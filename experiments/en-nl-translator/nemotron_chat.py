#!/usr/bin/env python3
"""Nemotron Chat v3 into Dutch (nemotron-chat-v3.md; Yeb 24 Nov ~14:50: first a spec and a small trial, the full run only
on his word): the chat split of nvidia/Nemotron-SFT-Instruction-Following-Chat-v3, lmarena seed prompts only, every turn
translated by G4-8-s140 (Gemma 4 31B-it with its LoRA) in pieces of at most 512 English tokens, once with the Dutch of
the conversation so far as context (ctx) and once without (noctx).

    python nemotron_chat.py sample --ranges 'DIR/part-*.jsonl' --tokenizer T --out DIR              (w0, CPU)
    python nemotron_chat.py translate --pieces F --cond ctx|noctx --cls c2k|n128|n512 --worker K --out gs://... \\
        --tokenizer T --lora DIR --work-dir DIR                                                       (a worker, TPU)
    python nemotron_chat.py collect --data DIR --gen DIR --out DIR                                    (w0, CPU)
    python nemotron_chat.py report --data DIR --collect DIR --qe 'GLOB' --out report.md               (w0, CPU)
    python nemotron_chat.py blind --data DIR --collect DIR --out DIR --key F                          (w0, CPU)

sample reads byte ranges of data/chat.jsonl at a pinned revision (only whole lines), keeps a conversation when its seed
prompt is lmarena's, its roles are user and assistant in turn after an empty system message, it is not a code task, it
is English, and its turns can be cut into pieces of at most PIECE_MAX tokens; it draws N_TRIAL conversations stratified
by assistant turns and length. translate keeps one engine for all waves: without context every piece of its class in one
wave; with context a wave takes the next piece of every conversation, whose prompt holds the Dutch already made of the
conversation so far (newest turns first, within CTX_TOKENS). A piece whose translation fails a hard check is translated
once more by sampling (RETRY_T); a piece that fails again breaks its conversation (with context its later pieces are
not translated): a conversation never goes on with a gap. collect assembles turns and conversations and counts the
checks; report and blind read no text themselves and print none: the texts stay in the bucket and on the VM."""

import argparse
import collections
import hashlib
import json
import random
import re
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
for _path in (HERE.parents[1] / "scripts", HERE.parent / "flan-nl", HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

PROTOCOL = "nemotron-chat-v3.md"
DATASET = "nvidia/Nemotron-SFT-Instruction-Following-Chat-v3"
REVISION = "be3b3e04ef605ac9d3f8f35b9d5a632f4a3a3402"
FILE_BYTES = 17_042_669_218  # data/chat.jsonl at REVISION (x-linked-size)
LMARENA = ("lmarena-ai/arena-human-preference-100k", "lmarena-ai/arena-human-preference-55k")
WILDCHAT = "allenai/WildChat-1M"
WILDCHAT_REVISION = "7d6490e462285cf85d91eabea0f9a954fbddcd1f"
WILDCHAT_FILES = 14
LMSYS = "lmsys/lmsys-chat-1m"
FULL_SEEDS = LMARENA + (WILDCHAT,)    # the full run (Yeb 24 Nov ~14:55: the whole chat split; lmsys never)
SHARD_BYTES = 128 * 2 ** 20           # a shard of the full run: the lines that start in 128 MiB of chat.jsonl
SHARDS = -(-FILE_BYTES // SHARD_BYTES)
FETCH_MARGIN = 8 * 2 ** 20
QE_MAX = 10.0                         # the full run's default: a conversation with a piece above it is left out
SEED = "nemotron-chat-v3/trial"
N_TRIAL = 200
WORKERS = 4
PIECE_MAX = 512        # English Gemma 4 tokens of a piece (the classes' max_source)
SHORT_MAX = 128        # a piece of at most this many tokens goes to n128 without context
TURN_MAX_TOKENS = 4096
CONV_MAX_TOKENS = 12288
CTX_TOKENS = 1408      # the context block: 2,176 (c2k's largest prefill) - 512 (a piece) - ~250 (instruction, template)
PROMPT_MAX = 2176
RETRY_T = 0.7
LABELS = {"user": "User", "assistant": "Assistant"}
DUTCH_LABELS = {"user": "Gebruiker", "assistant": "Assistent"}
CODE_SHARE_MAX = 0.10  # characters in fenced code blocks, of the conversation's
CODE_LINES_MAX = 2     # code-like lines outside fenced blocks, in the whole conversation
LANG_FIRST_LETTERS = 20
LANG_BLOCK_LETTERS = 100
CLASSES = {"c2k": ("ctx", 0, PIECE_MAX), "n128": ("noctx", 0, SHORT_MAX), "n512": ("noctx", SHORT_MAX, PIECE_MAX)}
# Marks of natural-v1-chat's context that a translation has no business to carry (unless its source has them)
MARKERS = ("<conversation_so_far", "</conversation_so_far", "[User]", "[Assistant", "(…)", "text to translate (")
CUT_MARK = "(…)"
CODE_LANGS = {
    "python", "py", "python3", "javascript", "js", "typescript", "ts", "java", "c", "cpp", "c++", "csharp", "cs", "c#",
    "go", "golang", "rust", "rs", "ruby", "rb", "php", "swift", "kotlin", "scala", "r", "matlab", "octave", "sql",
    "mysql", "postgresql", "sqlite", "bash", "sh", "shell", "zsh", "fish", "powershell", "ps1", "bat", "batch", "cmd",
    "html", "css", "scss", "sass", "less", "xml", "json", "jsonc", "yaml", "yml", "toml", "ini", "dockerfile", "makefile",
    "make", "cmake", "perl", "lua", "haskell", "hs", "julia", "dart", "vb", "vba", "vbnet", "assembly", "asm", "nasm",
    "jsx", "tsx", "vue", "svelte", "graphql", "solidity", "verilog", "vhdl", "systemverilog", "latex", "tex", "regex",
    "console", "terminal", "diff", "patch", "nginx", "apache", "groovy", "elixir", "erlang", "clojure", "fortran",
    "cobol", "pascal", "delphi", "objective-c", "objc", "sas", "stata", "jinja", "jinja2", "django", "twig",
    "handlebars", "glsl", "hlsl", "cuda", "arduino", "gdscript", "lisp", "elisp", "scheme", "racket", "prolog", "ocaml",
    "fsharp", "f#", "nim", "zig", "crystal", "awk", "sed", "jq", "proto", "protobuf", "terraform", "hcl", "gradle",
    "maven", "kusto", "kql", "dax", "m", "abap", "apex", "ahk", "autohotkey", "vim", "mermaid", "plantuml", "csv",
}
FENCE_OPEN = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")
SEP = re.compile(r"\n(?:[ \t]*\n)+")
SENTENCE = re.compile(r"(?<=[.!?:;])[ \t]+|\n")
LETTER = re.compile(r"[^\W\d_]")
INLINE_CODE = re.compile(r"(?<!`)`(?!`)[^`\n]+`(?!`)")
CODE_LINE = re.compile(
    r"^\s*(?:def \w+\(|class \w+\s*[:({]|import [\w.]+\s*;?\s*$|from [\w.]+ import |#include\s*[<\"]|"
    r"public (?:static |class |void |int |String )|private (?:static |void |int )|function \w*\s*\(|"
    r"(?:const|let|var) \w+\s*=|SELECT\s.+\sFROM\s|INSERT INTO |CREATE TABLE |<\?php|console\.log\(|System\.out\.|"
    r"printf\(|print\(.*\)\s*$|return [^.]*;\s*$|\}\s*$|for\s*\(.*;.*;.*\)|if\s*\(.+\)\s*\{|"
    r"<(?:div|html|script|body|span|head)\b)")


class Unsplittable(ValueError):
    """A turn with a block that does not fit a piece: a fenced code block or a sentence longer than PIECE_MAX."""


# ---------------------------------------------------------------------------------------------------- small helpers

def stable(key):
    return hashlib.sha1(key.encode()).hexdigest()


def seed_int(key):
    return int(stable(key)[:8], 16)


# Files are local paths or gs:// URLs: the full run reads and writes the bucket from Python, so that no copy is left on
# a worker's /dev/shm (w0 has ~10 GB free) and nothing has to be removed there.
_CLIENT = []


def gcs():
    if not _CLIENT:
        from google.cloud import storage

        _CLIENT.append(storage.Client())
    return _CLIENT[0]


def blob(url):
    bucket, _, name = url[len("gs://"):].partition("/")
    if not bucket or not name:
        raise ValueError(f"not a gs://bucket/object URL: {url}")
    return gcs().bucket(bucket).blob(name)


def join(base, *parts):
    return "/".join([str(base).rstrip("/")] + [str(p).strip("/") for p in parts])


def read_bytes(path):
    path = str(path)
    return blob(path).download_as_bytes() if path.startswith("gs://") else Path(path).read_bytes()


def write_bytes(path, data):
    path = str(path)
    if path.startswith("gs://"):
        from google.cloud.storage.retry import DEFAULT_RETRY

        blob(path).upload_from_string(data, retry=DEFAULT_RETRY)
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(data)


def exists(path):
    path = str(path)
    return blob(path).exists() if path.startswith("gs://") else Path(path).exists()


def list_files(root, pattern):
    """The files under root (recursively) whose path relative to root matches the regex pattern, sorted"""
    root = str(root).rstrip("/")
    regex = re.compile(pattern)
    if root.startswith("gs://"):
        bucket, _, prefix = root[len("gs://"):].partition("/")
        names = [b.name for b in gcs().list_blobs(bucket, prefix=prefix + "/")]
        rel = [n[len(prefix) + 1:] for n in names]
    else:
        rel = [str(q.relative_to(root)) for q in Path(root).rglob("*") if q.is_file()]
    return [join(root, r) for r in sorted(rel) if regex.fullmatch(r)]


def read_json(path):
    return json.loads(read_bytes(path))


def write_json(obj, path):
    write_bytes(path, (json.dumps(obj, indent=1) + "\n").encode())


def read_rows(path, columns=None):
    import io

    import pyarrow.parquet as pq

    return pq.read_table(io.BytesIO(read_bytes(path)), columns=columns).to_pylist()


def write_rows(rows, path):
    import io

    import pyarrow as pa
    import pyarrow.parquet as pq

    buf = io.BytesIO()
    pq.write_table(pa.Table.from_pylist(rows), buf, compression="zstd")
    write_bytes(path, buf.getvalue())


def quantiles(xs, qs=(0.05, 0.5, 0.9, 0.95, 0.99)):
    xs = sorted(xs)
    if not xs:
        return {}
    return {f"p{int(q * 100)}": xs[min(len(xs) - 1, int(q * len(xs)))] for q in qs} | {"max": xs[-1], "mean": round(
        statistics.mean(xs), 2), "n": len(xs)}


def langid_identifier(languages=None):
    """identify(text) -> language code: langid 1.1.6 over all its languages, or over `languages` (FLAN-nl's set)"""
    from langid.langid import LanguageIdentifier, model

    ident = LanguageIdentifier.from_modelstring(model, norm_probs=False)
    if languages:
        ident.set_languages(languages)
    return lambda text: ident.classify(text)[0]


# ---------------------------------------------------------------------------------------------------- the source

def range_rows(path, offset):
    """The whole JSON lines of a byte range of chat.jsonl that starts at offset: the first line is cut unless the range
    starts the file, and the last element after the final newline is cut (or empty)."""
    lines = Path(path).read_bytes().split(b"\n")
    if offset > 0:
        lines = lines[1:]
    return [json.loads(line.decode("utf-8")) for line in lines[:-1] if line.strip()]


def offset_of(path):
    m = re.search(r"part-(\d+)\.jsonl$", str(path))
    if not m:
        raise ValueError(f"{path}: not part-<offset>.jsonl")
    return int(m.group(1))


def conversation(row, seeds=LMARENA):
    """(conversation, None) or (None, reason): a conversation of a seed in seeds (the trial: lmarena) of user and
    assistant turns in turn, starting with the user and ending with the assistant, after an empty system message
    (lmarena's rows have one; WildChat's is null, also after restore_row). lmsys-chat-1m never (its licence)."""
    meta = row.get("metadata") or {}
    seed = meta.get("seed_dataset")
    if seed == LMSYS:
        return None, "seed_lmsys"
    if seed not in seeds:
        return None, "seed_not_lmarena" if seeds == LMARENA else "seed_other"
    turns = []
    for k, m in enumerate(row.get("messages") or []):
        role, content = m.get("role"), m.get("content")
        if role == "system" and k == 0:
            if content:
                return None, "system_prompt"
            continue
        if role not in LABELS or not isinstance(content, str) or not content.strip():
            return None, "roles"
        turns.append({"role": role, "text": content, "message": k})
    roles = [t["role"] for t in turns]
    if not turns or roles != [("user", "assistant")[i % 2] for i in range(len(turns))] or roles[-1] != "assistant":
        return None, "roles"
    reasoning = sum(len(m.get("reasoning_content") or "") for m in row.get("messages") or [])
    return {"uuid": str(row["uuid"]), "seed": seed, "turns": turns, "reasoning_chars": reasoning,
            "train_turns": json.dumps(meta.get("train_turns"))}, None


# ---------------------------------------------------------------------------------------------------- markdown

def fence_spans(text):
    """[(start, end, info)] of the fenced code blocks of text (``` or ~~~, at most 3 spaces in), the fences included;
    a block that is not closed runs to the end of the text."""
    spans, pos, open_ = [], 0, None
    for line in text.split("\n"):
        end = pos + len(line)
        if open_ is None:
            m = FENCE_OPEN.match(line)
            if m:
                open_ = (pos, m.group(1)[0], len(m.group(1)), m.group(2).strip())
        else:
            stripped = line.strip()
            if stripped and set(stripped) == {open_[1]} and len(stripped) >= open_[2]:
                spans.append((open_[0], end, open_[3]))
                open_ = None
        pos = end + 1
    if open_ is not None:
        spans.append((open_[0], len(text), open_[3]))
    return spans


def fences(text):
    return [text[s:e] for s, e, _ in fence_spans(text)]


def without_fences(text):
    out, last = [], 0
    for s, e, _ in fence_spans(text):
        out.append(text[last:s])
        out.append("\n")
        last = e
    out.append(text[last:])
    return "".join(out)


def prose(text):
    """The text without its fenced code blocks and inline code: what the language checks look at"""
    return INLINE_CODE.sub(" ", without_fences(text))


def letters(text):
    return len(LETTER.findall(text))


def blocks(core):
    """[(block, separator after it)] of a stripped text: cut at blank lines outside fenced code blocks (a fenced block
    with blank lines in it stays one block); the last separator is ''. ''.join(b + s) is the text."""
    spans = fence_spans(core)
    out, start = [], 0
    for m in SEP.finditer(core):
        if any(s < m.start() < e for s, e, _ in spans):
            continue
        out.append((core[start:m.start()], m.group(0)))
        start = m.end()
    out.append((core[start:], ""))
    return out


def sentences(block, sep_after):
    """[(part, separator)] of a block cut after sentence ends and at line breaks; the last part takes sep_after. An
    empty part (a line break right after the spaces behind a sentence end) gives its separator to the part before it,
    so that ''.join(p + s) is the block + sep_after, character for character."""
    out, start, pending = [], 0, ""
    for m in SENTENCE.finditer(block):
        part = block[start:m.start()]
        if part:
            out.append([pending + part, m.group(0)])
            pending = ""
        elif out:
            out[-1][1] += m.group(0)
        else:
            pending += m.group(0)
        start = m.end()
    last = pending + block[start:]
    if last:
        out.append([last, sep_after])
    elif out:
        out[-1][1] += sep_after
    else:
        return [(block, sep_after)]
    return [tuple(x) for x in out]


def pack(units, count, limit):
    """[[piece, separator after it]]: consecutive units joined with their separators while a piece has at most limit
    tokens (a unit longer than limit stays alone; split_turn refuses those first)"""
    pieces = []
    for text, sep in units:
        if pieces and count(pieces[-1][0] + pieces[-1][1] + text) <= limit:
            pieces[-1] = [pieces[-1][0] + pieces[-1][1] + text, sep]
        else:
            pieces.append([text, sep])
    return pieces


def split_turn(text, count, limit=PIECE_MAX):
    """(lead, [[piece, separator after it]], trail): a turn in pieces of at most limit tokens, cut between blocks, a
    block longer than limit between sentences or lines; Unsplittable for a fenced block or a sentence longer than
    limit. lead + ''.join(p + s) + trail is the turn, character for character."""
    core = text.strip()
    lead, trail = text[:len(text) - len(text.lstrip())], text[len(text.rstrip()):]
    units = []
    for block, sep in blocks(core):
        if count(block) <= limit:
            units.append((block, sep))
            continue
        if fence_spans(block):
            raise Unsplittable("fence")
        parts = sentences(block, sep)
        if any(count(p) > limit for p, _ in parts):
            raise Unsplittable("sentence")
        units.extend(parts)
    pieces = pack(units, count, limit)
    if lead + "".join(p + s for p, s in pieces) + trail != text:
        raise AssertionError("split_turn does not give the turn back")
    return lead, pieces, trail


def assemble(lead, parts, seps, trail):
    """A turn from its pieces' texts and the source's separators between them"""
    return lead + "".join(p + s for p, s in zip(parts, seps)) + trail


# ---------------------------------------------------------------------------------------------------- the filters

def code_reason(conv):
    """Why a conversation counts as a code task, or None: a fenced block with a programming language's tag, more than
    CODE_SHARE_MAX of its characters in fenced blocks, or more than CODE_LINES_MAX code-like lines outside them."""
    total = code = lines = 0
    for t in conv["turns"]:
        text = t["text"]
        total += len(text)
        for s, e, info in fence_spans(text):
            code += e - s
            tag = re.split(r"[\s{,]", info.lower().lstrip("{."), maxsplit=1)[0] if info else ""
            if tag in CODE_LANGS:
                return "code_fence_lang"
        lines += sum(bool(CODE_LINE.match(line)) for line in without_fences(text).split("\n"))
    if total and code / total > CODE_SHARE_MAX:
        return "code_share"
    if lines > CODE_LINES_MAX:
        return "code_lines"
    return None


def language_reason(conv, identify):
    """Why a conversation is not (all) English, or None: the first user turn (with at least LANG_FIRST_LETTERS letters
    of prose) or any paragraph of LANG_BLOCK_LETTERS letters or more that langid does not take for English."""
    first = prose(conv["turns"][0]["text"])
    if letters(first) >= LANG_FIRST_LETTERS and identify(first) != "en":
        return "lang_first"
    for t in conv["turns"]:
        for block, _ in blocks(prose(t["text"]).strip()):
            if letters(block) >= LANG_BLOCK_LETTERS and identify(block) != "en":
                return "lang_paragraph"
    return None


def stratum(n_assistant, tokens):
    """a1-a4 (assistant turns, 4 or more as a4) x s/m/l (at most 1,024 / 3,072 English tokens, or more)"""
    return f"a{min(n_assistant, 4)}{'s' if tokens <= 1024 else 'm' if tokens <= 3072 else 'l'}"


def prepare(conv, count, identify):
    """The conversation with tokens, pieces and stratum, or the reason it is left out"""
    why = code_reason(conv) or language_reason(conv, identify)
    if why:
        return None, why
    for t in conv["turns"]:
        t["tokens"] = count(t["text"])
    tokens = [t["tokens"] for t in conv["turns"]]
    if max(tokens) > TURN_MAX_TOKENS:
        return None, "turn_long"
    if sum(tokens) > CONV_MAX_TOKENS:
        return None, "conv_long"
    for t in conv["turns"]:
        try:
            t["lead"], pieces, t["trail"] = split_turn(t["text"], count)
        except Unsplittable as error:
            return None, f"unsplittable_{error}"
        t["pieces"] = [{"text": p, "sep": s, "tokens": count(p)} for p, s in pieces]
    conv["tokens"] = sum(tokens)
    conv["n_assistant"] = sum(t["role"] == "assistant" for t in conv["turns"])
    conv["stratum"] = stratum(conv["n_assistant"], conv["tokens"])
    return conv, None


def pick(pool, n, seed=SEED):
    """n conversations, as evenly over the strata as they allow (a stratum with fewer gives all it has, the rest goes to
    the others), within a stratum in the order of sha1(seed/uuid)"""
    by = collections.defaultdict(list)
    for c in pool:
        by[c["stratum"]].append(c)
    for s in by:
        by[s].sort(key=lambda c: stable(f"{seed}/{c['uuid']}"))
    quota = {s: 0 for s in by}
    remaining = n
    while remaining > 0:
        open_ = sorted(s for s in by if quota[s] < len(by[s]))
        if not open_:
            break
        share = max(1, remaining // len(open_))
        for s in open_:
            take = min(share, len(by[s]) - quota[s], remaining)
            quota[s] += take
            remaining -= take
            if not remaining:
                break
    return [c for s in sorted(by) for c in by[s][:quota[s]]]


def assign_workers(convs, workers=WORKERS):
    """{uuid: worker}: longest conversations first, each to the worker with the fewest tokens so far (ties: lowest)"""
    load = [0] * workers
    out = {}
    for c in sorted(convs, key=lambda c: (-c["tokens"], stable(c["uuid"]))):
        k = min(range(workers), key=lambda w: (load[w], w))
        out[c["uuid"]] = k
        load[k] += c["tokens"]
    return out


def pool_meta_row(c):
    pieces = [p for t in c["turns"] for p in t["pieces"]]
    return {"uuid": c["uuid"], "seed": c["seed"], "stratum": c["stratum"], "turns": len(c["turns"]),
            "assistant": c["n_assistant"], "tokens": c["tokens"],
            "tokens_user": sum(t["tokens"] for t in c["turns"] if t["role"] == "user"),
            "tokens_assistant": sum(t["tokens"] for t in c["turns"] if t["role"] == "assistant"),
            "chars": sum(len(t["text"]) for t in c["turns"]), "pieces": len(pieces),
            "pieces_short": sum(p["tokens"] <= SHORT_MAX for p in pieces),
            "tokens_short": sum(p["tokens"] for p in pieces if p["tokens"] <= SHORT_MAX),
            "tokens_long": sum(p["tokens"] for p in pieces if p["tokens"] > SHORT_MAX),
            "fences": sum(len(fence_spans(t["text"])) for t in c["turns"]), "reasoning_chars": c["reasoning_chars"]}


def sample(args):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    def count(text):
        return len(tok(text, add_special_tokens=False)["input_ids"])

    identify = langid_identifier()
    paths = sorted(Path(args.ranges_dir).glob("part-*.jsonl"), key=offset_of)
    if not paths:
        raise SystemExit(f"no part-*.jsonl in {args.ranges_dir}")
    rows, seen, dup = [], set(), 0
    for path in paths:
        for row in range_rows(path, offset_of(path)):
            if row["uuid"] in seen:
                dup += 1
                continue
            seen.add(row["uuid"])
            rows.append(row)
    seeds = collections.Counter((r.get("metadata") or {}).get("seed_dataset") for r in rows)
    reasons, pool = collections.Counter(), []
    for row in rows:
        conv, why = conversation(row)
        if conv is not None:
            conv, why = prepare(conv, count, identify)
        if why:
            reasons[why] += 1
            continue
        pool.append(conv)
    chosen = pick(pool, args.n)
    workers = assign_workers(chosen)
    out = Path(args.out)
    conversations, turns, pieces = [], [], []
    for c in chosen:
        conversations.append({"conv": c["uuid"], "seed": c["seed"], "stratum": c["stratum"], "turns": len(c["turns"]),
                              "assistant": c["n_assistant"], "tokens": c["tokens"], "worker": workers[c["uuid"]],
                              "train_turns": c["train_turns"], "reasoning_chars": c["reasoning_chars"]})
        for i, t in enumerate(c["turns"]):
            turns.append({"conv": c["uuid"], "turn": i, "role": t["role"], "message": t["message"], "text": t["text"],
                          "lead": t["lead"], "trail": t["trail"], "pieces": len(t["pieces"]), "tokens": t["tokens"],
                          "worker": workers[c["uuid"]]})
            for j, p in enumerate(t["pieces"]):
                pieces.append({"pid": f"{c['uuid']}|t{i}|p{j}", "conv": c["uuid"], "turn": i, "piece": j,
                               "n_pieces": len(t["pieces"]), "role": t["role"], "text": p["text"], "sep_after": p["sep"],
                               "tokens": p["tokens"], "worker": workers[c["uuid"]], "stratum": c["stratum"]})
    write_rows(conversations, out / "conversations.parquet")
    write_rows(turns, out / "turns.parquet")
    write_rows(pieces, out / "pieces.parquet")
    write_rows([{k: v for k, v in p.items() if k not in ("text", "sep_after")} for p in pieces], out / "pieces-meta.parquet")
    write_rows([pool_meta_row(c) for c in pool], out / "pool-meta.parquet")
    per_worker = collections.Counter()
    for c in chosen:
        per_worker[f"w{workers[c['uuid']]}"] += c["tokens"]
    info = {"protocol": PROTOCOL, "dataset": DATASET, "revision": REVISION, "ranges": len(paths),
            "range_bytes": sum(p.stat().st_size for p in paths), "file_bytes": FILE_BYTES, "rows": len(rows),
            "duplicates": dup, "seeds": dict(seeds), "left_out": dict(reasons), "pool": len(pool),
            "pool_strata": dict(collections.Counter(c["stratum"] for c in pool)),
            "chosen": len(chosen), "chosen_strata": dict(collections.Counter(c["stratum"] for c in chosen)),
            "chosen_seeds": dict(collections.Counter(c["seed"] for c in chosen)),
            "turns": len(turns), "pieces": len(pieces), "pieces_short": sum(p["tokens"] <= SHORT_MAX for p in pieces),
            "tokens": sum(c["tokens"] for c in chosen), "worker_tokens": dict(per_worker),
            "pool_tokens": quantiles([c["tokens"] for c in pool]),
            "pool_turn_tokens": {r: quantiles([t["tokens"] for c in pool for t in c["turns"] if t["role"] == r])
                                 for r in LABELS},
            "pool_assistant_turns": dict(collections.Counter(c["n_assistant"] for c in pool))}
    (out / "sample.json").write_text(json.dumps(info, indent=1) + "\n")
    print(f"NEMOSAMPLE {json.dumps({k: info[k] for k in ('rows', 'pool', 'chosen', 'turns', 'pieces', 'tokens', 'left_out')})}",
          flush=True)
    return info


# ---------------------------------------------------------------------------------------------------- prompts

def history(turn_rows, pieces_of, results, turn, piece):
    """The context of piece `piece` of turn `turn`, oldest first: [{label, translation}] of every earlier turn whole
    (its pieces' Dutch with the source's separators), then the Dutch of this turn's earlier pieces"""
    items = []
    for t in range(turn):
        ps = pieces_of[t]
        dutch = [results[p["pid"]]["text_nl"] for p in ps]
        items.append({"label": LABELS[turn_rows[t]["role"]],
                      "translation": assemble("", dutch, [p["sep_after"] for p in ps[:-1]] + [""], "")})
    if piece:
        ps = pieces_of[turn][:piece]
        dutch = [results[p["pid"]]["text_nl"] for p in ps]
        items.append({"label": f"{LABELS[turn_rows[turn]['role']]}, this turn so far",
                      "translation": assemble("", dutch, [p["sep_after"] for p in ps[:-1]] + [""], "")})
    return items


def item_cost(item, count):
    return count(f"[{item['label']}]\n{item['translation'].strip()}\n") + 1


def fit_context(items, count, budget=CTX_TOKENS):
    """(items, cut): the newest items whose rendering takes at most budget tokens, oldest first; when the newest alone
    is too long, its last blocks (cut at blank lines) after CUT_MARK, or nothing when its last block alone is too long"""
    chosen, used = [], 0
    for item in reversed(items):
        cost = item_cost(item, count)
        if used + cost <= budget:
            chosen.append(item)
            used += cost
            continue
        if not chosen:
            parts = blocks(item["translation"].strip())
            for k in range(1, len(parts)):
                tail = dict(item, translation=CUT_MARK + "\n\n" + "".join(b + s for b, s in parts[k:]))
                if item_cost(tail, count) <= budget:
                    return [tail], True
            return [], True
        break
    return list(reversed(chosen)), False


def current_label(role, piece):
    return LABELS[role] + (", continued" if piece else "")


# ---------------------------------------------------------------------------------------------------- checks

def leak(nl, en):
    return [m for m in MARKERS if m in nl and m not in en]


def hard_flags(en, nl, ended, identify_nl):
    """The hard checks of a piece's translation (a failed one is retried, then breaks its conversation): FLAN-nl's
    segment checks (flan_nl.check_translation: truncated, empty, length ratio, commentary, alternatives, foreign script,
    not Dutch) on the prose (without code), its detectors decode_loop and english_left, and a mark of the prompt"""
    import flan_nl
    import publish

    flags = []
    if not ended:
        flags.append("truncated")
    if not nl.strip():
        return flags + ["empty"]
    p_en, p_nl = prose(en), prose(nl)
    if p_en.strip():
        why = flan_nl.check_translation(p_en, p_nl, True, identify_nl)
        if why:
            flags.append(why)
        if p_nl.strip() and publish.decode_loop(p_nl, p_en):
            flags.append("decode_loop")
        if p_nl.strip() and publish.english_left(p_nl, p_en):
            flags.append("english_left")
    if leak(nl, en):
        flags.append("tag_leak")
    return flags


def structure_of(en, nl):
    """The soft checks (counted, not retried): structure.differences (markdown categories), whether the fenced code
    blocks are byte for byte the same (None without any) and the inline code spans the same"""
    import structure

    f_en, f_nl = fences(en), fences(nl)
    i_en = INLINE_CODE.findall(without_fences(en))
    i_nl = INLINE_CODE.findall(without_fences(nl))
    a, b = structure.features(en), structure.features(nl)
    return {"diffs": structure.differences(en, nl), "fences_en": len(f_en), "fences_nl": len(f_nl),
            "fences_same": (collections.Counter(f_en) == collections.Counter(f_nl)) if (f_en or f_nl) else None,
            "inline_en": len(i_en), "inline_same": (collections.Counter(i_en) == collections.Counter(i_nl))
            if (i_en or i_nl) else None,
            "headings_en": a["heading"], "headings_nl": b["heading"], "items_en": a["bullet"] + a["number"],
            "items_nl": b["bullet"] + b["number"], "rows_en": a["table_row"], "rows_nl": b["table_row"]}


# ---------------------------------------------------------------------------------------------------- the waves

def next_wave(units_by_conv, results, broken, cond):
    """The units of the next wave: without context every unit not done (one wave); with context the first unit not
    done of every conversation that is not broken"""
    wave = []
    for conv in sorted(units_by_conv):
        if conv in broken:
            continue
        pending = [u for u in units_by_conv[conv] if u["pid"] not in results]
        if cond == "noctx":
            wave.extend(pending)
        elif pending:
            wave.append(pending[0])
    return wave


def run_waves(units_by_conv, cond, translate_wave, results=None, broken=None, first_wave=0):
    """translate_wave(wave, units, results) -> {pid: row} for every unit (row["failed"] after the retry) until no
    unit is left; a failed unit breaks its conversation (with context its later units are never translated).
    Returns (results, broken, waves run)."""
    results = dict(results or {})
    broken = set(broken or ())
    w = first_wave
    while True:
        wave = next_wave(units_by_conv, results, broken, cond)
        if not wave:
            return results, broken, w - first_wave
        rows = translate_wave(w, wave, results)
        if set(rows) != {u["pid"] for u in wave}:
            raise RuntimeError(f"wave {w}: rows for {len(rows)} units, not the wave's {len(wave)}")
        for u in wave:
            results[u["pid"]] = rows[u["pid"]]
            if rows[u["pid"]]["failed"]:
                broken.add(u["conv"])
        w += 1


def translate_jobs(args):
    """[(pieces, out)]: --pieces and --out, or the lines "<pieces> <out>" of --jobs (the full run: one per shard)"""
    if bool(args.jobs) == bool(args.pieces or args.out):
        raise SystemExit("give --pieces and --out, or --jobs")
    if args.jobs:
        jobs = [tuple(line.split()) for line in Path(args.jobs).read_text().splitlines() if line.strip()]
        if any(len(j) != 2 for j in jobs):
            raise SystemExit(f"{args.jobs}: a line is not '<pieces> <out>'")
        return jobs
    if not (args.pieces and args.out):
        raise SystemExit("--pieces and --out go together")
    return [(args.pieces, args.out)]


def translate_run(args):
    """Every job of translate_jobs with one engine, made at the first job with work: a job with DONE.json in its out
    is skipped, one with waves there goes on after them; DONE.json after its last wave."""
    import io

    import jax
    import pyarrow.parquet as pq
    import translate as tr
    from transformers import AutoTokenizer

    cond, lo, hi = CLASSES[args.cls]
    if cond != args.cond:
        raise SystemExit(f"--cls {args.cls} is a class of {cond}, not {args.cond}")
    model = tr.MODELS[args.model]
    shapes = dict(model["length_classes"][args.cls], slots=model["slots"][args.cls])
    if (jax.device_count(), jax.process_count()) != (model["chips"], 1):
        raise SystemExit(f"needs one process with {model['chips']} chips, has {jax.process_count()} process(es) with "
                         f"{jax.device_count()} devices")
    jobs = translate_jobs(args)
    client = tr.gcs_client()
    st = {}

    def engine():
        """tokenizer, codec, identify_nl and the device, made once"""
        if st:
            return st
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        codec = tr.ModelPrompts(tokenizer, model)
        codec.check(["Hello.", "How are you?"], [None, [{"label": "User", "translation": "Hallo."}]], [None, "Assistant"])
        print(f"MAXTEXT {json.dumps({'model': args.model, 'prompt': model['prompt'], 'context_prompt': model.get('context_prompt'), 'lora': args.lora, 'ckpt': args.ckpt, **tr.maxtext_source(), 'patches': tr.patch_maxtext()})}", flush=True)
        started = time.time()
        engine_args = argparse.Namespace(model=args.model, ckpt=args.ckpt, work_dir=args.work_dir,
                                         tokenizer=args.tokenizer, copies=None)
        eng, params = tr.make_engine(engine_args, shapes, model["chips"])
        if args.lora:
            params = tr.load_lora(params, args.lora)
        st.update(tokenizer=tokenizer, codec=codec, device=tr.EngineSlots(eng, params, shapes["prefill_lengths"]),
                  identify_nl=langid_identifier(["nl", "en", "de", "fr", "es", "it", "pt", "af"]),
                  engine_seconds=time.time() - started)
        print(f"engine ready in {st['engine_seconds']:.0f} s", flush=True)
        return st

    def read_pieces(src):
        if src.startswith("gs://"):
            return pq.read_table(io.BytesIO(tr.blob_of(src, client).download_as_bytes())).to_pylist()
        return read_rows(src)

    def upload_json(stats, out, name):
        local = Path(args.work_dir) / "nemotron"
        local.mkdir(parents=True, exist_ok=True)
        marker = local / f"{name}-{stable(out)[:12]}.json"
        marker.write_text(json.dumps(stats) + "\n")
        tr.upload(marker, f"{out}/{name}.json", client)
        marker.unlink()

    all_stats = []
    for src, out in jobs:
        out = out.rstrip("/")
        if tr.blob_of(f"{out}/DONE.json", client).exists():
            print(f"NEMODONE {json.dumps({'out': out, 'note': 'DONE.json there before'})}", flush=True)
            continue
        pieces = [p for p in read_pieces(src) if p["worker"] == args.worker]
        turn_roles = {(p["conv"], p["turn"]): {"role": p["role"]} for p in pieces}
        units = [p for p in pieces if cond == "ctx" or lo < p["tokens"] <= hi]
        by_conv = collections.defaultdict(list)
        for u in sorted(units, key=lambda u: (u["conv"], u["turn"], u["piece"])):
            by_conv[u["conv"]].append(u)
        pieces_of = collections.defaultdict(lambda: collections.defaultdict(list))
        for p in sorted(pieces, key=lambda p: (p["conv"], p["turn"], p["piece"])):
            pieces_of[p["conv"]][p["turn"]].append(p)
        done = sorted(tr.done_shards(out, client))
        results, broken = {}, set()
        for name in done:
            for r in pq.read_table(io.BytesIO(tr.blob_of(f"{out}/{name}", client).download_as_bytes())).to_pylist():
                results[r["pid"]] = r
                if r["failed"]:
                    broken.add(r["conv"])
        first_wave = len(done)
        print(f"{cond}/{args.cls} w{args.worker} {out}: {len(units)} units of {len(by_conv)} conversations, "
              f"{len(results)} done in {first_wave} waves before; shapes {json.dumps(shapes)}", flush=True)
        base = {"cond": cond, "cls": args.cls, "worker": args.worker, "out": out, "units": len(units),
                "convs": len(by_conv)}
        if not next_wave(by_conv, results, broken, cond):
            upload_json(base | {"note": "nothing left to translate at this start", "broken": len(broken)}, out, "DONE")
            print(f"NEMODONE {json.dumps(base | {'note': 'nothing left'})}", flush=True)
            continue
        e = engine()
        tokenizer, codec, device, identify_nl = e["tokenizer"], e["codec"], e["device"], e["identify_nl"]

        def count(text):
            return len(tokenizer(text, add_special_tokens=False)["input_ids"])

        local = Path(args.work_dir) / "nemotron" / stable(out)[:12]
        local.mkdir(parents=True, exist_ok=True)
        totals = collections.Counter()

        def prompt_of(u, done_results):
            if cond == "noctx":
                return codec.encode(u["text"]), {"context_items": 0, "context_tokens": 0, "context_cut": False}
            items = history([turn_roles[(u["conv"], t)] for t in range(u["turn"] + 1)], pieces_of[u["conv"]],
                            done_results, u["turn"], u["piece"])
            chosen, cut = fit_context(items, count)
            while True:
                ids = codec.encode(u["text"], chosen or None, current_label(u["role"], u["piece"]) if chosen else None)
                if len(ids) <= PROMPT_MAX or not chosen:
                    break
                chosen, cut = chosen[1:], True
            if len(ids) > shapes["prefill_lengths"][-1]:
                raise RuntimeError(f"{u['pid']}: a prompt of {len(ids)} tokens, more than {shapes['prefill_lengths'][-1]}")
            return ids, {"context_items": len(chosen), "context_tokens": sum(item_cost(i, count) for i in chosen),
                         "context_cut": cut}

        def decode(prompts):
            t0 = time.time()
            outputs = tr.decode_all(prompts, shapes["slots"], device, shapes["max_decode"], args.sync_every,
                                    model["eos_ids"])
            return outputs, time.time() - t0

        def translate_wave(w, wave, done_results, out=out, local=local, totals=totals):
            prompts, info = zip(*(prompt_of(u, done_results) for u in wave))
            outputs, seconds = decode(list(prompts))
            rows = {}
            for u, ids, meta, (tokens, ended) in zip(wave, prompts, info, outputs):
                nl = tokenizer.decode(tokens, skip_special_tokens=True).strip()
                flags = hard_flags(u["text"], nl, ended, identify_nl)
                rows[u["pid"]] = {"pid": u["pid"], "conv": u["conv"], "turn": u["turn"], "piece": u["piece"],
                                  "role": u["role"], "cond": cond, "cls": args.cls, "worker": args.worker, "wave": w,
                                  "attempt": 0, "text_nl": nl, "ended": bool(ended), "tokens_in": len(ids),
                                  "tokens_out": len(tokens) + int(bool(ended)), "flags0": json.dumps(flags),
                                  "flags": json.dumps(flags), "failed": bool(flags), **meta}
            retry = [k for k, u in enumerate(wave) if rows[u["pid"]]["failed"]]
            retry_seconds = 0.0
            if retry:
                device.sample(RETRY_T, seed_int(f"{SEED}/retry/{out}/wave{w}"))
                outputs2, retry_seconds = decode([prompts[k] for k in retry])
                device.sampling = {}
                totals["retry_tokens_out"] += sum(len(t) + int(bool(x)) for t, x in outputs2)
                for k, (tokens, ended) in zip(retry, outputs2):
                    u = wave[k]
                    nl = tokenizer.decode(tokens, skip_special_tokens=True).strip()
                    flags = hard_flags(u["text"], nl, ended, identify_nl)
                    rows[u["pid"]].update(attempt=1, text_nl=nl, ended=bool(ended),
                                          tokens_out=len(tokens) + int(bool(ended)), flags=json.dumps(flags),
                                          failed=bool(flags))
            n_in = sum(r["tokens_in"] for r in rows.values())
            n_out0 = sum(len(o[0]) + int(bool(o[1])) for o in outputs)
            stat = {"wave": w, "units": len(wave), "tokens_in": n_in, "tokens_out": n_out0,
                    "seconds": round(seconds, 2), "retried": len(retry), "retry_seconds": round(retry_seconds, 2),
                    "failed": sum(r["failed"] for r in rows.values()), "full": len(wave) >= shapes["slots"]}
            for key in ("units", "tokens_in", "tokens_out", "seconds", "retried", "retry_seconds", "failed"):
                totals[key] += stat[key]
            if stat["full"]:
                totals["full_tokens_out"] += n_out0
                totals["full_seconds"] += seconds
            totals["waves"] += 1
            path = local / f"wave-{w:03d}.parquet"
            write_rows([rows[u["pid"]] for u in wave], path)
            tr.upload(path, f"{out}/wave-{w:03d}.parquet", client)
            path.unlink()
            print(f"WAVE {json.dumps(stat | {'out_per_s': round(n_out0 / max(seconds, 1e-9))})}", flush=True)
            return rows

        results, broken, waves = run_waves(by_conv, cond, translate_wave, results, broken, first_wave)
        stats = base | {"engine_seconds": round(e["engine_seconds"], 1) if len(all_stats) == 0 else 0.0,
                        "waves_run": waves, "waves_before": first_wave, "shapes": shapes,
                        **{k: (round(v, 2) if isinstance(v, float) else v) for k, v in totals.items()},
                        "broken": len(broken), "skipped": sum(1 for u in units if u["pid"] not in results),
                        "lora": args.lora, "model": args.model}
        upload_json(stats, out, "DONE")
        all_stats.append(stats)
        print(f"NEMODONE {json.dumps(stats)}", flush=True)
    print(f"NEMOJOBS {json.dumps({'jobs': len(jobs), 'translated': len(all_stats)})}", flush=True)
    return all_stats


# ---------------------------------------------------------------------------------------------------- collect

def read_gen(gen_dir, leaf=r"w\d+"):
    """{cond: {pid: row}} and the DONE stats [{...}] of gen/<cond>/<cls>/<leaf>/ (the trial: w<k>; the full run:
    shard-<r>), a local directory or a gs:// prefix"""
    results = collections.defaultdict(dict)
    stats = []
    for path in list_files(gen_dir, rf"(ctx|noctx)/(c2k|n128|n512)/{leaf}/wave-\d+\.parquet"):
        for r in read_rows(path):
            if r["pid"] in results[r["cond"]]:
                raise SystemExit(f"{path}: {r['pid']} translated twice in {r['cond']}")
            results[r["cond"]][r["pid"]] = r
    for path in list_files(gen_dir, rf"(ctx|noctx)/(c2k|n128|n512)/{leaf}/DONE\.json"):
        stats.append(read_json(path))
    return results, stats


def collect(args):
    data = args.data
    pieces = read_rows(join(data, "pieces.parquet"))
    turns = read_rows(join(data, "turns.parquet"))
    convs = read_rows(join(data, "conversations.parquet"))
    conds = tuple(args.conds.split(","))
    results, stats = read_gen(args.gen, args.leaf)
    workers = sorted({p["worker"] for p in pieces})
    expected = {(s["cond"], s["cls"], s["worker"]) for s in stats}
    want = {(c, k, w) for k, (c, _, _) in CLASSES.items() if c in conds for w in workers}
    if expected != want:
        raise SystemExit(f"{args.gen}: DONE of {sorted(expected)}, not of every class of {conds} and worker {workers}")
    for cond in conds:
        results.setdefault(cond, {})
    pieces_of = collections.defaultdict(list)
    for p in sorted(pieces, key=lambda p: (p["conv"], p["turn"], p["piece"])):
        pieces_of[(p["conv"], p["turn"])].append(p)
    turns_of = collections.defaultdict(list)
    for t in sorted(turns, key=lambda t: (t["conv"], t["turn"])):
        turns_of[t["conv"]].append(t)
    out = args.out
    meta_rows, turn_rows, qe = [], [], collections.defaultdict(list)
    summary = {}
    for cond in conds:
        res = results[cond]
        missing = [p["pid"] for p in pieces if p["pid"] not in res]
        broken = {r["conv"] for r in res.values() if r["failed"]}
        unexplained = [pid for pid in missing if pid.split("|")[0] not in broken or cond == "noctx"]
        if unexplained:
            raise SystemExit(f"{cond}: {len(unexplained)} pieces without a translation, e.g. {unexplained[:2]}")
        c = collections.Counter()
        flags0, flags1 = collections.Counter(), collections.Counter()
        by_role = collections.defaultdict(collections.Counter)
        for p in pieces:
            r = res.get(p["pid"])
            if r is None:
                continue
            f0, f1 = json.loads(r["flags0"]), json.loads(r["flags"])
            st = structure_of(p["text"], r["text_nl"]) if not r["failed"] else None
            meta_rows.append({"cond": cond, "pid": p["pid"], "conv": p["conv"], "turn": p["turn"], "piece": p["piece"],
                              "role": p["role"], "stratum": p["stratum"], "worker": p["worker"], "cls": r["cls"],
                              "wave": r["wave"], "tokens": p["tokens"], "tokens_in": r["tokens_in"],
                              "tokens_out": r["tokens_out"], "chars_en": len(p["text"]), "chars_nl": len(r["text_nl"]),
                              "ended": r["ended"], "attempt": r["attempt"], "flags0": r["flags0"], "flags": r["flags"],
                              "failed": r["failed"], "context_items": r["context_items"],
                              "context_tokens": r["context_tokens"], "context_cut": r["context_cut"],
                              "structure": json.dumps(st)})
            c["pieces"] += 1
            c["failed0"] += bool(f0)
            c["failed"] += bool(f1)
            c["retried"] += r["attempt"]
            for f in f0:
                flags0[f] += 1
            for f in f1:
                flags1[f] += 1
            by_role[p["role"]]["pieces"] += 1
            by_role[p["role"]]["failed0"] += bool(f0)
            by_role[p["role"]]["failed"] += bool(f1)
            if r["text_nl"].strip():
                qe[p["worker"]].append({"id": f"{cond}|{p['pid']}", "source": p["text"], "hypothesis": r["text_nl"]})
        complete = 0
        chat = []
        tc = collections.Counter()
        for conv in convs:
            ts = turns_of[conv["conv"]]
            nl_turns, ok = [], True
            for t in ts:
                ps = pieces_of[(conv["conv"], t["turn"])]
                rs = [res.get(p["pid"]) for p in ps]
                if any(r is None or r["failed"] for r in rs):
                    ok = False
                    turn_rows.append({"cond": cond, "conv": conv["conv"], "turn": t["turn"], "role": t["role"],
                                      "text_nl": None, "structure": None})
                    continue
                text_nl = assemble(t["lead"], [r["text_nl"] for r in rs], [p["sep_after"] for p in ps], t["trail"])
                st = structure_of(t["text"], text_nl)
                tc["turns"] += 1
                tc["fences_en"] += st["fences_en"]
                tc["turns_fences"] += bool(st["fences_en"] or st["fences_nl"])
                tc["turns_fences_same"] += bool(st["fences_same"])
                tc["turns_headings"] += bool(st["headings_en"] or st["headings_nl"])
                tc["turns_headings_same"] += bool((st["headings_en"] or st["headings_nl"])
                                                  and st["headings_en"] == st["headings_nl"])
                tc["turns_items"] += bool(st["items_en"] or st["items_nl"])
                tc["turns_items_same"] += bool((st["items_en"] or st["items_nl"]) and st["items_en"] == st["items_nl"])
                tc["turns_rows"] += bool(st["rows_en"] or st["rows_nl"])
                tc["turns_rows_same"] += bool((st["rows_en"] or st["rows_nl"]) and st["rows_en"] == st["rows_nl"])
                tc["turns_inline"] += bool(st["inline_en"])
                tc["turns_inline_same"] += bool(st["inline_en"] and st["inline_same"])
                tc["turns_any_diff"] += bool(st["diffs"])
                for d in st["diffs"]:
                    tc[f"diff_{d}"] += 1
                turn_rows.append({"cond": cond, "conv": conv["conv"], "turn": t["turn"], "role": t["role"],
                                  "text_nl": text_nl, "structure": json.dumps(st)})
                nl_turns.append({"role": t["role"], "content": text_nl})
            if ok:
                complete += 1
                chat.append({"uuid": conv["conv"], "seed_dataset": conv["seed"], "messages": nl_turns,
                             "train_turns": json.loads(conv["train_turns"]),
                             "metadata": {"source": f"{DATASET}@{REVISION}", "generator": "zai-org/GLM-5",
                                          "translator": "G4-8-s140 (Gemma 4 31B-it + LoRA)",
                                          "prompt": "natural-v1-chat" if cond == "ctx" else "natural-v1",
                                          "context": cond == "ctx"}})
        chat_bytes = "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in chat).encode()
        write_bytes(join(out, f"chat-nl-{cond}.jsonl"), chat_bytes)
        summary[cond] = {"pieces": dict(c), "flags0": dict(flags0), "flags": dict(flags1),
                         "by_role": {k: dict(v) for k, v in by_role.items()}, "convs": len(convs),
                         "complete": complete, "broken": len(broken), "missing_pieces": len(missing),
                         "turns": dict(tc), "chat_bytes": len(chat_bytes),
                         "chat_bytes_en": sum(len(t["text"].encode()) for t in turns)}
    write_rows(meta_rows, join(out, "pieces-result-meta.parquet"))
    write_rows(turn_rows, join(out, "turns-nl.parquet"))
    for k in workers:
        if not qe.get(k):
            raise SystemExit(f"no piece with a translation on w{k}: nothing for its QE")
        write_rows(qe[k], join(out, f"qe-in-w{k}.parquet"))
    info = {"protocol": PROTOCOL, "conds": conds, "summary": summary, "done": stats,
            "qe_rows": {f"w{k}": len(qe.get(k, [])) for k in workers}}
    write_json(info, join(out, "collect.json"))
    print(f"NEMOCOLLECT {json.dumps({c: {k: summary[c][k] for k in ('complete', 'convs', 'broken')} | summary[c]['pieces'] for c in summary})}",
          flush=True)
    return info


# ---------------------------------------------------------------------------------------------------- report

def wilson(k, n, z=1.96):
    if not n:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (c - h, c + h)


def pct(k, n):
    return f"{100 * k / n:.1f}%" if n else "–"


def kn(k, n):
    return f"{k}/{n} ({pct(k, n)})"


def read_qe(pattern):
    import glob

    out = {}
    for path in sorted(glob.glob(pattern)):
        for r in read_rows(path):
            if r["id"] in out:
                raise SystemExit(f"{path}: QE of {r['id']} twice")
            out[r["id"]] = r["qe_source_hypothesis"]
    return out


def turn_qe(meta, qe, cond):
    """{(conv, turn): QE of the turn}: its pieces' QE weighted by their English tokens (None if a piece has none)"""
    by = collections.defaultdict(list)
    for m in meta:
        if m["cond"] == cond:
            by[(m["conv"], m["turn"])].append((qe.get(f"{cond}|{m['pid']}"), m["tokens"], m["failed"]))
    out = {}
    for key, xs in by.items():
        if any(q is None or f for q, _, f in xs):
            continue
        out[key] = sum(q * n for q, n, _ in xs) / sum(n for _, n, _ in xs)
    return out


def extrapolate(pool_meta, rates, n=100_000):
    """Hours on one v4-32 (4 workers) and on two for n conversations drawn like the pool: the pool's mean English tokens
    per conversation per class, times the measured ratio of output to input tokens, over the measured output tokens per
    second of a worker (rates: {cls: (out_per_s, out_per_in)})"""
    m = len(pool_meta)
    short = sum(r["tokens_short"] for r in pool_meta) / m
    long_ = sum(r["tokens_long"] for r in pool_meta) / m
    def seconds(cls, tokens):
        if not tokens:
            return 0.0
        if cls not in rates:
            raise SystemExit(f"the pool has {tokens:.0f} tokens per conversation for {cls}, and the trial no rate of it")
        return n * tokens * rates[cls][1] / rates[cls][0]

    est = {"noctx": seconds("n128", short) + seconds("n512", long_), "ctx": seconds("c2k", short + long_)}
    return {k: {"worker_hours": v / 3600, "hours_4": v / 3600 / 4, "hours_8": v / 3600 / 8} for k, v in est.items()} | {
        "tokens_en_per_conv": short + long_, "tokens_short_per_conv": short, "tokens_long_per_conv": long_,
        "pool": m}


def report(args):
    data, col = Path(args.data), Path(args.collect)
    sample_info = json.loads((data / "sample.json").read_text())
    info = json.loads((col / "collect.json").read_text())
    meta = read_rows(col / "pieces-result-meta.parquet")
    pool_meta = read_rows(data / "pool-meta.parquet")
    qe = read_qe(args.qe)
    s = info["summary"]
    lines = [f"# Nemotron Chat v3, proef met G4-8-s140 ({PROTOCOL})", "",
             "Gemaakt door `nemotron_chat.py report` uit tellingen en scores; geen tekst gelezen of afgedrukt.", ""]
    # the pool
    left = sample_info["left_out"]
    lines += ["## Bron en filters", "",
              f"{sample_info['ranges']} bytebereiken ({sample_info['range_bytes'] / 1e6:.0f} MB van "
              f"{sample_info['file_bytes'] / 1e9:.2f} GB `data/chat.jsonl`, `{DATASET}@{REVISION[:7]}`), "
              f"{sample_info['rows']} hele gesprekken.", "",
              "| herkomst eerste vraag | gesprekken |", "|---|---:|"]
    lines += [f"| {k} | {v} ({pct(v, sample_info['rows'])}) |" for k, v in sorted(sample_info["seeds"].items(), key=str)]
    n_lm = sum(v for k, v in sample_info["seeds"].items() if k in LMARENA)
    lines += ["", f"Van de {n_lm} lmarena-gesprekken vallen af:", "", "| reden | gesprekken | van lmarena |", "|---|---:|---:|"]
    lines += [f"| {k} | {v} | {pct(v, n_lm)} |" for k, v in sorted(left.items()) if k != "seed_not_lmarena"]
    lines += ["", f"Over: **{sample_info['pool']}** ({pct(sample_info['pool'], n_lm)} van lmarena). Gekozen: "
              f"{sample_info['chosen']} gesprekken, {sample_info['turns']} beurten, {sample_info['pieces']} stukken "
              f"({sample_info['pieces_short']} van ≤ {SHORT_MAX} tokens), {sample_info['tokens']} Engelse tokens.", "",
              "| laag | pool | proef |", "|---|---:|---:|"]
    for k in sorted(set(sample_info["pool_strata"]) | set(sample_info["chosen_strata"])):
        lines.append(f"| {k} | {sample_info['pool_strata'].get(k, 0)} | {sample_info['chosen_strata'].get(k, 0)} |")
    # throughput
    lines += ["", "## Tempo", "", "| conditie | klasse | worker | stukken | golven | laden (s) | decode (s) | herhaling (s) | "
              "tokens in | tokens uit | uit/s | uit/s volle golven |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    rates = {}
    agg = collections.defaultdict(collections.Counter)
    for d in sorted(info["done"], key=lambda d: (d["cond"], d["cls"], d["worker"])):
        full = d.get("full_tokens_out", 0) / d["full_seconds"] if d.get("full_seconds") else None
        lines.append(f"| {d['cond']} | {d['cls']} | w{d['worker']} | {d['units']} | {d.get('waves', 0)} | "
                     f"{d.get('engine_seconds', 0):.0f} | {d.get('seconds', 0):.0f} | {d.get('retry_seconds', 0):.0f} | "
                     f"{d.get('tokens_in', 0)} | {d.get('tokens_out', 0)} | "
                     f"{d.get('tokens_out', 0) / max(d.get('seconds', 0), 1e-9):.0f} | {'' if full is None else f'{full:.0f}'} |")
        a = agg[d["cls"]]
        for k in ("seconds", "retry_seconds", "tokens_in", "tokens_out", "full_tokens_out", "full_seconds", "engine_seconds"):
            a[k] += d.get(k, 0)
    src_tokens = collections.Counter()
    for m in meta:
        src_tokens[m["cls"]] += m["tokens"]
    for cls, a in agg.items():
        if not a["tokens_out"]:
            continue
        sec = a["full_seconds"] if cls == "c2k" and a["full_seconds"] else a["seconds"]
        out_tok = a["full_tokens_out"] if cls == "c2k" and a["full_seconds"] else a["tokens_out"]
        rates[cls] = (out_tok / max(sec, 1e-9), a["tokens_out"] / max(src_tokens[cls], 1))
    per_conv = {}
    for cond in ("ctx", "noctx"):
        sec = sum(a["seconds"] + a["retry_seconds"] for cls, a in agg.items() if CLASSES[cls][0] == cond)
        per_conv[cond] = sec / s[cond]["convs"]
    lines += ["", "Per gesprek (decode en herhaling, opgeteld over de workers, gedeeld door het aantal gesprekken): "
              + ", ".join(f"{c} {v:.1f} s" for c, v in per_conv.items()) + ".", ""]
    # QE
    lines += ["## MetricX-24 QE (lager is beter)", "", "| conditie | rol | stukken met QE | gemiddeld | mediaan | p90 | > 5 | > 10 |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    qe_stats = {}
    for cond in ("ctx", "noctx"):
        for role in ("user", "assistant", None):
            xs = [qe[f"{cond}|{m['pid']}"] for m in meta if m["cond"] == cond and not m["failed"]
                  and (role is None or m["role"] == role) and qe.get(f"{cond}|{m['pid']}") is not None]
            if not xs:
                continue
            q = quantiles(xs)
            qe_stats[f"{cond}/{role or 'all'}"] = q
            lines.append(f"| {cond} | {role or 'alle'} | {len(xs)} | {statistics.mean(xs):.2f} | {q['p50']:.2f} | "
                         f"{q['p90']:.2f} | {pct(sum(x > 5 for x in xs), len(xs))} | {pct(sum(x > 10 for x in xs), len(xs))} |")
    paired = collections.Counter()
    diffs = []
    for m in meta:
        if m["cond"] != "ctx" or m["failed"]:
            continue
        a, b = qe.get(f"ctx|{m['pid']}"), qe.get(f"noctx|{m['pid']}")
        if a is None or b is None:
            continue
        diffs.append(a - b)
        paired["ctx beter" if a < b - 0.1 else "noctx beter" if b < a - 0.1 else "gelijk (|Δ| ≤ 0,1)"] += 1
    if diffs:
        lines += ["", f"Gepaard per stuk ({len(diffs)} stukken met beide QE's): Δ = ctx − noctx gemiddeld "
                  f"{statistics.mean(diffs):+.3f}, mediaan {statistics.median(diffs):+.3f}; "
                  + ", ".join(f"{k} {kn(v, len(diffs))}" for k, v in sorted(paired.items())) + "."]
    tq = {c: turn_qe(meta, qe, c) for c in ("ctx", "noctx")}
    both = sorted(set(tq["ctx"]) & set(tq["noctx"]))
    if both:
        d = [tq["ctx"][k] - tq["noctx"][k] for k in both]
        lines += [f"Per beurt ({len(both)} beurten, QE per stuk gewogen naar Engelse tokens): Δ gemiddeld "
                  f"{statistics.mean(d):+.3f}, mediaan {statistics.median(d):+.3f}."]
    # failures
    lines += ["", "## Mislukte stukken", "", "| conditie | stukken | 1e poging mislukt | na herhaling mislukt | gesprekken heel |",
              "|---|---:|---:|---:|---:|"]
    for cond in ("ctx", "noctx"):
        p = s[cond]["pieces"]
        lines.append(f"| {cond} | {p.get('pieces', 0)} | {kn(p.get('failed0', 0), p.get('pieces', 0))} | "
                     f"{kn(p.get('failed', 0), p.get('pieces', 0))} | {kn(s[cond]['complete'], s[cond]['convs'])} |")
    flags = sorted(set(s["ctx"]["flags0"]) | set(s["noctx"]["flags0"]) | set(s["ctx"]["flags"]) | set(s["noctx"]["flags"]))
    lines += ["", "| controle | ctx 1e poging | ctx na herhaling | noctx 1e poging | noctx na herhaling |", "|---|---:|---:|---:|---:|"]
    for f in flags:
        lines.append(f"| {f} | {s['ctx']['flags0'].get(f, 0)} | {s['ctx']['flags'].get(f, 0)} | "
                     f"{s['noctx']['flags0'].get(f, 0)} | {s['noctx']['flags'].get(f, 0)} |")
    lines += ["", "| conditie | rol | stukken | 1e poging mislukt | na herhaling mislukt |", "|---|---|---:|---:|---:|"]
    for cond in ("ctx", "noctx"):
        for role, v in sorted(s[cond]["by_role"].items()):
            lines.append(f"| {cond} | {role} | {v['pieces']} | {kn(v['failed0'], v['pieces'])} | {kn(v['failed'], v['pieces'])} |")
    # structure
    lines += ["", "## Codeblokken en markdown (beurten die heel zijn)", "",
              "| conditie | beurten | codeblokken gelijk (byte voor byte) | koppen gelijk | lijstitems gelijk | tabelrijen gelijk | "
              "inline code gelijk | een categorie wijkt af |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for cond in ("ctx", "noctx"):
        t = s[cond]["turns"]
        lines.append(f"| {cond} | {t.get('turns', 0)} | {kn(t.get('turns_fences_same', 0), t.get('turns_fences', 0))} | "
                     f"{kn(t.get('turns_headings_same', 0), t.get('turns_headings', 0))} | "
                     f"{kn(t.get('turns_items_same', 0), t.get('turns_items', 0))} | "
                     f"{kn(t.get('turns_rows_same', 0), t.get('turns_rows', 0))} | "
                     f"{kn(t.get('turns_inline_same', 0), t.get('turns_inline', 0))} | "
                     f"{kn(t.get('turns_any_diff', 0), t.get('turns', 0))} |")
    cats = sorted({k for c in ("ctx", "noctx") for k in s[c]["turns"] if k.startswith("diff_")})
    if cats:
        lines += ["", "| categorie wijkt af (beurten) | ctx | noctx |", "|---|---:|---:|"]
        lines += [f"| {k[5:]} | {s['ctx']['turns'].get(k, 0)} | {s['noctx']['turns'].get(k, 0)} |" for k in cats]
    # context
    cm = [m for m in meta if m["cond"] == "ctx"]
    with_ctx = [m for m in cm if m["context_items"]]
    lines += ["", "## Context (ctx)", "", f"Stukken met context: {kn(len(with_ctx), len(cm))}; contexttokens "
              f"{json.dumps(quantiles([m['context_tokens'] for m in with_ctx]))}; ingekort (budget {CTX_TOKENS}): "
              f"{kn(sum(m['context_cut'] for m in with_ctx), len(with_ctx))}; prompttokens "
              f"{json.dumps(quantiles([m['tokens_in'] for m in cm]))}."]
    ratio = {c: quantiles([m["chars_nl"] / m["chars_en"] for m in meta if m["cond"] == c and not m["failed"]
                           and m["chars_en"] >= 30]) for c in ("ctx", "noctx")}
    lines += ["", "Lengteverhouding (tekens nl/en, stukken ≥ 30 tekens): " + "; ".join(
        f"{c} p5 {q.get('p5', 0):.2f}, p50 {q.get('p50', 0):.2f}, p95 {q.get('p95', 0):.2f}" for c, q in ratio.items()) + "."]
    # extrapolation
    ex = extrapolate(pool_meta, rates)
    lines += ["", "## Schatting voor 100.000 gesprekken", "",
              f"Pool: gemiddeld {ex['tokens_en_per_conv']:.0f} Engelse tokens per gesprek ({ex['tokens_short_per_conv']:.0f} "
              f"in stukken ≤ {SHORT_MAX}, {ex['tokens_long_per_conv']:.0f} in langere). Gemeten tempo per worker (uit/s; "
              f"uittokens per Engelse token): " + ", ".join(f"{k} {v[0]:.0f}; {v[1]:.2f}" for k, v in sorted(rates.items()))
              + (" (c2k: alleen golven met minstens zoveel stukken als slots)" if "c2k" in rates else "") + ".", "",
              "| conditie | workeruren | uren op mbert-mix (4 workers) | uren op 2 × v4-32 (8 workers) |", "|---|---:|---:|---:|"]
    for cond in ("ctx", "noctx"):
        if cond in ex:
            e = ex[cond]
            lines.append(f"| {cond} | {e['worker_hours']:.0f} | {e['hours_4']:.1f} | {e['hours_8']:.1f} |")
    lines += ["", "Zonder laadtijd van de engine (~2 min per proces), QE (MetricX) en de herhalingen; zie het rapport."]
    bytes_per = {c: s[c]["chat_bytes"] / max(s[c]["complete"], 1) for c in ("ctx", "noctx")}
    lines += ["", "Opslag: " + ", ".join(f"{c} {v / 1e3:.1f} kB per heel gesprek (JSONL, Nederlands)" for c, v in bytes_per.items())
              + f"; × 100.000 ≈ {max(bytes_per.values()) * 1e5 / 1e9:.1f} GB."]
    report_json = {"rates": rates, "per_conv_seconds": per_conv, "qe": qe_stats, "extrapolation": ex,
                   "paired": dict(paired), "paired_mean": statistics.mean(diffs) if diffs else None,
                   "bytes_per_conv": bytes_per, "summary": s}
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    Path(args.out).with_suffix(".json").write_text(json.dumps(report_json, indent=1) + "\n")
    print(f"NEMOREPORT {json.dumps({'rates': rates, 'extrapolation': {k: v for k, v in ex.items() if k in ('ctx', 'noctx')}})}",
          flush=True)
    return report_json


# ---------------------------------------------------------------------------------------------------- blind

N_BLIND = 40
BLIND_SEED = "nemotron-chat-v3/blind"
BLIND_CONTEXT_TURNS = 3
REVIEWER = """# Blinde beoordeling

In deze map staat `judge-input.jsonl`: {n} regels, per regel één JSON-object met `blind_id`, `context` (de eerdere
beurten van een gesprek tussen een gebruiker en een AI-assistent, hoogstens de laatste {k}: per beurt `rol`, de Engelse
tekst `en`, en de Nederlandse vertaling van die beurt bij A, `A`, en bij B, `B`), `rol` en `source` (de Engelse beurt
die beoordeeld wordt), en `A` en `B` (twee Nederlandse vertalingen van die beurt).

Beoordeel elk paar volgens de instructie hieronder. Schrijf je oordelen naar `judgments.jsonl` in deze map: één
JSON-regel per `blind_id`, voor alle {n} paren, in de volgorde van de invoer, en verder niets.

## Instructie

```
Je beoordeelt vertalingen uit het Engels in het Nederlands van beurten uit gesprekken tussen een gebruiker en een
AI-assistent. Per beurt krijg je de eerdere beurten (Engels, en per kant de Nederlandse vertaling die bij die kant
hoort), de Engelse beurt zelf en twee Nederlandse vertalingen, A en B. Je weet niet hoe of door wie ze zijn gemaakt;
probeer dat niet te raden.

Beoordeel elke beurt in deze volgorde:

1. Betekenisfout, apart voor A en voor B: "ja" of "nee". Een betekenisfout is:
   - een weglating: iets uit de bron ontbreekt in de vertaling;
   - een toevoeging: de vertaling zegt iets wat niet in de bron staat (ook: de vraag beantwoorden in plaats van haar
     vertalen);
   - een verkeerde betekenis: de vertaling zegt iets anders dan de bron;
   - een misplaatste uitdrukking: een idioom of vaste uitdrukking uit de bron is letterlijk vertaald waar dat in
     het Nederlands niet werkt, of vertaald met een Nederlandse uitdrukking die iets anders betekent;
   - een codeblok of opmaak (koppen, lijsten, tabellen) die niet behouden is.
   Woordkeus, stijl en zinsbouw zijn geen betekenisfout zolang de betekenis klopt.
2. Oordeel: "A beter", "B beter" of "gelijk". Eerst op juistheid: de vertaling zonder betekenisfout, of met de
   minst ernstige, is beter. Zijn ze even juist, dan op hoe goed de vertaling in het gesprek past (dezelfde
   aanspreekvorm, namen en termen als in de eigen eerdere beurten van die kant, kloppende verwijzingen naar eerdere
   beurten) en op natuurlijk, oorspronkelijk klinkend Nederlands. "Gelijk" als er op deze punten geen duidelijk
   verschil is.
3. Zekerheid: "zeker" of "twijfel", over je hele oordeel over deze beurt (de betekenisfouten en de keuze).
   Twijfel je aan één van die drie onderdelen, kies dan "twijfel".
4. Uitleg: één zin die je oordeel onderbouwt.

Geef per beurt één JSON-regel, en verder niets:
{{"blind_id": "...", "fout_A": "ja|nee", "fout_B": "ja|nee", "oordeel": "A beter|B beter|gelijk", "zekerheid": "zeker|twijfel", "uitleg": "..."}}
```
"""


def norm(text):
    return " ".join((text or "").split())


def blind(args):
    data, col = Path(args.data), Path(args.collect)
    turns = {(t["conv"], t["turn"]): t for t in read_rows(data / "turns.parquet")}
    nl = {(r["cond"], r["conv"], r["turn"]): r["text_nl"] for r in read_rows(col / "turns-nl.parquet")}
    eligible = collections.defaultdict(list)
    counts = collections.Counter()
    for (conv, turn), t in turns.items():
        if turn == 0:
            continue
        sides = [[nl.get((c, conv, k)) for k in range(turn + 1)] for c in ("ctx", "noctx")]
        if any(x is None for side in sides for x in side):
            counts["not_complete_up_to_it"] += 1
            continue
        if norm(sides[0][-1]) == norm(sides[1][-1]):
            counts["same_text"] += 1
            continue
        eligible[t["role"]].append((stable(f"{BLIND_SEED}/{conv}/{turn}"), conv, turn))
    chosen, used = [], set()
    for role in ("user", "assistant"):
        for _, conv, turn in sorted(eligible[role]):
            if len([c for c in chosen if c[2] == role]) == N_BLIND // 2:
                break
            if conv in used:
                continue
            used.add(conv)
            chosen.append((conv, turn, role))
    if len(chosen) != N_BLIND:
        raise SystemExit(f"only {len(chosen)} turns for the blind package (eligible {({k: len(v) for k, v in eligible.items()})}, "
                         f"{dict(counts)})")
    rng = random.Random(BLIND_SEED)
    rng.shuffle(chosen)
    items, key = [], []
    for conv, turn, role in chosen:
        bid = stable(f"{BLIND_SEED}/id/{conv}/{turn}")[:12]
        a_is_ctx = rng.random() < 0.5
        a, b = ("ctx", "noctx") if a_is_ctx else ("noctx", "ctx")
        context = [{"rol": DUTCH_LABELS[turns[(conv, k)]["role"]], "en": turns[(conv, k)]["text"],
                    "A": nl[(a, conv, k)], "B": nl[(b, conv, k)]} for k in range(max(0, turn - BLIND_CONTEXT_TURNS), turn)]
        items.append({"blind_id": bid, "context": context, "rol": DUTCH_LABELS[role], "source": turns[(conv, turn)]["text"],
                      "A": nl[(a, conv, turn)], "B": nl[(b, conv, turn)]})
        key.append({"blind_id": bid, "A": "ctx" if a_is_ctx else "base", "B": "base" if a_is_ctx else "ctx",
                    "set": role, "base_system": "noctx", "conv": conv, "turn": turn,
                    "context_turns": len(context), "earlier_turns": turn})
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "judge-input.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in items),
                                           encoding="utf-8")
    (out / "REVIEWER.md").write_text(REVIEWER.format(n=N_BLIND, k=BLIND_CONTEXT_TURNS), encoding="utf-8")
    Path(args.key).write_text("".join(json.dumps(x) + "\n" for x in key))
    info = {"items": len(items), "by_role": dict(collections.Counter(k["set"] for k in key)),
            "a_is_ctx": sum(k["A"] == "ctx" for k in key), "eligible": {k: len(v) for k, v in eligible.items()},
            "left_out": dict(counts)}
    (Path(args.key).parent / "blind.json").write_text(json.dumps(info, indent=1) + "\n")
    print(f"NEMOBLIND {json.dumps(info)}", flush=True)
    return info


# ---------------------------------------------------------------------------------------------------- the full run

def hf_token():
    """The HF token of this VM (~/.cache/huggingface/token, put there through stdin); never printed"""
    path = Path.home() / ".cache" / "huggingface" / "token"
    token = path.read_text().strip() if path.exists() else ""
    if not token:
        raise SystemExit(f"no HF token in {path}")
    return token


def http_range(url, start, end, token, tries=5):
    """Bytes start..end (inclusive) of url, retried with a pause; exactly that many bytes or an error"""
    import urllib.request

    for attempt in range(tries):
        try:
            request = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}",
                                                           "Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(request, timeout=600) as response:
                data = response.read()
            if len(data) != end - start + 1:
                raise IOError(f"{len(data)} bytes for the range {start}-{end}")
            return data
        except Exception as error:  # noqa: BLE001 - every failure is retried, the last one raised
            if attempt == tries - 1:
                raise
            print(f"range {start}-{end}: {error!r}, again in {30 * (attempt + 1)} s", flush=True)
            time.sleep(30 * (attempt + 1))
    raise AssertionError("unreachable")


def shard_lines(r, fetch, size=FILE_BYTES, shard_bytes=SHARD_BYTES, margin=FETCH_MARGIN):
    """The lines of the file whose first byte lies in shard r's bytes [r x shard_bytes, (r + 1) x shard_bytes), each
    without its newline; fetch(start, end) gives bytes start..end inclusive. Over all shards every line once."""
    start, end = r * shard_bytes, min((r + 1) * shard_bytes, size)
    if start >= size:
        return []
    s0 = max(start - 1, 0)
    data = fetch(s0, min(end + margin, size) - 1)
    if start == 0:
        pos = 0
    else:
        k = data.find(b"\n")
        while k < 0:
            if s0 + len(data) >= size:
                return []
            data += fetch(s0 + len(data), min(s0 + len(data) + margin, size) - 1)
            k = data.find(b"\n")
        pos = k + 1
    lines = []
    while s0 + pos < end:
        k = data.find(b"\n", pos)
        while k < 0:
            if s0 + len(data) >= size:
                k = len(data)
                break
            data += fetch(s0 + len(data), min(s0 + len(data) + margin, size) - 1)
            k = data.find(b"\n", pos)
        lines.append(data[pos:k])
        pos = k + 1
    return [line for line in lines if line.strip()]


def seed_messages(conversation):
    """(system, first user content) of a WildChat-1M row's conversation, as prepare_chat_prompts.py of the Nemotron
    repo @ REVISION (extract_source_seed_messages): the first system content, the first user content, else the first
    message's content"""
    system = user = None
    for m in conversation or []:
        if not isinstance(m, dict) or m.get("content") is None:
            continue
        if m.get("role") == "system" and system is None:
            system = m["content"]
        if m.get("role") == "user":
            user = m["content"]
            break
    if user is None and conversation and isinstance(conversation[0], dict) and isinstance(conversation[0].get("content"), str):
        user = conversation[0]["content"]
    return system, user


def prompt_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def restore_row(row, prompts):
    """A WildChat row with its withheld system and first user content put back from prompts {sha256: (system, user)}
    (prepare_chat_prompts.py's restore_chat_jsonl); None when its hash is not there. Other rows unchanged."""
    meta = row.get("metadata") or {}
    if meta.get("seed_dataset") != WILDCHAT:
        return row
    found = prompts.get(meta.get("seed_prompt_sha256"))
    if found is None:
        return None
    messages = [dict(m) for m in row.get("messages") or []]
    if messages and messages[0].get("role") == "system":
        messages[0]["content"] = found[0]
    first = next((k for k, m in enumerate(messages) if m.get("role") == "user"), None)
    if first is None:
        return None
    messages[first]["content"] = found[1]
    return dict(row, messages=messages)


def wildchat(args):
    """The first prompts of WildChat-1M at WILDCHAT_REVISION, straight from HF (huggingface_hub's file system, nothing
    on disk), as 16 parquet files by the first hex digit of their sha256: {sha256, system, user}"""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem(token=hf_token())
    files = sorted(fs.glob(f"datasets/{WILDCHAT}@{WILDCHAT_REVISION}/data/*.parquet"))
    if len(files) != WILDCHAT_FILES:
        raise SystemExit(f"{WILDCHAT}@{WILDCHAT_REVISION}: {len(files)} parquet files, not {WILDCHAT_FILES}")
    per = collections.defaultdict(dict)
    counts = collections.Counter()
    for f in files:
        with fs.open(f, "rb") as handle:
            for batch in pq.ParquetFile(handle).iter_batches(columns=["conversation"], batch_size=4096):
                for conv in batch.column(0).to_pylist():
                    counts["rows"] += 1
                    system, user = seed_messages(conv)
                    if user is None:
                        counts["no_user"] += 1
                        continue
                    h = prompt_hash(user)
                    if h in per[h[0]]:
                        counts["same_prompt"] += 1
                        continue
                    per[h[0]][h] = (system, user)
        print(f"WILDCHAT {json.dumps({'file': f.rsplit('/', 1)[-1], **counts})}", flush=True)
    for d in "0123456789abcdef":
        write_rows([{"sha256": h, "system": v[0], "user": v[1]} for h, v in sorted(per[d].items())],
                   join(args.out, f"prompts-{d}.parquet"))
    info = {"dataset": WILDCHAT, "revision": WILDCHAT_REVISION, "files": len(files), **counts,
            "prompts": sum(len(v) for v in per.values())}
    write_json(info, join(args.out, "wildchat.json"))
    print(f"NEMOWILDCHAT {json.dumps(info)}", flush=True)
    return info


def read_prompts(root):
    out = {}
    for d in "0123456789abcdef":
        for r in read_rows(join(root, f"prompts-{d}.parquet")):
            out[r["sha256"]] = (r["system"], r["user"])
    return out


def shards_of(worker, workers=WORKERS, shards=SHARDS):
    return [r for r in range(shards) if r % workers == worker]


def prepare_shard(lines, r, count, identify, prompts):
    """The tables of shard r: every conversation that passes the filters (no draw), all on worker r % WORKERS"""
    worker = r % WORKERS
    reasons, seeds = collections.Counter(), collections.Counter()
    conversations, turns, pieces, pool = [], [], [], []
    for line in lines:
        row = json.loads(line.decode("utf-8"))
        seeds[(row.get("metadata") or {}).get("seed_dataset")] += 1
        restored = restore_row(row, prompts)
        if restored is None:
            reasons["wildchat_not_restored"] += 1
            continue
        conv, why = conversation(restored, FULL_SEEDS)
        if conv is not None:
            conv, why = prepare(conv, count, identify)
        if why:
            reasons[why] += 1
            continue
        pool.append(conv)
        c = conv
        conversations.append({"conv": c["uuid"], "seed": c["seed"], "stratum": c["stratum"], "turns": len(c["turns"]),
                              "assistant": c["n_assistant"], "tokens": c["tokens"], "worker": worker,
                              "train_turns": c["train_turns"], "reasoning_chars": c["reasoning_chars"], "shard": r})
        for i, t in enumerate(c["turns"]):
            turns.append({"conv": c["uuid"], "turn": i, "role": t["role"], "message": t["message"], "text": t["text"],
                          "lead": t["lead"], "trail": t["trail"], "pieces": len(t["pieces"]), "tokens": t["tokens"],
                          "worker": worker})
            for j, p in enumerate(t["pieces"]):
                pieces.append({"pid": f"{c['uuid']}|t{i}|p{j}", "conv": c["uuid"], "turn": i, "piece": j,
                               "n_pieces": len(t["pieces"]), "role": t["role"], "text": p["text"],
                               "sep_after": p["sep"], "tokens": p["tokens"], "worker": worker, "stratum": c["stratum"]})
    info = {"shard": r, "worker": worker, "rows": len(lines), "seeds": dict(seeds), "left_out": dict(reasons),
            "kept": len(conversations), "kept_seeds": dict(collections.Counter(c["seed"] for c in conversations)),
            "turns": len(turns), "pieces": len(pieces), "pieces_short": sum(p["tokens"] <= SHORT_MAX for p in pieces),
            "tokens": sum(c["tokens"] for c in conversations),
            "tokens_short": sum(p["tokens"] for p in pieces if p["tokens"] <= SHORT_MAX),
            "tokens_long": sum(p["tokens"] for p in pieces if p["tokens"] > SHORT_MAX),
            "chars_en": sum(len(t["text"].encode()) for t in turns)}
    return conversations, turns, pieces, [pool_meta_row(c) for c in pool], info


def prepare_run(args):
    """Every shard of this worker (r = worker mod 4) that has no prepare.json yet: its lines straight from HF (HTTP
    ranges at REVISION, nothing on disk), WildChat's prompts put back, the filters, the pieces, to <out>/shard-<r>/"""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    def count(text):
        return len(tok(text, add_special_tokens=False)["input_ids"])

    identify = langid_identifier()
    token = hf_token()
    url = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/data/chat.jsonl"
    todo = [r for r in (shards_of(args.worker) if args.shards is None else [int(x) for x in args.shards.split(",")])
            if not exists(join(args.out, f"shard-{r:05d}", "prepare.json"))]
    print(f"prepare w{args.worker}: {len(todo)} shards to do", flush=True)
    prompts = read_prompts(args.wildchat) if todo else {}
    for r in todo:
        t0 = time.time()
        lines = shard_lines(r, lambda a, b: http_range(url, a, b, token))
        conversations, turns, pieces, pool, info = prepare_shard(lines, r, count, identify, prompts)
        d = join(args.out, f"shard-{r:05d}")
        write_rows(conversations, join(d, "conversations.parquet"))
        write_rows(turns, join(d, "turns.parquet"))
        write_rows(pieces, join(d, "pieces.parquet"))
        write_rows([{k: v for k, v in p.items() if k not in ("text", "sep_after")} for p in pieces],
                   join(d, "pieces-meta.parquet"))
        write_rows(pool, join(d, "pool-meta.parquet"))
        info["seconds"] = round(time.time() - t0, 1)
        write_json(info, join(d, "prepare.json"))  # last: the shard's marker
        print(f"PREPARED {json.dumps({k: info[k] for k in ('shard', 'rows', 'kept', 'pieces', 'tokens', 'seconds')})}",
              flush=True)
    print(f"NEMOPREPARE {json.dumps({'worker': args.worker, 'done': len(todo)})}", flush=True)


def final_shard(chat_rows, meta, qe, cond, qe_max=QE_MAX):
    """(rows, counts): the shard's conversations of chat-nl-<cond>.jsonl with each message's QE (its pieces' MetricX-24
    QE weighted by English tokens); a conversation with a message without QE, or with a piece above qe_max, is left
    out (no gaps). qe_max None keeps them all."""
    tq = turn_qe(meta, qe, cond)
    worst = collections.defaultdict(float)
    for m in meta:
        if m["cond"] == cond:
            q = qe.get(f"{cond}|{m['pid']}")
            if q is not None:
                worst[m["conv"]] = max(worst[m["conv"]], q)
    out, counts = [], collections.Counter()
    for row in chat_rows:
        counts["complete"] += 1
        qs = [tq.get((row["uuid"], k)) for k in range(len(row["messages"]))]
        if any(q is None for q in qs):
            counts["no_qe"] += 1
            continue
        if qe_max is not None and worst[row["uuid"]] > qe_max:
            counts["qe_above_max"] += 1
            continue
        out.append(dict(row, messages=[dict(m, qe=round(q, 3)) for m, q in zip(row["messages"], qs)]))
    counts["kept"] = len(out)
    return out, dict(counts)


def final_run(args):
    """Every shard of this worker with a collect.json and QE scores and no final.json: out/chat-nl-<r>.jsonl"""
    qe_max = None if args.qe_max.lower() == "none" else float(args.qe_max)
    done = 0
    for r in shards_of(args.worker):
        c = join(args.collect, f"shard-{r:05d}")
        target = join(args.out, f"chat-nl-{r:05d}.jsonl")
        marker = join(args.out, f"final-{r:05d}.json")
        if exists(marker) or not exists(join(c, "collect.json")):
            continue
        scores = join(args.scores, f"qe-{r:05d}.parquet")
        if not exists(scores):
            raise SystemExit(f"shard {r}: collect.json but no {scores}")
        chat = [json.loads(x) for x in read_bytes(join(c, f"chat-nl-{args.cond}.jsonl")).decode().splitlines() if x]
        meta = read_rows(join(c, "pieces-result-meta.parquet"))
        qe = {x["id"]: x["qe_source_hypothesis"] for x in read_rows(scores)}
        rows, counts = final_shard(chat, meta, qe, args.cond, qe_max)
        write_bytes(target, "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows).encode())
        write_json({"shard": r, "cond": args.cond, "qe_max": qe_max, **counts}, marker)
        done += 1
        print(f"FINAL {json.dumps({'shard': r, **counts})}", flush=True)
    print(f"NEMOFINAL {json.dumps({'worker': args.worker, 'done': done})}", flush=True)


def summary_run(args):
    """The full run in numbers (text-free): per stage the shards done and the sums of their counts"""
    agg = {}
    for name, root, pattern in (("prepare", args.data, r"shard-\d+/prepare\.json"),
                                ("final", args.out, r"final-\d+\.json")):
        files = list_files(root, pattern)
        tot = collections.Counter()
        for f in files:
            for k, v in read_json(f).items():
                if isinstance(v, (int, float)) and not isinstance(v, bool) and k not in ("shard", "worker", "qe_max"):
                    tot[k] += v
                elif isinstance(v, dict):
                    for kk, vv in v.items():
                        if isinstance(vv, (int, float)):
                            tot[f"{k}.{kk}"] += vv
        agg[name] = {"shards": len(files), **dict(sorted(tot.items()))}
    gen = collections.Counter()
    for f in list_files(args.gen, r"(ctx|noctx)/(c2k|n128|n512)/shard-\d+/DONE\.json"):
        d = read_json(f)
        gen["shards"] += 1
        for k in ("units", "tokens_in", "tokens_out", "seconds", "retry_seconds", "failed", "broken"):
            gen[k] += d.get(k, 0)
    agg["gen"] = dict(gen)
    agg["shards_total"] = SHARDS
    write_json(agg, args.json)
    print(f"NEMOSUMMARY {json.dumps(agg)}", flush=True)
    return agg


def qe_join(args):
    """The QE inputs of some shards of this worker as one file (ids "<shard>#<id>"), for one MetricX run"""
    rows = []
    for r in [int(x) for x in args.shards.split(",")]:
        for x in read_rows(join(args.collect, f"shard-{r:05d}", f"qe-in-w{args.worker}.parquet")):
            rows.append(dict(x, id=f"{r:05d}#{x['id']}"))
    write_rows(rows, args.out)
    print(f"QEJOIN {json.dumps({'shards': args.shards, 'rows': len(rows)})}", flush=True)


def qe_split(args):
    """MetricX's scores of a qe_join file back per shard: <scores>/qe-<shard>.parquet with the ids of collect"""
    per = collections.defaultdict(list)
    for x in read_rows(args.scores):
        shard, sep, rest = x["id"].partition("#")
        if not sep:
            raise SystemExit(f"{args.scores}: id {x['id']!r} without a shard")
        per[shard].append(dict(x, id=rest))
    want = {f"{int(r):05d}" for r in args.shards.split(",")}
    if set(per) != want:
        raise SystemExit(f"{args.scores}: scores of shards {sorted(per)}, not {sorted(want)}")
    for shard, rows in per.items():
        write_rows(rows, join(args.out, f"qe-{shard}.parquet"))
    print(f"QESPLIT {json.dumps({k: len(v) for k, v in sorted(per.items())})}", flush=True)


# ---------------------------------------------------------------------------------------------------- main

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("sample")
    p.add_argument("--ranges-dir", required=True, help="a directory of part-<byte offset>.jsonl")
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--n", type=int, default=N_TRIAL)
    p.add_argument("--out", required=True)
    p = sub.add_parser("translate")
    p.add_argument("--pieces", default=None, help="the pieces (a local file or gs://), with --out")
    p.add_argument("--jobs", default=None, help="a file of lines '<pieces> <out>' (the full run: one per shard)")
    p.add_argument("--cond", required=True, choices=("ctx", "noctx"))
    p.add_argument("--cls", required=True, choices=sorted(CLASSES))
    p.add_argument("--worker", type=int, required=True)
    p.add_argument("--out", default=None, help="the bucket directory of this condition, class and worker")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--model", default="gemma4-31b-chat")
    p.add_argument("--lora", default=None)
    p.add_argument("--work-dir", required=True)
    p.add_argument("--sync-every", type=int, default=8)
    p = sub.add_parser("collect")
    p.add_argument("--data", required=True)
    p.add_argument("--gen", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--conds", default="ctx,noctx", help="the conditions translated (the full run: one)")
    p.add_argument("--leaf", default=r"w\d+", help="the directory of a class's output (the full run: shard-<r>)")
    p = sub.add_parser("report")
    p.add_argument("--data", required=True)
    p.add_argument("--collect", required=True)
    p.add_argument("--qe", required=True, help="glob of the QE files (id, qe_source_hypothesis)")
    p.add_argument("--out", required=True)
    p = sub.add_parser("blind")
    p.add_argument("--data", required=True)
    p.add_argument("--collect", required=True)
    p.add_argument("--out", required=True, help="the package's directory (judge-input.jsonl, REVIEWER.md)")
    p.add_argument("--key", required=True)
    p = sub.add_parser("wildchat")
    p.add_argument("--out", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--worker", type=int, required=True)
    p.add_argument("--shards", default=None, help="comma-separated shards (default: those of the worker)")
    p.add_argument("--wildchat", required=True, help="the prompts of `wildchat`")
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("final")
    p.add_argument("--worker", type=int, required=True)
    p.add_argument("--cond", required=True, choices=("ctx", "noctx"))
    p.add_argument("--collect", required=True)
    p.add_argument("--scores", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--qe-max", default=str(QE_MAX))
    p = sub.add_parser("qe-join")
    p.add_argument("--collect", required=True)
    p.add_argument("--shards", required=True)
    p.add_argument("--worker", type=int, required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("qe-split")
    p.add_argument("--scores", required=True)
    p.add_argument("--shards", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("summary")
    p.add_argument("--data", required=True)
    p.add_argument("--gen", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--json", required=True)
    args = parser.parse_args(argv)
    return {"sample": sample, "translate": translate_run, "collect": collect, "report": report, "blind": blind,
            "wildchat": wildchat, "prepare": prepare_run, "final": final_run, "summary": summary_run,
            "qe-join": qe_join, "qe-split": qe_split}[args.command](args)


if __name__ == "__main__":
    # As translate.py: a thread that is no daemon would keep a failed process alive, holding the TPU and never reaching
    # the job's EXIT line; so the process ends here with the error's traceback and code.
    import os
    import traceback

    code = 0
    try:
        main()
    except SystemExit as stop:
        code = stop.code if isinstance(stop.code, int) else (0 if stop.code is None else 1)
        if not isinstance(stop.code, int) and stop.code is not None:
            print(stop.code, file=sys.stderr)
    except BaseException:
        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
