#!/usr/bin/env python3
"""Length trial of 13 Nov (Yeb): the longest text a back-translation pair can have, by what the model can do and by
the quality of the translation. Gemma 4 31B-it (prompt natural-v1) translates contiguous stretches of real Dutch
documents of five length bins from Dutch into English, each stretch whole (one prompt) and in pieces (whole
paragraphs up to ~512 tokens per prompt, joined afterwards); TranslateGemma 27B translates that English back into
Dutch the same two ways (part B, on mbert-mix). The decision rule is in mbert-ops/translator-sft/bt-length-trial.md.

    python bt_length.py docs --tokenizer TOK --wordlist WORDS --out DIR [--per-source 8] [--bins 512,...,16384]
        [--sources raad]
    python bt_length.py docs-merge --inputs DIR ... --out DIR
    python bt_length.py segments --docs docs.parquet --tokenizer TOK --model gemma4-31b --doc-model gemma4-31b-doc \\
        --source nl --target en --workers 2 --out DIR
    python bt_length.py collect --docs docs.parquet --heel DIR --stukken DIR --source nl --target en --out DIR
    python bt_length.py pending --units units.parquet --scored qe-*.parquet --workers 2 --out DIR
    python bt_length.py report --measured measured.parquet --units units.parquet --qe qe-*.parquet --logs DIR \\
        --out bt-length.md
    python bt_length.py english --docs docs.parquet --measured measured.parquet --tokenizer TOK --out docs-en.parquet
    python bt_length.py merge --measured m1.parquet m2.parquet --units u1.parquet u2.parquet --out DIR

docs: per source (SOURCES of select_passages.py, read straight from Hugging Face, row group by row group) documents
that pass the document filters of the back-translation's selection; from each at most one stretch of whole,
consecutive paragraphs (the document's own paragraphs, as the selection reads them; in the Handelingen whole turns,
each opened by a paragraph with its speaker) whose joined text has more than LOW x and at most L Gemma tokens, for the
largest bin L that the source still needs. Every paragraph of a stretch holds at most MAX_PARAGRAPH_TOKENS (it must
fit a piece), at least MIN_PROSE_SHARE of its tokens are prose paragraphs, and the stretch as a whole passes the
selection's language, spelling and character filters. Per bin --per-source stretches of each source; a bin a source
cannot fill takes its remainder from the stretches the other sources found beyond theirs. With --sources a run takes
some of the sources (one process per source); docs-merge chooses over their stretches.parquet.

segments: translate.py's segments of both conditions, split over --workers: heel (a stretch whole, the length classes
of --doc-model) and stukken (the pieces of each stretch, --model's classes). A piece is the longest run of whole
paragraphs from where the last one ended whose joined text has at most PIECE_TOKENS tokens.

collect: per stretch and condition the translation (the pieces joined with an empty line), the checks of the pairs
(not ended, decode loop, source language left, structure, chatter), the paragraph counts and the alignment of the
translation's paragraphs to the source's (align), the omissions (a source paragraph aligned to nothing, or whose unit
lies outside R2's length bound of its source), and the units for MetricX-24-QE (units.parquet: id, src, cand).

pending: the units not scored yet (no id in --scored), split over --workers for metricx_qe.py.

report: the tables per bin and condition, the decision rule (exact on k/n) and the costs from translate.py's logs.
With --relative also heel against stukken per bin as differences, and the outcome of a proposed relative rule
(RELATIVE), marked as no part of the decision rule; with --before an earlier report's .json, its heel per bin beside
this one's (part A's heel, before the fork's fix of the decode cache, beside the repeat of 14 Nov).

smoke: a check of a translate.py run of whole stretches (--now, its shards) against an earlier run of the same
stretches (--before): the number of stretches, how many ended with a stop token, and the median of their output tokens
over the earlier ones; exit 1 when one is outside the given bounds.

setup: the setup lines of a report (key, value) as its --setup json.

merge: the measured rows and units of collects over different bins into one set (a row or unit that two inputs both
have must be the same there: the condition stukken of the same pieces).

english: the English of the condition stukken of part A as the documents of part B (its paragraphs, TranslateGemma's
token count), for the stretches whose pieces all ended.
"""

import argparse
import collections
import json
import math
import random
import re
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for _path in (HERE.parent / "flan-nl", HERE):
    sys.path.insert(0, str(_path))

BINS = (512, 1024, 2048, 4096, 8192, 16384)
LOW = 0.75                      # a stretch of bin L has more than LOW x L tokens
PER_SOURCE = 8                  # stretches per source and bin (5 sources: 40 per bin)
MAX_PARAGRAPH_TOKENS = 480      # every paragraph fits one piece
PIECE_TOKENS = 512              # the condition stukken: pieces of whole paragraphs up to this many tokens
MIN_PROSE_SHARE = 0.6           # of a stretch's tokens in prose paragraphs (select_passages.is_prose)
STARTS_PER_DOC = 12             # starting paragraphs tried per document and bin
RESERVE = 8                     # stretches kept per source and bin beyond its quota, for another source's shortfall
SEED = "bt-length/0"
# yhavinga/dutch-handelingen-modern: the selection's revision abdba14 is gone from the Hub; its history was squashed
# into ef8e7e5 (nl-backtranslation README: the published passages were checked against it)
HANDELINGEN_REV = "ef8e7e53159cb631f6c24c40b8b8288c68a487d9"
PARAGRAPH_SPLIT = re.compile(r"\n[ \t]*\n\s*")
# R2's length bound (bt_natural.R2_BOUNDS): characters Dutch / English within 4/3 of the source's median
R2_BOUNDS = {
    "dpc": {"median": 0.9836, "lo": 0.7377, "hi": 1.3114},
    "fineweb": {"median": 1.003, "lo": 0.7522, "hi": 1.3373},
    "handelingen": {"median": 1.0061, "lo": 0.7546, "hi": 1.3414},
    "raad": {"median": 0.993, "lo": 0.7448, "hi": 1.3239},
    "rechtspraak": {"median": 1.0287, "lo": 0.7715, "hi": 1.3716},
}
SOURCE_NAMES = {"handelingen": "Handelingen", "raad": "raad", "dpc": "dpc", "rechtspraak": "rechtspraak",
                "fineweb": "FineWeb-2"}
SOURCE_ORDER = ("handelingen", "raad", "dpc", "rechtspraak", "fineweb")
# The alignment's units (source paragraphs, translation paragraphs) and their fixed costs. The cost of a unit with
# text on both sides adds (d / sigma)^2 / 2, d the log of its length ratio over the expected one, sigma wider for
# short paragraphs (headings, speaker labels); a paragraph left out or added costs SKIP whatever its length, so that a
# missing paragraph is found as one (a length-only cost would rather merge it with its neighbour).
UNITS = {(1, 1): 0.0, (2, 1): 2.0, (1, 2): 2.0, (2, 2): 3.0, (3, 1): 3.5, (1, 3): 3.5}
SKIP = 4.5
SIGMA = 0.2
SIGMA_SHORT = 2.0               # sigma = SIGMA + SIGMA_SHORT / sqrt(characters of the source side)
RULE = {"ended": 0.98, "paragraphs": 0.95, "omission_pp": 1.0, "qe": 0.2}
# A proposed amendment (14 Nov, after part A and the fork fix check; not the rule of bt-length-trial.md 1): heel against
# stukken in the same bin only, as the rule of the condition ctx (13 Nov): at most this many documents fewer done, the
# omissions and the QE within the margins of RULE; the paragraph count reported, not in it
RELATIVE = {"ended_docs": 1, "omission_pp": 1.0, "qe": 0.2}
# Formatting that the translation adds: these categories of structure.py more often in the translation than in the
# source (paragraph and line breaks are the paragraph count's business)
FORMAT_CATEGORIES = ("heading", "bullet", "number", "bold", "italic", "code_block", "inline_code", "table_row",
                     "table_rule")


def write_parquet(rows, path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")


def read_rows(path, columns=None):
    import pyarrow.parquet as pq

    return pq.read_table(path, columns=columns).to_pylist()


def paragraphs_of_text(text):
    """The paragraphs of a text: blocks between empty lines, stripped, none empty"""
    return [p.strip() for p in PARAGRAPH_SPLIT.split(text.strip()) if p.strip()]


def join(paragraphs):
    return "\n\n".join(paragraphs)


def bin_of(tokens, bins=BINS, low=LOW):
    """The bin L with low x L < tokens <= L, or None"""
    for b in bins:
        if low * b < tokens <= b:
            return b
    return None


# ---------------------------------------------------------------------------------------------------- docs

def handelingen_paragraphs(doc):
    """The paragraphs of a debate (an agenda item): per turn a paragraph with its speaker, then the turn's lines.
    Returns (paragraphs, starts): starts marks the speaker paragraphs (a stretch begins at a turn)."""
    import select_passages as sp

    paragraphs, starts = [], []
    for turn in doc["turns"] or []:
        lines = [p for p in (sp.clean(x) for x in (turn.get("text") or "").split("\n")) if p]
        if not lines:
            continue
        speaker = sp.clean(turn.get("speaker") or "")
        party = sp.clean(turn.get("party") or "")
        if speaker:
            label = f"{speaker} ({party}):" if party and party not in speaker else f"{speaker}:"
            paragraphs.append(label)
            starts.append(True)
        for line in lines:
            paragraphs.append(line)
            starts.append(not speaker and line is lines[0])
    return paragraphs, starts


def document_paragraphs(source, doc, wordset):
    """(paragraphs, starts) of a document as the selection reads it; starts: where a stretch may begin"""
    import select_passages as sp

    if source == "handelingen":
        return handelingen_paragraphs(doc)
    if source == "fineweb":
        paragraphs = [p for p in (sp.clean(x) for x in doc["text"].split("\n")) if p]
    else:
        paragraphs = sp.paragraphs_of(source, doc["text"], wordset)
    return paragraphs, [True] * len(paragraphs)


def stretch_fails(text, wordset, langid):
    """The selection's filters of a passage (select_passages Builder.checks) that make sense for a whole stretch:
    language, spelling before 1947, words outside the word list, digits, punctuation, capitals"""
    import select_passages as sp

    lower = [w for w in sp.words_of(text) if w.islower()]
    oov = sum(w not in wordset for w in lower) / max(len(lower), 1)
    chars = [c for c in text if not c.isspace()]
    digits = sum(c.isdigit() for c in chars) / max(len(chars), 1)
    punct = len(sp.PUNCT.findall(text)) / max(len(chars), 1)
    words = sp.words_of(text)
    capitalized = sum(w[0].isupper() for w in words) / max(len(words), 1)
    fails = []
    if langid.classify(text)[0] != "nl":
        fails.append("langid")
    if any(w.lower() in sp.OLD_SPELLING for w in words):
        fails.append("old_spelling")
    if oov > sp.MAX_OOV:
        fails.append("oov")
    if digits > sp.MAX_DIGITS:
        fails.append("digits")
    if punct > sp.MAX_PUNCT:
        fails.append("punct")
    if capitalized > sp.MAX_CAPITALIZED:
        fails.append("capitals")
    return fails


def find_stretch(paragraphs, starts, counts, prose, count_text, target, rng, accept, low=LOW,
                 max_paragraph=MAX_PARAGRAPH_TOKENS, min_prose=MIN_PROSE_SHARE, tries=STARTS_PER_DOC):
    """(first, last + 1, tokens) of a stretch of bin target, or None: from up to `tries` starting paragraphs (in a
    random order), whole paragraphs as long as the joined text stays within target tokens; the stretch must have more
    than low x target tokens, every paragraph at most max_paragraph, a prose share of at least min_prose, and pass
    accept(text). count_text counts the tokens of a joined text (the paragraphs' sum misses the joins)."""
    candidates = [i for i, s in enumerate(starts) if s and counts[i] <= max_paragraph]
    rng.shuffle(candidates)
    for i in candidates[:tries]:
        j, total = i, 0
        while j < len(paragraphs) and counts[j] <= max_paragraph and total + counts[j] + 2 * (j > i) <= target:
            total += counts[j] + 2 * (j > i)
            j += 1
        if total <= low * target:
            continue
        # the exact count of the joined text, shrunk from the end until it fits
        while j > i:
            n = count_text(join(paragraphs[i:j]))
            if n <= target:
                break
            j -= 1
        if j == i or n <= low * target:
            continue
        prose_tokens = sum(c for c, p in zip(counts[i:j], prose[i:j]) if p)
        if prose_tokens < min_prose * sum(counts[i:j]):
            continue
        if not accept(join(paragraphs[i:j])):
            continue
        return i, j, n
    return None


class Quota:
    """Stretches per source and bin: the quota first, then up to RESERVE more for another source's shortfall"""

    def __init__(self, sources, bins, per_source, reserve=RESERVE):
        self.bins, self.per_source, self.reserve = bins, per_source, reserve
        self.taken = {s: {b: [] for b in bins} for s in sources}
        self.extra = {s: {b: [] for b in bins} for s in sources}

    def wanted(self, source):
        """The bins this source still takes a stretch for, largest first"""
        out = [b for b in self.bins if len(self.taken[source][b]) < self.per_source]
        out += [b for b in self.bins if len(self.taken[source][b]) >= self.per_source
                and len(self.extra[source][b]) < self.reserve]
        return sorted(set(out), reverse=True)

    def add(self, source, b, stretch):
        if len(self.taken[source][b]) < self.per_source:
            self.taken[source][b].append(stretch)
        elif len(self.extra[source][b]) < self.reserve:
            self.extra[source][b].append(stretch)
        else:
            raise ValueError(f"{source} bin {b} is full")

    def full(self, source):
        return not self.wanted(source)

    def quota_full(self, source):
        return all(len(self.taken[source][b]) >= self.per_source for b in self.bins)

    def chosen(self):
        """Every bin's stretches: each source's quota, a shortfall made up from the other sources' reserves in turn"""
        out = []
        for b in self.bins:
            rows = [x for s in self.taken for x in self.taken[s][b]]
            short = len(self.taken) * self.per_source - len(rows)
            pools = {s: list(self.extra[s][b]) for s in self.extra}
            while short > 0 and any(pools.values()):
                for s in pools:
                    if short > 0 and pools[s]:
                        x = dict(pools[s].pop(0), filled_for="shortfall")
                        rows.append(x)
                        short -= 1
            out += rows
        return out


def docs(args):
    import langid
    import select_passages as sp
    from huggingface_hub import HfFileSystem
    from tokenizers import Tokenizer

    sp.SOURCES["handelingen"] = dict(sp.SOURCES["handelingen"], repo=(sp.HANDELINGEN[0], HANDELINGEN_REV))
    tokenizer = Tokenizer.from_file(str(Path(args.tokenizer) / "tokenizer.json"))

    def count(texts):
        return [len(e.ids) for e in tokenizer.encode_batch(texts, add_special_tokens=False)] if texts else []

    def count_text(text):
        return count([text])[0]

    with open(args.wordlist, encoding="utf-8") as f:
        wordset = {line.strip().lower() for line in f if line.strip()}
    langid.classify("init")
    bins = tuple(int(b) for b in args.bins.split(","))
    sources = [s for s in SOURCE_ORDER if s in args.sources.split(",")]
    quota = Quota(sources, bins, args.per_source)
    fs = HfFileSystem()
    report = {"bins": list(bins), "low": LOW, "per_source": args.per_source, "sources": {},
              "handelingen_revision": HANDELINGEN_REV}
    for source in sources:
        groups = sp.row_groups(source, args.max_groups, f"{SEED}/groups", fs)
        rng_groups = random.Random(f"{SEED}/{source}/order")
        rng_groups.shuffle(groups)
        counters = collections.Counter()
        for g in groups:
            # the reserves fill only while the quota still reads row groups
            if quota.quota_full(source):
                break
            documents, c = sp.read_docs(source, [g], fs)
            counters.update(c)
            counters["row_groups"] += 1
            for doc in documents:
                wanted = quota.wanted(source)
                if not wanted:
                    break
                doc_id = str(doc["id"] if source in ("handelingen", "fineweb") else doc["doc_id"])
                paragraphs, starts = document_paragraphs(source, doc, wordset)
                # Dutch has ~4-5 characters per Gemma token: a document under 3 per token of the smallest bin it
                # could serve is too short, without counting its tokens
                if not paragraphs or sum(map(len, paragraphs)) < 3 * LOW * min(wanted):
                    counters["docs_too_short"] += 1
                    continue
                counts = count(paragraphs)
                total = sum(counts) + 2 * (len(counts) - 1)
                if total <= LOW * min(wanted):
                    counters["docs_too_short"] += 1
                    continue
                counters["docs_tried"] += 1
                prose = [sp.is_prose(p) for p in paragraphs]
                rng = random.Random(f"{SEED}/{source}/{doc_id}")
                for b in wanted:
                    if total <= LOW * b:
                        continue
                    found = find_stretch(paragraphs, starts, counts, prose, count_text, b, rng,
                                         lambda text: not stretch_fails(text, wordset, langid))
                    if found is None:
                        counters[f"no_stretch_{b}"] += 1
                        continue
                    i, j, n = found
                    quota.add(source, b, {
                        "id": f"{source}:{doc_id}:p{i}-{j}", "source": source, "doc_id": doc_id, "bin": b,
                        "tokens": n, "paragraphs": paragraphs[i:j], "first": i, "last": j - 1,
                        "doc_paragraphs": len(paragraphs), "text": join(paragraphs[i:j]),
                        "url": doc.get("url"), "date": str(doc.get("date")) if doc.get("date") is not None else None,
                        "content_type": list(doc.get("content_type") or []) if source not in (
                            "handelingen", "fineweb") else [],
                        "filled_for": "quota"})
                    counters[f"stretch_{b}"] += 1
                    break
            print(f"DOCS {json.dumps({'source': source, 'group': g[:2], **{str(b): len(quota.taken[source][b]) for b in bins}})}",
                  flush=True)
        report["sources"][source] = {"counters": dict(counters),
                                     "found": {str(b): len(quota.taken[source][b]) + len(quota.extra[source][b])
                                               for b in bins}}
    out = Path(args.out)
    # every stretch found, the quota's and the reserve's: docs-merge chooses over the sources' separate runs
    write_parquet([dict(x, filled_for=kind) for s in quota.taken for kind, lists in (("quota", quota.taken[s]),
                                                                                  ("reserve", quota.extra[s]))
                   for b in bins for x in lists[b]], out / "stretches.parquet")
    (out / "docs.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    choose(quota, bins, report, out)


def choose(quota, bins, report, out):
    rows = quota.chosen()
    for r in rows:
        if bin_of(r["tokens"], bins) != r["bin"]:
            raise SystemExit(f"{r['id']}: {r['tokens']} tokens is not bin {r['bin']}")
    report["chosen"] = {str(b): dict(collections.Counter(r["source"] for r in rows if r["bin"] == b)) for b in bins}
    write_parquet(rows, Path(out) / "docs.parquet")
    (Path(out) / "docs.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(f"CHOSEN {json.dumps(report['chosen'])}", flush=True)


def docs_merge(args):
    """docs.parquet of the sources' separate docs runs (each a directory with stretches.parquet and docs.json): each
    source's quota per bin, a shortfall made up from the other sources' reserves"""
    reports = [json.loads((Path(d) / "docs.json").read_text()) for d in args.inputs]
    bins = tuple(reports[0]["bins"])
    per_source = reports[0]["per_source"]
    if any(tuple(r["bins"]) != bins or r["per_source"] != per_source for r in reports):
        raise SystemExit("the docs runs differ in bins or per_source")
    rows = [r for d in args.inputs for r in read_rows(Path(d) / "stretches.parquet")]
    sources = [s for s in SOURCE_ORDER if any(r["source"] == s for r in rows) or any(s in x["sources"] for x in reports)]
    quota = Quota(sources, bins, per_source)
    for r in rows:
        (quota.taken if r["filled_for"] == "quota" else quota.extra)[r["source"]][r["bin"]].append(r)
    report = {"bins": list(bins), "low": LOW, "per_source": per_source, "handelingen_revision": HANDELINGEN_REV,
              "sources": {s: x for rep in reports for s, x in rep["sources"].items()}}
    choose(quota, bins, report, args.out)


# ---------------------------------------------------------------------------------------------------- segments

def pieces(paragraphs, count_text, limit=PIECE_TOKENS):
    """[[paragraph, ...]]: whole consecutive paragraphs, each piece's joined text at most limit tokens (a paragraph
    longer than limit stands alone)"""
    out, current = [], []
    for p in paragraphs:
        if current and count_text(join(current + [p])) > limit:
            out.append(current)
            current = []
        current.append(p)
    if current:
        out.append(current)
    return out


def balanced(items, workers, weight):
    """items split over workers, heaviest first to the lightest worker: [[item, ...] per worker]"""
    loads = [0] * workers
    out = [[] for _ in range(workers)]
    for item in sorted(items, key=lambda x: -weight(x)):
        k = loads.index(min(loads))
        out[k].append(item)
        loads[k] += weight(item)
    return out


def segments(args):
    import translate as tr
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    def count_text(text):
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])

    documents = read_rows(args.docs)
    if args.bins:
        keep = {int(b) for b in args.bins.split(",")}
        documents = [d for d in documents if d["bin"] in keep]
    heel, stukken = [], []
    for d in documents:
        text = join(d["paragraphs"])
        heel.append({"id": d["id"], "text": text, "tokens": count_text(text), "bin": d["bin"]})
        for k, piece in enumerate(pieces(d["paragraphs"], count_text, args.piece_tokens)):
            t = join(piece)
            stukken.append({"id": f"{d['id']}#c{k:03d}", "text": t, "tokens": count_text(t), "bin": d["bin"]})
    out = Path(args.out)
    summary = {}
    for name, rows, model_name in (("heel", heel, args.doc_model), ("stukken", stukken, args.model)):
        model = tr.with_prompt(tr.MODELS[model_name], args.prompt)
        classes = model["length_classes"]
        codec = tr.ModelPrompts(tokenizer, model, args.source, args.target)
        codec.check([r["text"] for r in rows[:2]])
        used = collections.Counter()
        for r in rows:
            cls = next((c for c, v in classes.items() if r["tokens"] <= v["max_source"]), None)
            if cls is None:
                raise SystemExit(f"{name} {r['id']}: {r['tokens']} tokens, more than {model_name}'s largest class")
            n = len(codec.encode(r["text"]))
            if n > classes[cls]["prefill_lengths"][-1]:
                raise SystemExit(f"{name} {r['id']}: a prompt of {n} tokens, more than class {cls}'s prefill")
            r["class"] = cls
            used[cls] += 1
        for k, part in enumerate(balanced(rows, args.workers, lambda r: r["tokens"])):
            part.sort(key=lambda r: r["id"])
            write_parquet([{"id": r["id"], "text": r["text"], "tokens": r["tokens"]} for r in part],
                          out / name / f"segments-w{k}.parquet")
        order = [c for c in classes if used[c]]
        (out / name / "classes.txt").write_text(" ".join(order) + "\n")
        summary[name] = {"model": model_name, "segments": len(rows), "classes": dict(used),
                         "tokens": sum(r["tokens"] for r in rows)}
    (out / "segments.json").write_text(json.dumps(summary, indent=1))
    print(f"SEGMENTS {json.dumps(summary)}", flush=True)


# ---------------------------------------------------------------------------------------------------- collect

def unit_cost(ls, lt, expected, kind):
    """The cost of aligning ls source characters to lt translation characters in a unit of kind (k source, m
    translation paragraphs); expected: translation characters per source character"""
    if kind not in UNITS:
        raise ValueError(f"no unit {kind}")
    d = math.log((lt + 1) / ((ls + 1) * expected))
    sigma = SIGMA + SIGMA_SHORT / math.sqrt(max(ls, 1))
    return UNITS[kind] + (d / sigma) ** 2 / 2


def align(source, target, expected):
    """A monotone alignment of the paragraphs target to source (lists of strings): [(source indices, target
    indices)], every paragraph in exactly one unit, units in order. expected: translation characters per source
    character. Dynamic programming over UNITS and SKIP (a paragraph left out or added)."""
    ls, lt = [len(p) for p in source], [len(p) for p in target]
    n, m = len(ls), len(lt)
    inf = float("inf")
    cost = [[inf] * (m + 1) for _ in range(n + 1)]
    back = [[None] * (m + 1) for _ in range(n + 1)]
    cost[0][0] = 0.0
    for i in range(n + 1):
        for j in range(m + 1):
            c = cost[i][j]
            if c == inf:
                continue
            if i < n and c + SKIP < cost[i + 1][j]:
                cost[i + 1][j], back[i + 1][j] = c + SKIP, (1, 0)
            if j < m and c + SKIP < cost[i][j + 1]:
                cost[i][j + 1], back[i][j + 1] = c + SKIP, (0, 1)
            for (a, b) in UNITS:
                if i + a <= n and j + b <= m:
                    x = c + unit_cost(sum(ls[i:i + a]), sum(lt[j:j + b]), expected, (a, b))
                    if x < cost[i + a][j + b]:
                        cost[i + a][j + b], back[i + a][j + b] = x, (a, b)
    units, i, j = [], n, m
    while i or j:
        a, b = back[i][j]
        units.append((list(range(i - a, i)), list(range(j - b, j))))
        i, j = i - a, j - b
    return units[::-1]


def nl_en_ratio(src_chars, tgt_chars, source_lang):
    """Characters Dutch / English of a unit"""
    if source_lang == "nl":
        return src_chars / max(tgt_chars, 1)
    return tgt_chars / max(src_chars, 1)


def omissions(units, source_paras, target_paras, bounds, source_lang):
    """Per source paragraph None (kept) or why it counts as omitted: 'none' (aligned to nothing), 'short' (its unit's
    translation is too short for R2's bound), 'long' (too long). Returns (per paragraph, per unit ratio)."""
    status = [None] * len(source_paras)
    ratios = []
    for s, t in units:
        if not s:
            ratios.append(None)
            continue
        if not t:
            for k in s:
                status[k] = "none"
            ratios.append(None)
            continue
        cs = sum(len(source_paras[k]) for k in s)
        ct = sum(len(target_paras[k]) for k in t)
        r = nl_en_ratio(cs, ct, source_lang)
        ratios.append(round(r, 4))
        if not bounds["lo"] <= r <= bounds["hi"]:
            # nl->en: NL/EN above the bound is English too short; en->nl: NL/EN below it is Dutch too short
            too_short = r > bounds["hi"] if source_lang == "nl" else r < bounds["lo"]
            for k in s:
                status[k] = "short" if too_short else "long"
    return status, ratios


def read_translations(root, column):
    """{id: row} of translate.py's shards anywhere under root"""
    out = {}
    if root is None or not Path(root).exists():
        return out
    for path in sorted(Path(root).rglob("shard-*.parquet")):
        for r in read_rows(path):
            if r["id"] in out:
                raise SystemExit(f"{r['id']} is in two shards under {root}")
            out[r["id"]] = {"text": r[column], "ended": r["ended"], "tokens_in": r["tokens_in"],
                            "tokens_out": r["tokens_out"]}
    return out


def added_formatting(source, target):
    """The categories of FORMAT_CATEGORIES that the translation has more often than the source"""
    import structure

    a, b = structure.features(source), structure.features(target)
    return [k for k in FORMAT_CATEGORIES if b[k] > a[k]]


def measure_doc(doc, translation, ended, source_lang, target_lang):
    """The measures of one stretch's translation (a condition) and its alignment units"""
    import bt_natural as bn
    import pairs
    import publish
    import structure

    src = join(doc["paragraphs"])
    src_paras = list(doc["paragraphs"])
    tgt = (translation or "").strip()
    tgt_paras = paragraphs_of_text(tgt) if tgt else []
    bounds = R2_BOUNDS[doc["source"]]
    expected = (1 / bounds["median"]) if source_lang == "nl" else bounds["median"]
    units = align(src_paras, tgt_paras, expected)
    status, ratios = omissions(units, src_paras, tgt_paras, bounds, source_lang)
    loop = bool(tgt) and publish.decode_loop(tgt, src)
    left = (pairs.dutch_left(tgt, src) if source_lang == "nl" else publish.english_left(tgt, src)) if tgt else False
    # cut: the end of the source has no translation, or one too short for R2's bound (a short last paragraph that
    # went missing can also be merged into the unit before it)
    cut = bool(src_paras) and status[-1] in ("none", "short")
    row = {
        "ended": bool(ended), "decode_loop": bool(loop), "cut": cut, "empty": not tgt,
        "done": bool(ended) and not loop and not cut and bool(tgt),
        "source_left": bool(left), "chatter": bn.chatter(tgt, src) if tgt else False,
        "structure": structure.differences(src, tgt) if tgt else [],
        "added_formatting": added_formatting(src, tgt) if tgt else [],
        "paragraphs_source": len(src_paras), "paragraphs_target": len(tgt_paras),
        "paragraphs_kept": len(src_paras) == len(tgt_paras),
        "omitted": sum(s is not None for s in status),
        "omitted_none": status.count("none"), "omitted_short": status.count("short"),
        "omitted_long": status.count("long"),
        "added_paragraphs": sum(len(t) for s, t in units if not s),
        "units": len(units), "units_merged": sum(1 for s, t in units if len(s) > 1 or len(t) > 1),
        "words_source": len(src.split()), "words_target": len(tgt.split()),
        "word_ratio_nl_en": round((len(src.split()) / max(len(tgt.split()), 1)) if source_lang == "nl"
                                  else (len(tgt.split()) / max(len(src.split()), 1)), 4),
        "char_ratio_nl_en": round(nl_en_ratio(len(src), len(tgt), source_lang), 4),
    }
    unit_rows = []
    for k, ((s, t), r) in enumerate(zip(units, ratios)):
        if not s or not t:
            continue
        unit_rows.append({"src": join([src_paras[i] for i in s]), "cand": join([tgt_paras[i] for i in t]),
                          "n_source": len(s), "n_target": len(t), "ratio_nl_en": r,
                          "omitted": status[s[0]] is not None, "unit": k})
    return row, unit_rows, units


def collect(args):
    documents = read_rows(args.docs)
    column = f"text_{args.target}"
    heel = read_translations(args.heel, column)
    stukken = read_translations(args.stukken, column)
    measured, units_out, missing = [], [], collections.Counter()
    for d in documents:
        conditions = {}
        h = heel.get(d["id"])
        if h is not None:
            conditions["heel"] = (h["text"], h["ended"], h["tokens_out"], h["tokens_in"])
        parts = sorted(k for k in stukken if k.startswith(d["id"] + "#c"))
        if parts:
            expected = int(args.expect_pieces.get(d["id"], len(parts))) if args.expect_pieces else len(parts)
            if len(parts) != expected:
                missing["stukken_incomplete"] += 1
            else:
                conditions["stukken"] = (join([(stukken[k]["text"] or "").strip() for k in parts]),
                                         all(stukken[k]["ended"] for k in parts),
                                         sum(stukken[k]["tokens_out"] for k in parts),
                                         sum(stukken[k]["tokens_in"] for k in parts))
        for cond in ("heel", "stukken"):
            if cond not in conditions:
                missing[f"no_{cond}"] += 1
                continue
            text, ended, tokens_out, tokens_in = conditions[cond]
            row, unit_rows, units = measure_doc(d, text, ended, args.source, args.target)
            measured.append({"id": d["id"], "source": d["source"], "bin": d["bin"], "condition": cond,
                             "tokens_source": d["tokens"], "tokens_in": tokens_in, "tokens_out": tokens_out,
                             "translation": text, "pieces": len(parts) if cond == "stukken" else 1,
                             "alignment": json.dumps(units), **row})
            for u in unit_rows:
                units_out.append({"id": f"{d['id']}|{cond}|u{u['unit']:04d}", "doc": d["id"], "source": d["source"],
                                  "bin": d["bin"], "condition": cond, **u})
    out = Path(args.out)
    write_parquet(measured, out / "measured.parquet")
    write_parquet(units_out, out / "units.parquet")
    print(f"COLLECT {json.dumps({'measured': len(measured), 'units': len(units_out), 'missing': dict(missing)})}",
          flush=True)


def pending(args):
    scored = set()
    for path in args.scored or []:
        if Path(path).exists():
            scored |= {r["id"] for r in read_rows(path, ["id", args.column]) if r[args.column] is not None}
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = [r for r in read_rows(args.units, ["id", "src", "cand"]) if r["id"] not in scored]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    schema = pa.schema([("id", pa.string()), ("src", pa.string()), ("cand", pa.string())])
    for k, part in enumerate(balanced(rows, args.workers, lambda r: len(r["src"]) + len(r["cand"]))):
        # an empty share keeps its columns; the run script skips MetricX for it
        pq.write_table(pa.Table.from_pylist(part, schema=schema), out / f"pending-w{k}.parquet")
        print(f"PENDING w{k}: {len(part)} units", flush=True)


def english(args):
    """Part B's documents: the English of the condition stukken (its paragraphs), where every piece ended"""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    documents = {d["id"]: d for d in read_rows(args.docs)}
    rows, skipped = [], collections.Counter()
    for m in read_rows(args.measured):
        if m["condition"] != "stukken":
            continue
        if not m["ended"] or not m["translation"]:
            skipped["not_ended"] += 1
            continue
        d = documents[m["id"]]
        paras = paragraphs_of_text(m["translation"])
        text = join(paras)
        rows.append({"id": d["id"], "source": d["source"], "doc_id": d["doc_id"], "bin": d["bin"],
                     "paragraphs": paras, "text": text,
                     "tokens": len(tokenizer(text, add_special_tokens=False)["input_ids"]),
                     "tokens_nl": d["tokens"], "nl_paragraphs": d["paragraphs"]})
    write_parquet(rows, args.out)
    print(f"ENGLISH {json.dumps({'documents': len(rows), 'skipped': dict(skipped)})}", flush=True)


# ---------------------------------------------------------------------------------------------------- report

def frac(k, n):
    return k / n if n else None


def pct(x, digits=1):
    return "–" if x is None else f"{100 * x:.{digits}f}%".replace(".", ",")


def num(x, digits=2, signed=False):
    if x is None:
        return "–"
    s = f"{x:+.{digits}f}" if signed else f"{x:.{digits}f}"
    return s.replace(".", ",")


def kn(k, n):
    return f"{k}/{n} ({pct(frac(k, n))})"


def qe_per_paragraph(units, scores, column):
    """{(doc, condition): [score per source paragraph with a translation]} from the units and their MetricX scores"""
    out = collections.defaultdict(list)
    for u in units:
        s = scores.get(u["id"])
        if s is None:
            continue
        out[(u["doc"], u["condition"])] += [s] * u["n_source"]
    return out


def bin_stats(rows, qe, bins, conditions=("heel", "stukken")):
    """{bin: {condition: stats}} of the measured rows"""
    stats = {}
    for b in bins:
        stats[b] = {}
        for cond in conditions:
            rs = [r for r in rows if r["bin"] == b and r["condition"] == cond]
            n = len(rs)
            paras = sum(r["paragraphs_source"] for r in rs)
            scores = [s for r in rs for s in qe.get((r["id"], cond), [])]
            scored_docs = sum(1 for r in rs if (r["id"], cond) in qe)
            stats[b][cond] = {
                "docs": n,
                "done": sum(r["done"] for r in rs), "not_ended": sum(not r["ended"] for r in rs),
                "decode_loop": sum(r["decode_loop"] for r in rs), "cut": sum(r["cut"] for r in rs),
                "empty": sum(r["empty"] for r in rs),
                "source_left": sum(r["source_left"] for r in rs), "chatter": sum(r["chatter"] for r in rs),
                "added_formatting": sum(bool(r["added_formatting"]) for r in rs),
                "formatting": dict(collections.Counter(c for r in rs for c in r["added_formatting"])),
                "paragraphs_kept": sum(r["paragraphs_kept"] for r in rs),
                "paragraphs": paras, "omitted": sum(r["omitted"] for r in rs),
                "omitted_none": sum(r["omitted_none"] for r in rs),
                "omitted_short": sum(r["omitted_short"] for r in rs),
                "omitted_long": sum(r["omitted_long"] for r in rs),
                "added_paragraphs": sum(r["added_paragraphs"] for r in rs),
                "qe_mean": statistics.fmean(scores) if scores else None,
                "qe_median": statistics.median(scores) if scores else None,
                "qe_paragraphs": len(scores), "qe_docs": scored_docs,
                "word_ratio_median": statistics.median([r["word_ratio_nl_en"] for r in rs if not r["empty"]])
                if any(not r["empty"] for r in rs) else None,
                "tokens_source_median": statistics.median([r["tokens_source"] for r in rs]) if rs else None,
                "tokens_out_median": statistics.median([r["tokens_out"] for r in rs]) if rs else None,
            }
    return stats


def rule(stats, rule_spec=RULE):
    """Per bin the four parts of the decision rule (exact on k/n) and L: the largest bin that passes all four, and
    every smaller measured bin too is not required (the rule names the largest bin that passes)"""
    verdict = {}
    for b, s in stats.items():
        h, p = s.get("heel"), s.get("stukken")
        if not h or not h["docs"] or not p or not p["docs"]:
            verdict[b] = {"measured": False, "pass": False}
            continue
        i = h["done"] / h["docs"] >= rule_spec["ended"]
        ii = h["paragraphs_kept"] / h["docs"] >= rule_spec["paragraphs"]
        om_h, om_p = h["omitted"] / h["paragraphs"], p["omitted"] / p["paragraphs"]
        iii = (om_h - om_p) * 100 <= rule_spec["omission_pp"] + 1e-12
        qe_known = h["qe_mean"] is not None and p["qe_mean"] is not None
        iv = qe_known and h["qe_mean"] - p["qe_mean"] <= rule_spec["qe"] + 1e-12
        verdict[b] = {"measured": True, "i": i, "ii": ii, "iii": iii, "iv": iv, "qe_known": qe_known,
                      "omission_diff_pp": (om_h - om_p) * 100,
                      "qe_diff": h["qe_mean"] - p["qe_mean"] if qe_known else None,
                      "pass": bool(i and ii and iii and iv)}
    return verdict, largest_passing(verdict)


def largest_passing(verdict):
    """The largest bin that passes, below the smallest bin that could not be measured (bt-length-trial.md 1: a bin
    that could not be measured does not pass, and a larger bin does not count then)"""
    unmeasured = [b for b, v in verdict.items() if not v["measured"]]
    passing = [b for b, v in verdict.items() if v["pass"] and (not unmeasured or b < min(unmeasured))]
    return max(passing) if passing else None


def relative(stats, spec=RELATIVE):
    """Per bin heel against stukken as differences (documents done and with their paragraph count, omissions in
    percentage points, mean QE) and whether the proposed relative rule passes; L of that rule as largest_passing"""
    out = {}
    for b, s in stats.items():
        h, p = s.get("heel"), s.get("stukken")
        if not h or not h["docs"] or not p or not p["docs"]:
            out[b] = {"measured": False, "pass": False}
            continue
        om = (h["omitted"] / h["paragraphs"] - p["omitted"] / p["paragraphs"]) * 100
        qe = h["qe_mean"] - p["qe_mean"] if h["qe_mean"] is not None and p["qe_mean"] is not None else None
        done = h["done"] - p["done"]
        parts = {"done": done >= -spec["ended_docs"], "omissions": om <= spec["omission_pp"] + 1e-12,
                 "qe": qe is not None and qe <= spec["qe"] + 1e-12}
        out[b] = {"measured": True, "done_diff": done, "paragraphs_kept_diff": h["paragraphs_kept"] - p["paragraphs_kept"],
                  "omission_diff_pp": om, "qe_diff": qe, **parts, "pass": all(parts.values())}
    return out, largest_passing(out)


def throughput(logs_dir):
    """Per log file and length class: the engine's load time, segments, tokens, decode seconds (translate.py's SHARD
    and DONE lines), the class's shapes and the peak HBM per chip (bt_length_translate.py's MEM line)"""
    out = []
    for path in sorted(Path(logs_dir).glob("*.log")) if logs_dir else []:
        current = None
        for line in path.read_text(errors="replace").splitlines():
            m = re.match(r"^(\w+): ([\d,]+) segments in \d+ shards, \d+ done before; shapes (\{.*\})$", line)
            if m:
                current = {"log": path.name, "class": m.group(1), "shapes": json.loads(m.group(3)),
                           "segments": 0, "tokens_in": 0, "tokens_out": 0, "seconds": 0.0, "load_s": None,
                           "peak_gib": None, "chips": None, "oom": False}
                out.append(current)
                continue
            if current is None:
                continue
            m = re.match(r"^engine ready in (\d+) s$", line)
            if m:
                current["load_s"] = int(m.group(1))
            elif line.startswith("SHARD "):
                s = json.loads(line[len("SHARD "):])
                current["segments"] += s["segments"]
                current["tokens_in"] += s["tokens_in"]
                current["tokens_out"] += s["tokens_out"]
                current["seconds"] += s["seconds"]
            elif line.startswith("MEM "):
                s = json.loads(line[len("MEM "):])
                current["peak_gib"] = s.get("peak_gib_max")
                current["chips"] = len(s["peak_gib"]) if s.get("peak_gib") else None
            elif "RESOURCE_EXHAUSTED" in line or "Ran out of memory" in line:
                current["oom"] = True
    return out


def report(args):
    measured = read_rows(args.measured)
    units = read_rows(args.units, ["id", "doc", "condition", "n_source"])
    scores = {}
    for path in args.qe or []:
        for r in read_rows(path):
            if r.get(args.column) is not None:
                scores[r["id"]] = r[args.column]
    qe = qe_per_paragraph(units, scores, args.column)
    bins = sorted({r["bin"] for r in measured})
    stats = bin_stats(measured, qe, bins)
    verdict, length = rule(stats)
    speed = throughput(args.logs)
    setup = json.loads(Path(args.setup).read_text()) if args.setup else {}
    r = {"title": args.title, "direction": f"{args.source}->{args.target}", "stats": stats, "verdict": verdict,
         "L": length, "throughput": speed, "setup": setup, "qe_column": args.column,
         "per_source": {b: {s: bin_stats([x for x in measured if x["source"] == s], qe, [b])[b]
                            for s in SOURCE_ORDER} for b in bins}}
    if getattr(args, "relative", False):
        r["relative"], r["L_relative"] = relative(stats)
        r["relative_spec"] = RELATIVE
    if getattr(args, "before", None):
        earlier = json.loads(Path(args.before).read_text())
        r["before"] = {"title": earlier.get("title"), "setup": earlier.get("setup"),
                       "heel": {int(b): s.get("heel") for b, s in earlier["stats"].items()}}
    md = report_md(r, measured, args)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(md)
    Path(args.out).with_suffix(".json").write_text(json.dumps(r, indent=1, ensure_ascii=False, default=str))
    print(md)


def report_md(r, measured, args):
    stats, verdict = r["stats"], r["verdict"]
    bins = sorted(stats)
    lines = [f"## {r['title']}", ""]
    if r["setup"]:
        lines += [f"- {k}: {v}" for k, v in r["setup"].items()] + [""]
    head = "| maat | " + " | ".join(f"{b} heel | {b} stukken" for b in bins) + " |"
    lines += [head, "|---|" + "---:|" * (2 * len(bins))]

    def row(label, f):
        cells = []
        for b in bins:
            for cond in ("heel", "stukken"):
                s = stats[b].get(cond)
                cells.append(f(s) if s and s["docs"] else "–")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    row("documenten", lambda s: str(s["docs"]))
    row("brontokens, mediaan", lambda s: num(s["tokens_source_median"], 0))
    row("**af (zonder lus of afkapping)**", lambda s: kn(s["done"], s["docs"]))
    row("niet af (decodeerlimiet)", lambda s: str(s["not_ended"]))
    row("lus", lambda s: str(s["decode_loop"]))
    row("afgekapt (laatste alinea weg)", lambda s: str(s["cut"]))
    row("brontaal over", lambda s: str(s["source_left"]))
    row("inleiding of noot (`chatter`)", lambda s: str(s["chatter"]))
    row("toegevoegde opmaak (documenten)", lambda s: str(s["added_formatting"]))
    row("**alinea-aantal behouden**", lambda s: kn(s["paragraphs_kept"], s["docs"]))
    row("bronalinea's", lambda s: str(s["paragraphs"]))
    row("**weglatingen**", lambda s: kn(s["omitted"], s["paragraphs"]))
    row("waarvan op niets uitgelijnd", lambda s: str(s["omitted_none"]))
    row("waarvan vertaling te kort (R2)", lambda s: str(s["omitted_short"]))
    row("waarvan vertaling te lang (R2)", lambda s: str(s["omitted_long"]))
    row("toegevoegde alinea's", lambda s: str(s["added_paragraphs"]))
    row(f"**MetricX-24-QE {r['direction']} per alinea, gem.**", lambda s: num(s["qe_mean"]))
    row("idem, mediaan", lambda s: num(s["qe_median"]))
    row("alinea's met score / documenten met score", lambda s: f"{s['qe_paragraphs']} / {s['qe_docs']}")
    row("NL/EN-woorden per document, mediaan", lambda s: num(s["word_ratio_median"], 3))
    lines.append("")
    lines += ["### Beslisregel per bak (conditie heel tegen in stukken)", "",
              "| bak | (i) af ≥ 98% | (ii) alinea's ≥ 95% | (iii) weglatingen ≤ +1 pp | (iv) QE ≤ +0,2 | haalt de regel |",
              "|---:|---|---|---|---|---|"]
    for b in bins:
        v, h = verdict[b], stats[b].get("heel")
        if not v["measured"]:
            lines.append(f"| {b} | niet gemeten | | | | nee |")
            continue
        lines.append(
            f"| {b} | {'ja' if v['i'] else 'nee'} ({kn(h['done'], h['docs'])}) "
            f"| {'ja' if v['ii'] else 'nee'} ({kn(h['paragraphs_kept'], h['docs'])}) "
            f"| {'ja' if v['iii'] else 'nee'} ({num(v['omission_diff_pp'], 2, True)} pp) "
            f"| {'ja' if v['iv'] else 'nee'} ({num(v['qe_diff'], 3, True) if v['qe_known'] else 'geen QE'}) "
            f"| **{'ja' if v['pass'] else 'nee'}** |")
    lines += ["", f"**L = {r['L'] if r['L'] else 'geen bak haalt de regel'}**", ""]
    if "relative" in r:
        lines += relative_md(r, stats, bins)
    if "before" in r:
        lines += before_md(r, stats, bins)
    lines += ["### Per bron (heel / in stukken)", "",
              "| bak | bron | n | af | alinea's behouden | weglatingen | QE gem. |", "|---:|---|---:|---|---|---|---|"]
    for b in bins:
        for s in SOURCE_ORDER:
            x = r["per_source"][b][s]
            h, p = x.get("heel"), x.get("stukken")
            if not h or not h["docs"]:
                continue
            p = p if p and p["docs"] else None
            lines.append(
                f"| {b} | {SOURCE_NAMES[s]} | {h['docs']} | {h['done']} / {p['done'] if p else '–'} "
                f"| {h['paragraphs_kept']} / {p['paragraphs_kept'] if p else '–'} "
                f"| {pct(frac(h['omitted'], h['paragraphs']))} / {pct(frac(p['omitted'], p['paragraphs'])) if p else '–'} "
                f"| {num(h['qe_mean'])} / {num(p['qe_mean']) if p else '–'} |")
    lines += ["", "### Kosten (translate.py per lengteklasse en proces)", "",
              "| log | klasse | chips | slots | prefill / decodeerlimiet | segmenten | laden (s) | decodeertijd (s) | s per segment | uitvoertokens/s | piek HBM per chip (GiB) | geheugen op |",
              "|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---|"]
    for t in r["throughput"]:
        sh = t["shapes"]
        lines.append(
            f"| {t['log']} | {t['class']} | {t.get('chips') or '–'} | {sh.get('slots')} | {sh.get('prefill_lengths', ['?'])[-1]} / "
            f"{sh.get('max_decode')} | {t['segments']} | {t['load_s'] if t['load_s'] is not None else '–'} "
            f"| {num(t['seconds'], 0)} | {num(t['seconds'] / t['segments'], 1) if t['segments'] else '–'} "
            f"| {num(t['tokens_out'] / t['seconds'], 0) if t['seconds'] else '–'} "
            f"| {num(t['peak_gib'], 2) if t['peak_gib'] is not None else '–'} | {'ja' if t['oom'] else ''} |")
    failing = [x for x in measured if x["condition"] == "heel" and (not x["done"] or not x["paragraphs_kept"])]
    if failing:
        lines += ["", "### Documenten in de conditie heel die niet af zijn of het alinea-aantal niet houden", "",
                  "| bak | document | brontokens | uitvoertokens | af | niet af / lus / afgekapt | alinea's bron → vertaling | weglatingen |",
                  "|---:|---|---:|---:|---|---|---|---:|"]
        for x in sorted(failing, key=lambda x: (x["bin"], x["id"]))[:80]:
            lines.append(f"| {x['bin']} | `{x['id']}` | {x['tokens_source']} | {x['tokens_out']} | "
                         f"{'ja' if x['done'] else 'nee'} | {int(not x['ended'])} / {int(x['decode_loop'])} / "
                         f"{int(x['cut'])} | {x['paragraphs_source']} → {x['paragraphs_target']} | {x['omitted']} |")
    return "\n".join(lines) + "\n"


def relative_md(r, stats, bins):
    """The section of --relative: heel against stukken per bin, and the proposed relative rule"""
    spec = r["relative_spec"]
    lines = ["### Heel tegen in stukken per bak, relatief (geen deel van de beslisregel)", "",
             "Verschil heel − in stukken op dezelfde documenten van de bak. Alinea-aantal: documenten met behouden "
             "alinea-aantal.", "",
             "| bak | af: heel / stukken (verschil) | alinea-aantal: heel / stukken (verschil) | weglatingen: heel / stukken (verschil) | QE gem.: heel / stukken (verschil) |",
             "|---:|---|---|---|---|"]
    for b in bins:
        v, h, p = r["relative"][b], stats[b].get("heel"), stats[b].get("stukken")
        if not v["measured"]:
            lines.append(f"| {b} | niet gemeten | | | |")
            continue
        lines.append(
            f"| {b} | {h['done']} / {p['done']} ({v['done_diff']:+d}) "
            f"| {h['paragraphs_kept']} / {p['paragraphs_kept']} ({v['paragraphs_kept_diff']:+d}) "
            f"| {pct(frac(h['omitted'], h['paragraphs']))} / {pct(frac(p['omitted'], p['paragraphs']))} "
            f"({num(v['omission_diff_pp'], 2, True)} pp) "
            f"| {num(h['qe_mean'])} / {num(p['qe_mean'])} ({num(v['qe_diff'], 3, True)}) |")
    lines += ["", f"**Voorstel voor een amendement (beslist Yeb; niet de regel van §1).** Een bak haalt de relatieve "
              f"regel als heel, op dezelfde documenten, (a) hoogstens {spec['ended_docs']} document minder af heeft "
              f"dan in stukken, (b) hoogstens {num(spec['omission_pp'], 0)} procentpunt meer weglatingen en (c) een "
              f"gemiddelde QE per alinea hoogstens {num(spec['qe'], 1)} hoger; het alinea-aantal wordt gerapporteerd "
              f"en beslist niet mee; een bak die niet gemeten kon worden haalt de regel niet en een grotere bak telt "
              f"dan niet.", "",
              "| bak | (a) af ≥ stukken − 1 | (b) weglatingen ≤ +1 pp | (c) QE ≤ +0,2 | haalt de relatieve regel |",
              "|---:|---|---|---|---|"]
    for b in bins:
        v = r["relative"][b]
        if not v["measured"]:
            lines.append(f"| {b} | niet gemeten | | | nee |")
            continue
        lines.append(f"| {b} | {'ja' if v['done'] else 'nee'} ({v['done_diff']:+d}) "
                     f"| {'ja' if v['omissions'] else 'nee'} ({num(v['omission_diff_pp'], 2, True)} pp) "
                     f"| {'ja' if v['qe'] else 'nee'} ({num(v['qe_diff'], 3, True)}) "
                     f"| **{'ja' if v['pass'] else 'nee'}** |")
    lines += ["", f"**L volgens het voorstel = {r['L_relative'] if r['L_relative'] else 'geen bak'}** "
              "(geen deel van de beslisregel)", ""]
    return lines


def before_md(r, stats, bins):
    """The section of --before: the earlier report's heel per bin beside this one's"""
    lines = [f"### Heel nu tegen heel in het eerdere rapport ({r['before']['title']})", "",
             "| bak | af: nu / eerder | lus: nu / eerder | alinea-aantal: nu / eerder | weglatingen: nu / eerder | QE gem.: nu / eerder |",
             "|---:|---|---|---|---|---|"]
    for b in bins:
        h, e = stats[b].get("heel"), r["before"]["heel"].get(b)
        h = h if h and h["docs"] else None
        e = e if e and e.get("docs") else None
        if not h and not e:
            continue

        def cell(f, x):
            return f(x) if x else "niet gemeten"
        lines.append(
            f"| {b} | {cell(lambda x: kn(x['done'], x['docs']), h)} / {cell(lambda x: kn(x['done'], x['docs']), e)} "
            f"| {cell(lambda x: str(x['decode_loop']), h)} / {cell(lambda x: str(x['decode_loop']), e)} "
            f"| {cell(lambda x: kn(x['paragraphs_kept'], x['docs']), h)} / {cell(lambda x: kn(x['paragraphs_kept'], x['docs']), e)} "
            f"| {cell(lambda x: kn(x['omitted'], x['paragraphs']), h)} / {cell(lambda x: kn(x['omitted'], x['paragraphs']), e)} "
            f"| {cell(lambda x: num(x['qe_mean']), h)} / {cell(lambda x: num(x['qe_mean']), e)} |")
    return lines + [""]


def merge(args):
    """One measured.parquet and units.parquet from several collects; a row or unit in two inputs must be equal"""
    out = Path(args.out)
    for name, paths, key, same in (("measured", args.measured, lambda r: (r["id"], r["condition"]),
                                    ("translation", "ended", "done", "omitted")),
                                   ("units", args.units, lambda r: r["id"], ("src", "cand", "n_source"))):
        rows, seen = [], {}
        for path in paths:
            for r in read_rows(path):
                k = key(r)
                if k in seen:
                    if any(seen[k][c] != r[c] for c in same):
                        raise SystemExit(f"{name}: {k} differs between the inputs ({path})")
                    continue
                seen[k] = r
                rows.append(r)
        write_parquet(rows, out / f"{name}.parquet")
        print(f"MERGE {json.dumps({'table': name, 'rows': len(rows), 'inputs': len(paths)})}", flush=True)


def complete(args):
    """Every class of --classes whole under --root/<class>: all its segments of --segments translated (translate.py's
    choice of a class: more tokens than the class below, at most its own) or the class marked out of memory (oom.txt).
    Else SystemExit: on 14 Nov a run's report would have taken a half or missing class as a bin with fewer or no
    documents, and the run would have ended as if it were done"""
    import translate as tr

    classes = tr.with_prompt(tr.MODELS[args.model], args.prompt)["length_classes"]
    order = list(classes)
    rows = read_rows(args.segments, ["id", "tokens"])
    bad = []
    for cls in args.classes:
        if cls not in classes:
            raise SystemExit(f"{cls} is not a class of {args.model}: {', '.join(order)}")
        k = order.index(cls)
        low = classes[order[k - 1]]["max_source"] if k else 0
        want = {r["id"] for r in rows if low < r["tokens"] <= classes[cls]["max_source"]}
        root = Path(args.root) / cls
        got = set()
        for path in sorted(root.glob("shard-*.parquet")) if root.exists() else []:
            got |= {r["id"] for r in read_rows(path, ["id"])}
        oom = (root / "oom.txt").exists()
        print(f"COMPLETE {json.dumps({'class': cls, 'segments': len(want), 'translated': len(want & got), 'oom': oom})}",
              flush=True)
        if not want:
            bad.append(f"{cls}: no segments in {args.segments}")
        elif not oom and not want <= got:
            bad.append(f"{cls}: {len(want - got)} of {len(want)} segments not translated in {root}")
    if bad:
        raise SystemExit("not complete: " + "; ".join(bad))


def smoke(args):
    def rows(root):
        out = {}
        for path in sorted(Path(root).rglob("shard-*.parquet")):
            for r in read_rows(path, ["id", "ended", "tokens_out"]):
                out[r["id"]] = r
        return out

    now, before = rows(args.now), rows(args.before)
    ended = sum(bool(r["ended"]) for r in now.values())
    ratios = [now[i]["tokens_out"] / before[i]["tokens_out"] for i in now if before.get(i, {}).get("tokens_out")]
    median = statistics.median(ratios) if ratios else None
    low, high = (float(x) for x in args.ratio.split(","))
    ok = len(now) == args.docs and ended >= args.ended and median is not None and low <= median <= high
    print(f"SMOKE {json.dumps({'documents': len(now), 'ended': ended, 'compared': len(ratios), 'median_ratio': median, 'pass': ok})}",
          flush=True)
    if not ok:
        raise SystemExit(f"smoke failed: {len(now)} documents (need {args.docs}), {ended} ended (need {args.ended}), "
                         f"median output ratio {median} (need {low}-{high})")


def setup_json(args):
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(dict(args.item), ensure_ascii=False))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("docs")
    a.add_argument("--tokenizer", required=True)
    a.add_argument("--wordlist", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--bins", default=",".join(str(b) for b in BINS))
    a.add_argument("--sources", default=",".join(SOURCE_ORDER))
    a.add_argument("--per-source", type=int, default=PER_SOURCE)
    a.add_argument("--max-groups", type=int, default=40, help="Row groups read per source at most")
    a = sub.add_parser("docs-merge")
    a.add_argument("--inputs", nargs="+", required=True)
    a.add_argument("--out", required=True)
    a = sub.add_parser("segments")
    a.add_argument("--docs", required=True)
    a.add_argument("--tokenizer", required=True)
    a.add_argument("--model", required=True, help="translate_models.yml model of the pieces")
    a.add_argument("--doc-model", required=True, help="translate_models.yml model of the whole stretches")
    a.add_argument("--prompt", default=None)
    a.add_argument("--source", required=True)
    a.add_argument("--target", required=True)
    a.add_argument("--workers", type=int, default=1)
    a.add_argument("--bins", default=None, help="Only these bins (default: all of --docs)")
    a.add_argument("--piece-tokens", type=int, default=PIECE_TOKENS)
    a.add_argument("--out", required=True)
    a = sub.add_parser("collect")
    a.add_argument("--docs", required=True)
    a.add_argument("--heel", default=None)
    a.add_argument("--stukken", default=None)
    a.add_argument("--pieces", default=None, help="segments.parquet files of stukken (the number of pieces per stretch)",
                   nargs="*")
    a.add_argument("--source", required=True)
    a.add_argument("--target", required=True)
    a.add_argument("--out", required=True)
    a = sub.add_parser("pending")
    a.add_argument("--units", required=True)
    a.add_argument("--scored", nargs="*", default=[])
    a.add_argument("--column", default="qe_src_cand")
    a.add_argument("--workers", type=int, default=1)
    a.add_argument("--out", required=True)
    a = sub.add_parser("report")
    a.add_argument("--measured", required=True)
    a.add_argument("--units", required=True)
    a.add_argument("--qe", nargs="*", default=[])
    a.add_argument("--column", default="qe_src_cand")
    a.add_argument("--logs", default=None)
    a.add_argument("--setup", default=None)
    a.add_argument("--source", required=True)
    a.add_argument("--target", required=True)
    a.add_argument("--title", required=True)
    a.add_argument("--relative", action="store_true", help="heel against stukken per bin and the proposed relative rule")
    a.add_argument("--before", default=None, help="an earlier report's .json: its heel per bin beside this one's")
    a.add_argument("--out", required=True)
    a = sub.add_parser("smoke")
    a.add_argument("--now", required=True)
    a.add_argument("--before", required=True)
    a.add_argument("--docs", type=int, required=True)
    a.add_argument("--ended", type=int, required=True)
    a.add_argument("--ratio", required=True, help="low,high bounds of the median output ratio")
    a = sub.add_parser("complete")
    a.add_argument("--segments", required=True)
    a.add_argument("--model", required=True, help="translate_models.yml model of the segments")
    a.add_argument("--prompt", default=None)
    a.add_argument("--classes", nargs="+", required=True)
    a.add_argument("--root", required=True, help="the classes' shards under <root>/<class>/")
    a = sub.add_parser("setup")
    a.add_argument("--out", required=True)
    a.add_argument("--item", nargs=2, action="append", default=[], metavar=("KEY", "VALUE"))
    a = sub.add_parser("merge")
    a.add_argument("--measured", nargs="+", required=True)
    a.add_argument("--units", nargs="+", required=True)
    a.add_argument("--out", required=True)
    a = sub.add_parser("english")
    a.add_argument("--docs", required=True)
    a.add_argument("--measured", required=True)
    a.add_argument("--tokenizer", required=True)
    a.add_argument("--out", required=True)
    args = p.parse_args(argv)
    if args.cmd == "collect":
        args.expect_pieces = None
        if args.pieces:
            counts = collections.Counter()
            for path in args.pieces:
                for r in read_rows(path, ["id"]):
                    counts[r["id"].split("#c")[0]] += 1
            args.expect_pieces = counts
    {"docs": docs, "docs-merge": docs_merge, "segments": segments, "collect": collect, "pending": pending, "report": report,
     "english": english, "merge": merge, "smoke": smoke, "complete": complete, "setup": setup_json}[args.cmd](args)


if __name__ == "__main__":
    main()
