#!/usr/bin/env python3
"""Gemma 4 31B-it against TranslateGemma 27B on the translator's pilot pairs (hammond mbert/designs/en-nl-translator.md,
§3 and §9), with translate.py's model gemma4-31b.

    python gemma4_pilot.py segments --pairs pairs.parquet --tokenizer <Gemma 4 tokenizer> --out DIR
    python gemma4_pilot.py context-data --pairs pairs.parquet --handelingen <dir of its parquet files> \\
        --nemotron <chunk of chat.jsonl> --tokenizer <Gemma 4 tokenizer> --out DIR
    python gemma4_pilot.py context-run --data DIR --tokenizer ... --ckpt ... --out DIR     (TPU, 4 chips)
    python gemma4_pilot.py merge --pairs pairs.parquet --en-nl <shards> --nl-en <shards> --out compare.parquet
    python gemma4_pilot.py report --compare compare.parquet --qe qe.parquet --context context.parquet \\
        --pids pairs30-pids.txt --logs <dir> --setup setup.json --out gemma4-vs-tg27b.md

segments: the inputs of translate.py: en->nl from the pilot's synthetic English (TranslateGemma 27B's nl->en, the
English that TranslateGemma's own en->nl had), nl->en from the Dutch originals; the token counts by Gemma 4's
tokenizer choose the length class.

context-data: the context trial. 20 passages of the Handelingen of the pilot, each with the 1-2 speaking turns before
its own in the same debate (yhavinga/dutch-handelingen-modern @ abdba14, the source of the pilot), and 5 short
conversations of the lmarena part of nvidia/Nemotron-SFT-Instruction-Following-Chat-v3 (user and assistant content,
no reasoning_content). A chain is the turns of one conversation in order; its last turn is the passage (Handelingen).

context-run: on one process of 4 chips, one engine for all rounds. The Handelingen turns before a passage are
translated nl->en first (plain prompt), as the pilot's back-translation was. Then every chain en->nl, turn by turn: a
turn with the earlier turns of its chain (their English and the Dutch Gemma 4 made of them) as context (chat-v1's
context prompt, the speaker as label), and every turn on its own (plain prompt). The passage's English is the
pilot's synthetic English.

merge: per pair the pilot's columns (TranslateGemma: en = nl->en, back = en->nl) with Gemma 4's (en_g4, nl_g4) and
the measures of both: ended, decode_loop, english_left / dutch_left, structure, chrF.

report: gemma4-vs-tg27b.md for Yeb.
"""

import argparse
import hashlib
import json
import re
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
for _path in (HERE.parent / "flan-nl", HERE.parents[1] / "scripts", HERE):
    sys.path.insert(0, str(_path))

N_HANDELINGEN = 20
N_NEMOTRON = 5
# Context of a Handelingen passage: the 1-2 turns before it, at most this many Dutch tokens together (with their
# translation the context is about twice that), each at most TURN_MAX
CONTEXT_MAX = 600
TURN_MAX = 384
# A short Nemotron conversation: at least two user turns, every turn at most NEMO_TURN_MAX tokens, at most CHAT_MAX
# together. Assistant turns have a median of ~650 tokens, so the Handelingen limit of TURN_MAX (384) left none of 387
# lmarena conversations; 800 keeps a Dutch turn under the engine's max_decode of 1024, and 1600 keeps the prompt (earlier
# turns twice, source and translation) under its largest prefill of 4096.
NEMO_TURN_MAX = 800
CHAT_MAX = 1600
# The engine of the trial: prompts up to 4096 tokens (earlier turns twice, source and translation), turns up to 1024
CONTEXT_SHAPES = {"prefill_lengths": [512, 1024, 2048, 3072, 4096], "max_decode": 1024, "slots": 8}
NEMOTRON_LABELS = {"system": "Systeem", "user": "Gebruiker", "assistant": "Assistent"}


def stable(key):
    return hashlib.sha1(key.encode()).hexdigest()


def write_parquet(rows, path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")


def read_rows(path, columns=None):
    import pyarrow.parquet as pq

    return pq.read_table(path, columns=columns).to_pylist()


class Counter:
    """Gemma 4 token counts of texts."""

    def __init__(self, tokenizer_dir):
        from tokenizers import Tokenizer

        self.tok = Tokenizer.from_file(str(Path(tokenizer_dir) / "tokenizer.json"))

    def __call__(self, text):
        return len(self.tok.encode(text, add_special_tokens=False).ids)


def segments(args):
    pairs = read_rows(args.pairs, ["id", "nl", "en"])
    count = Counter(args.tokenizer)
    out = Path(args.out)
    for name, column in (("en", "en"), ("nl", "nl")):
        rows = [{"id": p["id"], "text": p[column], "tokens": count(p[column])} for p in pairs if p[column]]
        rows.sort(key=lambda r: r["id"])
        write_parquet(rows, out / f"segments-{name}.parquet")
        print(f"SEGMENTS {json.dumps({'source': name, 'rows': len(rows), 'max_tokens': max(r['tokens'] for r in rows)})}")


# ------------------------------------------------------------------ context data


def handelingen_rows(directory, doc_ids):
    """{id: row} of the debates doc_ids, from the dataset's parquet files."""
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    found = {}
    for path in sorted(Path(directory).glob("*.parquet")):
        table = pq.read_table(path, columns=["id", "chamber", "date", "title", "turns"])
        table = table.filter(pc.is_in(table.column("id"), value_set=__import__("pyarrow").array(sorted(doc_ids))))
        for row in table.to_pylist():
            found[str(row["id"])] = row
    missing = set(doc_ids) - set(found)
    if missing:
        raise SystemExit(f"debates not found in {directory}: {sorted(missing)[:3]}")
    return found


def turn_text(turn):
    """A speaking turn's text as select_passages makes its paragraphs: lines with their whitespace evened, empty
    ones out."""
    import select_passages as sp

    return "\n".join(p for p in (sp.clean(x) for x in (turn.get("text") or "").split("\n")) if p)


def speaker_label(turn):
    speaker = (turn.get("speaker") or "").strip()
    party = (turn.get("party") or "").strip()
    return f"{speaker} ({party})" if speaker and party else speaker or None


def handelingen_chains(pairs, debates, count):
    """Up to N_HANDELINGEN chains: the 1-2 turns before a pilot passage (2 when they fit) and the passage."""
    candidates = []
    for p in pairs:
        if p["source"] != "handelingen":
            continue
        _, doc_id, block, _ = p["pid"].split(":")
        t = int(block[1:])
        turns = debates[doc_id]["turns"] or []
        before = []
        for j in (t - 1, t - 2):
            if j < 0:
                break
            text = turn_text(turns[j])
            if not text or count(text) > TURN_MAX:
                break
            if sum(count(b["nl"]) for b in before) + count(text) > CONTEXT_MAX:
                break
            before.insert(0, {"label": speaker_label(turns[j]), "nl": text, "turn": j})
        if not before:
            continue
        # A passage that opens its turn reads on from the turns before it; one from the middle less so
        first_line = p["nl"].split("\n")[0]
        opens = turn_text(turns[t]).startswith(first_line)
        candidates.append((not opens, -len(before), stable(p["pid"]), p, before, turns[t]))
    candidates.sort(key=lambda c: c[:3])
    chains = []
    for _, _, _, p, before, own in candidates[:N_HANDELINGEN]:
        turns = [{"kind": "context", "label": b["label"], "nl": b["nl"], "en": None, "turn": b["turn"]} for b in before]
        turns.append({"kind": "passage", "label": speaker_label(own), "nl": p["nl"], "en": p["en"],
                      "turn": int(p["pid"].split(":")[2][1:])})
        chains.append({"chain": p["pid"], "set": "handelingen", "pair_id": p["id"], "turns": turns})
    return chains


def nemotron_conversations(path):
    """The complete JSON lines of a chunk of chat.jsonl (a byte range: the first and last line may be cut)."""
    rows = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def is_lmarena(row):
    return "lmarena" in json.dumps(row.get("metadata") or {}, ensure_ascii=False).lower()


def nemotron_chains(rows, count):
    """Up to N_NEMOTRON short lmarena conversations in English with at least two user turns."""
    import langid

    chosen = []
    for row in rows:
        if not is_lmarena(row):
            continue
        messages = row.get("messages") or []
        turns = [{"role": m.get("role"), "text": m.get("content")} for m in messages]
        if any(t["role"] not in NEMOTRON_LABELS or not isinstance(t["text"], str) or not t["text"].strip()
               for t in turns):
            continue
        if sum(t["role"] == "user" for t in turns) < 2 or turns[-1]["role"] != "assistant":
            continue
        lengths = [count(t["text"]) for t in turns]
        if max(lengths) > NEMO_TURN_MAX or sum(lengths) > CHAT_MAX:
            continue
        if langid.classify(" ".join(t["text"] for t in turns))[0] != "en":
            continue
        uuid = str(row.get("uuid") or row.get("id") or stable(json.dumps(messages)))
        chosen.append((stable(uuid), uuid, turns))
    chosen.sort()
    return [{"chain": uuid, "set": "nemotron", "pair_id": None,
             "turns": [{"kind": "turn", "label": NEMOTRON_LABELS[t["role"]], "nl": None, "en": t["text"].strip(),
                        "turn": k} for k, t in enumerate(turns)]}
            for _, uuid, turns in chosen[:N_NEMOTRON]]


def context_data(args):
    pairs = read_rows(args.pairs, ["id", "pid", "source", "nl", "en"])
    count = Counter(args.tokenizer)
    doc_ids = {p["pid"].split(":")[1] for p in pairs if p["source"] == "handelingen"}
    debates = handelingen_rows(args.handelingen, doc_ids)
    chains = handelingen_chains(pairs, debates, count)
    rows = nemotron_conversations(args.nemotron)
    chats = nemotron_chains(rows, count)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "chains.json").write_text(json.dumps(chains + chats, ensure_ascii=False, indent=1), encoding="utf-8")
    print("CONTEXT-DATA " + json.dumps({
        "handelingen": len(chains), "with_two_turns": sum(len(c["turns"]) == 3 for c in chains),
        "nemotron_lines": len(rows), "nemotron_lmarena": sum(map(is_lmarena, rows)), "nemotron": len(chats),
        "turns_per_chat": [len(c["turns"]) for c in chats]}))
    if len(chains) < N_HANDELINGEN or len(chats) < N_NEMOTRON:
        raise SystemExit(f"only {len(chains)} Handelingen chains and {len(chats)} conversations")


# ------------------------------------------------------------------ context run (TPU)


def run_round(device, codec, tokenizer, items, eos_ids, sync_every):
    """[(text, ended, tokens_in, tokens_out)] of items [(text, context, label)], translated in one decode_all."""
    import translate as tr

    prompts = [codec.encode(text, context, label) for text, context, label in items]
    outputs = tr.decode_all(prompts, CONTEXT_SHAPES["slots"], device, CONTEXT_SHAPES["max_decode"], sync_every,
                            eos_ids)
    return [(tokenizer.decode(tokens, skip_special_tokens=True).strip(), ended, len(prompt), len(tokens) + int(ended))
            for prompt, (tokens, ended) in zip(prompts, outputs)]


def translate_chains(chains, translate):
    """The trial's rounds on chains (in place). translate(name, direction, [(text, context, label)]) gives
    [(translation, ended, prompt tokens, output tokens)]. A turn without English (a Handelingen turn before a passage)
    is first translated nl -> en on its own; then every turn en -> nl on its own (nl_plain), and turn by turn with the
    earlier turns of its chain as context, their English and the Dutch made with context before (nl_context; the
    first turn has no earlier turns, so its nl_context is its nl_plain)."""
    todo = [t for c in chains for t in c["turns"] if t["en"] is None]
    if todo:
        for t, (text, ended, _, _) in zip(todo, translate("context-nl-en", "nl-en", [(t["nl"], None, None) for t in todo])):
            t["en"], t["en_ended"] = text, ended
    turns = [t for c in chains for t in c["turns"]]
    for t, (text, ended, n_in, n_out) in zip(turns, translate("plain-en-nl", "en-nl", [(t["en"], None, None) for t in turns])):
        t.update(nl_plain=text, plain_ended=ended, plain_tokens_in=n_in, plain_tokens_out=n_out)
    for c in chains:
        first = c["turns"][0]
        first.update(nl_context=first["nl_plain"], context_ended=first["plain_ended"],
                     context_tokens_in=first["plain_tokens_in"], context_tokens_out=first["plain_tokens_out"])
    for k in range(1, max(len(c["turns"]) for c in chains)):
        live = [c for c in chains if len(c["turns"]) > k]
        items = []
        for c in live:
            context = [{"label": e["label"], "source": e["en"], "translation": e["nl_context"]} for e in c["turns"][:k]]
            items.append((c["turns"][k]["en"], context, c["turns"][k]["label"]))
        for c, (text, ended, n_in, n_out) in zip(live, translate(f"context-en-nl-{k}", "en-nl", items)):
            c["turns"][k].update(nl_context=text, context_ended=ended, context_tokens_in=n_in, context_tokens_out=n_out)
    return chains


def context_run(args):
    import jax
    import translate as tr
    from transformers import AutoTokenizer

    model = tr.MODELS[args.model]
    if (jax.device_count(), jax.process_count()) != (model["chips"], 1):
        raise SystemExit(f"needs one process with {model['chips']} chips, has {jax.device_count()} devices")
    chains = json.loads((Path(args.data) / "chains.json").read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    en_nl = tr.ModelPrompts(tokenizer, model, "en", "nl")
    nl_en = tr.ModelPrompts(tokenizer, model, "nl", "en")
    sample = [c for c in chains if len(c["turns"]) > 1][0]["turns"]
    en_nl.check([sample[-1]["en"] or "x"], [[{"label": sample[0]["label"], "source": "a", "translation": "b"}]],
                [sample[-1]["label"]])
    print(f"MAXTEXT {json.dumps({'model': args.model, 'ckpt': args.ckpt, **tr.maxtext_source(), 'patches': tr.patch_maxtext()})}",
          flush=True)
    started = time.time()
    shapes = dict(CONTEXT_SHAPES)
    engine_args = argparse.Namespace(model=args.model, ckpt=args.ckpt, work_dir=args.work_dir)
    engine, params = tr.make_engine(engine_args, shapes, model["chips"])
    device = tr.EngineSlots(engine, params, shapes["prefill_lengths"])
    print(f"engine ready in {time.time() - started:.0f} s", flush=True)
    eos = model["eos_ids"]

    codecs = {"nl-en": nl_en, "en-nl": en_nl}

    def timed(name, direction, items):
        t0 = time.time()
        out = run_round(device, codecs[direction], tokenizer, items, eos, args.sync_every)
        seconds = time.time() - t0
        print("ROUND " + json.dumps({"round": name, "items": len(items), "seconds": round(seconds, 1),
                                     "tokens_in": sum(o[2] for o in out), "tokens_out": sum(o[3] for o in out),
                                     "longest_prompt": max(o[2] for o in out),
                                     "not_ended": sum(not o[1] for o in out)}), flush=True)
        return out

    translate_chains(chains, timed)
    rows = []
    for c in chains:
        for k, t in enumerate(c["turns"]):
            rows.append({"set": c["set"], "chain": c["chain"], "pair_id": c["pair_id"], "position": k,
                         "kind": t["kind"], "label": t["label"], "turn": t["turn"], "nl": t["nl"], "en": t["en"],
                         "en_ended": t.get("en_ended"), "nl_plain": t["nl_plain"], "plain_ended": t["plain_ended"],
                         "nl_context": t["nl_context"], "context_ended": t["context_ended"],
                         "plain_tokens_in": t["plain_tokens_in"], "context_tokens_in": t["context_tokens_in"]})
    write_parquet(rows, Path(args.out) / "context.parquet")
    print(f"CONTEXT-RUN {json.dumps({'chains': len(chains), 'turns': len(rows), 'seconds': round(time.time() - started)})}",
          flush=True)


# ------------------------------------------------------------------ merge and measures


def merge(args):
    import pairs as pm
    import publish
    import sacrebleu
    import structure

    pairs = read_rows(args.pairs)
    en_g4 = pm.read_shards(args.nl_en, "text_en")
    nl_g4 = pm.read_shards(args.en_nl, "text_nl")
    rows = []
    for p in pairs:
        # The pilot's pairs.parquet predates the structure check, so those two columns are computed here when absent.
        for key, (src, tgt) in (("structure", ("nl", "en")), ("back_structure", ("en", "back"))):
            if key not in p:
                p[key] = structure.differences(p[src], p[tgt]) if p[tgt] else None
        e, e_ended, e_tokens = en_g4.get(p["id"], (None, None, None))
        n, n_ended, n_tokens = nl_g4.get(p["id"], (None, None, None))
        row = {k: p[k] for k in ("id", "pid", "source", "half", "nl", "en", "back", "ended", "back_ended",
                                 "tokens_nl", "tokens_en", "chrf", "decode_loop", "dutch_left", "back_decode_loop",
                                 "structure", "back_structure")}
        row["back_english_left"] = bool(p["back"]) and publish.english_left(p["back"], p["en"])
        row.update({"en_g4": e, "en_g4_ended": e_ended, "en_g4_tokens": e_tokens, "nl_g4": n, "nl_g4_ended": n_ended,
                    "nl_g4_tokens": n_tokens})
        if e is not None:
            g = pm.measure(p["nl"], e, None)
            row.update({"g4_decode_loop": g["decode_loop"], "g4_dutch_left": g["dutch_left"],
                        "g4_structure": g["structure"], "g4_len_ratio": g["len_ratio"],
                        "chrf_en_g4_vs_en": round(sacrebleu.sentence_chrf(e, [p["en"]]).score, 3)})
        if n is not None:
            row.update({"chrf_g4": round(sacrebleu.sentence_chrf(n, [p["nl"]]).score, 3),
                        "g4_back_decode_loop": bool(n) and publish.decode_loop(n, p["en"]),
                        "g4_english_left": bool(n) and publish.english_left(n, p["en"]),
                        "g4_back_structure": structure.differences(p["en"], n)})
        rows.append(row)
    write_parquet(rows, args.out)
    print("MERGE " + json.dumps({"pairs": len(rows), "en_g4": sum(r["en_g4"] is not None for r in rows),
                                 "nl_g4": sum(r["nl_g4"] is not None for r in rows)}))


# ------------------------------------------------------------------ report


def num(x, digits=1):
    if x is None:
        return "–"
    return f"{x:,.{digits}f}".replace(",", " ").replace(".", ",").replace(" ", ".")


def quantiles(values):
    values = sorted(v for v in values if v is not None)
    if not values:
        return {"n": 0}
    pick = lambda q: values[min(len(values) - 1, int(q * len(values)))]  # noqa: E731
    return {"n": len(values), "mean": statistics.fmean(values), "median": statistics.median(values),
            "p5": pick(0.05), "p95": pick(0.95)}


def quote(text):
    if text is None:
        return "> –"
    return "\n".join("> " + line if line else ">" for line in text.split("\n"))


def log_lines(directory, prefix):
    """The JSON of every line starting with prefix in the logs of directory: {log name: [dict]}."""
    out = {}
    for path in sorted(Path(directory).glob("*.log")):
        found = []
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith(prefix + " "):
                found.append(json.loads(line[len(prefix) + 1:]))
        out[path.stem] = found
    return out


def throughput_rows(logs):
    rows = []
    for name, shards in sorted(log_lines(logs, "SHARD").items()):
        for s in shards:
            rows.append({"log": name, **s})
    return rows


def report(args):
    import pyarrow.parquet as pq

    rows = pq.read_table(args.compare).to_pylist()
    qe = {r["id"]: r for r in pq.read_table(args.qe).to_pylist()}
    for r in rows:
        r.update({k: v for k, v in qe.get(r["id"], {}).items() if k != "id"})
    setup = json.loads(Path(args.setup).read_text(encoding="utf-8")) if args.setup else {}
    by_pid = {r["pid"]: r for r in rows}
    pids = [line.strip() for line in Path(args.pids).read_text().splitlines() if line.strip()]
    context = pq.read_table(args.context).to_pylist() if args.context else []
    L = ["# Gemma 4 31B tegen TranslateGemma 27B op de pilotparen van de vertaler", ""]
    L += [f"Gemaakt {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())} door `gemma4_pilot.py report` "
          "(branch `translate-generic` van mbert-jax). Alle vertalingen zijn greedy, door de modellen zelf; niets is "
          "met de hand vertaald of verbeterd.", ""]
    L += ["## Opzet", ""]
    L += ["- **Paren**: de 2.000 paren van de pilot (`gs://t5-dutch-english-usc2/mbert/translator/pilot/pairs.parquet`). "
          "en→nl: Gemma 4 vertaalt hetzelfde synthetische Engels (TranslateGemma 27B, nl→en) dat TranslateGemma's eigen "
          "terugvertaling als invoer had; nl→en: Gemma 4 vertaalt het Nederlandse origineel.",
          "- **TranslateGemma 27B**: de uitvoer van de pilot (8–9 nov, MaxText 0.2.5, één v4-host van 4 chips per "
          "worker, eigen chat-template).",
          "- **Gemma 4 31B-it** (`google/gemma-4-31B-it` @ 842da37): `translate.py --model gemma4-31b`, prompt `chat-v1` "
          "zonder context: TranslateGemma's eigen woorden in Gemma 4's chat-template (lege denkruimte, thinking uit).",
          f"- **MaxText**: {setup.get('maxtext', '–')}.",
          f"- **VM**: {setup.get('vm', '–')}.", ""]

    # Throughput
    L += ["## Doorvoer", ""]
    L += ["Per lengteklasse, alleen decodeertijd (laden en compileren apart). Eén proces met 4 chips per richting; de "
          "twee richtingen draaiden tegelijk op de twee helften van de v6e-8.", ""]
    L += ["| richting | klasse | slots | segmenten | prompt-tokens | uitvoertokens | s | uitvoer tok/s | prompt+uitvoer tok/s |",
          "|---|---|---|---|---|---|---|---|---|"]
    totals = {}
    for s in throughput_rows(args.logs):
        direction = "en→nl" if "en_nl" in s["log"] else "nl→en"
        t = totals.setdefault(direction, [0, 0, 0, 0.0])
        t[0] += s["segments"]
        t[1] += s["tokens_in"]
        t[2] += s["tokens_out"]
        t[3] += s["seconds"]
        L.append(f"| {direction} | {s['class']} | {setup.get('slots', {}).get(s['class'], '–')} | {s['segments']} | "
                 f"{num(s['tokens_in'], 0)} | {num(s['tokens_out'], 0)} | {num(s['seconds'])} | {num(s['out_per_s'], 0)} | "
                 f"{num(s['all_per_s'], 0)} |")
    for direction, (n, tin, tout, sec) in sorted(totals.items()):
        L.append(f"| **{direction} samen** | | | {n} | {num(tin, 0)} | {num(tout, 0)} | {num(sec)} | "
                 f"**{num(tout / sec, 0)}** | {num((tin + tout) / sec, 0)} |")
    L += ["", "TranslateGemma 27B in de pilot (HANDOVER van `en-nl-translator`, per v4-worker van 4 chips, ~500 "
          "segmenten per worker): nl→en 504–535 uitvoertokens/s (1.474–1.558 prompt+uitvoer), en→nl 543–559 "
          "(1.227–1.263).", "",
          "Niet gelijk: v6e-chips hebben meer rekenkracht en geheugenbandbreedte dan v4-chips (918 tegen 275 TFLOPS "
          "bf16, 1,6 tegen 1,2 TB/s), allebei 32 GB HBM; Gemma 4 31B heeft ~2× zoveel KV-cache per token (0,90 tegen "
          "0,51 MB), dus minder slots; en deze run had ~2.000 segmenten per proces tegen ~500, dus vollere rondes. "
          "Per chip en per model zijn de getallen daarom alleen een grove vergelijking.", ""]

    # Quality
    def share(key, subset, test=lambda v: v is True):
        known = [r.get(key) for r in subset if r.get(key) is not None]
        hits = sum(1 for v in known if test(v))
        return f"{hits} ({num(100 * hits / len(known), 1)}%)" if known else "–"

    both = [r for r in rows if r.get("nl_g4") is not None and r.get("en_g4") is not None]
    L += ["## Kwaliteit", "", f"{len(both)} paren met beide Gemma 4-vertalingen.", ""]
    L += ["### en→nl (het synthetische Engels naar het Nederlands; het origineel is de referentie)", ""]
    q_tg, q_g4 = quantiles([r.get("qe_en_back") for r in both]), quantiles([r.get("qe_en_nl_g4") for r in both])
    c_tg, c_g4 = quantiles([r.get("chrf") for r in both]), quantiles([r.get("chrf_g4") for r in both])
    L += ["| maat | TranslateGemma 27B | Gemma 4 31B |", "|---|---|---|",
          f"| chrF tegen origineel, mediaan (p5–p95) | {num(c_tg.get('median'))} ({num(c_tg.get('p5'))}–{num(c_tg.get('p95'))}) | "
          f"{num(c_g4.get('median'))} ({num(c_g4.get('p5'))}–{num(c_g4.get('p95'))}) |",
          f"| chrF, gemiddeld | {num(c_tg.get('mean'))} | {num(c_g4.get('mean'))} |",
          f"| MetricX-24-QE, mediaan (p5–p95; lager is beter) | {num(q_tg.get('median'), 2)} ({num(q_tg.get('p5'), 2)}–{num(q_tg.get('p95'), 2)}) | "
          f"{num(q_g4.get('median'), 2)} ({num(q_g4.get('p5'), 2)}–{num(q_g4.get('p95'), 2)}) |",
          f"| MetricX-24-QE, gemiddeld | {num(q_tg.get('mean'), 2)} | {num(q_g4.get('mean'), 2)} |",
          f"| niet afgelopen (decodeerlimiet) | {share('back_ended', both, lambda v: v is False)} | {share('nl_g4_ended', both, lambda v: v is False)} |",
          f"| lus (`decode_loop`) | {share('back_decode_loop', both)} | {share('g4_back_decode_loop', both)} |",
          f"| Engels blijven staan (`english_left`) | {share('back_english_left', both)} | {share('g4_english_left', both)} |",
          f"| opmaak anders dan de bron (`structure`) | {share('back_structure', both, lambda v: bool(v))} | "
          f"{share('g4_back_structure', both, lambda v: bool(v))} |"]
    wins = [(r["qe_en_nl_g4"] < r["qe_en_back"]) for r in both if r.get("qe_en_nl_g4") is not None and r.get("qe_en_back") is not None]
    cw = [(r["chrf_g4"] > r["chrf"]) for r in both if r.get("chrf_g4") is not None and r.get("chrf") is not None]
    L += [f"| Gemma 4 beter per paar (QE lager / chrF hoger) | | {num(100 * sum(wins) / max(len(wins), 1), 1)}% / "
          f"{num(100 * sum(cw) / max(len(cw), 1), 1)}% |", ""]
    L += ["### nl→en (het origineel naar het Engels: kan Gemma 4 de back-translation doen?)", ""]
    q_tg, q_g4 = quantiles([r.get("qe_nl_en") for r in both]), quantiles([r.get("qe_nl_en_g4") for r in both])
    agree = quantiles([r.get("chrf_en_g4_vs_en") for r in both])
    L += ["| maat | TranslateGemma 27B | Gemma 4 31B |", "|---|---|---|",
          f"| MetricX-24-QE, mediaan (p5–p95) | {num(q_tg.get('median'), 2)} ({num(q_tg.get('p5'), 2)}–{num(q_tg.get('p95'), 2)}) | "
          f"{num(q_g4.get('median'), 2)} ({num(q_g4.get('p5'), 2)}–{num(q_g4.get('p95'), 2)}) |",
          f"| MetricX-24-QE, gemiddeld | {num(q_tg.get('mean'), 2)} | {num(q_g4.get('mean'), 2)} |",
          f"| niet afgelopen | {share('ended', both, lambda v: v is False)} | {share('en_g4_ended', both, lambda v: v is False)} |",
          f"| lus (`decode_loop`) | {share('decode_loop', both)} | {share('g4_decode_loop', both)} |",
          f"| Nederlands blijven staan (`dutch_left`) | {share('dutch_left', both)} | {share('g4_dutch_left', both)} |",
          f"| opmaak anders dan de bron (`structure`) | {share('structure', both, lambda v: bool(v))} | "
          f"{share('g4_structure', both, lambda v: bool(v))} |"]
    wins = [(r["qe_nl_en_g4"] < r["qe_nl_en"]) for r in both if r.get("qe_nl_en_g4") is not None and r.get("qe_nl_en") is not None]
    L += [f"| Gemma 4 beter per paar (QE lager) | | {num(100 * sum(wins) / max(len(wins), 1), 1)}% |",
          f"| chrF van Gemma 4's Engels tegen dat van TranslateGemma (overeenkomst, geen kwaliteit), mediaan | | "
          f"{num(agree.get('median'))} |", ""]
    L += ["MetricX-24 hybrid XXL v2p6 in QE-modus (zonder referentie), `metricx_qe.py` (eigen JAX-port, in de pilot op de "
          "CPU gelijk aan transformers), hier op 4 chips van de v6e-8. Kolommen: `qe_en_back` (TranslateGemma en→nl), "
          "`qe_en_nl_g4`, `qe_nl_en` (TranslateGemma nl→en), `qe_nl_en_g4`. chrF: sacrebleu chrF2 per zin.", ""]

    # Per source
    L += ["### Per bron (mediaan MetricX-QE, TranslateGemma / Gemma 4)", "", "| bron | n | en→nl | nl→en |",
          "|---|---|---|---|"]
    for source in sorted({r["source"] for r in both}):
        sub = [r for r in both if r["source"] == source]
        a, b = quantiles([r.get("qe_en_back") for r in sub]), quantiles([r.get("qe_en_nl_g4") for r in sub])
        c, d = quantiles([r.get("qe_nl_en") for r in sub]), quantiles([r.get("qe_nl_en_g4") for r in sub])
        L.append(f"| {source} | {len(sub)} | {num(a.get('median'), 2)} / {num(b.get('median'), 2)} | "
                 f"{num(c.get('median'), 2)} / {num(d.get('median'), 2)} |")
    L.append("")

    # Problem cases
    def listing(title, items, fields):
        out = [f"### {title}", ""]
        if not items:
            return out + ["Geen.", ""]
        for r in items:
            out.append(f"- `{r['pid']}`: " + ", ".join(f"{k} {v}" for k, v in fields(r)))
        return out + [""]

    L += listing("Gemma 4 en→nl niet afgelopen, met lus of met Engels", [r for r in both if r.get("nl_g4_ended") is False or
                 r.get("g4_back_decode_loop") or r.get("g4_english_left")],
                 lambda r: [("ended", r.get("nl_g4_ended")), ("lus", r.get("g4_back_decode_loop")),
                            ("Engels", r.get("g4_english_left")), ("tokens", r.get("nl_g4_tokens"))])
    L += listing("Gemma 4 nl→en niet afgelopen, met lus of met Nederlands", [r for r in both if r.get("en_g4_ended") is False or
                 r.get("g4_decode_loop") or r.get("g4_dutch_left")],
                 lambda r: [("ended", r.get("en_g4_ended")), ("lus", r.get("g4_decode_loop")),
                            ("Nederlands", r.get("g4_dutch_left")), ("tokens", r.get("en_g4_tokens"))])

    # The 30 pairs
    L += ["## De 30 paren van `pairs30.md` naast elkaar", "",
          "Volgorde en selectie als in `pairs30.md` (seed 0, gestratificeerd naar bron). chrF: tegen het origineel; "
          "QE: MetricX-24-QE, lager is beter.", ""]
    for k, pid in enumerate(pids, 1):
        r = by_pid.get(pid)
        if r is None:
            raise SystemExit(f"pair {pid} of {args.pids} is not in {args.compare}")
        L += [f"### {k}. {r['source']} — `{pid}`", "",
              f"en→nl: TranslateGemma chrF **{num(r.get('chrf'))}**, QE **{num(r.get('qe_en_back'), 2)}** · Gemma 4 chrF "
              f"**{num(r.get('chrf_g4'))}**, QE **{num(r.get('qe_en_nl_g4'), 2)}** · nl→en QE: TranslateGemma "
              f"{num(r.get('qe_nl_en'), 2)} · Gemma 4 {num(r.get('qe_nl_en_g4'), 2)}", "",
              "**Nederlands origineel:**", "", quote(r["nl"]), "",
              "**TranslateGemma 27B, en→nl** (uit het synthetische Engels):", "", quote(r["back"]), "",
              "**Gemma 4 31B, en→nl** (uit hetzelfde Engels):", "", quote(r["nl_g4"]), "",
              "**TranslateGemma 27B, nl→en** (het synthetische Engels van de pilot):", "", quote(r["en"]), "",
              "**Gemma 4 31B, nl→en:**", "", quote(r["en_g4"]), ""]

    # Context trial
    if context:
        L += context_section(context)
    Path(args.out).write_text("\n".join(L) + "\n", encoding="utf-8")
    summary = {"pairs": len(both)}
    for key in ("qe_en_back", "qe_en_nl_g4", "qe_nl_en", "qe_nl_en_g4", "chrf", "chrf_g4", "chrf_en_g4_vs_en"):
        summary[key] = quantiles([r.get(key) for r in both])
    Path(args.out).with_suffix(".json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"REPORT {args.out}")


def context_section(context):
    import sacrebleu

    L = ["## Proef met context: per beurt vertaald, met en zonder de eerdere beurten", "",
         "Prompt `chat-v1` met context: TranslateGemma's eigen woorden, daarna de eerdere beurten (spreker, Engels en de "
         "Nederlandse vertaling die Gemma 4 er eerder in dezelfde keten van maakte), daarna de huidige beurt, afgebakend, "
         "met de opdracht alleen die beurt te vertalen. Zonder context: dezelfde beurt met het prompt zonder context. "
         "Beide in één engine (prefill tot 4.096 tokens), greedy.", "",
         "- **Handelingen** (20): een pilotpassage met de 1–2 spreekbeurten ervoor in hetzelfde debat. Het Engels van "
         "die eerdere beurten maakte Gemma 4 (nl→en, zonder context); het Engels van de passage is het synthetische "
         "Engels van de pilot (TranslateGemma 27B). chrF tegen het origineel.",
         "- **Nemotron** (5): korte lmarena-gesprekken uit `nvidia/Nemotron-SFT-Instruction-Following-Chat-v3` "
         "(alleen de content van user en assistant). Geen referentie.", ""]
    chains = {}
    for r in context:
        chains.setdefault((r["set"], r["chain"]), []).append(r)
    hand = [v for (s, _), v in chains.items() if s == "handelingen"]
    passages = [sorted(v, key=lambda r: r["position"])[-1] for v in hand]
    with_c = [sacrebleu.sentence_chrf(p["nl_context"], [p["nl"]]).score for p in passages if p["nl_context"]]
    without = [sacrebleu.sentence_chrf(p["nl_plain"], [p["nl"]]).score for p in passages if p["nl_plain"]]
    tags = sum(bool(re.search(r"</?(current_turn|turn|earlier_turns|en|nl)\b", r["nl_context"] or "")) for r in context)
    L += ["| maat | zonder context | met context |", "|---|---|---|",
          f"| Handelingen-passages, chrF tegen origineel, mediaan | {num(statistics.median(without))} | "
          f"{num(statistics.median(with_c))} |",
          f"| Handelingen-passages, chrF, gemiddeld | {num(statistics.fmean(without))} | {num(statistics.fmean(with_c))} |",
          f"| beurten niet afgelopen | {sum(r['plain_ended'] is False for r in context)} | "
          f"{sum(r['context_ended'] is False for r in context)} |",
          f"| uitvoer met een tag van het prompt (`<turn>`, `<current_turn>` ...) | – | {tags} |",
          f"| langste prompt (tokens) | {max(r['plain_tokens_in'] for r in context)} | "
          f"{max(r['context_tokens_in'] for r in context)} |", ""]
    for (s, chain), turns in chains.items():
        turns.sort(key=lambda r: r["position"])
        title = "Handelingen" if s == "handelingen" else "Nemotron"
        L += [f"### {title} — `{chain}`", ""]
        for t in turns:
            head = f"**Beurt {t['position'] + 1}, {t['label'] or 'zonder label'}**"
            if t["kind"] == "context":
                head += " (context; Engels door Gemma 4)"
            elif t["kind"] == "passage":
                head += " (de pilotpassage)"
            L += [head, ""]
            if t["nl"] is not None and t["kind"] == "passage":
                L += ["Nederlands origineel:", "", quote(t["nl"]), ""]
            elif t["nl"] is not None:
                L += ["<details><summary>Nederlands origineel</summary>", "", quote(t["nl"]), "", "</details>", ""]
            L += ["Engels (bron):", "", quote(t["en"]), "", "Gemma 4 zonder context:", "", quote(t["nl_plain"]), ""]
            if t["position"] == 0:
                L += ["Met context: gelijk aan zonder (de eerste beurt heeft geen eerdere beurten).", ""]
            else:
                L += ["Gemma 4 met context:", "", quote(t["nl_context"]), ""]
            if s == "handelingen" and t["kind"] == "passage":
                L += [f"chrF tegen origineel: zonder context {num(sacrebleu.sentence_chrf(t['nl_plain'], [t['nl']]).score)}, "
                      f"met context {num(sacrebleu.sentence_chrf(t['nl_context'], [t['nl']]).score)}", ""]
    return L


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("segments")
    p.add_argument("--pairs", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("context-data")
    p.add_argument("--pairs", required=True)
    p.add_argument("--handelingen", required=True)
    p.add_argument("--nemotron", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("context-run")
    p.add_argument("--data", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--model", default="gemma4-31b")
    p.add_argument("--out", required=True)
    p.add_argument("--work-dir", default="/dev/shm/translate-generic")
    p.add_argument("--sync-every", type=int, default=8)
    p = sub.add_parser("merge")
    p.add_argument("--pairs", required=True)
    p.add_argument("--en-nl", required=True)
    p.add_argument("--nl-en", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("report")
    p.add_argument("--compare", required=True)
    p.add_argument("--qe", required=True)
    p.add_argument("--context", default=None)
    p.add_argument("--pids", required=True)
    p.add_argument("--logs", required=True)
    p.add_argument("--setup", default=None)
    p.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    commands = {"segments": segments, "context-data": context_data, "context-run": context_run, "merge": merge,
                "report": report}
    return commands[args.command](args)


if __name__ == "__main__":
    # As in translate.py: a thread that is no daemon must not keep a failed process (and the TPU) alive
    import os
    import traceback

    code = 0
    try:
        code = main() or 0
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
