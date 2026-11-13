#!/usr/bin/env python3
"""Pilot of 12 Nov: a back-translation (nl->en) in more natural and complete English, so that the back-translation pairs
of the en->nl translator need no QE filter (Yeb's aim). TranslateGemma 27B's English (the published pairs) is wordier
than the Dutch it comes from, and every trained arm learned to compress. Gemma 4 31B-it translates a stratified sample
of the same Dutch passages again with translate.py's prompts natural-v1 and bare-v1; the measures are those of the
pairs (pairs.py), MetricX-24-QE in both directions (metricx_qe.py) and R2's length bound (sft_build.py).

    python bt_natural.py fetch --out DIR
    python bt_natural.py sample --data DIR --tokenizer TOK --out DIR
    python bt_natural.py show --segments probe.parquet --shards NAME=DIR ... [--tokenizer TOK --source nl --target en]
    python bt_natural.py merge --sample sample.parquet --shards natural=DIR --shards bare=DIR --out compare.parquet
    python bt_natural.py report --compare compare.parquet --qe qe-0.parquet qe-1.parquet --logs DIR \\
        --fullset fullset.json --setup setup.json --out bt-natural.md
    python bt_natural.py refqe --evalsets four/evalsets.parquet --qe refqe.parquet --base tg27b --out refqe.md
    python bt_natural.py evalsegments --four four/evalsets.parquet --x x/evalsets.parquet --tokenizer TOK --out DIR
    python bt_natural.py evalcompare --data DIR/four --pairs pairs-four.parquet --metricx metricx-four.parquet \\
        --comet comet-four.parquet --flan-pairs pairs-flan.parquet --flan-metricx metricx-flan.parquet \\
        --base tg27b --logs DIR --out eval-gemma4.md

fetch: the five published datasets of collection yhavinga/dutch-back-translation-nlen (TranslateGemma 27B's English
of each Dutch passage, with its scores and checks) at fixed revisions, straight from Hugging Face to this VM.

sample: per source STRATA (random first, then the long passages, then QE en->nl > 5, then the lowest decile of Dutch
words per English word; each from the train rows not drawn yet, in the order of a hash of the id), the segments for
translate.py (Gemma 4's token counts; every prompt of PROMPTS must fit its class's largest prefill), six probe passages
of the class medium, and fullset.json: the passages and Gemma 4 tokens per source and class of the whole published BT
(train and validation, and the Handelingen's context turns), for the estimate of a full new back-translation.

merge: per passage TranslateGemma's English (tg) and each setup's, with the measures of pairs.py (pairs.measure,
pairs.hard_fails), Dutch words per English word, characters, R2's length bound and `chatter` (an introduction or a
note around the translation).

report: bt-natural.md and .json: per setup and source the length ratio, MetricX-24-QE nl->en (source Dutch, candidate
English) and en->nl (source English, candidate the Dutch original, the measure of the pairs' filter), the share a
filter would still hit, the hard checks, the throughput and the TPU time of a full back-translation, and examples.

refqe: the calibration of rule (a): MetricX-24-QE en->nl of the four sets' human references (source the English,
candidate the Dutch reference; for Europarl-direct the Dutch original), against TranslateGemma 27B's QE.

evalsegments / evalcompare: Gemma 4 31B-it as an en->nl translator (natural-v1) on the translator's four evaluation
sets (evalsets.py) and the 1,000 FLAN rows of the classifier (armeval.py data --kind x), against TranslateGemma 27B's
starting point (protocol rule (a): MetricX-24-QE at most +0.20 on WMT24++ (960), FLORES and NTREX).
"""

import argparse
import collections
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

# The published datasets (collection yhavinga/dutch-back-translation-nlen-6af45e2ba548b444c6362255), revisions of 11 Nov
SOURCES = {
    "handelingen": ("yhavinga/handelingen-tweede-kamer-nl-backtranslated-en", "b12d17adf134e06718182b2e81e95a5672782f05"),
    "raad": ("yhavinga/openraadsinformatie-nl-backtranslated-en", "16c10acce1d5336f420c11bb267d9b8c572cedd4"),
    "dpc": ("yhavinga/rijksoverheid-websites-nl-backtranslated-en", "881f04a20059aeaa51c616fa312ee387b7f6bf36"),
    "rechtspraak": ("yhavinga/rechtspraak-nl-backtranslated-en", "b187a9a22745876d7c74f1e49282f5ecb1b03402"),
    "fineweb": ("yhavinga/fineweb-2-nl-backtranslated-en", "763f2cbc76f326b221357aca7c0e9534f7a0d57d"),
}
SOURCE_NAMES = {"handelingen": "Handelingen", "raad": "raad", "dpc": "dpc", "rechtspraak": "rechtspraak",
                "fineweb": "FineWeb-2"}
# R2's length bound (protocol.md §9, sft_build.py --len-factor 4/3): characters Dutch / English within a factor 4/3 of
# the source's median over its train passages without hard_fails (mbert-ops/translator-sft/data-r2-report.md)
R2_BOUNDS = {
    "dpc": {"median": 0.9836, "lo": 0.7377, "hi": 1.3114},
    "fineweb": {"median": 1.003, "lo": 0.7522, "hi": 1.3373},
    "handelingen": {"median": 1.0061, "lo": 0.7546, "hi": 1.3414},
    "raad": {"median": 0.993, "lo": 0.7448, "hi": 1.3239},
    "rechtspraak": {"median": 1.0287, "lo": 0.7715, "hi": 1.3716},
}
# Per source 1,000 passages: a uniform sample (representative), then on purpose the hard cases
STRATA = (("random", 500), ("long", 150), ("qe_high", 175), ("ratio_low", 175))
QE_HIGH = 5.0
QE_FILTERS = (5.0, 7.5)
RATIO_LOW_QUANTILE = 0.10
HUMAN_RATIO = 1.05  # Dutch words per English word of human translations (Europarl 1.07, Global Voices 1.05)
SEED = "bt-natural/0"
PROMPTS = ("natural-v1", "bare-v1")
MODEL = "gemma4-31b"
PROBE_CLASS = "medium"
COLUMNS = ["id", "nl", "en", "qe_en_nl", "length_class", "hard_fails", "ended", "tokens_nl", "tokens_en"]
# An introduction before the translation, or a note after it that the source has no counterpart of. "Of course," or
# "Certainly," alone opens many a faithful translation ("Natuurlijk, ..."), so an introduction must name the translation
INTRO = re.compile(r"^\s*(?:\*\*)?(?:(?:sure|certainly|of course)[,!.]?\s+)?(?:here(?:'s| is)\b[^\n]{0,60}\btranslation|"
                   r"(?:english |dutch )?translation\s*:|hier is\b[^\n]{0,60}\bvertaling|vertaling\s*:)", re.IGNORECASE)
NOTE = re.compile(r"\n\s*(?:\*\*)?(?:note|translator'?s note|opmerking|noot|n\.b\.)\s*:", re.IGNORECASE)


def stable(key):
    return hashlib.sha1(key.encode()).hexdigest()


def words(text):
    return len(text.split())


def word_ratio(nl, en):
    """Dutch words per English word (split on white space), as sft_build.word_ratio"""
    return words(nl) / max(1, words(en))


def char_ratio(nl, en):
    """Dutch characters over English characters (stripped), as sft_build.char_ratio"""
    return len(nl.strip()) / max(1, len(en.strip()))


def quantile(values, q):
    values = sorted(values)
    if not values:
        raise ValueError("quantile of nothing")
    return values[min(len(values) - 1, int(q * (len(values) - 1) + 0.5))]


def chatter(text, source=""):
    """An introduction to the translation, or a note the source does not have"""
    if not text:
        return False
    return bool(INTRO.search(text)) or (bool(NOTE.search(text)) and not NOTE.search(source or ""))


def write_parquet(rows, path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")


def read_rows(path, columns=None):
    import pyarrow.parquet as pq

    return pq.read_table(path, columns=columns).to_pylist()


# ---------------------------------------------------------------------------------------------------- data

def fetch(args):
    from huggingface_hub import snapshot_download

    out = Path(args.out)
    for key, (repo, revision) in SOURCES.items():
        patterns = ["data/*.parquet"] + (["context_turns/*.parquet"] if key == "handelingen" else [])
        snapshot_download(repo, repo_type="dataset", revision=revision, local_dir=str(out / key),
                          allow_patterns=patterns)
        files = sorted(p.relative_to(out / key).as_posix() for p in (out / key).rglob("*.parquet"))
        print(f"FETCH {json.dumps({'source': key, 'repo': repo, 'revision': revision, 'files': files})}", flush=True)


def source_files(data, key, part="data", split="train"):
    files = sorted((Path(data) / key / part).glob(f"{split}-*.parquet"))
    if not files:
        raise SystemExit(f"no {part}/{split}-*.parquet of {key} under {data}")
    return files


def draw(rows, source, strata=STRATA, qe_high=QE_HIGH, ratio_quantile=RATIO_LOW_QUANTILE, seed=SEED):
    """The sample of one source: per stratum, in the order of a hash of the id, the rows that pass its test and were
    not drawn before. Returns (rows with source and stratum, {ratio_cut, short: {stratum: [taken, wanted]}})."""
    cut = quantile([word_ratio(r["nl"], r["en"]) for r in rows], ratio_quantile)
    tests = {"random": lambda r: True,
             "long": lambda r: r["length_class"] == "long",
             "qe_high": lambda r: r["qe_en_nl"] is not None and r["qe_en_nl"] > qe_high,
             "ratio_low": lambda r: word_ratio(r["nl"], r["en"]) <= cut}
    order = sorted(rows, key=lambda r: stable(f"{seed}/{r['id']}"))
    chosen, out, short = set(), [], {}
    for name, n in strata:
        taken = [r for r in order if r["id"] not in chosen and tests[name](r)][:n]
        if len(taken) < n:
            short[name] = [len(taken), n]
        for r in taken:
            chosen.add(r["id"])
            out.append(dict(r, source=source, stratum=name))
    return out, {"ratio_cut": cut, "short": short}


def class_of(tokens, classes):
    """translate.py's length class of a segment: the first whose max_source holds it"""
    for name, c in classes.items():
        if tokens <= c["max_source"]:
            return name
    raise SystemExit(f"a segment of {tokens} tokens is longer than the largest class")


def check_prompts(segments, tokenizer, source, target, prompts=PROMPTS, model=MODEL):
    """Every segment's prompt (each of prompts, in the model's chat turn) within its class's largest prefill length,
    checked against the chat template on a few; returns {prompt: largest prompt overhead in tokens}"""
    import translate as tr

    classes = tr.MODELS[model]["length_classes"]
    overhead = {}
    for name in prompts:
        codec = tr.ModelPrompts(tokenizer, tr.with_prompt(tr.MODELS[model], name), source, target)
        codec.check([s["text"] for s in segments[:3]])
        worst = 0
        for s in segments:
            n = len(codec.encode(s["text"]))
            limit = classes[class_of(s["tokens"], classes)]["prefill_lengths"][-1]
            if n > limit:
                raise SystemExit(f"{s['id']}: prompt {name} has {n} tokens, more than its class's prefill {limit}")
            worst = max(worst, n - s["tokens"])
        overhead[name] = worst
    return overhead


class Counter:
    """Gemma 4 token counts of texts (as gemma4_pilot.Counter)."""

    def __init__(self, tokenizer_dir):
        from tokenizers import Tokenizer

        self.tok = Tokenizer.from_file(str(Path(tokenizer_dir) / "tokenizer.json"))

    def __call__(self, text):
        return len(self.tok.encode(text, add_special_tokens=False).ids)

    def many(self, texts):
        return [len(e.ids) for e in self.tok.encode_batch(list(texts), add_special_tokens=False)]


def fullset(data, count, classes):
    """{source: {class: {passages, tokens}}} of every published passage (train, validation; the Handelingen's context
    turns under handelingen_context), in Gemma 4 tokens; a turn beyond the largest class under over_largest"""
    import pyarrow.parquet as pq

    out = {}
    largest = max(c["max_source"] for c in classes.values())
    parts = [(k, k, "data") for k in SOURCES] + [("handelingen_context", "handelingen", "context_turns")]
    for name, key, part in parts:
        per = collections.defaultdict(lambda: {"passages": 0, "tokens": 0})
        for split in ("train", "validation"):
            for path in source_files(data, key, part, split):
                texts = pq.read_table(path, columns=["nl"]).column("nl").to_pylist()
                for n in count.many(texts):
                    # a context turn is a whole turn and may be longer than the largest class
                    c = per[class_of(n, classes) if n <= largest else "over_largest"]
                    c["passages"] += 1
                    c["tokens"] += n
        out[name] = dict(per)
    return out


def sample(args):
    import translate as tr
    from transformers import AutoTokenizer

    out = Path(args.out)
    rows, info = [], {}
    for key in SOURCES:
        source_rows = [r for path in source_files(args.data, key) for r in read_rows(path, COLUMNS)]
        drawn, info[key] = draw(source_rows, key)
        info[key]["train_rows"] = len(source_rows)
        rows += drawn
        print(f"SAMPLE {json.dumps({'source': key, **info[key], 'drawn': len(drawn)})}", flush=True)
    count = Counter(args.tokenizer)
    classes = tr.MODELS[MODEL]["length_classes"]
    for r, n in zip(rows, count.many(r["nl"] for r in rows)):
        r["tokens_g4"] = n
        r["class_g4"] = class_of(n, classes)
    rows.sort(key=lambda r: r["id"])
    write_parquet(rows, out / "sample.parquet")
    segments = [{"id": r["id"], "text": r["nl"], "tokens": r["tokens_g4"]} for r in rows]
    write_parquet(segments, out / "segments-nl.parquet")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    overhead = check_prompts(segments, tokenizer, "nl", "en")
    # The probe: one passage of class medium per source from the uniform sample, and a second Handelingen passage
    probe = []
    for key in list(SOURCES) + ["handelingen"]:
        pick = next(s for s, r in zip(segments, rows) if r["source"] == key and r["stratum"] == "random"
                    and r["class_g4"] == PROBE_CLASS and s not in probe)
        probe.append(pick)
    write_parquet(probe, out / "probe-nl.parquet")
    by_class = collections.Counter(r["class_g4"] for r in rows)
    info["_all"] = {"rows": len(rows), "classes": dict(sorted(by_class.items())), "prompt_overhead": overhead,
                    "strata": dict(collections.Counter(r["stratum"] for r in rows))}
    if not args.no_fullset:
        info["_fullset"] = fullset(args.data, count, classes)
        (out / "fullset.json").write_text(json.dumps(info["_fullset"], indent=1))
    (out / "sample.json").write_text(json.dumps(info, indent=1))
    (out / "classes.txt").write_text(" ".join(c for c in classes if by_class.get(c)) + "\n")
    print(f"SEGMENTS {json.dumps(info['_all'])}", flush=True)


def evalsegments(args):
    """The segments of the en->nl evaluation: four/ (the four sets as evalsets.py wrote them) and flan/ (the 1,000 FLAN
    rows of armeval.py's kind x), each with evalsets.parquet, segments-w0.parquet (token counts of --model's tokenizer,
    every prompt checked against its chat turn) and classes.txt; and probe-en.parquet (one segment of class small per
    set). With --blind also blind/: the 100 passages of the blind assessment (set blind of the same x), as
    run_g4_blind.sh made them for the 31B."""
    import shutil

    import translate as tr
    from transformers import AutoTokenizer

    model = getattr(args, "model", MODEL)
    count = Counter(args.tokenizer)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    classes = tr.MODELS[model]["length_classes"]
    out = Path(args.out)
    four = read_rows(args.four)
    x = read_rows(args.x)
    flan = [r for r in x if r["set"] == "flan"]
    if len(flan) != 1000:
        raise SystemExit(f"{args.x}: {len(flan)} FLAN rows, expected 1,000")
    parts = [("four", four), ("flan", flan)]
    if getattr(args, "blind", False):
        blind = [r for r in x if r["set"] == "blind"]
        if len(blind) != 100 or len({r["id"] for r in blind}) != 100:
            raise SystemExit(f"{args.x}: {len(blind)} rows of set blind ({len({r['id'] for r in blind})} ids), "
                             f"expected 100")
        parts.append(("blind", blind))
    probe = []
    for name, rows in parts:
        d = out / name
        d.mkdir(parents=True, exist_ok=True)
        write_parquet(rows, d / "evalsets.parquet")
        manifest = Path(args.four if name == "four" else args.x).with_name("manifest.json")
        if manifest.exists():
            shutil.copy(manifest, d / "manifest.json")
        segments = [{"id": r["id"], "text": r["source"], "tokens": n}
                    for r, n in zip(rows, count.many(r["source"] for r in rows))]
        overhead = check_prompts(segments, tokenizer, "en", "nl", prompts=("natural-v1",), model=model)
        write_parquet(segments, d / "segments-w0.parquet")
        present = collections.Counter(class_of(s["tokens"], classes) for s in segments)
        (d / "classes.txt").write_text(" ".join(c for c in classes if present.get(c)) + "\n")
        print(f"EVALSEGMENTS {json.dumps({'part': name, 'rows': len(rows), 'classes': dict(present), 'overhead': overhead})}",
              flush=True)
        by_id = {r["id"]: r for r in rows}
        for set_name in sorted({r["set"] for r in rows} - {"blind"}):
            pick = next((s for s in segments if by_id[s["id"]]["set"] == set_name
                         and class_of(s["tokens"], classes) == "small"), None)
            if pick:
                probe.append(pick)
    write_parquet(probe, out / "probe-en.parquet")


# ---------------------------------------------------------------------------------------------------- probe

def read_shard_rows(root):
    """{id: row} of translate.py's shards anywhere under root"""
    out = {}
    for path in sorted(Path(root).rglob("shard-*.parquet")):
        for r in read_rows(path):
            if r["id"] in out:
                raise SystemExit(f"{r['id']} is in two shards under {root}")
            out[r["id"]] = r
    return out


def show(args):
    """The rendered prompt of the first probe segment per setup and every probe translation, for reading"""
    import translate as tr
    from transformers import AutoTokenizer

    segments = read_rows(args.segments)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    column = f"text_{args.target}"
    for spec in args.shards:
        name, prompt, root = spec.split("=", 2)
        codec = tr.ModelPrompts(tokenizer, tr.with_prompt(tr.MODELS[MODEL], prompt), args.source, args.target)
        print(f"===== {name} ({prompt}): the prompt of {segments[0]['id']} as the model gets it =====")
        print(codec.plain.text(segments[0]["text"]))
        done = read_shard_rows(root)
        for s in segments:
            r = done.get(s["id"])
            print(f"----- {name} {s['id']} ({s['tokens']} tokens)")
            print("SOURCE:", s["text"])
            print("OUTPUT:", None if r is None else r[column])
            if r is not None:
                print(f"ended {r['ended']}, tokens out {r['tokens_out']}, words source/output "
                      f"{words(s['text'])}/{words(r[column] or '')}")


# ---------------------------------------------------------------------------------------------------- merge

def measure(nl, en, source, ended):
    """The measures of one (Dutch original, English) pair: pairs.measure and pairs.hard_fails, word and character
    ratios, R2's length bound and chatter"""
    import pairs as pm

    m = pm.measure(nl, en, None)
    row = {"ended": ended, **{k: m[k] for k in ("decode_loop", "dutch_left", "len_ratio", "numbers_missing",
                                                 "structure", "sentences_nl", "sentences_en")}}
    row["hard_fails"] = pm.hard_fails(row)
    row["chatter"] = chatter(en, nl)
    row["word_ratio"] = round(word_ratio(nl, en), 4)
    row["char_ratio"] = round(char_ratio(nl, en), 4)
    b = R2_BOUNDS[source]
    row["r2_length_out"] = not b["lo"] <= row["char_ratio"] <= b["hi"]
    return row


def merge(args):
    passages = read_rows(args.sample)
    setups = {"tg": None}
    for spec in args.shards:
        name, root = spec.split("=", 1)
        setups[name] = read_shard_rows(root)
    rows = []
    for p in passages:
        row = {k: p[k] for k in ("id", "source", "stratum", "length_class", "class_g4", "tokens_nl", "tokens_g4",
                                 "nl")}
        row["qe_en_nl_published"] = p["qe_en_nl"]
        for name, done in setups.items():
            if done is None:
                en, ended, tokens = p["en"], p["ended"], p["tokens_en"]
            else:
                t = done.get(p["id"])
                if t is None:
                    raise SystemExit(f"{name}: no translation of {p['id']}")
                en, ended, tokens = t["text_en"], t["ended"], t["tokens_out"]
            row[f"en_{name}"] = en
            row[f"tokens_{name}"] = tokens
            for k, v in measure(p["nl"], en, p["source"], ended).items():
                row[f"{k}_{name}"] = v
        rows.append(row)
    write_parquet(rows, args.out)
    print(f"MERGE {json.dumps({'passages': len(rows), 'setups': list(setups)})}", flush=True)


# ---------------------------------------------------------------------------------------------------- report

def nl_num(x, digits=2, signed=False):
    if x is None:
        return "–"
    s = f"{x:+,.{digits}f}" if signed else f"{x:,.{digits}f}"
    return s.replace(",", " ").replace(".", ",").replace(" ", ".")


def pct(x, digits=1):
    return "–" if x is None else nl_num(100 * x, digits) + "%"


def share(rows, test):
    return sum(1 for r in rows if test(r)) / len(rows) if rows else None


def med(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def avg(values):
    values = [v for v in values if v is not None]
    return statistics.fmean(values) if values else None


def setup_stats(rows, s):
    """The measures of setup s over rows (None for no rows)"""
    if not rows:
        return None
    ratios = [r[f"word_ratio_{s}"] for r in rows]
    qe_nl_en, qe_en_nl = f"qe_nl_en_{s}", f"qe_en_{s}_nl"
    out = {"n": len(rows),
           "word_ratio": {"median": med(ratios), "p10": quantile(ratios, 0.1), "p25": quantile(ratios, 0.25),
                          "p75": quantile(ratios, 0.75), "p90": quantile(ratios, 0.9)},
           "under_0_9": share(rows, lambda r: r[f"word_ratio_{s}"] < 0.9),
           "qe_nl_en": {"mean": avg(r[qe_nl_en] for r in rows), "median": med(r[qe_nl_en] for r in rows)},
           "qe_en_nl": {"mean": avg(r[qe_en_nl] for r in rows), "median": med(r[qe_en_nl] for r in rows)},
           "r2_length_out": share(rows, lambda r: r[f"r2_length_out_{s}"]),
           "hard": share(rows, lambda r: bool(r[f"hard_fails_{s}"])),
           "chatter": share(rows, lambda r: r[f"chatter_{s}"]),
           "numbers_missing": share(rows, lambda r: bool(r[f"numbers_missing_{s}"]))}
    for q in QE_FILTERS:
        out[f"qe_over_{q}"] = share(rows, lambda r: r[qe_en_nl] is not None and r[qe_en_nl] > q)
        out[f"filter_{q}"] = share(rows, lambda r: bool(r[f"hard_fails_{s}"]) or r[f"r2_length_out_{s}"]
                                   or r[qe_en_nl] is None or r[qe_en_nl] > q)
    out["structure_categories"] = dict(collections.Counter(c for r in rows for c in (r[f"structure_{s}"] or [])))
    checks = collections.Counter(f for r in rows for f in r[f"hard_fails_{s}"])
    out["checks"] = {k: checks.get(k, 0) for k in ("not_ended", "decode_loop", "dutch_left", "len_ratio", "structure",
                                                     "newline_doubled")}
    out["checks"]["chatter"] = sum(r[f"chatter_{s}"] for r in rows)
    return out


def against_tg(rows, s):
    """Per passage, setup s against TranslateGemma's English"""
    def diff(key_s, key_t):
        d = [r[key_s] - r[key_t] for r in rows if r[key_s] is not None and r[key_t] is not None]
        return {"mean": avg(d), "better": share([x for x in d], lambda x: x < 0)}
    return {"qe_nl_en": diff(f"qe_nl_en_{s}", "qe_nl_en_tg"), "qe_en_nl": diff(f"qe_en_{s}_nl", "qe_en_tg_nl"),
            "words_vs_tg": med(words(r[f"en_{s}"]) / max(1, words(r["en_tg"])) for r in rows)}


def throughput(logs_dir):
    """Per setup (log name translate_<setup>-*.log): SHARD totals per class and the engine loads"""
    out = collections.defaultdict(lambda: {"classes": collections.defaultdict(
        lambda: {"segments": 0, "tokens_in": 0, "tokens_out": 0, "seconds": 0.0}), "engine_seconds": []})
    for path in sorted(Path(logs_dir).glob("translate_*.log")):
        setup = path.stem[len("translate_"):].split("-")[0]
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("SHARD "):
                s = json.loads(line[6:])
                c = out[setup]["classes"][s["class"]]
                for k in ("segments", "tokens_in", "tokens_out", "seconds"):
                    c[k] += s[k]
            m = re.match(r"engine ready in (\d+) s", line)
            if m:
                out[setup]["engine_seconds"].append(int(m.group(1)))
    return {k: {"classes": {c: dict(v) for c, v in d["classes"].items()}, "engine_seconds": d["engine_seconds"]}
            for k, d in out.items()}


def full_estimate(speed, rows, s, full, chips=4):
    """Decode hours of a full back-translation with setup s: per class the published passages' Gemma 4 tokens times the
    sample's output tokens per source token, at the sample's output tokens per second of that class"""
    per_class = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        per_class[r["class_g4"]][0] += r["tokens_g4"]
        per_class[r["class_g4"]][1] += r[f"tokens_{s}"] or 0
    seconds, passages = 0.0, 0
    for parts in full.values():
        for c, v in parts.items():
            src, gen = per_class.get(c, (0, 0))
            measured = speed["classes"].get(c)
            if not src or not measured:
                # a class the sample does not have: the speed and ratio over all of the sample's classes
                src, gen = sum(x[0] for x in per_class.values()), sum(x[1] for x in per_class.values())
                tok_s = sum(m["tokens_out"] for m in speed["classes"].values()) / sum(
                    m["seconds"] for m in speed["classes"].values())
            else:
                tok_s = measured["tokens_out"] / measured["seconds"]
            seconds += v["tokens"] * gen / src / tok_s
            passages += v["passages"]
    hours = seconds / 3600
    return {"passages": passages, "process_hours": hours, "chip_hours": hours * chips, "v6e8_hours": hours * chips / 8}


def cell(text):
    return (text or "").replace("|", "\\|").replace("\n", "<br>")


def examples(rows, s, n_inflate=4, n_omit=4, n_worse=2, max_words=130):
    """Passages where TranslateGemma inflates (fewest Dutch words per English word, where s is closer to the human
    ratio) or omits (highest MetricX-QE nl->en of TranslateGemma above s's), one per source first; then where s is
    worse than TranslateGemma on QE nl->en"""
    ok = [r for r in rows if words(r["nl"]) <= max_words and r["ended_tg"] and r[f"ended_{s}"]
          and r[f"qe_nl_en_{s}"] is not None and r["qe_nl_en_tg"] is not None]

    def pick(candidates, n, taken):
        chosen, sources = [], set()
        for spread in (True, False):
            for r in candidates:
                if len(chosen) == n:
                    return chosen
                if r["id"] in taken or (spread and r["source"] in sources):
                    continue
                chosen.append(r)
                taken.add(r["id"])
                sources.add(r["source"])
        return chosen

    taken = set()
    inflate = sorted((r for r in ok if r["word_ratio_tg"] < 0.9 and r[f"word_ratio_{s}"] >= r["word_ratio_tg"] + 0.08),
                     key=lambda r: r["word_ratio_tg"])
    omit = sorted((r for r in ok if r["qe_nl_en_tg"] - r[f"qe_nl_en_{s}"] >= 1.0),
                  key=lambda r: r[f"qe_nl_en_{s}"] - r["qe_nl_en_tg"])
    worse = sorted(ok, key=lambda r: r["qe_nl_en_tg"] - r[f"qe_nl_en_{s}"])
    return ([("TranslateGemma blaast op", r) for r in pick(inflate, n_inflate, taken)]
            + [("TranslateGemma laat weg of verdraait", r) for r in pick(omit, n_omit, taken)]
            + [(f"{s} slechter dan TranslateGemma (QE nl→en)", r) for r in pick(worse, n_worse, taken)])


def table(stats, setups, labels):
    head = "| maat | " + " | ".join(labels[s] for s in setups) + " |"
    lines = [head, "|---|" + "---:|" * len(setups)]

    def row(name, f):
        lines.append(f"| {name} | " + " | ".join(f(stats[s]) for s in setups) + " |")

    row("passages", lambda x: nl_num(x["n"], 0))
    row("NL-woorden per EN-woord, mediaan", lambda x: nl_num(x["word_ratio"]["median"], 3))
    row("idem p10 / p25 / p75 / p90", lambda x: " / ".join(nl_num(x["word_ratio"][k], 2) for k in ("p10", "p25", "p75", "p90")))
    row("aandeel < 0,9 NL/EN-woorden", lambda x: pct(x["under_0_9"]))
    row("MetricX-QE nl→en, gemiddeld / mediaan", lambda x: f"{nl_num(x['qe_nl_en']['mean'])} / {nl_num(x['qe_nl_en']['median'])}")
    row("MetricX-QE en→nl, gemiddeld / mediaan", lambda x: f"{nl_num(x['qe_en_nl']['mean'])} / {nl_num(x['qe_en_nl']['median'])}")
    row("QE en→nl > 5", lambda x: pct(x["qe_over_5.0"]))
    row("QE en→nl > 7,5", lambda x: pct(x["qe_over_7.5"]))
    row("buiten de lengtegrens van R2", lambda x: pct(x["r2_length_out"]))
    row("harde controle faalt", lambda x: pct(x["hard"]))
    row("**filter raakt (hard, R2-lengte of QE > 5)**", lambda x: "**" + pct(x["filter_5.0"]) + "**")
    row("filter raakt (hard, R2-lengte of QE > 7,5)", lambda x: pct(x["filter_7.5"]))
    row("getal uit het Nederlands ontbreekt", lambda x: pct(x["numbers_missing"]))
    for k in ("not_ended", "decode_loop", "dutch_left", "structure", "newline_doubled", "len_ratio", "chatter"):
        row(f"`{k}`", lambda x, k=k: nl_num(x["checks"][k], 0))
    row("`structure`: categorieën", lambda x: ", ".join(f"{k} {v}" for k, v in sorted(
        x["structure_categories"].items(), key=lambda kv: -kv[1])) or "–")
    return lines


def report(args):
    rows = read_rows(args.compare)
    qe = {}
    for path in args.qe:
        for r in read_rows(path):
            qe.setdefault(r["id"], {}).update({k: v for k, v in r.items() if k != "id"})
    for r in rows:
        r.update(qe[r["id"]])
    setups = ["tg"] + [s for s in args.setups if s != "tg"]
    labels = {"tg": "TranslateGemma 27B (gepubliceerd)", "natural": "Gemma 4 31B natural-v1", "bare": "Gemma 4 31B bare-v1"}
    labels = {s: labels.get(s, s) for s in setups}
    setup = json.loads(Path(args.setup).read_text()) if args.setup else {}
    full = json.loads(Path(args.fullset).read_text())
    speed = throughput(args.logs)
    random_rows = [r for r in rows if r["stratum"] == "random"]
    result = {"created": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), "setup": setup, "setups": setups,
              "random": {s: setup_stats(random_rows, s) for s in setups},
              "all": {s: setup_stats(rows, s) for s in setups},
              "strata": {st: {s: setup_stats([r for r in rows if r["stratum"] == st], s) for s in setups}
                         for st in dict(STRATA) if any(r["stratum"] == st for r in rows)},
              "sources": {src: {s: setup_stats([r for r in random_rows if r["source"] == src], s) for s in setups}
                          for src in SOURCES if any(r["source"] == src for r in random_rows)},
              "against_tg": {s: {"random": against_tg(random_rows, s), "all": against_tg(rows, s)}
                             for s in setups if s != "tg"},
              "throughput": speed, "estimate": {}}
    published = [(r["qe_en_tg_nl"], r["qe_en_nl_published"]) for r in rows
                 if r["qe_en_tg_nl"] is not None and r["qe_en_nl_published"] is not None]
    deltas = [abs(a - b) for a, b in published]
    result["port_check"] = {"n": len(deltas), "max_abs": max(deltas) if deltas else None,
                            "mean_abs": avg(deltas), "over_0_1": sum(d > 0.1 for d in deltas)}
    for s in setups:
        if s in speed and speed[s]["classes"]:
            c = speed[s]["classes"]
            segs, secs = sum(v["segments"] for v in c.values()), sum(v["seconds"] for v in c.values())
            result["estimate"][s] = {"segments": segs, "decode_seconds": secs,
                                     "passages_per_chip_hour": segs / (secs / 3600) / 4,
                                     "out_tokens_per_s": sum(v["tokens_out"] for v in c.values()) / secs,
                                     **full_estimate(speed[s], rows, s, full)}
    Path(args.out).with_suffix(".json").write_text(json.dumps(result, indent=1, ensure_ascii=False))
    Path(args.out).write_text("\n".join(report_md(result, rows, labels, args)) + "\n")
    print(f"REPORT {args.out}", flush=True)


def report_md(r, rows, labels, args):
    setups = r["setups"]
    L = [f"# Terugvertaling nl→en in natuurlijker Engels: pilot op {nl_num(len(rows), 0)} passages", "",
         f"Gemaakt {r['created']} door `bt_natural.py report` (branch `bt-natural` van mbert-jax). Alle vertalingen zijn "
         "greedy, door de modellen zelf; niets is met de hand vertaald of verbeterd.", "", "## Opzet", ""]
    for k, v in r["setup"].items():
        L.append(f"- **{k}**: {v}")
    L += ["", "## Uitkomst op de uniforme steekproef (stratum `random`, representatief)", "",
          "Per bron 500 passages, uniform uit de trainrijen van de gepubliceerde datasets; samen 2.500. MetricX-24-QE: "
          "lager is beter (0–25). nl→en: bron het Nederlandse origineel, kandidaat het Engels (staat alles erin?). "
          "en→nl: bron het Engels, kandidaat het Nederlandse origineel (de maat van de huidige paarfilter). 'Filter "
          "raakt': een harde controle (`pairs.hard_fails`: niet af, lus, Nederlands over, lengteverhouding > 1,5, "
          "structuur, `newline_doubled`), buiten de lengtegrens van R2 (tekens NL/EN binnen factor 4/3 van de mediaan "
          "van de bron) of QE en→nl boven de grens. `chatter`: een inleiding of noot om de vertaling.", ""]
    L += table(r["random"], setups, labels)
    L += ["", "### Per passage tegen TranslateGemma (uniforme steekproef)", "",
          "| opzet | QE nl→en: verschil gemiddeld / aandeel beter | QE en→nl: verschil gemiddeld / aandeel beter | "
          "Engelse woorden t.o.v. TG, mediaan |", "|---|---:|---:|---:|"]
    for s, v in r["against_tg"].items():
        a = v["random"]
        L.append(f"| {labels[s]} | {nl_num(a['qe_nl_en']['mean'], 2, True)} / {pct(a['qe_nl_en']['better'])} | "
                 f"{nl_num(a['qe_en_nl']['mean'], 2, True)} / {pct(a['qe_en_nl']['better'])} | "
                 f"{nl_num(a['words_vs_tg'], 3)} |")
    L += ["", "### Per bron (uniforme steekproef)", "",
          "| bron | opzet | NL/EN-woorden mediaan (p10–p90) | QE nl→en gem. | QE en→nl gem. / mediaan | QE en→nl > 5 | "
          "R2-lengte | hard | filter raakt (> 5) |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for src, per in r["sources"].items():
        for s in setups:
            x = per[s]
            L.append(f"| {SOURCE_NAMES[src]} | {labels[s]} | {nl_num(x['word_ratio']['median'], 3)} "
                     f"({nl_num(x['word_ratio']['p10'])}–{nl_num(x['word_ratio']['p90'])}) | "
                     f"{nl_num(x['qe_nl_en']['mean'])} | {nl_num(x['qe_en_nl']['mean'])} / {nl_num(x['qe_en_nl']['median'])} | "
                     f"{pct(x['qe_over_5.0'])} | {pct(x['r2_length_out'])} | {pct(x['hard'])} | {pct(x['filter_5.0'])} |")
    L += ["", "## De moeilijke strata (bewust gekozen, niet representatief)", "",
          "`long`: 385–511 TranslateGemma-tokens; `qe_high`: TranslateGemma's paar had QE en→nl > 5; `ratio_low`: het "
          "laagste deciel NL/EN-woorden van de bron (TranslateGemma's Engels relatief lang). Elk 150–175 per bron.", "",
          "| stratum | opzet | n | NL/EN-woorden mediaan | QE nl→en gem. | QE en→nl gem. | QE en→nl > 5 | > 7,5 | "
          "R2-lengte | hard | filter raakt (> 5) |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for st, per in r["strata"].items():
        for s in setups:
            x = per[s]
            L.append(f"| `{st}` | {labels[s]} | {nl_num(x['n'], 0)} | {nl_num(x['word_ratio']['median'], 3)} | "
                     f"{nl_num(x['qe_nl_en']['mean'])} | {nl_num(x['qe_en_nl']['mean'])} | {pct(x['qe_over_5.0'])} | "
                     f"{pct(x['qe_over_7.5'])} | {pct(x['r2_length_out'])} | {pct(x['hard'])} | {pct(x['filter_5.0'])} |")
    pc = r["port_check"]
    L += ["", f"Controle van MetricX op deze VM: TranslateGemma's QE en→nl opnieuw berekend tegen de gepubliceerde "
          f"`qe_en_nl`: {nl_num(pc['n'], 0)} paren, grootste afwijking {nl_num(pc['max_abs'], 4)}, gemiddeld "
          f"{nl_num(pc['mean_abs'], 4)}, {pc['over_0_1']} boven 0,1.", "", "## Doorvoer en een volledige nieuwe BT", ""]
    L += ["| opzet | passages | decodeertijd | uitvoertokens/s per proces (4 chips) | passages per chip-uur | "
          "volledige BT: passages | procesuren (4 chips) | chip-uren | v6e-8-uren |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for s, e in r["estimate"].items():
        L.append(f"| {labels[s]} | {nl_num(e['segments'], 0)} | {nl_num(e['decode_seconds'] / 60, 1)} min | "
                 f"{nl_num(e['out_tokens_per_s'], 0)} | {nl_num(e['passages_per_chip_hour'], 0)} | "
                 f"{nl_num(e['passages'], 0)} | {nl_num(e['process_hours'], 1)} | {nl_num(e['chip_hours'], 1)} | "
                 f"{nl_num(e['v6e8_hours'], 1)} |")
    L += ["", "Volledige BT: alle gepubliceerde passages (train en validation van de vijf bronnen, plus de "
          "contextbeurten van de Handelingen), per lengteklasse hun Gemma 4-tokens maal de uitvoertokens per brontoken "
          "van de steekproef, gedeeld door de gemeten uitvoertokens per seconde van die klasse. Alleen decodeertijd; "
          "het laden van de engine (per klasse) staat hieronder.", ""]
    for s, t in r["throughput"].items():
        L.append(f"- {labels.get(s, s)}: engine geladen in {', '.join(str(x) for x in t['engine_seconds'])} s; per klasse "
                 + "; ".join(f"{c} {v['segments']} passages in {nl_num(v['seconds'], 0)} s "
                             f"({nl_num(v['tokens_out'] / max(v['seconds'], 1e-9), 0)} tok/s)"
                             for c, v in sorted(t["classes"].items())))
    L += ["", f"## Tien voorbeelden naast elkaar ({labels[args.example_setup]})", "",
          "Gekozen uit alle strata, tot 130 Nederlandse woorden: waar TranslateGemma's Engels het meest opblaast "
          "(minste NL-woorden per EN-woord, waar de nieuwe opzet dichter bij 1,05 komt), waar TranslateGemma's Engels "
          "het meest verliest op QE nl→en, en twee waar de nieuwe opzet het slechtst is tegen TranslateGemma. Eén per "
          "bron waar het kan.", ""]
    s = args.example_setup
    for k, (why, x) in enumerate(examples(rows, s), 1):
        L += [f"### {k}. {why} — {SOURCE_NAMES[x['source']]}, `{x['id']}` (stratum `{x['stratum']}`)", "",
              f"NL/EN-woorden: TG {nl_num(x['word_ratio_tg'])} · {s} {nl_num(x[f'word_ratio_{s}'])} · QE nl→en: TG "
              f"{nl_num(x['qe_nl_en_tg'])} · {s} {nl_num(x[f'qe_nl_en_{s}'])} · QE en→nl: TG {nl_num(x['qe_en_tg_nl'])} · "
              f"{s} {nl_num(x[f'qe_en_{s}_nl'])}", "",
              f"| Nederlands (origineel) | TranslateGemma 27B | {labels[s]} |", "|---|---|---|",
              f"| {cell(x['nl'])} | {cell(x['en_tg'])} | {cell(x[f'en_{s}'])} |", ""]
    return L


# ---------------------------------------------------------------------------------------------------- en->nl evaluation

def score_stats(rows):
    """{metric: {mean, median}} of MetricX (reference and QE) and COMET-22 over rows"""
    return {k: {"mean": avg(r[k] for r in rows), "median": med(r[k] for r in rows)}
            for k in ("metricx_qe", "metricx_ref", "comet22")}


def evalcompare(args):
    """Gemma 4's en->nl scores against TranslateGemma 27B's starting point: rule (a), the length of the output against
    the reference, per segment against TranslateGemma, FLAN-QE, and examples. --base: the directory of
    TranslateGemma's scores.parquet and translations.parquet (evalsets.py summary)"""
    import armeval

    evalsets = {r["id"]: r for r in read_rows(Path(args.data) / "evalsets.parquet")}
    pairs = {r["id"]: r for r in read_rows(args.pairs)}
    metricx = {r["id"]: r for r in read_rows(args.metricx)}
    comet = {r["id"]: r for r in read_rows(args.comet)}
    rows = []
    for i, p in pairs.items():
        m = metricx[i]
        rows.append({**p, "metricx_ref": m.get("ref_source_hypothesis_reference"),
                     "metricx_qe": m["qe_source_hypothesis"], "comet22": comet[i]["comet22"] if i in comet else None})
    by_set = collections.defaultdict(list)
    for r in rows:
        by_set[f"{r['set']}_all" if r["set"] == "wmt24pp" else r["set"]].append(r)
        if r["set"] == "wmt24pp" and not armeval.bad_source(evalsets[r["id"]]["meta"]):
            by_set["wmt24pp"].append(r)
    texts = {r["id"]: r for r in read_rows(Path(args.base) / "translations.parquet")}
    base_rows = [{**texts[r["id"]], **r} for r in read_rows(Path(args.base) / "scores.parquet")]
    base = collections.defaultdict(list)
    for r in base_rows:
        base[f"{r['set']}_all" if r["set"] == "wmt24pp" else r["set"]].append(r)
        if r["set"] == "wmt24pp" and not armeval.bad_source(evalsets[r["id"]]["meta"]):
            base["wmt24pp"].append(r)
    base_by_id = {r["id"]: r for r in base_rows}
    sets = ["wmt24pp", "wmt24pp_all", "flores", "ntrex", "europarl"]
    result = {"created": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), "sets": {}, "rule_a": armeval.rule_a(
        base, by_set)}
    for s in sets:
        g, b = by_set[s], base[s]
        result["sets"][s] = {"n": len(g), "gemma4": {**score_stats(g), "length": armeval.length_ratios(g),
                                                     **armeval.counts(g)},
                             "tg": {**score_stats(b), "length": armeval.length_ratios(b)}}
        paired = [(r, base_by_id[r["id"]]) for r in g]
        result["sets"][s]["per_segment"] = {
            "qe_better": share(paired, lambda x: x[0]["metricx_qe"] < x[1]["metricx_qe"]),
            "qe_worse_by_1": share(paired, lambda x: x[0]["metricx_qe"] > x[1]["metricx_qe"] + 1),
            "words_vs_tg": med(words(x[0]["hypothesis"] or "") / max(1, words(x[1]["hypothesis"] or "")) for x in paired)}
    flan = read_rows(args.flan_pairs)
    flan_qe = {r["id"]: r["qe_source_hypothesis"] for r in read_rows(args.flan_metricx)}
    result["flan"] = {"n": len(flan), "qe_mean": avg(flan_qe[r["id"]] for r in flan),
                      "qe_median": med(flan_qe[r["id"]] for r in flan), **armeval.counts(flan)}
    result["throughput"] = throughput(args.logs)
    label = getattr(args, "label", None) or "Gemma 4 31B-it"
    result["label"] = label
    Path(args.out).with_suffix(".json").write_text(json.dumps(result, indent=1, ensure_ascii=False, default=str))
    L = [f"# {label} als en→nl-vertaler (natural-v1) op de evaluatiesets van de vertaler", "",
         f"Gemaakt {result['created']} door `bt_natural.py evalcompare`. Greedy; MetricX-24 hybrid "
         "XXL (JAX-port), COMET-22 op de CPU; TranslateGemma 27B = het uitgangspunt van 10 nov (`baseline-tg27b`). "
         "Per cel gemiddelde / mediaan.", "",
         "| set | n | MetricX-QE Gemma 4 | TG | verschil | MetricX-ref Gemma 4 | TG | COMET-22 Gemma 4 | TG | "
         "uitvoer/referentie (woorden, mediaan) Gemma 4 | TG | < 0,7 × ref Gemma 4 | TG | niet af / lus / Engels over / leeg |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for s in sets:
        x = result["sets"][s]
        g, b = x["gemma4"], x["tg"]
        gl, bl = g["length"] or {}, b["length"] or {}
        L.append(f"| {s} | {x['n']} | {nl_num(g['metricx_qe']['mean'], 3)} / {nl_num(g['metricx_qe']['median'], 3)} | "
                 f"{nl_num(b['metricx_qe']['mean'], 3)} | {nl_num(g['metricx_qe']['mean'] - b['metricx_qe']['mean'], 3, True)} | "
                 f"{nl_num(g['metricx_ref']['mean'], 3)} | {nl_num(b['metricx_ref']['mean'], 3)} | "
                 f"{nl_num(g['comet22']['mean'], 3)} | {nl_num(b['comet22']['mean'], 3)} | "
                 f"{nl_num(gl.get('out_per_ref'), 3)} | {nl_num(bl.get('out_per_ref'), 3)} | "
                 f"{pct(gl.get('under_0_7_of_ref'))} | {pct(bl.get('under_0_7_of_ref'))} | "
                 f"{g['not_ended']} / {g['decode_loop']} / {g['english_left']} / {g['empty']} |")
    ra = result["rule_a"]
    L += ["", "Regel (a), MetricX-24-QE tegen TG + 0,20: " + "; ".join(
        f"{k} {nl_num(v['arm'], 3)} tegen grens {nl_num(v['limit'], 3)} ({'gehaald' if v['holds'] else 'niet gehaald'})"
        for k, v in ra["sets"].items()) + f". **{'Gehaald' if ra['holds'] else 'Niet gehaald'}.**", "",
          "Per segment tegen TG: " + "; ".join(
              f"{s} QE beter {pct(result['sets'][s]['per_segment']['qe_better'])}, > 1 slechter "
              f"{pct(result['sets'][s]['per_segment']['qe_worse_by_1'])}, woorden t.o.v. TG "
              f"{nl_num(result['sets'][s]['per_segment']['words_vs_tg'], 3)}" for s in ("wmt24pp", "flores", "ntrex", "europarl")),
          "", f"FLAN (1.000 rijen van de classifier, alleen QE): MetricX-QE {nl_num(result['flan']['qe_mean'])} / "
          f"{nl_num(result['flan']['qe_median'])}; niet af {result['flan']['not_ended']}, lus {result['flan']['decode_loop']}, "
          f"Engels over {result['flan']['english_left']}, leeg {result['flan']['empty']}.", ""]
    for s, t in result["throughput"].items():
        L.append(f"Doorvoer {s}: engine {', '.join(str(x) for x in t['engine_seconds'])} s; " + "; ".join(
            f"{c} {v['segments']} segmenten in {nl_num(v['seconds'], 0)} s ({nl_num(v['tokens_out'] / max(v['seconds'], 1e-9), 0)} tok/s)"
            for c, v in sorted(t["classes"].items())))
    L += ["", "## Tien voorbeelden: bron, TranslateGemma 27B, Gemma 4, referentie", "",
          "Gekozen uit WMT24++ (960), FLORES en NTREX, tot 90 bronwoorden: de vijf segmenten waar Gemma 4 op MetricX-QE "
          "het meest wint, en de vijf waar het het meest verliest, één per set waar het kan.", ""]
    pool = [r for s in ("wmt24pp", "flores", "ntrex") for r in by_set[s] if words(r["source"]) <= 90]
    gain = sorted(pool, key=lambda r: r["metricx_qe"] - base_by_id[r["id"]]["metricx_qe"])
    picked = []
    for group, ordered in (("Gemma 4 beter", gain), ("Gemma 4 slechter", list(reversed(gain)))):
        sets_seen, n = set(), 0
        for spread in (True, False):
            for r in ordered:
                if n == 5:
                    break
                if r["id"] in {p[1]["id"] for p in picked} or (spread and r["set"] in sets_seen):
                    continue
                picked.append((group, r))
                sets_seen.add(r["set"])
                n += 1
    for k, (group, r) in enumerate(picked, 1):
        b = base_by_id[r["id"]]
        L += [f"### {k}. {group} — {r['set']} `{r['id']}`", "",
              f"MetricX-QE: TG {nl_num(b['metricx_qe'])} · Gemma 4 {nl_num(r['metricx_qe'])} · MetricX-ref: TG "
              f"{nl_num(b['metricx_ref'])} · Gemma 4 {nl_num(r['metricx_ref'])}", "",
              f"| bron (Engels) | TranslateGemma 27B | {label} natural-v1 | referentie |", "|---|---|---|---|",
              f"| {cell(r['source'])} | {cell(b['hypothesis'])} | {cell(r['hypothesis'])} | {cell(r['reference'])} |", ""]
    Path(args.out).write_text("\n".join(L) + "\n")
    print(f"EVALCOMPARE {args.out}", flush=True)


def refqe_groups(evalsets):
    """{group: [ids]} of the four sets: wmt24pp (the 960 without is_bad_source, rule (a)), wmt24pp_all (998), flores,
    ntrex, europarl"""
    import armeval

    groups = collections.defaultdict(list)
    for r in evalsets:
        if r["set"] == "wmt24pp":
            groups["wmt24pp_all"].append(r["id"])
            if not armeval.bad_source(r["meta"]):
                groups["wmt24pp"].append(r["id"])
        else:
            groups[r["set"]].append(r["id"])
    return groups


def refqe(args):
    """The calibration of rule (a): MetricX-24-QE en->nl of the human reference (source the English, candidate the Dutch
    reference, as an arm's translation is scored) against TranslateGemma 27B's QE on the same segments"""
    evalsets = read_rows(args.evalsets)
    qe = {r["id"]: r["qe_source_reference"] for r in read_rows(args.qe)}
    base = {r["id"]: r["metricx_qe"] for r in read_rows(Path(args.base) / "scores.parquet")}
    out = {}
    for group, ids in refqe_groups(evalsets).items():
        ref, tg = [qe[i] for i in ids], [base[i] for i in ids]
        if None in ref or None in tg:
            raise SystemExit(f"{group}: a segment without a score")
        delta = statistics.fmean(ref) - statistics.fmean(tg)
        out[group] = {"n": len(ids), "ref_mean": statistics.fmean(ref), "ref_median": statistics.median(ref),
                      "tg_mean": statistics.fmean(tg), "tg_median": statistics.median(tg), "delta": delta,
                      "within_0_20": delta <= 0.20 + 1e-9,
                      "ref_worse_share": sum(a > b for a, b in zip(ref, tg)) / len(ids)}
    order = ["wmt24pp", "wmt24pp_all", "flores", "ntrex", "europarl"]
    L = ["| set | n | MetricX-QE menselijke referentie (gem. / mediaan) | TG 27B (gem. / mediaan) | verschil (gem.) | "
         "binnen +0,20? | segmenten waar de referentie slechter scoort dan TG |", "|---|---:|---:|---:|---:|---|---:|"]
    for g in order:
        x = out[g]
        L.append(f"| {g} | {x['n']} | {nl_num(x['ref_mean'], 3)} / {nl_num(x['ref_median'], 3)} | "
                 f"{nl_num(x['tg_mean'], 3)} / {nl_num(x['tg_median'], 3)} | {nl_num(x['delta'], 3, True)} | "
                 f"{'ja' if x['within_0_20'] else 'nee'} | {pct(x['ref_worse_share'])} |")
    Path(args.out).with_suffix(".json").write_text(json.dumps(out, indent=1))
    Path(args.out).write_text("\n".join(L) + "\n")
    print("\n".join(L), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("fetch")
    p.add_argument("--out", required=True)
    p = sub.add_parser("sample")
    p.add_argument("--data", required=True)
    p.add_argument("--tokenizer", required=True, help="Gemma 4's tokenizer directory")
    p.add_argument("--out", required=True)
    p.add_argument("--no-fullset", action="store_true", help="Skip the token counts of the whole published BT")
    p = sub.add_parser("show")
    p.add_argument("--segments", required=True)
    p.add_argument("--shards", nargs="+", required=True, help="NAME=PROMPT=DIR of translate.py's shards")
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--source", default="nl")
    p.add_argument("--target", default="en")
    p = sub.add_parser("merge")
    p.add_argument("--sample", required=True)
    p.add_argument("--shards", nargs="*", default=[],
                   help="SETUP=DIR of translate.py's shards (none: TranslateGemma's English only)")
    p.add_argument("--out", required=True)
    p = sub.add_parser("report")
    p.add_argument("--compare", required=True)
    p.add_argument("--qe", nargs="+", required=True)
    p.add_argument("--setups", nargs="+", default=["natural", "bare"])
    p.add_argument("--example-setup", default="natural")
    p.add_argument("--logs", required=True)
    p.add_argument("--fullset", required=True)
    p.add_argument("--setup", default=None, help="JSON file: {name: description} of the setup")
    p.add_argument("--out", required=True)
    p = sub.add_parser("evalsegments")
    p.add_argument("--four", required=True, help="evalsets.parquet of the four sets (evalsets.py fetch)")
    p.add_argument("--x", required=True, help="evalsets.parquet of armeval.py data --kind x (its set flan)")
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--model", default=MODEL, help="The model of translate_models.yml whose classes and chat turn count")
    p.add_argument("--blind", action="store_true", help="Also blind/: the 100 passages of x's set blind")
    p.add_argument("--out", required=True)
    p = sub.add_parser("evalcompare")
    for name in ("--data", "--pairs", "--metricx", "--comet", "--flan-pairs", "--flan-metricx", "--base",
                 "--logs", "--out"):
        p.add_argument(name, required=True)
    p.add_argument("--label", default=None, help="The model's name in the report (default: Gemma 4 31B-it)")
    p = sub.add_parser("refqe")
    p.add_argument("--evalsets", required=True, help="evalsets.parquet of the four sets")
    p.add_argument("--qe", required=True, help="metricx_qe.py score with --pair source:reference")
    p.add_argument("--base", required=True, help="TranslateGemma 27B's evalsets.py summary directory (scores.parquet)")
    p.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    {"refqe": refqe, "fetch": fetch, "sample": sample, "show": show, "merge": merge, "report": report, "evalsegments": evalsegments,
     "evalcompare": evalcompare}[args.command](args)


if __name__ == "__main__":
    main()
