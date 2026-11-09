#!/usr/bin/env python3
"""Steps 1-2 of the en-nl translator (hammond mbert/designs/en-nl-translator.md, §3) around translate.py: the English
segments for the back-translation, and the measures of step 2 for every triple (original Dutch, synthetic English,
back-translation into Dutch).

    python pairs.py back-segments --select <dir> --nl-en <shards> --tokenizer <dir> --out <dir>
    python pairs.py measures --select <dir> --nl-en <shards> --en-nl <shards> --out <dir>/pairs.parquet

<shards> is a local copy of a run's shards (w<k>/<class>/shard-*.parquet). Measures per pair (no threshold is fixed
here: Yeb sets them on the pilot):
- ended: the nl->en translation stopped by itself (not at the decode limit)
- decode_loop: publish.decode_loop of FLAN-nl, the English against its Dutch source
- dutch_left: the mirror of publish.english_left, a stretch of the Dutch source left in the English
- len_ratio: the longer of the two texts' characters over the shorter's
- sentences_nl, sentences_en: sentence counts (select_passages.split_sentences)
- numbers_missing: numbers of the Dutch that the English lacks (missing_numbers: separators, times, number words up
  to twenty and months in dates aside)
- structure: structure.differences of the Dutch original and its English (formatting kept: [], or the categories that
  differ; ["newline_doubled"] for TranslateGemma's empty line where the original has a single newline)
- chrf: chrF (sacrebleu, chrF2) of the back-translation against the original
- the back-translation's own ended and decode_loop, for reading
MetricX-24 QE comes from metricx_qe.py.
"""

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# select_passages.py beside this file; publish.py (and what it imports) of FLAN-nl in the repo, or beside this file on
# a VM
for _path in (HERE.parent / "flan-nl", HERE.parents[1] / "scripts", HERE):
    sys.path.insert(0, str(_path))

import publish  # noqa: E402
import select_passages as sp  # noqa: E402
import structure  # noqa: E402

MAX_BACK_TOKENS = 512  # translate.py's longest class
ENGLISH_ABBREVIATIONS = sp.ABBREVIATIONS | frozenset("mrs ms e.g i.e no vs u.s u.k para sec fig approx pp".split())
NUMBER = re.compile(r"\d+(?:[.,\u00a0]\d{3})*(?:[.,]\d+)?")  # a plain space is no thousands separator: "30 2015"
# A number of the Dutch that the English writes as a word, or as a month in a date
NUMBER_WORDS = {w: str(k) for k, w in enumerate("""zero one two three four five six seven eight nine ten eleven twelve
    thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty""".split())}
MONTHS = frozenset("""january february march april may june july august september october november december jan feb mar
    apr jun jul aug sep sept oct nov dec""".split())
NUMERIC_DATE = re.compile(r"\b\d{1,2}[-/](\d{1,2})[-/]\d{2,4}\b")


def source_left(translation, source, source_words, target_words, shared=publish.SHARED_WORDS,
                least=publish.ENGLISH_LEFT):
    """publish.english_left with the languages as parameters: a run of words of the source in the translation, none a
    function word of the target language (target_words), with at least least[0] function words of the source language
    (source_words) and least[1] other lowercase words of 3 letters or more that are not in shared"""
    src = {w.lower() for w in publish.LETTERS.findall(publish.NOT_PROSE.sub(" ", source))}
    function = other = 0
    for w in publish.LETTERS.findall(publish.NOT_PROSE.sub(" ", translation)):
        if not (w.islower() or w.istitle()):
            continue
        lw = w.lower()
        if lw in target_words or lw not in src:
            function = other = 0
        elif lw in source_words:
            function += 1
        elif w.islower() and len(w) >= 3 and lw not in shared:
            other += 1
        if function >= least[0] and other >= least[1]:
            return True
    return False


def dutch_left(en, nl):
    """The English translation en keeps a stretch of its Dutch source nl"""
    return source_left(en, nl, publish.DUTCH_WORDS, publish.ENGLISH_WORDS)


def numbers(text):
    return {re.sub(r"\D", "", m) for m in NUMBER.findall(text)}


def _value(digits):
    return digits.lstrip("0") or "0"


def missing_numbers(nl, en):
    """Numbers of the Dutch that the English lacks. Present: the same digits (separators and leading zeros aside:
    1.000,5 = 1,000.5), each group of a time or decimal (01.30 = 1:30), a number word up to twenty (2 = two), or the
    month of a numeric date when the English names a month (30-04-2015 = April 30, 2015)"""
    words = {w.lower() for w in re.findall(r"[A-Za-z]+", en)}
    present = {_value(n) for n in numbers(en)} | {v for w, v in NUMBER_WORDS.items() if w in words}
    months = {_value(m) for m in NUMERIC_DATE.findall(nl)} if words & MONTHS else set()
    missing = set()
    for raw in NUMBER.findall(nl):
        value = _value(re.sub(r"\D", "", raw))
        groups = [_value(g) for g in re.split(r"[.,:\s]", raw) if g]
        if value in present or (len(groups) > 1 and all(g in present for g in groups)) or value in months:
            continue
        missing.add(value)
    return sorted(missing)


def sentences(text, abbreviations):
    return len(sp.split_sentences(text, abbreviations))


def read_shards(root, column):
    """{id: (text, ended, tokens_out)} of every shard under root"""
    import pyarrow.parquet as pq

    out = {}
    paths = sorted(Path(root).glob("w*/*/shard-*.parquet"))
    if not paths:
        raise SystemExit(f"no shards under {root}")
    for path in paths:
        t = pq.read_table(path).to_pydict()
        for i, text, ended, n in zip(t["id"], t[column], t["ended"], t["tokens_out"]):
            if i in out:
                raise SystemExit(f"{path}: segment {i} translated twice")
            out[i] = (text, ended, n)
    return out


def read_passages(select):
    import pyarrow.parquet as pq

    return pq.read_table(Path(select) / "passages.parquet").to_pylist()


def back_segments(args):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer

    passages = read_passages(args.select)
    english = read_shards(args.nl_en, "text_en")
    missing = [p["id"] for p in passages if p["id"] not in english]
    if missing:
        raise SystemExit(f"{len(missing)} passages have no English, e.g. {missing[:5]}")
    tok = Tokenizer.from_file(str(Path(args.tokenizer) / "tokenizer.json"))
    rows, skipped = [], {"not_ended": [], "empty": [], "too_long": []}
    for p in passages:
        text, ended, _ = english[p["id"]]
        if not ended:
            skipped["not_ended"].append(p["id"])
        elif not text.strip():
            skipped["empty"].append(p["id"])
        else:
            n = len(tok.encode(text, add_special_tokens=False).ids)
            if n > MAX_BACK_TOKENS:
                skipped["too_long"].append(p["id"])
            else:
                rows.append({"id": p["id"], "text": text, "tokens": n})
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for k, part in enumerate(sp.balanced_workers(rows, args.workers)):
        pq.write_table(pa.table({c: [r[c] for r in part] for c in ("id", "text", "tokens")}),
                       out / f"segments-w{k}.parquet")
    (out / "back_segments.json").write_text(json.dumps({"segments": len(rows), "skipped": skipped}, indent=1))
    print(f"BACK {len(rows)} segments, skipped {({k: len(v) for k, v in skipped.items()})}")


def measure(nl, en, back):
    """The measures of one triple; back is None when the English was not back-translated"""
    import sacrebleu

    sentences_nl, sentences_en = sentences(nl, sp.ABBREVIATIONS), sentences(en, ENGLISH_ABBREVIATIONS)
    missing = missing_numbers(nl, en)
    row = {
        "decode_loop": bool(en) and publish.decode_loop(en, nl),
        "dutch_left": dutch_left(en, nl),
        "len_ratio": round(max(len(nl), len(en)) / max(min(len(nl), len(en)), 1), 4),
        "sentences_nl": sentences_nl, "sentences_en": sentences_en,
        "numbers_missing": missing,
        "structure": structure.differences(nl, en),
        "chrf": None, "back_decode_loop": None, "back_structure": None,
    }
    if back is not None:
        row["chrf"] = round(sacrebleu.sentence_chrf(back, [nl]).score, 3)
        row["back_decode_loop"] = bool(back) and publish.decode_loop(back, en)
        row["back_structure"] = structure.differences(en, back)
    return row


def measures(args):
    import pyarrow as pa
    import pyarrow.parquet as pq

    passages = read_passages(args.select)
    english = read_shards(args.nl_en, "text_en")
    back = read_shards(args.en_nl, "text_nl")
    rows = []
    for p in passages:
        en, ended, en_tokens = english[p["id"]]
        b = back.get(p["id"])
        row = {"id": p["id"], "pid": p["pid"], "source": p["source"], "half": p["half"], "nl": p["text"],
               "tokens_nl": p["tokens"], "idiom_score": p["idiom_score"], "en": en, "ended": ended,
               "tokens_en": en_tokens - int(ended), "back": b[0] if b else None,
               "back_ended": b[1] if b else None}
        row.update(measure(p["text"], en, b[0] if b else None))
        rows.append(row)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({c: [r[c] for r in rows] for c in rows[0]}), out, compression="zstd")
    print(f"MEASURES {len(rows)} pairs, {sum(r['back'] is not None for r in rows)} back-translated -> {out}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("back-segments", help="The English of the nl->en run as segments for the en->nl run")
    p.add_argument("--select", required=True)
    p.add_argument("--nl-en", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=4)
    p = sub.add_parser("measures", help="The measures of step 2 per pair")
    p.add_argument("--select", required=True)
    p.add_argument("--nl-en", required=True)
    p.add_argument("--en-nl", required=True)
    p.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    {"back-segments": back_segments, "measures": measures}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
