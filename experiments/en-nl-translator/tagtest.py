#!/usr/bin/env python3
"""The tag test of the fine-tune (protocol.md, "Proef met de tag"): TranslateGemma 12B with a LoRA adapter per form of
arm N's tag (sft_build.py: token, instruction), each translating FLAN's evaluation rows and WMT24++ segments with and
without its tag; the base model beside them.

    python tagtest.py segments --flan-eval eval-rows.parquet --evalsets evalsets.parquet --tokenizer TOK --out DIR
    python tagtest.py collect --items DIR/items.parquet --shards SHARDS --out pairs.parquet
    python tagtest.py report --pairs pairs.parquet --qe qe.parquet --out tag-test.md [--json tag-test.json]

segments: N_FLAN rows of FLAN's evaluation rows (set classifier; the whole instruction, `inputs`) and N_WMT segments of
WMT24++ en->nl_NL without is_bad_source, drawn with a fixed seed; segments.parquet (id, text, tokens) for translate.py
and items.parquet (id, set, task, source, reference).
collect: the translations of every variant (SHARDS/<variant>/<class>/shard-*.parquet) beside their item, with the
counts below per translation; pairs.parquet has the columns source and hypothesis for metricx_qe.py (QE).
report: per variant and mode the counts and MetricX-QE, and examples, as Markdown.

What is counted per translation (heuristics, named so in the report):
- tag: the tag token (<unused...>) or the tag sentence (English, or a Dutch rendering of its key words) in the output;
- intro: the output opens with a sentence about the translation ("Hieronder volgt", "Hier is de vertaling", "Vertaling:")
  where the source does not open that way;
- longer / shorter: the output is more than LONGER x or less than SHORTER x as long (characters) as the base model's
  translation of the same segment and differs from it (chrF < CHRF_DIFFERENT): a sign of an answer added to or put in
  place of the translation, or of a part left out; read by hand in the report's list;
- loop: publish.decode_loop; not ended: the decode limit was hit; english_left: publish.english_left.
"""

import argparse
import collections
import json
import random
import re
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for _path in (HERE.parent / "flan-nl", HERE.parents[1] / "scripts", HERE):
    sys.path.insert(0, str(_path))

import publish  # noqa: E402

N_FLAN = 300
N_WMT = 300
SEED = "en-nl-sft/tagtest/0"
VARIANTS = {  # variant: (adapter, tag in the prompt)
    "base": (None, False),
    "token_tag": ("token", True), "token_plain": ("token", False),
    "instruction_tag": ("instruction", True), "instruction_plain": ("instruction", False),
}
TAG_TEXT = re.compile(r"<unused\d+>|<BT>|original[- ]sounding|natural, original|"
                      r"(?:natuurlijk|origineel|oorspronkelijk)(?:,)? (?:en )?(?:origineel|oorspronkelijk|natuurlijk) "
                      r"klinkend|klinkend nederlands", re.I)
INTRO = re.compile(r"^\W*(?:hieronder (?:volgt|staat|vind|de)|hier (?:is|volgt|staat|de)|de (?:nederlandse )?vertaling"
                   r"|vertaling\s*:|vertaald\s*:|nederlandse vertaling|natuurlijk[,!.]|zeker[,!.]|uiteraard|graag[,!.]"
                   r"|sure\b|here is|here's|translation\s*:|okay|oké)", re.I)
SOURCE_INTRO = re.compile(r"^\W*(?:here|below|sure|of course|translation|okay|ok\b|certainly)", re.I)
LONGER, SHORTER, CHRF_DIFFERENT = 1.4, 0.6, 50.0


def segments(args):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    flan = [r for r in pq.read_table(args.flan_eval).to_pylist() if r["set"] == "classifier"]
    wmt = [r for r in pq.read_table(args.evalsets).to_pylist()
           if r["set"] == "wmt24pp" and not json.loads(r["meta"]).get("is_bad_source")]
    flan = random.Random(f"{args.seed}/flan").sample(flan, args.n_flan)
    wmt = random.Random(f"{args.seed}/wmt").sample(wmt, args.n_wmt)
    items = [{"id": f"flan:{r['host']}-{r['position']}", "set": "flan", "task": r["source"],
              "source": r["inputs"].strip(), "reference": None} for r in flan]
    items += [{"id": r["id"], "set": "wmt24pp", "task": json.loads(r["meta"]).get("domain"),
               "source": r["source"].strip(), "reference": r["reference"]} for r in wmt]
    if len({i["id"] for i in items}) != len(items):
        raise SystemExit("duplicate ids among the items")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tokens = [len(tokenizer(i["source"], add_special_tokens=False)["input_ids"]) for i in items]
    pq.write_table(pa.table({"id": [i["id"] for i in items], "text": [i["source"] for i in items], "tokens": tokens}),
                   out / "segments.parquet")
    pq.write_table(pa.Table.from_pylist(items), out / "items.parquet")
    print(json.dumps({"items": len(items), "flan": len(flan), "wmt24pp": len(wmt), "max_tokens": max(tokens)}))


def chrf(a, b):
    import sacrebleu

    return sacrebleu.sentence_chrf(a, [b]).score


def flags(item, text, ended, base):
    """The counts of one translation (base: the base model's translation of the item, or None for the base)."""
    out = {"tag": bool(TAG_TEXT.search(text)),
           "intro": bool(INTRO.search(text)) and not SOURCE_INTRO.search(item["source"]),
           "loop": publish.decode_loop(text, item["source"]), "not_ended": not ended,
           "english_left": publish.english_left(text, item["source"]), "longer": False, "shorter": False,
           "empty": not text.strip()}
    if base is not None and base.strip():
        ratio = len(text) / len(base)
        different = chrf(text, base) < CHRF_DIFFERENT
        out["longer"] = ratio > LONGER and different
        out["shorter"] = ratio < SHORTER and different
        out["ratio_to_base"] = round(ratio, 3)
    return out


def collect(args):
    import pyarrow as pa
    import pyarrow.parquet as pq

    items = {r["id"]: r for r in pq.read_table(args.items).to_pylist()}
    translations = {}
    for variant in VARIANTS:
        shards = sorted(Path(args.shards, variant).glob("*/shard-*.parquet"))
        rows = [r for s in shards for r in pq.read_table(s).to_pylist()]
        got = {r["id"]: r for r in rows}
        if set(got) != set(items):
            raise SystemExit(f"{variant}: {len(set(items) - set(got))} items without a translation, "
                             f"{len(set(got) - set(items))} unknown ids")
        translations[variant] = got
    out = []
    for variant in VARIANTS:
        for i, item in items.items():
            t = translations[variant][i]
            base = translations["base"][i]["text_nl"] if variant != "base" else None
            out.append({"variant": variant, "id": i, "set": item["set"], "task": item["task"],
                        "source": item["source"], "hypothesis": t["text_nl"], "ended": t["ended"],
                        "tokens_out": t["tokens_out"], **flags(item, t["text_nl"], t["ended"], base)})
    pq.write_table(pa.Table.from_pylist(out), args.out)
    print(json.dumps({"rows": len(out), "variants": len(VARIANTS)}))


COUNTS = ["tag", "intro", "longer", "shorter", "loop", "not_ended", "english_left", "empty"]
LABELS = {"base": "basis 12B", "token_tag": "token, met tag", "token_plain": "token, zonder tag",
          "instruction_tag": "instructie, met tag", "instruction_plain": "instructie, zonder tag"}


def summarize(rows):
    table = {}
    for variant in VARIANTS:
        for s in ("flan", "wmt24pp", "all"):
            part = [r for r in rows if r["variant"] == variant and (s == "all" or r["set"] == s)]
            qe = [r["qe"] for r in part]
            table[f"{variant}/{s}"] = {"n": len(part), **{c: sum(bool(r[c]) for r in part) for c in COUNTS},
                                       "qe_mean": round(statistics.mean(qe), 3), "qe_median": round(
                                           statistics.median(qe), 3)}
    return table


def report(args):
    import pyarrow.parquet as pq

    rows = pq.read_table(args.pairs).to_pylist()
    qe = pq.read_table(args.qe).to_pylist()
    if len(qe) != len(rows):
        raise SystemExit(f"{len(rows)} pairs, {len(qe)} QE scores")
    for r, q in zip(rows, qe):
        if q["id"] != r["id"] or q[args.qe_column] is None:
            raise SystemExit(f"no QE score in the order of the pairs at {r['variant']} {r['id']}")
        r["qe"] = q[args.qe_column]
    table = summarize(rows)
    if args.json:
        Path(args.json).write_text(json.dumps(table, indent=1) + "\n", encoding="utf-8")
    lines = [f"# {args.title}", "", args.intro or "", "",
             "| variant | set | n | tag letterlijk | inleiding | langer | korter | lus | onafgemaakt | Engels over | "
             "QE gem. | QE med. |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for key, t in table.items():
        variant, s = key.split("/")
        lines.append(f"| {LABELS[variant]} | {s} | {t['n']} | {t['tag']} | {t['intro']} | {t['longer']} | "
                     f"{t['shorter']} | {t['loop']} | {t['not_ended']} | {t['english_left']} | {t['qe_mean']:.3f} | "
                     f"{t['qe_median']:.3f} |")
    by_item = collections.defaultdict(dict)
    for r in rows:
        by_item[r["id"]][r["variant"]] = r
    for adapter in ("token", "instruction"):
        lines += ["", f"## Voorbeelden: adapter met de tag als {'token' if adapter == 'token' else 'instructiezin'}", ""]
        flagged = [i for i, v in by_item.items()
                   if any(v[f"{adapter}_{m}"][c] for m in ("tag", "plain") for c in COUNTS if c != "english_left")]
        rng = random.Random(f"{SEED}/examples/{adapter}")
        rest = [i for i in sorted(by_item) if i not in flagged]
        chosen = sorted(flagged, key=lambda i: (by_item[i]["base"]["set"], i))[:args.examples // 2]
        chosen += rng.sample(rest, args.examples - len(chosen)) if len(rest) >= args.examples - len(chosen) else rest
        for i in chosen[:args.examples]:
            v = by_item[i]
            marks = {m: [c for c in COUNTS if v[f"{adapter}_{m}"][c]] for m in ("tag", "plain")}
            lines += [f"### {i} ({v['base']['set']}, {v['base']['task']})", "",
                      f"Bron:\n\n{quote(v['base']['source'], args.max_chars)}", "",
                      f"Basis 12B (QE {v['base']['qe']:.2f}):\n\n{quote(v['base']['hypothesis'], args.max_chars)}", ""]
            for m, label in (("tag", "met tag"), ("plain", "zonder tag")):
                r = v[f"{adapter}_{m}"]
                lines += [f"Adapter, {label} (QE {r['qe']:.2f}; vlaggen: {', '.join(marks[m]) or 'geen'}):\n\n"
                          f"{quote(r['hypothesis'], args.max_chars)}", ""]
    if args.flagged:
        flagged_rows = [r for r in rows if r["variant"] != "base" and any(r[c] for c in COUNTS if c != "english_left")]
        Path(args.flagged).write_text("\n\n".join(
            f"## {r['variant']} {r['id']} ({r['set']}): {', '.join(c for c in COUNTS if r[c])}\n\nBron:\n\n"
            f"{quote(r['source'], 1500)}\n\nBasis:\n\n{quote(by_item[r['id']]['base']['hypothesis'], 1500)}\n\n"
            f"Variant:\n\n{quote(r['hypothesis'], 1500)}" for r in flagged_rows) + "\n", encoding="utf-8")
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")


def quote(text, limit):
    text = text if len(text) <= limit else text[:limit] + " […]"
    return "\n".join("> " + line for line in text.split("\n"))


def main(argv=None):
    parser = argparse.ArgumentParser(description="The tag test of the fine-tune")
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("segments")
    s.add_argument("--flan-eval", required=True)
    s.add_argument("--evalsets", required=True)
    s.add_argument("--tokenizer", required=True)
    s.add_argument("--n-flan", type=int, default=N_FLAN)
    s.add_argument("--n-wmt", type=int, default=N_WMT)
    s.add_argument("--seed", default=SEED)
    s.add_argument("--out", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--items", required=True)
    c.add_argument("--shards", required=True, help="<variant>/<class>/shard-*.parquet below it")
    c.add_argument("--out", required=True)
    r = sub.add_parser("report")
    r.add_argument("--pairs", required=True)
    r.add_argument("--qe", required=True, help="metricx_qe.py's scores of the pairs, in their order")
    r.add_argument("--qe-column", default="qe_source_hypothesis")
    r.add_argument("--out", required=True)
    r.add_argument("--json", default=None)
    r.add_argument("--flagged", default=None, help="Markdown of every flagged translation, to read by hand")
    r.add_argument("--examples", type=int, default=10)
    r.add_argument("--max-chars", type=int, default=700)
    r.add_argument("--title", default="Proef met de tag op TranslateGemma 12B")
    r.add_argument("--intro", default="")
    args = parser.parse_args(argv)
    {"segments": segments, "collect": collect, "report": report}[args.command](args)


if __name__ == "__main__":
    main()
