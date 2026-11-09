#!/usr/bin/env python3
"""The side-by-side of the LoRA smoke test as Markdown: per compared passage (sft_data.py's compare.parquet) the
English source, for a pilot passage its original Dutch, and the translations of the base model and the adapters
(run_compare.sh's shards), with a table of what differs. The texts are the data's and the models'; nothing is edited.

    python smoke_report.py --data DIR --compare DIR --out passages.md
--compare is a local copy of gs://.../smoke-lora/compare (<variant>/<class>/shard-*.parquet).
"""

import argparse
import json
from pathlib import Path

VARIANTS = [("base", "basismodel, zonder tag"), ("N_tag", "LoRA arm N, met tag"),
            ("N_plain", "LoRA arm N, zonder tag"), ("B_plain", "LoRA arm B, zonder tag")]


def read_variant(directory):
    """{id: row} of a variant's shards over all its length classes; refuses an id twice."""
    import pyarrow.parquet as pq

    rows = {}
    for shard in sorted(Path(directory).glob("*/shard-*.parquet")):
        for row in pq.read_table(shard).to_pylist():
            if row["id"] in rows:
                raise SystemExit(f"{shard}: id {row['id']} twice")
            rows[row["id"]] = row
    return rows


def quote(text):
    return "\n".join("> " + line if line.strip() else ">" for line in text.split("\n"))


def render(compare, outputs):
    """The Markdown of the comparison: a table of counts, then the passages."""
    lines = ["| variant | passages | not ended | identical to base | output tokens |", "|---|---|---|---|---|"]
    for key, label in VARIANTS:
        rows = outputs[key]
        same = sum(rows[c["id"]]["text_nl"] == outputs["base"][c["id"]]["text_nl"] for c in compare)
        lines.append(f"| {label} | {len(rows)} | {sum(not r['ended'] for r in rows.values())} | "
                     f"{same if key != 'base' else '-'} | {sum(r['tokens_out'] for r in rows.values())} |")
    for k, c in enumerate(compare, 1):
        lines += ["", f"### {k}. {c['origin']} `{c['ref']}` ({c['task']})", "", "**Engels (bron)**", "",
                  quote(c["en"])]
        if c["nl_original"]:
            lines += ["", "**Nederlands origineel (de doeltekst van de pilot)**", "", quote(c["nl_original"])]
        base = outputs["base"][c["id"]]["text_nl"]
        for key, label in VARIANTS:
            row = outputs[key][c["id"]]
            notes = [] if row["ended"] else ["niet afgelopen"]
            if key != "base" and row["text_nl"] == base:
                notes.append("gelijk aan het basismodel")
            lines += ["", f"**{label}**" + (f" ({', '.join(notes)})" if notes else ""), "", quote(row["text_nl"])]
    return "\n".join(lines) + "\n"


def main(argv=None):
    import pyarrow.parquet as pq

    parser = argparse.ArgumentParser(description="Side-by-side of the LoRA smoke test")
    parser.add_argument("--data", required=True, help="sft_data.py's output directory (compare.parquet)")
    parser.add_argument("--compare", required=True, help="Local copy of the compare/ directory of the bucket")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    compare = pq.read_table(Path(args.data) / "compare.parquet").to_pylist()
    outputs = {key: read_variant(Path(args.compare) / key) for key, _ in VARIANTS}
    for key, rows in outputs.items():
        missing = sorted({c["id"] for c in compare} - set(rows))
        if missing:
            raise SystemExit(f"variant {key}: no translation of passages {missing}")
    Path(args.out).write_text(render(compare, outputs), encoding="utf-8")
    print(json.dumps({key: len(rows) for key, rows in outputs.items()}))


if __name__ == "__main__":
    main()
