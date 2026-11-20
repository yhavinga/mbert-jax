#!/usr/bin/env python3
"""The decision rule of the shape ablation of dmbert v3 (experiments/v3/protocol.md, part 2, "Beslisregel"), exactly
on counts, with the loaders and comparisons of experiments/final/decide.py (T2's rule) and its DBRD@512 of
experiments/final-v2/decide_v2.py.

    python3 -I experiments/v3/decide_v3.py [--bucket gs://.../mbert] [--output decision.json] [--markdown decision.md]

Arms (results in <bucket>/v3-ablation/{dumb,dumb-extra,blimp-nl}/ of eval_v3.sh, name abl-<arm>-100000): A (the shape
of T2 and v2), B, C, D, and A2 (A with --seed 43: a second pretraining run of A, for the pretraining noise; optional).
Each candidate X of B, C, D against A, per task:
  threshold   the larger seed range of X and A (decide.compare); with A2 also |mean A - mean A2|, so that a
              difference must exceed the noise of a second pretraining run as well; BLiMP-NL 0.5 point (or |A - A2|)
  outcome     "beter" / "slechter" above / below the threshold, otherwise "gelijk"
  verdict     decide.primary_rule on SICK-NL and DBRD@512: "schaadt" if worse on either, else "helpt" if better on
              SICK-NL, else "geen verschil"
  qualifies   "helpt" and no extra task (SQuAD-NL F1, CoNLL-2002 F1, POS LassySmall, POS Alpino, BLiMP-NL) worse; or
              "geen verschil" with at least one extra task better and none worse
Outcome: no candidate qualifies -> A stays; one -> that one; more -> the one with the most "beter" over all seven
tasks; a tie there -> Yeb decides ("Yeb"). A missing or inexact result is FOUT (exit code 1).
"""

import argparse
import json
import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "final"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "final-v2"))

import decide  # noqa: E402
import decide_v2  # noqa: E402

BUCKET = "gs://t5-dutch-english-usc2/mbert"
STEP = 100_000
# The shapes of protocol.md part 2: (layers, hidden, heads, intermediate, global attention every n layers)
SHAPES = {"A": (22, 1024, 16, 1536, 3), "B": (29, 768, 12, 2048, 3), "C": (16, 1024, 16, 2560, 3),
          "D": (22, 1024, 16, 1536, 2), "A2": (22, 1024, 16, 1536, 3)}
CANDIDATES = ("B", "C", "D")
DECIDING = ("sicknl", "dbrd512")
EXTRA = ("squadnl", "conll2002", "pos-lassysmall", "pos-alpino", "blimp-nl")
TITLES = {"sicknl": "SICK-NL", "dbrd512": "DBRD@512", "squadnl": "SQuAD-NL v1.1 F1", "conll2002": "CoNLL-2002 entity-F1",
          "pos-lassysmall": "POS LassySmall", "pos-alpino": "POS Alpino", "blimp-nl": "BLiMP-NL"}


def name_of(arm):
    return f"abl-{arm.lower()}-{STEP}"


def checkpoint_of(arm, bucket):
    return f"{bucket}/v3-ablation/v3-abl-{arm.lower()}/ckpt_keep/{STEP}"


def load(read, bucket, arm):
    """{task: (mean, range)} of one arm, from the exact counts."""
    root = f"{bucket}/v3-ablation"
    name = name_of(arm)
    stats = {}
    for task in DECIDING:
        spec = decide_v2.PRIMARY_512[task]
        stats[task] = decide.stat_counts(decide.load_primary(read, root, name, task, spec), spec["n"])
    for task in EXTRA[:-1]:
        stats[task] = decide.extra_stat(decide.load_extra(read, root, name, task, decide.EXTRA[task]))
    k = decide.load_blimp(read, root, name, checkpoint_of(arm, bucket))
    stats["blimp-nl"] = (Fraction(k, decide.BLIMP_PAIRS), None)
    return stats


def compare(task, x, a, noise):
    (mean_x, range_x), (mean_a, range_a) = x, a
    difference = mean_x - mean_a
    if task == "blimp-nl":
        threshold = decide.BLIMP_MARGIN
    else:
        threshold = max(range_x, range_a)
    if noise is not None:
        threshold = max(threshold, noise[task])
    side = "below" if -difference > threshold else "above" if difference > threshold else "within"
    return {"difference": difference, "threshold": threshold, "side": side, "outcome": decide.OUTCOME[side]}


def decision(stats):
    a = stats["A"]
    noise = ({task: abs(a[task][0] - stats["A2"][task][0]) for task in a} if "A2" in stats else None)
    per = {}
    for x in CANDIDATES:
        cells = {task: compare(task, stats[x][task], a[task], noise) for task in DECIDING + EXTRA}
        verdict = decide.primary_rule({task: cells[task] for task in DECIDING})
        worse = sorted(t for t in EXTRA if cells[t]["outcome"] == "slechter")
        better = sorted(t for t in EXTRA if cells[t]["outcome"] == "beter")
        qualifies = (verdict == "helpt" and not worse) or (verdict == "geen verschil" and better and not worse)
        per[x] = {"cells": cells, "verdict": verdict, "extra_worse": worse, "extra_better": better,
                  "qualifies": bool(qualifies),
                  "better_count": sum(1 for c in cells.values() if c["outcome"] == "beter")}
    qualified = [x for x in CANDIDATES if per[x]["qualifies"]]
    if not qualified:
        outcome, why = "A", "geen kandidaat voldoet: de vorm van T2 en v2 blijft"
    elif len(qualified) == 1:
        outcome, why = qualified[0], f"alleen {qualified[0]} voldoet"
    else:
        most = max(per[x]["better_count"] for x in qualified)
        top = [x for x in qualified if per[x]["better_count"] == most]
        if len(top) == 1:
            outcome, why = top[0], f"{', '.join(qualified)} voldoen; {top[0]} is op de meeste taken beter ({most})"
        else:
            outcome, why = "Yeb", f"{', '.join(top)} voldoen en zijn op evenveel taken beter ({most}): Yeb beslist"
    return {"candidates": per, "outcome": outcome, "reason": why, "with_a2": "A2" in stats,
            "pretraining_noise": None if noise is None else {t: str(v) for t, v in noise.items()}}


def pct(fraction):
    return float(fraction) * 100


def markdown(stats, d):
    arms = [arm for arm in ("A", "A2", "B", "C", "D") if arm in stats]
    lines = ["| taak | " + " | ".join(arms) + " |", "|---|" + "---:|" * len(arms)]
    for task in DECIDING + EXTRA:
        row = []
        for arm in arms:
            mean, spread = stats[arm][task]
            row.append(f"{pct(mean):.2f}" + ("" if spread is None else f" [{pct(spread):.2f}]"))
        lines.append(f"| {TITLES[task]} | " + " | ".join(row) + " |")
    lines += ["", "| kandidaat | taak | X − A | drempel | oordeel |", "|---|---|---:|---:|---|"]
    for x, p in d["candidates"].items():
        for task in DECIDING + EXTRA:
            c = p["cells"][task]
            lines.append(f"| {x} | {TITLES[task]} | {pct(c['difference']):+.2f} | {pct(c['threshold']):.2f} | {c['outcome']} |")
        lines.append(f"| {x} | **oordeel** | {p['verdict']} | | {'voldoet' if p['qualifies'] else 'voldoet niet'} |")
    lines += ["", f"**Uitkomst: {d['outcome']}** — {d['reason']}."]
    return "\n".join(lines) + "\n"


def main(argv=None, read=decide.read_json):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bucket", default=BUCKET)
    parser.add_argument("--without-a2", action="store_true", help="A2 was not run (Yeb's word): thresholds are seed ranges")
    parser.add_argument("--output")
    parser.add_argument("--markdown")
    args = parser.parse_args(argv)
    try:
        stats = {arm: load(read, args.bucket, arm) for arm in ("A", "B", "C", "D") + (() if args.without_a2 else ("A2",))}
    except decide.Fout as error:
        print(f"FOUT: {error}")
        return 1
    d = decision(stats)
    text = markdown(stats, d)
    print(text)
    if args.output:
        out = {**d, "candidates": {x: {**p, "cells": {t: {k: (str(v) if isinstance(v, Fraction) else v) for k, v in c.items()}
                                                         for t, c in p["cells"].items()}}
                                   for x, p in d["candidates"].items()},
               "stats": {arm: {t: [str(m), None if r is None else str(r)] for t, (m, r) in s.items()} for arm, s in stats.items()}}
        Path(args.output).write_text(json.dumps(out, indent=1) + "\n")
    if args.markdown:
        Path(args.markdown).write_text(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
