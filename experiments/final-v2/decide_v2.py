#!/usr/bin/env python3
"""The decision rule of experiments/final-v2/protocol.md ("Beslisregel"), exactly on counts: V (dmbert v2) against T2
(step 1) and against RobBERT-2023 (step 2), and the outcome A (V replaces T2), B (V next to T2) or C (Yeb decides
again). The counting, the checks of every result and the comparisons are those of experiments/final/decide.py,
imported from there; this file adds only the arms, DBRD@512 and the rule.

Usage (where gsutil reads the bucket; ~70 small result files):
    python3 -I experiments/final-v2/decide_v2.py [--output decision.json] [--markdown decision.md]
    python3 -I experiments/final-v2/decide_v2.py --arm V        # V's 22 results checked and summarized

Results (a local directory with the same layout works, --bucket):
    V    <bucket>/final-v2/{dumb,dumb-extra,blimp-nl}/v2-1306000--...         eval_v2.sh (DBRD@512 in dumb/ too)
    T2   <bucket>/final/{dumb,dumb-extra,blimp-nl}/epoch2-1306000--...       the final run's evaluation
         <bucket>/final/robbert/dumb/epoch2-1306000--dbrd512--seed<n>.json   its DBRD@512 (robbert-compare)
    R23  <bucket>/final/robbert/{dumb,dumb-extra,blimp-nl}/robbert-2023--...  robbert-compare
A result missing, not exact or at other settings is FOUT (exit code 1).
"""

import argparse
import json
import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "final"))

import decide  # noqa: E402

BUCKET = "gs://t5-dutch-english-usc2/mbert"
V_NAME, V_CHECKPOINT = "v2-1306000", f"{BUCKET}/v2-fw2optout-230m/ckpt_keep/1306000"
T2_NAME = decide.ARMS["T2"]["name"]
R23_NAME = "robbert-2023"

# DBRD@512 as robbert-compare ran it (experiments/robbert/protocol.md, report.py): finetune.py --task dbrd
# --max-length 512, update 192, padding to a multiple of 128; RobBERT's SICK-NL is checked the same way
PRIMARY_512 = {
    "dbrd512": {"n": 2112, "update": 192, "checkpoint_name": False,
                "settings": {"task": "dbrd", "learning_rate": 5e-5, "epochs": 3, "max_length": 512, "pad_to_multiple_of": 128}},
    "sicknl": {"n": 4864, "update": 256, "checkpoint_name": False,
               "settings": {"task": "sicknl", "learning_rate": 5e-5, "epochs": 3, "max_length": 512, "pad_to_multiple_of": 1}},
}
BLIMP_HF_SETTINGS = {"metric": "PLL-word-l2r", "max_pairs": None, "compute_dtype": "float32", "pairs": decide.BLIMP_PAIRS}
# Step 1: the extra tests of V against T2; step 2: the tests against RobBERT-2023 that decide, and those reported
STEP1_EXTRA = ("dbrd512", "squadnl", "conll2002", "pos-lassysmall", "pos-alpino", "blimp-nl")
STEP2_DECIDING = ("dbrd512", "sicknl", "blimp-nl")
STEP2_REPORTED = ("squadnl", "pos-lassysmall", "pos-alpino")
OUTCOMES = {"A": "vervangt T2", "B": "naast T2", "C": "Yeb beslist opnieuw"}


def counts(ks, n):
    return {"kind": "counts", "stat": decide.stat_counts(ks, n), "values": [f"{k}/{n}" for k in ks]}


def extra_cell(entry):
    kind, values, n = entry
    return {"kind": kind, "stat": decide.extra_stat(entry), "entry": entry,
            "values": [f"{v}/{n}" if kind == "words" else str(v) for v in values]}


def load_v(read, bucket):
    root = f"{bucket}/final-v2"
    arm = decide.load_arm(read, root, V_NAME, V_CHECKPOINT)
    cells = {task: counts(ks, decide.PRIMARY[task]["n"]) for task, ks in arm["primary"].items()}
    cells["dbrd512"] = counts(decide.load_primary(read, root, V_NAME, "dbrd512", PRIMARY_512["dbrd512"]), 2112)
    cells.update({task: extra_cell(entry) for task, entry in arm["extra"].items()})
    cells["blimp-nl"] = {"kind": "blimp", "k": arm["blimp"], "values": [f"{arm['blimp']}/{decide.BLIMP_PAIRS}"]}
    return cells


def load_t2(read, bucket):
    root = f"{bucket}/final"
    arm = decide.load_arm(read, root, T2_NAME, decide.ARMS["T2"]["checkpoint"])
    cells = {task: counts(ks, decide.PRIMARY[task]["n"]) for task, ks in arm["primary"].items()}
    cells["dbrd512"] = counts(decide.load_primary(read, f"{root}/robbert", T2_NAME, "dbrd512", PRIMARY_512["dbrd512"]), 2112)
    cells.update({task: extra_cell(entry) for task, entry in arm["extra"].items()})
    cells["blimp-nl"] = {"kind": "blimp", "k": arm["blimp"], "values": [f"{arm['blimp']}/{decide.BLIMP_PAIRS}"]}
    return cells


def load_r23(read, bucket):
    root = f"{bucket}/final/robbert"
    cells = {task: counts(decide.load_primary(read, root, R23_NAME, task, PRIMARY_512[task]), PRIMARY_512[task]["n"])
             for task in ("dbrd512", "sicknl")}
    for task in STEP2_REPORTED + ("conll2002",):
        cells[task] = extra_cell(decide.load_extra(read, root, R23_NAME, task, decide.EXTRA[task]))
    relative = f"blimp-nl/{R23_NAME}.json"
    result = read(root, relative)
    decide.check_settings(result, BLIMP_HF_SETTINGS, f"{root}/{relative}")
    k = decide.exact_count(result.get("accuracy"), decide.BLIMP_PAIRS, f"{root}/{relative} accuracy")
    cells["blimp-nl"] = {"kind": "blimp", "k": k, "values": [f"{k}/{decide.BLIMP_PAIRS}"]}
    return cells


def compare_cells(task, a, b):
    """a against b on one task: difference, threshold (the larger seed range; BLiMP-NL 0.5 point), side, outcome."""
    if a["kind"] == "blimp":
        return decide.compare_blimp(a["k"], b["k"])
    if a["kind"] in ("words", "f1"):
        return decide.compare_extra(task, a["entry"], b["entry"])
    c = decide.compare(a["stat"], b["stat"])
    return {**c, "outcome": decide.OUTCOME[c["side"]]}


def decision(v, t2, r23):
    step1 = {task: compare_cells(task, v[task], t2[task]) for task in ("dbrd1024", "sicknl")}
    verdict = decide.primary_rule(step1)
    extra1 = {task: compare_cells(task, v[task], t2[task]) for task in STEP1_EXTRA}
    step2 = {task: compare_cells(task, v[task], r23[task]) for task in STEP2_DECIDING}
    reported2 = {task: compare_cells(task, v[task], r23[task]) for task in STEP2_REPORTED}
    worse_than_t2 = sorted(t for t, c in extra1.items() if c["outcome"] == "slechter")
    worse_than_r23 = sorted(t for t, c in step2.items() if c["outcome"] == "slechter")
    if verdict in ("helpt", "geen verschil") and not worse_than_t2:
        outcome, why = "A", f"stap 1: V {verdict} tegen T2 en geen extra toets slechter"
    elif not worse_than_r23:
        reasons = [f"V {verdict} tegen T2"] if verdict == "schaadt" else []
        reasons += [f"slechter dan T2 op {', '.join(worse_than_t2)}"] if worse_than_t2 else []
        outcome, why = "B", f"niet A ({'; '.join(reasons)}); tegen RobBERT-2023 op DBRD@512, SICK-NL en BLiMP-NL nergens slechter"
    else:
        outcome, why = "C", f"niet A en niet B: tegen RobBERT-2023 slechter op {', '.join(worse_than_r23)}"
    return {"step1": {"primary": step1, "verdict": verdict, "extra": extra1},
            "step2": {"deciding": step2, "reported": reported2},
            "outcome": outcome, "outcome_name": OUTCOMES[outcome], "reason": why}


def pct(fraction):
    return float(100 * fraction)


def mean_of(cell):
    return Fraction(cell["k"], decide.BLIMP_PAIRS) if cell["kind"] == "blimp" else cell["stat"][0]


def range_of(cell):
    return None if cell["kind"] == "blimp" else cell["stat"][1]


def exact(c):
    return {key: str(value) if isinstance(value, Fraction) else value for key, value in c.items()}


ROWS = [("dbrd1024", "DBRD@1024"), ("sicknl", "SICK-NL"), ("dbrd512", "DBRD@512"), ("squadnl", "SQuAD-NL v1.1 F1"),
        ("conll2002", "CoNLL-2002 entity-F1"), ("pos-lassysmall", "POS LassySmall"), ("pos-alpino", "POS Alpino"),
        ("blimp-nl", "BLiMP-NL")]


def nl(fraction, digits=2):
    return f"{pct(fraction):.{digits}f}".replace(".", ",")


def markdown(arms, d):
    word = {"slechter": "slechter", "beter": "beter", "gelijk": "geen verschil"}
    lines = ["| taak | V | T2 | RobBERT-2023 |", "|---|---:|---:|---:|"]
    for task, label in ROWS:
        cells = []
        for key in ("V", "T2", "R23"):
            cell = arms[key].get(task)
            if cell is None:
                cells.append("—")
            elif range_of(cell) is None:
                cells.append(nl(mean_of(cell)))
            else:
                cells.append(f"{nl(mean_of(cell))} [{nl(range_of(cell))}]")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines += ["", "| vergelijking | taak | verschil | drempel | oordeel |", "|---|---|---:|---:|---|"]
    for title, group in (("V − T2 (stap 1, beslissend)", d["step1"]["primary"]), ("V − T2 (stap 1, extra)", d["step1"]["extra"]),
                         ("V − RobBERT-2023 (stap 2, beslissend)", d["step2"]["deciding"]),
                         ("V − RobBERT-2023 (gerapporteerd)", d["step2"]["reported"])):
        for task, c in group.items():
            outcome = c.get("outcome", decide.OUTCOME[c["side"]])
            lines.append(f"| {title} | {dict(ROWS)[task]} | {pct(c['difference']):+.2f} | {pct(c['threshold']):.2f} | {word[outcome]} |")
    lines += ["", f"Stap 1: V **{d['step1']['verdict']}** tegen T2 (DBRD@1024 en SICK-NL).",
              f"Uitkomst: **{d['outcome']}, {d['outcome_name']}** ({d['reason']})."]
    return "\n".join(lines) + "\n"


def main(argv=None, read=decide.read_json):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bucket", default=BUCKET, help="gs://…/mbert or a local directory with the same layout")
    parser.add_argument("--arm", choices=["V"], default=None, help="only check and summarize V's results")
    parser.add_argument("--output", default=None, help="the decision as JSON (fractions as strings)")
    parser.add_argument("--markdown", default=None, help="the tables of the decision, for the report")
    args = parser.parse_args(argv)
    try:
        v = load_v(read, args.bucket)
        for task, label in ROWS:
            cell = v[task]
            spread = range_of(cell)
            print(f"V {label}: {pct(mean_of(cell)):.2f}" + ("" if spread is None else f" [range {pct(spread):.2f}]")
                  + f" ({', '.join(cell['values'])})")
        if args.arm:
            print("ARM V: 22 results, exact and at the protocol's settings")
            if args.output:
                Path(args.output).write_text(json.dumps({task: cell["values"] for task, cell in v.items()}, indent=2) + "\n")
            return 0
        t2, r23 = load_t2(read, args.bucket), load_r23(read, args.bucket)
        d = decision(v, t2, r23)
    except decide.Fout as error:
        print(f"FOUT: {error}")
        return 1
    for name, group in (("STAP 1 (V tegen T2)", d["step1"]["primary"]), ("STAP 1 extra", d["step1"]["extra"]),
                        ("STAP 2 (V tegen RobBERT-2023)", d["step2"]["deciding"]), ("STAP 2 gerapporteerd", d["step2"]["reported"])):
        print(f"{name}: " + "  ".join(f"{task} {pct(c['difference']):+.3f} (drempel {pct(c['threshold']):.3f}) "
                                      f"{c.get('outcome', decide.OUTCOME[c['side']])}" for task, c in group.items()))
    print(f"STAP 1: V {d['step1']['verdict']} tegen T2")
    print(f"UITKOMST {d['outcome']}: {d['outcome_name']} ({d['reason']})")
    arms = {"V": v, "T2": t2, "R23": r23}
    if args.output:
        out = {"arms": {key: {task: {"values": cell["values"], "mean": str(mean_of(cell)),
                                     "range": None if range_of(cell) is None else str(range_of(cell))}
                              for task, cell in arm.items()} for key, arm in arms.items()},
               "step1": {"primary": {t: exact(c) for t, c in d["step1"]["primary"].items()}, "verdict": d["step1"]["verdict"],
                         "extra": {t: exact(c) for t, c in d["step1"]["extra"].items()}},
               "step2": {k: {t: exact(c) for t, c in g.items()} for k, g in d["step2"].items()},
               "outcome": d["outcome"], "outcome_name": d["outcome_name"], "reason": d["reason"]}
        Path(args.output).write_text(json.dumps(out, indent=2) + "\n")
    if args.markdown:
        Path(args.markdown).write_text(markdown(arms, d))
    return 0


if __name__ == "__main__":
    sys.exit(main())
