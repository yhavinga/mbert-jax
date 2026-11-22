#!/usr/bin/env python3
"""The decision rule of T3 (experiments/t3/protocol.md, section 11), exactly on counts.

    python3 -I experiments/t3/decide_t3.py [--bucket gs://.../mbert] [--output decision.json] [--markdown decision.md]
    python3 -I experiments/t3/decide_t3.py --branch-only ...     # question 2 alone, after the branch (decides nothing)

Arms (results of eval_t3.sh in <bucket>/t3/{dumb,dumb-extra,blimp-nl}/):
    T3   the main leg, t3-1959000            <bucket>/t3/t3-culturax-317m/ckpt_keep/1959000
    T3b  the branch after two epochs, t3-epoch2-1306000   <bucket>/t3/t3-culturax-317m-epoch2/ckpt_keep/1306000
    T2   epoch2-1306000: seeds 42-44 the results of the final run (its DBRD@512 of robbert-compare), as decide_v2.py
         loads them; seeds 45 and 46 of the five tasks below from eval_t3.sh, in <bucket>/t3/ (amendment 1)
    A, A2  of the shape ablation (decide_v3.py): only |mean A - mean A2| per task, the noise of a second pretraining run
Seeds (protocol.md, amendment 1): DBRD@1024, DBRD@512, SICK-NL, SQuAD-NL and CoNLL-2002 have the five seeds 42-46 in
every arm, the POS tasks 42-44; a missing seed is FOUT. The difference X - Y is that of the means over all seeds of the
task (five or three); the seed ranges in the threshold are those over 42-44, so that the threshold stays that of the
ablation.
Per task the threshold is the largest of the two arms' seed ranges and the noise (BLiMP-NL: of 0.5 point and the
noise); DBRD@1024 has no noise of its own in the ablation and takes DBRD@512's. "beter"/"slechter" above/below the
threshold (strictly), else "gelijk". Verdict (decide.primary_rule) on DBRD@1024 and SICK-NL: "schaadt", "helpt" or
"geen verschil". Extra tasks: DBRD@512, SQuAD-NL F1, CoNLL-2002 F1, POS LassySmall, POS Alpino, BLiMP-NL.
  question 1  T3 against T2: the verdict and per task beter / gelijk / slechter. T3 is a new model next to T2, not its
              successor (amendment 1): no publication label; publishing it, under a name of its own, only on Yeb's word
  question 2  T3b against T2: shape and size at T2's data and tokens
  question 3  T3 against T3b: the third epoch; at "schaadt" question 2 is reported as the alternative for Yeb
A missing or inexact result, or one at other settings, is FOUT (exit code 1).
"""

import argparse
import json
import sys
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "final"))
sys.path.insert(0, str(HERE.parents[1] / "final-v2"))
sys.path.insert(0, str(HERE.parents[1] / "v3"))

import decide  # noqa: E402
import decide_v2  # noqa: E402
import decide_v3  # noqa: E402

BUCKET = "gs://t5-dutch-english-usc2/mbert"
MAIN = {"name": "t3-1959000", "run": "t3-culturax-317m", "step": 1959000}
BRANCH = {"name": "t3-epoch2-1306000", "run": "t3-culturax-317m-epoch2", "step": 1306000}
PRIMARY = ("dbrd1024", "sicknl")
EXTRA = ("dbrd512", "squadnl", "conll2002", "pos-lassysmall", "pos-alpino", "blimp-nl")
TASKS = PRIMARY + EXTRA
TITLES = dict(decide_v2.ROWS)
# Question 1 reports one of these verdicts; amendment 1 dropped the publication labels (vervangt T2 / naast T2 / niet
# publiceren): T3 is a new model, not T2's successor
OUTCOMES = ("helpt", "geen verschil", "schaadt")
SEEDS = decide.SEEDS  # 42-44: every fine-tune task, and the seed ranges of the threshold
MORE_SEEDS = (45, 46)  # amendment 1: these five tasks have the seeds 42-46
FIVE_SEEDS = ("dbrd1024", "dbrd512", "sicknl", "squadnl", "conll2002")
PRIMARY_SPECS = {"dbrd1024": decide.PRIMARY["dbrd1024"], "sicknl": decide.PRIMARY["sicknl"],
                 "dbrd512": decide_v2.PRIMARY_512["dbrd512"]}


def checkpoint_of(arm, bucket):
    return f"{bucket}/t3/{arm['run']}/ckpt_keep/{arm['step']}"


def seeds_of(task):
    return SEEDS + MORE_SEEDS if task in FIVE_SEEDS else SEEDS


def cell(kind, values, n):
    """One arm on one task from its values in seed order (42, 43, 44[, 45, 46]): "stat" is (the mean over all seeds,
    the seed range over 42-44), the two numbers of the rule (amendment 1); "range_all" (over all seeds) is reported."""
    fractions = list(values) if kind == "f1" else [Fraction(k, n) for k in values]
    first = fractions[:len(SEEDS)]
    return {"kind": kind, "n": n, "stat": (sum(fractions, Fraction(0)) / len(fractions), max(first) - min(first)),
            "range_all": max(fractions) - min(fractions),
            "values": [str(v) for v in values] if kind == "f1" else [f"{k}/{n}" for k in values]}


def load_task(read, name, task, parts):
    """One arm on one fine-tune task; parts: [(root, seeds), ...], in seed order."""
    if task in PRIMARY_SPECS:
        spec = PRIMARY_SPECS[task]
        return cell("counts", [k for root, seeds in parts
                               for k in decide.load_primary(read, root, name, task, spec, seeds=seeds)], spec["n"])
    loaded = [decide.load_extra(read, root, name, task, decide.EXTRA[task], seeds=seeds) for root, seeds in parts]
    if len({(kind, n) for kind, _, n in loaded}) != 1:
        raise decide.Fout(f"{name} {task}: the seeds are not counted alike ({[(kind, n) for kind, _, n in loaded]})")
    return cell(loaded[0][0], [v for _, values, _ in loaded for v in values], loaded[0][2])


def blimp_cell(k):
    return {"kind": "blimp", "k": k, "values": [f"{k}/{decide.BLIMP_PAIRS}"]}


def load_t3(read, bucket, arm):
    """The cells of one T3 arm: every result in <bucket>/t3."""
    root = f"{bucket}/t3"
    cells = {task: load_task(read, arm["name"], task, [(root, seeds_of(task))]) for task in TASKS if task != "blimp-nl"}
    cells["blimp-nl"] = blimp_cell(decide.load_blimp(read, root, arm["name"], checkpoint_of(arm, bucket)))
    return cells


def load_t2(read, bucket):
    """T2's cells: seeds 42-44 where decide_v2.py reads them (DBRD@512 in final/robbert), 45 and 46 in <bucket>/t3."""
    final = f"{bucket}/final"
    cells = {}
    for task in TASKS:
        if task == "blimp-nl":
            continue
        parts = [(f"{final}/robbert" if task == "dbrd512" else final, SEEDS)]
        if task in FIVE_SEEDS:
            parts.append((f"{bucket}/t3", MORE_SEEDS))
        cells[task] = load_task(read, decide_v2.T2_NAME, task, parts)
    cells["blimp-nl"] = blimp_cell(decide.load_blimp(read, final, decide_v2.T2_NAME, decide.ARMS["T2"]["checkpoint"]))
    return cells


def load_noise(read, bucket):
    """|mean A - mean A2| per task from the ablation's results; DBRD@1024 takes DBRD@512's."""
    a, a2 = decide_v3.load(read, bucket, "A"), decide_v3.load(read, bucket, "A2")
    noise = {task: abs(a[task][0] - a2[task][0]) for task in a}
    noise["dbrd1024"] = noise["dbrd512"]
    return noise


def compare(task, x, y, noise):
    """x against y on one task: difference, threshold, side, outcome."""
    if x["kind"] != y["kind"]:
        raise decide.Fout(f"{task}: the arms are not counted alike ({x['kind']} against {y['kind']})")
    if x["kind"] == "blimp":
        difference = Fraction(x["k"] - y["k"], decide.BLIMP_PAIRS)
        threshold = max(decide.BLIMP_MARGIN, noise[task])
    else:
        if x["kind"] == "words" and x["n"] != y["n"]:
            raise decide.Fout(f"{task}: the arms count different test words ({x['n']} against {y['n']})")
        (mean_x, range_x), (mean_y, range_y) = x["stat"], y["stat"]
        difference = mean_x - mean_y
        threshold = max(range_x, range_y, noise[task])
    side = "below" if -difference > threshold else "above" if difference > threshold else "within"
    return {"difference": difference, "threshold": threshold, "side": side, "outcome": decide.OUTCOME[side]}


def question(x, y, noise):
    cells = {task: compare(task, x[task], y[task], noise) for task in TASKS}
    verdict = decide.primary_rule({task: cells[task] for task in PRIMARY})
    return {"cells": cells, "verdict": verdict,
            "extra_worse": sorted(t for t in EXTRA if cells[t]["outcome"] == "slechter"),
            "extra_better": sorted(t for t in EXTRA if cells[t]["outcome"] == "beter"),
            "better": sorted(t for t in TASKS if cells[t]["outcome"] == "beter"),
            "worse": sorted(t for t in TASKS if cells[t]["outcome"] == "slechter")}


def summary(q):
    """The verdict and the tasks better and worse, in Dutch."""
    def names(tasks):
        return ", ".join(TITLES[t] for t in tasks) or "geen"

    return f"{q['verdict']}; beter op {names(q['better'])}; slechter op {names(q['worse'])}"


MEANING_Q1 = {"helpt": "T3 is beter dan T2", "geen verschil": "geen meetbaar verschil tussen T3 en T2",
              "schaadt": "T3 is slechter dan T2"}
MEANING_Q2 = {"helpt": "de grotere vorm (met de fix) is bij T2's data en tokens beter",
              "geen verschil": "vorm en grootte geven bij twee epochs CulturaX geen meetbare winst",
              "schaadt": "de grotere vorm is bij T2's data en tokens slechter"}
MEANING_Q3 = {"helpt": "het derde epoch helpt", "geen verschil": "geen meetbare winst van het derde epoch",
              "schaadt": "het derde epoch schaadt"}


def decision(t3, t3b, t2, noise):
    out = {}
    if t3b is not None:
        q2 = question(t3b, t2, noise)
        out["question2"] = {**q2, "meaning": MEANING_Q2[q2["verdict"]]}
    if t3 is not None:
        q1 = question(t3, t2, noise)
        out["question1"] = {**q1, "meaning": MEANING_Q1[q1["verdict"]]}
        if t3b is not None:
            q3 = question(t3, t3b, noise)
            out["question3"] = {**q3, "meaning": MEANING_Q3[q3["verdict"]]}
            if q3["verdict"] == "schaadt":
                out["alternative"] = {"arm": "T3b", "summary": summary(out["question2"]),
                                      "note": "het derde epoch schaadt: Yeb kiest of de tak in plaats van de hoofdtak "
                                              "het nieuwe model wordt"}
    return out


def pct(fraction):
    return float(100 * fraction)


def nl(fraction, digits=2):
    return f"{pct(fraction):.{digits}f}".replace(".", ",")


def signed(fraction):
    return ("+" if fraction >= 0 else "−") + nl(abs(fraction))


def mean_of(cell):
    return Fraction(cell["k"], decide.BLIMP_PAIRS) if cell["kind"] == "blimp" else cell["stat"][0]


def range_of(cell):
    return None if cell["kind"] == "blimp" else cell["stat"][1]


def markdown(arms, noise, d):
    keys = [k for k in ("T3", "T3b", "T2") if arms.get(k) is not None]
    label = {"T3": "T3-hoofd", "T3b": "T3-tak", "T2": "T2"}
    lines = ["| taak | " + " | ".join(label[k] for k in keys) + " | ruis |A − A2| |", "|---|" + "---:|" * (len(keys) + 1)]
    for task in TASKS:
        cells = []
        for key in keys:
            cell = arms[key][task]
            spread = range_of(cell)
            cells.append(nl(mean_of(cell)) + ("" if spread is None else f" [{nl(spread)}]"))
        lines.append(f"| {TITLES[task]} | " + " | ".join(cells) + f" | {nl(noise[task])} |")
    lines += ["", "Gemiddelde over de seeds 42–46 (DBRD@1024, DBRD@512, SICK-NL, SQuAD-NL, CoNLL-2002) of 42–44 (POS); "
              "tussen haken de seed-range over 42–44, die in de drempel telt (amendement 1)."]
    word = {"slechter": "slechter", "beter": "beter", "gelijk": "gelijk"}
    for key, title in (("question1", "Vraag 1: T3-hoofd − T2"), ("question2", "Vraag 2: T3-tak − T2"),
                       ("question3", "Vraag 3: T3-hoofd − T3-tak")):
        if key not in d:
            continue
        q = d[key]
        lines += ["", f"**{title}**", "", "| taak | verschil | drempel | oordeel |", "|---|---:|---:|---|"]
        for task in TASKS:
            c = q["cells"][task]
            lines.append(f"| {TITLES[task]} | {signed(c['difference'])} | {nl(c['threshold'])} | {word[c['outcome']]} |")
        lines.append("")
        lines.append(f"Oordeel: **{q['verdict']}**" + (f" — {q['meaning']}" if "meaning" in q else "") + ".")
    if "question2" in d and "question1" not in d:
        lines += ["", f"Voorlopig (beslist niets): T3-tak tegen T2: **{summary(d['question2'])}**."]
    if "question1" in d:
        lines += ["", f"**Vraag 1, T3-hoofd tegen T2: {summary(d['question1'])}.** T3 is een nieuw model, geen vervanging "
                      "van T2 (amendement 1); publiceren, onder een eigen naam, alleen op Yebs woord."]
    if "alternative" in d:
        a = d["alternative"]
        lines += ["", f"Alternatief: {a['note']}; T3-tak tegen T2 (vraag 2): **{a['summary']}**."]
    return "\n".join(lines) + "\n"


def exact(c):
    return {key: str(value) if isinstance(value, Fraction) else value for key, value in c.items()}


def serializable(d):
    out = {}
    for key, value in d.items():
        if isinstance(value, dict) and "cells" in value:
            out[key] = {**value, "cells": {t: exact(c) for t, c in value["cells"].items()}}
        else:
            out[key] = value
    return out


def main(argv=None, read=decide.read_json):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bucket", default=BUCKET, help="gs://.../mbert or a local directory with the same layout")
    parser.add_argument("--branch-only", action="store_true", help="question 2 alone (T3-tak against T2)")
    parser.add_argument("--output", default=None)
    parser.add_argument("--markdown", default=None)
    args = parser.parse_args(argv)
    try:
        noise = load_noise(read, args.bucket)
        t2 = load_t2(read, args.bucket)
        t3b = load_t3(read, args.bucket, BRANCH)
        t3 = None if args.branch_only else load_t3(read, args.bucket, MAIN)
        d = decision(t3, t3b, t2, noise)
    except decide.Fout as error:
        print(f"FOUT: {error}")
        return 1
    arms = {"T3": t3, "T3b": t3b, "T2": t2}
    text = markdown(arms, noise, d)
    print(text)
    for key in ("question1", "question2", "question3"):
        if key in d:
            print(f"{key.upper()}: {d[key]['verdict']}; " + "  ".join(
                f"{task} {pct(c['difference']):+.3f} (drempel {pct(c['threshold']):.3f}) {c['outcome']}" for task, c in d[key]["cells"].items()))
    if "question1" in d:
        print(f"VRAAG 1 (T3-hoofd tegen T2): {summary(d['question1'])}")
    else:
        print(f"VOORLOPIG (vraag 2, tak tegen T2, beslist niets): {summary(d['question2'])}")
    if args.output:
        out = {"arms": {key: None if arm is None else {task: {"values": c["values"], "mean": str(mean_of(c)),
                                                              "range": None if range_of(c) is None else str(range_of(c)),
                                                              "range_all": str(c["range_all"]) if "range_all" in c else None}
                                                       for task, c in arm.items()} for key, arm in arms.items()},
               "noise": {task: str(v) for task, v in noise.items()}, "branch_only": args.branch_only, **serializable(d)}
        Path(args.output).write_text(json.dumps(out, indent=2) + "\n")
    if args.markdown:
        Path(args.markdown).write_text(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
