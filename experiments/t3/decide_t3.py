#!/usr/bin/env python3
"""The decision rule of T3 (experiments/t3/protocol.md, section 11), exactly on counts.

    python3 -I experiments/t3/decide_t3.py [--bucket gs://.../mbert] [--output decision.json] [--markdown decision.md]
    python3 -I experiments/t3/decide_t3.py --branch-only ...     # question 2 alone, after the branch (decides nothing)

Arms (results of eval_t3.sh in <bucket>/t3/{dumb,dumb-extra,blimp-nl}/):
    T3   the main leg, t3-1959000            <bucket>/t3/t3-culturax-317m/ckpt_keep/1959000
    T3b  the branch after two epochs, t3-epoch2-1306000   <bucket>/t3/t3-culturax-317m-epoch2/ckpt_keep/1306000
    T2   epoch2-1306000, the results of the final run (and its DBRD@512 of robbert-compare), as decide_v2.py loads them
    A, A2  of the shape ablation (decide_v3.py): only |mean A - mean A2| per task, the noise of a second pretraining run
Per task the threshold is the largest of the two arms' seed ranges and the noise (BLiMP-NL: of 0.5 point and the
noise); DBRD@1024 has no noise of its own in the ablation and takes DBRD@512's. "beter"/"slechter" above/below the
threshold (strictly), else "gelijk". Verdict (decide.primary_rule) on DBRD@1024 and SICK-NL: "schaadt", "helpt" or
"geen verschil". Extra tasks: DBRD@512, SQuAD-NL F1, CoNLL-2002 F1, POS LassySmall, POS Alpino, BLiMP-NL.
  question 1 (deciding)  T3 against T2: "vervangt T2" if "helpt" and no extra task worse, or "geen verschil" with an
                         extra task better and none worse; else "naast T2" if not "schaadt" and better on one of the
                         eight tasks; else "niet publiceren"
  question 2 (reported)  T3b against T2: shape and size at T2's data and tokens
  question 3 (reported)  T3 against T3b: the third epoch; at "schaadt" the outcome of question 1's rule for T3b against
                         T2 is reported as an alternative for Yeb
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
OUTCOMES = ("vervangt T2", "naast T2", "niet publiceren")


def checkpoint_of(arm, bucket):
    return f"{bucket}/t3/{arm['run']}/ckpt_keep/{arm['step']}"


def load_t3(read, bucket, arm):
    """The cells of one T3 arm, in decide_v2's form (counts, extra entries, BLiMP k)."""
    root = f"{bucket}/t3"
    loaded = decide.load_arm(read, root, arm["name"], checkpoint_of(arm, bucket))
    cells = {task: decide_v2.counts(ks, decide.PRIMARY[task]["n"]) for task, ks in loaded["primary"].items()}
    cells["dbrd512"] = decide_v2.counts(decide.load_primary(read, root, arm["name"], "dbrd512", decide_v2.PRIMARY_512["dbrd512"]), 2112)
    cells.update({task: decide_v2.extra_cell(entry) for task, entry in loaded["extra"].items()})
    cells["blimp-nl"] = {"kind": "blimp", "k": loaded["blimp"], "values": [f"{loaded['blimp']}/{decide.BLIMP_PAIRS}"]}
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
        if x["kind"] == "words" and x["entry"][2] != y["entry"][2]:
            raise decide.Fout(f"{task}: the arms count different test words ({x['entry'][2]} against {y['entry'][2]})")
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
            "better": sorted(t for t in TASKS if cells[t]["outcome"] == "beter")}


def outcome(q):
    """Question 1's rule on a comparison against T2: (outcome, reason)."""
    verdict, worse, better_extra, better = q["verdict"], q["extra_worse"], q["extra_better"], q["better"]
    if verdict == "helpt" and not worse:
        return "vervangt T2", "helpt en geen extra taak slechter"
    if verdict == "geen verschil" and better_extra and not worse:
        return "vervangt T2", f"geen verschil, beter op {', '.join(better_extra)} en nergens slechter"
    if verdict != "schaadt" and better:
        why = f"{verdict}, beter op {', '.join(better)}"
        return "naast T2", why + (f", maar slechter op {', '.join(worse)}" if worse else ", zonder extra taak beter")
    if verdict == "schaadt":
        return "niet publiceren", "schaadt (slechter op DBRD@1024 of SICK-NL)"
    return "niet publiceren", f"{verdict} en op geen taak beter"


MEANING_Q2 = {"helpt": "de grotere vorm (met de fix) is bij T2's data en tokens beter",
              "geen verschil": "vorm en grootte geven bij twee epochs CulturaX geen meetbare winst",
              "schaadt": "de grotere vorm is bij T2's data en tokens slechter"}
MEANING_Q3 = {"helpt": "het derde epoch helpt", "geen verschil": "geen meetbare winst van het derde epoch",
              "schaadt": "het derde epoch schaadt"}


def decision(t3, t3b, t2, noise):
    out = {}
    if t3b is not None:
        q2 = question(t3b, t2, noise)
        out["question2"] = {**q2, "meaning": MEANING_Q2[q2["verdict"]], "rule_of_question1": outcome(q2)}
    if t3 is not None:
        q1 = question(t3, t2, noise)
        result, why = outcome(q1)
        out["question1"] = {**q1, "outcome": result, "reason": why}
        out["outcome"], out["reason"] = result, why
        if t3b is not None:
            q3 = question(t3, t3b, noise)
            out["question3"] = {**q3, "meaning": MEANING_Q3[q3["verdict"]]}
            if q3["verdict"] == "schaadt":
                alt, alt_why = out["question2"]["rule_of_question1"]
                out["alternative"] = {"arm": "T3b", "outcome": alt, "reason": alt_why,
                                      "note": "het derde epoch schaadt: Yeb kiest of de tak de kandidaat wordt"}
    return out


def pct(fraction):
    return float(100 * fraction)


def nl(fraction, digits=2):
    return f"{pct(fraction):.{digits}f}".replace(".", ",")


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
    word = {"slechter": "slechter", "beter": "beter", "gelijk": "gelijk"}
    for key, title in (("question1", "Vraag 1 (beslissend): T3-hoofd − T2"), ("question2", "Vraag 2: T3-tak − T2"),
                       ("question3", "Vraag 3: T3-hoofd − T3-tak")):
        if key not in d:
            continue
        q = d[key]
        lines += ["", f"**{title}**", "", "| taak | verschil | drempel | oordeel |", "|---|---:|---:|---|"]
        for task in TASKS:
            c = q["cells"][task]
            lines.append(f"| {TITLES[task]} | {pct(c['difference']):+.2f} | {pct(c['threshold']):.2f} | {word[c['outcome']]} |")
        lines.append("")
        lines.append(f"Oordeel: **{q['verdict']}**" + (f" — {q['meaning']}" if "meaning" in q else "") + ".")
    if "question2" in d and "question1" not in d:
        rule, why = d["question2"]["rule_of_question1"]
        lines += ["", f"Voorlopig (beslist niets): de regel van vraag 1 op T3-tak tegen T2 gaf **{rule}** ({why})."]
    if "outcome" in d:
        lines += ["", f"**Uitkomst: {d['outcome']}** ({d['reason']}). Publiceren alleen op Yebs woord."]
    if "alternative" in d:
        a = d["alternative"]
        lines += ["", f"Alternatief: {a['note']}; de regel van vraag 1 op T3-tak tegen T2 geeft **{a['outcome']}** ({a['reason']})."]
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
        t2 = decide_v2.load_t2(read, args.bucket)
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
    if "outcome" in d:
        print(f"UITKOMST {d['outcome']} ({d['reason']})")
    else:
        rule, why = d["question2"]["rule_of_question1"]
        print(f"VOORLOPIG (tak tegen T2, beslist niets): {d['question2']['verdict']}; regel van vraag 1: {rule} ({why})")
    if args.output:
        out = {"arms": {key: None if arm is None else {task: {"values": cell["values"], "mean": str(mean_of(cell)),
                                                              "range": None if range_of(cell) is None else str(range_of(cell))}
                                                       for task, cell in arm.items()} for key, arm in arms.items()},
               "noise": {task: str(v) for task, v in noise.items()}, "branch_only": args.branch_only, **serializable(d)}
        Path(args.output).write_text(json.dumps(out, indent=2) + "\n")
    if args.markdown:
        Path(args.markdown).write_text(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
