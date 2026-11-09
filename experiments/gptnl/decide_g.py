#!/usr/bin/env python3
"""The decision rule of experiments/gptnl/protocol.md, exactly on the counts: G against X on DBRD and SICK-NL.

Usage (where gsutil reads the bucket; 24 small result files, the region of the runs):
    python3 experiments/gptnl/decide_g.py [--g cxgptnl-300000] [--output decision.json]

Reads the results of scripts/dumb/finetune.py, gs://t5-dutch-english-usc2/mbert/masking/dumb/
<name>--{dbrd,sicknl}--seed{42,43,44}.json, of G (--g), X (300000), F (fw2-300000) and R (fw2rh3-300000).
test_accuracy is k / n with n = 2,112 for DBRD and 4,864 for SICK-NL; k = round(test_accuracy x n) must give
test_accuracy back, and X's and F's k must be those of the protocol's table. Means and seed ranges are fractions, so
no rounding decides. G against F and R is reported the same way and decides nothing. Secondary, decides nothing:
DBRD@1024 (gs://t5-dutch-english-usc2/mbert/dbrd1024/dumb/, where X is u15-300000), once G's three files are there.

--g fw2rh3-300000 runs it all on R, an earlier run of the same form, in place of G (R itself then drops out of the
references). Its outcome, 2026-11-09: R against X DBRD +0.268 within 0.710, SICK-NL -5.195 below by more than 1.521,
"GPT-NL schaadt" in this rule's words; the numbers of R in mbert/LEARNINGS.md (hammond), DBRD@1024 included.
"""

import argparse
import json
import subprocess
from fractions import Fraction

BUCKET = "gs://t5-dutch-english-usc2/mbert"
DUMB = f"{BUCKET}/masking/dumb"
DBRD1024 = f"{BUCKET}/dbrd1024/dumb"
SEEDS = (42, 43, 44)
N = {"dbrd": 2112, "sicknl": 4864}
REFERENCES = {"X": "300000", "F": "fw2-300000", "R": "fw2rh3-300000"}
# The k of the protocol's table "Arms" (seeds 42, 43, 44): the files read are the ones the protocol fixed
PROTOCOL_K = {"X": {"dbrd": [1967, 1973, 1958], "sicknl": [3701, 3738, 3775]},
              "F": {"dbrd": [1950, 1933, 1936], "sicknl": [3667, 3651, 3633]}}
# DBRD@1024 at 300,000: X's results are named after the start name of the 2048 phase
DBRD1024_NAMES = {"X": "u15-300000", "F": "fw2-300000", "R": "fw2rh3-300000"}
DBRD1024_N = 2112


def gsutil_cat(url):
    done = subprocess.run(["gsutil", "cat", url], capture_output=True)
    if done.returncode != 0:
        raise FileNotFoundError(f"{url}: {done.stderr.decode().strip()}")
    return json.loads(done.stdout)


def gsutil_exists(url):
    return subprocess.run(["gsutil", "-q", "stat", url], capture_output=True).returncode == 0


def count(accuracy, n, where):
    """k of test_accuracy = k / n; an accuracy that is not a count over n is an error."""
    k = round(accuracy * n)
    if abs(k / n - accuracy) >= 1e-9:
        raise ValueError(f"{where}: test_accuracy {accuracy!r} is not k / {n}")
    return k


def load_counts(directory, name, task, n, read=gsutil_cat):
    counts = []
    for seed in SEEDS:
        url = f"{directory}/{name}--{task}--seed{seed}.json"
        counts.append(count(read(url)["test_accuracy"], n, url))
    return counts


def stat(ks, n):
    """Mean and seed range (max - min) of the accuracies k / n, as fractions."""
    return Fraction(sum(ks), len(ks) * n), Fraction(max(ks) - min(ks), n)


def compare(a, b, n):
    """a against b on one task: difference of means, the threshold (the larger of the two seed ranges), and where a lies."""
    mean_a, range_a = stat(a, n)
    mean_b, range_b = stat(b, n)
    threshold = max(range_a, range_b)
    difference = mean_a - mean_b
    # "More than" is strict
    side = "below" if -difference > threshold else "above" if difference > threshold else "within"
    return {"difference": difference, "threshold": threshold, "side": side}


def rule(g, x):
    """The decision rule of protocol.md for G against X: ({task: comparison}, verdict)."""
    tasks = {task: compare(g[task], x[task], N[task]) for task in N}
    # 1. "GPT-NL schaadt" if G's mean is below X's by more than the threshold on DBRD or on SICK-NL. Checked first.
    if any(c["side"] == "below" for c in tasks.values()):
        verdict = "GPT-NL schaadt"
    # 2. Otherwise "GPT-NL helpt" if G's mean is above X's by more than the threshold on SICK-NL.
    elif tasks["sicknl"]["side"] == "above":
        verdict = "GPT-NL helpt"
    # 3. Otherwise "geen verschil" (a gain on DBRD alone included).
    else:
        verdict = "geen verschil"
    return tasks, verdict


def pct(fraction):
    return float(100 * fraction)


def describe(ks, n):
    mean, spread = stat(ks, n)
    return {"k": ks, "n": n, "mean": str(mean), "mean_pct": pct(mean), "min_pct": pct(Fraction(min(ks), n)),
            "max_pct": pct(Fraction(max(ks), n)), "range": str(spread), "range_pct": pct(spread)}


def exact(comparison):
    return {"difference": str(comparison["difference"]), "difference_pct": pct(comparison["difference"]),
            "threshold": str(comparison["threshold"]), "threshold_pct": pct(comparison["threshold"]),
            "side": comparison["side"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--g", default="cxgptnl-300000", help="G's name in masking/dumb (fw2rh3-300000: the test on R)")
    parser.add_argument("--output", default=None, help="The decision as JSON (fractions as strings)")
    args = parser.parse_args()

    names = {"G": args.g, **{arm: name for arm, name in REFERENCES.items() if name != args.g}}
    if "X" not in names:
        parser.error("--g must not be X")
    counts = {arm: {task: load_counts(DUMB, name, task, n) for task, n in N.items()} for arm, name in names.items()}
    for arm, expected in PROTOCOL_K.items():
        if arm in counts and counts[arm] != expected:
            raise ValueError(f"{arm} ({names[arm]}): k {counts[arm]} are not the protocol's {expected}")
    out = {"names": names, "arms": {arm: {task: describe(ks, N[task]) for task, ks in by_task.items()}
                                    for arm, by_task in counts.items()}}
    for arm, by_task in out["arms"].items():
        print(f"{arm} {names[arm]:16s} " + "  ".join(
            f"{task} k {d['k']} of {d['n']}: {d['mean_pct']:.2f} [{d['min_pct']:.2f}-{d['max_pct']:.2f}] range {d['range_pct']:.3f}"
            for task, d in by_task.items()))

    out["rule"] = {}
    for reference in (arm for arm in names if arm != "G"):
        tasks, verdict = rule(counts["G"], counts[reference])
        out["rule"][reference] = {"tasks": {task: exact(c) for task, c in tasks.items()}, "verdict": verdict}
        print(f"G vs {reference} ({names[reference]}): " + "  ".join(
            f"{task} {pct(c['difference']):+.3f} threshold {pct(c['threshold']):.3f} {c['side']}" for task, c in tasks.items())
              + f"  -> {verdict}" + ("" if reference == "X" else " (reported, no consequence)"))

    # Secondary: DBRD@1024, reported as the rule reports, without consequence
    g_files = [f"{DBRD1024}/{args.g}--dbrd1024--seed{seed}.json" for seed in SEEDS]
    missing = [url for url in g_files if not gsutil_exists(url)]
    if missing:
        out["dbrd1024"] = {"missing": missing}
        print(f"DBRD@1024 (secondary): not reported, G's files missing: {' '.join(missing)}")
    else:
        g1024 = load_counts(DBRD1024, args.g, "dbrd1024", DBRD1024_N)
        out["dbrd1024"] = {"G": describe(g1024, DBRD1024_N)}
        d = out["dbrd1024"]["G"]
        print(f"DBRD@1024 (secondary) G {d['mean_pct']:.2f} [{d['min_pct']:.2f}-{d['max_pct']:.2f}]")
        for reference, name in DBRD1024_NAMES.items():
            if name == args.g:
                continue
            ref = load_counts(DBRD1024, name, "dbrd1024", DBRD1024_N)
            c = compare(g1024, ref, DBRD1024_N)
            out["dbrd1024"][reference] = {"name": name, **describe(ref, DBRD1024_N), "G_against": exact(c)}
            r = out["dbrd1024"][reference]
            print(f"  vs {reference} ({name}) {r['mean_pct']:.2f} [{r['min_pct']:.2f}-{r['max_pct']:.2f}]: "
                  f"{pct(c['difference']):+.3f} threshold {pct(c['threshold']):.3f} {c['side']}")

    print(f"VERDICT G ({args.g}) vs X: {out['rule']['X']['verdict']}")
    if args.output:
        with open(args.output, "w") as f:
            f.write(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
