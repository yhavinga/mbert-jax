#!/usr/bin/env python3
"""The decision rule of experiments/final/protocol.md ("Beslisregel"), exactly on counts: question 1 (T1 against S)
and questions 2 and 3 (T2 against T1, which gives the final model), from the results of the evaluation chain.

Usage (where gsutil reads the bucket; at most 57 small result files):
    python3 -I experiments/final/decide.py [--question 1|2|all] [--output decision.json]
    python3 -I experiments/final/decide.py --arm s-653000       # one arm's 19 results checked and summarized

Results under --root (default gs://t5-dutch-english-usc2/mbert/final; a local directory with the same layout works):
    dumb/<name>--{dbrd1024,sicknl}--seed{42,43,44}.json       scripts/dumb/finetune.py (primary)
    dumb-extra/<task>/<name>--<task>--seed{42,43,44}.json     scripts/dumb/finetune_extra.py
    blimp-nl/<name>.json                                      scripts/eval_blimp_nl.py
Arms (protocol "Evaluatie"): S = s-653000, T1 = epoch1-653000, T2 = epoch2-1306000.

Exact, so no rounding decides: primary k = round(test_accuracy x n) with n = 2,112 (DBRD@1024) and 4,864 (SICK-NL);
POS k = round(test_metric x words) over the test words; BLiMP-NL k = round(accuracy x pairs) over the 8,400 pairs.
Each k / n must give the stored number back exactly (the scripts divide two integers), otherwise FOUT. F1 (SQuAD-NL,
CoNLL-2002) is the Fraction of the stored float. Means and seed ranges are fractions; "more than" is strict. Every
result is also checked against the protocol's settings (LR, epochs, update, length, seed, full data); a result at
other settings is FOUT, and so is a missing one. FOUT ends the program with exit code 1.
"""

import argparse
import json
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

ROOT = "gs://t5-dutch-english-usc2/mbert/final"
BUCKET = "gs://t5-dutch-english-usc2/mbert"
SEEDS = (42, 43, 44)
ARMS = {
    "S": {"name": "s-653000", "checkpoint": f"{BUCKET}/culturax-nl-230m-chunk1024/ckpt_keep/653000"},
    "T1": {"name": "epoch1-653000", "checkpoint": f"{BUCKET}/final-culturax-230m-epoch1/ckpt_keep/653000"},
    "T2": {"name": "epoch2-1306000", "checkpoint": f"{BUCKET}/final-culturax-230m-epoch2/ckpt_keep/1306000"},
}

# Primary tasks (finetune.py): n of the test split as finetune.py counts it (whole updates only), the update, and the
# fields of the protocol each result must carry. "checkpoint_name": the result's checkpoint is <dir>/ckpt<name>.
PRIMARY = {
    "dbrd1024": {"n": 2112, "update": 192, "checkpoint_name": True,
                 "settings": {"task": "dbrd", "learning_rate": 5e-5, "epochs": 3, "max_length": 1024}},
    "sicknl": {"n": 4864, "update": 256, "checkpoint_name": True,
               "settings": {"task": "sicknl", "learning_rate": 5e-5, "epochs": 3, "max_length": 512}},
}
# Extra tasks (finetune_extra.py, experiments/dumb-extra/protocol.md): "words" are counted exactly on the test words,
# "f1" is the stored float
_EXTRA_COMMON = {"learning_rate": 5e-5, "epochs": 3, "update_size": 32, "lr_schedule": "linear", "warmup_steps": 0,
                 "pad_to_multiple_of": 128}
EXTRA = {
    "squadnl": {"kind": "f1", "settings": {**_EXTRA_COMMON, "task": "squadnl", "metric": "f1", "max_length": 384,
                                           "doc_stride": 128}},
    "conll2002": {"kind": "f1", "settings": {**_EXTRA_COMMON, "task": "conll2002", "metric": "entity_f1",
                                             "max_length": 512}},
    "pos-lassysmall": {"kind": "words", "settings": {**_EXTRA_COMMON, "task": "pos-lassysmall", "metric": "accuracy",
                                                     "max_length": 512}},
    "pos-alpino": {"kind": "words", "settings": {**_EXTRA_COMMON, "task": "pos-alpino", "metric": "accuracy",
                                                 "max_length": 512}},
}
BLIMP_PAIRS = 8400
BLIMP_SETTINGS = {"metric": "PLL-word-l2r", "max_pairs": None, "export_head": 0, "compute_dtype": "float32",
                  "pairs": BLIMP_PAIRS}
# BLiMP-NL is "slechter" (or "beter") at more than 0.5 point, i.e. more than 42 of the 8,400 pairs
BLIMP_MARGIN = Fraction(1, 200)


class Fout(Exception):
    """A result that is missing, not exact, or not at the protocol's settings."""


def read_json(root, relative):
    """One result file under root (gs:// through gsutil, or a local directory)."""
    url = f"{root}/{relative}"
    if root.startswith("gs://"):
        try:
            done = subprocess.run(["gsutil", "cat", url], capture_output=True, timeout=600)
        except subprocess.TimeoutExpired:
            raise Fout(f"{url}: gsutil cat did not end within 600 s") from None
        if done.returncode != 0:
            raise Fout(f"{url}: {done.stderr.decode(errors='replace').strip() or 'gsutil cat failed'}")
        text = done.stdout
    else:
        path = Path(url)
        if not path.is_file():
            raise Fout(f"{url}: missing")
        text = path.read_bytes()
    try:
        return json.loads(text)
    except json.JSONDecodeError as error:
        raise Fout(f"{url}: not JSON ({error})") from None


def exact_count(value, n, where):
    """k of value = k / n; a value that k / n does not give back exactly is FOUT."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Fout(f"{where}: {value!r} is not a number")
    k = round(value * n)
    if not 0 <= k <= n or k / n != value:
        raise Fout(f"{where}: {value!r} is not k / {n} (k = round(value x n) = {k} gives {k / n!r})")
    return k


def check_settings(result, settings, where):
    for key, expected in settings.items():
        if key not in result:
            raise Fout(f"{where}: no field {key!r} (the protocol's {expected!r})")
        if result[key] != expected or type(result[key]) is not type(expected):
            raise Fout(f"{where}: {key} is {result[key]!r}, the protocol's is {expected!r}")


def load_primary(read, root, name, task, spec):
    """The k of the three seeds of one arm on one primary task."""
    ks = []
    for seed in SEEDS:
        relative = f"dumb/{name}--{task}--seed{seed}.json"
        where = f"{root}/{relative}"
        result = read(root, relative)
        check_settings(result, spec["settings"], where)
        update = result.get("batch_size", 0) * result.get("num_devices", 0) * result.get("gradient_accumulation_steps", 0)
        if update != spec["update"]:
            raise Fout(f"{where}: an update of {update} (batch_size x num_devices x gradient_accumulation_steps), "
                       f"the protocol's is {spec['update']}")
        if spec["checkpoint_name"] and not str(result.get("checkpoint", "")).endswith(f"/ckpt{name}"):
            raise Fout(f"{where}: checkpoint {result.get('checkpoint')!r} is not .../ckpt{name}")
        ks.append(exact_count(result.get("test_accuracy"), spec["n"], f"{where} test_accuracy"))
    return ks


def load_extra(read, root, name, task, spec):
    """("words", ks, n) or ("f1", fractions, None) of the three seeds of one arm on one extra task."""
    values, words = [], set()
    for seed in SEEDS:
        relative = f"dumb-extra/{task}/{name}--{task}--seed{seed}.json"
        where = f"{root}/{relative}"
        result = read(root, relative)
        check_settings(result, {**spec["settings"], "seed": seed}, where)
        data = result.get("data") or {}
        if data.get("max_train_examples") is not None or data.get("max_eval_examples") is not None:
            raise Fout(f"{where}: a smoke run (data {data}), not the whole data")
        test = result.get("test") or {}
        value = result.get("test_metric")
        if value != test.get("metric") or value != test.get(result["metric"]):
            raise Fout(f"{where}: test_metric {value!r} is not test[{result['metric']!r}] {test.get(result['metric'])!r}")
        if spec["kind"] == "words":
            n = test.get("words")
            if not isinstance(n, int) or n <= 0:
                raise Fout(f"{where}: no number of test words ({n!r})")
            words.add(n)
            values.append(exact_count(value, n, f"{where} test_metric"))
        else:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
                raise Fout(f"{where}: test_metric {value!r} is not an F1 in [0, 1]")
            values.append(Fraction(value))
    if spec["kind"] == "words":
        if len(words) != 1:
            raise Fout(f"{root}/dumb-extra/{task}/{name}--{task}: the seeds count different test words {sorted(words)}")
        return "words", values, words.pop()
    return "f1", values, None


def load_blimp(read, root, name, checkpoint):
    relative = f"blimp-nl/{name}.json"
    where = f"{root}/{relative}"
    result = read(root, relative)
    check_settings(result, BLIMP_SETTINGS, where)
    if checkpoint is not None and result.get("checkpoint") != checkpoint:
        raise Fout(f"{where}: checkpoint {result.get('checkpoint')!r}, the arm's is {checkpoint!r}")
    return exact_count(result.get("accuracy"), BLIMP_PAIRS, f"{where} accuracy")


def load_arm(read, root, name, checkpoint, primary=None, extra=None):
    """All results of one arm: {"primary": {task: ks}, "extra": {task: (kind, values, n)}, "blimp": k}."""
    primary = PRIMARY if primary is None else primary
    extra = EXTRA if extra is None else extra
    return {"name": name,
            "primary": {task: load_primary(read, root, name, task, spec) for task, spec in primary.items()},
            "extra": {task: load_extra(read, root, name, task, spec) for task, spec in extra.items()},
            "blimp": load_blimp(read, root, name, checkpoint)}


def stat_counts(ks, n):
    """Mean and seed range (max - min) of the accuracies k / n, as fractions."""
    return Fraction(sum(ks), len(ks) * n), Fraction(max(ks) - min(ks), n)


def stat_values(values):
    return sum(values, Fraction(0)) / len(values), max(values) - min(values)


def compare(a, b):
    """a against b, each (mean, range): the difference of the means, the threshold (the larger seed range), the side."""
    (mean_a, range_a), (mean_b, range_b) = a, b
    difference, threshold = mean_a - mean_b, max(range_a, range_b)
    # "More than" is strict
    side = "below" if -difference > threshold else "above" if difference > threshold else "within"
    return {"difference": difference, "threshold": threshold, "side": side}


def primary_rule(comparisons):
    """Steps 1-3 of the rule, A against B: "schaadt", "helpt" or "geen verschil"."""
    # 1. "schaadt" if A's mean is more than the threshold below B's on DBRD@1024 or on SICK-NL. Checked first.
    if any(c["side"] == "below" for c in comparisons.values()):
        return "schaadt"
    # 2. Otherwise "helpt" if A's mean is more than the threshold above B's on SICK-NL.
    if comparisons["sicknl"]["side"] == "above":
        return "helpt"
    # 3. Otherwise "geen verschil"; a gain on DBRD@1024 alone included.
    return "geen verschil"


OUTCOME = {"below": "slechter", "above": "beter", "within": "gelijk"}


def extra_stat(entry):
    kind, values, n = entry
    return stat_counts(values, n) if kind == "words" else stat_values(values)


def compare_extra(task, a, b):
    if a[0] != b[0] or a[2] != b[2]:
        raise Fout(f"{task}: the arms are not counted alike ({a[0]} over {a[2]} against {b[0]} over {b[2]})")
    c = compare(extra_stat(a), extra_stat(b))
    return {**c, "outcome": OUTCOME[c["side"]]}


def compare_blimp(k_a, k_b):
    difference = Fraction(k_a - k_b, BLIMP_PAIRS)
    side = "below" if -difference > BLIMP_MARGIN else "above" if difference > BLIMP_MARGIN else "within"
    return {"difference": difference, "threshold": BLIMP_MARGIN, "side": side, "outcome": OUTCOME[side],
            "pairs": k_a - k_b}


def question(arm_a, arm_b, primary=None):
    """A against B: the primary comparisons and verdict, and the outcome of every extra eval."""
    primary = PRIMARY if primary is None else primary
    comparisons = {task: compare(stat_counts(arm_a["primary"][task], spec["n"]), stat_counts(arm_b["primary"][task], spec["n"]))
                   for task, spec in primary.items()}
    extra = {task: compare_extra(task, arm_a["extra"][task], arm_b["extra"][task]) for task in arm_a["extra"]}
    extra["blimp-nl"] = compare_blimp(arm_a["blimp"], arm_b["blimp"])
    return {"primary": comparisons, "verdict": primary_rule(comparisons), "extra": extra}


def final_model(verdict, extra):
    """Questions 2 and 3 (A = T2, B = T1): the final model and why."""
    worse = sorted(task for task, c in extra.items() if c["outcome"] == "slechter")
    better = sorted(task for task, c in extra.items() if c["outcome"] == "beter")
    if verdict == "schaadt":
        return "T1", "T2 schaadt (regel 1)"
    if verdict == "helpt":
        if worse:
            return "T1", f"T2 helpt, maar is slechter op {', '.join(worse)} (regel 2)"
        return "T2", "T2 helpt en geen extra eval is slechter (regel 2)"
    if better and not worse:
        return "T2", f"geen verschil; beter op {', '.join(better)}, op geen enkele slechter (regel 3)"
    if worse:
        return "T1", f"geen verschil; slechter op {', '.join(worse)} (regel 3)"
    return "T1", "geen verschil en geen extra eval beter (regel 3)"


def pct(fraction):
    return float(100 * fraction)


def describe_arm(arm, primary=None):
    primary = PRIMARY if primary is None else primary
    out = {"name": arm["name"], "primary": {}, "extra": {}, "blimp": {"k": arm["blimp"], "n": BLIMP_PAIRS,
                                                                   "accuracy": str(Fraction(arm["blimp"], BLIMP_PAIRS))}}
    for task, ks in arm["primary"].items():
        mean, spread = stat_counts(ks, primary[task]["n"])
        out["primary"][task] = {"k": ks, "n": primary[task]["n"], "mean": str(mean), "range": str(spread)}
    for task, entry in arm["extra"].items():
        mean, spread = extra_stat(entry)
        kind, values, n = entry
        out["extra"][task] = {"kind": kind, "values": [str(v) for v in values], "n": n, "mean": str(mean),
                              "range": str(spread)}
    return out


def arm_lines(arm, primary=None):
    d = describe_arm(arm, primary)
    lines = [f"{arm['name']}: " + "  ".join(
        f"{task} {pct(Fraction(v['mean'])):.2f} [range {pct(Fraction(v['range'])):.2f}] k {v['k']} of {v['n']}"
        for task, v in d["primary"].items())]
    lines.append(f"{arm['name']}: " + "  ".join(
        f"{task} {pct(Fraction(v['mean'])):.2f} [range {pct(Fraction(v['range'])):.2f}]" for task, v in d["extra"].items())
        + f"  blimp-nl {pct(Fraction(arm['blimp'], BLIMP_PAIRS)):.2f} ({arm['blimp']} of {BLIMP_PAIRS})")
    return lines


def exact(c):
    return {key: str(value) if isinstance(value, Fraction) else value for key, value in c.items()}


def question_lines(label, q):
    lines = [f"{label}: " + "  ".join(
        f"{task} {pct(c['difference']):+.3f} threshold {pct(c['threshold']):.3f} {c['side']}" for task, c in q["primary"].items())
        + f"  -> {q['verdict']}"]
    lines.append(f"{label} extra: " + "  ".join(
        f"{task} {pct(c['difference']):+.3f} (threshold {pct(c['threshold']):.3f}) {c['outcome']}" for task, c in q["extra"].items()))
    return lines


def main(argv=None, read=read_json):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=ROOT, help="gs:// directory or local directory with dumb/, dumb-extra/, blimp-nl/")
    parser.add_argument("--question", choices=["1", "2", "all"], default="all",
                        help="1: T1 against S; 2: T2 against T1 and the final model; all: both (default)")
    parser.add_argument("--arm", default=None, help="Only check and summarize this arm's results (an arm's name)")
    parser.add_argument("--output", default=None, help="The decision as JSON (fractions as strings)")
    args = parser.parse_args(argv)
    by_name = {arm["name"]: key for key, arm in ARMS.items()}
    try:
        if args.arm is not None:
            if args.arm not in by_name:
                parser.error(f"--arm must be one of {sorted(by_name)}")
            arm = load_arm(read, args.root, args.arm, ARMS[by_name[args.arm]]["checkpoint"])
            for line in arm_lines(arm):
                print(line)
            print(f"ARM {args.arm}: 19 results, exact and at the protocol's settings")
            out = {"arm": describe_arm(arm)}
        else:
            needed = {"1": ["S", "T1"], "2": ["T1", "T2"], "all": ["S", "T1", "T2"]}[args.question]
            arms = {key: load_arm(read, args.root, ARMS[key]["name"], ARMS[key]["checkpoint"]) for key in needed}
            for key in needed:
                for line in arm_lines(arms[key]):
                    print(f"{key} {line}")
            out = {"arms": {key: describe_arm(arm) for key, arm in arms.items()}}
            if args.question in ("1", "all"):
                q1 = question(arms["T1"], arms["S"])
                for line in question_lines("VRAAG 1 (T1 tegen S)", q1):
                    print(line)
                out["question1"] = {"primary": {t: exact(c) for t, c in q1["primary"].items()}, "verdict": q1["verdict"],
                                    "extra": {t: exact(c) for t, c in q1["extra"].items()}}
                print(f"VRAAG 1: T1 {q1['verdict']} tegen S"
                      + (" (WSD hier slechter dan cosine)" if q1["verdict"] == "schaadt" else "")
                      + "; beslist niet over het eindmodel")
            if args.question in ("2", "all"):
                q2 = question(arms["T2"], arms["T1"])
                for line in question_lines("VRAAG 2 (T2 tegen T1)", q2):
                    print(line)
                model, why = final_model(q2["verdict"], q2["extra"])
                out["question2"] = {"primary": {t: exact(c) for t, c in q2["primary"].items()}, "verdict": q2["verdict"],
                                    "extra": {t: exact(c) for t, c in q2["extra"].items()}}
                out["final_model"] = {"arm": model, "name": ARMS[model]["name"], "reason": why}
                print(f"VRAAG 2: T2 {q2['verdict']} tegen T1")
                print(f"EINDMODEL (vraag 3): {model} ({ARMS[model]['name']}): {why}")
    except Fout as error:
        print(f"FOUT: {error}")
        return 1
    if args.output:
        Path(args.output).write_text(json.dumps(out, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
