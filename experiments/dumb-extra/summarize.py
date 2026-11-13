#!/usr/bin/env python3
"""The tables of experiments/dumb-extra/protocol.md from result files: mean and [min-max] over seeds per model and
task, the protocol's comparison with the reference model, and BLiMP-NL per model.

Usage (after gsutil -m cp of the result files into local directories):
    python experiments/dumb-extra/summarize.py --results res/squadnl res/conll2002 res/dbrd --reference S \\
        --compare mix mixheads [--blimp res/blimp-nl] [--output summary.json]

A fine-tune result is NAME--TASK--seed<n>.json of finetune.py or finetune_extra.py; its number is test_metric
(finetune_extra.py) or test_accuracy (finetune.py: DBRD, SICK-NL). The comparison: model M beats the reference R on a
task when mean(M) - mean(R) is larger than the larger of the two seed ranges (max - min); the other way round, R beats M.
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

RESULT_NAME = re.compile(r"(?P<name>.+)--(?P<task>[^-][^/]*?)--seed(?P<seed>\d+)\.json")
PROTOCOL_SEEDS = {42, 43, 44}


def read_results(directories):
    """{(name, task): {seed: number}} of every result file in the directories."""
    table = defaultdict(dict)
    for directory in directories:
        for path in sorted(Path(directory).glob("*--*--seed*.json")):
            match = RESULT_NAME.fullmatch(path.name)
            if match is None:
                raise ValueError(f"{path}: not NAME--TASK--seed<n>.json")
            result = json.loads(path.read_text())
            value = result["test_metric"] if "test_metric" in result else result["test_accuracy"]
            key = (match["name"], match["task"])
            seed = int(match["seed"])
            if seed in table[key]:
                raise ValueError(f"{path}: a second result of {key} seed {seed}")
            table[key][seed] = value
    return table


def summary(values):
    numbers = list(values.values())
    return {"mean": sum(numbers) / len(numbers), "min": min(numbers), "max": max(numbers), "range": max(numbers) - min(numbers),
            "seeds": sorted(values), "complete": set(values) == PROTOCOL_SEEDS}


def compare(table, reference, models):
    """Per model and task both have: the difference of means, the larger seed range, and which (if any) beats the other."""
    rows = []
    for model in models:
        for task in sorted({task for name, task in table if name == model}):
            if (reference, task) not in table:
                continue
            a, b = summary(table[(model, task)]), summary(table[(reference, task)])
            difference = a["mean"] - b["mean"]
            larger_range = max(a["range"], b["range"])
            verdict = model if difference > larger_range else reference if -difference > larger_range else "neither"
            rows.append({"model": model, "reference": reference, "task": task, "difference": difference,
                         "larger_range": larger_range, "beats": verdict, "complete": a["complete"] and b["complete"]})
    return rows


def read_blimp(directory):
    out = {}
    for path in sorted(Path(directory).glob("*.json")):
        result = json.loads(path.read_text())
        out[path.stem] = {"accuracy": result["accuracy"], "handcrafted": result["subsets"]["handcrafted"]["accuracy"],
                          "generated": result["subsets"]["generated"]["accuracy"], "pairs": result["pairs"],
                          "complete": result["max_pairs"] is None,
                          "phenomena": {v["name"]: v["accuracy"] for v in result["phenomena"].values()}}
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", nargs="+", default=[], help="Directories of NAME--TASK--seed<n>.json files")
    parser.add_argument("--reference", default="S")
    parser.add_argument("--compare", nargs="*", default=["mix", "mixheads"])
    parser.add_argument("--blimp", default=None, help="Directory of NAME.json files of eval_blimp_nl.py")
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)
    table = read_results(args.results)
    out = {"tasks": {f"{name}--{task}": summary(values) for (name, task), values in sorted(table.items())},
           "comparison": compare(table, args.reference, args.compare)}
    for key, s in out["tasks"].items():
        flag = "" if s["complete"] else f"  (seeds {s['seeds']}, not the protocol's)"
        print(f"{key:40s} {100 * s['mean']:6.2f} [{100 * s['min']:6.2f}-{100 * s['max']:6.2f}]{flag}")
    for row in out["comparison"]:
        print(f"{row['model']} vs {row['reference']} on {row['task']}: {100 * row['difference']:+.2f} points, larger seed range "
              f"{100 * row['larger_range']:.2f}: {row['beats']}{'' if row['complete'] else ' (incomplete seeds)'}")
    if args.blimp:
        out["blimp"] = read_blimp(args.blimp)
        for name, b in out["blimp"].items():
            print(f"BLiMP-NL {name:24s} {100 * b['accuracy']:6.2f} handcrafted {100 * b['handcrafted']:6.2f} "
                  f"generated {100 * b['generated']:6.2f}{'' if b['complete'] else ' (--max-pairs)'}")
    if args.output:
        Path(args.output).write_text(json.dumps(out, indent=2) + "\n")
    return out


if __name__ == "__main__":
    main()
