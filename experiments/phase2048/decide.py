#!/usr/bin/env python3
"""The decision rule of protocol.md, applied to the DUMB results of the 2048 phase.

Usage:
    gsutil -m cp 'gs://t5-dutch-english-usc2/mbert/phase2048/dumb/*.json' results/
    python experiments/phase2048/decide.py --results results --start u15 --output decision.json

Reads <start>-{S,C,L}--{dbrd512,dbrd1024,dbrd2048,sicknl512}--seed{42,43,44}.json, the results.json of
scripts/dumb/finetune.py, and decides on test_accuracy. Every cell needs exactly these three seeds.
"""

import argparse
import json
import statistics
from pathlib import Path

MODELS = ("S", "C", "L")
CELLS = ("dbrd512", "dbrd1024", "dbrd2048", "sicknl512")
SEEDS = (42, 43, 44)
METRIC = "test_accuracy"


def load_cells(results_dir, start):
    """{model: {cell: [value per seed]}}; a missing or extra file is an error, not a smaller table."""
    cells = {}
    for model in MODELS:
        for cell in CELLS:
            values = []
            for seed in SEEDS:
                path = Path(results_dir) / f"{start}-{model}--{cell}--seed{seed}.json"
                if not path.exists():
                    raise FileNotFoundError(f"{path}: every cell needs seeds {SEEDS}")
                values.append(float(json.loads(path.read_text())[METRIC]))
            cells.setdefault(model, {})[cell] = values
    return cells


def summarize(values):
    return {"values": values, "mean": statistics.mean(values), "range": max(values) - min(values)}


def decide(cells):
    """The rule of protocol.md, word for word in the comments."""
    table = {model: {cell: summarize(values) for cell, values in by_cell.items()} for model, by_cell in cells.items()}
    C, L, S = table["C"], table["L"], table["S"]

    # "the best mean of C over {512, 1024, 2048}"
    best_c_cell = max(("dbrd512", "dbrd1024", "dbrd2048"), key=lambda cell: C[cell]["mean"])
    # "by more than the larger seed range (max-min over the 3 seeds) of the two compared cells"
    help_margin = max(L["dbrd2048"]["range"], C[best_c_cell]["range"])
    help_gain = L["dbrd2048"]["mean"] - C[best_c_cell]["mean"]
    exceeds = help_gain > help_margin

    # "L is not below C on SICK-NL or on DBRD@512 by more than C's seed range for that cell"
    harm_checks = {}
    for cell in ("sicknl512", "dbrd512"):
        shortfall = C[cell]["mean"] - L[cell]["mean"]
        harm_checks[cell] = {"C_minus_L": shortfall, "C_range": C[cell]["range"], "harmed": shortfall > C[cell]["range"]}
    harmed = any(check["harmed"] for check in harm_checks.values())

    if harmed:
        decision = "2048 harms"
    elif exceeds:
        decision = "2048 helps"
    else:
        decision = "no measurable difference"
    return {
        "metric": METRIC,
        "table": table,
        "help": {"best_C_dbrd_cell": best_c_cell, "L_dbrd2048_minus_best_C": help_gain, "margin": help_margin, "exceeds": exceeds},
        "harm": harm_checks,
        # "S vs C shows the effect of the extra steps themselves": reported, it decides nothing
        "C_minus_S": {cell: C[cell]["mean"] - S[cell]["mean"] for cell in CELLS},
        "decision": decision,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", required=True, help="Directory with the result files")
    parser.add_argument("--start", required=True, help="START_NAME of the runs, e.g. u15")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    result = decide(load_cells(args.results, args.start))
    for model in MODELS:
        print(model, "  ".join(f"{cell} {100 * s['mean']:.2f} (range {100 * s['range']:.2f})" for cell, s in result["table"][model].items()))
    print("help:", json.dumps(result["help"]))
    print("harm:", json.dumps(result["harm"]))
    print("C - S:", json.dumps(result["C_minus_S"]))
    print("DECISION:", result["decision"])
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
