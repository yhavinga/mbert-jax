#!/usr/bin/env python3
"""The decision rule of protocol.md, with its amendment 1, applied to the results of the 2048 phase.

Usage:
    gsutil -m cp 'gs://t5-dutch-english-usc2/mbert/phase2048/dumb/*.json' results/
    gsutil -m cp 'gs://t5-dutch-english-usc2/mbert/phase2048/eval/*.json' eval/
    python experiments/phase2048/decide.py --results results --eval eval --start u15 --output decision.json

Reads <start>-{S,C,L}--{dbrd512,dbrd1024,dbrd2048,sicknl512}--seed{42,43,44}.json, the results.json of
scripts/dumb/finetune.py, and decides on test_accuracy. Every cell needs exactly these three seeds. Reads
<start>-{S,C,L}.json, the output of scripts/eval_windows.py, for the paired window test of amendment 1.
"""

import argparse
import json
import statistics
from pathlib import Path

MODELS = ("S", "C", "L")
CELLS = ("dbrd512", "dbrd1024", "dbrd2048", "sicknl512")
SEEDS = (42, 43, 44)
METRIC = "test_accuracy"
WINDOWS = ("1024", "2048")


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


def load_windows(eval_dir, start):
    """{model: {window: result}} of scripts/eval_windows.py; a window cut short by --max-batches is an error."""
    windows = {}
    for model in MODELS:
        path = Path(eval_dir) / f"{start}-{model}.json"
        if not path.exists():
            raise FileNotFoundError(f"{path}: amendment 1 needs the window eval of every model")
        result = json.loads(path.read_text())["windows"]
        for window in WINDOWS:
            if window not in result:
                raise ValueError(f"{path}: no window {window}")
            if not result[window]["complete"]:
                raise ValueError(f"{path}: window {window} did not count all documents")
        windows[model] = result
    return windows


def uses_context(windows):
    """Amendment 1 of protocol.md: does L use the context beyond 1024 tokens?"""
    # The test is paired: every model and both windows must have scored the same masked tokens
    for window in WINDOWS:
        for group in windows["L"][window]["groups"]:
            counts = {model: windows[model][window]["groups"][group]["masked"] for model in MODELS}
            if len(set(counts.values())) != 1:
                raise ValueError(f"window {window}, {group}: masked tokens differ between models: {counts}")
    for model in MODELS:
        counts = {window: windows[model][window]["groups"]["long_documents"]["masked"] for window in WINDOWS}
        if counts["1024"] != counts["2048"]:
            raise ValueError(f"{model}: the long documents have {counts} masked tokens per window, not the same tokens")

    def loss(model, window, group):
        return windows[model][window]["groups"][group]["loss"]

    # "L's loss on the long documents at window 2048 is lower than at window 1024"
    gain = loss("L", "1024", "long_documents") - loss("L", "2048", "long_documents")
    # "and L's loss on positions 1024-2047 is lower than C's"
    below_c = loss("L", "2048", "positions_high") < loss("C", "2048", "positions_high")
    return {
        "L_long_documents_1024_minus_2048": gain,
        "positions_high_2048": {model: loss(model, "2048", "positions_high") for model in MODELS},
        "long_documents": {model: {window: loss(model, window, "long_documents") for window in WINDOWS} for model in MODELS},
        "uses_context": gain > 0 and below_c,
    }


def summarize(values):
    return {"values": values, "mean": statistics.mean(values), "range": max(values) - min(values)}


def decide(cells, windows):
    """The rule of protocol.md and its amendment 1, word for word in the comments."""
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
    context = uses_context(windows)

    if harmed:
        decision = "2048 harms"
    # Amendment 1: a gain on DBRD@2048, or no harm with L using the longer context
    elif exceeds or context["uses_context"]:
        decision = "2048 helps"
    else:
        decision = "no measurable difference"
    return {
        "metric": METRIC,
        "table": table,
        "help": {"best_C_dbrd_cell": best_c_cell, "L_dbrd2048_minus_best_C": help_gain, "margin": help_margin, "exceeds": exceeds},
        "context": context,
        "harm": harm_checks,
        # "S vs C shows the effect of the extra steps themselves": reported, it decides nothing
        "C_minus_S": {cell: C[cell]["mean"] - S[cell]["mean"] for cell in CELLS},
        "decision": decision,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", required=True, help="Directory with the result files")
    parser.add_argument("--eval", required=True, help="Directory with the window eval files")
    parser.add_argument("--start", required=True, help="START_NAME of the runs, e.g. u15")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    result = decide(load_cells(args.results, args.start), load_windows(args.eval, args.start))
    for model in MODELS:
        print(model, "  ".join(f"{cell} {100 * s['mean']:.2f} (range {100 * s['range']:.2f})" for cell, s in result["table"][model].items()))
    print("help:", json.dumps(result["help"]))
    print("context:", json.dumps(result["context"]))
    print("harm:", json.dumps(result["harm"]))
    print("C - S:", json.dumps(result["C_minus_S"]))
    print("DECISION:", result["decision"])
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
