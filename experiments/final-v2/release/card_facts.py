#!/usr/bin/env python3
"""The training facts of a model card, from the run's own records after its last step (standard library only).

  python3 card_facts.py --data-state process_*.json --trainer-logs w0.log w1.log w2.log w3.log \\
      --last-step 1306000 --parameters 230110592 --out facts.json

data       from the data state of every process at the last step (ckpt_data_state/<step>/process_<i>.json): tokens
           and documents handed to training since step 0, passes over the corpus, the share of real tokens among the
           positions, and 6 x parameters x tokens.
precision  from the coordinator's trainer log (the one log with the "Compute dtype:" lines; train_minimal.py prints
           them on process 0 only, which need not be worker 0): every start of the trainer, the step it resumed
           from and its compute dtype. A later start that resumes from an earlier step discards what earlier starts
           computed past that step, so the steps that made the final checkpoint are the step ranges (from, to] that
           survive, each with its dtype.
nonfinite  the NONFINITE lines of those ranges only (a discarded stretch, such as a bf16 burst before a resume in
           float32, does not count), and the Skipped count of the last step's line, which must equal the last
           cumulative count of those lines.
Exit 1, with the reason, when a record is missing or does not add up; nothing is guessed.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# Searched, not anchored: a progress bar's text can precede a print on the same line
DTYPE = re.compile(r"Compute dtype: (\w+)")
RESUME = re.compile(r"Resuming from micro_step: (\d+) \(optimizer_step: (\d+)\)")
NONFINITE = re.compile(r"NONFINITE step=(\d+) skipped=(\d+) in_a_row=(\d+)")
STEP = re.compile(r"Step\s+(\d+) \|.*\| Skipped: (\d+)\s*$")
FINAL = re.compile(r"Final checkpoint saved at optimizer_step (\d+)")
# The detailed Step line comes every 5 x log_every (10) steps: a start stopped right after a checkpoint may not have
# logged the checkpoint's own step
STEP_LINE_EVERY = 50


class FactsError(Exception):
    pass


def data_facts(states: list[dict], last_step: int, positions_per_step: int, parameters: int) -> dict:
    """Tokens, documents and passes summed over the processes' data states at the last step."""
    if not states:
        raise FactsError("no data state files")
    counts = {s["identity"]["process_count"] for s in states}
    if counts != {len(states)}:
        raise FactsError(f"{len(states)} data state files, process_count {sorted(counts)}")
    per = []
    for i, s in enumerate(states):
        st = s["state"]
        if s["step"] != last_step:
            raise FactsError(f"data state {i} is of step {s['step']}, not {last_step}")
        if not st.get("counted_from_start"):
            raise FactsError(f"data state {i}: its counts do not run from step 0 (counted_from_start false)")
        epoch = st["epoch"]
        finished = st["documents_of_finished_epochs"]
        into = st["documents_completed"] - finished
        if epoch < 1 or into < 0:
            raise FactsError(f"data state {i}: epoch {epoch}, documents completed {st['documents_completed']}, "
                             f"of finished passes {finished}")
        # Every pass hands the same documents, in another order
        per.append({"tokens_read": st["tokens_read"], "documents_completed": st["documents_completed"],
                    "finished_passes": epoch, "documents_per_pass": finished / epoch,
                    "passes": epoch + into / (finished / epoch)})
    tokens = sum(p["tokens_read"] for p in per)
    return {
        "processes": len(per),
        "tokens_read": tokens,
        "documents_completed": sum(p["documents_completed"] for p in per),
        "finished_passes": sorted({p["finished_passes"] for p in per}),
        "passes_min": min(p["passes"] for p in per),
        "passes_max": max(p["passes"] for p in per),
        "real_token_share": tokens / (last_step * positions_per_step),
        "training_flop": 6 * parameters * tokens,
        "per_process": per,
    }


def coordinator_log(texts: dict[str, str]) -> tuple[str, list[str]]:
    """(name, lines) of the one log with "Compute dtype:" lines; lines split at \\n and \\r (progress bars)."""
    with_dtype = {}
    for name, text in texts.items():
        lines = re.split(r"[\r\n]", text)
        if any(DTYPE.search(line) for line in lines):
            with_dtype[name] = lines
    if len(with_dtype) != 1:
        raise FactsError(f"'Compute dtype:' lines in {sorted(with_dtype) or 'no log'}: need exactly one coordinator log")
    return next(iter(with_dtype.items()))


def starts_of(lines: list[str]) -> list[dict]:
    """Every start of the trainer in the log: its dtype, the step it resumed from, its NONFINITE and Step lines."""
    starts = []
    for line in lines:
        if m := DTYPE.search(line):
            starts.append({"dtype": m.group(1), "resumed_from": None, "nonfinite": [], "steps": [], "final": None})
            continue
        if not starts:
            continue
        cur = starts[-1]
        if m := RESUME.search(line):
            if cur["resumed_from"] is not None or cur["steps"]:
                raise FactsError(f"a second 'Resuming from' line, or one after Step lines, in one start: {line!r}")
            cur["resumed_from"] = int(m.group(2))
        elif m := NONFINITE.search(line):
            cur["nonfinite"].append((int(m.group(1)), int(m.group(2))))
        elif m := STEP.search(line):
            cur["steps"].append((int(m.group(1)), int(m.group(2))))
        elif m := FINAL.search(line):
            cur["final"] = int(m.group(1))
    return starts


def kept_ranges(starts: list[dict], last_step: int) -> list[dict]:
    """The step ranges (from, to] that made the checkpoint of the last step, each with its start and dtype."""
    ranges: list[dict] = []
    seen_training = False
    for k, s in enumerate(starts):
        if s["resumed_from"] is None:
            if not s["steps"]:
                continue  # stopped before training: computed nothing
            if seen_training:
                raise FactsError(f"start {k} trained from step 0 after an earlier start had trained")
            frm = 0
        else:
            frm = s["resumed_from"]
        seen_training = True
        ranges = [r for r in ranges if r["from"] < frm]
        if ranges:
            prev = starts[ranges[-1]["start"]]
            reached = max([st for st, _ in prev["steps"]] + [prev["final"] or 0])
            if reached < frm - STEP_LINE_EVERY:
                raise FactsError(f"start {k} resumed from {frm}, but the start before it (from {ranges[-1]['from']}) "
                                 f"logged no step past {reached}")
        ranges.append({"start": k, "from": frm, "dtype": s["dtype"]})
    if not ranges:
        raise FactsError("no start of the trainer logged a step")
    last = starts[ranges[-1]["start"]]
    if last["final"] != last_step:
        raise FactsError(f"the last start's final checkpoint is {last['final']}, not {last_step}")
    for r, nxt in zip(ranges, ranges[1:] + [None]):
        r["to"] = nxt["from"] if nxt else last_step
    return ranges


def precision_and_nonfinite(starts: list[dict], last_step: int) -> tuple[dict, dict]:
    ranges = kept_ranges(starts, last_step)
    merged: list[dict] = []
    for r in ranges:
        if merged and merged[-1]["dtype"] == r["dtype"]:
            merged[-1]["to"] = r["to"]
        else:
            merged.append({"from": r["from"], "to": r["to"], "dtype": r["dtype"]})
    steps: dict[str, int] = {}
    for r in merged:
        steps[r["dtype"]] = steps.get(r["dtype"], 0) + r["to"] - r["from"]
    kept = []  # (step, cumulative skipped, dtype)
    for r in ranges:
        kept += [(st, c, r["dtype"]) for st, c in starts[r["start"]]["nonfinite"] if r["from"] < st <= r["to"]]
    last = starts[ranges[-1]["start"]]
    final_lines = [c for st, c in last["steps"] if st == last_step]
    if not final_lines:
        raise FactsError(f"no Step line of step {last_step} with a Skipped count")
    skipped = final_lines[-1]
    cumulative = max([c for _, c, _ in kept], default=0)
    if cumulative != skipped:
        raise FactsError(f"Skipped {skipped} at step {last_step}, but the NONFINITE lines of the kept steps count to "
                         f"{cumulative}")
    by_dtype: dict[str, list[int]] = {}
    for st, _, dt in kept:
        by_dtype.setdefault(dt, []).append(st)
    precision = {"ranges": merged, "steps": steps}
    nonfinite = {
        "skipped": skipped,
        "share_of_steps": skipped / last_step,
        "lines": len(kept),
        "first_step": min((st for st, _, _ in kept), default=None),
        "last_step": max((st for st, _, _ in kept), default=None),
        "lines_by_dtype": {dt: len(v) for dt, v in by_dtype.items()},
        "steps_by_dtype": {dt: [min(v), max(v)] for dt, v in by_dtype.items()},
        "discarded_lines": sum(len(s["nonfinite"]) for s in starts) - len(kept),
    }
    return precision, nonfinite


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-state", nargs="+", required=True, help="process_<i>.json of the last step")
    parser.add_argument("--trainer-logs", nargs="+", required=True, help="the trainer log of every worker")
    parser.add_argument("--last-step", type=int, required=True)
    parser.add_argument("--parameters", type=int, required=True)
    parser.add_argument("--positions-per-step", type=int, default=128 * 1024)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    try:
        states = [json.loads(Path(p).read_text()) for p in args.data_state]
        data = data_facts(states, args.last_step, args.positions_per_step, args.parameters)
        name, lines = coordinator_log({p: Path(p).read_text(errors="replace") for p in args.trainer_logs})
        starts = starts_of(lines)
        precision, nonfinite = precision_and_nonfinite(starts, args.last_step)
    except FactsError as e:
        print(f"FAILED: {e}")
        return 1
    facts = {"last_step": args.last_step, "parameters": args.parameters, "data": data, "coordinator_log": name,
             "starts": [{"dtype": s["dtype"], "resumed_from": s["resumed_from"], "final": s["final"],
                         "last_logged_step": max((st for st, _ in s["steps"]), default=None)} for s in starts],
             "precision": precision, "nonfinite": nonfinite}
    Path(args.out).write_text(json.dumps(facts, indent=1) + "\n")
    print(json.dumps({"tokens_read": data["tokens_read"], "passes": [round(data["passes_min"], 4), round(data["passes_max"], 4)],
                      "real_token_share": round(data["real_token_share"], 4), "training_flop": f"{data['training_flop']:.3e}",
                      "precision": precision, "nonfinite": {k: v for k, v in nonfinite.items()}}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
