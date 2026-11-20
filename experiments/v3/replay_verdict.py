"""The verdict of the replay test of dmbert v3 (experiments/v3/protocol.md, part 1), from the coordinator's trainer logs.

    python replay_verdict.py --replay REPLAY_W0_LOG --v2 V2_W0_LOG [--control CONTROL_W0_LOG] --out verdict.json
                             [--markdown verdict.md]

A log holds the starts of one run: each start prints "Compute dtype: ..." in its model config and, when it resumes,
"Resuming from micro_step: N". Step lines (every 50 steps, the mean loss of the last 10) carry the cumulative count
of skipped steps (apply_if_finite), NONFINITE lines its new value (every 10 steps); a resume runs from a checkpoint
and repeats the steps after it, so a later start replaces what an earlier one logged after its start step.

Criteria of protocol.md part 1, fixed before the start:
  S1 stability, over the optimizer steps 250,001-350,000: no self-stop, at most 2 skipped steps in a row, fewer than
     10 skipped steps in every window of 1,000 steps, at most 10 skipped steps in all;
  S2 loss, per block of 10,000 steps that v2 ran in float32 (250-270k, 280-320k, 340-350k): the mean logged MLM
     loss of the replay at most MARGIN above v2's at the same steps;
  S3 held-out, at 275k, 300k, 325k and 350k: the replay's EVAL loss at most MARGIN above v2's.
Passed = S1, S2 and S3. The control (plain bf16 from 250k) and the comparisons with v2's bf16 are reported only.
"""

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

START, END = 250_000, 350_000
MARGIN = 0.010
MAX_IN_A_ROW = 2
WINDOW, MAX_IN_WINDOW = 1_000, 9
MAX_TOTAL = 10
F32_BLOCKS = [(250_000, 260_000), (260_000, 270_000), (280_000, 290_000), (290_000, 300_000), (300_000, 310_000),
              (310_000, 320_000), (340_000, 350_000)]
EVAL_STEPS = [275_000, 300_000, 325_000, 350_000]
# The bf16 stretches of v2 whose batches the replay's legs repeat exactly (same resume step, so the same masks),
# up to where v2's bursts began
BF16_EXACT = [(270_000, 281_700), (320_000, 344_200)]

STEP = re.compile(r"Step +(\d+) \| Loss: (\S+) \|.*?\| Step: ([\d.]+)s.*?\| Skipped: (\d+)(?: \| Z: ([\d.naif]+))?")
EVAL = re.compile(r"EVAL step=(\d+) loss=([\d.naif]+)")
NONFINITE = re.compile(r"NONFINITE step=(\d+) skipped=(\d+) in_a_row=(\d+): (\d+) step")
GROUPS = re.compile(r"NONFINITE_GROUPS step=(\d+) logits_nonfinite=(\w+) groups=(\S*)")
SELFSTOP = re.compile(r"(\d+) steps in a row up to optimizer_step (\d+)")
DTYPE = re.compile(r"Compute dtype: (\w+)")
ATTENTION = re.compile(r"Attention logits: (\w+)")
ZLOSS = re.compile(r"Z-loss: ([\d.e+-]+)")
RESUME = re.compile(r"Resuming from micro_step: (\d+)")
PROCESS = re.compile(r"Data state restored on process (\d+) at step (\d+)")


def number(text):
    return float(text) if text not in ("nan", "inf", "-inf") else float(text)


def parse_starts(text):
    """The starts of a run in one log, in order: their settings and what each logged."""
    marks = [m.start() for m in DTYPE.finditer(text)]
    if not marks:
        raise ValueError("no 'Compute dtype:' line: not a trainer log of this code")
    starts = []
    for k, begin in enumerate(marks):
        part = text[begin: marks[k + 1] if k + 1 < len(marks) else len(text)]
        resume = RESUME.search(part)
        attention, z = ATTENTION.search(part), ZLOSS.search(part)
        starts.append({
            "dtype": DTYPE.match(part).group(1),
            "attention": attention.group(1) if attention else None,
            "z_loss": float(z.group(1)) if z else 0.0,
            "from": int(resume.group(1)) if resume else 0,
            "steps": {int(m.group(1)): {"loss": number(m.group(2)), "seconds": float(m.group(3)),
                                        "skipped": int(m.group(4)), "z": number(m.group(5)) if m.group(5) else None}
                      for m in STEP.finditer(part)},
            "evals": {int(m.group(1)): number(m.group(2)) for m in EVAL.finditer(part)},
            "nonfinite": [(int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)))
                          for m in NONFINITE.finditer(part)],
            "groups": [(int(m.group(1)), m.group(2) == "True", m.group(3)) for m in GROUPS.finditer(part)],
            "selfstop": [int(m.group(2)) for m in SELFSTOP.finditer(part)],
            "process": sorted({int(m.group(1)) for m in PROCESS.finditer(part)}),
        })
    return starts


def merged(starts):
    """One run's record over its starts: a start from step r replaces what earlier starts logged after r."""
    out = {"steps": {}, "evals": {}, "nonfinite": [], "groups": [], "selfstop": [], "starts": []}
    for s in starts:
        r = s["from"]
        out["steps"] = {k: v for k, v in out["steps"].items() if k <= r}
        out["evals"] = {k: v for k, v in out["evals"].items() if k <= r}
        out["nonfinite"] = [x for x in out["nonfinite"] if x[0] <= r]
        out["groups"] = [x for x in out["groups"] if x[0] <= r]
        out["selfstop"] = [x for x in out["selfstop"] if x <= r]
        out["steps"].update(s["steps"])
        out["evals"].update(s["evals"])
        out["nonfinite"] += s["nonfinite"]
        out["groups"] += s["groups"]
        out["selfstop"] += s["selfstop"]
        out["starts"].append({k: s[k] for k in ("from", "dtype", "attention", "z_loss", "process")})
    return out


def by_dtype(starts, dtype):
    """Step losses of the starts in one compute dtype; a later start in that dtype replaces an earlier one's steps
    after its start step, one in the other dtype does not (v2's bf16 run before a burst keeps its own steps)."""
    steps, evals = {}, {}
    for k, s in enumerate(starts):
        if s["dtype"] != dtype:
            continue
        stop = min((t["from"] for t in starts[k + 1:] if t["dtype"] == dtype), default=None)
        for step, v in s["steps"].items():
            if stop is None or step <= stop:
                steps[step] = v["loss"]
        evals.update(s["evals"])
    return steps, evals


def skipped_profile(record, start_count, lo=START, hi=END):
    """Skipped steps in (lo, hi]: total, most in a window of WINDOW steps, most in a row, from the cumulative counts."""
    points = sorted({(s, c) for s, c, _, _ in record["nonfinite"] if lo < s <= hi} |
                    {(s, v["skipped"]) for s, v in record["steps"].items() if lo < s <= hi})
    counts = [(lo, start_count)] + points

    def at(step):
        value = start_count
        for s, c in counts:
            if s > step:
                break
            value = c
        return value

    total = (counts[-1][1] if counts else start_count) - start_count
    worst, worst_at = 0, None
    for s, _ in counts:
        n = at(s) - at(s - WINDOW)
        if n > worst:
            worst, worst_at = n, s
    in_a_row = max((r for s, _, r, _ in record["nonfinite"] if lo < s <= hi), default=0)
    return {"total": total, "max_in_window": worst, "window_ending": worst_at, "max_in_a_row": in_a_row}


def count_at(record, lo):
    """The cumulative skipped count at step lo: that of the first Step line after lo less the steps that NONFINITE
    lines add up to it."""
    first = min((s for s in record["steps"] if s > lo), default=None)
    if first is None:
        raise ValueError(f"no Step line after {lo}")
    return record["steps"][first]["skipped"] - sum(n for s, _, _, n in record["nonfinite"] if lo < s <= first)


def block_means(a, b, lo, hi):
    """Mean logged loss of a and of b over the steps in (lo, hi] that both have and that are finite."""
    common = [s for s in sorted(a) if lo < s <= hi and s in b and a[s] == a[s] and b[s] == b[s]]
    if not common:
        return None
    return {"steps": len(common), "a": statistics.fmean(a[s] for s in common),
            "b": statistics.fmean(b[s] for s in common)}


def verdict(replay_text, v2_text, control_text=None):
    v2 = parse_starts(v2_text)
    v2_f32, v2_f32_evals = by_dtype(v2, "float32")
    v2_bf16, v2_bf16_evals = by_dtype(v2, "bfloat16")
    v2_evals = merged(v2)["evals"]
    starts = parse_starts(replay_text)
    rec = merged(starts)
    losses = {s: v["loss"] for s, v in rec["steps"].items()}
    start_count = count_at(rec, START)
    last = max(rec["steps"])
    s1 = skipped_profile(rec, start_count)
    s1["selfstop"] = [s for s in rec["selfstop"] if START < s <= END]
    s1["reached"] = last
    s1["ok"] = (not s1["selfstop"] and s1["max_in_a_row"] <= MAX_IN_A_ROW and s1["max_in_window"] <= MAX_IN_WINDOW
                and s1["total"] <= MAX_TOTAL and last >= END)
    s2 = []
    for lo, hi in F32_BLOCKS:
        m = block_means(losses, v2_f32, lo, hi)
        row = {"block": [lo, hi], **(m or {"steps": 0})}
        if m:
            row["difference"] = m["a"] - m["b"]
            row["ok"] = row["difference"] <= MARGIN
        else:
            row["ok"] = False
        s2.append(row)
    s3 = []
    for step in EVAL_STEPS:
        mine, theirs = rec["evals"].get(step), v2_evals.get(step)
        row = {"step": step, "replay": mine, "v2": theirs,
               "v2_dtype": "float32" if step in v2_f32_evals and v2_f32_evals[step] == theirs else "bfloat16"}
        row["ok"] = mine is not None and theirs is not None and mine - theirs <= MARGIN
        if mine is not None and theirs is not None:
            row["difference"] = mine - theirs
        s3.append(row)
    report = {
        "passed": s1["ok"] and all(r["ok"] for r in s2) and all(r["ok"] for r in s3),
        "S1_stability": s1, "S2_loss_vs_v2_float32": s2, "S3_eval_vs_v2": s3,
        "margin": MARGIN, "replay_starts": rec["starts"], "v2_starts": [{k: s[k] for k in ("from", "dtype", "process")} for s in v2],
        "skipped_at_start": start_count,
        "nonfinite_groups": rec["groups"][:50],
        "exact_batches_vs_v2_bf16": [dict(stretch=[lo, hi], **(block_means(losses, v2_bf16, lo, hi) or {"steps": 0}))
                                     for lo, hi in BF16_EXACT],
        "z_mean_per_10k": {f"{lo}-{lo + 10_000}": statistics.fmean(v["z"] for s, v in rec["steps"].items()
                                                                     if lo < s <= lo + 10_000 and v["z"] is not None)
                           for lo in range(START, END, 10_000)
                           if any(lo < s <= lo + 10_000 and v["z"] is not None for s, v in rec["steps"].items())},
        "step_seconds_median": statistics.median(v["seconds"] for s, v in rec["steps"].items() if START < s <= END),
        "v2_step_seconds_median": {d: statistics.median(v["seconds"] for st in v2 if st["dtype"] == d
                                                        for s, v in st["steps"].items() if START < s <= END)
                                   for d in ("bfloat16", "float32")},
    }
    if control_text:
        crec = merged(parse_starts(control_text))
        report["control_bf16"] = {**skipped_profile(crec, count_at(crec, START), START, max(crec["steps"])),
                                  "reached": max(crec["steps"]),
                                  "first_nonfinite": min((s for s, *_ in crec["nonfinite"] if s > START), default=None)}
    return report


def s1_so_far(text):
    """S1 over what a running replay has logged so far: the chain's guard stops the replay once it is violated."""
    rec = merged(parse_starts(text))
    if not any(s > START for s in rec["steps"]):
        return {"reached": None}
    profile = skipped_profile(rec, count_at(rec, START), START, END)
    selfstop = [s for s in rec["selfstop"] if START < s <= END]
    violated = (bool(selfstop) or profile["max_in_a_row"] > MAX_IN_A_ROW or profile["max_in_window"] > MAX_IN_WINDOW
                or profile["total"] > MAX_TOTAL)
    last = max(rec["steps"])
    return {"reached": last, **profile, "selfstop": selfstop, "violated": violated,
            "last_loss": rec["steps"][last]["loss"], "last_z": rec["steps"][last]["z"],
            "last_seconds": rec["steps"][last]["seconds"], "groups": rec["groups"][-3:]}


SMOKE_STEPS = list(range(START + 50, START + 301, 50))


def smoke_check(smoke_text, v2_text):
    """The smoke test of protocol.md part 1: fix on, 250,000 -> 250,300 on the batches of v2's float32 piece."""
    starts = parse_starts(smoke_text)
    rec = merged(starts)
    f32, _ = by_dtype(parse_starts(v2_text), "float32")
    problems, rows = [], []
    if not any(s["from"] == START for s in starts):
        problems.append(f"no resume from {START}")
    if not any(s["attention"] == "float32" and s["z_loss"] == 1e-4 for s in starts):
        problems.append("not started with float32 attention logits and z-loss 1e-4")
    for step in SMOKE_STEPS:
        mine, theirs = rec["steps"].get(step), f32.get(step)
        if mine is None or theirs is None:
            problems.append(f"no Step line {step} in {'the smoke test' if mine is None else 'v2'}")
            continue
        rows.append({"step": step, "loss": mine["loss"], "v2_float32": theirs, "difference": mine["loss"] - theirs,
                     "z": mine["z"], "seconds": mine["seconds"], "skipped": mine["skipped"]})
        if not abs(mine["loss"] - theirs) <= 0.02:
            problems.append(f"loss at {step} {mine['loss']} not within 0.02 of v2's float32 {theirs}")
        if mine["z"] is None:
            problems.append(f"no Z at {step}")
    if rows:
        mean = statistics.fmean(r["difference"] for r in rows)
        if not abs(mean) <= 0.01:
            problems.append(f"mean loss difference {mean:+.4f} not within 0.01")
        seconds = statistics.median(r["seconds"] for r in rows)
        if not seconds <= 0.35:
            problems.append(f"median step time {seconds:.3f} s above 0.35 s")
        if len({r["skipped"] for r in rows}) != 1:
            problems.append("skipped steps in the smoke test")
    if any(START < s <= START + 300 for s, *_ in rec["nonfinite"]):
        problems.append("NONFINITE lines")
    if f"Final checkpoint saved at optimizer_step {START + 300}" not in smoke_text:
        problems.append(f"no final checkpoint at {START + 300}")
    return {"ok": not problems, "problems": problems, "steps": rows,
            "mean_difference": statistics.fmean(r["difference"] for r in rows) if rows else None,
            "median_seconds": statistics.median(r["seconds"] for r in rows) if rows else None}


def _fmt(x):
    return f"{x:.4f}" if x is not None else "–"


def markdown(r):
    lines = [f"**Uitslag: {'geslaagd' if r['passed'] else 'niet geslaagd'}** (marge {r['margin']}).", ""]
    s1 = r["S1_stability"]
    lines += ["| criterium | gemeten | eis | oordeel |", "|---|---|---|---|",
              f"| S1 tot stap | {s1['reached']:,} | ≥ {END:,} | |",
              f"| S1 overgeslagen in totaal | {s1['total']} | ≤ {MAX_TOTAL} | |",
              f"| S1 meeste in {WINDOW:,} stappen | {s1['max_in_window']} (tot {s1['window_ending']}) | ≤ {MAX_IN_WINDOW} | |",
              f"| S1 meeste op rij | {s1['max_in_a_row']} | ≤ {MAX_IN_A_ROW} | |",
              f"| S1 zelfstop | {s1['selfstop'] or 'geen'} | geen | {'ja' if s1['ok'] else 'nee'} |", ""]
    lines += ["| blok (S2) | stappen | replay | v2 float32 | verschil | oordeel |", "|---|---:|---:|---:|---:|---|"]
    for row in r["S2_loss_vs_v2_float32"]:
        if row["steps"]:
            lines.append(f"| {row['block'][0]:,}–{row['block'][1]:,} | {row['steps']} | {row['a']:.4f} | {row['b']:.4f} | "
                         f"{row['difference']:+.4f} | {'ja' if row['ok'] else 'nee'} |")
        else:
            lines.append(f"| {row['block'][0]:,}–{row['block'][1]:,} | 0 | – | – | – | nee |")
    lines += ["", "| EVAL (S3) | replay | v2 | v2 in | verschil | oordeel |", "|---|---:|---:|---|---:|---|"]
    for row in r["S3_eval_vs_v2"]:
        diff = f"{row['difference']:+.4f}" if "difference" in row else "–"
        lines.append(f"| {row['step']:,} | {_fmt(row['replay'])} | {_fmt(row['v2'])} | {row['v2_dtype']} | {diff} | "
                     f"{'ja' if row['ok'] else 'nee'} |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--replay")
    parser.add_argument("--v2")
    parser.add_argument("--control")
    parser.add_argument("--out")
    parser.add_argument("--markdown")
    parser.add_argument("--s1-so-far", metavar="LOG", help="print S1 over this running log as one JSON line")
    parser.add_argument("--smoke", metavar="LOG", help="print the smoke check of this log against --v2 as JSON")
    args = parser.parse_args()
    if args.s1_so_far:
        print(json.dumps(s1_so_far(Path(args.s1_so_far).read_text(errors="replace"))))
        return 0
    if args.smoke:
        r = smoke_check(Path(args.smoke).read_text(errors="replace"), Path(args.v2).read_text(errors="replace"))
        print(json.dumps(r))
        return 0 if r["ok"] else 1
    if not (args.replay and args.v2 and args.out):
        parser.error("--replay, --v2 and --out are needed for the verdict")
    texts = {name: Path(path).read_text(errors="replace") if path else None
             for name, path in (("replay", args.replay), ("v2", args.v2), ("control", args.control))}
    r = verdict(texts["replay"], texts["v2"], texts["control"])
    Path(args.out).write_text(json.dumps(r, indent=1) + "\n")
    if args.markdown:
        Path(args.markdown).write_text(markdown(r))
    print(markdown(r))
    print("PASSED" if r["passed"] else "NOT PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
