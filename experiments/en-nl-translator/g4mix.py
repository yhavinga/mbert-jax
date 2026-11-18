#!/usr/bin/env python3
"""Arm G4-5 (draft amendment G4-5 of protocol-g4.md; Yeb, 17 Nov): one LoRA per source, branched from a shared warmup,
combined as W0 + Delta_s + sum_i w_i (Delta_i - Delta_s). numpy and the standard library only (no JAX), so that the
arithmetic is tested on any machine; sft_lora.py --combine and g4arm.py use it on the TPU workers.

  stack   one adapter of rank sum(r_k) whose update is sum_k c_k * update_k: the A's side by side along the rank, each B
          times its coefficient (float32); with the shared warmup first, c = (shared - sum(w), w_1, ..., w_N)
  spec    the search's specification for sft_lora.py --combine (components, objective, coordinate search, top K, the
          tolerance of the repeated first point, and with --warmup-check the check of the merged warmup before it)

    python g4mix.py stack --component shared=DIR --component bt_raad=DIR ... --weights 0.5,0.25,... --out DIR
    python g4mix.py stack --component ... --from-summary summary.json --candidate 1 --out DIR
    python g4mix.py spec --component shared=FILE.npz ... --objective wmt24pp_s=1 ... --out spec.json

A component DIR holds adapter.npz and adapter.json of sft_lora.py (all of one rank, alpha and scale).
"""

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for _path in (HERE.parent / "flan-nl", HERE):
    sys.path.insert(0, str(_path))

import translate  # noqa: E402  (lora_modules, lora_delta: the merge that translate.py --lora does)

SHARED = "shared"
DECIMALS = 6            # a point of the search is its weights rounded to this many decimals (the cache key)
# addendum G4-5.7b (Yeb, 18 Nov): the repeated first point is "equal" within this; float32 rounding of a broadcast
# value is ~1e-7, a rest of an earlier point after a failed restore of the base kernels would be >= 1e-2
REPEAT_MAX_DIFF = 1e-4
# addendum G4-5.7b A: the search's objective, the loss at T = 1 or at the point's best T (calibrated_objective)
OBJECTIVE_MODES = ("t1", "calibrated")


# ---------------------------------------------------------------------------------------------------- stacking

def coefficients(weights, shared=1.0):
    """The coefficient of each component, shared warmup first: W0 + shared * Delta_s + sum_i w_i (Delta_i - Delta_s)
    = W0 + (shared - sum w) Delta_s + sum_i w_i Delta_i"""
    weights = [float(w) for w in weights]
    return [float(shared) - sum(weights)] + weights


def module_factors(components):
    """[{module: (a, b)}] of the components, checked: the same modules, the same shapes, one rank per component"""
    if not components:
        raise ValueError("no component")
    mods = [translate.lora_modules(c) for c in components]
    names = set(mods[0])
    for k, m in enumerate(mods[1:], 1):
        if set(m) != names:
            raise ValueError(f"component {k} has other modules than component 0: "
                             f"{sorted(set(m) ^ names)[:3]}")
        for module, (a, b) in m.items():
            a0, b0 = mods[0][module]
            if a.shape != a0.shape or b.shape != b0.shape:
                raise ValueError(f"{module}: component {k} has shapes {a.shape} {b.shape}, component 0 {a0.shape} "
                                 f"{b0.shape}")
    return mods


def stack_module(factors, coefs):
    """(A, B) of one module: the components' A side by side along the rank (last axis), their B one under the other
    (first axis), each B times its coefficient in float32"""
    if len(factors) != len(coefs):
        raise ValueError(f"{len(factors)} components, {len(coefs)} coefficients")
    a = np.concatenate([np.asarray(f[0], np.float32) for f in factors], axis=-1)
    b = np.concatenate([np.asarray(f[1], np.float32) * np.float32(c) for f, c in zip(factors, coefs)], axis=0)
    return a, b


def stack(components, coefs):
    """{key: array}: an adapter of sft_lora.py's keys whose update (translate.lora_delta) is sum_k coefs[k] times the
    update of components[k], exactly in real arithmetic (one product over the stacked rank)"""
    mods = module_factors(components)
    if len(coefs) != len(mods):
        raise ValueError(f"{len(mods)} components, {len(coefs)} coefficients")
    out = {}
    for module in sorted(mods[0]):
        a, b = stack_module([m[module] for m in mods], coefs)
        out[f"{module}/kernel_lora_a"] = a
        out[f"{module}/kernel_lora_b"] = b
    return out


def stacked_meta(metas, names, sources, coefs, weights, shared):
    """adapter.json of a stacked adapter: rank the sum, scale the components' one (translate.py takes meta['scale']),
    alpha = scale * rank; the components with their coefficients"""
    scales = {float(m["scale"]) for m in metas}
    ranks = [int(m["rank"]) for m in metas]
    if len(scales) != 1:
        raise ValueError(f"the components differ in scale: {sorted(scales)}")
    for m in metas:
        if abs(float(m["alpha"]) / int(m["rank"]) - float(m["scale"])) > 1e-12:
            raise ValueError(f"an adapter.json with alpha / rank != scale: {m}")
        if "lora_b_factor" in m:
            raise ValueError("a component that is itself scaled (lora_b_factor): stack the source adapters")
    scale = scales.pop()
    rank = sum(ranks)
    return {"rank": rank, "alpha": scale * rank, "scale": scale, "stacked": True,
            "shared_coef": float(shared), "weights": {n: float(w) for n, w in zip(names[1:], weights)},
            "components": [{"name": n, "source": s, "rank": r, "coef": float(c), "steps": m.get("steps"),
                            "train_files": m.get("train_files")}
                           for n, s, r, c, m in zip(names, sources, ranks, coefs, metas)],
            "model": metas[0].get("model"), "checkpoint": metas[0].get("checkpoint"),
            "module_path": metas[0].get("module_path")}


def read_adapter(directory):
    d = Path(directory)
    meta = json.loads((d / "adapter.json").read_text())
    with np.load(d / "adapter.npz") as npz:
        arrays = {k: npz[k] for k in npz.files}
    return arrays, meta


# ---------------------------------------------------------------------------------------------------- the search

def objective_value(records, weights):
    """The search's objective: the weighted mean of the token-weighted losses of the objective's sets"""
    missing = [s for s in weights if s not in records]
    if missing:
        raise ValueError(f"no loss of {missing}")
    total = sum(weights.values())
    if total <= 0:
        raise ValueError(f"objective weights {weights}")
    return sum(w * records[s] for s, w in weights.items()) / total


def calibrated_objective(curves, weights):
    """Addendum G4-5.7b A: the objective at the point's best temperature, (value, T): the lowest weighted mean of the
    objective's sets over their common grid of T (curves {set: {"t": [...], "loss": [...]}}; equal values: the T
    nearest to 1)"""
    missing = [s for s in weights if s not in curves]
    if missing:
        raise ValueError(f"no temperature curve of {missing}")
    grid = curves[next(iter(weights))]["t"]
    if any(curves[s]["t"] != grid for s in weights):
        raise ValueError("the objective's sets have different grids of T")
    values = [objective_value({s: curves[s]["loss"][k] for s in weights}, weights) for k in range(len(grid))]
    k = min(range(len(grid)), key=lambda i: (values[i], abs(grid[i] - 1.0)))
    return values[k], grid[k]


def key_of(w):
    return tuple(round(float(x), DECIMALS) for x in w)


class Budget(Exception):
    pass


class Evaluator:
    """Each distinct point once, in order, up to `budget` evaluations; the points as evaluated"""

    def __init__(self, objective, budget):
        self.objective, self.budget = objective, budget
        self.cache, self.points = {}, []

    def __call__(self, w, phase):
        key = key_of(w)
        if key in self.cache:
            return self.cache[key]
        if len(self.points) >= self.budget:
            raise Budget()
        value = float(self.objective(list(key)))
        if not np.isfinite(value):
            raise ValueError(f"objective {value} at {key}")
        self.cache[key] = value
        self.points.append({"order": len(self.points), "w": list(key), "value": value, "phase": phase})
        return value


def anchors(n, uniform=True):
    """The points evaluated before the search: the warmup alone (all 0), each branch alone (its weight 1), and the
    uniform mean of the branches (1/N each)"""
    out = [("warmup", [0.0] * n)]
    out += [(f"alone/{i}", [1.0 if j == i else 0.0 for j in range(n)]) for i in range(n)]
    if uniform and n > 1:
        out.append(("uniform", [1.0 / n] * n))
    return out


def coordinate_search(evaluate, start, step, min_step, lower, upper):
    """Deterministic coordinate search: from the best point, each coordinate +h then -h (clipped to [lower, upper]);
    a strict improvement moves there and goes on with the next coordinate; a sweep without one halves h; stops below
    min_step or when the evaluator's budget is spent. Returns (best, value, reason)."""
    if not 0 < min_step <= step:
        raise ValueError(f"step {step}, min_step {min_step}")
    best = [min(max(float(x), lower), upper) for x in start]
    value = evaluate(best, "start")
    h = step
    try:
        while h >= min_step - 1e-12:
            improved = False
            for i in range(len(best)):
                for sign in (1, -1):
                    cand = list(best)
                    cand[i] = min(max(cand[i] + sign * h, lower), upper)
                    if key_of(cand) == key_of(best):
                        continue
                    v = evaluate(cand, f"h={h:g}")
                    if v < value:
                        best, value, improved = cand, v, True
                        break
            if not improved:
                h /= 2
    except Budget:
        return key_of(best), value, "budget"
    return key_of(best), value, "converged"


def top_candidates(points, k, min_distance):
    """The k best points (lowest value, then earliest), each at least min_distance (largest coordinate difference)
    from the ones chosen before it"""
    chosen = []
    for p in sorted(points, key=lambda p: (p["value"], p["order"])):
        if all(max(abs(a - b) for a, b in zip(p["w"], c["w"])) >= min_distance - 1e-12 for c in chosen):
            chosen.append(p)
        if len(chosen) == k:
            break
    return chosen


def run_search(objective, spec):
    """The anchors, the coordinate search from spec['start'], the top K; every evaluation within spec['budget']"""
    s = spec["search"]
    n = len(spec["components"]) - 1
    ev = Evaluator(objective, s["budget"])
    for name, w in anchors(n, s.get("uniform_anchor", True)):
        try:
            ev(w, f"anchor/{name}")
        except Budget:
            raise ValueError(f"budget {s['budget']} below the {len(anchors(n))} anchors") from None
    start = s["start"] if s["start"] != "uniform" else [1.0 / n] * n
    if len(start) != n:
        raise ValueError(f"start has {len(start)} weights, {n} branches")
    best, value, reason = coordinate_search(ev, start, s["step"], s["min_step"], s["lower"], s["upper"])
    top = top_candidates(ev.points, s["top_k"], s["min_distance"])
    return {"points": ev.points, "best": {"w": list(best), "value": value}, "stopped": reason,
            "evaluations": len(ev.points), "top": top}


# ---------------------------------------------------------------------------------------------------- spec, split

def spec_dict(components, objective, *, shared=1.0, start="uniform", step=0.25, min_step=0.0625, lower=0.0,
              upper=1.5, budget=80, top_k=4, min_distance=0.125, uniform_anchor=True, scale=None,
              repeat_max_diff=REPEAT_MAX_DIFF, warmup_check=None, objective_rule=None):
    names = [n for n, _ in components]
    if not names or names[0] != SHARED:
        raise ValueError(f"the first component is the shared warmup ('{SHARED}'), not {names[:1]}")
    if len(set(names)) != len(names):
        raise ValueError(f"a component name twice: {names}")
    if not objective or any(w <= 0 for w in objective.values()):
        raise ValueError(f"objective {objective}")
    if not lower <= upper:
        raise ValueError(f"bounds {lower} {upper}")
    if not repeat_max_diff >= 0:
        raise ValueError(f"repeat_max_diff {repeat_max_diff}")
    mode = "t1"
    if objective_rule is not None:
        # addendum G4-5.7b A: the rule fixed before the search chooses the objective (temp5's search_rule)
        if sorted(objective_rule.get("sets") or []) != sorted(objective) or objective_rule.get("objective") not in OBJECTIVE_MODES:
            raise ValueError(f"the objective rule {objective_rule} is not one of this objective {sorted(objective)}")
        mode = objective_rule["objective"]
    out = {"components": [{"name": n, "adapter": str(p)} for n, p in components], "objective": dict(objective),
           "objective_mode": mode, "objective_rule": objective_rule,
           "shared_coef": float(shared), "scale": scale, "repeat_max_diff": float(repeat_max_diff),
           "search": {"start": start, "step": step, "min_step": min_step, "lower": lower, "upper": upper,
                      "budget": budget, "top_k": top_k, "min_distance": min_distance,
                      "uniform_anchor": uniform_anchor}}
    if warmup_check is not None:
        out["warmup_check"] = warmup_check
    return out


def warmup_check_of(summary, sets, max_diff, source=None):
    """Addendum G4-5.7b: what the merged warmup alone (w = 0, c_s = 1) must give before the search goes on: the warmup
    run's own eval at its last step on these sets (loss and target tokens, so that the batches are the same)"""
    step = summary["steps"]
    got = {e["set"]: e for e in summary.get("eval") or [] if e["step"] == step}
    missing = [s for s in sets if s not in got]
    if missing or not sets:
        raise ValueError(f"the warmup has no eval of {missing or sets} at its last step {step}")
    if not max_diff > 0:
        raise ValueError(f"max_diff {max_diff}")
    return {"step": step, "losses": {s: got[s]["loss"] for s in sets}, "tokens": {s: got[s]["tokens"] for s in sets},
            "max_diff": float(max_diff), "source": source}


def warmup_check_result(check, losses, tokens):
    """The merged warmup's losses and target tokens against the check: holds if the tokens are the same and every loss
    is within max_diff"""
    diff = {s: abs(losses[s] - want) for s, want in check["losses"].items()}
    tokens_equal = all(tokens[s] == want for s, want in check["tokens"].items())
    return {"step": check["step"], "want": dict(check["losses"]), "got": {s: losses[s] for s in check["losses"]},
            "diff": diff, "tokens_equal": tokens_equal, "max_diff": check["max_diff"],
            "holds": tokens_equal and max(diff.values()) <= check["max_diff"]}


def repeat_result(w, first, again, max_diff):
    """The first point of the search evaluated again after the top K: equal within max_diff (the base came back)"""
    diff = abs(again - first)
    return {"w": list(w), "first": first, "again": again, "diff": diff, "max_diff": max_diff, "equal": diff <= max_diff}


def stratum_of(row):
    """The stratum a segment is split in: its domain where the set has one (WMT24++, FLORES), so that both parts keep
    the set's mix of domains"""
    meta = json.loads(row["meta"]) if row.get("meta") else {}
    return f"{row['set']}/{meta.get('domain', '')}" if row["set"] in ("wmt24pp", "flores") else row["set"]


def unit_of(row):
    """The unit a segment is split by: its article for FLORES (URL, a few sentences each), else the segment itself
    (WMT24++'s documents are too few and too uneven per domain to split by: 8 literary documents hold 206 segments)"""
    if row["set"] == "flores":
        meta = json.loads(row["meta"]) if row.get("meta") else {}
        return f"flores/{meta['URL']}"
    return f"{row['set']}/{row['id']}"


def split_set(rows, seed, share, forced_decision=()):
    """{id: 'search' | 'decision'} of one set's segments: per stratum (stratum_of) the units (unit_of) with a segment in
    forced_decision go to the decision part; the others in a seeded order (seed/<stratum>), each to the search part if
    that brings the stratum's search count nearer to share x the stratum's segments"""
    forced = set(forced_decision)
    strata = {}
    for r in sorted(rows, key=lambda r: r["id"]):
        strata.setdefault(stratum_of(r), {}).setdefault(unit_of(r), []).append(r["id"])
    out = {}
    for stratum, units in sorted(strata.items()):
        free = sorted(u for u, ids in units.items() if not forced & set(ids))
        random.Random(f"{seed}/{stratum}").shuffle(free)
        target = share * sum(map(len, units.values()))
        n = 0
        for u in free:
            size = len(units[u])
            if abs(n + size - target) < abs(n - target):
                out.update((i, "search") for i in units[u])
                n += size
        for ids in units.values():
            for i in ids:
                out.setdefault(i, "decision")
    return out


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


# ---------------------------------------------------------------------------------------------------- CLI

def parse_named(specs, what):
    out = []
    for spec in specs:
        name, sep, path = spec.partition("=")
        if not sep or not name or not path:
            raise SystemExit(f"--{what} {spec!r}: NAME=PATH")
        out.append((name, path))
    return out


def stack_cli(args):
    comps = parse_named(args.component, "component")
    if comps[0][0] != SHARED:
        raise SystemExit(f"the first --component is {SHARED}=DIR, not {comps[0][0]}")
    if args.from_summary:
        combine = json.loads(Path(args.from_summary).read_text()).get("combine")
        if not combine:
            raise SystemExit(f"{args.from_summary}: no combine result")
        names = [c["name"] for c in combine["spec"]["components"]]
        if names != [n for n, _ in comps]:
            raise SystemExit(f"the components {[n for n, _ in comps]} are not the search's {names}")
        top = combine["search"]["top"]
        if not 1 <= args.candidate <= len(top):
            raise SystemExit(f"--candidate {args.candidate}: the search has {len(top)}")
        weights, shared = top[args.candidate - 1]["w"], combine["spec"]["shared_coef"]
    else:
        weights, shared = [float(x) for x in args.weights.split(",")], args.shared_coef
    if len(weights) != len(comps) - 1:
        raise SystemExit(f"{len(weights)} weights for {len(comps) - 1} branches")
    out = Path(args.out)
    for name in ("adapter.npz", "adapter.json"):
        if (out / name).exists():
            raise SystemExit(f"{out / name} exists")
    arrays, metas = [], []
    for _, d in comps:
        a, m = read_adapter(d)
        arrays.append(a)
        metas.append(m)
    coefs = coefficients(weights, shared)
    stacked = stack(arrays, coefs)
    meta = stacked_meta(metas, [n for n, _ in comps], [d for _, d in comps], coefs, weights, shared)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "adapter.npz", **stacked)
    (out / "adapter.json").write_text(json.dumps(meta, indent=1) + "\n")
    print("STACKED " + json.dumps({"out": str(out), "rank": meta["rank"], "scale": meta["scale"],
                                   "coefs": dict(zip([n for n, _ in comps], coefs)),
                                   "modules": sum(k.endswith("/kernel_lora_b") for k in stacked)}), flush=True)


def spec_cli(args):
    comps = parse_named(args.component, "component")
    objective = {n: float(w) for n, w in parse_named(args.objective, "objective")}
    start = "uniform" if args.start == "uniform" else [float(x) for x in args.start.split(",")]
    check = None
    if args.warmup_check:
        check = warmup_check_of(json.loads(Path(args.warmup_check).read_text()), args.check_set,
                                args.check_max_diff, source=args.warmup_check)
    elif args.check_set:
        raise SystemExit("--check-set goes with --warmup-check")
    rule = None
    if args.objective_rule:
        rule = json.loads(Path(args.objective_rule).read_text()).get("search_rule")
        if not rule:
            raise SystemExit(f"{args.objective_rule}: no search_rule (g4arm.py temp5)")
        rule = {**rule, "source": args.objective_rule}
    spec = spec_dict(comps, objective, shared=args.shared_coef, start=start, step=args.step, min_step=args.min_step,
                     lower=args.lower, upper=args.upper, budget=args.budget, top_k=args.top_k,
                     min_distance=args.min_distance, scale=args.scale, repeat_max_diff=args.repeat_max_diff,
                     warmup_check=check, objective_rule=rule)
    Path(args.out).write_text(json.dumps(spec, indent=1) + "\n")
    print(f"SPEC {json.dumps(spec)}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("stack")
    p.add_argument("--component", action="append", required=True, help="NAME=DIR, the shared warmup first")
    p.add_argument("--weights", help="w_1,...,w_N of the branches in --component order")
    p.add_argument("--shared-coef", type=float, default=1.0)
    p.add_argument("--from-summary", help="summary.json of the search run (sft_lora.py --combine)")
    p.add_argument("--candidate", type=int, default=1, help="with --from-summary: the k-th of the search's top K")
    p.add_argument("--out", required=True)
    p = sub.add_parser("spec")
    p.add_argument("--component", action="append", required=True, help="NAME=adapter.npz, the shared warmup first")
    p.add_argument("--objective", action="append", required=True, help="EVAL_NAME=WEIGHT")
    p.add_argument("--shared-coef", type=float, default=1.0)
    p.add_argument("--start", default="uniform", help="'uniform' or w_1,...,w_N")
    p.add_argument("--step", type=float, default=0.25)
    p.add_argument("--min-step", type=float, default=0.0625)
    p.add_argument("--lower", type=float, default=0.0)
    p.add_argument("--upper", type=float, default=1.5)
    p.add_argument("--budget", type=int, default=80)
    p.add_argument("--top-k", type=int, default=4)
    p.add_argument("--min-distance", type=float, default=0.125)
    p.add_argument("--scale", type=float, default=0.5, help="the components' LoRA scale (alpha / rank)")
    p.add_argument("--repeat-max-diff", type=float, default=REPEAT_MAX_DIFF,
                   help="the repeated first point is equal within this (addendum G4-5.7b)")
    p.add_argument("--warmup-check", default=None,
                   help="summary.json of the warmup run: the merged warmup alone must give its own eval at its last "
                        "step on the --check-set sets before the search goes on (addendum G4-5.7b)")
    p.add_argument("--check-set", action="append", default=[], help="an eval set of the warmup check (repeatable)")
    p.add_argument("--check-max-diff", type=float, default=0.02)
    p.add_argument("--objective-rule", default=None,
                   help="temp5's json: its search_rule chooses the objective, t1 or calibrated (addendum G4-5.7b A)")
    p.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if args.command == "stack":
        if bool(args.weights) == bool(args.from_summary):
            raise SystemExit("stack takes --weights or --from-summary")
        stack_cli(args)
    else:
        spec_cli(args)


if __name__ == "__main__":
    main()
