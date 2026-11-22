#!/usr/bin/env python3
"""Meting G4-bon (protocol-g4.md §15, 22 Nov 2026): the ceiling of best-of-8 when the choice among a model's 8 samples
at T = 1 is made on C2 (G4-7's reward classifier) and MetricX-24-QE instead of on QE alone. A measurement on the search
part of G4-5.4 only (WMT24++ without is_bad_source and FLAN); it decides nothing. C1 (the classifier of rule (b)) scores
the chosen outputs and never chooses.

Models: G4-0 (its 8 samples of the sampling probe, $G/sampling/G4-0/t10s1..t10s8, with C2's scores of c2cal7) and the
candidate chosen by G4-7 (8 new samples of the search part with the probe's settings, scored the same way).

Selections among the 8 samples of a source (fixed in §15 before the first step):
  qe   the lowest QE (ties: the earlier sample), the sampling probe's best8
  r7   the highest reward of G4-7 (g4rl.reward7: clip(logit C2, -12, 12) - 2 max(0, QE - QE0) - 2 max(0, |ln l/l0| -
       ln 1.15), a fail -30), the anchor G4-0's greedy translation (QE0, l0); ties: lower QE, then the earlier sample
  c2q  the highest clipped logit of C2 among the samples that are no fail of G4-7 and have QE <= QE0; none: qe's choice
  c2   the highest clipped logit of C2 among the samples that are no fail of G4-7; none: qe's choice (the variant of §15)

    python g4bon.py bondata --evalsets E --split S --segments DIR --out DIR
    python g4bon.py select --model G4-0 --sample t10s1=DIR ... --anchor DIR --c2-predictions P --c2-texts T \\
        --evalsets E --split S [--best8 choice.parquet] [--probe-report report.json] --out DIR
    python g4bon.py c2texts --model M --sample t10s1=GLOB ... --greedy GLOB --g40 GLOB --evalsets E --split S \\
        --checkpoint DIR --out texts.parquet
    python g4bon.py texts --pairs NAME=GLOB ... --evalsets E --split S --out texts.parquet
    python g4bon.py report --evalsets E --split S --refqe R --anchor DIR --predictions P --probe-predictions PP \\
        --model NAME=GREEDY_DIR,T10S1_DIR,SELECT_DIR ... [--c2-check NAME=PREDICTIONS,CAL7_PREDICTIONS] --out report.md

Pure Python and numpy; no model, no TPU (C2's token counter in c2texts loads a tokenizer)."""

import argparse
import collections
import glob
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for _path in (HERE.parent / "flan-nl", HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import armeval  # noqa: E402
import g4arm  # noqa: E402
import g4rl  # noqa: E402

PART = "search"                                  # G4-5.4's search part; the decision part stays unread
SETS = ("wmt24pp", "flan")                       # rule (b)'s sets
N_SEARCH = {"wmt24pp": 480, "flan": 500}         # split.json: WMT24++ good 480 of 497, FLAN 500 [measured 22 Nov]
SAMPLES = tuple((f"t10s{k}", 1.0, k) for k in range(1, 9))   # the sampling probe's T = 1 settings (name, T, seed)
SAMPLE_NAMES = tuple(n for n, _, _ in SAMPLES)
SELECTIONS = ("qe", "r7", "c2q", "c2")
BASELINES = ("greedy", "t10s1")
OUTPUTS = BASELINES + SELECTIONS
BASE = g4arm.BASE                                # G4-0
C2_MAX_TOKENS = 1022                             # G4-7.0 (Yeb 21 Nov): more is c2_overflow (g4c2.C2_MAX_TOKENS)
LOGIT_CLIP = g4rl.LOGIT_CLIP                     # 12 (G4-7.4e)
QE_TOL = 1e-9
BOOT = 10000                                     # paired bootstrap resamples per comparison
BOOT_SEED = "en-nl-g4arm/0/G4-bon/boot"
HACK_C2, HACK_C1 = 0.10, 0.10                    # report7's flag: C2 up >= 0.10 while C1 up < 0.10 (G4-7.4c)
# what each output's choice used: a measure listed here is not an independent test of that output (§15)
SELECTION_MEASURES = {
    "greedy": [], "t10s1": [],
    "qe": ["QE", "MetricX-ref (zelfde model als QE)"],
    "r7": ["C2", "QE", "MetricX-ref (zelfde model als QE)", "lengte", "fail-vlaggen"],
    "c2q": ["C2", "QE", "MetricX-ref (zelfde model als QE)", "fail-vlaggen"],
    "c2": ["C2", "fail-vlaggen (QE > 7,5 en lengteband)"],
}
LABELS = {"greedy": "greedy", "t10s1": "één sample (T = 1, seed 1)", "qe": "best-of-8: laagste QE",
          "r7": "best-of-8: hoogste reward van G4-7", "c2q": "best-of-8: hoogste C2 met QE ≤ QE₀",
          "c2": "best-of-8: hoogste C2 (geen QE-grens)"}


# ---------------------------------------------------------------------------------------------------- the search part

def search_ids(evalsets, split):
    """{set: sorted ids} of the search part of G4-5.4 on rule (b)'s sets: WMT24++ without is_bad_source, FLAN. The
    assignment must be the one of its digest and that digest G4-5.4's (g4arm.SPLIT_DIGEST6C)."""
    import g4mix

    if g4mix.digest(split["assign"]) != split["digest"] or split["digest"] != g4arm.SPLIT_DIGEST6C:
        raise SystemExit(f"the split is not G4-5.4's ({split['digest'][:12]}…, expected {g4arm.SPLIT_DIGEST6C[:12]}…)")
    ids = g4arm.part_ids5(evalsets, split["assign"], PART)
    out = {s: ids[s] for s in SETS}
    if {s: len(v) for s, v in out.items()} != N_SEARCH:
        raise SystemExit(f"the search part has {({s: len(v) for s, v in out.items()})} segments, expected {N_SEARCH}")
    return out


def load_evalsets(path):
    return {r["id"]: r for r in g4arm.read_rows(path, ["id", "set", "meta"])}


def load_split(path):
    return json.loads(Path(path).read_text())


def bondata(args):
    """The sampling probe's segments (segments-w<k>.parquet, classes-w<k>.txt; G4-5.7b B) kept to the search part,
    each on its worker as there; every search segment exactly once"""
    split = load_split(args.split)
    if g4arm.sha256_file(args.evalsets) != split["inputs"]["evalsets"]:
        raise SystemExit(f"{args.evalsets} is not the evalsets.parquet of the split ({split['inputs']['evalsets'][:12]}…)")
    want = search_ids(load_evalsets(args.evalsets), split)
    keep = {i for v in want.values() for i in v}
    src, out = Path(args.segments), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seen, per = collections.Counter(), {}
    for k in range(args.workers):
        mine = [r for r in g4arm.read_rows(src / f"segments-w{k}.parquet") if r["id"] in keep]
        g4arm.write_parquet(mine, out / f"segments-w{k}.parquet")
        (out / f"classes-w{k}.txt").write_text((src / f"classes-w{k}.txt").read_text())
        seen.update(r["id"] for r in mine)
        per[f"w{k}"] = len(mine)
    if set(seen) != keep or max(seen.values()) != 1:
        raise SystemExit(f"the probe's segments hold {len(seen)} of the {len(keep)} search segments (or one twice)")
    print(f"BONDATA {json.dumps({**{s: len(v) for s, v in want.items()}, 'workers': per})}", flush=True)


# ---------------------------------------------------------------------------------------------------- C2

def logit_of(p):
    """clip(logit, -12, 12) from P(original) as g4c2.logit_of (c2cal7): p 0 or 1 clip to the bound"""
    if p <= 0.0:
        return -LOGIT_CLIP
    if p >= 1.0:
        return LOGIT_CLIP
    return max(-LOGIT_CLIP, min(LOGIT_CLIP, math.log(p) - math.log1p(-p)))


def c2_scores(predictions, texts, names, ids):
    """{name: {id: {"logit_c2", "c2_overflow"}}} from a classify run of C2 (predictions: model, set, id, p_original)
    and its texts (model, set, id, tokens): every (name, id) exactly once; a text of more than 1,022 tokens of C2's
    tokenizer is c2_overflow with no logit, as the judge's rows (G4-7.4a)"""
    want = {(n, i) for n in names for i in ids}
    tokens = {}
    for r in texts:
        key = (r["model"], r["id"])
        if key in want:
            if key in tokens:
                raise SystemExit(f"C2's texts: {key} twice")
            tokens[key] = r["tokens"]
    out = {n: {} for n in names}
    for r in predictions:
        key = (r["model"], r["id"])
        if key not in want:
            continue
        if r["id"] in out[r["model"]]:
            raise SystemExit(f"C2's predictions: {key} twice")
        if key not in tokens:
            raise SystemExit(f"C2's predictions: {key} has no row in C2's texts")
        over = tokens[key] > C2_MAX_TOKENS
        out[r["model"]][r["id"]] = {"logit_c2": None if over else logit_of(r["p_original"]), "c2_overflow": over}
    missing = [(n, i) for n, i in sorted(want) if i not in out[n]]
    if missing:
        raise SystemExit(f"C2 scored {len(want) - len(missing)} of the {len(want)} texts, e.g. missing {missing[:3]}")
    return out


# ---------------------------------------------------------------------------------------------------- the choice

def anchor_of(row):
    """G4-7's anchor of an evaluation segment: G4-0's greedy translation, l0 its tokens_out, qe0 its QE"""
    if row["metricx_qe"] is None or not row["tokens_out"]:
        raise SystemExit(f"{row['id']}: G4-0's greedy translation has no QE or no tokens")
    return {"l0": int(row["tokens_out"]), "qe0": float(row["metricx_qe"])}


def sample_view(row):
    return {"uid": row["id"], "tokens_out": int(row["tokens_out"]), "ended": bool(row["ended"]),
            "decode_loop": bool(row["decode_loop"]), "empty": bool(row["empty"]),
            "english_left": bool(row["english_left"])}


def scored_samples(rows, c2, anchor):
    """Per sample of one source (rows: the 8 scored rows in SAMPLE_NAMES's order): QE, the clipped logit, G4-7's
    reward and its fail reasons (g4rl.reward7, unchanged)"""
    out = []
    for row, c in zip(rows, c2):
        r = g4rl.reward7(sample_view(row), row["metricx_qe"], c, anchor)
        out.append({"qe": row["metricx_qe"], "logit": r["logit"], "reward": r["reward"], "fail": r["fail"],
                    "ratio": r["ratio"]})
    return out


def _qe_key(s):
    return math.inf if s["qe"] is None else s["qe"]


def choose(samples, qe0):
    """{selection: (index, fallback)} among one source's scored samples (§15): qe, r7, c2q, c2"""
    idx = range(len(samples))
    qe = min(idx, key=lambda k: (_qe_key(samples[k]), k))
    r7 = min(idx, key=lambda k: (-samples[k]["reward"], _qe_key(samples[k]), k))
    ok = [k for k in idx if not samples[k]["fail"]]
    bound = [k for k in ok if samples[k]["qe"] is not None and samples[k]["qe"] <= qe0 + QE_TOL]

    def best_c2(cands):
        return min(cands, key=lambda k: (-samples[k]["logit"], _qe_key(samples[k]), k))

    return {"qe": (qe, False), "r7": (r7, False),
            "c2q": (best_c2(bound), False) if bound else (qe, True),
            "c2": (best_c2(ok), False) if ok else (qe, True)}, len(bound)


def read_sample_dir(d):
    """A sample's raw files (pairs/pairs-w*.parquet, scores/metricx-w*.parquet, scores/comet-w*.parquet) by id"""
    comet = f"{d}/scores/comet-w*.parquet"
    return {"pairs": armeval.read_rows(f"{d}/pairs/pairs-w*.parquet"),
            "metricx": armeval.read_rows(f"{d}/scores/metricx-w*.parquet"),
            "comet": armeval.read_rows(comet) if glob.glob(comet) else {}}


def check_probe_settings(report):
    """The probe's T = 1 settings are SAMPLES (t10s<k> at T 1.0 with seed k)"""
    got = {n: (float(t), int(seed)) for n, t, seed in report["settings"]["settings"]}
    bad = [n for n, t, seed in SAMPLES if got.get(n) != (t, seed)]
    if bad:
        raise SystemExit(f"the probe's settings differ for {bad}: {({n: got.get(n) for n in bad})}")


def select(args):
    evalsets = load_evalsets(args.evalsets)
    ids = search_ids(evalsets, load_split(args.split))
    all_ids = [i for s in SETS for i in ids[s]]
    set_of = {i: s for s in SETS for i in ids[s]}
    runs = [x.split("=", 1) for x in args.sample]
    if [n for n, _ in runs] != list(SAMPLE_NAMES):
        raise SystemExit(f"--sample: {[n for n, _ in runs]}, expected {list(SAMPLE_NAMES)} in this order")
    if args.probe_report:
        check_probe_settings(json.loads(Path(args.probe_report).read_text()))
    raw = {n: read_sample_dir(d) for n, d in runs}
    scored = {n: armeval.scored(d) for n, d in runs}
    for n in SAMPLE_NAMES:
        missing = [i for i in all_ids if i not in scored[n]]
        if missing:
            raise SystemExit(f"{n}: no scored sample of {len(missing)} search segments, e.g. {missing[:3]}")
    anchor_rows = armeval.scored(args.anchor)
    anchors = {i: anchor_of(anchor_rows[i]) for i in all_ids}
    c2 = c2_scores(g4arm.read_rows(args.c2_predictions, ["model", "set", "id", "p_original"]),
                   g4arm.read_rows(args.c2_texts, ["model", "set", "id", "tokens"]),
                   [args.model] + [f"{args.model}/{n}" for n in SAMPLE_NAMES], all_ids)
    out = Path(args.out)
    chosen_rows = {s: {"pairs": [], "metricx": [], "comet": []} for s in SELECTIONS}
    choice = []
    for i in all_ids:
        rows = [scored[n][i] for n in SAMPLE_NAMES]
        cs = scored_samples(rows, [c2[f"{args.model}/{n}"][i] for n in SAMPLE_NAMES], anchors[i])
        picks, n_bound = choose(cs, anchors[i]["qe0"])
        g = c2[args.model][i]
        rec = {"id": i, "set": set_of[i], "qe0": anchors[i]["qe0"], "l0": anchors[i]["l0"],
               "greedy_logit": None if g["logit_c2"] is None else max(-LOGIT_CLIP, min(LOGIT_CLIP, g["logit_c2"])),
               "greedy_overflow": g["c2_overflow"], "qe": [c["qe"] for c in cs], "logit": [c["logit"] for c in cs],
               "reward": [c["reward"] for c in cs], "fail": [",".join(c["fail"]) for c in cs],
               "ratio": [c["ratio"] for c in cs], "c2q_eligible": n_bound}
        for sel, (k, fallback) in picks.items():
            name = SAMPLE_NAMES[k]
            rec[f"{sel}_choice"] = name
            rec[f"{sel}_fallback"] = fallback
            chosen_rows[sel]["pairs"].append(raw[name]["pairs"][i])
            chosen_rows[sel]["metricx"].append(raw[name]["metricx"][i])
            if i in raw[name]["comet"]:
                chosen_rows[sel]["comet"].append(raw[name]["comet"][i])
        choice.append(rec)
    if args.best8:
        best8 = {r["id"]: r["chosen"] for r in g4arm.read_rows(args.best8, ["id", "chosen"])}
        differ = [r["id"] for r in choice if best8.get(r["id"]) != r["qe_choice"]]
        if differ:
            raise SystemExit(f"the QE-best choice differs from the probe's best8 at {len(differ)} search segments, "
                             f"e.g. {differ[:3]}")
    for sel, x in chosen_rows.items():
        g4arm.write_parquet(x["pairs"], out / sel / "pairs" / "pairs-wall.parquet")
        g4arm.write_parquet(x["metricx"], out / sel / "scores" / "metricx-wall.parquet")
        if x["comet"]:
            g4arm.write_parquet(x["comet"], out / sel / "scores" / "comet-wall.parquet")
    g4arm.write_parquet(choice, out / "choice.parquet")
    stats = select_stats(choice)
    (out / "select.json").write_text(json.dumps({"model": args.model, "part": PART, "n": {s: len(v) for s, v in ids.items()},
                                                 "best8_checked": bool(args.best8), "stats": stats}, indent=1) + "\n")
    fallbacks = {s: {sel: x[sel]["fallback"] for sel in SELECTIONS} for s, x in stats.items()}
    print(f"BONSELECT {json.dumps({'model': args.model, 'fallbacks': fallbacks})}", flush=True)


def select_stats(choice):
    """Per set: per selection the fallbacks, the share equal to qe's choice, the chosen samples; the c2q candidates"""
    out = {}
    for st in SETS:
        rows = [r for r in choice if r["set"] == st]
        x = {}
        for sel in SELECTIONS:
            x[sel] = {"fallback": sum(r[f"{sel}_fallback"] for r in rows),
                      "same_as_qe": sum(r[f"{sel}_choice"] == r["qe_choice"] for r in rows) / len(rows),
                      "chosen": dict(sorted(collections.Counter(r[f"{sel}_choice"] for r in rows).items()))}
        x["c2q"]["eligible_mean"] = statistics.fmean(r["c2q_eligible"] for r in rows)
        x["all_fail"] = sum(all(f for f in r["fail"]) for r in rows)
        out[st] = x
    return out


# ---------------------------------------------------------------------------------------------------- texts

def pairs_of(pattern, ids):
    rows = armeval.read_rows(pattern, ["id", "set", "hypothesis"])
    missing = [i for i in ids if i not in rows]
    if missing:
        raise SystemExit(f"{pattern}: no translation of {len(missing)} search segments, e.g. {missing[:3]}")
    return rows


def c2texts(args):
    """C2's texts of a candidate (model, set, id, text, tokens): its 8 samples (<M>/t10s<k>), its greedy translation
    (<M>) and G4-0's greedy translation (G4-0, the check against c2cal7's run), the search part only"""
    import g4c2
    import pyarrow as pa
    import pyarrow.parquet as pq

    ids = search_ids(load_evalsets(args.evalsets), load_split(args.split))
    runs = [x.split("=", 1) for x in args.sample]
    if [n for n, _ in runs] != list(SAMPLE_NAMES):
        raise SystemExit(f"--sample: {[n for n, _ in runs]}, expected {list(SAMPLE_NAMES)} in this order")
    specs = [(f"{args.model}/{n}", p) for n, p in runs] + [(args.model, args.greedy), (BASE, args.g40)]
    out = []
    for name, pattern in specs:
        rows = pairs_of(pattern, [i for s in SETS for i in ids[s]])
        out += [{"model": name, "set": s, "id": i, "text": rows[i]["hypothesis"] or ""} for s in SETS for i in ids[s]]
    counts = g4c2.token_counter(g4c2.tokenizer_of(args.checkpoint))([r["text"] for r in out])
    for r, n in zip(out, counts):
        r["tokens"] = n
    pq.write_table(pa.Table.from_pylist(out), args.out)
    print(f"BONC2TEXTS {json.dumps(collections.Counter(r['model'] for r in out))} over_{C2_MAX_TOKENS} "
          f"{sum(n > C2_MAX_TOKENS for n in counts)}", flush=True)


def texts(args):
    """C1's texts (model, set, id, text) of every output, the search part only, for one classify run"""
    ids = search_ids(load_evalsets(args.evalsets), load_split(args.split))
    out = []
    for spec in args.pairs:
        name, pattern = spec.split("=", 1)
        rows = pairs_of(pattern, [i for s in SETS for i in ids[s]])
        out += [{"model": name, "set": s, "id": i, "text": rows[i]["hypothesis"] or ""} for s in SETS for i in ids[s]]
    if len({(r["model"], r["id"]) for r in out}) != len(out):
        raise SystemExit("a (model, id) twice among the texts")
    g4arm.write_parquet(out, args.out)
    print(f"BONTEXTS {json.dumps(collections.Counter(r['model'] for r in out))}", flush=True)


# ---------------------------------------------------------------------------------------------------- report

def seed_int(text):
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little")


def paired_ci(diffs, name, boot=BOOT):
    """The mean of paired differences pooled over the sets (each set resampled within itself, so the pooled share
    keeps the sets' sizes) and its 95% percentile interval; diffs {set: [values]}"""
    import numpy as np

    rng = np.random.default_rng(seed_int(f"{BOOT_SEED}/{name}"))
    arrays = {s: np.asarray(v, dtype=np.float64) for s, v in diffs.items() if len(v)}
    n = sum(len(a) for a in arrays.values())
    point = sum(float(a.sum()) for a in arrays.values()) / n
    total = np.zeros(boot)
    for a in arrays.values():
        total += a[rng.integers(0, len(a), size=(boot, len(a)))].sum(axis=1)
    lo, hi = np.percentile(total / n, [2.5, 97.5])
    return {"delta": point, "lo": float(lo), "hi": float(hi), "n": n}


def output_rows(model, out, scored, choice, c1, anchors):
    """{id: view} of one output: C1's P(original), C2's clipped logit (and overflow), the scores, the flags, G4-7's
    fail reasons and the token ratio against G4-0's greedy translation"""
    rows = {}
    for i, ch in choice.items():
        r = scored[i]
        if out == "greedy":
            c = {"logit_c2": ch["greedy_logit"], "c2_overflow": ch["greedy_overflow"]}
            rw = g4rl.reward7(sample_view(r), r["metricx_qe"], c, anchors[i])
            logit, fail, ratio = rw["logit"], rw["fail"], rw["ratio"]
        else:
            k = SAMPLE_NAMES.index("t10s1" if out == "t10s1" else ch[f"{out}_choice"])
            logit, fail, ratio = ch["logit"][k], [f for f in ch["fail"][k].split(",") if f], ch["ratio"][k]
            if r["metricx_qe"] != ch["qe"][k]:
                raise SystemExit(f"{model}/{out} {i}: the output's QE {r['metricx_qe']} is not the chosen sample's {ch['qe'][k]}")
        name = model if out == "greedy" else f"{model}/{out}"
        if (name, i) not in c1:
            raise SystemExit(f"C1 has no score of {name} {i}")
        rows[i] = {**r, "c1": c1[(name, i)], "logit": logit, "fail": fail, "ratio": ratio}
    return rows


def measures(rows, ids, with_ref):
    rs = [rows[i] for i in ids]
    n = len(rs)
    out = {"n": n, "c1": sum(r["c1"] >= 0.5 for r in rs) / n,
           "c2": sum(r["logit"] is not None and r["logit"] >= 0.0 for r in rs) / n,
           "c2_logit": g4arm.mean(r["logit"] for r in rs), "qe": g4arm.mean(r["metricx_qe"] for r in rs),
           "fail": sum(bool(r["fail"]) for r in rs) / n, "not_ended": sum(not r["ended"] for r in rs) / n,
           "decode_loop": sum(bool(r["decode_loop"]) for r in rs) / n,
           "english_left": sum(bool(r["english_left"]) for r in rs) / n, "empty": sum(bool(r["empty"]) for r in rs) / n,
           "chars_source": sum(len(r["hypothesis"] or "") for r in rs) / sum(len(r["source"]) for r in rs),
           "ratio_median": statistics.median(r["ratio"] for r in rs)}
    if with_ref:
        out.update({"metricx_ref": g4arm.mean(r["metricx_ref"] for r in rs), "comet22": g4arm.mean(r["comet22"] for r in rs),
                    "chars_reference": sum(len(r["hypothesis"] or "") for r in rs) / sum(len(r["reference"]) for r in rs)})
    return out


def c1_delta(a, b, ids, name):
    return paired_ci({s: [(a[i]["c1"] >= 0.5) - (b[i]["c1"] >= 0.5) for i in ids[s]] for s in SETS}, name)


def value_delta(a, b, ids, key, name):
    return paired_ci({"wmt24pp": [a[i][key] - b[i][key] for i in ids["wmt24pp"]]}, name)


def report(args):
    evalsets = load_evalsets(args.evalsets)
    ids = search_ids(evalsets, load_split(args.split))
    all_ids = [i for s in SETS for i in ids[s]]
    refqe = {r["id"]: r["qe_source_reference"] for r in g4arm.read_rows(args.refqe, ["id", "qe_source_reference"])}
    human = g4arm.mean(refqe[i] for i in ids["wmt24pp"])
    c1 = {(r["model"], r["id"]): r["p_original"] for r in g4arm.read_rows(args.predictions, ["model", "id", "p_original"])}
    anchor_rows = armeval.scored(args.anchor)
    anchors = {i: anchor_of(anchor_rows[i]) for i in all_ids}
    result = {"protocol": "protocol-g4.md §15", "part": PART, "n": N_SEARCH, "human_qe_wmt24pp": human,
              "boot": BOOT, "models": {}, "checks": {}}
    views = {}
    for spec in args.model:
        model, dirs = spec.split("=", 1)
        greedy_dir, t1_dir, select_dir = dirs.split(",")
        choice = {r["id"]: r for r in g4arm.read_rows(f"{select_dir}/choice.parquet")}
        if set(choice) != set(all_ids):
            raise SystemExit(f"{model}: the choice has {len(choice)} segments, not the {len(all_ids)} of the search part")
        scored = {"greedy": armeval.scored(greedy_dir), "t10s1": armeval.scored(t1_dir),
                  **{sel: armeval.scored(f"{select_dir}/{sel}") for sel in SELECTIONS}}
        views[model] = {o: output_rows(model, o, scored[o], choice, c1, anchors) for o in OUTPUTS}
        result["models"][model] = {"select": json.loads(Path(f"{select_dir}/select.json").read_text())}
    if BASE not in views:
        raise SystemExit(f"no {BASE} among the models: the deltas are against its greedy translation")
    g40 = views[BASE]["greedy"]
    for model, v in views.items():
        outs = {}
        for o in OUTPUTS:
            x = {"sets": {s: measures(v[o], ids[s], s == "wmt24pp") for s in SETS},
                 "c1_pooled": sum(v[o][i]["c1"] >= 0.5 for i in all_ids) / len(all_ids),
                 "c2_pooled": sum(v[o][i]["logit"] is not None and v[o][i]["logit"] >= 0 for i in all_ids) / len(all_ids),
                 "selection_measures": SELECTION_MEASURES[o]}
            if o != "greedy":
                x["c1_vs_greedy"] = c1_delta(v[o], v["greedy"], ids, f"{model}/{o}-greedy")
                d_c2 = x["c2_pooled"] - sum(v["greedy"][i]["logit"] is not None and v["greedy"][i]["logit"] >= 0
                                            for i in all_ids) / len(all_ids)
                x["c2_vs_greedy"] = d_c2
                x["hack_flag"] = d_c2 >= HACK_C2 and x["c1_vs_greedy"]["delta"] < HACK_C1
            if not (model == BASE and o == "greedy"):
                x["c1_vs_g40"] = c1_delta(v[o], g40, ids, f"{model}/{o}-G4-0")
            outs[o] = x
        contrasts = {}
        for o in ("r7", "c2q", "c2"):
            contrasts[f"{o}-qe"] = {"c1": c1_delta(v[o], v["qe"], ids, f"{model}/{o}-qe"),
                                    "comet22": value_delta(v[o], v["qe"], ids, "comet22", f"{model}/{o}-qe/comet"),
                                    "metricx_ref": value_delta(v[o], v["qe"], ids, "metricx_ref", f"{model}/{o}-qe/mxref")}
        for o in SELECTIONS:
            contrasts[f"{o}-greedy"] = {"comet22": value_delta(v[o], v["greedy"], ids, "comet22", f"{model}/{o}-greedy/comet"),
                                        "metricx_ref": value_delta(v[o], v["greedy"], ids, "metricx_ref", f"{model}/{o}-greedy/mxref")}
        result["models"][model].update({"outputs": outs, "contrasts": contrasts})
    # checks, reported: C1 of this run against the probe's run on the same segments; C2 of G4-0's greedy across runs
    probe = {(r["model"], r["id"]): r["p_original"]
             for r in g4arm.read_rows(args.probe_predictions, ["model", "id", "p_original"])}
    checks = {}
    for here, there in ((f"{BASE}/t10s1", f"{BASE}/t10s1"), (f"{BASE}/qe", f"{BASE}/best8"), (BASE, BASE)):
        got = [(c1[(here, i)] >= 0.5, probe[(there, i)] >= 0.5) for i in all_ids if (there, i) in probe]
        if len(got) != len(all_ids):
            raise SystemExit(f"the probe's run has {len(got)} of the {len(all_ids)} search segments of {there}")
        checks[f"{here} ~ probe {there}"] = {"share_here": sum(a for a, _ in got) / len(got),
                                             "share_probe": sum(b for _, b in got) / len(got),
                                             "flipped": sum(a != b for a, b in got)}
    result["checks"]["c1_against_probe"] = checks
    for spec in args.c2_check or []:
        model, paths = spec.split("=", 1)
        here_path, cal7_path = paths.split(",")
        here = {r["id"]: r["p_original"] for r in g4arm.read_rows(here_path, ["model", "id", "p_original"]) if r["model"] == BASE}
        cal7 = {r["id"]: r["p_original"] for r in g4arm.read_rows(cal7_path, ["model", "id", "p_original"]) if r["model"] == BASE}
        got = [(here[i] >= 0.5, cal7[i] >= 0.5, abs(logit_of(here[i]) - logit_of(cal7[i]))) for i in all_ids]
        result["checks"][f"c2_{model}_run"] = {"share_here": sum(a for a, _, _ in got) / len(got),
                                               "share_cal7": sum(b for _, b, _ in got) / len(got),
                                               "flipped": sum(a != b for a, b, _ in got),
                                               "max_abs_dlogit": max(d for _, _, d in got)}
    Path(args.out).with_suffix(".json").write_text(json.dumps(result, indent=1) + "\n")
    Path(args.out).write_text(report_text(result), encoding="utf-8")
    print(f"BONREPORT {json.dumps({m: {o: round(x['outputs'][o]['c1_pooled'], 4) for o in OUTPUTS} for m, x in result['models'].items()})}",
          flush=True)


def ci_text(d, scale=100.0, unit=" pp", digits=1):
    def f(x):
        return f"{scale * x:+.{digits}f}".replace(".", ",")
    return f"{f(d['delta'])}{unit} [{f(d['lo'])}; {f(d['hi'])}]"


def report_text(r):
    pct, f3 = g4arm.pct, g4arm.f3
    L = ["# Meting G4-bon: best-of-8 met selectie op C2 + QE (zoekdeel; rapporteren, beslist niets)", "",
         "Gemaakt door `g4bon.py report` (protocol-g4.md §15). Zoekdeel van G4-5.4: WMT24++ zonder slechte bronnen "
         f"({r['n']['wmt24pp']}) en FLAN ({r['n']['flan']}). Per bron 8 samples bij T = 1 (seeds 1–8, de instellingen "
         "van de sampling-proef); de selecties kiezen één sample per bron. C1 (\"origineel\" = P ≥ 0,5) uit één "
         "classify-run over alle uitvoer; C1 kiest niets. Intervallen: 95%, gepaarde bootstrap over de segmenten "
         f"({r['boot']} trekkingen, per set binnen de set).", "",
         "## Tabel 1: aandeel \"origineel\" (C1)", "",
         "| model | uitvoer | WMT24++ | FLAN | samen | Δ eigen greedy (95%-BI) | Δ G4-0 greedy (95%-BI) | C2 samen | ΔC2 eigen greedy | C2 ↑, C1 niet |",
         "|---|---|---:|---:|---:|---|---|---:|---:|---|"]
    for m, x in r["models"].items():
        for o in OUTPUTS:
            y = x["outputs"][o]
            d_c2 = f"{100 * y['c2_vs_greedy']:+.1f} pp".replace(".", ",") if "c2_vs_greedy" in y else "–"
            hack = ("ja" if y["hack_flag"] else "nee") if "hack_flag" in y else "–"
            L.append(f"| {m} | {LABELS[o]} | {pct(y['sets']['wmt24pp']['c1'])} | {pct(y['sets']['flan']['c1'])} | "
                     f"{pct(y['c1_pooled'])} | {ci_text(y['c1_vs_greedy']) if 'c1_vs_greedy' in y else '–'} | "
                     f"{ci_text(y['c1_vs_g40']) if 'c1_vs_g40' in y else '–'} | {pct(y['c2_pooled'])} | {d_c2} | {hack} |")
    L += ["", f"## Tabel 2: kwaliteit en vorm (QE van de menselijke referentie op WMT24++ van het zoekdeel: "
          f"{f3(r['human_qe_wmt24pp'])})", "",
          "| model | uitvoer | set | QE | MetricX-ref | COMET-22 | fail (G4-7) | niet af | lus | Engels over | leeg | "
          "tekens / bron | tekens / ref | ℓ/ℓ₀ mediaan | C2-logit gem. | niet onafhankelijk (selectiemaat) |",
          "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for m, x in r["models"].items():
        for o in OUTPUTS:
            y = x["outputs"][o]
            for s in SETS:
                z = y["sets"][s]
                L.append(f"| {m} | {o} | {s} | {f3(z['qe'])} | {f3(z.get('metricx_ref'))} | {f3(z.get('comet22'))} | "
                         f"{pct(z['fail'])} | {pct(z['not_ended'])} | {pct(z['decode_loop'])} | {pct(z['english_left'])} | "
                         f"{pct(z['empty'])} | {f3(z['chars_source'])} | {f3(z.get('chars_reference'))} | "
                         f"{f3(z['ratio_median'])} | {f3(z['c2_logit'])} | {', '.join(y['selection_measures']) or '–'} |")
    L += ["", "## Tabel 3: contrasten binnen een model (WMT24++ voor COMET-22 en MetricX-ref; C1 samen)", "",
          "| model | contrast | ΔC1 samen (95%-BI) | ΔCOMET-22 (95%-BI) | ΔMetricX-ref (95%-BI) |", "|---|---|---|---|---|"]
    for m, x in r["models"].items():
        for name, c in x["contrasts"].items():
            L.append(f"| {m} | {name} | {ci_text(c['c1']) if 'c1' in c else '–'} | "
                     f"{ci_text(c['comet22'], 1.0, '', 4)} | {ci_text(c['metricx_ref'], 1.0, '', 3)} |")
    L += ["", "## Tabel 4: de selectie", "",
          "| model | set | selectie | terugval op qe | gelijk aan qe-keuze | gekozen samples |", "|---|---|---|---:|---:|---|"]
    for m, x in r["models"].items():
        for s, st in x["select"]["stats"].items():
            for sel in SELECTIONS:
                y = st[sel]
                L.append(f"| {m} | {s} | {sel} | {y['fallback']} | {pct(y['same_as_qe'])} | "
                         f"{', '.join(f'{k} {v}' for k, v in y['chosen'].items())} |")
            L.append(f"| {m} | {s} | (c2q: gem. toegelaten samples {f3(st['c2q']['eligible_mean'])}; bronnen waar alle 8 "
                     f"falen {st['all_fail']}) | | | |")
    L += ["", "## Controles (gerapporteerd)", ""]
    for k, v in r["checks"]["c1_against_probe"].items():
        L.append(f"- C1 {k}: hier {pct(v['share_here'])}, sampling-proef {pct(v['share_probe'])}, label anders bij "
                 f"{v['flipped']} segmenten.")
    for k, v in r["checks"].items():
        if k.startswith("c2_"):
            L.append(f"- C2 van G4-0 greedy, run van {k[3:-4]} tegen c2cal7: hier {pct(v['share_here'])}, c2cal7 "
                     f"{pct(v['share_cal7'])}, label anders bij {v['flipped']}, max |Δlogit| {f3(v['max_abs_dlogit'])}.")
    L += ["", "## Wat dit niet zegt", "",
          "- Een selectie uit 8 samples is geen getraind model: de tabel zegt wat er in de verdeling zit en met deze "
          "maten te vinden is, niet hoeveel daarvan een fine-tune (G4-8) naar greedy decoding overbrengt.",
          "- Een maat waarop gekozen is, is voor die uitvoer geen onafhankelijke toets (kolom \"niet onafhankelijk\"); "
          "MetricX-ref gebruikt hetzelfde model als QE. Voor FLAN is er geen referentie: de kwaliteitsprijs op FLAN is "
          "voor qe, r7 en c2q alleen met een selectiemaat gemeten.",
          "- C1 en C2 hangen samen (Spearman per segment ~0,6, c2cal7): een C1-winst na selectie op C2 is geen bewijs "
          "van natuurlijker Nederlands; daarvoor is een blinde beoordeling nodig, die hier niet gebeurt.",
          "- Alleen het zoekdeel (n = 480 + 500); de kandidaat van G4-7 is op dit deel gekozen (op C1 van zijn greedy "
          "vertaling), dus zijn cijfers hier zijn optimistisch; verschillen tussen zijn eigen uitvoer minder.",
          "- C2 van de samples van G4-0 komt uit c2cal7, die van de kandidaat uit een eigen run (controle hierboven).", ""]
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------------------------------- CLI

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("bondata")
    a.add_argument("--evalsets", required=True)
    a.add_argument("--split", required=True)
    a.add_argument("--segments", required=True, help="the probe's data: segments-w<k>.parquet, classes-w<k>.txt")
    a.add_argument("--workers", type=int, default=4)
    a.add_argument("--out", required=True)
    a = sub.add_parser("select")
    a.add_argument("--model", required=True)
    a.add_argument("--sample", action="append", required=True, help="t10s<k>=DIR (pairs/, scores/), in order")
    a.add_argument("--anchor", required=True, help="G4-0's evaluation (pairs/, scores/): QE0 and l0")
    a.add_argument("--c2-predictions", required=True)
    a.add_argument("--c2-texts", required=True)
    a.add_argument("--evalsets", required=True)
    a.add_argument("--split", required=True)
    a.add_argument("--best8", help="the probe's best8/choice.parquet: qe's choice must equal it (G4-0)")
    a.add_argument("--probe-report", help="the probe's report.json: its T = 1 settings must be SAMPLES (G4-0)")
    a.add_argument("--out", required=True)
    a = sub.add_parser("c2texts")
    a.add_argument("--model", required=True)
    a.add_argument("--sample", action="append", required=True, help="t10s<k>=GLOB of pairs, in order")
    a.add_argument("--greedy", required=True, help="GLOB of the model's greedy pairs")
    a.add_argument("--g40", required=True, help="GLOB of G4-0's greedy pairs")
    a.add_argument("--evalsets", required=True)
    a.add_argument("--split", required=True)
    a.add_argument("--checkpoint", required=True, help="a directory with C2's config.json (its tokenizer)")
    a.add_argument("--out", required=True)
    a = sub.add_parser("texts")
    a.add_argument("--pairs", action="append", required=True, help="NAME=GLOB")
    a.add_argument("--evalsets", required=True)
    a.add_argument("--split", required=True)
    a.add_argument("--out", required=True)
    a = sub.add_parser("report")
    a.add_argument("--evalsets", required=True)
    a.add_argument("--split", required=True)
    a.add_argument("--refqe", required=True)
    a.add_argument("--anchor", required=True)
    a.add_argument("--predictions", required=True, help="C1 of this measurement's classify run")
    a.add_argument("--probe-predictions", required=True, help="C1 of the sampling probe's classify run")
    a.add_argument("--model", action="append", required=True, help="NAME=GREEDY_DIR,T10S1_DIR,SELECT_DIR")
    a.add_argument("--c2-check", action="append", help="NAME=PREDICTIONS,CAL7_PREDICTIONS (G4-0's greedy in both)")
    a.add_argument("--out", required=True)
    args = p.parse_args(argv)
    {"bondata": bondata, "select": select, "c2texts": c2texts, "texts": texts, "report": report}[args.cmd](args)


if __name__ == "__main__":
    main()
