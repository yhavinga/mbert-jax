#!/usr/bin/env python3
"""Arm G4-7 (spec translator-sft/g4-rl-spec.md; amendment G4-7, protocol-g4.md §14): online RL (GRPO) with LoRA on
Gemma 4 31B-it as the en->nl translator, route C-phase. Per iteration i the four workers of mbert-corpus translate their
part of the iteration's sources G = 8 times each (translate.py --copies, sampling at T = 1, the LoRA of iteration i
merged), the judges of mbert-mix score every sample (MetricX-24-QE; C2, the reward classifier), this module turns the
samples and scores into rewards, advantages and a packed training file, and one job of sft_lora.py --grpo over the 16
chips computes the reference's and the old policy's log-probabilities of every sample token and takes one optimizer step
per batch, the optimizer state continued from iteration to iteration (--resume-step).

Decisions of Yeb (20-21 Nov) over the spec: C2 on dmbert v2 (its token bound 1,022: the judges' c2_overflow); stop rules
S1-S4 and S6, S4's band [0.85, 1.15]; sources of at most 256 Gemma 4 tokens; the rest as the spec advises: start G4-0
(the reference is the model with its LoRA at zero), G = 8, 512 sources per iteration, LR 5e-6, beta 0.04 with k2 and
sg(rho), Dr. GRPO's advantage (no division by the group's std), no overlap.

Pure parts (numpy, no JAX):
  pool7     the sources: FLAN 40%, novels 30%, BT 30% (BT equal per source), English of at most 256 Gemma 4 tokens, no
            13-gram of the evaluation sets or the blind passages, the excluded documents out; RL-val (128 per part)
            drawn first, its documents kept out of the pool; the order of the iterations (seed SEED7/order); a reserve
            of RESERVE7 per part for the sources the anchor drops
  plan7     the anchor (G4-0's greedy translation of every pool and RL-val source, with its QE and C2): a source whose
            anchor did not end, loops, is empty or leaves English is dropped; per part the first sources that remain
            make the iterations (512 each: 205 FLAN, 154 novels, 153 BT) and RL-val (128 per part)
  segments7 an iteration's (or RL-val's) sources for the workers' translate.py
  iter7     an iteration's samples and scores (the bucket protocol of spec §3.4, checked: markers, sha256, the uids of
            samples and scores equal) -> reward (§4.1), advantage (Dr. GRPO, §4.3; groups with max - min < 0.05 left out),
            the packed training file (one sample a segment of MaxText's SFT row: the prompt of g4lora.prompt_text and the
            sampled token ids, stop token included; per token adv and sampler_logp beside the targets), the report
  stop7     the stop rules after an iteration (RL-val greedy with the new adapter, the iteration's samples, the passes)
  grpo_terms, the loss's arithmetic (numpy or jax.numpy), and the sampler/learner gap.
JAX parts (sft_lora.py --grpo): logp_passes (the reference, the model with its LoRA at zero, and the old policy, the model
as loaded, in one jitted forward per batch), make_grpo_loss (MaxText's SFT loss signature).

The bucket protocol (spec §3.4; RL = gs://.../g4arm/rl/G4-7): iter/<i>/samples/w<k>/shard-<n>.parquet (SAMPLE_COLUMNS,
written by translate.py --copies) and its DONE ({"rows", "shards", "sha256": {shard: digest}}); iter/<i>/scores/<role>/
w<m>/shard-<k>-<n>.parquet (SCORE_COLUMNS[role]) and DONE-<k> (the rows scored of generation worker k), role metricx or
c2. RL-val and the anchor take the same layout under val/<i>/ and data/anchor/.
"""

import argparse
import hashlib
import json
import math
import random
import statistics
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
# translate.py (and translate_models.yml) of FLAN-nl in the repo, or beside this file on a worker
for _path in (HERE.parent / "flan-nl", HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import g4dpo  # noqa: E402

ARM7 = "G4-7"
SEED7 = "en-nl-g4arm/0/G4-7"
G7 = 8                               # samples per source
SOURCES7 = 512                       # sources per iteration
ITERATIONS7 = 30
PARTS7 = {"flan": 0.4, "novels": 0.3, "bt": 0.3}
RLVAL_PER_PART7 = 128
MAX_SOURCE7 = 256                    # Gemma 4 tokens of the English (Yeb 20-21 Nov; class medium of translate.py)
LENGTH_CLASS7 = "medium"             # prefill 256/384, decode limit 448, 52 slots: one engine per worker
RESERVE7 = 0.03                      # pool7 draws this share more per part, for the sources plan7 drops
SEQ7 = 4096
ROWS7 = 16                           # a global batch: 16 rows of 4,096 (per_device_batch_size 1 on 16 chips)
LR7 = 5e-6
WARMUP7 = 10                         # linear warmup steps, then constant (decay 0)
# the reward (spec §4.1)
FAIL_REWARD = -10.0
LOGIT_CLIP = 6.0
QE_WEIGHT = 2.0
QE_FAIL = 7.5
LEN_WEIGHT = 2.0
LEN_FREE = math.log(1.15)
LEN_BAND = (0.67, 1.5)
MIN_SPREAD = 0.05                    # a group with max - min of its rewards below this is not in the training file
FAILS = ("not_ended", "decode_loop", "empty", "english_left", "length", "qe", "c2_overflow")
# the loss (spec §4.3, §4.4)
GRPO_DEFAULTS = {"beta": 0.04, "epsilon": 0.2, "log_ratio_clip": 20.0, "tis": False, "tis_threshold": 2.0}
TIS_GAP = 0.05                       # mean |log pi_old - log pi_sampler| per token above this: TIS from the next iteration
# the stop rules (spec §6.4; Yeb 20-21 Nov: S4 [0.85, 1.15], no S5)
S1_MAX = 0.02
S2_MAX = 0.15
S3_MAX = 0.30
S4_BAND = (0.85, 1.15)
S6_MAX = 0.10
# the bucket protocol (spec §3.4)
SAMPLE_COLUMNS = ("uid", "source_id", "j", "source", "hypothesis", "token_ids", "sampler_logp", "ended", "decode_loop",
                  "english_left", "empty", "tokens_out")
SCORE_COLUMNS = {"metricx": ("uid", "qe"), "c2": ("uid", "p_c2", "logit_c2", "c2_tokens", "c2_overflow")}
DONE = "DONE"
# the packed training file: g4dpo.KEYS, and per token the advantage and the sampler's log-probability (float32)
TOKEN_KEYS = ("adv", "sampler_logp")
PASS_KEYS = ("ref_logp", "old_logp")
TIS_KEY = "tis_weight"
ROW_KEY = g4dpo.ROW_KEY
PASSES_FILE = "grpo-passes.npz"
PASSES_META = "grpo-passes.json"


# ---------------------------------------------------------------------------------------------------- small helpers

def uid(source_id, j):
    return f"{source_id}#{j}"


def split_uid(text):
    source_id, sep, j = str(text).rpartition("#")
    if not sep or not source_id or not j.isdigit():
        raise ValueError(f"not a uid <source id>#<j>: {text!r}")
    return source_id, int(j)


def seed_int(text):
    """A string seed as the int translate.py --sample takes (the first 8 bytes of its sha256)"""
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big") >> 1


def sample_seed(iteration, worker):
    """The sampling seed of worker k in iteration i: SEED7/sample/<i>/w<k>"""
    return seed_int(f"{SEED7}/sample/{iteration}/w{worker}")


def file_sha256(path):
    return g4dpo.file_sha256(path)


def write_parquet(rows, path, schema=None):
    import pyarrow as pa
    import pyarrow.parquet as pq

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path, compression="zstd")


def read_rows(path, columns=None):
    import pyarrow.parquet as pq

    return pq.read_table(path, columns=columns).to_pylist()


def part_counts(total=SOURCES7, shares=PARTS7):
    """{part: sources per iteration}: the shares of `total`, largest remainders first (ties in the parts' order)"""
    raw = {p: total * s for p, s in shares.items()}
    out = {p: int(math.floor(v)) for p, v in raw.items()}
    rest = total - sum(out.values())
    for p in sorted(raw, key=lambda p: -(raw[p] - out[p]))[:rest]:
        out[p] += 1
    return out


def median(values):
    values = list(values)
    return statistics.median(values) if values else None


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else None


# ---------------------------------------------------------------------------------------------------- the pool

def ordered(rows, seed):
    """A part's candidates in the pool's order: each stratum shuffled with its own seed (seed/<stratum>), then the strata
    taken in turn (sorted by name), so that any first n are equal per stratum within one"""
    by = {}
    for r in sorted(rows, key=lambda r: r["id"]):
        by.setdefault(r.get("stratum") or "", []).append(r)
    for name, part in by.items():
        random.Random(f"{seed}/{name}").shuffle(part)
    out, k = [], 0
    names = sorted(by)
    while any(k < len(by[n]) for n in names):
        out += [by[n][k] for n in names if k < len(by[n])]
        k += 1
    return out


def candidate_filter(rows, counter, blocked, excluded_docs=frozenset(), max_tokens=MAX_SOURCE7):
    """(the candidates that may be sources, {reason: count}): English of at most max_tokens Gemma 4 tokens (the ids of
    the text alone, as translate.py's segments count them), not empty, no 13-gram of the evaluation sets or the blind
    passages, not of an excluded document, one source per distinct English text (the first id)"""
    import select_passages as sp

    lengths = counter.lengths([r["en"] for r in rows])
    keep, drops, seen = [], {"empty": 0, "tokens": 0, "ngram": 0, "excluded_doc": 0, "duplicate": 0}, set()
    for r, n in sorted(zip(rows, lengths), key=lambda x: x[0]["id"]):
        text = (r["en"] or "").strip()
        if not text:
            drops["empty"] += 1
        elif n > max_tokens:
            drops["tokens"] += 1
        elif r["doc"] in excluded_docs:
            drops["excluded_doc"] += 1
        elif sp.ngrams(text) & blocked:
            drops["ngram"] += 1
        elif text in seen:
            drops["duplicate"] += 1
        else:
            seen.add(text)
            keep.append({**r, "en": text, "tokens": int(n)})
    return keep, drops


def draw_pool(cands, iterations=ITERATIONS7, per_iter=SOURCES7, rlval=RLVAL_PER_PART7, reserve=RESERVE7, seed=SEED7):
    """{part: rows} of filtered candidates -> rows with part, role (rlval or pool) and rank (the order within the part
    and role): per part its order (ordered, seed/order/<part>), RL-val its first rlval x (1 + reserve), then the pool the
    next iterations x count x (1 + reserve) sources of other documents than RL-val's"""
    counts = part_counts(per_iter)
    out = []
    for part in PARTS7:
        rows = ordered(cands[part], f"{seed}/order/{part}")
        n_val = math.ceil(rlval * (1 + reserve))
        n_pool = math.ceil(iterations * counts[part] * (1 + reserve))
        val = rows[:n_val]
        val_docs = {r["doc"] for r in val}
        pool = [r for r in rows[n_val:] if r["doc"] not in val_docs][:n_pool]
        if len(val) < n_val or len(pool) < n_pool:
            raise SystemExit(f"pool7: {part} has {len(val)} RL-val and {len(pool)} pool sources, {n_val} and {n_pool} "
                             f"wanted")
        out += [{**r, "part": part, "role": "rlval", "rank": k} for k, r in enumerate(val)]
        out += [{**r, "part": part, "role": "pool", "rank": k} for k, r in enumerate(pool)]
    return out


POOL_SCHEMA = (("id", "string"), ("part", "string"), ("stratum", "string"), ("doc", "string"), ("en", "string"),
               ("tokens", "int64"), ("role", "string"), ("rank", "int64"))


def schema(fields):
    import pyarrow as pa

    return pa.schema([(name, getattr(pa, kind)()) for name, kind in fields])


def pool7(args, counter=None):
    import g4arm

    counter = counter or g4arm.TokenCounter(args.tokenizer)
    blocked = g4arm.blocked_grams(read_rows(args.evalsets), g4arm.read_jsonl(args.blind))
    excluded = set()
    for path in args.exclude_docs or []:
        excluded |= {line.strip() for line in Path(path).read_text().splitlines() if line.strip()}
    cands, report = {}, {"arm": ARM7, "seed": SEED7, "parts": {}, "excluded_docs": len(excluded)}
    for part in PARTS7:
        rows = read_rows(Path(args.cands) / f"cands-{part}.parquet")
        missing = sorted({"id", "en", "doc"} - set(rows[0] if rows else {}))
        if missing or len({r["id"] for r in rows}) != len(rows):
            raise SystemExit(f"pool7: cands-{part}.parquet needs unique ids and the columns id, en, doc (missing {missing})")
        cands[part], drops = candidate_filter(rows, counter, blocked, excluded)
        report["parts"][part] = {"candidates": len(rows), "kept": len(cands[part]), "drops": drops}
    rows = draw_pool(cands, args.iterations, args.per_iter)
    out = Path(args.out)
    write_parquet([{k: (r.get(k) or "") if k == "stratum" else r[k] for k, _ in POOL_SCHEMA} for r in rows],
                  out / "pool.parquet", schema(POOL_SCHEMA))
    for role in ("pool", "rlval"):
        report[role] = {p: sum(r["part"] == p and r["role"] == role for r in rows) for p in PARTS7}
    report["per_iteration"] = part_counts(args.per_iter)
    report["iterations"] = args.iterations
    (out / "pool.json").write_text(json.dumps(report, indent=1) + "\n")
    print(f"POOL7 {json.dumps(report)}", flush=True)
    return report


def anchor_rows(samples, scores):
    """{source id: anchor} from the anchor's translations (the bucket protocol, one sample #0 per source, greedy) and
    their scores: the flags, l0 (tokens_out), qe0, C2's logit"""
    by_uid = {r["uid"]: r for r in samples}
    out = {}
    for u, r in by_uid.items():
        source_id, j = split_uid(u)
        if j != 0 or source_id in out:
            raise SystemExit(f"anchor: {u} is not one greedy translation #0 per source")
        out[source_id] = {"ended": bool(r["ended"]), "decode_loop": bool(r["decode_loop"]), "empty": bool(r["empty"]),
                          "english_left": bool(r["english_left"]), "l0": int(r["tokens_out"]),
                          "qe0": float(scores["metricx"][u]["qe"]), "logit_c2_0": float(scores["c2"][u]["logit_c2"]),
                          "hypothesis0": r["hypothesis"]}
    return out


def anchor_ok(a):
    """The anchor says something: it ended, does not loop, is not empty and leaves no English (spec §6.2, G4-7.2)"""
    return a["ended"] and not a["decode_loop"] and not a["empty"] and not a["english_left"]


def plan_sources(pool, anchors, iterations=ITERATIONS7, per_iter=SOURCES7, rlval=RLVAL_PER_PART7):
    """(plan rows, report): per part and role the sources in rank order whose anchor is ok; RL-val the first rlval
    (iteration -1), the iterations the first iterations x count, iteration = position // count, then within an
    iteration the parts' sources shuffled together (seed SEED7/order/<i>)"""
    counts = part_counts(per_iter)
    missing = [r["id"] for r in pool if r["id"] not in anchors]
    if missing:
        raise SystemExit(f"plan7: {len(missing)} sources without an anchor, e.g. {missing[:3]}")
    plan, report = [], {"dropped": {}, "kept": {}}
    for part in PARTS7:
        for role, need in (("rlval", rlval), ("pool", iterations * counts[part])):
            rows = sorted((r for r in pool if r["part"] == part and r["role"] == role), key=lambda r: r["rank"])
            ok = [r for r in rows if anchor_ok(anchors[r["id"]])]
            report["dropped"][f"{part}/{role}"] = {k: sum(not anchors[r["id"]][k] if k == "ended" else
                                                         bool(anchors[r["id"]][k]) for r in rows)
                                                   for k in ("ended", "decode_loop", "empty", "english_left")}
            if len(ok) < need:
                raise SystemExit(f"plan7: {part} {role}: {len(ok)} sources with a good anchor, {need} needed")
            report["kept"][f"{part}/{role}"] = need
            for k, r in enumerate(ok[:need]):
                a = anchors[r["id"]]
                plan.append({"id": r["id"], "part": part, "stratum": r.get("stratum") or "", "doc": r["doc"],
                             "en": r["en"], "tokens": int(r["tokens"]),
                             "iteration": -1 if role == "rlval" else k // counts[part],
                             "qe0": a["qe0"], "l0": a["l0"], "logit_c2_0": a["logit_c2_0"],
                             "hypothesis0": a["hypothesis0"]})
    by_iter = {}
    for r in plan:
        by_iter.setdefault(r["iteration"], []).append(r)
    out = []
    for i in sorted(by_iter):
        rows = sorted(by_iter[i], key=lambda r: r["id"])
        random.Random(f"{SEED7}/order/{i}").shuffle(rows)
        out += [{**r, "position": k} for k, r in enumerate(rows)]
    return out, report


PLAN_SCHEMA = (("id", "string"), ("part", "string"), ("stratum", "string"), ("doc", "string"), ("en", "string"),
               ("tokens", "int64"), ("iteration", "int64"), ("position", "int64"), ("qe0", "double"), ("l0", "int64"),
               ("logit_c2_0", "double"), ("hypothesis0", "string"))


def plan7(args):
    pool = read_rows(Path(args.pool) / "pool.parquet")
    samples = read_samples(args.anchor_samples)
    scores = read_scores(args.anchor_scores, {r["uid"] for r in samples})
    anchors = anchor_rows(samples, scores)
    plan, report = plan_sources(pool, anchors, args.iterations, args.per_iter)
    write_parquet(plan, Path(args.out) / "plan.parquet", schema(PLAN_SCHEMA))
    report["sources"] = {"rlval": sum(r["iteration"] < 0 for r in plan), "pool": sum(r["iteration"] >= 0 for r in plan)}
    (Path(args.out) / "plan.json").write_text(json.dumps(report, indent=1) + "\n")
    print(f"PLAN7 {json.dumps(report)}", flush=True)
    return report


def assign_workers7(rows, workers):
    """{worker: rows}: by tokens, longest first, each to the worker with the fewest tokens so far (ties: the lowest)"""
    load, out = [0] * workers, {k: [] for k in range(workers)}
    for r in sorted(rows, key=lambda r: (-r["tokens"], r["id"])):
        k = min(range(workers), key=lambda w: (load[w], w))
        out[k].append(r)
        load[k] += r["tokens"]
    return out


def segments7(args):
    """translate.py's segments (id, text, tokens) of an iteration (or RL-val: --iteration -1), per worker"""
    plan = read_rows(args.plan)
    rows = [r for r in plan if r["iteration"] == args.iteration]
    if not rows:
        raise SystemExit(f"segments7: no source of iteration {args.iteration}")
    for k, part in assign_workers7(rows, args.workers).items():
        write_parquet([{"id": r["id"], "text": r["en"], "tokens": r["tokens"]} for r in part],
                      Path(args.out) / f"segments-w{k}.parquet",
                      schema((("id", "string"), ("text", "string"), ("tokens", "int64"))))
    print(f"SEGMENTS7 {json.dumps({'iteration': args.iteration, 'sources': len(rows), 'workers': args.workers})}")


# ---------------------------------------------------------------------------------------------------- the bucket protocol

def samples_done(shards):
    """A samples DONE marker: {"rows", "shards", "sha256": {name: digest}} of {name: (rows, digest)}"""
    return {"rows": int(sum(n for n, _ in shards.values())), "shards": len(shards),
            "sha256": {name: digest for name, (_, digest) in sorted(shards.items())}}


def read_samples(root):
    """Every sample of the w<k> directories under root (bucket protocol): each directory's DONE names its shards with
    their sha256 and the rows; exactly those shards, unchanged; the columns SAMPLE_COLUMNS; uids unique and of the form
    <source id>#<j>, each row's source_id and j its uid's. Each row gets "worker"."""
    import pyarrow.parquet as pq

    root = Path(root)
    dirs = sorted(p for p in root.glob("w*") if p.is_dir())
    if not dirs:
        raise SystemExit(f"{root}: no w<k> directory of samples")
    rows, seen = [], set()
    for d in dirs:
        if not (d / DONE).is_file():
            raise SystemExit(f"{d}: no {DONE} marker (the generation is not done)")
        done = json.loads((d / DONE).read_text())
        names = sorted(p.name for p in d.glob("*.parquet"))
        if names != sorted(done["sha256"]) or len(names) != done["shards"]:
            raise SystemExit(f"{d}: shards {names[:3]}... ({len(names)}) are not the {done['shards']} of {DONE}")
        n = 0
        for name in names:
            if file_sha256(d / name) != done["sha256"][name]:
                raise SystemExit(f"{d / name}: its sha256 is not the one {DONE} names")
            table = pq.read_table(d / name)
            lacking = sorted(set(SAMPLE_COLUMNS) - set(table.column_names))
            if lacking:
                raise SystemExit(f"{d / name}: no columns {lacking}")
            for r in table.to_pylist():
                if r["uid"] in seen:
                    raise SystemExit(f"{d / name}: {r['uid']} twice")
                if split_uid(r["uid"]) != (str(r["source_id"]), int(r["j"])):
                    raise SystemExit(f"{d / name}: uid {r['uid']} is not source_id {r['source_id']} # j {r['j']}")
                if len(r["token_ids"]) != len(r["sampler_logp"]) or len(r["token_ids"]) != r["tokens_out"]:
                    raise SystemExit(f"{d / name}: {r['uid']} has {len(r['token_ids'])} token ids, "
                                     f"{len(r['sampler_logp'])} log-probabilities, tokens_out {r['tokens_out']}")
                seen.add(r["uid"])
                rows.append({**r, "source_id": str(r["source_id"]), "worker": d.name})
                n += 1
        if n != done["rows"]:
            raise SystemExit(f"{d}: {n} rows, {DONE} says {done['rows']}")
    return rows


def done_rows(path):
    """The rows a judge's DONE-<k> counts: a json object with "rows", or a bare number"""
    value = json.loads(Path(path).read_text())
    return int(value["rows"] if isinstance(value, dict) else value)


def read_scores(root, uids):
    """{role: {uid: row}} of the judges' score shards under root/<role>/w<m>/ (bucket protocol), every role of
    SCORE_COLUMNS: each judge directory has a DONE-<k> for every generation worker k whose shards it holds, with its row
    count; no uid twice in a role; each role's uids exactly the samples' (spec §3.4: else FAILED with the missing)"""
    import pyarrow.parquet as pq

    out = {}
    for role, columns in SCORE_COLUMNS.items():
        judges = sorted(p for p in (Path(root) / role).glob("w*") if p.is_dir())
        if not judges:
            raise SystemExit(f"{root}/{role}: no judge directory")
        got = {}
        for d in judges:
            per_worker = {}
            for path in sorted(d.glob("shard-*.parquet")):
                k = path.name.split("-")[1]
                table = pq.read_table(path)
                lacking = sorted(set(columns) - set(table.column_names))
                if lacking:
                    raise SystemExit(f"{path}: no columns {lacking}")
                for r in table.select(list(columns)).to_pylist():
                    if r["uid"] in got:
                        raise SystemExit(f"{role}: {r['uid']} scored twice ({path})")
                    got[r["uid"]] = r
                    per_worker[k] = per_worker.get(k, 0) + 1
            markers = {p.name[len("DONE-"):]: done_rows(p) for p in d.glob("DONE-*")}
            if set(per_worker) - set(markers) or any(markers[k] != per_worker.get(k, 0) for k in markers):
                raise SystemExit(f"{d}: scored rows per generation worker {per_worker}, markers {markers}")
        missing, extra = sorted(set(uids) - set(got)), sorted(set(got) - set(uids))
        if missing or extra:
            raise SystemExit(f"{role}: {len(missing)} samples without a score (e.g. {missing[:5]}), {len(extra)} scores "
                             f"of no sample (e.g. {extra[:5]})")
        for u, r in got.items():
            values = [v for k, v in r.items() if k not in ("uid", "c2_overflow")]
            if any(v is None or not math.isfinite(float(v)) for v in values):
                raise SystemExit(f"{role}: {u} has a score that is not finite: {r}")
        out[role] = got
    return out


# ---------------------------------------------------------------------------------------------------- reward, advantage

def reward7(sample, qe, c2, anchor):
    """The reward of one sample (spec §4.1): fail -> FAIL_REWARD (no classifier, no QE); else clip(logit C2, -6, 6) -
    2 max(0, QE - QE0) - 2 max(0, |ln(l / l0)| - ln 1.15), with l and l0 the tokens_out of the sample and of the anchor
    (G4-0's greedy translation of the source). Returns {"reward", "fail": [reasons], "logit", "qe_pen", "len_pen",
    "ratio"}"""
    ratio = sample["tokens_out"] / anchor["l0"]
    checks = {"not_ended": not sample["ended"], "decode_loop": bool(sample["decode_loop"]), "empty": bool(sample["empty"]),
              "english_left": bool(sample["english_left"]), "length": not LEN_BAND[0] <= ratio <= LEN_BAND[1],
              "qe": qe > QE_FAIL, "c2_overflow": bool(c2["c2_overflow"])}
    fail = [k for k in FAILS if checks[k]]
    logit = min(max(float(c2["logit_c2"]), -LOGIT_CLIP), LOGIT_CLIP)
    qe_pen = QE_WEIGHT * max(0.0, qe - anchor["qe0"])
    len_pen = LEN_WEIGHT * max(0.0, abs(math.log(ratio)) - LEN_FREE)
    reward = FAIL_REWARD if fail else logit - qe_pen - len_pen
    return {"reward": reward, "fail": fail, "logit": logit, "qe_pen": qe_pen, "len_pen": len_pen, "ratio": ratio}


def advantages7(rewards, min_spread=MIN_SPREAD):
    """Dr. GRPO's group advantage (Tunix's "drgrpo"): reward - the group's mean, no division by its std; (advantages,
    the group has signal: max - min >= min_spread)"""
    r = np.asarray(rewards, np.float64)
    return r - r.mean(), bool(r.max() - r.min() >= min_spread)


def groups7(samples, g=G7):
    """{source id: [sample of j = 0 .. g-1]}: every source exactly g samples"""
    by = {}
    for s in samples:
        by.setdefault(s["source_id"], {})[int(s["j"])] = s
    bad = {k: sorted(v) for k, v in by.items() if sorted(v) != list(range(g))}
    if bad:
        raise SystemExit(f"{len(bad)} sources without exactly the samples 0..{g - 1}, e.g. {list(bad.items())[:3]}")
    return {k: [v[j] for j in range(g)] for k, v in by.items()}


# ---------------------------------------------------------------------------------------------------- packing

def pack_samples(lengths, seq, rows):
    """[batch: [row: [sample index, ...]]] of samples of the given lengths: the order kept, a sample to the first row of
    the open batch with room, a sample that fits no row closes the batch (g4dpo.pack_pairs with a sample as the unit)"""
    batches, cur, free = [], None, None
    for k, n in enumerate(lengths):
        if n > seq:
            raise ValueError(f"sample {k}: {n} tokens, more than a row of {seq}")
        if cur is None:
            cur, free = [[] for _ in range(rows)], [seq] * rows
        for i in range(rows):
            if free[i] >= n:
                break
        else:
            batches.append(cur)
            cur, free, i = [[] for _ in range(rows)], [seq] * rows, 0
        cur[i].append(k)
        free[i] -= n
    if cur is not None:
        batches.append(cur)
    return batches


def row_arrays7(segments, seq, pad_id=g4dpo.PAD_ID, bos_id=g4dpo.BOS_ID):
    """One packed row as MaxText's SFT makes it (g4dpo.row_arrays: the prompt masked, ShiftData's shift) with per token
    the sample's advantage and the sampler's log-probability of each completion token, shifted with the targets, so that
    adv[t] and sampler_logp[t] belong to targets[t]. segments: [(prompt ids, completion ids, advantage, sampler logps)]."""
    arrays = g4dpo.row_unshifted([(p, c) for p, c, _, _ in segments], seq, pad_id)
    adv, slp = np.zeros(seq, np.float32), np.zeros(seq, np.float32)
    at = 0
    for prompt, completion, a, logps in segments:
        if len(logps) != len(completion):
            raise ValueError(f"{len(completion)} completion tokens, {len(logps)} log-probabilities")
        lo = at + len(prompt)
        adv[lo:lo + len(completion)] = a
        slp[lo:lo + len(completion)] = logps
        at = lo + len(completion)
    out = g4dpo.shift_row(arrays, pad_id, bos_id)
    out["adv"] = np.append(adv[1:], np.float32(0)).astype(np.float32)
    out["sampler_logp"] = np.append(slp[1:], np.float32(0)).astype(np.float32)
    return out


def pack_rows7(samples, seq, rows, pad_id=g4dpo.PAD_ID, bos_id=g4dpo.BOS_ID):
    """The packed rows of samples ({"uid", "prompt", "completion", "adv", "sampler_logp"}) in their order, every batch
    `rows` rows (empty rows all zero)"""
    lengths = [len(s["prompt"]) + len(s["completion"]) for s in samples]
    out = []
    for b, batch in enumerate(pack_samples(lengths, seq, rows)):
        for row in batch:
            arrays = row_arrays7([(samples[k]["prompt"], samples[k]["completion"], samples[k]["adv"],
                                   samples[k]["sampler_logp"]) for k in row], seq, pad_id, bos_id)
            out.append({**{k: arrays[k].tolist() for k in (*g4dpo.KEYS, *TOKEN_KEYS)}, "batch": b,
                        "uids": [samples[k]["uid"] for k in row],
                        "tokens": int(np.count_nonzero(arrays["inputs_segmentation"]))})
    return out


def write_rows7(rows, path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    fields = [(k, pa.list_(pa.int32())) for k in g4dpo.KEYS] + [(k, pa.list_(pa.float32())) for k in TOKEN_KEYS]
    fields += [("batch", pa.int64()), ("uids", pa.list_(pa.string())), ("tokens", pa.int64())]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=pa.schema(fields)), path, compression="zstd", row_group_size=1024)


def read_packed7(path):
    """{key: array [rows, seq]} of a packed file of iter7 (KEYS int32, TOKEN_KEYS float32), and its batch per row"""
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=[*g4dpo.KEYS, *TOKEN_KEYS, "batch"])
    rows = table.num_rows
    arrays = {}
    for k in (*g4dpo.KEYS, *TOKEN_KEYS):
        dtype = np.float32 if k in TOKEN_KEYS else np.int32
        arrays[k] = np.asarray(table.column(k).combine_chunks().flatten(), dtype).reshape(rows, -1)
    return arrays, np.asarray(table.column("batch").to_pylist(), np.int64)


def packed_info7(path, rows, seq):
    """{batches, rows, tokens, loss_tokens, samples} of a packed file of rows of `seq` tokens, `rows` per batch"""
    import pyarrow.parquet as pq

    arrays, batch = read_packed7(path)
    if arrays["inputs"].shape[1] != seq:
        raise SystemExit(f"{path}: rows of {arrays['inputs'].shape[1]} tokens, the run's max_target_length is {seq}")
    samples = sum(len(u) for u in pq.read_table(path, columns=["uids"]).column("uids").to_pylist())
    return {"batches": g4dpo.batch_count(batch, rows), "rows": len(batch), "samples": samples,
            "tokens": int(np.count_nonzero(arrays["inputs_segmentation"])),
            "loss_tokens": int(np.count_nonzero(arrays["targets_segmentation"]))}


def tis_weights(old, sampler, mask, threshold):
    """Truncated importance sampling (Tunix's sampler_is, spec §4.4): min(pi_old / pi_sampler, threshold) per target
    token, 0 off the targets"""
    w = np.minimum(np.exp(old.astype(np.float64) - sampler.astype(np.float64)), threshold)
    return np.where(mask, w, 0.0).astype(np.float32)


def local_batches7(arrays, process, processes, rows, batches, padding=0, passes=None, tis=None, row_ids=False):
    """This process's part of each global batch (rows / processes consecutive rows) of an iter7 file: KEYS and
    TOKEN_KEYS; with passes ({"ref_logp", "old_logp"}: [file rows, seq]) those; with tis the TIS_KEY weights; with row_ids
    ROW_KEY (the passes); then `padding` all-zero batches (Tunix's look-ahead, g4dpo.local_batches)"""
    if rows % processes:
        raise SystemExit(f"{rows} rows per batch over {processes} processes")
    local = rows // processes
    seq = arrays["inputs"].shape[1]
    for b in range(batches):
        lo = b * rows + process * local
        out = {k: arrays[k][lo:lo + local] for k in (*g4dpo.KEYS, *TOKEN_KEYS)}
        if passes is not None:
            out.update({k: np.asarray(passes[k][lo:lo + local], np.float32) for k in PASS_KEYS})
        if tis is not None:
            out[TIS_KEY] = np.asarray(tis[lo:lo + local], np.float32)
        if row_ids:
            out[ROW_KEY] = np.arange(lo, lo + local, dtype=np.int32)
        yield out
    for _ in range(padding):
        out = {k: np.zeros((local, seq), np.float32 if k in TOKEN_KEYS else np.int32)
               for k in (*g4dpo.KEYS, *TOKEN_KEYS)}
        if passes is not None:
            out.update({k: np.zeros((local, seq), np.float32) for k in PASS_KEYS})
        if tis is not None:
            out[TIS_KEY] = np.zeros((local, seq), np.float32)
        yield out


def global_iterator7(path, mesh, rows, batches, padding=0, passes=None, tis=None, row_ids=False):
    """MaxText's multi-host iterator over an iter7 file: each process puts its rows of a batch on its devices"""
    import jax
    from maxtext.input_pipeline.multihost_dataloading import MultiHostDataLoadIterator

    arrays, batch = read_packed7(path)
    have = g4dpo.batch_count(batch, rows)
    if batches > have:
        raise SystemExit(f"{path}: {have} batches, {batches} wanted")
    gen = list(local_batches7(arrays, jax.process_index(), jax.process_count(), rows, batches, padding, passes, tis,
                              row_ids))
    return MultiHostDataLoadIterator(gen, mesh)


# ---------------------------------------------------------------------------------------------------- iter7

def iteration_samples(samples, scores, plan, iteration, g=G7):
    """Every sample of an iteration with its reward and advantage, and the groups' signal: (rows, report). The sources
    must be exactly the plan's of the iteration, each with g samples."""
    sources = {r["id"]: r for r in plan if r["iteration"] == iteration}
    groups = groups7(samples, g)
    if set(groups) != set(sources):
        raise SystemExit(f"iteration {iteration}: samples of {len(groups)} sources, the plan has {len(sources)} "
                         f"(not in the plan: {sorted(set(groups) - set(sources))[:3]}, without samples: "
                         f"{sorted(set(sources) - set(groups))[:3]})")
    out, flat = [], 0
    for sid in sorted(groups):
        src = sources[sid]
        if any(s["source"] != src["en"] for s in groups[sid]):
            raise SystemExit(f"{sid}: a sample's source is not the plan's English")
        terms = [reward7(s, float(scores["metricx"][s["uid"]]["qe"]), scores["c2"][s["uid"]], src) for s in groups[sid]]
        adv, signal = advantages7([t["reward"] for t in terms])
        flat += not signal
        for s, t, a in zip(groups[sid], terms, adv):
            out.append({"uid": s["uid"], "source_id": sid, "j": int(s["j"]), "part": src["part"], "worker": s["worker"],
                        "reward": t["reward"], "adv": float(a), "signal": signal, "fail": t["fail"],
                        "logit": t["logit"], "qe_pen": t["qe_pen"], "len_pen": t["len_pen"], "ratio": t["ratio"],
                        "qe": float(scores["metricx"][s["uid"]]["qe"]),
                        "logit_c2": float(scores["c2"][s["uid"]]["logit_c2"]),
                        "p_c2": float(scores["c2"][s["uid"]]["p_c2"]), "c2_tokens": int(scores["c2"][s["uid"]]["c2_tokens"]),
                        "tokens_out": int(s["tokens_out"]), "ended": bool(s["ended"])})
    return out, {"sources": len(groups), "samples": len(out), "groups_without_signal": flat}


def iteration_report(rows):
    """What an iteration's samples are (spec G4-7.10): the reward and its terms, the fail shares per kind (c2_overflow
    too), QE, C2, the length ratio"""
    n = len(rows)
    ok = [r for r in rows if not r["fail"]]
    return {"samples": n, "fail": sum(bool(r["fail"]) for r in rows), "fail_share": sum(bool(r["fail"]) for r in rows) / n,
            "fails": {k: sum(k in r["fail"] for r in rows) for k in FAILS},
            "reward_mean": mean(r["reward"] for r in rows), "reward_mean_ok": mean(r["reward"] for r in ok),
            "logit_mean_ok": mean(r["logit"] for r in ok), "qe_pen_mean_ok": mean(r["qe_pen"] for r in ok),
            "len_pen_mean_ok": mean(r["len_pen"] for r in ok), "qe_mean": mean(r["qe"] for r in rows),
            "p_c2_mean": mean(r["p_c2"] for r in rows), "origineel_share": sum(r["p_c2"] >= 0.5 for r in rows) / n,
            "ratio_median": median(r["ratio"] for r in rows), "adv_abs_mean": mean(abs(r["adv"]) for r in rows),
            "c2_tokens_max": max(r["c2_tokens"] for r in rows)}


def decoded_ok(counter, sample):
    """translate.py's text is the decode of the sampled token ids without the stop token (skip_special_tokens, strip)"""
    ids = sample["token_ids"][:-1] if sample["ended"] else sample["token_ids"]
    return counter.tok.decode(list(ids), skip_special_tokens=True).strip() == sample["hypothesis"]


def iter7(args, counter=None):
    import g4arm
    import g4lora

    counter = counter or g4arm.TokenCounter(args.tokenizer)
    plan = read_rows(args.plan)
    samples = read_samples(args.samples)
    scores = read_scores(args.scores, {s["uid"] for s in samples})
    rows, report = iteration_samples(samples, scores, plan, args.iteration)
    report.update({"iteration": args.iteration, **iteration_report(rows)})
    bad = [s["uid"] for s in samples if not decoded_ok(counter, s)]
    if bad:
        raise SystemExit(f"iter7: the token ids of {len(bad)} samples do not decode to their hypothesis, e.g. {bad[:3]}")
    by_uid = {s["uid"]: s for s in samples}
    en = {r["id"]: r["en"] for r in plan if r["iteration"] == args.iteration}
    prompt_of = dict(zip(sorted(en), counter.ids([g4lora.prompt_text(en[k]) for k in sorted(en)])))
    if any("tokens_in" in s and s["tokens_in"] != len(prompt_of[s["source_id"]]) for s in samples):
        raise SystemExit("iter7: a prompt of g4lora.prompt_text has another length than translate.py's tokens_in")
    train = [r for r in rows if r["signal"]]
    random.Random(f"{SEED7}/pack/{args.iteration}").shuffle(train)
    units = [{"uid": r["uid"], "prompt": prompt_of[r["source_id"]], "completion": list(by_uid[r["uid"]]["token_ids"]),
              "adv": r["adv"], "sampler_logp": list(by_uid[r["uid"]]["sampler_logp"])} for r in train]
    packed = pack_rows7(units, args.seq, args.rows, counter.special_id("<pad>"), counter.special_id("<bos>"))
    out = Path(args.out)
    write_rows7(packed, out / "train.parquet")
    loss_tokens = sum(np.count_nonzero(r["targets_segmentation"]) for r in packed)
    report["train"] = {"samples": len(units), "rows": len(packed), "batches": 1 + max(r["batch"] for r in packed),
                       "tokens": sum(r["tokens"] for r in packed), "loss_tokens": int(loss_tokens),
                       "completion_tokens": sum(len(u["completion"]) for u in units),
                       "fill": sum(r["tokens"] for r in packed) / (len(packed) * args.seq)}
    # a completion token MaxText's shift masks (the pad or the bos id) has no loss: counted, not trained
    report["train"]["masked_completion_tokens"] = report["train"]["completion_tokens"] - report["train"]["loss_tokens"]
    write_parquet([{**r, "fail": ",".join(r["fail"])} for r in rows], out / "samples.parquet")
    (out / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    print(f"ITER7 {json.dumps(report)}", flush=True)
    return report


# ---------------------------------------------------------------------------------------------------- the loss

def grpo_spec(**kw):
    """The GRPO loss's settings (GRPO_DEFAULTS, overridden), checked"""
    out = {**GRPO_DEFAULTS, **{k: v for k, v in kw.items() if v is not None}}
    unknown = sorted(set(out) - set(GRPO_DEFAULTS))
    if unknown:
        raise ValueError(f"unknown settings {unknown}")
    if not out["beta"] >= 0 or not 0 < out["epsilon"] < 1 or not out["log_ratio_clip"] > 0 or not out["tis_threshold"] >= 1:
        raise ValueError(f"beta >= 0, 0 < epsilon < 1, log_ratio_clip > 0, tis_threshold >= 1: {out}")
    out["tis"] = bool(out["tis"])
    return out


def read_grpo_spec(path):
    return grpo_spec(**json.loads(Path(path).read_text()))


SUMS7 = ("tokens", "loss_sum", "pg_sum", "kl_sum", "k2_sum", "k1_ref_sum", "log_ratio_sum", "log_ratio_abs_sum",
         "log_ratio_abs_max", "clip_sum", "adv_sum", "tis_sum")


def grpo_terms(logp, old_logp, ref_logp, adv, mask, opts, xp=np, tis=None):
    """The GRPO loss of a batch and its sums (SUMS7), per target token (mask) of the samples (spec §4.3):
    rho = exp(clip(log pi - log pi_old, +-log_ratio_clip)); policy term -min(rho A, clip(rho, 1 - eps, 1 + eps) A) (Tunix's
    grpo_loss_fn, symmetric clip), times the TIS weight if given (Tunix's sampler_is_weights, the policy term only); KL
    term beta sg(rho) 1/2 (log pi - log pi_ref)^2 (k2, Tunix's "mse_kl" at rho = 1, weighted by the ratio without its
    gradient); the loss the sum of both over the batch's target tokens divided by their count (token-mean)."""
    f32 = xp.float32
    if xp is np:
        def sg(x):
            return x
    else:
        import jax

        sg = jax.lax.stop_gradient
    m = mask.astype(f32)
    logp, old_logp, ref_logp, adv = (x.astype(f32) for x in (logp, old_logp, ref_logp, adv))
    raw = logp - old_logp
    c = float(opts["log_ratio_clip"])
    log_ratio = xp.clip(raw, -c, c)
    ratio = xp.exp(log_ratio)
    eps = float(opts["epsilon"])
    pg1 = -adv * ratio
    pg2 = -adv * xp.clip(ratio, 1.0 - eps, 1.0 + eps)
    pg = xp.maximum(pg1, pg2)
    if tis is not None:
        pg = pg * tis.astype(f32)
    diff = logp - ref_logp
    k2 = 0.5 * diff * diff
    kl = sg(ratio) * k2
    per_token = pg + float(opts["beta"]) * kl
    tokens = m.sum()
    sums = {"tokens": tokens, "loss_sum": (per_token * m).sum(), "pg_sum": (pg * m).sum(), "kl_sum": (kl * m).sum(),
            "k2_sum": (k2 * m).sum(), "k1_ref_sum": (diff * m).sum(), "log_ratio_sum": (raw * m).sum(),
            "log_ratio_abs_sum": (xp.abs(raw) * m).sum(), "log_ratio_abs_max": (xp.abs(raw) * m).max(),
            "clip_sum": ((pg2 > pg1).astype(f32) * m).sum(), "adv_sum": (adv * m).sum(),
            "tis_sum": ((tis.astype(f32) if tis is not None else xp.ones_like(m)) * m).sum()}
    return sums["loss_sum"] / xp.maximum(tokens, 1.0), sums


def step_record7(step, sums):
    """A training step's GRPO measures (the batch's sums as floats)"""
    n = max(sums["tokens"], 1.0)
    return {"step": step, "tokens": int(round(sums["tokens"])), "loss": sums["loss_sum"] / n, "pg": sums["pg_sum"] / n,
            "kl": sums["kl_sum"] / n, "k2": sums["k2_sum"] / n, "k1_ref": sums["k1_ref_sum"] / n,
            "log_ratio": sums["log_ratio_sum"] / n, "log_ratio_abs": sums["log_ratio_abs_sum"] / n,
            "log_ratio_abs_max": sums["log_ratio_abs_max"], "clip_frac": sums["clip_sum"] / n,
            "adv_mean": sums["adv_sum"] / n, "tis_mean": sums["tis_sum"] / n}


def pass_record(ref, old, sampler, mask):
    """The passes over an iteration's training batches (spec §4.4, §6.4 S3): per target token the mean of log pi_old -
    log pi_ref (k1, S3), the sampler/learner gap mean |log pi_old - log pi_sampler| and the k1 estimate of KL(pi_sampler ||
    pi_old) = mean of log pi_sampler - log pi_old, max |old - ref|, whether old is ref bit for bit"""
    m = mask.astype(bool)
    n = int(m.sum())
    if not n:
        raise SystemExit("the passes: no target token")
    o, r, s = (x.astype(np.float64)[m] for x in (old, ref, sampler))
    return {"tokens": n, "k1_old_ref": float((o - r).mean()), "old_ref_abs_max": float(np.abs(o - r).max()),
            "old_is_ref": bool(np.array_equal(old[m], ref[m])), "sampler_gap": float(np.abs(o - s).mean()),
            "kl_sampler_old": float((s - o).mean()), "old_nll": float(-o.mean()), "ref_nll": float(-r.mean()),
            "sampler_nll": float(-s.mean())}


# ---------------------------------------------------------------------------------------------------- stop rules

def validation_measures(samples, scores, plan):
    """RL-val after an iteration (greedy, the new adapter; the bucket protocol with #0 per source): the failure share of
    S1, the mean QE against the anchor's of the same sources (S2), the median l / l0 (S4), C2 (reported)"""
    sources = {r["id"]: r for r in plan if r["iteration"] == -1}
    groups = groups7(samples, 1)
    if set(groups) != set(sources):
        raise SystemExit(f"RL-val: translations of {len(groups)} sources, the plan has {len(sources)}")
    rows = [(groups[k][0], sources[k]) for k in sorted(groups)]
    failed = [s for s, _ in rows if not s["ended"] or s["decode_loop"] or s["empty"] or s["english_left"]]
    qe = [float(scores["metricx"][s["uid"]]["qe"]) for s, _ in rows]
    return {"sources": len(rows), "failed": len(failed), "fail_share": len(failed) / len(rows),
            "qe_mean": mean(qe), "qe0_mean": mean(a["qe0"] for _, a in rows),
            "qe_delta": mean(qe) - mean(a["qe0"] for _, a in rows),
            "ratio_median": median(s["tokens_out"] / a["l0"] for s, a in rows),
            "p_c2_mean": mean(float(scores["c2"][s["uid"]]["p_c2"]) for s, _ in rows),
            "c2_origineel_share": mean(float(float(scores["c2"][s["uid"]]["p_c2"]) >= 0.5) for s, _ in rows)}


def stop_rules7(val, iteration_fail_share, k1_old_ref, previous=None):
    """The stop rules after an iteration (spec §6.4; Yeb 20-21 Nov: S4 [0.85, 1.15], no S5): S1 RL-val's failures
    (not ended, loop, empty, English left) > 2%; S2 RL-val's mean QE - the anchor's > 0.15 in this and the previous
    validation; S3 the k1 KL per token of the iteration's samples (mean of log pi_old - log pi_ref) > 0.30; S4 the median
    l / l0 on RL-val outside [0.85, 1.15]; S6 the failures among the iteration's samples > 10% (c2_overflow included).
    previous: the previous iteration's result (its S2 value)."""
    s2_before = (previous or {}).get("rules", {}).get("S2", {}).get("value")
    rules = {
        "S1": {"value": val["fail_share"], "bound": S1_MAX, "hit": val["fail_share"] > S1_MAX},
        "S2": {"value": val["qe_delta"], "bound": S2_MAX, "previous": s2_before,
               "hit": val["qe_delta"] > S2_MAX and s2_before is not None and s2_before > S2_MAX},
        "S3": {"value": k1_old_ref, "bound": S3_MAX, "hit": k1_old_ref > S3_MAX},
        "S4": {"value": val["ratio_median"], "bound": list(S4_BAND),
               "hit": not S4_BAND[0] <= val["ratio_median"] <= S4_BAND[1]},
        "S6": {"value": iteration_fail_share, "bound": S6_MAX, "hit": iteration_fail_share > S6_MAX},
    }
    return {"rules": rules, "stop": [k for k, v in rules.items() if v["hit"]], "validation": val}


def stop7(args):
    plan = read_rows(args.plan)
    samples = read_samples(args.val_samples)
    scores = read_scores(args.val_scores, {s["uid"] for s in samples})
    val = validation_measures(samples, scores, plan)
    report = json.loads(Path(args.iter_report).read_text())
    summary = json.loads(Path(args.train_summary).read_text())
    previous = json.loads(Path(args.previous).read_text()) if args.previous else None
    out = stop_rules7(val, report["fail_share"], summary["grpo"]["passes"]["k1_old_ref"], previous)
    out.update({"iteration": args.iteration, "sampler_gap": summary["grpo"]["passes"]["sampler_gap"],
                "tis_next": summary["grpo"]["passes"]["sampler_gap"] > TIS_GAP})
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    print(f"STOP7 {json.dumps({'iteration': args.iteration, 'stop': out['stop'], 'tis_next': out['tis_next']})}")
    return out


# ---------------------------------------------------------------------------------------------------- JAX

_PASS_STEPS = {}


def pass_step(config):
    """The jitted passes of a batch -> (ref log-probabilities, old log-probabilities, the batch's ROW_KEY): the model with
    its LoRA at zero (g4dpo.without_lora: the reference, the start model) and the model as it is (the LoRA of the
    iteration: the old policy), both g4dpo.model_logp in one program; one per run"""
    from flax import nnx

    key = id(config)
    if key not in _PASS_STEPS:
        def both(model, data):
            micro = config.micro_batch_size_to_eval_on
            d = {k: data[k][:micro] for k in g4dpo.KEYS}
            ref = g4dpo.model_logp(g4dpo.without_lora(model), config, d, False)
            old = g4dpo.model_logp(model, config, d, False)
            return ref, old, data[ROW_KEY][:micro]

        _PASS_STEPS[key] = nnx.jit(both)
    return _PASS_STEPS[key]


def logp_passes(model, config, path, mesh, rows, batches):
    """The two passes over the first `batches` global batches of an iter7 file, gathered from every process: ({"ref_logp",
    "old_logp"}: float32 [batches x rows, seq] in the file's row order, 0 off the targets), the seconds of each batch"""
    import time

    import jax
    from jax.experimental import multihost_utils

    step = pass_step(config)
    arrays, _ = read_packed7(path)
    seq = arrays["inputs"].shape[1]
    out = {k: np.full((batches * rows, seq), np.nan, np.float32) for k in PASS_KEYS}
    seconds = []
    for batch in global_iterator7(path, mesh, rows, batches, row_ids=True):
        t0 = time.perf_counter()
        ref, old, ids = jax.block_until_ready(step(model, batch))
        ids = np.asarray(multihost_utils.process_allgather(ids, tiled=True)).reshape(-1)
        for k, x in (("ref_logp", ref), ("old_logp", old)):
            x = np.asarray(multihost_utils.process_allgather(x, tiled=True), np.float32)
            if x.shape != (rows, seq) or len(ids) != rows:
                raise SystemExit(f"{path}: the passes gave {x.shape} for rows {ids.shape}")
            out[k][ids] = x
        seconds.append(time.perf_counter() - t0)
    for k, x in out.items():
        if not np.isfinite(x).all():
            raise SystemExit(f"{path}: {k}: {int((~np.isfinite(x)).any(axis=1).sum())} rows not finite (or missing)")
    return out, seconds


def write_passes(out_dir, passes, identity, record):
    """The passes of a run (PASSES_FILE: ref_logp, old_logp) and PASSES_META: what they are the passes of, the record,
    the shape, the file's sha256"""
    out_dir = Path(out_dir)
    np.savez(out_dir / PASSES_FILE, **{k: np.ascontiguousarray(passes[k], np.float32) for k in PASS_KEYS})
    meta = {"identity": g4dpo.as_json(identity), "record": g4dpo.as_json(record),
            "shape": list(passes["ref_logp"].shape), "sha256": file_sha256(out_dir / PASSES_FILE)}
    (out_dir / PASSES_META).write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n")
    return meta


def read_passes(path, identity, rows, seq):
    """--passes-in: a run's PASSES_FILE with its PASSES_META beside it, refused unless the meta's identity is this run's
    (every key), the file is the one the meta names, and both arrays are finite float32 [rows, seq]"""
    npz = Path(path)
    meta_path = npz.parent / PASSES_META
    if not npz.is_file() or not meta_path.is_file():
        raise SystemExit(f"--passes-in {path}: no such file or no {meta_path} beside it")
    meta = json.loads(meta_path.read_text())
    have, want = meta.get("identity") or {}, g4dpo.as_json(identity)
    differ = sorted(k for k in set(have) | set(want) if have.get(k) != want.get(k))
    if differ:
        raise SystemExit(f"--passes-in {path}: not this run's passes, they differ in {differ}: "
                         f"{json.dumps({k: [have.get(k), want.get(k)] for k in differ}, default=str)[:600]}")
    if file_sha256(npz) != meta.get("sha256"):
        raise SystemExit(f"--passes-in {path}: its sha256 is not the one {meta_path} names")
    with np.load(npz) as z:
        passes = {k: z[k] for k in PASS_KEYS}
    for k, x in passes.items():
        if x.dtype != np.float32 or x.shape != (rows, seq) or not np.isfinite(x).all():
            raise SystemExit(f"--passes-in {path}: {k} is {x.dtype} {x.shape}, not finite float32 {(rows, seq)}")
    return passes, meta


def make_grpo_loss(config, opts, is_train):
    """The GRPO loss in the signature of MaxText's SFT loss (train_sft.use_maxtext_loss_function): the policy's
    log-probabilities of the batch (g4dpo.model_logp), the batch's adv, ref_logp and old_logp (and tis_weight with TIS);
    aux holds xent_sum and total_weights (an SFT eval file's loss, as MaxText's) and the batch's sums (grpo_<sum>). A
    batch without adv is an SFT eval file's; a training batch always has the four."""
    import jax.numpy as jnp

    micro = config.micro_batch_size_to_train_on if is_train else config.micro_batch_size_to_eval_on
    dropout = bool(config.enable_dropout) if is_train else False

    def loss_func(model, inputs, inputs_position, inputs_segmentation, targets, targets_position,
                  targets_segmentation, adv=None, sampler_logp=None, ref_logp=None, old_logp=None, tis_weight=None):
        data = {"inputs": inputs, "inputs_position": inputs_position, "inputs_segmentation": inputs_segmentation,
                "targets": targets, "targets_position": targets_position,
                "targets_segmentation": targets_segmentation}
        data = {k: v[:micro] for k, v in data.items()}
        logp = g4dpo.model_logp(model, config, data, dropout)
        aux = {"xent_sum": -jnp.sum(logp), "total_weights": jnp.sum(data["targets_segmentation"] != 0)}
        if adv is None:
            if is_train:
                raise ValueError("--grpo: a training batch without adv")
            return aux["xent_sum"] / jnp.maximum(aux["total_weights"], 1), aux
        if ref_logp is None or old_logp is None:
            raise ValueError("--grpo: a batch with adv but without ref_logp and old_logp (the passes)")
        if opts["tis"] and tis_weight is None:
            raise ValueError(f"--grpo with TIS: a batch without {TIS_KEY}")
        loss, sums = grpo_terms(logp, old_logp[:micro], ref_logp[:micro], adv[:micro],
                                data["targets_segmentation"] != 0, opts, jnp,
                                tis=tis_weight[:micro] if opts["tis"] else None)
        return loss, {**aux, **{f"grpo_{k}": v for k, v in sums.items()}}

    return loss_func


# ---------------------------------------------------------------------------------------------------- main

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pool7")
    p.add_argument("--cands", required=True, help="a directory with cands-flan, cands-novels, cands-bt .parquet (id, en, "
                                                   "doc, optionally stratum)")
    p.add_argument("--evalsets", required=True, help="$E/data/evalsets.parquet (13-grams)")
    p.add_argument("--blind", required=True, help="blind100.jsonl (13-grams)")
    p.add_argument("--exclude-docs", action="append", default=[], help="a text file of documents to leave out")
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--iterations", type=int, default=ITERATIONS7)
    p.add_argument("--per-iter", type=int, default=SOURCES7)
    p.add_argument("--out", required=True)
    p = sub.add_parser("plan7")
    p.add_argument("--pool", required=True, help="pool7's directory")
    p.add_argument("--anchor-samples", required=True, help="the anchor's translations (w<k>/ shards and DONE)")
    p.add_argument("--anchor-scores", required=True, help="the anchor's scores (<role>/w<m>/ shards and DONE-<k>)")
    p.add_argument("--iterations", type=int, default=ITERATIONS7)
    p.add_argument("--per-iter", type=int, default=SOURCES7)
    p.add_argument("--out", required=True)
    p = sub.add_parser("segments7")
    p.add_argument("--plan", required=True, help="plan7's plan.parquet")
    p.add_argument("--iteration", type=int, required=True, help="the iteration (-1: RL-val)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", required=True)
    p = sub.add_parser("iter7")
    p.add_argument("--plan", required=True)
    p.add_argument("--iteration", type=int, required=True)
    p.add_argument("--samples", required=True, help="iter/<i>/samples/ (w<k>/ shards and DONE)")
    p.add_argument("--scores", required=True, help="iter/<i>/scores/ (<role>/w<m>/ shards and DONE-<k>)")
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--seq", type=int, default=SEQ7)
    p.add_argument("--rows", type=int, default=ROWS7)
    p.add_argument("--out", required=True)
    p = sub.add_parser("stop7")
    p.add_argument("--plan", required=True)
    p.add_argument("--iteration", type=int, required=True)
    p.add_argument("--val-samples", required=True, help="val/<i>/samples/")
    p.add_argument("--val-scores", required=True, help="val/<i>/scores/")
    p.add_argument("--iter-report", required=True, help="iter7's report.json of the iteration")
    p.add_argument("--train-summary", required=True, help="the iteration's sft_lora.py --grpo summary.json")
    p.add_argument("--previous", default=None, help="the previous iteration's stop7 result (S2)")
    p.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    {"pool7": pool7, "plan7": plan7, "segments7": segments7, "iter7": iter7, "stop7": stop7}[args.cmd](args)


if __name__ == "__main__":
    main()
