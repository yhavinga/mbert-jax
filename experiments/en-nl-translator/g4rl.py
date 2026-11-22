#!/usr/bin/env python3
"""Arm G4-7 (spec translator-sft/g4-rl-spec.md; amendment G4-7, protocol-g4.md §14): online RL (GRPO) with LoRA on
Gemma 4 31B-it as the en->nl translator, route C-phase. Per iteration i the four workers of mbert-corpus translate their
part of the iteration's sources G = 8 times each (translate.py --copies, sampling at T = 1, the LoRA of iteration i
merged), the judges of mbert-mix score every sample (MetricX-24-QE; C2, the reward classifier), this module turns the
samples and scores into rewards, advantages and a packed training file, and one job of sft_lora.py --grpo over the 16
chips computes the reference's and the old policy's log-probabilities of every sample token and takes one optimizer step
per batch, the optimizer state continued from iteration to iteration (--resume-step).

Decisions of Yeb (20-21 Nov) over the spec: C2 on dmbert v2 (its token bound 1,022: the judges' c2_overflow); stop rules
S1-S4 and S6, S4's band [0.85, 1.15]; sources of at most 256 Gemma 4 tokens; the mix FLAN 10%, novels 45%, BT 45% (21 Nov
~11:50, addendum G4-7.4d) with sources of at least 20 Gemma 4 tokens (the main session's reading of it); the rest as the
spec advises: start G4-0
(the reference is the model with its LoRA at zero), G = 8, 512 sources per iteration, LR 5e-6, beta 0.04 with k2 and
sg(rho), Dr. GRPO's advantage (no division by the group's std), no overlap.

Pure parts (numpy, no JAX):
  pool7     the sources: FLAN 10%, novels 45%, BT 45% (BT equal per source), English of 20 to 256 Gemma 4 tokens, no
            13-gram of the evaluation sets or the blind passages, the excluded documents out; RL-val (128 per part)
            drawn first, its documents kept out of the pool; the order of the iterations (seed SEED7/order); a reserve
            of RESERVE7 per part for the sources the anchor drops
  plan7     the anchor (G4-0's greedy translation of every pool and RL-val source, with its QE and C2): a source whose
            anchor did not end, loops, is empty or leaves English is dropped; per part the first sources that remain
            make the iterations (512 each: 51 FLAN, 231 novels, 230 BT) and RL-val (128 per part)
  segments7 an iteration's (or RL-val's) sources for the workers' translate.py
  iter7     an iteration's samples and scores (the bucket protocol of spec §3.4, checked: markers, sha256, the uids of
            samples and scores equal) -> reward (§4.1), advantage (Dr. GRPO, §4.3; groups with max - min < 0.05 left out),
            the packed training file (one sample a segment of MaxText's SFT row: the prompt of g4lora.prompt_text and the
            sampled token ids, stop token included; per token adv and sampler_logp beside the targets), the report
  stop7     the stop rules after an iteration (RL-val greedy with the new adapter, the iteration's samples, the passes)
  grpo_terms, the loss's arithmetic (numpy or jax.numpy), and the sampler/learner gap.
JAX parts (sft_lora.py --grpo): logp_passes (the reference, the model with its LoRA at zero, and the old policy, the model
as loaded, in one jitted forward per batch), make_grpo_loss (MaxText's SFT loss signature).

The bucket protocol (amendment G4-7.4a; RL = gs://.../g4arm/rl/G4-7), per job (iter/<i>, val/<i>, anchor):
<job>/samples/w<k>/shard-<n>.parquet (SAMPLE_COLUMNS, written by translate.py --copies) and, last, its DONE ({"rows",
"shards": {shard: {"rows", "sha256"}}}); <job>/scores/<role>/w<m>/shard-<k>-<n>.parquet (SCORE_COLUMNS[role]; qe null
for an empty hypothesis, p_c2 and logit_c2 null with c2_overflow) and DONE-<k> ({"rows", "shards": {shard: {"rows",
"samples_sha256"}}}), role metricx (w0, w2: the halves of the uids) or c2 (w1). The readers here check all of it.
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
PARTS7 = {"flan": 0.1, "novels": 0.45, "bt": 0.45}   # Yeb 21 Nov ~11:50 (G4-7.4d); was 0.4 / 0.3 / 0.3
RLVAL_PER_PART7 = 128
MAX_SOURCE7 = 256                    # Gemma 4 tokens of the English (Yeb 20-21 Nov; class medium of translate.py)
# a shorter source gives a group of 8 nearly identical samples and so no signal (G4-7.4d; the main session, 21 Nov)
MIN_SOURCE7 = 20
LENGTH_CLASS7 = "medium"             # prefill 256/384, decode limit 448, 52 slots: one engine per worker
RESERVE7 = 0.03                      # pool7 draws this share more per part, for the sources plan7 drops
SEQ7 = 4096
ROWS7 = 16                           # a global batch: 16 rows of 4,096 (per_device_batch_size 1 on 16 chips)
LR7 = 5e-6
WARMUP7 = 10                         # linear warmup steps, then constant (decay 0)
# the reward (spec §4.1)
FAIL_REWARD = -30.0                  # G4-7.4e: below the lowest non-fail reward, -12 - 2 x 7.5 - 2 (ln 1.5 - ln 1.15) = -27.5
LOGIT_CLIP = 12.0                    # G4-7.4e (Yeb 21 Nov): C2's logits of G4-0's output lie at -12.2 .. 8.3; +-6 clipped them flat
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
SAMPLE_COLUMNS = ("uid", "id", "j", "source", "hypothesis", "token_ids", "sampler_logp", "ended", "decode_loop",
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


def candidate_filter(rows, counter, blocked, excluded_docs=frozenset(), max_tokens=MAX_SOURCE7, min_tokens=MIN_SOURCE7):
    """(the candidates that may be sources, {reason: count}): English of min_tokens to max_tokens Gemma 4 tokens (the
    ids of the text alone, as translate.py's segments count them), not empty, no 13-gram of the evaluation sets or the blind
    passages, not of an excluded document, one source per distinct English text (the first id)"""
    import select_passages as sp

    lengths = counter.lengths([r["en"] for r in rows])
    keep, drops, seen = [], {"empty": 0, "tokens": 0, "short": 0, "ngram": 0, "excluded_doc": 0, "duplicate": 0}, set()
    for r, n in sorted(zip(rows, lengths), key=lambda x: x[0]["id"]):
        text = (r["en"] or "").strip()
        if not text:
            drops["empty"] += 1
        elif n > max_tokens:
            drops["tokens"] += 1
        elif n < min_tokens:
            drops["short"] += 1
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


def read_docs(path):
    return {line.strip() for line in Path(path).read_text().splitlines() if line.strip()}


def pool7(args, counter=None):
    """The pool (G4-7.2): C2's documents ($RL/c2/data/docs.txt, <source>:<document> as g4c2.doc_key makes them) out of the
    BT part, which must hold some of them (else its doc keys are not C2's); 13-grams of the evaluation sets, the FLAN
    evaluation rows and the blind passages out"""
    import g4arm
    import sft_build

    if not Path(args.c2_docs).is_file():
        raise SystemExit(f"pool7: no {args.c2_docs} (g4c2.py data7 runs before pool7, G4-7.2)")
    counter = counter or g4arm.TokenCounter(args.tokenizer)
    blocked = g4arm.checked_blocked(args.evalsets, args.blind)
    blocked |= sft_build.flan_eval_ngrams(read_rows(args.flan_eval, columns=["inputs", "targets"]))
    c2_docs = read_docs(args.c2_docs)
    excluded = set(c2_docs)
    for path in args.exclude_docs or []:
        excluded |= read_docs(path)
    cands, report = {}, {"arm": ARM7, "seed": SEED7, "shares": PARTS7, "min_tokens": MIN_SOURCE7, "max_tokens": MAX_SOURCE7,
                         "parts": {}, "c2_docs": len(c2_docs), "excluded_docs": len(excluded)}
    for part in PARTS7:
        rows = read_rows(Path(args.cands) / f"cands-{part}.parquet")
        missing = sorted({"id", "en", "doc"} - set(rows[0] if rows else {}))
        if missing or len({r["id"] for r in rows}) != len(rows):
            raise SystemExit(f"pool7: cands-{part}.parquet needs unique ids and the columns id, en, doc (missing {missing})")
        if part == "bt" and not {r["doc"] for r in rows} & c2_docs:
            raise SystemExit("pool7: no BT candidate is of a document of C2: the doc keys are not <source>:<document>")
        cands[part], drops = candidate_filter(rows, counter, blocked, excluded)
        report["parts"][part] = {"candidates": len(rows), "kept": len(cands[part]), "drops": drops}
    rows = draw_pool(cands, args.iterations, args.per_iter, args.rlval)
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


# ---------------------------------------------------------------------------------------------------- the candidates

CANDS_MARGIN7 = 2.0                  # cands7 draws this many times what pool7 takes of a part (RL-val and pool)
CANDS_CHUNK7 = 16384                 # texts per call of the token counter in cands7's draw
CANDS_SCHEMA = (("id", "string"), ("en", "string"), ("doc", "string"), ("stratum", "string"))
# arm F's pool, and so FLAN-nl, has no task named after SICK or DBRD (flan-nl/protocol.md): checked, a hit stops cands7
FLAN_BARRED_TASKS = ("sick", "dbrd")


def pool_need(iterations=ITERATIONS7, per_iter=SOURCES7, rlval=RLVAL_PER_PART7, reserve=RESERVE7):
    """{part: the sources draw_pool takes of it}: RL-val and the pool, each with its reserve"""
    counts = part_counts(per_iter)
    return {p: math.ceil(rlval * (1 + reserve)) + math.ceil(iterations * counts[p] * (1 + reserve)) for p in PARTS7}


def draw_candidates(rows, n, counter, seed, max_tokens=MAX_SOURCE7, chunk=CANDS_CHUNK7, min_tokens=MIN_SOURCE7):
    """(rows, funnel): of rows in a seeded order (sorted by id, then shuffled with seed) the first n whose English,
    stripped, is not empty, not a text looked at before and of min_tokens to max_tokens Gemma 4 tokens (pool7's count of
    the same text). The tokens are counted a chunk at a time, so about as many texts as the draw needs."""
    order = sorted(rows, key=lambda r: r["id"])
    if len({r["id"] for r in order}) != len(order):
        raise SystemExit(f"cands7: an id twice among the rows of {seed}")
    random.Random(seed).shuffle(order)
    out, seen, i = [], set(), 0
    funnel = {"looked_at": 0, "empty": 0, "duplicate": 0, "tokens": 0, "short": 0}
    while len(out) < n and i < len(order):
        part = order[i:i + chunk]
        i += len(part)
        texts = sorted({(r["en"] or "").strip() for r in part} - seen - {""})
        lengths = dict(zip(texts, counter.lengths(texts)))
        for r in part:
            if len(out) == n:
                break
            funnel["looked_at"] += 1
            text = (r["en"] or "").strip()
            if not text:
                funnel["empty"] += 1
            elif text in seen:
                funnel["duplicate"] += 1
            else:
                seen.add(text)
                if lengths[text] > max_tokens:
                    funnel["tokens"] += 1
                elif lengths[text] < min_tokens:
                    funnel["short"] += 1
                else:
                    out.append({**r, "en": text})
    funnel["drawn"] = len(out)
    return out, funnel


def published_key(row_id):
    """(host, position) of a FLAN-nl row id w<host>-<position>"""
    host, sep, position = str(row_id)[1:].partition("-")
    if str(row_id)[:1] != "w" or not sep or not host.isdigit() or not position.isdigit():
        raise SystemExit(f"cands7: {row_id!r} is not a FLAN-nl row id w<host>-<position>")
    return int(host), int(position)


def flan_cands(plan_dir, kept_dir, build_report, flan_eval):
    """(rows, info): FLAN-nl's English segments of the examples in the published set. An example is in it when the plan
    has it to translate (status translate), assemble kept it (kept_dir/w<host>.parquet, its position) and no check of
    the publication's build left it out (build_report.json, checks.<check>.ids); its segments are the S pieces of its
    instruction and answer that are not one of the plan's template strings. A segment is one distinct text of the plan
    and can be in several examples: its doc is the first of them by (host, position), flan:w<host>-<position> (the
    published row's id). Out: the segments of the FLAN evaluation rows and every segment of their examples."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    plan_dir, kept_dir = Path(plan_dir), Path(kept_dir)
    files = sorted(kept_dir.glob("w*.parquet"))
    if not files:
        raise SystemExit(f"cands7: no w<host>.parquet in {kept_dir}")
    kept, assembled = set(), 0
    for path in files:
        host = int(path.stem[1:])
        positions = pq.read_table(path, columns=["position"]).column("position").to_pylist()
        assembled += len(positions)
        kept |= {(host, int(p)) for p in positions}
    if len(kept) != assembled:
        raise SystemExit(f"cands7: an example twice in {kept_dir}")
    build = json.loads(Path(build_report).read_text())
    left = {published_key(x): check for check, entry in build["checks"].items() for x in entry["ids"]}
    if (assembled, assembled - len(left)) != (build["rows_of_the_experiment"], build["rows"]) or not set(left) <= kept:
        raise SystemExit(f"cands7: {kept_dir} has {assembled} rows and {build_report} leaves out {len(left)} of them "
                         f"({len(set(left) - kept)} not among them); the build has {build['rows_of_the_experiment']} "
                         f"and {build['rows']}: not the published set's examples")
    published = kept - set(left)
    templates = {int(v) for v in json.loads((plan_dir / "plan.json").read_text()).get("template_segments", {}).values()}
    ev = read_rows(flan_eval, columns=["host", "position", "segments"])
    eval_examples = {(int(r["host"]), int(r["position"])) for r in ev}
    eval_segments = {int(s) for r in ev for s in r["segments"]}
    ex = pq.read_table(plan_dir / "examples.parquet",
                       columns=["host", "position", "source", "status", "instruction", "answer"]).to_pydict()
    doc_of, examples_of = {}, {}
    info = {"assembled": assembled, "left_by_checks": {c: sum(v == c for v in left.values()) for c in build["checks"]},
            "published": len(published), "published_to_translate": 0, "published_other_status": {},
            "template_pieces": 0}
    for h, p, source, status, ins, ans in zip(*(ex[c] for c in ("host", "position", "source", "status", "instruction",
                                                                 "answer"))):
        key = (int(h), int(p))
        is_eval, is_published = key in eval_examples, key in published
        if not (is_eval or is_published):
            continue
        segments = [int(v) for kind, v in json.loads(ins) + json.loads(ans) if kind == "S"]
        if is_eval:
            eval_segments.update(segments)
        if not is_published:
            continue
        if status != "translate":
            info["published_other_status"][status] = info["published_other_status"].get(status, 0) + 1
            continue
        if any(t in str(source).lower() for t in FLAN_BARRED_TASKS):
            raise SystemExit(f"cands7: w{key[0]}-{key[1]} is of {source}, a task FLAN-nl does not have (SICK, DBRD)")
        info["published_to_translate"] += 1
        for s in segments:
            if s in templates:
                info["template_pieces"] += 1
                continue
            examples_of[s] = examples_of.get(s, 0) + 1
            if s not in doc_of or key < doc_of[s]:
                doc_of[s] = key
    del ex
    missing = len(published) - info["published_to_translate"] - sum(info["published_other_status"].values())
    if missing:
        raise SystemExit(f"cands7: {missing} published examples are not in {plan_dir}/examples.parquet")
    want = sorted(set(doc_of) - eval_segments)
    rows = []
    for path in sorted(plan_dir.glob("segments-w*.parquet")):
        t = pq.read_table(path, columns=["id", "text"])
        t = t.filter(pc.is_in(t["id"], value_set=pa.array(want, pa.int64()))).to_pydict()
        for i, text in zip(t["id"], t["text"]):
            h, p = doc_of[int(i)]
            rows.append({"id": f"flan:{int(i)}", "en": text, "doc": f"flan:w{h}-{p}", "stratum": ""})
    if len(rows) != len(want) or len({r["id"] for r in rows}) != len(rows):
        raise SystemExit(f"cands7: {len(want)} segments wanted, {len(rows)} texts in {plan_dir}/segments-w*.parquet")
    info.update({"eval_examples": len(eval_examples), "segments": len(doc_of),
                 "segments_in_several_examples": sum(n > 1 for n in examples_of.values()),
                 "segments_of_the_eval_rows_out": len(doc_of) - len(want), "candidates": len(rows)})
    return rows, info


def novels_cands(b1, g44_report, novels_eval):
    """(rows, info): the English passages (group en_orig) of the -b1- novels as G4-4 took them, build4's steps (arm RE's
    checks, one Dutch edition per English book, one English id per work, no near duplicate) that must give G4-4's books
    and rows (its data report), without the books of the eval set novels (G4-4.10); id novels:<row id>, doc
    novels:<English book id>"""
    import g4arm

    report4 = json.loads(Path(g44_report).read_text())
    sha = g4arm.sha256_file(b1)
    if report4.get("arm") != g4arm.ARM4 or sha != report4["novels"]["sha256"]:
        raise SystemExit(f"cands7: {b1} is not the -b1- file of G4-4's set ({g44_report})")
    rows, groups = g4arm.read_b1(b1)
    chosen, _ = g4arm.b1_editions(rows)
    chosen, _ = g4arm.same_works(rows, chosen)
    del rows
    chosen, near = g4arm.near_duplicates(chosen)
    want = report4["novels"]["near_duplicates"]
    if (near["books_left"], near["rows_left"]) != (want["books_left"], want["rows_left"]):
        raise SystemExit(f"cands7: build4's steps give {near['books_left']} books, {near['rows_left']} rows; G4-4's "
                         f"report {want['books_left']}, {want['rows_left']}")
    books = {str(en): rs for en, rs in chosen.items()}
    held = {str(b["en_book_id"]) for b in json.loads(Path(novels_eval).read_text())["draw"]["books"]}
    if len(held) != g4arm.NOVELS_EVAL_BOOKS or not held <= set(books):
        raise SystemExit(f"cands7: {novels_eval} names {len(held)} books, {len(held - set(books))} of them not G4-4's; "
                         f"the eval set novels has {g4arm.NOVELS_EVAL_BOOKS}")
    out = [{"id": f"novels:{r['id']}", "en": r["en"], "doc": f"novels:{en}", "stratum": ""}
           for en, rs in books.items() if en not in held for r in rs]
    info = {"sha256": sha, "groups": groups, "books": len(books), "rows": near["rows_left"],
            "books_of_the_eval_set_out": sorted(held, key=int), "rows_of_the_eval_set_out": sum(len(books[b]) for b in held),
            "candidates": len(out)}
    return out, info


def bt_cands(bt):
    """({source: rows}, revisions, info): the English (en) of the train split of the five <source>-nl-en-gemma4 sets at
    G4-3's revisions (fetch --arm G4-3, every file's sha256 checked), without the rows with hard_fails (a check of the
    English failed: not_ended, decode_loop, dutch_left, len_ratio, structure, newline_doubled, chatter); id bt:<id>,
    stratum the source. The doc comes after the draw (g4c2.doc_key, C2's key)."""
    import g4arm

    revisions = g4arm.revisions3(bt)
    out = {s: [] for s in g4arm.BT_REPOS3}
    info = {s: {"rows": 0, "hard_fails": 0} for s in g4arm.BT_REPOS3}
    for r in g4arm.read_bt(bt, revisions, split="train"):
        info[r["source"]]["rows"] += 1
        if r["hard_fails"]:
            info[r["source"]]["hard_fails"] += 1
        else:
            out[r["source"]].append({"id": f"bt:{r['pid']}", "en": r["en"], "pid": r["pid"], "stratum": r["source"]})
    for s, rows in out.items():
        info[s]["candidates"] = len(rows)
    return out, revisions, info


def cands7(args, counter=None):
    """pool7's candidates (G4-7.2, spec §6.2): cands-flan, cands-novels, cands-bt .parquet (id, en, doc, stratum) and
    cands.json (the inputs with their sha256, the counts, what fell out and why). Per part, and per source for BT in
    equal numbers, margin x what pool7 takes of it, drawn in a seeded order (SEED7/cands/<part>[/<source>]) from the
    texts that pass pool7's own filters on the English: not empty, a text once, MIN_SOURCE7 to MAX_SOURCE7 Gemma 4 tokens
    (the same counter on the same text, so pool7 drops none of them for these). Left to pool7: the 13-grams, C2's documents
    and RL-val's documents."""
    import g4arm
    import g4c2

    counter = counter or g4arm.TokenCounter(args.tokenizer)
    need = pool_need(args.iterations, args.per_iter, args.rlval)
    want = {p: math.ceil(args.margin * n) for p, n in need.items()}
    out = Path(args.out)
    report = {"arm": ARM7, "seed": f"{SEED7}/cands", "margin": args.margin, "max_tokens": MAX_SOURCE7,
              "min_tokens": MIN_SOURCE7, "shares": PARTS7,
              "pool7_takes": need, "wanted": want, "parts": {},
              "left_to_pool7": ["13-grams of the evaluation sets, the blind passages and the FLAN evaluation rows",
                                "C2's documents (docs.txt)", "RL-val's documents in the pool"]}
    drawn = {}
    print("PROGRESS flan", flush=True)
    rows, info = flan_cands(args.flan_plan, args.flan_kept, args.flan_build_report, args.flan_eval)
    drawn["flan"], info["draw"] = draw_candidates(rows, want["flan"], counter, f"{SEED7}/cands/flan")
    report["parts"]["flan"] = info
    print("PROGRESS novels", flush=True)
    rows, info = novels_cands(args.novels, args.novels_report, args.novels_eval)
    drawn["novels"], info["draw"] = draw_candidates(rows, want["novels"], counter, f"{SEED7}/cands/novels")
    report["parts"]["novels"] = info
    print("PROGRESS bt", flush=True)
    by_source, revisions, info = bt_cands(args.bt)
    per_source = math.ceil(want["bt"] / len(by_source))
    drawn["bt"] = []
    for source, rows in by_source.items():
        got, info[source]["draw"] = draw_candidates(rows, per_source, counter, f"{SEED7}/cands/bt/{source}")
        drawn["bt"] += [{**r, "doc": g4c2.doc_key(r["pid"])} for r in got]
    report["parts"]["bt"] = {"per_source": per_source, "sources": info}
    del rows, by_source
    short = {p: len(drawn[p]) for p in PARTS7 if len(drawn[p]) < want[p]}
    short.update({f"bt/{s}": i["draw"]["drawn"] for s, i in info.items() if i["draw"]["drawn"] < per_source})
    if short:
        raise SystemExit(f"cands7: too few candidates {short}; wanted {want}, {per_source} per BT source")
    ids = [r["id"] for p in PARTS7 for r in drawn[p]]
    if len(set(ids)) != len(ids):
        raise SystemExit("cands7: an id in two parts")
    report["files"] = {}
    for part in PARTS7:
        path = out / f"cands-{part}.parquet"
        write_parquet([{k: r[k] for k, _ in CANDS_SCHEMA} for r in sorted(drawn[part], key=lambda r: r["id"])], path,
                      schema(CANDS_SCHEMA))
        report["files"][path.name] = {"rows": len(drawn[part]), "sha256": file_sha256(path)}
    def files(d, pattern):
        return {f.name: g4arm.sha256_file(f) for f in sorted(Path(d).glob(pattern))}

    report["inputs"] = {
        "flan_plan": {"path": str(args.flan_plan), "sha256": {**files(args.flan_plan, "examples.parquet"),
                                                              **files(args.flan_plan, "plan.json"),
                                                              **files(args.flan_plan, "segments-w*.parquet")}},
        "flan_kept": {"path": str(args.flan_kept), "sha256": files(args.flan_kept, "w*.parquet")},
        **{k: {"path": str(getattr(args, k)), "sha256": g4arm.sha256_file(getattr(args, k))}
           for k in ("flan_build_report", "flan_eval", "novels_report", "novels_eval")},
        "novels": {"path": str(args.novels), "sha256": report["parts"]["novels"]["sha256"]},
        "bt": {"path": str(args.bt), "revisions": revisions},
        "tokenizer": {"path": str(args.tokenizer),
                      "sha256": g4arm.sha256_file(Path(args.tokenizer) / "tokenizer.json")
                      if (Path(args.tokenizer) / "tokenizer.json").is_file() else None}}
    (out / "cands.json").write_text(json.dumps(report, indent=1) + "\n")
    print(f"CANDS7 {json.dumps({'wanted': want, 'files': report['files']})}", flush=True)
    return report


def anchor_rows(samples, scores):
    """{source id: anchor} from the anchor's translations (the job anchor, one sample #0 per source, greedy) and their
    scores: the flags, l0 (tokens_out), qe0 (null for an empty one), C2's logit (null with c2_overflow)"""
    by_uid = {r["uid"]: r for r in samples}
    out = {}
    for u, r in by_uid.items():
        source_id, j = split_uid(u)
        if j != 0 or source_id in out:
            raise SystemExit(f"anchor: {u} is not one greedy translation #0 per source")
        out[source_id] = {"ended": bool(r["ended"]), "decode_loop": bool(r["decode_loop"]), "empty": bool(r["empty"]),
                          "english_left": bool(r["english_left"]), "l0": int(r["tokens_out"]),
                          "qe0": scores["metricx"][u]["qe"], "logit_c2_0": scores["c2"][u]["logit_c2"],
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
               ("tokens", "int64"), ("iteration", "int64"), ("position", "int64"), ("qe0", "float64"), ("l0", "int64"),
               ("logit_c2_0", "float64"), ("hypothesis0", "string"))


def plan7(args):
    pool = read_rows(Path(args.pool) / "pool.parquet")
    samples = read_samples(args.anchor_samples)
    scores = read_scores(args.anchor_scores, {r["uid"] for r in samples}, sample_dones(args.anchor_samples))
    anchors = anchor_rows(samples, scores)
    plan, report = plan_sources(pool, anchors, args.iterations, args.per_iter, args.rlval)
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
    """translate.py's segments (id, text, tokens) per worker: of an iteration of the plan (or RL-val: --iteration -1),
    or with --pool of every source of pool7's pool.parquet, RL-val's included (the job anchor, G4-7.4a: G4-0's greedy
    translation of the pool and RL-val, before plan7)"""
    if bool(args.plan) == bool(args.pool) or (args.plan and args.iteration is None) or (args.pool and
                                                                                         args.iteration is not None):
        raise SystemExit("segments7: --plan with --iteration, or --pool alone")
    if args.pool:
        rows, what = read_rows(args.pool), "anchor"
    else:
        rows, what = [r for r in read_rows(args.plan) if r["iteration"] == args.iteration], args.iteration
    if not rows:
        raise SystemExit(f"segments7: no source of {'the pool' if args.pool else f'iteration {args.iteration}'}")
    if len({r["id"] for r in rows}) != len(rows):
        raise SystemExit("segments7: a source id twice")
    for k, part in assign_workers7(rows, args.workers).items():
        write_parquet([{"id": r["id"], "text": r["en"], "tokens": r["tokens"]} for r in part],
                      Path(args.out) / f"segments-w{k}.parquet",
                      schema((("id", "string"), ("text", "string"), ("tokens", "int64"))))
    print(f"SEGMENTS7 {json.dumps({'iteration': what, 'sources': len(rows), 'workers': args.workers})}")


# ---------------------------------------------------------------------------------------------------- the bucket protocol

def samples_done(shards):
    """A samples DONE marker (G4-7.4a): {"rows", "shards": {name: {"rows", "sha256"}}} of {name: (rows, digest)}"""
    return {"rows": int(sum(n for n, _ in shards.values())),
            "shards": {name: {"rows": int(n), "sha256": digest} for name, (n, digest) in sorted(shards.items())}}


def sample_dones(root):
    """{k: the DONE of samples/w<k>} under root (the job's samples directory)"""
    out = {}
    for d in sorted(p for p in Path(root).glob("w*") if p.is_dir()):
        if not (d / DONE).is_file():
            raise SystemExit(f"{d}: no {DONE} marker (the generation is not done)")
        out[d.name[1:]] = json.loads((d / DONE).read_text())
    return out


def read_samples(root):
    """Every sample of the w<k> directories under root (a job's samples): each directory's DONE names its shards with
    their rows and sha256; exactly those shards, unchanged; the columns SAMPLE_COLUMNS; uids unique and <id>#<j> of each
    row's id and j. Each row gets "source_id" (its id as a string) and "worker"."""
    import pyarrow.parquet as pq

    root = Path(root)
    dirs = sorted(p for p in root.glob("w*") if p.is_dir())
    if not dirs:
        raise SystemExit(f"{root}: no w<k> directory of samples")
    dones = sample_dones(root)
    rows, seen = [], set()
    for d in dirs:
        done = dones[d.name[1:]]
        names = sorted(p.name for p in d.glob("*.parquet"))
        if names != sorted(done["shards"]):
            raise SystemExit(f"{d}: shards {names[:3]}... ({len(names)}) are not the {len(done['shards'])} of {DONE}")
        n = 0
        for name in names:
            if file_sha256(d / name) != done["shards"][name]["sha256"]:
                raise SystemExit(f"{d / name}: its sha256 is not the one {DONE} names")
            table = pq.read_table(d / name)
            lacking = sorted(set(SAMPLE_COLUMNS) - set(table.column_names))
            if lacking:
                raise SystemExit(f"{d / name}: no columns {lacking}")
            shard_rows = table.to_pylist()
            if len(shard_rows) != done["shards"][name]["rows"]:
                raise SystemExit(f"{d / name}: {len(shard_rows)} rows, {DONE} says {done['shards'][name]['rows']}")
            for r in shard_rows:
                if r["uid"] in seen:
                    raise SystemExit(f"{d / name}: {r['uid']} twice")
                if split_uid(r["uid"]) != (str(r["id"]), int(r["j"])):
                    raise SystemExit(f"{d / name}: uid {r['uid']} is not id {r['id']} # j {r['j']}")
                if len(r["token_ids"]) != len(r["sampler_logp"]) or len(r["token_ids"]) != r["tokens_out"]:
                    raise SystemExit(f"{d / name}: {r['uid']} has {len(r['token_ids'])} token ids, "
                                     f"{len(r['sampler_logp'])} log-probabilities, tokens_out {r['tokens_out']}")
                seen.add(r["uid"])
                rows.append({**r, "source_id": str(r["id"]), "worker": d.name})
                n += 1
        if n != done["rows"]:
            raise SystemExit(f"{d}: {n} rows, {DONE} says {done['rows']}")
    return rows


def read_scores(root, uids, dones):
    """{role: {uid: row}} of the judges' score shards under root/<role>/w<m>/ (G4-7.4a), every role of SCORE_COLUMNS:
    each judge directory has a DONE-<k> for every generation worker k of `dones` (the samples' DONE per k), naming exactly
    that DONE's shards with the sha256 the judge read (= the DONE's) and the rows it scored; no uid twice in a role; each
    role's uids exactly the samples' (else FAILED with the missing). A score is finite or null: qe null only for an
    empty hypothesis, p_c2 and logit_c2 null only with c2_overflow (checked against the samples by the reward)."""
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
            for k, done in dones.items():
                marker = d / f"DONE-{k}"
                if not marker.is_file():
                    raise SystemExit(f"{marker}: missing (the judge has not scored every shard of w{k})")
                m = json.loads(marker.read_text())
                if set(m["shards"]) != set(done["shards"]) or any(
                        m["shards"][s]["samples_sha256"] != done["shards"][s]["sha256"] for s in done["shards"]):
                    raise SystemExit(f"{marker}: its shards or their sha256 are not those of samples/w{k}/{DONE}")
                if m["rows"] != per_worker.get(k, 0):
                    raise SystemExit(f"{marker}: {m['rows']} rows, {d} has {per_worker.get(k, 0)} of w{k}")
            if set(per_worker) - set(dones):
                raise SystemExit(f"{d}: scores of generation workers {sorted(set(per_worker) - set(dones))} without samples")
        missing, extra = sorted(set(uids) - set(got)), sorted(set(got) - set(uids))
        if missing or extra:
            raise SystemExit(f"{role}: {len(missing)} samples without a score (e.g. {missing[:5]}), {len(extra)} scores "
                             f"of no sample (e.g. {extra[:5]})")
        for u, r in got.items():
            values = [v for k, v in r.items() if k not in ("uid", "c2_overflow") and v is not None]
            if any(not math.isfinite(float(v)) for v in values):
                raise SystemExit(f"{role}: {u} has a score that is not finite: {r}")
            if role == "c2" and (r["c2_tokens"] is None or r["c2_overflow"] is None):
                raise SystemExit(f"c2: {u} has no c2_tokens or c2_overflow: {r}")
        out[role] = got
    return out


# ---------------------------------------------------------------------------------------------------- reward, advantage

def reward7(sample, qe, c2, anchor):
    """The reward of one sample (spec §4.1): fail -> FAIL_REWARD (no classifier, no QE); else clip(logit C2, -12, 12) -
    2 max(0, QE - QE0) - 2 max(0, |ln(l / l0)| - ln 1.15), with l and l0 the tokens_out of the sample and of the anchor
    (G4-0's greedy translation of the source). Returns {"reward", "fail": [reasons], "logit", "qe_pen", "len_pen",
    "ratio"}"""
    if qe is None and not sample["empty"]:
        raise SystemExit(f"{sample.get('uid')}: no QE, and its hypothesis is not empty")
    if c2["logit_c2"] is None and not c2["c2_overflow"]:
        raise SystemExit(f"{sample.get('uid')}: no logit of C2, and no c2_overflow")
    ratio = sample["tokens_out"] / anchor["l0"]
    checks = {"not_ended": not sample["ended"], "decode_loop": bool(sample["decode_loop"]), "empty": bool(sample["empty"]),
              "english_left": bool(sample["english_left"]), "length": not LEN_BAND[0] <= ratio <= LEN_BAND[1],
              "qe": qe is not None and qe > QE_FAIL, "c2_overflow": bool(c2["c2_overflow"])}
    fail = [k for k in FAILS if checks[k]]
    logit = None if c2["logit_c2"] is None else min(max(float(c2["logit_c2"]), -LOGIT_CLIP), LOGIT_CLIP)
    qe_pen = None if qe is None else QE_WEIGHT * max(0.0, qe - anchor["qe0"])
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
        terms = [reward7(s, scores["metricx"][s["uid"]]["qe"], scores["c2"][s["uid"]], src) for s in groups[sid]]
        adv, signal = advantages7([t["reward"] for t in terms])
        flat += not signal
        for s, t, a in zip(groups[sid], terms, adv):
            out.append({"uid": s["uid"], "source_id": sid, "j": int(s["j"]), "part": src["part"], "worker": s["worker"],
                        "reward": t["reward"], "adv": float(a), "signal": signal, "fail": t["fail"],
                        "logit": t["logit"], "qe_pen": t["qe_pen"], "len_pen": t["len_pen"], "ratio": t["ratio"],
                        "qe": scores["metricx"][s["uid"]]["qe"], "logit_c2": scores["c2"][s["uid"]]["logit_c2"],
                        "p_c2": scores["c2"][s["uid"]]["p_c2"], "c2_tokens": int(scores["c2"][s["uid"]]["c2_tokens"]),
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
            "len_pen_mean_ok": mean(r["len_pen"] for r in ok), "qe_mean": mean(r["qe"] for r in rows if r["qe"] is not None),
            "p_c2_mean": mean(r["p_c2"] for r in rows if r["p_c2"] is not None),
            "origineel_share": sum(r["p_c2"] is not None and r["p_c2"] >= 0.5 for r in rows) / n,
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
    scores = read_scores(args.scores, {s["uid"] for s in samples}, sample_dones(args.samples))
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
    # S2 over the sources whose translation has a QE (an empty one has none, and counts in S1)
    scored = [(float(scores["metricx"][s["uid"]]["qe"]), a["qe0"]) for s, a in rows
              if scores["metricx"][s["uid"]]["qe"] is not None]
    p = [scores["c2"][s["uid"]]["p_c2"] for s, _ in rows]
    return {"sources": len(rows), "failed": len(failed), "fail_share": len(failed) / len(rows), "qe_sources": len(scored),
            "qe_mean": mean(q for q, _ in scored), "qe0_mean": mean(q0 for _, q0 in scored),
            "qe_delta": mean(q - q0 for q, q0 in scored),
            "ratio_median": median(s["tokens_out"] / a["l0"] for s, a in rows),
            "c2_overflow": sum(bool(scores["c2"][s["uid"]]["c2_overflow"]) for s, _ in rows),
            "p_c2_mean": mean(x for x in p if x is not None),
            "c2_origineel_share": mean(float(x >= 0.5) for x in p if x is not None)}


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
    scores = read_scores(args.val_scores, {s["uid"] for s in samples}, sample_dones(args.val_samples))
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


# ---------------------------------------------------------------------------------------------------- the smoke test

# spec §8.3 and amendment G4-7.9: two mini-iterations of 32 sources x G = 8 on mbert-corpus (A: G4-0; B: phase 1's
# adapter merged), the judges of mbert-mix; the checks of smoke7 below. The bounds are the spec's.
SMOKE_BASE_MIN7 = 900                # the smoke's iterations N, N+1 under the same $RL, far from the run's 0..29
SMOKE_SOURCES7 = 32
SMOKE_SAMPLES7 = SMOKE_SOURCES7 * G7  # 256
SMOKE_MIN_OK7 = 248                  # §8.3 point 1: at least this many of the 256 ended (no loop, not empty)
SMOKE_QE_TOL7 = 1e-3                 # §8.3 point 2: the judges' QE against a loose metricx_qe.py score
SMOKE_C2_TOL7 = 1e-4                 # §8.3 point 2: the judge's p_c2 against a loose naturalness.py classify
SMOKE_OVERFLOW_MAX7 = 1              # G4-7.9: c2_overflow at more than 1 of the 256 samples is a stop
SMOKE_STEP1_TOL7 = 1e-4              # §8.3 point 4: phase 1's step 1, rho = 1 and KL = 0; old = ref
SMOKE_HOURS_MAX7 = 24.0              # §8.3 point 7: the full run's estimate
SMOKE_FILL7 = 0.95                   # §7.2's packing fill of the full run's batches (the smoke's ~2 are not full)
SMOKE_EVAL_SETS7 = ("books", "btval", "own")   # phase 2's SFT evals (G4-7.10: every 5 iterations in the run)
EVAL_EVERY7 = 5
# G4-7.4h: no SFT evals in the GRPO runs: their eval program does not fit beside the training state (smoke7 s905:
# RESOURCE_EXHAUSTED, 21 MB asked, 5 MB free, as G4-6c's evals); they decided nothing (G4-7.10)
SFT_EVALS7 = False
WORKERS7 = 4                         # the generating workers of mbert-corpus
JUDGES7 = (("metricx", "w0", 0), ("metricx", "w2", 1), ("c2", "w1", None))
# smoke7-loose's inputs of the loose scorers and the names their outputs get in D/loose/
LOOSE7 = {"pairs": "pairs.parquet", "texts": "texts.parquet", "metricx": "metricx.parquet", "c2": "c2.parquet"}
LOOSE_QE7 = "qe_source_hypothesis"   # the column of metricx_qe.py score --pair source:hypothesis
# each check of smoke7: (point of spec §8.3, its bound, what it is)
CHECKS7 = {
    "gen_A_samples": (1, "256 = 32 bronnen × 8 (j = 0..7), de bronnen en het Engels van iteratie N van het plan",
                      "generatie A: aantal samples en bronnen"),
    "gen_A_decoded": (1, "alle 256", "de token-id's decoderen tot de opgeslagen hypothese (decoded_ok)"),
    "gen_A_logp": (1, "alle eindig, even lang als de token-id's", "sampler-logprobs"),
    "gen_A_ended": (1, f">= {SMOKE_MIN_OK7} van 256", "af (stoptoken), geen lus, niet leeg"),
    "gen_A_setup": (1, "geen LORA-regel, MAXTEXT lora null, T = 1, zaad sample_seed(N, k)",
                    "G4-0 zonder adapter, met de zaden van de keten"),
    "scores_A": (2, "read_scores zonder fout; staart na de laatste DONE van de generatie <= 900 s (g4judge's timeout)",
                 "alle scores van A er, binnen de timeout (markers, sha256, uid's, eindig of null volgens G4-7.4a)"),
    "qe_loose": (2, f"max |verschil| <= {SMOKE_QE_TOL7}, null op dezelfde rijen",
                 "QE van de judges gelijk aan een losse metricx_qe.py score op dezelfde rijen"),
    "c2_loose": (2, f"max |verschil| <= {SMOKE_C2_TOL7} (rijen zonder c2_overflow); truncated = c2_overflow",
                 "p_c2 van de judge gelijk aan een losse naturalness.py classify met C2"),
    "c2_overflow": (2, f"<= {SMOKE_OVERFLOW_MAX7} van 256 (G4-7.9)", "c2_overflow in A"),
    "heartbeats": (2, "elke hartslag aanwezig, <= 300 s oud bij het verzamelen, toestand loading/idle/scoring",
                   "hartslagen van de drie judges"),
    "rewards_A": (3, "gelijk (|verschil| <= 1e-9), dezelfde fails en signalen",
                  "reward en voordeel van A opnieuw berekend (reward7, advantages7) tegen iter7's samples.parquet"),
    "rewards_B": (3, "gelijk (|verschil| <= 1e-9), dezelfde fails en signalen", "idem voor B"),
    "check7_A": (3, "ok, op het trainbestand van A (sha256)", "check7: MaxText's SFT-stappen en shift_and_refine"),
    "check7_B": (3, "ok, op het trainbestand van B (sha256)", "idem voor B"),
    "train1_data": (4, "getraind op A's trainbestand (sha256), stappen = de batches van dat bestand",
                    "fase 1: data en stappen"),
    "train1_step1": (4, f"stap 1: max |log pi - log pi_old| <= {SMOKE_STEP1_TOL7} en |KL| <= {SMOKE_STEP1_TOL7}",
                     "fase 1: rho = 1 en KL = 0 op stap 1"),
    "train1_old_ref": (4, f"max |log pi_old - log pi_ref| <= {SMOKE_STEP1_TOL7}", "fase 1: old = ref (LoRA op nul)"),
    "train1_peak": (4, "<= 29,5 GiB per chip", "fase 1: piekgeheugen"),
    "train1_finite": (4, "alle verliezen en stapmaten eindig", "fase 1: verliezen"),
    "train1_files": (4, "adapter.npz en checkpoints/<stappen van fase 1> met commit_success",
                     "fase 1: adapter en Tunix-checkpoint in de bucket"),
    "gen_B_samples": (5, "256 = 32 × 8 van iteratie N+1, alle scores, decoded_ok, logprobs eindig",
                      "generatie B: samples en scores"),
    "gen_B_lora": (5, "elke generatielog: LORA met 410 modules en norm > 0; T = 1, zaad sample_seed(N+1, k)",
                   "generatie B met de ingebakken adapter van fase 1"),
    "train2_data": (5, "getraind op B's trainbestand (sha256), stappen = de batches van dat bestand",
                    "fase 2: data en stappen"),
    "train2_resume": (5, "hervat op stap = de stappen van fase 1, eerste stap die + 1",
                      "fase 2: --resume-step (optimizer-state doorgezet)"),
    "train2_peak": (5, "<= 29,5 GiB per chip", "fase 2: piekgeheugen"),
    "train2_finite": (5, "alle verliezen eindig; geen SFT-evals in de run (G4-7.4h)", "fase 2: verliezen"),
    "train2_files": (5, "adapter.npz en checkpoints/<stappen na fase 2> met commit_success",
                     "fase 2: adapter en Tunix-checkpoint in de bucket"),
    "translate_S_lora": (6, "LORA met 410 modules en norm > 0", "de 50 segmenten met de adapter van fase 2 ingebakken"),
    "translate_ended": (6, ">= 48 van 50 af (geen lus, niet leeg), in beide", "vertaling S en G4-0, greedy, klasse small"),
    "translate_qe": (6, "alle QE eindig, |gemiddelde QE S - G4-0| <= 0,5", "MetricX-QE van de 50 segmenten"),
    "estimate": (7, f"<= {SMOKE_HOURS_MAX7:g} u", "schatting van de volle RL-run (30 iteraties van 512 bronnen × 8)"),
}


def smoke_plan7(plan, base, sources=SMOKE_SOURCES7):
    """The smoke test's plan rows (spec §8.3): the first `sources` sources (by position) of the plan's iteration 0 as
    iteration `base` (A) and those of its iteration 1 as `base` + 1 (B), positions 0.. again; RL-val (iteration -1) as
    it is. Every other column unchanged (the anchor's qe0, l0, ...)."""
    if base < SMOKE_BASE_MIN7:
        raise SystemExit(f"smokeplan7: --base {base}: at least {SMOKE_BASE_MIN7} (the run's iterations are "
                         f"0..{ITERATIONS7 - 1})")
    out = []
    for old, new in ((0, base), (1, base + 1)):
        rows = sorted((r for r in plan if r["iteration"] == old), key=lambda r: r["position"])[:sources]
        if len(rows) < sources:
            raise SystemExit(f"smokeplan7: iteration {old} of the plan has {len(rows)} sources, {sources} wanted")
        out += [{**r, "iteration": new, "position": k} for k, r in enumerate(rows)]
    val = [r for r in plan if r["iteration"] == -1]
    if not val:
        raise SystemExit("smokeplan7: the plan has no RL-val (iteration -1)")
    return out + val


def smokeplan7(args):
    plan = read_rows(args.plan)
    rows = smoke_plan7(plan, args.base, args.sources)
    out = Path(args.out)
    write_parquet([{k: r[k] for k, _ in PLAN_SCHEMA} for r in rows], out / "plan.parquet", schema(PLAN_SCHEMA))
    report = {"plan": str(args.plan), "plan_sha256": file_sha256(args.plan), "iterations": {"A": args.base,
                                                                                             "B": args.base + 1},
              "sources": args.sources, "rlval": sum(r["iteration"] == -1 for r in rows),
              "ids": {str(i): [r["id"] for r in rows if r["iteration"] == i] for i in (args.base, args.base + 1)}}
    (out / "plan.json").write_text(json.dumps(report, indent=1) + "\n")
    print(f"SMOKEPLAN7 {json.dumps({k: v for k, v in report.items() if k != 'ids'})}", flush=True)
    return report


def loose_inputs7(samples):
    """The loose scorers' inputs of the samples in the order the judges read them (worker, shard, row): metricx_qe.py
    score's pairs (id = uid, source, hypothesis; QE mode as the judge, an empty hypothesis unscored) and naturalness.py
    classify's texts (model, set, id = uid, text = the hypothesis)"""
    pairs = [{"id": s["uid"], "source": s["source"], "hypothesis": s["hypothesis"]} for s in samples]
    texts = [{"model": ARM7, "set": "smoke7", "id": s["uid"], "text": s["hypothesis"]} for s in samples]
    return pairs, texts


def smoke7_loose(args):
    """The inputs of the loose scorers of §8.3 point 2 from the samples of A (checked as iter7 reads them)"""
    samples = read_samples(args.samples)
    pairs, texts = loose_inputs7(samples)
    out = Path(args.out)
    write_parquet(pairs, out / LOOSE7["pairs"], schema((("id", "string"), ("source", "string"),
                                                        ("hypothesis", "string"))))
    write_parquet(texts, out / LOOSE7["texts"], schema((("model", "string"), ("set", "string"), ("id", "string"),
                                                        ("text", "string"))))
    report = {"samples": len(samples), "dones": {k: file_sha256(Path(args.samples) / f"w{k}" / DONE)
                                                 for k in sample_dones(args.samples)},
              "files": {LOOSE7[k]: file_sha256(out / LOOSE7[k]) for k in ("pairs", "texts")},
              "outputs": {"metricx": LOOSE7["metricx"], "c2": LOOSE7["c2"]}}
    (out / "loose.json").write_text(json.dumps(report, indent=1) + "\n")
    print(f"SMOKE7_LOOSE {json.dumps(report)}", flush=True)
    return report


# ---------------------------------------------------------------------------------------------------- check7

def _check_chunk7(job):
    import g4arm
    import g4lora

    messages, ens, labels, tokenizer = job
    return g4lora.check_examples(messages, ens, tokenizer, g4arm.MODEL, labels)


def check_packed7(train, samples, scored, tokenizer, ids_of, bos_id, shift, rows_per_batch=ROWS7, processes=1,
                  chunk=256):
    """check7's comparison (see check7). samples: read_samples of the job; scored: iter7's samples.parquet rows; ids_of:
    the model tokenizer's ids of texts (no special tokens added); shift: MaxText's shift_and_refine"""
    import multiprocessing

    import g4lora
    import pyarrow.parquet as pq

    by_uid = {s["uid"]: s for s in samples}
    adv = {r["uid"]: r["adv"] for r in scored}
    signal = {r["uid"] for r in scored if r["signal"]}
    rows = pq.read_table(train).to_pylist()
    if not rows:
        raise SystemExit(f"check7: {train} has no row")
    seq = len(rows[0]["inputs"])
    packed = [u for r in rows for u in r["uids"]]
    if len(set(packed)) != len(packed) or set(packed) != signal or not signal <= set(by_uid):
        raise SystemExit(f"check7: the file's {len(packed)} samples ({len(set(packed))} distinct) are not the "
                         f"{len(signal)} with signal of samples.parquet (or not of the job's samples)")
    if len(rows) % rows_per_batch or [r["batch"] for r in rows] != [k // rows_per_batch for k in range(len(rows))]:
        raise SystemExit(f"check7: {len(rows)} rows are not whole batches of {rows_per_batch} in order")
    # MaxText's SFT steps per source: its user turn and (as the assistant turn) a trained sample's hypothesis, one that
    # is not empty if there is one
    sources = {}
    for u in sorted(packed, key=lambda u: (not by_uid[u]["hypothesis"], u)):
        s = by_uid[u]
        sources.setdefault(s["source_id"], (s["source"], s["hypothesis"]))
    names = sorted(sources)
    jobs = []
    for k in range(0, len(names), chunk):
        part = names[k:k + chunk]
        jobs.append(([[{"role": "user", "content": g4lora.user_turn(sources[n][0])},
                       {"role": "assistant", "content": sources[n][1]}] for n in part],
                     [sources[n][0] for n in part], part, tokenizer))
    if processes > 1:
        with multiprocessing.get_context("spawn").Pool(processes) as pool:
            results = pool.map(_check_chunk7, jobs, chunksize=1)
    else:
        results = [_check_chunk7(j) for j in jobs]
    pads = {pad for _, _, pad in results}
    if len(pads) != 1:
        raise SystemExit(f"check7: MaxText's pad ids {pads}")
    pad = pads.pop()
    prompt_of = dict(zip(names, ids_of([g4lora.prompt_text(sources[n][0]) for n in names])))
    loss_tokens = completion_tokens = 0
    for r, row in enumerate(rows):
        if not row["uids"]:
            if any(any(row[k]) for k in (*g4dpo.KEYS, *TOKEN_KEYS)):
                raise SystemExit(f"check7: row {r} has no sample and is not all zero")
            continue
        parts = [(list(prompt_of[by_uid[u]["source_id"]]), list(by_uid[u]["token_ids"])) for u in row["uids"]]
        want = g4dpo.row_unshifted(parts, seq, pad)
        want = {k: np.asarray(v)[0] for k, v in shift({k: v[None] for k, v in want.items()},
                                                      ignored_ids=[pad, bos_id], axis=1).items()}
        differ = [k for k in g4dpo.KEYS if list(want[k]) != list(row[k])]
        # the advantage and the sampler's log-probability of each completion token at the place whose target it is
        adv_w, slp_w = np.zeros(seq, np.float32), np.zeros(seq, np.float32)
        at = 0
        for (p, c), u in zip(parts, row["uids"]):
            lo = at + len(p) - 1
            adv_w[lo:lo + len(c)] = adv[u]
            slp_w[lo:lo + len(c)] = by_uid[u]["sampler_logp"]
            at += len(p) + len(c)
            completion_tokens += len(c)
        differ += [k for k, w in (("adv", adv_w), ("sampler_logp", slp_w))
                   if not np.array_equal(np.asarray(row[k], np.float32), w)]
        if differ:
            raise SystemExit(f"check7: row {r} (batch {row['batch']}, samples {row['uids'][:2]}...): {differ} differ "
                             f"from MaxText's prompt + the sampled ids, packed and shifted by MaxText's shift_and_refine")
        loss_tokens += int(np.count_nonzero(want["targets_segmentation"]))
    return {"rows": len(rows), "batches": len(rows) // rows_per_batch, "samples": len(packed), "sources": len(names),
            "seq": seq, "pad_id": pad, "bos_id": bos_id, "loss_tokens": loss_tokens,
            "completion_tokens": completion_tokens, "masked_completion_tokens": completion_tokens - loss_tokens}


def check7(args):
    """Every packed row of an iter7 file against MaxText's own SFT steps (as check6, amendment G4-6.3; spec §8.3 point
    3). What is compared:
    (1) per source, g4lora.check_examples: MaxText's apply_chat_template (g4lora.TEMPLATE), tokenization and
        SFTPromptMasking on the source's user turn give as prompt the model's own chat template's prompt, which is the
        tokenizer's ids of g4lora.prompt_text(en): the prompt iter7 packs (the assistant turn there is a sampled
        hypothesis; its text tokenization is not what the row holds, see (2));
    (2) every row = for each of its samples those prompt ids followed by the sampled token ids exactly as translate.py
        wrote them (the stop token included; never tokenized again from the text), packed in the row's order (segments
        1, 2, ..., positions from 0 per sample), then shifted by MaxText's shift_and_refine with MaxText's pad id and the
        tokenizer's bos id ignored: every key of g4dpo.KEYS at every place equals the file's;
    (3) adv and sampler_logp: at the place whose target is a completion token, that sample's advantage (iter7's
        samples.parquet) and the sampler's log-probability of that token (float32), 0 elsewhere, exactly;
    (4) the file holds exactly the samples with signal, each once, in whole batches of --rows rows (an empty row all 0).
    Not compared: the completion's text tokenization (SFT tokenizes the Dutch + "<turn|>\\n"; RL trains on the sampled
    ids, which iter7's decoded_ok checks against the hypothesis). Writes --out (JSON, also on a failure) and CHECK7."""
    out = {"train": str(args.train), "train_sha256": file_sha256(args.train), "samples": str(args.samples), "ok": False}
    try:
        import transformers
        from maxtext.input_pipeline import input_pipeline_utils as ipu

        own = transformers.AutoTokenizer.from_pretrained(args.tokenizer)

        def ids_of(texts):
            return [own(t, add_special_tokens=False)["input_ids"] for t in texts]

        scored = read_rows(Path(args.train).parent / "samples.parquet")
        out.update(check_packed7(args.train, read_samples(args.samples), scored, args.tokenizer, ids_of,
                                 own.bos_token_id, ipu.shift_and_refine, args.rows, args.processes, args.chunk))
        out["ok"] = True
    except SystemExit as stop:
        out["error"] = str(stop.code)
    except Exception as e:  # noqa: BLE001 - written to --out for the chain and printed with its traceback, then exit 1
        import traceback

        traceback.print_exc()
        out["error"] = f"{type(e).__name__}: {e}"
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    print(f"CHECK7 {json.dumps(out)}", flush=True)
    if not out["ok"]:
        raise SystemExit(1)
    return out


# ---------------------------------------------------------------------------------------------------- smoke7

def log_json(text, tag):
    """The JSON of every line 'TAG {...}' of a log, in order"""
    return [json.loads(line[len(tag) + 1:]) for line in text.splitlines() if line.startswith(tag + " {")]


def finite(x):
    return x is not None and isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def generation7(samples, plan, iteration, counter):
    """What a generation of the smoke is (§8.3 points 1 and 5): its samples against the plan's sources of the
    iteration, the decode check, the sampler's log-probabilities, the ended ones, the different samples per group"""
    sources = {r["id"]: r for r in plan if r["iteration"] == iteration}
    groups = groups7(samples)
    same = set(groups) == set(sources) and all(s["source"] == sources[k]["en"] for k, g in groups.items() for s in g
                                               if k in sources)
    bad_decode = [s["uid"] for s in samples if not decoded_ok(counter, s)]
    bad_logp = [s["uid"] for s in samples if len(s["sampler_logp"]) != len(s["token_ids"])
                or not all(finite(x) for x in s["sampler_logp"])]
    ok = [s for s in samples if s["ended"] and not s["decode_loop"] and not s["empty"]]
    distinct = [len({s["hypothesis"] for s in g}) for g in groups.values()]
    return {"samples": len(samples), "sources": len(groups), "plan_sources": len(sources), "same_sources": same,
            "decode_bad": len(bad_decode), "decode_bad_uids": bad_decode[:5], "logp_bad": len(bad_logp),
            "logp_bad_uids": bad_logp[:5], "ok": len(ok), "ended": sum(bool(s["ended"]) for s in samples),
            "decode_loop": sum(bool(s["decode_loop"]) for s in samples), "empty": sum(bool(s["empty"]) for s in samples),
            "english_left": sum(bool(s["english_left"]) for s in samples),
            "logp_max": max((max(s["sampler_logp"]) for s in samples if s["sampler_logp"]), default=None),
            "distinct_per_group": {"min": min(distinct), "median": median(distinct), "max": max(distinct),
                                   "groups_all_same": sum(d == 1 for d in distinct)}}


def gen_logs7(logs, phase, workers):
    """{k: the LORA, MAXTEXT and DONE lines of D/logs/gen-<phase>-w<k>.log} (translate.py's)"""
    out = {}
    for k in workers:
        path = Path(logs) / f"gen-{phase}-w{k}.log"
        if not path.is_file():
            raise SystemExit(f"no {path}")
        text = path.read_text(errors="replace")
        done = log_json(text, "DONE")
        out[k] = {"lora": log_json(text, "LORA"), "maxtext": log_json(text, "MAXTEXT"),
                  "done": done[-1] if done else None}
    return out


def setup_ok7(logs, iteration, lora):
    """translate.py's lines of every worker: T = 1 and seed sample_seed(iteration, k) on its DONE line; with lora a LORA
    line of SMOKE_MODULES modules and a relative update norm > 0, without one no LORA line and MAXTEXT's lora null"""
    import g4arm

    problems = []
    for k, x in logs.items():
        done = x["done"] or {}
        if done.get("temperature") != 1.0 or done.get("seed") != sample_seed(iteration, int(k)):
            problems.append(f"w{k}: DONE has T {done.get('temperature')} and seed {done.get('seed')}, not 1.0 and "
                            f"{sample_seed(iteration, int(k))}")
        if lora:
            last = x["lora"][-1] if x["lora"] else {}
            if last.get("modules") != g4arm.SMOKE_MODULES or not (last.get("relative_update_norm") or 0) > 0:
                problems.append(f"w{k}: LORA {last or 'missing'}")
        elif x["lora"] or any(m.get("lora") for m in x["maxtext"]) or not x["maxtext"]:
            problems.append(f"w{k}: an adapter (LORA {x['lora'][:1]}, MAXTEXT lora "
                            f"{[m.get('lora') for m in x['maxtext']]})")
    return problems


def loose_compare7(samples, scores, loose):
    """§8.3 point 2: the judges' scores of A against the loose scorers' outputs of the same rows (D/loose/): the loose
    inputs are the samples' texts; QE (both null for an empty hypothesis) within SMOKE_QE_TOL7; p_c2 within SMOKE_C2_TOL7
    on the rows without c2_overflow, and classify's truncated flag is c2_overflow (C2's tokenizer puts only <s> in
    front, G4-7.4b: more than 1,022 tokens is 1,024 with it)"""
    loose = Path(loose)
    pairs = {r["id"]: r for r in read_rows(loose / LOOSE7["pairs"])}
    texts = {r["id"]: r for r in read_rows(loose / LOOSE7["texts"])}
    mx = {r["id"]: r for r in read_rows(loose / LOOSE7["metricx"])}
    c2 = {r["id"]: r for r in read_rows(loose / LOOSE7["c2"])}
    uids = {s["uid"] for s in samples}
    inputs_bad = [s["uid"] for s in samples if s["uid"] not in pairs or s["uid"] not in texts
                  or (pairs[s["uid"]]["source"], pairs[s["uid"]]["hypothesis"]) != (s["source"], s["hypothesis"])
                  or texts[s["uid"]]["text"] != s["hypothesis"]]
    qe = {"rows": 0, "null_both": 0, "max_abs_diff": 0.0, "mismatch": []}
    p = {"rows": 0, "overflow": 0, "max_abs_diff": 0.0, "mismatch": []}
    for u in sorted(uids):
        a, b = scores["metricx"][u]["qe"], (mx.get(u) or {}).get(LOOSE_QE7)
        if a is None and b is None and u in mx:
            qe["null_both"] += 1
        elif a is None or b is None:
            qe["mismatch"].append(u)
        else:
            qe["rows"] += 1
            qe["max_abs_diff"] = max(qe["max_abs_diff"], abs(float(a) - float(b)))
        j, c = scores["c2"][u], c2.get(u)
        if c is None or bool(c["truncated"]) != bool(j["c2_overflow"]):
            p["mismatch"].append(u)
        elif j["c2_overflow"]:
            p["overflow"] += 1
        else:
            p["rows"] += 1
            p["max_abs_diff"] = max(p["max_abs_diff"], abs(float(j["p_c2"]) - float(c["p_original"])))
    for x in (qe, p):
        x["mismatch_count"] = len(x["mismatch"])
        x["mismatch"] = x["mismatch"][:5]
    extra = {"metricx": len(set(mx) - uids), "c2": len(set(c2) - uids)}
    return {"inputs_bad": len(inputs_bad), "inputs_bad_uids": inputs_bad[:5], "qe": qe, "p_c2": p, "extra_rows": extra}


class _HeartbeatDir:
    """g4judge's store interface over D/heartbeats/ (judges/<role>-w<m>.json -> heartbeats/<role>-w<m>.json)"""

    def __init__(self, root):
        self.root = Path(root)

    def _path(self, name):
        return self.root / name.split("/", 1)[1]

    def exists(self, name):
        return self._path(name).is_file()

    def read(self, name):
        return self._path(name).read_bytes()


def rewards_compare7(samples, scores, plan, iteration, train_dir):
    """§8.3 point 3: the rewards and advantages computed again (reward7, advantages7 through iteration_samples) from the
    samples, the scores and the plan, against iter7's samples.parquet and report.json"""
    rows, info = iteration_samples(samples, scores, plan, iteration)
    got = {r["uid"]: r for r in read_rows(Path(train_dir) / "samples.parquet")}
    report = json.loads((Path(train_dir) / "report.json").read_text())
    differ, d_reward, d_adv = [], 0.0, 0.0
    for r in rows:
        g = got.get(r["uid"])
        if g is None or g["signal"] != r["signal"] or g["fail"] != ",".join(r["fail"]):
            differ.append(r["uid"])
            continue
        d_reward = max(d_reward, abs(g["reward"] - r["reward"]))
        d_adv = max(d_adv, abs(g["adv"] - r["adv"]))
    return {"samples": len(rows), "file_rows": len(got), "report_iteration": report.get("iteration"),
            "groups_without_signal": info["groups_without_signal"],
            "report_groups_without_signal": report.get("groups_without_signal"), "differ": len(differ),
            "differ_uids": differ[:5], "reward_max_abs_diff": d_reward, "adv_max_abs_diff": d_adv,
            "fail_share": iteration_report(rows)["fail_share"]}


def judge_seconds7(scores_root):
    """{judge: (scoring seconds, sample rows read)} of a job's score shards (the metadata g4judge.py writes)"""
    import pyarrow.parquet as pq

    out = {}
    for path in sorted(Path(scores_root).glob("*/w*/shard-*.parquet")):
        meta = json.loads((pq.read_schema(path).metadata or {})[b"g4judge"])
        key = f"{path.parent.parent.name}-{path.parent.name}"
        s, n = out.get(key, (0.0, 0))
        out[key] = (s + float(meta["seconds"]), n + int(meta["samples_rows"]))
    return out


def evals_between(summary):
    """phase 2's evals after its first step and before its last (they would sit inside a step's duration)"""
    loss = summary.get("loss") or []
    if len(loss) < 2:
        return []
    return [e for e in summary.get("eval") or [] if loss[0][0] <= e["step"] < loss[-1][0]]


def smoke_estimate7(gen, judges, iter7_runs, trains, tokens_per_sample, val_s, iterations=ITERATIONS7,
                    sources=SOURCES7, g=G7, workers=WORKERS7):
    """The full RL run's hours from the smoke's measured phases (spec §7.2 as the model; every measure the larger of A
    and B, so the estimate leans long):
      S = sources x g samples per iteration, S / workers per generating worker;
      generation = startup + per sample x S / workers, startup = the phase's wall time - its slowest worker's decode
        seconds (translate.py's DONE), per sample = the slowest worker's decode seconds per sample;
      judges = the tail after the generation's last DONE + max(0, a judge's scoring seconds per iteration sample x S -
        the generation's decode time x S / workers) (the judges stream: only what they cannot keep up with adds);
      iter7 = its wall time per sample x S (linear: the fixed part scaled too);
      steps = ceil(S x the trained tokens per sample (iter7's report) / (ROWS7 x SEQ7 x SMOKE_FILL7));
      training of iteration 0 = phase 1's wall time (startup, the passes, compilation, its steps, saving, the chain's
        copies) - its evals + (steps - its steps) x (step + passes per batch); iterations 1..29 the same from phase 2
        (its first step with the compile cache: first_step_s); step = the median step time (phase 1's, and phase 2's
        when no eval sits between its steps); passes per batch = min(the phase's median passes batch, step) (two
        forwards cost less than a forward and backward; a median of two batches holds a compilation);
      evals = (iterations // EVAL_EVERY7) x the longest eval round of phase 2 (books, btval, own);
      RL-val = its wall time after every iteration (on the critical path: the stop rules wait for it).
    total = iterations x (generation + judges + iter7 + RL-val) + training of iteration 0 + (iterations - 1) x the
    next iterations' training + evals."""
    n = sources * g
    per_worker = n / workers
    startup = max(x["wall_s"] - x["decode_s"] for x in gen.values())
    per_sample = max(x["per_sample_s"] for x in gen.values())
    decode_full = per_sample * per_worker
    gen_s = startup + decode_full
    tail = max(x["tail_s"] for x in judges.values())
    judge_rate = max(x["s_per_sample"] for x in judges.values())
    judge_s = tail + max(0.0, judge_rate * n - decode_full)
    iter7_s = max(x["wall_s"] / x["samples"] for x in iter7_runs.values()) * n
    steps = math.ceil(n * tokens_per_sample / (ROWS7 * SEQ7 * SMOKE_FILL7))
    s1, s2 = trains["1"]["summary"], trains["2"]["summary"]
    step_times = [(s1.get("throughput") or {}).get("step_s_median")]
    if not evals_between(s2):
        step_times.append((s2.get("throughput") or {}).get("step_s_median"))
    step_times = [x for x in step_times if finite(x)]
    if not step_times:
        raise SystemExit("estimate: no step time (phase 1 and 2 have one step each, or phase 2 evaluates between)")
    step_s = max(step_times)
    train = {}
    for k, x in trains.items():
        s = x["summary"]
        passes = (s.get("grpo") or {}).get("passes") or {}
        pbs = min(passes.get("batch_s_median") or step_s, step_s)
        evals = sum(e.get("seconds") or 0.0 for e in s.get("eval") or [])
        train[k] = {"wall_s": x["wall_s"], "evals_s": evals, "own_steps": s["own_steps"], "passes_batch_s": pbs,
                    "first_step_s": (s.get("times") or {}).get("first_step_s"),
                    "iteration_s": x["wall_s"] - evals + (steps - s["own_steps"]) * (step_s + pbs)}
    rounds = {}
    for e in s2.get("eval") or []:
        rounds[e["step"]] = rounds.get(e["step"], 0.0) + (e.get("seconds") or 0.0)
    eval_round = max(rounds.values(), default=0.0)
    total = (iterations * (gen_s + judge_s + iter7_s + val_s) + train["1"]["iteration_s"]
             + (iterations - 1) * train["2"]["iteration_s"] + (iterations // EVAL_EVERY7) * eval_round)
    per_iteration = {"generation_s": gen_s, "judges_s": judge_s, "iter7_s": iter7_s, "val_s": val_s,
                     "training_s": train["2"]["iteration_s"]}
    return {"iterations": iterations, "samples": n, "steps": steps, "tokens_per_sample": tokens_per_sample,
            "fill": SMOKE_FILL7, "generation": {"startup_s": startup, "per_sample_s": per_sample,
                                                "decode_s": decode_full},
            "judges": {"tail_s": tail, "s_per_sample": judge_rate}, "step_s": step_s, "train": train,
            "eval_round_s": eval_round, "eval_rounds": iterations // EVAL_EVERY7,
            "per_iteration": {k: round(v, 1) for k, v in per_iteration.items()},
            "minutes_per_iteration": round(sum(per_iteration.values()) / 60, 1),
            "total_s": round(total, 1), "hours": round(total / 3600, 2)}


def smoke_verdict7(d, counter):
    """Every check of spec §8.3 points 1-7 (and G4-7.9) on the smoke's directory d, and what is reported beside them.
    A check whose files are missing or wrong fails with the error as its value; the others still run."""
    import g4arm
    import g4judge

    d = Path(d)
    checks, out = {}, {}

    def put(name, holds, value):
        point, bound, what = CHECKS7[name]
        checks[name] = {"holds": bool(holds), "value": value, "bound": bound, "what": what, "point": point}

    def guard(names, fn):
        try:
            fn()
        except (SystemExit, Exception) as e:  # a missing or broken input fails its checks, not the whole verdict
            text = str(e.code if isinstance(e, SystemExit) else f"{type(e).__name__}: {e}")[:600]
            for name in names:
                if name not in checks:
                    put(name, False, {"error": text})

    plan = read_rows(d / "plan.parquet")
    smoke_iters = sorted({r["iteration"] for r in plan if r["iteration"] >= 0})
    if len(smoke_iters) != 2 or smoke_iters[1] != smoke_iters[0] + 1 or smoke_iters[0] < SMOKE_BASE_MIN7:
        raise SystemExit(f"{d}/plan.parquet: iterations {smoke_iters}, not N, N+1 with N >= {SMOKE_BASE_MIN7} "
                         "(smokeplan7's plan)")
    it = {"A": smoke_iters[0], "B": smoke_iters[1]}
    out["iteration"] = it
    data = {}

    def job(phase):
        if phase not in data:
            root = d / f"iter-{phase}"
            samples = read_samples(root / "samples")
            scores = read_scores(root / "scores", {s["uid"] for s in samples}, sample_dones(root / "samples"))
            data[phase] = (samples, scores)
        return data[phase]

    # (1) the generation of A (G4-0)
    def point1():
        samples = read_samples(d / "iter-A" / "samples")
        x = generation7(samples, plan, it["A"], counter)
        out["generation_A"] = x
        put("gen_A_samples", x["samples"] == SMOKE_SAMPLES7 and x["sources"] == SMOKE_SOURCES7
            and x["plan_sources"] == SMOKE_SOURCES7 and x["same_sources"],
            {k: x[k] for k in ("samples", "sources", "plan_sources", "same_sources")})
        put("gen_A_decoded", x["decode_bad"] == 0, {"bad": x["decode_bad"], "uids": x["decode_bad_uids"]})
        put("gen_A_logp", x["logp_bad"] == 0, {"bad": x["logp_bad"], "uids": x["logp_bad_uids"], "max": x["logp_max"]})
        put("gen_A_ended", x["ok"] >= SMOKE_MIN_OK7,
            {k: x[k] for k in ("ok", "ended", "decode_loop", "empty", "english_left")})
        logs = gen_logs7(d / "logs", "A", sorted(int(k) for k in sample_dones(d / "iter-A" / "samples")))
        problems = setup_ok7(logs, it["A"], lora=False)
        put("gen_A_setup", not problems, problems or {"workers": sorted(logs)})

    guard(("gen_A_samples", "gen_A_decoded", "gen_A_logp", "gen_A_ended", "gen_A_setup"), point1)

    # (2) the judges
    def scores_a():
        samples, scores = job("A")
        tail = json.loads((d / "timings.json").read_text())["judge_tail_s"]["A"]
        put("scores_A", finite(tail) and tail <= g4judge.SCORE_TIMEOUT,
            {**{role: len(v) for role, v in scores.items()}, "tail_s": tail})
        n = sum(bool(r["c2_overflow"]) for r in scores["c2"].values())
        put("c2_overflow", n <= SMOKE_OVERFLOW_MAX7, {"overflow": n, "of": len(scores["c2"]),
                                                      "c2_tokens_max": max(r["c2_tokens"] for r in scores["c2"].values())})

    guard(("scores_A", "c2_overflow"), scores_a)

    def loose():
        samples, scores = job("A")
        x = loose_compare7(samples, scores, d / "loose")
        out["loose"] = x
        put("qe_loose", not x["inputs_bad"] and not x["qe"]["mismatch_count"] and not x["extra_rows"]["metricx"]
            and x["qe"]["max_abs_diff"] <= SMOKE_QE_TOL7, {"inputs_bad": x["inputs_bad"], **x["qe"]})
        put("c2_loose", not x["inputs_bad"] and not x["p_c2"]["mismatch_count"] and not x["extra_rows"]["c2"]
            and x["p_c2"]["max_abs_diff"] <= SMOKE_C2_TOL7, {"inputs_bad": x["inputs_bad"], **x["p_c2"]})

    guard(("qe_loose", "c2_loose"), loose)

    def heartbeats():
        now = json.loads((d / "now.json").read_text())["epoch"]
        problems = g4judge.check_judges(_HeartbeatDir(d / "heartbeats"), JUDGES7, now=now)
        ages = {}
        for role, worker, _ in JUDGES7:
            path = d / "heartbeats" / f"{role}-{worker}.json"
            if path.is_file():
                hb = json.loads(path.read_text())
                ages[f"{role}-{worker}"] = {"age_s": round(now - hb["epoch"], 1), "state": hb["state"]}
        put("heartbeats", not problems, {"problems": problems, "judges": ages})

    guard(("heartbeats",), heartbeats)

    # (3) the iterations' files
    for phase in ("A", "B"):
        def rewards(phase=phase):
            samples, scores = job(phase)
            x = rewards_compare7(samples, scores, plan, it[phase], d / f"iter-{phase}" / "train")
            out[f"rewards_{phase}"] = x
            put(f"rewards_{phase}", x["differ"] == 0 and x["samples"] == x["file_rows"] and x["report_iteration"]
                == it[phase] and x["report_groups_without_signal"] == x["groups_without_signal"]
                and x["reward_max_abs_diff"] <= 1e-9 and x["adv_max_abs_diff"] <= 1e-9, x)

        def check(phase=phase):
            x = json.loads((d / f"check7-{phase}.json").read_text())
            sha = file_sha256(d / f"iter-{phase}" / "train" / "train.parquet")
            put(f"check7_{phase}", x.get("ok") is True and x.get("train_sha256") == sha,
                {k: x.get(k) for k in ("ok", "rows", "samples", "sources", "loss_tokens", "error")}
                | {"same_file": x.get("train_sha256") == sha})

        guard((f"rewards_{phase}",), rewards)
        guard((f"check7_{phase}",), check)

    # (4), (5) the training phases
    summaries = {}

    def phase_checks(k, phase):
        s = json.loads((d / f"train-{k}" / "summary.json").read_text())
        summaries[k] = s
        report = json.loads((d / f"iter-{phase}" / "train" / "report.json").read_text())
        grpo = s.get("grpo") or {}
        sha = file_sha256(d / f"iter-{phase}" / "train" / "train.parquet")
        put(f"train{k}_data", (grpo.get("identity") or {}).get("grpo_train_sha256") == sha
            and s.get("own_steps") == s.get("steps_done") == report["train"]["batches"],
            {"same_file": (grpo.get("identity") or {}).get("grpo_train_sha256") == sha, "own_steps": s.get("own_steps"),
             "steps_done": s.get("steps_done"), "batches": report["train"]["batches"]})
        put(f"train{k}_peak", finite(s.get("peak_gib_max")) and s["peak_gib_max"] <= g4arm.MEM_MAX_GIB,
            s.get("peak_gib_max"))
        losses = [v for _, v in s.get("loss") or []]
        measures = [v for st in grpo.get("steps") or [] for key, v in st.items() if key != "step"]
        holds = bool(losses) and all(finite(v) for v in losses + measures) and len(grpo.get("steps") or []) == len(losses)
        value = {"loss": s.get("loss"), "grpo_steps": len(grpo.get("steps") or [])}
        if k == "2":
            evals = s.get("eval") or []
            names = {e["set"] for e in evals}
            if SFT_EVALS7:
                holds = holds and set(SMOKE_EVAL_SETS7) <= names and all(finite(e.get("loss")) for e in evals)
            else:
                holds = holds and not evals
            value["eval"] = [[e["set"], e["step"], e.get("loss")] for e in evals]
        put(f"train{k}_finite", holds, value)
        files = json.loads((d / f"train-{k}" / "files.json").read_text())
        ck = str(files.get("checkpoint") or "")
        put(f"train{k}_files", files.get("adapter.npz") is True and files.get("commit_success") is True
            and ck.rstrip("/").endswith(f"/checkpoints/{s.get('steps')}"), files)
        passes = grpo.get("passes") or {}
        out[f"train_{k}"] = {"steps": s.get("steps"), "own_steps": s.get("own_steps"), "resume": s.get("resume"),
                             "times": s.get("times"), "throughput": s.get("throughput"),
                             "peak_gib_max": s.get("peak_gib_max"),
                             "passes": {key: passes.get(key) for key in (
                                 "tokens", "k1_old_ref", "old_ref_abs_max", "old_is_ref", "sampler_gap",
                                 "kl_sampler_old", "seconds", "batch_s_median")},
                             "steps_grpo": grpo.get("steps")}
        return s

    def phase1():
        s = phase_checks("1", "A")
        steps = (s.get("grpo") or {}).get("steps") or []
        first = steps[0] if steps else {}
        put("train1_step1", first.get("step") == 1 and finite(first.get("log_ratio_abs_max"))
            and first["log_ratio_abs_max"] <= SMOKE_STEP1_TOL7 and finite(first.get("kl"))
            and abs(first["kl"]) <= SMOKE_STEP1_TOL7,
            {k: first.get(k) for k in ("step", "log_ratio_abs_max", "kl", "loss", "adv_mean", "clip_frac")})
        passes = (s.get("grpo") or {}).get("passes") or {}
        put("train1_old_ref", finite(passes.get("old_ref_abs_max")) and passes["old_ref_abs_max"] <= SMOKE_STEP1_TOL7,
            {k: passes.get(k) for k in ("old_ref_abs_max", "old_is_ref", "k1_old_ref")})

    guard(("train1_data", "train1_peak", "train1_finite", "train1_files", "train1_step1", "train1_old_ref"), phase1)

    def gen_b():
        samples, scores = job("B")
        x = generation7(samples, plan, it["B"], counter)
        out["generation_B"] = x
        out["c2_overflow_B"] = sum(bool(r["c2_overflow"]) for r in scores["c2"].values())
        put("gen_B_samples", x["samples"] == SMOKE_SAMPLES7 and x["sources"] == SMOKE_SOURCES7 and x["same_sources"]
            and x["plan_sources"] == SMOKE_SOURCES7 and x["decode_bad"] == 0 and x["logp_bad"] == 0,
            {k: x[k] for k in ("samples", "sources", "same_sources", "decode_bad", "logp_bad", "ok", "ended")})
        logs = gen_logs7(d / "logs", "B", sorted(int(k) for k in sample_dones(d / "iter-B" / "samples")))
        problems = setup_ok7(logs, it["B"], lora=True)
        put("gen_B_lora", not problems, problems or {f"w{k}": x["lora"][-1] for k, x in logs.items()})

    guard(("gen_B_samples", "gen_B_lora"), gen_b)

    def phase2():
        s = phase_checks("2", "B")
        s1 = summaries.get("1") or json.loads((d / "train-1" / "summary.json").read_text())
        steps1 = s1.get("steps")
        loss = s.get("loss") or []
        grpo_steps = (s.get("grpo") or {}).get("steps") or []
        put("train2_resume", (s.get("resume") or {}).get("step") == steps1 and bool(loss) and loss[0][0] == steps1 + 1
            and bool(grpo_steps) and grpo_steps[0]["step"] == steps1 + 1
            and s.get("steps") == steps1 + (s.get("own_steps") or 0),
            {"steps_1": steps1, "resume": s.get("resume"), "first_step": loss[0][0] if loss else None,
             "steps": s.get("steps")})

    guard(("train2_data", "train2_peak", "train2_finite", "train2_files", "train2_resume"), phase2)
    gap = (((summaries.get("2") or {}).get("grpo") or {}).get("passes") or {}).get("sampler_gap")
    out["sampler_gap"] = gap
    out["sampler_gap_1"] = (((summaries.get("1") or {}).get("grpo") or {}).get("passes") or {}).get("sampler_gap")
    out["tis"] = bool(finite(gap) and gap > TIS_GAP)
    out["first_step_s"] = {k: (s.get("times") or {}).get("first_step_s") for k, s in summaries.items()}
    out["compile_s_est"] = {k: (s.get("times") or {}).get("compile_s_est") for k, s in summaries.items()}

    # (6) the 50 WMT24++ segments
    def translation():
        lora = g4arm.lora_line((d / "logs" / "translate-S.log").read_text(errors="replace"))
        put("translate_S_lora", bool(lora) and lora.get("modules") == g4arm.SMOKE_MODULES
            and (lora.get("relative_update_norm") or 0) > 0, lora)
        rows = {}
        for name in ("S", "G4-0"):
            p = {r["id"]: r for r in read_rows(d / "translate" / name / "pairs-w0.parquet")}
            m = {r["id"]: r for r in read_rows(d / "translate" / name / "metricx-w0.parquet")}
            rows[name] = [{**r, "qe": m.get(i, {}).get(LOOSE_QE7)} for i, r in p.items()]
        ok = {k: sum(bool(r["ended"]) and not r["decode_loop"] and not r["empty"] for r in v) for k, v in rows.items()}
        n = {k: len(v) for k, v in rows.items()}
        put("translate_ended", all(n[k] == g4arm.SMOKE_N and ok[k] >= g4arm.SMOKE_MIN_ENDED for k in rows),
            {"segments": n, "ok": ok})
        qe = {k: [r["qe"] for r in v] for k, v in rows.items()}
        all_finite = all(len(v) == g4arm.SMOKE_N and all(finite(q) for q in v) for v in qe.values())
        means = {k: mean(v) if all(finite(q) for q in v) else None for k, v in qe.items()}
        delta = means["S"] - means["G4-0"] if all_finite else None
        put("translate_qe", all_finite and abs(delta) <= g4arm.SMOKE_QE_MAX_DELTA,
            {"mean": means, "delta": delta})
        same = {r["id"]: r["hypothesis"] for r in rows["G4-0"]}
        out["identical_translations"] = sum(same.get(r["id"]) == r["hypothesis"] for r in rows["S"])

    guard(("translate_S_lora", "translate_ended", "translate_qe"), translation)

    # (7) the estimate
    def estimate():
        t = json.loads((d / "timings.json").read_text())
        gen, judges, runs, tokens = {}, {}, {}, []
        for phase in ("A", "B"):
            dones = sample_dones(d / f"iter-{phase}" / "samples")
            logs = gen_logs7(d / "logs", phase, sorted(int(k) for k in dones))
            per = [(x["done"]["seconds"], x["done"]["segments"]) for x in logs.values()]
            samples = sum(v["rows"] for v in dones.values())
            gen[phase] = {"wall_s": float(t["gen_wall_s"][phase]), "decode_s": max(s for s, _ in per),
                          "per_sample_s": max(s / n for s, n in per)}
            js = judge_seconds7(d / f"iter-{phase}" / "scores")
            judges[phase] = {"tail_s": float(t["judge_tail_s"][phase]),
                             "s_per_sample": max(s for s, _ in js.values()) / samples, "judges": js}
            runs[phase] = {"wall_s": float(t["iter7_wall_s"][phase]), "samples": samples}
            report = json.loads((d / f"iter-{phase}" / "train" / "report.json").read_text())
            tokens.append(report["train"]["tokens"] / report["train"]["samples"])
        trains = {k: {"summary": json.loads((d / f"train-{k}" / "summary.json").read_text()),
                      "wall_s": float(t["train_wall_s"][k])} for k in ("1", "2")}
        x = smoke_estimate7(gen, judges, runs, trains, max(tokens), float(t["val_wall_s"]))
        x["measured"] = {"generation": gen, "judges": {p: {k: v for k, v in j.items() if k != "judges"}
                                                       for p, j in judges.items()},
                         "judge_seconds": {p: j["judges"] for p, j in judges.items()}, "iter7": runs,
                         "timings": t}
        out["estimate"] = x
        put("estimate", finite(x["hours"]) and x["hours"] <= SMOKE_HOURS_MAX7, x["hours"])

    guard(("estimate",), estimate)
    out.setdefault("estimate", {"hours": None})

    # reported, deciding nothing: stop7 on RL-val after phase 1
    stop = d / "val" / "stop.json"
    out["val"] = json.loads(stop.read_text()) if stop.is_file() else None
    ordered_checks = {k: checks[k] for k in CHECKS7 if k in checks}
    failed = [k for k, c in ordered_checks.items() if not c["holds"]]
    return {"checks": ordered_checks, "passed": not failed, "failed": failed, **out}


def cell7(text):
    """A markdown table cell: its | escaped"""
    return str(text).replace("|", "\\|")


def f3_7(x):
    return "–" if x is None else f"{x:.3f}" if isinstance(x, float) else str(x)


def smoke7_md(r):
    """The smoke test's report in Dutch"""
    e = r.get("estimate") or {}
    L = [f"# Smoke-test G4-7 (spec §8.3, amendement G4-7.9): iteraties A = {r['iteration']['A']}, "
         f"B = {r['iteration']['B']}", "",
         f"**{'Geslaagd' if r['passed'] else 'Niet geslaagd: ' + ', '.join(r['failed'])}.** Schatting van de volle "
         f"RL-run: {f3_7(e.get('hours'))} u (grens {SMOKE_HOURS_MAX7:g} u).", "",
         "| punt | controle | wat | waarde | grens | gehaald |", "|---:|---|---|---|---|---|"]
    for name, c in r["checks"].items():
        value = json.dumps(c["value"], default=str, ensure_ascii=False)[:220]
        L.append(f"| {c['point']} | {name} | {cell7(c['what'])} | `{cell7(value)}` | {cell7(c['bound'])} | "
                 f"{'ja' if c['holds'] else 'nee'} |")
    gen = r.get("generation_A") or {}
    L += ["", "## Gerapporteerd (beslist niets)", "",
          f"- Verschillende hypothesen per groep van 8 in A: {json.dumps(gen.get('distinct_per_group'))}; in B: "
          f"{json.dumps((r.get('generation_B') or {}).get('distinct_per_group'))}.",
          f"- Verschil sampler/leerder (gemiddelde |log π_old − log π_sampler| per token): fase 2 "
          f"{f3_7(r.get('sampler_gap'))}, fase 1 {f3_7(r.get('sampler_gap_1'))}; grens {TIS_GAP} (spec §4.4): "
          + ("**boven de grens: TIS vanaf de eerste RL-iteratie**." if r.get("tis") else "eronder: geen TIS."),
          f"- `first_step_s` (compileren en de eerste stap; fase 2 met de compileercache): "
          f"{json.dumps(r.get('first_step_s'))}; `compile_s_est` {json.dumps(r.get('compile_s_est'))}.",
          f"- `c2_overflow` in B: {r.get('c2_overflow_B')}; gelijke vertalingen S en G4-0 (50 segmenten): "
          f"{r.get('identical_translations')}.",
          f"- stop7 op RL-val na fase 1: {json.dumps({k: (r.get('val') or {}).get(k) for k in ('stop', 'tis_next')})}."]
    if e.get("per_iteration"):
        L += ["", "## Schatting (spec §7.2 als model; elke maat de grootste van A en B)", "",
              f"S = {e['samples']} samples per iteratie, {e['iterations']} iteraties, {e['steps']} stappen per iteratie "
              f"(⌈S × {f3_7(e['tokens_per_sample'])} getrainde tokens per sample / (16 × 4.096 × {e['fill']})⌉), "
              f"stap {f3_7(e['step_s'])} s.", "",
              "- generatie = opstart (wandtijd − decodeertijd van de traagste worker) + decodeertijd per sample × S / 4: "
              f"{f3_7(e['generation']['startup_s'])} + {f3_7(e['generation']['per_sample_s'])} × {e['samples'] // WORKERS7} s;",
              "- judges = staart na de laatste DONE + max(0, scoortijd per sample × S − decodeertijd): "
              f"{f3_7(e['judges']['tail_s'])} s staart, {f3_7(e['judges']['s_per_sample'])} s per sample;",
              "- iter7 = wandtijd per sample × S (lineair, ook het vaste deel);",
              "- trainen (iteratie 0 uit fase 1, de andere uit fase 2 met de cache) = wandtijd − evals + (stappen − "
              "eigen stappen) × (stap + passes per batch), passes per batch = min(mediaan, stap): "
              f"{json.dumps({k: round(v['iteration_s'], 1) for k, v in e['train'].items()})} s;",
              f"- evals: {e['eval_rounds']} × {f3_7(e['eval_round_s'])} s; RL-val na elke iteratie.", "",
              f"Per iteratie (s): {json.dumps(e['per_iteration'])}, samen {e['minutes_per_iteration']} min; totaal "
              f"{e['total_s']} s = **{e['hours']} u**."]
    return "\n".join(L) + "\n"


def smoke7(args, counter=None):
    import g4arm

    counter = counter or g4arm.TokenCounter(args.tokenizer)
    result = smoke_verdict7(args.dir, counter)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "smoke7.json").write_text(json.dumps(result, indent=1, default=str) + "\n")
    (out / "smoke7.md").write_text(smoke7_md(result), encoding="utf-8")
    short = {"passed": result["passed"], "failed": result["failed"], "hours": result["estimate"].get("hours"),
             "sampler_gap": result.get("sampler_gap"), "tis": result.get("tis"),
             "first_step_s": result.get("first_step_s")}
    print(f"SMOKE7 {json.dumps(short, default=str)}", flush=True)
    print("SMOKE7_OK" if result["passed"] else f"SMOKE7_FAIL {' '.join(result['failed'])}", flush=True)
    if not result["passed"]:
        raise SystemExit(1)
    return result


# ---------------------------------------------------------------------------------------------------- main

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("cands7")
    p.add_argument("--flan-plan", required=True, help="FLAN-nl's plan (mbert/flan-nl/plan2: examples.parquet, plan.json, "
                                                       "segments-w*.parquet)")
    p.add_argument("--flan-kept", required=True, help="the examples assemble kept for the publication "
                                                       "(mbert/flan-nl/nl-g4b: w<host>.parquet)")
    p.add_argument("--flan-build-report", required=True,
                   help="the published build's build_report.json (mbert/flan-nl/nl-g4b/checks/build/): rows left out")
    p.add_argument("--flan-eval", required=True, help="FLAN's evaluation rows (mainset/flan-eval/eval-rows.parquet)")
    p.add_argument("--novels", required=True, help="the -b1- inventory's b1-selected.parquet ($S/data/books-b1/)")
    p.add_argument("--novels-report", required=True, help="G4-4's data report ($G/data/report-G4-4.json)")
    p.add_argument("--novels-eval", required=True, help="the eval set novels' manifest ($G/data/heldout4/manifest.json)")
    p.add_argument("--bt", required=True, help="fetch --arm G4-3's train directory ($G/input/bt3: revisions.json, "
                                                "<source>/data/train-*.parquet)")
    p.add_argument("--tokenizer", required=True, help="Gemma 4's tokenizer directory (tokenizer.json), as pool7's")
    p.add_argument("--iterations", type=int, default=ITERATIONS7, help="pool7's")
    p.add_argument("--per-iter", type=int, default=SOURCES7, help="pool7's")
    p.add_argument("--rlval", type=int, default=RLVAL_PER_PART7, help="pool7's")
    p.add_argument("--margin", type=float, default=CANDS_MARGIN7, help="candidates per part over what pool7 takes")
    p.add_argument("--out", required=True)
    p = sub.add_parser("pool7")
    p.add_argument("--cands", required=True, help="a directory with cands-flan, cands-novels, cands-bt .parquet (id, en, "
                                                   "doc, optionally stratum)")
    p.add_argument("--evalsets", required=True, help="$E/data/evalsets.parquet (13-grams)")
    p.add_argument("--blind", required=True, help="blind100.jsonl (13-grams)")
    p.add_argument("--c2-docs", required=True, help="$RL/c2/data/docs.txt (g4c2.py data7): C2's documents, out of the pool")
    p.add_argument("--flan-eval", required=True, help="FLAN's evaluation rows (inputs, targets): their 13-grams out")
    p.add_argument("--exclude-docs", action="append", default=[],
                   help="a text file of further documents to leave out (one per line)")
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--iterations", type=int, default=ITERATIONS7)
    p.add_argument("--per-iter", type=int, default=SOURCES7)
    p.add_argument("--rlval", type=int, default=RLVAL_PER_PART7, help="RL-val's sources per part")
    p.add_argument("--out", required=True)
    p = sub.add_parser("plan7")
    p.add_argument("--pool", required=True, help="pool7's directory")
    p.add_argument("--anchor-samples", required=True, help="the job anchor's samples/ (w<k>/ shards and DONE)")
    p.add_argument("--anchor-scores", required=True, help="the job anchor's scores/ (<role>/w<m>/ shards and DONE-<k>)")
    p.add_argument("--iterations", type=int, default=ITERATIONS7)
    p.add_argument("--per-iter", type=int, default=SOURCES7)
    p.add_argument("--rlval", type=int, default=RLVAL_PER_PART7, help="RL-val's sources per part")
    p.add_argument("--out", required=True)
    p = sub.add_parser("segments7")
    p.add_argument("--plan", default=None, help="plan7's plan.parquet (with --iteration)")
    p.add_argument("--iteration", type=int, default=None, help="the iteration (-1: RL-val)")
    p.add_argument("--pool", default=None, help="pool7's pool.parquet: every source, the job anchor (no --iteration)")
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
    # the smoke test (spec §8.3, G4-7.9)
    p = sub.add_parser("smokeplan7", help="the smoke's plan: iterations N (A) and N+1 (B) of 32 sources, RL-val")
    p.add_argument("--plan", required=True, help="plan7's plan.parquet")
    p.add_argument("--base", type=int, required=True, help=f"N (>= {SMOKE_BASE_MIN7}): A is iteration N, B N+1")
    p.add_argument("--sources", type=int, default=SMOKE_SOURCES7)
    p.add_argument("--out", required=True)
    p = sub.add_parser("smoke7-loose", help="the loose scorers' inputs of A's samples (metricx_qe.py, naturalness.py)")
    p.add_argument("--samples", required=True, help="iter/<N>/samples/ (w<k>/ shards and DONE)")
    p.add_argument("--out", required=True)
    p = sub.add_parser("check7", help="every packed row of an iter7 file against MaxText's own SFT steps")
    p.add_argument("--train", required=True, help="iter7's train.parquet (its samples.parquet beside it)")
    p.add_argument("--samples", required=True, help="the iteration's samples/ (w<k>/ shards and DONE)")
    p.add_argument("--tokenizer", required=True, help="Gemma 4's tokenizer directory")
    p.add_argument("--rows", type=int, default=ROWS7)
    p.add_argument("--processes", type=int, default=1)
    p.add_argument("--chunk", type=int, default=256)
    p.add_argument("--out", required=True, help="the check's JSON")
    p = sub.add_parser("smoke7", help="the smoke test's verdict (spec §8.3 points 1-7)")
    p.add_argument("--dir", required=True, help="the smoke's files as the chain puts them together")
    p.add_argument("--tokenizer", required=True, help="Gemma 4's tokenizer directory (decoded_ok)")
    p.add_argument("--out", required=True, help="a directory: smoke7.json, smoke7.md")
    args = parser.parse_args(argv)
    {"cands7": cands7, "pool7": pool7, "plan7": plan7, "segments7": segments7, "iter7": iter7, "stop7": stop7,
     "smokeplan7": smokeplan7, "smoke7-loose": smoke7_loose, "check7": check7, "smoke7": smoke7}[args.cmd](args)


if __name__ == "__main__":
    main()
