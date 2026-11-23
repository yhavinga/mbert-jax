#!/usr/bin/env python3
"""G4-8-data (protocol-g4.md §18; spec mbert-ops/translator-sft/g4-8-spec-judge.md): the selection data of arm G4-8,
best-of-8 of G4-0 with the Gemma 4 judge against the anchor (janchor). No training here.

24,000 English sources (FLAN 10 / novels 45 / BT 45, Yeb 23 Nov ~16:40): G4-7's 15,360 planned sources with their anchor
and fresh ones drawn as cands7/pool7 do; G4-0 samples each 8 times at T = 1 (translate.py --copies 8 on the workers);
MetricX-24-QE scores every sample and the anchor; a sample that is no fail of G4-7 (without c2_overflow), has QE <= QE0
and differs from the anchor is judged against the anchor in both orders (pairjudge.py score); per source the candidate
with the highest P̄(sample better) is the target if that P̄ > 0.5, else the source drops.

    python g4arm8.py fresh --g47-plan P --g47-pool P --flan-plan D --flan-kept D --flan-build-report F --flan-eval F \\
        --novels F --novels-report F --novels-eval F --bt D --evalsets E --blind B --c2-docs T --tokenizer T --out DIR
    python g4arm8.py plan --g47-plan P --fresh DIR --anchor-samples DIR --out DIR     (+ <out>/<chunk>/segments-w<k>)
    python g4arm8.py qepairs --plan P --chunk C --samples DIR --out pairs.parquet      (one worker's samples + anchors)
    python g4arm8.py pairs --plan P --chunk C --samples DIR --qe GLOB --out DIR        (judge prompts, cands.parquet)
    python g4arm8.py select --plan P --chunk C --samples DIR --cands F --scores GLOB --out DIR
    python g4arm8.py v3 --plan P --targets F --out DIR                                 (the blind package G4-8-V3)
    python g4arm8.py report --plan P --chunk-dir C=DIR ... --out DIR                    (report.md/.json, targets.parquet)

All CPU. The texts stay in the bucket and on the VM: no command prints a text; the reports hold counts and scores."""

import argparse
import collections
import json
import math
import random
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for _path in (HERE.parent / "flan-nl", HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import g4rl  # noqa: E402
import pairjudge  # noqa: E402

PROTOCOL = "protocol-g4.md §18"
ARM8 = "G4-8"
SEED8 = "en-nl-g4arm/0/G4-8"
TOTAL8 = 24000
PARTS8 = dict(g4rl.PARTS7)                 # Yeb 23 Nov ~16:40: FLAN 10 / novels 45 / BT 45, as G4-7 (G4-7.4d)
REUSED7 = {"flan": 1530, "novels": 6930, "bt": 6900}   # G4-7's plan, iterations 0-29 [read 23 Nov, g4-7-plan.json]
CHUNKS8 = (("c1", 4000), ("c2", 10000), ("c3", 10000))
MARGIN8 = 2.0                              # the fresh draw: this many times what the anchor takes
RESERVE8 = g4rl.RESERVE7                   # 3% more to the anchor, for the sources it drops
G8 = g4rl.G7                               # samples per source
WORKERS = 4
FAILS8 = tuple(f for f in g4rl.FAILS if f != "c2_overflow")   # spec §3: c2_overflow falls away
QE_TOL = 1e-9
ANCHOR_P = pairjudge.ANCHOR_P              # the target only if P̄(sample better) > 0.5 (janchor, G4-judge.12)
QE0_MEDIAN_MAX, QE0_BIG, QE0_BIG_SHARE = 0.05, 0.5, 0.02   # G4-8-data.2: the reused anchors' QE0 against G4-7's
V3_PARTS = ("bt", "novels")
V3_PER_PART = 50
V3_TARGET = "G4-8-janchor"
SFT_SEQ, SFT_ROWS, SFT_FILL, SFT_EPOCHS = 4096, 16, 0.95, 2      # spec §4 (the estimate of the report only)
TARGET_MEASURES = ("p_sample", "p_anchor", "p_gelijk", "qe", "ratio", "tokens_out", "tokens_in", "chrf")
READ_COLUMNS = ("uid", "id", "j", "source", "hypothesis", "ended", "decode_loop", "english_left", "empty", "tokens_in",
                "tokens_out")

FRESH_SCHEMA = (("id", "string"), ("part", "string"), ("stratum", "string"), ("doc", "string"), ("en", "string"),
                ("tokens", "int64"), ("rank", "int64"))
PLAN_SCHEMA = (("id", "string"), ("part", "string"), ("stratum", "string"), ("doc", "string"), ("en", "string"),
               ("tokens", "int64"), ("origin", "string"), ("chunk", "string"), ("hypothesis0", "string"),
               ("l0", "int64"), ("qe0_g47", "float64"))


# ---------------------------------------------------------------------------------------------------- small helpers

def read_rows(path, columns=None):
    return g4rl.read_rows(path, columns)


def write_parquet(rows, path, fields=None):
    g4rl.write_parquet(rows, path, g4rl.schema(fields) if fields else None)


def norm(text):
    """A text with its whitespace normalized (G4-8-data.4)"""
    return " ".join((text or "").split())


def need8():
    return g4rl.part_counts(TOTAL8, PARTS8)


def chunk_counts():
    """{chunk: {part: sources}}: each chunk the shares of PARTS8; together exactly need8()"""
    out = {c: g4rl.part_counts(n, PARTS8) for c, n in CHUNKS8}
    total = {p: sum(out[c][p] for c in out) for p in PARTS8}
    if total != need8():
        raise SystemExit(f"the chunks {out} do not add up to {need8()}")
    return out


def mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.fmean(xs) if xs else None


def median(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def share(k, n):
    return k / n if n else None


def chrf(hyp, ref, order=6, beta=2.0):
    """chrF of sacrebleu 2 (CHRF(char_order=6, word_order=0, beta=2), whitespace removed, no eps smoothing), 0-100"""
    def grams(text, n):
        return collections.Counter(text[i:i + n] for i in range(len(text) - n + 1))

    h, r = "".join(hyp.split()), "".join(ref.split())
    eps, factor = 1e-16, beta ** 2
    avg_p = avg_r = 0.0
    effective = 0
    for n in range(1, order + 1):
        hg, rg = grams(h, n), grams(r, n)
        n_hyp, n_ref = sum(hg.values()), sum(rg.values())
        match = sum((hg & rg).values())
        avg_p += match / n_hyp if n_hyp > 0 else eps
        avg_r += match / n_ref if n_ref > 0 else eps
        effective += n_hyp > 0 and n_ref > 0
    if effective == 0:
        return 0.0
    avg_p, avg_r = avg_p / effective, avg_r / effective
    if avg_p + avg_r == 0:
        return 0.0
    return 100 * (1 + factor) * avg_p * avg_r / (factor * avg_p + avg_r)


# ---------------------------------------------------------------------------------------------------- the fresh sources

def reused_counts(plan7):
    """{part: sources of G4-7's plan in iterations 0-29}; must be REUSED7"""
    got = {p: sum(r["part"] == p and r["iteration"] >= 0 for r in plan7) for p in PARTS8}
    if got != REUSED7:
        raise SystemExit(f"G4-7's plan has {got} sources in its iterations, not {REUSED7}")
    return got


def fresh_need(reused):
    """{part: (fresh sources wanted, taken to the anchor with the reserve, drawn)}"""
    out = {}
    for p, n in need8().items():
        fresh = n - reused[p]
        take = math.ceil(fresh * (1 + RESERVE8))
        out[p] = (fresh, take, math.ceil(MARGIN8 * take))
    return out


def unused(rows, old_ids, old_texts, counter):
    """rows whose id and stripped English are not G4-7's pool's"""
    out = []
    for r in rows:
        if r["id"] in old_ids:
            counter["g47_id"] += 1
        elif (r["en"] or "").strip() in old_texts:
            counter["g47_text"] += 1
        else:
            out.append(r)
    return out


def fresh_draw(parts, bt_sources, old, need, counter, blocked, excluded, doc_of=None):
    """(rows, report): the fresh sources. parts: {"flan": rows, "novels": rows} of cands7's builders (id, en, doc,
    stratum); bt_sources: {source: rows} (id, en, pid, stratum); old: (ids, texts) of G4-7's pool; need: fresh_need();
    per part (BT per source) without G4-7's ids and texts, drawn as cands7 (seed SEED8/cands/<part>[/<source>]), then
    pool7's candidate_filter (13-grams, excluded documents, one per text), then g4rl.ordered (SEED8/order/<part>) and the
    first `take` with their rank"""
    import g4c2

    doc_of = doc_of or g4c2.doc_key
    old_ids, old_texts = old
    report, drawn = {}, {}
    for part in ("flan", "novels"):
        c = collections.Counter()
        rows = unused(parts[part], old_ids, old_texts, c)
        drawn[part], funnel = g4rl.draw_candidates(rows, need[part][2], counter, f"{SEED8}/cands/{part}")
        report[part] = {"rows": len(parts[part]), "not_g47": len(rows), "out_g47": dict(c), "draw": funnel}
    per_source = math.ceil(need["bt"][2] / len(bt_sources))
    drawn["bt"], report["bt"] = [], {"per_source": per_source, "sources": {}}
    for source, rows_all in bt_sources.items():
        c = collections.Counter()
        rows = unused(rows_all, old_ids, old_texts, c)
        got, funnel = g4rl.draw_candidates(rows, per_source, counter, f"{SEED8}/cands/bt/{source}")
        drawn["bt"] += [{**r, "doc": doc_of(r["pid"])} for r in got]
        report["bt"]["sources"][source] = {"rows": len(rows_all), "not_g47": len(rows), "out_g47": dict(c),
                                           "draw": funnel}
    short = {p: len(drawn[p]) for p in PARTS8 if len(drawn[p]) < need[p][2]}
    short.update({f"bt/{s}": x["draw"]["drawn"] for s, x in report["bt"]["sources"].items()
                  if x["draw"]["drawn"] < per_source})
    if short:
        raise SystemExit(f"fresh: too few candidates {short}; wanted {({p: n[2] for p, n in need.items()})}")
    out = []
    for part in PARTS8:
        kept, drops = g4rl.candidate_filter(drawn[part], counter, blocked, excluded)
        rows = g4rl.ordered(kept, f"{SEED8}/order/{part}")
        take = need[part][1]
        if len(rows) < take:
            raise SystemExit(f"fresh: {part} keeps {len(rows)} after pool7's filters, {take} wanted")
        out += [{**{k: r.get(k) for k in ("id", "en", "doc", "tokens")}, "stratum": r.get("stratum") or "",
                 "part": part, "rank": k} for k, r in enumerate(rows[:take])]
        report[part]["filter"] = {"drawn": len(drawn[part]), "kept": len(kept), "drops": drops, "taken": take,
                                  "per_stratum": dict(sorted(collections.Counter(r.get("stratum") or "" for r in
                                                                                 rows[:take]).items()))}
    ids = [r["id"] for r in out]
    texts = [r["en"] for r in out]
    if len(set(ids)) != len(ids) or len(set(texts)) != len(texts):
        raise SystemExit("fresh: an id or a text twice among the fresh sources")
    if set(ids) & old_ids or {t.strip() for t in texts} & old_texts:
        raise SystemExit("fresh: a fresh source is G4-7's")
    return out, report


def fresh(args, counter=None):
    import g4arm
    import sft_build

    counter = counter or g4arm.TokenCounter(args.tokenizer)
    plan7 = read_rows(args.g47_plan, columns=["id", "part", "iteration"])
    pool7 = read_rows(args.g47_pool, columns=["id", "en", "doc", "role", "part"])
    reused = reused_counts(plan7)
    if not {r["id"] for r in plan7} <= {r["id"] for r in pool7}:
        raise SystemExit("G4-7's plan has a source that is not in its pool")
    old = ({r["id"] for r in pool7}, {(r["en"] or "").strip() for r in pool7})
    rlval_docs = {r["doc"] for r in pool7 if r["role"] == "rlval"}
    need = fresh_need(reused)
    blocked = g4arm.checked_blocked(args.evalsets, args.blind)
    blocked |= sft_build.flan_eval_ngrams(read_rows(args.flan_eval, columns=["inputs", "targets"]))
    c2_docs = g4rl.read_docs(args.c2_docs)
    print("PROGRESS flan", flush=True)
    flan, flan_info = g4rl.flan_cands(args.flan_plan, args.flan_kept, args.flan_build_report, args.flan_eval)
    print("PROGRESS novels", flush=True)
    novels, novels_info = g4rl.novels_cands(args.novels, args.novels_report, args.novels_eval)
    print("PROGRESS bt", flush=True)
    bt, revisions, bt_info = g4rl.bt_cands(args.bt)
    print("PROGRESS draw", flush=True)
    rows, report = fresh_draw({"flan": flan, "novels": novels}, bt, old, need, counter, blocked, c2_docs | rlval_docs)
    del flan, novels, bt
    out = Path(args.out)
    write_parquet(sorted(rows, key=lambda r: r["id"]), out / "fresh.parquet", FRESH_SCHEMA)
    for k, part in g4rl.assign_workers7(rows, WORKERS).items():
        write_parquet([{"id": r["id"], "text": r["en"], "tokens": r["tokens"]} for r in part],
                      out / f"segments-w{k}.parquet", (("id", "string"), ("text", "string"), ("tokens", "int64")))
    summary = {"protocol": PROTOCOL, "seed": SEED8, "shares": PARTS8, "total": TOTAL8, "need": need8(),
               "reused": reused, "fresh": {p: n[0] for p, n in need.items()}, "to_anchor": {p: n[1] for p, n in need.items()},
               "drawn": {p: n[2] for p, n in need.items()}, "g47_pool": {"ids": len(old[0]), "rlval_docs": len(rlval_docs)},
               "c2_docs": len(c2_docs), "parts": report,
               "builders": {"flan": flan_info, "novels": novels_info, "bt": {"revisions": revisions, "info": bt_info}}}
    (out / "fresh.json").write_text(json.dumps(summary, indent=1, default=str) + "\n")
    print(f"FRESH8 {json.dumps({'to_anchor': summary['to_anchor'], 'kept': {p: report[p]['filter']['kept'] for p in PARTS8}})}",
          flush=True)


# ---------------------------------------------------------------------------------------------------- samples

def read_samples8(root):
    """The samples (or anchors) of the w<k> directories under root, as g4rl.read_samples checks them (DONE, sha256,
    rows, uids), only READ_COLUMNS read"""
    import pyarrow.parquet as pq

    root = Path(root)
    dirs = sorted(p for p in root.glob("w*") if p.is_dir())
    if not dirs:
        raise SystemExit(f"{root}: no w<k> directory of samples")
    dones = g4rl.sample_dones(root)
    rows, seen = [], set()
    for d in dirs:
        done = dones[d.name[1:]]
        names = sorted(p.name for p in d.glob("*.parquet"))
        if names != sorted(done["shards"]):
            raise SystemExit(f"{d}: shards {names[:3]}... ({len(names)}) are not the {len(done['shards'])} of DONE")
        n = 0
        for name in names:
            if g4rl.file_sha256(d / name) != done["shards"][name]["sha256"]:
                raise SystemExit(f"{d / name}: its sha256 is not the one DONE names")
            shard = pq.read_table(d / name, columns=list(READ_COLUMNS)).to_pylist()
            if len(shard) != done["shards"][name]["rows"]:
                raise SystemExit(f"{d / name}: {len(shard)} rows, DONE says {done['shards'][name]['rows']}")
            for r in shard:
                if r["uid"] in seen:
                    raise SystemExit(f"{d / name}: {r['uid']} twice")
                if g4rl.split_uid(r["uid"]) != (str(r["id"]), int(r["j"])):
                    raise SystemExit(f"{d / name}: uid {r['uid']} is not id {r['id']} # j {r['j']}")
                seen.add(r["uid"])
                rows.append({**r, "id": str(r["id"]), "worker": d.name})
                n += 1
        if n != done["rows"]:
            raise SystemExit(f"{d}: {n} rows, DONE says {done['rows']}")
    return rows


def groups(samples, sources, g=G8):
    """{source id: [sample j = 0..g-1]} with every source of `sources` ({id: plan row}) exactly once, its English"""
    by = collections.defaultdict(dict)
    for s in samples:
        if s["id"] not in sources:
            raise SystemExit(f"{s['uid']}: not a source of this chunk")
        if (s["source"] or "").strip() != (sources[s["id"]]["en"] or "").strip():
            raise SystemExit(f"{s['uid']}: its source is not the plan's English")
        by[s["id"]][s["j"]] = s
    for i, js in by.items():
        if sorted(js) != list(range(g)):
            raise SystemExit(f"{i}: samples {sorted(js)}, not 0..{g - 1}")
    if set(by) != set(sources):
        raise SystemExit(f"{len(set(sources) - set(by))} sources of the chunk have no samples")
    return {i: [js[j] for j in range(g)] for i, js in by.items()}


def anchor_uid(source_id):
    return f"{source_id}#a"


# ---------------------------------------------------------------------------------------------------- the plan

def plan_rows(plan7, fresh_rows, anchors):
    """(the 24,000 plan rows, report): G4-7's planned sources with their anchor, and per part the first fresh sources in
    rank order whose anchor (G4-0 greedy, sample #0) is ok (g4rl.anchor_ok); each with its chunk"""
    reused_counts(plan7)
    need = need8()
    rows, report = [], {"anchor_dropped": {}, "fresh": {}, "reused": {}}
    for r in plan7:
        if r["iteration"] < 0:
            continue
        rows.append({"id": r["id"], "part": r["part"], "stratum": r.get("stratum") or "", "doc": r["doc"], "en": r["en"],
                     "tokens": int(r["tokens"]), "origin": "g4-7", "hypothesis0": r["hypothesis0"], "l0": int(r["l0"]),
                     "qe0_g47": r["qe0"]})
    fresh_ids = {r["id"] for r in fresh_rows}
    if set(anchors) != fresh_ids:
        raise SystemExit(f"the anchor has {len(anchors)} sources, the fresh sources are {len(fresh_ids)} "
                         f"({len(set(anchors) ^ fresh_ids)} differ)")
    for part in PARTS8:
        n_reused = sum(r["part"] == part for r in rows)
        want = need[part] - n_reused
        cands = sorted((r for r in fresh_rows if r["part"] == part), key=lambda r: r["rank"])
        ok = [r for r in cands if g4rl.anchor_ok(anchors[r["id"]])]
        report["anchor_dropped"][part] = {k: sum(not anchors[r["id"]][k] if k == "ended" else bool(anchors[r["id"]][k])
                                                 for r in cands) for k in ("ended", "decode_loop", "empty", "english_left")}
        if len(ok) < want:
            raise SystemExit(f"plan: {part} has {len(ok)} fresh sources with a good anchor, {want} needed")
        for r in ok[:want]:
            a = anchors[r["id"]]
            if (a["source"] or "").strip() != (r["en"] or "").strip():
                raise SystemExit(f"{r['id']}: the anchor's source is not the fresh source's English")
            rows.append({"id": r["id"], "part": part, "stratum": r.get("stratum") or "", "doc": r["doc"], "en": r["en"],
                         "tokens": int(r["tokens"]), "origin": "fresh", "hypothesis0": a["hypothesis"],
                         "l0": int(a["tokens_out"]), "qe0_g47": None})
        report["fresh"][part] = want
        report["reused"][part] = n_reused
    if len({r["id"] for r in rows}) != len(rows) or len({r["en"] for r in rows}) != len(rows):
        raise SystemExit("plan: an id or an English text twice")
    counts = chunk_counts()
    for part in PARTS8:
        mine = sorted((r for r in rows if r["part"] == part), key=lambda r: r["id"])
        random.Random(f"{SEED8}/chunk/{part}").shuffle(mine)
        k = 0
        for c, _ in CHUNKS8:
            for r in mine[k:k + counts[c][part]]:
                r["chunk"] = c
            k += counts[c][part]
    report["chunks"] = {c: {p: sum(r["chunk"] == c and r["part"] == p for r in rows) for p in PARTS8} for c, _ in CHUNKS8}
    report["origin_by_chunk"] = {c: dict(collections.Counter(r["origin"] for r in rows if r["chunk"] == c))
                                 for c, _ in CHUNKS8}
    report["bt_per_set"] = dict(sorted(collections.Counter(r["stratum"] for r in rows if r["part"] == "bt").items()))
    report["tokens_mean"] = {p: mean(r["tokens"] for r in rows if r["part"] == p) for p in PARTS8}
    report["sources"] = len(rows)
    return sorted(rows, key=lambda r: (r["chunk"], r["id"])), report


def plan(args):
    plan7 = read_rows(args.g47_plan)
    fresh_rows = read_rows(Path(args.fresh) / "fresh.parquet")
    anchors = {}
    for s in read_samples8(args.anchor_samples):
        if s["j"] != 0 or s["id"] in anchors:
            raise SystemExit(f"anchor: {s['uid']} is not one greedy translation #0 per source")
        anchors[s["id"]] = s
    rows, report = plan_rows(plan7, fresh_rows, anchors)
    out = Path(args.out)
    write_parquet(rows, out / "plan.parquet", PLAN_SCHEMA)
    for c, _ in CHUNKS8:
        for k, part in g4rl.assign_workers7([r for r in rows if r["chunk"] == c], WORKERS).items():
            write_parquet([{"id": r["id"], "text": r["en"], "tokens": r["tokens"]} for r in part],
                          out / c / f"segments-w{k}.parquet", (("id", "string"), ("text", "string"), ("tokens", "int64")))
    report["protocol"] = PROTOCOL
    (out / "plan.json").write_text(json.dumps(report, indent=1) + "\n")
    print(f"PLAN8 {json.dumps({k: report[k] for k in ('sources', 'chunks', 'anchor_dropped')})}", flush=True)


def chunk_sources(plan_path, chunk):
    rows = {r["id"]: r for r in read_rows(plan_path) if r["chunk"] == chunk}
    if not rows:
        raise SystemExit(f"{plan_path}: no source of chunk {chunk}")
    return rows


# ---------------------------------------------------------------------------------------------------- QE pairs

def qepairs(args):
    """MetricX's rows of one worker: its samples (id = uid) and the anchor of each of their sources (id = <id>#a)"""
    sources = chunk_sources(args.plan, args.chunk)
    samples = read_samples8(args.samples)
    mine = {s["id"] for s in samples}
    if not mine <= set(sources):
        raise SystemExit(f"{len(mine - set(sources))} sources of these samples are not of chunk {args.chunk}")
    grouped = groups(samples, {i: sources[i] for i in mine})
    rows = [{"id": s["uid"], "source": sources[i]["en"], "hypothesis": s["hypothesis"]}
            for i in sorted(grouped) for s in grouped[i]]
    rows += [{"id": anchor_uid(i), "source": sources[i]["en"], "hypothesis": sources[i]["hypothesis0"]}
             for i in sorted(grouped)]
    write_parquet(rows, args.out, (("id", "string"), ("source", "string"), ("hypothesis", "string")))
    print(f"QEPAIRS {json.dumps({'chunk': args.chunk, 'sources': len(grouped), 'rows': len(rows)})}", flush=True)


def read_qe(pattern):
    out = {}
    for r in pairjudge.read_glob(pattern, ["id", "qe_source_hypothesis"]):
        if r["id"] in out:
            raise SystemExit(f"QE: {r['id']} twice")
        out[r["id"]] = r["qe_source_hypothesis"]
    return out


# ---------------------------------------------------------------------------------------------------- fail, candidates

def fails8(sample, qe, l0):
    """(the fail reasons of G4-7 without c2_overflow, ratio l / l0)"""
    if qe is None and not sample["empty"]:
        raise SystemExit(f"{sample.get('uid')}: no QE, and its hypothesis is not empty")
    ratio = sample["tokens_out"] / l0
    checks = {"not_ended": not sample["ended"], "decode_loop": bool(sample["decode_loop"]),
              "empty": bool(sample["empty"]), "english_left": bool(sample["english_left"]),
              "length": not g4rl.LEN_BAND[0] <= ratio <= g4rl.LEN_BAND[1],
              "qe": qe is not None and qe > g4rl.QE_FAIL}
    return [k for k in FAILS8 if checks[k]], ratio


def source_candidates(src, samples, qe, qe0):
    """Per sample of one source: {j, fail, qe, ratio, same_as_anchor, candidate, key}; the key (<id>/j<j> of the first
    sample with that normalized text) of each candidate; the judged keys in order"""
    anchor = norm(src["hypothesis0"])
    out, key_of = [], {}
    for s in samples:
        q = qe[s["uid"]]
        fail, ratio = fails8(s, q, src["l0"])
        same = norm(s["hypothesis"]) == anchor
        cand = not fail and q is not None and q <= qe0 + QE_TOL and not same
        key = None
        if cand:
            key = key_of.setdefault(norm(s["hypothesis"]), f"{src['id']}/j{s['j']}")
        out.append({"uid": s["uid"], "id": src["id"], "j": s["j"], "fail": ",".join(fail), "qe": q, "qe0": qe0,
                    "ratio": ratio, "tokens_in": s["tokens_in"], "tokens_out": s["tokens_out"],
                    "same_as_anchor": same, "candidate": cand, "key": key})
    return out, key_of


def qe0_check(rows):
    """G4-8-data.2: the reused anchors' new QE0 against G4-7's; (stats, problem or None)"""
    deltas = [abs(r["qe0"] - r["qe0_g47"]) for r in rows if r["qe0_g47"] is not None and r["qe0"] is not None]
    if not deltas:
        return {"n": 0}, None
    stats = {"n": len(deltas), "median_abs": statistics.median(deltas), "max_abs": max(deltas),
             "share_over": sum(d > QE0_BIG for d in deltas) / len(deltas)}
    if stats["median_abs"] > QE0_MEDIAN_MAX or stats["share_over"] > QE0_BIG_SHARE:
        return stats, (f"the reused anchors' QE0 differs from G4-7's: median |d| {stats['median_abs']:.4f} "
                       f"(max {QE0_MEDIAN_MAX}), share > {QE0_BIG} {stats['share_over']:.4f} (max {QE0_BIG_SHARE})")
    return stats, None


def pairs(args):
    sources = chunk_sources(args.plan, args.chunk)
    grouped = groups(read_samples8(args.samples), sources)
    qe = read_qe(args.qe)
    want = {s["uid"] for g in grouped.values() for s in g} | {anchor_uid(i) for i in sources}
    if set(qe) != want:
        raise SystemExit(f"QE: {len(want - set(qe))} rows missing, {len(set(qe) - want)} extra")
    cands, prompts, anchors = [], [], []
    for i in sorted(sources):
        src = sources[i]
        qe0 = qe[anchor_uid(i)]
        if qe0 is None:
            raise SystemExit(f"{i}: the anchor has no QE (an empty anchor is not in the plan)")
        anchors.append({"qe0": qe0, "qe0_g47": src["qe0_g47"]})
        rows, key_of = source_candidates(src, grouped[i], qe, qe0)
        cands += [{**r, "part": src["part"], "distinct": len({norm(s["hypothesis"]) for s in grouped[i]})} for r in rows]
        first = {}
        for r in rows:
            if r["candidate"]:
                first.setdefault(r["key"], r["j"])
        for key in key_of.values():
            j = int(key.rsplit("/j", 1)[1])
            if first[key] != j:
                raise SystemExit(f"{key}: not the first sample of its text")
            hyp = grouped[i][j]["hypothesis"]
            prompts += [{"uid": u, "text": t} for u, t, _ in pairjudge.pair_prompts(key, src["en"], hyp, src["hypothesis0"])]
    stats, problem = qe0_check(anchors)
    if problem:
        raise SystemExit(problem)
    out = Path(args.out)
    workers = pairjudge.write_prompts(prompts, out)
    write_parquet(cands, out / "cands.parquet")
    summary = {"protocol": PROTOCOL, "chunk": args.chunk, "qe0_check": stats, "prompts": len(prompts), "workers": workers,
               "parts": pairs_stats(cands, sources)}
    (out / "pairs.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(f"PAIRS8 {json.dumps({'chunk': args.chunk, 'prompts': len(prompts), 'qe0_check': stats})}", flush=True)


def pairs_stats(cands, sources):
    out = {}
    for part in PARTS8:
        rs = [r for r in cands if r["part"] == part]
        ids = {r["id"] for r in rs}
        per = collections.Counter(r["id"] for r in rs if r["candidate"])
        keys = {r["key"] for r in rs if r["candidate"]}
        out[part] = {"sources": len(ids), "samples": len(rs),
                     "fail": {f: sum(f in r["fail"].split(",") for r in rs) for f in FAILS8},
                     "fail_any": sum(bool(r["fail"]) for r in rs), "same_as_anchor": sum(r["same_as_anchor"] for r in rs),
                     "qe_above_qe0": sum(not r["fail"] and r["qe"] is not None and r["qe"] > r["qe0"] + QE_TOL
                                         for r in rs),
                     "candidates": sum(r["candidate"] for r in rs), "keys": len(keys),
                     "sources_with_candidate": len(per)}
    return out


# ---------------------------------------------------------------------------------------------------- the selection

def choose8(cands, p_of):
    """(the chosen candidate row or None, reason): k = the highest P̄ (ties: lower QE, then the earlier sample); the
    target if P̄(k) > 0.5 ('sample'), else None ('p'); no judged candidate: None ('none')"""
    judged = [r for r in cands if r["candidate"] and p_of.get(r["key"]) is not None]
    if not judged:
        return None, "none"
    k = min(judged, key=lambda r: (-p_of[r["key"]]["A"], r["qe"], r["j"]))
    return (k, "sample") if p_of[k["key"]]["A"] > ANCHOR_P else (None, "p")


def select(args):
    sources = chunk_sources(args.plan, args.chunk)
    grouped = groups(read_samples8(args.samples), sources)
    cands = collections.defaultdict(list)
    for r in read_rows(args.cands):
        cands[r["id"]].append(r)
    if set(cands) != set(sources):
        raise SystemExit(f"{args.cands}: not the {len(sources)} sources of chunk {args.chunk}")
    scores = pairjudge.scores_by_uid(pairjudge.read_glob(args.scores))
    keys = {r["key"] for rs in cands.values() for r in rs if r["candidate"]}
    want = {f"{k}/{o}" for k in keys for o in pairjudge.ORDERS}
    if set(scores) != want:
        raise SystemExit(f"judge scores: {len(want - set(scores))} prompts missing, {len(set(scores) - want)} extra")
    p_of = {k: pairjudge.combine(scores[f"{k}/ab"], scores[f"{k}/ba"]) for k in keys}
    targets, choice = [], []
    for i in sorted(sources):
        src, rs = sources[i], sorted(cands[i], key=lambda r: r["j"])
        k, reason = choose8(rs, p_of)
        n_cand = sum(r["candidate"] for r in rs)
        n_keys = len({r["key"] for r in rs if r["candidate"]})
        not_scored = sum(1 for key in {r["key"] for r in rs if r["candidate"]} if p_of[key] is None)
        best = max((p_of[r["key"]]["A"] for r in rs if r["candidate"] and p_of[r["key"]] is not None), default=None)
        row = {"id": i, "part": src["part"], "stratum": src["stratum"], "origin": src["origin"], "chunk": args.chunk,
               "reason": reason, "j": -1 if k is None else k["j"], "p_best": best, "qe0": rs[0]["qe0"],
               "candidates": n_cand, "keys": n_keys, "not_scored": not_scored, "distinct": rs[0]["distinct"],
               "fails": sum(bool(r["fail"]) for r in rs), **dict.fromkeys(TARGET_MEASURES)}
        if k is not None:
            p = p_of[k["key"]]
            target = grouped[i][k["j"]]["hypothesis"]
            row.update({"p_sample": p["A"], "p_anchor": p["B"], "p_gelijk": p["gelijk"], "qe": k["qe"],
                        "ratio": k["ratio"], "tokens_out": k["tokens_out"], "tokens_in": k["tokens_in"],
                        "chrf": chrf(target, src["hypothesis0"])})
            targets.append({**row, "en": src["en"], "target": target, "doc": src["doc"], "l0": src["l0"]})
        choice.append(row)
    out = Path(args.out)
    write_parquet(choice, out / "choice.parquet")
    write_parquet(targets, out / "targets.parquet")
    stats = {p: {"sources": sum(r["part"] == p for r in choice),
                 "reasons": dict(collections.Counter(r["reason"] for r in choice if r["part"] == p))} for p in PARTS8}
    (out / "select.json").write_text(json.dumps({"protocol": PROTOCOL, "chunk": args.chunk, "stats": stats}, indent=1) + "\n")
    print(f"SELECT8 {json.dumps({'chunk': args.chunk, 'targets': len(targets), 'stats': stats})}", flush=True)


# ---------------------------------------------------------------------------------------------------- V3

def v3_package(targets, anchors):
    """(inputs, keys, report) of the blind package G4-8-V3 (G4-8-data.8): per part of V3_PARTS a seeded sample of
    V3_PER_PART of the sorted ids with a target; A/B per item seeded"""
    inputs, keys, report = [], [], {"protocol": PROTOCOL, "target": V3_TARGET, "sets": {}}
    by = {r["id"]: r for r in targets}
    n = 0
    for part in V3_PARTS:
        ids = sorted(i for i, r in by.items() if r["part"] == part)
        seed = f"{SEED8}/v3/{part}"
        picked = random.Random(seed).sample(ids, min(V3_PER_PART, len(ids)))
        report["sets"][part] = {"targets": len(ids), "picked": len(picked), "seed": seed,
                                "strata": dict(sorted(collections.Counter(by[i]["stratum"] for i in picked).items()))}
        for i in picked:
            n += 1
            b = f"v3-{n:03d}"
            ab_seed = f"{SEED8}/v3/ab/{b}"
            base_first = random.Random(ab_seed).random() < 0.5
            base, own = anchors[i], by[i]["target"]
            if norm(base) == norm(own):
                raise SystemExit(f"{i}: the target is the anchor's text")
            a, bb = (base, own) if base_first else (own, base)
            inputs.append({"blind_id": b, "source": by[i]["en"], "A": a, "B": bb})
            keys.append({"blind_id": b, "A": "base" if base_first else V3_TARGET, "B": V3_TARGET if base_first else "base",
                         "seed": ab_seed, "set": part, "item": i, "stratum": by[i]["stratum"]})
    report["n"] = len(keys)
    report["a_is_base"] = sum(k["A"] == "base" for k in keys)
    return inputs, keys, report


def v3(args):
    sources = {r["id"]: r for r in read_rows(args.plan, columns=["id", "hypothesis0"])}
    targets = read_rows(args.targets)
    inputs, keys, report = v3_package(targets, {i: r["hypothesis0"] for i, r in sources.items()})
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "judge-input.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in inputs)
    with open(out / "key.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(x) + "\n" for x in keys)
    (out / "select-report.json").write_text(json.dumps(report, indent=1) + "\n")
    print(f"V3 {json.dumps(report)}", flush=True)


# ---------------------------------------------------------------------------------------------------- the report

def part_report(choice, cands, qe0s):
    """The measures of G4-8-data.7 over one group of sources (their choice rows, their sample rows, their QE0)"""
    n, samples = len(choice), len(cands)
    kept = [r for r in choice if r["reason"] == "sample"]
    sft_tokens = sum(r["tokens_in"] + r["tokens_out"] for r in kept)
    return {
        "sources": n, "samples": samples,
        "fail": {f: share(sum(f in r["fail"].split(",") for r in cands), samples) for f in FAILS8},
        "fail_any": share(sum(bool(r["fail"]) for r in cands), samples),
        "same_as_anchor": share(sum(r["same_as_anchor"] for r in cands), samples),
        "candidates_per_source": share(sum(r["candidates"] for r in choice), n),
        "keys_per_source": share(sum(r["keys"] for r in choice), n),
        "no_candidate": sum(r["reason"] == "none" for r in choice),
        "not_scored": sum(r["not_scored"] for r in choice),
        "distinct_median": median(r["distinct"] for r in choice),
        "all_equal": share(sum(r["distinct"] == 1 for r in choice), n),
        "kept": len(kept), "kept_share": share(len(kept), n),
        "p_target_mean": mean(r["p_sample"] for r in kept),
        "p_best_mean": mean(r["p_best"] for r in choice),
        "qe0_mean": mean(qe0s), "qe_target_mean": mean(r["qe"] for r in kept),
        "dqe_mean": mean(r["qe"] - r["qe0"] for r in kept), "dqe_median": median(r["qe"] - r["qe0"] for r in kept),
        "ratio_median": median(r["ratio"] for r in kept), "ratio_mean": mean(r["ratio"] for r in kept),
        "chrf_median": median(r["chrf"] for r in kept), "chrf_mean": mean(r["chrf"] for r in kept),
        "chosen": dict(sorted(collections.Counter(f"s{r['j']}" for r in kept).items())),
        "sft_tokens": sft_tokens,
    }


def report(args):
    plan_rows_ = {r["id"]: r for r in read_rows(args.plan, columns=["id", "part", "origin", "chunk"])}
    choice, cands, targets, pairs_json = [], [], [], {}
    for spec in args.chunk_dir:
        c, d = spec.split("=", 1)
        choice += read_rows(Path(d) / "choice.parquet")
        cands += read_rows(Path(d) / "cands.parquet")
        targets += read_rows(Path(d) / "targets.parquet")
        pairs_json[c] = json.loads((Path(d) / "pairs.json").read_text())
    if sorted(r["id"] for r in choice) != sorted(plan_rows_):
        raise SystemExit(f"the chunks' choices cover {len(choice)} sources, the plan has {len(plan_rows_)}")
    origin = {i: r["origin"] for i, r in plan_rows_.items()}
    qe0 = {r["id"]: r["qe0"] for r in choice}
    groups_ = [("samen", lambda r: True)] + [(p, lambda r, p=p: r["part"] == p) for p in PARTS8]
    rep = {"protocol": PROTOCOL, "chunks": sorted(pairs_json), "parts": {}, "origin": {}, "qe0_check": {
        c: x["qe0_check"] for c, x in pairs_json.items()}, "prompts": {c: x["prompts"] for c, x in pairs_json.items()}}
    for name, f in groups_:
        ch = [r for r in choice if f(r)]
        rep["parts"][name] = part_report(ch, [r for r in cands if f(r)], [qe0[r["id"]] for r in ch])
    for p in PARTS8:
        for o in ("g4-7", "fresh"):
            ch = [r for r in choice if r["part"] == p and origin[r["id"]] == o]
            rep["origin"][f"{p}/{o}"] = {"sources": len(ch), "kept_share": share(sum(r["reason"] == "sample" for r in ch),
                                                                                 len(ch))}
    tokens = rep["parts"]["samen"]["sft_tokens"]
    rep["sft_estimate"] = {"tokens": tokens, "steps_per_epoch": math.ceil(tokens / (SFT_SEQ * SFT_ROWS * SFT_FILL)),
                           "epochs": SFT_EPOCHS, "seq": SFT_SEQ, "rows": SFT_ROWS, "fill": SFT_FILL}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_parquet(sorted(targets, key=lambda r: r["id"]), out / "targets.parquet")
    (out / "report.json").write_text(json.dumps(rep, indent=1) + "\n")
    (out / "report.md").write_text(report_text(rep), encoding="utf-8")
    print(f"REPORT8 {json.dumps({'targets': len(targets), 'kept_share': {k: v['kept_share'] for k, v in rep['parts'].items()}})}",
          flush=True)


def nl(x, digits=3):
    return "–" if x is None else f"{x:,.{digits}f}".replace(",", " ").replace(".", ",").replace(" ", ".")


def pc(x, digits=1):
    return "–" if x is None else nl(100 * x, digits) + "%"


def report_text(r):
    names = list(r["parts"])
    lines = [f"# G4-8-data: de selectiedata van G4-8 (janchor), rapport vóór de training ({r['protocol']})", "",
             "Alleen tellingen en scores; geen tekst gelezen. Doel = het sample met de hoogste P̄(sample beter dan het "
             "anker) onder de kandidaten (geen fail, QE ≤ QE₀, anders dan het anker), alleen als P̄ > 0,5.", "",
             "| maat | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    rows = [("bronnen", "sources", lambda x: f"{x:,}".replace(",", ".")),
            ("samples", "samples", lambda x: f"{x:,}".replace(",", ".")),
            ("fail (een reden)", "fail_any", pc)]
    rows += [(f"fail: {f}", None, f) for f in FAILS8]
    rows += [("gelijk aan het anker", "same_as_anchor", pc),
             ("kandidaten per bron", "candidates_per_source", lambda x: nl(x, 2)),
             ("beoordeelde teksten per bron", "keys_per_source", lambda x: nl(x, 2)),
             ("bronnen zonder kandidaat", "no_candidate", str), ("niet gescoorde kandidaten", "not_scored", str),
             ("verschillende samples per groep (mediaan)", "distinct_median", lambda x: nl(x, 1)),
             ("groepen met 8 gelijke", "all_equal", pc),
             ("**sample verslaat anker (P̄ > 0,5) = gehouden**", "kept_share", pc), ("gehouden bronnen", "kept", str),
             ("P̄ van het doel (gemiddeld)", "p_target_mean", nl), ("hoogste P̄ per bron (gemiddeld)", "p_best_mean", nl),
             ("QE₀ anker (gemiddeld, alle bronnen)", "qe0_mean", nl), ("QE doel (gemiddeld)", "qe_target_mean", nl),
             ("ΔQE doel − anker (gemiddeld)", "dqe_mean", nl), ("ΔQE (mediaan)", "dqe_median", nl),
             ("ℓ/ℓ₀ doel (mediaan)", "ratio_median", nl), ("ℓ/ℓ₀ doel (gemiddeld)", "ratio_mean", nl),
             ("chrF(doel, anker) (mediaan)", "chrf_median", lambda x: nl(x, 1)),
             ("chrF(doel, anker) (gemiddeld)", "chrf_mean", lambda x: nl(x, 1)),
             ("SFT-tokens van de doelen (prompt + doel)", "sft_tokens", lambda x: f"{x:,}".replace(",", "."))]
    for label, key, fmt in rows:
        if key is None:
            cells = [pc(r["parts"][n]["fail"][fmt]) for n in names]
        else:
            cells = [fmt(r["parts"][n][key]) if r["parts"][n][key] is not None else "–" for n in names]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines += ["", "Gekozen sample (s0…s7, gehouden bronnen): " + "; ".join(
        f"{n} {json.dumps(r['parts'][n]['chosen'])}" for n in names), "",
        "## Herkomst", "", "| deel/herkomst | bronnen | gehouden |", "|---|---|---|"]
    lines += [f"| {k} | {v['sources']:,} | {pc(v['kept_share'])} |".replace(",", ".") for k, v in r["origin"].items()]
    e = r["sft_estimate"]
    lines += ["", "## Controles en schatting", "",
              "QE₀ van de hergebruikte ankers tegen G4-7's qe0 (G4-8-data.2), per deel: " + "; ".join(
                  f"{c}: n {x.get('n', 0)}, mediaan |Δ| {nl(x.get('median_abs'), 4)}, max {nl(x.get('max_abs'), 3)}, "
                  f"aandeel > 0,5 {pc(x.get('share_over'), 2)}" for c, x in r["qe0_check"].items()) + ".",
              "", "Judge-prompts per deel: " + ", ".join(f"{c} {n:,}".replace(",", ".") for c, n in r["prompts"].items()) + ".",
              "", f"[schatting] SFT: {e['tokens']:,} tokens".replace(",", ".") +
              f" ≈ {e['steps_per_epoch']} stappen per epoch van {e['rows']} × {e['seq']:,}".replace(",", ".") +
              f" (vulling {nl(e['fill'], 2)}), {e['epochs']} epochs ≈ {e['epochs'] * e['steps_per_epoch']} stappen.", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------------- main

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("fresh")
    for name in ("g47-plan", "g47-pool", "flan-plan", "flan-kept", "flan-build-report", "flan-eval", "novels",
                 "novels-report", "novels-eval", "bt", "evalsets", "blind", "c2-docs", "tokenizer", "out"):
        a.add_argument(f"--{name}", required=True)
    a = sub.add_parser("plan")
    for name in ("g47-plan", "fresh", "anchor-samples", "out"):
        a.add_argument(f"--{name}", required=True)
    a = sub.add_parser("qepairs")
    for name in ("plan", "chunk", "samples", "out"):
        a.add_argument(f"--{name}", required=True)
    a = sub.add_parser("pairs")
    for name in ("plan", "chunk", "samples", "qe", "out"):
        a.add_argument(f"--{name}", required=True)
    a = sub.add_parser("select")
    for name in ("plan", "chunk", "samples", "cands", "scores", "out"):
        a.add_argument(f"--{name}", required=True)
    a = sub.add_parser("v3")
    for name in ("plan", "targets", "out"):
        a.add_argument(f"--{name}", required=True)
    a = sub.add_parser("report")
    a.add_argument("--plan", required=True)
    a.add_argument("--chunk-dir", action="append", required=True,
                   help="CHUNK=DIR with choice.parquet, targets.parquet, cands.parquet, pairs.json")
    a.add_argument("--out", required=True)
    args = p.parse_args(argv)
    if getattr(args, "chunk", None) is not None and args.chunk not in dict(CHUNKS8):
        raise SystemExit(f"--chunk {args.chunk}: one of {[c for c, _ in CHUNKS8]}")
    {"fresh": fresh, "plan": plan, "qepairs": qepairs, "pairs": pairs, "select": select, "v3": v3,
     "report": report}[args.cmd](args)


if __name__ == "__main__":
    main()
