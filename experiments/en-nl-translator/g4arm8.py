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
    python g4arm8.py v3 --set n1 --plan P --targets F --choice F --rubric F --n1report F --out DIR
        (§22 G4-9.V3N1, the blind package G4-9-V3N1: §21's N1 targets of c1, select-n1's targets and choice, §18's choice)
    python g4arm8.py report --plan P --chunk-dir C=DIR ... --out DIR                    (report.md/.json, targets.parquet)

§21 (G4-8-data N1-scores, a preparation that decides nothing): the same pairs judged with FB.8's prompt N1 (pairjudge.N1_PROMPT);
select unchanged on N1's scores; the report beside the rubric's choice of §18:
    python g4arm8.py n1pairs --plan P --chunk C --samples DIR --cands F --rubric DIR --out DIR   (§18's prompts in DIR)
    python g4arm8.py n1report --plan P --chunk-dir C=DIR ... --out DIR   (DIR: n1pairs.json, cands.parquet, choice.parquet,
        choice-n1.parquet, scores/w<k>/shard-*.parquet, scores-n1/w<k>/shard-*.parquet)

G4-8's training (protocol-g4.md §19), and G4-9's (§22, a draft until Yeb decides: §19's training on the N1 choice of
§21) with --arm G4-9 (its seed, names and places; default --arm G4-8):
    python g4arm8.py n1targets --plan P --n1report F --chunk-dir C=DIR ... --out DIR   (G4-9's targets: §21's select-n1)
    python g4arm8.py sftset --targets T --plan P --evalsets E --blind B --flan-eval F --g47-pool P --g47-plan P \
        --sft-eval books=F --sft-eval btval=F --sft-eval own=F --tokenizer T --out DIR   (set, smoke set, RL-val)
    python g4arm8.py smoke8 --run R --summary S --adapter A --adapter-step A --smoke-manifest M --manifest M --out F
    python g4arm8.py trained --run R --summary S --manifest M --adapter G4-8-s<k>=JSON ... --out F
    python g4arm8.py loraline --logs 'DIR/g4a_translate_<candidate>_*' --out F       (translate.py's LORA line)
    python g4arm8.py report8 --evalsets E --split S --refqe Q --base DIR --candidate N=DIR ... [--reference N=DIR]
        --predictions P --predictions-c2 P [--pll P] --summary S --manifest M [--smoke F] --lora N=JSON ...
        --div N=DIR ... --rlval G47PLAN --o6 R-oud|R-nieuw|open [--tally F] [--compare N] --out report.md

All CPU. The texts stay in the bucket and on the VM: no command prints a text; the reports hold counts and scores."""

import argparse
import collections
import hashlib
import json
import math
import random
import re
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
# the blind packages in V3's form (v3 --set): G4-8-V3 (G4-8-data.8) and G4-9-V3N1 (§22 G4-9.V3N1: the N1 targets of c1);
# parts as (part, picked, per stratum)
V3_SETS = {"": {"protocol": PROTOCOL, "target": V3_TARGET, "seed": f"{SEED8}/v3", "prefix": "v3", "chunk": "c1",
                "parts": tuple((part, V3_PER_PART, False) for part in V3_PARTS)},
           "n1": {"protocol": "protocol-g4.md §22 G4-9.V3N1", "target": "G4-9-n1", "seed": "en-nl-g4arm/0/G4-9/v3n1",
                  "prefix": "v3n1", "chunk": "c1",
                  "parts": (("bt", 8, True), ("novels", 40, False), ("flan", 20, False))}}
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

def rubric_label(t, rubric):
    """§22 G4-9.V3N1: what §18's rubric chose for a source with an N1 target (G4-8-N1.7.2): zelfde sample, ander
    sample or anker"""
    label = n1_label(("sample", t["j"]), rubric[t["id"]])
    if label not in ("sample/zelfde sample", "sample/ander sample", "sample/anker"):
        raise SystemExit(f"{t['id']}: an N1 target where §18's rubric chose {rubric[t['id']]}")
    return label.split("/", 1)[1]


def v3_package(targets, anchors, spec=None, rubric=None):
    """(inputs, keys, report) of a blind package in V3's form: G4-8-V3 (G4-8-data.8, V3_SETS[""]) or G4-9-V3N1
    (§22 G4-9.V3N1, V3_SETS["n1"], with rubric = {id: (reason, j)} of §18's choice): per part a seeded sample of the
    sorted ids with a target (per stratum if the spec says so, the strata by name); A/B per item seeded"""
    spec = spec or V3_SETS[""]
    name, seed0 = spec["target"], spec["seed"]
    inputs, keys, report = [], [], {"protocol": spec["protocol"], "target": name, "sets": {}}
    by = {r["id"]: r for r in targets}
    n = 0
    for part, per, per_stratum in spec["parts"]:
        ids = sorted(i for i, r in by.items() if r["part"] == part)
        if per_stratum:
            groups_ = {s: [i for i in ids if by[i]["stratum"] == s] for s in sorted({by[i]["stratum"] for i in ids})}
            seed = {s: f"{seed0}/{part}/{s}" for s in groups_}
            picked = [i for s, g in groups_.items() for i in random.Random(seed[s]).sample(g, min(per, len(g)))]
        else:
            seed = f"{seed0}/{part}"
            picked = random.Random(seed).sample(ids, min(per, len(ids)))
        rep = report["sets"][part] = {
            "targets": len(ids), "picked": len(picked), "seed": seed,
            "strata": dict(sorted(collections.Counter(by[i]["stratum"] for i in picked).items()))}
        if per_stratum:
            rep["targets_strata"] = {s: len(g) for s, g in groups_.items()}
        if rubric is not None:
            label = {i: rubric_label(by[i], rubric) for i in ids}
            rep["rubric"] = dict(sorted(collections.Counter(label[i] for i in picked).items()))
            rep["rubric_targets"] = dict(sorted(collections.Counter(label.values()).items()))
        for i in picked:
            n += 1
            b = f"{spec['prefix']}-{n:03d}"
            ab_seed = f"{seed0}/ab/{b}"
            base_first = random.Random(ab_seed).random() < 0.5
            base, own = anchors[i], by[i]["target"]
            if norm(base) == norm(own):
                raise SystemExit(f"{i}: the target is the anchor's text")
            a, bb = (base, own) if base_first else (own, base)
            inputs.append({"blind_id": b, "source": by[i]["en"], "A": a, "B": bb})
            keys.append({"blind_id": b, "A": "base" if base_first else name, "B": name if base_first else "base",
                         "seed": ab_seed, "set": part, "item": i, "stratum": by[i]["stratum"],
                         **({} if rubric is None else {"rubric": label[i]})})
    report["n"] = len(keys)
    report["a_is_base"] = sum(k["A"] == "base" for k in keys)
    return inputs, keys, report


def v3n1_inputs(args, c):
    """§22 G4-9.V3N1: the N1 targets of chunk c with G4-9.2's checks (n1_chunk), and §18's choice of c; the plan and
    §18's choice must be the ones §21's n1report read. (targets, {id: (reason, j)} of §18, sha256 of the inputs)"""
    rep = json.loads(Path(args.n1report).read_text())
    if rep.get("protocol") != PROTOCOL_N1 or c not in rep.get("chunks", []):
        raise SystemExit(f"{args.n1report} is not {PROTOCOL_N1}'s report over {c}")
    if g4rl.file_sha256(args.plan) != rep["inputs"][c]["n1pairs"]["plan"]:
        raise SystemExit(f"{args.plan} is not the plan that §21's n1pairs read ({c})")
    if g4rl.file_sha256(args.rubric) != rep["inputs"][c]["choice.parquet"]:
        raise SystemExit(f"{args.rubric} is not §18's choice.parquet that §21's n1report checked ({c})")
    plan = {r["id"]: r for r in read_rows(args.plan, ["id", "part", "chunk", "en", "hypothesis0"])}
    targets, _ = n1_chunk(c, Path(args.choice), Path(args.targets), plan, rep)
    rubric = {r["id"]: (r["reason"], r["j"]) for r in read_rows(args.rubric, ["id", "reason", "j"])}
    if set(rubric) != {i for i, r in plan.items() if r["chunk"] == c}:
        raise SystemExit(f"{args.rubric}: not the sources of chunk {c} in the plan")
    inputs = {k: g4rl.file_sha256(getattr(args, k)) for k in ("plan", "targets", "choice", "rubric", "n1report")}
    return targets, rubric, inputs


def v3(args):
    spec = V3_SETS[args.set]
    extra = [k for k in ("choice", "rubric", "n1report") if getattr(args, k) is not None]
    if args.set == "n1" and len(extra) < 3:
        raise SystemExit("v3 --set n1 takes --choice, --rubric and --n1report")
    if args.set != "n1" and extra:
        raise SystemExit(f"--{', --'.join(extra)}: only with v3 --set n1")
    sources = {r["id"]: r for r in read_rows(args.plan, columns=["id", "hypothesis0"])}
    if args.set == "n1":
        targets, rubric, used = v3n1_inputs(args, spec["chunk"])
    else:
        targets, rubric, used = read_rows(args.targets), None, None
    inputs, keys, report = v3_package(targets, {i: r["hypothesis0"] for i, r in sources.items()}, spec, rubric)
    if used is not None:
        report.update({"chunk": spec["chunk"], "inputs": used})
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


# ---------------------------------------------------------------------------------------------------- §21: N1

PROTOCOL_N1 = "protocol-g4.md §21"
N1_BINS = 10                               # G4-8-N1.7: P̄ in tenths


def shards_sha256(paths):
    """One sha256 over the (name, sha256) of each file, in the given order"""
    h = hashlib.sha256()
    for path in paths:
        h.update(f"{Path(path).parent.name}/{Path(path).name} {g4rl.file_sha256(path)}\n".encode())
    return h.hexdigest()


def text_digests(rows):
    return {r["uid"]: hashlib.sha256(r["text"].encode("utf-8")).hexdigest() for r in rows}


def judged_keys(cands_rows, sources):
    """{key: (source id, j)} of the candidates; the key must be <id>/j<j> of the first candidate with its text"""
    first = {}
    for r in cands_rows:
        if r["id"] not in sources:
            raise SystemExit(f"cands: {r['id']} is not a source of this chunk")
        if r["candidate"]:
            i, j = first.get(r["key"], (r["id"], r["j"]))
            if i != r["id"]:
                raise SystemExit(f"cands: the key {r['key']} in sources {i} and {r['id']}")
            first[r["key"]] = (i, min(j, r["j"]))
    for key, (i, j) in first.items():
        if key != f"{i}/j{j}":
            raise SystemExit(f"cands: {key} is not the first sample of its text ({i}/j{j})")
    return first


def n1pairs(args):
    """G4-8-N1.2-3: N1's prompts of §18's judged keys of one chunk, written only after the rubric prompts of the same
    inputs are §18's per uid (sha256 of the text) and per worker"""
    sources = chunk_sources(args.plan, args.chunk)
    grouped = groups(read_samples8(args.samples), sources)
    first = judged_keys(read_rows(args.cands, ["id", "j", "candidate", "key"]), sources)
    rubric, n1 = [], []
    for key in sorted(first):
        i, j = first[key]
        src, hyp = sources[i], grouped[i][j]["hypothesis"]
        for template, rows in ((pairjudge.PROMPT, rubric), (pairjudge.N1_PROMPT, n1)):
            rows += [{"uid": u, "text": t} for u, t, _ in
                     pairjudge.pair_prompts(key, src["en"], hyp, src["hypothesis0"], template)]
    was = {}
    for k in range(WORKERS):
        rows = read_rows(Path(args.rubric) / f"prompts-w{k}.parquet")
        if any(pairjudge.worker_of(r["uid"]) != k for r in rows):
            raise SystemExit(f"§18's prompts-w{k}.parquet holds a uid of another worker")
        for uid, h in text_digests(rows).items():
            if uid in was:
                raise SystemExit(f"§18's prompts hold {uid} twice")
            was[uid] = h
    now = text_digests(rubric)
    if was != now:
        raise SystemExit(f"the rubric prompts of these inputs are not §18's: {sum(was.get(u) != h for u, h in now.items())} "
                         f"of {len(now)} differ or are new, {len(set(was) - set(now))} of §18's missing")
    inputs = {"plan": g4rl.file_sha256(args.plan), "cands": g4rl.file_sha256(args.cands),
              **{f"rubric_prompts_w{k}": g4rl.file_sha256(Path(args.rubric) / f"prompts-w{k}.parquet")
                 for k in range(WORKERS)},
              **{f"samples_{d.name}_DONE": g4rl.file_sha256(d / "DONE")
                 for d in sorted(p for p in Path(args.samples).glob("w*") if p.is_dir())}}
    out = Path(args.out)
    workers = pairjudge.write_prompts(n1, out)
    summary = {"protocol": PROTOCOL_N1, "chunk": args.chunk, "prompt": "n1", "keys": len(first), "prompts": len(n1),
               "workers": workers, "same_as_18": {"rubric_prompts": len(now)}, "inputs": inputs}
    (out / "n1pairs.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(f"N1PAIRS {json.dumps({k: summary[k] for k in ('chunk', 'keys', 'prompts', 'workers', 'same_as_18')})}",
          flush=True)


def p_of_keys(pattern, keys):
    """{key: P̄ or None (not scored)} from score shards that hold exactly the keys in both orders"""
    scores = pairjudge.scores_by_uid(pairjudge.read_glob(pattern))
    want = {f"{k}/{o}" for k in keys for o in pairjudge.ORDERS}
    if set(scores) != want:
        raise SystemExit(f"{pattern}: {len(want - set(scores))} prompts missing, {len(set(scores) - want)} extra")
    return {k: pairjudge.combine(scores[f"{k}/ab"], scores[f"{k}/ba"]) for k in keys}


def p_dist(ps):
    """G4-8-N1.7.3: mean, quartiles, share > 0.5 and the histogram in tenths of P̄ values (None left out)"""
    ps = sorted(p for p in ps if p is not None)
    if not ps:
        return {"n": 0}
    q = statistics.quantiles(ps, n=4, method="inclusive") if len(ps) > 1 else [ps[0]] * 3
    hist = collections.Counter(min(int(p * N1_BINS), N1_BINS - 1) for p in ps)
    return {"n": len(ps), "mean": statistics.fmean(ps), "q1": q[0], "median": q[1], "q3": q[2],
            "over_half": share(sum(p > ANCHOR_P for p in ps), len(ps)), "hist": [hist.get(b, 0) for b in range(N1_BINS)]}


def n1_label(n1, rubric):
    """G4-8-N1.7.2: N1's choice against the rubric's, from (reason, j) of both"""
    names = {"sample": "sample", "p": "anker", "none": "geen"}
    if n1[0] == rubric[0] == "sample":
        return "sample/" + ("zelfde sample" if n1[1] == rubric[1] else "ander sample")
    return f"{names[n1[0]]}/{names[rubric[0]]}"


def n1report(args):
    import g4c2

    plan_rows_ = {r["id"]: r for r in read_rows(args.plan, ["id", "part", "chunk"])}
    rows, keys_rows, inputs, prompts = [], [], {}, {}
    for spec in args.chunk_dir:
        c, d = spec.split("=", 1)
        d = Path(d)
        info = json.loads((d / "n1pairs.json").read_text())
        if (info.get("protocol"), info.get("chunk")) != (PROTOCOL_N1, c):
            raise SystemExit(f"{d / 'n1pairs.json'} is not {PROTOCOL_N1}'s of chunk {c}")
        if g4rl.file_sha256(d / "cands.parquet") != info["inputs"]["cands"]:
            raise SystemExit(f"{d / 'cands.parquet'} is not the one n1pairs read")
        sources = {i: r for i, r in plan_rows_.items() if r["chunk"] == c}
        cands = collections.defaultdict(list)
        for r in read_rows(d / "cands.parquet"):
            cands[r["id"]].append(r)
        keys = set(judged_keys([r for rs in cands.values() for r in rs], sources))
        p_r = p_of_keys(str(d / "scores" / "w*" / "shard-*.parquet"), keys)
        p_n = p_of_keys(str(d / "scores-n1" / "w*" / "shard-*.parquet"), keys)
        made = {}
        for name in ("choice.parquet", "choice-n1.parquet"):
            made[name] = {r["id"]: (r["reason"], r["j"]) for r in read_rows(d / name, ["id", "reason", "j"])}
            if set(made[name]) != set(sources):
                raise SystemExit(f"{d / name}: not the {len(sources)} sources of chunk {c}")
        for i in sorted(sources):
            rs = sorted(cands[i], key=lambda r: r["j"])
            got = {}
            for name, p_of in (("choice.parquet", p_r), ("choice-n1.parquet", p_n)):
                k, reason = choose8(rs, p_of)
                got[name] = (reason, -1 if k is None else k["j"])
                if got[name] != made[name][i]:
                    raise SystemExit(f"{c} {i}: {name} says {made[name][i]}, the rule on its scores {got[name]}")
            best = {n: max((p[r["key"]]["A"] for r in rs if r["candidate"] and p[r["key"]] is not None), default=None)
                    for n, p in (("rubric", p_r), ("n1", p_n))}
            rows.append({"id": i, "part": sources[i]["part"], "n1": got["choice-n1.parquet"],
                         "rubric": got["choice.parquet"], "best_n1": best["n1"], "best_rubric": best["rubric"]})
        part_of = {k: sources[k.rsplit("/j", 1)[0]]["part"] for k in keys}
        keys_rows += [{"part": part_of[k], "n1": None if p_n[k] is None else p_n[k]["A"],
                       "rubric": None if p_r[k] is None else p_r[k]["A"]} for k in sorted(keys)]
        prompts[c] = info["prompts"]
        inputs[c] = {"n1pairs": info["inputs"],
                     **{name: g4rl.file_sha256(d / name) for name in ("choice.parquet", "choice-n1.parquet")},
                     **{f"{s}_shards": shards_sha256(sorted(d.glob(f"{s}/w*/shard-*.parquet")))
                        for s in ("scores", "scores-n1")}}
    if sorted(r["id"] for r in rows) != sorted(plan_rows_):
        raise SystemExit(f"the chunks cover {len(rows)} sources, the plan has {len(plan_rows_)}")
    rep = {"protocol": PROTOCOL_N1, "chunks": sorted(prompts), "prompts": prompts, "inputs": inputs, "parts": {}}
    for name in ("samen", *PARTS8):
        rs = [r for r in rows if name == "samen" or r["part"] == name]
        ks = [r for r in keys_rows if name == "samen" or r["part"] == name]
        both = [(r["n1"], r["rubric"]) for r in ks if r["n1"] is not None and r["rubric"] is not None]
        judged = [r for r in rs if r["n1"][0] != "none" or r["rubric"][0] != "none"]
        rep["parts"][name] = {
            "sources": len(rs), "judged": len(judged),
            "n1_kept": sum(r["n1"][0] == "sample" for r in rs),
            "n1_kept_share": share(sum(r["n1"][0] == "sample" for r in rs), len(rs)),
            "n1_kept_share_judged": share(sum(r["n1"][0] == "sample" for r in rs), len(judged)),
            "rubric_kept": sum(r["rubric"][0] == "sample" for r in rs),
            "rubric_kept_share": share(sum(r["rubric"][0] == "sample" for r in rs), len(rs)),
            "not_scored": {"n1": sum(r["n1"] is None for r in ks), "rubric": sum(r["rubric"] is None for r in ks)},
            "overlap": dict(sorted(collections.Counter(n1_label(r["n1"], r["rubric"]) for r in judged).items())),
            "p_keys": {"n1": p_dist(r["n1"] for r in ks), "rubric": p_dist(r["rubric"] for r in ks)},
            "p_best": {"n1": p_dist(r["best_n1"] for r in rs), "rubric": p_dist(r["best_rubric"] for r in rs)},
            "spearman_keys": g4c2.spearman([a for a, _ in both], [b for _, b in both]), "keys_both": len(both)}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(rep, indent=1) + "\n")
    (out / "report.md").write_text(n1report_text(rep), encoding="utf-8")
    print(f"N1REPORT {json.dumps({k: {m: v[m] for m in ('sources', 'n1_kept', 'rubric_kept', 'overlap')} for k, v in rep['parts'].items()})}",
          flush=True)


N1_OVERLAP = (("sample/zelfde sample", "N1 kiest een sample, de rubriek hetzelfde"),
              ("sample/ander sample", "N1 kiest een sample, de rubriek een ander"),
              ("sample/anker", "N1 kiest een sample, de rubriek het anker"),
              ("anker/sample", "N1 kiest het anker, de rubriek een sample"),
              ("anker/anker", "beide het anker"))
N1_PART_NAMES = {"samen": "samen", "flan": "FLAN", "novels": "romans", "bt": "BT"}


def n1report_text(r):
    names = list(r["parts"])
    head = ["| maat | " + " | ".join(N1_PART_NAMES[n] for n in names) + " |", "|---|" + "---|" * len(names)]

    def row(label, f):
        return f"| {label} | " + " | ".join(f(r["parts"][n]) for n in names) + " |"

    def count(x):
        return f"{x:,}".replace(",", ".")

    lines = [f"# G4-8-data N1-scores: voorbereiding, beslist niets ({r['protocol']})", "",
             "Alleen tellingen en scores; geen tekst gelezen. Dezelfde paren als de rubriek-scoring van §18 (kandidaat = "
             "geen fail, QE ≤ QE₀, anders dan het anker; A = sample, B = anker, beide volgordes), beoordeeld met de "
             "natuurlijkheidsprompt N1 van §20 FB.8. Keuze per bron: de kandidaat met de hoogste P̄ als P̄ > 0,5, anders "
             "het anker (de regel van G4-8-data.6, voor N1 en voor de rubriek).", "", "## Keuzes", ""] + head
    lines += [row("bronnen", lambda p: count(p["sources"])),
              row("bronnen met een beoordeelde kandidaat", lambda p: count(p["judged"])),
              row("**N1 zet een sample boven het anker (P̄ > 0,5, QE ≤ QE₀)**",
                  lambda p: f"{count(p['n1_kept'])} ({pc(p['n1_kept_share'])})"),
              row("idem, van de bronnen met een kandidaat", lambda p: pc(p["n1_kept_share_judged"])),
              row("rubriek (§18) zet een sample boven het anker", lambda p: f"{count(p['rubric_kept'])} ({pc(p['rubric_kept_share'])})"),
              row("niet gescoorde teksten (N1 / rubriek)", lambda p: f"{p['not_scored']['n1']} / {p['not_scored']['rubric']}")]
    lines += ["", "## Overlap met de rubriekkeuze van §18 (bronnen met een beoordeelde kandidaat)", ""] + head
    for key, label in N1_OVERLAP:
        lines.append(row(label, lambda p, key=key: f"{count(p['overlap'].get(key, 0))} ({pc(share(p['overlap'].get(key, 0), p['judged']))})"))
    rest = sorted({k for p in r["parts"].values() for k in p["overlap"]} - {k for k, _ in N1_OVERLAP})
    for key in rest:
        lines.append(row(f"overig: {key}", lambda p, key=key: count(p["overlap"].get(key, 0))))
    lines += ["", "## Verdeling van P̄(sample beter dan het anker)", "",
              "Per beoordeelde tekst (sleutel) en de hoogste per bron; kwartielen inclusief; histogram in tienden "
              "[0,0–0,1) … [0,9–1,0].", "", "| deel | wat | prompt | n | gemiddeld | Q1 | mediaan | Q3 | > 0,5 | histogram |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for n in names:
        for what, label in (("p_keys", "per tekst"), ("p_best", "hoogste per bron")):
            for prompt in ("n1", "rubric"):
                x = r["parts"][n][what][prompt]
                lines.append(f"| {N1_PART_NAMES[n]} | {label} | {'N1' if prompt == 'n1' else 'rubriek'} | {count(x['n'])} | "
                             f"{nl(x.get('mean'))} | {nl(x.get('q1'))} | {nl(x.get('median'))} | {nl(x.get('q3'))} | "
                             f"{pc(x.get('over_half'))} | {' '.join(map(str, x.get('hist', [])))} |")
    lines += ["", "Spearman tussen P̄_N1 en P̄_rubriek per tekst: " + "; ".join(
        f"{N1_PART_NAMES[n]} {nl(r['parts'][n]['spearman_keys'])} (n {count(r['parts'][n]['keys_both'])})" for n in names) + ".",
              "", "N1-prompts per deel: " + ", ".join(f"{c} {count(k)}" for c, k in r["prompts"].items()) + ".", "",
              "Wat het niet zegt (§21 G4-8-N1.8): N1 sluit juistheid uit; de enige juistheidsbewaking is fail + QE ≤ QE₀. "
              "De tellingen zeggen hoe vaak N1 en de rubriek hetzelfde kiezen, niet welke keuze beter is. Beslist niets.", ""]
    return "\n".join(lines)


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


# ---------------------------------------------------------------------------------------------------- G4-8 training

PROTOCOL_T = "protocol-g4.md §19"
EPOCHS_T = 2                               # §19 G4-8.2 (spec O5)
LR_T = 5e-5                                # G4-8.3 (spec O5, as G4-7.4k)
WARMUP_T = 10                              # G4-8.3: linear warmup steps, then constant (WSD without decay)
SEQ_T = 4096                               # G4-8.2: rows of max_target_length (G4-3's SEQ3)
EVAL_ROUNDS_T = 6                          # G4-8.3: an eval every ceil(steps / 6) steps
SFT_EVALS_T = ("books", "btval", "own")    # G4-2.5's held-out files
RUN_T = re.compile(r"^g4arm_G4-8[a-z]?$")  # the full run (a restart takes a new letter)
SMOKE_RUN_T = re.compile(r"^g4arm_G4-8_s8[a-z]?$")
SMOKE_STEPS_T, SMOKE_SAVE_T, SMOKE_EVERY_T = 8, 4, 4
SMOKE_EVAL_STEPS_T = [0, 4, 8]
SMOKE_LONGEST_T, SMOKE_RANDOM_T = 256, 512  # as G4-3.5: the longest of pass 1 first, then others
OWN_MAX_LOSS_T = 1.0                       # G4-2.4: the base model's loss on its own greedy output at step 0
CAND_T = re.compile(r"^(G4-[89])-s([1-9][0-9]*)$")
RULES_T = ("R-oud", "R-nieuw")
DIV_FLAG_T = 4                             # G4-8.6: a median below 4 distinct samples of 8 is a flag
BOOT_NAME_T = "{arm}/b"                    # g4bon.paired_ci's seed: en-nl-g4arm/0/G4-bon/boot/<arm>/b
# §22 (G4-9, a draft): §19's training on other targets; the arm names its seed (en-nl-g4arm/0/<arm>), runs, candidates
# and places. compare: the earlier arm's choice beside it in the same classify run (G4-9.8) and against it on the
# decision part (G4-9.11, reported)
PROTOCOL_9 = "protocol-g4.md §22"
ARMS_T = {"G4-8": {"protocol": PROTOCOL_T, "targets": "G4-8-data", "report": "G4-8-data (judge-of-anker)",
                   "compare": None},
          "G4-9": {"protocol": PROTOCOL_9, "targets": "de N1-keuze van G4-8-data (§21)",
                   "report": "de N1-keuze van G4-8-data (natuurlijkheidsprompt N1, QE ≤ QE₀ als bewaker)",
                   "compare": "G4-8-s140"}}
META_FIELDS_T = (("id", "string"), ("kind", "string"), ("source", "string"), ("prompt", "string"), ("en", "string"),
                 ("tokens_prompt", "int64"), ("tokens_target", "int64"), ("word_ratio", "float64"),
                 ("char_ratio", "float64"), ("chrf", "float64"), ("dump", "string"), ("long", "bool_"),
                 ("epoch", "int64"))       # g4arm.META_SCHEMA3_FIELDS, what g4arm.py check --arm G4-8 reads
SEGMENT_FIELDS = (("id", "string"), ("text", "string"), ("tokens", "int64"))


def schedule_t(steps):
    """(E, [K1, K2]) of G4-8.3/G4-8.4: an eval every E = ceil(steps / 6) steps, the candidates after 2E and 4E (eval
    steps, so their SFT losses are theirs) and the last step"""
    e = math.ceil(steps / EVAL_ROUNDS_T)
    save = [2 * e, 4 * e]
    if save[-1] >= steps:
        raise SystemExit(f"{steps} steps: too few for G4-8.4's candidates (an eval every {e}, saved after {save})")
    return e, save


def seed_t(arm):
    return f"en-nl-g4arm/0/{arm}"


def arm_of(args):
    """The arm of a training command (--arm; G4-8 unless given)"""
    arm = getattr(args, "arm", None) or ARM8
    if arm not in ARMS_T:
        raise SystemExit(f"--arm {arm}: one of {list(ARMS_T)}")
    return arm


def candidates_t(manifest, arm=ARM8):
    """The candidates of G4-8.4 (G4-9.5) in the order of their steps"""
    return [f"{arm}-s{k}" for k in [*manifest["save_steps"], manifest["steps"]]]


def cand_step_t(name, arm=ARM8):
    m = CAND_T.match(name)
    if not m or m.group(1) != arm:
        raise SystemExit(f"{name} is no candidate of {arm} ({arm}-s<step>)")
    return int(m.group(2))


def n1targets(args):
    """§22 G4-9.2: the N1 choice of §21 (select-n1 per chunk) as one targets file in §18's form (G4-8-data.9), checked
    against the plan, against §21's report (each chunk's choice by sha256, the counts per part) and QE <= QE0 again;
    nothing is written if a check fails"""
    plan = {r["id"]: r for r in read_rows(args.plan, ["id", "part", "chunk", "en", "hypothesis0"])}
    rep = json.loads(Path(args.n1report).read_text())
    chunks = [c for c, _ in CHUNKS8]
    if rep.get("protocol") != PROTOCOL_N1 or rep.get("chunks") != chunks:
        raise SystemExit(f"{args.n1report} is not {PROTOCOL_N1}'s report over {chunks}")
    dirs = dict(spec.split("=", 1) for spec in args.chunk_dir)
    if sorted(dirs) != chunks:
        raise SystemExit(f"--chunk-dir {sorted(dirs)}: one per chunk {chunks}")
    targets, inputs = [], {"plan": g4rl.file_sha256(args.plan), "n1report": g4rl.file_sha256(args.n1report)}
    for c in chunks:
        d = Path(dirs[c])
        rows, sha = n1_chunk(c, d / "choice.parquet", d / "targets.parquet", plan, rep)
        targets += rows
        inputs[c] = {"choice": sha, "targets": g4rl.file_sha256(d / "targets.parquet")}
    counts = {p: sum(t["part"] == p for t in targets) for p in PARTS8}
    want = {p: rep["parts"][p]["n1_kept"] for p in PARTS8}
    if counts != want or len(targets) != rep["parts"]["samen"]["n1_kept"]:
        raise SystemExit(f"targets per part {counts}, §21's report {want} (together {rep['parts']['samen']['n1_kept']})")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_parquet(sorted(targets, key=lambda r: r["id"]), out / "targets.parquet")
    info = {"protocol": PROTOCOL_9, "arm": "G4-9", "source": f"{PROTOCOL_N1}: select-n1 of {chunks}",
            "targets": len(targets), "parts": counts, "inputs": inputs,
            "targets_sha256": g4rl.file_sha256(out / "targets.parquet")}
    (out / "targets.json").write_text(json.dumps(info, indent=1) + "\n")
    print(f"N1TARGETS {json.dumps({'targets': len(targets), 'parts': counts})}", flush=True)


def n1_chunk(c, choice_path, targets_path, plan, rep):
    """G4-9.2's checks of one chunk's N1 choice (select-n1's choice.parquet and targets.parquet) against the plan and
    §21's report: (its target rows, the sha256 of its choice.parquet); a failed check stops"""
    sha = g4rl.file_sha256(choice_path)
    if sha != rep["inputs"][c]["choice-n1.parquet"]:
        raise SystemExit(f"{choice_path} is not the choice-n1.parquet that §21's n1report checked ({c})")
    choice = {r["id"]: r for r in read_rows(choice_path)}
    if set(choice) != {i for i, r in plan.items() if r["chunk"] == c}:
        raise SystemExit(f"{choice_path}: not the sources of chunk {c} in the plan")
    rows = read_rows(targets_path)
    ids = [t["id"] for t in rows]
    kept = {i for i, r in choice.items() if r["reason"] == "sample"}
    if len(ids) != len(set(ids)) or set(ids) != kept:
        raise SystemExit(f"{targets_path}: {len(ids)} rows, not one per source of choice.parquet with a "
                         f"sample ({len(kept)})")
    for t in rows:
        i, ch, src = t["id"], choice[t["id"]], plan[t["id"]]
        if (t["part"], t["chunk"], t["reason"]) != (src["part"], c, "sample"):
            raise SystemExit(f"{c} {i}: part, chunk or reason {(t['part'], t['chunk'], t['reason'])} is not the "
                             f"plan's {(src['part'], c)} with a sample")
        if (t["j"], t["p_sample"], t["qe"], t["qe0"]) != (ch["j"], ch["p_sample"], ch["qe"], ch["qe0"]):
            raise SystemExit(f"{c} {i}: the target's sample, P or QE is not choice.parquet's")
        if t["p_sample"] is None or not t["p_sample"] > ANCHOR_P:
            raise SystemExit(f"{c} {i}: P {t['p_sample']!r}, not above {ANCHOR_P}")
        if t["qe"] is None or t["qe0"] is None or t["qe"] > t["qe0"] + QE_TOL:
            raise SystemExit(f"{c} {i}: QE {t['qe']!r} above QE0 {t['qe0']!r} (the guard of G4-9.2)")
        if (t["en"] or "").strip() != (src["en"] or "").strip():
            raise SystemExit(f"{c} {i}: the English is not the plan's")
        if not (t["target"] or "").strip() or norm(t["target"]) == norm(src["hypothesis0"] or ""):
            raise SystemExit(f"{c} {i}: the target is empty or the anchor's text")
    return rows, sha


def checked_targets(targets, plan):
    """§19 G4-8.2: every target as §18.6 made it: one per source id of the plan, its part and English the plan's,
    reason 'sample' with P̄ > 0.5, a text that is not empty and not the anchor's (whitespace normalized)"""
    by = {r["id"]: r for r in plan}
    seen = set()
    for t in targets:
        i = t["id"]
        src = by.get(i)
        if i in seen:
            raise SystemExit(f"targets: {i} twice")
        seen.add(i)
        if src is None:
            raise SystemExit(f"targets: {i} is no source of the plan")
        if t["part"] not in PARTS8 or t["part"] != src["part"]:
            raise SystemExit(f"targets: {i} has part {t['part']!r}, the plan {src['part']!r}")
        if t["reason"] != "sample" or t["p_sample"] is None or not t["p_sample"] > ANCHOR_P:
            raise SystemExit(f"targets: {i} is no janchor target (reason {t['reason']!r}, P {t['p_sample']!r})")
        if (t["en"] or "").strip() != (src["en"] or "").strip():
            raise SystemExit(f"targets: {i}'s English is not the plan's")
        if not (t["target"] or "").strip() or norm(t["target"]) == norm(src["hypothesis0"] or ""):
            raise SystemExit(f"targets: {i}'s target is empty or the anchor's text")
    return {p: sum(t["part"] == p for t in targets) for p in PARTS8}


def leak_check(targets, blocked, exact, rlval, sft_users):
    """§19 G4-8.2: per kind the ids of the targets whose source meets the evaluation: a 13-gram of `blocked` (the
    evaluation sets' sources and references, the blind passages, FLAN's evaluation rows), an English text of `exact`,
    an id, English or document of G4-7's RL-val (rlval: (ids, texts, docs)), a user turn of the SFT evals (sft_users,
    whitespace normalized)"""
    import g4lora
    import select_passages as sp

    ids, texts, docs = rlval
    hits = {k: [] for k in ("ngram", "eval_text", "rlval", "rlval_doc", "sft_eval")}
    for t in targets:
        en = (t["en"] or "").strip()
        if sp.ngrams(en) & blocked:
            hits["ngram"].append(t["id"])
        if en in exact:
            hits["eval_text"].append(t["id"])
        if t["id"] in ids or en in texts:
            hits["rlval"].append(t["id"])
        if t.get("doc") and t["doc"] in docs:
            hits["rlval_doc"].append(t["id"])
        if norm(g4lora.user_turn(en)) in sft_users:
            hits["sft_eval"].append(t["id"])
    return hits


def sft_example(t, tokens_prompt, tokens_target, epoch):
    """(messages row, meta row) of one target in the format of G4-1...G4-5 (g4arm.example)"""
    import g4lora
    import sft_build

    en, nl_text = t["en"].strip(), t["target"].strip()
    return ({"messages": [{"role": "user", "content": g4lora.user_turn(en)}, {"role": "assistant", "content": nl_text}]},
            {"id": t["id"], "kind": t["part"], "source": t["stratum"] or t["part"], "prompt": g4lora.PROMPT, "en": en,
             "tokens_prompt": tokens_prompt, "tokens_target": tokens_target,
             "word_ratio": round(sft_build.word_ratio(nl_text, en), 4),
             "char_ratio": round(sft_build.char_ratio(nl_text, en), 4), "chrf": t["chrf"], "dump": None, "long": False,
             "epoch": epoch})


def pack_t(lengths):
    import g4arm

    return g4arm.pack_batches(lengths, seq=SEQ_T)


def prefix_for_steps(lengths, steps):
    """The fewest first examples whose packing gives the multi-host job at least `steps` steps (g4arm.shortest_prefix
    at SEQ_T)"""
    import g4arm

    if g4arm.steps_of(pack_t(lengths)) < steps:
        raise SystemExit(f"{len(lengths)} examples give {g4arm.steps_of(pack_t(lengths))} steps, {steps} wanted")
    lo, hi = 1, len(lengths)
    while lo < hi:
        mid = (lo + hi) // 2
        if g4arm.steps_of(pack_t(lengths[:mid])) >= steps:
            hi = mid
        else:
            lo = mid + 1
    return lo


def write_set(rows, d, extra):
    """train.parquet, train-meta.parquet, template.jinja and manifest.json of a set of (messages, meta) rows"""
    import g4arm
    import g4lora

    d.mkdir(parents=True, exist_ok=True)
    lengths = [m["tokens_prompt"] + m["tokens_target"] for _, m in rows]
    if max(lengths) > SEQ_T:
        raise SystemExit(f"an example of {max(lengths)} tokens, more than {SEQ_T}")
    batches = pack_t(lengths)
    g4rl.write_parquet([x for x, _ in rows], d / "train.parquet")
    write_parquet([m for _, m in rows], d / "train-meta.parquet", META_FIELDS_T)
    (d / "template.jinja").write_text(g4lora.TEMPLATE, encoding="utf-8")
    manifest = {"examples": len(rows), "tokens": sum(lengths), "target_tokens": sum(m["tokens_target"] for _, m in rows),
                "max_tokens": max(lengths), "seq": SEQ_T, "batches_per_process": batches,
                "steps": g4arm.steps_of(batches), "fill": round(sum(lengths) / (sum(batches) * g4arm.ROWS_PER_HOST * SEQ_T), 4),
                "prompt": g4lora.PROMPT, "template": "g4lora.TEMPLATE (Gemma 4, thinking off)", **extra}
    (d / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


def sftset(args, counter=None):
    """§19 G4-8.2: the SFT set of the targets (two passes, each shuffled), checked against the plan and the evaluation;
    the smoke set (G4-8.5); RL-val's segments per worker (G4-8.6); manifest, report"""
    import g4arm
    import sft_build

    arm = arm_of(args)
    seed, protocol = seed_t(arm), ARMS_T[arm]["protocol"]
    counter = counter or g4arm.TokenCounter(args.tokenizer)
    out = Path(args.out)
    plan = read_rows(args.plan, columns=["id", "part", "en", "hypothesis0"])
    targets = sorted(read_rows(args.targets), key=lambda r: r["id"])
    per_part = checked_targets(targets, plan)
    evalsets = read_rows(args.evalsets)
    blind = g4arm.read_jsonl(args.blind)
    blocked = g4arm.checked_blocked(args.evalsets, args.blind)
    blocked |= sft_build.flan_eval_ngrams(read_rows(args.flan_eval, columns=["inputs", "targets"]))
    exact = {(r["source"] or "").strip() for r in evalsets} | {(b["source"] or "").strip() for b in blind}
    val = [r for r in read_rows(args.g47_pool, columns=["id", "en", "doc", "role"]) if r["role"] == "rlval"]
    if not val:
        raise SystemExit(f"{args.g47_pool}: no RL-val row")
    rlval = ({r["id"] for r in val}, {(r["en"] or "").strip() for r in val}, {r["doc"] for r in val if r["doc"]})
    sft_users, sft_files = set(), {}
    for spec in args.sft_eval:
        name, path = spec.split("=", 1)
        rows = read_rows(path, columns=["messages"])
        sft_users |= {norm(x["messages"][0]["content"]) for x in rows}
        sft_files[name] = {"rows": len(rows), "sha256": g4rl.file_sha256(path)}
    if sorted(sft_files) != sorted(SFT_EVALS_T):
        raise SystemExit(f"--sft-eval {sorted(sft_files)}: G4-8.2 checks {list(SFT_EVALS_T)}")
    hits = leak_check(targets, blocked, exact, rlval, sft_users)
    if any(hits.values()):
        raise SystemExit("sftset: sources that meet the evaluation "
                         f"{ {k: len(v) for k, v in hits.items()} }, e.g. { {k: v[:3] for k, v in hits.items() if v} }")
    prompts, tgts = counter.examples([{"en": t["en"].strip(), "nl": t["target"]} for t in targets])
    passes = []
    for e in range(1, EPOCHS_T + 1):
        one = [sft_example(t, a, b, e) for t, a, b in zip(targets, prompts, tgts)]
        random.Random(f"{seed}/train/shuffle/{e}").shuffle(one)
        passes.append(one)
    rows = [p for one in passes for p in one]
    batches = pack_t([m["tokens_prompt"] + m["tokens_target"] for _, m in rows])
    steps = g4arm.steps_of(batches)
    every, save = schedule_t(steps)
    inputs = {"targets": g4rl.file_sha256(args.targets), "plan": g4rl.file_sha256(args.plan),
              "evalsets": g4rl.file_sha256(args.evalsets), "blind": g4rl.file_sha256(args.blind),
              "flan_eval": g4rl.file_sha256(args.flan_eval), "g47_pool": g4rl.file_sha256(args.g47_pool),
              "g47_plan": g4rl.file_sha256(args.g47_plan), "sft_eval": sft_files}
    kinds = {p: {"examples": n, "tokens": sum(a + b for t, a, b in zip(targets, prompts, tgts) if t["part"] == p)}
             for p, n in per_part.items()}
    manifest = write_set(rows, out, {
        "set": arm, "protocol": protocol, "seed": seed, "epochs": EPOCHS_T, "distinct": len(targets),
        "examples_per_epoch": [len(one) for one in passes], "parts": kinds, "eval_every": every, "save_steps": save,
        "lr": LR_T, "warmup_steps": WARMUP_T, "sft_evals": list(SFT_EVALS_T),
        "leak_check": {k: len(v) for k, v in hits.items()}, "inputs": inputs})
    manifest["candidates"] = candidates_t(manifest, arm)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    # G4-8.5: pass 1's longest first, then the others in a seeded order, as few as give the smoke run its 8 steps
    first = passes[0]
    longest = sorted(first, key=lambda p: (-(p[1]["tokens_prompt"] + p[1]["tokens_target"]), p[1]["id"]))
    longest = longest[:SMOKE_LONGEST_T]
    taken = {m["id"] for _, m in longest}
    others = sorted((p for p in first if p[1]["id"] not in taken), key=lambda p: p[1]["id"])
    random.Random(f"{seed}/smoke").shuffle(others)
    smoke_all = longest + others
    n = max(prefix_for_steps([m["tokens_prompt"] + m["tokens_target"] for _, m in smoke_all], SMOKE_STEPS_T),
            min(len(smoke_all), SMOKE_LONGEST_T + SMOKE_RANDOM_T))
    smoke = write_set(smoke_all[:n], out / "smoke", {"set": f"{arm}-smoke", "protocol": protocol,
                                                     "longest": len(longest), "seed": f"{seed}/smoke"})
    # G4-8.6: RL-val of G4-7 (iteration -1 of its plan) per worker, for the samples of every model
    rl = [r for r in read_rows(args.g47_plan, columns=["id", "en", "tokens", "iteration"]) if r["iteration"] == -1]
    if not rl or len({r["id"] for r in rl}) != len(rl):
        raise SystemExit(f"{args.g47_plan}: no RL-val (iteration -1), or an id twice")
    for k, part in g4rl.assign_workers7(rl, WORKERS).items():
        write_parquet([{"id": r["id"], "text": r["en"], "tokens": r["tokens"]} for r in part],
                      out / "rlval" / f"segments-w{k}.parquet", SEGMENT_FIELDS)
    report = {"protocol": protocol, "manifest": manifest, "smoke": smoke, "rlval": len(rl),
              "targets": {p: {"examples": n_, "p_mean": mean(t["p_sample"] for t in targets if t["part"] == p),
                              "dqe_median": median(t["qe"] - t["qe0"] for t in targets if t["part"] == p),
                              "chrf_median": median(t["chrf"] for t in targets if t["part"] == p),
                              "ratio_median": median(t["ratio"] for t in targets if t["part"] == p)}
                          for p, n_ in per_part.items()}}
    (out / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    (out / "report.md").write_text(sftset_text(report), encoding="utf-8")
    print(f"SFTSET8 {json.dumps({'targets': len(targets), 'parts': per_part, 'tokens': manifest['tokens'], 'steps': steps, 'eval_every': every, 'candidates': manifest['candidates'], 'smoke_examples': smoke['examples'], 'smoke_steps': smoke['steps'], 'rlval': len(rl)})}",
          flush=True)


def n0(x):
    return "–" if x is None else f"{x:,}".replace(",", ".")


def sftset_text(r):
    m, s = r["manifest"], r["smoke"]
    lines = [f"# {m['set']}: de trainingsset ({r['protocol']})", "",
             f"Gemaakt door `g4arm8.py sftset`; alleen tellingen. Elk doel van {ARMS_T[m['set']]['targets']} één "
             "voorbeeld (prompt "
             f"`{m['prompt']}`), {m['epochs']} doorgangen, elk apart geschud.", "",
             "| deel | doelen | tokens (prompt + doel, één doorgang) | P̄ (gem.) | ΔQE (med.) | chrF doel–anker (med.) | "
             "ℓ/ℓ₀ (med.) |", "|---|---:|---:|---:|---:|---:|---:|"]
    for p, x in r["targets"].items():
        lines.append(f"| {p} | {n0(x['examples'])} | {n0(m['parts'][p]['tokens'])} | {nl(x['p_mean'])} | "
                     f"{nl(x['dqe_median'])} | {nl(x['chrf_median'], 1)} | {nl(x['ratio_median'])} |")
    lines += ["", f"Samen {n0(m['examples'])} rijen, {n0(m['tokens'])} tokens, langste voorbeeld {n0(m['max_tokens'])}; "
              f"batches per proces {m['batches_per_process']} → **{m['steps']} stappen** (vulling {nl(m['fill'], 3)}); "
              f"evaluatie elke {m['eval_every']} stappen; kandidaten {', '.join(m['candidates'])}.", "",
              f"Lekcontrole (G4-8.2): {json.dumps(m['leak_check'])} (alles 0).", "",
              f"Smoke-set: {s['examples']} voorbeelden ({s['longest']} langste eerst), {s['steps']} stappen mogelijk. "
              f"RL-val: {r['rlval']} bronnen.", ""]
    return "\n".join(lines)


def adapter_check(meta, step):
    """rank 32 / alpha 16 / scale 0.5 and the adapter of that step"""
    return {"value": {k: meta.get(k) for k in ("rank", "alpha", "scale", "steps")},
            "holds": (meta.get("rank"), meta.get("alpha"), meta.get("scale"), meta.get("steps")) == (32, 16.0, 0.5, step)}


def run_checks_t(summary, manifest, steps, eval_steps):
    """The checks of a training run of §19 (G4-8.3, G4-8.5): its steps, finite losses, the peak, the sequence length,
    the pipeline's batches per process against sftset's count, the LR's peak, every SFT eval finite at eval_steps"""
    import g4arm

    checks = {}
    loss = summary.get("loss") or []
    checks["steps"] = {"value": summary.get("steps_done"), "holds": summary.get("steps_done") == steps}
    checks["loss_finite"] = {"value": len(loss), "holds": len(loss) == steps and all(math.isfinite(v) for _, v in loss)}
    checks["peak_gib"] = {"value": summary.get("peak_gib_max"), "holds": summary.get("peak_gib_max") is not None
                          and summary["peak_gib_max"] <= g4arm.MEM_MAX_GIB}
    seq = (summary.get("config") or {}).get("max_target_length")
    checks["seq"] = {"value": seq, "holds": seq == SEQ_T}
    got = (summary.get("epoch") or {}).get("batches_per_process")
    checks["batches_as_sftset"] = {"value": {"pipeline": got, "sftset": manifest["batches_per_process"]},
                                   "holds": got == manifest["batches_per_process"]}
    lrs = list((summary.get("lr_at") or {}).values())
    # the LR is logged as float32: 5e-5 reads back as 4.999999873689376e-05, so compare within float32 precision
    checks["lr_peak"] = {"value": max(lrs) if lrs else None,
                         "holds": bool(lrs) and abs(max(lrs) - LR_T) <= 1e-6 * LR_T}
    evals = summary.get("eval") or []
    for name in SFT_EVALS_T:
        rows = {e["step"]: e for e in evals if e["set"] == name}
        checks[f"eval_{name}"] = {"value": {k: round(rows[k]["loss"], 4) for k in eval_steps if k in rows},
                                  "holds": all(k in rows and math.isfinite(rows[k]["loss"]) and rows[k]["tokens"] >= 1
                                               for k in eval_steps)}
    return checks


def verdict_t(checks, what):
    failed = [k for k, v in checks.items() if not v["holds"]]
    return {"checks": checks, "passed": not failed, "failed": failed, "what": what}


def smoke8(args):
    """G4-8.5: the smoke run's checks and the estimate of the full run (g4arm.training_time); exit 1 if a check fails"""
    import g4arm

    s = json.loads(Path(args.summary).read_text())
    sm, full = json.loads(Path(args.smoke_manifest).read_text()), json.loads(Path(args.manifest).read_text())
    checks = run_checks_t(s, sm, SMOKE_STEPS_T, SMOKE_EVAL_STEPS_T)
    checks["adapter"] = adapter_check(json.loads(Path(args.adapter).read_text()), SMOKE_STEPS_T)
    checks["adapter_step4"] = adapter_check(json.loads(Path(args.adapter_step).read_text()), SMOKE_SAVE_T)
    own0 = next((e["loss"] for e in s.get("eval") or [] if e["set"] == "own" and e["step"] == 0), None)
    checks["own_step0"] = {"value": own0, "holds": own0 is not None and own0 <= OWN_MAX_LOSS_T}
    out = verdict_t(checks, f"the smoke run {args.run}")
    out.update({"protocol": ARMS_T[arm_of(args)]["protocol"], "run": args.run,
                "estimate": g4arm.training_time(s, full["steps"], full["tokens"], sm["tokens"], every=full["eval_every"])})
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    print(f"SMOKE8 {json.dumps({k: out[k] for k in ('run', 'passed', 'failed', 'estimate')})}", flush=True)
    if not out["passed"]:
        raise SystemExit(f"SMOKE8 failed: {out['failed']} (G4-8.5: the chain stops before the training)")


def trained(args):
    """G4-8.3/G4-8.4: the full run against sftset's manifest (its steps, the batches, finite losses, the LR, the SFT
    evals at 0, at each kept step and at the last), each candidate's adapter of its step; exit 1 if a check fails"""
    arm = arm_of(args)
    s = json.loads(Path(args.summary).read_text())
    m = json.loads(Path(args.manifest).read_text())
    steps = m["steps"]
    checks = run_checks_t(s, m, steps, [0, *m["save_steps"], steps])
    lr = s.get("lr_at") or {}
    checks["lr_constant"] = {"value": {k: lr.get(str(k)) for k in (WARMUP_T, steps - 1)},
                             "holds": all(lr.get(str(k)) is not None and abs(lr[str(k)] - LR_T) <= 1e-6 * LR_T  # float32, as lr_peak
                                          for k in (WARMUP_T, steps - 1))}
    metas = dict(x.split("=", 1) for x in args.adapter)
    if sorted(metas) != sorted(candidates_t(m, arm)):
        raise SystemExit(f"--adapter {sorted(metas)}: the candidates are {candidates_t(m, arm)}")
    for name in candidates_t(m, arm):
        checks[f"adapter_{name}"] = adapter_check(json.loads(Path(metas[name]).read_text()), cand_step_t(name, arm))
    out = verdict_t(checks, f"the run {args.run}")
    out.update({"protocol": ARMS_T[arm]["protocol"], "run": args.run, "candidates": candidates_t(m, arm)})
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    print(f"TRAINED8 {json.dumps({k: out[k] for k in ('run', 'passed', 'failed', 'candidates')})}", flush=True)
    if not out["passed"]:
        raise SystemExit(f"TRAINED8 failed: {out['failed']}")


# ---------------------------------------------------------------------------------------------------- report8

def lora_from_logs(paths):
    """translate.py's LORA line of one candidate (G4-8.6: the merge's update norm) from its translate logs: every LORA
    line in them the same apart from the local directory"""
    found = set()
    for path in paths:
        for line in Path(path).read_text(errors="replace").splitlines():
            if line.startswith("LORA "):
                found.add(json.dumps({k: v for k, v in json.loads(line[5:]).items() if k != "dir"}, sort_keys=True))
    if len(found) != 1:
        raise SystemExit(f"{len(paths)} translate logs: {len(found)} different LORA lines, not one")
    return json.loads(found.pop())


def loraline(args):
    paths = sorted(str(p) for p in Path(args.logs).parent.glob(Path(args.logs).name))
    x = lora_from_logs(paths)
    Path(args.out).write_text(json.dumps(x) + "\n")
    print(f"LORA8 {json.dumps(x)}", flush=True)


def div_measures(samples, sources):
    """G4-8.6 on RL-val: per group of 8 samples at T = 1 the distinct texts (whitespace normalized) and the fails of
    G4-7 that need no QE (not ended, loop, empty, English left, l / l0 outside the band against G4-7's anchor)"""
    grouped = groups(samples, sources)
    distinct, fails, n = [], 0, 0
    for i, rs in grouped.items():
        distinct.append(len({norm(r["hypothesis"] or "") for r in rs}))
        for r in rs:
            n += 1
            ratio = r["tokens_out"] / sources[i]["l0"]
            fails += bool(not r["ended"] or r["decode_loop"] or r["empty"] or r["english_left"]
                          or not g4rl.LEN_BAND[0] <= ratio <= g4rl.LEN_BAND[1])
    med = statistics.median(distinct)
    return {"sources": len(grouped), "samples": n, "distinct_median": med, "distinct_mean": mean(distinct),
            "all_equal": share(sum(d == 1 for d in distinct), len(distinct)), "fail_share": share(fails, n),
            "flag": med < DIV_FLAG_T}


def rule_variants(r):
    """G4-8.9: the verdict of both variants (R-oud: (b) decides; R-nieuw: (c) after Yeb's doubt cases with the C1 guard)"""
    out = {}
    for rule in RULES_T:
        gate = r["b"]["holds"] if rule == "R-oud" else r["c1_guard"]["holds"]
        if not (r["a1"]["holds"] and r["a3"]["holds"] and gate):
            out[rule] = "helpt niet"
        elif r["blind"] is None:
            out[rule] = "wacht op de blinde beoordeling ((a2), (c))"
        elif rule == "R-nieuw" and r["blind"]["doubt"]:
            out[rule] = "wacht op Yebs oordeel over de twijfelgevallen ((c) van R-nieuw)"
        else:
            out[rule] = "helpt" if r["blind"]["a2"] and r["blind"]["c"] else "helpt niet"
    return out


def compare_t(scored, part, ids, c1, c2_part, chosen, ref, arm):
    """§22 G4-9.11 (reported, decides nothing): the chosen candidate against an earlier arm's choice on one part: per
    (a1) set the mean QE of both and the paired difference on the segments both scored with its 95% interval, (a1) of
    both; COMET-22 per set and (a3) against it; C1 "original" on WMT24++ (good) + FLAN pooled of both with the
    difference and its interval; C2 of both"""
    import g4arm
    import g4bon

    m, o = part["models"][chosen], part["models"][ref]
    boot = f"{arm}/vs-{ref.rsplit('-s', 1)[0]}"
    qe = {}
    for s in g4arm.HUMAN_QE:
        diffs = [scored[chosen][i]["metricx_qe"] - scored[ref][i]["metricx_qe"] for i in ids[s]
                 if scored[chosen][i]["metricx_qe"] is not None and scored[ref][i]["metricx_qe"] is not None]
        if not diffs:
            raise SystemExit(f"{s}: no segment of the part with a QE of both {chosen} and {ref}")
        qe[s] = {"model": m["qe"][s], "ref": o["qe"][s], "ci": g4bon.paired_ci({s: diffs}, f"{boot}/qe/{s}")}
    c1_ci = g4bon.paired_ci({s: [(c1[(chosen, s, i)] >= 0.5) - (c1[(ref, s, i)] >= 0.5) for i in ids[s]]
                             for s in g4arm.B_SETS7}, f"{boot}/c1")
    if abs(c1_ci["delta"] - (m["share"] - o["share"])) > 1e-9:
        raise SystemExit(f"the comparison with {ref}: the bootstrap's point {c1_ci['delta']} is not the shares' "
                         f"{m['share'] - o['share']}")
    return {"ref": ref, "qe": qe, "a1": {"model": m["a1"], "ref": o["a1"]},
            "a3": g4arm.rule_a3(scored, ids, chosen, ref),
            "c1": {"model": m["share"], "ref": o["share"], "ci": c1_ci},
            "c2": {"model": c2_part[chosen]["c2"]["pooled"], "ref": c2_part[ref]["c2"]["pooled"],
                   "delta": c2_part[chosen]["c2"]["pooled"] - c2_part[ref]["c2"]["pooled"]}}


def report8(args):
    """§19 G4-8.7-G4-8.10: the candidate chosen on the search part (select5), (a1), (a3), (b) with its interval and the
    C1 guard against G4-0 on the decision part, both variants of the rule (O6), C2, PLL and the other measures per
    candidate on both parts, per candidate the update norm, the SFT losses at its step and the samples on RL-val; with
    --arm G4-9 (§22) the same, and the comparison with G4-8's choice (G4-9.11, --compare)"""
    import g4arm
    import g4bon
    import g4mix

    arm = arm_of(args)
    compare = getattr(args, "compare", None)
    if compare != ARMS_T[arm]["compare"]:
        raise SystemExit(f"--compare {compare}: {arm} compares with {ARMS_T[arm]['compare']}")
    evalsets = {r["id"]: r for r in g4arm.read_rows(args.evalsets)}
    split = json.loads(Path(args.split).read_text())
    assign = split["assign"]
    if g4mix.digest(assign) != split["digest"] or split["digest"] != g4arm.SPLIT_DIGEST6C:
        raise SystemExit(f"{args.split}: not the split of G4-5.4 (digest {split['digest']})")
    manifest = json.loads(Path(args.manifest).read_text())
    cands = [c.split("=", 1) for c in args.candidate]
    names = [n for n, _ in cands]
    if names != candidates_t(manifest, arm):
        raise SystemExit(f"the candidates {names}: G4-8.4 takes {candidates_t(manifest, arm)} in this order")
    refs = [c.split("=", 1) for c in args.reference or []]
    if compare is not None and compare not in [n for n, _ in refs]:
        raise SystemExit(f"--compare {compare}: not among the references {[n for n, _ in refs]}")
    base = g4arm.BASE
    scored = {name: g4arm.scored_dir(d) for name, d in cands + [(base, args.base)] + refs}
    for name, rows in scored.items():
        if set(rows) != set(evalsets):
            raise SystemExit(f"{name}: scores for {len(rows)} of {len(evalsets)} segments")
    refqe = {r["id"]: r["qe_source_reference"] for r in g4arm.read_rows(args.refqe)}
    check = {}
    for s, want in g4arm.HUMAN_QE.items():
        ids = [i for i, r in evalsets.items() if r["set"] == s and not (s == "wmt24pp" and g4arm.bad_source(r["meta"]))]
        if any(refqe.get(i) is None for i in ids):
            raise SystemExit(f"{args.refqe}: no QE of the human reference of every {s} segment")
        got = g4arm.mean(refqe[i] for i in ids)
        check[s] = {"again": got, "protocol": want, "diff": got - want}
        if abs(got - want) > g4arm.REFQE_MAX_DIFF5:
            raise SystemExit(f"the human references' QE of {s} again: {got:.4f}, the protocol's {want:.4f}")
    predictions, predictions_c2 = g4arm.read_rows(args.predictions), g4arm.read_rows(args.predictions_c2)
    keys = [(r["model"], r["set"], r["id"]) for r in predictions]
    keys_c2 = [(r["model"], r["set"], r["id"]) for r in predictions_c2]
    if len(set(keys_c2)) != len(keys_c2) or set(keys_c2) != set(keys):
        raise SystemExit(f"{args.predictions_c2}: not C2 on the texts of {args.predictions} (each once)")
    part_ids = {p: g4arm.part_ids5(evalsets, assign, p) for p in ("search", "decision")}
    parts = {p: g4arm.part_measures5(scored, predictions, refqe, part_ids[p], base) for p in part_ids}
    chosen, why = g4arm.select5(names, parts["search"])
    d = parts["decision"]["models"][chosen]
    a3 = {p: {name: g4arm.rule_a3(scored, part_ids[p], name, base) for name in scored} for p in parts}
    c1 = {k: r["p_original"] for k, r in zip(keys, predictions)}
    c2 = {k: r["p_original"] for k, r in zip(keys_c2, predictions_c2)}
    dec = part_ids["decision"]
    b_ci = g4bon.paired_ci({s: [(c1[(chosen, s, i)] >= 0.5) - (c1[(base, s, i)] >= 0.5) for i in dec[s]]
                            for s in g4arm.B_SETS7}, BOOT_NAME_T.format(arm=arm))
    if abs(b_ci["delta"] - d["b_delta"]) > 1e-9:
        raise SystemExit(f"(b): the bootstrap's point {b_ci['delta']} is not part_measures5's {d['b_delta']}")
    summary = json.loads(Path(args.summary).read_text())
    lora = {}
    for spec in args.lora:
        name, path = spec.split("=", 1)
        lora[name] = json.loads(Path(path).read_text())
    rl = {r["id"]: r for r in g4arm.read_rows(args.rlval) if r["iteration"] == -1}
    div = {}
    for spec in args.div:
        name, root = spec.split("=", 1)
        div[name] = div_measures(read_samples8(root), rl)
    if sorted(lora) != sorted(names) or sorted(div) != sorted([*names, base]):
        raise SystemExit(f"--lora {sorted(lora)} and --div {sorted(div)}: every candidate (and G4-0's samples)")
    step0 = g4arm.eval_at(summary, 0)
    training = {}
    for name in names:
        k = cand_step_t(name, arm)
        at = g4arm.eval_at(summary, k)
        if sorted(at) != sorted(SFT_EVALS_T):
            raise SystemExit(f"the run's summary has no SFT eval of every set at step {k}: {sorted(at)}")
        training[name] = {"step": k, "update_norm": lora[name]["relative_update_norm"],
                          "elements_changed": lora[name].get("elements_changed"), "modules": lora[name]["modules"],
                          "sft": at, "sft_delta": {s: at[s] - step0[s] for s in SFT_EVALS_T}, "div": div[name]}
    truncated = {label: {name: sum(bool(r.get("truncated")) for r in rows if r["model"] == name) for name in scored}
                 for label, rows in (("c1", predictions), ("c2", predictions_c2))}
    result = {"protocol": ARMS_T[arm]["protocol"], "arm": arm, "base": base, "o6": args.o6, "candidates": names,
              "references": [n for n, _ in refs], "selected": chosen, "selection": why, "refqe_check": check,
              "parts": parts, "a1": {"holds": d["a1"], "sets": d["a1_sets"]}, "a3": a3["decision"][chosen],
              "a3_all": a3, "b": {"holds": d["b"], "delta": d["b_delta"], "ci": b_ci},
              "c1_guard": {"holds": d["b_delta"] >= -g4arm.TOLERANCE, "delta": d["b_delta"]},
              "c2": {p: g4arm.c2_part7(c1, c2, part_ids[p], list(scored), base) for p in parts},
              "truncated": truncated, "counts": {p: g4arm.part_counts7(scored, part_ids[p]) for p in parts},
              "pll": ({p: g4arm.pll_part7({(r["model"], r["set"], r["id"]): r for r in g4arm.read_rows(args.pll)},
                                          part_ids[p], list(scored)) for p in parts} if args.pll else None),
              "training": training, "div_base": div[base], "sft_step0": step0,
              "run": {"steps": manifest["steps"], "eval_every": manifest["eval_every"], "lr": LR_T,
                      "warmup": WARMUP_T, "tokens": manifest["tokens"], "examples": manifest["examples"],
                      "parts": manifest["parts"], "step_s_median": (summary.get("throughput") or {}).get("step_s_median"),
                      "peak_gib_max": summary.get("peak_gib_max")},
              "smoke": json.loads(Path(args.smoke).read_text()) if args.smoke else None}
    result["compare"] = (compare_t(scored, parts["decision"], dec, c1, result["c2"]["decision"], chosen, compare, arm)
                         if compare else None)
    result["blind_due"] = result["a1"]["holds"] and result["a3"]["holds"]
    result["blind"] = None
    if args.tally:
        t = json.loads(Path(args.tally).read_text())
        if not result["blind_due"]:
            raise SystemExit(f"{args.tally}: a blind tally, but (a1) and (a3) do not both hold for {chosen} (G4-8.9)")
        if t.get("arm") != chosen:
            raise SystemExit(f"{args.tally}: a tally of {t.get('arm')}, the selected candidate is {chosen}")
        result["blind"] = t
    result["verdicts"] = rule_variants(result)
    result["verdict"] = result["verdicts"][args.o6] if args.o6 in RULES_T else "open (O6): zie beide varianten"
    Path(args.out).with_suffix(".json").write_text(json.dumps(result, indent=1, ensure_ascii=False, default=str) + "\n")
    Path(args.out).write_text(report8_text(result), encoding="utf-8")
    print("REPORT8_TRAIN " + json.dumps({"selected": chosen, "a1": d["a1"], "a3": result["a3"]["holds"],
                                          "b": d["b_delta"], "b_holds": d["b"], "c1_guard": result["c1_guard"]["holds"],
                                          "blind_due": result["blind_due"], "o6": args.o6,
                                          "verdicts": result["verdicts"]}, ensure_ascii=False), flush=True)


def f3(x, signed=False):
    if x is None:
        return "–"
    return (f"{x:+.3f}" if signed else f"{x:.3f}").replace(".", ",")


def f4(x, signed=False):
    if x is None:
        return "–"
    return (f"{x:+.4f}" if signed else f"{x:.4f}").replace(".", ",")


def report8_text(r):
    import g4arm

    sets = list(g4arm.HUMAN_QE)
    dec = r["parts"]["decision"]
    sel, base = r["selected"], r["base"]
    m, b0 = dec["models"][sel], dec["models"][base]
    ci = r["b"]["ci"]
    decisive = r["o6"] if r["o6"] in RULES_T else None
    L = [f"# Arm {r['arm']}: uitslag volgens {r['protocol']}", "",
         f"Gemaakt door `g4arm8.py report8`. SFT met LoRA vanaf G4-0 op de doelen van {ARMS_T[r['arm']]['report']}; de "
         "kandidaten vertaald en gescoord als G4-7.7, de keuze op het zoekdeel (`select5`), het besluit tegen G4-0 op "
         "het beslisdeel (G4-8.9). C2, PLL en alles onder \"rapporteren\" beslissen niets.", "",
         f"**Uitslag ({'beslissend: ' + decisive if decisive else 'O6 nog open'}): {r['verdict']}.** Gekozen kandidaat: "
         f"**{sel}** ({r['selection']}); kandidaten {', '.join(r['candidates'])}.", "",
         "| variant (G4-8.9) | regel | uitslag |", "|---|---|---|",
         f"| R-oud{' (beslissend)' if decisive == 'R-oud' else ''} | (a1) ∧ (a3) ∧ (b) ≥ +0,10, dan (a2) ∧ (c) | "
         f"{r['verdicts']['R-oud']} |",
         f"| R-nieuw{' (beslissend)' if decisive == 'R-nieuw' else ''} | (a1) ∧ (a3) ∧ C1 − G4-0 ≥ 0, dan (a2) ∧ (c) "
         f"na Yebs twijfelgevallen | {r['verdicts']['R-nieuw']} |", "",
         "## Besluit op het beslisdeel", "", f"| set | n | {sel} QE | G4-0 QE | mens | (a1) |", "|---|---:|---:|---:|---:|---|"]
    for s in sets:
        L.append(f"| {s} | {dec['n'][s]} | {f3(m['qe'][s])} | {f3(b0['qe'][s])} | {f3(dec['human'][s])} | "
                 f"{'ja' if m['a1_sets'][s] else 'nee'} |")
    L += ["", f"(a1) **{'gehaald' if r['a1']['holds'] else 'niet gehaald'}**.", "",
          f"| set | n | {sel} COMET-22 | G4-0 | verschil | (a3) |", "|---|---:|---:|---:|---:|---|"]
    for s, v in r["a3"]["sets"].items():
        L.append(f"| {s} | {v['n']} | {f4(v['model'])} | {f4(v['base'])} | {f4(v['delta'], True)} | "
                 f"{'ja' if v['holds'] else 'nee'} |")
    L += ["", f"(a3) **{'gehaald' if r['a3']['holds'] else 'niet gehaald'}**. C1 \"origineel\" op WMT24++ (goed) + FLAN: "
          f"{pc(m['share'])} tegen G4-0 {pc(b0['share'])}, verschil {f3(r['b']['delta'], True)} "
          f"(95%-BI {f3(ci['lo'], True)}; {f3(ci['hi'], True)}, gepaarde bootstrap, n = {ci['n']}): (b) ≥ +0,10 "
          f"**{'gehaald' if r['b']['holds'] else 'niet gehaald'}**; C1-vangrail ≥ 0 "
          f"**{'gehaald' if r['c1_guard']['holds'] else 'niet gehaald'}**.", ""]
    t = r["blind"]
    if t:
        L += [f"Blind tegen G4-0: {t['arm_better']} beter, {t['base_better']} slechter, {t['equal']} gelijk "
              f"(twijfel {t['doubt']}); fouten {t['errors_arm']} tegen {t['errors_base']} → (a2) "
              f"{'ja' if t['a2'] else 'nee'}, (c) {'ja' if t['c'] else 'nee'}.", ""]
    elif r["blind_due"]:
        L += [f"(a2) en (c): (a1) en (a3) gelden; het blinde pakket (`$S/blind/{r['arm']}/`) is nog niet geteld.", ""]
    else:
        L += ["(a2) en (c): niet beoordeeld; zonder (a1) en (a3) geen blinde beoordeling.", ""]
    if r.get("compare"):
        L += compare_text(r)
    for part, title in (("search", "Zoekdeel (de keuze)"), ("decision", "Beslisdeel, alle modellen (rapporteren)")):
        p = r["parts"][part]
        L += [f"## {title}", "", "| model | " + " | ".join(f"QE {s}" for s in sets) +
              " | (a1) | max. QE-overschrijding | (a3) | C1 origineel | C1 − G4-0 |", "|---|" + "---:|" * len(sets) +
              "---|---:|---|---:|---:|", "| mens | " + " | ".join(f3(p["human"][s]) for s in sets) + " | | | | | |"]
        for name, x in p["models"].items():
            L.append(f"| {name} | " + " | ".join(f3(x["qe"][s]) for s in sets) +
                     f" | {'ja' if x['a1'] else 'nee'} | {f3(x['excess'], True)} | "
                     f"{'ja' if r['a3_all'][part][name]['holds'] else 'nee'} | {pc(x['share'])} | {f3(x['b_delta'], True)} |")
        L.append("")
    L += ["## C1 en C2 (rapporteren)", "", "| deel | model | C1 samen | C1 − G4-0 | C2 samen | C2 − G4-0 | C2 stijgt, C1 niet |",
          "|---|---|---:|---:|---:|---:|---|"]
    for part, label in (("search", "zoek"), ("decision", "beslis")):
        for name, x in r["c2"][part].items():
            L.append(f"| {label} | {name} | {pc(x['c1']['pooled'])} | {f3(x['c1_delta']['pooled'], True)} | "
                     f"{pc(x['c2']['pooled'])} | {f3(x['c2_delta']['pooled'], True)} | "
                     f"{'**ja**' if x['c2_up_c1_not'] else 'nee'} |")
    L += ["", "Afgekapt op 1.024 tokens: " + "; ".join(f"{n} C1 {r['truncated']['c1'][n]}, C2 {r['truncated']['c2'][n]}"
                                                    for n in r["truncated"]["c2"]) + ".", "",
          "## Per model en set (rapporteren)", "",
          "| deel | model | set | n | QE | MetricX-ref | COMET-22 | niet af | lus | Engels over | leeg | uitvoer / ref | "
          "MTLD |", "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for part, label in (("search", "zoek"), ("decision", "beslis")):
        for name, per in r["counts"][part].items():
            for s, c in per.items():
                L.append(f"| {label} | {name} | {s} | {c['n']} | {f3(c['metricx_qe'])} | {f3(c['metricx_ref'])} | "
                         f"{f4(c['comet22'])} | {c['not_ended']} | {c['decode_loop']} | {c['english_left']} | "
                         f"{c['empty']} | {f3(c['out_per_ref'])} | {f3(c['mtld'])} |")
    L += ["", "## PLL (rapporteren)", ""]
    if r["pll"] is None:
        L += ["PLL: niet gemeten.", ""]
    else:
        L += ["| deel | model | set | n | weggelaten | PLL per vertaling | PLL per token |", "|---|---|---|---:|---:|---:|---:|"]
        for part, label in (("search", "zoek"), ("decision", "beslis")):
            for name, per in r["pll"][part].items():
                for s, v in per.items():
                    L.append(f"| {label} | {name} | {s} | {v['n']} | {v['skipped']} | {f3(v['pll'])} | "
                             f"{f3(v['pll_per_token'])} |")
        L.append("")
    run = r["run"]
    L += ["## De training en de kandidaten (G4-8.6, rapporteren)", "",
          f"{n0(run['examples'])} rijen ({json.dumps({p: x['examples'] for p, x in run['parts'].items()})} per "
          f"doorgang), {n0(run['tokens'])} tokens, {run['steps']} stappen, LR {run['lr']} constant na {run['warmup']} "
          f"stappen warmup, evaluatie elke {run['eval_every']} stappen; {f3(run['step_s_median'])} s per stap "
          f"(mediaan), piek {f3(run['peak_gib_max'])} GiB per chip.", "",
          "SFT-verlies per token (G4-0 op stap 0: " + ", ".join(f"{s} {f3(r['sft_step0'].get(s))}" for s in SFT_EVALS_T) +
          "). RL-val: 8 samples bij T = 1 per bron; vlag bij een mediaan < 4 verschillende.", "",
          "| model | stap | update-norm | elementen veranderd | " + " | ".join(f"{s} (Δ)" for s in SFT_EVALS_T) +
          " | verschillend (mediaan) | 8 gelijk | fail | vlag |", "|---|---:|---:|---:|" + "---:|" * len(SFT_EVALS_T) +
          "---:|---:|---:|---|"]
    dv = r["div_base"]
    L.append("| G4-0 | 0 | | | " + " | ".join(f3(r["sft_step0"].get(s)) for s in SFT_EVALS_T) +
             f" | {nl(dv['distinct_median'], 1)} | {pc(dv['all_equal'])} | {pc(dv['fail_share'])} | "
             f"{'**ja**' if dv['flag'] else 'nee'} |")
    for name, x in r["training"].items():
        dv = x["div"]
        L.append(f"| {name} | {x['step']} | {f4(x['update_norm'])} | {pc(x['elements_changed'])} | " +
                 " | ".join(f"{f3(x['sft'][s])} ({f3(x['sft_delta'][s], True)})" for s in SFT_EVALS_T) +
                 f" | {nl(dv['distinct_median'], 1)} | {pc(dv['all_equal'])} | {pc(dv['fail_share'])} | "
                 f"{'**ja**' if dv['flag'] else 'nee'} |")
    L += ["", "## Smoke-test (G4-8.5)", ""]
    s = r["smoke"]
    if s is None:
        L += ["Geen verslag van `smoke8` bij dit verslag.", ""]
    else:
        L += [f"Geslaagd: {'ja' if s['passed'] else 'nee'} ({len(s['checks'])} controles; niet gehaald: "
              f"{', '.join(s['failed']) or 'geen'}); schatting {json.dumps(s.get('estimate'))}.", ""]
    L += [f"Controle menselijke referenties (QE opnieuw, hele set): {json.dumps(r['refqe_check'])}", ""]
    return "\n".join(L) + "\n"


def compare_text(r):
    """§22 G4-9.11: the comparison with the earlier arm's choice on the decision part"""
    import g4arm

    x, sel, dec = r["compare"], r["selected"], r["parts"]["decision"]
    ref = x["ref"]
    L = [f"## Vergelijking met {ref} op het beslisdeel (G4-9.11, rapporteren, beslist niets)", "",
         f"{sel} tegen {ref} (de keuze van de vorige arm, in dezelfde classify-run). QE: lager is beter; het interval is "
         "dat van het gepaarde verschil (bootstrap per set, 10.000 trekkingen).", "",
         f"| set | n | {sel} QE | {ref} QE | verschil (95%-BI) |", "|---|---:|---:|---:|---|"]
    for s in g4arm.HUMAN_QE:
        q = x["qe"][s]
        L.append(f"| {s} | {dec['n'][s]} | {f3(q['model'])} | {f3(q['ref'])} | {f3(q['ci']['delta'], True)} "
                 f"({f3(q['ci']['lo'], True)}; {f3(q['ci']['hi'], True)}) |")
    L += ["", f"(a1): {sel} {'ja' if x['a1']['model'] else 'nee'}, {ref} {'ja' if x['a1']['ref'] else 'nee'}.", "",
          f"| set | n | {sel} COMET-22 | {ref} | verschil | ≥ {ref} − 0,005 |", "|---|---:|---:|---:|---:|---|"]
    for s, v in x["a3"]["sets"].items():
        L.append(f"| {s} | {v['n']} | {f4(v['model'])} | {f4(v['base'])} | {f4(v['delta'], True)} | "
                 f"{'ja' if v['holds'] else 'nee'} |")
    ci = x["c1"]["ci"]
    L += ["", f"C1 \"origineel\" op WMT24++ (goed) + FLAN: {pc(x['c1']['model'])} tegen {pc(x['c1']['ref'])}, verschil "
          f"{f3(ci['delta'], True)} (95%-BI {f3(ci['lo'], True)}; {f3(ci['hi'], True)}, n = {ci['n']}). C2: "
          f"{pc(x['c2']['model'])} tegen {pc(x['c2']['ref'])}, verschil {f3(x['c2']['delta'], True)}.", ""]
    return L


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
    a.add_argument("--set", default="", choices=tuple(V3_SETS), help="n1: G4-9-V3N1 (§22 G4-9.V3N1); default G4-8-V3")
    a.add_argument("--choice", default=None, help="--set n1: select-n1's choice.parquet of the chunk")
    a.add_argument("--rubric", default=None, help="--set n1: §18's select/choice.parquet of the chunk")
    a.add_argument("--n1report", default=None, help="--set n1: §21's report.json")
    a = sub.add_parser("report")
    a.add_argument("--plan", required=True)
    a.add_argument("--chunk-dir", action="append", required=True,
                   help="CHUNK=DIR with choice.parquet, targets.parquet, cands.parquet, pairs.json")
    a.add_argument("--out", required=True)
    a = sub.add_parser("n1pairs")
    for name in ("plan", "chunk", "samples", "cands", "rubric", "out"):
        a.add_argument(f"--{name}", required=True)
    a = sub.add_parser("n1report")
    a.add_argument("--plan", required=True)
    a.add_argument("--chunk-dir", action="append", required=True,
                   help="CHUNK=DIR with n1pairs.json, cands.parquet, choice.parquet, choice-n1.parquet, scores/, scores-n1/")
    a.add_argument("--out", required=True)
    a = sub.add_parser("n1targets")
    for name in ("plan", "n1report", "out"):
        a.add_argument(f"--{name}", required=True)
    a.add_argument("--chunk-dir", action="append", required=True, help="CHUNK=DIR with select-n1's choice.parquet, targets.parquet")
    a = sub.add_parser("sftset")
    for name in ("targets", "plan", "evalsets", "blind", "flan-eval", "g47-pool", "g47-plan", "tokenizer", "out"):
        a.add_argument(f"--{name}", required=True)
    a.add_argument("--sft-eval", action="append", required=True, help="NAME=FILE: books, btval, own (G4-2.5)")
    a.add_argument("--arm", default=ARM8, choices=tuple(ARMS_T))
    a = sub.add_parser("smoke8")
    for name in ("run", "summary", "adapter", "adapter-step", "smoke-manifest", "manifest", "out"):
        a.add_argument(f"--{name}", required=True)
    a.add_argument("--arm", default=ARM8, choices=tuple(ARMS_T))
    a = sub.add_parser("trained")
    for name in ("run", "summary", "manifest", "out"):
        a.add_argument(f"--{name}", required=True)
    a.add_argument("--adapter", action="append", required=True, help="<arm>-s<step>=adapter json of that step")
    a.add_argument("--arm", default=ARM8, choices=tuple(ARMS_T))
    a = sub.add_parser("loraline")
    for name in ("logs", "out"):
        a.add_argument(f"--{name}", required=True)
    a = sub.add_parser("report8")
    for name in ("evalsets", "split", "refqe", "base", "predictions", "predictions-c2", "summary", "manifest", "rlval",
                 "out"):
        a.add_argument(f"--{name}", required=True)
    a.add_argument("--candidate", action="append", required=True)
    a.add_argument("--reference", action="append", default=[])
    a.add_argument("--lora", action="append", required=True, help="NAME=the json of translate.py's LORA line")
    a.add_argument("--div", action="append", required=True, help="NAME=the root of its RL-val samples (w<k>/)")
    a.add_argument("--o6", required=True, choices=(*RULES_T, "open"))
    a.add_argument("--pll", default=None)
    a.add_argument("--smoke", default=None)
    a.add_argument("--tally", default=None)
    a.add_argument("--arm", default=ARM8, choices=tuple(ARMS_T))
    a.add_argument("--compare", default=None, help="G4-9: G4-8's choice, one of the references (G4-9.11)")
    args = p.parse_args(argv)
    if getattr(args, "chunk", None) is not None and args.chunk not in dict(CHUNKS8):
        raise SystemExit(f"--chunk {args.chunk}: one of {[c for c, _ in CHUNKS8]}")
    {"fresh": fresh, "plan": plan, "qepairs": qepairs, "pairs": pairs, "select": select, "v3": v3,
     "report": report, "n1pairs": n1pairs, "n1report": n1report, "n1targets": n1targets, "sftset": sftset,
     "smoke8": smoke8, "trained": trained, "loraline": loraline, "report8": report8}[args.cmd](args)


if __name__ == "__main__":
    main()
