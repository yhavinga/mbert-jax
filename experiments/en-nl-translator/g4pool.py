"""The evaluation pool (protocol-g4.md §24, part 2, 24 Nov 2026): ~440 held-out English sources (novels 100, BT 100,
WMT24++ 100, FLORES 40, NTREX 40, idiom, FLAN <= 10%; the search part of G4-5.4 only), per source the greedy translations
of G4-0, G4-8-s140 and G4-7-i15, G4-0's 8 samples at T = 1 with four fixed choices against the anchor G4-0 greedy
(rubric-or-anchor, N1, rubric ∩ N1, A1-veto -> N1), the human translation where there is one, QE, COMET-22 (news),
C1, C2, and the judge's five prompts on the pairs a later human trial would use. A stock, no result: nothing is drawn for
anyone.

    python g4pool.py data --novels-meta F --b1 F --g47-pool F --g48-fresh F --g48-plan F --g48-targets F --c2-data F \\
        --bt DIR --evalsets F --split F --mix-data J --mix-key J --yeb1-key J --package-key NAME=F ... --sample2-key J \\
        --tokenizer T --out DIR                                                            (w0, CPU)
    python g4pool.py texts --data DIR --samples DIR --greedy MODEL=DIR ... --eval MODEL=GLOB ... --m23-qe F --out DIR
    python g4pool.py cands --data DIR --texts F --qe GLOB --m23-prompts GLOB --out DIR
    python g4pool.py choose --data DIR --texts F --cands F --scores GLOB --out DIR --prompts-out DIR
    python g4pool.py report --data DIR --texts-meta F --texts-json J --cands-json J --choose DIR --qe GLOB --comet F \\
        --c1 F --c2 F --judge1 GLOB --judge2 GLOB --m23-metricx F --m23-judge GLOB --out report.md

All CPU. The texts stay in the bucket and on the VM: no command prints a text; the json files and the report hold ids,
counts and scores."""

import argparse
import collections
import json
import random
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for _path in (HERE, HERE.parent / "flan-nl"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import g4arm  # noqa: E402
import g4arm8  # noqa: E402
import g4measure2 as m2  # noqa: E402
import g4mixsample as mx  # noqa: E402
import g4rl  # noqa: E402
import pairjudge  # noqa: E402

PROTOCOL = "protocol-g4.md §24 (Pool)"
SEED = "en-nl-g4arm/0/G4-maat2/pool"
NOVELS_PER_BOOK = 10
BT_PER_SOURCE = 20
EVAL_COUNTS = {"wmt24pp": 100, "flores": 40, "ntrex": 40}
FLAN_MAX = 40
PARTS = ("novels", "bt", "wmt24pp", "flores", "ntrex", "idiom", "flan")
NEWS = ("wmt24pp", "flores", "ntrex")
GREEDY = ("G4-0", "G4-8-s140", "G4-7-i15")
SAMPLES = g4arm8.G8
KINDS = GREEDY + tuple(f"s{j}" for j in range(SAMPLES)) + ("mens",)
CHOICES = {"bo8-rubriek": "rubriek-of-anker", "bo8-N1": "N1", "bo8-rubriek∩N1": "rubriek ∩ N1",
           "bo8-A1-veto→N1": "A1-veto → N1"}
SYSTEMS = GREEDY + tuple(CHOICES) + ("mens",)
JUDGE1 = ("rubric", "n1", "a1")
JUDGE2 = ("rubric", "n1", "n2", "a1", "jnat")
PROMPT_LABELS = {"rubric": "rubriek", "n1": "N1", "n2": "N2", "a1": "A1", "jnat": "J-nat"}
TRIALS = {"mens": ("G4-8-s140", "mens"), "bo8": ("G4-8-s140", "bo8-rubriek∩N1"), "i15": ("G4-8-s140", "G4-7-i15"),
          "keuze": ("bo8-rubriek", "bo8-N1")}
ANCHOR_P = g4arm8.ANCHOR_P
VETO = m2.VETO
REPRO_QE_N = 200
REPRO_QE_SEED = "en-nl-g4arm/0/G4-pool/repro-qe"
REPRO_QE_TOL = 0.06
REPRO_JUDGE_N = 100
REPRO_JUDGE_SEED = "en-nl-g4arm/0/G4-pool/repro-judge"
REPRO = "repro|"
SOURCE_FIELDS = (("id", "string"), ("part", "string"), ("stratum", "string"), ("doc", "string"), ("en", "string"),
                 ("human", "string"), ("tokens", "int64"))
SEGMENT_FIELDS = (("id", "string"), ("text", "string"), ("tokens", "int64"))


def norm(text):
    return " ".join((text or "").split())


def read_rows(path, columns=None):
    return g4arm.read_rows(path, columns)


def write(rows, path, fields=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if fields:
        g4rl.write_parquet(rows, path, g4rl.schema(fields))
    else:
        pairjudge.write_rows(rows, path)


def named(specs, what):
    out = {}
    for s in specs or []:
        name, _, value = s.partition("=")
        if not name or not value or name in out:
            raise SystemExit(f"{what} {s!r}: NAME=VALUE, each name once")
        out[name] = value
    return out


def mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.fmean(xs) if xs else None


# ---------------------------------------------------------------------------------------------------- data

def yeb_items(mix_key, yeb1_key, package_keys):
    """The items Yeb saw: the mixed sample's and V1's (through the packages' keys)"""
    items = {k["item"] for k in mix_key["pairs"]}
    for k in yeb1_key:
        key = package_keys.get(k["package"])
        if key is None or k["blind_id"] not in key:
            raise SystemExit(f"V1 {k['n']}: no key row of {k['package']}/{k['blind_id']}")
        items.add(key[k["blind_id"]]["item"])
    return items


def eval_rows(ev, part, assign, blind_items, seen):
    """(rows, excluded counts) of one evaluation set: search part, not is_bad_source, not a fixed blind item, not Yeb's"""
    rows, excluded = [], collections.Counter()
    for r in ev:
        if r["set"] != part:
            continue
        if assign.get(r["id"]) != "search":
            excluded["decision_or_none"] += 1
        elif part == "wmt24pp" and g4arm.bad_source(r["meta"]):
            excluded["bad_source"] += 1
        elif r["id"] in blind_items:
            excluded["fixed_blind_item"] += 1
        elif r["id"] in seen:
            excluded["yeb"] += 1
        else:
            human = (r["reference"] or "").strip() or None
            if part in NEWS and human is None:
                raise SystemExit(f"{r['id']}: no reference")
            rows.append({"id": r["id"], "doc": r["id"], "en": (r["source"] or "").strip(), "stratum": part,
                         "human": human})
    return rows, dict(excluded)


def idiom_rows(ev, sample2, seen):
    rows, excluded = [], collections.Counter()
    for r in ev:
        if r["set"] != "blind" or json.loads(r["meta"] or "{}").get("blind_set") != "idiom":
            continue
        if r["id"] in sample2 or r["id"] in seen:
            excluded["yeb"] += 1
            continue
        rows.append({"id": r["id"], "doc": r["id"], "en": (r["source"] or "").strip(), "stratum": "idiom", "human": None})
    return rows, dict(excluded)


def checked(part, rows, blocked, counter):
    """The checks and the filter of G4-8.15 (a shared id, document or English with G4-7's pool or G4-8's fresh, plan,
    targets stops; the filter counts): (kept rows, info)"""
    hits = mx.overlap_of(rows, blocked)
    if any(hits.values()):
        raise SystemExit(f"{part}: sources in G4-7's pool or G4-8's fresh, plan or targets: {hits}")
    keep, drops = g4rl.candidate_filter(rows, counter, blocked["grams"])
    return keep, {"sources": len(rows), "overlap": hits, "filter": drops, "kept": len(keep)}


def data(args, counter=None):
    import pyarrow.parquet as pq

    counter = counter or g4arm.TokenCounter(args.tokenizer)
    paths = {"g47_pool": args.g47_pool, "g48_fresh": args.g48_fresh, "g48_plan": args.g48_plan,
             "g48_targets": args.g48_targets}
    tables = {k: read_rows(p, ["id", "doc", "en"]) for k, p in paths.items()}
    blocked = mx.blocked_of(tables)
    mix = json.loads(Path(args.mix_data).read_text())
    mix_ids = set(mix["novels"]["ids"]) | set(mix["bt"]["ids"])
    package_keys = {name: pairjudge.by_id(pairjudge.read_jsonl(p), name) for name, p in named(args.package_key, "--package-key").items()}
    seen = yeb_items(json.loads(Path(args.mix_key).read_text()), json.loads(Path(args.yeb1_key).read_text()), package_keys)
    sample2 = {k["blind_id"] for k in json.loads(Path(args.sample2_key).read_text())}
    info = {"protocol": PROTOCOL, "seed": SEED, "blocked_rows": {k: len(v) for k, v in tables.items()},
            "yeb_items": len(seen), "sample2_blind_ids": len(sample2), "mix_sources": len(mix_ids), "parts": {}}
    drawn = {}
    # novels: the set novels with the human translation of the -b1- row, without G4-8.15's 60
    meta = read_rows(args.novels_meta, ["pid", "en_book_id"])
    want = sorted({str(m["pid"]) for m in meta})
    id_type = pq.read_schema(args.b1).field("id").type
    key = [int(x) for x in want] if str(id_type).startswith(("int", "uint")) else want
    b1 = pq.read_table(args.b1, columns=["id", "group", "en_book_id", "en", "nl"], filters=[("id", "in", key)]).to_pylist()
    nl_of = {str(r["id"]): (r["nl"] or "").strip() for r in b1}
    novels = []
    for r in mx.novel_rows(meta, b1):
        human = nl_of[r["id"][len("novels:"):]]
        if not human:
            raise SystemExit(f"{r['id']}: no human translation")
        if r["id"] not in mix_ids:
            novels.append({**r, "human": human})
    del b1
    keep, part_info = checked("novels", novels, blocked, counter)
    drawn["novels"], part_info["eligible"] = mx.draw(keep, NOVELS_PER_BOOK, lambda s: f"{SEED}/novels/{s}")
    part_info["without_mix"] = 500 - len(novels)
    info["parts"]["novels"] = part_info
    # BT: §20's pairs (C2's test split) with the Dutch original, without G4-8.15's 60
    import g4c2

    c2_rows = read_rows(args.c2_data, ["pid", "source", "split", "label", "text"])
    bt_in = g4arm.read_bt(args.bt, g4arm.revisions3(args.bt), split="train", extra=("nl_back",))
    fb, fb_info = pairjudge.fb_bt(c2_rows, bt_in, tables["g47_pool"], tables["g48_plan"])
    del c2_rows, bt_in
    bt = [{"id": f"bt:{p['id']}", "doc": g4c2.doc_key(p["id"]), "en": p["en"], "stratum": p["stratum"],
           "human": (p["first"] or "").strip()} for p in fb]
    n_bt = len(bt)
    bt = [r for r in bt if r["id"] not in mix_ids]
    keep, part_info = checked("bt", bt, blocked, counter)
    drawn["bt"], part_info["eligible"] = mx.draw(keep, BT_PER_SOURCE, lambda s: f"{SEED}/bt/{s}")
    part_info["without_mix"] = n_bt - len(bt)
    part_info["fb_bt"] = fb_info
    info["parts"]["bt"] = part_info
    # the evaluation sets: search part only; idiom of the fixed blind passages outside Yeb's samples
    ev = read_rows(args.evalsets, ["id", "set", "source", "reference", "meta"])
    assign = mx.split_assign(args.split)
    blind_items = {json.loads(r["meta"] or "{}").get("item") for r in ev if r["set"] == "blind"}
    for part, n in EVAL_COUNTS.items():
        rows, excluded = eval_rows(ev, part, assign, blind_items, seen)
        keep, part_info = checked(part, rows, blocked, counter)
        drawn[part], part_info["eligible"] = mx.draw(keep, n, lambda s: f"{SEED}/{s}")
        part_info["excluded"] = excluded
        info["parts"][part] = part_info
    rows, excluded = idiom_rows(ev, sample2, seen)
    keep, part_info = checked("idiom", rows, blocked, counter)
    drawn["idiom"] = sorted(keep, key=lambda r: r["id"])
    part_info["excluded"], part_info["eligible"] = excluded, {"idiom": len(keep)}
    info["parts"]["idiom"] = part_info
    others = sum(len(v) for v in drawn.values())
    n_flan = min(FLAN_MAX, others // 9)
    rows, excluded = eval_rows(ev, "flan", assign, blind_items, seen)
    keep, part_info = checked("flan", rows, blocked, counter)
    drawn["flan"], part_info["eligible"] = mx.draw(keep, n_flan, lambda s: f"{SEED}/{s}")
    part_info["excluded"] = excluded
    info["parts"]["flan"] = part_info
    out_rows = []
    for part in PARTS:
        for r in drawn[part]:
            out_rows.append({"id": r["id"], "part": part, "stratum": r["stratum"], "doc": r["doc"], "en": r["en"],
                             "human": r["human"], "tokens": int(r["tokens"])})
    if len({r["id"] for r in out_rows}) != len(out_rows) or len({r["en"] for r in out_rows}) != len(out_rows):
        raise SystemExit("an id or an English text twice in the pool")
    if any(r["tokens"] > 256 for r in out_rows):
        raise SystemExit("a source above 256 tokens (class medium)")
    total = len(out_rows)
    for part in PARTS:
        info["parts"][part]["drawn"] = len(drawn[part])
        info["parts"][part]["ids"] = [r["id"] for r in drawn[part]]
    info["total"] = total
    info["flan_share"] = len(drawn["flan"]) / total
    if info["flan_share"] > 0.10:
        raise SystemExit(f"FLAN {info['flan_share']:.3f} of the pool, above 10%")
    info["with_human"] = sum(r["human"] is not None for r in out_rows)
    info["inputs"] = {k: g4arm.sha256_file(p) for k, p in
                      (("novels_meta", args.novels_meta), ("b1", args.b1), ("c2_data", args.c2_data),
                       ("evalsets", args.evalsets), ("split", args.split), ("mix_data", args.mix_data),
                       ("mix_key", args.mix_key), ("yeb1_key", args.yeb1_key), ("sample2_key", args.sample2_key),
                       *paths.items())}
    out = Path(args.out)
    write(out_rows, out / "sources.parquet", SOURCE_FIELDS)
    write([{k: r[k] for k in ("id", "part", "stratum", "tokens")} | {"has_human": r["human"] is not None}
           for r in out_rows], out / "sources-meta.parquet")
    write([{"id": r["id"], "text": r["en"], "tokens": r["tokens"]} for r in out_rows], out / "segments.parquet",
          SEGMENT_FIELDS)
    (out / "data.json").write_text(json.dumps(info, indent=1) + "\n")
    print(f"POOLDATA {json.dumps({'total': total, 'flan_share': round(info['flan_share'], 4), 'with_human': info['with_human'], 'parts': {p: {k: v for k, v in x.items() if k not in ('ids', 'fb_bt')} for p, x in info['parts'].items()}})}",
          flush=True)
    return info


# ---------------------------------------------------------------------------------------------------- texts

def sources_of(data_dir):
    rows = read_rows(Path(data_dir) / "sources.parquet")
    return {r["id"]: r for r in rows}


def greedy_rows(root, sources):
    """{id: the one greedy translation (j = 0)} of a model, every source once, its English the source's"""
    out = {}
    for s in g4arm8.read_samples8(root):
        if s["j"] != 0 or s["id"] in out:
            raise SystemExit(f"{root}: {s['uid']} is not one greedy translation #0 per source")
        out[s["id"]] = s
    g4arm8.groups(list(out.values()), sources, g=1)
    return out


def text_row(i, part, kind, s):
    return {"tid": f"{i}|{kind}", "id": i, "part": part, "kind": kind, "text": s["hypothesis"] or "",
            "ended": bool(s["ended"]), "decode_loop": bool(s["decode_loop"]), "empty": bool(s["empty"]),
            "english_left": bool(s["english_left"]), "tokens_in": int(s["tokens_in"]), "tokens_out": int(s["tokens_out"])}


def texts(args):
    sources = sources_of(args.data)
    grouped = g4arm8.groups(g4arm8.read_samples8(args.samples), sources, g=SAMPLES)
    greedy = {m: greedy_rows(p, sources) for m, p in named(args.greedy, "--greedy").items()}
    if tuple(sorted(greedy)) != tuple(sorted(GREEDY)):
        raise SystemExit(f"--greedy names {sorted(greedy)}, not {sorted(GREEDY)}")
    rows = []
    for i in sorted(sources):
        src = sources[i]
        for m in GREEDY:
            rows.append(text_row(i, src["part"], m, greedy[m][i]))
        for s in grouped[i]:
            rows.append(text_row(i, src["part"], f"s{s['j']}", s))
        if src["human"] is not None:
            rows.append({"tid": f"{i}|mens", "id": i, "part": src["part"], "kind": "mens", "text": src["human"],
                         "ended": True, "decode_loop": False, "empty": not src["human"].strip(), "english_left": False,
                         "tokens_in": None, "tokens_out": None})
    qe = [{"id": r["tid"], "source": sources[r["id"]]["en"], "hypothesis": r["text"]} for r in rows]
    m23 = sorted(read_rows(args.m23_qe, ["id", "source", "hypothesis"]), key=lambda r: r["id"])
    pairs = sorted({r["id"].rsplit("/", 1)[0] for r in m23})
    repro = set(random.Random(REPRO_QE_SEED).sample(pairs, REPRO_QE_N))
    rq = [{"id": REPRO + r["id"], "source": r["source"], "hypothesis": r["hypothesis"]} for r in m23
          if r["id"].rsplit("/", 1)[0] in repro]
    if len(rq) != 2 * REPRO_QE_N:
        raise SystemExit(f"the QE reproduction: {len(rq)} sides, not {2 * REPRO_QE_N}")
    out = Path(args.out)
    per = collections.defaultdict(list)
    for r in qe + rq:
        per[pairjudge.worker_of(r["id"])].append(r)
    for k in range(pairjudge.WORKERS):
        write(per.get(k, []), out / f"qe-in-w{k}.parquet", (("id", "string"), ("source", "string"), ("hypothesis", "string")))
    comet = [{"id": r["tid"], "source": sources[r["id"]]["en"], "hypothesis": r["text"],
              "reference": sources[r["id"]]["human"]} for r in rows if r["part"] in NEWS and r["kind"] != "mens"]
    write(comet, out / "comet-in.parquet")
    write(rows, out / "texts.parquet")
    write([{k: v for k, v in r.items() if k != "text"} for r in rows], out / "texts-meta.parquet")
    write([{"model": "g4pool", "set": r["part"], "id": r["tid"], "text": r["text"]} for r in rows],
          out / "nat-texts.parquet")
    # Pool.3: the greedy texts against the evaluation's own output of the same model (reported)
    same = {}
    for m, pattern in named(args.eval, "--eval").items():
        ev = mx.read_pairs(sorted(Path(pattern).parent.glob(Path(pattern).name)))
        c = collections.Counter()
        for i, src in sources.items():
            if src["part"] in ("novels", "bt"):
                continue
            if i not in ev:
                c["not_in_eval"] += 1
            else:
                c["same" if norm(ev[i]["hypothesis"]) == norm(greedy[m][i]["hypothesis"]) else "differ"] += 1
        same[m] = dict(c)
    info = {"protocol": PROTOCOL, "sources": len(sources), "texts": len(rows),
            "kinds": dict(collections.Counter(r["kind"] for r in rows)), "qe_rows": len(qe), "qe_repro_rows": len(rq),
            "qe_workers": {f"w{k}": len(per.get(k, [])) for k in range(pairjudge.WORKERS)}, "comet_rows": len(comet),
            "empty": dict(collections.Counter(r["kind"] for r in rows if r["empty"])),
            "not_ended": dict(collections.Counter(r["kind"] for r in rows if not r["ended"])),
            "greedy_vs_eval": same, "qe_repro_pairs": sorted(repro)}
    (out / "texts.json").write_text(json.dumps(info, indent=1) + "\n")
    print(f"POOLTEXTS {json.dumps({k: v for k, v in info.items() if k != 'qe_repro_pairs'})}", flush=True)
    return info


# ---------------------------------------------------------------------------------------------------- candidates

def read_qe(pattern):
    return g4arm8.read_qe(pattern)


def as_sample(t):
    return {"uid": t["tid"], "id": t["id"], "j": int(t["kind"][1:]), "hypothesis": t["text"], "ended": t["ended"],
            "decode_loop": t["decode_loop"], "empty": t["empty"], "english_left": t["english_left"],
            "tokens_in": t["tokens_in"], "tokens_out": t["tokens_out"]}


def by_tid(path):
    rows = read_rows(path)
    out = {r["tid"]: r for r in rows}
    if len(out) != len(rows):
        raise SystemExit(f"{path}: a tid twice")
    return out


def cands(args):
    sources = sources_of(args.data)
    texts_ = by_tid(args.texts)
    qe = read_qe(args.qe)
    pool_qe = {k: v for k, v in qe.items() if not k.startswith(REPRO)}
    if set(pool_qe) != set(texts_):
        raise SystemExit(f"QE: {len(set(texts_) - set(pool_qe))} texts without a row, {len(set(pool_qe) - set(texts_))} extra")
    rows, prompts = [], []
    for i in sorted(sources):
        src = sources[i]
        anchor = texts_[f"{i}|G4-0"]
        samples = [as_sample(texts_[f"{i}|s{j}"]) for j in range(SAMPLES)]
        qe0 = pool_qe[anchor["tid"]]
        if g4rl.anchor_ok(anchor) and qe0 is not None:
            plan = {"id": i, "hypothesis0": anchor["text"], "l0": anchor["tokens_out"]}
            found, key_of = g4arm8.source_candidates(plan, samples, pool_qe, qe0)
            for key in key_of.values():
                j = int(key.rsplit("/j", 1)[1])
                for name in JUDGE1:
                    prompts += [{"uid": u, "text": t} for u, t, _ in
                                pairjudge.pair_prompts(f"{key}/{name}", src["en"], samples[j]["hypothesis"],
                                                       anchor["text"], m2.ALL_PROMPTS[name])]
        else:
            found = [{"uid": s["uid"], "id": i, "j": s["j"], "fail": "anchor_not_ok", "qe": pool_qe[s["uid"]],
                      "qe0": qe0, "ratio": None, "tokens_in": s["tokens_in"], "tokens_out": s["tokens_out"],
                      "same_as_anchor": norm(s["hypothesis"]) == norm(anchor["text"]), "candidate": False, "key": None}
                     for s in samples]
        rows += [{**r, "part": src["part"]} for r in found]
    m23 = m2.texts_of(args.m23_prompts)
    pairs = sorted({u.rsplit("/", 2)[0] for u in m23})
    repro = sorted(random.Random(REPRO_JUDGE_SEED).sample(pairs, REPRO_JUDGE_N))
    prompts += [{"uid": f"repro/{p}/{n}/{o}", "text": m23[f"{p}/{n}/{o}"]} for p in repro for n in m2.OLD_PROMPTS
                for o in pairjudge.ORDERS]
    out = Path(args.out)
    write(rows, out / "cands.parquet")
    workers = pairjudge.write_prompts(prompts, Path(args.prompts_out))
    stats = {}
    for part in PARTS:
        rs = [r for r in rows if r["part"] == part]
        stats[part] = {"sources": len({r["id"] for r in rs}),
                       "anchor_not_ok": len({r["id"] for r in rs if r["fail"] == "anchor_not_ok"}),
                       "samples": len(rs), "fail_any": sum(bool(r["fail"]) for r in rs),
                       "fail": {f: sum(f in (r["fail"] or "").split(",") for r in rs) for f in g4arm8.FAILS8},
                       "same_as_anchor": sum(bool(r["same_as_anchor"]) for r in rs),
                       "candidates": sum(bool(r["candidate"]) for r in rs),
                       "keys": len({r["key"] for r in rs if r["candidate"]}),
                       "sources_with_candidate": len({r["id"] for r in rs if r["candidate"]})}
    info = {"protocol": PROTOCOL, "prompts": len(prompts), "repro_prompts": 6 * REPRO_JUDGE_N, "workers": workers,
            "parts": stats, "repro_pairs": repro}
    (out / "cands.json").write_text(json.dumps(info, indent=1) + "\n")
    print(f"POOLCANDS {json.dumps({k: v for k, v in info.items() if k != 'repro_pairs'})}", flush=True)
    return info


# ---------------------------------------------------------------------------------------------------- choices

def choose_a1veto(cands_rows, p_n1, p_a1):
    """Pool.5 A1-veto -> N1: per candidate A1 decides at |P̄_A1(sample) - P̄_A1(anchor)| >= 0.2, else N1's rule
    (P̄_N1 > 0.5); among the candidates that gives the sample, the highest P̄_N1 (ties: lower QE, earlier sample)"""
    judged = [r for r in cands_rows if r["candidate"] and p_n1.get(r["key"]) is not None
              and p_a1.get(r["key"]) is not None]
    if not judged:
        return None, "none"
    ok = []
    for r in judged:
        a1, n1 = p_a1[r["key"]], p_n1[r["key"]]
        if abs(a1["A"] - a1["B"]) >= VETO:
            if a1["A"] > a1["B"]:
                ok.append(r)
        elif n1["A"] > ANCHOR_P:
            ok.append(r)
    if not ok:
        return None, "p"
    return min(ok, key=lambda r: (-p_n1[r["key"]]["A"], r["qe"], r["j"])), "sample"


def choices_of(rows, p_of):
    """{choice: (row or None, reason)} of one source's candidate rows (Pool.5)"""
    kr, rr = g4arm8.choose8(rows, p_of["rubric"])
    kn, rn = g4arm8.choose8(rows, p_of["n1"])
    same = kr is not None and kn is not None and kr["j"] == kn["j"]
    ka, ra = choose_a1veto(rows, p_of["n1"], p_of["a1"])
    return {"bo8-rubriek": (kr, rr), "bo8-N1": (kn, rn),
            "bo8-rubriek∩N1": (kr if same else None, "sample" if same else "differ"), "bo8-A1-veto→N1": (ka, ra)}


def choose(args):
    sources = sources_of(args.data)
    texts_ = by_tid(args.texts)
    rows = collections.defaultdict(list)
    for r in read_rows(args.cands):
        rows[r["id"]].append(r)
    if set(rows) != set(sources):
        raise SystemExit(f"{args.cands}: not the {len(sources)} sources of the pool")
    scores = {u: s for u, s in pairjudge.scores_by_uid(pairjudge.read_glob(args.scores)).items()
              if not u.startswith("repro/")}
    keys = {r["key"] for rs in rows.values() for r in rs if r["candidate"]}
    want = {f"{k}/{n}/{o}" for k in keys for n in JUDGE1 for o in pairjudge.ORDERS}
    if set(scores) != want:
        raise SystemExit(f"judge scores: {len(want - set(scores))} prompts missing, {len(set(scores) - want)} extra")
    p_of = {n: {k: pairjudge.combine(scores[f"{k}/{n}/ab"], scores[f"{k}/{n}/ba"]) for k in keys} for n in JUDGE1}
    choice, outputs, pairs_meta, pairs, prompts = [], [], [], [], []
    for i in sorted(sources):
        src, rs = sources[i], sorted(rows[i], key=lambda r: r["j"])
        picked = choices_of(rs, p_of)
        tid = {m: f"{i}|{m}" for m in GREEDY}
        rec = {"id": i, "part": src["part"], "stratum": src["stratum"],
               "candidates": sum(bool(r["candidate"]) for r in rs), "keys": len({r["key"] for r in rs if r["candidate"]}),
               "not_scored": {n: sum(1 for k in {r["key"] for r in rs if r["candidate"]} if p_of[n][k] is None)
                              for n in JUDGE1},
               "anchor_not_ok": any(r["fail"] == "anchor_not_ok" for r in rs)}
        for c, (k, reason) in picked.items():
            tid[c] = f"{i}|s{k['j']}" if k is not None else f"{i}|G4-0"
            rec[f"{c}|j"], rec[f"{c}|reason"] = (-1 if k is None else int(k["j"])), reason
        if src["human"] is not None:
            tid["mens"] = f"{i}|mens"
        rec["not_scored"] = json.dumps(rec["not_scored"])
        choice.append(rec)
        for system, t in tid.items():
            outputs.append({"id": i, "part": src["part"], "system": system, "tid": t, "text": texts_[t]["text"]})
        for ptype, (a, b) in TRIALS.items():
            if a not in tid or b not in tid:
                continue
            ta, tb = texts_[tid[a]]["text"], texts_[tid[b]]["text"]
            same = norm(ta) == norm(tb)
            if ptype == "keuze" and same:
                continue
            pid = f"{ptype}/{i}"
            pairs_meta.append({"pair": pid, "type": ptype, "id": i, "part": src["part"], "a_system": a, "b_system": b,
                               "a_tid": tid[a], "b_tid": tid[b], "identical": same})
            if same:
                continue
            pairs.append({"pair": pid, "source": src["en"], "A": ta, "B": tb})
            for name in JUDGE2:
                prompts += [{"uid": u, "text": t} for u, t, _ in
                            pairjudge.pair_prompts(f"{pid}/{name}", src["en"], ta, tb, m2.ALL_PROMPTS[name])]
    out = Path(args.out)
    write(choice, out / "choice.parquet")
    write(outputs, out / "outputs.parquet")
    write([{k: v for k, v in r.items() if k != "text"} for r in outputs], out / "outputs-meta.parquet")
    write(pairs_meta, out / "pairs-meta.parquet")
    write(pairs, out / "pairs.parquet")
    workers = pairjudge.write_prompts(prompts, Path(args.prompts_out))
    stats = {p: {c: dict(collections.Counter(r[f"{c}|reason"] for r in choice if r["part"] == p)) for c in CHOICES}
             for p in PARTS}
    trial = {t: {"pairs": sum(r["type"] == t for r in pairs_meta),
                 "identical": sum(r["type"] == t and r["identical"] for r in pairs_meta)} for t in TRIALS}
    info = {"protocol": PROTOCOL, "sources": len(choice), "reasons": stats, "trial": trial, "prompts": len(prompts),
            "workers": workers}
    (out / "choose.json").write_text(json.dumps(info, indent=1, ensure_ascii=False) + "\n")
    print(f"POOLCHOOSE {json.dumps(info, ensure_ascii=False)}", flush=True)
    return info


# ---------------------------------------------------------------------------------------------------- the report

def read_scores(path, column, key="id"):
    out = {}
    for r in pairjudge.read_rows(path, [key, column]):
        if r[key] in out:
            raise SystemExit(f"{path}: {r[key]} twice")
        out[r[key]] = r[column]
    return out


def qe_repro(qe, m23):
    """Pool.7: the reproduced QE of §23's sides within REPRO_QE_TOL of §23's metricx.parquet; a deviation stops"""
    got = {k[len(REPRO):]: v for k, v in qe.items() if k.startswith(REPRO)}
    if len(got) != 2 * REPRO_QE_N:
        raise SystemExit(f"the QE reproduction has {len(got)} sides, not {2 * REPRO_QE_N}")
    deltas = []
    for k, v in got.items():
        if k not in m23 or (v is None) != (m23[k] is None):
            raise SystemExit(f"the QE reproduction: {k} is not scored as in §23")
        if v is not None:
            deltas.append(abs(v - m23[k]))
    worst = max(deltas, default=0.0)
    result = {"sides": len(got), "max_abs": worst, "median_abs": statistics.median(deltas) if deltas else None,
              "tolerance": REPRO_QE_TOL}
    if worst > REPRO_QE_TOL:
        raise SystemExit(f"the QE reproduction deviates from §23: max |Δ| {worst:.4f} > {REPRO_QE_TOL}")
    return result


def system_table(meta, outputs, tmeta, qe, comet, c1, c2):
    """{part or 'alle': {system: means}} of Pool.8"""
    out = {}
    for part in PARTS + ("alle",):
        out[part] = {}
        for system in SYSTEMS:
            ts = [o["tid"] for o in outputs if o["system"] == system and (part == "alle" or o["part"] == part)]
            if not ts:
                continue
            c1s = [c1[t] for t in ts]
            out[part][system] = {
                "n": len(ts), "qe": mean(qe[t] for t in ts), "qe_missing": sum(qe[t] is None for t in ts),
                "comet": mean(comet.get(t) for t in ts), "comet_n": sum(comet.get(t) is not None for t in ts),
                "c1": mean(c1s), "c1_original": sum(p >= 0.5 for p in c1s) / len(c1s), "c2": mean(c2[t] for t in ts),
                "empty": sum(bool(tmeta[t]["empty"]) for t in ts),
                "not_ended": sum(not tmeta[t]["ended"] for t in ts)}
    return out


def agreement_table(outputs):
    tid = collections.defaultdict(dict)
    for o in outputs:
        tid[o["id"]][o["system"]] = o["tid"]
    names = list(CHOICES)
    out = {}
    for a in names:
        out[a] = {b: sum(t[a] == t[b] for t in tid.values()) / len(tid) for b in names}
    out["anker"] = {c: sum(t[c] == t["G4-0"] for t in tid.values()) / len(tid) for c in names}
    return out


def trial_table(pairs_meta, judge2):
    """Per pair type (and per part) and prompt: n, A's direction (P̄ credit), mean P̄"""
    out = {}
    for t in TRIALS:
        mine = [r for r in pairs_meta if r["type"] == t]
        for part in ("alle",) + PARTS:
            rs = [r for r in mine if part == "alle" or r["part"] == part]
            if not rs:
                continue
            row = {"pairs": len(rs), "identical": sum(r["identical"] for r in rs)}
            for name in JUDGE2:
                ps, skipped = [], 0
                for r in rs:
                    if r["identical"]:
                        continue
                    ab, ba = judge2.get(f"{r['pair']}/{name}/ab"), judge2.get(f"{r['pair']}/{name}/ba")
                    if ab is None or ba is None:
                        raise SystemExit(f"{r['pair']}/{name}: no score of both orders")
                    p = pairjudge.combine(ab, ba)
                    if p is None:
                        skipped += 1
                    else:
                        ps.append(p)
                row[name] = {"direction": pairjudge.share(pairjudge.direction_credit(p, "A") for p in ps) if ps else
                             {"n": 0, "share": None, "lo": None, "hi": None},
                             "mean_a": mean(p["A"] for p in ps), "mean_b": mean(p["B"] for p in ps),
                             "mean_gelijk": mean(p["gelijk"] for p in ps), "not_scored": skipped}
            out[f"{t}|{part}"] = row
    return out


def report(args):
    meta = {r["id"]: r for r in read_rows(Path(args.data) / "sources-meta.parquet")}
    info = json.loads((Path(args.data) / "data.json").read_text())
    tinfo = json.loads(Path(args.texts_json).read_text())
    cinfo = json.loads(Path(args.cands_json).read_text())
    choose_dir = Path(args.choose)
    chinfo = json.loads((choose_dir / "choose.json").read_text())
    tmeta = by_tid(args.texts_meta)
    if "text" in next(iter(tmeta.values())):
        raise SystemExit(f"{args.texts_meta} holds text")
    outputs = read_rows(choose_dir / "outputs-meta.parquet")
    pairs_meta = read_rows(choose_dir / "pairs-meta.parquet")
    choice = read_rows(choose_dir / "choice.parquet")
    qe = read_qe(args.qe)
    comet = read_scores(args.comet, "comet22")
    c1 = read_scores(args.c1, "p_original")
    c2 = read_scores(args.c2, "p_original")
    for name, d in (("C1", c1), ("C2", c2)):
        if set(d) != set(tmeta):
            raise SystemExit(f"{name}: {len(set(tmeta) - set(d))} texts missing, {len(set(d) - set(tmeta))} extra")
    news = {t for t, r in tmeta.items() if r["part"] in NEWS and r["kind"] != "mens"}
    if set(comet) != news:
        raise SystemExit(f"COMET: {len(news - set(comet))} missing, {len(set(comet) - news)} extra")
    checks = {"qe_repro": qe_repro(qe, read_scores(args.m23_metricx, "qe_source_hypothesis"))}
    judge1 = pairjudge.scores_by_uid(pairjudge.read_glob(args.judge1))
    repro = {u[len("repro/"):]: s for u, s in judge1.items() if u.startswith("repro/")}
    checks["judge_repro"] = m2.repro_check(repro, pairjudge.scores_by_uid(pairjudge.read_glob(args.m23_judge)),
                                           cinfo["repro_pairs"])
    judge2 = pairjudge.scores_by_uid(pairjudge.read_glob(args.judge2))
    want = {f"{r['pair']}/{n}/{o}" for r in pairs_meta if not r["identical"] for n in JUDGE2 for o in pairjudge.ORDERS}
    if set(judge2) != want:
        raise SystemExit(f"judge2: {len(want - set(judge2))} scores missing, {len(set(judge2) - want)} extra")
    checks["greedy_vs_eval"] = tinfo["greedy_vs_eval"]
    r = {"protocol": PROTOCOL, "sources": len(meta), "parts": {p: {k: v for k, v in x.items() if k not in ("ids", "fb_bt")}
                                                               for p, x in info["parts"].items()},
         "flan_share": info["flan_share"], "with_human": info["with_human"],
         "systems": system_table(meta, outputs, tmeta, qe, comet, c1, c2),
         "choices": {p: {c: dict(collections.Counter(x[f"{c}|reason"] for x in choice if p == "alle" or x["part"] == p))
                         for c in CHOICES} for p in ("alle",) + PARTS},
         "agreement": agreement_table(outputs), "cands": cinfo["parts"], "trial": trial_table(pairs_meta, judge2),
         "trial_counts": chinfo["trial"], "texts": {k: tinfo[k] for k in ("texts", "kinds", "empty", "not_ended")},
         "checks": checks}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report_text(r), encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(r, indent=1, ensure_ascii=False) + "\n")
    print("POOLREPORT " + json.dumps({"sources": r["sources"], "qe_repro": checks["qe_repro"]["max_abs"],
                                      "judge_repro": checks["judge_repro"]["max_abs"]}), flush=True)
    return r


def num(x, digits=3):
    return "–" if x is None else f"{x:.{digits}f}".replace(".", ",")


def report_text(r):
    pct = m2.gm.pct
    L = [f"# Evaluatiepool G4 ({PROTOCOL})", "",
         f"Een voorraad achtergehouden bronnen voor een eventuele latere menselijke proef; er is niets getrokken of "
         f"voorgelegd. {r['sources']} bronnen, waarvan {r['with_human']} met een menselijke tekst; FLAN "
         f"{pct(r['flan_share'])}. Alleen tellingen en scores, geen tekst. Rapportage; beslist niets.", "",
         "## 1. Bronnen per stratum", "",
         "| stratum | bronnen | uitgesloten | na filter | getrokken | filter (gedropt) |", "|---|---:|---|---:|---:|---|"]
    for part, x in r["parts"].items():
        excl = x.get("excluded") or ({"G4-8.15": x["without_mix"]} if "without_mix" in x else {})
        drops = {k: v for k, v in x["filter"].items() if v}
        L.append(f"| {part} | {x['sources']} | {json.dumps(excl)} | {x['kept']} | {x['drawn']} | {json.dumps(drops)} |")
    L += ["", "## 2. Gemiddelde scores per systeem", "",
          "QE: MetricX-24 XXL zonder referentie (lager is beter); COMET-22 met de referentie (alleen WMT24++, FLORES, "
          "NTREX); C1: gem. P(origineel) en aandeel ≥ 0,5; C2: gem. P(origineel). `bo8-*`: G4-0 best-of-8 met de keuze, "
          "terugval G4-0 greedy.", ""]
    for part, systems in r["systems"].items():
        L += [f"**{part}**", "", "| systeem | n | QE | COMET-22 (n) | C1 | C1 ≥ 0,5 | C2 | leeg | niet af |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for s, x in systems.items():
            L.append(f"| {s} | {x['n']} | {num(x['qe'])} | {num(x['comet'], 4)} ({x['comet_n']}) | {num(x['c1'])} | "
                     f"{pct(x['c1_original'])} | {num(x['c2'])} | {x['empty']} | {x['not_ended']} |")
        L.append("")
    L += ["## 3. Best-of-8: keuzes", "", "Reden per keuze: `sample`, `p` (geen kandidaat boven de grens: anker), "
          "`none` (geen beoordeelde kandidaat: anker), `differ` (rubriek en N1 kozen niet hetzelfde sample: anker).", "",
          "| stratum | " + " | ".join(CHOICES.values()) + " |", "|---|" + "---|" * len(CHOICES)]
    for part, x in r["choices"].items():
        L.append(f"| {part} | " + " | ".join(json.dumps(x[c]) for c in CHOICES) + " |")
    L += ["", "Aandeel bronnen met dezelfde tekst-id (rij tegen kolom; `anker` = de keuze is G4-0 greedy):", "",
          "| | " + " | ".join(CHOICES.values()) + " |", "|---|" + "---:|" * len(CHOICES)]
    for a, row in r["agreement"].items():
        L.append(f"| {CHOICES.get(a, a)} | " + " | ".join(pct(row[b]) for b in CHOICES) + " |")
    L += ["", "Kandidaten per stratum:", "", "| stratum | bronnen | anker niet ok | samples | fail | gelijk aan anker | "
          "kandidaten | verschillende | bronnen met kandidaat |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for part, x in r["cands"].items():
        L.append(f"| {part} | {x['sources']} | {x['anchor_not_ok']} | {x['samples']} | {x['fail_any']} | "
                 f"{x['same_as_anchor']} | {x['candidates']} | {x['keys']} | {x['sources_with_candidate']} |")
    L += ["", "## 4. Judge-scores van de paren voor een latere proef", "",
          "Per paartype en prompt de richting voor A (aandeel P̄(A) > P̄(B), gelijk ½; 95%-Wilson) en n. "
          + "; ".join(f"**{t}**: A = {a}, B = {b}" for t, (a, b) in TRIALS.items())
          + ". Paren met gelijke tekst zijn niet gescoord.", "",
          "| paartype | stratum | paren | gelijke tekst | " + " | ".join(PROMPT_LABELS[n] for n in JUDGE2) + " |",
          "|---|---|---:|---:|" + "---:|" * len(JUDGE2)]
    for key, x in r["trial"].items():
        t, part = key.split("|")
        L.append(f"| {t} | {part} | {x['pairs']} | {x['identical']} | " + " | ".join(
            (f"{pct(x[n]['direction']['share'])} (n {x[n]['direction']['n']})" if x[n]["direction"]["n"] else "–")
            for n in JUDGE2) + " |")
    L += ["", "Gemiddelde P̄(A) / P̄(B) over alle strata:", "",
          "| paartype | " + " | ".join(PROMPT_LABELS[n] for n in JUDGE2) + " |", "|---|" + "---|" * len(JUDGE2)]
    for t in TRIALS:
        x = r["trial"].get(f"{t}|alle")
        if x:
            L.append(f"| {t} | " + " | ".join(f"{num(x[n]['mean_a'])} / {num(x[n]['mean_b'])}" for n in JUDGE2) + " |")
    c = r["checks"]
    L += ["", "## 5. Controles", "",
          f"- QE-reproductie: {c['qe_repro']['sides']} kanten van §23, grootste |Δ| {num(c['qe_repro']['max_abs'], 4)}, "
          f"mediaan {num(c['qe_repro']['median_abs'], 4)} (grens {c['qe_repro']['tolerance']}).",
          f"- Judge-reproductie: {c['judge_repro']['pairs']} §23-paren, {c['judge_repro']['prompts']} prompts, grootste "
          f"|Δp| {c['judge_repro']['max_abs']:.3g} (grens {c['judge_repro']['tolerance']:g}), "
          f"{c['judge_repro']['deviations']} afwijkingen.",
          "- Greedy tegen de bestaande evaluatie-uitvoer (bronnen uit de evaluatiesets; witruimte genormaliseerd): "
          + json.dumps(c["greedy_vs_eval"]) + ".",
          "- Teksten: " + json.dumps(r["texts"], ensure_ascii=False) + ".",
          "", "## 6. Wat het niet zegt", "",
          "- Een voorraad, geen uitslag: automatische maten en judge-oordelen van Gemma 4 31B-it, het model achter G4-0, "
          "G4-8-s140 en G4-7-i15.",
          "- FLAN, WMT24++, FLORES en NTREX komen uit het zoekdeel, dat bij G4-7 en G4-8 voor de keuze van de kandidaat "
          "gelezen is; de romanboeken zitten in G4-4's training; het BT-origineel is geen vertaling van het Engels.", ""]
    return "\n".join(L)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("data")
    for f in ("novels-meta", "b1", "g47-pool", "g48-fresh", "g48-plan", "g48-targets", "c2-data", "bt", "evalsets",
              "split", "mix-data", "mix-key", "yeb1-key", "sample2-key", "tokenizer", "out"):
        a.add_argument(f"--{f}", required=True)
    a.add_argument("--package-key", action="append", required=True, help="NAME=key.jsonl of V1's packages")
    a = sub.add_parser("texts")
    for f in ("data", "samples", "m23-qe", "out"):
        a.add_argument(f"--{f}", required=True)
    a.add_argument("--greedy", action="append", required=True, help="MODEL=DIR of its w<k> translations")
    a.add_argument("--eval", action="append", default=[], help="MODEL=GLOB of the evaluation's pairs")
    a = sub.add_parser("cands")
    for f in ("data", "texts", "qe", "m23-prompts", "out", "prompts-out"):
        a.add_argument(f"--{f}", required=True)
    a = sub.add_parser("choose")
    for f in ("data", "texts", "cands", "scores", "out", "prompts-out"):
        a.add_argument(f"--{f}", required=True)
    a = sub.add_parser("report")
    for f in ("data", "texts-meta", "texts-json", "cands-json", "choose", "qe", "comet", "c1", "c2", "judge1", "judge2",
              "m23-metricx", "m23-judge", "out"):
        a.add_argument(f"--{f}", required=True)
    args = p.parse_args(argv)
    {"data": data, "texts": texts, "cands": cands, "choose": choose, "report": report}[args.cmd](args)


if __name__ == "__main__":
    main()
