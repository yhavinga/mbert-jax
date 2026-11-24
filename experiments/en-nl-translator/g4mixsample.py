#!/usr/bin/env python3
"""G4-8.15 (protocol-g4.md §19): a human sample of the final model G4-8-s140 against G4-0 from all sources, reported, it
decides nothing. 40 blind pairs: novels 8 and BT 10 newly translated by both models (60 + 60 sources), FLAN 6, WMT24++ 6,
FLORES 3, NTREX 2 and idiom 5 from the evaluation's translations; only pairs whose two translations differ. Beside it a
blind package of every differing novel and BT pair for a Claude reviewer.

    python g4mixsample.py data --novels-meta F --b1 F --g47-pool F --g48-fresh F --g48-plan F --g48-targets F \\
        --c2-data F --bt DIR --tokenizer T --out DIR
        (evalsets.parquet, segments.parquet, classes.txt, data.json: the 120 sources in evalsets.py's data format)
    python g4mixsample.py pick --data DIR --mix base=F --mix G4-8-s140=F --eval base=GLOB --eval G4-8-s140=GLOB \\
        --evalsets F --split F --exclude-items F ... --used-blind F --out DIR
        (steekproef.md, key.json, judge-input.jsonl, judge-key.jsonl, select-report.json, pick.json)

All CPU. The texts stay in the bucket and on the VM: no command prints a text; the json files hold ids and counts. The
sample file for Yeb is the one output with text that leaves the VM."""

import argparse
import collections
import glob
import hashlib
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for _path in (HERE, HERE.parent / "flan-nl"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import g4arm  # noqa: E402
import g4rl  # noqa: E402

PROTOCOL = "protocol-g4.md §19 G4-8.15"
SEED = "en-nl-g4arm/0/G4-8/mix"
MODEL8 = "G4-8-s140"
BASE = "base"                                # G4-0 without an adapter, as the other blind packages name it
MODELS = (BASE, MODEL8)
NOVEL_BOOKS = 10                             # the set novels (G4-4.10)
NOVEL_PAIRS = 500
NOVELS_PER_BOOK = 6                          # 60 novel sources
BT_PER_SOURCE = 12                           # 60 BT sources
PICK = {"novels": 8, "bt": 10, "flan": 6, "wmt24pp": 6, "flores": 3, "ntrex": 2, "idiom": 5}
BT_PICK_PER_SOURCE = 2
EVAL_PICK = ("flan", "wmt24pp", "flores", "ntrex")   # from the search part of G4-5.4
BLOCKED_FILES = ("g47_pool", "g48_fresh", "g48_plan", "g48_targets")
SEGMENT_SCHEMA = (("id", "string"), ("text", "string"), ("tokens", "int64"))
EVALSET_SCHEMA = (("id", "string"), ("set", "string"), ("source", "string"), ("reference", "string"),
                  ("tokens", "int64"), ("over_512", "bool_"), ("length_class", "string"), ("meta", "string"))
SAMPLE_FILE = "steekproef-yeb-g4-8-mix.md"
MONTHS = ("jan", "feb", "mrt", "apr", "mei", "jun", "jul", "aug", "sep", "okt", "nov", "dec")


def norm(text):
    """A text with its whitespace normalized (G4-8-data.4): two translations differ if this differs"""
    return " ".join((text or "").split())


def write_parquet(rows, path, fields):
    g4rl.write_parquet(rows, path, g4rl.schema(fields))


def read_rows(path, columns=None):
    return g4arm.read_rows(path, columns)


def shuffled(items, seed):
    """The sorted items in the order of a seeded shuffle"""
    out = sorted(items)
    random.Random(seed).shuffle(out)
    return out


# ---------------------------------------------------------------------------------------------------- data

def blocked_of(tables):
    """{name: rows (id, doc, en)} -> the ids, documents, stripped English texts and English 13-grams no source may have"""
    import select_passages as sp

    out = {"id": set(), "doc": set(), "en": set(), "grams": set()}
    for rows in tables.values():
        for r in rows:
            en = (r["en"] or "").strip()
            out["id"].add(r["id"])
            out["doc"].add(r["doc"])
            out["en"].add(en)
            out["grams"] |= sp.ngrams(en)
    return out


def overlap_of(rows, blocked):
    """The sources that share an id, a document or a stripped English text with the blocked files (a hit stops)"""
    hits = {"id": 0, "doc": 0, "en": 0}
    for r in rows:
        hits["id"] += r["id"] in blocked["id"]
        hits["doc"] += r["doc"] in blocked["doc"]
        hits["en"] += (r["en"] or "").strip() in blocked["en"]
    return hits


def novel_rows(meta, b1_rows):
    """The 500 passages of the set novels with the English of their -b1- row: id novels:<pid>, doc novels:<English
    book id> (cands7's form), stratum the English book id"""
    by = collections.defaultdict(list)
    for r in b1_rows:
        by[str(r["id"])].append(r)
    out = []
    for m in meta:
        found = by.get(str(m["pid"]), [])
        if len(found) != 1 or found[0]["group"] != "en_orig" or str(found[0]["en_book_id"]) != str(m["en_book_id"]):
            raise SystemExit(f"novels {m['pid']}: {len(found)} rows in b1-selected (or not en_orig of book "
                             f"{m['en_book_id']})")
        en = (found[0]["en"] or "").strip()
        if not en:
            raise SystemExit(f"novels {m['pid']}: an empty English passage")
        book = str(m["en_book_id"])
        out.append({"id": f"novels:{m['pid']}", "doc": f"novels:{book}", "en": en, "stratum": book})
    books = {r["stratum"] for r in out}
    if len(out) != NOVEL_PAIRS or len({r["id"] for r in out}) != NOVEL_PAIRS or len(books) != NOVEL_BOOKS:
        raise SystemExit(f"the set novels: {len(out)} passages of {len(books)} books, not {NOVEL_PAIRS} of {NOVEL_BOOKS}")
    return out


def bt_rows(fb_pairs):
    """§20's BT pairs (pairjudge.fb_bt) as sources: id bt:<pid>, doc C2's document key, stratum the source"""
    import g4c2

    return [{"id": f"bt:{p['id']}", "doc": g4c2.doc_key(p["id"]), "en": p["en"], "stratum": p["stratum"]}
            for p in fb_pairs]


def draw(rows, per, seed_of):
    """Per stratum (sorted) `per` sources: its ids sorted, shuffled with seed_of(stratum), the first `per`"""
    by = collections.defaultdict(list)
    for r in rows:
        by[r["stratum"]].append(r)
    out, eligible = [], {}
    for s in sorted(by):
        eligible[s] = len(by[s])
        if len(by[s]) < per:
            raise SystemExit(f"{s}: {len(by[s])} eligible sources, {per} wanted")
        of = {r["id"]: r for r in by[s]}
        out += [of[i] for i in shuffled(of, seed_of(s))[:per]]
    return out, eligible


def segments_of(drawn):
    """(evalsets rows, segment rows, classes) of the drawn sources, the class from the English's Gemma 4 tokens"""
    ev, seg = [], []
    for part, rows in drawn.items():
        for r in rows:
            cls = g4arm.class_of(r["tokens"])
            ev.append({"id": r["id"], "set": part, "source": r["en"], "reference": None, "tokens": r["tokens"],
                       "over_512": r["tokens"] > 512, "length_class": cls,
                       "meta": json.dumps({"stratum": r["stratum"], "doc": r["doc"]})})
            seg.append({"id": r["id"], "text": r["en"], "tokens": r["tokens"]})
    if len({r["id"] for r in ev}) != len(ev):
        raise SystemExit("a source twice")
    classes = [c for c in g4arm.EVAL_CLASSES if any(r["length_class"] == c for r in ev)]
    return ev, seg, classes


def data(args, counter=None):
    import pairjudge
    import pyarrow.parquet as pq

    counter = counter or g4arm.TokenCounter(args.tokenizer)
    paths = {"g47_pool": args.g47_pool, "g48_fresh": args.g48_fresh, "g48_plan": args.g48_plan,
             "g48_targets": args.g48_targets}
    tables = {k: read_rows(p, ["id", "doc", "en"]) for k, p in paths.items()}
    blocked = blocked_of(tables)
    meta = read_rows(args.novels_meta, ["pid", "en_book_id"])
    want = sorted({str(m["pid"]) for m in meta})
    id_type = pq.read_schema(args.b1).field("id").type
    key = [int(x) for x in want] if str(id_type).startswith(("int", "uint")) else want
    b1 = pq.read_table(args.b1, columns=["id", "group", "en_book_id", "en"], filters=[("id", "in", key)]).to_pylist()
    novels = novel_rows(meta, b1)
    del b1
    c2_rows = read_rows(args.c2_data, ["pid", "source", "split", "label", "text"])
    bt_in = g4arm.read_bt(args.bt, g4arm.revisions3(args.bt), split="train", extra=("nl_back",))
    # §20's pairs and checks (C2's training documents, G4-7's pool, G4-8's plan), unchanged
    fb, fb_info = pairjudge.fb_bt(c2_rows, bt_in, tables["g47_pool"], tables["g48_plan"])
    del c2_rows, bt_in
    bt = bt_rows(fb)
    info = {"protocol": PROTOCOL, "seed": SEED, "blocked_rows": {k: len(v) for k, v in tables.items()},
            "inputs": {k: g4arm.sha256_file(p) for k, p in
                       (("novels_meta", args.novels_meta), ("b1", args.b1), ("c2_data", args.c2_data), *paths.items())}}
    drawn = {}
    for part, rows, per in (("novels", novels, NOVELS_PER_BOOK), ("bt", bt, BT_PER_SOURCE)):
        hits = overlap_of(rows, blocked)
        if any(hits.values()):
            raise SystemExit(f"{part}: sources in G4-7's pool or G4-8's fresh, plan or targets: {hits}")
        keep, drops = g4rl.candidate_filter(rows, counter, blocked["grams"])
        drawn[part], eligible = draw(keep, per, lambda s, p=part: f"{SEED}/{p}/{s}")
        info[part] = {"sources": len(rows), "strata": len({r["stratum"] for r in rows}), "overlap": hits,
                      "filter": drops, "eligible": eligible, "drawn": len(drawn[part]),
                      "ids": [r["id"] for r in drawn[part]]}
    info["bt"]["fb_bt"] = fb_info
    ev, seg, classes = segments_of(drawn)
    info["classes"] = {p: dict(collections.Counter(r["length_class"] for r in ev if r["set"] == p)) for p in drawn}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_parquet(ev, out / "evalsets.parquet", EVALSET_SCHEMA)
    write_parquet(seg, out / "segments.parquet", SEGMENT_SCHEMA)
    (out / "classes.txt").write_text(" ".join(classes) + "\n")
    (out / "data.json").write_text(json.dumps(info, indent=1) + "\n")
    print(f"MIXDATA {json.dumps({p: {k: info[p][k] for k in ('sources', 'overlap', 'filter', 'eligible', 'drawn')} for p in drawn} | {'classes': info['classes']})}",
          flush=True)


# ---------------------------------------------------------------------------------------------------- pick

def read_pairs(paths):
    """{id: row} of collected pairs (evalsets.py collect) of one model"""
    out = {}
    for f in paths:
        for r in read_rows(f, ["id", "set", "source", "hypothesis", "ended", "decode_loop", "empty"]):
            if r["id"] in out:
                raise SystemExit(f"{r['id']} twice ({f})")
            out[r["id"]] = r
    if not out:
        raise SystemExit(f"no pairs in {paths}")
    return out


def paired(by_model, what):
    """The two models' pairs checked: the same ids with the same sets and sources"""
    a, b = (by_model[m] for m in MODELS)
    if set(a) != set(b):
        raise SystemExit(f"{what}: {len(set(a) ^ set(b))} ids of one model only")
    bad = [i for i in a if (a[i]["set"], a[i]["source"]) != (b[i]["set"], b[i]["source"])]
    if bad:
        raise SystemExit(f"{what}: {len(bad)} ids whose set or source differs between the models ({bad[:3]})")
    return a, b


def differs(a, b, i):
    return norm(a[i]["hypothesis"]) != norm(b[i]["hypothesis"])


def split_assign(path):
    """G4-5.4's split (search or decision per id), checked against its digest (g4mix.digest)"""
    split = json.loads(Path(path).read_text())
    assign = split["assign"]
    got = hashlib.sha256(json.dumps(assign, sort_keys=True).encode()).hexdigest()
    if got != split["digest"] or got != g4arm.SPLIT_DIGEST6C:
        raise SystemExit(f"{path}: not the split of G4-5.4 (digest {got}, want {g4arm.SPLIT_DIGEST6C})")
    return assign


def pick_eval(a, b, evalsets, assign, excluded, used_blind):
    """{set: picked ids} and {set: counts} of FLAN, WMT24++, FLORES, NTREX (search part, not excluded) and idiom (the
    fixed blind passages' idioms outside sample 2), each only where the translations differ"""
    blind = {i: json.loads(r["meta"]) for i, r in evalsets.items() if r["set"] == "blind"}
    blind_items = {m["item"] for m in blind.values()}
    picked, counts = {}, {}
    for s in EVAL_PICK:
        ids = [i for i, r in evalsets.items() if r["set"] == s]
        pool = [i for i in ids if assign.get(i) == "search" and i not in blind_items and i not in excluded
                and not (s == "wmt24pp" and g4arm.bad_source(evalsets[i]["meta"]))]
        cands = sorted(i for i in pool if differs(a, b, i))
        if len(cands) < PICK[s]:
            raise SystemExit(f"{s}: {len(cands)} differing pairs in the search part, {PICK[s]} wanted")
        picked[s] = random.Random(f"{SEED}/pick/{s}").sample(cands, PICK[s])
        counts[s] = {"segments": len(ids), "search_not_excluded": len(pool), "differ": len(cands)}
    idioms = sorted(i for i, m in blind.items() if m["blind_set"] == "idiom")
    pool = [i for i in idioms if i not in used_blind]
    cands = sorted(i for i in pool if differs(a, b, i))
    if len(cands) < PICK["idiom"]:
        raise SystemExit(f"idiom: {len(cands)} differing pairs outside sample 2, {PICK['idiom']} wanted")
    picked["idiom"] = random.Random(f"{SEED}/pick/idiom").sample(cands, PICK["idiom"])
    counts["idiom"] = {"segments": len(idioms), "not_in_sample_2": len(pool), "differ": len(cands)}
    return picked, counts, {i: blind[i]["item"] for i in picked["idiom"]}


def pick_mix(a, b, strata):
    """{part: picked ids} and counts of the novels (one per book in a drawn order of the books, a second round only if
    fewer books than wanted have a differing pair) and BT (BT_PICK_PER_SOURCE per source)"""
    picked, counts = {}, {}
    for part in ("novels", "bt"):
        ids = [i for i in a if a[i]["set"] == part]
        by = collections.defaultdict(list)
        for i in ids:
            if differs(a, b, i):
                by[strata[i]].append(i)
        all_strata = sorted({strata[i] for i in ids})
        counts[part] = {"translated": len(ids), "differ": sum(map(len, by.values())),
                        "differ_per_stratum": {s: len(by.get(s, [])) for s in all_strata}}
        if part == "novels":
            queues = {s: shuffled(by[s], f"{SEED}/pick/novels/{s}") for s in by}
            order = [s for s in shuffled(all_strata, f"{SEED}/pick/novels") if s in queues]
            out = []
            while len(out) < PICK["novels"] and any(queues.values()):
                for s in order:
                    if queues[s] and len(out) < PICK["novels"]:
                        out.append(queues[s].pop(0))
            if len(out) < PICK["novels"]:
                raise SystemExit(f"novels: {len(out)} differing pairs, {PICK['novels']} wanted")
        else:
            out = []
            for s in all_strata:
                if len(by.get(s, [])) < BT_PICK_PER_SOURCE:
                    raise SystemExit(f"bt {s}: {len(by.get(s, []))} differing pairs, {BT_PICK_PER_SOURCE} wanted")
                out += random.Random(f"{SEED}/pick/bt/{s}").sample(sorted(by[s]), BT_PICK_PER_SOURCE)
        picked[part] = out
    return picked, counts


def ab_of(seed):
    """True if G4-8-s140 is A"""
    return random.Random(seed).random() < 0.5


def sample_text(pairs, date):
    lines = [f"# Steekproef 3 voor Yeb: het eindmodel G4-8-s140 tegen G4-0 uit alle bronnen (40 blinde paren, {date})", "",
             "Zelfde opzet als de vorige steekproeven, nu gemengd uit alle bronnen (ook romans en BT): per paar de Engelse",
             "bron en twee vertalingen, A en B. Eerst juistheid, dan natuurlijk Nederlands. Antwoord per nummer met \"A\",",
             "\"B\" of \"=\" (gelijk), eventueel \"fout A\" / \"fout B\". Sleutel apart.", ""]
    for p in pairs:
        lines += [f"## {p['n']}", "", "**Bron:**", "", p["source"].strip(), "", "**A:**", "", p["A_text"].strip(), "",
                  "**B:**", "", p["B_text"].strip(), ""]
    return "\n".join(lines)


def pick(args, today=None):
    import datetime

    evalsets = {r["id"]: r for r in read_rows(args.evalsets, ["id", "set", "meta"])}
    mix = paired({m: read_pairs([p]) for m, p in named(args.mix, "--mix").items()}, "the new translations")
    ev = paired({m: read_pairs(sorted(glob.glob(p))) for m, p in named(args.eval, "--eval").items()}, "the evaluation")
    if set(ev[0]) != set(evalsets):
        raise SystemExit(f"the evaluation's pairs are not the {len(evalsets)} segments of {args.evalsets}")
    data_rows = {r["id"]: r for r in read_rows(Path(args.data) / "evalsets.parquet", ["id", "set", "meta"])}
    if set(mix[0]) != set(data_rows):
        raise SystemExit(f"the new translations are not the {len(data_rows)} sources of {args.data}")
    strata = {i: json.loads(r["meta"])["stratum"] for i, r in data_rows.items()}
    excluded = set()
    for f in args.exclude_items:
        excluded |= {r["item"] for r in g4arm.read_jsonl(f)}
    used_blind = {r["blind_id"] for r in json.loads(Path(args.used_blind).read_text())}
    assign = split_assign(args.split)
    picked, counts, idiom_items = pick_eval(*ev, evalsets, assign, excluded, used_blind)
    pm, cm = pick_mix(*mix, strata)
    picked |= pm
    counts |= cm
    if {s: len(v) for s, v in picked.items()} != PICK:
        raise SystemExit(f"picked {({s: len(v) for s, v in picked.items()})}, not {PICK}")
    items = sorted((s, i) for s, ids in picked.items() for i in ids)
    items = random.Random(f"{SEED}/pick/order").sample(items, len(items))
    pairs, keys = [], []
    for n, (s, i) in enumerate(items, 1):
        a, b = (mix if s in ("novels", "bt") else ev)
        seed = f"{SEED}/ab/{s}/{i}"
        g48_first = ab_of(seed)
        base, own = a[i]["hypothesis"] or "", b[i]["hypothesis"] or ""
        pairs.append({"n": n, "source": a[i]["source"], "A_text": own if g48_first else base,
                      "B_text": base if g48_first else own})
        k = {"n": n, "set": s, "item": i, "g48": "A" if g48_first else "B",
             "A": MODEL8 if g48_first else "G4-0", "B": "G4-0" if g48_first else MODEL8, "seed": seed}
        if i in strata:
            k["stratum"] = strata[i]
        if s == "idiom":
            k["blind_item"] = idiom_items[i]
        for m, side in ((BASE, a), (MODEL8, b)):
            r = side[i]
            k[f"flags_{'G4-0' if m == BASE else m}"] = [f for f in ("decode_loop", "empty") if r[f]] + \
                ([] if r["ended"] else ["not_ended"])
        keys.append(k)
    today = today or datetime.date.today()
    date = f"{today.day} {MONTHS[today.month - 1]} {today.year}"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / SAMPLE_FILE).write_text(sample_text(pairs, date), encoding="utf-8")
    key = {"protocol": PROTOCOL, "file": SAMPLE_FILE, "models": {"G4-8": MODEL8, "G4-0": "G4-0 (no adapter)"},
           "answer": "per n: A, B or = (equal), optionally 'fout A' / 'fout B'", "seed": SEED, "pairs": keys,
           "per_set": dict(collections.Counter(k["set"] for k in keys)), "g48_is_a": sum(k["g48"] == "A" for k in keys)}
    (out / "key.json").write_text(json.dumps(key, indent=1) + "\n")
    in_sample = {(k["set"], k["item"]): k["n"] for k in keys}
    inputs, jkeys, report = judge_package(*mix, strata, in_sample)
    with open(out / "judge-input.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in inputs)
    with open(out / "judge-key.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(x) + "\n" for x in jkeys)
    (out / "select-report.json").write_text(json.dumps(report, indent=1) + "\n")
    info = {"protocol": PROTOCOL, "seed": SEED, "counts": counts, "picked": picked, "g48_is_a": key["g48_is_a"],
            "excluded_items": len(excluded), "used_blind": len(used_blind),
            "flags_new": {("G4-0" if m == BASE else m): {f: sum(bool(r[f]) for r in side.values())
                                                         for f in ("decode_loop", "empty")}
                          | {"not_ended": sum(not r["ended"] for r in side.values())}
                          for m, side in zip(MODELS, mix)},
            "package": {k: report[k] for k in ("n", "per_set", "a_is_base")}}
    (out / "pick.json").write_text(json.dumps(info, indent=1) + "\n")
    print(f"MIXPICK {json.dumps({'counts': counts, 'g48_is_a': key['g48_is_a'], 'flags_new': info['flags_new'], 'package': info['package']})}",
          flush=True)


def judge_package(a, b, strata, in_sample):
    """(inputs, keys, report) of the blind package G4-8-romans-bt: every new pair whose translations differ, in a seeded
    order, A/B per blind_id seeded (as G4-8-V3)"""
    ids = sorted(i for i in a if differs(a, b, i))
    ids = random.Random(f"{SEED}/judge/order").sample(ids, len(ids))
    inputs, keys = [], []
    for n, i in enumerate(ids, 1):
        bid = f"rb-{n:03d}"
        seed = f"{SEED}/judge/ab/{bid}"
        base_first = not ab_of(seed)
        base, own = a[i]["hypothesis"] or "", b[i]["hypothesis"] or ""
        inputs.append({"blind_id": bid, "source": a[i]["source"], "A": base if base_first else own,
                       "B": own if base_first else base})
        keys.append({"blind_id": bid, "A": BASE if base_first else MODEL8, "B": MODEL8 if base_first else BASE,
                     "seed": seed, "set": a[i]["set"], "item": i, "stratum": strata[i],
                     "steekproef_n": in_sample.get((a[i]["set"], i))})
    per_set = dict(sorted(collections.Counter(k["set"] for k in keys).items()))
    report = {"protocol": PROTOCOL, "target": MODEL8, "base": "G4-0 (no adapter)", "n": len(keys), "per_set": per_set,
              "translated": dict(sorted(collections.Counter(r["set"] for r in a.values()).items())),
              "a_is_base": sum(k["A"] == BASE for k in keys),
              "in_steekproef": sum(k["steekproef_n"] is not None for k in keys)}
    return inputs, keys, report


def named(specs, what):
    out = {}
    for spec in specs:
        name, _, value = spec.partition("=")
        if name not in MODELS or not value or name in out:
            raise SystemExit(f"{what} takes {' and '.join(f'{m}=…' for m in MODELS)} once each, not {spec!r}")
        out[name] = value
    if set(out) != set(MODELS):
        raise SystemExit(f"{what}: {sorted(out)}, want {list(MODELS)}")
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("data")
    for a in ("novels-meta", "b1", "g47-pool", "g48-fresh", "g48-plan", "g48-targets", "c2-data", "bt", "tokenizer",
              "out"):
        p.add_argument(f"--{a}", required=True)
    p = sub.add_parser("pick")
    for a in ("data", "evalsets", "split", "used-blind", "out"):
        p.add_argument(f"--{a}", required=True)
    p.add_argument("--mix", action="append", required=True)
    p.add_argument("--eval", action="append", required=True)
    p.add_argument("--exclude-items", action="append", required=True)
    args = parser.parse_args(argv)
    {"data": data, "pick": pick}[args.cmd](args)


if __name__ == "__main__":
    main()
