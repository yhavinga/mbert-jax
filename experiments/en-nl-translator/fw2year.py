#!/usr/bin/env python3
"""Are FineWeb-2 texts of later crawls more often not original Dutch? The classifier "original or translated" of rule
(b) (protocol.md §4.2.1, S @ 653k fine-tuned on 10 Nov, accuracy 0.9575) per crawl year of the Dutch side of
yhavinga/fineweb-2-nl-backtranslated-en (Yeb, 13 Nov: whether the back-translation should leave out FineWeb-2 after
some crawl). Measures and reports; decides nothing.

    python fw2year.py data --classifier-data classifier.parquet --tokenizer DIR --wordlist F --idioms F --out DIR
    python fw2year.py report --meta meta.parquet --predictions predictions.parquet --data-json data.json \
        --classifier-results results.json [--bt-sources summary.json] --commit C --out report.md --json report.json
    python fw2year.py g4texts --base-texts arms-texts.parquet --texts NAME=texts.parquet ... --out texts.parquet
    python fw2year.py g4report --predictions predictions.parquet --names names.json --classifier-results results.json \
        --out g4.md --json g4.json

data: the dataset's train and validation splits at a fixed revision (BT_REPO); a passage's crawl is its column `dump`
(CC-MAIN-YYYY-WW), its crawl year YYYY, an upper bound of the year it was written. Passages of the classifier's own
data (its 2,000 eval passages, by pid or text) are left out. A crawl year with fewer than MIN_PER_YEAR passages gets a
supplement up to MIN_PER_YEAR: passages of HuggingFaceFW/fineweb-2 nld_Latn of that year (the revision and the 45 files
select_passages.py read), cut and filtered as select_passages.py cuts FineWeb-2 (Builder, the page's address,
paragraph_breaks), from documents outside the dataset, drawn with a fixed seed: the row groups whose crawls (the row
group's min/max of `dump`; the files are ordered by crawl in runs) touch the year, in a seeded random order, read
SUPPLEMENT_ROW_GROUPS at a time until SUPPLEMENT_MAX_DOCS documents of that year are in (a thin year needs more row
groups: 2015 has few), then SUPPLEMENT_MAX_DOCS of them at random, one passage per document, LONG_SHARE of them long. Written: texts.parquet
(model = group "bt" or "extra", set = crawl, id, text; what naturalness.py classify reads), meta.parquet (id, group,
dump, year, half, length_class, tokens, url) and data.json (counts per year and group, what was left out, what was read).
report: the share "original" (P(original) >= 0.5) per crawl year and group with its 95% Wilson interval and n, per crawl
when n >= PER_CRAWL_MIN, the rule fixed before the measurement (RULE_NL) and the first year it holds for.
g4texts / g4report: the smaller Gemma 4 models' translations (WMT24++ 960 and FLAN 1,000, as armeval.py texts writes
them) beside TranslateGemma's (model base of the arms' texts), and rule (b) for each as RE's reclassify computes it for
the 31B (armeval.classifier_shares, pooled share minus the base model's, >= RULE_B).
"""

import argparse
import collections
import hashlib
import json
import math
import random
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import select_passages as sp  # noqa: E402

BT_REPO = ("yhavinga/fineweb-2-nl-backtranslated-en", "763f2cbc76f326b221357aca7c0e9534f7a0d57d")
BT_FILES = [f"data/train-{i:05d}-of-00003.parquet" for i in range(3)] + ["data/validation-00000-of-00001.parquet"]
BT_COLUMNS = ["id", "fineweb2_id", "url", "dump", "date", "nl", "half", "length_class", "tokens_nl"]
BT_ROWS = 59408  # train 58,252 + validation 1,156
# The resources select_passages.py cut FineWeb-2 with (mainset pool-fineweb/report.json, 9 Nov)
WORDLIST_SHA256 = "12e5fb5e3c73840b583b30016926d1f63a75e9bf1652a3a6634b2ba7c49ad7be"
IDIOMS_SHA256 = "bccb353244293c83d9032163e076dacc75dcf00a2fd26d6b99a8b8d86997141c"
FINEWEB_COLUMNS = ["id", "text", "url", "dump", "date"]
MIN_PER_YEAR = 1000
SUPPLEMENT_ROW_GROUPS = 30
SUPPLEMENT_MAX_DOCS = 6000
SEED = "fw2year"
PER_CRAWL_MIN = 200
DUMP = re.compile(r"^CC-MAIN-(\d{4})-(\d{2})$")
Z95 = 1.959963984540054

# The rule, fixed before the measurement (the main session's wording for Yeb, made exact here before any passage was
# classified; this text goes into the report unchanged)
BASELINE_YEARS = (2017, 2018, 2019, 2020, 2021)
DROP_PP = 0.05
RULE_FIXED_AT = "13 nov 2026 11:31 CET"
RULE_NL = (
    "Een daling: het aandeel \"origineel\" (kans >= 0,5) van een crawljaar ligt meer dan 5 procentpunt onder het "
    "gemiddelde van 2017-2021 én het 95%-Wilson-interval van dat jaar overlapt niet met dat van 2017-2021. "
    "Het gemiddelde van 2017-2021 is het aandeel over alle BT-passages met crawljaar 2017 tot en met 2021 samen "
    "(k/n samengenomen, dus naar aantal passages gewogen), met zijn eigen Wilson-interval; 'overlapt niet' betekent: "
    "de bovengrens van het jaar ligt onder de ondergrens van 2017-2021. De regel wordt per crawljaar in "
    "chronologische volgorde toegepast op de BT-passages (de groep waarover beslist wordt); het eerste jaar waarvoor "
    "hij geldt wordt gerapporteerd, of 'geen'. De aanvulpassages uit FineWeb-2 (alleen jaren met minder dan 1.000 "
    "BT-passages) worden apart tegen dezelfde basis getoetst en apart gerapporteerd. Kans >= 0,5 als 'origineel', "
    "95%-Wilson met z = 1,96. De regel beslist niets.")


def crawl_year(dump):
    m = DUMP.match(dump or "")
    if not m:
        raise ValueError(f"not a crawl: {dump!r}")
    return int(m.group(1))


def wilson(k, n, z=Z95):
    """(low, high) of the Wilson score interval of k successes in n"""
    if n <= 0:
        raise ValueError("wilson: n must be positive")
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def stat(flags):
    """{n, k, share, low, high} of a list of booleans (P(original) >= 0.5)"""
    n, k = len(flags), sum(flags)
    low, high = wilson(k, n)
    return {"n": n, "k": k, "share": k / n, "low": low, "high": high}


def decline(year, base):
    """The fixed rule for one year's stat against the baseline's: (holds, below by more than DROP_PP, intervals apart)"""
    below = year["share"] < base["share"] - DROP_PP
    apart = year["high"] < base["low"]
    return below and apart, below, apart


def first_decline(per_year, base):
    """The first year (in order) the rule holds for, or None; per_year {year: stat}"""
    for year in sorted(per_year):
        if decline(per_year[year], base)[0]:
            return year
    return None


def supplement_needs(counts, min_per_year=MIN_PER_YEAR):
    """{year: passages wanted} for every year with fewer than min_per_year passages"""
    return {y: min_per_year - n for y, n in sorted(counts.items()) if n < min_per_year}


def groups_touching(groups, year):
    """The row groups [(file, group, min dump, max dump)] whose crawls touch the year"""
    return [g for g in groups if crawl_year(g[2]) <= year <= crawl_year(g[3])]


def read_in_batches(groups, batch, read, enough):
    """Reads the groups batch by batch in their order (read(batch) -> (docs, counts)) until enough(docs) or none are
    left: (docs, counts, the groups read)"""
    docs, counts, done = [], collections.Counter(), []
    for k in range(0, len(groups), batch):
        part = sorted(groups[k:k + batch])
        found, c = read(part)
        docs += found
        counts.update(c)
        done += part
        if enough(docs):
            break
    return docs, dict(counts), done


def classifier_exclusions(bt_rows, classifier_rows):
    """The ids of the dataset's passages that are in the classifier's data (its pid, or its text as label 1), with
    the count of each way"""
    pids = {r["pid"] for r in classifier_rows}
    texts = {(r["text"] or "").strip() for r in classifier_rows if int(r["label"]) == 1}
    by_pid = {r["id"] for r in bt_rows if r["id"] in pids}
    by_text = {r["id"] for r in bt_rows if (r["nl"] or "").strip() in texts}
    return by_pid | by_text, {"by_pid": len(by_pid), "by_text": len(by_text), "either": len(by_pid | by_text)}


def draw_supplement(passages, n, year, seed=SEED):
    """n of the passages that pass every filter, one per document, LONG_SHARE of them long (as select_passages.py's
    length classes), drawn with a fixed seed; fails when a class has too few"""
    ok = [p for p in passages if not p["fails"]]
    rng = random.Random(f"{seed}/{year}/draw")
    chooser = sp.Chooser(1)
    n_long = round(sp.LONG_SHARE * n)
    chosen = []
    for name, k in (("long", n_long), ("short", n - n_long)):
        pool = sorted((p for p in ok if sp.length_class(p) == name), key=lambda p: p["pid"])
        picked = chooser.take(pool, k, rng)
        if len(picked) < k:
            raise SystemExit(f"{year}: only {len(picked)} {name} passages for {k} (pool {len(pool)}, skipped "
                             f"{dict(chooser.skipped)}): read more row groups (--row-groups) or documents (--max-docs)")
        for p in picked:
            p["length_class"] = name
        chosen += picked
    return chosen


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------------------------------- data

def read_bt(fs):
    import pyarrow.parquet as pq

    repo, revision = BT_REPO
    rows = []
    for name in BT_FILES:
        with fs.open(f"datasets/{repo}@{revision}/{name}", "rb", block_size=1 << 23) as f:
            part = pq.read_table(f, columns=BT_COLUMNS).to_pylist()
        split = "validation" if "validation" in name else "train"
        for r in part:
            r["split"] = split
        rows += part
    if len(rows) != BT_ROWS or len({r["id"] for r in rows}) != BT_ROWS:
        raise SystemExit(f"{repo}@{revision}: {len(rows)} rows ({len({r['id'] for r in rows})} ids), expected {BT_ROWS}")
    return rows


def fineweb_groups(fs):
    """[(file, group, min dump, max dump)] of every row group of the 45 files, from the files' footers"""
    import pyarrow.parquet as pq

    repo, revision = sp.FINEWEB
    out = []
    for name in sp.SOURCES["fineweb"]["files"]:
        with fs.open(f"datasets/{repo}@{revision}/{name}", "rb") as f:
            md = pq.ParquetFile(f).metadata
        col = md.schema.names.index("dump")
        for g in range(md.num_row_groups):
            s = md.row_group(g).column(col).statistics
            if s is None or not s.has_min_max:
                raise SystemExit(f"{name} row group {g}: no min/max of dump")
            out.append((name, g, s.min, s.max))
    return out


def read_year_docs(fs, groups, year, skip_ids):
    """The documents of these row groups crawled in the year whose page passes select_passages.py's address check and
    that are not in the dataset, with the counts"""
    import pyarrow.parquet as pq

    repo, revision = sp.FINEWEB
    counters, docs = collections.Counter(), []
    by_file = collections.defaultdict(list)
    for name, g, _, _ in groups:
        by_file[name].append(g)
    for name, gs in sorted(by_file.items()):
        with fs.open(f"datasets/{repo}@{revision}/{name}", "rb", block_size=1 << 23) as f:
            table = pq.ParquetFile(f).read_row_groups(sorted(gs), columns=FINEWEB_COLUMNS)
        for doc in table.to_pylist():
            counters["docs_read"] += 1
            if crawl_year(doc["dump"]) != year:
                counters["docs_other_year"] += 1
                continue
            reason = sp.fineweb_url(doc["url"])
            if reason:
                counters[f"docs_url_{reason}"] += 1
                continue
            if doc["id"] in skip_ids:
                counters["docs_in_dataset"] += 1
                continue
            docs.append(doc)
    counters["docs_kept"] = len(docs)
    return docs, dict(counters)


def data(args):
    import concurrent.futures
    import multiprocessing

    import pyarrow as pa
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    started = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.glob("*.parquet")):
        raise SystemExit(f"{out} has parquet files already: a new run goes to a new directory")
    for path, want in ((args.wordlist, WORDLIST_SHA256), (args.idioms, IDIOMS_SHA256)):
        if sha256(path) != want:
            raise SystemExit(f"{path}: sha256 is not select_passages.py's resource ({want})")
    fs = HfFileSystem()
    bt = read_bt(fs)
    classifier_rows = pq.read_table(args.classifier_data, columns=["pid", "label", "text"]).to_pylist()
    left_out, how = classifier_exclusions(bt, classifier_rows)
    kept = [r for r in bt if r["id"] not in left_out]
    for r in kept:
        r["year"] = crawl_year(r["dump"])
    counts = collections.Counter(r["year"] for r in kept)
    needs = supplement_needs(counts, args.min_per_year)
    print(f"BT {len(bt)} rows, {len(left_out)} in the classifier's data left out ({how}); per year "
          f"{json.dumps(dict(sorted(counts.items())))}; supplement {json.dumps(needs)} ({time.time() - started:.0f} s)",
          flush=True)

    supplement, read = [], {}
    if needs:
        groups = fineweb_groups(fs)
        bt_docs = {r["fineweb2_id"] for r in bt}
        ctx = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(args.processes, mp_context=ctx, initializer=sp._init,
                                                    initargs=(args.tokenizer, args.wordlist, args.idioms)) as pool:
            for year, need in needs.items():
                touching = groups_touching(groups, year)
                order = list(touching)
                random.Random(f"{SEED}/{year}/groups").shuffle(order)
                docs, c, chosen_groups = read_in_batches(
                    order, args.row_groups, lambda part: read_year_docs(fs, part, year, bt_docs),
                    lambda found: len(found) >= args.max_docs)
                c["docs_kept"] = len(docs)
                docs.sort(key=lambda d: d["id"])
                if len(docs) > args.max_docs:
                    docs = random.Random(f"{SEED}/{year}/docs").sample(docs, args.max_docs)
                    c["docs_sampled"] = len(docs)
                tasks = [("fineweb", docs[k:k + 50]) for k in range(0, len(docs), 50)]
                passages = [p for _, found, _ in pool.map(sp._work, tasks) for p in found]
                drawn = draw_supplement(passages, need, year)
                dumps = {d["id"]: d for d in docs}
                for p in drawn:
                    doc = dumps[p["doc_id"]]
                    supplement.append({"id": f"extra:{p['pid']}", "group": "extra", "dump": doc["dump"],
                                       "year": year, "half": "random", "length_class": p["length_class"],
                                       "tokens": p["tokens"], "url": doc["url"], "date": doc["date"],
                                       "split": "fineweb-2", "text": sp.paragraph_breaks(p["text"])})
                read[year] = {"row_groups_touching": len(touching), "row_groups_read": [[n, g] for n, g, _, _ in chosen_groups],
                              "documents": c, "passages": len(passages),
                              "passed": sum(not p["fails"] for p in passages), "drawn": len(drawn)}
                print(f"supplement {year}: {len(chosen_groups)} of {len(touching)} row groups, {c}, "
                      f"{len(passages)} passages, {read[year]['passed']} pass, {len(drawn)} drawn "
                      f"({time.time() - started:.0f} s)", flush=True)

    rows = [{"id": r["id"], "group": "bt", "dump": r["dump"], "year": r["year"], "half": r["half"],
             "length_class": r["length_class"], "tokens": r["tokens_nl"], "url": r["url"], "date": r["date"],
             "split": r["split"], "text": r["nl"]} for r in kept] + supplement
    if len({r["id"] for r in rows}) != len(rows):
        raise SystemExit("an id twice among the texts")
    if any(not (r["text"] or "").strip() for r in rows):
        raise SystemExit("an empty text")
    pq.write_table(pa.Table.from_pylist([{"model": r["group"], "set": r["dump"], "id": r["id"], "text": r["text"]}
                                         for r in rows]), out / "texts.parquet")
    pq.write_table(pa.Table.from_pylist([{k: v for k, v in r.items() if k != "text"} for r in rows]),
                   out / "meta.parquet")
    per_year = collections.defaultdict(dict)
    for (g, y), n in sorted(collections.Counter((r["group"], r["year"]) for r in rows).items()):
        per_year[str(y)][g] = n
    report = {
        "bt": {"repo": BT_REPO[0], "revision": BT_REPO[1], "rows": len(bt), "kept": len(kept),
               "left_out_classifier": how},
        "fineweb": {"repo": sp.FINEWEB[0], "revision": sp.FINEWEB[1], "min_per_year": args.min_per_year,
                    "row_groups_batch": args.row_groups, "max_docs_per_year": args.max_docs, "seed": SEED,
                    "needs": {str(y): n for y, n in needs.items()}, "read": {str(y): v for y, v in read.items()}},
        "per_year": dict(per_year), "texts": len(rows), "commit": args.commit,
        "resources": {"wordlist": WORDLIST_SHA256, "idioms": IDIOMS_SHA256},
        "rule": {"text": RULE_NL, "fixed_at": RULE_FIXED_AT}, "seconds": round(time.time() - started),
    }
    (out / "data.json").write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    print(f"FW2_DATA {json.dumps(report['per_year'])} texts {len(rows)} ({report['seconds']} s)", flush=True)


# ---------------------------------------------------------------------------------------------------- report

def pct(x):
    return f"{100 * x:.1f}%".replace(".", ",")


def interval(s):
    return f"[{pct(s['low'])}, {pct(s['high'])}]"


def year_check(meta, predictions):
    """{group: {year: stat}}, {group: {dump: stat}}, the baseline's stat and the halves of the BT passages"""
    p = {r["id"]: r["p_original"] >= 0.5 for r in predictions}
    if set(p) != {r["id"] for r in meta}:
        raise SystemExit(f"predictions and meta disagree: {len(set(p) ^ {r['id'] for r in meta})} ids")
    years, dumps, halves = (collections.defaultdict(lambda: collections.defaultdict(list)) for _ in range(3))
    base = []
    for r in meta:
        years[r["group"]][r["year"]].append(p[r["id"]])
        dumps[r["group"]][r["dump"]].append(p[r["id"]])
        if r["group"] == "bt":
            halves[r["half"]][r["year"]].append(p[r["id"]])
            if r["year"] in BASELINE_YEARS:
                base.append(p[r["id"]])
    if not base:
        raise SystemExit(f"no BT passages of {BASELINE_YEARS}")
    per_year = {g: {y: stat(v) for y, v in sorted(d.items())} for g, d in years.items()}
    per_dump = {g: {k: stat(v) for k, v in sorted(d.items()) if len(v) >= PER_CRAWL_MIN} for g, d in dumps.items()}
    per_half = {h: {y: stat(v) for y, v in sorted(d.items())} for h, d in halves.items()}
    return per_year, per_dump, stat(base), per_half


def report(args):
    import pyarrow.parquet as pq

    meta = pq.read_table(args.meta).to_pylist()
    predictions = pq.read_table(args.predictions).to_pylist()
    data_json = json.loads(Path(args.data_json).read_text())
    accuracy = json.loads(Path(args.classifier_results).read_text())["test_accuracy"]
    if data_json["rule"]["text"] != RULE_NL or data_json["rule"]["fixed_at"] != RULE_FIXED_AT:
        raise SystemExit("the rule in data.json is not this code's RULE_NL: the rule changed after the data")
    per_year, per_dump, base, per_half = year_check(meta, predictions)
    unweighted = sum(per_year["bt"][y]["share"] for y in BASELINE_YEARS) / len(BASELINE_YEARS)
    first = {g: first_decline(per_year[g], base) for g in sorted(per_year)}
    truncated = sum(bool(r.get("truncated")) for r in predictions)

    L = ["# FineWeb-2 per crawljaar: aandeel \"origineel\" volgens de classifier van regel (b)", "",
         f"Gemaakt {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())} door `fw2year.py report`, mbert-jax "
         f"`en-nl-fw2year` @ `{args.commit}`. Meet en rapporteert; beslist niets.", "",
         "## De regel, vastgelegd vóór de meting", "",
         f"Vastgelegd {RULE_FIXED_AT}, vóór er één passage geclassificeerd was (de code met deze tekst stond toen al "
         f"in `fw2year.py`; ook in `data.json` van de data, gemaakt vóór de meting):", "", f"> {RULE_NL}", "",
         "## Uitslag volgens de regel", "",
         f"- Basis 2017-2021 (BT, samengenomen): {pct(base['share'])} origineel ({base['k']} van {base['n']}), "
         f"Wilson {interval(base)}; drempel 5 pp lager: {pct(base['share'] - DROP_PP)}. Ter informatie, geen deel "
         f"van de regel: het ongewogen gemiddelde van de vijf jaren is {pct(unweighted)}.",
         f"- **Eerste \"dalende\" crawljaar (BT-passages): {first.get('bt') or 'geen'}.**"]
    if "extra" in first:
        L.append(f"- Aanvulpassages uit FineWeb-2 (apart, alleen jaren met minder dan {MIN_PER_YEAR} BT-passages): "
                 f"eerste jaar waarvoor de regel geldt: {first['extra'] or 'geen'}.")
    L += ["", "## Per crawljaar", "",
          "| crawljaar | groep | n | origineel | aandeel | 95%-Wilson | > 5 pp onder basis | intervallen gescheiden | daling |",
          "|---|---|---:|---:|---:|---|---|---|---|"]
    names = {"bt": "BT", "extra": "aanvulling"}
    yes = {True: "ja", False: "nee"}
    for year in sorted({y for d in per_year.values() for y in d}):
        for g in ("bt", "extra"):
            s = per_year.get(g, {}).get(year)
            if s is None:
                continue
            holds, below, apart = decline(s, base)
            mark = " (basis)" if g == "bt" and year in BASELINE_YEARS else ""
            L.append(f"| {year}{mark} | {names[g]} | {s['n']} | {s['k']} | {pct(s['share'])} | {interval(s)} | "
                     f"{yes[below]} | {yes[apart]} | **{yes[holds]}** |")
    L += ["", f"## Per crawl (n >= {PER_CRAWL_MIN})", "", "| crawl | groep | n | aandeel | 95%-Wilson |",
          "|---|---|---:|---:|---|"]
    for g in ("bt", "extra"):
        for dump, s in per_dump.get(g, {}).items():
            L.append(f"| {dump} | {names[g]} | {s['n']} | {pct(s['share'])} | {interval(s)} |")
    L += ["", "## BT-passages per helft (ter informatie)", "",
          "De BT-selectie nam per bron de helft uit de hoogste idioomklasse en de helft willekeurig "
          "(`select_passages.py` choose); de willekeurige helft is de zuiverste steekproef van FineWeb-2.", "",
          "| crawljaar | helft | n | aandeel | 95%-Wilson |", "|---|---|---:|---:|---|"]
    for year in sorted({y for d in per_half.values() for y in d}):
        for h in sorted(per_half):
            s = per_half[h].get(year)
            if s is not None:
                L.append(f"| {year} | {h} | {s['n']} | {pct(s['share'])} | {interval(s)} |")
    if args.bt_sources:
        src = json.loads(Path(args.bt_sources).read_text())["targets"]["bt"]
        L += ["", "## De vijf BT-bronnen ter vergelijking", "",
              "Dezelfde classifier op de Nederlandse doelteksten (originelen) van de BT-voorbeelden in set R2, per "
              "bron (de stap reclassify van de RE-keten, 13 nov, `$S/re/classify/summary.json`; niet opnieuw gemeten).", "",
              "| bron | n | aandeel origineel |", "|---|---:|---:|"]
        for k, v in src.items():
            L.append(f"| {'alle vijf' if k == 'all' else k[4:]} | {v['n']} | {pct(v['share'])} |")
    bt_info = data_json["bt"]
    fw = data_json["fineweb"]
    L += ["", "## Werkwijze en beperkingen", "",
          f"- Teksten: de kolom `nl` van `{bt_info['repo']}` @ `{bt_info['revision'][:12]}` (train en validation, "
          f"{bt_info['rows']} passages); crawl = kolom `dump`, crawljaar = het jaar daarin. Een pagina kan niet na "
          f"haar crawl geschreven zijn: het crawljaar is een bovengrens voor het schrijfjaar; oude pagina's in latere "
          f"crawls verdunnen een effect.",
          f"- Weggelaten: {bt_info['left_out_classifier']['either']} passages die in de trainings- of testdata van de "
          f"classifier zitten (op pid {bt_info['left_out_classifier']['by_pid']}, op tekst "
          f"{bt_info['left_out_classifier']['by_text']}).",
          f"- Aanvulling: jaren met minder dan {fw['min_per_year']} BT-passages, aangevuld tot ~{fw['min_per_year']} "
          f"uit `{fw['repo']}` nld_Latn @ `{fw['revision'][:12]}`: row groups die het jaar raken in willekeurige "
          f"volgorde (seed `{fw['seed']}`), per {fw['row_groups_batch']} gelezen tot er {fw['max_docs_per_year']} "
          f"documenten van dat jaar waren, daaruit {fw['max_docs_per_year']} willekeurig; geknipt en gefilterd als "
          f"`select_passages.py`, één passage per document, 25% lang; documenten van de dataset uitgesloten. Wat "
          f"gelezen werd: `data.json`.",
          f"- Classifier: S @ 653k, fijngetuned op 10 nov (nauwkeurigheid {accuracy:.4f}) op origineel Nederlands "
          f"tegenover de terugvertaling (nl->en->nl) van TranslateGemma 27B, op passages van dezelfde vijf bronnen. "
          f"Hij leerde vertaalstijl van TranslateGemma herkennen; of hij tekst van andere vertaalsystemen of door een "
          f"taalmodel geschreven tekst herkent, is niet gemeten.",
          f"- Uitgevoerd als `run_arms.sh classify` (NAT = de map van deze meting): `naturalness.py classify`, "
          f"maximaal 1.024 tokens; {truncated} teksten afgekapt.",
          "- De BT-passages komen per document met hoogstens 3; het Wilson-interval neemt onafhankelijke passages "
          "aan en is daardoor iets te smal."]
    Path(args.out).write_text("\n".join(L) + "\n")
    summary = {"rule": {"text": RULE_NL, "fixed_at": RULE_FIXED_AT}, "base": base, "unweighted_base": unweighted,
               "first_decline": first, "per_year": {g: {str(y): s for y, s in d.items()} for g, d in per_year.items()},
               "per_dump": per_dump, "per_half": {h: {str(y): s for y, s in d.items()} for h, d in per_half.items()},
               "truncated": truncated, "classifier_accuracy": accuracy, "commit": args.commit}
    Path(args.json).write_text(json.dumps(summary, indent=1) + "\n")
    for year in sorted(per_year["bt"]):
        s = per_year["bt"][year]
        e = per_year.get("extra", {}).get(year)
        print(f"FW2_YEAR {year} bt n={s['n']} share={s['share']:.4f} [{s['low']:.4f}, {s['high']:.4f}]"
              + (f" extra n={e['n']} share={e['share']:.4f} [{e['low']:.4f}, {e['high']:.4f}]" if e else ""), flush=True)
    print(f"FW2_RULE base={base['share']:.4f} [{base['low']:.4f}, {base['high']:.4f}] n={base['n']} "
          f"first_decline {json.dumps(first)}", flush=True)


# ---------------------------------------------------------------------------------------------------- Gemma 4 sizes

def g4_combine(base_rows, named):
    """The rows of every named texts file (one model each, not base, the base model's sets and ids) and the base
    model's rows; {name: model}"""
    keys = lambda rows: sorted((r["set"], r["id"]) for r in rows)  # noqa: E731
    base = [r for r in base_rows if r["model"] == "base"]
    if not base:
        raise SystemExit("no rows of model base in the arms' texts")
    out, models = list(base), {}
    for name, rows in named.items():
        labels = {r["model"] for r in rows}
        if len(labels) != 1 or "base" in labels:
            raise SystemExit(f"{name}: models {sorted(labels)}, expected one model other than base")
        model = labels.pop()
        if model in models.values():
            raise SystemExit(f"{name}: model {model} is also another file's")
        if keys(rows) != keys(base):
            raise SystemExit(f"{name}: its sets and ids are not the base model's")
        models[name] = model
        out += rows
    return out, models


def g4texts(args):
    import pyarrow as pa
    import pyarrow.parquet as pq

    columns = ["model", "set", "id", "text"]
    named = {}
    for item in args.texts:
        name, path = item.split("=", 1)
        named[name] = pq.read_table(path, columns=columns).to_pylist()
    rows, models = g4_combine(pq.read_table(args.base_texts, columns=columns).to_pylist(), named)
    pq.write_table(pa.Table.from_pylist(rows), args.out)
    Path(args.names).write_text(json.dumps(models, indent=1) + "\n")
    print(f"FW2_G4TEXTS {json.dumps(dict(collections.Counter(r['model'] + '/' + r['set'] for r in rows)))}", flush=True)


def g4report(args):
    import armeval
    import pyarrow.parquet as pq

    models = json.loads(Path(args.names).read_text())
    accuracy = json.loads(Path(args.classifier_results).read_text())["test_accuracy"]
    shares = armeval.classifier_shares(pq.read_table(args.predictions).to_pylist())
    base = shares["base"]
    out = {}
    L = ["## Kleinere Gemma 4-modellen als en->nl-vertaler: regel (b) tegen TranslateGemma", "",
         f"Dezelfde classifier (nauwkeurigheid {accuracy:.4f}), zoals de stap reclassify van de RE-keten hem voor "
         f"Gemma 4 31B toepaste: aandeel \"origineel\" van WMT24++ (960) en FLAN (1.000) samen, min dat van "
         f"TranslateGemma 27B (hier opnieuw gemeten; RE-keten: 19,1%); regel (b): verschil >= "
         f"{armeval.RULE_B:.2f}. Beslist niets.", "",
         "| bestand | model | WMT24++ | FLAN | samen | TranslateGemma | verschil | regel (b) |",
         "|---|---|---:|---:|---:|---:|---:|---|"]
    for name, model in sorted(models.items()):
        s = shares[model]
        delta = s["pooled"] - base["pooled"]
        holds = delta >= armeval.RULE_B - armeval.TOLERANCE
        out[name] = {"model": model, "shares": s, "base": base, "delta": delta, "holds": holds}
        L.append(f"| {name} | {model} | {pct(s['wmt24pp'])} | {pct(s['flan'])} | {pct(s['pooled'])} | "
                 f"{pct(base['pooled'])} | {f'{100 * delta:+.1f}'.replace('.', ',')} pp | "
                 f"{'gehaald' if holds else 'niet gehaald'} |")
        print("FW2_G4 " + json.dumps({"file": name, model: round(s["pooled"], 4), "base": round(base["pooled"], 4),
                                      "delta": round(delta, 4), "holds": holds, "accuracy": accuracy}), flush=True)
    Path(args.out).write_text("\n".join(L) + "\n")
    Path(args.json).write_text(json.dumps({"classifier_accuracy": accuracy, "rule_b": armeval.RULE_B,
                                           "models": out}, indent=1) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("data", help="The texts per crawl year, with a supplement for thin years")
    p.add_argument("--classifier-data", required=True, help="The classifier's data (armeval.py classifier-data)")
    p.add_argument("--tokenizer", required=True, help="TranslateGemma's tokenizer directory (tokenizer.json)")
    p.add_argument("--wordlist", required=True)
    p.add_argument("--idioms", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--min-per-year", type=int, default=MIN_PER_YEAR)
    p.add_argument("--row-groups", type=int, default=SUPPLEMENT_ROW_GROUPS)
    p.add_argument("--max-docs", type=int, default=SUPPLEMENT_MAX_DOCS)
    p.add_argument("--processes", type=int, default=32)
    p.add_argument("--commit", default="unknown")
    p = sub.add_parser("report", help="The share original per crawl year and the fixed rule")
    for name in ("meta", "predictions", "data-json", "classifier-results", "out", "json"):
        p.add_argument(f"--{name}", required=True)
    p.add_argument("--bt-sources", help="RE's reclassify summary.json (the five BT sources)")
    p.add_argument("--commit", default="unknown")
    p = sub.add_parser("g4texts", help="The Gemma 4 sizes' texts beside the base model's")
    p.add_argument("--base-texts", required=True, help="The arms' texts.parquet (model base)")
    p.add_argument("--texts", required=True, nargs="+", help="NAME=texts.parquet")
    p.add_argument("--out", required=True)
    p.add_argument("--names", required=True, help="Written: {NAME: model}")
    p = sub.add_parser("g4report", help="Rule (b) for each Gemma 4 size")
    for name in ("predictions", "names", "classifier-results", "out", "json"):
        p.add_argument(f"--{name}", required=True)
    args = parser.parse_args(argv)
    {"data": data, "report": report, "g4texts": g4texts, "g4report": g4report}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
