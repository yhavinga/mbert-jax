"""The measurement C2F (protocol-g4.md §16, 23 Nov 2026): do C2 (G4-7's reward classifier) and C1 (rule (b)'s
classifier) tell novel passages apart? Existing texts only (nothing is translated): Dutch original fiction (the held-out
set books of G4-2.5, nl_books, and G4-1's other books, nl_books31), the human Dutch translations of English novels
(-b1-, group en_orig; human) and G4-0's greedy translations of the same passages (G4-7's anchor; g40).
  texts:  the groups, whole in the encoder (at most 1,022 tokens of C1's and of C2's tokenizer), drawn with fixed
          seeds; texts.parquet (model = the group, set, id, text) for naturalness.py classify, meta.parquet (no text)
  report: the measures from the two classify runs and meta.parquet; reads no text. Reports, decides nothing."""

import argparse
import collections
import glob
import json
import random
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import g4arm  # noqa: E402
import g4c2  # noqa: E402

SEED = "c2fiction/0"
N = 500
MAX_TOKENS = g4c2.C2_MAX_TOKENS                 # 1,022 without the special tokens: whole in 1,024 (G4-7.6)
GROUPS = ("nl_books", "nl_books31", "human", "g40")
PAIRED = ("human", "g40")
CLASSIFIERS = ("C1", "C2")
AUC_PAIRS = (("nl_books", "human"), ("nl_books", "g40"), ("human", "g40"), ("nl_books31", "human"),
             ("nl_books31", "g40"), ("nl_books", "nl_books31"))
BANDS = (("<=128", 0, 128), ("129-256", 129, 256), (">256", 257, 10 ** 9))
BOOT = 2000
CHECK_TOL = 1e-3
HELDOUT_PREFIX = "heldout:books:"               # heldout:books:<G4-1's id book:books:<nl>-<en>-b1-<k>>
NOVEL_PREFIX = "novels:"                        # the anchor's novels: novels:<-b1- row id> (g4rl.novels_cands)
ANCHOR_COLUMNS = ["uid", "id", "j", "source", "hypothesis", "ended", "decode_loop", "english_left", "empty"]


def draw(rows, n, seed):
    """min(n, len(rows)) rows of rows sorted by id, random.Random(seed).sample; the result sorted by id"""
    rows = sorted(rows, key=lambda r: r["id"])
    if len({r["id"] for r in rows}) != len(rows):
        raise SystemExit(f"draw {seed}: an id twice")
    return sorted(random.Random(seed).sample(rows, min(n, len(rows))), key=lambda r: r["id"])


def fits(rows, counters, key="text"):
    """(rows whose text has at most MAX_TOKENS tokens under every counter, with their counts; lost)"""
    counts = [c([r[key] for r in rows]) for c in counters]
    keep, lost = [], 0
    for r, *ns in zip(rows, *counts):
        if max(ns) <= MAX_TOKENS:
            keep.append({**r, f"tokens_{key}": ns})
        else:
            lost += 1
    return keep, lost


def anchor_eligible(r):
    return bool(r["ended"]) and not r["decode_loop"] and not r["english_left"] and not r["empty"] \
        and bool((r["hypothesis"] or "").strip())


def build(heldout_rows, g41_pairs, heldout_books, anchor_rows, b1_rows, counters):
    """(texts, meta, eligible anchor ids, report). heldout_rows: the held-out evalsets (id, set, reference); g41_pairs:
    [(messages row, meta row)] of G4-1's set; heldout_books: the book keys of G4-2.5; anchor_rows: G4-7's anchor
    samples (ANCHOR_COLUMNS); b1_rows: the -b1- rows of the anchor's novels (id, group, en, nl); counters: C1's and
    C2's token counters (without the special tokens)."""
    report = {"seed": SEED, "n": N, "max_tokens": MAX_TOKENS, "groups": {}}
    # nl_books: the held-out set books of G4-2.5 (its references are the Dutch originals)
    books = [{"id": r["id"], "text": r["reference"], "book": g4arm.book_key(r["id"][len(HELDOUT_PREFIX):])}
             for r in heldout_rows if r["set"] == "books"]
    if not books or any(not r["id"].startswith(HELDOUT_PREFIX) for r in books):
        raise SystemExit(f"the held-out evalsets have {len(books)} rows of set books, or an id not {HELDOUT_PREFIX}…")
    if {r["book"] for r in books} - set(heldout_books):
        raise SystemExit("a row of set books is not of a held-out book of books.json")
    g41 = [{"id": m["id"], "text": x["messages"][1]["content"].strip(), "book": g4arm.book_key(m["id"])}
           for x, m in g41_pairs if m["kind"] == "book"]
    g41_text = {r["id"]: r["text"] for r in g41}
    others = [r for r in g41 if r["book"] not in set(heldout_books)]
    out = {}
    for name, rows in (("nl_books", books), ("nl_books31", others)):
        rows = [r for r in rows if r["text"].strip()]
        kept, lost = fits(rows, counters)
        out[name] = draw(kept, N, f"{SEED}/{name}")
        report["groups"][name] = {"rows": len(rows), "too_long": lost, "eligible": len(kept),
                                  "books_eligible": len({r["book"] for r in kept}), "drawn": len(out[name])}
    # the drawn references of set books against G4-1's Dutch of the same pair (nl_books31's text)
    report["nl_books_equal_to_g41"] = sum(g41_text.get(r["id"][len(HELDOUT_PREFIX):]) == r["text"]
                                          for r in out["nl_books"])
    # human and g40: G4-7's anchor (pool and RL-val), its novels, paired with the -b1- row of the same passage
    novels = [r for r in anchor_rows if r["id"].startswith(NOVEL_PREFIX)]
    if len({r["id"] for r in novels}) != len(novels) or any(r["j"] != 0 for r in novels):
        raise SystemExit("the anchor has a novel id twice, or a row with j != 0")
    ok = [r for r in novels if anchor_eligible(r)]
    by_id = collections.defaultdict(list)
    for r in b1_rows:
        by_id[r["id"]].append(r)
    pairs, english_differs = [], []
    for r in ok:
        found = by_id.get(r["id"][len(NOVEL_PREFIX):], [])
        if len(found) != 1 or found[0]["group"] != "en_orig":
            raise SystemExit(f"{r['id']}: {len(found)} rows in b1-selected (or not of group en_orig)")
        b = found[0]
        if (r["source"] or "").strip() != (b["en"] or "").strip():
            english_differs.append(r["id"])
            continue
        pairs.append({"id": r["id"], "human": b["nl"] or "", "g40": r["hypothesis"],
                      "book": "novels:" + b["id"].rsplit("-b1-", 1)[0]})
    if english_differs:
        raise SystemExit(f"{len(english_differs)} anchor rows whose English is not the -b1- row's, e.g. "
                         f"{english_differs[:3]}")
    pairs = [p for p in pairs if p["human"].strip()]
    kept_h, lost_h = fits(pairs, counters, key="human")
    kept, lost_g = fits(kept_h, counters, key="g40")
    chosen = draw(kept, N, f"{SEED}/novels")
    report["groups"]["novels"] = {"anchor_novels": len(novels), "anchor_eligible": len(ok), "pairs": len(pairs),
                                  "human_too_long": lost_h, "g40_too_long": lost_g, "eligible": len(kept),
                                  "books_eligible": len({p["book"] for p in kept}), "drawn": len(chosen)}
    out["human"] = [{"id": p["id"], "text": p["human"], "book": p["book"], "tokens_text": p["tokens_human"]}
                    for p in chosen]
    out["g40"] = [{"id": p["id"], "text": p["g40"], "book": p["book"], "tokens_text": p["tokens_g40"]}
                  for p in chosen]
    texts, meta = [], []
    for g in GROUPS:
        for r in out[g]:
            texts.append({"model": g, "set": "fiction", "id": r["id"], "text": r["text"]})
            meta.append({"group": g, "id": r["id"], "book": r["book"], "tokens_c1": r["tokens_text"][0],
                         "tokens_c2": r["tokens_text"][1]})
        report["groups"].setdefault(g, {})["texts"] = len(out[g])
        report["groups"][g]["books"] = len({r["book"] for r in out[g]})
    eligible = sorted(p["id"] for p in pairs)
    return texts, meta, eligible, report


def texts(args):
    import pyarrow.parquet as pq

    counters = [g4c2.token_counter(g4c2.tokenizer_of(c)) for c in (args.c1_config, args.c2_config)]
    heldout_rows = g4arm.read_rows(args.heldout_evalsets, columns=["id", "set", "reference"])
    g41_pairs, _ = g4arm.g41_set(args.g41)
    heldout_books = sorted(json.loads(Path(args.heldout_books).read_text())["books"])
    anchor_rows = []
    shards = sorted(glob.glob(args.anchor))
    if not shards:
        raise SystemExit(f"no anchor shards {args.anchor}")
    for s in shards:
        anchor_rows += g4arm.read_rows(s, columns=ANCHOR_COLUMNS)
    want = sorted({r["id"][len(NOVEL_PREFIX):] for r in anchor_rows if r["id"].startswith(NOVEL_PREFIX)})
    b1_rows = pq.read_table(args.b1, columns=["id", "group", "en", "nl"], filters=[("id", "in", want)]).to_pylist()
    rows, meta, eligible, report = build(heldout_rows, g41_pairs, heldout_books, anchor_rows, b1_rows, counters)
    report["inputs"] = {"anchor_shards": len(shards), "b1_sha256": g4arm.sha256_file(args.b1),
                        "heldout_evalsets_sha256": g4arm.sha256_file(args.heldout_evalsets)}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    g4arm.write_parquet(rows, out / "texts.parquet")
    g4arm.write_parquet(meta, out / "meta.parquet")
    (out / "anchor-eligible.txt").write_text("".join(f"{i}\n" for i in eligible))
    (out / "texts.json").write_text(json.dumps(report, indent=1) + "\n")
    print(f"C2F_TEXTS {json.dumps({g: v.get('texts', v.get('drawn')) for g, v in report['groups'].items()})}",
          flush=True)


# ---------------------------------------------------------------------------------------------------- report

def auc(a, b):
    """P(a > b) + ½ P(a = b) over all pairs (Mann–Whitney)"""
    import numpy as np

    a, b = np.asarray(a, dtype=np.float64), np.sort(np.asarray(b, dtype=np.float64))
    left, right = np.searchsorted(b, a, "left"), np.searchsorted(b, a, "right")
    return float((left + 0.5 * (right - left)).sum() / (len(a) * len(b)))


def interval(values):
    import numpy as np

    lo, hi = np.percentile(np.asarray(values), [2.5, 97.5])
    return [float(lo), float(hi)]


def auc_boot(a, b, paired, rng):
    import numpy as np

    a, b = np.asarray(a), np.asarray(b)
    out = []
    for _ in range(BOOT):
        if paired:
            i = rng.integers(0, len(a), len(a))
            out.append(auc(a[i], b[i]))
        else:
            out.append(auc(a[rng.integers(0, len(a), len(a))], b[rng.integers(0, len(b), len(b))]))
    return interval(out)


def band_of(n):
    return next(name for name, lo, hi in BANDS if lo <= n <= hi)


def measures(meta, preds, judge, eligible):
    """The report of C2F.4. meta: rows of meta.parquet; preds: {classifier: rows of its predictions}; judge: the anchor's
    C2 scores (uid, p_c2, logit_c2, c2_tokens, c2_overflow); eligible: the anchor's eligible novel ids."""
    import numpy as np

    keys = {(m["group"], m["id"]) for m in meta}
    if len(keys) != len(meta):
        raise SystemExit("meta.parquet has a (group, id) twice")
    score = {}
    trunc = {}
    for c in CLASSIFIERS:
        got = {(r["model"], r["id"]): r for r in preds[c] if r["set"] == "fiction"}
        if set(got) != keys or len(preds[c]) != len(meta):
            raise SystemExit(f"{c}'s predictions are not the texts of meta.parquet")
        score[c] = {k: got[k]["p_original"] for k in keys}
        trunc[c] = [k for k in keys if got[k]["truncated"]]
    tokens = {(m["group"], m["id"]): m["tokens_c2"] for m in meta}
    tokens_c1 = {(m["group"], m["id"]): m["tokens_c1"] for m in meta}
    for c in CLASSIFIERS:
        bad = [k for k in trunc[c] if max(tokens[k], tokens_c1[k]) != MAX_TOKENS]
        if bad:
            raise SystemExit(f"{c}: {len(bad)} texts marked truncated with fewer than {MAX_TOKENS} tokens, e.g. {bad[:3]}")
    ids = {g: sorted(m["id"] for m in meta if m["group"] == g) for g in GROUPS}
    books = {g: len({m["book"] for m in meta if m["group"] == g}) for g in GROUPS}
    if ids["human"] != ids["g40"]:
        raise SystemExit("human and g40 are not the same passages")
    r = {"groups": {}, "auc": {}, "paired": {}, "bands": {}, "spearman": {}, "checks": {}}
    for g in GROUPS:
        r["groups"][g] = {"n": len(ids[g]), "books": books[g],
                          "median_tokens": statistics.median(tokens[(g, i)] for i in ids[g]) if ids[g] else None}
        for c in CLASSIFIERS:
            p = [score[c][(g, i)] for i in ids[g]]
            lg = [g4c2.logit_of(x) for x in p]
            r["groups"][g][c] = {"share_original": sum(x >= 0.5 for x in p) / len(p), "mean_p": statistics.fmean(p),
                                 "mean_logit": statistics.fmean(lg), "median_logit": statistics.median(lg),
                                 "share_p_at_1": sum(x >= 1.0 for x in p) / len(p)} if p else None
    rng = np.random.default_rng(int.from_bytes(f"{SEED}/boot".encode(), "big") % (2 ** 63))
    for a, b in AUC_PAIRS:
        if not ids[a] or not ids[b]:
            continue
        paired = (a, b) == PAIRED
        for c in CLASSIFIERS:
            xa = [score[c][(a, i)] for i in ids[a]]
            xb = [score[c][(b, i)] for i in ids[b]]
            r["auc"].setdefault(f"{a}-{b}", {})[c] = {"auc": auc(xa, xb), "ci95": auc_boot(xa, xb, paired, rng),
                                                      "paired": paired}
    h, g0 = PAIRED
    for c in CLASSIFIERS:
        ph = np.array([score[c][(h, i)] for i in ids[h]])
        pg = np.array([score[c][(g0, i)] for i in ids[h]])
        d = np.array([g4c2.logit_of(x) for x in ph]) - np.array([g4c2.logit_of(x) for x in pg])
        boot = [float(d[rng.integers(0, len(d), len(d))].mean()) for _ in range(BOOT)]
        r["paired"][c] = {"pairs": len(d), "share_human_higher": float(((ph > pg) + 0.5 * (ph == pg)).mean()),
                          "mean_dlogit": float(d.mean()), "ci95": interval(boot), "median_dlogit": float(np.median(d))}
    for name, _, _ in BANDS:
        r["bands"][name] = {}
        for g in GROUPS:
            ks = [(g, i) for i in ids[g] if band_of(tokens[(g, i)]) == name]
            r["bands"][name][g] = {"n": len(ks), **{c: (sum(score[c][k] >= 0.5 for k in ks) / len(ks) if ks else None)
                                                    for c in CLASSIFIERS}}
    for g in GROUPS:
        ks = [(g, i) for i in ids[g]]
        r["spearman"][g] = {**{f"{c}_logit_tokens": g4c2.spearman([g4c2.logit_of(score[c][k]) for k in ks],
                                                                    [tokens[k] for k in ks]) for c in CLASSIFIERS},
                            "C1_C2": g4c2.spearman([score["C1"][k] for k in ks], [score["C2"][k] for k in ks])}
    every = sorted(keys)
    r["spearman"]["all"] = {"C1_C2": g4c2.spearman([score["C1"][k] for k in every], [score["C2"][k] for k in every])}
    # checks: C2 on g40 against the anchor's judge score; the tokens against c2_tokens; the anchor's reference row
    jd = {j["uid"]: j for j in judge}
    diffs, tok_equal, missing = [], 0, 0
    for i in ids[g0]:
        j = jd.get(f"{i}#0")
        if j is None or j["p_c2"] is None:
            missing += 1
            continue
        diffs.append(abs(score["C2"][(g0, i)] - j["p_c2"]))
        tok_equal += j["c2_tokens"] == tokens[(g0, i)]
    r["checks"] = {
        "c2_vs_judge_on_g40": {"compared": len(diffs), "missing": missing, "max_abs_dp": max(diffs) if diffs else None,
                               "over_tol": sum(x > CHECK_TOL for x in diffs), "tol": CHECK_TOL,
                               "tokens_equal": tok_equal},
        "truncated_flags": {c: len(trunc[c]) for c in CLASSIFIERS}}
    ref = [jd[f"{i}#0"] for i in eligible if f"{i}#0" in jd and jd[f"{i}#0"]["p_c2"] is not None]
    r["anchor_novels_judge_c2"] = {
        "eligible": len(eligible), "scored": len(ref),
        "share_original": sum(j["p_c2"] >= 0.5 for j in ref) / len(ref) if ref else None,
        "mean_logit": statistics.fmean(max(-g4c2.LOGIT_CLIP, min(g4c2.LOGIT_CLIP, j["logit_c2"])) for j in ref)
        if ref else None}
    return r


def f(x, d=3):
    if x is None:
        return "–"
    if isinstance(x, float):
        return f"{x:.{d}f}".replace(".", ",")
    return str(x)


def report_md(r, info):
    L = ["# C2 en C1 op fictie (meting C2F, `protocol-g4.md` §16)", "",
         "Rapporteert, beslist niets. Geen teksten; alleen tellingen en scores. P = P(origineel); "
         "\"origineel\" = P ≥ 0,5; logit = clip(logit, −12, 12) uit P (als `c2cal7`).", "",
         "Groepen: `nl_books` = Nederlandse originelen, set `books` (G4-2.5); `nl_books31` = Nederlandse originelen, "
         "de andere boeken van G4-1's boekenset; `human` = menselijke Nederlandse vertalingen van Engelse romans (-b1-, "
         "`en_orig`); `g40` = G4-0 greedy van dezelfde passages (anker van G4-7). `human` en `g40` zijn gepaard.", "",
         "## 1. Per groep", "",
         "| groep | n | boeken | mediaan tokens | C1 origineel | C1 gem. P | C1 gem. logit | C1 med. logit | "
         "C2 origineel | C2 gem. P | C2 gem. logit | C2 med. logit |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for g in GROUPS:
        x = r["groups"][g]
        cells = []
        for c in CLASSIFIERS:
            y = x[c] or {}
            cells += [f(y.get("share_original")), f(y.get("mean_p")), f(y.get("mean_logit"), 2),
                      f(y.get("median_logit"), 2)]
        L.append(f"| {g} | {x['n']} | {x['books']} | {f(x['median_tokens'], 1)} | " + " | ".join(cells) + " |")
    L += ["", "## 2. AUC (de eerste groep hoort hoger; `nl_books`–`nl_books31` is een controle)", "",
          "| paar | C1 AUC | C1 95%-interval | C2 AUC | C2 95%-interval |", "|---|---:|---|---:|---|"]
    for k, v in r["auc"].items():
        cells = []
        for c in CLASSIFIERS:
            cells += [f(v[c]["auc"]), f"[{f(v[c]['ci95'][0])}; {f(v[c]['ci95'][1])}]"]
        L.append(f"| {k}{' (gepaard)' if v['C1']['paired'] else ''} | " + " | ".join(cells) + " |")
    L += ["", "## 3. Gepaard `human` – `g40` (dezelfde passages)", "",
          "| classifier | paren | aandeel P(human) > P(g40) | gem. Δlogit | 95%-interval | mediaan Δlogit |",
          "|---|---:|---:|---:|---|---:|"]
    for c in CLASSIFIERS:
        v = r["paired"][c]
        L.append(f"| {c} | {v['pairs']} | {f(v['share_human_higher'])} | {f(v['mean_dlogit'], 2)} | "
                 f"[{f(v['ci95'][0], 2)}; {f(v['ci95'][1], 2)}] | {f(v['median_dlogit'], 2)} |")
    L += ["", "## 4. Aandeel \"origineel\" per lengteband (tokens van C2's tokenizer)", "",
          "| band | " + " | ".join(f"{g} n | {g} C1 | {g} C2" for g in GROUPS) + " |",
          "|---|" + "---:|" * (3 * len(GROUPS))]
    for name, v in r["bands"].items():
        L.append(f"| {name} | " + " | ".join(f"{v[g]['n']} | {f(v[g]['C1'])} | {f(v[g]['C2'])}" for g in GROUPS) + " |")
    L += ["", "## 5. Spearman", "", "| groep | C1 logit~tokens | C2 logit~tokens | C1~C2 |", "|---|---:|---:|---:|"]
    for g in GROUPS:
        v = r["spearman"][g]
        L.append(f"| {g} | {f(v['C1_logit_tokens'])} | {f(v['C2_logit_tokens'])} | {f(v['C1_C2'])} |")
    L.append(f"| alle teksten | | | {f(r['spearman']['all']['C1_C2'])} |")
    ch = r["checks"]["c2_vs_judge_on_g40"]
    a = r["anchor_novels_judge_c2"]
    L += ["", "## 6. Controles en herkomst", "",
          f"- C2 op `g40` tegen de judge-score van het anker: {ch['compared']} vergeleken, {ch['missing']} ontbreken, "
          f"grootste |ΔP| {f(ch['max_abs_dp'], 4)}, {ch['over_tol']} boven {f(ch['tol'], 3)}; tokens gelijk aan `c2_tokens` "
          f"bij {ch['tokens_equal']}." if ch["max_abs_dp"] is not None else "- C2 tegen de judge: niets vergeleken.",
          f"- Door `naturalness.py` als `truncated` gemarkeerd (alleen teksten van precies {MAX_TOKENS} tokens): "
          + ", ".join(f"{c} {n}" for c, n in r["checks"]["truncated_flags"].items()) + ".",
          f"- Referenties van `nl_books` gelijk aan het Nederlands van G4-1 voor hetzelfde paar: "
          f"{info.get('nl_books_equal_to_g41')} van {r['groups']['nl_books']['n']}.",
          f"- Ter vergelijking, C2 volgens de judge op alle romanrijen van het anker die in aanmerking komen: "
          f"{a['scored']} rijen, \"origineel\" {f(a['share_original'])}, gem. logit (±12) {f(a['mean_logit'], 2)}.", "",
          "Trechter (`texts.json`):", "", "| groep | rijen | te lang | in aanmerking | boeken in aanmerking | getrokken |",
          "|---|---:|---:|---:|---:|---:|"]
    for g, v in info["groups"].items():
        if g == "novels":
            L.append(f"| human/g40 (anker: {v['anchor_novels']} romanrijen, {v['anchor_eligible']} af en schoon) | "
                     f"{v['pairs']} | {v['human_too_long']} + {v['g40_too_long']} | {v['eligible']} | "
                     f"{v['books_eligible']} | {v['drawn']} |")
        elif "rows" in v:
            L.append(f"| {g} | {v['rows']} | {v['too_long']} | {v['eligible']} | {v['books_eligible']} | {v['drawn']} |")
    L += ["", "Wat het niet zegt: geen maat van kwaliteit; de groepen verschillen ook in lengte, boeken, auteurs en tijd "
          "van uitgave; `nl_books` is 2 boeken; alleen `human`–`g40` is gepaard (protocol §16, C2F.5)."]
    return "\n".join(L) + "\n"


def report(args):
    meta = g4arm.read_rows(args.meta)
    preds = {"C1": g4arm.read_rows(args.pred_c1), "C2": g4arm.read_rows(args.pred_c2)}
    judge = []
    for s in sorted(glob.glob(args.judge)):
        judge += g4arm.read_rows(s, columns=["uid", "p_c2", "logit_c2", "c2_tokens", "c2_overflow"])
    if not judge:
        raise SystemExit(f"no judge scores {args.judge}")
    eligible = [x for x in Path(args.eligible).read_text().splitlines() if x]
    info = json.loads(Path(args.texts_json).read_text())
    r = measures(meta, preds, judge, eligible)
    r["texts"] = info
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(r, indent=1) + "\n")
    (out / "report.md").write_text(report_md(r, info))
    print("C2F_REPORT " + json.dumps({g: {c: round(r["groups"][g][c]["share_original"], 4) for c in CLASSIFIERS}
                                      for g in GROUPS if r["groups"][g]["n"]}), flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("texts")
    t.add_argument("--heldout-evalsets", required=True, help="$G/data/heldout/evalsets.parquet (set books)")
    t.add_argument("--heldout-books", required=True, help="$G/data/heldout/books.json")
    t.add_argument("--g41", required=True, help="G4-1's set ($G/data/G4-1/)")
    t.add_argument("--anchor", required=True, help="glob of G4-7's anchor samples ($RL/anchor/samples/w*/shard-*)")
    t.add_argument("--b1", required=True, help="$S/data/books-b1/b1-selected.parquet")
    t.add_argument("--c1-config", required=True, help="a directory with C1's config.json")
    t.add_argument("--c2-config", required=True, help="a directory with C2's config.json")
    t.add_argument("--out", required=True)
    k = sub.add_parser("report")
    k.add_argument("--meta", required=True)
    k.add_argument("--texts-json", required=True)
    k.add_argument("--eligible", required=True, help="anchor-eligible.txt of texts")
    k.add_argument("--pred-c1", required=True)
    k.add_argument("--pred-c2", required=True)
    k.add_argument("--judge", required=True, help="glob of the anchor's C2 scores ($RL/anchor/scores/c2/w1/shard-*)")
    k.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    {"texts": texts, "report": report}[args.cmd](args)


if __name__ == "__main__":
    main()
