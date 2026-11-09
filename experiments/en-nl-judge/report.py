#!/usr/bin/env python3
"""Summaries of the judge's scores, for Yeb.

    python report.py sanity --scores <dir of score shards> --out <dir>      sanity.json and sanity.md
    python report.py scan --pairs <dir> --scores <dir> --out europarl-scan.md --manifest <pairs>/manifest.json \
        [--sanity-md sanity.md] [--compare-md compare.md]
    python report.py compare --reference ref.json --scores <dir of the sanity scores>
    python report.py pilot --pairs <dir> --scores <dir> --ids pairs30_ids.txt --out pilot-scores.md
    python report.py source --pairs <dir> --scores <dir> --pair-scores <dir> --ids pairs30_ids.txt --out source-quality.md

sanity: per kind of pair (the originals and each deterministic error of data.py) the number of rows, the spread of
the scores and their counts per score class, and per error the AUC of the score against the originals: the chance
that an original scores higher than a corrupted pair (ties half), against all originals and against the originals of
the same paragraphs.

scan: the distribution of the scores over the classes 0-0.1, ..., 0.9-1 (the last includes 1), by period and year, and
per class 10 random paragraphs (fixed seed) with score, sitting date, speaker and both texts.
"""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

CLASS_EDGES = [k / 10 for k in range(11)]


def score_class(score):
    return min(int(score * 10), 9)


def class_label(k):
    return f"{CLASS_EDGES[k]:.1f}–{CLASS_EDGES[k + 1]:.1f}"


def auc(good, bad):
    """P(a good row scores higher than a bad one), ties counted half (the Mann-Whitney U over n_good x n_bad)."""
    good, bad = np.sort(np.asarray(good, float)), np.asarray(bad, float)
    if not len(good) or not len(bad):
        return float("nan")
    right = np.searchsorted(good, bad, side="right")
    left = np.searchsorted(good, bad, side="left")
    return float(((len(good) - right) + 0.5 * (right - left)).sum() / (len(good) * len(bad)))


def spread(scores):
    s = np.asarray(scores, float)
    q = np.quantile(s, [0.1, 0.25, 0.5, 0.75, 0.9]) if len(s) else [float("nan")] * 5
    return {"n": int(len(s)), "mean": float(s.mean()) if len(s) else float("nan"), "p10": float(q[0]),
            "p25": float(q[1]), "median": float(q[2]), "p75": float(q[3]), "p90": float(q[4]),
            "share_ge_0.5": float((s >= 0.5).mean()) if len(s) else float("nan")}


def histogram(scores):
    counts = Counter(score_class(s) for s in scores)
    return [counts.get(k, 0) for k in range(10)]


def read_dir(directory, columns=None):
    import pyarrow as pa
    import pyarrow.parquet as pq

    files = sorted(Path(directory).glob("shard-*.parquet"))
    if not files:
        raise SystemExit(f"no shard-*.parquet in {directory}")
    return pa.concat_tables([pq.read_table(f, columns=columns) for f in files]).to_pylist()


def fmt(x, digits=3):
    return "–" if x != x else f"{x:.{digits}f}"


def sanity_summary(rows):
    scored = [r for r in rows if r["skipped"] is None]
    originals = {r["source_id"]: r["score"] for r in scored if r["kind"] == "original"}
    kinds = sorted({r["kind"] for r in scored}, key=lambda k: (k != "original", k))
    summary = {"skipped": dict(Counter(r["skipped"] for r in rows if r["skipped"] is not None)), "kinds": {}}
    for kind in kinds:
        members = [r for r in scored if r["kind"] == kind]
        scores = [r["score"] for r in members]
        entry = {**spread(scores), "histogram": histogram(scores),
                 "answer_mass_min": float(min(r["answer_mass"] for r in members)),
                 "answer_mass_median": float(np.median([r["answer_mass"] for r in members])),
                 "top_tokens": dict(Counter(r["top_token"] for r in members).most_common(5))}
        if kind != "original":
            paired = [originals[r["source_id"]] for r in members if r["source_id"] in originals]
            entry["auc_all_originals"] = auc(list(originals.values()), scores)
            entry["auc_same_paragraphs"] = auc(paired, scores)
            entry["paired_wins"] = float(np.mean([originals[r["source_id"]] > r["score"] for r in members
                                                  if r["source_id"] in originals]))
        summary["kinds"][kind] = entry
    return summary


def sanity_markdown(summary):
    lines = ["| kind | n | mean | p10 | median | p90 | share ≥ 0.5 | AUC vs all originals | AUC vs same paragraphs |"
             " original higher |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for kind, e in summary["kinds"].items():
        lines.append(f"| {kind} | {e['n']} | {fmt(e['mean'])} | {fmt(e['p10'])} | {fmt(e['median'])} | "
                     f"{fmt(e['p90'])} | {fmt(e['share_ge_0.5'], 2)} | {fmt(e.get('auc_all_originals', float('nan')))}"
                     f" | {fmt(e.get('auc_same_paragraphs', float('nan')))} | "
                     f"{fmt(e.get('paired_wins', float('nan')), 2)} |")
    lines += ["", "Counts per score class:", "",
              "| kind | " + " | ".join(class_label(k) for k in range(10)) + " |",
              "|---|" + "---:|" * 10]
    for kind, e in summary["kinds"].items():
        lines.append(f"| {kind} | " + " | ".join(str(c) for c in e["histogram"]) + " |")
    lines += ["", "Answer mass P(yes) + P(no) and the most likely first tokens:", "",
              "| kind | min | median | top tokens |", "|---|---:|---:|---|"]
    for kind, e in summary["kinds"].items():
        tops = ", ".join(f"`{t}` {n}" for t, n in e["top_tokens"].items())
        lines.append(f"| {kind} | {fmt(e['answer_mass_min'])} | {fmt(e['answer_mass_median'])} | {tops} |")
    if summary["skipped"]:
        lines += ["", f"Not scored: {summary['skipped']}"]
    return "\n".join(lines) + "\n"


def sanity_command(args):
    rows = read_dir(args.scores)
    summary = sanity_summary(rows)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "sanity.json").write_text(json.dumps(summary, indent=1))
    (out / "sanity.md").write_text(sanity_markdown(summary))
    print(sanity_markdown(summary))


def quote(text):
    return "\n".join("> " + line for line in text.splitlines()) if text else "> (leeg)"


def scan_markdown(rows, meta, seed, per_class):
    scored = [r for r in rows if r["skipped"] is None]
    skipped = Counter(r["skipped"] for r in rows if r["skipped"] is not None)
    scores = [r["score"] for r in scored]
    total = spread(scores)
    hist = histogram(scores)
    lines = [f"# {meta.get('title', 'Scan van de beoordelaar')}", ""]
    lines += [f"- {line}" for line in meta.get("intro", [])]
    for heading, text in meta.get("sections", []):
        lines += ["", f"## {heading}", "", text.rstrip()]
    lines += ["", "## Deze scan", ""]
    lines += [f"- Paren: {len(rows):,}; gescoord {len(scored):,}; niet gescoord: "
              f"{dict(skipped) if skipped else 'geen'}.",
              f"- Score = P(yes) / (P(yes) + P(no)) van het eerste antwoordtoken. Gemiddeld {fmt(total['mean'])}, "
              f"p10 {fmt(total['p10'])}, p25 {fmt(total['p25'])}, mediaan {fmt(total['median'])}, "
              f"p75 {fmt(total['p75'])}, p90 {fmt(total['p90'])}; aandeel ≥ 0,5: {fmt(total['share_ge_0.5'], 3)}.",
              f"- P(yes) + P(no) (de kans dat het model met yes of no begint): mediaan "
              f"{fmt(float(np.median([r['answer_mass'] for r in scored])))}, minimum "
              f"{fmt(min(r['answer_mass'] for r in scored))}; meest waarschijnlijke eerste tokens: "
              + ", ".join(f"`{t}` {n:,}" for t, n in Counter(r["top_token"] for r in scored).most_common(6)) + ".",
              "", "## Verdeling", "", "| scoreklasse | paren | aandeel | cumulatief vanaf 1 |", "|---|---:|---:|---:|"]
    above = len(scores)
    for k in range(10):
        lines.append(f"| {class_label(k)} | {hist[k]:,} | {hist[k] / max(len(scores), 1):.3f} | "
                     f"{above / max(len(scores), 1):.3f} |")
        above -= hist[k]
    odds = np.array([r["logp_yes"] - r["logp_no"] for r in scored])
    edges = [-np.inf, -10, -5, -2, 0, 2, 5, 10, np.inf]
    counts = np.histogram(odds, bins=edges)[0]
    q = np.quantile(odds, [0.05, 0.25, 0.5, 0.75, 0.95])
    lines += ["", "De score is bijna binair; de log-odds log P(yes) − log P(no) (score = 1 / (1 + e^−log-odds)) "
              "ordenen fijner. Kwantielen: p5 " + ", p25 ".join([f"{q[0]:.1f}", f"{q[1]:.1f}"]) +
              f", mediaan {q[2]:.1f}, p75 {q[3]:.1f}, p95 {q[4]:.1f}.", "",
              "| log-odds | paren | aandeel |", "|---|---:|---:|"]
    for k, c in enumerate(counts):
        low, high = edges[k], edges[k + 1]
        label = f"< {high:g}" if low == -np.inf else (f"≥ {low:g}" if high == np.inf else f"{low:g} tot {high:g}")
        lines.append(f"| {label} | {c:,} | {c / max(len(odds), 1):.3f} |")
    if "year" in scored[0]:
        lines += ["", "Per periode (aandeel per scoreklasse):", "",
                  "| periode | paren | " + " | ".join(class_label(k) for k in range(10)) + " |",
                  "|---|---:|" + "---:|" * 10]
        for name, test in (("1999–2003", lambda y: y <= 2003), ("2004–2012", lambda y: y >= 2004)):
            part = [r["score"] for r in scored if test(r["year"])]
            h = histogram(part)
            lines.append(f"| {name} | {len(part):,} | " + " | ".join(f"{c / max(len(part), 1):.3f}" for c in h) + " |")
        lines += ["", "Per jaar:", "", "| jaar | paren | gemiddelde | mediaan | aandeel ≥ 0,5 |", "|---|---:|---:|---:|---:|"]
        for year in sorted({r["year"] for r in scored}):
            s = spread([r["score"] for r in scored if r["year"] == year])
            lines.append(f"| {year} | {s['n']:,} | {fmt(s['mean'])} | {fmt(s['median'])} | {fmt(s['share_ge_0.5'])} |")
    lines += ["", "Per lengte (Engelse woorden):", "", "| woorden | paren | gemiddelde | aandeel ≥ 0,5 |",
              "|---|---:|---:|---:|"]
    for low, high in ((0, 20), (20, 50), (50, 100), (100, 200), (200, 400), (400, 10 ** 9)):
        s = spread([r["score"] for r in scored if low <= r["en_words"] < high])
        if s["n"]:
            lines.append(f"| {low}–{high if high < 10 ** 9 else ''} | {s['n']:,} | {fmt(s['mean'])} | "
                         f"{fmt(s['share_ge_0.5'])} |")
    rng = random.Random(seed)
    lines += ["", f"## {per_class} willekeurige alinea's per scoreklasse (seed {seed})", ""]
    for k in reversed(range(10)):
        members = sorted((r for r in scored if score_class(r["score"]) == k), key=lambda r: r["id"])
        chosen = sorted(rng.sample(members, min(per_class, len(members))), key=lambda r: -r["score"])
        lines += [f"### {class_label(k)} ({len(members):,} paren)", ""]
        for r in chosen:
            who = r.get("speaker") or "?"
            if r.get("nationality"):
                who += f" ({r['nationality']})"
            lines += [f"**{r['score']:.4f}** · {r.get('date', '?')} · {who} · `{r['id']}` · en {r['en_words']} w, "
                      f"nl {r['nl_words']} w", "", "EN:", "", quote(r["en"]), "", "NL:", "", quote(r["nl"]), ""]
    return "\n".join(lines) + "\n"


def europarl_meta(manifest, sanity_md, compare_md):
    """Title, introduction and the sections before the scan's own numbers: where the pairs come from, the model, the
    prompt word for word, the sanity check and the comparison with transformers."""
    import judge

    counts, periods = manifest["counts"], manifest["by_period"]
    intro = [
        "Bron: Multilingual Parallel Direct Europarl (Amponsah-Kaakyire et al., MoTra 2021), "
        "github.com/UDS-SFB-B6-Datasets/Multilingual-Parallel-Direct-Europarl, commit "
        f"`{manifest.get('repo_commit')}`, `proceedings/xml/{{en,nl}}` ({manifest['sittings_en']} zittingen).",
        "Koppeling zoals `scripts/utils.py`: een alinea met `sl=\"en\"` in het EN-bestand gaat naar de alinea van het "
        "NL-bestand met dezelfde interventie-id, dezelfde index onder de `sl=\"en\"`-alinea's van die interventie en "
        "hetzelfde aantal daarvan. Zonder het langdetect-filter van utils.py; tekst is de hele alinea.",
        f"Engelse alinea's met `sl=\"en\"`: {counts['en_paragraphs']:,}; gekoppeld {counts['paired']:,} "
        f"(1999–2003: {periods['1999-2003']['paired']:,}; 2004–2012: {periods['2004-2012']['paired']:,}); zonder "
        f"NL-alinea {counts.get('no_nl_paragraph', 0):,}. Vanaf medio 2011 koppelt de regel niets: de NL-bestanden "
        "markeren alles `sl=\"nl\"` en bevatten de onvertaalde toespraken.",
        f"Beoordelaar: `google/gemma-4-31B-it` @ `{judge.REVISION[:7]}`, omgezet naar MaxText 0.2.5, bf16, op een "
        "v6e-8; één forward pass per paar. Score = P(yes) / (P(yes) + P(no)) van het eerste antwoordtoken, opgeteld "
        "over de schrijfwijzen (yes, Yes, YES, ...). Er is geen drempel gekozen.",
    ]
    prompt = judge.TEMPLATE_PREFIX + judge.QUESTION + judge.TEMPLATE_SUFFIX
    sections = [["Prompt (letterlijk; {en} en {nl} zijn de alinea's)", f"````\n{prompt}\n````"]]
    if sanity_md:
        sections.append(["Sanity-check (500 alinea's en hun deterministische fouten, data.py sanity)", sanity_md])
    if compare_md:
        sections.append(["Controle van de omzetting: TPU (MaxText, bf16) tegen transformers (CPU, float32)",
                         compare_md])
    return {"title": "Europarl-direct en→nl: scan met Gemma 4 31B als beoordelaar", "intro": intro,
            "sections": sections}


def scan_command(args):
    pairs = {r["id"]: r for r in read_dir(args.pairs, columns=["id", "en", "nl"])}
    rows = read_dir(args.scores)
    for r in rows:
        r["en"], r["nl"] = pairs[r["id"]]["en"], pairs[r["id"]]["nl"]
    missing = len(pairs) - len(rows)
    if missing:
        raise SystemExit(f"{missing} pairs of {args.pairs} have no score in {args.scores}")
    meta = europarl_meta(json.loads(Path(args.manifest).read_text()),
                         Path(args.sanity_md).read_text() if args.sanity_md else None,
                         Path(args.compare_md).read_text() if args.compare_md else None)
    Path(args.out).write_text(scan_markdown(rows, meta, args.seed, args.per_class))
    print(f"{args.out}: {len(rows):,} pairs")


def ranks(values):
    """Ranks from 1, ties sharing their mean rank."""
    values = np.asarray(values, float)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    out = np.empty(len(values))
    start = 0
    while start < len(values):
        end = start
        while end + 1 < len(values) and sorted_values[end + 1] == sorted_values[start]:
            end += 1
        out[order[start:end + 1]] = (start + end) / 2 + 1
        start = end + 1
    return out


def spearman(x, y):
    """Spearman's rho over the rows where both are numbers: the Pearson correlation of their ranks."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    keep = ~(np.isnan(x) | np.isnan(y))
    rx, ry = ranks(x[keep]), ranks(y[keep])
    return float(np.corrcoef(rx, ry)[0, 1]), int(keep.sum())


def pilot_markdown(rows, ids30):
    """The Gemma scores of the translator's pilot: the 30 pairs Yeb reads (in his order), Spearman with chrF and
    MetricX-QE over all pairs, and the 10 lowest."""
    by_id = {r["id"]: r for r in rows}
    missing = [i for i in ids30 if i not in by_id]
    if missing:
        raise SystemExit(f"{len(missing)} of the 30 ids have no score: {missing[:3]}")
    scored = [r for r in rows if r["skipped"] is None]
    g = [r["score"] for r in scored]
    lines = ["# Pilot en→nl-vertaler: score van Gemma 4 31B als beoordelaar", "",
             f"- {len(rows):,} paren, gescoord {len(scored):,}. Beoordeeld als trainingspaar: Engelse bron = het "
             "synthetische Engels (TranslateGemma 27B, nl→en), Nederlandse vertaling = het Nederlandse origineel.",
             "- Gemma-score = P(yes) / (P(yes) + P(no)) van het eerste antwoordtoken (0–1, hoger = beter); "
             "chrF = terugvertaling tegen origineel (0–100, hoger = beter); MetricX-24-QE (0–25, lager = beter).",
             f"- Verdeling: gemiddeld {fmt(spread(g)['mean'])}, mediaan {fmt(spread(g)['median'])}, p10 "
             f"{fmt(spread(g)['p10'])}, aandeel ≥ 0,5: {fmt(spread(g)['share_ge_0.5'])}.", "",
             "## Spearman over alle paren", "", "| met | rho | n |", "|---|---:|---:|"]
    for name, key, note in (("chrF", "chrf", "verwacht positief"),
                            ("MetricX-QE en→nl", "qe_en_nl", "lager is beter, dus verwacht negatief"),
                            ("MetricX-QE nl→en", "qe_nl_en", "lager is beter, dus verwacht negatief")):
        rho, n = spearman(g, [r.get(key, float("nan")) for r in scored])
        lines.append(f"| {name} ({note}) | {rho:.3f} | {n:,} |")
    lines += ["", "Per scoreklasse:", "", "| scoreklasse | paren |", "|---|---:|"]
    for k, c in enumerate(histogram(g)):
        lines.append(f"| {class_label(k)} | {c} |")
    lines += ["", "## De 30 paren van pairs30.md (zelfde volgorde)", "",
              "| # | id | bron | Gemma | chrF | QE en→nl | QE nl→en |", "|---:|---|---|---:|---:|---:|---:|"]
    for k, i in enumerate(ids30, 1):
        r = by_id[i]
        lines.append(f"| {k} | `{i}` | {r.get('source', '')} | {fmt(r['score'])} | {fmt(r.get('chrf', float('nan')), 1)}"
                     f" | {fmt(r.get('qe_en_nl', float('nan')), 2)} | {fmt(r.get('qe_nl_en', float('nan')), 2)} |")
    lowest = sorted(scored, key=lambda r: r["score"])[:10]
    lines += ["", "## De 10 paren met de laagste Gemma-score", ""]
    for k, r in enumerate(lowest, 1):
        lines += [f"### {k}. `{r['id']}` — Gemma {r['score']:.4f} · chrF {fmt(r.get('chrf', float('nan')), 1)} · "
                  f"QE en→nl {fmt(r.get('qe_en_nl', float('nan')), 2)} · {r.get('source', '')}", "",
                  "Engelse bron (synthetisch):", "", quote(r["en"]), "", "Nederlands origineel:", "", quote(r["nl"]), ""]
    return "\n".join(lines) + "\n"


def pilot_command(args):
    pairs = {r["id"]: r for r in read_dir(args.pairs, columns=["id", "en", "nl"])}
    rows = read_dir(args.scores)
    for r in rows:
        r["en"], r["nl"] = pairs[r["id"]]["en"], pairs[r["id"]]["nl"]
    ids30 = [line.strip() for line in Path(args.ids).read_text().splitlines() if line.strip()]
    if len(ids30) != 30:
        raise SystemExit(f"{args.ids} has {len(ids30)} ids, not 30")
    Path(args.out).write_text(pilot_markdown(rows, ids30))
    print(f"{args.out}: {len(rows):,} pairs")


def log_odds(r):
    return r["logp_yes"] - r["logp_no"]


def passage(r, pair):
    return [f"**{r['score']:.4f}** (log-odds {log_odds(r):.1f}) · `{r['id']}` · {r.get('source', '')} · "
            f"paarscore {fmt(pair['score'])} · chrF {fmt(r.get('chrf', float('nan')), 1)} · QE en→nl "
            f"{fmt(r.get('qe_en_nl', float('nan')), 2)} · {r['nl_words']} woorden", "", quote(r["nl"]), ""]


def source_markdown(rows, pair_rows, ids30, seed=0):
    """The source check of the pilot's Dutch originals: distribution (also per source), overlap with the pair score
    and with MetricX-QE, the 30 pairs of pairs30.md, the 15 lowest and 15 random rejected passages with their text."""
    import judge

    pairs = {r["id"]: r for r in pair_rows}
    missing = [r["id"] for r in rows if r["id"] not in pairs] + [i for i in ids30 if i not in pairs]
    if missing:
        raise SystemExit(f"{len(missing)} ids without a pair score: {missing[:3]}")
    scored = [r for r in rows if r["skipped"] is None]
    g = [r["score"] for r in scored]
    sp = spread(g)
    odds = np.array([log_odds(r) for r in scored])
    q = np.quantile(odds, [0.05, 0.25, 0.5, 0.75, 0.95])
    prompt = judge.TEMPLATE_PREFIX + judge.SOURCE_QUESTION + judge.TEMPLATE_SUFFIX
    lines = ["# Pilot en→nl-vertaler: bronkwaliteit van de Nederlandse originelen (Gemma 4 31B)", "",
             f"- {len(rows):,} Nederlandse originelen uit `pairs.parquet`, monolinguaal beoordeeld (zonder het Engels); "
             f"gescoord {len(scored):,}.",
             "- Score = P(yes) / (P(yes) + P(no)) van het eerste antwoordtoken, zoals bij de paren. Er is geen drempel "
             "gekozen. Waar hieronder \"afgekeurd\" staat, is dat beschrijvend: het model zegt eerder no dan yes "
             "(score < 0,5); < 0,1 staat er steeds naast. De 652 afgekeurde paren zijn de paren met paarscore < 0,1 "
             "(pilot-scores.md).",
             f"- Verdeling: gemiddeld {fmt(sp['mean'])}, mediaan {fmt(sp['median'])}, p10 {fmt(sp['p10'])}; "
             f"aandeel ≥ 0,5: {fmt(sp['share_ge_0.5'])}. Log-odds log P(yes) − log P(no): p5 {q[0]:.1f}, p25 "
             f"{q[1]:.1f}, mediaan {q[2]:.1f}, p75 {q[3]:.1f}, p95 {q[4]:.1f}.", "",
             "## Prompt (letterlijk; {nl} is de passage)", "", f"````\n{prompt}\n````", "",
             "## Verdeling, ook per bron", "",
             "| bron | passages | gemiddeld | mediaan | < 0,1 | < 0,5 | " +
             " | ".join(class_label(k) for k in range(10)) + " |", "|---|---:|---:|---:|---:|---:|" + "---:|" * 10]
    for name in ["alle"] + sorted({r.get("source", "") for r in scored}):
        part = [r["score"] for r in scored if name == "alle" or r.get("source", "") == name]
        s = spread(part)
        lines.append(f"| {name} | {len(part):,} | {fmt(s['mean'])} | {fmt(s['median'])} | "
                     f"{sum(x < 0.1 for x in part):,} | {sum(x < 0.5 for x in part):,} | "
                     + " | ".join(str(c) for c in histogram(part)) + " |")

    def overlap(selected, label):
        n = len(selected)
        b01 = sum(r["score"] < 0.1 for r in selected)
        b05 = sum(r["score"] < 0.5 for r in selected)
        return f"| {label} | {n:,} | {b01:,} ({b01 / max(n, 1):.1%}) | {b05:,} ({b05 / max(n, 1):.1%}) |"

    pair_low = [r for r in scored if pairs[r["id"]]["score"] < 0.1]
    pair_ok = [r for r in scored if pairs[r["id"]]["score"] >= 0.1]
    qe_high = [r for r in scored if r.get("qe_en_nl") is not None and r["qe_en_nl"] > 10]
    qe_low = [r for r in scored if r.get("qe_en_nl") is not None and r["qe_en_nl"] <= 10]
    lines += ["", "## Overlap met de paarscore en met MetricX-QE en→nl", "",
              "| groep | passages | bronscore < 0,1 | bronscore < 0,5 |", "|---|---:|---:|---:|",
              overlap(scored, "alle passages"),
              overlap(pair_low, "paarscore < 0,1 (de afgekeurde paren)"),
              overlap(pair_ok, "paarscore ≥ 0,1"),
              overlap(qe_high, "MetricX-QE en→nl > 10"),
              overlap(qe_low, "MetricX-QE en→nl ≤ 10")]
    src_low = [r for r in scored if r["score"] < 0.5]
    both = sum(pairs[r["id"]]["score"] < 0.1 for r in src_low)
    lines += ["", f"Omgekeerd: van de {len(src_low):,} passages met bronscore < 0,5 hebben er {both:,} ook een "
              "paarscore < 0,1.", "", "Spearman van de bronscore met:", "", "| met | rho | n |", "|---|---:|---:|"]
    for name, values in (("paarscore", [pairs[r["id"]]["score"] for r in scored]),
                         ("chrF", [r.get("chrf", float("nan")) for r in scored]),
                         ("MetricX-QE en→nl", [r.get("qe_en_nl", float("nan")) for r in scored])):
        rho, n = spearman(g, values)
        lines.append(f"| {name} | {rho:.3f} | {n:,} |")
    by_id = {r["id"]: r for r in rows}
    lines += ["", "## De 30 paren van pairs30.md (zelfde volgorde)", "",
              "| # | id | bron | bronscore | paarscore | chrF | QE en→nl |", "|---:|---|---|---:|---:|---:|---:|"]
    for k, i in enumerate(ids30, 1):
        r = by_id[i]
        lines.append(f"| {k} | `{i}` | {r.get('source', '')} | {fmt(r['score'])} | {fmt(pairs[i]['score'])} | "
                     f"{fmt(r.get('chrf', float('nan')), 1)} | {fmt(r.get('qe_en_nl', float('nan')), 2)} |")
    lowest = sorted(scored, key=log_odds)[:15]
    lines += ["", "## De 15 laagste passages (op log-odds, want veel scores zijn 0,0000)", ""]
    for k, r in enumerate(lowest, 1):
        lines += [f"### {k}."] + passage(r, pairs[r["id"]])
    taken = {r["id"] for r in lowest}
    rest = sorted((r for r in src_low if r["id"] not in taken), key=lambda r: r["id"])
    chosen = random.Random(seed).sample(rest, min(15, len(rest)))
    lines += [f"## 15 willekeurige afgekeurde passages (bronscore < 0,5, buiten de 15 laagste; seed {seed}, uit "
              f"{len(rest):,})", ""]
    for k, r in enumerate(sorted(chosen, key=lambda r: r["score"]), 1):
        lines += [f"### {k}."] + passage(r, pairs[r["id"]])
    return "\n".join(lines) + "\n"


def source_command(args):
    texts = {r["id"]: r for r in read_dir(args.pairs, columns=["id", "nl"])}
    rows = read_dir(args.scores)
    for r in rows:
        r["nl"] = texts[r["id"]]["nl"]
    ids30 = [line.strip() for line in Path(args.ids).read_text().splitlines() if line.strip()]
    Path(args.out).write_text(source_markdown(rows, read_dir(args.pair_scores), ids30))
    print(f"{args.out}: {len(rows):,} passages")


def compare_command(args):
    """The TPU's scores of the reference rows next to those of transformers on the CPU (float32)."""
    reference = json.loads(Path(args.reference).read_text())
    tpu = {r["id"]: r for r in read_dir(args.scores)}
    lines = ["| id | kind | tokens | score CPU | score TPU | logp yes CPU / TPU | logp no CPU / TPU | top CPU / TPU |",
             "|---|---|---:|---:|---:|---:|---:|---|"]
    worst = {"score": 0.0, "logp_yes": 0.0, "logp_no": 0.0}
    for ref in reference["rows"]:
        mine = tpu[ref["id"]]
        for key in worst:
            worst[key] = max(worst[key], abs(ref[key] - mine[key]))
        lines.append(f"| `{ref['id']}` | {ref['kind']} | {len(ref['prompt_ids'])} | {ref['score']:.4f} | "
                     f"{mine['score']:.4f} | {ref['logp_yes']:.3f} / {mine['logp_yes']:.3f} | "
                     f"{ref['logp_no']:.3f} / {mine['logp_no']:.3f} | `{ref['top'][0][0]}` / `{mine['top_token']}` |")
    lines += ["", "Largest differences: " + ", ".join(f"{k} {v:.4f}" for k, v in worst.items())]
    text = "\n".join(lines) + "\n"
    if args.out:
        Path(args.out).write_text(text)
    print(text)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Reports of the en-nl judge")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("sanity")
    p.add_argument("--scores", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("scan")
    p.add_argument("--pairs", required=True)
    p.add_argument("--scores", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--manifest", required=True, help="manifest.json of data.py motra")
    p.add_argument("--sanity-md", default=None)
    p.add_argument("--compare-md", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--per-class", type=int, default=10)
    p = sub.add_parser("pilot")
    p.add_argument("--pairs", required=True)
    p.add_argument("--scores", required=True)
    p.add_argument("--ids", required=True, help="The 30 ids of pairs30.md, one per line, in its order")
    p.add_argument("--out", required=True)
    p = sub.add_parser("source")
    p.add_argument("--pairs", required=True)
    p.add_argument("--scores", required=True, help="The source scores (judge.py score --task source)")
    p.add_argument("--pair-scores", required=True, help="The pair scores of the same pilot")
    p.add_argument("--ids", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("compare")
    p.add_argument("--reference", required=True)
    p.add_argument("--scores", required=True)
    p.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    {"sanity": sanity_command, "scan": scan_command, "compare": compare_command,
     "pilot": pilot_command, "source": source_command}[args.command](args)


if __name__ == "__main__":
    main()
