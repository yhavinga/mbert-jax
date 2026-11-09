#!/usr/bin/env python3
"""Summaries of the judge's scores, for Yeb.

    python report.py sanity --scores <dir of score shards> --out <dir>      sanity.json and sanity.md
    python report.py scan --pairs <dir> --scores <dir> --out europarl-scan.md [--meta meta.json]
    python report.py compare --reference ref.json --scores <dir of the sanity scores>

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
from collections import Counter
from pathlib import Path

import numpy as np

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


def scan_command(args):
    pairs = {r["id"]: r for r in read_dir(args.pairs, columns=["id", "en", "nl"])}
    rows = read_dir(args.scores)
    for r in rows:
        r["en"], r["nl"] = pairs[r["id"]]["en"], pairs[r["id"]]["nl"]
    missing = len(pairs) - len(rows)
    if missing:
        raise SystemExit(f"{missing} pairs of {args.pairs} have no score in {args.scores}")
    meta = json.loads(Path(args.meta).read_text()) if args.meta else {}
    Path(args.out).write_text(scan_markdown(rows, meta, args.seed, args.per_class))
    print(f"{args.out}: {len(rows):,} pairs")


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
    p.add_argument("--meta", default=None, help="JSON with title and intro lines")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--per-class", type=int, default=10)
    p = sub.add_parser("compare")
    p.add_argument("--reference", required=True)
    p.add_argument("--scores", required=True)
    p.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    {"sanity": sanity_command, "scan": scan_command, "compare": compare_command}[args.command](args)


if __name__ == "__main__":
    main()
