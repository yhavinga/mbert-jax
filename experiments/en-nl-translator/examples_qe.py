#!/usr/bin/env python3
"""Examples for two of Yeb's open points of the fine-tune (protocol.md §8, 10 Nov): the 0.2 of decision rule (a) and the
query's bound QE <= 7.5 (§2.1). Nothing here is a translation of ours: every text comes from the tag test or the pairs
tables.

    python examples_qe.py rule-pairs --pairs pairs.parquet --items items.parquet --out rule-pairs.parquet
    python examples_qe.py rule --pairs pairs.parquet --qe qe.parquet --items items.parquet \\
        --scores metricx.parquet --comet comet.parquet --out examples-qe-rule.md --json examples-qe-rule.json
    python examples_qe.py cutoff --run RUN --long LONG --evalsets evalsets.parquet --tokenizer TOK \\
        --out examples-qe-cutoff.md --json examples-qe-cutoff.json
    python examples_qe.py length --run RUN --long LONG [--real real.parquet] --qe-max 5 --factor 1.3333 \\
        --out examples-r2-length.md --json examples-r2-length.json
    python examples_qe.py real-edges --real real-qe.parquet --source books --qe-max 5 --factor 1.3333 \\
        --out examples-rb-books.md --json examples-rb-books.json

rule-pairs: the tag test's WMT24++ translations of the base model 12B and of the adapter with the instruction sentence,
with its tag (variant instruction_tag), beside the WMT24++ reference: the input of metricx_qe.py (source:hypothesis:
reference and source:hypothesis) and comet_score.py. Ids are <variant>|<segment>.
rule: per segment the QE difference variant - base (from the tag test's own QE, tagtest/report/qe.parquet), its
distribution, and RULE_PER_CLASS examples per class around +0.2, +0.5, +1, +2 and -0.5, drawn with a fixed seed, each
with source, reference, both translations, QE, MetricX-24 with the reference and COMET-22.
cutoff: the training passages of both runs that pass every other bound of the query (sft_build.py), what each QE bound
would cost, and pairs drawn with a fixed seed per QE band, each with the synthetic English, the original Dutch, QE,
chrF and its source.
length (arm R2, protocol §9, amendment of 11 Nov): the bound on the length ratio (Dutch over English characters within a
factor of the source's median, sft_build.py --len-factor): per source the medians (characters and words), what each
factor costs on either side, and pairs drawn with a fixed seed just outside and just inside the chosen bound, for the
back-translation pairs (training passages that pass hard_fails and QE <= --qe-max) and the real pairs (real_pairs.py
candidates that pass the hard checks; their QE where real.parquet has it).
real-edges (arm RB, protocol §9, amendment of 11 Nov ~12:00): for one source of real pairs (the Dutch books), what the
hard checks, the QE bound and the length bound cost, in that order as sft_build.py applies them, in total and per
author (meta "author"); what other QE bounds would cost; and pairs drawn with a fixed seed just outside and just inside
the QE bound and the length bound, and the pairs with the highest QE (where a misaligned passage shows). The 13-grams
and the tokens come after these in sft_build.py and cost what its report says.
"""

import argparse
import bisect
import collections
import json
import math
import random
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for _path in (HERE.parent / "flan-nl", HERE):
    sys.path.insert(0, str(_path))

SEED = "en-nl-sft/examples/0"
RULE_VARIANTS = ("base", "instruction_tag")
# (label, lower, upper): a difference d is in the class when lower <= d <= upper
RULE_CLASSES = [("+0,2", 0.1, 0.3), ("+0,5", 0.4, 0.6), ("+1", 0.8, 1.2), ("+2", 1.6, 2.4), ("-0,5", -0.6, -0.4)]
RULE_PER_CLASS = 4
RULE_BINS = [-math.inf, -2, -1, -0.5, -0.1, 0.1, 0.5, 1, 2, math.inf]
# (label, lower, upper, n): lower < QE <= upper
CUTOFF_BANDS = [("6,5–7,0", 6.5, 7.0, 6), ("7,0–7,5", 7.0, 7.5, 6), ("7,5–8,0", 7.5, 8.0, 6), ("8,0–9,0", 8.0, 9.0, 6),
                ("> 10", 10.0, math.inf, 4)]
CUTOFF_BOUNDS = [6, 7, 7.5, 8, 9, 10]
MAX_CHARS = 3000


def nl(x, digits=2):
    """A number the Dutch way (decimal comma)"""
    if x is None:
        return "–"
    return f"{x:,.{digits}f}".replace(",", "_").replace(".", ",").replace("_", ".")


def n_(x):
    return f"{x:,}".replace(",", ".")


def quote(text, limit=MAX_CHARS):
    text = text.strip()
    cut = len(text) > limit
    text = text[:limit] + (" […]" if cut else "")
    return "\n".join("> " + line if line.strip() else ">" for line in text.split("\n"))


def tagtest_rows(pairs_path, qe_path):
    """The tag test's pairs with their QE (qe.parquet is in the order of pairs.parquet, as tagtest.py report reads it)"""
    import pyarrow.parquet as pq

    rows = pq.read_table(pairs_path, columns=["variant", "id", "set", "task", "source", "hypothesis"]).to_pylist()
    qe = pq.read_table(qe_path).to_pylist()
    if len(qe) != len(rows):
        raise SystemExit(f"{len(rows)} pairs, {len(qe)} QE scores")
    for r, q in zip(rows, qe):
        if q["id"] != r["id"] or q["qe_source_hypothesis"] is None:
            raise SystemExit(f"no QE in the order of the pairs at {r['variant']} {r['id']}")
        r["qe"] = q["qe_source_hypothesis"]
    return rows


def rule_pairs(args):
    import pyarrow as pa
    import pyarrow.parquet as pq

    items = {r["id"]: r for r in pq.read_table(args.items).to_pylist()}
    rows = pq.read_table(args.pairs, columns=["variant", "id", "set", "source", "hypothesis"]).to_pylist()
    out = []
    for r in rows:
        if r["set"] != "wmt24pp" or r["variant"] not in RULE_VARIANTS:
            continue
        ref = items[r["id"]]["reference"]
        if not ref:
            raise SystemExit(f"{r['id']}: no reference")
        out.append({"id": f"{r['variant']}|{r['id']}", "source": r["source"], "hypothesis": r["hypothesis"],
                    "reference": ref})
    pq.write_table(pa.Table.from_pylist(out), args.out)
    print(json.dumps({"rows": len(out)}))


def quantile(values, q):
    values = sorted(values)
    k = (len(values) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            out[order[k]] = (i + j) / 2
        i = j + 1
    return out


def spearman(a, b):
    return statistics.correlation(ranks(a), ranks(b))


def rule(args):
    import pyarrow.parquet as pq

    rows = [r for r in tagtest_rows(args.pairs, args.qe) if r["set"] == "wmt24pp" and r["variant"] in RULE_VARIANTS]
    items = {r["id"]: r for r in pq.read_table(args.items).to_pylist()}
    by = collections.defaultdict(dict)
    for r in rows:
        by[r["id"]][r["variant"]] = r
    metricx = {r["id"]: r for r in pq.read_table(args.scores).to_pylist()}
    comet = {r["id"]: r["comet22"] for r in pq.read_table(args.comet).to_pylist()}
    segs = []
    for i in sorted(by):
        v = by[i]
        if set(v) != set(RULE_VARIANTS):
            raise SystemExit(f"{i}: variants {sorted(v)}")
        seg = {"id": i, "domain": items[i]["task"], "source": items[i]["source"], "reference": items[i]["reference"]}
        for name in RULE_VARIANTS:
            key = f"{name}|{i}"
            m = metricx[key]
            seg[name] = {"hypothesis": v[name]["hypothesis"], "qe": v[name]["qe"], "qe_again": m["qe_source_hypothesis"],
                         "ref": m["ref_source_hypothesis_reference"], "comet": comet[key]}
        seg["diff"] = seg["instruction_tag"]["qe"] - seg["base"]["qe"]
        seg["diff_ref"] = seg["instruction_tag"]["ref"] - seg["base"]["ref"]
        seg["diff_comet"] = seg["instruction_tag"]["comet"] - seg["base"]["comet"]
        seg["same_text"] = seg["instruction_tag"]["hypothesis"].strip() == seg["base"]["hypothesis"].strip()
        segs.append(seg)
    diffs = [s["diff"] for s in segs]
    n = len(segs)
    again = max(abs(s[v]["qe"] - s[v]["qe_again"]) for s in segs for v in RULE_VARIANTS)
    mean = {v: {k: statistics.mean(s[v][k] for s in segs) for k in ("qe", "ref", "comet")} for v in RULE_VARIANTS}
    hist = collections.Counter(bisect.bisect_left(RULE_BINS, d) - 1 for d in diffs)
    top = sorted(diffs, reverse=True)
    stats = {
        "n": n, "mean_diff": statistics.mean(diffs), "median_diff": statistics.median(diffs),
        "sd_diff": statistics.stdev(diffs), **{f"p{q}": quantile(diffs, q / 100) for q in (5, 25, 75, 95)},
        "same_text": sum(s["same_text"] for s in segs),
        "same_text_absdiff_max": max((abs(s["diff"]) for s in segs if s["same_text"]), default=None),
        "same_text_absdiff_mean": statistics.mean([abs(s["diff"]) for s in segs if s["same_text"]] or [0.0]),
        "worse": sum(d > 0 for d in diffs),
        "better": sum(d < 0 for d in diffs), "equal": sum(d == 0 for d in diffs),
        "over": {str(c): sum(d > c for d in diffs) for c in (0.2, 0.5, 1, 2)},
        "under_-0.5": sum(d < -0.5 for d in diffs),
        "mean_without_top": {str(k): statistics.mean(top[k:]) for k in (5, 10, 20)},
        "share_of_sum_top10": sum(top[:10]) / sum(diffs) if sum(diffs) else None,
        "mean": mean, "qe_again_max_abs": again,
        "spearman_qe_ref": spearman(diffs, [s["diff_ref"] for s in segs]),
        "spearman_qe_comet": spearman(diffs, [s["diff_comet"] for s in segs]),
        "hist": {f"{RULE_BINS[b]}..{RULE_BINS[b + 1]}": hist[b] for b in range(len(RULE_BINS) - 1)},
    }
    chosen = {}
    for label, lo, hi in RULE_CLASSES:
        pool = sorted(s["id"] for s in segs if lo <= s["diff"] <= hi)
        k = min(RULE_PER_CLASS, len(pool))
        chosen[label] = {"pool": len(pool), "ids": sorted(random.Random(f"{SEED}/rule/{label}").sample(pool, k))}
    if args.json:
        Path(args.json).write_text(json.dumps({"stats": stats, "chosen": chosen,
                                               "segments": segs}, indent=1, ensure_ascii=False) + "\n")
    Path(args.out).write_text(rule_md(stats, chosen, {s["id"]: s for s in segs}), encoding="utf-8")
    print(json.dumps({"n": n, "mean_diff": round(stats["mean_diff"], 4),
                      "chosen": {k: v["pool"] for k, v in chosen.items()}}))


def rule_md(st, chosen, segs):
    m = st["mean"]
    hist_labels = {"-inf..-2": "≤ −2", "-2..-1": "−2 tot −1", "-1..-0.5": "−1 tot −0,5", "-0.5..-0.1": "−0,5 tot −0,1",
                   "-0.1..0.1": "−0,1 tot +0,1", "0.1..0.5": "+0,1 tot +0,5", "0.5..1": "+0,5 tot +1",
                   "1..2": "+1 tot +2", "2..inf": "> +2"}
    lines = [
        "# Voorbeelden bij beslisregel (a): wat betekent 0,2 slechter op MetricX-QE?", "",
        "Voor Yebs open punt 5(a) (protocol §5, regel (a): een arm mag op MetricX-24-QE gemiddeld hoogstens 0,20 "
        "slechter zijn dan het basismodel). Gemaakt op 10 nov 2026 met `examples_qe.py rule` (branch `en-nl-sft`) op "
        "mbert-mix. Alle vertalingen komen uit de tagproef (`gs://t5-dutch-english-usc2/mbert/translator/sft/tagtest/`): "
        "TranslateGemma 12B zonder adapter (**basis**) tegen de LoRA-adapter die met de instructiezin is getraind, "
        "vertaald mét die zin (**variant**, `instruction_tag`), op dezelfde "
        f"{st['n']} WMT24++-segmenten (en→nl_NL, zonder `is_bad_source`, seed `en-nl-sft/tagtest/0/wmt`). "
        "Greedy, dezelfde code. MetricX: lager is beter; COMET-22: hoger is beter.", "",
        "## Het gemiddelde en de verdeling per segment", "",
        f"- Gemiddeld QE-verschil (variant − basis): **{nl(st['mean_diff'], 3)}** (basis {nl(m['base']['qe'], 3)}, "
        f"variant {nl(m['instruction_tag']['qe'], 3)}); dat is de ~0,21 uit protocol §6. Mediaan van de verschillen per "
        f"segment: {nl(st['median_diff'], 3)}; standaardafwijking {nl(st['sd_diff'], 2)}; p5 {nl(st['p5'])}, "
        f"p25 {nl(st['p25'])}, p75 {nl(st['p75'])}, p95 {nl(st['p95'])}.",
        f"- Van de {st['n']} segmenten: {st['same_text']} met letterlijk dezelfde vertaling, {st['worse']} met een "
        f"hogere QE (slechter), {st['better']} met een lagere, {st['equal']} gelijk. Meer dan 0,2 slechter: "
        f"{st['over']['0.2']}; meer dan 0,5: {st['over']['0.5']}; meer dan 1: {st['over']['1']}; meer dan 2: "
        f"{st['over']['2']}. Meer dan 0,5 beter: {st['under_-0.5']}.",
        f"- Ruis: bij de {st['same_text']} segmenten met letterlijk dezelfde vertaling verschilt de QE toch, gemiddeld "
        f"{nl(st['same_text_absdiff_mean'], 3)} en hoogstens {nl(st['same_text_absdiff_max'], 3)}: MetricX rekent in "
        "bf16, en een segment staat in elke run in een andere batch [gemeten; de oorzaak is een gevolgtrekking].",
        f"- Het gemiddelde komt vooral uit een staart: de 10 grootste verschillen dragen "
        f"{nl(100 * st['share_of_sum_top10'], 0)}% van de som; zonder de 5, 10 of 20 grootste is het gemiddelde "
        f"{nl(st['mean_without_top']['5'], 3)}, {nl(st['mean_without_top']['10'], 3)} of "
        f"{nl(st['mean_without_top']['20'], 3)} [berekend uit de verschillen].",
        f"- Met referentie (dezelfde segmenten, WMT24++-referentie): MetricX-24 basis {nl(m['base']['ref'], 3)}, "
        f"variant {nl(m['instruction_tag']['ref'], 3)} (verschil {nl(m['instruction_tag']['ref'] - m['base']['ref'], 3)}); "
        f"COMET-22 basis {nl(m['base']['comet'], 4)}, variant {nl(m['instruction_tag']['comet'], 4)} (verschil "
        f"{nl(m['instruction_tag']['comet'] - m['base']['comet'], 4)}). Rangcorrelatie (Spearman) van het QE-verschil "
        f"per segment met het verschil in MetricX met referentie: {nl(st['spearman_qe_ref'], 2)}; met het verschil in "
        f"COMET-22: {nl(st['spearman_qe_comet'], 2)} (negatief = samen slechter).",
        f"- Controle: de QE die hier opnieuw is berekend (met de referentiescores in dezelfde run) wijkt hoogstens "
        f"{nl(st['qe_again_max_abs'], 4)} af van de QE van de tagproef; de tabellen tonen die van de tagproef.", "",
        "| QE-verschil per segment | segmenten |", "|---|---:|"]
    lines += [f"| {hist_labels[k]} | {v} |" for k, v in st["hist"].items()]
    lines += ["", "Lezing [eigen gevolgtrekking]: een gemiddeld verschil van 0,2 betekent niet dat elk segment 0,2 "
              "slechter is. De meeste segmenten liggen dicht bij 0, en een klein aantal segmenten met een groot verschil "
              "bepaalt het gemiddelde. De voorbeelden hieronder laten per klasse zien hoe zo'n verschil eruitziet.", "",
              "## Voorbeelden", "",
              f"Per klasse {RULE_PER_CLASS} segmenten, getrokken met seed `{SEED}/rule/<klasse>` uit alle segmenten "
              "waarvan het QE-verschil in het interval van de klasse ligt (grenzen inclusief): "
              + "; ".join(f"{label}: [{nl(lo, 1)}; {nl(hi, 1)}], {chosen[label]['pool']} segmenten"
                          for label, lo, hi in RULE_CLASSES) + ". Een positief verschil: de variant scoort slechter.",
              ""]
    for label, lo, hi in RULE_CLASSES:
        title = "variant beter" if lo < 0 else "variant slechter"
        lines += [f"### Klasse {label} ({title})", ""]
        for i in chosen[label]["ids"]:
            s = segs[i]
            b, v = s["base"], s["instruction_tag"]
            lines += [f"#### {i} ({s['domain']}): QE-verschil {nl(s['diff'])}", "",
                      "| | MetricX-QE | MetricX met ref. | COMET-22 |", "|---|---:|---:|---:|",
                      f"| basis | {nl(b['qe'])} | {nl(b['ref'])} | {nl(b['comet'], 3)} |",
                      f"| variant | {nl(v['qe'])} | {nl(v['ref'])} | {nl(v['comet'], 3)} |",
                      f"| verschil | {nl(s['diff'])} | {nl(s['diff_ref'])} | {nl(s['diff_comet'], 3)} |", "",
                      "Engelse bron:", "", quote(s["source"]), "",
                      "Referentie (WMT24++):", "", quote(s["reference"]), "",
                      "Basis 12B:", "", quote(b["hypothesis"]), "",
                      "Variant (adapter met instructiezin, met de zin):", "", quote(v["hypothesis"]), ""]
    return "\n".join(lines) + "\n"


def cutoff(args):
    import pyarrow.parquet as pq
    import sft_build as sb
    from transformers import AutoTokenizer

    report = {}
    evalsets = pq.read_table(args.evalsets, columns=["source", "reference"]).to_pylist()
    blocked = sb.blocked_ngrams(evalsets)
    build_args = argparse.Namespace(run=args.run, long=args.long, qe_max=math.inf)
    rows = sb.candidates_bt(build_args, blocked, report)
    other = [r for r in rows if not r["fails"]]
    # The bound on tokens (an example fits 2,048 tokens): counted here for the passages above QE 6 in their plain
    # prompt; with a context that is too long the build drops the context, not the passage
    counter = sb.TokenCounter(AutoTokenizer.from_pretrained(args.tokenizer))
    high = [r for r in other if r["qe_en_nl"] > min(CUTOFF_BOUNDS)]
    prompts = counter.prompt_lengths([sb.user_turn("chat-v1", r["en"]) for r in high])
    targets = counter.target_lengths([r["nl"].strip() for r in high])
    too_long = {r["id"] for r, p, t in zip(high, prompts, targets) if p + t > sb.MAX_TOKENS}
    other = [r for r in other if r["id"] not in too_long]
    train = [r for r in other if r["split"] == "train"]
    evals = [r for r in other if r["split"] == "eval"]
    # context turns that pass the other bounds of the query; a turn above the bound ends its passage's context there
    cols = ["id", "pid", "kind", "nl", "en", "hard_fails", "qe_en_nl"]
    context_pairs = {r["pid"]: r for r in pq.read_table(Path(args.run) / "pairs.parquet", columns=cols).to_pylist()
                     if r["kind"] == "context"}
    turns = [t for t in context_pairs.values() if not sb.query_fails(t, None, blocked, math.inf)]
    turn_rows = {t["id"]: t for t in pq.read_table(Path(args.run) / "context-turns.parquet",
                                                   columns=["id", "speaker", "party", "source_score"]).to_pylist()}
    marked = [r for r in train if r["passage"]["context_ids"]]

    def context_lost(qe_max):
        """passages within the bound that have a context without a QE bound (candidates_bt ran with an infinite one) and
        lose all of it under this bound (before the bound on tokens): they stay as plain examples"""
        return sum(r["qe_en_nl"] <= qe_max and bool(r["context"]) and not sb.context_of(
            r["passage"], context_pairs, turn_rows, blocked, qe_max)[0] for r in marked)

    ctx_inf = sum(bool(r["context"]) for r in marked)
    costs = []
    for c in CUTOFF_BOUNDS:
        costs.append({"bound": c, "train": sum(r["qe_en_nl"] > c for r in train),
                      "eval": sum(r["qe_en_nl"] > c for r in evals),
                      "turns": sum(t["qe_en_nl"] > c for t in turns),
                      "context_examples_lost": context_lost(c)})
    by_source = collections.Counter(r["source"] for r in train)
    over_source = {c: collections.Counter(r["source"] for r in train if r["qe_en_nl"] > c) for c in CUTOFF_BOUNDS}
    chosen = []
    for label, lo, hi, k in CUTOFF_BANDS:
        pool = sorted((r for r in train if lo < r["qe_en_nl"] <= hi), key=lambda r: r["id"])
        pick = random.Random(f"{SEED}/cutoff/{label}").sample(pool, min(k, len(pool)))
        chosen.append({"band": label, "pool": len(pool),
                       "pairs": [{"id": r["id"], "source": r["source"], "length_class": r["length_class"],
                                  "qe": r["qe_en_nl"], "chrf": r["chrf"], "en": r["en"], "nl": r["nl"]}
                                 for r in sorted(pick, key=lambda r: r["qe_en_nl"])]})
    qe_train = sorted(r["qe_en_nl"] for r in train)
    out = {"train_other_bounds": len(train), "eval_other_bounds": len(evals), "turns_other_bounds": len(turns),
           "too_long_above_6": len(too_long), "context_examples_without_qe_bound": ctx_inf,
           "by_source": dict(by_source), "over_source": {str(c): dict(v) for c, v in over_source.items()},
           "costs": costs, "chosen": chosen,
           "quantiles": {f"p{q}": quantile(qe_train, q / 100) for q in (50, 90, 95, 99, 99.5, 99.9)},
           "max": qe_train[-1]}
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    Path(args.out).write_text(cutoff_md(out), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("train_other_bounds", "eval_other_bounds", "too_long_above_6")}))


def cutoff_md(o):
    sources = sorted(o["by_source"])
    lines = [
        "# Voorbeelden bij de grens QE ≤ 7,5 in de trainingsdata", "",
        "Voor Yebs open punt 5(b) (protocol §2.1 punt 3: MetricX-24-QE en→nl ≤ 7,5 tegen uitschieters). Gemaakt op "
        "10 nov 2026 met `examples_qe.py cutoff` (branch `en-nl-sft`) op mbert-mix, uit de pairs-tabellen "
        "`gs://t5-dutch-english-usc2/mbert/translator/mainset/run/pairs/pairs.parquet` en "
        "`…/mainset/run-long/pairs/pairs.parquet`. Elk paar is een BT-paar: Engels dat TranslateGemma 27B uit het "
        "Nederlandse origineel maakte (synthetisch), en het originele Nederlands. De QE scoort het Nederlands als "
        "vertaling van het Engels; een hoge QE betekent dat Engels en Nederlands slecht bij elkaar passen (lager is "
        "beter).", "",
        f"Alleen trainpassages die door alle andere grenzen van de query komen (`hard_fails` leeg, bronscore en "
        f"herkomst, geen 13-gram met een evaluatieset, ≤ 2.048 tokens): {n_(o['train_other_bounds'])} "
        f"(eval: {n_(o['eval_other_bounds'])}). Van de passages met QE > 6 past er {o['too_long_above_6']} niet in "
        f"2.048 tokens. QE van die trainpassages: "
        + ", ".join(f"{k} {nl(v, 3)}" for k, v in o["quantiles"].items()) + f", max {nl(o['max'], 2)}.", "",
        "## Wat elke grens kost", "",
        "Trainpassages, evalpassages en contextbeurten met QE boven de grens (zij vallen af); "
        "contextvoorbeelden die hun hele context verliezen doordat de dichtstbijzijnde beurt boven de grens ligt (de "
        "passage blijft dan als gewoon voorbeeld, zie protocol §2.2). Bij de huidige grens 7,5 telt de query 2.117 "
        "trainpassages die als eerste op QE vallen (protocol §2.1); hier staan alle passages boven de grens die door de "
        "andere grenzen komen.", "",
        "| grens QE ≤ | train weg | aandeel | eval weg | contextbeurten weg | contextvoorbeelden weg | "
        + " | ".join(sources) + " |",
        "|---:|---:|---:|---:|---:|---:|" + "---:|" * len(sources)]
    for c in o["costs"]:
        b = c["bound"]
        lines.append(f"| {nl(b, 1)} | {n_(c['train'])} | {nl(100 * c['train'] / o['train_other_bounds'], 2)}% | "
                     f"{n_(c['eval'])} | {n_(c['turns'])} | {n_(c['context_examples_lost'])} | "
                     + " | ".join(n_(o["over_source"][str(b)].get(s, 0)) for s in sources) + " |")
    lines += ["", "Per bron in aanmerking zonder QE-grens (train): "
              + ", ".join(f"{s} {n_(o['by_source'][s])}" for s in sources) + ".", "",
              "## Paren per QE-band", "",
              f"Per band getrokken met seed `{SEED}/cutoff/<band>` uit alle trainpassages in die band (ondergrens "
              "exclusief, bovengrens inclusief), gesorteerd op QE. Teksten langer dan "
              f"{n_(MAX_CHARS)} tekens zijn afgekapt (`[…]`). Banden: "
              + "; ".join(f"{c['band']}: {n_(c['pool'])} passages" for c in o["chosen"]) + ".", ""]
    for c in o["chosen"]:
        lines += [f"### QE {c['band']}", ""]
        for p in c["pairs"]:
            lines += [f"#### {p['id']}", "",
                      f"QE {nl(p['qe'])}, chrF {nl(p['chrf'], 1)}; bron: {p['source']}"
                      f"{', lange passage (385–511 tokens)' if p['length_class'] == 'long' else ''}.", "",
                      "Synthetisch Engels (TranslateGemma 27B, nl→en):", "", quote(p["en"]), "",
                      "Nederlands origineel:", "", quote(p["nl"]), ""]
    return "\n".join(lines) + "\n"


LENGTH_FACTORS = [1.2, 1.25, 1.3333, 1.5]
LENGTH_PER_SIDE = (4, 2)  # pairs just outside, just inside the bound, per side and group
LENGTH_BAND = 1.06        # "just": within this factor of the bound


def length_groups(args):
    """{group: rows (id, source, en, nl, qe_en_nl)}: 'bt' (training passages, hard_fails empty, QE <= qe_max) and per
    real source 'real/<source>' (hard checks passed; QE <= qe_max where measured)"""
    import pyarrow.parquet as pq

    cols = ["id", "pid", "kind", "source", "split", "nl", "en", "hard_fails", "qe_en_nl"]
    rows = pq.read_table(Path(args.run) / "pairs.parquet", columns=cols).to_pylist() + \
        pq.read_table(Path(args.long) / "pairs.parquet", columns=cols).to_pylist()
    bt_all = [r for r in rows if r["kind"] == "passage" and r["split"] == "train" and not r["hard_fails"]]
    groups = {"bt": (bt_all, [r for r in bt_all if r["qe_en_nl"] is not None and r["qe_en_nl"] <= args.qe_max])}
    if args.real:
        real = [r for r in pq.read_table(args.real).to_pylist() if not r["fails"]]
        for source in sorted({r["source"] for r in real}):
            mine = [r for r in real if r["source"] == source]
            kept = [r for r in mine if r.get("qe_en_nl") is None or r["qe_en_nl"] <= args.qe_max]
            groups[f"real/{source}"] = (mine, kept)
    return groups


def length(args):
    import sft_build as sb

    out = {"factor": args.factor, "qe_max": args.qe_max, "groups": {}}
    for group, (base_rows, rows) in length_groups(args).items():
        bounds = sb.length_bounds(base_rows, args.factor)  # the medians over the hard checks, as sft_build.py
        g = {"n": len(rows), "with_qe": sum(r.get("qe_en_nl") is not None for r in rows), "sources": {}, "costs": [],
             "examples": {}}
        for source, b in bounds.items():
            mine = [r for r in rows if r["source"] == source]
            cr = sorted(sb.char_ratio(r["nl"], r["en"]) for r in mine)
            wr = sorted(sb.word_ratio(r["nl"], r["en"]) for r in mine)
            g["sources"][source] = {"n": len(mine), "median_chars": b["median"], "median_words": sb.median(wr) if wr else None,
                                    "lo": b["lo"], "hi": b["hi"], "below": sum(x < b["lo"] for x in cr),
                                    "above": sum(x > b["hi"] for x in cr),
                                    "p1": quantile(cr, 0.01), "p99": quantile(cr, 0.99)}
        for f in LENGTH_FACTORS:
            fb = sb.length_bounds(base_rows, f)
            below = sum(sb.char_ratio(r["nl"], r["en"]) < fb[r["source"]]["lo"] for r in rows)
            above = sum(sb.char_ratio(r["nl"], r["en"]) > fb[r["source"]]["hi"] for r in rows)
            g["costs"].append({"factor": f, "below": below, "above": above})
        rng = random.Random(f"{SEED}/length/{group}")
        for side in ("below", "above"):
            for where, k in zip(("outside", "inside"), LENGTH_PER_SIDE):
                def near(r):
                    x, b = sb.char_ratio(r["nl"], r["en"]), bounds[r["source"]]
                    if side == "below":
                        return b["lo"] / LENGTH_BAND <= x < b["lo"] if where == "outside" else b["lo"] <= x < b["lo"] * LENGTH_BAND
                    return b["hi"] < x <= b["hi"] * LENGTH_BAND if where == "outside" else b["hi"] / LENGTH_BAND < x <= b["hi"]
                pool = sorted((r for r in rows if near(r) and len(r["en"]) <= 1200), key=lambda r: r["id"])
                pick = rng.sample(pool, min(k, len(pool)))
                g["examples"][f"{side}/{where}"] = {"pool": len(pool), "pairs": [
                    {"id": r["id"], "source": r["source"], "chars": sb.char_ratio(r["nl"], r["en"]),
                     "words": sb.word_ratio(r["nl"], r["en"]), "qe": r.get("qe_en_nl"), "en": r["en"], "nl": r["nl"]}
                    for r in pick]}
        out["groups"][group] = g
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    Path(args.out).write_text(length_md(out), encoding="utf-8")
    print("LENGTH " + json.dumps({k: {"n": g["n"], "costs": g["costs"]} for k, g in out["groups"].items()}))


GROUP_NAMES = {"bt": "BT-paren (synthetisch Engels → origineel Nederlands)", "real/europarl": "Europarl-direct en→nl",
               "real/globalvoices": "Global Voices en→nl"}


def length_md(o):
    f = o["factor"]
    L = ["# Voorbeelden bij de lengtegrens van arm R2", "",
         f"Maat per paar: tekens van het Nederlands gedeeld door tekens van het Engels. Grens: binnen een factor {nl(f, 4)} "
         "van de mediaan van de bron (mediaan / factor tot mediaan × factor; de medianen over de paren die door de harde "
         "controles komen), aan beide kanten. Tekens en niet woorden, omdat Nederlandse samenstellingen het aantal woorden "
         "drukken (\"gladheidbestrijdingsplan\"); de mediaan in woorden staat ernaast. Gemaakt met `examples_qe.py length` "
         f"(branch `en-nl-sft`); BT: trainpassages met lege `hard_fails` en MetricX-QE ≤ {nl(o['qe_max'], 1)}; echte paren: "
         "kandidaten van `real_pairs.py` die door de harde controles komen, met QE-grens waar de QE gemeten is.", ""]
    for group, g in o["groups"].items():
        L += [f"## {GROUP_NAMES.get(group, group)}", "",
              f"{n_(g['n'])} paren ({n_(g['with_qe'])} met MetricX-QE).", "",
              "| bron | n | mediaan tekens | mediaan woorden | ondergrens | bovengrens | eronder | erboven | p1 | p99 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for s, v in g["sources"].items():
            L.append(f"| {s} | {n_(v['n'])} | {nl(v['median_chars'], 3)} | {nl(v['median_words'], 3)} | {nl(v['lo'], 3)} | "
                     f"{nl(v['hi'], 3)} | {n_(v['below'])} | {n_(v['above'])} | {nl(v['p1'], 3)} | {nl(v['p99'], 3)} |")
        L += ["", "Wat de grens kost bij andere factoren (paren eronder / erboven):", "",
              "| factor | eronder | erboven | samen | aandeel |", "|---:|---:|---:|---:|---:|"]
        for c in g["costs"]:
            total = c["below"] + c["above"]
            L.append(f"| {nl(c['factor'], 4)}{' (gekozen)' if abs(c['factor'] - f) < 1e-9 else ''} | {n_(c['below'])} | "
                     f"{n_(c['above'])} | {n_(total)} | {nl(100 * total / g['n'], 2) if g['n'] else '–'}% |")
        names = {"below/outside": "Net onder de ondergrens (valt af)", "below/inside": "Net boven de ondergrens (blijft)",
                 "above/outside": "Net boven de bovengrens (valt af)", "above/inside": "Net onder de bovengrens (blijft)"}
        for key, title in names.items():
            e = g["examples"][key]
            L += ["", f"### {title} ({n_(e['pool'])} paren binnen {nl(LENGTH_BAND, 2)} × de grens)", ""]
            for x in e["pairs"]:
                L += [f"**`{x['id']}`** ({x['source']}; tekens {nl(x['chars'], 3)}, woorden {nl(x['words'], 3)}; QE "
                      f"{nl(x['qe'])})", "", "Engels:", "", quote(x["en"], 1500), "", "Nederlands:", "", quote(x["nl"], 1500), ""]
        L.append("")
    return "\n".join(L) + "\n"


EDGE_QE_BOUNDS = [3, 4, 5, 6, 7.5, 10]
EDGE_PER_SIDE = 3      # pairs just outside and just inside a bound, per side
EDGE_QE_BAND = 1.0     # "just" at the QE bound: within this many points of it
EDGE_HIGHEST = 3       # the pairs with the highest QE


def real_edges(args):
    import pyarrow.parquet as pq
    import sft_build as sb

    rows = [r for r in pq.read_table(args.real).to_pylist() if r["source"] == args.source]
    if not rows:
        raise SystemExit(f"{args.real}: no pair of source {args.source}")
    hard_ok = [r for r in rows if not r["fails"]]
    missing = [r["id"] for r in hard_ok if r.get("qe_en_nl") is None]
    if missing:
        raise SystemExit(f"{len(missing)} pairs that pass the hard checks have no MetricX-QE, e.g. {missing[0]}")
    b = sb.length_bounds(hard_ok, args.factor)[args.source]  # the median over the hard checks, as sft_build.py

    def author(r):
        return json.loads(r["meta"] or "{}").get("author") or "–"

    def ratio(r):
        return sb.char_ratio(r["nl"], r["en"])

    def length_ok(r):
        return b["lo"] <= ratio(r) <= b["hi"]

    per = collections.defaultdict(collections.Counter)
    for r in rows:
        c = per[author(r)]
        c["rows"] += 1
        if r["fails"]:
            c["hard:" + r["fails"][0]] += 1
        elif r["qe_en_nl"] > args.qe_max:
            c["qe"] += 1
        elif not length_ok(r):
            c["length"] += 1
        else:
            c["kept"] += 1
    total = collections.Counter()
    for c in per.values():
        total.update(c)
    qe = sorted(r["qe_en_nl"] for r in hard_ok)
    qe_ok = [r for r in hard_ok if r["qe_en_nl"] <= args.qe_max]
    rng = random.Random(f"{SEED}/real-edges/{args.source}")

    def pick(pool, k=EDGE_PER_SIDE):
        pool = sorted(pool, key=lambda r: r["id"])
        return [{"id": r["id"], "author": author(r), "qe": r["qe_en_nl"], "chars": ratio(r),
                 "words": sb.word_ratio(r["nl"], r["en"]), "en": r["en"], "nl": r["nl"]}
                for r in rng.sample(pool, min(k, len(pool)))]

    examples = {
        "qe/outside": pick([r for r in hard_ok if length_ok(r) and args.qe_max < r["qe_en_nl"] <= args.qe_max + EDGE_QE_BAND]),
        "qe/inside": pick([r for r in hard_ok if length_ok(r) and args.qe_max - EDGE_QE_BAND < r["qe_en_nl"] <= args.qe_max]),
        "qe/highest": [{"id": r["id"], "author": author(r), "qe": r["qe_en_nl"], "chars": ratio(r),
                        "words": sb.word_ratio(r["nl"], r["en"]), "en": r["en"], "nl": r["nl"]}
                       for r in sorted(hard_ok, key=lambda r: (-r["qe_en_nl"], r["id"]))[:EDGE_HIGHEST]],
        "length/below_outside": pick([r for r in qe_ok if b["lo"] / LENGTH_BAND <= ratio(r) < b["lo"]]),
        "length/below_inside": pick([r for r in qe_ok if b["lo"] <= ratio(r) < b["lo"] * LENGTH_BAND]),
        "length/above_outside": pick([r for r in qe_ok if b["hi"] < ratio(r) <= b["hi"] * LENGTH_BAND]),
        "length/above_inside": pick([r for r in qe_ok if b["hi"] / LENGTH_BAND < ratio(r) <= b["hi"]]),
    }
    out = {"source": args.source, "rows": len(rows), "hard_ok": len(hard_ok), "qe_max": args.qe_max,
           "factor": args.factor, "bounds": b,
           "qe": {f"p{q}": quantile(qe, q / 100) for q in (10, 25, 50, 75, 90, 95, 99)},
           "qe_costs": {str(c): sum(x > c for x in qe) for c in EDGE_QE_BOUNDS},
           "length_costs_after_qe": {"below": sum(ratio(r) < b["lo"] for r in qe_ok),
                                     "above": sum(ratio(r) > b["hi"] for r in qe_ok)},
           "length_costs_alone": {"below": sum(ratio(r) < b["lo"] for r in hard_ok),
                                  "above": sum(ratio(r) > b["hi"] for r in hard_ok)},
           "total": dict(total), "authors": {a: dict(c) for a, c in sorted(per.items())}, "examples": examples}
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    Path(args.out).write_text(real_edges_md(out), encoding="utf-8")
    print("REAL_EDGES " + json.dumps({k: out[k] for k in ("source", "rows", "hard_ok", "qe_costs",
                                                         "length_costs_after_qe", "total")}))


def real_edges_md(o):
    t = o["total"]
    hard = sum(v for k, v in t.items() if k.startswith("hard:"))
    L = [f"# Grenzen op de echte paren van bron {o['source']} (arm RB)", "",
         f"Gemaakt met `examples_qe.py real-edges` (branch `en-nl-sft`). Volgorde zoals `sft_build.py`: harde controles "
         f"(`real_pairs.py`), dan MetricX-24-QE en→nl ≤ {nl(o['qe_max'], 1)}, dan de lengtegrens (tekens NL / tekens EN "
         f"binnen een factor {nl(o['factor'], 4)} van de mediaan {nl(o['bounds']['median'], 3)}: "
         f"{nl(o['bounds']['lo'], 3)} tot {nl(o['bounds']['hi'], 3)}; de mediaan over de {n_(o['hard_ok'])} paren die "
         "door de harde controles komen). De 13-grammen en de tokens volgen in `sft_build.py` (zijn rapport).", "",
         f"Van {n_(o['rows'])} paren: harde controles {n_(hard)}, QE {n_(t.get('qe', 0))}, lengtegrens "
         f"{n_(t.get('length', 0))}; over {n_(t.get('kept', 0))}.", "",
         "| auteur | paren | harde controles | QE > grens | lengtegrens | over | aandeel over |",
         "|---|---:|---:|---:|---:|---:|---:|"]
    for a, c in sorted(o["authors"].items(), key=lambda x: -x[1]["rows"]):
        h = sum(v for k, v in c.items() if k.startswith("hard:"))
        hk = ", ".join(f"{k[5:]} {v}" for k, v in sorted(c.items()) if k.startswith("hard:"))
        L.append(f"| {a} | {n_(c['rows'])} | {n_(h)}{f' ({hk})' if hk else ''} | {n_(c.get('qe', 0))} | "
                 f"{n_(c.get('length', 0))} | {n_(c.get('kept', 0))} | {nl(100 * c.get('kept', 0) / c['rows'], 1)}% |")
    L += ["", "MetricX-24-QE van de paren die door de harde controles komen: "
          + ", ".join(f"{k} {nl(v)}" for k, v in o["qe"].items()) + ".", "",
          "| QE > | paren (na de harde controles) |", "|---:|---:|"]
    L += [f"| {nl(float(c), 1)}{' (grens)' if abs(float(c) - o['qe_max']) < 1e-9 else ''} | {n_(n)} |"
          for c, n in o["qe_costs"].items()]
    la, lq = o["length_costs_alone"], o["length_costs_after_qe"]
    L += ["", f"Lengtegrens: eronder / erboven {n_(la['below'])} / {n_(la['above'])} van de paren na de harde controles; "
          f"{n_(lq['below'])} / {n_(lq['above'])} van die ook de QE-grens halen.", ""]
    names = {"qe/outside": f"Net boven de QE-grens (valt af; QE tot {nl(o['qe_max'] + EDGE_QE_BAND, 1)})",
             "qe/inside": f"Net onder de QE-grens (blijft; QE vanaf {nl(o['qe_max'] - EDGE_QE_BAND, 1)})",
             "qe/highest": "De hoogste QE (valt af)",
             "length/below_outside": "Net onder de ondergrens van de lengte (valt af)",
             "length/below_inside": "Net boven de ondergrens van de lengte (blijft)",
             "length/above_outside": "Net boven de bovengrens van de lengte (valt af)",
             "length/above_inside": "Net onder de bovengrens van de lengte (blijft)"}
    for key, title in names.items():
        L += [f"## {title}", ""]
        for x in o["examples"][key]:
            L += [f"**`{x['id']}`** ({x['author']}; QE {nl(x['qe'])}; tekens {nl(x['chars'], 3)}, woorden "
                  f"{nl(x['words'], 3)})", "", "Engels:", "", quote(x["en"], 1500), "", "Nederlands:", "",
                  quote(x["nl"], 1500), ""]
        if not o["examples"][key]:
            L += ["(geen paar in deze band)", ""]
    return "\n".join(L) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Examples for the QE rule and the QE bound")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("rule-pairs")
    p.add_argument("--pairs", required=True, help="tagtest/report/pairs.parquet")
    p.add_argument("--items", required=True, help="tagtest/segments/items.parquet")
    p.add_argument("--out", required=True)
    r = sub.add_parser("rule")
    r.add_argument("--pairs", required=True, help="tagtest/report/pairs.parquet")
    r.add_argument("--qe", required=True, help="tagtest/report/qe.parquet")
    r.add_argument("--items", required=True)
    r.add_argument("--scores", required=True, help="metricx_qe.py of rule-pairs (reference and QE)")
    r.add_argument("--comet", required=True, help="comet_score.py of rule-pairs")
    r.add_argument("--out", required=True)
    r.add_argument("--json", default=None)
    c = sub.add_parser("cutoff")
    c.add_argument("--run", required=True)
    c.add_argument("--long", required=True)
    c.add_argument("--evalsets", required=True)
    c.add_argument("--tokenizer", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--json", default=None)
    g = sub.add_parser("length")
    g.add_argument("--run", required=True)
    g.add_argument("--long", required=True)
    g.add_argument("--real", default=None, help="real_pairs.py candidates (real.parquet or real-qe.parquet)")
    g.add_argument("--qe-max", type=float, default=5.0)
    g.add_argument("--factor", type=float, default=1.3333)
    g.add_argument("--out", required=True)
    g.add_argument("--json", default=None)
    e = sub.add_parser("real-edges")
    e.add_argument("--real", required=True, help="real_pairs.py join-qe (real-qe.parquet)")
    e.add_argument("--source", required=True)
    e.add_argument("--qe-max", type=float, default=5.0)
    e.add_argument("--factor", type=float, default=1.3333)
    e.add_argument("--out", required=True)
    e.add_argument("--json", default=None)
    args = parser.parse_args(argv)
    {"rule-pairs": rule_pairs, "rule": rule, "cutoff": cutoff, "length": length,
     "real-edges": real_edges}[args.command](args)


if __name__ == "__main__":
    main()
