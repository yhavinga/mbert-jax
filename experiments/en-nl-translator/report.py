#!/usr/bin/env python3
"""The pilot's report for Yeb (hammond mbert/designs/en-nl-translator.md, §3 step 2): per filter what it measures and
what would fall off at a few candidate thresholds, the distributions, the throughput, 30 random pairs stratified by
source and a handful of filtered cases per filter. It fixes no threshold.

    python report.py --pairs pairs.parquet --qe qe.parquet --select-report report.json --logs <dir of logs> \\
        --out <dir>

Writes <out>/report.md (Dutch), <out>/pairs30.md (Dutch headings, the texts as they are) and <out>/summary.json.
"""

import argparse
import collections
import json
import random
import re
import sys
from pathlib import Path

SOURCES = ["handelingen", "raad", "dpc", "rechtspraak"]
SOURCE_NAMES = {"handelingen": "Handelingen TK", "raad": "raadsinformatie", "dpc": "dpc", "rechtspraak": "rechtspraak"}
CHRF = [30, 40, 50, 60, 70]
QE = [2, 3, 5, 7, 10]
RATIO = [1.3, 1.5, 2.0]
SAMPLE = {"handelingen": 15, "raad": 9, "dpc": 3, "rechtspraak": 3}
# The filters of §3 step 2 whose verdict is yes or no; chrF and QE get candidate thresholds instead
BINARY = {
    "niet_afgelopen": ("ended = false: de nl→en-vertaling stopte op de decodeerlimiet", lambda r: not r["ended"]),
    "decode_loop": ("`decode_loop` van FLAN-nl (`publish.py`), Engels tegen zijn Nederlandse bron",
                    lambda r: bool(r["decode_loop"])),
    "nederlands_over": ("Nederlandse resten in het Engels (`dutch_left`, spiegeling van `english_left`)",
                        lambda r: bool(r["dutch_left"])),
    "lengte_1_5": ("lengteverhouding in tekens (langste/kortste) > 1,5", lambda r: r["len_ratio"] > 1.5),
    "zinnen": ("ander aantal zinnen (Nederlands tegen Engels)", lambda r: r["sentences_nl"] != r["sentences_en"]),
    "getallen": ("getal uit het Nederlands ontbreekt in het Engels", lambda r: bool(r["numbers_missing"])),
}


def pct(values, qs=(1, 5, 10, 25, 50, 75, 90, 95, 99)):
    v = sorted(x for x in values if x is not None)
    if not v:
        return {}
    return {q: v[min(len(v) - 1, int(round(q / 100 * (len(v) - 1))))] for q in qs}


def fmt(x, digits=1):
    if x is None:
        return "–"
    if isinstance(x, float):
        return f"{x:.{digits}f}".replace(".", ",")
    return f"{x:,}".replace(",", ".")


def share(n, total):
    return f"{fmt(n)} ({fmt(100 * n / total if total else 0.0)}%)"


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(out)


def throughput(logs):
    """Per direction and worker: segments, tokens and seconds of the SHARD lines, and out tokens/s"""
    out = collections.defaultdict(lambda: collections.Counter())
    for path in sorted(Path(logs).glob("translator_*-w*.log")):
        m = re.match(r"translator_(nl_en|en_nl)-w(\d)\.log", path.name)
        if not m:
            continue
        for line in path.read_text(errors="replace").splitlines():
            if line.startswith("SHARD "):
                s = json.loads(line[6:])
                c = out[(m.group(1), int(m.group(2)))]
                for key in ("segments", "tokens_in", "tokens_out", "seconds"):
                    c[key] += s[key]
                c["not_ended"] += s["not_ended"]
            elif "engine ready in" in line:
                out[(m.group(1), int(m.group(2)))]["load_seconds"] += int(re.search(r"in (\d+) s", line).group(1))
    return out


def flags(r):
    hit = [name for name, (_, fn) in BINARY.items() if fn(r)]
    if r["back"] is None:
        hit.append("niet_teruggevertaald")
    return hit


def pair_block(r, k, why=None):
    lines = [f"### {k}. {SOURCE_NAMES[r['source']]} — `{r['pid']}` ({r['half']})", ""]
    if why:
        lines += [f"**Reden:** {why}", ""]
    qe = r.get("qe_en_nl")
    qe2 = r.get("qe_nl_en")
    lines += [f"chrF terug/origineel: **{fmt(r['chrf'])}** · MetricX-24-QE en→nl: **{fmt(qe, 2)}** · "
              f"nl→en: {fmt(qe2, 2)} · lengteverhouding {fmt(r['len_ratio'], 2)} · zinnen nl/en "
              f"{r['sentences_nl']}/{r['sentences_en']} · tokens nl/en {r['tokens_nl']}/{r['tokens_en']}"
              + (f" · ontbrekende getallen: {', '.join(r['numbers_missing'])}" if r["numbers_missing"] else ""),
              "", f"Filters geraakt: {', '.join(flags(r)) or 'geen'}", ""]
    for title, text in (("Nederlands origineel", r["nl"]), ("Synthetisch Engels (TranslateGemma 27B, nl→en)", r["en"]),
                        ("Terugvertaling (TranslateGemma 27B, en→nl)", r["back"])):
        lines += [f"**{title}:**", ""]
        lines += ["> " + line if line else ">" for line in (text or "(geen)").split("\n")]
        lines += [""]
    return "\n".join(lines)


def main(argv=None):
    import pyarrow.parquet as pq

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--qe", required=True)
    parser.add_argument("--select-report", required=True)
    parser.add_argument("--logs", required=True)
    parser.add_argument("--metricx-model", required=True, help="Name of the MetricX-24 variant, for the text")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wordlist", help="OpenTaal's word list: count hyphenation artifacts in the originals")
    parser.add_argument("--metricx-note", default="", help="A line on how the MetricX port was checked")
    args = parser.parse_args(argv)

    rows = pq.read_table(args.pairs).to_pylist()
    qe = {r["id"]: r for r in pq.read_table(args.qe).to_pylist()}
    for r in rows:
        r.update({k: v for k, v in qe[r["id"]].items() if k != "id"})
    select = json.loads(Path(args.select_report).read_text())
    n = len(rows)
    by_source = {s: [r for r in rows if r["source"] == s] for s in SOURCES}
    summary = {"pairs": n}
    md = ["# Pilot en→nl-vertaler: back-translation van 2.000 passages", "",
          "Gemaakt door `experiments/en-nl-translator/report.py` (branch `en-nl-translator` van mbert-jax). Alle "
          "Nederlandse en Engelse teksten hieronder zijn de originelen en de uitvoer van TranslateGemma 27B, "
          "ongewijzigd. Er is geen drempel vastgesteld: per filter staat wat er bij een paar kandidaatdrempels zou "
          "afvallen.", ""]

    # Selection
    md += ["## Stap 0: selectie", ""]
    srows = []
    for s in SOURCES:
        v = select["sources"][s]
        srows.append([SOURCE_NAMES[s], fmt(v["documents"]["docs_kept"]), fmt(v["counts"]["passages"]),
                      fmt(v["passed_train"]), fmt(v["chosen"]), v["chosen_halves"].get("idiom", 0),
                      fmt(v["choice"]["idiom_class_threshold"], 2),
                      ", ".join(f"{k} {fmt(c)}" for k, c in sorted(v["fails"].items(), key=lambda x: -x[1]))])
    md += [table(["bron", "documenten", "kandidaatpassages", "door filters (train-kant)", "gekozen",
                  "uit hoogste idioomklasse", "idioomklasse vanaf (treffers/100 woorden)", "afgevallen per filter"],
                 srows), ""]
    idi = select["idioms"]
    md += [f"Idioomlijst: {fmt(idi['kept_found'] + idi['left_out'])} uitdrukkingen gevonden in de kandidaten; "
           f"{idi['left_out']} kwamen in meer dan {fmt(100 * idi['max_share'], 1)}% van de passages voor en tellen "
           f"niet mee (vaste verbindingen, zoals "
           f"{', '.join(repr(i) for i, _ in idi['left_out_list'][:6])}). Vaakst gevonden die wel meetellen: "
           f"{', '.join(f'{i!r} ({c})' for i, c in idi['kept_most_frequent'][:10])}.", ""]

    # Throughput
    tp = throughput(args.logs)
    md += ["## Tempo (TranslateGemma 27B, MaxText, één v4-host = 4 chips per worker)", ""]
    trows = []
    for (direction, w), c in sorted(tp.items()):
        trows.append([direction.replace("_", "→"), f"w{w}", fmt(c["segments"]), fmt(c["tokens_in"]),
                      fmt(c["tokens_out"]), fmt(float(c["seconds"])), fmt(c["tokens_out"] / max(c["seconds"], 1e-9)),
                      fmt((c["tokens_in"] + c["tokens_out"]) / max(c["seconds"], 1e-9)), fmt(c["load_seconds"]),
                      c["not_ended"]])
    md += [table(["richting", "worker", "segmenten", "prompt-tokens", "uitvoertokens", "decodeerseconden",
                  "uitvoertokens/s", "prompt+uitvoer tokens/s", "laden+compileren (s, alle klassen)",
                  "niet afgelopen"], trows), "",
           "Decodeerseconden tellen alleen het vertalen van de shards; het laden van het model en het compileren "
           "per lengteklasse staat apart. Met maar ~500 segmenten per worker vult de laatste ronde de slots niet, "
           "dus de tokens/s liggen onder die van een volle run.", ""]
    summary["throughput"] = {f"{d}-w{w}": dict(c) for (d, w), c in tp.items()}

    # Binary filters
    md += ["## Stap 2: filters met ja/nee", ""]
    brows, summary["binary"] = [], {}
    for name, (desc, fn) in BINARY.items():
        hit = [r for r in rows if fn(r)]
        summary["binary"][name] = {s: sum(fn(r) for r in by_source[s]) for s in SOURCES} | {"all": len(hit)}
        brows.append([f"`{name}`", desc, share(len(hit), n)] +
                     [share(summary["binary"][name][s], len(by_source[s])) for s in SOURCES])
    any_binary = [r for r in rows if any(fn(r) for _, fn in BINARY.values())]
    brows.append(["**samen**", "minstens één van de filters hierboven", share(len(any_binary), n)] +
                 [share(sum(any(fn(r) for _, fn in BINARY.values()) for r in by_source[s]), len(by_source[s]))
                  for s in SOURCES])
    md += [table(["filter", "wat", "valt af (alle)"] + [SOURCE_NAMES[s] for s in SOURCES], brows), ""]
    not_back = [r for r in rows if r["back"] is None]
    md += [f"Niet teruggevertaald (Engels niet afgelopen of leeg): {fmt(len(not_back))}; die hebben geen chrF.", ""]
    zd = collections.Counter(abs(r["sentences_nl"] - r["sentences_en"]) for r in rows)
    md += ["Verschil in aantal zinnen: " + ", ".join(f"{k}: {fmt(v)}" for k, v in sorted(zd.items())) + ".", ""]

    # Continuous measures
    md += ["## Verdelingen", ""]
    measures = {"chrF terug/origineel": "chrf", "MetricX-24-QE en→nl (bron Engels, kandidaat origineel)": "qe_en_nl",
                "MetricX-24-QE nl→en (bron origineel, kandidaat Engels)": "qe_nl_en",
                "lengteverhouding (tekens)": "len_ratio"}
    qs = (1, 5, 10, 25, 50, 75, 90, 95, 99)
    drows = []
    for label, key in measures.items():
        for s in [None] + SOURCES:
            part = rows if s is None else by_source[s]
            p = pct([r[key] for r in part], qs)
            drows.append([label if s is None else "", "alle" if s is None else SOURCE_NAMES[s]] +
                         [fmt(p.get(q), 2) for q in qs])
    md += [table(["maat", "bron"] + [f"p{q}" for q in qs], drows), "",
           "MetricX: lager is beter (0–25, als MQM-foutpunten); chrF: hoger is beter (0–100).", ""]
    summary["distributions"] = {key: {s or "all": pct([r[key] for r in (rows if s is None else by_source[s])])
                                      for s in [None] + SOURCES} for key in measures.values()}

    md += ["## Wat zou afvallen bij kandidaatdrempels", ""]
    crow = []
    for t in CHRF:
        crow.append([f"chrF < {t}"] + [share(sum(r["chrf"] is not None and r["chrf"] < t for r in part), len(part))
                                       for part in [rows] + [by_source[s] for s in SOURCES]])
    for key, label in (("qe_en_nl", "QE en→nl"), ("qe_nl_en", "QE nl→en")):
        for t in QE:
            crow.append([f"{label} > {t}"] + [share(sum(r[key] is not None and r[key] > t for r in part), len(part))
                                              for part in [rows] + [by_source[s] for s in SOURCES]])
    for t in (1, 2):
        crow.append([f"verschil in zinnen ≥ {t}"] + [share(sum(abs(r["sentences_nl"] - r["sentences_en"]) >= t
                                                            for r in part), len(part))
                                                     for part in [rows] + [by_source[s] for s in SOURCES]])
    for t in RATIO:
        crow.append([f"lengteverhouding > {fmt(t)}"] + [share(sum(r["len_ratio"] > t for r in part), len(part))
                                                        for part in [rows] + [by_source[s] for s in SOURCES]])
    md += [table(["drempel", "alle"] + [SOURCE_NAMES[s] for s in SOURCES], crow), ""]

    md += ["### Over na alle ja/nee-filters en een combinatie van chrF en QE en→nl", ""]
    base = [r for r in rows if not any(fn(r) for _, fn in BINARY.values()) and r["back"] is not None]
    grid = []
    for t in CHRF:
        grid.append([f"chrF ≥ {t}"] + [share(sum(r["chrf"] >= t and r["qe_en_nl"] is not None and r["qe_en_nl"] <= q
                                                 for r in base), n) for q in QE])
    md += [f"Na de ja/nee-filters (en met terugvertaling) blijven {share(len(base), n)} over. Daarvan, als deel van "
           f"alle {fmt(n)}:", "", table(["", *[f"QE en→nl ≤ {q}" for q in QE]], grid), ""]
    summary["kept_after_binary"] = len(base)
    summary["grid"] = {f"chrf>={t}": {f"qe<={q}": sum(r["chrf"] >= t and r["qe_en_nl"] is not None and
                                                      r["qe_en_nl"] <= q for r in base) for q in QE} for t in CHRF}

    md += ["## MetricX", "",
           f"Variant: `{args.metricx_model}` (MetricX-24 hybrid, bfloat16-gewichten), in QE-modus (zonder referentie), "
           "eigen JAX-implementatie op de TPU (`metricx_qe.py`), op de CPU gecontroleerd tegen transformers. Per paar "
           "twee scores: en→nl (het Engels als bron, het Nederlandse origineel als kandidaat: de trainingsrichting) en "
           "nl→en (het origineel als bron, het synthetische Engels als kandidaat: wat de back-translation verloor of "
           "toevoegde)."] + ([args.metricx_note] if args.metricx_note else []) + [""]
    if args.wordlist:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import select_passages as sp

        with open(args.wordlist, encoding="utf-8") as f:
            words = {line.strip().lower() for line in f if line.strip()}
        broken = {r["id"]: sp.hyphen_artifacts(r["nl"], words) for r in rows}
        hit = [r for r in rows if broken[r["id"]]]
        md += ["## Doeltekst: afbreekfouten in het origineel", "",
               "Woorden die een regelafbreking in de bron heeft laten staan (\"voor-ge-legd\", \"be-ant-woord\"): "
               "zonder koppeltekens wel in de OpenTaal-lijst, met niet. Dat raakt de doelkant van een trainingspaar. "
               "Dit filter (`hyphenation`) is na de selectie van de pilot aan stap 0 toegevoegd; hier staat wat het in de "
               f"pilot had weggelaten: {share(len(hit), n)} — "
               + ", ".join(f"{SOURCE_NAMES[s]} {sum(r['source'] == s for r in hit)}" for s in SOURCES) + ".", ""]
        md += [f"- `{r['pid']}`: {', '.join(broken[r['id']][:5])}" for r in hit[:8]] + [""]
        summary["hyphenation"] = {s: sum(r["source"] == s for r in hit) for s in SOURCES} | {"all": len(hit)}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text("\n".join(md))

    # 30 pairs and the filtered cases
    rng = random.Random(args.seed)
    sample = []
    for s in SOURCES:
        sample += rng.sample(by_source[s], SAMPLE[s])
    p = ["# Pilot en→nl-vertaler: 30 paren en weggefilterde gevallen", "",
         "30 willekeurige paren (seed 0), gestratificeerd naar bron (15 Handelingen, 9 raadsinformatie, 3 dpc, 3 "
         "rechtspraak, zoals 50/30/10/10), ongeacht de filters. Daarna per filter een handvol gevallen die het filter "
         "raakt. De teksten zijn het origineel en de uitvoer van TranslateGemma 27B (greedy), ongewijzigd. chrF: "
         "terugvertaling tegen origineel (0–100, hoger is beter); MetricX-24-QE: 0–25, lager is beter.", "",
         "## 30 willekeurige paren", ""]
    for k, r in enumerate(sample, 1):
        p += [pair_block(r, k)]
    p += ["## Weggefilterde gevallen per filter", ""]
    k = 0
    for name, (desc, fn) in BINARY.items():
        hit = [r for r in rows if fn(r)]
        p += [f"### Filter `{name}`: {desc} ({fmt(len(hit))} paren)", ""]
        for r in rng.sample(hit, min(4, len(hit))):
            k += 1
            p += [pair_block(r, f"{name}-{k}", why=desc)]
    for key, label, worst in (("chrf", "laagste chrF", lambda r: r["chrf"]),
                              ("qe_en_nl", "slechtste QE en→nl", lambda r: -r["qe_en_nl"]),
                              ("qe_nl_en", "slechtste QE nl→en", lambda r: -r["qe_nl_en"])):
        scored = sorted((r for r in rows if r[key] is not None), key=worst)
        p += [f"### {label[0].upper() + label[1:]} (de 4 met de {label})", ""]
        for r in scored[:4]:
            k += 1
            p += [pair_block(r, f"{key}-{k}", why=f"{label}: {fmt(r[key], 2)}")]
    (out / "pairs30.md").write_text("\n".join(p))
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(f"REPORT {out}/report.md, {out}/pairs30.md")


if __name__ == "__main__":
    sys.exit(main())
