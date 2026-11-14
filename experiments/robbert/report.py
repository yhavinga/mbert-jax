#!/usr/bin/env python3
"""The report of experiments/robbert/protocol.md: RobBERT-2023, RobBERT-2026, T2 and S per task (mean and seed range
over 42, 43, 44), and per pair of models the comparison of experiments/final/decide.py: "beter" or "slechter" when the
difference of the means is more than the larger of the two seed ranges (BLiMP-NL: more than 0.5 point, 42 of 8,400
pairs), else "geen verschil". Exact as decide.py: counts k / n and the Fractions of stored F1s, every result checked
against the protocol's settings; a missing or other result is FOUT (exit code 1).

Usage (where gsutil reads the bucket; ~90 small result files):
    python3 -I experiments/robbert/report.py --output-dir ~/Developer/yhavinga/mbert-ops/dx/final \\
        [--code-commit <commit of the runs>] [--report-commit <commit of this file>]
Results: RobBERT and T2's DBRD@512 under <root>/robbert/ (eval_robbert.sh), T2's and S's other tasks under <root>/
(the final run's evaluation). Writes robbert-compare.md and robbert-compare.json; internal, as they hold CoNLL-2002
numbers. decide.py is imported from this file's directory (the chain copies both) or from experiments/final/.
"""

import argparse
import datetime
import json
import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(1, str(Path(__file__).resolve().parents[1] / "final"))

import decide  # noqa: E402

COLUMNS = ["robbert-2023", "robbert-2026", "T2", "S"]
LABEL = {"robbert-2023": "RobBERT-2023", "robbert-2026": "RobBERT-2026", "T2": "T2", "S": "S"}
FINAL_NAME = {"T2": decide.ARMS["T2"]["name"], "S": decide.ARMS["S"]["name"]}
PRIMARY_512 = {
    "dbrd512": {"n": 2112, "update": 192, "checkpoint_name": False,
                "settings": {"task": "dbrd", "learning_rate": 5e-5, "epochs": 3, "max_length": 512, "pad_to_multiple_of": 128}},
    "sicknl": {"n": 4864, "update": 256, "checkpoint_name": False,
               "settings": {"task": "sicknl", "learning_rate": 5e-5, "epochs": 3, "max_length": 512, "pad_to_multiple_of": 1}},
}
BLIMP_HF_SETTINGS = {"metric": "PLL-word-l2r", "max_pairs": None, "compute_dtype": "float32", "pairs": decide.BLIMP_PAIRS}
# (key, task, metric) in the order of the table
ROWS = [("dbrd512", "DBRD@512", "acc"), ("dbrd1024", "DBRD@1024", "acc"), ("sicknl", "SICK-NL", "acc"),
        ("squadnl", "SQuAD-NL v1.1", "F1"), ("squadnl-em", "SQuAD-NL v1.1", "EM"), ("conll2002", "CoNLL-2002", "entity-F1"),
        ("pos-lassysmall", "POS LassySmall", "UPOS-acc"), ("pos-alpino", "POS Alpino", "UPOS-acc"),
        ("blimp", "BLiMP-NL", "acc, PLL-word-l2r")]
PAIRS = [("T2", "robbert-2023"), ("T2", "robbert-2026"), ("robbert-2026", "robbert-2023")]
# pdelobelle/robbert-2026, README.md at a406b1a3, section "Results" (DUMB test split, 2 epochs, LR 5e-5, bf16)
MAKERS = {"squadnl": (69.3, 69.6), "squadnl-em": (46.7, 65.5), "dbrd512": (93.5, 93.3), "sicknl": (85.9, 82.4)}


def squad_em(read, root, name, seeds=decide.SEEDS):
    """EM per seed as k / questions (squad_scores divides two integers)."""
    out = []
    for seed in seeds:
        relative = f"dumb-extra/squadnl/{name}--squadnl--seed{seed}.json"
        test = read(root, relative).get("test") or {}
        questions = test.get("questions")
        if not isinstance(questions, int) or questions <= 0:
            raise decide.Fout(f"{root}/{relative}: no number of test questions ({questions!r})")
        out.append((decide.exact_count(test.get("exact_match"), questions, f"{root}/{relative} exact_match"), questions))
    if len({q for _, q in out}) != 1:
        raise decide.Fout(f"{root}/dumb-extra/squadnl/{name}: the seeds count different questions")
    return [k for k, _ in out], out[0][1]


def load_blimp_hf(read, root, name):
    relative = f"blimp-nl/{name}.json"
    result = read(root, relative)
    decide.check_settings(result, BLIMP_HF_SETTINGS, f"{root}/{relative}")
    return decide.exact_count(result.get("accuracy"), decide.BLIMP_PAIRS, f"{root}/{relative} accuracy")


def load_all(read, root):
    """{column: {row: (mean, range, values-as-strings)}}; a row a model has no number for is missing."""
    robbert = f"{root}/robbert"
    table = {column: {} for column in COLUMNS}

    def counts(ks, n):
        mean, spread = decide.stat_counts(ks, n)
        return {"mean": mean, "range": spread, "values": [f"{k}/{n}" for k in ks]}

    def extra(entry):
        kind, values, n = entry
        mean, spread = decide.extra_stat(entry)
        return {"mean": mean, "range": spread, "values": [f"{v}/{n}" if kind == "words" else str(v) for v in values],
                "kind": kind, "n": n}

    sources = {"robbert-2023": (robbert, "robbert-2023"), "robbert-2026": (robbert, "robbert-2026"),
               "T2": (root, FINAL_NAME["T2"]), "S": (root, FINAL_NAME["S"])}
    for column, (where, name) in sources.items():
        cells = table[column]
        if column.startswith("robbert"):
            for task in ("dbrd512", "sicknl"):
                cells[task] = counts(decide.load_primary(read, where, name, task, PRIMARY_512[task]), PRIMARY_512[task]["n"])
            cells["blimp"] = counts([load_blimp_hf(read, where, name)], decide.BLIMP_PAIRS)
        else:
            for task in ("dbrd1024", "sicknl"):
                cells[task] = counts(decide.load_primary(read, where, name, task, decide.PRIMARY[task]), decide.PRIMARY[task]["n"])
            cells["blimp"] = counts([decide.load_blimp(read, where, name, decide.ARMS[column]["checkpoint"])], decide.BLIMP_PAIRS)
        for task, spec in decide.EXTRA.items():
            cells[task] = extra(decide.load_extra(read, where, name, task, spec))
        ks, questions = squad_em(read, where, name)
        cells["squadnl-em"] = counts(ks, questions)
    table["T2"]["dbrd512"] = counts(decide.load_primary(read, robbert, FINAL_NAME["T2"], "dbrd512", PRIMARY_512["dbrd512"]),
                                    PRIMARY_512["dbrd512"]["n"])
    return table


def load_diagnostic(read, root):
    """SQuAD-NL seed 42 of RobBERT-2023 with the tokenizer's own (shifted) offsets, next to seed 42 with fixed ones."""
    robbert = f"{root}/robbert"
    out = {}
    for name in ("robbert-2023-rawoffsets", "robbert-2023"):
        relative = f"dumb-extra/squadnl/{name}--squadnl--seed42.json"
        result = read(robbert, relative)
        decide.check_settings(result, {**decide.EXTRA["squadnl"]["settings"], "seed": 42}, f"{robbert}/{relative}")
        test = result["test"]
        out[name] = {"f1": Fraction(test["f1"]), "exact_match": Fraction(decide.exact_count(test["exact_match"], test["questions"],
                     f"{robbert}/{relative} exact_match"), test["questions"]), "no_valid_span": test.get("no_valid_span")}
    return out


def compare(row, a, b):
    if row == "blimp":
        k_a, k_b = (int(cell["values"][0].split("/")[0]) for cell in (a, b))
        return decide.compare_blimp(k_a, k_b)
    c = decide.compare((a["mean"], a["range"]), (b["mean"], b["range"]))
    return {**c, "outcome": decide.OUTCOME[c["side"]]}


def nl(fraction, digits=2):
    return f"{float(100 * fraction):.{digits}f}".replace(".", ",")


WORD = {"beter": "beter", "slechter": "slechter", "gelijk": "geen verschil"}


def markdown(table, comparisons, diagnostic, args):
    stamp = datetime.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
    lines = [
        "# RobBERT-2023 en RobBERT-2026 door onze evaluatie, naast T2 en S",
        "",
        f"Stand {stamp}. Intern (bevat CoNLL-2002-getallen). Protocol: `experiments/robbert/protocol.md` op mbert-jax-branch",
        f"`robbert-compare` (runs met code `{args.code_commit}`; dit rapport met `report.py` van `{args.report_commit}`).",
        f"Uitslagen: `{args.root}/robbert/` (RobBERT, T2 DBRD@512) en `{args.root}/{{dumb,dumb-extra,blimp-nl}}/` (T2 en S).",
        "",
        "## Uitslagen [gemeten]",
        "",
        "Gemiddelde over de seeds 42, 43, 44, tussen haken de range (max − min), in procentpunten. BLiMP-NL is één getal",
        "per model (zero-shot). DBRD@1024 kan RoBERTa niet (514 geleerde posities); RobBERT-2026 is op 512 voorgetraind.",
        "S is op DBRD@512 niet gemeten.",
        "",
        "| taak | maat | " + " | ".join(LABEL[c] for c in COLUMNS) + " |",
        "|---|---|" + "---:|" * len(COLUMNS),
    ]
    for row, task, metric in ROWS:
        cells = []
        for column in COLUMNS:
            cell = table[column].get(row)
            if cell is None:
                cells.append("—")
            elif row == "blimp":
                cells.append(nl(cell["mean"]))
            else:
                cells.append(f"{nl(cell['mean'])} [{nl(cell['range'])}]")
        lines.append(f"| {task} | {metric} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Vergelijkingen [gemeten, met de drempelvorm van `decide.py`]",
        "",
        "Verschil van de gemiddelden (eerste min tweede) in procentpunten; drempel = de grootste seed-range van de twee",
        "(BLiMP-NL: 0,5 punt); \"beter\"/\"slechter\" alleen bij strikt meer dan de drempel.",
        "",
        "| taak | maat | " + " | ".join(f"{LABEL[a]} − {LABEL[b]}" for a, b in PAIRS) + " |",
        "|---|---|" + "---|" * len(PAIRS),
    ]
    for row, task, metric in ROWS:
        cells = []
        for a, b in PAIRS:
            c = comparisons.get(f"{a}|{b}", {}).get(row)
            cells.append("—" if c is None else f"{'+' if c['difference'] >= 0 else '−'}{nl(abs(c['difference']))} "
                                               f"(drempel {nl(c['threshold'])}): {WORD[c['outcome']]}")
        lines.append(f"| {task} | {metric} | " + " | ".join(cells) + " |")
    raw, fixed = diagnostic["robbert-2023-rawoffsets"], diagnostic["robbert-2023"]
    lines += [
        "",
        "## Diagnose: de offsets van de tokenizer van RobBERT-2023 [gemeten]",
        "",
        "De fast tokenizer van `DTAI-KULeuven/robbert-2023-dutch-base` geeft elk woord een begin-offset één teken te laat",
        "(\"De hoofdstad is Amsterdam\": `ĠDe` (1, 2) = \"e\", `ĠAmsterdam` (17, 25) = \"msterdam\"). SQuAD-NL haalt de",
        "antwoordposities voor de training en de antwoordtekst van de voorspelling uit die offsets. Onze evaluatie zet",
        "`trim_offsets` van de post-processor uit (dezelfde ids, offsets vanaf het woord). Eén run met de offsets zoals de",
        "tokenizer ze geeft, naast dezelfde seed met gecorrigeerde offsets:",
        "",
        "| RobBERT-2023, SQuAD-NL v1.1, seed 42 | F1 | EM | vragen zonder geldige span |",
        "|---|---:|---:|---:|",
        f"| offsets van de tokenizer | {nl(raw['f1'])} | {nl(raw['exact_match'])} | {raw['no_valid_span']} |",
        f"| gecorrigeerde offsets (de tabel hierboven) | {nl(fixed['f1'])} | {nl(fixed['exact_match'])} | {fixed['no_valid_span']} |",
        "",
        "## Naast de opgave van de RobBERT-makers",
        "",
        "Opgave [citaat, `pdelobelle/robbert-2026`, README.md op `a406b1a3`, sectie \"Results\"]: \"Evaluated on DUMB (test",
        "split; fine-tuned for 2 epochs at lr 5e-5, bf16, identical recipe for both models)\". Onze getallen uit de tabel",
        "hierboven [gemeten]:",
        "",
        "| taak | maat | RobBERT-2023 opgave | RobBERT-2023 ons | RobBERT-2026 opgave | RobBERT-2026 ons |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row, task, metric in [("squadnl", "SQuAD-NL", "F1"), ("squadnl-em", "SQuAD-NL", "EM"), ("dbrd512", "DBRD", "acc"),
                              ("sicknl", "SICK-NL", "acc")]:
        theirs = MAKERS[row]
        ours = [nl(table[c][row]["mean"]) for c in ("robbert-2023", "robbert-2026")]
        lines.append(f"| {task} | {metric} | {str(theirs[0]).replace('.', ',')} | {ours[0]} | {str(theirs[1]).replace('.', ',')} | {ours[1]} |")
    lines += [
        "",
        "Verschillen in de opzet [vastgesteld uit de opgave en onze protocollen]:",
        "",
        "- SQuAD-NL: DUMB's `squadnl` is SQuAD-NL v2.0, met onbeantwoordbare vragen (`experiments/dumb-extra/protocol.md`);",
        "  wij gebruiken v1.1 (alleen beantwoordbare vragen, `GroNLP/squad-nl-v1.1`). De getallen zijn dus niet dezelfde maat.",
        "- Fine-tunen: zij 2 epochs in bf16; wij 3 epochs in float32, met updates van 192 (DBRD), 256 (SICK-NL) en 32",
        "  (SQuAD-NL), constante LR voor DBRD en SICK-NL en lineair naar 0 voor SQuAD-NL; gemiddelde over 3 seeds.",
        "- DBRD: wij kappen af tot 512 tokens van links (het einde van de recensie blijft) en tellen 2.112 van de 2.224",
        "  testrecensies (hele updates); SICK-NL 4.864 van de 4.906 testparen.",
        "- RobBERT-2023 EM: de diagnose hierboven meet wat de verschoven offsets kosten [gemeten, één seed]. Of hun 46,7 EM",
        "  daardoor komt, is niet vast te stellen zonder hun run; de README zelf noemt een offset-bug in EuroEval's",
        "  `squad-nl`-pijplijn [citaat: \"EuroEval's `squad-nl` pipeline has an offset bug\"], terwijl de verschuiving",
        "  hier al in de offsets van de tokenizer zelf zit [gemeten op de tokenizer, eigen gevolgtrekking over de oorzaak].",
    ]
    return "\n".join(lines) + "\n"


def jsonable(value):
    if isinstance(value, Fraction):
        return {"exact": str(value), "pct": float(100 * value)}
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [jsonable(v) for v in value]
    return value


def main(argv=None, read=decide.read_json):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=decide.ROOT)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--code-commit", default="?")
    parser.add_argument("--report-commit", default="?")
    args = parser.parse_args(argv)
    try:
        table = load_all(read, args.root)
        diagnostic = load_diagnostic(read, args.root)
    except decide.Fout as error:
        print(f"FOUT: {error}")
        return 1
    comparisons = {}
    for a, b in PAIRS:
        comparisons[f"{a}|{b}"] = {row: compare(row, table[a][row], table[b][row]) for row, _, _ in ROWS
                                   if row in table[a] and row in table[b]}
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "robbert-compare.md").write_text(markdown(table, comparisons, diagnostic, args))
    (out / "robbert-compare.json").write_text(json.dumps(jsonable({
        "root": args.root, "code_commit": args.code_commit, "report_commit": args.report_commit, "columns": COLUMNS,
        "table": table, "comparisons": comparisons, "diagnostic": diagnostic, "makers": MAKERS}), indent=2) + "\n")
    for row, task, metric in ROWS:
        print(f"{task} {metric}: " + "  ".join(
            f"{LABEL[c]} {nl(table[c][row]['mean'])} [{nl(table[c][row]['range'])}]" for c in COLUMNS if row in table[c]))
    print(f"wrote {out}/robbert-compare.md and robbert-compare.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
