"""experiments/en-nl-translator/examples_qe.py: the pairs for the scorers and the examples of rule (a) from the tag test's
tables, on small tables. cutoff runs sft_build.py's query with a real tokenizer and is run on a VM."""

import json
import statistics
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "en-nl-translator"))

import examples_qe as ex  # noqa: E402

DIFFS = [0.2, 0.15, 0.25, 0.3, 0.5, 0.45, 1.0, 0.9, 2.0, -0.5, -0.45, 0.0, 0.0, 3.5, -1.2] + [0.05] * 8  # 23 > 20


def write(path, rows):
    pq.write_table(pa.Table.from_pylist(rows), path)


def tables(tmp_path):
    ids = [f"wmt24pp-{k:04d}" for k in range(len(DIFFS))]
    items = [{"id": i, "set": "wmt24pp", "task": "news", "source": f"Source {i}.", "reference": f"Referentie {i}."}
             for i in ids] + [{"id": "flan:0-1", "set": "flan", "task": "t", "source": "Q?", "reference": None}]
    pairs, qe = [], []
    for variant in ("base", "token_tag", "instruction_tag"):
        for item in items:
            k = ids.index(item["id"]) if item["id"] in ids else 0
            score = 2.0 + (DIFFS[k] if variant == "instruction_tag" else 0.0)
            text = f"{variant} {item['id']}" if DIFFS[k] or variant != "instruction_tag" else f"base {item['id']}"
            pairs.append({"variant": variant, "id": item["id"], "set": item["set"], "task": item["task"],
                          "source": item["source"], "hypothesis": text})
            qe.append({"id": item["id"], "qe_source_hypothesis": score})
    write(tmp_path / "items.parquet", items)
    write(tmp_path / "pairs.parquet", pairs)
    write(tmp_path / "qe.parquet", qe)
    return ids


def test_rule_pairs_and_examples(tmp_path):
    ids = tables(tmp_path)
    ex.main(["rule-pairs", "--pairs", str(tmp_path / "pairs.parquet"), "--items", str(tmp_path / "items.parquet"),
             "--out", str(tmp_path / "rule-pairs.parquet")])
    rp = pq.read_table(tmp_path / "rule-pairs.parquet").to_pylist()
    assert len(rp) == 2 * len(ids) and {r["id"].split("|")[0] for r in rp} == {"base", "instruction_tag"}
    assert all(r["reference"].startswith("Referentie") for r in rp)
    write(tmp_path / "metricx.parquet", [{"id": r["id"], "ref_source_hypothesis_reference": 3.0 + 0.001 * k * k,
                                          "qe_source_hypothesis": 2.0} for k, r in enumerate(rp)])
    write(tmp_path / "comet.parquet", [{"id": r["id"], "comet22": 0.8 - 0.0001 * k * k} for k, r in enumerate(rp)])
    out = tmp_path / "rule.md"
    ex.main(["rule", "--pairs", str(tmp_path / "pairs.parquet"), "--qe", str(tmp_path / "qe.parquet"), "--items",
             str(tmp_path / "items.parquet"), "--scores", str(tmp_path / "metricx.parquet"), "--comet",
             str(tmp_path / "comet.parquet"), "--out", str(out), "--json", str(tmp_path / "rule.json")])
    got = json.loads((tmp_path / "rule.json").read_text())
    assert got["stats"]["n"] == len(ids)
    assert abs(got["stats"]["mean_diff"] - statistics.mean(DIFFS)) < 1e-9
    assert got["stats"]["same_text"] == 2 and got["stats"]["over"]["2"] == 1
    diffs = dict(zip(ids, DIFFS))
    for label, lo, hi in ex.RULE_CLASSES:
        chosen = got["chosen"][label]
        assert chosen["pool"] == sum(lo <= d <= hi for d in DIFFS)
        assert len(chosen["ids"]) == min(ex.RULE_PER_CLASS, chosen["pool"])
        assert all(lo <= diffs[i] <= hi for i in chosen["ids"])
    text = out.read_text()
    assert "## Het gemiddelde en de verdeling per segment" in text and "Referentie (WMT24++)" in text
    # the same seed, the same examples
    first = (tmp_path / "rule.json").read_text()
    ex.main(["rule", "--pairs", str(tmp_path / "pairs.parquet"), "--qe", str(tmp_path / "qe.parquet"), "--items",
             str(tmp_path / "items.parquet"), "--scores", str(tmp_path / "metricx.parquet"), "--comet",
             str(tmp_path / "comet.parquet"), "--out", str(out), "--json", str(tmp_path / "rule.json")])
    assert (tmp_path / "rule.json").read_text() == first


def test_spearman_and_ranks():
    assert ex.ranks([3, 1, 2, 2]) == [3, 0, 1.5, 1.5]
    assert abs(ex.spearman([1, 2, 3, 4], [10, 20, 30, 45]) - 1) < 1e-12
    assert abs(ex.spearman([1, 2, 3, 4], [4, 3, 2, 1]) + 1) < 1e-12


def test_numbers_the_dutch_way():
    assert ex.nl(1234.5, 1) == "1.234,5" and ex.nl(-0.213, 3) == "-0,213" and ex.nl(None) == "–"


def test_length_examples_and_what_each_factor_costs(tmp_path):
    """Arm R2's length bound: Dutch over English characters within a factor of the source's median"""
    (tmp_path / "run").mkdir()
    (tmp_path / "long").mkdir()
    rows = []
    for k, n in enumerate([100] * 20 + [70, 74, 140, 128, 300]):  # medians 100/100 = 1.0; 300 fails hard_fails
        rows.append({"id": f"h{k}", "pid": f"h{k}", "kind": "passage", "source": "handelingen", "split": "train",
                     "en": "e" * 100, "nl": "n" * n, "hard_fails": ["len_ratio"] if n == 300 else [], "qe_en_nl": 1.0})
    rows.append({"id": "q", "pid": "q", "kind": "passage", "source": "handelingen", "split": "train", "en": "e" * 100,
                 "nl": "n" * 60, "hard_fails": [], "qe_en_nl": 9.0})  # above the QE bound: not counted
    write(tmp_path / "run" / "pairs.parquet", rows)
    write(tmp_path / "long" / "pairs.parquet", [rows[0] | {"id": "l0", "pid": "l0"}])
    real = [{"id": f"europarl:{k}", "source": "europarl", "en": "e" * 100, "nl": "n" * n, "fails": [], "qe_en_nl": None}
            for k, n in enumerate([115] * 10 + [80, 160])]
    write(tmp_path / "real.parquet", real)
    out = tmp_path / "length.md"
    ex.main(["length", "--run", str(tmp_path / "run"), "--long", str(tmp_path / "long"), "--real",
             str(tmp_path / "real.parquet"), "--factor", "1.3333", "--out", str(out), "--json", str(tmp_path / "l.json")])
    o = json.loads((tmp_path / "l.json").read_text())
    bt = o["groups"]["bt"]
    assert bt["n"] == 25 and bt["sources"]["handelingen"]["median_chars"] == 1.0
    chosen = [c for c in bt["costs"] if c["factor"] == 1.3333][0]
    assert chosen == {"factor": 1.3333, "below": 2, "above": 1}  # 70 and 74 under 0.75; 140 over 1.3333; 128 stays
    assert [c["below"] + c["above"] for c in bt["costs"]] == [4, 4, 3, 0]  # 1.2, 1.25, 1.3333, 1.5 (0.667 to 1.5)
    ep = o["groups"]["real/europarl"]
    assert ep["sources"]["europarl"]["median_chars"] == 1.15 and ep["sources"]["europarl"]["below"] == 1
    md = out.read_text()
    assert "# Voorbeelden bij de lengtegrens van arm R2" in md and "## Europarl-direct en→nl" in md
    assert "1,3333 (gekozen)" in md


def test_real_edges_cost_per_author_and_examples_on_the_bounds(tmp_path):
    """Arm RB: the books' hard checks, QE bound and length bound in sft_build.py's order, per author, with examples"""
    rows = []
    for k in range(20):
        author = "Nescio" if k < 10 else "Harry Mulisch"
        qe = 4.5 if k == 3 else 5.5 if k == 4 else 12.0 if k == 5 else 1.0
        nl_chars = 60 if k == 6 else 100
        rows.append({"id": f"books:{k}", "source": "books", "en": "e" * 100, "nl": "n" * nl_chars, "fails": [],
                     "qe_en_nl": qe, "meta": json.dumps({"author": author})})
    rows.append({"id": "books:x", "source": "books", "en": "e" * 100, "nl": "n" * 100, "fails": ["not_dutch_original"],
                 "qe_en_nl": None, "meta": json.dumps({"author": "Paulo Coelho"})})
    write(tmp_path / "real-qe.parquet", rows)
    out = tmp_path / "edges.md"
    ex.main(["real-edges", "--real", str(tmp_path / "real-qe.parquet"), "--source", "books", "--qe-max", "5",
             "--factor", "1.3333", "--out", str(out), "--json", str(tmp_path / "e.json")])
    o = json.loads((tmp_path / "e.json").read_text())
    assert o["rows"] == 21 and o["hard_ok"] == 20 and o["bounds"]["median"] == 1.0
    assert o["total"] == {"rows": 21, "hard:not_dutch_original": 1, "qe": 2, "length": 1, "kept": 17}
    assert o["authors"]["Nescio"] == {"rows": 10, "qe": 2, "length": 1, "kept": 7}
    assert o["qe_costs"]["5"] == 2 and o["qe_costs"]["10"] == 1
    assert [x["id"] for x in o["examples"]["qe/outside"]] == ["books:4"]
    assert [x["id"] for x in o["examples"]["qe/highest"]][0] == "books:5"
    md = out.read_text()
    assert "# Grenzen op de echte paren van bron books (arm RB)" in md and "| Paulo Coelho | 1 | 1 (not_dutch_original 1) |" in md
    # a pair that passed the hard checks without a QE: an error
    rows[0]["qe_en_nl"] = None
    write(tmp_path / "real-qe.parquet", rows)
    with pytest.raises(SystemExit, match="no MetricX-QE"):
        ex.main(["real-edges", "--real", str(tmp_path / "real-qe.parquet"), "--source", "books", "--out", str(out)])
