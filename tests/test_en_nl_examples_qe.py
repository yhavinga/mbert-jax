"""experiments/en-nl-translator/examples_qe.py: the pairs for the scorers and the examples of rule (a) from the tag test's
tables, on small tables. cutoff runs sft_build.py's query with a real tokenizer and is run on a VM."""

import json
import statistics
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

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
