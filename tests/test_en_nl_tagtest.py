"""experiments/en-nl-translator/tagtest.py: what the tag test counts per translation, and collect and report on small
tables (no model, no TPU; chrF is replaced where sacrebleu is not installed)."""

import argparse
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "en-nl-translator"))

import tagtest as tt  # noqa: E402

ITEM = {"source": "The House agrees with the motion. What does the minister think?"}


@pytest.fixture(autouse=True)
def plain_chrf(monkeypatch):
    # a stand-in for sacrebleu's chrF: 100 for the same text, 0 otherwise
    monkeypatch.setattr(tt, "chrf", lambda a, b: 100.0 if a == b else 0.0)


@pytest.mark.parametrize("text, tag", [
    ("<unused0>De Kamer stemt in.", True), ("De Kamer stemt in. <BT>", True),
    ("Vertaal in natuurlijk, oorspronkelijk klinkend Nederlands. De Kamer stemt in.", True),
    ("Translate into natural, original-sounding Dutch.", True), ("De Kamer stemt in met de motie.", False),
    ("Het klinkt natuurlijk goed.", False)])
def test_the_tag_in_the_output(text, tag):
    assert tt.flags(ITEM, text, True, None)["tag"] is tag


@pytest.mark.parametrize("text, intro", [
    ("Hieronder volgt de vertaling van de tekst:\n\nDe Kamer stemt in.", True), ("Hier is de vertaling: De Kamer.", True),
    ("Vertaling: De Kamer stemt in.", True), ("Natuurlijk! De Kamer stemt in.", True),
    ("De Kamer stemt in met de motie.", False), ("Natuurlijk stemt de Kamer in.", False)])
def test_an_opening_about_the_translation(text, intro):
    assert tt.flags(ITEM, text, True, None)["intro"] is intro


def test_an_opening_the_source_has_too_is_no_intro():
    assert not tt.flags({"source": "Here is the list: a, b."}, "Hier is de lijst: a, b.", True, None)["intro"]
    # the base model's own translation of "All right, whoa, ..." (WMT24++, 10 Nov)
    assert not tt.flags({"source": "All right, whoa, hold it down."}, "Oké, rustig, rustig.", True, None)["intro"]


def test_longer_and_shorter_than_the_base_model():
    base = "De Kamer stemt in met de motie. Wat vindt de minister?"
    f = tt.flags(ITEM, base + " De minister vindt dat de motie een goed idee is en steunt haar.", True, base)
    assert f["longer"] and not f["shorter"]
    assert tt.flags(ITEM, "Ja.", True, base)["shorter"]
    same = tt.flags(ITEM, base, True, base)
    assert not same["longer"] and not same["shorter"] and same["ratio_to_base"] == 1.0
    assert tt.flags(ITEM, "", False, base)["empty"] and tt.flags(ITEM, "", False, base)["not_ended"]


def test_collect_and_report(tmp_path):
    items = [{"id": "flan:0-1", "set": "flan", "task": "t", "source": "Is it red? Answer yes or no.", "reference": None},
             {"id": "w1", "set": "wmt24pp", "task": "news", "source": "The House agrees.", "reference": "De Kamer."}]
    pq.write_table(pa.Table.from_pylist(items), tmp_path / "items.parquet")
    outputs = {"flan:0-1": "Is het rood? Antwoord ja of nee.", "w1": "De Kamer stemt in."}
    for variant in tt.VARIANTS:
        d = tmp_path / "shards" / variant / "small"
        d.mkdir(parents=True)
        texts = dict(outputs)
        if variant == "token_tag":
            texts["w1"] = "<unused0>De Kamer stemt in."
        if variant == "instruction_tag":
            texts["flan:0-1"] = "Is het rood? Antwoord ja of nee.\n\nJa, het is rood, want de vraag zegt dat het zo is."
        pq.write_table(pa.table({"id": list(texts), "text_nl": list(texts.values()), "ended": [True, True],
                                 "tokens_in": [1, 1], "tokens_out": [1, 1]}), d / "shard-00000.parquet")
    tt.collect(argparse.Namespace(items=tmp_path / "items.parquet", shards=tmp_path / "shards",
                                  out=tmp_path / "pairs.parquet"))
    pairs = pq.read_table(tmp_path / "pairs.parquet").to_pylist()
    assert len(pairs) == 10 and set(pairs[0]) >= {"source", "hypothesis", "variant", "tag", "longer"}
    assert [p["tag"] for p in pairs if p["variant"] == "token_tag"] == [False, True]
    assert [p["longer"] for p in pairs if p["variant"] == "instruction_tag"] == [True, False]
    pq.write_table(pa.table({"id": [p["id"] for p in pairs], "qe_source_hypothesis": [1.0] * len(pairs)}),
                   tmp_path / "qe.parquet")
    args = argparse.Namespace(pairs=tmp_path / "pairs.parquet", qe=tmp_path / "qe.parquet",
                              qe_column="qe_source_hypothesis", out=tmp_path / "tag-test.md",
                              json=tmp_path / "t.json", flagged=tmp_path / "flagged.md", examples=2, max_chars=100,
                              title="T", intro="")
    tt.report(args)
    md = (tmp_path / "tag-test.md").read_text()
    assert "| token, met tag | wmt24pp | 1 | 1 |" in md and "### w1" in md
    assert "instruction_tag flan:0-1" in (tmp_path / "flagged.md").read_text()
    # a missing translation stops collect
    (tmp_path / "shards" / "base" / "small" / "shard-00000.parquet").unlink()
    with pytest.raises(SystemExit):
        tt.collect(argparse.Namespace(items=tmp_path / "items.parquet", shards=tmp_path / "shards",
                                      out=tmp_path / "p2.parquet"))
