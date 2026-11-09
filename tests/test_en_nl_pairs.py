"""experiments/en-nl-translator/pairs.py and the parts of metricx_qe.py that need no model: the mirror of english_left,
numbers, sentence counts, the measures of a pair, MetricX's input. The forward pass of metricx_qe.py is checked against
transformers on a VM (`metricx_qe.py check`). Run with --noconftest where the conftest's imports are not installed."""

import json
import math
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "en-nl-translator"))

import pairs  # noqa: E402

publish = pairs.publish


@pytest.mark.parametrize("nl, en", [
    ("Hij zei dat the quick brown fox jumps over the lazy dog and runs into the forest with them.",
     "He said that the quick brown fox jumps over the lazy dog and runs into the forest with them."),
    ("Het boek The Lord of the Rings is mooi.", "The book The Lord of the Rings is beautiful."),
    ("Dit is helemaal vertaald naar het Nederlands.", "This is completely translated into Dutch."),
])
def test_source_left_with_english_as_source_is_english_left(nl, en):
    assert pairs.source_left(nl, en, publish.ENGLISH_WORDS, publish.DUTCH_WORDS) == publish.english_left(nl, en)


def test_dutch_left_finds_a_stretch_of_untranslated_dutch():
    nl = "De minister zegt dat het kabinet de knoop heeft doorgehakt over de bouw van woningen in de steden."
    left = "The minister says that het kabinet de knoop heeft doorgehakt over de bouw van woningen in de steden."
    assert pairs.dutch_left(left, nl)
    assert not pairs.dutch_left("The minister says the cabinet has made a decision on building homes in cities.", nl)
    assert not pairs.dutch_left("Mr Van der Ploeg of the VVD visited Den Haag and Zwolle.", "De heer Van der Ploeg "
                                "van de VVD bezocht Den Haag en Zwolle.")


def test_numbers_ignore_separators_and_decimal_commas():
    assert pairs.numbers("Er zijn 1.000.000 woningen, 12,5% meer dan in 2023 (art. 7).") == \
        pairs.numbers("There are 1,000,000 homes, 12.5% more than in 2023 (Art. 7).")
    assert pairs.numbers("Op 3 mei") - pairs.numbers("On May 4") == {"3"}


def test_numbers_the_english_writes_otherwise_are_not_missing():
    assert pairs.missing_numbers("Een plantekst en 2 plankaarten.", "A plan text and two plan maps.") == []
    assert pairs.missing_numbers("Om 01.30 uur vannacht.", "At 1:30 last night.") == []
    assert pairs.missing_numbers("Den Haag, 30-04-2015.", "The Hague, April 30, 2015.") == []
    assert pairs.missing_numbers("Den Haag, 30-04-2015.", "The Hague, 30 2015.") == ["4"]
    assert pairs.missing_numbers("Op de Janneboulevard 78 en in 2023.", "On Janneboulevard in 2023.") == ["78"]
    assert pairs.missing_numbers("Er zijn 1.000.000 woningen, 12,5% meer.", "There are 1,000,000 homes, 12.5% more.") == []


def test_sentences_know_dutch_references():
    text = "Het proces-verbaal d.d. 7 december is opgemaakt. Zie de stukken (pag. 6) en het verslag."
    assert pairs.sentences(text, pairs.sp.ABBREVIATIONS) == 2


def test_sentences_know_english_abbreviations():
    assert pairs.sentences("Mrs. Smith spoke. Mr. Jones, e.g. here, did not.", pairs.ENGLISH_ABBREVIATIONS) == 2


def test_measures_of_a_pair():
    nl = "Het kabinet heeft in 2023 besloten. Dat was laat."
    en = "The cabinet decided. That was late."
    row = pairs.measure(nl, en, "Het kabinet besloot. Dat was laat.")
    assert row["numbers_missing"] == ["2023"]
    assert row["sentences_nl"] == row["sentences_en"] == 2
    assert row["len_ratio"] == pytest.approx(len(nl) / len(en), abs=1e-3)
    assert 0 < row["chrf"] < 100 and row["decode_loop"] is False and row["dutch_left"] is False
    assert pairs.measure(nl, en, None)["chrf"] is None


def test_a_looping_translation():
    nl = "Wij moeten nu doorpakken."
    en = "We must now act, act, act, act, act, act, act, act and act again."
    assert pairs.measure(nl, en, None)["decode_loop"] is True


def test_metricx_input_and_eos():
    sys.path.insert(0, str(ROOT / "experiments" / "en-nl-translator"))
    import metricx_qe as mx

    assert mx.make_input("A house.", "Een huis.") == "source: A house. candidate: Een huis."

    class Tok:
        def __call__(self, texts, max_length, truncation):
            assert max_length == mx.MAX_INPUT and truncation
            return {"input_ids": [[len(t), 7, 1] for t in texts]}

    assert mx.encode(Tok(), ["ab", "abc"]) == [[2, 7], [3, 7]]


def reference_bucket(rel, bidirectional, num_buckets=32, max_distance=128):
    """transformers' T5Attention._relative_position_bucket for one int"""
    out = 0
    if bidirectional:
        num_buckets //= 2
        out += (rel > 0) * num_buckets
        n = abs(rel)
    else:
        n = -min(rel, 0)
    exact = num_buckets // 2
    if n < exact:
        return out + n
    large = exact + int(math.log(n / exact) / math.log(max_distance / exact) * (num_buckets - exact))
    return out + min(large, num_buckets - 1)


def test_relative_buckets_match_t5():
    jnp = pytest.importorskip("jax.numpy")
    import metricx_qe as mx

    rel = jnp.arange(-300, 301, dtype=jnp.int32)
    for bidirectional in (True, False):
        got = [int(x) for x in mx.relative_buckets(rel, bidirectional, 32, 128)]
        assert got == [reference_bucket(int(r), bidirectional) for r in range(-300, 301)]


def test_measures_record_the_structure_of_both_directions():
    row = pairs.measure("Een zin.\nNog een zin.", "A sentence.\n\nAnother sentence.", "Een zin.\nNog een zin.")
    # nl -> en doubled the newline; en -> nl went back to one, which is a difference of its own
    assert row["structure"] == ["newline_doubled"] and row["back_structure"] == ["paragraph_break", "line_break"]
    row = pairs.measure("Een **vette** zin.", "A **bold** sentence.", None)
    assert row["structure"] == [] and row["back_structure"] is None


# ---------------------------------------------------------------- the main run: context turns, hard checks, summary

def test_hard_checks_of_a_pair():
    nl = "Het kabinet besloot gisteren.\n\nDat was laat."
    row = {"ended": True, **pairs.measure(nl, "The cabinet decided yesterday.\n\nThat was late.", None)}
    assert pairs.hard_fails(row) == []
    row = {"ended": False, **pairs.measure(nl, "The cabinet decided yesterday. " * 8, None)}
    assert pairs.hard_fails(row) == ["not_ended", "decode_loop", "len_ratio", "structure"]
    # the known empty line for a single newline is counted apart from other structure
    row = {"ended": True, **pairs.measure("Een zin hier.\nNog een zin.", "A sentence here.\n\nOne more sentence.", None)}
    assert pairs.hard_fails(row) == ["newline_doubled"]


def make_run(tmp_path):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    sel = tmp_path / "select"
    sel.mkdir()
    passages = [{"id": f"raad:{k}:d:0", "pid": f"raad:{k}:d:0", "source": "raad", "split": "train", "half": "random",
                 "idiom_score": 0.0, "text": f"De raad besloot punt {k}.\n\nDaarna was het klaar.", "tokens": 12}
                for k in range(3)]
    turns = [{"id": "handelingen:9:t3", "doc_id": "9", "turn": 3, "speaker": "De voorzitter", "party": None,
              "text": "Ik geef het woord aan de minister.", "text_pool": "Ik geef het woord aan de minister.",
              "tokens": 8, "source_score": 1.0, "used_by": 1}]
    pq.write_table(pa.table({c: [p[c] for p in passages] for c in passages[0]}), sel / "passages.parquet")
    pq.write_table(pa.table({c: [t[c] for t in turns] for c in turns[0]}), sel / "context-turns.parquet")
    ids = [p["id"] for p in passages] + [turns[0]["id"]]
    en = [f"The council decided point {k}.\n\nThen it was done." for k in range(3)] + ["I give the floor to the minister."]
    back = [f"De raad besliste punt {k}.\n\nToen was het klaar." for k in range(3)] + ["Ik geef het woord aan de minister."]
    for name, column, texts in (("nl-en", "text_en", en), ("en-nl", "text_nl", back)):
        d = tmp_path / name / "w0" / "small"
        d.mkdir(parents=True)
        pq.write_table(pa.table({"id": ids, column: texts, "ended": [True] * 4, "tokens_in": [80] * 4,
                                 "tokens_out": [10] * 4}), d / "shard-00000.parquet")
    return sel, pa, pq


def test_measures_cover_the_context_turns(tmp_path):
    sel, pa, pq = make_run(tmp_path)
    rows = pairs.read_passages(sel)
    assert [r["kind"] for r in rows] == ["passage"] * 3 + ["context"]
    out = tmp_path / "pairs.parquet"
    pairs.main(["measures", "--select", str(sel), "--nl-en", str(tmp_path / "nl-en"), "--en-nl",
                str(tmp_path / "en-nl"), "--out", str(out), "--processes", "1"])
    got = pq.read_table(out).to_pylist()
    assert len(got) == 4 and got[-1]["kind"] == "context" and got[-1]["source"] == "handelingen"
    assert all(r["hard_fails"] == [] and r["structure"] == [] and r["chrf"] > 0 for r in got)
    pairs.main(["qe-split", "--pairs", str(out), "--workers", "2", "--out", str(tmp_path / "qe-in")])
    split = [pq.read_table(tmp_path / "qe-in" / f"qe-in-w{k}.parquet").to_pylist() for k in range(2)]
    assert sorted(r["id"] for part in split for r in part) == sorted(r["id"] for r in got)
    for k, part in enumerate(split):
        pq.write_table(pa.table({"id": [r["id"] for r in part], "qe_en_nl": [1.5] * len(part)}),
                       tmp_path / f"qe-w{k}.parquet")
    pairs.main(["summary", "--pairs", str(out), "--qe", str(tmp_path / "qe-w0.parquet"), str(tmp_path / "qe-w1.parquet"),
                "--out", str(tmp_path / "final")])
    final = pq.read_table(tmp_path / "final" / "pairs.parquet").to_pylist()
    assert all(r["qe_en_nl"] == 1.5 for r in final)
    summary = (tmp_path / "final" / "summary.md").read_text()
    assert "| all | 4 | 4 |" in summary


def test_the_summary_refuses_pairs_without_qe(tmp_path):
    sel, pa, pq = make_run(tmp_path)
    out = tmp_path / "pairs.parquet"
    pairs.main(["measures", "--select", str(sel), "--nl-en", str(tmp_path / "nl-en"), "--en-nl",
                str(tmp_path / "en-nl"), "--out", str(out)])
    pq.write_table(pa.table({"id": ["raad:0:d:0"], "qe_en_nl": [1.0]}), tmp_path / "qe.parquet")
    with pytest.raises(SystemExit, match="3 pairs without a QE row"):
        pairs.main(["summary", "--pairs", str(out), "--qe", str(tmp_path / "qe.parquet"), "--out", str(tmp_path / "f")])


def test_a_long_run_carries_its_length_class_and_caps_the_back_translation(tmp_path, monkeypatch):
    sel, pa, pq = make_run(tmp_path)
    (sel / "context-turns.parquet").unlink()
    rows = pq.read_table(sel / "passages.parquet").to_pylist()
    for r in rows:
        r["length_class"] = "long"
    pq.write_table(pa.table({c: [r[c] for r in rows] for c in rows[0]}), sel / "passages.parquet")
    out = tmp_path / "pairs.parquet"
    pairs.main(["measures", "--select", str(sel), "--nl-en", str(tmp_path / "nl-en"), "--en-nl",
                str(tmp_path / "en-nl"), "--out", str(out)])
    got = pq.read_table(out).to_pylist()
    assert [r["length_class"] for r in got] == ["long"] * 3
    pq.write_table(pa.table({"id": [r["id"] for r in got], "qe_en_nl": [2.0] * 3}), tmp_path / "qe.parquet")
    pairs.main(["summary", "--pairs", str(out), "--qe", str(tmp_path / "qe.parquet"), "--out", str(tmp_path / "f"),
                "--title", "Aanvullende run: passages van 385–511 tokens"])
    assert (tmp_path / "f" / "summary.md").read_text().startswith("# Aanvullende run: passages van 385–511 tokens\n")
    assert pq.read_table(tmp_path / "f" / "pairs.parquet").column("length_class").to_pylist() == ["long"] * 3

    class WordTokenizer:
        @classmethod
        def from_file(cls, path):
            return cls()

        def encode(self, text, add_special_tokens=False):
            return types.SimpleNamespace(ids=text.split())

    monkeypatch.setitem(sys.modules, "tokenizers", types.SimpleNamespace(Tokenizer=WordTokenizer))
    # "The council decided point k. Then it was done." has 9 words: above a cap of 8 every segment stays out
    pairs.main(["back-segments", "--select", str(sel), "--nl-en", str(tmp_path / "nl-en"), "--tokenizer", "x",
                "--out", str(tmp_path / "back"), "--max-tokens", "8"])
    report = json.loads((tmp_path / "back" / "back_segments.json").read_text())
    assert report["segments"] == 0 and len(report["skipped"]["too_long"]) == 3
    pairs.main(["back-segments", "--select", str(sel), "--nl-en", str(tmp_path / "nl-en"), "--tokenizer", "x",
                "--out", str(tmp_path / "back9"), "--max-tokens", "9"])
    assert json.loads((tmp_path / "back9" / "back_segments.json").read_text())["segments"] == 3
