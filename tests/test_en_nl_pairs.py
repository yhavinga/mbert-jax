"""experiments/en-nl-translator/pairs.py and the parts of metricx_qe.py that need no model: the mirror of english_left,
numbers, sentence counts, the measures of a pair, MetricX's input. The forward pass of metricx_qe.py is checked against
transformers on a VM (`metricx_qe.py check`). Run with --noconftest where the conftest's imports are not installed."""

import math
import sys
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
