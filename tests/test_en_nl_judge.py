"""experiments/en-nl-judge without a model: the prompt, the answer tokens, batching and padding, the score of a shard
with a made-up judge, the MoTra pairing on made-up sittings, the deterministic errors of the sanity set, and the AUC.
No jax: run with --noconftest where the conftest's imports (jax, flax) are not installed."""

import math
import random
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "en-nl-judge"))

import data  # noqa: E402
import judge  # noqa: E402
import report  # noqa: E402

# ----------------------------------------------------------------------------------------------------------------- judge


def test_prompt_is_template_around_trimmed_user_text():
    text = judge.prompt_text("  The House rose.\n", "\nDe vergadering wordt gesloten. ")
    assert text.startswith("<bos><|turn>user\nHere is an English text")
    assert text.endswith("Answer with one word: yes or no.<turn|>\n<|turn>model\n<|channel>thought\n<channel|>")
    assert "English:\nThe House rose.\n\nDutch:\nDe vergadering wordt gesloten.\n\n" in text
    user = judge.user_text("a", "b")
    assert user == user.strip(), "the template trims the user text, so the prompt must not depend on that"


def test_answer_ids_take_every_spelling():
    vocab = ["<pad>", "yes", "▁Yes", "YES", "no", "▁no", "No", "yesterday", "none", "▁", "nos"]
    yes, no = judge.answer_ids(vocab)
    assert yes == [1, 2, 3]
    assert no == [4, 5, 6]
    with pytest.raises(RuntimeError):
        judge.answer_ids(["<pad>", "yes"])


def test_plan_batches_sorts_buckets_and_caps_rows():
    lengths = [300, 100, 5000, 130, 120, 190, 700]
    batches, too_long = judge.plan_batches(lengths, token_budget=512, multiple=1, lengths=(128, 192, 384, 768))
    assert too_long == [2]
    assert batches == [(128, [1, 4]), (192, [3, 5]), (384, [0]), (768, [6])]
    batches, _ = judge.plan_batches([10] * 5, token_budget=256, multiple=1, lengths=(128,))
    assert [len(rows) for _, rows in batches] == [2, 2, 1]


def test_batch_rows_is_a_multiple_and_at_least_one_multiple():
    assert judge.batch_rows(192, 32768, 8) == 168
    assert judge.batch_rows(4096, 16384, 8) == 8
    assert judge.batch_rows(128, 32768, 8) == 256


def test_pack_left_pads_with_segment_zero():
    tokens, positions, segments = judge.pack([[5, 6, 7], [9]], length=4, rows=3)
    assert tokens.tolist() == [[0, 5, 6, 7], [0, 0, 0, 9], [0, 0, 0, 0]]
    assert positions.tolist() == [[0, 0, 1, 2], [0, 0, 0, 0], [0, 0, 0, 0]]
    assert segments.tolist() == [[0, 1, 1, 1], [0, 0, 0, 1], [0, 0, 0, 0]]
    with pytest.raises(ValueError):
        judge.pack([[1] * 5], length=4, rows=1)


def test_score_from_logps():
    assert judge.score_from_logps(math.log(0.6), math.log(0.2)) == pytest.approx(0.75)
    assert judge.score_from_logps(0.0, -1000.0) == 1.0
    assert judge.score_from_logps(-1000.0, 0.0) == 0.0


class FakeTokenizer:
    vocab = ["<pad>", "yes", "no", "x"]

    def encode_batch(self, texts):
        # one token per word, so the prompt's length follows the pair's
        return [[3] * len(t.split()) for t in texts]

    def token(self, i):
        return self.vocab[i]


class FakeJudge:
    """P(yes) depends on the prompt's length, so a mix-up of rows shows."""

    tokenizer = FakeTokenizer()

    def __init__(self):
        self.calls = []

    def run(self, prompts, length, rows):
        self.calls.append((length, rows, len(prompts)))
        p = np.array([1.0 / (1 + len(p) % 7) for p in prompts])
        top = np.tile(np.array([[1, 2, 3]]), (len(prompts), 1))
        return np.log(p) - 0.1, np.log(1 - p) - 0.1, top, np.zeros((len(prompts), 3))


def test_score_rows_keeps_row_order_and_skips():
    en = ["one two", "", "a b c d e f g h", "word " * 5000, "x y z"]
    nl = ["een twee", "leeg", "a b c d e f g h", "woord", "x y z"]
    fake = FakeJudge()
    cols = judge.score_rows(fake, en, nl, token_budget=1024, multiple=8)
    assert cols["skipped"] == [None, "empty", None, "too_long", None]
    for i in (0, 2, 4):
        n = cols["prompt_tokens"][i]
        p = 1.0 / (1 + n % 7)
        assert cols["score"][i] == pytest.approx(p)
        assert cols["answer_mass"][i] == pytest.approx(math.exp(-0.1))
        assert cols["top_token"][i] == "yes"
    assert math.isnan(cols["score"][1]) and math.isnan(cols["score"][3])
    assert all(rows % 8 == 0 for _, rows, _ in fake.calls)


# ------------------------------------------------------------------------------------------------------------------ data

EN_SITTING = """<?xml version='1.0' encoding='utf-8'?>
<text id="20050309.EN" lang="en" date="2005-03-09">
  <section id="s1">
    <intervention id="1-001" speaker_id="7" name="Ann SMITH" nationality="Ireland" mode="spoken">
      <p sl="en">First paragraph.</p>
      <p sl="de">Translated from German.</p>
      <p sl="en">Second <i>paragraph</i> here.</p>
    </intervention>
    <intervention id="1-002" speaker_id="8" name="Bo LIND" mode="spoken">
      <p sl="en">Has no Dutch.</p>
    </intervention>
    <intervention id="1-003" name="Twice"><p sl="en">A.</p></intervention>
    <intervention id="1-003" name="Twice"><p sl="en">B.</p></intervention>
  </section>
</text>
"""

NL_SITTING = """<?xml version='1.0' encoding='utf-8'?>
<text id="20050309.NL" lang="nl" date="2005-03-09">
  <section id="s1">
    <intervention id="1-001" speaker_id="7" name="Ann SMITH" nationality="Ireland" mode="spoken">
      <p sl="en">Eerste alinea.</p>
      <p sl="de">Uit het Duits.</p>
      <p sl="en">Tweede alinea   hier.</p>
    </intervention>
    <intervention id="1-002" speaker_id="8" name="Bo LIND" mode="spoken">
      <p sl="en">Een.</p>
      <p sl="en">Twee: het aantal klopt niet.</p>
    </intervention>
    <intervention id="1-003" name="Twice"><p sl="en">A.</p></intervention>
  </section>
</text>
"""


def test_pair_sitting_matches_on_intervention_index_and_count(tmp_path):
    en, nl = tmp_path / "20050309.EN.xml", tmp_path / "20050309.NL.xml"
    en.write_text(EN_SITTING)
    nl.write_text(NL_SITTING)
    date, rows, counts = data.pair_sitting((en, nl))
    assert date == "2005-03-09"
    assert [(r["id"], r["en"], r["nl"]) for r in rows] == [
        ("20050309:1-001:0:2", "First paragraph.", "Eerste alinea."),
        ("20050309:1-001:1:2", "Second paragraph here.", "Tweede alinea hier."),
    ]
    r = rows[0]
    assert (r["year"], r["speaker"], r["nationality"], r["sl"], r["en_words"], r["nl_words"]) == (
        2005, "Ann SMITH", "Ireland", "en", 2, 2)
    # 1-002 has one English paragraph against two in the Dutch file: the count differs, no pair
    assert counts["no_nl_paragraph"] == 1
    assert counts["en_ambiguous"] == 2 and counts["en_paragraphs"] == 5
    assert counts["paired"] == 2 and counts["with_child_elements"] == 1
    _, rows, counts = data.pair_sitting((en, None))
    assert rows == [] and counts["no_nl_file"] == 3


def test_sentences_keep_closing_quotes():
    text = 'He said "no." Then he left! Is it 5 p.m. now? yes. Mr. Smith agreed.'
    assert data.sentences(text) == ['He said "no."', "Then he left!", "Is it 5 p.m. now? yes.", "Mr.",
                                    "Smith agreed."]


def test_drop_sentence_is_deterministic_and_needs_two():
    nl = "Eén. Twee. Drie."
    a = data.drop_sentence(nl, random.Random("s"))
    assert a == data.drop_sentence(nl, random.Random("s"))
    assert a in ("Twee. Drie.", "Eén. Drie.", "Eén. Twee.")
    assert data.drop_sentence("Maar één zin.", random.Random(0)) is None


def test_change_number_only_a_number_of_the_english():
    assert data.change_number("In 1999 we had 3,000 cases.", "In 1999 hadden wij 3.000 gevallen.") == \
        "In 1990 hadden wij 3.000 gevallen."
    assert data.change_number("We had 3,000 cases.", "Wij hadden 3.000 gevallen in 2004.") == \
        "Wij hadden 3.001 gevallen in 2004."
    assert data.change_number("No numbers here.", "Hier 12 stuks.") is None


def test_remove_negation_only_with_one_simple_negation():
    assert data.remove_negation("Dat is niet waar.") == "Dat is waar."
    assert data.remove_negation("Er is geen probleem.") == "Er is een probleem."
    assert data.remove_negation("Geen enkel land stemde.") == "Een enkel land stemde."
    assert data.remove_negation("Dat is niet waar en geen grap.") is None
    assert data.remove_negation("Niet iedereen kwam.") is None
    assert data.remove_negation("Hij kwam niet, nooit.") is None
    assert data.remove_negation("Hij zegt: niet.") is None
    assert data.remove_negation("Nietsnut en geenszins.") is None


def test_corruptions_apply_where_they_can():
    rows = [{"id": "a", "en": "We have 12 votes. It is not over.", "nl": "Wij hebben 12 stemmen. Het is niet voorbij."},
            {"id": "b", "en": "Thank you.", "nl": "Dank u."},
            {"id": "c", "en": "Fine. Good.", "nl": "Goed. Prima."}]
    out = data.corruptions(rows, seed=0)
    kinds = {(r["source_id"], r["kind"]) for r in out}
    assert ("a", "number_changed") in kinds and ("a", "negation_removed") in kinds
    assert ("b", "sentence_dropped") not in kinds and ("c", "sentence_dropped") in kinds
    assert ("b", "number_changed") not in kinds
    mismatched = {r["source_id"]: r["nl"] for r in out if r["kind"] == "mismatch"}
    assert set(mismatched) == {"a", "b", "c"}
    assert all(mismatched[r["id"]] != r["nl"] for r in rows), "a derangement: no row keeps its own Dutch"
    assert {r["nl"] for r in out if r["kind"] == "english_kept"} == {r["en"] for r in rows}
    assert out == data.corruptions(rows, seed=0)


def test_draw_strata_disjoint_and_seeded():
    rows = [{"id": str(i), "en": f"It is {i}. Not so.", "nl": f"Het is {i}. Niet zo." if i % 2 else f"Het is {i}.",
             "en_words": 5, "nl_words": 4} for i in range(40)]
    rows += [{"id": f"n{i}", "en": "It is not.", "nl": "Het is niet zo.", "en_words": 3, "nl_words": 4}
             for i in range(10)]
    chosen = data.draw_strata(rows, seed=1, per_stratum=5)
    assert len({r["id"] for r in chosen}) == 20
    assert chosen == data.draw_strata(rows, seed=1, per_stratum=5)
    with pytest.raises(SystemExit):
        data.draw_strata(rows, seed=1, per_stratum=30)


# ---------------------------------------------------------------------------------------------------------------- report


def test_auc():
    assert report.auc([0.9, 0.8], [0.1, 0.2]) == 1.0
    assert report.auc([0.1], [0.9]) == 0.0
    assert report.auc([0.5, 0.5], [0.5]) == 0.5
    assert report.auc([0.9, 0.4], [0.5, 0.1]) == 0.75


def test_score_class_edges():
    assert [report.score_class(s) for s in (0.0, 0.0999, 0.1, 0.95, 1.0)] == [0, 0, 1, 9, 9]


def test_sanity_summary_pairs_with_own_originals():
    rows = []
    for i in range(10):
        base = {"skipped": None, "answer_mass": 0.99, "top_token": "yes"}
        rows.append({**base, "kind": "original", "source_id": str(i), "score": 0.85 - i * 0.005})
        rows.append({**base, "kind": "mismatch", "source_id": str(i), "score": 0.05})
    rows.append({"kind": "english_kept", "source_id": "0", "skipped": "too_long", "score": float("nan"),
                 "answer_mass": float("nan"), "top_token": None})
    summary = report.sanity_summary(rows)
    assert summary["kinds"]["mismatch"]["auc_all_originals"] == 1.0
    assert summary["kinds"]["mismatch"]["auc_same_paragraphs"] == 1.0
    assert summary["kinds"]["original"]["histogram"][8] == 10
    assert summary["skipped"] == {"too_long": 1}
    assert "| mismatch | 10 |" in report.sanity_markdown(summary)


def test_scan_markdown_samples_per_class():
    rows = [{"id": f"2001:{i}", "score": i / 100, "skipped": None, "answer_mass": 1.0, "top_token": "yes",
             "year": 2001 + i % 12, "date": "2001-01-01", "speaker": "S", "nationality": None, "en_words": 30,
             "nl_words": 31, "en": "E", "nl": "N"} for i in range(100)]
    md = report.scan_markdown(rows, {"title": "T", "intro": ["x"]}, seed=0, per_class=3)
    assert md.count("**0.") == 30
    assert "| 0.9–1.0 | 10 | 0.100 |" in md


def test_europarl_meta_has_source_counts_and_prompt():
    manifest = {"repo_commit": "abc", "sittings_en": 798, "counts": {"en_paragraphs": 10, "paired": 8,
                                                                     "no_nl_paragraph": 2},
                "by_period": {"1999-2003": {"paired": 3}, "2004-2012": {"paired": 5}}}
    meta = report.europarl_meta(manifest, "| sanity |", None)
    rows = [{"id": "x", "score": 0.5, "skipped": None, "answer_mass": 1.0, "top_token": "yes", "year": 2001,
             "date": "2001-01-01", "speaker": "S", "nationality": None, "en_words": 3, "nl_words": 3, "en": "E",
             "nl": "N"}]
    md = report.scan_markdown(rows, meta, seed=0, per_class=1)
    assert "commit `abc`" in md and "gekoppeld 8" in md and "| sanity |" in md
    assert judge.QUESTION in md and "<channel|>" in md


def test_spearman_with_ties_and_nan():
    assert report.ranks([3, 1, 2, 2]).tolist() == [4.0, 1.0, 2.5, 2.5]
    rho, n = report.spearman([1, 2, 3, 4, float("nan")], [10, 20, 30, 40, 50])
    assert rho == pytest.approx(1.0) and n == 4
    rho, _ = report.spearman([1, 2, 3], [3, 2, 1])
    assert rho == pytest.approx(-1.0)


def test_pilot_markdown_keeps_the_order_of_the_30():
    rows = [{"id": f"p{i}", "score": i / 40, "skipped": None, "chrf": float(i), "qe_en_nl": 40.0 - i,
             "qe_nl_en": 1.0, "source": "raad", "en": "E", "nl": "N"} for i in range(40)]
    ids30 = [f"p{i}" for i in reversed(range(30))]
    md = report.pilot_markdown(rows, ids30)
    assert md.index("`p29`") < md.index("`p0`")
    assert "| chrF (verwacht positief) | 1.000 | 40 |" in md
    assert "| MetricX-QE en→nl (lager is beter, dus verwacht negatief) | -1.000 | 40 |" in md
    with pytest.raises(SystemExit):
        report.pilot_markdown(rows, ["nope"] * 30)
