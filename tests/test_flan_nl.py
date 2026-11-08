"""experiments/flan-nl on made-up FLAN rows and a made-up translator: options, the substring rule, filters, the Dutch mix
in place of F's FLAN rows, and shards that a restart skips. No model and no TPU: run with --noconftest where the
conftest's imports (jax, flax) are not installed."""

import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "flan-nl"))
sys.path.insert(0, str(ROOT / "scripts"))

import flan_nl as fn  # noqa: E402
import translate as tr  # noqa: E402


def words(text):
    return len(text.split())


def fake_dutch(text):
    """A translator that is easy to check: every segment becomes NL[<segment>]."""
    return f"NL[{text}]"


def translated(inputs, targets, translate=fake_dutch):
    instruction, answer, info = fn.plan_example(inputs, targets, words)
    table = {s: translate(s) for s in fn.segments_of(instruction) + fn.segments_of(answer)}
    return fn.render(instruction, table), fn.render(answer, table), info


@pytest.mark.parametrize("inputs, targets", [
    ("Premise: A man.\nHypothesis: A dog.\nOPTIONS:\n- yes\n- it is not possible to tell\n- no", "it is not possible to tell"),
    ("What is the best way to measure?\n\nOPTIONS: [+] Grab a teaspoon.; [+] Grab a tablespoon.;\nAnswer:", "Grab a teaspoon."),
    ("Tell me who it is.\nAvailable choices:\n +a Barbie.\n +the microwave.", "the microwave"),
    ("Available options: -Two.. -One.. -Four.. -Three..\nGiven those answer options, answer: Bob is six.", "Three."),
    ("pick from the following. -- Yes -- It's impossible to say -- No\nI think the answer is", "It's impossible to say"),
])
def test_an_answer_that_is_an_option_gets_the_options_translation(inputs, targets):
    instruction, answer, info = fn.plan_example(inputs, targets, words)
    assert info["answer"] == "option"
    nl_inputs, nl_targets, _ = translated(inputs, targets)
    option = fn._strip_end(targets)
    assert nl_targets == f"NL[{option}]" + targets[len(option):]
    # The Dutch answer is word for word one of the Dutch options
    assert f"NL[{option}]" in nl_inputs
    assert not fn.answer_lost(inputs, targets, nl_inputs, nl_targets)


@pytest.mark.parametrize("inputs, targets, labels", [
    ("What will happen to Lee?\n\nOPTIONS: (i). be frustrated (ii). be happy (iii). stand trial\nThe answer is:", "(ii).",
     ["(i).", "(ii).", "(iii)."]),
    ("Who is \"it\"?\nChoose your answer from:\n a). The cat\n b). the glass", "a).", ["a).", "b)."]),
    ("Select from the following. i. insulin; ii. lipids; iii. amino acids; iv. glucose;\n\nQ: What?\nAnswer:", "iv.",
     ["i.", "ii.", "iii.", "iv."]),
    ("Options: (A) How to change (B) Improve students (C) Only a little\nSolution:", "B", ["(A)", "(B)", "(C)"]),
    ("Question: Is it?\nChoices:\n1). yes.\n2). it is not possible to tell.\n3). no.", "1).", ["1).", "2).", "3)."]),
])
def test_labels_stay_as_they_are_and_every_option_is_a_segment_of_its_own(inputs, targets, labels):
    instruction, answer, info = fn.plan_example(inputs, targets, words)
    assert info["answer"] == "label" and info["options"] == len(labels)
    assert answer == [("L", targets)]
    literal = "".join(v for k, v in instruction if k == "L")
    segments = fn.segments_of(instruction)
    for label in labels:
        assert label in literal and not any(label in s for s in segments[1:])
    options = [inputs[start:stop] for _, _, start, stop in fn.find_options(inputs)]
    assert all(option in segments for option in options)
    nl_inputs, nl_targets, _ = translated(inputs, targets)
    assert nl_targets == targets and all(label in nl_inputs for label in labels)


def test_identity_translation_gives_back_every_example_unchanged():
    cases = [("Is this review positive?\nOPTIONS:\n- No\n- Yes", "Yes"), ("[73, 878, 639]\nOutput:", "[878, 11]"),
             ("  Leading and trailing  \n", "  answer with spaces  "), ("Answer with \"plausible\" or \"not plausible\".", "not plausible"),
             ("A. first. B. second. C. third.", "C."), ("Question?\n\nAvailable choices: [I] negative [II] positive", "[I]")]
    for inputs, targets in cases:
        nl_inputs, nl_targets, _ = translated(inputs, targets, lambda s: s)
        assert (nl_inputs, nl_targets) == (inputs, targets)


def test_text_without_words_is_not_translated():
    # The instruction has a word, so it is one segment, list and all; the answer has none and stays as it is
    instruction, answer, _ = fn.plan_example("[73, 878, 639, 977]\nOutput:", "[878, 11]", words)
    assert answer == [("L", "[878, 11]")] and instruction == [("S", "[73, 878, 639, 977]\nOutput:")]
    assert fn.text_pieces(" 4.5 ", words) == [("L", " 4.5 ")] and not fn.needs_translation("(B). 3 - 4")


def test_long_text_is_cut_at_paragraphs_then_sentences():
    paragraph = " ".join(["word"] * 300) + "."
    text = "\n\n".join([paragraph] * 3)
    pieces = fn.text_pieces(text, words, limit=512, splits=(splits := []))
    assert [k for k, _ in pieces] == ["S", "L", "S", "L", "S"] and splits == [2]
    assert fn.render(pieces, {s: s for s in fn.segments_of(pieces)}) == text
    sentences = " ".join(["Short sentence number one."] * 200)
    assert all(words(s) <= 512 for s in fn.segments_of(fn.text_pieces(sentences, words, limit=512)))
    with pytest.raises(fn.SegmentTooLong):
        fn.text_pieces(" ".join(["word"] * 600), words, limit=512)


@pytest.mark.parametrize("inputs_en, targets_en, inputs_nl, targets_nl, lost", [
    # Extractive: the answer is a span of the passage in both languages
    ("The papers were found by Sinister Foxx.", "Sinister Foxx", "De papieren zijn gevonden door Sinister Foxx.", "Sinister Foxx", False),
    ("Answer \"plausible\" or \"not plausible\".", "not plausible", "Antwoord \"aannemelijk\" of \"niet aannemelijk\".", "niet aannemelijk", False),
    # The span's own translation differs from its translation inside the passage
    ("Answer \"plausible\" or \"not plausible\".", "not plausible", "Antwoord \"plausibel\" of \"niet plausibel\".", "niet aannemelijk", True),
    # Whole words only, case and spacing aside: "no" is not in "know"
    ("I know nothing.", "no", "Ik weet niets.", "nee", False),
    ("Say Yes  or No.", "no", "Zeg Ja of Nee.", "NEE", False),
    # Not in the English instruction: no rule
    ("Write a review.", "A fine film.", "Schrijf een recensie.", "Een mooie film.", False),
])
def test_substring_rule(inputs_en, targets_en, inputs_nl, targets_nl, lost):
    assert fn.answer_lost(inputs_en, targets_en, inputs_nl, targets_nl) == lost


def test_task_filters():
    assert fn.task_drop_reason("glue/cola:2.0.0") == "english_form"
    assert fn.task_drop_reason("task107_splash_question_to_sql") == "code"
    assert fn.task_drop_reason("task183_rhyme_generation") == "word_game"
    assert fn.task_drop_reason("task583_udeps_eng_coarse_pos_tagging") == "token_level"
    assert fn.task_drop_reason("task560_alt_translation_en_entk") == "translation"
    assert fn.task_drop_reason("super_glue/wic:1.0.2") == "english_words"
    assert fn.task_drop_reason("glue/sst2:2.0.0") is None and fn.task_drop_reason("task1312_amazonreview_polarity_classification") is None
    assert "task1543_conll2002_parts_of_speech_tagging_answer_generation" in fn.DUTCH_TASKS
    assert not fn.DUTCH_TASKS & set(fn._REASON)
    assert fn.task_of("flan/niv2_zsopt_data/task391_causal_relationship") == "task391_causal_relationship"
    with pytest.raises(ValueError):
        fn.task_of("culturax")


def test_generic_checks_of_a_translation():
    dutch = lambda _: "nl"  # noqa: E731
    english = lambda _: "en"  # noqa: E731
    source = "The weather is lovely today and we will go to the beach."
    good = "Het weer is vandaag heerlijk en we gaan naar het strand."
    assert fn.check_translation(source, good, True, dutch) is None
    assert fn.check_translation(source, good, False, dutch) == "truncated"
    assert fn.check_translation(source, source, True, english) == "not_dutch"
    assert fn.check_translation(source, "Ja.", True, dutch) == "length_ratio"
    assert fn.check_translation(source, good + " " + good + " " + good, True, dutch) == "length_ratio"
    assert fn.check_translation(source, "Hier is de vertaling: " + good, True, dutch) == "commentary"
    assert fn.check_translation("not liking my parents", "niet overweg kunnen met / een hekel hebben aan", True, dutch) == "alternatives"
    assert fn.check_translation("km/h or mph", "km/u of mph / mijl", True, dutch) is None
    assert fn.check_translation("Note: bring a towel.", "Opmerking: neem een handdoek mee.", True, dutch) is None
    # Too short for a language ID: names and single words pass
    assert fn.check_translation("Sinister Foxx", "Sinister Foxx", True, english) is None


def _mix_rows():
    """F's sequence of a host: web rows with odd bytes, FLAN rows in between, in position order."""
    texts = ["Eerste document.\n\nMet één alinea.", "  spaties  en ⁇ tekens ", "Derde\tdocument", "laatste"]
    rows, position = [], 0
    for k, text in enumerate(texts):
        rows.append((position, text, None, None, "culturax"))
        position += 1
        for j in range(2):
            rows.append((position, None, f"instruction {k}.{j}", f"answer {k}.{j}", "flan/t0_zsopt_data/some_task"))
            position += 1
    return rows


def test_dutch_mix_replaces_flan_rows_in_place_and_keeps_web_rows_byte_identical(tmp_path):
    import build_decay_mix as bdm

    rows = _mix_rows()
    flan_positions = [p for p, _, _, _, s in rows if s != "culturax"]
    dropped = flan_positions[3]
    dutch = {p: (f"instructie {p}", f"antwoord {p}") for p in flan_positions if p != dropped}
    sequence, web_texts, pool, counts = fn.dutch_sequence(rows, dutch)
    assert counts == {"web": 4, "flan_in": 8, "flan_dropped": 1, "flan_kept": 7}
    # The order of F with the dropped row taken out, every FLAN row at the place of its English one
    expected = [("web", p) if s == "culturax" else ("flan", p) for p, _, _, _, s in rows if p != dropped]
    got = [("web", None) if kind == "web" else ("flan", pool[i][3]) for kind, i in sequence]
    assert got == [(kind, None if kind == "web" else p) for kind, p in expected]
    assert all(pool[i][:2] == dutch[pool[i][3]] for kind, i in sequence if kind == "flan")

    # Written as build_decay_mix writes a mix (here with the trainer's shuffle replaced by a fixed permutation)
    pool_rows = [(x, y, s, 0) for x, y, s, _ in pool]
    order = np.random.default_rng(0).permutation(len(sequence))
    path = tmp_path / "mix.parquet"
    bdm.write_mix(path, sequence, web_texts, pool_rows, order, "test-build")
    table = pq.read_table(path).sort_by("position")
    web_in = [t for _, t, _, _, s in rows if s == "culturax"]
    web_out = [t for t, s in zip(table.column("text").to_pylist(), table.column("source").to_pylist()) if s == "culturax"]
    assert [t.encode() for t in web_out] == [t.encode() for t in web_in]
    assert pq.ParquetFile(path).schema_arrow.metadata[bdm.METADATA_KEY] == b"test-build"


def test_dutch_mix_refuses_rows_out_of_order_and_dutch_rows_without_a_place():
    rows = _mix_rows()
    with pytest.raises(ValueError, match="position order"):
        fn.dutch_sequence(rows[1:] + rows[:1], {})
    with pytest.raises(ValueError, match="no FLAN row"):
        fn.dutch_sequence(rows, {0: ("instructie", "antwoord")})


def test_a_restart_translates_only_the_shards_that_are_not_stored():
    ids = list(range(10))
    texts = [f"segment {i}" for i in ids]
    shards = tr.shards_of(ids, texts, 4)
    assert [name for name, _, _ in shards] == ["shard-00000.parquet", "shard-00001.parquet", "shard-00002.parquet"]
    assert tr.shards_of(ids, texts, 4) == shards
    store, calls = {}, []

    def translate(shard_ids, shard_texts):
        calls.append(list(shard_ids))
        if len(calls) == 2 and not store.get("crashed"):
            store["crashed"] = True
            raise RuntimeError("preempted")
        return {"id": list(shard_ids), "text_nl": [fake_dutch(t) for t in shard_texts]}, 1.0

    def keep(name, rows, seconds):
        store[name] = rows

    with pytest.raises(RuntimeError, match="preempted"):
        tr.translate_pending(shards, set(), translate, keep)
    assert set(store) - {"crashed"} == {"shard-00000.parquet"}
    done = {name for name in store if name.endswith(".parquet")}
    calls.clear()
    assert tr.translate_pending(shards, done, translate, keep) == ["shard-00001.parquet", "shard-00002.parquet"]
    assert calls == [[4, 5, 6, 7], [8, 9]]
    assert tr.translate_pending(shards, set(store) - {"crashed"}, translate, keep) == []


def test_a_shard_with_translations_of_other_segments_is_refused():
    shards = tr.shards_of([1, 2], ["a", "b"], 2)
    with pytest.raises(RuntimeError, match="not those"):
        tr.translate_pending(shards, set(), lambda i, t: ({"id": [2, 1]}, 1.0), lambda *a: None)
