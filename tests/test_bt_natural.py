"""experiments/en-nl-translator/bt_natural.py without a model: the stratified sample, the prompts' fit in the length
classes, the measures of a pair (R2's length bound, chatter), the merge and report of the pilot, the calibration of
rule (a) and the en->nl comparison. No TPU: run with --noconftest where the conftest's imports (jax, flax) are not
installed. The test of the real prompt lengths runs where Gemma 4's tokenizer is (GEMMA4_TOKENIZER)."""

import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "en-nl-translator"))
sys.path.insert(0, str(ROOT / "experiments" / "flan-nl"))
sys.path.insert(0, str(ROOT / "tests"))

import bt_natural as bn  # noqa: E402
import translate as tr  # noqa: E402

GEMMA4_TOKENIZER = Path(os.environ.get("GEMMA4_TOKENIZER", "/dev/shm/translate-generic/tokenizer-gemma4"))


def row(i, nl_words=10, en_words=10, qe=1.0, length_class="short"):
    return {"id": f"s:{i:04d}", "nl": " ".join(["woord"] * nl_words), "en": " ".join(["word"] * en_words),
            "qe_en_nl": qe, "length_class": length_class}


def test_draw_takes_the_strata_in_order_without_a_passage_twice():
    rows = ([row(i) for i in range(100)] + [row(100 + i, length_class="long") for i in range(10)]
            + [row(200 + i, qe=6.0) for i in range(10)] + [row(300 + i, nl_words=5) for i in range(10)])
    strata = (("random", 40), ("long", 5), ("qe_high", 5), ("ratio_low", 5))
    out, info = bn.draw(rows, "dpc", strata=strata)
    ids = [r["id"] for r in out]
    assert len(ids) == len(set(ids)) == 55
    assert [r["stratum"] for r in out] == ["random"] * 40 + ["long"] * 5 + ["qe_high"] * 5 + ["ratio_low"] * 5
    assert all(r["source"] == "dpc" for r in out)
    by = {r["id"]: r for r in rows}
    assert all(by[r["id"]]["length_class"] == "long" for r in out if r["stratum"] == "long")
    assert all(by[r["id"]]["qe_en_nl"] > 5 for r in out if r["stratum"] == "qe_high")
    assert all(bn.word_ratio(r["nl"], r["en"]) <= info["ratio_cut"] for r in out if r["stratum"] == "ratio_low")
    assert info["short"] == {}
    again, _ = bn.draw(list(reversed(rows)), "dpc", strata=strata)
    assert [r["id"] for r in again] == ids


def test_draw_reports_a_stratum_whose_pool_is_too_small():
    rows = [row(i) for i in range(20)] + [row(100, length_class="long")]
    out, info = bn.draw(rows, "dpc", strata=(("random", 5), ("long", 3)))
    assert sum(r["stratum"] == "long" for r in out) <= 1
    assert info["short"]["long"][1] == 3


def test_class_of_is_translate_pys_class():
    classes = tr.MODELS["gemma4-31b"]["length_classes"]
    assert bn.class_of(32, classes) == "tiny" and bn.class_of(33, classes) == "small"
    assert bn.class_of(512, classes) == "xlong" and bn.class_of(513, classes) == "xxlong"
    with pytest.raises(SystemExit):
        bn.class_of(769, classes)


class CharTokenizer:
    """Gemma 4's chat turn with a token per character (as test_translate_models.Gemma4Tokenizer)"""

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        text = "<bos><|turn>user\n" + messages[0]["content"].strip() + "<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
        return [ord(c) for c in text] if tokenize else text

    def __call__(self, text, add_special_tokens):
        return {"input_ids": [ord(c) for c in text]}


def test_check_prompts_refuses_a_prompt_beyond_its_class_prefill():
    # a token per character: the prompt is ~150 tokens, within the prefill of 384 of the class medium
    short = [{"id": "a", "text": "x", "tokens": 200}]
    assert set(bn.check_prompts(short, CharTokenizer(), "nl", "en", prompts=("bare-v1",))) == {"bare-v1"}
    # 300 characters of text in the class tiny (its count says 30 tokens): far beyond its prefill of 128
    with pytest.raises(SystemExit):
        bn.check_prompts([{"id": "b", "text": "y" * 300, "tokens": 30}], CharTokenizer(), "nl", "en", prompts=("bare-v1",))


def test_the_new_prompts_name_the_languages_and_keep_the_text():
    text = tr.user_text(tr.PROMPTS["natural-v1"], " Dit is een zin.\n", "nl", "en")
    assert text.startswith("Translate the following Dutch text into natural, idiomatic English")
    assert "leave nothing out and add nothing" in text and text.endswith("\n\n\nDit is een zin.")
    assert tr.user_text(tr.PROMPTS["bare-v1"], "Een zin.", "en", "nl") == (
        "Translate the following English text into Dutch. Output only the translation.\n\n\nEen zin.")
    model = tr.with_prompt(tr.MODELS["gemma4-31b"], "natural-v1")
    assert model["prompt"] == "natural-v1" and model["context_prompt"] == "chat-v1"


@pytest.mark.skipif(not (GEMMA4_TOKENIZER / "tokenizer.json").exists(), reason="Gemma 4's tokenizer is not here")
def test_the_new_prompts_fit_the_smallest_headroom_of_the_classes():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(GEMMA4_TOKENIZER))
    classes = tr.MODELS["gemma4-31b"]["length_classes"]
    headroom = min(c["prefill_lengths"][-1] - c["max_source"] for c in classes.values())
    for name in bn.PROMPTS:
        for source, target in (("nl", "en"), ("en", "nl")):
            prompts = tr.ModelPrompts(tok, tr.with_prompt(tr.MODELS["gemma4-31b"], name), source, target)
            prompts.check(["Dit is een zin."])
            assert len(prompts.encode("x")) - 1 <= headroom


def test_chatter_finds_an_introduction_or_a_note_and_not_plain_text():
    assert bn.chatter("Here is the translation:\n\nThe House agrees.")
    assert bn.chatter("Sure! The House agrees.")
    assert bn.chatter("The House agrees.\n\nNote: 'Kamer' is the House of Representatives.")
    assert bn.chatter("**Translation:** The House agrees.")
    assert not bn.chatter("The House agrees. There is a note on the table.")
    assert not bn.chatter("Of the motions, two are accepted.")
    assert not bn.chatter("")


def test_measure_applies_r2s_bound_of_the_source():
    nl = "De Kamer stemt in met de motie over de begroting van het ministerie."
    ok = bn.measure(nl, "The House agrees with the motion on the ministry's budget.", "handelingen", True)
    assert not ok["r2_length_out"] and ok["hard_fails"] == [] and not ok["chatter"]
    assert ok["word_ratio"] == round(13 / 10, 4)
    long_en = bn.measure(nl, "The House of Representatives agrees, after a long and careful debate, with the motion "
                             "on the budget of the ministry.", "handelingen", True)
    assert long_en["r2_length_out"] and long_en["char_ratio"] < bn.R2_BOUNDS["handelingen"]["lo"]
    unfinished = bn.measure(nl, "The House agrees with the motion on the ministry's budget.", "handelingen", False)
    assert unfinished["hard_fails"] == ["not_ended"]


def write(rows, path):
    bn.write_parquet(rows, path)
    return path


def make_pilot(tmp_path):
    texts = [("De minister zegt dat het kabinet de motie uitvoert en de Kamer voor de zomer informeert.",
              "The minister states that the cabinet will implement the motion and that it will inform the House of "
              "Representatives before the summer.", "The minister says the cabinet will carry out the motion and "
              "inform the House before the summer."),
             ("Het college heeft besloten om 12 bomen te kappen.", "The board decided to cut trees.",
              "The board has decided to fell 12 trees."),
             ("Wij vragen u om dit formulier vóór 1 mei in te vullen.", "We ask you to complete this form before 1 May.",
              "Please fill in this form before 1 May.")]
    sample = []
    for k, (nl, en, new) in enumerate(texts):
        for source in bn.SOURCES:
            sample.append({"id": f"{source}:{k}", "source": source, "stratum": "random" if k < 2 else "qe_high",
                           "length_class": "short", "class_g4": "small", "tokens_nl": 20, "tokens_g4": 20, "nl": nl,
                           "en": en, "qe_en_nl": 2.0 + k, "ended": True, "tokens_en": 25, "_new": new})
    write([{k: v for k, v in r.items() if k != "_new"} for r in sample], tmp_path / "sample.parquet")
    for setup in ("natural", "bare"):
        write([{"id": r["id"], "text_en": r["_new"], "ended": True, "tokens_in": 120, "tokens_out": 18} for r in sample],
              tmp_path / "shards" / setup / "small" / "shard-00000.parquet")
    return sample


def test_merge_and_report_of_the_pilot(tmp_path):
    sample = make_pilot(tmp_path)
    bn.main(["merge", "--sample", str(tmp_path / "sample.parquet"), "--shards", f"natural={tmp_path}/shards/natural",
             f"bare={tmp_path}/shards/bare", "--out", str(tmp_path / "compare.parquet")])
    rows = bn.read_rows(tmp_path / "compare.parquet")
    assert len(rows) == len(sample)
    first = next(r for r in rows if r["id"] == "dpc:0")
    assert first["en_tg"].startswith("The minister states") and first["en_natural"].startswith("The minister says")
    assert first["word_ratio_natural"] > first["word_ratio_tg"]
    second = next(r for r in rows if r["id"] == "dpc:1")
    assert second["numbers_missing_tg"] and not second["numbers_missing_natural"]
    qe = [{"id": r["id"], "qe_nl_en_tg": 3.0, "qe_en_tg_nl": r["qe_en_nl_published"] + 0.01, "qe_nl_en_natural": 2.0,
           "qe_en_natural_nl": 1.5, "qe_nl_en_bare": 2.5, "qe_en_bare_nl": 6.0} for r in rows]
    write(qe, tmp_path / "qe.parquet")
    logs = tmp_path / "logs"
    logs.mkdir()
    shard = {"class": "small", "shard": "shard-00000.parquet", "segments": 15, "tokens_in": 1800, "tokens_out": 270,
             "seconds": 10.0}
    for setup in ("natural", "bare"):
        (logs / f"translate_{setup}-run1.log").write_text(f"engine ready in 60 s\nSHARD {json.dumps(shard)}\n")
    full = {"dpc": {"small": {"passages": 1000, "tokens": 20000}}, "fineweb": {"medium": {"passages": 10, "tokens": 2000}}}
    (tmp_path / "full.json").write_text(json.dumps(full))
    (tmp_path / "setup.json").write_text(json.dumps({"VM": "test"}))
    out = tmp_path / "bt.md"
    bn.main(["report", "--compare", str(tmp_path / "compare.parquet"), "--qe", str(tmp_path / "qe.parquet"),
             "--logs", str(logs), "--fullset", str(tmp_path / "full.json"), "--setup", str(tmp_path / "setup.json"),
             "--out", str(out)])
    result = json.loads(out.with_suffix(".json").read_text())
    assert result["random"]["natural"]["n"] == 10 and result["strata"]["qe_high"]["natural"]["n"] == 5
    assert result["all"]["bare"]["qe_over_5.0"] == 1.0 and result["all"]["natural"]["qe_over_5.0"] == 0.0
    assert result["port_check"]["max_abs"] == pytest.approx(0.01)
    e = result["estimate"]["natural"]
    # 15 passages in 10 s of decoding on 4 chips; the full set: 22,000 source tokens at 270/300 output tokens per source
    # token (the sample's) and 27 tokens/s
    assert e["passages_per_chip_hour"] == pytest.approx(15 / (10 / 3600) / 4)
    assert e["process_hours"] == pytest.approx(22000 * 18 / 20 / 27 / 3600)
    assert e["passages"] == 1010
    text = out.read_text()
    assert "filter raakt" in text and "Tien voorbeelden" in text and "| Nederlands (origineel) |" in text


def test_refqe_compares_the_human_references_with_translategemma(tmp_path):
    evalsets = [{"id": "w1", "set": "wmt24pp", "source": "a", "reference": "b", "meta": json.dumps({"is_bad_source": False})},
                {"id": "w2", "set": "wmt24pp", "source": "a", "reference": "b", "meta": json.dumps({"is_bad_source": True})},
                {"id": "f1", "set": "flores", "source": "a", "reference": "b", "meta": "{}"},
                {"id": "n1", "set": "ntrex", "source": "a", "reference": "b", "meta": "{}"},
                {"id": "e1", "set": "europarl", "source": "a", "reference": "b", "meta": "{}"}]
    write(evalsets, tmp_path / "evalsets.parquet")
    write([{"id": r["id"], "qe_source_reference": {"w1": 2.0, "w2": 9.0}.get(r["id"], 1.0)} for r in evalsets],
          tmp_path / "refqe.parquet")
    (tmp_path / "tg").mkdir()
    write([{"id": r["id"], "metricx_qe": 1.5} for r in evalsets], tmp_path / "tg" / "scores.parquet")
    out = tmp_path / "refqe.md"
    bn.main(["refqe", "--evalsets", str(tmp_path / "evalsets.parquet"), "--qe", str(tmp_path / "refqe.parquet"),
             "--base", str(tmp_path / "tg"), "--out", str(out)])
    r = json.loads(out.with_suffix(".json").read_text())
    assert r["wmt24pp"]["n"] == 1 and r["wmt24pp"]["delta"] == pytest.approx(0.5) and not r["wmt24pp"]["within_0_20"]
    assert r["wmt24pp_all"]["n"] == 2 and r["wmt24pp_all"]["ref_mean"] == pytest.approx(5.5)
    assert r["flores"]["within_0_20"] and r["flores"]["ref_worse_share"] == 0
    assert "| wmt24pp | 1 |" in out.read_text()


def test_evalcompare_against_translategemma(tmp_path):
    meta = json.dumps({"is_bad_source": False})
    sets = {"wmt24pp": 3, "flores": 2, "ntrex": 2, "europarl": 2}
    evalsets, pairs, metricx, comet, base_scores, base_texts = [], [], [], [], [], []
    for name, n in sets.items():
        for k in range(n):
            i = f"{name}-{k}"
            src, ref = "The House agrees with the motion today.", "De Kamer stemt vandaag in met de motie."
            evalsets.append({"id": i, "set": name, "source": src, "reference": ref, "meta": meta})
            hyp = "De Kamer stemt in met de motie." if k == 0 else ref
            pairs.append({"id": i, "set": name, "source": src, "hypothesis": hyp, "reference": ref, "ended": True,
                          "decode_loop": False, "english_left": False, "empty": False})
            metricx.append({"id": i, "ref_source_hypothesis_reference": 1.0, "qe_source_hypothesis": 1.0 + k})
            comet.append({"id": i, "comet22": 0.85})
            base_scores.append({"id": i, "set": name, "metricx_ref": 1.2, "metricx_qe": 1.2, "comet22": 0.86,
                                "ended": True, "decode_loop": False, "english_left": False, "empty": False})
            base_texts.append({"id": i, "set": name, "source": src, "reference": ref,
                               "hypothesis": "De Kamer is het vandaag eens met de motie.", "meta": meta})
    (tmp_path / "data").mkdir()
    write(evalsets, tmp_path / "data" / "evalsets.parquet")
    write(pairs, tmp_path / "pairs.parquet")
    write(metricx, tmp_path / "metricx.parquet")
    write(comet, tmp_path / "comet.parquet")
    (tmp_path / "tg").mkdir()
    write(base_scores, tmp_path / "tg" / "scores.parquet")
    write(base_texts, tmp_path / "tg" / "translations.parquet")
    flan = [{"id": "x1", "set": "flan", "source": "Answer the question.", "hypothesis": "Beantwoord de vraag.",
             "ended": True, "decode_loop": False, "english_left": False, "empty": False}]
    write(flan, tmp_path / "flan.parquet")
    write([{"id": "x1", "qe_source_hypothesis": 2.5}], tmp_path / "flan-qe.parquet")
    (tmp_path / "logs").mkdir()
    out = tmp_path / "eval.md"
    bn.main(["evalcompare", "--data", str(tmp_path / "data"), "--pairs", str(tmp_path / "pairs.parquet"),
             "--metricx", str(tmp_path / "metricx.parquet"), "--comet", str(tmp_path / "comet.parquet"),
             "--flan-pairs", str(tmp_path / "flan.parquet"), "--flan-metricx", str(tmp_path / "flan-qe.parquet"),
             "--base", str(tmp_path / "tg"), "--logs", str(tmp_path / "logs"), "--out", str(out)])
    r = json.loads(out.with_suffix(".json").read_text())
    assert r["rule_a"]["sets"]["wmt24pp"]["delta"] == pytest.approx(2.0 - 1.2)
    assert not r["rule_a"]["holds"]
    assert r["sets"]["flores"]["gemma4"]["length"]["out_per_ref"] == pytest.approx((7 / 8 + 1) / 2)
    assert r["flan"]["qe_mean"] == 2.5
    assert "Regel (a)" in out.read_text() and "| bron (Engels) |" in out.read_text()


def test_fullset_counts_every_passage_and_a_context_turn_beyond_the_largest_class(tmp_path):
    for key in bn.SOURCES:
        write([{"nl": "een twee drie"}], tmp_path / key / "data" / "train-00000-of-00001.parquet")
        write([{"nl": "vier"}], tmp_path / key / "data" / "validation-00000-of-00001.parquet")
    write([{"nl": "x " * 900}, {"nl": "kort"}], tmp_path / "handelingen" / "context_turns" / "train-00000-of-00001.parquet")
    write([{"nl": "kort"}], tmp_path / "handelingen" / "context_turns" / "validation-00000-of-00001.parquet")

    class Words:
        def many(self, texts):
            return [len(t.split()) for t in texts]

    full = bn.fullset(tmp_path, Words(), tr.MODELS["gemma4-31b"]["length_classes"])
    assert full["dpc"] == {"tiny": {"passages": 2, "tokens": 4}}
    assert full["handelingen_context"] == {"over_largest": {"passages": 1, "tokens": 900}, "tiny": {"passages": 2, "tokens": 2}}


@pytest.mark.skipif(not (GEMMA4_TOKENIZER / "tokenizer.json").exists(), reason="Gemma 4's tokenizer is not here")
def test_sample_writes_the_segments_the_probe_and_the_fullset(tmp_path):
    medium = "De Kamer heeft vandaag uitgebreid gesproken over de begroting van het ministerie. " * 12
    for key in bn.SOURCES:
        rows = [dict(row(i, qe=6.0 if i % 7 == 0 else 1.0, length_class="long" if i % 5 == 0 else "short"),
                     id=f"{key}:{i:03d}", nl=medium if i % 2 else "Een korte zin over de begroting van dit jaar.",
                     hard_fails=[], ended=True, tokens_nl=150, tokens_en=150) for i in range(40)]
        write(rows, tmp_path / "hf" / key / "data" / "train-00000-of-00001.parquet")
        write(rows[:2], tmp_path / "hf" / key / "data" / "validation-00000-of-00001.parquet")
    write([{"nl": "kort"}], tmp_path / "hf" / "handelingen" / "context_turns" / "train-00000-of-00001.parquet")
    write([{"nl": "kort"}], tmp_path / "hf" / "handelingen" / "context_turns" / "validation-00000-of-00001.parquet")
    bn.main(["sample", "--data", str(tmp_path / "hf"), "--tokenizer", str(GEMMA4_TOKENIZER), "--out", str(tmp_path / "out")])
    out = tmp_path / "out"
    sample = bn.read_rows(out / "sample.parquet")
    assert len(sample) == 5 * 40 and len({r["id"] for r in sample}) == 200
    segments = bn.read_rows(out / "segments-nl.parquet")
    assert [s["id"] for s in segments] == [r["id"] for r in sample]
    probe = bn.read_rows(out / "probe-nl.parquet")
    assert len(probe) == 6 and len({p["id"] for p in probe}) == 6
    assert all(129 <= p["tokens"] <= 256 for p in probe)
    assert set(json.loads((out / "fullset.json").read_text())) == set(bn.SOURCES) | {"handelingen_context"}
    assert (out / "classes.txt").read_text().split() == ["tiny", "medium"]
