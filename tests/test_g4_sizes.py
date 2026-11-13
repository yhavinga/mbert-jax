"""The smaller Gemma 4 models as en->nl translators (experiments/en-nl-translator/g4_sizes.py, translate_models.yml):
their entries, the 12B's MaxText overrides, the conversion's command line, the check against Hugging Face, (a1) and
the comparison table. The tests of the real chat templates run where the tokenizers are (G4_E4B_TOKENIZER,
G4_12B_TOKENIZER: a directory with tokenizer.json and chat_template.jinja of that model)."""
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "en-nl-translator"))
sys.path.insert(0, str(ROOT / "experiments" / "flan-nl"))

import bt_natural as bn  # noqa: E402
import g4_sizes as g  # noqa: E402
import translate as tr  # noqa: E402

E4B_TOKENIZER = Path(os.environ.get("G4_E4B_TOKENIZER", "/dev/shm/translate-generic/tok-e4b"))
TOKENIZER_12B = Path(os.environ.get("G4_12B_TOKENIZER", "/dev/shm/translate-generic/tok-12b"))


def write(rows, path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    pq.write_table(pa.Table.from_pylist(rows), path)


@pytest.mark.parametrize("size", sorted(g.SIZES))
def test_each_size_has_the_31bs_prompt_classes_and_stop_tokens(size):
    model, big = tr.MODELS[g.SIZES[size]], tr.MODELS["gemma4-31b"]
    assert model["length_classes"] == big["length_classes"]
    assert model["eos_ids"] == big["eos_ids"] == [1, 106, 50]
    assert model["turn"]["prefix"] == big["turn"]["prefix"] and model["turn"]["check"]
    assert set(model["slots"]) == set(model["length_classes"])
    assert tr.with_prompt(model, g.PROMPT)["prompt"] == "natural-v1"
    assert len(model["hf_revision"]) == 40 and model["hf_repo"].startswith("google/gemma-4-")
    assert set(model["checkpoints"]) == {"europe-west4", "us-east1"}
    assert all(f"gemma-4-{size}-it-maxtext/0/items" in c for c in model["checkpoints"].values())


def test_chips_follow_the_fewest_kv_heads():
    # MaxText refuses tensor parallelism beyond a layer's KV heads: E2B 1, E4B 2, 12B 1 global, 26B 2 global; a v6e
    # process takes 1 or 4 chips (2 failed with START_SESSION), so the 26B is tensor 2 x data 2 on 4
    meshes = {s: (tr.MODELS[m]["chips"], tr.mesh_of(tr.MODELS[m], tr.MODELS[m]["chips"])) for s, m in g.SIZES.items()}
    assert meshes == {"e2b": (1, (1, 1)), "e4b": (1, (1, 1)), "12b": (1, (1, 1)), "26b": (4, (2, 2))}
    assert tr.mesh_of(tr.MODELS["gemma4-31b"], 4) == (4, 1)


def test_a_tensor_parallelism_that_does_not_divide_the_chips_is_an_error():
    with pytest.raises(ValueError, match="does not divide"):
        tr.mesh_of({"tensor": 3}, 4)


def test_only_the_small_models_turn_closes_no_thought_channel():
    # Their own chat templates: E2B/E4B add nothing after the model's turn with thinking off, 12B/26B/31B an empty channel
    for size in ("e2b", "e4b"):
        assert tr.MODELS[g.SIZES[size]]["turn"]["suffix"] == "<turn|>\n<|turn>model\n"
    for name in ("gemma4-12b", "gemma4-26b"):
        assert tr.MODELS[name]["turn"]["suffix"] == tr.MODELS["gemma4-31b"]["turn"]["suffix"]


def test_the_12b_runs_as_the_31b_with_its_own_sizes():
    model = tr.MODELS["gemma4-12b"]
    assert model["maxtext_model"] == "gemma4-31b"
    assert tr.engine_overrides(model) == {"override_model_config": True, "base_num_decoder_layers": 48,
                                          "base_emb_dim": 3840, "base_num_query_heads": 16, "base_num_kv_heads": 8,
                                          "base_mlp_dim": 15360, "global_num_kv_heads": 1}
    assert tr.engine_overrides(tr.MODELS["gemma4-31b"]) == {}


def test_overrides_may_not_set_what_make_engine_sets():
    with pytest.raises(ValueError, match="ici_tensor_parallelism"):
        tr.engine_overrides({"maxtext_overrides": {"ici_tensor_parallelism": 8}})


def test_convert_argv_is_the_judges_conversion_with_the_overrides():
    argv = g.convert_argv(tr.MODELS["gemma4-12b"], "/hf", "/out", "run")
    assert argv[:2] == ["to_maxtext", "model_name=gemma4-31b"]
    assert "scan_layers=false" in argv and "use_multimodal=false" in argv and "base_output_directory=/out" in argv
    assert "override_model_config=true" in argv and "global_num_kv_heads=1" in argv
    assert argv[-3:] == ["--hf_model_path=/hf", "--lazy_load_tensors=True", "--save_dtype=bfloat16"]
    plain = g.convert_argv(tr.MODELS["gemma4-e2b"], "/hf", "/out", "run")
    assert "model_name=gemma4-e2b" in plain and not any(a.startswith("override_model_config") for a in plain)


def test_agreement_and_the_segments_of_the_check():
    assert g.agreement([1, 2, 3, 4], [1, 2, 0, 4]) == 0.75
    assert g.agreement([], []) is None
    rows = {"b": {"ended": True, "text_nl": "twee"}, "a": {"ended": False, "text_nl": "een"},
            "c": {"ended": True, "text_nl": "  "}, "d": {"ended": True, "text_nl": "vier"}}
    assert g.pick(rows, 2) == ["b", "d"]


def evalcompare_fixture(tmp_path, qe_shift):
    """A small evalcompare result (as test_bt_natural's) with the model's QE shifted by qe_shift"""
    meta = json.dumps({"is_bad_source": False})
    evalsets, pairs, metricx, comet, base_scores, base_texts = [], [], [], [], [], []
    for name, n in {"wmt24pp": 3, "flores": 2, "ntrex": 2, "europarl": 2}.items():
        for k in range(n):
            i = f"{name}-{k}"
            src, ref = "The House agrees with the motion today.", "De Kamer stemt vandaag in met de motie."
            evalsets.append({"id": i, "set": name, "source": src, "reference": ref, "meta": meta})
            pairs.append({"id": i, "set": name, "source": src, "hypothesis": ref, "reference": ref, "ended": True,
                          "decode_loop": False, "english_left": k == 1, "empty": False})
            metricx.append({"id": i, "ref_source_hypothesis_reference": 1.0, "qe_source_hypothesis": 1.0 + qe_shift})
            comet.append({"id": i, "comet22": 0.85})
            base_scores.append({"id": i, "set": name, "metricx_ref": 1.2, "metricx_qe": 1.2, "comet22": 0.86,
                                "ended": True, "decode_loop": False, "english_left": False, "empty": False})
            base_texts.append({"id": i, "set": name, "source": src, "reference": ref,
                               "hypothesis": "De Kamer is het vandaag eens met de motie.", "meta": meta})
    d = tmp_path / f"shift{qe_shift}"
    (d / "data").mkdir(parents=True)
    write(evalsets, d / "data" / "evalsets.parquet")
    write(pairs, d / "pairs.parquet")
    write(metricx, d / "metricx.parquet")
    write(comet, d / "comet.parquet")
    (d / "tg").mkdir()
    write(base_scores, d / "tg" / "scores.parquet")
    write(base_texts, d / "tg" / "translations.parquet")
    write([{"id": "x1", "set": "flan", "source": "Answer.", "hypothesis": "Vertaal de volgende tekst",
            "ended": True, "decode_loop": False, "english_left": False, "empty": False}], d / "flan.parquet")
    write([{"id": "x1", "qe_source_hypothesis": 2.5}], d / "flan-qe.parquet")
    (d / "logs").mkdir()
    (d / "logs" / "translate_eval-x-1.log").write_text(
        'engine ready in 40 s\nSHARD {"class": "tiny", "segments": 9, "tokens_in": 90, "tokens_out": 300, '
        '"seconds": 2.0}\nengine ready in 50 s\n')
    out = d / "eval.md"
    bn.main(["evalcompare", "--data", str(d / "data"), "--pairs", str(d / "pairs.parquet"),
             "--metricx", str(d / "metricx.parquet"), "--comet", str(d / "comet.parquet"),
             "--flan-pairs", str(d / "flan.parquet"), "--flan-metricx", str(d / "flan-qe.parquet"),
             "--base", str(d / "tg"), "--logs", str(d / "logs"), "--label", "Gemma 4 E2B-it", "--out", str(out)])
    return d, out


def test_evalcompare_names_the_model_it_was_given(tmp_path):
    _, out = evalcompare_fixture(tmp_path, 0.0)
    assert out.read_text().startswith("# Gemma 4 E2B-it als en→nl-vertaler")
    assert json.loads(out.with_suffix(".json").read_text())["label"] == "Gemma 4 E2B-it"


def test_a1_is_the_mean_qe_against_the_human_reference_per_set():
    refqe = {s: {"ref_mean": 2.0} for s in g.A1_SETS}
    sets = {s: {"gemma4": {"metricx_qe": {"mean": 2.0}}} for s in g.A1_SETS}
    assert g.a1(sets, refqe)["holds"]  # equal to the human reference is not higher
    sets["ntrex"]["gemma4"]["metricx_qe"]["mean"] = 2.001
    r = g.a1(sets, refqe)
    assert not r["holds"] and not r["sets"]["ntrex"]["holds"] and r["sets"]["flores"]["holds"]


def test_report_puts_the_sizes_beside_the_31b_tg_and_the_humans(tmp_path):
    d, out = evalcompare_fixture(tmp_path, 0.0)
    (d / "extra-e2b.json").write_text(json.dumps({"chips": 1, "leak": "0 / 1"}))
    big, big_out = evalcompare_fixture(tmp_path, 2.0)
    refqe = {s: {"ref_mean": 2.5, "delta": 1.3} for s in ("wmt24pp", "flores", "ntrex", "europarl")}
    (tmp_path / "refqe.json").write_text(json.dumps(refqe))
    md = tmp_path / "sizes.md"
    g.main(["report", "--eval", f"e2b={out.with_suffix('.json')}", "--g4", str(big_out.with_suffix(".json")),
            "--refqe", str(tmp_path / "refqe.json"), "--out", str(md)])
    lines = md.read_text().splitlines()
    assert lines[0] == "| maat | Gemma 4 E2B-it | Gemma 4 31B-it | TG 27B | menselijke referentie |"
    cells = [[c.strip() for c in line.strip().strip("|").split(" | ")] for line in lines[2:]]
    row = {c[0]: c[1:] for c in cells}
    assert row["MetricX-QE WMT24++ (960)"] == ["1,000", "3,000", "1,200", "2,500"]
    assert row["(a1): QE niet hoger dan de mens (WMT / FLORES / NTREX)"][:2] == ["**ja**", "nee (wmt24pp, flores, ntrex)"]
    assert row["niet af / lus / Engels over / leeg, vier sets"][0] == "0 / 0 / 4 / 0"
    assert row["prompt in de uitvoer (regex van de pilot), vier sets / FLAN"][0] == "0 / 1"
    assert row["engine laden (s, per lengteklasse)"][0] == "40–50"
    assert row["uitvoertokens/s per proces (vier sets + FLAN)"][0] == "150"
    summary = json.loads(md.with_suffix(".json").read_text())
    assert summary["Gemma 4 E2B-it"]["a1"]["holds"] and not summary["Gemma 4 31B-it"]["a1"]["holds"]


def test_leak_counts_the_pilots_regex(tmp_path):
    write([{"id": "a", "hypothesis": "Vertaal de volgende Engelse tekst"}, {"id": "b", "hypothesis": "Goed."},
           {"id": "c", "hypothesis": None}], tmp_path / "four.parquet")
    write([{"id": "x", "hypothesis": "als een native speaker"}], tmp_path / "flan.parquet")
    assert g.leak_counts(tmp_path / "four.parquet", tmp_path / "flan.parquet") == "1 / 1"


@pytest.mark.parametrize("size,directory", [("e4b", E4B_TOKENIZER), ("12b", TOKENIZER_12B)])
def test_real_turn_matches_the_models_own_chat_template(size, directory):
    if not (directory / "tokenizer.json").exists():
        pytest.skip(f"{size}'s tokenizer is not in {directory}")
    AutoTokenizer = pytest.importorskip("transformers").AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(directory))
    model = tr.with_prompt(tr.MODELS[g.SIZES[size]], g.PROMPT)
    prompts = tr.ModelPrompts(tok, model, "en", "nl")
    prompts.check(["The House agrees.", "  Leading and trailing space.  ", "Two\n\nparagraphs."])
