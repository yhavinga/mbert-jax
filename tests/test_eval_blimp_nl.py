"""scripts/eval_blimp_nl.py on a tiny model on the CPU: words, the PLL-word-l2r masks, pair scoring, packed against
unpacked scoring, and the heads of a --mix-heads checkpoint."""
import sys
import types
from pathlib import Path
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dumbx_tokenizer import make_tokenizer  # noqa: E402
from test_multihost import _stub_wandb_if_missing  # noqa: E402


@pytest.fixture(scope="module")
def eb():
    _stub_wandb_if_missing()
    # Other test modules already started JAX, after which train_minimal's module-level initialize() can't run
    with mock.patch("jax.distributed.initialize", side_effect=ValueError):
        import eval_blimp_nl

    return eval_blimp_nl


def test_words_start_at_whitespace_and_around_punctuation(eb):
    # The real tokenizer's pieces of "Het merendeel lijkt een ijzige kalmte, zegt 's-Hertogenbosch."
    tokens = ["▁Het", "▁merendeel", "▁lijkt", "▁een", "▁ijz", "ige", "▁kalm", "te", ",", "▁zegt", "▁'", "s", "-", "Hertogenbosch", "."]
    assert eb.word_starts(tokens) == [True, True, True, True, True, False, True, False, True, True, True, True, True, True, True]
    # Byte fallback continues a word; a lone ▁ starts one that the next piece continues
    assert eb.word_starts(["▁caf", "<0xC3>", "<0xA9>", "▁is"]) == [True, False, False, True]
    assert eb.word_starts(["▁", "1", "0", "%"]) == [True, False, False, True]
    assert eb.word_starts(["kamer", "."]) == [True, True]


def test_l2r_masks_the_token_and_the_rest_of_its_word(eb):
    assert eb.l2r_masks([True, False, False, True, False, True]) == [[0, 1, 2], [1, 2], [2], [3, 4], [4], [5]]


def test_each_token_gets_a_copy_with_its_l2r_mask_and_specials_stay(eb):
    BOS, EOS, MASK = 1, 2, 99
    copies = eb.masked_copies([10, 11, 12, 13], ["▁ijz", "ige", "▁kalm", "."], BOS, EOS, MASK)
    assert [(c.tolist(), p) for c, p in copies] == [
        ([BOS, MASK, MASK, 12, 13, EOS], 1),
        ([BOS, 10, MASK, 12, 13, EOS], 2),
        ([BOS, 10, 11, MASK, 13, EOS], 3),
        ([BOS, 10, 11, 12, MASK, EOS], 4),
    ]


def test_pairs_are_right_when_the_grammatical_sentence_scores_strictly_higher(eb):
    pairs = [{"item_nr": i, "phenomenon_nr": 1 + i // 4, "phenomenon": f"p{1 + i // 4}", "paradigm_nr": 1 + i // 2,
              "paradigm": f"q{1 + i // 2}", "handcrafted": i < 2} for i in range(8)]
    grammatical = [-10.0, -10.0, -5.0, -7.0, -3.0, -3.0, -8.0, -1.0]
    ungrammatical = [-12.0, -10.0, -6.0, -6.0, -4.0, -9.0, -2.0, -2.0]
    scores = eb.score_pairs(pairs, grammatical, ungrammatical)
    assert scores["accuracy"] == pytest.approx(5 / 8) and scores["ties"] == 1 and scores["pairs"] == 8
    assert {k: v["accuracy"] for k, v in scores["paradigms"].items()} == {"1": 0.5, "2": 0.5, "3": 1.0, "4": 0.5}
    assert scores["paradigms"]["2"]["phenomenon"] == "p1"
    assert {k: (v["name"], v["accuracy"]) for k, v in scores["phenomena"].items()} == {"1": ("p1", 0.5), "2": ("p2", 0.75)}
    assert scores["subsets"]["handcrafted"] == {"accuracy": 0.5, "pairs": 2, "ties": 1}
    assert scores["subsets"]["generated"]["accuracy"] == pytest.approx(4 / 6)
    assert scores["paradigm_mean_accuracy"] == pytest.approx(0.625)


ARGS = ["--hidden-size", "16", "--num-layers", "2", "--num-heads", "2", "--intermediate-size", "32"]


def _model(eb):
    tokenizer = make_tokenizer()
    tokenizer.add_special_tokens({"mask_token": "<mask>"})
    args = types.SimpleNamespace(hidden_size=16, num_layers=2, num_heads=2, intermediate_size=32, global_attn_every_n_layers=3,
                                 compute_dtype="float32")
    config, model = eb.build_model(args, tokenizer)
    return tokenizer, model


def _save(eb, directory, params, model):
    """A checkpoint the way the trainer saves one."""
    import optax
    from flax.training import train_state

    tm = eb.tm
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optax.adamw(1e-3))
    manager = tm.ocp.CheckpointManager(directory, options=tm.ocp.CheckpointManagerOptions(create=True))
    manager.save(5, args=tm.ocp.args.StandardSave(state))
    manager.wait_until_finished()
    return directory / "5"


@pytest.fixture(scope="module")
def checkpoints(tmp_path_factory, eb):
    """A one-head checkpoint and a two-head one whose head 0 is the one-head model's."""
    tokenizer, model = _model(eb)
    ones = jnp.ones((1, 8), dtype=jnp.int32)
    params = model.init(jax.random.PRNGKey(3), input_ids=ones, attention_mask=ones, deterministic=True)["params"]
    other = model.init(jax.random.PRNGKey(4), input_ids=ones, attention_mask=ones, deterministic=True)["params"]
    two_heads = {**params, "prediction_head_1": other["prediction_head"],
                 "decoder_1": jax.tree_util.tree_map(lambda x: x + 0.5, other["decoder"])}
    one = _save(eb, tmp_path_factory.mktemp("one"), params, model)
    two = _save(eb, tmp_path_factory.mktemp("two"), two_heads, model)
    return model, params, two_heads, one, two


def _pairs_file(path):
    sentences = [("Wij bekijken de foto van mezelf in de kamer.", "Ik bekijken de foto van mezelf in de kamer."),
                 ("De man zit in de kamer.", "De man zitten in de kamer."),
                 ("Jan woont in Amsterdam bij de vrouw.", "Jan wonen in Amsterdam bij de vrouw."),
                 ("De ijzige kalmte van de man.", "Het ijzige kalmte van de man."),
                 ("Piet zag Jan in de kamer.", "Piet zag in Jan de kamer."),
                 ("Ik bekijk de foto.", "Ik bekijken de foto.")]
    pairs = [{"item_nr": i + 1, "phenomenon_nr": 1 + i // 3, "phenomenon": f"p{1 + i // 3}", "paradigm_nr": 1 + i // 2,
              "paradigm": f"q{1 + i // 2}", "handcrafted": i < 3, "grammatical": g, "ungrammatical": u}
             for i, (g, u) in enumerate(sentences)]
    pq.write_table(pa.Table.from_pylist(pairs), path)
    return pairs


def _reference_pll(eb, model, params, sentence):
    """Each masked copy alone, unpacked: what the packed rows must give."""
    tokenizer = make_tokenizer()
    tokenizer.add_special_tokens({"mask_token": "<mask>"})
    ids = tokenizer(sentence, add_special_tokens=False)["input_ids"]
    total = 0.0
    for inputs, position in eb.masked_copies(ids, tokenizer.convert_ids_to_tokens(ids), tokenizer.bos_token_id,
                                             tokenizer.eos_token_id, tokenizer.mask_token_id):
        logits = model.apply({"params": params}, input_ids=jnp.asarray(inputs[None]), deterministic=True).logits[0]
        total += float(jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)[position, ids[position - 1]])
    return total


def _run(eb, monkeypatch, checkpoint, data, extra=()):
    monkeypatch.setattr(eb, "AutoTokenizer", types.SimpleNamespace(from_pretrained=lambda *_a, **_k: make_tokenizer()))
    return eb.main(["--checkpoint", str(checkpoint), "--data", str(data), "--tokenizer", "fake", "--row-length", "48",
                    "--rows-per-device", "2", *ARGS, *extra])


def test_packed_scores_are_those_of_each_copy_alone(eb, checkpoints, tmp_path, monkeypatch):
    model, params, _, one, _ = checkpoints
    pairs = _pairs_file(tmp_path / "large.parquet")
    result = _run(eb, monkeypatch, one, tmp_path / "large.parquet", ["--output", str(tmp_path / "out.json")])
    assert (tmp_path / "out.json").exists()
    assert result["pairs"] == 6 and result["rows"] < result["masked_copies"], "copies must share rows"
    assert result["prediction_heads"] == 1 and result["metric"] == "PLL-word-l2r"
    for pair, (item, g, u, _, _) in zip(pairs, result["pair_scores"]):
        assert item == pair["item_nr"]
        assert g == pytest.approx(_reference_pll(eb, model, params, pair["grammatical"]), rel=1e-5)
        assert u == pytest.approx(_reference_pll(eb, model, params, pair["ungrammatical"]), rel=1e-5)
    right = [g > u for _, g, u, _, _ in result["pair_scores"]]
    assert result["accuracy"] == pytest.approx(np.mean(right))
    assert set(result["phenomena"]) == {"1", "2"} and set(result["paradigms"]) == {"1", "2", "3"}


def test_a_mix_heads_checkpoint_is_scored_with_the_chosen_head(eb, checkpoints, tmp_path, monkeypatch):
    model, params, two_heads, one, two = checkpoints
    pairs = _pairs_file(tmp_path / "large.parquet")
    head0 = _run(eb, monkeypatch, two, tmp_path / "large.parquet")
    assert head0["prediction_heads"] == 2 and head0["export_head"] == 0
    single = _run(eb, monkeypatch, one, tmp_path / "large.parquet")
    np.testing.assert_allclose(np.asarray(head0["pair_scores"]), np.asarray(single["pair_scores"]), rtol=1e-6)
    head1 = _run(eb, monkeypatch, two, tmp_path / "large.parquet", ["--export-head", "1"])
    swapped = {key: value for key, value in two_heads.items() if key not in ("prediction_head_1", "decoder_1")}
    swapped["prediction_head"], swapped["decoder"] = two_heads["prediction_head_1"], two_heads["decoder_1"]
    assert head1["pair_scores"][0][1] == pytest.approx(_reference_pll(eb, model, swapped, pairs[0]["grammatical"]), rel=1e-5)
    assert head1["pair_scores"][0][1] != pytest.approx(head0["pair_scores"][0][1], rel=1e-3)
    with pytest.raises(ValueError, match="heads 0 to 1"):
        _run(eb, monkeypatch, two, tmp_path / "large.parquet", ["--export-head", "2"])
