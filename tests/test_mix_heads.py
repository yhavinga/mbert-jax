"""--mix-heads: a prediction head per group of the denoisers of --masking mix, with the encoder and the decoder weight shared."""
import sys
import types
from pathlib import Path
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import traverse_util
from flax.training import train_state

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from mmbjax.configuration_mmbbert import MMBertConfig  # noqa: E402
from mmbjax.heads.masked_lm import ModernBertForMaskedLM  # noqa: E402
from mmbjax.training import DENOISER_NAMES, MIX_HEADS, mix_head_count  # noqa: E402

# Computed with the code of 652e335, before --mix-heads, on this config, init and batch
GOLDEN_TREE = {
    "decoder/bias": (64,),
    "model/embeddings/layer_norm/scale": (16,),
    "model/embeddings/word_embeddings/embedding": (64, 16),
    "model/encoder/final_layer_norm/scale": (16,),
    "model/encoder/layers_0/attention/out/kernel": (16, 16),
    "model/encoder/layers_0/attention/qkv/kernel": (16, 48),
    "model/encoder/layers_0/mlp/wi/kernel": (16, 64),
    "model/encoder/layers_0/mlp/wo/kernel": (32, 16),
    "model/encoder/layers_0/mlp_norm/scale": (16,),
    "model/encoder/layers_1/attention/out/kernel": (16, 16),
    "model/encoder/layers_1/attention/qkv/kernel": (16, 48),
    "model/encoder/layers_1/attention_norm/scale": (16,),
    "model/encoder/layers_1/mlp/wi/kernel": (16, 64),
    "model/encoder/layers_1/mlp/wo/kernel": (32, 16),
    "model/encoder/layers_1/mlp_norm/scale": (16,),
    "prediction_head/dense/kernel": (16, 16),
    "prediction_head/layer_norm/scale": (16,),
}
GOLDEN_FORWARD = {"loss": 4.137704372406006, "accuracy": 0.09090909361839294, "logits_sum": 5.642856121063232}
GOLDEN_TRAIN_STEPS = [
    {"loss": 4.137704372406006, "accuracy": 0.09090909361839294, "grad_norm": 1.6656708717346191},
    {"loss": 4.102840900421143, "accuracy": 0.09090909361839294, "grad_norm": 1.5777173042297363},
]
HEAD_1 = {"prediction_head_1/dense/kernel": (16, 16), "prediction_head_1/layer_norm/scale": (16,), "decoder_1/bias": (64,)}


@pytest.fixture(scope="module")
def tm():
    # Other test modules already started JAX, after which its module-level initialize() can't run
    with mock.patch("jax.distributed.initialize", side_effect=ValueError):
        import train_minimal

    return train_minimal


def _config():
    return MMBertConfig(
        vocab_size=64, hidden_size=16, num_hidden_layers=2, num_attention_heads=2, intermediate_size=32,
        max_position_embeddings=64, local_attention=8, global_attn_every_n_layers=2, pad_token_id=0, bos_token_id=1,
        eos_token_id=2, compute_dtype="float32", param_dtype="float32",
    )


def _model(heads=1):
    return ModernBertForMaskedLM(_config(), dtype=jnp.float32, param_dtype=jnp.float32, num_prediction_heads=heads)


def _init(model):
    ones = jnp.ones((1, 8), dtype=jnp.int32)
    return model.init(jax.random.PRNGKey(0), input_ids=ones, attention_mask=ones, deterministic=True)["params"]


def _batch():
    """Two rows of two documents of 8 tokens, 40% of the non-<s> tokens masked."""
    rng = np.random.RandomState(0)
    ids = rng.randint(5, 64, size=(2, 16)).astype(np.int32)
    ids[:, 0] = 1
    ids[:, 8] = 1
    labels = np.where(rng.random_sample((2, 16)) < 0.4, ids, -100).astype(np.int32)
    labels[:, [0, 8]] = -100
    return {
        "input_ids": np.where(labels != -100, 4, ids).astype(np.int32),
        "attention_mask": np.ones((2, 16), dtype=np.int32),
        "labels": labels,
        "position_ids": np.tile(np.r_[np.arange(8), np.arange(8)], (2, 1)).astype(np.int32),
        "segment_ids": np.tile(np.r_[np.zeros(8), np.ones(8)], (2, 1)).astype(np.int32),
    }


def _shapes(params):
    return {"/".join(path): tuple(leaf.shape) for path, leaf in traverse_util.flatten_dict(params).items()}


def _logits(model, params, batch, head_ids=None):
    return model.apply({"params": params}, input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                       position_ids=batch["position_ids"], segment_ids=batch["segment_ids"], head_ids=head_ids,
                       deterministic=True).logits


def _distinct_heads(params, seed=1):
    """Head 1 moved away from its init, so a mix-up of heads can't go unnoticed."""
    noise = jax.random.PRNGKey(seed)
    moved = dict(params)
    for name in ("prediction_head_1", "decoder_1"):
        leaves, tree = jax.tree_util.tree_flatten(params[name])
        keys = jax.random.split(noise, len(leaves))
        moved[name] = jax.tree_util.tree_unflatten(tree, [leaf + jax.random.normal(k, leaf.shape) for k, leaf in zip(keys, leaves)])
    return moved


def test_the_table_groups_every_denoiser_and_keeps_the_held_out_masks_on_head_0():
    for choice, heads in MIX_HEADS.items():
        assert sorted(heads) == sorted(DENOISER_NAMES), choice
        assert sorted(set(heads.values())) == list(range(mix_head_count(choice))), choice
        # Head 0 predicts the 15% single tokens of --eval-file, so it must be the head that learns them
        assert heads["single_r15"] == 0, choice
    assert [mix_head_count(choice) for choice in ("shared", "rate2", "per-denoiser")] == [1, 2, 8]
    rate2 = MIX_HEADS["rate2"]
    assert {name for name, head in rate2.items() if head == 0} == {"single_r15", "span_mu3_r15", "span_mu8_r15", "span_mu64_r15"}
    assert {name for name, head in rate2.items() if head == 1} == {"span_mu3_r50", "span_mu8_r50", "span_mu64_r50", "short_prefix"}


def test_shared_is_the_model_and_the_numbers_of_the_code_before_the_flag(tm):
    for model in (ModernBertForMaskedLM(_config(), dtype=jnp.float32, param_dtype=jnp.float32), _model(heads=1)):
        params = _init(model)
        assert _shapes(params) == GOLDEN_TREE
        batch = {key: jnp.asarray(value) for key, value in _batch().items()}
        logits = _logits(model, params, batch)
        loss, accuracy, _ = tm.compute_mlm_loss_and_accuracy(logits, batch["labels"])
        assert float(loss) == pytest.approx(GOLDEN_FORWARD["loss"], rel=1e-6)
        assert float(accuracy) == pytest.approx(GOLDEN_FORWARD["accuracy"], rel=1e-6)
        assert float(jnp.sum(logits)) == pytest.approx(GOLDEN_FORWARD["logits_sum"], rel=1e-5)

        state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optax.adamw(1e-3))
        step = jax.jit(tm.train_step, static_argnums=(4,))
        dropout_rng = jax.random.PRNGKey(1)
        for golden in GOLDEN_TRAIN_STEPS:
            state, metrics, dropout_rng = step(state, batch, dropout_rng, 1e-3, False)
            for name, value in golden.items():
                assert float(metrics[name]) == pytest.approx(value, rel=1e-6), name
            assert not any(name.startswith("head") for name in metrics)


def test_a_head_adds_its_own_prediction_head_and_bias_and_leaves_the_rest_as_it_was():
    single, double = _init(_model(1)), _init(_model(2))
    assert _shapes(double) == {**GOLDEN_TREE, **HEAD_1}
    # Flax draws a parameter from its path, so the shared parameters and head 0 start where a one-head model starts
    flat_single, flat_double = traverse_util.flatten_dict(single), traverse_util.flatten_dict(double)
    for path, leaf in flat_single.items():
        assert np.array_equal(np.asarray(flat_double[path]), np.asarray(leaf)), path
    assert not np.array_equal(np.asarray(double["prediction_head_1"]["dense"]["kernel"]), np.asarray(double["prediction_head"]["dense"]["kernel"]))
    hidden, vocab = 16, 64
    added = sum(leaf.size for path, leaf in flat_double.items() if path not in flat_single)
    assert added == hidden * hidden + hidden + vocab
    assert _shapes(_init(_model(8))).keys() == GOLDEN_TREE.keys() | {
        f"{name}_{k}/{leaf}" for k in range(1, 8) for name, leaf in
        [("prediction_head", "dense/kernel"), ("prediction_head", "layer_norm/scale"), ("decoder", "bias")]
    }


def test_each_position_gets_the_logits_of_the_head_of_its_id():
    model = _model(2)
    params = _distinct_heads(_init(model))
    batch = {key: jnp.asarray(value) for key, value in _batch().items()}
    head_ids = jnp.asarray(np.random.RandomState(3).randint(0, 2, size=(2, 16)), dtype=jnp.int32)
    assert 0 < int(head_ids.sum()) < head_ids.size

    routed = np.asarray(_logits(model, params, batch, head_ids))
    each = [np.asarray(_logits(model, params, batch, jnp.full((2, 16), k, dtype=jnp.int32))) for k in range(2)]
    assert not np.allclose(each[0], each[1])
    expected = np.where(np.asarray(head_ids)[..., None] == 1, each[1], each[0])
    np.testing.assert_allclose(routed, expected, rtol=1e-6, atol=1e-6)

    # Without head ids: head 0, which is what a one-head model with the same parameters computes
    single = _model(1)
    single_params = {name: value for name, value in params.items() if name not in ("prediction_head_1", "decoder_1")}
    unrouted = np.asarray(_logits(model, params, batch))
    assert np.array_equal(unrouted, np.asarray(_logits(single, single_params, batch)))
    assert np.array_equal(unrouted, each[0])

    with pytest.raises(ValueError, match="head_ids given to a model with one prediction head"):
        _logits(single, single_params, batch, head_ids)


def test_a_head_gets_gradient_from_its_own_tokens_only(tm):
    model = _model(2)
    params = _distinct_heads(_init(model))
    batch = {key: jnp.asarray(value) for key, value in _batch().items()}
    head_ids = jnp.asarray(np.tile(np.r_[np.zeros(8), np.ones(8)], (2, 1)), dtype=jnp.int32)

    def grads_with_labels_of(head):
        labels = jnp.where(head_ids == head, batch["labels"], -100)
        assert int(jnp.sum(labels != -100)) > 0

        def loss(p):
            return tm.compute_mlm_loss_and_accuracy(_logits(model, p, batch, head_ids), labels)[0]

        return jax.grad(loss)(params)

    def norm(tree):
        return float(jnp.sqrt(sum(jnp.sum(jnp.square(leaf)) for leaf in jax.tree_util.tree_leaves(tree))))

    own = {0: ("prediction_head", "decoder"), 1: ("prediction_head_1", "decoder_1")}
    for head in (0, 1):
        grads = grads_with_labels_of(head)
        for name in own[head]:
            assert norm(grads[name]) > 0, (head, name)
        for name in own[1 - head]:
            assert norm(grads[name]) == 0, (head, name)
        # The encoder and the decoder weight, the embedding matrix, learn from both
        assert norm(grads["model"]["encoder"]) > 0
        assert norm(grads["model"]["embeddings"]["word_embeddings"]) > 0


def test_train_step_logs_each_head_and_trains_on_the_mean_over_all_masked_tokens(tm):
    model = _model(2)
    params = _distinct_heads(_init(model))
    batch = {key: jnp.asarray(value) for key, value in _batch().items()}
    head_ids = jnp.asarray(np.random.RandomState(5).randint(0, 2, size=(2, 16)), dtype=jnp.int32)
    batch["head_ids"] = head_ids

    # By hand, from the routed logits
    logits = _logits(model, params, batch, head_ids)
    log_probs = np.asarray(jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1))
    labels = np.asarray(batch["labels"])
    masked = labels != -100
    nll = -np.take_along_axis(log_probs, np.where(masked, labels, 0)[..., None], axis=-1)[..., 0]
    hit = np.asarray(jnp.argmax(logits, axis=-1)) == labels
    expected = {}
    for k in range(2):
        mine = masked & (np.asarray(head_ids) == k)
        assert mine.sum() > 0
        expected[k] = (nll[mine].mean(), hit[mine].mean(), int(mine.sum()))

    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optax.adamw(1e-3))
    step = jax.jit(lambda s, b, r, lr: tm.train_step(s, b, r, lr, False, num_heads=2))
    new_state, metrics, _ = step(state, batch, jax.random.PRNGKey(1), 1e-3)
    for k, (loss, accuracy, count) in expected.items():
        assert float(metrics[f"head{k}_loss"]) == pytest.approx(loss, rel=1e-5)
        assert float(metrics[f"head{k}_accuracy"]) == pytest.approx(accuracy, rel=1e-6)
        assert int(metrics[f"head{k}_masked"]) == count
    # The loss trained on: one mean over all masked tokens, so each head weighs by its count
    total = sum(loss * count for loss, _, count in expected.values()) / sum(count for *_, count in expected.values())
    assert float(metrics["loss"]) == pytest.approx(total, rel=1e-5)
    assert int(metrics["n_masked"]) == int(masked.sum())
    moved = traverse_util.flatten_dict(jax.tree_util.tree_map(lambda a, b: float(jnp.max(jnp.abs(a - b))), new_state.params, state.params))
    assert all(value > 0 for path, value in moved.items() if path[0] in ("prediction_head", "prediction_head_1")), moved


def test_with_every_position_on_head_0_two_heads_train_like_one(tm):
    """Head 1 then gets no gradient, and the shared parameters and head 0 take the steps of the one-head model."""
    model = _model(2)
    params = _init(model)
    batch = {key: jnp.asarray(value) for key, value in _batch().items()}
    batch["head_ids"] = jnp.zeros((2, 16), dtype=jnp.int32)
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optax.adamw(1e-3))
    step = jax.jit(lambda s, b, r, lr: tm.train_step(s, b, r, lr, False, num_heads=2))
    dropout_rng = jax.random.PRNGKey(1)
    for golden in GOLDEN_TRAIN_STEPS:
        state, metrics, dropout_rng = step(state, batch, dropout_rng, 1e-3)
        for name, value in golden.items():
            assert float(metrics[name]) == pytest.approx(value, rel=1e-6), name
        assert float(metrics["head0_loss"]) == pytest.approx(golden["loss"], rel=1e-6)
        assert int(metrics["head1_masked"]) == 0 and float(metrics["head1_loss"]) == 0


def test_per_layer_norms_keep_lm_head_for_head_0_and_name_the_others(tm):
    params = _init(_model(2))
    norms = tm.compute_per_layer_gradient_norms(jax.tree_util.tree_map(jnp.ones_like, params))
    assert {"lm_head", "lm_head_1"} <= set(norms) and "other" not in norms
    assert float(norms["lm_head"]) ** 2 == pytest.approx(16 * 16 + 16 + 64)
    assert float(norms["lm_head_1"]) ** 2 == pytest.approx(16 * 16 + 16 + 64)
    total = sum(float(v) ** 2 for v in norms.values())
    assert total == pytest.approx(sum(leaf.size for leaf in jax.tree_util.tree_leaves(params)), rel=1e-5)


def _args(tm, *flags):
    with mock.patch.object(sys, "argv", ["train_minimal.py", *flags]):
        return tm.parse_args()


def test_the_flag_is_for_the_mix_only_and_shared_keeps_the_data_state(tm, tmp_path):
    assert _args(tm).mix_heads == "shared"
    assert _args(tm, "--use-packing", "--masking", "mix").mix_heads == "shared"
    assert _args(tm, "--use-packing", "--masking", "mix", "--mix-heads", "rate2").mix_heads == "rate2"
    for flags in (["--mix-heads", "rate2"], ["--use-packing", "--mix-heads", "rate2"],
                  ["--use-packing", "--masking", "uniform15", "--mix-heads", "per-denoiser"],
                  ["--use-packing", "--masking", "mix", "--mix-heads", "rate3"]):
        with pytest.raises(SystemExit):
            _args(tm, *flags)
    tokenizer = types.SimpleNamespace(pad_token_id=0, mask_token_id=3, all_special_ids=[0, 1, 2, 3], vocab_size=100)
    with pytest.raises(ValueError, match="routes the denoisers of --masking mix"):
        tm.create_masking_collator("uniform15", tokenizer, "rate2")
    assert tm.create_masking_collator("mix", tokenizer, "rate2").head_of_denoiser.tolist() == [MIX_HEADS["rate2"][n] for n in DENOISER_NAMES]
    assert tm.create_masking_collator("mix", tokenizer).head_of_denoiser is None

    args = types.SimpleNamespace(dataset="d", dataset_config="c", datasets_config=None, max_length=1024, tokenizer="t", shuffle_buffer_size=10)
    # Data states of mix runs saved before the flag stay valid for a shared run
    assert tm.data_state_identity(args, 32, ["a"], "mix", "shared") == tm.data_state_identity(args, 32, ["a"], "mix")
    assert tm.data_state_identity(args, 32, ["a"], "mix", "rate2")["mix_heads"] == "rate2"
    path = tm.epath.Path(tmp_path) / "state.json"
    loader = mock.Mock(get_state=mock.Mock(return_value={"epoch": 0}))
    for saved, resumed in [("rate2", "shared"), ("shared", "rate2"), ("rate2", "per-denoiser")]:
        tm.save_data_state(path, loader, tm.data_state_identity(args, 32, ["a"], "mix", saved), 5000)
        with pytest.raises(RuntimeError, match="another data setup"):
            tm.restore_data_state(path, loader, tm.data_state_identity(args, 32, ["a"], "mix", resumed), allow_restart=True)


def _save_params(tm, directory, params):
    checkpointer = tm.ocp.PyTreeCheckpointer()
    checkpointer.save(directory / "default", {"params": params})
    return directory


def test_init_from_a_multi_head_run_into_the_same_heads_or_into_head_0_of_one(tm, tmp_path, capsys):
    saved_params = _distinct_heads(_init(_model(2)))
    step_dir = _save_params(tm, tmp_path / "rate2" / "12", saved_params)

    # The same heads: every parameter, head 1 included
    loaded = tm.load_params_of_run(step_dir, _init(_model(2)))
    for path, leaf in traverse_util.flatten_dict(saved_params).items():
        assert np.array_equal(np.asarray(traverse_util.flatten_dict(loaded)[path]), np.asarray(leaf)), path

    # One head, as an anneal with single tokens: head 0 and the shared parameters, head 1 left behind
    capsys.readouterr()
    loaded = tm.load_params_of_run(step_dir, _init(_model(1)))
    assert _shapes(loaded) == GOLDEN_TREE
    for path, leaf in traverse_util.flatten_dict(loaded).items():
        assert np.array_equal(np.asarray(leaf), np.asarray(traverse_util.flatten_dict(saved_params)[path])), path
    assert "head 0 becomes the one prediction head; left behind: decoder_1, prediction_head_1" in capsys.readouterr().out

    # Other heads than the run had, or a one-head run into two heads: refused
    with pytest.raises(ValueError, match="the saved parameters are not those of this model"):
        tm.load_params_of_run(step_dir, _init(_model(8)))
    single_dir = _save_params(tm, tmp_path / "shared" / "12", _init(_model(1)))
    with pytest.raises(ValueError, match="the saved parameters are not those of this model"):
        tm.load_params_of_run(single_dir, _init(_model(2)))
