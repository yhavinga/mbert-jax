"""The stability flags of dmbert v3: --attention-float32 (query, key, RoPE and attention logits in float32) and --z-loss.

Without them the model and the train step are those of dmbert v2 bit for bit (fixture made on final-v2 @ 494d4d9 by
tests/v2_reference_outputs.py); with them the bf16 attention comes closer to the float32 model, and the objective
gains the z-loss while the logged loss stays the MLM loss.
"""

import sys
from pathlib import Path
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

import v2_reference_outputs as ref  # noqa: E402

from mmbjax.configuration_mmbbert import MMBertConfig  # noqa: E402
from mmbjax.heads.masked_lm import ModernBertForMaskedLM  # noqa: E402

FIXTURE = ROOT / "tests" / "fixtures" / "v2_reference.npz"


@pytest.fixture(scope="module")
def tm():
    with mock.patch("jax.distributed.initialize", side_effect=ValueError):
        import train_minimal

    return train_minimal


@pytest.fixture(scope="module")
def v2():
    with np.load(FIXTURE) as data:
        return {name: data[name] for name in data.files}


def test_without_the_flags_model_and_train_step_are_v2s_bit_for_bit(v2):
    now = ref.outputs(ROOT)
    assert sorted(now) == sorted(v2)
    differ = [name for name in v2 if not np.array_equal(now[name], v2[name], equal_nan=True)]
    assert not differ, f"{len(differ)} of {len(v2)} arrays differ from dmbert v2's, e.g. {differ[:3]}"


def test_the_flags_off_explicitly_are_v2s_too(v2):
    now = ref.outputs(ROOT, attention_float32=False, z_loss_coef=0.0)
    assert all(np.array_equal(now[name], v2[name], equal_nan=True) for name in v2)


def test_attention_float32_changes_only_the_bf16_model(v2):
    now = ref.outputs(ROOT, attention_float32=True)
    # In float32 compute query and key are float32 already: the flag is no op there
    assert all(np.array_equal(now[name], v2[name]) for name in v2 if name.startswith("float32/"))
    # In bf16 the attention changes (at this small init the logits only after training steps)
    plain, _ = _attention_probs("bfloat16", False, scale=1.0)
    fixed, _ = _attention_probs("bfloat16", True, scale=1.0)
    assert all(not np.array_equal(p, f) for p, f in zip(plain, fixed))
    assert any(not np.array_equal(now[name], v2[name]) for name in v2 if name.startswith("bfloat16/params/"))


def _attention_probs(dtype, attention_float32, scale=40.0):
    """Attention probabilities of every layer, with query/key weights scaled up so that the logits reach tens, where
    bf16 rounding of the logits (8 bits of mantissa) shows."""
    batch = ref.batch_arrays()
    config = MMBertConfig(**ref.config_kwargs(dtype), attention_float32=attention_float32)
    model = ModernBertForMaskedLM(config, dtype=getattr(jnp, dtype), param_dtype=jnp.float32)
    params = model.init(jax.random.PRNGKey(0), input_ids=jnp.asarray(batch["input_ids"]), deterministic=True)["params"]
    hidden = config.hidden_size
    for name, layer in params["model"]["encoder"].items():
        if name.startswith("layers_"):
            kernel = layer["attention"]["qkv"]["kernel"]
            layer["attention"]["qkv"]["kernel"] = kernel.at[:, : 2 * hidden].multiply(scale)
    out = model.apply({"params": params}, **{k: jnp.asarray(v) for k, v in batch.items() if k != "labels"},
                      deterministic=True, output_attentions=True)
    return [np.asarray(a, dtype=np.float64) for a in out.attentions], batch["attention_mask"]


def _exact_attention(dtype, attention_float32, scale=4.0, seq=48):
    """One global attention layer whose query, key and value are exact bf16 numbers (identity projections times a
    power of 2 on bf16 inputs), so that everything after the projection can be computed exactly in float64: its
    probabilities and those of the float64 reference."""
    from mmbjax.modules.attention import ModernBertAttention

    hidden, heads = 32, 4
    config = MMBertConfig(**{**ref.config_kwargs(dtype), "hidden_size": hidden, "num_attention_heads": heads},
                          attention_float32=attention_float32)
    layer = ModernBertAttention(config=config, layer_idx=0, dtype=getattr(jnp, dtype))
    eye = np.eye(hidden, dtype=np.float32)
    params = {"qkv": {"kernel": jnp.asarray(np.concatenate([scale * eye, scale * eye, eye], axis=1))},
              "out": {"kernel": jnp.asarray(eye)}}
    x = np.asarray(jnp.asarray(np.random.default_rng(7).normal(0, 1.5, (1, seq, hidden)), jnp.bfloat16), np.float32)
    _, probs = layer.apply({"params": params}, jnp.asarray(x), deterministic=True, output_attentions=True)
    # The float64 reference: RoPE of the global layers (theta 160,000), logits, softmax
    head_dim = hidden // heads
    q = (scale * x[0]).astype(np.float64).reshape(seq, heads, head_dim).transpose(1, 0, 2)
    inv_freq = 1.0 / (config.global_rope_theta ** (np.arange(0, head_dim, 2) / head_dim))
    angles = np.arange(seq)[:, None] * inv_freq[None, :]
    cos, sin = np.cos(np.concatenate([angles, angles], -1)), np.sin(np.concatenate([angles, angles], -1))
    rotate = np.concatenate([-q[..., head_dim // 2:], q[..., : head_dim // 2]], -1)
    q = q * cos + rotate * sin
    logits = np.einsum("hqd,hkd->hqk", q, q) / np.sqrt(head_dim)
    expected = np.exp(logits - logits.max(-1, keepdims=True))
    expected /= expected.sum(-1, keepdims=True)
    return np.asarray(probs, np.float64)[0], expected, np.abs(logits).max()


def test_attention_float32_gives_the_float64_attention_from_the_same_bf16_query_and_key():
    plain, expected, largest = _exact_attention("bfloat16", False)
    fixed, _, _ = _exact_attention("bfloat16", True)
    assert largest > 30, largest
    error_plain, error_fixed = np.abs(plain - expected).max(), np.abs(fixed - expected).max()
    # Only float32 rounding is left: the logits and the rotation's cos/sin are no longer bf16
    assert error_fixed < 1e-4, error_fixed
    assert error_plain > 50 * error_fixed, (error_plain, error_fixed)
    # And it is what the float32 model computes
    reference, _, _ = _exact_attention("float32", False)
    assert np.abs(fixed - reference).max() < 1e-5


def test_attention_float32_brings_the_bf16_model_closer_to_the_float32_model():
    reference, mask = _attention_probs("float32", False)
    plain, _ = _attention_probs("bfloat16", False)
    fixed, _ = _attention_probs("bfloat16", True)
    real = np.broadcast_to(mask.astype(bool)[:, None, :, None], reference[0].shape)  # query rows of real tokens
    for layer, (r, p, f) in enumerate(zip(reference, plain, fixed)):
        assert r.max() > 0.9, f"layer {layer}: the scaled logits should make some attention nearly one-hot"
        # What remains is the bf16 of the projections and the layers before, which the flag leaves alone
        error_plain, error_fixed = np.abs(p - r)[real].mean(), np.abs(f - r)[real].mean()
        assert error_fixed < 0.85 * error_plain, (layer, error_fixed, error_plain)


def test_z_loss_is_the_mean_squared_log_normalizer_at_masked_positions(tm):
    logits = jnp.array([[[0.0, 0.0], [3.0, 1.0], [10.0, -5.0]]], dtype=jnp.bfloat16)
    labels = jnp.array([[1, -100, 0]])
    expected = (np.log(2.0) ** 2 + np.logaddexp(10.0, -5.0) ** 2) / 2
    assert float(tm.compute_z_loss(logits, labels)) == pytest.approx(expected, rel=1e-6)
    assert float(tm.compute_z_loss(logits, jnp.full_like(labels, -100))) == 0.0


def _train(tm, z_loss_coef, steps, learning_rate=1e-2):
    from flax.training import train_state

    batch = {k: jnp.asarray(v) for k, v in ref.batch_arrays().items()}
    config = MMBertConfig(**ref.config_kwargs("bfloat16"), attention_float32=True)
    model = ModernBertForMaskedLM(config, dtype=jnp.bfloat16, param_dtype=jnp.float32)
    params = model.init(jax.random.PRNGKey(0), input_ids=batch["input_ids"], deterministic=True)["params"]
    optimizer, _ = tm.create_train_state(model, learning_rate=learning_rate, num_train_steps=1000, warmup_steps=1,
                                         lr_schedule_type="constant", layer_norm_lr_factor=1.0,
                                         skip_nonfinite_updates=True)
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)
    step = jax.jit(lambda s, b, r: tm.train_step(s, b, r, learning_rate, False, z_loss_coef=z_loss_coef))
    rng, history = jax.random.PRNGKey(1), []
    for _ in range(steps):
        state, metrics, rng = step(state, batch, rng)
        history.append({k: np.asarray(v) for k, v in metrics.items()})
    return state, history, model, batch


def test_z_loss_keeps_the_logged_loss_the_mlm_loss_and_adds_its_gradient(tm):
    _, plain, _, _ = _train(tm, 0.0, 1)
    state, with_z, model, batch = _train(tm, 1e-4, 1)
    assert "z_loss" not in plain[0] and "z_loss" in with_z[0]
    # Same parameters before the first step: the MLM loss is the same number
    assert np.array_equal(plain[0]["loss"], with_z[0]["loss"])
    logits = model.apply({"params": _train(tm, 0.0, 0)[0].params}, input_ids=batch["input_ids"],
                         attention_mask=batch["attention_mask"], position_ids=batch["position_ids"],
                         segment_ids=batch["segment_ids"], deterministic=True).logits
    assert float(with_z[0]["z_loss"]) == pytest.approx(float(tm.compute_z_loss(logits, batch["labels"])), rel=1e-4)
    assert not np.array_equal(plain[0]["grad_norm"], with_z[0]["grad_norm"])


def test_z_loss_pulls_the_log_normalizer_towards_zero(tm):
    _, history, _, _ = _train(tm, 1.0, 30)
    z = [float(h["z_loss"]) for h in history]
    assert z[0] > 1.0, z[0]
    assert z[-1] < 0.2 * z[0], z


@pytest.mark.parametrize("argv, expected", [([], 0.0), (["--z-loss"], 1e-4), (["--z-loss", "3e-4"], 3e-4)])
def test_z_loss_flag(tm, argv, expected):
    with mock.patch.object(sys, "argv", ["train_minimal.py", *argv]):
        args = tm.parse_args()
    assert args.z_loss == expected
    assert args.attention_float32 is False


def test_attention_float32_flag_and_a_negative_z_loss(tm):
    with mock.patch.object(sys, "argv", ["train_minimal.py", "--attention-float32"]):
        assert tm.parse_args().attention_float32 is True
    with mock.patch.object(sys, "argv", ["train_minimal.py", "--z-loss", "-1e-4"]), pytest.raises(SystemExit):
        tm.parse_args()


def test_nonfinite_diagnostics_change_no_update(v2):
    now = ref.outputs(ROOT, nonfinite_diagnostics=True)
    assert all(np.array_equal(now[name], v2[name], equal_nan=True) for name in v2)


def test_nonfinite_groups_name_the_layer_types_with_nan_or_inf(tm):
    grads = {"model": {"embeddings": {"w": jnp.ones(2)},
                       "encoder": {"layers_0": {"attention": {"qkv": {"kernel": jnp.array([1.0, jnp.nan])}},
                                                "mlp": {"wi": {"kernel": jnp.ones(2)}}},
                                   "layers_1": {"attention": {"qkv": {"kernel": jnp.ones(2)}},
                                                "mlp": {"wo": {"kernel": jnp.array([jnp.inf, 0.0])}}},
                                   "final_layer_norm": {"scale": jnp.ones(2)}}},
             "decoder": {"bias": jnp.ones(2)}}
    bad = {name for name, flag in tm.nonfinite_gradient_groups(grads).items() if bool(flag)}
    assert bad == {"encoder_0_attention", "encoder_1_ffn"}
    names = ["embeddings", "encoder_0_attention", "encoder_10_ffn", "encoder_2_ffn", "lm_head", "encoder_10_attention",
             "final_layer_norm", "encoder_2_attention"]
    assert tm.order_from_output(names) == ["lm_head", "final_layer_norm", "encoder_10_ffn", "encoder_10_attention",
                                           "encoder_2_ffn", "encoder_2_attention", "encoder_0_attention", "embeddings"]


def test_nonfinite_diagnostics_need_the_step_skip(tm):
    with mock.patch.object(sys, "argv", ["train_minimal.py", "--nonfinite-diagnostics"]), pytest.raises(SystemExit):
        tm.parse_args()
    with mock.patch.object(sys, "argv", ["train_minimal.py", "--nonfinite-diagnostics", "--skip-nonfinite-updates"]):
        assert tm.parse_args().nonfinite_diagnostics is True
