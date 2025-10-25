import os

import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import jax.numpy as jnp
import jax.random as jr

from src.mmbjax.modules import attention as attention_mod
from src.mmbjax.modules.attention import ModernBertAttention
from src.mmbjax.modules.rope import clear_rotary_cache


def test_attention_switches_rope_scale_between_layers(monkeypatch, tiny_config):
    config = tiny_config.copy(
        global_rope_theta=1000.0,
        local_rope_theta=10.0,
        global_attn_every_n_layers=2,
    )
    clear_rotary_cache()

    recorded_thetas = []
    original_get_rotary_cache = attention_mod.get_rotary_cache

    def tracking_get_rotary_cache(head_dim, theta, dtype=jnp.float32):
        recorded_thetas.append(theta)
        return original_get_rotary_cache(head_dim, theta, dtype=dtype)

    monkeypatch.setattr(attention_mod, "get_rotary_cache", tracking_get_rotary_cache)

    hidden_states = jnp.ones((1, 3, config.hidden_size), dtype=jnp.float32)
    attention_mask = jnp.ones((1, 3), dtype=jnp.int32)

    global_attn = ModernBertAttention(config, layer_idx=0)
    global_attn.init(jr.PRNGKey(0), hidden_states, attention_mask=attention_mask, deterministic=True)

    local_attn = ModernBertAttention(config, layer_idx=1)
    local_attn.init(jr.PRNGKey(1), hidden_states, attention_mask=attention_mask, deterministic=True)

    assert recorded_thetas[:2] == [
        pytest.approx(config.global_rope_theta),
        pytest.approx(config.local_rope_theta),
    ]


def test_local_attention_sliding_window_masks_out_distant_tokens(tiny_config):
    config = tiny_config.copy(
        local_attention=4,
        global_attn_every_n_layers=10,
    )
    clear_rotary_cache()

    seq_len = 6
    hidden_states = jnp.arange(seq_len * config.hidden_size, dtype=jnp.float32).reshape(1, seq_len, config.hidden_size)
    attention_mask = jnp.ones((1, seq_len), dtype=jnp.int32)

    module = ModernBertAttention(config, layer_idx=1)
    variables = module.init(jr.PRNGKey(2), hidden_states, attention_mask=attention_mask, deterministic=True)
    _, attn_probs = module.apply(
        variables,
        hidden_states,
        attention_mask=attention_mask,
        deterministic=True,
        output_attentions=True,
    )

    assert attn_probs is not None
    probs = np.asarray(attn_probs)
    radius = config.local_attention // 2
    distances = np.abs(
        np.arange(seq_len, dtype=np.int32)[None, :] - np.arange(seq_len, dtype=np.int32)[:, None]
    )
    disallowed = distances > radius
    disallowed = disallowed[None, None, :, :]
    disallowed = np.broadcast_to(disallowed, probs.shape)

    masked_probs = probs[disallowed]
    if masked_probs.size > 0:
        np.testing.assert_allclose(masked_probs, 0.0, atol=1e-6)

    allowed_mask = np.logical_not(disallowed).astype(probs.dtype)
    row_sums = (probs * allowed_mask).sum(axis=-1)
    expected = np.ones_like(row_sums)
    np.testing.assert_allclose(row_sums, expected, atol=1e-6)


def test_attention_padding_mask_blocks_padded_keys(tiny_config):
    config = tiny_config.copy(global_attn_every_n_layers=1)
    clear_rotary_cache()

    seq_len = 5
    hidden_states = jnp.ones((1, seq_len, config.hidden_size), dtype=jnp.float32)
    attention_mask = jnp.array([[1, 1, 1, 0, 0]], dtype=jnp.int32)

    module = ModernBertAttention(config, layer_idx=0)
    variables = module.init(jr.PRNGKey(4), hidden_states, attention_mask=attention_mask, deterministic=True)
    _, attn_probs = module.apply(
        variables,
        hidden_states,
        attention_mask=attention_mask,
        deterministic=True,
        output_attentions=True,
    )
    assert attn_probs is not None
    probs = np.asarray(attn_probs)

    pad_cols = probs[..., 3:]
    if pad_cols.size > 0:
        np.testing.assert_allclose(pad_cols, 0.0, atol=1e-6)
    row_sums = probs[..., :3].sum(axis=-1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)


def test_attention_dropout_respects_deterministic_flag(tiny_config):
    config = tiny_config.copy(attention_dropout=0.5, global_attn_every_n_layers=1)
    clear_rotary_cache()

    hidden_states = jnp.ones((1, 4, config.hidden_size), dtype=jnp.float32)
    attention_mask = jnp.ones((1, 4), dtype=jnp.int32)

    module = ModernBertAttention(config, layer_idx=0)
    variables = module.init(jr.PRNGKey(5), hidden_states, attention_mask=attention_mask, deterministic=True)

    ctx_det_1, _ = module.apply(
        variables,
        hidden_states,
        attention_mask=attention_mask,
        deterministic=True,
        rngs={"dropout": jr.PRNGKey(6)},
    )
    ctx_det_2, _ = module.apply(
        variables,
        hidden_states,
        attention_mask=attention_mask,
        deterministic=True,
        rngs={"dropout": jr.PRNGKey(7)},
    )
    np.testing.assert_allclose(np.asarray(ctx_det_1), np.asarray(ctx_det_2), atol=1e-7)

    ctx_stoch_1, _ = module.apply(
        variables,
        hidden_states,
        attention_mask=attention_mask,
        deterministic=False,
        rngs={"dropout": jr.PRNGKey(8)},
    )
    ctx_stoch_2, _ = module.apply(
        variables,
        hidden_states,
        attention_mask=attention_mask,
        deterministic=False,
        rngs={"dropout": jr.PRNGKey(9)},
    )
    assert not np.allclose(np.asarray(ctx_stoch_1), np.asarray(ctx_stoch_2))
