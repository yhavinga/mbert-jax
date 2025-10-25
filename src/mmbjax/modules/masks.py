"""Mask helpers for ModernBERT attention."""

from __future__ import annotations

import jax.numpy as jnp


def build_sliding_window_mask(
    attention_mask: jnp.ndarray,
    window_size: int,
    *,
    mask_value: float | None = None,
    dtype: jnp.dtype | None = None,
) -> jnp.ndarray:
    """Return an additive mask that enforces the ±window_size/2 attention span."""

    if attention_mask.ndim != 2:
        raise ValueError("`attention_mask` must be shaped [batch, seq_length].")
    if window_size <= 0 or window_size % 2 != 0:
        raise ValueError("`window_size` must be a positive, even integer.")

    batch_size, seq_len = attention_mask.shape
    radius = window_size // 2
    dtype = dtype or jnp.result_type(attention_mask.dtype, jnp.float32)
    mask_value = -1e9 if mask_value is None else mask_value
    neg = jnp.array(mask_value, dtype=dtype)

    positions = jnp.arange(seq_len)
    distances = positions[None, :] - positions[:, None]
    window = jnp.where(jnp.abs(distances) <= radius, 0.0, neg).astype(dtype)
    window = jnp.broadcast_to(window, (batch_size, seq_len, seq_len))
    window = window[:, None, :, :]

    key_pad = (1.0 - attention_mask.astype(dtype)) * neg
    key_pad = key_pad[:, None, None, :]
    return window + key_pad


__all__ = ["build_sliding_window_mask"]
