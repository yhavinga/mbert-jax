"""Rotary positional embedding helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import jax.numpy as jnp


def rotate_half(x: jnp.ndarray) -> jnp.ndarray:
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate((-x2, x1), axis=-1)


def build_rotary_cache(
    max_seq: int,
    dim: int,
    theta: float,
    dtype: jnp.dtype = jnp.float32,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    if dim % 2 != 0:
        raise ValueError(f"RoPE dimension must be even (got {dim}).")
    work_dtype = jnp.float32 if dtype in (jnp.float16, jnp.bfloat16) else dtype
    positions = jnp.arange(max_seq, dtype=work_dtype)
    idx = jnp.arange(0, dim, 2, dtype=work_dtype)
    inv_freq = 1.0 / (jnp.power(theta, idx / dim))
    angles = positions[:, None] * inv_freq[None, :]
    # Stack (concatenate) instead of interleaving (repeat) to match PyTorch
    emb = jnp.concatenate([angles, angles], axis=-1)
    cos = jnp.cos(emb)
    sin = jnp.sin(emb)
    return cos.astype(dtype), sin.astype(dtype)


@dataclass
class RotaryCache:
    head_dim: int
    theta: float
    dtype: jnp.dtype = jnp.float32
    max_seq: int = 0
    _cos: jnp.ndarray | None = None
    _sin: jnp.ndarray | None = None

    def get(self, seq_len: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
        target = max(seq_len, 1)
        if self._cos is None or target > self.max_seq:
            self.max_seq = target
            self._cos, self._sin = build_rotary_cache(target, self.head_dim, self.theta, self.dtype)
        assert self._cos is not None and self._sin is not None
        return self._cos[:seq_len], self._sin[:seq_len]


_CACHE: Dict[Tuple[int, float, str], RotaryCache] = {}


def get_rotary_cache(
    head_dim: int,
    theta: float,
    dtype: jnp.dtype = jnp.float32,
) -> RotaryCache:
    key = (head_dim, float(theta), str(jnp.dtype(dtype)))
    if key not in _CACHE:
        _CACHE[key] = RotaryCache(head_dim=head_dim, theta=theta, dtype=dtype)
    return _CACHE[key]


def clear_rotary_cache() -> None:
    _CACHE.clear()


def apply_rotary_emb(
    q: jnp.ndarray,
    k: jnp.ndarray,
    cos: jnp.ndarray,
    sin: jnp.ndarray,
    position_ids: jnp.ndarray | None = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Apply rotary embeddings to query/key tensors."""

    if position_ids is not None:
        if position_ids.ndim == 1:
            position_ids = position_ids[None, :]
        gather_idx = position_ids.astype("int32")
        cos = jnp.take(cos, gather_idx, axis=0)
        sin = jnp.take(sin, gather_idx, axis=0)
        cos = cos[:, None, :, :]
        sin = sin[:, None, :, :]
    else:
        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]

    # Keep rotation in float32 for precision - BF16 causes position encoding errors
    # See: https://openreview.net/forum?id=gwXfZ3xkUq (RoPE precision issues in BF16)
    original_dtype = q.dtype
    q_f32 = q.astype(jnp.float32)
    k_f32 = k.astype(jnp.float32)
    # cos/sin are already float32 from build_rotary_cache
    
    q_out = (q_f32 * cos) + (rotate_half(q_f32) * sin)
    k_out = (k_f32 * cos) + (rotate_half(k_f32) * sin)
    
    return q_out.astype(original_dtype), k_out.astype(original_dtype)


__all__ = [
    "RotaryCache",
    "apply_rotary_emb",
    "build_rotary_cache",
    "clear_rotary_cache",
    "get_rotary_cache",
    "rotate_half",
]
