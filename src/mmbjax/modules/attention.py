"""Multi-head attention with ModernBERT specifics."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import flax.linen as nn
import jax.nn as jnn
import jax.numpy as jnp

from ..configuration_mmbbert import MMBertConfig
from ..utils import truncated_normal_initializer
from .masks import build_sliding_window_mask
from .rope import apply_rotary_emb, get_rotary_cache


MaskArray = Optional[jnp.ndarray]


def _ensure_attention_mask(
    attention_mask: Optional[jnp.ndarray],
    batch: int,
    seq_len: int,
    dtype: jnp.dtype,
) -> jnp.ndarray:
    if attention_mask is None:
        return jnp.ones((batch, seq_len), dtype=dtype)
    if attention_mask.ndim != 2:
        raise ValueError("`attention_mask` must be [batch, seq_len].")
    return attention_mask.astype(dtype)


class ModernBertAttention(nn.Module):
    config: MMBertConfig
    layer_idx: int
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    def setup(self) -> None:
        self.num_heads = self.config.num_attention_heads
        self.head_dim = self.config.head_dim
        self.scale = 1.0 / math.sqrt(self.head_dim)
        kernel_init = truncated_normal_initializer(self.config.initializer_range)
        self.qkv = nn.Dense(
            features=3 * self.config.hidden_size,
            use_bias=self.config.attention_bias,
            kernel_init=kernel_init,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        self.out = nn.Dense(
            features=self.config.hidden_size,
            use_bias=self.config.attention_bias,
            kernel_init=kernel_init,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        self.attn_dropout = nn.Dropout(rate=self.config.attention_dropout)
        self.out_dropout = nn.Dropout(rate=self.config.attention_dropout)
        self.is_global = self.config.is_global_layer(self.layer_idx)
        local_theta = (
            self.config.local_rope_theta
            if self.config.local_rope_theta is not None
            else self.config.global_rope_theta
        )
        self.theta = self.config.global_rope_theta if self.is_global else local_theta

    def _compute_padding_mask(
        self, attention_mask: jnp.ndarray, mask_value: jnp.ndarray
    ) -> Optional[jnp.ndarray]:
        if attention_mask is None:
            return None
        padding = (1.0 - attention_mask) * mask_value
        return padding[:, None, None, :]

    def _maybe_sliding_window_mask(
        self,
        attention_mask: jnp.ndarray,
        sliding_window_mask: MaskArray,
        mask_value: jnp.ndarray,
    ) -> MaskArray:
        if self.is_global:
            return None
        if sliding_window_mask is not None:
            return sliding_window_mask
        window = build_sliding_window_mask(
            attention_mask,
            window_size=self.config.local_attention,
            mask_value=float(mask_value),
            dtype=self.dtype,
        )
        return window

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        *,
        attention_mask: Optional[jnp.ndarray] = None,
        sliding_window_mask: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        segment_ids: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
        output_attentions: bool = False,
    ) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
        batch, seq_len, _ = hidden_states.shape
        mask_dtype = hidden_states.dtype if hidden_states.dtype in (jnp.float32, jnp.float64) else jnp.float32
        attention_mask = _ensure_attention_mask(attention_mask, batch, seq_len, mask_dtype)
        # Use minimum float value (like PyTorch's -inf) instead of -1e9
        mask_value = jnp.array(jnp.finfo(mask_dtype).min, dtype=mask_dtype)

        # Build block-diagonal mask from segment_ids for packed sequences
        # When sequences are packed, tokens from different documents must not attend to each other
        segment_mask = None
        if segment_ids is not None:
            # segment_ids shape: [batch, seq_len]
            # Create mask where tokens only attend within their segment (document)
            # Output shape: [batch, num_heads, seq_len, seq_len] for broadcasting with attn_scores
            seg_i = segment_ids[:, None, :, None]  # [b, 1, s, 1]
            seg_j = segment_ids[:, None, None, :]  # [b, 1, 1, s]
            segment_mask = (seg_i == seg_j).astype(mask_dtype)  # 1 where same segment, 0 otherwise
            # Convert to additive mask (0 -> keep, 1 -> mask out)
            segment_mask = (1.0 - segment_mask) * mask_value

        qkv = self.qkv(hidden_states)
        # PyTorch packs qkv as [batch, seq, 3, heads, head_dim]; keep that ordering
        qkv = qkv.reshape(batch, seq_len, 3, self.num_heads, self.head_dim)
        qkv = jnp.transpose(qkv, (0, 3, 1, 2, 4))  # -> [batch, heads, seq, 3, head_dim]
        query = qkv[:, :, :, 0, :]
        key = qkv[:, :, :, 1, :]
        value = qkv[:, :, :, 2, :]

        cache = get_rotary_cache(self.head_dim, self.theta, dtype=self.dtype)
        cos, sin = cache.get(self.config.max_position_embeddings)
        cos_slice = cos if position_ids is not None else cos[:seq_len]
        sin_slice = sin if position_ids is not None else sin[:seq_len]
        query, key = apply_rotary_emb(query, key, cos_slice, sin_slice, position_ids=position_ids)

        attn_scores = jnp.einsum("bhqd,bhkd->bhqk", query, key) * self.scale
        attn_scores = attn_scores.astype(mask_dtype)

        padding_mask = self._compute_padding_mask(attention_mask, mask_value)
        if padding_mask is not None:
            attn_scores = attn_scores + padding_mask

        local_mask = self._maybe_sliding_window_mask(attention_mask, sliding_window_mask, mask_value)
        if local_mask is not None:
            attn_scores = attn_scores + local_mask.astype(mask_dtype)

        # Apply block-diagonal segment mask to prevent cross-document attention
        if segment_mask is not None:
            attn_scores = attn_scores + segment_mask

        attn_probs = jnn.softmax(attn_scores, axis=-1).astype(self.dtype)
        attn_probs = self.attn_dropout(attn_probs, deterministic=deterministic)
        context = jnp.einsum("bhqk,bhkd->bhqd", attn_probs, value)
        context = jnp.transpose(context, (0, 2, 1, 3))
        context = context.reshape(batch, seq_len, self.config.hidden_size)
        context = self.out(context)
        context = self.out_dropout(context, deterministic=deterministic)

        return context, (attn_probs if output_attentions else None)


__all__ = ["ModernBertAttention"]
