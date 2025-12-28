"""Transformer layer for ModernBERT."""

from __future__ import annotations

from typing import Optional, Tuple

import flax.linen as nn
import jax.numpy as jnp

from ..configuration_mmbbert import MMBertConfig
from .attention import ModernBertAttention
from .mlp import ModernBertMLP


class ModernBertLayer(nn.Module):
    config: MMBertConfig
    layer_idx: int
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    def setup(self) -> None:
        # Keep LayerNorm in FP32 for numerical stability in mixed precision
        self.attention_norm = nn.LayerNorm(
            epsilon=self.config.norm_eps,
            use_bias=self.config.norm_bias,
            use_scale=True,
            dtype=jnp.float32,  # Always FP32 for stability
            param_dtype=self.param_dtype,
        )
        self.attention = ModernBertAttention(
            config=self.config,
            layer_idx=self.layer_idx,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        # Keep LayerNorm in FP32 for numerical stability in mixed precision
        self.mlp_norm = nn.LayerNorm(
            epsilon=self.config.norm_eps,
            use_bias=self.config.norm_bias,
            use_scale=True,
            dtype=jnp.float32,  # Always FP32 for stability
            param_dtype=self.param_dtype,
        )
        self.mlp = ModernBertMLP(
            config=self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )

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
        attn_input = hidden_states if self.layer_idx == 0 else self.attention_norm(hidden_states)
        attn_output, attn_weights = self.attention(
            attn_input,
            attention_mask=attention_mask,
            sliding_window_mask=sliding_window_mask,
            position_ids=position_ids,
            segment_ids=segment_ids,
            deterministic=deterministic,
            output_attentions=output_attentions,
        )
        hidden_states = hidden_states + attn_output
        mlp_input = self.mlp_norm(hidden_states)
        mlp_output = self.mlp(mlp_input, deterministic=deterministic)
        hidden_states = hidden_states + mlp_output
        return hidden_states, attn_weights


__all__ = ["ModernBertLayer"]
