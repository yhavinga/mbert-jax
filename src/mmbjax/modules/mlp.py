"""GeGLU/MLP block."""

from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp

from ..configuration_mmbbert import MMBertConfig
from ..utils import get_activation, truncated_normal_initializer


class ModernBertMLP(nn.Module):
    config: MMBertConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    def setup(self) -> None:
        kernel_init = truncated_normal_initializer(self.config.initializer_range)
        self.wi = nn.Dense(
            features=2 * self.config.intermediate_size,
            use_bias=self.config.mlp_bias,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=kernel_init,
        )
        self.wo = nn.Dense(
            features=self.config.hidden_size,
            use_bias=self.config.mlp_bias,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=kernel_init,
        )
        self.dropout = nn.Dropout(rate=self.config.mlp_dropout)
        self.activation = get_activation(self.config.hidden_activation)

    def __call__(self, hidden_states: jnp.ndarray, *, deterministic: bool = True) -> jnp.ndarray:
        hidden_states = hidden_states.astype(self.dtype)
        fused = self.wi(hidden_states)
        act_input, gate = jnp.split(fused, 2, axis=-1)
        act = self.activation(act_input)
        # Keep gate multiply in float32 for precision - small gate values after GeLU
        # lose precision in BF16 (7-bit mantissa)
        act = (act.astype(jnp.float32) * gate.astype(jnp.float32)).astype(self.dtype)
        act = self.dropout(act, deterministic=deterministic)
        return self.wo(act)


__all__ = ["ModernBertMLP"]
