"""Classifier/prediction head shared across tasks."""

from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp

from ..configuration_mmbbert import MMBertConfig
from ..utils import get_activation, truncated_normal_initializer


class ModernBertPredictionHead(nn.Module):
    config: MMBertConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    def setup(self) -> None:
        kernel_init = truncated_normal_initializer(self.config.initializer_range)
        self.dense = nn.Dense(
            features=self.config.hidden_size,
            use_bias=self.config.classifier_bias,
            kernel_init=kernel_init,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        self.activation = get_activation(self.config.classifier_activation)
        self.layer_norm = nn.LayerNorm(
            epsilon=self.config.norm_eps,
            use_bias=self.config.norm_bias,
            use_scale=True,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )

    def __call__(self, hidden_states: jnp.ndarray) -> jnp.ndarray:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        return self.layer_norm(hidden_states)


__all__ = ["ModernBertPredictionHead"]
