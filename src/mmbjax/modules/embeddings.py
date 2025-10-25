"""Embedding stack for ModernBERT."""

from __future__ import annotations

from typing import Optional

import flax.linen as nn
import jax.numpy as jnp

from ..configuration_mmbbert import MMBertConfig
from ..utils import truncated_normal_initializer


class ModernBertEmbeddings(nn.Module):
    """Token embeddings + layer norm + dropout (no token types or position embeds)."""

    config: MMBertConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    def setup(self) -> None:
        embedding_init = truncated_normal_initializer(self.config.initializer_range)
        self.word_embeddings = nn.Embed(
            num_embeddings=self.config.vocab_size,
            features=self.config.hidden_size,
            embedding_init=embedding_init,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        self.layer_norm = nn.LayerNorm(
            epsilon=self.config.norm_eps,
            use_bias=self.config.norm_bias,
            use_scale=True,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        self.dropout = nn.Dropout(rate=self.config.embedding_dropout)

    def __call__(
        self,
        input_ids: Optional[jnp.ndarray] = None,
        inputs_embeds: Optional[jnp.ndarray] = None,
        *,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Pass either `input_ids` or `inputs_embeds`, but not both.")

        if inputs_embeds is None:
            if input_ids.ndim != 2:
                raise ValueError("`input_ids` should be [batch, seq_length].")
            embeddings = self.word_embeddings(input_ids.astype("int32"))
        else:
            embeddings = inputs_embeds

        embeddings = embeddings.astype(self.dtype)
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings, deterministic=deterministic)
        return embeddings


__all__ = ["ModernBertEmbeddings"]
