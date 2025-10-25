"""ModernBERT encoder stack implemented in Flax/Linen."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import flax.linen as nn
import jax.numpy as jnp

from .configuration_mmbbert import MMBertConfig
from .modules import ModernBertEmbeddings, build_sliding_window_mask
from .modules.layer import ModernBertLayer


@dataclass
class ModernBertModelOutput:
    last_hidden_state: jnp.ndarray
    hidden_states: Optional[Tuple[jnp.ndarray, ...]] = None
    attentions: Optional[Tuple[jnp.ndarray, ...]] = None


class ModernBertEncoder(nn.Module):
    config: MMBertConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    def setup(self) -> None:
        self.layers = [
            ModernBertLayer(
                config=self.config,
                layer_idx=idx,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
            )
            for idx in range(self.config.num_hidden_layers)
        ]
        self.final_layer_norm = nn.LayerNorm(
            epsilon=self.config.norm_eps,
            use_bias=self.config.norm_bias,
            use_scale=True,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        self.has_local_layers = any(
            not self.config.is_global_layer(i) for i in range(self.config.num_hidden_layers)
        )

    def _default_position_ids(self, seq_len: int) -> jnp.ndarray:
        return jnp.arange(seq_len)[None, :]

    def _maybe_sliding_window_mask(
        self,
        attention_mask: jnp.ndarray,
        sliding_window_mask: Optional[jnp.ndarray],
    ) -> Optional[jnp.ndarray]:
        if not self.has_local_layers:
            return None
        if sliding_window_mask is not None:
            return sliding_window_mask
        mask_value = -1e9
        return build_sliding_window_mask(
            attention_mask,
            window_size=self.config.local_attention,
            mask_value=mask_value,
            dtype=jnp.float32,
        )

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        *,
        attention_mask: Optional[jnp.ndarray] = None,
        sliding_window_mask: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
    ) -> ModernBertModelOutput:
        batch, seq_len, _ = hidden_states.shape
        self.config.ensure_can_handle_sequence(seq_len)

        if attention_mask is None:
            attention_mask = jnp.ones((batch, seq_len), dtype=jnp.float32)
        else:
            attention_mask = attention_mask.astype(jnp.float32)

        if position_ids is None:
            position_ids = self._default_position_ids(seq_len)
        position_ids = position_ids.astype("int32")

        sliding_window_mask = self._maybe_sliding_window_mask(attention_mask, sliding_window_mask)

        all_hidden_states: Tuple[jnp.ndarray, ...] | None = () if output_hidden_states else None
        all_attentions: Tuple[jnp.ndarray, ...] | None = () if output_attentions else None

        for layer in self.layers:
            if output_hidden_states and all_hidden_states is not None:
                all_hidden_states = all_hidden_states + (hidden_states,)

            hidden_states, attn = layer(
                hidden_states,
                attention_mask=attention_mask,
                sliding_window_mask=sliding_window_mask,
                position_ids=position_ids,
                deterministic=deterministic,
                output_attentions=output_attentions,
            )
            if output_attentions and attn is not None and all_attentions is not None:
                all_attentions = all_attentions + (attn,)

        if output_hidden_states and all_hidden_states is not None:
            all_hidden_states = all_hidden_states + (hidden_states,)

        hidden_states = self.final_layer_norm(hidden_states)

        return ModernBertModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )


class ModernBertModel(nn.Module):
    config: MMBertConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    def setup(self) -> None:
        self.embeddings = ModernBertEmbeddings(
            config=self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        self.encoder = ModernBertEncoder(
            config=self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )

    def __call__(
        self,
        input_ids: Optional[jnp.ndarray] = None,
        *,
        inputs_embeds: Optional[jnp.ndarray] = None,
        attention_mask: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        sliding_window_mask: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
    ) -> ModernBertModelOutput:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of `input_ids` or `inputs_embeds`.")

        if inputs_embeds is None and input_ids is not None:
            hidden_states = self.embeddings(input_ids=input_ids, deterministic=deterministic)
            batch, seq_len = input_ids.shape
        else:
            hidden_states = self.embeddings(inputs_embeds=inputs_embeds, deterministic=deterministic)
            batch, seq_len = hidden_states.shape[:2]

        if attention_mask is None:
            attention_mask = jnp.ones((batch, seq_len), dtype=jnp.float32)
        else:
            attention_mask = attention_mask.astype(jnp.float32)

        if position_ids is None:
            position_ids = jnp.arange(seq_len)[None, :]

        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            sliding_window_mask=sliding_window_mask,
            position_ids=position_ids,
            deterministic=deterministic,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )
        return encoder_outputs


__all__ = ["ModernBertModel", "ModernBertEncoder", "ModernBertModelOutput"]
