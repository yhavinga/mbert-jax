"""Masked language modeling head."""

from __future__ import annotations

from typing import Optional, Tuple

import flax.linen as nn
import jax.numpy as jnp

from ..configuration_mmbbert import MMBertConfig
from ..modeling_mmbbert import ModernBertModel
from ..modeling_outputs import MaskedLMOutput
from .prediction import ModernBertPredictionHead
from .utils import cross_entropy_with_ignore_index


class ModernBertLMHead(nn.Module):
    config: MMBertConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    def setup(self) -> None:
        if self.config.decoder_bias:
            self.bias = self.param(
                "bias",
                nn.initializers.zeros,
                (self.config.vocab_size,),
                self.param_dtype,
            )

    def __call__(self, hidden_states: jnp.ndarray, embedding_kernel: jnp.ndarray) -> jnp.ndarray:
        logits = jnp.einsum("bsh,vh->bsv", hidden_states, embedding_kernel)
        if self.config.decoder_bias:
            logits = logits + self.bias
        return logits


class ModernBertForMaskedLM(nn.Module):
    config: MMBertConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    def setup(self) -> None:
        self.model = ModernBertModel(
            config=self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        self.prediction_head = ModernBertPredictionHead(
            config=self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        self.decoder = ModernBertLMHead(
            config=self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )

    def _get_embedding_kernel(self) -> jnp.ndarray:
        params = self.variables.get("params")
        if params is None:
            raise ValueError("Parameters must be initialized before calling the LM head.")
        return params["model"]["embeddings"]["word_embeddings"]["embedding"]

    def __call__(
        self,
        input_ids: Optional[jnp.ndarray] = None,
        *,
        attention_mask: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        inputs_embeds: Optional[jnp.ndarray] = None,
        labels: Optional[jnp.ndarray] = None,
        sliding_window_mask: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> MaskedLMOutput | Tuple[jnp.ndarray, ...]:
        output_attentions = (
            self.config.output_attentions if output_attentions is None else output_attentions
        )
        output_hidden_states = (
            self.config.output_hidden_states if output_hidden_states is None else output_hidden_states
        )
        return_dict = self.config.use_return_dict if return_dict is None else return_dict

        outputs = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            sliding_window_mask=sliding_window_mask,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        hidden_states = outputs.last_hidden_state

        hidden_states = self.prediction_head(hidden_states)
        embedding_kernel = self._get_embedding_kernel()
        logits = self.decoder(hidden_states, embedding_kernel)

        loss = None
        if labels is not None:
            flat_logits = logits.reshape(-1, self.config.vocab_size)
            flat_labels = labels.reshape(-1)
            loss = cross_entropy_with_ignore_index(
                flat_logits,
                flat_labels,
                ignore_index=self.config.sparse_pred_ignore_index,
            )

        if not return_dict:
            output = (logits,)
            extras = tuple(
                value
                for value in (outputs.hidden_states, outputs.attentions)
                if value is not None
            )
            output = output + extras
            return ((loss,) + output) if loss is not None else output

        return MaskedLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = ["ModernBertForMaskedLM"]
