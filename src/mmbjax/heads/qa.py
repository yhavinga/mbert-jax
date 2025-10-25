"""Question answering head."""

from __future__ import annotations

from typing import Optional, Tuple

import flax.linen as nn
import jax.numpy as jnp

from ..configuration_mmbbert import MMBertConfig
from ..modeling_mmbbert import ModernBertModel
from ..modeling_outputs import QuestionAnsweringModelOutput
from ..utils import truncated_normal_initializer
from .prediction import ModernBertPredictionHead
from .utils import cross_entropy_with_ignore_index


class ModernBertForQuestionAnswering(nn.Module):
    config: MMBertConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    def setup(self) -> None:
        self.model = ModernBertModel(config=self.config, dtype=self.dtype, param_dtype=self.param_dtype)
        self.head = ModernBertPredictionHead(config=self.config, dtype=self.dtype, param_dtype=self.param_dtype)
        self.dropout = nn.Dropout(rate=self.config.classifier_dropout)
        kernel_init = truncated_normal_initializer(self.config.initializer_range)
        self.classifier = nn.Dense(
            features=self.config.num_labels,
            use_bias=True,
            kernel_init=kernel_init,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )

    def __call__(
        self,
        input_ids: jnp.ndarray,
        *,
        attention_mask: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        sliding_window_mask: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
        start_positions: Optional[jnp.ndarray] = None,
        end_positions: Optional[jnp.ndarray] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> QuestionAnsweringModelOutput | Tuple[jnp.ndarray, ...]:
        output_attentions = (
            self.config.output_attentions if output_attentions is None else output_attentions
        )
        output_hidden_states = (
            self.config.output_hidden_states if output_hidden_states is None else output_hidden_states
        )
        return_dict = self.config.use_return_dict if return_dict is None else return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            sliding_window_mask=sliding_window_mask,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        hidden_states = outputs.last_hidden_state
        hidden_states = self.head(hidden_states)
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        logits = self.classifier(hidden_states)

        start_logits, end_logits = jnp.split(logits, 2, axis=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        loss = None
        if start_positions is not None and end_positions is not None:
            start_positions = start_positions.reshape(-1)
            end_positions = end_positions.reshape(-1)
            start_loss = cross_entropy_with_ignore_index(
                start_logits,
                start_positions,
                ignore_index=-1,
            )
            end_loss = cross_entropy_with_ignore_index(
                end_logits,
                end_positions,
                ignore_index=-1,
            )
            loss = 0.5 * (start_loss + end_loss)

        if not return_dict:
            output = (start_logits, end_logits)
            extras = tuple(
                value
                for value in (outputs.hidden_states, outputs.attentions)
                if value is not None
            )
            output = output + extras
            return ((loss,) + output) if loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = ["ModernBertForQuestionAnswering"]
