"""Classification-style task heads."""

from __future__ import annotations

from typing import Optional, Tuple

import flax.linen as nn
import jax.numpy as jnp

from ..configuration_mmbbert import MMBertConfig
from ..modeling_mmbbert import ModernBertModel
from ..modeling_outputs import (
    MultipleChoiceModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ..utils import truncated_normal_initializer
from .prediction import ModernBertPredictionHead
from .utils import cross_entropy_loss, cross_entropy_with_ignore_index, mse_loss, pool_hidden_states


class ModernBertForSequenceClassification(nn.Module):
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
    ) -> SequenceClassifierOutput | Tuple[jnp.ndarray, ...]:
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
        if attention_mask is None:
            if input_ids is not None:
                attention_mask = jnp.ones_like(input_ids, dtype=jnp.float32)
            elif inputs_embeds is not None:
                attention_mask = jnp.ones(hidden_states.shape[:2], dtype=jnp.float32)

        pooled = pool_hidden_states(
            hidden_states,
            attention_mask,
            pooling=self.config.classifier_pooling,
        )
        pooled = self.head(pooled)
        pooled = self.dropout(pooled, deterministic=deterministic)
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.config.num_labels == 1:
                    self.config.problem_type = "regression"
                else:
                    self.config.problem_type = "single_label_classification"

            if self.config.problem_type == "regression":
                loss = mse_loss(logits, labels.astype(self.dtype))
            else:
                loss = cross_entropy_loss(logits, labels.astype(jnp.int32))

        if not return_dict:
            output = (logits,)
            extras = tuple(
                value
                for value in (outputs.hidden_states, outputs.attentions)
                if value is not None
            )
            output = output + extras
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class ModernBertForTokenClassification(nn.Module):
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
    ) -> TokenClassifierOutput | Tuple[jnp.ndarray, ...]:
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
        hidden_states = self.head(hidden_states)
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        logits = self.classifier(hidden_states)

        loss = None
        if labels is not None:
            flat_logits = logits.reshape(-1, self.config.num_labels)
            flat_labels = labels.reshape(-1)
            loss = cross_entropy_with_ignore_index(flat_logits, flat_labels)

        if not return_dict:
            output = (logits,)
            extras = tuple(
                value
                for value in (outputs.hidden_states, outputs.attentions)
                if value is not None
            )
            output = output + extras
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class ModernBertForMultipleChoice(nn.Module):
    config: MMBertConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    def setup(self) -> None:
        self.model = ModernBertModel(config=self.config, dtype=self.dtype, param_dtype=self.param_dtype)
        self.head = ModernBertPredictionHead(config=self.config, dtype=self.dtype, param_dtype=self.param_dtype)
        self.dropout = nn.Dropout(rate=self.config.classifier_dropout)
        kernel_init = truncated_normal_initializer(self.config.initializer_range)
        self.classifier = nn.Dense(
            features=1,
            use_bias=True,
            kernel_init=kernel_init,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )

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
    ) -> MultipleChoiceModelOutput | Tuple[jnp.ndarray, ...]:
        if input_ids is None and inputs_embeds is None:
            raise ValueError("Provide either `input_ids` or `inputs_embeds` for multiple choice tasks.")

        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        def _reshape(tensor: Optional[jnp.ndarray]) -> Optional[jnp.ndarray]:
            if tensor is None:
                return None
            new_shape = (-1,) + tensor.shape[2:]
            return tensor.reshape(new_shape)

        flat_input_ids = _reshape(input_ids)
        flat_mask = _reshape(attention_mask)
        flat_position_ids = _reshape(position_ids)
        flat_embeds = (
            inputs_embeds.reshape((-1,) + inputs_embeds.shape[2:]) if inputs_embeds is not None else None
        )

        output_attentions = (
            self.config.output_attentions if output_attentions is None else output_attentions
        )
        output_hidden_states = (
            self.config.output_hidden_states if output_hidden_states is None else output_hidden_states
        )
        return_dict = self.config.use_return_dict if return_dict is None else return_dict

        outputs = self.model(
            input_ids=flat_input_ids,
            inputs_embeds=flat_embeds,
            attention_mask=flat_mask,
            position_ids=flat_position_ids,
            sliding_window_mask=sliding_window_mask,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        hidden_states = outputs.last_hidden_state
        pooled = pool_hidden_states(
            hidden_states,
            flat_mask,
            pooling=self.config.classifier_pooling,
            prefer_first_nonpad=True,
        )
        pooled = self.head(pooled)
        pooled = self.dropout(pooled, deterministic=deterministic)
        logits = self.classifier(pooled).reshape(-1, num_choices)

        loss = None
        if labels is not None:
            loss = cross_entropy_loss(logits, labels)

        if not return_dict:
            output = (logits,)
            extras = tuple(
                value
                for value in (outputs.hidden_states, outputs.attentions)
                if value is not None
            )
            output = output + extras
            return ((loss,) + output) if loss is not None else output

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = [
    "ModernBertForSequenceClassification",
    "ModernBertForTokenClassification",
    "ModernBertForMultipleChoice",
]
