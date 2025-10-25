"""Dataclass outputs mirroring Hugging Face's modeling outputs but for JAX arrays."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import jax.numpy as jnp


@dataclass
class MaskedLMOutput:
    loss: Optional[jnp.ndarray] = None
    logits: Optional[jnp.ndarray] = None
    hidden_states: Optional[Tuple[jnp.ndarray, ...]] = None
    attentions: Optional[Tuple[jnp.ndarray, ...]] = None


@dataclass
class SequenceClassifierOutput:
    loss: Optional[jnp.ndarray] = None
    logits: Optional[jnp.ndarray] = None
    hidden_states: Optional[Tuple[jnp.ndarray, ...]] = None
    attentions: Optional[Tuple[jnp.ndarray, ...]] = None


@dataclass
class TokenClassifierOutput:
    loss: Optional[jnp.ndarray] = None
    logits: Optional[jnp.ndarray] = None
    hidden_states: Optional[Tuple[jnp.ndarray, ...]] = None
    attentions: Optional[Tuple[jnp.ndarray, ...]] = None


@dataclass
class QuestionAnsweringModelOutput:
    loss: Optional[jnp.ndarray] = None
    start_logits: Optional[jnp.ndarray] = None
    end_logits: Optional[jnp.ndarray] = None
    hidden_states: Optional[Tuple[jnp.ndarray, ...]] = None
    attentions: Optional[Tuple[jnp.ndarray, ...]] = None


@dataclass
class MultipleChoiceModelOutput:
    loss: Optional[jnp.ndarray] = None
    logits: Optional[jnp.ndarray] = None
    hidden_states: Optional[Tuple[jnp.ndarray, ...]] = None
    attentions: Optional[Tuple[jnp.ndarray, ...]] = None


__all__ = [
    "MaskedLMOutput",
    "SequenceClassifierOutput",
    "TokenClassifierOutput",
    "QuestionAnsweringModelOutput",
    "MultipleChoiceModelOutput",
]
