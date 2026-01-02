"""Helper utilities shared across ModernBERT heads."""

from __future__ import annotations

from typing import Optional

import jax.nn as jnn
import jax.numpy as jnp


def cross_entropy_loss(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    labels = labels.astype(jnp.int32)
    log_probs = jnn.log_softmax(logits, axis=-1)
    gathered = jnp.take_along_axis(log_probs, labels[..., None], axis=-1).squeeze(-1)
    return -jnp.mean(gathered)


def cross_entropy_with_ignore_index(
    logits: jnp.ndarray,
    labels: jnp.ndarray,
    *,
    ignore_index: int = -100,
) -> jnp.ndarray:
    # Force float32 for precision - log_softmax over 32K vocab loses precision in BF16
    logits = logits.astype(jnp.float32)
    mask = (labels != ignore_index).astype(jnp.float32)
    safe_labels = jnp.where(mask > 0, labels, 0)
    log_probs = jnn.log_softmax(logits, axis=-1)
    gathered = jnp.take_along_axis(log_probs, safe_labels[..., None], axis=-1).squeeze(-1)
    numerator = -(gathered * mask).sum()
    denom = jnp.maximum(mask.sum(), 1.0)
    return numerator / denom


def mse_loss(predictions: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    diff = predictions.squeeze(-1) - targets.squeeze(-1)
    return jnp.mean(jnp.square(diff))


def pool_hidden_states(
    hidden_states: jnp.ndarray,
    attention_mask: Optional[jnp.ndarray],
    *,
    pooling: str,
    prefer_first_nonpad: bool = False,
) -> jnp.ndarray:
    if pooling == "cls":
        if prefer_first_nonpad and attention_mask is not None:
            indices = jnp.argmax(attention_mask, axis=-1)
            gathered = jnp.take_along_axis(hidden_states, indices[:, None, None], axis=1)
            return gathered.squeeze(axis=1)
        return hidden_states[:, 0]

    if attention_mask is None:
        return hidden_states.mean(axis=1)

    mask = attention_mask[..., None].astype(hidden_states.dtype)
    summed = (hidden_states * mask).sum(axis=1)
    denom = jnp.clip(mask.sum(axis=1), a_min=1.0)
    return summed / denom


def mean_pool_last_dim(hidden_states: jnp.ndarray, attention_mask: jnp.ndarray) -> jnp.ndarray:
    mask = attention_mask[..., None].astype(hidden_states.dtype)
    summed = (hidden_states * mask).sum(axis=1)
    denom = jnp.clip(mask.sum(axis=1), a_min=1.0)
    return summed / denom


__all__ = [
    "cross_entropy_loss",
    "cross_entropy_with_ignore_index",
    "mse_loss",
    "pool_hidden_states",
    "mean_pool_last_dim",
]
