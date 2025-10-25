"""Initializer helpers for ModernBERT modules."""

from __future__ import annotations

from typing import Callable

import flax.linen as nn
import jax.numpy as jnp


Initializer = Callable[..., jnp.ndarray]


def truncated_normal_initializer(stddev: float) -> Initializer:
    """Return a truncated normal initializer with the provided stddev."""

    def init(key, shape, dtype=jnp.float32):
        return nn.initializers.truncated_normal(stddev=stddev, dtype=dtype)(key, shape)

    return init


def scaled_truncated_normal(stddev: float, scale: float) -> Initializer:
    """Helper when ModernBERT scales certain projection outputs."""
    return truncated_normal_initializer(stddev * scale)


__all__ = ["Initializer", "truncated_normal_initializer", "scaled_truncated_normal"]
