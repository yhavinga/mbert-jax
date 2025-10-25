"""Activation helpers."""

from __future__ import annotations

import math
from typing import Callable, Dict

import functools
import jax.nn as jnn
import jax.numpy as jnp

Activation = Callable[[jnp.ndarray], jnp.ndarray]


def _gelu(x: jnp.ndarray) -> jnp.ndarray:
    return 0.5 * x * (1.0 + jnp.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))


def _gelu_fast(x: jnp.ndarray) -> jnp.ndarray:
    return 0.5 * x * (1.0 + jnp.tanh(0.79788456 * (x + 0.044715 * x**3)))


def _gelu_approx(x: jnp.ndarray) -> jnp.ndarray:
    return x * jnp.reciprocal(1.0 + jnp.exp(-1.702 * x))


def _silu(x: jnp.ndarray) -> jnp.ndarray:
    return x * jnp.reciprocal(1.0 + jnp.exp(-x))


ACT2FN: Dict[str, Activation] = {
    "relu": lambda x: jnp.maximum(x, 0.0),
    "gelu": functools.partial(jnn.gelu, approximate=False),
    "gelu_fast": _gelu_fast,
    "gelu_new": _gelu,
    "approximate_gelu": _gelu_approx,
    "silu": _silu,
    "swish": _silu,
    "tanh": jnp.tanh,
}


def get_activation(name_or_callable: str | Activation) -> Activation:
    if callable(name_or_callable):
        return name_or_callable  # type: ignore[return-value]
    name = name_or_callable.lower()
    if name not in ACT2FN:
        raise ValueError(f"Unknown activation '{name}'. Supported: {sorted(ACT2FN)}")
    return ACT2FN[name]


__all__ = ["Activation", "ACT2FN", "get_activation"]
