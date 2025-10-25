import numpy as np
import pytest

import jax.numpy as jnp

from src.mmbjax.heads.utils import (
    cross_entropy_loss,
    cross_entropy_with_ignore_index,
    pool_hidden_states,
)


def test_pool_hidden_states_cls_prefers_first_nonpad():
    hidden = jnp.arange(2 * 4 * 3, dtype=jnp.float32).reshape(2, 4, 3)
    attention_mask = jnp.array(
        [
            [0, 0, 1, 1],
            [1, 1, 0, 0],
        ],
        dtype=jnp.int32,
    )
    pooled = pool_hidden_states(hidden, attention_mask, pooling="cls", prefer_first_nonpad=True)
    expected = jnp.stack([hidden[0, 2], hidden[1, 0]])
    np.testing.assert_allclose(np.asarray(pooled), np.asarray(expected), rtol=1e-6)


def test_pool_hidden_states_mean_matches_masked_average():
    hidden = jnp.array(
        [
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[2.0, 0.0], [4.0, 2.0], [0.0, 0.0]],
        ]
    )
    mask = jnp.array([[1, 1, 0], [1, 1, 1]], dtype=jnp.float32)
    pooled = pool_hidden_states(hidden, mask, pooling="mean")

    manual = []
    for row, m in zip(hidden, mask):
        weights = m[:, None]
        denom = np.clip(np.sum(weights), a_min=1.0, a_max=None)
        manual.append(np.sum(np.asarray(row) * np.asarray(weights), axis=0) / denom)
    manual = np.stack(manual)
    np.testing.assert_allclose(np.asarray(pooled), manual, rtol=1e-6)


def test_cross_entropy_loss_matches_manual():
    logits = jnp.array([[2.0, 0.0], [0.0, 2.0]], dtype=jnp.float32)
    labels = jnp.array([0, 1], dtype=jnp.int32)
    loss = cross_entropy_loss(logits, labels)
    log_probs = jnp.log(jnp.array([np.exp(2) / (np.exp(2) + 1), np.exp(2) / (np.exp(2) + 1)], dtype=jnp.float32))
    expected = -jnp.mean(log_probs)
    np.testing.assert_allclose(np.asarray(loss), np.asarray(expected), rtol=1e-6)


def test_cross_entropy_with_ignore_index_masks_positions():
    logits = jnp.array(
        [
            [10.0, 0.0],
            [0.0, 10.0],
            [5.0, 5.0],
        ],
        dtype=jnp.float32,
    )
    labels = jnp.array([0, 1, -100], dtype=jnp.int32)
    loss = cross_entropy_with_ignore_index(logits, labels)

    keep = logits[:2]
    keep_labels = labels[:2]
    log_probs = jnp.log(jnp.exp(keep) / jnp.sum(jnp.exp(keep), axis=-1, keepdims=True))
    manual = -jnp.mean(jnp.take_along_axis(log_probs, keep_labels[:, None], axis=-1))
    np.testing.assert_allclose(np.asarray(loss), np.asarray(manual), rtol=1e-4, atol=1e-8)
