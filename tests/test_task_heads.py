import numpy as np

import jax.numpy as jnp
import jax.random as jr

from src.mmbjax.heads.classification import (
    ModernBertForMultipleChoice,
    ModernBertForSequenceClassification,
    ModernBertForTokenClassification,
)
from src.mmbjax.heads.qa import ModernBertForQuestionAnswering
from src.mmbjax.heads.utils import cross_entropy_loss, cross_entropy_with_ignore_index


def test_sequence_classification_logits_and_loss_path(tiny_config):
    config = tiny_config.copy(num_labels=3, classifier_pooling="mean")
    model = ModernBertForSequenceClassification(config)

    input_ids = jnp.array(
        [
            [1, 2, 3, 4],
            [5, 6, 7, 0],
        ],
        dtype=jnp.int32,
    )
    attention_mask = jnp.array(
        [
            [1, 1, 1, 1],
            [1, 1, 1, 0],
        ],
        dtype=jnp.int32,
    )
    labels = jnp.array([0, 2], dtype=jnp.int32)

    variables = model.init(
        jr.PRNGKey(0),
        input_ids=input_ids,
        attention_mask=attention_mask,
        deterministic=True,
    )
    outputs = model.apply(
        variables,
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        deterministic=True,
    )

    assert outputs.logits.shape == (input_ids.shape[0], config.num_labels)
    expected = cross_entropy_loss(outputs.logits, labels)
    np.testing.assert_allclose(
        np.asarray(outputs.loss),
        np.asarray(expected),
        rtol=1e-6,
    )


def test_token_classification_logits_and_loss_path(tiny_config):
    config = tiny_config.copy(num_labels=4)
    model = ModernBertForTokenClassification(config)

    input_ids = jnp.array(
        [
            [1, 2, 3],
            [4, 5, 0],
        ],
        dtype=jnp.int32,
    )
    attention_mask = jnp.array(
        [
            [1, 1, 1],
            [1, 1, 0],
        ],
        dtype=jnp.int32,
    )
    labels = jnp.array(
        [
            [0, 1, -100],
            [2, -100, -100],
        ],
        dtype=jnp.int32,
    )

    variables = model.init(
        jr.PRNGKey(1),
        input_ids=input_ids,
        attention_mask=attention_mask,
        deterministic=True,
    )
    outputs = model.apply(
        variables,
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        deterministic=True,
    )

    assert outputs.logits.shape == (input_ids.shape[0], input_ids.shape[1], config.num_labels)
    flat_logits = outputs.logits.reshape(-1, config.num_labels)
    flat_labels = labels.reshape(-1)
    expected = cross_entropy_with_ignore_index(flat_logits, flat_labels)
    np.testing.assert_allclose(
        np.asarray(outputs.loss),
        np.asarray(expected),
        rtol=1e-6,
    )


def test_question_answering_logits_and_loss_path(tiny_config):
    config = tiny_config.copy(num_labels=2)
    model = ModernBertForQuestionAnswering(config)

    input_ids = jnp.array(
        [
            [1, 2, 3, 4],
            [5, 6, 0, 0],
        ],
        dtype=jnp.int32,
    )
    attention_mask = jnp.array(
        [
            [1, 1, 1, 1],
            [1, 1, 0, 0],
        ],
        dtype=jnp.int32,
    )
    start_positions = jnp.array([0, -1], dtype=jnp.int32)
    end_positions = jnp.array([2, -1], dtype=jnp.int32)

    variables = model.init(
        jr.PRNGKey(2),
        input_ids=input_ids,
        attention_mask=attention_mask,
        deterministic=True,
    )
    outputs = model.apply(
        variables,
        input_ids=input_ids,
        attention_mask=attention_mask,
        start_positions=start_positions,
        end_positions=end_positions,
        deterministic=True,
    )

    assert outputs.start_logits.shape == input_ids.shape
    assert outputs.end_logits.shape == input_ids.shape

    start_loss = cross_entropy_with_ignore_index(
        outputs.start_logits,
        start_positions,
        ignore_index=-1,
    )
    end_loss = cross_entropy_with_ignore_index(
        outputs.end_logits,
        end_positions,
        ignore_index=-1,
    )
    expected = 0.5 * (start_loss + end_loss)
    np.testing.assert_allclose(
        np.asarray(outputs.loss),
        np.asarray(expected),
        rtol=1e-6,
    )


def test_multiple_choice_logits_and_loss_path(tiny_config):
    config = tiny_config.copy(classifier_pooling="cls")
    model = ModernBertForMultipleChoice(config)

    input_ids = jnp.array(
        [
            [[0, 1, 2, 0], [5, 6, 0, 0], [7, 8, 9, 10]],
            [[3, 4, 0, 0], [11, 12, 13, 14], [0, 0, 0, 0]],
        ],
        dtype=jnp.int32,
    )
    attention_mask = jnp.array(
        [
            [[0, 1, 1, 0], [1, 1, 0, 0], [1, 1, 1, 1]],
            [[1, 1, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0]],
        ],
        dtype=jnp.int32,
    )
    labels = jnp.array([2, 1], dtype=jnp.int32)

    variables = model.init(
        jr.PRNGKey(3),
        input_ids=input_ids,
        attention_mask=attention_mask,
        deterministic=True,
    )
    outputs = model.apply(
        variables,
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        deterministic=True,
    )

    batch_size, num_choices = input_ids.shape[:2]
    assert outputs.logits.shape == (batch_size, num_choices)
    expected = cross_entropy_loss(outputs.logits, labels)
    np.testing.assert_allclose(
        np.asarray(outputs.loss),
        np.asarray(expected),
        rtol=1e-6,
    )
