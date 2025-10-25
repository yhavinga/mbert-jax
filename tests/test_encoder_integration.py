import numpy as np
import pytest

import jax.numpy as jnp
import jax.random as jr

from src.mmbjax.modeling_mmbbert import ModernBertEncoder, ModernBertModel
from src.mmbjax.modules import ModernBertEmbeddings
from src.mmbjax.modules.masks import build_sliding_window_mask


@pytest.fixture
def encoder_config(tiny_config):
    return tiny_config.copy(
        num_hidden_layers=2,
        global_attn_every_n_layers=2,
        max_position_embeddings=64,
    )


def _dummy_batch():
    input_ids = jnp.array(
        [
            [1, 2, 3, 0, 0, 0],
            [4, 5, 6, 7, 8, 0],
        ],
        dtype=jnp.int32,
    )
    attention_mask = jnp.array(
        [
            [1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 0],
        ],
        dtype=jnp.int32,
    )
    return input_ids, attention_mask


def test_model_encoder_outputs_align(encoder_config):
    model = ModernBertModel(encoder_config)
    encoder = ModernBertEncoder(encoder_config)
    embeddings_module = ModernBertEmbeddings(encoder_config)

    input_ids, attention_mask = _dummy_batch()
    variables = model.init(jr.PRNGKey(0), input_ids=input_ids, deterministic=True)

    model_outputs = model.apply(
        variables,
        input_ids=input_ids,
        attention_mask=attention_mask,
        deterministic=True,
        output_hidden_states=True,
        output_attentions=True,
    )

    embeddings = embeddings_module.apply(
        {"params": variables["params"]["embeddings"]},
        input_ids=input_ids,
        deterministic=True,
    )
    encoder_outputs = encoder.apply(
        {"params": variables["params"]["encoder"]},
        embeddings,
        attention_mask=attention_mask,
        deterministic=True,
        output_hidden_states=True,
        output_attentions=True,
    )

    assert model_outputs.hidden_states is not None
    assert encoder_outputs.hidden_states is not None
    assert len(model_outputs.hidden_states) == encoder_config.num_hidden_layers + 1
    assert len(encoder_outputs.hidden_states) == encoder_config.num_hidden_layers + 1
    assert model_outputs.attentions is not None
    assert encoder_outputs.attentions is not None
    assert len(model_outputs.attentions) == encoder_config.num_hidden_layers
    assert len(encoder_outputs.attentions) == encoder_config.num_hidden_layers

    np.testing.assert_allclose(
        np.asarray(model_outputs.last_hidden_state),
        np.asarray(encoder_outputs.last_hidden_state),
        rtol=1e-6,
    )

    for lhs, rhs in zip(model_outputs.hidden_states, encoder_outputs.hidden_states):
        np.testing.assert_allclose(np.asarray(lhs), np.asarray(rhs), rtol=1e-6)
    for lhs, rhs in zip(model_outputs.attentions, encoder_outputs.attentions):
        np.testing.assert_allclose(np.asarray(lhs), np.asarray(rhs), rtol=1e-6)


def test_manual_sliding_window_mask_matches_encoder_default(encoder_config):
    model = ModernBertModel(encoder_config)
    input_ids, attention_mask = _dummy_batch()
    variables = model.init(jr.PRNGKey(1), input_ids=input_ids, deterministic=True)

    auto_outputs = model.apply(
        variables,
        input_ids=input_ids,
        attention_mask=attention_mask,
        deterministic=True,
        output_attentions=True,
    )

    mask_value = -1e9
    manual_mask = build_sliding_window_mask(
        attention_mask.astype(jnp.float32),
        window_size=encoder_config.local_attention,
        mask_value=mask_value,
        dtype=jnp.float32,
    )
    manual_outputs = model.apply(
        variables,
        input_ids=input_ids,
        attention_mask=attention_mask,
        sliding_window_mask=manual_mask,
        deterministic=True,
        output_attentions=True,
    )

    np.testing.assert_allclose(
        np.asarray(auto_outputs.last_hidden_state),
        np.asarray(manual_outputs.last_hidden_state),
        rtol=1e-6,
    )
    assert auto_outputs.attentions is not None
    assert manual_outputs.attentions is not None
    for lhs, rhs in zip(auto_outputs.attentions, manual_outputs.attentions):
        np.testing.assert_allclose(np.asarray(lhs), np.asarray(rhs), rtol=1e-6)


def test_inputs_embeds_path_matches_input_ids(encoder_config):
    model = ModernBertModel(encoder_config)
    input_ids, attention_mask = _dummy_batch()
    variables = model.init(jr.PRNGKey(2), input_ids=input_ids, deterministic=True)

    outputs_from_ids = model.apply(
        variables,
        input_ids=input_ids,
        attention_mask=attention_mask,
        deterministic=True,
    )

    embedding_table = variables["params"]["embeddings"]["word_embeddings"]["embedding"]
    inputs_embeds = jnp.take(embedding_table, input_ids, axis=0)
    outputs_from_embeds = model.apply(
        variables,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        deterministic=True,
    )

    np.testing.assert_allclose(
        np.asarray(outputs_from_ids.last_hidden_state),
        np.asarray(outputs_from_embeds.last_hidden_state),
        rtol=1e-6,
    )
