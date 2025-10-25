import numpy as np

import jax.numpy as jnp
import jax.random as jr

from src.mmbjax.heads.masked_lm import ModernBertForMaskedLM, ModernBertLMHead
from src.mmbjax.heads.prediction import ModernBertPredictionHead


def test_embedding_kernel_reuses_token_embeddings(tiny_config):
    model = ModernBertForMaskedLM(tiny_config)
    rng = jr.PRNGKey(0)
    input_ids = jnp.ones((1, 2), dtype=jnp.int32)
    variables = model.init(rng, input_ids=input_ids)

    tied_kernel = model.apply(variables, method=ModernBertForMaskedLM._get_embedding_kernel)
    embedding_matrix = variables["params"]["model"]["embeddings"]["word_embeddings"]["embedding"]
    np.testing.assert_allclose(np.asarray(tied_kernel), np.asarray(embedding_matrix), rtol=1e-6)


def test_decoder_logits_match_manual_projection(tiny_config):
    model = ModernBertForMaskedLM(tiny_config)
    variables = model.init(jr.PRNGKey(1), input_ids=jnp.ones((1, 2), dtype=jnp.int32))

    hidden_states = jnp.arange(1 * 2 * tiny_config.hidden_size, dtype=jnp.float32).reshape(
        1, 2, tiny_config.hidden_size
    )

    head_module = ModernBertPredictionHead(tiny_config)
    head_params = {"params": variables["params"]["prediction_head"]}
    projected = head_module.apply(head_params, hidden_states)

    decoder_module = ModernBertLMHead(tiny_config)
    decoder_params = {"params": variables["params"]["decoder"]}
    embedding_kernel = variables["params"]["model"]["embeddings"]["word_embeddings"]["embedding"]
    logits = decoder_module.apply(decoder_params, projected, embedding_kernel)

    manual = jnp.einsum("bsh,vh->bsv", projected, embedding_kernel)
    if tiny_config.decoder_bias:
        manual = manual + decoder_params["params"]["bias"]

    np.testing.assert_allclose(np.asarray(logits), np.asarray(manual), rtol=1e-6)
