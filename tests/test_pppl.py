"""Tests for Pseudo-Perplexity (PPPL) evaluation."""

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from src.mmbjax.heads.masked_lm import ModernBertForMaskedLM


def test_pppl_single_token_masking(tiny_config):
    """Test that PPPL correctly masks and predicts single tokens."""
    model = ModernBertForMaskedLM(tiny_config)
    rng = jr.PRNGKey(42)
    
    # Create a simple sequence: [CLS, A, B, C, SEP] = [1, 2, 3, 4, 5]
    # Using vocab_size=16, we use tokens 1-5
    input_ids = jnp.array([[1, 2, 3, 4, 5]], dtype=jnp.int32)
    attention_mask = jnp.ones_like(input_ids)
    
    # Initialize model
    variables = model.init(rng, input_ids=input_ids)
    
    # For PPPL, we mask position 2 (token "3") and check we get valid log probs
    mask_token_id = 0  # Use 0 as mask token for tiny_config
    masked_ids = input_ids.at[0, 2].set(mask_token_id)
    
    outputs = model.apply(
        {"params": variables["params"]},
        input_ids=masked_ids,
        attention_mask=attention_mask,
        deterministic=True
    )
    
    # Extract logits at masked position
    logits = outputs.logits[0, 2]
    log_probs = jax.nn.log_softmax(logits)
    
    # Log prob should be finite and negative
    log_prob_true_token = float(log_probs[3])  # Original token was 3
    assert np.isfinite(log_prob_true_token)
    assert log_prob_true_token <= 0  # Log probs are always <= 0


def test_pppl_formula_calculation(tiny_config):
    """Test that PPPL = exp(-mean_log_prob) is computed correctly."""
    model = ModernBertForMaskedLM(tiny_config)
    rng = jr.PRNGKey(123)
    
    input_ids = jnp.array([[1, 2, 3, 4, 5]], dtype=jnp.int32)
    attention_mask = jnp.ones_like(input_ids)
    
    variables = model.init(rng, input_ids=input_ids)
    
    mask_token_id = 0
    total_log_prob = 0.0
    num_tokens = 0
    
    # Skip first and last (typically CLS/SEP), mask positions 1-3
    for i in [1, 2, 3]:
        true_token = int(input_ids[0, i])
        masked_ids = input_ids.at[0, i].set(mask_token_id)
        
        outputs = model.apply(
            {"params": variables["params"]},
            input_ids=masked_ids,
            attention_mask=attention_mask,
            deterministic=True
        )
        
        logits = outputs.logits[0, i].astype(jnp.float32)
        log_probs = jax.nn.log_softmax(logits)
        total_log_prob += float(log_probs[true_token])
        num_tokens += 1
    
    # Compute PPPL
    mean_log_prob = total_log_prob / num_tokens
    pppl = np.exp(-mean_log_prob)
    
    # PPPL should be >= 1 (since log probs are <= 0, mean is <= 0, -mean >= 0, exp >= 1)
    assert pppl >= 1.0
    # PPPL should be finite
    assert np.isfinite(pppl)
    # For random weights, PPPL should roughly be around vocab_size (uniform distribution)
    # With vocab_size=16, PPPL should be reasonable (not astronomical)
    assert pppl < 1000  # Sanity check - random model shouldn't have astronomical PPPL


def test_pppl_deterministic(tiny_config):
    """Test that PPPL calculation is deterministic (no dropout in eval)."""
    model = ModernBertForMaskedLM(tiny_config)
    rng = jr.PRNGKey(999)
    
    input_ids = jnp.array([[1, 2, 3, 4, 5]], dtype=jnp.int32)
    attention_mask = jnp.ones_like(input_ids)
    
    variables = model.init(rng, input_ids=input_ids)
    
    mask_token_id = 0
    masked_ids = input_ids.at[0, 2].set(mask_token_id)
    
    # Run twice with deterministic=True
    outputs1 = model.apply(
        {"params": variables["params"]},
        input_ids=masked_ids,
        attention_mask=attention_mask,
        deterministic=True
    )
    outputs2 = model.apply(
        {"params": variables["params"]},
        input_ids=masked_ids,
        attention_mask=attention_mask,
        deterministic=True
    )
    
    # Outputs should be identical
    np.testing.assert_allclose(
        np.asarray(outputs1.logits),
        np.asarray(outputs2.logits),
        rtol=1e-5
    )
