"""Tests for sequence packing and block-diagonal attention masking."""

import os

import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import jax.numpy as jnp
import jax.random as jr

from src.mmbjax.configuration_mmbbert import MMBertConfig
from src.mmbjax.modules.attention import ModernBertAttention
from src.mmbjax.modeling_mmbbert import ModernBertEncoder
from src.mmbjax.modules.rope import clear_rotary_cache


def test_segment_mask_blocks_cross_document_attention(tiny_config):
    """Tokens from different segments must not attend to each other."""
    clear_rotary_cache()

    # Two segments: [0,0,0,1,1,1] - tokens 0-2 are doc1, 3-5 are doc2
    batch_size, seq_len = 1, 6
    segment_ids = jnp.array([[0, 0, 0, 1, 1, 1]], dtype=jnp.int32)

    config = tiny_config.copy(global_attn_every_n_layers=1)
    hidden_states = jnp.ones((batch_size, seq_len, config.hidden_size), dtype=jnp.float32)
    attention_mask = jnp.ones((batch_size, seq_len), dtype=jnp.float32)

    module = ModernBertAttention(config, layer_idx=0)
    variables = module.init(
        jr.PRNGKey(0), hidden_states, attention_mask=attention_mask,
        segment_ids=segment_ids, deterministic=True
    )

    _, attn_probs = module.apply(
        variables, hidden_states, attention_mask=attention_mask,
        segment_ids=segment_ids, deterministic=True, output_attentions=True
    )

    probs = np.asarray(attn_probs)  # [1, heads, 6, 6]

    # Cross-segment attention must be exactly zero
    # Segment 0 tokens (indices 0-2) should have zero attention to Segment 1 (indices 3-5)
    cross_seg_0_to_1 = probs[0, :, 0:3, 3:6]  # Doc1 attending to Doc2
    np.testing.assert_allclose(cross_seg_0_to_1, 0.0, atol=1e-6)

    cross_seg_1_to_0 = probs[0, :, 3:6, 0:3]  # Doc2 attending to Doc1
    np.testing.assert_allclose(cross_seg_1_to_0, 0.0, atol=1e-6)

    # Within-segment attention should sum to 1.0 (proper normalization)
    within_seg_0 = probs[0, :, 0:3, 0:3].sum(axis=-1)
    np.testing.assert_allclose(within_seg_0, 1.0, atol=1e-6)

    within_seg_1 = probs[0, :, 3:6, 3:6].sum(axis=-1)
    np.testing.assert_allclose(within_seg_1, 1.0, atol=1e-6)


def test_segment_mask_with_three_documents(tiny_config):
    """Block-diagonal pattern with 3 segments: [0,0,1,1,2,2]."""
    clear_rotary_cache()

    segment_ids = jnp.array([[0, 0, 1, 1, 2, 2]], dtype=jnp.int32)
    batch_size, seq_len = 1, 6

    config = tiny_config.copy(global_attn_every_n_layers=1)
    hidden_states = jnp.ones((batch_size, seq_len, config.hidden_size), dtype=jnp.float32)
    attention_mask = jnp.ones((batch_size, seq_len), dtype=jnp.float32)

    module = ModernBertAttention(config, layer_idx=0)
    variables = module.init(
        jr.PRNGKey(0), hidden_states, attention_mask=attention_mask,
        segment_ids=segment_ids, deterministic=True
    )

    _, attn_probs = module.apply(
        variables, hidden_states, attention_mask=attention_mask,
        segment_ids=segment_ids, deterministic=True, output_attentions=True
    )

    probs = np.asarray(attn_probs)

    # All cross-segment attention must be zero
    for i in range(3):
        for j in range(3):
            if i != j:
                # Segment i tokens attending to segment j tokens
                i_start, i_end = i*2, i*2+2
                j_start, j_end = j*2, j*2+2
                cross_seg = probs[0, :, i_start:i_end, j_start:j_end]
                np.testing.assert_allclose(cross_seg, 0.0, atol=1e-6,
                                           err_msg=f"Segment {i} attending to segment {j}")


def test_attention_without_segment_ids_unchanged(tiny_config):
    """When segment_ids=None, attention should work as before."""
    clear_rotary_cache()

    batch_size, seq_len = 1, 4
    hidden_states = jnp.ones((batch_size, seq_len, tiny_config.hidden_size), dtype=jnp.float32)
    attention_mask = jnp.ones((batch_size, seq_len), dtype=jnp.float32)

    config = tiny_config.copy(global_attn_every_n_layers=1)

    # Run without segment_ids
    module = ModernBertAttention(config, layer_idx=0)
    variables = module.init(
        jr.PRNGKey(0), hidden_states, attention_mask=attention_mask, deterministic=True
    )

    _, attn_probs = module.apply(
        variables, hidden_states, attention_mask=attention_mask,
        deterministic=True, output_attentions=True
    )

    # Verify attention is fully connected (all-to-all)
    probs = np.asarray(attn_probs)
    np.testing.assert_allclose(probs.sum(axis=-1), 1.0, atol=1e-6)


def test_segment_mask_respects_padding(tiny_config):
    """Padding tokens should still be masked even with segment_ids."""
    clear_rotary_cache()

    # Packed with padding: [0,0,0,PAD,1,1,1,PAD]
    segment_ids = jnp.array([[0, 0, 0, 0, 1, 1, 1, 1]], dtype=jnp.int32)
    attention_mask = jnp.array([[1, 1, 1, 0, 1, 1, 1, 0]], dtype=jnp.float32)

    batch_size, seq_len = 1, 8
    config = tiny_config.copy(global_attn_every_n_layers=1)
    hidden_states = jnp.ones((batch_size, seq_len, config.hidden_size), dtype=jnp.float32)

    module = ModernBertAttention(config, layer_idx=0)
    variables = module.init(
        jr.PRNGKey(0), hidden_states, attention_mask=attention_mask,
        segment_ids=segment_ids, deterministic=True
    )

    _, attn_probs = module.apply(
        variables, hidden_states, attention_mask=attention_mask,
        segment_ids=segment_ids, deterministic=True, output_attentions=True
    )

    probs = np.asarray(attn_probs)

    # Padding positions (3, 7) should receive zero attention
    np.testing.assert_allclose(probs[:, :, :, 3], 0.0, atol=1e-6)
    np.testing.assert_allclose(probs[:, :, :, 7], 0.0, atol=1e-6)

    # Cross-segment still blocked
    np.testing.assert_allclose(probs[0, :, 0:3, 4:7], 0.0, atol=1e-6)


def test_segment_mask_with_local_attention(tiny_config):
    """Local attention window should be bounded by segment boundaries."""
    clear_rotary_cache()

    # Local layer with window=4: tokens can attend ±2 positions
    # But segment boundary should still block
    segment_ids = jnp.array([[0, 0, 0, 0, 1, 1, 1, 1]], dtype=jnp.int32)
    attention_mask = jnp.ones((1, 8), dtype=jnp.float32)

    config = tiny_config.copy(local_attention=4, global_attn_every_n_layers=10)  # Layer 1 is local
    hidden_states = jnp.ones((1, 8, config.hidden_size), dtype=jnp.float32)

    module = ModernBertAttention(config, layer_idx=1)  # Local attention layer
    variables = module.init(
        jr.PRNGKey(0), hidden_states, attention_mask=attention_mask,
        segment_ids=segment_ids, deterministic=True
    )

    _, attn_probs = module.apply(
        variables, hidden_states, attention_mask=attention_mask,
        segment_ids=segment_ids, deterministic=True, output_attentions=True
    )

    probs = np.asarray(attn_probs)

    # Token at position 3 (last of segment 0) is within window of position 4 (first of segment 1)
    # But should NOT attend due to segment boundary
    np.testing.assert_allclose(probs[0, :, 3, 4], 0.0, atol=1e-6,
                               err_msg="Segment boundary should block attention despite sliding window")


def test_segment_mask_batch_processing(tiny_config):
    """Each batch element should have independent segment masking."""
    clear_rotary_cache()

    # Batch 0: [0,0,1,1], Batch 1: [0,1,1,1] (different segmentations)
    segment_ids = jnp.array([[0, 0, 1, 1], [0, 1, 1, 1]], dtype=jnp.int32)
    attention_mask = jnp.ones((2, 4), dtype=jnp.float32)

    config = tiny_config.copy(global_attn_every_n_layers=1)
    hidden_states = jnp.ones((2, 4, config.hidden_size), dtype=jnp.float32)

    module = ModernBertAttention(config, layer_idx=0)
    variables = module.init(
        jr.PRNGKey(0), hidden_states, attention_mask=attention_mask,
        segment_ids=segment_ids, deterministic=True
    )

    _, attn_probs = module.apply(
        variables, hidden_states, attention_mask=attention_mask,
        segment_ids=segment_ids, deterministic=True, output_attentions=True
    )

    probs = np.asarray(attn_probs)

    # Batch 0: positions 0-1 should not attend to 2-3
    np.testing.assert_allclose(probs[0, :, 0:2, 2:4], 0.0, atol=1e-6)

    # Batch 1: position 0 should not attend to 1-3
    np.testing.assert_allclose(probs[1, :, 0, 1:4], 0.0, atol=1e-6)


def test_segment_mask_shape_is_broadcastable(tiny_config):
    """Segment mask should broadcast to [batch, num_heads, seq_len, seq_len]."""
    batch_size, seq_len, num_heads = 2, 8, 4
    segment_ids = jnp.array([[0, 0, 0, 0, 1, 1, 1, 1],
                             [0, 0, 1, 1, 2, 2, 3, 3]], dtype=jnp.int32)

    # Replicate the mask construction logic from attention.py
    mask_dtype = jnp.float32
    mask_value = jnp.array(jnp.finfo(mask_dtype).min, dtype=mask_dtype)

    seg_i = segment_ids[:, None, :, None]  # [b, 1, s, 1]
    seg_j = segment_ids[:, None, None, :]  # [b, 1, 1, s]

    assert seg_i.shape == (batch_size, 1, seq_len, 1)
    assert seg_j.shape == (batch_size, 1, 1, seq_len)

    segment_mask = (seg_i == seg_j).astype(mask_dtype)
    segment_mask = (1.0 - segment_mask) * mask_value

    assert segment_mask.shape == (batch_size, 1, seq_len, seq_len)

    # Verify it broadcasts to attention scores shape
    attn_scores = jnp.zeros((batch_size, num_heads, seq_len, seq_len), dtype=mask_dtype)
    broadcasted = attn_scores + segment_mask
    assert broadcasted.shape == (batch_size, num_heads, seq_len, seq_len)


def test_encoder_with_segment_ids(tiny_config):
    """Encoder should pass segment_ids through all layers."""
    clear_rotary_cache()

    config = tiny_config.copy(num_hidden_layers=2, global_attn_every_n_layers=1)

    encoder = ModernBertEncoder(config)
    hidden_states = jnp.ones((1, 8, config.hidden_size), dtype=jnp.float32)
    segment_ids = jnp.array([[0, 0, 0, 0, 1, 1, 1, 1]], dtype=jnp.int32)
    attention_mask = jnp.ones((1, 8), dtype=jnp.float32)

    variables = encoder.init(
        jr.PRNGKey(0),
        hidden_states,
        attention_mask=attention_mask,
        segment_ids=segment_ids,
    )

    output = encoder.apply(
        variables,
        hidden_states,
        attention_mask=attention_mask,
        segment_ids=segment_ids,
        output_attentions=True,
    )

    assert output.last_hidden_state.shape == (1, 8, config.hidden_size)
    assert output.attentions is not None
    assert len(output.attentions) == 2  # 2 layers

    # Verify cross-segment attention is blocked in both layers
    for layer_idx, attn in enumerate(output.attentions):
        attn_np = np.asarray(attn)
        # First segment should not attend to second
        cross_seg = attn_np[0, :, 0:4, 4:8]
        np.testing.assert_allclose(cross_seg, 0.0, atol=1e-6,
                                   err_msg=f"Layer {layer_idx} has cross-segment attention")


@pytest.mark.parametrize("batch_size,seq_len,num_segments", [
    (1, 8, 2),
    (1, 12, 3),
    (2, 16, 4),
])
def test_segment_mask_properties(batch_size, seq_len, num_segments):
    """Verify properties: symmetric, block-diagonal, zero cross-segment."""
    # Create segment_ids with equal-sized segments
    seg_len = seq_len // num_segments
    segment_ids = jnp.array([
        [i // seg_len for i in range(seq_len)]
        for _ in range(batch_size)
    ], dtype=jnp.int32)

    # Build mask - same as attention.py
    mask_dtype = jnp.float32
    mask_value = jnp.array(jnp.finfo(mask_dtype).min, dtype=mask_dtype)

    seg_i = segment_ids[:, None, :, None]
    seg_j = segment_ids[:, None, None, :]

    # Property 1: Correct shape
    assert seg_i.shape == (batch_size, 1, seq_len, 1)
    assert seg_j.shape == (batch_size, 1, 1, seq_len)

    # Property 2: Symmetric (comparison matrix is symmetric)
    comparison = seg_i == seg_j  # [b, 1, s, s]
    np.testing.assert_array_equal(
        np.asarray(comparison),
        np.swapaxes(np.asarray(comparison), -1, -2)
    )

    # Property 3: Block-diagonal structure
    # Within segment: comparison should be True (1)
    # Cross segment: comparison should be False (0)
    for b in range(batch_size):
        for i in range(num_segments):
            for j in range(num_segments):
                block = np.asarray(comparison[b, 0, i*seg_len:(i+1)*seg_len, j*seg_len:(j+1)*seg_len])
                if i == j:
                    # Within segment: should be all 1s (True)
                    np.testing.assert_allclose(block, 1.0)
                else:
                    # Cross segment: should be all 0s (False)
                    np.testing.assert_allclose(block, 0.0)
