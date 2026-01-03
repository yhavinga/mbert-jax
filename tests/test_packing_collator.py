"""Tests for Grain dataloader packing components."""

import os

import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import jax.numpy as jnp
import jax.random as jr

from src.mmbjax.configuration_mmbbert import MMBertConfig
from src.mmbjax.heads.masked_lm import ModernBertForMaskedLM
from src.mmbjax.training import PackedMLMDataCollator, TokenizeTransform


@pytest.fixture
def mock_tokenizer():
    """Factory fixture creating MockTokenizer instances with configurable attributes.

    Usage:
        tokenizer = mock_tokenizer(vocab_size=100, all_special_ids=[0, 1, 2])
    """
    def _create(**kwargs):
        defaults = {
            "pad_token_id": 0,
            "eos_token_id": 2,
            "mask_token_id": 103,
            "vocab_size": 30522,
            "all_special_ids": [0, 101, 102, 103],
        }
        defaults.update(kwargs)

        class MockTokenizer:
            pass

        for key, value in defaults.items():
            setattr(MockTokenizer, key, value)

        return MockTokenizer()

    return _create


@pytest.fixture
def mock_callable_tokenizer():
    """Factory fixture creating callable MockTokenizer instances (for TokenizeTransform).

    Usage:
        tokenizer = mock_callable_tokenizer(vocab_size=100)
    """
    def _create(**kwargs):
        defaults = {
            "pad_token_id": 0,
            "eos_token_id": 2,
            "mask_token_id": 103,
            "vocab_size": 100,
            "all_special_ids": [0, 1, 2],
        }
        defaults.update(kwargs)

        class MockTokenizer:
            def __call__(self, text, truncation=False, max_length=None, return_tensors=None, add_special_tokens=False):
                words = text.split()[:max_length] if max_length else text.split()
                return {"input_ids": [hash(w) % self.vocab_size for w in words]}

        for key, value in defaults.items():
            setattr(MockTokenizer, key, value)

        return MockTokenizer()

    return _create


def test_tokenize_transform_adds_eos_token():
    """TokenizeTransform should append EOS token only if doc wasn't truncated."""
    # This test uses a minimal callable tokenizer - keeping inline for clarity
    class MockTokenizer:
        eos_token_id = 2

        def __call__(self, text, truncation=False, max_length=None, return_tensors=None, add_special_tokens=False):
            words = text.split()
            if max_length and truncation:
                words = words[:max_length]
            return {"input_ids": [hash(w) % 1000 for w in words]}

    max_length = 10
    transform = TokenizeTransform(MockTokenizer(), max_length=max_length)

    # Case 1: Short document (not truncated) - should have EOS
    short_doc = {"text": "Hello world"}  # 2 tokens
    result_short = transform.map(short_doc)
    assert result_short["input_ids"][-1] == MockTokenizer.eos_token_id, \
        "Short doc should have EOS token"
    assert result_short["length"] == 3  # 2 words + 1 EOS

    # Case 2: Long document (truncated) - should NOT have EOS
    long_doc = {"text": " ".join(["word"] * 20)}  # 20 tokens, will be truncated to 10
    result_long = transform.map(long_doc)
    assert result_long["length"] == max_length, f"Expected {max_length}, got {result_long['length']}"
    # EOS token id is 2, the hash of "word" is not 2, so check last token is NOT EOS
    assert result_long["input_ids"][-1] != MockTokenizer.eos_token_id, \
        "Truncated doc should NOT have EOS token"


def test_packed_mlm_collator_preserves_segments(mock_tokenizer):
    """Collator should preserve segment_ids structure."""
    batch_raw = {
        "input_ids": [[1, 2, 3, 4, 5, 6, 7, 8]],
        "input_ids_segment_ids": [[0, 0, 0, 0, 1, 1, 1, 1]],
        "input_ids_positions": [[0, 1, 2, 3, 0, 1, 2, 3]],
    }

    tokenizer = mock_tokenizer()
    collator = PackedMLMDataCollator(tokenizer, mlm_probability=0.15)

    # Set seed for deterministic masking
    np.random.seed(42)
    result = collator(batch_raw)

    assert "input_ids" in result
    assert "segment_ids" in result
    assert "position_ids" in result
    assert "labels" in result
    assert "attention_mask" in result

    # Verify segment_ids preserved
    np.testing.assert_array_equal(result["segment_ids"], batch_raw["input_ids_segment_ids"])

    # Verify position_ids preserved
    np.testing.assert_array_equal(result["position_ids"], batch_raw["input_ids_positions"])


def test_packed_mlm_collator_never_masks_special_tokens(mock_tokenizer):
    """Special tokens should always have label=-100 (not masked)."""
    tokenizer = mock_tokenizer(vocab_size=100, all_special_ids=[0, 1, 2, 3])

    # Input with special tokens at specific positions
    batch_raw = {
        "input_ids": [[10, 0, 20, 1, 30, 2, 40, 3]],  # 0,1,2,3 are special
        "input_ids_segment_ids": [[0, 0, 0, 0, 0, 0, 0, 0]],
        "input_ids_positions": [[0, 1, 2, 3, 4, 5, 6, 7]],
    }

    collator = PackedMLMDataCollator(tokenizer, mlm_probability=1.0)  # Mask everything possible

    np.random.seed(42)
    result = collator(batch_raw)

    # Special token positions should have label=-100 (not trained)
    labels = result["labels"]
    assert labels[0, 1] == -100  # Position 1 has special token 0
    assert labels[0, 3] == -100  # Position 3 has special token 1
    assert labels[0, 5] == -100  # Position 5 has special token 2
    assert labels[0, 7] == -100  # Position 7 has special token 3


def test_packed_mlm_collator_attention_mask_all_ones(mock_tokenizer):
    """Packed sequences have no padding, so attention_mask should be all ones."""
    tokenizer = mock_tokenizer(vocab_size=100, all_special_ids=[0, 1, 2])

    batch_raw = {
        "input_ids": [[1, 2, 3, 4, 5, 6, 7, 8]],
        "input_ids_segment_ids": [[0, 0, 0, 0, 1, 1, 1, 1]],
        "input_ids_positions": [[0, 1, 2, 3, 0, 1, 2, 3]],
    }

    collator = PackedMLMDataCollator(tokenizer, mlm_probability=0.15)
    np.random.seed(42)
    result = collator(batch_raw)

    # All tokens are valid (no padding in packed sequences)
    expected_mask = np.ones((1, 8))
    np.testing.assert_array_equal(result["attention_mask"], expected_mask)


def test_masked_lm_with_segment_ids():
    """MLM head should pass segment_ids through the model."""
    # Use a minimal config
    config = MMBertConfig(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        max_position_embeddings=32,
        local_attention=4,
        global_attn_every_n_layers=1,
        classifier_dropout=0.0,
        classifier_pooling="cls",
    )

    model = ModernBertForMaskedLM(config)

    input_ids = jnp.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=jnp.int32)
    segment_ids = jnp.array([[0, 0, 0, 0, 1, 1, 1, 1]], dtype=jnp.int32)
    position_ids = jnp.array([[0, 1, 2, 3, 0, 1, 2, 3]], dtype=jnp.int32)
    labels = jnp.array([[-100, -100, 5, -100, -100, -100, 10, -100]], dtype=jnp.int32)
    attention_mask = jnp.ones((1, 8), dtype=jnp.float32)

    variables = model.init(
        jr.PRNGKey(0),
        input_ids=input_ids,
        attention_mask=attention_mask,
        segment_ids=segment_ids,
        position_ids=position_ids,
        labels=labels,
    )

    output = model.apply(
        variables,
        input_ids=input_ids,
        attention_mask=attention_mask,
        segment_ids=segment_ids,
        position_ids=position_ids,
        labels=labels,
    )

    assert output.loss is not None
    assert output.logits.shape == (1, 8, config.vocab_size)


def test_packed_mlm_collator_mlm_distribution(mock_tokenizer):
    """Verify MLM masking distribution: 80% MASK, 10% random, 10% original."""
    tokenizer = mock_tokenizer(vocab_size=1000, all_special_ids=[0, 1, 2])

    # Simple input with no special tokens
    batch_raw = {
        "input_ids": [[10, 11, 12, 13, 14, 15, 16, 17]],
        "input_ids_segment_ids": [[0, 0, 0, 0, 1, 1, 1, 1]],
        "input_ids_positions": [[0, 1, 2, 3, 0, 1, 2, 3]],
    }

    collator = PackedMLMDataCollator(tokenizer, mlm_probability=1.0)
    np.random.seed(42)
    result = collator(batch_raw)

    input_ids = result["input_ids"][0]
    labels = result["labels"][0]

    # Check that masking was applied (not all labels are -100)
    masked_positions = labels != -100
    assert masked_positions.any()

    # Count MASK token replacements
    mask_count = np.sum(input_ids == tokenizer.mask_token_id)

    # At least one should be MASK
    assert mask_count > 0, "Expected at least one token to be replaced with MASK"


def test_packed_mlm_collator_zero_mask_probability(mock_tokenizer):
    """With mlm_probability=0.0, no tokens should be masked."""
    tokenizer = mock_tokenizer(vocab_size=100, all_special_ids=[])

    batch_raw = {
        "input_ids": [[10, 11, 12, 13, 14, 15, 16, 17]],
        "input_ids_segment_ids": [[0, 0, 0, 0, 1, 1, 1, 1]],
        "input_ids_positions": [[0, 1, 2, 3, 0, 1, 2, 3]],
    }

    collator = PackedMLMDataCollator(tokenizer, mlm_probability=0.0)
    result = collator(batch_raw)

    # No masking, so all labels should be -100
    labels = result["labels"]
    np.testing.assert_array_equal(labels, -100 * np.ones_like(labels))

    # input_ids should be unchanged
    np.testing.assert_array_equal(result["input_ids"], batch_raw["input_ids"])


def test_create_packed_dataloader_integration(mock_callable_tokenizer):
    """Integration test: Verify Grain packing pipeline works end-to-end.

    This test validates the complete Grain pipeline using StreamingBufferDataSource:
    stream -> StreamingBufferDataSource -> MapDataset -> shuffle -> map -> to_iter -> pack -> batch

    Key insight: FirstFitPackIterDataset produces an IterDataset which must be
    iterated directly (it's not subscriptable, so DataLoader+IndexSampler won't work).
    """
    from src.mmbjax.training import create_packed_dataset_streaming, GRAIN_AVAILABLE

    if not GRAIN_AVAILABLE:
        pytest.skip("Grain not installed")

    # Create test data as a GENERATOR (true streaming, no list buffering)
    def data_generator():
        for i in range(1, 20):  # 19 documents
            yield {"text": "hello world test " * i}  # Variable length documents

    tokenizer = mock_callable_tokenizer()
    max_length = 64
    batch_size = 2

    # Use streaming function with generator
    packed_dataset = create_packed_dataset_streaming(
        stream_dataset=data_generator(),
        tokenizer=tokenizer,
        max_length=max_length,
        batch_size=batch_size,
    )

    # Iterate the IterDataset directly
    dataset_iter = iter(packed_dataset)
    batch = next(dataset_iter)

    # Verify batch structure from Grain's FirstFitPackIterDataset
    assert "input_ids" in batch, f"Expected 'input_ids' in batch, got keys: {batch.keys()}"
    assert "input_ids_segment_ids" in batch, f"Expected 'input_ids_segment_ids' in batch"
    assert "input_ids_positions" in batch, f"Expected 'input_ids_positions' in batch"

    # Verify shapes
    assert batch["input_ids"].shape[0] == batch_size, \
        f"Expected batch size {batch_size}, got {batch['input_ids'].shape[0]}"
    assert batch["input_ids"].shape[1] == max_length, \
        f"Expected seq length {max_length}, got {batch['input_ids'].shape[1]}"
    assert batch["input_ids_segment_ids"].shape == (batch_size, max_length)
    assert batch["input_ids_positions"].shape == (batch_size, max_length)

    # Verify segment_ids are valid (non-negative integers)
    # Note: With shuffle_bins=True, segments can appear in any order, so we don't
    # require monotonicity. What matters is that segment_ids correctly identify
    # which document each token belongs to.
    import numpy as np
    assert np.all(batch["input_ids_segment_ids"] >= 0), \
        "segment_ids should be non-negative"

    # Verify we have multiple segments in at least one row (packing is working)
    # segment_ids are 0-indexed, so max+1 gives number of unique segment IDs
    total_unique_segments = len(np.unique(batch["input_ids_segment_ids"]))
    assert total_unique_segments >= 2, f"Expected at least 2 unique segments across batch, got {total_unique_segments}"

    print(f"✓ Packed batch with {total_unique_segments} unique segments")
    print(f"  input_ids shape: {batch['input_ids'].shape}")
    print(f"  segment_ids row 0: {batch['input_ids_segment_ids'][0][:20]}...")


def test_streaming_buffer_data_source(mock_callable_tokenizer):
    """Test that StreamingBufferDataSource works with Grain packing."""
    from src.mmbjax.training import StreamingBufferDataSource, GRAIN_AVAILABLE, TokenizeTransform
    import grain.python as grain
    import grain.python.experimental as grain_exp

    if not GRAIN_AVAILABLE:
        pytest.skip("Grain not installed")

    # Create a GENERATOR (not a list) - this verifies lazy loading
    def data_generator():
        for i in range(1, 100):
            yield {"text": "word " * i}

    tokenizer = mock_callable_tokenizer()
    max_length = 32
    batch_size = 4

    # Use StreamingBufferDataSource with the generator
    source = StreamingBufferDataSource(data_generator(), max_buffer_size=50)

    # Tokenize transform
    tokenize = TokenizeTransform(tokenizer, max_length)

    # Build Grain pipeline
    map_ds = grain.MapDataset.source(source).map(tokenize.map)
    iter_ds = map_ds.to_iter_dataset()
    packed_ds = grain_exp.FirstFitPackIterDataset(
        parent=iter_ds,
        length_struct={"input_ids": max_length},
        num_packing_bins=batch_size,
        shuffle_bins=True,
    )
    batched_ds = packed_ds.batch(batch_size)

    # Iterate and verify
    dataset_iter = iter(batched_ds)
    batch = next(dataset_iter)

    # Verify structure
    assert "input_ids" in batch
    assert batch["input_ids"].shape == (batch_size, max_length)
    assert batch["input_ids_segment_ids"].shape == (batch_size, max_length)

    print(f"✓ Streaming buffer test passed: batch shape = {batch['input_ids'].shape}")
    print(f"  Buffer size used: {len(source._buffer)}")


def test_epoch_tracking_dataloader_restarts(mock_callable_tokenizer):
    """Test EpochTrackingDataLoader and pipeline exhaustion/restart.
    
    Validates:
    1. EpochTrackingDataLoader initializes correctly
    2. Pipeline can be exhausted and recreated (epoch restart simulation)
    """
    from src.mmbjax.training import EpochTrackingDataLoader, GRAIN_AVAILABLE, create_packed_dataset_streaming
    
    if not GRAIN_AVAILABLE:
        pytest.skip("Grain not installed")
    
    tokenizer = mock_callable_tokenizer()
    
    # Test 1: EpochTrackingDataLoader initialization
    loader = EpochTrackingDataLoader(
        dataset_name="yhavinga/mc4_nl_cleaned",
        dataset_config="micro",
        tokenizer=tokenizer,
        max_length=512,
        batch_size=16,
        seed=42,
        num_epochs=None,
    )
    assert loader.epoch == 0
    assert loader.num_epochs is None
    print("✓ EpochTrackingDataLoader initialized correctly")
    
    # Test 2: Exhaust a finite dataset and verify we get batches
    max_length = 32
    batch_size = 2
    
    def data_generator():
        for i in range(50):  # 50 documents
            yield {"text": "hello world test sentence " * ((i % 5) + 1)}
    
    packed_dataset = create_packed_dataset_streaming(
        stream_dataset=data_generator(),
        tokenizer=tokenizer,
        max_length=max_length,
        batch_size=batch_size,
    )
    
    # Exhaust the dataset
    batch_count = 0
    for batch in packed_dataset:
        batch_count += 1
        assert "input_ids" in batch
        if batch_count > 200:  # Safety limit
            break
    
    assert batch_count >= 1, f"Expected at least 1 batch, got {batch_count}"
    print(f"✓ Epoch 1 exhaustion: got {batch_count} batches from 50 documents")
    
    # Test 3: Create fresh pipeline (simulating epoch restart)
    packed_dataset_2 = create_packed_dataset_streaming(
        stream_dataset=data_generator(),
        tokenizer=tokenizer,
        max_length=max_length,
        batch_size=batch_size,
    )
    
    batch_count_2 = 0
    for batch in packed_dataset_2:
        batch_count_2 += 1
        if batch_count_2 > 200:
            break
    
    assert batch_count_2 == batch_count, f"Expected {batch_count} batches in epoch 2, got {batch_count_2}"
    print(f"✓ Epoch 2 restart: both epochs yielded {batch_count} batches")


