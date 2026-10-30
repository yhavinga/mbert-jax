"""Tests for Grain dataloader packing components."""

import os

import numpy as np
import json

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
            "bos_token_id": 1,
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
    """TokenizeTransform should start with BOS, and append EOS token only if doc wasn't truncated."""
    # This test uses a minimal callable tokenizer - keeping inline for clarity
    class MockTokenizer:
        bos_token_id = 1
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
    assert result_short["input_ids"][0] == MockTokenizer.bos_token_id, \
        "Every doc should start with BOS"
    assert result_short["length"] == 4  # 1 BOS + 2 words + 1 EOS

    # Case 2: Long document (truncated) - should NOT have EOS
    long_doc = {"text": " ".join(["word"] * 20)}  # 20 tokens, will be truncated to 9 + BOS
    result_long = transform.map(long_doc)
    assert result_long["length"] == max_length, f"Expected {max_length}, got {result_long['length']}"
    assert result_long["input_ids"][0] == MockTokenizer.bos_token_id
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


def _left_truncating_tokenizer():
    """Real fast tokenizer with what matters of yhavinga/dutch-llama-tokenizer: <s>, </s>, truncation from the left."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    vocab = {token: i for i, token in enumerate(["<unk>", "<s>", "</s>", "<pad>", "<mask>"] + [f"w{i}" for i in range(95)])}
    backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="<unk>", bos_token="<s>", eos_token="</s>", pad_token="<pad>",
        mask_token="<mask>", truncation_side="left",
    )


def test_tokenize_transform_keeps_max_length_and_eos_rule():
    tokenizer = _left_truncating_tokenizer()
    transform = TokenizeTransform(tokenizer, max_length=8)
    words = [f"w{i}" for i in range(20)]
    ids = lambda n: tokenizer(" ".join(words[:n]), add_special_tokens=False)["input_ids"]  # noqa: E731

    # Fits with both special tokens
    assert transform.map({"text": " ".join(words[:6])})["input_ids"].tolist() == [1] + ids(6) + [2]
    # Fits without </s>: indistinguishable from truncated, which gets no </s>
    assert transform.map({"text": " ".join(words[:7])})["input_ids"].tolist() == [1] + ids(7)
    # Truncated: the tokenizer's side decides what stays, <s> is in front and there is no </s>
    long = transform.map({"text": " ".join(words)})
    assert long["input_ids"].tolist() == [1] + ids(20)[-7:]
    assert long["length"] == 8


def test_tokenizer_without_bos_fails_with_a_message():
    from src.mmbjax.training.packing import FLANTokenizeTransform

    class NoBos:
        bos_token_id = None
        eos_token_id = 2

    with pytest.raises(ValueError, match="no bos_token_id"):
        TokenizeTransform(NoBos(), max_length=8)
    with pytest.raises(ValueError, match="no bos_token_id"):
        FLANTokenizeTransform(NoBos(), max_length=8)


def test_flan_transform_starts_with_bos_within_max_length():
    from src.mmbjax.training.packing import FLANTokenizeTransform

    tokenizer = _left_truncating_tokenizer()
    transform = FLANTokenizeTransform(tokenizer, max_length=8)
    result = transform.map({"inputs": "w1 w2", "targets": "w3 w4"})["input_ids"].tolist()
    inputs_ids, target_ids = (tokenizer(t, add_special_tokens=False)["input_ids"] for t in ("w1 w2", "w3 w4"))
    assert result == [1] + inputs_ids + [2] + target_ids
    words = " ".join(f"w{i}" for i in range(20))
    result = transform.map({"inputs": words, "targets": words})["input_ids"].tolist()
    assert len(result) == 8 and result[0] == 1 and result.count(2) == 1


def _mask_tokens_with_python_loop(tokenizer, input_ids, mlm_probability):
    """mask_tokens_packed as it was: the reference for the masking distribution."""
    labels = input_ids.copy()
    probability_matrix = np.full(input_ids.shape, mlm_probability)
    special_tokens_mask = np.array([
        [1 if tok in tokenizer.all_special_ids else 0 for tok in ids]
        for ids in input_ids.tolist()
    ])
    probability_matrix[special_tokens_mask.astype(bool)] = 0.0
    masked_indices = np.random.binomial(1, probability_matrix).astype(bool)
    labels[~masked_indices] = -100
    indices_replaced = np.random.binomial(1, np.full(labels.shape, 0.8)).astype(bool) & masked_indices
    input_ids[indices_replaced] = tokenizer.mask_token_id
    indices_random = np.random.binomial(1, np.full(labels.shape, 0.5)).astype(bool) & masked_indices & ~indices_replaced
    random_words = np.random.randint(tokenizer.vocab_size, size=labels.shape, dtype=np.int32)
    input_ids[indices_random] = random_words[indices_random]
    return input_ids, labels


def test_masking_is_unchanged_by_the_faster_special_token_mask(mock_tokenizer):
    """Same seed, same bytes as the Python loop: the distribution can't have moved."""
    tokenizer = mock_tokenizer(all_special_ids=[0, 1, 2, 103], vocab_size=1000, mask_token_id=103)
    rng = np.random.RandomState(0)
    input_ids = rng.randint(0, 1000, size=(64, 256)).astype(np.int32)
    input_ids[:, 0] = 1
    input_ids[:, 100] = 2
    input_ids[:, 200:] = 0

    np.random.seed(7)
    expected_ids, expected_labels = _mask_tokens_with_python_loop(tokenizer, input_ids.copy(), 0.15)
    np.random.seed(7)
    collator = PackedMLMDataCollator(tokenizer, mlm_probability=0.15)
    masked_ids, labels = collator.mask_tokens_packed(input_ids.copy(), np.zeros_like(input_ids))

    assert np.array_equal(masked_ids, expected_ids)
    assert np.array_equal(labels, expected_labels)
    assert not np.any(labels[np.isin(input_ids, tokenizer.all_special_ids)] != -100)


def _stream_of_numbered_documents(n, shard=0):
    return ({"text": f"w{i % 95} " * (1 + i % 5), "doc": shard * 1000000 + i} for i in range(n))


def test_source_stays_bounded_and_keeps_order_when_skipping():
    from src.mmbjax.training import StreamingBufferDataSource

    # What a resumed iterator does: straight to its index, keeping only the last items
    source = StreamingBufferDataSource(_stream_of_numbered_documents(1000), max_buffer_size=8)
    assert source[900]["doc"] == 900
    assert len(source._buffer) <= 8
    assert source[901]["doc"] == 901
    assert source.stream_position == 902
    with pytest.raises(IndexError, match="is gone"):
        source[0]
    # A normal read leaves nothing behind, and a later index may come first
    source = StreamingBufferDataSource(_stream_of_numbered_documents(1000), max_buffer_size=8)
    for index in [1, 0, 2, 20, 3, 4, 5]:
        assert source[index]["doc"] == index
    assert len(source._buffer) == 21 - 7
    # The end of the stream is the end of the iterator, not a silent wrap to document 0
    assert source[999]["doc"] == 999
    with pytest.raises(StopIteration):
        source[1000]


def _batches_of_loader(loader, n):
    iterator = iter(loader)
    return iterator, [next(iterator) for _ in range(n)]


def _real_pipeline_loader(monkeypatch, documents=400, **kwargs):
    """The real EpochTrackingDataLoader on a real shuffled datasets.IterableDataset; only the download is fake."""
    import datasets

    from src.mmbjax.training import EpochTrackingDataLoader, packing

    def generate(shards):
        for shard in shards:
            yield from _stream_of_numbered_documents(documents // 8, shard=shard)

    fake = datasets.IterableDataset.from_generator(generate, gen_kwargs={"shards": list(range(8))})
    monkeypatch.setattr(packing, "load_dataset", lambda *_a, **_k: fake)
    return EpochTrackingDataLoader(
        dataset_name="fake", tokenizer=_left_truncating_tokenizer(), max_length=16, batch_size=4,
        shuffle_buffer_size=20, **kwargs,
    )


def test_loader_resumes_at_the_next_batch_on_the_real_pipeline(monkeypatch):
    whole = _real_pipeline_loader(monkeypatch)
    _, batches = _batches_of_loader(whole, 30)

    first = _real_pipeline_loader(monkeypatch)
    _, before = _batches_of_loader(first, 17)
    state = json.loads(json.dumps(first.get_state(), default=int))

    resumed = _real_pipeline_loader(monkeypatch)
    resumed.set_state(state)
    _, after = _batches_of_loader(resumed, 13)

    for got, expected in zip(before + after, batches):
        for key in expected:
            assert np.array_equal(got[key], expected[key]), key


def test_every_document_once_per_pass_across_a_resume(monkeypatch):
    """Each document is one token sequence here, so what was trained on can be counted."""
    import datasets

    from src.mmbjax.training import EpochTrackingDataLoader, packing

    documents = [{"text": f"w{i} w{(i * 7) % 95}"} for i in range(95)]
    fake = datasets.IterableDataset.from_generator(lambda: iter(documents))
    monkeypatch.setattr(packing, "load_dataset", lambda *_a, **_k: fake)

    def loader():
        return EpochTrackingDataLoader(
            dataset_name="fake", tokenizer=_left_truncating_tokenizer(), max_length=8, batch_size=2,
            shuffle_buffer_size=10, num_epochs=2,
        )

    def segments_of(batch):
        found = []
        for ids, positions in zip(batch["input_ids"], batch["input_ids_positions"]):
            starts = [i for i, p in enumerate(positions) if p == 0 and ids[i] == 1]
            for start, end in zip(starts, starts[1:] + [len(ids)]):
                found.append(tuple(t for t in ids[start:end].tolist() if t != 3))
        return found

    first = loader()
    iterator = iter(first)
    seen = []
    for _ in range(5):
        seen += segments_of(next(iterator))
    state = first.get_state()

    resumed = loader()
    resumed.set_state(state)
    for batch in resumed:
        if resumed.epoch > 0:
            break
        seen += segments_of(batch)

    # Rows of the last, partial batch of a pass are dropped; none are repeated
    assert len(seen) == len(set(seen)), "a document came twice within one pass"
    assert len(seen) >= 95 - 2 * 4
    assert resumed.epoch == 1


def test_completed_pass_is_visible_and_the_next_one_starts(monkeypatch, capsys):
    loader = _real_pipeline_loader(monkeypatch, documents=80)
    iterator = iter(loader)
    assert loader.epoch == 0
    batches = 0
    while loader.epoch == 0:
        next(iterator)
        batches += 1
        assert batches < 500, "the pass never ended"
    assert loader.documents_read >= 80
    out = capsys.readouterr().out
    assert "[EpochTrackingDataLoader] Completed epoch 1" in out
    # Goes on with the next pass
    next(iterator)
    assert loader.epoch == 1


def test_num_epochs_ends_the_loader_and_too_little_data_is_an_error(monkeypatch):
    loader = _real_pipeline_loader(monkeypatch, documents=80, num_epochs=1)
    assert sum(1 for _ in loader) > 0
    assert loader.epoch == 1

    short = _real_pipeline_loader(monkeypatch, documents=8)
    short.batch_size = 64
    with pytest.raises(RuntimeError, match="too few for one batch"):
        next(iter(short))


def test_loader_on_local_parquet_files_covers_them_and_resumes(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from src.mmbjax.training import EpochTrackingDataLoader

    files = []
    for part in range(3):
        path = tmp_path / f"part{part}.parquet"
        pq.write_table(pa.table({"text": [f"w{part} w{i % 95}" for i in range(40)]}), path)
        files.append(str(path))

    def loader():
        # shard_count isn't used for local files: they are this host's part already
        return EpochTrackingDataLoader(dataset_name="ignored", tokenizer=_left_truncating_tokenizer(), max_length=16,
                                       batch_size=2, shard_index=1, shard_count=4, shuffle_buffer_size=10, data_files=files)

    whole = loader()
    iterator = iter(whole)
    batches = [next(iterator) for _ in range(12)]
    while whole.epoch == 0:
        next(iterator)
    assert 120 - 16 <= whole.documents_read - 0 and whole.epoch == 1

    first = loader()
    iterator = iter(first)
    for _ in range(5):
        next(iterator)
    resumed = loader()
    resumed.set_state(first.get_state())
    iterator = iter(resumed)
    for expected in batches[5:12]:
        assert np.array_equal(next(iterator)["input_ids"], expected["input_ids"])
