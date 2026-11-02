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
from src.mmbjax.training import DENOISER_NAMES, MixedDenoiserCollator, PackedMLMDataCollator, TokenizeTransform


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
            def __call__(self, text, truncation=False, max_length=None, return_tensors=None, add_special_tokens=False,
                         split_special_tokens=False):
                words = text.split()[:max_length] if max_length else text.split()
                return {"input_ids": [hash(w) % self.vocab_size for w in words]}

        for key, value in defaults.items():
            setattr(MockTokenizer, key, value)

        return MockTokenizer()

    return _create


def test_tokenize_transform_adds_eos_token():
    """Every piece starts with BOS; only the piece with the end of the document has EOS."""
    # This test uses a minimal callable tokenizer - keeping inline for clarity
    class MockTokenizer:
        bos_token_id = 1
        eos_token_id = 2

        def __call__(self, text, truncation=False, max_length=None, return_tensors=None, add_special_tokens=False,
                     split_special_tokens=False):
            words = text.split()
            if max_length and truncation:
                words = words[:max_length]
            return {"input_ids": [hash(w) % 1000 + 10 for w in words]}

    max_length = 10
    transform = TokenizeTransform(MockTokenizer(), max_length=max_length)

    # Case 1: Short document - one piece, with EOS
    pieces = transform.map({"text": "Hello world"})  # 2 tokens
    assert len(pieces) == 1
    assert pieces[0]["input_ids"][-1] == MockTokenizer.eos_token_id, "Short doc should have EOS token"
    assert pieces[0]["input_ids"][0] == MockTokenizer.bos_token_id, "Every doc should start with BOS"
    assert pieces[0]["length"] == 4  # 1 BOS + 2 words + 1 EOS

    # Case 2: Long document - pieces of 9 tokens + BOS, and EOS on the last only
    pieces = transform.map({"text": " ".join(["word"] * 20)})
    assert [piece["length"] for piece in pieces] == [10, 10, 4]
    assert all(piece["input_ids"][0] == MockTokenizer.bos_token_id for piece in pieces)
    assert [piece["input_ids"][-1] == MockTokenizer.eos_token_id for piece in pieces] == [False, False, True]


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
    iter_ds = grain_exp.FlatMapIterDataset(map_ds.to_iter_dataset(), tokenize)
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
    pieces = lambda n: [p["input_ids"].tolist() for p in transform.map({"text": " ".join(words[:n])})]  # noqa: E731

    # Fits with both special tokens
    assert pieces(6) == [[1] + ids(6) + [2]]
    # Fits without </s>: this is the end of the document, so </s> takes a last token along to a new piece
    assert pieces(7) == [[1] + ids(6), [1] + ids(7)[6:] + [2]]
    # Longer: all of it, from the start. The tokenizer's truncation side has no say.
    assert pieces(20) == [[1] + ids(7), [1] + ids(14)[7:], [1] + ids(20)[14:] + [2]]
    assert all(p["length"] == 8 for p in transform.map({"text": " ".join(words)})[:2])


@pytest.mark.parametrize("n", [0, 1, 2, 7, 8, 9, 10, 15, 16, 17, 18, 19, 26, 27, 28, 29, 5000])
@pytest.mark.parametrize("max_length", [3, 4, 10])
def test_pieces_hold_the_document_exactly_within_the_limits(n, max_length):
    from src.mmbjax.training.packing import split_into_pieces

    tokens = list(range(100, 100 + n))
    pieces = split_into_pieces(tokens, max_length, 1, 2)

    assert all(2 <= len(piece) <= max_length for piece in pieces)
    assert all(piece[0] == 1 for piece in pieces)
    # The real end has the only </s>
    assert [piece[-1] == 2 for piece in pieces] == [False] * (len(pieces) - 1) + [True]
    rebuilt = [t for piece in pieces[:-1] for t in piece[1:]] + pieces[-1][1:-1]
    assert rebuilt == tokens
    # No waste: every piece but the last two is full, and only a document without tokens has an empty piece
    assert all(len(piece) == max_length for piece in pieces[:-2])
    assert n == 0 or len(pieces[-1]) >= 3


def _tokenizer_with_the_pieces_of_special_tokens():
    """Special tokens as in yhavinga/dutch-llama-tokenizer, and a vocabulary that can spell them as text."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    words = ["<", "</", ">", "s", "pad", "unk", "mask", "a", "b"]
    vocab = {token: i for i, token in enumerate(["<unk>", "<s>", "</s>", "<pad>", "<mask>"] + words)}
    backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="<unk>", bos_token="<s>", eos_token="</s>", pad_token="<pad>",
        mask_token="<mask>", truncation_side="left",
    )


LITERAL_SPECIAL_TOKENS = "a <s> b </s> a <pad> b <unk> a <mask> b"


def test_literal_special_tokens_in_a_document_are_text():
    tokenizer = _tokenizer_with_the_pieces_of_special_tokens()
    specials = set(tokenizer.all_special_ids)
    assert specials == {0, 1, 2, 3, 4}
    # Without the option the strings do become those tokens, or this test says nothing
    assert specials <= set(tokenizer(LITERAL_SPECIAL_TOKENS, add_special_tokens=False)["input_ids"]) | {0}

    for max_length in (6, 1024):
        pieces = [p["input_ids"].tolist() for p in TokenizeTransform(tokenizer, max_length).map({"text": LITERAL_SPECIAL_TOKENS})]
        # The real ones the packer needs are there, each in its place
        assert all(piece[0] == 1 for piece in pieces)
        assert pieces[-1][-1] == 2
        inner = [t for piece in pieces[:-1] for t in piece[1:]] + pieces[-1][1:-1]
        assert not specials & set(inner), inner
        assert len(inner) == 6 + 3 + 3 + 3 + 3 + 3


def test_literal_special_tokens_in_flan_documents_are_text():
    from src.mmbjax.training.packing import FLANTokenizeTransform

    tokenizer = _tokenizer_with_the_pieces_of_special_tokens()
    ids = FLANTokenizeTransform(tokenizer, 64).map({"inputs": "a <s> </s>", "targets": "b <pad> <mask> <unk>"})["input_ids"].tolist()
    assert ids[0] == 1
    assert ids.count(1) == 1 and ids.count(2) == 1, "only the <s> and the separator it adds"
    assert not {0, 3, 4} & set(ids)


def test_literal_special_tokens_through_the_packed_pipeline():
    """The route of training and of the held-out batches."""
    from src.mmbjax.training import create_packed_dataset_streaming

    tokenizer = _tokenizer_with_the_pieces_of_special_tokens()
    documents = [{"text": LITERAL_SPECIAL_TOKENS} for _ in range(8)]
    batch = next(iter(create_packed_dataset_streaming(documents, tokenizer=tokenizer, max_length=8, batch_size=4)))
    for ids, positions in zip(batch["input_ids"], batch["input_ids_positions"]):
        padding = (ids == 3) & (positions == 0)
        assert not padding[:1].any()
        real = ids[~padding]
        assert not (real == 3).any() and not (real == 0).any() and not (real == 4).any()
        # <s> is at the start of each segment and nowhere else
        assert np.array_equal(real == 1, positions[~padding] == 0)


def test_too_short_max_length_is_an_error():
    from src.mmbjax.training.packing import split_into_pieces

    with pytest.raises(ValueError, match="no room"):
        split_into_pieces([5], 2, 1, 2)


def test_document_far_longer_than_the_limit_is_kept_whole():
    tokenizer = _left_truncating_tokenizer()
    transform = TokenizeTransform(tokenizer, max_length=16)
    text = " ".join(f"w{i % 95}" for i in range(100_000))
    pieces = [p["input_ids"].tolist() for p in transform.map({"text": text})]

    assert sum(len(piece) - 1 for piece in pieces) - 1 == 100_000
    assert [t for piece in pieces for t in piece if t not in (1, 2)] == tokenizer(text, add_special_tokens=False)["input_ids"]
    assert pieces[0][1] == tokenizer("w0", add_special_tokens=False)["input_ids"][0], "starts at the start of the document"


def test_truncation_settings_of_the_tokenizer_do_not_change_the_pieces():
    text = " ".join(f"w{i % 95}" for i in range(200))

    def pieces_with(side, preset_truncation=False):
        tokenizer = _left_truncating_tokenizer()
        tokenizer.truncation_side = side
        if preset_truncation:
            tokenizer.backend_tokenizer.enable_truncation(5, direction=side)
        return [p["input_ids"].tolist() for p in TokenizeTransform(tokenizer, max_length=16).map({"text": text})]

    left = pieces_with("left")
    assert left == pieces_with("right")
    assert left == pieces_with("left", preset_truncation=True)
    assert left == pieces_with("right", preset_truncation=True)
    assert len(left) == 14


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


def _segments(batch):
    """The packed segments of a batch as token tuples, as the training step gets them apart."""
    found = []
    for ids, positions in zip(batch["input_ids"], batch["input_ids_positions"]):
        starts = [i for i, p in enumerate(positions) if p == 0 and ids[i] == 1]
        for start, end in zip(starts, starts[1:] + [len(ids)]):
            found.append(tuple(t for t in ids[start:end].tolist() if t != 3))
    return found


def _long_documents_loader(monkeypatch, documents, window=None, **kwargs):
    """The real loader on a real shuffled datasets.IterableDataset of documents of all lengths; only the download is fake."""
    import datasets

    from src.mmbjax.training import EpochTrackingDataLoader, packing

    fake = datasets.IterableDataset.from_generator(lambda: iter(documents))
    monkeypatch.setattr(packing, "load_dataset", lambda *_a, **_k: fake)
    if window is not None:
        monkeypatch.setattr(packing, "PIECE_SHUFFLE_WINDOW", window)
    return EpochTrackingDataLoader(
        dataset_name="fake", tokenizer=_left_truncating_tokenizer(), max_length=16, batch_size=4,
        shuffle_buffer_size=7, **kwargs,
    )


def _documents_of_all_lengths(n=60):
    """Random words, of 1 to 90 tokens: a piece of some length tells its document."""
    rng = np.random.RandomState(0)
    return [
        {"text": " ".join(f"w{w}" for w in rng.randint(0, 95, size=int(rng.choice([1, 3, 14, 15, 16, 30, 31, 90]))))}
        for _ in range(n)
    ]


def _pieces_expected(documents):
    transform = TokenizeTransform(_left_truncating_tokenizer(), max_length=16)
    return [tuple(p["input_ids"].tolist()) for document in documents for p in transform.map(document)]


@pytest.mark.parametrize("window", [1, 6, 5000])
def test_resume_in_the_middle_of_a_document_loses_and_repeats_no_piece(monkeypatch, window):
    documents = _documents_of_all_lengths()
    transform = TokenizeTransform(_left_truncating_tokenizer(), max_length=16)
    # Full pieces only: those can't be the same by chance
    document_of_piece = {
        tuple(p["input_ids"].tolist()): i
        for i, document in enumerate(documents) for p in transform.map(document) if p["length"] == 16
    }
    assert len(document_of_piece) > 20, "the documents must split or this says nothing"

    whole = _long_documents_loader(monkeypatch, documents, window=window, num_epochs=1)
    all_batches = list(whole)

    documents_cut_by_a_kill = 0
    for kill_at in range(1, len(all_batches), 3):
        first = _long_documents_loader(monkeypatch, documents, window=window, num_epochs=1)
        iterator = iter(first)
        trained = [next(iterator) for _ in range(kill_at)]
        state = json.loads(json.dumps(first.get_state(), default=int))
        iterator.close()

        resumed = _long_documents_loader(monkeypatch, documents, window=window, num_epochs=1)
        resumed.set_state(state)
        rest = list(resumed)

        for got, wanted in zip(trained + rest, all_batches):
            for key in wanted:
                assert np.array_equal(got[key], wanted[key]), (kill_at, key)
        assert len(trained) + len(rest) == len(all_batches)
        assert resumed.pieces_read == whole.pieces_read

        documents_before = {document_of_piece.get(seg) for batch in trained for seg in _segments(batch)} - {None}
        documents_after = {document_of_piece.get(seg) for batch in rest for seg in _segments(batch)} - {None}
        documents_cut_by_a_kill += len(documents_before & documents_after)
    assert documents_cut_by_a_kill > 0, "no kill fell in the middle of a document"


def test_every_piece_once_per_pass_and_the_counters_are_right(monkeypatch):
    from collections import Counter

    documents = _documents_of_all_lengths()
    expected = _pieces_expected(documents)

    loader = _long_documents_loader(monkeypatch, documents, num_epochs=1)
    seen = []
    for batch in loader:
        seen += _segments(batch)
    assert len(seen) == loader.pieces_read
    # Only rows of the last, partial batch are dropped, nothing comes twice
    missing = Counter(expected) - Counter(seen)
    assert not Counter(seen) - Counter(expected)
    assert sum(missing.values()) <= 4 * 16

    # Documents stay documents and a pass stays a pass, however many pieces they make
    assert loader.documents_read == len(documents)
    assert loader.epoch == 1
    assert loader.pieces_read > loader.documents_read


def test_pieces_of_a_document_are_spread_over_batches(monkeypatch):
    """One document of 500 pieces among short ones: without a window it takes over whole batches."""
    long_document = {"text": " ".join(f"w{i % 95}" for i in range(500 * 15))}
    documents = [{"text": f"w{i % 95} w{(i + 1) % 95}"} for i in range(2000)] + [long_document]

    def most_in_one_batch(window):
        loader = _long_documents_loader(monkeypatch, documents, window=window, num_epochs=1)
        return max(sum(len(segment) >= 15 for segment in _segments(batch)) for batch in loader)

    assert most_in_one_batch(1) == 4
    assert most_in_one_batch(5000) < 4


def test_multi_dataset_route_splits_too_and_counts_documents(monkeypatch):
    import datasets

    from src.mmbjax.training import EpochTrackingDataLoader, packing

    text_documents = _documents_of_all_lengths()
    flan_documents = [{"inputs": "w1 w2", "targets": "w3 w4"} for _ in range(30)]
    streams = {"text": text_documents, "flan": flan_documents}
    monkeypatch.setattr(
        packing, "load_dataset",
        lambda name, *_a, **_k: datasets.IterableDataset.from_generator(lambda: iter(streams[name])),
    )
    loader = EpochTrackingDataLoader(
        dataset_configs=[{"name": "text"}, {"name": "flan", "format": "inputs_targets"}],
        tokenizer=_left_truncating_tokenizer(), max_length=16, batch_size=4, shuffle_buffer_size=7,
        stopping_strategy="all_exhausted", num_epochs=1,
    )
    seen = [segment for batch in loader for segment in _segments(batch)]

    assert set(seen) <= set(_pieces_expected(text_documents)) | {(1, 6, 7, 2, 8, 9)}
    assert len(seen) == loader.pieces_read
    assert loader.documents_read == len(text_documents) + len(flan_documents)


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


def test_a_resume_after_any_batch_of_three_passes_continues_with_the_next(tmp_path):
    """mC4 nl takes 2.21 passes in 653,000 steps: a checkpoint can fall in any pass, also on its last batch."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from src.mmbjax.training import EpochTrackingDataLoader

    files = []
    for part in range(3):
        path = tmp_path / f"part{part}.parquet"
        pq.write_table(pa.table({"text": [f"w{part} w{i % 95}" for i in range(40)]}), path)
        files.append(str(path))

    def loader():
        return EpochTrackingDataLoader(dataset_name="ignored", tokenizer=_left_truncating_tokenizer(), max_length=16,
                                       batch_size=2, shuffle_buffer_size=10, data_files=files)

    whole = loader()
    iterator = iter(whole)
    batches, epochs, counts = [], [], []
    while whole.epoch < 3:
        batches.append(next(iterator))
        epochs.append(whole.epoch)
        counts.append(json.dumps(whole.accounting()))
    batches.pop(), epochs.pop(), counts.pop()  # the first batch of the fourth pass
    last_of_pass = [i + 1 for i in range(len(epochs) - 1) if epochs[i + 1] != epochs[i]]
    assert len(last_of_pass) == 2, "the batches must span three passes"
    assert not np.array_equal(batches[0]["input_ids"], batches[last_of_pass[0]]["input_ids"]), "each pass has its own order"

    for kill_at in range(1, len(batches)):
        first = loader()
        iterator = iter(first)
        for _ in range(kill_at):
            next(iterator)
        state = json.loads(json.dumps(first.get_state(), default=int))
        iterator.close()

        resumed = loader()
        resumed.set_state(state)
        iterator = iter(resumed)
        for i in range(kill_at, len(batches)):
            batch = next(iterator)
            for key in batches[i]:
                assert np.array_equal(batch[key], batches[i][key]), (kill_at, i, key)
            assert (resumed.epoch, json.dumps(resumed.accounting())) == (epochs[i], counts[i]), (kill_at, i)


# --------------------------------------------------------------------------- --masking

GOLDEN_UNIFORM15 = "8b98bdaf1f5ac53837e87b468f5a3a9d0a66783539e87ec3e8e26cc7c687b687"
GOLDEN_UNIFORM15_NEXT_DRAW = 1395185717


def _masking_tokenizer(mock_tokenizer):
    return mock_tokenizer(pad_token_id=0, bos_token_id=1, eos_token_id=2, mask_token_id=3,
                          all_special_ids=[0, 1, 2, 3], vocab_size=1000)


def _packed_batch(rows=16, length=256, seed=0, longest=120):
    """What Grain's packing hands the collator: <s> tokens </s> per document, with padding in segment 0 at position 0."""
    rng = np.random.RandomState(seed)
    input_ids = np.zeros((rows, length), dtype=np.int32)
    segment_ids = np.zeros((rows, length), dtype=np.int32)
    positions = np.zeros((rows, length), dtype=np.int32)
    for row in range(rows):
        at, segment = 0, 0
        while True:
            n = rng.randint(1, longest)
            if at + n + 2 > length:
                break
            input_ids[row, at:at + n + 2] = np.r_[1, rng.randint(10, 1000, size=n), 2]
            segment_ids[row, at:at + n + 2] = segment
            positions[row, at:at + n + 2] = np.arange(n + 2)
            at += n + 2
            segment += 1
    return {"input_ids": input_ids, "input_ids_segment_ids": segment_ids, "input_ids_positions": positions}


def _documents_of_row(batch, row):
    """(start, end) of each document in a row of a raw batch."""
    ids = batch["input_ids"][row]
    starts = np.flatnonzero(ids == 1)
    ends = np.flatnonzero(ids == 2) + 1
    return list(zip(starts.tolist(), ends.tolist()))


def _digest(collated):
    import hashlib

    digest = hashlib.sha256()
    for key in sorted(collated):
        digest.update(key.encode())
        digest.update(collated[key].dtype.str.encode())
        digest.update(collated[key].tobytes())
    return digest.hexdigest()


def test_uniform15_collates_to_the_bytes_of_the_collator_before_the_flag(mock_tokenizer):
    """The digest was computed with packing.py of cabf5d4, the code of the running long run."""
    np.random.seed(42)
    collator = PackedMLMDataCollator(_masking_tokenizer(mock_tokenizer), mlm_probability=0.15)
    batches = [collator(_packed_batch(seed=seed)) for seed in range(3)]
    assert _digest({f"{i}{key}": batch[key] for i, batch in enumerate(batches) for key in batch}) == GOLDEN_UNIFORM15
    # And the RNG stands where it stood, for all that comes after
    assert np.random.randint(2**31) == GOLDEN_UNIFORM15_NEXT_DRAW


@pytest.mark.parametrize("denoiser", [i for i, name in enumerate(DENOISER_NAMES) if name.startswith("span")])
def test_span_denoiser_masks_its_rate_in_separate_spans_around_its_mean(denoiser):
    mean, rate = [(int(name.split("_")[1][2:]), int(name.split("_")[2][1:]) / 100) for name in [DENOISER_NAMES[denoiser]]][0]
    np.random.seed(denoiser)
    n = 2000
    spans, masked = [], 0
    for _ in range(200):
        chosen = MixedDenoiserCollator._draw_masked(denoiser, n)
        assert len(set(chosen.tolist())) == len(chosen) and chosen.min() >= 0 and chosen.max() < n
        assert np.all(np.diff(chosen) > 0)
        assert abs(len(chosen) - rate * n) <= 1
        masked += len(chosen)
        breaks = np.flatnonzero(np.diff(chosen) > 1) + 1
        spans += [len(span) for span in np.split(chosen, breaks)]
    assert masked / (200 * n) == pytest.approx(rate, abs=0.002)
    # The last span of a segment is cut to the budget, which pulls the mean down a little
    assert 0.8 * mean < np.mean(spans) < 1.1 * mean
    assert min(spans) >= 1


@pytest.mark.parametrize("n", [1, 2, 3, 5, 9, 17, 40, 63])
def test_a_short_segment_caps_the_span_and_keeps_the_rate(n):
    for denoiser, name in enumerate(DENOISER_NAMES):
        if not name.startswith("span"):
            continue
        rate = int(name.split("_")[2][1:]) / 100
        np.random.seed(n)
        sizes = [len(MixedDenoiserCollator._draw_masked(denoiser, n)) for _ in range(2000)]
        assert max(sizes) <= rate * n + 1
        assert np.mean(sizes) == pytest.approx(rate * n, abs=0.05 * max(rate * n, 1))
        assert max(sizes) < n or n == 1, "a whole segment masked leaves nothing to predict from"


def test_no_maskable_tokens_is_no_error():
    for denoiser in range(len(DENOISER_NAMES)):
        assert len(MixedDenoiserCollator._draw_masked(denoiser, 0)) == 0


def test_short_prefix_masks_one_block_of_1_to_16_at_a_uniform_cut():
    prefix = DENOISER_NAMES.index("short_prefix")
    np.random.seed(0)
    draws = [MixedDenoiserCollator._draw_masked(prefix, 100) for _ in range(20000)]
    assert all(np.all(np.diff(chosen) == 1) for chosen in draws)
    # Blocks at the end of the segment are cut short, the others are 1 to 16 with equal chance
    far_from_the_end = [len(chosen) for chosen in draws if chosen[0] < 100 - 16]
    assert set(far_from_the_end) == set(range(1, 17))
    assert np.mean(far_from_the_end) == pytest.approx(8.5, abs=0.1)
    assert np.mean([chosen[0] for chosen in draws]) == pytest.approx(49.5, abs=1)
    assert {chosen[0] for chosen in draws} == set(range(100))
    assert all(chosen[-1] < 100 for chosen in draws)


def test_mix_draws_are_fixed_by_the_seed(mock_tokenizer):
    tokenizer = _masking_tokenizer(mock_tokenizer)

    def run(seed):
        np.random.seed(seed)
        collator = MixedDenoiserCollator(tokenizer)
        return [_digest(collator(_packed_batch(seed=i))) for i in range(3)], collator.segments.tolist()

    assert run(1) == run(1)
    assert run(1)[0] != run(2)[0]


def test_mix_gives_each_denoiser_an_equal_chance_and_logs_its_share(mock_tokenizer):
    np.random.seed(5)
    collator = MixedDenoiserCollator(_masking_tokenizer(mock_tokenizer))
    for seed in range(100):
        collator(_packed_batch(seed=seed))
    assert collator.segments.sum() > 5000
    assert np.all(np.abs(collator.segments / collator.segments.sum() - 1 / 8) < 0.02)

    stats = collator.denoiser_stats()
    assert set(stats) == set(DENOISER_NAMES) and len(stats) == 8
    assert sum(s["masked_share"] for s in stats.values()) == pytest.approx(1.0)
    assert stats["single_r15"]["masked_rate"] == pytest.approx(0.15, abs=0.01)
    for name, s in stats.items():
        if name.startswith("span"):
            assert s["masked_rate"] == pytest.approx(int(name.split("_")[2][1:]) / 100, abs=0.02), name
    # What makes this mix top-heavy: r=0.5 on 3 of 8 segments gives most of the masked tokens, so most of a mean loss
    high = sum(s["masked_share"] for name, s in stats.items() if name.endswith("_r50"))
    assert high > 0.6
    assert stats["short_prefix"]["hidden_tokens"] > 0


def test_mix_never_masks_specials_and_keeps_the_80_10_10(mock_tokenizer):
    np.random.seed(11)
    collator = MixedDenoiserCollator(_masking_tokenizer(mock_tokenizer))
    outcome = {"mask": 0, "kept": 0, "random": 0}
    for seed in range(60):
        raw = _packed_batch(seed=seed)
        batch = collator(raw)
        original = raw["input_ids"]
        assert np.all(batch["labels"][np.isin(original, [0, 1, 2, 3])] == -100)
        # Specials stay what they are, unless a short prefix hid them
        special = np.isin(original, [0, 1, 2])
        assert np.all((batch["input_ids"][special] == original[special]) | (batch["attention_mask"][special] == 0))
        assert np.all(batch["input_ids"][original == 1] == 1), "<s> is before any masked block, so never hidden"
        masked = batch["labels"] != -100
        assert np.all(batch["labels"][masked] == original[masked])
        outcome["mask"] += int(np.sum(batch["input_ids"][masked] == 3))
        outcome["kept"] += int(np.sum(batch["input_ids"][masked] == original[masked]))
        outcome["random"] += int(np.sum((batch["input_ids"][masked] != 3) & (batch["input_ids"][masked] != original[masked])))
    total = sum(outcome.values())
    assert outcome["mask"] / total == pytest.approx(0.8, abs=0.01)
    assert outcome["kept"] / total == pytest.approx(0.1, abs=0.01)
    assert outcome["random"] / total == pytest.approx(0.1, abs=0.01)


def _short_prefix_only(monkeypatch):
    """Every segment gets the short prefix; what happens to the RNG doesn't change."""
    real_randint = np.random.randint
    prefix = DENOISER_NAMES.index("short_prefix")

    def randint(low, high=None, **kwargs):
        value = real_randint(low, high, **kwargs)
        return prefix if low == len(DENOISER_NAMES) and high is None and not kwargs else value

    monkeypatch.setattr(np.random, "randint", randint)


def test_short_prefix_leaves_nothing_behind_its_block_in_the_batch(mock_tokenizer, monkeypatch):
    _short_prefix_only(monkeypatch)
    np.random.seed(3)
    collator = MixedDenoiserCollator(_masking_tokenizer(mock_tokenizer))
    raw = _packed_batch(seed=4)
    batch = collator(raw)
    hidden_somewhere = False
    for row in range(raw["input_ids"].shape[0]):
        for start, end in _documents_of_row(raw, row):
            masked = np.flatnonzero(batch["labels"][row, start:end] != -100)
            assert len(masked) >= 1 and 1 <= len(masked) <= 16
            assert np.all(np.diff(masked) == 1)
            last = start + masked[-1]
            # Up to the block: the document as it was
            assert np.array_equal(batch["input_ids"][row, start:start + masked[0]], raw["input_ids"][row, start:start + masked[0]])
            assert np.all(batch["segment_ids"][row, start:last + 1] == raw["input_ids_segment_ids"][row, start])
            assert np.all(batch["attention_mask"][row, start:last + 1] == 1)
            assert np.array_equal(batch["position_ids"][row, start:last + 1], raw["input_ids_positions"][row, start:last + 1])
            # Behind it: the same as padding, in every array, and no target
            assert np.all(batch["input_ids"][row, last + 1:end] == 0)
            assert np.all(batch["attention_mask"][row, last + 1:end] == 0)
            assert np.all(batch["segment_ids"][row, last + 1:end] == -1)
            assert np.all(batch["position_ids"][row, last + 1:end] == 0)
            assert np.all(batch["labels"][row, last + 1:end] == -100)
            hidden_somewhere |= last + 1 < end - 1
    assert hidden_somewhere


def test_other_tokens_behind_the_prefix_make_the_same_batch(mock_tokenizer, monkeypatch):
    """Behind the block the collator's output can't depend on the document, so the model can't either."""
    _short_prefix_only(monkeypatch)
    tokenizer = _masking_tokenizer(mock_tokenizer)
    raw = _packed_batch(seed=4)
    np.random.seed(3)
    first = MixedDenoiserCollator(tokenizer)(raw)

    other = {key: value.copy() for key, value in raw.items()}
    changed = (first["attention_mask"] == 0) & (raw["input_ids"] >= 10)
    assert changed.sum() > 100
    other["input_ids"][changed] = 10 + (other["input_ids"][changed] + 7) % 900
    np.random.seed(3)
    second = MixedDenoiserCollator(tokenizer)(other)
    assert _digest(first) == _digest(second)


def test_model_sees_nothing_of_the_tokens_behind_the_prefix():
    """On the model itself, with local and global layers: hidden positions take any token id, without effect on any other position."""
    config = MMBertConfig(
        vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=4, num_attention_heads=2,
        max_position_embeddings=64, local_attention=8, global_attn_every_n_layers=2, pad_token_id=0,
    )
    model = ModernBertForMaskedLM(config)
    # Two documents; the first is cut behind its masked block at position 9, the second is padding-free up to 24
    input_ids = jnp.array([[1, 5, 6, 7, 8, 9, 10, 3, 3, 3, 11, 12, 13, 14, 15, 2, 1, 21, 22, 23, 24, 25, 26, 2, 0, 0]], dtype=jnp.int32)
    segment_ids = jnp.array([[0] * 16 + [1] * 8 + [-1] * 2], dtype=jnp.int32)
    position_ids = jnp.array([list(range(16)) + list(range(8)) + [0, 0]], dtype=jnp.int32)
    attention_mask = jnp.array([[1] * 24 + [0, 0]], dtype=jnp.int32)
    hidden_from, hidden_to = 10, 16

    variables = model.init(jr.PRNGKey(0), input_ids=input_ids, attention_mask=attention_mask, deterministic=True)

    def logits(ids, mask, segments, positions):
        return np.asarray(model.apply(variables, input_ids=ids, attention_mask=mask, segment_ids=segments,
                                      position_ids=positions, deterministic=True).logits)

    hidden_mask = attention_mask.at[0, hidden_from:hidden_to].set(0)
    hidden_segments = segment_ids.at[0, hidden_from:hidden_to].set(-1)
    hidden_positions = position_ids.at[0, hidden_from:hidden_to].set(0)
    base = logits(input_ids, hidden_mask, hidden_segments, hidden_positions)
    other_ids = input_ids.at[0, hidden_from:hidden_to].set(jnp.array([40, 41, 42, 43, 44, 45]))
    other = logits(other_ids, hidden_mask, hidden_segments, hidden_positions)

    seen = np.r_[0:hidden_from, hidden_to:24]
    assert np.array_equal(base[0, seen], other[0, seen])
    assert np.all(np.isfinite(base[0, seen]))
    # A position of a hidden token can't reach them either
    moved = logits(input_ids, hidden_mask, hidden_segments, hidden_positions.at[0, hidden_from:hidden_to].set(50))
    assert np.array_equal(base[0, seen], moved[0, seen])

    # Negative controls: each of the two arrays hides them on its own, and with neither the same tokens do show
    only_mask = logits(other_ids, hidden_mask, segment_ids, position_ids), logits(input_ids, hidden_mask, segment_ids, position_ids)
    assert np.array_equal(only_mask[0][0, seen], only_mask[1][0, seen])
    only_segments = logits(other_ids, attention_mask, hidden_segments, position_ids), logits(input_ids, attention_mask, hidden_segments, position_ids)
    assert np.array_equal(only_segments[0][0, seen], only_segments[1][0, seen])
    visible = logits(other_ids, attention_mask, segment_ids, position_ids), logits(input_ids, attention_mask, segment_ids, position_ids)
    assert not np.array_equal(visible[0][0, :hidden_from], visible[1][0, :hidden_from])
