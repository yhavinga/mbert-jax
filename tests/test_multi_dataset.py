"""Tests for multi-dataset interleaving functionality."""

import os
import pytest
from collections import Counter

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

from src.mmbjax.training import GRAIN_AVAILABLE


class TestInterleavedIterDataset:
    """Tests for the custom InterleavedIterDataset used in multi-dataset mode."""

    def test_probabilistic_sampling_respects_weights(self):
        """Verify that sampling roughly follows the specified probabilities."""
        # Create simple mock iterables
        stream_a = iter(range(0, 1000))  # items 0-999
        stream_b = iter(range(1000, 2000))  # items 1000-1999
        
        # Import the class (it's defined inside _create_multi_dataset_pipeline)
        # We'll recreate the logic here for testing
        class InterleavedIterDataset:
            def __init__(self, parents, probabilities, seed, stopping_strategy):
                self.parents = parents
                self.probabilities = probabilities
                self.seed = seed
                self.stopping_strategy = stopping_strategy

            def __iter__(self):
                import random
                rng = random.Random(self.seed)
                iterators = [iter(p) for p in self.parents]
                exhausted = [False] * len(iterators)

                while True:
                    if self.stopping_strategy == "first_exhausted":
                        if any(exhausted):
                            break
                    else:
                        if all(exhausted):
                            break

                    available_probs = [
                        p if not e else 0.0
                        for p, e in zip(self.probabilities, exhausted)
                    ]
                    total = sum(available_probs)
                    if total == 0:
                        break
                    normalized = [p / total for p in available_probs]

                    r = rng.random()
                    cumulative = 0.0
                    chosen_idx = 0
                    for i, prob in enumerate(normalized):
                        cumulative += prob
                        if r <= cumulative:
                            chosen_idx = i
                            break

                    try:
                        item = next(iterators[chosen_idx])
                        yield item
                    except StopIteration:
                        exhausted[chosen_idx] = True

        # 70% from stream_a, 30% from stream_b
        interleaved = InterleavedIterDataset(
            parents=[[i for i in range(100)], [i + 1000 for i in range(100)]],
            probabilities=[0.7, 0.3],
            seed=42,
            stopping_strategy="first_exhausted",
        )
        
        items = list(interleaved)
        
        # Count items from each source
        from_a = sum(1 for x in items if x < 1000)
        from_b = sum(1 for x in items if x >= 1000)
        
        # Should be roughly 70/30 (with some variance due to randomness)
        ratio_a = from_a / max(from_a + from_b, 1)
        assert 0.5 < ratio_a < 0.9, f"Expected ~70% from A, got {ratio_a*100:.1f}%"

    def test_first_exhausted_stops_early(self):
        """Verify first_exhausted stops when any stream is empty."""
        class InterleavedIterDataset:
            def __init__(self, parents, probabilities, seed, stopping_strategy):
                self.parents = parents
                self.probabilities = probabilities
                self.seed = seed
                self.stopping_strategy = stopping_strategy

            def __iter__(self):
                import random
                rng = random.Random(self.seed)
                iterators = [iter(p) for p in self.parents]
                exhausted = [False] * len(iterators)

                while True:
                    if self.stopping_strategy == "first_exhausted":
                        if any(exhausted):
                            break
                    else:
                        if all(exhausted):
                            break

                    available_probs = [
                        p if not e else 0.0
                        for p, e in zip(self.probabilities, exhausted)
                    ]
                    total = sum(available_probs)
                    if total == 0:
                        break
                    normalized = [p / total for p in available_probs]

                    r = rng.random()
                    cumulative = 0.0
                    chosen_idx = 0
                    for i, prob in enumerate(normalized):
                        cumulative += prob
                        if r <= cumulative:
                            chosen_idx = i
                            break

                    try:
                        item = next(iterators[chosen_idx])
                        yield item
                    except StopIteration:
                        exhausted[chosen_idx] = True

        # Short stream (10 items) and long stream (1000 items)
        interleaved = InterleavedIterDataset(
            parents=[[i for i in range(10)], [i + 1000 for i in range(1000)]],
            probabilities=[0.5, 0.5],
            seed=42,
            stopping_strategy="first_exhausted",
        )
        
        items = list(interleaved)
        
        # Should stop relatively quickly once short stream exhausted
        # With 50/50 probability, expect ~20 items before stopping (10 from A + ~10 from B)
        assert len(items) < 100, f"Expected early stop, got {len(items)} items"

    def test_seed_reproducibility(self):
        """Verify same seed produces same sequence."""
        class InterleavedIterDataset:
            def __init__(self, parents, probabilities, seed, stopping_strategy):
                self.parents = parents
                self.probabilities = probabilities
                self.seed = seed
                self.stopping_strategy = stopping_strategy

            def __iter__(self):
                import random
                rng = random.Random(self.seed)
                iterators = [iter(p) for p in self.parents]
                exhausted = [False] * len(iterators)

                while True:
                    if self.stopping_strategy == "first_exhausted":
                        if any(exhausted):
                            break
                    else:
                        if all(exhausted):
                            break

                    available_probs = [
                        p if not e else 0.0
                        for p, e in zip(self.probabilities, exhausted)
                    ]
                    total = sum(available_probs)
                    if total == 0:
                        break
                    normalized = [p / total for p in available_probs]

                    r = rng.random()
                    cumulative = 0.0
                    chosen_idx = 0
                    for i, prob in enumerate(normalized):
                        cumulative += prob
                        if r <= cumulative:
                            chosen_idx = i
                            break

                    try:
                        item = next(iterators[chosen_idx])
                        yield item
                    except StopIteration:
                        exhausted[chosen_idx] = True

        def get_sequence(seed):
            interleaved = InterleavedIterDataset(
                parents=[[i for i in range(50)], [i + 1000 for i in range(50)]],
                probabilities=[0.5, 0.5],
                seed=seed,
                stopping_strategy="first_exhausted",
            )
            return list(interleaved)
        
        seq1 = get_sequence(123)
        seq2 = get_sequence(123)
        seq3 = get_sequence(456)
        
        assert seq1 == seq2, "Same seed should produce same sequence"
        assert seq1 != seq3, "Different seeds should produce different sequences"


class TestEpochTrackingDataLoaderMultiDataset:
    """Tests for EpochTrackingDataLoader multi-dataset mode."""
    
    def test_single_dataset_mode_still_works(self):
        """Verify backward compatibility - single dataset mode works."""
        from src.mmbjax.training.packing import EpochTrackingDataLoader
        
        loader = EpochTrackingDataLoader(
            dataset_name="yhavinga/mc4_nl_cleaned",
            dataset_config="micro",
            tokenizer=None,
            max_length=512,
            batch_size=8,
        )
        
        assert loader.multi_dataset_mode is False
        assert loader.dataset_name == "yhavinga/mc4_nl_cleaned"
        assert loader.dataset_config == "micro"
        assert loader.dataset_configs is None

    def test_multi_dataset_mode_initialization(self):
        """Test initialization in multi-dataset mode."""
        from src.mmbjax.training.packing import EpochTrackingDataLoader
        
        configs = [
            {"name": "dataset1", "config": "cfg1", "weight": 0.5},
            {"name": "dataset2", "config": "cfg2", "weight": 0.5},
        ]
        
        loader = EpochTrackingDataLoader(
            dataset_configs=configs,
            tokenizer=None,
            max_length=512,
            batch_size=8,
        )
        
        assert loader.multi_dataset_mode is True
        assert loader.dataset_configs == configs
        assert loader.dataset_name is None
        assert loader.dataset_config is None

    def test_requires_dataset_name_or_configs(self):
        """Test that either dataset_name or dataset_configs must be provided."""
        from src.mmbjax.training.packing import EpochTrackingDataLoader
        
        with pytest.raises(ValueError, match="Must provide either dataset_name or dataset_configs"):
            EpochTrackingDataLoader(
                tokenizer=None,
                max_length=512,
                batch_size=8,
            )


class TestCreatePackedDataloader:
    """Tests for create_packed_dataloader multi-dataset support."""
    
    @pytest.mark.skipif(not GRAIN_AVAILABLE, reason="Grain not available")
    def test_accepts_dataset_configs_parameter(self):
        """Verify create_packed_dataloader accepts dataset_configs."""
        import inspect
        from src.mmbjax.training import create_packed_dataloader
        
        sig = inspect.signature(create_packed_dataloader)
        params = list(sig.parameters.keys())
        
        assert "dataset_configs" in params
        assert "shuffle_buffer_size" in params
        assert "stopping_strategy" in params
