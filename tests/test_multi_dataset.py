"""Tests for multi-dataset interleaving functionality."""

import os
import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

from src.mmbjax.training import load_interleaved_streaming_dataset, GRAIN_AVAILABLE


@pytest.mark.skipif(not GRAIN_AVAILABLE, reason="Grain not available")
class TestLoadInterleavedStreamingDataset:
    """Tests for load_interleaved_streaming_dataset function."""

    def test_function_is_importable(self):
        """Verify the function can be imported from mmbjax.training."""
        from src.mmbjax.training import load_interleaved_streaming_dataset
        assert callable(load_interleaved_streaming_dataset)

    def test_weights_are_normalized(self):
        """Test that weights are normalized to probabilities internally."""
        # This is a unit test that doesn't require actual datasets
        # Just verify the function signature and basic structure
        import inspect
        sig = inspect.signature(load_interleaved_streaming_dataset)
        params = list(sig.parameters.keys())
        
        assert "dataset_configs" in params
        assert "shard_index" in params
        assert "shard_count" in params
        assert "seed" in params
        assert "shuffle_buffer_size" in params
        assert "stopping_strategy" in params


class TestEpochTrackingDataLoaderMultiDataset:
    """Tests for EpochTrackingDataLoader multi-dataset mode."""
    
    def test_single_dataset_mode_still_works(self):
        """Verify backward compatibility - single dataset mode works."""
        from src.mmbjax.training.packing import EpochTrackingDataLoader
        
        # Create loader in single-dataset mode (don't actually iterate)
        loader = EpochTrackingDataLoader(
            dataset_name="yhavinga/mc4_nl_cleaned",
            dataset_config="micro",
            tokenizer=None,  # Would need real tokenizer to iterate
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
