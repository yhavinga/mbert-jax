"""Training utilities for mmbjax.

This module provides data loading and preprocessing utilities for training
ModernBERT models, including sequence packing with Google's Grain library.
"""

from .packing import (
    GRAIN_AVAILABLE,
    EpochTrackingDataLoader,
    PackedMLMDataCollator,
    StreamingBufferDataSource,
    TokenizeTransform,
    create_packed_dataloader,
    create_packed_dataset_streaming,
    load_interleaved_streaming_dataset,
)

__all__ = [
    "GRAIN_AVAILABLE",
    "EpochTrackingDataLoader",
    "PackedMLMDataCollator",
    "StreamingBufferDataSource",
    "TokenizeTransform",
    "create_packed_dataloader",
    "create_packed_dataset_streaming",
    "load_interleaved_streaming_dataset",
]

