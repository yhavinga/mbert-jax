"""Training utilities for mmbjax.

This module provides data loading and preprocessing utilities for training
ModernBERT models, including sequence packing with Google's Grain library.
"""

from .packing import (
    GRAIN_AVAILABLE,
    PIECE_SHUFFLE_WINDOW,
    EpochTrackingDataLoader,
    HybridMaskingCollator,
    InstructionResponseMaskingCollator,
    PackedMLMDataCollator,
    StreamingBufferDataSource,
    TokenizeTransform,
    create_packed_dataloader,
    create_packed_dataset_streaming,
)

__all__ = [
    "GRAIN_AVAILABLE",
    "PIECE_SHUFFLE_WINDOW",
    "EpochTrackingDataLoader",
    "HybridMaskingCollator",
    "InstructionResponseMaskingCollator",
    "PackedMLMDataCollator",
    "StreamingBufferDataSource",
    "TokenizeTransform",
    "create_packed_dataloader",
    "create_packed_dataset_streaming",
]

