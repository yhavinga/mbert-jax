"""Sequence packing utilities for efficient MLM training with Grain."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Set HuggingFace Hub timeout before any HF imports - TPU networks can be slow
os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '120')

# Grain imports are optional - only needed when --use-packing is enabled
try:
    import grain.python as grain
    from grain.python import experimental as grain_exp
    from datasets import load_dataset, interleave_datasets
    GRAIN_AVAILABLE = True
except ImportError:
    GRAIN_AVAILABLE = False
    # Create stubs for type checking
    grain = None  # type: ignore
    grain_exp = None  # type: ignore


__all__ = [
    "TokenizeTransform",
    "PackedMLMDataCollator",
    "StreamingBufferDataSource",
    "EpochTrackingDataLoader",
    "create_packed_dataloader",
    "create_packed_dataset_streaming",
    "load_interleaved_streaming_dataset",
    "GRAIN_AVAILABLE",
]


class StreamingBufferDataSource:
    """A data source that lazily fills from a stream.
    
    Acts as a RandomAccessDataSource (has __len__ and __getitem__) but fetches
    from stream on-demand. This allows Grain's MapDataset.source() to work with
    streaming data while maintaining bounded memory usage.
    
    The buffer grows as items are accessed, capped at max_buffer_size. For truly
    infinite streams, use max_buffer_size to limit memory usage.
    
    Thread-safe: Grain's prefetch uses multiple threads, so we protect stream access.
    
    Usage:
        stream = load_dataset(..., streaming=True)
        source = StreamingBufferDataSource(stream, max_buffer_size=100000)
        dataset = grain.MapDataset.source(source).map(tokenize)
    """
    
    def __init__(self, stream, max_buffer_size: Optional[int] = None, stream_length: Optional[int] = None):
        """Initialize with a stream and optional buffer limit.
        
        Args:
            stream: Any Python iterable (HF streaming dataset, generator, etc.)
            max_buffer_size: Maximum items to buffer. None = buffer everything.
                            For large datasets, set this to limit memory.
            stream_length: Expected number of items in stream. If None, assumes
                          very large (for streaming datasets). Used by Grain to
                          know when dataset is "exhausted".
        """
        import threading
        self._stream = iter(stream)
        self._buffer = []
        self._max_buffer_size = max_buffer_size
        self._exhausted = False
        self._stream_position = 0  # Track how far we've read
        self._lock = threading.Lock()  # Thread-safety for Grain's prefetch
        # For streaming datasets with unknown length, use a very large number
        # so Grain doesn't stop early. 10B is enough for any pretraining run.
        self._stream_length = stream_length if stream_length is not None else 10_000_000_000
    
    def _ensure_available(self, idx: int) -> None:
        """Fetch from stream until idx is available. Thread-safe.
        
        Note: Buffer grows unbounded because Grain's shuffling requires random
        access to all previously seen items. For large datasets, memory will grow
        over time - this is required for the packing + shuffling pipeline.
        """
        with self._lock:
            while idx >= len(self._buffer) and not self._exhausted:
                try:
                    item = next(self._stream)
                    self._stream_position += 1
                    self._buffer.append(item)
                except StopIteration:
                    self._exhausted = True
                    break
    
    def __len__(self) -> int:
        """Return expected stream length for Grain.
        
        For streaming datasets, returns a very large number so Grain keeps
        fetching. The actual data will be served via __getitem__ as long as
        the stream has items.
        """
        with self._lock:
            if self._exhausted:
                return len(self._buffer)
            return self._stream_length
    
    def __getitem__(self, idx: int):
        """Get item at index, fetching from stream if needed. Thread-safe.
        
        Handles race condition: if Grain's prefetch thread calls __len__ before
        exhaustion (returns 10B) but __getitem__ is called after exhaustion,
        we wrap the index to avoid IndexError. This effectively enables dataset
        cycling for finite streams.
        """
        self._ensure_available(idx)
        with self._lock:
            if idx < len(self._buffer):
                return self._buffer[idx]
            # Stream exhausted and idx >= buffer size: wrap around for cycling
            if self._exhausted and len(self._buffer) > 0:
                wrapped_idx = idx % len(self._buffer)
                return self._buffer[wrapped_idx]
            raise IndexError(f'Index {idx} out of range (buffer size: {len(self._buffer)})')


def _stack_batch(batch: list) -> Dict[str, np.ndarray]:
    """Stack a list of dicts into a single dict of stacked arrays."""
    if not batch:
        return {}
    keys = batch[0].keys()
    return {
        key: np.stack([item[key] for item in batch], axis=0)
        for key in keys
    }


class TokenizeTransform:
    """Tokenize text for sequence packing (variable length).

    Note: This class extends grain.MapTransform when Grain is available.
    We use composition over inheritance to avoid import issues.
    """

    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def map(self, element: Dict[str, Any]) -> Dict[str, Any]:
        """Tokenize a single text element."""
        # Tokenize with truncation at max_length
        tokens = self.tokenizer(
            element["text"],
            truncation=True,
            max_length=self.max_length,
            return_tensors=None,
            add_special_tokens=False,
        )
        input_ids = tokens["input_ids"]
        
        # Only add EOS if the document wasn't truncated
        # If it was truncated, the text didn't actually end here - adding EOS would be misleading
        # Grain's packing still works correctly: segment boundaries are tracked separately
        was_truncated = len(input_ids) >= self.max_length
        if not was_truncated:
            input_ids = input_ids + [self.tokenizer.eos_token_id]
        
        # Grain's FirstFitPackIterDataset requires numpy arrays, not Python lists
        return {
            "input_ids": np.array(input_ids, dtype=np.int32),
            "length": len(input_ids),
        }


class PackedMLMDataCollator:
    """
    MLM collator for packed sequences from Grain.
    Handles segment_ids and position_ids for block-diagonal attention.
    """

    def __init__(self, tokenizer, mlm_probability: float = 0.15):
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability

    def __call__(self, batch: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Process a batch from Grain's FirstFitPackIterDataset.
        batch contains:
        - input_ids: (batch_size, max_length) - packed token IDs
        - input_ids_segment_ids: (batch_size, max_length) - which doc each token belongs to
        - input_ids_positions: (batch_size, max_length) - position within original doc
        """
        input_ids = np.array(batch["input_ids"])
        segment_ids = np.array(batch["input_ids_segment_ids"])
        positions = np.array(batch["input_ids_positions"])

        # CRITICAL FIX: Padding tokens from Grain have segment_id=0, which conflicts with
        # real document tokens that also have segment_id=0. This causes padding to attend
        # to real content and vice versa, corrupting training gradients.
        # Fix: Assign padding tokens a unique segment_id (-1) that won't match any real segment.
        is_padding = (input_ids == self.tokenizer.pad_token_id)
        segment_ids = np.where(is_padding, -1, segment_ids)

        # Apply MLM masking AFTER packing
        input_ids, labels = self.mask_tokens_packed(input_ids, segment_ids)

        # CRITICAL: attention_mask must be 0 for padding tokens!
        # With all 1s, model attends to garbage padding values, corrupting representations
        attention_mask = np.where(is_padding, 0, 1).astype(np.int32)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "position_ids": positions,
            "segment_ids": segment_ids,
        }

    def mask_tokens_packed(
        self, input_ids: np.ndarray, segment_ids: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply 15% MLM masking, respecting special tokens."""
        labels = input_ids.copy()

        # Create mask - same probability for all tokens
        probability_matrix = np.full(input_ids.shape, self.mlm_probability)

        # Don't mask special tokens (pad, bos, eos, cls, sep, mask)
        special_tokens_mask = np.array([
            [
                1 if tok in self.tokenizer.all_special_ids else 0
                for tok in ids
            ]
            for ids in input_ids.tolist()
        ])
        probability_matrix[special_tokens_mask.astype(bool)] = 0.0

        # Sample positions to mask
        masked_indices = np.random.binomial(1, probability_matrix).astype(bool)
        labels[~masked_indices] = -100

        # 80% MASK, 10% random, 10% original
        indices_replaced = (
            np.random.binomial(1, np.full(labels.shape, 0.8)).astype(bool)
            & masked_indices
        )
        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        indices_random = (
            np.random.binomial(1, np.full(labels.shape, 0.5)).astype(bool)
            & masked_indices
            & ~indices_replaced
        )
        random_words = np.random.randint(
            self.tokenizer.vocab_size,
            size=labels.shape,
            dtype=np.int32
        )
        input_ids[indices_random] = random_words[indices_random]

        return input_ids, labels


def create_packed_dataloader(
    dataset_name: Optional[str] = None,
    dataset_config: Optional[str] = None,
    tokenizer=None,
    max_length: int = 512,
    batch_size: int = 16,
    num_workers: int = 0,
    shard_index: int = 0,
    shard_count: int = 1,
    seed: int = 42,
    num_epochs: Optional[int] = None,
    # Multi-dataset mode parameters
    dataset_configs: Optional[List[Dict[str, Any]]] = None,
    shuffle_buffer_size: int = 10000,
    stopping_strategy: str = "first_exhausted",
):
    """Create Grain dataset iterator with sequence packing and epoch restart.

    Uses EpochTrackingDataLoader to automatically restart the data pipeline
    when the source dataset is exhausted, enabling multi-epoch training.
    
    Can be called in single-dataset mode (dataset_name) or multi-dataset
    mode (dataset_configs).

    Args:
        dataset_name: HuggingFace dataset name (e.g., "yhavinga/mc4_nl_cleaned")
        dataset_config: HuggingFace dataset config (e.g., "micro")
        tokenizer: Tokenizer with eos_token_id, all_special_ids, mask_token_id, vocab_size
        max_length: Maximum sequence length for packing
        batch_size: Number of sequences per batch (also num_packing_bins for Grain)
        num_workers: Number of workers (unused, kept for API compatibility)
        shard_index: Index of this shard for multi-host training
        shard_count: Total number of shards
        seed: Random seed for shuffling (epoch number is added for per-epoch variation)
        num_epochs: Number of epochs to run (None = infinite, auto-restart forever)
        dataset_configs: List of dataset configs for multi-dataset mode.
            Each config is a dict with keys: name, config, weight
        shuffle_buffer_size: Buffer size for streaming shuffle
        stopping_strategy: "first_exhausted" or "all_exhausted"

    Returns:
        EpochTrackingDataLoader with sequence packing and epoch tracking
    """
    if grain is None:
        raise ImportError(
            "Grain is not installed. Install it with: pip install grain"
        )

    if dataset_configs is not None:
        # Multi-dataset mode
        print(f"Creating EpochTrackingDataLoader with {len(dataset_configs)} interleaved datasets:")
        for cfg in dataset_configs:
            config_str = f" ({cfg.get('config')})" if cfg.get('config') else ""
            weight_pct = cfg.get('weight', 1.0) * 100 / sum(c.get('weight', 1.0) for c in dataset_configs)
            print(f"  - {cfg['name']}{config_str}: {weight_pct:.1f}%")
    else:
        # Single-dataset mode
        config_str = f" ({dataset_config})" if dataset_config else " (no config)"
        print(f"Creating EpochTrackingDataLoader for: {dataset_name}{config_str}")
    print(f"  Epochs: {'infinite' if num_epochs is None else num_epochs}")
    print(f"  Shuffle buffer: {shuffle_buffer_size}")
    
    return EpochTrackingDataLoader(
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        tokenizer=tokenizer,
        max_length=max_length,
        batch_size=batch_size,
        shard_index=shard_index,
        shard_count=shard_count,
        seed=seed,
        num_epochs=num_epochs,
        dataset_configs=dataset_configs,
        shuffle_buffer_size=shuffle_buffer_size,
        stopping_strategy=stopping_strategy,
    )


def create_packed_dataset_streaming(
    stream_dataset,
    tokenizer,
    max_length: int,
    batch_size: int,
):
    """Create a packed IterDataset from a streaming dataset.

    Uses StreamingBufferDataSource to lazily load from the stream while
    providing the RandomAccessDataSource interface that Grain requires.
    
    Note: Buffer grows unbounded as items are accessed. For large datasets,
    memory will grow over time. This is required because Grain's shuffling
    needs random access to previously-seen items.

    Args:
        stream_dataset: Any Python iterable yielding dicts with "text" field
        tokenizer: Tokenizer with eos_token_id
        max_length: Maximum sequence length for packing
        batch_size: Number of packing bins (sequences per batch)

    Returns:
        IterDataset that yields packed batches
    """
    if grain is None or grain_exp is None:
        raise ImportError(
            "Grain is not available. Install with: pip install grain"
        )

    # Wrap stream as a lazy-loading RandomAccessDataSource
    # No buffer limit - Grain needs random access to all items for shuffling
    source = StreamingBufferDataSource(stream_dataset)
    
    # Create tokenizer transform
    tokenize = TokenizeTransform(tokenizer, max_length)

    # Build Grain pipeline: source -> tokenize -> to_iter -> pack -> batch
    map_dataset = grain.MapDataset.source(source).map(tokenize.map)
    iter_dataset = map_dataset.to_iter_dataset()

    # Apply packing - FirstFitPackIterDataset yields one packed sequence at a time
    # num_packing_bins controls parallel packing efficiency, not output batch size
    packed_dataset = grain_exp.FirstFitPackIterDataset(
        parent=iter_dataset,
        length_struct={"input_ids": max_length},
        num_packing_bins=batch_size,
        shuffle_bins=True,
        padding_struct={"input_ids": tokenizer.pad_token_id},
    )

    # Batch the packed sequences to get (batch_size, max_length) shaped tensors
    batched_dataset = packed_dataset.batch(batch_size)

    return batched_dataset


def load_interleaved_streaming_dataset(
    dataset_configs: List[Dict[str, Any]],
    shard_index: int = 0,
    shard_count: int = 1,
    seed: int = 42,
    shuffle_buffer_size: int = 10000,
    stopping_strategy: str = "first_exhausted",
):
    """Load multiple datasets and interleave them with probability weights.
    
    Uses HuggingFace's interleave_datasets() to combine multiple streaming
    datasets into a single stream with configurable sampling probabilities.
    
    Args:
        dataset_configs: List of dicts with keys:
            - name: HuggingFace dataset name
            - config: Dataset config/subset (or None)
            - weight: Sampling probability (weights are normalized automatically)
        shard_index: Index of this shard for multi-host training
        shard_count: Total number of shards
        seed: Random seed for shuffling
        shuffle_buffer_size: Buffer size for streaming shuffle
        stopping_strategy: "first_exhausted" or "all_exhausted"
    
    Returns:
        Interleaved IterableDataset with shuffling applied
    
    Example:
        >>> configs = [
        ...     {"name": "yhavinga/mc4_nl_cleaned", "config": "micro", "weight": 0.5},
        ...     {"name": "uonlp/CulturaX", "config": "nl", "weight": 0.25},
        ... ]
        >>> dataset = load_interleaved_streaming_dataset(configs)
    """
    if not GRAIN_AVAILABLE:
        raise ImportError("Grain and datasets must be installed for interleaving")
    
    # Load each dataset as streaming
    datasets_list = []
    weights = []
    
    for cfg in dataset_configs:
        ds = load_dataset(
            cfg["name"],
            cfg.get("config"),
            split="train",
            streaming=True,
            download_mode="reuse_cache_if_exists",
            token=True,
        )
        datasets_list.append(ds)
        weights.append(cfg.get("weight", 1.0))
    
    # Normalize weights to probabilities
    total_weight = sum(weights)
    probabilities = [w / total_weight for w in weights]
    
    # Interleave datasets with probability weights
    interleaved = interleave_datasets(
        datasets_list,
        probabilities=probabilities,
        seed=seed,
        stopping_strategy=stopping_strategy,
    )
    
    # Apply sharding for multi-host training
    if shard_count > 1:
        try:
            from datasets.distributed import split_dataset_by_node
            interleaved = split_dataset_by_node(
                interleaved, rank=shard_index, world_size=shard_count
            )
        except ImportError:
            # Fallback: manual strided iteration
            original = interleaved
            def sharded_iter():
                for i, item in enumerate(original):
                    if i % shard_count == shard_index:
                        yield item
            interleaved = sharded_iter()
    
    # Apply shuffle buffer
    if hasattr(interleaved, 'shuffle'):
        interleaved = interleaved.shuffle(seed=seed, buffer_size=shuffle_buffer_size)
    
    return interleaved


class EpochTrackingDataLoader:

    """Wrapper that auto-restarts the data pipeline on epoch boundaries.
    
    When the underlying HuggingFace streaming dataset is exhausted, this class:
    1. Logs the epoch completion
    2. Recreates a fresh data pipeline for the next epoch
    3. Continues yielding batches seamlessly
    
    This enables multi-epoch training without buffering the entire dataset in memory.
    
    Attributes:
        epoch: Current epoch number (0-indexed, incremented after each full pass)
        num_epochs: Maximum epochs to run (None = infinite)
    """
    
    def __init__(
        self,
        dataset_name: Optional[str] = None,
        dataset_config: Optional[str] = None,
        tokenizer=None,
        max_length: int = 512,
        batch_size: int = 16,
        shard_index: int = 0,
        shard_count: int = 1,
        seed: int = 42,
        num_epochs: Optional[int] = None,
        # Multi-dataset mode parameters
        dataset_configs: Optional[List[Dict[str, Any]]] = None,
        shuffle_buffer_size: int = 10000,
        stopping_strategy: str = "first_exhausted",
    ):
        """Initialize the epoch-tracking data loader.
        
        Can be initialized in single-dataset mode (dataset_name + dataset_config)
        or multi-dataset mode (dataset_configs list).
        
        Args:
            dataset_name: HuggingFace dataset name (single-dataset mode)
            dataset_config: Dataset configuration/subset (single-dataset mode)
            tokenizer: Tokenizer for text encoding
            max_length: Maximum sequence length for packing
            batch_size: Number of sequences per batch
            shard_index: Index of this shard for multi-host training
            shard_count: Total number of shards
            seed: Random seed (epoch number is added for per-epoch variation)
            num_epochs: Maximum epochs (None = infinite)
            dataset_configs: List of dataset configs for multi-dataset mode.
                Each config is a dict with keys: name, config, weight
            shuffle_buffer_size: Buffer size for streaming shuffle
            stopping_strategy: "first_exhausted" or "all_exhausted"
        """
        # Determine mode
        self.multi_dataset_mode = dataset_configs is not None
        
        if self.multi_dataset_mode:
            self.dataset_configs = dataset_configs
            self.dataset_name = None
            self.dataset_config = None
        else:
            if dataset_name is None:
                raise ValueError("Must provide either dataset_name or dataset_configs")
            self.dataset_name = dataset_name
            self.dataset_config = dataset_config
            self.dataset_configs = None
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size
        self.shard_index = shard_index
        self.shard_count = shard_count
        self.base_seed = seed
        self.num_epochs = num_epochs
        self.shuffle_buffer_size = shuffle_buffer_size
        self.stopping_strategy = stopping_strategy
        self.epoch = 0
        self._current_iter = None
    
    def _load_stream_dataset(self):
        """Load a fresh HuggingFace streaming dataset for one epoch."""
        epoch_seed = self.base_seed + self.epoch
        
        if self.multi_dataset_mode:
            # Multi-dataset mode: use interleaved loader
            return load_interleaved_streaming_dataset(
                dataset_configs=self.dataset_configs,
                shard_index=self.shard_index,
                shard_count=self.shard_count,
                seed=epoch_seed,
                shuffle_buffer_size=self.shuffle_buffer_size,
                stopping_strategy=self.stopping_strategy,
            )
        
        # Single-dataset mode (original behavior)
        # reuse_cache_if_exists avoids timeout checking metadata for large datasets
        # token=True for gated datasets like CulturaX (use_auth_token is deprecated)
        stream_dataset = load_dataset(
            self.dataset_name,
            self.dataset_config,
            split="train",
            streaming=True,
            download_mode="reuse_cache_if_exists",
            token=True,
        )
        
        # Apply sharding for multi-host training
        if self.shard_count > 1:
            try:
                from datasets.distributed import split_dataset_by_node
                stream_dataset = split_dataset_by_node(
                    stream_dataset, rank=self.shard_index, world_size=self.shard_count
                )
            except ImportError:
                # Fallback: manual strided iteration
                original_dataset = stream_dataset
                shard_index = self.shard_index
                shard_count = self.shard_count
                def sharded_iter():
                    for i, item in enumerate(original_dataset):
                        if i % shard_count == shard_index:
                            yield item
                stream_dataset = sharded_iter()
        
        # Shuffle with epoch-specific seed for different order each epoch
        if hasattr(stream_dataset, 'shuffle'):
            stream_dataset = stream_dataset.shuffle(seed=epoch_seed, buffer_size=self.shuffle_buffer_size)
        
        return stream_dataset
    
    def _create_pipeline(self):
        """Create a fresh packed data pipeline for one epoch."""
        stream_dataset = self._load_stream_dataset()
        return create_packed_dataset_streaming(
            stream_dataset,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            batch_size=self.batch_size,
        )
    
    def __iter__(self):
        """Iterate through batches, auto-restarting on epoch boundaries."""
        while self.num_epochs is None or self.epoch < self.num_epochs:
            pipeline = self._create_pipeline()
            try:
                for batch in pipeline:
                    yield batch
            except StopIteration:
                pass  # Normal end of epoch
            
            # Epoch completed
            self.epoch += 1
            print(f"[EpochTrackingDataLoader] Completed epoch {self.epoch} "
                  f"(shard {self.shard_index}/{self.shard_count})")
        
        # If num_epochs is set, we stop here
        print(f"[EpochTrackingDataLoader] Finished {self.num_epochs} epochs")

