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
    "FLANTokenizeTransform",
    "PackedMLMDataCollator",
    "InstructionResponseMaskingCollator",
    "HybridMaskingCollator",
    "StreamingBufferDataSource",
    "EpochTrackingDataLoader",
    "create_packed_dataloader",
    "create_packed_dataset_streaming",
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


class FLANTokenizeTransform:
    """Tokenize FLAN-format (inputs, targets) with separator for instruction-response masking.

    Format: [inputs] </s> [targets] → 2 segments per example for differential masking.
    """

    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.sep_token_id = tokenizer.eos_token_id  # </s> token
        import threading
        self._lock = threading.Lock()

    def map(self, element: Dict[str, Any]) -> Dict[str, Any]:
        """Tokenize FLAN format (inputs, targets) with separator.

        The dataset has 'inputs' (instruction) and 'targets' (response) fields.
        We concatenate them with EOS token as separator.

        Note: We reserve 1 token for the separator, so inputs+targets must fit in max_length-1.
        """
        with self._lock:  # Thread-safe tokenization for fast tokenizers
            # Reserve 1 token for the separator
            max_tokens_per_part = (self.max_length - 1) // 2

            # Tokenize inputs (instruction)
            inputs = self.tokenizer(
                element["inputs"],
                truncation=True,
                max_length=max_tokens_per_part,
                return_tensors=None,
                add_special_tokens=False,
            )["input_ids"]

            # Calculate remaining space for targets
            remaining_space = self.max_length - len(inputs) - 1  # -1 for separator

            # Tokenize targets (response) with remaining space
            targets = self.tokenizer(
                element["targets"],
                truncation=True,
                max_length=remaining_space,
                return_tensors=None,
                add_special_tokens=False,
            )["input_ids"]

        # Concatenate: inputs + </s> + targets
        input_ids = inputs + [self.sep_token_id] + targets

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


class InstructionResponseMaskingCollator:
    """MLM collator with instruction/response masking for FLAN-style data.

    Uses position-based heuristic to identify instruction vs response tokens:
    - Scans each segment for first </s> (EOS token, id=2)
    - Tokens before </s> → instruction part → mask_prob = 0.0 (no masking)
    - Tokens after </s> → response part → mask_prob = 0.25 (harder training)

    This works even when Grain splits sequences across packing boundaries.
    """

    def __init__(
        self,
        tokenizer,
        instruction_mask_prob: float = 0.0,
        response_mask_prob: float = 0.25,
        sep_token_id: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.instruction_mask_prob = instruction_mask_prob
        self.response_mask_prob = response_mask_prob
        # Default to EOS token (</s> for LLaMA tokenizers)
        self.sep_token_id = sep_token_id if sep_token_id is not None else tokenizer.eos_token_id

    def __call__(self, batch: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Process a batch from Grain's FirstFitPackIterDataset."""
        input_ids = np.array(batch["input_ids"])
        segment_ids = np.array(batch["input_ids_segment_ids"])
        positions = np.array(batch["input_ids_positions"])

        # Fix padding token segment_id (same as PackedMLMDataCollator)
        is_padding = (input_ids == self.tokenizer.pad_token_id)
        segment_ids = np.where(is_padding, -1, segment_ids)

        # Apply segment-aware MLM masking
        input_ids, labels = self.mask_tokens_packed(input_ids, segment_ids)

        # CRITICAL: attention_mask must be 0 for padding tokens
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
        """Apply position-aware MLM masking based on instruction/response segments.

        For each segment:
        1. Find first `</s>` token position
        2. Tokens before `</s>` → instruction masking (0% by default)
        3. Tokens after `</s>` → response masking (25% by default)
        4. No `</s>` found → treat as instruction (0% masking)
        """
        labels = input_ids.copy()
        probability_matrix = np.zeros(input_ids.shape, dtype=np.float32)

        # Process each batch row independently
        # (segment IDs are only meaningful within a single sequence)
        for batch_idx in range(input_ids.shape[0]):
            row_input_ids = input_ids[batch_idx]
            row_segment_ids = segment_ids[batch_idx]

            for seg_id in np.unique(row_segment_ids):
                if seg_id == -1:  # Skip padding
                    continue

                # 1D mask for this segment within this row
                seg_mask = (row_segment_ids == seg_id)
                seg_tokens = row_input_ids[seg_mask]

                # Find first </s> position in this segment
                sep_positions = np.where(seg_tokens == self.sep_token_id)[0]

                if len(sep_positions) == 0:
                    # No separator → entire segment is instruction
                    probability_matrix[batch_idx, seg_mask] = self.instruction_mask_prob
                else:
                    # Has separator - check if it's truly FLAN format or just trailing EOS
                    first_sep_idx = sep_positions[0]
                    # Get column indices where this segment is located
                    seg_col_indices = np.where(seg_mask)[0]

                    # Check if there are tokens AFTER the separator
                    response_cols = seg_col_indices[first_sep_idx + 1:]
                    
                    if len(response_cols) == 0:
                        # Separator is at the end (no response) → treat as instruction
                        probability_matrix[batch_idx, seg_mask] = self.instruction_mask_prob
                    else:
                        # Has tokens after separator → true FLAN format → differential masking
                        # Tokens before </s> → instruction (lower/no masking)
                        instruction_cols = seg_col_indices[:first_sep_idx]
                        probability_matrix[batch_idx, instruction_cols] = self.instruction_mask_prob

                        # Tokens after </s> → response (higher masking)
                        probability_matrix[batch_idx, response_cols] = self.response_mask_prob

        # Don't mask special tokens (including </s> separator)
        special_tokens_mask = np.array([
            [1 if tok in self.tokenizer.all_special_ids else 0 for tok in ids]
            for ids in input_ids.tolist()
        ])
        probability_matrix[special_tokens_mask.astype(bool)] = 0.0

        # Sample and apply masking (80% MASK, 10% random, 10% original)
        masked_indices = np.random.binomial(1, probability_matrix).astype(bool)
        labels[~masked_indices] = -100

        indices_replaced = (
            np.random.binomial(1, np.full(labels.shape, 0.8)).astype(bool)
            & masked_indices
        )
        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        indices_random = (
            np.random.binomial(1, np.full(labels.shape, 0.5)).astype(bool)
            & masked_indices & ~indices_replaced
        )
        random_words = np.random.randint(
            self.tokenizer.vocab_size, size=labels.shape, dtype=np.int32
        )
        input_ids[indices_random] = random_words[indices_random]

        return input_ids, labels


class HybridMaskingCollator:
    """Hybrid MLM collator that handles both standard and FLAN-style data in mixed batches.

    Uses position-based heuristic:
    - Segments WITH `</s>` (EOS token): 0% instruction / 25% response masking
    - Segments WITHOUT `</s>`: 15% standard MLM masking

    This enables mixing normal text datasets (MC4, web text) with instruction-tuning
    datasets (FLAN) in a single training run.

    The masking detection is automatic - no need to specify which datasets are which format.
    Just ensure FLAN datasets use the `inputs_targets` format (which adds `</s>` separator).
    """

    def __init__(
        self,
        tokenizer,
        standard_mask_prob: float = 0.15,
        instruction_mask_prob: float = 0.0,
        response_mask_prob: float = 0.25,
        sep_token_id: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.standard_mask_prob = standard_mask_prob
        self.instruction_mask_prob = instruction_mask_prob
        self.response_mask_prob = response_mask_prob
        self.sep_token_id = sep_token_id if sep_token_id is not None else tokenizer.eos_token_id

    def __call__(self, batch: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Process a batch from Grain's FirstFitPackIterDataset."""
        input_ids = np.array(batch["input_ids"])
        segment_ids = np.array(batch["input_ids_segment_ids"])
        positions = np.array(batch["input_ids_positions"])

        # Fix padding token segment_id
        is_padding = (input_ids == self.tokenizer.pad_token_id)
        segment_ids = np.where(is_padding, -1, segment_ids)

        # Apply hybrid MLM masking
        input_ids, labels = self.mask_tokens_packed(input_ids, segment_ids)

        # CRITICAL: attention_mask must be 0 for padding tokens
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
        """Apply hybrid MLM masking based on segment content.

        For each segment:
        - If `</s>` is found → FLAN format (0% instruction / 25% response)
        - If no `</s>` → Standard format (15% uniform)
        """
        labels = input_ids.copy()
        probability_matrix = np.zeros(input_ids.shape, dtype=np.float32)

        # Process each batch row independently
        # (segment IDs are only meaningful within a single sequence)
        for batch_idx in range(input_ids.shape[0]):
            row_input_ids = input_ids[batch_idx]
            row_segment_ids = segment_ids[batch_idx]

            for seg_id in np.unique(row_segment_ids):
                if seg_id == -1:  # Skip padding
                    continue

                # 1D mask for this segment within this row
                seg_mask = (row_segment_ids == seg_id)
                seg_tokens = row_input_ids[seg_mask]

                # Find first </s> position in this segment
                sep_positions = np.where(seg_tokens == self.sep_token_id)[0]

                if len(sep_positions) == 0:
                    # No separator → standard text dataset → 15% masking
                    probability_matrix[batch_idx, seg_mask] = self.standard_mask_prob
                else:
                    # Has separator - check if it's truly FLAN format or just trailing EOS
                    first_sep_idx = sep_positions[0]
                    # Get column indices where this segment is located
                    seg_col_indices = np.where(seg_mask)[0]
                    
                    # Check if there are tokens AFTER the separator
                    response_cols = seg_col_indices[first_sep_idx + 1:]
                    
                    if len(response_cols) == 0:
                        # Separator is at the end (no response) → standard text → 15% masking
                        # This happens when TokenizeTransform adds EOS at the end
                        probability_matrix[batch_idx, seg_mask] = self.standard_mask_prob
                    else:
                        # Has tokens after separator → true FLAN format → differential masking
                        # Tokens before </s> → instruction (0% masking)
                        instruction_cols = seg_col_indices[:first_sep_idx]
                        probability_matrix[batch_idx, instruction_cols] = self.instruction_mask_prob
                        
                        # Tokens after </s> → response (25% masking)
                        probability_matrix[batch_idx, response_cols] = self.response_mask_prob

        # Don't mask special tokens (including </s> separator)
        special_tokens_mask = np.array([
            [1 if tok in self.tokenizer.all_special_ids else 0 for tok in ids]
            for ids in input_ids.tolist()
        ])
        probability_matrix[special_tokens_mask.astype(bool)] = 0.0

        # Sample and apply masking (80% MASK, 10% random, 10% original)
        masked_indices = np.random.binomial(1, probability_matrix).astype(bool)
        labels[~masked_indices] = -100

        indices_replaced = (
            np.random.binomial(1, np.full(labels.shape, 0.8)).astype(bool)
            & masked_indices
        )
        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        indices_random = (
            np.random.binomial(1, np.full(labels.shape, 0.5)).astype(bool)
            & masked_indices & ~indices_replaced
        )
        random_words = np.random.randint(
            self.tokenizer.vocab_size, size=labels.shape, dtype=np.int32
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
    # Dataset format parameter
    dataset_format: str = "text",
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
            Each config is a dict with keys: name, config, weight, format (optional)
        shuffle_buffer_size: Buffer size for streaming shuffle
        stopping_strategy: "first_exhausted" or "all_exhausted"
        dataset_format: Dataset format - "text" or "inputs_targets" (single-dataset mode only)

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
        dataset_format=dataset_format,
    )


def create_packed_dataset_streaming(
    stream_dataset,
    tokenizer,
    max_length: int,
    batch_size: int,
    dataset_format: str = "text",
):
    """Create a packed IterDataset from a streaming dataset.

    Uses StreamingBufferDataSource to lazily load from the stream while
    providing the RandomAccessDataSource interface that Grain requires.

    Note: Buffer grows unbounded as items are accessed. For large datasets,
    memory will grow over time. This is required because Grain's shuffling
    needs random access to previously-seen items.

    Args:
        stream_dataset: Any Python iterable yielding dicts with "text" field
            (for format="text") or "inputs"/"targets" fields (for format="inputs_targets")
        tokenizer: Tokenizer with eos_token_id
        max_length: Maximum sequence length for packing
        batch_size: Number of packing bins (sequences per batch)
        dataset_format: "text" for standard text datasets, "inputs_targets" for FLAN-style

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

    # Create tokenizer transform based on dataset format
    if dataset_format == "inputs_targets":
        tokenize = FLANTokenizeTransform(tokenizer, max_length)
    else:  # Default to "text" format
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
        # Dataset format parameter
        dataset_format: str = "text",
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
                Each config is a dict with keys: name, config, weight, format (optional)
            shuffle_buffer_size: Buffer size for streaming shuffle
            stopping_strategy: "first_exhausted" or "all_exhausted"
            dataset_format: Dataset format - "text" or "inputs_targets" (single-dataset mode)
        """
        # Determine mode
        self.multi_dataset_mode = dataset_configs is not None

        if self.multi_dataset_mode:
            self.dataset_configs = dataset_configs
            self.dataset_name = None
            self.dataset_config = None
            self.dataset_format = None  # Not used in multi-dataset mode
        else:
            if dataset_name is None:
                raise ValueError("Must provide either dataset_name or dataset_configs")
            self.dataset_name = dataset_name
            self.dataset_config = dataset_config
            self.dataset_configs = None
            self.dataset_format = dataset_format

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
        """Create a fresh packed data pipeline for one epoch.

        For multi-dataset mode with different formats:
        1. Load each dataset separately
        2. Apply format-specific tokenization
        3. Interleave tokenized streams
        4. Pack the interleaved stream
        """
        if grain is None or grain_exp is None:
            raise ImportError("Grain is not available")

        if self.multi_dataset_mode:
            # Multi-dataset mode: process each dataset separately, then interleave
            return self._create_multi_dataset_pipeline()
        else:
            # Single-dataset mode: use the specified format
            stream_dataset = self._load_stream_dataset()
            return create_packed_dataset_streaming(
                stream_dataset,
                tokenizer=self.tokenizer,
                max_length=self.max_length,
                batch_size=self.batch_size,
                dataset_format=self.dataset_format,
            )

    def _create_multi_dataset_pipeline(self):
        """Create a data pipeline for multi-dataset mode with format support.

        Each dataset is tokenized according to its format, then the tokenized
        streams are interleaved and packed together.
        """
        epoch_seed = self.base_seed + self.epoch

        # Load and tokenize each dataset separately
        tokenized_streams = []
        weights = []

        for cfg in self.dataset_configs:
            # Get dataset format (default to "text")
            dataset_format = cfg.get("format", "text")

            # Load the streaming dataset
            stream = load_dataset(
                cfg["name"],
                cfg.get("config"),
                split="train",
                streaming=True,
                download_mode="reuse_cache_if_exists",
                token=True,
            )

            # Apply sharding if needed
            if self.shard_count > 1:
                try:
                    from datasets.distributed import split_dataset_by_node
                    stream = split_dataset_by_node(
                        stream, rank=self.shard_index, world_size=self.shard_count
                    )
                except ImportError:
                    # Fallback to manual strided iteration
                    original_stream = stream
                    def sharded_iter():
                        for i, item in enumerate(original_stream):
                            if i % self.shard_count == self.shard_index:
                                yield item
                    stream = sharded_iter()

            # Shuffle
            if hasattr(stream, 'shuffle'):
                stream = stream.shuffle(seed=epoch_seed, buffer_size=self.shuffle_buffer_size)

            # Wrap as buffer source for Grain
            source = StreamingBufferDataSource(stream)

            # Create appropriate tokenizer transform
            if dataset_format == "inputs_targets":
                tokenize = FLANTokenizeTransform(self.tokenizer, self.max_length)
            else:
                tokenize = TokenizeTransform(self.tokenizer, self.max_length)

            # Create tokenized map dataset
            # Disable prefetching to avoid thread safety issues with fast tokenizers
            # (HuggingFace fast tokenizers use Rust and aren't thread-safe)
            map_dataset = grain.MapDataset.source(source).map(tokenize.map)
            iter_dataset = map_dataset.to_iter_dataset(
                read_options=grain.ReadOptions(num_threads=1, prefetch_buffer_size=0)
            )

            tokenized_streams.append(iter_dataset)
            weights.append(cfg.get("weight", 1.0))

        # Normalize weights
        total_weight = sum(weights)
        probabilities = [w / total_weight for w in weights]

        # Interleave tokenized streams using probability sampling
        # (grain.MixIterDataset doesn't exist in all Grain versions)
        class InterleavedIterDataset:
            """Simple interleaving iterator that samples from multiple streams."""
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
                    # Check stopping condition
                    if self.stopping_strategy == "first_exhausted":
                        if any(exhausted):
                            break
                    else:  # all_exhausted
                        if all(exhausted):
                            break

                    # Sample which dataset to draw from (skip exhausted ones)
                    available_probs = [
                        p if not e else 0.0
                        for p, e in zip(self.probabilities, exhausted)
                    ]
                    total = sum(available_probs)
                    if total == 0:
                        break
                    normalized = [p / total for p in available_probs]

                    # Weighted random choice
                    r = rng.random()
                    cumulative = 0.0
                    chosen_idx = 0
                    for i, prob in enumerate(normalized):
                        cumulative += prob
                        if r <= cumulative:
                            chosen_idx = i
                            break

                    # Try to get next item from chosen iterator
                    try:
                        item = next(iterators[chosen_idx])
                        yield item
                    except StopIteration:
                        exhausted[chosen_idx] = True

        interleaved = InterleavedIterDataset(
            parents=tokenized_streams,
            probabilities=probabilities,
            seed=epoch_seed,
            stopping_strategy=self.stopping_strategy,
        )

        # Wrap interleaved iterator as a proper Grain data source
        # FirstFitPackIterDataset expects a Grain IterDataset, not a plain Python iterator
        interleaved_source = StreamingBufferDataSource(iter(interleaved))
        interleaved_map = grain.MapDataset.source(interleaved_source).map(lambda x: x)
        interleaved_iter = interleaved_map.to_iter_dataset(
            read_options=grain.ReadOptions(num_threads=1, prefetch_buffer_size=0)
        )

        # Apply packing to the interleaved tokenized stream
        packed_dataset = grain_exp.FirstFitPackIterDataset(
            parent=interleaved_iter,
            length_struct={"input_ids": self.max_length},
            num_packing_bins=self.batch_size,
            shuffle_bins=True,
            padding_struct={"input_ids": self.tokenizer.pad_token_id},
        )

        # Batch the packed sequences
        batched_dataset = packed_dataset.batch(self.batch_size)
        return batched_dataset
    
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

