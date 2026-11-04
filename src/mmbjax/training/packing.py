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


# Items a source keeps for Grain's read threads (16, with a prefetch buffer of 500), and the
# most it holds on to after a resume reads the stream up to its index
SOURCE_BUFFER_ITEMS = 2048

# Pieces in the window that shuffles them before packing. Packing fills one batch from the elements
# that come next, so without this the pieces of a long document would share a batch.
PIECE_SHUFFLE_WINDOW = 5000

# Batches kept ready by a thread of their own, 54 seconds of training on the 16-chip VM. Reading a part
# of the parquet that isn't in the page cache held the pipeline up for 12 seconds in a measurement,
# and 256 batches take only 100 MB.
BATCH_PREFETCH = 256

# The span denoisers of Table 1 of UL2 (Tay et al. 2022, arXiv:2205.05131): (mean span length, corruption rate)
SPAN_DENOISERS = ((3, 0.15), (8, 0.15), (3, 0.5), (8, 0.5), (64, 0.15), (64, 0.5))
# UL2 says "a normal distribution with mean of µ" and no spread. A share of µ keeps µ=3 and µ=64 apart.
SPAN_LENGTH_STD_OVER_MEAN = 0.25
PREFIX_MAX_MASKED = 16
DENOISER_NAMES = ("single_r15", *(f"span_mu{mean}_r{round(rate * 100)}" for mean, rate in SPAN_DENOISERS), "short_prefix")

# --mix-heads: the MLM prediction head that predicts each denoiser's masked tokens. Head 0 also predicts the held-out
# masks (15% single tokens) and is the head an export keeps. UltraBERT (arXiv:2610.15028) found mask rates mixed under
# one head to behave like their mean rate, and a head per rate to do better; rate2 splits the mix by corruption.
MIX_HEADS = {
    "shared": {name: 0 for name in DENOISER_NAMES},
    "rate2": {
        "single_r15": 0, "span_mu3_r15": 0, "span_mu8_r15": 0, "span_mu64_r15": 0,
        "span_mu3_r50": 1, "span_mu8_r50": 1, "span_mu64_r50": 1, "short_prefix": 1,
    },
    "per-denoiser": {name: index for index, name in enumerate(DENOISER_NAMES)},
}


def mix_head_count(mix_heads: str) -> int:
    """Prediction heads of a --mix-heads choice."""
    return max(MIX_HEADS[mix_heads].values()) + 1


__all__ = [
    "PIECE_SHUFFLE_WINDOW",
    "split_into_pieces",
    "TokenizeTransform",
    "FLANTokenizeTransform",
    "PackedMLMDataCollator",
    "MixedDenoiserCollator",
    "DENOISER_NAMES",
    "MIX_HEADS",
    "mix_head_count",
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
    
    The pipeline reads every index once, in order give or take its prefetch threads, so with
    max_buffer_size an item is dropped once read and memory stays bounded. A resumed iterator
    starts far into the stream: the first index asked for reads the stream up to it, and of
    what comes before it only the last max_buffer_size items are kept.
    
    Past the end of the stream __getitem__ raises StopIteration, which Grain's prefetch
    passes on as the end of the iterator: the pass is over and the loader decides what next.
    
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
            max_buffer_size: See above. Must exceed Grain's prefetch buffer, because a later
                            index can arrive first. None = keep everything and allow reading
                            an item again.
            stream_length: Expected number of items in stream. If None, assumes
                          very large (for streaming datasets). Used by Grain to
                          know when dataset is "exhausted".
        """
        import threading
        self._stream = iter(stream)
        self._buffer = {}
        self._max_buffer_size = max_buffer_size
        self._exhausted = False
        self._stream_position = 0  # Track how far we've read
        self._keep_from = None  # Items before this index are not wanted, set by the first request
        self._lock = threading.Lock()  # Thread-safety for Grain's prefetch
        # For streaming datasets with unknown length, use a very large number
        # so Grain doesn't stop early. 10B is enough for any pretraining run.
        self._stream_length = stream_length if stream_length is not None else 10_000_000_000
    
    @property
    def stream_position(self) -> int:
        """Items read from the stream so far, including those still in prefetch."""
        return self._stream_position
    
    def _ensure_available(self, idx: int) -> None:
        """Fetch from stream until idx is available. Thread-safe."""
        with self._lock:
            if self._keep_from is None:
                self._keep_from = 0 if self._max_buffer_size is None else idx - self._max_buffer_size
            while idx >= self._stream_position and not self._exhausted:
                try:
                    item = next(self._stream)
                except StopIteration:
                    self._exhausted = True
                    break
                if self._stream_position >= self._keep_from:
                    self._buffer[self._stream_position] = item
                self._stream_position += 1
    
    def __len__(self) -> int:
        """Return expected stream length for Grain.
        
        For streaming datasets, returns a very large number so Grain keeps
        fetching. The actual data will be served via __getitem__ as long as
        the stream has items. It stays that number past the end of the stream:
        Grain takes indices modulo this length, and a smaller one would wrap them.
        """
        return self._stream_length
    
    def __getitem__(self, idx: int):
        """Get item at index, fetching from stream if needed. Thread-safe."""
        self._ensure_available(idx)
        with self._lock:
            if idx in self._buffer:
                return self._buffer[idx] if self._max_buffer_size is None else self._buffer.pop(idx)
            if self._exhausted and idx >= self._stream_position:
                raise StopIteration
            raise IndexError(
                f'Index {idx} is gone: stream position {self._stream_position}, '
                f'max_buffer_size {self._max_buffer_size}'
            )


def _require_bos_token(tokenizer) -> None:
    if getattr(tokenizer, "bos_token_id", None) is None:
        raise ValueError(
            f"{type(tokenizer).__name__} has no bos_token_id: packed documents must start with "
            "the token that inference puts first, and classifier_pooling=cls reads that token."
        )


def _stack_batch(batch: list) -> Dict[str, np.ndarray]:
    """Stack a list of dicts into a single dict of stacked arrays."""
    if not batch:
        return {}
    keys = batch[0].keys()
    return {
        key: np.stack([item[key] for item in batch], axis=0)
        for key in keys
    }


def split_into_pieces(token_ids: List[int], max_length: int, bos_token_id: int, eos_token_id: int) -> List[List[int]]:
    """A document's tokens, in order, in pieces of at most max_length tokens including specials.

    Every piece starts with <s>, because inference puts it first and classifier_pooling=cls reads
    it. Only the piece with the real end of the document ends with </s>. A last piece always has a
    token besides its specials: </s> alone carries nothing to learn from.
    """
    if max_length < 3:
        raise ValueError(f"max_length {max_length} has no room for <s>, a token and </s>")
    pieces = []
    start = 0
    room = max_length - 1
    while len(token_ids) - start > room - 1:
        # With exactly `room` tokens left, </s> wouldn't fit behind them: one of them moves on with it
        end = start + min(room, len(token_ids) - start - 1)
        pieces.append([bos_token_id] + token_ids[start:end])
        start = end
    pieces.append([bos_token_id] + token_ids[start:] + [eos_token_id])
    return pieces


class TokenizeTransform:
    """Tokenize text for sequence packing: a document becomes one or more pieces (variable length).

    map() does the work and belongs in Grain's read threads, as a long document takes long to
    tokenize. flat_map() then hands out its pieces one by one, and with a parent's checkpointable
    index plus the position in the document Grain can resume in the middle of a document.
    """

    def __init__(self, tokenizer, max_length: int):
        _require_bos_token(tokenizer)
        self.tokenizer = tokenizer
        self.max_length = max_length
        # A fast tokenizer drops a truncation setting on its first call, which mustn't happen with
        # read threads already using it.
        self._tokenize("")

    def _tokenize(self, text: str) -> List[int]:
        return self.tokenizer(
            text,
            truncation=False,
            return_tensors=None,
            add_special_tokens=False,
            split_special_tokens=True,  # A literal <s> in a document is text, not token id 1
        )["input_ids"]

    def map(self, element: Dict[str, Any]) -> List[Dict[str, Any]]:
        """The pieces of a single text element."""
        pieces = split_into_pieces(
            self._tokenize(element["text"]), self.max_length, self.tokenizer.bos_token_id, self.tokenizer.eos_token_id
        )
        # Grain's FirstFitPackIterDataset requires numpy arrays, not Python lists
        return [{"input_ids": np.array(piece, dtype=np.int32), "length": len(piece)} for piece in pieces]

    def flat_map(self, pieces: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return pieces


class FLANTokenizeTransform:
    """Tokenize FLAN-format (inputs, targets) with separator for instruction-response masking.

    Format: <s> [inputs] </s> [targets] → 2 segments per example for differential masking.
    """

    def __init__(self, tokenizer, max_length: int):
        _require_bos_token(tokenizer)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.sep_token_id = tokenizer.eos_token_id  # </s> token
        import threading
        self._lock = threading.Lock()

    def map(self, element: Dict[str, Any]) -> Dict[str, Any]:
        """Tokenize FLAN format (inputs, targets) with separator.

        The dataset has 'inputs' (instruction) and 'targets' (response) fields.
        We concatenate them with EOS token as separator.

        Note: We reserve 2 tokens for <s> and the separator, so inputs+targets must fit in max_length-2.
        """
        with self._lock:  # Thread-safe tokenization for fast tokenizers
            # Reserve 2 tokens for <s> and the separator
            max_tokens_per_part = (self.max_length - 2) // 2

            # Tokenize inputs (instruction)
            inputs = self.tokenizer(
                element["inputs"],
                truncation=True,
                max_length=max_tokens_per_part,
                return_tensors=None,
                add_special_tokens=False,
                split_special_tokens=True,  # A literal </s> mustn't pass for the separator
            )["input_ids"]

            # Calculate remaining space for targets
            remaining_space = self.max_length - len(inputs) - 2  # -2 for <s> and separator

            # Tokenize targets (response) with remaining space
            targets = self.tokenizer(
                element["targets"],
                truncation=True,
                max_length=remaining_space,
                return_tensors=None,
                add_special_tokens=False,
                split_special_tokens=True,
            )["input_ids"]

        # Concatenate: <s> + inputs + </s> + targets
        input_ids = [self.tokenizer.bos_token_id] + inputs + [self.sep_token_id] + targets

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
        probability_matrix[np.isin(input_ids, self.tokenizer.all_special_ids)] = 0.0

        # Sample positions to mask
        masked_indices = np.random.binomial(1, probability_matrix).astype(bool)
        labels[~masked_indices] = -100

        self._replace_masked_tokens(input_ids, masked_indices)

        return input_ids, labels

    def _replace_masked_tokens(self, input_ids: np.ndarray, masked_indices: np.ndarray) -> None:
        """80% MASK, 10% random, 10% original."""
        indices_replaced = (
            np.random.binomial(1, np.full(input_ids.shape, 0.8)).astype(bool)
            & masked_indices
        )
        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        indices_random = (
            np.random.binomial(1, np.full(input_ids.shape, 0.5)).astype(bool)
            & masked_indices
            & ~indices_replaced
        )
        random_words = np.random.randint(
            self.tokenizer.vocab_size,
            size=input_ids.shape,
            dtype=np.int32
        )
        input_ids[indices_random] = random_words[indices_random]


class MixedDenoiserCollator(PackedMLMDataCollator):
    """Each packed segment gets one of 8 denoisers, with equal chance: DENOISER_NAMES.

    All draws come from NumPy's global RNG, as in the parent, so a seed fixes them. The short prefix
    leaves the tokens behind its masked block out by turning them into padding, in every array the
    model reads: the batch then holds nothing of them, whatever the attention does.

    mix_heads (MIX_HEADS) other than "shared" adds head_ids to the batch: at every position of a segment the
    prediction head of that segment's denoiser, 0 at padding and hidden tokens. It takes no draws, so the masks stay
    those of "shared".
    """

    def __init__(self, tokenizer, mix_heads: str = "shared"):
        super().__init__(tokenizer, mlm_probability=0.15)
        heads = MIX_HEADS[mix_heads]
        self.head_of_denoiser = None if mix_heads == "shared" else np.array([heads[name] for name in DENOISER_NAMES], dtype=np.int32)
        self.segments = np.zeros(len(DENOISER_NAMES), dtype=np.int64)
        self.maskable_tokens = np.zeros(len(DENOISER_NAMES), dtype=np.int64)
        self.masked_tokens = np.zeros(len(DENOISER_NAMES), dtype=np.int64)
        self.hidden_tokens = np.zeros(len(DENOISER_NAMES), dtype=np.int64)

    def __call__(self, batch: Dict[str, Any]) -> Dict[str, np.ndarray]:
        input_ids = np.array(batch["input_ids"])
        segment_ids = np.array(batch["input_ids_segment_ids"])
        positions = np.array(batch["input_ids_positions"])

        # See the parent: padding comes with segment_id 0, which a real document also has
        is_padding = (input_ids == self.tokenizer.pad_token_id)
        segment_ids = np.where(is_padding, -1, segment_ids)

        maskable = ~np.isin(input_ids, self.tokenizer.all_special_ids)
        masked_indices = np.zeros(input_ids.shape, dtype=bool)
        hidden = np.zeros(input_ids.shape, dtype=bool)
        head_ids = np.zeros(input_ids.shape, dtype=np.int32)
        for row in range(input_ids.shape[0]):
            boundaries = np.flatnonzero(segment_ids[row, 1:] != segment_ids[row, :-1]) + 1
            for start, end in zip(np.r_[0, boundaries], np.r_[boundaries, input_ids.shape[1]]):
                if segment_ids[row, start] == -1:
                    continue
                candidates = start + np.flatnonzero(maskable[row, start:end])
                denoiser = np.random.randint(len(DENOISER_NAMES))
                chosen = candidates[self._draw_masked(denoiser, len(candidates))]
                masked_indices[row, chosen] = True
                if self.head_of_denoiser is not None:
                    head_ids[row, start:end] = self.head_of_denoiser[denoiser]
                if DENOISER_NAMES[denoiser] == "short_prefix" and len(chosen):
                    hidden[row, chosen[-1] + 1:end] = True
                    self.hidden_tokens[denoiser] += end - chosen[-1] - 1
                self.segments[denoiser] += 1
                self.maskable_tokens[denoiser] += len(candidates)
                self.masked_tokens[denoiser] += len(chosen)

        labels = np.where(masked_indices, input_ids, -100)
        input_ids[hidden] = self.tokenizer.pad_token_id
        segment_ids[hidden] = -1
        positions[hidden] = 0
        head_ids[hidden] = 0
        self._replace_masked_tokens(input_ids, masked_indices)
        attention_mask = np.where(is_padding | hidden, 0, 1).astype(np.int32)

        collated = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "position_ids": positions,
            "segment_ids": segment_ids,
        }
        if self.head_of_denoiser is not None:
            collated["head_ids"] = head_ids
        return collated

    @staticmethod
    def _draw_masked(denoiser: int, n: int) -> np.ndarray:
        """Which of a segment's n maskable tokens this denoiser masks, as ascending indices."""
        name = DENOISER_NAMES[denoiser]
        if name == "single_r15":
            return np.flatnonzero(np.random.binomial(1, 0.15, size=n))
        if name == "short_prefix":
            if n == 0:
                return np.arange(0)
            cut = np.random.randint(n)
            k = np.random.randint(1, PREFIX_MAX_MASKED + 1)
            return np.arange(cut, min(cut + k, n))

        mean, rate = SPAN_DENOISERS[denoiser - 1]
        # Rounded at random, or the short segments would all land on the same side of the rate
        budget = int(rate * n + np.random.random())
        if budget == 0:
            return np.arange(0)
        # At least 1 token each, so this many draws always reach the budget
        lengths = np.maximum(np.rint(np.random.normal(mean, SPAN_LENGTH_STD_OVER_MEAN * mean, size=budget)).astype(int), 1)
        ends = np.cumsum(lengths)
        spans = int(np.searchsorted(ends, budget)) + 1
        # The cap for a short segment: a span stops at the rate's budget, so µ=64 can't blank a segment of 40
        lengths = lengths[:spans]
        lengths[-1] = budget - (ends[spans - 2] if spans > 1 else 0)
        # Spans keep an unmasked token between them, or two neighbours would be one longer span. That
        # token is always there: budget <= rate * n + 1, with rate <= 0.5 and every span at least 1 long.
        room = n - budget - (spans - 1)
        gaps = np.sort(np.random.randint(0, room + 1, size=spans)) + np.arange(spans)
        starts = gaps + np.r_[0, np.cumsum(lengths)[:-1]]
        return np.concatenate([np.arange(start, start + length) for start, length in zip(starts, lengths)])

    def denoiser_stats(self) -> Dict[str, Dict[str, float]]:
        """Per denoiser since the start of this process, on this host."""
        total_masked = max(int(self.masked_tokens.sum()), 1)
        return {
            name: {
                "segments": int(self.segments[i]),
                "masked_share": float(self.masked_tokens[i]) / total_masked,
                "masked_rate": float(self.masked_tokens[i]) / max(int(self.maskable_tokens[i]), 1),
                "hidden_tokens": int(self.hidden_tokens[i]),
            }
            for i, name in enumerate(DENOISER_NAMES)
        }


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
        probability_matrix[np.isin(input_ids, self.tokenizer.all_special_ids)] = 0.0

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
        probability_matrix[np.isin(input_ids, self.tokenizer.all_special_ids)] = 0.0

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
    # This host's own parquet files on local disk, instead of streaming and sharding dataset_name
    data_files: Optional[List[str]] = None,
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
        data_files: Parquet files of this host (single-dataset mode only)

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
        data_files=data_files,
    )


def create_packed_dataset_streaming(
    stream_dataset,
    tokenizer,
    max_length: int,
    batch_size: int,
    dataset_format: str = "text",
    seed: int = 0,
):
    """Create a packed IterDataset from a streaming dataset.

    Uses StreamingBufferDataSource to lazily load from the stream while
    providing the RandomAccessDataSource interface that Grain requires.

    Args:
        stream_dataset: Any Python iterable yielding dicts with "text" field
            (for format="text") or "inputs"/"targets" fields (for format="inputs_targets"),
            or a StreamingBufferDataSource around one
        tokenizer: Tokenizer with eos_token_id
        max_length: Maximum sequence length for packing
        batch_size: Number of packing bins (sequences per batch)
        dataset_format: "text" for standard text datasets, "inputs_targets" for FLAN-style
        seed: Seed of the window that shuffles pieces before packing (format "text")

    Returns:
        IterDataset that yields packed batches
    """
    if grain is None or grain_exp is None:
        raise ImportError(
            "Grain is not available. Install with: pip install grain"
        )

    # Wrap stream as a lazy-loading RandomAccessDataSource
    if isinstance(stream_dataset, StreamingBufferDataSource):
        source = stream_dataset
    else:
        source = StreamingBufferDataSource(stream_dataset, max_buffer_size=SOURCE_BUFFER_ITEMS)

    # Create tokenizer transform based on dataset format
    if dataset_format == "inputs_targets":
        tokenize = FLANTokenizeTransform(tokenizer, max_length)
    else:  # Default to "text" format
        tokenize = TokenizeTransform(tokenizer, max_length)

    # Build Grain pipeline: source -> tokenize -> to_iter -> (pieces -> shuffle) -> pack -> batch
    map_dataset = grain.MapDataset.source(source).map(tokenize.map)
    iter_dataset = map_dataset.to_iter_dataset()
    if isinstance(tokenize, TokenizeTransform):
        iter_dataset = grain_exp.WindowShuffleIterDataset(
            grain_exp.FlatMapIterDataset(iter_dataset, tokenize), window_size=PIECE_SHUFFLE_WINDOW, seed=seed
        )

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
    # drop_remainder: every host must hand in the same number of rows, also at the end of a pass
    batched_dataset = packed_dataset.batch(batch_size, drop_remainder=True)

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
        data_files: Optional[List[str]] = None,
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
            data_files: Parquet files of this host, instead of streaming and sharding
                dataset_name. Their order is shuffled with the seed, as a stream's files are.
        """
        # Determine mode
        self.multi_dataset_mode = dataset_configs is not None
        self.data_files = data_files

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
        self._document_sources = []
        self._documents_of_finished_epochs = 0
        self._pieces_read = 0
        self._tokens_read = 0
        self._documents_completed = 0
        # False after a resume from a data state saved before these two counts existed
        self._counted_from_start = True
        self._restore_state = None
    
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
        
        if self.data_files is not None:
            # Already this host's part of the data: nothing to shard
            stream_dataset = load_dataset("parquet", data_files=self.data_files, split="train", streaming=True)
            return stream_dataset.shuffle(seed=epoch_seed, buffer_size=self.shuffle_buffer_size)

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
            source = StreamingBufferDataSource(self._load_stream_dataset(), max_buffer_size=SOURCE_BUFFER_ITEMS)
            self._document_sources = [source]
            # A window of pieces reads thousands of documents at once, measured at 2.3 seconds:
            # in the training loop's thread that would cost 10% of the throughput
            return grain_exp.ThreadPrefetchIterDataset(
                create_packed_dataset_streaming(
                    source,
                    tokenizer=self.tokenizer,
                    max_length=self.max_length,
                    batch_size=self.batch_size,
                    dataset_format=self.dataset_format,
                    seed=self.base_seed + self.epoch,
                ),
                prefetch_buffer_size=BATCH_PREFETCH,
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
        self._document_sources = []

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
            source = StreamingBufferDataSource(stream, max_buffer_size=SOURCE_BUFFER_ITEMS)
            self._document_sources.append(source)

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
            if isinstance(tokenize, TokenizeTransform):
                iter_dataset = grain_exp.FlatMapIterDataset(iter_dataset, tokenize)

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
        interleaved_source = StreamingBufferDataSource(iter(interleaved), max_buffer_size=SOURCE_BUFFER_ITEMS)
        interleaved_map = grain.MapDataset.source(interleaved_source).map(lambda x: x)
        interleaved_iter = grain_exp.WindowShuffleIterDataset(
            interleaved_map.to_iter_dataset(
                read_options=grain.ReadOptions(num_threads=1, prefetch_buffer_size=0)
            ),
            window_size=PIECE_SHUFFLE_WINDOW,
            seed=epoch_seed,
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
        # drop_remainder: every host must hand in the same number of rows, also at the end of a pass
        batched_dataset = packed_dataset.batch(self.batch_size, drop_remainder=True)
        return batched_dataset
    
    @property
    def documents_read(self) -> int:
        """Documents this host read from the stream, over all epochs, including those in prefetch."""
        return self._documents_of_finished_epochs + sum(source.stream_position for source in self._document_sources)

    @property
    def pieces_read(self) -> int:
        """Segments this host handed to training, over all epochs: the pieces of its documents."""
        return self._pieces_read

    @property
    def tokens_read(self) -> int:
        """Tokens other than padding this host handed to training, <s> and </s> included."""
        return self._tokens_read

    @property
    def documents_completed(self) -> int:
        """Documents whose last piece this host handed to training: only that piece has </s>.

        Unlike documents_read this leaves out what still waits in prefetch, so it is what
        training consumed.
        """
        return self._documents_completed

    def accounting(self) -> Dict[str, Any]:
        """What this host handed to training since the start of the run."""
        return {
            "documents_read": self.documents_read,
            "documents_completed": self._documents_completed,
            "pieces_read": self._pieces_read,
            "tokens_read": self._tokens_read,
            "epoch": self.epoch,
            "counted_from_start": self._counted_from_start,
        }

    def get_state(self) -> Dict[str, Any]:
        """Where this host is: enough for a new loader to yield the very next batch.

        Grain's state holds indices only, so it needs the same stream order again: the same
        dataset, seed and number of hosts. On restore the stream is read up to that index.
        """
        return {
            "epoch": self.epoch,
            "pipeline": self._current_iter.get_state() if self._current_iter is not None else None,
            "documents_of_finished_epochs": self._documents_of_finished_epochs,
            "pieces_read": self._pieces_read,
            "tokens_read": self._tokens_read,
            "documents_completed": self._documents_completed,
            "counted_from_start": self._counted_from_start,
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Continue from get_state(). Call before the first batch."""
        self.epoch = state["epoch"]
        self._documents_of_finished_epochs = state["documents_of_finished_epochs"]
        self._pieces_read = state["pieces_read"]
        if "tokens_read" in state:
            self._tokens_read = state["tokens_read"]
            self._documents_completed = state["documents_completed"]
            self._counted_from_start = state["counted_from_start"]
        else:
            # Saved before these counts: they start here, and accounting() says so
            self._counted_from_start = False
            print("[EpochTrackingDataLoader] data state has no tokens_read or documents_completed: "
                  "those count from this resume on, counted_from_start=False", flush=True)
        self._restore_state = state["pipeline"]

    def __iter__(self):
        """Iterate through batches, auto-restarting on epoch boundaries."""
        while self.num_epochs is None or self.epoch < self.num_epochs:
            self._current_iter = iter(self._create_pipeline())
            restored = self._restore_state is not None
            if restored:
                self._current_iter.set_state(self._restore_state)
                self._restore_state = None
            batches = 0
            for batch in self._current_iter:
                batches += 1
                real = batch["input_ids"] != self.tokenizer.pad_token_id
                # Padding has position 0 too
                self._pieces_read += int(np.count_nonzero((batch["input_ids_positions"] == 0) & real))
                self._tokens_read += int(np.count_nonzero(real))
                # A literal </s> in a text is tokenized as text (split_special_tokens), so this id ends a document
                self._documents_completed += int(np.count_nonzero(batch["input_ids"] == self.tokenizer.eos_token_id))
                yield batch
            # A restored pass can have no batch left: its state was saved after the pass's last batch
            if batches == 0 and not restored:
                raise RuntimeError(
                    f"Epoch {self.epoch} of shard {self.shard_index}/{self.shard_count} has "
                    f"{self.documents_read - self._documents_of_finished_epochs} documents, "
                    f"too few for one batch of {self.batch_size} rows."
                )
            
            # Epoch completed
            self._documents_of_finished_epochs = self.documents_read
            self._document_sources = []  # Counted now; left in place they would count twice
            self.epoch += 1
            print(f"[EpochTrackingDataLoader] Completed epoch {self.epoch} "
                  f"(shard {self.shard_index}/{self.shard_count}): "
                  f"{self.documents_read:,} documents read so far, "
                  f"{self.pieces_read:,} pieces handed to training", flush=True)
        
        # If num_epochs is set, we stop here
        print(f"[EpochTrackingDataLoader] Finished {self.num_epochs} epochs")
