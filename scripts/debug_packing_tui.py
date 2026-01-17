#!/usr/bin/env python3
"""Interactive TUI for visualizing Grain sequence packing.

Usage:
    python scripts/debug_packing_tui.py \
        --dataset yhavinga/mc4_nl_cleaned \
        --dataset-config micro \
        --tokenizer jhu-clsp/mmBERT-small \
        --max-length 512 \
        --prefetch 1000

Navigation:
    ←/→     Previous/next sequence
    ↑/↓     Select segment within sequence
    m       Toggle masking visualization
    q       Quit

UI Layout:
    Top-left:  Segment layout (position-ordered)
    Top-right: Block-diagonal attention mask
    Bottom-left: Cumulative statistics (all prefetched sequences)
    Bottom-right: Decoded text OR Masking visualization (toggle with 'm')
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

import numpy as np


# Textual imports
try:
    from textual.app import App, ComposeResult
    from textual.containers import Horizontal, Vertical, Container, ScrollableContainer
    from textual.widgets import Header, Static, Label
    from textual.reactive import reactive
    from textual import events
    from textual.geometry import Size
    from textual.scroll_view import ScrollView
    TEXTUAL_AVAILABLE = True
except ImportError:
    TEXTUAL_AVAILABLE = False
    App = object


# Import masking collator for visualization
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from mmbjax.training.packing import HybridMaskingCollator, FLANTokenizeTransform, EpochTrackingDataLoader


# Grain and tokenizer imports
try:
    import grain.python as grain
    from grain.python import experimental as grain_exp
    from datasets import load_dataset
    from transformers import AutoTokenizer
    GRAIN_AVAILABLE = True
except ImportError:
    GRAIN_AVAILABLE = False


# =============================================================================
# Data Loading
# =============================================================================

class StreamingBufferDataSource:
    """Lazily fills from a stream, providing random access for Grain."""

    def __init__(self, stream, max_buffer_size: Optional[int] = None, stream_length: Optional[int] = None):
        import threading
        self._stream = iter(stream)
        self._buffer = []
        self._max_buffer_size = max_buffer_size
        self._exhausted = False
        self._stream_position = 0
        self._lock = threading.Lock()
        self._stream_length = stream_length if stream_length is not None else 10_000_000_000

    def _ensure_available(self, idx: int) -> None:
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
        with self._lock:
            if self._exhausted:
                return len(self._buffer)
            return self._stream_length

    def __getitem__(self, idx: int):
        self._ensure_available(idx)
        with self._lock:
            if idx < len(self._buffer):
                return self._buffer[idx]
            raise IndexError(f'Index {idx} out of range (buffer size: {len(self._buffer)})')


class TokenizeTransform:
    """Tokenize text for sequence packing."""

    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        import threading
        self._lock = threading.Lock()

    def map(self, element: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:  # Thread-safe tokenization for fast tokenizers
            tokens = self.tokenizer(
                element["text"],
                truncation=True,
                max_length=self.max_length,
                return_tensors=None,
                add_special_tokens=False,
            )
        input_ids = tokens["input_ids"]
        was_truncated = len(input_ids) >= self.max_length
        if not was_truncated:
            input_ids = input_ids + [self.tokenizer.eos_token_id]
        return {
            "input_ids": np.array(input_ids, dtype=np.int32),
            "length": len(input_ids),
        }


def create_packed_dataset_streaming(
    stream_dataset,
    tokenizer,
    max_length: int,
    batch_size: int,
    dataset_format: str = "text",
    num_workers: int = 0,  # 0 = no prefetch (single-threaded)
):
    """Create a packed IterDataset from a streaming dataset."""
    source = StreamingBufferDataSource(stream_dataset)

    # Choose tokenizer transform based on dataset format
    if dataset_format == "inputs_targets":
        tokenize = FLANTokenizeTransform(tokenizer, max_length)
    else:
        tokenize = TokenizeTransform(tokenizer, max_length)

    map_dataset = grain.MapDataset.source(source).map(tokenize.map)
    iter_dataset = map_dataset.to_iter_dataset()

    # Use tokenizer's pad_token_id for padding (not unk!)
    # This ensures filler segments use the correct pad token
    packed_dataset = grain_exp.FirstFitPackIterDataset(
        parent=iter_dataset,
        length_struct={"input_ids": max_length},
        num_packing_bins=batch_size,
        shuffle_bins=True,
        padding_struct={"input_ids": tokenizer.pad_token_id},
    )

    batched_dataset = packed_dataset.batch(batch_size)
    return batched_dataset


def get_individual_sequences(packed_dataset, n_sequences: int = 1000) -> List[Dict[str, np.ndarray]]:
    """Extract N individual sequences from packed dataset.

    Grain returns batches of (batch_size, max_length). We flatten to
    individual sequences for TUI navigation.
    """
    sequences = []
    for batch in packed_dataset:
        batch_size = batch["input_ids"].shape[0]
        for i in range(batch_size):
            sequences.append({
                "input_ids": batch["input_ids"][i],
                "segment_ids": batch["input_ids_segment_ids"][i],
                "positions": batch["input_ids_positions"][i],
            })
            if len(sequences) >= n_sequences:
                return sequences
    return sequences


# =============================================================================
# Statistics Tracking
# =============================================================================

@dataclass
class PackingStats:
    """Track cumulative statistics across viewed sequences."""
    sequences_viewed: int = 0
    total_segments: int = 0
    total_real_tokens: int = 0  # Only real document tokens, not padding
    total_token_capacity: int = 0
    segment_lengths: List[int] = field(default_factory=list)
    pad_token_id: int = 0  # Pad token ID to exclude from efficiency

    def update(self, input_ids: np.ndarray, segment_ids: np.ndarray, max_length: int) -> None:
        """Update stats with a new sequence.

        Excludes segments that consist entirely of pad tokens from efficiency.
        These are filler segments Grain uses to occupy leftover space.
        """
        self.sequences_viewed += 1
        self.total_token_capacity += max_length

        for seg_id in np.unique(segment_ids):
            seg_mask = segment_ids == seg_id
            seg_tokens = input_ids[seg_mask]
            seg_length = int(seg_mask.sum())

            # Check if this segment is all padding (filler segment)
            is_filler = np.all(seg_tokens == self.pad_token_id)

            if not is_filler:
                # Only count real document tokens
                self.total_segments += 1
                self.total_real_tokens += seg_length
                self.segment_lengths.append(seg_length)

    @property
    def avg_docs_per_sequence(self) -> float:
        if self.sequences_viewed == 0:
            return 0.0
        return self.total_segments / self.sequences_viewed

    @property
    def packing_efficiency(self) -> float:
        if self.total_token_capacity == 0:
            return 0.0
        return 100.0 * self.total_real_tokens / self.total_token_capacity

    @property
    def wasted_tokens(self) -> int:
        return self.total_token_capacity - self.total_real_tokens

    def get_segment_length_distribution(self) -> Dict[str, int]:
        """Get histogram of segment lengths."""
        if not self.segment_lengths:
            return {"1-50": 0, "51-100": 0, "101-200": 0, "201+": 0}

        bins = {"1-50": 0, "51-100": 0, "101-200": 0, "201+": 0}
        for length in self.segment_lengths:
            if length <= 50:
                bins["1-50"] += 1
            elif length <= 100:
                bins["51-100"] += 1
            elif length <= 200:
                bins["101-200"] += 1
            else:
                bins["201+"] += 1
        return bins


# =============================================================================
# Color Palette
# =============================================================================

SEGMENT_COLORS = [
    "red", "green", "blue", "yellow", "magenta", "cyan",
    "bright_red", "bright_green", "bright_blue", "bright_yellow",
    "bright_magenta", "bright_cyan", "orange", "purple", "pink",
]


def get_segment_color(segment_id: int) -> str:
    """Get color for a segment ID."""
    return SEGMENT_COLORS[segment_id % len(SEGMENT_COLORS)]


# =============================================================================
# Textual Widgets
# =============================================================================

class SequenceView(Static):
    """Display single sequence with color-coded segment blocks."""

    sequence_idx = reactive(0)
    selected_segment = reactive(0)

    def __init__(self, sequences: List[Dict[str, np.ndarray]], **kwargs):
        super().__init__(**kwargs)
        self.sequences = sequences
        self.max_length = sequences[0]["input_ids"].shape[0] if sequences else 512

    def get_segment_info(self) -> List[Dict[str, Any]]:
        """Extract segment information for current sequence."""
        if not self.sequences:
            return []

        seq = self.sequences[self.sequence_idx]
        segment_ids = seq["segment_ids"]
        input_ids = seq["input_ids"]
        positions = seq["positions"]

        segments = []
        for seg_id in np.unique(segment_ids):
            mask = segment_ids == seg_id
            indices = np.where(mask)[0]
            if len(indices) == 0:
                continue
            start = int(indices[0])
            end = int(indices[-1]) + 1
            seg_positions = positions[mask]

            segments.append({
                "id": int(seg_id),
                "start": start,
                "end": end,
                "length": int(mask.sum()),
                "color": get_segment_color(int(seg_id)),
                "position_ids": seg_positions,
                "pos_start": int(seg_positions[0]) if len(seg_positions) > 0 else 0,
                "pos_end": int(seg_positions[-1]) if len(seg_positions) > 0 else 0,
            })
        # Sort by START position (actual position in sequence, not segment ID)
        # This ensures the visualization matches the actual token layout
        segments.sort(key=lambda s: s["start"])
        return segments

    def render_inline_blocks(self, segments: List[Dict], width: int) -> str:
        """Render segments as inline horizontal blocks showing actual packing.

        This shows how segments are concatenated in the sequence.
        NOTE: Segments shown in POSITION order (left to right in sequence).
        Segment IDs are assigned in DOCUMENT ORDER (as they come from the stream).
        So Seg #0 might appear at the end if it was a leftover "pad" segment.
        """
        if not segments:
            return "[dim italic]No sequence data[/]"

        lines = []

        # Header line with position markers
        header = "    "
        ruler = "   "
        for seg in segments:
            # Midpoint marker
            mid = seg["start"] + seg["length"] // 2
            header += f" {mid:3d}"
            # Segment bar
            bar_len = max(3, seg["length"] // 10)  # Scale down
            ruler += f"[{seg['color']}]{'█' * bar_len}[/]"

        lines.append("[dim]Segments in POSITION ORDER (left→right = sequence order)[/]")
        lines.append("[dim]Segment IDs are assigned by document order, not position[/]")
        lines.append("")
        lines.append("[dim]Position markers (seq_offset):[/]")
        lines.append(header)
        lines.append(ruler)
        lines.append("")

        # Detail lines for each segment
        for i, seg in enumerate(segments):
            is_selected = seg["id"] == self.selected_segment
            color = seg["color"]
            prefix = "▶ " if is_selected else "  "

            # Segment info line
            line = f"{prefix}[{color}]●[/] [{color}]Seg #{seg['id']}[/]"
            line += f" [dim]│[/] seq_offset: [cyan]{seg['start']}[/]-[cyan]{seg['end']-1}[/]"
            line += f" [dim]│[/] len: [cyan]{seg['length']}[/]"
            line += f" [dim]│[/] position_ids: [cyan]{seg['pos_start']}[/]-[cyan]{seg['pos_end']}[/]"
            if is_selected:
                line += " [dim]◀[/]"
            lines.append(line)

        return "\n".join(lines)


    def watch_sequence_idx(self) -> None:
        self.update(self.render_content())

    def watch_selected_segment(self) -> None:
        self.update(self.render_content())

    def render_content(self) -> str:
        """Render the segment blocks view."""
        if not self.sequences:
            return "[dim italic]Loading...[/]"

        container_size = self.size
        width = container_size.width - 4 if container_size else 80

        segments = self.get_segment_info()

        header = f"\n[b u]SEQUENCE {self.sequence_idx + 1}/{len(self.sequences)}[/] [dim]│[/] [b green]SEGMENT LAYOUT[/]\n"

        view = self.render_inline_blocks(segments, width)

        # Footer stats
        num_docs = len(segments)
        density = 100.0 * sum(s["length"] for s in segments) / self.max_length
        used = sum(s["length"] for s in segments)
        footer = f"\n[dim]Segments: [cyan]{num_docs}[/] docs[/]  [dim]│[/]  [cyan]{used}[/][dim]/{self.max_length} tokens ([cyan]{density:.1f}%[/])[/]"

        return header + view + footer


class AttentionView(Static):
    """Display attention mask visualization."""

    sequence_idx = reactive(0)
    sequences: List[Dict[str, np.ndarray]] = field(default_factory=list)

    def __init__(self, sequences: List[Dict[str, np.ndarray]], **kwargs):
        super().__init__(**kwargs)
        self.sequences = sequences
        self.max_length = sequences[0]["input_ids"].shape[0] if sequences else 512

    def watch_sequence_idx(self) -> None:
        self.update(self.render_content())

    def render_content(self) -> str:
        """Render the attention mask as a proportional 2D grid."""
        if not self.sequences:
            return "[dim italic]Loading...[/]"

        seq = self.sequences[self.sequence_idx]
        segment_ids = seq["segment_ids"]
        max_len = len(segment_ids)

        lines = []
        lines.append("[b u]BLOCK-DIAGONAL ATTENTION[/]\n")
        lines.append("[dim]Block sizes ∝ actual segment lengths[/]")

        # Get segments in position order with their lengths
        seg_positions = []
        for seg_id in np.unique(segment_ids):
            mask = segment_ids == seg_id
            indices = np.where(mask)[0]
            if len(indices) > 0:
                seg_positions.append((int(indices[0]), int(seg_id), int(mask.sum())))
        seg_positions.sort()  # Sort by start position
        segments = [(seg_id, length) for _, seg_id, length in seg_positions]

        if not segments:
            return "\n".join(lines) + "\n[dim]No segments[/]"

        # Create a downsampled 2D attention mask
        # Determine grid size based on available space (approximate)
        container_size = self.size
        grid_w = min(64, container_size.width - 10) if container_size else 50
        grid_h = min(24, container_size.height - 10) if container_size else 16

        # Build the attention mask matrix (actual size)
        # attention_mask[i, j] = True if position i can attend to position j
        attention_mask = np.zeros((max_len, max_len), dtype=bool)
        for i in range(max_len):
            # Position i can attend to any position j in the same segment
            for j in range(max_len):
                attention_mask[i, j] = (segment_ids[i] == segment_ids[j])

        # Downsample to fit grid by averaging blocks
        # Each "pixel" represents a range of positions
        block_size_h = max(1, (max_len + grid_h - 1) // grid_h)
        block_size_w = max(1, (max_len + grid_w - 1) // grid_w)

        # Header with segment markers
        header = "\n    "
        pixels_per_seg = []
        for start, seg_id, length in seg_positions:
            # How many grid columns this segment spans
            start_col = start // block_size_w
            end_col = (start + length) // block_size_w
            span = max(1, end_col - start_col)
            pixels_per_seg.append((start_col, span, seg_id))

        # Place segment labels in header
        label_line = [" "] * (grid_w + 5)
        for start_col, span, seg_id in pixels_per_seg:
            if span >= 2:
                label_pos = 4 + start_col + span // 2
                label = f"{seg_id}"
                idx = label_pos - len(label) // 2
                for i, c in enumerate(label):
                    if 0 <= idx + i < len(label_line):
                        label_line[idx + i] = c

        lines.append("".join(label_line))
        lines.append("")

        # Render the downsampled grid
        for row in range(grid_h):
            row_start = row * block_size_h
            row_end = min((row + 1) * block_size_h, max_len)
            row_label = f"{row_start:4d}:"

            row_visual = ""
            for col in range(grid_w):
                col_start = col * block_size_w
                col_end = min((col + 1) * block_size_w, max_len)

                # Check if this block has any attending cells
                has_attention = False
                total_cells = 0
                attending_cells = 0

                for i in range(row_start, row_end):
                    for j in range(col_start, col_end):
                        if i < max_len and j < max_len:
                            total_cells += 1
                            if attention_mask[i, j]:
                                attending_cells += 1

                # Character based on density of attention in this block
                if total_cells > 0:
                    density = attending_cells / total_cells
                    if density >= 0.75:
                        row_visual += "█"  # Mostly can attend
                    elif density >= 0.5:
                        row_visual += "▓"  # Half can attend
                    elif density >= 0.25:
                        row_visual += "░"  # Some can attend
                    else:
                        row_visual += "·"  # Mostly blocked
                else:
                    row_visual += " "

            lines.append(f"[dim]{row_label}[/]{row_visual}")

        lines.append("")
        lines.append("[dim]Block sizes ∝ segment lengths  █=attend  ░=blocked[/]")

        return "\n".join(lines)


class DecodedTextPanel(Static):
    """Display decoded text for selected segment with scroll."""

    sequence_idx = reactive(0)
    selected_segment = reactive(0)
    sequences: List[Dict[str, np.ndarray]] = field(default_factory=list)
    tokenizer = None
    full_text: str = ""
    metadata: Dict = field(default_factory=dict)

    def __init__(self, sequences: List[Dict[str, np.ndarray]], tokenizer, **kwargs):
        super().__init__(**kwargs)
        self.sequences = sequences
        self.tokenizer = tokenizer

    def decode_segment(self, segment_id: int) -> tuple[str, Dict]:
        """Decode a segment's tokens back to text."""
        if not self.sequences:
            return "No data", {}

        seq = self.sequences[self.sequence_idx]
        segment_ids = seq["segment_ids"]
        input_ids = seq["input_ids"]
        positions = seq["positions"]

        mask = segment_ids == segment_id
        if not mask.any():
            return "[dim italic]Segment not found[/]", {}

        segment_tokens = input_ids[mask]
        segment_positions = positions[mask]

        # Decode tokens
        text = self.tokenizer.decode(segment_tokens, skip_special_tokens=False)

        # Get metadata
        start_idx = np.where(mask)[0][0] if mask.any() else 0
        end_idx = np.where(mask)[0][-1] + 1 if mask.any() else 0

        # First 20 tokens for preview
        tokens_preview = segment_tokens[:20].tolist()

        metadata = {
            "segment_id": int(segment_id),
            "length": int(mask.sum()),
            "position_range": f"{segment_positions[0]}-{segment_positions[-1]}",
            "seq_offset": f"{start_idx}-{end_idx-1}",
            "tokens_preview": tokens_preview,
            "all_tokens": segment_tokens.tolist(),
        }

        return text, metadata

    def watch_sequence_idx(self) -> None:
        self._update_content()

    def watch_selected_segment(self) -> None:
        self._update_content()

    def _update_content(self):
        self.full_text, self.metadata = self.decode_segment(self.selected_segment)
        self.update(self.render_content())

    def render_content(self) -> str:
        """Render decoded text panel."""
        if not self.sequences:
            return "[b u]DECODED TEXT[/]\n\n[dim italic]Loading...[/]"

        if not self.metadata:
            return "[b u]DECODED TEXT[/]\n\n[dim italic]No segment selected[/]"

        m = self.metadata
        content = "[b u]DECODED TEXT[/]\n\n"

        # Metadata header
        content += f"  [cyan]Doc #{m['segment_id']}[/]  [dim]•[/]  "
        content += f"[dim]len:[/] [cyan]{m['length']}[/] [dim]tokens[/]  [dim]•[/]  "
        content += f"[dim]seq_offset:[/] [cyan]{m['seq_offset']}[/]  [dim]•[/]  "
        content += f"[dim]pos_ids:[/] [cyan]{m['position_range']}[/]\n"
        content += "  " + "─" * 60 + "\n\n"

        # Full text (no truncation - parent ScrollView handles scrolling)
        content += f"[white]{self.full_text}[/]\n\n"

        # Token preview
        if "tokens_preview" in m:
            tokens = m["tokens_preview"]
            content += f"[dim]First 20 tokens: {tokens}"
            if len(m.get("all_tokens", [])) > 20:
                content += f" ... ({len(m['all_tokens'])} total)"
            content += "[/]"

        return content


class MaskingView(Static):
    """Display masking information for the current sequence.

    Shows which tokens would be masked using the HybridMaskingCollator logic.
    """

    sequence_idx = reactive(0)
    selected_segment = reactive(0)
    show_masking = reactive(False)  # Make reactive so toggle triggers re-render
    sequences: List[Dict[str, np.ndarray]] = field(default_factory=list)
    tokenizer = None
    mlm_collator: Optional[HybridMaskingCollator] = None

    def __init__(self, sequences: List[Dict[str, np.ndarray]], tokenizer, **kwargs):
        super().__init__(**kwargs)
        self.sequences = sequences

        # Get the EOS token ID BEFORE adding mask token
        # (sequences were tokenized with this vocab)
        original_eos_id = tokenizer.eos_token_id

        # Ensure mask token exists (required for MLM)
        if tokenizer.mask_token is None:
            tokenizer.add_special_tokens({'mask_token': '<mask>'})

        # Create collator with the ORIGINAL EOS ID (before mask token was added)
        self.mlm_collator = HybridMaskingCollator(
            tokenizer=tokenizer,
            sep_token_id=original_eos_id
        )

        self.tokenizer = tokenizer

    def watch_sequence_idx(self) -> None:
        self.update(self.render_content())

    def watch_selected_segment(self) -> None:
        self.update(self.render_content())

    def watch_show_masking(self) -> None:
        """Re-render when masking view is toggled."""
        self.update(self.render_content())

    def toggle_view(self) -> None:
        self.show_masking = not self.show_masking
        self.update(self.render_content())

    def render_masking_pattern(self, seq: Dict, max_display: int = 128) -> str:
        """Render the actual masking pattern by running the masking collator."""
        if not seq:
            return "[dim italic]No data[/]"

        try:
            input_ids = seq["input_ids"].copy()
            segment_ids = seq["segment_ids"].copy()

            sep_token_id = self.mlm_collator.sep_token_id

            # CRITICAL: Mark padding tokens as segment_id=-1 (matches HybridMaskingCollator.__call__)
            pad_token_id = self.tokenizer.pad_token_id
            is_padding = (input_ids == pad_token_id)
            segment_ids = np.where(is_padding, -1, segment_ids)

            # Run the actual masking FIRST so we can show results at top
            input_ids_2d = input_ids.reshape(1, -1)
            segment_ids_2d = segment_ids.reshape(1, -1)
            _, labels_2d = self.mlm_collator.mask_tokens_packed(input_ids_2d, segment_ids_2d)
            labels = labels_2d[0]

            # Calculate masking stats
            masked_positions = (labels != -100)
            total_masked = int(masked_positions.sum())
            padding_count = int(is_padding.sum())
            real_tokens = len(input_ids) - padding_count

            lines = []
            lines.append("[b]MASKING VISUALIZATION[/b]")
            lines.append("")

            # === MASKING RESULTS AT TOP (most important info) ===
            lines.append("[b green]Masking Results:[/]")
            mask_pct = 100 * total_masked / max(real_tokens, 1)
            lines.append(f"  [cyan]Total masked: {total_masked}/{real_tokens} tokens ({mask_pct:.1f}%)[/]")
            if total_masked > 0:
                positions = np.where(masked_positions)[0][:10].tolist()
                lines.append(f"  [dim]First 10 masked positions: {positions}{'...' if total_masked > 10 else ''}[/]")
            lines.append("")

            # === PER-SEGMENT BREAKDOWN ===
            lines.append("[b]Per-Segment Breakdown:[/]")
            for seg_id in np.unique(segment_ids):
                if seg_id == -1:  # Skip padding
                    continue

                seg_mask = (segment_ids == seg_id)
                seg_tokens = input_ids[seg_mask]
                seg_labels = labels[seg_mask]
                sep_positions = np.where(seg_tokens == sep_token_id)[0]

                if len(sep_positions) == 0:
                    # Standard text format - no separator
                    n_masked = int((seg_labels != -100).sum())
                    pct = 100 * n_masked / max(len(seg_tokens), 1)
                    lines.append(f"  [cyan]Segment {seg_id}[/] [dim](standard text, {len(seg_tokens)} tokens)[/]")
                    lines.append(f"    Masked: [green]{n_masked}[/]/{len(seg_tokens)} ({pct:.1f}%)")
                else:
                    first_sep = sep_positions[0]
                    resp_tokens = seg_tokens[first_sep + 1:]
                    
                    if len(resp_tokens) == 0:
                        # Separator at end (no response) → standard text with trailing EOS
                        n_masked = int((seg_labels != -100).sum())
                        pct = 100 * n_masked / max(len(seg_tokens), 1)
                        lines.append(f"  [cyan]Segment {seg_id}[/] [dim](standard text, {len(seg_tokens)} tokens, trailing EOS)[/]")
                        lines.append(f"    Masked: [green]{n_masked}[/]/{len(seg_tokens)} ({pct:.1f}%)")
                    else:
                        # True FLAN format - has instruction/response separator
                        inst_tokens = seg_tokens[:first_sep]
                        inst_labels = seg_labels[:first_sep]
                        resp_labels = seg_labels[first_sep + 1:]

                        n_inst_masked = int((inst_labels != -100).sum())
                        n_resp_masked = int((resp_labels != -100).sum())
                        inst_pct = 100 * n_inst_masked / max(len(inst_tokens), 1)
                        resp_pct = 100 * n_resp_masked / max(len(resp_tokens), 1)

                        lines.append(f"  [cyan]Segment {seg_id}[/] [dim](FLAN format, sep at {first_sep})[/]")
                        lines.append(f"    [green]Instruction[/]: {n_inst_masked}/{len(inst_tokens)} masked ({inst_pct:.1f}%)")
                        lines.append(f"    [yellow]Response[/]:    {n_resp_masked}/{len(resp_tokens)} masked ({resp_pct:.1f}%)")

            lines.append("")
            lines.append("[dim]─" + "─" * 50 + "[/]")

            # === COMPACT DEBUG INFO AT BOTTOM ===
            lines.append("[dim]Debug info:[/]")
            lines.append(f"  [dim]Seq len: {len(input_ids)} | Padding: {padding_count} | Sep token: {sep_token_id}[/]")
            # Only show first 10 tokens, not the whole array
            tokens_preview = input_ids[:10].tolist()
            lines.append(f"  [dim]First 10 tokens: {tokens_preview}...[/]")
            lines.append("")
            lines.append("[dim]Press [m] to return to text view[/]")

            return "\n".join(lines)
        except Exception as e:
            import traceback
            return f"[red]Error:[/]\n[dim]{str(e)}[/]\n[dim]{traceback.format_exc()}[/]"

    def render_content(self) -> str:
        """Render the masking view."""
        if not self.sequences:
            return "[b]MASKING VISUALIZATION[/]\n\n[dim italic]Loading...[/]"

        if not self.show_masking:
            return "[b]MASKING VISUALIZATION[/]\n\n[dim italic]Press [m] to toggle masking view[/]"

        seq = self.sequences[self.sequence_idx]
        return self.render_masking_pattern(seq)


class StatsPanel(Static):
    """Display cumulative statistics."""

    stats: Optional[PackingStats] = None
    max_length: int = 512

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def set_stats(self, stats: PackingStats, max_length: int):
        self.stats = stats
        self.max_length = max_length
        self.update(self.render_content())

    def render_histogram(self, distribution: Dict[str, int]) -> str:
        """Render segment length histogram as ASCII bars."""
        if not distribution:
            return "[dim]No data[/]"

        max_count = max(distribution.values()) if distribution.values() else 1
        max_bar_width = 16

        lines = []
        for label, count in distribution.items():
            bar_width = int(count * max_bar_width / max_count) if max_count > 0 else 0
            bar = "█" * bar_width + "░" * (max_bar_width - bar_width)
            pct = 100.0 * count / sum(distribution.values()) if distribution.values() else 0
            lines.append(f"    {label:8s} {bar} [cyan]{count}[/] ([dim]{pct:.0f}%[/])")

        return "\n".join(lines)

    def render_content(self) -> str:
        """Render stats panel."""
        if self.stats is None:
            return "[b u]CUMULATIVE STATS[/]\n\n[dim italic]No data yet[/]"

        s = self.stats
        dist = s.get_segment_length_distribution()

        content = "[b u]CUMULATIVE STATS[/]\n\n"
        content += f"  Sequences viewed: [cyan]{s.sequences_viewed}[/]\n"
        content += f"  Total segments: [cyan]{s.total_segments}[/]\n"
        content += f"  Avg docs/seq: [cyan]{s.avg_docs_per_sequence:.1f}[/]\n"
        content += f"  Efficiency: [cyan]{s.packing_efficiency:.1f}%[/] ([cyan]{s.total_real_tokens}[/][dim]/{s.total_token_capacity}[/])\n"
        content += f"  Wasted tokens: [cyan]{s.wasted_tokens}[/]\n"
        content += "\n[b u]Segment length dist[/]\n"
        content += self.render_histogram(dist)

        return content


# =============================================================================
# Main App
# =============================================================================

class PackingDebugger(App):
    """Interactive TUI for debugging Grain sequence packing."""

    CSS = """
    Screen {
        layout: vertical;
    }

    #main-container {
        height: 1fr;
    }

    #top-panel {
        height: 40%;
        layout: horizontal;
    }

    #layout-container {
        width: 55%;
        padding: 0 1;
        border-right: solid $primary;
    }

    #attention-container {
        width: 45%;
        padding: 0 1;
    }

    #bottom-panel {
        height: 1fr;
        layout: horizontal;
    }

    #stats-container {
        width: 35%;
        padding: 0 1;
        border-right: solid $primary;
    }

    #text-container {
        width: 65%;
        padding: 0 1;
    }

    #masking-container {
        width: 65%;
        padding: 0 1;
        display: none;  /* Hidden by default */
    }

    SequenceView {
        height: 100%;
    }

    AttentionView {
        height: 100%;
    }

    StatsPanel {
        height: 100%;
    }

    DecodedTextPanel {
        height: 100%;
    }

    MaskingView {
        height: 100%;
    }

    .help-text {
        text-style: dim;
        text-align: center;
        padding: 0 1;
        min-height: 1;
    }
    """

    TITLE = "Packing Debug TUI"

    def __init__(
        self,
        sequences: List[Dict[str, np.ndarray]],
        tokenizer,
        max_length: int,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Calculate stats on ALL sequences upfront
        self.stats = PackingStats(pad_token_id=tokenizer.pad_token_id)
        for seq in sequences:
            self.stats.update(seq["input_ids"], seq["segment_ids"], self.max_length)

        self.current_sequence = 0
        self.current_segment = 0
        self.show_masking = False  # Toggle between text and masking view

    def _get_segments_in_sequence(self, seq_idx: int) -> List[int]:
        if not self.sequences:
            return []
        segment_ids = self.sequences[seq_idx]["segment_ids"]
        segments = []
        for seg_id in np.unique(segment_ids):
            mask = segment_ids == seg_id
            indices = np.where(mask)[0]
            if len(indices) > 0:
                segments.append((int(indices[0]), int(seg_id)))
        segments.sort()
        return [seg_id for _, seg_id in segments]

    def compose(self) -> ComposeResult:
        yield Header()

        with Container(id="main-container"):
            # Top panel: side-by-side segment layout and attention view
            with Horizontal(id="top-panel"):
                with Vertical(id="layout-container"):
                    yield SequenceView(
                        self.sequences,
                        id="sequence-view"
                    )

                with Vertical(id="attention-container"):
                    yield AttentionView(
                        self.sequences,
                        id="attention-view"
                    )

            # Bottom panel: stats and decoded text/masking view
            with Horizontal(id="bottom-panel"):
                with Vertical(id="stats-container"):
                    yield StatsPanel(id="stats-panel")

                # Two separate scrollable containers - only one visible at a time
                # This ensures the visible content starts at top (no hidden sibling pushing it down)
                with ScrollableContainer(id="text-container"):
                    yield DecodedTextPanel(
                        self.sequences,
                        self.tokenizer,
                        id="text-panel"
                    )
                with ScrollableContainer(id="masking-container"):
                    yield MaskingView(
                        self.sequences,
                        self.tokenizer,
                        id="masking-panel"
                    )

        yield Static("[dim]←/→ Navigate sequences  │  ↑/↓ Select segment  │  [m] Toggle masking  │  [q] Quit[/]", id="help-line")

    def on_mount(self) -> None:
        stats_panel = self.query_one("#stats-panel", StatsPanel)
        stats_panel.set_stats(self.stats, self.max_length)

        segments = self._get_segments_in_sequence(0)
        if segments:
            self.current_segment = segments[0]

        # Initialize masking view state
        masking_panel = self.query_one("#masking-panel", MaskingView)
        masking_panel.show_masking = False

        self._update_widgets()

    def _update_widgets(self):
        sequence_view = self.query_one("#sequence-view", SequenceView)
        attention_view = self.query_one("#attention-view", AttentionView)
        text_panel = self.query_one("#text-panel", DecodedTextPanel)
        masking_panel = self.query_one("#masking-panel", MaskingView)
        stats_panel = self.query_one("#stats-panel", StatsPanel)

        sequence_view.sequence_idx = self.current_sequence
        sequence_view.selected_segment = self.current_segment

        attention_view.sequence_idx = self.current_sequence

        text_panel.sequence_idx = self.current_sequence
        text_panel.selected_segment = self.current_segment

        masking_panel.sequence_idx = self.current_sequence
        masking_panel.selected_segment = self.current_segment
        masking_panel.show_masking = self.show_masking

        stats_panel.set_stats(self.stats, self.max_length)

        # Toggle container visibility (not just panel visibility)
        # This ensures visible content starts at top of its own scroll container
        text_container = self.query_one("#text-container")
        masking_container = self.query_one("#masking-container")
        text_container.display = not self.show_masking
        masking_container.display = self.show_masking

    def _move_sequence(self, delta: int):
        new_idx = self.current_sequence + delta
        if 0 <= new_idx < len(self.sequences):
            self.current_sequence = new_idx
            segments = self._get_segments_in_sequence(new_idx)
            if segments:
                self.current_segment = segments[0]
            self._update_widgets()

    def _move_segment(self, delta: int):
        segments = self._get_segments_in_sequence(self.current_sequence)
        if not segments:
            return

        try:
            current_idx = segments.index(self.current_segment)
            new_idx = current_idx + delta
            if 0 <= new_idx < len(segments):
                self.current_segment = segments[new_idx]
                self._update_widgets()
        except ValueError:
            self.current_segment = segments[0]
            self._update_widgets()

    def on_key(self, event: events.Key) -> None:
        if event.key == "q":
            self.exit()
            return
        elif event.key == "left":
            self._move_sequence(-1)
        elif event.key == "right":
            self._move_sequence(1)
        elif event.key == "up":
            self._move_segment(-1)
        elif event.key == "down":
            self._move_segment(1)
        elif event.key == "m":
            self.show_masking = not self.show_masking
            self._update_widgets()



# =============================================================================
# CLI Entry Point
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Interactive TUI for visualizing Grain sequence packing"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="yhavinga/mc4_nl_cleaned",
        help="HuggingFace dataset name"
    )
    parser.add_argument(
        "--dataset-config",
        type=str,
        default="micro",
        help="HuggingFace dataset config"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="jhu-clsp/mmBERT-small",
        help="HuggingFace tokenizer name"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--prefetch",
        type=int,
        default=1000,
        help="Number of sequences to prefetch"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for packing (num_packing_bins)"
    )
    parser.add_argument(
        "--dataset-format",
        type=str,
        choices=["text", "inputs_targets"],
        default="text",
        help="Dataset format: 'text' or 'inputs_targets' for FLAN-style"
    )
    # Multi-dataset config (same format as train_minimal.py)
    parser.add_argument(
        "--datasets-config",
        type=str,
        default=None,
        help="Path to JSON config file for multiple datasets. Overrides --dataset/--dataset-config/--dataset-format"
    )
    return parser.parse_args()


def main():
    if not TEXTUAL_AVAILABLE:
        print("Error: textual is not installed. Install with: pip install textual")
        return 1

    if not GRAIN_AVAILABLE:
        print("Error: Grain or transformers not installed.")
        print("Install with: pip install grain transformers")
        return 1

    args = parse_args()

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    print(f"  Tokenizer: {args.tokenizer}")
    print(f"  Vocab size: {len(tokenizer)}")

    # Check for multi-dataset config
    if args.datasets_config:
        import json
        print(f"\nLoading datasets config: {args.datasets_config}")
        with open(args.datasets_config) as f:
            config = json.load(f)
        dataset_configs = config["datasets"]
        
        print(f"  Found {len(dataset_configs)} datasets:")
        total_weight = sum(d.get('weight', 1.0) for d in dataset_configs)
        for d in dataset_configs:
            pct = 100 * d.get('weight', 1.0) / total_weight
            fmt = d.get('format', 'text')
            print(f"    - {d['name']} ({d.get('config', 'default')}): {pct:.0f}%, format={fmt}")
        
        print("\nBuilding multi-dataset packing pipeline...")
        # Use EpochTrackingDataLoader which supports multi-dataset mode
        dataloader = EpochTrackingDataLoader(
            dataset_name=None,  # Multi-dataset mode
            dataset_config=None,
            tokenizer=tokenizer,
            max_length=args.max_length,
            batch_size=args.batch_size,
            shard_index=0,
            shard_count=1,
            seed=42,
            num_epochs=1,  # Just need one epoch for debugging
            dataset_configs=dataset_configs,
            shuffle_buffer_size=1000,
            stopping_strategy="first_exhausted",
        )
        # Get batches from the dataloader
        print(f"\nPrefetching {args.prefetch} sequences...")
        sequences = []
        for batch in dataloader:
            batch_size = batch["input_ids"].shape[0]
            for i in range(batch_size):
                sequences.append({
                    "input_ids": batch["input_ids"][i],
                    "segment_ids": batch["input_ids_segment_ids"][i],
                    "positions": batch["input_ids_positions"][i],
                })
                if len(sequences) >= args.prefetch:
                    break
            if len(sequences) >= args.prefetch:
                break
    else:
        # Single dataset mode (original behavior)
        print("\nLoading dataset...")
        stream_dataset = load_dataset(
            args.dataset,
            args.dataset_config,
            split="train",
            streaming=True
        )
        print(f"  Dataset: {args.dataset} ({args.dataset_config})")

        print("\nBuilding packing pipeline...")
        packed_dataset = create_packed_dataset_streaming(
            stream_dataset=stream_dataset,
            tokenizer=tokenizer,
            max_length=args.max_length,
            batch_size=args.batch_size,
            dataset_format=args.dataset_format,
        )
        print(f"  Max length: {args.max_length}")
        print(f"  Batch size: {args.batch_size}")
        print(f"  Dataset format: {args.dataset_format}")

        print(f"\nPrefetching {args.prefetch} sequences...")
        sequences = get_individual_sequences(packed_dataset, n_sequences=args.prefetch)
    
    print(f"  Loaded {len(sequences)} sequences")

    if not sequences:
        print("Error: No sequences loaded. Check dataset configuration.")
        return 1

    print("\nStarting TUI...")
    print()

    app = PackingDebugger(
        sequences=sequences,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )
    app.run()


if __name__ == "__main__":
    import sys
    sys.exit(main() or 0)
