"""Configuration objects for the ModernBERT → JAX port.

The goal is to mirror the PyTorch `ModernBertConfig` surface while remaining
framework agnostic so we can instantiate Flax modules or run conversion scripts
without pulling in `transformers`. The fields are intentionally aligned with
`vendor/pytorch_reference/modernbert/configuration_modernbert.py`.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Literal, Mapping, Optional


ClassifierPooling = Literal["cls", "mean"]


@dataclass
class MMBertConfig:
    """Lightweight clone of Hugging Face's `ModernBertConfig`."""

    vocab_size: int = 50368
    hidden_size: int = 768
    intermediate_size: int = 1152
    num_hidden_layers: int = 22
    num_attention_heads: int = 12
    hidden_activation: str = "gelu"
    max_position_embeddings: int = 8192
    initializer_range: float = 0.02
    initializer_cutoff_factor: float = 2.0
    norm_eps: float = 1e-5
    norm_bias: bool = False
    pad_token_id: int = 50283
    eos_token_id: int = 50282
    bos_token_id: int = 50281
    cls_token_id: int = 50281
    sep_token_id: int = 50282
    global_rope_theta: float = 160000.0
    attention_bias: bool = False
    attention_dropout: float = 0.0
    global_attn_every_n_layers: int = 3
    local_attention: int = 128
    local_rope_theta: Optional[float] = 10000.0
    embedding_dropout: float = 0.0
    mlp_bias: bool = False
    mlp_dropout: float = 0.0
    decoder_bias: bool = True
    classifier_pooling: ClassifierPooling = "cls"
    classifier_dropout: float = 0.0
    classifier_bias: bool = False
    classifier_activation: str = "gelu"
    deterministic_flash_attn: bool = False
    sparse_prediction: bool = False
    sparse_pred_ignore_index: int = -100
    reference_compile: Optional[str] = None
    repad_logits_with_grad: bool = False
    num_labels: int = 2
    problem_type: Optional[str] = None
    output_attentions: bool = False
    output_hidden_states: bool = False
    use_return_dict: bool = True

    # Non-HF helpers
    model_type: str = field(default="modernbert", init=False)

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                "`hidden_size` must be divisible by `num_attention_heads` "
                f"(got {self.hidden_size} vs {self.num_attention_heads})."
            )

        if self.local_attention <= 0 or self.local_attention % 2 != 0:
            raise ValueError(
                "`local_attention` should be a positive, even window size; "
                f"received {self.local_attention}."
            )

        if self.global_attn_every_n_layers <= 0:
            raise ValueError("`global_attn_every_n_layers` must be >= 1.")

        if self.num_hidden_layers <= 0:
            raise ValueError("`num_hidden_layers` must be positive.")

        if self.max_position_embeddings <= 0:
            raise ValueError("`max_position_embeddings` must be positive.")

        if self.classifier_pooling not in ("cls", "mean"):
            raise ValueError(
                "`classifier_pooling` must be either 'cls' or 'mean', "
                f"got {self.classifier_pooling}."
            )

        if self.hidden_activation is None:
            raise ValueError("`hidden_activation` cannot be None.")

    # --------------------------------------------------------------------- #
    # Derived convenience helpers
    # --------------------------------------------------------------------- #
    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def num_global_layers(self) -> int:
        return math.ceil(self.num_hidden_layers / self.global_attn_every_n_layers)

    def is_global_layer(self, layer_idx: int) -> bool:
        if layer_idx < 0:
            raise ValueError("`layer_idx` must be >= 0.")
        return (layer_idx % self.global_attn_every_n_layers) == 0

    # --------------------------------------------------------------------- #
    # Serialization helpers
    # --------------------------------------------------------------------- #
    def to_dict(self) -> Dict[str, Any]:
        output = asdict(self)
        output.pop("model_type", None)
        return output

    def to_json_string(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True) + "\n"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MMBertConfig":
        return cls(**dict(data))

    @classmethod
    def from_json_file(cls, path: str | Path) -> "MMBertConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    # ------------------------------------------------------------------ #
    # Small helpers used by scripts/modules
    # ------------------------------------------------------------------ #
    def ensure_can_handle_sequence(self, seq_len: int) -> None:
        if seq_len > self.max_position_embeddings:
            raise ValueError(
                f"Sequence length {seq_len} exceeds "
                f"max_position_embeddings={self.max_position_embeddings}."
            )

    def copy(self, **updates: Any) -> "MMBertConfig":
        payload = self.to_dict()
        payload.update(updates)
        return type(self)(**payload)


__all__ = ["MMBertConfig", "ClassifierPooling"]
