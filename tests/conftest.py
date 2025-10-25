from __future__ import annotations

import sys
from pathlib import Path
import os

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

from src.mmbjax.configuration_mmbbert import MMBertConfig


@pytest.fixture
def tiny_config() -> MMBertConfig:
    return MMBertConfig(
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
