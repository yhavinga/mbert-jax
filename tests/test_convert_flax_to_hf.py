"""Tests for Flax to HuggingFace ModernBERT checkpoint conversion."""

from __future__ import annotations

import json
import pickle
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

# Import the conversion functions
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from convert_flax_to_hf import (
    flatten_dict,
    map_flax_key_to_hf,
    convert_flax_to_hf,
)


class TestFlattenDict:
    """Tests for flatten_dict utility."""

    def test_flat_dict(self):
        d = {"a": 1, "b": 2}
        assert flatten_dict(d) == {"a": 1, "b": 2}

    def test_nested_dict(self):
        d = {"a": {"b": {"c": 1}}}
        assert flatten_dict(d) == {"a.b.c": 1}

    def test_mixed_dict(self):
        d = {"a": 1, "b": {"c": 2, "d": {"e": 3}}}
        result = flatten_dict(d)
        assert result == {"a": 1, "b.c": 2, "b.d.e": 3}


class TestKeyMapping:
    """Tests for Flax to HuggingFace key mapping."""

    def test_embeddings_mapping(self):
        assert map_flax_key_to_hf(
            "model.embeddings.word_embeddings.embedding"
        ) == "model.embeddings.tok_embeddings.weight"

        assert map_flax_key_to_hf(
            "model.embeddings.layer_norm.scale"
        ) == "model.embeddings.norm.weight"

    def test_final_norm_mapping(self):
        assert map_flax_key_to_hf(
            "model.encoder.final_layer_norm.scale"
        ) == "model.final_norm.weight"

    def test_prediction_head_mapping(self):
        assert map_flax_key_to_hf(
            "prediction_head.dense.kernel"
        ) == "head.dense.weight"

        assert map_flax_key_to_hf(
            "prediction_head.layer_norm.scale"
        ) == "head.norm.weight"

    def test_decoder_bias_mapping(self):
        assert map_flax_key_to_hf("decoder.bias") == "decoder.bias"

    def test_layer_attention_mapping(self):
        assert map_flax_key_to_hf(
            "model.encoder.layers_0.attention.qkv.kernel"
        ) == "model.layers.0.attn.Wqkv.weight"

        assert map_flax_key_to_hf(
            "model.encoder.layers_0.attention.out.kernel"
        ) == "model.layers.0.attn.Wo.weight"

        assert map_flax_key_to_hf(
            "model.encoder.layers_1.attention_norm.scale"
        ) == "model.layers.1.attn_norm.weight"

    def test_layer_mlp_mapping(self):
        assert map_flax_key_to_hf(
            "model.encoder.layers_0.mlp.wi.kernel"
        ) == "model.layers.0.mlp.Wi.weight"

        assert map_flax_key_to_hf(
            "model.encoder.layers_0.mlp.wo.kernel"
        ) == "model.layers.0.mlp.Wo.weight"

        assert map_flax_key_to_hf(
            "model.encoder.layers_0.mlp_norm.scale"
        ) == "model.layers.0.mlp_norm.weight"

    def test_multiple_layers(self):
        for i in range(6):
            assert map_flax_key_to_hf(
                f"model.encoder.layers_{i}.mlp.wi.kernel"
            ) == f"model.layers.{i}.mlp.Wi.weight"

    def test_unknown_key_raises(self):
        with pytest.raises(ValueError, match="Unknown Flax key"):
            map_flax_key_to_hf("unknown.key.path")


class TestConversion:
    """Integration tests for full checkpoint conversion."""

    @pytest.fixture
    def tiny_flax_checkpoint(self, tmp_path):
        """Create a minimal Flax checkpoint for testing."""
        hidden_size = 8
        intermediate_size = 16
        vocab_size = 100
        num_layers = 2

        # Create minimal params structure matching real checkpoint
        params = {
            "model": {
                "embeddings": {
                    "word_embeddings": {
                        "embedding": np.random.randn(vocab_size, hidden_size).astype(np.float32)
                    },
                    "layer_norm": {
                        "scale": np.ones(hidden_size, dtype=np.float32)
                    }
                },
                "encoder": {
                    "final_layer_norm": {
                        "scale": np.ones(hidden_size, dtype=np.float32)
                    }
                }
            },
            "prediction_head": {
                "dense": {
                    "kernel": np.random.randn(hidden_size, hidden_size).astype(np.float32)
                },
                "layer_norm": {
                    "scale": np.ones(hidden_size, dtype=np.float32)
                }
            },
            "decoder": {
                "bias": np.zeros(vocab_size, dtype=np.float32)
            }
        }

        # Add layers
        for i in range(num_layers):
            layer = {
                "attention": {
                    "qkv": {
                        "kernel": np.random.randn(hidden_size, 3 * hidden_size).astype(np.float32)
                    },
                    "out": {
                        "kernel": np.random.randn(hidden_size, hidden_size).astype(np.float32)
                    }
                },
                "mlp": {
                    "wi": {
                        "kernel": np.random.randn(hidden_size, 2 * intermediate_size).astype(np.float32)
                    },
                    "wo": {
                        "kernel": np.random.randn(intermediate_size, hidden_size).astype(np.float32)
                    }
                },
                "mlp_norm": {
                    "scale": np.ones(hidden_size, dtype=np.float32)
                }
            }
            # Layer 0 has no attn_norm (by design)
            if i > 0:
                layer["attention_norm"] = {
                    "scale": np.ones(hidden_size, dtype=np.float32)
                }
            params["model"]["encoder"][f"layers_{i}"] = layer

        # Save checkpoint
        input_dir = tmp_path / "flax_checkpoint"
        input_dir.mkdir()

        with open(input_dir / "params.pkl", "wb") as f:
            pickle.dump(params, f)

        config = {
            "vocab_size": vocab_size,
            "hidden_size": hidden_size,
            "num_hidden_layers": num_layers,
            "num_attention_heads": 2,
            "intermediate_size": intermediate_size,
            "max_position_embeddings": 32,
        }
        with open(input_dir / "config.json", "w") as f:
            json.dump(config, f)

        return input_dir, params, config

    def test_conversion_creates_files(self, tiny_flax_checkpoint, tmp_path):
        """Test that conversion creates expected output files."""
        input_dir, _, _ = tiny_flax_checkpoint
        output_dir = tmp_path / "hf_checkpoint"

        convert_flax_to_hf(input_dir, output_dir, verify=False)

        assert (output_dir / "model.safetensors").exists()
        assert (output_dir / "config.json").exists()

    def test_conversion_config_has_required_fields(self, tiny_flax_checkpoint, tmp_path):
        """Test that output config has all required HuggingFace fields."""
        input_dir, _, _ = tiny_flax_checkpoint
        output_dir = tmp_path / "hf_checkpoint"

        convert_flax_to_hf(input_dir, output_dir, verify=False)

        with open(output_dir / "config.json") as f:
            config = json.load(f)

        required_fields = [
            "architectures",
            "model_type",
            "vocab_size",
            "hidden_size",
            "num_hidden_layers",
            "num_attention_heads",
            "intermediate_size",
        ]
        for field in required_fields:
            assert field in config, f"Missing required field: {field}"

        assert config["model_type"] == "modernbert"
        assert "ModernBertForMaskedLM" in config["architectures"]

    def test_conversion_tensor_shapes(self, tiny_flax_checkpoint, tmp_path):
        """Test that converted tensors have correct shapes (transposed)."""
        input_dir, params, config = tiny_flax_checkpoint
        output_dir = tmp_path / "hf_checkpoint"

        convert_flax_to_hf(input_dir, output_dir, verify=False)

        from safetensors.torch import load_file
        hf_state = load_file(output_dir / "model.safetensors")

        hidden_size = config["hidden_size"]
        intermediate_size = config["intermediate_size"]

        # Check embedding (no transpose)
        assert hf_state["model.embeddings.tok_embeddings.weight"].shape == (
            config["vocab_size"], hidden_size
        )

        # Check attention kernels (transposed)
        assert hf_state["model.layers.0.attn.Wqkv.weight"].shape == (
            3 * hidden_size, hidden_size  # transposed from [hidden, 3*hidden]
        )

        # Check MLP kernels (transposed)
        assert hf_state["model.layers.0.mlp.Wi.weight"].shape == (
            2 * intermediate_size, hidden_size  # transposed
        )
        assert hf_state["model.layers.0.mlp.Wo.weight"].shape == (
            hidden_size, intermediate_size  # transposed
        )

    def test_conversion_includes_decoder_weight(self, tiny_flax_checkpoint, tmp_path):
        """Test that decoder.weight is created (tied to embeddings)."""
        input_dir, _, _ = tiny_flax_checkpoint
        output_dir = tmp_path / "hf_checkpoint"

        convert_flax_to_hf(input_dir, output_dir, verify=False)

        from safetensors.torch import load_file
        hf_state = load_file(output_dir / "model.safetensors")

        assert "decoder.weight" in hf_state
        assert "decoder.bias" in hf_state

        # decoder.weight should equal embeddings
        assert torch.equal(
            hf_state["decoder.weight"],
            hf_state["model.embeddings.tok_embeddings.weight"]
        )

    def test_layer0_has_no_attn_norm(self, tiny_flax_checkpoint, tmp_path):
        """Test that layer 0 has no attn_norm (by design)."""
        input_dir, _, _ = tiny_flax_checkpoint
        output_dir = tmp_path / "hf_checkpoint"

        convert_flax_to_hf(input_dir, output_dir, verify=False)

        from safetensors.torch import load_file
        hf_state = load_file(output_dir / "model.safetensors")

        # Layer 0 should NOT have attn_norm
        assert "model.layers.0.attn_norm.weight" not in hf_state

        # Layer 1 should have attn_norm
        assert "model.layers.1.attn_norm.weight" in hf_state
