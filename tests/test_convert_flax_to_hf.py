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
    select_export_head,
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


class TestExportHead:
    """--export-head: one prediction head of a --mix-heads checkpoint as the head of a plain ModernBertForMaskedLM."""

    @staticmethod
    def _params(heads):
        params = {"model": {"x": np.zeros(1)}, "prediction_head": {"dense": {"kernel": np.full((2, 2), 0.0)}},
                  "decoder": {"bias": np.full(3, 0.0)}}
        for k in range(1, heads):
            params[f"prediction_head_{k}"] = {"dense": {"kernel": np.full((2, 2), float(k))}}
            params[f"decoder_{k}"] = {"bias": np.full(3, float(k))}
        return params

    def test_a_multi_head_checkpoint_needs_the_choice(self):
        with pytest.raises(ValueError, match="choose it with --export-head"):
            select_export_head(self._params(2))
        single = self._params(1)
        assert select_export_head(single) is single
        selected = select_export_head(single, 0)
        assert selected.keys() == single.keys() and all(selected[key] is single[key] for key in single)

    @pytest.mark.parametrize("heads,head", [(2, 0), (2, 1), (8, 0), (8, 5)])
    def test_the_chosen_head_under_the_names_of_the_single_head(self, heads, head):
        selected = select_export_head(self._params(heads), head)
        assert set(selected) == {"model", "prediction_head", "decoder"}
        assert np.all(selected["prediction_head"]["dense"]["kernel"] == head)
        assert np.all(selected["decoder"]["bias"] == head)

    def test_a_head_that_is_not_there_is_refused(self):
        for heads, head in [(1, 1), (2, 2), (2, -1)]:
            with pytest.raises(ValueError, match="this checkpoint has heads 0 to"):
                select_export_head(self._params(heads), head)
        gap = self._params(3)
        del gap["prediction_head_1"], gap["decoder_1"]
        with pytest.raises(ValueError, match="expected 1 to"):
            select_export_head(gap, 0)


def _word_tokenizer_dir(tmp_path, words):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    specials = ["<pad>", "<s>", "</s>", "<unk>", "<mask>"]
    vocab = {token: i for i, token in enumerate(specials + [f"w{i}" for i in range(words)])}
    backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="<pad>", bos_token="<s>", eos_token="</s>",
                                        unk_token="<unk>", mask_token="<mask>")
    directory = tmp_path / "tokenizer"
    tokenizer.save_pretrained(directory)
    return directory


def test_exported_heads_give_the_logits_of_jax_in_transformers(tmp_path):
    """HF parity of an export of each head of a two-head JAX model, on CPU: a plain ModernBertForMaskedLM, no custom code."""
    import jax
    import jax.numpy as jnp
    from transformers import ModernBertForMaskedLM as HFModernBertForMaskedLM

    sys.path.insert(0, str(ROOT / "src"))
    from mmbjax.configuration_mmbbert import MMBertConfig
    from mmbjax.heads.masked_lm import ModernBertForMaskedLM

    config = MMBertConfig(vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=3, num_attention_heads=2,
                          max_position_embeddings=64, local_attention=8, global_attn_every_n_layers=3, pad_token_id=0,
                          bos_token_id=1, eos_token_id=2)
    model = ModernBertForMaskedLM(config, num_prediction_heads=2)
    ones = jnp.ones((1, 8), dtype=jnp.int32)
    params = model.init(jax.random.PRNGKey(0), input_ids=ones, attention_mask=ones)["params"]
    # Heads far apart, so an export of the wrong one can't pass
    noise = jax.random.normal(jax.random.PRNGKey(1), (32, 32))
    params = {**params, "prediction_head_1": {**params["prediction_head_1"], "dense": {"kernel": noise}},
              "decoder_1": {"bias": jax.random.normal(jax.random.PRNGKey(2), (64,))}}

    input_dir = tmp_path / "flax"
    input_dir.mkdir()
    (input_dir / "params.pkl").write_bytes(pickle.dumps(jax.tree_util.tree_map(np.asarray, jax.device_get(params))))
    (input_dir / "config.json").write_text(json.dumps({
        "vocab_size": 64, "hidden_size": 32, "num_hidden_layers": 3, "num_attention_heads": 2, "intermediate_size": 48,
        "max_position_embeddings": 64, "global_attn_every_n_layers": 3, "local_attention": 8,
    }))
    tokenizer_dir = _word_tokenizer_dir(tmp_path, 40)

    rng = np.random.RandomState(0)
    input_ids = rng.randint(5, 45, size=(2, 24)).astype(np.int32)
    input_ids[:, 0] = 1
    attention_mask = np.ones_like(input_ids)
    jax_logits = {
        0: np.asarray(model.apply({"params": params}, input_ids=jnp.asarray(input_ids), attention_mask=jnp.asarray(attention_mask)).logits),
        1: np.asarray(model.apply({"params": params}, input_ids=jnp.asarray(input_ids), attention_mask=jnp.asarray(attention_mask),
                                  head_ids=jnp.ones_like(jnp.asarray(input_ids))).logits),
    }
    assert np.abs(jax_logits[0] - jax_logits[1]).max() > 1.0

    with pytest.raises(ValueError, match="choose it with --export-head"):
        convert_flax_to_hf(input_dir, tmp_path / "refused", tokenizer_override=str(tokenizer_dir), verify=False)
    for head in (0, 1):
        output_dir = tmp_path / f"hf_head{head}"
        convert_flax_to_hf(input_dir, output_dir, tokenizer_override=str(tokenizer_dir), verify=False, export_head=head)
        hf = HFModernBertForMaskedLM.from_pretrained(output_dir, attn_implementation="eager").eval()
        assert type(hf).__module__.startswith("transformers.")
        from safetensors.torch import load_file
        assert set(load_file(output_dir / "model.safetensors")) == set(hf.state_dict())
        with torch.no_grad():
            hf_logits = hf(input_ids=torch.from_numpy(input_ids.astype(np.int64)),
                           attention_mask=torch.from_numpy(attention_mask.astype(np.int64))).logits.numpy()
        np.testing.assert_allclose(hf_logits, jax_logits[head], atol=2e-4, rtol=1e-4)
