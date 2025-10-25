"""Parity tests against PyTorch ModernBERT reference outputs."""

import os
from pathlib import Path

import numpy as np
import pytest
import torch

import jax.numpy as jnp
import jax.random as jr

from src.mmbjax.heads.masked_lm import ModernBertForMaskedLM
from src.mmbjax.modeling_mmbbert import ModernBertModel
from src.mmbjax.utils import convert_pytorch_state_dict_to_flax, load_hf_config_as_mmbert

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")


ARTIFACTS_DIR = Path("artifacts/reference-configs")
CONFIG_PATH = ARTIFACTS_DIR / "jhu-clsp_mmBERT-small_config.json"
BATCH_PATH = ARTIFACTS_DIR / "jhu-clsp_mmBERT-small_batch.npz"
PT_CKPT_PATH = Path.home() / ".cache/huggingface/hub"  # default HF cache


@pytest.fixture
def mmbert_config():
    """Load mmBERT-small config."""
    if not CONFIG_PATH.exists():
        pytest.skip(f"Reference config not found: {CONFIG_PATH}")
    return load_hf_config_as_mmbert(CONFIG_PATH)


@pytest.fixture
def reference_batch():
    """Load golden PyTorch outputs."""
    if not BATCH_PATH.exists():
        pytest.skip(f"Reference batch not found: {BATCH_PATH}")
    data = np.load(BATCH_PATH)
    return {
        "input_ids": jnp.array(data["input_ids"], dtype=jnp.int32),
        "attention_mask": jnp.array(data["attention_mask"], dtype=jnp.int32),
        "last_hidden_state": data["last_hidden_state"],
        "hidden_states": data["hidden_states"],
        "logits": data["logits"],
    }


@pytest.fixture
def pytorch_state_dict():
    """Load PyTorch checkpoint weights."""
    # Attempt to find cached model
    model_id = "jhu-clsp/mmBERT-small"
    cache_dirs = list(PT_CKPT_PATH.glob("models--jhu-clsp--mmBERT-small/snapshots/*"))
    
    if not cache_dirs:
        pytest.skip(f"PyTorch checkpoint not found in {PT_CKPT_PATH}")
    
    # Use first snapshot directory
    snapshot_dir = cache_dirs[0]
    pt_file = snapshot_dir / "pytorch_model.bin"
    
    if not pt_file.exists():
        pytest.skip(f"pytorch_model.bin not found in {snapshot_dir}")
    
    state_dict = torch.load(pt_file, map_location="cpu")
    # Convert to numpy
    return {k: v.numpy() for k, v in state_dict.items()}


def test_config_loads_correctly(mmbert_config):
    """Verify config mapping from HF to MMBertConfig."""
    assert mmbert_config.vocab_size == 256000
    assert mmbert_config.hidden_size == 384
    assert mmbert_config.intermediate_size == 1152
    assert mmbert_config.num_hidden_layers == 22
    assert mmbert_config.num_attention_heads == 6
    assert mmbert_config.head_dim == 64
    assert mmbert_config.max_position_embeddings == 8192
    assert mmbert_config.global_attn_every_n_layers == 3
    assert mmbert_config.local_attention == 128
    assert mmbert_config.global_rope_theta == 160000
    assert mmbert_config.local_rope_theta == 160000
    assert not mmbert_config.attention_bias
    assert not mmbert_config.mlp_bias
    assert not mmbert_config.norm_bias
    assert mmbert_config.decoder_bias


def test_flax_model_forward_shape(mmbert_config, reference_batch):
    """Test Flax model produces correct output shapes."""
    model = ModernBertForMaskedLM(mmbert_config)
    
    variables = model.init(
        jr.PRNGKey(0),
        input_ids=reference_batch["input_ids"],
        deterministic=True,
    )
    
    outputs = model.apply(
        variables,
        input_ids=reference_batch["input_ids"],
        attention_mask=reference_batch["attention_mask"],
        deterministic=True,
        output_hidden_states=True,
    )
    
    batch_size, seq_len = reference_batch["input_ids"].shape
    assert outputs.logits.shape == (batch_size, seq_len, mmbert_config.vocab_size)
    assert outputs.hidden_states is not None
    assert len(outputs.hidden_states) == mmbert_config.num_hidden_layers + 1


@pytest.mark.skipif(not BATCH_PATH.exists(), reason="Reference batch not available")
def test_logits_parity_with_converted_weights(mmbert_config, reference_batch, pytorch_state_dict):
    """Test Flax model with converted PyTorch weights matches reference logits.
    
    This is the key parity test - we load PyTorch weights, convert them to Flax,
    and verify the outputs match within tolerance.
    """
    # Convert PyTorch weights to Flax
    flax_params = convert_pytorch_state_dict_to_flax(pytorch_state_dict, mmbert_config)
    
    # Create model and run forward pass
    model = ModernBertForMaskedLM(mmbert_config)
    outputs = model.apply(
        flax_params,
        input_ids=reference_batch["input_ids"],
        attention_mask=reference_batch["attention_mask"],
        deterministic=True,
        output_hidden_states=True,
    )
    
    # Compare logits
    flax_logits = np.asarray(outputs.logits)
    pt_logits = reference_batch["logits"]
    
    max_diff = np.max(np.abs(flax_logits - pt_logits))
    mean_diff = np.mean(np.abs(flax_logits - pt_logits))
    rel_diff = max_diff / (np.abs(pt_logits).max() + 1e-8)
    
    print(f"\nLogits comparison:")
    print(f"  Max absolute diff: {max_diff:.6e}")
    print(f"  Mean absolute diff: {mean_diff:.6e}")
    print(f"  Max relative diff: {rel_diff:.6e}")
    
    # Initially use loose tolerance; tighten as we debug
    assert max_diff < 1e-3, f"Logits differ by {max_diff:.6e} (max)"
    assert rel_diff < 1e-3, f"Relative logits diff {rel_diff:.6e} exceeds tolerance"


@pytest.mark.skipif(not BATCH_PATH.exists(), reason="Reference batch not available")
def test_hidden_states_parity_with_converted_weights(
    mmbert_config, reference_batch, pytorch_state_dict
):
    """Test hidden states at each layer match PyTorch reference."""
    flax_params = convert_pytorch_state_dict_to_flax(pytorch_state_dict, mmbert_config)
    
    model = ModernBertForMaskedLM(mmbert_config)
    outputs = model.apply(
        flax_params,
        input_ids=reference_batch["input_ids"],
        attention_mask=reference_batch["attention_mask"],
        deterministic=True,
        output_hidden_states=True,
    )
    
    # PyTorch hidden_states: [num_layers+1, batch, seq, hidden]
    # Flax hidden_states: tuple of (batch, seq, hidden) tensors
    pt_hidden = reference_batch["hidden_states"]
    assert outputs.hidden_states is not None
    
    for i, flax_h in enumerate(outputs.hidden_states):
        pt_h = pt_hidden[i]
        flax_h_np = np.asarray(flax_h)
        
        max_diff = np.max(np.abs(flax_h_np - pt_h))
        mean_diff = np.mean(np.abs(flax_h_np - pt_h))
        
        print(f"\nLayer {i} hidden state:")
        print(f"  Max absolute diff: {max_diff:.6e}")
        print(f"  Mean absolute diff: {mean_diff:.6e}")
        
        assert max_diff < 1e-3, f"Layer {i} hidden differs by {max_diff:.6e}"


@pytest.mark.skipif(not BATCH_PATH.exists(), reason="Reference batch not available")
def test_last_hidden_state_matches_reference(mmbert_config, reference_batch, pytorch_state_dict):
    """Focused test on final encoder output."""
    flax_params = convert_pytorch_state_dict_to_flax(pytorch_state_dict, mmbert_config)
    
    model = ModernBertForMaskedLM(mmbert_config)
    outputs = model.apply(
        flax_params,
        input_ids=reference_batch["input_ids"],
        attention_mask=reference_batch["attention_mask"],
        deterministic=True,
    )
    
    # Access last hidden from encoder directly for parity
    encoder = ModernBertModel(mmbert_config)
    encoder_output = encoder.apply(
        {"params": flax_params["params"]["model"]},
        input_ids=reference_batch["input_ids"],
        attention_mask=reference_batch["attention_mask"],
        deterministic=True,
    )
    
    flax_last = np.asarray(encoder_output.last_hidden_state)

    pt_pre_final = reference_batch["hidden_states"][-1]
    scale = pytorch_state_dict["model.final_norm.weight"]
    bias = pytorch_state_dict.get("model.final_norm.bias")
    mean = pt_pre_final.mean(axis=-1, keepdims=True)
    var = pt_pre_final.var(axis=-1, keepdims=True)
    pt_last = (pt_pre_final - mean) / np.sqrt(var + mmbert_config.norm_eps)
    pt_last = pt_last * scale
    if bias is not None:
        pt_last = pt_last + bias
    
    max_diff = np.max(np.abs(flax_last - pt_last))
    mean_diff = np.mean(np.abs(flax_last - pt_last))
    
    print(f"\nLast hidden state:")
    print(f"  Max absolute diff: {max_diff:.6e}")
    print(f"  Mean absolute diff: {mean_diff:.6e}")
    
    assert max_diff < 1e-3, f"Last hidden differs by {max_diff:.6e}"
