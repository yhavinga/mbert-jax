"""Parity tests against PyTorch ModernBERT reference outputs."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

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
PT_CKPT_PATH = Path.home() / ".cache/huggingface/hub"  # default HF cache


@dataclass(frozen=True)
class ReferenceCase:
    model_id: str
    safe_name: str
    config_path: Path
    batch_path: Path
    expected: Dict[str, int | bool]


REFERENCE_CASES = [
    ReferenceCase(
        model_id="jhu-clsp/mmBERT-small",
        safe_name="jhu-clsp_mmBERT-small",
        config_path=ARTIFACTS_DIR / "jhu-clsp_mmBERT-small_config.json",
        batch_path=ARTIFACTS_DIR / "jhu-clsp_mmBERT-small_batch.npz",
        expected={
            "vocab_size": 256000,
            "hidden_size": 384,
            "intermediate_size": 1152,
            "num_hidden_layers": 22,
            "num_attention_heads": 6,
            "head_dim": 64,
            "max_position_embeddings": 8192,
            "global_attn_every_n_layers": 3,
            "local_attention": 128,
            "global_rope_theta": 160000,
            "local_rope_theta": 160000,
            "attention_bias": False,
            "mlp_bias": False,
            "norm_bias": False,
            "decoder_bias": True,
        },
    ),
    ReferenceCase(
        model_id="jhu-clsp/mmBERT-base",
        safe_name="jhu-clsp_mmBERT-base",
        config_path=ARTIFACTS_DIR / "jhu-clsp_mmBERT-base_config.json",
        batch_path=ARTIFACTS_DIR / "jhu-clsp_mmBERT-base_batch.npz",
        expected={
            "vocab_size": 256000,
            "hidden_size": 768,
            "intermediate_size": 1152,
            "num_hidden_layers": 22,
            "num_attention_heads": 12,
            "head_dim": 64,
            "max_position_embeddings": 8192,
            "global_attn_every_n_layers": 3,
            "local_attention": 128,
            "global_rope_theta": 160000,
            "local_rope_theta": 160000,
            "attention_bias": False,
            "mlp_bias": False,
            "norm_bias": False,
            "decoder_bias": True,
        },
    ),
]


@pytest.fixture(scope="module", params=REFERENCE_CASES, ids=lambda case: case.safe_name)
def reference_case(request):
    """Parameterize tests over all available reference artifacts."""
    case = request.param
    missing = []
    if not case.config_path.exists():
        missing.append(case.config_path)
    if not case.batch_path.exists():
        missing.append(case.batch_path)
    if missing:
        pytest.skip(f"Missing reference files for {case.safe_name}: {missing}")
    return case


@pytest.fixture(scope="module")
def mmbert_config(reference_case):
    """Load ModernBERT config for the current reference case."""
    return load_hf_config_as_mmbert(reference_case.config_path)


@pytest.fixture(scope="module")
def reference_batch(reference_case):
    """Load golden PyTorch outputs."""
    data = np.load(reference_case.batch_path)
    return {
        "input_ids": jnp.array(data["input_ids"], dtype=jnp.int32),
        "attention_mask": jnp.array(data["attention_mask"], dtype=jnp.int32),
        "last_hidden_state": data["last_hidden_state"],
        "hidden_states": data["hidden_states"],
        "logits": data["logits"],
    }


def _resolve_snapshot_dir(model_dir: Path) -> Path | None:
    """Return the snapshot directory pointed to by refs/main if available."""
    refs_dir = model_dir / "refs"
    snapshots_dir = model_dir / "snapshots"
    if not snapshots_dir.exists():
        return None

    ref_main = refs_dir / "main"
    if ref_main.exists():
        revision = ref_main.read_text().strip()
        candidate = snapshots_dir / revision
        if candidate.exists():
            return candidate

    snapshot_paths = sorted(p for p in snapshots_dir.iterdir() if p.is_dir())
    return snapshot_paths[0] if snapshot_paths else None


@pytest.fixture(scope="module")
def pytorch_state_dict(reference_case):
    """Load PyTorch checkpoint weights."""
    model_dir = PT_CKPT_PATH / f"models--{reference_case.model_id.replace('/', '--')}"
    snapshot_dir = _resolve_snapshot_dir(model_dir)
    if snapshot_dir is None:
        pytest.skip(f"No snapshot directory found for {reference_case.model_id}")

    pt_file = snapshot_dir / "pytorch_model.bin"
    if not pt_file.exists():
        pytest.skip(f"pytorch_model.bin not found in {snapshot_dir}")

    state_dict = torch.load(pt_file, map_location="cpu")
    # Convert to numpy
    return {k: v.numpy() for k, v in state_dict.items()}


def test_config_loads_correctly(mmbert_config, reference_case):
    """Verify config mapping from HF to MMBertConfig."""
    expected = reference_case.expected
    assert mmbert_config.vocab_size == expected["vocab_size"]
    assert mmbert_config.hidden_size == expected["hidden_size"]
    assert mmbert_config.intermediate_size == expected["intermediate_size"]
    assert mmbert_config.num_hidden_layers == expected["num_hidden_layers"]
    assert mmbert_config.num_attention_heads == expected["num_attention_heads"]
    assert mmbert_config.head_dim == expected["head_dim"]
    assert mmbert_config.max_position_embeddings == expected["max_position_embeddings"]
    assert mmbert_config.global_attn_every_n_layers == expected["global_attn_every_n_layers"]
    assert mmbert_config.local_attention == expected["local_attention"]
    assert mmbert_config.global_rope_theta == expected["global_rope_theta"]
    assert mmbert_config.local_rope_theta == expected["local_rope_theta"]
    assert mmbert_config.attention_bias == expected["attention_bias"]
    assert mmbert_config.mlp_bias == expected["mlp_bias"]
    assert mmbert_config.norm_bias == expected["norm_bias"]
    assert mmbert_config.decoder_bias == expected["decoder_bias"]


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


def test_logits_parity_with_converted_weights(
    mmbert_config, reference_batch, pytorch_state_dict, reference_case
):
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
    
    print(f"\nLogits comparison ({reference_case.safe_name}):")
    print(f"  Max absolute diff: {max_diff:.6e}")
    print(f"  Mean absolute diff: {mean_diff:.6e}")
    print(f"  Max relative diff: {rel_diff:.6e}")
    
    # Initially use loose tolerance; tighten as we debug
    assert max_diff < 1e-3, f"Logits differ by {max_diff:.6e} (max)"
    assert rel_diff < 1e-3, f"Relative logits diff {rel_diff:.6e} exceeds tolerance"


def test_hidden_states_parity_with_converted_weights(
    mmbert_config, reference_batch, pytorch_state_dict, reference_case
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
        pt_scale = np.max(np.abs(pt_h)) + 1e-8
        rel_diff = max_diff / pt_scale
    
        print(f"\n{reference_case.safe_name} layer {i} hidden state:")
        print(f"  Max absolute diff: {max_diff:.6e}")
        print(f"  Mean absolute diff: {mean_diff:.6e}")
        print(f"  Max relative diff: {rel_diff:.6e}")
        
        assert (max_diff < 1e-3) or (
            rel_diff < 1e-4
        ), f"Layer {i} hidden differs by {max_diff:.6e} (rel {rel_diff:.6e})"


def test_last_hidden_state_matches_reference(
    mmbert_config, reference_batch, pytorch_state_dict, reference_case
):
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
    
    print(f"\nLast hidden state ({reference_case.safe_name}):")
    print(f"  Max absolute diff: {max_diff:.6e}")
    print(f"  Mean absolute diff: {mean_diff:.6e}")
    
    assert max_diff < 1e-3, f"Last hidden differs by {max_diff:.6e}"
