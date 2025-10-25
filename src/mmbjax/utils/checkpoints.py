"""Checkpoint conversion utilities for PyTorch → Flax."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping

import jax.numpy as jnp
import numpy as np

from ..configuration_mmbbert import MMBertConfig


def load_hf_config_as_mmbert(source: str | Path | Mapping[str, Any]) -> MMBertConfig:
    """Load HuggingFace ModernBERT config (path or dict) and map to MMBertConfig."""
    if isinstance(source, (str, Path)):
        with Path(source).open("r", encoding="utf-8") as f:
            hf_cfg = json.load(f)
    elif isinstance(source, Mapping):
        hf_cfg = dict(source)
    else:
        raise TypeError(f"Unsupported config source type: {type(source)!r}")
    
    # Map HF field names to our config
    mapped = {
        "vocab_size": hf_cfg["vocab_size"],
        "hidden_size": hf_cfg["hidden_size"],
        "intermediate_size": hf_cfg["intermediate_size"],
        "num_hidden_layers": hf_cfg["num_hidden_layers"],
        "num_attention_heads": hf_cfg["num_attention_heads"],
        "hidden_activation": hf_cfg["hidden_activation"],
        "max_position_embeddings": hf_cfg["max_position_embeddings"],
        "initializer_range": hf_cfg["initializer_range"],
        "initializer_cutoff_factor": hf_cfg["initializer_cutoff_factor"],
        "norm_eps": hf_cfg.get("norm_eps", hf_cfg.get("layer_norm_eps", 1e-5)),
        "norm_bias": hf_cfg["norm_bias"],
        "pad_token_id": hf_cfg["pad_token_id"],
        "eos_token_id": hf_cfg["eos_token_id"],
        "bos_token_id": hf_cfg["bos_token_id"],
        "cls_token_id": hf_cfg.get("cls_token_id", hf_cfg["bos_token_id"]),
        "sep_token_id": hf_cfg.get("sep_token_id", hf_cfg["eos_token_id"]),
        "global_rope_theta": hf_cfg["global_rope_theta"],
        "attention_bias": hf_cfg["attention_bias"],
        "attention_dropout": hf_cfg["attention_dropout"],
        "global_attn_every_n_layers": hf_cfg["global_attn_every_n_layers"],
        "local_attention": hf_cfg["local_attention"],
        "local_rope_theta": hf_cfg.get("local_rope_theta"),
        "embedding_dropout": hf_cfg["embedding_dropout"],
        "mlp_bias": hf_cfg["mlp_bias"],
        "mlp_dropout": hf_cfg["mlp_dropout"],
        "decoder_bias": hf_cfg["decoder_bias"],
        "classifier_pooling": hf_cfg["classifier_pooling"],
        "classifier_dropout": hf_cfg["classifier_dropout"],
        "classifier_bias": hf_cfg["classifier_bias"],
        "classifier_activation": hf_cfg["classifier_activation"],
        "deterministic_flash_attn": hf_cfg.get("deterministic_flash_attn", False),
        "sparse_prediction": hf_cfg.get("sparse_prediction", False),
        "sparse_pred_ignore_index": hf_cfg.get("sparse_pred_ignore_index", -100),
        "repad_logits_with_grad": hf_cfg.get("repad_logits_with_grad", False),
        "num_labels": hf_cfg.get("num_labels", 2),
        "problem_type": hf_cfg.get("problem_type"),
        "output_attentions": hf_cfg.get("output_attentions", False),
        "output_hidden_states": hf_cfg.get("output_hidden_states", False),
        "use_return_dict": hf_cfg.get("return_dict", True),
    }
    
    return MMBertConfig(**mapped)


def convert_pytorch_state_dict_to_flax(
    pytorch_state_dict: Dict[str, np.ndarray],
    config: MMBertConfig,
) -> Dict[str, Any]:
    """Convert PyTorch ModernBERT state dict to Flax parameter tree.
    
    PyTorch naming (from transformers 4.57.1 ModernBERT):
      - model.embeddings.tok_embeddings.weight
      - model.embeddings.norm.weight, .bias
      - model.layers.{i}.attn.Wqkv.weight, .bias (optional)
      - model.layers.{i}.attn.Wo.weight, .bias (optional)
      - model.layers.{i}.attn_norm.weight, .bias (optional, layer 0 missing)
      - model.layers.{i}.mlp.Wi.weight, .bias (optional)
      - model.layers.{i}.mlp.Wo.weight, .bias (optional)
      - model.layers.{i}.mlp_norm.weight, .bias (optional)
      - model.final_norm.weight, .bias (optional)
      - head.dense.weight, .bias (optional)
      - head.norm.weight, .bias (optional)
      - decoder.weight (tied to embeddings)
      - decoder.bias (optional)
    
    Flax naming (our implementation):
      - embeddings/word_embeddings/embedding
      - embeddings/layer_norm/scale, /bias
      - encoder/layers_{i}/attention/qkv/kernel, /bias
      - encoder/layers_{i}/attention/out/kernel, /bias
      - encoder/layers_{i}/attention_norm/scale, /bias
      - encoder/layers_{i}/mlp/wi/kernel, /bias
      - encoder/layers_{i}/mlp/wo/kernel, /bias
      - encoder/layers_{i}/mlp_norm/scale, /bias
      - encoder/final_layer_norm/scale, /bias
      - prediction_head/dense/kernel, /bias
      - prediction_head/layer_norm/scale, /bias
      - decoder/bias
    """
    params = {}
    
    # Embeddings
    params["embeddings"] = {
        "word_embeddings": {
            "embedding": _to_flax(pytorch_state_dict["model.embeddings.tok_embeddings.weight"])
        },
        "layer_norm": {
            "scale": _to_flax(pytorch_state_dict["model.embeddings.norm.weight"])
        }
    }
    if "model.embeddings.norm.bias" in pytorch_state_dict:
        params["embeddings"]["layer_norm"]["bias"] = _to_flax(
            pytorch_state_dict["model.embeddings.norm.bias"]
        )
    
    # Encoder layers
    params["encoder"] = {}
    for i in range(config.num_hidden_layers):
        layer_params = {}
        
        # Attention
        qkv_weight = pytorch_state_dict[f"model.layers.{i}.attn.Wqkv.weight"]
        layer_params["attention"] = {
            "qkv": {"kernel": _transpose_dense(qkv_weight)},
            "out": {"kernel": _transpose_dense(pytorch_state_dict[f"model.layers.{i}.attn.Wo.weight"])},
        }
        if f"model.layers.{i}.attn.Wqkv.bias" in pytorch_state_dict:
            layer_params["attention"]["qkv"]["bias"] = _to_flax(
                pytorch_state_dict[f"model.layers.{i}.attn.Wqkv.bias"]
            )
        if f"model.layers.{i}.attn.Wo.bias" in pytorch_state_dict:
            layer_params["attention"]["out"]["bias"] = _to_flax(
                pytorch_state_dict[f"model.layers.{i}.attn.Wo.bias"]
            )
        
        # Attention norm (layer 0 doesn't have this in PyTorch)
        if f"model.layers.{i}.attn_norm.weight" in pytorch_state_dict:
            layer_params["attention_norm"] = {
                "scale": _to_flax(pytorch_state_dict[f"model.layers.{i}.attn_norm.weight"])
            }
            if f"model.layers.{i}.attn_norm.bias" in pytorch_state_dict:
                layer_params["attention_norm"]["bias"] = _to_flax(
                    pytorch_state_dict[f"model.layers.{i}.attn_norm.bias"]
                )
        else:
            # Layer 0: create identity norm parameters
            layer_params["attention_norm"] = {
                "scale": jnp.ones((config.hidden_size,), dtype=jnp.float32)
            }
            if config.norm_bias:
                layer_params["attention_norm"]["bias"] = jnp.zeros((config.hidden_size,), dtype=jnp.float32)
        
        # MLP
        layer_params["mlp"] = {
            "wi": {"kernel": _transpose_dense(pytorch_state_dict[f"model.layers.{i}.mlp.Wi.weight"])},
            "wo": {"kernel": _transpose_dense(pytorch_state_dict[f"model.layers.{i}.mlp.Wo.weight"])},
        }
        if f"model.layers.{i}.mlp.Wi.bias" in pytorch_state_dict:
            layer_params["mlp"]["wi"]["bias"] = _to_flax(
                pytorch_state_dict[f"model.layers.{i}.mlp.Wi.bias"]
            )
        if f"model.layers.{i}.mlp.Wo.bias" in pytorch_state_dict:
            layer_params["mlp"]["wo"]["bias"] = _to_flax(
                pytorch_state_dict[f"model.layers.{i}.mlp.Wo.bias"]
            )
        
        # MLP norm
        layer_params["mlp_norm"] = {
            "scale": _to_flax(pytorch_state_dict[f"model.layers.{i}.mlp_norm.weight"])
        }
        if f"model.layers.{i}.mlp_norm.bias" in pytorch_state_dict:
            layer_params["mlp_norm"]["bias"] = _to_flax(
                pytorch_state_dict[f"model.layers.{i}.mlp_norm.bias"]
            )
        
        params["encoder"][f"layers_{i}"] = layer_params
    
    # Final encoder norm
    params["encoder"]["final_layer_norm"] = {
        "scale": _to_flax(pytorch_state_dict["model.final_norm.weight"])
    }
    if "model.final_norm.bias" in pytorch_state_dict:
        params["encoder"]["final_layer_norm"]["bias"] = _to_flax(
            pytorch_state_dict["model.final_norm.bias"]
        )
    
    # MLM prediction head (if present)
    if "head.dense.weight" in pytorch_state_dict:
        params["prediction_head"] = {
            "dense": {"kernel": _transpose_dense(pytorch_state_dict["head.dense.weight"])},
            "layer_norm": {"scale": _to_flax(pytorch_state_dict["head.norm.weight"])},
        }
        if "head.dense.bias" in pytorch_state_dict:
            params["prediction_head"]["dense"]["bias"] = _to_flax(
                pytorch_state_dict["head.dense.bias"]
            )
        if "head.norm.bias" in pytorch_state_dict:
            params["prediction_head"]["layer_norm"]["bias"] = _to_flax(
                pytorch_state_dict["head.norm.bias"]
            )
        
        # Decoder bias (weight is tied to embeddings)
        params["decoder"] = {}
        if "decoder.bias" in pytorch_state_dict:
            params["decoder"]["bias"] = _to_flax(pytorch_state_dict["decoder.bias"])
    
    # Wrap model params under "model" key for ModernBertForMaskedLM structure
    flax_params = {
        "model": {
            "embeddings": params["embeddings"],
            "encoder": params["encoder"],
        }
    }
    
    # Add head params if present
    if "prediction_head" in params:
        flax_params["prediction_head"] = params["prediction_head"]
    if "decoder" in params:
        flax_params["decoder"] = params["decoder"]
    
    return {"params": flax_params}


def _to_flax(tensor: np.ndarray) -> jnp.ndarray:
    """Convert numpy array to JAX array."""
    return jnp.array(tensor)


def _transpose_dense(weight: np.ndarray) -> jnp.ndarray:
    """Transpose PyTorch dense weight [out, in] to Flax [in, out]."""
    return jnp.array(weight.T)


__all__ = [
    "load_hf_config_as_mmbert",
    "convert_pytorch_state_dict_to_flax",
]
