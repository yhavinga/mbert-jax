#!/usr/bin/env python3
"""
Convert Flax/JAX checkpoint to HuggingFace ModernBERT format.

This script converts checkpoints from the mmbjax training format
to HuggingFace-compatible ModernBERT format for inference and finetuning.

Usage:
    python scripts/convert_flax_to_hf.py \\
        --input experiments/checkpoints/flax/MODEL_NAME \\
        --output experiments/checkpoints/hf/MODEL_NAME
"""

import argparse
import pickle
import json
import re
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file


def flatten_dict(d, parent_key='', sep='.'):
    """Flatten a nested dictionary."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def map_flax_key_to_hf(flax_key: str) -> str:
    """Map a Flax parameter key to HuggingFace ModernBERT key."""

    # Embeddings
    if flax_key == "model.embeddings.word_embeddings.embedding":
        return "model.embeddings.tok_embeddings.weight"
    if flax_key == "model.embeddings.layer_norm.scale":
        return "model.embeddings.norm.weight"

    # Final layer norm
    if flax_key == "model.encoder.final_layer_norm.scale":
        return "model.final_norm.weight"

    # Prediction head
    if flax_key == "prediction_head.dense.kernel":
        return "head.dense.weight"
    if flax_key == "prediction_head.layer_norm.scale":
        return "head.norm.weight"

    # Decoder bias (same key)
    if flax_key == "decoder.bias":
        return "decoder.bias"

    # Layer-specific mappings
    # Pattern: model.encoder.layers_{i}.{component}
    layer_match = re.match(r"model\.encoder\.layers_(\d+)\.(.+)", flax_key)
    if layer_match:
        layer_idx = layer_match.group(1)
        component = layer_match.group(2)

        # Attention
        if component == "attention.qkv.kernel":
            return f"model.layers.{layer_idx}.attn.Wqkv.weight"
        if component == "attention.out.kernel":
            return f"model.layers.{layer_idx}.attn.Wo.weight"
        if component == "attention_norm.scale":
            return f"model.layers.{layer_idx}.attn_norm.weight"

        # MLP
        if component == "mlp.wi.kernel":
            return f"model.layers.{layer_idx}.mlp.Wi.weight"
        if component == "mlp.wo.kernel":
            return f"model.layers.{layer_idx}.mlp.Wo.weight"
        if component == "mlp_norm.scale":
            return f"model.layers.{layer_idx}.mlp_norm.weight"

    raise ValueError(f"Unknown Flax key: {flax_key}")


def save_tokenizer_with_mask(tokenizer_name: str, output_dir: Path, vocab_size: int):
    """Load tokenizer, add mask token if needed, resize to vocab_size, and save."""
    from transformers import AutoTokenizer

    print(f"\nLoading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Add mask token if missing (matches training script logic)
    if tokenizer.mask_token is None:
        tokenizer.add_special_tokens({"mask_token": "<mask>"})
        print(f"  Added <mask> token at index {tokenizer.mask_token_id}")

    # Ensure vocab size matches model (training may have added padding tokens)
    if len(tokenizer) < vocab_size:
        # Add padding tokens to reach target vocab_size
        num_to_add = vocab_size - len(tokenizer)
        print(f"  Adding {num_to_add} padding tokens to match vocab_size={vocab_size}")
        # These won't be used but ensure vocab sizes match
        tokenizer.add_tokens([f"<unused{i}>" for i in range(num_to_add)])

    tokenizer.save_pretrained(output_dir)
    print(f"✓ Saved tokenizer to {output_dir}")
    print(f"  Vocab size: {len(tokenizer)}, mask_token_id: {tokenizer.mask_token_id}")


def convert_flax_to_hf(input_dir: Path, output_dir: Path, tokenizer_override: str = None, verify: bool = True):
    """Convert Flax checkpoint to HuggingFace format."""

    print(f"Loading Flax checkpoint from {input_dir}")

    # Load Flax params
    params_path = input_dir / "params.pkl"
    if not params_path.exists():
        raise FileNotFoundError(f"params.pkl not found in {input_dir}")

    with open(params_path, 'rb') as f:
        flax_params = pickle.load(f)

    # Load config to get model dimensions
    config_path = input_dir / "config.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        # Default config
        config = {
            "vocab_size": 256000,
            "hidden_size": 768,
            "num_hidden_layers": 6,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "max_position_embeddings": 8192,
        }

    # Flatten the nested dict
    flat_params = flatten_dict(flax_params)

    print(f"Found {len(flat_params)} parameter tensors")

    # Convert to HuggingFace format
    hf_state_dict = {}
    embeddings_weight = None

    for flax_key, tensor in flat_params.items():
        # Convert numpy to tensor
        if isinstance(tensor, np.ndarray):
            tensor_np = tensor
        else:
            tensor_np = np.array(tensor)

        # Map key
        hf_key = map_flax_key_to_hf(flax_key)

        # Transpose kernels (JAX [in, out] -> PyTorch [out, in])
        if ".kernel" in flax_key:
            tensor_np = tensor_np.T

        # Convert to torch tensor
        hf_state_dict[hf_key] = torch.from_numpy(tensor_np.copy())

        # Save embeddings for decoder.weight
        if hf_key == "model.embeddings.tok_embeddings.weight":
            embeddings_weight = hf_state_dict[hf_key]

        print(f"  {flax_key} -> {hf_key} {tuple(tensor_np.shape)}")

    # Add tied decoder.weight (copy of embeddings)
    if embeddings_weight is not None:
        hf_state_dict["decoder.weight"] = embeddings_weight.clone()
        print(f"  (tied) -> decoder.weight {tuple(embeddings_weight.shape)}")

    print(f"\nTotal HF tensors: {len(hf_state_dict)}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save as safetensors
    safetensors_path = output_dir / "model.safetensors"
    save_file(hf_state_dict, str(safetensors_path))
    print(f"\nSaved weights to {safetensors_path}")

    # Write HuggingFace config
    hf_config = {
        "architectures": ["ModernBertForMaskedLM"],
        "model_type": "modernbert",
        "vocab_size": config.get("vocab_size", 256000),
        "hidden_size": config.get("hidden_size", 768),
        "num_hidden_layers": config.get("num_hidden_layers", 6),
        "num_attention_heads": config.get("num_attention_heads", 12),
        "intermediate_size": config.get("intermediate_size", 3072),
        "hidden_activation": "gelu",
        "max_position_embeddings": config.get("max_position_embeddings", 8192),
        "initializer_range": 0.02,
        "norm_eps": 1e-5,
        "pad_token_id": 0,
        "classifier_activation": "gelu",
        "classifier_dropout": 0.0,
        "attention_dropout": 0.0,
        "embedding_dropout": 0.0,
        "mlp_dropout": 0.0,
        "global_attn_every_n_layers": config.get("global_attn_every_n_layers", 3),
        "local_attention": config.get("local_attention", 128),
        "torch_dtype": "float32",
        "transformers_version": "4.48.0",
    }

    config_path = output_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(hf_config, f, indent=2)
    print(f"Saved config to {config_path}")

    # Save tokenizer (use override if provided, otherwise from config)
    tokenizer_name = tokenizer_override or config.get("tokenizer_name")
    if tokenizer_name:
        save_tokenizer_with_mask(tokenizer_name, output_dir, hf_config["vocab_size"])
    else:
        print("\nWARNING: No tokenizer specified. You must provide tokenizer separately.")

    # Verify if requested
    if verify:
        print("\n" + "="*50)
        print("VERIFICATION")
        print("="*50)
        verify_with_transformers(output_dir)

    return output_dir


def verify_with_transformers(output_dir: Path):
    """Verify the converted model loads and runs inference."""
    try:
        from transformers import ModernBertForMaskedLM, AutoTokenizer
    except ImportError:
        print("transformers not installed, skipping verification")
        return

    print(f"\nLoading model from {output_dir}...")
    model = ModernBertForMaskedLM.from_pretrained(output_dir)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Try to load tokenizer from output_dir (if saved with model)
    # Fall back to mmBERT tokenizer for older checkpoints
    tokenizer_path = output_dir / "tokenizer.json"
    if tokenizer_path.exists():
        print(f"Loading tokenizer from {output_dir}...")
        tokenizer = AutoTokenizer.from_pretrained(output_dir)
    else:
        print("Loading tokenizer (jhu-clsp/mmBERT-small, vocab=256k)...")
        tokenizer = AutoTokenizer.from_pretrained("jhu-clsp/mmBERT-small")

    # Test inference - use tokenizer's mask token
    mask = tokenizer.mask_token
    test_sentences = [
        f"Amsterdam is de {mask} van Nederland.",
        f"De kat zit op de {mask}.",
        f"Ik drink graag een kopje {mask}.",
    ]

    print("\nInference test:")
    model.eval()

    for text in test_sentences:
        inputs = tokenizer(text, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)

        # Find mask position
        mask_positions = (inputs.input_ids == tokenizer.mask_token_id).nonzero()
        if len(mask_positions) == 0:
            print(f"  No mask found in: {text}")
            continue

        mask_idx = mask_positions[0, 1]

        # Get top predictions with probabilities
        logits = outputs.logits[0, mask_idx]
        probs = torch.softmax(logits, dim=-1)
        top5_ids = logits.topk(5).indices.tolist()
        top5_probs = probs[top5_ids].tolist()
        top5_tokens = [tokenizer.decode([i]).strip() for i in top5_ids]

        print(f"  '{text}'")
        print(f"    -> {top5_tokens[0]} ({top5_probs[0]:.1%}), "
              f"{top5_tokens[1]} ({top5_probs[1]:.1%}), "
              f"{top5_tokens[2]} ({top5_probs[2]:.1%})")

    print("\n" + "="*50)
    print("VERIFICATION COMPLETE")
    print("="*50)


def main():
    parser = argparse.ArgumentParser(
        description="Convert Flax checkpoint to HuggingFace ModernBERT format"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Input directory containing params.pkl and config.json"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output directory for HuggingFace checkpoint"
    )
    parser.add_argument(
        "--tokenizer", "-t",
        type=str,
        required=False,
        help="Tokenizer name/path (overrides config.json if specified)"
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip verification with transformers"
    )

    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    convert_flax_to_hf(input_dir, output_dir, 
                       tokenizer_override=args.tokenizer if hasattr(args, 'tokenizer') else None,
                       verify=not args.no_verify)

    print(f"\nConversion complete!")
    print(f"Load with: ModernBertForMaskedLM.from_pretrained('{output_dir}')")


if __name__ == "__main__":
    main()
