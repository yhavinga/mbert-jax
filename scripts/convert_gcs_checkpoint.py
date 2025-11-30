#!/usr/bin/env python3
"""Convert Orbax checkpoint from GCS to CPU-friendly format locally.

This script loads multi-host TPU checkpoints on a local CPU and converts
them to pickle + safetensors format for downstream use.

Usage:
    # Download from GCS first
    gcloud storage cp -r gs://bucket/checkpoints/STEP experiments/gcs/STEP

    # Convert locally
    python scripts/convert_gcs_checkpoint.py \
        --checkpoint experiments/gcs/5000 \
        --output experiments/converted/5000
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse
import pickle
import json
import numpy as np

# Force CPU-only JAX (must be before jax import)
import jax
jax.config.update('jax_platforms', 'cpu')

import orbax.checkpoint as ocp
from flax.traverse_util import flatten_dict
from safetensors.numpy import save_file


def load_orbax_checkpoint(checkpoint_path: Path):
    """Load Orbax checkpoint as raw pytree."""
    # Orbax requires absolute paths
    checkpoint_path = checkpoint_path.absolute()
    print(f"Loading from {checkpoint_path}")

    checkpointer = ocp.PyTreeCheckpointer()

    # Orbax stores in 'default' subdirectory
    default_path = checkpoint_path / "default"
    if default_path.exists():
        restore_path = default_path
    else:
        restore_path = checkpoint_path

    raw = checkpointer.restore(restore_path)

    # Navigate checkpoint structure to find params
    if isinstance(raw, dict):
        if "params" in raw:
            print("  Found params at top level")
            return raw["params"]
        elif "state" in raw:
            state = raw["state"]
            if isinstance(state, dict) and "params" in state:
                print("  Found params in state dict")
                return state["params"]
            elif hasattr(state, "params"):
                print("  Found params in state object")
                return state.params
    elif hasattr(raw, "params"):
        print("  Found params attribute")
        return raw.params

    # Debug: show what we got
    if isinstance(raw, dict):
        print(f"  Checkpoint keys: {list(raw.keys())}")
    raise ValueError("Could not find params in checkpoint")


def infer_config_from_params(params) -> dict:
    """Infer model config from parameter shapes."""
    config = {}

    # Navigate to find embedding shape: params.model.embeddings.word_embeddings.embedding
    try:
        emb = params["model"]["embeddings"]["word_embeddings"]["embedding"]
        config["vocab_size"] = emb.shape[0]
        config["hidden_size"] = emb.shape[1]
        print(f"  Inferred vocab_size={config['vocab_size']}, hidden_size={config['hidden_size']} from embeddings")
    except (KeyError, TypeError):
        pass

    # Count layers by looking for layers_N patterns
    try:
        layers = params["model"]["encoder"]
        layer_keys = [k for k in layers.keys() if k.startswith("layers_")]
        config["num_hidden_layers"] = len(layer_keys)
        print(f"  Inferred num_hidden_layers={config['num_hidden_layers']}")
    except (KeyError, TypeError):
        pass

    # Get intermediate_size from MLP wi kernel: [hidden, intermediate*2]
    try:
        wi = params["model"]["encoder"]["layers_0"]["mlp"]["wi"]["kernel"]
        # JAX kernel is [in, out] = [hidden, intermediate*2]
        config["intermediate_size"] = wi.shape[1] // 2  # GeGLU splits in half
        print(f"  Inferred intermediate_size={config['intermediate_size']} from MLP")
    except (KeyError, TypeError):
        pass

    # Get num_heads from QKV kernel: [hidden*3, hidden] -> hidden/64 heads typical
    try:
        qkv = params["model"]["encoder"]["layers_0"]["attention"]["qkv"]["kernel"]
        hidden = config.get("hidden_size", qkv.shape[1])
        # Assuming head_dim=64 (ModernBERT default)
        config["num_attention_heads"] = hidden // 64
        print(f"  Inferred num_attention_heads={config['num_attention_heads']} (assuming head_dim=64)")
    except (KeyError, TypeError):
        pass

    return config


def save_converted(params, output_dir: Path, config: dict):
    """Save params as pickle + safetensors + config.json."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert JAX/numpy arrays to numpy
    def to_numpy(x):
        if hasattr(x, '__array__'):
            return np.array(x)
        return x

    numpy_params = jax.tree_util.tree_map(to_numpy, params)

    # Save pickle (for convert_flax_to_hf.py)
    pickle_path = output_dir / "params.pkl"
    with open(pickle_path, 'wb') as f:
        pickle.dump(numpy_params, f)
    print(f"✓ Saved {pickle_path}")

    # Save safetensors
    flat_params = flatten_dict(numpy_params, sep='.')
    safetensor_dict = {str(k): np.array(v) for k, v in flat_params.items()}
    safetensors_path = output_dir / "model.safetensors"
    save_file(safetensor_dict, str(safetensors_path))
    print(f"✓ Saved {safetensors_path}")

    # Save config
    config_path = output_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"✓ Saved {config_path}")

    # Stats
    total = sum(v.size for v in flat_params.values())
    print(f"\nTotal parameters: {total:,}")


def main():
    parser = argparse.ArgumentParser(description="Convert Orbax checkpoint locally")
    parser.add_argument("--checkpoint", "-c", required=True,
                        help="Path to downloaded Orbax checkpoint directory")
    parser.add_argument("--output", "-o", required=True,
                        help="Output directory for converted checkpoint")
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--vocab-size", type=int, default=256000)
    parser.add_argument("--num-heads", type=int, default=12)
    parser.add_argument("--intermediate-size", type=int, default=3072)
    parser.add_argument("--global-attn-every-n-layers", type=int, default=3)
    parser.add_argument("--tokenizer", type=str, default=None,
                        help="Tokenizer name/path (saved in config for HF conversion)")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output)

    print("="*60)
    print("LOCAL CHECKPOINT CONVERTER")
    print("="*60)
    print(f"JAX devices: {jax.devices()}")
    print(f"Input: {checkpoint_path}")
    print(f"Output: {output_dir}")
    print("="*60)

    # Load
    params = load_orbax_checkpoint(checkpoint_path)

    # Infer config from params
    print("\nInferring config from checkpoint...")
    inferred = infer_config_from_params(params)

    # Build config: inferred values, then CLI overrides for non-default values
    config = {
        "model_type": "modernbert",
        "vocab_size": inferred.get("vocab_size", args.vocab_size),
        "hidden_size": inferred.get("hidden_size", args.hidden_size),
        "num_hidden_layers": inferred.get("num_hidden_layers", args.num_layers),
        "num_attention_heads": inferred.get("num_attention_heads", args.num_heads),
        "intermediate_size": inferred.get("intermediate_size", args.intermediate_size),
        "max_position_embeddings": 8192,
        "global_attn_every_n_layers": args.global_attn_every_n_layers,
        "local_attention": 128,
    }

    # Add tokenizer if specified (for HF conversion)
    if args.tokenizer:
        config["tokenizer_name"] = args.tokenizer
        print(f"  Tokenizer: {args.tokenizer}")

    print(f"\nFinal config: vocab={config['vocab_size']}, hidden={config['hidden_size']}, "
          f"layers={config['num_hidden_layers']}, heads={config['num_attention_heads']}, "
          f"intermediate={config['intermediate_size']}")

    # Save
    save_converted(params, output_dir, config)

    print("\n" + "="*60)
    print("✓ CONVERSION COMPLETE")
    print("="*60)
    print(f"\nNext: python scripts/convert_flax_to_hf.py -i {output_dir} -o hf_output/")


if __name__ == "__main__":
    main()
