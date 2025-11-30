#!/usr/bin/env python3
"""
Convert checkpoint to CPU-friendly format ON THE TPU VM.
This script should be run on the TPU where the checkpoint was created.
It will save the parameters in a format that can be loaded on CPU.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse
import pickle
import json
import numpy as np
import jax
import jax.numpy as jnp
from flax.traverse_util import flatten_dict, unflatten_dict
from safetensors.numpy import save_file
import orbax.checkpoint as ocp
from flax.training import train_state
import optax
from mmbjax.configuration_mmbbert import MMBertConfig
from mmbjax.heads.masked_lm import ModernBertForMaskedLM


def load_checkpoint_on_tpu(checkpoint_path: str, config: MMBertConfig):
    """Load checkpoint on TPU (where it was created)."""
    checkpoint_path = Path(checkpoint_path).absolute()
    print(f"Loading checkpoint from {checkpoint_path}")

    # Create the same model and optimizer used in training
    model = ModernBertForMaskedLM(config)

    # Initialize with dummy data
    rng = jax.random.PRNGKey(0)
    dummy_input = jnp.ones((1, 512), dtype=jnp.int32)
    dummy_mask = jnp.ones((1, 512), dtype=jnp.int32)

    variables = model.init(
        rng,
        input_ids=dummy_input,
        attention_mask=dummy_mask,
        deterministic=True
    )

    # Create optimizer (matching train_minimal.py)
    # Note: Using adamw with warmup to match actual training
    warmup_fn = optax.linear_schedule(
        init_value=0,
        end_value=5e-4,
        transition_steps=10000
    )
    decay_fn = optax.linear_schedule(
        init_value=5e-4,
        end_value=0,
        transition_steps=990000
    )
    lr_schedule = optax.join_schedules(
        schedules=[warmup_fn, decay_fn],
        boundaries=[10000]
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(
            learning_rate=lr_schedule,
            b1=0.9,
            b2=0.999,
            eps=1e-8,
            weight_decay=0.01
        )
    )

    # Create train state
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=optimizer
    )

    # Create checkpoint manager
    checkpoint_manager = ocp.CheckpointManager(
        checkpoint_path.parent,
        options=ocp.CheckpointManagerOptions(max_to_keep=1),
    )

    # Get step from directory name
    step = int(checkpoint_path.name)

    # Restore checkpoint - use partial restore to only get params
    try:
        # Try standard restore first
        restored_state = checkpoint_manager.restore(
            step,
            args=ocp.args.StandardRestore(state)
        )
        print(f"✓ Loaded checkpoint from step {step}")
        return restored_state.params
    except ValueError as e:
        print(f"Standard restore failed: {e}")
        print("Attempting to load params only...")

        # Load raw checkpoint and extract params
        checkpointer = ocp.Checkpointer(ocp.PyTreeCheckpointHandler())
        raw_checkpoint = checkpointer.restore(checkpoint_path / "default")

        # Navigate to find params
        if isinstance(raw_checkpoint, dict):
            if "params" in raw_checkpoint:
                params = raw_checkpoint["params"]
                print("✓ Found params directly")
            elif "state" in raw_checkpoint:
                if hasattr(raw_checkpoint["state"], "params"):
                    params = raw_checkpoint["state"].params
                elif isinstance(raw_checkpoint["state"], dict) and "params" in raw_checkpoint["state"]:
                    params = raw_checkpoint["state"]["params"]
                else:
                    raise ValueError("Could not find params in state")
            else:
                raise ValueError("Could not find params in checkpoint")
        elif hasattr(raw_checkpoint, "params"):
            params = raw_checkpoint.params
        else:
            raise ValueError("Unknown checkpoint structure")

        print(f"✓ Loaded params from step {step}")
        return params


def convert_to_cpu_format(params, output_dir: str, config: MMBertConfig, format: str = "both"):
    """Convert params to CPU-friendly format."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert JAX arrays to numpy arrays
    def to_numpy(x):
        if isinstance(x, jnp.ndarray):
            # Move to CPU and convert to numpy
            return np.array(jax.device_get(x))
        return x

    numpy_params = jax.tree.map(to_numpy, params)

    # Save as pickle
    if format in ["pickle", "both"]:
        pickle_path = output_dir / "params.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(numpy_params, f)
        print(f"✓ Saved pickle to {pickle_path}")

    # Save as safetensors (HF format)
    if format in ["safetensors", "both"]:
        # Flatten for safetensors
        flat_params = flatten_dict(numpy_params, sep='.')

        # Convert keys to strings and ensure numpy arrays
        safetensor_dict = {}
        for key, value in flat_params.items():
            # Create string key
            if isinstance(key, tuple):
                str_key = '.'.join(str(k) for k in key)
            else:
                str_key = str(key)

            # Ensure numpy array
            if not isinstance(value, np.ndarray):
                value = np.array(value)

            safetensor_dict[str_key] = value

        # Save safetensors
        safetensors_path = output_dir / "model.safetensors"
        save_file(safetensor_dict, str(safetensors_path))
        print(f"✓ Saved safetensors to {safetensors_path}")

        # Also save config for HF compatibility
        config_dict = {
            "model_type": "modernbert",
            "vocab_size": config.vocab_size,
            "hidden_size": config.hidden_size,
            "num_hidden_layers": config.num_hidden_layers,
            "num_attention_heads": config.num_attention_heads,
            "intermediate_size": config.intermediate_size,
            "max_position_embeddings": config.max_position_embeddings,
            "global_attn_every_n_layers": config.global_attn_every_n_layers,
            "local_attention": config.local_attention,
        }

        config_path = output_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        print(f"✓ Saved config to {config_path}")

    # Print statistics
    flat_params = flatten_dict(params)
    total_params = sum(
        x.size if hasattr(x, 'size') else len(x)
        for x in flat_params.values()
    )
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Number of parameter tensors: {len(flat_params)}")

    # Component stats
    component_stats = {}
    for key, value in flat_params.items():
        comp = key[0] if isinstance(key, tuple) else str(key).split('.')[0]
        if comp not in component_stats:
            component_stats[comp] = 0
        if hasattr(value, 'size'):
            component_stats[comp] += value.size

    print("\nParameters by component:")
    for comp, size in sorted(component_stats.items()):
        print(f"  {comp}: {size:,}")


def main():
    parser = argparse.ArgumentParser(description="Convert checkpoint on TPU to CPU format")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint directory (e.g., ~/mbert-jax/checkpoints/1000000)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory for converted checkpoint")
    parser.add_argument("--format", type=str, default="both",
                        choices=["pickle", "safetensors", "both"],
                        help="Output format (default: both)")
    parser.add_argument("--num-layers", type=int, default=22,
                        help="Number of model layers")
    parser.add_argument("--hidden-size", type=int, default=512,
                        help="Hidden size")
    parser.add_argument("--vocab-size", type=int, default=32128,
                        help="Vocabulary size")
    parser.add_argument("--num-heads", type=int, default=8,
                        help="Number of attention heads")
    parser.add_argument("--intermediate-size", type=int, default=3072,
                        help="MLP intermediate size")
    args = parser.parse_args()

    print("="*60)
    print("TPU CHECKPOINT CONVERTER")
    print("="*60)
    print(f"Running on: {jax.devices()}")
    print(f"Input checkpoint: {args.checkpoint}")
    print(f"Output directory: {args.output}")
    print(f"Format: {args.format}")
    print("="*60)

    # Create config
    config = MMBertConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        intermediate_size=args.intermediate_size,
        max_position_embeddings=8192,
    )

    # Load checkpoint on TPU
    print("\n1. Loading checkpoint on TPU...")
    params = load_checkpoint_on_tpu(args.checkpoint, config)

    # Convert to CPU format
    print("\n2. Converting to CPU-friendly format...")
    convert_to_cpu_format(params, args.output, config, args.format)

    print("\n" + "="*60)
    print("✓ CONVERSION COMPLETE!")
    print("="*60)
    print(f"\nNext steps:")
    print(f"1. Download the converted files from TPU:")
    print(f"   rsync -avz tpu-vm:{args.output}/ ./converted_checkpoint/")
    print(f"2. Load on CPU with:")
    print(f"   import pickle")
    print(f"   with open('converted_checkpoint/params.pkl', 'rb') as f:")
    print(f"       params = pickle.load(f)")


if __name__ == "__main__":
    main()