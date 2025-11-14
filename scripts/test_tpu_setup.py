#!/usr/bin/env python3
"""
Quick test script to verify TPU setup and run basic benchmarks.
Checks JAX installation, TPU availability, and runs a minimal training step.
"""

import time
import sys
from typing import Dict, Any
import numpy as np

def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f"{title:^60}")
    print('='*60)

def check_imports() -> bool:
    """Check if all required imports are available."""
    print_section("Checking Required Imports")

    required_modules = {
        'jax': 'JAX',
        'flax': 'Flax',
        'optax': 'Optax',
        'transformers': 'Transformers',
        'datasets': 'Datasets'
    }

    all_available = True
    for module, name in required_modules.items():
        try:
            __import__(module)
            print(f"✓ {name:20} available")
        except ImportError as e:
            print(f"✗ {name:20} NOT available - {e}")
            all_available = False

    return all_available

def check_tpu_devices():
    """Check TPU device availability and configuration."""
    import jax

    print_section("TPU Configuration")

    # Basic info
    backend = jax.default_backend()
    devices = jax.devices()
    process_count = jax.process_count()
    process_index = jax.process_index()

    print(f"JAX version:        {jax.__version__}")
    print(f"Backend:            {backend}")
    print(f"Process count:      {process_count}")
    print(f"Process index:      {process_index}")
    print(f"Device count:       {len(devices)}")

    if backend != 'tpu':
        print(f"\n⚠️  WARNING: Expected TPU backend, got '{backend}'")
        print("JAX will use CPU/GPU fallback mode")
        return False

    print(f"\n✓ TPU backend detected!")
    print(f"\nTPU Devices:")
    for i, device in enumerate(devices):
        print(f"  [{i}] {device}")
        # Try to get more device info
        try:
            from jax._src.lib import xla_client
            local_devices = xla_client.get_local_devices()
            if i < len(local_devices):
                device_kind = local_devices[i].device_kind
                print(f"      Type: {device_kind}")
        except:
            pass

    return True

def run_simple_computation():
    """Run a simple JAX computation to verify everything works."""
    import jax
    import jax.numpy as jnp
    from jax import random

    print_section("Simple Computation Test")

    # Create random data
    key = random.PRNGKey(42)
    size = 1000
    x = random.normal(key, (size, size))

    # JIT compile a simple function
    @jax.jit
    def compute(x):
        return jnp.dot(x, x.T).mean()

    # Warm up
    print("Warming up JIT compilation...")
    _ = compute(x)

    # Benchmark
    print("Running benchmark (10 iterations)...")
    start = time.time()
    for _ in range(10):
        result = compute(x).block_until_ready()
    elapsed = time.time() - start

    print(f"✓ Computation successful!")
    print(f"  Matrix size: {size}x{size}")
    print(f"  Time for 10 iterations: {elapsed:.3f}s")
    print(f"  Time per iteration: {elapsed/10:.3f}s")
    print(f"  TFLOPS estimate: {(2*size**3*10/elapsed)/1e12:.2f}")

    return True

def test_model_loading():
    """Test loading and running the ModernBERT model."""
    print_section("Model Loading Test")

    try:
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        from src.mmbjax.configuration_mmbbert import MmBertConfig
        from src.mmbjax.modeling_mmbbert import FlaxMmBertForPreTraining
        import jax
        import jax.numpy as jnp
        from jax import random

        print("Creating model configuration...")
        config = MmBertConfig(
            d_model=256,
            n_layers=6,
            n_heads=8,
            vocab_size=20000,
            max_seq_len=512
        )

        print("Initializing model...")
        model = FlaxMmBertForPreTraining(config)

        # Create dummy inputs
        batch_size = 4
        seq_len = 128
        key = random.PRNGKey(0)
        params_key, dropout_key = random.split(key)

        input_ids = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
        attention_mask = jnp.ones((batch_size, seq_len), dtype=jnp.int32)

        # Initialize parameters
        variables = model.init(
            params_key,
            input_ids=input_ids,
            attention_mask=attention_mask,
            deterministic=True
        )

        print("Running forward pass...")

        # JIT compile forward pass
        @jax.jit
        def forward(variables, input_ids, attention_mask):
            return model.apply(
                variables,
                input_ids=input_ids,
                attention_mask=attention_mask,
                deterministic=True
            )

        # Run forward pass
        start = time.time()
        outputs = forward(variables, input_ids, attention_mask)
        outputs.prediction_logits.block_until_ready()
        elapsed = time.time() - start

        print(f"✓ Model forward pass successful!")
        print(f"  Model size: {config.n_layers} layers, {config.d_model} hidden dim")
        print(f"  Batch size: {batch_size}, Sequence length: {seq_len}")
        print(f"  Forward pass time: {elapsed:.3f}s")
        print(f"  Output shape: {outputs.prediction_logits.shape}")

        # Count parameters
        param_count = sum(x.size for x in jax.tree_util.tree_leaves(variables['params']))
        print(f"  Total parameters: {param_count:,}")

        return True

    except Exception as e:
        print(f"✗ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_loading():
    """Test HuggingFace dataset loading."""
    print_section("Data Loading Test")

    try:
        from datasets import load_dataset
        import time

        print("Loading sample from mc4_nl_cleaned dataset...")
        start = time.time()

        # Load just one sample to test
        dataset = load_dataset(
            "yhavinga/mc4_nl_cleaned",
            split="train",
            streaming=True
        )

        # Get first sample
        sample = next(iter(dataset))
        elapsed = time.time() - start

        print(f"✓ Dataset loading successful!")
        print(f"  Dataset: yhavinga/mc4_nl_cleaned")
        print(f"  Time to load first sample: {elapsed:.2f}s")
        print(f"  Sample text length: {len(sample['text'])} chars")
        print(f"  Sample preview: {sample['text'][:100]}...")

        return True

    except Exception as e:
        print(f"✗ Data loading failed: {e}")
        return False

def main():
    """Run all tests."""
    print("="*60)
    print("mbert-jax TPU Setup Verification")
    print("="*60)

    results = {}

    # Run tests
    results['imports'] = check_imports()

    if not results['imports']:
        print("\n⚠️  Missing required packages.")
        sys.exit(1)

    results['tpu'] = check_tpu_devices()
    results['computation'] = run_simple_computation()
    results['model'] = test_model_loading()
    results['data'] = test_data_loading()

    # Summary
    print_section("Test Summary")

    all_passed = True
    for test, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test.capitalize():20} {status}")
        all_passed = all_passed and passed

    print("\n" + "="*60)
    if all_passed:
        print("✓ All tests passed! TPU is ready for training.")
        print("\nYou can now start training with:")
        print("  python scripts/train_minimal.py")
    else:
        print("⚠️  Some tests failed. Please check the output above.")
        print("You may still be able to train, but with reduced functionality.")
    print("="*60)

    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())