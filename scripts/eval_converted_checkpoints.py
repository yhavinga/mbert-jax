#!/usr/bin/env python3
"""
Evaluate converted checkpoints (from .pkl files) on Dutch validation data.
This script loads the pre-converted checkpoint files and calculates MLM metrics.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pickle
import json
import numpy as np
import jax
import jax.numpy as jnp
from typing import Dict, Any, Tuple
from datasets import load_dataset
from transformers import AutoTokenizer
import optax
from tqdm import tqdm

from mmbjax.configuration_mmbbert import MMBertConfig
from mmbjax.heads.masked_lm import ModernBertForMaskedLM


def load_converted_checkpoint(checkpoint_path: Path) -> Tuple[Dict[str, Any], MMBertConfig]:
    """Load a converted checkpoint from pickle file."""
    print(f"Loading checkpoint from {checkpoint_path}")

    # Load params
    params_path = checkpoint_path / "params.pkl"
    with open(params_path, 'rb') as f:
        params = pickle.load(f)

    # Load config
    config_path = checkpoint_path / "config.json"
    with open(config_path, 'r') as f:
        config_dict = json.load(f)

    # Create config object
    config = MMBertConfig(
        vocab_size=config_dict.get("vocab_size", 256000),
        hidden_size=config_dict.get("hidden_size", 768),
        num_hidden_layers=config_dict.get("num_hidden_layers", 6),
        num_attention_heads=config_dict.get("num_attention_heads", 12),
        intermediate_size=config_dict.get("intermediate_size", 3072),
        max_position_embeddings=config_dict.get("max_position_embeddings", 8192),
    )

    print(f"  Loaded config: {config.num_hidden_layers} layers, {config.hidden_size} hidden size")

    # Count parameters
    def count_params(tree):
        return sum(x.size for x in jax.tree_util.tree_leaves(tree) if hasattr(x, 'size'))

    total_params = count_params(params)
    print(f"  Total parameters: {total_params:,}")

    return params, config


def prepare_mlm_batch(batch, tokenizer, mlm_probability=0.15):
    """Prepare MLM inputs from a batch of text."""
    # Tokenize
    encoded = tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="np"
    )

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    # Create MLM labels
    labels = input_ids.copy()

    # Create probability matrix
    probability_matrix = np.full(labels.shape, mlm_probability)

    # Don't mask special tokens
    special_tokens_mask = np.isin(input_ids, tokenizer.all_special_ids)
    probability_matrix[special_tokens_mask] = 0

    # Create mask
    masked_indices = np.random.binomial(1, probability_matrix).astype(bool)

    # 80% of the time, replace with [MASK]
    indices_replaced = (
        np.random.binomial(1, 0.8, size=labels.shape).astype(bool)
        & masked_indices
    )
    input_ids[indices_replaced] = tokenizer.mask_token_id

    # 10% of the time, replace with random word
    indices_random = (
        np.random.binomial(1, 0.5, size=labels.shape).astype(bool)
        & masked_indices & ~indices_replaced
    )
    random_words = np.random.randint(len(tokenizer), size=labels.shape)
    input_ids[indices_random] = random_words[indices_random]

    # The rest of the time (10%), keep original

    # Only compute loss on masked tokens
    labels[~masked_indices] = -100

    return {
        "input_ids": jnp.array(input_ids),
        "attention_mask": jnp.array(attention_mask),
        "labels": jnp.array(labels)
    }


def evaluate_checkpoint(params, config, tokenizer, dataset, num_samples=1000):
    """Evaluate a checkpoint on MLM task."""
    print(f"\nEvaluating on {num_samples} samples...")

    # Create model
    model = ModernBertForMaskedLM(config)

    # Prepare data
    total_loss = 0.0
    total_accuracy = 0.0
    total_tokens = 0

    # Process in batches
    batch_size = 8
    num_batches = (num_samples + batch_size - 1) // batch_size

    for i in tqdm(range(num_batches), desc="Evaluating"):
        # Get batch of texts
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, num_samples)
        batch_texts = dataset["text"][start_idx:end_idx]

        # Prepare MLM inputs
        batch = prepare_mlm_batch(
            {"text": batch_texts},
            tokenizer,
            mlm_probability=0.15
        )

        # Forward pass
        outputs = model.apply(
            {"params": params},
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            deterministic=True
        )

        # Accumulate metrics
        batch_loss = outputs.loss

        # Calculate accuracy on masked tokens only
        predictions = jnp.argmax(outputs.logits, axis=-1)
        mask = batch["labels"] != -100

        if jnp.any(mask):
            correct = (predictions == batch["labels"]) & mask
            batch_accuracy = jnp.sum(correct) / jnp.sum(mask)

            total_loss += float(batch_loss) * jnp.sum(mask)
            total_accuracy += float(batch_accuracy) * jnp.sum(mask)
            total_tokens += int(jnp.sum(mask))

    # Calculate final metrics
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    avg_accuracy = total_accuracy / total_tokens if total_tokens > 0 else 0.0
    perplexity = np.exp(avg_loss)

    return {
        "loss": avg_loss,
        "accuracy": avg_accuracy,
        "perplexity": perplexity,
        "total_tokens": total_tokens
    }


def main():
    print("=" * 60)
    print("EVALUATING CONVERTED CHECKPOINTS")
    print("=" * 60)

    # Setup paths
    converted_dir = Path(__file__).parent.parent / "experiments" / "converted"
    red_path = converted_dir / "red_1m"
    yellow_path = converted_dir / "yellow_2m"

    # Check paths exist
    if not red_path.exists():
        print(f"❌ Red checkpoint not found at {red_path}")
        return
    if not yellow_path.exists():
        print(f"❌ Yellow checkpoint not found at {yellow_path}")
        return

    print(f"✓ Found converted checkpoints at {converted_dir}")

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("jhu-clsp/mmBERT-small")

    # Load validation dataset
    print("\nLoading Dutch validation data...")
    dataset = load_dataset(
        "yhavinga/mc4_nl_cleaned",
        "micro",
        split="validation",
        streaming=False
    )

    # Take subset for evaluation
    num_samples = 500  # Start with 500 samples for speed
    dataset = dataset.select(range(min(num_samples, len(dataset))))
    print(f"  Using {len(dataset)} validation samples")

    # Load and evaluate Red checkpoint (1M steps)
    print("\n" + "=" * 40)
    print("RED CHECKPOINT (1M steps - Stable)")
    print("=" * 40)
    red_params, red_config = load_converted_checkpoint(red_path)
    red_metrics = evaluate_checkpoint(
        red_params, red_config, tokenizer, dataset, num_samples=len(dataset)
    )

    # Load and evaluate Yellow checkpoint (2M steps)
    print("\n" + "=" * 40)
    print("YELLOW CHECKPOINT (2M steps - Exploded)")
    print("=" * 40)
    yellow_params, yellow_config = load_converted_checkpoint(yellow_path)
    yellow_metrics = evaluate_checkpoint(
        yellow_params, yellow_config, tokenizer, dataset, num_samples=len(dataset)
    )

    # Print comparison
    print("\n" + "=" * 60)
    print("QUALITY COMPARISON RESULTS")
    print("=" * 60)

    print("\n| Metric | Red (1M) | Yellow (2M) | Winner |")
    print("|--------|----------|-------------|---------|")

    # Loss comparison
    red_loss = red_metrics["loss"]
    yellow_loss = yellow_metrics["loss"]
    loss_winner = "Red ✓" if red_loss < yellow_loss else "Yellow ✓"
    print(f"| Loss | {red_loss:.4f} | {yellow_loss:.4f} | {loss_winner} |")

    # Accuracy comparison
    red_acc = red_metrics["accuracy"] * 100
    yellow_acc = yellow_metrics["accuracy"] * 100
    acc_winner = "Red ✓" if red_acc > yellow_acc else "Yellow ✓"
    print(f"| Accuracy | {red_acc:.2f}% | {yellow_acc:.2f}% | {acc_winner} |")

    # Perplexity comparison
    red_ppl = red_metrics["perplexity"]
    yellow_ppl = yellow_metrics["perplexity"]
    ppl_winner = "Red ✓" if red_ppl < yellow_ppl else "Yellow ✓"
    print(f"| Perplexity | {red_ppl:.2f} | {yellow_ppl:.2f} | {ppl_winner} |")

    # Analysis
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)

    if red_ppl < yellow_ppl:
        ppl_diff = (yellow_ppl - red_ppl) / red_ppl * 100
        print(f"✓ Red (stable) performs {ppl_diff:.1f}% better in perplexity")
        print("  → Gradient explosion significantly hurt model quality")
    else:
        ppl_diff = (red_ppl - yellow_ppl) / yellow_ppl * 100
        print(f"⚠ Yellow (exploded) performs {ppl_diff:.1f}% better in perplexity")
        print("  → Surprising: explosion didn't prevent learning")

    if red_ppl < 10 and yellow_ppl < 10:
        print("\n✓ Both models show reasonable perplexity (< 10)")
        print("  → Core training setup is working")
    elif red_ppl < 10:
        print("\n✓ Red model shows good perplexity (< 10)")
        print("  → Stable training produces usable model")
    else:
        print("\n⚠ High perplexity values indicate potential issues")
        print("  → May need to investigate training setup")

    # Save results
    results_path = Path(__file__).parent.parent / "analysis" / "perplexity_evaluation.json"
    results_path.parent.mkdir(exist_ok=True)

    # Convert metrics to JSON-serializable format
    def to_python_types(obj):
        """Convert JAX/numpy types to Python native types."""
        if isinstance(obj, dict):
            return {k: to_python_types(v) for k, v in obj.items()}
        elif hasattr(obj, 'item'):  # numpy/jax scalar
            return obj.item()
        elif hasattr(obj, 'tolist'):  # numpy/jax array
            return obj.tolist()
        else:
            return obj

    import json
    with open(results_path, 'w') as f:
        json.dump({
            "red_1m": to_python_types(red_metrics),
            "yellow_2m": to_python_types(yellow_metrics),
            "winner": "red" if red_ppl < yellow_ppl else "yellow",
            "evaluation_samples": len(dataset)
        }, f, indent=2)

    print(f"\n✓ Results saved to {results_path}")

    # Recommendation
    print("\n" + "=" * 60)
    print("RECOMMENDATION")
    print("=" * 60)

    if red_ppl < yellow_ppl and red_ppl < 10:
        print("→ Use RED checkpoint (1M steps) for production")
        print("  Reason: More stable training, better perplexity")
    elif yellow_ppl < red_ppl and yellow_ppl < 10:
        print("→ Use YELLOW checkpoint (2M steps) with caution")
        print("  Reason: Better perplexity despite instability")
        print("  Warning: Parameter explosion may cause issues")
    else:
        print("→ Neither checkpoint is production-ready")
        print("  Action: Debug training setup and retrain")


if __name__ == "__main__":
    main()