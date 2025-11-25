#!/usr/bin/env python3
"""
Unified checkpoint evaluation script with SQLite storage.
Simplifies evaluation to: checkpoint → evaluation → database.

Usage:
    # Evaluate perplexity
    python eval_unified.py --checkpoint experiments/converted/red_1m

    # Evaluate inference
    python eval_unified.py --checkpoint experiments/converted/yellow_2m \
        --eval-type inference --wandb-name vivid-firebrand-18
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse
import json
import pickle
import sqlite3
import time
from datetime import datetime
from typing import Dict, Any, Optional

import numpy as np
import jax
import jax.numpy as jnp
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

from mmbjax.configuration_mmbbert import MMBertConfig
from mmbjax.heads.masked_lm import ModernBertForMaskedLM


def setup_database(db_path: str):
    """Create database and tables if they don't exist."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS evaluations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            checkpoint_path TEXT NOT NULL,
            checkpoint_name TEXT,
            training_step INTEGER,
            wandb_run_name TEXT,
            eval_type TEXT NOT NULL,
            dataset TEXT,
            num_samples INTEGER,
            -- Metrics
            loss REAL,
            accuracy REAL,
            perplexity REAL,
            -- Metadata
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            duration_seconds REAL,
            config TEXT,
            predictions TEXT
        )
    """)

    # Create indices for common queries
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_checkpoint
        ON evaluations(checkpoint_path)
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_eval_type
        ON evaluations(eval_type)
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_perplexity
        ON evaluations(perplexity)
    """)

    conn.commit()
    conn.close()


def load_checkpoint(checkpoint_path: Path) -> tuple[Dict, MMBertConfig]:
    """Load checkpoint from pickle file or Orbax format."""
    checkpoint_path = Path(checkpoint_path)

    # Try pickle format first (converted checkpoints)
    pkl_path = checkpoint_path / "params.pkl"
    config_path = checkpoint_path / "config.json"

    if pkl_path.exists() and config_path.exists():
        with open(pkl_path, 'rb') as f:
            params = pickle.load(f)

        with open(config_path, 'r') as f:
            config_dict = json.load(f)

        config = MMBertConfig(
            vocab_size=config_dict.get("vocab_size", 256000),
            hidden_size=config_dict.get("hidden_size", 768),
            num_hidden_layers=config_dict.get("num_hidden_layers", 6),
            num_attention_heads=config_dict.get("num_attention_heads", 12),
            intermediate_size=config_dict.get("intermediate_size", 3072),
            max_position_embeddings=config_dict.get(
                "max_position_embeddings", 8192
            ),
        )
        return params, config

    raise ValueError(
        f"Could not load checkpoint from {checkpoint_path}. "
        f"Expected params.pkl and config.json files."
    )


def prepare_mlm_batch(texts, tokenizer, max_length=512):
    """Prepare batch for MLM evaluation."""
    # Tokenize
    encoded = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="np"
    )

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    labels = input_ids.copy()

    # Create MLM masks (15% of tokens)
    probability_matrix = np.full(labels.shape, 0.15)
    special_tokens_mask = np.isin(input_ids, tokenizer.all_special_ids)
    probability_matrix[special_tokens_mask] = 0

    masked_indices = np.random.binomial(1, probability_matrix).astype(bool)

    # 80% mask, 10% random, 10% keep
    indices_replaced = (
        np.random.binomial(1, 0.8, size=labels.shape).astype(bool)
        & masked_indices
    )
    input_ids[indices_replaced] = tokenizer.mask_token_id

    indices_random = (
        np.random.binomial(1, 0.5, size=labels.shape).astype(bool)
        & masked_indices & ~indices_replaced
    )
    random_words = np.random.randint(len(tokenizer), size=labels.shape)
    input_ids[indices_random] = random_words[indices_random]

    # Only compute loss on masked tokens
    labels[~masked_indices] = -100

    return {
        "input_ids": jnp.array(input_ids),
        "attention_mask": jnp.array(attention_mask),
        "labels": jnp.array(labels)
    }


def eval_perplexity(params, config, args) -> Dict[str, Any]:
    """Evaluate perplexity on validation data."""
    print("Loading tokenizer and dataset...")
    tokenizer = AutoTokenizer.from_pretrained("jhu-clsp/mmBERT-small")

    dataset = load_dataset(
        args.dataset or "yhavinga/mc4_nl_cleaned",
        args.dataset_config or "micro",
        split=args.split or "validation",
        streaming=False
    )

    # Limit samples
    num_samples = min(args.num_samples, len(dataset))
    dataset = dataset.select(range(num_samples))

    print(f"Evaluating on {num_samples} samples...")
    model = ModernBertForMaskedLM(config)

    # Evaluation loop
    total_loss = 0.0
    total_accuracy = 0.0
    total_tokens = 0

    batch_size = args.batch_size
    num_batches = (num_samples + batch_size - 1) // batch_size

    for i in tqdm(range(num_batches), desc="Evaluating"):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, num_samples)
        batch_texts = dataset["text"][start_idx:end_idx]

        batch = prepare_mlm_batch(batch_texts, tokenizer, args.max_length)

        outputs = model.apply(
            {"params": params},
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            deterministic=True
        )

        # Calculate metrics
        predictions = jnp.argmax(outputs.logits, axis=-1)
        mask = batch["labels"] != -100

        if jnp.any(mask):
            correct = (predictions == batch["labels"]) & mask
            batch_accuracy = jnp.sum(correct) / jnp.sum(mask)

            total_loss += float(outputs.loss) * jnp.sum(mask)
            total_accuracy += float(batch_accuracy) * jnp.sum(mask)
            total_tokens += int(jnp.sum(mask))

    # Final metrics
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    avg_accuracy = total_accuracy / total_tokens if total_tokens > 0 else 0.0
    perplexity = np.exp(avg_loss)

    return {
        "loss": avg_loss,
        "accuracy": avg_accuracy,
        "perplexity": perplexity,
        "total_tokens": total_tokens,
        "num_samples": num_samples
    }


def eval_inference(params, config, args) -> Dict[str, Any]:
    """Evaluate inference on test sentences."""
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("jhu-clsp/mmBERT-small")
    model = ModernBertForMaskedLM(config)

    # Test sentences
    test_sentences = [
        "Amsterdam is de <mask> van Nederland.",
        "De kat zit op de <mask>.",
        "Het weer is vandaag erg <mask>.",
        "Ik drink graag een kopje <mask>.",
        "Nederland is bekend om zijn <mask>.",
        "De trein vertrekt om <mask> uur.",
    ]

    predictions_list = []

    for sentence in test_sentences:
        inputs = tokenizer(sentence, return_tensors="np")
        input_ids = jnp.array(inputs["input_ids"])
        attention_mask = jnp.array(inputs["attention_mask"])

        # Find mask position
        mask_token_id = tokenizer.mask_token_id
        mask_positions = jnp.where(input_ids == mask_token_id)

        if len(mask_positions[0]) == 0:
            continue

        outputs = model.apply(
            {"params": params},
            input_ids=input_ids,
            attention_mask=attention_mask,
            deterministic=True
        )

        mask_idx = mask_positions[1][0]
        logits = outputs.logits[0, mask_idx]

        # Get top 5 predictions
        top_k = 5
        top_indices = jnp.argsort(logits)[-top_k:][::-1]
        top_tokens = [tokenizer.decode([idx]).strip() for idx in top_indices]
        top_probs = jax.nn.softmax(logits)[top_indices]

        predictions_list.append({
            "input": sentence,
            "predictions": list(zip(
                top_tokens,
                [float(p) for p in top_probs]
            ))
        })

    # Calculate average confidence
    avg_confidence = np.mean([
        pred["predictions"][0][1]
        for pred in predictions_list
        if pred["predictions"]
    ])

    return {
        "predictions": predictions_list,
        "avg_top1_confidence": avg_confidence,
        "num_sentences": len(test_sentences)
    }


def save_to_db(db_path: str, results: Dict, args, duration: float):
    """Save evaluation results to database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Extract checkpoint info
    checkpoint_path = str(args.checkpoint)
    checkpoint_name = Path(checkpoint_path).name

    # Try to get training step from path
    training_step = None
    try:
        # For paths like experiments/red_1m/1000000
        parts = checkpoint_path.split('/')
        if parts[-1].isdigit():
            training_step = int(parts[-1])
        elif '_' in checkpoint_name:
            # For names like red_1m or yellow_2m
            step_part = checkpoint_name.split('_')[-1]
            if 'm' in step_part:
                training_step = int(step_part[:-1]) * 1_000_000
            elif 'k' in step_part:
                training_step = int(step_part[:-1]) * 1_000
    except:
        pass

    # Prepare data for insertion - ensure Python types
    def to_python(val):
        """Convert numpy/JAX values to Python types."""
        if val is None:
            return None
        if hasattr(val, 'item'):  # numpy/JAX scalar
            return val.item()
        return val

    cursor.execute("""
        INSERT INTO evaluations (
            checkpoint_path, checkpoint_name, training_step,
            wandb_run_name, eval_type, dataset, num_samples,
            loss, accuracy, perplexity,
            duration_seconds, config, predictions
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        checkpoint_path,
        checkpoint_name,
        training_step,
        args.wandb_name,
        args.eval_type,
        args.dataset or "mc4_nl_cleaned",
        to_python(results.get("num_samples")),
        to_python(results.get("loss")),
        to_python(results.get("accuracy")),
        to_python(results.get("perplexity")),
        duration,
        json.dumps(vars(args)),
        json.dumps(results.get("predictions")) if "predictions" in results else None
    ))

    conn.commit()
    conn.close()

    print(f"\n✓ Results saved to {db_path}")


def print_usage_examples():
    """Print usage examples when called without arguments."""
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                    Unified Checkpoint Evaluator                      ║
╚══════════════════════════════════════════════════════════════════════╝

Evaluate ModernBERT checkpoints and store results in SQLite database.

USAGE EXAMPLES:

1. Evaluate perplexity (default):
   python eval_unified.py --checkpoint experiments/converted/red_1m

2. Add training run annotation (for tracking):
   python eval_unified.py --checkpoint experiments/converted/yellow_2m \\
       --wandb-name vivid-firebrand-18

3. Quick test with fewer samples:
   python eval_unified.py --checkpoint experiments/converted/red_1m \\
       --num-samples 100 --batch-size 16

4. Inference evaluation (masked word predictions):
   python eval_unified.py --checkpoint experiments/converted/yellow_2m \\
       --eval-type inference

5. Custom dataset evaluation:
   python eval_unified.py --checkpoint /path/to/checkpoint \\
       --dataset yhavinga/mc4_nl_cleaned \\
       --dataset-config tiny \\
       --split test

6. Use custom database:
   python eval_unified.py --checkpoint experiments/converted/red_1m \\
       --db my_evaluations.db

CHECKPOINT FORMAT:
- Expects converted checkpoints (params.pkl + config.json)
- Use convert_checkpoint_on_tpu.py to convert Orbax checkpoints first

OUTPUTS:
- Console: Immediate results (perplexity, accuracy, loss)
- Database: All results stored for later analysis
- Use query_evaluations.py to analyze stored results

For detailed options: python eval_unified.py --help
""")


def main():
    # Show examples if called without arguments
    if len(sys.argv) == 1:
        print_usage_examples()
        sys.exit(0)

    parser = argparse.ArgumentParser(
        description="Unified checkpoint evaluation with SQLite storage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  # Basic perplexity evaluation
  %(prog)s --checkpoint experiments/converted/red_1m

  # With training run annotation
  %(prog)s --checkpoint experiments/converted/yellow_2m --wandb-name vivid-firebrand-18

  # Inference evaluation
  %(prog)s --checkpoint experiments/converted/yellow_2m --eval-type inference

  # Quick test
  %(prog)s --checkpoint experiments/converted/red_1m --num-samples 100
"""
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint directory"
    )
    parser.add_argument(
        "--eval-type",
        choices=["perplexity", "inference"],
        default="perplexity",
        help="Type of evaluation to run"
    )
    parser.add_argument(
        "--db",
        type=str,
        default="evaluations.db",
        help="SQLite database path"
    )
    parser.add_argument(
        "--wandb-name",
        type=str,
        help="Training run name for tracking (optional annotation)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset to use (default: yhavinga/mc4_nl_cleaned)"
    )
    parser.add_argument(
        "--dataset-config",
        type=str,
        help="Dataset configuration (default: micro)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        help="Dataset split (default: validation)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=500,
        help="Number of samples to evaluate (default: 500)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for evaluation (default: 8)"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length (default: 512)"
    )

    args = parser.parse_args()

    # Setup database
    setup_database(args.db)

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    params, config = load_checkpoint(args.checkpoint)

    # Run evaluation
    start_time = time.time()

    if args.eval_type == "perplexity":
        results = eval_perplexity(params, config, args)
        print(f"\n{'=' * 40}")
        print(f"Perplexity: {results['perplexity']:.2f}")
        print(f"Accuracy: {results['accuracy']*100:.2f}%")
        print(f"Loss: {results['loss']:.4f}")
    else:  # inference
        results = eval_inference(params, config, args)
        print(f"\n{'=' * 40}")
        print(f"Average confidence: {results['avg_top1_confidence']*100:.1f}%")
        print("\nSample predictions:")
        for pred in results["predictions"][:3]:
            print(f"  {pred['input']}")
            top = pred["predictions"][0]
            print(f"    → {top[0]} ({top[1]*100:.1f}%)")

    duration = time.time() - start_time
    print(f"\nEvaluation completed in {duration:.1f} seconds")

    # Save to database
    save_to_db(args.db, results, args, duration)


if __name__ == "__main__":
    main()