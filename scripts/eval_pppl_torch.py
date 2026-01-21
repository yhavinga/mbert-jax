#!/usr/bin/env python3
"""
PPPL (Pseudo-Perplexity) evaluation using PyTorch with MPS acceleration.

This script evaluates masked language models using the true PPPL metric,
which masks each token one-at-a-time and measures prediction quality.
Uses Apple Silicon MPS acceleration when available.

Usage:
    # First convert Flax checkpoint to HuggingFace format:
    python scripts/convert_flax_to_hf.py -i experiments/checkpoints/flax/22L-mc4-lowlr \\
        -o experiments/checkpoints/hf/22L-mc4-lowlr

    # Then run PPPL evaluation:
    python scripts/eval_pppl_torch.py --checkpoint experiments/checkpoints/hf/22L-mc4-lowlr \\
        --num-samples 200
"""

import argparse
import json
import sqlite3
import time
from pathlib import Path
from typing import Optional

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM


def get_device():
    """Get the best available device (MPS > CUDA > CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def eval_pppl(
    checkpoint_path: str,
    dataset_name: str = "yhavinga/mc4_nl_cleaned",
    dataset_config: str = "micro",
    split: str = "validation",
    num_samples: int = 200,
    max_length: int = 512,
    batch_size: int = 32,  # Can batch mask positions for same sequence
) -> dict:
    """Evaluate PPPL on validation data.
    
    PPPL masks each token one-at-a-time and computes:
    PPPL = exp(-mean(log P(token_i | context_without_i)))
    """
    device = get_device()
    print(f"Using device: {device}")
    
    print(f"Loading model from {checkpoint_path}...")
    model = AutoModelForMaskedLM.from_pretrained(checkpoint_path)
    model = model.to(device)
    model.eval()
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print(f"Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    
    # Add mask token if not present
    if tokenizer.mask_token is None:
        tokenizer.add_special_tokens({'mask_token': '<mask>'})
        print(f"  Added <mask> token (ID: {tokenizer.mask_token_id})")
    
    mask_token_id = tokenizer.mask_token_id
    special_ids = set(tokenizer.all_special_ids)
    
    print(f"Loading dataset {dataset_name}/{dataset_config}...")
    dataset = load_dataset(
        dataset_name,
        dataset_config,
        split=split,
        streaming=False
    )
    
    num_samples = min(num_samples, len(dataset))
    dataset = dataset.select(range(num_samples))
    
    print(f"Evaluating PPPL on {num_samples} samples...")
    
    total_log_prob = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for sample in tqdm(dataset, desc="Computing PPPL", total=num_samples):
            text = sample["text"]
            
            # Tokenize
            inputs = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
            
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            seq_len = int(attention_mask.sum().item())
            
            # Collect non-special positions
            positions = []
            true_tokens = []
            for i in range(seq_len):
                token_id = int(input_ids[0, i].item())
                if token_id not in special_ids:
                    positions.append(i)
                    true_tokens.append(token_id)
            
            if not positions:
                continue
            
            # Batch process: create copies of input with one mask each
            # Process in batches to avoid OOM
            for batch_start in range(0, len(positions), batch_size):
                batch_end = min(batch_start + batch_size, len(positions))
                batch_positions = positions[batch_start:batch_end]
                batch_true_tokens = true_tokens[batch_start:batch_end]
                batch_len = len(batch_positions)
                
                # Create batched inputs
                batch_input_ids = input_ids.repeat(batch_len, 1)
                batch_attention_mask = attention_mask.repeat(batch_len, 1)
                
                # Mask one position in each batch item
                for j, pos in enumerate(batch_positions):
                    batch_input_ids[j, pos] = mask_token_id
                
                # Forward pass
                outputs = model(
                    input_ids=batch_input_ids,
                    attention_mask=batch_attention_mask
                )
                
                # Get log probabilities for true tokens
                for j, (pos, true_token) in enumerate(zip(batch_positions, batch_true_tokens)):
                    logits = outputs.logits[j, pos]
                    log_probs = torch.log_softmax(logits.float(), dim=-1)
                    total_log_prob += log_probs[true_token].item()
                    total_tokens += 1
    
    # PPPL = exp(-mean_log_prob)
    if total_tokens > 0:
        mean_log_prob = total_log_prob / total_tokens
        pppl = float(torch.exp(torch.tensor(-mean_log_prob)).item())
    else:
        pppl = float('inf')
        mean_log_prob = None
    
    return {
        "pppl": pppl,
        "mean_log_prob": mean_log_prob,
        "total_tokens": total_tokens,
        "num_samples": num_samples,
        "device": str(device),
    }


def save_to_db(db_path: str, results: dict, args):
    """Save evaluation results to SQLite database."""
    # Simple append to evaluations table
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create table if not exists (simplified schema)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS pppl_evaluations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            checkpoint_path TEXT NOT NULL,
            checkpoint_name TEXT,
            pppl REAL,
            mean_log_prob REAL,
            total_tokens INTEGER,
            num_samples INTEGER,
            device TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    checkpoint_name = Path(args.checkpoint).name
    
    cursor.execute("""
        INSERT INTO pppl_evaluations (
            checkpoint_path, checkpoint_name, pppl, mean_log_prob,
            total_tokens, num_samples, device
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        str(args.checkpoint),
        checkpoint_name,
        results["pppl"],
        results.get("mean_log_prob"),
        results["total_tokens"],
        results["num_samples"],
        results["device"],
    ))
    
    conn.commit()
    conn.close()
    print(f"✓ Results saved to {db_path}")


def main():
    parser = argparse.ArgumentParser(
        description="PPPL evaluation with PyTorch MPS acceleration"
    )
    parser.add_argument(
        "--checkpoint", "-c",
        type=str,
        required=True,
        help="Path to HuggingFace checkpoint directory"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="yhavinga/mc4_nl_cleaned",
        help="Dataset to use"
    )
    parser.add_argument(
        "--dataset-config",
        type=str,
        default="micro",
        help="Dataset configuration"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        help="Dataset split"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=200,
        help="Number of samples to evaluate"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for parallel mask evaluation"
    )
    parser.add_argument(
        "--db",
        type=str,
        default="evaluations.db",
        help="SQLite database path"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("PPPL EVALUATION (PyTorch MPS)")
    print("=" * 60)
    
    start_time = time.time()
    
    results = eval_pppl(
        checkpoint_path=args.checkpoint,
        dataset_name=args.dataset,
        dataset_config=args.dataset_config,
        split=args.split,
        num_samples=args.num_samples,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )
    
    duration = time.time() - start_time
    
    print(f"\n{'=' * 40}")
    print(f"Pseudo-Perplexity (PPPL): {results['pppl']:.2f}")
    print(f"Mean Log Prob: {results.get('mean_log_prob', 'N/A'):.4f}")
    print(f"Total Tokens: {results['total_tokens']:,}")
    print(f"Device: {results['device']}")
    print(f"Duration: {duration:.1f}s ({duration/results['num_samples']:.2f}s/sample)")
    
    # Save to database
    save_to_db(args.db, results, args)


if __name__ == "__main__":
    main()
