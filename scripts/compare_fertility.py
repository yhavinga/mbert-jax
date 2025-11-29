#!/usr/bin/env python3
"""Compare tokenizer fertility (bytes-per-token) on Dutch text."""

import argparse
import math
from datasets import load_dataset
from transformers import AutoTokenizer


def calculate_fertility(tokenizer, texts):
    """Calculate fertility (bytes per token) for a tokenizer on given texts."""
    total_bytes = 0
    total_tokens = 0

    for text in texts:
        total_bytes += len(text.encode('utf-8'))
        tokens = tokenizer.encode(text, add_special_tokens=False)
        total_tokens += len(tokens)

    fertility = total_bytes / total_tokens if total_tokens > 0 else 0
    return {
        'total_bytes': total_bytes,
        'total_tokens': total_tokens,
        'fertility': fertility,
        'vocab_size': len(tokenizer),
    }


def compute_bpb(loss_per_token, fertility):
    """Convert per-token loss (nats) to bits-per-byte."""
    return loss_per_token / fertility * math.log2(math.e)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare tokenizer fertility on Dutch text"
    )
    parser.add_argument(
        "--tokenizers",
        nargs="+",
        default=["yhavinga/dutch-llama-tokenizer", "jhu-clsp/mmBERT-base"],
        help="HuggingFace tokenizer names to compare",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="yhavinga/mc4_nl_cleaned",
        help="Dataset to use (default: yhavinga/mc4_nl_cleaned)",
    )
    parser.add_argument(
        "--dataset-config",
        type=str,
        default="micro",
        help="Dataset config (default: micro)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10000,
        help="Number of samples to analyze (default: 10000)",
    )
    parser.add_argument(
        "--loss",
        type=str,
        default=None,
        help="Comma-separated loss values (nats) for BPB calculation",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load dataset
    print(f"Loading {args.dataset} ({args.dataset_config})...")
    dataset = load_dataset(
        args.dataset,
        args.dataset_config,
        split="train",
        streaming=True,
    )

    # Collect texts
    texts = []
    for i, sample in enumerate(dataset):
        if i >= args.num_samples:
            break
        texts.append(sample["text"])
    print(f"Loaded {len(texts)} samples\n")

    # Calculate fertility for each tokenizer
    results = []
    for tok_name in args.tokenizers:
        print(f"Processing {tok_name}...")
        tokenizer = AutoTokenizer.from_pretrained(tok_name)
        stats = calculate_fertility(tokenizer, texts)
        stats['name'] = tok_name
        results.append(stats)

    # Print results
    print(f"\nTokenizer Fertility Analysis on {args.dataset} ({len(texts)} samples)")
    print("=" * 80)
    print(f"{'Tokenizer':<40} {'Vocab':>8} {'Tokens':>12} {'Bytes':>12} {'Fertility':>10}")
    print("-" * 80)

    for r in results:
        print(f"{r['name']:<40} {r['vocab_size']:>8} {r['total_tokens']:>12} {r['total_bytes']:>12} {r['fertility']:>10.2f}")

    # BPB calculation if loss values provided
    if args.loss:
        losses = [float(x) for x in args.loss.split(",")]
        if len(losses) != len(results):
            print(f"\nError: {len(losses)} loss values provided but {len(results)} tokenizers")
            return

        print(f"\nBPB Comparison (with provided losses):")
        print("-" * 80)
        print(f"{'Tokenizer':<40} {'Loss':>8} {'Fertility':>10} {'BPB':>10}")
        print("-" * 80)

        for r, loss in zip(results, losses):
            bpb = compute_bpb(loss, r['fertility'])
            print(f"{r['name']:<40} {loss:>8.3f} {r['fertility']:>10.2f} {bpb:>10.3f}")


if __name__ == "__main__":
    main()
