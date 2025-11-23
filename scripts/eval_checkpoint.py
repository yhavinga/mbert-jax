#!/usr/bin/env python3
"""
Evaluate Orbax checkpoints from training.
Loads checkpoints and runs MLM evaluation on validation data.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse
import time
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import orbax.checkpoint as ocp

from mmbjax.configuration_mmbbert import MMBertConfig
from mmbjax.heads.masked_lm import ModernBertForMaskedLM


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate ModernBERT checkpoints"
    )

    # Checkpoint arguments
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        required=True,
        help="Path to checkpoint directory (e.g., checkpoints/1000000)"
    )

    # Dataset arguments
    parser.add_argument(
        "--dataset",
        type=str,
        default="yhavinga/mc4_nl_cleaned",
        help="HuggingFace dataset name (default: yhavinga/mc4_nl_cleaned)"
    )
    parser.add_argument(
        "--dataset-config",
        type=str,
        default="micro",
        help="Dataset configuration/subset (default: micro)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        help="Dataset split to evaluate (default: validation)"
    )

    # Evaluation arguments
    parser.add_argument(
        "--max-samples",
        type=int,
        default=1000,
        help="Maximum samples to evaluate (default: 1000)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for evaluation (default: 32)"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length (default: 512)"
    )
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Quick test with only 100 samples"
    )

    # Model configuration (should match training)
    parser.add_argument(
        "--num-layers",
        type=int,
        default=6,
        help="Number of model layers (default: 6)"
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=768,
        help="Hidden size (default: 768)"
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=12,
        help="Number of attention heads (default: 12)"
    )

    return parser.parse_args()


class FlaxDataCollatorForLanguageModeling:
    """
    Data collator for MLM - handles tokenization + masking.
    Same as in train_minimal.py for consistency.
    """
    def __init__(self, tokenizer, mlm_probability: float = 0.15):
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability

        if tokenizer.mask_token is None:
            raise ValueError("Tokenizer must have a mask token for MLM")

    def __call__(self, examples, max_length: int = 512):
        """Tokenize and mask a batch of examples."""
        # Extract texts
        texts = [ex["text"] if isinstance(ex, dict) else ex for ex in examples]

        # Tokenize
        batch = self.tokenizer(
            texts,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="np",
        )

        # Apply MLM masking
        input_ids, labels = self.mask_tokens(
            batch["input_ids"],
            batch.get("special_tokens_mask")
        )

        return {
            "input_ids": jnp.array(input_ids),
            "attention_mask": jnp.array(batch["attention_mask"]),
            "labels": jnp.array(labels),
        }

    def mask_tokens(self, inputs, special_tokens_mask=None):
        """Prepare masked tokens: 80% MASK, 10% random, 10% original."""
        labels = inputs.copy()

        # Probability matrix for masking
        probability_matrix = np.full(labels.shape, self.mlm_probability)

        # Don't mask special tokens
        if special_tokens_mask is None:
            special_tokens_mask = np.array([
                self.tokenizer.get_special_tokens_mask(ids, already_has_special_tokens=True)
                for ids in inputs.tolist()
            ])

        probability_matrix[special_tokens_mask.astype(bool)] = 0.0

        # Sample positions to mask
        masked_indices = np.random.binomial(1, probability_matrix).astype(bool)
        labels[~masked_indices] = -100  # Only compute loss on masked tokens

        # 80% of time: replace with [MASK]
        indices_replaced = (
            np.random.binomial(1, np.full(labels.shape, 0.8)).astype(bool)
            & masked_indices
        )
        inputs[indices_replaced] = self.tokenizer.mask_token_id

        # 10% of time: replace with random token
        indices_random = (
            np.random.binomial(1, np.full(labels.shape, 0.5)).astype(bool)
            & masked_indices
            & ~indices_replaced
        )
        random_words = np.random.randint(
            self.tokenizer.vocab_size,
            size=labels.shape,
            dtype=np.int32
        )
        inputs[indices_random] = random_words[indices_random]

        # 10% of time: keep original token (no change needed)

        return inputs, labels


def compute_mlm_metrics(logits, labels):
    """Compute MLM loss, accuracy, and perplexity."""
    # Flatten
    logits_flat = logits.reshape(-1, logits.shape[-1])
    labels_flat = labels.reshape(-1)

    # Mask: only compute loss where labels != -100
    mask = labels_flat != -100

    # Cross-entropy
    log_probs = jax.nn.log_softmax(logits_flat, axis=-1)
    target_log_probs = jnp.take_along_axis(
        log_probs, labels_flat[:, None], axis=-1
    ).squeeze(-1)

    # Masked mean loss
    loss = -jnp.sum(target_log_probs * mask) / jnp.maximum(jnp.sum(mask), 1)

    # Compute accuracy on masked tokens
    predictions = jnp.argmax(logits_flat, axis=-1)
    correct = (predictions == labels_flat) * mask
    accuracy = jnp.sum(correct) / jnp.maximum(jnp.sum(mask), 1)

    # Compute perplexity
    perplexity = jnp.exp(loss)

    # Count masked tokens
    n_masked = jnp.sum(mask)

    return {
        "loss": loss,
        "accuracy": accuracy,
        "perplexity": perplexity,
        "n_masked_tokens": n_masked,
        "n_correct": jnp.sum(correct),
        "n_total_tokens": mask.size,
    }


def load_checkpoint(checkpoint_dir, model, learning_rate=5e-5):
    """Load checkpoint using Orbax CheckpointManager."""
    checkpoint_path = Path(checkpoint_dir).absolute()

    if not checkpoint_path.exists():
        raise ValueError(f"Checkpoint directory not found: {checkpoint_path}")

    print(f"\nLoading checkpoint from: {checkpoint_path}")

    # Create checkpoint manager
    options = ocp.CheckpointManagerOptions(
        max_to_keep=1,
        save_interval_steps=1,  # Not used for loading
    )

    checkpoint_manager = ocp.CheckpointManager(
        checkpoint_path.parent,  # Parent directory containing numbered checkpoints
        options=options,
    )

    # Get step from directory name
    step = int(checkpoint_path.name)

    # Create a simple optimizer (needed for TrainState, but not used for eval)
    optimizer = optax.adam(learning_rate)

    # Create initial state
    dummy_input = jnp.ones((1, 512), dtype=jnp.int32)
    dummy_mask = jnp.ones((1, 512), dtype=jnp.int32)

    variables = model.init(
        jax.random.PRNGKey(0),
        input_ids=dummy_input,
        attention_mask=dummy_mask,
        deterministic=True,
    )

    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=optimizer,
    )

    # Restore checkpoint
    state = checkpoint_manager.restore(
        step,
        args=ocp.args.StandardRestore(state)
    )

    print(f"✓ Loaded checkpoint from step {step}")

    # Get parameter count
    param_count = sum(x.size for x in jax.tree.leaves(state.params))
    print(f"✓ Model parameters: {param_count:,} ({param_count/1e6:.1f}M)")

    return state


@jax.jit
def eval_step(state, batch):
    """Single evaluation step."""
    output = state.apply_fn(
        {"params": state.params},
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        deterministic=True,
    )

    # Extract logits
    logits = output.logits if hasattr(output, "logits") else output["logits"]

    # Compute metrics
    metrics = compute_mlm_metrics(logits, batch["labels"])

    return metrics


def evaluate(args):
    """Main evaluation function."""
    print("=" * 70)
    print("MODERNBERT CHECKPOINT EVALUATION")
    print("=" * 70)

    # Device info
    backend = jax.default_backend()
    devices = jax.devices()
    n_devices = len(devices)

    print(f"\nDevice Configuration:")
    print(f"  Backend: {backend}")
    print(f"  Device count: {n_devices}")

    # Model configuration (should match training)
    config = MMBertConfig(
        vocab_size=256000,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        intermediate_size=args.hidden_size * 4,
        max_position_embeddings=8192,
    )

    print(f"\nModel Configuration:")
    print(f"  Layers: {config.num_hidden_layers}")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Attention heads: {config.num_attention_heads}")
    print(f"  Max position embeddings: {config.max_position_embeddings}")

    # Initialize model
    print("\nInitializing model...")
    model = ModernBertForMaskedLM(config)

    # Load checkpoint
    state = load_checkpoint(args.checkpoint_dir, model)

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("jhu-clsp/mmBERT-small")
    print(f"  Vocab size: {tokenizer.vocab_size}")

    # Create data collator
    collator = FlaxDataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=0.15,
    )

    # Load dataset
    print(f"\nLoading dataset: {args.dataset} ({args.dataset_config})...")
    print(f"  Split: {args.split}")

    max_samples = args.max_samples
    if args.quick_test:
        max_samples = 100
        print("  Quick test mode: evaluating only 100 samples")

    # Load dataset (not streaming for evaluation)
    try:
        dataset = load_dataset(
            args.dataset,
            args.dataset_config,
            split=args.split,
            streaming=False,
        )
        # Take subset
        if max_samples < len(dataset):
            dataset = dataset.select(range(max_samples))
    except:
        # If validation split doesn't exist, use train split
        print(f"  Warning: '{args.split}' split not found, using 'train' split")
        dataset = load_dataset(
            args.dataset,
            args.dataset_config,
            split="train",
            streaming=False,
        )
        # Take subset from end (more likely to be validation-like)
        if max_samples < len(dataset):
            start_idx = max(0, len(dataset) - max_samples)
            dataset = dataset.select(range(start_idx, len(dataset)))

    print(f"  Evaluating {len(dataset)} samples")

    # Evaluation loop
    print("\n" + "=" * 70)
    print("EVALUATION START")
    print("=" * 70)

    total_loss = 0.0
    total_accuracy = 0.0
    total_perplexity = 0.0
    total_masked_tokens = 0
    total_correct_predictions = 0
    total_samples = 0
    batch_buffer = []

    start_time = time.time()

    with tqdm(total=len(dataset), desc="Evaluating") as pbar:
        for example in dataset:
            batch_buffer.append(example)

            if len(batch_buffer) >= args.batch_size:
                # Create batch
                batch = collator(batch_buffer, max_length=args.max_length)
                batch_buffer = []

                # Evaluate
                metrics = eval_step(state, batch)

                # Accumulate metrics
                batch_size = batch["input_ids"].shape[0]
                total_loss += float(metrics["loss"]) * batch_size
                total_accuracy += float(metrics["accuracy"]) * float(metrics["n_masked_tokens"])
                total_perplexity += float(metrics["perplexity"]) * batch_size
                total_masked_tokens += int(metrics["n_masked_tokens"])
                total_correct_predictions += int(metrics["n_correct"])
                total_samples += batch_size

                # Update progress bar
                pbar.update(batch_size)
                pbar.set_postfix({
                    "loss": f"{float(metrics['loss']):.4f}",
                    "acc": f"{float(metrics['accuracy'])*100:.1f}%",
                    "ppl": f"{float(metrics['perplexity']):.1f}",
                })

        # Process remaining samples
        if batch_buffer:
            batch = collator(batch_buffer, max_length=args.max_length)
            metrics = eval_step(state, batch)

            batch_size = batch["input_ids"].shape[0]
            total_loss += float(metrics["loss"]) * batch_size
            total_accuracy += float(metrics["accuracy"]) * float(metrics["n_masked_tokens"])
            total_perplexity += float(metrics["perplexity"]) * batch_size
            total_masked_tokens += int(metrics["n_masked_tokens"])
            total_correct_predictions += int(metrics["n_correct"])
            total_samples += batch_size

            pbar.update(batch_size)

    # Calculate final metrics
    avg_loss = total_loss / total_samples
    avg_accuracy = total_correct_predictions / max(total_masked_tokens, 1)
    avg_perplexity = total_perplexity / total_samples

    elapsed_time = time.time() - start_time
    samples_per_sec = total_samples / elapsed_time
    tokens_per_sec = total_samples * args.max_length / elapsed_time

    # Print results
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)

    print(f"\nDataset:")
    print(f"  Name: {args.dataset} ({args.dataset_config})")
    print(f"  Split: {args.split}")
    print(f"  Samples evaluated: {total_samples}")
    print(f"  Total masked tokens: {total_masked_tokens:,}")

    print(f"\nMetrics:")
    print(f"  Loss: {avg_loss:.4f}")
    print(f"  Accuracy: {avg_accuracy*100:.2f}%")
    print(f"  Perplexity: {avg_perplexity:.2f}")

    print(f"\nPerformance:")
    print(f"  Evaluation time: {elapsed_time:.1f} seconds")
    print(f"  Samples/second: {samples_per_sec:.1f}")
    print(f"  Tokens/second: {tokens_per_sec:.0f}")

    print("\n✓ Evaluation completed successfully!")

    return {
        "loss": avg_loss,
        "accuracy": avg_accuracy,
        "perplexity": avg_perplexity,
        "n_samples": total_samples,
        "n_masked_tokens": total_masked_tokens,
        "elapsed_time": elapsed_time,
    }


def main():
    args = parse_args()
    results = evaluate(args)


if __name__ == "__main__":
    main()