#!/usr/bin/env python3
"""Fine-tuning script for classification tasks.

Usage:
    python scripts/dumb/finetune.py \
        --checkpoint experiments/checkpoints/flax/22L-mc4-lowlr \
        --task dbrd \
        --output experiments/dumb/22L-mc4-lowlr/dbrd
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import argparse
import json
import pickle
from functools import partial
from typing import Any, Dict

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state
from tqdm import tqdm
from transformers import AutoTokenizer

from mmbjax.configuration_mmbbert import MMBertConfig
from mmbjax.heads.classification import ModernBertForSequenceClassification

from data import TASKS, get_task


class TrainState(train_state.TrainState):
    """Train state with dropout RNG."""
    dropout_rng: jax.Array


def load_pretrained_params(checkpoint_path: Path) -> tuple[dict, MMBertConfig]:
    """Load pretrained parameters and config."""
    pkl_path = checkpoint_path / "params.pkl"
    config_path = checkpoint_path / "config.json"

    if not pkl_path.exists() or not config_path.exists():
        raise ValueError(f"Expected params.pkl and config.json in {checkpoint_path}")

    with open(pkl_path, "rb") as f:
        params = pickle.load(f)

    with open(config_path, "r") as f:
        config_dict = json.load(f)

    # Filter to only valid MMBertConfig parameters
    import inspect
    valid_params = set(inspect.signature(MMBertConfig.__init__).parameters.keys())
    valid_params.discard("self")
    filtered_config = {k: v for k, v in config_dict.items() if k in valid_params}

    # Store tokenizer_name separately before filtering
    tokenizer_name = config_dict.get("tokenizer_name", "yhavinga/dutch-llama-tokenizer")

    config = MMBertConfig(**filtered_config)
    # Attach tokenizer_name as attribute for later use
    config.tokenizer_name = tokenizer_name

    return params, config


def create_model_and_state(
    config: MMBertConfig,
    pretrained_params: dict,
    num_labels: int,
    learning_rate: float,
    rng: jax.Array,
) -> tuple[ModernBertForSequenceClassification, TrainState]:
    """Create model and training state with pretrained weights."""
    # Update config for classification
    config.num_labels = num_labels

    model = ModernBertForSequenceClassification(config=config)

    # Initialize with dummy input to get parameter structure
    dummy_input = jnp.ones((1, 128), dtype=jnp.int32)
    init_rng, dropout_rng = jax.random.split(rng)
    params = model.init(init_rng, dummy_input, deterministic=True)

    # Copy pretrained weights into model params
    # Pretrained has: model, prediction_head, decoder
    # Classification needs: model, head, classifier
    new_params = params["params"].copy()

    # Copy encoder (embeddings + transformer layers)
    if "model" in pretrained_params:
        new_params["model"] = pretrained_params["model"]

    # Copy prediction head -> head (same structure: dense + layer_norm)
    if "prediction_head" in pretrained_params:
        new_params["head"] = pretrained_params["prediction_head"]

    # classifier is randomly initialized (new task-specific layer)

    # Create optimizer
    optimizer = optax.adamw(learning_rate=learning_rate, weight_decay=0.01)

    state = TrainState.create(
        apply_fn=model.apply,
        params=new_params,
        tx=optimizer,
        dropout_rng=dropout_rng,
    )

    return model, state


def collate_fn(batch: list[dict], pad_token_id: int, max_length: int) -> dict:
    """Collate batch with padding."""
    input_ids = [ex["input_ids"][:max_length] for ex in batch]
    labels = [ex["label"] for ex in batch]

    # Pad to max length in batch
    max_len = max(len(ids) for ids in input_ids)
    padded_ids = []
    attention_masks = []

    for ids in input_ids:
        pad_len = max_len - len(ids)
        padded_ids.append(ids + [pad_token_id] * pad_len)
        attention_masks.append([1] * len(ids) + [0] * pad_len)

    return {
        "input_ids": np.array(padded_ids, dtype=np.int32),
        "attention_mask": np.array(attention_masks, dtype=np.float32),
        "labels": np.array(labels, dtype=np.int32),
    }


@jax.jit
def train_step(
    state: TrainState,
    batch: dict,
) -> tuple[TrainState, dict]:
    """Single training step."""
    dropout_rng, new_dropout_rng = jax.random.split(state.dropout_rng)

    def loss_fn(params):
        outputs = state.apply_fn(
            {"params": params},
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            deterministic=False,
            rngs={"dropout": dropout_rng},
        )
        return outputs.loss, outputs.logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)

    # Compute accuracy
    predictions = jnp.argmax(logits, axis=-1)
    accuracy = jnp.mean(predictions == batch["labels"])

    state = state.apply_gradients(grads=grads)
    state = state.replace(dropout_rng=new_dropout_rng)

    metrics = {"loss": loss, "accuracy": accuracy}
    return state, metrics


@jax.jit
def eval_step(
    state: TrainState,
    batch: dict,
) -> dict:
    """Single evaluation step."""
    outputs = state.apply_fn(
        {"params": state.params},
        batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
        deterministic=True,
    )

    predictions = jnp.argmax(outputs.logits, axis=-1)
    correct = jnp.sum(predictions == batch["labels"])
    total = batch["labels"].shape[0]

    return {"loss": outputs.loss, "correct": correct, "total": total}


def evaluate(state: TrainState, dataset, collate, batch_size: int) -> dict:
    """Evaluate model on dataset."""
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    # Create batches
    indices = list(range(len(dataset)))
    for i in range(0, len(indices), batch_size):
        batch_indices = indices[i : i + batch_size]
        batch = [dataset[j] for j in batch_indices]
        batch = collate(batch)
        batch = {k: jnp.array(v) for k, v in batch.items()}

        metrics = eval_step(state, batch)
        total_loss += float(metrics["loss"]) * len(batch_indices)
        total_correct += int(metrics["correct"])
        total_samples += int(metrics["total"])

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
    }


def main():
    parser = argparse.ArgumentParser(description="Fine-tune on classification tasks")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to pretrained checkpoint")
    parser.add_argument("--task", type=str, required=True, choices=list(TASKS.keys()), help="Task to fine-tune on")
    parser.add_argument("--output", type=str, required=True, help="Output directory for fine-tuned model")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--max-length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    print(f"Fine-tuning on task: {args.task}")
    print(f"Checkpoint: {args.checkpoint}")

    # Load pretrained model
    checkpoint_path = Path(args.checkpoint)
    pretrained_params, config = load_pretrained_params(checkpoint_path)

    # Get tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name or "yhavinga/dutch-llama-tokenizer")

    # Load dataset
    task_config = get_task(args.task)
    print(f"Loading dataset for {args.task}...")

    dataset = task_config["loader"](tokenizer, args.max_length)
    print(f"Train: {len(dataset['train'])}, Val: {len(dataset['validation'])}, Test: {len(dataset['test'])}")

    # Create model and state
    rng = jax.random.PRNGKey(args.seed)
    model, state = create_model_and_state(
        config=config,
        pretrained_params=pretrained_params,
        num_labels=task_config["num_labels"],
        learning_rate=args.learning_rate,
        rng=rng,
    )

    # Collate function
    collate = partial(collate_fn, pad_token_id=tokenizer.pad_token_id, max_length=args.max_length)

    # Training loop
    best_val_accuracy = 0.0
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        # Shuffle training data
        train_indices = list(range(len(dataset["train"])))
        np.random.shuffle(train_indices)

        # Training
        epoch_loss = 0.0
        epoch_correct = 0
        num_batches = 0

        pbar = tqdm(range(0, len(train_indices), args.batch_size), desc="Training")
        for i in pbar:
            batch_indices = train_indices[i : i + args.batch_size]
            batch = [dataset["train"][j] for j in batch_indices]
            batch = collate(batch)
            batch = {k: jnp.array(v) for k, v in batch.items()}

            state, metrics = train_step(state, batch)

            epoch_loss += float(metrics["loss"])
            epoch_correct += float(metrics["accuracy"]) * len(batch_indices)
            num_batches += 1

            pbar.set_postfix({"loss": f"{metrics['loss']:.4f}", "acc": f"{metrics['accuracy']:.4f}"})

        train_accuracy = epoch_correct / len(train_indices)
        print(f"Train Loss: {epoch_loss / num_batches:.4f}, Train Accuracy: {train_accuracy:.4f}")

        # Validation
        val_metrics = evaluate(state, dataset["validation"], collate, args.batch_size)
        print(f"Val Loss: {val_metrics['loss']:.4f}, Val Accuracy: {val_metrics['accuracy']:.4f}")

        # Save best model
        if val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            print(f"New best! Saving checkpoint...")

            with open(output_path / "params.pkl", "wb") as f:
                pickle.dump(state.params, f)

            # Save config with num_labels
            config_dict = config.to_dict()
            config_dict["num_labels"] = task_config["num_labels"]
            with open(output_path / "config.json", "w") as f:
                json.dump(config_dict, f, indent=2)

    # Final evaluation on test set
    print("\nEvaluating on test set...")
    test_metrics = evaluate(state, dataset["test"], collate, args.batch_size)
    print(f"Test Loss: {test_metrics['loss']:.4f}, Test Accuracy: {test_metrics['accuracy']:.4f}")

    # Save results
    results = {
        "task": args.task,
        "checkpoint": args.checkpoint,
        "best_val_accuracy": best_val_accuracy,
        "test_accuracy": test_metrics["accuracy"],
        "test_loss": test_metrics["loss"],
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
    }
    with open(output_path / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_path}")
    print(f"Best validation accuracy: {best_val_accuracy:.4f}")
    print(f"Test accuracy: {test_metrics['accuracy']:.4f}")


if __name__ == "__main__":
    main()
