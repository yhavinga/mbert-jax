#!/usr/bin/env python3
"""
Production training script - with HF best practices and TPU optimizations.
Trains ModernBERT on Dutch data (mc4_nl_cleaned micro config).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse
import os
import json
import time
from datetime import datetime
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import traverse_util
from flax.training import train_state
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple, Any
import orbax.checkpoint as ocp
import wandb

from mmbjax.configuration_mmbbert import MMBertConfig
from mmbjax.heads.masked_lm import ModernBertForMaskedLM


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train ModernBERT model with JAX/Flax"
    )

    # Core training arguments
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
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length (default: 512)"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=500,
        help="Maximum training steps (default: 500)"
    )

    # Additional training arguments
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (default: auto-detect based on backend)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
        help="Learning rate (default: 5e-5)"
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=50,
        help="Number of warmup steps (default: 50)"
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay (default: 0.01)"
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=100,
        help="Checkpoint frequency in steps (default: 100)"
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=10,
        help="Logging frequency in steps (default: 10)"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints (default: checkpoints)"
    )

    # W&B arguments
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="modernbert-mlm",
        help="W&B project name (default: modernbert-mlm)"
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="W&B run name (default: auto-generated)"
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable W&B logging"
    )

    return parser.parse_args()


class FlaxDataCollatorForLanguageModeling:
    """
    Data collator for MLM - handles tokenization + masking.
    Based on HF transformers pattern.
    """
    def __init__(self, tokenizer, mlm_probability: float = 0.15):
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        
        if tokenizer.mask_token is None:
            raise ValueError("Tokenizer must have a mask token for MLM")
    
    def __call__(self, examples: List[Dict], max_length: int = 512) -> Dict[str, jnp.ndarray]:
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
    
    def mask_tokens(
        self, 
        inputs: np.ndarray, 
        special_tokens_mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare masked tokens: 80% MASK, 10% random, 10% original.
        """
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


def compute_mlm_loss_and_accuracy(logits, labels):
    """Compute cross-entropy loss and accuracy on masked positions only."""
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

    return loss, accuracy, perplexity


def compute_gradient_norm(gradients):
    """Compute global L2 norm of gradients."""
    leaves = jax.tree_util.tree_leaves(gradients)
    norm = jnp.sqrt(sum([jnp.sum(jnp.square(g)) for g in leaves]))
    return norm


def compute_parameter_norm(params):
    """Compute global L2 norm of parameters."""
    leaves = jax.tree_util.tree_leaves(params)
    norm = jnp.sqrt(sum([jnp.sum(jnp.square(p)) for p in leaves]))
    return norm


def compute_update_ratio(gradients, params, learning_rate):
    """Compute update/parameter ratio (Karpathy's favorite metric).

    This should be around 0.001 for healthy training.
    """
    grad_norm = compute_gradient_norm(gradients)
    param_norm = compute_parameter_norm(params)
    # The actual update is grad * lr (for SGD-like optimizers)
    update_norm = grad_norm * learning_rate
    ratio = update_norm / jnp.maximum(param_norm, 1e-7)
    return ratio


def compute_per_layer_gradient_norms(gradients):
    """Compute gradient norms for each layer type."""
    flat_grads = traverse_util.flatten_dict(gradients)

    layer_norms = {}

    # Embedding layer
    embedding_grads = {k: v for k, v in flat_grads.items() if "embeddings" in str(k)}
    if embedding_grads:
        layer_norms["embeddings"] = jnp.sqrt(
            sum([jnp.sum(jnp.square(g)) for g in embedding_grads.values()])
        )

    # Encoder layers (attention and FFN)
    for i in range(12):  # Assuming max 12 layers
        layer_key = f"layer_{i}"
        layer_grads = {k: v for k, v in flat_grads.items() if layer_key in str(k)}
        if layer_grads:
            # Attention sublayer
            attn_grads = {k: v for k, v in layer_grads.items() if "attention" in str(k)}
            if attn_grads:
                layer_norms[f"encoder_{i}_attention"] = jnp.sqrt(
                    sum([jnp.sum(jnp.square(g)) for g in attn_grads.values()])
                )

            # FFN sublayer
            ffn_grads = {k: v for k, v in layer_grads.items() if "ffn" in str(k) or "intermediate" in str(k) or "output" in str(k)}
            if ffn_grads:
                layer_norms[f"encoder_{i}_ffn"] = jnp.sqrt(
                    sum([jnp.sum(jnp.square(g)) for g in ffn_grads.values()])
                )

    # LM head
    lm_head_grads = {k: v for k, v in flat_grads.items() if "lm_head" in str(k) or "predictions" in str(k)}
    if lm_head_grads:
        layer_norms["lm_head"] = jnp.sqrt(
            sum([jnp.sum(jnp.square(g)) for g in lm_head_grads.values()])
        )

    return layer_norms


def create_train_state(
    model, 
    learning_rate: float,
    num_train_steps: int,
    warmup_steps: int,
    weight_decay: float = 0.01,
):
    """Create training state with proper optimizer."""
    
    # Learning rate schedule: linear warmup + linear decay
    warmup_fn = optax.linear_schedule(
        init_value=0.0,
        end_value=learning_rate,
        transition_steps=warmup_steps,
    )
    decay_fn = optax.linear_schedule(
        init_value=learning_rate,
        end_value=0.0,
        transition_steps=num_train_steps - warmup_steps,
    )
    lr_schedule = optax.join_schedules(
        schedules=[warmup_fn, decay_fn],
        boundaries=[warmup_steps],
    )
    
    # Weight decay mask: exclude bias and LayerNorm from decay
    def decay_mask_fn(params):
        flat_params = traverse_util.flatten_dict(params)
        
        # Find LayerNorm parameters
        layer_norm_candidates = ["layer_norm", "layernorm", "ln"]
        layer_norm_params = set()
        for layer_norm_name in layer_norm_candidates:
            for layer in flat_params.keys():
                if layer_norm_name in "".join(layer).lower():
                    layer_norm_params.add(layer[-2:])
        
        # Mask: decay everything EXCEPT bias and LayerNorm params
        flat_mask = {
            path: (path[-1] != "bias" and path[-2:] not in layer_norm_params)
            for path in flat_params
        }
        return traverse_util.unflatten_dict(flat_mask)
    
    # Optimizer: gradient clipping + AdamW with masked weight decay
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),  # Gradient clipping
        optax.adamw(
            learning_rate=lr_schedule,
            weight_decay=weight_decay,
            mask=decay_mask_fn,
        ),
    )
    
    return optimizer, lr_schedule


def train_step(state, batch, dropout_rng, learning_rate, compute_per_layer=False):
    """Single training step with proper RNG handling and comprehensive metrics."""
    dropout_rng, new_dropout_rng = jax.random.split(dropout_rng)

    def loss_fn(params):
        output = state.apply_fn(
            {"params": params},
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            deterministic=False,
            rngs={"dropout": dropout_rng},
        )
        # Extract logits
        logits = output.logits if hasattr(output, "logits") else output["logits"]
        loss, accuracy, perplexity = compute_mlm_loss_and_accuracy(logits, batch["labels"])
        return loss, (logits, accuracy, perplexity)

    (loss, (logits, accuracy, perplexity)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)

    # Compute gradient metrics
    grad_norm = compute_gradient_norm(grads)
    param_norm = compute_parameter_norm(state.params)

    # Use the passed learning rate
    update_ratio = compute_update_ratio(grads, state.params, learning_rate)

    # Apply gradients
    state = state.apply_gradients(grads=grads)

    # Build metrics dict
    metrics = {
        "loss": loss,
        "accuracy": accuracy,
        "perplexity": perplexity,
        "grad_norm": grad_norm,
        "param_norm": param_norm,
        "update_ratio": update_ratio,
    }

    # Optionally compute per-layer gradient norms
    if compute_per_layer:
        layer_norms = compute_per_layer_gradient_norms(grads)
        metrics["layer_grad_norms"] = layer_norms

    return state, metrics, new_dropout_rng


def load_dataset_for_training(dataset_name="yhavinga/mc4_nl_cleaned",
                              dataset_config="micro",
                              split="train"):
    """Load dataset for training."""
    print(f"Loading dataset: {dataset_name} ({dataset_config})...")

    # Load streaming dataset
    dataset = load_dataset(
        dataset_name,
        dataset_config,
        split=split,
        streaming=True,
    )

    # Shuffle dataset for better training
    dataset = dataset.shuffle(seed=42, buffer_size=1000)

    return dataset


def create_checkpoint_manager(checkpoint_dir: str, checkpoint_every: int, max_to_keep: int = 3):
    """Create CheckpointManager for async checkpointing on TPU."""
    checkpoint_dir = Path(checkpoint_dir).absolute()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Configure async checkpointing
    options = ocp.CheckpointManagerOptions(
        max_to_keep=max_to_keep,
        save_interval_steps=checkpoint_every,
        enable_async_checkpointing=True,  # Enables background saves
    )

    # Create manager with StandardCheckpointer for TrainState
    checkpoint_manager = ocp.CheckpointManager(
        checkpoint_dir,
        options=options,
        metadata={"model": "ModernBERT", "version": "1.0"},
    )

    return checkpoint_manager


def main(args):
    print("=" * 70)
    print("MODERNBERT TRAINING - Production Run with TPU")
    print("=" * 70)

    # Detect TPU and adjust configuration
    backend = jax.default_backend()
    devices = jax.devices()
    n_devices = len(devices)

    print(f"\nDevice Configuration:")
    print(f"  Backend: {backend}")
    print(f"  Device count: {n_devices}")

    # Batch size logic
    if args.batch_size is not None:
        batch_size = args.batch_size
        print(f"  Using specified batch size: {batch_size}")
    elif backend == 'tpu':
        print(f"  TPU devices: {devices}")
        # Adjust batch size for TPU v6e-1 (31GB HBM memory)
        batch_size = 16  # Conservative for 254M model on TPU v6e-1
        print(f"  ✓ TPU detected! Using optimized batch size: {batch_size}")
    else:
        batch_size = 8
        print(f"  Using CPU/GPU with batch size: {batch_size}")

    # Training hyperparameters from args
    max_length = args.max_length
    num_train_steps = args.max_steps
    warmup_steps = args.warmup_steps
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    log_every = args.log_every
    checkpoint_every = args.checkpoint_every
    checkpoint_dir = args.checkpoint_dir
    
    # Model config (small for now)
    config = MMBertConfig(
        vocab_size=256000,
        hidden_size=768,
        num_hidden_layers=6,  # Medium size for testing
        num_attention_heads=12,
        intermediate_size=3072,
        max_position_embeddings=2048,
    )
    print(f"\nModel config:")
    print(f"  Layers: {config.num_hidden_layers}")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Attention heads: {config.num_attention_heads}")
    print(f"  Max seq length: {config.max_position_embeddings}")

    # Initialize W&B
    if not args.no_wandb:
        print("\nInitializing W&B...")
        wandb_config = {
            # Model config
            "model_layers": config.num_hidden_layers,
            "model_hidden_size": config.hidden_size,
            "model_attention_heads": config.num_attention_heads,
            "model_intermediate_size": config.intermediate_size,
            "model_max_position_embeddings": config.max_position_embeddings,
            "model_vocab_size": config.vocab_size,

            # Training config
            "batch_size": batch_size,
            "max_length": max_length,
            "num_train_steps": num_train_steps,
            "warmup_steps": warmup_steps,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "mlm_probability": 0.15,
            "gradient_clipping": 1.0,

            # Dataset
            "dataset": args.dataset,
            "dataset_config": args.dataset_config,

            # Infrastructure
            "backend": backend,
            "n_devices": n_devices,
            "device_type": str(devices[0].device_kind) if devices else "unknown",
        }

        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=wandb_config,
            tags=["modernbert", "mlm", backend, args.dataset_config],
        )
        print(f"  ✓ W&B run initialized: {wandb.run.name}")
        print(f"  ✓ Project: {args.wandb_project}")
    else:
        print("\n✓ W&B logging disabled (--no-wandb)")

    # Initialize model
    print("\nInitializing model...")
    model = ModernBertForMaskedLM(config)
    
    # Init params
    rng = jax.random.PRNGKey(42)
    init_rng, dropout_rng = jax.random.split(rng)
    
    dummy_input = jnp.ones((1, max_length), dtype=jnp.int32)
    dummy_mask = jnp.ones((1, max_length), dtype=jnp.int32)
    
    variables = model.init(
        init_rng,
        input_ids=dummy_input,
        attention_mask=dummy_mask,
        deterministic=True,
    )
    params = variables["params"]
    
    # Count params
    param_count = sum(x.size for x in jax.tree.leaves(params))
    print(f"  Parameters: {param_count:,} ({param_count/1e6:.1f}M)")
    
    # Create optimizer and training state
    print("\nCreating optimizer...")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Warmup steps: {warmup_steps}")
    print(f"  Weight decay: {weight_decay}")
    
    optimizer, lr_schedule = create_train_state(
        model,
        learning_rate=learning_rate,
        num_train_steps=num_train_steps,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
    )
    
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
    )
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("jhu-clsp/mmBERT-small")
    print(f"  Vocab size: {tokenizer.vocab_size}")
    print(f"  Mask token: {tokenizer.mask_token} (ID: {tokenizer.mask_token_id})")
    
    # Create data collator
    collator = FlaxDataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=0.15,
    )
    
    # Load dataset
    dataset = load_dataset_for_training(
        dataset_name=args.dataset,
        dataset_config=args.dataset_config
    )
    
    # JIT compile training step
    print("\nCompiling train_step...")
    # JIT compile with static argnums for compute_per_layer
    train_step_jit = jax.jit(train_step, static_argnums=(4,))

    # Create checkpoint manager for async checkpointing
    print("\nInitializing checkpoint manager...")
    checkpoint_manager = create_checkpoint_manager(
        checkpoint_dir=checkpoint_dir,
        checkpoint_every=checkpoint_every,
        max_to_keep=3,
    )
    print(f"  Checkpoint directory: {checkpoint_dir}")
    print(f"  Async checkpointing: Enabled")
    print(f"  Max checkpoints to keep: 3")

    # Check for existing checkpoints to resume from
    start_step = 0
    latest_step = checkpoint_manager.latest_step()

    if latest_step is not None:
        # Restore from latest checkpoint
        print(f"\nRestoring checkpoint from step {latest_step}...")
        restored = checkpoint_manager.restore(
            latest_step,
            args=ocp.args.StandardRestore(state)
        )
        state = restored
        start_step = latest_step
        print(f"✓ Resumed from step {start_step}")
    else:
        print("\n✓ Starting fresh training")

    # Training loop
    print("\n" + "=" * 70)
    print("TRAINING START")
    print("=" * 70)
    print(f"Total steps: {num_train_steps}")
    print(f"Batch size: {batch_size}")
    print(f"Sequence length: {max_length}")
    print(f"Checkpoint every: {checkpoint_every} steps")
    if start_step > 0:
        print(f"Resuming from step: {start_step}")
    print()

    step = start_step
    batch_buffer = []
    training_start_time = time.time()

    with tqdm(total=num_train_steps - start_step, desc="Training", initial=0) as pbar:
        for example in dataset:
            batch_buffer.append(example)

            if len(batch_buffer) >= batch_size:
                # Create batch
                batch = collator(batch_buffer, max_length=max_length)
                batch_buffer = []

                # Training step - use per-layer metrics every 100 steps
                compute_per_layer = (step % 100 == 0)
                current_lr = float(lr_schedule(step))
                state, metrics, dropout_rng = train_step_jit(state, batch, dropout_rng, current_lr, compute_per_layer)

                # Logging
                if step % log_every == 0:
                    # Extract metrics
                    loss_value = float(metrics["loss"])
                    accuracy_value = float(metrics["accuracy"]) * 100  # Convert to percentage
                    perplexity_value = float(metrics["perplexity"])
                    grad_norm_value = float(metrics["grad_norm"])
                    param_norm_value = float(metrics["param_norm"])
                    update_ratio_value = float(metrics["update_ratio"])

                    lr_value = current_lr  # Already computed above
                    n_masked = int(jnp.sum(batch["labels"] != -100))

                    # Calculate throughput
                    elapsed = time.time() - training_start_time
                    tokens_per_sec = (step - start_step) * batch_size * max_length / elapsed if elapsed > 0 else 0
                    samples_per_sec = tokens_per_sec / max_length
                    steps_per_sec = (step - start_step) / elapsed if elapsed > 0 else 0

                    # Update progress bar
                    pbar.set_postfix({
                        "loss": f"{loss_value:.4f}",
                        "acc": f"{accuracy_value:.1f}%",
                        "ppl": f"{perplexity_value:.1f}",
                        "lr": f"{lr_value:.2e}",
                        "grad": f"{grad_norm_value:.2f}",
                        "ratio": f"{update_ratio_value:.4f}",
                    })

                    # Log to W&B
                    if not args.no_wandb:
                        wandb_log = {
                            "train/loss": loss_value,
                            "train/accuracy": accuracy_value,
                            "train/perplexity": perplexity_value,
                            "train/learning_rate": lr_value,
                            "train/grad_norm": grad_norm_value,
                            "train/param_norm": param_norm_value,
                            "train/update_ratio": update_ratio_value,
                            "train/n_masked_tokens": n_masked,
                            "train/masking_percentage": n_masked / (batch_size * max_length) * 100,
                            "performance/tokens_per_sec": tokens_per_sec,
                            "performance/samples_per_sec": samples_per_sec,
                            "performance/steps_per_sec": steps_per_sec,
                            "step": step,
                        }

                        # Add per-layer gradient norms if available
                        if "layer_grad_norms" in metrics:
                            for layer_name, grad_norm in metrics["layer_grad_norms"].items():
                                wandb_log[f"gradients/{layer_name}"] = float(grad_norm)

                        wandb.log(wandb_log, step=step)

                    if step % (log_every * 5) == 0:  # Print detailed stats occasionally
                        print(f"\nStep {step:4d} | Loss: {loss_value:.4f} | Acc: {accuracy_value:.1f}% | "
                              f"PPL: {perplexity_value:.1f} | LR: {lr_value:.2e} | "
                              f"Grad: {grad_norm_value:.2f} | Ratio: {update_ratio_value:.4f} | "
                              f"Tok/s: {tokens_per_sec:.0f}")

                # Checkpointing - happens async in background
                if step > 0 and step % checkpoint_every == 0:
                    checkpoint_manager.save(
                        step,
                        args=ocp.args.StandardSave(state)
                    )
                    print(f"\n✓ Checkpoint saved (async) at step {step}")
                    print(f"  Elapsed time: {(time.time() - training_start_time)/60:.1f} minutes")

                step += 1
                pbar.update(1)

                if step >= num_train_steps:
                    break

    # Save final checkpoint
    checkpoint_manager.save(
        step,
        args=ocp.args.StandardSave(state)
    )
    print(f"\n✓ Final checkpoint saved (async) at step {step}")

    # Wait for all async checkpoints to complete before exiting
    print("Waiting for async checkpoint operations to complete...")
    checkpoint_manager.wait_until_finished()
    print("✓ All checkpoints saved successfully")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Final loss: {float(metrics['loss']):.4f}")
    print(f"Final accuracy: {float(metrics['accuracy']) * 100:.1f}%")
    print(f"Final perplexity: {float(metrics['perplexity']):.1f}")
    print(f"Total steps: {step}")
    print(f"Training time: {(time.time() - training_start_time)/60:.1f} minutes")

    if backend == 'tpu':
        print(f"\nTPU Performance:")
        print(f"  Tokens/second: {tokens_per_sec:.0f}")
        print(f"  Samples/second: {tokens_per_sec / max_length:.1f}")

    # Log final metrics to W&B
    if not args.no_wandb:
        wandb.summary["final_loss"] = float(metrics['loss'])
        wandb.summary["final_accuracy"] = float(metrics['accuracy']) * 100
        wandb.summary["final_perplexity"] = float(metrics['perplexity'])
        wandb.summary["total_steps"] = step
        wandb.summary["training_time_minutes"] = (time.time() - training_start_time) / 60
        wandb.finish()
        print("\n✓ W&B run finished successfully")

    print()
    print("✓ Training completed successfully!")
    print(f"✓ Checkpoints saved to: {checkpoint_dir}")
    print("\nTo continue training, simply run the script again!")


if __name__ == "__main__":
    args = parse_args()
    main(args)
