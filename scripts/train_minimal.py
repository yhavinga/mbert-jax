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
        help="Training sequence length in tokens (default: 512)"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=500,
        help="Maximum training steps (default: 500)"
    )

    # Additional training arguments
    parser.add_argument(
        "--per-device-batch-size",
        type=int,
        default=None,
        help="Batch size per device (default: 16 for TPU, 8 for GPU/CPU)"
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
        default=10000,
        help="Number of warmup steps (default: 10000)"
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay (default: 0.01)"
    )
    parser.add_argument(
        "--lr-schedule",
        type=str,
        choices=["inverse_sqrt", "linear", "cosine", "constant"],
        default="inverse_sqrt",
        help="Learning rate schedule type (default: inverse_sqrt)"
    )
    parser.add_argument(
        "--layer-norm-lr-factor",
        type=float,
        default=0.1,
        help="Learning rate factor for layer norm parameters (default: 0.1 = 10x smaller)"
    )
    parser.add_argument(
        "--min-lr",
        type=float,
        default=1e-7,
        help="Minimum learning rate (default: 1e-7, prevents decay to zero)"
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

    # Mixed precision arguments
    parser.add_argument(
        "--compute-dtype",
        type=str,
        choices=["float32", "bfloat16", "float16"],
        default=None,
        help="Dtype for computations (default: bfloat16 on TPU, float32 on GPU/CPU)"
    )
    parser.add_argument(
        "--param-dtype",
        type=str,
        choices=["float32", "bfloat16", "float16"],
        default="float32",
        help="Dtype for parameter storage (default: float32 for stability)"
    )

    # Tokenizer arguments
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="jhu-clsp/mmBERT-small",
        help="HuggingFace tokenizer name (default: jhu-clsp/mmBERT-small)"
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=None,
        help="Override vocab size (default: auto-detect from tokenizer, rounded to multiple of 128 for TPU)"
    )

    # Model architecture arguments
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=768,
        help="Hidden size / embedding dimension (default: 768)"
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=22,
        help="Number of transformer layers (default: 22)"
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=12,
        help="Number of attention heads (default: 12)"
    )
    parser.add_argument(
        "--intermediate-size",
        type=int,
        default=3072,
        help="MLP intermediate size (default: 3072)"
    )
    parser.add_argument(
        "--global-attn-every-n-layers",
        type=int,
        default=3,
        help="Global attention frequency (default: 3, meaning layers 0,3,6,... use global attention)"
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
    lr_schedule_type: str = "inverse_sqrt",
    layer_norm_lr_factor: float = 0.1,
    min_lr: float = 1e-7,
):
    """Create training state with proper optimizer.

    Args:
        model: The model to train
        learning_rate: Peak learning rate
        num_train_steps: Total number of training steps
        warmup_steps: Number of warmup steps
        weight_decay: Weight decay coefficient
        lr_schedule_type: Type of LR schedule ("inverse_sqrt", "linear", "cosine", "constant")
        layer_norm_lr_factor: LR multiplier for layer norm parameters (e.g., 0.1 for 10x smaller)
        min_lr: Minimum learning rate (prevents decay to zero)
    """

    # Create the selected learning rate schedule
    if lr_schedule_type == "inverse_sqrt":
        # Classic Transformer schedule: warmup then inverse sqrt decay
        def inverse_sqrt_schedule(step):
            # Handle step 0 with JAX control flow (not Python if)
            step = jnp.maximum(step, 1)  # Avoid divide by zero

            warmup_factor = jnp.minimum(step / warmup_steps, 1.0)
            decay_factor = jnp.where(
                step < warmup_steps,
                1.0,
                jnp.sqrt(warmup_steps / step)
            )
            lr = learning_rate * warmup_factor * decay_factor

            # Special case: step 0 should have LR = 0
            lr = jnp.where(step == 1, 0.0, lr)

            return jnp.maximum(lr, min_lr)
        lr_schedule = inverse_sqrt_schedule

    elif lr_schedule_type == "linear":
        # Original linear warmup + linear decay (but with min_lr)
        warmup_fn = optax.linear_schedule(
            init_value=0.0,
            end_value=learning_rate,
            transition_steps=warmup_steps,
        )
        decay_fn = optax.linear_schedule(
            init_value=learning_rate,
            end_value=min_lr,  # Changed from 0.0 to min_lr
            transition_steps=num_train_steps - warmup_steps,
        )
        lr_schedule = optax.join_schedules(
            schedules=[warmup_fn, decay_fn],
            boundaries=[warmup_steps],
        )

    elif lr_schedule_type == "cosine":
        # Cosine annealing with warmup
        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=learning_rate,
            warmup_steps=warmup_steps,
            decay_steps=num_train_steps,
            end_value=min_lr,
        )

    elif lr_schedule_type == "constant":
        # Constant LR after warmup (useful for debugging)
        warmup_fn = optax.linear_schedule(
            init_value=0.0,
            end_value=learning_rate,
            transition_steps=warmup_steps,
        )
        constant_fn = optax.constant_schedule(learning_rate)
        lr_schedule = optax.join_schedules(
            schedules=[warmup_fn, constant_fn],
            boundaries=[warmup_steps],
        )
    else:
        raise ValueError(f"Unknown lr_schedule_type: {lr_schedule_type}")
    
    # Weight decay mask: exclude bias and LayerNorm from decay
    def decay_mask_fn(params):
        """Create mask for weight decay: True = apply decay, False = no decay.

        Following standard practice, we exclude:
        1. All bias parameters
        2. All LayerNorm/normalization parameters
        """
        flat_params = traverse_util.flatten_dict(params)

        # Create mask
        flat_mask = {}
        for path in flat_params:
            path_str = "/".join(str(p) for p in path).lower()

            # Check if this parameter should be excluded from weight decay
            no_decay = (
                # Bias terms
                path[-1] == "bias" or
                # LayerNorm parameters (various naming conventions)
                "layer_norm" in path_str or
                "layernorm" in path_str or
                ".ln." in path_str or
                "_ln." in path_str or
                # Normalization layers
                "_norm/" in path_str or  # Added: catches mlp_norm/scale
                "_norm." in path_str or
                "norm/scale" in path_str or  # Added: catches any norm/scale
                "norm/bias" in path_str or   # Added: catches any norm/bias
                "norm.scale" in path_str or
                "norm.bias" in path_str or
                # Common variants
                "norm_" in path_str or
                ".norm_" in path_str or
                ".norm/" in path_str or  # Added: another variant
                # Final layer norm
                "final_layer_norm" in path_str or
                "final_norm" in path_str
            )

            # True = apply weight decay, False = no weight decay
            flat_mask[path] = not no_decay

        return traverse_util.unflatten_dict(flat_mask)
    
    # Create optimizer with optional separate learning rate for layer norms
    if layer_norm_lr_factor != 1.0:
        # Create schedule for layer norm parameters (scaled by factor)
        if callable(lr_schedule):
            # For custom schedules (like inverse_sqrt), wrap them
            ln_lr_schedule = lambda step: lr_schedule(step) * layer_norm_lr_factor
        else:
            # For optax schedules, we need to scale them differently
            # This is a bit hacky but works for optax schedules
            ln_lr_schedule = optax.scale_by_schedule(
                lambda step: layer_norm_lr_factor
            )

        # Create label function to identify layer norm parameters
        def label_fn(path):
            """Label parameters as 'layer_norm' or 'other'."""
            path_str = "/".join(path)
            # Check if this is a layer norm parameter
            if any(ln in path_str.lower() for ln in ["layer_norm", "layernorm", "ln", "norm.scale", "norm.bias"]):
                return "layer_norm"
            return "other"

        # Create the multi-transform optimizer
        # First, we need the two different AdamW optimizers
        main_adamw = optax.adamw(
            learning_rate=lr_schedule,
            weight_decay=weight_decay,
            mask=decay_mask_fn,
        )

        ln_adamw = optax.adamw(
            learning_rate=ln_lr_schedule,
            weight_decay=0.0,  # Usually no weight decay for layer norms
            mask=None,  # Layer norms are already excluded from weight decay
        )

        # Use multi_transform to apply different optimizers based on parameter path
        multi_optimizer = optax.multi_transform(
            transforms={"layer_norm": ln_adamw, "other": main_adamw},
            param_labels=label_fn,
        )

        # Combine with gradient clipping
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),  # Gradient clipping
            multi_optimizer,
        )

        # For logging purposes, return both schedules
        return optimizer, (lr_schedule, ln_lr_schedule)
    else:
        # Standard optimizer without special layer norm treatment
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),  # Gradient clipping
            optax.adamw(
                learning_rate=lr_schedule,
                weight_decay=weight_decay,
                mask=decay_mask_fn,
            ),
        )

        return optimizer, (lr_schedule, None)


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


def create_checkpoint_manager(checkpoint_dir: str, checkpoint_every: int, max_to_keep: int = 3,
                             wandb_run_id: Optional[str] = None):
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
    metadata = {"model": "ModernBERT", "version": "1.0"}
    if wandb_run_id:
        metadata["wandb_run_id"] = wandb_run_id

    checkpoint_manager = ocp.CheckpointManager(
        checkpoint_dir,
        options=options,
        metadata=metadata,
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
    if backend == 'tpu':
        print(f"  TPU devices: {devices}")

    # Batch size logic
    if args.per_device_batch_size is not None:
        per_device_batch_size = args.per_device_batch_size
    elif backend == 'tpu':
        per_device_batch_size = 16  # Optimized for TPU v4-8
    else:
        per_device_batch_size = 8   # Conservative default for GPU/CPU

    # Calculate total batch size across all devices
    total_batch_size = per_device_batch_size * n_devices

    # Display batch size information clearly
    print(f"\nBatch Size Configuration:")
    print(f"  Per-device batch size: {per_device_batch_size}")
    print(f"  Total batch size: {total_batch_size} (= {per_device_batch_size} × {n_devices} devices)")

    # Note: batch_size represents per-device batch size throughout the code
    # JAX automatically handles data parallelism across devices internally
    batch_size = per_device_batch_size

    # Training hyperparameters from args
    max_length = args.max_length
    num_train_steps = args.max_steps
    warmup_steps = args.warmup_steps
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    lr_schedule_type = args.lr_schedule
    layer_norm_lr_factor = args.layer_norm_lr_factor
    min_lr = args.min_lr
    log_every = args.log_every
    checkpoint_every = args.checkpoint_every
    checkpoint_dir = args.checkpoint_dir
    
    # Determine dtype based on backend and arguments
    if args.compute_dtype is not None:
        compute_dtype_str = args.compute_dtype
    elif backend == 'tpu':
        compute_dtype_str = "bfloat16"  # Default to bf16 on TPU for efficiency
        print(f"\n✓ Using bfloat16 precision on TPU (use --compute-dtype to override)")
    else:
        compute_dtype_str = "float32"  # Default to fp32 on GPU/CPU

    param_dtype_str = args.param_dtype

    # Load tokenizer BEFORE model config (to determine vocab_size)
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    # Ensure mask token exists (required for MLM)
    if tokenizer.mask_token is None:
        tokenizer.add_special_tokens({'mask_token': '<mask>'})
        print(f"  Added <mask> token (ID: {tokenizer.mask_token_id})")

    # Determine vocab size
    tokenizer_vocab_size = len(tokenizer)  # Includes added special tokens

    if args.vocab_size is not None:
        # User override
        if args.vocab_size < tokenizer_vocab_size:
            raise ValueError(f"--vocab-size ({args.vocab_size}) must be >= tokenizer size ({tokenizer_vocab_size})")
        vocab_size = args.vocab_size
    else:
        # Auto-round to multiple of 128 for TPU efficiency
        vocab_size = ((tokenizer_vocab_size + 127) // 128) * 128

    print(f"  Tokenizer: {args.tokenizer}")
    print(f"  Tokenizer vocab: {tokenizer_vocab_size}")
    print(f"  Model vocab_size: {vocab_size} (padded for TPU)")
    print(f"  Mask token: {tokenizer.mask_token} (ID: {tokenizer.mask_token_id})")

    # Model config from CLI arguments
    # Following ModernBERT's strategy: set max capacity to 8192 from the start
    # but train with shorter sequences initially for efficiency
    config = MMBertConfig(
        vocab_size=vocab_size,  # Auto-detected from tokenizer
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        intermediate_size=args.intermediate_size,
        global_attn_every_n_layers=args.global_attn_every_n_layers,
        max_position_embeddings=8192,  # Model capacity for long sequences
        compute_dtype=compute_dtype_str,
        param_dtype=param_dtype_str,
    )
    print(f"\nModel config:")
    print(f"  Vocab size: {config.vocab_size}")
    print(f"  Layers: {config.num_hidden_layers}")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Attention heads: {config.num_attention_heads}")
    print(f"  Intermediate size: {config.intermediate_size}")
    print(f"  Global attn every N layers: {config.global_attn_every_n_layers}")
    print(f"  Max pos embeddings: {config.max_position_embeddings}")
    print(f"  Compute dtype: {compute_dtype_str}")
    print(f"  Param dtype: {param_dtype_str}")

    # Check for existing checkpoints early to determine if we're resuming
    print("\nChecking for existing checkpoints...")
    checkpoint_dir_path = Path(checkpoint_dir).absolute()
    wandb_run_id = None
    resuming_from_checkpoint = False

    if checkpoint_dir_path.exists():
        # Create a temporary checkpoint manager to check for checkpoints
        temp_options = ocp.CheckpointManagerOptions(
            max_to_keep=3,
            save_interval_steps=checkpoint_every,
            enable_async_checkpointing=True,
        )
        temp_checkpoint_manager = ocp.CheckpointManager(
            checkpoint_dir_path,
            options=temp_options,
        )
        latest_step = temp_checkpoint_manager.latest_step()

        if latest_step is not None:
            resuming_from_checkpoint = True
            print(f"  Found checkpoint at step {latest_step}")

            # Try to get W&B run ID from metadata
            try:
                metadata = temp_checkpoint_manager.metadata()
                if metadata and "wandb_run_id" in metadata:
                    wandb_run_id = metadata["wandb_run_id"]
                    print(f"  Found W&B run ID in checkpoint: {wandb_run_id}")
            except:
                print("  No W&B run ID found in checkpoint metadata")
        else:
            print("  No checkpoints found in directory")
    else:
        print("  Checkpoint directory does not exist yet")

    # Initialize W&B
    if not args.no_wandb:
        print("\nInitializing W&B...")
        wandb_config = {
            # Model config
            "model_layers": config.num_hidden_layers,
            "model_hidden_size": config.hidden_size,
            "model_attention_heads": config.num_attention_heads,
            "model_intermediate_size": config.intermediate_size,
            "model_global_attn_every_n_layers": config.global_attn_every_n_layers,
            "model_max_position_embeddings": config.max_position_embeddings,
            "model_vocab_size": config.vocab_size,

            # Training config
            "per_device_batch_size": per_device_batch_size,
            "total_batch_size": total_batch_size,
            "max_length": max_length,
            "num_train_steps": num_train_steps,
            "warmup_steps": warmup_steps,
            "learning_rate": learning_rate,
            "lr_schedule": lr_schedule_type,
            "layer_norm_lr_factor": layer_norm_lr_factor,
            "min_lr": min_lr,
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

        # Initialize W&B - either resume existing run or create new one
        if wandb_run_id and resuming_from_checkpoint:
            # Resume existing W&B run
            wandb.init(
                project=args.wandb_project,
                id=wandb_run_id,
                resume="must",
                config=wandb_config,
            )
            print(f"  ✓ W&B run resumed: {wandb.run.name} (ID: {wandb_run_id})")
            print(f"  ✓ Project: {args.wandb_project}")
        else:
            # Create new W&B run
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                config=wandb_config,
                tags=["modernbert", "mlm", backend, args.dataset_config],
            )
            wandb_run_id = wandb.run.id
            print(f"  ✓ W&B run initialized: {wandb.run.name} (ID: {wandb_run_id})")
            print(f"  ✓ Project: {args.wandb_project}")
    else:
        print("\n✓ W&B logging disabled (--no-wandb)")

    # Initialize model with dtype configuration
    print("\nInitializing model...")
    # Convert string dtypes to JAX dtypes
    compute_dtype = getattr(jnp, compute_dtype_str)
    param_dtype = getattr(jnp, param_dtype_str)

    model = ModernBertForMaskedLM(
        config,
        dtype=compute_dtype,
        param_dtype=param_dtype
    )
    
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
    print(f"  LR schedule: {lr_schedule_type}")
    print(f"  Warmup steps: {warmup_steps}")
    print(f"  Min LR: {min_lr}")
    print(f"  Weight decay: {weight_decay}")
    if layer_norm_lr_factor != 1.0:
        print(f"  Layer norm LR factor: {layer_norm_lr_factor} (LR = {learning_rate * layer_norm_lr_factor})")

    optimizer, lr_schedules = create_train_state(
        model,
        learning_rate=learning_rate,
        num_train_steps=num_train_steps,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        lr_schedule_type=lr_schedule_type,
        layer_norm_lr_factor=layer_norm_lr_factor,
        min_lr=min_lr,
    )

    # Unpack schedules
    if isinstance(lr_schedules, tuple):
        lr_schedule, ln_lr_schedule = lr_schedules
    else:
        lr_schedule, ln_lr_schedule = lr_schedules, None
    
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
    )

    # Log weight decay assignment for verification (only when using same LR for all)
    if layer_norm_lr_factor == 1.0:
        print("\nWeight Decay Assignment:")
        # Re-create the decay mask function to check assignments
        flat_params_check = traverse_util.flatten_dict(params)
        decay_info = []

        for path in flat_params_check.keys():
            path_str = "/".join(str(p) for p in path).lower()

            # Same logic as in decay_mask_fn
            no_decay = (
                path[-1] == "bias" or
                "layer_norm" in path_str or
                "layernorm" in path_str or
                ".ln." in path_str or
                "_ln." in path_str or
                "_norm/" in path_str or
                "_norm." in path_str or
                "norm/scale" in path_str or
                "norm/bias" in path_str or
                "norm.scale" in path_str or
                "norm.bias" in path_str or
                "norm_" in path_str or
                ".norm_" in path_str or
                ".norm/" in path_str or
                "final_layer_norm" in path_str or
                "final_norm" in path_str
            )

            decay_info.append((path, not no_decay))

        with_decay = sum(1 for _, has_decay in decay_info if has_decay)
        without_decay = sum(1 for _, has_decay in decay_info if not has_decay)

        print(f"  Parameters WITH weight decay: {with_decay}")
        print(f"  Parameters WITHOUT weight decay: {without_decay}")

        # Show examples
        print("\n  Examples excluded from weight decay:")
        shown = 0
        for path, has_decay in decay_info:
            if not has_decay and shown < 5:
                print(f"    - {'/'.join(str(p) for p in path)}")
                shown += 1
    else:
        print(f"\nUsing separate layer norm LR (factor={layer_norm_lr_factor}):")
        print("  Layer norms automatically have weight_decay=0")

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
        wandb_run_id=wandb_run_id if not args.no_wandb else None,
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
        # For now, just restore state directly (old format)
        # This avoids memory leak from trying to restore into pre-populated dict
        state = checkpoint_manager.restore(
            latest_step,
            args=ocp.args.StandardRestore(state)
        )
        print(f"✓ Restored checkpoint from step {latest_step}")
        # Note: RNG state will be reset for old checkpoints
        # New checkpoints with RNG state will be supported after all old checkpoints expire

        start_step = latest_step
        print(f"✓ Resumed from step {start_step}")
    else:
        print("\n✓ Starting fresh training")

    # Skip already-processed batches when resuming
    if start_step > 0:
        examples_to_skip = start_step * batch_size
        print(f"\nResuming from step {start_step}...")
        print(f"Skipping {examples_to_skip:,} examples in streaming dataset")
        print("Note: HuggingFace .skip() must download/process examples sequentially")
        print("This may take 5-15 minutes depending on network speed...")

        # Use the built-in skip method - it's slow but correct
        # This modifies the dataset to skip the first N examples
        dataset = dataset.skip(examples_to_skip)

        print(f"✓ Dataset positioned at step {start_step}")

    # Training loop
    print("\n" + "=" * 70)
    print("TRAINING START")
    print("=" * 70)
    print(f"Total steps: {num_train_steps}")
    print(f"Per-device batch size: {per_device_batch_size}")
    print(f"Total batch size: {total_batch_size} (across {n_devices} devices)")
    print(f"Training seq length: {max_length} tokens")
    print(f"Checkpoint every: {checkpoint_every} steps")
    if start_step > 0:
        print(f"Resuming from step: {start_step}")
    print()

    step = start_step
    batch_buffer = []
    training_start_time = time.time()

    with tqdm(total=num_train_steps, desc="Training", initial=start_step) as pbar:
        for example in dataset:
            batch_buffer.append(example)

            # Collect examples until we have enough for one per-device batch
            # JAX will automatically replicate this batch across all devices
            if len(batch_buffer) >= batch_size:
                # Create batch - this will be replicated to all devices by JAX's pmap
                batch = collator(batch_buffer, max_length=max_length)
                batch_buffer = []

                # Training step - use per-layer metrics every 100 steps
                compute_per_layer = (step % 100 == 0)
                current_lr = float(lr_schedule(step))
                # Note: We pass the main LR to train_step, layer norm LR is handled internally by optimizer
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
                    # Compute layer norm LR if different
                    ln_lr_value = float(ln_lr_schedule(step)) if ln_lr_schedule else lr_value
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

                        # Add layer norm LR if different
                        if ln_lr_schedule:
                            wandb_log["train/layer_norm_lr"] = ln_lr_value

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
                if step > start_step and step % checkpoint_every == 0:
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
