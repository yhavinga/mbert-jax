#!/usr/bin/env python3
"""
Production training script - with HF best practices and TPU optimizations.
Trains ModernBERT on Dutch data (mc4_nl_cleaned micro config).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

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
from typing import Dict, List, Optional, Tuple
import orbax.checkpoint as ocp

from mmbjax.configuration_mmbbert import MMBertConfig
from mmbjax.heads.masked_lm import ModernBertForMaskedLM


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


def compute_mlm_loss(logits, labels):
    """Compute cross-entropy loss on masked positions only."""
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
    
    # Masked mean
    loss = -jnp.sum(target_log_probs * mask) / jnp.maximum(jnp.sum(mask), 1)
    
    return loss


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


def train_step(state, batch, dropout_rng):
    """Single training step with proper RNG handling."""
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
        return compute_mlm_loss(logits, batch["labels"])
    
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    
    return state, loss, new_dropout_rng


def load_dutch_dataset(tokenizer, split="train", max_samples=1000):
    """Load Dutch mc4_nl_cleaned micro dataset."""
    print(f"Loading Dutch dataset (yhavinga/mc4_nl_cleaned micro)...")

    # Load streaming dataset
    dataset = load_dataset(
        "yhavinga/mc4_nl_cleaned",
        "micro",
        split=split,
        streaming=True,
    )

    # Take limited samples and shuffle
    dataset = dataset.shuffle(seed=42, buffer_size=1000)
    dataset = dataset.take(max_samples)

    return dataset


def save_checkpoint(state, step: int, checkpoint_dir: str = "checkpoints"):
    """Save checkpoint using orbax for TPU-optimized I/O."""
    # Ensure absolute path for orbax
    checkpoint_dir = Path(checkpoint_dir).absolute()
    checkpoint_path = checkpoint_dir / f"checkpoint_{step}"
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    # Save using orbax (TPU-optimized)
    checkpointer = ocp.PyTreeCheckpointer()
    checkpointer.save(
        checkpoint_path,
        {"state": state, "step": step},
        force=True,
    )

    # Also save metadata
    metadata = {
        "step": step,
        "timestamp": datetime.now().isoformat(),
        "jax_version": jax.__version__,
        "backend": jax.default_backend(),
    }

    with open(checkpoint_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✓ Checkpoint saved to {checkpoint_path}")
    return checkpoint_path


def load_checkpoint(checkpoint_dir: str, step: Optional[int] = None):
    """Load checkpoint from directory."""
    checkpoint_dir = Path(checkpoint_dir)

    if step is not None:
        checkpoint_path = checkpoint_dir / f"checkpoint_{step}"
    else:
        # Find latest checkpoint, excluding temporary orbax directories
        checkpoints = sorted([
            cp for cp in checkpoint_dir.glob("checkpoint_*")
            if not cp.name.endswith('.orbax-checkpoint-tmp')
        ])
        if not checkpoints:
            return None
        checkpoint_path = checkpoints[-1]
        # Extract step number from checkpoint_XXX format
        try:
            step = int(checkpoint_path.name.split("_")[-1])
        except ValueError:
            print(f"Warning: Could not parse step from checkpoint {checkpoint_path.name}")
            return None

    if not checkpoint_path.exists():
        return None

    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpointer = ocp.PyTreeCheckpointer()
    # Orbax requires absolute paths
    absolute_path = checkpoint_path.resolve()
    restored = checkpointer.restore(absolute_path)

    print(f"✓ Checkpoint loaded from step {step}")
    return restored


def main():
    print("=" * 70)
    print("MODERNBERT DUTCH TRAINING - Production Run with TPU")
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
        # Adjust batch size for TPU v6e-1 (31GB HBM memory)
        batch_size = 16  # Conservative for 254M model on TPU v6e-1
        print(f"  ✓ TPU detected! Using optimized batch size: {batch_size}")
    else:
        batch_size = 8
        print(f"  Using CPU/GPU with batch size: {batch_size}")

    # Training hyperparameters
    max_length = 512
    num_train_steps = 500
    warmup_steps = 50
    learning_rate = 5e-5
    weight_decay = 0.01
    eval_every = 50
    log_every = 10
    checkpoint_every = 100  # Checkpoint frequency
    
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
    dataset = load_dutch_dataset(tokenizer, max_samples=5000)
    
    # JIT compile training step
    print("\nCompiling train_step...")
    train_step_jit = jax.jit(train_step)
    
    # Check for existing checkpoints to resume from
    checkpoint_dir = Path("checkpoints")

    # Clean up any stale temporary checkpoint directories
    if checkpoint_dir.exists():
        temp_dirs = list(checkpoint_dir.glob("*.orbax-checkpoint-tmp"))
        if temp_dirs:
            print(f"\nCleaning up {len(temp_dirs)} temporary checkpoint directories...")
            for temp_dir in temp_dirs:
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
                print(f"  Removed: {temp_dir.name}")

    start_step = 0
    resume_checkpoint = load_checkpoint(str(checkpoint_dir))

    if resume_checkpoint is not None:
        # Reconstruct TrainState from the loaded dict
        state_dict = resume_checkpoint["state"]
        # Create a fresh state with loaded params, but let optimizer init its state
        state = train_state.TrainState.create(
            apply_fn=model.apply,
            params=state_dict["params"],
            tx=optimizer,
        )
        # Update the step count to match checkpoint
        state = state.replace(step=state_dict["step"])
        start_step = resume_checkpoint["step"]
        print(f"\n✓ Resuming from step {start_step}")
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

                # Training step
                state, loss, dropout_rng = train_step_jit(state, batch, dropout_rng)

                # Logging
                if step % log_every == 0:
                    loss_value = float(loss)
                    lr_value = float(lr_schedule(step))
                    n_masked = int(jnp.sum(batch["labels"] != -100))

                    # Calculate throughput
                    elapsed = time.time() - training_start_time
                    tokens_per_sec = (step - start_step) * batch_size * max_length / elapsed if elapsed > 0 else 0

                    pbar.set_postfix({
                        "loss": f"{loss_value:.4f}",
                        "lr": f"{lr_value:.2e}",
                        "masked": n_masked,
                        "tok/s": f"{tokens_per_sec:.0f}",
                    })

                    if step % (log_every * 5) == 0:  # Print occasionally
                        print(f"\nStep {step:4d} | Loss: {loss_value:.4f} | LR: {lr_value:.2e} | Tokens/s: {tokens_per_sec:.0f}")

                # Checkpointing
                if step > 0 and step % checkpoint_every == 0:
                    save_checkpoint(state, step, str(checkpoint_dir))
                    print(f"  Elapsed time: {(time.time() - training_start_time)/60:.1f} minutes")

                step += 1
                pbar.update(1)

                if step >= num_train_steps:
                    break

    # Save final checkpoint
    save_checkpoint(state, step, str(checkpoint_dir))

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Final loss: {float(loss):.4f}")
    print(f"Total steps: {step}")
    print(f"Training time: {(time.time() - training_start_time)/60:.1f} minutes")

    if backend == 'tpu':
        print(f"\nTPU Performance:")
        print(f"  Tokens/second: {tokens_per_sec:.0f}")
        print(f"  Samples/second: {tokens_per_sec / max_length:.1f}")

    print()
    print("✓ Training completed successfully!")
    print(f"✓ Checkpoints saved to: {checkpoint_dir}")
    print("\nTo continue training, simply run the script again!")


if __name__ == "__main__":
    main()
