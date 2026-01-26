#!/usr/bin/env python3
"""Task runner for spot TPU VMs.

Runs fine-tuning with GCS heartbeat and checkpoint management.
Designed to be launched by orchestrator.py on a fresh spot VM.

Usage:
    python -m infra.spot.task_runner \
        --task-id 22l-mc4--dbrd \
        --checkpoint gs://mbert-checkpoints/dumb-eval/checkpoints/22L-mc4-lowlr \
        --task dbrd \
        --output gs://mbert-checkpoints/dumb-eval/runs/22l-mc4--dbrd
"""

import argparse
import os
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

# Add src to path for mmbjax imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from . import state as state_ops
from .config import GCS_RESULTS_DIR, HEARTBEAT_INTERVAL


class HeartbeatThread(threading.Thread):
    """Background thread that sends heartbeats to GCS."""

    def __init__(self, task_id: str, interval: int = HEARTBEAT_INTERVAL):
        super().__init__(daemon=True)
        self.task_id = task_id
        self.interval = interval
        self.step = 0
        self.best_score = None
        self.stop_event = threading.Event()

    def update(self, step: int, best_score: float | None = None):
        """Update the current step and score for the next heartbeat."""
        self.step = step
        if best_score is not None:
            self.best_score = best_score

    def stop(self):
        """Signal the thread to stop."""
        self.stop_event.set()

    def run(self):
        """Main loop: send heartbeat every interval seconds."""
        while not self.stop_event.wait(self.interval):
            try:
                state_ops.heartbeat(self.task_id, self.step, self.best_score)
                print(f"[heartbeat] step={self.step}, score={self.best_score}")
            except Exception as e:
                print(f"[heartbeat] Error: {e}")


def download_from_gcs(gcs_path: str, local_path: str) -> bool:
    """Download a file or directory from GCS."""
    # Create parent directory
    os.makedirs(local_path, exist_ok=True)
    result = subprocess.run(
        ["gcloud", "storage", "cp", "-r", f"{gcs_path}/*", local_path],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"[error] GCS download failed: {result.stderr}")
        return False
    return True


def upload_to_gcs(local_path: str, gcs_path: str) -> bool:
    """Upload a file or directory to GCS."""
    result = subprocess.run(
        ["gcloud", "storage", "cp", "-r", local_path, gcs_path],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"[error] GCS upload failed: {result.stderr}")
        return False
    return True


def install_dependencies():
    """Install JAX and other dependencies on a fresh VM."""
    print("[setup] Installing dependencies...")

    # Use python3 -m pip to ensure packages go to the right location
    commands = [
        "python3 -m pip install --upgrade pip",
        "python3 -m pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html",
        "python3 -m pip install flax optax transformers datasets tqdm",
    ]

    for cmd in commands:
        print(f"  Running: {cmd}")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  [error] {result.stderr}")
            return False

    # Add user site-packages to sys.path so we can import the installed packages
    import site
    site.main()  # Reload site-packages
    user_site = site.getusersitepackages()
    if user_site not in sys.path:
        sys.path.insert(0, user_site)
        print(f"  Added {user_site} to sys.path")

    # Invalidate import caches so Python sees newly installed packages
    import importlib
    importlib.invalidate_caches()

    # Verify JAX is importable
    print("  Verifying JAX installation...")
    try:
        import jax
        print(f"  JAX {jax.__version__}, {len(jax.devices())} devices")
    except ImportError as e:
        print(f"  [error] JAX import failed: {e}")
        return False

    print("[setup] Dependencies installed")
    return True


def run_finetuning(
    task_id: str,
    checkpoint_path: str,
    task: str,
    output_path: str,
    heartbeat: HeartbeatThread,
) -> dict:
    """Run the fine-tuning process.

    Returns results dict with test metrics.
    """
    # Import here after dependencies are installed
    import json
    import pickle
    from functools import partial

    import jax
    import jax.numpy as jnp
    import numpy as np
    from flax import jax_utils
    from tqdm import tqdm
    from transformers import AutoTokenizer

    # Import from the project
    from mmbjax.configuration_mmbbert import MMBertConfig
    from mmbjax.heads.classification import ModernBertForSequenceClassification

    # Import finetune utilities
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts" / "dumb"))
    from data import TASKS, get_task
    from finetune import (
        NUM_DEVICES,
        TrainState,
        collate_fn,
        create_model_and_state,
        evaluate,
        load_pretrained_params,
        p_train_step,
        shard_batch,
    )

    print(f"[train] Devices: {NUM_DEVICES}")

    # Load checkpoint
    print(f"[train] Loading checkpoint from {checkpoint_path}")
    pretrained_params, config = load_pretrained_params(Path(checkpoint_path))

    # Get tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.tokenizer_name or "yhavinga/dutch-llama-tokenizer"
    )

    # Load dataset
    task_config = get_task(task)
    print(f"[train] Loading dataset for {task}")
    dataset = task_config["loader"](tokenizer, 512)
    print(
        f"[train] Train: {len(dataset['train'])}, Val: {len(dataset['validation'])}, Test: {len(dataset['test'])}"
    )

    # Create model and state
    rng = jax.random.PRNGKey(42)
    model, state = create_model_and_state(
        config=config,
        pretrained_params=pretrained_params,
        num_labels=task_config["num_labels"],
        learning_rate=2e-5,
        rng=rng,
    )

    # Replicate state across devices for pmap
    state = jax_utils.replicate(state)

    # Calculate total steps
    per_device_batch_size = 32
    # Reduce memory pressure for dbrd (22L) to avoid HBM OOMs
    if task == "dbrd":
        per_device_batch_size = 24

    total_batch_size = per_device_batch_size * NUM_DEVICES
    epochs = 3
    steps_per_epoch = len(dataset["train"]) // total_batch_size
    total_steps = steps_per_epoch * epochs

    print(f"[train] Per-device batch: {per_device_batch_size}, Total batch: {total_batch_size}")

    # Update state with total_steps
    s = state_ops.get_state(task_id)
    if s:
        s["total_steps"] = total_steps
        state_ops.put_state(task_id, s)

    # Collate function
    collate = partial(collate_fn, pad_token_id=tokenizer.pad_token_id, max_length=512)

    # Training loop
    best_val_accuracy = 0.0
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0

    for epoch in range(epochs):
        print(f"\n[train] Epoch {epoch + 1}/{epochs}")

        # Shuffle training data
        train_indices = list(range(len(dataset["train"])))
        np.random.shuffle(train_indices)

        # Training
        epoch_loss = 0.0
        epoch_correct = 0
        num_batches = 0

        pbar = tqdm(range(0, len(train_indices), total_batch_size), desc="Training")
        for i in pbar:
            batch_indices = train_indices[i : i + total_batch_size]
            # Skip incomplete batches
            if len(batch_indices) < total_batch_size:
                continue
            batch = [dataset["train"][j] for j in batch_indices]
            batch = collate(batch)
            batch = shard_batch(batch)

            state, metrics = p_train_step(state, batch)

            # Metrics are replicated, take first device's value
            loss = float(metrics["loss"][0])
            accuracy = float(metrics["accuracy"][0])

            epoch_loss += loss
            epoch_correct += accuracy * len(batch_indices)
            num_batches += 1
            global_step += 1

            # Update heartbeat
            heartbeat.update(global_step, best_val_accuracy)

            pbar.set_postfix({"loss": f"{loss:.4f}", "acc": f"{accuracy:.4f}"})

        train_accuracy = epoch_correct / len(train_indices)
        print(
            f"[train] Train Loss: {epoch_loss / num_batches:.4f}, Train Accuracy: {train_accuracy:.4f}"
        )

        # Validation
        val_metrics = evaluate(state, dataset["validation"], collate, total_batch_size)
        print(
            f"[train] Val Loss: {val_metrics['loss']:.4f}, Val Accuracy: {val_metrics['accuracy']:.4f}"
        )

        # Save best model
        if val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            print(f"[train] New best! Saving checkpoint...")

            # Unreplicate state to save single copy of params
            unreplicated_state = jax_utils.unreplicate(state)
            with open(output_dir / "params.pkl", "wb") as f:
                pickle.dump(unreplicated_state.params, f)

            config_dict = config.to_dict()
            config_dict["num_labels"] = task_config["num_labels"]
            with open(output_dir / "config.json", "w") as f:
                json.dump(config_dict, f, indent=2)

            # Update heartbeat with new best score
            heartbeat.update(global_step, best_val_accuracy)

    # Final evaluation on test set
    print("\n[train] Evaluating on test set...")
    test_metrics = evaluate(state, dataset["test"], collate, total_batch_size)
    print(
        f"[train] Test Loss: {test_metrics['loss']:.4f}, Test Accuracy: {test_metrics['accuracy']:.4f}"
    )

    # Save results
    results = {
        "task_id": task_id,
        "task": task,
        "checkpoint": checkpoint_path,
        "best_val_accuracy": best_val_accuracy,
        "test_accuracy": test_metrics["accuracy"],
        "test_loss": test_metrics["loss"],
        "epochs": epochs,
        "total_steps": total_steps,
        "num_devices": NUM_DEVICES,
        "total_batch_size": total_batch_size,
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser(description="Run fine-tuning task on spot VM")
    parser.add_argument("--task-id", required=True, help="Task ID for state tracking")
    parser.add_argument("--checkpoint", required=True, help="GCS path to checkpoint")
    parser.add_argument("--task", required=True, help="Task name (dbrd, sicknl, etc.)")
    parser.add_argument("--output", required=True, help="GCS path for output")
    parser.add_argument(
        "--skip-install", action="store_true", help="Skip dependency installation"
    )
    args = parser.parse_args()

    print(f"[task_runner] Starting task {args.task_id}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Task: {args.task}")
    print(f"  Output: {args.output}")

    # Update state to running
    s = state_ops.get_state(args.task_id)
    if s is None:
        print(f"[error] No state found for task {args.task_id}")
        sys.exit(1)

    s["status"] = "running"
    s["started_at"] = datetime.now(timezone.utc).isoformat()
    state_ops.put_state(args.task_id, s)

    # Start heartbeat thread
    heartbeat = HeartbeatThread(args.task_id)
    heartbeat.start()

    try:
        # Install dependencies
        if not args.skip_install:
            if not install_dependencies():
                raise RuntimeError("Failed to install dependencies")

        # Create temp directory for local work
        with tempfile.TemporaryDirectory() as tmpdir:
            local_checkpoint = os.path.join(tmpdir, "checkpoint")
            local_output = os.path.join(tmpdir, "output")

            # Download checkpoint from GCS
            print(f"[task_runner] Downloading checkpoint...")
            if not download_from_gcs(args.checkpoint, local_checkpoint):
                raise RuntimeError("Failed to download checkpoint")

            # Run fine-tuning
            results = run_finetuning(
                task_id=args.task_id,
                checkpoint_path=local_checkpoint,
                task=args.task,
                output_path=local_output,
                heartbeat=heartbeat,
            )

            # Upload results to GCS
            print(f"[task_runner] Uploading results to {args.output}")
            if not upload_to_gcs(local_output, args.output):
                raise RuntimeError("Failed to upload results")

            # Also save results.json to results dir
            results_gcs = f"{GCS_RESULTS_DIR}/{args.task_id}.json"
            results_local = os.path.join(local_output, "results.json")
            upload_to_gcs(results_local, results_gcs)

        # Mark task as completed
        s = state_ops.get_state(args.task_id)
        s["status"] = "completed"
        s["best_dev_score"] = results["best_val_accuracy"]
        state_ops.put_state(args.task_id, s)

        print(f"[task_runner] Task completed successfully!")
        print(f"  Best val accuracy: {results['best_val_accuracy']:.4f}")
        print(f"  Test accuracy: {results['test_accuracy']:.4f}")

    except Exception as e:
        print(f"[task_runner] Error: {e}")
        # Mark task as failed
        s = state_ops.get_state(args.task_id)
        if s:
            s["status"] = "failed"
            s["error"] = str(e)
            state_ops.put_state(args.task_id, s)
        sys.exit(1)

    finally:
        heartbeat.stop()


if __name__ == "__main__":
    main()
