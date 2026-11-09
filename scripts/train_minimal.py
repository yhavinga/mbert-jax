#!/usr/bin/env python3
"""
Production training script - with HF best practices and TPU optimizations.
Trains ModernBERT on Dutch data (mc4_nl_cleaned micro config).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse
import gc
import os
import json
import re
import resource
import sqlite3
import subprocess
import tempfile
import threading
import time
import traceback
from datetime import datetime
import jax
# Initialize distributed system for multi-host TPU (must be before other JAX calls)
# On TPU/Slurm/MPI: auto-detects config. On local: fails gracefully (single-host).
# A caller that set up the runtime itself (tests/test_multihost.py) must not be re-initialized.
if not jax.distributed.is_initialized():
    try:
        jax.distributed.initialize()
    except ValueError:
        pass  # No coordinator available - running locally, single-host mode
import jax.numpy as jnp
import numpy as np
import optax
from optax import contrib as optax_contrib
from etils import epath
from flax import traverse_util
from flax.training import train_state
from jax.experimental import multihost_utils
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple, Any
import orbax.checkpoint as ocp

from mmbjax.configuration_mmbbert import MMBertConfig
from mmbjax.heads.masked_lm import ModernBertForMaskedLM

# Grain for sequence packing (optional, requires --use-packing)
from mmbjax.training import (
    GRAIN_AVAILABLE,
    PIECE_SHUFFLE_WINDOW,
    PackedMLMDataCollator,
    MixedDenoiserCollator,
    InstructionResponseMaskingCollator,
    HybridMaskingCollator,
    TokenizeTransform,
    create_packed_dataloader,
    create_packed_dataset_streaming,
)
from mmbjax.training.url_blocklist import parse_blocklist


def is_coordinator():
    """Returns True if this is process 0 (coordinator).

    In multi-host TPU training, only the coordinator should:
    - Initialize W&B
    - Log to W&B
    - Print progress updates
    - Show tqdm progress bars

    All processes should:
    - Run training steps (JAX SPMD)
    - Call checkpoint save (Orbax coordinates internally)
    """
    return jax.process_index() == 0


def create_shardings():
    """Return (replicated, batch-split) shardings on a mesh over every device of every host."""
    mesh = jax.sharding.Mesh(np.array(jax.devices()), ("batch",))
    return (
        jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec()),
        jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("batch")),
    )


def to_global(tree, sharding):
    """Build global arrays from this host's data.

    With the replicated sharding every host must pass the same values (nothing checks
    this); with the batch sharding each host passes its own rows, and the global batch
    is the hosts' rows in process order.
    """
    return jax.make_array_from_process_local_data(sharding, jax.tree_util.tree_map(np.asarray, tree))


def write_file(path, data):
    """To a local path or gs://, where a directory needs no creating and mkdir fails with the gcsfs backend."""
    path = epath.Path(path)
    if not str(path).startswith("gs://"):
        path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def load_tracker(name):
    """The module behind --tracker. Trackio follows wandb's init/log/finish, as in `import trackio as wandb`."""
    if name == "trackio":
        import trackio as wandb
    elif name == "wandb":
        import wandb
    else:
        return None
    return wandb


class TrackerDatabaseSync(threading.Thread):
    """Copies Trackio's SQLite database to sync_dir, so progress can be read off the VM.

    Trackio keeps one file per project, <TRACKIO_DIR>/<project>.db, in WAL mode. A plain copy of
    that file can miss what is still in the WAL, so this copies through SQLite's backup API.
    """

    def __init__(self, project, sync_dir, every):
        super().__init__(daemon=True)
        self.project = project
        self.sync_dir = sync_dir
        self.every = every
        self.stop_event = threading.Event()

    def sync_once(self):
        from trackio.sqlite_storage import SQLiteStorage

        database = SQLiteStorage.get_project_db_path(self.project)
        if not database.exists():  # Trackio creates it with the first log
            return
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory) / database.name
            source, target = sqlite3.connect(database), sqlite3.connect(snapshot)
            source.backup(target)
            source.close()
            target.close()
            write_file(epath.Path(self.sync_dir) / database.name, snapshot.read_bytes())

    def run(self):
        while not self.stop_event.wait(self.every):
            try:
                self.sync_once()
            except Exception:
                # Not worth losing a training run over; the next round tries again
                print(f"Tracker database sync to {self.sync_dir} failed:", flush=True)
                traceback.print_exc()

    def finish(self):
        self.stop_event.set()
        self.join()
        self.sync_once()


class DutyCycleSampler(threading.Thread):
    """Runs tpu-info in the background: it takes seconds, and a step must not wait for that."""

    ROW = re.compile(r"\|\s*(\d+)\s*\|\s*[\d.]+ GiB / [\d.]+ GiB\s*\|\s*([\d.]+)%")

    def __init__(self, command, every):
        super().__init__(daemon=True)
        self.command = command
        self.every = every
        self.latest = {}

    def sample(self):
        output = subprocess.run([self.command], capture_output=True, text=True, timeout=120).stdout
        return {int(chip): float(duty) for chip, duty in self.ROW.findall(output)}

    def run(self):
        while True:
            try:
                self.latest = self.sample()
            except Exception:
                self.latest = {}
                print(f"{self.command} failed on process {jax.process_index()}:", flush=True)
                traceback.print_exc()
            time.sleep(self.every)


def start_duty_cycle_sampler(every_seconds):
    """A running sampler, or None without tpu-info (CPU runs); the metric then reads -1."""
    command = Path(sys.executable).with_name("tpu-info")
    if not command.exists():
        if is_coordinator():
            print(f"  No {command}: duty cycle per chip will read -1")
        return None
    sampler = DutyCycleSampler(str(command), every_seconds)
    sampler.start()
    return sampler


def gather_system_metrics(data_loader, duty_sampler, input_seconds_per_step, dataset_num_rows):
    """Every chip of every host, gathered to all hosts. A collective: every host calls this at the same step.

    Per-chip numbers are here because the bug that trained on one chip of sixteen looked fine
    in everything else.
    """
    duty = duty_sampler.latest if duty_sampler is not None else {}
    chips = []
    for device in jax.local_devices():
        stats = device.memory_stats() or {}
        # tpu-info numbers the chips like device.id: 4 to 7 on the second host of a v4-32
        chips.append([stats.get("bytes_in_use", -1), stats.get("peak_bytes_in_use", -1), duty.get(device.id, -1)])
    peak_rss_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    host = [data_loader.documents_read, data_loader.epoch, input_seconds_per_step, peak_rss_bytes, data_loader.pieces_read]
    gathered_chips = multihost_utils.process_allgather(np.array(chips, dtype=np.float32))
    gathered_hosts = multihost_utils.process_allgather(np.array(host, dtype=np.float32))
    gathered_chips = np.asarray(gathered_chips).reshape(jax.process_count(), -1, 3)
    gathered_hosts = np.asarray(gathered_hosts).reshape(jax.process_count(), 5)

    metrics = {}
    for h, host_chips in enumerate(gathered_chips):
        for c, (in_use, peak, duty_cycle) in enumerate(host_chips):
            metrics[f"system/host{h}_chip{c}_bytes_in_use"] = float(in_use)
            metrics[f"system/host{h}_chip{c}_peak_bytes_in_use"] = float(peak)
            metrics[f"system/host{h}_chip{c}_duty_cycle"] = float(duty_cycle)
    for h, (documents, epoch, seconds, peak_rss, pieces) in enumerate(gathered_hosts):
        metrics[f"data/host{h}_documents_read"] = float(documents)
        metrics[f"data/host{h}_pieces_read"] = float(pieces)
        metrics[f"data/host{h}_passes_completed"] = float(epoch)
        metrics[f"performance/host{h}_input_seconds_per_step"] = float(seconds)
        metrics[f"system/host{h}_peak_rss_bytes"] = float(peak_rss)
    if dataset_num_rows:
        metrics["data/passes"] = float(gathered_hosts[:, 0].sum()) / dataset_num_rows
    return metrics


def data_state_path(checkpoint_dir, step, process_index):
    """Next to the checkpoints, because Grain's state is per host and Orbax's save holds what all hosts share."""
    return epath.Path(f"{str(checkpoint_dir).rstrip('/')}_data_state") / str(step) / f"process_{process_index}.json"


def data_state_identity(args, rows_per_host, data_files, masking="uniform15"):
    """What must be the same for the saved indices to point at the same documents, and for a run to stay one experiment."""
    identity = {
        "data_files": data_files,
        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "datasets_config": args.datasets_config,
        "process_count": jax.process_count(),
        "rows_per_host": rows_per_host,
        "max_length": args.max_length,
        "long_documents": "split_into_pieces",
        "piece_shuffle_window": PIECE_SHUFFLE_WINDOW,
        "special_tokens_in_text": "split",
        "tokenizer": args.tokenizer,
        "shuffle_buffer_size": args.shuffle_buffer_size,
    }
    # Only when it isn't the default: data states saved before --masking existed have no such key
    if masking != "uniform15":
        identity["masking"] = masking
    if getattr(args, "url_blocklist", None):
        identity["url_blocklist"] = list(parse_blocklist(args.url_blocklist))
    return identity


def save_data_state(path, data_loader, identity, step):
    write_file(path, json.dumps({"step": step, "identity": identity, "state": data_loader.get_state()}, default=int).encode())


def restore_data_state(path, data_loader, identity, allow_restart):
    """Put data_loader where the checkpoint left it; a resume that would repeat data must be asked for."""
    if not path.exists():
        if allow_restart:
            print(f"No data state at {path}: this host reads its data from the start (--allow-data-restart)", flush=True)
            return
        raise RuntimeError(
            f"No data state at {path}, so this resume would train on the same data again. "
            "Pass --allow-data-restart if that is what you want."
        )
    saved = json.loads(path.read_text())
    if saved["identity"] != identity and identity["data_files"]:
        # A reboot can give a host another process index, and its data files stay with the host
        for other in sorted(path.parent.glob("process_*.json")):
            candidate = json.loads(other.read_text())
            if candidate["identity"] == identity:
                print(f"Process {jax.process_index()} continues the data state {other}, of the same data files", flush=True)
                path, saved = other, candidate
                break
    if saved["identity"] != identity:
        raise RuntimeError(
            f"Data state {path} is of another data setup, its indices would point at other documents: "
            f"saved {saved['identity']}, this run {identity}"
        )
    data_loader.set_state(saved["state"])
    print(f"Data state restored on process {jax.process_index()} at step {saved['step']}: {saved['state']}", flush=True)


def load_params_of_run(step_dir, target_params):
    """The parameters saved in step_dir by another run, in the shapes, dtypes and shardings of target_params.

    Only the parameters are read: that run's optimizer state and step stay behind. All hosts call this, as
    they do the restore of a resume, and each gets the arrays in target_params' (replicated) sharding.
    """
    item_dir = epath.Path(step_dir) / "default"
    if not item_dir.exists():
        raise FileNotFoundError(f"--init-from {step_dir}: no {item_dir}, so this is no step directory of an Orbax checkpoint")
    if str(step_dir).startswith("gs://") and not (epath.Path(step_dir) / "commit_success.txt").exists():
        # On GCS Orbax marks a finished save with this file; without it the save may have been cut off
        raise FileNotFoundError(f"--init-from {step_dir}: no commit_success.txt, the save of this checkpoint did not finish")
    checkpointer = ocp.PyTreeCheckpointer()
    saved = checkpointer.metadata(item_dir).item_metadata
    if "params" not in saved:
        raise ValueError(f"--init-from {step_dir}: no 'params' in the checkpoint, only {sorted(saved)}")
    saved_shapes = {path: (tuple(leaf.shape), np.dtype(leaf.dtype)) for path, leaf in traverse_util.flatten_dict(dict(saved["params"])).items()}
    target_shapes = {path: (tuple(leaf.shape), np.dtype(leaf.dtype)) for path, leaf in traverse_util.flatten_dict(target_params).items()}
    if saved_shapes != target_shapes:
        # partial_restore would skip saved parameters this model lacks: that must not pass unnoticed
        differ = sorted((path for path in saved_shapes.keys() | target_shapes.keys() if saved_shapes.get(path) != target_shapes.get(path)), key=str)
        examples = "; ".join(f"{'/'.join(path)}: saved {saved_shapes.get(path)}, model {target_shapes.get(path)}" for path in differ[:3])
        raise ValueError(f"--init-from {step_dir}: the saved parameters are not those of this model, {len(differ)} differ, e.g. {examples}")
    item = {"params": target_params}
    restored = checkpointer.restore(item_dir, args=ocp.args.PyTreeRestore(
        item=item, restore_args=ocp.checkpoint_utils.construct_restore_args(item), partial_restore=True,
    ))
    return restored["params"]


def log_data_accounting(data_loader, optimizer_step):
    """What each host handed to training in this run, gathered on every host. A collective: all hosts call it."""
    names = ["documents_read", "documents_completed", "pieces_read", "tokens_read", "epoch"]
    mine = data_loader.accounting()
    # Two int32 words per count: without x64 JAX gathers int64 as int32, and tokens pass 2**31 in long runs
    words = np.array([[mine[name] >> 31, mine[name] & (2**31 - 1)] for name in names] + [[0, int(mine["counted_from_start"])]], dtype=np.int32)
    gathered = np.asarray(multihost_utils.process_allgather(words)).reshape(jax.process_count(), len(names) + 1, 2).astype(np.int64)
    hosts = {
        f"host{h}": {**{name: int(gathered[h, i, 0] << 31 | gathered[h, i, 1]) for i, name in enumerate(names)},
                     "counted_from_start": bool(gathered[h, -1, 1])}
        for h in range(jax.process_count())
    }
    if is_coordinator():
        totals = {name: sum(host[name] for host in hosts.values()) for name in names if name != "epoch"}
        print(f"DATA step={optimizer_step} {json.dumps({'hosts': hosts, 'total': totals})}", flush=True)
    return hosts


def create_masking_collator(masking, tokenizer):
    """The training collator of --masking, for packed plain-text data."""
    if masking == "mix":
        return MixedDenoiserCollator(tokenizer)
    return PackedMLMDataCollator(tokenizer=tokenizer, mlm_probability={"uniform15": 0.15, "uniform30": 0.30}[masking])


def create_eval_collator(masking, train_collator, tokenizer):
    """Held-out masks are those of 15% single tokens in every mode, or EVAL lines couldn't be compared across modes."""
    return train_collator if masking == "uniform15" else PackedMLMDataCollator(tokenizer=tokenizer, mlm_probability=0.15)


def load_eval_batches(path, tokenizer, collator, max_length, rows_per_host, n_batches):
    """Held-out packed batches, built once: every evaluation and every run sees the same masks."""
    import pyarrow.parquet as pq

    def documents():
        index = 0
        for record_batch in pq.ParquetFile(path).iter_batches(batch_size=1024, columns=["text"]):
            for text in record_batch.column("text").to_pylist():
                if index % jax.process_count() == jax.process_index():
                    yield {"text": text}
                index += 1

    packed = iter(create_packed_dataset_streaming(documents(), tokenizer=tokenizer, max_length=max_length, batch_size=rows_per_host))
    np.random.seed(1234 + jax.process_index())
    try:
        return [collator(next(packed)) for _ in range(n_batches)]
    except StopIteration:
        raise ValueError(f"{path} has too few documents for {n_batches} eval batches of {rows_per_host} rows per host") from None


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
        default=None,
        help="Dataset configuration/subset (default: None, for datasets without configs)"
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
    parser.add_argument(
        "--stop-at-step",
        type=int,
        default=None,
        help="Stop after this optimizer step while the LR schedule still runs to --max-steps (default: run to --max-steps)"
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
        help="Weight decay (default: 0.01 for AdamW, recommend 0.1 for Muon)"
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=["adamw", "muon"],
        default="adamw",
        help="Optimizer to use (default: adamw). Muon uses less memory but needs weight_decay=0.1"
    )
    parser.add_argument(
        "--lr-schedule",
        type=str,
        choices=["inverse_sqrt", "linear", "cosine", "constant", "wsd"],
        default="inverse_sqrt",
        help="Learning rate schedule type; wsd: warmup, the peak until --decay-steps before --max-steps, then a "
             "cooldown to --min-lr at --max-steps (default: inverse_sqrt)"
    )
    parser.add_argument(
        "--decay-steps",
        type=int,
        default=None,
        help="wsd only: steps of the cooldown, which ends at --max-steps (default: none, required with wsd)"
    )
    parser.add_argument(
        "--decay-shape",
        type=str,
        choices=["1-sqrt", "linear"],
        default="1-sqrt",
        help="wsd only: the cooldown's shape, 1 - sqrt(fraction of the cooldown) as in Hägele et al. 2024 "
             "(arXiv:2405.18392), or linear (default: 1-sqrt)"
    )
    parser.add_argument(
        "--skip-nonfinite-updates",
        action="store_true",
        help="optax.apply_if_finite around the optimizer: a step whose gradients are not finite changes neither "
             "the parameters nor the optimizer state. A finite step is the same as without it"
    )
    parser.add_argument(
        "--max-consecutive-nonfinite",
        type=int,
        default=50,
        help="With --skip-nonfinite-updates: stop the run with an error after this many skipped steps in a row, "
             "with healthy parameters (default: 50)"
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
    parser.add_argument(
        "--init-from",
        type=str,
        default=None,
        help="Step directory of another run's checkpoint, e.g. gs://bucket/run/ckpt/653000: start from its parameters "
             "only, with a fresh optimizer, step 0 and fresh data. A checkpoint in --checkpoint-dir wins: then this "
             "run resumes from its own (default: random initialization)"
    )

    # W&B arguments
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="modernbert-mlm",
        help="Tracker project name (default: modernbert-mlm)"
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="Tracker run name (default: auto-generated). With trackio a resume continues the run of this name"
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Same as --tracker none"
    )
    parser.add_argument(
        "--tracker",
        type=str,
        choices=["wandb", "trackio", "none"],
        default="trackio",
        help="Experiment tracker, on the coordinator only (default: trackio, stored locally in TRACKIO_DIR)"
    )
    parser.add_argument(
        "--tracking-sync-dir",
        type=str,
        default=None,
        help="Copy Trackio's database here, e.g. gs://bucket/run/tracking (default: no copy)"
    )
    parser.add_argument(
        "--tracking-sync-every",
        type=int,
        default=300,
        help="Seconds between copies of Trackio's database (default: 300)"
    )
    parser.add_argument(
        "--system-metrics-every",
        type=int,
        default=1000,
        help="Optimizer steps between per-chip metrics; a step count because all hosts must gather at the same step (default: 1000)"
    )
    parser.add_argument(
        "--per-layer-norms-every",
        type=int,
        default=1000,
        help="Optimizer steps between per-layer gradient norms, a multiple of --log-every. These steps run a second "
             "compiled program; measured on a v4-32 at every 100 steps: 3.7%% of the throughput (default: 1000)"
    )
    parser.add_argument(
        "--dataset-num-rows",
        type=int,
        default=None,
        help="Documents in the dataset, to log the fraction of a pass; streaming doesn't always know (default: only completed passes)"
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
    parser.add_argument(
        "--use-packing",
        action="store_true",
        help="Enable sequence packing using Grain library (more efficient training)"
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=None,
        help="Number of epochs for packing mode (default: None = infinite, auto-restart on exhaustion)"
    )
    parser.add_argument(
        "--data-files-list",
        type=str,
        default=None,
        help="Packing mode: text file on each host with its own local parquet files, one per line. "
             "Replaces streaming: on the v4-32 HF reads timed out up to 4 of the 5 retries that are fatal when exhausted"
    )
    parser.add_argument(
        "--url-blocklist",
        type=str,
        default=None,
        help="With --data-files-list: comma-separated host names, e.g. hebban.nl, whose documents are left out, "
             "with those of their subdomains, by the files' column url (default: none)"
    )
    parser.add_argument(
        "--allow-data-restart",
        action="store_true",
        help="Packing mode: accept a resume from a checkpoint without data state; that host's data starts over"
    )
    parser.add_argument(
        "--eval-file",
        type=str,
        default=None,
        help="Local parquet file with held-out documents in a 'text' column (default: no evaluation)"
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=1000,
        help="Optimizer steps between evaluations on --eval-file (default: 1000)"
    )
    parser.add_argument(
        "--eval-batches",
        type=int,
        default=50,
        help="Global batches in the held-out set (default: 50)"
    )
    
    # Multi-dataset interleaving arguments
    parser.add_argument(
        "--datasets-config",
        type=str,
        default=None,
        help="Path to JSON config file for multiple datasets. Format: {\"datasets\": [{\"name\": \"...\", \"config\": \"...\", \"weight\": 0.5}, ...]}"
    )
    parser.add_argument(
        "--shuffle-buffer-size",
        type=int,
        default=10000,
        help="Buffer size for streaming shuffle (default: 10000)"
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Number of micro-batches to accumulate before optimizer update (default: 1 = disabled)"
    )
    parser.add_argument(
        "--stopping-strategy",
        type=str,
        choices=["first_exhausted", "all_exhausted"],
        default="first_exhausted",
        help="Multi-dataset stopping strategy (default: first_exhausted)"
    )

    # FLAN / instruction-response masking arguments
    parser.add_argument(
        "--dataset-format",
        type=str,
        choices=["text", "inputs_targets"],
        default="text",
        help="Dataset format: 'text' for standard text datasets, 'inputs_targets' for FLAN-style datasets (default: text)"
    )
    parser.add_argument(
        "--masking-mode",
        type=str,
        choices=["standard", "instruction_response"],
        default="standard",
        help="Masking mode: 'standard' for uniform 15%% masking, 'instruction_response' for 0%% instruction/25%% response (default: standard)"
    )
    parser.add_argument(
        "--masking",
        type=str,
        choices=["uniform15", "uniform30", "mix"],
        default="uniform15",
        help="Masking of training batches: single tokens at 15%% or 30%%, or 'mix' for one of 8 denoisers per packed segment; "
             "--eval-file always gets 15%% single tokens (default: uniform15)"
    )

    args = parser.parse_args()
    if args.no_wandb:
        args.tracker = "none"
    if args.eval_file and not args.use_packing:
        parser.error("--eval-file needs --use-packing")
    if args.masking != "uniform15" and (
        not args.use_packing or args.dataset_format != "text" or args.masking_mode != "standard" or args.datasets_config
    ):
        parser.error(f"--masking {args.masking} needs --use-packing on one plain-text dataset, with --masking-mode standard")
    if args.per_layer_norms_every % args.log_every != 0:
        parser.error("--per-layer-norms-every must be a multiple of --log-every: other steps aren't logged")
    if args.lr_schedule == "wsd":
        if args.decay_steps is None or not 0 < args.decay_steps <= args.max_steps - args.warmup_steps:
            parser.error("--lr-schedule wsd needs --decay-steps between 1 and --max-steps minus --warmup-steps")
    elif args.decay_steps is not None:
        parser.error("--decay-steps is for --lr-schedule wsd only")
    if args.url_blocklist:
        if not args.data_files_list:
            parser.error("--url-blocklist needs --data-files-list: it reads the column url of local parquet files")
        try:
            parse_blocklist(args.url_blocklist)
        except ValueError as error:
            parser.error(str(error))
    if args.max_consecutive_nonfinite < 1:
        parser.error("--max-consecutive-nonfinite must be at least 1")
    return args


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
    
    def __call__(self, examples: List[Dict], max_length: int = 512) -> Dict[str, np.ndarray]:
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
            "input_ids": input_ids,
            "attention_mask": batch["attention_mask"],
            "labels": labels,
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
    # Force float32 - log_softmax over 32K vocab loses precision in BF16 on TPU
    logits = logits.astype(jnp.float32)
    
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
    """Compute gradient norms for each layer type, from the module names in the parameter tree."""
    squares = {}

    def add(name, grad):
        squares[name] = squares.get(name, 0.0) + jnp.sum(jnp.square(grad))

    for path, grad in traverse_util.flatten_dict(gradients).items():
        layer = next((part for part in path if re.fullmatch(r"layers_\d+", part)), None)
        if layer is not None:
            index = layer.removeprefix("layers_")
            sublayer = "attention" if any(part.startswith("attention") for part in path) else "ffn"
            add(f"encoder_{index}_{sublayer}", grad)
        elif "embeddings" in path:
            add("embeddings", grad)
        elif "final_layer_norm" in path:
            add("final_layer_norm", grad)
        elif "prediction_head" in path or "decoder" in path:
            add("lm_head", grad)
        else:
            add("other", grad)

    return {name: jnp.sqrt(square) for name, square in squares.items()}


# max_consecutive_errors of apply_if_finite: past it optax would apply a non-finite update. Far above any
# --max-consecutive-nonfinite, and within the int32 of its counter.
NONFINITE_NEVER_APPLIED = 2**30


def wsd_schedule(peak, warmup_steps, total_steps, decay_steps, end_value, shape="1-sqrt"):
    """Warmup-stable-decay: linear warmup from 0 to peak, peak until total_steps - decay_steps, then the cooldown
    to end_value at total_steps, and end_value after it.

    The cooldown of Hägele et al. 2024 (arXiv:2405.18392, "Scaling Laws and Compute-Optimal Training Beyond Fixed
    Training Durations"), with 1-sqrt the factor 1 - sqrt(fraction of the cooldown done) and linear 1 - fraction,
    here from the peak to end_value instead of to 0. The warmup is that of optax.warmup_cosine_decay_schedule, so a
    run on this schedule has the learning rates of a cosine run up to the end of the warmup. Runs with the same
    peak and warmup that differ in total_steps and decay_steps agree on every step before the earlier cooldown,
    which is what lets a cooldown branch start from a checkpoint of the longer run.
    """
    if not 0 < decay_steps <= total_steps - warmup_steps:
        raise ValueError(f"decay_steps {decay_steps} must be between 1 and total_steps - warmup_steps "
                         f"({total_steps} - {warmup_steps})")
    if shape not in ("1-sqrt", "linear"):
        raise ValueError(f"cooldown shape {shape} is not 1-sqrt or linear")
    decay_start = total_steps - decay_steps

    def cooldown(count):
        # join_schedules passes the steps since the boundary
        fraction = jnp.clip(count / decay_steps, 0.0, 1.0)
        done = jnp.sqrt(fraction) if shape == "1-sqrt" else fraction
        # Each half from its own end, so that in float32 the first cooldown step has the stable phase's rate and the
        # last one end_value, both to the last bit
        return jnp.where(done < 0.5, peak - (peak - end_value) * done, end_value + (peak - end_value) * (1.0 - done))

    return optax.join_schedules(
        schedules=[
            optax.linear_schedule(init_value=0.0, end_value=peak, transition_steps=warmup_steps),
            optax.constant_schedule(peak),
            cooldown,
        ],
        boundaries=[warmup_steps, decay_start],
    )


def nonfinite_counts(opt_state):
    """(skipped steps in a row, skipped steps in all) of the apply_if_finite in opt_state, or None without it."""
    found = [node for node in jax.tree_util.tree_leaves(
        opt_state, is_leaf=lambda node: isinstance(node, optax.ApplyIfFiniteState))
        if isinstance(node, optax.ApplyIfFiniteState)]
    if not found:
        return None
    if len(found) > 1:
        raise ValueError(f"{len(found)} apply_if_finite states in the optimizer state, expected one")
    return int(found[0].notfinite_count), int(found[0].total_notfinite)


def create_train_state(
    model,
    learning_rate: float,
    num_train_steps: int,
    warmup_steps: int,
    weight_decay: float = 0.01,
    lr_schedule_type: str = "inverse_sqrt",
    layer_norm_lr_factor: float = 0.1,
    min_lr: float = 1e-7,
    gradient_accumulation_steps: int = 1,
    optimizer_type: str = "adamw",
    decay_steps: Optional[int] = None,
    decay_shape: str = "1-sqrt",
    skip_nonfinite_updates: bool = False,
):
    """Create training state with proper optimizer.

    Args:
        model: The model to train
        learning_rate: Peak learning rate
        num_train_steps: Total number of training steps (optimizer steps, not micro-steps)
        warmup_steps: Number of warmup steps (optimizer steps)
        weight_decay: Weight decay coefficient (recommend 0.01 for AdamW, 0.1 for Muon)
        lr_schedule_type: Type of LR schedule ("inverse_sqrt", "linear", "cosine", "constant")
        layer_norm_lr_factor: LR multiplier for layer norm parameters (only for AdamW)
        min_lr: Minimum learning rate (prevents decay to zero)
        gradient_accumulation_steps: Number of micro-batches to accumulate before update
        optimizer_type: "adamw" or "muon" (Muon uses less memory, good for large models)
        decay_steps: wsd only, the steps of the cooldown that ends at num_train_steps
        decay_shape: wsd only, "1-sqrt" or "linear"
        skip_nonfinite_updates: wrap the optimizer in optax.apply_if_finite (see finish below)
    """

    def finish(optimizer):
        """The optimizer of the branches below, around which the step skip and the accumulation go.

        apply_if_finite looks at the gradients it is given: when one is NaN or inf, the update is zeros and the inner
        state (Adam's moments and count) stays as it was; otherwise the inner optimizer runs as without it. Its
        max_consecutive_errors would apply non-finite updates after that many in a row; the trainer stops itself
        earlier (--max-consecutive-nonfinite), so that number is never reached.
        """
        if skip_nonfinite_updates:
            optimizer = optax.apply_if_finite(optimizer, max_consecutive_errors=NONFINITE_NEVER_APPLIED)
        if gradient_accumulation_steps > 1:
            optimizer = optax.MultiSteps(optimizer, every_k_schedule=gradient_accumulation_steps)
        return optimizer

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

    elif lr_schedule_type == "wsd":
        lr_schedule = wsd_schedule(learning_rate, warmup_steps, num_train_steps, decay_steps, min_lr, decay_shape)

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

    # Create optimizer based on optimizer_type
    if optimizer_type == "muon":
        # Muon optimizer: uses Newton-Schulz orthogonalization for 2D params (weights)
        # Automatically uses AdamW for non-2D params (biases, embeddings, layer norms)
        # Uses less memory than AdamW (only stores momentum, not variance)
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),  # Gradient clipping
            optax_contrib.muon(
                learning_rate=lr_schedule,
                weight_decay=weight_decay,
                # Muon-specific defaults
                beta=0.95,  # Momentum
                nesterov=True,
                ns_steps=5,  # Newton-Schulz iterations
            ),
        )

        return finish(optimizer), (lr_schedule, None)

    # AdamW optimizer with optional separate learning rate for layer norms
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
        def label_fn(params):
            """Label each parameter 'layer_norm' or 'other' by its path: optax passes the whole tree."""
            return traverse_util.path_aware_map(
                lambda path, _: "layer_norm" if any(part == "norm" or part.endswith("_norm") for part in path) else "other",
                params,
            )

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
        return finish(optimizer), (lr_schedule, ln_lr_schedule)
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

        return finish(optimizer), (lr_schedule, None)


def wants_per_layer_norms(optimizer_step, micro_step, gradient_accumulation_steps, every):
    """True for the last micro-step before an optimizer step that is a multiple of `every`: that step logs its metrics."""
    return (optimizer_step + 1) % every == 0 and (micro_step + 1) % gradient_accumulation_steps == 0


def train_step(state, batch, dropout_rng, learning_rate, compute_per_layer=False):
    """Single training step with proper RNG handling and comprehensive metrics."""
    dropout_rng, new_dropout_rng = jax.random.split(dropout_rng)

    def loss_fn(params):
        # Prepare model inputs
        model_kwargs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "deterministic": False,
            "rngs": {"dropout": dropout_rng},
        }
        # Add optional parameters
        if "position_ids" in batch:
            model_kwargs["position_ids"] = batch["position_ids"]
        if "segment_ids" in batch:
            model_kwargs["segment_ids"] = batch["segment_ids"]

        output = state.apply_fn(
            {"params": params},
            **model_kwargs
        )
        # Extract logits
        logits = output.logits if hasattr(output, "logits") else output["logits"]
        loss, accuracy, perplexity = compute_mlm_loss_and_accuracy(logits, batch["labels"])
        
        # Diagnostics for debugging bfloat16 issues
        logits_min = jnp.min(logits)
        logits_max = jnp.max(logits)
        logits_has_nan = jnp.any(jnp.isnan(logits))
        logits_has_inf = jnp.any(jnp.isinf(logits))
        
        return loss, (logits, accuracy, perplexity, logits_min, logits_max, logits_has_nan, logits_has_inf)

    (loss, (logits, accuracy, perplexity, logits_min, logits_max, logits_has_nan, logits_has_inf)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)

    # Check for NaN in gradients
    grad_leaves = jax.tree_util.tree_leaves(grads)
    grads_has_nan = jnp.any(jnp.array([jnp.any(jnp.isnan(g)) for g in grad_leaves]))

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
        # Here because batch is a global array: the coordinator can't count it on its own
        "n_masked": jnp.sum(batch["labels"] != -100),
        "n_tokens": jnp.sum(batch["attention_mask"]),
        # Diagnostics
        "logits_min": logits_min,
        "logits_max": logits_max,
        "logits_has_nan": logits_has_nan,
        "logits_has_inf": logits_has_inf,
        "grads_has_nan": grads_has_nan,
    }

    # Optionally compute per-layer gradient norms
    if compute_per_layer:
        layer_norms = compute_per_layer_gradient_norms(grads)
        metrics["layer_grad_norms"] = layer_norms

    return state, metrics, new_dropout_rng


def load_dataset_for_training(dataset_name="yhavinga/mc4_nl_cleaned",
                              dataset_config="micro",
                              split="train"):
    """Load dataset for training with multi-host sharding support."""
    from datasets.distributed import split_dataset_by_node

    if is_coordinator():
        config_str = f"({dataset_config})" if dataset_config else "(no config)"
        print(f"Loading dataset: {dataset_name} {config_str}...")

    # Load streaming dataset (mc4_nl_cleaned is pre-shuffled)
    # Only pass dataset_config if it's not None (for datasets without configs)
    # reuse_cache_if_exists avoids timeout checking metadata for large datasets
    # token=True for gated datasets like CulturaX (use_auth_token is deprecated)
    dataset = load_dataset(
        dataset_name,
        dataset_config,
        split=split,
        streaming=True,
        download_mode="reuse_cache_if_exists",
        token=True,
    )

    # Shard across processes for multi-host training
    process_count = jax.process_count()
    if process_count > 1:
        dataset = split_dataset_by_node(
            dataset,
            rank=jax.process_index(),
            world_size=process_count
        )
        if is_coordinator():
            print(f"  Dataset sharded across {process_count} processes")

    return dataset


def create_checkpoint_manager(checkpoint_dir: str, checkpoint_every: int, max_to_keep: int = 3,
                             wandb_run_id: Optional[str] = None):
    """Create CheckpointManager for checkpointing.

    IMPORTANT: For multi-host TPU training, checkpoint_dir MUST be a GCS path (gs://...).
    Local paths don't work because Orbax requires shared storage that all processes can access.
    See: https://github.com/google/orbax/issues/999

    All processes must call this function and participate in checkpointing - Orbax
    coordinates internally.
    """
    is_gcs = str(checkpoint_dir).startswith('gs://')
    is_multi_host = jax.process_count() > 1

    # For multi-host, local storage doesn't work - Orbax requires shared storage
    # Return None for ALL processes to disable checkpointing
    if is_multi_host and not is_gcs:
        if is_coordinator():
            print("\n" + "=" * 70)
            print("WARNING: Multi-host training requires GCS for checkpointing!")
            print("=" * 70)
            print(f"  Current path: {checkpoint_dir}")
            print("  Local paths don't work because each TPU host has separate storage.")
            print("  Checkpointing will be DISABLED for this run.")
            print("  To enable checkpointing, use: --checkpoint-dir gs://your-bucket/path")
            print("=" * 70 + "\n")
        return None

    # Set up checkpoint directory (only for single-host local paths)
    if not is_gcs:
        checkpoint_dir_path = Path(checkpoint_dir).absolute()
        checkpoint_dir_path.mkdir(parents=True, exist_ok=True)
        checkpoint_dir = checkpoint_dir_path

    # Only use async checkpointing for multi-host setups where it's beneficial
    # For single-host (even with GCS), sync checkpointing avoids memory fragmentation
    use_async = is_gcs and is_multi_host

    options = ocp.CheckpointManagerOptions(
        max_to_keep=max_to_keep,
        save_interval_steps=checkpoint_every,
        enable_async_checkpointing=use_async,
        create=True,  # Auto-create checkpoint directory if it doesn't exist
    )

    # Create manager - all processes must create this for Orbax coordination
    metadata = {"model": "ModernBERT", "version": "1.0"}
    if wandb_run_id:
        metadata["wandb_run_id"] = wandb_run_id

    # Orbax mkdirs <dir>/metadata and then requires it to exist, which fails on gs:// with the gcsfs
    # backend (no TensorFlow): GCS has no empty directories. Nothing in this repo reads it back.
    if is_gcs and is_coordinator():
        print(f"  Not writing checkpoint root metadata on gs://: {metadata}")

    checkpoint_manager = ocp.CheckpointManager(
        checkpoint_dir,
        options=options,
        metadata=None if is_gcs else metadata,
    )

    return checkpoint_manager


def main(args):
    # Detect TPU and adjust configuration
    backend = jax.default_backend()
    devices = jax.devices()
    n_devices = len(devices)
    process_count = jax.process_count()
    process_idx = jax.process_index()

    if is_coordinator():
        print("=" * 70)
        print("MODERNBERT TRAINING - Production Run with TPU")
        print("=" * 70)
        print(f"\nDevice Configuration:")
        print(f"  Backend: {backend}")
        print(f"  Device count: {n_devices}")
        if process_count > 1:
            print(f"  Multi-host: {process_count} processes (this is process {process_idx})")
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

    # Display batch size information clearly (gradient accumulation info added later after extraction)

    # Note: batch_size is the rows this host loads per micro-step; the hosts' rows together
    # form the global batch of total_batch_size rows, split over all devices
    batch_size = per_device_batch_size * jax.local_device_count()

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
    gradient_accumulation_steps = args.gradient_accumulation_steps

    # Compute effective batch size
    effective_batch_size = per_device_batch_size * n_devices * gradient_accumulation_steps

    # Display batch size configuration
    if is_coordinator():
        print(f"\nBatch Size Configuration:")
        print(f"  Per-device batch size: {per_device_batch_size}")
        print(f"  Devices: {n_devices}")
        print(f"  Gradient accumulation steps: {gradient_accumulation_steps}")
        print(f"  Effective batch size: {effective_batch_size} (= {per_device_batch_size} × {n_devices} × {gradient_accumulation_steps})")
        print(f"  Tokens per optimizer step: {effective_batch_size * max_length:,}")
    
    # Determine dtype based on backend and arguments
    if args.compute_dtype is not None:
        compute_dtype_str = args.compute_dtype
    elif backend == 'tpu':
        compute_dtype_str = "bfloat16"  # Default to bf16 on TPU for efficiency
        if is_coordinator():
            print(f"\n✓ Using bfloat16 precision on TPU (use --compute-dtype to override)")
    else:
        compute_dtype_str = "float32"  # Default to fp32 on GPU/CPU

    param_dtype_str = args.param_dtype

    # Load tokenizer BEFORE model config (to determine vocab_size)
    if is_coordinator():
        print("\nLoading tokenizer...")
    # On every route: a special token's string in a document is text, not that token
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, split_special_tokens=True)

    # Ensure mask token exists (required for MLM)
    if tokenizer.mask_token is None:
        tokenizer.add_special_tokens({'mask_token': '<mask>'})
        if is_coordinator():
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

    if is_coordinator():
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
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        compute_dtype=compute_dtype_str,
        param_dtype=param_dtype_str,
    )
    if is_coordinator():
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
    # Only coordinator checks - W&B run ID recovery happens after checkpoint_manager is created
    wandb_run_id = None
    resuming_from_checkpoint = False

    if is_coordinator():
        print("\nChecking for existing checkpoints...")
        is_gcs = str(checkpoint_dir).startswith('gs://')
        if not is_gcs:
            checkpoint_dir_path = Path(checkpoint_dir).absolute()
            if checkpoint_dir_path.exists():
                # Check for Orbax checkpoint subdirectories (they're named by step number)
                checkpoint_subdirs = [d for d in checkpoint_dir_path.iterdir()
                                      if d.is_dir() and d.name.isdigit()]
                if checkpoint_subdirs:
                    latest_step = max(int(d.name) for d in checkpoint_subdirs)
                    resuming_from_checkpoint = True
                    print(f"  Found checkpoint at step {latest_step}")
                else:
                    print("  No checkpoints found in directory")
            else:
                print("  Checkpoint directory does not exist yet")
        else:
            # For GCS, we'll check via checkpoint_manager later
            print("  Will check GCS for existing checkpoints...")

    # Initialize the tracker (only on coordinator); None on the other hosts and with --tracker none
    wandb = load_tracker(args.tracker) if is_coordinator() else None
    tracker_sync = None
    if wandb is not None:
        print(f"\nInitializing {args.tracker}...")
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
            "optimizer": args.optimizer,
            "per_device_batch_size": per_device_batch_size,
            "total_batch_size": total_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "effective_batch_size": effective_batch_size,
            "max_length": max_length,
            "num_train_steps": num_train_steps,
            "warmup_steps": warmup_steps,
            "learning_rate": learning_rate,
            "lr_schedule": lr_schedule_type,
            "layer_norm_lr_factor": layer_norm_lr_factor,
            "min_lr": min_lr,
            "weight_decay": weight_decay,
            "mlm_probability": {"uniform15": 0.15, "uniform30": 0.30, "mix": None}[args.masking],
            "masking": args.masking,
            "init_from": args.init_from,
            "gradient_clipping": 1.0,
            "decay_steps": args.decay_steps,
            "decay_shape": args.decay_shape if lr_schedule_type == "wsd" else None,
            "skip_nonfinite_updates": args.skip_nonfinite_updates,
            "url_blocklist": args.url_blocklist,

            # Dataset
            "dataset": args.dataset,
            "dataset_config": args.dataset_config,
            "dataset_num_rows": args.dataset_num_rows,
            "tokenizer": args.tokenizer,
            "shuffle_buffer_size": args.shuffle_buffer_size,

            # Infrastructure
            "backend": backend,
            "n_devices": n_devices,
            "process_count": process_count,
            "device_type": str(devices[0].device_kind) if devices else "unknown",
        }

        # Initialize W&B - either resume existing run or create new one
        if args.tracker == "trackio":
            # Local storage only: no space_id, so no HF Space and no write token
            run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                config=wandb_config,
                resume="allow",
                auto_log_gpu=False,
            )
            print(f"  ✓ Trackio run: {run.name}")
            if args.tracking_sync_dir:
                tracker_sync = TrackerDatabaseSync(args.wandb_project, args.tracking_sync_dir, args.tracking_sync_every)
                tracker_sync.start()
                print(f"  ✓ Database copied to {args.tracking_sync_dir} every {args.tracking_sync_every} s")
        elif wandb_run_id and resuming_from_checkpoint:
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
                tags=["modernbert", "mlm", backend],
            )
            wandb_run_id = wandb.run.id
            print(f"  ✓ W&B run initialized: {wandb.run.name} (ID: {wandb_run_id})")
            print(f"  ✓ Project: {args.wandb_project}")
    elif is_coordinator():
        print("\n✓ Tracker disabled (--tracker none)")

    # Initialize model with dtype configuration
    if is_coordinator():
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
    if is_coordinator():
        print(f"  Parameters: {param_count:,} ({param_count/1e6:.1f}M)")

    # Create optimizer and training state
    optimizer_type = args.optimizer
    if is_coordinator():
        print("\nCreating optimizer...")
        print(f"  Optimizer: {optimizer_type.upper()}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  LR schedule: {lr_schedule_type}")
        print(f"  Warmup steps: {warmup_steps}")
        if lr_schedule_type == "wsd":
            print(f"  Cooldown: {args.decay_shape} over the last {args.decay_steps} of {num_train_steps} steps, "
                  f"from step {num_train_steps - args.decay_steps}")
        if args.skip_nonfinite_updates:
            print(f"  Non-finite gradients: step skipped (optax.apply_if_finite), the run stops after "
                  f"{args.max_consecutive_nonfinite} in a row")
        print(f"  Min LR: {min_lr}")
        print(f"  Weight decay: {weight_decay}")
        if optimizer_type == "adamw" and layer_norm_lr_factor != 1.0:
            print(f"  Layer norm LR factor: {layer_norm_lr_factor} (LR = {learning_rate * layer_norm_lr_factor})")
        if optimizer_type == "muon":
            print(f"  Muon momentum (beta): 0.95")
            print(f"  Muon Newton-Schulz steps: 5")
        if gradient_accumulation_steps > 1:
            print(f"  Gradient accumulation: {gradient_accumulation_steps} micro-batches per optimizer step")

    optimizer, lr_schedules = create_train_state(
        model,
        learning_rate=learning_rate,
        num_train_steps=num_train_steps,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        lr_schedule_type=lr_schedule_type,
        layer_norm_lr_factor=layer_norm_lr_factor,
        min_lr=min_lr,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optimizer_type=optimizer_type,
        decay_steps=args.decay_steps,
        decay_shape=args.decay_shape,
        skip_nonfinite_updates=args.skip_nonfinite_updates,
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

    # The same seed gives every host the same values, so these global arrays are consistent.
    # Before the restore below, because Orbax restores into the shardings of this state.
    replicated_sharding, batch_sharding = create_shardings()
    state = to_global(state, replicated_sharding)
    dropout_rng = to_global(dropout_rng, replicated_sharding)

    # Log weight decay assignment for verification (only when using same LR for all)
    if is_coordinator():
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

    # These still pin a host-local copy of the params to the first chip of each host
    del variables, params

    # Create data collator
    if args.use_packing:
        # Check if using multi-dataset config
        dataset_configs = None
        has_flan_dataset = False
        if args.datasets_config:
            with open(args.datasets_config, 'r') as f:
                config_data = json.load(f)
            dataset_configs = config_data.get("datasets", config_data)
            # Check if any dataset uses inputs_targets format
            has_flan_dataset = any(
                cfg.get("format") == "inputs_targets" for cfg in dataset_configs
            )

        # Select collator based on configuration
        if dataset_configs and has_flan_dataset:
            # Mixed datasets: use HybridMaskingCollator
            # This handles both standard (15%) and FLAN (0%/25%) masking automatically
            if args.masking_mode == "instruction_response":
                if is_coordinator():
                    print("\nUsing HybridMaskingCollator for mixed datasets:")
                    print("  - Standard datasets: 15% uniform masking")
                    print("  - FLAN datasets: 0% instruction / 25% response masking")
                collator = HybridMaskingCollator(
                    tokenizer=tokenizer,
                    standard_mask_prob=0.15,
                    instruction_mask_prob=0.0,
                    response_mask_prob=0.25,
                )
            else:
                # Standard mode for all
                collator = PackedMLMDataCollator(
                    tokenizer=tokenizer,
                    mlm_probability=0.15,
                )
        elif args.dataset_format == "inputs_targets":
            # Single FLAN dataset
            collator = InstructionResponseMaskingCollator(
                tokenizer=tokenizer,
                instruction_mask_prob=0.0,
                response_mask_prob=0.25,
            )
            if is_coordinator():
                print("\nUsing InstructionResponseMaskingCollator:")
                print("  - Instruction tokens: 0% masking")
                print("  - Response tokens: 25% masking")
        else:
            # Standard single dataset
            collator = create_masking_collator(args.masking, tokenizer)
    else:
        collator = FlaxDataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm_probability=0.15,
        )

    # Load dataset
    if args.use_packing:
        # Use Grain dataloader with packing and epoch tracking
        if is_coordinator():
            print("\nCreating Grain dataloader with sequence packing...")

        # Parse multi-dataset config if provided
        if dataset_configs is None and args.datasets_config:
            with open(args.datasets_config, 'r') as f:
                config_data = json.load(f)
            dataset_configs = config_data.get("datasets", config_data)
            if is_coordinator():
                print(f"  Loading {len(dataset_configs)} datasets from config:")
                for cfg in dataset_configs:
                    print(f"    - {cfg['name']} ({cfg.get('config', 'default')}): weight={cfg.get('weight', 1.0)}")

        data_files = Path(args.data_files_list).read_text().split() if args.data_files_list else None
        if data_files and dataset_configs:
            raise ValueError("--data-files-list replaces --dataset; it can't be combined with --datasets-config")
        url_blocklist = parse_blocklist(args.url_blocklist) if args.url_blocklist else None
        if url_blocklist:
            import pyarrow.parquet as pq

            # Here and not in the loader's read threads, where a missing column would surface far from its cause
            without_url = [path for path in data_files if "url" not in pq.read_schema(path).names]
            if without_url:
                raise ValueError(f"--url-blocklist reads the column url, which {len(without_url)} of the "
                                 f"--data-files-list files lack, e.g. {without_url[0]}")
            if is_coordinator():
                print(f"  URL blocklist: documents of {', '.join(url_blocklist)} and their subdomains are left out")

        packed_dataloader = create_packed_dataloader(
            dataset_name=args.dataset if not dataset_configs else None,
            dataset_config=args.dataset_config if not dataset_configs else None,
            tokenizer=tokenizer,
            max_length=max_length,
            batch_size=batch_size,  # Rows per host (same as non-packed)
            num_workers=0,  # JAX compatible
            shard_index=jax.process_index(),
            shard_count=jax.process_count(),
            seed=42,
            num_epochs=args.num_epochs,
            dataset_configs=dataset_configs,
            shuffle_buffer_size=args.shuffle_buffer_size,
            stopping_strategy=args.stopping_strategy,
            data_files=data_files,
            url_blocklist=url_blocklist,
        )
        if is_coordinator():
            print(f"  ✓ Packing enabled: documents will be packed into {max_length}-token sequences")
            print(f"  ✓ Rows per host: {batch_size}")
            epochs_str = "infinite" if args.num_epochs is None else str(args.num_epochs)
            print(f"  ✓ Epoch mode: {epochs_str} (auto-restart on exhaustion)")
        data_iter = iter(packed_dataloader)
    else:
        # Use existing data loading
        dataset = load_dataset_for_training(
            dataset_name=args.dataset,
            dataset_config=args.dataset_config
        )

    # JIT compile training step
    if is_coordinator():
        print("\nCompiling train_step...")
    # JIT compile with static argnums for compute_per_layer
    # out_shardings: state must stay replicated for the checkpoints and the next step, and
    # metrics must stay readable by float() on each host
    train_step_jit = jax.jit(train_step, static_argnums=(4,), out_shardings=replicated_sharding)

    eval_batches = None
    if args.eval_file:
        if is_coordinator():
            print(f"\nBuilding {args.eval_batches} held-out batches from {args.eval_file}...")
        eval_collator = create_eval_collator(args.masking, collator, tokenizer)
        eval_batches = load_eval_batches(args.eval_file, tokenizer, eval_collator, max_length, batch_size, args.eval_batches)

        def eval_step(params, batch):
            output = state.apply_fn(
                {"params": params},
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                position_ids=batch.get("position_ids"),
                segment_ids=batch.get("segment_ids"),
                deterministic=True,
            )
            logits = output.logits if hasattr(output, "logits") else output["logits"]
            loss, accuracy, _ = compute_mlm_loss_and_accuracy(logits, batch["labels"])
            n_masked = jnp.sum(batch["labels"] != -100)
            return loss * n_masked, accuracy * n_masked, n_masked

        eval_step_jit = jax.jit(eval_step, out_shardings=replicated_sharding)

    def evaluate(step):
        """Masked-token mean over the held-out set. All hosts run it, the coordinator reports it."""
        loss_sum, correct, n_masked = 0.0, 0.0, 0
        for local_batch in eval_batches:
            batch_loss, batch_correct, batch_masked = eval_step_jit(state.params, to_global(local_batch, batch_sharding))
            loss_sum += float(batch_loss)
            correct += float(batch_correct)
            n_masked += int(batch_masked)
        if is_coordinator():
            # Plain floats: a NumPy float kills Trackio's writer thread, and with it all later logs
            loss, accuracy, perplexity = loss_sum / n_masked, correct / n_masked * 100, float(np.exp(loss_sum / n_masked))
            print(f"EVAL step={step} loss={loss:.5f} accuracy={accuracy:.3f} perplexity={perplexity:.4f} masked={n_masked}", flush=True)
            if wandb is not None:
                wandb.log({"eval/loss": loss, "eval/accuracy": accuracy, "eval/perplexity": perplexity}, step=step)

    # Masking draws from NumPy's global RNG; seeded so runs differ by what they change only
    np.random.seed(42 + process_idx)

    # Not on the TPU: it would wait for the running step, and the next batch can be prepared during that step
    cpu_device = jax.local_devices(backend="cpu")[0]
    duty_sampler = start_duty_cycle_sampler(every_seconds=60)

    # Create checkpoint manager for async checkpointing
    if is_coordinator():
        print("\nInitializing checkpoint manager...")
    checkpoint_manager = create_checkpoint_manager(
        checkpoint_dir=checkpoint_dir,
        checkpoint_every=checkpoint_every,
        max_to_keep=3,
        wandb_run_id=wandb_run_id if is_coordinator() else None,
    )
    if is_coordinator():
        print(f"  Checkpoint directory: {checkpoint_dir}")
        is_gcs = str(checkpoint_dir).startswith('gs://')
        is_multi_host = jax.process_count() > 1
        if is_gcs and is_multi_host:
            print(f"  Async checkpointing: Enabled (multi-host)")
        else:
            print(f"  Async checkpointing: Disabled (single-host, sync for memory efficiency)")
        print(f"  Max checkpoints to keep: 3")

    # Check for existing checkpoints to resume from
    # For multi-host with local storage: checkpoint_manager is None on non-coordinators
    # They start fresh (no resume support without shared storage)
    start_step = 0
    latest_step = checkpoint_manager.latest_step() if checkpoint_manager is not None else None

    if latest_step is not None:
        # Restore from latest checkpoint
        if is_coordinator():
            print(f"\nRestoring checkpoint from step {latest_step}...")
        state = checkpoint_manager.restore(
            latest_step,
            args=ocp.args.StandardRestore(state)
        )
        if is_coordinator():
            print(f"✓ Restored checkpoint from step {latest_step}")

        start_step = latest_step
        if is_coordinator():
            print(f"✓ Resumed from step {start_step}")
            if args.init_from:
                print(f"  --init-from {args.init_from} not used: this run resumes from its own checkpoint")
    elif args.init_from:
        if is_coordinator():
            print(f"\nLoading the parameters of {args.init_from} (--init-from)...")
        state = state.replace(params=load_params_of_run(args.init_from, state.params))
        if is_coordinator():
            print(f"✓ Parameters of {args.init_from}; optimizer, step 0 and data start fresh")
    elif is_coordinator():
        print("\n✓ Starting fresh training")

    # The other half of a packed checkpoint: where this host's data was, so it isn't trained on again
    if args.use_packing:
        data_identity = data_state_identity(args, batch_size, data_files, args.masking)
        if start_step > 0:
            restore_data_state(data_state_path(checkpoint_dir, start_step, process_idx), packed_dataloader, data_identity, args.allow_data_restart)

    # Calculate examples to skip when resuming (only for non-packed)
    # Note: We can't use dataset.skip() because it doesn't work with split_dataset_by_node
    examples_to_skip = start_step * batch_size if start_step > 0 and not args.use_packing else 0
    if start_step > 0 and is_coordinator() and not args.use_packing:
        print(f"\nResuming from step {start_step}...")
        print(f"Will skip {examples_to_skip:,} examples per process during iteration")

    # Training loop
    if is_coordinator():
        print("\n" + "=" * 70)
        print("TRAINING START")
        print("=" * 70)
        print(f"Total optimizer steps: {num_train_steps}")
        print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
        print(f"Per-device batch size: {per_device_batch_size}")
        print(f"Effective batch size: {effective_batch_size} (= {per_device_batch_size} × {n_devices} × {gradient_accumulation_steps})")
        if process_count > 1:
            print(f"Multi-host: {process_count} processes")
        print(f"Training seq length: {max_length} tokens")
        print(f"Tokens per optimizer step: {effective_batch_size * max_length:,}")
        print(f"Checkpoint every: {checkpoint_every} optimizer steps")
        if args.use_packing:
            print(f"Sequence packing: ENABLED")
            # Check Adam optimizer state dtype - critical for bfloat16 stability
            try:
                opt_state = state.opt_state
                # Navigate through optax structure to find first momentum buffer
                first_leaf = jax.tree_util.tree_leaves(opt_state)[0]
                print(f"Optimizer state dtype: {first_leaf.dtype} (should be float32)")
            except Exception as e:
                print(f"Could not check optimizer state dtype: {e}")
        if start_step > 0:
            print(f"Resuming from micro_step: {start_step} (optimizer_step: {start_step // gradient_accumulation_steps})")
        print()

    # With gradient accumulation:
    # - micro_step: increments every forward pass (corresponds to state.step with MultiSteps)
    # - optimizer_step: increments every gradient_accumulation_steps micro-steps (actual param updates)
    # - start_step from checkpoint is micro_step
    micro_step = start_step
    start_optimizer_step = start_step // gradient_accumulation_steps
    optimizer_step = start_optimizer_step

    # Metric accumulators for averaging over accumulation window
    accumulated_loss = 0.0
    accumulated_accuracy = 0.0
    accumulated_perplexity = 0.0
    accumulated_grad_norm = 0.0
    accumulated_param_norm = 0.0
    accumulated_update_ratio = 0.0

    batch_buffer = []
    training_start_time = time.time()
    skipped_examples = 0
    last_optimizer_step = num_train_steps if args.stop_at_step is None else min(args.stop_at_step, num_train_steps)

    # Real tokens (padding excluded) and time to prepare batches, since the last time they were logged
    window_metrics = []
    pending_tokens = []
    tokens_seen = 0
    window_start_time = training_start_time
    window_start_step = start_optimizer_step
    input_seconds = 0.0
    system_input_seconds = 0.0
    system_window_start_step = start_optimizer_step

    # Steps skipped by apply_if_finite so far, resumes included: the counts are in the optimizer state
    skipped_steps = nonfinite_counts(state.opt_state)[1] if args.skip_nonfinite_updates else 0

    # Progress bar shows optimizer steps (not micro-steps); in a log file only every 30 s
    pbar = tqdm(total=num_train_steps, desc="Training", initial=start_optimizer_step,
                mininterval=0.1 if sys.stderr.isatty() else 30) if is_coordinator() else None

    try:
        if args.use_packing:
            # Grain returns pre-batched data
            # Total micro-batches needed: (last_optimizer_step - start_optimizer_step) * gradient_accumulation_steps
            total_micro_batches = (last_optimizer_step - start_optimizer_step) * gradient_accumulation_steps

            for _ in range(total_micro_batches):
                input_start_time = time.time()

                # Get next packed batch from Grain
                try:
                    batch_raw = next(data_iter)
                except StopIteration:
                    # --num-epochs is over on this host. Every host must run every step: see the unpacked route below
                    if process_count > 1:
                        raise RuntimeError(
                            f"Data of process {process_idx} ended after {args.num_epochs} epochs at optimizer_step "
                            f"{optimizer_step} of {last_optimizer_step}; multi-host training cannot continue without this host."
                        ) from None
                    break

                # Process batch through collator (applies MLM masking)
                local_batch = collator(batch_raw)

                # Convert to JAX arrays
                batch = to_global(local_batch, batch_sharding)
                input_seconds += time.time() - input_start_time

                # Debug: Print batch info for first batch only
                # Row contents come from local_batch: a global array can't be computed on by one host
                if micro_step == start_step and is_coordinator():
                    seg_ids = local_batch["segment_ids"]
                    pos_ids = local_batch["position_ids"]
                    labels = local_batch["labels"]
                    print("\n" + "="*60)
                    print(f"PACKED BATCH DIAGNOSTICS (micro_step {micro_step})")
                    print("="*60)
                    print(f"Batch shape: {batch['input_ids'].shape}")
                    print(f"Input IDs row 0: {local_batch['input_ids'][0, :30]}... (first 30)")
                    print(f"Segment IDs row 0: {seg_ids[0, :30]}... (first 30)")
                    print(f"Position IDs row 0: {pos_ids[0, :30]}... (first 30)")
                    unique_segs = [len(set(int(x) for x in row)) for row in seg_ids]
                    print(f"Unique segments per row: {unique_segs}")
                    segment_starts = (pos_ids == 0) & (seg_ids != -1)
                    n_with_bos = int(np.sum(local_batch["input_ids"][segment_starts] == tokenizer.bos_token_id))
                    print(f"Segments on this host starting with {tokenizer.bos_token}: {n_with_bos}/{int(segment_starts.sum())}")
                    n_masked = int(np.sum(labels != -100))
                    n_total = labels.size
                    print(f"Masked tokens on this host: {n_masked}/{n_total} ({100*n_masked/n_total:.1f}%)")
                    # Check for segment -1 (padding)
                    n_padding = int(np.sum(seg_ids == -1))
                    print(f"Padding tokens (segment=-1): {n_padding}/{seg_ids.size} ({100*n_padding/seg_ids.size:.1f}%)")
                    # Check attention_mask
                    attn_mask = local_batch["attention_mask"]
                    n_attn_zeros = int(np.sum(attn_mask == 0))
                    print(f"Attention mask zeros (padding): {n_attn_zeros}/{attn_mask.size}")
                    print("="*60 + "\n", flush=True)


                # Training step (with MultiSteps, optimizer accumulates gradients internally)
                # LR is passed for update_ratio logging; actual LR is handled by optax schedule
                compute_per_layer = wants_per_layer_norms(optimizer_step, micro_step, gradient_accumulation_steps, args.per_layer_norms_every)
                with jax.default_device(cpu_device):
                    current_lr = float(lr_schedule(optimizer_step))
                state, metrics, dropout_rng = train_step_jit(state, batch, dropout_rng, current_lr, compute_per_layer)

                # Kept as arrays until a log step: float() here would wait for the TPU, and the next
                # batch can be prepared while this step runs
                window_metrics.append(metrics)
                pending_tokens.append(metrics["n_tokens"])

                micro_step += 1

                # Check if we completed an accumulation cycle (optimizer step completed)
                if micro_step % gradient_accumulation_steps == 0:
                    optimizer_step += 1

                    if optimizer_step % log_every == 0:
                        # On every host: the others would otherwise queue steps without limit
                        jax.block_until_ready(metrics["loss"])
                        window_tokens = sum(int(t) for t in pending_tokens)
                        tokens_seen += window_tokens
                        pending_tokens = []
                        if args.skip_nonfinite_updates:
                            # Replicated counts, so every host comes to the same decision at the same step
                            in_a_row, skipped = nonfinite_counts(state.opt_state)
                            if skipped != skipped_steps and is_coordinator():
                                print(f"NONFINITE step={optimizer_step} skipped={skipped} in_a_row={in_a_row}: "
                                      f"{skipped - skipped_steps} step(s) with non-finite gradients left the "
                                      f"parameters and optimizer state unchanged", flush=True)
                            skipped_steps = skipped
                            if in_a_row >= args.max_consecutive_nonfinite:
                                raise RuntimeError(
                                    f"{in_a_row} steps in a row up to optimizer_step {optimizer_step} had non-finite "
                                    f"gradients and were skipped; the parameters are still those before them. "
                                    f"Resume from the last checkpoint, e.g. with --compute-dtype float32")

                    # Logging (only on coordinator, using optimizer_step)
                    if optimizer_step % log_every == 0 and is_coordinator():
                        # Average over accumulation window
                        def window_mean(name):
                            return sum(float(m[name]) for m in window_metrics) / len(window_metrics)

                        loss_value = window_mean("loss")
                        accuracy_value = window_mean("accuracy") * 100
                        perplexity_value = window_mean("perplexity")
                        grad_norm_value = window_mean("grad_norm")
                        param_norm_value = window_mean("param_norm")
                        update_ratio_value = window_mean("update_ratio")

                        lr_value = current_lr
                        with jax.default_device(cpu_device):
                            ln_lr_value = float(ln_lr_schedule(optimizer_step)) if ln_lr_schedule else lr_value
                        n_masked = int(metrics["n_masked"])

                        # Diagnostic metrics (from last micro-batch)
                        logits_min = float(metrics["logits_min"])
                        logits_max = float(metrics["logits_max"])
                        logits_has_nan = bool(metrics["logits_has_nan"])
                        logits_has_inf = bool(metrics["logits_has_inf"])
                        grads_has_nan = bool(metrics["grads_has_nan"])

                        # Print diagnostics for first 10 optimizer steps
                        if optimizer_step < start_optimizer_step + 10:
                            print(f"Step {optimizer_step}: loss={loss_value:.4f}, logits=[{logits_min:.2f}, {logits_max:.2f}], "
                                  f"nan={logits_has_nan}, inf={logits_has_inf}, grad_nan={grads_has_nan}, grad_norm={grad_norm_value:.4f}")

                        # Since the previous log step, with padding excluded from the tokens
                        now = time.time()
                        window_seconds = now - window_start_time
                        window_steps = optimizer_step - window_start_step
                        tokens_per_sec = window_tokens / window_seconds
                        step_seconds = window_seconds / window_steps
                        input_seconds_per_step = input_seconds / window_steps
                        optimizer_steps_per_sec = window_steps / window_seconds
                        window_start_time, window_start_step = now, optimizer_step

                        if pbar:
                            pbar.set_postfix({
                                "loss": f"{loss_value:.4f}",
                                "acc": f"{accuracy_value:.1f}%",
                                "ppl": f"{perplexity_value:.1f}",
                                "lr": f"{lr_value:.2e}",
                                "grad": f"{grad_norm_value:.2f}",
                                "ratio": f"{update_ratio_value:.2e}",
                                "tok/s": f"{tokens_per_sec:.0f}",
                                "epoch": packed_dataloader.epoch,
                            }, refresh=False)

                        if optimizer_step % (log_every * 5) == 0:  # Print detailed stats occasionally
                            print(f"\nStep {optimizer_step:4d} | Loss: {loss_value:.4f} | Acc: {accuracy_value:.1f}% | "
                                  f"PPL: {perplexity_value:.1f} | LR: {lr_value:.2e} | "
                                  f"Grad: {grad_norm_value:.2f} | Ratio: {update_ratio_value:.2e} | "
                                  f"Tok/s: {tokens_per_sec:.0f} | Step: {step_seconds:.3f}s | Input: {input_seconds_per_step:.3f}s"
                                  + (f" | Skipped: {skipped_steps}" if args.skip_nonfinite_updates else ""), flush=True)

                        if wandb is not None:
                            wandb_log = {
                                "train/loss": loss_value,
                                "train/accuracy": accuracy_value,
                                "train/perplexity": perplexity_value,
                                "train/learning_rate": lr_value,
                                "train/grad_norm": grad_norm_value,
                                "train/param_norm": param_norm_value,
                                "train/update_ratio": update_ratio_value,
                                "train/n_masked_tokens": n_masked,
                                "train/epoch": packed_dataloader.epoch,
                                "train/effective_batch_size": effective_batch_size,
                                "train/gradient_accumulation_steps": gradient_accumulation_steps,
                                "train/micro_step": micro_step,
                                "performance/tokens_per_sec": tokens_per_sec,
                                "performance/optimizer_steps_per_sec": optimizer_steps_per_sec,
                                "performance/step_seconds": step_seconds,
                                "performance/input_seconds_per_step": input_seconds_per_step,
                            }
                            if ln_lr_schedule:
                                wandb_log["train/layer_norm_lr"] = ln_lr_value
                            if args.skip_nonfinite_updates:
                                wandb_log["train/skipped_steps"] = skipped_steps

                            # Add per-layer gradient norms if available
                            if "layer_grad_norms" in metrics:
                                for layer_name, grad_norm in metrics["layer_grad_norms"].items():
                                    wandb_log[f"gradients/{layer_name}"] = float(grad_norm)

                            wandb.log(wandb_log, step=optimizer_step)

                    if optimizer_step % log_every == 0:
                        system_input_seconds += input_seconds
                        input_seconds = 0.0

                    # Per-chip metrics (a collective: all hosts, at the same step)
                    if optimizer_step % args.system_metrics_every == 0:
                        system_input_seconds += input_seconds
                        input_seconds = 0.0
                        host_input_seconds = system_input_seconds / (optimizer_step - system_window_start_step)
                        system_input_seconds = 0.0
                        system_window_start_step = optimizer_step
                        system_metrics = gather_system_metrics(packed_dataloader, duty_sampler, host_input_seconds, args.dataset_num_rows)
                        if is_coordinator():
                            print(f"SYSTEM step={optimizer_step} {json.dumps(system_metrics)}", flush=True)
                            if wandb is not None:
                                wandb.log(system_metrics, step=optimizer_step)
                            if args.masking == "mix":
                                denoiser_stats = collator.denoiser_stats()
                                print(f"MASKING step={optimizer_step} {json.dumps(denoiser_stats)}", flush=True)
                                if wandb is not None:
                                    wandb.log({f"masking/{name}_{key}": value for name, stats in denoiser_stats.items() for key, value in stats.items()}, step=optimizer_step)

                    if eval_batches is not None and optimizer_step % args.eval_every == 0:
                        evaluate(optimizer_step)

                    # Checkpointing (using micro_step for state, but keyed by optimizer_step)
                    if optimizer_step > start_optimizer_step and optimizer_step % checkpoint_every == 0 and checkpoint_manager is not None:
                        # Before the save: a checkpoint that exists has its data state
                        save_data_state(data_state_path(checkpoint_dir, micro_step, process_idx), packed_dataloader, data_identity, micro_step)
                        checkpoint_manager.save(
                            micro_step,  # Save micro_step for correct resume
                            args=ocp.args.StandardSave(state)
                        )
                        # Wait for checkpoint to complete before continuing
                        checkpoint_manager.wait_until_finished()
                        # Force garbage collection to release checkpoint references
                        gc.collect()
                        if is_coordinator():
                            print(f"\n✓ Checkpoint saved at optimizer_step {optimizer_step} (micro_step {micro_step})", flush=True)

                    # Reset for next accumulation window
                    window_metrics = []

                    # Update progress bar (one step per optimizer step)
                    if pbar:
                        pbar.update(1)

                    # Check termination (based on optimizer steps)
                    if optimizer_step >= last_optimizer_step:
                        break
        else:
            # Original non-packed training loop (with gradient accumulation support)
            for example in dataset:
                # Skip examples when resuming (can't use dataset.skip() with split_dataset_by_node)
                if skipped_examples < examples_to_skip:
                    skipped_examples += 1
                    if skipped_examples % 10000 == 0 and is_coordinator():
                        print(f"  Skipping... {skipped_examples:,}/{examples_to_skip:,} examples")
                    continue

                batch_buffer.append(example)

                # Collect examples until we have this host's rows of the global batch
                if len(batch_buffer) >= batch_size:
                    # Create batch - split over all devices by to_global
                    batch = to_global(collator(batch_buffer, max_length=max_length), batch_sharding)
                    batch_buffer = []

                    if micro_step == start_step and is_coordinator():
                        print(f"Batch shape: {batch['input_ids'].shape}")

                    # Training step (with MultiSteps, optimizer accumulates gradients internally)
                    compute_per_layer = wants_per_layer_norms(optimizer_step, micro_step, gradient_accumulation_steps, args.per_layer_norms_every)
                    current_lr = float(lr_schedule(optimizer_step))
                    state, metrics, dropout_rng = train_step_jit(state, batch, dropout_rng, current_lr, compute_per_layer)

                    # Accumulate metrics for averaging over accumulation window
                    accumulated_loss += float(metrics["loss"])
                    accumulated_accuracy += float(metrics["accuracy"])
                    accumulated_perplexity += float(metrics["perplexity"])
                    accumulated_grad_norm += float(metrics["grad_norm"])
                    accumulated_param_norm += float(metrics["param_norm"])
                    accumulated_update_ratio += float(metrics["update_ratio"])
                    tokens_seen += int(metrics["n_tokens"])

                    micro_step += 1

                    # Check if we completed an accumulation cycle (optimizer step completed)
                    if micro_step % gradient_accumulation_steps == 0:
                        optimizer_step += 1

                        # Compute averaged metrics over accumulation window
                        avg_loss = accumulated_loss / gradient_accumulation_steps
                        avg_accuracy = accumulated_accuracy / gradient_accumulation_steps
                        avg_perplexity = accumulated_perplexity / gradient_accumulation_steps
                        avg_grad_norm = accumulated_grad_norm / gradient_accumulation_steps
                        avg_param_norm = accumulated_param_norm / gradient_accumulation_steps
                        avg_update_ratio = accumulated_update_ratio / gradient_accumulation_steps

                        # Logging (only on coordinator, using optimizer_step)
                        if optimizer_step % log_every == 0 and is_coordinator():
                            loss_value = avg_loss
                            accuracy_value = avg_accuracy * 100
                            perplexity_value = avg_perplexity
                            grad_norm_value = avg_grad_norm
                            param_norm_value = avg_param_norm
                            update_ratio_value = avg_update_ratio

                            lr_value = current_lr
                            ln_lr_value = float(ln_lr_schedule(optimizer_step)) if ln_lr_schedule else lr_value
                            n_masked = int(metrics["n_masked"])

                            # Calculate throughput
                            elapsed = time.time() - training_start_time
                            tokens_per_sec = tokens_seen / elapsed if elapsed > 0 else 0
                            samples_per_sec = (micro_step - start_step) * total_batch_size / elapsed if elapsed > 0 else 0
                            optimizer_steps_per_sec = (optimizer_step - start_optimizer_step) / elapsed if elapsed > 0 else 0

                            # Update progress bar
                            if pbar:
                                pbar.set_postfix({
                                    "loss": f"{loss_value:.4f}",
                                    "acc": f"{accuracy_value:.1f}%",
                                    "ppl": f"{perplexity_value:.1f}",
                                    "lr": f"{lr_value:.2e}",
                                    "grad": f"{grad_norm_value:.2f}",
                                    "ratio": f"{update_ratio_value:.2e}",
                                })

                            # Log to the tracker (only on coordinator, already guarded)
                            if wandb is not None:
                                wandb_log = {
                                    "train/loss": loss_value,
                                    "train/accuracy": accuracy_value,
                                    "train/perplexity": perplexity_value,
                                    "train/learning_rate": lr_value,
                                    "train/grad_norm": grad_norm_value,
                                    "train/param_norm": param_norm_value,
                                    "train/update_ratio": update_ratio_value,
                                    "train/n_masked_tokens": n_masked,
                                    "train/masking_percentage": n_masked / (total_batch_size * max_length) * 100,
                                    "train/effective_batch_size": effective_batch_size,
                                    "train/gradient_accumulation_steps": gradient_accumulation_steps,
                                    "train/micro_step": micro_step,
                                    "performance/tokens_per_sec": tokens_per_sec,
                                    "performance/samples_per_sec": samples_per_sec,
                                    "performance/optimizer_steps_per_sec": optimizer_steps_per_sec,
                                    "step": optimizer_step,
                                }

                                # Add layer norm LR if different
                                if ln_lr_schedule:
                                    wandb_log["train/layer_norm_lr"] = ln_lr_value

                                # Add per-layer gradient norms if available
                                if "layer_grad_norms" in metrics:
                                    for layer_name, grad_norm in metrics["layer_grad_norms"].items():
                                        wandb_log[f"gradients/{layer_name}"] = float(grad_norm)

                                wandb.log(wandb_log, step=optimizer_step)

                            if optimizer_step % (log_every * 5) == 0:  # Print detailed stats occasionally
                                print(f"\nStep {optimizer_step:4d} | Loss: {loss_value:.4f} | Acc: {accuracy_value:.1f}% | "
                                      f"PPL: {perplexity_value:.1f} | LR: {lr_value:.2e} | "
                                      f"Grad: {grad_norm_value:.2f} | Ratio: {update_ratio_value:.2e} | "
                                      f"Tok/s: {tokens_per_sec:.0f}")

                        # Checkpointing (using micro_step for state, keyed by optimizer_step)
                        if optimizer_step > start_optimizer_step and optimizer_step % checkpoint_every == 0 and checkpoint_manager is not None:
                            checkpoint_manager.save(
                                micro_step,  # Save micro_step for correct resume
                                args=ocp.args.StandardSave(state)
                            )
                            checkpoint_manager.wait_until_finished()
                            gc.collect()
                            if is_coordinator():
                                print(f"\n✓ Checkpoint saved at optimizer_step {optimizer_step} (micro_step {micro_step})")
                                print(f"  Elapsed time: {(time.time() - training_start_time)/60:.1f} minutes")

                        # Reset accumulators for next accumulation window
                        accumulated_loss = 0.0
                        accumulated_accuracy = 0.0
                        accumulated_perplexity = 0.0
                        accumulated_grad_norm = 0.0
                        accumulated_param_norm = 0.0
                        accumulated_update_ratio = 0.0

                        # Update progress bar (one step per optimizer step)
                        if pbar:
                            pbar.update(1)

                        # Check termination (based on optimizer steps)
                        if optimizer_step >= last_optimizer_step:
                            break

            # Every host must run every step. The others now wait in a collective, and all hosts
            # abort when the shutdown barrier of this one times out (5 minutes, measured on CPU)
            if process_count > 1 and optimizer_step < last_optimizer_step:
                raise RuntimeError(
                    f"Dataset shard of process {process_idx} ran out at optimizer_step {optimizer_step} "
                    f"of {last_optimizer_step}; multi-host training cannot continue without this host."
                )

    finally:
        # Close progress bar if it exists
        if pbar:
            pbar.close()

    # Save final checkpoint (only if checkpoint_manager exists)
    if checkpoint_manager is not None:
        # force: without it the manager skips a step that isn't a multiple of its interval, and this
        # run would resume from the last multiple while the print below says otherwise
        if checkpoint_manager.latest_step() != micro_step:
            if args.use_packing:
                save_data_state(data_state_path(checkpoint_dir, micro_step, process_idx), packed_dataloader, data_identity, micro_step)
            checkpoint_manager.save(
                micro_step,  # Save micro_step for correct resume
                args=ocp.args.StandardSave(state),
                force=True,
            )
        if is_coordinator():
            print(f"\n✓ Final checkpoint saved at optimizer_step {optimizer_step} (micro_step {micro_step})")

        # Wait for all async checkpoints to complete before exiting
        if is_coordinator():
            print("Waiting for checkpoint operations to complete...")
        checkpoint_manager.wait_until_finished()
        if is_coordinator():
            print("✓ All checkpoints saved successfully")

    # Over the whole run, resumes included: two runs on the same stream can be held against each other
    if args.use_packing:
        log_data_accounting(packed_dataloader, optimizer_step)

    # Calculate final metrics for logging
    elapsed = time.time() - training_start_time
    tokens_seen += sum(int(t) for t in pending_tokens)
    tokens_per_sec = tokens_seen / elapsed if elapsed > 0 else 0

    if is_coordinator():
        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)
        print(f"Final loss: {float(metrics['loss']):.4f}")
        print(f"Final accuracy: {float(metrics['accuracy']) * 100:.1f}%")
        print(f"Final perplexity: {float(metrics['perplexity']):.1f}")
        print(f"Total optimizer steps: {optimizer_step}")
        print(f"Total micro steps: {micro_step}")
        print(f"Training time: {elapsed/60:.1f} minutes")

        if backend == 'tpu':
            print(f"\nTPU Performance:")
            print(f"  Tokens/second: {tokens_per_sec:.0f}")
            print(f"  Samples/second: {(micro_step - start_step) * total_batch_size / elapsed:.1f}")

    # Log final metrics to the tracker (only on coordinator)
    if args.tracker == "trackio" and is_coordinator():
        wandb.finish()
        if tracker_sync is not None:
            tracker_sync.finish()
        print("\n✓ Trackio run finished successfully")
    if args.tracker == "wandb" and is_coordinator():
        wandb.summary["final_loss"] = float(metrics['loss'])
        wandb.summary["final_accuracy"] = float(metrics['accuracy']) * 100
        wandb.summary["final_perplexity"] = float(metrics['perplexity'])
        wandb.summary["total_optimizer_steps"] = optimizer_step
        wandb.summary["total_micro_steps"] = micro_step
        wandb.summary["gradient_accumulation_steps"] = gradient_accumulation_steps
        wandb.summary["effective_batch_size"] = effective_batch_size
        wandb.summary["training_time_minutes"] = elapsed / 60
        wandb.finish()
        print("\n✓ W&B run finished successfully")

    if is_coordinator():
        print()
        print("✓ Training completed successfully!")
        print(f"✓ Checkpoints saved to: {checkpoint_dir}")
        print("\nTo continue training, simply run the script again!")

    return state


if __name__ == "__main__":
    args = parse_args()
    try:
        main(args)
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        # Not a normal exit: atexit would call jax.distributed.shutdown(), whose barrier waits for
        # peers that are stuck in a collective. Measured on the v4-32: still there after 10 minutes.
        os._exit(1)
