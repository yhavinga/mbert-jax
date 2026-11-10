#!/usr/bin/env python3
"""LoRA SFT of TranslateGemma with MaxText 0.2.5's train_sft (Tunix's PeftTrainer, qwix LoRA), with the measurements
of the smoke test, and the adapter exported for translate.py --lora.

    python sft_lora.py --out DIR -- <MaxText overrides key=value ...>      (run_sft.sh)

The config is MaxText's post_train/sft.yml plus the overrides. One epoch: the training data first go once through
MaxText's own pipeline (create_data_iterator on the training mesh) to count its batches, and `steps` becomes that count,
so the cosine schedule ends with the epoch. Then MaxText's setup_trainer_state and train_model, unchanged, with two
wrapped hooks that keep, per step, the loss, the time it ended and the batch's token counts.

Writes to DIR (process 0): summary.json (configuration, times, throughput, memory per chip, loss per step), adapter.npz
(the LoRA parameters in float32, keys as decoder/layers_0/self_attention/query/kernel_lora_a), adapter.json (rank,
alpha, module regex, model, checkpoint, original dtype); MaxText's output (metrics file, tensorboard, Tunix's checkpoint
of the LoRA parameters) under DIR/maxtext.
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path


def split_argv(argv):
    """(own arguments, MaxText overrides) around the first `--`."""
    if "--" not in argv:
        return argv, []
    k = argv.index("--")
    return argv[:k], argv[k + 1:]


def adapter_arrays(lora_pure):
    """({"/"-joined path: float32 numpy array}, {dtypes}) of LoRA parameters given as a pure dict; each must be
    replicated (MaxText's apply_lora_to_model replicates them), so one device holds the whole array."""
    import jax
    import numpy as np

    flat, dtypes = {}, set()
    for path, leaf in jax.tree_util.tree_flatten_with_path(lora_pure)[0]:
        key = "/".join(str(getattr(p, "key", getattr(p, "name", getattr(p, "idx", p)))) for p in path)
        if hasattr(leaf, "sharding") and not leaf.sharding.is_fully_replicated:
            raise SystemExit(f"LoRA parameter {key} is not replicated: {leaf.sharding}")
        dtypes.add(str(leaf.dtype))
        value = leaf.addressable_data(0) if hasattr(leaf, "addressable_data") else leaf
        flat[key] = np.asarray(value).astype(np.float32)
    if not flat:
        raise SystemExit("the model has no LoRA parameters")
    return flat, dtypes


def throughput(records, tokens, loss_tokens):
    """Steady-state rates from per-step end times: the first step carries the compilation, so the rates run from the
    end of step 1 to the end of the last; compile_s is step 1's time minus the median of the others."""
    times = [r["t"] for r in records]
    durations = [b - a for a, b in zip(times, times[1:])]
    if not durations:
        return {}
    span = times[-1] - times[0]
    steady = len(durations)
    return {
        "steady_steps": steady,
        "step_s_median": round(statistics.median(durations), 4),
        "step_s_mean": round(span / steady, 4),
        "steps_per_s": round(steady / span, 4),
        "tokens_per_s": round(sum(tokens[1:]) / span, 1),
        "loss_tokens_per_s": round(sum(loss_tokens[1:]) / span, 1),
    }


def main(argv=None):
    own, overrides = split_argv(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description="LoRA SFT with MaxText's train_sft, measured")
    parser.add_argument("--out", required=True, help="Local output directory")
    parser.add_argument("--steps", default="auto", help="'auto' (one epoch, counted) or a number")
    parser.add_argument("--maxtext-output", default=None,
                        help="MaxText's base_output_directory (default: OUT/maxtext). A multi-host job needs a shared "
                             "one (gs://): Orbax's final save waits for every process's commit in the same directory")
    parser.add_argument("--distributed", action="store_true",
                        help="One job over several hosts: jax.distributed.initialize() here, before anything touches "
                             "the TPU (pass skip_jax_distributed_system=True to MaxText)")
    args = parser.parse_args(own)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    started = time.time()
    import jax
    from absl import flags

    # train_sft's main runs under absl's app.run; grain reads absl flags, so they must count as parsed here too
    flags.FLAGS(sys.argv[:1])
    from absl import logging as absl_logging

    absl_logging.set_verbosity(absl_logging.INFO)  # MaxText's own messages (LoRA match, step metrics) go to the log

    if args.distributed:
        jax.distributed.initialize()
    import jax.numpy as jnp
    import maxtext
    import numpy as np
    from flax import nnx
    from maxtext.configs import pyconfig
    from maxtext.input_pipeline.input_pipeline_interface import create_data_iterator
    from maxtext.trainers.post_train.sft import train_sft
    from maxtext.utils import lora_utils, maxtext_utils

    sft_yml = Path(maxtext.__file__).parent / "configs" / "post_train" / "sft.yml"
    base = ["sft_lora.py", str(sft_yml), *overrides, f"base_output_directory={args.maxtext_output or out / 'maxtext'}",
            f"metrics_file={out}/metrics.txt"]
    config = pyconfig.initialize_pydantic(base + ["steps=1"])
    main_process = jax.process_index() == 0

    count = {"batches": 0, "tokens": 0, "loss_tokens": 0}
    if args.steps == "auto":
        mesh = maxtext_utils.get_mesh_from_config(config)
        iterator, _ = create_data_iterator(config, mesh)
        try:
            while True:
                batch = next(iterator)
                count["batches"] += 1
                count["tokens"] += int(jnp.sum(batch["inputs_segmentation"] != 0))
                count["loss_tokens"] += int(jnp.sum(batch["targets_segmentation"] != 0))
        except Exception as end:  # noqa: BLE001 - the pipeline's way to say the data ran out, checked below
            text = f"{type(end).__name__}: {end}"
            if not isinstance(end, StopIteration) and "Run out of shards" not in text:
                raise
            count["end"] = text[:300]
        del iterator
        steps = count["batches"]
        if jax.process_count() > 1:
            # Each process counts the batches of its own shard of the data; they can differ by a few, and a process
            # whose data run out first would stop while the others wait in a collective: all take the smallest count
            from jax.experimental import multihost_utils

            counts = multihost_utils.process_allgather(np.array([count["batches"]], np.int64))
            count["batches_per_process"] = [int(x) for x in np.asarray(counts).reshape(-1)]
            steps = min(count["batches_per_process"])
        if not steps:
            raise SystemExit("the training data gave no batch")
    else:
        steps = int(args.steps)
    print(f"EPOCH {json.dumps(count | {'steps': steps})}", flush=True)
    config = pyconfig.initialize_pydantic(base + [f"steps={steps}"])

    t_setup = time.time()
    trainer, mesh = train_sft.setup_trainer_state(config)
    setup_s = time.time() - t_setup

    hooks = trainer.training_hooks
    records, batch_counts = [], []
    marks = {}
    on_start, on_step_start, on_step_end = hooks.on_train_start, hooks.on_train_step_start, hooks.on_train_step_end

    def train_start(ctx):
        marks["train_start"] = time.perf_counter()
        return on_start(ctx)

    def step_start(ctx):
        batch = ctx.data_hooks.train_batch
        # Device scalars, read after training, so the steps stay asynchronous
        batch_counts.append((jnp.sum(batch["inputs_segmentation"] != 0), jnp.sum(batch["targets_segmentation"] != 0)))
        return on_step_start(ctx)

    def step_end(ctx, train_step, train_loss, step_time=0.0):
        records.append({"step": int(train_step), "loss": float(np.asarray(train_loss)), "t": time.perf_counter()})
        return on_step_end(ctx, train_step, train_loss, step_time)

    hooks.on_train_start, hooks.on_train_step_start, hooks.on_train_step_end = train_start, step_start, step_end

    t_train = time.time()
    train_sft.train_model(config, trainer, mesh)
    train_s = time.time() - t_train

    tokens = [int(a) for a, _ in batch_counts]
    loss_tokens = [int(b) for _, b in batch_counts]
    memory = []
    for device in jax.local_devices():
        stats = device.memory_stats() or {}
        memory.append({"device": str(device), "peak_gib": round(stats.get("peak_bytes_in_use", 0) / 2 ** 30, 2),
                       "limit_gib": round(stats.get("bytes_limit", 0) / 2 ** 30, 2)})

    flat, dtypes = adapter_arrays(nnx.to_pure_dict(nnx.state(trainer.model, nnx.LoRAParam)))
    first_compile = (records[0]["t"] - marks["train_start"]) if records else None
    later = [b["t"] - a["t"] for a, b in zip(records, records[1:])]
    lora = config.lora
    summary = {
        "model": config.model_name, "checkpoint": config.load_parameters_path,
        "train_files": config.hf_train_files, "chat_template_path": config.chat_template_path,
        "maxtext": maxtext.__version__ if hasattr(maxtext, "__version__") else None,
        "devices": jax.device_count(), "processes": jax.process_count(), "device_kind": jax.devices()[0].device_kind,
        "lora": {"rank": lora.lora_rank, "alpha": lora.lora_alpha,
                 "module_path": lora_utils._get_lora_module_path(config),  # noqa: SLF001 - the regex qwix used
                 "params": int(sum(v.size for v in flat.values())), "dtype": sorted(dtypes)},
        "config": {k: getattr(config, k) for k in (
            "per_device_batch_size", "global_batch_size_to_train_on", "gradient_accumulation_steps",
            "max_target_length", "packing", "learning_rate", "learning_rate_final_fraction", "warmup_steps_fraction",
            "opt_type", "adam_b1", "adam_b2", "adam_weight_decay", "gradient_clipping_threshold", "weight_dtype",
            "dtype", "remat_policy", "attention", "scan_layers", "ici_fsdp_parallelism", "ici_tensor_parallelism",
            "dcn_data_parallelism", "dcn_fsdp_parallelism", "sft_train_on_completion_only", "enable_data_shuffling")},
        "epoch": count, "steps": steps, "steps_done": len(records),
        "times": {"python_to_setup_s": round(t_setup - started, 1), "setup_s": round(setup_s, 1),
                  "train_s": round(train_s, 1),
                  "first_step_s": round(first_compile, 1) if first_compile is not None else None,
                  "compile_s_est": round(first_compile - statistics.median(later), 1) if later else None},
        "throughput": throughput(records, tokens, loss_tokens),
        "tokens": {"total": sum(tokens), "loss": sum(loss_tokens), "per_step": tokens, "loss_per_step": loss_tokens},
        "memory": memory, "peak_gib_max": max(m["peak_gib"] for m in memory),
        "loss": [[r["step"], round(r["loss"], 5)] for r in records],
    }
    if summary["throughput"]:
        summary["throughput"]["tokens_per_s_per_chip"] = round(summary["throughput"]["tokens_per_s"] / jax.device_count(), 1)
    if main_process:
        np.savez(out / "adapter.npz", **flat)
        adapter = {"rank": lora.lora_rank, "alpha": lora.lora_alpha, "scale": lora.lora_alpha / lora.lora_rank,
                   "module_path": summary["lora"]["module_path"], "model": config.model_name,
                   "checkpoint": config.load_parameters_path, "trained_dtype": sorted(dtypes),
                   "steps": len(records), "learning_rate": config.learning_rate, "train_files": config.hf_train_files}
        (out / "adapter.json").write_text(json.dumps(adapter, indent=1) + "\n")
        (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str) + "\n")
    short = {k: summary[k] for k in ("model", "devices", "steps", "steps_done", "times", "throughput", "peak_gib_max")}
    print(f"SUMMARY {json.dumps(short | {'loss_first_last': [summary['loss'][:3], summary['loss'][-3:]]}, default=str)}",
          flush=True)
    if jax.process_count() > 1:
        # Process 0 writes the adapter after the others are done; a process that exits first takes the coordination
        # service down under it (EXIT 134 on worker 0, 9 Nov)
        from jax.experimental import multihost_utils

        multihost_utils.sync_global_devices("sft_lora files written")


if __name__ == "__main__":
    # As translate.py: a thread that is no daemon would keep a failed process alive, holding the TPU
    import os
    import traceback

    code = 0
    try:
        main()
    except SystemExit as stop:
        code = stop.code if isinstance(stop.code, int) else (0 if stop.code is None else 1)
        if not isinstance(stop.code, int) and stop.code is not None:
            print(stop.code, file=sys.stderr)
    except BaseException:
        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
