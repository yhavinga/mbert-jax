#!/usr/bin/env python3
"""LoRA SFT of TranslateGemma with MaxText 0.2.5's train_sft (Tunix's PeftTrainer, qwix LoRA), with the measurements
of the smoke test, and the adapter exported for translate.py --lora.

    python sft_lora.py --out DIR -- <MaxText overrides key=value ...>      (run_sft.sh)

The config is MaxText's post_train/sft.yml plus the overrides. One epoch: the training data first go once through
MaxText's own pipeline (create_data_iterator on the training mesh) to count its batches, and `steps` becomes that count,
so the cosine schedule ends with the epoch; a multi-host job takes the smallest count of its processes minus a margin
(step_limit), and holds a given number of steps to that limit too. Then MaxText's setup_trainer_state and train_model,
unchanged, with two wrapped hooks that keep, per step, the loss, the time it ended and the batch's token counts.

Before the first step every process prints LORA_MODULES: the LoRA parameters per module type and the modules that the
LoRA regex matches but that got none (qwix adds LoRA only where it intercepts a matmul of a module's own weight).

With --eval NAME=FILE (repeatable) and --eval-every N, the loss on each held-out file (examples in the training data's
format) before the first step (LoRA's B is zero, so the base model), every N steps (Tunix's eval_every_n_steps) and
after the last: each file in full, token-weighted (MaxText's xent_sum over total_weights). Every eval makes a fresh
iterator (MaxText's HF source cannot restart), and every process runs the same number of batches, the largest count of
any process (counted before training): a process past its data gets MaxText's zero padding batch
(generate_padding_batch_eval), which weighs nothing. Each eval prints an EVAL line; summary.json keeps them under "eval".
Writes to DIR (process 0): summary.json (configuration, times, throughput, memory per chip, loss per step), adapter.npz
(the LoRA parameters in float32, keys as decoder/layers_0/self_attention/query/kernel_lora_a), adapter.json (rank,
alpha, module regex, model, checkpoint, original dtype); MaxText's output (metrics file, tensorboard, Tunix's checkpoint
of the LoRA parameters) under DIR/maxtext.
"""

import argparse
import json
import math
import re
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


def path_key(path):
    """A pytree path as "/"-joined names, as adapter_arrays makes the adapter's keys."""
    return "/".join(str(getattr(p, "key", getattr(p, "name", getattr(p, "idx", p)))) for p in path)


LORA_SUFFIXES = ("_lora_a", "_lora_b")
LAYER_INDEX = re.compile(r"(?<=layers_)[0-9]+")


def module_type(module_path):
    """A module's path with its layer index as *: decoder/layers_3/self_attention/query -> decoder/layers_*/..."""
    return LAYER_INDEX.sub("*", module_path)


def lora_module_type(key):
    """The module type of a LoRA parameter (adapter key, e.g. decoder/layers_3/self_attention/query/kernel_lora_a ->
    decoder/layers_*/self_attention/query). qwix names a LoRA pair after the weight it adapts; a weight other than a
    kernel (a MoE block's own expert weights wi_0, wi_1, wo) stays in the type."""
    module, _, name = key.rpartition("/")
    for suffix in LORA_SUFFIXES:
        if name.endswith(suffix):
            weight = name[: -len(suffix)]
            return module_type(module) if weight == "kernel" else f"{module_type(module)}/{weight}"
    raise ValueError(f"{key} is not a LoRA parameter (it ends in neither of {LORA_SUFFIXES})")


def from_decoder(path):
    """A module path from its first "decoder" component on, as the LoRA regex sees it (a wrapper's prefix dropped)."""
    parts = path.split("/")
    return "/".join(parts[parts.index("decoder"):]) if "decoder" in parts else path


def lora_report(regex, sizes, module_paths):
    """What LoRA reached: per module type the modules with LoRA and their parameters (sizes: {adapter key: number of
    values}), and per module type the modules whose path the regex fullmatches (as qwix matches it) that have none."""
    by_type, modules = {}, {}
    for key, size in sizes.items():
        kind = lora_module_type(key)
        entry = by_type.setdefault(kind, {"modules": 0, "params": 0})
        entry["params"] += int(size)
        modules.setdefault(kind, set()).add(key.rpartition("/")[0])
    for kind, paths in modules.items():
        by_type[kind]["modules"] = len(paths)
    with_lora = {from_decoder(key.rpartition("/")[0]) for key in sizes}
    pattern = re.compile(regex)
    without = {}
    for path in module_paths:
        path = from_decoder(path)
        if pattern.fullmatch(path) and path not in with_lora:
            kind = module_type(path)
            without[kind] = without.get(kind, 0) + 1
    return {"regex": regex, "params": int(sum(sizes.values())), "by_type": dict(sorted(by_type.items())),
            "matched_without_lora": dict(sorted(without.items()))}


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


def local_nonzero(array):
    """The non-zero entries of this process's part of a (global) array, read from its addressable shards on the host.
    A jnp operation on a global array is one program over all processes; in the epoch count the processes see different
    numbers of batches, so such programs run out of step (10 Nov: "an unexpected peer shows up in the launch group").
    A shard held twice (a replicated axis) counts once."""
    import numpy as np

    seen, total = set(), 0
    for shard in array.addressable_shards:
        key = tuple((s.start, s.stop, s.step) for s in shard.index)
        if key not in seen:
            seen.add(key)
            total += int(np.count_nonzero(np.asarray(shard.data)))
    return total


# A multi-host job stops this many batches before the smallest count of its processes. Tunix's PeftTrainer.train loads
# the next batch before it checks max_steps, so at steps == the smallest count the process with that count gets
# StopIteration while the others load (and device_put over the global mesh) one more batch: the processes leave the
# loop by different paths and their TPU programs run out of step (12B over 16 chips, 10 Nov: 431 steps, process 0
# "program continuator has halted unexpectedly", the others "Failed to cancel preemption barrier"). One batch would
# do; the margin also covers a count that comes out a little different in the training pass.
STEP_MARGIN_MIN = 8
STEP_MARGIN_FRACTION = 0.01


def step_limit(batches_per_process):
    """The most steps a multi-host job may take: the smallest batch count of its processes minus the margin, so every
    process still has a batch left when the loop ends at max_steps."""
    smallest = min(batches_per_process)
    return smallest - max(STEP_MARGIN_MIN, math.ceil(STEP_MARGIN_FRACTION * smallest))


def choose_steps(requested, batches_per_process):
    """The job's step count: 'auto' is one epoch (a single process: all its batches; several: step_limit); a number
    must stay within step_limit when several processes share the job. batches_per_process is None where nothing was
    counted (a single process with a number)."""
    if batches_per_process is None:
        return int(requested)
    if not min(batches_per_process):
        raise SystemExit(f"a process got no batch from the training data: {batches_per_process}")
    if len(batches_per_process) == 1:
        return batches_per_process[0] if requested == "auto" else int(requested)
    limit = step_limit(batches_per_process)
    if limit < 1:
        raise SystemExit(f"too few batches for a multi-host job: {batches_per_process}, limit {limit}")
    if requested == "auto":
        return limit
    if int(requested) > limit:
        raise SystemExit(f"steps={requested} exceeds the multi-host limit {limit} (batches per process "
                         f"{batches_per_process}): a process would run out of data before the others")
    return int(requested)


def parse_evals(specs):
    """{name: file} of --eval NAME=FILE, in the given order; names unique"""
    out = {}
    for spec in specs or []:
        name, sep, path = spec.partition("=")
        if not sep or not name or not path:
            raise SystemExit(f"--eval {spec!r}: NAME=FILE")
        if name in out:
            raise SystemExit(f"--eval {name} twice")
        out[name] = path
    return out


def eval_overrides(evals, every):
    """MaxText's keys for Tunix's eval schedule; the data hooks build an eval iterator of the first file, which this
    script replaces by its own per file"""
    if not evals:
        return []
    if every < 1:
        raise SystemExit("--eval needs --eval-every >= 1")
    return [f"eval_interval={every}", f"hf_eval_files={next(iter(evals.values()))}", "hf_eval_split=train",
            "eval_steps=-1"]


def eval_loss(name, step, parts, batches, seconds):
    """One file's eval at a step from the (xent_sum, total_weights) of its batches"""
    if len(parts) != batches:
        raise SystemExit(f"eval {name} at step {step}: {len(parts)} batches ran, {batches} planned")
    xent = sum(x for x, _ in parts)
    weights = sum(w for _, w in parts)
    if weights <= 0:
        raise SystemExit(f"eval {name} at step {step}: no target token")
    return {"step": step, "set": name, "batches": batches, "tokens": int(round(weights)), "loss": xent / weights,
            "seconds": round(seconds, 1)}


def needs_final_eval(evals, steps):
    """Tunix evaluates at step 0 and every N steps; the last step only when N divides it"""
    return bool(evals) and evals[-1]["step"] != steps


def lr_points(steps, warmup_fraction, decay_fraction):
    """The steps whose learning rate summary.json records: start, end of warmup, middle, start of the decay, last"""
    warmup = int(steps * warmup_fraction)
    decay = int(steps * decay_fraction)
    return sorted({0, 1, warmup, warmup + 1, steps // 2, max(0, steps - decay), steps - 1} & set(range(steps)))


def learning_rates(config, steps):
    """{step: the learning rate of MaxText's schedule} at lr_points (optax counts updates from 0)"""
    from maxtext.utils import maxtext_utils

    schedule = maxtext_utils.create_learning_rate_schedule(config)
    decay = config.wsd_decay_steps_fraction if str(config.lr_schedule_type).lower().endswith("wsd") else 0.0
    return {str(k): float(schedule(k)) for k in lr_points(steps, config.warmup_steps_fraction, decay)}


def scalar(x):
    """A replicated device scalar on the host (this process's copy)"""
    import numpy as np

    value = x.addressable_data(0) if hasattr(x, "addressable_data") else x
    return float(np.asarray(value))


def count_batches(iterator):
    """(batches, target tokens of this process, how it ended) of a MaxText iterator without padding batches"""
    n, loss_tokens = 0, 0
    try:
        while True:
            batch = next(iterator)
            n += 1
            loss_tokens += local_nonzero(batch["targets_segmentation"])
    except Exception as end:  # noqa: BLE001 - the pipeline's way to say the data ran out, checked below
        text = f"{type(end).__name__}: {end}"
        if not isinstance(end, StopIteration) and "Run out of shards" not in text:
            raise
        return n, loss_tokens, text[:300]


def eval_iterator(config, mesh):
    """MaxText's HF eval iterator of config.hf_eval_files, as create_data_iterator makes it, without a train iterator"""
    from maxtext.input_pipeline import input_pipeline_interface as ipi
    from maxtext.input_pipeline.hf_data_processing import make_hf_eval_iterator

    indices = ipi.get_process_loading_real_data(config.data_sharding, config.global_batch_size_to_load_eval,
                                                config.global_batch_size_to_eval_on, config.max_target_length, mesh)
    return ipi.create_process_specific_iterator(config, mesh, indices, make_hf_eval_iterator, is_training=False)


def close(iterator):
    if iterator is not None and hasattr(iterator, "close"):
        iterator.close()


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
    parser.add_argument("--steps", default="auto",
                        help="'auto' (one epoch, counted; multi-host: step_limit) or a number (multi-host: at most "
                             "step_limit)")
    parser.add_argument("--maxtext-output", default=None,
                        help="MaxText's base_output_directory (default: OUT/maxtext). A multi-host job needs a shared "
                             "one (gs://): Orbax's final save waits for every process's commit in the same directory")
    parser.add_argument("--distributed", action="store_true",
                        help="One job over several hosts: jax.distributed.initialize() here, before anything touches "
                             "the TPU (pass skip_jax_distributed_system=True to MaxText)")
    parser.add_argument("--eval", action="append", default=[],
                        help="NAME=FILE: a held-out file in the training data's format, its loss at step 0, every "
                             "--eval-every steps and after the last (repeatable)")
    parser.add_argument("--eval-every", type=int, default=0)
    args = parser.parse_args(own)
    evals_wanted = parse_evals(args.eval)
    overrides = overrides + eval_overrides(evals_wanted, args.eval_every)
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
    batches_per_process = None
    # A multi-host job counts also when given a number of steps, to hold that number to step_limit
    if args.steps == "auto" or jax.process_count() > 1:
        mesh = maxtext_utils.get_mesh_from_config(config)
        iterator, unused_eval = create_data_iterator(config, mesh)
        close(unused_eval)
        try:
            while True:
                batch = next(iterator)
                count["batches"] += 1
                count["tokens"] += local_nonzero(batch["inputs_segmentation"])
                count["loss_tokens"] += local_nonzero(batch["targets_segmentation"])
        except Exception as end:  # noqa: BLE001 - the pipeline's way to say the data ran out, checked below
            text = f"{type(end).__name__}: {end}"
            if not isinstance(end, StopIteration) and "Run out of shards" not in text:
                raise
            count["end"] = text[:300]
        del iterator
        batches_per_process = [count["batches"]]
        if jax.process_count() > 1:
            # Each process counts the batches of its own shard of the data (tokens: its own part); they can differ by
            # a few, and a process whose data run out first would stop while the others wait in a collective: all take
            # step_limit of the counts, gathered in one collective once every process is done counting
            from jax.experimental import multihost_utils

            counts = multihost_utils.process_allgather(np.array([count["batches"]], np.int64))
            batches_per_process = [int(x) for x in np.asarray(counts).reshape(-1)]
            count["batches_per_process"] = batches_per_process
            count["step_limit"] = step_limit(batches_per_process)
    steps = choose_steps(args.steps, batches_per_process)
    print(f"EPOCH {json.dumps(count | {'requested': args.steps, 'steps': steps})}", flush=True)
    config = pyconfig.initialize_pydantic(base + [f"steps={steps}"])
    # before the training, so that a schedule this cannot read stops the run before it costs anything
    lr_at = learning_rates(config, steps)
    print(f"LR {json.dumps({'type': str(config.lr_schedule_type), 'at': lr_at})}", flush=True)

    # The held-out files: per file a config of its own; the batches every process runs, the largest count of any
    # process (counted without padding batches, so the count ends with the data)
    eval_plan = {}
    count_mesh = maxtext_utils.get_mesh_from_config(config) if evals_wanted else None
    for name, path in evals_wanted.items():
        it = eval_iterator(pyconfig.initialize_pydantic(base + [f"steps={steps}", f"hf_eval_files={path}",
                                                                "generate_padding_batch_eval=false"]), count_mesh)
        n, loss_tokens, end = count_batches(it)
        close(it)
        counts = [n]
        if jax.process_count() > 1:
            from jax.experimental import multihost_utils

            counts = [int(x) for x in np.asarray(multihost_utils.process_allgather(np.array([n], np.int64))).reshape(-1)]
        if not min(counts):
            raise SystemExit(f"eval {name}: a process got no batch of {path}: {counts}")
        eval_plan[name] = {"path": path, "batches": max(counts), "batches_per_process": counts,
                           "loss_tokens_here": loss_tokens, "end": end,
                           # MaxText requires eval_steps > 0 with padding batches; this script runs that many itself
                           "config": pyconfig.initialize_pydantic(base + [f"steps={steps}", f"hf_eval_files={path}",
                                                                          "generate_padding_batch_eval=true",
                                                                          f"eval_steps={max(counts)}"])}
        print(f"EVAL_PLAN {json.dumps({k: v for k, v in eval_plan[name].items() if k != 'config'} | {'set': name})}",
              flush=True)

    t_setup = time.time()
    trainer, mesh = train_sft.setup_trainer_state(config)
    setup_s = time.time() - t_setup
    # Shapes only (no transfer): what the LoRA regex reached, before the first step, so a run that fails in compilation
    # still shows it
    lora_sizes = {path_key(path): math.prod(leaf.shape) for path, leaf in jax.tree_util.tree_flatten_with_path(
        nnx.to_pure_dict(nnx.state(trainer.model, nnx.LoRAParam)))[0]}
    module_paths = ["/".join(str(p) for p in path) for path, value in nnx.iter_graph(trainer.model)
                    if path and isinstance(value, nnx.Module)]
    lora_seen = lora_report(lora_utils._get_lora_module_path(config), lora_sizes, module_paths)  # noqa: SLF001
    print(f"LORA_MODULES {json.dumps(lora_seen)}", flush=True)

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

    evals = []
    if eval_plan:
        # Tunix's eval loop, file by file: our batches (data_hooks.load_next_eval_batch), the planned number of them
        # from a fresh iterator; the jitted eval step wrapped to keep each batch's xent_sum and total_weights
        data_hooks = trainer.data_hooks
        close(data_hooks.eval_data_iterator)          # the hooks' own iterator of the first file, never read
        data_hooks.eval_data_iterator = [None]        # truthy for Tunix's `if eval_ds`; iterated, never read
        feed = {"iter": None, "left": 0, "parts": [], "eval_step": None}

        def load_next_eval_batch(ctx):  # noqa: ARG001 - Tunix's signature
            if feed["left"] <= 0:
                data_hooks.eval_batch = None
                return None
            feed["left"] -= 1
            data_hooks.eval_batch = next(feed["iter"])
            return data_hooks.eval_batch

        data_hooks.load_next_eval_batch = load_next_eval_batch
        jit_both = trainer.jit_train_and_eval_step

        def jit_recorded(*a, **k):
            train_step, eval_step = jit_both(*a, **k)

            def eval_recorded(*args_, **kwargs_):
                loss, aux = eval_step(*args_, **kwargs_)
                feed["parts"].append((aux["xent_sum"], aux["total_weights"]))
                return loss, aux

            feed["eval_step"] = eval_recorded
            return train_step, eval_recorded

        trainer.jit_train_and_eval_step = jit_recorded
        run_eval_tunix = trainer._run_eval  # noqa: SLF001 - wrapped, not replaced

        def run_eval(eval_ds, eval_step_fn):
            step = trainer.train_steps
            for name, plan in eval_plan.items():
                feed["iter"] = eval_iterator(plan["config"], mesh)
                feed["left"], feed["parts"] = plan["batches"], []
                t0 = time.perf_counter()
                run_eval_tunix(eval_ds, eval_step_fn)
                close(feed["iter"])
                if feed["left"]:
                    raise SystemExit(f"eval {name} at step {step}: {feed['left']} planned batches not run")
                record = eval_loss(name, step, [(scalar(x), scalar(w)) for x, w in feed["parts"]], plan["batches"],
                                   time.perf_counter() - t0)
                evals.append(record)
                print(f"EVAL {json.dumps(record)}", flush=True)

        trainer._run_eval = run_eval  # noqa: SLF001
        on_end = hooks.on_train_end

        def train_end(ctx):
            # after the loop, before MaxText's hook flushes its metric logger and Tunix closes the trainer
            if needs_final_eval(evals, ctx.train_steps):
                run_eval(data_hooks.eval_data_iterator, feed["eval_step"])
            return on_end(ctx)

        hooks.on_train_end = train_end

    t_train = time.time()
    train_sft.train_model(config, trainer, mesh)
    train_s = time.time() - t_train
    if trainer.train_steps != steps or len(records) != steps:
        # the loop ended early (the data ran out) or a step went unlogged: not the run that was set up
        raise SystemExit(f"trained {trainer.train_steps} steps ({len(records)} logged), not the {steps} set")

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
                 "params": int(sum(v.size for v in flat.values())), "dtype": sorted(dtypes),
                 "by_module": lora_seen["by_type"], "matched_without_lora": lora_seen["matched_without_lora"]},
        # the MoE's settings (a dense model: num_experts 1); getattr with a default, so that a key another MaxText
        # lacks costs no finished run its summary
        "moe": {k: getattr(config, k, None) for k in (
            "num_experts", "num_experts_per_tok", "sparse_matmul", "megablox", "capacity_factor",
            "ici_expert_parallelism", "load_balance_loss_weight")},
        "config": {k: getattr(config, k) for k in (
            "per_device_batch_size", "global_batch_size_to_train_on", "gradient_accumulation_steps",
            "max_target_length", "packing", "learning_rate", "learning_rate_final_fraction", "warmup_steps_fraction",
            "lr_schedule_type", "wsd_decay_steps_fraction", "wsd_decay_style", "eval_interval", "opt_type", "adam_b1",
            "adam_b2", "adam_weight_decay", "gradient_clipping_threshold", "weight_dtype",
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
        "eval": evals,
        "eval_plan": {k: {x: v[x] for x in ("path", "batches", "batches_per_process", "end")}
                      for k, v in eval_plan.items()},
        "lr_at": lr_at,
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
