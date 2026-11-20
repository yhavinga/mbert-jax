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
after the last: each file (see below for the batches), token-weighted (MaxText's xent_sum over total_weights). Every eval makes a fresh
iterator (MaxText's HF source cannot restart), and every process runs the same number of batches, the smallest count
of any process (counted before training), without padding batches; the last batches of the larger shares are left out,
the same ones at every eval. Each eval prints an EVAL line; summary.json keeps them under "eval".
With --init-adapter FILE (an adapter.npz of this script) the LoRA parameters are that adapter's before the first step,
each in the model's own dtype and sharding, every key and shape checked both ways and read back: the eval at step 0 is
then that adapter's model (protocol-g4.md amendment G4-4.11: G4-3's final model on G4-4's novels).
Arm G4-5 (draft amendment G4-5): with --resume-step N the run continues the training state that Tunix restores from
its own checkpoint directory (MaxText's checkpoint_dir, where the chain copied the warmup's checkpoint of step N): the
LoRA parameters and the optimizer state (AdamW's moments and step count); this script checks that Tunix restored step
N, that the LoRA parameters equal --resume-adapter (the warmup's adapter.npz) and that the optimizer's step counters
are N. --steps then counts this run's own steps (its data from the first batch: MaxText's data hooks skip nothing);
MaxText's `steps` and the schedule run over N + own steps, so the learning rate continues at step N. With
--warmup-steps W and --decay-fraction F the WSD schedule has exactly W warmup steps and int(F x own) decay steps at
the end (the fractions of MaxText's schedule computed for N + own). With --combine SPEC (g4mix.py spec) the first
eval runs, after its usual round, the search over the weights of a shared warmup and its branches (KernelMerger: the
model's base kernels replaced by W0 + the stacked update, one point at a time, forward only), the top K on every eval
file, and puts the base kernels back; summary.json keeps it under "combine". Addendum G4-5.7b: the repeated first point
is equal within the spec's repeat_max_diff; with the spec's warmup_check the merged warmup alone (w = 0) is evaluated
first on the check's sets and must give the warmup's own eval (same target tokens, within max_diff), else the run stops
before any other point.
With --temperatures GRID (addendum G4-5.7b) the run is eval only: the first eval (before any step) also gives, per
file and per temperature T of the grid, the token-weighted loss of softmax(logits / T) from the same forward pass
(MaxText's loss on the full logits: num_vocab_tiling=1), and the run stops after that round (and, with --combine, after
the search) without a training step; summary.json keeps it per eval under "temperatures". With --combine every point
then has both objectives, at T = 1 and at its best T (calibrated); the spec's objective_mode says which one the search
follows (addendum G4-5.7b A).
Arm G4-6 (draft amendment G4-6): with --dpo SPEC (g4dpo.spec) the run is preference training: the training data are
--pref-train, a packed file of g4dpo.py (pairs, chosen and rejected side by side in a row; one epoch is every batch of
the file, the same on every process, so nothing is counted through MaxText's pipeline), and the loss is g4dpo.make_loss
for training and eval. The reference (the model with its LoRA at zero) is computed once before the first step
(amendment G4-6.4a): a forward pass of the model while every LoRA B is zero over the batches that the run trains on and
every --pref-eval file, its per-segment sums carried by each batch; the training step runs only the policy. MaxText's
own data hooks are still made from the config's hf_train_files (any SFT file; never read). --pref-eval NAME=FILE adds a
packed eval file (its pairs' loss, accuracy, margin and rewards, and the NLL of both sides, per eval); the --eval files
keep their token-weighted loss, as before. summary.json keeps the per-step preference measures under "dpo" (with the
reference pass under "ref"); DIR/dpo_ref.parquet the reference's sums per pair.
Arm G4-6b (amendment G4-6b): with --dpo and --base-adapter NAME=FILE (an adapter.npz of this script, its adapter.json
beside it) that adapter is merged into the base kernels before the first step and before any eval (KernelMerger, as
G4-5's search merged a point: W0 + scale * A B in float32, then the kernel's own dtype and sharding; the model casts a
kernel to bf16 at use, so the model is the one translate.py --lora decodes with that adapter); the run's own LoRA
starts at zero on that model, and the reference (the LoRA at zero) is that model. summary.json keeps it under
"base_adapter".
Arm G4-6c (amendment G4-6c): a spec with end_tokens = n leaves the last n target tokens of every side out of the
log-ratio (the reference pass too); before JAX each packed file is checked to hold every segment's targets as one run of
more than n tokens (ENDS). --save-steps K1,K2,... keeps the LoRA parameters after those steps as
DIR/adapter-step<K>.npz and .json (process 0; the same format as adapter.npz), and a line SAVED per step. Each K is an
eval step (a multiple of --eval-every): the adapter is kept in that eval round, so it is the model of that eval (Tunix
calls on_train_step_end for step K only after step K + 1, addendum G4-6c.6a).
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


def load_adapter(model, path):
    """Puts adapter.npz (adapter_arrays' keys, float32) into the model's LoRA parameters, each cast to the parameter's
    dtype and placed with its sharding; every parameter needs a key of its shape and every key a parameter. Read back
    through adapter_arrays and compared, so that a load that did not reach the model stops the run."""
    import hashlib

    import jax
    import numpy as np
    from flax import nnx

    state = nnx.state(model, nnx.LoRAParam)
    leaves, treedef = jax.tree_util.tree_flatten_with_path(nnx.to_pure_dict(state))
    with np.load(path) as z:
        arrays = {k: z[k] for k in z.files}
    new, want = [], {}
    for p, leaf in leaves:
        key = path_key(p)
        if key not in arrays:
            raise SystemExit(f"--init-adapter {path}: no {key}")
        if tuple(arrays[key].shape) != tuple(leaf.shape):
            raise SystemExit(f"--init-adapter {path}: {key} is {arrays[key].shape}, the model's {leaf.shape}")
        a = arrays[key].astype(leaf.dtype)
        want[key] = np.asarray(a).astype(np.float32)
        new.append(jax.make_array_from_callback(leaf.shape, leaf.sharding, lambda index, a=a: a[index]))
    extra = sorted(set(arrays) - set(want))
    if extra:
        raise SystemExit(f"--init-adapter {path}: {len(extra)} keys the model has no LoRA parameter for, e.g. {extra[:3]}")
    nnx.replace_by_pure_dict(state, jax.tree_util.tree_unflatten(treedef, new))
    nnx.update(model, state)
    back, dtypes = adapter_arrays(nnx.to_pure_dict(nnx.state(model, nnx.LoRAParam)))
    differ = [k for k in want if not np.array_equal(back[k], want[k])]
    if set(back) != set(want) or differ:
        raise SystemExit(f"--init-adapter {path}: read back, {len(differ)} parameters differ, e.g. {differ[:3]}")
    with open(path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    return {"path": str(path), "sha256": digest, "keys": len(want), "params": int(sum(a.size for a in want.values())),
            "dtype": sorted(dtypes), "nonzero_b": sum(bool(np.any(want[k])) for k in want if k.endswith("_lora_b"))}


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


def base_adapter_arg(spec):
    """--base-adapter NAME=FILE (arm G4-6b), checked before JAX: the adapter.npz and its adapter.json (rank, alpha, scale
    = alpha / rank) -> {"name", "path", "meta"}"""
    name, sep, path = spec.partition("=")
    if not sep or not name or not path:
        raise SystemExit(f"--base-adapter {spec}: NAME=FILE")
    npz = Path(path)
    meta_path = npz.parent / "adapter.json"
    if not npz.is_file() or not meta_path.is_file():
        raise SystemExit(f"--base-adapter {spec}: no {npz} or no {meta_path}")
    meta = json.loads(meta_path.read_text())
    if any(k not in meta for k in ("rank", "alpha", "scale")) or abs(meta["alpha"] / meta["rank"] - meta["scale"]) > 1e-12:
        raise SystemExit(f"--base-adapter {spec}: {meta_path} has no rank, alpha and scale = alpha / rank")
    return {"name": name, "path": str(npz), "meta": meta}


def bake_adapter(model, mesh, base):
    """--base-adapter: the adapter merged into the base kernels with KernelMerger (every adapter module has exactly one
    base kernel, else SystemExit). Nothing restores the base, so (addendum G4-6c.6d) each kernel's merge takes the
    kernel itself and donates its buffer: the kernels from before the merge leave the device even where the trainer
    still holds them (G4-6b's peak 29.13 GiB against G4-6's 25.18), and no host copy of the base is made."""
    import numpy as np

    t = time.perf_counter()
    with np.load(base["path"]) as z:
        arrays = {k: z[k] for k in z.files}
    merger = KernelMerger(model, mesh, [arrays], base["meta"]["scale"], keep_base=False)
    before = device_bytes()
    merger.bake()
    after = device_bytes()
    modules = len(merger.modules)
    return {"name": base["name"], "path": base["path"], "rank": base["meta"]["rank"], "alpha": base["meta"]["alpha"],
            "scale": base["meta"]["scale"], "modules": modules, "keys": len(arrays),
            "seconds": round(time.perf_counter() - t, 1), "device_gib_before": before, "device_gib_after": after}


def device_bytes():
    """The largest bytes in use of this process's devices now, GiB (None where the backend does not say)"""
    import jax

    used = [(d.memory_stats() or {}).get("bytes_in_use") for d in jax.local_devices()]
    return round(max(used) / 2 ** 30, 2) if all(u is not None for u in used) and used else None


def dpo_args(args, evals_wanted):
    """--dpo's spec and its preference eval files {name: file}, checked before JAX: a measurement of its own (no
    --combine, --temperatures, --resume-step, --init-adapter), names apart from the --eval files"""
    if not args.dpo:
        if args.pref_train or args.pref_eval or args.base_adapter:
            raise SystemExit("--pref-train, --pref-eval and --base-adapter belong to --dpo")
        return None, {}
    import g4dpo

    if not args.pref_train:
        raise SystemExit("--dpo needs --pref-train (a packed file of g4dpo.py)")
    if args.combine or args.temperatures or args.resume_step or args.init_adapter:
        raise SystemExit("--dpo: no --combine, --temperatures, --resume-step or --init-adapter")
    try:
        opts = g4dpo.read_spec(args.dpo)
    except ValueError as bad:
        raise SystemExit(f"--dpo {args.dpo}: {bad}") from None
    prefs = parse_evals(args.pref_eval)
    both = sorted(set(prefs) & set(evals_wanted))
    if both:
        raise SystemExit(f"--pref-eval and --eval both name {both}")
    if "train" in prefs:
        raise SystemExit("--pref-eval: the name train is the training file's (its precomputed reference)")
    if opts["end_tokens"]:
        # amendment G4-6c: end_mask takes the last end_tokens targets of a segment only if they are one run
        for name, path in [("train", args.pref_train), *prefs.items()]:
            print(f"ENDS {json.dumps({'set': name, 'end_tokens': opts['end_tokens'], **g4dpo.check_ends(path, opts['end_tokens'])})}",
                  flush=True)
    return opts, prefs


def save_steps_arg(text, steps, eval_every=0):
    """--save-steps K1,K2,...: distinct steps in 1..steps-1 (the last step's adapter is adapter.npz), sorted, each an eval
    step (a multiple of --eval-every: the adapter is kept in that step's eval round)"""
    if not text:
        return []
    try:
        out = sorted({int(x) for x in text.split(",")})
    except ValueError:
        raise SystemExit(f"--save-steps {text}: integers separated by commas") from None
    if not out or out[0] < 1 or out[-1] >= steps:
        raise SystemExit(f"--save-steps {text}: every step in 1..{steps - 1} (step {steps} is adapter.npz)")
    if eval_every <= 0 or any(k % eval_every for k in out):
        raise SystemExit(f"--save-steps {text}: every step an eval step (a multiple of --eval-every {eval_every})")
    return out


def peak_gib():
    """The largest peak of this process's devices so far, GiB"""
    import jax

    return round(max((d.memory_stats() or {}).get("peak_bytes_in_use", 0) for d in jax.local_devices()) / 2 ** 30, 2)


def needs_final_eval(evals, steps):
    """Tunix evaluates at step 0 and every N steps; the last step only when N divides it"""
    return bool(evals) and evals[-1]["step"] != steps


class EvalOnlyDone(Exception):
    """--temperatures: the first eval round is done; no training step follows"""


def temperature_grid(text):
    """--temperatures LOW:HIGH:STEP (both ends included) or T1,T2,...: ascending, distinct, all > 0, with 1.0 among
    them (the training stack's own loss, the check of the measurement)"""
    if ":" in text:
        parts = text.split(":")
        if len(parts) != 3:
            raise SystemExit(f"--temperatures {text!r}: LOW:HIGH:STEP or T1,T2,...")
        low, high, step = (float(x) for x in parts)
        if not 0 < low <= high or step <= 0:
            raise SystemExit(f"--temperatures {text!r}: 0 < LOW <= HIGH, STEP > 0")
        n = int(round((high - low) / step))
        if abs(low + n * step - high) > 1e-9:
            raise SystemExit(f"--temperatures {text!r}: HIGH is not LOW + k x STEP")
        temps = [round(low + k * step, 6) for k in range(n + 1)]
    else:
        temps = sorted({float(x) for x in text.split(",")})
    if not temps or temps[0] <= 0 or 1.0 not in temps:
        raise SystemExit(f"--temperatures {text!r}: every T > 0, and 1.0 among them")
    return temps


def temperature_losses(logits, one_hot, temps):
    """Per temperature T the per-token cross entropy of softmax(logits / T), logsumexp(z / T) - z_target / T, in
    float32, shape [len(temps), *logits.shape[:-1]]; one T at a time (lax.map), so one scaled copy of the logits at most
    is live. T changes no argmax: the greedy output is the same at every T."""
    import jax
    import jax.numpy as jnp

    z = logits.astype(jnp.float32)
    target = jnp.sum(one_hot * z, axis=-1)
    t = jnp.asarray(temps, jnp.float32)
    lse = jax.lax.map(lambda x: jax.scipy.special.logsumexp(z / x, axis=-1), t)
    return lse - target[None] / t.reshape((-1,) + (1,) * target.ndim)


def with_temperatures(eval_loss_fn, temps, micro_batch, module):
    """MaxText's eval loss with aux["xent_temps"]: per temperature the cross-entropy sum over the target tokens
    (targets_segmentation != 0, as MaxText's xent_sum) from the same forward pass. MaxText's loss_fn hands the full
    logits (num_vocab_tiling 1) to module.cross_entropy_with_logits; that function is wrapped so that, while this loss
    is traced, it gives the logits to temperature_losses too (under jit the Python here runs once, at tracing)."""
    import jax.numpy as jnp

    original = module.cross_entropy_with_logits
    seen = {"on": False}

    def cross_entropy(logits, targets, z_loss=0.0):
        if seen["on"]:
            if "per_token" in seen:
                raise SystemExit("--temperatures: two cross entropies in one eval loss")
            seen["per_token"] = temperature_losses(logits, targets, temps)
        return original(logits, targets, z_loss=z_loss)

    module.cross_entropy_with_logits = cross_entropy

    def loss_fn(model, **inputs):
        seen.pop("per_token", None)
        seen["on"] = True
        try:
            loss, aux = eval_loss_fn(model, **inputs)
        finally:
            seen["on"] = False
        if "per_token" not in seen:
            raise SystemExit("--temperatures: MaxText's eval loss did not call cross_entropy_with_logits "
                             "(num_vocab_tiling must be 1)")
        per_token = seen.pop("per_token")
        mask = (inputs["targets_segmentation"][:micro_batch] != 0).astype(jnp.float32)
        return loss, {**aux, "xent_temps": jnp.sum(per_token * mask[None], axis=tuple(range(1, per_token.ndim)))}

    return loss_fn


def temperature_record(temps, sums, weights, batches):
    """{"t": temps, "loss": per T the file's cross entropy over its target tokens} from the per-batch sums"""
    if len(sums) != batches or any(len(x) != len(temps) for x in sums) or weights <= 0:
        raise SystemExit(f"--temperatures: {len(sums)} batches of {[len(x) for x in sums[:3]]} sums for "
                         f"{len(temps)} temperatures, {batches} planned, {weights} tokens")
    return {"t": list(temps), "loss": [sum(x[k] for x in sums) / weights for k in range(len(temps))]}


def lr_points(steps, warmup_fraction, decay_fraction, extra=()):
    """The steps whose learning rate summary.json records: start, end of warmup, middle, start of the decay, last (and
    `extra`, e.g. a resumed run's first steps)"""
    warmup = int(steps * warmup_fraction)
    decay = int(steps * decay_fraction)
    points = {0, 1, warmup, warmup + 1, steps // 2, max(0, steps - decay), steps - 1}
    return sorted((points | set(extra)) & set(range(steps)))


def learning_rates(config, steps, extra=()):
    """{step: the learning rate of MaxText's schedule} at lr_points (optax counts updates from 0)"""
    from maxtext.utils import maxtext_utils

    schedule = maxtext_utils.create_learning_rate_schedule(config)
    decay = config.wsd_decay_steps_fraction if str(config.lr_schedule_type).lower().endswith("wsd") else 0.0
    return {str(k): float(schedule(k)) for k in lr_points(steps, config.warmup_steps_fraction, decay, extra)}


def schedule_overrides(total, warmup, decay):
    """MaxText's WSD fractions that give exactly `warmup` linear warmup steps and `decay` decay steps in a schedule of
    `total` steps (MaxText takes int(total * fraction)); the schedule's length is `steps`"""
    if not (0 <= warmup and 0 <= decay and warmup + decay <= total):
        raise SystemExit(f"warmup {warmup} and decay {decay} steps do not fit a schedule of {total}")
    fw = (warmup + 0.5) / total
    fd = (decay + 0.5) / total if decay else 0.0
    if int(total * fw) != warmup or int(total * fd) != decay or fw + fd > 1.0:
        raise SystemExit(f"no fractions for {warmup} / {decay} of {total} steps ({fw!r}, {fd!r})")
    return [f"warmup_steps_fraction={fw!r}", f"wsd_decay_steps_fraction={fd!r}", "learning_rate_schedule_steps=-1"]


def resumed_counters(opt_pure):
    """{key: value} of the optimizer state's step counters (leaves named step or count) of a pure dict"""
    import jax
    import numpy as np

    out = {}
    for path, leaf in jax.tree_util.tree_flatten_with_path(opt_pure)[0]:
        key = path_key(path)
        if key.rpartition("/")[2] in ("step", "count"):
            value = leaf.addressable_data(0) if hasattr(leaf, "addressable_data") else leaf
            out[key] = int(np.asarray(value))
    return out


def check_resume(trainer, step, adapter_path):
    """--resume-step: Tunix restored step `step` from the run's checkpoint directory, the LoRA parameters are those of
    adapter_path (the warmup's adapter.npz), and every step counter of the optimizer state is `step`"""
    import numpy as np
    from flax import nnx

    if trainer.train_steps != step:
        where = getattr(getattr(trainer, "config", None), "checkpoint_root_directory", "the run's checkpoint directory")
        raise SystemExit(f"--resume-step {step}: Tunix starts at step {trainer.train_steps} (no checkpoint of step "
                         f"{step} in {where}?)")
    flat, _ = adapter_arrays(nnx.to_pure_dict(nnx.state(trainer.model, nnx.LoRAParam)))
    with np.load(adapter_path) as z:
        want = {k: z[k] for k in z.files}
    differ = sorted(k for k in want if k not in flat or not np.array_equal(flat[k], want[k]))
    if set(flat) != set(want) or differ:
        raise SystemExit(f"--resume-adapter {adapter_path}: the restored LoRA parameters differ ({len(differ)} keys, "
                         f"e.g. {differ[:3]}; keys only in one: {sorted(set(flat) ^ set(want))[:3]})")
    opt = nnx.to_pure_dict(nnx.state(trainer.optimizer, nnx.optimizer.OptState))
    counters = resumed_counters(opt)
    if not counters or any(v != step for v in counters.values()):
        raise SystemExit(f"--resume-step {step}: the optimizer's step counters are {counters or 'not found'}")
    return {"step": step, "adapter": str(adapter_path), "lora_keys": len(want), "counters": counters}


class KernelMerger:
    """sft_lora.py --combine: the base kernels of the adapters' modules replaced by W0 + scale * sum_k c_k A_k B_k
    (g4mix.stack_module, translate.lora_delta in float32, then the kernel's own dtype and sharding), one module at a
    time; W0 from a host copy of this process's own shards, so that every point starts from the base exactly and
    restore() puts it back. The kernels keep their shapes, dtypes and shardings: the jitted eval step is not compiled
    again. The model casts a kernel to its compute dtype (bf16) at use, so a point is the model translate.py --lora
    decodes with the stacked adapter."""

    def __init__(self, model, mesh, components, scale, keep_base=True):
        import g4mix
        import jax
        import numpy as np
        from flax import nnx

        self.jax, self.np, self.scale, self.mesh = jax, np, float(scale), mesh
        self.factors = g4mix.module_factors(components)
        self.modules = sorted(self.factors[0])
        found = {}
        for path, value in nnx.iter_graph(model):
            if isinstance(value, nnx.Param) and not isinstance(value, nnx.LoRAParam) and path and path[-1] == "kernel":
                found.setdefault(from_decoder("/".join(str(p) for p in path[:-1])), []).append(value)
        missing = [m for m in self.modules if len(found.get(from_decoder(m), [])) != 1]
        if missing:
            raise SystemExit(f"--combine: {len(missing)} adapter modules without exactly one base kernel, e.g. "
                             f"{[(m, len(found.get(from_decoder(m), []))) for m in missing[:3]]}")
        self.vars = {m: found[from_decoder(m)][0] for m in self.modules}
        self.host, self.fns = {}, {}
        for m, var in self.vars.items():
            if not keep_base:
                break
            w = self.get(var)
            self.host[m] = {self.index_key(s.index): np.asarray(s.data) for s in w.addressable_shards}
        self.replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    @staticmethod
    def get(var):
        return var.get_value() if hasattr(var, "get_value") else var.value

    @staticmethod
    def put(var, value):
        if hasattr(var, "set_value"):
            var.set_value(value)
        else:
            var.value = value

    @staticmethod
    def index_key(index):
        return tuple((s.start, s.stop, s.step) for s in index)

    def base(self, module):
        w = self.get(self.vars[module])
        shards = self.host[module]

        def shard(index):
            key = self.index_key(index)
            if key not in shards:
                raise SystemExit(f"--combine: {module} has no host copy of the shard {key}")
            return shards[key]

        return self.jax.make_array_from_callback(w.shape, w.sharding, shard)

    def merge_fn(self, w, donate=False):
        import jax.numpy as jnp
        import translate

        key = (tuple(w.shape), str(w.dtype), str(w.sharding), donate)
        if key not in self.fns:
            scale = self.scale
            self.fns[key] = self.jax.jit(
                lambda w0, a, b: (w0.astype(jnp.float32) + translate.lora_delta(a, b, w0.shape, scale, jnp)).astype(
                    w0.dtype), out_shardings=w.sharding, donate_argnums=(0,) if donate else ())
        return self.fns[key]

    def bake(self):
        """Every module's kernel = itself + the one component's update (coefficient 1), its buffer donated to the merge
        (addendum G4-6c.6d): one-way, nothing restores the base"""
        import g4mix

        np = self.np
        for m in self.modules:
            var = self.vars[m]
            w = self.get(var)
            a, b = g4mix.stack_module([f[m] for f in self.factors], [1.0])
            put = [self.jax.make_array_from_callback(x.shape, self.replicated, lambda i, x=x: x[i])
                   for x in (np.ascontiguousarray(a), np.ascontiguousarray(b))]
            self.put(var, self.merge_fn(w, donate=True)(w, *put))
            del w

    def apply(self, coefs):
        """Every module's kernel = W0 + its stacked update with these coefficients (all zero: W0 itself)"""
        import g4mix

        np = self.np
        for m in self.modules:
            w0 = self.base(m)
            if not any(coefs):
                self.put(self.vars[m], w0)
                continue
            a, b = g4mix.stack_module([f[m] for f in self.factors], coefs)
            put = [self.jax.make_array_from_callback(x.shape, self.replicated, lambda i, x=x: x[i])
                   for x in (np.ascontiguousarray(a), np.ascontiguousarray(b))]
            self.put(self.vars[m], self.merge_fn(w0)(w0, *put))

    def restore(self):
        self.apply([0.0] * len(self.factors))


def broadcast_float(value):
    """Process 0's float on every process, so that the search's choices are the same everywhere"""
    import jax

    if jax.process_count() == 1:
        return float(value)
    import numpy as np
    from jax.experimental import multihost_utils

    return float(np.asarray(multihost_utils.broadcast_one_to_all(np.array(value, np.float64))))


def run_combine(spec, model, mesh, run_files, names):
    """--combine: the anchors, the coordinate search on the objective's eval files, the top K on every eval file, the
    first anchor once more (the same loss: the base came back), the base kernels restored"""
    import g4mix
    import numpy as np

    t0 = time.perf_counter()
    components = []
    for c in spec["components"]:
        with np.load(c["adapter"]) as z:
            components.append({k: z[k] for k in z.files})
    merger = KernelMerger(model, mesh, components, spec["scale"])
    objective_sets = list(spec["objective"])
    records = []

    def evaluate(w, tag, sets, into=records):
        coefs = g4mix.coefficients(w, spec["shared_coef"])
        t = time.perf_counter()
        merger.apply(coefs)
        merge_s = time.perf_counter() - t
        rows = run_files(sets, tag)
        losses = {r["set"]: r["loss"] for r in rows}
        record = {"tag": tag, "w": list(w), "coefs": coefs, "losses": losses,
                  "tokens": {r["set"]: r["tokens"] for r in rows}, "merge_s": round(merge_s, 1),
                  "seconds": round(time.perf_counter() - t, 1)}
        if all("temperatures" in r for r in rows):
            record["curves"] = {r["set"]: r["temperatures"] for r in rows}
        into.append(record)
        return losses

    def objectives(record):
        """addendum G4-5.7b A: both objectives of a point, at T = 1 and at its best T (from the same forward pass)"""
        out = {"objective_t1": g4mix.objective_value(record["losses"], spec["objective"])}
        if "curves" in record:
            out["objective_calibrated"], out["t_calibrated"] = g4mix.calibrated_objective(record["curves"],
                                                                                          spec["objective"])
        record.update(out)
        return out["objective_calibrated" if spec["objective_mode"] == "calibrated" else "objective_t1"]

    warmup = None
    if spec.get("warmup_check"):
        # addendum G4-5.7b: the merged warmup alone against the warmup's own eval, before any point of the search
        check, kept = spec["warmup_check"], []
        losses = evaluate([0.0] * (len(spec["components"]) - 1), "check/warmup", list(check["losses"]), into=kept)
        warmup = g4mix.warmup_check_result(check, losses, kept[0]["tokens"]) | {"seconds": kept[0]["seconds"]}
        # process 0's verdict everywhere, so that all processes stop or go on together
        warmup["holds"] = broadcast_float(1.0 if warmup["holds"] else 0.0) == 1.0
        print(f"WARMUP_CHECK {json.dumps(warmup)}", flush=True)
        if not warmup["holds"]:
            raise SystemExit(f"--combine: the merged warmup alone is not the warmup's own eval at step {check['step']} "
                             f"(same tokens, within {check['max_diff']}; addendum G4-5.7b): {json.dumps(warmup)}")

    def objective(w):
        evaluate(w, f"search/{len(records)}", objective_sets)
        return broadcast_float(objectives(records[-1]))

    search = g4mix.run_search(objective, spec)
    for p in search["points"]:
        r = records[p["order"]]
        if r["tag"] != f"search/{p['order']}" or g4mix.key_of(r["w"]) != g4mix.key_of(p["w"]):
            raise SystemExit(f"--combine: point {p['order']} is not the evaluation {r['tag']}")
        p["losses"] = r["losses"]
        p.update({k: r[k] for k in ("objective_t1", "objective_calibrated", "t_calibrated") if k in r})
    for k, p in enumerate(search["top"], 1):
        p["losses"] = evaluate(p["w"], f"top/{k}", names)
    first = search["points"][0]
    evaluate(first["w"], "repeat", objective_sets)
    again = objectives(records[-1])
    merger.restore()
    out = {"spec": spec, "search": search, "evals": records,
           "repeat": g4mix.repeat_result(first["w"], first["value"], again, spec["repeat_max_diff"]),
           "warmup_check": warmup, "seconds": round(time.perf_counter() - t0, 1)}
    print(f"COMBINE {json.dumps({k: out[k] for k in ('repeat', 'seconds')} | {'objective_mode': spec['objective_mode'], 'best': search['best'], 'top': [(p['w'], p['value']) for p in search['top']], 'evaluations': search['evaluations'], 'stopped': search['stopped']})}",
          flush=True)
    return out


def scalar(x):
    """A replicated device scalar on the host (this process's copy)"""
    import numpy as np

    value = x.addressable_data(0) if hasattr(x, "addressable_data") else x
    return float(np.asarray(value))


def vector(x):
    """A device vector on the host (this process's copy if replicated, else gathered from every process)"""
    import numpy as np

    if getattr(x, "is_fully_replicated", True):
        value = x.addressable_data(0) if hasattr(x, "addressable_data") else x
    else:
        from jax.experimental import multihost_utils

        value = multihost_utils.process_allgather(x, tiled=True)
    return [float(v) for v in np.asarray(value).reshape(-1)]


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
    parser.add_argument("--init-adapter", default=None,
                        help="an adapter.npz of this script: the LoRA parameters before the first step (its eval at "
                             "step 0 is that adapter's model)")
    parser.add_argument("--resume-step", type=int, default=0,
                        help="continue the training state of this step, which Tunix restores from the run's own "
                             "checkpoint directory (copied there before); --steps counts this run's own steps")
    parser.add_argument("--resume-adapter", default=None,
                        help="with --resume-step: the adapter.npz of the restored step (checked against the restore)")
    parser.add_argument("--warmup-steps", type=int, default=None,
                        help="WSD with exactly this many warmup steps (with --decay-fraction)")
    parser.add_argument("--decay-fraction", type=float, default=None,
                        help="WSD decay over int(this x the run's own steps) steps at its end (0: none)")
    parser.add_argument("--combine", default=None,
                        help="g4mix.py spec: the search over the weights of a warmup and its branches at the first eval")
    parser.add_argument("--temperatures", default=None,
                        help="LOW:HIGH:STEP or T1,T2,...: eval only, the first eval's loss per file also at logits / T "
                             "for every T (with 1.0), from the same forward pass; needs num_vocab_tiling=1")
    parser.add_argument("--dpo", default=None,
                        help="g4dpo spec (json): preference training on --pref-train with the base model as reference")
    parser.add_argument("--pref-train", default=None, help="with --dpo: the packed training file of g4dpo.py")
    parser.add_argument("--pref-eval", action="append", default=[],
                        help="with --dpo: NAME=FILE, a packed eval file of pairs (repeatable)")
    parser.add_argument("--base-adapter", default=None,
                        help="with --dpo: NAME=FILE, an adapter.npz of this script merged into the base kernels before "
                             "the first step (the start model and the reference; arm G4-6b)")
    parser.add_argument("--save-steps", default=None,
                        help="K1,K2,...: the LoRA parameters after these steps as adapter-step<K>.npz/.json (arm G4-6c)")
    args = parser.parse_args(own)
    if args.resume_step < 0 or bool(args.resume_step) != bool(args.resume_adapter):
        raise SystemExit("--resume-step N (> 0) and --resume-adapter go together")
    if (args.warmup_steps is None) != (args.decay_fraction is None):
        raise SystemExit("--warmup-steps and --decay-fraction go together")
    if args.decay_fraction is not None and not 0 <= args.decay_fraction < 1:
        raise SystemExit(f"--decay-fraction {args.decay_fraction}: in [0, 1)")
    combine_spec = None
    if args.combine:
        if args.steps != "1" or args.resume_step or args.init_adapter:
            raise SystemExit("--combine is a measurement: --steps 1, no --resume-step, no --init-adapter")
        combine_spec = json.loads(Path(args.combine).read_text())
    evals_wanted = parse_evals(args.eval)
    if combine_spec:
        if "repeat_max_diff" not in combine_spec or combine_spec.get("objective_mode") not in ("t1", "calibrated"):
            raise SystemExit("--combine: the spec has no repeat_max_diff or objective_mode (g4mix.py spec of addendum "
                             "G4-5.7b)")
        unknown = [s for s in combine_spec["objective"] if s not in evals_wanted]
        unknown += [s for s in (combine_spec.get("warmup_check") or {}).get("losses", {}) if s not in evals_wanted]
        if unknown or not evals_wanted:
            raise SystemExit(f"--combine: the objective's or the warmup check's sets {unknown} are no --eval")
    temps = None
    if args.temperatures:
        if args.resume_step or not evals_wanted:
            raise SystemExit("--temperatures is an eval-only measurement: --eval files, no --resume-step")
        temps = temperature_grid(args.temperatures)
    if combine_spec and combine_spec["objective_mode"] == "calibrated" and not temps:
        raise SystemExit("--combine: the calibrated objective needs --temperatures (addendum G4-5.7b A)")
    dpo_opts, prefs_wanted = dpo_args(args, evals_wanted)
    base_spec = base_adapter_arg(args.base_adapter) if args.base_adapter else None
    if prefs_wanted and not evals_wanted:
        # Tunix runs its eval rounds only with an eval iterator, which eval_overrides makes from the first --eval file
        raise SystemExit("--pref-eval needs at least one --eval file")
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
        # the job's processes are started over ssh on four workers: JAX's default of 300 s for all of them to meet
        # here leaves little margin (19 Nov: starts 6 min apart, every process gave up after 300 s)
        jax.distributed.initialize(initialization_timeout=900)
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
    pref_rows = config.global_batch_size_to_train_on
    if dpo_opts:
        # --dpo: the packed file's global batches (every process takes its rows of each), counted from the file
        import g4dpo

        count = {**g4dpo.packed_info(args.pref_train, pref_rows, config.max_target_length), "file": args.pref_train}
        own_steps = count["batches"] if args.steps == "auto" else int(args.steps)
        if not 1 <= own_steps <= count["batches"]:
            raise SystemExit(f"--dpo: {own_steps} steps, the packed file has {count['batches']} batches")
    # A multi-host job counts also when given a number of steps, to hold that number to step_limit
    elif args.steps == "auto" or jax.process_count() > 1:
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
    if not dpo_opts:
        own_steps = choose_steps(args.steps, batches_per_process)
    # --resume-step: MaxText's steps (Tunix's max_steps) and the schedule count from the restored step
    steps = args.resume_step + own_steps
    save_at = save_steps_arg(args.save_steps, steps, args.eval_every if evals_wanted else 0)
    if save_at and (temps or combine_spec):
        raise SystemExit("--save-steps belongs to a training run (no --temperatures, no --combine)")
    print(f"EPOCH {json.dumps(count | {'requested': args.steps, 'steps': steps, 'own_steps': own_steps, 'resume_step': args.resume_step})}",
          flush=True)
    if args.warmup_steps is not None:
        decay = int(args.decay_fraction * own_steps)
        base = base + schedule_overrides(steps, args.warmup_steps, decay)
    config = pyconfig.initialize_pydantic(base + [f"steps={steps}"])
    if temps and (config.num_vocab_tiling != 1 or config.z_loss_multiplier != 0):
        # the wrapped cross entropy must see the full logits at the top of the loss (no tiles in a scan), and the
        # loss must be the plain cross entropy
        raise SystemExit(f"--temperatures needs num_vocab_tiling=1 and z_loss_multiplier=0, not "
                         f"{config.num_vocab_tiling} and {config.z_loss_multiplier}")
    # before the training, so that a schedule this cannot read stops the run before it costs anything
    lr_at = learning_rates(config, steps, extra=(args.resume_step, args.resume_step + 1) if args.resume_step else ())
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
        # Every process runs the same number of eval batches, the smallest count, without padding batches: on a
        # v4-32 (4 processes) a padding batch on one host while the others still had data aborted the slice
        # (SLICE_FAILURE_SW_INJECT_ERROR, G4-2 smoke test 14 Nov). The last batches of the larger shares are left out.
        eval_plan[name] = {"path": path, "batches": min(counts), "batches_per_process": counts,
                           "loss_tokens_here": loss_tokens, "end": end,
                           "config": pyconfig.initialize_pydantic(base + [f"steps={steps}", f"hf_eval_files={path}",
                                                                          "generate_padding_batch_eval=false",
                                                                          f"eval_steps={min(counts)}"])}
        print(f"EVAL_PLAN {json.dumps({k: v for k, v in eval_plan[name].items() if k != 'config'} | {'set': name})}",
              flush=True)
    for name, path in prefs_wanted.items():
        # --dpo: a packed eval file, its global batches the same on every process (the eval step's batch shape: as
        # many rows as MaxText's eval batches, so that one compiled eval step serves both kinds)
        if config.global_batch_size_to_load_eval != pref_rows:
            raise SystemExit(f"--pref-eval: eval batches of {config.global_batch_size_to_load_eval} rows, training "
                             f"batches of {pref_rows}; the packed files have one size")
        info = g4dpo.packed_info(path, pref_rows, config.max_target_length)
        eval_plan[name] = {"path": path, "batches": info["batches"], "kind": "pref", "pairs": info["pairs"],
                           "batches_per_process": [info["batches"]] * jax.process_count(), "end": "packed file"}
        print(f"EVAL_PLAN {json.dumps(eval_plan[name] | {'set': name})}", flush=True)

    t_setup = time.time()
    trainer, mesh = train_sft.setup_trainer_state(config)
    setup_s = time.time() - t_setup
    dpo_steps = []
    if dpo_opts:
        # --dpo: MaxText's loss replaced by the preference loss (training and eval), MaxText's training data by the
        # packed file's batches (below, with the reference), and every step's sums kept
        trainer = trainer.with_loss_fn(g4dpo.make_loss(config, dpo_opts, True), has_aux=True)
        trainer.eval_loss_fn = g4dpo.make_loss(config, dpo_opts, False)
        post_train = trainer._post_process_train_step  # noqa: SLF001 - wrapped, not replaced

        def keep_dpo_step(aux):
            dpo_steps.append({k: aux[f"pref_{k}"] for k in g4dpo.SUMS})   # device scalars, read after training
            return post_train(aux)

        trainer._post_process_train_step = keep_dpo_step  # noqa: SLF001
        print(f"DPO {json.dumps({'spec': dpo_opts, 'train': count, 'own_steps': own_steps})}", flush=True)
    # Shapes only (no transfer): what the LoRA regex reached, before the first step, so a run that fails in compilation
    # still shows it
    lora_sizes = {path_key(path): math.prod(leaf.shape) for path, leaf in jax.tree_util.tree_flatten_with_path(
        nnx.to_pure_dict(nnx.state(trainer.model, nnx.LoRAParam)))[0]}
    module_paths = ["/".join(str(p) for p in path) for path, value in nnx.iter_graph(trainer.model)
                    if path and isinstance(value, nnx.Module)]
    lora_seen = lora_report(lora_utils._get_lora_module_path(config), lora_sizes, module_paths)  # noqa: SLF001
    print(f"LORA_MODULES {json.dumps(lora_seen)}", flush=True)
    # before the first eval and step: the start model, and so the reference of --dpo
    base_adapter = bake_adapter(trainer.model, mesh, base_spec) if base_spec else None
    if base_adapter:
        print(f"BASE_ADAPTER {json.dumps(base_adapter)}", flush=True)
    init_adapter = load_adapter(trainer.model, args.init_adapter) if args.init_adapter else None
    if init_adapter:
        print(f"INIT_ADAPTER {json.dumps(init_adapter)}", flush=True)
    resumed = check_resume(trainer, args.resume_step, args.resume_adapter) if args.resume_step else None
    if resumed:
        print(f"RESUMED {json.dumps(resumed)}", flush=True)
    refs, ref_record, ref_s = {}, {}, 0.0
    if dpo_opts:
        # --dpo, amendment G4-6.4a: the model is now the one of the first eval and step (every LoRA B zero, checked):
        # the reference's sums of the batches the run trains on and of every preference eval file, forward only, in the
        # training's mesh and rules; then the training data are the packed file's batches with their reference (one
        # all-zero batch after them for Tunix's look-ahead)
        from flax.core.spmd import logical_axis_rules

        t_ref = time.time()
        ref_files = [("train", args.pref_train, own_steps)] + [(k, v, eval_plan[k]["batches"])
                                                                for k, v in prefs_wanted.items()]
        with jax.set_mesh(mesh), logical_axis_rules(config.logical_axis_rules):
            for name, path, n in ref_files:
                refs[name], secs = g4dpo.precompute_reference(trainer.model, config, path, mesh, pref_rows, n,
                                                              dpo_opts["max_segments"], dpo_opts["end_tokens"])
                ref_record[name] = {"file": path, "batches": n, "seconds": round(sum(secs), 1),
                                    "batch_s_median": round(statistics.median(secs), 3),
                                    "ref_sum": float(np.sum(refs[name], dtype=np.float64)),
                                    "device_gib": device_bytes()}
                print(f"REF {json.dumps(ref_record[name] | {'set': name})}", flush=True)
        ref_s = time.time() - t_ref
        if main_process:
            import pyarrow as pa
            import pyarrow.parquet as pq

            rows_out = [{"set": name, **r} for name, path, n in ref_files
                        for r in g4dpo.reference_rows(path, refs[name], n, pref_rows)]
            pq.write_table(pa.Table.from_pylist(rows_out), out / "dpo_ref.parquet")
        hooks_data = trainer.data_hooks
        close(hooks_data.train_data_iterator)
        pref_iter = g4dpo.global_iterator(args.pref_train, mesh, pref_rows, own_steps, padding=1, ref=refs["train"],
                                          end_tokens=dpo_opts["end_tokens"])
        hooks_data.train_data_iterator = pref_iter
        hooks_data.train_data_loader.data_iterator = pref_iter
    if temps:
        from maxtext.utils import max_utils

        trainer.eval_loss_fn = with_temperatures(trainer.eval_loss_fn, temps, config.micro_batch_size_to_eval_on,
                                                 max_utils)
        print(f"TEMPERATURES {json.dumps(temps)}", flush=True)

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

    saved = {}

    def keep_adapter(k):
        # amendment G4-6c: the LoRA parameters after step k, in its eval round (the model that eval measures)
        arrays, kinds = adapter_arrays(nnx.to_pure_dict(nnx.state(trainer.model, nnx.LoRAParam)))
        lora_cfg = config.lora
        meta = {"rank": lora_cfg.lora_rank, "alpha": lora_cfg.lora_alpha,
                "scale": lora_cfg.lora_alpha / lora_cfg.lora_rank, "module_path": lora_utils._get_lora_module_path(config),  # noqa: SLF001
                "model": config.model_name, "checkpoint": config.load_parameters_path,
                "trained_dtype": sorted(kinds), "steps": k, "learning_rate": config.learning_rate,
                "train_files": config.hf_train_files}
        if main_process:
            np.savez(out / f"adapter-step{k}.npz", **arrays)
            (out / f"adapter-step{k}.json").write_text(json.dumps(meta, indent=1) + "\n")
        saved[k] = {"keys": len(arrays), "sum_abs": float(sum(np.abs(v).sum(dtype=np.float64) for v in arrays.values())),
                    "peak_gib": peak_gib()}
        print(f"SAVED {json.dumps({'step': k, **saved[k]})}", flush=True)

    def step_end(ctx, train_step, train_loss, step_time=0.0):
        records.append({"step": int(train_step), "loss": float(np.asarray(train_loss)), "t": time.perf_counter()})
        if int(train_step) == args.resume_step + 1:
            # Tunix calls this for step 1 after step 2: the peak of the compiled step, before a run's end shows it
            print(f"PEAK {json.dumps({'step': int(train_step), 'peak_gib': peak_gib()})}", flush=True)
        return on_step_end(ctx, train_step, train_loss, step_time)

    hooks.on_train_start, hooks.on_train_step_start, hooks.on_train_step_end = train_start, step_start, step_end

    evals, combined = [], {}
    if eval_plan:
        # Tunix's eval loop, file by file: our batches (data_hooks.load_next_eval_batch), the planned number of them
        # from a fresh iterator; the jitted eval step wrapped to keep each batch's xent_sum and total_weights
        data_hooks = trainer.data_hooks
        close(data_hooks.eval_data_iterator)          # the hooks' own iterator of the first file, never read
        data_hooks.eval_data_iterator = [None]        # truthy for Tunix's `if eval_ds`; iterated, never read
        feed = {"iter": None, "left": 0, "parts": [], "temps": [], "prefs": [], "eval_step": None}

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
                if dpo_opts and "pref_pairs" in aux:      # a preference file's batch (an SFT file's has no pairs)
                    feed["prefs"].append({k: aux[f"pref_{k}"] for k in g4dpo.SUMS})
                if temps:
                    feed["temps"].append(aux["xent_temps"])
                return loss, aux

            feed["eval_step"] = eval_recorded
            return train_step, eval_recorded

        trainer.jit_train_and_eval_step = jit_recorded
        run_eval_tunix = trainer._run_eval  # noqa: SLF001 - wrapped, not replaced

        def run_files(names, eval_ds, eval_step_fn, tag=None):
            step = trainer.train_steps
            out = []
            for name in names:
                plan = eval_plan[name]
                pref = plan.get("kind") == "pref"
                if pref:
                    feed["iter"] = g4dpo.global_iterator(plan["path"], mesh, pref_rows, plan["batches"], ref=refs[name],
                                                         end_tokens=dpo_opts["end_tokens"])
                else:
                    feed["iter"] = eval_iterator(plan["config"], mesh)
                feed["left"], feed["parts"], feed["temps"], feed["prefs"] = plan["batches"], [], [], []
                t0 = time.perf_counter()
                run_eval_tunix(eval_ds, eval_step_fn)
                close(feed["iter"])
                if feed["left"]:
                    raise SystemExit(f"eval {name} at step {step}: {feed['left']} planned batches not run")
                if pref:
                    sums = [{k: scalar(v) for k, v in p.items()} for p in feed["prefs"]]
                    record = g4dpo.pref_record(name, step, sums, plan["batches"], time.perf_counter() - t0)
                    if record["pairs"] != plan["pairs"]:
                        raise SystemExit(f"eval {name} at step {step}: {record['pairs']} pairs, the file has "
                                         f"{plan['pairs']}")
                    out.append(record)
                    print(f"EVAL {json.dumps(record)}", flush=True)
                    continue
                parts = [(scalar(x), scalar(w)) for x, w in feed["parts"]]
                record = eval_loss(name, step, parts, plan["batches"], time.perf_counter() - t0)
                if temps:
                    record["temperatures"] = temperature_record(temps, [vector(x) for x in feed["temps"]],
                                                                sum(w for _, w in parts), plan["batches"])
                if tag is not None:
                    record["combo"] = tag
                out.append(record)
                print(f"{'COMBO_EVAL' if tag is not None else 'EVAL'} {json.dumps(record)}", flush=True)
            return out

        def run_eval(eval_ds, eval_step_fn):
            if trainer.train_steps in save_at and trainer.train_steps not in saved:
                keep_adapter(trainer.train_steps)
            evals.extend(run_files(list(eval_plan), eval_ds, eval_step_fn))
            if combine_spec and "result" not in combined:
                combined["result"] = run_combine(
                    combine_spec, trainer.model, mesh,
                    lambda names, tag: run_files(names, eval_ds, eval_step_fn, tag), list(eval_plan))
            if temps:
                # eval only: Tunix's first eval comes before the first train step, which is then never compiled
                raise EvalOnlyDone()

        trainer._run_eval = run_eval  # noqa: SLF001
        on_end = hooks.on_train_end

        def train_end(ctx):
            # after the loop, before MaxText's hook flushes its metric logger and Tunix closes the trainer
            if needs_final_eval(evals, ctx.train_steps):
                run_eval(data_hooks.eval_data_iterator, feed["eval_step"])
            return on_end(ctx)

        hooks.on_train_end = train_end

    t_train = time.time()
    try:
        train_sft.train_model(config, trainer, mesh)
    except EvalOnlyDone:
        if not temps:
            raise
    train_s = time.time() - t_train
    if temps:
        if records or not evals or any("temperatures" not in e for e in evals):
            raise SystemExit(f"--temperatures: {len(records)} steps ran, {len(evals)} evals: not the eval-only round")
    elif trainer.train_steps != steps or len(records) != own_steps:
        # the loop ended early (the data ran out) or a step went unlogged: not the run that was set up
        raise SystemExit(f"trained up to step {trainer.train_steps} ({len(records)} logged), not {steps} "
                         f"({own_steps} own steps)")
    if combine_spec and "result" not in combined:
        raise SystemExit("--combine: the search did not run (no eval before the first step?)")
    if sorted(saved) != save_at:
        raise SystemExit(f"--save-steps {save_at}: saved after {sorted(saved)}")

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
        "epoch": count, "steps": steps, "steps_done": len(records), "own_steps": own_steps,
        "resume": resumed, "schedule": {"warmup_steps": args.warmup_steps, "decay_fraction": args.decay_fraction},
        "times": {"python_to_setup_s": round(t_setup - started, 1), "setup_s": round(setup_s, 1),
                  "train_s": round(train_s, 1), "ref_s": round(ref_s, 1),
                  "first_step_s": round(first_compile, 1) if first_compile is not None else None,
                  "compile_s_est": round(first_compile - statistics.median(later), 1) if later else None},
        "throughput": throughput(records, tokens, loss_tokens),
        "tokens": {"total": sum(tokens), "loss": sum(loss_tokens), "per_step": tokens, "loss_per_step": loss_tokens},
        "memory": memory, "peak_gib_max": max(m["peak_gib"] for m in memory),
        # beside --save-steps' sums (amendment G4-6c.6: a kept adapter is not the last one)
        "adapter_sum_abs": float(sum(np.abs(v).sum(dtype=np.float64) for v in flat.values())),
        "loss": [[r["step"], round(r["loss"], 5)] for r in records],
        "eval": evals,
        "eval_plan": {k: {x: v[x] for x in ("path", "batches", "batches_per_process", "end")}
                      for k, v in eval_plan.items()},
        "lr_at": lr_at,
        "init_adapter": init_adapter,
        "base_adapter": base_adapter,
        "saved_adapters": {str(k): v for k, v in sorted(saved.items())},
        "combine": combined.get("result"),
        "temperatures": temps,
        # --dpo: the spec, the packed training file and every step's preference measures (its batch's pairs)
        "dpo": ({"spec": dpo_opts, "train": count, "pref_eval": prefs_wanted, "ref": ref_record,
                 "steps": [g4dpo.step_record(k, {n: scalar(v) for n, v in s.items()})
                           for k, s in enumerate(dpo_steps, args.resume_step + 1)]} if dpo_opts else None),
    }
    if summary["throughput"]:
        summary["throughput"]["tokens_per_s_per_chip"] = round(summary["throughput"]["tokens_per_s"] / jax.device_count(), 1)
    if main_process:
        if temps and not (out / "metrics.txt").exists():
            (out / "metrics.txt").write_text("")   # eval only: MaxText logged no step; run_sft.sh copies the file
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
