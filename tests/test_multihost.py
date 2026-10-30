"""Multi-host training in scripts/train_minimal.py, on two real JAX processes on the CPU.

Each test starts this file as worker processes (see worker_main) with
--xla_force_host_platform_device_count and jax.distributed.initialize, and runs the real
train_minimal.main() on a tiny model and made-up data. Only the data sources, the tokenizer
and, for "same global batch" runs, the masking collator are replaced.

Tolerance, fixed before the first run: PARAM_ABS_TOL = 1e-5 absolute on every parameter of
a run of a few steps at lr 1e-3, which is 1% of one AdamW step (an AdamW step moves a
parameter by about lr). Two topologies that sum the same gradients may only differ by float32
rounding, so anything larger is a bug. Comparisons of one host against another and of a
restored checkpoint against the saved state are exact (hashes of the raw bytes).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.machinery
import json
import os
import socket
import subprocess
import sys
import threading
import time
import traceback
import types
from pathlib import Path

import numpy as np

PARAM_ABS_TOL = 1e-5

# tiny config: 16 tokens, 2 layers, 16 hidden
MAX_LENGTH = 16
VOCAB_WORDS = 123
LEARNING_RATE = 1e-3
WORKER_TIMEOUT = 300


# --------------------------------------------------------------------------- worker side


def _stub_wandb_if_missing():
    """train_minimal imports wandb at the top, but these --no-wandb runs never call it."""
    try:
        import wandb  # noqa: F401
    except ImportError:
        stub = types.ModuleType("wandb")
        # accelerate, imported through transformers, calls find_spec("wandb")
        stub.__spec__ = importlib.machinery.ModuleSpec("wandb", None)
        sys.modules["wandb"] = stub


def _make_tokenizer():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    specials = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    words = [f"w{i}" for i in range(VOCAB_WORDS)]
    vocab = {token: i for i, token in enumerate(specials + words)}
    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="[PAD]",
        unk_token="[UNK]",
        bos_token="[CLS]",
        eos_token="[SEP]",
        mask_token="[MASK]",
    )


def _row_arrays(rows, with_packing_fields):
    """Input, mask and labels that depend on the row id only, so any topology builds the same row."""
    input_ids, labels = [], []
    for row in rows:
        rng = np.random.RandomState(int(row))
        ids = rng.randint(5, 128, size=MAX_LENGTH)
        target = np.where(rng.random_sample(MAX_LENGTH) < 0.3, ids, -100)
        target[0] = ids[0]
        input_ids.append(np.where(target != -100, 4, ids))
        labels.append(target)
    n = len(rows)
    batch = {
        "input_ids": np.stack(input_ids),
        "attention_mask": np.ones((n, MAX_LENGTH), dtype=np.int64),
        "labels": np.stack(labels),
    }
    if with_packing_fields:
        # Two documents per row, as a packed row would have
        batch["position_ids"] = np.tile(np.r_[np.arange(MAX_LENGTH // 2), np.arange(MAX_LENGTH // 2)], (n, 1))
        batch["segment_ids"] = np.tile(np.r_[np.zeros(MAX_LENGTH // 2), np.ones(MAX_LENGTH // 2)].astype(np.int64), (n, 1))
    return batch


def _digest_per_device(tree):
    """sha256 of the raw parameter bytes held by each local device."""
    import jax

    digests = {}
    for path, leaf in sorted(jax.tree_util.tree_leaves_with_path(tree), key=lambda item: str(item[0])):
        for shard in leaf.addressable_shards:
            digest = digests.setdefault(str(shard.device), hashlib.sha256())
            digest.update(str(path).encode())
            digest.update(np.asarray(shard.data).tobytes())
    return {device: digest.hexdigest() for device, digest in digests.items()}


def _flat_numpy(tree):
    import jax

    return {str(path): np.asarray(leaf) for path, leaf in jax.tree_util.tree_leaves_with_path(tree)}


def _save_numpy_state_like_before_the_fix(tm, ocp, real_standard_save):
    """What prepare_state_for_checkpoint used to do, for the old-checkpoint test."""
    import jax

    def to_numpy_save(state):
        return real_standard_save(jax.tree_util.tree_map(np.array, state))

    tm.ocp.args.StandardSave = to_numpy_save


def worker_main(spec):
    # Before train_minimal is imported: its own module-level initialize must see this and stay out.
    import jax

    if spec["num_processes"] > 1:
        jax.distributed.initialize(
            coordinator_address=f"localhost:{spec['port']}",
            num_processes=spec["num_processes"],
            process_id=spec["process_id"],
        )

    _stub_wandb_if_missing()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import train_minimal as tm

    ocp = tm.ocp
    process_id = jax.process_index()
    record = {"first_rows": None, "restored": None}
    output = Path(spec["output"])

    tm.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda *_a, **_k: _make_tokenizer())

    # Local paths are refused on multi-host because hosts normally share no disk; these two
    # processes do. gs:// also selects the async save that real multi-host runs use.
    real_manager = ocp.CheckpointManager

    class LocalManager(real_manager):
        def __init__(self, directory, **kwargs):
            super().__init__(str(directory).removeprefix("gs://"), **kwargs)

        def restore(self, step, args=None, **kwargs):
            restored = super().restore(step, args=args, **kwargs)
            record["restored"] = {
                "step": step,
                "digests": _digest_per_device(restored.params),
                "replicated": all(leaf.is_fully_replicated for leaf in jax.tree_util.tree_leaves(restored)),
                "state_step": int(restored.step),
            }
            np.savez(output / f"restored_{process_id}.npz", **_flat_numpy(restored.params))
            return restored

    ocp.CheckpointManager = LocalManager

    real_data_state_path = tm.data_state_path
    tm.data_state_path = lambda directory, step, process: real_data_state_path(str(directory).removeprefix("gs://"), step, process)

    if spec.get("legacy_numpy_save"):
        _save_numpy_state_like_before_the_fix(tm, ocp, ocp.args.StandardSave)

    per_host = spec["per_device"] * jax.local_device_count()
    global_rows = per_host * jax.process_count()

    def rows_of(step):
        first = step * global_rows + process_id * per_host
        return list(range(first, first + per_host))

    def fixed_rows():
        step = 0
        while process_id != spec.get("exhaust_process") or step < spec["exhaust_after"]:
            yield from ({"text": "unused", "row": row} for row in rows_of(step))
            step += 1

    class FixedCollator:
        def __init__(self, tokenizer=None, mlm_probability=0.15, **_kwargs):
            self.tokenizer = tokenizer

        def __call__(self, examples, max_length=None):
            packed = isinstance(examples, dict)
            rows = examples["rows"] if packed else [example["row"] for example in examples]
            if record["first_rows"] is None:
                record["first_rows"] = [int(row) for row in rows]
            return _row_arrays(rows, with_packing_fields=packed)

    class FixedPackedLoader:
        epoch = 0
        documents_read = 0

        def get_state(self):
            return {}

        def __iter__(self):
            step = 0
            while True:
                yield {"rows": rows_of(step)}
                step += 1

    data = spec["data"]
    if data == "fixed":
        tm.FlaxDataCollatorForLanguageModeling = FixedCollator
        tm.PackedMLMDataCollator = FixedCollator
        tm.load_dataset_for_training = lambda **_kwargs: fixed_rows()
        tm.create_packed_dataloader = lambda **_kwargs: FixedPackedLoader()
    elif data == "real_pipeline":
        # Real split_dataset_by_node, real Grain packing, real masking; only the download is fake.
        import datasets

        from mmbjax.training import packing

        class RecordingCollator(tm.PackedMLMDataCollator):
            """Keeps what the host read before masking, one line per batch, for a process that gets killed."""

            batches = 0

            def __call__(self, batch):
                with open(output / f"raw_batches_{process_id}.txt", "a") as f:
                    f.write(hashlib.sha256(np.asarray(batch["input_ids"]).tobytes()).hexdigest() + "\n")
                type(self).batches += 1
                if type(self).batches == spec.get("wait_to_be_killed_at_batch"):
                    (output / f"ready_to_be_killed_{process_id}").touch()
                    time.sleep(WORKER_TIMEOUT)
                return super().__call__(batch)

        tm.PackedMLMDataCollator = RecordingCollator

        def generate(shards):
            rng = np.random.RandomState(0)
            for shard in shards:
                for i in range(spec["rows_per_shard"]):
                    words = rng.randint(0, VOCAB_WORDS, size=rng.randint(3, 11))
                    yield {"text": " ".join(f"w{w}" for w in words), "row": shard * 100000 + i}

        fake = datasets.IterableDataset.from_generator(generate, gen_kwargs={"shards": list(range(8))})
        fake_load = lambda *_a, **_k: fake  # noqa: E731
        tm.load_dataset = fake_load
        packing.load_dataset = fake_load

    argv = [
        "train_minimal.py",
        "--dataset", "fake",
        "--tokenizer", "fake",
        "--tracker", spec.get("tracker", "none"),
        "--compute-dtype", "float32",
        "--max-length", str(MAX_LENGTH),
        "--hidden-size", "16",
        "--num-layers", "2",
        "--num-heads", "2",
        "--intermediate-size", "32",
        "--learning-rate", str(LEARNING_RATE),
        "--warmup-steps", "2",
        "--log-every", "1",
        "--max-steps", str(spec["max_steps"]),
        "--per-device-batch-size", str(spec["per_device"]),
        "--checkpoint-every", str(spec.get("checkpoint_every", 1000)),
        "--checkpoint-dir", "gs://" + spec["checkpoint_dir"],
        "--gradient-accumulation-steps", str(spec.get("accumulation", 1)),
    ] + (["--use-packing"] if spec["route"] == "packed" else []) + spec.get("extra_args", [])
    sys.argv = argv

    # to_global's first call receives the freshly initialized state: keep it to see that params move.
    real_to_global = tm.to_global

    def to_global_recording(tree, sharding):
        if "init" not in record:
            record["init"] = True
            np.savez(output / f"init_{process_id}.npz", **_flat_numpy(tree.params))
        return real_to_global(tree, sharding)

    tm.to_global = to_global_recording

    state = tm.main(tm.parse_args())

    np.savez(output / f"params_{process_id}.npz", **_flat_numpy(state.params))
    leaves = jax.tree_util.tree_leaves(state)
    (output / f"result_{process_id}.json").write_text(json.dumps({
        "process_index": jax.process_index(),
        "process_count": jax.process_count(),
        "device_count": jax.device_count(),
        "local_device_count": jax.local_device_count(),
        "digests": _digest_per_device(state.params),
        "all_replicated": all(leaf.is_fully_replicated for leaf in leaves),
        "state_step": int(state.step),
        "first_rows": record["first_rows"],
        "restored": record["restored"],
    }))


def legacy_local_worker(spec):
    """Negative control: the step as it ran before the fix, on host-local arrays."""
    import jax
    import jax.numpy as jnp

    jax.distributed.initialize(
        coordinator_address=f"localhost:{spec['port']}",
        num_processes=spec["num_processes"],
        process_id=spec["process_id"],
    )
    _stub_wandb_if_missing()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import optax
    import train_minimal as tm
    from flax.training import train_state

    from mmbjax.configuration_mmbbert import MMBertConfig
    from mmbjax.heads.masked_lm import ModernBertForMaskedLM

    config = MMBertConfig(vocab_size=128, hidden_size=16, num_hidden_layers=2, num_attention_heads=2,
                          intermediate_size=32, max_position_embeddings=64, pad_token_id=0,
                          compute_dtype="float32", param_dtype="float32")
    model = ModernBertForMaskedLM(config, dtype=jnp.float32, param_dtype=jnp.float32)
    init_rng, dropout_rng = jax.random.split(jax.random.PRNGKey(42))
    dummy = jnp.ones((1, MAX_LENGTH), dtype=jnp.int32)
    params = model.init(init_rng, input_ids=dummy, attention_mask=dummy, deterministic=True)["params"]
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optax.adamw(LEARNING_RATE))
    step_fn = jax.jit(tm.train_step, static_argnums=(4,))
    for step in range(3):
        rows = [step * 4 + jax.process_index() * 2 + j for j in range(2)]
        batch = {k: jnp.array(v) for k, v in _row_arrays(rows, with_packing_fields=False).items()}
        state, _metrics, dropout_rng = step_fn(state, batch, dropout_rng, LEARNING_RATE, False)
    Path(spec["output"], f"result_{jax.process_index()}.json").write_text(json.dumps({
        "digests": _digest_per_device(state.params),
    }))


# --------------------------------------------------------------------------- test side


def _free_port():
    with socket.socket() as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


def _start(tmp_path, name, num_processes, devices_per_process, worker="worker_main", **spec):
    """Start the workers; return the output directory and the (process, log file) pairs."""
    output = tmp_path / name
    output.mkdir()
    port = _free_port()
    spec.setdefault("checkpoint_dir", str(output / "ckpt"))
    env = {
        **os.environ,
        "JAX_PLATFORMS": "cpu",
        "OMP_NUM_THREADS": "1",
        "XLA_FLAGS": f"--xla_force_host_platform_device_count={devices_per_process}",
        "TOKENIZERS_PARALLELISM": "false",
        "HF_DATASETS_OFFLINE": "1",
        "HF_HUB_OFFLINE": "1",
    }
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    procs = []
    for process_id in range(num_processes):
        full = {**spec, "num_processes": num_processes, "process_id": process_id, "port": port,
                "output": str(output), "worker": worker}
        log = open(output / f"stdout_{process_id}.log", "w")
        procs.append((subprocess.Popen(
            [sys.executable, __file__, "--spec", json.dumps(full)],
            stdout=log, stderr=subprocess.STDOUT, env={**env, "TRACKIO_DIR": str(output / f"trackio_{process_id}")}), log))
    return output, procs


def _kill_workers(procs):
    # SIGKILL, because a worker left alive aborts on the runtime's fatal error once its peer is
    # gone, and on macOS each abort opens a "Python quit unexpectedly" dialog.
    for proc, log in procs:
        if proc.poll() is None:
            proc.kill()
        proc.wait()
        log.close()


def _wait_for_workers(procs):
    """Exit codes, as soon as all workers ended or one failed; a worker still running then gets None."""
    deadline = time.time() + WORKER_TIMEOUT
    while time.time() < deadline:
        codes = [proc.poll() for proc, _ in procs]
        if None not in codes or any(code not in (None, 0) for code in codes):
            return codes
        time.sleep(0.2)
    return [proc.poll() for proc, _ in procs]


def _run(tmp_path, name, num_processes, devices_per_process, **spec):
    """Start the workers, wait, return (per-process results, per-process stdout)."""
    output, procs = _start(tmp_path, name, num_processes, devices_per_process, **spec)
    try:
        codes = _wait_for_workers(procs)
    finally:
        _kill_workers(procs)
    failures = [
        f"worker {i} of {name} exited with {code}:\n{(output / f'stdout_{i}.log').read_text()[-3000:]}"
        for i, code in enumerate(codes) if code != 0
    ]
    assert not failures, "\n".join(failures)
    results = [json.loads((output / f"result_{i}.json").read_text()) for i in range(num_processes)]
    logs = [(output / f"stdout_{i}.log").read_text() for i in range(num_processes)]
    return results, logs


def _params(tmp_path, name, kind="params", process_id=0):
    with np.load(tmp_path / name / f"{kind}_{process_id}.npz") as f:
        return {k: f[k] for k in f.files}


def _max_abs_diff(a, b):
    assert a.keys() == b.keys()
    return max(float(np.max(np.abs(a[k] - b[k]))) for k in a)


def _assert_identical_across_hosts_and_chips(results):
    """(b): one digest for every device of every process, so all chips hold the same bytes."""
    digests = [digest for result in results for digest in result["digests"].values()]
    assert len(digests) == results[0]["device_count"]
    assert len(set(digests)) == 1, digests
    assert all(result["all_replicated"] for result in results)


def _assert_topologies_agree(tmp_path, name_a, name_b):
    a = _params(tmp_path, name_a)
    b = _params(tmp_path, name_b)
    diff = _max_abs_diff(a, b)
    moved = _max_abs_diff(_params(tmp_path, name_a), _params(tmp_path, name_a, kind="init"))
    print(f"{name_a} vs {name_b}: max abs diff {diff:.3e}, largest move from init {moved:.3e}")
    assert diff <= PARAM_ABS_TOL, f"max abs param diff {diff} > {PARAM_ABS_TOL}"
    # Without this a run that didn't train would pass
    assert moved > 100 * PARAM_ABS_TOL, f"params hardly moved: {moved}"


def test_one_process_and_two_processes_train_the_same_unpacked(tmp_path):
    spec = dict(route="unpacked", data="fixed", max_steps=5, per_device=2)
    one, one_logs = _run(tmp_path, "one", 1, 4, **spec)
    two, two_logs = _run(tmp_path, "two", 2, 2, **spec)
    assert (one[0]["process_count"], one[0]["device_count"]) == (1, 4)
    assert [(r["process_count"], r["device_count"], r["local_device_count"]) for r in two] == [(2, 4, 2)] * 2
    _assert_identical_across_hosts_and_chips(one)
    _assert_identical_across_hosts_and_chips(two)
    _assert_topologies_agree(tmp_path, "one", "two")
    # The batch shape printed is the global one in both runs
    assert f"Batch shape: (8, {MAX_LENGTH})" in one_logs[0]
    assert f"Batch shape: (8, {MAX_LENGTH})" in two_logs[0]
    # Each host read its own part of the global batch
    assert two[0]["first_rows"] == [0, 1, 2, 3]
    assert two[1]["first_rows"] == [4, 5, 6, 7]
    assert one[0]["first_rows"] == list(range(8))


def test_one_process_and_two_processes_train_the_same_packed(tmp_path):
    spec = dict(route="packed", data="fixed", max_steps=5, per_device=2)
    one, one_logs = _run(tmp_path, "one", 1, 4, **spec)
    two, two_logs = _run(tmp_path, "two", 2, 2, **spec)
    _assert_identical_across_hosts_and_chips(one)
    _assert_identical_across_hosts_and_chips(two)
    _assert_topologies_agree(tmp_path, "one", "two")
    assert f"Batch shape: (8, {MAX_LENGTH})" in one_logs[0]
    assert f"Batch shape: (8, {MAX_LENGTH})" in two_logs[0]


def test_gradient_accumulation_across_processes(tmp_path):
    spec = dict(route="unpacked", data="fixed", max_steps=3, per_device=2, accumulation=2)
    one, _ = _run(tmp_path, "one", 1, 4, **spec)
    two, _ = _run(tmp_path, "two", 2, 2, **spec)
    _assert_identical_across_hosts_and_chips(two)
    _assert_topologies_agree(tmp_path, "one", "two")
    # TrainState.step counts micro-steps with MultiSteps
    assert one[0]["state_step"] == two[0]["state_step"] == 6


def test_single_device_run_still_works_and_agrees(tmp_path):
    """The whole batch on one device against the same batch split over four."""
    single, logs = _run(tmp_path, "single", 1, 1, route="unpacked", data="fixed", max_steps=5, per_device=8)
    split, _ = _run(tmp_path, "split", 1, 4, route="unpacked", data="fixed", max_steps=5, per_device=2)
    assert single[0]["device_count"] == 1
    assert f"Batch shape: (8, {MAX_LENGTH})" in logs[0]
    _assert_topologies_agree(tmp_path, "single", "split")


def test_checkpoint_roundtrip_on_two_processes(tmp_path):
    ckpt = str(tmp_path / "shared_ckpt")
    spec = dict(route="unpacked", data="fixed", per_device=2, checkpoint_every=2, checkpoint_dir=ckpt)
    saved, _ = _run(tmp_path, "save", 2, 2, max_steps=4, **spec)
    _assert_identical_across_hosts_and_chips(saved)
    assert saved[0]["restored"] is None

    resumed, logs = _run(tmp_path, "resume", 2, 2, max_steps=6, **spec)
    saved_digest = next(iter(saved[0]["digests"].values()))
    for result in resumed:
        # (c): the restored bytes on each chip of each host are the bytes that were saved
        assert result["restored"]["step"] == 4
        assert result["restored"]["state_step"] == 4
        assert result["restored"]["replicated"]
        assert set(result["restored"]["digests"].values()) == {saved_digest}
    for process_id in range(2):
        assert _max_abs_diff(_params(tmp_path, "save", process_id=process_id),
                             _params(tmp_path, "resume", kind="restored", process_id=process_id)) == 0.0
    _assert_identical_across_hosts_and_chips(resumed)
    assert resumed[0]["state_step"] == 6
    # Each host continues exactly after the rows of the 4 steps it already trained on
    assert "Will skip 16 examples per process" in logs[0]
    assert resumed[0]["first_rows"] == [32, 33, 34, 35]
    assert resumed[1]["first_rows"] == [36, 37, 38, 39]


def test_checkpoint_from_two_processes_restores_on_one(tmp_path):
    """Another topology, because a checkpoint must not tie a run to one VM shape."""
    ckpt = str(tmp_path / "shared_ckpt")
    spec = dict(route="unpacked", data="fixed", per_device=2, checkpoint_every=2, checkpoint_dir=ckpt)
    saved, _ = _run(tmp_path, "save", 2, 2, max_steps=4, **spec)
    resumed, _ = _run(tmp_path, "resume", 1, 4, max_steps=5, **spec)
    assert set(resumed[0]["restored"]["digests"].values()) == set(saved[0]["digests"].values())


def test_numpy_checkpoint_from_before_the_fix_restores(tmp_path):
    """Checkpoints of the published runs were written as NumPy arrays."""
    ckpt = str(tmp_path / "old_ckpt")
    spec = dict(route="unpacked", data="fixed", per_device=2, checkpoint_every=2, checkpoint_dir=ckpt)
    saved, _ = _run(tmp_path, "save", 1, 4, max_steps=2, legacy_numpy_save=True, **spec)
    resumed, _ = _run(tmp_path, "resume", 2, 2, max_steps=3, **spec)
    for result in resumed:
        assert result["restored"]["replicated"]
        assert set(result["restored"]["digests"].values()) == set(saved[0]["digests"].values())


def test_real_data_pipeline_unpacked(tmp_path):
    """Real sharding and unseeded masking per host: bytes must still agree across hosts."""
    results, _ = _run(tmp_path, "run", 2, 2, route="unpacked", data="real_pipeline", max_steps=4, per_device=2,
                      rows_per_shard=50)
    _assert_identical_across_hosts_and_chips(results)


def test_real_data_pipeline_packed_on_a_short_dataset(tmp_path):
    """Far fewer rows than the 8 steps need: the loader starts another pass, no host may fall out of step."""
    results, logs = _run(tmp_path, "run", 2, 2, route="packed", data="real_pipeline", max_steps=8, per_device=2,
                         rows_per_shard=3)
    _assert_identical_across_hosts_and_chips(results)
    assert results[0]["state_step"] == 8
    # A pass that ends is in the log of each host
    for process_id, log in enumerate(logs):
        assert f"Completed epoch 1 (shard {process_id}/2)" in log


def _raw_batches(tmp_path, name, process_id):
    return (tmp_path / name / f"raw_batches_{process_id}.txt").read_text().split()


def test_packed_resume_after_a_kill_continues_with_the_next_batch(tmp_path):
    """Real sharding, shuffle, packing and Orbax; every host killed between two checkpoints."""
    spec = dict(route="packed", data="real_pipeline", per_device=2, rows_per_shard=500, checkpoint_every=3)
    _run(tmp_path, "whole", 2, 2, max_steps=8, checkpoint_dir=str(tmp_path / "ckpt_whole"), **spec)

    ckpt = str(tmp_path / "ckpt")
    # Killed while reading the data of the 5th step, after the checkpoint of step 3 and with step 4 trained
    output, procs = _start(tmp_path, "killed", 2, 2, max_steps=8, checkpoint_dir=ckpt, wait_to_be_killed_at_batch=5, **spec)
    try:
        deadline = time.time() + WORKER_TIMEOUT
        ready = [output / f"ready_to_be_killed_{i}" for i in range(2)]
        while time.time() < deadline and not all(path.exists() for path in ready):
            assert all(proc.poll() is None for proc, _ in procs), [(output / f"stdout_{i}.log").read_text()[-2000:] for i in range(2)]
            time.sleep(0.2)
    finally:
        # Both at once: a worker that sees its peer gone aborts, and on macOS that opens a dialog
        _kill_workers(procs)
    assert all(path.exists() for path in ready)

    resumed, logs = _run(tmp_path, "resumed", 2, 2, max_steps=8, checkpoint_dir=ckpt, **spec)
    _assert_identical_across_hosts_and_chips(resumed)
    assert resumed[0]["restored"]["step"] == 3
    assert resumed[0]["state_step"] == 8
    for process_id in range(2):
        assert "Data state restored" in logs[process_id]
        whole = _raw_batches(tmp_path, "whole", process_id)
        assert _raw_batches(tmp_path, "killed", process_id) == whole[:5]
        # The 4th batch again because step 4 wasn't in a checkpoint; from there on what the
        # uninterrupted run read, byte for byte
        assert _raw_batches(tmp_path, "resumed", process_id) == whole[3:8]
        assert len(set(whole)) == 8, "this data must differ per batch or the comparison says nothing"


def test_packed_resume_without_data_state_refuses_to_repeat_data(tmp_path):
    ckpt = str(tmp_path / "ckpt")
    spec = dict(route="packed", data="real_pipeline", per_device=2, rows_per_shard=500, checkpoint_every=2, checkpoint_dir=ckpt)
    _run(tmp_path, "save", 2, 2, max_steps=2, **spec)
    state_dir = Path(ckpt + "_data_state")
    assert (state_dir / "2" / "process_1.json").exists()
    (state_dir / "2" / "process_1.json").unlink()

    output, procs = _start(tmp_path, "refused", 2, 2, max_steps=4, **spec)
    try:
        codes = _wait_for_workers(procs)
    finally:
        _kill_workers(procs)
    assert codes[1] == 1
    assert "this resume would train on the same data again" in (output / "stdout_1.log").read_text()

    resumed, _ = _run(tmp_path, "allowed", 2, 2, max_steps=4, extra_args=["--allow-data-restart"], **spec)
    assert resumed[0]["state_step"] == 4


def test_stop_at_step_saves_there_for_a_resume_on_the_same_schedule(tmp_path):
    ckpt = str(tmp_path / "ckpt")
    spec = dict(route="packed", data="real_pipeline", per_device=2, rows_per_shard=500, max_steps=6, checkpoint_dir=ckpt)
    stopped, _ = _run(tmp_path, "stopped", 2, 2, extra_args=["--stop-at-step", "3"], **spec)
    assert stopped[0]["state_step"] == 3
    # Step 3 is no multiple of --checkpoint-every; the save at the end happens anyway
    resumed, _ = _run(tmp_path, "resumed", 2, 2, **spec)
    assert resumed[0]["restored"]["step"] == 3
    assert resumed[0]["state_step"] == 6


def _write_eval_parquet(path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    rng = np.random.RandomState(1)
    texts = [" ".join(f"w{w}" for w in rng.randint(0, VOCAB_WORDS, size=rng.randint(3, 11))) for _ in range(400)]
    pq.write_table(pa.table({"text": texts}), path)


def test_held_out_loss_is_logged_and_reproducible(tmp_path):
    eval_file = tmp_path / "held_out.parquet"
    _write_eval_parquet(eval_file)
    spec = dict(route="packed", data="real_pipeline", per_device=2, rows_per_shard=500, max_steps=4,
                extra_args=["--eval-file", str(eval_file), "--eval-every", "2", "--eval-batches", "5"])

    def eval_lines(name):
        _, logs = _run(tmp_path, name, 2, 2, **spec)
        return [line for line in logs[0].splitlines() if line.startswith("EVAL")]

    first = eval_lines("first")
    assert len(first) == 2 and "step=2" in first[0] and "step=4" in first[1]
    assert first[0] != first[1], "two evaluations of a model in training gave the same loss"
    # Same data, same seeds, same masks: this is what a learning-rate comparison stands on
    assert eval_lines("second") == first


def test_trackio_on_the_coordinator_with_per_chip_metrics_and_sync(tmp_path):
    import pytest

    pytest.importorskip("trackio")
    import sqlite3

    sync_dir = tmp_path / "synced"
    eval_file = tmp_path / "held_out.parquet"
    _write_eval_parquet(eval_file)
    results, logs = _run(tmp_path, "run", 2, 2, route="packed", data="real_pipeline", max_steps=10, per_device=2,
                         rows_per_shard=500, tracker="trackio", extra_args=[
                             "--wandb-project", "proj", "--wandb-run-name", "r1", "--dataset-num-rows", "8000",
                             "--system-metrics-every", "5", "--tracking-sync-dir", str(sync_dir), "--tracking-sync-every", "1",
                             "--eval-file", str(eval_file), "--eval-every", "5", "--eval-batches", "2",
                         ])
    assert "Exception in thread" not in logs[0], "a dead writer thread of Trackio loses the logs after it"
    _assert_identical_across_hosts_and_chips(results)
    assert (tmp_path / "run" / "trackio_0" / "proj.db").exists()
    assert not (tmp_path / "run" / "trackio_1" / "proj.db").exists(), "only the coordinator tracks"

    rows = sqlite3.connect(sync_dir / "proj.db").execute("SELECT step, metrics FROM metrics").fetchall()
    logged = {step: json.loads(metrics) for step, metrics in rows if "train/loss" in json.loads(metrics)}
    system = {step: json.loads(metrics) for step, metrics in rows if "data/passes" in json.loads(metrics)}
    assert sorted(system) == [5, 10]
    # Logged after an evaluation, so these only arrive if its log didn't stop the writer
    assert sorted(step for step, metrics in rows if "eval/loss" in json.loads(metrics)) == [5, 10]
    # Every chip of every host, and the data position of every host
    for host in range(2):
        for chip in range(2):
            assert f"system/host{host}_chip{chip}_bytes_in_use" in system[10]
            assert f"system/host{host}_chip{chip}_duty_cycle" in system[10]
        assert system[10][f"data/host{host}_documents_read"] > 0
        assert system[10][f"performance/host{host}_input_seconds_per_step"] > 0
        assert system[10][f"system/host{host}_peak_rss_bytes"] > 0
    assert 0 < system[10]["data/passes"] < 1

    # Real tokens: padding excluded, so fewer than rows x length per step
    assert 0 < logged[10]["performance/tokens_per_sec"]
    assert logged[10]["performance/step_seconds"] > 0
    per_second_with_padding = 8 * MAX_LENGTH / logged[10]["performance/step_seconds"]
    assert logged[10]["performance/tokens_per_sec"] < per_second_with_padding
    # Per-layer norms reach the tracker on the packed route, and the logged step has them
    assert "gradients/encoder_0_attention" in logged[10]
    assert "gradients/encoder_1_ffn" in logged[10]
    assert "gradients/lm_head" in logged[10]


def test_exhausted_shard_stops_that_host_with_a_message(tmp_path):
    """Host 1 runs out after 2 of 6 steps; host 0 is then stuck in a collective and gets killed here."""
    output, procs = _start(tmp_path, "run", 2, 2, route="unpacked", data="fixed", max_steps=6, per_device=2,
                           exhaust_process=1, exhaust_after=2)
    try:
        codes = _wait_for_workers(procs)
    finally:
        _kill_workers(procs)
    assert codes[1] == 1
    assert codes[0] is None, "host 0 cannot finish 6 steps without host 1"
    assert "Dataset shard of process 1 ran out at optimizer_step 2 of 6" in (output / "stdout_1.log").read_text()


def test_hash_check_detects_the_bug_this_fix_removes(tmp_path):
    """Negative control: host-local training on different rows must show up as different hashes."""
    results, _ = _run(tmp_path, "legacy", 2, 2, worker="legacy_local_worker", max_steps=3, per_device=1)
    assert next(iter(results[0]["digests"].values())) != next(iter(results[1]["digests"].values()))


def _exit_if_parent_dies():
    """An orphaned worker would abort on the runtime's fatal error after its 5-minute shutdown timeout."""
    parent = os.getppid()

    def watch():
        while os.getppid() == parent:
            time.sleep(1)
        os._exit(1)

    threading.Thread(target=watch, daemon=True).start()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    worker_spec = json.loads(parser.parse_args().spec)
    _exit_if_parent_dies()
    try:
        {"worker_main": worker_main, "legacy_local_worker": legacy_local_worker}[worker_spec["worker"]](worker_spec)
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        # Not a normal exit: atexit would call jax.distributed.shutdown(), whose barrier waits 5
        # minutes for a peer that is stuck in a collective, and then aborts both processes.
        # The exit code tells the parent, which kills the peer.
        os._exit(1)
    import jax

    if jax.distributed.is_initialized():
        jax.distributed.shutdown()
