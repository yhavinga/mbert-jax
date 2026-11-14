#!/usr/bin/env python3
"""Checks for translate.py with other MaxText models (translate_models.yml).

    JAX_PLATFORMS=cpu python translate_checks.py decode
    JAX_PLATFORMS=cpu python translate_checks.py mesh
    python translate_checks.py prompts --old <translate.py before translate_models.yml> --pairs pairs.parquet \\
        --tokenizer <TranslateGemma's tokenizer directory>

decode: does MaxEngine's decode of a Gemma 4 model (5 sliding-window layers, then a global one with its own head size,
fewer KV heads and K = V) give the logits of a train-mode forward over the same tokens? A tiny model with random
weights on the CPU, after MaxText's tests/unit/sliding_window_decode_test.py (upstream main, which tests Gemma 2 and
GPT-OSS): prefill a prompt, decode greedily, then one forward pass over prompt and completion; every decode step's
logits must equal the forward pass's at that position (relative max-abs difference <= 1e-3). Cases, with W the
sliding window:
- within: prompt and completion inside W.
- past_window: prefill and decode part each shorter than W, prompt and completion longer (translate.py's classes on
  Gemma 4: W 1024, prefill <= 640, decode <= 832).
- prefill_past_window: a prefill part longer than W (a prompt with earlier turns as context).
- decode_past_window: a decode part longer than W (the case of MaxText's commit 2bbe8006c, 8 Oct 2026).
Prints a line per case and exits 1 when one fails.

mesh: may translate.py put Gemma 4 on 8 chips of one process (a v6e-8, 14 Nov)? Tensor parallelism splits the KV
heads, and Gemma 4 31B's global layers have 4, so the mesh of 8 is tensor 4 x context 2 (translate.mesh_of). A tiny
Gemma 4 with the same ratio of heads (16 query, 8 KV, 4 global KV) prefills two prompts into two slots and decodes 12
steps, each layout in a process of its own with that many CPU devices: one device (the reference), tensor 4 (part A's
4 chips) and tensor 4 x context 2. Every logit of a layout must equal the reference's (relative max-abs difference <=
1e-3). Prints a line per layout and exits 1 when one fails.

prompts: the prompt ids of TranslateGemma for every text of the pilot pairs (nl -> en from the Dutch, en -> nl from
the English), from translate.py before translate_models.yml (--old) and from this one: they must be equal.
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for _path in (HERE.parent / "flan-nl", HERE):
    sys.path.insert(0, str(_path))

TINY = {
    "model_name": "gemma4-31b",
    "override_model_config": True,
    "base_num_decoder_layers": 6,
    "base_emb_dim": 128,
    "base_num_query_heads": 4,
    "base_num_kv_heads": 2,
    "head_dim": 32,
    "global_head_dim": 64,
    "global_num_kv_heads": 1,
    "base_mlp_dim": 256,
    "vocab_size": 128,
    "use_multimodal": False,
    "per_device_batch_size": 1,
    "scan_layers": False,
    "attention": "dot_product",
    "dtype": "float32",
    "weight_dtype": "float32",
    "matmul_precision": "highest",
    "decode_sampling_strategy": "greedy",
    "enable_checkpointing": False,
    "async_checkpointing": False,
    "skip_jax_distributed_system": True,
    "log_config": False,
    "run_name": "decode_check",
}
# name: (sliding window, max_prefill_predict_length, max_target_length, prompt tokens, decode steps)
CASES = {
    "within": (16, 12, 24, 5, 8),
    "past_window": (16, 12, 24, 10, 11),
    "prefill_past_window": (8, 16, 24, 12, 7),
    "decode_past_window": (8, 16, 64, 5, 30),
}
REL_TOL = 1e-3
# mesh: layout -> (CPU devices, translate.mesh_of(chips, context))
MESH_LAYOUTS = {"one": (1, 1), "tensor4": (4, 1), "tensor4-context2": (8, 2)}
MESH_HEADS = {"base_num_query_heads": 16, "base_num_kv_heads": 8, "global_num_kv_heads": 4}


def decode_case(window, max_prefill, max_target, prompt_len, steps, work_dir):
    """Per decode step, max |decode logits - forward logits| / max |forward logits|."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from flax import nnx
    from flax.linen import partitioning as nn_partitioning
    from maxtext.common.common_types import MODEL_MODE_PREFILL, MODEL_MODE_TRAIN
    from maxtext.configs import pyconfig
    from maxtext.inference.maxengine import maxengine
    from maxtext.utils import maxtext_utils, model_creation_utils

    cfg = pyconfig.initialize([""], **TINY, sliding_window_size=window, max_prefill_predict_length=max_prefill,
                              max_target_length=max_target, base_output_directory=f"{work_dir}/maxtext_output")
    mesh = jax.sharding.Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
    with nn_partitioning.axis_rules(cfg.logical_axis_rules), mesh:
        model = model_creation_utils.create_model(cfg, mesh, model_mode=MODEL_MODE_PREFILL, rngs=nnx.Rngs(params=0))
    _, params_state, _ = nnx.split(model, nnx.Param, ...)
    # Default init is close to degenerate for attention; random weights make the window visible (as upstream)
    leaves, treedef = jax.tree.flatten(params_state)
    keys = jax.random.split(jax.random.PRNGKey(7), len(leaves))
    leaves = [jax.random.normal(k, x.shape, x.dtype) * 0.2 for k, x in zip(keys, leaves)]
    params_state = jax.tree.unflatten(treedef, leaves)

    engine = maxengine.MaxEngine(cfg, jax.devices())
    params = engine.load_params(params=params_state)
    prompt = [int(x) for x in np.random.default_rng(prompt_len).integers(3, cfg.vocab_size, prompt_len)]
    padded = jnp.asarray(prompt + [0] * (max_prefill - prompt_len), dtype=jnp.int32)
    prefix, first = engine.prefill(params=params, padded_tokens=padded, true_length=prompt_len)
    tokens = prompt + [int(first.data[0, 0])]
    decode_logits = [np.asarray(prefix["logits"], np.float64).reshape(-1)[-cfg.vocab_size:]]
    state = engine.insert(prefix, engine.init_decode_state(), slot=0)
    for _ in range(steps):
        state, result = engine.generate(params, state)
        decode_logits.append(np.asarray(state["logits"][0], np.float64).reshape(-1)[-cfg.vocab_size:])
        tokens.append(int(result.data[0, 0]))

    with nn_partitioning.axis_rules(cfg.logical_axis_rules), mesh:
        train_model = model_creation_utils.create_model(cfg, mesh, model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(params=0))
    nnx.update(train_model, params_state)
    seq, batch = max_target, cfg.micro_batch_size_to_train_on
    ids = jnp.tile(jnp.asarray([tokens + [0] * (seq - len(tokens))], dtype=jnp.int32), (batch, 1))
    positions = jnp.tile(jnp.arange(seq, dtype=jnp.int32)[None], (batch, 1))
    segments = jnp.tile(jnp.asarray([[1] * len(tokens) + [0] * (seq - len(tokens))], dtype=jnp.int32), (batch, 1))
    with nn_partitioning.axis_rules(cfg.logical_axis_rules), mesh:
        ref = train_model(ids, positions, decoder_segment_ids=segments, enable_dropout=False,
                          model_mode=MODEL_MODE_TRAIN)
    ref = np.asarray(ref[0] if isinstance(ref, tuple) else ref, np.float64)[0]
    return [float(np.max(np.abs(d - ref[prompt_len - 1 + k])) / np.max(np.abs(ref[prompt_len - 1 + k])))
            for k, d in enumerate(decode_logits)]


def decode(args):
    import translate

    patches = translate.patch_maxtext()
    print("MAXTEXT " + json.dumps({**translate.maxtext_source(), "patches": patches}), flush=True)
    failed = []
    for name, (window, max_prefill, max_target, prompt_len, steps) in CASES.items():
        rels = decode_case(window, max_prefill, max_target, prompt_len, steps, args.work_dir)
        worst = max(range(len(rels)), key=rels.__getitem__)
        # The first step past the window: the position prompt_len - 1 + k reaches W at k = W - prompt_len + 1
        first_bad = next((k for k, r in enumerate(rels) if r > REL_TOL), None)
        ok = rels[worst] <= REL_TOL
        print("CASE " + json.dumps({
            "case": name, "window": window, "max_prefill": max_prefill, "max_target": max_target,
            "prompt": prompt_len, "steps": steps, "last_position": prompt_len + steps - 1, "ok": ok,
            "worst_step": worst, "worst_rel": float(f"{rels[worst]:.3g}"), "first_step_over_tol": first_bad}),
            flush=True)
        if not ok:
            failed.append(name)
    print("DECODE " + json.dumps({"failed": failed}), flush=True)
    return 1 if failed else 0


def mesh_logits(chips, context, work_dir, window=8, max_prefill=16, max_target=40, steps=12):
    """The logits of two prompts prefilled into two slots and 12 decode steps, on this process's devices"""
    import jax
    import jax.numpy as jnp
    import numpy as np
    import translate
    from flax import nnx
    from flax.linen import partitioning as nn_partitioning
    from maxtext.common.common_types import MODEL_MODE_PREFILL
    from maxtext.configs import pyconfig
    from maxtext.inference.maxengine import maxengine
    from maxtext.utils import maxtext_utils, model_creation_utils

    if jax.device_count() != chips:
        raise SystemExit(f"needs {chips} devices, has {jax.device_count()}")
    cfg = pyconfig.initialize([""], **dict(TINY, **MESH_HEADS, per_device_batch_size=2 / chips),
                              **translate.mesh_of(chips, context), sliding_window_size=window,
                              max_prefill_predict_length=max_prefill, max_target_length=max_target,
                              base_output_directory=f"{work_dir}/maxtext_output")
    mesh = jax.sharding.Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
    with nn_partitioning.axis_rules(cfg.logical_axis_rules), mesh:
        model = model_creation_utils.create_model(cfg, mesh, model_mode=MODEL_MODE_PREFILL, rngs=nnx.Rngs(params=0))
    _, params_state, _ = nnx.split(model, nnx.Param, ...)
    leaves, treedef = jax.tree.flatten(params_state)
    keys = jax.random.split(jax.random.PRNGKey(7), len(leaves))
    params_state = jax.tree.unflatten(treedef, [jax.random.normal(k, x.shape, x.dtype) * 0.2
                                                for k, x in zip(keys, leaves)])
    engine = maxengine.MaxEngine(cfg, jax.devices())
    params = engine.load_params(params=params_state)
    state, out = engine.init_decode_state(), []
    for slot, length in ((0, 10), (1, 13)):
        prompt = [int(x) for x in np.random.default_rng(length).integers(3, cfg.vocab_size, length)]
        padded = jnp.asarray(prompt + [0] * (max_prefill - length), dtype=jnp.int32)
        prefix, _ = engine.prefill(params=params, padded_tokens=padded, true_length=length)
        out.append(np.asarray(prefix["logits"], np.float64).reshape(-1)[-cfg.vocab_size:])
        state = engine.insert(prefix, state, slot=slot)
    for _ in range(steps):
        state, _ = engine.generate(params, state)
        out.append(np.asarray(state["logits"], np.float64).reshape(2, -1)[:, -cfg.vocab_size:].reshape(-1))
    return np.concatenate(out)


def mesh_one(args):
    import numpy as np
    import translate

    translate.patch_maxtext()
    devices, context = MESH_LAYOUTS[args.layout]
    np.save(args.out, mesh_logits(devices, context, args.work_dir))
    return 0


def mesh(args):
    import os
    import subprocess

    import numpy as np

    Path(args.work_dir).mkdir(parents=True, exist_ok=True)
    logits, failed = {}, []
    for layout, (devices, context) in MESH_LAYOUTS.items():
        out = Path(args.work_dir) / f"mesh-{layout}.npy"
        env = dict(os.environ, JAX_PLATFORMS="cpu", XLA_FLAGS=f"--xla_force_host_platform_device_count={devices}")
        run = subprocess.run([sys.executable, __file__, "mesh-one", "--layout", layout, "--out", str(out),
                              "--work-dir", args.work_dir], env=env, capture_output=True, text=True)
        if run.returncode:
            print(f"MESH {json.dumps({'layout': layout, 'ok': False, 'error': run.stderr.strip()[-400:]})}", flush=True)
            failed.append(layout)
            continue
        logits[layout] = np.load(out)
    ref = logits.get("one")
    for layout, x in logits.items():
        if layout == "one" or ref is None:
            continue
        rel = float(np.max(np.abs(x - ref)) / np.max(np.abs(ref)))
        devices, context = MESH_LAYOUTS[layout]
        print("MESH " + json.dumps({"layout": layout, "devices": devices, "context": context, "ok": rel <= REL_TOL,
                                    "rel_to_one_device": float(f"{rel:.3g}")}), flush=True)
        if rel > REL_TOL:
            failed.append(layout)
    print("MESHES " + json.dumps({"failed": failed if ref is not None else ["one"] + failed}), flush=True)
    return 1 if failed or ref is None else 0


def prompts(args):
    import pyarrow.parquet as pq
    import translate
    from transformers import AutoTokenizer

    spec = importlib.util.spec_from_file_location("translate_old", args.old)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    pairs = pq.read_table(args.pairs, columns=["nl", "en"]).to_pydict()
    counts = {}
    for source, target, column in (("nl", "en", "nl"), ("en", "nl", "en")):
        texts = [t for t in pairs[column] if t]
        before = old.PromptCodec(tok, source, target)
        after = translate.ModelPrompts(tok, translate.MODELS[args.model], source, target)
        differ = [t for t in texts if before.encode(t) != after.encode(t)]
        after.check(texts[:5])
        counts[f"{source}-{target}"] = {"texts": len(texts), "differ": len(differ)}
        if differ:
            print(f"differs for {differ[0][:80]!r}", flush=True)
    same = {k: getattr(old, k) == getattr(translate, k) for k in ("LENGTH_CLASSES", "CLASS_ORDER", "EOS_IDS")}
    same["SLOTS"] = all(old.SLOTS[m] == translate.SLOTS[m] for m in old.SLOTS)
    print("PROMPTS " + json.dumps({"model": args.model, "counts": counts, "same": same}), flush=True)
    return 0 if all(c["differ"] == 0 for c in counts.values()) and all(same.values()) else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("decode", help="Gemma 4 decode against a forward pass, tiny model on the CPU")
    p.add_argument("--work-dir", default="/tmp/translate_checks")
    p = sub.add_parser("mesh", help="Gemma 4 on 1, 4 and 8 CPU devices (tensor 4 x context 2): the same logits")
    p.add_argument("--work-dir", default="/tmp/translate_checks")
    p = sub.add_parser("mesh-one", help="one layout of mesh, in a process with its devices")
    p.add_argument("--layout", required=True, choices=sorted(MESH_LAYOUTS))
    p.add_argument("--out", required=True)
    p.add_argument("--work-dir", default="/tmp/translate_checks")
    p = sub.add_parser("prompts", help="TranslateGemma's prompt ids before and after translate_models.yml")
    p.add_argument("--old", required=True)
    p.add_argument("--pairs", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--model", default="gemma3-27b")
    args = parser.parse_args(argv)
    return {"decode": decode, "prompts": prompts, "mesh": mesh, "mesh-one": mesh_one}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
