#!/usr/bin/env python3
"""The translator G4-8-s140 for Hugging Face (protocol-g4.md §26): its MaxText LoRA adapter (sft_lora.py's adapter.npz
and adapter.json) as a PEFT adapter for google/gemma-4-31B-it in transformers, and that adapter merged into the base
model's bf16 weights (the input of llama.cpp's convert_hf_to_gguf.py).

  convert  adapter.npz + adapter.json -> adapter_model.safetensors (float32, as trained) + adapter_config.json. Each
           MaxText module decoder/layers_<i>/<type> becomes model.language_model.layers.<i>.<hf name> of
           Gemma4ForConditionalGeneration (what AutoModelForCausalLM loads for the base model). MaxText's kernels are the
           HF weights transposed and split per head (MaxText 0.2.5's Gemma 4 hook: kernel = W.T.reshape(shape)), and
           qwix's LoRA adds x.A.B.scale with A over the kernel's contracted axes and B over the rest (translate.lora_delta),
           so with A2 = A as (in, rank) and B2 = B as (rank, out): PEFT's lora_A = A2.T (rank, in), lora_B = B2.T
           (out, rank), and PEFT's scaling lora_alpha / r is the adapter's scale. With --base-dir every module's
           (out, in) is checked against the base model's weight in its safetensors headers, and the modules against
           the base model's config: every layer's seven projections, the global (full_attention) layers without v_proj
           when attention_k_eq_v (their value is the key projection, in MaxText and in transformers).
  merge    the PEFT adapter added to the base model's weights: W + scaling * (lora_B @ lora_A) in float32, then bf16 (as
           translate.py --lora merges in MaxText), shard by shard with the base model's own index; the other files of
           the base model (config, tokenizer, chat template, processor) copied unchanged. Prints the relative norm of the
           update and the share of changed elements over the adapter's modules, which translate.py's LORA line gave for
           the MaxText merge of the same adapter.

Prints ids, counts and statistics only.
"""

import argparse
import hashlib
import json
import re
import shutil
import struct
import sys
from pathlib import Path

import numpy as np

BASE_REPO = "google/gemma-4-31B-it"
BASE_REVISION = "842da3794eaa0b77d5f08bae87a17459d91ff475"
LAYER_PREFIX = "model.language_model.layers."   # Gemma4ForConditionalGeneration's text decoder
PEFT_PREFIX = "base_model.model."               # PEFT's prefix of a saved adapter's keys
# MaxText module type (unscanned Gemma 4, run_sft.sh's regex) -> the projection's name in transformers' Gemma 4
HF_NAMES = {
    "self_attention/query": "self_attn.q_proj",
    "self_attention/key": "self_attn.k_proj",
    "self_attention/value": "self_attn.v_proj",
    "self_attention/out": "self_attn.o_proj",
    "mlp/wi_0": "mlp.gate_proj",
    "mlp/wi_1": "mlp.up_proj",
    "mlp/wo": "mlp.down_proj",
}
MT_MODULE = re.compile(r"^decoder/layers_(\d+)/(" + "|".join(re.escape(k) for k in HF_NAMES) + r")$")
# PEFT matches a str target_modules with re.fullmatch on the module's name: the text decoder's seven projections, not
# the vision tower's (whose attention also has q_proj ... o_proj)
TARGET_MODULES = (r"model\.language_model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)"
                  r"|mlp\.(gate_proj|up_proj|down_proj))")
SUFFIXES = ("kernel_lora_a", "kernel_lora_b")
COPY_SKIP = re.compile(r"^(model.*\.safetensors(\.index\.json)?|\.gitattributes|README\.md|\.cache|\.eval_results)$")


def mt_modules(arrays):
    """{MaxText module path: (lora_a, lora_b)} from adapter.npz's arrays (keys <module>/kernel_lora_a and _b, the
    module path from its "decoder" component on); any other key or a module without both stops."""
    out = {}
    for key, value in arrays.items():
        module, _, name = key.rpartition("/")
        parts = module.split("/")
        if name not in SUFFIXES or "decoder" not in parts:
            raise SystemExit(f"adapter key {key}: not <...>/decoder/.../kernel_lora_a or _b")
        module = "/".join(parts[parts.index("decoder"):])
        if not MT_MODULE.match(module):
            raise SystemExit(f"adapter key {key}: module {module} is none of {sorted(HF_NAMES)}")
        out.setdefault(module, {})[name] = value
    incomplete = sorted(m for m, ab in out.items() if set(ab) != set(SUFFIXES))
    if incomplete:
        raise SystemExit(f"adapter modules without both lora_a and lora_b: {incomplete[:3]}")
    return {m: (ab["kernel_lora_a"], ab["kernel_lora_b"]) for m, ab in out.items()}


def hf_module(mt_module):
    """decoder/layers_12/self_attention/query -> model.language_model.layers.12.self_attn.q_proj"""
    m = MT_MODULE.match(mt_module)
    if not m:
        raise ValueError(f"{mt_module} is not a MaxText module of the adapter")
    return f"{LAYER_PREFIX}{int(m.group(1))}.{HF_NAMES[m.group(2)]}"


def peft_factors(a, b, rank):
    """(lora_A (rank, in), lora_B (out, rank)) in float32 from qwix's lora_a (contracted axes..., rank) and lora_b (rank,
    remaining axes...): their product A2 @ B2 is the MaxText kernel's update as (in, out), its transpose the HF
    weight's update as (out, in)."""
    if a.shape[-1] != rank or b.shape[0] != rank:
        raise ValueError(f"lora_a {a.shape} and lora_b {b.shape} are not of rank {rank}")
    a2 = np.reshape(np.asarray(a, dtype=np.float32), (-1, rank))
    b2 = np.reshape(np.asarray(b, dtype=np.float32), (rank, -1))
    return np.ascontiguousarray(a2.T), np.ascontiguousarray(b2.T)


def text_config(config):
    return config.get("text_config", config)


def layer_dims(t, i, full):
    """(head_dim, KV heads) of layer i: per_layer_config's override when the config has one (transformers 5.23 writes
    it), else for a full_attention layer global_head_dim and, with attention_k_eq_v, num_global_key_value_heads (as
    Gemma4TextConfig builds per_layer_config from them; config.json of the base model at its revision has these)."""
    per_layer = t.get("per_layer_config")
    if per_layer is not None:
        lc = per_layer.get(str(i), per_layer.get(i, {})) or {}
        return lc.get("head_dim", t["head_dim"]), lc.get("num_key_value_heads", t["num_key_value_heads"])
    if not full:
        return t["head_dim"], t["num_key_value_heads"]
    hd = t.get("global_head_dim") or t["head_dim"]
    kv = t.get("num_global_key_value_heads") if t.get("attention_k_eq_v") else None
    return hd, kv or t["num_key_value_heads"]


def expected_shapes(config):
    """{HF module name: (out, in)} of the base model's text decoder from its config: per layer q, k, (v), o, gate, up,
    down; a full_attention layer has the global head_dim and KV heads and, with attention_k_eq_v, no v_proj."""
    t = text_config(config)
    hidden, inter, heads = t["hidden_size"], t["intermediate_size"], t["num_attention_heads"]
    out = {}
    for i, kind in enumerate(t["layer_types"][: t["num_hidden_layers"]]):
        full = kind == "full_attention"
        hd, kv = layer_dims(t, i, full)
        p = f"{LAYER_PREFIX}{i}."
        out[p + "self_attn.q_proj"] = (heads * hd, hidden)
        out[p + "self_attn.k_proj"] = (kv * hd, hidden)
        if not (full and t.get("attention_k_eq_v")):
            out[p + "self_attn.v_proj"] = (kv * hd, hidden)
        out[p + "self_attn.o_proj"] = (hidden, heads * hd)
        out[p + "mlp.gate_proj"] = (inter, hidden)
        out[p + "mlp.up_proj"] = (inter, hidden)
        out[p + "mlp.down_proj"] = (hidden, inter)
    return out


def safetensors_header(path):
    """The JSON header of a safetensors file ({name: {dtype, shape, data_offsets}}), read without loading tensors."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def base_weights(base_dir):
    """{weight name: (shard file name, dtype, shape)} of a safetensors model directory (with or without an index)."""
    base = Path(base_dir)
    index = base / "model.safetensors.index.json"
    files = sorted(set(json.loads(index.read_text())["weight_map"].values())) if index.exists() else ["model.safetensors"]
    out = {}
    for name in files:
        for key, info in safetensors_header(base / name).items():
            if key != "__metadata__":
                out[key] = (name, info["dtype"], tuple(info["shape"]))
    return out


def adapter_config(rank, alpha):
    """PEFT's adapter_config.json for the converted adapter (LoraConfig's fields; the base model at its revision)."""
    return {
        "peft_type": "LORA", "task_type": "CAUSAL_LM", "base_model_name_or_path": BASE_REPO,
        "revision": BASE_REVISION, "inference_mode": True, "r": int(rank), "lora_alpha": float(alpha),
        "lora_dropout": 0.0, "bias": "none", "target_modules": TARGET_MODULES, "fan_in_fan_out": False,
        "use_rslora": False, "use_dora": False, "init_lora_weights": True, "modules_to_save": None,
        "layers_to_transform": None, "layers_pattern": None, "rank_pattern": {}, "alpha_pattern": {},
    }


def convert_arrays(arrays, meta, config, base_shapes=None):
    """(PEFT tensors {key: float32 array}, adapter_config, report) from adapter.npz's arrays and adapter.json. Every
    module the config expects must be in the adapter and nothing else; with base_shapes ({weight name: shape}) each
    module's (out, in) must be its base weight's shape."""
    rank, alpha, scale = meta["rank"], meta["alpha"], meta["scale"]
    if abs(alpha / rank - scale) > 1e-12:
        raise SystemExit(f"adapter.json: scale {scale} is not alpha / rank = {alpha} / {rank}")
    modules = mt_modules(arrays)
    expected = expected_shapes(config)
    got = {hf_module(m): m for m in modules}
    missing, extra = sorted(set(expected) - set(got)), sorted(set(got) - set(expected))
    if missing or extra:
        raise SystemExit(f"adapter modules against the base model's config: {len(missing)} missing (e.g. {missing[:3]}), "
                         f"{len(extra)} not in the model (e.g. {extra[:3]})")
    tensors, params, by_type = {}, 0, {}
    for name in sorted(got):
        lora_a, lora_b = peft_factors(*modules[got[name]], rank)
        shape = (lora_b.shape[0], lora_a.shape[1])
        if shape != expected[name]:
            raise SystemExit(f"{got[name]} -> {name}: update (out, in) {shape}, the config's {expected[name]}")
        if base_shapes is not None and tuple(base_shapes.get(name + ".weight", ())) != shape:
            raise SystemExit(f"{name}: update (out, in) {shape}, the base weight {base_shapes.get(name + '.weight')}")
        tensors[f"{PEFT_PREFIX}{name}.lora_A.weight"] = lora_a
        tensors[f"{PEFT_PREFIX}{name}.lora_B.weight"] = lora_b
        params += lora_a.size + lora_b.size
        kind = name.rsplit(".", 1)[-1]
        by_type[kind] = by_type.get(kind, 0) + 1
    t = text_config(config)
    full = [i for i, k in enumerate(t["layer_types"][: t["num_hidden_layers"]]) if k == "full_attention"]
    report = {"modules": len(got), "params": int(params), "by_type": dict(sorted(by_type.items())), "rank": rank,
              "alpha": alpha, "scaling": alpha / rank, "full_attention_layers": full,
              "full_without_v_proj": sum(1 for i in full if f"{LAYER_PREFIX}{i}.self_attn.v_proj" not in got)}
    return tensors, adapter_config(rank, alpha), report


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def cmd_convert(args):
    from safetensors.numpy import save_file

    meta = json.loads(Path(args.meta).read_text())
    config = json.loads(Path(args.config).read_text())
    with np.load(args.npz) as z:
        arrays = {k: z[k] for k in z.files}
    base_shapes = None
    if args.base_dir:
        base_shapes = {k: v[2] for k, v in base_weights(args.base_dir).items()}
    tensors, cfg, report = convert_arrays(arrays, meta, config, base_shapes)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out / "adapter_model.safetensors"), metadata={"format": "pt"})
    (out / "adapter_config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    report.update(npz_sha256=sha256(args.npz), safetensors_sha256=sha256(out / "adapter_model.safetensors"),
                  checked_against_base=bool(args.base_dir))
    (out / "convert.json").write_text(json.dumps(report, indent=1) + "\n")
    print("CONVERT " + json.dumps({k: v for k, v in report.items() if k != "full_attention_layers"}), flush=True)


# ------------------------------------------------------------------------------------------------------------ merge

def merge_weight(weight, lora_a, lora_b, scaling, torch):
    """(merged bf16 weight, sum of squares of the update, of the weight, number of changed elements): W + scaling * B A
    in float32, then the weight's dtype."""
    w32 = weight.to(torch.float32)
    delta = (torch.as_tensor(lora_b, dtype=torch.float32) @ torch.as_tensor(lora_a, dtype=torch.float32)) * scaling
    if delta.shape != w32.shape:
        raise SystemExit(f"update {tuple(delta.shape)} against weight {tuple(w32.shape)}")
    merged = (w32 + delta).to(weight.dtype)
    return merged, float((delta.double() ** 2).sum()), float((w32.double() ** 2).sum()), int((merged != weight).sum())


def cmd_merge(args):
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    base, adapter, out = Path(args.base_dir), Path(args.adapter_dir), Path(args.out)
    cfg = json.loads((adapter / "adapter_config.json").read_text())
    scaling = cfg["lora_alpha"] / cfg["r"]
    factors = {}
    with safe_open(adapter / "adapter_model.safetensors", framework="np") as f:
        for key in f.keys():
            m = re.match(r"^" + re.escape(PEFT_PREFIX) + r"(.+)\.lora_([AB])\.weight$", key)
            if not m:
                raise SystemExit(f"adapter key {key} is not {PEFT_PREFIX}<module>.lora_A/B.weight")
            factors.setdefault(m.group(1) + ".weight", {})[m.group(2)] = f.get_tensor(key)
    weights = base_weights(base)
    missing = sorted(k for k in factors if k not in weights)
    if missing or any(set(v) != {"A", "B"} for v in factors.values()):
        raise SystemExit(f"adapter modules without a base weight or without both factors: {missing[:3]}")
    if out.exists():
        raise SystemExit(f"{out} exists; merge into a new directory")
    out.mkdir(parents=True)
    shards = sorted({v[0] for v in weights.values()})
    delta_sq = weight_sq = 0.0
    changed = total = merged_n = 0
    for shard in shards:
        tensors = {}
        with safe_open(base / shard, framework="pt") as f:
            metadata = f.metadata()
            for key in f.keys():
                t = f.get_tensor(key)
                if key in factors:
                    if t.dtype != torch.bfloat16:
                        raise SystemExit(f"{key}: base weight {t.dtype}, expected bfloat16")
                    t, d2, w2, n = merge_weight(t, factors[key]["A"], factors[key]["B"], scaling, torch)
                    delta_sq, weight_sq, changed, total, merged_n = (delta_sq + d2, weight_sq + w2, changed + n,
                                                                     total + t.numel(), merged_n + 1)
                tensors[key] = t.contiguous()
        save_file(tensors, str(out / shard), metadata=metadata or {"format": "pt"})
        print(f"SHARD {json.dumps({'shard': shard, 'tensors': len(tensors)})}", flush=True)
        del tensors
    if merged_n != len(factors):
        raise SystemExit(f"merged {merged_n} weights, the adapter has {len(factors)}")
    for p in sorted(base.iterdir()):
        if p.is_file() and not COPY_SKIP.match(p.name):
            shutil.copy2(p, out / p.name)
    if (base / "model.safetensors.index.json").exists():
        shutil.copy2(base / "model.safetensors.index.json", out / "model.safetensors.index.json")
    stats = {"modules": merged_n, "scaling": scaling, "relative_update_norm": (delta_sq / weight_sq) ** 0.5,
             "elements_changed": changed / total, "shards": len(shards),
             "copied": sorted(p.name for p in out.iterdir() if not p.name.endswith(".safetensors"))}
    (out / "merge.json").write_text(json.dumps(stats, indent=1) + "\n")
    print("MERGE " + json.dumps(stats), flush=True)
    if args.maxtext_norm is not None:
        rel = stats["relative_update_norm"] / args.maxtext_norm - 1
        print(f"MERGE_VS_MAXTEXT {json.dumps({'maxtext_norm': args.maxtext_norm, 'relative_difference': rel})}")
        if abs(rel) > args.norm_tolerance:
            raise SystemExit(f"relative update norm {stats['relative_update_norm']} differs from MaxText's "
                             f"{args.maxtext_norm} by {rel:+.2%}, more than {args.norm_tolerance:.0%}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="G4-8-s140's MaxText LoRA as a PEFT adapter, and merged")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("convert")
    p.add_argument("--npz", required=True, help="sft_lora.py's adapter.npz")
    p.add_argument("--meta", required=True, help="its adapter.json (rank, alpha, scale)")
    p.add_argument("--config", required=True, help="the base model's config.json")
    p.add_argument("--base-dir", help="the base model's safetensors, to check every module's shape")
    p.add_argument("--out", required=True)
    p = sub.add_parser("merge")
    p.add_argument("--base-dir", required=True)
    p.add_argument("--adapter-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--maxtext-norm", type=float, help="relative_update_norm of translate.py's LORA line")
    p.add_argument("--norm-tolerance", type=float, default=0.01)
    args = parser.parse_args(argv)
    {"convert": cmd_convert, "merge": cmd_merge}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
