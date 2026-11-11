#!/usr/bin/env python3
"""A LoRA adapter of sft_lora.py with its contribution scaled by a factor s (protocol.md §9, amendment of 11 Nov:
arms N50 and N25, N's adapter at 50% and 25%): every kernel_lora_b times s; kernel_lora_a and adapter.json's rank,
alpha and scale stay as they are, so translate.py --lora adds s times the source adapter's update (A.B.scale) to each
kernel. adapter.json gains scaled_from and lora_b_factor, so the adapter says what it is.

    python scale_adapter.py --adapter DIR --factor 0.5 --source gs://.../arms/N --out DIR

DIR holds adapter.npz and adapter.json; --out must not hold either yet.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def scale_adapter(adapter, factor):
    """{key: array} with every kernel_lora_b times factor (in its own dtype) and every kernel_lora_a unchanged."""
    if not 0 < factor <= 1:
        raise ValueError(f"factor {factor} is not in (0, 1]")
    out = {}
    for key, value in adapter.items():
        name = key.rpartition("/")[2]
        if name == "kernel_lora_b":
            out[key] = (value * np.asarray(factor, dtype=value.dtype)).astype(value.dtype)
        elif name == "kernel_lora_a":
            out[key] = value
        else:
            raise ValueError(f"unexpected key in the adapter: {key}")
    if not any(k.endswith("/kernel_lora_b") for k in out):
        raise ValueError("the adapter has no kernel_lora_b")
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--adapter", required=True, help="directory with adapter.npz and adapter.json")
    parser.add_argument("--factor", type=float, required=True)
    parser.add_argument("--source", required=True, help="where the adapter comes from (written to adapter.json)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    src, out = Path(args.adapter), Path(args.out)
    meta = json.loads((src / "adapter.json").read_text())
    if "lora_b_factor" in meta:
        raise SystemExit(f"{src}: already scaled (lora_b_factor {meta['lora_b_factor']}); scale the source adapter")
    for name in ("adapter.npz", "adapter.json"):
        if (out / name).exists():
            raise SystemExit(f"{out / name} exists")
    with np.load(src / "adapter.npz") as npz:
        adapter = {k: npz[k] for k in npz.files}
    scaled = scale_adapter(adapter, args.factor)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "adapter.npz", **scaled)
    meta.update(scaled_from=args.source, lora_b_factor=args.factor)
    (out / "adapter.json").write_text(json.dumps(meta, indent=1) + "\n")
    b_keys = [k for k in adapter if k.endswith("/kernel_lora_b")]
    norm = lambda d: float(np.sqrt(sum(float(np.sum(np.square(d[k].astype(np.float64)))) for k in b_keys)))
    print("SCALED " + json.dumps({"out": str(out), "factor": args.factor, "modules": len(b_keys),
                                  "lora_b_norm": norm(adapter), "lora_b_norm_scaled": norm(scaled),
                                  "dtypes": sorted({str(v.dtype) for v in scaled.values()})}), flush=True)


if __name__ == "__main__":
    main()
