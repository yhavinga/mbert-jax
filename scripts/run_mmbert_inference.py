#!/usr/bin/env python
"""Run ModernBERT masked-LM inference comparing JAX and PyTorch implementations."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForMaskedLM

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.mmbjax.heads.masked_lm import ModernBertForMaskedLM
from src.mmbjax.utils import convert_pytorch_state_dict_to_flax, load_hf_config_as_mmbert

HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-name",
        default="jhu-clsp/mmBERT-small",
        help="Hugging Face model identifier (must exist in local HF cache unless --remote-files-ok).",
    )
    parser.add_argument(
        "--tokenizer-name",
        help="Tokenizer to use (defaults to --model-name).",
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        help="Path to a dumped HF config JSON. If omitted we load it from the HF cache.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        help="Optional explicit path to pytorch_model.bin. Falls back to the HF cache snapshot.",
    )
    parser.add_argument(
        "--texts",
        nargs="+",
        required=True,
        help="Input sentences. Include the tokenizer mask token (e.g., '[MASK]') to inspect predictions.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="How many predictions to report per mask position.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
        help="Maximum sequence length for tokenization.",
    )
    parser.add_argument(
        "--remote-files-ok",
        action="store_true",
        help="Allow transformers to hit the network if files are missing locally.",
    )
    parser.add_argument(
        "--jax-only",
        action="store_true",
        help="Only run JAX inference (skip PyTorch comparison).",
    )
    return parser.parse_args()


def _resolve_snapshot_dir(model_name: str) -> Path:
    safe_name = model_name.replace("/", "--")
    model_dir = HF_CACHE / f"models--{safe_name}"
    snapshots_dir = model_dir / "snapshots"
    refs_dir = model_dir / "refs" / "main"

    if refs_dir.exists():
        revision = refs_dir.read_text().strip()
        candidate = snapshots_dir / revision
        if candidate.exists():
            return candidate

    if snapshots_dir.exists():
        for candidate in sorted(snapshots_dir.iterdir()):
            if candidate.is_dir():
                return candidate

    raise FileNotFoundError(
        f"Could not find cached snapshot for {model_name}. "
        "Download it first with `transformers` (e.g., via py_reference_dump.py)."
    )


def _load_state_dict(checkpoint_path: Path) -> dict[str, np.ndarray]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    return {k: v.numpy() for k, v in state_dict.items()}


def _topk(values: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    k = min(k, values.shape[-1])
    indices = np.argpartition(-values, k - 1)[:k]
    scores = values[indices]
    order = np.argsort(-scores)
    return indices[order], scores[order]


def _format_prediction(tokenizer, input_ids: np.ndarray, mask_idx: int, token_id: int) -> str:
    filled = input_ids.copy()
    filled[mask_idx] = token_id
    return tokenizer.decode(filled, skip_special_tokens=True)


def _run_jax_inference(
    model: ModernBertForMaskedLM,
    params: dict,
    input_ids: jnp.ndarray,
    attention_mask: jnp.ndarray,
) -> np.ndarray:
    outputs = model.apply(
        params,
        input_ids=input_ids,
        attention_mask=attention_mask,
        deterministic=True,
    )
    return np.asarray(outputs.logits)


def _run_pytorch_inference(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> np.ndarray:
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    return outputs.logits.cpu().numpy()


def main() -> None:
    args = _parse_args()
    tokenizer_name = args.tokenizer_name or args.model_name
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            local_files_only=not args.remote_files_ok,
        )
    except ValueError as exc:
        if "tiktoken" in str(exc).lower():
            raise SystemExit(
                "Tokenizer conversion requires the `tiktoken` package. "
                "Install it via `pip install tiktoken` or provide a compatible tokenizer via --tokenizer-name."
            ) from exc
        raise
    if tokenizer.mask_token_id is None:
        raise ValueError(f"Tokenizer {tokenizer_name} does not define a mask token.")

    if args.config_path:
        config_source = args.config_path
    else:
        hf_cfg = AutoConfig.from_pretrained(
            args.model_name,
            local_files_only=not args.remote_files_ok,
        )
        config_source = hf_cfg.to_dict()
    config = load_hf_config_as_mmbert(config_source)

    checkpoint_path = args.checkpoint_path
    if checkpoint_path is None:
        snapshot_dir = _resolve_snapshot_dir(args.model_name)
        checkpoint_path = snapshot_dir / "pytorch_model.bin"

    encodings = tokenizer(
        args.texts,
        return_tensors="np",
        padding=True,
        truncation=True,
        max_length=args.max_length,
    )
    input_ids_np = np.asarray(encodings["input_ids"])
    attention_mask_np = np.asarray(encodings["attention_mask"])
    mask_token_id = tokenizer.mask_token_id

    # Run JAX inference
    pt_params = _load_state_dict(checkpoint_path)
    flax_params = convert_pytorch_state_dict_to_flax(pt_params, config)
    jax_model = ModernBertForMaskedLM(config)
    input_ids_jax = jnp.array(input_ids_np, dtype=jnp.int32)
    attention_mask_jax = jnp.array(attention_mask_np, dtype=jnp.int32)
    jax_logits = _run_jax_inference(jax_model, flax_params, input_ids_jax, attention_mask_jax)

    # Run PyTorch inference (unless --jax-only)
    pt_logits = None
    if not args.jax_only:
        pt_model = AutoModelForMaskedLM.from_pretrained(
            args.model_name,
            local_files_only=not args.remote_files_ok,
        )
        pt_model.eval()
        input_ids_pt = torch.from_numpy(input_ids_np)
        attention_mask_pt = torch.from_numpy(attention_mask_np)
        pt_logits = _run_pytorch_inference(pt_model, input_ids_pt, attention_mask_pt)

    print("\n=== ModernBERT Masked-LM Inference Comparison ===")
    for idx, text in enumerate(args.texts):
        mask_positions = np.where(input_ids_np[idx] == mask_token_id)[0]
        print(f"\nPrompt {idx}: {text}")
        if mask_positions.size == 0:
            print("  (no mask token found in this prompt)")
            continue

        for mask_pos in mask_positions:
            jax_token_logits = jax_logits[idx, mask_pos]
            jax_token_probs = np.asarray(jax.nn.softmax(jnp.array(jax_token_logits)))
            jax_top_ids, jax_top_scores = _topk(jax_token_logits, args.top_k)

            print(f"  Mask position {mask_pos}:")
            print("    JAX predictions:")
            for rank, (token_id, logit) in enumerate(zip(jax_top_ids, jax_top_scores), start=1):
                prob = float(jax_token_probs[token_id])
                token_str = tokenizer.convert_ids_to_tokens(int(token_id))
                suggestion = _format_prediction(tokenizer, input_ids_np[idx], mask_pos, int(token_id))
                print(
                    f"      {rank:>2}. {token_str:<12} logit={logit: .4f} prob={prob: .4f} -> {suggestion}"
                )

            if pt_logits is not None:
                pt_token_logits = pt_logits[idx, mask_pos]
                pt_token_probs = torch.softmax(torch.from_numpy(pt_token_logits), dim=-1).numpy()
                pt_top_ids, pt_top_scores = _topk(pt_token_logits, args.top_k)

                print("    PyTorch predictions:")
                for rank, (token_id, logit) in enumerate(zip(pt_top_ids, pt_top_scores), start=1):
                    prob = float(pt_token_probs[token_id])
                    token_str = tokenizer.convert_ids_to_tokens(int(token_id))
                    suggestion = _format_prediction(tokenizer, input_ids_np[idx], mask_pos, int(token_id))
                    print(
                        f"      {rank:>2}. {token_str:<12} logit={logit: .4f} prob={prob: .4f} -> {suggestion}"
                    )

                # Show max absolute difference
                max_diff = np.abs(jax_logits[idx, mask_pos] - pt_logits[idx, mask_pos]).max()
                mean_diff = np.abs(jax_logits[idx, mask_pos] - pt_logits[idx, mask_pos]).mean()
                print(f"    Logit differences: max={max_diff:.6f}, mean={mean_diff:.6f}")


if __name__ == "__main__":
    main()
