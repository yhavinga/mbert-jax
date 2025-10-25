#!/usr/bin/env python
"""Run ModernBERT masked-LM inference using converted PyTorch weights."""

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
from transformers import AutoConfig, AutoTokenizer

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


def _run_inference(
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

    pt_params = _load_state_dict(checkpoint_path)
    flax_params = convert_pytorch_state_dict_to_flax(pt_params, config)

    encodings = tokenizer(
        args.texts,
        return_tensors="np",
        padding=True,
        truncation=True,
        max_length=args.max_length,
    )
    input_ids = jnp.array(encodings["input_ids"], dtype=jnp.int32)
    attention_mask = jnp.array(encodings["attention_mask"], dtype=jnp.int32)

    model = ModernBertForMaskedLM(config)
    logits = _run_inference(model, flax_params, input_ids, attention_mask)

    input_ids_np = np.asarray(encodings["input_ids"])
    mask_token_id = tokenizer.mask_token_id

    print("\n=== ModernBERT Masked-LM Inference ===")
    for idx, text in enumerate(args.texts):
        mask_positions = np.where(input_ids_np[idx] == mask_token_id)[0]
        print(f"\nPrompt {idx}: {text}")
        if mask_positions.size == 0:
            print("  (no mask token found in this prompt)")
            continue

        for mask_pos in mask_positions:
            token_logits = logits[idx, mask_pos]
            token_probs = np.asarray(jax.nn.softmax(jnp.array(token_logits)))
            top_ids, top_scores = _topk(token_logits, args.top_k)
            print(f"  Mask position {mask_pos} (token idx {mask_pos}):")
            for rank, (token_id, logit) in enumerate(zip(top_ids, top_scores), start=1):
                prob = float(token_probs[token_id])
                token_str = tokenizer.convert_ids_to_tokens(int(token_id))
                suggestion = _format_prediction(tokenizer, input_ids_np[idx], mask_pos, int(token_id))
                print(
                    f"    {rank:>2}. {token_str:<12} logit={logit: .4f} prob={prob: .4f} -> {suggestion}"
                )


if __name__ == "__main__":
    main()
