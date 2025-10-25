#!/usr/bin/env python
"""Dump ModernBERT PyTorch reference outputs for JAX/Flax parity tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForMaskedLM


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-name",
        default="jhu-clsp/mmBERT-small",
        help="HF model identifier to dump.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/reference-configs"),
        help="Directory to store config + tensors.",
    )
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size for the dummy batch.")
    parser.add_argument("--seq-len", type=int, default=16, help="Sequence length for the dummy batch.")
    parser.add_argument("--seed", type=int, default=0, help="Torch RNG seed for reproducibility.")
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device to run the PyTorch forward pass on (e.g., cpu, cuda).",
    )
    return parser.parse_args()


def _format_model_name(model_name: str) -> str:
    return model_name.replace("/", "_")


def _prepare_inputs(
    config: Any,
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    input_ids = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(batch_size, seq_len),
        device=device,
    )
    attention_mask = torch.ones_like(input_ids)

    for row in range(batch_size):
        pad_tokens = row + 1
        if pad_tokens >= seq_len:
            continue
        attention_mask[row, -pad_tokens:] = 0
        input_ids[row, -pad_tokens:] = config.pad_token_id

    return {"input_ids": input_ids, "attention_mask": attention_mask}


def _save_config(config: AutoConfig, path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(config.to_dict(), handle, indent=2, sort_keys=True)
        handle.write("\n")


def _save_batch(npz_path: Path, payload: Dict[str, np.ndarray]) -> None:
    np.savez(npz_path, **payload)


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = _format_model_name(args.model_name)

    device = torch.device(args.device)
    config = AutoConfig.from_pretrained(args.model_name)
    model = AutoModelForMaskedLM.from_pretrained(args.model_name, config=config).to(device)
    model.eval()

    torch.manual_seed(args.seed)
    inputs = _prepare_inputs(config, batch_size=args.batch_size, seq_len=args.seq_len, device=device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    hidden = torch.stack(outputs.hidden_states, dim=0).cpu().numpy()
    logits = outputs.logits.cpu().numpy()
    payload = {
        "input_ids": inputs["input_ids"].cpu().numpy(),
        "attention_mask": inputs["attention_mask"].cpu().numpy(),
        "last_hidden_state": outputs.hidden_states[-1].cpu().numpy(),
        "hidden_states": hidden,
        "logits": logits,
    }

    config_path = args.output_dir / f"{safe_name}_config.json"
    npz_path = args.output_dir / f"{safe_name}_batch.npz"
    _save_config(config, config_path)
    _save_batch(npz_path, payload)
    print(f"Wrote config to {config_path}")
    print(f"Wrote tensors to {npz_path}")


if __name__ == "__main__":
    main()
