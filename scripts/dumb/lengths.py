#!/usr/bin/env python3
"""Share of a DUMB task's examples longer than given lengths, in tokens as finetune.py makes them.

Usage:
    python scripts/dumb/lengths.py --task dbrd --lengths 512 1024 2048 --output lengths.json

The loader of data.py with a limit no example reaches, so these are the lengths before
finetune.py's --max-length cuts them: <s> included, as finetune.py's tokenizer adds it.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from data import get_task  # noqa: E402

# More tokens than any example of these tasks has; truncation=True then cuts nothing
NO_LIMIT = 10**9


def length_shares(lengths, limits):
    """Per limit: how many examples are longer, their share, and the share of tokens beyond it."""
    lengths = np.asarray(lengths)
    total_tokens = int(lengths.sum())
    return {
        str(limit): {
            "longer": int((lengths > limit).sum()),
            "share_longer": float((lengths > limit).mean()),
            "share_of_tokens_beyond": float(np.maximum(lengths - limit, 0).sum() / total_tokens),
        }
        for limit in limits
    }


def summarize(lengths, limits):
    lengths = np.asarray(lengths)
    return {
        "examples": int(lengths.size),
        "tokens": int(lengths.sum()),
        "mean": float(lengths.mean()),
        "median": float(np.median(lengths)),
        "p90": float(np.percentile(lengths, 90)),
        "p99": float(np.percentile(lengths, 99)),
        "max": int(lengths.max()),
        "limits": length_shares(lengths, limits),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", required=True, choices=["dbrd", "sicknl"])
    parser.add_argument("--tokenizer", default="yhavinga/dutch-llama-tokenizer")
    parser.add_argument("--lengths", type=int, nargs="+", default=[512, 1024, 2048])
    parser.add_argument("--output", default=None, help="JSON file for the result (default: print only)")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    dataset = get_task(args.task)["loader"](tokenizer, NO_LIMIT)
    result = {
        "task": args.task,
        "tokenizer": args.tokenizer,
        "splits": {split: summarize([len(ids) for ids in dataset[split]["input_ids"]], args.lengths) for split in dataset},
    }
    text = json.dumps(result, indent=2)
    print(text)
    if args.output:
        Path(args.output).write_text(text + "\n")


if __name__ == "__main__":
    main()
