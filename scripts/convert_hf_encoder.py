#!/usr/bin/env python3
"""A Hugging Face RoBERTa or NeoBERT checkpoint (RobBERT-2023, RobBERT-2026) as params.pkl and config.json for
scripts/dumb/finetune.py, finetune_extra.py and eval_blimp_nl_hf.py (src/mmbjax/hf_encoders.py).

Usage:
    python scripts/convert_hf_encoder.py --model DTAI-KULeuven/robbert-2023-dutch-base \\
        --revision 1164b2df6d3ceb1735900c839da8de9bad1a99f9 --output /dev/shm/robbert_eval/ckpt-robbert-2023

Downloads config.json and model.safetensors of that revision only (no remote code is run), converts them to float32
in the trees of mmbjax.hf_encoders, runs the MLM head once on a sentence as a check that the weights are read (the
masked word must be among the top 10 of its predictions), and writes params.pkl ({"model": encoder, MLM head}) and
config.json (HFEncoderConfig, with the model and revision as those of the tokenizer).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse  # noqa: E402
import json  # noqa: E402
import pickle  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from mmbjax.hf_encoders import HFEncoderForMaskedLM, load_hf_checkpoint, load_tokenizer  # noqa: E402

# A sentence whose masked word an encoder of Dutch should rank high
CHECK = ("De hoofdstad van Nederland is", "Amsterdam", ".")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help="Hugging Face repository")
    parser.add_argument("--revision", required=True, help="Commit of that repository (a 40-character sha)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache-dir", default=None, help="Hugging Face cache for the download (default: its own)")
    parser.add_argument("--keep-tokenizer-offsets", action="store_true",
                        help="Leave the tokenizer's offsets as they are, also when they are shifted (hf_encoders.load_tokenizer)")
    args = parser.parse_args(argv)
    if len(args.revision) != 40:
        raise SystemExit(f"--revision {args.revision}: give the full commit sha, so the weights are fixed")
    from huggingface_hub import snapshot_download

    directory = snapshot_download(args.model, revision=args.revision, allow_patterns=["config.json", "model.safetensors"],
                                  cache_dir=args.cache_dir)
    params, config = load_hf_checkpoint(directory)
    config.model_name = config.tokenizer_name = args.model
    config.model_revision = config.tokenizer_revision = args.revision
    config.tokenizer_offsets = "tokenizer" if args.keep_tokenizer_offsets else "fixed"
    count = sum(int(np.prod(x.shape)) for x in jax.tree_util.tree_leaves(params))
    print(f"{args.model}@{args.revision}: {config.model_type}, {config.num_hidden_layers} layers, {count:,} parameters")

    tokenizer = load_tokenizer(config)
    before = tokenizer(CHECK[0])["input_ids"][:-1]  # without </s>
    after = tokenizer(CHECK[2], add_special_tokens=False)["input_ids"]
    ids = before + [tokenizer.mask_token_id] + after + [tokenizer.sep_token_id]
    target = tokenizer(" " + CHECK[1], add_special_tokens=False)["input_ids"][0]
    logits = HFEncoderForMaskedLM(config).apply({"params": params}, jnp.asarray([ids]), jnp.ones((1, len(ids))),
                                                positions=jnp.asarray([len(before)]))
    top = np.argsort(-np.asarray(logits[0]))[:10]
    print(f"check: '{CHECK[0]} <mask>{CHECK[2]}' -> {tokenizer.convert_ids_to_tokens(top.tolist())}")
    if target not in top:
        raise SystemExit(f"check failed: {tokenizer.convert_ids_to_tokens([target])} is not among the top 10")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with open(output / "params.pkl", "wb") as f:
        pickle.dump(params, f)
    (output / "config.json").write_text(json.dumps(config.to_dict(), indent=2) + "\n")
    print(f"wrote {output}/params.pkl and config.json")


if __name__ == "__main__":
    main()
