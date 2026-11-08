#!/usr/bin/env python3
"""BLiMP-NL zero-shot with the MLM head: which sentence of each minimal pair gets the higher pseudo-log-likelihood.

Usage (one host as a TPU of its own 4 chips, see experiments/dumb-extra/blimp.sh):
    python scripts/eval_blimp_nl.py --checkpoint gs://bucket/run/ckpt_keep/653000 --data ~/dumbx/data/blimp-nl/large.parquet \\
        --tokenizer yhavinga/dutch-llama-tokenizer --hidden-size 1024 --num-layers 22 --num-heads 16 \\
        --intermediate-size 1536 --output gs://t5-dutch-english-usc2/mbert/blimp-nl/S.json

The score of a sentence is PLL-word-l2r (Kauf and Ivanova 2023, "A Better Way to Do Masked Language Model Scoring",
ACL 2023), the metric Suijkerbuijk et al. (2025) use for BLiMP-NL: per token, the log-probability of that token with
the token itself and the tokens after it in the same word masked, summed over the sentence. A pair counts as right when
the grammatical sentence scores strictly higher (a tie counts as wrong, and is counted). No fine-tuning, no sampling:
one number per model.

A sentence is a document as the trainer makes one, <s> tokens </s>; <s> and </s> are never scored. Words: this
tokenizer has no pre-tokenizer, so its word_ids() put a whole sentence in one word (the route of minicons, Kauf and
Ivanova's code). A word here is what BERT's basic tokenizer makes one, as for the masked models of the BLiMP-NL paper:
a new word starts at a token with ▁ (whitespace before it), at a token that starts with punctuation, and after a token
that ends with punctuation. So "kamer." is two words and "'s-Hertogenbosch" four.

Every masked copy of every sentence is a piece of its own, packed into rows with segment ids and position ids as in
eval_windows.py, so no copy sees another. float32 by default: the pair compares two sums of log-probabilities.
A --mix-heads checkpoint is read with all its heads and scored with one (--export-head, default 0: the head of the
15% single-token denoisers, which predicts the held-out masks).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402
import unicodedata  # noqa: E402
from collections import defaultdict  # noqa: E402

import jax  # noqa: E402

# One host of a slice as a TPU of its own (blimp.sh sets TPU_PROCESS_BOUNDS=1,1,1): jax.distributed.initialize() in
# train_minimal's import takes the slice from the TPU metadata, not from these variables, and aborts after waiting for
# the other hosts. A runtime of one process set up first makes that import skip its own.
if os.environ.get("TPU_PROCESS_BOUNDS") == "1,1,1" and not jax.distributed.is_initialized():
    jax.distributed.initialize(coordinator_address="localhost:8482", num_processes=1, process_id=0)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import train_minimal as tm  # noqa: E402  # Its import initializes the runtime, as the trainer's does
from etils import epath  # noqa: E402
from eval_windows import build_model, pack_rows, pad_to_rows  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

# Head k > 0 of a --mix-heads checkpoint, next to head 0 under the names of the single head
EXTRA_HEAD = re.compile(r"(prediction_head|decoder)_(\d+)")
BYTE_TOKEN = re.compile(r"<0x[0-9A-F]{2}>")


def is_punctuation(char):
    """BERT's _is_punctuation: the ASCII non-alphanumerics and every Unicode P* category."""
    code = ord(char)
    if 33 <= code <= 47 or 58 <= code <= 64 or 91 <= code <= 96 or 123 <= code <= 126:
        return True
    return unicodedata.category(char).startswith("P")


def word_starts(tokens):
    """Per token of a sentence (no <s>, </s>): whether it starts a word. Byte-fallback tokens continue the word."""
    starts = []
    previous = ""
    for i, token in enumerate(tokens):
        if BYTE_TOKEN.fullmatch(token):
            starts.append(i == 0)
            previous = ""
            continue
        text = token[1:] if token.startswith("▁") else token
        starts.append(i == 0 or token.startswith("▁") or (text != "" and is_punctuation(text[0]))
                      or (previous != "" and is_punctuation(previous[-1])))
        previous = text
    return starts


def l2r_masks(starts):
    """Per token: the positions masked when it is scored, itself and the later tokens of its word (PLL-word-l2r)."""
    ends = [0] * len(starts)
    end = len(starts)
    for i in range(len(starts) - 1, -1, -1):
        ends[i] = end
        if starts[i]:
            end = i
    return [list(range(i, ends[i])) for i in range(len(starts))]


def masked_copies(token_ids, tokens, bos_token_id, eos_token_id, mask_token_id):
    """The copies that score one sentence: per token, (<s> ids </s> with its l2r mask applied, position of the token)."""
    sentence = np.asarray([bos_token_id, *token_ids, eos_token_id], dtype=np.int32)
    copies = []
    for target, masked in enumerate(l2r_masks(word_starts(tokens))):
        inputs = sentence.copy()
        inputs[[1 + m for m in masked]] = mask_token_id
        copies.append((inputs, 1 + target))
    return copies


def score_pairs(pairs, pll_grammatical, pll_ungrammatical):
    """Accuracy overall, per phenomenon, per paradigm and for the hand-crafted (small-version) and generated pairs."""
    right = np.asarray(pll_grammatical) > np.asarray(pll_ungrammatical)
    ties = np.asarray(pll_grammatical) == np.asarray(pll_ungrammatical)

    def summary(indices):
        return {"accuracy": float(right[indices].mean()), "pairs": len(indices), "ties": int(ties[indices].sum())}

    groups = {"phenomena": defaultdict(list), "paradigms": defaultdict(list)}
    for i, pair in enumerate(pairs):
        groups["phenomena"][pair["phenomenon_nr"]].append(i)
        groups["paradigms"][pair["paradigm_nr"]].append(i)
    phenomena = {str(k): {"name": pairs[v[0]]["phenomenon"], **summary(v)} for k, v in sorted(groups["phenomena"].items())}
    paradigms = {str(k): {"name": pairs[v[0]]["paradigm"], "phenomenon_nr": pairs[v[0]]["phenomenon_nr"],
                          "phenomenon": pairs[v[0]]["phenomenon"], **summary(v)} for k, v in sorted(groups["paradigms"].items())}
    handcrafted = [i for i, pair in enumerate(pairs) if pair["handcrafted"]]
    generated = [i for i, pair in enumerate(pairs) if not pair["handcrafted"]]
    return {
        **summary(list(range(len(pairs)))),
        # The mean over paradigms, the "Model Average" of the paper's Table 3; equal to the accuracy when every
        # paradigm has the same number of pairs, as in the large version
        "paradigm_mean_accuracy": float(np.mean([p["accuracy"] for p in paradigms.values()])),
        "subsets": {"handcrafted": summary(handcrafted), "generated": summary(generated)},
        "phenomena": phenomena,
        "paradigms": paradigms,
    }


def read_pairs(path, max_pairs=None):
    import pyarrow.parquet as pq

    pairs = pq.read_table(path).to_pylist()
    return pairs if max_pairs is None else pairs[:max_pairs]


def mlm_params(checkpoint, model_params, export_head):
    """The parameters of a checkpoint in the tree of a one-head model, with head export_head as its prediction head.

    load_params_of_run wants the saved tree exactly, so a --mix-heads checkpoint is read into this tree with its extra
    heads added (each the shape of head 0), and then all heads but the chosen one are left out.
    """
    item_dir = epath.Path(checkpoint) / "default"
    if not item_dir.exists():
        raise FileNotFoundError(f"{checkpoint}: no {item_dir}, so this is no step directory of an Orbax checkpoint")
    saved = tm.ocp.PyTreeCheckpointer().metadata(item_dir).item_metadata["params"]
    extra = sorted(key for key in saved if EXTRA_HEAD.fullmatch(key))
    heads = 1 + len({EXTRA_HEAD.fullmatch(key).group(2) for key in extra})
    if not 0 <= export_head < heads:
        raise ValueError(f"--export-head {export_head}: {checkpoint} has heads 0 to {heads - 1}")
    target = dict(model_params)
    for key in extra:
        target[key] = model_params[EXTRA_HEAD.fullmatch(key).group(1)]
    params = dict(tm.load_params_of_run(checkpoint, target))
    if export_head > 0:
        params["prediction_head"] = params[f"prediction_head_{export_head}"]
        # The decoder has parameters (its bias) only with config.decoder_bias
        if "decoder" in params:
            params["decoder"] = params[f"decoder_{export_head}"]
    for key in extra:
        del params[key]
    return params, heads


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="BLiMP-NL accuracy of a checkpoint's MLM head, PLL-word-l2r")
    parser.add_argument("--checkpoint", required=True, help="Step directory of a train_minimal.py checkpoint")
    parser.add_argument("--data", required=True, help="blimp-nl/large.parquet of experiments/dumb-extra/stage_data.py")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--export-head", type=int, default=0, help="The prediction head of a --mix-heads checkpoint (default: 0)")
    parser.add_argument("--max-pairs", type=int, default=None, help="The first this many pairs (smoke tests)")
    parser.add_argument("--row-length", type=int, default=512, help="Rows of the packed batches")
    parser.add_argument("--rows-per-device", type=int, default=16)
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--num-layers", type=int, default=22)
    parser.add_argument("--num-heads", type=int, default=12)
    parser.add_argument("--intermediate-size", type=int, default=3072)
    parser.add_argument("--global-attn-every-n-layers", type=int, default=3)
    parser.add_argument("--compute-dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--output", default=None, help="JSON file, local or gs:// (default: print only)")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if jax.process_count() != 1 or jax.device_count() != jax.local_device_count():
        raise SystemExit(f"{jax.process_count()} processes, {jax.device_count()} devices: run this on one host as a TPU of its own "
                         "(experiments/dumb-extra/blimp.sh)")
    started = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, split_special_tokens=True)
    if tokenizer.mask_token is None:
        # As the trainer does, so the vocabulary and the mask id are the trainer's
        tokenizer.add_special_tokens({"mask_token": "<mask>"})
    config, model = build_model(args, tokenizer)

    replicated, batch_sharding = tm.create_shardings()
    ones = jnp.ones((1, 8), dtype=jnp.int32)
    # Eagerly, as the trainer does: the first call fills the module-level RoPE cache (see eval_windows.py)
    init_params = tm.to_global(model.init(jax.random.PRNGKey(0), input_ids=ones, attention_mask=ones, deterministic=True)["params"], replicated)
    params, heads = mlm_params(args.checkpoint, init_params, args.export_head)
    del init_params

    pairs = read_pairs(args.data, args.max_pairs)
    sentences = [pair[side] for pair in pairs for side in ("grammatical", "ungrammatical")]
    token_lists = tokenizer(sentences, add_special_tokens=False, split_special_tokens=True)["input_ids"]
    pieces, targets, owner = [], [], []
    for s, token_ids in enumerate(token_lists):
        if not token_ids:
            raise ValueError(f"sentence {s} has no tokens: {sentences[s]!r}")
        if len(token_ids) + 2 > args.row_length:
            raise ValueError(f"sentence {s} has {len(token_ids)} tokens, more than --row-length {args.row_length} holds")
        for inputs, position in masked_copies(token_ids, tokenizer.convert_ids_to_tokens(token_ids), tokenizer.bos_token_id,
                                              tokenizer.eos_token_id, tokenizer.mask_token_id):
            # labels carry the index of the copy at its scored position, so each log-probability finds its copy after packing
            labels = np.full(inputs.shape, -100, dtype=np.int32)
            labels[position] = len(targets)
            pieces.append((inputs, labels, False))
            targets.append(token_ids[position - 1])
            owner.append(s)
    rows = pack_rows(pieces, args.row_length, tokenizer.pad_token_id)
    rows_per_batch = args.rows_per_device * jax.device_count()
    n_batches = -(-len(rows["input_ids"]) // rows_per_batch)
    padded = pad_to_rows(rows, n_batches * rows_per_batch, tokenizer.pad_token_id)
    target_ids = np.asarray(targets, dtype=np.int32)
    padded["target_ids"] = np.where(padded["labels"] >= 0, target_ids[np.maximum(padded["labels"], 0)], 0).astype(np.int32)

    def forward(params, batch):
        logits = model.apply({"params": params}, input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                             position_ids=batch["position_ids"], segment_ids=batch["segment_ids"], deterministic=True).logits
        log_probs = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
        return jnp.take_along_axis(log_probs, batch["target_ids"][..., None], axis=-1)[..., 0]

    forward_jit = jax.jit(forward, out_shardings=replicated)
    copy_log_probs = np.full(len(targets), np.nan)
    keys = ("input_ids", "attention_mask", "position_ids", "segment_ids", "target_ids")
    for b in range(n_batches):
        window = slice(b * rows_per_batch, (b + 1) * rows_per_batch)
        local = {key: padded[key][window] for key in keys}
        log_probs = np.asarray(forward_jit(params, tm.to_global(local, batch_sharding)), dtype=np.float64)
        labels = padded["labels"][window]
        scored = labels >= 0
        if not (padded["input_ids"][window][scored] == tokenizer.mask_token_id).all():
            raise RuntimeError(f"batch {b}: a scored position without <mask>")
        copy_log_probs[labels[scored]] = log_probs[scored]
    if np.isnan(copy_log_probs).any():
        raise RuntimeError(f"{int(np.isnan(copy_log_probs).sum())} masked copies have no log-probability")
    pll = np.zeros(len(sentences))
    np.add.at(pll, np.asarray(owner), copy_log_probs)
    pll_grammatical, pll_ungrammatical = pll[0::2], pll[1::2]
    lengths = [len(t) for t in token_lists]

    result = {
        "checkpoint": args.checkpoint, "data": args.data, "tokenizer": args.tokenizer, "metric": "PLL-word-l2r",
        "export_head": args.export_head, "prediction_heads": heads, "compute_dtype": args.compute_dtype,
        "vocab_size": config.vocab_size, "mask_token_id": tokenizer.mask_token_id, "max_pairs": args.max_pairs,
        "sentences": len(sentences), "tokens": int(sum(lengths)), "masked_copies": len(targets), "rows": len(rows["input_ids"]),
        "batches": n_batches,
        **score_pairs(pairs, pll_grammatical, pll_ungrammatical),
        # Per pair, for paired comparisons of models: item, PLL of the grammatical and the ungrammatical sentence, tokens
        "pair_scores": [[pair["item_nr"], float(g), float(u), lengths[2 * i], lengths[2 * i + 1]]
                        for i, (pair, g, u) in enumerate(zip(pairs, pll_grammatical, pll_ungrammatical))],
    }
    result["seconds"] = time.time() - started
    print(f"BLIMP accuracy={result['accuracy']:.4f} handcrafted={result['subsets']['handcrafted']['accuracy']:.4f} "
          f"generated={result['subsets']['generated']['accuracy']:.4f} pairs={result['pairs']} ties={result['ties']} "
          f"copies={len(targets)} batches={n_batches} seconds={result['seconds']:.0f}", flush=True)
    for number, phenomenon in result["phenomena"].items():
        print(f"BLIMP phenomenon {number} {phenomenon['name']}: {phenomenon['accuracy']:.4f}", flush=True)
    if args.output:
        tm.write_file(epath.Path(args.output), (json.dumps(result, indent=2) + "\n").encode())
    return result


if __name__ == "__main__":
    import traceback

    try:
        main()
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        # As in train_minimal.py: a normal exit would wait in the shutdown barrier for peers stuck in a collective
        os._exit(1)
