#!/usr/bin/env python3
"""BLiMP-NL zero-shot for RoBERTa and NeoBERT (RobBERT-2023, RobBERT-2026; mmbjax.hf_encoders), with the scoring of
scripts/eval_blimp_nl.py: PLL-word-l2r (Kauf and Ivanova 2023), per token the log-probability of that token with the
token itself and the later tokens of its word masked, summed over the sentence; a pair is right when the grammatical
sentence scores strictly higher. The masks (l2r_masks), the pair scores (score_pairs) and the words (BERT's basic
tokenizer: a word starts after whitespace, at a token that starts with punctuation and after a token that ends with
one) are those of eval_blimp_nl.py; only reading the words off the tokens differs, as these tokenizers are byte-level
BPE: a token's bytes (GPT-2's byte-to-unicode map undone) start with a space where a word starts after whitespace, a
token whose bytes are no whole UTF-8 text continues its word, as a byte-fallback token does there, and a bare space
token (RobBERT-2023's tokenizer puts one before every punctuation mark: "Ġ" ",") starts the word of the token after it.

A sentence is <s> tokens </s> as the model's own tokenizer makes it; <s> and </s> are never scored. Every masked copy
is a row of its own (no packing: these models have no segment ids), padded to the longest; float32.

Usage (one host as a TPU of its own 4 chips):
    python scripts/eval_blimp_nl_hf.py --checkpoint /dev/shm/robbert_eval/ckpt-robbert-2023 \\
        --data /dev/shm/robbert_eval/data/blimp-nl/large.parquet --output /dev/shm/robbert_eval/blimp-robbert-2023.json
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

import argparse  # noqa: E402
import json  # noqa: E402
import pickle  # noqa: E402
import time  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from flax import jax_utils  # noqa: E402

from mmbjax.hf_encoders import HFEncoderConfig, HFEncoderForMaskedLM, load_tokenizer  # noqa: E402


def blimp_helpers():
    """l2r_masks, is_punctuation, read_pairs and score_pairs of eval_blimp_nl.py. Its import starts train_minimal's
    runtime (as for our model), so it is imported only when needed."""
    import eval_blimp_nl

    return eval_blimp_nl


def byte_decoder():
    from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode

    return {char: byte for byte, char in bytes_to_unicode().items()}


def word_starts_bytelevel(tokens, is_punctuation, decoder=None):
    """Per token of a sentence (no <s>, </s>): whether it starts a word, as eval_blimp_nl.word_starts does for ▁."""
    decoder = decoder or byte_decoder()
    starts = []
    previous = ""
    bare = False
    for i, token in enumerate(tokens):
        raw = bytes(decoder[char] for char in token)
        space = raw.startswith(b" ")
        try:
            text = (raw[1:] if space else raw).decode("utf-8")
        except UnicodeDecodeError:
            starts.append(i == 0 or (space and not bare))
            previous = ""
            bare = False
            continue
        if i > 0 and bare and not space:
            starts.append(False)
        else:
            starts.append(i == 0 or space or (text != "" and is_punctuation(text[0]))
                          or (previous != "" and is_punctuation(previous[-1])))
        previous = text
        bare = space and text == ""
    return starts


def masked_rows(token_lists, tokenizer, eb):
    """Per masked copy: (<s> ids </s> with its l2r mask, the scored position, the target id, the sentence)."""
    decoder = byte_decoder()
    copies = []
    for s, token_ids in enumerate(token_lists):
        if not token_ids:
            raise ValueError(f"sentence {s} has no tokens")
        tokens = tokenizer.convert_ids_to_tokens(token_ids)
        sentence = np.asarray([tokenizer.cls_token_id, *token_ids, tokenizer.sep_token_id], dtype=np.int32)
        for target, masked in enumerate(eb.l2r_masks(word_starts_bytelevel(tokens, eb.is_punctuation, decoder))):
            inputs = sentence.copy()
            inputs[[1 + m for m in masked]] = tokenizer.mask_token_id
            copies.append((inputs, 1 + target, token_ids[target], s))
    return copies


def score_copies(model, params, copies, pad_token_id, rows_per_device):
    """The log-probability of each copy's target at its position, in batches over the local devices; params replicated
    on them (flax.jax_utils.replicate)."""
    devices = jax.local_device_count()
    length = -(-max(len(c[0]) for c in copies) // 8) * 8
    per_batch = rows_per_device * devices

    def forward(params, ids, mask, positions, targets):
        logits = model.apply({"params": params}, ids, mask, positions=positions)
        log_probs = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
        return jnp.take_along_axis(log_probs, targets[:, None], axis=-1)[:, 0]

    p_forward = jax.pmap(forward)
    out = np.full(len(copies), np.nan)
    for start in range(0, len(copies), per_batch):
        chunk = copies[start:start + per_batch]
        filled = chunk + [chunk[0]] * (per_batch - len(chunk))
        ids = np.full((per_batch, length), pad_token_id, dtype=np.int32)
        mask = np.zeros((per_batch, length), dtype=np.float32)
        for r, (inputs, _, _, _) in enumerate(filled):
            ids[r, :len(inputs)] = inputs
            mask[r, :len(inputs)] = 1
        positions = np.asarray([c[1] for c in filled], dtype=np.int32)
        targets = np.asarray([c[2] for c in filled], dtype=np.int32)
        shards = [x.reshape(devices, rows_per_device, *x.shape[1:]) for x in (ids, mask, positions, targets)]
        log_probs = np.asarray(p_forward(params, *shards), dtype=np.float64)
        out[start:start + len(chunk)] = log_probs.reshape(-1)[:len(chunk)]
    if np.isnan(out).any():
        raise RuntimeError(f"{int(np.isnan(out).sum())} masked copies have no log-probability")
    return out


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="BLiMP-NL accuracy of a RoBERTa or NeoBERT MLM head, PLL-word-l2r")
    parser.add_argument("--checkpoint", required=True, help="Directory of scripts/convert_hf_encoder.py")
    parser.add_argument("--data", required=True, help="blimp-nl/large.parquet of experiments/dumb-extra/stage_data.py")
    parser.add_argument("--max-pairs", type=int, default=None, help="The first this many pairs (smoke tests)")
    parser.add_argument("--rows-per-device", type=int, default=256)
    parser.add_argument("--output", required=True, help="Local JSON file")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    started = time.time()
    eb = blimp_helpers()
    checkpoint = Path(args.checkpoint)
    config = HFEncoderConfig.from_dict(json.loads((checkpoint / "config.json").read_text()))
    with open(checkpoint / "params.pkl", "rb") as f:
        params = pickle.load(f)
    tokenizer = load_tokenizer(config)
    model = HFEncoderForMaskedLM(config)

    pairs = eb.read_pairs(args.data, args.max_pairs)
    sentences = [pair[side] for pair in pairs for side in ("grammatical", "ungrammatical")]
    token_lists = tokenizer(sentences, add_special_tokens=False, split_special_tokens=True)["input_ids"]
    copies = masked_rows(token_lists, tokenizer, eb)
    if max(len(c[0]) for c in copies) > config.max_length:
        raise ValueError(f"a sentence is longer than the {config.max_length} tokens of {config.model_name}")
    log_probs = score_copies(model, jax_utils.replicate(params), copies,
                             tokenizer.pad_token_id, args.rows_per_device)
    pll = np.zeros(len(sentences))
    np.add.at(pll, np.asarray([c[3] for c in copies]), log_probs)
    pll_grammatical, pll_ungrammatical = pll[0::2], pll[1::2]
    lengths = [len(t) for t in token_lists]

    result = {
        "checkpoint": args.checkpoint, "model": config.model_name, "revision": config.model_revision, "data": args.data,
        "tokenizer": config.tokenizer_name, "metric": "PLL-word-l2r", "compute_dtype": "float32",
        "vocab_size": config.vocab_size, "mask_token_id": tokenizer.mask_token_id, "max_pairs": args.max_pairs,
        "sentences": len(sentences), "tokens": int(sum(lengths)), "masked_copies": len(copies),
        **eb.score_pairs(pairs, pll_grammatical, pll_ungrammatical),
        "pair_scores": [[pair["item_nr"], float(g), float(u), lengths[2 * i], lengths[2 * i + 1]]
                        for i, (pair, g, u) in enumerate(zip(pairs, pll_grammatical, pll_ungrammatical))],
    }
    result["seconds"] = time.time() - started
    print(f"BLIMP accuracy={result['accuracy']:.4f} handcrafted={result['subsets']['handcrafted']['accuracy']:.4f} "
          f"generated={result['subsets']['generated']['accuracy']:.4f} pairs={result['pairs']} ties={result['ties']} "
          f"copies={len(copies)} seconds={result['seconds']:.0f}", flush=True)
    for number, phenomenon in result["phenomena"].items():
        print(f"BLIMP phenomenon {number} {phenomenon['name']}: {phenomenon['accuracy']:.4f}", flush=True)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    main()
