#!/usr/bin/env python3
"""MLM loss of a checkpoint on held-out documents, cut into windows of several lengths.

Usage (on every host of a multi-host VM, like the trainer):
    python scripts/eval_windows.py --checkpoint gs://bucket/run/ckpt/20000 --eval-file ~/heldout.parquet \\
        --tokenizer yhavinga/dutch-llama-tokenizer --hidden-size 1024 --num-layers 22 --num-heads 16 \\
        --intermediate-size 1536 --windows 1024 2048 --output gs://bucket/run/eval/20000.json

Each document is tokenized as the trainer does and masked once, with 15% single tokens and the
80/10/10 replacement, from a generator seeded with (--seed, document index). The mask belongs to
the document, not to the window: every window length and every model is measured on the same
masked tokens, whatever masking the model was trained with. For a window w the document is cut
into the pieces the trainer would make at --max-length w (each with <s>, the last with </s>), and
the pieces are packed into rows of w with segment ids, so no token sees another document.

Reported per window: the loss on (a) all masked tokens, (b) the masked tokens of documents longer
than the smallest window (more tokens than it holds with <s> and </s>), and, for longer windows,
(c) per position bucket, the position within the piece: [0, smallest window) and [smallest window, w).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import train_minimal as tm  # noqa: E402  # Its import initializes the multi-host runtime, as the trainer's does
from etils import epath  # noqa: E402
from jax.experimental import multihost_utils  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from mmbjax.configuration_mmbbert import MMBertConfig  # noqa: E402
from mmbjax.heads.masked_lm import ModernBertForMaskedLM  # noqa: E402
from mmbjax.training.packing import split_into_pieces  # noqa: E402

MLM_PROBABILITY = 0.15
# The groups of masked tokens a window reports on; see group_masks
GROUPS = ("all", "long_documents", "positions_low", "positions_high", "long_documents_positions_low")


def mask_document(tokens, document_index, seed, special_ids, mask_token_id, vocab_size):
    """(input ids, labels) of one document's tokens, the same for every window, model and host count.

    The same draws as PackedMLMDataCollator (15%, then 80% <mask>, 10% a random id below
    vocab_size, 10% unchanged), from a generator of this document only.
    """
    tokens = np.asarray(tokens, dtype=np.int64)
    rng = np.random.default_rng([seed, document_index])
    masked = (rng.random(tokens.size) < MLM_PROBABILITY) & ~np.isin(tokens, special_ids)
    replace = rng.random(tokens.size)
    random_ids = rng.integers(vocab_size, size=tokens.size)
    inputs = tokens.copy()
    inputs[masked & (replace < 0.8)] = mask_token_id
    random_positions = masked & (replace >= 0.8) & (replace < 0.9)
    inputs[random_positions] = random_ids[random_positions]
    return inputs, np.where(masked, tokens, -100)


def document_pieces(inputs, labels, window, bos_token_id, eos_token_id):
    """The pieces the trainer cuts at --max-length window: (input ids, labels) per piece, <s> and </s> unmasked."""
    pieces = []
    for indices in split_into_pieces(list(range(inputs.size)), window, -1, -2):
        indices = np.asarray(indices)
        body = indices[(indices >= 0)]
        piece_inputs = np.r_[bos_token_id, inputs[body]]
        piece_labels = np.r_[-100, labels[body]]
        if indices[-1] == -2:
            piece_inputs = np.r_[piece_inputs, eos_token_id]
            piece_labels = np.r_[piece_labels, -100]
        pieces.append((piece_inputs, piece_labels))
    return pieces


def pack_rows(pieces, window, pad_token_id, open_rows=64):
    """Pieces (input ids, labels, is_long) into rows of length window, first fit among the last open_rows rows.

    Deterministic, so every model sees the same rows. Returns arrays of shape (rows, window).
    """
    rows = []  # each: list of (inputs, labels, is_long)
    room = []
    open_indices = []
    for piece in pieces:
        length = piece[0].size
        target = next((i for i in open_indices if room[i] >= length), None)
        if target is None:
            rows.append([])
            room.append(window)
            target = len(rows) - 1
            open_indices.append(target)
            if len(open_indices) > open_rows:
                open_indices.pop(0)
        rows[target].append(piece)
        room[target] -= length
    n = len(rows)
    arrays = {
        "input_ids": np.full((n, window), pad_token_id, dtype=np.int32),
        "labels": np.full((n, window), -100, dtype=np.int32),
        "attention_mask": np.zeros((n, window), dtype=np.int32),
        "position_ids": np.zeros((n, window), dtype=np.int32),
        # Padding gets a segment of its own, as in PackedMLMDataCollator
        "segment_ids": np.full((n, window), -1, dtype=np.int32),
        "long_document": np.zeros((n, window), dtype=np.int32),
    }
    for r, row in enumerate(rows):
        start = 0
        for segment, (inputs, labels, is_long) in enumerate(row):
            end = start + inputs.size
            arrays["input_ids"][r, start:end] = inputs
            arrays["labels"][r, start:end] = labels
            arrays["attention_mask"][r, start:end] = 1
            arrays["position_ids"][r, start:end] = np.arange(inputs.size)
            arrays["segment_ids"][r, start:end] = segment
            arrays["long_document"][r, start:end] = int(is_long)
            start = end
    return arrays


def pad_to_rows(rows, n, pad_token_id):
    """The first n rows, with rows of padding added when there are fewer."""
    have = len(rows["input_ids"])
    if have >= n:
        return {key: value[:n] for key, value in rows.items()}
    fill = {"input_ids": pad_token_id, "labels": -100, "segment_ids": -1}
    return {key: np.concatenate([value, np.full((n - have,) + value.shape[1:], fill.get(key, 0), dtype=value.dtype)])
            for key, value in rows.items()}


def group_masks(labels, position_ids, long_document, boundary):
    """Boolean (..., len(GROUPS)) of which group each token counts in; only masked tokens count."""
    masked = labels != -100
    long_masked = masked & (long_document == 1)
    low = position_ids < boundary
    return jnp.stack([masked, long_masked, masked & low, masked & ~low, long_masked & low], axis=-1)


def read_documents(path, num_documents):
    """The first num_documents texts of the parquet file, or all of them for None."""
    import pyarrow.parquet as pq

    texts = []
    for record_batch in pq.ParquetFile(path).iter_batches(batch_size=1024, columns=["text"]):
        texts.extend(record_batch.column("text").to_pylist())
        if num_documents is not None and len(texts) >= num_documents:
            return texts[:num_documents]
    if num_documents is not None and len(texts) < num_documents:
        raise ValueError(f"{path} has {len(texts)} documents, fewer than --num-documents {num_documents}")
    return texts


def build_model(args, tokenizer):
    """The model of train_minimal.py for these flags: the same vocabulary rounding and config defaults."""
    vocab_size = ((len(tokenizer) + 127) // 128) * 128
    config = MMBertConfig(
        vocab_size=vocab_size,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        intermediate_size=args.intermediate_size,
        global_attn_every_n_layers=args.global_attn_every_n_layers,
        max_position_embeddings=8192,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        compute_dtype=args.compute_dtype,
        param_dtype="float32",
    )
    model = ModernBertForMaskedLM(config, dtype=getattr(jnp, args.compute_dtype), param_dtype=jnp.float32)
    return config, model


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="MLM loss of a checkpoint per window length, on fixed masks")
    parser.add_argument("--checkpoint", required=True, help="Step directory of a train_minimal.py checkpoint")
    parser.add_argument("--eval-file", required=True, help="Parquet file with a 'text' column")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--windows", type=int, nargs="+", default=[1024, 2048])
    parser.add_argument("--num-documents", type=int, default=None, help="The first this many documents (default: all)")
    parser.add_argument("--max-batches", type=int, default=None, help="Stop each window after this many global batches (smoke tests)")
    parser.add_argument("--tokens-per-device", type=int, default=8192,
                        help="Rows per device per batch are this divided by the window; 8192 is the trainer's 8 x 1024 "
                             "and 4 x 2048 (default: 8192)")
    parser.add_argument("--seed", type=int, default=0, help="Seed of the masks; keep it fixed across models (default: 0)")
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--num-layers", type=int, default=22)
    parser.add_argument("--num-heads", type=int, default=12)
    parser.add_argument("--intermediate-size", type=int, default=3072)
    parser.add_argument("--global-attn-every-n-layers", type=int, default=3)
    parser.add_argument("--compute-dtype", choices=["float32", "bfloat16"], default=None,
                        help="Default: bfloat16 on TPU, float32 elsewhere, as in the trainer")
    parser.add_argument("--output", default=None, help="JSON file, local or gs:// (default: print only)")
    args = parser.parse_args(argv)
    if sorted(set(args.windows)) != args.windows:
        parser.error("--windows must be distinct and ascending")
    for window in args.windows:
        if args.tokens_per_device % window:
            parser.error(f"--tokens-per-device {args.tokens_per_device} is no multiple of window {window}")
    if args.compute_dtype is None:
        args.compute_dtype = "bfloat16" if jax.default_backend() == "tpu" else "float32"
    return args


def main(argv=None):
    args = parse_args(argv)
    started = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, split_special_tokens=True)
    if tokenizer.mask_token is None:
        # As the trainer does, so the vocabulary and the mask id are the trainer's
        tokenizer.add_special_tokens({"mask_token": "<mask>"})
    config, model = build_model(args, tokenizer)
    if tm.is_coordinator():
        print(f"Model: {config.num_hidden_layers} layers, hidden {config.hidden_size}, vocab {config.vocab_size}, "
              f"global attention every {config.global_attn_every_n_layers} layers, global_rope_theta "
              f"{config.global_rope_theta}, local_rope_theta {config.local_rope_theta}, local_attention "
              f"{config.local_attention}, compute dtype {args.compute_dtype}", flush=True)

    replicated, batch_sharding = tm.create_shardings()
    ones = jnp.ones((1, args.windows[0]), dtype=jnp.int32)
    # Shapes only: the checkpoint fills them, in the replicated sharding the trainer uses
    shapes = jax.eval_shape(lambda: model.init(jax.random.PRNGKey(0), input_ids=ones, attention_mask=ones, deterministic=True)["params"])
    target = jax.tree_util.tree_map(lambda leaf: jax.ShapeDtypeStruct(leaf.shape, leaf.dtype, sharding=replicated), shapes)
    params = tm.load_params_of_run(args.checkpoint, target)

    texts = read_documents(args.eval_file, args.num_documents)
    # Each host tokenizes and packs its own share of the documents; the masks don't depend on the split
    mine = list(range(jax.process_index(), len(texts), jax.process_count()))
    token_lists = tokenizer([texts[i] for i in mine], add_special_tokens=False, split_special_tokens=True)["input_ids"]
    special_ids = np.asarray(tokenizer.all_special_ids)
    masked_documents = [
        mask_document(tokens, index, args.seed, special_ids, tokenizer.mask_token_id, tokenizer.vocab_size)
        for index, tokens in zip(mine, token_lists)
    ]
    smallest = args.windows[0]
    is_long = [len(tokens) + 2 > smallest for tokens in token_lists]

    def eval_step(params, batch):
        logits = model.apply({"params": params}, input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                             position_ids=batch["position_ids"], segment_ids=batch["segment_ids"], deterministic=True).logits
        # float32 as in the trainer's loss: log_softmax over the vocabulary loses precision in bfloat16
        log_probs = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
        labels = batch["labels"]
        nll = -jnp.take_along_axis(log_probs, jnp.maximum(labels, 0)[..., None], axis=-1)[..., 0]
        correct = jnp.argmax(log_probs, axis=-1) == labels
        groups = group_masks(labels, batch["position_ids"], batch["long_document"], smallest)
        return (
            jnp.sum(jnp.where(groups, nll[..., None], 0.0), axis=(0, 1)),
            jnp.sum(correct[..., None] & groups, axis=(0, 1)),
            jnp.sum(groups, axis=(0, 1)),
        )

    eval_step_jit = jax.jit(eval_step, out_shardings=replicated)
    result = {
        "checkpoint": args.checkpoint, "eval_file": args.eval_file, "seed": args.seed, "mlm_probability": MLM_PROBABILITY,
        "documents": len(texts), "process_count": jax.process_count(), "device_count": jax.device_count(),
        "compute_dtype": args.compute_dtype, "long_document_threshold": smallest,
        "model": {"global_rope_theta": config.global_rope_theta, "local_rope_theta": config.local_rope_theta,
                  "local_attention": config.local_attention, "max_position_embeddings": config.max_position_embeddings},
        "windows": {},
    }
    for window in args.windows:
        pieces = [
            (inputs, labels, long_document)
            for (masked_inputs, masked_labels), long_document in zip(masked_documents, is_long)
            for inputs, labels in document_pieces(masked_inputs, masked_labels, window, tokenizer.bos_token_id, tokenizer.eos_token_id)
        ]
        rows = pack_rows(pieces, window, tokenizer.pad_token_id)
        rows_per_host = args.tokens_per_device // window * jax.local_device_count()
        # Every host runs every batch: the one with the most rows decides how many, the others pad
        local_batches = -(-len(rows["input_ids"]) // rows_per_host)
        needed = int(np.max(np.asarray(multihost_utils.process_allgather(np.array([local_batches], dtype=np.int32)))))
        n_batches = needed if args.max_batches is None else min(needed, args.max_batches)
        padded = pad_to_rows(rows, n_batches * rows_per_host, tokenizer.pad_token_id)
        loss_sum = np.zeros(len(GROUPS))
        correct = np.zeros(len(GROUPS))
        count = np.zeros(len(GROUPS), dtype=np.int64)
        for b in range(n_batches):
            local = {key: value[b * rows_per_host:(b + 1) * rows_per_host] for key, value in padded.items()}
            batch_loss, batch_correct, batch_count = eval_step_jit(params, tm.to_global(local, batch_sharding))
            loss_sum += np.asarray(batch_loss, dtype=np.float64)
            correct += np.asarray(batch_correct, dtype=np.float64)
            count += np.asarray(batch_count, dtype=np.int64)
        groups = {}
        for g, name in enumerate(GROUPS):
            if window == smallest and name.endswith(("positions_high", "positions_low")):
                continue  # every position of the smallest window is below the boundary: same as "all" and "long_documents"
            groups[name] = {"loss": loss_sum[g] / count[g] if count[g] else None,
                            "accuracy": correct[g] / count[g] if count[g] else None, "masked": int(count[g])}
        host_counts = np.asarray(multihost_utils.process_allgather(np.array(
            [len(pieces), len(rows["input_ids"]), int(rows["attention_mask"].sum()), sum(is_long)], dtype=np.int32))).reshape(-1, 4).sum(axis=0)
        result["windows"][str(window)] = {
            "groups": groups, "position_boundary": smallest, "batches": n_batches, "rows_per_batch": rows_per_host * jax.process_count(),
            # False when --max-batches cut it short: then not all documents counted
            "complete": n_batches == needed,
            "pieces": int(host_counts[0]), "rows": int(host_counts[1]), "tokens": int(host_counts[2]), "long_documents": int(host_counts[3]),
        }
        if tm.is_coordinator():
            print(f"WINDOW {window} {json.dumps(result['windows'][str(window)])}", flush=True)
    result["seconds"] = time.time() - started
    if tm.is_coordinator():
        print(f"WINDOW_EVAL {json.dumps(result)}", flush=True)
        if args.output:
            tm.write_file(epath.Path(args.output), (json.dumps(result, indent=2) + "\n").encode())
    return result


if __name__ == "__main__":
    import os
    import traceback

    try:
        main()
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        # As in train_minimal.py: a normal exit would wait in the shutdown barrier for peers stuck in a collective
        os._exit(1)
