#!/usr/bin/env python3
"""The memorization test of dmbert v2 (protocol.md, "Memorisatietest"): does a checkpoint reconstruct documents of its
training data better than documents of the same source that it never saw?

Input: the sample of stage_optout.py memsample, N documents of kept/train and N of the held-out set (kept/test less
what is also in kept/train), the i-th of each split matched on token count, each one piece of the trainer's (at most
1,022 tokens plus <s> and </s>). Every document is one row, so no token sees another document. Per document, on masks
that depend on the document only (the same for every model):

  mlm   15% single tokens with the 80/10/10 replacement of the trainer (eval_windows.mask_document), mask seeds 0, 1, 2:
        the loss and the accuracy on the masked tokens;
  span  SPAN consecutive tokens replaced by <mask>, at a start drawn per document and seed (seeds 0, 1, 2): the
        accuracy of the argmax on them, and whether all SPAN are right.

Reported per split the mean over documents, and per metric the difference train - test over the matched pairs: its
mean, a 95% bootstrap interval (10,000 resamples of the pairs, seed 0) and a two-sided sign-flip permutation p-value
(10,000 flips, seed 0). Single host only (a TPU of 4 chips, as the evaluations on mbert-mix). It reports; it decides
nothing.

    python experiments/final-v2/memtest.py --checkpoint gs://.../ckpt_keep/1306000 --sample sample.parquet \\
        --tokenizer yhavinga/dutch-llama-tokenizer --hidden-size 1024 --num-layers 22 --num-heads 16 \\
        --intermediate-size 1536 --output gs://.../memtest/v2-1306000.json
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import eval_windows as ew  # noqa: E402  # imports train_minimal, which initializes the runtime
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from etils import epath  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

tm = ew.tm
SPAN = 16
SEEDS = (0, 1, 2)
METRICS = ("mlm_loss", "mlm_accuracy", "span_accuracy", "span_exact")
RESAMPLES = 10_000


def span_mask(tokens, document_index, seed, mask_token_id, span=SPAN):
    """(input ids, labels): span consecutive tokens of the document as <mask>, from a start of this document and seed."""
    tokens = np.asarray(tokens, dtype=np.int64)
    if tokens.size < span:
        raise ValueError(f"document {document_index} has {tokens.size} tokens, fewer than a span of {span}")
    start = int(np.random.default_rng([1000 + seed, document_index]).integers(0, tokens.size - span + 1))
    inputs, labels = tokens.copy(), np.full(tokens.size, -100)
    inputs[start:start + span] = mask_token_id
    labels[start:start + span] = tokens[start:start + span]
    return inputs, labels


def row_of(inputs, labels, window, bos, eos, pad):
    """One document as one row of the window: <s> tokens </s>, then padding in a segment of its own."""
    n = inputs.size + 2
    if n > window:
        raise ValueError(f"a document of {inputs.size} tokens does not fit a row of {window} with <s> and </s>")
    row = {"input_ids": np.full(window, pad, np.int32), "labels": np.full(window, -100, np.int32),
           "attention_mask": np.zeros(window, np.int32), "position_ids": np.zeros(window, np.int32),
           "segment_ids": np.full(window, -1, np.int32)}
    row["input_ids"][:n] = np.r_[bos, inputs, eos]
    row["labels"][1:n - 1] = labels
    row["attention_mask"][:n] = 1
    row["position_ids"][:n] = np.arange(n)
    row["segment_ids"][:n] = 0
    return row


def paired(train, test, seed=0, resamples=RESAMPLES):
    """Mean of train - test over the pairs, its 95% bootstrap interval, and the two-sided sign-flip p-value."""
    d = np.asarray(train, dtype=np.float64) - np.asarray(test, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, d.size, size=(resamples, d.size))].mean(axis=1)
    flips = rng.choice([-1.0, 1.0], size=(resamples, d.size))
    null = np.abs((flips * d).mean(axis=1))
    p = (1 + np.sum(null >= abs(d.mean()) - 1e-12)) / (resamples + 1)
    return {"difference": float(d.mean()), "ci95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
            "p_sign_flip": float(p), "pairs": int(d.size)}


def read_sample(path):
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=["text", "split", "tokens"])
    texts, splits = table.column("text").to_pylist(), table.column("split").to_pylist()
    n = splits.count("test")
    if splits != ["test"] * n + ["train"] * n or n == 0:
        raise ValueError(f"{path}: not N test documents followed by N train documents ({splits.count('test')} test, "
                         f"{splits.count('train')} train)")
    return texts, splits, table.column("tokens").to_pylist()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Memorization test: kept/train against held-out documents, matched pairs")
    parser.add_argument("--checkpoint", required=True, help="Step directory of a train_minimal.py checkpoint")
    parser.add_argument("--sample", required=True, help="memsample parquet of stage_optout.py")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--window", type=int, default=1024)
    parser.add_argument("--rows-per-device", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--num-layers", type=int, default=22)
    parser.add_argument("--num-heads", type=int, default=12)
    parser.add_argument("--intermediate-size", type=int, default=3072)
    parser.add_argument("--global-attn-every-n-layers", type=int, default=3)
    parser.add_argument("--compute-dtype", choices=["float32", "bfloat16"], default="float32",
                        help="float32 by default: the differences are small, and every model is measured alike")
    parser.add_argument("--output", default=None, help="JSON file, local or gs:// (default: print only)")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if jax.process_count() != 1:
        raise RuntimeError(f"memtest.py runs on one host, not {jax.process_count()}: every host would need the same batches")
    started = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, split_special_tokens=True)
    if tokenizer.mask_token is None:
        tokenizer.add_special_tokens({"mask_token": "<mask>"})
    config, model = ew.build_model(args, tokenizer)
    replicated, batch_sharding = tm.create_shardings()
    ones = jnp.ones((1, args.window), dtype=jnp.int32)
    init_params = tm.to_global(model.init(jax.random.PRNGKey(0), input_ids=ones, attention_mask=ones, deterministic=True)["params"], replicated)
    params = tm.load_params_of_run(args.checkpoint, init_params)
    del init_params

    texts, splits, stored = read_sample(args.sample)
    token_lists = tokenizer(texts, add_special_tokens=False, split_special_tokens=True)["input_ids"]
    counted = [len(t) for t in token_lists]
    if counted != stored:
        raise ValueError(f"{args.sample}: the tokenizer counts other lengths than the sample stored "
                         f"(first difference at document {next(i for i, (a, b) in enumerate(zip(counted, stored)) if a != b)})")
    special_ids = np.asarray(tokenizer.all_special_ids)
    rows, owner = [], []
    for kind in ("mlm", "span"):
        for seed in SEEDS:
            for index, tokens in enumerate(token_lists):
                if kind == "mlm":
                    inputs, labels = ew.mask_document(tokens, index, seed, special_ids, tokenizer.mask_token_id, tokenizer.vocab_size)
                else:
                    inputs, labels = span_mask(tokens, index, seed, tokenizer.mask_token_id)
                rows.append(row_of(inputs, labels, args.window, tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id))
                owner.append((kind, index))

    def eval_step(params, batch):
        logits = model.apply({"params": params}, input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                             position_ids=batch["position_ids"], segment_ids=batch["segment_ids"], deterministic=True).logits
        log_probs = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
        labels = batch["labels"]
        labeled = labels != -100
        nll = -jnp.take_along_axis(log_probs, jnp.maximum(labels, 0)[..., None], axis=-1)[..., 0]
        correct = (jnp.argmax(log_probs, axis=-1) == labels) & labeled
        return jnp.sum(jnp.where(labeled, nll, 0.0), axis=1), jnp.sum(correct, axis=1), jnp.sum(labeled, axis=1)

    eval_step_jit = jax.jit(eval_step, out_shardings=replicated)
    per_batch = args.rows_per_device * jax.local_device_count()
    empty = row_of(np.zeros(0, np.int64), np.zeros(0, np.int64), args.window, tokenizer.bos_token_id,
                   tokenizer.eos_token_id, tokenizer.pad_token_id)
    nll_sum, correct, labeled = (np.zeros(len(rows)) for _ in range(3))
    for start in range(0, len(rows), per_batch):
        chunk = rows[start:start + per_batch]
        chunk = chunk + [empty] * (per_batch - len(chunk))
        local = {key: np.stack([r[key] for r in chunk]) for key in empty}
        a, b, c = (np.asarray(x, dtype=np.float64) for x in eval_step_jit(params, tm.to_global(local, batch_sharding)))
        n = min(per_batch, len(rows) - start)
        nll_sum[start:start + n], correct[start:start + n], labeled[start:start + n] = a[:n], b[:n], c[:n]

    documents = len(texts)
    values = {m: np.zeros(documents) for m in METRICS}
    mlm = {k: [0.0, 0.0, 0.0] for k in range(documents)}
    span = {k: [0.0, 0.0] for k in range(documents)}
    for r, (kind, index) in enumerate(owner):
        if kind == "mlm":
            mlm[index][0] += nll_sum[r]
            mlm[index][1] += correct[r]
            mlm[index][2] += labeled[r]
        else:
            if labeled[r] != SPAN:
                raise RuntimeError(f"a span row of document {index} has {labeled[r]} labels, not {SPAN}")
            span[index][0] += correct[r] / SPAN
            span[index][1] += float(correct[r] == SPAN)
    for k in range(documents):
        if mlm[k][2] == 0:
            raise ValueError(f"document {k} got no masked token in {len(SEEDS)} seeds")
        values["mlm_loss"][k] = mlm[k][0] / mlm[k][2]
        values["mlm_accuracy"][k] = mlm[k][1] / mlm[k][2]
        values["span_accuracy"][k] = span[k][0] / len(SEEDS)
        values["span_exact"][k] = span[k][1] / len(SEEDS)
    half = documents // 2
    result = {
        "checkpoint": args.checkpoint, "sample": args.sample, "documents_per_split": half, "span": SPAN, "seeds": list(SEEDS),
        "compute_dtype": args.compute_dtype, "window": args.window,
        "mean_tokens": {"test": float(np.mean(counted[:half])), "train": float(np.mean(counted[half:]))},
        "splits": {split: {m: float(values[m][sl].mean()) for m in METRICS}
                   for split, sl in (("test", slice(0, half)), ("train", slice(half, documents)))},
        "train_minus_test": {m: paired(values[m][half:], values[m][:half]) for m in METRICS},
        "per_document": {m: [round(float(v), 6) for v in values[m]] for m in METRICS},
        "seconds": None,
    }
    result["seconds"] = time.time() - started
    summary = {k: v for k, v in result.items() if k != "per_document"}
    print(f"MEMTEST {json.dumps(summary)}", flush=True)
    if args.output:
        tm.write_file(epath.Path(args.output), (json.dumps(result, indent=1) + "\n").encode())
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
        os._exit(1)
