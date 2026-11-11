#!/usr/bin/env python3
"""The naturalness measures of the fine-tune's arms with our Dutch ModernBERT S @ 653k (protocol.md §4.2.1 and
§4.2.3), on one host as a TPU of its own 4 chips, in the venv of the DUMB rounds with a checkout of branch dumb-extra
(--dumb-code; its scripts and src are imported, nothing of them is copied here).

    python naturalness.py classifier --dumb-code DIR --checkpoint CONVERTED --data classifier.parquet --out DIR
    python naturalness.py classify --dumb-code DIR --model DIR --texts texts.parquet --out predictions.parquet
    python naturalness.py pll --dumb-code DIR --checkpoint gs://.../ckpt_keep/653000 --texts texts.parquet --out pll.parquet

classifier: binary "original or translated" (label 1 = the Dutch original, 0 = the base model's back-translation;
armeval.py classifier-data), fine-tuned with scripts/dumb/finetune.py's model, optimizer, collate function and train
step (AdamW at a constant LR, weight decay 0.01): 3 epochs at LR 5e-5, seed 42, 8 per chip, at most 1,024 tokens,
padded to a multiple of 128, shuffled per epoch as finetune.py does. No validation split and no choice of a best epoch:
the state after the last epoch is tested on every example of the test split (the last batch padded, its padding left
out) and saved (params.pkl, config.json); results.json has the accuracy. The CONVERTED checkpoint is the output of
scripts/convert_gcs_checkpoint.py --params-only (params.pkl, config.json), as dumb-extra's dumbx.sh makes it.
classify: P(original) per text of texts.parquet (model, set, id, text) with the saved classifier.
pll: PLL-word-l2r per text under the MLM head (scripts/eval_blimp_nl.py's words, masks, packing and model), float32,
head 0; a text longer than ROW_LENGTH - 2 tokens of S's tokenizer is left out (pll null) and counted.
"""

import argparse
import json
import sys
import time
from functools import partial
from pathlib import Path

S_SHAPE = {"hidden_size": 1024, "num_layers": 22, "num_heads": 16, "intermediate_size": 1536,
           "global_attn_every_n_layers": 3}  # experiments/dumb-extra/blimp.sh, the models of the mbert runs
ROW_LENGTH = 1024


def dumb_paths(code):
    code = Path(code)
    for p in (code / "src", code / "scripts", code / "scripts" / "dumb"):
        if not p.is_dir():
            raise SystemExit(f"--dumb-code {code}: no {p}")
        sys.path.insert(0, str(p))


def warm_rotary_cache(model, params, max_length):
    """One eager forward pass at max_length. mmbjax's rotary cache is global and keeps what its first build made; built
    inside the first pmap trace that is a tracer, which a later trace of another padded length reuses
    (UnexpectedTracerError, classify on 11 Nov). Built eagerly it holds concrete arrays for every length up to
    max_length; finetune.py's eager model.init does the same up to 128."""
    import jax.numpy as jnp

    ones = jnp.ones((1, max_length), jnp.int32)
    model.apply({"params": params}, ones, attention_mask=ones, deterministic=True)


def make_predict(model):
    import jax

    def logits(params, input_ids, attention_mask):
        out = model.apply({"params": params}, input_ids, attention_mask=attention_mask, deterministic=True)
        return jax.nn.softmax(out.logits.astype("float32"), axis=-1)[..., 1]

    return jax.pmap(logits, axis_name="batch")


def predict(ft, p_predict, params, examples, collate, total):
    """P(label 1) of every example, in order: full batches of `total`, the last one filled with copies of the first
    example and those left out."""
    import numpy as np

    out = []
    for i in range(0, len(examples), total):
        chunk = examples[i:i + total]
        filled = chunk + [examples[0]] * (total - len(chunk))
        batch = ft.shard_batch(collate(filled))
        probs = np.asarray(p_predict(params, batch["input_ids"], batch["attention_mask"])).reshape(-1)
        out.extend(float(p) for p in probs[:len(chunk)])
    return out


def classifier(args):
    dumb_paths(args.dumb_code)
    import finetune as ft  # its import counts the devices
    import jax
    import numpy as np
    import pyarrow.parquet as pq
    from flax import jax_utils
    from transformers import AutoTokenizer

    if (jax.device_count(), jax.process_count()) != (4, 1):
        raise SystemExit(f"{jax.device_count()} devices, {jax.process_count()} processes: one host of 4 chips")
    started = time.time()
    np.random.seed(args.seed)  # finetune.py's epoch shuffles draw from NumPy
    params, config = ft.load_pretrained_params(Path(args.checkpoint))
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name or "yhavinga/dutch-llama-tokenizer")
    rows = pq.read_table(args.data).to_pylist()
    split = {s: [r for r in rows if r["split"] == s] for s in ("train", "test")}

    def encode(part):
        ids = tokenizer([r["text"] for r in part], truncation=True, max_length=args.max_length)["input_ids"]
        return [{"input_ids": i, "label": int(r["label"])} for i, r in zip(ids, part)]

    train, test = encode(split["train"]), encode(split["test"])
    truncated = {s: sum(len(e["input_ids"]) >= args.max_length for e in d) for s, d in (("train", train), ("test", test))}
    model, state = ft.create_model_and_state(config=config, pretrained_params=params, num_labels=2,
                                             learning_rate=args.learning_rate, rng=jax.random.PRNGKey(args.seed))
    state = jax_utils.replicate(state)
    collate = partial(ft.collate_fn, pad_token_id=tokenizer.pad_token_id, max_length=args.max_length,
                      pad_to_multiple_of=args.pad_to_multiple_of)
    total = args.batch_size * ft.NUM_DEVICES
    epochs = []
    steps = 0
    for epoch in range(args.epochs):
        order = list(range(len(train)))
        np.random.shuffle(order)
        losses, accs = [], []
        for i in range(0, len(order), total):
            chunk = order[i:i + total]
            if len(chunk) < total:  # as finetune.py: the incomplete last batch is skipped
                continue
            state, m = ft.p_train_step(state, ft.shard_batch(collate([train[j] for j in chunk])))
            losses.append(float(m["loss"][0]))
            accs.append(float(m["accuracy"][0]))
            steps += 1
        epochs.append({"epoch": epoch + 1, "loss": float(np.mean(losses)), "train_accuracy": float(np.mean(accs))})
        print(f"EPOCH {json.dumps(epochs[-1])}", flush=True)
    p_predict = make_predict(model)
    probs = predict(ft, p_predict, state.params, test, collate, total)
    right = [(p >= 0.5) == bool(e["label"]) for p, e in zip(probs, test)]
    by_source = {}
    for r, ok in zip(split["test"], right):
        by_source.setdefault(r["source"], []).append(ok)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    import pickle

    with open(out / "params.pkl", "wb") as f:
        pickle.dump(jax.device_get(jax_utils.unreplicate(state.params)), f)
    config_dict = config.to_dict()
    config_dict["num_labels"] = 2
    (out / "config.json").write_text(json.dumps(config_dict, indent=2))
    results = {
        "checkpoint": args.checkpoint, "data": args.data, "train_examples": len(train), "test_examples": len(test),
        "test_accuracy": sum(right) / len(right), "test_accuracy_by_source": {s: sum(v) / len(v) for s, v in by_source.items()},
        "test_accuracy_original": sum(ok for ok, e in zip(right, test) if e["label"] == 1) / sum(e["label"] == 1 for e in test),
        "test_accuracy_translated": sum(ok for ok, e in zip(right, test) if e["label"] == 0) / sum(e["label"] == 0 for e in test),
        "truncated_at_max_length": truncated, "epochs": epochs, "steps": steps, "learning_rate": args.learning_rate,
        "batch_size_per_device": args.batch_size, "total_batch_size": total, "max_length": args.max_length,
        "pad_to_multiple_of": args.pad_to_multiple_of, "seed": args.seed, "seconds": round(time.time() - started, 1),
        "code": "scripts/dumb/finetune.py (create_model_and_state, collate_fn, shard_batch, p_train_step) of --dumb-code",
    }
    (out / "results.json").write_text(json.dumps(results, indent=1))
    print(f"CLASSIFIER test_accuracy={results['test_accuracy']:.4f} on {len(test)} texts, original "
          f"{results['test_accuracy_original']:.4f}, translated {results['test_accuracy_translated']:.4f}, "
          f"{steps} steps, {results['seconds']} s", flush=True)


def classify(args):
    dumb_paths(args.dumb_code)
    import finetune as ft
    import jax
    import pyarrow as pa
    import pyarrow.parquet as pq
    from flax import jax_utils
    from transformers import AutoTokenizer

    if (jax.device_count(), jax.process_count()) != (4, 1):
        raise SystemExit(f"{jax.device_count()} devices, {jax.process_count()} processes: one host of 4 chips")
    params, config = ft.load_pretrained_params(Path(args.model))
    if getattr(config, "num_labels", None) != 2:
        raise SystemExit(f"{args.model}: num_labels {getattr(config, 'num_labels', None)}, not the classifier")
    from mmbjax.heads.classification import ModernBertForSequenceClassification

    model = ModernBertForSequenceClassification(config=config)
    warm_rotary_cache(model, params, args.max_length)
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name or "yhavinga/dutch-llama-tokenizer")
    rows = pq.read_table(args.texts).to_pylist()
    ids = tokenizer([r["text"] for r in rows], truncation=True, max_length=args.max_length)["input_ids"]
    examples = [{"input_ids": i, "label": 0} for i in ids]
    collate = partial(ft.collate_fn, pad_token_id=tokenizer.pad_token_id, max_length=args.max_length,
                      pad_to_multiple_of=args.pad_to_multiple_of)
    probs = predict(ft, make_predict(model), jax_utils.replicate(params), examples, collate,
                    args.batch_size * ft.NUM_DEVICES)
    out = [{"model": r["model"], "set": r["set"], "id": r["id"], "p_original": p,
            "truncated": len(i) >= args.max_length} for r, p, i in zip(rows, probs, ids)]
    pq.write_table(pa.Table.from_pylist(out), args.out)
    shares = {}
    for r in out:
        shares.setdefault(f"{r['model']}/{r['set']}", []).append(r["p_original"] >= 0.5)
    print(f"CLASSIFY {json.dumps({k: round(sum(v) / len(v), 4) for k, v in sorted(shares.items())})}", flush=True)


def pll(args):
    dumb_paths(args.dumb_code)
    import eval_blimp_nl as eb  # its import sets up the runtime of one host (blimp.sh's variables)
    import jax
    import jax.numpy as jnp
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    tm = eb.tm
    if jax.process_count() != 1 or jax.device_count() != jax.local_device_count():
        raise SystemExit(f"{jax.process_count()} processes: one host of 4 chips")
    started = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, split_special_tokens=True)
    if tokenizer.mask_token is None:
        tokenizer.add_special_tokens({"mask_token": "<mask>"})  # as the trainer and eval_blimp_nl.py do
    shape = argparse.Namespace(**S_SHAPE, compute_dtype="float32")
    config, model = eb.build_model(shape, tokenizer)
    replicated, batch_sharding = tm.create_shardings()
    ones = jnp.ones((1, 8), dtype=jnp.int32)
    init = tm.to_global(model.init(jax.random.PRNGKey(0), input_ids=ones, attention_mask=ones, deterministic=True)["params"],
                        replicated)
    params, heads = eb.mlm_params(args.checkpoint, init, 0)
    del init
    rows = pq.read_table(args.texts).to_pylist()
    token_lists = tokenizer([r["text"] for r in rows], add_special_tokens=False, split_special_tokens=True)["input_ids"]
    pieces, targets, owner = [], [], []
    kept = [0 < len(t) <= ROW_LENGTH - 2 for t in token_lists]
    for s, token_ids in enumerate(token_lists):
        if not kept[s]:
            continue
        for inputs, position in eb.masked_copies(token_ids, tokenizer.convert_ids_to_tokens(token_ids),
                                                 tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.mask_token_id):
            labels = np.full(inputs.shape, -100, dtype=np.int32)
            labels[position] = len(targets)
            pieces.append((inputs, labels, False))
            targets.append(token_ids[position - 1])
            owner.append(s)
    packed = eb.pack_rows(pieces, ROW_LENGTH, tokenizer.pad_token_id)
    rows_per_batch = args.rows_per_device * jax.device_count()
    n_batches = -(-len(packed["input_ids"]) // rows_per_batch)
    padded = eb.pad_to_rows(packed, n_batches * rows_per_batch, tokenizer.pad_token_id)
    target_ids = np.asarray(targets, dtype=np.int32)
    padded["target_ids"] = np.where(padded["labels"] >= 0, target_ids[np.maximum(padded["labels"], 0)], 0).astype(np.int32)

    def forward(params, batch):
        logits = model.apply({"params": params}, input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                             position_ids=batch["position_ids"], segment_ids=batch["segment_ids"],
                             deterministic=True).logits
        log_probs = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
        return jnp.take_along_axis(log_probs, batch["target_ids"][..., None], axis=-1)[..., 0]

    forward_jit = jax.jit(forward, out_shardings=replicated)
    copy_log_probs = np.full(len(targets), np.nan)
    keys = ("input_ids", "attention_mask", "position_ids", "segment_ids", "target_ids")
    for b in range(n_batches):
        window = slice(b * rows_per_batch, (b + 1) * rows_per_batch)
        log_probs = np.asarray(forward_jit(params, tm.to_global({k: padded[k][window] for k in keys}, batch_sharding)),
                               dtype=np.float64)
        labels = padded["labels"][window]
        scored = labels >= 0
        if not (padded["input_ids"][window][scored] == tokenizer.mask_token_id).all():
            raise RuntimeError(f"batch {b}: a scored position without <mask>")
        copy_log_probs[labels[scored]] = log_probs[scored]
        if b % 200 == 0:
            print(f"PLL batch {b + 1}/{n_batches}, {time.time() - started:.0f} s", flush=True)
    if np.isnan(copy_log_probs).any():
        raise RuntimeError(f"{int(np.isnan(copy_log_probs).sum())} masked copies have no log-probability")
    total = np.zeros(len(rows))
    np.add.at(total, np.asarray(owner, dtype=np.int64), copy_log_probs)
    out = [{"model": r["model"], "set": r["set"], "id": r["id"], "tokens": len(t),
            "pll": float(total[s]) if kept[s] else None} for s, (r, t) in enumerate(zip(rows, token_lists))]
    pq.write_table(pa.Table.from_pylist(out), args.out)
    print(f"PLL texts {len(rows)}, scored {sum(kept)}, left out {len(rows) - sum(kept)}, copies {len(targets)}, "
          f"rows {len(packed['input_ids'])}, batches {n_batches}, heads {heads}, {time.time() - started:.0f} s", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("classifier", "classify", "pll"):
        p = sub.add_parser(name)
        p.add_argument("--dumb-code", required=True, help="A checkout of mbert-jax branch dumb-extra")
        p.add_argument("--out", required=True)
        if name in ("classifier", "classify"):
            p.add_argument("--max-length", type=int, default=1024)
            p.add_argument("--batch-size", type=int, default=8, help="Per device")
            p.add_argument("--pad-to-multiple-of", type=int, default=128)
    c = sub.choices["classifier"]
    c.add_argument("--checkpoint", required=True, help="S @ 653k converted with convert_gcs_checkpoint.py --params-only")
    c.add_argument("--data", required=True)
    c.add_argument("--epochs", type=int, default=3)
    c.add_argument("--learning-rate", type=float, default=5e-5)
    c.add_argument("--seed", type=int, default=42)
    k = sub.choices["classify"]
    k.add_argument("--model", required=True, help="The classifier's directory (params.pkl, config.json)")
    k.add_argument("--texts", required=True)
    p = sub.choices["pll"]
    p.add_argument("--checkpoint", required=True, help="Step directory of the Orbax checkpoint (gs://)")
    p.add_argument("--texts", required=True)
    p.add_argument("--tokenizer", default="yhavinga/dutch-llama-tokenizer")
    p.add_argument("--rows-per-device", type=int, default=8)
    args = parser.parse_args(argv)
    {"classifier": classifier, "classify": classify, "pll": pll}[args.command](args)


if __name__ == "__main__":
    import os
    import traceback

    code = 0
    try:
        main()
    except SystemExit as stop:
        code = stop.code if isinstance(stop.code, int) else (0 if stop.code is None else 1)
        if not isinstance(stop.code, int) and stop.code is not None:
            print(stop.code, file=sys.stderr)
    except BaseException:
        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    # As eval_blimp_nl.py and train_minimal.py: a normal exit can wait in the runtime's shutdown
    os._exit(code)
