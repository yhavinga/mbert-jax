#!/usr/bin/env python3
"""Fine-tuning for the token and span tasks of experiments/dumb-extra: NER (CoNLL-2002), POS (UD) and QA (SQuAD-NL).

Usage:
    python scripts/dumb/finetune_extra.py --checkpoint CKPT_DIR --task squadnl --data-dir ~/dumbx/data \\
        --output OUT --batch-size 8 --learning-rate 5e-5 --seed 42

CKPT_DIR as for finetune.py (params.pkl and config.json of convert_gcs_checkpoint.py); --data-dir a local copy of
gs://t5-dutch-english-usc2/mbert/dumb-extra/data/. finetune.py itself is not changed: DBRD and SICK-NL keep their
computation (tests/test_dumb_pinned.py). From finetune.py: the checkpoint loading, the encoder and the MLM prediction
head as the start of the task model's encoder and head, data parallelism with pmap over the per-device batch, gradient
accumulation with optax.MultiSteps, AdamW with weight decay 0.01, float32, and the epoch shuffles of NumPy seeded with
--seed. Different, for these tasks:
- the head: a classifier per token (ModernBertForTokenClassification) or a start/end classifier per token
  (ModernBertForQuestionAnswering), new and randomly initialized;
- the learning rate falls linearly to 0 over all updates, after --warmup-steps (default 0), as Hugging Face's Trainer in
  DUMB (warmup ratio 0); --lr-schedule constant is finetune.py's;
- every batch is padded to a multiple of --pad-to-multiple-of (default 128), so each step compiles once per multiple;
- evaluation counts every example: a last, incomplete batch is filled with copies of its first row, whose outputs are
  dropped. The dev split is evaluated after every epoch for the log only; the test numbers are those of the last state.
No parameters are saved.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from functools import partial  # noqa: E402

import extra_metrics  # noqa: E402
import extra_tasks  # noqa: E402
import finetune  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402
from flax import jax_utils  # noqa: E402
from tqdm import tqdm  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from mmbjax import hf_encoders  # noqa: E402
from mmbjax.heads.classification import ModernBertForTokenClassification  # noqa: E402
from mmbjax.heads.qa import ModernBertForQuestionAnswering  # noqa: E402

WEIGHT_DECAY = 0.01  # finetune.py's


def learning_rate_schedule(learning_rate, kind, total_updates, warmup_steps):
    """The rate per update: constant (finetune.py), or linear from 0 to learning_rate over warmup_steps and then
    linear to 0 at total_updates (Hugging Face's get_linear_schedule_with_warmup)."""
    if kind == "constant":
        return learning_rate
    if not 0 <= warmup_steps < total_updates:
        raise ValueError(f"--warmup-steps {warmup_steps} must be below the {total_updates} updates")
    decay = optax.linear_schedule(learning_rate, 0.0, total_updates - warmup_steps)
    if warmup_steps == 0:
        return decay
    return optax.join_schedules([optax.linear_schedule(0.0, learning_rate, warmup_steps), decay], [warmup_steps])


def create_model_and_state(kind, config, pretrained_params, num_labels, schedule, rng, gradient_accumulation_steps=1):
    """As finetune.create_model_and_state, for a token or span head, and failing when the checkpoint lacks the encoder
    or the prediction head instead of leaving it at its random initialization. RoBERTa and NeoBERT
    (mmbjax.hf_encoders) take only their encoder; their task heads are new, as in their Hugging Face classes."""
    config.num_labels = num_labels
    hf = isinstance(config, hf_encoders.HFEncoderConfig)
    if hf:
        model = (hf_encoders.HFEncoderForQuestionAnswering if kind == "qa" else hf_encoders.HFEncoderForTokenClassification)(config=config)
    else:
        model = (ModernBertForQuestionAnswering if kind == "qa" else ModernBertForTokenClassification)(config=config)
    init_rng, dropout_rng = jax.random.split(rng)
    params = model.init(init_rng, jnp.ones((1, 128), dtype=jnp.int32), deterministic=True)["params"]
    missing = [name for name in (("model",) if hf else ("model", "prediction_head")) if name not in pretrained_params]
    if missing:
        raise ValueError(f"the checkpoint has no {missing}, only {sorted(pretrained_params)}")
    new_params = dict(params)
    new_params["model"] = pretrained_params["model"]
    if not hf:
        new_params["head"] = pretrained_params["prediction_head"]
    optimizer = optax.adamw(learning_rate=schedule, weight_decay=WEIGHT_DECAY)
    if gradient_accumulation_steps > 1:
        optimizer = optax.MultiSteps(optimizer, every_k_schedule=gradient_accumulation_steps)
    state = finetune.TrainState.create(apply_fn=model.apply, params=new_params, tx=optimizer, dropout_rng=dropout_rng)
    return model, state


def collate(rows, kind, pad_token_id, max_length, pad_to_multiple_of):
    """Padded arrays of rows; the length is the longest row's rounded up to pad_to_multiple_of, at most max_length.

    Padding is masked; for a token task its labels are -100. The span loss of ModernBertForQuestionAnswering runs over
    every position of the row, padding included, as Hugging Face's does over rows padded to max_length.
    """
    longest = max(len(row["input_ids"]) for row in rows)
    if longest > max_length:
        raise ValueError(f"a row of {longest} tokens, longer than --max-length {max_length}")
    length = min(-(-longest // pad_to_multiple_of) * pad_to_multiple_of, max_length)
    batch = {
        "input_ids": np.full((len(rows), length), pad_token_id, dtype=np.int32),
        "attention_mask": np.zeros((len(rows), length), dtype=np.float32),
    }
    if kind == "token":
        batch["labels"] = np.full((len(rows), length), -100, dtype=np.int32)
    else:
        batch["start_positions"] = np.array([row["start_position"] for row in rows], dtype=np.int32)
        batch["end_positions"] = np.array([row["end_position"] for row in rows], dtype=np.int32)
    for i, row in enumerate(rows):
        n = len(row["input_ids"])
        batch["input_ids"][i, :n] = row["input_ids"]
        batch["attention_mask"][i, :n] = 1
        if kind == "token":
            batch["labels"][i, :n] = row["labels"]
    return batch


def _targets(batch):
    if "labels" in batch:
        return {"labels": batch["labels"]}
    return {"start_positions": batch["start_positions"], "end_positions": batch["end_positions"]}


def _train_step(state, batch):
    """finetune.py's step with the loss of the task head."""
    dropout_rng, new_dropout_rng = jax.random.split(state.dropout_rng)

    def loss_fn(params):
        return state.apply_fn({"params": params}, batch["input_ids"], attention_mask=batch["attention_mask"],
                              deterministic=False, rngs={"dropout": dropout_rng}, **_targets(batch)).loss

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    grads = jax.lax.pmean(grads, axis_name="batch")
    loss = jax.lax.pmean(loss, axis_name="batch")
    state = state.apply_gradients(grads=grads)
    state = state.replace(dropout_rng=new_dropout_rng)
    return state, {"loss": loss}


p_train_step = jax.pmap(_train_step, axis_name="batch")


def _token_logits(state, batch):
    return state.apply_fn({"params": state.params}, batch["input_ids"], attention_mask=batch["attention_mask"],
                          deterministic=True).logits.astype(jnp.float32)


def _span_logits(state, batch):
    outputs = state.apply_fn({"params": state.params}, batch["input_ids"], attention_mask=batch["attention_mask"],
                             deterministic=True)
    return jnp.stack([outputs.start_logits, outputs.end_logits], axis=-1).astype(jnp.float32)


p_token_logits = jax.pmap(_token_logits, axis_name="batch")
p_span_logits = jax.pmap(_span_logits, axis_name="batch")


def predict(state, rows, collate_rows, batch_size, forward):
    """The logits of every row, each cut to its own length: (length, labels) or (length, 2) for start and end."""
    outputs = []
    for i in range(0, len(rows), batch_size):
        chunk = rows[i:i + batch_size]
        filled = chunk + [chunk[0]] * (batch_size - len(chunk))
        batch = collate_rows(filled)
        logits = np.asarray(forward(state, finetune.shard_batch({k: batch[k] for k in ("input_ids", "attention_mask")})))
        logits = logits.reshape(batch_size, *logits.shape[2:])
        outputs.extend(logits[k, :len(row["input_ids"])] for k, row in enumerate(chunk))
    return outputs


def log_softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=axis, keepdims=True))


def token_scores(task, examples, logits):
    """Loss over the labelled positions, word accuracy and, for NER, entity scores; predictions at first sub-tokens,
    the rows of a sentence cut in chunks joined again first."""
    labels = extra_tasks.TASKS[task]["labels"]
    nll, count = 0.0, 0
    true, predicted = {}, {}
    for example, row_logits in zip(examples, logits, strict=True):
        first = np.asarray(example["first"])
        log_probs = log_softmax(row_logits[first].astype(np.float64))
        nll -= log_probs[np.arange(first.size), example["word_labels"]].sum()
        count += first.size
        true.setdefault(example["sentence"], []).extend(labels[i] for i in example["word_labels"])
        predicted.setdefault(example["sentence"], []).extend(labels[i] for i in log_probs.argmax(axis=-1))
    true, predicted = list(true.values()), list(predicted.values())
    accuracy = extra_metrics.word_accuracy(true, predicted)
    scores = {"loss": nll / count, "accuracy": accuracy["accuracy"], "words": accuracy["words"], "sentences": len(true),
              "rows": len(examples)}
    if extra_tasks.TASKS[task]["metric"] == "entity_f1":
        entities = extra_metrics.entity_scores(true, predicted)
        scores.update({"entity_f1": entities["f1"], "entity_precision": entities["precision"], "entity_recall": entities["recall"],
                       "entities": entities["true"], "entities_predicted": entities["predicted"], "per_type": entities["per_type"]})
    scores["metric"] = scores[extra_tasks.TASKS[task]["metric"]]
    return scores


def qa_scores(rows, features, logits, n_best, max_answer_length):
    """Exact match and F1 of the predicted answers; the loss of the features' first-answer positions (real tokens only)."""
    nll = 0.0
    for feature, row_logits in zip(features, logits, strict=True):
        log_probs = log_softmax(row_logits.astype(np.float64), axis=0)
        nll -= 0.5 * (log_probs[feature["start_position"], 0] + log_probs[feature["end_position"], 1])
    predictions, no_span = extra_tasks.qa_predictions(rows, features, [x[:, 0] for x in logits], [x[:, 1] for x in logits],
                                                      n_best, max_answer_length)
    squad = extra_metrics.squad_scores(predictions, {row["id"]: row["answers"]["text"] for row in rows})
    return {"loss": nll / len(features), "accuracy": squad["exact_match"], "exact_match": squad["exact_match"], "f1": squad["f1"],
            "metric": squad["f1"], "questions": squad["questions"], "features": len(features), "no_valid_span": no_span}


def load_task_data(args, tokenizer):
    """Per split role: the rows, and the examples the model sees (sentences, or windows for QA)."""
    task = extra_tasks.get_task(args.task)
    data = {}
    for role in ("train", "validation", "test"):
        limit = args.max_train_examples if role == "train" else args.max_eval_examples
        rows = extra_tasks.read_split(args.data_dir, args.task, role, limit)
        if task["kind"] == "qa":
            examples = extra_tasks.qa_features(tokenizer, rows, args.max_length, args.doc_stride)
        else:
            examples = extra_tasks.token_examples(tokenizer, rows, task["labels"], task["labels_column"], args.max_length)
        data[role] = (rows, examples)
    return data


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Fine-tune on the token and span tasks of experiments/dumb-extra")
    parser.add_argument("--checkpoint", required=True, help="Directory with params.pkl and config.json, as for finetune.py")
    parser.add_argument("--task", required=True, choices=list(extra_tasks.TASKS))
    parser.add_argument("--data-dir", required=True, help="Local copy of gs://t5-dutch-english-usc2/mbert/dumb-extra/data/")
    parser.add_argument("--output", required=True, help="Directory for results.json")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8, help="Per device; 8 x 4 chips = 32, DUMB's update (default: 8)")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--lr-schedule", choices=["linear", "constant"], default="linear")
    parser.add_argument("--warmup-steps", type=int, default=0, help="Updates of linear warmup before the decay (default: 0)")
    parser.add_argument("--max-length", type=int, default=None, help="Default: 384 for squadnl, 512 for the token tasks")
    parser.add_argument("--doc-stride", type=int, default=128, help="squadnl: tokens of overlap between windows")
    parser.add_argument("--n-best", type=int, default=20, help="squadnl: starts and ends considered per window")
    parser.add_argument("--max-answer-length", type=int, default=30, help="squadnl: tokens")
    parser.add_argument("--pad-to-multiple-of", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-examples", type=int, default=None, help="The first this many train rows (smoke tests)")
    parser.add_argument("--max-eval-examples", type=int, default=None, help="The first this many dev and test rows (smoke tests)")
    args = parser.parse_args(argv)
    if args.max_length is None:
        args.max_length = 384 if extra_tasks.TASKS[args.task]["kind"] == "qa" else 512
    return args


def main(argv=None):
    args = parse_args(argv)
    started = time.time()
    # The epoch shuffles below draw from NumPy, and two checkpoints should see the same order
    np.random.seed(args.seed)
    task = extra_tasks.get_task(args.task)
    manifest = json.loads((Path(args.data_dir) / "manifest.json").read_text())["datasets"][task["dir"]]
    print(f"Fine-tuning on task: {args.task} ({manifest['name']}, {manifest['revision']})")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Devices: {finetune.NUM_DEVICES}, Per-device batch size: {args.batch_size}, Total batch size: {args.batch_size * finetune.NUM_DEVICES}")

    pretrained_params, config = finetune.load_pretrained_params(Path(args.checkpoint))
    if isinstance(config, hf_encoders.HFEncoderConfig):
        tokenizer = hf_encoders.load_tokenizer(config, split_into_words=task["kind"] == "token")
    else:
        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name or "yhavinga/dutch-llama-tokenizer")
    data = load_task_data(args, tokenizer)
    print("Rows / examples: " + ", ".join(f"{role} {len(rows)} / {len(examples)}" for role, (rows, examples) in data.items()))

    total_batch_size = args.batch_size * finetune.NUM_DEVICES
    micro_batches = len(data["train"][1]) // total_batch_size
    updates = micro_batches * args.epochs // args.gradient_accumulation_steps
    num_labels = 2 if task["kind"] == "qa" else len(task["labels"])
    schedule = learning_rate_schedule(args.learning_rate, args.lr_schedule, updates, args.warmup_steps)
    _, state = create_model_and_state(task["kind"], config, pretrained_params, num_labels, schedule, jax.random.PRNGKey(args.seed),
                                      args.gradient_accumulation_steps)
    state = jax_utils.replicate(state)
    collate_rows = partial(collate, kind=task["kind"], pad_token_id=tokenizer.pad_token_id, max_length=args.max_length,
                           pad_to_multiple_of=args.pad_to_multiple_of)
    forward = p_span_logits if task["kind"] == "qa" else p_token_logits

    def evaluate(role):
        rows, examples = data[role]
        logits = predict(state, examples, collate_rows, total_batch_size, forward)
        if task["kind"] == "qa":
            return qa_scores(rows, examples, logits, args.n_best, args.max_answer_length)
        return token_scores(args.task, examples, logits)

    validation = []
    train_examples = data["train"][1]
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        train_indices = list(range(len(train_examples)))
        np.random.shuffle(train_indices)
        epoch_loss, num_batches = 0.0, 0
        pbar = tqdm(range(0, len(train_indices), total_batch_size), desc="Training")
        for i in pbar:
            batch_indices = train_indices[i:i + total_batch_size]
            # Skip incomplete batches, as finetune.py does
            if len(batch_indices) < total_batch_size:
                continue
            batch = collate_rows([train_examples[j] for j in batch_indices])
            state, metrics = p_train_step(state, finetune.shard_batch(batch))
            loss = float(metrics["loss"][0])
            epoch_loss += loss
            num_batches += 1
            pbar.set_postfix({"loss": f"{loss:.4f}"})
        print(f"Train Loss: {epoch_loss / max(num_batches, 1):.4f}")
        peaks = [(device.memory_stats() or {}).get("peak_bytes_in_use", -1) / 2**30 for device in jax.local_devices()]
        print(f"Peak memory per device: {' '.join(f'{peak:.2f}' for peak in peaks)} GiB")
        scores = evaluate("validation")
        validation.append({"epoch": epoch + 1, "train_loss": epoch_loss / max(num_batches, 1),
                           **{k: v for k, v in scores.items() if k != "per_type"}})
        print(f"Val Loss: {scores['loss']:.4f}, Val Accuracy: {scores['accuracy']:.4f}, Val {task['metric']}: {scores['metric']:.4f}")

    print("\nEvaluating on test set...")
    test = evaluate("test")
    print(f"Test Loss: {test['loss']:.4f}, Test Accuracy: {test['accuracy']:.4f}, Test {task['metric']}: {test['metric']:.4f}")

    results = {
        # The fields of finetune.py's results.json; accuracy is word accuracy for the token tasks, exact match for squadnl
        "task": args.task,
        "checkpoint": args.checkpoint,
        "best_val_accuracy": max(v["accuracy"] for v in validation),
        "test_accuracy": test["accuracy"],
        "test_loss": test["loss"],
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "total_batch_size": total_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_devices": finetune.NUM_DEVICES,
        "max_length": args.max_length,
        "pad_to_multiple_of": args.pad_to_multiple_of,
        # The task metric and what else fixes the run
        "metric": task["metric"],
        "test_metric": test["metric"],
        "best_val_metric": max(v["metric"] for v in validation),
        "test": test,
        "validation": validation,
        "seed": args.seed,
        "update_size": total_batch_size * args.gradient_accumulation_steps,
        "updates": updates,
        "lr_schedule": args.lr_schedule,
        "warmup_steps": args.warmup_steps,
        "weight_decay": WEIGHT_DECAY,
        "train_examples": len(train_examples),
        "data": {"dir": task["dir"], "revision": manifest["revision"], "max_train_examples": args.max_train_examples,
                 "max_eval_examples": args.max_eval_examples},
        "seconds": time.time() - started,
    }
    if task["kind"] == "qa":
        results.update({"doc_stride": args.doc_stride, "n_best": args.n_best, "max_answer_length": args.max_answer_length})
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")
    print(f"Test {task['metric']}: {test['metric']:.4f}")
    return results


if __name__ == "__main__":
    main()
