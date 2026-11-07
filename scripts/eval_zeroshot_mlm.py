#!/usr/bin/env python3
"""Zero-shot classification with the MLM head: Dutch prompts, one <mask> after <answer>, verbalizers of one token.

Usage (one host as a TPU of its own 4 chips, see experiments/flan-decay/eval_zeroshot.sh):
    python scripts/eval_zeroshot_mlm.py --checkpoint gs://bucket/run/ckpt/20000 --tokenizer yhavinga/dutch-llama-tokenizer \\
        --hidden-size 1024 --num-layers 22 --num-heads 16 --intermediate-size 1536 --output gs://bucket/zeroshot/F.json

Each example becomes <s> {prompt} <answer> <mask> </s>, the format of the instruction examples of --masking decay with
the answer masked. The prediction is the verbalizer with the highest logit at the <mask>, among the task's verbalizers
only. Tasks: DBRD test (positief/negatief), SICK-NL test (A/B/C with the options in the prompt) and Belebele nld_Latn
(A-D), each with the 3 fixed templates of TEMPLATES; reported per template and as the mean over templates.

S and W were trained without <answer>: their row for it (id 32001, inside the vocabulary of 32,128) is the untrained
one of the start. That is the point of the comparison, so the model is built as the trainer of S built it, from the
tokenizer with <mask> only, and <answer> must have a row in that vocabulary.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import train_minimal as tm  # noqa: E402  # Its import initializes the runtime, as the trainer's does
from datasets import load_dataset  # noqa: E402
from etils import epath  # noqa: E402
from eval_windows import build_model, pack_rows, pad_to_rows  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from mmbjax.training import add_answer_token  # noqa: E402

PROMPT_FORMAT = "<s> {prompt} <answer> <mask> </s>"

# The datasets of scripts/dumb/data.py for DBRD and SICK-NL, and Belebele's Dutch config; test splits only
DATASETS = {
    "dbrd": ("dbrd", None, "test"),
    "sicknl": ("maximedb/sick_nl", None, "test"),
    "belebele": ("facebook/belebele", "nld_Latn", "test"),
}
# Words whose tokenization alone is one token: ▁negatief, ▁positief, ▁A ... with yhavinga/dutch-llama-tokenizer, as an
# answer of the decay mix is tokenized on its own
VERBALIZERS = {
    "dbrd": ["negatief", "positief"],
    "sicknl": ["A", "B", "C"],
    "belebele": ["A", "B", "C", "D"],
}
# The field cut when a prompt is longer than the window; the end of every template stays
LONG_FIELD = {"dbrd": "text", "sicknl": None, "belebele": "passage"}
# Fixed before any run (experiments/flan-decay/protocol.md). Options are listed in the order of the verbalizers.
TEMPLATES = {
    "dbrd": [
        "{text}\n\nIs deze boekrecensie positief of negatief?",
        "Recensie: {text}\nIs het oordeel van de recensent over het boek positief of negatief?\nAntwoord:",
        "Lees de volgende recensie van een boek en zeg of die positief of negatief is.\n\n{text}",
    ],
    "sicknl": [
        "Zin 1: {sentence_a}\nZin 2: {sentence_b}\nWat is de relatie tussen de twee zinnen?\n"
        "A: zin 2 volgt uit zin 1\nB: neutraal\nC: zin 2 spreekt zin 1 tegen",
        "Premisse: {sentence_a}\nHypothese: {sentence_b}\nVolgt de hypothese uit de premisse?\n"
        "A: ja\nB: misschien\nC: nee\nAntwoord:",
        "{sentence_a}\nVraag: is het volgende dan waar, onwaar of geen van beide? {sentence_b}\n"
        "A: waar\nB: geen van beide\nC: onwaar",
    ],
    "belebele": [
        "{passage}\n\nVraag: {question}\nA: {a}\nB: {b}\nC: {c}\nD: {d}\nAntwoord:",
        "Lees de tekst en beantwoord de vraag.\n\nTekst: {passage}\nVraag: {question}\n\nOpties:\nA. {a}\nB. {b}\nC. {c}\nD. {d}",
        "{passage}\n{question}\nKies het juiste antwoord: A) {a} B) {b} C) {c} D) {d}",
    ],
}
SICK_LABELS = {"ENTAILMENT": 0, "NEUTRAL": 1, "CONTRADICTION": 2}


def task_examples(task, max_examples=None):
    """[{fields of the templates..., "label": index of the verbalizer}] of the task's test split."""
    name, config, split = DATASETS[task]
    rows = load_dataset(name, config, split=split)
    if max_examples is not None:
        rows = rows.select(range(min(max_examples, len(rows))))
    if task == "dbrd":
        # ClassLabel neg = 0, pos = 1: the order of VERBALIZERS["dbrd"]
        return [{"text": row["text"], "label": int(row["label"])} for row in rows]
    if task == "sicknl":
        # The label column of maximedb/sick_nl is 0 contradiction, 1 neutral, 2 entailment; the prompts list A = entailment
        return [{"sentence_a": row["sentence_A"], "sentence_b": row["sentence_B"], "label": SICK_LABELS[row["entailment_label"]]}
                for row in rows]
    return [{"passage": row["flores_passage"], "question": row["question"], "a": row["mc_answer1"], "b": row["mc_answer2"],
             "c": row["mc_answer3"], "d": row["mc_answer4"], "label": int(row["correct_answer_num"]) - 1} for row in rows]


def verbalizer_ids(tokenizer, task):
    ids = []
    for word in VERBALIZERS[task]:
        tokens = tokenizer(word, add_special_tokens=False, split_special_tokens=True)["input_ids"]
        if len(tokens) != 1 or tokens[0] == tokenizer.unk_token_id:
            raise ValueError(f"verbalizer {word!r} of {task} is not one token of this tokenizer: {tokens}")
        ids.append(tokens[0])
    return ids


def prompt_ids(tokenizer, template, fields, long_field, budget):
    """Token ids of the filled template within budget, and whether the long field was cut (at a word, from the end)."""
    def encode(values):
        return tokenizer(template.format(**values), add_special_tokens=False, split_special_tokens=True)["input_ids"]

    ids = encode(fields)
    if len(ids) <= budget:
        return ids, False
    if long_field is None:
        raise ValueError(f"a prompt of {len(ids)} tokens without a field to cut: {template!r}")
    words = fields[long_field].split()
    low, high = 0, len(words)
    # The most words that fit
    while low < high:
        middle = (low + high + 1) // 2
        if len(encode({**fields, long_field: " ".join(words[:middle])})) <= budget:
            low = middle
        else:
            high = middle - 1
    ids = encode({**fields, long_field: " ".join(words[:low])})
    if len(ids) > budget:
        raise ValueError(f"the template alone is longer than {budget} tokens: {template!r}")
    return ids, True


def build_sequences(tokenizer, answer_token_id, tasks, max_length):
    """Per example and template: (task, template index, example index, label, ids, mask position, cut)."""
    sequences = []
    for task, examples in tasks.items():
        for t, template in enumerate(TEMPLATES[task]):
            for e, example in enumerate(examples):
                ids, cut = prompt_ids(tokenizer, template, example, LONG_FIELD[task], max_length - 4)
                sequence = [tokenizer.bos_token_id, *ids, answer_token_id, tokenizer.mask_token_id, tokenizer.eos_token_id]
                sequences.append((task, t, e, example["label"], np.asarray(sequence, dtype=np.int32), len(ids) + 2, cut))
    return sequences


def score(sequences, predictions, verbalizers, top_ids):
    """Accuracy per task and template from the logits of the verbalizers at each <mask>."""
    result = {}
    for task, ids in verbalizers.items():
        rows = [(i, s) for i, s in enumerate(sequences) if s[0] == task]
        labels = Counter(s[3] for i, s in rows if s[1] == 0)
        n = sum(labels.values())
        templates = []
        for t in range(len(TEMPLATES[task])):
            mine = [(i, s) for i, s in rows if s[1] == t]
            predicted = [int(np.argmax(predictions[task][i])) for i, _ in mine]
            templates.append({
                "accuracy": float(np.mean([p == s[3] for p, (_, s) in zip(predicted, mine)])),
                "predicted": {VERBALIZERS[task][k]: v for k, v in sorted(Counter(predicted).items())},
                # How often the head's own first choice over the whole vocabulary is one of the verbalizers
                "top1_is_verbalizer": float(np.mean([top_ids[i] in ids for i, _ in mine])),
                "cut": sum(s[6] for _, s in mine),
                "examples": len(mine),
            })
        result[task] = {
            "mean_accuracy": float(np.mean([t["accuracy"] for t in templates])),
            "templates": templates,
            "examples": n,
            "labels": {VERBALIZERS[task][k]: v for k, v in sorted(labels.items())},
            "chance": 1 / len(ids),
            "majority": max(labels.values()) / n,
            "verbalizers": dict(zip(VERBALIZERS[task], ids)),
        }
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Zero-shot accuracy of a checkpoint's MLM head on DBRD, SICK-NL and Belebele")
    parser.add_argument("--checkpoint", required=True, help="Step directory of a train_minimal.py checkpoint")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--tasks", nargs="+", choices=list(DATASETS), default=list(DATASETS))
    parser.add_argument("--max-examples", type=int, default=None, help="The first this many test examples per task (smoke tests)")
    parser.add_argument("--max-length", type=int, default=1024, help="Rows of the packed batches; prompts are cut to fit")
    parser.add_argument("--rows-per-device", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--num-layers", type=int, default=22)
    parser.add_argument("--num-heads", type=int, default=12)
    parser.add_argument("--intermediate-size", type=int, default=3072)
    parser.add_argument("--global-attn-every-n-layers", type=int, default=3)
    parser.add_argument("--compute-dtype", choices=["float32", "bfloat16"], default=None,
                        help="Default: bfloat16 on TPU, float32 elsewhere, as in the trainer")
    parser.add_argument("--output", default=None, help="JSON file, local or gs:// (default: print only)")
    args = parser.parse_args(argv)
    if args.compute_dtype is None:
        args.compute_dtype = "bfloat16" if jax.default_backend() == "tpu" else "float32"
    return args


def main(argv=None):
    args = parse_args(argv)
    if jax.process_count() != 1 or jax.device_count() != jax.local_device_count():
        raise SystemExit(f"{jax.process_count()} processes, {jax.device_count()} devices: run this on one host as a TPU of its own "
                         "(experiments/flan-decay/eval_zeroshot.sh)")
    started = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, split_special_tokens=True)
    if tokenizer.mask_token is None:
        tokenizer.add_special_tokens({"mask_token": "<mask>"})
    # The vocabulary of S: built before <answer> is added
    config, model = build_model(args, tokenizer)
    answer_token_id = add_answer_token(tokenizer)
    if answer_token_id >= config.vocab_size:
        raise ValueError(f"<answer> ({answer_token_id}) has no row in a vocabulary of {config.vocab_size}")
    verbalizers = {task: verbalizer_ids(tokenizer, task) for task in args.tasks}
    union = sorted({i for ids in verbalizers.values() for i in ids})

    replicated, batch_sharding = tm.create_shardings()
    ones = jnp.ones((1, 8), dtype=jnp.int32)
    # Eagerly, as the trainer does: the first call fills the module-level RoPE cache (see eval_windows.py)
    init_params = tm.to_global(model.init(jax.random.PRNGKey(0), input_ids=ones, attention_mask=ones, deterministic=True)["params"], replicated)
    params = tm.load_params_of_run(args.checkpoint, init_params)
    del init_params

    tasks = {task: task_examples(task, args.max_examples) for task in args.tasks}
    sequences = build_sequences(tokenizer, answer_token_id, tasks, args.max_length)
    # labels carry the index of the sequence at its <mask>, so each prediction finds its example after packing
    pieces = []
    for index, (_, _, _, _, ids, mask_at, _) in enumerate(sequences):
        labels = np.full(ids.shape, -100, dtype=np.int32)
        labels[mask_at] = index
        pieces.append((ids, labels, False))
    rows = pack_rows(pieces, args.max_length, tokenizer.pad_token_id)
    rows_per_batch = args.rows_per_device * jax.device_count()
    n_batches = -(-len(rows["input_ids"]) // rows_per_batch)
    padded = pad_to_rows(rows, n_batches * rows_per_batch, tokenizer.pad_token_id)
    union_ids = jnp.asarray(union)

    def forward(params, batch):
        logits = model.apply({"params": params}, input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                             position_ids=batch["position_ids"], segment_ids=batch["segment_ids"], deterministic=True).logits
        logits = logits.astype(jnp.float32)
        return logits[..., union_ids], jnp.argmax(logits, axis=-1)

    forward_jit = jax.jit(forward, out_shardings=replicated)
    union_logits = [None] * len(sequences)
    top_ids = [None] * len(sequences)
    for b in range(n_batches):
        local = {key: padded[key][b * rows_per_batch:(b + 1) * rows_per_batch] for key in ("input_ids", "attention_mask", "position_ids", "segment_ids")}
        logits, top = forward_jit(params, tm.to_global(local, batch_sharding))
        logits, top = np.asarray(logits), np.asarray(top)
        labels = padded["labels"][b * rows_per_batch:(b + 1) * rows_per_batch]
        for row, column in zip(*np.nonzero(labels != -100)):
            index = labels[row, column]
            if tokenizer.mask_token_id != padded["input_ids"][b * rows_per_batch + row, column]:
                raise RuntimeError(f"sequence {index}: no <mask> where its label is")
            union_logits[index] = logits[row, column]
            top_ids[index] = int(top[row, column])
    missing = [i for i, value in enumerate(union_logits) if value is None]
    if missing:
        raise RuntimeError(f"{len(missing)} sequences have no prediction, e.g. {missing[:5]}")
    position = {token: k for k, token in enumerate(union)}
    predictions = {task: [None] * len(sequences) for task in args.tasks}
    for i, sequence in enumerate(sequences):
        predictions[sequence[0]][i] = union_logits[i][[position[token] for token in verbalizers[sequence[0]]]]

    result = {
        "checkpoint": args.checkpoint, "tokenizer": args.tokenizer, "format": PROMPT_FORMAT, "answer_token_id": answer_token_id,
        "mask_token_id": tokenizer.mask_token_id, "vocab_size": config.vocab_size, "compute_dtype": args.compute_dtype,
        "max_length": args.max_length, "max_examples": args.max_examples, "datasets": {task: DATASETS[task] for task in args.tasks},
        "templates": {task: TEMPLATES[task] for task in args.tasks}, "batches": n_batches, "rows": len(rows["input_ids"]),
        "tasks": score(sequences, predictions, verbalizers, top_ids),
    }
    result["seconds"] = time.time() - started
    for task, values in result["tasks"].items():
        print(f"ZEROSHOT {task} mean={values['mean_accuracy']:.4f} templates={[round(t['accuracy'], 4) for t in values['templates']]} "
              f"chance={values['chance']:.3f} majority={values['majority']:.3f} n={values['examples']}", flush=True)
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
        os._exit(1)
