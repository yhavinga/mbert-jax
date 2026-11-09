#!/usr/bin/env python3
"""Gemma 4 31B-it in MaxText as a yes/no judge of English-to-Dutch translations, on the 8 chips of a v6e-8 host.

One forward pass per pair: the prompt is the model's chat template with one user turn (the English source, the Dutch
translation, the question whether the Dutch is a correct and complete translation, "answer yes or no"), and the score
is P(yes) / (P(yes) + P(no)) of the first answer token, summed over the spellings of each word (yes, Yes, YES, " yes",
...). No decoding: the model runs in MaxText's train mode (no KV cache) on left-padded batches, and only the hidden
state of the last position goes through the output head.

Usage (run_judge.sh, on the VM):
    python judge.py score --pairs <dir of shard-NNNNN.parquet with id, en, nl> --out gs://.../scores/<set> \\
        --ckpt gs://.../gemma-4-31b-it-maxtext/0/items --tokenizer /dev/shm/judge/tokenizer
    python judge.py probe --pairs probe.parquet --reference ref.json --ckpt ... --tokenizer ...

Every input shard gives one output shard of the same name, written to the bucket when it is complete; a restart skips
the shards that are there, so a preempted spot VM loses at most the shard in progress.
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

# The chat template of google/gemma-4-31B-it (chat_template.jinja at 842da37) for one user message, no system turn,
# add_generation_prompt=True and enable_thinking left at its default (false), which closes an empty thought channel
# so that the next token is the answer. The template trims the user text. hf_reference.py checks these strings
# against transformers' apply_chat_template on the VM; tests/test_en_nl_judge.py renders the jinja file itself.
TEMPLATE_PREFIX = "<bos><|turn>user\n"
TEMPLATE_SUFFIX = "<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
REVISION = "842da3794eaa0b77d5f08bae87a17459d91ff475"

QUESTION = (
    "Here is an English text and a translation of it into Dutch.\n\n"
    "English:\n{en}\n\n"
    "Dutch:\n{nl}\n\n"
    "Is the Dutch a correct and complete translation of the English? It is correct only if it conveys the full "
    "meaning of the English text: nothing left out, nothing added, no sentence that belongs to a different text, and "
    "written in Dutch. Answer with one word: yes or no."
)

# Padded prompt lengths; a batch holds token_budget // length rows (a multiple of the batch axis), so a step costs
# about the same at every length and the number of compiled shapes stays small.
LENGTHS = (128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096)
PAD_ID = 0


def user_text(en, nl):
    return QUESTION.format(en=en.strip(), nl=nl.strip())


def prompt_text(en, nl):
    return TEMPLATE_PREFIX + user_text(en, nl) + TEMPLATE_SUFFIX


def answer_ids(vocab):
    """(yes ids, no ids): every token whose text, without the word-boundary mark and spaces, is yes or no in any case.

    vocab: the token strings, indexed by id."""
    yes, no = [], []
    for i, token in enumerate(vocab):
        word = token.replace("▁", " ").strip().lower()
        if word == "yes":
            yes.append(i)
        elif word == "no":
            no.append(i)
    if not yes or not no:
        raise RuntimeError(f"no yes or no token in the vocabulary: yes {yes}, no {no}")
    return yes, no


def bucket_length(n, lengths=LENGTHS):
    for length in lengths:
        if n <= length:
            return length
    return None


def batch_rows(length, token_budget, multiple):
    return max(multiple, token_budget // length // multiple * multiple)


def plan_batches(prompt_lengths, token_budget, multiple, lengths=LENGTHS):
    """([(padded length, [row indices])], [rows too long]): rows sorted by length, each batch from one length bucket,
    at most batch_rows(length) rows (the last batch of a bucket fewer; pack() fills it with empty rows)."""
    order = sorted(range(len(prompt_lengths)), key=lambda i: prompt_lengths[i])
    batches, too_long = [], []
    for i in order:
        length = bucket_length(prompt_lengths[i], lengths)
        if length is None:
            too_long.append(i)
            continue
        if batches and batches[-1][0] == length and len(batches[-1][1]) < batch_rows(length, token_budget, multiple):
            batches[-1][1].append(i)
        else:
            batches.append((length, [i]))
    return batches, too_long


def pack(prompts, length, rows):
    """Left-padded (tokens, positions, segment ids) of shape [rows, length]: every prompt ends at the last position,
    so one slice gives the next-token logits of all rows. Padding has segment id 0 and attends only to padding; the
    empty rows that fill a short batch are all padding and are dropped by the caller."""
    tokens = np.full((rows, length), PAD_ID, np.int32)
    positions = np.zeros((rows, length), np.int32)
    segments = np.zeros((rows, length), np.int32)
    for r, ids in enumerate(prompts):
        n = len(ids)
        if n > length:
            raise ValueError(f"prompt of {n} tokens in a batch of length {length}")
        tokens[r, length - n:] = ids
        positions[r, length - n:] = np.arange(n)
        segments[r, length - n:] = 1
    return tokens, positions, segments


def score_from_logps(logp_yes, logp_no):
    """P(yes) / (P(yes) + P(no)) from the two log probabilities, stable at both ends."""
    return 1.0 / (1.0 + math.exp(logp_no - logp_yes)) if logp_no - logp_yes < 700 else 0.0


def word_count(text):
    return len(text.split())


class Tokenizer:
    """The model's tokenizer.json through the tokenizers library; the special tokens in the template text (<bos>,
    <|turn>, ...) are added tokens, which the library matches in the text."""

    def __init__(self, directory):
        from tokenizers import Tokenizer as RustTokenizer

        self.inner = RustTokenizer.from_file(str(Path(directory) / "tokenizer.json"))
        self.vocab = [self.inner.id_to_token(i) or "" for i in range(self.inner.get_vocab_size())]

    def encode_batch(self, texts):
        return [e.ids for e in self.inner.encode_batch(texts, add_special_tokens=False)]

    def token(self, i):
        return self.vocab[i]


def keep_last_position_only():
    """MaxText computes the logits of every position; the judge needs the last one, and [batch, length, 262144]
    float32 logits would take more memory than the model. The decoder's output head gets the last position only
    (the prompts are left-padded, so that is every row's last token)."""
    from maxtext.layers import nnx_decoders

    classes = [c for c in vars(nnx_decoders).values() if isinstance(c, type) and "apply_output_head" in vars(c)]
    if len(classes) != 1:
        raise RuntimeError(f"expected one decoder class with apply_output_head, found {classes}")
    cls = classes[0]
    original = cls.apply_output_head

    def apply_output_head(self, shared_embedding, y, deterministic, model_mode, normalize_y=True):
        return original(self, shared_embedding, y[:, -1:, :], deterministic, model_mode, normalize_y)

    cls.apply_output_head = apply_output_head


class Judge:
    """The model on the host's chips and the jitted score of one batch."""

    def __init__(self, ckpt, tokenizer, fsdp, tensor, top_k=5):
        import jax
        import jax.numpy as jnp
        from flax import nnx
        from flax.linen import partitioning as nn_partitioning
        from maxtext.configs import pyconfig
        from maxtext.utils import maxtext_utils, model_creation_utils

        if fsdp * tensor != jax.device_count():
            raise SystemExit(f"fsdp {fsdp} x tensor {tensor} is not the {jax.device_count()} devices of this host")
        keep_last_position_only()
        self.config = pyconfig.initialize(
            [""], model_name="gemma4-31b", load_parameters_path=ckpt, run_name="en_nl_judge", log_config=False,
            base_output_directory="/dev/shm/judge/maxtext_output", enable_checkpointing=False,
            async_checkpointing=False, skip_jax_distributed_system=True, scan_layers=False, use_multimodal=False,
            attention="dot_product", dtype="bfloat16", weight_dtype="bfloat16", ici_fsdp_parallelism=fsdp,
            ici_tensor_parallelism=tensor, per_device_batch_size=1, max_target_length=max(LENGTHS),
            max_prefill_predict_length=max(LENGTHS))
        devices = maxtext_utils.create_device_mesh(self.config, jax.devices())
        self.mesh = jax.sharding.Mesh(devices, self.config.mesh_axes)
        self.rules = self.config.logical_axis_rules
        with jax.set_mesh(self.mesh), nn_partitioning.axis_rules(self.rules):
            self.model = model_creation_utils.from_pretrained(self.config, mesh=self.mesh)
        self.tokenizer = tokenizer
        yes, no = answer_ids(tokenizer.vocab)
        self.yes_ids, self.no_ids = yes, no
        yes_ids, no_ids = jnp.asarray(yes, jnp.int32), jnp.asarray(no, jnp.int32)

        @nnx.jit
        def step(model, tokens, positions, segments):
            out = model(decoder_input_tokens=tokens, decoder_positions=positions, decoder_segment_ids=segments,
                        enable_dropout=False)
            logits = out[0] if isinstance(out, tuple) else out
            logp = jax.nn.log_softmax(logits[:, -1, :].astype(jnp.float32), axis=-1)
            logp_yes = jax.nn.logsumexp(logp[:, yes_ids], axis=-1)
            logp_no = jax.nn.logsumexp(logp[:, no_ids], axis=-1)
            top_logp, top_ids = jax.lax.top_k(logp, top_k)
            return logp_yes, logp_no, top_ids, top_logp

        self._step = step

    def run(self, prompts, length, rows):
        import jax
        from flax.linen import partitioning as nn_partitioning

        tokens, positions, segments = pack(prompts, length, rows)
        with jax.set_mesh(self.mesh), nn_partitioning.axis_rules(self.rules):
            out = self._step(self.model, tokens, positions, segments)
        logp_yes, logp_no, top_ids, top_logp = (np.asarray(jax.device_get(x)) for x in out)
        n = len(prompts)
        return logp_yes[:n], logp_no[:n], top_ids[:n], top_logp[:n]


def score_rows(judge, en, nl, token_budget, multiple, log=print):
    """Scores of all rows of one shard: a dict of columns in the order of the rows; a row whose prompt is longer than
    the longest padded length, or with an empty side, has no score and its reason in `skipped`."""
    n = len(en)
    skipped = [None] * n
    for i in range(n):
        if not en[i].strip() or not nl[i].strip():
            skipped[i] = "empty"
    live = [i for i in range(n) if skipped[i] is None]
    ids = judge.tokenizer.encode_batch([prompt_text(en[i], nl[i]) for i in live])
    prompt_tokens = [0] * n
    for i, row in zip(live, ids):
        prompt_tokens[i] = len(row)
    by_row = dict(zip(live, ids))
    batches, too_long = plan_batches([len(row) for row in ids], token_budget, multiple)
    for k in too_long:
        skipped[live[k]] = "too_long"
    nan = float("nan")
    cols = {"score": [nan] * n, "logp_yes": [nan] * n, "logp_no": [nan] * n, "answer_mass": [nan] * n,
            "top_token": [None] * n, "top_logp": [nan] * n}
    for length, members in batches:
        rows = [live[k] for k in members]
        logp_yes, logp_no, top_ids, top_logp = judge.run([by_row[i] for i in rows], length,
                                                         batch_rows(length, token_budget, multiple))
        for j, i in enumerate(rows):
            y, m = float(logp_yes[j]), float(logp_no[j])
            cols["score"][i] = score_from_logps(y, m)
            cols["logp_yes"][i], cols["logp_no"][i] = y, m
            cols["answer_mass"][i] = math.exp(y) + math.exp(m)
            cols["top_token"][i] = judge.tokenizer.token(int(top_ids[j, 0]))
            cols["top_logp"][i] = float(top_logp[j, 0])
    cols["prompt_tokens"] = prompt_tokens
    cols["skipped"] = skipped
    return cols


def done_shards(out):
    listed = subprocess.run(["gsutil", "ls", f"{out}/"], capture_output=True, text=True)
    if listed.returncode != 0:
        if "matched no objects" in listed.stderr:
            return set()
        raise RuntimeError(f"gsutil ls {out}/ failed: {listed.stderr.strip()}")
    return {Path(line).name for line in listed.stdout.split() if line.endswith(".parquet")}


# The columns of the input shards that the score shards carry along (all but the texts)
PASSTHROUGH = ("id", "kind", "source_id", "date", "year", "intervention_id", "paragraph", "paragraphs", "speaker",
               "speaker_id", "nationality", "mode", "role", "sl")


def sync_compilation_cache():
    """The local compilation cache to the bucket, so that a new spot VM need not compile again."""
    local, remote = os.environ.get("JUDGE_JAX_CACHE"), os.environ.get("JUDGE_JAX_CACHE_GCS")
    if local and remote:
        subprocess.run(["gsutil", "-m", "-q", "rsync", "-r", local, remote], check=True)


def score_command(args):
    """Every --pairs directory into its --out directory, in that order, with one model."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    if len(args.pairs) != len(args.out):
        raise SystemExit("give one --out per --pairs")
    jobs = []
    for pairs, out in zip(args.pairs, args.out):
        shards = sorted(Path(pairs).glob("shard-*.parquet"))
        if not shards:
            raise SystemExit(f"no shard-*.parquet in {pairs}")
        done = done_shards(out)
        pending = [s for s in shards if s.name not in done]
        print(f"{len(shards)} shards in {pairs}, {len(shards) - len(pending)} done before in {out}", flush=True)
        jobs.extend((shard, out) for shard in pending)
    if args.max_shards is not None:
        jobs = jobs[:args.max_shards]
    if not jobs:
        return
    started = time.time()
    judge = Judge(args.ckpt, Tokenizer(args.tokenizer), args.fsdp, args.tensor)
    print(f"model ready in {time.time() - started:.0f} s; yes ids {judge.yes_ids}, no ids {judge.no_ids}", flush=True)
    local = Path("/dev/shm/judge/out_shards")
    local.mkdir(parents=True, exist_ok=True)
    totals = {"pairs": 0, "scored": 0, "seconds": 0.0}
    for shard, out_dir in jobs:
        table = pq.read_table(shard)
        en, nl = table.column("en").to_pylist(), table.column("nl").to_pylist()
        t0 = time.time()
        cols = score_rows(judge, en, nl, args.token_budget, args.fsdp)
        seconds = time.time() - t0
        out = {c: table.column(c).to_pylist() for c in PASSTHROUGH if c in table.column_names}
        out.update({"en_chars": [len(x) for x in en], "nl_chars": [len(x) for x in nl],
                    "en_words": [word_count(x) for x in en], "nl_words": [word_count(x) for x in nl]})
        out.update(cols)
        path = local / shard.name
        pq.write_table(pa.table(out), path, compression="zstd")
        subprocess.run(["gsutil", "-q", "cp", str(path), f"{out_dir}/{shard.name}"], check=True)
        path.unlink()
        sync_compilation_cache()
        scored = sum(s is None for s in cols["skipped"])
        tokens = sum(cols["prompt_tokens"])
        totals["pairs"] += len(en)
        totals["scored"] += scored
        totals["seconds"] += seconds
        print("SHARD " + json.dumps({"out": out_dir, "shard": shard.name, "pairs": len(en), "scored": scored,
                                     "skipped": len(en) - scored, "prompt_tokens": tokens,
                                     "seconds": round(seconds, 1), "pairs_per_s": round(scored / seconds, 1),
                                     "tokens_per_s": round(tokens / seconds)}), flush=True)
    print("DONE " + json.dumps({**totals, "pairs_per_s": round(totals["scored"] / max(totals["seconds"], 1e-9), 1)}),
          flush=True)


def probe_command(args):
    """Scores a few rows and compares them with hf_reference.py's numbers (transformers on the CPU): the prompt ids
    must be equal, and the log probabilities close."""
    import pyarrow.parquet as pq

    table = pq.read_table(args.pairs)
    en, nl = table.column("en").to_pylist(), table.column("nl").to_pylist()
    tokenizer = Tokenizer(args.tokenizer)
    ids = tokenizer.encode_batch([prompt_text(a, b) for a, b in zip(en, nl)])
    reference = json.loads(Path(args.reference).read_text()) if args.reference else None
    if reference is not None:
        for k, row in enumerate(reference["rows"]):
            if row["prompt_ids"] != ids[k]:
                raise SystemExit(f"row {k}: prompt ids differ from the chat template's (transformers)")
        print(f"prompt ids equal to apply_chat_template for {len(reference['rows'])} rows", flush=True)
    judge = Judge(args.ckpt, tokenizer, args.fsdp, args.tensor)
    if reference is not None and (reference["yes_ids"], reference["no_ids"]) != (judge.yes_ids, judge.no_ids):
        raise SystemExit(f"answer ids differ: {reference['yes_ids']} {reference['no_ids']} here {judge.yes_ids} "
                         f"{judge.no_ids}")
    results = []
    for length in sorted({bucket_length(len(row)) for row in ids}):
        members = [k for k, row in enumerate(ids) if bucket_length(len(row)) == length]
        t0 = time.time()
        logp_yes, logp_no, top_ids, top_logp = judge.run([ids[k] for k in members], length,
                                                         batch_rows(length, args.token_budget, args.fsdp))
        print(f"length {length}: {len(members)} rows in {time.time() - t0:.1f} s (incl. compile)", flush=True)
        for j, k in enumerate(members):
            results.append((k, {"logp_yes": float(logp_yes[j]), "logp_no": float(logp_no[j]),
                                "score": score_from_logps(float(logp_yes[j]), float(logp_no[j])),
                                "top": [[tokenizer.token(int(t)), round(float(p), 4)]
                                        for t, p in zip(top_ids[j], top_logp[j])]}))
    results = [r for _, r in sorted(results, key=lambda kr: kr[0])]
    worst = 0.0
    for k, r in enumerate(results):
        line = {"row": k, "tokens": len(ids[k]), **{key: round(v, 4) if isinstance(v, float) else v
                                                    for key, v in r.items()}}
        if reference is not None:
            ref = reference["rows"][k]
            line["ref_score"] = round(ref["score"], 4)
            line["ref_top"] = ref["top"][:3]
            worst = max(worst, abs(ref["score"] - r["score"]))
        print("PROBE " + json.dumps(line, ensure_ascii=False), flush=True)
    if reference is not None:
        print(f"largest score difference to the reference: {worst:.4f}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Gemma 4 31B as a yes/no judge of en-nl translations")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("score", "probe"):
        p = sub.add_parser(name)
        p.add_argument("--pairs", required=True, action="append" if name == "score" else "store",
                       help="score: a directory of shard-NNNNN.parquet, repeatable; probe: one parquet file")
        p.add_argument("--ckpt", required=True, help="The MaxText parameters, <checkpoint>/0/items")
        p.add_argument("--tokenizer", required=True, help="Directory with the model's tokenizer.json")
        p.add_argument("--fsdp", type=int, default=8)
        p.add_argument("--tensor", type=int, default=1)
        p.add_argument("--token-budget", type=int, default=32768, help="Padded tokens per step (default: 32768)")
    sub.choices["score"].add_argument("--out", required=True, action="append",
                                      help="Bucket directory of the score shards, one per --pairs")
    sub.choices["score"].add_argument("--max-shards", type=int, default=None)
    sub.choices["probe"].add_argument("--reference", default=None, help="hf_reference.py's JSON")
    args = parser.parse_args(argv)
    cache = os.environ.get("JUDGE_JAX_CACHE")
    if cache:
        import jax

        jax.config.update("jax_compilation_cache_dir", cache)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 5)
    {"score": score_command, "probe": probe_command}[args.command](args)


if __name__ == "__main__":
    # As in translate.py of FLAN-nl: a thread that is no daemon would keep a failed process alive, holding the TPU
    # and never reaching the EXIT line of the log; the process ends here with the error's traceback and code.
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
    os._exit(code)
