#!/usr/bin/env python3
"""The decay mix of one host for arm F of the FLAN-decay proof: this host's CulturaX share and a disjoint part of FLAN.

Usage (on each worker of the v4-32, on the CPU; it does not touch the TPU, so it can run while a trainer runs):
    python scripts/build_decay_mix.py --rank 0 --web-files ~/culturax_nl_files.txt --out ~/flan_decay \\
        --list ~/flan_decay_files.txt --niv2-languages experiments/flan-decay/niv2_languages.json

Web: the documents of --web-files in the order the trainer reads them (load_dataset parquet + shuffle with seed 42 and a
buffer of 10,000, the route of --data-files-list), from the start, until their tokens are what --steps need on this host
at the web share, plus --margin. These are the documents arm W reads first, in W's order.

FLAN: Open-Orca/FLAN at a fixed revision, only the zero-shot folders. Host r takes shards r, r + world, ... of a folder
(every shard is a random sample of its folder, measured); a folder of fewer shards than hosts is split by row index.
Filters, in this order: tasks that leak DUMB or the zero-shot eval (sick, dbrd, belebele, flores), tasks with
GPT-generated parts (HellaSwag, MOCHA), tasks with text in other languages than English and Dutch (WMT, para_crawl,
NIv2 by the languages in --niv2-languages), rows with U+2047, rows with an empty instruction or answer, and examples of
more than --max-length tokens. Then at most ceil(--cap / world) rows per (folder, task) on this host, in shard order.

The mix: after each web document, FLAN examples from the shuffled pool until FLAN is --flan-share of the tokens so
far, counted as the trainer counts them (TokenizeTransform with <answer>). The rows are written in an order that the
trainer's own shuffle (seed 42, buffer 10,000) turns into exactly this sequence, and that is checked by reading the file
back through that route. One parquet file per host, so the trainer has no file order to shuffle.

Output: <out>/mix-<build>.parquet with columns text | inputs, targets, source, position (the place in the sequence),
the build id in its metadata (train_minimal.py puts it in the data state's identity), <out>/manifest-<build>.json, and
--list with the parquet file for --data-files-list.

Reads Hugging Face with the token in ~/.cache/huggingface/token (huggingface_hub finds it); the token is never printed.
"""

import os

# The TPU may be in use by a trainer: this script must never take it
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import argparse  # noqa: E402
import concurrent.futures  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np  # noqa: E402

FLAN_REPO = "Open-Orca/FLAN"
# The revision whose shards were measured for the design (2026-11-03) and for this script
FLAN_REVISION = "6845b1b3b53c6d5c5b1e49767ed759df3fc246cc"
# All zero-shot folders of the revision; niv2, cot and dialog have no zsnoopt variant there
FLAN_FOLDERS = ("cot_zsopt_data", "dialog_zsopt_data", "flan_zsopt_data", "flan_zsnoopt_data", "niv2_zsopt_data",
                "t0_zsopt_data", "t0_zsnoopt_data")
FLAN_COLUMNS = ["inputs", "targets", "_task_name", "_template_type"]
# What the tokenizer of T5 put for characters it lacked; FLAN rows with it lost text
UNKNOWN_CHARACTER = "⁇"
LANGUAGES_KEPT = {"English", "Dutch"}
# The trainer's data route: seed=42 in main() of train_minimal.py, --shuffle-buffer-size default 10000
TRAIN_SEED = 42
TRAIN_SHUFFLE_BUFFER = 10000
METADATA_KEY = b"flan_decay_build"
WEB_SOURCE = "culturax"
TOKENIZE_BATCH = 2000


def task_exclusion(folder, task, niv2_languages):
    """Why a task of FLAN is left out, or None."""
    name = task.lower()
    if any(word in name for word in ("sick", "dbrd")):
        return "dumb_leak"  # task1612-task1615 are SICK: SICK-NL is its translation
    if any(word in name for word in ("belebele", "flores")):
        return "eval_leak"  # Belebele's passages are FLORES-200
    if "hellaswag" in name or "mocha" in name:
        return "gpt_generated"  # HellaSwag's wrong endings (Zellers et al. 2019, App. B), MOCHA's candidates (DPI)
    if folder.startswith("flan_") and (name.startswith("wmt") or name.startswith("para_crawl")):
        return "language"
    if folder.startswith("niv2_"):
        if task not in niv2_languages:
            raise KeyError(f"{folder}/{task} is not in --niv2-languages: that file is not of this revision of {FLAN_REPO}")
        languages = niv2_languages[task]
        if not set(languages["input"]) | set(languages["output"]) | set(languages["instruction"]) <= LANGUAGES_KEPT:
            return "language"
    return None


def host_units(n_shards, rank, world):
    """(shard index, row residue or None) of this host: whole shards, or every world-th row of all of them."""
    if n_shards >= world:
        return [(index, None) for index in range(rank, n_shards, world)]
    return [(index, rank) for index in range(n_shards)]


def web_target_tokens(steps, rows_per_host, max_length, flan_share, margin):
    """Web tokens this host needs: the slots of --steps at the web share, plus the margin (padding makes it more)."""
    return int(steps * rows_per_host * max_length * (1 - flan_share) * (1 + margin))


def interleave(web_tokens, flan_tokens, flan_share):
    """The sequence of ("web", i) and ("flan", j): after each web document, FLAN until it is flan_share of the tokens.

    Returns (sequence, FLAN examples used). Raises when the FLAN pool runs out before the web documents.
    """
    sequence, used = [], 0
    web_total = flan_total = 0
    for i, tokens in enumerate(web_tokens):
        sequence.append(("web", i))
        web_total += tokens
        while flan_total < flan_share * (web_total + flan_total):
            if used == len(flan_tokens):
                raise RuntimeError(f"the FLAN pool of {len(flan_tokens)} examples ran out after web document {i + 1} of "
                                   f"{len(web_tokens)} ({web_total} web tokens): raise --cap")
            sequence.append(("flan", used))
            flan_total += flan_tokens[used]
            used += 1
    return sequence, used


def trainer_shuffle_order(n):
    """order[t] = the row of a one-file parquet of n rows that the trainer's stream yields t-th.

    The buffer shuffle of datasets draws its indices from the seed alone, whatever the rows hold; the file is read back
    through the real route afterwards (verify_order), so a datasets version where this is not so fails there.
    """
    import datasets

    stream = datasets.IterableDataset.from_generator(lambda: ({"row": row} for row in range(n)))
    order = np.fromiter((example["row"] for example in stream.shuffle(seed=TRAIN_SEED, buffer_size=TRAIN_SHUFFLE_BUFFER)),
                        dtype=np.int64, count=n)
    if not np.array_equal(np.sort(order), np.arange(n)):
        raise RuntimeError("the shuffle of datasets did not yield every row once")
    return order


def trainer_stream(data_files, tokenizer):
    """The stream of rows the trainer reads from these files in its first pass, from the trainer's own code."""
    from mmbjax.training import EpochTrackingDataLoader

    loader = EpochTrackingDataLoader(dataset_name="uonlp/CulturaX", dataset_config="nl", tokenizer=tokenizer,
                                     seed=TRAIN_SEED, shuffle_buffer_size=TRAIN_SHUFFLE_BUFFER, data_files=list(data_files))
    return loader._load_stream_dataset()


def token_counts(tokenizer, texts, max_length):
    """Tokens of each text as pieces of the trainer, <s> and </s> included (TokenizeTransform's split)."""
    from mmbjax.training.packing import split_into_pieces

    ids = tokenizer(texts, truncation=False, add_special_tokens=False, split_special_tokens=True)["input_ids"]
    return [len(tokens) + len(split_into_pieces(tokens, max_length, tokenizer.bos_token_id, tokenizer.eos_token_id)) + 1
            for tokens in ids]


def read_web(web_files, tokenizer, target, max_length):
    """This host's first web documents in the trainer's order until their tokens reach target."""
    texts, tokens, total = [], [], 0
    stream = iter(trainer_stream(web_files, tokenizer))
    while total < target:
        batch = []
        for row in stream:
            batch.append(row["text"])
            if len(batch) == TOKENIZE_BATCH:
                break
        if not batch:
            raise RuntimeError(f"{web_files} hold {total} web tokens, fewer than the {target} needed")
        for text, count in zip(batch, token_counts(tokenizer, batch, max_length)):
            if total >= target:
                break
            texts.append(text)
            tokens.append(count)
            total += count
        if len(texts) % 100_000 < TOKENIZE_BATCH:
            print(f"web: {len(texts):,} documents, {total:,} of {target:,} tokens", flush=True)
    return texts, tokens


class HubShards:
    """The shards of FLAN_REPO at FLAN_REVISION, downloaded a few ahead into download_dir and deleted once read."""

    def __init__(self, download_dir, ahead=4):
        from huggingface_hub import HfApi

        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.ahead = ahead
        self.files = HfApi().list_repo_files(FLAN_REPO, repo_type="dataset", revision=FLAN_REVISION)

    def shards(self, folder):
        names = [f for f in self.files if f.startswith(folder + "/") and f.endswith(".parquet")]
        return sorted(names, key=lambda name: int(re.search(r"part\.(\d+)\.parquet$", name).group(1)))

    def _download(self, name):
        from huggingface_hub import hf_hub_download

        return hf_hub_download(FLAN_REPO, name, repo_type="dataset", revision=FLAN_REVISION, local_dir=self.download_dir)

    def tables(self, names):
        """(name, table) in the order of names; the caller may stop early."""
        import pyarrow.parquet as pq

        executor = concurrent.futures.ThreadPoolExecutor(self.ahead)
        futures = [executor.submit(self._download, name) for name in names]
        try:
            for name, future in zip(names, futures):
                path = future.result()
                table = pq.read_table(path, columns=FLAN_COLUMNS)
                os.remove(path)
                yield name, table
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
            for future in futures:
                if future.done() and not future.cancelled() and future.exception() is None and os.path.exists(future.result()):
                    os.remove(future.result())


def select_flan(args, tokenizer, source, niv2_languages):
    """This host's FLAN pool: [(inputs, targets, source, tokens)] in shard order, and what was left out why."""
    per_host_cap = math.ceil(args.cap / args.world)
    pool, report = [], {"per_host_cap": per_host_cap, "folders": {}}
    for folder in FLAN_FOLDERS:
        names = source.shards(folder)
        if not names:
            raise RuntimeError(f"{FLAN_REPO}@{FLAN_REVISION} has no shards in {folder}")
        units = host_units(len(names), args.rank, args.world)
        counts, dropped, excluded = Counter(), Counter(), {}
        shards_read = 0
        for (index, residue), (name, table) in zip(units, source.tables([names[index] for index, _ in units])):
            shards_read += 1
            # The folder is the template type: the cap is per (folder, task)
            template_types = set(table.column("_template_type").unique().to_pylist())
            if template_types != {"zs_opt" if "_zsopt_" in folder else "zs_noopt"}:
                raise RuntimeError(f"{name} has template types {template_types}, not those of {folder}")
            tasks = table.column("_task_name").to_pylist()
            rows = range(len(tasks)) if residue is None else range(residue, len(tasks), args.world)
            taken_here = Counter()
            candidates = []
            for row in rows:
                task = tasks[row]
                if task not in excluded:
                    excluded[task] = task_exclusion(folder, task, niv2_languages)
                if excluded[task] is not None:
                    dropped[excluded[task]] += 1
                    continue
                # A few more than the cap leaves room, as some fail the filters below
                if counts[task] + taken_here[task] >= per_host_cap * 1.2 + 5:
                    continue
                taken_here[task] += 1
                candidates.append(row)
            # Only these become Python strings: a shard holds ~300,000 rows, of which a few thousand are wanted
            inputs = table.column("inputs").take(candidates).to_pylist()
            targets = table.column("targets").take(candidates).to_pylist()
            lengths_in = [len(ids) for ids in tokenizer(inputs, add_special_tokens=False, split_special_tokens=True)["input_ids"]]
            lengths_out = [len(ids) for ids in tokenizer(targets, add_special_tokens=False, split_special_tokens=True)["input_ids"]]
            for row, text_in, text_out, n_in, n_out in zip(candidates, inputs, targets, lengths_in, lengths_out):
                task = tasks[row]
                if counts[task] >= per_host_cap:
                    continue
                if UNKNOWN_CHARACTER in text_in or UNKNOWN_CHARACTER in text_out:
                    dropped["unknown_character"] += 1
                elif not text_in.strip() or not text_out.strip():
                    dropped["empty"] += 1
                elif n_in + n_out + 3 > args.max_length:
                    dropped["too_long"] += 1
                else:
                    counts[task] += 1
                    pool.append((text_in, text_out, f"flan/{folder}/{task}", n_in + n_out + 3))
            print(f"flan: {name}: {sum(counts.values()):,} rows of {len(counts)} tasks so far in {folder}, pool {len(pool):,}", flush=True)
            if counts and all(counts[task] >= per_host_cap for task, why in excluded.items() if why is None):
                break
        report["folders"][folder] = {
            "shards": len(names), "units_of_this_host": len(units), "shards_read": shards_read,
            "rows": sum(counts.values()), "tasks": len(counts), "tasks_at_cap": sum(n >= per_host_cap for n in counts.values()),
            "dropped": dict(dropped), "excluded_tasks": {task: why for task, why in sorted(excluded.items()) if why is not None},
            "rows_per_task": dict(sorted(counts.items())),
        }
    return pool, report


def build_id(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]


def write_mix(path, sequence, web_texts, pool, order, build):
    """Row order[t] of the file holds sequence[t], so that the trainer's shuffle yields the sequence."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    schema = pa.schema([("text", pa.string()), ("inputs", pa.string()), ("targets", pa.string()), ("source", pa.string()),
                        ("position", pa.int64())], metadata={METADATA_KEY: build.encode()})
    with pq.ParquetWriter(path, schema, compression="zstd") as writer:
        for start in range(0, len(order), 10_000):
            positions = inverse[start:start + 10_000]
            rows = [sequence[t] for t in positions]
            writer.write_table(pa.table({
                "text": [web_texts[i] if kind == "web" else None for kind, i in rows],
                "inputs": [pool[i][0] if kind == "flan" else None for kind, i in rows],
                "targets": [pool[i][1] if kind == "flan" else None for kind, i in rows],
                "source": [WEB_SOURCE if kind == "web" else pool[i][2] for kind, i in rows],
                "position": positions,
            }, schema=schema))


def verify_order(path, tokenizer, n):
    """The trainer's stream of the file must yield position 0, 1, ..., n - 1."""
    expected = 0
    for row in trainer_stream([str(path)], tokenizer):
        if row["position"] != expected:
            raise RuntimeError(f"{path}: the trainer's stream yields position {row['position']} as row {expected}: "
                               "the shuffle of this datasets version is not the one the order was made for")
        expected += 1
    if expected != n:
        raise RuntimeError(f"{path}: the trainer's stream has {expected} rows, the mix {n}")


def check_counts(tokenize, sequence, web_texts, web_tokens, pool, rows=2000):
    """The token counts of the interleave are those of the trainer's TokenizeTransform, on the first rows."""
    for kind, i in sequence[:rows]:
        element = {"text": web_texts[i], "inputs": None, "targets": None} if kind == "web" else \
            {"text": None, "inputs": pool[i][0], "targets": pool[i][1]}
        counted = web_tokens[i] if kind == "web" else pool[i][3]
        actual = sum(len(piece["input_ids"]) for piece in tokenize.map(element))
        if actual != counted:
            raise RuntimeError(f"{kind} row {i}: counted {counted} tokens, the trainer makes {actual}")


def build(args, tokenizer, source):
    from mmbjax.training import TokenizeTransform

    started = time.time()
    answer_token_id = tokenizer.convert_tokens_to_ids("<answer>")
    niv2_languages = json.loads(Path(args.niv2_languages).read_text())["tasks"]
    web_files = Path(args.web_files).read_text().split()
    config = {
        "rank": args.rank, "world": args.world, "web_files": {path: os.path.getsize(path) for path in web_files},
        "flan": f"{FLAN_REPO}@{FLAN_REVISION}",
        "folders": FLAN_FOLDERS, "cap": args.cap, "flan_share": args.flan_share, "margin": args.margin, "steps": args.steps,
        "rows_per_host": args.rows_per_host, "max_length": args.max_length, "seed": args.seed, "tokenizer": args.tokenizer,
        "niv2_languages": hashlib.sha256(Path(args.niv2_languages).read_bytes()).hexdigest(),
        "train_seed": TRAIN_SEED, "train_shuffle_buffer": TRAIN_SHUFFLE_BUFFER,
    }
    build = build_id(config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"mix-{build}.parquet"
    print(f"build {build} of rank {args.rank}/{args.world}: {path}", flush=True)

    pool, flan_report = select_flan(args, tokenizer, source, niv2_languages)
    # The pool is in shard order, which groups rows by folder: shuffled, every stretch of the mix has every folder
    shuffled = np.random.default_rng([args.seed, args.rank]).permutation(len(pool))
    pool = [pool[i] for i in shuffled]
    print(f"flan pool: {len(pool):,} examples, {sum(p[3] for p in pool):,} tokens ({time.time() - started:.0f} s)", flush=True)

    target = web_target_tokens(args.steps, args.rows_per_host, args.max_length, args.flan_share, args.margin)
    web_texts, web_tokens = read_web(web_files, tokenizer, target, args.max_length)
    sequence, used = interleave(web_tokens, [p[3] for p in pool], args.flan_share)
    check_counts(TokenizeTransform(tokenizer, args.max_length, answer_token_id=answer_token_id), sequence, web_texts, web_tokens, pool)
    flan_tokens = sum(pool[i][3] for i in range(used))
    print(f"mix: {len(web_texts):,} web documents ({sum(web_tokens):,} tokens) and {used:,} FLAN examples "
          f"({flan_tokens:,} tokens) in {len(sequence):,} rows ({time.time() - started:.0f} s)", flush=True)

    order = trainer_shuffle_order(len(sequence))
    partial = path.with_suffix(".partial")
    write_mix(partial, sequence, web_texts, pool, order, build)
    verify_order(partial, tokenizer, len(sequence))
    partial.rename(path)

    used_per_folder = Counter(pool[i][2].split("/")[1] for i in range(used))
    used_tokens_per_folder = Counter()
    used_answer_tasks = Counter()
    for i in range(used):
        used_tokens_per_folder[pool[i][2].split("/")[1]] += pool[i][3]
        used_answer_tasks[pool[i][2]] += 1
    total = sum(web_tokens) + flan_tokens
    manifest = {
        "build": build, "file": str(path), "config": config, "seconds": round(time.time() - started),
        "web": {"documents": len(web_texts), "tokens": sum(web_tokens), "target_tokens": target,
                "order": "the first documents of the trainer's stream of web_files (seed 42, buffer 10,000), in that order"},
        "flan": {"pool_examples": len(pool), "pool_tokens": sum(p[3] for p in pool), "examples": used, "tokens": flan_tokens,
                 "examples_per_folder": dict(used_per_folder), "tokens_per_folder": dict(used_tokens_per_folder),
                 "tasks_used": len(used_answer_tasks), "max_examples_per_task": max(used_answer_tasks.values()), **flan_report},
        "rows": len(sequence), "tokens": total, "flan_token_share": flan_tokens / total,
        # Lower bound: the trainer pads rows, so it reads more steps from these tokens than this
        "steps_at_full_rows": total / (args.rows_per_host * args.max_length),
    }
    (out / f"manifest-{build}.json").write_text(json.dumps(manifest, indent=1) + "\n")
    Path(args.list).write_text(f"{path}\n")
    print(f"MIX {json.dumps({k: manifest[k] for k in ('build', 'file', 'rows', 'tokens', 'flan_token_share', 'steps_at_full_rows')})}", flush=True)
    print(f"done: {path}, listed in {args.list}; manifest {out / f'manifest-{build}.json'}", flush=True)
    return manifest


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="One host's mix of web text and FLAN for the decay of arm F")
    parser.add_argument("--rank", type=int, required=True, help="Worker number 0..world-1: which part of FLAN this host takes")
    parser.add_argument("--world", type=int, default=4)
    parser.add_argument("--web-files", required=True, help="This host's --data-files-list of arm W, e.g. ~/culturax_nl_files.txt")
    parser.add_argument("--out", required=True, help="Directory of the parquet file and its manifest")
    parser.add_argument("--list", required=True, help="File list for the trainer's --data-files-list")
    parser.add_argument("--niv2-languages", required=True, help="experiments/flan-decay/niv2_languages.json")
    parser.add_argument("--download-dir", default="/dev/shm/flan_decay", help="FLAN shards while they are read (default: /dev/shm/flan_decay)")
    parser.add_argument("--tokenizer", default="yhavinga/dutch-llama-tokenizer")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--rows-per-host", type=int, default=32, help="Rows per host per step: 8 per chip x 4 chips")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--flan-share", type=float, default=0.1, help="FLAN's share of the tokens (default: 0.1)")
    parser.add_argument("--margin", type=float, default=0.2, help="Web tokens beyond the steps' need (default: 0.2)")
    parser.add_argument("--cap", type=int, default=2000, help="Most rows per (folder, task) over all hosts (default: 2000)")
    parser.add_argument("--seed", type=int, default=42, help="Seed of the order of the FLAN pool")
    parser.add_argument("--threads", type=int, default=32, help="Tokenizer threads; leave a trainer on this host its CPU (default: 32)")
    args = parser.parse_args(argv)
    if not 0 <= args.rank < args.world:
        parser.error(f"--rank {args.rank} is not in 0..{args.world - 1}")
    if not 0 < args.flan_share < 1:
        parser.error("--flan-share must be in (0, 1)")
    return args


def main(argv=None):
    args = parse_args(argv)
    os.environ["RAYON_NUM_THREADS"] = str(args.threads)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    from huggingface_hub import get_token
    from transformers import AutoTokenizer

    from mmbjax.training import add_answer_token

    if get_token() is None:
        raise SystemExit("no Hugging Face token: put it in ~/.cache/huggingface/token first (commands.txt)")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, split_special_tokens=True)
    if tokenizer.mask_token is None:
        # As the trainer does: the text is tokenized the same with or without these, but the ids must be the trainer's
        tokenizer.add_special_tokens({"mask_token": "<mask>"})
    add_answer_token(tokenizer)
    build(args, tokenizer, HubShards(args.download_dir))


if __name__ == "__main__":
    main()
