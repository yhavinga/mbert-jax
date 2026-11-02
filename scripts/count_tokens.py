"""Count the training tokens of a Dutch pretraining corpus the way scripts/train_minimal.py sees them.

Three steps, each its own subcommand:

  count    Download every file of the corpus, tokenize every document with the trainer's TokenizeTransform
           and keep per document its token count, UTF-8 length and a 64-bit hash of its text. Writes per
           source file an .npz and a "lengths" parquet: one row per document, in the file's order, whose
           column `text` holds the document's token count.
  pack     Run the trainer's own data pipeline (EpochTrackingDataLoader: shuffle buffer, pieces of
           --max-length, window shuffle, first-fit packing, batches) over the lengths parquets of one host,
           with a tokenizer that turns a token count into that many tokens. Counts batches = optimizer steps.
  summary  Documents, tokens, percentiles, shares of long documents, pieces, steps per epoch and epochs.

Example, on a TPU VM with the token in ~/.cache/huggingface/token:
  PYTHONPATH=src python scripts/count_tokens.py count --corpus culturax --work ~/tc --workers 70 \
      --gcs gs://t5-dutch-english-usc2/mbert/token-counts
  PYTHONPATH=src python scripts/count_tokens.py pack --corpus culturax --work ~/tc --rank 3
  PYTHONPATH=src python scripts/count_tokens.py summary --work ~/tc
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import gzip
import json
import multiprocessing
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
# mmbjax imports JAX; nothing here computes on it, and the TPU must stay free of these processes
os.environ.setdefault("JAX_PLATFORMS", "cpu")

TOKENIZER = "yhavinga/dutch-llama-tokenizer"
MAX_LENGTH = 1024
SEED = 42  # train_minimal.py passes seed=42 to create_packed_dataloader
SHUFFLE_BUFFER = 10000  # train_minimal.py --shuffle-buffer-size default
HOSTS = 4  # the v4-32 of the long run: 4 hosts of 4 chips
ROWS_PER_HOST = 32  # --per-device-batch-size 8 on 4 chips per host; global batch 128

# Revisions are pinned: CulturaX at the one stage.py staged for the long run, the other two at the
# revision of 2026-11-02, as the old runs did not record one.
CORPORA = {
    "culturax": {"repo": "uonlp/CulturaX", "config": "nl", "revision": "6a8734bc69fefcbb7735f4f9250f43e4cd7a442e",
                 "prefix": "nl/", "suffix": ".parquet"},
    "fineweb2": {"repo": "HuggingFaceFW/fineweb-2", "config": "nld_Latn",
                 "revision": "af9c13333eb981300149d5ca60a8e9d659b276b9", "prefix": "data/nld_Latn/train/",
                 "suffix": ".parquet"},
    "mc4": {"repo": "yhavinga/mc4_nl_cleaned", "config": "full", "revision": "187180b617793d6415c93fd2f950e34f8bf3e199",
            "prefix": "mc4_nl_cleaned/train/", "suffix": ".json.gz"},
}

TEXT_CHUNK = 256  # documents per tokenizer call


def piece_lengths(n_tokens: np.ndarray, max_length: int) -> tuple[np.ndarray, np.ndarray]:
    """Per document: the number of pieces split_into_pieces makes, and its tokens including <s> per piece and </s>.

    Vectorized form of split_into_pieces: full pieces take max_length - 1 text tokens while at least max_length
    remain; with exactly max_length - 1 left, one of them moves on with </s>, which makes one extra piece.
    """
    n = np.asarray(n_tokens, dtype=np.int64)
    room = max_length - 1
    full = np.where(n >= max_length, (n - max_length) // room + 1, 0)
    rest = n - full * room
    pieces = full + np.where(rest == room, 2, 1)
    return pieces, n + pieces + 1


def load_tokenizer(name: str = TOKENIZER):
    """As train_minimal.py main() does: split_special_tokens, and a <mask> token added if the tokenizer has none."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(name, split_special_tokens=True)
    if tokenizer.mask_token is None:
        tokenizer.add_special_tokens({"mask_token": "<mask>"})
    return tokenizer


# ----------------------------------------------------------------------------------------------- count

_TRANSFORM = None


def _init_worker(tokenizer_name: str, max_length: int) -> None:
    global _TRANSFORM
    from mmbjax.training.packing import TokenizeTransform

    _TRANSFORM = TokenizeTransform(load_tokenizer(tokenizer_name), max_length)


def _read_texts(path: str, row_groups: list[int] | None) -> list[str]:
    if row_groups is None:
        import orjson

        with gzip.open(path, "rb") as f:
            return [orjson.loads(line)["text"] for line in f]
    import pyarrow.parquet as pq

    return pq.ParquetFile(path).read_row_groups(row_groups, columns=["text"]).column("text").to_pylist()


def count_texts(texts: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Token count (no specials), UTF-8 bytes and xxh64 of each text, with the trainer's tokenization."""
    import xxhash

    counts = np.empty(len(texts), dtype=np.uint32)
    for start in range(0, len(texts), TEXT_CHUNK):
        ids = _TRANSFORM._tokenize(texts[start:start + TEXT_CHUNK])
        counts[start:start + len(ids)] = [len(x) for x in ids]
    encoded = [t.encode("utf-8") for t in texts]
    nbytes = np.fromiter((len(b) for b in encoded), dtype=np.uint32, count=len(texts))
    hashes = np.fromiter((xxhash.xxh64_intdigest(b) for b in encoded), dtype=np.uint64, count=len(texts))
    # The closed form of the pieces against the trainer's own map(), on the first documents of every task
    for text, n in zip(texts[:3], counts[:3]):
        real = _TRANSFORM.map({"text": text})
        pieces, tokens = piece_lengths(np.array([n]), _TRANSFORM.max_length)
        if len(real) != pieces[0] or sum(p["length"] for p in real) != tokens[0]:
            raise AssertionError(f"piece_lengths({n}) = {pieces[0]}, {tokens[0]}; TokenizeTransform.map: "
                                 f"{len(real)}, {sum(p['length'] for p in real)}")
    return counts, nbytes, hashes


def _count_task(path: str, row_groups: list[int] | None):
    return count_texts(_read_texts(path, row_groups))


def _tasks(path: str, min_rows: int = 50000) -> list[list[int] | None]:
    """Consecutive row groups of a parquet file in tasks of at least min_rows; a json.gz file is one task.

    FineWeb-2 has row groups of 1,000 documents, CulturaX of ~76,000.
    """
    if not path.endswith(".parquet"):
        return [None]
    import pyarrow.parquet as pq

    metadata = pq.ParquetFile(path).metadata
    tasks, current, rows = [], [], 0
    for i in range(metadata.num_row_groups):
        current.append(i)
        rows += metadata.row_group(i).num_rows
        if rows >= min_rows:
            tasks.append(current)
            current, rows = [], 0
    if current:
        tasks.append(current)
    return tasks


def source_files(corpus: str) -> list[str]:
    from huggingface_hub import HfApi

    spec = CORPORA[corpus]
    names = HfApi().list_repo_files(spec["repo"], repo_type="dataset", revision=spec["revision"])
    files = sorted(n for n in names if n.startswith(spec["prefix"]) and n.endswith(spec["suffix"]))
    if not files:
        raise RuntimeError(f"No {spec['suffix']} files under {spec['prefix']} in {spec['repo']}@{spec['revision']}")
    return files


def output_stem(repo_path: str) -> str:
    name = repo_path.rsplit("/", 1)[1]
    return name[: -len(".json.gz")] if name.endswith(".json.gz") else name[: -len(".parquet")]


def _run(cmd: list[str]) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} exited {result.returncode}: {result.stderr[-2000:]}")


def cmd_count(args) -> None:
    from huggingface_hub import hf_hub_download

    spec = CORPORA[args.corpus]
    work = Path(args.work).expanduser()
    download_dir = work / "download" / args.corpus
    out = work / args.corpus
    (out / "counts").mkdir(parents=True, exist_ok=True)
    (out / "lengths").mkdir(parents=True, exist_ok=True)
    files = source_files(args.corpus)
    if args.limit_files:
        files = files[: args.limit_files]
    done = {p.stem for p in (out / "counts").glob("*.npz") if not p.name.endswith(".tmp.npz")}
    if args.gcs:
        # A spot VM loses its disk on preemption; what reached the bucket is done
        base = f"{args.gcs.rstrip('/')}/{args.corpus}"
        listing = subprocess.run(["gcloud", "storage", "ls", f"{base}/counts/*.npz"], capture_output=True, text=True)
        if listing.returncode != 0 and "matched no objects" not in listing.stderr:
            raise RuntimeError(f"gcloud storage ls {base}/counts/ exited {listing.returncode}: {listing.stderr[-2000:]}")
        done |= {line.rsplit("/", 1)[1][: -len(".npz")] for line in listing.stdout.split()}
    todo = [f for f in files if output_stem(f) not in done]
    (out / "files.json").write_text(json.dumps({**spec, "files": files}, indent=1))
    if args.gcs:
        _run(["gcloud", "storage", "cp", "--quiet", str(out / "files.json"), f"{base}/files.json"])
    print(f"{args.corpus}: {len(files)} files, {len(todo)} to do, {args.workers} tokenizer processes", flush=True)

    on_disk = threading.Semaphore(args.max_on_disk)

    def download(repo_path):
        on_disk.acquire()
        t = time.time()
        local = hf_hub_download(spec["repo"], repo_path, repo_type="dataset", revision=spec["revision"],
                                local_dir=download_dir)
        size = os.path.getsize(local)
        print(f"  downloaded {repo_path}: {size / 1e9:.2f} GB in {time.time() - t:.0f} s", flush=True)
        return local

    def finish(repo_path, local, parts):
        import pyarrow as pa
        import pyarrow.parquet as pq

        counts = np.concatenate([p[0] for p in parts])
        nbytes = np.concatenate([p[1] for p in parts])
        hashes = np.concatenate([p[2] for p in parts])
        stem = output_stem(repo_path)
        npz = out / "counts" / f"{stem}.npz"
        lengths = out / "lengths" / f"{stem}.parquet"
        pq.write_table(pa.table({"text": counts.astype(np.int32)}), lengths, row_group_size=1 << 20)
        if args.gcs:
            _run(["gcloud", "storage", "cp", "--quiet", str(lengths), f"{base}/lengths/{stem}.parquet"])
        # Written last and uploaded last: its presence means the file is done
        np.savez(npz.with_suffix(".tmp.npz"), counts=counts, utf8_bytes=nbytes, xxh64=hashes)
        os.replace(npz.with_suffix(".tmp.npz"), npz)
        if args.gcs:
            _run(["gcloud", "storage", "cp", "--quiet", str(npz), f"{base}/counts/{stem}.npz"])
        os.remove(local)
        on_disk.release()
        return repo_path, len(counts), int(counts.sum(dtype=np.int64)), int(nbytes.sum(dtype=np.int64))

    t0 = time.time()
    done_docs = done_tokens = done_files = 0
    with cf.ThreadPoolExecutor(args.downloads) as downloads, cf.ThreadPoolExecutor(4) as finishers, \
            cf.ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("spawn"),
                                   initializer=_init_worker, initargs=(args.tokenizer, args.max_length)) as pool:
        pending = {downloads.submit(download, f): ("download", f) for f in todo}
        parts: dict[str, list] = {}
        waiting: dict[str, int] = {}
        local_of: dict[str, str] = {}
        while pending:
            finished, _ = cf.wait(pending, return_when=cf.FIRST_COMPLETED)
            for future in finished:
                kind, key = pending.pop(future)
                result = future.result()  # raises the task's own error, with its traceback
                if kind == "download":
                    local_of[key] = result
                    tasks = _tasks(result)
                    parts[key] = [None] * len(tasks)
                    waiting[key] = len(tasks)
                    for i, task in enumerate(tasks):
                        pending[pool.submit(_count_task, result, task)] = ("tokenize", (key, i))
                elif kind == "tokenize":
                    repo_path, i = key
                    parts[repo_path][i] = result
                    waiting[repo_path] -= 1
                    if waiting[repo_path] == 0:
                        pending[finishers.submit(finish, repo_path, local_of.pop(repo_path), parts.pop(repo_path))] = ("finish", repo_path)
                else:
                    repo_path, docs, tokens, nbytes = result
                    done_files += 1
                    done_docs += docs
                    done_tokens += tokens
                    elapsed = time.time() - t0
                    print(f"DONE {done_files}/{len(todo)} {repo_path}: {docs:,} docs, {tokens:,} tokens, "
                          f"{nbytes / 1e9:.2f} GB text | total {done_docs:,} docs, {done_tokens:,} tokens, "
                          f"{elapsed:.0f} s, {done_tokens / elapsed / 1e6:.1f} M tokens/s", flush=True)
    print(f"{args.corpus}: count complete, {done_files} files in {time.time() - t0:.0f} s", flush=True)


# ------------------------------------------------------------------------------------------------ pack

class LengthTokenizer:
    """Stands in for the tokenizer in the trainer's pipeline: a document's `text` is its token count.

    The packer only sees lengths, so batches, pieces and padding come out as with the real text.
    """

    def __init__(self, bos_token_id: int, eos_token_id: int, pad_token_id: int, token_id: int):
        if token_id in (bos_token_id, eos_token_id, pad_token_id):
            raise ValueError(f"token_id {token_id} is a special token")
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self._token_id = token_id

    def __call__(self, text, **kwargs):
        if isinstance(text, list):
            return {"input_ids": [[self._token_id] * (0 if t == "" else int(t)) for t in text]}
        return {"input_ids": [self._token_id] * (0 if text == "" else int(text))}


def long_run_file_order(rank: int, world_size: int = HOSTS) -> list[str]:
    """The CulturaX files host `rank` of the long run read, in its --data-files-list order (as stage.py made it).

    stage.py takes the files split_dataset_by_node gives the host and the order in which shuffle() reads
    them, then lists them so that the local parquet route reads them in that same order.
    """
    from datasets import load_dataset
    from datasets.distributed import split_dataset_by_node

    spec = CORPORA["culturax"]
    ds = load_dataset(spec["repo"], spec["config"], split="train", streaming=True,
                      download_mode="reuse_cache_if_exists", token=True)
    node = split_dataset_by_node(ds, rank=rank, world_size=world_size).shuffle(seed=SEED, buffer_size=SHUFFLE_BUFFER)
    files = node._prepare_ex_iterable_for_iteration().ex_iterable.kwargs["files"]
    return [re.search(r"nl_part_\d+\.parquet", str(f)).group(0) for fs in files for f in fs]


def _local_read_order(paths: list[str]) -> list[str]:
    from datasets import load_dataset

    local = load_dataset("parquet", data_files=paths, split="train", streaming=True).shuffle(
        seed=SEED, buffer_size=SHUFFLE_BUFFER)
    return [Path(str(f)).name for fs in local._prepare_ex_iterable_for_iteration().ex_iterable.kwargs["files"] for f in fs]


def listed_for_read_order(directory: Path, wanted: list[str]) -> list[str]:
    """Paths in `directory` listed so that the trainer's local route (parquet + shuffle) reads `wanted` in order."""
    reads = _local_read_order([str(directory / n) for n in wanted])
    position = {n: i for i, n in enumerate(reads)}
    listed = [wanted[position[n]] for n in wanted]
    paths = [str(directory / n) for n in listed]
    if _local_read_order(paths) != wanted:
        raise AssertionError(f"local route over {directory} doesn't read {wanted[:3]}... in order")
    return paths


def cmd_pack(args) -> None:
    from datasets import load_dataset
    from datasets.distributed import split_dataset_by_node

    from mmbjax.training.packing import EpochTrackingDataLoader

    work = Path(args.work).expanduser()
    lengths_dir = work / args.corpus / "lengths"
    spec = CORPORA[args.corpus]
    real = load_tokenizer(args.tokenizer)
    tokenizer = LengthTokenizer(real.bos_token_id, real.eos_token_id, real.pad_token_id, token_id=100)
    common = dict(tokenizer=tokenizer, max_length=args.max_length, batch_size=args.rows_per_host,
                  shard_index=args.rank, shard_count=args.hosts, seed=SEED, num_epochs=1,
                  shuffle_buffer_size=SHUFFLE_BUFFER)

    if args.corpus == "culturax" and args.hosts == HOSTS:
        # The long run's route: this host's own files, through --data-files-list
        files = listed_for_read_order(lengths_dir, long_run_file_order(args.rank, args.hosts))
        loader = EpochTrackingDataLoader(dataset_name=spec["repo"], dataset_config=spec["config"], data_files=files, **common)
        route = "data-files-list of the long run"
    else:
        # The trainer's streaming route, over the lengths files instead of the hub repo
        files = [str(lengths_dir / f"{output_stem(f)}.parquet") for f in json.loads((work / args.corpus / "files.json").read_text())["files"]]

        class HubRouteLoader(EpochTrackingDataLoader):
            def _load_stream_dataset(self):
                stream = load_dataset("parquet", data_files=files, split="train", streaming=True)
                stream = split_dataset_by_node(stream, rank=self.shard_index, world_size=self.shard_count)
                return stream.shuffle(seed=self.base_seed + self.epoch, buffer_size=self.shuffle_buffer_size)

        loader = HubRouteLoader(dataset_name=spec["repo"], dataset_config=spec["config"], **common)
        route = "streaming route (split_dataset_by_node + shuffle)"

    t0 = time.time()
    batches = real_tokens = 0
    for batch in loader:
        batches += 1
        real_tokens += int(np.count_nonzero(batch["input_ids"] != tokenizer.pad_token_id))
        if batches % 100000 == 0:
            print(f"  {args.corpus} rank {args.rank}: {batches:,} batches, {loader.documents_read:,} docs, "
                  f"{loader.pieces_read:,} pieces, {time.time() - t0:.0f} s", flush=True)
    result = {"corpus": args.corpus, "rank": args.rank, "hosts": args.hosts, "rows_per_host": args.rows_per_host,
              "max_length": args.max_length, "route": route, "files": len(files), "batches": batches,
              "documents": loader.documents_read, "pieces": loader.pieces_read, "real_tokens": real_tokens,
              "seconds": round(time.time() - t0)}
    path = work / args.corpus / "pack" / f"rank{args.rank}_of{args.hosts}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=1))
    print("PACK " + json.dumps(result), flush=True)


# --------------------------------------------------------------------------------------------- summary

def corpus_stats(counts: np.ndarray, hashes: np.ndarray, nbytes: np.ndarray, max_length: int) -> dict:
    counts = counts.astype(np.int64)
    pieces, trained = piece_lengths(counts, max_length)
    total = int(counts.sum())
    p50, p90, p99 = (int(v) for v in np.percentile(counts, [50, 90, 99], method="inverted_cdf"))
    unique, first = np.unique(hashes, return_index=True)
    repeated = np.ones(len(hashes), dtype=bool)
    repeated[first] = False
    stats = {
        "documents": int(len(counts)),
        "tokens": total,
        "tokens_with_specials": int(trained.sum()),
        "utf8_bytes": int(nbytes.astype(np.int64).sum()),
        "mean_tokens_per_doc": total / len(counts),
        "p50": p50, "p90": p90, "p99": p99, "max": int(counts.max()),
        "empty_documents": int(np.count_nonzero(counts == 0)),
        "pieces": int(pieces.sum()),
        "split_documents": int(np.count_nonzero(pieces > 1)),
        "exact_duplicate_documents": int(np.count_nonzero(repeated)),
        "exact_duplicate_token_share": float(counts[repeated].sum() / total),
    }
    for limit in (1024, 2048):
        long = counts > limit
        stats[f"doc_share_over_{limit}"] = float(long.mean())
        stats[f"token_share_over_{limit}"] = float(counts[long].sum() / total)
    return stats


def cmd_summary(args) -> None:
    work = Path(args.work).expanduser()
    rows = {}
    for corpus in args.corpora:
        meta = json.loads((work / corpus / "files.json").read_text())
        parts = [np.load(work / corpus / "counts" / f"{output_stem(f)}.npz") for f in meta["files"]]
        stats = corpus_stats(np.concatenate([p["counts"] for p in parts]), np.concatenate([p["xxh64"] for p in parts]),
                             np.concatenate([p["utf8_bytes"] for p in parts]), args.max_length)
        packs = [json.loads((work / corpus / "pack" / f"rank{r}_of{HOSTS}.json").read_text()) for r in range(HOSTS)]
        if sum(p["documents"] for p in packs) != stats["documents"] or sum(p["pieces"] for p in packs) > stats["pieces"]:
            raise AssertionError(f"{corpus}: pack runs read {sum(p['documents'] for p in packs)} documents and "
                                 f"{sum(p['pieces'] for p in packs)} pieces, the counts have {stats['documents']} and {stats['pieces']}")
        stats["steps_per_epoch_per_host"] = [p["batches"] for p in packs]
        # Every host must have read all its documents once
        stats["steps_per_epoch"] = max(p["batches"] for p in packs)
        stats["mean_steps_per_epoch"] = sum(p["batches"] for p in packs) / HOSTS
        stats["pieces_per_host"] = [p["pieces"] for p in packs]
        stats["packing_fill"] = sum(p["real_tokens"] for p in packs) / (sum(p["batches"] for p in packs) * ROWS_PER_HOST * args.max_length)
        stats["epochs_at_653000_steps"] = 653000 / stats["mean_steps_per_epoch"]
        rows[corpus] = stats
    smallest = min(rows, key=lambda c: rows[c]["mean_steps_per_epoch"])
    for corpus, stats in rows.items():
        stats["epochs_at_one_epoch_of_smallest"] = rows[smallest]["mean_steps_per_epoch"] / stats["mean_steps_per_epoch"]
    summary = {"tokenizer": args.tokenizer, "max_length": args.max_length, "hosts": HOSTS, "rows_per_host": ROWS_PER_HOST,
               "smallest": smallest, "corpora": rows}
    (work / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tokenizer", default=TOKENIZER)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    sub = parser.add_subparsers(dest="command", required=True)
    count = sub.add_parser("count")
    count.add_argument("--corpus", choices=CORPORA, required=True)
    count.add_argument("--work", required=True, help="Directory for downloads and results")
    count.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    count.add_argument("--downloads", type=int, default=4, help="Files downloaded at once")
    count.add_argument("--max-on-disk", type=int, default=8, help="Source files on disk at once")
    count.add_argument("--gcs", default=None, help="Bucket prefix for the results; files already there are skipped")
    count.add_argument("--limit-files", type=int, default=None, help="Only the first N files (a smoke-test)")
    pack = sub.add_parser("pack")
    pack.add_argument("--corpus", choices=CORPORA, required=True)
    pack.add_argument("--work", required=True)
    pack.add_argument("--rank", type=int, required=True)
    pack.add_argument("--hosts", type=int, default=HOSTS)
    pack.add_argument("--rows-per-host", type=int, default=ROWS_PER_HOST)
    summary = sub.add_parser("summary")
    summary.add_argument("--work", required=True)
    summary.add_argument("--corpora", nargs="+", default=list(CORPORA))
    args = parser.parse_args(argv)
    {"count": cmd_count, "pack": cmd_pack, "summary": cmd_summary}[args.command](args)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
