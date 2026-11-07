"""Stage one host's share of a corpus on its own disk, for scripts/train_minimal.py --data-files-list.

Every document is in exactly one share, and the shares differ by at most one document: of a file of n rows
each host gets n // hosts, and the n % hosts rows left over go to the hosts in turn, the turn carried on from
file to file. A host's rows of a file are not one block but segments of SEGMENT_ROWS rows, taken in rounds
(in round r the hosts take their next segment starting with host r % hosts). FineWeb-2 nl has longer
documents towards the end of each file: a quarter of every file as one block gave 4.4% more tokens on the
last host than on the first, segments of 1,000 rows 0.08%, as striding row by row does (0.09%).

  plan   Where the lengths parquets of scripts/count_tokens.py (branch token-counts; one row per document,
         its token count in `text`) are: rows and tokens per source file, and documents and tokens per share,
         to experiments/corpus-runs/<corpus>_plan.json. The stage step reads that plan from the checkout.
  stage  On a worker: each source file of the plan's revision is downloaded from the Hugging Face repo (into
         --download-dir, /dev/shm on a TPU VM, as the boot disk writes 27 MB/s), checked against the plan's
         row count, and this host's rows are written as one parquet file with the column `text`; then the
         download is deleted. Then the list --data-files-list reads. A file already staged with the planned
         rows is kept, so a second start continues.
  rehydrate  On a worker, after stage: FineWeb-2's rehydration (misc/rehydration of github.com/huggingface/fineweb-2)
         of the staged share, as a selection without repeats. A document of weight w (of its minhash_cluster_size)
         is kept with probability w / --max-reps, one draw per row of the source file, whichever host has it.
         The cluster sizes are read from the Hugging Face files, that column only; before anything is written,
         some segments of every file are compared row by row with the source. The kept rows go to --out under the
         staged names, another directory, so the trainer's data state (its list of paths) is another; the summary
         has per file the rows and tokens (of the lengths parquets) in and kept. A file already written with the
         rows to keep is kept, so a second start continues.

  python scripts/stage_share.py plan --corpus fineweb2 --lengths ~/tc/fineweb2/lengths
  python scripts/stage_share.py stage --corpus fineweb2 --rank 0 --out ~/fineweb2_nl --list ~/fineweb2_nl_files.txt
  python scripts/stage_share.py rehydrate --corpus fineweb2 --rank 0 --staged ~/fineweb2_nl --out ~/fineweb2_nl_rh3 \\
      --list ~/fineweb2_nl_rh3_files.txt --lengths ~/tc/fineweb2/lengths --max-reps 3 --seed 42
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import gzip
import io
import json
import os
import shutil
import sys
import time
from pathlib import Path

PLAN_DIR = Path(__file__).resolve().parents[1] / "experiments" / "corpus-runs"

# Revisions are those of the token count (count_tokens.py on branch token-counts)
CORPORA = {
    "fineweb2": {"repo": "HuggingFaceFW/fineweb-2", "config": "nld_Latn",
                 "revision": "af9c13333eb981300149d5ca60a8e9d659b276b9", "prefix": "data/nld_Latn/train/",
                 "suffix": ".parquet"},
    "mc4": {"repo": "yhavinga/mc4_nl_cleaned", "config": "full", "revision": "187180b617793d6415c93fd2f950e34f8bf3e199",
            "prefix": "mc4_nl_cleaned/train/", "suffix": ".json.gz"},
}

# Rows of a segment: a row group of FineWeb-2, so a host decompresses only its own row groups
SEGMENT_ROWS = 1000
# Rows per written row group: the trainer's parquet stream reads one row group at a time
ROWS_PER_GROUP = 50_000
# The boot disk of a v4 worker writes 27 MB/s: level 9 writes 0.81 of what Arrow's default (level 1, the size
# of FineWeb-2's own text column) does, at 75 MB/s of text per process
ZSTD_LEVEL = 9


def share_ranges(rows: list[int], hosts: int, segment: int = SEGMENT_ROWS) -> list[list[list[tuple[int, int]]]]:
    """ranges[host][file]: the rows [start, stop) of that file in the host's share, in the file's order."""
    if hosts < 1 or segment < 1:
        raise ValueError(f"hosts and segment must be at least 1, not {hosts} and {segment}")
    ranges: list[list[list[tuple[int, int]]]] = [[] for _ in range(hosts)]
    turn = 0  # the host that gets the next row left over
    for n in rows:
        if n < 0:
            raise ValueError(f"a file can't have {n} rows")
        base, rest = divmod(n, hosts)
        extra = {(turn + j) % hosts for j in range(rest)}
        turn = (turn + rest) % hosts
        left = [base + (k in extra) for k in range(hosts)]
        mine: list[list[tuple[int, int]]] = [[] for _ in range(hosts)]
        position, round_ = 0, 0
        while position < n:
            for i in range(hosts):
                k = (round_ + i) % hosts
                take = min(segment, left[k])
                if take:
                    mine[k].append((position, position + take))
                    position += take
                    left[k] -= take
            round_ += 1
        for k in range(hosts):
            ranges[k].append(mine[k])
    return ranges


def output_stem(repo_path: str) -> str:
    """As count_tokens.py names a source file's outputs."""
    name = repo_path.rsplit("/", 1)[1]
    return name[: -len(".json.gz")] if name.endswith(".json.gz") else name[: -len(".parquet")]


def share_file_name(repo_path: str, rank: int, hosts: int) -> str:
    """The share's part of a source file. The rank is in the name because a data state's identity is its list of
    paths: with the same names on every host, a host whose process index changed would take another's state."""
    return f"{output_stem(repo_path)}-w{rank}of{hosts}.parquet"


def plan_path(corpus: str) -> Path:
    return PLAN_DIR / f"{corpus}_plan.json"


def load_plan(corpus: str) -> dict:
    plan = json.loads(plan_path(corpus).read_text())
    spec = CORPORA[corpus]
    if (plan["repo"], plan["revision"], plan["segment_rows"]) != (spec["repo"], spec["revision"], SEGMENT_ROWS):
        raise ValueError(f"{plan_path(corpus)} is of {plan['repo']}@{plan['revision']} in segments of "
                         f"{plan['segment_rows']}, not {spec['repo']}@{spec['revision']} in segments of {SEGMENT_ROWS}")
    return plan


def take_rows(ranges: list[tuple[int, int]], offset: int, texts):
    """The parts of texts (rows offset, offset + 1, ...) that fall in the sorted ranges, as slices of texts."""
    end = offset + len(texts)
    return [texts[max(start, offset) - offset:min(stop, end) - offset] for start, stop in ranges if start < end and stop > offset]


# ------------------------------------------------------------------------------------------------- plan


def cmd_plan(args) -> None:
    import numpy as np
    import pyarrow.parquet as pq

    spec = CORPORA[args.corpus]
    lengths = Path(args.lengths).expanduser()
    stems = sorted(p.stem for p in lengths.glob("*.parquet"))
    if not stems:
        raise FileNotFoundError(f"no lengths parquets in {lengths}")
    files = [f"{spec['prefix']}{stem}{spec['suffix']}" for stem in stems]
    counts = [pq.read_table(lengths / f"{stem}.parquet", columns=["text"]).column("text").to_numpy().astype(np.int64)
              for stem in stems]
    ranges = share_ranges([len(c) for c in counts], args.hosts)
    shares = []
    for k, host in enumerate(ranges):
        tokens = sum(int(c[start:stop].sum()) for c, file_ranges in zip(counts, host) for start, stop in file_ranges)
        documents = sum(stop - start for file_ranges in host for start, stop in file_ranges)
        shares.append({"rank": k, "documents": documents, "tokens": tokens})
    plan = {
        "corpus": args.corpus, "repo": spec["repo"], "config": spec["config"], "revision": spec["revision"],
        "hosts": args.hosts, "segment_rows": SEGMENT_ROWS,
        "documents": sum(len(c) for c in counts),
        "tokens": sum(int(c.sum()) for c in counts),
        "tokens_note": "without <s> and </s>, of yhavinga/dutch-llama-tokenizer as the trainer tokenizes (count_tokens.py)",
        "shares": shares,
        "files": [{"path": f, "rows": len(c), "tokens": int(c.sum())} for f, c in zip(files, counts)],
    }
    out = plan_path(args.corpus)
    out.write_text(json.dumps(plan, indent=1) + "\n")
    tokens = [s["tokens"] for s in shares]
    print(json.dumps({k: plan[k] for k in ("corpus", "documents", "tokens", "shares")}, indent=1))
    print(f"token spread {(max(tokens) - min(tokens)) / (sum(tokens) / len(tokens)) * 100:.3f}%; wrote {out}")


# ------------------------------------------------------------------------------------------------ stage


def _write(writer, parts) -> None:
    import pyarrow as pa

    chunks = []
    for part in parts:
        # A list of str from json.gz, or a slice of a parquet file's column
        chunks.extend([pa.array(part, type=pa.string())] if isinstance(part, list) else part.cast(pa.string()).chunks)
    texts = pa.chunked_array(chunks, type=pa.string())
    if len(texts):
        writer.write_table(pa.table({"text": texts}), row_group_size=ROWS_PER_GROUP)


def _parquet_rows(local: Path, rows: int, ranges: list[tuple[int, int]], writer) -> None:
    import pyarrow.parquet as pq

    source = pq.ParquetFile(local)
    if source.metadata.num_rows != rows:
        raise ValueError(f"{local} has {source.metadata.num_rows} rows, the plan {rows}")
    parts, pending, offset = [], 0, 0
    for i in range(source.metadata.num_row_groups):
        n = source.metadata.row_group(i).num_rows
        if any(start < offset + n and stop > offset for start, stop in ranges):
            texts = source.read_row_group(i, columns=["text"]).column("text")
            for part in take_rows(ranges, offset, texts):
                parts.append(part)
                pending += len(part)
            if pending >= ROWS_PER_GROUP:
                _write(writer, parts)
                parts, pending = [], 0
        offset += n
    _write(writer, parts)


def _json_gz_rows(local: Path, rows: int, ranges: list[tuple[int, int]], writer) -> None:
    import orjson

    wanted = iter(ranges)
    current = next(wanted, None)
    texts, row = [], 0
    with gzip.open(local, "rb") as f:
        for line in f:
            while current is not None and row >= current[1]:
                current = next(wanted, None)
            if current is not None and current[0] <= row:
                texts.append(orjson.loads(line)["text"])
                if len(texts) == ROWS_PER_GROUP:
                    _write(writer, [texts])
                    texts = []
            row += 1
    _write(writer, [texts])
    if row != rows:
        raise ValueError(f"{local} has {row} rows, the plan {rows}")


def stage_file(corpus: str, path: str, rows: int, ranges: list[tuple[int, int]], out: str,
               download_dir: str) -> tuple[str, int, int, float]:
    """One source file's part of a share, written under a temporary name first. Returns (out, rows, bytes, seconds)."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    spec = CORPORA[corpus]
    target = Path(out)
    wanted = sum(stop - start for start, stop in ranges)
    t = time.time()
    if not (target.exists() and pq.ParquetFile(target).metadata.num_rows == wanted):
        partial = target.with_name(target.name + ".partial")
        download = Path(download_dir) / (target.name + ".download")
        partial.unlink(missing_ok=True)
        shutil.rmtree(download, ignore_errors=True)
        try:
            local = Path(hf_hub_download(spec["repo"], path, repo_type="dataset", revision=spec["revision"], local_dir=download))
            writer = pq.ParquetWriter(partial, pa.schema([("text", pa.string())]), compression="zstd",
                                      compression_level=ZSTD_LEVEL)
            (_parquet_rows if path.endswith(".parquet") else _json_gz_rows)(local, rows, ranges, writer)
            writer.close()
        finally:
            shutil.rmtree(download, ignore_errors=True)
        written = pq.ParquetFile(partial).metadata.num_rows
        if written != wanted:
            raise ValueError(f"{partial}: {written} rows written, {wanted} wanted")
        os.replace(partial, target)
    return out, wanted, target.stat().st_size, time.time() - t


def cmd_stage(args) -> None:
    import pyarrow.parquet as pq

    plan = load_plan(args.corpus)
    hosts = plan["hosts"]
    if not 0 <= args.rank < hosts:
        raise ValueError(f"--rank {args.rank} is not one of the plan's {hosts} hosts")
    mine = share_ranges([f["rows"] for f in plan["files"]], hosts)[args.rank]
    planned = plan["shares"][args.rank]
    if sum(stop - start for file_ranges in mine for start, stop in file_ranges) != planned["documents"]:
        raise ValueError(f"the split gives rank {args.rank} another number of documents than the plan's {planned}")
    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    download_dir = Path(args.download_dir).expanduser() if args.download_dir else out_dir
    download_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(args.corpus, f["path"], f["rows"], file_ranges, str(out_dir / share_file_name(f["path"], args.rank, hosts)),
             str(download_dir)) for f, file_ranges in zip(plan["files"], mine)]
    print(f"rank {args.rank} of {hosts}: {len(jobs)} files, {planned['documents']:,} documents, "
          f"{planned['tokens']:,} tokens planned, {args.workers} files at a time", flush=True)
    t = time.time()
    done = 0
    with cf.ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(stage_file, *job) for job in jobs]
        for future in cf.as_completed(futures):
            path, n, nbytes, seconds = future.result()
            done += 1
            print(f"[{done}/{len(jobs)}] {Path(path).name}: {n:,} rows, {nbytes / 1e9:.2f} GB, {seconds:.0f} s", flush=True)
    paths = [job[4] for job in jobs]
    documents = sum(pq.ParquetFile(p).metadata.num_rows for p in paths)
    nbytes = sum(Path(p).stat().st_size for p in paths)
    if documents != planned["documents"]:
        raise ValueError(f"{out_dir} holds {documents} documents, the plan {planned['documents']}")
    summary = {"corpus": args.corpus, "repo": plan["repo"], "revision": plan["revision"], "rank": args.rank,
               "hosts": hosts, "files": len(paths), "documents": documents, "tokens_planned": planned["tokens"],
               "bytes": nbytes, "seconds": round(time.time() - t)}
    (out_dir / "share.json").write_text(json.dumps(summary, indent=1) + "\n")
    Path(args.list).expanduser().write_text("\n".join(paths) + "\n")
    print(f"SHARE {json.dumps(summary)}", flush=True)


# -------------------------------------------------------------------------------------------- rehydrate

# FineWeb-2's rehydration weights of nld_Latn per maximum of repetitions: misc/rehydration/weights/up_to_03_reps.json
# of github.com/huggingface/fineweb-2 (commit 995a1ca). With them nld_Latn's 147,301,270 documents become 233,693,423
# (weights/resulting_ds_sizes.json).
REHYDRATION_WEIGHTS = {3: {1: 1, 4: 2, 10: 3, 45: 2, 124: 1}}


def expand_weights(weights: dict[int, int]) -> list[int]:
    """table[s]: the weight of cluster size s, built as the Rehydrater of misc/rehydration builds it. A size takes the
    weight of the largest key at or below it; a size past the table takes the last entry."""
    table = [1]
    for size in range(1, max(weights) + 1):
        table.append(weights.get(size, table[-1]))
    return table


def cluster_weights(sizes, weights: dict[int, int]):
    import numpy as np

    if len(sizes) and sizes.min() < 0:
        raise ValueError(f"a cluster size of {sizes.min()}")
    table = np.array(expand_weights(weights))
    return table[np.minimum(sizes, len(table) - 1)]


def keep_rows(seed: int, file_index: int, rows: int, ranges: list[tuple[int, int]], weights, max_reps: int):
    """Whether each of this host's rows of a file is kept, with probability weight / max_reps. The draws are one per
    row of the whole source file, so a document's fate is the same whichever host has it."""
    import numpy as np

    if len(weights) and weights.max() > max_reps:
        raise ValueError(f"a weight of {weights.max()} is more than --max-reps {max_reps}")
    draws = np.random.default_rng([seed, file_index]).random(rows)
    mine = np.concatenate([draws[start:stop] for start, stop in ranges] + [np.empty(0)])
    if len(mine) != len(weights):
        raise ValueError(f"{len(weights)} weights for {len(mine)} rows")
    return mine < weights / max_reps


class _RangeFile(io.RawIOBase):
    """A file read by HTTP range requests, at the address the Hub's /resolve/ URL redirects to. pyarrow reads a
    column as one range per row group, ~1,000 per FineWeb-2 file and host; HfFileSystem sends each range to /resolve/,
    of which the Hub allows a free account 5,000 per 5 minutes. This asks /resolve/ once per file."""

    def __init__(self, url: str, size: int):
        super().__init__()
        self.url, self.size, self.position = url, size, 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        self.position = (offset, self.position + offset, self.size + offset)[whence]
        return self.position

    def read(self, n: int = -1) -> bytes:
        from huggingface_hub.utils import http_backoff

        n = self.size - self.position if n is None or n < 0 else min(n, self.size - self.position)
        if n <= 0:
            return b""
        span = f"bytes={self.position}-{self.position + n - 1}"
        r = http_backoff("GET", self.url, headers={"Range": span}, retry_on_status_codes=(429, 500, 502, 503, 504))
        r.raise_for_status()
        if r.status_code != 206 or len(r.content) != n:
            raise IOError(f"{self.url.split('?')[0]}: {span} gave status {r.status_code} with {len(r.content)} bytes")
        self.position += n
        return r.content


def open_source(repo: str, revision: str, path: str):
    """A file of the dataset repo at that revision, for pyarrow, which then reads only what it needs of it."""
    from huggingface_hub import get_hf_file_metadata, hf_hub_url

    meta = get_hf_file_metadata(hf_hub_url(repo, path, repo_type="dataset", revision=revision))
    if meta.size is None:
        raise ValueError(f"the Hub gives no size of {repo}@{revision}/{path}")
    return _RangeFile(meta.location, meta.size)


def _rows(parquet, ranges: list[tuple[int, int]], column: str):
    """The rows in the sorted ranges of one column of a parquet file, in its order: only the row groups they are in are
    read."""
    import pyarrow as pa

    parts, offset = [], 0
    for i in range(parquet.metadata.num_row_groups):
        n = parquet.metadata.row_group(i).num_rows
        if any(start < offset + n and stop > offset for start, stop in ranges):
            parts.extend(take_rows(ranges, offset, parquet.read_row_group(i, columns=[column]).column(column)))
        offset += n
    return pa.chunked_array([chunk for part in parts for chunk in part.chunks], type=parquet.schema_arrow.field(column).type)


def check_file(corpus: str, index: int, path: str, rows: int, tokens: int, ranges: list[tuple[int, int]], staged: str,
               lengths: str, max_reps: int, seed: int, rank: int):
    """This host's rows of one source file: the rows to keep, and the counts of the summary. Before that, the staged
    file is compared with the source on its first, its last and one random segment. Returns (counts, keep)."""
    import numpy as np
    import pyarrow.parquet as pq

    spec = CORPORA[corpus]
    wanted = sum(stop - start for start, stop in ranges)
    staged_file = pq.ParquetFile(staged)
    if staged_file.metadata.num_rows != wanted:
        raise ValueError(f"{staged} has {staged_file.metadata.num_rows} rows, rank {rank}'s share of {path} {wanted}")
    source = pq.ParquetFile(open_source(spec["repo"], spec["revision"], path))
    if source.metadata.num_rows != rows:
        raise ValueError(f"{path} has {source.metadata.num_rows} rows, the plan {rows}")
    counts = pq.read_table(lengths, columns=["text"]).column("text").to_numpy().astype(np.int64)
    if len(counts) != rows or int(counts.sum()) != tokens:
        raise ValueError(f"{lengths} has {len(counts)} rows of {int(counts.sum())} tokens, the plan {rows} of {tokens}")

    firsts = np.cumsum([0] + [stop - start for start, stop in ranges])  # where each segment starts in the staged file
    segments = []
    if ranges:
        segments = sorted({0, len(ranges) - 1, int(np.random.default_rng([seed, index, rank]).integers(len(ranges)))})
    for s in segments:
        start, stop = ranges[s]
        theirs = _rows(source, [(start, stop)], "text").to_pylist()
        if _rows(staged_file, [(firsts[s], firsts[s + 1])], "text").to_pylist() != theirs:
            raise ValueError(f"rows {firsts[s]}-{firsts[s + 1]} of {staged} are not rows {start}-{stop} of {path}")

    sizes = _rows(source, ranges, "minhash_cluster_size")
    if sizes.null_count:
        raise ValueError(f"{path}: {sizes.null_count} rows of rank {rank} without a minhash_cluster_size")
    weights = cluster_weights(sizes.to_numpy(), REHYDRATION_WEIGHTS[max_reps])
    keep = keep_rows(seed, index, rows, ranges, weights, max_reps)
    mine = np.concatenate([counts[start:stop] for start, stop in ranges] + [np.empty(0, np.int64)])
    by_weight = {str(w): {"rows_in": int((weights == w).sum()), "rows_kept": int((keep & (weights == w)).sum()),
                          "tokens_in": int(mine[weights == w].sum()), "tokens_kept": int(mine[keep & (weights == w)].sum())}
                 for w in sorted(set(REHYDRATION_WEIGHTS[max_reps].values()))}
    return {"path": path, "rows_in": wanted, "rows_kept": int(keep.sum()), "tokens_in": int(mine.sum()),
            "tokens_kept": int(mine[keep].sum()), "by_weight": by_weight,
            "segments_checked": [[int(a), int(b)] for a, b in (ranges[s] for s in segments)]}, keep


def write_kept(staged: str, keep, out: str) -> tuple[str, int, int, float]:
    """The kept rows of a staged file, in its order, written under a temporary name first. A file already written with
    as many rows is kept. Returns (out, rows, bytes, seconds)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    target = Path(out)
    wanted = int(keep.sum())
    t = time.time()
    if not (target.exists() and pq.ParquetFile(target).metadata.num_rows == wanted):
        partial = target.with_name(target.name + ".partial")
        partial.unlink(missing_ok=True)
        source = pq.ParquetFile(staged)
        if source.metadata.num_rows != len(keep):
            raise ValueError(f"{staged} has {source.metadata.num_rows} rows, {len(keep)} decided")
        writer = pq.ParquetWriter(partial, pa.schema([("text", pa.string())]), compression="zstd",
                                  compression_level=ZSTD_LEVEL)
        parts, pending, offset = [], 0, 0
        for i in range(source.metadata.num_row_groups):
            texts = source.read_row_group(i, columns=["text"]).column("text")
            part = texts.filter(pa.array(keep[offset:offset + len(texts)]))
            offset += len(texts)
            parts.append(part)
            pending += len(part)
            if pending >= ROWS_PER_GROUP:
                _write(writer, parts)
                parts, pending = [], 0
        _write(writer, parts)
        writer.close()
        written = pq.ParquetFile(partial).metadata.num_rows
        if written != wanted:
            raise ValueError(f"{partial}: {written} rows written, {wanted} kept")
        os.replace(partial, target)
    return out, wanted, target.stat().st_size, time.time() - t


def cmd_rehydrate(args) -> None:
    plan = load_plan(args.corpus)
    hosts = plan["hosts"]
    if not 0 <= args.rank < hosts:
        raise ValueError(f"--rank {args.rank} is not one of the plan's {hosts} hosts")
    # The same call as the stage step's, so the sizes read line up with the staged rows
    mine = share_ranges([f["rows"] for f in plan["files"]], hosts)[args.rank]
    staged_dir, out_dir = Path(args.staged).expanduser(), Path(args.out).expanduser()
    if staged_dir.resolve() == out_dir.resolve():
        raise ValueError(f"--out is --staged ({staged_dir}): the rehydrated share needs paths of its own")
    lengths = Path(args.lengths).expanduser()
    files = plan["files"][:args.max_files]
    names = [share_file_name(f["path"], args.rank, hosts) for f in files]
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"rank {args.rank} of {hosts}: {len(files)} of {len(plan['files'])} files, keeping each document with "
          f"probability w / {args.max_reps} of {REHYDRATION_WEIGHTS[args.max_reps]}, seed {args.seed}; "
          f"{args.readers} files read at a time, {args.workers} written", flush=True)
    t = time.time()
    # Every file is checked before any is written: a share that does not line up with the source writes nothing
    with cf.ThreadPoolExecutor(max_workers=args.readers) as pool:
        futures = [pool.submit(check_file, args.corpus, i, f["path"], f["rows"], f["tokens"], file_ranges,
                               str(staged_dir / name), str(lengths / f"{output_stem(f['path'])}.parquet"),
                               args.max_reps, args.seed, args.rank)
                   for i, (f, file_ranges, name) in enumerate(zip(files, mine, names))]
        try:
            for done, future in enumerate(cf.as_completed(futures), 1):
                counts, _ = future.result()
                print(f"[{done}/{len(files)}] checked {counts['path']}: {counts['rows_kept']:,} of {counts['rows_in']:,} "
                      f"rows kept, segments {counts['segments_checked']} as the source", flush=True)
        except BaseException:
            # Else the error waits for the files still queued, minutes of range requests
            pool.shutdown(cancel_futures=True)
            raise
        checked = [future.result() for future in futures]
    print(f"all {len(files)} files checked in {time.time() - t:.0f} s", flush=True)
    written = {}
    with cf.ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(write_kept, str(staged_dir / name), keep, str(out_dir / name))
                   for name, (_, keep) in zip(names, checked)]
        for done, future in enumerate(cf.as_completed(futures), 1):
            path, n, nbytes, seconds = future.result()
            written[path] = nbytes
            print(f"[{done}/{len(files)}] {Path(path).name}: {n:,} rows, {nbytes / 1e9:.2f} GB, {seconds:.0f} s", flush=True)
    paths = [str(out_dir / name) for name in names]
    per_file = [{**counts, "out": p, "bytes": written[p]} for p, (counts, _) in zip(paths, checked)]
    totals = {key: sum(f[key] for f in per_file) for key in ("rows_in", "rows_kept", "tokens_in", "tokens_kept", "bytes")}
    by_weight = {w: {key: sum(f["by_weight"][w][key] for f in per_file) for key in per_file[0]["by_weight"][w]}
                 for w in per_file[0]["by_weight"]} if per_file else {}
    summary = {"corpus": args.corpus, "repo": plan["repo"], "revision": plan["revision"], "rank": args.rank, "hosts": hosts,
               "max_reps": args.max_reps, "weights": {str(k): v for k, v in REHYDRATION_WEIGHTS[args.max_reps].items()},
               "seed": args.seed, "staged": str(staged_dir), "files": len(paths), **totals, "by_weight": by_weight,
               "tokens_note": plan["tokens_note"], "seconds": round(time.time() - t), "per_file": per_file}
    if len(files) < len(plan["files"]):
        # No list and no share.json: those are of the whole share, which a smoke test of a few files is not
        print(f"PART {json.dumps({k: v for k, v in summary.items() if k != 'per_file'})}", flush=True)
        return
    if totals["rows_in"] != plan["shares"][args.rank]["documents"]:
        raise ValueError(f"{totals['rows_in']} rows read, rank {args.rank}'s share in the plan {plan['shares'][args.rank]}")
    (out_dir / "share.json").write_text(json.dumps(summary, indent=1) + "\n")
    Path(args.list).expanduser().write_text("\n".join(paths) + "\n")
    print(f"SHARE {json.dumps({k: v for k, v in summary.items() if k != 'per_file'})}", flush=True)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--corpus", required=True, choices=sorted(CORPORA))
    p.add_argument("--lengths", required=True, help="directory of the lengths parquets of count_tokens.py")
    p.add_argument("--hosts", type=int, default=4)
    p.set_defaults(func=cmd_plan)
    s = sub.add_parser("stage")
    s.add_argument("--corpus", required=True, choices=sorted(CORPORA))
    s.add_argument("--rank", type=int, required=True, help="this host's share, the worker number")
    s.add_argument("--out", required=True, help="directory of the share's parquet files")
    s.add_argument("--list", required=True, help="file to write the share's paths to, for --data-files-list")
    s.add_argument("--workers", type=int, default=6, help="source files downloaded at a time; each needs its size on disk")
    s.add_argument("--download-dir", default=None,
                   help="where a source file waits while its rows are taken, e.g. /dev/shm: then the disk only writes "
                        "the share (default: --out)")
    s.set_defaults(func=cmd_stage)
    r = sub.add_parser("rehydrate")
    # Only FineWeb-2 has a minhash_cluster_size
    r.add_argument("--corpus", required=True, choices=["fineweb2"])
    r.add_argument("--rank", type=int, required=True, help="this host's share, the worker number")
    r.add_argument("--staged", required=True, help="directory of the share the stage step wrote")
    r.add_argument("--out", required=True, help="directory of the kept rows' parquet files, not --staged")
    r.add_argument("--list", required=True, help="file to write their paths to, for --data-files-list")
    r.add_argument("--lengths", required=True, help="directory of the lengths parquets of count_tokens.py, for the tokens")
    r.add_argument("--max-reps", type=int, required=True, choices=sorted(REHYDRATION_WEIGHTS))
    r.add_argument("--seed", type=int, default=42)
    r.add_argument("--readers", type=int, default=16,
                   help="source files read at a time; a range request takes ~0.2 s, a file ~1,000 of them per host")
    r.add_argument("--workers", type=int, default=4, help="files written at a time")
    r.add_argument("--max-files", type=int, default=None,
                   help="only the plan's first files, for a smoke test: then no list and no share.json")
    r.set_defaults(func=cmd_rehydrate)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
