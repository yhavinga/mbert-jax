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
  stage  On a worker: each source file of the plan's revision is downloaded from the Hugging Face repo,
         checked against the plan's row count, and this host's rows are written as one parquet file with
         the column `text`; then the download is deleted. Then the list --data-files-list reads. A file
         already staged with the planned rows is kept, so a second start continues.

  python scripts/stage_share.py plan --corpus fineweb2 --lengths ~/tc/fineweb2/lengths
  python scripts/stage_share.py stage --corpus fineweb2 --rank 0 --out ~/fineweb2_nl --list ~/fineweb2_nl_files.txt
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import gzip
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


def stage_file(corpus: str, path: str, rows: int, ranges: list[tuple[int, int]], out: str) -> tuple[str, int, int, float]:
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
        download = target.with_name(target.name + ".download")
        partial.unlink(missing_ok=True)
        shutil.rmtree(download, ignore_errors=True)
        try:
            local = Path(hf_hub_download(spec["repo"], path, repo_type="dataset", revision=spec["revision"], local_dir=download))
            writer = pq.ParquetWriter(partial, pa.schema([("text", pa.string())]), compression="zstd")
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
    jobs = [(args.corpus, f["path"], f["rows"], file_ranges, str(out_dir / share_file_name(f["path"], args.rank, hosts)))
            for f, file_ranges in zip(plan["files"], mine)]
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
    paths = [job[-1] for job in jobs]
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
    s.set_defaults(func=cmd_stage)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
