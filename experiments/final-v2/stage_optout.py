"""The data of dmbert v2 (experiments/final-v2/protocol.md): kept/train of the opt-out-filtered FineWeb-2 nl, one share
per host of the v4-32 for scripts/train_minimal.py --data-files-list, and kept/test as the held-out set.

Source: gs://t5-dutch-english-usc2/mbert/data/fineweb2-nl-optout/ (us-central2, the region of the workers), written by
github.com/yhavinga/fineweb2-nl-optout; its manifest.json lists every output file with rows, bytes and md5.

  plan       From manifest.json: the kept/train files in name order with their rows, bytes and md5, kept/test, and
             every host's rows of every file as scripts/stage_share.py splits a corpus (segments of 1,000 rows taken in
             rounds, the rows left over in turn): a JSON plan with the sha256 of the manifest. Metadata only.
  stage      On a worker, by the plan: each file copied from the bucket into --download-dir (/dev/shm), its size, md5
             and rows checked against the plan, this host's rows written with the columns text and url (zstd 9, row
             groups of 50,000) as <out>/<stem>-w<rank>of<hosts>.parquet, then the copy deleted. Every row of the share
             is compared with kept/test by its id and by the sha256 of its text. A file already staged with its planned
             rows (and its comparison) is kept, so a second start continues. Then the list for --data-files-list (the
             plan's order; a file without rows for this host is left out), share.json and overlap.json.
  heldout    kept/test less every document that some host found in kept/train (the overlap.json of every host): one
             parquet file with the columns text, url and id, for --eval-file, and heldout.json with its md5.
  memsample  The sample of the memorization test (memtest.py): --documents documents of the held-out file and as many of
             this host's share, matched one to one on token count, each between --min-tokens and --max-tokens (one
             piece of 1,024 with <s> and </s>), drawn with --seed.

CPU and network only; nothing imports JAX. Deletes only its own downloads in --download-dir.

  python experiments/final-v2/stage_optout.py plan --manifest gs://.../manifest.json --out plan.json
  python experiments/final-v2/stage_optout.py stage --plan plan.json --rank 0 --out ~/fw2optout --list ~/fw2optout_files.txt \\
      --download-dir /dev/shm/fw2optout_stage
"""
from __future__ import annotations

import argparse
import base64
import concurrent.futures as cf
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from stage_share import share_ranges  # noqa: E402  # the split of the corpus runs, unchanged

SOURCE = "gs://t5-dutch-english-usc2/mbert/data/fineweb2-nl-optout"
SEGMENT_ROWS = 1000
ROWS_PER_GROUP = 50_000
ZSTD_LEVEL = 9
COLUMNS = ("text", "url")


def read_bytes(location: str) -> bytes:
    """A file's bytes, local or gs:// (through gsutil, which checks the object's crc32c)."""
    if location.startswith("gs://"):
        done = subprocess.run(["gsutil", "-q", "cat", location], capture_output=True, timeout=900)
        if done.returncode != 0:
            raise RuntimeError(f"gsutil cat {location}: {done.stderr.decode(errors='replace').strip()}")
        return done.stdout
    return Path(location).read_bytes()


def fetch(location: str, target: Path) -> None:
    """Copy one file of the source to target, local or gs://."""
    if location.startswith("gs://"):
        done = subprocess.run(["gsutil", "-q", "cp", location, str(target)], capture_output=True, timeout=1800)
        if done.returncode != 0:
            raise RuntimeError(f"gsutil cp {location}: {done.stderr.decode(errors='replace').strip()}")
    else:
        shutil.copyfile(location, target)


def md5_base64(path: Path) -> str:
    """The md5 as the manifest (and GCS) writes it: base64 of the digest."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(16 * 2**20):
            h.update(chunk)
    return base64.b64encode(h.digest()).decode()


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stem(file: str) -> str:
    name = file.rsplit("/", 1)[-1]
    if not name.endswith(".parquet"):
        raise ValueError(f"not a parquet file: {file}")
    return name[: -len(".parquet")]


def share_file_name(file: str, rank: int, hosts: int) -> str:
    """The rank is in the name: a data state names its files, so a host can't take another host's state."""
    return f"{stem(file)}-w{rank}of{hosts}.parquet"


# -------------------------------------------------------------------------------------------------------------- plan


def make_plan(manifest_bytes: bytes, source: str, hosts: int = 4, segment: int = SEGMENT_ROWS) -> dict:
    manifest = json.loads(manifest_bytes)
    kept = manifest["outputs"]["kept"]
    train = sorted((f for f in kept if f["file"].startswith("kept/train/")), key=lambda f: f["file"])
    test = [f for f in kept if f["file"].startswith("kept/test/")]
    other = [f["file"] for f in kept if not f["file"].startswith(("kept/train/", "kept/test/"))]
    if other or not train or len(test) != 1:
        raise ValueError(f"manifest: {len(train)} kept/train files, {len(test)} kept/test files, others {other[:3]}")
    if len({f["file"] for f in train}) != len(train):
        raise ValueError("manifest: a kept/train file is listed twice")
    rows = [int(f["rows"]) for f in train]
    counted = manifest["counts"]["split_kept"]
    if sum(rows) != counted["train"] or int(test[0]["rows"]) != counted["test"]:
        raise ValueError(f"manifest: the files hold {sum(rows)} train and {test[0]['rows']} test rows, its counts "
                         f"{counted['train']} and {counted['test']}")
    ranges = share_ranges(rows, hosts, segment)
    shares = [{"rank": k, "documents": sum(b - a for file_ranges in host for a, b in file_ranges),
               "files": sum(1 for file_ranges in host if file_ranges)} for k, host in enumerate(ranges)]
    return {
        "source": source,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "pipeline": manifest.get("code"),
        "fineweb2": {k: manifest["source"][k] for k in ("repo", "revision", "config")},
        "optout_snapshot_utc": manifest.get("optout_snapshot_utc"),
        "hosts": hosts,
        "segment_rows": segment,
        "documents": sum(rows),
        "columns": list(COLUMNS),
        "test": {k: test[0][k] for k in ("file", "rows", "bytes", "md5")},
        "shares": shares,
        "files": [{"file": f["file"], "rows": int(f["rows"]), "bytes": int(f["bytes"]), "md5": f["md5"],
                   "host_rows": [sum(b - a for a, b in host[i]) for host in ranges]} for i, f in enumerate(train)],
    }


def cmd_plan(args) -> None:
    plan = make_plan(read_bytes(args.manifest), args.source, args.hosts)
    text = json.dumps(plan, indent=1) + "\n"
    Path(args.out).write_text(text)
    docs = [s["documents"] for s in plan["shares"]]
    print(f"PLAN {len(plan['files'])} files, {plan['documents']:,} documents, per host {docs} "
          f"(spread {max(docs) - min(docs)}), manifest sha256 {plan['manifest_sha256'][:12]}, "
          f"plan sha256 {hashlib.sha256(text.encode()).hexdigest()[:12]}", flush=True)


def load_plan(path: str) -> tuple[dict, list]:
    """The plan and every host's ranges of every file, recomputed from its rows and checked against it."""
    plan = json.loads(Path(path).read_text())
    ranges = share_ranges([f["rows"] for f in plan["files"]], plan["hosts"], plan["segment_rows"])
    for i, f in enumerate(plan["files"]):
        if [sum(b - a for a, b in host[i]) for host in ranges] != f["host_rows"]:
            raise ValueError(f"{path}: the split of {f['file']} is not the plan's")
    return plan, ranges


# ------------------------------------------------------------------------------------------------------------- stage

_TEST_IDS: set = set()
_TEST_HASHES: dict = {}


def _init_test(ids: set, hashes: dict) -> None:
    global _TEST_IDS, _TEST_HASHES
    _TEST_IDS, _TEST_HASHES = ids, hashes


def read_test(path: Path) -> tuple[list, list, list]:
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=["id", "text", "url"])
    return table.column("id").to_pylist(), table.column("text").to_pylist(), table.column("url").to_pylist()


def test_index(path: Path) -> tuple[set, dict]:
    ids, texts, _ = read_test(path)
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: an id occurs twice in kept/test")
    hashes: dict = {}
    for i, t in zip(ids, texts):
        hashes.setdefault(text_hash(t), []).append(i)
    return set(ids), hashes


def overlaps_of(ids, texts, file: str, first_row: int) -> list:
    """The rows (of file, numbered from first_row) whose id or text is that of a document of kept/test."""
    found = []
    for offset, (i, t) in enumerate(zip(ids, texts)):
        if i in _TEST_IDS:
            found.append({"test_id": i, "by": "id", "file": file, "row": first_row + offset})
        for test_id in _TEST_HASHES.get(text_hash(t), ()):
            found.append({"test_id": test_id, "by": "text", "file": file, "row": first_row + offset})
    return found


def stage_file(job: dict) -> dict:
    """One source file's part of this host's share. Returns its summary, with the rows found in kept/test."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    target = Path(job["out"])
    sidecar = target.with_name(target.name + ".overlap.json")
    wanted = sum(b - a for a, b in job["ranges"])
    t = time.time()
    if target.exists() and sidecar.exists() and pq.ParquetFile(target).metadata.num_rows == wanted:
        found = json.loads(sidecar.read_text())
        return {"out": str(target), "rows": wanted, "bytes": target.stat().st_size, "seconds": 0.0, "kept": True,
                "overlaps": found}
    partial = target.with_name(target.name + ".partial")
    download = Path(job["download_dir"]) / (target.name + ".download")
    partial.unlink(missing_ok=True)
    download.unlink(missing_ok=True)
    found = []
    try:
        fetch(job["location"], download)
        if download.stat().st_size != job["bytes"] or md5_base64(download) != job["md5"]:
            raise ValueError(f"{job['location']}: {download.stat().st_size} bytes, md5 {md5_base64(download)}; "
                             f"the plan {job['bytes']} bytes, md5 {job['md5']}")
        source = pq.ParquetFile(download)
        if source.metadata.num_rows != job["rows"]:
            raise ValueError(f"{job['location']} has {source.metadata.num_rows} rows, the plan {job['rows']}")
        missing = {"text", "url", "id"} - set(source.schema_arrow.names)
        if missing:
            raise ValueError(f"{job['location']} lacks the columns {sorted(missing)}")
        schema = pa.schema([(c, pa.string()) for c in COLUMNS])
        writer = pq.ParquetWriter(partial, schema, compression="zstd", compression_level=ZSTD_LEVEL)
        pending: list = []
        n_pending, offset = 0, 0
        for g in range(source.metadata.num_row_groups):
            n = source.metadata.row_group(g).num_rows
            if any(a < offset + n and b > offset for a, b in job["ranges"]):
                group = source.read_row_group(g, columns=["text", "url", "id"])
                for a, b in job["ranges"]:
                    lo, hi = max(a, offset), min(b, offset + n)
                    if lo >= hi:
                        continue
                    part = group.slice(lo - offset, hi - lo)
                    found += overlaps_of(part.column("id").to_pylist(), part.column("text").to_pylist(), job["file"], lo)
                    pending.append(pa.table({c: part.column(c).cast(pa.string()) for c in COLUMNS}, schema=schema))
                    n_pending += hi - lo
                if n_pending >= ROWS_PER_GROUP:
                    writer.write_table(pa.concat_tables(pending), row_group_size=ROWS_PER_GROUP)
                    pending, n_pending = [], 0
            offset += n
        if pending:
            writer.write_table(pa.concat_tables(pending), row_group_size=ROWS_PER_GROUP)
        writer.close()
    finally:
        download.unlink(missing_ok=True)
    written = pq.ParquetFile(partial).metadata.num_rows
    if written != wanted:
        raise ValueError(f"{partial}: {written} rows written, {wanted} wanted")
    tmp = sidecar.with_name(sidecar.name + ".tmp")
    tmp.write_text(json.dumps(found))
    os.replace(tmp, sidecar)
    os.replace(partial, target)
    return {"out": str(target), "rows": wanted, "bytes": target.stat().st_size, "seconds": time.time() - t,
            "kept": False, "overlaps": found}


def fetch_test(plan: dict, download_dir: Path) -> Path:
    """kept/test, copied and checked against the plan (it is small: kept in download_dir)."""
    test = plan["test"]
    local = download_dir / "kept_test.parquet"
    if not (local.exists() and local.stat().st_size == test["bytes"] and md5_base64(local) == test["md5"]):
        fetch(f"{plan['source']}/{test['file']}", local)
    if local.stat().st_size != test["bytes"] or md5_base64(local) != test["md5"]:
        raise ValueError(f"{local}: not the plan's {test['file']} ({test['bytes']} bytes, md5 {test['md5']})")
    return local


def cmd_stage(args) -> None:
    import pyarrow.parquet as pq

    plan, ranges = load_plan(args.plan)
    hosts = plan["hosts"]
    if not 0 <= args.rank < hosts:
        raise ValueError(f"--rank {args.rank} is not one of the plan's {hosts} hosts")
    out_dir = Path(args.out).expanduser()
    download_dir = Path(args.download_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    download_dir.mkdir(parents=True, exist_ok=True)
    only = set(Path(args.files).read_text().split()) if args.files else None
    jobs = []
    for i, f in enumerate(plan["files"]):
        mine = ranges[args.rank][i]
        if not mine or (only is not None and f["file"] not in only):
            continue
        jobs.append({"file": f["file"], "location": f"{plan['source']}/{f['file']}", "rows": f["rows"], "bytes": f["bytes"],
                     "md5": f["md5"], "ranges": mine, "out": str(out_dir / share_file_name(f["file"], args.rank, hosts)),
                     "download_dir": str(download_dir)})
    if only is not None and len(jobs) != len(only):
        raise ValueError(f"--files: {len(only)} files listed, {len(jobs)} of them in the plan with rows for this host")
    planned = sum(sum(b - a for a, b in job["ranges"]) for job in jobs)
    if only is None and planned != plan["shares"][args.rank]["documents"]:
        raise ValueError(f"rank {args.rank}: {planned} rows in its ranges, the plan {plan['shares'][args.rank]['documents']}")
    ids, hashes = test_index(fetch_test(plan, download_dir))
    print(f"rank {args.rank} of {hosts}: {len(jobs)} files, {planned:,} documents; kept/test {len(ids):,} documents; "
          f"{args.workers} files at a time", flush=True)
    t = time.time()
    results = {}
    with cf.ProcessPoolExecutor(max_workers=args.workers, initializer=_init_test, initargs=(ids, hashes)) as pool:
        futures = {pool.submit(stage_file, job): job["out"] for job in jobs}
        for done, future in enumerate(cf.as_completed(futures), 1):
            r = future.result()
            results[r["out"]] = r
            if done % 50 == 0 or done == len(jobs):
                print(f"[{done}/{len(jobs)}] {time.time() - t:.0f} s, last {Path(r['out']).name}: {r['rows']:,} rows, "
                      f"{r['bytes'] / 1e6:.1f} MB{' (there already)' if r['kept'] else ''}", flush=True)
    paths = [job["out"] for job in jobs]
    documents = sum(pq.ParquetFile(p).metadata.num_rows for p in paths)
    if documents != planned:
        raise ValueError(f"{out_dir} holds {documents} documents, the plan {planned}")
    overlaps = [o for p in paths for o in results[p]["overlaps"]]
    summary = {"plan_sha256": hashlib.sha256(Path(args.plan).read_bytes()).hexdigest(), "rank": args.rank,
               "hosts": hosts, "files": len(paths), "documents": documents,
               "bytes": sum(Path(p).stat().st_size for p in paths), "overlaps": len(overlaps),
               "overlap_test_ids": len({o["test_id"] for o in overlaps}), "seconds": round(time.time() - t),
               "subset": only is not None}
    (out_dir / "overlap.json").write_text(json.dumps(overlaps, indent=0) + "\n")
    (out_dir / "share.json").write_text(json.dumps(summary, indent=1) + "\n")
    lst = Path(args.list).expanduser()
    tmp = lst.with_name(lst.name + ".new")
    tmp.write_text("".join(p + "\n" for p in paths))
    os.replace(tmp, lst)
    print(f"SHARE {json.dumps(summary)}", flush=True)


# ----------------------------------------------------------------------------------------------------------- heldout


def cmd_heldout(args) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    plan, _ = load_plan(args.plan)
    download_dir = Path(args.download_dir).expanduser()
    download_dir.mkdir(parents=True, exist_ok=True)
    ids, texts, urls = read_test(fetch_test(plan, download_dir))
    found = [o for path in args.overlap for o in json.loads(Path(path).read_text())]
    if len(args.overlap) != plan["hosts"]:
        raise ValueError(f"{len(args.overlap)} overlap files, the plan has {plan['hosts']} hosts")
    out_of = {o["test_id"] for o in found}
    unknown = out_of - set(ids)
    if unknown:
        raise ValueError(f"overlap files name {len(unknown)} ids that are not in kept/test, e.g. {sorted(unknown)[:2]}")
    keep = [k for k, i in enumerate(ids) if i not in out_of]
    table = pa.table({"text": pa.array([texts[k] for k in keep], pa.string()),
                      "url": pa.array([urls[k] for k in keep], pa.string()),
                      "id": pa.array([ids[k] for k in keep], pa.string())})
    out = Path(args.out).expanduser()
    tmp = out.with_name(out.name + ".partial")
    pq.write_table(table, tmp, compression="zstd", compression_level=ZSTD_LEVEL, row_group_size=ROWS_PER_GROUP)
    os.replace(tmp, out)
    md5 = hashlib.md5(out.read_bytes()).hexdigest()
    summary = {"plan_sha256": hashlib.sha256(Path(args.plan).read_bytes()).hexdigest(), "test_file": plan["test"]["file"],
               "test_documents": len(ids), "in_train": len(out_of), "by_id": len({o["test_id"] for o in found if o["by"] == "id"}),
               "by_text": len({o["test_id"] for o in found if o["by"] == "text"}), "documents": len(keep),
               "file": out.name, "md5": md5, "bytes": out.stat().st_size}
    Path(args.summary).expanduser().write_text(json.dumps(summary, indent=1) + "\n")
    print(f"HELDOUT {json.dumps(summary)}", flush=True)


# --------------------------------------------------------------------------------------------------------- memsample


def match_lengths(test_lengths, pool_lengths) -> list:
    """For each test document in order, the unused pool document of the nearest length (the shorter on a tie)."""
    import bisect

    order = sorted(range(len(pool_lengths)), key=lambda k: (pool_lengths[k], k))
    keys = [pool_lengths[k] for k in order]
    used = [False] * len(order)
    chosen = []
    for length in test_lengths:
        at = bisect.bisect_left(keys, length)
        lo, hi = at - 1, at
        while lo >= 0 and used[lo]:
            lo -= 1
        while hi < len(order) and used[hi]:
            hi += 1
        if lo < 0 and hi >= len(order):
            raise ValueError("the pool has fewer documents than the test sample")
        pick = hi if lo < 0 or (hi < len(order) and keys[hi] - length < length - keys[lo]) else lo
        used[pick] = True
        chosen.append(order[pick])
    return chosen


def cmd_memsample(args) -> None:
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    def count(texts):
        return [len(x) for x in tokenizer(texts, add_special_tokens=False, split_special_tokens=True)["input_ids"]]

    rng = np.random.default_rng(args.seed)
    held = pq.read_table(args.heldout, columns=["text", "id"])
    test_texts, test_ids = held.column("text").to_pylist(), held.column("id").to_pylist()
    test_lengths = count(test_texts)
    fit = [k for k, n in enumerate(test_lengths) if args.min_tokens <= n <= args.max_tokens]
    if len(fit) < args.documents:
        raise ValueError(f"{len(fit)} held-out documents of {args.min_tokens}-{args.max_tokens} tokens, {args.documents} wanted")
    test_pick = sorted(rng.choice(fit, size=args.documents, replace=False).tolist())

    files = Path(args.files_list).read_text().split()
    rows = [pq.ParquetFile(f).metadata.num_rows for f in files]
    total = sum(rows)
    draw = np.sort(rng.choice(total, size=min(args.pool, total), replace=False))
    starts = np.cumsum([0] + rows)
    pool_texts, pool_where = [], []
    for f, path in enumerate(files):
        mine = draw[(draw >= starts[f]) & (draw < starts[f + 1])] - starts[f]
        if mine.size:
            column = pq.read_table(path, columns=["text"]).column("text")
            pool_texts += [column[int(r)].as_py() for r in mine]
            pool_where += [f"{Path(path).name}:{int(r)}" for r in mine]
    pool_lengths = count(pool_texts)
    usable = [k for k, n in enumerate(pool_lengths) if args.min_tokens <= n <= args.max_tokens]
    chosen = match_lengths([test_lengths[k] for k in test_pick], [pool_lengths[k] for k in usable])
    train_pick = [usable[c] for c in chosen]
    table = pa.table({
        "text": [test_texts[k] for k in test_pick] + [pool_texts[k] for k in train_pick],
        "split": ["test"] * len(test_pick) + ["train"] * len(train_pick),
        "source": [f"heldout:{test_ids[k]}" for k in test_pick] + [pool_where[k] for k in train_pick],
        "tokens": pa.array([test_lengths[k] for k in test_pick] + [pool_lengths[k] for k in train_pick], pa.int32()),
    })
    pq.write_table(table, args.out)
    tl, rl = np.array(table.column("tokens").to_pylist()).reshape(2, -1)
    summary = {"documents_per_split": args.documents, "seed": args.seed, "pool": len(pool_texts), "usable_pool": len(usable),
               "min_tokens": args.min_tokens, "max_tokens": args.max_tokens, "test_mean_tokens": float(tl.mean()),
               "train_mean_tokens": float(rl.mean()), "max_abs_length_difference": int(np.abs(tl - rl).max()),
               "md5": hashlib.md5(Path(args.out).read_bytes()).hexdigest()}
    Path(args.summary).write_text(json.dumps(summary, indent=1) + "\n")
    print(f"MEMSAMPLE {json.dumps(summary)}", flush=True)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--manifest", default=f"{SOURCE}/manifest.json")
    p.add_argument("--source", default=SOURCE, help="the directory the manifest's paths are relative to")
    p.add_argument("--hosts", type=int, default=4)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_plan)
    s = sub.add_parser("stage")
    s.add_argument("--plan", required=True)
    s.add_argument("--rank", type=int, required=True)
    s.add_argument("--out", required=True, help="directory of the share's files")
    s.add_argument("--list", required=True, help="the share's paths, for --data-files-list")
    s.add_argument("--download-dir", required=True, help="where a source file waits while its rows are taken (/dev/shm)")
    s.add_argument("--workers", type=int, default=8, help="source files at a time")
    s.add_argument("--files", default=None, help="only these source files of the plan (one per line), e.g. for a smoke test")
    s.set_defaults(func=cmd_stage)
    h = sub.add_parser("heldout")
    h.add_argument("--plan", required=True)
    h.add_argument("--overlap", nargs="+", required=True, help="overlap.json of every host")
    h.add_argument("--download-dir", required=True)
    h.add_argument("--out", required=True)
    h.add_argument("--summary", required=True)
    h.set_defaults(func=cmd_heldout)
    m = sub.add_parser("memsample")
    m.add_argument("--heldout", required=True)
    m.add_argument("--files-list", required=True, help="this host's --data-files-list")
    m.add_argument("--tokenizer", default="yhavinga/dutch-llama-tokenizer")
    m.add_argument("--documents", type=int, default=1000)
    m.add_argument("--pool", type=int, default=50_000)
    m.add_argument("--min-tokens", type=int, default=64)
    m.add_argument("--max-tokens", type=int, default=1022)
    m.add_argument("--seed", type=int, default=42)
    m.add_argument("--out", required=True)
    m.add_argument("--summary", required=True)
    m.set_defaults(func=cmd_memsample)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
