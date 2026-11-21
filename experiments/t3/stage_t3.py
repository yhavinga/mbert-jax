"""The data of T3 (experiments/t3/protocol.md, sections 4 and 5): T2's CulturaX files on a worker, the held-out taken
out of them, the proof by hashes that it is out, and the sample of the memorization test.

  files      This worker's 32 files of T2 (the reference culturax-w<N>.tsv: path, size, sha256, in the order of its list)
             checked by size and sha256; with --fetch a missing file, or one of another size, is fetched first from
             Hugging Face (uonlp/CulturaX at T2's revision). A file that is not the reference's stays and is an error.
  keys       Two sha256 keys of every held-out document: "exact" (the text in UTF-8, as the trainer reads it) and
             "normalized" (NFC, all whitespace joined to one space, none at the ends); with the held-out file's md5 and
             one sha256 over both lists (the same on every worker, or the workers hashed another file).
  scan       Every row of every file of a list (columns text and url): its two keys, the documents of the URL blocklist
             (the trainer's function), every row whose exact or normalized key is a held-out key, and the file's
             sha256. --require-none makes a matching row an error: the proof that the list holds no held-out document.
  write      For every file of a scan with matching rows, a copy without those rows in --out-dir (all columns, the
             source's compression and row group size; the original is not touched), then the list for
             --data-files-list: the scanned list with those paths replaced, and its rows.
  memsample  The sample of the memorization test (experiments/final-v2/memtest.py): --documents documents of the held-out
             file and as many of the list's documents that the trainer trains on (not of the blocklist), matched one to
             one on token count, each between --min-tokens and --max-tokens, drawn with --seed.

CPU, disk and network only; nothing imports JAX. Deletes nothing.

  python3 experiments/t3/stage_t3.py keys --heldout ~/heldout.parquet --out keys.json
  python3 experiments/t3/stage_t3.py scan --keys keys.json --list ~/culturax_nl_files.txt --out scan.json
  python3 experiments/t3/stage_t3.py write --scan scan.json --out-dir ~/culturax_nl/t3 --list-out ~/t3_files.txt
  python3 experiments/t3/stage_t3.py scan --keys keys.json --list ~/t3_files.txt --out proof.json --require-none
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import sys
import time
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "mmbjax" / "training"))
sys.path.insert(0, str(ROOT / "experiments" / "final-v2"))
sys.path.insert(0, str(ROOT / "scripts"))

# The trainer's own functions, standard library only (a scan next to other work must not import JAX)
from url_blocklist import parse_blocklist, url_host, url_is_blocked  # noqa: E402

REPO, REVISION = "uonlp/CulturaX", "6a8734bc69fefcbb7735f4f9250f43e4cd7a442e"
HELDOUT_MD5 = "b2f579a876d06eef40c94270537bb7af"
BATCH_ROWS = 20_000


def exact_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalized_key(text: str) -> str:
    return hashlib.sha256(" ".join(unicodedata.normalize("NFC", text).split()).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(16 * 2**20):
            h.update(chunk)
    return h.hexdigest()


def file_md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(16 * 2**20):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: str, data) -> None:
    tmp = Path(f"{path}.tmp")
    tmp.write_text(json.dumps(data, indent=1) + "\n")
    os.replace(tmp, path)


# ------------------------------------------------------------------------------------------------------------- files


def cmd_files(args) -> None:
    ref = [line.split("\t") for line in Path(args.reference).read_text().splitlines() if line.strip()]
    listed = [line.strip() for line in Path(args.list).read_text().splitlines() if line.strip()]
    if any(len(r) != 3 for r in ref) or [r[0] for r in ref] != listed:
        sys.exit(f"{args.reference} is not one line of path, size and sha256 per file of {args.list}, in its order")
    if len(listed) != args.expect:
        sys.exit(f"{args.list} lists {len(listed)} files, not {args.expect}")
    home = Path(args.home)
    wrong_place = [p for p in listed if Path(p).parent != home / "culturax_nl" / "nl"]
    if wrong_place:
        sys.exit(f"{wrong_place[0]} is not under {home / 'culturax_nl' / 'nl'}: the lists of T2 name these paths")
    size_of = {r[0]: int(r[1]) for r in ref}
    sha_of = {r[0]: r[2].strip() for r in ref}
    missing = [p for p in listed if not (Path(p).is_file() and Path(p).stat().st_size == size_of[p])]
    if missing and not args.fetch:
        sys.exit(f"{len(missing)} files missing or of another size (e.g. {missing[0]}); --fetch fetches them")
    if missing:
        from huggingface_hub import hf_hub_download

        need = sum(size_of[p] for p in missing)
        free = os.statvfs(home).f_bavail * os.statvfs(home).f_frsize
        if need + 3 * 2**30 > free:
            sys.exit(f"{len(missing)} files to fetch, {need / 2**30:.1f} GiB and 3 GiB to spare; {free / 2**30:.1f} GiB free")
        for p in missing:
            t = time.time()
            got = hf_hub_download(REPO, f"nl/{Path(p).name}", repo_type="dataset", revision=REVISION,
                                  local_dir=str(home / "culturax_nl"))
            if Path(got) != Path(p):
                sys.exit(f"nl/{Path(p).name} went to {got}, not {p}")
            print(f"fetched {p} in {time.time() - t:.0f} s", flush=True)
    with cf.ThreadPoolExecutor(args.workers) as pool:
        sums = dict(zip(listed, pool.map(lambda p: file_sha256(Path(p)), listed)))
    files = [{"path": p, "size": Path(p).stat().st_size, "sha256": sums[p]} for p in listed]
    bad = [f"{f['path']}: size {f['size']} sha256 {f['sha256']}, the reference {size_of[f['path']]} {sha_of[f['path']]}"
           for f in files if f["size"] != size_of[f["path"]] or f["sha256"] != sha_of[f["path"]]]
    write_json(args.out, {"files": files, "fetched": missing, "wrong": bad})
    if bad:
        sys.exit(f"{len(bad)} of {len(files)} files are not the reference's (they stay):\n" + "\n".join(bad))
    print(f"FILES {len(files)} files with the reference's size and sha256 ({sum(f['size'] for f in files) / 2**30:.1f} GiB), "
          f"{len(missing)} fetched", flush=True)


# -------------------------------------------------------------------------------------------------------------- keys


def cmd_keys(args) -> None:
    import pyarrow.parquet as pq

    md5 = file_md5(Path(args.heldout))
    if args.md5 and md5 != args.md5:
        sys.exit(f"{args.heldout} has md5 {md5}, not {args.md5}")
    texts = pq.read_table(args.heldout, columns=["text"]).column("text").to_pylist()
    if any(t is None for t in texts):
        sys.exit(f"{args.heldout} has rows without text")
    exact = [exact_key(t) for t in texts]
    normalized = [normalized_key(t) for t in texts]
    digest = hashlib.sha256(("\n".join(exact) + "\n" + "\n".join(normalized) + "\n").encode()).hexdigest()
    write_json(args.out, {"file": str(args.heldout), "md5": md5, "rows": len(texts), "keys_sha256": digest,
                          "distinct_exact": len(set(exact)), "distinct_normalized": len(set(normalized)),
                          "exact": exact, "normalized": normalized})
    print(f"KEYS {len(texts)} held-out documents, md5 {md5}, keys sha256 {digest}", flush=True)


def load_keys(path: str):
    keys = json.loads(Path(path).read_text())
    exact, normalized = {}, {}
    for row, (e, n) in enumerate(zip(keys["exact"], keys["normalized"])):
        exact.setdefault(e, row)
        normalized.setdefault(n, row)
    return keys, exact, normalized


# -------------------------------------------------------------------------------------------------------------- scan

_KEYS = None


def _init_scan(keys_path: str, blocklist) -> None:
    global _KEYS
    _, exact, normalized = load_keys(keys_path)
    _KEYS = (exact, normalized, blocklist)


def scan_file(path: str) -> dict:
    """One file: rows, the blocklist's rows by host, the rows with a held-out key, the file's sha256."""
    import pyarrow.parquet as pq

    exact, normalized, blocklist = _KEYS
    rows = blocked = 0
    hosts, matches = {}, []
    for batch in pq.ParquetFile(path).iter_batches(batch_size=BATCH_ROWS, columns=["text", "url"]):
        for text, url in zip(batch.column("text").to_pylist(), batch.column("url").to_pylist()):
            if url_is_blocked(url, blocklist):
                blocked += 1
                host = url_host(url)
                hosts[host] = hosts.get(host, 0) + 1
            if text is not None:
                e = exact_key(text)
                if e in exact:
                    matches.append({"row": rows, "heldout_row": exact[e], "kind": "exact"})
                else:
                    n = normalized_key(text)
                    if n in normalized:
                        matches.append({"row": rows, "heldout_row": normalized[n], "kind": "normalized"})
            rows += 1
    return {"path": path, "sha256": file_sha256(Path(path)), "rows": rows, "blocked": blocked, "blocked_hosts": hosts,
            "matches": matches}


def cmd_scan(args) -> None:
    keys, _, _ = load_keys(args.keys)
    blocklist = parse_blocklist(args.url_blocklist)
    listed = [line.strip() for line in Path(args.list).read_text().splitlines() if line.strip()]
    if not listed:
        sys.exit(f"{args.list} lists no file")
    t = time.time()
    with cf.ProcessPoolExecutor(args.processes, initializer=_init_scan, initargs=(args.keys, blocklist)) as pool:
        files = list(pool.map(scan_file, listed))
    matched = sum(len(f["matches"]) for f in files)
    out = {"list": str(args.list), "list_md5": file_md5(Path(args.list)), "keys_sha256": keys["keys_sha256"],
           "heldout_md5": keys["md5"], "heldout_rows": keys["rows"], "url_blocklist": list(blocklist), "files": files,
           "rows": sum(f["rows"] for f in files), "blocked": sum(f["blocked"] for f in files), "matching_rows": matched,
           "exact_matches": sum(m["kind"] == "exact" for f in files for m in f["matches"]),
           "normalized_matches": sum(m["kind"] == "normalized" for f in files for m in f["matches"]),
           "heldout_rows_found": len({m["heldout_row"] for f in files for m in f["matches"]}), "seconds": round(time.time() - t)}
    write_json(args.out, out)
    print(f"SCAN {len(files)} files, {out['rows']} rows, {out['blocked']} of the blocklist, {matched} with a held-out key "
          f"({out['exact_matches']} exact, {out['normalized_matches']} normalized; {out['heldout_rows_found']} distinct "
          f"held-out documents) in {out['seconds']} s", flush=True)
    if args.require_none and matched:
        where = [f"{f['path']}: {len(f['matches'])}" for f in files if f["matches"]]
        sys.exit(f"the list still holds {matched} rows with a held-out key: {', '.join(where)}")


# ------------------------------------------------------------------------------------------------------------- write


def write_without(source: Path, target: Path, drop: set) -> int:
    """source without the rows in drop, to target (via a temporary file); its rows."""
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    meta = pq.ParquetFile(source).metadata
    if any(r >= meta.num_rows or r < 0 for r in drop):
        raise ValueError(f"{source} has {meta.num_rows} rows; rows to drop outside it: {sorted(r for r in drop if r >= meta.num_rows)[:5]}")
    compression = meta.row_group(0).column(0).compression.lower() if meta.num_row_groups else "snappy"
    group = meta.row_group(0).num_rows if meta.num_row_groups else 100_000
    table = pq.read_table(source)
    keep = np.ones(table.num_rows, dtype=bool)
    keep[sorted(drop)] = False
    kept = table.filter(pa.array(keep))
    tmp = target.with_name(target.name + ".tmp")
    pq.write_table(kept, tmp, compression=compression, row_group_size=group)
    os.replace(tmp, target)
    return kept.num_rows


def cmd_write(args) -> None:
    scan = json.loads(Path(args.scan).read_text())
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    replaced, paths, rows = [], [], 0
    for f in scan["files"]:
        source = Path(f["path"])
        if not f["matches"]:
            paths.append(str(source))
            rows += f["rows"]
            continue
        if file_sha256(source) != f["sha256"]:
            sys.exit(f"{source} changed since the scan (sha256)")
        target = out_dir / source.name
        if target.resolve() == source.resolve():
            sys.exit(f"--out-dir {out_dir} holds the source {source}")
        free = os.statvfs(out_dir).f_bavail * os.statvfs(out_dir).f_frsize
        if free < source.stat().st_size + 2**30:
            sys.exit(f"{free / 2**30:.1f} GiB free in {out_dir}, {source.stat().st_size / 2**30:.1f} GiB + 1 needed; nothing deleted")
        drop = {m["row"] for m in f["matches"]}
        n = write_without(source, target, drop)
        if n != f["rows"] - len(drop):
            sys.exit(f"{target}: {n} rows, {f['rows']} - {len(drop)} expected")
        replaced.append({"source": str(source), "path": str(target), "rows_before": f["rows"], "removed": len(drop),
                         "rows": n, "sha256": file_sha256(target)})
        paths.append(str(target))
        rows += n
        print(f"WROTE {target}: {f['rows']} - {len(drop)} = {n} rows", flush=True)
    tmp = Path(f"{args.list_out}.tmp")
    tmp.write_text("".join(f"{p}\n" for p in paths))
    os.replace(tmp, args.list_out)
    Path(args.list_out).with_suffix(".rows").write_text(f"{rows}\n")
    md5 = file_md5(Path(args.list_out))
    Path(args.list_out).with_suffix(".md5").write_text(f"{md5}\n")
    write_json(args.summary, {"list": str(args.list_out), "list_md5": md5, "files": len(paths), "rows": rows,
                              "replaced": replaced, "scan_list_md5": scan["list_md5"]})
    print(f"LIST {args.list_out}: {len(paths)} files, {rows} rows, {len(replaced)} replaced, md5 {md5}", flush=True)


# --------------------------------------------------------------------------------------------------------- memsample


def cmd_memsample(args) -> None:
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    from stage_optout import match_lengths
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    blocklist = parse_blocklist(args.url_blocklist)
    _, exact, normalized = load_keys(args.keys)

    def count(texts):
        return [len(x) for x in tokenizer(texts, add_special_tokens=False, split_special_tokens=True)["input_ids"]]

    rng = np.random.default_rng(args.seed)
    test_texts = pq.read_table(args.heldout, columns=["text"]).column("text").to_pylist()
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
            table = pq.read_table(path, columns=["text", "url"])
            texts, urls = table.column("text"), table.column("url")
            for r in mine:
                text, url = texts[int(r)].as_py(), urls[int(r)].as_py()
                # Only what the trainer trains on: not the blocklist's, not a held-out document
                if text is None or url_is_blocked(url, blocklist):
                    continue
                if exact_key(text) in exact or normalized_key(text) in normalized:
                    raise ValueError(f"{path} row {int(r)} is a held-out document: the list is not the proven one")
                pool_texts.append(text)
                pool_where.append(f"{Path(path).name}:{int(r)}")
    pool_lengths = count(pool_texts)
    usable = [k for k, n in enumerate(pool_lengths) if args.min_tokens <= n <= args.max_tokens]
    chosen = match_lengths([test_lengths[k] for k in test_pick], [pool_lengths[k] for k in usable])
    train_pick = [usable[c] for c in chosen]
    table = pa.table({
        "text": [test_texts[k] for k in test_pick] + [pool_texts[k] for k in train_pick],
        "split": ["test"] * len(test_pick) + ["train"] * len(train_pick),
        "source": [f"heldout:{k}" for k in test_pick] + [pool_where[k] for k in train_pick],
        "tokens": pa.array([test_lengths[k] for k in test_pick] + [pool_lengths[k] for k in train_pick], pa.int32()),
    })
    pq.write_table(table, args.out)
    tl, rl = np.array(table.column("tokens").to_pylist()).reshape(2, -1)
    summary = {"documents_per_split": args.documents, "seed": args.seed, "pool": len(pool_texts), "usable_pool": len(usable),
               "min_tokens": args.min_tokens, "max_tokens": args.max_tokens, "test_mean_tokens": float(tl.mean()),
               "train_mean_tokens": float(rl.mean()), "max_abs_length_difference": int(np.abs(tl - rl).max()),
               "md5": file_md5(Path(args.out))}
    write_json(args.summary, summary)
    print(f"MEMSAMPLE {json.dumps(summary)}", flush=True)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    f = sub.add_parser("files")
    f.add_argument("--reference", required=True, help="culturax-w<N>.tsv of T2's staging")
    f.add_argument("--list", required=True, help="culturax_nl_files-w<N>.txt of T2's staging")
    f.add_argument("--home", default=str(Path.home()))
    f.add_argument("--expect", type=int, default=32)
    f.add_argument("--fetch", action="store_true")
    f.add_argument("--workers", type=int, default=4)
    f.add_argument("--out", required=True)
    f.set_defaults(func=cmd_files)
    k = sub.add_parser("keys")
    k.add_argument("--heldout", required=True)
    k.add_argument("--md5", default=HELDOUT_MD5, help="the held-out file's md5 ('' to skip the check)")
    k.add_argument("--out", required=True)
    k.set_defaults(func=cmd_keys)
    s = sub.add_parser("scan")
    s.add_argument("--keys", required=True)
    s.add_argument("--list", required=True)
    s.add_argument("--url-blocklist", default="hebban.nl")
    s.add_argument("--processes", type=int, default=16)
    s.add_argument("--require-none", action="store_true", help="a row with a held-out key is an error")
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_scan)
    w = sub.add_parser("write")
    w.add_argument("--scan", required=True)
    w.add_argument("--out-dir", required=True)
    w.add_argument("--list-out", required=True, help="the list for --data-files-list (.rows and .md5 next to it)")
    w.add_argument("--summary", required=True)
    w.set_defaults(func=cmd_write)
    m = sub.add_parser("memsample")
    m.add_argument("--heldout", required=True)
    m.add_argument("--keys", required=True)
    m.add_argument("--files-list", required=True, help="this host's --data-files-list")
    m.add_argument("--tokenizer", default="yhavinga/dutch-llama-tokenizer")
    m.add_argument("--url-blocklist", default="hebban.nl")
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
