"""The data of culturax-gptnl-nl-230m-chunk1024 (protocol.md): per host a share of 30% GPT-NL Public Corpus and 70%
CulturaX nl in tokens, interleaved document by document, for scripts/train_minimal.py --data-files-list.

GPT-NL: only the collections that are originally Dutch by provenance (INCLUDED), and of those only the documents whose
`language` is nl. Left out besides: empty documents, and exact duplicates (xxh64 of the text) within GPT-NL and of a
CulturaX nl document, so that no text is in the shares twice. A seeded random sample of the rest, no reweighting
between collections. CulturaX: each host's own 32 files of culturax-nl-230m-chunk1024 (CULTURAX_FILES), so a host's
CulturaX documents are some of those the same host had in that run, except the documents of the trainer's held-out
file (--eval-file, the first 40,000 of nl_part_00097), which are not trained on.

  count  On a worker: its part of the included GPT-NL files (the largest first, each to the host with the fewest bytes),
         read by range requests. Per document, in the file's order: its tokens (column `text`, as the lengths parquets
         of count_tokens.py on branch token-counts), language, license, UTF-8 bytes and xxh64; one lengths parquet per
         source file, uploaded to --gcs. A file already there is skipped. First the token counts of this script are
         compared with count_tokens.py's on documents of CulturaX.
  plan   Where all lengths are: the documents of every share, drawn with --seed. The counts to --plan, to be committed
         as cxgptnl_plan.json next to this script, the rows per host to --out-dir as cxgptnl-w<rank>.npz (to the bucket;
         the plan has their sha256).
  mix    On a worker: this host's documents, read from the Hugging Face files by range requests into --spill (on
         /dev/shm) in the order of their position in the share, then written as SHARE_FILES parquet files with the
         column `text` under --out, and the list for --data-files-list. A spill or a share file already written with the
         planned rows is kept, so a second start continues.

The tokenizer is loaded and used as the trainer does, without importing mmbjax, which imports JAX: the TPU of the VM
trains meanwhile, so nothing here may open it.

  python experiments/gptnl/gptnl_mix.py count --rank 0 --work /dev/shm/gptnl_work/lengths \\
      --gcs gs://t5-dutch-english-usc2/mbert/gptnl/lengths --check-lengths /dev/shm/gptnl_work/nl_part_00000.parquet
  python experiments/gptnl/gptnl_mix.py plan --gptnl-lengths ... --culturax-counts ... --out-dir ...
  python experiments/gptnl/gptnl_mix.py mix --rank 0 --selection .../cxgptnl-w0.npz --spill /dev/shm/gptnl_work/spill \\
      --out ~/cxgptnl_nl --list ~/cxgptnl_nl_files.txt
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import multiprocessing
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import stage_share  # noqa: E402

PLAN_PATH = Path(__file__).resolve().parent / "cxgptnl_plan.json"

GPTNL = {"repo": "GPT-NL/GPT-NL_Public_Corpus", "revision": "117b13ee7246aa5c83a42ff77e15d4d50b4156b9"}
# Originally Dutch by provenance: collected by GPT-NL from Dutch (and Flemish) institutions. KB and the archives stay
# in, texts before 1950 included.
INCLUDED = (
    "openraadsinformatie-part1", "openraadsinformatie-part2", "officiele-bekendmakingen", "woogle", "tweedekamer", "dpc",
    "pbl", "auditdienstrijk", "rechtspraak", "nationaal-archief", "utrechts-archief", "noordhollandsarchief",
    "zeeuws-archief", "kb", "kb-pd-books", "kb-open-kranten", "dansknaw", "naturalis", "wikiwijs", "belgian-journal",
)
# Translated (youtube-commons), synthetic (wikidata), translations of the EU (european-parliament, multi-eurlex,
# cc_eurovoc), web of unknown provenance (common-crawl), and English, German or code (the rest)
EXCLUDED = (
    "youtube-commons", "wikidata", "european-parliament", "multi-eurlex", "cc_eurovoc", "common-crawl", "cc_openalex",
    "cc_english-pd", "cc_german-pd", "cc_loc-pd-books", "american-stories", "cc_github_open_source",
)
DUTCH = "nl"

# The revision of culturax-nl-230m-chunk1024's staging (stage.py) and of count_tokens.py
CULTURAX = {"repo": "uonlp/CulturaX", "revision": "6a8734bc69fefcbb7735f4f9250f43e4cd7a442e", "files": 128}
# Per worker number the CulturaX files (nl/nl_part_<n>.parquet) of culturax-nl-230m-chunk1024, as stage.py split them:
# split_dataset_by_node by the worker number of the hostname (recomputed so on 2026-11-08, as count_tokens.py's
# long_run_file_order does). They are the 4 sets of data_files of its data states at step 300,000
# (gs://t5-dutch-english-usc2/mbert/culturax-nl-230m-chunk1024/ckpt_data_state/300000/), whose process indices were
# other worker numbers by then: process_<k>.json has the files of worker (k + 2) % 4
CULTURAX_FILES = (
    (6, 9, 14, 16, 26, 29, 33, 34, 35, 39, 40, 43, 56, 57, 63, 68, 70, 83, 84, 88, 92, 97, 100, 101, 105, 109, 111, 112,
     116, 121, 124, 125),
    (2, 7, 10, 12, 13, 15, 24, 25, 28, 30, 41, 42, 44, 62, 64, 65, 69, 73, 78, 82, 86, 90, 91, 93, 94, 96, 102, 104, 108,
     115, 126, 127),
    (0, 1, 4, 20, 21, 22, 23, 31, 32, 38, 45, 46, 47, 48, 51, 53, 54, 55, 58, 59, 66, 71, 75, 77, 81, 85, 98, 106, 110,
     114, 118, 119),
    (3, 5, 8, 11, 17, 18, 19, 27, 36, 37, 49, 50, 52, 60, 61, 67, 72, 74, 76, 79, 80, 87, 89, 95, 99, 103, 107, 113, 117,
     120, 122, 123),
)
# Their documents per host, long_run_hosts of gs://t5-dutch-english-usc2/mbert/token-counts/summary.json (from
# stage.py's split by worker number): the plan checks the split against them
CULTURAX_HOST_DOCUMENTS = (29_348_172, 29_348_135, 29_348_224, 29_348_135)

TOKENIZER = "yhavinga/dutch-llama-tokenizer"
MAX_LENGTH = 1024
HOSTS = 4
# Enough for 330,000 steps, 300,000 and 10% to spare: 128 rows of 1024 places per step, 94.6% of them real tokens
# (CulturaX nl's packing fill, 0.9464 in token-counts/summary.json), over the hosts. Tokens as the lengths parquets
# count them, without <s> and </s> (with them 0.3% more for CulturaX)
STEPS = 330_000
HOST_TOKENS = STEPS * 128 * MAX_LENGTH * 946 // (1000 * HOSTS)
GPTNL_PERCENT = 30
# Files per share: ~200,000 documents and ~650 MB of text each, so a share file is one string array in memory
SHARE_FILES = 64
# The spill is read back and rewritten soon: fast compression, large strings so a source file's rows are one array
SPILL_ZSTD_LEVEL = 1

# Of one tokenizer call: a KB book has up to 2.7 million characters
CHUNK_DOCS = 256
CHUNK_CHARS = 4_000_000


def load_tokenizer(name: str = TOKENIZER):
    """As train_minimal.py's main() and count_tokens.py load it: split_special_tokens, and a <mask> token added if the
    tokenizer has none."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(name, split_special_tokens=True)
    if tokenizer.mask_token is None:
        tokenizer.add_special_tokens({"mask_token": "<mask>"})
    return tokenizer


def token_counts(tokenizer, texts: list[str]) -> np.ndarray:
    """The tokens of each text without <s> and </s>, as TokenizeTransform._tokenize (src/mmbjax/training/packing.py)
    makes them: a literal <s> in a document is text."""
    counts = np.empty(len(texts), dtype=np.int64)
    start = 0
    while start < len(texts):
        stop, chars = start + 1, len(texts[start])
        while stop < len(texts) and stop - start < CHUNK_DOCS and chars + len(texts[stop]) <= CHUNK_CHARS:
            chars += len(texts[stop])
            stop += 1
        ids = tokenizer(texts[start:stop], truncation=False, return_tensors=None, add_special_tokens=False,
                        split_special_tokens=True)["input_ids"]
        counts[start:stop] = [len(x) for x in ids]
        start = stop
    return counts


def tokens_with_specials(tokens: np.ndarray, max_length: int = MAX_LENGTH) -> int:
    """What the trainer reads of these documents: <s> per piece of max_length and one </s> (count_tokens.py,
    piece_lengths)."""
    n = np.asarray(tokens, dtype=np.int64)
    room = max_length - 1
    full = np.where(n >= max_length, (n - max_length) // room + 1, 0)
    rest = n - full * room
    pieces = full + np.where(rest == room, 2, 1)
    return int((n + pieces + 1).sum())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def culturax_path(index: int) -> str:
    return f"nl/nl_part_{index:05d}.parquet"


def share_file_name(index: int, rank: int, hosts: int) -> str:
    # The rank is in the name, as in stage_share.share_file_name: a data state's identity is its list of paths
    return f"cxgptnl_{index:03d}-w{rank}of{hosts}.parquet"


# ------------------------------------------------------------------------------------------------ count


def gptnl_files() -> list[tuple[str, int]]:
    """The included collections' parquet files at GPTNL's revision, with their sizes, sorted by path. Refuses a repo whose
    collections are not those of INCLUDED and EXCLUDED."""
    from huggingface_hub import HfApi

    items = HfApi().list_repo_tree(GPTNL["repo"], repo_type="dataset", revision=GPTNL["revision"], recursive=True)
    # Folders have no size
    files = {item.path: item.size for item in items if getattr(item, "size", None) is not None}
    folders = {path.split("/")[0] for path in files if "/" in path}
    if folders != set(INCLUDED) | set(EXCLUDED):
        raise ValueError(f"{GPTNL['repo']}@{GPTNL['revision']} has the collections {sorted(folders)}: "
                         f"not in INCLUDED or EXCLUDED {sorted(folders - set(INCLUDED) - set(EXCLUDED))}, "
                         f"missing {sorted((set(INCLUDED) | set(EXCLUDED)) - folders)}")
    chosen = sorted((path, size) for path, size in files.items()
                    if path.split("/")[0] in INCLUDED and path.endswith(".parquet"))
    deeper = [path for path, _ in chosen if path.count("/") != 1]
    if deeper:
        raise ValueError(f"parquet files below a collection's folder: {deeper[:3]}")
    stems = [stage_share.output_stem(path) for path, _ in chosen]
    if len(set(stems)) != len(stems):
        raise ValueError("two included files have the same name, which the lengths parquets need to be unique")
    return chosen


def assign_files(sizes: list[int], hosts: int) -> list[int]:
    """The host that counts each file: the largest first, each to the host with the fewest bytes so far."""
    load = [0] * hosts
    host_of = [0] * len(sizes)
    for i in sorted(range(len(sizes)), key=lambda i: -sizes[i]):
        k = min(range(hosts), key=lambda h: (load[h], h))
        host_of[i] = k
        load[k] += sizes[i]
    return host_of


_TOKENIZER = None


def _init_counter(name: str) -> None:
    global _TOKENIZER
    _TOKENIZER = load_tokenizer(name)


def count_file(path: str, out: str) -> tuple[str, int, int, float]:
    """Per document of one GPT-NL file, in its order: tokens (column `text`), language, license, UTF-8 bytes and
    xxh64, written under a temporary name first. Returns (out, rows, tokens, seconds)."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    import xxhash

    t = time.time()
    source = pq.ParquetFile(stage_share.open_source(GPTNL["repo"], GPTNL["revision"], path))
    parts = []
    for i in range(source.metadata.num_row_groups):
        table = source.read_row_group(i, columns=["text", "language", "license"])
        if table.column("text").null_count:
            raise ValueError(f"{path}: {table.column('text').null_count} rows without a text in row group {i}")
        texts = table.column("text").to_pylist()
        parts.append(pa.table({
            "text": pa.array(token_counts(_TOKENIZER, texts), pa.int32()),
            "language": table.column("language").cast(pa.string()),
            "license": table.column("license").cast(pa.string()),
            "utf8_bytes": pc.binary_length(table.column("text")).cast(pa.int64()),
            "xxh64": pa.array([xxhash.xxh64_intdigest(x.encode("utf-8")) for x in texts], pa.uint64()),
        }))
    table = pa.concat_tables(parts)
    if table.num_rows != source.metadata.num_rows:
        raise ValueError(f"{path}: {table.num_rows} rows counted, the file has {source.metadata.num_rows}")
    partial = Path(out + ".partial")
    pq.write_table(table, partial)
    os.replace(partial, out)
    return out, table.num_rows, int(pc.sum(table.column("text")).as_py() or 0), time.time() - t


def check_tokenizer(tokenizer, lengths: str, documents: int = 2000, seed: int = 0) -> None:
    """The token counts of this script against count_tokens.py's, which used the trainer's own TokenizeTransform: a
    seeded sample of the first row group of CulturaX nl_part_00000 against its lengths parquet."""
    import pyarrow.parquet as pq

    path = culturax_path(0)
    expected = pq.read_table(lengths, columns=["text"]).column("text").to_numpy()
    source = pq.ParquetFile(stage_share.open_source(CULTURAX["repo"], CULTURAX["revision"], path))
    if source.metadata.num_rows != len(expected):
        raise ValueError(f"{path} has {source.metadata.num_rows} rows, {lengths} {len(expected)}")
    n = source.metadata.row_group(0).num_rows
    picks = np.sort(np.random.default_rng(seed).choice(n, size=min(documents, n), replace=False))
    texts = source.read_row_group(0, columns=["text"]).column("text").take(picks).to_pylist()
    got = token_counts(tokenizer, texts)
    wrong = np.flatnonzero(got != expected[picks])
    if len(wrong):
        raise AssertionError(f"{len(wrong)} of {len(picks)} documents of {path} have other token counts than "
                             f"count_tokens.py's, e.g. row {picks[wrong[0]]}: {got[wrong[0]]} against {expected[picks[wrong[0]]]}")
    print(f"tokenizer checked: {len(picks)} documents of {path}, {int(got.sum()):,} tokens, as count_tokens.py "
          f"counted them", flush=True)


def _gcs_stems(gcs: str) -> set[str]:
    listing = subprocess.run(["gsutil", "ls", f"{gcs.rstrip('/')}/*.parquet"], capture_output=True, text=True)
    if listing.returncode != 0:
        if "matched no objects" in listing.stderr:
            return set()
        raise RuntimeError(f"gsutil ls {gcs} exited {listing.returncode}: {listing.stderr[-2000:]}")
    return {url.rsplit("/", 1)[1][: -len(".parquet")] for url in listing.stdout.split()}


def _upload(local: str, gcs: str) -> None:
    subprocess.run(["gsutil", "-q", "cp", local, f"{gcs.rstrip('/')}/{Path(local).name}"], check=True)


def cmd_count(args) -> None:
    files = gptnl_files()
    host_of = assign_files([size for _, size in files], args.hosts)
    mine = [path for (path, _), k in zip(files, host_of) if k == args.rank]
    done = _gcs_stems(args.gcs)
    todo = [path for path in mine if stage_share.output_stem(path) not in done][:args.max_files]
    work = Path(args.work).expanduser()
    work.mkdir(parents=True, exist_ok=True)
    print(f"rank {args.rank} of {args.hosts}: {len(mine)} of {len(files)} included files "
          f"({sum(s for (_, s), k in zip(files, host_of) if k == args.rank) / 1e9:.1f} GB), {len(todo)} to count, "
          f"{args.workers} at a time", flush=True)
    check_tokenizer(load_tokenizer(args.tokenizer), args.check_lengths)
    t = time.time()
    with cf.ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("spawn"),
                                initializer=_init_counter, initargs=(args.tokenizer,)) as pool:
        futures = [pool.submit(count_file, path, str(work / f"{stage_share.output_stem(path)}.parquet")) for path in todo]
        try:
            for done_files, future in enumerate(cf.as_completed(futures), 1):
                out, rows, tokens, seconds = future.result()
                _upload(out, args.gcs)
                print(f"[{done_files}/{len(todo)}] {Path(out).stem}: {rows:,} documents, {tokens:,} tokens, "
                      f"{seconds:.0f} s", flush=True)
        except BaseException:
            pool.shutdown(cancel_futures=True)
            raise
    print(f"COUNTED {len(todo)} files in {time.time() - t:.0f} s", flush=True)


# ------------------------------------------------------------------------------------------------- plan


def take_until(tokens: np.ndarray, target: int) -> int:
    """How many of the first documents it takes to reach target tokens."""
    total = np.cumsum(tokens, dtype=np.int64)
    if not len(total) or total[-1] < target:
        raise ValueError(f"{int(total[-1]) if len(total) else 0:,} tokens, fewer than the {target:,} wanted")
    return int(np.searchsorted(total, target)) + 1


def eligible_documents(dutch: np.ndarray, tokens: np.ndarray, hashes: np.ndarray, culturax_hashes: np.ndarray):
    """Which documents may be drawn: Dutch, not empty, the first of their text (in the given order) and no text of
    CulturaX. Returns (mask, what each rule left out)."""
    candidate = dutch & (tokens > 0)
    index = np.flatnonzero(candidate)
    _, first = np.unique(hashes[index], return_index=True)
    unique = np.zeros(len(tokens), dtype=bool)
    unique[index[first]] = True
    in_culturax = unique & np.isin(hashes, culturax_hashes)
    mask = unique & ~in_culturax
    dropped = {"not_dutch": ~dutch, "empty": dutch & (tokens == 0), "duplicate": candidate & ~unique,
               "in_culturax": in_culturax}
    return mask, dropped


def gptnl_selection(tokens: np.ndarray, seed: int, hosts: int, target: int) -> list[np.ndarray]:
    """Per host the indices of its documents among the eligible ones, sorted: a permutation drawn with seed is dealt to
    the hosts in turn, and each takes its documents until it has target tokens. No document is on two hosts."""
    order = np.random.default_rng([seed, 0]).permutation(len(tokens))
    picks = []
    for k in range(hosts):
        dealt = order[k::hosts]
        picks.append(np.sort(dealt[:take_until(tokens[dealt], target)]))
    return picks


def culturax_selection(seed: int, files: tuple[int, ...], tokens: list[np.ndarray], target: int, allowed=None):
    """A host's CulturaX documents: the rows of its files in the order of a draw per row of each whole file
    (default_rng([seed, 1, file]).random(rows)), taken until target tokens; rows not allowed are passed over.
    Returns (file, row), sorted."""
    if list(files) != sorted(set(files)):
        raise ValueError(f"the files {files} are not sorted and distinct")
    draws = np.concatenate([np.random.default_rng([seed, 1, f]).random(len(t)) for f, t in zip(files, tokens)])
    order = np.argsort(draws, kind="stable")
    if allowed is not None:
        order = order[np.concatenate(allowed)[order]]
    chosen = np.sort(order[:take_until(np.concatenate(tokens)[order], target)])
    offsets = np.cumsum([0] + [len(t) for t in tokens])
    file_of = np.searchsorted(offsets, chosen, side="right") - 1
    return np.asarray(files, dtype=np.int32)[file_of], (chosen - offsets[file_of]).astype(np.int32)


def positions(seed: int, rank: int, documents: int) -> np.ndarray:
    """Each document's place in the share, its GPT-NL documents first and then its CulturaX ones, each in (file, row)
    order: a permutation drawn with seed, so the two are interleaved document by document."""
    return np.random.default_rng([seed, 2, rank]).permutation(documents)


def _counts(selected, tokens) -> dict:
    return {"documents": int(selected.sum()), "tokens": int(tokens[selected].sum())}


def _codes(column) -> tuple[list[str], np.ndarray]:
    """The distinct values of a string column and each row's index among them, -1 for null."""
    encoded = column.combine_chunks().dictionary_encode()
    return encoded.dictionary.to_pylist(), encoded.indices.fill_null(-1).to_numpy(zero_copy_only=False).astype(np.int64)


def cmd_plan(args) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    import xxhash

    seed, hosts = args.seed, HOSTS
    files = gptnl_files()
    lengths_dir = Path(args.gptnl_lengths).expanduser()
    missing = [path for path, _ in files if not (lengths_dir / f"{stage_share.output_stem(path)}.parquet").exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} included files have no lengths parquet in {lengths_dir}, e.g. {missing[:3]}")
    tables = [pq.read_table(lengths_dir / f"{stage_share.output_stem(path)}.parquet") for path, _ in files]
    rows = np.array([t.num_rows for t in tables], dtype=np.int64)
    file_index = np.repeat(np.arange(len(files), dtype=np.int32), rows)
    row_index = np.concatenate([np.arange(n, dtype=np.int32) for n in rows])
    table = pa.concat_tables(tables)
    tokens = table.column("text").to_numpy().astype(np.int64)
    utf8_bytes = table.column("utf8_bytes").to_numpy()
    hashes = table.column("xxh64").to_numpy()
    languages, language = _codes(table.column("language"))
    dutch = language == (languages.index(DUTCH) if DUTCH in languages else -2)
    collection = np.array([path.split("/")[0] for path, _ in files])[file_index]

    counts_dir = Path(args.culturax_counts).expanduser()
    culturax = []
    for f in range(CULTURAX["files"]):
        saved = np.load(counts_dir / f"nl_part_{f:05d}.npz")
        culturax.append({"tokens": saved["counts"].astype(np.int64), "utf8_bytes": saved["utf8_bytes"].astype(np.int64),
                         "xxh64": saved["xxh64"]})
    if sorted(f for host in CULTURAX_FILES for f in host) != list(range(CULTURAX["files"])):
        raise ValueError("CULTURAX_FILES is not a split of the CulturaX files")
    host_documents = tuple(sum(len(culturax[f]["tokens"]) for f in host) for host in CULTURAX_FILES)
    if host_documents != CULTURAX_HOST_DOCUMENTS:
        raise ValueError(f"the CulturaX files of CULTURAX_FILES hold {host_documents} documents per host, the long run's "
                         f"hosts {CULTURAX_HOST_DOCUMENTS}")
    culturax_hashes = np.concatenate([c["xxh64"] for c in culturax])
    # The trainer's --eval-file is CulturaX documents: none of them goes into training
    heldout = np.array([xxhash.xxh64_intdigest(t.encode("utf-8"))
                        for t in pq.read_table(args.heldout, columns=["text"]).column("text").to_pylist()], dtype=np.uint64)
    allowed = [~np.isin(c["xxh64"], heldout) for c in culturax]
    print(f"GPT-NL: {len(files)} files, {len(tokens):,} documents, {int(tokens.sum()):,} tokens; CulturaX: "
          f"{len(culturax_hashes):,} documents, {sum(int((~a).sum()) for a in allowed):,} of them in the "
          f"{len(heldout):,} of {args.heldout}", flush=True)

    eligible, dropped = eligible_documents(dutch, tokens, hashes, culturax_hashes)
    candidates = np.flatnonzero(eligible)
    gptnl_target = HOST_TOKENS * GPTNL_PERCENT // 100
    picks = [candidates[p] for p in gptnl_selection(tokens[candidates], seed, hosts, gptnl_target)]
    selected_host = np.full(len(tokens), -1, dtype=np.int8)
    for k, p in enumerate(picks):
        selected_host[p] = k

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    shares = []
    for k, p in enumerate(picks):
        gptnl_tokens = int(tokens[p].sum())
        # CulturaX tokens for exactly GPTNL_PERCENT of GPT-NL in the share
        culturax_target = -(-gptnl_tokens * (100 - GPTNL_PERCENT) // GPTNL_PERCENT)
        host_files = CULTURAX_FILES[k]
        cx_file, cx_row = culturax_selection(seed, host_files, [culturax[f]["tokens"] for f in host_files],
                                             culturax_target, [allowed[f] for f in host_files])
        cx_tokens = np.concatenate([culturax[f]["tokens"][cx_row[cx_file == f]] for f in host_files])
        cx_bytes = np.concatenate([culturax[f]["utf8_bytes"][cx_row[cx_file == f]] for f in host_files])
        documents = len(p) + len(cx_row)
        selection = out_dir / f"cxgptnl-w{k}.npz"
        np.savez(selection, gptnl_file=file_index[p], gptnl_row=row_index[p], gptnl_tokens=tokens[p].astype(np.int32),
                 gptnl_bytes=utf8_bytes[p].astype(np.int64), cx_file=cx_file, cx_row=cx_row,
                 cx_tokens=cx_tokens.astype(np.int32), cx_bytes=cx_bytes, pos=positions(seed, k, documents))
        everything = np.concatenate([tokens[p], cx_tokens])
        by_collection = {c: {"documents": int((collection[p] == c).sum()), "tokens": int(tokens[p][collection[p] == c].sum())}
                         for c in INCLUDED}
        shares.append({
            "rank": k, "documents": documents, "tokens": int(everything.sum()),
            "tokens_with_specials": tokens_with_specials(everything),
            "gptnl": {"documents": len(p), "tokens": gptnl_tokens, "utf8_bytes": int(utf8_bytes[p].sum()),
                      "by_collection": by_collection},
            "culturax": {"documents": len(cx_row), "tokens": int(cx_tokens.sum()), "utf8_bytes": int(cx_bytes.sum()),
                         "of_documents": host_documents[k], "of_tokens": int(sum(culturax[f]["tokens"].sum() for f in host_files)),
                         "heldout_passed_over": sum(int((~allowed[f]).sum()) for f in host_files),
                         "files": [culturax_path(f) for f in host_files]},
            "gptnl_token_percent": round(100 * gptnl_tokens / int(everything.sum()), 4),
            "selection": selection.name, "sha256": sha256(selection),
        })
        print(f"share {k}: {len(p):,} GPT-NL documents ({gptnl_tokens:,} tokens) and {len(cx_row):,} CulturaX "
              f"({int(cx_tokens.sum()):,}), {shares[-1]['gptnl_token_percent']}% GPT-NL", flush=True)

    collections = {}
    for c in INCLUDED:
        here = collection == c
        collections[c] = {
            "files": sum(path.split("/")[0] == c for path, _ in files), **_counts(here, tokens),
            "languages": {("null" if v < 0 else languages[v]): _counts(here & (language == v), tokens)
                          for v in np.unique(language[here])},
            "dropped": {rule: _counts(here & mask, tokens) for rule, mask in dropped.items()},
            "eligible": _counts(here & eligible, tokens), "selected": _counts(here & (selected_host >= 0), tokens),
        }
    licenses, license_code = _codes(table.column("license"))
    values, times = np.unique(license_code[selected_host >= 0], return_counts=True)
    plan = {
        "run": "culturax-gptnl-nl-230m-chunk1024", "seed": seed, "hosts": hosts, "steps": STEPS,
        "host_tokens": HOST_TOKENS, "gptnl_percent": GPTNL_PERCENT, "share_files": SHARE_FILES,
        "tokens_note": "without <s> and </s>, of yhavinga/dutch-llama-tokenizer as the trainer tokenizes (count_tokens.py "
                       "for CulturaX, gptnl_mix.py count for GPT-NL); tokens_with_specials as the trainer reads them",
        "dataset_num_rows": sum(s["documents"] for s in shares),
        "gptnl": {**GPTNL, "included": list(INCLUDED), "excluded": list(EXCLUDED), "documents": len(tokens),
                  "tokens": int(tokens.sum()), "eligible": _counts(eligible, tokens),
                  "dropped": {rule: _counts(mask, tokens) for rule, mask in dropped.items()},
                  "selected": _counts(selected_host >= 0, tokens),
                  "selected_licenses": {("null" if v < 0 else licenses[v]): int(n) for v, n in zip(values, times)},
                  "files": [{"path": path, "rows": int(n), "tokens": int(t.column("text").to_numpy().sum())}
                            for (path, _), n, t in zip(files, rows, tables)]},
        "culturax": {**CULTURAX, "rows": [len(c["tokens"]) for c in culturax], "files_per_host": [list(h) for h in CULTURAX_FILES]},
        "collections": collections,
        "shares": shares,
    }
    Path(args.plan).expanduser().write_text(json.dumps(plan, indent=1) + "\n")
    totals = [s["tokens"] for s in shares]
    print(f"token spread {(max(totals) - min(totals)) / (sum(totals) / len(totals)) * 100:.4f}%; "
          f"{plan['dataset_num_rows']:,} documents; wrote {args.plan} and {out_dir}/cxgptnl-w*.npz", flush=True)


# -------------------------------------------------------------------------------------------------- mix


def spill_rows(repo: str, revision: str, path: str, rows: int, wanted, nbytes, pos, per_file: int, dutch: bool,
               spill: str) -> tuple[str, int, float]:
    """This host's rows of one source file, checked against the plan (the file's rows, each document's UTF-8 bytes and,
    for GPT-NL, its language), written to spill in the order of their position in the share, one row group per share
    file they go to. Returns (spill, rows, seconds)."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    target = Path(spill)
    t = time.time()
    if target.exists() and pq.ParquetFile(target).metadata.num_rows == len(wanted):
        return spill, len(wanted), time.time() - t
    source = pq.ParquetFile(stage_share.open_source(repo, revision, path))
    if source.metadata.num_rows != rows:
        raise ValueError(f"{path} has {source.metadata.num_rows} rows, the plan {rows}")
    columns = ["text", "language"] if dutch else ["text"]
    parts, offset = [], 0
    for i in range(source.metadata.num_row_groups):
        n = source.metadata.row_group(i).num_rows
        lo, hi = np.searchsorted(wanted, [offset, offset + n])
        if hi > lo:
            table = source.read_row_group(i, columns=columns)
            parts.append(table.take(pa.array(wanted[lo:hi] - offset)).cast(
                pa.schema([(name, pa.large_string()) for name in columns])))
        offset += n
    table = pa.concat_tables(parts)
    if table.num_rows != len(wanted):
        raise ValueError(f"{path}: {table.num_rows} of the plan's {len(wanted)} rows read")
    got = pc.binary_length(table.column("text")).to_numpy()
    if (got != nbytes).any():
        wrong = np.flatnonzero(got != nbytes)[0]
        raise ValueError(f"{path}: row {wanted[wrong]} has {got[wrong]} bytes, the plan {nbytes[wrong]}: the plan's rows "
                         f"are not these")
    if dutch and not pc.all(pc.fill_null(pc.equal(table.column("language"), DUTCH), False)).as_py():
        raise ValueError(f"{path}: a row of the plan's is not {DUTCH}")
    order = np.argsort(pos, kind="stable")
    texts, place = table.column("text").take(pa.array(order)), pos[order]
    edges = (np.flatnonzero(np.diff(place // per_file)) + 1).tolist()
    partial = target.with_name(target.name + ".partial")
    writer = pq.ParquetWriter(partial, pa.schema([("text", pa.large_string()), ("pos", pa.int64())]), compression="zstd",
                              compression_level=SPILL_ZSTD_LEVEL)
    for lo, hi in zip([0] + edges, edges + [len(place)]):
        writer.write_table(pa.table({"text": texts[lo:hi], "pos": pa.array(place[lo:hi])}), row_group_size=hi - lo)
    writer.close()
    written = pq.ParquetFile(partial).metadata.num_rows
    if written != len(wanted):
        raise ValueError(f"{partial}: {written} rows written, {len(wanted)} wanted")
    os.replace(partial, target)
    return spill, len(wanted), time.time() - t


def write_share_file(parts: list[tuple[str, int]], lo: int, hi: int, out: str) -> tuple[str, int, int, float]:
    """The documents at positions lo to hi of the share, from the row groups of the spills that hold them, written in
    that order with the column `text` under a temporary name first. Returns (out, rows, bytes, seconds)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    target = Path(out)
    t = time.time()
    if not (target.exists() and pq.ParquetFile(target).metadata.num_rows == hi - lo):
        table = pa.concat_tables([pq.ParquetFile(spill).read_row_group(group) for spill, group in parts])
        order = np.argsort(table.column("pos").to_numpy(), kind="stable")
        if not (table.column("pos").to_numpy()[order] == np.arange(lo, hi)).all():
            raise ValueError(f"{out}: the spills hold other positions than {lo} to {hi}")
        partial = target.with_name(target.name + ".partial")
        writer = pq.ParquetWriter(partial, pa.schema([("text", pa.string())]), compression="zstd",
                                  compression_level=stage_share.ZSTD_LEVEL)
        writer.write_table(pa.table({"text": table.column("text").take(pa.array(order)).cast(pa.string())}),
                           row_group_size=stage_share.ROWS_PER_GROUP)
        writer.close()
        written = pq.ParquetFile(partial).metadata.num_rows
        if written != hi - lo:
            raise ValueError(f"{partial}: {written} rows written, {hi - lo} wanted")
        os.replace(partial, target)
    return out, hi - lo, target.stat().st_size, time.time() - t


def _groups(starts: np.ndarray) -> list[tuple[int, int, int]]:
    """(value, start, stop) of the runs of a sorted array."""
    values, first = np.unique(starts, return_index=True)
    return [(int(v), int(a), int(b)) for v, a, b in zip(values, first, list(first[1:]) + [len(starts)])]


def cmd_mix(args) -> None:
    import pyarrow.parquet as pq

    plan = json.loads(PLAN_PATH.read_text())
    hosts = plan["hosts"]
    if not 0 <= args.rank < hosts:
        raise ValueError(f"--rank {args.rank} is not one of the plan's {hosts} hosts")
    share = plan["shares"][args.rank]
    selection = Path(args.selection).expanduser()
    if sha256(selection) != share["sha256"]:
        raise ValueError(f"{selection} is not the selection of the plan ({share['selection']}, sha256 {share['sha256']})")
    # Each item of an NpzFile is read from the file again, and a slice keeps its own copy alive: read them once
    s = dict(np.load(selection))
    pos = s["pos"]
    documents = len(pos)
    if documents != share["documents"] or not (np.sort(pos) == np.arange(documents)).all():
        raise ValueError(f"{selection}: {documents} positions, not a permutation of the plan's {share['documents']}")
    per_file = -(-documents // plan["share_files"])
    n_gptnl = len(s["gptnl_row"])
    gptnl_files = plan["gptnl"]["files"]
    sources = []  # (repo, revision, path, rows, wanted, bytes, pos, dutch)
    for f, a, b in _groups(s["gptnl_file"]):
        sources.append((plan["gptnl"]["repo"], plan["gptnl"]["revision"], gptnl_files[f]["path"], gptnl_files[f]["rows"],
                        s["gptnl_row"][a:b], s["gptnl_bytes"][a:b], pos[a:b], True))
    for f, a, b in _groups(s["cx_file"]):
        sources.append((plan["culturax"]["repo"], plan["culturax"]["revision"], culturax_path(f), plan["culturax"]["rows"][f],
                        s["cx_row"][a:b], s["cx_bytes"][a:b], pos[n_gptnl + a:n_gptnl + b], False))
    if sum(len(x[4]) for x in sources) != documents:
        raise ValueError(f"{selection}: the sources hold another number of documents than {documents}")

    spill_dir, out_dir = Path(args.spill).expanduser(), Path(args.out).expanduser()
    spill_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    spills = [str(spill_dir / f"{stage_share.output_stem(x[2])}.parquet") for x in sources]
    names = [share_file_name(j, args.rank, hosts) for j in range(plan["share_files"])]
    text_bytes = share["gptnl"]["utf8_bytes"] + share["culturax"]["utf8_bytes"]
    # zstd level 9 keeps 0.34 of FineWeb-2 nl's text (the rehydrated shares); level 1 keeps more. What a first start
    # wrote counts as written
    for directory, ratio, ours in ((out_dir, 0.36, [out_dir / n for n in names]), (spill_dir, 0.5, map(Path, spills))):
        free = shutil.disk_usage(directory).free + sum(p.stat().st_size for p in ours if p.exists())
        if free < ratio * text_bytes + 3e9:
            raise OSError(f"{directory}: {free / 1e9:.1f} GB free, {ratio * text_bytes / 1e9:.1f} GB needed and 3 to spare")
    print(f"rank {args.rank} of {hosts}: {documents:,} documents ({n_gptnl:,} GPT-NL) of {text_bytes / 1e9:.1f} GB text "
          f"from {len(sources)} files, {args.readers} read at a time; {plan['share_files']} share files of {per_file:,} "
          f"documents, {args.workers} written at a time", flush=True)
    t = time.time()
    # The CulturaX files first: the largest, and all hosts then ask /resolve/ less often at the start
    order = sorted(range(len(sources)), key=lambda i: (sources[i][7], i))
    with cf.ProcessPoolExecutor(max_workers=args.readers) as pool:
        futures = [pool.submit(spill_rows, *sources[i][:7], per_file, sources[i][7], spills[i]) for i in order]
        try:
            for done, future in enumerate(cf.as_completed(futures), 1):
                path, n, seconds = future.result()
                if done % 25 == 0 or done == len(futures) or seconds > 60:
                    print(f"[{done}/{len(futures)}] spilled {Path(path).stem}: {n:,} rows, {seconds:.0f} s", flush=True)
        except BaseException:
            pool.shutdown(cancel_futures=True)
            raise
    print(f"all {len(sources)} files read in {time.time() - t:.0f} s", flush=True)

    parts: list[list[tuple[str, int]]] = [[] for _ in range(plan["share_files"])]
    for spill, x in zip(spills, sources):
        for group, j in enumerate(np.unique(x[6] // per_file)):
            parts[j].append((spill, group))
    written = {}
    with cf.ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(write_share_file, parts[j], j * per_file, min((j + 1) * per_file, documents),
                               str(out_dir / names[j])) for j in range(plan["share_files"])]
        for done, future in enumerate(cf.as_completed(futures), 1):
            path, n, nbytes, seconds = future.result()
            written[path] = nbytes
            print(f"[{done}/{len(names)}] {Path(path).name}: {n:,} rows, {nbytes / 1e9:.2f} GB, {seconds:.0f} s", flush=True)
    paths = [str(out_dir / name) for name in names]
    rows = [pq.ParquetFile(p).metadata.num_rows for p in paths]
    if sum(rows) != documents:
        raise ValueError(f"{out_dir} holds {sum(rows)} documents, the plan {documents}")

    tokens = np.concatenate([s["gptnl_tokens"], s["cx_tokens"]]).astype(np.int64)
    is_gptnl = np.arange(documents) < n_gptnl
    by_place = np.empty(documents, dtype=np.int64)
    by_place[pos] = np.arange(documents)
    per_share_file = []
    for j, (p, n) in enumerate(zip(paths, rows)):
        here = by_place[j * per_file:j * per_file + n]
        per_share_file.append({"path": p, "rows": n, "bytes": written[p], "tokens": int(tokens[here].sum()),
                               "gptnl_documents": int(is_gptnl[here].sum()), "gptnl_tokens": int(tokens[here][is_gptnl[here]].sum())})
    percent = [100 * f["gptnl_tokens"] / f["tokens"] for f in per_share_file]
    summary = {"run": plan["run"], "rank": args.rank, "hosts": hosts, "seed": plan["seed"],
               "gptnl": {"repo": plan["gptnl"]["repo"], "revision": plan["gptnl"]["revision"], **share["gptnl"]},
               "culturax": {"repo": plan["culturax"]["repo"], "revision": plan["culturax"]["revision"], **share["culturax"]},
               "documents": documents, "tokens": share["tokens"], "tokens_with_specials": share["tokens_with_specials"],
               "gptnl_token_percent": share["gptnl_token_percent"], "selection_sha256": share["sha256"],
               "files": len(paths), "bytes": sum(written.values()),
               "gptnl_token_percent_per_file": [round(min(percent), 3), round(max(percent), 3)],
               "tokens_note": plan["tokens_note"], "seconds": round(time.time() - t), "per_file": per_share_file}
    if summary["tokens"] != int(tokens.sum()):
        raise ValueError(f"{selection} has {int(tokens.sum())} tokens, the plan {summary['tokens']}")
    (out_dir / "share.json").write_text(json.dumps(summary, indent=1) + "\n")
    Path(args.list).expanduser().write_text("\n".join(paths) + "\n")
    print(f"SHARE {json.dumps({k: v for k, v in summary.items() if k != 'per_file'})}", flush=True)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("count")
    c.add_argument("--rank", type=int, required=True, help="this host's part of the files, the worker number")
    c.add_argument("--hosts", type=int, default=HOSTS)
    c.add_argument("--work", required=True, help="directory of the lengths parquets before they are uploaded")
    c.add_argument("--gcs", required=True, help="bucket directory of the lengths parquets")
    c.add_argument("--check-lengths", required=True, help="count_tokens.py's lengths parquet of CulturaX nl_part_00000")
    c.add_argument("--tokenizer", default=TOKENIZER)
    c.add_argument("--workers", type=int, default=16, help="files counted at a time, one process each")
    c.add_argument("--max-files", type=int, default=None, help="only the first files to do, for a smoke test")
    c.set_defaults(func=cmd_count)
    p = sub.add_parser("plan")
    p.add_argument("--gptnl-lengths", required=True, help="directory of the lengths parquets of count")
    p.add_argument("--culturax-counts", required=True, help="directory of count_tokens.py's counts/*.npz of CulturaX nl")
    p.add_argument("--heldout", required=True, help="the trainer's --eval-file, whose documents are not trained on")
    p.add_argument("--out-dir", required=True, help="where the selections cxgptnl-w<rank>.npz go")
    p.add_argument("--plan", default=str(PLAN_PATH), help="where the plan goes; mix reads it at PLAN_PATH")
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=cmd_plan)
    m = sub.add_parser("mix")
    m.add_argument("--rank", type=int, required=True, help="this host's share, the worker number")
    m.add_argument("--selection", required=True, help="this host's cxgptnl-w<rank>.npz of plan")
    m.add_argument("--spill", required=True, help="directory of the rows read, by position (/dev/shm)")
    m.add_argument("--out", required=True, help="directory of the share's parquet files")
    m.add_argument("--list", required=True, help="file to write the share's paths to, for --data-files-list")
    m.add_argument("--readers", type=int, default=8, help="source files read at a time, one process each")
    m.add_argument("--workers", type=int, default=4, help="share files written at a time")
    m.set_defaults(func=cmd_mix)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
