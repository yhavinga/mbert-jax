"""Which documents of a host's staged files the trainer's --url-blocklist leaves out, counted before the run.

Reads only the column `url` of each file in --files-list (the trainer's --data-files-list) and applies
src/mmbjax/training/url_blocklist.py, the function the trainer's loader applies. Writes per file its rows and the
row numbers of the blocked documents, and over the host the documents per blocked host name and some example URLs.
The row numbers are the plan's input for the mix of CulturaX and GPT-NL (gptnl_mix.py plan --exclude-culturax).

Imports no JAX and opens no TPU: it may run next to a trainer, at nice 19.

  python experiments/final/blocklist_scan.py --files-list ~/culturax_nl_files.txt --blocklist hebban.nl \\
      --out ~/final_blocklist_w0.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq

# url_blocklist.py by its path: importing the mmbjax package would import JAX
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src" / "mmbjax" / "training"))

import url_blocklist  # noqa: E402

CULTURAX_FILE = re.compile(r"nl_part_(\d{5})\.parquet$")
EXAMPLES = 20


def scan_file(path: str, blocked: tuple) -> dict:
    """Rows of the file, and the row numbers (in file order) whose URL is blocked, with their hosts."""
    parquet = pq.ParquetFile(path)
    if "url" not in parquet.schema_arrow.names:
        raise ValueError(f"{path} has no column url, only {parquet.schema_arrow.names}")
    rows, blocked_rows, hosts, examples = 0, [], Counter(), []
    for batch in parquet.iter_batches(batch_size=65536, columns=["url"]):
        urls = batch.column(0)
        # Every blocked URL has a blocked host name in it; only those candidates go through the exact test
        candidates = pc.match_substring(urls, blocked[0], ignore_case=True)
        for host in blocked[1:]:
            candidates = pc.or_(candidates, pc.match_substring(urls, host, ignore_case=True))
        for index in pc.indices_nonzero(pc.fill_null(candidates, False)).to_pylist():
            url = urls[index].as_py()
            if url_blocklist.url_is_blocked(url, blocked):
                blocked_rows.append(rows + index)
                hosts[url_blocklist.url_host(url)] += 1
                if len(examples) < EXAMPLES:
                    examples.append(url)
        rows += len(urls)
    if rows != parquet.metadata.num_rows:
        raise RuntimeError(f"{path}: read {rows} rows, its metadata says {parquet.metadata.num_rows}")
    match = CULTURAX_FILE.search(path)
    return {"path": path, "culturax_file": int(match.group(1)) if match else None, "rows": rows,
            "blocked": len(blocked_rows), "blocked_rows": blocked_rows, "hosts": dict(hosts), "examples": examples}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--files-list", required=True, help="the trainer's --data-files-list of this host")
    parser.add_argument("--blocklist", required=True, help="comma-separated host names, as the trainer's --url-blocklist")
    parser.add_argument("--out", required=True, help="JSON with the counts and the blocked rows per file")
    args = parser.parse_args()

    blocked = url_blocklist.parse_blocklist(args.blocklist)
    files = Path(args.files_list).read_text().split()
    start = time.time()
    scanned = []
    for path in files:
        scanned.append(scan_file(path, blocked))
        print(f"{path}: {scanned[-1]['rows']:,} rows, {scanned[-1]['blocked']:,} blocked", flush=True)
    hosts = Counter()
    for f in scanned:
        hosts.update(f["hosts"])
    summary = {
        "blocklist": list(blocked),
        "files_list": args.files_list,
        "files": scanned,
        "documents": sum(f["rows"] for f in scanned),
        "blocked": sum(f["blocked"] for f in scanned),
        "hosts": dict(hosts.most_common()),
        "examples": [url for f in scanned for url in f["examples"]][:EXAMPLES],
        "seconds": round(time.time() - start, 1),
    }
    Path(args.out).write_text(json.dumps(summary, indent=1))
    print(f"{summary['blocked']:,} of {summary['documents']:,} documents blocked in {len(files)} files, "
          f"by host {summary['hosts']}", flush=True)


if __name__ == "__main__":
    main()
