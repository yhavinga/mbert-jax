#!/usr/bin/env python3
"""The Dutch mix of one host for arm F-nl: F's mix with every FLAN row replaced in place by its Dutch version.

Usage (build_mix.sh, on each worker, on the CPU, with the trainer's venv and the builder of F's commit on the path):
    PYTHONPATH=~/mbert-flan/src:~/mbert-flan/scripts python build_nl_mix.py --rank 0 --english-list ~/flan_decay_files.txt \\
        --dutch /dev/shm/flan_nl/nl/w0.parquet --out ~/flan_decay_nl --list ~/flan_decay_nl_files.txt

The sequence is F's: the web rows of F's mix in F's order, byte for byte, and at the place of each FLAN row its Dutch
version, or nothing where it was dropped. The file is written the way scripts/build_decay_mix.py wrote F's (its
write_mix, trainer_shuffle_order and verify_order, imported from F's commit), so the trainer's shuffle yields exactly
this sequence. A dropped row makes the file shorter, and the shuffle of another length is another permutation, so the
order is made for the new length and checked by reading the file back through the trainer's route. A Dutch example of
more than --max-length tokens as the trainer counts them is dropped too: the trainer never splits an example.

Output: <out>/mix-<build>.parquet (F's columns, the build id in the metadata key F uses), <out>/manifest-<build>.json,
<out>/ids-<build>.parquet (Dutch position -> F's position of every FLAN row), and --list for --data-files-list.
"""

import os

# The TPU may be in use: this script must never take it
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

import flan_nl  # noqa: E402

TOKENIZE_BATCH = 2000


def read_f_mix(path):
    """F's mix in position order: [(position, text, inputs, targets, source)] and its build id."""
    import pyarrow.parquet as pq
    from build_decay_mix import METADATA_KEY

    table = pq.read_table(path, columns=["position", "text", "inputs", "targets", "source"]).sort_by("position")
    build = pq.ParquetFile(path).schema_arrow.metadata[METADATA_KEY].decode()
    columns = [table.column(c).to_pylist() for c in ("position", "text", "inputs", "targets", "source")]
    return list(zip(*columns)), build


def flan_tokens(tokenizer, inputs, targets):
    """Tokens of each example as the builder counted them for F: inputs, targets, <s>, <answer> and </s>."""
    counts = []
    for k in range(0, len(inputs), TOKENIZE_BATCH):
        x = tokenizer(inputs[k:k + TOKENIZE_BATCH], add_special_tokens=False, split_special_tokens=True)["input_ids"]
        y = tokenizer(targets[k:k + TOKENIZE_BATCH], add_special_tokens=False, split_special_tokens=True)["input_ids"]
        counts += [len(a) + len(b) + 3 for a, b in zip(x, y)]
    return counts


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def build(args, tokenizer):
    import build_decay_mix as bdm
    import pyarrow as pa
    import pyarrow.parquet as pq

    from mmbjax.training import TokenizeTransform

    started = time.time()
    english = Path(args.english_list).read_text().split()
    if len(english) != 1:
        raise SystemExit(f"{args.english_list} lists {len(english)} files, F has one mix per host")
    rows, f_build = read_f_mix(english[0])
    f_manifest = json.loads((Path(english[0]).parent / f"manifest-{f_build}.json").read_text())
    if f_manifest["rows"] != len(rows):
        raise RuntimeError(f"F's manifest has {f_manifest['rows']} rows, its mix {len(rows)}")
    source_at = {p: s for p, _, _, _, s in rows}

    table = pq.read_table(args.dutch)
    positions = table.column("position").to_pylist()
    sources = table.column("source").to_pylist()
    inputs = table.column("inputs").to_pylist()
    targets = table.column("targets").to_pylist()
    for p, s in zip(positions, sources):
        if source_at.get(p) != s:
            raise RuntimeError(f"Dutch row at position {p} is of {s}, F's row there of {source_at.get(p)}")
    counts = flan_tokens(tokenizer, inputs, targets)
    dutch, too_long = {}, Counter()
    for p, s, x, y, n in zip(positions, sources, inputs, targets, counts):
        if n > args.max_length:
            too_long[s.split("/")[1]] += 1
        else:
            dutch[p] = (x, y, n)

    sequence, web_texts, pool, sequence_counts = flan_nl.dutch_sequence(rows, {p: v[:2] for p, v in dutch.items()})
    # write_mix takes F's pool layout: (inputs, targets, source, tokens)
    pool_rows = [(x, y, s, dutch[p][2]) for x, y, s, p in pool]
    tokenize = TokenizeTransform(tokenizer, args.max_length, answer_token_id=tokenizer.convert_tokens_to_ids("<answer>"))
    for x, y, s, n in pool_rows[:2000]:
        actual = sum(len(piece["input_ids"]) for piece in tokenize.map({"text": None, "inputs": x, "targets": y}))
        if actual != n:
            raise RuntimeError(f"a Dutch example counted {n} tokens, the trainer makes {actual}: {x[:60]!r}")

    config = {
        "rank": args.rank, "f_build": f_build, "f_mix": english[0], "dutch": file_sha256(args.dutch),
        "max_length": args.max_length, "tokenizer": args.tokenizer,
        "code": {Path(module).name: hashlib.sha256(Path(module).read_bytes()).hexdigest()
                 for module in (__file__, flan_nl.__file__, bdm.__file__, sys.modules["mmbjax.training.packing"].__file__)},
    }
    build_id = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"mix-{build_id}.parquet"
    manifest_path = out / f"manifest-{build_id}.json"
    print(f"build {build_id} of rank {args.rank} from F's {f_build}: {path}", flush=True)
    if path.exists() and manifest_path.exists():
        Path(args.list).write_text(f"{path}\n")
        print(f"done before: {path}, listed in {args.list}", flush=True)
        return json.loads(manifest_path.read_text())

    order = bdm.trainer_shuffle_order(len(sequence))
    partial = path.with_suffix(".partial")
    bdm.write_mix(partial, sequence, web_texts, pool_rows, order, build_id)
    bdm.verify_order(partial, tokenizer, len(sequence))
    # The web rows of the file, in sequence order, are F's byte for byte
    written = pq.read_table(partial, columns=["position", "text", "source"]).sort_by("position")
    web_written = [t for t, s in zip(written.column("text").to_pylist(), written.column("source").to_pylist())
                   if s == bdm.WEB_SOURCE]
    if web_written != [t for _, t, _, _, s in rows if s == bdm.WEB_SOURCE]:
        raise RuntimeError(f"{partial}: the web rows differ from F's")
    partial.rename(path)
    flan_positions = [t for t, (kind, _) in enumerate(sequence) if kind == "flan"]
    pq.write_table(pa.table({"position": flan_positions, "f_position": [p for _, _, _, p in pool]}),
                   out / f"ids-{build_id}.parquet", compression="zstd")

    dutch_flan_tokens = sum(n for _, _, _, n in pool_rows)
    web_tokens = f_manifest["web"]["tokens"]
    kept_per_folder = Counter(s.split("/")[1] for _, _, s, _ in pool_rows)
    f_per_folder = Counter(s.split("/")[1] for _, _, _, _, s in rows if s != bdm.WEB_SOURCE)
    manifest = {
        "build": build_id, "file": str(path), "config": config, "seconds": round(time.time() - started),
        "web": {"documents": sequence_counts["web"], "tokens": web_tokens,
                "order": "F's web rows in F's order, byte for byte (checked)"},
        "flan": {"f_rows": sequence_counts["flan_in"], "f_tokens": f_manifest["flan"]["tokens"],
                 "dutch_rows_in": len(positions), "dropped_too_long": dict(too_long),
                 "dropped_before_the_mix": sequence_counts["flan_in"] - len(positions),
                 "rows": sequence_counts["flan_kept"], "tokens": dutch_flan_tokens,
                 "f_rows_per_folder": dict(f_per_folder), "rows_per_folder": dict(kept_per_folder)},
        "rows": len(sequence), "tokens": web_tokens + dutch_flan_tokens,
        "flan_token_share": dutch_flan_tokens / (web_tokens + dutch_flan_tokens),
        "f_flan_token_share": f_manifest["flan_token_share"],
        "steps_at_full_rows": (web_tokens + dutch_flan_tokens) / (32 * args.max_length),
    }
    manifest_path.write_text(json.dumps(manifest, indent=1) + "\n")
    Path(args.list).write_text(f"{path}\n")
    print(f"MIX {json.dumps({k: manifest[k] for k in ('build', 'file', 'rows', 'tokens', 'flan_token_share', 'f_flan_token_share')})}", flush=True)
    print(f"done: {path}, listed in {args.list}; manifest {manifest_path}", flush=True)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description="One host's Dutch mix of arm F-nl")
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--english-list", required=True, help="~/flan_decay_files.txt: F's mix of this host")
    parser.add_argument("--dutch", required=True, help="This host's Dutch FLAN rows (flan_nl.py assemble)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--list", required=True, help="File list for the trainer's --data-files-list")
    parser.add_argument("--tokenizer", default="yhavinga/dutch-llama-tokenizer")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--threads", type=int, default=32)
    args = parser.parse_args(argv)
    os.environ["RAYON_NUM_THREADS"] = str(args.threads)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    from transformers import AutoTokenizer

    from mmbjax.training import add_answer_token

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, split_special_tokens=True)
    if tokenizer.mask_token is None:
        # As the builder and the trainer do, so the ids are the trainer's
        tokenizer.add_special_tokens({"mask_token": "<mask>"})
    add_answer_token(tokenizer)
    build(args, tokenizer)


if __name__ == "__main__":
    main()
