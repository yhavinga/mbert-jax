"""Tokenizer (a) against dmbert v2's tokenizer on a sample of CulturaX-nl documents: are the ids the same?

    python tokenizer_sample_check.py --tokenizer DIR --out sample.json [--docs 40000] [--files nl_part_00005 ...]

Per file one row group from the middle of the file (streamed from Hugging Face, uonlp/CulturaX at the revision of the
runs, 6a8734bc), its first --docs documents. Compared, per document:
  train    the trainer's call (add_special_tokens=False, split_special_tokens=True): must be equal everywhere;
  default  the default mode without the template (add_special_tokens=False): equal for every document without a
           literal special token of tokenizer (a) (<mask>, <|pad|>, <unusedN>) in its text.
v2's tokenizer is the base yhavinga/dutch-llama-tokenizer at the same revision plus <mask>, as the trainer makes it.
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import build_tokenizer_v3 as bt  # noqa: E402

REPO, REVISION = "uonlp/CulturaX", "6a8734bc"
FILES = ["nl_part_00005", "nl_part_00064", "nl_part_00121"]
NEW_LITERAL = re.compile(r"<mask>|<\|pad\|>|<unused\d+>")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tokenizer", required=True, help="directory of tokenizer (a), from build_tokenizer_v3.py")
    parser.add_argument("--out", required=True)
    parser.add_argument("--docs", type=int, default=40_000, help="documents per file")
    parser.add_argument("--files", nargs="+", default=FILES)
    args = parser.parse_args()
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    from transformers import AutoTokenizer

    base = bt.base_dir(bt.BASE_REPO, bt.BASE_REVISION)
    old_train = AutoTokenizer.from_pretrained(base, split_special_tokens=True)
    old_train.add_special_tokens({"mask_token": "<mask>"})
    old = AutoTokenizer.from_pretrained(base)
    old.add_special_tokens({"mask_token": "<mask>"})
    new_train = AutoTokenizer.from_pretrained(args.tokenizer, split_special_tokens=True)
    new = AutoTokenizer.from_pretrained(args.tokenizer)
    fs = HfFileSystem()
    report = {"repo": REPO, "revision": REVISION, "tokenizer": args.tokenizer, "files": {}, "documents": 0,
              "tokens": 0, "train_different": 0, "default_different": 0, "default_different_without_literal": 0,
              "with_literal": 0, "examples": []}
    t0 = time.time()
    for name in args.files:
        f = pq.ParquetFile(fs.open(f"datasets/{REPO}@{REVISION}/nl/{name}.parquet"))
        group = f.metadata.num_row_groups // 2
        texts = f.read_row_group(group, columns=["text"]).column("text").to_pylist()[: args.docs]
        report["files"][name] = {"row_group": group, "documents": len(texts)}
        for k in range(0, len(texts), 1000):
            chunk = texts[k: k + 1000]
            a = old_train(chunk, add_special_tokens=False)["input_ids"]
            b = new_train(chunk, add_special_tokens=False)["input_ids"]
            c = old(chunk, add_special_tokens=False)["input_ids"]
            d = new(chunk, add_special_tokens=False)["input_ids"]
            for text, x, y, u, v in zip(chunk, a, b, c, d):
                report["documents"] += 1
                report["tokens"] += len(x)
                literal = bool(NEW_LITERAL.search(text))
                report["with_literal"] += literal
                if x != y:
                    report["train_different"] += 1
                if u != v:
                    report["default_different"] += 1
                    report["default_different_without_literal"] += not literal
                if (x != y or (u != v and not literal)) and len(report["examples"]) < 10:
                    report["examples"].append({"file": name, "text": text[:200], "train_equal": x == y,
                                               "default_equal": u == v})
        print(f"{name}: {report['documents']} documents, {report['tokens']:,} tokens, train different "
              f"{report['train_different']}, default different {report['default_different']} "
              f"({report['default_different_without_literal']} without a literal), {time.time() - t0:.0f} s", flush=True)
    report["passed"] = report["train_different"] == 0 and report["default_different_without_literal"] == 0
    Path(args.out).write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    print("PASSED" if report["passed"] else "DIFFERENT")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
