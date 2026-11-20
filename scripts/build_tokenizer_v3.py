"""Tokenizer (a) of dmbert v3: the 32k vocabulary of yhavinga/dutch-llama-tokenizer with repaired special tokens.

    python scripts/build_tokenizer_v3.py --out DIR [--base DIR_OR_REPO] [--revision REV]

What changes against the tokenizer of dmbert v2 (that of the trainer: the base plus <mask> at 32,000):
  <mask> (32,000)  normalized=True: it is found after the normalizer (which prepends "▁" and turns spaces into "▁"),
                   so its pattern is "▁<mask>" and the space before it belongs to the mask, like the space that the
                   masked token "▁hoofdstad" carried in pretraining; and the whole text is normalized once, so the
                   text after it gets no extra "▁". "de <mask> van" and "van <mask>." give the pretraining ids.
  <|pad|> (32,001) its own padding token; v2 padded with id 20,277, the vocabulary entry "<pad>", which stays a
                   special token at its id (a literal "<pad>" in a text is tokenized as before).
  <s>, </s>        CLS and SEP: single texts "<s> A </s>", pairs "<s> A </s> B </s>" (v2: "<s> A" and "<s> A <s> B").
                   Both were trained in pretraining (every piece starts with <s>, a document ends with </s>); new
                   CLS/SEP tokens in unused slots would start untrained.
  <unused1..126>   32,002-32,127, special, so that the tokenizer has the 32,128 entries of the embedding.
Plain text gives the same ids as before, in the trainer's mode (add_special_tokens=False, split_special_tokens=True)
and in the default mode: the normalizer, pre-tokenizer, BPE model and decoder are unchanged, and an added token
changes nothing in a text that does not contain it. tokenizer_class is PreTrainedTokenizerFast, so that
transformers takes tokenizer.json as it is (a Llama tokenizer class rebuilds the post-processor from add_bos/add_eos);
model_input_names has no token_type_ids, which ModernBERT does not take. truncation_side stays "left", as in v2.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path

BASE_REPO = "yhavinga/dutch-llama-tokenizer"
BASE_REVISION = "d6a3a55a4ddf51a05d98d6a07f170f61a3c7c1ff"
VOCAB = 32_000
SIZE = 32_128
MASK, PAD = "<mask>", "<|pad|>"


def added(token_id, content, special=True, normalized=False):
    return {"id": token_id, "content": content, "single_word": False, "lstrip": False, "rstrip": False,
            "normalized": normalized, "special": special}


def build(base: Path, out: Path) -> dict:
    tj = json.loads((base / "tokenizer.json").read_text())
    if len(tj["model"]["vocab"]) != VOCAB:
        raise ValueError(f"{base}: {len(tj['model']['vocab'])} vocabulary entries, not {VOCAB}")
    have = {a["content"]: a["id"] for a in tj["added_tokens"]}
    if have != {"<unk>": 0, "<s>": 1, "</s>": 2, "<pad>": 20277}:
        raise ValueError(f"{base}: added tokens {have}, not those of {BASE_REPO}@{BASE_REVISION[:7]}")
    for content in [MASK, PAD] + [f"<unused{k}>" for k in range(1, SIZE - VOCAB - 1)]:
        if content in tj["model"]["vocab"]:
            raise ValueError(f"{content} is in the vocabulary already")
    new = copy.deepcopy(tj)
    new["added_tokens"] += [added(VOCAB, MASK, normalized=True), added(VOCAB + 1, PAD)]
    new["added_tokens"] += [added(VOCAB + 1 + k, f"<unused{k}>") for k in range(1, SIZE - VOCAB - 1)]
    new["post_processor"] = {
        "type": "TemplateProcessing",
        "single": [{"SpecialToken": {"id": "<s>", "type_id": 0}}, {"Sequence": {"id": "A", "type_id": 0}},
                   {"SpecialToken": {"id": "</s>", "type_id": 0}}],
        "pair": [{"SpecialToken": {"id": "<s>", "type_id": 0}}, {"Sequence": {"id": "A", "type_id": 0}},
                 {"SpecialToken": {"id": "</s>", "type_id": 0}}, {"Sequence": {"id": "B", "type_id": 1}},
                 {"SpecialToken": {"id": "</s>", "type_id": 1}}],
        "special_tokens": {"<s>": {"id": "<s>", "ids": [1], "tokens": ["<s>"]},
                           "</s>": {"id": "</s>", "ids": [2], "tokens": ["</s>"]}},
    }
    base_config = json.loads((base / "tokenizer_config.json").read_text())
    decoder = {str(a["id"]): {k: a[k] for k in ("content", "lstrip", "normalized", "rstrip", "single_word", "special")}
               for a in new["added_tokens"]}
    config = {
        "added_tokens_decoder": decoder,
        "bos_token": "<s>", "eos_token": "</s>", "unk_token": "<unk>", "cls_token": "<s>", "sep_token": "</s>",
        "pad_token": PAD, "mask_token": MASK,
        "clean_up_tokenization_spaces": False,
        "model_input_names": ["input_ids", "attention_mask"],
        "model_max_length": base_config["model_max_length"],
        "truncation_side": base_config["truncation_side"],
        "tokenizer_class": "PreTrainedTokenizerFast",
    }
    special_map = {name: {"content": content, "lstrip": False, "normalized": content == MASK, "rstrip": False,
                          "single_word": False}
                   for name, content in (("bos_token", "<s>"), ("eos_token", "</s>"), ("unk_token", "<unk>"),
                                         ("cls_token", "<s>"), ("sep_token", "</s>"), ("pad_token", PAD),
                                         ("mask_token", MASK))}
    out.mkdir(parents=True, exist_ok=True)
    files = {"tokenizer.json": json.dumps(new, ensure_ascii=False, indent=2) + "\n",
             "tokenizer_config.json": json.dumps(config, ensure_ascii=False, indent=2) + "\n",
             "special_tokens_map.json": json.dumps(special_map, ensure_ascii=False, indent=2) + "\n"}
    for name, text in files.items():
        (out / name).write_text(text)
    return {name: hashlib.sha256(text.encode()).hexdigest() for name, text in files.items()}


def base_dir(base: str, revision: str) -> Path:
    if Path(base).is_dir():
        return Path(base)
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(base, revision=revision,
                                  allow_patterns=["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--base", default=BASE_REPO, help=f"directory or Hugging Face repo (default {BASE_REPO})")
    parser.add_argument("--revision", default=BASE_REVISION)
    args = parser.parse_args()
    sums = build(base_dir(args.base, args.revision), Path(args.out))
    for name, digest in sums.items():
        print(f"{digest}  {name}")


if __name__ == "__main__":
    main()
