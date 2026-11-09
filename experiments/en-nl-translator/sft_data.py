#!/usr/bin/env python3
"""The data of the LoRA smoke test (hammond mbert/designs/en-nl-translator.md §3 step 3): the pilot's pairs as
(synthetic English -> original Dutch) in TranslateGemma's own prompt, for MaxText's SFT pipeline, and the passages that
compare the base model with the adapters.

    python sft_data.py build --pairs pairs.parquet --flan-en flan-en-w0.parquet --flan-examples examples.parquet \\
        --tokenizer /dev/shm/flan_nl/tokenizer-12b --out DIR
    python sft_data.py check --data DIR --tokenizer /dev/shm/flan_nl/tokenizer-12b   (SFT venv: MaxText's pipeline)

build writes to DIR:
- train-N.parquet, train-B.parquet: one column `messages`, [{role: user, content}, {role: assistant, content: Dutch}],
  the same pairs in the same shuffled order; arm N's user content is TAG, a space and the English, arm B's the English.
- heldout.parquet: the HOLDOUT pairs left out of both (random, seeded).
- compare.parquet: the passages translated by the base model and the adapters: COMPARE_FLAN rows of FLAN (`inputs` of
  rows FLAN-nl translated, w0) and COMPARE_PILOT of the held-out pairs; segments-plain.parquet and segments-tag.parquet
  are their input for translate.py (id, text, tokens), without and with the tag.
- translategemma_en_nl.jinja: the chat template MaxText's pipeline gets (chat_template_path). MaxText casts message
  content to a string, while TranslateGemma's own template takes the user turn as a list with one mapping (type,
  source_lang_code, target_lang_code, text) and raises on a string. This template renders a string user turn as
  TranslateGemma's template renders {type: text, source_lang_code: en, target_lang_code: nl, text}, and the assistant
  turn as the model's output follows the prompt when it translates: "<start_of_turn>model\\n", the text,
  "<end_of_turn>\\n" (TranslateGemma's own template puts a newline and 8 spaces before the text); `check` compares the
  ids on every example.
- manifest.json: counts, seed, tag, the ids held out and compared, the token statistics.

The loss falls on the Dutch only: MaxText's SFTPromptMasking (sft_train_on_completion_only) masks the prompt; what is
trained is the completion that TranslateGemma's template makes of the assistant turn, the Dutch and `<end_of_turn>\\n`.
"""

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

# The fixed tag of arm N, in front of the source text inside the model's own prompt (design §3 step 3). Caswell et al.
# 2019 used a reserved token <BT>; here it is plain text, so that its tokens have trained embeddings (embeddings stay
# frozen under LoRA). A choice of this smoke test, not of the design.
TAG = "<BT>"
HOLDOUT = 10
COMPARE_FLAN = 5
COMPARE_PILOT = 5
# Source tokens of the compared passages: within translate.py's small and medium classes, so two engine loads per run
COMPARE_TOKENS = (33, 256)
SEED = 0

PROMPT_HEAD = ("<start_of_turn>user\nYou are a professional English (en) to Dutch (nl) translator. Your goal is to "
               "accurately convey the meaning and nuances of the original English text while adhering to Dutch "
               "grammar, vocabulary, and cultural sensitivities.\nProduce only the Dutch translation, without any "
               "additional explanations or commentary. Please translate the following English text into Dutch:\n\n\n")

# TranslateGemma's chat_template.jinja for en -> nl text, with the user's text as the content string
CHAT_TEMPLATE = (
    "{{ bos_token }}\n"
    "{%- if (messages[0]['role'] != 'user') -%}{{ raise_exception('Conversations must start with a user prompt.') }}"
    "{%- endif -%}\n"
    "{%- for message in messages -%}\n"
    "{%- if (message['role'] == 'user') != (loop.index0 % 2 == 0) -%}"
    "{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{%- endif -%}\n"
    "{%- if message['content'] is none or message['content'] is not string -%}"
    "{{ raise_exception('content must be a string') }}{%- endif -%}\n"
    "{%- if message['role'] == 'assistant' -%}{{ '<start_of_turn>model\\n' }}{{ message['content'] | trim }}\n"
    "{%- elif message['role'] == 'user' -%}{{ " + json.dumps(PROMPT_HEAD) + " + message['content'] | trim }}\n"
    "{%- else -%}{{ raise_exception('Conversations must only contain user or assistant roles.') }}{%- endif -%}\n"
    "{{ '<end_of_turn>\\n' }}\n"
    "{%- endfor -%}\n"
    "{%- if add_generation_prompt -%}{{ '<start_of_turn>model\\n' }}{%- endif -%}\n"
)


def source_text(en, tagged):
    return f"{TAG} {en}" if tagged else en


def messages(en, nl, tagged):
    return [{"role": "user", "content": source_text(en, tagged)}, {"role": "assistant", "content": nl}]


def structured(text):
    """The user turn as TranslateGemma's own template takes it (translate.py's messages())."""
    return [{"role": "user", "content": [{"type": "text", "source_lang_code": "en", "target_lang_code": "nl",
                                          "text": text}]}]


def split_pairs(ids, holdout, seed):
    """(train ids in a seeded random order, held-out ids): disjoint, together all ids."""
    order = list(ids)
    random.Random(seed).shuffle(order)
    return order[holdout:], order[:holdout]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(args):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    pairs = pq.read_table(args.pairs, columns=["id", "pid", "source", "half", "en", "nl", "tokens_en", "tokens_nl",
                                               "ended"]).to_pylist()
    if not all(p["ended"] for p in pairs):
        raise SystemExit(f"{args.pairs}: a pair whose nl->en translation did not end")
    by_id = {p["id"]: p for p in pairs}
    if len(by_id) != len(pairs):
        raise SystemExit(f"{args.pairs}: duplicate ids")
    for p in pairs:
        p["en"], p["nl"] = p["en"].strip(), p["nl"].strip()
        if not p["en"] or not p["nl"]:
            raise SystemExit(f"{args.pairs}: pair {p['id']} has an empty side")
    train_ids, held_ids = split_pairs(sorted(by_id), args.holdout, args.seed)

    for arm, tagged in (("N", True), ("B", False)):
        rows = [messages(by_id[i]["en"], by_id[i]["nl"], tagged) for i in train_ids]
        pq.write_table(pa.table({"messages": rows}), out / f"train-{arm}.parquet")
    held = [by_id[i] for i in held_ids]
    pq.write_table(pa.Table.from_pylist([{k: p[k] for k in ("id", "pid", "source", "half", "en", "nl", "tokens_en",
                                                             "tokens_nl")} for p in held]), out / "heldout.parquet")

    def n_tokens(text):
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])

    low, high = COMPARE_TOKENS
    pilot = [p for p in held if low <= n_tokens(p["en"]) <= high][:COMPARE_PILOT]
    if len(pilot) < COMPARE_PILOT:
        raise SystemExit(f"only {len(pilot)} held-out pairs with {low}-{high} English tokens")
    # FLAN: rows of w0 that FLAN-nl translated, drawn at random until COMPARE_FLAN fit the token band
    status = pq.read_table(args.flan_examples, columns=["host", "position", "status"])
    translated = {pos for host, pos, st in zip(*(status.column(c).to_pylist() for c in ("host", "position", "status")))
                  if host == 0 and st == "translate"}
    flan = pq.read_table(args.flan_en, columns=["inputs", "source", "position", "host"]).to_pylist()
    rng = random.Random(args.seed)
    flan_rows = []
    for k in rng.sample(range(len(flan)), len(flan)):
        row = flan[k]
        if row["host"] != 0 or row["position"] not in translated:
            continue
        text = row["inputs"].strip()
        if low <= n_tokens(text) <= high:
            flan_rows.append(row | {"inputs": text})
        if len(flan_rows) == COMPARE_FLAN:
            break
    compare = [{"id": k, "origin": "flan", "ref": f"w0-{r['position']}", "task": r["source"], "en": r["inputs"],
                "nl_original": None} for k, r in enumerate(flan_rows)]
    compare += [{"id": COMPARE_FLAN + k, "origin": "pilot", "ref": p["pid"], "task": p["source"], "en": p["en"],
                 "nl_original": p["nl"]} for k, p in enumerate(pilot)]
    pq.write_table(pa.Table.from_pylist(compare), out / "compare.parquet")
    for name, tagged in (("plain", False), ("tag", True)):
        texts = [source_text(c["en"], tagged) for c in compare]
        pq.write_table(pa.table({"id": [c["id"] for c in compare], "text": texts,
                                 "tokens": [n_tokens(t) for t in texts]}), out / f"segments-{name}.parquet")
    (out / "translategemma_en_nl.jinja").write_text(CHAT_TEMPLATE, encoding="utf-8")

    manifest = {
        "pairs": str(args.pairs), "pairs_sha256": sha256(args.pairs), "seed": args.seed, "tag": TAG,
        "train": len(train_ids), "heldout_ids": held_ids, "heldout_pids": [p["pid"] for p in held],
        "compare": [{k: c[k] for k in ("id", "origin", "ref", "task")} for c in compare],
        "train_sources": {s: sum(by_id[i]["source"] == s for i in train_ids) for s in sorted({p["source"] for p in pairs})},
        "files": {f.name: sha256(f) for f in sorted(out.glob("*.parquet")) if f.name.startswith(("train-", "heldout",
                                                                                                   "compare", "segments-"))},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("train", "heldout_ids", "compare", "train_sources")}, ensure_ascii=False))


def check(args):
    """MaxText's own SFT steps (apply_chat_template, tokenization, SFTPromptMasking, as hf_data_processing runs them)
    on every training example: the inputs are TranslateGemma's prompt (its own template, and translate.py's) followed
    by the Dutch and <end_of_turn>\\n, and the targets mask the prompt."""
    import numpy as np
    import pyarrow.parquet as pq
    import transformers
    from maxtext.input_pipeline import input_pipeline_utils as ipu

    # translate.py of experiments/flan-nl is copied beside this script on the workers
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from translate import PromptCodec

    data = Path(args.data)
    template = (data / "translategemma_en_nl.jinja").read_text(encoding="utf-8")
    # As hf_data_processing.preprocessing_pipeline loads it for SFT, then validate_and_configure_sft_columns
    mt_tok = transformers.AutoTokenizer.from_pretrained(args.tokenizer, add_bos_token=False, add_eos_token=False,
                                                        legacy=False, extra_special_tokens={})
    mt_tok.chat_template = template
    own = transformers.AutoTokenizer.from_pretrained(args.tokenizer)
    codec = PromptCodec(own, "en", "nl")
    pad_id = mt_tok.pad_token_id
    stats = {}
    for arm in ("N", "B"):
        rows = pq.read_table(data / f"train-{arm}.parquet").column("messages").to_pylist()
        masking = ipu.SFTPromptMasking("messages", completion_only=True, max_target_length=10 ** 9, unk_id=pad_id)
        lengths, prompts, completions = [], [], []
        for k, msgs in enumerate(rows):
            user, assistant = msgs[0]["content"], msgs[1]["content"]
            example = ipu.apply_chat_template({"messages": msgs}, mt_tok, "messages")
            example = ipu.tokenization(example, mt_tok, truncation=False, max_length=args.max_target_length,
                                       column_names=["messages"])
            got = masking.map(example)
            prompt = own.apply_chat_template(structured(user), tokenize=True, add_generation_prompt=True)
            prompt = list(prompt["input_ids"] if hasattr(prompt, "keys") else prompt)
            if prompt != codec.encode(user):
                raise SystemExit(f"arm {arm} row {k}: TranslateGemma's prompt differs from translate.py's")
            # What the model generates after the prompt when it translates: the Dutch, then <end_of_turn>. (Its own
            # template renders an assistant turn as "model\n" + "\n" + 8 spaces + text, whitespace between two of its
            # {{ }} blocks; the generated translations never start that way, so the target here is the text itself.)
            completion = own(assistant + "<end_of_turn>\n", add_special_tokens=False)["input_ids"]
            full = prompt + completion
            expected_targets = [pad_id] * len(prompt) + completion
            if got["inputs"].tolist() != full or got["targets"].tolist() != expected_targets:
                raise SystemExit(f"arm {arm} row {k}: MaxText's SFT ids differ from prompt + Dutch + <end_of_turn>\n"
                                 f"got      {got['inputs'].tolist()[len(prompt) - 4:len(prompt) + 8]}\n"
                                 f"expected {full[len(prompt) - 4:len(prompt) + 8]}")
            if completion[-2:] != own.convert_tokens_to_ids(["<end_of_turn>", "\n"]):
                raise SystemExit(f"arm {arm} row {k}: the completion does not end in <end_of_turn>\\n")
            lengths.append(len(full))
            prompts.append(len(prompt))
            completions.append(len(completion))
        over = sum(n > args.max_target_length for n in lengths)
        if over:
            raise SystemExit(f"arm {arm}: {over} examples longer than {args.max_target_length} tokens would be cut")
        stats[arm] = {"examples": len(rows), "tokens": int(np.sum(lengths)), "prompt_tokens": int(np.sum(prompts)),
                      "loss_tokens": int(np.sum(completions)), "max_tokens": int(np.max(lengths)),
                      "mean_tokens": round(float(np.mean(lengths)), 1)}
    print("CHECK " + json.dumps(stats))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Data of the LoRA smoke test")
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--pairs", required=True)
    b.add_argument("--flan-en", required=True, help="FLAN-nl's English extract of w0 (en/w0.parquet)")
    b.add_argument("--flan-examples", required=True, help="FLAN-nl's plan2/examples.parquet (status per row)")
    b.add_argument("--tokenizer", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--holdout", type=int, default=HOLDOUT)
    b.add_argument("--seed", type=int, default=SEED)
    c = sub.add_parser("check")
    c.add_argument("--data", required=True)
    c.add_argument("--tokenizer", required=True)
    c.add_argument("--max-target-length", type=int, default=1024)
    args = parser.parse_args(argv)
    {"build": build, "check": check}[args.command](args)


if __name__ == "__main__":
    main()
