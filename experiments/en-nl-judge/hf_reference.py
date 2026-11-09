#!/usr/bin/env python3
"""The judge's numbers from transformers on the CPU: a check of the MaxText conversion and of the prompt ids.

    python hf_reference.py --model /dev/shm/judge/hf-31b --sanity /dev/shm/judge/sanity --out ref.json

The rows: every kind of the first --sources paragraphs (by id) of the sanity set, which the TPU scores too, so that
report.py compare can set the two side by side. Per row: the ids of the chat template (transformers'
apply_chat_template of one user message with judge.user_text), which must equal judge.prompt_text tokenized by
transformers and by judge.Tokenizer; and from Gemma4ForConditionalGeneration in float32 the log probabilities of yes
and no for the next token, the score, and the 5 most likely tokens.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import judge  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description="Reference scores of the judge from transformers on the CPU")
    parser.add_argument("--model", required=True, help="Local snapshot of google/gemma-4-31B-it")
    parser.add_argument("--sanity", required=True, help="Directory of the sanity set (data.py sanity)")
    parser.add_argument("--sources", type=int, default=2, help="Paragraphs whose rows are taken (default: 2)")
    parser.add_argument("--threads", type=int, default=96, help="CPU threads; the TPU's host process needs some")
    parser.add_argument("--out", required=True)
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    args = parser.parse_args(argv)

    import pyarrow.parquet as pq
    import torch
    from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

    torch.set_num_threads(min(args.threads, os.cpu_count()))
    table = pq.read_table(Path(args.sanity) / "shard-00000.parquet").to_pylist()
    sources = sorted({r["source_id"] for r in table})[:args.sources]
    chosen = [r for r in table if r["source_id"] in sources]
    en, nl = [r["en"] for r in chosen], [r["nl"] for r in chosen]
    tok = AutoTokenizer.from_pretrained(args.model)
    fast = judge.Tokenizer(args.model)
    vocab = fast.vocab
    yes_ids, no_ids = judge.answer_ids(vocab)
    rows = []
    for r, a, b in zip(chosen, en, nl):
        messages = [{"role": "user", "content": judge.user_text(a, b)}]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if text != judge.prompt_text(a, b):
            raise SystemExit(f"the chat template renders {text[:80]!r}...{text[-80:]!r}, not judge.prompt_text")
        ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        if hasattr(ids, "keys"):
            ids = ids["input_ids"]
        ids = list(ids)
        mine = tok(judge.prompt_text(a, b), add_special_tokens=False)["input_ids"]
        if mine != ids:
            raise SystemExit("judge.prompt_text tokenized differs from the chat template's ids")
        if fast.encode_batch([judge.prompt_text(a, b)])[0] != ids:
            raise SystemExit("judge.Tokenizer (tokenizers) gives other ids than transformers")
        rows.append({"id": r["id"], "kind": r["kind"], "prompt_ids": ids})
    print(f"{len(rows)} prompts: the template renders judge.prompt_text and its ids match", flush=True)

    started = time.time()
    model = Gemma4ForConditionalGeneration.from_pretrained(args.model, dtype=getattr(torch, args.dtype))
    model.eval()
    print(f"model loaded in {time.time() - started:.0f} s", flush=True)
    yes, no = torch.tensor(yes_ids), torch.tensor(no_ids)
    for k, row in enumerate(rows):
        t0 = time.time()
        with torch.no_grad():
            logits = model(input_ids=torch.tensor([row["prompt_ids"]]), logits_to_keep=1).logits
        logp = torch.log_softmax(logits[0, -1].float(), dim=-1)
        row["logp_yes"] = float(torch.logsumexp(logp[yes], 0))
        row["logp_no"] = float(torch.logsumexp(logp[no], 0))
        row["score"] = judge.score_from_logps(row["logp_yes"], row["logp_no"])
        top = torch.topk(logp, 5)
        row["top"] = [[vocab[int(i)], round(float(v), 4)] for v, i in zip(top.values, top.indices)]
        print(json.dumps({"row": k, "tokens": len(row["prompt_ids"]), "seconds": round(time.time() - t0, 1),
                          "score": round(row["score"], 4), "top": row["top"][:3]}, ensure_ascii=False), flush=True)
    Path(args.out).write_text(json.dumps({"dtype": args.dtype, "yes_ids": yes_ids, "no_ids": no_ids, "rows": rows}))


if __name__ == "__main__":
    main()
