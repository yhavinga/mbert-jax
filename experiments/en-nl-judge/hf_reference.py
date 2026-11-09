#!/usr/bin/env python3
"""The judge's numbers from transformers on the CPU: a check of the MaxText conversion and of the prompt ids.

    python hf_reference.py --model /dev/shm/judge/hf-31b --pairs probe.parquet --out ref.json

Per row: the ids of the chat template (transformers' apply_chat_template of one user message with judge.user_text),
which must equal judge.prompt_text tokenized; and from Gemma4ForConditionalGeneration in float32, the log probabilities
of yes and no for the next token, the score, and the 5 most likely tokens. judge.py probe compares its own numbers
with these.
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
    parser.add_argument("--pairs", required=True, help="Parquet with en and nl")
    parser.add_argument("--out", required=True)
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    args = parser.parse_args(argv)

    import pyarrow.parquet as pq
    import torch
    from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

    torch.set_num_threads(os.cpu_count())
    table = pq.read_table(args.pairs)
    en, nl = table.column("en").to_pylist(), table.column("nl").to_pylist()
    tok = AutoTokenizer.from_pretrained(args.model)
    vocab = judge.Tokenizer(args.model).vocab
    yes_ids, no_ids = judge.answer_ids(vocab)
    rows = []
    for a, b in zip(en, nl):
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
        rows.append({"prompt_ids": ids})
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
