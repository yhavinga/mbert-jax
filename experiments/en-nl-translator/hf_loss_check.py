#!/usr/bin/env python3
"""A reference for the loss of sft_lora.py's first step: TranslateGemma in Hugging Face transformers on the CPU, the
mean cross-entropy per token of the Dutch and <end_of_turn>\\n given TranslateGemma's prompt, over the first examples
of a training file (MaxText's pipeline packs them in file order, so the first batch holds the first ~34). Step 1 of
the LoRA run is the base model (lora_b starts at zero), so the two should agree to about the spread between examples.

    JAX_PLATFORMS=cpu python hf_loss_check.py --hf-dir /dev/shm/translator/smoke-lora/hf-12b \\
        --data /dev/shm/translator/smoke-lora/data/train-N.parquet --n 34
"""

import argparse
import json
import sys
import time
from pathlib import Path


def main(argv=None):
    import pyarrow.parquet as pq
    import torch
    import transformers

    parser = argparse.ArgumentParser(description="Reference loss of the base model with transformers on the CPU")
    parser.add_argument("--hf-dir", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--n", type=int, default=34)
    parser.add_argument("--threads", type=int, default=64)
    args = parser.parse_args(argv)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from sft_data import structured

    torch.set_num_threads(args.threads)
    tok = transformers.AutoTokenizer.from_pretrained(args.hf_dir)
    started = time.time()
    model = transformers.Gemma3ForConditionalGeneration.from_pretrained(args.hf_dir, dtype=torch.float32)
    model.eval()
    print(f"loaded in {time.time() - started:.0f} s", flush=True)
    rows = pq.read_table(args.data).column("messages").to_pylist()[:args.n]
    total_nll, total_tokens, per_example = 0.0, 0, []
    for k, msgs in enumerate(rows):
        user, assistant = msgs[0]["content"], msgs[1]["content"]
        prompt = tok.apply_chat_template(structured(user), tokenize=True, add_generation_prompt=True)
        prompt = list(prompt["input_ids"] if hasattr(prompt, "keys") else prompt)
        completion = tok(assistant + "<end_of_turn>\n", add_special_tokens=False)["input_ids"]
        ids = torch.tensor([prompt + completion])
        with torch.no_grad():
            logits = model(input_ids=ids).logits[0, :-1].float()
        targets = ids[0, 1:]
        nll = torch.nn.functional.cross_entropy(logits, targets, reduction="none")[len(prompt) - 1:]
        total_nll += float(nll.sum())
        total_tokens += len(completion)
        per_example.append(round(float(nll.mean()), 3))
        print(f"example {k}: {len(completion)} tokens, loss {per_example[-1]}, first tokens "
              f"{[round(float(x), 2) for x in nll[:6]]}", flush=True)
    print("HFLOSS " + json.dumps({"examples": len(rows), "tokens": total_tokens,
                                  "mean_loss": round(total_nll / total_tokens, 4), "per_example": per_example}))


if __name__ == "__main__":
    main()
