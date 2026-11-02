#!/usr/bin/env python3
"""Does a fine-tune of finetune.py fit at this per-device batch? The worst case: every row at --max-length.

Usage:
    python scripts/dumb/probe_memory.py --checkpoint CKPT_DIR --max-length 2048 --batch-size 2 --gradient-accumulation-steps 24

Runs the train and eval steps of finetune.py on one batch of rows of max_length tokens, with the
model of the checkpoint. Prints one PROBE line. Exit code 0: it fits; 3: out of device memory;
anything else is an error of its own.
"""

import argparse
import sys
from pathlib import Path

import jax
import numpy as np
from flax import jax_utils

sys.path.insert(0, str(Path(__file__).parent))
import finetune  # noqa: E402

OUT_OF_MEMORY = 3


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--max-length", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True, help="Per device, as in finetune.py")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--num-labels", type=int, default=2)
    args = parser.parse_args()
    head = (f"PROBE max_length={args.max_length} batch_size={args.batch_size} "
            f"accumulation={args.gradient_accumulation_steps} devices={finetune.NUM_DEVICES}")

    params, config = finetune.load_pretrained_params(Path(args.checkpoint))
    _, state = finetune.create_model_and_state(config, params, args.num_labels, 2e-5, jax.random.PRNGKey(0),
                                               args.gradient_accumulation_steps)
    state = jax_utils.replicate(state)
    rows = args.batch_size * finetune.NUM_DEVICES
    rng = np.random.RandomState(0)
    batch = {
        "input_ids": rng.randint(5, config.vocab_size, size=(rows, args.max_length)).astype(np.int32),
        "attention_mask": np.ones((rows, args.max_length), dtype=np.float32),
        "labels": rng.randint(0, args.num_labels, size=rows).astype(np.int32),
    }
    try:
        # Twice: the second call compiles again, for the state the first one returned
        for _ in range(2):
            state, metrics = finetune.p_train_step(state, finetune.shard_batch(batch))
            float(metrics["loss"][0])
        int(finetune.p_eval_step(state, finetune.shard_batch(batch))["correct"].sum())
    except Exception as error:
        if "RESOURCE_EXHAUSTED" not in str(error):
            raise
        print(f"{head} OUT_OF_MEMORY {str(error).splitlines()[0][:300]}", flush=True)
        sys.exit(OUT_OF_MEMORY)
    stats = [device.memory_stats() or {} for device in jax.local_devices()]
    peak = max(s.get("peak_bytes_in_use", -1) for s in stats) / 2**30
    limit = min(s.get("bytes_limit", -1) for s in stats) / 2**30
    print(f"{head} FITS peak_GiB={peak:.2f} limit_GiB={limit:.2f}", flush=True)


if __name__ == "__main__":
    main()
