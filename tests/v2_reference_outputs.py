"""Outputs of the model and of two train steps on a fixed small packed batch: the reference the v3 flags leave alone.

Run on the code of dmbert v2 (mbert-jax final-v2 @ 494d4d9) it wrote tests/fixtures/v2_reference.npz:
    cd <final-v2 checkout> && <python> -I <this file> --root . --out <v3 checkout>/tests/fixtures/v2_reference.npz
tests/test_v3_flags.py calls outputs() on the code of its own checkout, without the flags, and wants every array
equal bit for bit. Not a test module itself (no test_ prefix).
"""

import argparse
import sys
from pathlib import Path
from unittest import mock

VOCAB, MASK_ID, PAD_ID = 64, 63, 0


def batch_arrays():
    """Two packed rows as the trainer's collator makes them: row 0 two documents, row 1 one document and padding."""
    import numpy as np

    rng = np.random.default_rng(20261120)
    seq = 24
    segment_ids = np.array([[0] * 10 + [1] * 14, [0] * 18 + [-1] * 6], dtype=np.int32)
    positions = np.array([list(range(10)) + list(range(14)), list(range(18)) + [0] * 6], dtype=np.int32)
    ids = rng.integers(3, MASK_ID, size=(2, seq)).astype(np.int32)
    ids[segment_ids < 0] = PAD_ID
    labels = np.full_like(ids, -100)
    chosen = (rng.random((2, seq)) < 0.3) & (segment_ids >= 0)
    labels[chosen] = ids[chosen]
    inputs = ids.copy()
    inputs[chosen] = MASK_ID
    return {"input_ids": inputs, "attention_mask": (segment_ids >= 0).astype(np.int32), "labels": labels,
            "position_ids": positions, "segment_ids": segment_ids}


def config_kwargs(dtype):
    # Global attention every 3rd layer and a local window smaller than the documents, as in the 230M runs
    return dict(vocab_size=VOCAB, hidden_size=32, num_hidden_layers=4, num_attention_heads=4, intermediate_size=48,
                global_attn_every_n_layers=3, local_attention=8, max_position_embeddings=64, pad_token_id=PAD_ID,
                bos_token_id=1, eos_token_id=2, compute_dtype=dtype, param_dtype="float32")


def outputs(root, **flags):
    """name -> array: logits of the forward pass, then the metrics and parameters after two train steps, in float32
    and in bfloat16 compute. flags go to MMBertConfig (attention_float32) and train_step (z_loss_coef,
    nonfinite_diagnostics)."""
    root = Path(root).resolve()
    for path in (root / "scripts", root / "src"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import jax
    import jax.numpy as jnp
    import numpy as np
    from flax import traverse_util
    from flax.training import train_state

    from mmbjax.configuration_mmbbert import MMBertConfig
    from mmbjax.heads.masked_lm import ModernBertForMaskedLM

    with mock.patch("jax.distributed.initialize", side_effect=ValueError):
        import train_minimal as tm
    if Path(tm.__file__).resolve().parent != root / "scripts":
        raise RuntimeError(f"train_minimal from {tm.__file__}, not from {root / 'scripts'}")

    step_flags = {name: flags.pop(name) for name in ("z_loss_coef", "nonfinite_diagnostics") if name in flags}
    batch = {name: jnp.asarray(array) for name, array in batch_arrays().items()}
    out = {}
    for dtype in ("float32", "bfloat16"):
        config = MMBertConfig(**config_kwargs(dtype), **flags)
        model = ModernBertForMaskedLM(config, dtype=getattr(jnp, dtype), param_dtype=jnp.float32)
        params = model.init(jax.random.PRNGKey(0), input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                            deterministic=True)["params"]
        logits = model.apply({"params": params}, input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                             position_ids=batch["position_ids"], segment_ids=batch["segment_ids"], deterministic=True).logits
        out[f"{dtype}/logits"] = np.asarray(logits)
        optimizer, _ = tm.create_train_state(
            model, learning_rate=4.8e-4, num_train_steps=100, warmup_steps=2, weight_decay=0.01, lr_schedule_type="wsd",
            layer_norm_lr_factor=1.0, min_lr=1e-6, decay_steps=20, skip_nonfinite_updates=True,
        )
        state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)
        step = jax.jit(lambda s, b, r, lr, per_layer: tm.train_step(s, b, r, lr, per_layer, **step_flags),
                       static_argnums=(4,))
        rng = jax.random.PRNGKey(1)
        for k in range(2):
            state, metrics, rng = step(state, batch, rng, 4.8e-4, False)
            for name in ("loss", "accuracy", "grad_norm", "param_norm", "logits_min", "logits_max"):
                out[f"{dtype}/step{k}/{name}"] = np.asarray(metrics[name])
        for path, leaf in traverse_util.flatten_dict(state.params).items():
            out[f"{dtype}/params/{'/'.join(path)}"] = np.asarray(leaf)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="checkout whose scripts/ and src/ to run")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    import numpy as np

    out = outputs(args.root)
    np.savez(args.out, **out)
    print(f"{len(out)} arrays to {args.out}")


if __name__ == "__main__":
    main()
