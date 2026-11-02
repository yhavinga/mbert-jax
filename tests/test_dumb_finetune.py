"""scripts/dumb/finetune.py: padding to a multiple changes no result of the classifier."""
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "dumb"))

import finetune  # noqa: E402

from mmbjax.configuration_mmbbert import MMBertConfig  # noqa: E402
from mmbjax.heads.classification import ModernBertForSequenceClassification  # noqa: E402


def test_padded_length_rounds_up_and_stops_at_max_length():
    batch = [{"input_ids": list(range(1, 6)), "label": 0}, {"input_ids": list(range(1, 10)), "label": 1}]
    assert finetune.collate_fn(batch, 0, 32)["input_ids"].shape == (2, 9)
    assert finetune.collate_fn(batch, 0, 32, pad_to_multiple_of=8)["input_ids"].shape == (2, 16)
    assert finetune.collate_fn(batch, 0, 12, pad_to_multiple_of=8)["input_ids"].shape == (2, 12)
    assert finetune.collate_fn(batch, 0, 7, pad_to_multiple_of=8)["input_ids"].shape == (2, 7)


def test_padding_to_a_multiple_gives_the_same_logits_and_loss():
    # cls pooling, local and global layers, as in the converted checkpoints the fine-tunes start from
    config = MMBertConfig(vocab_size=64, hidden_size=32, intermediate_size=32, num_hidden_layers=4, num_attention_heads=2,
                          max_position_embeddings=64, local_attention=4, global_attn_every_n_layers=3, num_labels=2,
                          pad_token_id=0, classifier_pooling="cls")
    model = ModernBertForSequenceClassification(config)
    rng = np.random.RandomState(0)
    batch = [{"input_ids": [1] + list(rng.randint(5, 64, size=n)), "label": int(n % 2)} for n in (4, 11, 19)]
    tight = finetune.collate_fn(batch, 0, 64)
    rounded = finetune.collate_fn(batch, 0, 64, pad_to_multiple_of=16)
    assert tight["input_ids"].shape[1] == 20 and rounded["input_ids"].shape[1] == 32
    params = model.init(jax.random.PRNGKey(0), jnp.array(tight["input_ids"]), deterministic=True)

    def run(collated):
        out = model.apply(params, jnp.array(collated["input_ids"]), attention_mask=jnp.array(collated["attention_mask"]),
                          labels=jnp.array(collated["labels"]), deterministic=True)
        return np.asarray(out.logits), float(out.loss)

    tight_logits, tight_loss = run(tight)
    rounded_logits, rounded_loss = run(rounded)
    np.testing.assert_allclose(rounded_logits, tight_logits, rtol=1e-5, atol=1e-6)
    assert abs(rounded_loss - tight_loss) < 1e-6


def test_memory_probe_runs_the_worst_case_and_says_it_fits(tmp_path):
    import json
    import os
    import pickle
    import subprocess

    config = MMBertConfig(vocab_size=64, hidden_size=32, intermediate_size=32, num_hidden_layers=3, num_attention_heads=2,
                          max_position_embeddings=256, global_attn_every_n_layers=3, pad_token_id=0)
    model = ModernBertForSequenceClassification(config)
    params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 8), dtype=jnp.int32), deterministic=True)["params"]
    # The layout of a converted checkpoint: the masked-LM tree, whose model and prediction head the fine-tune takes
    (tmp_path / "params.pkl").write_bytes(pickle.dumps({"model": jax.device_get(params["model"]), "prediction_head": jax.device_get(params["head"])}))
    (tmp_path / "config.json").write_text(json.dumps({**config.to_dict(), "tokenizer_name": "unused"}))
    script = Path(__file__).resolve().parents[1] / "scripts" / "dumb" / "probe_memory.py"
    done = subprocess.run([sys.executable, str(script), "--checkpoint", str(tmp_path), "--max-length", "32", "--batch-size", "2",
                           "--gradient-accumulation-steps", "3"], capture_output=True, text=True, timeout=300,
                          env={**os.environ, "JAX_PLATFORMS": "cpu", "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")})
    assert done.returncode == 0, done.stderr[-2000:]
    assert "PROBE max_length=32 batch_size=2 accumulation=3 devices=1 FITS" in done.stdout
