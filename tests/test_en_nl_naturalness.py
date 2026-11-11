"""naturalness.py's classify pattern on CPU with a tiny model, each case in a fresh process as classify runs: params
that come without an eager forward pass (as from load_pretrained_params), then pmap predictions at two padded lengths.
Without the warm-up the global rotary cache keeps the first trace's tracer (UnexpectedTracerError, classify on
mbert-mix, 11 Nov). A leaked tracer also outlives rope.clear_rotary_cache() in the same process, hence the subprocesses."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("flax")
ROOT = Path(__file__).resolve().parents[1]

CASE = r"""
import sys
sys.path.insert(0, sys.argv[1] + "/src")
sys.path.insert(0, sys.argv[1] + "/experiments/en-nl-translator")
import jax, jax.numpy as jnp, numpy as np
from flax import jax_utils
from mmbjax.configuration_mmbbert import MMBertConfig
from mmbjax.heads.classification import ModernBertForSequenceClassification
from mmbjax.modules import rope
import naturalness as nat

config = MMBertConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=3, num_attention_heads=2,
                      local_attention=8, pad_token_id=0, num_labels=2, max_position_embeddings=64)
model = ModernBertForSequenceClassification(config=config)
params = jax.jit(lambda r: model.init(r, jnp.ones((1, 8), jnp.int32), deterministic=True))(jax.random.PRNGKey(0))["params"]
rope.clear_rotary_cache()
if sys.argv[2] == "warm":
    nat.warm_rotary_cache(model, params, 32)
p_predict = nat.make_predict(model)
replicated = jax_utils.replicate(params)
ids = np.random.default_rng(0).integers(1, 64, size=(2, 32)).astype(np.int32)
for n in (32, 16):
    batch = ids[None, :, :n]
    got = np.asarray(p_predict(replicated, batch, np.ones_like(batch))).reshape(-1)
    eager = model.apply({"params": params}, jnp.asarray(ids[:, :n]), attention_mask=jnp.ones((2, n), jnp.int32),
                        deterministic=True)
    expected = np.asarray(jax.nn.softmax(eager.logits.astype("float32"), axis=-1)[..., 1])
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-6)
print("PREDICTIONS MATCH")
"""


def run_case(case):
    env = dict(os.environ, JAX_PLATFORMS="cpu")
    return subprocess.run([sys.executable, "-c", CASE, str(ROOT), case], capture_output=True, text=True, env=env,
                          timeout=600)


def test_classify_without_warm_up_leaks_the_rotary_tracer():
    result = run_case("cold")
    assert result.returncode != 0
    assert "UnexpectedTracerError" in result.stderr


def test_warm_rotary_cache_lets_pmap_run_at_any_padded_length_and_matches_eager():
    result = run_case("warm")
    assert result.returncode == 0, result.stderr[-3000:]
    assert "PREDICTIONS MATCH" in result.stdout
