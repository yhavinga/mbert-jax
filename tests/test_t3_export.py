"""Export of a model of T3's shape to Hugging Face (scripts/convert_flax_to_hf.py) with tokenizer (a): a plain
ModernBertForMaskedLM with the logits of the JAX model (experiments/t3/protocol.md, section 3). The width is T3's
(hidden 1152, 18 heads of 64, GeGLU 2944, a global layer every 3rd); 4 layers instead of 18 keep it on a small CPU,
and Hugging Face's own count of the full shape on the meta device is T3's 317,106,304. Needs torch and transformers
(the venv of hf-release)."""

import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

import jax.numpy as jnp  # noqa: E402
from test_v3_export import _export, tokenizer_dir  # noqa: E402, F401  (the fixture of v3's export test)


def test_t3s_width_exports_as_modernbert_with_the_logits_of_the_jax_model(tmp_path, tokenizer_dir):  # noqa: F811
    model, params, hf_dir = _export(tmp_path, tokenizer_dir, 4, 1152, 18, 2944, 3)
    hf = transformers.ModernBertForMaskedLM.from_pretrained(hf_dir, attn_implementation="eager").eval()
    assert type(hf).__name__ == "ModernBertForMaskedLM"
    c = hf.config
    assert (c.hidden_size, c.num_attention_heads, c.intermediate_size, c.global_attn_every_n_layers) == (1152, 18, 2944, 3)
    assert (c.local_attention, c.global_rope_theta, c.local_rope_theta) == (128, 160000.0, 10000.0)
    tok = transformers.AutoTokenizer.from_pretrained(hf_dir)
    text = "Amsterdam is de <mask> van Nederland. " * 25
    batch = tok([text, "Een korte zin met <mask>."], padding=True, return_tensors="np")
    assert batch["input_ids"].shape[1] > 160  # beyond the local window of 128
    with torch.no_grad():
        theirs = hf(input_ids=torch.tensor(batch["input_ids"]), attention_mask=torch.tensor(batch["attention_mask"])).logits.numpy()
    ours = np.asarray(model.apply({"params": params}, input_ids=jnp.asarray(batch["input_ids"]),
                                  attention_mask=jnp.asarray(batch["attention_mask"]), deterministic=True).logits)
    real = batch["attention_mask"].astype(bool)
    difference = np.abs(theirs - ours)[real].max()
    assert difference < 1e-3 * max(1.0, np.abs(ours[real]).max()), difference
    assert (theirs.argmax(-1)[real] == ours.argmax(-1)[real]).all()


def test_hugging_face_counts_317106304_parameters_for_t3s_full_shape(tmp_path, tokenizer_dir):  # noqa: F811
    _, _, hf_dir = _export(tmp_path, tokenizer_dir, 2, 64, 1, 128, 3)  # only for the exporter's config keys
    config = transformers.ModernBertConfig.from_pretrained(hf_dir)
    config.update({"hidden_size": 1152, "num_hidden_layers": 18, "num_attention_heads": 18, "intermediate_size": 2944,
                   "global_attn_every_n_layers": 3})
    with torch.device("meta"):
        model = transformers.ModernBertForMaskedLM(config)
    assert sum(p.numel() for p in model.parameters()) == 317_106_304  # the decoder is tied to the embeddings
    assert model.config.vocab_size == 32_128
