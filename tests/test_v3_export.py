"""Export of a dmbert v3 model to Hugging Face (scripts/convert_flax_to_hf.py) with tokenizer (a): it loads as a plain
ModernBertForMaskedLM and gives the logits of the JAX model, for the shapes of the ablation (a global layer every 2nd
or 3rd layer, other widths) and sequences longer than the local window. The stability flags add no parameter, so a
model trained with them exports the same way. Needs torch and transformers (the venv of hf-release)."""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

import build_tokenizer_v3 as bt  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from convert_flax_to_hf import convert_flax_to_hf  # noqa: E402
from mmbjax.configuration_mmbbert import MMBertConfig  # noqa: E402
from mmbjax.heads.masked_lm import ModernBertForMaskedLM  # noqa: E402


@pytest.fixture(scope="module")
def tokenizer_dir(tmp_path_factory):
    from huggingface_hub import snapshot_download

    try:
        base = Path(snapshot_download(bt.BASE_REPO, revision=bt.BASE_REVISION, local_files_only=True,
                                      allow_patterns=["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]))
    except Exception:
        pytest.skip(f"{bt.BASE_REPO}@{bt.BASE_REVISION[:7]} is not in the local Hugging Face cache")
    out = tmp_path_factory.mktemp("tok")
    bt.build(base, out)
    return out


def _export(tmp_path, tokenizer_dir, layers, hidden, heads, intermediate, every):
    config = MMBertConfig(vocab_size=32_128, hidden_size=hidden, num_hidden_layers=layers, num_attention_heads=heads,
                          intermediate_size=intermediate, global_attn_every_n_layers=every, max_position_embeddings=8192,
                          pad_token_id=32_001, bos_token_id=1, eos_token_id=2, attention_float32=True)
    model = ModernBertForMaskedLM(config)
    params = model.init(jax.random.PRNGKey(0), input_ids=jnp.ones((1, 8), jnp.int32), deterministic=True)["params"]
    # Every parameter away from its initial value (norm scales 1, biases 0), so that a wrong mapping shows
    leaves, tree = jax.tree_util.tree_flatten(params)
    keys = jax.random.split(jax.random.PRNGKey(1), len(leaves))
    leaves = [leaf + 0.05 * jax.random.normal(k, leaf.shape) for leaf, k in zip(leaves, keys)]
    params = jax.tree_util.tree_unflatten(tree, leaves)
    flax_dir = tmp_path / "flax"
    flax_dir.mkdir()
    with open(flax_dir / "params.pkl", "wb") as f:
        pickle.dump(jax.tree_util.tree_map(np.asarray, params), f)
    (flax_dir / "config.json").write_text(json.dumps(config.to_dict()))
    convert_flax_to_hf(flax_dir, tmp_path / "hf", tokenizer_override=str(tokenizer_dir), verify=False)
    return model, params, tmp_path / "hf"


@pytest.mark.parametrize("shape", [(5, 64, 4, 128, 2), (4, 48, 3, 96, 3)], ids=["global-every-2", "global-every-3"])
def test_export_loads_as_modernbert_with_the_logits_of_the_jax_model(tmp_path, tokenizer_dir, shape):
    model, params, hf_dir = _export(tmp_path, tokenizer_dir, *shape)
    hf = transformers.ModernBertForMaskedLM.from_pretrained(hf_dir, attn_implementation="eager").eval()
    assert type(hf).__name__ == "ModernBertForMaskedLM"
    assert hf.config.global_attn_every_n_layers == shape[4] and hf.config.num_hidden_layers == shape[0]
    tok = transformers.AutoTokenizer.from_pretrained(hf_dir)
    assert (tok.pad_token_id, tok.mask_token_id, hf.config.pad_token_id, hf.config.cls_token_id,
            hf.config.sep_token_id) == (32_001, 32_000, 32_001, 1, 2)
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
