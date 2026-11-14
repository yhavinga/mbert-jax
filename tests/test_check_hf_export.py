"""Tests of scripts/check_hf_export.py and of the export path it checks (convert_flax_to_hf.py against the JAX model)."""

from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import check_hf_export as chk  # noqa: E402

SPECIAL = {"<s>", "</s>", "<unk>", "<pad>", "<mask>"}


def test_sentencepiece_spans_with_the_prepended_space_pass():
    text = "De kat, café"
    tokens = ["<s>", "▁De", "▁kat", ",", "▁caf", "é"]
    offsets = [(0, 0), (0, 2), (2, 6), (6, 7), (7, 11), (11, 12)]
    problems, counts = chk.offset_problems(text, tokens, offsets, SPECIAL)
    assert problems == []
    assert counts["segment_starts"] == 1 and counts["space_then_word"] == 2 and counts["tokens"] == 5


def test_a_start_one_character_late_is_reported():
    # RobBERT-2023's kind of error: every word's span starts one character late ("ĠDe" -> "e")
    text = "De kat zit"
    tokens = ["<s>", "▁De", "▁kat", "▁zit"]
    offsets = [(0, 0), (1, 2), (3, 6), (7, 10)]
    problems, _ = chk.offset_problems(text, tokens, offsets, SPECIAL)
    spans = [p["span"] for p in problems if p["why"] == "span is not the token's text"]
    assert spans == [[1, 2], [3, 6], [7, 10]]


def test_byte_tokens_of_one_character_share_its_span():
    text = "Prijs €5"
    tokens = ["<s>", "▁Prijs", "▁", "<0xE2>", "<0x82>", "<0xAC>", "5"]
    offsets = [(0, 0), (0, 5), (5, 6), (6, 7), (6, 7), (6, 7), (7, 8)]
    problems, counts = chk.offset_problems(text, tokens, offsets, SPECIAL)
    assert problems == [] and counts["byte_groups"] == 1
    # The wrong bytes for the character are an error
    bad = tokens[:3] + ["<0xE2>", "<0x82>", "<0xAD>"] + tokens[6:]
    problems, _ = chk.offset_problems(text, bad, offsets, SPECIAL)
    assert [p["span"] for p in problems] == [[6, 7]]


def test_a_gap_and_a_literal_special_token():
    text = "de<mask>van"
    tokens = ["<s>", "▁de", "<mask>", "▁van"]
    assert chk.offset_problems(text, tokens, [(0, 0), (0, 2), (2, 8), (8, 11)], SPECIAL)[0] == []
    problems, _ = chk.offset_problems(text, tokens, [(0, 0), (0, 2), (3, 8), (8, 11)], SPECIAL)
    assert any(p["why"].startswith("gap or overlap") for p in problems)


def test_a_window_inside_a_text_has_no_prepended_space():
    text = "aa bb cc"
    tokens = ["▁bb", "▁cc"]
    offsets = [(2, 5), (5, 8)]
    assert chk.offset_problems(text, tokens, offsets, SPECIAL, first_is_segment_start=False)[0] == []
    assert chk.offset_problems(text, tokens, offsets, SPECIAL, first_is_segment_start=True)[0] != []


def test_compare_logits_ignores_padding():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(2, 4, 7)).astype(np.float32)
    b = a.copy()
    mask = np.array([[1, 1, 1, 1], [1, 1, 0, 0]])
    b[1, 2:] = 99.0  # padding: never compared
    b[0, 1, 3] += 1e-3
    r = chk.compare_logits(a, b, mask)
    assert r["positions"] == 6 and r["top1_equal"] == 6
    assert r["max_abs_diff"] == pytest.approx(1e-3, rel=1e-3)
    b[0, 0] = -a[0, 0]
    assert chk.compare_logits(a, b, mask)["top1_equal"] == 5


def test_mask_notations():
    n = chk.mask_notations("Amsterdam is de [hoofdstad] van Nederland.", "<mask>")
    assert n == {"spaces": "Amsterdam is de <mask> van Nederland.",
                 "no space before": "Amsterdam is de<mask> van Nederland.",
                 "no space before or after": "Amsterdam is de<mask>van Nederland."}
    with pytest.raises(ValueError):
        chk.split_marked("no word")


def _tokenizer_or_skip():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained("yhavinga/dmbert-230m-culturax", local_files_only=True)
    except OSError:
        pytest.skip("the tokenizer of yhavinga/dmbert-230m-culturax is not in the local Hugging Face cache")


def test_the_dutch_tokenizer_offsets_are_the_tokens_text():
    tokenizer = _tokenizer_or_skip()
    report = chk.check_offsets(tokenizer)
    assert report["problems"] == 0, [t["problems"] for t in report["texts"] if t["problems"]]
    assert len(report["qa_windows"]) > 1


def test_ids_with_word_masked_replaces_one_token():
    tokenizer = _tokenizer_or_skip()
    ids, word = chk.ids_with_word_masked(tokenizer, "Amsterdam is de [hoofdstad] van Nederland.")
    assert tokenizer.convert_ids_to_tokens(word) == "▁hoofdstad"
    assert tokenizer.convert_ids_to_tokens(ids) == ["<s>", "▁Amsterdam", "▁is", "▁de", "<mask>", "▁van", "▁Nederland", "."]
    # The notation without spaces around <mask> gives the same ids when a word follows
    assert tokenizer("Amsterdam is de<mask>van Nederland.")["input_ids"] == ids


def test_converted_tiny_model_gives_the_jax_logits(tmp_path):
    """convert_flax_to_hf.py on a random tiny model: HF/PyTorch logits equal the JAX model's, with padding and
    sequences longer than the local window."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    import jax
    import jax.numpy as jnp
    from convert_flax_to_hf import convert_flax_to_hf

    from mmbjax.configuration_mmbbert import MMBertConfig
    from mmbjax.heads.masked_lm import ModernBertForMaskedLM

    cfg = {"vocab_size": 96, "hidden_size": 32, "num_hidden_layers": 4, "num_attention_heads": 2,
           "intermediate_size": 48, "max_position_embeddings": 64, "global_attn_every_n_layers": 3,
           "local_attention": 8}
    model = ModernBertForMaskedLM(MMBertConfig(**cfg))
    ones = jnp.ones((1, 8), dtype=jnp.int32)
    params = model.init(jax.random.PRNGKey(0), input_ids=ones, attention_mask=ones, deterministic=True)["params"]
    # Every leaf away from its initial value, so that a swapped norm or bias would show
    leaves, tree = jax.tree_util.tree_flatten(params)
    keys = jax.random.split(jax.random.PRNGKey(1), len(leaves))
    leaves = [np.asarray(x + 0.2 * jax.random.normal(k, x.shape), dtype=np.float32) for x, k in zip(leaves, keys)]
    params = jax.tree_util.tree_unflatten(tree, leaves)
    flax_dir = tmp_path / "flax"
    flax_dir.mkdir()
    (flax_dir / "params.pkl").write_bytes(pickle.dumps(jax.tree_util.tree_map(np.asarray, params)))
    (flax_dir / "config.json").write_text(json.dumps(dict(cfg, model_type="modernbert")))
    hf_dir = tmp_path / "hf"
    convert_flax_to_hf(flax_dir, hf_dir, verify=False)
    # Without a tokenizer the converter writes no special ids; HF's defaults are outside this vocabulary
    hf_cfg = json.loads((hf_dir / "config.json").read_text())
    hf_cfg.update(pad_token_id=0, bos_token_id=1, eos_token_id=2, cls_token_id=1, sep_token_id=2)
    (hf_dir / "config.json").write_text(json.dumps(hf_cfg))

    rng = np.random.default_rng(0)
    ids, mask = chk.pad_rows([list(rng.integers(3, 96, size=n)) for n in (30, 19, 5)], pad_id=0)
    ref = np.asarray(model.apply({"params": params}, input_ids=jnp.asarray(ids), attention_mask=jnp.asarray(mask),
                                 deterministic=True).logits)
    hf = transformers.AutoModelForMaskedLM.from_pretrained(hf_dir)
    hf.eval()
    with torch.no_grad():
        out = hf(input_ids=torch.tensor(ids, dtype=torch.long), attention_mask=torch.tensor(mask, dtype=torch.long)).logits.numpy()
    r = chk.compare_logits(ref, out, mask)
    assert r["top1_equal"] == r["positions"] == 54
    assert r["max_abs_diff"] < 1e-4, r
