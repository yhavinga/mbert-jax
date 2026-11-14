"""mmbjax.hf_encoders against the PyTorch models on the CPU: RoBERTa (transformers' RobertaForMaskedLM) and NeoBERT
(model.py of pdelobelle/robbert-2026, at a fixed revision) give the same logits; the task heads plug into
finetune.py and finetune_extra.py; BLiMP-NL's words for byte-level BPE tokens.

The tiny tests need torch (skipped without it); the NeoBERT ones also fetch model.py and rotary.py (25 kB) from the
Hugging Face hub. HF_ENCODER_REAL=1 also compares the real RobBERT-2023 and RobBERT-2026 checkpoints (~1 GB download).
"""
import os
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "dumb"))

from mmbjax import hf_encoders  # noqa: E402

ROBBERT_2023 = ("DTAI-KULeuven/robbert-2023-dutch-base", "1164b2df6d3ceb1735900c839da8de9bad1a99f9")
ROBBERT_2026 = ("pdelobelle/robbert-2026", "a406b1a3eabf41401293e2a096167729bbd4d479")


def torch_state(model):
    return {k: v.detach().float().numpy() for k, v in model.state_dict().items()}


def scramble(torch, model, seed):
    """Weights far from their initialization (norm scales and biases too), so a swapped or transposed matrix shows."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.3 + (1.0 if parameter.dim() == 1 else 0.0))


def batch(pad, lengths, vocab, seed=0):
    """Rows of random ids (never pad) of these lengths, padded on the right with pad, and their mask."""
    rng = np.random.RandomState(seed)
    ids = np.full((len(lengths), max(lengths)), pad, dtype=np.int64)
    for i, n in enumerate(lengths):
        ids[i, :n] = rng.choice([t for t in range(3, vocab) if t != pad], size=n)
    return ids, (ids != pad).astype(np.int64)


def assert_logits_match(ours, theirs, mask, atol):
    ours, theirs = np.asarray(ours), np.asarray(theirs)
    real = mask.astype(bool)
    difference = np.abs(ours[real] - theirs[real]).max()
    assert difference < atol, f"largest difference {difference}"
    assert (ours[real].argmax(-1) == theirs[real].argmax(-1)).all()


# ---------------------------------------------------------------- RoBERTa


@pytest.fixture(scope="module")
def tiny_roberta():
    torch = pytest.importorskip("torch")
    from transformers import RobertaConfig, RobertaForMaskedLM

    hf = RobertaConfig(vocab_size=61, hidden_size=32, num_hidden_layers=2, num_attention_heads=4, intermediate_size=48,
                       max_position_embeddings=20, pad_token_id=1, bos_token_id=0, eos_token_id=2, type_vocab_size=1,
                       layer_norm_eps=1e-5)
    torch.manual_seed(0)
    model = RobertaForMaskedLM(hf).eval()
    scramble(torch, model, 1)
    model.tie_weights()
    config = hf_encoders.HFEncoderConfig.from_hf(hf.to_dict())
    return torch, model, config


def test_roberta_config_is_read_from_hugging_face(tiny_roberta):
    _, _, config = tiny_roberta
    assert (config.model_type, config.max_length, config.hidden_dropout, config.attention_dropout) == ("roberta", 18, 0.1, 0.1)
    assert config.classifier_dropout == 0.1  # classifier_dropout None: hidden_dropout_prob, as RobertaClassificationHead


def test_roberta_logits_match_pytorch_with_padding(tiny_roberta):
    torch, model, config = tiny_roberta
    ids, mask = batch(1, [18, 5, 11], 61)
    with torch.no_grad():
        theirs = model(input_ids=torch.tensor(ids), attention_mask=torch.tensor(mask)).logits.numpy()
    params = hf_encoders.params_from_state_dict(config, torch_state(model))
    ours = hf_encoders.HFEncoderForMaskedLM(config).apply({"params": params}, jnp.asarray(ids), jnp.asarray(mask))
    assert_logits_match(ours, theirs, mask, 1e-4)
    # One position per row, as BLiMP-NL scores
    positions = np.asarray([3, 0, 10])
    one = hf_encoders.HFEncoderForMaskedLM(config).apply({"params": params}, jnp.asarray(ids), jnp.asarray(mask), positions=jnp.asarray(positions))
    np.testing.assert_allclose(np.asarray(one), theirs[np.arange(3), positions], atol=1e-4)


def test_roberta_refuses_more_tokens_than_its_positions(tiny_roberta):
    torch, model, config = tiny_roberta
    params = hf_encoders.params_from_state_dict(config, torch_state(model))
    with pytest.raises(ValueError, match="more than the 18"):
        hf_encoders.HFEncoderForMaskedLM(config).apply({"params": params}, jnp.ones((1, 19), dtype=jnp.int32))


def test_roberta_conversion_uses_every_tensor(tiny_roberta):
    torch, model, config = tiny_roberta
    state = torch_state(model)
    state["roberta.pooler.dense.weight"] = np.zeros((32, 32), dtype=np.float32)
    with pytest.raises(ValueError, match="pooler"):
        hf_encoders.params_from_state_dict(config, state)
    state = torch_state(model)
    state["lm_head.decoder.weight"] = state["lm_head.decoder.weight"] + 1
    with pytest.raises(ValueError, match="not the word embeddings"):
        hf_encoders.params_from_state_dict(config, state)


# ---------------------------------------------------------------- NeoBERT


@pytest.fixture(scope="module")
def neobert_code():
    torch = pytest.importorskip("torch")
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    try:
        lm_class = get_class_from_dynamic_module("model.NeoBERTLMHead", ROBBERT_2026[0], revision=ROBBERT_2026[1])
    except OSError as error:  # offline
        pytest.skip(f"model.py of {ROBBERT_2026[0]} not available: {error}")
    return torch, lm_class


@pytest.fixture(scope="module")
def tiny_neobert(neobert_code):
    torch, lm_class = neobert_code
    hf = lm_class.config_class(hidden_size=32, num_hidden_layers=2, num_attention_heads=4, intermediate_size=48,
                               vocab_size=61, pad_token_id=1, max_length=64, norm_eps=1e-5)
    torch.manual_seed(0)
    model = lm_class(hf).eval()
    scramble(torch, model, 2)
    config = hf_encoders.HFEncoderConfig.from_hf(hf.to_dict())
    return torch, model, config


def test_neobert_config_is_read_from_hugging_face(tiny_neobert):
    _, model, config = tiny_neobert
    # int(2 * 48 / 3) = 32, a multiple of 8: the SwiGLU's hidden size of EncoderBlock
    assert (config.model_type, config.intermediate_size, config.classifier_dropout, config.hidden_dropout) == ("neobert", 32, 0.1, 0.0)
    assert tuple(model.model.transformer_encoder[0].ffn.w3.weight.shape) == (32, 32)


def test_neobert_logits_match_pytorch_with_padding(tiny_neobert):
    torch, model, config = tiny_neobert
    ids, mask = batch(1, [23, 7, 16], 61)
    with torch.no_grad():
        theirs = model(input_ids=torch.tensor(ids), attention_mask=torch.tensor(mask)).logits.numpy()
    params = hf_encoders.params_from_state_dict(config, torch_state(model))
    ours = hf_encoders.HFEncoderForMaskedLM(config).apply({"params": params}, jnp.asarray(ids), jnp.asarray(mask))
    assert_logits_match(ours, theirs, mask, 1e-4)


def test_neobert_rope_table_is_that_of_rotary_py(tiny_neobert):
    _, model, config = tiny_neobert
    table = model.model.freqs_cis_real.numpy()
    cos, sin = hf_encoders.rope_tables(config.head_dim, table.shape[0])
    np.testing.assert_allclose(cos, table[..., 0], atol=1e-6)
    np.testing.assert_allclose(sin, table[..., 1], atol=1e-6)


# ---------------------------------------------------------------- task heads in the fine-tuning scripts


def tiny_params_and_config(model_type):
    """Random encoder and MLM-head parameters of a tiny model, as convert_hf_encoder.py writes them."""
    config = hf_encoders.HFEncoderConfig(model_type=model_type, vocab_size=61, hidden_size=32, num_hidden_layers=2,
                                         num_attention_heads=4, intermediate_size=32, pad_token_id=1, norm_eps=1e-5,
                                         # 128 tokens: the dummy input of the scripts' model.init
                                         max_position_embeddings=130 if model_type == "roberta" else 0,
                                         hidden_dropout=0.1 if model_type == "roberta" else 0.0,
                                         attention_dropout=0.1 if model_type == "roberta" else 0.0,
                                         max_length=128)
    params = hf_encoders.HFEncoderForMaskedLM(config).init(jax.random.PRNGKey(0), jnp.ones((1, 8), dtype=jnp.int32))["params"]
    return jax.tree_util.tree_map(np.asarray, params), config


@pytest.mark.parametrize("model_type", hf_encoders.MODEL_TYPES)
def test_sequence_head_takes_the_encoder_and_trains(model_type):
    import finetune
    from flax import jax_utils

    pretrained, config = tiny_params_and_config(model_type)
    model, state = finetune.create_model_and_state(config, pretrained, num_labels=3, learning_rate=1e-3,
                                                   rng=jax.random.PRNGKey(42), gradient_accumulation_steps=1)
    assert isinstance(model, hf_encoders.HFEncoderForSequenceClassification)
    assert set(state.params) == {"model", "dense", "classifier"}
    for ours, theirs in zip(jax.tree_util.tree_leaves(state.params["model"]), jax.tree_util.tree_leaves(pretrained["model"])):
        np.testing.assert_array_equal(np.asarray(ours), theirs)
    ids, mask = batch(1, [12, 4], 61)
    rows = [{"input_ids": list(ids[i, :n]), "label": i} for i, n in enumerate([12, 4])] * jax.local_device_count()
    collated = finetune.collate_fn(rows, pad_token_id=1, max_length=16, pad_to_multiple_of=8)
    state, metrics = finetune.p_train_step(jax_utils.replicate(state), finetune.shard_batch(collated))
    assert np.isfinite(float(metrics["loss"][0]))


@pytest.mark.parametrize("model_type", hf_encoders.MODEL_TYPES)
@pytest.mark.parametrize("kind", ["token", "qa"])
def test_token_and_span_heads_take_the_encoder_and_train(model_type, kind):
    import finetune
    import finetune_extra
    from flax import jax_utils

    pretrained, config = tiny_params_and_config(model_type)
    del pretrained["lm_head" if model_type == "roberta" else "decoder"]  # the MLM head is not needed
    model, state = finetune_extra.create_model_and_state(kind, config, pretrained, 5 if kind == "token" else 2, 1e-3,
                                                         jax.random.PRNGKey(43))
    head = "classifier" if kind == "token" else "qa_outputs"
    assert set(state.params) == {"model", head}
    rows = [{"input_ids": [0, 7, 8, 9, 2], "labels": [-100, 1, 2, 3, -100], "start_position": 1, "end_position": 2}]
    collated = finetune_extra.collate(rows * jax.local_device_count(), kind, pad_token_id=1, max_length=16, pad_to_multiple_of=8)
    state, metrics = finetune_extra.p_train_step(jax_utils.replicate(state), finetune.shard_batch(collated))
    assert np.isfinite(float(metrics["loss"][0]))


def test_a_modernbert_checkpoint_still_needs_its_prediction_head():
    import finetune_extra

    from mmbjax.configuration_mmbbert import MMBertConfig

    with pytest.raises(ValueError, match="prediction_head"):
        finetune_extra.create_model_and_state("token", MMBertConfig(vocab_size=16, hidden_size=8, intermediate_size=16,
                                              num_hidden_layers=1, num_attention_heads=2), {"model": {}}, 3, 1e-3,
                                              jax.random.PRNGKey(0))


# ---------------------------------------------------------------- BLiMP-NL words of byte-level tokens


def test_bytelevel_words_start_as_basic_tokenizer_words():
    import eval_blimp_nl_hf as ebh
    from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode

    def is_punctuation(char):  # eval_blimp_nl.is_punctuation, without importing train_minimal
        import unicodedata

        code = ord(char)
        return 33 <= code <= 47 or 58 <= code <= 64 or 91 <= code <= 96 or 123 <= code <= 126 or unicodedata.category(char).startswith("P")

    encode = bytes_to_unicode()

    def token(text):
        return "".join(encode[b] for b in text.encode("utf-8"))

    # The pieces of "Het merendeel lijkt een ijzige kalmte, zegt 's-Hertogenbosch." as in test_eval_blimp_nl
    tokens = [token(t) for t in ["Het", " merendeel", " lijkt", " een", " ijz", "ige", " kalm", "te", ",", " zegt", " '", "s", "-",
                                 "Hertogenbosch", "."]]
    assert ebh.word_starts_bytelevel(tokens, is_punctuation) == [True, True, True, True, True, False, True, False, True, True,
                                                                 True, True, True, True, True]
    # é whole in one token continues the word; split over two tokens, its bytes continue it too
    assert ebh.word_starts_bytelevel([token(" caf"), token("é"), token(" is")], is_punctuation) == [True, False, True]
    raw = " café".encode("utf-8")
    split = ["".join(encode[b] for b in raw[:4]), "".join(encode[b] for b in raw[4:5]), "".join(encode[b] for b in raw[5:])]
    assert ebh.word_starts_bytelevel(split + [token(" is")], is_punctuation) == [True, False, False, True]
    # « is punctuation as a character, not as one of the bytes of ë (c3 ab, shown as Ã«)
    assert ebh.word_starts_bytelevel([token(" Belgi"), token("ë"), token(" «"), token("ja")], is_punctuation) == [True, False, True, True]
    # RobBERT-2023's pieces of "Amsterdam, 's-Hertogenbosch.": a bare space before each punctuation mark, in its word
    tokens = [token(t) for t in [" Amsterdam", " ", ",", " ", "'", " s", " ", "-", " Hertogenbosch", " ", "."]]
    assert ebh.word_starts_bytelevel(tokens, is_punctuation) == [True, True, False, True, False, True, True, False, True, True, False]


def test_robbert_2023_offsets_start_at_the_word_with_the_same_ids():
    pytest.importorskip("transformers")
    name, revision = ROBBERT_2023
    config = hf_encoders.HFEncoderConfig(model_type="roberta", vocab_size=50000, hidden_size=768, num_hidden_layers=1,
                                         num_attention_heads=12, intermediate_size=1, pad_token_id=1, norm_eps=1e-5,
                                         tokenizer_name=name, tokenizer_revision=revision)
    try:
        fixed = hf_encoders.load_tokenizer(config)
    except OSError as error:  # offline
        pytest.skip(f"tokenizer of {name} not available: {error}")
    raw = hf_encoders.load_tokenizer(hf_encoders.HFEncoderConfig(**{**config.to_dict(), "tokenizer_offsets": "tokenizer"}))
    text = "De hoofdstad is Amsterdam, zei hij."
    a, b = fixed(text, return_offsets_mapping=True), raw(text, return_offsets_mapping=True)
    assert a["input_ids"] == b["input_ids"]
    assert [text[x:y] for x, y in b["offset_mapping"][1:5]] == ["e", "oofdstad", "s", "msterdam"]  # one character late
    assert [text[x:y] for x, y in a["offset_mapping"][1:5]] == ["De", "hoofdstad", "is", "Amsterdam"]
    assert fixed.truncation_side == "left"


# ---------------------------------------------------------------- the real checkpoints (HF_ENCODER_REAL=1)


@pytest.mark.skipif(os.environ.get("HF_ENCODER_REAL") != "1", reason="HF_ENCODER_REAL=1 downloads both checkpoints (~1 GB)")
@pytest.mark.parametrize("repo", [ROBBERT_2023, ROBBERT_2026], ids=["robbert-2023", "robbert-2026"])
def test_real_checkpoint_logits_match_pytorch(repo):
    torch = pytest.importorskip("torch")
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForMaskedLM

    name, revision = repo
    model = AutoModelForMaskedLM.from_pretrained(name, revision=revision, trust_remote_code=name == ROBBERT_2026[0],
                                                 torch_dtype=torch.float32).eval()
    params, config = hf_encoders.load_hf_checkpoint(snapshot_download(name, revision=revision, allow_patterns=["config.json", "model.safetensors"]))
    config.tokenizer_name, config.tokenizer_revision = name, revision
    tokenizer = hf_encoders.load_tokenizer(config)
    sentences = ["De hoofdstad van Nederland is <mask>.", "Het boek was erg <mask>, ik raad het iedereen aan.",
                 "Gisteren liep de man met zijn hond door het park, terwijl het zachtjes begon te <mask> en de kinderen naar huis renden."]
    encoded = tokenizer(sentences, padding=True, return_tensors="np")
    ids, mask = encoded["input_ids"], encoded["attention_mask"]
    with torch.no_grad():
        theirs = model(input_ids=torch.tensor(ids), attention_mask=torch.tensor(mask)).logits.numpy()
    ours = hf_encoders.HFEncoderForMaskedLM(config).apply({"params": params}, jnp.asarray(ids), jnp.asarray(mask))
    assert_logits_match(ours, theirs, mask, 2e-3)
