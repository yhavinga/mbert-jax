"""RoBERTa and NeoBERT encoders in Flax with the weights of their Hugging Face checkpoints, so that
scripts/dumb/finetune.py, finetune_extra.py and eval_blimp_nl_hf.py evaluate RobBERT exactly as our ModernBERT
(experiments/robbert/protocol.md). transformers 5 has no Flax models, so both are written out here; the forward passes
are those of the PyTorch code, checked logit for logit in tests/test_hf_encoders.py.

- roberta (RobBERT-2023, DTAI-KULeuven/robbert-2023-dutch-base): transformers' RobertaModel. Learned positions that
  start at pad_token_id + 1 and skip padding (so at most max_position_embeddings - 2 = 512 tokens), post-LN layers
  with exact GELU, dropout in the embeddings, the attention probabilities and both residual branches.
- neobert (RobBERT-2026, pdelobelle/robbert-2026): the model.py of that repository. Pre-RMSNorm layers, one qkv
  projection packed per head as [q k v], RoPE on adjacent pairs of each head's dimensions, SwiGLU with the packed w12,
  a final RMSNorm, no dropout in the encoder.

Task heads are each model's own Hugging Face heads, new and drawn from a normal distribution with std 0.02 (bias 0):
- sequences: <s> -> dropout -> dense -> tanh -> dropout -> classifier (RobertaClassificationHead and
  NeoBERTForSequenceClassification have this same form);
- tokens: dropout -> classifier (RobertaForTokenClassification; NeoBERT ships no token head, so the same form);
- spans: one linear layer to start and end logits (qa_outputs of RobertaForQuestionAnswering and
  NeoBERTForQuestionAnswering).
The losses are those of our ModernBERT heads (mmbjax.heads), so the fine-tuning scripts treat both alike. The MLM heads
(HFEncoderForMaskedLM) serve BLiMP-NL only.
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from .heads.utils import cross_entropy_loss, cross_entropy_with_ignore_index
from .modeling_outputs import (
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)

MODEL_TYPES = ("roberta", "neobert")


@dataclasses.dataclass
class HFEncoderConfig:
    model_type: str
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    intermediate_size: int  # roberta: the FFN; neobert: the SwiGLU's hidden size (after its 2/3 and multiple of 8)
    pad_token_id: int
    norm_eps: float
    max_position_embeddings: int = 0  # roberta: rows of the position table
    type_vocab_size: int = 1
    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0
    classifier_dropout: float = 0.1
    initializer_range: float = 0.02
    rope_theta: float = 10000.0
    max_length: int = 512  # the longest input the fine-tunes may give it
    num_labels: int = 2
    model_name: str = ""
    model_revision: str = ""
    tokenizer_name: str = ""
    tokenizer_revision: str = ""
    # "fixed": offsets that start at the word (see load_tokenizer); "tokenizer": as the tokenizer gives them
    tokenizer_offsets: str = "fixed"

    def __post_init__(self):
        if self.model_type not in MODEL_TYPES:
            raise ValueError(f"model_type {self.model_type!r} is none of {MODEL_TYPES}")
        if self.hidden_size % self.num_attention_heads:
            raise ValueError(f"hidden_size {self.hidden_size} is not a multiple of {self.num_attention_heads} heads")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, values: dict) -> "HFEncoderConfig":
        fields = {field.name for field in dataclasses.fields(cls)}
        unknown = sorted(set(values) - fields)
        if unknown:
            raise ValueError(f"unknown fields of HFEncoderConfig: {unknown}")
        return cls(**values)

    @classmethod
    def from_hf(cls, hf: dict, *, model_name: str = "", model_revision: str = "") -> "HFEncoderConfig":
        """From the config.json of a Hugging Face checkpoint of either kind."""
        common = dict(model_name=model_name, model_revision=model_revision, tokenizer_name=model_name,
                      tokenizer_revision=model_revision, vocab_size=hf["vocab_size"], hidden_size=hf["hidden_size"],
                      num_hidden_layers=hf["num_hidden_layers"], num_attention_heads=hf["num_attention_heads"],
                      pad_token_id=hf["pad_token_id"])
        if hf["model_type"] == "roberta":
            if hf.get("hidden_act") != "gelu" or hf.get("position_embedding_type", "absolute") != "absolute":
                raise ValueError(f"a RoBERTa with {hf.get('hidden_act')} and {hf.get('position_embedding_type')} positions")
            dropout = hf["classifier_dropout"] if hf.get("classifier_dropout") is not None else hf["hidden_dropout_prob"]
            return cls(model_type="roberta", intermediate_size=hf["intermediate_size"], norm_eps=hf["layer_norm_eps"],
                       max_position_embeddings=hf["max_position_embeddings"], type_vocab_size=hf["type_vocab_size"],
                       hidden_dropout=hf["hidden_dropout_prob"], attention_dropout=hf["attention_probs_dropout_prob"],
                       classifier_dropout=dropout, initializer_range=hf["initializer_range"],
                       max_length=hf["max_position_embeddings"] - hf["pad_token_id"] - 1, **common)
        if hf["model_type"] == "neobert":
            # EncoderBlock of model.py: int(2 * intermediate_size / 3), rounded up to a multiple of 8
            ffn = int(2 * hf["intermediate_size"] / 3)
            ffn = 8 * ((ffn + 7) // 8)
            # NeoBERTForSequenceClassification: getattr(config, "classifier_dropout", 0.1)
            return cls(model_type="neobert", intermediate_size=ffn, norm_eps=hf["norm_eps"],
                       classifier_dropout=hf.get("classifier_dropout", 0.1),
                       initializer_range=hf.get("classifier_init_range", 0.02), max_length=512, **common)
        raise ValueError(f"model_type {hf['model_type']!r} is none of {MODEL_TYPES}")


def _dense(config: HFEncoderConfig, features: int, use_bias: bool = True, name: Optional[str] = None) -> nn.Dense:
    return nn.Dense(features, use_bias=use_bias, kernel_init=nn.initializers.normal(config.initializer_range),
                    bias_init=nn.initializers.zeros, name=name)


def _attention(q, k, v, mask, dropout: Optional[nn.Dropout] = None, deterministic: bool = True):
    """softmax(q k^T / sqrt(d)) v over the keys where mask is true; q, k, v (batch, length, heads, head_dim)."""
    scores = jnp.einsum("bqhd,bkhd->bhqk", q, k) / math.sqrt(q.shape[-1])
    scores = jnp.where(mask[:, None, None, :], scores, jnp.finfo(scores.dtype).min)
    probs = jax.nn.softmax(scores, axis=-1)
    if dropout is not None:
        probs = dropout(probs, deterministic=deterministic)
    return jnp.einsum("bhqk,bkhd->bqhd", probs, v).reshape(q.shape[0], q.shape[1], -1)


class RobertaLayer(nn.Module):
    config: HFEncoderConfig

    @nn.compact
    def __call__(self, x, mask, deterministic: bool = True):
        c = self.config
        shape = (*x.shape[:2], c.num_attention_heads, c.head_dim)
        q = _dense(c, c.hidden_size, name="query")(x).reshape(shape)
        k = _dense(c, c.hidden_size, name="key")(x).reshape(shape)
        v = _dense(c, c.hidden_size, name="value")(x).reshape(shape)
        context = _attention(q, k, v, mask, nn.Dropout(c.attention_dropout), deterministic)
        attended = nn.Dropout(c.hidden_dropout)(_dense(c, c.hidden_size, name="attention_output")(context), deterministic=deterministic)
        x = nn.LayerNorm(epsilon=c.norm_eps, name="attention_norm")(attended + x)
        h = jax.nn.gelu(_dense(c, c.intermediate_size, name="intermediate")(x), approximate=False)
        h = nn.Dropout(c.hidden_dropout)(_dense(c, c.hidden_size, name="output")(h), deterministic=deterministic)
        return nn.LayerNorm(epsilon=c.norm_eps, name="output_norm")(h + x)


def rope_tables(head_dim: int, length: int, theta: float = 10000.0):
    """cos and sin (length, head_dim / 2) of precompute_freqs_cis in RobBERT-2026's rotary.py, in float32 as there."""
    freqs = 1.0 / (np.float32(theta) ** (np.arange(0, head_dim, 2, dtype=np.float32)[: head_dim // 2] / np.float32(head_dim)))
    angles = np.outer(np.arange(length, dtype=np.float32), freqs.astype(np.float32)).astype(np.float32)
    return np.cos(angles), np.sin(angles)


def apply_rope(x, cos, sin):
    """apply_rotary_emb of rotary.py: dimensions (2i, 2i+1) of each head as one complex number, times e^(i angle)."""
    pairs = x.reshape(*x.shape[:-1], -1, 2)
    real, imag = pairs[..., 0], pairs[..., 1]
    cos, sin = cos[None, :, None, :], sin[None, :, None, :]
    return jnp.stack([real * cos - imag * sin, real * sin + imag * cos], axis=-1).reshape(x.shape)


class NeoBertLayer(nn.Module):
    config: HFEncoderConfig

    @nn.compact
    def __call__(self, x, mask, cos, sin):
        c = self.config
        h = nn.RMSNorm(epsilon=c.norm_eps, name="attention_norm")(x)
        qkv = _dense(c, 3 * c.hidden_size, use_bias=False, name="qkv")(h)
        q, k, v = jnp.split(qkv.reshape(*x.shape[:2], c.num_attention_heads, 3 * c.head_dim), 3, axis=-1)
        context = _attention(apply_rope(q, cos, sin), apply_rope(k, cos, sin), v, mask)
        x = x + _dense(c, c.hidden_size, use_bias=False, name="wo")(context)
        h = nn.RMSNorm(epsilon=c.norm_eps, name="ffn_norm")(x)
        gate, up = jnp.split(_dense(c, 2 * c.intermediate_size, use_bias=False, name="w12")(h), 2, axis=-1)
        return x + _dense(c, c.hidden_size, use_bias=False, name="w3")(jax.nn.silu(gate) * up)


class HFEncoder(nn.Module):
    """The encoder ("model" in every parameter tree here): (batch, length) ids to (batch, length, hidden) states."""

    config: HFEncoderConfig

    def setup(self):
        c = self.config
        init = nn.initializers.normal(c.initializer_range)
        self.word_embeddings = nn.Embed(c.vocab_size, c.hidden_size, embedding_init=init)
        if c.model_type == "roberta":
            self.position_embeddings = nn.Embed(c.max_position_embeddings, c.hidden_size, embedding_init=init)
            self.token_type_embeddings = nn.Embed(c.type_vocab_size, c.hidden_size, embedding_init=init)
            self.embeddings_norm = nn.LayerNorm(epsilon=c.norm_eps)
            self.embeddings_dropout = nn.Dropout(c.hidden_dropout)
            self.layers = [RobertaLayer(c) for _ in range(c.num_hidden_layers)]
        else:
            self.layers = [NeoBertLayer(c) for _ in range(c.num_hidden_layers)]
            self.final_norm = nn.RMSNorm(epsilon=c.norm_eps)

    def __call__(self, input_ids, attention_mask=None, deterministic: bool = True):
        c = self.config
        if input_ids.shape[1] > c.max_length:
            raise ValueError(f"{input_ids.shape[1]} tokens, more than the {c.max_length} of {c.model_name or c.model_type}")
        mask = jnp.ones(input_ids.shape, dtype=bool) if attention_mask is None else attention_mask > 0
        x = self.word_embeddings(input_ids)
        if c.model_type == "roberta":
            # create_position_ids_from_input_ids: positions count the non-padding tokens from pad_token_id + 1 on
            real = (input_ids != c.pad_token_id).astype(jnp.int32)
            positions = jnp.cumsum(real, axis=1) * real + c.pad_token_id
            x = x + self.token_type_embeddings(jnp.zeros_like(input_ids)) + self.position_embeddings(positions)
            x = self.embeddings_dropout(self.embeddings_norm(x), deterministic=deterministic)
            for layer in self.layers:
                x = layer(x, mask, deterministic)
            return x
        cos, sin = rope_tables(c.head_dim, input_ids.shape[1], c.rope_theta)
        for layer in self.layers:
            x = layer(x, mask, cos, sin)
        return self.final_norm(x)

    def attend(self, x):
        """x times the word embeddings transposed: RoBERTa's MLM decoder, tied to them."""
        return self.word_embeddings.attend(x)


class RobertaLMHead(nn.Module):
    config: HFEncoderConfig

    @nn.compact
    def __call__(self, x, attend):
        c = self.config
        x = nn.LayerNorm(epsilon=c.norm_eps, name="layer_norm")(jax.nn.gelu(_dense(c, c.hidden_size, name="dense")(x), approximate=False))
        return attend(x) + self.param("bias", nn.initializers.zeros, (c.vocab_size,))


class HFEncoderForMaskedLM(nn.Module):
    """Logits over the vocabulary, at every position or, with positions (batch,), at one position per row."""

    config: HFEncoderConfig

    def setup(self):
        self.model = HFEncoder(self.config)
        if self.config.model_type == "roberta":
            self.lm_head = RobertaLMHead(self.config)
        else:
            self.decoder = _dense(self.config, self.config.vocab_size)

    def __call__(self, input_ids, attention_mask=None, positions=None, deterministic: bool = True):
        hidden = self.model(input_ids, attention_mask, deterministic)
        if positions is not None:
            hidden = jnp.take_along_axis(hidden, positions[:, None, None], axis=1)[:, 0]
        if self.config.model_type == "roberta":
            return self.lm_head(hidden, self.model.attend)
        return self.decoder(hidden)


class HFEncoderForSequenceClassification(nn.Module):
    config: HFEncoderConfig

    @nn.compact
    def __call__(self, input_ids, *, attention_mask=None, labels=None, deterministic: bool = True):
        c = self.config
        hidden = HFEncoder(c, name="model")(input_ids, attention_mask, deterministic)
        x = nn.Dropout(c.classifier_dropout)(hidden[:, 0], deterministic=deterministic)
        x = jnp.tanh(_dense(c, c.hidden_size, name="dense")(x))
        x = nn.Dropout(c.classifier_dropout)(x, deterministic=deterministic)
        logits = _dense(c, c.num_labels, name="classifier")(x)
        loss = None if labels is None else cross_entropy_loss(logits, labels.astype(jnp.int32))
        return SequenceClassifierOutput(loss=loss, logits=logits)


class HFEncoderForTokenClassification(nn.Module):
    config: HFEncoderConfig

    @nn.compact
    def __call__(self, input_ids, *, attention_mask=None, labels=None, deterministic: bool = True):
        c = self.config
        hidden = HFEncoder(c, name="model")(input_ids, attention_mask, deterministic)
        logits = _dense(c, c.num_labels, name="classifier")(nn.Dropout(c.classifier_dropout)(hidden, deterministic=deterministic))
        loss = None
        if labels is not None:
            loss = cross_entropy_with_ignore_index(logits.reshape(-1, c.num_labels), labels.reshape(-1))
        return TokenClassifierOutput(loss=loss, logits=logits)


class HFEncoderForQuestionAnswering(nn.Module):
    config: HFEncoderConfig

    @nn.compact
    def __call__(self, input_ids, *, attention_mask=None, start_positions=None, end_positions=None, deterministic: bool = True):
        c = self.config
        hidden = HFEncoder(c, name="model")(input_ids, attention_mask, deterministic)
        start_logits, end_logits = jnp.split(_dense(c, 2, name="qa_outputs")(hidden), 2, axis=-1)
        start_logits, end_logits = start_logits[..., 0], end_logits[..., 0]
        loss = None
        if start_positions is not None and end_positions is not None:
            # The loss of ModernBertForQuestionAnswering
            loss = 0.5 * (cross_entropy_with_ignore_index(start_logits, start_positions.reshape(-1), ignore_index=-1)
                          + cross_entropy_with_ignore_index(end_logits, end_positions.reshape(-1), ignore_index=-1))
        return QuestionAnsweringModelOutput(loss=loss, start_logits=start_logits, end_logits=end_logits)


# ---------------------------------------------------------------- weights of a Hugging Face checkpoint


def _linear(state, name):
    out = {"kernel": state.pop(f"{name}.weight").T}
    if f"{name}.bias" in state:
        out["bias"] = state.pop(f"{name}.bias")
    return out


def _norm(state, name, scale_only=False):
    out = {"scale": state.pop(f"{name}.weight")}
    if not scale_only:
        out["bias"] = state.pop(f"{name}.bias")
    return out


def params_from_state_dict(config: HFEncoderConfig, state: dict) -> dict:
    """{"model": encoder, MLM head} in the trees of HFEncoderForMaskedLM, float32, from a PyTorch state dict of numpy
    arrays (RobertaForMaskedLM or NeoBERTLMHead). Every tensor must be used, except buffers that hold no weights."""
    state = {k: np.asarray(v, dtype=np.float32) for k, v in state.items()}
    if config.model_type == "roberta":
        state.pop("roberta.embeddings.position_ids", None)
        state.pop("roberta.embeddings.token_type_ids", None)
        # A RobertaForMaskedLM saved with its tied decoder: the same matrix as the word embeddings
        decoder = state.pop("lm_head.decoder.weight", None)
        if decoder is not None and not np.array_equal(decoder, state["roberta.embeddings.word_embeddings.weight"]):
            raise ValueError("lm_head.decoder.weight is not the word embeddings; this code ties them")
        decoder_bias = state.pop("lm_head.decoder.bias", None)
        if decoder_bias is not None and not np.array_equal(decoder_bias, state["lm_head.bias"]):
            raise ValueError("lm_head.decoder.bias is not lm_head.bias; this code ties them")
        e = "roberta.embeddings"
        model = {"word_embeddings": {"embedding": state.pop(f"{e}.word_embeddings.weight")},
                 "position_embeddings": {"embedding": state.pop(f"{e}.position_embeddings.weight")},
                 "token_type_embeddings": {"embedding": state.pop(f"{e}.token_type_embeddings.weight")},
                 "embeddings_norm": _norm(state, f"{e}.LayerNorm")}
        for i in range(config.num_hidden_layers):
            p = f"roberta.encoder.layer.{i}"
            model[f"layers_{i}"] = {
                "query": _linear(state, f"{p}.attention.self.query"), "key": _linear(state, f"{p}.attention.self.key"),
                "value": _linear(state, f"{p}.attention.self.value"),
                "attention_output": _linear(state, f"{p}.attention.output.dense"),
                "attention_norm": _norm(state, f"{p}.attention.output.LayerNorm"),
                "intermediate": _linear(state, f"{p}.intermediate.dense"), "output": _linear(state, f"{p}.output.dense"),
                "output_norm": _norm(state, f"{p}.output.LayerNorm")}
        head = {"lm_head": {"dense": _linear(state, "lm_head.dense"), "layer_norm": _norm(state, "lm_head.layer_norm"),
                            "bias": state.pop("lm_head.bias")}}
    else:
        table = state.pop("model.freqs_cis_real", None)
        if table is not None:
            cos, sin = rope_tables(config.head_dim, table.shape[0], config.rope_theta)
            if not (np.allclose(table[..., 0], cos, atol=1e-5) and np.allclose(table[..., 1], sin, atol=1e-5)):
                raise ValueError("model.freqs_cis_real is not the RoPE table of rotary.py with this head size and theta")
        model = {"word_embeddings": {"embedding": state.pop("model.encoder.weight")},
                 "final_norm": _norm(state, "model.layer_norm", scale_only=True)}
        for i in range(config.num_hidden_layers):
            p = f"model.transformer_encoder.{i}"
            model[f"layers_{i}"] = {"attention_norm": _norm(state, f"{p}.attention_norm", scale_only=True),
                                    "qkv": _linear(state, f"{p}.qkv"), "wo": _linear(state, f"{p}.wo"),
                                    "ffn_norm": _norm(state, f"{p}.ffn_norm", scale_only=True),
                                    "w12": _linear(state, f"{p}.ffn.w12"), "w3": _linear(state, f"{p}.ffn.w3")}
        head = {"decoder": _linear(state, "decoder")}
    if state:
        raise ValueError(f"tensors of the checkpoint that this code does not use: {sorted(state)}")
    return {"model": model, **head}


def load_hf_checkpoint(directory) -> tuple[dict, HFEncoderConfig]:
    """The parameters and config of a Hugging Face checkpoint directory with config.json and model.safetensors."""
    from safetensors.flax import load_file

    directory = Path(directory)
    hf = json.loads((directory / "config.json").read_text())
    config = HFEncoderConfig.from_hf(hf)
    state = {k: np.asarray(v.astype(jnp.float32)) for k, v in load_file(str(directory / "model.safetensors")).items()}
    return params_from_state_dict(config, state), config


OFFSET_PROBE = "De kat zat."


def load_tokenizer(config: HFEncoderConfig, *, split_into_words: bool = False):
    """The model's own tokenizer at its revision, truncating from the left as yhavinga/dutch-llama-tokenizer does
    (truncation_side "left" in its tokenizer_config.json), so a DBRD review longer than max_length keeps the same end
    for both. Pre-split words need a space before each word (add_prefix_space), as run_ner.py gives RoBERTa.

    Offsets: RobBERT-2023's tokenizer (pre-tokenizers Whitespace, ByteLevel with add_prefix_space, Punctuation) trims
    from every word's offsets a leading space that the text does not have, so each word starts one character late
    ("ĠDe" of "De kat" at (1, 2)); SQuAD-NL's answer positions and answer texts come from the offsets. When the probe
    shows that, the post-processor's trim_offsets is turned off: the same ids, offsets that start at the word. With
    config.tokenizer_offsets "tokenizer" they are left as they are (to measure what the shifted offsets cost)."""
    from transformers import AutoTokenizer

    extra = {"add_prefix_space": True} if split_into_words else {}

    def load(**more):
        return AutoTokenizer.from_pretrained(config.tokenizer_name, revision=config.tokenizer_revision or None,
                                             truncation_side="left", **extra, **more)

    def probe(tok):
        encoded = tok(OFFSET_PROBE, add_special_tokens=False, return_offsets_mapping=True)
        return encoded["input_ids"], [OFFSET_PROBE[a:b] for a, b in encoded["offset_mapping"]][:2]

    tokenizer = load()
    if not tokenizer.is_fast:
        raise ValueError(f"{config.tokenizer_name}: no fast tokenizer; word_ids and offsets need one")
    if config.tokenizer_offsets not in ("fixed", "tokenizer"):
        raise ValueError(f"tokenizer_offsets {config.tokenizer_offsets!r} is neither 'fixed' nor 'tokenizer'")
    ids, words = probe(tokenizer)
    if words != ["De", "kat"] and config.tokenizer_offsets == "fixed":
        tokenizer = load(trim_offsets=False)
        fixed_ids, fixed_words = probe(tokenizer)
        if fixed_ids != ids or fixed_words != ["De", "kat"]:
            raise ValueError(f"{config.tokenizer_name}: offsets of {OFFSET_PROBE!r} are {words}, and {fixed_words} "
                             "without trim_offsets")
    return tokenizer


__all__ = [
    "MODEL_TYPES",
    "HFEncoderConfig",
    "HFEncoder",
    "HFEncoderForMaskedLM",
    "HFEncoderForSequenceClassification",
    "HFEncoderForTokenClassification",
    "HFEncoderForQuestionAnswering",
    "params_from_state_dict",
    "load_hf_checkpoint",
    "load_tokenizer",
    "rope_tables",
]
