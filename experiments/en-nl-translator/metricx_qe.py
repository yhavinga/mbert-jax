#!/usr/bin/env python3
"""MetricX-24 (hybrid, v2p6, Apache-2.0) in quality-estimation mode or with a reference, in JAX, on the 4 chips of one
TPU host (or on the CPU for the check).

    python metricx_qe.py score --pairs pairs.parquet --pair en:nl --pair nl:en --model xxl --out qe.parquet
    python metricx_qe.py score --pairs pairs.parquet --pair source:hypothesis:reference --pair source:hypothesis ...
    JAX_PLATFORMS=cpu python metricx_qe.py check --model large --pairs pairs.parquet --n 6

What google-research/metricx (metricx24/predict.py and models.py, commit fc4978e) does, rewritten for JAX: the input
is "source: <source> candidate: <candidate>" (QE: no reference) or "source: <source> candidate: <candidate>
reference: <reference>" (predict.py's _make_input with is_qe false; the hybrid models take both), tokenized by mT5's
tokenizer with at most 1,536 tokens, its final </s> removed; the model is mT5 (MT5ForConditionalGeneration's weights),
the encoder over the input and one decoder step from token 0; the score is the logit of token 250089 (<extra_id_10>)
of that step, clipped to [0, 25]. Lower is better (an MQM-like error score). A score per --pair of columns of --pairs:
--pair en:nl scores the Dutch column as a translation of the English one (QE, column qe_en_nl); --pair
source:hypothesis:reference scores hypothesis against reference as well (column ref_source_hypothesis_reference).

The weights are the bfloat16 releases (the README: within 0.0x of the float32 ones). On the TPU the matrix products
take bfloat16 inputs and accumulate in float32; norms, softmax and the residual stream are float32. The layers are
sharded over the host's 4 chips by heads and by the feed-forward dimension (tensor parallel). `check` compares this
code on the CPU in float32 against transformers' MT5ForConditionalGeneration in PyTorch, float32, on the same weights.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

MODELS = {
    "xxl": ("google/metricx-24-hybrid-xxl-v2p6-bfloat16", "3d9c6f7fd11b58d3eceeb68ee39fe7d2c761ee20"),
    "xl": ("google/metricx-24-hybrid-xl-v2p6-bfloat16", "1aa765b1705d36c85c9fab40e2f7ce1e0ff69705"),
    "large": ("google/metricx-24-hybrid-large-v2p6-bfloat16", "febb720e29a059df2e8af3ffd71dcdc9e0a24910"),
}
TOKENIZER = ("google/mt5-xl", "63fc6450d80515b48e026b69ef2fbbd426433e84")
SCORE_TOKEN = 250089  # <extra_id_10>
MAX_INPUT = 1536
LENGTHS = [128, 256, 384, 512, 768, 1024, 1536]  # inputs are padded to one of these
TOKENS_PER_BATCH = 16384


def make_input(source, candidate, reference=None):
    """MetricX-24's input: QE without a reference, the reference-based form with one"""
    if reference is None:
        return "source: " + source + " candidate: " + candidate
    return "source: " + source + " candidate: " + candidate + " reference: " + reference


def parse_pair(pair):
    """"src:cand" (QE) or "src:cand:ref" -> (src, cand, ref or None, output column)"""
    parts = pair.split(":")
    if len(parts) == 2:
        return parts[0], parts[1], None, f"qe_{parts[0]}_{parts[1]}"
    if len(parts) == 3:
        return parts[0], parts[1], parts[2], f"ref_{parts[0]}_{parts[1]}_{parts[2]}"
    raise SystemExit(f"--pair {pair!r}: source:candidate or source:candidate:reference")


def rows_to_score(table, src, cand, ref, empty_candidates):
    """The rows of table that get a score: a source (and a reference) that is not empty, and a candidate that is not
    empty, or with empty_candidates one that is not missing (an empty translation is scored as one)"""
    out = []
    for k, (s, c) in enumerate(zip(table[src], table[cand])):
        r = table[ref][k] if ref else "-"
        if s and r and (c is not None if empty_candidates else c):
            out.append(k)
    return out


def encode(tokenizer, texts):
    """Token ids per text as metricx24/predict.py makes them: truncated to MAX_INPUT, the last token (</s>) removed"""
    return [ids[:-1] for ids in tokenizer(texts, max_length=MAX_INPUT, truncation=True)["input_ids"]]


def download(model, root):
    from huggingface_hub import snapshot_download

    repo, revision = MODELS[model]
    path = Path(root) / repo.split("/")[1]
    snapshot_download(repo, revision=revision, local_dir=str(path))
    tok = Path(root) / "mt5-tokenizer"
    snapshot_download(TOKENIZER[0], revision=TOKENIZER[1], local_dir=str(tok),
                      allow_patterns=["spiece.model", "special_tokens_map.json", "tokenizer_config.json", "config.json"])
    return path, tok


def load_weights(path, dtype):
    """{name: numpy array} of the checkpoint's shards, as dtype (ml_dtypes.bfloat16 or float32)"""
    import torch

    files = sorted(Path(path).glob("pytorch_model*.bin"))
    if not files:
        raise SystemExit(f"no pytorch_model*.bin in {path}")
    out = {}
    for f in files:
        for name, t in torch.load(f, map_location="cpu", mmap=True, weights_only=True).items():
            out[name] = t.to(torch.float32).numpy().astype(dtype)
    return out


def stack_params(w, config):
    """The weights in the layout of forward(): stacked per layer, linear weights as (in, out)"""
    def lin(name):
        return w[name].T

    def stack(prefix, n, parts):
        return {key: np.stack([fn(f"{prefix}.block.{i}.{part}") for i in range(n)])
                for key, (part, fn) in parts.items()}

    attn = {"q": "SelfAttention.q.weight", "k": "SelfAttention.k.weight", "v": "SelfAttention.v.weight",
            "o": "SelfAttention.o.weight"}
    enc_parts = {k: (f"layer.0.{v}", lin) for k, v in attn.items()}
    enc_parts.update({"ln1": ("layer.0.layer_norm.weight", w.__getitem__),
                      "ln2": ("layer.1.layer_norm.weight", w.__getitem__),
                      "wi0": ("layer.1.DenseReluDense.wi_0.weight", lin),
                      "wi1": ("layer.1.DenseReluDense.wi_1.weight", lin),
                      "wo": ("layer.1.DenseReluDense.wo.weight", lin)})
    dec_parts = {"v": ("layer.0.SelfAttention.v.weight", lin), "o": ("layer.0.SelfAttention.o.weight", lin),
                 "ln1": ("layer.0.layer_norm.weight", w.__getitem__),
                 "cq": ("layer.1.EncDecAttention.q.weight", lin), "ck": ("layer.1.EncDecAttention.k.weight", lin),
                 "cv": ("layer.1.EncDecAttention.v.weight", lin), "co": ("layer.1.EncDecAttention.o.weight", lin),
                 "ln2": ("layer.1.layer_norm.weight", w.__getitem__),
                 "wi0": ("layer.2.DenseReluDense.wi_0.weight", lin),
                 "wi1": ("layer.2.DenseReluDense.wi_1.weight", lin),
                 "wo": ("layer.2.DenseReluDense.wo.weight", lin),
                 "ln3": ("layer.2.layer_norm.weight", w.__getitem__)}
    return {
        "embed": w["shared.weight"],
        "enc": stack("encoder", config["num_layers"], enc_parts),
        "enc_rel": w["encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"],
        "enc_ln": w["encoder.final_layer_norm.weight"],
        # one decoder step: its self-attention attends to itself alone, so only v and o matter
        "dec": stack("decoder", config["num_decoder_layers"], dec_parts),
        "dec_ln": w["decoder.final_layer_norm.weight"],
        "score_row": w["lm_head.weight"][SCORE_TOKEN],
    }


def relative_buckets(rel, bidirectional, num_buckets, max_distance):
    """T5's relative position buckets (transformers' T5Attention._relative_position_bucket)"""
    import jax.numpy as jnp

    out = jnp.zeros_like(rel)
    if bidirectional:
        num_buckets //= 2
        out = out + (rel > 0).astype(rel.dtype) * num_buckets
        n = jnp.abs(rel)
    else:
        n = -jnp.minimum(rel, 0)
    exact = num_buckets // 2
    large = exact + (jnp.log(jnp.maximum(n, 1).astype(jnp.float32) / exact) / math.log(max_distance / exact)
                     * (num_buckets - exact)).astype(rel.dtype)
    large = jnp.minimum(large, num_buckets - 1)
    return out + jnp.where(n < exact, n, large)


def forward(params, ids, mask, config, compute_dtype):
    """Scores of a batch: ids and mask (batch, length) -> (batch,) clipped to [0, 25]"""
    import jax
    import jax.numpy as jnp

    f32 = jnp.float32
    eps, heads, dk = config["layer_norm_epsilon"], config["num_heads"], config["d_kv"]

    def dense(x, w):
        return jnp.dot(x.astype(compute_dtype), w.astype(compute_dtype), preferred_element_type=f32)

    def norm(x, w):
        return x * jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + eps) * w.astype(f32)

    def ff(x, w):
        return dense(jax.nn.gelu(dense(x, w["wi0"]), approximate=True) * dense(x, w["wi1"]), w["wo"])

    def split(x):
        return x.reshape(x.shape[0], x.shape[1], heads, dk)

    def attend(q, k, v, bias, masked):
        scores = jnp.einsum("bqhd,bkhd->bhqk", q.astype(compute_dtype), k.astype(compute_dtype),
                            preferred_element_type=f32) + bias + masked
        p = jax.nn.softmax(scores, axis=-1)
        o = jnp.einsum("bhqk,bkhd->bqhd", p.astype(compute_dtype), v.astype(compute_dtype),
                       preferred_element_type=f32)
        return o.reshape(o.shape[0], o.shape[1], heads * dk)

    batch, length = ids.shape
    masked = (1.0 - mask.astype(f32))[:, None, None, :] * jnp.finfo(f32).min
    pos = jnp.arange(length, dtype=jnp.int32)
    buckets = relative_buckets(pos[None, :] - pos[:, None], True, config["relative_attention_num_buckets"],
                               config["relative_attention_max_distance"])
    bias = params["enc_rel"].astype(f32)[buckets].transpose(2, 0, 1)[None]  # (1, heads, length, length)

    def enc_layer(x, w):
        h = norm(x, w["ln1"])
        x = x + dense(attend(split(dense(h, w["q"])), split(dense(h, w["k"])), split(dense(h, w["v"])), bias, masked),
                      w["o"])
        return x + ff(norm(x, w["ln2"]), w), None

    x = params["embed"][ids].astype(f32)
    x, _ = jax.lax.scan(enc_layer, x, params["enc"])
    enc = norm(x, params["enc_ln"])

    def dec_layer(y, w):
        y = y + dense(dense(norm(y, w["ln1"]), w["v"]), w["o"])
        h = norm(y, w["ln2"])
        y = y + dense(attend(split(dense(h, w["cq"])), split(dense(enc, w["ck"])), split(dense(enc, w["cv"])), 0.0,
                             masked), w["co"])
        return y + ff(norm(y, w["ln3"]), w), None

    y = jnp.broadcast_to(params["embed"][0].astype(f32), (batch, 1, x.shape[-1]))
    y, _ = jax.lax.scan(dec_layer, y, params["dec"])
    y = norm(y, params["dec_ln"])[:, 0]
    return jnp.clip(y @ params["score_row"].astype(f32), 0.0, 25.0)


def shardings(params, mesh):
    """Tensor parallel over the mesh's axis m: heads (q, k, v by their output, o by its input) and the feed-forward
    dimension; everything else replicated"""
    import jax
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    out_cols = {"q", "k", "v", "cq", "ck", "cv", "wi0", "wi1"}
    in_rows = {"o", "co", "wo"}

    def spec(path, leaf):
        key = path[-1].key
        if key in out_cols and leaf.ndim == 3:
            return NamedSharding(mesh, P(None, None, "m"))
        if key in in_rows and leaf.ndim == 3:
            return NamedSharding(mesh, P(None, "m", None))
        return NamedSharding(mesh, P())

    return jax.tree_util.tree_map_with_path(spec, params)


class Scorer:
    def __init__(self, model, root, compute_dtype_name, devices=None):
        import jax
        import jax.numpy as jnp
        import ml_dtypes
        from jax.sharding import Mesh
        from transformers import AutoTokenizer

        path, tok = download(model, root)
        self.config = json.loads((path / "config.json").read_text())
        self.tokenizer = AutoTokenizer.from_pretrained(str(tok))
        compute_dtype = jnp.bfloat16 if compute_dtype_name == "bfloat16" else jnp.float32
        store = ml_dtypes.bfloat16 if compute_dtype_name == "bfloat16" else np.float32
        started = time.time()
        params = stack_params(load_weights(path, store), self.config)
        devices = devices or jax.devices()
        self.mesh = Mesh(np.array(devices), ("m",))
        self.params = jax.device_put(params, shardings(params, self.mesh))
        print(f"{model} loaded on {len(devices)} {devices[0].platform} device(s) in {time.time() - started:.0f} s",
              flush=True)
        config = self.config
        self.fn = jax.jit(lambda p, i, m: forward(p, i, m, config, compute_dtype))

    def score(self, sources, candidates, references=None):
        """MetricX-24 of each (source, candidate), in order: QE, or against references when they are given"""
        import jax

        references = references if references is not None else [None] * len(sources)
        ids = encode(self.tokenizer, [make_input(s, c, r) for s, c, r in zip(sources, candidates, references)])
        order = sorted(range(len(ids)), key=lambda i: len(ids[i]))
        out = np.zeros(len(ids), np.float32)
        k = 0
        while k < len(order):
            length = next(n for n in LENGTHS if n >= len(ids[order[k]]))
            batch = max(1, TOKENS_PER_BATCH // length)
            chunk = [i for i in order[k:k + batch] if len(ids[i]) <= length]
            tokens = np.zeros((batch, length), np.int32)
            mask = np.zeros((batch, length), np.int32)
            for row, i in enumerate(chunk):
                tokens[row, :len(ids[i])] = ids[i]
                mask[row, :len(ids[i])] = 1
            mask[len(chunk):, 0] = 1  # padding rows: one token, so no row is all masked
            scores = np.asarray(jax.device_get(self.fn(self.params, tokens, mask)))
            out[chunk] = scores[:len(chunk)]
            k += len(chunk)
        return out, [len(x) for x in ids]


def score(args):
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pq.read_table(args.pairs).to_pydict()
    scorer = Scorer(args.model, args.root, "bfloat16")
    out = {"id": table["id"]}
    for pair in args.pair:
        src, cand, ref, name = parse_pair(pair)
        rows = rows_to_score(table, src, cand, ref, args.empty_candidates)
        started = time.time()
        if rows:
            scores, lengths = scorer.score([table[src][k] for k in rows], [table[cand][k] for k in rows],
                                           [table[ref][k] for k in rows] if ref else None)
        else:
            # A worker's share can have no row for a pair (no reference in it): the tokenizer fails on an empty list,
            # so the column stays all None.
            scores, lengths = [], []
        column = [None] * len(table["id"])
        for k, s in zip(rows, scores):
            column[k] = round(float(s), 4)
        out[name] = column
        seconds = time.time() - started
        print(f"{'QE' if ref is None else 'REF'} {json.dumps({'pair': pair, 'rows': len(rows), 'of': len(column), 'seconds': round(seconds, 1), 'mean': round(float(np.mean(scores)), 3) if rows else None, 'truncated': sum(n >= MAX_INPUT - 1 for n in lengths)})}", flush=True)
    pq.write_table(pa.table(out), args.out)
    print(f"wrote {args.out}")


def check(args):
    """This code (CPU, float32) against transformers' MT5ForConditionalGeneration (PyTorch, float32)"""
    import jax
    import pyarrow.parquet as pq
    import torch
    from transformers import MT5ForConditionalGeneration

    table = pq.read_table(args.pairs).to_pydict()
    src, cand, ref, _ = parse_pair(args.pair[0])
    rows = rows_to_score(table, src, cand, ref, False)[:args.n]
    sources, candidates = [table[src][k] for k in rows], [table[cand][k] for k in rows]
    references = [table[ref][k] for k in rows] if ref else [None] * len(rows)
    scorer = Scorer(args.model, args.root, "float32", devices=jax.devices("cpu")[:1])
    mine, _ = scorer.score(sources, candidates, references)
    path = Path(args.root) / MODELS[args.model][0].split("/")[1]
    model = MT5ForConditionalGeneration.from_pretrained(str(path), dtype=torch.float32).eval()
    ids = encode(scorer.tokenizer, [make_input(s, c, r) for s, c, r in zip(sources, candidates, references)])
    ref = []
    with torch.no_grad():
        for x in ids:
            logits = model(input_ids=torch.tensor([x]), attention_mask=torch.ones(1, len(x), dtype=torch.long),
                           decoder_input_ids=torch.zeros(1, 1, dtype=torch.long)).logits
            ref.append(float(torch.clamp(logits[0, 0, SCORE_TOKEN], 0, 25)))
    diff = np.abs(np.asarray(ref) - mine)
    print(f"CHECK {json.dumps({'model': args.model, 'pair': args.pair[0], 'n': len(ids), 'jax': [round(float(v), 4) for v in mine], 'torch': [round(v, 4) for v in ref], 'max_abs_diff': round(float(diff.max()), 5)})}")
    if diff.max() > args.tolerance:
        raise SystemExit(f"the JAX scores differ from transformers' by {diff.max():.4f} > {args.tolerance}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("score", "check"):
        p = sub.add_parser(name)
        p.add_argument("--pairs", required=True, help="parquet with id and the text columns")
        p.add_argument("--pair", action="append", required=True,
                       help="source column:candidate column (QE), or source:candidate:reference column")
        p.add_argument("--model", default="xxl", choices=sorted(MODELS))
        p.add_argument("--root", default="/dev/shm/translator/metricx", help="Where the weights are downloaded")
        if name == "score":
            p.add_argument("--out", required=True)
            p.add_argument("--empty-candidates", action="store_true",
                           help="Score an empty candidate too (an evaluation set's empty translation); only a missing "
                                "one stays unscored")
        else:
            p.add_argument("--n", type=int, default=6)
            p.add_argument("--tolerance", type=float, default=0.01)
    args = parser.parse_args(argv)
    {"score": score, "check": check}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
