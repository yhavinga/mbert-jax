"""translate.py's LoRA merge against qwix's LoRA forward pass on MaxText's DenseGeneral, the layer of Gemma 3's
attention (query: one axis in, heads x head_dim out; out: heads x head_dim in) and MLP projections: the merged kernels
without LoRA give the output of the base kernels with the adapter. The adapter goes through sft_lora.py's export, as
after training. Needs JAX, Flax, qwix and MaxText (the SFT venv on a worker, JAX_PLATFORMS=cpu); skipped elsewhere."""

import sys
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
nnx = pytest.importorskip("flax.nnx")
qwix = pytest.importorskip("qwix")
linears = pytest.importorskip("maxtext.layers.linears")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "en-nl-translator"))
sys.path.insert(0, str(ROOT / "experiments" / "flan-nl"))

import sft_lora  # noqa: E402
import translate  # noqa: E402

RANK, ALPHA = 3, 6.0


class Attention(nnx.Module):
    def __init__(self, rngs):
        self.query = linears.DenseGeneral(in_features_shape=8, out_features_shape=(2, 4), axis=-1,
                                          kernel_axes=("embed", "q_heads", "kv"), rngs=rngs)
        self.out = linears.DenseGeneral(in_features_shape=(2, 4), out_features_shape=8, axis=(-2, -1),
                                        kernel_axes=("heads", "kv", "embed"), rngs=rngs)


class Mlp(nnx.Module):
    def __init__(self, rngs):
        self.wi_0 = linears.DenseGeneral(in_features_shape=8, out_features_shape=12, kernel_axes=("embed", "mlp"),
                                         rngs=rngs)
        self.wo = linears.DenseGeneral(in_features_shape=12, out_features_shape=8, kernel_axes=("mlp", "embed"),
                                       rngs=rngs)


class Layer(nnx.Module):
    def __init__(self, rngs):
        self.self_attention = Attention(rngs)
        self.mlp = Mlp(rngs)

    def __call__(self, x):
        h = x + self.self_attention.out(self.self_attention.query(x))
        return h + self.mlp.wo(jax.nn.gelu(self.mlp.wi_0(h)))


class Decoder(nnx.Module):
    def __init__(self, rngs):
        self.layers_0 = Layer(rngs)

    def __call__(self, x):
        return self.layers_0(x)


def test_merged_kernels_reproduce_qwix_lora():
    x = jax.random.normal(jax.random.key(1), (2, 3, 8))
    base_out = Decoder(nnx.Rngs(0))(x)
    provider = qwix.LoraProvider(module_path=r"layers_[0-9]+/.*(self_attention/(query|out)|mlp/(wi_0|wo))",
                                 rank=RANK, alpha=ALPHA)
    model = qwix.apply_lora_to_model(Decoder(nnx.Rngs(0)), provider, x, rngs=nnx.Rngs(3))
    # lora_b starts at zero; random values in every LoRA parameter make the adapter change the output
    key = jax.random.key(2)
    count = 0
    for _, variable in nnx.iter_graph(model):
        if isinstance(variable, nnx.LoRAParam):
            key, sub = jax.random.split(key)
            variable.set_value(0.3 * jax.random.normal(sub, variable.get_value().shape))
            count += 1
    assert count == 8
    lora_out = model(x)
    assert float(np.max(np.abs(np.asarray(lora_out - base_out)))) > 0.1

    adapter, dtypes = sft_lora.adapter_arrays(nnx.to_pure_dict(nnx.state(model, nnx.LoRAParam)))
    assert sorted(adapter) == sorted(f"layers_0/{m}/kernel_lora_{s}" for m in (
        "self_attention/query", "self_attention/out", "mlp/wi_0", "mlp/wo") for s in "ab")
    merged = Decoder(nnx.Rngs(0))
    state = nnx.state(merged, nnx.Param)
    tree = nnx.to_pure_dict(state)
    stats = translate.merge_lora(tree, adapter, ALPHA / RANK)
    assert stats["modules"] == 4 and stats["elements_changed"] > 0.9
    nnx.replace_by_pure_dict(state, tree)
    nnx.update(merged, state)
    np.testing.assert_allclose(np.asarray(merged(x)), np.asarray(lora_out), rtol=1e-4, atol=1e-4)


def test_init_adapter_puts_an_exported_adapter_back_into_a_fresh_model(tmp_path):
    """sft_lora.py --init-adapter (protocol-g4.md amendment G4-4.11): an adapter exported after training, loaded into a
    model whose lora_b is zero, gives the trained model's output; a missing key or another shape stops the load"""
    x = jax.random.normal(jax.random.key(1), (2, 3, 8))
    provider = qwix.LoraProvider(module_path=r"layers_[0-9]+/.*(self_attention/(query|out)|mlp/(wi_0|wo))",
                                 rank=RANK, alpha=ALPHA)
    trained = qwix.apply_lora_to_model(Decoder(nnx.Rngs(0)), provider, x, rngs=nnx.Rngs(3))
    key = jax.random.key(2)
    for _, variable in nnx.iter_graph(trained):
        if isinstance(variable, nnx.LoRAParam):
            key, sub = jax.random.split(key)
            variable.set_value(0.3 * jax.random.normal(sub, variable.get_value().shape))
    adapter, _ = sft_lora.adapter_arrays(nnx.to_pure_dict(nnx.state(trained, nnx.LoRAParam)))
    np.savez(tmp_path / "adapter.npz", **adapter)
    fresh = qwix.apply_lora_to_model(Decoder(nnx.Rngs(0)), provider, x, rngs=nnx.Rngs(4))
    assert float(np.max(np.abs(np.asarray(fresh(x) - trained(x))))) > 0.1
    info = sft_lora.load_adapter(fresh, tmp_path / "adapter.npz")
    assert info["keys"] == 8 and info["nonzero_b"] == 4 and info["params"] == sum(a.size for a in adapter.values())
    np.testing.assert_allclose(np.asarray(fresh(x)), np.asarray(trained(x)), rtol=1e-6, atol=1e-6)
    np.savez(tmp_path / "missing.npz", **{k: v for k, v in adapter.items() if not k.endswith("wo/kernel_lora_b")})
    with pytest.raises(SystemExit, match="no layers_0/mlp/wo/kernel_lora_b"):
        sft_lora.load_adapter(qwix.apply_lora_to_model(Decoder(nnx.Rngs(0)), provider, x, rngs=nnx.Rngs(4)),
                              tmp_path / "missing.npz")
    np.savez(tmp_path / "shape.npz", **{**adapter, "layers_0/mlp/wo/kernel_lora_a": np.zeros((12, RANK + 1), np.float32)})
    with pytest.raises(SystemExit, match="is \\(12, 4\\)"):
        sft_lora.load_adapter(qwix.apply_lora_to_model(Decoder(nnx.Rngs(0)), provider, x, rngs=nnx.Rngs(4)),
                              tmp_path / "shape.npz")
    np.savez(tmp_path / "extra.npz", **{**adapter, "layers_9/mlp/wo/kernel_lora_a": np.zeros((12, RANK), np.float32)})
    with pytest.raises(SystemExit, match="no LoRA parameter for"):
        sft_lora.load_adapter(qwix.apply_lora_to_model(Decoder(nnx.Rngs(0)), provider, x, rngs=nnx.Rngs(4)),
                              tmp_path / "extra.npz")
