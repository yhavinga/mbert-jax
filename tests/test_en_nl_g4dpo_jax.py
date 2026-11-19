"""Arm G4-6 (draft amendment G4-6): the JAX parts of g4dpo.py on the CPU with a toy NNX model in MaxText's shape (a
decoder that sows its hidden states, logits_from_hidden_states_for_vocab_tiling, qwix-style LoRA parameters) and MaxText's
two helpers stubbed: the per-token log-probabilities in pieces, the reference as the model with its LoRA at zero, the loss
in MaxTextPeftTrainer's train step (jax.value_and_grad over the LoRA parameters only), and the reference computed before
training (amendment G4-6.4a) against the reference in the same step that it replaces. Needs JAX and Flax NNX (skipped
otherwise; run with the SFT venv or a JAX 0.11 / Flax 0.12 CPU venv)."""

import math
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
nnx = pytest.importorskip("flax.nnx")
import jax.numpy as jnp  # noqa: E402

# test_en_nl_g4arm6 puts experiments/en-nl-translator on the path (through test_en_nl_g4arm4)
import test_en_nl_g4arm6 as t6  # noqa: E402

g4dpo = t6.g4dpo
V, D, R = 41, 8, 2


@pytest.fixture
def maxtext_stub(monkeypatch):
    """maxtext.utils.max_utils.cross_entropy_with_logits and maxtext_utils.get_nested_value, as MaxText 0.2.5 has them
    (per-token cross entropy and z-loss in float32; a nested dict lookup)"""

    def cross_entropy_with_logits(logits, targets, z_loss=0.0):
        logits = logits.astype(jnp.float32)
        lse = jax.nn.logsumexp(logits, axis=-1)
        return lse - jnp.sum(targets * logits, axis=-1), z_loss * lse ** 2

    def get_nested_value(d, keys):
        for k in keys:
            d = d[k]
        return d

    maxtext, utils = types.ModuleType("maxtext"), types.ModuleType("maxtext.utils")
    mu, mxu = types.ModuleType("maxtext.utils.max_utils"), types.ModuleType("maxtext.utils.maxtext_utils")
    mu.cross_entropy_with_logits, mxu.get_nested_value = cross_entropy_with_logits, get_nested_value
    maxtext.utils, utils.max_utils, utils.maxtext_utils = utils, mu, mxu
    for name, module in (("maxtext", maxtext), ("maxtext.utils", utils), ("maxtext.utils.max_utils", mu),
                         ("maxtext.utils.maxtext_utils", mxu)):
        monkeypatch.setitem(sys.modules, name, module)


class LoRADense(nnx.Module):
    """A kernel with qwix's LoRA pair: x W + (x A) B x scale"""

    def __init__(self, din, dout, rngs):
        self.kernel = nnx.Param(jax.random.normal(rngs.params(), (din, dout)) * 0.4)
        self.kernel_lora_a = nnx.LoRAParam(jax.random.normal(rngs.params(), (din, R)) * 0.4)
        self.kernel_lora_b = nnx.LoRAParam(jnp.zeros((R, dout)))

    def __call__(self, x):
        return x @ self.kernel[...] + (x @ self.kernel_lora_a[...]) @ self.kernel_lora_b[...] * 0.5


class Decoder(nnx.Module):
    """Attention within a segment (causal, by position), one LoRA layer; sows its hidden states as MaxText's decoder
    does with num_vocab_tiling > 1"""

    def __init__(self, rngs):
        self.mix = LoRADense(D, D, rngs)
        self.out = LoRADense(D, D, rngs)

    def __call__(self, x, positions, segments):
        same = (segments[:, :, None] == segments[:, None, :]) & (positions[:, None, :] <= positions[:, :, None])
        weights = same / jnp.maximum(same.sum(-1, keepdims=True), 1)
        h = jnp.tanh(self.mix(jnp.einsum("bij,bjd->bid", weights, x)) + x)
        h = jnp.tanh(self.out(h) + h)
        self.sow(nnx.Intermediate, "hidden_states", h)


class Toy(nnx.Module):
    def __init__(self, rngs):
        self.token_embedder = nnx.Param(jax.random.normal(rngs.params(), (V, D)))
        self.decoder = Decoder(rngs)

    def __call__(self, decoder_input_tokens, decoder_positions, decoder_segment_ids=None, encoder_images=None,
                 encoder_image_masks=None, enable_dropout=False, decoder_target_tokens=None, decoder_target_mask=None):
        self.decoder(self.token_embedder[...][decoder_input_tokens], decoder_positions, decoder_segment_ids)
        return None

    def logits_from_hidden_states_for_vocab_tiling(self, hidden, deterministic, model_mode):
        assert model_mode == "train"
        return hidden @ self.token_embedder[...].T


CONFIG = SimpleNamespace(num_vocab_tiling=4, vocab_size=V, micro_batch_size_to_train_on=4,
                         micro_batch_size_to_eval_on=4, enable_dropout=False)


def toy_batch(seed=0, n=8, seq=64, rows=4):
    rng = np.random.default_rng(seed)
    pairs = []
    for k in range(n):
        prompt = [2] + rng.integers(5, V, size=int(rng.integers(3, 7))).tolist()
        pairs.append({"id": f"p{k}", "prompt": prompt, "chosen": rng.integers(5, V, size=int(rng.integers(2, 5))).tolist(),
                      "rejected": rng.integers(5, V, size=int(rng.integers(2, 5))).tolist()})
    packed = g4dpo.pack_rows(pairs, seq, rows)
    assert 1 + max(r["batch"] for r in packed) == 1, "one batch"
    return {k: jnp.asarray([r[k] for r in packed], jnp.int32) for k in g4dpo.KEYS}


def full_logp(model, data):
    """The reference computation: the whole logits at once, log_softmax, the targets' entries"""
    model(decoder_input_tokens=data["inputs"], decoder_positions=data["inputs_position"],
          decoder_segment_ids=data["inputs_segmentation"])
    h = nnx.pop(model, nnx.Intermediate).to_pure_dict()["decoder"]["hidden_states"][0]
    logits = model.logits_from_hidden_states_for_vocab_tiling(h, True, "train")
    lp = jnp.take_along_axis(jax.nn.log_softmax(logits, -1), data["targets"][..., None], -1)[..., 0]
    return lp * (data["targets_segmentation"] != 0)


def test_model_logp_in_pieces_is_the_whole_log_softmax_and_so_is_its_gradient(maxtext_stub):
    model = Toy(nnx.Rngs(0))
    data = toy_batch()
    np.testing.assert_allclose(g4dpo.model_logp(model, CONFIG, data, False), full_logp(model, data), rtol=1e-5,
                               atol=1e-6)
    assert not nnx.state(model, nnx.Intermediate)          # the sown hidden states are popped
    graphdef, lora, rest = nnx.split(model, nnx.LoRAParam, ...)

    def total(fn):
        return jax.grad(lambda lo: jnp.sum(fn(nnx.merge(graphdef, lo, rest, copy=True))))(lora)

    a = total(lambda m: g4dpo.model_logp(m, CONFIG, data, False))
    b = total(lambda m: full_logp(m, data))
    for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)):
        np.testing.assert_allclose(x, y, rtol=1e-4, atol=1e-6)


def test_without_lora_is_the_base_model_and_leaves_the_models_lora_alone(maxtext_stub):
    model = Toy(nnx.Rngs(1))
    model.decoder.mix.kernel_lora_b[...] = jnp.full((R, D), 0.3)
    data = toy_batch(1)
    ref = g4dpo.model_logp(g4dpo.without_lora(model), CONFIG, data, False)
    base = Toy(nnx.Rngs(1))                                  # the same seed: the same kernels, B = 0
    assert np.array_equal(ref, g4dpo.model_logp(base, CONFIG, data, False))
    assert not np.array_equal(ref, g4dpo.model_logp(model, CONFIG, data, False))
    assert float(model.decoder.mix.kernel_lora_b[...][0, 0]) == pytest.approx(0.3)


def peft_train_step(loss_fn, graphdef):
    """MaxTextPeftTrainer.create_train_step_fn's pattern: split off the LoRA parameters, jax.value_and_grad over them,
    the loss on the merged model; plain SGD for the test"""

    @jax.jit
    def step(lora, rest, data, lr):
        def wrapper(lora, rest):
            local = nnx.merge(graphdef, lora, rest)
            loss, aux = loss_fn(local, **data)
            _, _, new_rest = nnx.split(local, nnx.LoRAParam, ...)
            return loss, (aux, new_rest)

        (loss, (aux, new_rest)), grads = jax.value_and_grad(wrapper, has_aux=True)(lora, rest)
        return loss, aux, jax.tree.map(lambda p, g: p - lr * g, lora, grads), new_rest, grads

    return step


def test_the_loss_at_step_0_is_log_2_plus_the_nll_and_training_moves_only_the_lora(maxtext_stub):
    model = Toy(nnx.Rngs(2))
    data = toy_batch(2)
    opts = g4dpo.spec(beta=0.5, nll_alpha=0.1)
    # amendment G4-6.4a: the batch carries the reference, computed once before the first step
    data[g4dpo.REF_KEY] = g4dpo.reference_sums(model, CONFIG, data, opts["max_segments"])
    loss_fn = g4dpo.make_loss(CONFIG, opts, True)
    graphdef, lora, rest = nnx.split(model, nnx.LoRAParam, ...)
    step = peft_train_step(loss_fn, graphdef)
    loss, aux, new_lora, new_rest, grads = step(lora, rest, data, 0.0)
    lp = full_logp(Toy(nnx.Rngs(2)), data)
    seg = np.asarray(data["targets_segmentation"])
    chosen = (seg % 2 == 1)
    nll = -float(np.sum(np.where(chosen, lp, 0.0))) / int(chosen.sum())
    assert float(loss) == pytest.approx(math.log(2) + 0.1 * nll, rel=1e-5)
    assert float(aux["pref_pairs"]) == 8 and float(aux["pref_margin_sum"]) == 0.0
    assert float(aux["xent_sum"]) == pytest.approx(-float(np.sum(lp)), rel=1e-5)
    assert float(aux["total_weights"]) == int((seg != 0).sum())
    # B is zero, so only B gets a gradient at step 0 (d/dA of (x A) 0 is 0); every gradient is a LoRA parameter's
    flat = {"/".join(map(str, k)): v for k, v in jax.tree_util.tree_leaves_with_path(grads)}
    assert all("lora" in k for k in flat) and any(np.any(np.asarray(v)) for k, v in flat.items() if "lora_b" in k)
    first = {k: float(aux[k]) for k in ("pref_ref_chosen_logp_sum", "pref_ref_rejected_logp_sum")}
    for _ in range(60):
        loss, aux, lora, rest, _ = step(lora, rest, data, 0.5)
    # the preference moves (the chosen side's reward above the rejected's), the reference stays the base model
    assert float(aux["pref_correct"]) >= 7 and float(aux["pref_dpo_sum"]) / 8 < math.log(2) - 0.1
    assert {k: float(aux[k]) for k in first} == pytest.approx(first, rel=1e-6)
    kernels = nnx.state(nnx.merge(graphdef, lora, rest), nnx.Param)
    assert np.array_equal(kernels["decoder"]["mix"]["kernel"][...], model.decoder.mix.kernel[...])


def test_dpo_terms_is_the_same_in_numpy_and_jax():
    data = toy_batch(3)
    seg = np.asarray(data["targets_segmentation"])
    rng = np.random.default_rng(3)
    logp = np.where(seg != 0, -rng.uniform(0.1, 2.0, seg.shape), 0.0).astype(np.float32)
    ref = np.where(seg != 0, -rng.uniform(0.1, 2.0, seg.shape), 0.0).astype(np.float32)
    for opts in (g4dpo.spec(), g4dpo.spec(loss="ipo", length_norm=True), g4dpo.spec(label_smoothing=0.2)):
        a, sa = g4dpo.dpo_terms(logp, ref, seg, opts)
        b, sb = jax.jit(lambda x, y, s, o=opts: g4dpo.dpo_terms(x, y, s, o, jnp))(logp, ref, seg)
        assert float(b) == pytest.approx(float(a), rel=1e-5)
        assert {k: float(v) for k, v in sb.items()} == pytest.approx({k: float(v) for k, v in sa.items()}, rel=1e-5)


def test_bake_adapter_merges_into_the_base_kernels_and_the_reference_is_the_merged_model(tmp_path):
    # amendment G4-6b.2: sft_lora.py --base-adapter (KernelMerger with one component at coefficient 1)
    import sft_lora

    model = Toy(nnx.Rngs(0))
    w0 = {m: np.asarray(getattr(model.decoder, m).kernel[...]) for m in ("mix", "out")}
    rng = np.random.default_rng(1)
    adapter = {}
    for m in ("mix", "out"):
        adapter[f"decoder/{m}/kernel_lora_a"] = rng.normal(size=(D, R)).astype(np.float32)
        adapter[f"decoder/{m}/kernel_lora_b"] = rng.normal(size=(R, D)).astype(np.float32)
    np.savez(tmp_path / "adapter.npz", **adapter)
    (tmp_path / "adapter.json").write_text('{"rank": 2, "alpha": 1.0, "scale": 0.5}')
    # the same adapter as LoRA in the forward pass, on an unmerged copy
    lora_model = Toy(nnx.Rngs(0))
    for m in ("mix", "out"):
        layer = getattr(lora_model.decoder, m)
        layer.kernel_lora_a[...] = jnp.asarray(adapter[f"decoder/{m}/kernel_lora_a"])
        layer.kernel_lora_b[...] = jnp.asarray(adapter[f"decoder/{m}/kernel_lora_b"])
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]), ("x",))
    base = sft_lora.base_adapter_arg(f"G4-4={tmp_path / 'adapter.npz'}")
    record = sft_lora.bake_adapter(model, mesh, base)
    assert (record["name"], record["modules"], record["keys"], record["scale"]) == ("G4-4", 2, 4, 0.5)
    for m in ("mix", "out"):
        want = w0[m] + 0.5 * adapter[f"decoder/{m}/kernel_lora_a"] @ adapter[f"decoder/{m}/kernel_lora_b"]
        assert np.allclose(np.asarray(getattr(model.decoder, m).kernel[...]), want, atol=1e-6)
        assert not np.any(np.asarray(getattr(model.decoder, m).kernel_lora_b[...]))   # the run's own LoRA starts at 0
    x = jax.random.normal(jax.random.key(2), (1, 5, D))
    pos, seg = jnp.arange(5)[None], jnp.ones((1, 5), jnp.int32)

    def hidden(mdl):
        mdl.decoder(x, pos, seg)
        return np.asarray(nnx.pop(mdl, nnx.Intermediate).to_pure_dict()["decoder"]["hidden_states"][0])

    merged = hidden(model)
    assert np.allclose(merged, hidden(lora_model), atol=1e-5)
    # the reference of --dpo (the LoRA at zero) is the merged model, not the base
    assert np.allclose(hidden(g4dpo.without_lora(model)), merged, atol=1e-6)
    # an adapter module the model has no kernel for stops the run
    np.savez(tmp_path / "adapter.npz", **adapter, **{"decoder/gone/kernel_lora_a": adapter["decoder/mix/kernel_lora_a"],
                                                     "decoder/gone/kernel_lora_b": adapter["decoder/mix/kernel_lora_b"]})
    with pytest.raises(SystemExit, match="without exactly one base kernel"):
        sft_lora.bake_adapter(Toy(nnx.Rngs(0)), mesh, base)


def in_step_loss(config, opts):
    """The reference in the same step, as make_loss computed it before amendment G4-6.4a (without_lora under
    stop_gradient, then the policy): the loss the precomputed reference replaces"""
    micro = config.micro_batch_size_to_train_on

    def loss_func(model, **batch):
        data = {k: batch[k][:micro] for k in g4dpo.KEYS}
        ref = jax.lax.stop_gradient(g4dpo.model_logp(g4dpo.without_lora(model), config, data, False))
        logp = g4dpo.model_logp(model, config, data, False)
        loss, sums = g4dpo.dpo_terms(logp, ref, data["targets_segmentation"], opts, jnp)
        return loss, {"xent_sum": -jnp.sum(logp), **{f"pref_{k}": v for k, v in sums.items()}}

    return loss_func


def test_the_precomputed_reference_gives_the_in_step_references_loss_and_gradients(maxtext_stub):
    # amendment G4-6.4a: the reference's sums computed once before the first step (every LoRA B zero) give the same loss,
    # sums and gradients as the reference computed in each step, at step 0 and after the LoRA has moved
    model = Toy(nnx.Rngs(4))
    data = toy_batch(4)
    opts = g4dpo.spec()
    assert g4dpo.lora_b_zero(model)
    ref_seg = g4dpo.reference_sums(model, CONFIG, data, opts["max_segments"])
    base = g4dpo.segment_sums(g4dpo.model_logp(g4dpo.without_lora(model), CONFIG, data, False),
                              data["targets_segmentation"], opts["max_segments"], jnp)
    np.testing.assert_allclose(ref_seg, base, rtol=1e-6, atol=1e-6)
    graphdef, lora, rest = nnx.split(model, nnx.LoRAParam, ...)
    pre = peft_train_step(g4dpo.make_loss(CONFIG, opts, True), graphdef)
    old = peft_train_step(in_step_loss(CONFIG, opts), graphdef)
    with_ref = {**data, g4dpo.REF_KEY: ref_seg}
    for step in range(6):
        loss_a, aux_a, new_lora, new_rest, grads_a = pre(lora, rest, with_ref, 0.5)
        loss_b, aux_b, _, _, grads_b = old(lora, rest, data, 0.5)
        assert float(loss_a) == pytest.approx(float(loss_b), rel=1e-5, abs=1e-6), step
        for k in g4dpo.SUMS:
            assert float(aux_a[f"pref_{k}"]) == pytest.approx(float(aux_b[f"pref_{k}"]), rel=1e-5, abs=1e-5), (step, k)
        for x, y in zip(jax.tree_util.tree_leaves(grads_a), jax.tree_util.tree_leaves(grads_b)):
            np.testing.assert_allclose(x, y, rtol=1e-4, atol=1e-6)
        lora, rest = new_lora, new_rest
    moved = nnx.merge(graphdef, lora, rest)
    assert not g4dpo.lora_b_zero(moved) and float(aux_a["pref_margin_sum"]) != 0.0


def test_make_loss_needs_the_reference_in_training_and_gives_an_sft_files_loss_without_it(maxtext_stub):
    model = Toy(nnx.Rngs(5))
    data = toy_batch(5)
    with pytest.raises(ValueError, match="without ref_seg"):
        g4dpo.make_loss(CONFIG, g4dpo.spec(), True)(model, **data)
    loss, aux = g4dpo.make_loss(CONFIG, g4dpo.spec(), False)(model, **data)
    lp = full_logp(Toy(nnx.Rngs(5)), data)
    assert set(aux) == {"xent_sum", "total_weights"}
    assert float(aux["xent_sum"]) == pytest.approx(-float(np.sum(lp)), rel=1e-5)
    assert float(loss) == pytest.approx(-float(np.sum(lp)) / int((np.asarray(data["targets_segmentation"]) != 0).sum()),
                                        rel=1e-5)


def test_precompute_reference_places_each_row_by_its_index_and_refuses_a_moved_lora(maxtext_stub, monkeypatch,
                                                                                    tmp_path):
    # the reference pass over a packed file: rows come back in the mesh's order (here reversed within each batch) and
    # are put back in the file's order by their ROW_KEY; the sums are reference_sums of each batch
    rng = np.random.default_rng(6)
    pairs = []
    for k in range(20):
        prompt = [2] + rng.integers(5, V, size=int(rng.integers(3, 7))).tolist()
        pairs.append({"id": f"p{k}", "prompt": prompt, "chosen": rng.integers(5, V, size=int(rng.integers(2, 5))).tolist(),
                      "rejected": rng.integers(5, V, size=int(rng.integers(2, 5))).tolist()})
    packed = g4dpo.pack_rows(pairs, 32, 4)
    g4dpo.write_rows(packed, tmp_path / "rows.parquet")
    arrays, batch = g4dpo.read_packed(tmp_path / "rows.parquet")
    batches = g4dpo.batch_count(batch, 4)
    assert batches >= 2

    def fake_iterator(path, mesh, rows, n, padding=0, ref=None, row_ids=False):
        for b in g4dpo.local_batches(arrays, 0, 1, rows, n, padding, ref, row_ids):
            yield {k: jnp.asarray(v[::-1]) for k, v in b.items()}

    monkeypatch.setattr(g4dpo, "global_iterator", fake_iterator)
    model = Toy(nnx.Rngs(6))
    ref, seconds = g4dpo.precompute_reference(model, CONFIG, tmp_path / "rows.parquet", None, 4, batches, 64)
    assert ref.shape == (batches * 4, 64) and len(seconds) == batches
    for b in range(batches):
        data = {k: jnp.asarray(arrays[k][4 * b:4 * b + 4]) for k in g4dpo.KEYS}
        np.testing.assert_allclose(ref[4 * b:4 * b + 4], g4dpo.reference_sums(model, CONFIG, data, 64), rtol=1e-6,
                                   atol=1e-6)
    model.decoder.out.kernel_lora_b[...] = jnp.full((R, D), 0.1)
    assert not g4dpo.lora_b_zero(model)
    with pytest.raises(SystemExit, match="not zero before the reference pass"):
        g4dpo.precompute_reference(model, CONFIG, tmp_path / "rows.parquet", None, 4, batches, 64)


def test_the_precomputed_reference_after_the_bake_is_the_merged_model(maxtext_stub, tmp_path):
    # amendments G4-6b.2 and G4-6b.5a: the reference pass runs after --base-adapter's merge, so its sums are the merged
    # model's (every LoRA B still zero), not the base model's
    import sft_lora

    model = Toy(nnx.Rngs(7))
    data = toy_batch(7)
    base_sums = g4dpo.reference_sums(Toy(nnx.Rngs(7)), CONFIG, data, 64)
    rng = np.random.default_rng(8)
    adapter = {f"decoder/{m}/kernel_lora_{x}": rng.normal(size=(D, R) if x == "a" else (R, D)).astype(np.float32)
               for m in ("mix", "out") for x in ("a", "b")}
    np.savez(tmp_path / "adapter.npz", **adapter)
    (tmp_path / "adapter.json").write_text('{"rank": 2, "alpha": 1.0, "scale": 0.5}')
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]), ("x",))
    sft_lora.bake_adapter(model, mesh, sft_lora.base_adapter_arg(f"G4-4={tmp_path / 'adapter.npz'}"))
    assert g4dpo.lora_b_zero(model)
    ref = g4dpo.reference_sums(model, CONFIG, data, 64)
    merged = g4dpo.segment_sums(g4dpo.model_logp(g4dpo.without_lora(model), CONFIG, data, False),
                                data["targets_segmentation"], 64, jnp)
    np.testing.assert_allclose(ref, merged, rtol=1e-6, atol=1e-6)
    assert np.abs(np.asarray(ref) - np.asarray(base_sums)).max() > 1e-3
