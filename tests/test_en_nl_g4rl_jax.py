"""Arm G4-7 (spec translator-sft/g4-rl-spec.md §4.3, §8.1 points 1 and 2): g4rl.py's loss and passes on the CPU.

Equal to Tunix @ a8d7058 (the version MaxText 0.2.5 pins): the advantage is compute_drgrpo_advantages ("drgrpo"); the
KL term at rho = 1 is compute_kl_divergence(..., "mse_kl"); the policy term is grpo_loss_fn's with the same inputs (a toy
model in Tunix's signature, Tunix computing its own log-probabilities); the aggregation is aggregate_loss(...,
"token-mean") with segment_ids (packed rows). Tunix comes from the venv (MaxText's SFT venv has it) or, with TUNIX_SRC (a
source tree of that commit), from there without its package __init__, which imports Tunix's whole training stack.

The properties of §8.1 point 2 with test_en_nl_g4dpo_jax's toy model in MaxText's shape: at theta = theta_old = theta_ref
rho is 1, the KL 0 and the loss minus the token mean of A; the gradient of the k2 term is beta rho (log pi - log
pi_ref) grad log pi; only LoRA parameters change; the two passes give per token the base model's log-probabilities and
the loaded model's. Needs JAX and Flax NNX (skipped otherwise; run with the SFT venv or a JAX 0.11 / Flax 0.12 CPU
venv)."""

import os
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
nnx = pytest.importorskip("flax.nnx")
import jax.numpy as jnp  # noqa: E402
import test_en_nl_g4dpo_jax as tj  # noqa: E402

maxtext_stub = tj.maxtext_stub          # the fixture: MaxText's two helpers the toy model needs
g4dpo = tj.g4dpo
import g4rl  # noqa: E402

CONFIG = tj.CONFIG
V = tj.V


def tunix_rl():
    """(algo_core, common) of Tunix"""
    src = os.environ.get("TUNIX_SRC")
    if src and "tunix" not in sys.modules:
        for name, sub in (("tunix", "tunix"), ("tunix.rl", "tunix/rl"), ("tunix.sft", "tunix/sft"),
                          ("tunix.oss", "tunix/oss")):
            module = types.ModuleType(name)
            module.__path__ = [os.path.join(src, sub)]
            sys.modules[name] = module
    try:
        from tunix.rl import algo_core, common
    except ImportError as missing:
        pytest.skip(f"no Tunix ({missing}); MaxText's SFT venv has it, or set TUNIX_SRC")
    return algo_core, common


# ---------------------------------------------------------------------------------------------------- Tunix

def test_the_advantage_is_tunixs_drgrpo():
    algo_core, _ = tunix_rl()
    rewards = np.random.default_rng(0).normal(size=5 * 8) * 3
    rewards[8:16] = -10.0
    tunix = np.asarray(algo_core.compute_drgrpo_advantages(jnp.asarray(rewards), 8))
    ours = np.concatenate([g4rl.advantages7(rewards[k:k + 8])[0] for k in range(0, 40, 8)])
    np.testing.assert_allclose(ours, tunix, rtol=1e-6, atol=1e-6)
    assert not g4rl.advantages7(rewards[8:16])[1]
    # GRPO's own advantage divides by the group's std (ddof 1): not ours
    grpo = np.asarray(algo_core.compute_advantages(rewards, 8))
    assert not np.allclose(grpo[:8], ours[:8])


def toy_logps(seed=0, shape=(4, 12)):
    rng = np.random.default_rng(seed)
    mask = rng.random(shape) < 0.7
    logp = np.where(mask, -rng.uniform(0.05, 4.0, shape), 0.0).astype(np.float32)
    return rng, mask, logp


def test_the_kl_term_at_rho_1_is_tunixs_mse_kl_and_the_aggregation_its_token_mean_with_segments():
    algo_core, common = tunix_rl()
    rng, mask, logp = toy_logps(1)
    ref = np.where(mask, logp + rng.normal(scale=0.3, size=mask.shape), 0.0).astype(np.float32)
    kl = np.asarray(common.compute_kl_divergence(jnp.asarray(logp), jnp.asarray(ref), "mse_kl"))
    opts = g4rl.grpo_spec()
    zero = np.zeros_like(logp)
    loss, sums = g4rl.grpo_terms(logp, logp, ref, zero, mask, opts)
    assert float(sums["kl_sum"]) == pytest.approx(float((kl * mask).sum()), rel=1e-6)
    # packed rows: segment ids per sample, token-mean over every target token of the batch
    seg = np.cumsum(rng.random(mask.shape) < 0.25, axis=1) + 1
    agg = common.aggregate_loss(jnp.asarray(kl), jnp.asarray(mask.astype(np.float32)), "token-mean",
                                segment_ids=jnp.asarray(seg), num_segments=int(seg.max()) + 1)
    assert float(loss) == pytest.approx(opts["beta"] * float(agg.compute()), rel=1e-6)
    plain = common.aggregate_loss(jnp.asarray(kl), jnp.asarray(mask.astype(np.float32)), "token-mean")
    assert float(agg.compute()) == pytest.approx(float(plain.compute()), rel=1e-6)


class Bigram(nnx.Module):
    """A model in Tunix's signature: the logits of a position are a row of a table, by the position's token"""

    def __init__(self, rngs):
        self.table = nnx.Param(jax.random.normal(rngs.params(), (V, V)) * 2.0)

    def __call__(self, input_tokens, positions, cache, attention_mask):
        return self.table[...][input_tokens], None


def bigram_logps(table, prompt, completion):
    """Per completion token: log_softmax of the table's row of the token before it"""
    seq = np.concatenate([prompt, completion], axis=1)
    rows = table[seq[:, prompt.shape[1] - 1:-1]]
    rows = rows - rows.max(-1, keepdims=True)
    lsm = rows - np.log(np.exp(rows).sum(-1, keepdims=True))
    return np.take_along_axis(lsm, completion[..., None], -1)[..., 0].astype(np.float32)


@pytest.mark.parametrize("beta", [0.0, 0.04])
def test_the_loss_is_tunixs_grpo_loss_fn_with_the_same_inputs(beta):
    algo_core, common = tunix_rl()
    model = Bigram(nnx.Rngs(0))
    table = np.asarray(model.table[...], np.float64)
    rng = np.random.default_rng(2)
    b, p, c = 4, 5, 9
    prompt = rng.integers(3, V, size=(b, p)).astype(np.int32)
    completion = rng.integers(3, V, size=(b, c)).astype(np.int32)
    mask = np.ones((b, c), np.float32)
    mask[0, 6:] = mask[2, 3:] = 0.0
    completion = np.where(mask > 0, completion, 0).astype(np.int32)
    logp = bigram_logps(table, prompt, completion) * mask
    adv = rng.normal(size=b).astype(np.float32)
    if beta:
        old = logp.copy()                                        # rho = 1: Tunix's mse_kl is our k2 term
    else:
        old = np.where(mask > 0, logp + rng.normal(scale=0.4, size=logp.shape), 0.0).astype(np.float32)
    ref = np.where(mask > 0, logp + rng.normal(scale=0.3, size=logp.shape), 0.0).astype(np.float32)
    example = common.TrainExample(prompt_ids=jnp.asarray(prompt), prompt_mask=jnp.ones((b, p), jnp.int32),
                                  completion_ids=jnp.asarray(completion), completion_mask=jnp.asarray(mask),
                                  advantages=jnp.asarray(adv), ref_per_token_logps=jnp.asarray(ref),
                                  old_per_token_logps=jnp.asarray(old))
    config = SimpleNamespace(beta=beta, epsilon=0.2, epsilon_high=0.2, epsilon_c=None, loss_algo="grpo",
                             loss_agg_mode="token-mean", temperature=1.0, kl_loss_mode="mse_kl", kl_clamp_value=None)
    out = algo_core.grpo_loss_fn(model, example, config, pad_id=0, eos_id=1)
    tunix = float(out.primary_loss.compute())
    ours, sums = g4rl.grpo_terms(logp, old, ref, adv[:, None] * mask, mask > 0, g4rl.grpo_spec(beta=beta))
    assert float(ours) == pytest.approx(tunix, rel=2e-5, abs=1e-6)
    tokens = float(mask.sum())
    assert float(sums["clip_sum"]) / tokens == pytest.approx(float(out.aux_metrics["pg_clipfrac"].compute()), abs=1e-6)
    if not beta:
        assert float(sums["clip_sum"]) > 0                      # the clip was in play


# ---------------------------------------------------------------------------------------------------- the toy model

def grpo_batch(seed=0, n=10, seq=64, rows=4):
    """One packed batch of random samples (iter7's rows) with their advantages and sampler log-probabilities"""
    rng = np.random.default_rng(seed)
    units = []
    for k in range(n):
        completion = rng.integers(5, V, size=int(rng.integers(2, 6))).tolist()
        units.append({"uid": f"u{k}", "prompt": [2] + rng.integers(5, V, size=int(rng.integers(3, 6))).tolist(),
                      "completion": completion, "adv": float(rng.normal()),
                      "sampler_logp": (-rng.uniform(0.1, 3.0, size=len(completion))).tolist()})
    packed = g4rl.pack_rows7(units, seq, rows)
    assert 1 + max(r["batch"] for r in packed) == 1, "one batch"
    batch = {k: jnp.asarray([r[k] for r in packed], jnp.int32) for k in g4dpo.KEYS}
    batch.update({k: jnp.asarray([r[k] for r in packed], jnp.float32) for k in g4rl.TOKEN_KEYS})
    batch[g4rl.ROW_KEY] = jnp.arange(rows, dtype=jnp.int32)
    return batch


def with_passes(model, batch):
    ref, old, _ = g4rl.pass_step(CONFIG)(model, batch)
    return {**{k: v for k, v in batch.items() if k != g4rl.ROW_KEY}, "ref_logp": ref, "old_logp": old}


def test_the_passes_give_the_base_models_and_the_loaded_models_logprobs(maxtext_stub):
    model = tj.Toy(nnx.Rngs(3))
    model.decoder.mix.kernel_lora_b[...] = jnp.full((tj.R, tj.D), 0.2)
    batch = grpo_batch(3)
    ref, old, rows = g4rl.pass_step(CONFIG)(model, batch)
    data = {k: batch[k] for k in g4dpo.KEYS}
    base = g4dpo.model_logp(tj.Toy(nnx.Rngs(3)), CONFIG, data, False)            # the same seed, every B zero
    np.testing.assert_allclose(ref, base, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(old, g4dpo.model_logp(model, CONFIG, data, False), rtol=1e-6, atol=1e-6)
    assert np.abs(np.asarray(old) - np.asarray(ref)).max() > 1e-3 and rows.tolist() == [0, 1, 2, 3]
    on = np.asarray(batch["targets_segmentation"]) != 0
    assert not np.asarray(ref)[~on].any() and not np.asarray(old)[~on].any()
    assert float(model.decoder.mix.kernel_lora_b[...][0, 0]) == pytest.approx(0.2)   # the model's LoRA left alone


def test_at_theta_old_ref_rho_is_1_kl_0_and_the_loss_minus_the_token_mean_of_a(maxtext_stub):
    model = tj.Toy(nnx.Rngs(4))
    batch = with_passes(model, grpo_batch(4))
    on = np.asarray(batch["targets_segmentation"]) != 0
    assert np.array_equal(np.asarray(batch["ref_logp"]), np.asarray(batch["old_logp"]))   # every LoRA B is zero
    opts = g4rl.grpo_spec()
    graphdef, lora, rest = nnx.split(model, nnx.LoRAParam, ...)
    step = tj.peft_train_step(g4rl.make_grpo_loss(CONFIG, opts, True), graphdef)
    loss, aux, new_lora, new_rest, grads = step(lora, rest, batch, 0.0)
    want = -float(np.asarray(batch["adv"])[on].mean())
    assert float(loss) == pytest.approx(want, rel=1e-5, abs=1e-7)
    assert float(aux["grpo_log_ratio_abs_max"]) <= 1e-6 and abs(float(aux["grpo_kl_sum"])) <= 1e-10
    assert float(aux["grpo_tokens"]) == on.sum() and float(aux["grpo_clip_sum"]) == 0.0
    # a moved LoRA: theta = theta_old (the passes again), theta_ref the base: rho 1, the KL the k2 of old against ref
    model.decoder.out.kernel_lora_b[...] = jnp.full((tj.R, tj.D), 0.3)
    batch = with_passes(model, grpo_batch(4))
    graphdef, lora, rest = nnx.split(model, nnx.LoRAParam, ...)
    loss, aux, *_ = tj.peft_train_step(g4rl.make_grpo_loss(CONFIG, opts, True), graphdef)(lora, rest, batch, 0.0)
    diff = (np.asarray(batch["old_logp"]) - np.asarray(batch["ref_logp"]))[on]
    assert float(aux["grpo_log_ratio_abs_max"]) <= 1e-6 and np.abs(diff).max() > 1e-3
    assert float(loss) == pytest.approx(want + 0.04 * float((0.5 * diff ** 2).mean()), rel=1e-5, abs=1e-7)


def test_the_k2_terms_gradient_is_beta_rho_times_the_log_ratio_to_the_reference_times_grad_log_pi(maxtext_stub):
    model = tj.Toy(nnx.Rngs(5))
    model.decoder.mix.kernel_lora_b[...] = jnp.full((tj.R, tj.D), 0.25)
    batch = with_passes(model, grpo_batch(5))
    on = np.asarray(batch["targets_segmentation"]) != 0
    rng = np.random.default_rng(5)
    # old off the policy (rho != 1), no advantage: only the KL term moves the model
    batch = {**batch, "adv": jnp.zeros_like(batch["adv"]),
             "old_logp": batch["old_logp"] + jnp.asarray(np.where(on, rng.normal(scale=0.3, size=on.shape), 0.0),
                                                        jnp.float32)}
    opts = g4rl.grpo_spec(beta=0.5)
    graphdef, lora, rest = nnx.split(model, nnx.LoRAParam, ...)
    _, _, _, _, grads = tj.peft_train_step(g4rl.make_grpo_loss(CONFIG, opts, True), graphdef)(lora, rest, batch, 0.0)
    data = {k: batch[k] for k in g4dpo.KEYS}
    lp = np.asarray(g4dpo.model_logp(model, CONFIG, data, False))
    rho = np.exp(np.clip(lp - np.asarray(batch["old_logp"]), -20, 20))
    weight = jnp.asarray(np.where(on, 0.5 * rho * (lp - np.asarray(batch["ref_logp"])), 0.0) / on.sum(), jnp.float32)

    def surrogate(lo):
        m = nnx.merge(graphdef, lo, rest, copy=True)
        return jnp.sum(weight * g4dpo.model_logp(m, CONFIG, data, False))

    want = jax.grad(surrogate)(lora)
    flat_g, flat_w = jax.tree_util.tree_leaves(grads), jax.tree_util.tree_leaves(want)
    assert any(np.abs(np.asarray(x)).max() > 1e-6 for x in flat_w)
    for x, y in zip(flat_g, flat_w):
        np.testing.assert_allclose(x, y, rtol=1e-4, atol=1e-7)


def test_training_moves_only_the_lora_and_raises_the_likelihood_of_the_better_samples(maxtext_stub):
    model = tj.Toy(nnx.Rngs(6))
    batch = with_passes(model, grpo_batch(6, n=12))
    on = np.asarray(batch["targets_segmentation"]) != 0
    opts = g4rl.grpo_spec(epsilon=0.99)                       # a wide clip: 30 steps of SGD stay inside it
    graphdef, lora, rest = nnx.split(model, nnx.LoRAParam, ...)
    step = tj.peft_train_step(g4rl.make_grpo_loss(CONFIG, opts, True), graphdef)
    _, aux0, _, _, grads = step(lora, rest, batch, 0.0)
    flat = {"/".join(map(str, k)): v for k, v in jax.tree_util.tree_leaves_with_path(grads)}
    assert all("lora" in k for k in flat) and any(np.any(np.asarray(v)) for k, v in flat.items() if "lora_b" in k)
    first = float(aux0["grpo_loss_sum"]) / float(aux0["grpo_tokens"])
    for _ in range(30):
        loss, aux, lora, rest, _ = step(lora, rest, batch, 0.5)
    assert float(loss) < first - 0.05
    moved = nnx.merge(graphdef, lora, rest)
    kernels = nnx.state(moved, nnx.Param)
    assert np.array_equal(kernels["decoder"]["mix"]["kernel"][...], model.decoder.mix.kernel[...])
    assert not g4dpo.lora_b_zero(moved)
    # the samples with A > 0 became likelier, those with A < 0 less likely (against the old policy)
    data = {k: batch[k] for k in g4dpo.KEYS}
    delta = np.asarray(g4dpo.model_logp(moved, CONFIG, data, False)) - np.asarray(batch["old_logp"])
    adv = np.asarray(batch["adv"])
    assert delta[on & (adv > 0)].mean() > 0 > delta[on & (adv < 0)].mean()


def test_make_grpo_loss_needs_the_passes_in_training_and_gives_an_sft_files_loss_without_them(maxtext_stub):
    model = tj.Toy(nnx.Rngs(7))
    batch = grpo_batch(7)
    data = {k: batch[k] for k in g4dpo.KEYS}
    with pytest.raises(ValueError, match="without adv"):
        g4rl.make_grpo_loss(CONFIG, g4rl.grpo_spec(), True)(model, **data)
    with pytest.raises(ValueError, match="without ref_logp"):
        g4rl.make_grpo_loss(CONFIG, g4rl.grpo_spec(), True)(model, **data, adv=batch["adv"])
    with pytest.raises(ValueError, match="tis_weight"):
        g4rl.make_grpo_loss(CONFIG, g4rl.grpo_spec(tis=True), True)(model, **{**with_passes(model, batch)})
    loss, aux = g4rl.make_grpo_loss(CONFIG, g4rl.grpo_spec(), False)(model, **data)
    lp = tj.full_logp(tj.Toy(nnx.Rngs(7)), data)
    assert set(aux) == {"xent_sum", "total_weights"}
    assert float(aux["xent_sum"]) == pytest.approx(-float(np.sum(lp)), rel=1e-5)


def test_grpo_terms_is_the_same_in_numpy_and_jax():
    rng, mask, logp = toy_logps(8)
    old = np.where(mask, logp + rng.normal(scale=0.5, size=mask.shape), 0.0).astype(np.float32)
    ref = np.where(mask, logp + rng.normal(scale=0.5, size=mask.shape), 0.0).astype(np.float32)
    adv = np.where(mask, rng.normal(size=(mask.shape[0], 1)), 0.0).astype(np.float32)
    tis = np.where(mask, rng.uniform(0.5, 2.0, size=mask.shape), 0.0).astype(np.float32)
    opts = g4rl.grpo_spec()
    a, sa = g4rl.grpo_terms(logp, old, ref, adv, mask, opts, tis=tis)
    b, sb = jax.jit(lambda *x: g4rl.grpo_terms(*x, opts, jnp, tis=jnp.asarray(tis)))(logp, old, ref, adv, mask)
    assert float(b) == pytest.approx(float(a), rel=1e-5)
    assert {k: float(v) for k, v in sb.items()} == pytest.approx({k: float(v) for k, v in sa.items()}, rel=1e-5)


def test_logp_passes_places_each_row_by_its_index(maxtext_stub, monkeypatch, tmp_path):
    rng = np.random.default_rng(9)
    units = []
    for k in range(30):
        completion = rng.integers(5, V, size=int(rng.integers(2, 6))).tolist()
        units.append({"uid": f"u{k}", "prompt": [2] + rng.integers(5, V, size=4).tolist(), "completion": completion,
                      "adv": float(rng.normal()), "sampler_logp": [-1.0] * len(completion)})
    g4rl.write_rows7(g4rl.pack_rows7(units, 32, 4), tmp_path / "train.parquet")
    arrays, batch = g4rl.read_packed7(tmp_path / "train.parquet")
    batches = g4dpo.batch_count(batch, 4)
    assert batches >= 2

    def fake_iterator(path, mesh, rows, n, padding=0, passes=None, tis=None, row_ids=False):
        for b in g4rl.local_batches7(arrays, 0, 1, rows, n, padding, passes, tis, row_ids):
            yield {k: jnp.asarray(v[::-1]) for k, v in b.items()}

    monkeypatch.setattr(g4rl, "global_iterator7", fake_iterator)
    model = tj.Toy(nnx.Rngs(9))
    model.decoder.out.kernel_lora_b[...] = jnp.full((tj.R, tj.D), 0.1)
    passes, seconds = g4rl.logp_passes(model, CONFIG, tmp_path / "train.parquet", None, 4, batches)
    assert passes["old_logp"].shape == (batches * 4, 32) and len(seconds) == batches
    for b in range(batches):
        data = {k: jnp.asarray(arrays[k][4 * b:4 * b + 4]) for k in g4dpo.KEYS}
        np.testing.assert_allclose(passes["old_logp"][4 * b:4 * b + 4], g4dpo.model_logp(model, CONFIG, data, False),
                                   rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(passes["ref_logp"][4 * b:4 * b + 4],
                                   g4dpo.model_logp(g4dpo.without_lora(model), CONFIG, data, False), rtol=1e-6,
                                   atol=1e-6)
