"""The final run's changes to scripts/train_minimal.py: the WSD schedule with cooldown branches, and
optax.apply_if_finite around the optimizer (--skip-nonfinite-updates)."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

PEAK, WARMUP, END = 4.8e-4, 1250, 1e-6
# The final run: tak 1 ends after one epoch, tak 2 after two, each with a cooldown of 20.4% of its steps
EPOCH = 653_000
TAK1 = dict(total_steps=EPOCH, decay_steps=133_000)
TAK2 = dict(total_steps=2 * EPOCH, decay_steps=266_000)


@pytest.fixture(scope="module")
def tm():
    # Other test modules may have started JAX already, after which its module-level initialize() can't run
    with mock.patch("jax.distributed.initialize", side_effect=ValueError):
        import train_minimal

    return train_minimal


def lr(schedule, step):
    return float(schedule(step))


def test_wsd_warms_up_as_cosine_holds_the_peak_and_cools_down_with_1_sqrt(tm):
    wsd = tm.wsd_schedule(PEAK, WARMUP, end_value=END, **TAK1)
    cosine = optax.warmup_cosine_decay_schedule(0.0, PEAK, WARMUP, EPOCH, END)
    # The warmup of S's cosine run, step for step
    for step in (0, 1, 2, 625, 1249, 1250):
        assert lr(wsd, step) == lr(cosine, step), step
    for step in (1251, 100_000, 519_999, 520_000):
        assert lr(wsd, step) == pytest.approx(PEAK, rel=1e-6), step
    start, length = EPOCH - 133_000, 133_000
    for done in (0.01, 0.25, 0.5, 0.81, 0.99):
        step = start + round(done * length)
        expected = END + (PEAK - END) * (1 - np.sqrt((step - start) / length))
        assert lr(wsd, step) == pytest.approx(expected, rel=1e-5), done
    assert lr(wsd, EPOCH) == pytest.approx(END, rel=1e-5)
    assert lr(wsd, EPOCH + 1000) == pytest.approx(END, rel=1e-5)
    # 1-sqrt falls fast at first: a quarter into the cooldown the rate is at half the peak
    assert lr(wsd, start + length // 4) == pytest.approx(END + (PEAK - END) / 2, rel=1e-4)


def test_wsd_linear_shape():
    import train_minimal as tm  # noqa: F811 (loaded by the fixture of another test)

    wsd = tm.wsd_schedule(1.0, 10, 110, 50, 0.0, shape="linear")
    assert [round(lr(wsd, s), 6) for s in (5, 10, 60, 85, 110, 200)] == [0.5, 1.0, 1.0, 0.5, 0.0, 0.0]


def test_branches_share_every_step_before_the_first_cooldown(tm):
    """Tak 1 starts from the main run's checkpoint at its cooldown start: up to there the two schedules agree, and
    the main run (stopped at tak 2's cooldown start) has tak 2's schedule."""
    tak1 = tm.wsd_schedule(PEAK, WARMUP, end_value=END, **TAK1)
    tak2 = tm.wsd_schedule(PEAK, WARMUP, end_value=END, **TAK2)
    steps = np.r_[np.arange(0, 3000), np.arange(3000, 520_001, 997), 520_000]
    assert all(lr(tak1, s) == lr(tak2, s) for s in steps)
    assert lr(tak1, 520_001) < lr(tak2, 520_001) == pytest.approx(PEAK, rel=1e-6)
    assert lr(tak2, 1_040_000) == pytest.approx(PEAK, rel=1e-6)
    assert lr(tak2, 2 * EPOCH) == pytest.approx(END, rel=1e-5)
    # Both cooldowns are the same fraction of their run
    assert TAK1["decay_steps"] / TAK1["total_steps"] == TAK2["decay_steps"] / TAK2["total_steps"]


def test_wsd_refuses_a_cooldown_that_does_not_fit(tm):
    for decay in (0, EPOCH - WARMUP + 1):
        with pytest.raises(ValueError):
            tm.wsd_schedule(PEAK, WARMUP, EPOCH, decay, END)
    with pytest.raises(ValueError):
        tm.wsd_schedule(PEAK, WARMUP, EPOCH, 1000, END, shape="cosine")


def _args(tm, *extra):
    argv = ["train_minimal.py", "--max-steps", "1000", "--warmup-steps", "10", *extra]
    with mock.patch.object(sys, "argv", argv):
        return tm.parse_args()


def test_arguments_of_the_final_run(tm, tmp_path, capsys):
    args = _args(tm, "--lr-schedule", "wsd", "--decay-steps", "200", "--skip-nonfinite-updates", "--use-packing",
                 "--data-files-list", str(tmp_path / "files.txt"), "--url-blocklist", "hebban.nl")
    assert (args.decay_steps, args.decay_shape, args.skip_nonfinite_updates, args.max_consecutive_nonfinite) == (200, "1-sqrt", True, 50)
    for bad in (["--lr-schedule", "wsd"], ["--lr-schedule", "wsd", "--decay-steps", "991"],
                ["--lr-schedule", "cosine", "--decay-steps", "100"], ["--url-blocklist", "hebban.nl"],
                ["--use-packing", "--data-files-list", "x", "--url-blocklist", "https://hebban.nl/"]):
        with pytest.raises(SystemExit):
            _args(tm, *bad)


def _tiny_params():
    rng = np.random.RandomState(0)
    return {
        "encoder": {"dense": {"kernel": jnp.asarray(rng.randn(8, 4), jnp.float32), "bias": jnp.zeros(4, jnp.float32)},
                    "mlp_norm": {"scale": jnp.ones(4, jnp.float32)}},
        "decoder": {"kernel": jnp.asarray(rng.randn(4, 3), jnp.float32)},
    }


def _optimizer(tm, skip, layer_norm_lr_factor=1.0):
    optimizer, _ = tm.create_train_state(
        None, learning_rate=PEAK, num_train_steps=100, warmup_steps=5, weight_decay=0.01, lr_schedule_type="wsd",
        layer_norm_lr_factor=layer_norm_lr_factor, min_lr=END, decay_steps=40, skip_nonfinite_updates=skip)
    return optimizer


def _gradients(step, params):
    rng = np.random.RandomState(100 + step)
    # Some steps far above the clipping norm of 1.0, so the clip is exercised too
    scale = 50.0 if step % 3 == 0 else 0.1
    return jax.tree_util.tree_map(lambda p: jnp.asarray(rng.randn(*p.shape) * scale, jnp.float32), params)


def _run_steps(optimizer, params, grads_of_step, steps):
    state = optimizer.init(params)

    @jax.jit
    def step(params, state, grads):
        updates, state = optimizer.update(grads, state, params)
        return optax.apply_updates(params, updates), state

    for i in range(steps):
        params, state = step(params, state, grads_of_step(i, params))
    return params, state


def _inner(state):
    """The inner chain's state of an apply_if_finite state."""
    assert isinstance(state, optax.ApplyIfFiniteState)
    return state.inner_state


def _same_bytes(a, b):
    leaves_a, leaves_b = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
    assert len(leaves_a) == len(leaves_b)
    return all(np.asarray(x).tobytes() == np.asarray(y).tobytes() for x, y in zip(leaves_a, leaves_b))


def _max_ulp(a, b):
    return max(int(np.max(np.abs(np.asarray(x).view(np.int32).astype(np.int64) - np.asarray(y).view(np.int32).astype(np.int64))))
               for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)))


@pytest.mark.parametrize("layer_norm_lr_factor", [1.0, 0.1])
def test_finite_steps_are_the_same_with_and_without_apply_if_finite(tm, layer_norm_lr_factor):
    """The same updates when every gradient is finite. One step is byte for byte that of the plain optimizer. Over 30
    steps (warmup, peak, cooldown, some clipped) at most 2 units in the last place differ: optax compiles the inner
    update inside a lax.cond, and XLA may round a fused branch in the last bit otherwise (measured on the CPU with jit: 1
    of the 52 numbers, by 1 ulp). Adam's step count, and so the learning rate, stays the same."""
    params = _tiny_params()
    one_plain, one_plain_state = _run_steps(_optimizer(tm, False, layer_norm_lr_factor), params, _gradients, 1)
    one_skip, one_skip_state = _run_steps(_optimizer(tm, True, layer_norm_lr_factor), params, _gradients, 1)
    assert _same_bytes(one_plain, one_skip) and _same_bytes(one_plain_state, _inner(one_skip_state))

    plain_params, plain_state = _run_steps(_optimizer(tm, False, layer_norm_lr_factor), params, _gradients, 30)
    skip_params, skip_state = _run_steps(_optimizer(tm, True, layer_norm_lr_factor), params, _gradients, 30)
    assert _max_ulp(plain_params, skip_params) <= 2
    assert _max_ulp(plain_state, _inner(skip_state)) <= 2
    counts = lambda state: sorted(int(leaf) for leaf in jax.tree_util.tree_leaves(state)  # noqa: E731
                                  if np.issubdtype(np.asarray(leaf).dtype, np.integer))
    assert counts(plain_state) == counts(_inner(skip_state)) and set(counts(plain_state)) == {30}
    assert tm.nonfinite_counts(skip_state) == (0, 0)
    assert tm.nonfinite_counts(plain_state) is None


def test_a_non_finite_step_changes_nothing_and_is_counted(tm):
    """Steps 4 (NaN) and 9-10 (inf, NaN) are skipped: parameters and Adam's state are byte for byte those of the
    same optimizer given only the 8 finite gradients, and the counts say 3 skipped, the last 2 in a row."""
    params = _tiny_params()
    bad = {4: np.nan, 9: np.inf, 10: np.nan}

    def with_bad_steps(i, p):
        grads = _gradients(i, p)
        if i in bad:
            grads["encoder"]["dense"]["kernel"] = grads["encoder"]["dense"]["kernel"].at[0, 0].set(bad[i])
        return grads

    finite_steps = [i for i in range(14) if i not in bad]
    skip = _optimizer(tm, True)
    skipped_params, skipped_state = _run_steps(skip, params, with_bad_steps, 11)
    assert tm.nonfinite_counts(skipped_state) == (2, 3)
    expected_params, expected_state = _run_steps(_optimizer(tm, True), params,
                                                 lambda i, p: _gradients(finite_steps[i], p), 8)
    assert _same_bytes(skipped_params, expected_params)
    assert _same_bytes(_inner(skipped_state), _inner(expected_state))
    assert all(np.isfinite(np.asarray(leaf)).all() for leaf in jax.tree_util.tree_leaves(skipped_params))
    # A finite step after them resets the count in a row, not the total
    _, state = jax.jit(skip.update)(_gradients(11, skipped_params), skipped_state, skipped_params)
    assert tm.nonfinite_counts(state) == (0, 3)

    # Without the wrap one NaN step poisons every parameter: the global norm of the clip is NaN
    plain_params, _ = _run_steps(_optimizer(tm, False), params, with_bad_steps, 5)
    assert all(np.isnan(np.asarray(leaf)).all() for leaf in jax.tree_util.tree_leaves(plain_params))
