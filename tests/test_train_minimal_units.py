"""Optimizer and metric code of scripts/train_minimal.py, on the parameter tree of the real 22-layer model."""
import importlib.machinery
import sys
import types
from pathlib import Path
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import traverse_util

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from mmbjax.configuration_mmbbert import MMBertConfig  # noqa: E402
from mmbjax.heads.masked_lm import ModernBertForMaskedLM  # noqa: E402

LAYERS = 22


@pytest.fixture(scope="module")
def tm():
    # Other test modules already started JAX, after which its module-level initialize() can't run
    with mock.patch("jax.distributed.initialize", side_effect=ValueError):
        import train_minimal

    return train_minimal


@pytest.fixture(scope="module")
def model_and_params():
    # The architecture of the 230M runs; only the widths are small
    config = MMBertConfig(
        vocab_size=64, hidden_size=32, num_hidden_layers=LAYERS, num_attention_heads=2, intermediate_size=32,
        global_attn_every_n_layers=3, max_position_embeddings=64, pad_token_id=0, bos_token_id=1, eos_token_id=2,
        compute_dtype="float32", param_dtype="float32",
    )
    model = ModernBertForMaskedLM(config, dtype=jnp.float32, param_dtype=jnp.float32)
    ones = jnp.ones((1, 8), dtype=jnp.int32)
    params = model.init(jax.random.PRNGKey(0), input_ids=ones, attention_mask=ones, deterministic=True)["params"]
    return model, params


def _updates(tm, model, params, factor, weight_decay=0.0):
    optimizer, _ = tm.create_train_state(
        model, learning_rate=1e-3, num_train_steps=100, warmup_steps=1, weight_decay=weight_decay,
        lr_schedule_type="constant", layer_norm_lr_factor=factor,
    )
    opt_state = optimizer.init(params)
    grads = jax.tree_util.tree_map(jnp.ones_like, params)
    for _ in range(3):
        updates, opt_state = optimizer.update(grads, opt_state, params)
    return traverse_util.flatten_dict(updates)


def _is_norm(path):
    return any(part == "norm" or part.endswith("_norm") for part in path)


def test_layer_norm_lr_factor_applies_to_norms_only(tm, model_and_params):
    model, params = model_and_params
    plain = _updates(tm, model, params, factor=1.0)
    scaled = _updates(tm, model, params, factor=0.1)
    norms = [path for path in plain if _is_norm(path)]
    # 22 mlp_norm, 21 attention_norm (layer 0 has none), embeddings, final and prediction head
    assert len(norms) == 22 + 21 + 3
    for path in plain:
        ratio = float(jnp.mean(scaled[path] / plain[path]))
        assert ratio == pytest.approx(0.1 if _is_norm(path) else 1.0, rel=1e-3), path


def test_weight_decay_still_reaches_the_kernels_with_a_factor(tm, model_and_params):
    model, params = model_and_params
    without = _updates(tm, model, params, factor=0.1, weight_decay=0.0)
    with_decay = _updates(tm, model, params, factor=0.1, weight_decay=0.1)
    kernel = ("model", "encoder", "layers_3", "mlp", "wi", "kernel")
    norm = ("model", "encoder", "layers_3", "mlp_norm", "scale")
    assert not np.allclose(with_decay[kernel], without[kernel])
    assert np.allclose(with_decay[norm], without[norm])


def test_per_layer_gradient_norms_cover_all_22_layers_without_mixing(tm, model_and_params):
    _, params = model_and_params
    flat = traverse_util.flatten_dict(params)
    # A gradient of ones in layer 1 only: "layer_1" as a substring would also catch 10 to 19 and 21
    grads = {path: (jnp.ones_like(v) if "layers_1" in path else jnp.zeros_like(v)) for path, v in flat.items()}
    norms = tm.compute_per_layer_gradient_norms(traverse_util.unflatten_dict(grads))

    assert {f"encoder_{i}_attention" for i in range(LAYERS)} <= set(norms)
    assert {f"encoder_{i}_ffn" for i in range(LAYERS)} <= set(norms)
    assert {"embeddings", "lm_head"} <= set(norms)
    assert "other" not in norms
    for name, value in norms.items():
        if name.startswith("encoder_1_"):
            assert float(value) > 0, name
        else:
            assert float(value) == 0, name

    # Nothing counted twice, nothing left out
    ones = tm.compute_per_layer_gradient_norms(jax.tree_util.tree_map(jnp.ones_like, params))
    total = sum(float(v) ** 2 for v in ones.values())
    assert total == pytest.approx(sum(v.size for v in flat.values()), rel=1e-5)


def test_per_layer_norms_are_requested_on_a_step_that_logs(tm):
    every = 100
    for accumulation in (1, 3):
        requested = [
            optimizer_step
            for micro_step in range(1000 * accumulation)
            for optimizer_step in [micro_step // accumulation]
            if tm.wants_per_layer_norms(optimizer_step, micro_step, accumulation, every)
        ]
        assert requested == [99, 199, 299, 399, 499, 599, 699, 799, 899, 999]
        # The loop logs when optimizer_step + 1, after this micro-step, is a multiple of log_every
        assert all((step + 1) % 10 == 0 for step in requested)


def test_tracker_choice(tm):
    assert tm.load_tracker("none") is None
    fake = types.ModuleType("trackio")
    fake.__spec__ = importlib.machinery.ModuleSpec("trackio", None)
    sys.modules.setdefault("trackio", fake)
    assert tm.load_tracker("trackio") is sys.modules["trackio"]


def test_duty_cycle_is_read_from_the_table_of_tpu_info(tm):
    # Literal output on the second host of a v4-32, during a run: chips are numbered 4 to 7
    output = """
TPU Runtime Utilization

| Chip | HBM Usage (GiB)       | Duty cycle |
|------|-----------------------|------------|
| 4    | 18.10 GiB / 30.75 GiB | 95.72%     |
| 5    | 18.10 GiB / 30.75 GiB | 95.73%     |

TensorCore Utilization

| Core ID | TensorCore Utilization |
|---------|------------------------|
| 0       | N/A                    |
"""
    assert {int(chip): float(duty) for chip, duty in tm.DutyCycleSampler.ROW.findall(output)} == {4: 95.72, 5: 95.73}
