"""scripts/convert_gcs_checkpoint.py --params-only: the params of a training checkpoint without its optimizer state."""
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import convert_gcs_checkpoint as convert  # noqa: E402


def _save_training_checkpoint(directory):
    """A step directory as train_minimal.py saves one: params, an AdamW state of twice their size, the step."""
    rng = np.random.RandomState(0)
    params = {"model": {"embeddings": {"embedding": jnp.asarray(rng.randn(32, 8), dtype=jnp.float32)},
                        "encoder": {"layers_0": {"kernel": jnp.asarray(rng.randn(8, 8), dtype=jnp.float32)}}}}
    state = {"params": params, "opt_state": optax.adamw(1e-3).init(params), "step": jnp.asarray(7)}
    manager = ocp.CheckpointManager(str(directory))
    manager.save(7, args=ocp.args.StandardSave(state))
    manager.wait_until_finished()
    return params


def test_params_only_restores_the_same_params_as_the_whole_checkpoint(tmp_path):
    params = _save_training_checkpoint(tmp_path / "ckpt")
    step_dir = tmp_path / "ckpt" / "7"
    whole = convert.load_orbax_checkpoint(step_dir)
    only = convert.load_orbax_checkpoint(step_dir, params_only=True)
    expected = {"model": {"embeddings": {"embedding": np.asarray(params["model"]["embeddings"]["embedding"])},
                          "encoder": {"layers_0": {"kernel": np.asarray(params["model"]["encoder"]["layers_0"]["kernel"])}}}}
    for restored in (whole, only):
        assert isinstance(restored["model"]["embeddings"]["embedding"], np.ndarray)
        np.testing.assert_array_equal(restored["model"]["embeddings"]["embedding"], expected["model"]["embeddings"]["embedding"])
        np.testing.assert_array_equal(restored["model"]["encoder"]["layers_0"]["kernel"], expected["model"]["encoder"]["layers_0"]["kernel"])


def test_a_gcs_checkpoint_is_read_in_place_only_with_params_only():
    with pytest.raises(ValueError, match="with --params-only only"):
        convert.load_orbax_checkpoint("gs://bucket/run/ckpt/300000")
