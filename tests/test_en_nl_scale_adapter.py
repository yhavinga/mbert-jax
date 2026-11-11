"""experiments/en-nl-translator/scale_adapter.py: an adapter whose update is s times its source's, through translate.py's
own lora_delta (numpy, no JAX)."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "en-nl-translator"))
sys.path.insert(0, str(ROOT / "experiments" / "flan-nl"))

import scale_adapter as sa  # noqa: E402
import translate  # noqa: E402

SHAPES = {"decoder/layers_0/self_attention/query": ((8, 2, 4), (8, 3), (3, 2, 4)),
          "decoder/layers_0/mlp/wo": ((12, 8), (12, 3), (3, 8))}


def adapter(seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for module, (_, a, b) in SHAPES.items():
        out[f"{module}/kernel_lora_a"] = rng.standard_normal(a).astype(np.float32)
        out[f"{module}/kernel_lora_b"] = rng.standard_normal(b).astype(np.float32)
    return out


@pytest.mark.parametrize("factor", [0.5, 0.25])
def test_update_is_factor_times_the_source(factor):
    src = adapter()
    scaled = sa.scale_adapter(src, factor)
    assert set(scaled) == set(src)
    for module, (shape, _, _) in SHAPES.items():
        a, b = src[f"{module}/kernel_lora_a"], src[f"{module}/kernel_lora_b"]
        a2, b2 = scaled[f"{module}/kernel_lora_a"], scaled[f"{module}/kernel_lora_b"]
        assert a2.dtype == b2.dtype == np.float32
        assert np.array_equal(a2, a)
        assert np.array_equal(b2, b * np.float32(factor))
        # a power of two: the merged update is exactly factor times the source's
        assert np.array_equal(translate.lora_delta(a2, b2, shape, 1.0), factor * translate.lora_delta(a, b, shape, 1.0))


def test_source_is_not_changed():
    src = adapter()
    before = {k: v.copy() for k, v in src.items()}
    sa.scale_adapter(src, 0.5)
    assert all(np.array_equal(src[k], before[k]) for k in src)


@pytest.mark.parametrize("factor", [0.0, -0.5, 1.5])
def test_factor_out_of_range(factor):
    with pytest.raises(ValueError, match="not in"):
        sa.scale_adapter(adapter(), factor)


def test_unexpected_key():
    bad = {**adapter(), "decoder/layers_0/mlp/wo/kernel": np.zeros((12, 8), np.float32)}
    with pytest.raises(ValueError, match="unexpected key"):
        sa.scale_adapter(bad, 0.5)


def write(directory, arrays, meta):
    directory.mkdir(parents=True, exist_ok=True)
    np.savez(directory / "adapter.npz", **arrays)
    (directory / "adapter.json").write_text(json.dumps(meta))


def test_main_writes_the_scaled_adapter(tmp_path, capsys):
    meta = {"rank": 3, "alpha": 3.0, "scale": 1.0, "steps": 7}
    write(tmp_path / "N", adapter(), meta)
    sa.main(["--adapter", str(tmp_path / "N"), "--factor", "0.25", "--source", "gs://b/arms/N", "--out",
             str(tmp_path / "N25")])
    out_meta = json.loads((tmp_path / "N25" / "adapter.json").read_text())
    assert out_meta == {**meta, "scaled_from": "gs://b/arms/N", "lora_b_factor": 0.25}
    with np.load(tmp_path / "N25" / "adapter.npz") as npz:
        got = {k: npz[k] for k in npz.files}
    want = sa.scale_adapter(adapter(), 0.25)
    assert set(got) == set(want) and all(np.array_equal(got[k], want[k]) for k in want)
    line = json.loads(capsys.readouterr().out.split("SCALED ", 1)[1])
    assert line["modules"] == 2 and line["lora_b_norm_scaled"] == pytest.approx(0.25 * line["lora_b_norm"])


def test_main_refuses_an_existing_output_and_a_scaled_source(tmp_path):
    write(tmp_path / "N", adapter(), {"rank": 3, "alpha": 3.0, "scale": 1.0})
    write(tmp_path / "N50", adapter(), {"rank": 3, "alpha": 3.0, "scale": 1.0, "lora_b_factor": 0.5})
    with pytest.raises(SystemExit, match="exists"):
        sa.main(["--adapter", str(tmp_path / "N"), "--factor", "0.5", "--source", "s", "--out", str(tmp_path / "N50")])
    with pytest.raises(SystemExit, match="already scaled"):
        sa.main(["--adapter", str(tmp_path / "N50"), "--factor", "0.5", "--source", "s", "--out", str(tmp_path / "x")])
