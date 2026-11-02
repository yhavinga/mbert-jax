"""The decision rule of experiments/phase2048/protocol.md, on made-up results."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "phase2048"))

import decide  # noqa: E402


def cells(**overrides):
    """Every model at 0.95 / 0.85 with seed spread 0.002, unless overridden as model_cell=[3 values]."""
    base = {"dbrd512": [0.950, 0.951, 0.952], "dbrd1024": [0.950, 0.951, 0.952], "dbrd2048": [0.950, 0.951, 0.952],
            "sicknl512": [0.850, 0.851, 0.852]}
    result = {model: dict(base) for model in decide.MODELS}
    for key, values in overrides.items():
        model, cell = key.split("_")
        result[model][cell] = values
    return result


def test_equal_models_make_no_measurable_difference():
    assert decide.decide(cells())["decision"] == "no measurable difference"


def test_a_gain_beyond_the_larger_seed_range_helps():
    result = decide.decide(cells(L_dbrd2048=[0.955, 0.956, 0.957]))
    assert result["help"]["L_dbrd2048_minus_best_C"] == pytest.approx(0.005)
    assert result["help"]["margin"] == pytest.approx(0.002)
    assert result["decision"] == "2048 helps"


def test_the_gain_is_measured_against_the_best_length_of_C():
    # L@2048 beats C@2048 but not C@1024 by more than the range
    result = decide.decide(cells(L_dbrd2048=[0.955, 0.956, 0.957], C_dbrd1024=[0.954, 0.955, 0.956]))
    assert result["help"]["best_C_dbrd_cell"] == "dbrd1024"
    assert result["decision"] == "no measurable difference"


def test_the_larger_range_of_the_two_cells_is_the_margin():
    # A gain of 0.005, but L's own seeds spread 0.006
    result = decide.decide(cells(L_dbrd2048=[0.953, 0.956, 0.959]))
    assert result["help"]["margin"] == pytest.approx(0.006)
    assert result["decision"] == "no measurable difference"


def test_a_gain_with_a_loss_on_sicknl_harms():
    result = decide.decide(cells(L_dbrd2048=[0.955, 0.956, 0.957], L_sicknl512=[0.840, 0.841, 0.842]))
    assert result["harm"]["sicknl512"]["harmed"]
    assert result["decision"] == "2048 harms"


def test_a_loss_on_dbrd512_within_c_range_is_no_harm():
    result = decide.decide(cells(L_dbrd512=[0.9495, 0.9505, 0.9515]))
    assert not result["harm"]["dbrd512"]["harmed"]
    assert decide.decide(cells(L_dbrd512=[0.945, 0.946, 0.947]))["decision"] == "2048 harms"


def test_files_are_read_per_seed_and_a_missing_seed_fails(tmp_path):
    for model, by_cell in cells().items():
        for cell, values in by_cell.items():
            for seed, value in zip(decide.SEEDS, values):
                (tmp_path / f"u15-{model}--{cell}--seed{seed}.json").write_text(json.dumps({"test_accuracy": value, "best_val_accuracy": 0.0}))
    assert decide.load_cells(tmp_path, "u15")["L"]["dbrd2048"] == [0.950, 0.951, 0.952]
    (tmp_path / "u15-L--dbrd2048--seed43.json").unlink()
    with pytest.raises(FileNotFoundError, match="seed43"):
        decide.load_cells(tmp_path, "u15")
