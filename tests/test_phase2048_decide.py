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


def windows(L_long2048=0.85, L_high=0.90, C_high=0.90):
    """Every model as good at window 2048 as at 1024 on the long documents, and equal on positions 1024-2047."""
    def model(long2048, high):
        return {
            "1024": {"complete": True, "groups": {"all": {"loss": 0.80, "masked": 4000}, "long_documents": {"loss": 0.85, "masked": 1000}}},
            "2048": {"complete": True, "groups": {
                "all": {"loss": 0.80, "masked": 4000}, "long_documents": {"loss": long2048, "masked": 1000},
                "positions_low": {"loss": 0.78, "masked": 3700}, "positions_high": {"loss": high, "masked": 300},
                "long_documents_positions_low": {"loss": 0.83, "masked": 700}}},
        }
    return {"S": model(0.86, 0.90), "C": model(0.86, C_high), "L": model(L_long2048, L_high)}


def test_equal_models_make_no_measurable_difference():
    assert decide.decide(cells(), windows())["decision"] == "no measurable difference"


def test_a_gain_beyond_the_larger_seed_range_helps():
    result = decide.decide(cells(L_dbrd2048=[0.955, 0.956, 0.957]), windows())
    assert result["help"]["L_dbrd2048_minus_best_C"] == pytest.approx(0.005)
    assert result["help"]["margin"] == pytest.approx(0.002)
    assert result["decision"] == "2048 helps"


def test_the_gain_is_measured_against_the_best_length_of_C():
    # L@2048 beats C@2048 but not C@1024 by more than the range
    result = decide.decide(cells(L_dbrd2048=[0.955, 0.956, 0.957], C_dbrd1024=[0.954, 0.955, 0.956]), windows())
    assert result["help"]["best_C_dbrd_cell"] == "dbrd1024"
    assert result["decision"] == "no measurable difference"


def test_the_larger_range_of_the_two_cells_is_the_margin():
    # A gain of 0.005, but L's own seeds spread 0.006
    result = decide.decide(cells(L_dbrd2048=[0.953, 0.956, 0.959]), windows())
    assert result["help"]["margin"] == pytest.approx(0.006)
    assert result["decision"] == "no measurable difference"


def test_a_gain_with_a_loss_on_sicknl_harms():
    result = decide.decide(cells(L_dbrd2048=[0.955, 0.956, 0.957], L_sicknl512=[0.840, 0.841, 0.842]), windows())
    assert result["harm"]["sicknl512"]["harmed"]
    assert result["decision"] == "2048 harms"


def test_a_loss_on_dbrd512_within_c_range_is_no_harm():
    result = decide.decide(cells(L_dbrd512=[0.9495, 0.9505, 0.9515]), windows())
    assert not result["harm"]["dbrd512"]["harmed"]
    assert decide.decide(cells(L_dbrd512=[0.945, 0.946, 0.947]), windows())["decision"] == "2048 harms"


def test_using_the_longer_context_without_harm_helps():
    result = decide.decide(cells(), windows(L_long2048=0.84, L_high=0.86))
    assert result["context"]["L_long_documents_1024_minus_2048"] == pytest.approx(0.01)
    assert result["context"]["uses_context"]
    assert not result["help"]["exceeds"]
    assert result["decision"] == "2048 helps"


def test_using_the_longer_context_with_a_loss_on_sicknl_harms():
    result = decide.decide(cells(L_sicknl512=[0.840, 0.841, 0.842]), windows(L_long2048=0.84, L_high=0.86))
    assert result["decision"] == "2048 harms"


def test_a_window_gain_without_beating_c_on_positions_1024_2047_is_no_difference():
    result = decide.decide(cells(), windows(L_long2048=0.84, L_high=0.86, C_high=0.86))
    assert not result["context"]["uses_context"]
    assert result["decision"] == "no measurable difference"


def test_beating_c_on_positions_1024_2047_without_a_window_gain_is_no_difference():
    result = decide.decide(cells(), windows(L_long2048=0.85, L_high=0.86))
    assert not result["context"]["uses_context"]
    assert result["decision"] == "no measurable difference"


def test_the_window_test_needs_the_same_masked_tokens():
    unequal_models = windows()
    unequal_models["C"]["2048"]["groups"]["positions_high"]["masked"] = 299
    with pytest.raises(ValueError, match="differ between models"):
        decide.decide(cells(), unequal_models)
    unequal_windows = windows()
    for model in decide.MODELS:
        unequal_windows[model]["2048"]["groups"]["long_documents"]["masked"] = 999
    with pytest.raises(ValueError, match="not the same tokens"):
        decide.decide(cells(), unequal_windows)


def test_window_files_must_exist_and_be_complete(tmp_path):
    for model, result in windows().items():
        (tmp_path / f"u15-{model}.json").write_text(json.dumps({"windows": result}))
    assert decide.load_windows(tmp_path, "u15")["L"]["2048"]["groups"]["positions_high"]["loss"] == 0.90
    cut_short = windows()["L"]
    cut_short["2048"]["complete"] = False
    (tmp_path / "u15-L.json").write_text(json.dumps({"windows": cut_short}))
    with pytest.raises(ValueError, match="did not count all documents"):
        decide.load_windows(tmp_path, "u15")
    (tmp_path / "u15-C.json").unlink()
    with pytest.raises(FileNotFoundError, match="u15-C.json"):
        decide.load_windows(tmp_path, "u15")


def test_files_are_read_per_seed_and_a_missing_seed_fails(tmp_path):
    for model, by_cell in cells().items():
        for cell, values in by_cell.items():
            for seed, value in zip(decide.SEEDS, values):
                (tmp_path / f"u15-{model}--{cell}--seed{seed}.json").write_text(json.dumps({"test_accuracy": value, "best_val_accuracy": 0.0}))
    assert decide.load_cells(tmp_path, "u15")["L"]["dbrd2048"] == [0.950, 0.951, 0.952]
    (tmp_path / "u15-L--dbrd2048--seed43.json").unlink()
    with pytest.raises(FileNotFoundError, match="seed43"):
        decide.load_cells(tmp_path, "u15")
