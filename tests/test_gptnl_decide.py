"""The decision rule of experiments/gptnl/protocol.md (decide_g.py), on made-up counts around X's real ones."""
import sys
from fractions import Fraction
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "gptnl"))

import decide_g  # noqa: E402

# X's k of the protocol's table: seed ranges 15 of 2,112 (DBRD) and 74 of 4,864 (SICK-NL)
X = decide_g.PROTOCOL_K["X"]


def shifted(dbrd=0, sicknl=0):
    """X with every seed moved by the given number of examples: the same ranges, the mean moved by exactly that."""
    return {"dbrd": [k + dbrd for k in X["dbrd"]], "sicknl": [k + sicknl for k in X["sicknl"]]}


def test_g_equal_to_x_is_no_difference():
    tasks, verdict = decide_g.rule(shifted(), X)
    assert verdict == "geen verschil"
    assert tasks["sicknl"]["threshold"] == Fraction(74, 4864)


def test_a_sicknl_gain_beyond_the_threshold_helps():
    tasks, verdict = decide_g.rule(shifted(sicknl=75), X)
    assert tasks["sicknl"] == {"difference": Fraction(75, 4864), "threshold": Fraction(74, 4864), "side": "above"}
    assert verdict == "GPT-NL helpt"


def test_more_than_is_strict():
    assert decide_g.rule(shifted(sicknl=74), X)[1] == "geen verschil"
    assert decide_g.rule(shifted(dbrd=-15), X)[1] == "geen verschil"
    assert decide_g.rule(shifted(dbrd=-16), X)[1] == "GPT-NL schaadt"


def test_harm_is_checked_first():
    # Below on DBRD, far above on SICK-NL
    assert decide_g.rule(shifted(dbrd=-16, sicknl=200), X)[1] == "GPT-NL schaadt"
    assert decide_g.rule(shifted(sicknl=-75), X)[1] == "GPT-NL schaadt"


def test_a_dbrd_gain_alone_is_no_difference():
    tasks, verdict = decide_g.rule(shifted(dbrd=100), X)
    assert tasks["dbrd"]["side"] == "above"
    assert verdict == "geen verschil"


def test_the_threshold_is_the_larger_seed_range():
    # G's mean 93 above X's on SICK-NL, more than X's range of 74, but G's own seeds spread 100
    g = {"dbrd": list(X["dbrd"]), "sicknl": [3781, 3881, 3831]}
    tasks, verdict = decide_g.rule(g, X)
    assert tasks["sicknl"]["difference"] == Fraction(93, 4864)
    assert tasks["sicknl"]["threshold"] == Fraction(100, 4864)
    assert verdict == "geen verschil"


def test_counts_are_exact():
    assert decide_g.count(1967 / 2112, 2112, "x") == 1967
    assert decide_g.count(0.931344696969697, 2112, "x") == 1967
    with pytest.raises(ValueError, match="is not k / 2112"):
        decide_g.count(0.9313, 2112, "x")


def test_load_counts_reads_the_three_seeds_in_order():
    read = {f"d/G--sicknl--seed{s}.json": {"test_accuracy": k / 4864} for s, k in zip(decide_g.SEEDS, (3701, 3738, 3775))}
    assert decide_g.load_counts("d", "G", "sicknl", 4864, read=read.__getitem__) == [3701, 3738, 3775]
