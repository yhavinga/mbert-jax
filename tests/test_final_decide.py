"""The decision rule of experiments/final/protocol.md (decide.py): on the real results of G, X, F and R at 300k
(tests/fixtures/final_decide/), against the outcomes decide_g.py, scripts/decide_r.py and LEARNINGS recorded, and on
made-up edge cases."""
import json
import math
import re
import sys
from fractions import Fraction
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "experiments" / "final"))
sys.path.insert(0, str(HERE.parent / "experiments" / "gptnl"))

import decide  # noqa: E402
import decide_g  # noqa: E402

FIXTURES = str(HERE / "fixtures" / "final_decide")
G, X, F, R = "cxgptnl-300000", "300000", "fw2-300000", "fw2rh3-300000"
M = "gs://t5-dutch-english-usc2/mbert"
CHECKPOINTS = {G: f"{M}/culturax-gptnl-nl-230m-chunk1024/ckpt_keep/300000", X: f"{M}/culturax-nl-230m-chunk1024/ckpt_keep/300000"}

# The primary tasks as the 300k runs had them: LR 2e-5; DBRD at 512 (masking/dumb) or 1024 (dbrd1024/dumb). X's
# SICK-NL files predate the fields max_length and pad_to_multiple_of; X's DBRD@1024 files say ckptu15-300000.
P512 = {"dbrd": {"n": 2112, "update": 192, "checkpoint_name": True, "settings": {"task": "dbrd", "learning_rate": 2e-5, "epochs": 3}},
        "sicknl": {"n": 4864, "update": 256, "checkpoint_name": True, "settings": {"task": "sicknl", "learning_rate": 2e-5, "epochs": 3}}}
P1024 = {"dbrd1024": {"n": 2112, "update": 192, "checkpoint_name": False,
                      "settings": {"task": "dbrd", "learning_rate": 2e-5, "epochs": 3, "max_length": 1024}},
         "sicknl": P512["sicknl"]}


# ---------------------------------------------------------------------------------------------- made-up results

def primary_result(name, task, k, **override):
    spec = decide.PRIMARY[task]
    result = {"task": spec["settings"]["task"], "checkpoint": f"/dev/shm/final_eval/ckpt{name}",
              "test_accuracy": k / spec["n"], "epochs": 3, "learning_rate": 5e-05,
              "batch_size": 12 if task == "dbrd1024" else 64, "gradient_accumulation_steps": 4 if task == "dbrd1024" else 1,
              "num_devices": 4, "max_length": spec["settings"]["max_length"], "pad_to_multiple_of": 128 if task == "dbrd1024" else 1}
    result.update(override)
    return result


def extra_result(name, task, seed, value, words=None, **override):
    spec = decide.EXTRA[task]["settings"]
    test = {"metric": value, spec["metric"]: value}
    if words is not None:
        test["words"] = words
    result = {**spec, "checkpoint": f"/dev/shm/final_eval/ckpt{name}", "test_accuracy": value if words else 0.5,
              "test_metric": value, "test": test, "seed": seed,
              "data": {"max_train_examples": None, "max_eval_examples": None}}
    result.update(override)
    return result


def blimp_result(k, checkpoint, **override):
    result = {**decide.BLIMP_SETTINGS, "checkpoint": checkpoint, "accuracy": k / decide.BLIMP_PAIRS}
    result.update(override)
    return result


POS_WORDS = {"pos-lassysmall": 28995, "pos-alpino": 11046}


def arm_files(name, checkpoint, dbrd=(2026, 2030, 2022), sicknl=(4200, 4210, 4190), squadnl=(0.83, 0.832, 0.828),
              conll2002=(0.91, 0.912, 0.908), lassysmall=(28400, 28410, 28390), alpino=(10790, 10800, 10780), blimp=7950):
    """{relative path: result} of one arm with the given numbers (k for counts, floats for F1)."""
    files = {}
    for seed, k_d, k_s in zip(decide.SEEDS, dbrd, sicknl):
        files[f"dumb/{name}--dbrd1024--seed{seed}.json"] = primary_result(name, "dbrd1024", k_d)
        files[f"dumb/{name}--sicknl--seed{seed}.json"] = primary_result(name, "sicknl", k_s)
    for task, values in (("squadnl", squadnl), ("conll2002", conll2002)):
        for seed, v in zip(decide.SEEDS, values):
            files[f"dumb-extra/{task}/{name}--{task}--seed{seed}.json"] = extra_result(name, task, seed, v)
    for task, ks in (("pos-lassysmall", lassysmall), ("pos-alpino", alpino)):
        for seed, k in zip(decide.SEEDS, ks):
            n = POS_WORDS[task]
            files[f"dumb-extra/{task}/{name}--{task}--seed{seed}.json"] = extra_result(name, task, seed, k / n, words=n)
    files[f"blimp-nl/{name}.json"] = blimp_result(blimp, checkpoint)
    return files


def reader(files):
    def read(root, relative):
        if relative not in files:
            raise decide.Fout(f"{root}/{relative}: missing")
        return json.loads(json.dumps(files[relative]))
    return read


def load(files, arm):
    return decide.load_arm(reader(files), "mem", decide.ARMS[arm]["name"], decide.ARMS[arm]["checkpoint"])


def arm(key, **numbers):
    return arm_files(decide.ARMS[key]["name"], decide.ARMS[key]["checkpoint"], **numbers)


# ---------------------------------------------------------------------------------------------- real results

def test_the_counts_of_x_and_f_are_those_of_the_gptnl_protocol():
    for arm_name, name in (("X", X), ("F", F)):
        counts = {task: decide.load_primary(decide.read_json, FIXTURES, name, task, spec) for task, spec in P512.items()}
        assert counts == decide_g.PROTOCOL_K[arm_name]


def comparisons(a, b, spec):
    return {task: decide.compare(decide.stat_counts(decide.load_primary(decide.read_json, FIXTURES, a, task, s), s["n"]),
                                 decide.stat_counts(decide.load_primary(decide.read_json, FIXTURES, b, task, s), s["n"]))
            for task, s in spec.items()}


def test_g_against_x_is_no_difference_as_decide_g_found():
    # decide_g.py, 10 Nov (LEARNINGS): DBRD -0.06 within 1.23, SICK-NL -2.63 within 3.48, "geen verschil"
    c = comparisons(G, X, P512)
    assert c["dbrd"] == {"difference": Fraction(5894 - 5898, 3 * 2112), "threshold": Fraction(26, 2112), "side": "within"}
    assert c["sicknl"] == {"difference": Fraction(10830 - 11214, 3 * 4864), "threshold": Fraction(169, 4864), "side": "within"}
    assert decide.primary_rule(c) == "geen verschil"


def test_r_against_f_and_against_x_harms_as_decide_r_found():
    # scripts/decide_r.py, 9 Nov: R vs F DBRD +1.52 above 0.80, SICK-NL -3.39 below 1.25; R vs X SICK-NL -5.195 below 1.521
    c = comparisons(R, F, P512)
    assert (c["dbrd"]["side"], c["sicknl"]["side"]) == ("above", "below")
    assert c["sicknl"]["difference"] == Fraction(10456 - 10951, 3 * 4864) and c["sicknl"]["threshold"] == Fraction(61, 4864)
    assert decide.primary_rule(c) == "schaadt"
    c = comparisons(R, X, P512)
    assert c["dbrd"] == {"difference": Fraction(17, 3 * 2112), "threshold": Fraction(15, 2112), "side": "within"}
    assert c["sicknl"]["difference"] == Fraction(10456 - 11214, 3 * 4864) and c["sicknl"]["side"] == "below"
    assert decide.primary_rule(c) == "schaadt"


def gx_reader():
    """The fixture files of G and X, with the same made-up CoNLL-2002 results for both (not published, see README)."""
    made_up = {}
    for name in (G, X):
        for seed, v in zip(decide.SEEDS, (0.85, 0.86, 0.855)):
            made_up[f"dumb-extra/conll2002/{name}--conll2002--seed{seed}.json"] = extra_result(name, "conll2002", seed, v)

    def read(root, relative):
        return made_up[relative] if relative in made_up else decide.read_json(root, relative)
    return read


def test_g_against_x_as_question_2_gives_the_reference_as_the_gptnl_decision_did():
    # LEARNINGS 10 Nov: DUMB "geen verschil", SQuAD-NL beter (+0.72 against 0.41), BLiMP-NL slechter (92.10 against
    # 92.86), POS gelijk; rule (3) gave "zonder GPT-NL", the reference. DBRD@1024 (92.66 against 92.99) within.
    read = gx_reader()
    g = decide.load_arm(read, FIXTURES, G, CHECKPOINTS[G], primary=P1024)
    x = decide.load_arm(read, FIXTURES, X, CHECKPOINTS[X], primary=P1024)
    assert g["primary"]["dbrd1024"] == [1947, 1980, 1944] and x["primary"]["dbrd1024"] == [1965, 1979, 1948]
    assert (g["blimp"], x["blimp"]) == (7736, 7800)
    assert g["extra"]["pos-alpino"][2] == 11046 and g["extra"]["pos-lassysmall"][2] == 28995
    q = decide.question(g, x, primary=P1024)
    assert q["primary"]["dbrd1024"]["side"] == "within" and q["primary"]["dbrd1024"]["threshold"] == Fraction(36, 2112)
    assert q["verdict"] == "geen verschil"
    assert {task: c["outcome"] for task, c in q["extra"].items()} == {
        "squadnl": "beter", "conll2002": "gelijk", "pos-lassysmall": "gelijk", "pos-alpino": "gelijk", "blimp-nl": "slechter"}
    assert q["extra"]["blimp-nl"]["pairs"] == -64
    assert round(float(100 * q["extra"]["squadnl"]["difference"]), 2) == 0.72
    assert round(float(100 * q["extra"]["squadnl"]["threshold"]), 2) == 0.41
    assert decide.final_model(q["verdict"], q["extra"])[0] == "T1"


def test_k_2026_gives_back_the_protocols_0_959280():
    assert repr(2026 / 2112).startswith("0.959280")
    assert decide.exact_count(2026 / 2112, 2112, "x") == 2026


# ---------------------------------------------------------------------------------------------- exactness and settings

def test_a_number_that_is_not_k_over_n_is_an_error():
    with pytest.raises(decide.Fout, match="is not k / 2112"):
        decide.exact_count(0.95, 2112, "x")
    with pytest.raises(decide.Fout, match="is not k / 2112"):
        decide.exact_count(math.nextafter(2026 / 2112, 1), 2112, "x")
    with pytest.raises(decide.Fout, match="is not a number"):
        decide.exact_count(None, 2112, "x")


def test_a_primary_result_not_k_over_n_stops_the_decision():
    files = arm("S")
    files["dumb/s-653000--sicknl--seed43.json"]["test_accuracy"] = 0.8635
    with pytest.raises(decide.Fout, match=r"s-653000--sicknl--seed43.json test_accuracy: 0.8635 is not k / 4864"):
        load(files, "S")


@pytest.mark.parametrize("path, change, message", [
    ("dumb/epoch1-653000--dbrd1024--seed42.json", {"learning_rate": 2e-5}, "learning_rate is 2e-05, the protocol's is 5e-05"),
    ("dumb/epoch1-653000--dbrd1024--seed43.json", {"max_length": 512}, "max_length is 512, the protocol's is 1024"),
    ("dumb/epoch1-653000--dbrd1024--seed44.json", {"gradient_accumulation_steps": 2}, "an update of 96"),
    ("dumb/epoch1-653000--sicknl--seed42.json", {"epochs": 2}, "epochs is 2"),
    ("dumb/epoch1-653000--sicknl--seed42.json", {"checkpoint": "/dev/shm/final_eval/ckpts-653000"}, "is not .../ckptepoch1-653000"),
    ("dumb-extra/squadnl/epoch1-653000--squadnl--seed43.json", {"seed": 42}, "seed is 42"),
    ("dumb-extra/squadnl/epoch1-653000--squadnl--seed43.json", {"doc_stride": 64}, "doc_stride is 64"),
    ("dumb-extra/conll2002/epoch1-653000--conll2002--seed44.json", {"lr_schedule": "constant"}, "lr_schedule is 'constant'"),
    ("dumb-extra/pos-alpino/epoch1-653000--pos-alpino--seed42.json",
     {"data": {"max_train_examples": 256, "max_eval_examples": 64}}, "a smoke run"),
    ("blimp-nl/epoch1-653000.json", {"max_pairs": 200}, "max_pairs is 200"),
    ("blimp-nl/epoch1-653000.json", {"checkpoint": decide.ARMS["S"]["checkpoint"]}, "the arm's is"),
])
def test_a_result_at_other_settings_is_an_error(path, change, message):
    files = arm("T1")
    files[path].update(change)
    with pytest.raises(decide.Fout, match=re.escape(message)):
        load(files, "T1")


def test_a_missing_seed_is_an_error():
    files = arm("T2")
    del files["dumb-extra/pos-lassysmall/epoch2-1306000--pos-lassysmall--seed44.json"]
    with pytest.raises(decide.Fout, match="epoch2-1306000--pos-lassysmall--seed44.json: missing"):
        load(files, "T2")


def test_pos_is_counted_on_the_test_words_alike():
    files = arm("S")
    files["dumb-extra/pos-alpino/s-653000--pos-alpino--seed43.json"]["test"]["words"] = 11045
    files["dumb-extra/pos-alpino/s-653000--pos-alpino--seed43.json"]["test_metric"] = 10790 / 11045
    files["dumb-extra/pos-alpino/s-653000--pos-alpino--seed43.json"]["test"]["metric"] = 10790 / 11045
    files["dumb-extra/pos-alpino/s-653000--pos-alpino--seed43.json"]["test"]["accuracy"] = 10790 / 11045
    with pytest.raises(decide.Fout, match=r"different test words \[11045, 11046\]"):
        load(files, "S")
    a = ("words", [1, 2, 3], 11046)
    with pytest.raises(decide.Fout, match="not counted alike"):
        decide.compare_extra("pos-alpino", a, ("words", [1, 2, 3], 11045))


def test_the_stored_f1_counts_exactly_where_floats_would_decide_otherwise():
    # A's mean is above B's by exactly A's seed range: "gelijk" ("more than" is strict). Added left to right in
    # floats (as sum() did before Python 3.12) the difference comes out larger than the range: "beter".
    a_values, y = (0.8052, 0.79, 0.7989), 0.7828333333333334
    assert sum(map(Fraction, a_values)) / 3 - Fraction(y) == Fraction(0.8052) - Fraction(0.79)
    assert (a_values[0] + a_values[1] + a_values[2]) / 3 - y > max(a_values) - min(a_values)
    a = ("f1", [Fraction(v) for v in a_values], None)
    b = ("f1", [Fraction(y)] * 3, None)
    c = decide.compare_extra("squadnl", a, b)
    assert c["difference"] == c["threshold"] and c["outcome"] == "gelijk"


# ---------------------------------------------------------------------------------------------- the rule

S_KS = {"dbrd1024": [2026, 2030, 2022], "sicknl": [4200, 4210, 4190]}  # ranges 8 of 2,112 and 20 of 4,864


def rule_for(dbrd_shift=0, sicknl_shift=0, dbrd=None, sicknl=None):
    a = {"dbrd1024": dbrd or [k + dbrd_shift for k in S_KS["dbrd1024"]],
         "sicknl": sicknl or [k + sicknl_shift for k in S_KS["sicknl"]]}
    c = {task: decide.compare(decide.stat_counts(a[task], decide.PRIMARY[task]["n"]),
                              decide.stat_counts(S_KS[task], decide.PRIMARY[task]["n"])) for task in S_KS}
    return c, decide.primary_rule(c)


def test_more_than_is_strict():
    assert rule_for(sicknl_shift=20)[1] == "geen verschil"
    assert rule_for(sicknl_shift=21)[1] == "helpt"
    assert rule_for(dbrd_shift=-8)[1] == "geen verschil"
    assert rule_for(dbrd_shift=-9)[1] == "schaadt"
    assert rule_for(sicknl_shift=-21)[1] == "schaadt"


def test_harm_is_checked_first_and_a_dbrd_gain_alone_is_no_difference():
    assert rule_for(dbrd_shift=-9, sicknl_shift=300)[1] == "schaadt"
    c, verdict = rule_for(dbrd_shift=50)
    assert c["dbrd1024"]["side"] == "above" and verdict == "geen verschil"


def test_the_threshold_is_the_larger_seed_range():
    # A 25 above on SICK-NL, more than the reference's range of 20 but not than A's own of 40
    c, verdict = rule_for(sicknl=[4205, 4245, 4225])
    assert c["sicknl"] == {"difference": Fraction(75, 3 * 4864), "threshold": Fraction(40, 4864), "side": "within"}
    assert verdict == "geen verschil"


@pytest.mark.parametrize("pairs, outcome", [(-42, "gelijk"), (-43, "slechter"), (42, "gelijk"), (43, "beter"), (0, "gelijk")])
def test_blimp_nl_counts_more_than_half_a_point(pairs, outcome):
    assert decide.compare_blimp(7950 + pairs, 7950)["outcome"] == outcome


def extras(**outcomes):
    tasks = ["squadnl", "conll2002", "pos-lassysmall", "pos-alpino", "blimp-nl"]
    return {task: {"outcome": outcomes.get(task.replace("-", "_"), "gelijk")} for task in tasks}


@pytest.mark.parametrize("verdict, outcomes, model", [
    ("schaadt", {"squadnl": "beter", "blimp_nl": "beter"}, "T1"),
    ("helpt", {}, "T2"),
    ("helpt", {"squadnl": "beter"}, "T2"),
    ("helpt", {"blimp_nl": "slechter"}, "T1"),
    ("helpt", {"squadnl": "beter", "pos_alpino": "slechter"}, "T1"),
    ("geen verschil", {}, "T1"),
    ("geen verschil", {"conll2002": "beter"}, "T2"),
    ("geen verschil", {"conll2002": "beter", "pos_lassysmall": "beter"}, "T2"),
    ("geen verschil", {"squadnl": "beter", "blimp_nl": "slechter"}, "T1"),
    ("geen verschil", {"pos_alpino": "slechter"}, "T1"),
])
def test_the_final_model(verdict, outcomes, model):
    assert decide.final_model(verdict, extras(**outcomes))[0] == model


# ---------------------------------------------------------------------------------------------- the program

def write_root(tmp_path, *files):
    for group in files:
        for relative, result in group.items():
            path = tmp_path / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(result))
    return str(tmp_path)


def test_the_program_on_three_arms(tmp_path, capsys):
    # T1 equal to S; T2 above T1 on SICK-NL by more than the ranges, but 50 pairs worse on BLiMP-NL: T1
    root = write_root(tmp_path, arm("S"), arm("T1"), arm("T2", sicknl=(4230, 4240, 4220), blimp=7900))
    output = tmp_path / "decision.json"
    assert decide.main(["--root", root, "--output", str(output)]) == 0
    printed = capsys.readouterr().out
    assert "VRAAG 1: T1 geen verschil tegen S" in printed
    assert "VRAAG 2: T2 helpt tegen T1" in printed
    assert "EINDMODEL (vraag 3): T1 (epoch1-653000): T2 helpt, maar is slechter op blimp-nl (regel 2)" in printed
    decision = json.loads(output.read_text())
    assert decision["final_model"]["arm"] == "T1"
    assert decision["question2"]["primary"]["sicknl"]["difference"] == str(Fraction(90, 3 * 4864))
    assert decision["question2"]["extra"]["blimp-nl"]["pairs"] == -50


def test_question_1_needs_no_t2_and_a_missing_t2_is_an_error(tmp_path, capsys):
    root = write_root(tmp_path, arm("S"), arm("T1", sicknl=(4170, 4180, 4160)))
    assert decide.main(["--root", root, "--question", "1"]) == 0
    assert "VRAAG 1: T1 schaadt tegen S (WSD hier slechter dan cosine)" in capsys.readouterr().out
    assert decide.main(["--root", root]) == 1
    printed = capsys.readouterr().out
    assert printed.startswith("FOUT: ") and printed.rstrip().endswith("epoch2-1306000--dbrd1024--seed42.json: missing")


def test_one_arm_is_checked_on_its_own(tmp_path, capsys):
    root = write_root(tmp_path, arm("S"))
    assert decide.main(["--root", root, "--arm", "s-653000"]) == 0
    assert "ARM s-653000: 19 results, exact and at the protocol's settings" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        decide.main(["--root", root, "--arm", "653000"])
