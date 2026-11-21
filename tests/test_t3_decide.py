"""The decision rule of T3 (experiments/t3/decide_t3.py) on made-up results: T3-hoofd and T3-tak against T2 and each
other, with the noise of the shape ablation (A against A2) in the thresholds."""
import json
import sys
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "experiments" / "t3"))
sys.path.insert(0, str(HERE))

import decide_t3  # noqa: E402
from test_final_decide import arm_files  # noqa: E402
from test_final_v2_decide import dbrd512_files  # noqa: E402
from test_v3_decide import arm as ablation_arm  # noqa: E402

decide = decide_t3.decide
T2_NAME = decide.ARMS["T2"]["name"]


def t3_files(arm, root, dbrd512=(2028, 2030, 2026), **numbers):
    return {**arm_files(arm["name"], decide_t3.checkpoint_of(arm, root), **numbers), **dbrd512_files(arm["name"], dbrd512)}


def bucket(tmp_path, main=None, branch=None, t2=None, a2=None):
    root = str(tmp_path)
    layout = {
        "t3": {**t3_files(decide_t3.BRANCH, root, **(branch or {})), **({} if main is False else t3_files(decide_t3.MAIN, root, **(main or {})))},
        "final": arm_files(T2_NAME, decide.ARMS["T2"]["checkpoint"], **(t2 or {})),
        "final/robbert": dbrd512_files(T2_NAME, (2028, 2030, 2026), "/dev/shm/robbert_eval/ckpt-"),
        "v3-ablation": {**ablation_arm("A"), **ablation_arm("A2", **(a2 or {}))},
    }
    for where, files in layout.items():
        for relative, result in files.items():
            if where == "v3-ablation" and relative.startswith("blimp-nl/"):
                arm = "A2" if "-a2-" in relative else "A"
                result = {**result, "checkpoint": decide_t3.decide_v3.checkpoint_of(arm, root)}
            path = tmp_path / where / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(result))
    return root


def run(tmp_path, root, capsys, *flags):
    out, md = tmp_path / "d.json", tmp_path / "d.md"
    code = decide_t3.main(["--bucket", root, "--output", str(out), "--markdown", str(md), *flags])
    printed = capsys.readouterr().out
    return code, printed, (json.loads(out.read_text()) if out.exists() else None), (md.read_text() if md.exists() else None)


def test_t3_equal_to_t2_is_not_published(tmp_path, capsys):
    code, printed, d, md = run(tmp_path, bucket(tmp_path), capsys)
    assert code == 0, printed
    assert d["question1"]["verdict"] == "geen verschil" and d["outcome"] == "niet publiceren"
    assert d["question2"]["verdict"] == "geen verschil" and d["question3"]["verdict"] == "geen verschil"
    assert "Uitkomst: niet publiceren" in md


def test_better_on_sicknl_and_nowhere_worse_replaces_t2(tmp_path, capsys):
    root = bucket(tmp_path, main={"sicknl": (4300, 4310, 4290)})
    code, printed, d, _ = run(tmp_path, root, capsys)
    assert code == 0, printed
    assert d["question1"]["verdict"] == "helpt" and d["outcome"] == "vervangt T2"
    assert d["question3"]["verdict"] == "helpt", "the main leg against the branch"


def test_no_difference_with_an_extra_task_better_replaces_t2(tmp_path, capsys):
    code, printed, d, _ = run(tmp_path, bucket(tmp_path, main={"blimp": 8100}), capsys)
    assert code == 0, printed
    assert d["question1"]["extra_better"] == ["blimp-nl"] and d["outcome"] == "vervangt T2"


def test_better_on_sicknl_but_worse_on_an_extra_task_is_next_to_t2(tmp_path, capsys):
    root = bucket(tmp_path, main={"sicknl": (4300, 4310, 4290), "conll2002": (0.89, 0.892, 0.888)})
    code, printed, d, _ = run(tmp_path, root, capsys)
    assert code == 0, printed
    assert d["question1"]["extra_worse"] == ["conll2002"] and d["outcome"] == "naast T2"


def test_harming_on_dbrd1024_is_not_published_even_if_better_elsewhere(tmp_path, capsys):
    root = bucket(tmp_path, main={"dbrd": (1990, 1992, 1988), "blimp": 8100})
    code, printed, d, _ = run(tmp_path, root, capsys)
    assert code == 0, printed
    assert d["question1"]["verdict"] == "schaadt" and d["outcome"] == "niet publiceren"


def test_the_pretraining_noise_of_the_ablation_raises_a_threshold(tmp_path, capsys):
    # SQuAD-NL 0.5 point better than T2 with seed ranges of 0.4: "beter" without noise, "gelijk" with A2 1 point off A
    numbers = {"squadnl": (0.835, 0.837, 0.833)}
    code, printed, d, _ = run(tmp_path, bucket(tmp_path, main=numbers), capsys)
    assert code == 0, printed
    assert d["question1"]["cells"]["squadnl"]["outcome"] == "beter"
    code, printed, d, _ = run(tmp_path, bucket(tmp_path, main=numbers, a2={"squadnl": (0.82, 0.822, 0.818)}), capsys)
    assert code == 0, printed
    assert Fraction(d["question1"]["cells"]["squadnl"]["threshold"]) == Fraction(d["noise"]["squadnl"]) > Fraction(1, 200)
    assert d["question1"]["cells"]["squadnl"]["outcome"] == "gelijk"


def test_dbrd1024_takes_the_noise_of_dbrd512(tmp_path, capsys):
    code, printed, d, _ = run(tmp_path, bucket(tmp_path, a2={"dbrd512": (2000, 2004, 1996)}), capsys)
    assert code == 0, printed
    assert d["noise"]["dbrd1024"] == d["noise"]["dbrd512"] != "0"


def test_a_harmful_third_epoch_reports_the_branch_as_an_alternative(tmp_path, capsys):
    root = bucket(tmp_path, branch={"sicknl": (4300, 4310, 4290)})
    code, printed, d, md = run(tmp_path, root, capsys)
    assert code == 0, printed
    assert d["question3"]["verdict"] == "schaadt" and d["question2"]["verdict"] == "helpt"
    assert d["outcome"] == "niet publiceren", "the outcome stays that of question 1"
    assert d["alternative"]["outcome"] == "vervangt T2" and "Alternatief" in md


def test_branch_only_reports_question_2_and_decides_nothing(tmp_path, capsys):
    root = bucket(tmp_path, main=False, branch={"blimp": 8100})
    code, printed, d, md = run(tmp_path, root, capsys, "--branch-only")
    assert code == 0, printed
    assert "outcome" not in d and "question1" not in d and d["question2"]["extra_better"] == ["blimp-nl"]
    assert "VOORLOPIG" in printed and "Voorlopig (beslist niets)" in md


def test_a_missing_result_is_an_error(tmp_path, capsys):
    root = bucket(tmp_path)
    (tmp_path / "t3" / "dumb" / "t3-1959000--sicknl--seed43.json").unlink()
    code, printed, d, _ = run(tmp_path, root, capsys)
    assert code == 1 and printed.startswith("FOUT") and d is None


def test_t2s_results_as_both_t3_arms_give_not_published(tmp_path, capsys):
    """The control of protocol.md section 11: an equally good, larger model does not replace T2."""
    root = bucket(tmp_path)
    for arm in (decide_t3.MAIN, decide_t3.BRANCH):
        for path in (tmp_path / "final").rglob(f"{T2_NAME}*.json"):
            # T2's DBRD@512 is that of robbert-compare, in final/robbert/dumb/
            relative = path.relative_to(tmp_path / "final").as_posix().removeprefix("robbert/")
            target = tmp_path / "t3" / relative.replace(T2_NAME, arm["name"])
            result = json.loads(path.read_text())
            if "checkpoint" in result:
                result["checkpoint"] = (decide_t3.checkpoint_of(arm, root) if "blimp-nl" in path.parts
                                        else str(result["checkpoint"]).replace(T2_NAME, arm["name"]))
            target.write_text(json.dumps(result))
    code, printed, d, _ = run(tmp_path, root, capsys)
    assert code == 0, printed
    assert all(c["outcome"] == "gelijk" for c in d["question1"]["cells"].values())
    assert d["outcome"] == "niet publiceren"
