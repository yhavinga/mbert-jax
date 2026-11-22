"""The decision rule of T3 (experiments/t3/decide_t3.py) on made-up results: T3-hoofd and T3-tak against T2 and each
other, with the noise of the shape ablation (A against A2) in the thresholds; five seeds (42-46) on DBRD@1024, DBRD@512,
SICK-NL, SQuAD-NL and CoNLL-2002, the means over all seeds and the seed ranges over 42-44 (protocol.md, amendment 1)."""
import json
import sys
from fractions import Fraction
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "experiments" / "t3"))
sys.path.insert(0, str(HERE))

import decide_t3  # noqa: E402
from test_final_decide import arm_files, extra_result, primary_result  # noqa: E402
from test_final_v2_decide import dbrd512_files  # noqa: E402
from test_v3_decide import arm as ablation_arm  # noqa: E402

decide = decide_t3.decide
T2_NAME = decide.ARMS["T2"]["name"]


# arm_files' defaults for the tasks with five seeds; each made-up triple is (m, m + d, m - d), with mean m
FIVE_DEFAULTS = {"dbrd": (2026, 2030, 2022), "sicknl": (4200, 4210, 4190), "squadnl": (0.83, 0.832, 0.828),
                 "conll2002": (0.91, 0.912, 0.908), "dbrd512": (2028, 2030, 2026)}


def five(values):
    """Seeds 42-46 of a made-up result: a triple gets its first value (its mean) for 45 and 46."""
    return tuple(values) if len(values) == 5 else (*values, values[0], values[0])


def seeds_45_46(name, numbers):
    """The results of seeds 45 and 46 of the five tasks, in eval_t3.sh's layout (numbers as arm_files takes them)."""
    values = {task: five(numbers.get(task, default)) for task, default in FIVE_DEFAULTS.items()}
    files = {}
    for i, seed in ((3, 45), (4, 46)):
        files[f"dumb/{name}--dbrd1024--seed{seed}.json"] = primary_result(name, "dbrd1024", values["dbrd"][i])
        files[f"dumb/{name}--sicknl--seed{seed}.json"] = primary_result(name, "sicknl", values["sicknl"][i])
        files[f"dumb/{name}--dbrd512--seed{seed}.json"] = primary_result(name, "dbrd1024", values["dbrd512"][i], max_length=512)
        for task in ("squadnl", "conll2002"):
            files[f"dumb-extra/{task}/{name}--{task}--seed{seed}.json"] = extra_result(name, task, seed, values[task][i])
    return files


def three(numbers):
    return {task: tuple(v)[:3] if isinstance(v, tuple) else v for task, v in numbers.items()}


def t3_files(arm, root, **numbers):
    dbrd512 = tuple(numbers.pop("dbrd512", FIVE_DEFAULTS["dbrd512"]))[:3]
    return {**arm_files(arm["name"], decide_t3.checkpoint_of(arm, root), **three(numbers)), **dbrd512_files(arm["name"], dbrd512),
            **seeds_45_46(arm["name"], {**numbers, "dbrd512": five(dbrd512) if len(numbers.get("dbrd512", ())) != 5 else numbers["dbrd512"]})}


def bucket(tmp_path, main=None, branch=None, t2=None, a2=None):
    root = str(tmp_path)
    t2 = dict(t2 or {})
    t2_dbrd512 = t2.pop("dbrd512", FIVE_DEFAULTS["dbrd512"])
    layout = {
        "t3": {**t3_files(decide_t3.BRANCH, root, **(branch or {})), **({} if main is False else t3_files(decide_t3.MAIN, root, **(main or {}))),
               **seeds_45_46(T2_NAME, {**t2, "dbrd512": t2_dbrd512})},
        "final": arm_files(T2_NAME, decide.ARMS["T2"]["checkpoint"], **three(t2)),
        "final/robbert": dbrd512_files(T2_NAME, tuple(t2_dbrd512)[:3], "/dev/shm/robbert_eval/ckpt-"),
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


def no_label(d, md, printed):
    """Amendment 1: question 1 is a comparison; T3 is a new model, so no outcome replaces, joins or drops T2."""
    assert "outcome" not in d and "reason" not in d and "outcome" not in d["question1"]
    assert d["question1"]["verdict"] in decide_t3.OUTCOMES
    for label in ("vervangt T2", "naast T2", "niet publiceren", "UITKOMST", "Uitkomst"):
        assert label not in md and label not in printed, label


def test_t3_equal_to_t2_is_no_difference_without_a_publication_label(tmp_path, capsys):
    code, printed, d, md = run(tmp_path, bucket(tmp_path), capsys)
    assert code == 0, printed
    assert d["question1"]["verdict"] == "geen verschil" and d["question1"]["better"] == d["question1"]["worse"] == []
    assert d["question2"]["verdict"] == "geen verschil" and d["question3"]["verdict"] == "geen verschil"
    no_label(d, md, printed)
    assert "VRAAG 1 (T3-hoofd tegen T2): geen verschil; beter op geen; slechter op geen" in printed
    assert "nieuw model, geen vervanging van T2" in md and "alleen op Yebs woord" in md


def test_better_on_sicknl_and_nowhere_worse_helps(tmp_path, capsys):
    root = bucket(tmp_path, main={"sicknl": (4300, 4310, 4290)})
    code, printed, d, md = run(tmp_path, root, capsys)
    assert code == 0, printed
    assert d["question1"]["verdict"] == "helpt" and d["question1"]["better"] == ["sicknl"]
    assert d["question3"]["verdict"] == "helpt", "the main leg against the branch"
    no_label(d, md, printed)


def test_no_difference_with_an_extra_task_better_is_reported_per_task(tmp_path, capsys):
    code, printed, d, md = run(tmp_path, bucket(tmp_path, main={"blimp": 8100}), capsys)
    assert code == 0, printed
    assert d["question1"]["verdict"] == "geen verschil" and d["question1"]["extra_better"] == ["blimp-nl"]
    no_label(d, md, printed)
    assert "beter op BLiMP-NL; slechter op geen" in printed


def test_better_on_sicknl_but_worse_on_an_extra_task_reports_both(tmp_path, capsys):
    root = bucket(tmp_path, main={"sicknl": (4300, 4310, 4290), "conll2002": (0.89, 0.892, 0.888)})
    code, printed, d, md = run(tmp_path, root, capsys)
    assert code == 0, printed
    assert d["question1"]["verdict"] == "helpt" and d["question1"]["extra_worse"] == d["question1"]["worse"] == ["conll2002"]
    no_label(d, md, printed)


def test_harming_on_dbrd1024_is_schaadt_even_if_better_elsewhere(tmp_path, capsys):
    root = bucket(tmp_path, main={"dbrd": (1990, 1992, 1988), "blimp": 8100})
    code, printed, d, md = run(tmp_path, root, capsys)
    assert code == 0, printed
    assert d["question1"]["verdict"] == "schaadt" and d["question1"]["better"] == ["blimp-nl"]
    no_label(d, md, printed)


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
    assert d["question1"]["verdict"] == "geen verschil" and "outcome" not in d
    assert d["alternative"]["summary"].startswith("helpt; beter op SICK-NL") and "Alternatief" in md


def test_branch_only_reports_question_2_and_decides_nothing(tmp_path, capsys):
    root = bucket(tmp_path, main=False, branch={"blimp": 8100})
    code, printed, d, md = run(tmp_path, root, capsys, "--branch-only")
    assert code == 0, printed
    assert "outcome" not in d and "question1" not in d and d["question2"]["extra_better"] == ["blimp-nl"]
    assert "VOORLOPIG" in printed and "Voorlopig (beslist niets)" in md
    assert "VOORLOPIG (vraag 2, tak tegen T2, beslist niets): geen verschil; beter op BLiMP-NL" in printed


def test_a_missing_result_is_an_error(tmp_path, capsys):
    root = bucket(tmp_path)
    (tmp_path / "t3" / "dumb" / "t3-1959000--sicknl--seed43.json").unlink()
    code, printed, d, _ = run(tmp_path, root, capsys)
    assert code == 1 and printed.startswith("FOUT") and d is None


def test_t2s_results_as_both_t3_arms_give_no_difference(tmp_path, capsys):
    """The control of protocol.md section 11 (amendment 1): T2's results as both T3 arms are equal everywhere."""
    root = bucket(tmp_path)
    for arm in (decide_t3.MAIN, decide_t3.BRANCH):
        for path in [*(tmp_path / "final").rglob(f"{T2_NAME}*.json"), *(tmp_path / "t3").rglob(f"{T2_NAME}*.json")]:
            # T2's DBRD@512 is that of robbert-compare, in final/robbert/dumb/
            relative = path.relative_to(tmp_path / ("t3" if "t3" in path.relative_to(tmp_path).parts[:1] else "final")).as_posix().removeprefix("robbert/")
            target = tmp_path / "t3" / relative.replace(T2_NAME, arm["name"])
            result = json.loads(path.read_text())
            if "checkpoint" in result:
                result["checkpoint"] = (decide_t3.checkpoint_of(arm, root) if "blimp-nl" in path.parts
                                        else str(result["checkpoint"]).replace(T2_NAME, arm["name"]))
            target.write_text(json.dumps(result))
    code, printed, d, _ = run(tmp_path, root, capsys)
    assert code == 0, printed
    assert all(c["outcome"] == "gelijk" for c in d["question1"]["cells"].values())
    assert d["question1"]["verdict"] == "geen verschil" and "outcome" not in d


def test_the_difference_is_over_five_seeds_and_the_threshold_takes_the_ranges_over_42_to_44(tmp_path, capsys):
    """Amendment 1: seeds 45 and 46 move the mean (4,280 against T2's 4,200 of 4,864); the threshold is the seed range
    over 42-44 (20/4,864), not that over 42-46 (210/4,864), which is reported only."""
    root = bucket(tmp_path, main={"sicknl": (4200, 4210, 4190, 4400, 4400)})
    code, printed, d, md = run(tmp_path, root, capsys)
    assert code == 0, printed
    c = d["question1"]["cells"]["sicknl"]
    assert Fraction(c["difference"]) == Fraction(80, 4864) and Fraction(c["threshold"]) == Fraction(20, 4864)
    assert c["outcome"] == "beter" and d["question1"]["verdict"] == "helpt"
    arm = d["arms"]["T3"]["sicknl"]
    assert len(arm["values"]) == 5 and Fraction(arm["mean"]) == Fraction(4280, 4864)
    assert Fraction(arm["range"]) == Fraction(20, 4864) and Fraction(arm["range_all"]) == Fraction(210, 4864)
    assert "seed-range over 42–44" in md


def test_an_f1_task_counts_its_five_seeds_and_pos_its_three(tmp_path, capsys):
    root = bucket(tmp_path, main={"squadnl": (0.83, 0.832, 0.828, 0.85, 0.85)})
    code, printed, d, _ = run(tmp_path, root, capsys)
    assert code == 0, printed
    squad = d["question1"]["cells"]["squadnl"]
    assert Fraction(squad["difference"]) == (Fraction(0.83) + Fraction(0.832) + Fraction(0.828) + 2 * Fraction(0.85)) / 5 - (
        Fraction(0.83) + Fraction(0.832) + Fraction(0.828) + 2 * Fraction(0.83)) / 5
    assert len(d["arms"]["T3"]["pos-alpino"]["values"]) == len(d["arms"]["T2"]["pos-lassysmall"]["values"]) == 3
    assert [len(d["arms"][a][t]["values"]) for a in ("T3", "T3b", "T2") for t in decide_t3.FIVE_SEEDS] == [5] * 15


def test_t2s_seeds_45_and_46_are_read_from_t3s_results(tmp_path, capsys):
    """T2 gets 45 and 46 from eval_t3.sh (in <bucket>/t3/), next to its own 42-44: here they make T2 better on SICK-NL."""
    root = bucket(tmp_path, t2={"sicknl": (4200, 4210, 4190, 4400, 4400)})
    code, printed, d, _ = run(tmp_path, root, capsys)
    assert code == 0, printed
    assert Fraction(d["question1"]["cells"]["sicknl"]["difference"]) == Fraction(-80, 4864)
    assert d["question1"]["verdict"] == d["question2"]["verdict"] == "schaadt"
    assert d["arms"]["T2"]["sicknl"]["values"][3:] == ["4400/4864", "4400/4864"]


@pytest.mark.parametrize("missing", ["t3/dumb/t3-1959000--dbrd512--seed46.json",
                                     "t3/dumb-extra/conll2002/t3-epoch2-1306000--conll2002--seed45.json",
                                     "t3/dumb/epoch2-1306000--sicknl--seed45.json",
                                     "t3/dumb-extra/squadnl/epoch2-1306000--squadnl--seed46.json"])
def test_a_missing_seed_45_or_46_is_an_error(tmp_path, capsys, missing):
    root = bucket(tmp_path)
    (tmp_path / missing).unlink()
    code, printed, d, _ = run(tmp_path, root, capsys)
    assert code == 1 and printed.startswith("FOUT") and missing.rsplit("/", 1)[1] in printed and d is None


def test_pos_needs_no_seed_45(tmp_path, capsys):
    root = bucket(tmp_path)
    assert not list((tmp_path / "t3").rglob("*pos-*--seed45.json"))
    code, printed, _, _ = run(tmp_path, root, capsys)
    assert code == 0, printed
