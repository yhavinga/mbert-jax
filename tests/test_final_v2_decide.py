"""The decision rule of experiments/final-v2/protocol.md (decide_v2.py) on made-up results: V against T2 (step 1) and
against RobBERT-2023 (step 2), and the outcomes A, B and C."""
import json
import sys
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "experiments" / "final-v2"))
sys.path.insert(0, str(HERE))

import decide_v2  # noqa: E402
from test_final_decide import arm_files, primary_result  # noqa: E402

decide = decide_v2.decide
T2_CHECKPOINT = decide.ARMS["T2"]["checkpoint"]


def dbrd512_files(name, ks, checkpoint_dir="/dev/shm/final_eval/ckpt"):
    return {f"dumb/{name}--dbrd512--seed{seed}.json": primary_result(name, "dbrd1024", k, max_length=512, checkpoint=f"{checkpoint_dir}{name}")
            for seed, k in zip(decide.SEEDS, ks)}


def v_files(dbrd512=(2028, 2030, 2026), **numbers):
    return {**arm_files(decide_v2.V_NAME, decide_v2.V_CHECKPOINT, **numbers), **dbrd512_files(decide_v2.V_NAME, dbrd512)}


def r23_files(dbrd512=(2000, 2006, 1994), sicknl=(4080, 4090, 4070), blimp=7030):
    files = {}
    for path, result in arm_files(decide_v2.R23_NAME, None).items():
        if path.startswith("dumb-extra/"):
            files[path] = result
    for seed, k in zip(decide.SEEDS, sicknl):
        files[f"dumb/robbert-2023--sicknl--seed{seed}.json"] = primary_result("robbert-2023", "sicknl", k, checkpoint="/dev/shm/robbert_eval/ckpt-robbert-2023")
    files.update(dbrd512_files("robbert-2023", dbrd512, checkpoint_dir="/dev/shm/robbert_eval/ckpt-"))
    files["blimp-nl/robbert-2023.json"] = {**decide_v2.BLIMP_HF_SETTINGS, "checkpoint": "DTAI-KULeuven/robbert-2023-dutch-base",
                                           "accuracy": blimp / decide.BLIMP_PAIRS}
    return files


def bucket(tmp_path, v, t2=None, t2_dbrd512=(2028, 2030, 2026), r23=None):
    t2 = arm_files(decide.ARMS["T2"]["name"], T2_CHECKPOINT) if t2 is None else t2
    r23 = r23_files() if r23 is None else r23
    layout = {"final-v2": v, "final": t2, "final/robbert": {**r23, **dbrd512_files(decide.ARMS["T2"]["name"], t2_dbrd512, "/dev/shm/robbert_eval/ckpt-")}}
    for where, files in layout.items():
        for relative, result in files.items():
            path = tmp_path / where / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(result))
    return str(tmp_path)


def run(tmp_path, root, capsys):
    out, md = tmp_path / "d.json", tmp_path / "d.md"
    code = decide_v2.main(["--bucket", root, "--output", str(out), "--markdown", str(md)])
    printed = capsys.readouterr().out
    return code, printed, (json.loads(out.read_text()) if out.exists() else None), (md.read_text() if md.exists() else None)


def test_v_equal_to_t2_replaces_it(tmp_path, capsys):
    code, printed, d, md = run(tmp_path, bucket(tmp_path, v_files()), capsys)
    assert code == 0, printed
    assert "STAP 1: V geen verschil tegen T2" in printed
    assert d["outcome"] == "A" and d["outcome_name"] == "vervangt T2"
    assert "**A, vervangt T2**" in md
    assert d["step1"]["extra"]["dbrd512"]["outcome"] == "gelijk"


def test_v_harming_on_sicknl_but_not_below_robbert_is_next_to_t2(tmp_path, capsys):
    code, printed, d, _ = run(tmp_path, bucket(tmp_path, v_files(sicknl=(4150, 4160, 4140))), capsys)
    assert code == 0, printed
    assert d["step1"]["verdict"] == "schaadt"
    assert d["outcome"] == "B" and "V schaadt tegen T2" in d["reason"]
    assert all(c["outcome"] != "slechter" for c in d["step2"]["deciding"].values())


def test_no_difference_but_an_extra_test_worse_is_not_a(tmp_path, capsys):
    # POS LassySmall 60 words lower on every seed: more than the ranges (20 words)
    code, printed, d, _ = run(tmp_path, bucket(tmp_path, v_files(lassysmall=(28340, 28350, 28330))), capsys)
    assert code == 0, printed
    assert d["step1"]["verdict"] == "geen verschil"
    assert d["step1"]["extra"]["pos-lassysmall"]["outcome"] == "slechter"
    assert d["outcome"] == "B" and "slechter dan T2 op pos-lassysmall" in d["reason"]


def test_worse_than_robbert_2023_on_a_deciding_task_is_c(tmp_path, capsys):
    v = v_files(sicknl=(4050, 4060, 4040))
    code, printed, d, _ = run(tmp_path, bucket(tmp_path, v), capsys)
    assert code == 0, printed
    assert d["step2"]["deciding"]["sicknl"]["outcome"] == "slechter"
    assert d["outcome"] == "C" and d["outcome_name"] == "Yeb beslist opnieuw"
    # BLiMP-NL decides too: 0.5 point is 42 pairs
    code, _, d, _ = run(tmp_path, bucket(tmp_path, v_files(blimp=7030 - 43)), capsys)
    assert d["step2"]["deciding"]["blimp-nl"]["outcome"] == "slechter" and d["outcome"] == "C"
    code, _, d, _ = run(tmp_path, bucket(tmp_path, v_files(blimp=7030 - 42)), capsys)
    assert d["step2"]["deciding"]["blimp-nl"]["outcome"] == "gelijk"


def test_dbrd512_is_compared_with_t2s_run_of_robbert_compare(tmp_path, capsys):
    code, printed, d, _ = run(tmp_path, bucket(tmp_path, v_files(dbrd512=(2000, 2002, 1998)), t2_dbrd512=(2030, 2032, 2028)), capsys)
    assert code == 0, printed
    c = d["step1"]["extra"]["dbrd512"]
    assert c["outcome"] == "slechter" and c["difference"] == str(Fraction(-90, 3 * 2112)) and c["threshold"] == str(Fraction(4, 2112))


def test_a_missing_result_of_v_is_an_error(tmp_path, capsys):
    v = v_files()
    del v[f"dumb/{decide_v2.V_NAME}--dbrd512--seed43.json"]
    code, printed, d, _ = run(tmp_path, bucket(tmp_path, v), capsys)
    assert code == 1 and d is None
    assert printed.rstrip().endswith(f"{decide_v2.V_NAME}--dbrd512--seed43.json: missing")


def test_v_alone_is_checked(tmp_path, capsys):
    root = bucket(tmp_path, v_files())
    assert decide_v2.main(["--bucket", root, "--arm", "V"]) == 0
    assert "ARM V: 22 results, exact and at the protocol's settings" in capsys.readouterr().out
