"""The decision rule of the shape ablation (experiments/v3/decide_v3.py) on made-up results of the five arms."""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "experiments" / "v3"))
sys.path.insert(0, str(HERE))

import decide_v3  # noqa: E402
from test_final_decide import arm_files, primary_result  # noqa: E402
from test_final_v2_decide import dbrd512_files  # noqa: E402

decide = decide_v3.decide
BUCKET = "gs://bucket/mbert"
BASE = dict(sicknl=(4200, 4210, 4190), dbrd512=(2026, 2030, 2022))


def arm(name_arm, sicknl=(4200, 4210, 4190), dbrd512=(2026, 2030, 2022), **extra):
    name = decide_v3.name_of(name_arm)
    files = {path: result for path, result in arm_files(name, decide_v3.checkpoint_of(name_arm, BUCKET), **extra).items()
             if "--dbrd1024--" not in path and "--sicknl--" not in path}
    for seed, k in zip(decide.SEEDS, sicknl):
        files[f"dumb/{name}--sicknl--seed{seed}.json"] = primary_result(name, "sicknl", k)
    files.update(dbrd512_files(name, dbrd512))
    return files


def run(tmp_path, capsys, arms, *flags):
    for files in arms:
        for relative, result in files.items():
            path = tmp_path / "v3-ablation" / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(result))
    out = tmp_path / "d.json"
    code = decide_v3.main(["--bucket", str(tmp_path), "--output", str(out), *flags])
    printed = capsys.readouterr().out
    return code, printed, (json.loads(out.read_text()) if out.exists() else None)


def checkpoint_fix(files, tmp_path, name_arm):
    # The BLiMP result carries the checkpoint of its arm in the bucket under test
    name = decide_v3.name_of(name_arm)
    files[f"blimp-nl/{name}.json"]["checkpoint"] = decide_v3.checkpoint_of(name_arm, str(tmp_path))
    return files


def arms(tmp_path, **changes):
    return [checkpoint_fix(arm(a, **changes.get(a, {})), tmp_path, a) for a in ("A", "B", "C", "D", "A2")]


def test_all_equal_keeps_a(tmp_path, capsys):
    code, printed, d = run(tmp_path, capsys, arms(tmp_path))
    assert code == 0, printed
    assert d["outcome"] == "A" and all(not p["qualifies"] for p in d["candidates"].values())


def test_a_candidate_better_on_sicknl_and_nowhere_worse_wins(tmp_path, capsys):
    code, printed, d = run(tmp_path, capsys, arms(tmp_path, B={"sicknl": (4300, 4310, 4290)}))
    assert code == 0, printed
    assert d["candidates"]["B"]["verdict"] == "helpt" and d["outcome"] == "B"


def test_better_on_sicknl_but_worse_on_an_extra_task_does_not_qualify(tmp_path, capsys):
    code, printed, d = run(tmp_path, capsys, arms(tmp_path, C={"sicknl": (4300, 4310, 4290),
                                                                "conll2002": (0.89, 0.892, 0.888)}))
    assert code == 0, printed
    assert d["candidates"]["C"]["verdict"] == "helpt" and d["candidates"]["C"]["extra_worse"] == ["conll2002"]
    assert d["outcome"] == "A"


def test_no_difference_with_an_extra_task_better_qualifies_and_two_tie_go_to_yeb(tmp_path, capsys):
    code, printed, d = run(tmp_path, capsys, arms(tmp_path, D={"squadnl": (0.85, 0.852, 0.848)},
                                                  B={"blimp": 8100}))
    assert code == 0, printed
    assert d["candidates"]["D"]["qualifies"] and d["candidates"]["B"]["qualifies"]
    assert d["outcome"] == "Yeb"


def test_worse_on_dbrd_harms(tmp_path, capsys):
    code, printed, d = run(tmp_path, capsys, arms(tmp_path, D={"dbrd512": (1990, 1994, 1986), "squadnl": (0.85, 0.852, 0.848)}))
    assert code == 0, printed
    assert d["candidates"]["D"]["verdict"] == "schaadt" and d["outcome"] == "A"


def test_the_pretraining_noise_of_a2_raises_the_threshold(tmp_path, capsys):
    # B 1.6 points better on SICK-NL: more than the seed ranges (0.4), less than |A - A2| (2.1)
    changes = {"B": {"sicknl": (4280, 4290, 4270)}, "A2": {"sicknl": (4300, 4310, 4290)}}
    code, printed, d = run(tmp_path, capsys, arms(tmp_path, **changes))
    assert code == 0, printed
    assert d["candidates"]["B"]["cells"]["sicknl"]["outcome"] == "gelijk" and d["outcome"] == "A"
    without = [a for a in arms(tmp_path, **changes)][:4]
    code, printed, d = run(tmp_path / "w", capsys, [checkpoint_fix(a, tmp_path / "w", n) for a, n in zip(without, "ABCD")],
                           "--without-a2")
    assert code == 0, printed
    assert d["candidates"]["B"]["cells"]["sicknl"]["outcome"] == "beter" and d["outcome"] == "B"


def test_a_missing_result_is_fout(tmp_path, capsys):
    files = arms(tmp_path)
    del files[1][f"dumb/{decide_v3.name_of('B')}--sicknl--seed43.json"]
    code, printed, _ = run(tmp_path, capsys, files)
    assert code == 1 and printed.startswith("FOUT")
