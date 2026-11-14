"""experiments/robbert/report.py on made-up results (the helpers of test_final_decide.py): the table, the comparisons
in decide.py's form, the offset diagnostic, and FOUT for a missing or other result."""
import json
import sys
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "experiments" / "robbert"))

import report  # noqa: E402
from test_final_decide import arm, decide, extra_result  # noqa: E402

QUESTIONS = 1183


def squad_em(files, name, ks):
    for seed, k in zip(decide.SEEDS, ks):
        files[f"dumb-extra/squadnl/{name}--squadnl--seed{seed}.json"]["test"].update(exact_match=k / QUESTIONS, questions=QUESTIONS)


def primary_512(name, task, k):
    n = report.PRIMARY_512[task]["n"]
    dbrd = task == "dbrd512"
    return {"task": "dbrd" if dbrd else "sicknl", "checkpoint": f"/dev/shm/robbert_eval/ckpt-{name}", "test_accuracy": k / n,
            "epochs": 3, "learning_rate": 5e-05, "batch_size": 12 if dbrd else 64, "gradient_accumulation_steps": 4 if dbrd else 1,
            "num_devices": 4, "max_length": 512, "pad_to_multiple_of": 128 if dbrd else 1}


def robbert_files(name, dbrd, squadnl, blimp):
    files = {}
    for seed, k in zip(decide.SEEDS, dbrd):
        files[f"dumb/{name}--dbrd512--seed{seed}.json"] = primary_512(name, "dbrd512", k)
        files[f"dumb/{name}--sicknl--seed{seed}.json"] = primary_512(name, "sicknl", 4100 + seed)
    for task, values in (("squadnl", squadnl), ("conll2002", (0.9, 0.901, 0.899))):
        for seed, v in zip(decide.SEEDS, values):
            files[f"dumb-extra/{task}/{name}--{task}--seed{seed}.json"] = extra_result(name, task, seed, v)
    for task, n in (("pos-lassysmall", 28995), ("pos-alpino", 11046)):
        for seed in decide.SEEDS:
            files[f"dumb-extra/{task}/{name}--{task}--seed{seed}.json"] = extra_result(name, task, seed, (n - 600 - seed) / n, words=n)
    squad_em(files, name, (700, 702, 698))
    files[f"blimp-nl/{name}.json"] = {**report.BLIMP_HF_SETTINGS, "accuracy": blimp / decide.BLIMP_PAIRS}
    return files


def world():
    files = {f"mem/{k}": v for k, v in {**arm("T2", squadnl=(0.85, 0.851, 0.849)), **arm("S")}.items()}
    for key, ks in (("T2", (850, 851, 849)), ("S", (840, 841, 839))):
        name = decide.ARMS[key]["name"]
        for seed, k in zip(decide.SEEDS, ks):
            files[f"mem/dumb-extra/squadnl/{name}--squadnl--seed{seed}.json"]["test"].update(exact_match=k / QUESTIONS, questions=QUESTIONS)
    robbert = {**robbert_files("robbert-2023", (1970, 1975, 1972), (0.80, 0.802, 0.798), 7800),
               **robbert_files("robbert-2026", (1980, 1960, 1990), (0.81, 0.79, 0.80), 7990)}
    for seed in decide.SEEDS:
        robbert[f"dumb/epoch2-1306000--dbrd512--seed{seed}.json"] = primary_512("epoch2-1306000", "dbrd512", 2020 + seed)
    raw = extra_result("robbert-2023-rawoffsets", "squadnl", 42, 0.70)
    raw["test"].update(exact_match=500 / QUESTIONS, questions=QUESTIONS, f1=0.70, no_valid_span=3)
    robbert["dumb-extra/squadnl/robbert-2023-rawoffsets--squadnl--seed42.json"] = raw
    files.update({f"mem/robbert/{k}": v for k, v in robbert.items()})
    return files


def reader(files):
    def read(root, relative):
        key = f"{root}/{relative}"
        if key not in files:
            raise decide.Fout(f"{key}: missing")
        return json.loads(json.dumps(files[key]))
    return read


def test_the_report_has_every_model_and_task_and_compares_as_decide(tmp_path, capsys):
    assert report.main(["--root", "mem", "--output-dir", str(tmp_path)], read=reader(world())) == 0
    md = (tmp_path / "robbert-compare.md").read_text()
    data = json.loads((tmp_path / "robbert-compare.json").read_text())
    table = data["table"]
    assert table["T2"]["dbrd512"]["mean"]["exact"] == str(Fraction(2020 * 3 + 129, 3 * 2112))
    assert "dbrd512" not in table["S"] and "dbrd1024" not in table["robbert-2023"]
    # SQuAD-NL F1: T2 0.85 [0.002] against RobBERT-2023 0.80 [0.004]: better; against RobBERT-2026 0.80 [0.02]: better
    assert data["comparisons"]["T2|robbert-2023"]["squadnl"]["outcome"] == "beter"
    # BLiMP-NL: 7990 against 7800 pairs is more than 42 pairs
    assert data["comparisons"]["robbert-2026|robbert-2023"]["blimp"]["outcome"] == "beter"
    # DBRD@512 of RobBERT-2026 has a range of 30/2112, more than the difference to RobBERT-2023
    assert data["comparisons"]["robbert-2026|robbert-2023"]["dbrd512"]["outcome"] == "gelijk"
    assert "| SQuAD-NL v1.1 | EM |" in md and "| offsets van de tokenizer | 70,00 | 42,27 | 3 |" in md
    assert "geen verschil" in md and "DBRD@1024 | acc | — | — |" in md


def test_a_missing_result_is_fout(tmp_path, capsys):
    files = world()
    del files["mem/robbert/dumb/robbert-2026--sicknl--seed43.json"]
    assert report.main(["--root", "mem", "--output-dir", str(tmp_path)], read=reader(files)) == 1
    assert "FOUT: mem/robbert/dumb/robbert-2026--sicknl--seed43.json: missing" in capsys.readouterr().out
    assert not (tmp_path / "robbert-compare.md").exists()


def test_a_result_at_another_length_is_fout(tmp_path, capsys):
    files = world()
    files["mem/robbert/dumb/robbert-2023--dbrd512--seed44.json"]["max_length"] = 1024
    assert report.main(["--root", "mem", "--output-dir", str(tmp_path)], read=reader(files)) == 1
    assert "max_length is 1024" in capsys.readouterr().out
