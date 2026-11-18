"""Tests of experiments/final-v2/release/card_facts.py on records shaped like those of run v2-fw2optout-230m: starts in
bf16 and float32, resumes from earlier checkpoints that discard bf16 bursts, and the data state at the last step."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "final-v2" / "release"))

import card_facts as cf  # noqa: E402

LAST = 1306000


def step(s, skipped):
    return (f"Step {s} | Loss: 0.9000 | Acc: 80.0% | PPL: 2.5 | LR: 4.80e-04 | Grad: 0.30 | Ratio: 1.00e-04 | "
            f"Tok/s: 400000 | Step: 0.300s | Input: 0.001s | Skipped: {skipped}")


def nonfinite(s, skipped):
    return (f"NONFINITE step={s} skipped={skipped} in_a_row=1: 1 step(s) with non-finite gradients left the "
            f"parameters and optimizer state unchanged")


def start(dtype, resumed=None):
    lines = ["", "Model config:", "  Vocab size: 32128", f"  Compute dtype: {dtype}", "  Param dtype: float32",
             "TRAINING START"]
    if resumed is not None:
        lines.append(f"Resuming from micro_step: {resumed} (optimizer_step: {resumed})")
    lines.append(f"Step {(resumed or 0) + 1}: loss=0.9000, logits=[-9.00, 9.00], ")  # the first steps' diagnostics
    return lines


def final(s):
    return f"\n✓ Final checkpoint saved at optimizer_step {s} (micro_step {s})"


def v2_history():
    """bf16 from 0 with a burst after 250,000; float32 250k-270k; bf16 from 270k with a burst after 280k; float32
    280k-320k; bf16 from 320k with a burst after 340k; float32 from 340k (to 380k, stopped at 352k) and from 350k to
    the end. Five NONFINITE lines in the discarded bursts, four in kept bf16 stretches."""
    lines = start("bfloat16")
    lines += [step(50, 0), nonfinite(173540, 5), step(174000, 5), nonfinite(223450, 31), step(250000, 31),
              nonfinite(252700, 40), "\r 19%|##  | 255190/1306000 [..]" + nonfinite(255190, 1837), step(255200, 1837)]
    lines += start("float32", 250000) + [step(250050, 31), step(270000, 31), final(270000)]
    lines += start("bfloat16", 270000) + [nonfinite(275000, 33), step(280000, 33), nonfinite(281800, 50),
                                          nonfinite(286190, 649), step(286200, 649)]
    lines += start("float32", 280000) + [step(320000, 33), final(320000)]
    lines += start("bfloat16", 320000) + [nonfinite(330010, 34), step(340000, 34), nonfinite(344300, 60), step(344300, 60)]
    lines += start("float32", 340000) + [step(350000, 34), step(352000, 34)]
    lines += start("float32", 350000) + [step(1040000, 34), step(LAST, 34), final(LAST)]
    return "\n".join(lines) + "\n"


def state(i, epoch=2, finished=58673718, completed=58709312, tokens=40503605562, counted=True, s=LAST):
    return {"step": s, "identity": {"process_count": 4, "tokenizer": "yhavinga/dutch-llama-tokenizer"},
            "state": {"epoch": epoch, "pipeline": None, "documents_of_finished_epochs": finished,
                      "pieces_read": 1, "tokens_read": tokens, "documents_completed": completed,
                      "counted_from_start": counted}}


def test_the_kept_steps_and_their_dtypes():
    starts = cf.starts_of(cf.coordinator_log({"w0.log": v2_history()})[1])
    precision, nf = cf.precision_and_nonfinite(starts, LAST)
    assert [(r["from"], r["to"], r["dtype"]) for r in precision["ranges"]] == [
        (0, 250000, "bfloat16"), (250000, 270000, "float32"), (270000, 280000, "bfloat16"),
        (280000, 320000, "float32"), (320000, 340000, "bfloat16"), (340000, LAST, "float32")]
    assert precision["steps"] == {"bfloat16": 280000, "float32": 1026000}
    # The bursts that a resume discarded do not count; the kept ones add up to the last Skipped count
    assert nf["skipped"] == 34 and nf["lines"] == 4 and nf["discarded_lines"] == 5
    assert (nf["first_step"], nf["last_step"]) == (173540, 330010)
    assert nf["lines_by_dtype"] == {"bfloat16": 4}


def test_the_coordinator_log_need_not_be_worker_0():
    others = "\n".join([step(50, 0), "Pass 1 of shard 1/4"])
    name, lines = cf.coordinator_log({"w0.log": others, "w2.log": v2_history(), "w3.log": ""})
    assert name == "w2.log"
    with pytest.raises(cf.FactsError, match="exactly one coordinator log"):
        cf.coordinator_log({"w0.log": v2_history(), "w2.log": v2_history()})
    with pytest.raises(cf.FactsError, match="exactly one coordinator log"):
        cf.coordinator_log({"w0.log": others})


def test_counts_that_do_not_add_up_are_errors():
    text = v2_history().replace(step(LAST, 34), step(LAST, 35))
    with pytest.raises(cf.FactsError, match="Skipped 35"):
        cf.precision_and_nonfinite(cf.starts_of(cf.coordinator_log({"w": text})[1]), LAST)
    # The run did not reach the last step
    text = v2_history().replace(final(LAST), final(1300000))
    with pytest.raises(cf.FactsError, match="final checkpoint is 1300000"):
        cf.precision_and_nonfinite(cf.starts_of(cf.coordinator_log({"w": text})[1]), LAST)
    # A resume from a checkpoint that the start before it cannot have written
    text = v2_history().replace(step(352000, 34), "").replace(step(350000, 34), step(345000, 34))
    with pytest.raises(cf.FactsError, match="logged no step past 345000"):
        cf.precision_and_nonfinite(cf.starts_of(cf.coordinator_log({"w": text})[1]), LAST)


def test_a_start_that_trained_nothing_is_passed_over_and_a_second_fresh_start_is_an_error():
    lines = start("bfloat16") + start("bfloat16") + [step(50, 0), step(LAST, 0), final(LAST)]
    precision, nf = cf.precision_and_nonfinite(cf.starts_of(lines), LAST)
    assert precision["ranges"] == [{"from": 0, "to": LAST, "dtype": "bfloat16"}] and nf["skipped"] == 0
    lines = start("bfloat16") + [step(50, 0)] + start("float32") + [step(50, 0), step(LAST, 0), final(LAST)]
    with pytest.raises(cf.FactsError, match="from step 0 after"):
        cf.precision_and_nonfinite(cf.starts_of(lines), LAST)


def test_data_facts_of_the_processes():
    d = cf.data_facts([state(i) for i in range(4)], LAST, 131072, 230110592)
    assert d["tokens_read"] == 4 * 40503605562
    assert d["real_token_share"] == pytest.approx(0.9464, abs=1e-4)
    assert d["passes_min"] == pytest.approx(2 + 35594 / 29336859)
    assert d["finished_passes"] == [2]
    assert d["training_flop"] == 6 * 230110592 * 4 * 40503605562
    with pytest.raises(cf.FactsError, match="counted_from_start"):
        cf.data_facts([state(0, counted=False)] + [state(i) for i in range(1, 4)], LAST, 131072, 1)
    with pytest.raises(cf.FactsError, match="not 1306000"):
        cf.data_facts([state(0, s=1300000)] + [state(i) for i in range(1, 4)], LAST, 131072, 1)
    with pytest.raises(cf.FactsError, match="3 data state files"):
        cf.data_facts([state(i) for i in range(3)], LAST, 131072, 1)


def test_main_writes_the_facts(tmp_path, monkeypatch):
    for i in range(4):
        (tmp_path / f"process_{i}.json").write_text(json.dumps(state(i)))
    (tmp_path / "w0.log").write_text(v2_history())
    (tmp_path / "w1.log").write_text(step(50, 0))
    out = tmp_path / "facts.json"
    argv = ["card_facts.py", "--data-state", *[str(tmp_path / f"process_{i}.json") for i in range(4)],
            "--trainer-logs", str(tmp_path / "w0.log"), str(tmp_path / "w1.log"), "--last-step", str(LAST),
            "--parameters", "230110592", "--out", str(out)]
    monkeypatch.setattr(sys, "argv", argv)
    assert cf.main() == 0
    facts = json.loads(out.read_text())
    assert facts["coordinator_log"].endswith("w0.log")
    assert facts["precision"]["steps"]["float32"] == 1026000
    (tmp_path / "w1.log").write_text(v2_history())
    assert cf.main() == 1
