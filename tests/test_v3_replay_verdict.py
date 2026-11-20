"""experiments/v3/replay_verdict.py on synthetic trainer logs: the starts of a run, and criteria S1-S3 of protocol.md."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "v3"))

import replay_verdict as rv  # noqa: E402


def start(dtype, resume=None, attention=None, z=None):
    lines = [f"  Compute dtype: {dtype}"]
    if attention:
        lines.append(f"  Attention logits: {attention}")
    if z is not None:
        lines.append(f"  Z-loss: {z:g} x mean log(Z)^2 of the MLM logits")
    if resume is not None:
        lines.append(f"Data state restored on process 0 at step {resume}: {{}}")
        lines.append(f"Resuming from micro_step: {resume} (optimizer_step: {resume})")
    return lines


def steps(lo, hi, loss, skipped=31, z=None, seconds=0.22, every=50):
    out = []
    for s in range(lo + every, hi + 1, every):
        value = loss(s) if callable(loss) else loss
        count = skipped(s) if callable(skipped) else skipped
        out.append(f"Step {s} | Loss: {value:.4f} | Acc: 78.0% | PPL: 2.6 | LR: 4.80e-04 | Grad: 0.22 | Ratio: 5e-08 | "
                   f"Tok/s: 580000 | Step: {seconds:.3f}s | Input: 0.010s | Skipped: {count}"
                   + (f" | Z: {z:.4f}" if z is not None else ""))
    return out


def v2_log():
    lines = start("bfloat16") + steps(0, 255_200, 0.96)
    lines += [f"NONFINITE step={s} skipped={31 + k + 1} in_a_row=1: 1 step(s) with non-finite gradients left the "
              f"parameters and optimizer state unchanged" for k, s in enumerate(range(252_700, 253_700, 10))]
    lines += ["EVAL step=250000 loss=0.95503 accuracy=78.0 perplexity=2.6 masked=1"]
    lines += start("float32", 250_000) + steps(250_000, 270_000, 0.96, seconds=0.30)
    lines += start("bfloat16", 270_000) + steps(270_000, 286_200, 0.955) + ["junk EVAL step=275000 loss=0.94703 accuracy=1"]
    lines += start("float32", 280_000) + steps(280_000, 320_000, 0.95, seconds=0.30) + ["EVAL step=300000 loss=0.94305 a"]
    lines += start("bfloat16", 320_000) + steps(320_000, 345_000, 0.945) + ["EVAL step=325000 loss=0.93619 a"]
    lines += start("float32", 340_000) + steps(340_000, 356_400, 0.94, seconds=0.30) + ["EVAL step=350000 loss=0.93193 a"]
    lines += start("float32", 350_000) + steps(350_000, 360_000, 0.93, seconds=0.30)
    return "\n".join(lines) + "\n"


def replay_log(loss_shift=0.0, nonfinite=(), stop=350_000, evals=None):
    evals = evals or {275_000: 0.948, 300_000: 0.944, 325_000: 0.937, 350_000: 0.932}
    lines = []
    for lo, hi in ((250_000, 270_000), (270_000, 320_000), (320_000, 350_000)):
        if lo >= stop:
            break
        hi = min(hi, stop)
        lines += start("bfloat16", lo, attention="float32", z=1e-4)

        def skipped(s):
            return 31 + sum(n for t, n in nonfinite if t <= s)

        def loss(s):
            return {True: 0.96, False: 0.95}[s <= 280_000] - (0.01 if 340_000 < s else 0) + loss_shift

        lines += steps(lo, hi, loss, skipped=skipped, z=4.2)
        for t, n in nonfinite:
            if lo < t <= hi:
                lines.append(f"NONFINITE step={t} skipped={skipped(t)} in_a_row={min(n, 3)}: {n} step(s) with "
                             f"non-finite gradients left the parameters and optimizer state unchanged")
                lines.append(f"NONFINITE_GROUPS step={t} logits_nonfinite=False groups=encoder_5_ffn,embeddings")
        lines += [f"EVAL step={s} loss={v} accuracy=78" for s, v in evals.items() if lo < s <= hi]
    return "\n".join(lines) + "\n"


def test_starts_and_dtypes_of_v2_and_its_float32_reference():
    starts = rv.parse_starts(v2_log())
    assert [(s["from"], s["dtype"]) for s in starts] == [
        (0, "bfloat16"), (250_000, "float32"), (270_000, "bfloat16"), (280_000, "float32"), (320_000, "bfloat16"),
        (340_000, "float32"), (350_000, "float32")]
    f32, _ = rv.by_dtype(starts, "float32")
    bf16, _ = rv.by_dtype(starts, "bfloat16")
    assert f32[260_000] == 0.96 and f32[300_000] == 0.95 and f32[345_000] == 0.94 and f32[355_000] == 0.93
    # A bf16 stretch keeps its own steps past a later float32 start, up to the next bf16 start
    assert bf16[285_000] == 0.955 and bf16[255_000] == 0.96 and 290_000 not in bf16
    # An EVAL line need not start the line
    assert rv.merged(starts)["evals"][275_000] == 0.94703


def test_a_quiet_replay_passes():
    r = rv.verdict(replay_log(), v2_log())
    assert r["passed"], r
    assert r["skipped_at_start"] == 31 and r["S1_stability"]["total"] == 0
    assert all(row["steps"] == 200 for row in r["S2_loss_vs_v2_float32"])


def test_a_few_loose_skipped_steps_still_pass():
    r = rv.verdict(replay_log(nonfinite=[(260_010, 1), (300_000, 2), (345_550, 1)]), v2_log())
    assert r["S1_stability"]["total"] == 4 and r["S1_stability"]["max_in_window"] == 2
    assert r["passed"]


@pytest.mark.parametrize("nonfinite, why", [
    ([(281_530 + 10 * k, 1) for k in range(10)], "max_in_window"),
    ([(281_530, 3)], "max_in_a_row"),
    ([(260_000 + 5_000 * k, 1) for k in range(11)], "total"),
])
def test_a_burst_or_too_many_skipped_steps_fail(nonfinite, why):
    r = rv.verdict(replay_log(nonfinite=nonfinite), v2_log())
    assert not r["S1_stability"]["ok"] and not r["passed"], (why, r["S1_stability"])


def test_a_replay_that_stops_early_fails():
    r = rv.verdict(replay_log(stop=320_000), v2_log())
    assert not r["S1_stability"]["ok"] and r["S1_stability"]["reached"] == 320_000


def test_loss_above_the_margin_fails_and_below_does_not():
    worse = rv.verdict(replay_log(loss_shift=0.011), v2_log())
    assert not worse["passed"] and not any(row["ok"] for row in worse["S2_loss_vs_v2_float32"])
    better = rv.verdict(replay_log(loss_shift=-0.05), v2_log())
    assert all(row["ok"] for row in better["S2_loss_vs_v2_float32"])


def test_eval_above_the_margin_fails():
    r = rv.verdict(replay_log(evals={275_000: 0.948, 300_000: 0.955, 325_000: 0.937, 350_000: 0.932}), v2_log())
    assert [row["ok"] for row in r["S3_eval_vs_v2"]] == [True, False, True, True]
    assert not r["passed"]


def test_a_resume_after_a_preemption_replaces_the_repeated_steps():
    text = replay_log()
    # A second start of the last piece from 330,000, with other losses for the steps it repeats
    text += "\n".join(start("bfloat16", 330_000, attention="float32", z=1e-4) + steps(330_000, 350_000, 0.90, z=4.0)) + "\n"
    rec = rv.merged(rv.parse_starts(text))
    assert rec["steps"][340_000]["loss"] == 0.90 and rec["steps"][330_000]["loss"] == 0.95


def test_control_is_reported():
    control = "\n".join(start("bfloat16", 250_000) + steps(250_000, 256_000, 0.96, skipped=lambda s: 31 + max(0, (s - 252_600) // 2))
                        + [f"NONFINITE step={s} skipped={31 + (s - 252_600) // 2} in_a_row=4: 5 step(s) with non-finite gradients"
                           for s in range(252_610, 256_000, 10)]) + "\n"
    r = rv.verdict(replay_log(), v2_log(), control)
    assert r["control_bf16"]["first_nonfinite"] == 252_610 and r["control_bf16"]["max_in_window"] >= 400
    assert r["passed"]
    assert "geslaagd" in rv.markdown(r)


def test_s1_so_far_flags_a_burst_while_running():
    quiet = rv.s1_so_far(replay_log(stop=300_000))
    assert quiet["reached"] == 300_000 and not quiet["violated"] and quiet["last_z"] == 4.2
    burst = rv.s1_so_far(replay_log(nonfinite=[(281_530 + 10 * k, 1) for k in range(10)], stop=300_000))
    assert burst["violated"] and burst["max_in_window"] == 10
    assert rv.s1_so_far("  Compute dtype: bfloat16\n")["reached"] is None


def _smoke_log(shift=0.0, seconds=0.24, final=True):
    lines = start("bfloat16", 250_000, attention="float32", z=1e-4) + steps(250_000, 250_300, 0.96 + shift, z=4.2,
                                                                               seconds=seconds)
    if final:
        lines.append("Final checkpoint saved at optimizer_step 250300")
    return "\n".join(lines) + "\n"


def test_smoke_check():
    assert rv.smoke_check(_smoke_log(), v2_log())["ok"]
    assert not rv.smoke_check(_smoke_log(shift=0.015), v2_log())["ok"]
    assert not rv.smoke_check(_smoke_log(seconds=0.36), v2_log())["ok"]
    assert not rv.smoke_check(_smoke_log(final=False), v2_log())["ok"]
    plain = _smoke_log().replace("  Attention logits: float32\n", "")
    assert not rv.smoke_check(plain, v2_log())["ok"]
