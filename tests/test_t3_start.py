"""T3's trainer commands (experiments/t3/start_t3.sh), its schedules, its shape and the shell scripts of
experiments/t3/ (protocol.md sections 3, 6 and 8)."""
from __future__ import annotations

import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
T3 = ROOT / "experiments" / "t3"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

PEAK, WARMUP, END = 4.8e-4, 1250, 1e-6
MAIN = dict(total_steps=1_959_000, decay_steps=399_000)
BRANCH = dict(total_steps=1_306_000, decay_steps=266_000)
T2_TAK2 = dict(total_steps=1_306_000, decay_steps=266_000)
SHAPE = {"--hidden-size": "1152", "--num-layers": "18", "--num-heads": "18", "--intermediate-size": "2944",
         "--global-attn-every-n-layers": "3"}


def run_script(script, *args, **env):
    done = subprocess.run(["bash", str(script), *args], capture_output=True, text=True,
                          env={**os.environ, "PRINT_RUN_SCRIPT": "1", "COMMIT": "c0ffee", **env})
    assert done.returncode == 0, done.stderr + done.stdout
    return done.stdout


def flags(script_text):
    """The trainer's options of a run script: {--flag: value or True}, and the part before train_minimal.py."""
    line = next(line for line in script_text.splitlines() if "train_minimal.py" in line)
    words = shlex.split(line.split(">>")[0])
    at = next(i for i, w in enumerate(words) if w.endswith("train_minimal.py"))
    out, i = {}, at + 1
    while i < len(words):
        if i + 1 < len(words) and not words[i + 1].startswith("--"):
            out[words[i]] = words[i + 1]
            i += 2
        else:
            out[words[i]] = True
            i += 1
    return out


def test_the_main_leg_is_t2s_command_with_t3s_shape_fix_schedule_data_and_names():
    t3 = flags(run_script(T3 / "start_t3.sh", "main"))
    t2 = flags(run_script(ROOT / "experiments" / "final" / "start_final.sh", "culturax", "stable"))
    for flag, value in SHAPE.items():
        assert t3[flag] == value, flag
    assert (t3["--max-steps"], t3["--decay-steps"]) == ("1959000", "399000") and "--stop-at-step" not in t3
    assert all(t3[f] is True for f in ("--attention-float32", "--nonfinite-diagnostics", "--skip-nonfinite-updates"))
    assert t3["--z-loss"] == "1e-4" and "--compute-dtype" not in t3 and "--seed" not in t3
    assert t3["--tokenizer"] == "$HOME/mbert-t3-tokenizer" and t3["--data-files-list"] == "$HOME/t3_files.txt"
    assert t3["--wandb-run-name"] == "t3-culturax-317m"
    assert t3["--checkpoint-dir"] == "gs://t5-dutch-english-usc2/mbert/t3/t3-culturax-317m/ckpt"
    differ = {"--tokenizer", "--max-steps", "--decay-steps", "--stop-at-step", "--dataset-num-rows", "--data-files-list",
              "--checkpoint-dir", "--wandb-project", "--wandb-run-name", "--tracking-sync-dir", "--attention-float32",
              "--z-loss", "--nonfinite-diagnostics", *SHAPE}
    same = {f: v for f, v in t2.items() if f not in differ}
    assert same == {f: t3[f] for f in same}, "every other option is T2's"
    assert set(t3) - set(t2) == {"--attention-float32", "--z-loss", "--nonfinite-diagnostics"}
    assert set(t2) - set(t3) == {"--stop-at-step"}
    # The same options as the ablation's arms, at another shape and length
    abl = flags(run_script(ROOT / "experiments" / "v3" / "start_ablation.sh", "c"))
    assert set(abl) == set(t3)
    # (--dataset-num-rows of both reads a file that is not here)
    assert {f for f in abl if abl[f] != t3[f]} == {"--tokenizer", "--max-steps", "--decay-steps", "--data-files-list", "--checkpoint-dir", "--wandb-project",
                                                    "--wandb-run-name", "--tracking-sync-dir", "--hidden-size",
                                                    "--num-layers", "--num-heads", "--intermediate-size"}


def test_the_branch_is_t2s_tak_2_schedule_and_a_stop_or_float32_piece_is_passed_on():
    branch = flags(run_script(T3 / "start_t3.sh", "branch"))
    tak2 = flags(run_script(ROOT / "experiments" / "final" / "start_final.sh", "culturax", "epoch2"))
    assert (branch["--max-steps"], branch["--decay-steps"]) == (tak2["--max-steps"], tak2["--decay-steps"]) == ("1306000", "266000")
    assert branch["--wandb-run-name"] == "t3-culturax-317m-epoch2"
    main = flags(run_script(T3 / "start_t3.sh", "main", STOP_AT="1040000", COMPUTE_DTYPE="float32"))
    assert main["--stop-at-step"] == "1040000" and main["--compute-dtype"] == "float32"
    assert main["--z-loss"] == "1e-4" and main["--attention-float32"] is True, "a float32 piece keeps the objective"


def test_the_smoke_leg_has_its_own_name_checkpoints_and_eval():
    smoke = flags(run_script(T3 / "start_t3.sh", "smoke", LEG_TAG="t3a1122a"))
    assert smoke["--wandb-run-name"] == "t3-culturax-317m-smoke-t3a1122a"
    assert (smoke["--stop-at-step"], smoke["--checkpoint-every"], smoke["--eval-every"], smoke["--system-metrics-every"]) == ("300", "100", "200", "100")
    assert (smoke["--max-steps"], smoke["--decay-steps"]) == ("1959000", "399000")
    done = subprocess.run(["bash", str(T3 / "start_t3.sh"), "smoke"], capture_output=True, text=True,
                          env={**os.environ, "PRINT_RUN_SCRIPT": "1", "COMMIT": "c0ffee"})
    assert done.returncode != 0 and "set LEG_TAG for a smoke run" in done.stderr


@pytest.fixture(scope="module")
def tm():
    # Other test modules may have started JAX already, after which its module-level initialize() can't run
    with mock.patch("jax.distributed.initialize", side_effect=ValueError):
        import train_minimal

    return train_minimal


def lr(schedule, step):
    return float(schedule(step))


def test_t3_schedules_share_every_step_before_the_branch_and_the_main_leg_holds_the_peak_to_1560000(tm):
    main = tm.wsd_schedule(PEAK, WARMUP, end_value=END, **MAIN)
    branch = tm.wsd_schedule(PEAK, WARMUP, end_value=END, **BRANCH)
    tak2 = tm.wsd_schedule(PEAK, WARMUP, end_value=END, **T2_TAK2)
    steps = np.r_[np.arange(0, 3000), np.arange(3000, 1_040_001, 9973), 1_040_000]
    assert all(lr(main, s) == lr(branch, s) for s in steps)
    assert all(lr(branch, s) == lr(tak2, s) for s in np.r_[steps, np.arange(1_040_000, 1_306_001, 997)])
    assert lr(branch, 1_040_001) < lr(main, 1_040_001) == pytest.approx(PEAK, rel=1e-6)
    assert lr(main, 1_560_000) == pytest.approx(PEAK, rel=1e-6)
    assert lr(main, 1_560_001) < lr(main, 1_560_000)
    assert lr(main, 1_959_000) == pytest.approx(END, rel=1e-5) and lr(branch, 1_306_000) == pytest.approx(END, rel=1e-5)
    # A quarter into each cooldown the rate is halfway (1-sqrt), and both cooldowns are the same share of their run
    assert lr(main, 1_560_000 + 399_000 // 4) == pytest.approx(END + (PEAK - END) / 2, rel=1e-4)
    assert MAIN["decay_steps"] * BRANCH["total_steps"] == BRANCH["decay_steps"] * MAIN["total_steps"]


def test_the_trainer_accepts_t3s_options(tm, tmp_path):
    script = run_script(T3 / "start_t3.sh", "main", STOP_AT="1040000")
    line = next(line for line in script.splitlines() if "train_minimal.py" in line)
    argv = shlex.split(line.split(">>")[0].replace("$HOME", str(tmp_path)).replace("$ROWS", "1"))
    argv = argv[next(i for i, w in enumerate(argv) if w.endswith("train_minimal.py")):]
    argv[argv.index("--dataset-num-rows") + 1] = "117306891"
    with mock.patch.object(sys, "argv", argv):
        args = tm.parse_args()
    assert (args.hidden_size, args.num_layers, args.num_heads, args.intermediate_size) == (1152, 18, 18, 2944)
    assert (args.max_steps, args.decay_steps, args.stop_at_step, args.z_loss) == (1959000, 399000, 1040000, 1e-4)
    assert args.attention_float32 and args.skip_nonfinite_updates and args.url_blocklist == "hebban.nl"


def test_t3s_shape_has_317106304_parameters_with_t2s_window_and_rope():
    import jax
    import jax.numpy as jnp

    from mmbjax.configuration_mmbbert import MMBertConfig
    from mmbjax.heads.masked_lm import ModernBertForMaskedLM

    def count(layers, hidden, heads, intermediate):
        config = MMBertConfig(vocab_size=32_128, hidden_size=hidden, num_hidden_layers=layers, num_attention_heads=heads,
                              intermediate_size=intermediate, global_attn_every_n_layers=3, max_position_embeddings=8192,
                              pad_token_id=32_001, attention_float32=True)
        shapes = jax.eval_shape(lambda: ModernBertForMaskedLM(config).init(
            jax.random.PRNGKey(0), input_ids=jnp.ones((1, 8), jnp.int32), deterministic=True))["params"]
        return sum(int(np.prod(x.shape)) for x in jax.tree_util.tree_leaves(shapes)), config

    t3, config = count(18, 1152, 18, 2944)
    assert t3 == 317_106_304
    assert count(22, 1024, 16, 1536)[0] == 230_110_592, "T2's count, as the model card"
    assert (config.local_attention, config.global_rope_theta, config.local_rope_theta) == (128, 160000.0, 10000.0)
    assert config.hidden_size // config.num_attention_heads == 64


SCRIPTS = sorted(T3.glob("*.sh"))


@pytest.mark.parametrize("script", SCRIPTS, ids=[s.name for s in SCRIPTS])
def test_the_shell_scripts_parse_and_no_parameter_message_holds_an_apostrophe(script):
    assert subprocess.run(["bash", "-n", str(script)], capture_output=True).returncode == 0
    # An apostrophe in ${X:?...} opens a quote that swallows the lines after it (eval_v3.sh, fixed in 356f8f2); in a
    # default (${X:-...}) only a quoted part of a command substitution may hold them, in pairs
    for match in re.finditer(r"\$\{[A-Za-z_][A-Za-z0-9_]*:?([?\-=+])([^}]*)\}", script.read_text()):
        kind, text = match.groups()
        assert "'" not in text if kind == "?" else text.count("'") % 2 == 0, f"{script.name}: {match.group(0)}"


def test_eval_t3_sets_every_variable_before_it_checks_the_jobs(tmp_path):
    """Run with only what the chain passes: an unset variable (a swallowed line) would stop it before the job check."""
    env = {"PATH": os.environ["PATH"], "HOME": str(tmp_path), "COMMIT": "ae11478", "T3_COMMIT": "c0ffee",
           "SHAPE": " ".join(f"{k} {v}" for k, v in SHAPE.items()), "WORKER": "0"}
    done = subprocess.run(["bash", str(T3 / "eval_t3.sh"), "t3-1959000", "gs://b/ckpt", "nosuchtask:42"],
                          capture_output=True, text=True, env=env)
    assert done.returncode == 1 and done.stdout.startswith("w0: not a job"), done.stdout + done.stderr
    done = subprocess.run(["bash", str(T3 / "eval_t3.sh"), "t3-1959000", "gs://b/ckpt", "blimp"],
                          capture_output=True, text=True, env={**env, "SHAPE": "--hidden-size 1152"})
    assert done.returncode == 1 and "SHAPE is not the five shape flags" in done.stdout
    del env["T3_COMMIT"]
    done = subprocess.run(["bash", str(T3 / "eval_t3.sh"), "t3-1959000", "gs://b/ckpt", "blimp"],
                          capture_output=True, text=True, env=env)
    assert done.returncode != 0 and "T3_COMMIT: set T3_COMMIT to the commit of branch t3" in done.stderr
