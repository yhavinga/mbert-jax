"""The LoRA memory and step test on Gemma 4 26B-A4B and 31B (experiments/en-nl-translator/g4lora.py, run_g4lora.sh,
the Gemma 4 models in run_sft.sh, sft_lora.py's LoRA report): the variants' regexes against the modules of the
converted checkpoints, the report of what LoRA reached, Gemma 4's chat template, the classification of a run from its
logs, the advice's rule and the report. The test of the real chat template runs where a Gemma 4 tokenizer directory is
(G4_31B_TOKENIZER: tokenizer.json and chat_template.jinja of google/gemma-4-31B-it)."""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "experiments" / "en-nl-translator"
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(ROOT / "experiments" / "flan-nl"))

import g4lora as g  # noqa: E402
import sft_lora as sl  # noqa: E402

TOKENIZER_31B = Path(os.environ.get("G4_31B_TOKENIZER", "/dev/shm/translator/g4lora/tok-gemma4-31b"))
GLOBAL = {5, 11, 17, 23, 29}  # Gemma 4's pattern: five sliding-window layers, then a global one (gemma4.py)


def modules_26b():
    """The 26B's module paths as MaxText 0.2.5 builds them unscanned (the converted checkpoint's tree, 13 Nov):
    global layers have no value projection (share_kv_projections); the MoE block holds the experts' own weights and
    the router `gate` as a module of its own."""
    out = ["decoder", "decoder/decoder_norm"]
    for i in range(30):
        layer = f"decoder/layers_{i}"
        attn = ["query", "key", "out", "query_norm", "key_norm"] + ([] if i in GLOBAL else ["value"])
        out += [layer, f"{layer}/self_attention"] + [f"{layer}/self_attention/{m}" for m in attn]
        out += [f"{layer}/{n}" for n in ("pre_self_attention_norm", "post_self_attention_norm", "pre_ffw_norm",
                                         "post_ffw_norm")]
        moe = f"{layer}/mlp/moe_block"
        out += [f"{layer}/mlp", moe, f"{moe}/MoeBlock_0", f"{moe}/MoeBlock_0/gate", f"{moe}/shared_experts"]
        out += [f"{moe}/shared_experts/{m}" for m in ("wi_0", "wi_1", "wo")]
        out += [f"{layer}/mlp/{n}" for n in ("pre_feedforward_layernorm_2", "post_feedforward_layernorm_1",
                                             "post_feedforward_layernorm_2", "gate_norm")]
    return out


def modules_31b():
    out = ["decoder", "decoder/decoder_norm"]
    for i in range(60):
        layer = f"decoder/layers_{i}"
        attn = ["query", "key", "out", "query_norm", "key_norm"] + ([] if i % 6 == 5 else ["value"])
        out += [layer, f"{layer}/self_attention"] + [f"{layer}/self_attention/{m}" for m in attn]
        out += [f"{layer}/mlp"] + [f"{layer}/mlp/{m}" for m in ("wi_0", "wi_1", "wo")]
        out += [f"{layer}/{n}" for n in ("pre_self_attention_norm", "post_self_attention_norm", "pre_ffw_norm",
                                         "post_ffw_norm")]
    return out


def spec(variant, **env):
    out = subprocess.run(["bash", str(CODE / "run_g4lora.sh"), "spec"], capture_output=True, text=True, timeout=30,
                         env={"PATH": os.environ["PATH"], "HOME": os.environ.get("HOME", "/tmp"), "VARIANT": variant,
                              **env})
    assert out.returncode == 0, out.stdout + out.stderr
    return json.loads(out.stdout)


def matched(regex, paths):
    return [p for p in paths if re.fullmatch(regex, p)]


def kinds(paths):
    return sorted({sl.module_type(p) for p in paths})


def test_the_variants_reach_exactly_their_modules_of_the_converted_checkpoints():
    a26, a31 = modules_26b(), modules_31b()
    attn26 = [p for p in a26 if re.search(r"self_attention/(query|key|value|out)$", p)]
    assert len(attn26) == 25 * 4 + 5 * 3
    v1 = spec("v1")
    assert v1["model"] == "gemma4-26b" and (v1["rank"], v1["alpha"], v1["length"]) == (64, 32, 2048)
    assert matched(v1["regex"], a26) == attn26
    v2 = matched(spec("v2")["regex"], a26)
    assert len(v2) == 115 + 30 and kinds(set(v2) - set(attn26)) == ["decoder/layers_*/mlp/moe_block/MoeBlock_0"]
    assert not [p for p in v2 if p.endswith("/gate")]   # the router stays frozen
    v2b = matched(spec("v2b")["regex"], a26)
    assert kinds(set(v2b) - set(attn26)) == [f"decoder/layers_*/mlp/moe_block/shared_experts/{m}"
                                             for m in ("wi_0", "wi_1", "wo")] and len(v2b) == 115 + 90
    v3 = spec("v3")
    assert v3["model"] == "gemma4-31b" and len(matched(v3["regex"], a31)) == 50 * 4 + 10 * 3 + 60 * 3
    # TranslateGemma's arms' regex, as run_sft.sh has it
    assert v3["regex"] in (CODE / "run_sft.sh").read_text()
    # on the 26B that regex reaches no MLP: the dense MLP sits under mlp/moe_block/shared_experts
    assert matched(v3["regex"], a26) == attn26
    v3b = spec("v3b")
    assert (v3b["rank"], v3b["alpha"], v3b["regex"]) == (32, 16, v3["regex"])


def test_a_variants_run_name_length_and_fallback():
    assert spec("v1")["run"] == "g4l_v1" and spec("v1")["extra"] == ""
    s = spec("v1", FALLBACK="1")
    assert s["run"] == "g4l_v1f" and s["extra"] == "attention=dot_product megablox=false" and s["fallback"] == 1
    s = spec("v3", FALLBACK="1", LEN="4096", LABEL="v4")
    assert (s["run"], s["extra"], s["length"], s["variant"], s["base"]) == \
        ("g4l_v3_4096f", "attention=dot_product", 4096, "v4", "v3")
    out = subprocess.run(["bash", str(CODE / "run_g4lora.sh"), "spec"], capture_output=True, text=True,
                         env={"PATH": os.environ["PATH"], "HOME": os.environ.get("HOME", "/tmp"), "VARIANT": "v9"})
    assert out.returncode != 0 and "VARIANT is" in out.stderr


def test_the_lora_report_shows_a_matched_module_without_lora_and_counts_per_type():
    regex = spec("v2")["regex"]
    sizes = {}
    for p in modules_26b():
        if re.search(r"self_attention/(query|key|value|out)$", p):
            sizes[f"{p}/kernel_lora_a"] = 10
            sizes[f"{p}/kernel_lora_b"] = 5
    report = sl.lora_report(regex, sizes, modules_26b())
    assert report["matched_without_lora"] == {"decoder/layers_*/mlp/moe_block/MoeBlock_0": 30}
    assert report["by_type"]["decoder/layers_*/self_attention/value"] == {"modules": 25, "params": 25 * 15}
    assert report["by_type"]["decoder/layers_*/self_attention/query"] == {"modules": 30, "params": 30 * 15}
    assert report["params"] == 115 * 15
    # experts reached (a MoE block's own weight, not a kernel): their type keeps the weight's name
    sizes["decoder/layers_0/mlp/moe_block/MoeBlock_0/wi_0_lora_a"] = 7
    report = sl.lora_report(regex, sizes, ["outer/" + p for p in modules_26b()])   # a wrapper's prefix drops
    assert report["by_type"]["decoder/layers_*/mlp/moe_block/MoeBlock_0/wi_0"] == {"modules": 1, "params": 7}
    assert report["matched_without_lora"] == {"decoder/layers_*/mlp/moe_block/MoeBlock_0": 29}
    with pytest.raises(ValueError):
        sl.lora_module_type("decoder/layers_0/self_attention/query/kernel")


def render(template, messages, add_generation_prompt):
    import jinja2

    def raise_exception(message):
        raise ValueError(message)

    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)
    env.globals["raise_exception"] = raise_exception
    return env.from_string(template).render(messages=messages, bos_token="<bos>",
                                            add_generation_prompt=add_generation_prompt)


def test_gemma4s_template_makes_the_prompt_of_the_pilot_and_ends_the_dutch_with_turn_end():
    en, nl = "  The house is red.\n", "Het huis is rood. "
    user = g.user_turn(en)
    assert user.startswith("Translate the following English text into natural, idiomatic Dutch") and \
        user.endswith("\n\nThe house is red.")
    prompt = render(g.TEMPLATE, [{"role": "user", "content": user}], True)
    full = render(g.TEMPLATE, [{"role": "user", "content": user}, {"role": "assistant", "content": nl}], False)
    assert prompt == g.prompt_text(en) == \
        "<bos><|turn>user\n" + user + "<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
    assert full == prompt + g.completion_text(nl) and g.completion_text(nl) == "Het huis is rood.<turn|>\n"
    # translate_models.yml's turn of both models, which the pilot's translations used
    for model in g.MODELS:
        prefix, suffix = g.model_turn(model)
        assert prefix + user + suffix == prompt


@pytest.mark.skipif(not (TOKENIZER_31B / "chat_template.jinja").exists(), reason="no Gemma 4 31B tokenizer here")
def test_the_models_own_chat_template_closes_the_same_empty_thought_channel():
    import jinja2

    own = (TOKENIZER_31B / "chat_template.jinja").read_text()
    user = g.user_turn("The house is red.")
    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])
    env.globals["raise_exception"] = lambda m: (_ for _ in ()).throw(ValueError(m))
    text = env.from_string(own).render(messages=[{"role": "user", "content": user}], bos_token="<bos>",
                                       add_generation_prompt=True)
    assert text == g.prompt_text("The house is red.")


def test_select_takes_the_first_examples_without_context_or_tag():
    metas = [{"context": None, "tag": "none"}, {"context": [{"source": "x"}], "tag": "none"},
             {"context": [], "tag": "token"}, {"context": [], "tag": "none"}, {"context": None}]
    assert g.select(metas, 3) == [0, 3, 4]
    with pytest.raises(SystemExit):
        g.select(metas, 4)
    assert round(g.batches_estimate(3_095_665 * 4, 2048), 0) == 420   # RL: 413 batches measured


def log(exit_code=0, peak=27.5, step=9.1, lora=True, extra=""):
    lines = ["preflight", "multi-host: jax.distributed.initialize in sft_lora.py",
             'EPOCH {"batches": 50, "batches_per_process": [50, 51, 49, 50], "steps": 8}']
    if lora:
        lines.append("LORA_MODULES " + json.dumps({"regex": "r", "params": 115,
                                                   "by_type": {"decoder/layers_*/self_attention/query": {
                                                       "modules": 30, "params": 115}},
                                                   "matched_without_lora": {
                                                       "decoder/layers_*/mlp/moe_block/MoeBlock_0": 30}}))
    lines += extra.splitlines()
    if exit_code == 0:
        lines.append("SUMMARY " + json.dumps({"peak_gib_max": peak, "throughput": {"step_s_median": step}}))
    if exit_code is not None:
        lines.append(f"EXIT {exit_code}")
    return "\n".join(lines) + "\n"


SPEC = {"variant": "v2", "base": "v2", "run": "g4l_v2", "model": "gemma4-26b", "rank": 64, "alpha": 32,
        "length": 2048, "extra": "", "regex": "r", "fallback": 0}
SUMMARY = {"throughput": {"step_s_mean": 9.3, "tokens_per_s": 3500.0, "tokens_per_s_per_chip": 218.8,
                          "loss_tokens_per_s": 1500.0, "step_s_median": 9.1},
           "loss": [[k, 2.0 - 0.1 * k] for k in range(1, 9)], "times": {"setup_s": 120.0, "compile_s_est": 400.0},
           "tokens": {"per_step": [30000] * 8, "loss_per_step": [13000] * 8}, "moe": {"num_experts": 128},
           "config": {"ici_fsdp_parallelism": -1, "per_device_batch_size": 1}, "devices": 16, "processes": 4}


def test_a_run_is_ok_oom_error_or_incomplete_from_its_four_logs():
    r = g.result(SPEC, [log(peak=p) for p in (27.1, 27.5, 26.9, 27.0)], SUMMARY)
    assert r["status"] == "ok" and r["peak_gib"] == 27.5 and r["fits"] and r["loss_finite"]
    assert r["step_s_median"] == 9.1 and r["tokens_per_s"] == 3500.0 and r["batches_per_process"] == [50, 51, 49, 50]
    assert r["matched_without_lora"] == {"decoder/layers_*/mlp/moe_block/MoeBlock_0": 30}
    assert not g.experts_reached(r)
    r = g.result(SPEC, [log(peak=29.6)] * 4, SUMMARY)
    assert r["status"] == "ok" and not r["fits"]
    bad = dict(SUMMARY, loss=SUMMARY["loss"][:7] + [[8, float("nan")]])
    assert not g.result(SPEC, [log()] * 4, bad)["loss_finite"]
    oom = ("jaxlib.xla_extension.XlaRuntimeError: RESOURCE_EXHAUSTED: XLA:TPU compile permanent error. Ran out of "
           "memory in memory space hbm. Used 33.12G of 30.75G hbm. Exceeded hbm capacity by 2.37G.")
    r = g.result(SPEC, [log(1, extra=oom)] + [log(1)] * 3, None)
    assert r["status"] == "oom" and r["hbm_used_of"] == [33.12, 30.75] and "RESOURCE_EXHAUSTED" in r["oom_line"]
    assert r["lora_params"] == 115   # the LoRA report comes before the first step
    err = ("Traceback (most recent call last):\n  File \"/dev/shm/x/venv/lib/python3.12/site-packages/qwix/_src/providers/"
           "lora.py\", line 518, in _get_or_create_lora_params\n    init()\n"
           "flax.errors.TraceContextError: Cannot mutate RngCount from a different trace level")
    r = g.result(SPEC, [log(1, extra=err)] * 4, None)
    assert r["status"] == "error" and r["error_line"].startswith("flax.errors.TraceContextError: Cannot mutate")
    assert r["error_frames"] == ["qwix/_src/providers/lora.py:518 _get_or_create_lora_params"]
    text, _ = g.report([dict(r, status="error")], {"models": {}}, {})
    assert "qwix/_src/providers/lora.py:518 _get_or_create_lora_params" in text
    assert g.result(SPEC, [log(None)] + [log(1)] * 3, None)["status"] == "incomplete"


def ok(variant, fits=True, experts=False, fallback=0, length=2048, model=None):
    r = {"variant": variant, "base": variant, "run": f"g4l_{variant}", "rank": 64, "alpha": 32, "length": length,
         "fallback": fallback, "model": model or ("gemma4-31b" if variant.startswith("v3") else "gemma4-26b"),
         "status": "ok", "fits": fits, "loss_finite": True, "peak_gib": 27.0 if fits else 30.2, "step_s_median": 8.0,
         "times": {"setup_s": 100.0, "compile_s_est": 400.0}, "lora_params": 1,
         "lora_by_module": {"decoder/layers_*/self_attention/query": {"modules": 1, "params": 1}}}
    if experts:
        r["lora_by_module"]["decoder/layers_*/mlp/moe_block/MoeBlock_0/wi_0"] = {"modules": 1, "params": 1}
    return r


def test_the_advice_takes_the_31b_first_and_the_experts_only_when_lora_reached_them():
    assert g.best([ok("v1"), ok("v2"), ok("v2b"), ok("v3")]) == "v3"
    assert g.best([ok("v1"), ok("v2"), ok("v2b"), ok("v3", fits=False), ok("v3b")]) == "v3b"
    assert g.best([ok("v1"), ok("v2"), ok("v2b"), ok("v3", fits=False), ok("v3b", fits=False)]) == "v2b"
    assert g.best([ok("v1"), ok("v2", experts=True), ok("v2b"), ok("v3", fits=False)]) == "v2"
    assert g.best([ok("v1"), dict(ok("v2b"), status="oom")]) == "v1"
    assert g.best([dict(ok("v1"), status="error"), ok("v1", fallback=1)]) == "v1"   # the fallback's attempt counts
    assert g.best([dict(ok("v1"), status="error")]) is None
    assert g.best([ok("v1", length=4096, fits=True)]) is None   # 4096 never picks the variant


def test_markers_are_read_in_the_variants_order_and_the_report_has_table_modules_and_advice(tmp_path):
    results = [dict(ok("v3"), variant="v4", length=4096, step_s_median=17.0, peak_gib=31.0, fits=False),
               ok("v3"), dict(ok("v1"), status="error", error_line="ValueError: x|y"), ok("v1", fallback=1),
               dict(ok("v2b"), status="oom", hbm_used_of=[31.2, 30.75], peak_gib=None, fits=None)]
    paths = []
    for k, r in enumerate(results):
        p = tmp_path / f"run_{k}.done"
        p.write_text("2026-11-13T21:00:00Z tag t commit c RESULT " + json.dumps(r) + "\n")
        paths.append(p)
    read = g.read_results(paths)
    assert [(r["variant"], r["fallback"]) for r in read] == [("v1", 0), ("v1", 1), ("v2b", 0), ("v3", 0), ("v4", 0)]
    data = {"models": {"gemma4-31b": {"ratio": {"RL": {"ratio": 1.1}, "R2": {"ratio": 1.2}}}}}
    tg = {"RL": {"steps": 403, "train_s": 3477.2}, "R2": {"steps": 3653, "train_s": 27604.5}}
    text, advice = g.report(read, data, tg)
    assert advice["variant"] == "v3" and advice["fits_4096"] is False
    assert advice["epoch_RL"] == {"steps": 443, "hours": round((443.3 * 8.0 + 500) / 3600, 2)}
    assert advice["epoch_R2"]["steps"] == round(3653 * 1.2)
    assert "geheugen op (31,20 van 30,75 GiB)" in text and "fout: ValueError: x/y" in text
    assert "| v3 (31B, TranslateGemma-modules (attention + MLP)) | `g4l_v3` |" in text
    assert "Eén epoch RL" in text and "past niet of traint niet" in text


def test_report_command_writes_markdown_and_json(tmp_path):
    p = tmp_path / "run_g4l_v1.done"
    p.write_text("x RESULT " + json.dumps(ok("v1")) + "\n")
    (tmp_path / "data.json").write_text(json.dumps({"models": {}, "template": "t"}))
    for name, steps in (("rl", 403), ("r2", 3653)):
        (tmp_path / f"{name}.json").write_text(json.dumps({"steps": steps, "times": {"train_s": 1.0},
                                                           "throughput": {"step_s_median": 7.1}}))
    head = tmp_path / "head.md"
    head.write_text("# kop\n\n")
    out = tmp_path / "rep" / "gemma4-moe-lora-test.md"
    g.main(["report", "--results", str(p), "--data", str(tmp_path / "data.json"), "--tg", f"RL={tmp_path}/rl.json",
            "--tg", f"R2={tmp_path}/r2.json", "--head", str(head), "--out", str(out)])
    assert out.read_text().startswith("# kop\n\n## Uitkomst")
    saved = json.loads(out.with_suffix(".json").read_text())
    assert saved["advice"]["variant"] == "v1" and "template" not in saved["data"]


def run_sft_job_text(tmp_path, **env):
    """The job text run_sft.sh writes to ~/run_<NAME>.sh, with /dev/shm moved under tmp_path (its venv, tokenizer
    and data made there) and gsutil, sudo and setsid faked (setsid starts nothing)."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    shm = tmp_path / "shm"
    text = (CODE / "run_sft.sh").read_text().replace("/dev/shm/", f"{shm}/")
    script = tmp_path / "run_sft.sh"
    script.write_text(text)
    fake = tmp_path / "bin"
    fake.mkdir(exist_ok=True)
    for name, body in {"gsutil": "exit 0\n", "sudo": "exit 0\n", "setsid": "exit 0\n", "hostname": "echo t1v-n-x-w-2\n"}.items():
        (fake / name).write_text("#!/bin/bash\n" + body)
        (fake / name).chmod(0o755)
    py = shm / "translator/smoke-lora/venv/bin/python"
    py.parent.mkdir(parents=True, exist_ok=True)
    py.write_text("#!/bin/bash\n")
    py.chmod(0o755)
    model = env["MODEL"]
    tok = shm / ("translator/g4lora/tok-" + model if model.startswith("gemma4") else "flan_nl/tokenizer-" + model)
    tok.mkdir(parents=True, exist_ok=True)
    (tok / "tokenizer.json").write_text("{}")
    data = tmp_path / "train.parquet"
    data.write_text("x")
    template = tmp_path / "t.jinja"
    template.write_text("x")
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    out = subprocess.run(["bash", str(script), "max_target_length=2048"], capture_output=True, text=True, timeout=30,
                         env={"HOME": str(home), "PATH": f"{fake}:{os.environ['PATH']}", "NAME": "n1",
                              "DATA": str(data), "TEMPLATE": str(template), "MULTIHOST": "1", **env})
    assert out.returncode == 0, out.stdout + out.stderr
    return (home / "run_n1.sh").read_text().replace(str(shm), "/dev/shm")


def test_run_sft_keeps_translategemma_and_takes_gemma4_with_its_own_regex(tmp_path):
    tg = run_sft_job_text(tmp_path / "a", MODEL="27b")
    assert "model_name=gemma3-27b load_parameters_path=gs://t5-dutch-english-usc2/mbert/flan-nl/translategemma-27b-" \
           "maxtext/0/items" in tg and "tokenizer_path=/dev/shm/flan_nl/tokenizer-27b " in tg
    assert "lora.lora_module_path=decoder/layers_\\[0-9\\]+/.\\*\\(self_attention/\\(query\\|key\\|value\\|out\\)" \
           "\\|mlp/\\(wi_0\\|wi_1\\|wo\\)\\)" in tg
    regex = spec("v2")["regex"]
    g4 = run_sft_job_text(tmp_path / "b", MODEL="gemma4-26b", LORA_MODULES=regex)
    assert "model_name=gemma4-26b load_parameters_path=gs://t5-dutch-english-usc2/mbert/translator/gemma4-ckpt/" \
           "gemma-4-26b-a4b-it/0/items" in g4 and "tokenizer_path=/dev/shm/translator/g4lora/tok-gemma4-26b " in g4
    module_path = re.search(r"lora\.lora_module_path=(\S+)", g4).group(1)
    assert subprocess.run(["bash", "-c", f"printf %s {module_path}"], capture_output=True, text=True).stdout == regex
    g31 = run_sft_job_text(tmp_path / "c", MODEL="gemma4-31b")
    assert "gemma4-ckpt/gemma-4-31b-it/0/items" in g31 and "model_name=gemma4-31b " in g31
