"""G4-8's training (protocol-g4.md §19, a draft until Yeb's sample and O6): g4arm8.py's sftset (the targets as SFT rows
in two passes, checked against the plan and the evaluation; the smoke set; RL-val's segments), smoke8, trained,
loraline and report8 (the choice on the search part, both variants of the rule, the measures per candidate) on small
synthetic inputs, and the stages of run_g4arm8.sh and of run_g4arm.sh with ARM=G4-8 rendered with a fake bucket (no
model, no TPU)."""

import json
import os
import random
import subprocess
import sys
import time
from argparse import Namespace
from pathlib import Path

import pyarrow.parquet as pq
import pytest
import test_en_nl_g4arm7 as t7
import test_en_nl_g4arm8 as t8d

g8, g4rl, g4arm = t8d.g8, t8d.g4rl, t7.g4arm
CODE = t8d.CODE


# ---------------------------------------------------------------------------------------------------- the schedule

def test_the_schedule_takes_an_eval_every_sixth_and_the_candidates_after_2e_4e_and_the_end():
    assert g8.schedule_t(190) == (32, [64, 128])
    assert g8.schedule_t(13) == (3, [6, 12])
    with pytest.raises(SystemExit, match="too few"):
        g8.schedule_t(7)
    assert g8.candidates_t({"save_steps": [64, 128], "steps": 190}) == ["G4-8-s64", "G4-8-s128", "G4-8-s190"]
    assert g8.cand_step_t("G4-8-s128") == 128
    with pytest.raises(SystemExit, match="no candidate"):
        g8.cand_step_t("G4-7-i5")
    assert (g8.EPOCHS_T, g8.LR_T, g8.WARMUP_T, g8.SEQ_T) == (2, 5e-5, 10, 4096)
    text = (CODE / "protocol-g4.md").read_text(encoding="utf-8")
    section = text[text.index("## 19. G4-8 training"):]
    # §19 is final since 23 Nov (Yeb chose O6): no draft line, both variants still described, exactly one O6 line
    assert "CONCEPT, NIET GESTART" not in section and "R-oud" in section and "R-nieuw" in section
    o6 = [line for line in section.splitlines() if line.startswith("- `O6 [besluit Yeb")]
    assert o6 == ["- `O6 [besluit Yeb 23 nov 2026]: R-nieuw`"]


# ---------------------------------------------------------------------------------------------------- sftset

class Counter:
    """Gemma 4 tokens as g4arm.TokenCounter.examples counts them, here words: the English's for the prompt, the
    Dutch's + 1 for the completion"""

    def examples(self, rows):
        rows = list(rows)
        return [len(r["en"].split()) for r in rows], [len(r["nl"].split()) + 1 for r in rows]


def sft_world(tmp_path, n=600):
    """A plan of n sources (FLAN 10 / novels 45 / BT 45), a target for every one but each tenth, the evaluation sets,
    the blind passages, FLAN's evaluation rows, G4-7's pool and plan (RL-val), the three SFT evals"""
    rng = random.Random(0)
    words = ("huis", "rivier", "groen", "loopt", "langzaam", "stad", "oud", "licht", "water", "ziet", "hond", "ver")
    plan, targets = [], []
    for k in range(n):
        part = "flan" if k % 10 == 0 else "novels" if k % 2 else "bt"
        i = f"{part}:{k:04d}"
        en = " ".join(rng.choice(t8d.WORDS) for _ in range(6)) + f" w{k}"
        plan.append({"id": i, "part": part, "stratum": "enwiki" if part == "bt" else "", "doc": f"doc{k}", "en": en,
                     "tokens": 24, "origin": "g4-7", "chunk": "c1", "hypothesis0": f"anker {k}", "l0": 20, "qe0_g47": 2.0})
        if k % 10 != 9:
            tgt = " ".join(rng.choice(words) for _ in range(3 + k % 18))
            targets.append({"id": i, "part": part, "stratum": plan[-1]["stratum"], "origin": "g4-7", "chunk": "c1",
                            "reason": "sample", "j": 1, "p_best": 0.8, "qe0": 2.0, "candidates": 3, "keys": 3,
                            "not_scored": 0, "distinct": 6, "fails": 0, "p_sample": 0.8, "p_anchor": 0.15,
                            "p_gelijk": 0.05, "qe": 1.8, "ratio": 1.0, "tokens_out": 20, "tokens_in": 50, "chrf": 60.0,
                            "en": en, "target": tgt, "doc": f"doc{k}", "l0": 20})
    d = tmp_path / "in"
    g8.write_parquet(plan, d / "plan.parquet", g8.PLAN_SCHEMA)
    long_en = "one two three four five six seven eight nine ten eleven twelve thirteen fourteen"
    evalsets = [{"id": "wmt24pp-0", "set": "wmt24pp", "source": long_en, "reference": "een twee", "meta": "{}"},
                {"id": "flores-0", "set": "flores", "source": "a short flores source", "reference": "kort", "meta": "{}"}]
    g8.write_parquet(evalsets, d / "evalsets.parquet")
    (d / "blind100.jsonl").write_text(json.dumps({"blind_id": "b1", "source": "the blind source text"}) + "\n")
    g8.write_parquet([{"inputs": "translate this flan input", "targets": "vertaal"}], d / "eval-rows.parquet")
    pool = [{"id": "novels:val0", "en": "an rl val text", "doc": "valdoc", "role": "rlval"},
            {"id": plan[1]["id"], "en": plan[1]["en"], "doc": "doc1", "role": "pool"}]
    g8.write_parquet(pool, d / "g47-pool.parquet")
    g47 = [{"id": f"val{k}", "en": f"rl val source {k}", "tokens": 20 + k, "iteration": -1} for k in range(10)]
    g47 += [{"id": "it0", "en": "a pool source", "tokens": 30, "iteration": 0}]
    g8.write_parquet(g47, d / "g47-plan.parquet")
    import g4lora

    for name in g8.SFT_EVALS_T:
        g4rl.write_parquet([{"messages": [{"role": "user", "content": g4lora.user_turn(f"held out {name}")},
                                          {"role": "assistant", "content": "x"}]}], d / f"eval-{name}.parquet")
    g8.write_parquet(targets, d / "targets.parquet")
    return d, plan, targets


def sft_args(d, out):
    return Namespace(targets=str(d / "targets.parquet"), plan=str(d / "plan.parquet"), evalsets=str(d / "evalsets.parquet"),
                     blind=str(d / "blind100.jsonl"), flan_eval=str(d / "eval-rows.parquet"),
                     g47_pool=str(d / "g47-pool.parquet"), g47_plan=str(d / "g47-plan.parquet"),
                     sft_eval=[f"{n}={d / f'eval-{n}.parquet'}" for n in g8.SFT_EVALS_T], tokenizer="x", out=str(out))


@pytest.fixture
def small(monkeypatch, tmp_path):
    monkeypatch.setattr(g8, "SEQ_T", 32)
    monkeypatch.setattr(g8, "SMOKE_LONGEST_T", 10)
    monkeypatch.setattr(g8, "SMOKE_RANDOM_T", 20)
    blind = tmp_path / "in" / "blind100.jsonl"
    monkeypatch.setattr(g4arm, "blind_sha256", lambda: g4rl.file_sha256(blind))


def test_sftset_makes_two_shuffled_passes_of_every_target_and_the_schedule(tmp_path, small, capsys):
    d, plan, targets = sft_world(tmp_path)
    out = tmp_path / "out"
    g8.sftset(sft_args(d, out), counter=Counter())
    rows = pq.read_table(out / "train.parquet").to_pylist()
    meta = pq.read_table(out / "train-meta.parquet").to_pylist()
    m = json.loads((out / "manifest.json").read_text())
    assert len(rows) == len(meta) == 2 * len(targets) == m["examples"] and m["distinct"] == len(targets)
    passes = [[x["id"] for x in meta if x["epoch"] == e] for e in (1, 2)]
    assert sorted(passes[0]) == sorted(passes[1]) == sorted(t["id"] for t in targets) and passes[0] != passes[1]
    assert [x["epoch"] for x in meta] == [1] * len(targets) + [2] * len(targets)
    by = {t["id"]: t for t in targets}
    import g4lora

    for x, y in zip(rows, meta):
        t = by[y["id"]]
        assert x["messages"] == [{"role": "user", "content": g4lora.user_turn(t["en"])},
                                 {"role": "assistant", "content": t["target"]}]
        assert y["kind"] == t["part"] and y["en"] == t["en"] and y["prompt"] == "natural-v1"
        assert (y["tokens_prompt"], y["tokens_target"]) == (len(t["en"].split()), len(t["target"].split()) + 1)
    assert (out / "template.jinja").read_text(encoding="utf-8") == g4lora.TEMPLATE
    lengths = [y["tokens_prompt"] + y["tokens_target"] for y in meta]
    batches = g4arm.pack_batches(lengths, seq=32)
    assert m["batches_per_process"] == batches and m["steps"] == g4arm.steps_of(batches)
    e, save = g8.schedule_t(m["steps"])
    assert (m["eval_every"], m["save_steps"]) == (e, save)
    assert m["candidates"] == [f"G4-8-s{save[0]}", f"G4-8-s{save[1]}", f"G4-8-s{m['steps']}"]
    assert m["parts"]["flan"]["examples"] == sum(t["part"] == "flan" for t in targets)
    assert m["leak_check"] == {"ngram": 0, "eval_text": 0, "rlval": 0, "rlval_doc": 0, "sft_eval": 0}
    assert (m["lr"], m["warmup_steps"], m["epochs"]) == (5e-5, 10, 2)
    # the smoke set: pass 1's 10 longest first, then as few others as give 8 steps of the multi-host job
    sm = json.loads((out / "smoke" / "manifest.json").read_text())
    smeta = pq.read_table(out / "smoke" / "train-meta.parquet").to_pylist()
    slen = [y["tokens_prompt"] + y["tokens_target"] for y in smeta]
    assert sm["steps"] >= 8 and g4arm.steps_of(g4arm.pack_batches(slen[:-1], seq=32)) < 8
    first = sorted((y for y in meta if y["epoch"] == 1), key=lambda y: (-(y["tokens_prompt"] + y["tokens_target"]), y["id"]))
    assert [y["id"] for y in smeta[:10]] == [y["id"] for y in first[:10]] and set(y["epoch"] for y in smeta) == {1}
    # RL-val per worker, by tokens
    seg = [pq.read_table(out / "rlval" / f"segments-w{k}.parquet").to_pylist() for k in range(4)]
    assert sorted(r["id"] for s in seg for r in s) == [f"val{k}" for k in range(10)]
    line = json.loads(next(x for x in capsys.readouterr().out.splitlines() if x.startswith("SFTSET8 "))[8:])
    assert line["steps"] == m["steps"] and line["rlval"] == 10
    report = (out / "report.md").read_text(encoding="utf-8")
    assert "Lekcontrole" in report and f"**{m['steps']} stappen**" in report
    # the same inputs give the same set
    g8.sftset(sft_args(d, tmp_path / "again"), counter=Counter())
    assert (tmp_path / "again" / "train.parquet").read_bytes() == (out / "train.parquet").read_bytes()


@pytest.mark.parametrize("kind", ["ngram", "eval_text", "blind_text", "rlval", "rlval_doc", "sft_eval"])
def test_sftset_stops_on_a_source_that_meets_the_evaluation(tmp_path, small, kind):
    d, plan, targets = sft_world(tmp_path)
    t = targets[5]
    if kind == "ngram":
        t["en"] = "zero one two three four five six seven eight nine ten eleven twelve thirteen"
    elif kind == "eval_text":
        t["en"] = "a short flores source"
    elif kind == "blind_text":
        t["en"] = "the blind source text"
    elif kind == "rlval":
        t["en"] = "an rl val text"
    elif kind == "rlval_doc":
        t["doc"] = "valdoc"
    else:
        t["en"] = "held out own"
    if t["en"] != plan[int(t["id"].split(":")[1])]["en"]:
        plan[int(t["id"].split(":")[1])]["en"] = t["en"]
        g8.write_parquet(plan, d / "plan.parquet", g8.PLAN_SCHEMA)
    g8.write_parquet(targets, d / "targets.parquet")
    want = {"blind_text": "eval_text"}.get(kind, kind)
    with pytest.raises(SystemExit, match=f"'{want}': 1"):
        g8.sftset(sft_args(d, tmp_path / "out"), counter=Counter())
    assert not (tmp_path / "out" / "manifest.json").exists()


@pytest.mark.parametrize("change, msg", [
    (lambda ts, p: ts[3].update(reason="p"), "no janchor target"),
    (lambda ts, p: ts[3].update(p_sample=0.5), "no janchor target"),
    (lambda ts, p: ts[3].update(target="anker   " + ts[3]["id"].split(":")[1].lstrip("0")), "anchor's text"),
    (lambda ts, p: ts[3].update(target="  "), "anchor's text"),
    (lambda ts, p: ts[3].update(id="bt:9999"), "no source of the plan"),
    (lambda ts, p: ts.append(dict(ts[3])), "twice"),
    (lambda ts, p: ts[3].update(en="other english"), "English is not the plan's"),
    (lambda ts, p: ts[3].update(part="flan"), "has part"),
])
def test_sftset_takes_janchor_targets_of_the_plan_only(tmp_path, small, change, msg):
    d, plan, targets = sft_world(tmp_path)
    change(targets, plan)
    g8.write_parquet(targets, d / "targets.parquet")
    with pytest.raises(SystemExit, match=msg):
        g8.sftset(sft_args(d, tmp_path / "out"), counter=Counter())


# ---------------------------------------------------------------------------------------------------- smoke8, trained

def summary(steps, evals, peak=28.0, batches=(30, 30, 31, 30), lr=None, own0=0.6):
    lr = lr if lr is not None else {"0": 5e-6, str(min(7, steps - 1)): 5e-5, str(steps - 1): 5e-5}
    return {"steps_done": steps, "loss": [[k + 1, 1.0] for k in range(steps)], "peak_gib_max": peak,
            "config": {"max_target_length": 4096}, "epoch": {"batches_per_process": list(batches)}, "lr_at": lr,
            "eval": [{"step": s, "set": n, "loss": own0 if (n == "own" and s == 0) else 0.9, "tokens": 100,
                      "seconds": 4.0} for s in evals for n in g8.SFT_EVALS_T],
            "throughput": {"step_s_median": 9.0}, "times": {"python_to_setup_s": 100.0, "setup_s": 60.0,
                                                           "compile_s_est": 300.0}}


def adapter_meta(step, rank=32):
    return {"rank": rank, "alpha": 16.0, "scale": 16.0 / rank, "steps": step}


def smoke_files(tmp_path, s, a8=None, a4=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    files = {"summary": s, "adapter": a8 or adapter_meta(8), "adapter_step": a4 or adapter_meta(4),
             "smoke_manifest": {"batches_per_process": [30, 30, 31, 30], "tokens": 400000},
             "manifest": {"steps": 190, "tokens": 12_000_000, "eval_every": 32, "save_steps": [64, 128],
                          "batches_per_process": [210, 210, 211, 210]}}
    for k, v in files.items():
        (tmp_path / f"{k}.json").write_text(json.dumps(v))
    return Namespace(run="g4arm_G4-8_s8", out=str(tmp_path / "smoke8.json"),
                     **{k: str(tmp_path / f"{k}.json") for k in files})


def test_smoke8_passes_a_good_smoke_run_and_estimates_the_full_run(tmp_path, capsys):
    g8.smoke8(smoke_files(tmp_path, summary(8, [0, 4, 8])))
    r = json.loads((tmp_path / "smoke8.json").read_text())
    assert r["passed"] and not r["failed"] and r["estimate"]["steps"] == 190 and r["estimate"]["hours"] > 0
    assert set(r["checks"]) >= {"steps", "loss_finite", "peak_gib", "seq", "batches_as_sftset", "lr_peak", "eval_books",
                                "eval_btval", "eval_own", "adapter", "adapter_step4", "own_step0"}
    assert '"passed": true' in capsys.readouterr().out


def test_smoke8_takes_the_peak_lr_as_logged_in_float32(tmp_path):
    # run g4arm_G4-8_s8 (23 Nov) logged its peak as 4.999999873689376e-05, 5e-5 in float32
    g8.smoke8(smoke_files(tmp_path, summary(8, [0, 4, 8], lr={"0": 0.0, "4": 2.857e-05, "7": 4.999999873689376e-05})))
    assert json.loads((tmp_path / "smoke8.json").read_text())["checks"]["lr_peak"]["holds"]


@pytest.mark.parametrize("kw, failed", [
    ({"s": summary(8, [0, 4, 8], lr={"0": 0.0, "7": 4.9e-05})}, ["lr_peak"]),
    ({"s": summary(8, [0, 4, 8], own0=1.2)}, ["own_step0"]),
    ({"s": summary(8, [0, 4, 8], peak=29.6)}, ["peak_gib"]),
    ({"s": summary(8, [0, 4, 8], batches=(30, 30, 30, 30))}, ["batches_as_sftset"]),
    ({"s": summary(8, [0, 8])}, ["eval_books", "eval_btval", "eval_own"]),
    ({"s": summary(8, [0, 4, 8], lr={"0": 5e-6, "7": 1e-4})}, ["lr_peak"]),
    ({"s": summary(8, [0, 4, 8]), "a4": adapter_meta(8)}, ["adapter_step4"]),
    ({"s": summary(8, [0, 4, 8]), "a8": adapter_meta(8, rank=16)}, ["adapter"]),
])
def test_smoke8_fails_on_each_check(tmp_path, kw, failed):
    with pytest.raises(SystemExit, match="SMOKE8 failed"):
        g8.smoke8(smoke_files(tmp_path, **kw))
    assert json.loads((tmp_path / "smoke8.json").read_text())["failed"] == failed


def trained_files(tmp_path, s, metas=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    m = {"steps": 190, "eval_every": 32, "save_steps": [64, 128], "batches_per_process": [30, 30, 31, 30]}
    (tmp_path / "summary.json").write_text(json.dumps(s))
    (tmp_path / "manifest.json").write_text(json.dumps(m))
    metas = metas or {f"G4-8-s{k}": adapter_meta(k) for k in (64, 128, 190)}
    adapters = []
    for name, meta in metas.items():
        (tmp_path / f"{name}.json").write_text(json.dumps(meta))
        adapters.append(f"{name}={tmp_path / f'{name}.json'}")
    return Namespace(run="g4arm_G4-8", summary=str(tmp_path / "summary.json"), manifest=str(tmp_path / "manifest.json"),
                     adapter=adapters, out=str(tmp_path / "trained.json"))


def test_trained_checks_the_run_against_sftset_and_each_candidates_adapter(tmp_path):
    evals = [0, 32, 64, 96, 128, 160, 190]
    lr = {"0": 5e-6, "10": 5e-5, "95": 5e-5, "189": 5e-5}
    g8.trained(trained_files(tmp_path / "a", summary(190, evals, lr=lr)))
    r = json.loads((tmp_path / "a" / "trained.json").read_text())
    assert r["passed"] and r["candidates"] == ["G4-8-s64", "G4-8-s128", "G4-8-s190"]
    with pytest.raises(SystemExit, match="TRAINED8 failed"):
        g8.trained(trained_files(tmp_path / "b", summary(190, [0, 32, 64, 96, 160, 190], lr=lr)))
    assert json.loads((tmp_path / "b" / "trained.json").read_text())["failed"] == ["eval_books", "eval_btval", "eval_own"]
    with pytest.raises(SystemExit, match="TRAINED8 failed"):
        g8.trained(trained_files(tmp_path / "c", summary(190, evals, lr={**lr, "189": 5e-6})))
    assert json.loads((tmp_path / "c" / "trained.json").read_text())["failed"] == ["lr_constant"]
    metas = {"G4-8-s64": adapter_meta(64), "G4-8-s128": adapter_meta(190), "G4-8-s190": adapter_meta(190)}
    with pytest.raises(SystemExit, match="TRAINED8 failed"):
        g8.trained(trained_files(tmp_path / "d", summary(190, evals, lr=lr), metas))
    with pytest.raises(SystemExit, match="the candidates are"):
        g8.trained(trained_files(tmp_path / "e", summary(190, evals, lr=lr), {"G4-8-s64": adapter_meta(64)}))


def test_loraline_takes_the_one_lora_line_of_a_candidates_logs(tmp_path, capsys):
    line = {"rank": 32, "alpha": 16.0, "modules": 410, "relative_update_norm": 0.0123, "elements_changed": 0.8}
    for k, d in enumerate(("/a", "/b")):
        (tmp_path / f"g4a_translate_G4-8-s64_t{k}_small-w{k}.log").write_text(
            f"== small\nLORA {json.dumps({'dir': d, **line})}\nDONE\n")
    (tmp_path / "g4a_translate_G4-8-s64_t0-w0.log").write_text("started\nEXIT 0\n")
    (tmp_path / "g4a_translate_G4-8-s640_t0_small-w0.log").write_text(f"LORA {json.dumps({**line, 'modules': 1})}\n")
    g8.main(["loraline", "--logs", str(tmp_path / "g4a_translate_G4-8-s64_*"), "--out", str(tmp_path / "x.json")])
    assert json.loads((tmp_path / "x.json").read_text()) == line
    (tmp_path / "g4a_translate_G4-8-s64_t2_tiny-w2.log").write_text(f"LORA {json.dumps({**line, 'modules': 409})}\n")
    with pytest.raises(SystemExit, match="2 different LORA lines"):
        g8.main(["loraline", "--logs", str(tmp_path / "g4a_translate_G4-8-s64_*"), "--out", str(tmp_path / "x.json")])


# ---------------------------------------------------------------------------------------------------- report8

NAMES = ["G4-8-s4", "G4-8-s8", "G4-8-s13"]


def qe_share(decision_share_s4=2, comet_s4=None):
    """test_en_nl_g4arm7's design with G4-8's names: s4 holds (a1) on the search part with the largest (b) there; s8
    holds (a1) with a smaller (b); s13 has the largest (b) but not (a1)"""
    qe = {"G4-8-s4": {"search": 1.9, "decision": 1.9}, "G4-8-s8": {"search": 1.8, "decision": 1.0},
          "G4-8-s13": {"search": 3.0, "decision": 3.0}, "G4-0": {"search": 1.5, "decision": 1.5},
          "G4-4": {"search": 1.6, "decision": 1.6}}
    share = {"G4-8-s4": {"search": 4, "decision": decision_share_s4}, "G4-8-s8": {"search": 3, "decision": 4},
             "G4-8-s13": {"search": 4, "decision": 4}, "G4-0": {"search": 1, "decision": 1},
             "G4-4": {"search": 1, "decision": 1}}
    return qe, share


def div_samples(root, rl, model):
    """8 samples per RL-val source: G4-0 and s4 six different texts, s8 three, s13 all eight the same (the flag); one
    sample of s4 not ended"""
    rows = []
    for k, r in enumerate(rl):
        for j in range(8):
            n = {"G4-0": 6, "G4-8-s4": 6, "G4-8-s8": 3, "G4-8-s13": 1}[model]
            flags = {"ended": False} if (model == "G4-8-s4" and k == 0 and j == 0) else {}
            rows.append(t8d.sample(r["id"], j, r["en"], f"tekst {min(j, n - 1)}", f"w{k % 2}", tokens_out=20, **flags))
    t8d.write_samples(root, rows)


def world8(tmp_path, monkeypatch, decision_share_s4=2, comet=None):
    qe, share = qe_share(decision_share_s4)
    a = t7.world7(tmp_path, monkeypatch, qe, share, comet=comet)
    manifest = {"steps": 13, "eval_every": 3, "save_steps": [4, 8], "tokens": 5000, "examples": 40,
                "parts": {"flan": {"examples": 4, "tokens": 400}, "novels": {"examples": 18, "tokens": 2300},
                          "bt": {"examples": 18, "tokens": 2300}}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    s = {"eval": [{"step": st, "set": n, "loss": 0.9 - 0.01 * st + (0.2 if n == "own" else 0), "tokens": 100}
                  for st in (0, 3, 4, 6, 8, 9, 12, 13) for n in g8.SFT_EVALS_T],
         "throughput": {"step_s_median": 9.1}, "peak_gib_max": 27.5}
    (tmp_path / "summary.json").write_text(json.dumps(s))
    rl = [{"id": f"val{k}", "en": f"rl val source {k}", "l0": 20, "iteration": -1, "tokens": 20} for k in range(4)]
    rl.append({"id": "it0", "en": "x", "l0": 20, "iteration": 0, "tokens": 20})
    g8.write_parquet(rl, tmp_path / "g47-plan.parquet")
    lora, div = [], []
    for k, name in enumerate(NAMES):
        (tmp_path / f"lora-{name}.json").write_text(json.dumps({"rank": 32, "alpha": 16.0, "modules": 410,
                                                               "relative_update_norm": 0.001 * (k + 1),
                                                               "elements_changed": 0.5}))
        lora.append(f"{name}={tmp_path / f'lora-{name}.json'}")
    for name in ["G4-0", *NAMES]:
        div_samples(tmp_path / "div" / name, rl[:4], name)
        div.append(f"{name}={tmp_path / 'div' / name}")
    (tmp_path / "smoke8.json").write_text(json.dumps({"passed": True, "checks": {"steps": {"holds": True}},
                                                      "failed": [], "estimate": {"hours": 0.8}}))
    return Namespace(evalsets=a.evalsets, split=a.split, refqe=a.refqe, base=a.base, predictions=a.predictions,
                     predictions_c2=a.predictions_c2, candidate=[f"{n}={tmp_path / n}" for n in NAMES],
                     reference=[f"G4-4={tmp_path / 'G4-4'}"], pll=None, summary=str(tmp_path / "summary.json"),
                     manifest=str(tmp_path / "manifest.json"), smoke=str(tmp_path / "smoke8.json"), lora=lora, div=div,
                     rlval=str(tmp_path / "g47-plan.parquet"), o6="open", tally=None, out=str(tmp_path / "report.md"))


def run8(args, capsys=None):
    g8.report8(args)
    r = json.loads(Path(args.out).with_suffix(".json").read_text())
    line = None
    if capsys:
        line = json.loads(next(x for x in capsys.readouterr().out.splitlines() if x.startswith("REPORT8_TRAIN "))[14:])
    return r, Path(args.out).read_text(encoding="utf-8"), line


def tally8(arm, doubt=0, a2=True, c=True):
    return {**t7.tally(arm, a2=a2, c=c), "doubt": doubt}


def test_report8_chooses_on_the_search_part_and_reports_both_variants(tmp_path, monkeypatch, capsys):
    args = world8(tmp_path, monkeypatch)
    r, md, line = run8(args, capsys)
    assert r["selected"] == "G4-8-s4" and r["selection"] == "a1 on the search part, max (b)"
    assert r["a1"]["holds"] and r["a3"]["holds"] and r["b"]["holds"] and r["c1_guard"]["holds"]
    assert r["b"]["delta"] == pytest.approx(0.25) and r["b"]["ci"]["n"] == 4
    assert r["b"]["ci"]["lo"] <= 0.25 <= r["b"]["ci"]["hi"]
    assert r["blind_due"] and r["blind"] is None
    assert r["verdicts"] == {"R-oud": "wacht op de blinde beoordeling ((a2), (c))",
                             "R-nieuw": "wacht op de blinde beoordeling ((a2), (c))"}
    assert r["verdict"].startswith("open") and line["selected"] == "G4-8-s4" and line["o6"] == "open"
    assert "O6 nog open" in md and "| R-oud |" in md and "| R-nieuw |" in md
    # G4-8.6 per candidate: the update norm, the SFT losses at its step, the samples on RL-val (s13: the flag)
    t = r["training"]
    assert [t[n]["step"] for n in NAMES] == [4, 8, 13] and t["G4-8-s8"]["update_norm"] == pytest.approx(0.002)
    assert t["G4-8-s8"]["sft"]["books"] == pytest.approx(0.82) and t["G4-8-s8"]["sft_delta"]["own"] == pytest.approx(-0.08)
    assert (t["G4-8-s4"]["div"]["distinct_median"], t["G4-8-s8"]["div"]["distinct_median"]) == (6, 3)
    assert t["G4-8-s13"]["div"]["flag"] and t["G4-8-s8"]["div"]["flag"] and not t["G4-8-s4"]["div"]["flag"]
    assert t["G4-8-s13"]["div"]["all_equal"] == 1.0 and t["G4-8-s4"]["div"]["fail_share"] == pytest.approx(1 / 32)
    assert r["div_base"]["distinct_median"] == 6 and not r["div_base"]["flag"]
    assert "8 gelijk" in md and "**ja**" in md and "Smoke-test" in md
    # the decisive variant once O6 is set
    args.o6 = "R-oud"
    r, md, _ = run8(args)
    assert r["verdict"] == "wacht op de blinde beoordeling ((a2), (c))" and "beslissend: R-oud" in md
    assert "| R-oud (beslissend) |" in md


def test_report8_with_a_tally_r_nieuw_waits_for_yebs_doubt_cases(tmp_path, monkeypatch):
    args = world8(tmp_path, monkeypatch)
    (tmp_path / "tally.json").write_text(json.dumps(tally8("G4-8-s4", doubt=3)))
    args.tally, args.o6 = str(tmp_path / "tally.json"), "R-nieuw"
    r, md, _ = run8(args)
    assert r["verdicts"] == {"R-oud": "helpt", "R-nieuw": "wacht op Yebs oordeel over de twijfelgevallen ((c) van R-nieuw)"}
    assert r["verdict"] == r["verdicts"]["R-nieuw"] and "twijfel 3" in md
    (tmp_path / "tally.json").write_text(json.dumps(tally8("G4-8-s4", doubt=0)))
    r, _, _ = run8(args)
    assert r["verdicts"] == {"R-oud": "helpt", "R-nieuw": "helpt"}
    (tmp_path / "tally.json").write_text(json.dumps(tally8("G4-8-s4", c=False)))
    r, _, _ = run8(args)
    assert r["verdicts"] == {"R-oud": "helpt niet", "R-nieuw": "helpt niet"}
    (tmp_path / "tally.json").write_text(json.dumps(tally8("G4-8-s8")))
    with pytest.raises(SystemExit, match="a tally of G4-8-s8"):
        run8(args)


def test_report8_b_decides_only_under_r_oud_and_the_c1_guard_under_r_nieuw(tmp_path, monkeypatch):
    # s4's C1 on the decision part equal to G4-0's: (b) fails, the guard (>= 0) holds
    args = world8(tmp_path / "eq", monkeypatch, decision_share_s4=1)
    r, _, _ = run8(args)
    assert not r["b"]["holds"] and r["c1_guard"]["holds"] and r["blind_due"]
    assert r["verdicts"] == {"R-oud": "helpt niet", "R-nieuw": "wacht op de blinde beoordeling ((a2), (c))"}
    # below G4-0: the guard fails too; the blind package is still due (a1 and a3), as a report
    args = world8(tmp_path / "lo", monkeypatch, decision_share_s4=0)
    r, _, _ = run8(args)
    assert not r["c1_guard"]["holds"] and r["blind_due"]
    assert r["verdicts"] == {"R-oud": "helpt niet", "R-nieuw": "helpt niet"}


def test_report8_without_a3_no_blind_review(tmp_path, monkeypatch):
    comet = {("G4-8-s4", "decision", "ntrex"): 0.844}
    args = world8(tmp_path, monkeypatch, comet=comet)
    r, md, _ = run8(args)
    assert not r["a3"]["holds"] and not r["blind_due"]
    assert r["verdicts"] == {"R-oud": "helpt niet", "R-nieuw": "helpt niet"}
    assert "zonder (a1) en (a3) geen blinde beoordeling" in md
    (tmp_path / "tally.json").write_text(json.dumps(tally8("G4-8-s4")))
    args.tally = str(tmp_path / "tally.json")
    with pytest.raises(SystemExit, match="do not both hold"):
        run8(args)


def test_report8_takes_the_manifests_candidates_and_every_measure(tmp_path, monkeypatch):
    args = world8(tmp_path, monkeypatch)
    args.candidate = args.candidate[:2]
    with pytest.raises(SystemExit, match="G4-8.4 takes"):
        run8(args)
    args = world8(tmp_path / "b", monkeypatch)
    args.div = args.div[1:]
    with pytest.raises(SystemExit, match="--div"):
        run8(args)
    args = world8(tmp_path / "c", monkeypatch)
    s = json.loads(Path(args.summary).read_text())
    s["eval"] = [e for e in s["eval"] if not (e["step"] == 8 and e["set"] == "btval")]
    Path(args.summary).write_text(json.dumps(s))
    with pytest.raises(SystemExit, match="at step 8"):
        run8(args)


# ---------------------------------------------------------------------------------------------------- run_g4arm8.sh

FAKE_GSUTIL = r"""#!/bin/bash
# gs://X -> $FAKEGS/X: stat, cat, cp (files); every call in $FAKELOG
echo "gsutil $*" >> "$FAKELOG"
args=(); for a in "$@"; do case $a in -q|-m) ;; *) args+=("$a") ;; esac; done
map() { case $1 in gs://*) echo "$FAKEGS/${1#gs://}" ;; *) echo "$1" ;; esac; }
case ${args[0]} in
  stat) [ -e "$(map "${args[1]}")" ] ;;
  cat) cat "$(map "${args[1]}")" ;;
  cp) n=${#args[@]}; dst=$(map "${args[$((n - 1))]}")
      for s in "${args[@]:1:$((n - 2))}"; do
        if [[ $dst == */ ]] || [ -d "$dst" ]; then mkdir -p "$dst"; else mkdir -p "$(dirname "$dst")"; fi
        cp "$(map "$s")" "$dst" || exit 1
      done ;;
  *) echo "fake gsutil: ${args[0]}" >&2; exit 2 ;;
esac
"""


def t8_world(tmp_path):
    """run_g4arm8.sh and run_sft.sh with /dev/shm under tmp_path, a fake bucket with what the training's stages read,
    fake gsutil, sudo, hostname and a setsid that only logs its call"""
    shm, gs, home, fake, code = (tmp_path / x for x in ("shm", "gs", "home", "bin", "code"))
    for d in (code, home, fake):
        d.mkdir(parents=True)
    for name in ("run_g4arm8.sh", "run_sft.sh"):
        (code / name).write_text((CODE / name).read_text().replace("/dev/shm/", f"{shm}/"))
    for f in CODE.glob("*.py"):
        (code / f.name).symlink_to(f)
    for name, body in {"gsutil": FAKE_GSUTIL, "sudo": "#!/bin/bash\nexit 0\n", "hostname": "#!/bin/bash\necho t1v-n-x-w-0\n",
                       "setsid": '#!/bin/bash\necho "SETSID $*" >> "$FAKELOG"\n'}.items():
        (fake / name).write_text(body)
        (fake / name).chmod(0o755)
    for py in ("translate-generic/venv-fork-f4d8615f4/bin/python", "translator/smoke-lora/venv/bin/python"):
        p = shm / py
        p.parent.mkdir(parents=True)
        p.write_text(f'#!/bin/bash\nexec {sys.executable} "$@"\n')
        p.chmod(0o755)
    for tok in ("translate-generic/tokenizer-gemma4", "translator/g4lora/tok-gemma4-31b"):
        (shm / tok).mkdir(parents=True)
        (shm / tok / "tokenizer.json").write_text("{}")
    tr = gs / "t5-dutch-english-usc2/mbert/translator"
    g = tr / "sft/g4arm"
    calls = tmp_path / "calls.log"

    def put(rel, text="x"):
        p = g / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)

    put("../../gemma4-ckpt/gemma-4-31b-it/done.txt")
    for n in g8.SFT_EVALS_T:
        put(f"data/heldout/eval-{n}.parquet")

    def run(stage, host="0", **env):
        (fake / "hostname").write_text(f"#!/bin/bash\necho t1v-n-x-w-{host}\n")
        e = {"HOME": str(home), "PATH": f"{fake}:{os.environ['PATH']}", "FAKEGS": str(gs), "FAKELOG": str(calls),
             "TAG": "t1", **env}
        calls.write_text("")
        out = subprocess.run(["bash", str(code / "run_g4arm8.sh"), stage], capture_output=True, text=True, timeout=120,
                             env=e, cwd=home)
        deadline = time.time() + 30
        while "started " in out.stdout and "SETSID" not in calls.read_text() and time.time() < deadline:
            time.sleep(0.2)
        return out, calls.read_text().splitlines()

    def body(name):
        f = home / f"job_body_{name}.sh"
        check = subprocess.run(["bash", "-n", str(f)], capture_output=True, text=True)
        assert check.returncode == 0, check.stderr
        return f.read_text()

    return run, body, put, shm, home, g


def test_run_g4arm8_sftset_and_check_render_their_jobs(tmp_path):
    run, body, put, shm, home, g = t8_world(tmp_path)
    out, _ = run("sftset", TAG="s1")
    assert out.returncode != 0 and "no " in out.stdout and "targets/targets.parquet" in out.stdout
    put("data/G4-8/targets/targets.parquet")
    out, _ = run("sftset", TAG="s2")
    assert out.returncode == 0 and "started g4a8_sftset_s2 on w0" in out.stdout, out.stdout + out.stderr
    b = body("g4a8_sftset_s2")
    for x in ("g4arm8.py sftset --targets", "/data/G4-8/plan/plan.parquet", "/rl/G4-7/data/pool/pool.parquet",
              "/rl/G4-7/data/plan/plan.parquet", "blind100.jsonl", "mainset/flan-eval/eval-rows.parquet",
              "--sft-eval books=", "--sft-eval btval=", "--sft-eval own=", "/data/G4-8/sft/smoke/",
              "/data/G4-8/sft/rlval/", "/data/G4-8/sft/manifest.json", "SFTSET8_DONE"):
        assert x in b, x
    assert b.index("/sft/rlval/") < b.index("/data/G4-8/sft/manifest.json") and "rm " not in b
    out, _ = run("sftset", host="1", TAG="s3")
    assert out.returncode != 0 and "runs on w0 only" in out.stdout
    put("data/G4-8/sft/manifest.json", "{}")
    out, _ = run("sftset", TAG="s4")
    assert "there before" in body("g4a8_sftset_s4")
    out, _ = run("check", TAG="c1")
    assert out.returncode == 0, out.stdout + out.stderr
    b = body("g4a8_check_c1")
    assert "g4arm.py check --data" in b and "--arm G4-8" in b and "/data/G4-8/sft/check.txt" in b


def full_manifest():
    return json.dumps({"steps": 190, "eval_every": 32, "save_steps": [64, 128],
                       "candidates": ["G4-8-s64", "G4-8-s128", "G4-8-s190"]})


def test_run_g4arm8_train_prepares_and_launches_the_smoke_and_the_full_run(tmp_path):
    run, body, put, shm, home, g = t8_world(tmp_path)
    for d in ("data/G4-8/sft", "data/G4-8/sft/smoke"):
        for f in ("train.parquet", "template.jinja"):
            put(f"{d}/{f}")
    put("data/G4-8/sft/smoke/manifest.json", "{}")
    put("data/G4-8/sft/manifest.json", full_manifest())
    out, _ = run("train", RUN="g4arm_G4-8_s8", PHASE="prep")
    assert out.returncode != 0 and "not checked" in out.stdout
    put("data/G4-8/sft/check.txt")
    out, calls = run("train", RUN="g4arm_G4-8_s8", PHASE="prep")
    assert out.returncode == 0 and out.stdout.strip().splitlines()[-1] == "prepared g4arm_G4-8_s8 on w0", out.stdout + out.stderr
    assert not any(c.startswith("SETSID") for c in calls)
    d = shm / "translator/g4arm8/data-g4arm_G4-8_s8"
    launch = (d / "launch.sh").read_text()
    for x in ("MULTIHOST=1", "NAME=g4arm_G4-8_s8", "STEPS=8", "ROOT=gs://t5-dutch-english-usc2/mbert/translator/sft/g4arm",
              "--eval-every 4", "--save-steps 4", "--warmup-steps 10", "--decay-fraction 0", "books=", "btval=", "own=",
              "max_target_length=4096", "lora.lora_rank=32", "lora.lora_alpha=16.0", "learning_rate=5e-05",
              "lr_schedule_type=wsd", f"DATA={d}/train.parquet"):
        assert x in launch.replace("\\ ", " "), x
    assert any("/data/G4-8/sft/smoke/train.parquet" in c for c in calls)
    out, calls = run("train", RUN="g4arm_G4-8_s8", PHASE="launch")
    assert out.returncode == 0 and "started g4arm_G4-8_s8 on t1v-n-x-w-0" in out.stdout, out.stdout + out.stderr
    assert sum(c.startswith("SETSID") for c in calls) == 1
    # the full run: sftset's steps, eval every and save steps
    out, _ = run("train", RUN="g4arm_G4-8", PHASE="prep")
    assert out.returncode == 0, out.stdout + out.stderr
    # printf %q escapes the blanks and the comma; bash reads them back
    launch = (shm / "translator/g4arm8/data-g4arm_G4-8/launch.sh").read_text().replace("\\", "")
    assert "STEPS=190" in launch and "--eval-every 32" in launch and "--save-steps 64,128" in launch
    for run_name, msg in (("g4arm_G4-7", "no run of G4-8"), ("g4arm_G4-8_s9", "no run of G4-8")):
        out, _ = run("train", RUN=run_name, PHASE="prep")
        assert out.returncode != 0 and msg in out.stdout, (run_name, out.stdout)
    put("runs/g4arm_G4-8b/summary.json")
    out, _ = run("train", RUN="g4arm_G4-8b", PHASE="prep")
    assert out.returncode != 0 and "a new run takes a new RUN" in out.stdout
    out, _ = run("train", RUN="g4arm_G4-8c", PHASE="launch")
    assert out.returncode != 0 and "PHASE=prep first" in out.stdout


def test_run_g4arm8_smoke8_adapter_div_and_report8_render_their_jobs(tmp_path):
    run, body, put, shm, home, g = t8_world(tmp_path)
    out, _ = run("smoke8", RUN="g4arm_G4-8", TAG="m1")
    assert out.returncode != 0 and "no smoke run" in out.stdout
    out, _ = run("smoke8", RUN="g4arm_G4-8_s8", TAG="m2")
    assert out.returncode == 0, out.stdout + out.stderr
    b = body("g4a8_smoke8_m2")
    for x in ("runs/g4arm_G4-8_s8/adapter-step4.json", "g4arm8.py smoke8 --run g4arm_G4-8_s8", "/sft/smoke8-g4arm_G4-8_s8.json",
              "exit $code", "SMOKE8_DONE"):
        assert x in b, x
    put("data/G4-8/sft/manifest.json", full_manifest())
    out, _ = run("adapter", RUN="g4arm_G4-8", TAG="a1")
    assert out.returncode == 0, out.stdout + out.stderr
    b = body("g4a8_adapter_a1")
    for x in ("g4arm8.py trained --run g4arm_G4-8", "--adapter G4-8-s190=", "--adapter G4-8-s64=", "--adapter G4-8-s128=",
              "runs/g4arm_G4-8/adapter-step$k.npz", "/G4-8-s$k/adapter.npz", "for k in 64 128; do",
              "/G4-8-s190/adapter.npz", "/data/G4-8/sft/trained.json", "TRAINED8_DONE"):
        assert x in b, x
    assert b.index("g4arm8.py trained") < b.index("/G4-8-s$k/adapter.npz") and "rm " not in b
    put("G4-8-s128/adapter.json", "{}")
    out, _ = run("adapter", RUN="g4arm_G4-8", TAG="a2")
    assert out.returncode != 0 and "G4-8-s128 has an adapter.json" in out.stdout
    # RL-val's samples: G4-0 without, a candidate with its LoRA (rank 32 / alpha 16), the same seed for both
    put("data/G4-8/sft/rlval/segments-w2.parquet")
    out, _ = run("generate", host="2", JOB="div", MODEL="G4-0", TAG="d1")
    assert out.returncode == 0, out.stdout + out.stderr
    b0 = body("g4a8_gen_div_G4-0_d1")
    seed = g4rl.seed_int("en-nl-g4arm/0/G4-8/div/w2")
    assert f"--copies 8 --sample samples=1.0:{seed}" in b0 and "--lora" not in b0 and "/sft/div/G4-0/{sample}/w2'" in b0
    put("G4-8-s64/adapter.npz")
    put("G4-8-s64/adapter.json", json.dumps({"rank": 32, "alpha": 16.0, "scale": 0.5}))
    out, _ = run("generate", host="2", JOB="div", MODEL="G4-8-s64", TAG="d2")
    assert out.returncode == 0, out.stdout + out.stderr
    b = body("g4a8_gen_div_G4-8-s64_d2")
    lora = f"{shm}/translator/g4arm8/adapter-g4a8_gen_div_G4-8-s64_d2"
    assert f"samples=1.0:{seed}" in b and f"--lora {lora}" in b and f"rm -rf -- {lora}" in b
    assert b.index("translate.py") < b.index(f"rm -rf -- {lora}")
    put("G4-8-s128/adapter.npz")
    put("G4-8-s128/adapter.json", json.dumps({"rank": 16, "alpha": 16.0, "scale": 1.0}))
    out, _ = run("generate", host="2", JOB="div", MODEL="G4-8-s128", TAG="d3")
    assert out.returncode != 0 and "not rank 32, alpha 16" in out.stdout
    assert not (shm / "translator/g4arm8/adapter-g4a8_gen_div_G4-8-s128_d3").exists()
    out, _ = run("generate", JOB="div", MODEL="G4-7-i15", TAG="d4")
    assert out.returncode != 0 and "MODEL is G4-0 or a candidate" in out.stdout
    # report8: refused before its inputs are there, then its job
    cands = "G4-8-s64 G4-8-s128 G4-8-s190"
    out, _ = run("report8", CANDIDATES=cands, O6="R-oud", TAG="r1")
    assert out.returncode != 0 and "naturalness-G4-8/predictions.parquet" in out.stdout
    for rel in ("naturalness-G4-8/predictions.parquet", "naturalness-G4-8/predictions-c2.parquet",
                "data/G4-5-split/split.json", "data/G4-5-split/refqe.parquet", "rl/G4-7/data/plan/plan.parquet",
                "data/G4-8/sft/smoke8-g4arm_G4-8_s8.json"):
        put(rel)
    put("data/G4-8/sft/trained.json", json.dumps({"run": "g4arm_G4-8b"}))
    for m in ("G4-0", *cands.split()):
        put(f"data/G4-8/sft/div/{m}/samples/w0/DONE")
    out, _ = run("report8", CANDIDATES=cands, O6="R-x", TAG="r2")
    assert out.returncode != 0 and "O6 is R-oud, R-nieuw or open" in out.stdout
    out, _ = run("report8", CANDIDATES="G4-8-s64 G4-7-i15", O6="open", TAG="r3")
    assert out.returncode != 0 and "G4-7-i15 is no candidate of G4-8" in out.stdout
    out, _ = run("report8", CANDIDATES=cands, O6="R-nieuw", SMOKE_RUN="g4arm_G4-8_s8", TAG="r4")
    assert out.returncode == 0, out.stdout + out.stderr
    b = body("g4a8_report8_r4")
    x8 = f"{shm}/translator/g4arm8/report8-r4"
    for x in (f"--candidate G4-8-s64={x8}/G4-8-s64", f"--lora G4-8-s190={x8}/lora-G4-8-s190.json",
              f"--div G4-0={x8}/div/G4-0", f"--reference G4-4={x8}/G4-4", "--o6 R-nieuw", f"--smoke {x8}/smoke8.json",
              "runs/g4arm_G4-8b/summary.json", "g4arm8.py loraline --logs", "g4a_translate_${m}_*",
              "/report-G4-8/tally.json", "/naturalness-G4-8/pll.parquet", "/report-G4-8/", "REPORT8_TRAIN_DONE"):
        assert x in b, x


# ---------------------------------------------------------------------------------------------------- run_g4arm.sh

def g48_world(tmp_path):
    run, shm, home, g = t7.g47_world(tmp_path)
    files = {"G4-8-s64/adapter.npz": "x", "G4-8-s64/adapter.json": json.dumps({"rank": 32, "alpha": 16.0, "scale": 0.5}),
             "G4-8-s128/adapter.npz": "x", "G4-8-s128/adapter.json": json.dumps({"rank": 16, "alpha": 16.0, "scale": 1.0}),
             "naturalness-G4-8/texts.parquet": "x",
             "report-G4-8/report.json": json.dumps({"selected": "G4-8-s64", "blind_due": True})}
    for rel, text in files.items():
        (g / rel).parent.mkdir(parents=True, exist_ok=True)
        (g / rel).write_text(text)
    return run, shm, home, g


def test_g48_stages_of_run_g4arm(tmp_path):
    run, shm, home, g = g48_world(tmp_path)
    for stage in ("train", "adapter", "report7", "report", "build", "smoke-translate", "check"):
        out, calls = run(stage, ARM="G4-8")
        assert out.returncode != 0 and "is no stage of G4-8" in out.stdout, (stage, out.stdout + out.stderr)
        assert not any(c.startswith(("SETSID", "gsutil")) for c in calls), stage
    out, _ = run("translate", ARM="G4-8", MODEL="G4-8-s64", TAG="t2")
    assert out.returncode == 0 and "started g4a_translate_G4-8-s64_t2 on w0" in out.stdout, out.stdout + out.stderr
    b = (home / "job_body_g4a_translate_G4-8-s64_t2.sh").read_text()
    copy = f"{shm}/translator/g4arm/adapter-g4a_translate_G4-8-s64_t2"
    assert f"--lora {copy}" in b and "/eval/G4-8-s64/translations/w0/tiny" in b
    assert b.index("translate_class ") < b.index(f"rm -rf -- {copy}") < b.index("TRANSLATED")
    out, _ = run("translate", ARM="G4-8", MODEL="G4-8-s128", TAG="t3")
    assert out.returncode != 0 and "G4-8-s128/adapter.json is not rank 32, alpha 16" in out.stdout
    for model, arm, msg in (("G4-0", "G4-8", "ARM=G4-8: MODEL is a candidate"),
                            ("G4-7-i5", "G4-8", "G4-7-i5 is G4-7's"), ("G4-8-s64", "G4-7", "ARM=G4-7: MODEL is a candidate"),
                            ("G4-8-s64", "G4-6c", "G4-8-s64 is G4-8's"),
                            ("G4-8-s0", "G4-8", "MODEL is G4-1")):
        for stage in ("translate", "score"):
            out, _ = run(stage, ARM=arm, MODEL=model, TAG="t4")
            assert out.returncode != 0 and msg in out.stdout, (model, arm, stage, out.stdout)
    out, _ = run("translate", ARM="G4-8", MODEL="G4-8-s64", SET="heldout", TAG="t5")
    assert out.returncode != 0 and "SET=eval only" in out.stdout
    out, _ = run("score", ARM="G4-8", MODEL="G4-8-s64", TAG="t6")
    assert out.returncode == 0, out.stdout + out.stderr
    b = (home / "job_body_g4a_score_G4-8-s64_t6.sh").read_text()
    assert "/eval/G4-8-s64/scores/metricx-w0.parquet" in b and "comet_score.py score" in b
    out, _ = run("texts", ARM="G4-8", CANDIDATES="G4-8-s64 G4-7-i5", TAG="t7")
    assert out.returncode != 0 and "G4-7-i5 is no candidate of G4-8" in out.stdout
    out, _ = run("texts", ARM="G4-8", CANDIDATES="G4-8-s64 G4-8-s128 G4-8-s190", TAG="t8")
    assert out.returncode == 0, out.stdout + out.stderr
    b = (home / "job_body_g4a_texts_t8.sh").read_text()
    assert "echo 'G4-8-s64 G4-8-s128 G4-8-s190 G4-0 G4-4' |" in b and "naturalness-G4-8/texts.parquet" in b
    for stage, x in (("classify-c2", "naturalness-G4-8/predictions-c2.parquet"), ("pll", "naturalness-G4-8/pll.parquet")):
        out, _ = run(stage, ARM="G4-8", TAG=f"t9{stage}")
        assert out.returncode == 0, out.stdout + out.stderr
        name = {"classify-c2": "g4a_classify_c2", "pll": "g4a_pll"}[stage]
        b = (home / f"job_body_{name}_t9{stage}.sh").read_text()
        assert x in b and "naturalness-G4-8/texts.parquet" in b and "naturalness-G4-7" not in b
    out, _ = run("classify", ARM="G4-8", TAG="t10")
    assert out.returncode == 0, out.stdout + out.stderr
    assert "naturalness-G4-8/predictions.parquet" in (home / "job_body_arms_classify_t10.sh").read_text()
    # the blind packages of report8's choice, only with (a1) and (a3) holding
    out, _ = run("blind", ARM="G4-8", SELECTED="G4-8-s64", TAG="t11")
    assert out.returncode == 0, out.stdout + out.stderr
    b = (home / "job_body_g4a_blind_t11.sh").read_text()
    assert "/eval/G4-8-s64/pairs/pairs-w*.parquet" in b and "sft/blind/G4-8/" in b and "sft/blind/G4-8-tg/" in b
    for selected, report, msg in (("G4-8-s128", {"selected": "G4-8-s64", "blind_due": True}, "not report8's choice"),
                                  ("G4-8-s64", {"selected": "G4-8-s64", "blind_due": False}, "not report8's choice"),
                                  ("G4-7-i10", {"selected": "G4-8-s64", "blind_due": True}, "no candidate of G4-8")):
        (g / "report-G4-8/report.json").write_text(json.dumps(report))
        out, _ = run("blind", ARM="G4-8", SELECTED=selected, TAG="t12")
        assert out.returncode != 0 and msg in out.stdout, (selected, out.stdout)
