"""The measurement G4-bon (protocol-g4.md §15): g4bon.py (the search part, the four selections with G4-7's reward
unchanged, C2's scores with c2_overflow, the selection's outputs in the evaluation's layout, the checks against the
sampling probe, C1's texts, the report with paired bootstrap intervals) and run_g4arm.sh's stages with ARM=G4-bon, on
synthetic inputs (no model, no TPU)."""

import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "en-nl-translator"))
sys.path.insert(0, str(ROOT / "experiments" / "flan-nl"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import armeval  # noqa: E402
import g4arm  # noqa: E402
import g4bon  # noqa: E402
import g4mix  # noqa: E402
import g4rl  # noqa: E402

N_W, N_F = 6, 5          # search segments per set in the synthetic world


def sample(qe, logit, fail=(), reward=None):
    return {"qe": qe, "logit": logit, "fail": list(fail), "reward": logit - 0.0 if reward is None else reward,
            "ratio": 1.0}


# ---------------------------------------------------------------------------------------------------- the choice

def test_choose_qe_r7_c2q_c2_and_their_ties_and_fallbacks():
    s = [sample(2.0, -3.0, reward=-3.0), sample(1.5, -9.0, reward=-9.0), sample(1.5, -1.0, ("length",), reward=-30.0),
         sample(2.5, 4.0, reward=2.0), sample(1.9, 3.0, reward=3.0), sample(3.0, 6.0, ("english_left",), reward=-30.0),
         sample(1.8, 3.0, reward=1.0), sample(4.0, -12.0, reward=-15.0)]
    picks, n_bound = g4bon.choose(s, qe0=2.0)
    assert picks["qe"] == (1, False)                       # 1.5 twice: the earlier sample
    assert picks["r7"] == (4, False)                       # the highest reward
    # c2q: no fail and QE <= 2.0 -> samples 0, 1, 4, 6; the highest logit 3.0 twice: the lower QE (6: 1.8)
    assert picks["c2q"] == (6, False) and n_bound == 4
    # c2: no fail, any QE -> sample 3 (logit 4.0; 5 has 6.0 but fails)
    assert picks["c2"] == (3, False)
    # nothing within the bound: c2q falls back on qe's choice; everything fails: c2 too
    t = [sample(2.5, 1.0), sample(2.4, 2.0), sample(2.6, -1.0, ("qe",))] + [sample(3.0, 0.0)] * 5
    picks, n_bound = g4bon.choose(t, qe0=2.0)
    assert picks["c2q"] == (1, True) and n_bound == 0 and picks["c2"] == (1, False)
    u = [sample(2.0 + k, 1.0, ("not_ended",), reward=-30.0) for k in range(8)]
    picks, _ = g4bon.choose(u, qe0=9.0)
    assert picks["c2"] == (0, True) and picks["c2q"] == (0, True) and picks["r7"] == (0, False)
    # r7 ties: lower QE, then the earlier sample
    v = [sample(2.0, 0.0, reward=1.0), sample(1.0, 0.0, reward=1.0)] + [sample(0.5, 0.0, reward=0.5)] * 6
    assert g4bon.choose(v, qe0=3.0)[0]["r7"] == (1, False)


def row(i, qe, tokens_out, ended=True, loop=False, english=False, empty=False, ref="ref", hyp="hyp"):
    return {"id": i, "set": "wmt24pp", "source": "src text", "hypothesis": hyp, "reference": ref, "ended": ended,
            "decode_loop": loop, "english_left": english, "empty": empty, "tokens_out": tokens_out,
            "metricx_qe": qe, "metricx_ref": qe, "comet22": 0.8}


def test_scored_samples_is_g4_7s_reward_unchanged():
    anchor = {"l0": 100, "qe0": 2.0}
    rows = [row("a", 2.5, 120), row("a", 1.0, 160), row("a", 1.0, 100, loop=True), row("a", 8.0, 100)]
    c2 = [{"logit_c2": 15.0, "c2_overflow": False}, {"logit_c2": -3.0, "c2_overflow": False},
          {"logit_c2": 1.0, "c2_overflow": False}, {"logit_c2": None, "c2_overflow": True}]
    out = g4bon.scored_samples(rows, c2, anchor)
    # clip 12, QE penalty 2 x 0.5, length 1.2 inside +-15%
    assert out[0]["reward"] == pytest.approx(12.0 - 1.0 - 2 * max(0.0, abs(math.log(1.2)) - math.log(1.15)))
    assert out[0]["logit"] == 12.0 and not out[0]["fail"]
    assert out[1]["fail"] == ["length"] and out[1]["reward"] == g4rl.FAIL_REWARD == -30.0
    assert out[2]["fail"] == ["decode_loop"]
    assert out[3]["fail"] == ["qe", "c2_overflow"] and out[3]["logit"] is None
    for o, r, c in zip(out, rows, c2):
        assert o["reward"] == g4rl.reward7(g4bon.sample_view(r), r["metricx_qe"], c, anchor)["reward"]


def test_c2_scores_take_the_texts_tokens_and_refuse_holes():
    preds = [{"model": "M", "set": "flan", "id": "x", "p_original": 0.5},
             {"model": "M/t10s1", "set": "flan", "id": "x", "p_original": 1e-7},
             {"model": "M/t10s2", "set": "flan", "id": "x", "p_original": 0.9}]
    texts = [{"model": "M", "set": "flan", "id": "x", "tokens": 10},
             {"model": "M/t10s1", "set": "flan", "id": "x", "tokens": 1022},
             {"model": "M/t10s2", "set": "flan", "id": "x", "tokens": 1023}]
    got = g4bon.c2_scores(preds, texts, ["M", "M/t10s1", "M/t10s2"], ["x"])
    assert got["M"]["x"] == {"logit_c2": 0.0, "c2_overflow": False}
    assert got["M/t10s1"]["x"]["logit_c2"] == -12.0          # clipped, as c2cal7
    assert got["M/t10s2"]["x"] == {"logit_c2": None, "c2_overflow": True}
    with pytest.raises(SystemExit, match="C2 scored 2 of the 3"):
        g4bon.c2_scores(preds[:2], texts, ["M", "M/t10s1", "M/t10s2"], ["x"])
    with pytest.raises(SystemExit, match="twice"):
        g4bon.c2_scores(preds + preds[:1], texts, ["M"], ["x"])


# ---------------------------------------------------------------------------------------------------- a world

def ids_of(prefix, n):
    return [f"{prefix}:{k:03d}" for k in range(n)]


def make_world(tmp_path, monkeypatch, seed=7):
    """evalsets, a split (search and decision part, its digest patched in as G4-5.4's), G4-0's greedy evaluation, the
    probe's 8 samples with best8 and report, C2's predictions and texts as c2cal7 wrote them, on worker files w0/w1"""
    import random

    rnd = random.Random(seed)
    w = tmp_path / "world"
    wmt = ids_of("wmt24pp", 2 * N_W + 2)          # two bad sources
    flan = ids_of("flan", 2 * N_F)
    flores = ids_of("flores", 4)
    evalsets = []
    for k, i in enumerate(wmt):
        evalsets.append({"id": i, "set": "wmt24pp", "meta": json.dumps({"is_bad_source": k >= 2 * N_W})})
    evalsets += [{"id": i, "set": "flan", "meta": None} for i in flan]
    evalsets += [{"id": i, "set": "flores", "meta": None} for i in flores]
    g4arm.write_parquet(evalsets, w / "evalsets.parquet")
    assign = {}
    for k, i in enumerate(wmt[:2 * N_W]):
        assign[i] = "search" if k % 2 == 0 else "decision"
    for i in wmt[2 * N_W:]:
        assign[i] = "search"
    for k, i in enumerate(flan):
        assign[i] = "search" if k % 2 == 0 else "decision"
    for k, i in enumerate(flores):
        assign[i] = "search" if k % 2 == 0 else "decision"
    split = {"assign": assign, "digest": g4mix.digest(assign),
             "inputs": {"evalsets": g4arm.sha256_file(w / "evalsets.parquet")}}
    (w / "split.json").write_text(json.dumps(split))
    monkeypatch.setattr(g4arm, "SPLIT_DIGEST6C", split["digest"])
    monkeypatch.setattr(g4bon, "N_SEARCH", {"wmt24pp": N_W, "flan": N_F})
    evaluated = wmt + flan + flores

    def pair(i, hyp, tokens_out, **flags):
        s = i.split(":")[0]
        return {"id": i, "set": s, "source": f"source {i} " * 3, "hypothesis": hyp,
                "reference": None if s == "flan" else f"reference {i} " * 3, "ended": flags.get("ended", True),
                "tokens_in": 30, "tokens_out": tokens_out, "decode_loop": flags.get("loop", False),
                "english_left": flags.get("english", False), "empty": not hyp.strip(), "tokens": 20, "over_512": False}

    def write_output(d, rows, qe):
        half = len(rows) // 2
        for k, part in ((0, rows[:half]), (1, rows[half:])):
            g4arm.write_parquet(part, d / "pairs" / f"pairs-w{k}.parquet")
            g4arm.write_parquet([{"id": r["id"], "qe_source_hypothesis": qe[r["id"]],
                                  "ref_source_hypothesis_reference": None if r["reference"] is None else qe[r["id"]] + 0.1}
                                 for r in part], d / "scores" / f"metricx-w{k}.parquet")
            g4arm.write_parquet([{"id": r["id"], "comet22": 0.9 - 0.02 * qe[r["id"]]} for r in part if r["reference"]],
                                d / "scores" / f"comet-w{k}.parquet")

    greedy = [pair(i, f"greedy {i}", 40) for i in evaluated]
    qe0 = {i: 2.0 for i in evaluated}
    write_output(w / "eval" / "G4-0", greedy, qe0)
    preds, texts, qes = [], [], {}
    for k, (name, _, _) in enumerate(g4bon.SAMPLES):
        qe = {i: round(rnd.uniform(0.5, 4.0), 4) for i in wmt + flan}
        rows = [pair(i, f"sample {k} {i}", rnd.choice([30, 40, 50, 70])) for i in wmt + flan]
        write_output(w / "sampling" / "G4-0" / name, rows, qe)
        qes[name] = qe
        for r in rows:
            p = rnd.uniform(0.0005, 0.9995)
            preds.append({"model": f"G4-0/{name}", "set": r["set"], "id": r["id"], "p_original": p})
            texts.append({"model": f"G4-0/{name}", "set": r["set"], "id": r["id"], "text": "t", "tokens": 30})
    for r in greedy:
        if r["set"] in g4bon.SETS:
            preds.append({"model": "G4-0", "set": r["set"], "id": r["id"], "p_original": 0.2})
            texts.append({"model": "G4-0", "set": r["set"], "id": r["id"], "text": "t", "tokens": 30})
    g4arm.write_parquet(preds, w / "cal7" / "predictions.parquet")
    g4arm.write_parquet(texts, w / "cal7" / "texts.parquet")
    best8 = [{"id": i, "chosen": min(qes, key=lambda n: (qes[n][i], g4bon.SAMPLE_NAMES.index(n))),
              "qe": [qes[n][i] for n in g4bon.SAMPLE_NAMES]} for i in wmt + flan]
    g4arm.write_parquet(best8, w / "sampling" / "G4-0" / "best8" / "choice.parquet")
    report = {"settings": {"settings": [["t07", 0.7, 1]] + [[n, t, s] for n, t, s in g4bon.SAMPLES], "note": "x"}}
    (w / "probe-report.json").write_text(json.dumps(report))
    return w


def select_args(w, out, model="G4-0", sampling=None, c2=None, **extra):
    sampling = sampling or w / "sampling" / "G4-0"
    c2 = c2 or w / "cal7"
    args = ["select", "--model", model]
    for n in g4bon.SAMPLE_NAMES:
        args += ["--sample", f"{n}={sampling / n}"]
    args += ["--anchor", str(w / "eval" / "G4-0"), "--c2-predictions", str(c2 / "predictions.parquet"),
             "--c2-texts", str(c2 / "texts.parquet"), "--evalsets", str(w / "evalsets.parquet"),
             "--split", str(w / "split.json"), "--out", str(out)]
    for k, v in extra.items():
        args += [f"--{k.replace('_', '-')}", str(v)]
    return args


def test_search_part_is_g4_5_4s_and_refuses_another_split(tmp_path, monkeypatch):
    w = make_world(tmp_path, monkeypatch)
    evalsets = g4bon.load_evalsets(w / "evalsets.parquet")
    split = g4bon.load_split(w / "split.json")
    ids = g4bon.search_ids(evalsets, split)
    assert [len(ids[s]) for s in g4bon.SETS] == [N_W, N_F]
    assert not any(i.startswith("flores") for v in ids.values() for i in v)
    # the bad sources are not in (b)'s WMT24++ even though they sit in the search part
    assert all(int(i.split(":")[1]) < 2 * N_W for i in ids["wmt24pp"])
    bad = dict(split, assign={**split["assign"], "flan:000": "decision"})
    with pytest.raises(SystemExit, match="not G4-5.4's"):
        g4bon.search_ids(evalsets, bad)
    monkeypatch.setattr(g4bon, "N_SEARCH", {"wmt24pp": N_W + 1, "flan": N_F})
    with pytest.raises(SystemExit, match="expected"):
        g4bon.search_ids(evalsets, split)


def test_bondata_keeps_the_search_part_on_its_workers(tmp_path, monkeypatch, capsys):
    w = make_world(tmp_path, monkeypatch)
    seg = tmp_path / "seg"
    evalsets = g4arm.read_rows(w / "evalsets.parquet")
    probe = [r["id"] for r in evalsets if r["set"] in ("wmt24pp", "flan")]
    for k in range(2):
        g4arm.write_parquet([{"id": i, "text": "x", "tokens": 5} for i in probe[k::2]], seg / f"segments-w{k}.parquet")
        (seg / f"classes-w{k}.txt").write_text("tiny small\n")
    out = tmp_path / "out"
    g4bon.main(["bondata", "--evalsets", str(w / "evalsets.parquet"), "--split", str(w / "split.json"),
                "--segments", str(seg), "--workers", "2", "--out", str(out)])
    kept = [r["id"] for k in range(2) for r in g4arm.read_rows(out / f"segments-w{k}.parquet")]
    ids = g4bon.search_ids(g4bon.load_evalsets(w / "evalsets.parquet"), g4bon.load_split(w / "split.json"))
    assert sorted(kept) == sorted(ids["wmt24pp"] + ids["flan"])
    assert (out / "classes-w1.txt").read_text() == "tiny small\n"
    assert "BONDATA" in capsys.readouterr().out
    # a search segment the probe did not translate; another evalsets.parquet
    g4arm.write_parquet([{"id": i, "text": "x", "tokens": 5} for i in probe[2::2]], seg / "segments-w0.parquet")
    with pytest.raises(SystemExit, match="search segments"):
        g4bon.main(["bondata", "--evalsets", str(w / "evalsets.parquet"), "--split", str(w / "split.json"),
                    "--segments", str(seg), "--workers", "2", "--out", str(out)])
    split = json.loads((w / "split.json").read_text())
    split["inputs"]["evalsets"] = "0" * 64
    (w / "split.json").write_text(json.dumps(split))
    with pytest.raises(SystemExit, match="not the evalsets.parquet of the split"):
        g4bon.main(["bondata", "--evalsets", str(w / "evalsets.parquet"), "--split", str(w / "split.json"),
                    "--segments", str(seg), "--workers", "2", "--out", str(out)])


def test_select_writes_each_choice_in_the_evaluations_layout_and_equals_best8(tmp_path, monkeypatch, capsys):
    w = make_world(tmp_path, monkeypatch)
    out = tmp_path / "sel"
    g4bon.main(select_args(w, out, best8=w / "sampling/G4-0/best8/choice.parquet", probe_report=w / "probe-report.json"))
    assert "BONSELECT" in capsys.readouterr().out
    choice = {r["id"]: r for r in g4arm.read_rows(out / "choice.parquet")}
    ids = g4bon.search_ids(g4bon.load_evalsets(w / "evalsets.parquet"), g4bon.load_split(w / "split.json"))
    assert set(choice) == set(ids["wmt24pp"] + ids["flan"])
    for sel in g4bon.SELECTIONS:
        scored = armeval.scored(out / sel)                  # the layout of an evaluation: pairs/, scores/
        assert set(scored) == set(choice)
        for i, r in choice.items():
            k = g4bon.SAMPLE_NAMES.index(r[f"{sel}_choice"])
            assert scored[i]["hypothesis"] == f"sample {k} {i}" and scored[i]["metricx_qe"] == r["qe"][k]
            if sel == "qe":
                assert r["qe"][k] == min(r["qe"])
            if sel == "r7":
                assert r["reward"][k] == max(r["reward"])
            if sel == "c2q" and not r["c2q_fallback"]:
                assert not r["fail"][k] and r["qe"][k] <= r["qe0"]
                assert r["logit"][k] == max(r["logit"][j] for j in range(8) if not r["fail"][j] and r["qe"][j] <= r["qe0"])
    for r in choice.values():
        assert r["greedy_logit"] == pytest.approx(math.log(0.2 / 0.8))
        # tokens_out 70 against l0 40 is outside [0.67, 1.5]: a fail of G4-7's length band
        for j, f in enumerate(r["fail"]):
            assert ("length" in f.split(",")) == (r["ratio"][j] > 1.5)
    stats = json.loads((out / "select.json").read_text())
    assert stats["best8_checked"] and set(stats["stats"]) == set(g4bon.SETS)
    assert stats["stats"]["flan"]["qe"]["same_as_qe"] == 1.0
    # another best8 (one segment chose another sample): refused
    rows = g4arm.read_rows(w / "sampling/G4-0/best8/choice.parquet")
    rows[0]["chosen"] = "t10s8" if rows[0]["chosen"] != "t10s8" else "t10s7"
    g4arm.write_parquet(rows, tmp_path / "best8-other.parquet")
    if rows[0]["id"] in choice:
        with pytest.raises(SystemExit, match="differs from the probe's best8 at 1"):
            g4bon.main(select_args(w, tmp_path / "sel2", best8=tmp_path / "best8-other.parquet"))
    # the probe made t10s3 with another seed: refused
    rep = json.loads((w / "probe-report.json").read_text())
    rep["settings"]["settings"][3][2] = 9
    (tmp_path / "rep.json").write_text(json.dumps(rep))
    with pytest.raises(SystemExit, match="t10s3"):
        g4bon.main(select_args(w, tmp_path / "sel3", probe_report=tmp_path / "rep.json"))


def test_select_refuses_missing_samples_and_c2(tmp_path, monkeypatch):
    w = make_world(tmp_path, monkeypatch)
    args = select_args(w, tmp_path / "s")
    with pytest.raises(SystemExit, match="in this order"):
        g4bon.main([a.replace("t10s2=", "t10s9=") for a in args])
    (w / "sampling/G4-0/t10s5/pairs/pairs-w1.parquet").unlink()
    (w / "sampling/G4-0/t10s5/scores/metricx-w1.parquet").unlink()
    (w / "sampling/G4-0/t10s5/scores/comet-w1.parquet").unlink()
    with pytest.raises(SystemExit, match="t10s5: no scored sample"):
        g4bon.main(args)


def test_texts_are_the_search_part_of_every_output(tmp_path, monkeypatch, capsys):
    w = make_world(tmp_path, monkeypatch)
    out = tmp_path / "texts.parquet"
    g4bon.main(["texts", "--pairs", f"G4-0={w}/eval/G4-0/pairs/pairs-w*.parquet",
                "--pairs", f"G4-0/t10s1={w}/sampling/G4-0/t10s1/pairs/pairs-w*.parquet",
                "--evalsets", str(w / "evalsets.parquet"), "--split", str(w / "split.json"), "--out", str(out)])
    rows = g4arm.read_rows(out)
    assert len(rows) == 2 * (N_W + N_F) and {r["model"] for r in rows} == {"G4-0", "G4-0/t10s1"}
    assert all(r["set"] in g4bon.SETS for r in rows)
    with pytest.raises(SystemExit, match="twice"):
        g4bon.main(["texts", "--pairs", f"G4-0={w}/eval/G4-0/pairs/pairs-w*.parquet",
                    "--pairs", f"G4-0={w}/eval/G4-0/pairs/pairs-w*.parquet",
                    "--evalsets", str(w / "evalsets.parquet"), "--split", str(w / "split.json"), "--out", str(out)])
    with pytest.raises(SystemExit, match="no translation of"):
        g4bon.main(["texts", "--pairs", f"G4-0={w}/eval/G4-0/pairs/pairs-w0.parquet",
                    "--evalsets", str(w / "evalsets.parquet"), "--split", str(w / "split.json"), "--out", str(out)])


def test_c2texts_names_the_samples_and_both_greedy_translations(tmp_path, monkeypatch):
    import g4c2

    w = make_world(tmp_path, monkeypatch)
    monkeypatch.setattr(g4c2, "tokenizer_of", lambda checkpoint: "tok")
    monkeypatch.setattr(g4c2, "token_counter", lambda tok: lambda texts: [len(t.split()) * 400 for t in texts])
    out = tmp_path / "c2texts.parquet"
    args = ["c2texts", "--model", "G4-7-i5"]
    for n in g4bon.SAMPLE_NAMES:
        args += ["--sample", f"{n}={w}/sampling/G4-0/{n}/pairs/pairs-w*.parquet"]
    g4bon.main(args + ["--greedy", f"{w}/eval/G4-0/pairs/pairs-w*.parquet", "--g40", f"{w}/eval/G4-0/pairs/pairs-w*.parquet",
                       "--evalsets", str(w / "evalsets.parquet"), "--split", str(w / "split.json"),
                       "--checkpoint", str(tmp_path), "--out", str(out)])
    rows = g4arm.read_rows(out)
    names = {r["model"] for r in rows}
    assert names == {"G4-7-i5", "G4-0"} | {f"G4-7-i5/{n}" for n in g4bon.SAMPLE_NAMES}
    assert len(rows) == 10 * (N_W + N_F) and all(r["tokens"] == 400 * len(r["text"].split()) for r in rows)


# ---------------------------------------------------------------------------------------------------- the report

def test_paired_ci_is_reproducible_and_keeps_the_sets_sizes():
    d = {"wmt24pp": [1] * 30 + [0] * 70, "flan": [0] * 100}
    a, b = g4bon.paired_ci(d, "x"), g4bon.paired_ci(d, "x")
    assert a == b and a["delta"] == pytest.approx(0.15) and a["n"] == 200
    assert a["lo"] < 0.15 < a["hi"] and a["hi"] - a["lo"] < 0.15
    # FLAN without variance: the pooled interval only has WMT24++'s spread, halved
    only = g4bon.paired_ci({"wmt24pp": d["wmt24pp"]}, "x")
    assert (a["hi"] - a["lo"]) == pytest.approx((only["hi"] - only["lo"]) / 2, rel=0.15)


def candidate_from(w, tmp_path, monkeypatch):
    """A candidate G4-7-i5: its samples are G4-0's probe samples with other C2 scores, its greedy evaluation G4-0's"""
    import shutil

    shutil.copytree(w / "eval" / "G4-0", w / "eval" / "G4-7-i5")
    preds = g4arm.read_rows(w / "cal7" / "predictions.parquet")
    texts = g4arm.read_rows(w / "cal7" / "texts.parquet")
    p2 = [{**r, "model": r["model"].replace("G4-0", "G4-7-i5"), "p_original": min(0.999, r["p_original"] * 1.5)}
          for r in preds]
    t2 = [{**r, "model": r["model"].replace("G4-0", "G4-7-i5")} for r in texts]
    p2 += [r for r in preds if r["model"] == "G4-0"]
    t2 += [r for r in texts if r["model"] == "G4-0"]
    g4arm.write_parquet(p2, w / "c2-i5" / "predictions.parquet")
    g4arm.write_parquet(t2, w / "c2-i5" / "texts.parquet")
    g4bon.main(select_args(w, tmp_path / "sel-i5", model="G4-7-i5", c2=w / "c2-i5"))
    return w / "c2-i5" / "predictions.parquet"


def c1_predictions(w, tmp_path, models, share_of):
    ids = g4bon.search_ids(g4bon.load_evalsets(w / "evalsets.parquet"), g4bon.load_split(w / "split.json"))
    rows = []
    for m in models:
        for o in g4bon.OUTPUTS:
            name = m if o == "greedy" else f"{m}/{o}"
            for s in g4bon.SETS:
                for k, i in enumerate(ids[s]):
                    rows.append({"model": name, "set": s, "id": i,
                                 "p_original": 0.9 if k < share_of(m, o, s) * len(ids[s]) else 0.1})
    g4arm.write_parquet(rows, tmp_path / "c1.parquet")
    probe = [{**r, "model": r["model"].replace("G4-0/qe", "G4-0/best8")} for r in rows if r["model"].startswith("G4-0")]
    g4arm.write_parquet(probe, tmp_path / "probe-c1.parquet")
    return tmp_path / "c1.parquet", tmp_path / "probe-c1.parquet"


def test_report_tables_both_models_with_intervals_flags_and_checks(tmp_path, monkeypatch, capsys):
    w = make_world(tmp_path, monkeypatch)
    g4bon.main(select_args(w, tmp_path / "sel-g40"))
    c2_i5 = candidate_from(w, tmp_path, monkeypatch)
    shares = {("G4-0", "greedy"): 0.2, ("G4-0", "c2"): 0.6, ("G4-7-i5", "greedy"): 0.4, ("G4-7-i5", "r7"): 0.4}
    c1, probe = c1_predictions(w, tmp_path, ("G4-0", "G4-7-i5"), lambda m, o, s: shares.get((m, o), 0.3))
    g4arm.write_parquet([{"id": i, "qe_source_reference": 2.5} for i in ids_of("wmt24pp", 2 * N_W + 2)],
                        tmp_path / "refqe.parquet")
    out = tmp_path / "rep" / "report.md"
    out.parent.mkdir()
    args = ["report", "--evalsets", str(w / "evalsets.parquet"), "--split", str(w / "split.json"),
            "--refqe", str(tmp_path / "refqe.parquet"), "--anchor", str(w / "eval" / "G4-0"),
            "--predictions", str(c1), "--probe-predictions", str(probe),
            "--model", f"G4-0={w}/eval/G4-0,{w}/sampling/G4-0/t10s1,{tmp_path}/sel-g40",
            "--model", f"G4-7-i5={w}/eval/G4-7-i5,{w}/sampling/G4-0/t10s1,{tmp_path}/sel-i5",
            "--c2-check", f"G4-7-i5={c2_i5},{w}/cal7/predictions.parquet", "--out", str(out)]
    # the candidate's t10s1 here is G4-0's (same files): its QE equals the choice's sample t10s1, as it must
    g4bon.main(args)
    assert "BONREPORT" in capsys.readouterr().out
    r = json.loads(out.with_suffix(".json").read_text())
    g = r["models"]["G4-0"]["outputs"]
    n = N_W + N_F
    # share_of 0.2: the first k with k < 0.2 n of a set are "original": 2 of 6 WMT24++, 1 of 5 FLAN; c2 0.6: 4 and 3
    assert g["greedy"]["c1_pooled"] == pytest.approx(3 / n) and g["c2"]["c1_pooled"] == pytest.approx(7 / n)
    assert g["c2"]["c1_vs_greedy"]["delta"] == pytest.approx(4 / n)
    assert g["greedy"]["sets"]["wmt24pp"]["c1"] == pytest.approx(2 / 6) and g["greedy"]["sets"]["flan"]["c1"] == pytest.approx(1 / 5)
    assert "c1_vs_g40" not in g["greedy"] and "c1_vs_greedy" in g["c2"]
    assert g["c2"]["c1_vs_greedy"]["delta"] > 0.2 and g["c2"]["c1_vs_greedy"]["lo"] <= g["c2"]["c1_vs_greedy"]["delta"]
    assert g["qe"]["selection_measures"][0] == "QE" and g["c2"]["selection_measures"][0] == "C2"
    c = r["models"]["G4-7-i5"]["outputs"]
    assert c["greedy"]["c1_vs_g40"]["delta"] > 0 and c["r7"]["c1_vs_greedy"]["delta"] == pytest.approx(0.0)
    # the flag of report7: C2 up >= 0.10 against the own greedy translation while C1 up < 0.10
    for m in ("G4-0", "G4-7-i5"):
        for o in g4bon.OUTPUTS[1:]:
            y = r["models"][m]["outputs"][o]
            assert y["hack_flag"] == (y["c2_vs_greedy"] >= 0.10 and y["c1_vs_greedy"]["delta"] < 0.10)
    assert set(r["models"]["G4-0"]["contrasts"]) == {"r7-qe", "c2q-qe", "c2-qe", "qe-greedy", "r7-greedy", "c2q-greedy", "c2-greedy"}
    checks = r["checks"]["c1_against_probe"]
    assert checks["G4-0/qe ~ probe G4-0/best8"]["flipped"] == 0
    assert r["checks"]["c2_G4-7-i5_run"]["flipped"] == 0 and r["checks"]["c2_G4-7-i5_run"]["max_abs_dlogit"] == 0
    md = out.read_text()
    for x in ("## Tabel 1", "## Tabel 2", "## Tabel 3", "## Tabel 4", "## Controles", "## Wat dit niet zegt",
              "best-of-8: hoogste C2 met QE ≤ QE₀", "niet onafhankelijk"):
        assert x in md, x
    assert r["human_qe_wmt24pp"] == pytest.approx(2.5)
    # C1 without one output's scores: refused
    rows = [x for x in g4arm.read_rows(c1) if x["model"] != "G4-7-i5/c2q"]
    g4arm.write_parquet(rows, c1)
    with pytest.raises(SystemExit, match="C1 has no score of G4-7-i5/c2q"):
        g4bon.main(args)


def test_report_needs_g4_0(tmp_path, monkeypatch):
    w = make_world(tmp_path, monkeypatch)
    candidate_from(w, tmp_path, monkeypatch)
    c1, probe = c1_predictions(w, tmp_path, ("G4-7-i5",), lambda m, o, s: 0.3)
    g4arm.write_parquet([{"id": i, "qe_source_reference": 2.5} for i in ids_of("wmt24pp", 2 * N_W + 2)],
                        tmp_path / "refqe.parquet")
    with pytest.raises(SystemExit, match="no G4-0 among the models"):
        g4bon.main(["report", "--evalsets", str(w / "evalsets.parquet"), "--split", str(w / "split.json"),
                    "--refqe", str(tmp_path / "refqe.parquet"), "--anchor", str(w / "eval" / "G4-0"),
                    "--predictions", str(c1), "--probe-predictions", str(probe),
                    "--model", f"G4-7-i5={w}/eval/G4-7-i5,{w}/sampling/G4-0/t10s1,{tmp_path}/sel-i5",
                    "--out", str(tmp_path / "r.md")])


def add_g4_4(w, p_greedy=0.7):
    """G4-4 in the world as the probe left it (G4-bon.10): its 8 samples and best8 (G4-0's files), its evaluation (G4-0's
    with other texts), C2 of c2cal7 for its samples (G4-4/t10s<k>) and for its greedy translation only under the
    calibration name calib:sampling:G4-4, with the probe's texts of it"""
    import shutil

    shutil.copytree(w / "sampling" / "G4-0", w / "sampling" / "G4-4")
    for f in (w / "eval" / "G4-0" / "pairs").glob("pairs-w*.parquet"):
        rows = [{**r, "hypothesis": r["hypothesis"].replace("greedy", "g44 greedy")} for r in g4arm.read_rows(f)]
        g4arm.write_parquet(rows, w / "eval" / "G4-4" / "pairs" / f.name)
    shutil.copytree(w / "eval" / "G4-0" / "scores", w / "eval" / "G4-4" / "scores")
    greedy = armeval.read_rows(f"{w}/eval/G4-4/pairs/pairs-w*.parquet")
    preds = g4arm.read_rows(w / "cal7" / "predictions.parquet")
    texts = g4arm.read_rows(w / "cal7" / "texts.parquet")
    preds += [{**r, "model": r["model"].replace("G4-0/", "G4-4/"), "p_original": 1 - r["p_original"]}
              for r in preds if r["model"].startswith("G4-0/")]
    texts += [{**r, "model": r["model"].replace("G4-0/", "G4-4/")} for r in texts if r["model"].startswith("G4-0/")]
    for r in [r for r in preds if r["model"] == "G4-0"]:
        preds.append({**r, "model": "calib:sampling:G4-4", "p_original": p_greedy})
        texts.append({"model": "calib:sampling:G4-4", "set": r["set"], "id": r["id"], "text": greedy[r["id"]]["hypothesis"],
                      "tokens": 30})
    g4arm.write_parquet(preds, w / "cal7" / "predictions.parquet")
    g4arm.write_parquet(texts, w / "cal7" / "texts.parquet")


def g44_args(w, out, **extra):
    return select_args(w, out, model="G4-4", sampling=w / "sampling" / "G4-4", best8=w / "sampling/G4-4/best8/choice.parquet",
                       probe_report=w / "probe-report.json", **extra)


def test_select_of_g4_4_reuses_the_probe_and_takes_c2_of_its_greedy_translation_from_the_calibration_rows(
        tmp_path, monkeypatch, capsys):
    w = make_world(tmp_path, monkeypatch)
    add_g4_4(w)
    out = tmp_path / "sel-g44"
    g4bon.main(g44_args(w, out, c2_greedy="calib:sampling:G4-4", greedy=w / "eval" / "G4-4"))
    assert '"model": "G4-4"' in capsys.readouterr().out
    choice = {r["id"]: r for r in g4arm.read_rows(out / "choice.parquet")}
    ids = g4bon.search_ids(g4bon.load_evalsets(w / "evalsets.parquet"), g4bon.load_split(w / "split.json"))
    assert set(choice) == set(ids["wmt24pp"] + ids["flan"])
    for r in choice.values():
        assert r["greedy_logit"] == pytest.approx(math.log(0.7 / 0.3))       # calib:sampling:G4-4, not G4-0's 0.2
        assert r["qe0"] == 2.0                                                # the anchor stays G4-0's greedy translation
    stats = json.loads((out / "select.json").read_text())
    assert stats["model"] == "G4-4" and stats["c2_greedy"] == "calib:sampling:G4-4"
    assert stats["best8_checked"] and stats["greedy_texts_checked"]
    # the samples' C2 is G4-4's (1 - p of G4-0's), so the C2 choices may differ from G4-0's; QE's equals best8 (checked)
    for i, r in choice.items():
        k = g4bon.SAMPLE_NAMES.index(r["qe_choice"])
        assert r["qe"][k] == min(r["qe"])
    # without the calibration name G4-4's greedy translation has no C2; the name without the evaluation is refused
    with pytest.raises(SystemExit, match="C2 scored"):
        g4bon.main(g44_args(w, tmp_path / "s1"))
    with pytest.raises(SystemExit, match="--greedy"):
        g4bon.main(g44_args(w, tmp_path / "s2", c2_greedy="calib:sampling:G4-4"))
    # a text of the calibration rows that is not G4-4's greedy translation: refused
    texts = g4arm.read_rows(w / "cal7" / "texts.parquet")
    k = next(n for n, r in enumerate(texts) if r["model"] == "calib:sampling:G4-4" and r["id"] in choice)
    texts[k] = {**texts[k], "text": "another text"}
    g4arm.write_parquet(texts, w / "cal7" / "texts.parquet")
    with pytest.raises(SystemExit, match="not the greedy translations at 1 of the"):
        g4bon.main(g44_args(w, tmp_path / "s3", c2_greedy="calib:sampling:G4-4", greedy=w / "eval" / "G4-4"))


def test_report_puts_g4_4_beside_g4_0_with_its_checks_against_the_probe(tmp_path, monkeypatch, capsys):
    w = make_world(tmp_path, monkeypatch)
    add_g4_4(w)
    g4bon.main(select_args(w, tmp_path / "sel-g40"))
    g4bon.main(g44_args(w, tmp_path / "sel-g44", c2_greedy="calib:sampling:G4-4", greedy=w / "eval" / "G4-4"))
    shares = {("G4-0", "greedy"): 0.2, ("G4-4", "greedy"): 0.4, ("G4-4", "c2"): 0.6}
    c1, _ = c1_predictions(w, tmp_path, ("G4-0", "G4-4"), lambda m, o, s: shares.get((m, o), 0.3))
    # the probe's classify run: G4-0 and G4-4 greedy, t10s1 and best8 (= qe here), with one G4-4/best8 label flipped
    rows = [x for x in g4arm.read_rows(c1) if x["model"].split("/")[-1] in ("G4-0", "G4-4", "t10s1", "qe")]
    probe = [{**x, "model": x["model"].replace("/qe", "/best8")} for x in rows]
    k = next(n for n, x in enumerate(probe) if x["model"] == "G4-4/best8")
    probe[k] = {**probe[k], "p_original": 1 - probe[k]["p_original"]}
    g4arm.write_parquet(probe, tmp_path / "probe-c1.parquet")
    g4arm.write_parquet([{"id": i, "qe_source_reference": 2.5} for i in ids_of("wmt24pp", 2 * N_W + 2)],
                        tmp_path / "refqe.parquet")
    out = tmp_path / "rep" / "report.md"
    out.parent.mkdir()
    g4bon.main(["report", "--evalsets", str(w / "evalsets.parquet"), "--split", str(w / "split.json"),
                "--refqe", str(tmp_path / "refqe.parquet"), "--anchor", str(w / "eval" / "G4-0"),
                "--predictions", str(c1), "--probe-predictions", str(tmp_path / "probe-c1.parquet"),
                "--model", f"G4-0={w}/eval/G4-0,{w}/sampling/G4-0/t10s1,{tmp_path}/sel-g40",
                "--model", f"G4-4={w}/eval/G4-4,{w}/sampling/G4-4/t10s1,{tmp_path}/sel-g44", "--out", str(out)])
    assert "BONREPORT" in capsys.readouterr().out
    r = json.loads(out.with_suffix(".json").read_text())
    n = N_W + N_F
    g = r["models"]["G4-4"]["outputs"]
    # 0.4: 3 of 6 WMT24++ and 2 of 5 FLAN original; G4-0 greedy 0.2: 2 and 1; c2 0.6: 4 and 3
    assert g["greedy"]["c1_pooled"] == pytest.approx(5 / n)
    assert g["greedy"]["c1_vs_g40"]["delta"] == pytest.approx(2 / n)
    assert g["c2"]["c1_vs_greedy"]["delta"] == pytest.approx(2 / n) and g["c2"]["c1_vs_g40"]["delta"] == pytest.approx(4 / n)
    # C2 of G4-4's greedy translation is the calibration rows' (p 0.7: all original)
    assert g["greedy"]["c2_pooled"] == 1.0
    for o in g4bon.OUTPUTS[1:]:
        assert g[o]["hack_flag"] == (g[o]["c2_vs_greedy"] >= 0.10 and g[o]["c1_vs_greedy"]["delta"] < 0.10)
    assert set(r["models"]["G4-4"]["contrasts"]) == set(r["models"]["G4-0"]["contrasts"])
    checks = r["checks"]["c1_against_probe"]
    assert set(checks) == {f"{m}{o} ~ probe {m}{p}" for m in ("G4-0", "G4-4")
                           for o, p in (("/t10s1", "/t10s1"), ("/qe", "/best8"), ("", ""))}
    assert checks["G4-4/qe ~ probe G4-4/best8"]["flipped"] == 1 and checks["G4-0/qe ~ probe G4-0/best8"]["flipped"] == 0
    assert not any(k.startswith("c2_") for k in r["checks"])
    md = out.read_text()
    assert "| G4-4 | greedy |" in md and "| G4-4 | c2-qe |" in md
    assert "- C2 van de samples van G4-0 en G4-4 komt uit c2cal7." in md
    assert "- C2 van G4-4 greedy komt uit de rijen `calib:sampling:G4-4` van c2cal7" in md
    assert md.endswith("zoekdeel.\n\n") or md.rstrip().endswith("zoekdeel.")


# ---------------------------------------------------------------------------------------------------- run_g4arm.sh

def bon_world(tmp_path):
    """test_en_nl_g4arm7's world for ARM=G4-bon: what the stages of the measurement read in the fake bucket"""
    import test_en_nl_g4arm7 as t7

    run, shm, home, g = t7.g47_world(tmp_path)
    files = {"bon/data/classes-w0.txt": "tiny small\n", "rl/G4-7/c2/cal7/predictions.parquet": "x",
             "rl/G4-7/c2/cal7/texts.parquet": "x", "bon/G4-7-i5/c2/predictions.parquet": "x",
             "bon/G4-7-i5/c2/texts.parquet": "x", "bon/naturalness/G4-0_G4-7-i5/texts.parquet": "x"}
    for rel, body in files.items():
        (g / rel).parent.mkdir(parents=True, exist_ok=True)
        (g / rel).write_text(body)
    return run, shm, home, g


def test_bon_stages_are_g4_bons_only(tmp_path):
    run, shm, home, g = bon_world(tmp_path)
    for stage in ("translate", "score", "texts", "classify", "report7", "train", "sample-translate"):
        out, calls = run(stage, ARM="G4-bon", MODEL="G4-7-i5")
        msg = "report7 is G4-7's" if stage == "report7" else "is no stage of G4-bon"
        assert out.returncode != 0 and msg in out.stdout, (stage, out.stdout + out.stderr)
        assert not any(c.startswith(("SETSID", "gsutil")) for c in calls), stage
    for arm in ("G4-7", "G4-5"):
        out, _ = run("bon-select", ARM=arm, MODEL="G4-0")
        assert out.returncode != 0 and ("bon-select is G4-bon's" in out.stdout or "is no stage of G4-7" in out.stdout)


def test_bon_translate_samples_a_candidate_as_the_probe(tmp_path):
    run, shm, home, g = bon_world(tmp_path)
    out, calls = run("bon-translate", ARM="G4-bon", MODEL="G4-7-i5", TAG="b1")
    assert out.returncode == 0 and "started g4b_translate_G4-7-i5_b1 on w0" in out.stdout, out.stdout + out.stderr
    body = (home / "job_body_g4b_translate_G4-7-i5_b1.sh").read_text()
    copy = f"{shm}/translator/g4arm/adapter-g4b_translate_G4-7-i5_b1"
    assert f"--lora {copy}" in body
    assert " ".join(f"--sample t10s{k}=1:{k}" for k in range(1, 9)) in body
    assert "/bon/G4-7-i5/{sample}/translations/w0/tiny" in body and "/bon/data/segments-w0.parquet" in body
    assert "--prompt natural-v1" in body and "venv-fork-f4d8615f4" in body
    assert body.index("translate_class ") < body.index(f"rm -rf -- {copy}") < body.index("TRANSLATED")
    for model, msg in (("G4-0", "candidate of G4-7"), ("G4-4", "candidate of G4-7"), ("G4-7-i31", "candidate of G4-7")):
        out, _ = run("bon-translate", ARM="G4-bon", MODEL=model, TAG="b2")
        assert out.returncode != 0 and msg in out.stdout
    out, _ = run("bon-translate", ARM="G4-bon", MODEL="G4-7-i10", TAG="b3")
    assert out.returncode != 0 and "not rank 32, alpha 16" in out.stdout
    assert not (shm / "translator/g4arm/adapter-g4b_translate_G4-7-i10_b3").exists()
    out, _ = run("bon-score", ARM="G4-bon", MODEL="G4-7-i5", TAG="b4")
    assert out.returncode == 0, out.stdout + out.stderr
    body = (home / "job_body_g4b_score_G4-7-i5_b4.sh").read_text()
    for x in ("evalsets.py collect", "metricx_qe.py score", "comet_score.py score", "/bon/G4-7-i5/t10s8/scores/metricx-w0.parquet",
              "g4arm.py samplejoin", "--pair source:hypothesis:reference --pair source:hypothesis"):
        assert x in body, x


def test_bon_c2_select_texts_classify_report(tmp_path):
    run, shm, home, g = bon_world(tmp_path)
    out, _ = run("bon-c2texts", ARM="G4-bon", MODEL="G4-7-i5", TAG="c1")
    assert out.returncode == 0, out.stdout + out.stderr
    body = (home / "job_body_g4b_c2texts_G4-7-i5_c1.sh").read_text()
    assert "g4bon.py c2texts --model G4-7-i5" in body and "/eval/G4-0/pairs/pairs-w*.parquet" in body
    assert "rl/G4-7/c2/model/config.json" in body and f"{home}/mbert-dumb/.venv/bin/python -I -u" in body
    out, _ = run("bon-c2", ARM="G4-bon", MODEL="G4-7-i5", TAG="c2")
    assert out.returncode == 0, out.stdout + out.stderr
    body = (home / "job_body_g4b_c2_G4-7-i5_c2.sh").read_text()
    assert "--max-length 1024" in body and "/bon/G4-7-i5/c2/predictions.parquet" in body
    out, _ = run("bon-c2", ARM="G4-bon", MODEL="G4-0", TAG="c3")
    assert out.returncode != 0 and "candidate of G4-7" in out.stdout
    # G4-0: the probe's samples, its best8 and report, C2 of c2cal7
    out, _ = run("bon-select", ARM="G4-bon", MODEL="G4-0", TAG="s1")
    assert out.returncode == 0, out.stdout + out.stderr
    body = (home / "job_body_g4b_select_G4-0_s1.sh").read_text()
    for x in ("/sampling/G4-0/t10s8", "/sampling/G4-0/best8/choice.parquet", "/sampling/report/report.json",
              "rl/G4-7/c2/cal7/predictions.parquet", "rl/G4-7/c2/cal7/texts.parquet", "--best8", "--probe-report",
              "/eval/G4-0 ", "/bon/G4-0/select"):
        assert x in body, x
    out, _ = run("bon-select", ARM="G4-bon", MODEL="G4-7-i5", TAG="s2")
    assert out.returncode == 0, out.stdout + out.stderr
    body = (home / "job_body_g4b_select_G4-7-i5_s2.sh").read_text()
    assert "/bon/G4-7-i5/t10s1" in body and "/bon/G4-7-i5/c2/texts.parquet" in body and "--best8" not in body
    (g / "bon/G4-7-i5/c2/texts.parquet").unlink()
    out, _ = run("bon-select", ARM="G4-bon", MODEL="G4-7-i5", TAG="s3")
    assert out.returncode != 0 and "C2's scores of G4-7-i5" in out.stdout
    # G4-bon.10 allows G4-4 after G4-0 (test_bon_select_and_texts_of_g4_4); another model is refused
    for models in ("G4-7-i5 G4-0", "G4-0 G4-5", "G4-4 G4-0"):
        out, _ = run("bon-texts", ARM="G4-bon", KEY="k", MODELS=models, TAG="t0")
        assert out.returncode != 0
    out, _ = run("bon-texts", ARM="G4-bon", KEY="G4-0_G4-7-i5", MODELS="G4-0 G4-7-i5", TAG="t1")
    assert out.returncode == 0, out.stdout + out.stderr
    body = (home / "job_body_g4b_texts_t1.sh").read_text()
    for x in ("'G4-0=", "'G4-0/t10s1=", "'G4-7-i5/c2q=", "/sampling/G4-0/t10s1", "/bon/G4-7-i5/t10s1",
              "/bon/naturalness/G4-0_G4-7-i5/texts.parquet"):
        assert x in body, x
    out, _ = run("bon-classify", ARM="G4-bon", KEY="G4-0_G4-7-i5", MODELS="G4-0 G4-7-i5", TAG="t2")
    assert out.returncode == 0 and "started arms_classify_t2b" in out.stdout, out.stdout + out.stderr
    body = (home / "job_body_arms_classify_t2b.sh").read_text()
    assert "/bon/naturalness/G4-0_G4-7-i5/texts.parquet" in body and "sft/arms/naturalness/classifier/params.pkl" in body
    out, _ = run("bon-report", ARM="G4-bon", KEY="G4-0_G4-7-i5", MODELS="G4-0 G4-7-i5", TAG="t3")
    assert out.returncode == 0, out.stdout + out.stderr
    body = (home / "job_body_g4b_report_t3.sh").read_text()
    for x in ("g4bon.py report", "--c2-check G4-7-i5=", "/sampling/naturalness/predictions.parquet",
              "/data/G4-5-split/refqe.parquet", "/bon/report/G4-0_G4-7-i5/", "--model G4-0="):
        assert x in body, x


def test_bon_select_and_texts_of_g4_4(tmp_path):
    """G4-bon.10: G4-4 reuses the probe as G4-0 does (its samples, best8, C2 of c2cal7 with its greedy translation's
    calibration name, checked against its evaluation); no new samples, no C2 run of its own"""
    run, shm, home, g = bon_world(tmp_path)
    out, _ = run("bon-select", ARM="G4-bon", MODEL="G4-4", TAG="s4")
    assert out.returncode == 0 and "started g4b_select_G4-4_s4 on w0" in out.stdout, out.stdout + out.stderr
    body = (home / "job_body_g4b_select_G4-4_s4.sh").read_text()
    for x in ("/sampling/G4-4/t10s1 ", "/sampling/G4-4/t10s8 ", "/sampling/G4-4/best8/choice.parquet",
              "/sampling/report/report.json", "rl/G4-7/c2/cal7/predictions.parquet", "rl/G4-7/c2/cal7/texts.parquet",
              "--best8", "--probe-report", "--c2-greedy calib:sampling:G4-4", "/eval/G4-0 ", "/bon/G4-4/select",
              "g4bon.py select --model G4-4 "):
        assert x in body, x
    greedy = f"{shm}/translator/g4arm/bonselect-G4-4-s4/greedy"
    assert f"--greedy {greedy} " in body and "rsync -r -x 'translations/.*' " in body and f"/eval/G4-4 {greedy}" in body
    assert "/sampling/G4-0/" not in body and "/bon/G4-4/c2/" not in body
    for stage in ("bon-translate", "bon-score", "bon-c2texts", "bon-c2"):
        out, _ = run(stage, ARM="G4-bon", MODEL="G4-4", TAG="s5")
        assert out.returncode != 0 and "candidate of G4-7" in out.stdout, stage
    (g / "bon/naturalness/G4-0_G4-4").mkdir(parents=True)
    (g / "bon/naturalness/G4-0_G4-4/texts.parquet").write_text("x")
    out, _ = run("bon-texts", ARM="G4-bon", KEY="G4-0_G4-4", MODELS="G4-0 G4-4", TAG="t4")
    assert out.returncode == 0, out.stdout + out.stderr
    body = (home / "job_body_g4b_texts_t4.sh").read_text()
    for x in ("'G4-4=", "'G4-4/t10s1=", "'G4-4/c2=", "/sampling/G4-4/t10s1 ", "/eval/G4-4 ", "/bon/G4-4/select ",
              "/bon/naturalness/G4-0_G4-4/texts.parquet"):
        assert x in body, x
    assert "/bon/G4-4/t10s1" not in body
    out, _ = run("bon-report", ARM="G4-bon", KEY="G4-0_G4-4", MODELS="G4-0 G4-4", TAG="t5")
    assert out.returncode == 0, out.stdout + out.stderr
    body = (home / "job_body_g4b_report_t5.sh").read_text()
    assert "--model G4-4=" in body and "--c2-check" not in body and "/bon/report/G4-0_G4-4/" in body
