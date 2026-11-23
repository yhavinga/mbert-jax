"""The measurement G4-judge (protocol-g4.md §17): pairjudge.py (the prompt as the protocol has it, the answer tokens,
the combination of both orders, the validation's measures and thresholds, the pairs of step 2, the selections judge and
judgeq, the report in G4-bon's form, the blind package) and run_pairjudge.sh's stages, on synthetic inputs (no model,
no TPU)."""

import glob
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "experiments" / "en-nl-translator"
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(ROOT / "experiments" / "flan-nl"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import g4arm  # noqa: E402
import g4bon  # noqa: E402
import pairjudge as pj  # noqa: E402
import test_en_nl_g4bon as tb  # noqa: E402

TOKENIZER = glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--google--gemma-4-31B-it/snapshots/842da3794eaa0b77d5f08bae87a17459d91ff475/tokenizer.json"))


# ---------------------------------------------------------------------------------------------------- the prompt

def test_the_prompt_is_the_protocols_text_letter_for_letter():
    text = (CODE / "protocol-g4.md").read_text(encoding="utf-8")
    section = text[text.index("## 17. Meting G4-judge"):]
    block = re.search(r"\*G4-judge\.3 Prompt\*.*?```\n(.*?)\n```", section, re.S).group(1)
    assert block.replace("{bron}", "{bron}") == pj.PROMPT
    filled = pj.user_text("  The source.\n", "\nDe A. ", " De B.")
    assert "Engelse bron:\nThe source.\n\nVertaling A:\nDe A.\n\nVertaling B:\nDe B.\n\nWelke vertaling" in filled
    assert filled.startswith("Je beoordeelt") and filled.endswith("Antwoord met één woord: A, B of gelijk.")
    # a text with braces or a placeholder's name is filled as it is
    assert "{A}" in pj.user_text("x {A} y", "a", "b").split("Vertaling A:")[0]


def test_both_orders_swap_the_translations():
    (u1, t1, o1), (u2, t2, o2) = pj.pair_prompts("p/b1", "src", "first", "second")
    assert (u1, o1, u2, o2) == ("p/b1/ab", "ab", "p/b1/ba", "ba")
    assert t1.index("Vertaling A:\nfirst") < t1.index("Vertaling B:\nsecond")
    assert t2.index("Vertaling A:\nsecond") < t2.index("Vertaling B:\nfirst")


@pytest.mark.skipif(not TOKENIZER, reason="no Gemma 4 tokenizer in the local Hugging Face cache")
def test_answer_ids_are_the_tokenizers():
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(TOKENIZER[0])
    for answer, tokens in pj.ANSWER_TOKENS.items():
        assert tuple(tok.id_to_token(i) for i in pj.ANSWER_IDS[answer]) == tokens
    assert tok.encode("gelijk", add_special_tokens=False).ids == [161224]
    assert tok.encode("<channel|>A", add_special_tokens=False).ids[-1] == 236776


def test_score_logits_takes_the_three_sets_and_the_top_tokens():
    import numpy as np

    logits = np.full(262144, -5.0, np.float32)
    logits[236776], logits[562], logits[236799], logits[161224], logits[7] = 3.0, 1.0, 2.0, 0.5, 4.0
    s = pj.score_logits(logits)
    lse = math.log(math.exp(3) + math.exp(1) + math.exp(2) + math.exp(0.5) + math.exp(4) + (262144 - 5) * math.exp(-5))
    assert s["logp_a"] == pytest.approx(math.log(math.exp(3) + math.exp(1)) - lse)
    assert s["logp_b"] == pytest.approx(math.log(math.exp(2) + math.exp(-5)) - lse)
    assert s["top_ids"][:2] == [7, 236776]
    v = pj.answer_view(s["logp_a"], s["logp_b"], s["logp_eq"])
    assert v["p_a"] + v["p_b"] + v["p_eq"] == pytest.approx(1.0) and 0 < v["mass"] < 1


# ---------------------------------------------------------------------------------------------------- the combination

def view(pa, pb, pe):
    return {"p_a": pa, "p_b": pb, "p_eq": pe, "skipped": None}


def test_combination_of_both_orders_removes_a_constant_position_preference():
    # the judge prefers the first position by 0.2 in both orders; the content favours the package's A
    ab, ba = view(0.6, 0.3, 0.1), view(0.45, 0.45, 0.1)
    p = pj.combine(ab, ba)
    assert p == pytest.approx({"A": 0.525, "B": 0.375, "gelijk": 0.1})
    assert pj.verdict_of(p) == "A" and pj.direction_credit(p, "A") == 1.0 and pj.direction_credit(p, "B") == 0.0
    assert pj.greedy_of(ab, "ab") == "A" and pj.greedy_of(ba, "ba") == "gelijk"
    # the same position chosen twice: opposite winners in the package's frame
    assert pj.greedy_of(view(0.7, 0.2, 0.1), "ba") == "B"
    assert pj.verdict_of({"A": 0.4, "B": 0.4, "gelijk": 0.2}) == "gelijk"
    assert pj.verdict_of({"A": 0.4, "B": 0.2, "gelijk": 0.4}) == "gelijk"
    assert pj.direction_credit({"A": 0.4, "B": 0.4, "gelijk": 0.2}, "A") == 0.5
    assert pj.combine(ab, {**ba, "skipped": "too_long"}) is None and pj.combine(None, ba) is None


def test_kappa_and_wilson():
    k, table = pj.kappa(["A", "A", "B", "gelijk"], ["A", "B", "B", "gelijk"])
    assert table["A"]["B"] == 1 and table["gelijk"]["gelijk"] == 1
    po, pe = 3 / 4, (2 * 1 + 1 * 2 + 1 * 1) / 16
    assert k == pytest.approx((po - pe) / (1 - pe))
    lo, hi = pj.wilson(70, 100)
    assert lo == pytest.approx(0.6041, abs=1e-3) and hi == pytest.approx(0.7811, abs=1e-3)
    assert pj.share([]) == {"n": 0, "share": None, "lo": None, "hi": None}


# ---------------------------------------------------------------------------------------------------- step 1

MARK = "QQTEXTQQ"                                    # in every synthetic text: no report may hold it


def make_packages(tmp_path, seed=3):
    """Five synthetic packages in the bucket's layout; Claude's labels cycle so that every kind of pair is there"""
    rnd = random.Random(seed)
    d = tmp_path / "blind"
    verdicts = ["A beter", "B beter", "gelijk", "A beter", "B beter"]
    for p, review in pj.PACKAGES.items():
        inputs, keys, judg, final = [], [], [], []
        for n in range(1, 101):
            b = f"blind-{n:03d}"
            inputs.append({"blind_id": b, "source": f"{MARK} source {p} {b}", "A": f"{MARK} A {b}", "B": f"{MARK} B {b}"})
            base_first = rnd.random() < 0.5
            keys.append({"blind_id": b, "A": "base" if base_first else "X", "B": "X" if base_first else "base",
                         "set": ("flan", "wmt24pp", "idiom")[n % 3], "item": f"i{n}"})
            v = verdicts[n % 5]
            judg.append({"blind_id": b, "fout_A": "ja" if n % 7 == 0 else "nee", "fout_B": "ja" if n % 11 == 0 else "nee",
                         "oordeel": v, "zekerheid": "zeker" if n % 2 else "twijfel", "uitleg": MARK})
            final.append({"blind_id": b, "oordeel": "gelijk" if n % 10 == 4 else v, "fout_A": "nee", "fout_B": "nee",
                          "uitleg": MARK, "zekerheid": "x"})
        (d / p / Path(review).parent).mkdir(parents=True, exist_ok=True)
        for name, rows in (("judge-input.jsonl", inputs), ("key.jsonl", keys), (review, judg)):
            (d / p / name).write_text("".join(json.dumps(r) + "\n" for r in rows))
        if p in pj.FINAL:
            (d / p / pj.FINAL[p]).write_text("".join(json.dumps(r) + "\n" for r in final))
    return d


def expected_counts(d):
    refs = []
    for p, review in pj.PACKAGES.items():
        for line in (d / p / review).read_text().splitlines():
            r = json.loads(line)
            refs.append((p, r["oordeel"], r["zekerheid"]))
    return {"pairs": len(refs), "noneq": sum(v != "gelijk" for _, v, _ in refs),
            "noneq_sure": sum(v != "gelijk" and z == "zeker" for _, v, z in refs),
            "close_noneq": sum(v != "gelijk" and p in pj.CLOSE for p, v, _ in refs)}


def fake_val_scores(data, out, agree_every=4, bias=0.15):
    """A judge that follows Claude's winner except at every agree_every-th pair, with a first-position bias"""
    refs = {f"{r['package']}/{r['blind_id']}": r for r in pj.read_rows(data / "refs.parquet")}
    rows = []
    for n, (key, r) in enumerate(sorted(refs.items())):
        winner = r["claude"] if n % agree_every else {"A": "B", "B": "A", "gelijk": "A"}[r["claude"]]
        content = {"A": (0.6, 0.2, 0.2), "B": (0.2, 0.6, 0.2), "gelijk": (0.25, 0.25, 0.5)}[winner]
        for order in pj.ORDERS:
            pa, pb, pe = content if order == "ab" else (content[1], content[0], content[2])
            pa, pb = pa + bias, pb - bias
            rows.append({"uid": f"{key}/{order}", "p_a": pa, "p_b": max(pb, 0.0), "p_eq": pe, "mass": 0.9,
                         "skipped": None, "top_ids": [236776], "prompt_tokens": 100})
    pj.write_rows(rows, out / "shard-00000.parquet")
    return refs


def test_valdata_refuses_other_counts_and_writes_no_text_beside_the_prompts(tmp_path, monkeypatch, capsys):
    d = make_packages(tmp_path)
    with pytest.raises(SystemExit, match="protocol §17 has"):
        pj.main(["valdata", "--blind", str(d), "--out", str(tmp_path / "x")])
    monkeypatch.setattr(pj, "EXPECTED", expected_counts(d))
    out = tmp_path / "data"
    pj.main(["valdata", "--blind", str(d), "--out", str(out)])
    assert "VALDATA" in capsys.readouterr().out
    prompts = [r for k in range(4) for r in pj.read_rows(out / f"prompts-w{k}.parquet")]
    assert len(prompts) == 1000 and len({r["uid"] for r in prompts}) == 1000
    for k in range(4):
        assert all(pj.worker_of(r["uid"]) == k for r in pj.read_rows(out / f"prompts-w{k}.parquet"))
    ab = next(r for r in prompts if r["uid"] == "G4-7/blind-002/ab")["text"]
    ba = next(r for r in prompts if r["uid"] == "G4-7/blind-002/ba")["text"]
    assert ab.index(f"{MARK} A blind-002") < ab.index(f"{MARK} B blind-002")
    assert ba.index(f"{MARK} B blind-002") < ba.index(f"{MARK} A blind-002")
    for f in ("refs.parquet", "valdata.json"):
        assert MARK not in (out / f).read_bytes().decode("latin-1")
    refs = pj.read_rows(out / "refs.parquet")
    assert {r["final"] is not None for r in refs if r["package"] in pj.FINAL} == {True}
    # a package with a missing judgment is refused
    lines = (d / "RB2" / pj.PACKAGES["RB2"]).read_text().splitlines()
    (d / "RB2" / pj.PACKAGES["RB2"]).write_text("\n".join(lines[:-1]) + "\n")
    with pytest.raises(SystemExit, match="RB2"):
        pj.main(["valdata", "--blind", str(d), "--out", str(tmp_path / "y")])


def test_valreport_measures_and_the_thresholds(tmp_path, monkeypatch, capsys):
    d = make_packages(tmp_path)
    monkeypatch.setattr(pj, "EXPECTED", expected_counts(d))
    data = tmp_path / "data"
    pj.main(["valdata", "--blind", str(d), "--out", str(data)])
    refs = fake_val_scores(data, tmp_path / "scores" / "w0")
    pj.main(["valreport", "--refs", str(data / "refs.parquet"), "--scores", str(tmp_path / "scores/w*/shard-*.parquet"),
             "--out", str(tmp_path / "report.md")])
    out = capsys.readouterr().out
    r = json.loads((tmp_path / "report.json").read_text())
    # by hand: the judge's direction is Claude's except at every 4th pair (sorted keys)
    keys = sorted(refs)
    agree = {k: n % 4 != 0 for n, k in enumerate(keys)}
    sure = [k for k in keys if refs[k]["claude"] != "gelijk" and refs[k]["sure"]]
    close = [k for k in keys if refs[k]["claude"] != "gelijk" and refs[k]["package"] in pj.CLOSE]
    assert r["gate"]["D1"]["share"] == pytest.approx(sum(agree[k] for k in sure) / len(sure))
    assert r["gate"]["D2"]["share"] == pytest.approx(sum(agree[k] for k in close) / len(close))
    assert r["gate"]["D1"]["n"] == len(sure) and r["gate"]["D2"]["n"] == len(close)
    # a first-position bias of 0.15 flips the greedy answer where the content gives gelijk (0.25 + 0.15 > 0.5 - no;
    # 0.40 < 0.5) never, and where it gives B (0.2 + 0.15 < 0.6 - 0.15) never: no flips
    assert r["gate"]["D3"]["share"] == 0.0
    assert r["gate"]["pass"] == (r["gate"]["D1"]["share"] >= 0.7 and r["gate"]["D2"]["share"] >= 0.6)
    assert f"GATE {'pass' if r['gate']['pass'] else 'fail'}" in out
    assert r["position"]["ab"]["mean_pa_minus_pb"] > r["position"]["ba"]["mean_pa_minus_pb"] - 1e-9
    md = (tmp_path / "report.md").read_text()
    assert MARK not in md and MARK not in (tmp_path / "report.json").read_text()
    assert "## Drempel voor stap 2" in md and "| D1 |" in md and "Tabel 5" in md
    assert set(r["final"]) == set(pj.FINAL) and set(r["packages"]) == set(pj.PACKAGES)
    # a strong position bias: the same position twice is a flip
    for f in (tmp_path / "scores" / "w0").glob("*.parquet"):
        f.unlink()
    fake_val_scores(data, tmp_path / "scores" / "w0", bias=0.45)
    pj.main(["valreport", "--refs", str(data / "refs.parquet"), "--scores", str(tmp_path / "scores/w*/shard-*.parquet"),
             "--out", str(tmp_path / "report2.md")])
    r2 = json.loads((tmp_path / "report2.json").read_text())
    assert r2["gate"]["D3"]["share"] > 0.2 and not r2["gate"]["pass"] and "GATE fail" in capsys.readouterr().out


def test_valreport_refuses_missing_scores(tmp_path, monkeypatch):
    d = make_packages(tmp_path)
    monkeypatch.setattr(pj, "EXPECTED", expected_counts(d))
    data = tmp_path / "data"
    pj.main(["valdata", "--blind", str(d), "--out", str(data)])
    fake_val_scores(data, tmp_path / "scores" / "w0")
    rows = pj.read_rows(tmp_path / "scores/w0/shard-00000.parquet")
    pj.write_rows(rows[:-1], tmp_path / "scores/w0/shard-00000.parquet")
    with pytest.raises(SystemExit, match="scores for"):
        pj.main(["valreport", "--refs", str(data / "refs.parquet"), "--scores", str(tmp_path / "scores/w*/shard-*.parquet"),
                 "--out", str(tmp_path / "r.md")])


# ---------------------------------------------------------------------------------------------------- step 2

def bon_setup(tmp_path, monkeypatch):
    """test_en_nl_g4bon's world; G4-bon's select of G4-0 and of G4-7-i15 (its samples G4-0's with other C2 scores)"""
    w = tb.make_world(tmp_path, monkeypatch)
    g4bon.main(tb.select_args(w, tmp_path / "bon-G4-0"))
    shutil.copytree(w / "eval" / "G4-0", w / "eval" / "G4-7-i15")
    shutil.copytree(w / "sampling" / "G4-0", w / "bon" / "G4-7-i15")
    preds = g4arm.read_rows(w / "cal7" / "predictions.parquet")
    texts = g4arm.read_rows(w / "cal7" / "texts.parquet")
    p2 = [{**r, "model": r["model"].replace("G4-0", "G4-7-i15"), "p_original": min(0.999, r["p_original"] * 1.5)}
          for r in preds] + [r for r in preds if r["model"] == "G4-0"]
    t2 = [{**r, "model": r["model"].replace("G4-0", "G4-7-i15")} for r in texts] + [r for r in texts if r["model"] == "G4-0"]
    g4arm.write_parquet(p2, w / "c2-i15" / "predictions.parquet")
    g4arm.write_parquet(t2, w / "c2-i15" / "texts.parquet")
    g4bon.main(tb.select_args(w, tmp_path / "bon-G4-7-i15", model="G4-7-i15", sampling=w / "bon" / "G4-7-i15",
                              c2=w / "c2-i15"))
    return w


def samples_of(w, m):
    return w / "sampling" / "G4-0" if m == "G4-0" else w / "bon" / "G4-7-i15"


def fake_bon_scores(data, out):
    """P̄(sample better) from the sample's number and the source: a deterministic judge without position bias"""
    rows = []
    for k in range(4):
        for r in pj.read_rows(data / f"prompts-w{k}.parquet"):
            model, i, name, order = r["uid"].rsplit("/", 3)
            s = (int(name[-1]) * 7 + sum(map(ord, i))) % 10 / 20 + 0.2      # 0.2 .. 0.65
            first, second = (s, 0.8 - s) if order == "ab" else (0.8 - s, s)
            rows.append({"uid": r["uid"], "p_a": first, "p_b": second, "p_eq": 0.2, "mass": 0.95, "skipped": None,
                         "top_ids": [236776], "prompt_tokens": 50})
    pj.write_rows(rows, out / "shard-00000.parquet")


def test_bonpairs_select_report_blind(tmp_path, monkeypatch, capsys):
    w = bon_setup(tmp_path, monkeypatch)
    ev, sp = str(w / "evalsets.parquet"), str(w / "split.json")
    data = tmp_path / "bondata"
    margs = []
    for m in pj.BON_MODELS:
        margs += ["--model", f"{m}={samples_of(w, m)},{tmp_path / f'bon-{m}' / 'choice.parquet'}"]
    pj.main(["bonpairs", *margs, "--anchor", str(w / "eval" / "G4-0"), "--evalsets", ev, "--split", sp, "--out", str(data)])
    info = json.loads((data / "bonpairs.json").read_text())
    prompts = [r for k in range(4) for r in pj.read_rows(data / f"prompts-w{k}.parquet")]
    for m in pj.BON_MODELS:
        choice = g4arm.read_rows(tmp_path / f"bon-{m}" / "choice.parquet")
        nonfail = sum(not f for r in choice for f in r["fail"])
        assert sum(r["uid"].startswith(f"{m}/") for r in prompts) == 2 * nonfail
        assert info["models"][m]["pairs_wmt24pp"] + info["models"][m]["pairs_flan"] == nonfail
        fails = {(r["id"], pj.SAMPLE_NAMES[k]) for r in choice for k, f in enumerate(r["fail"]) if f}
        assert not any(f"{m}/{i}/{n}/ab" == p["uid"] for i, n in fails for p in prompts)
    fake_bon_scores(data, tmp_path / "bonscores" / "w0")
    for m in pj.BON_MODELS:
        pj.main(["select", "--model", m, "--samples", str(samples_of(w, m)), "--choice",
                 str(tmp_path / f"bon-{m}" / "choice.parquet"), "--scores", str(tmp_path / "bonscores/w*/shard-*.parquet"),
                 "--evalsets", ev, "--split", sp, "--out", str(tmp_path / f"j-{m}")])
        choice = {r["id"]: r for r in g4arm.read_rows(tmp_path / f"j-{m}" / "choice.parquet")}
        for i, r in choice.items():
            ok = [k for k in range(8) if not r["fail"][k]]
            bound = [k for k in ok if r["qe"][k] <= r["qe0"]]
            k = pj.SAMPLE_NAMES.index(r["judge_choice"])
            if ok:
                assert not r["judge_fallback"] and r["p_sample"][k] == max(r["p_sample"][j] for j in ok)
                assert k == min(ok, key=lambda j: (-r["p_sample"][j], r["qe"][j], j))
            else:
                assert r["judge_fallback"] and r["judge_choice"] == r["qe_choice"]
            kq = pj.SAMPLE_NAMES.index(r["judgeq_choice"])
            if bound:
                assert kq in bound and r["p_sample"][kq] == max(r["p_sample"][j] for j in bound)
            else:
                assert r["judgeq_fallback"] and r["judgeq_choice"] == r["qe_choice"]
            assert all((r["p_sample"][j] is None) == bool(r["fail"][j]) for j in range(8))
        for sel in pj.JUDGE_SELECTIONS:
            scored = __import__("armeval").scored(tmp_path / f"j-{m}" / sel)
            assert set(scored) == set(choice)
    assert "JSELECT" in capsys.readouterr().out
    # C1's texts of the 12 outputs, as run_pairjudge.sh texts asks g4bon.py
    targs = []
    for m in pj.BON_MODELS:
        targs += ["--pairs", f"{m}={w / 'eval' / m}/pairs/pairs-w*.parquet",
                  "--pairs", f"{m}/t10s1={samples_of(w, m) / 't10s1'}/pairs/pairs-w*.parquet",
                  "--pairs", f"{m}/qe={tmp_path / f'bon-{m}'}/qe/pairs/pairs-w*.parquet",
                  "--pairs", f"{m}/c2q={tmp_path / f'bon-{m}'}/c2q/pairs/pairs-w*.parquet",
                  "--pairs", f"{m}/judge={tmp_path / f'j-{m}'}/judge/pairs/pairs-w*.parquet",
                  "--pairs", f"{m}/judgeq={tmp_path / f'j-{m}'}/judgeq/pairs/pairs-w*.parquet"]
    g4bon.main(["texts", "--evalsets", ev, "--split", sp, *targs, "--out", str(tmp_path / "texts.parquet")])
    texts = g4arm.read_rows(tmp_path / "texts.parquet")
    assert len({r["model"] for r in texts}) == 12
    c1 = [{"model": r["model"], "set": r["set"], "id": r["id"],
           "p_original": 0.9 if (hash(r["id"]) + len(r["model"])) % 3 == 0 else 0.1} for r in texts]
    g4arm.write_parquet(c1, tmp_path / "c1.parquet")
    probe = [{**r, "model": r["model"].replace("G4-0/qe", "G4-0/best8")} for r in c1 if r["model"].startswith("G4-0")]
    g4arm.write_parquet(probe, tmp_path / "probe.parquet")
    g4arm.write_parquet([r for r in c1 if r["model"].startswith("G4-7-i15")], tmp_path / "bonc1.parquet")
    rargs = []
    for m in pj.BON_MODELS:
        rargs += ["--model", f"{m}={w / 'eval' / m},{samples_of(w, m) / 't10s1'},{tmp_path / f'bon-{m}'},{tmp_path / f'j-{m}'}"]
    refqe = [{"id": r["id"], "qe_source_reference": 2.3} for r in g4arm.read_rows(tmp_path / "bon-G4-0/choice.parquet", ["id"])]
    g4arm.write_parquet(refqe, tmp_path / "refqe.parquet")
    pj.main(["bonreport", "--evalsets", ev, "--split", sp, "--refqe", str(tmp_path / "refqe.parquet"), "--anchor",
             str(w / "eval" / "G4-0"), "--predictions", str(tmp_path / "c1.parquet"), "--probe-predictions",
             str(tmp_path / "probe.parquet"), "--bon-predictions", str(tmp_path / "bonc1.parquet"), *rargs,
             "--out", str(tmp_path / "rep.md")])
    rep = json.loads((tmp_path / "rep.json").read_text())
    md = (tmp_path / "rep.md").read_text()
    for m in pj.BON_MODELS:
        x = rep["models"][m]
        assert set(x["outputs"]) == set(pj.OUTPUTS)
        assert {"judge-qe", "judgeq-qe", "judge-c2q", "judgeq-c2q", "judge-greedy"} <= set(x["contrasts"])
        ids = [r["id"] for r in texts if r["model"] == m]
        assert x["outputs"]["greedy"]["c1_pooled"] == pytest.approx(
            sum(r["p_original"] >= 0.5 for r in c1 if r["model"] == m) / len(ids))
    assert all(v["flipped"] == 0 for v in rep["checks"].values()) and len(rep["checks"]) == 7
    assert "## Tabel 3" in md and "judgeq-c2q" in md and "JBONREPORT" in capsys.readouterr().out
    # the blind package: (G4-7-i15, judgeq) against G4-0 greedy where they differ (all do in this world)
    pj.main(["blind", "--model", "G4-7-i15", "--select", str(tmp_path / "j-G4-7-i15"), "--anchor", str(w / "eval" / "G4-0"),
             "--evalsets", ev, "--split", sp, "--out", str(tmp_path / "blind")])
    inputs = [json.loads(x) for x in (tmp_path / "blind" / "judge-input.jsonl").read_text().splitlines()]
    keys = [json.loads(x) for x in (tmp_path / "blind" / "key.jsonl").read_text().splitlines()]
    rep = json.loads((tmp_path / "blind" / "select-report.json").read_text())
    assert len(inputs) == len(keys) == tb.N_W + tb.N_F and [k["blind_id"] for k in keys] == [x["blind_id"] for x in inputs]
    assert [k["set"] for k in keys] == ["wmt24pp"] * tb.N_W + ["flan"] * tb.N_F
    target = __import__("armeval").read_rows(str(tmp_path / "j-G4-7-i15" / "judgeq" / "pairs" / "pairs-w*.parquet"))
    for x, k in zip(inputs, keys):
        assert {k["A"], k["B"]} == {"base", pj.BLIND_TARGET}
        assert (random.Random(k["seed"]).random() < 0.5) == (k["A"] == "base")
        own = target[k["item"]]["hypothesis"]
        assert (x["A"] if k["A"] == pj.BLIND_TARGET else x["B"]) == own
        assert (x["B"] if k["A"] == pj.BLIND_TARGET else x["A"]) == f"greedy {k['item']}"
    assert rep["sets"]["wmt24pp"]["seed"] == "g4judge/blind/wmt24pp" and "greedy" not in json.dumps(rep)


def test_select_refuses_a_fail_with_a_score_and_a_sample_without(tmp_path, monkeypatch):
    w = bon_setup(tmp_path, monkeypatch)
    ev, sp = str(w / "evalsets.parquet"), str(w / "split.json")
    data = tmp_path / "bondata"
    pj.main(["bonpairs", "--model", f"G4-0={samples_of(w, 'G4-0')},{tmp_path / 'bon-G4-0' / 'choice.parquet'}",
             "--anchor", str(w / "eval" / "G4-0"), "--evalsets", ev, "--split", sp, "--out", str(data)])
    fake_bon_scores(data, tmp_path / "s" / "w0")
    rows = pj.read_rows(tmp_path / "s/w0/shard-00000.parquet")
    pj.write_rows(rows[2:], tmp_path / "s/w0/shard-00000.parquet")
    with pytest.raises(SystemExit, match="no fail without a judge score|judge score"):
        pj.main(["select", "--model", "G4-0", "--samples", str(samples_of(w, "G4-0")), "--choice",
                 str(tmp_path / "bon-G4-0" / "choice.parquet"), "--scores", str(tmp_path / "s/w*/shard-*.parquet"),
                 "--evalsets", ev, "--split", sp, "--out", str(tmp_path / "j")])


# ---------------------------------------------------------------------------------------------------- run_pairjudge.sh

def pj_world(tmp_path):
    """run_pairjudge.sh (and run_arms.sh) with /dev/shm under tmp_path, fake gsutil/sudo/hostname, a setsid that starts
    nothing, the fork's venv and tokenizer present"""
    import test_en_nl_g4arm5 as t5

    shm, gs, home, fake, code = (tmp_path / x for x in ("shm", "gs", "home", "bin", "code"))
    for d in (code, home, fake):
        d.mkdir(parents=True)
    for name in ("run_pairjudge.sh", "run_arms.sh"):
        (code / name).write_text((CODE / name).read_text().replace("/dev/shm/", f"{shm}/"))
    for name, body in {"gsutil": t5.FAKE_GSUTIL, "sudo": "#!/bin/bash\nexit 0\n", "hostname": "#!/bin/bash\necho t1v-n-x-w-0\n",
                       "setsid": '#!/bin/bash\necho "SETSID $*" >> "$FAKELOG"\n'}.items():
        (fake / name).write_text(body)
        (fake / name).chmod(0o755)
    fpy = shm / "translate-generic/venv-fork-f4d8615f4/bin/python"
    fpy.parent.mkdir(parents=True)
    fpy.write_text("#!/bin/bash\nexit 0\n")
    fpy.chmod(0o755)
    (shm / "translate-generic/tokenizer-gemma4").mkdir(parents=True)
    (shm / "translate-generic/tokenizer-gemma4/tokenizer.json").write_text("{}")
    g = gs / "t5-dutch-english-usc2/mbert/translator/sft/g4arm"
    calls = tmp_path / "calls.log"

    def run(stage, host="0", **env):
        (fake / "hostname").write_text(f"#!/bin/bash\necho t1v-n-x-w-{host}\n")
        e = {"HOME": str(home), "PATH": f"{fake}:{os.environ['PATH']}", "FAKEGS": str(gs), "FAKELOG": str(calls),
             "TAG": "t1", **env}
        calls.write_text("")
        out = subprocess.run(["bash", str(code / "run_pairjudge.sh"), stage], capture_output=True, text=True, timeout=60,
                             env=e, cwd=home)
        return out, calls.read_text().splitlines()

    return run, shm, home, g


def test_run_pairjudge_stages_render_their_jobs(tmp_path):
    run, shm, home, g = pj_world(tmp_path)
    out, calls = run("valdata", TAG="v1")
    assert out.returncode == 0 and "started pj_valdata_v1 on w0" in out.stdout, out.stdout + out.stderr
    body = (home / "job_body_pj_valdata_v1.sh").read_text()
    for p in pj.PACKAGES:
        assert f"/sft/blind/{p} " in body
    assert "pairjudge.py valdata" in body and "/judge/val/data/" in body
    out, _ = run("valdata", host="1", TAG="v2")
    assert out.returncode != 0 and "runs on w0 only" in out.stdout
    for host in "0123":
        out, calls = run("score", host=host, SET="val", TAG="s1")
        assert out.returncode == 0 and f"started pj_score_val_s1 on w{host}" in out.stdout, out.stdout + out.stderr
        body = (home / "job_body_pj_score_val_s1.sh").read_text()
        assert f"/judge/val/data/prompts-w{host}.parquet" in body and f"/judge/val/scores/w{host} " in body
        assert "judge/gemma-4-31b-it-maxtext/0/items" in body and "TPU_VISIBLE_CHIPS=0,1,2,3" in body
        assert "preflight failed" in body and "venv-fork-f4d8615f4" in body
        (home / "logs" / "pj_score_val_s1.log").unlink(missing_ok=True)
    out, _ = run("score", SET="test", TAG="s2")
    assert out.returncode != 0 and "SET is val or bon" in out.stdout
    out, _ = run("valreport", TAG="r1")
    assert out.returncode == 0, out.stdout + out.stderr
    body = (home / "job_body_pj_valreport_r1.sh").read_text()
    assert all(f"/judge/val/scores/w{k}/shard-*.parquet" in body for k in range(4)) and "pairjudge.py valreport" in body
    out, _ = run("bonpairs", TAG="b1")
    body = (home / "job_body_pj_bonpairs_b1.sh").read_text()
    assert "/sampling/G4-0/t10s8 " in body and "/bon/G4-7-i15/t10s8 " in body and "/bon/G4-7-i15/select/choice.parquet" in body
    assert "--model G4-0=" in body and "--model G4-7-i15=" in body and "-x 'translations/.*|scores/.*'" in body
    out, _ = run("select", MODEL="G4-7-i15", TAG="c1")
    body = (home / "job_body_pj_select_G4-7-i15_c1.sh").read_text()
    assert "pairjudge.py select --model G4-7-i15" in body and "/judge/G4-7-i15/select" in body and "/bon/scores/w3/" in body
    out, _ = run("texts", TAG="x1")
    body = (home / "job_body_pj_texts_x1.sh").read_text()
    assert body.count("--pairs ") == 12 and "g4bon.py texts" in body and "/judge/naturalness/texts.parquet" in body
    out, _ = run("bonreport", TAG="x2")
    body = (home / "job_body_pj_bonreport_x2.sh").read_text()
    assert "/bon/naturalness/G4-0_G4-7-i15/predictions.parquet" in body and "/sampling/naturalness/predictions.parquet" in body
    assert body.count("--model ") == 2 and "pairjudge.py bonreport" in body
    out, _ = run("classify", TAG="k1")
    assert out.returncode != 0 and "no " in out.stdout and "texts" in out.stdout
    out, _ = run("blind", TAG="z1")
    assert out.returncode == 0, out.stdout + out.stderr
    body = (home / "job_body_pj_blind_z1.sh").read_text()
    assert "pairjudge.py blind --model G4-7-i15" in body and "/sft/blind/G4-judge-i15/" in body
    key = g.parent / "blind" / "G4-judge-i15" / "key.jsonl"
    key.parent.mkdir(parents=True)
    key.write_text("x")
    out, _ = run("blind", TAG="z2")
    assert out.returncode != 0 and "stays as it is" in out.stdout


# ---------------------------------------------------------------------------------------------------- A: judge or anchor

def test_reason_prompt_is_the_protocols_text_letter_for_letter():
    text = (CODE / "protocol-g4.md").read_text(encoding="utf-8")
    section = text[text.index("*G4-judge.13 B:"):]
    block = re.search(r"\*\*Prompt B\*\*.*?```\n(.*?)\n```", section, re.S).group(1)
    assert block == pj.REASON_PROMPT
    head = pj.PROMPT[:pj.PROMPT.index("Welke vertaling is beter?")]
    assert pj.REASON_PROMPT.startswith(head) and pj.REASON_PROMPT != pj.PROMPT
    filled = pj.user_text("src", "a", "b", pj.REASON_PROMPT)
    assert filled.endswith('alleen: "Oordeel: A", "Oordeel: B" of "Oordeel: gelijk".') and "Vertaling A:\na\n" in filled
    (u1, t1, _), _ = pj.pair_prompts("p/b1", "src", "first", "second", pj.REASON_PROMPT)
    assert u1 == "p/b1/ab" and t1.startswith(head[:40]) and "Noem eerst" in t1


def anchor_flow(tmp_path, monkeypatch):
    """Step 2 in the synthetic world (bonpairs, scores, select), then janchor for both models"""
    w = bon_setup(tmp_path, monkeypatch)
    ev, sp = str(w / "evalsets.parquet"), str(w / "split.json")
    data = tmp_path / "bondata"
    margs = []
    for m in pj.BON_MODELS:
        margs += ["--model", f"{m}={samples_of(w, m)},{tmp_path / f'bon-{m}' / 'choice.parquet'}"]
    pj.main(["bonpairs", *margs, "--anchor", str(w / "eval" / "G4-0"), "--evalsets", ev, "--split", sp, "--out", str(data)])
    fake_bon_scores(data, tmp_path / "bonscores" / "w0")
    for m in pj.BON_MODELS:
        pj.main(["select", "--model", m, "--samples", str(samples_of(w, m)), "--choice",
                 str(tmp_path / f"bon-{m}" / "choice.parquet"), "--scores", str(tmp_path / "bonscores/w*/shard-*.parquet"),
                 "--evalsets", ev, "--split", sp, "--out", str(tmp_path / f"j-{m}")])
        pj.main(["anchorselect", "--model", m, "--select", str(tmp_path / f"j-{m}"), "--samples", str(samples_of(w, m)),
                 "--anchor", str(w / "eval" / "G4-0"), "--evalsets", ev, "--split", sp, "--out", str(tmp_path / f"a-{m}")])
    return w, ev, sp


def test_anchorselect_keeps_the_anchor_unless_the_judge_prefers_a_sample(tmp_path, monkeypatch, capsys):
    import armeval

    w, ev, sp = anchor_flow(tmp_path, monkeypatch)
    assert "ASELECT" in capsys.readouterr().out
    anchor = armeval.scored(w / "eval" / "G4-0")
    for m in pj.BON_MODELS:
        j = {r["id"]: r for r in g4arm.read_rows(tmp_path / f"j-{m}" / "choice.parquet")}
        a = {r["id"]: r for r in g4arm.read_rows(tmp_path / f"a-{m}" / "choice.parquet")}
        rows = armeval.scored(tmp_path / f"a-{m}" / "janchor")
        judgeq = armeval.scored(tmp_path / f"j-{m}" / "judgeq")
        assert set(a) == set(j) == set(rows)
        kinds = collections_counter(r["reason"] for r in a.values())
        assert kinds["sample"] and kinds["p"], kinds      # the synthetic judge gives both
        for i, r in a.items():
            k = pj.SAMPLE_NAMES.index(j[i]["judgeq_choice"])
            if j[i]["judgeq_fallback"]:
                assert r["janchor_choice"] == "anchor" and r["reason"] == "fallback"
            elif j[i]["p_sample"][k] > 0.5:
                assert r["janchor_choice"] == j[i]["judgeq_choice"] and r["reason"] == "sample"
                assert rows[i]["hypothesis"] == judgeq[i]["hypothesis"] and rows[i]["metricx_qe"] == judgeq[i]["metricx_qe"]
            else:
                assert r["janchor_choice"] == "anchor" and r["reason"] == "p" and r["p_judgeq"] == j[i]["p_sample"][k]
            if r["janchor_choice"] == "anchor":
                assert rows[i]["hypothesis"] == anchor[i]["hypothesis"] and rows[i]["comet22"] == anchor[i]["comet22"]
        stats = json.loads((tmp_path / f"a-{m}" / "anchorselect.json").read_text())["stats"]
        for s, x in stats.items():
            assert x["anchor"] + x["sample"] == x["n"] and x["fallback"] + x["p_at_most"] == x["anchor"]
            assert x["anchor"] == sum(r["janchor_choice"] == "anchor" for r in a.values() if r["set"] == s)
    # exactly 0.5 keeps the anchor; a missing score of the judgeq choice is refused
    rec = {"id": "x", "judgeq_fallback": False, "judgeq_choice": "t10s2", "p_sample": [None, 0.5] + [None] * 6}
    assert pj.anchor_pick(rec) == (None, "p")
    assert pj.anchor_pick({**rec, "p_sample": [None, 0.5000001] + [None] * 6}) == (1, "sample")
    with pytest.raises(SystemExit, match="no judge score"):
        pj.anchor_pick({**rec, "p_sample": [None] * 8})


def collections_counter(values):
    import collections

    return collections.Counter(values)


def test_anchor_report_and_blind(tmp_path, monkeypatch, capsys):
    import armeval

    w, ev, sp = anchor_flow(tmp_path, monkeypatch)
    targs = []
    for m in pj.BON_MODELS:
        targs += ["--pairs", f"{m}={w / 'eval' / m}/pairs/pairs-w*.parquet",
                  "--pairs", f"{m}/t10s1={samples_of(w, m) / 't10s1'}/pairs/pairs-w*.parquet",
                  "--pairs", f"{m}/qe={tmp_path / f'bon-{m}'}/qe/pairs/pairs-w*.parquet",
                  "--pairs", f"{m}/c2q={tmp_path / f'bon-{m}'}/c2q/pairs/pairs-w*.parquet",
                  "--pairs", f"{m}/judge={tmp_path / f'j-{m}'}/judge/pairs/pairs-w*.parquet",
                  "--pairs", f"{m}/judgeq={tmp_path / f'j-{m}'}/judgeq/pairs/pairs-w*.parquet",
                  "--pairs", f"{m}/janchor={tmp_path / f'a-{m}'}/janchor/pairs/pairs-w*.parquet"]
    g4bon.main(["texts", "--evalsets", ev, "--split", sp, *targs, "--out", str(tmp_path / "texts.parquet")])
    texts = g4arm.read_rows(tmp_path / "texts.parquet")
    assert len({r["model"] for r in texts}) == 14
    # C1 a function of the text: the same text has the same label in both runs
    c1 = [{"model": r["model"], "set": r["set"], "id": r["id"],
           "p_original": 0.9 if sum(map(ord, r["text"])) % 3 == 0 else 0.1} for r in texts]
    g4arm.write_parquet(c1, tmp_path / "c1.parquet")
    g4arm.write_parquet([r for r in c1 if not r["model"].endswith("/janchor")], tmp_path / "step2.parquet")
    probe = [{**r, "model": r["model"].replace("G4-0/qe", "G4-0/best8")} for r in c1 if r["model"].startswith("G4-0")]
    g4arm.write_parquet(probe, tmp_path / "probe.parquet")
    g4arm.write_parquet([r for r in c1 if r["model"].startswith("G4-7-i15")], tmp_path / "bonc1.parquet")
    refqe = [{"id": r["id"], "qe_source_reference": 2.3} for r in g4arm.read_rows(tmp_path / "bon-G4-0/choice.parquet", ["id"])]
    g4arm.write_parquet(refqe, tmp_path / "refqe.parquet")
    rargs = []
    for m in pj.BON_MODELS:
        rargs += ["--model", f"{m}={w / 'eval' / m},{samples_of(w, m) / 't10s1'},{tmp_path / f'bon-{m}'},"
                             f"{tmp_path / f'j-{m}'},{tmp_path / f'a-{m}'}"]
    common = ["bonreport", "--evalsets", ev, "--split", sp, "--refqe", str(tmp_path / "refqe.parquet"), "--anchor",
              str(w / "eval" / "G4-0"), "--predictions", str(tmp_path / "c1.parquet"), "--probe-predictions",
              str(tmp_path / "probe.parquet"), "--bon-predictions", str(tmp_path / "bonc1.parquet")]
    with pytest.raises(SystemExit, match="judge-predictions"):
        pj.main([*common, *rargs, "--out", str(tmp_path / "x.md")])
    with pytest.raises(SystemExit, match="four directories"):
        pj.main([*common, *rargs[:2], "--model", rargs[3].rsplit(",", 1)[0], "--judge-predictions",
                 str(tmp_path / "step2.parquet"), "--out", str(tmp_path / "x.md")])
    pj.main([*common, *rargs, "--judge-predictions", str(tmp_path / "step2.parquet"), "--out", str(tmp_path / "rep.md")])
    rep = json.loads((tmp_path / "rep.json").read_text())
    md = (tmp_path / "rep.md").read_text()
    assert rep["anchor_mode"] and rep["outputs"][-1] == "janchor"
    for m in pj.BON_MODELS:
        x = rep["models"][m]
        a = {r["id"]: r for r in g4arm.read_rows(tmp_path / f"a-{m}" / "choice.parquet")}
        assert {"janchor-greedy", "janchor-judgeq", "janchor-qe", "janchor-c2q"} <= set(x["contrasts"])
        assert ("janchor-G4-0 greedy" in x["contrasts"]) == (m != "G4-0")
        # janchor's C1 share by hand from the composition
        lab = {(r["model"], r["id"]): r["p_original"] >= 0.5 for r in c1}
        assert x["outputs"]["janchor"]["c1_pooled"] == pytest.approx(
            sum(lab[(f"{m}/janchor", i)] for i in a) / len(a))
        jq = armeval.scored(tmp_path / f"a-{m}" / "janchor")
        g0 = armeval.scored(w / "eval" / "G4-0")
        wmt = [i for i in a if a[i]["set"] == "wmt24pp"]
        assert x["outputs"]["janchor"]["sets"]["wmt24pp"]["comet22"] == pytest.approx(
            sum(jq[i]["comet22"] for i in wmt) / len(wmt))
        assert any(jq[i]["hypothesis"] == g0[i]["hypothesis"] for i in a)
        assert x["anchor"] == json.loads((tmp_path / f"a-{m}" / "anchorselect.json").read_text())["stats"]
    assert all(v["flipped"] == 0 for v in rep["checks"].values())
    assert len(rep["checks"]) == 7 + 2 * (6 + 1)
    assert "aanvulling A: judge of anker" in md and "## Tabel 5: judge of anker" in md and "janchor-judgeq" in md
    assert "JBONREPORT" in capsys.readouterr().out
    # without the fifth directory the report is step 2's, unchanged in form
    pj.main([*common, *[x if not x.startswith("G4") else x.rsplit(",", 1)[0] for x in rargs], "--out", str(tmp_path / "rep2.md")])
    rep2 = json.loads((tmp_path / "rep2.json").read_text())
    assert not rep2["anchor_mode"] and "janchor" not in rep2["models"]["G4-0"]["outputs"]
    assert rep2["models"]["G4-0"]["outputs"]["judgeq"]["c1_vs_g40"] == rep["models"]["G4-0"]["outputs"]["judgeq"]["c1_vs_g40"]
    assert "stap 2: best-of-8" in (tmp_path / "rep2.md").read_text()
    # the blind package of janchor; the overlap with an earlier package, by item ids only
    earlier = tmp_path / "earlier-key.jsonl"
    earlier.write_text("".join(json.dumps({"blind_id": f"judge-{n:03d}", "item": i}) + "\n"
                               for n, i in enumerate(sorted(armeval.scored(tmp_path / "a-G4-7-i15" / "janchor"))[:3])))
    pj.main(["blind", "--kind", "janchor", "--model", "G4-7-i15", "--select", str(tmp_path / "a-G4-7-i15"), "--anchor",
             str(w / "eval" / "G4-0"), "--evalsets", ev, "--split", sp, "--overlap", str(earlier), "--overlap-name",
             "G4-judge-i15", "--out", str(tmp_path / "ablind")])
    keys = [json.loads(x) for x in (tmp_path / "ablind" / "key.jsonl").read_text().splitlines()]
    inputs = [json.loads(x) for x in (tmp_path / "ablind" / "judge-input.jsonl").read_text().splitlines()]
    sel = json.loads((tmp_path / "ablind" / "select-report.json").read_text())
    target = armeval.scored(tmp_path / "a-G4-7-i15" / "janchor")
    g0 = armeval.scored(w / "eval" / "G4-0")
    differ = [i for i in target if (target[i]["hypothesis"] or "").strip() != (g0[i]["hypothesis"] or "").strip()]
    assert len(keys) == len(inputs) == len(differ) > 0 and sel["target"] == "G4-7-i15-janchor"
    assert all(k["blind_id"].startswith("anchor-") and k["seed"].startswith("g4judge-anchor/blind/ab/anchor-") for k in keys)
    assert sel["sets"]["wmt24pp"]["seed"] == "g4judge-anchor/blind/wmt24pp"
    assert sel["overlap"] == {"package": "G4-judge-i15", "items": sum(k["item"] in {
        json.loads(x)["item"] for x in earlier.read_text().splitlines()} for k in keys)}
    for x, k in zip(inputs, keys):
        assert (random.Random(k["seed"]).random() < 0.5) == (k["A"] == "base")
        assert (x["A"] if k["A"] == "base" else x["B"]) == g0[k["item"]]["hypothesis"]


# ---------------------------------------------------------------------------------------------------- B: reasoning first

def test_split_reasoning_finds_the_verdict_line_and_the_own_answer():
    assert pj.split_reasoning("A laat iets weg; B klopt.\n\nOordeel: B") == ("A laat iets weg; B klopt.", True, "B")
    assert pj.split_reasoning("Geen fouten.\n**Oordeel:** gelijk") == ("Geen fouten.", True, "gelijk")
    assert pj.split_reasoning("x\n## Oordeel: Vertaling A\nmeer") == ("x", True, "A")
    assert pj.split_reasoning("Oordeel: A") == ("", True, "A")
    assert pj.split_reasoning("x\n- oordeel: beide goed") == ("x", True, "anders")
    assert pj.split_reasoning("Alleen redenering, geen regel.  \n") == ("Alleen redenering, geen regel.", False, None)
    # "Oordeel" inside a line does not count
    assert pj.split_reasoning("Mijn Oordeel volgt.\nOordeel: A")[0] == "Mijn Oordeel volgt."
    assert pj.answer_text("r") == "r\n\nOordeel:" and pj.answer_text("") == "Oordeel:"


@pytest.mark.skipif(not TOKENIZER, reason="no Gemma 4 tokenizer in the local Hugging Face cache")
def test_the_answer_position_extends_the_prompt_and_takes_the_answer_ids():
    import translate
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(TOKENIZER[0])
    turn = translate.MODELS[pj.MODEL]["turn"]
    head = turn["prefix"] + pj.user_text("The cat sat.", "De kat zat.", "De poes zat.", pj.REASON_PROMPT) + turn["suffix"]
    ids = tok.encode(head, add_special_tokens=False).ids
    for reasoning in ("", "A en B kloppen; B klinkt natuurlijker.", "**A**: geen fout.\n- B: weglating"):
        full = tok.encode(head + pj.answer_text(reasoning), add_special_tokens=False).ids
        assert full[:len(ids)] == ids
    for answer, want in (("A", 562), ("B", 603), ("gelijk", 190506)):
        assert tok.encode(f"x\n\nOordeel: {answer}", add_special_tokens=False).ids[-1] == want


class FakeTok:
    """A character tokenizer (ids = code points) with the calls reason() makes"""

    def __call__(self, texts, add_special_tokens=False):
        return {"input_ids": [[ord(c) for c in t] for t in texts] if isinstance(texts, list) else [ord(c) for c in texts]}

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(i) for i in ids)


def fake_reason_env(tmp_path, monkeypatch, completion_of):
    """reason() without a TPU: translate, jax and transformers replaced; the engine's answer logits favour the own
    answer the completion wrote; the bucket is a local directory"""
    import types

    import numpy as np

    bucket = tmp_path / "bucket"

    def local(url):
        return bucket / url[len("gs://"):]

    def decode_all(prompts, slots, device, max_decode, sync, eos):
        assert (slots, max_decode, sync) == (pj.REASON_SLOTS, pj.REASON_MAX_DECODE, pj.REASON_SYNC)
        out = []
        for p in prompts:
            text = completion_of("".join(chr(i) for i in p))
            out.append(([ord(c) for c in text], not text.endswith("…")))
        return out

    class Engine:
        def prefill(self, params, padded_tokens, true_length):
            text = "".join(chr(int(i)) for i in np.asarray(padded_tokens)[:true_length])
            assert text.endswith(pj.ANSWER_CUE)
            own = pj.split_reasoning(completion_of(text))[2]
            logits = np.full(262144, -4.0, np.float32)
            hot = {"A": 562, "B": 603, "gelijk": 190506}.get(own, 7)
            logits[hot] = 6.0
            return {"logits": logits, "tokens": np.asarray([[hot]])}, None

    class Slots:
        def __init__(self, engine, params, lengths):
            assert lengths == list(pj.REASON_PREFILL)

        def prefill(self, tokens):
            return {}, 0

        def insert(self, prefix, slot):
            pass

        def generate(self):
            return np.zeros(pj.REASON_SLOTS)

        @staticmethod
        def fetch(x):
            return x

    def upload(path, url, client=None):
        local(url).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(path, local(url))

    tr = types.SimpleNamespace(
        MODELS={pj.MODEL: {"turn": {"prefix": "<bos>U:", "suffix": "<channel|>", "check": True}, "eos_ids": [1, 106, 50]}},
        gcs_client=lambda: None, upload=upload, maxtext_source=lambda: {}, patch_maxtext=lambda: [],
        done_shards=lambda out, client=None: {p.name for p in local(out).glob("*.parquet")} if local(out).exists() else set(),
        make_engine=lambda ns, shapes, chips: (Engine(), None), EngineSlots=Slots, decode_all=decode_all)
    jax = types.SimpleNamespace(device_count=lambda: 4, process_count=lambda: 1, device_get=lambda x: x)
    monkeypatch.setitem(sys.modules, "translate", tr)
    monkeypatch.setitem(sys.modules, "jax", jax)
    monkeypatch.setitem(sys.modules, "jax.numpy", types.SimpleNamespace(asarray=np.asarray))
    jax.numpy = sys.modules["jax.numpy"]
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda path: FakeTok())))
    monkeypatch.setattr(pj, "check_answer_tokens", lambda tok: None)
    monkeypatch.setattr(pj, "check_template", lambda tok, turn, texts: None)
    return local


def completion_for(text):
    """The synthetic judge's completion: it follows the A/B mark in the prompt's texts"""
    if "LONGLONG" in text:
        return "eindeloos…"
    if "Vertaling A:\nQQTEXTQQ A" in text and "blind-0" in text and int(re.search(r"blind-(\d+)", text).group(1)) % 3 == 0:
        return "Beide goed.\nOordeel: gelijk"
    first = re.search(r"Vertaling A:\n\S+ (\w)", text)
    return f"{first.group(1) if first else 'A'} is beter.\n\n**Oordeel:** A"


def test_reason_scores_the_answer_position_and_times_both_phases(tmp_path, monkeypatch, capsys):
    local = fake_reason_env(tmp_path, monkeypatch, completion_for)
    rows = [{"uid": f"P/b{n}/ab", "text": f"Vertaling A:\nQQTEXTQQ A {n}\nVertaling B:\nx"} for n in range(5)]
    rows.append({"uid": "P/long/ab", "text": "y" * (pj.REASON_MAX_PROMPT + 1)})
    rows.append({"uid": "P/loop/ab", "text": "LONGLONG"})
    pj.write_rows(rows, tmp_path / "prompts.parquet")
    out = "gs://b/reason/scores/w0"
    args = ["reason", "--prompts", str(tmp_path / "prompts.parquet"), "--out", out, "--ckpt", "c", "--tokenizer", "t",
            "--work-dir", str(tmp_path / "work")]
    pj.main(args)
    log = capsys.readouterr().out
    assert "WARMUP" in log and "TIMING" in log and "REASONED" in log
    got = {r["uid"]: r for r in pj.read_rows(local(out) / "shard-00000.parquet")}
    assert set(got) == {r["uid"] for r in rows}
    assert got["P/long/ab"]["skipped"] == "too_long" and got["P/long/ab"]["p_a"] is None
    r = got["P/b1/ab"]
    assert r["skipped"] is None and r["verdict_line"] and r["own_answer"] == "A" and r["ended"]
    assert r["p_a"] > 0.9 and r["first_token"] == 562 and r["completion"].endswith("**Oordeel:** A")
    assert r["reasoning_tokens"] == len("A is beter.") + len("\n\nOordeel:")
    loop = got["P/loop/ab"]
    assert loop["ended"] is False and loop["verdict_line"] is False and loop["skipped"] is None
    t = json.loads((local(out) / "timing-shard-00000.json").read_text())
    assert t["prompts"] == 6 and t["not_scored"] == 1 and t["slots"] == pj.REASON_SLOTS
    assert t["gen_tokens"] == sum(got[u]["gen_tokens"] for u in got if got[u]["gen_tokens"] is not None)
    # a restart skips the shard that is there
    pj.main(args)
    assert "TIMING" not in capsys.readouterr().out


def test_reasonreport_against_step_1(tmp_path, monkeypatch, capsys):
    d = make_packages(tmp_path)
    monkeypatch.setattr(pj, "EXPECTED", expected_counts(d))
    data = tmp_path / "data"
    pj.main(["valdata", "--blind", str(d), "--out", str(data)])
    rdata = tmp_path / "rdata"
    pj.main(["valdata", "--prompt", "reason", "--blind", str(d), "--out", str(rdata)])
    assert json.loads((rdata / "valdata.json").read_text())["prompt"] == "reason"
    assert pj.read_rows(rdata / "refs.parquet") == pj.read_rows(data / "refs.parquet")
    rp = {r["uid"]: r["text"] for k in range(4) for r in pj.read_rows(rdata / f"prompts-w{k}.parquet")}
    assert all("Noem eerst" in t for t in rp.values()) and len(rp) == 1000
    # step 1: wrong at every 4th pair; the reasoning judge: wrong at every 8th, with the extra columns
    fake_val_scores(data, tmp_path / "base" / "w0")
    fake_val_scores(data, tmp_path / "rs" / "w0", agree_every=8)
    rows = pj.read_rows(tmp_path / "rs/w0/shard-00000.parquet")
    for n, r in enumerate(rows):
        r.update({"gen_tokens": 40 + n % 7, "ended": n % 50 != 0, "verdict_line": n % 25 != 0,
                  "own_answer": "anders" if n % 25 == 0 else pj.verdict_of({"A": r["p_a"], "B": r["p_b"], "gelijk": r["p_eq"]}),
                  "reasoning_tokens": 30, "completion": f"{MARK} redenering"})
    pj.write_rows(rows, tmp_path / "rs/w0/shard-00000.parquet")
    for k, (dec, ans, n) in enumerate([(100.0, 20.0, 600), (90.0, 10.0, 400)]):
        (tmp_path / "rs" / "w0" / f"timing-shard-0000{k}.json").write_text(json.dumps(
            {"out": f"gs://b/scores/w{k}", "prompts": n, "decode_seconds": dec, "answer_seconds": ans, "gen_tokens": 40 * n,
             "engine_seconds": 150.0, "warmup_seconds": 200.0, "not_scored": 0}))
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "pj_score_bon_t-w0.log").write_text(
        'PROMPTS {"rows": 10}\nengine ready in 150 s\nCOMPILE {"bucket": 512, "seconds": 30.0}\n'
        'SHARD {"shard": "s", "rows": 100, "seconds": 50.0}\nSHARD {"shard": "t", "rows": 100, "seconds": 20.0}\n')
    pj.main(["reasonreport", "--refs", str(data / "refs.parquet"), "--scores", str(tmp_path / "rs/w*/shard-*.parquet"),
             "--base-scores", str(tmp_path / "base/w*/shard-*.parquet"), "--timing", str(tmp_path / "rs/w*/timing-*.json"),
             "--base-logs", str(tmp_path / "logs/*.log"), "--out", str(tmp_path / "rr.md")])
    r = json.loads((tmp_path / "rr.json").read_text())
    md = (tmp_path / "rr.md").read_text()
    assert MARK not in md and MARK not in (tmp_path / "rr.json").read_text()
    assert r["paired"]["D1"]["delta"] > 0 and r["paired"]["noneq"]["new"] > r["paired"]["noneq"]["old"]
    assert r["paired"]["D1"]["new"] == pytest.approx(r["gate"]["D1"]["share"])
    assert r["paired"]["D1"]["old"] == pytest.approx(r["base"]["gate"]["D1"]["share"])
    assert r["paired"]["fault"]["new"] == pytest.approx(r["fault"]["share"])
    assert r["paired"]["idiom"]["n"] == r["sets"]["idiom"]["direction_noneq"]["n"]
    t = r["timing"]
    assert t["per_pair"] == pytest.approx(2 * (100 + 20 + 90 + 10) / 1000)
    assert t["hours_one_v4_8"] == pytest.approx(pj.SELECTION_SAMPLES * t["per_pair"] / 3600)
    assert t["base_per_pair"] == pytest.approx(2 * (70 - 30) / 200)
    x = r["reasoning"]["ab"]
    assert x["n"] == 500 and x["no_verdict_line"] == sum(1 for n, rr in enumerate(rows) if n % 25 == 0 and rr["uid"].endswith("/ab"))
    assert x["own_differs"] == 0
    assert "aanvulling B" in md and "| D1 |" in md and "Tabel 8" in md and "[gevolgtrekking]" in md
    hours = f"{pj.SELECTION_SAMPLES * t['per_pair'] / 3600:.1f}".replace(".", ",")
    assert f"op 100.000 samples (elk tegen het anker, beide volgordes) ~{hours} uur op één v4-8" in md
    assert "RREPORT" in capsys.readouterr().out


def test_run_pairjudge_a_and_b_stages_render_their_jobs(tmp_path):
    run, shm, home, g = pj_world(tmp_path)
    out, _ = run("aselect", MODEL="G4-7-i15", TAG="a1")
    assert out.returncode == 0 and "started pj_aselect_G4-7-i15_a1 on w0" in out.stdout, out.stdout + out.stderr
    body = (home / "job_body_pj_aselect_G4-7-i15_a1.sh").read_text()
    assert "pairjudge.py anchorselect --model G4-7-i15" in body and "/judge/G4-7-i15/select/choice.parquet" in body
    assert "/bon/G4-7-i15/t10s8 " in body and "/judge/G4-7-i15/select-anchor" in body and "/eval/G4-0 " in body
    out, _ = run("atexts", TAG="a2")
    body = (home / "job_body_pj_atexts_a2.sh").read_text()
    assert body.count("--pairs ") == 14 and "/janchor/pairs/" in body and "/judge/anchor/naturalness/texts.parquet" in body
    out, _ = run("areport", TAG="a3")
    body = (home / "job_body_pj_areport_a3.sh").read_text()
    assert "--judge-predictions" in body and "/judge/naturalness/predictions.parquet" in body
    assert "/judge/anchor/naturalness/predictions.parquet" in body and "/judge/anchor/report/" in body
    assert body.count("/select-anchor ") == 2 and body.count(",/") >= 8
    out, _ = run("bonreport", TAG="a4")
    body = (home / "job_body_pj_bonreport_a4.sh").read_text()
    assert "--judge-predictions" not in body and "select-anchor" not in body and "/judge/bon/report/" in body
    out, _ = run("aclassify", TAG="a5")
    assert out.returncode != 0 and "atexts" in out.stdout
    out, _ = run("ablind", TAG="a6")
    assert out.returncode == 0, out.stdout + out.stderr
    body = (home / "job_body_pj_ablind_a6.sh").read_text()
    assert "blind --kind janchor --model G4-7-i15" in body and "/sft/blind/G4-judge-anchor-i15/" in body
    assert "/sft/blind/G4-judge-i15/key.jsonl" in body and "--overlap-name G4-judge-i15" in body
    out, _ = run("rvaldata", TAG="r1")
    body = (home / "job_body_pj_rvaldata_r1.sh").read_text()
    assert "valdata --prompt reason" in body and "/judge/reason/val/data/" in body
    for host in "0123":
        out, _ = run("reason", host=host, TAG="r2")
        assert out.returncode == 0 and f"started pj_reason_r2 on w{host}" in out.stdout, out.stdout + out.stderr
        body = (home / "job_body_pj_reason_r2.sh").read_text()
        assert f"/judge/reason/val/data/prompts-w{host}.parquet" in body and f"/judge/reason/val/scores/w{host} " in body
        assert "pairjudge.py reason" in body and "TPU_VISIBLE_CHIPS=0,1,2,3" in body and "preflight failed" in body
        (home / "logs" / "pj_reason_r2.log").unlink(missing_ok=True)
    out, _ = run("rreport", TAG="r3")
    body = (home / "job_body_pj_rreport_r3.sh").read_text()
    assert "pairjudge.py reasonreport" in body and "/judge/val/scores/w3/shard-*.parquet" in body
    assert "/judge/reason/val/scores/w2/timing-*.json" in body and "pj_score_bon_*-w*.log" in body
    key = g.parent / "blind" / "G4-judge-anchor-i15" / "key.jsonl"
    key.parent.mkdir(parents=True)
    key.write_text("x")
    out, _ = run("ablind", TAG="a7")
    assert out.returncode != 0 and "stays as it is" in out.stdout
