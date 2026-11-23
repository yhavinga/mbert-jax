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
