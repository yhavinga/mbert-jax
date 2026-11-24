"""G4-maat (protocol-g4.md §23): g4measure.py data (the 14 packages, Yeb's two samples, the mixed sample rebuilt and
checked against its file, the shared pairs, the scorers' inputs) and report (the side each signal picks, K1-K4, the
agreement with Yeb and Claude, the checks) on small synthetic inputs, and run_g4measure.sh's stages rendered (no model,
no TPU)."""

import json
import os
import random
import re
import subprocess
import sys
import time
from argparse import Namespace
from pathlib import Path

import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "experiments" / "en-nl-translator"
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(ROOT / "experiments" / "flan-nl"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import g4measure as gm  # noqa: E402
import g4mixsample as mx  # noqa: E402
import pairjudge  # noqa: E402
from test_en_nl_g4arm8t import FAKE_GSUTIL  # noqa: E402
from test_en_nl_g4rl import FakeCounter7  # noqa: E402

WORDS = ("de", "het", "huis", "rivier", "groen", "loopt", "langzaam", "en", "stad", "oud", "licht", "water", "ziet")
NONEQ = {"G4-7": 43, "G4-7-tg": 76, "G4-bon-i15c2q": 51, "G4-judge-i15": 65, "G4-judge-G4-0": 57,
         "G4-judge-i15-vs-qe": 58, "G4-judge-anchor-i15": 54, "G4-8": 37, "G4-8-tg": 68, "G4-8-replica": 59,
         "G4-8-romans-bt": 52, "G4-8-V3": 68, "G4-9-V3N1": 56, "RB3": 80}
NONBASE = {"G4-7": "G4-7-i15", "G4-7-tg": "G4-7-i15-tg", "G4-bon-i15c2q": "G4-7-i15-c2q", "G4-judge-i15": "G4-7-i15-judgeq",
           "G4-judge-G4-0": "G4-0-judgeq", "G4-judge-i15-vs-qe": "i15-judgeq", "G4-judge-anchor-i15": "G4-7-i15-janchor",
           "G4-8": "G4-8-s140", "G4-8-tg": "G4-8-s140-tg", "G4-8-replica": "G4-8-s140", "G4-8-romans-bt": "G4-8-s140",
           "G4-8-V3": "G4-8-janchor", "G4-9-V3N1": "G4-9-n1", "RB3": "RB3"}
MIX_SETS = {"bt": 10, "idiom": 5, "wmt24pp": 6, "ntrex": 2, "flan": 6, "novels": 8, "flores": 3}


def sentence(rng, n=12):
    return " ".join(rng.choice(WORDS) for _ in range(n)) + f" w{rng.randrange(10 ** 9)}"


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def write_pairs(path, rows):
    pairjudge.write_rows([{"id": i, "set": s, "source": src, "hypothesis": h, "reference": None, "ended": True,
                           "tokens_in": 1, "tokens_out": 1, "decode_loop": False, "english_left": False, "empty": False,
                           "tokens": 1, "over_512": False, "length_class": "tiny"} for i, s, src, h in rows], path)


def build_world(tmp_path, seed=7):
    """A blind directory with the 14 packages and Yeb's two samples, the evaluation's and the mixed sample's pairs, the
    evalsets and §17's prompts, consistent with each other as the real files are"""
    rng = random.Random(seed)
    blind, inp = tmp_path / "blind", tmp_path / "in"
    # the evaluation: sources, references and the outputs of base, G4-8-s140, G4-7-i15
    ev = {}
    for k in range(1, 101):
        bs = "flan" if k <= 35 else "wmt24pp" if k <= 70 else "idiom"
        ev[f"blind-{k:03d}"] = {"set": "blind", "blind_set": bs, "ref": sentence(rng) if bs == "wmt24pp" else None}
    for s, n in (("wmt24pp", 120), ("flores", 20), ("ntrex", 20), ("flan", 120)):
        for k in range(n):
            i = f"flan:0-{k}" if s == "flan" else f"{s}-{k:04d}"
            ev[i] = {"set": s, "blind_set": None, "ref": None if s == "flan" else sentence(rng)}
    for i, e in ev.items():
        e["source"] = sentence(rng)
        e["out"] = {m: sentence(rng) for m in ("base", "G4-8-s140", "G4-7-i15")}
    evalsets = [{"id": i, "set": e["set"], "source": e["source"], "reference": e["ref"], "tokens": 5, "over_512": False,
                 "length_class": "tiny", "meta": json.dumps({"blind_set": e["blind_set"], "item": f"x:{i}"} if e["set"] == "blind" else {})}
                for i, e in ev.items()]
    inp.mkdir(parents=True)
    pairjudge.write_rows(evalsets, inp / "evalsets.parquet")
    for m in ("base", "G4-8-s140", "G4-7-i15"):
        write_pairs(inp / "eval" / m / "pairs-w0.parquet",
                    [(i, e["set"], e["source"], e["out"][m]) for i, e in ev.items()])
    # the mixed sample's new pairs (novels, BT)
    new = {}
    for s in ("novels", "bt"):
        for k in range(30):
            i = f"novels:p{k}" if s == "novels" else f"bt:dpc:d{k}:0"
            new[i] = {"set": s, "source": sentence(rng), "out": {m: sentence(rng) for m in ("base", "G4-8-s140")}}
    for m in ("base", "G4-8-s140"):
        write_pairs(inp / f"mix-{m}.parquet", [(i, e["set"], e["source"], e["out"][m]) for i, e in new.items()])
    # the mixed sample's key and file
    pairs, n = [], 0
    evals = {s: [i for i, e in ev.items() if e["set"] == s] for s in ("wmt24pp", "flores", "ntrex", "flan")}
    evals["idiom"] = [f"blind-{k:03d}" for k in range(71, 76)]
    news = {s: [i for i, e in new.items() if e["set"] == s] for s in ("novels", "bt")}
    for s, count in MIX_SETS.items():
        for i in (news.get(s) or evals[s])[:count]:
            pairs.append((s, i))
    rng.shuffle(pairs)
    key, blocks = [], []
    for n, (s, i) in enumerate(pairs, 1):
        g48 = rng.random() < 0.5
        e = new.get(i) or ev[i]
        a, b = (e["out"]["G4-8-s140"], e["out"]["base"]) if g48 else (e["out"]["base"], e["out"]["G4-8-s140"])
        key.append({"n": n, "set": s, "item": i, "g48": "A" if g48 else "B", "A": "G4-8-s140" if g48 else "G4-0",
                    "B": "G4-0" if g48 else "G4-8-s140"})
        blocks.append({"n": n, "source": e["source"], "A_text": a, "B_text": b})
    (blind / gm.YEB2).mkdir(parents=True)
    (blind / gm.YEB2 / "key.json").write_text(json.dumps({"pairs": key}))
    (blind / gm.YEB2 / gm.YEB2_FILE).write_text(mx.sample_text(blocks, "24 nov 2026"), encoding="utf-8")
    mix_by_item = {k["item"]: k for k in key}
    # the 14 packages
    val_prompts = []
    search = {"wmt24pp": [i for i in ev if ev[i]["set"] == "wmt24pp"], "flan": [i for i in ev if ev[i]["set"] == "flan"]}
    for pkg, (review, count, base_system, _) in gm.PACKAGES.items():
        items = []
        if pkg in gm.FIXED:
            items = [(f"blind-{k:03d}", ev[f"blind-{k:03d}"]["blind_set"], f"blind-{k:03d}") for k in range(1, 101)]
        elif pkg in ("G4-8-romans-bt", "G4-8-V3", "G4-9-V3N1"):
            pool = [i for i in new] if pkg == "G4-8-romans-bt" else [f"novels:v{k}" for k in range(count)]
            items = [(f"x-{k:03d}", (new[i]["set"] if i in new else "novels"), i) for k, i in enumerate(pool[:count], 1)]
            if pkg == "G4-8-romans-bt":
                items += [(f"x-{k:03d}", "novels", f"novels:q{k}") for k in range(len(items) + 1, count + 1)]
        else:
            items = [(f"p-{k:03d}", s, search[s][k % 50 + (50 if pkg.endswith("G4-0") else 0)])
                     for k, s in enumerate(["wmt24pp"] * 50 + ["flan"] * 50)]
        inputs, keys, judg, second = [], [], [], []
        for j, (b, s, item) in enumerate(items):
            base_first = rng.random() < 0.5
            nb = NONBASE[pkg]
            if pkg in gm.FIXED or pkg not in ("G4-8-romans-bt", "G4-8-V3", "G4-9-V3N1"):
                e = ev[item]
                src = e["source"]
                bt = e["out"]["base"] if base_system == "G4-0" else sentence(rng)
                own = e["out"][nb] if nb in e["out"] else sentence(rng)
            elif item in new:
                e = new[item]
                src, bt, own = e["source"], e["out"]["base"], e["out"]["G4-8-s140"]
            else:
                src, bt, own = sentence(rng), sentence(rng), sentence(rng)
            a, bb = (bt, own) if base_first else (own, bt)
            inputs.append({"blind_id": b, "source": "  " + src + " ", "A": a, "B": bb})
            k = {"blind_id": b, "A": "base" if base_first else nb, "B": nb if base_first else "base", "set": s,
                 "item": item}
            if pkg == "G4-8-romans-bt":
                k["steekproef_n"] = mix_by_item[item]["n"] if item in mix_by_item else None
            keys.append(k)
            v = "gelijk" if j >= NONEQ[pkg] else ("A beter" if j % 2 else "B beter")
            judg.append({"blind_id": b, "oordeel": v, "zekerheid": "zeker" if j % 3 else "twijfel",
                         "fout_A": "ja" if j % 5 == 0 else "nee", "fout_B": "ja" if j % 7 == 0 else "nee",
                         "uitleg": "tekst"})
            second.append({"blind_id": b, "oordeel": ("A beter", "B beter", "gelijk")[j % 3]})
            if pkg in gm.VAL:
                for uid, text, _ in pairjudge.pair_prompts(f"{pkg}/{b}", src, a, bb):
                    val_prompts.append({"uid": uid, "text": text})
        d = blind / pkg
        write_jsonl(d / "judge-input.jsonl", inputs)
        write_jsonl(d / "key.jsonl", keys)
        write_jsonl(d / review, judg)
        if pkg in gm.SECOND:
            write_jsonl(d / gm.SECOND[pkg], second)
    (inp / "val").mkdir()
    pairjudge.write_prompts(val_prompts, inp / "val")
    # V1: 20 + 20 pairs of the two judge packages
    key1 = [{"n": n, "package": gm.YEB1_PACKAGES[(n - 1) % 2], "blind_id": f"p-{(n + 1) // 2 + 30:03d}"}
            for n in range(1, 41)]
    (blind / gm.YEB1).mkdir()
    (blind / gm.YEB1 / "key-map.json").write_text(json.dumps(key1))
    return blind, inp


def data_args(blind, inp, out):
    return Namespace(blind=str(blind), eval=[f"{m}={inp}/eval/{m}/pairs-w*.parquet" for m in ("base", "G4-8-s140", "G4-7-i15")],
                     mix=[f"{m}={inp}/mix-{m}.parquet" for m in ("base", "G4-8-s140")], evalsets=str(inp / "evalsets.parquet"),
                     val_prompts=str(inp / "val" / "prompts-w*.parquet"), tokenizer="unused", out=str(out))


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("world")
    blind, inp = build_world(tmp)
    out = tmp / "out"
    info = gm.data(data_args(blind, inp, out), counter=FakeCounter7())
    return blind, inp, out, info


# ---------------------------------------------------------------------------------------------------- data

def test_data_builds_every_pair_its_prompts_and_the_scorers_inputs(world):
    blind, inp, out, info = world
    assert info["pairs"] == 1398 + 40 and info["prompts"] == 6 * 1438 and info["val_prompts_checked"] == 800
    assert info["packages"]["mix"] == 40 and info["packages"]["G4-8-romans-bt"] == 98
    # the 18 novel and BT pairs of the sample in G4-8-romans-bt, the 5 idioms in G4-8
    assert info["shared"] == {"shared_novels": 8, "shared_bt": 10, "shared_idiom": 5}
    refs = pq.read_table(out / "refs.parquet").to_pylist()
    assert not {"source", "A", "B", "text"} & set(refs[0])
    assert sum(r["origin"] == "mix" and r["shared"] is not None for r in refs) == 23
    prompts = [r for k in range(4) for r in pq.read_table(out / f"prompts-w{k}.parquet").to_pylist()]
    assert len(prompts) == 6 * 1438 and len({p["uid"] for p in prompts}) == len(prompts)
    for k in range(4):
        assert all(pairjudge.worker_of(r["uid"]) == k for r in pq.read_table(out / f"prompts-w{k}.parquet").to_pylist())
    uids = {p["uid"] for p in prompts}
    assert {"G4-7/blind-001/rubric/ab", "G4-7/blind-001/n1/ba", "mix/1/n2/ab"} <= uids
    n2 = next(p["text"] for p in prompts if p["uid"] == "mix/1/n2/ab")
    assert "Engelse bron" not in n2 and n2.startswith("Je krijgt twee Nederlandse teksten")
    qe = pq.read_table(out / "qe-in.parquet").to_pylist()
    assert len(qe) == 2 * 1438 and all(r["source"] == r["source"].strip() for r in qe)
    comet = pq.read_table(out / "comet-in.parquet").to_pylist()
    news_pairs = sum(r["set"] in gm.NEWS for r in refs)
    assert len(comet) == 2 * news_pairs and all(r["reference"] for r in comet)
    assert sum(r["has_reference"] for r in refs) == news_pairs
    texts = pq.read_table(out / "texts.parquet").to_pylist()
    assert len(texts) == 2 * 1438 and {t["model"] for t in texts} == {"g4measure"}
    # the sides that are an evaluation output: G4-0's and G4-8-s140's of G4-8, not TranslateGemma's of G4-8-tg
    g48 = next(r for r in refs if r["pair"] == "G4-8/blind-001")
    assert {g48["eval_a"], g48["eval_b"]} == {"base|blind-001", "G4-8-s140|blind-001"}
    tg = next(r for r in refs if r["pair"] == "G4-8-tg/blind-001")
    assert tg["eval_a"] is None and tg["eval_b"] is None
    assert all(r["eval_a"] is None and r["eval_b"] is None for r in refs if r["package"] in gm.NOT_EVAL)
    data_json = json.loads((out / "data.json").read_text())
    assert data_json["inputs"]["sample_file"] and data_json["comet_rows"] == len(comet)


def test_data_stops_when_the_sample_file_is_not_the_rebuilt_pairs(tmp_path):
    blind, inp = build_world(tmp_path)
    md = blind / gm.YEB2 / gm.YEB2_FILE
    md.write_text(md.read_text(encoding="utf-8").replace("**A:**", "**A:** ", 1), encoding="utf-8")
    with pytest.raises(SystemExit, match="not the file Yeb read"):
        gm.data(data_args(blind, inp, tmp_path / "out"), counter=FakeCounter7())
    assert not (tmp_path / "out").exists()


def test_data_stops_on_a_prompt_that_is_not_17s(tmp_path):
    blind, inp = build_world(tmp_path)
    p = inp / "val" / "prompts-w0.parquet"
    rows = pq.read_table(p).to_pylist()
    rows[0]["text"] += " "
    pairjudge.write_rows(rows, p)
    with pytest.raises(SystemExit, match="not §17's prompt"):
        gm.data(data_args(blind, inp, tmp_path / "out"), counter=FakeCounter7())


def test_data_stops_on_a_reference_row_with_another_source(tmp_path):
    blind, inp = build_world(tmp_path)
    p = inp / "evalsets.parquet"
    rows = pq.read_table(p).to_pylist()
    for r in rows:
        if r["id"] == "blind-040":
            r["source"] = "iets anders"
    pairjudge.write_rows(rows, p)
    with pytest.raises(SystemExit, match="source differs"):
        gm.data(data_args(blind, inp, tmp_path / "out"), counter=FakeCounter7())


def test_data_stops_on_other_claude_counts_or_ids(tmp_path):
    blind, inp = build_world(tmp_path)
    f = blind / "G4-8" / "review" / "judge-judgments.jsonl"
    rows = [json.loads(x) for x in f.read_text().splitlines()]
    rows[90]["oordeel"] = "A beter"
    write_jsonl(f, rows)
    with pytest.raises(SystemExit, match="protocol §23 has"):
        gm.data(data_args(blind, inp, tmp_path / "out"), counter=FakeCounter7())
    write_jsonl(f, rows[:99])
    with pytest.raises(SystemExit, match="the same 100 blind ids"):
        gm.data(data_args(blind, inp, tmp_path / "out"), counter=FakeCounter7())


def test_a_shared_pair_with_another_text_is_not_shared(tmp_path):
    blind, inp = build_world(tmp_path)
    f = blind / "G4-8" / "judge-input.jsonl"
    rows = [json.loads(x) for x in f.read_text().splitlines()]
    for r in rows:
        if r["blind_id"] == "blind-071":
            r["A"] += " extra"
    write_jsonl(f, rows)
    info = gm.data(data_args(blind, inp, tmp_path / "out"), counter=FakeCounter7())
    assert info["shared"]["shared_idiom"] == 4 and info["shared"]["text_differs"] == 1


# ---------------------------------------------------------------------------------------------------- choices

def scores(qe=(1.0, 2.0), comet=(None, None), c1=(0.2, 0.8), c2=(0.6, 0.4), rubric=(0.6, 0.3), n1=(0.2, 0.7),
           n2=None):
    j = {"rubric": rubric, "n1": n1, "n2": n2}
    return {"qe": qe, "comet": comet, "c1": c1, "c2": c2,
            "judge": {k: None if v is None else {"A": v[0], "B": v[1], "gelijk": 1 - v[0] - v[1]} for k, v in j.items()}}


def test_pair_signals_and_the_four_combinations():
    r = {"tokens_a": 10, "tokens_b": 12, "base_side": "B"}
    c, m = gm.pair_signals(r, scores())
    assert (c["QE"], c["COMET-22"], c["C1"], c["C2"], c["rubriek"], c["N1"], c["N2"]) == ("A", None, "B", "A", "A", "B", None)
    assert c["lengte (korter)"] == "A" and c["nieuw systeem"] == "A" and c["positie A"] == "A"
    assert m["QE"] == pytest.approx(1.0) and m["rubriek"] == pytest.approx(0.3)
    # |dQE| >= 0.2: K1 and K2 are QE; K3 the majority of QE, rubric, N1; K4 QE (rubric and N1 differ)
    assert (c["K1"], c["K2"], c["K3"], c["K4"]) == ("A", "A", "A", "A")
    c, _ = gm.pair_signals(r, scores(qe=(1.0, 1.1)))
    assert (c["K1"], c["K2"], c["K3"], c["K4"]) == ("B", "A", "A", "A")
    c, _ = gm.pair_signals(r, scores(qe=(2.0, 1.0), rubric=(0.6, 0.3), n1=(0.6, 0.2)))
    assert (c["QE"], c["K1"], c["K2"], c["K3"], c["K4"]) == ("B", "B", "B", "A", "A")
    # ties and missing values: no choice; K3 needs two of the same side
    c, _ = gm.pair_signals({**r, "tokens_b": 10}, scores(qe=(1.0, 1.0), rubric=None, n1=(0.4, 0.4)))
    assert c["QE"] is None and c["rubriek"] is None and c["N1"] is None and c["lengte (korter)"] is None
    assert c["K1"] is None and c["K2"] is None and c["K3"] is None and c["K4"] is None
    c, _ = gm.pair_signals(r, scores(qe=(None, 1.0), rubric=(0.5, 0.1), n1=(0.1, 0.5)))
    assert c["QE"] is None and c["K1"] is None and c["K3"] is None and c["K4"] is None


def test_parse_answers_of_both_samples():
    one = ", ".join(f"{n} {'AB'[n % 2]}" for n in range(1, 41))
    a = gm.parse_answers(one, allow_equal=False)
    assert a[1] == ("B", None) and a[2] == ("A", None)
    two = one.replace("3 B", "3 =").replace("20 A", "20 B fout A")
    a = gm.parse_answers(two + "\n", allow_equal=True)
    assert a[3] == ("=", None) and a[20] == ("B", "A")
    with pytest.raises(SystemExit, match="answers"):
        gm.parse_answers(two, allow_equal=False)
    with pytest.raises(SystemExit, match="not 1..40"):
        gm.parse_answers(one.replace(", 40 A", ""), allow_equal=False)
    with pytest.raises(SystemExit, match="twice"):
        gm.parse_answers(one + ", 4 A", allow_equal=False)
    with pytest.raises(SystemExit, match="answers"):
        gm.parse_answers(one.replace("7 B", "7 C"), allow_equal=False)


def test_agreement_counts_no_choice_apart_and_claude_is_flipped_on_a_shared_pair():
    a = gm.agreement([("A", "A"), ("B", "A"), ("A", None), ("B", "B")])
    assert (a["n_ref"], a["n"], a["k"], a["no_choice"]) == (4, 3, 2, 1) and a["share"] == pytest.approx(2 / 3)
    assert a["lo"] < 2 / 3 < a["hi"]
    pkg = {"pair": "G4-8/blind-071", "origin": "package", "claude": "A"}
    mix = {"pair": "mix/2", "origin": "mix", "shared": "G4-8/blind-071", "flip": True}
    assert gm.claude_in_frame(mix, {"G4-8/blind-071": pkg}) == "B"
    assert gm.claude_in_frame({**mix, "flip": False}, {"G4-8/blind-071": pkg}) == "A"
    assert gm.claude_in_frame({**mix, "shared": None}, {}) is None


# ---------------------------------------------------------------------------------------------------- report

def fake_scores(out, refs, yeb_side, tmp):
    """Scores in which QE picks Yeb's side (or A where Yeb has none), the rubric the other side, N1 position A, N2
    not scored on one pair; C1 and C2 the higher side B; COMET on the news sides"""
    judge, mx_rows, comet, c1 = [], [], [], []
    for r in refs:
        p = r["pair"]
        want = yeb_side.get(p, "A")
        qa, qb = (1.0, 2.0) if want == "A" else (2.0, 1.0)
        mx_rows += [{"id": f"{p}/A", "qe_source_hypothesis": qa}, {"id": f"{p}/B", "qe_source_hypothesis": qb}]
        c1 += [{"model": "g4measure", "set": r["set"], "id": f"{p}/A", "p_original": 0.3, "truncated": False},
               {"model": "g4measure", "set": r["set"], "id": f"{p}/B", "p_original": 0.7, "truncated": False}]
        if r["has_reference"]:
            comet += [{"id": f"{p}/A", "comet22": 0.8}, {"id": f"{p}/B", "comet22": 0.8}]
        rubric_a = want != "A"
        for name in ("rubric", "n1", "n2"):
            for order in ("ab", "ba"):
                # P̄(A) = ½[p_a(ab) + p_b(ba)]
                a_wins = rubric_a if name == "rubric" else True
                pa, pb = (0.7, 0.2) if (a_wins == (order == "ab")) else (0.2, 0.7)
                row = {"uid": f"{p}/{name}/{order}", "skipped": None, "p_a": pa, "p_b": pb, "p_eq": 0.1}
                if name == "n2" and p == "mix/1":
                    row = {"uid": row["uid"], "skipped": "too_long", "p_a": None, "p_b": None, "p_eq": None}
                judge.append(row)
    d = tmp / "scores"
    pairjudge.write_rows(judge, d / "judge" / "shard-00000.parquet")
    pairjudge.write_rows(mx_rows, d / "metricx.parquet")
    pairjudge.write_rows(comet or [{"id": "x", "comet22": 0.0}], d / "comet.parquet")
    pairjudge.write_rows(c1, d / "c1.parquet")
    pairjudge.write_rows(c1, d / "c2.parquet")
    return d


def answers(tmp, world_blind):
    key1 = json.loads((world_blind / gm.YEB1 / "key-map.json").read_text())
    a1 = ", ".join(f"{k['n']} {'AB'[k['n'] % 2]}" for k in key1)
    a2 = ", ".join(f"{n} " + ("=" if n in (1, 3) else "B fout A" if n == 20 else "AB"[n % 2]) for n in range(1, 41))
    (tmp / "a1.txt").write_text(a1)
    (tmp / "a2.txt").write_text(a2)
    side = {f"{k['package']}/{k['blind_id']}": "AB"[k["n"] % 2] for k in key1}
    side.update({f"mix/{n}": "AB"[n % 2] for n in range(1, 41) if n not in (1, 3)})
    side["mix/20"] = "B"
    return side


def report_args(tmp, out, d, blind, **extra):
    return Namespace(refs=str(out / "refs.parquet"), judge=str(d / "judge" / "shard-*.parquet"),
                     metricx=str(d / "metricx.parquet"), comet=str(d / "comet.parquet"), c1=str(d / "c1.parquet"),
                     c2=str(d / "c2.parquet"), yeb1_key=str(blind / gm.YEB1 / "key-map.json"), yeb1_answers=str(tmp / "a1.txt"),
                     yeb2_key=str(blind / gm.YEB2 / "key.json"), yeb2_answers=str(tmp / "a2.txt"),
                     out=str(tmp / "report" / "report.md"), val_scores=None, existing_qe=[], existing_comet=[],
                     existing_c1=[], existing_c2=[], **extra)


def test_report_tables_against_yeb_and_claude(world, tmp_path, capsys):
    blind, inp, out, _ = world
    refs = pq.read_table(out / "refs.parquet").to_pylist()
    side = answers(tmp_path, blind)
    d = fake_scores(out, refs, side, tmp_path)
    r = gm.report(report_args(tmp_path, out, d, blind))
    m = r["main"]
    assert m["Yeb V1"]["QE"]["n_ref"] == 40 and m["Yeb gemengd"]["QE"]["n_ref"] == 38 and m["Yeb samen"]["QE"]["n_ref"] == 78
    assert m["Yeb samen"]["QE"]["share"] == 1.0 and m["Yeb samen"]["rubriek"]["share"] == 0.0
    yeb_a = sum(v == "A" for v in side.values())
    assert m["Yeb samen"]["N1"]["k"] == yeb_a and m["Yeb samen"]["positie A"]["k"] == yeb_a
    # N2 not scored on mix/1 (Yeb said "=" there, so no pair is lost); COMET only on the news pairs and all tied
    assert m["Yeb samen"]["N2"]["n"] == 78 and m["Yeb samen"]["COMET-22"]["n"] == 0
    assert m["Yeb samen"]["COMET-22"]["no_choice"] == 78
    # K1 = K2 = QE here (|dQE| = 1), K4 the QE winner as rubric and N1 differ where Yeb picks B
    assert m["Yeb samen"]["K1"]["share"] == 1.0 and m["Yeb samen"]["K2"]["share"] == 1.0
    assert m["Claude alle"]["QE"]["n_ref"] == 824 and m["Claude gemma"]["QE"]["n_ref"] == 600
    assert m["Claude zonder selectie"]["QE"]["n_ref"] == 191 and "Claude" not in m["Claude alle"]
    # Claude as a signal against Yeb only on the shared pairs (V1's 40 and 23 of the mixed sample, minus "=")
    assert m["Yeb samen"]["Claude"]["n_ref"] == 78 and r["coverage"]["shared_yeb_claude"] == 40 + 23
    assert sum(r["cross"].values()) == 63
    assert "V1 FLAN" in r["yeb_sets"] and "V1 judgeq" in r["yeb_sets"] and "samen romans" in r["yeb_sets"]
    assert "pakket RB3" in r["claude_sets"] and "set novels" in r["claude_sets"]
    assert r["faults"]["Yeb (fout aan één kant)"]["QE"]["n_ref"] == 1
    assert "Claude" in r["faults"]["Yeb (fout aan één kant)"] and "Claude" not in r["faults"]["Claude (fout aan één kant)"]
    # QE picks Yeb's side; where Yeb picks the new system, QE does too
    ns = r["new_system"]["Yeb samen"]["QE"]
    assert ns["n"] == 78 and ns["k"] == m["Yeb samen"]["nieuw systeem"]["k"]
    assert "## 8. Na de uitslag toegevoegd" in (tmp_path / "report" / "report.md").read_text()
    assert r["coverage"]["judge_not_scored"]["n2"] == 1
    md = (tmp_path / "report" / "report.md").read_text()
    assert md.startswith("# G4-maat") and "| QE |" in md and "K1: QE, bij |ΔQE| < 0,2 N1" in md
    assert "100,0% (78/78)" in md
    # no text of a pair in the report
    pairs = pq.read_table(out / "pairs.parquet").to_pylist()
    words = {w for p in pairs[:200] for w in p["A"].split() if w.startswith("w")}
    assert not any(w in md for w in words)
    assert "MEASUREREPORT" in capsys.readouterr().out
    assert json.loads((tmp_path / "report" / "report.json").read_text())["protocol"] == gm.PROTOCOL


def test_report_refuses_incomplete_scores_or_a_refs_file_with_text(world, tmp_path):
    blind, inp, out, _ = world
    refs = pq.read_table(out / "refs.parquet").to_pylist()
    answers(tmp_path, blind)
    d = fake_scores(out, refs, {}, tmp_path)
    rows = pq.read_table(d / "metricx.parquet").to_pylist()[1:]
    pairjudge.write_rows(rows, d / "metricx.parquet")
    with pytest.raises(SystemExit, match="metricx: 1 sides missing"):
        gm.report(report_args(tmp_path, out, d, blind))
    bad = tmp_path / "bad"
    bad.mkdir()
    pairjudge.write_rows([{**refs[0], "source": "x"}], bad / "refs.parquet")
    with pytest.raises(SystemExit, match="text columns"):
        gm.report(report_args(tmp_path, bad, d, blind))


def test_report_checks_against_17s_scores_and_the_existing_scores(world, tmp_path):
    blind, inp, out, _ = world
    refs = pq.read_table(out / "refs.parquet").to_pylist()
    answers(tmp_path, blind)
    d = fake_scores(out, refs, {}, tmp_path)
    judge = pq.read_table(d / "judge" / "shard-00000.parquet").to_pylist()
    val = [{**j, "uid": j["uid"].replace("/rubric/", "/")} for j in judge
           if "/rubric/" in j["uid"] and j["uid"].split("/")[0] in gm.VAL]
    pairjudge.write_rows(val, tmp_path / "val" / "shard-00000.parquet")
    qe = [{"id": "blind-001", "qe_source_hypothesis": 1.5}]
    pairjudge.write_rows(qe, tmp_path / "ex" / "metricx-w0.parquet")
    c1 = [{"model": "G4-0", "id": "wmt24pp-0031", "p_original": 0.9, "truncated": False},
          {"model": "G4-4", "id": "wmt24pp-0031", "p_original": 0.1, "truncated": False}]
    pairjudge.write_rows(c1, tmp_path / "ex" / "c1.parquet")
    args = report_args(tmp_path, out, d, blind)
    args.val_scores = str(tmp_path / "val" / "shard-*.parquet")
    args.existing_qe = [f"base={tmp_path}/ex/metricx-w*.parquet"]
    args.existing_c1 = [f"G4-0={tmp_path}/ex/c1.parquet"]
    r = gm.report(args)
    c = r["checks"]
    assert c["rubric_vs_17"]["n"] == 400 and c["rubric_vs_17"]["max_abs"] == 0 and c["rubric_vs_17"]["same_direction"] == 400
    # blind-001's base side in G4-7 and G4-8 (G4-8-tg's base is TranslateGemma); wmt24pp-0031 in the search packages
    assert c["qe_vs_existing"]["n"] == 2 and c["c1_vs_existing"]["n"] >= 1
    assert set(c["shared_same_scores"]) == {"qe", "c1", "c2", "rubric", "n1", "n2"}


# ---------------------------------------------------------------------------------------------------- run_g4measure.sh

def stage_world(tmp_path):
    shm, gs, home, fake, code = (tmp_path / x for x in ("shm", "gs", "home", "bin", "code"))
    for d in (code, home, fake):
        d.mkdir(parents=True)
    for name in ("run_g4measure.sh", "run_arms.sh"):
        (code / name).write_text((CODE / name).read_text().replace("/dev/shm/", f"{shm}/"))
    for f in CODE.glob("*.py"):
        (code / f.name).symlink_to(f)
    for name, body in {"gsutil": FAKE_GSUTIL, "sudo": "#!/bin/bash\nexit 0\n", "hostname": "#!/bin/bash\necho t1v-n-x-w-0\n",
                       "setsid": '#!/bin/bash\necho "SETSID $*" >> "$FAKELOG"\n'}.items():
        (fake / name).write_text(body)
        (fake / name).chmod(0o755)
    for py in ("translate-generic/venv-fork-f4d8615f4/bin/python", "translator/smoke-lora/venv/bin/python",
               "translator/comet-venv/bin/python"):
        p = shm / py
        p.parent.mkdir(parents=True)
        p.write_text(f'#!/bin/bash\nexec {sys.executable} "$@"\n')
        p.chmod(0o755)
    for tok in ("translate-generic/tokenizer-gemma4", "translator/g4lora/tok-gemma4-31b"):
        (shm / tok).mkdir(parents=True)
        (shm / tok / "tokenizer.json").write_text("{}")
    (shm / "translator/metricx/metricx-24-hybrid-xxl-v2p6-bfloat16").mkdir(parents=True)
    g = gs / "t5-dutch-english-usc2/mbert/translator/sft/g4arm"
    calls = tmp_path / "calls.log"

    def put(rel, text="x"):
        p = g / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)

    def run(stage, host="0", **env):
        (fake / "hostname").write_text(f"#!/bin/bash\necho t1v-n-x-w-{host}\n")
        e = {"HOME": str(home), "PATH": f"{fake}:{os.environ['PATH']}", "FAKEGS": str(gs), "FAKELOG": str(calls),
             "TAG": "t1", **env}
        calls.write_text("")
        out = subprocess.run(["bash", str(code / "run_g4measure.sh"), stage], capture_output=True, text=True,
                             timeout=120, env=e, cwd=home)
        deadline = time.time() + 30
        while "started " in out.stdout and "SETSID" not in calls.read_text() and time.time() < deadline:
            time.sleep(0.2)
        return out, calls.read_text().splitlines()

    def body(name):
        f = home / f"job_body_{name}.sh"
        check = subprocess.run(["bash", "-n", str(f)], capture_output=True, text=True)
        assert check.returncode == 0, check.stderr
        return f.read_text()

    return run, body, put, shm


def test_run_g4measure_renders_its_jobs(tmp_path):
    assert subprocess.run(["bash", "-n", str(CODE / "run_g4measure.sh")]).returncode == 0
    run, body, put, shm = stage_world(tmp_path)
    out, _ = run("data", host="1")
    assert out.returncode != 0 and "data runs on w0 only" in out.stdout
    out, _ = run("data", TAG="d1")
    assert out.returncode == 0 and "started gm_data_d1 on w0" in out.stdout, out.stdout + out.stderr
    b = body("gm_data_d1")
    for x in ("g4measure.py data --blind", "/sft/blind/G4-9-V3N1", "/sft/blind/steekproef-yeb-g4-8-mix",
              "/sft/blind/steekproef-yeb ", "/eval/G4-0/pairs/pairs-w*.parquet", "/eval/G4-7-i15/pairs/",
              "/mixsample/pairs/G4-0.parquet", "/judge/val/data/prompts-w*.parquet", "/data/eval/evalsets.parquet",
              "/g4arm/measure/data/", "/g4arm/measure/naturalness/texts.parquet", "MEASUREDATA_DONE"):
        assert x in b, x
    assert b.index("refs.parquet") < b.index("data.json") and "rm " not in b
    for p in gm.PACKAGES:
        assert f"/sft/blind/{p} " in b, p
    put("measure/data/data.json", "{}")
    out, _ = run("data", TAG="d2")
    assert "there before" in body("gm_data_d2")
    # judge: every worker its own prompts, pairjudge.py score unchanged
    out, _ = run("judge", host="2", TAG="j1")
    assert out.returncode != 0 and "no gs://" in out.stdout
    put("measure/data/prompts-w2.parquet")
    out, _ = run("judge", host="2", TAG="j2")
    assert out.returncode == 0 and "started gm_judge_j2 on w2" in out.stdout
    b = body("gm_judge_j2")
    assert "pairjudge.py score --prompts" in b and "/measure/judge/scores/w2" in b and "prompts-w2.parquet" in b
    # qe on w1, comet on w2, C2 on w0
    out, _ = run("qe", TAG="q1")
    assert out.returncode != 0 and "qe runs on w1 only" in out.stdout
    put("measure/data/qe-in.parquet")
    out, _ = run("qe", host="1", TAG="q2")
    b = body("gm_qe_q2")
    assert out.returncode == 0 and "metricx_qe.py score --pairs" in b and "--pair source:hypothesis --model xxl" in b
    assert "--empty-candidates" in b and "/measure/scores/metricx.parquet" in b
    put("measure/data/comet-in.parquet")
    out, _ = run("comet", host="2", TAG="c1")
    b = body("gm_comet_c1")
    assert out.returncode == 0 and "comet_score.py score" in b and "/measure/scores/comet.parquet" in b
    out, _ = run("classify-c2", TAG="k1")
    assert out.returncode != 0 and "c2/model/params.pkl" in out.stdout
    out, _ = run("classify", TAG="k2")
    assert out.returncode != 0 and "measure/naturalness/texts.parquet" in out.stdout
    put("measure/naturalness/texts.parquet")
    out, _ = run("classify", TAG="k3")
    # run_arms.sh's classify, reached with this measurement's NAT (it stops here: no classifier venv in the test)
    assert out.returncode != 0 and "mbert-dumb/.venv/bin/python" in out.stdout


# ---------------------------------------------------------------------------------------------------- the protocol

def test_protocol_has_23_with_the_packages_and_combinations():
    text = (CODE / "protocol-g4.md").read_text(encoding="utf-8")
    section = text[text.index("## 23. Meting G4-maat"):]
    for pkg, (review, n, _, _) in gm.PACKAGES.items():
        assert re.search(rf"\| {re.escape(pkg)} \| {n} \| .* \| `{re.escape(review)}` \|", section), pkg
    for k, want in (("K1", "de QE-winnaar; bij |ΔQE| < 0,2 de keuze van N1"),
                    ("K2", "de QE-winnaar; bij |ΔQE| < 0,2 de keuze van de rubriek"),
                    ("K3", "de meerderheid van QE, rubriek en N1"), ("K4", "kiezen rubriek en N1 dezelfde kant")):
        assert f"**{k}**: {want}" in section, k
    assert "1.398 paren" in section and "824 niet-gelijk" in section and "600 niet-gelijk" in section
    assert "191 niet-gelijk" in section and gm.EXPECTED == {"pairs": 1398, "noneq": 824, "gemma_noneq": 600,
                                                             "unselected_noneq": 191}
    assert sum(NONEQ.values()) == 824 and gm.MARGIN == 0.2
