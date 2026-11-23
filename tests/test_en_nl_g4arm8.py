"""G4-8-data (protocol-g4.md §18): g4arm8.py (the counts of the 10/45/45 mix, chrF, the fresh draw, the plan with the
anchor and the chunks, MetricX's rows, fail and candidates, the judge prompts, janchor's selection, the blind package
V3, the report) on small synthetic inputs, and run_g4arm8.sh's stages rendered with a fake bucket (no model, no TPU)."""

import hashlib
import json
import os
import random
import re
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "experiments" / "en-nl-translator"
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(ROOT / "experiments" / "flan-nl"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import g4arm8 as g8  # noqa: E402
import g4rl  # noqa: E402
import select_passages as sp  # noqa: E402
import translate as tr  # noqa: E402
from test_en_nl_g4rl import FakeCounter7  # noqa: E402

WORDS = ("the", "house", "river", "green", "walks", "slowly", "and", "city", "old", "light", "water", "sees", "a", "dog",
         "runs", "far", "away", "from", "home", "today")


def sentence(rng, n=24):
    return " ".join(rng.choice(WORDS) for _ in range(n))


# ---------------------------------------------------------------------------------------------------- counts, chrF

def test_the_mix_is_yebs_10_45_45_of_24000_and_the_chunks_add_up():
    assert g8.PARTS8 == {"flan": 0.1, "novels": 0.45, "bt": 0.45}
    assert g8.need8() == {"flan": 2400, "novels": 10800, "bt": 10800}
    assert g8.chunk_counts() == {"c1": {"flan": 400, "novels": 1800, "bt": 1800},
                                 "c2": {"flan": 1000, "novels": 4500, "bt": 4500},
                                 "c3": {"flan": 1000, "novels": 4500, "bt": 4500}}
    # fresh = need - G4-7's planned sources; 3% more to the anchor; twice that drawn
    assert g8.fresh_need(g8.REUSED7) == {"flan": (870, 897, 1794), "novels": (3870, 3987, 7974),
                                         "bt": (3900, 4017, 8034)}
    assert "c2_overflow" not in g8.FAILS8 and set(g8.FAILS8) == set(g4rl.FAILS) - {"c2_overflow"}
    text = (CODE / "protocol-g4.md").read_text(encoding="utf-8")
    section = text[text.index("## 18. G4-8-data"):]
    assert "Besluit Yeb 23 nov ~16:40: FLAN-aandeel als bij G4-7, 10 / 45 / 45" in section
    assert "wijkt af van de spec" in section


def test_chrf_is_sacrebleus_sentence_chrf():
    # values of sacrebleu 2.4.3 CHRF().sentence_score (computed 23 Nov 2026)
    cases = [("De kat zit op de mat.", "De kat zat op de mat.", 72.001124),
             ("Hij liep snel naar huis", "Zij ging langzaam weg", 10.179745), ("abc", "xyz", 0.0),
             ("a", "a b", 55.555556),
             ("Een lange zin met veel woorden erin, en komma.", "Een korte zin.", 27.717304)]
    for h, r, want in cases:
        assert g8.chrf(h, r) == pytest.approx(want, abs=1e-5)
    assert g8.chrf("zelfde tekst", "zelfde  tekst") == pytest.approx(100.0)


# ---------------------------------------------------------------------------------------------------- the fresh draw

def fresh_world(seed=0):
    rng = random.Random(seed)
    flan = [{"id": f"flan:{k}", "en": sentence(rng), "doc": f"flan:w0-{k}", "stratum": ""} for k in range(60)]
    novels = [{"id": f"novels:{k}", "en": sentence(rng), "doc": f"novels:{k % 5}", "stratum": ""} for k in range(80)]
    bt = {s: [{"id": f"bt:{s}:d{k}:d:{k}", "en": sentence(rng), "pid": f"{s}:d{k}:d:{k}", "stratum": s}
              for k in range(40)] for s in ("enwiki", "news")}
    return flan, novels, bt


def test_the_fresh_draw_leaves_out_g47_filters_like_pool7_and_takes_the_first_in_order():
    flan, novels, bt = fresh_world()
    flan[3]["en"] = "short text"                                   # < 20 tokens: never drawn
    old_ids = {flan[0]["id"], novels[0]["id"]}
    old_texts = {flan[1]["en"], bt["news"][0]["en"]}
    blocked = sp.ngrams(novels[1]["en"])                          # a 13-gram of the evaluation
    excluded = {"novels:2", "enwiki:d5"}                          # C2's or RL-val's documents
    need = {"flan": (10, 11, 22), "novels": (10, 11, 22), "bt": (10, 11, 22)}
    rows, report = g8.fresh_draw({"flan": flan, "novels": novels}, bt, (old_ids, old_texts), need, FakeCounter7(),
                                 blocked, excluded)
    ids = {r["id"] for r in rows}
    assert not ids & old_ids and not {r["en"] for r in rows} & old_texts
    assert flan[3]["id"] not in ids and novels[1]["id"] not in ids
    assert not {r["doc"] for r in rows} & excluded
    for part in g8.PARTS8:
        mine = sorted((r for r in rows if r["part"] == part), key=lambda r: r["rank"])
        assert [r["rank"] for r in mine] == list(range(11))
        assert all(20 <= r["tokens"] <= 256 for r in mine)
    strata = [r["stratum"] for r in sorted((r for r in rows if r["part"] == "bt"), key=lambda r: r["rank"])]
    assert strata[:10].count("enwiki") == strata[:10].count("news") == 5   # the BT sets in turn
    assert report["flan"]["out_g47"] == {"g47_id": 1, "g47_text": 1}
    assert report["bt"]["per_source"] == 11
    again, _ = g8.fresh_draw({"flan": flan, "novels": novels}, bt, (old_ids, old_texts), need, FakeCounter7(), blocked,
                             excluded)
    assert again == rows
    with pytest.raises(SystemExit, match="too few candidates"):
        g8.fresh_draw({"flan": flan[:5], "novels": novels}, bt, (old_ids, old_texts), need, FakeCounter7(), blocked,
                      excluded)


# ---------------------------------------------------------------------------------------------------- the plan

SMALL = {"TOTAL8": 200, "REUSED7": {"flan": 10, "novels": 60, "bt": 50},
         "CHUNKS8": (("c1", 40), ("c2", 80), ("c3", 80))}


@pytest.fixture
def small(monkeypatch):
    for k, v in SMALL.items():
        monkeypatch.setattr(g8, k, v)
    return g8.need8()


def plan_inputs(seed=1):
    rng = random.Random(seed)
    plan7 = []
    for part, n in SMALL["REUSED7"].items():
        for k in range(n):
            plan7.append({"id": f"{part}:old{k}", "part": part, "stratum": "s1" if part == "bt" else "", "doc": f"{part}:d",
                          "en": sentence(rng), "tokens": 24, "iteration": k % 3, "position": k, "qe0": 1.0 + k / 100,
                          "l0": 30, "logit_c2_0": 0.0, "hypothesis0": f"anker {part} {k}"})
        plan7.append({"id": f"{part}:val", "part": part, "stratum": "", "doc": "v", "en": sentence(rng), "tokens": 24,
                      "iteration": -1, "position": 0, "qe0": 1.0, "l0": 30, "logit_c2_0": 0.0, "hypothesis0": "val"})
    fresh, anchors = [], {}
    for part, n in {"flan": 14, "novels": 33, "bt": 43}.items():
        for k in range(n):
            i = f"{part}:new{k}"
            fresh.append({"id": i, "part": part, "stratum": "s1" if part == "bt" else "", "doc": f"{part}:n", "en": sentence(rng),
                          "tokens": 24, "rank": k})
            anchors[i] = {"source": fresh[-1]["en"], "hypothesis": f"nieuw anker {i}", "tokens_out": 20 + k, "ended": True,
                          "decode_loop": False, "empty": False, "english_left": False}
    return plan7, fresh, anchors


def test_the_plan_takes_g47s_sources_and_the_first_fresh_ones_with_a_good_anchor(small):
    assert small == {"flan": 20, "novels": 90, "bt": 90}
    plan7, fresh, anchors = plan_inputs()
    anchors["flan:new0"]["ended"] = False
    anchors["flan:new2"]["english_left"] = True
    rows, report = g8.plan_rows(plan7, fresh, anchors)
    assert len(rows) == 200 and report["sources"] == 200
    flan_fresh = [r["id"] for r in rows if r["part"] == "flan" and r["origin"] == "fresh"]
    assert sorted(flan_fresh, key=lambda i: int(i.split("new")[1])) == [f"flan:new{k}" for k in (1, 3, 4, 5, 6, 7, 8, 9, 10, 11)]
    assert report["anchor_dropped"]["flan"] == {"ended": 1, "decode_loop": 0, "empty": 0, "english_left": 1}
    assert not any(r["id"].endswith(":val") for r in rows)
    assert report["chunks"] == {"c1": {"flan": 4, "novels": 18, "bt": 18}, "c2": {"flan": 8, "novels": 36, "bt": 36},
                                "c3": {"flan": 8, "novels": 36, "bt": 36}}
    old = {r["id"]: r for r in rows if r["origin"] == "g4-7"}
    assert old["flan:old3"]["hypothesis0"] == "anker flan 3" and old["flan:old3"]["qe0_g47"] == pytest.approx(1.03)
    new = {r["id"]: r for r in rows if r["origin"] == "fresh"}
    assert new["bt:new4"]["l0"] == 24 and new["bt:new4"]["hypothesis0"] == "nieuw anker bt:new4"
    assert new["bt:new4"]["qe0_g47"] is None
    again, _ = g8.plan_rows(plan7, fresh, anchors)
    assert [(r["id"], r["chunk"]) for r in again] == [(r["id"], r["chunk"]) for r in rows]
    for k in range(4, 14):
        anchors[f"flan:new{k}"]["empty"] = True
    with pytest.raises(SystemExit, match="flan has 2 fresh sources with a good anchor, 10 needed"):
        g8.plan_rows(plan7, fresh, anchors)


# ---------------------------------------------------------------------------------------------------- the chunk stages

def write_samples(root, rows, shard_size=16):
    """Sample shards as translate.py --copies writes them (PROTOCOL_COLUMNS), with DONE, per worker directory"""
    by = {}
    for r in rows:
        by.setdefault(r["worker"], []).append({k: r[k] for k in tr.PROTOCOL_COLUMNS})
    for w, part in by.items():
        d = Path(root) / w
        d.mkdir(parents=True)
        digests = {}
        for k in range(0, len(part), shard_size):
            name = f"shard-{k // shard_size:05d}.parquet"
            pq.write_table(pa.Table.from_pylist(part[k:k + shard_size]), d / name)
            digests[name] = (len(part[k:k + shard_size]), hashlib.sha256((d / name).read_bytes()).hexdigest())
        (d / "DONE").write_text(json.dumps(tr.done_marker(list(digests), digests)))


def sample(i, j, src, hyp, worker, tokens_out=20, **flags):
    return {"uid": f"{i}#{j}", "id": i, "j": j, "source": src, "hypothesis": hyp, "token_ids": [1] * tokens_out,
            "sampler_logp": [-0.1] * tokens_out, "ended": True, "decode_loop": False, "english_left": False,
            "empty": not hyp.strip(), "tokens_in": 50, "tokens_out": tokens_out, "worker": worker, **flags}


def chunk_world(tmp_path):
    """A plan of 5 sources in c1 (and one in c2) with designed samples, QE and judge probabilities:
    A: samples 1 and 2 the same text (one judged key), 3 the anchor's text, 4 not ended, 5 QE > QE0, 6 too long;
       P̄(j1) 0.8, P̄(j7) 0.7 -> target j1
    B: two candidates, P̄ 0.45 and 0.40 -> dropped ('p')
    C: every sample the anchor's text or a fail -> no candidate ('none')
    D: two candidates with the same P̄ 0.9 -> the lower QE (j5)
    E: one candidate whose prompts were too long (not scored) and one with P̄ 0.6 -> j2"""
    rows = []
    for i, part, origin in (("A", "novels", "g4-7"), ("B", "bt", "fresh"), ("C", "flan", "g4-7"), ("D", "bt", "g4-7"),
                            ("E", "novels", "fresh")):
        rows.append({"id": i, "part": part, "stratum": "enwiki" if part == "bt" else "", "doc": f"d{i}",
                     "en": f"source {i} in English", "tokens": 24, "origin": origin, "chunk": "c1",
                     "hypothesis0": f"anker {i}", "l0": 20, "qe0_g47": 2.0 if origin == "g4-7" else None})
    rows.append({"id": "Z", "part": "bt", "stratum": "news", "doc": "dZ", "en": "other chunk", "tokens": 24, "origin": "fresh",
                 "chunk": "c2", "hypothesis0": "anker Z", "l0": 20, "qe0_g47": None})
    plan = tmp_path / "plan.parquet"
    g8.write_parquet(rows, plan, g8.PLAN_SCHEMA)
    src = {r["id"]: r["en"] for r in rows}
    samples, qe = [], {}
    texts = {"A": ["a0", "a1", "a1 ", "anker A", "a4", "a5", "a6", "a7"],
             "B": ["b0", "b1", "anker B", "anker B", "anker  B", "anker B", "anker B", "anker B"],
             "C": ["anker C"] * 7 + [""],
             "D": ["d0", "anker D", "anker D", "anker D", "anker D", "d5", "anker D", "anker D"],
             "E": ["e0", "anker E", "e2", "anker E", "anker E", "anker E", "anker E", "anker E"]}
    for i, hyps in texts.items():
        for j, h in enumerate(hyps):
            flags = {}
            if i == "A" and j == 4:
                flags["ended"] = False
            tokens = 40 if (i == "A" and j == 6) else 20      # ratio 2.0: the length fail
            samples.append(sample(i, j, src[i], h, "w0" if i in "AB" else "w1", tokens_out=tokens, **flags))
            qe[f"{i}#{j}"] = None if not h.strip() else 1.0
        qe[f"{i}#a"] = 2.0
    qe["A#5"] = 2.5                                             # QE > QE0
    qe["A#0"] = 1.9
    qe["D#0"], qe["D#5"] = 1.5, 1.2
    write_samples(tmp_path / "samples", samples)
    return plan, qe


def judge_scores(prompts, p_sample, skipped=()):
    """Score rows as pairjudge.py score writes them: per key in order ab (A = sample) and ba"""
    out = []
    for r in prompts:
        key, order = r["uid"].rsplit("/", 1)
        if key in skipped:
            out.append({"uid": r["uid"], "skipped": "too_long", "p_a": None, "p_b": None, "p_eq": None})
            continue
        ps = p_sample[key]
        pa_, pb_ = (ps, 1 - ps - 0.1) if order == "ab" else (1 - ps - 0.1, ps)
        out.append({"uid": r["uid"], "skipped": None, "p_a": pa_, "p_b": pb_, "p_eq": 0.1})
    return out


def test_the_chunk_stages_choose_janchor_and_build_v3_and_the_report(tmp_path):
    plan, qe = chunk_world(tmp_path)
    for w in ("w0", "w1"):
        (tmp_path / f"one-{w}" / w).mkdir(parents=True)
        for f in (tmp_path / "samples" / w).iterdir():
            (tmp_path / f"one-{w}" / w / f.name).write_bytes(f.read_bytes())
        g8.main(["qepairs", "--plan", str(plan), "--chunk", "c1", "--samples", str(tmp_path / f"one-{w}"),
                 "--out", str(tmp_path / f"qepairs-{w}.parquet")])
    w0 = pq.read_table(tmp_path / "qepairs-w0.parquet").to_pylist()
    assert sorted(r["id"] for r in w0) == sorted([f"{i}#{j}" for i in "AB" for j in range(8)] + ["A#a", "B#a"])
    assert {r["id"]: r for r in w0}["A#a"]["hypothesis"] == "anker A"
    with pytest.raises(SystemExit, match="not of chunk c2"):
        g8.main(["qepairs", "--plan", str(plan), "--chunk", "c2", "--samples", str(tmp_path / "one-w0"),
                 "--out", str(tmp_path / "x.parquet")])
    for w, ids in (("w0", "AB"), ("w1", "CDE")):
        pq.write_table(pa.Table.from_pylist([{"id": u, "qe_source_hypothesis": v} for u, v in qe.items() if u[0] in ids]),
                       tmp_path / f"qe-{w}.parquet")
    out = tmp_path / "pairs"
    g8.main(["pairs", "--plan", str(plan), "--chunk", "c1", "--samples", str(tmp_path / "samples"),
             "--qe", str(tmp_path / "qe-w*.parquet"), "--out", str(out)])
    prompts = [r for k in range(4) for r in pq.read_table(out / f"prompts-w{k}.parquet").to_pylist()]
    keys = sorted({r["uid"].rsplit("/", 1)[0] for r in prompts})
    # A: j1 and j2 one key, j3 the anchor, j4 not ended, j5 QE > QE0, j6 too long; B: j4 differs only in whitespace
    assert keys == ["A/j0", "A/j1", "A/j7", "B/j0", "B/j1", "D/j0", "D/j5", "E/j0", "E/j2"]
    by_uid = {r["uid"]: r["text"] for r in prompts}
    assert "Vertaling A:\na0\n\nVertaling B:\nanker A" in by_uid["A/j0/ab"]
    assert "Vertaling A:\nanker A\n\nVertaling B:\na0" in by_uid["A/j0/ba"]
    cands = {r["uid"]: r for r in pq.read_table(out / "cands.parquet").to_pylist()}
    assert cands["A#4"]["fail"] == "not_ended" and cands["A#6"]["fail"] == "length" and cands["C#7"]["fail"] == "empty"
    assert cands["A#2"]["key"] == "A/j1" and cands["B#4"]["same_as_anchor"] and not cands["A#5"]["candidate"]
    summary = json.loads((out / "pairs.json").read_text())
    assert summary["qe0_check"]["n"] == 3 and summary["qe0_check"]["median_abs"] == 0.0
    assert summary["parts"]["novels"]["keys"] == 5 and summary["parts"]["flan"]["candidates"] == 0
    p_sample = {"A/j0": 0.6, "A/j1": 0.8, "A/j7": 0.7, "B/j0": 0.45, "B/j1": 0.40, "D/j0": 0.9, "D/j5": 0.9,
                "E/j2": 0.6}
    d = tmp_path / "scores" / "w0"
    d.mkdir(parents=True)
    rows = judge_scores(prompts, p_sample, skipped=("E/j0",))
    pq.write_table(pa.Table.from_pylist(rows[:7]), d / "shard-00000.parquet")
    pq.write_table(pa.Table.from_pylist(rows[7:]), d / "shard-00001.parquet")
    sel = tmp_path / "select"
    g8.main(["select", "--plan", str(plan), "--chunk", "c1", "--samples", str(tmp_path / "samples"), "--cands",
             str(out / "cands.parquet"), "--scores", str(tmp_path / "scores/w*/shard-*.parquet"), "--out", str(sel)])
    choice = {r["id"]: r for r in pq.read_table(sel / "choice.parquet").to_pylist()}
    assert {i: (r["reason"], r["j"]) for i, r in choice.items()} == {
        "A": ("sample", 1), "B": ("p", -1), "C": ("none", -1), "D": ("sample", 5), "E": ("sample", 2)}
    assert choice["E"]["not_scored"] == 1 and choice["B"]["p_best"] == pytest.approx(0.45)
    assert choice["B"]["p_sample"] is None and choice["A"]["p_sample"] == pytest.approx(0.8)
    targets = {r["id"]: r for r in pq.read_table(sel / "targets.parquet").to_pylist()}
    assert {i: r["target"] for i, r in targets.items()} == {"A": "a1", "D": "d5", "E": "e2"}
    assert targets["D"]["qe"] == 1.2 and targets["A"]["chrf"] == pytest.approx(g8.chrf("a1", "anker A"))
    # a missing judge score is an error, not a fallback
    (d / "shard-00001.parquet").unlink()
    with pytest.raises(SystemExit, match="prompts missing"):
        g8.main(["select", "--plan", str(plan), "--chunk", "c1", "--samples", str(tmp_path / "samples"), "--cands",
                 str(out / "cands.parquet"), "--scores", str(tmp_path / "scores/w*/shard-*.parquet"), "--out",
                 str(tmp_path / "select2")])
    v3 = tmp_path / "v3"
    g8.main(["v3", "--plan", str(plan), "--targets", str(sel / "targets.parquet"), "--out", str(v3)])
    keys_ = [json.loads(x) for x in (v3 / "key.jsonl").read_text().splitlines()]
    inputs = [json.loads(x) for x in (v3 / "judge-input.jsonl").read_text().splitlines()]
    assert [k["blind_id"] for k in keys_] == ["v3-001", "v3-002", "v3-003"]
    assert [k["set"] for k in keys_] == ["bt", "novels", "novels"]
    for k, x in zip(keys_, inputs):
        own = targets[k["item"]]["target"]
        assert x[k["A"] == "base" and "B" or "A"] == own and {k["A"], k["B"]} == {"base", g8.V3_TARGET}
        assert set(x) == {"blind_id", "source", "A", "B"} and "target" not in k
    rep = json.loads((v3 / "select-report.json").read_text())
    assert rep["sets"]["novels"] == {"targets": 2, "picked": 2, "seed": "en-nl-g4arm/0/G4-8/v3/novels", "strata": {"": 2}}
    # the report over the chunks (here one chunk whose plan is all of a smaller plan)
    rows = [r for r in pq.read_table(plan).to_pylist() if r["chunk"] == "c1"]
    small_plan = tmp_path / "plan-c1.parquet"
    g8.write_parquet(rows, small_plan, g8.PLAN_SCHEMA)
    c1 = tmp_path / "c1"
    c1.mkdir()
    for f in ("choice.parquet", "targets.parquet"):
        (c1 / f).write_bytes((sel / f).read_bytes())
    for f in ("cands.parquet", "pairs.json"):
        (c1 / f).write_bytes((out / f).read_bytes())
    g8.main(["report", "--plan", str(small_plan), "--chunk-dir", f"c1={c1}", "--out", str(tmp_path / "report")])
    r = json.loads((tmp_path / "report/report.json").read_text())
    assert r["parts"]["samen"]["kept"] == 3 and r["parts"]["samen"]["kept_share"] == pytest.approx(0.6)
    assert r["parts"]["bt"]["kept_share"] == pytest.approx(0.5) and r["parts"]["flan"]["no_candidate"] == 1
    assert r["parts"]["novels"]["fail"]["length"] == pytest.approx(1 / 16)
    assert r["origin"]["bt/fresh"] == {"sources": 1, "kept_share": 0.0}
    md = (tmp_path / "report/report.md").read_text()
    assert "gehouden" in md and not any(t in md for t in ("anker A", "anker D", "source A", "| a1", "d5 "))
    assert len(pq.read_table(tmp_path / "report/targets.parquet")) == 3
    # the plan's other chunk is not covered: an error
    with pytest.raises(SystemExit, match="the plan has 6"):
        g8.main(["report", "--plan", str(plan), "--chunk-dir", f"c1={c1}", "--out", str(tmp_path / "report2")])


def test_a_reused_anchor_whose_qe0_is_far_from_g47s_stops_pairs():
    ok = [{"qe0": 2.0, "qe0_g47": 2.01}] * 98 + [{"qe0": 2.0, "qe0_g47": 2.9}] * 2 + [{"qe0": 1.0, "qe0_g47": None}]
    stats, problem = g8.qe0_check(ok)
    assert problem is None and stats["n"] == 100 and stats["share_over"] == pytest.approx(0.02)
    stats, problem = g8.qe0_check(ok[:95] + [{"qe0": 2.0, "qe0_g47": 2.9}] * 5)
    assert "QE0 differs" in problem
    stats, problem = g8.qe0_check([{"qe0": 2.0, "qe0_g47": 2.06}] * 10)
    assert "median" in problem


def test_fail_is_g47s_without_c2_overflow():
    s = {"tokens_out": 20, "ended": True, "decode_loop": False, "empty": False, "english_left": False}
    assert g8.fails8(s, 2.0, 20) == ([], 1.0)
    assert g8.fails8(s, 7.6, 20)[0] == ["qe"]
    assert g8.fails8({**s, "tokens_out": 31}, 2.0, 20)[0] == ["length"]
    assert g8.fails8({**s, "tokens_out": 13}, 2.0, 20)[0] == ["length"]      # 0.65 < 0.67
    assert g8.fails8({**s, "empty": True, "tokens_out": 0}, None, 20)[0] == ["empty", "length"]
    with pytest.raises(SystemExit, match="no QE"):
        g8.fails8(s, None, 20)


# ---------------------------------------------------------------------------------------------------- run_g4arm8.sh

FAKE_GSUTIL = r"""#!/bin/bash
# gs://X -> $FAKEGS/X; stat only (the stages' jobs are rendered, not run); every call in $FAKELOG
echo "gsutil $*" >> "$FAKELOG"
args=(); for a in "$@"; do case $a in -q|-m) ;; *) args+=("$a") ;; esac; done
case ${args[0]} in
  stat) [ -e "$FAKEGS/${args[1]#gs://}" ] ;;
  *) echo "fake gsutil: ${args[0]}" >&2; exit 2 ;;
esac
"""


def g8_world(tmp_path):
    shm, gs, home, fake, code = (tmp_path / x for x in ("shm", "gs", "home", "bin", "code"))
    for d in (code, home, fake):
        d.mkdir(parents=True)
    (code / "run_g4arm8.sh").write_text((CODE / "run_g4arm8.sh").read_text().replace("/dev/shm/", f"{shm}/"))
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
    (shm / "translator/metricx/metricx-24-hybrid-xxl-v2p6-bfloat16").mkdir(parents=True)
    g8dir = gs / "t5-dutch-english-usc2/mbert/translator/sft/g4arm/data/G4-8"
    calls = tmp_path / "calls.log"

    def put(rel, text="x"):
        p = gs / "t5-dutch-english-usc2/mbert/translator" / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)

    def run(stage, host="0", **env):
        (fake / "hostname").write_text(f"#!/bin/bash\necho t1v-n-x-w-{host}\n")
        e = {"HOME": str(home), "PATH": f"{fake}:{os.environ['PATH']}", "FAKEGS": str(gs), "FAKELOG": str(calls),
             "TAG": "t1", **env}
        calls.write_text("")
        out = subprocess.run(["bash", str(code / "run_g4arm8.sh"), stage], capture_output=True, text=True, timeout=60,
                             env=e, cwd=home)
        return out

    def body(name):
        text = (home / f"job_body_{name}.sh").read_text()
        check = subprocess.run(["bash", "-n", str(home / f"job_body_{name}.sh")], capture_output=True, text=True)
        assert check.returncode == 0, check.stderr
        return text

    return run, body, put, home, g8dir


def test_run_g4arm8_stages_render_their_jobs(tmp_path):
    run, body, put, home, g8dir = g8_world(tmp_path)
    out = run("fresh", TAG="f1")
    assert out.returncode == 0 and "started g4a8_fresh_f1 on w0" in out.stdout, out.stdout + out.stderr
    b = body("g4a8_fresh_f1")
    assert "g4arm8.py fresh --g47-plan" in b and "/rl/G4-7/data/plan/plan.parquet" in b and "/rl/G4-7/c2/data/docs.txt" in b
    assert "blind100.jsonl" in b and "/data/G4-8/fresh/fresh.json" in b and "rm " not in b
    assert run("fresh", host="2", TAG="f2").stdout.strip().endswith("runs on w0 only")
    # the anchor: greedy, 1 copy, the fresh segments of its worker
    put("sft/g4arm/data/G4-8/fresh/segments-w2.parquet")
    out = run("generate", host="2", JOB="anchor", TAG="a1")
    assert out.returncode == 0 and "started g4a8_gen_anchor_a1 on w2" in out.stdout, out.stdout + out.stderr
    b = body("g4a8_gen_anchor_a1")
    assert "--copies 1 " in b and "--sample" not in b and "/G4-8/anchor/samples/w2 " not in b
    assert "--out '" in b and "/G4-8/anchor/samples/w2'" in b and "--length-class medium" in b
    assert "--prompt natural-v1" in b and "--model gemma4-31b" in b and "--lora" not in b and "rm " not in b
    # a chunk: 8 samples at T = 1, the seed of the chunk and worker
    seeds = {}
    for c, host in (("c1", "0"), ("c1", "3"), ("c2", "0")):
        put(f"sft/g4arm/data/G4-8/{c}/segments/segments-w{host}.parquet")
        out = run("generate", host=host, JOB=c, TAG=f"g{c}{host}")
        assert out.returncode == 0, out.stdout + out.stderr
        b = body(f"g4a8_gen_{c}_g{c}{host}")
        seed = g4rl.seed_int(f"en-nl-g4arm/0/G4-8/sample/{c}/w{host}")
        assert f"--copies 8 --sample samples=1.0:{seed}" in b and f"/G4-8/{c}/{{sample}}/w{host}'" in b
        assert "--shard-size 256" in b
        seeds[(c, host)] = seed
    assert len(set(seeds.values())) == 3
    # a worker whose DONE is there: a job that only says so
    put("sft/g4arm/data/G4-8/c1/samples/w0/DONE")
    out = run("generate", host="0", JOB="c1", TAG="g9")
    assert out.returncode == 0 and "started g4a8_gen_c1_g9 on w0" in out.stdout
    assert "DONE there before" in body("g4a8_gen_c1_g9")
    assert run("generate", JOB="c4", TAG="g10").returncode != 0
    out = run("plan", TAG="p1")
    b = body("g4a8_plan_p1")
    assert "g4arm8.py plan --g47-plan" in b and "/G4-8/anchor/samples " in b and "for c in c1 c2 c3" in b
    out = run("qe", host="1", CHUNK="c2", TAG="q1")
    assert out.returncode != 0 and "no " in out.stdout and "/c2/samples/w1/DONE" in out.stdout
    put("sft/g4arm/data/G4-8/c2/samples/w1/DONE")
    out = run("qe", host="1", CHUNK="c2", TAG="q2")
    assert out.returncode == 0, out.stdout + out.stderr
    b = body("g4a8_qe_c2_q2")
    assert "g4arm8.py qepairs --plan" in b and "--chunk c2" in b and "metricx_qe.py score" in b
    assert "--pair source:hypothesis --model xxl" in b and "--empty-candidates" not in b and "/c2/qe/qe-w1.parquet" in b
    assert run("qe", CHUNK="c5", TAG="q3").returncode != 0
    out = run("pairs", CHUNK="c1", TAG="r1")
    b = body("g4a8_pairs_c1_r1")
    assert "g4arm8.py pairs" in b and "'" + str(tmp_path) in b and "/c1/judge/data/pairs.json" in b
    put("sft/g4arm/data/G4-8/c3/judge/data/prompts-w3.parquet")
    out = run("judge", host="3", CHUNK="c3", TAG="j1")
    assert out.returncode == 0, out.stdout + out.stderr
    b = body("g4a8_judge_c3_j1")
    assert "pairjudge.py score --prompts" in b and "/c3/judge/scores/w3 " in b and "TPU_VISIBLE_CHIPS=0,1,2,3" in b
    out = run("select", CHUNK="c1", TAG="s1")
    b = body("g4a8_select_c1_s1")
    assert "g4arm8.py select" in b and all(f"/c1/judge/scores/w{k}/shard-*.parquet" not in b for k in range(4))
    assert '/c1/judge/scores/w$k/shard-*.parquet' in b and "for k in 0 1 2 3" in b
    out = run("v3", TAG="v1")
    assert out.returncode != 0 and "c1/select/targets.parquet" in out.stdout
    put("sft/g4arm/data/G4-8/c1/select/targets.parquet")
    out = run("v3", TAG="v2")
    assert out.returncode == 0, out.stdout + out.stderr
    b = body("g4a8_v3_v2")
    assert "g4arm8.py v3" in b and "/sft/blind/G4-8-V3/key.jsonl" in b
    put("sft/blind/G4-8-V3/key.jsonl")
    out = run("v3", TAG="v3")
    assert out.returncode != 0 and "stays as it is" in out.stdout
    out = run("report", TAG="x1")
    b = body("g4a8_report_x1")
    assert b.count("--chunk-dir ") == 3 and "/G4-8/targets/targets.parquet" in b and "/G4-8/report/" in b
    # every rendered job copies its log to $G8/logs
    job = (home / "job_g4a8_report_x1.sh").read_text()
    assert job.startswith("bash ")
    assert re.search(r"/data/G4-8/fresh/", (home / "job_body_g4a8_fresh_f1.sh").read_text())
