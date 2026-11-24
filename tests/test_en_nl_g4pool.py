"""The evaluation pool (protocol-g4.md §24, part 2): g4pool.py data (the strata, their checks, filters and exclusions,
FLAN <= 10%), texts, cands, choose (the four best-of-8 choices and the pairs of a later human trial) and report (the
reproductions, the tables, no text) on small synthetic inputs (no model, no TPU)."""

import json
import random
import sys
from argparse import Namespace
from pathlib import Path

import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "experiments" / "en-nl-translator"
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(ROOT / "experiments" / "flan-nl"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import g4arm  # noqa: E402
import g4mixsample as mx  # noqa: E402
import g4pool as gp  # noqa: E402
import g4rl  # noqa: E402
import pairjudge  # noqa: E402
from test_en_nl_g4arm8 import sample, write_samples  # noqa: E402
from test_en_nl_g4measure import write_pairs  # noqa: E402
from test_en_nl_g4rl import FakeCounter7  # noqa: E402

WORDS = ("the", "house", "river", "green", "walks", "slowly", "and", "city", "old", "light", "water", "sees", "a", "dog",
         "runs", "far", "away", "from", "home", "today", "bright", "stone", "bridge", "under", "over", "night")
SOURCES = ("dpc", "fineweb", "handelingen", "raad", "rechtspraak")


def sentence(rng, n=24):
    return " ".join(rng.choice(WORDS) for _ in range(n)) + f" w{rng.randrange(10 ** 9)}"


def test_the_pool_is_fixed_as_section_24_has_it():
    text = (CODE / "protocol-g4.md").read_text(encoding="utf-8")
    section = text[text.index("### 24.2 Deel 2"):]
    assert "**romans 100**" in section and "10 per boek" in section and "20 per bron" in section
    assert "**WMT24++ 100**, **FLORES 40**, **NTREX 40**, **FLAN 40**" in section
    assert (gp.NOVELS_PER_BOOK, gp.BT_PER_SOURCE, gp.EVAL_COUNTS, gp.FLAN_MAX) == (10, 20, {"wmt24pp": 100, "flores": 40,
                                                                                           "ntrex": 40}, 40)
    assert "FLAN = min(40, ⌊(alle andere bronnen) / 9⌋)" in section
    assert '`random.Random("en-nl-g4arm/0/G4-pool/repro-qe")' in section and "≤ **0,06**" in section
    assert gp.REPRO_QE_SEED == "en-nl-g4arm/0/G4-pool/repro-qe" and gp.REPRO_QE_TOL == 0.06
    assert gp.REPRO_JUDGE_SEED == "en-nl-g4arm/0/G4-pool/repro-judge" and (gp.REPRO_QE_N, gp.REPRO_JUDGE_N) == (200, 100)


# ---------------------------------------------------------------------------------------------------- data

def data_world(tmp, rng, monkeypatch):
    novels_meta, b1 = [], []
    for k in range(mx.NOVEL_PAIRS):
        book, pid = str(900 + k % mx.NOVEL_BOOKS), str(10_000 + k)
        novels_meta.append({"pid": pid, "en_book_id": book})
        b1.append({"id": pid, "group": "en_orig", "en_book_id": book, "en": sentence(rng), "nl": "nl " + sentence(rng)})
    b1[499]["en"] = " ".join(["word"] * 300)              # above 256 tokens: filtered
    g4rl.write_parquet(novels_meta, tmp / "meta.parquet")
    g4rl.write_parquet(b1, tmp / "b1.parquet")
    blocked = [{"id": f"x{k}", "doc": f"dx{k}", "en": sentence(rng)} for k in range(10)]
    for name in ("pool", "fresh", "plan", "targets"):
        g4rl.write_parquet(blocked, tmp / f"{name}.parquet")
    g4rl.write_parquet([{"pid": "p", "source": "dpc", "split": "train", "label": 1, "text": "t"}], tmp / "c2.parquet")
    fb = [{"test": "bt", "id": f"{s}:doc{k}:d:0", "stratum": s, "en": sentence(rng), "first": "origineel " + sentence(rng),
           "second": "y"} for s in SOURCES for k in range(100)]
    monkeypatch.setattr(pairjudge, "fb_bt", lambda *a: (fb, {"pairs": len(fb)}))
    monkeypatch.setattr(g4arm, "revisions3", lambda d: {})
    monkeypatch.setattr(g4arm, "read_bt", lambda *a, **k: [])
    ev, assign = [], {}
    for s, n in (("wmt24pp", 160), ("flores", 60), ("ntrex", 60), ("flan", 80)):
        for k in range(n):
            i = f"flan:0-{k}" if s == "flan" else f"{s}-{k:04d}"
            meta = {"is_bad_source": k == 3} if s == "wmt24pp" else {}
            ev.append({"id": i, "set": s, "source": sentence(rng), "reference": None if s == "flan" else "ref " + sentence(rng),
                       "meta": json.dumps(meta)})
            assign[i] = "decision" if k % 4 == 0 else "search"
    for k in range(1, 101):
        bs = "idiom" if k > 70 else "wmt24pp"
        ev.append({"id": f"blind-{k:03d}", "set": "blind", "source": sentence(rng), "reference": None,
                   "meta": json.dumps({"blind_set": bs, "item": "wmt24pp-0001" if k == 1 else f"x:{k}"})})
    pairjudge.write_rows(ev, tmp / "evalsets.parquet")
    (tmp / "split.json").write_text("{}")
    monkeypatch.setattr(mx, "split_assign", lambda path: assign)
    mix_ids = {"novels": [f"novels:{10_000 + k}" for k in range(60)],
               "bt": [f"bt:{s}:doc{k}:d:0" for s in SOURCES for k in range(12)]}
    (tmp / "mix.json").write_text(json.dumps({"novels": {"ids": mix_ids["novels"]}, "bt": {"ids": mix_ids["bt"]}}))
    (tmp / "mixkey.json").write_text(json.dumps({"pairs": [{"set": "wmt24pp", "item": "wmt24pp-0002"},
                                                           {"set": "idiom", "item": "blind-071"}]}))
    (tmp / "yeb1.json").write_text(json.dumps([{"n": 1, "package": "G4-judge-i15", "blind_id": "judge-001"}]))
    (tmp / "pk.jsonl").write_text(json.dumps({"blind_id": "judge-001", "set": "flan", "item": "flan:0-1"}) + "\n")
    (tmp / "s2.json").write_text(json.dumps([{"n": 1, "blind_id": "blind-072"}, {"n": 2, "blind_id": "blind-005"}]))
    return Namespace(novels_meta=str(tmp / "meta.parquet"), b1=str(tmp / "b1.parquet"), g47_pool=str(tmp / "pool.parquet"),
                     g48_fresh=str(tmp / "fresh.parquet"), g48_plan=str(tmp / "plan.parquet"),
                     g48_targets=str(tmp / "targets.parquet"), c2_data=str(tmp / "c2.parquet"), bt=str(tmp / "bt"),
                     evalsets=str(tmp / "evalsets.parquet"), split=str(tmp / "split.json"), mix_data=str(tmp / "mix.json"),
                     mix_key=str(tmp / "mixkey.json"), yeb1_key=str(tmp / "yeb1.json"),
                     package_key=[f"G4-judge-i15={tmp / 'pk.jsonl'}"], sample2_key=str(tmp / "s2.json"), tokenizer="x",
                     out=str(tmp / "out"))


def test_data_draws_every_stratum_with_its_exclusions_and_keeps_flan_at_10_percent(tmp_path, monkeypatch, capsys):
    args = data_world(tmp_path, random.Random(3), monkeypatch)
    info = gp.data(args, counter=FakeCounter7())
    parts = info["parts"]
    assert {p: parts[p]["drawn"] for p in gp.PARTS} == {"novels": 100, "bt": 100, "wmt24pp": 100, "flores": 40,
                                                         "ntrex": 40, "idiom": 28, "flan": 40}
    assert info["total"] == 448 and info["flan_share"] <= 0.10
    # G4-8.15's 60 + 60 are out; the long novel passage is filtered
    assert parts["novels"]["without_mix"] == 60 and parts["bt"]["without_mix"] == 60
    assert parts["novels"]["filter"]["tokens"] == 1
    assert not set(parts["novels"]["ids"]) & {f"novels:{10_000 + k}" for k in range(60)}
    # the decision part, a bad source, a fixed blind item and Yeb's items are out
    assert parts["wmt24pp"]["excluded"] == {"decision_or_none": 40, "bad_source": 1, "fixed_blind_item": 1, "yeb": 1}
    assert parts["flan"]["excluded"]["yeb"] == 1
    assert "blind-071" not in parts["idiom"]["ids"] and "blind-072" not in parts["idiom"]["ids"]
    rows = pq.read_table(tmp_path / "out" / "sources.parquet").to_pylist()
    by = {r["id"]: r for r in rows}
    assert by[parts["novels"]["ids"][0]]["human"].startswith("nl ")
    assert by[parts["bt"]["ids"][0]]["human"].startswith("origineel ")
    assert by[parts["wmt24pp"]["ids"][0]]["human"].startswith("ref ") and by[parts["flan"]["ids"][0]]["human"] is None
    assert all(20 <= r["tokens"] <= 256 for r in rows)
    meta = pq.read_table(tmp_path / "out" / "sources-meta.parquet").to_pylist()
    assert "en" not in meta[0] and "human" not in meta[0] and len(meta) == 448
    seg = pq.read_table(tmp_path / "out" / "segments.parquet").to_pylist()
    assert [s["id"] for s in seg] == [r["id"] for r in rows]
    assert "POOLDATA" in capsys.readouterr().out


def test_data_stops_on_a_source_of_g48s_data(tmp_path, monkeypatch):
    args = data_world(tmp_path, random.Random(4), monkeypatch)
    rows = pq.read_table(tmp_path / "targets.parquet").to_pylist()
    rows.append({"id": "novels:10100", "doc": "zz", "en": "zz"})
    g4rl.write_parquet(rows, tmp_path / "targets.parquet")
    with pytest.raises(SystemExit, match="novels: sources in G4-7's pool or G4-8's"):
        gp.data(args, counter=FakeCounter7())


# ---------------------------------------------------------------------------------------------------- the choices

def cand(key, j, qe=1.0, candidate=True):
    return {"key": key if candidate else None, "j": j, "qe": qe, "candidate": candidate}


def p(a, b):
    return {"A": a, "B": b, "gelijk": 1 - a - b}


def test_a1_veto_n1_lets_a1_decide_at_a_clear_margin_and_n1s_rule_elsewhere():
    rows = [cand("S/j1", 1), cand("S/j2", 2), cand("S/j3", 3, qe=0.5), cand(None, 4, candidate=False)]
    n1 = {"S/j1": p(0.9, 0.05), "S/j2": p(0.6, 0.3), "S/j3": p(0.4, 0.5)}
    # j1: A1 clearly for the anchor (veto); j2: A1 unclear, N1 0.6 > 0.5; j3: A1 clearly for the sample
    a1 = {"S/j1": p(0.1, 0.8), "S/j2": p(0.4, 0.35), "S/j3": p(0.7, 0.2)}
    k, reason = gp.choose_a1veto(rows, n1, a1)
    assert reason == "sample" and k["j"] == 2                      # the highest P̄_N1 of j2 (0.6) and j3 (0.4)
    k, reason = gp.choose_a1veto(rows[:1], n1, a1)
    assert k is None and reason == "p"
    k, reason = gp.choose_a1veto(rows[3:], n1, a1)
    assert k is None and reason == "none"
    # rubric ∩ N1: the same sample only
    rubric = {"S/j1": p(0.3, 0.6), "S/j2": p(0.7, 0.2), "S/j3": p(0.2, 0.7)}
    c = gp.choices_of(rows, {"rubric": rubric, "n1": n1, "a1": a1})
    assert c["bo8-rubriek"][0]["j"] == 2 and c["bo8-N1"][0]["j"] == 1 and c["bo8-rubriek∩N1"] == (None, "differ")
    rubric["S/j1"] = p(0.95, 0.0)
    c = gp.choices_of(rows, {"rubric": rubric, "n1": n1, "a1": a1})
    assert c["bo8-rubriek∩N1"][0]["j"] == 1 and c["bo8-rubriek∩N1"][1] == "sample"


# ---------------------------------------------------------------------------------------------------- texts .. report

SRC = {"novels:1": "novels", "bt:dpc:d1:0": "bt", "wmt24pp-0001": "wmt24pp", "wmt24pp-0002": "wmt24pp", "flan:0-1": "flan",
       "blind-071": "idiom"}


def pipeline_world(tmp, monkeypatch):
    rng = random.Random(9)
    monkeypatch.setattr(gp, "REPRO_QE_N", 2)
    monkeypatch.setattr(gp, "REPRO_JUDGE_N", 2)
    rows = []
    for i, part in SRC.items():
        human = None if part in ("flan", "idiom") else f"mens {i}"
        rows.append({"id": i, "part": part, "stratum": part, "doc": i, "en": f"source {i}", "human": human, "tokens": 30})
    data = tmp / "data"
    gp.write(rows, data / "sources.parquet", gp.SOURCE_FIELDS)
    gp.write([{k: r[k] for k in ("id", "part", "stratum", "tokens")} | {"has_human": r["human"] is not None} for r in rows],
             data / "sources-meta.parquet")
    (data / "data.json").write_text(json.dumps({"parts": {p: {"sources": 1, "kept": 1, "drawn": 1, "filter": {"tokens": 0},
                                                              "excluded": {}} for p in gp.PARTS},
                                                "flan_share": 1 / 6, "with_human": 4}))
    # samples: wmt24pp-0001 has two candidates (j1, j2), j3 the anchor's text; the others one candidate (j1)
    samples, greedy = [], {m: [] for m in gp.GREEDY}
    for i in SRC:
        anchor = f"anker {i}"
        for j in range(8):
            hyp = f"s{j} {i}" if j in (1, 2) else anchor
            samples.append(sample(i, j, f"source {i}", hyp, "w0"))
        for k, m in enumerate(gp.GREEDY):
            hyp = anchor if m == "G4-0" else (f"s1 {i}" if m == "G4-8-s140" else f"i15 {i}")
            greedy[m].append(sample(i, 0, f"source {i}", hyp, f"w{k + 1}"))
    write_samples(tmp / "samples", samples)
    for m, rs in greedy.items():
        write_samples(tmp / "gen" / m, rs)
    # the evaluation's G4-0 output: the same text for wmt24pp-0001, another for -0002
    write_pairs(tmp / "eval" / "G4-0" / "pairs-w0.parquet", [("wmt24pp-0001", "wmt24pp", "s", "anker wmt24pp-0001"),
                                                            ("wmt24pp-0002", "wmt24pp", "s", "iets anders")])
    # §23's QE input and its scores, §23's prompts and judge scores (for the reproductions)
    m23_pairs = ["G4-7/blind-001", "G4-7/blind-002", "mix/1"]
    gp.write([{"id": f"{q}/{s}", "source": "x", "hypothesis": f"h {q} {s}"} for q in m23_pairs for s in "AB"],
             tmp / "m23" / "qe-in.parquet")
    gp.write([{"id": f"{q}/{s}", "qe_source_hypothesis": 2.0} for q in m23_pairs for s in "AB"],
             tmp / "m23" / "metricx.parquet")
    prompts, scores = [], []
    for q in m23_pairs:
        for n in ("rubric", "n1", "n2"):
            for o in ("ab", "ba"):
                prompts.append({"uid": f"{q}/{n}/{o}", "text": f"prompt {q} {n} {o}"})
                pa_, pb_ = (0.6, 0.3) if o == "ab" else (0.3, 0.6)
                scores.append({"uid": f"{q}/{n}/{o}", "skipped": None, "p_a": pa_, "p_b": pb_, "p_eq": 0.1})
    pairjudge.write_prompts(prompts, tmp / "m23" / "prompts")
    gp.write(scores, tmp / "m23" / "judge" / "shard-00000.parquet")
    return rng


def judge_rows(prompt_dir, p_of):
    """Scores of the prompts in prompt_dir: p_of(uid) -> (p_a, p_b) in order ab; ba mirrored"""
    out = []
    for k in range(4):
        for r in pq.read_table(prompt_dir / f"prompts-w{k}.parquet").to_pylist():
            key, order = r["uid"].rsplit("/", 1)
            a, b = p_of(key)
            pa_, pb_ = (a, b) if order == "ab" else (b, a)
            out.append({"uid": r["uid"], "skipped": None, "p_a": pa_, "p_b": pb_, "p_eq": round(1 - a - b, 6),
                        "mass": 0.9, "top_ids": [1]})
    return out


def test_texts_cands_choose_and_report(tmp_path, monkeypatch, capsys):
    pipeline_world(tmp_path, monkeypatch)
    t = tmp_path
    info = gp.texts(Namespace(data=str(t / "data"), samples=str(t / "samples"),
                              greedy=[f"{m}={t / 'gen' / m}" for m in gp.GREEDY],
                              eval=[f"G4-0={t / 'eval' / 'G4-0' / 'pairs-w*.parquet'}"], m23_qe=str(t / "m23" / "qe-in.parquet"),
                              out=str(t / "texts")))
    assert info["texts"] == 6 * 11 + 4 and info["qe_repro_rows"] == 4 and info["comet_rows"] == 2 * 11
    assert info["greedy_vs_eval"]["G4-0"] == {"same": 1, "differ": 1, "not_in_eval": 2}
    tmeta = pq.read_table(t / "texts" / "texts-meta.parquet").to_pylist()
    assert "text" not in tmeta[0] and {r["kind"] for r in tmeta} == set(gp.KINDS)
    nat = pq.read_table(t / "texts" / "nat-texts.parquet").to_pylist()
    assert {r["model"] for r in nat} == {"g4pool"} and len(nat) == 70
    # QE: every text 1.0, the anchor 1.5, j2 of wmt24pp-0001 above the anchor (no candidate); §23's sides 2.0 + 0.01
    qe = []
    for k in range(4):
        for r in pq.read_table(t / "texts" / f"qe-in-w{k}.parquet").to_pylist():
            v = 2.01 if r["id"].startswith(gp.REPRO) else 1.5 if r["id"].endswith("|G4-0") else 1.0
            if r["id"] == "wmt24pp-0002|s2":
                v = 1.6
            qe.append({"id": r["id"], "qe_source_hypothesis": v})
    gp.write(qe, t / "scores" / "qe-w0.parquet")
    c = gp.cands(Namespace(data=str(t / "data"), texts=str(t / "texts" / "texts.parquet"),
                           qe=str(t / "scores" / "qe-w*.parquet"), m23_prompts=str(t / "m23" / "prompts" / "prompts-w*.parquet"),
                           out=str(t / "choose"), prompts_out=str(t / "judge1")))
    # 6 sources x 2 candidates (j1, j2), minus wmt24pp-0002's j2 (QE above the anchor's); 3 prompts x 2 orders each
    assert sum(x["keys"] for x in c["parts"].values()) == 11 and c["prompts"] == 11 * 6 + 12
    # judge 1: rubric prefers j1, N1 prefers j2 for wmt24pp-0001 (else j1); A1 unclear everywhere
    def p1(key):
        if key.startswith("repro/"):
            return 0.6, 0.3
        k, name = key.rsplit("/", 1)
        if name == "a1":
            return 0.4, 0.4
        if k == "wmt24pp-0001/j2":
            return (0.6, 0.3) if name == "rubric" else (0.9, 0.05)
        return (0.8, 0.1) if name == "rubric" else (0.7, 0.2)
    gp.write(judge_rows(t / "judge1", p1), t / "judge1" / "scores" / "w0" / "shard-00000.parquet")
    ch = gp.choose(Namespace(data=str(t / "data"), texts=str(t / "texts" / "texts.parquet"),
                             cands=str(t / "choose" / "cands.parquet"), scores=str(t / "judge1" / "scores" / "w*" / "shard-*.parquet"),
                             out=str(t / "choose"), prompts_out=str(t / "judge2")))
    choice = {r["id"]: r for r in pq.read_table(t / "choose" / "choice.parquet").to_pylist()}
    w1 = choice["wmt24pp-0001"]
    assert (w1["bo8-rubriek|j"], w1["bo8-N1|j"], w1["bo8-rubriek∩N1|j"], w1["bo8-A1-veto→N1|j"]) == (1, 2, -1, 2)
    assert w1["bo8-rubriek∩N1|reason"] == "differ"
    assert all(r["bo8-rubriek∩N1|j"] == 1 for i, r in choice.items() if i != "wmt24pp-0001")
    # trial pairs: G4-8-s140 is s1 = the bo8 choice except for wmt24pp-0001 (identical, not scored); keuze only there
    assert ch["trial"]["mens"] == {"pairs": 4, "identical": 0}
    assert ch["trial"]["bo8"] == {"pairs": 6, "identical": 5} and ch["trial"]["keuze"] == {"pairs": 1, "identical": 0}
    assert ch["prompts"] == (4 + 1 + 6 + 1) * 5 * 2
    meta = pq.read_table(t / "choose" / "pairs-meta.parquet").to_pylist()
    assert "A" not in meta[0] and "source" not in meta[0]
    gp.write(judge_rows(t / "judge2", lambda key: (0.6, 0.3)), t / "judge2" / "scores" / "w0" / "shard-00000.parquet")
    tids = [r["tid"] for r in tmeta]
    gp.write([{"model": "g4pool", "set": "x", "id": x, "p_original": 0.7, "truncated": False} for x in tids], t / "c1.parquet")
    gp.write([{"model": "g4pool", "set": "x", "id": x, "p_original": 0.4, "truncated": False} for x in tids], t / "c2.parquet")
    gp.write([{"id": r["tid"], "comet22": 0.8} for r in tmeta if r["part"] in gp.NEWS and r["kind"] != "mens"],
             t / "comet.parquet")
    args = Namespace(data=str(t / "data"), texts_meta=str(t / "texts" / "texts-meta.parquet"),
                     texts_json=str(t / "texts" / "texts.json"), cands_json=str(t / "choose" / "cands.json"),
                     choose=str(t / "choose"), qe=str(t / "scores" / "qe-w*.parquet"), comet=str(t / "comet.parquet"),
                     c1=str(t / "c1.parquet"), c2=str(t / "c2.parquet"),
                     judge1=str(t / "judge1" / "scores" / "w*" / "shard-*.parquet"),
                     judge2=str(t / "judge2" / "scores" / "w*" / "shard-*.parquet"),
                     m23_metricx=str(t / "m23" / "metricx.parquet"), m23_judge=str(t / "m23" / "judge" / "shard-*.parquet"),
                     out=str(t / "report" / "report.md"))
    r = gp.report(args)
    assert r["checks"]["qe_repro"]["max_abs"] == pytest.approx(0.01) and r["checks"]["judge_repro"]["deviations"] == 0
    s = r["systems"]["alle"]
    assert s["G4-0"]["qe"] == 1.5 and s["G4-8-s140"]["qe"] == 1.0 and s["mens"]["n"] == 4 and s["G4-0"]["c1_original"] == 1.0
    assert r["systems"]["wmt24pp"]["G4-0"]["comet_n"] == 2 and r["systems"]["novels"]["G4-0"]["comet_n"] == 0
    assert r["trial"]["mens|alle"]["a1"]["direction"]["share"] == 1.0 and r["trial"]["bo8|alle"]["rubric"]["direction"]["n"] == 1
    md = (t / "report" / "report.md").read_text()
    assert md.startswith("# Evaluatiepool G4") and "## 4. Judge-scores" in md
    texts_ = pq.read_table(t / "texts" / "texts.parquet").to_pylist()
    assert not any(x["text"] in md for x in texts_ if len(x["text"]) > 8)
    assert "POOLREPORT" in capsys.readouterr().out
    # a QE reproduction off by more than 0.06 stops the report
    rows = pq.read_table(t / "scores" / "qe-w0.parquet").to_pylist()
    for x in rows:
        if x["id"].startswith(gp.REPRO):
            x["qe_source_hypothesis"] = 2.07
    gp.write(rows, t / "scores" / "qe-w0.parquet")
    with pytest.raises(SystemExit, match="QE reproduction deviates"):
        gp.report(args)


# ---------------------------------------------------------------------------------------------------- §25 blind packages

def test_the_blind_packages_are_fixed_as_section_25_has_it():
    text = (CODE / "protocol-g4.md").read_text(encoding="utf-8")
    section = text[text.index("## 25. Twee blinde pakketten"):]
    assert "`S25=en-nl-g4arm/0/G4-blind25`" in section and gp.BLIND_SEED == "en-nl-g4arm/0/G4-blind25"
    assert "**P2 `G4-8-vs-bo8ri`, 60 paren**" in section and "**P1 `G4-8-vs-mens`, 60 paren**" in section
    assert "**20 romans, 20 BT en 20 WMT24++**" in section and gp.BLIND_PER_PART == 20
    assert gp.BLIND_PARTS == ("novels", "bt", "wmt24pp")
    assert list(gp.BLIND_PACKAGES) == ["G4-8-vs-bo8ri", "G4-8-vs-mens"] and "Volgorde: eerst P2, dan P1." in section
    assert {k: (v["type"], v["prefix"], v["base"]) for k, v in gp.BLIND_PACKAGES.items()} == {
        "G4-8-vs-bo8ri": ("bo8", "vb", "bo8-rubriek∩N1"), "G4-8-vs-mens": ("mens", "vm", "mens")}
    assert "`vb-001`…`vb-060` (P2) en `vm-001`…`vm-060` (P1)" in section
    for seed in ('random.Random(f"{S25}/{pakket}/{stratum}/groups").shuffle',
                 'random.Random(f"{S25}/{pakket}/{stratum}/{groep}").shuffle',
                 'random.Random(f"{S25}/{pakket}/{stratum}/ab").shuffle', 'random.Random(f"{S25}/{pakket}/order").shuffle',
                 '`["A"] * 10 + ["B"] * 10`'):
        assert seed in section
    assert "dit beslist niets" in section


def test_the_reviewers_rubric_is_protocol_md_4_2_4_literally():
    rubric = gp.rubric_text(CODE / "protocol.md")
    lines = rubric.split("\n")
    assert lines[0].startswith("Je beoordeelt vertalingen uit het Engels in het Nederlands. Per passage krijg je")
    assert lines[-1] == ('{"blind_id": "...", "fout_A": "ja|nee", "fout_B": "ja|nee", "oordeel": "A beter|B beter|gelijk", '
                         '"zekerheid": "zeker|twijfel", "uitleg": "..."}')
    assert "```" not in rubric and lines[0] == lines[0].lstrip() and lines[-1] == lines[-1].lstrip()
    # the block in protocol.md is the rubric indented by five spaces
    assert "\n".join("     " + x if x else "" for x in lines) in (CODE / "protocol.md").read_text(encoding="utf-8")
    md = gp.reviewer_text(60, rubric)
    assert rubric in md and "`judge-input.jsonl`: 60 regels" in md and "`judgments.jsonl` in deze map" in md
    # nothing names the other side: no package name, no system
    assert not any(w in md for w in ("mens", "bo8", "G4-", "base", "menselijk", "origineel"))


def test_round_robin_spreads_over_the_groups_and_the_sides_are_blocked():
    groups = {"b1": [f"b1-{k}" for k in range(7)], "b2": ["b2-0"], "b3": [f"b3-{k}" for k in range(3)]}
    got = gp.round_robin(groups, 6, "s")
    assert got == gp.round_robin({g: list(reversed(v)) for g, v in groups.items()}, 6, "s")
    assert sum(x.startswith("b2") for x in got) == 1 and sum(x.startswith("b3") for x in got) == 2
    order = mx.shuffled(groups, "s/groups")
    assert got[:3] == [mx.shuffled(groups[g], f"s/{g}")[0] for g in order]
    assert len(gp.round_robin(groups, 11, "s")) == 11
    with pytest.raises(SystemExit, match="11 eligible sources, 12 wanted"):
        gp.round_robin(groups, 12, "s")
    sides = gp.blocked_sides(20, "x/ab")
    assert sides.count("A") == 10 and sides.count("B") == 10 and sides == gp.blocked_sides(20, "x/ab")
    assert sides != gp.blocked_sides(20, "y/ab")
    with pytest.raises(SystemExit, match="50/50"):
        gp.blocked_sides(19, "x/ab")


def blind_world(tmp, rng, n_sample=None):
    """A pool's choose/ of novels (10 books x 10), BT (5 sources x 20), WMT24++ 100 and FLORES 40: per source G4-0,
    G4-8-s140, the rubric ∩ N1 choice (a sample for n_sample[part] sources, else the anchor) and the human text"""
    n_sample = n_sample or {"novels": 37, "bt": 41, "wmt24pp": 38, "flores": 12}
    groups = {"novels": [str(90000 + 1111 * b) for b in range(10)], "bt": list(SOURCES), "wmt24pp": ["wmt24pp"],
              "flores": ["flores"]}
    per = {"novels": 10, "bt": 20, "wmt24pp": 100, "flores": 40}
    C = "bo8-rubriek∩N1"
    choice, outputs, meta, pairs = [], [], [], []
    for part, gs in groups.items():
        ids = [(f"{part}:{g}-{k:02d}", g) for g in gs for k in range(per[part])]
        sampled = set(rng.sample([i for i, _ in ids], n_sample[part]))
        for n, (i, g) in enumerate(ids):
            src = "  " + sentence(rng) + "\n"
            text = {"G4-0": sentence(rng), "G4-8-s140": sentence(rng) + " \n", "mens": "\n" + sentence(rng)}
            j = rng.randrange(8) if i in sampled else -1
            tid = {"G4-0": f"{i}|G4-0", "G4-8-s140": f"{i}|G4-8-s140", "mens": f"{i}|mens",
                   C: f"{i}|s{j}" if j >= 0 else f"{i}|G4-0"}
            if j >= 0:
                text[C] = sentence(rng) if n % 9 else " ".join(text["G4-8-s140"].split())  # some equal G4-8's text
            else:
                text[C] = text["G4-0"]
            choice.append({"id": i, "part": part, "stratum": g, f"{C}|j": j, f"{C}|reason": "sample" if j >= 0 else "differ"})
            for system in ("G4-0", "G4-8-s140", C, "mens"):
                outputs.append({"id": i, "part": part, "system": system, "tid": tid[system], "text": text[system]})
            for ptype, b in (("mens", "mens"), ("bo8", C)):
                same = gp.norm(text["G4-8-s140"]) == gp.norm(text[b])
                pid = f"{ptype}/{i}"
                meta.append({"pair": pid, "type": ptype, "id": i, "part": part, "a_system": "G4-8-s140", "b_system": b,
                             "a_tid": tid["G4-8-s140"], "b_tid": tid[b], "identical": same})
                if not same:
                    pairs.append({"pair": pid, "source": src, "A": text["G4-8-s140"], "B": text[b]})
    d = tmp / "choose"
    for name, rows in (("choice", choice), ("outputs", outputs), ("pairs-meta", meta), ("pairs", pairs)):
        gp.write(rows, d / f"{name}.parquet")
    return d, {r["tid"]: r["text"] for r in outputs}, {r["id"]: r for r in choice}


def test_blind_draws_p2_then_p1_blocked_and_blind(tmp_path, capsys):
    rng = random.Random(25)
    d, texts_, choice = blind_world(tmp_path, rng)
    args = Namespace(choose=str(d), out=str(tmp_path / "out"), protocol=str(CODE / "protocol.md"))
    summary = gp.blind(args)
    printed = capsys.readouterr().out
    assert "BLIND25" in printed and not any(t.strip() in printed for t in texts_.values() if len(t.strip()) > 8)
    drawn = {}
    for package, spec in gp.BLIND_PACKAGES.items():
        pkg = tmp_path / "out" / package
        assert sorted(x.name for x in pkg.iterdir()) == ["judge-input.jsonl", "key.jsonl", "review", "select-report.json"]
        inputs = g4arm.read_jsonl(pkg / "judge-input.jsonl")
        keys = g4arm.read_jsonl(pkg / "key.jsonl")
        assert [x["blind_id"] for x in inputs] == [k["blind_id"] for k in keys] == [f"{spec['prefix']}-{n:03d}"
                                                                                    for n in range(1, 61)]
        assert all(set(x) == {"blind_id", "source", "A", "B"} for x in inputs)
        s = summary[package]
        assert s["n"] == 60 and s["per_set"] == {"novels": 20, "bt": 20, "wmt24pp": 20}
        assert s["arm_is_a"] == {"novels": 10, "bt": 10, "wmt24pp": 10} and s["a_is_base"] == 30
        for x, k in zip(inputs, keys):
            assert k["base_system"] == spec["base"] and k["set"] in gp.BLIND_PARTS and {k["A"], k["B"]} == {"G4-8-s140", "base"}
            arm_side = "A" if k["A"] == "G4-8-s140" else "B"
            other = "B" if arm_side == "A" else "A"
            assert x[arm_side] == texts_[k["a_tid"]].strip() and x[other] == texts_[k["b_tid"]].strip()
            assert all(v == v.strip() for v in x.values()) and k["stratum"] == choice[k["item"]]["stratum"]
            if spec["type"] == "bo8":
                c = choice[k["item"]]
                assert c["bo8-rubriek∩N1|reason"] == "sample" and k["b_tid"] == f"{k['item']}|s{k['bo8_j']}"
                assert gp.norm(x[other]) != gp.norm(x[arm_side])
            else:
                assert k["b_tid"] == f"{k['item']}|mens"
        # the 20 novels spread over the books, BT over its five sources
        per_stratum = json.loads((pkg / "select-report.json").read_text())["per_stratum"]
        assert len(per_stratum["bt"]) == 5 and max(per_stratum["bt"].values()) <= 4 + (spec["type"] == "bo8") * 4
        if spec["type"] == "mens":
            assert set(per_stratum["novels"].values()) == {2} and set(per_stratum["bt"].values()) == {4}
        assert gp.rubric_text(CODE / "protocol.md") in (pkg / "review" / "REVIEWER.md").read_text(encoding="utf-8")
        drawn[package] = {k["item"] for k in keys}
        # the key works with the official tally (A/B name the arm or 'base')
        judg = [{"blind_id": k["blind_id"], "fout_A": "nee", "fout_B": "ja" if k["B"] == "base" else "nee",
                 "oordeel": "A beter" if k["A"] == "G4-8-s140" else "B beter", "zekerheid": "zeker"} for k in keys]
        t = g4arm.tally_rows(judg, keys, "G4-8-s140")
        assert (t["arm_better"], t["base_better"], t["equal"], t["errors_base"], t["errors_arm"]) == (60, 0, 0, 30, 0)
    assert not drawn["G4-8-vs-bo8ri"] & drawn["G4-8-vs-mens"]
    rep2 = json.loads((tmp_path / "out" / "G4-8-vs-bo8ri" / "select-report.json").read_text())
    rep1 = json.loads((tmp_path / "out" / "G4-8-vs-mens" / "select-report.json").read_text())
    assert rep1["drawn_before"] == ["G4-8-vs-bo8ri"] and rep2["drawn_before"] == []
    assert sum(rep1["excluded"][p].get("drawn_before", 0) for p in gp.BLIND_PARTS) == 60
    assert all(rep2["excluded"][p]["anchor"] > 0 for p in gp.BLIND_PARTS)
    # the same inputs draw the same packages; an existing package is not drawn again
    again = Namespace(choose=str(d), out=str(tmp_path / "again"), protocol=str(CODE / "protocol.md"))
    gp.blind(again)
    for package in gp.BLIND_PACKAGES:
        for name in ("judge-input.jsonl", "key.jsonl"):
            assert (tmp_path / "again" / package / name).read_bytes() == (tmp_path / "out" / package / name).read_bytes()
    with pytest.raises(SystemExit, match="a package is drawn once"):
        gp.blind(args)


def test_blind_stops_on_too_few_samples_and_on_a_choice_with_the_anchors_text(tmp_path):
    d, _, _ = blind_world(tmp_path, random.Random(3), {"novels": 37, "bt": 19, "wmt24pp": 38, "flores": 0})
    with pytest.raises(SystemExit, match="G4-8-vs-bo8ri/bt: 1[0-9] eligible sources, 20 wanted"):
        gp.blind(Namespace(choose=str(d), out=str(tmp_path / "a"), protocol=str(CODE / "protocol.md")))
    d, texts_, choice = blind_world(tmp_path / "w2", random.Random(4))
    rows = pq.read_table(d / "outputs.parquet").to_pylist()
    i = next(c["id"] for c in choice.values() if c["bo8-rubriek∩N1|reason"] == "sample")
    j = choice[i]["bo8-rubriek∩N1|j"]
    for r in rows:
        if r["tid"] == f"{i}|s{j}":
            r["text"] = texts_[f"{i}|G4-0"]
    gp.write(rows, d / "outputs.parquet")
    with pytest.raises(SystemExit, match="has G4-0 greedy's text"):
        gp.blind(Namespace(choose=str(d), out=str(tmp_path / "b"), protocol=str(CODE / "protocol.md")))
