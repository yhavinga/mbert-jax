"""G4-9's training (protocol-g4.md §22, a draft until Yeb decides): §19's training with --arm G4-9 / ARM8=G4-9 / ARM=G4-9
on the N1 choice of §21 (O9 n1), or on its intersection with §18's rubric choice (O9 n1∩rubriek, --mode intersect,
TARGETS=intersect). g4arm8.py's n1targets (§21's select-n1 merged and checked), sftset, trained and report8 (with
the comparison with G4-8's choice) on small synthetic inputs; run_g4arm8.sh's and run_g4arm.sh's stages of G4-9 rendered
with a fake bucket (no model, no TPU)."""

import collections
import json
import re
from argparse import Namespace

import pyarrow.parquet as pq
import pytest
import test_en_nl_g4arm7 as t7
import test_en_nl_g4arm8 as t8d
import test_en_nl_g4arm8t as t8t

g8, g4rl, g4arm = t8d.g8, t8d.g4rl, t7.g4arm
CODE = t8d.CODE


# ---------------------------------------------------------------------------------------------------- §22, the names

def test_section_22_is_a_draft_or_final_with_one_o6_and_o1_g4_0_and_the_code_agrees():
    text = (CODE / "protocol-g4.md").read_text(encoding="utf-8")
    start = text.index("## 22. G4-9 training")
    end = text.find("\n## 23.", start)
    section = text[start:] if end < 0 else text[start:end]
    o6 = re.findall(r"`O6 \[besluit Yeb[^]`]*\]: (R-oud|R-nieuw)`", section)
    o1 = re.findall(r"`O1 \[besluit Yeb[^]`]*\]: ([^`]*)`", section)
    if "CONCEPT, NIET GESTART" in section:
        # the draft names both variants of O6 and the default O1; the chain refuses it
        assert sorted(o6) == ["R-nieuw", "R-oud"] and o1 == ["G4-0"]
    else:
        # the final version as g4arm9_train_run.sh reads it: one decision on O6, O1 G4-0 (G4-8-s140 is not built)
        assert len(o6) == 1 and o1 == ["G4-0"]
    for x in ("`G4-9-s<stap>`", "`g4arm_G4-9`", "`g4arm_G4-9_s8`", "`$S/blind/G4-9/`", "`$S/blind/G4-9-tg/`",
              "`$G/report-G4-9/report.md`", "`$G/stages/g4-9-train/`", "`mbert-ops/g4arm9.train-go`",
              "`en-nl-g4arm/0/G4-9/train/shuffle/<doorgang>`", "`en-nl-g4arm/0/G4-9/div/w<k>`",
              "`en-nl-g4arm/0/G4-bon/boot/G4-9/b`", "**G4-8-s140**", "`g4arm8.py n1targets`", "16.349"):
        assert x in section, x
    assert g8.ARMS_T["G4-9"]["protocol"] == g8.PROTOCOL_9 == "protocol-g4.md §22"
    assert g8.ARMS_T["G4-9"]["compare"] == "G4-8-s140" and g8.ARMS_T["G4-8"]["compare"] is None
    for script in ("run_g4arm8.sh", "run_g4arm.sh"):
        assert re.search(r"^COMPARE9=G4-8-s140 ", (CODE / script).read_text(), re.M), script


def test_the_arm_names_the_candidates_and_their_steps():
    m = {"save_steps": [74, 148], "steps": 221}
    assert g8.candidates_t(m, "G4-9") == ["G4-9-s74", "G4-9-s148", "G4-9-s221"]
    assert g8.candidates_t(m) == ["G4-8-s74", "G4-8-s148", "G4-8-s221"]
    assert g8.cand_step_t("G4-9-s148", "G4-9") == 148 and g8.cand_step_t("G4-8-s140") == 140
    for name, arm in (("G4-8-s140", "G4-9"), ("G4-9-s148", "G4-8"), ("G4-9-s0", "G4-9"), ("G4-7-i5", "G4-9")):
        with pytest.raises(SystemExit, match="no candidate"):
            g8.cand_step_t(name, arm)
    assert g8.seed_t("G4-9") == "en-nl-g4arm/0/G4-9" and g8.seed_t("G4-8") == g8.SEED8
    with pytest.raises(SystemExit, match="--arm G4-10"):
        g8.arm_of(Namespace(arm="G4-10"))


# ---------------------------------------------------------------------------------------------------- n1targets

def n1_chunks(tmp_path, per_chunk=6):
    """A plan of 3 chunks; per chunk select-n1's choice.parquet and targets.parquet (a target for 4 of 6 sources, the
    others 'p' and 'none'), and §21's report.json with each choice's sha256 and the counts per part"""
    plan, dirs, inputs = [], {}, {}
    kept = {p: 0 for p in g8.PARTS8}
    parts = ("flan", "novels", "bt")
    for ci, c in enumerate(("c1", "c2", "c3")):
        choice, targets = [], []
        for k in range(per_chunk):
            part = parts[(ci + k) % 3]
            i = f"{part}:{c}-{k}"
            en = f"an english source {c} {k}"
            plan.append({"id": i, "part": part, "stratum": "", "doc": f"d{c}{k}", "en": en, "tokens": 24,
                         "origin": "g4-7", "chunk": c, "hypothesis0": f"anker {c} {k}", "l0": 20, "qe0_g47": 2.0})
            reason = "sample" if k < 4 else ("p" if k == 4 else "none")
            row = {"id": i, "part": part, "stratum": "", "origin": "g4-7", "chunk": c, "reason": reason,
                   "j": k % 8 if reason == "sample" else -1, "p_best": 0.7 if reason != "none" else None, "qe0": 2.0,
                   "candidates": 3, "keys": 3, "not_scored": 0, "distinct": 5, "fails": 0,
                   **dict.fromkeys(g8.TARGET_MEASURES)}
            if reason == "sample":
                row.update({"p_sample": 0.6 + 0.05 * k, "p_anchor": 0.3, "p_gelijk": 0.1, "qe": 1.5 + 0.1 * k,
                            "ratio": 1.0, "tokens_out": 20, "tokens_in": 50, "chrf": 70.0})
                targets.append({**row, "en": en, "target": f"doel {c} {k}", "doc": f"d{c}{k}", "l0": 20})
                kept[part] += 1
            choice.append(row)
        d = tmp_path / c
        g8.write_parquet(choice, d / "choice.parquet")
        g8.write_parquet(targets, d / "targets.parquet")
        dirs[c] = d
        inputs[c] = {"choice-n1.parquet": g4rl.file_sha256(d / "choice.parquet")}
    g8.write_parquet(plan, tmp_path / "plan.parquet", g8.PLAN_SCHEMA)
    rep = {"protocol": g8.PROTOCOL_N1, "chunks": ["c1", "c2", "c3"], "inputs": inputs,
           "parts": {"samen": {"n1_kept": sum(kept.values())}, **{p: {"n1_kept": n} for p, n in kept.items()}}}
    (tmp_path / "n1report.json").write_text(json.dumps(rep))
    return dirs, rep


def n1_args(tmp_path, dirs, out="out"):
    return ["n1targets", "--plan", str(tmp_path / "plan.parquet"), "--n1report", str(tmp_path / "n1report.json"),
            *[x for c, d in dirs.items() for x in ("--chunk-dir", f"{c}={d}")], "--out", str(tmp_path / out)]


def test_n1targets_merges_the_n1_choice_in_section_18s_form(tmp_path, capsys):
    dirs, rep = n1_chunks(tmp_path)
    g8.main(n1_args(tmp_path, dirs))
    rows = pq.read_table(tmp_path / "out" / "targets.parquet").to_pylist()
    assert [r["id"] for r in rows] == sorted(r["id"] for r in rows) and len(rows) == 12
    assert {r["reason"] for r in rows} == {"sample"} and all(r["qe"] <= r["qe0"] for r in rows)
    # §18's form: the columns of select's targets.parquet, as report writes $G8/targets/targets.parquet
    one = pq.read_table(dirs["c1"] / "targets.parquet").schema.names
    assert pq.read_table(tmp_path / "out" / "targets.parquet").schema.names == one
    info = json.loads((tmp_path / "out" / "targets.json").read_text())
    assert info["protocol"] == "protocol-g4.md §22" and info["arm"] == "G4-9" and info["targets"] == 12
    assert info["parts"] == {p: rep["parts"][p]["n1_kept"] for p in g8.PARTS8}
    assert info["inputs"]["c2"]["choice"] == rep["inputs"]["c2"]["choice-n1.parquet"]
    assert info["targets_sha256"] == g4rl.file_sha256(tmp_path / "out" / "targets.parquet")
    assert '"targets": 12' in capsys.readouterr().out
    # sftset takes them as G4-8's (§19's checks of a janchor target)
    plan = g8.read_rows(tmp_path / "plan.parquet")
    assert g8.checked_targets(rows, plan) == info["parts"]


def edit(path, change):
    rows = g8.read_rows(path)
    change(rows)
    g8.write_parquet(rows, path)


@pytest.mark.parametrize("what, msg", [
    ("qe", "above QE0"), ("p", "not above 0.5"), ("en", "English is not the plan's"), ("anchor", "the anchor's text"),
    ("drop", "not one per source"), ("choice", "is not the choice-n1.parquet"), ("count", "§21's report"),
    ("protocol", "is not protocol-g4.md §21's"), ("sample", "is not choice.parquet's"),
])
def test_n1targets_refuses_and_writes_nothing(tmp_path, what, msg):
    dirs, rep = n1_chunks(tmp_path)
    t2 = dirs["c2"] / "targets.parquet"
    if what == "qe":
        # the guard: a target above QE0 also in choice.parquet (and its sha256 in the report)
        for f in (t2, dirs["c2"] / "choice.parquet"):
            edit(f, lambda rows: rows[0].update(qe=2.5))
        rep["inputs"]["c2"]["choice-n1.parquet"] = g4rl.file_sha256(dirs["c2"] / "choice.parquet")
    elif what == "p":
        for f in (t2, dirs["c2"] / "choice.parquet"):
            edit(f, lambda rows: rows[0].update(p_sample=0.5))
        rep["inputs"]["c2"]["choice-n1.parquet"] = g4rl.file_sha256(dirs["c2"] / "choice.parquet")
    elif what == "en":
        edit(t2, lambda rows: rows[1].update(en="another english"))
    elif what == "anchor":
        edit(t2, lambda rows: rows[1].update(target="  anker   c2 1 "))
    elif what == "drop":
        edit(t2, lambda rows: rows.pop())
    elif what == "choice":
        edit(dirs["c3"] / "choice.parquet", lambda rows: rows[5].update(fails=1))
    elif what == "count":
        rep["parts"]["bt"]["n1_kept"] += 1
    elif what == "protocol":
        rep["protocol"] = "protocol-g4.md §18"
    elif what == "sample":
        edit(t2, lambda rows: rows[2].update(j=7))
    (tmp_path / "n1report.json").write_text(json.dumps(rep))
    with pytest.raises(SystemExit, match=re.escape(msg)):
        g8.main(n1_args(tmp_path, dirs))
    assert not (tmp_path / "out").exists()


def test_n1targets_takes_each_chunk_once(tmp_path):
    dirs, _ = n1_chunks(tmp_path)
    args = n1_args(tmp_path, {c: d for c, d in dirs.items() if c != "c3"})
    with pytest.raises(SystemExit, match="one per chunk"):
        g8.main(args)


# ---------------------------------------------------------------------------------------------------- §22 G4-9.V3N1

V3N1_TARGETS = {("bt", "dpc"): 10, ("bt", "raad"): 12, ("bt", "fineweb"): 5, ("novels", ""): 50, ("flan", ""): 11}


def v3n1_world(tmp_path):
    """Chunk c1 (and a source of c2) with select-n1's choice and targets, §18's choice (for each N1 target the same
    sample, another one or the anchor in turn) and §21's report with the sha256 of plan and both choices"""
    plan, n1, targets, rubric = [], [], [], []
    k = 0
    for (part, stratum), n in V3N1_TARGETS.items():
        for m in range(n + 2):                     # two sources per group without an N1 target ('p' and 'none')
            k += 1
            i = f"{part}:{stratum or 'x'}-{m:03d}"
            en = f"an english source {k}"
            plan.append({"id": i, "part": part, "stratum": stratum, "doc": f"d{k}", "en": en, "tokens": 24,
                         "origin": "g4-7", "chunk": "c1", "hypothesis0": f"anker {k}", "l0": 20, "qe0_g47": 2.0})
            reason = "sample" if m < n else ("p" if m == n else "none")
            j = m % 8 if reason == "sample" else -1
            row = {"id": i, "part": part, "stratum": stratum, "origin": "g4-7", "chunk": "c1", "reason": reason, "j": j,
                   "p_best": None if reason == "none" else 0.7, "qe0": 2.0, "candidates": 3, "keys": 3,
                   "not_scored": 0, "distinct": 5, "fails": 0, **dict.fromkeys(g8.TARGET_MEASURES)}
            if reason == "sample":
                row.update({"p_sample": 0.75, "p_anchor": 0.2, "p_gelijk": 0.05, "qe": 1.5, "ratio": 1.0,
                            "tokens_out": 20, "tokens_in": 50, "chrf": 70.0})
                targets.append({**row, "en": en, "target": f"doel {k}", "doc": f"d{k}", "l0": 20})
            n1.append(row)
            rubric_reason, rubric_j = (("sample", j), ("sample", (j + 1) % 8), ("p", -1))[m % 3] if reason == "sample" \
                else (reason, -1)
            rubric.append({**row, "reason": rubric_reason, "j": rubric_j})
    plan.append({"id": "bt:other-chunk", "part": "bt", "stratum": "dpc", "doc": "dz", "en": "another", "tokens": 24,
                 "origin": "g4-7", "chunk": "c2", "hypothesis0": "anker z", "l0": 20, "qe0_g47": 2.0})
    d = tmp_path / "in"
    g8.write_parquet(plan, d / "plan.parquet", g8.PLAN_SCHEMA)
    g8.write_parquet(n1, d / "choice.parquet")
    g8.write_parquet(targets, d / "targets.parquet")
    g8.write_parquet(rubric, d / "rubric-choice.parquet")
    rep = {"protocol": g8.PROTOCOL_N1, "chunks": ["c1", "c2", "c3"],
           "inputs": {"c1": {"n1pairs": {"plan": g4rl.file_sha256(d / "plan.parquet")},
                             "choice.parquet": g4rl.file_sha256(d / "rubric-choice.parquet"),
                             "choice-n1.parquet": g4rl.file_sha256(d / "choice.parquet")}}}
    (d / "n1report.json").write_text(json.dumps(rep))
    return d, rep, {r["id"]: r for r in plan}, {r["id"]: r for r in targets}, {r["id"]: r for r in rubric}


def v3n1_args(d, out, extra=("--set", "n1")):
    return ["v3", "--plan", str(d / "plan.parquet"), "--targets", str(d / "targets.parquet"), *extra,
            "--choice", str(d / "choice.parquet"), "--rubric", str(d / "rubric-choice.parquet"),
            "--n1report", str(d / "n1report.json"), "--out", str(out)]


def test_v3_with_set_n1_builds_g4_9_v3n1_from_c1s_n1_targets(tmp_path, capsys):
    import random

    d, rep, plan, targets, rubric = v3n1_world(tmp_path)
    out = tmp_path / "out"
    g8.main(v3n1_args(d, out))
    keys = [json.loads(x) for x in (out / "key.jsonl").read_text().splitlines()]
    inputs = [json.loads(x) for x in (out / "judge-input.jsonl").read_text().splitlines()]
    report = json.loads((out / "select-report.json").read_text())
    # BT 8 per source (fewer: all), the sources by name; then novels 40, FLAN 20 (fewer: all)
    want = []
    for part, stratum, n in (("bt", "dpc", 8), ("bt", "fineweb", 8), ("bt", "raad", 8), ("novels", "", 40), ("flan", "", 20)):
        ids = sorted(i for i, t in targets.items() if t["part"] == part and (part != "bt" or t["stratum"] == stratum))
        seed = f"en-nl-g4arm/0/G4-9/v3n1/{part}/{stratum}" if part == "bt" else f"en-nl-g4arm/0/G4-9/v3n1/{part}"
        want += random.Random(seed).sample(ids, min(n, len(ids)))
    assert [k["item"] for k in keys] == want and len(want) == 8 + 5 + 8 + 40 + 11
    assert [k["blind_id"] for k in keys] == [f"v3n1-{n:03d}" for n in range(1, len(want) + 1)]
    labels = {"sample": "zelfde sample", "p": "anker"}
    for k, x in zip(keys, inputs):
        t, base = targets[k["item"]], plan[k["item"]]["hypothesis0"]
        assert k["seed"] == f"en-nl-g4arm/0/G4-9/v3n1/ab/{k['blind_id']}"
        base_first = random.Random(k["seed"]).random() < 0.5
        assert (k["A"], k["B"]) == (("base", "G4-9-n1") if base_first else ("G4-9-n1", "base"))
        assert (x["A"], x["B"]) == ((base, t["target"]) if base_first else (t["target"], base))
        assert set(x) == {"blind_id", "source", "A", "B"} and x["source"] == t["en"] and x["blind_id"] == k["blind_id"]
        r = rubric[k["item"]]
        assert k["rubric"] == (labels[r["reason"]] if r["reason"] == "p" or r["j"] == t["j"] else "ander sample")
        assert (k["set"], k["stratum"]) == (t["part"], t["stratum"])
    assert {k["rubric"] for k in keys} == {"zelfde sample", "ander sample", "anker"}
    # the report: counts, seeds and sha256 only, no text
    assert report["protocol"] == "protocol-g4.md §22 G4-9.V3N1" and report["target"] == "G4-9-n1"
    assert report["n"] == len(keys) and report["chunk"] == "c1" and report["a_is_base"] == sum(k["A"] == "base" for k in keys)
    bt = report["sets"]["bt"]
    assert bt["targets"] == 27 and bt["picked"] == 21 and bt["strata"] == {"dpc": 8, "fineweb": 5, "raad": 8}
    assert bt["targets_strata"] == {"dpc": 10, "fineweb": 5, "raad": 12}
    assert bt["seed"] == {s: f"en-nl-g4arm/0/G4-9/v3n1/bt/{s}" for s in ("dpc", "fineweb", "raad")}
    assert report["sets"]["novels"]["seed"] == "en-nl-g4arm/0/G4-9/v3n1/novels" and report["sets"]["flan"]["picked"] == 11
    for part, s in report["sets"].items():
        assert sum(s["rubric"].values()) == s["picked"] and sum(s["rubric_targets"].values()) == s["targets"]
        assert s["rubric"] == dict(sorted(collections.Counter(k["rubric"] for k in keys if k["set"] == part).items()))
    assert report["inputs"] == {n: g4rl.file_sha256(d / f) for n, f in (
        ("plan", "plan.parquet"), ("targets", "targets.parquet"), ("choice", "choice.parquet"),
        ("rubric", "rubric-choice.parquet"), ("n1report", "n1report.json"))}
    shown = (out / "key.jsonl").read_text() + (out / "select-report.json").read_text() + capsys.readouterr().out
    assert "doel " not in shown and "anker " not in shown and "english" not in shown


def test_v3_without_a_set_is_g4_8_v3_and_n1s_inputs_only_with_set_n1(tmp_path):
    d, *_ = v3n1_world(tmp_path)
    assert g8.V3_SETS[""]["seed"] == "en-nl-g4arm/0/G4-8/v3" and g8.V3_SETS[""]["target"] == g8.V3_TARGET
    assert g8.V3_SETS[""]["parts"] == (("bt", 50, False), ("novels", 50, False))
    with pytest.raises(SystemExit, match="only with v3 --set n1"):
        g8.main(v3n1_args(d, tmp_path / "out", extra=()))
    with pytest.raises(SystemExit, match="takes --choice, --rubric and --n1report"):
        g8.main(["v3", "--set", "n1", "--plan", str(d / "plan.parquet"), "--targets", str(d / "targets.parquet"),
                 "--choice", str(d / "choice.parquet"), "--out", str(tmp_path / "out")])
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("what, msg", [
    ("plan", "is not the plan that §21's n1pairs read"), ("rubric", "is not §18's choice.parquet"),
    ("choice", "is not the choice-n1.parquet"), ("qe", "above QE0"), ("none", "where §18's rubric chose"),
    ("protocol", "is not protocol-g4.md §21's report over c1"), ("drop", "not one per source"),
])
def test_v3_with_set_n1_refuses_and_writes_nothing(tmp_path, what, msg):
    d, rep, plan, targets, rubric = v3n1_world(tmp_path)
    first = sorted(targets)[0]
    if what == "plan":
        edit(d / "plan.parquet", lambda rows: rows[-1].update(en="changed"))
    elif what == "rubric":
        edit(d / "rubric-choice.parquet", lambda rows: rows[0].update(fails=1))
    elif what == "choice":
        edit(d / "choice.parquet", lambda rows: rows[0].update(fails=1))
    elif what in ("qe", "none"):
        # also in the choice files, their sha256 in §21's report: the checks themselves stop
        name = "choice.parquet" if what == "qe" else "rubric-choice.parquet"
        change = {"qe": {"qe": 2.5}, "none": {"reason": "none", "j": -1}}[what]
        if what == "qe":
            edit(d / "targets.parquet", lambda rows: [r.update(change) for r in rows if r["id"] == first])
        edit(d / name, lambda rows: [r.update(change) for r in rows if r["id"] == first])
        rep["inputs"]["c1"]["choice-n1.parquet" if what == "qe" else "choice.parquet"] = g4rl.file_sha256(d / name)
    elif what == "protocol":
        rep["chunks"] = ["c2", "c3"]
    elif what == "drop":
        edit(d / "targets.parquet", lambda rows: rows.pop())
    (d / "n1report.json").write_text(json.dumps(rep))
    with pytest.raises(SystemExit, match=re.escape(msg)):
        g8.main(v3n1_args(d, tmp_path / "out"))
    assert not (tmp_path / "out").exists()


# ---------------------------------------------------------------------------------------------------- sftset, trained

def test_sftset_with_arm_g4_9_takes_its_seed_names_and_protocol(tmp_path, monkeypatch):
    monkeypatch.setattr(g8, "SEQ_T", 32)
    monkeypatch.setattr(g8, "SMOKE_LONGEST_T", 10)
    monkeypatch.setattr(g8, "SMOKE_RANDOM_T", 20)
    blind = tmp_path / "in" / "blind100.jsonl"
    monkeypatch.setattr(g4arm, "blind_sha256", lambda: g4rl.file_sha256(blind))
    d, plan, targets = t8t.sft_world(tmp_path)
    metas = {}
    for arm in ("G4-8", "G4-9"):
        args = t8t.sft_args(d, tmp_path / arm)
        args.arm = arm
        g8.sftset(args, counter=t8t.Counter())
        m = json.loads((tmp_path / arm / "manifest.json").read_text())
        metas[arm] = (m, g8.read_rows(tmp_path / arm / "train-meta.parquet"))
    m9, rows9 = metas["G4-9"]
    m8, rows8 = metas["G4-8"]
    assert (m9["set"], m9["seed"], m9["protocol"]) == ("G4-9", "en-nl-g4arm/0/G4-9", "protocol-g4.md §22")
    assert m9["candidates"] == [f"G4-9-s{k}" for k in [*m9["save_steps"], m9["steps"]]]
    assert m9["examples"] == m8["examples"] == 2 * len(targets) and m9["inputs"] == m8["inputs"]
    # the same examples, another shuffle per pass
    assert sorted(r["id"] for r in rows9) == sorted(r["id"] for r in rows8)
    assert [r["id"] for r in rows9] != [r["id"] for r in rows8]
    smoke = json.loads((tmp_path / "G4-9" / "smoke" / "manifest.json").read_text())
    assert smoke["set"] == "G4-9-smoke" and smoke["seed"] == "en-nl-g4arm/0/G4-9/smoke"
    assert (tmp_path / "G4-9" / "report.md").read_text().startswith("# G4-9: de trainingsset (protocol-g4.md §22)")
    assert "de N1-keuze van G4-8-data (§21)" in (tmp_path / "G4-9" / "report.md").read_text()


def test_trained_and_smoke8_with_arm_g4_9(tmp_path):
    evals = [0, 32, 64, 96, 128, 160, 190]
    lr = {"0": 5e-6, "10": 5e-5, "95": 5e-5, "189": 5e-5}
    metas = {f"G4-9-s{k}": t8t.adapter_meta(k) for k in (64, 128, 190)}
    args = t8t.trained_files(tmp_path / "a", t8t.summary(190, evals, lr=lr), metas)
    args.arm, args.run = "G4-9", "g4arm_G4-9"
    g8.trained(args)
    r = json.loads((tmp_path / "a" / "trained.json").read_text())
    assert r["passed"] and r["candidates"] == list(metas) and r["protocol"] == "protocol-g4.md §22"
    # G4-8's adapters are no candidates of G4-9
    args = t8t.trained_files(tmp_path / "b", t8t.summary(190, evals, lr=lr))
    args.arm = "G4-9"
    with pytest.raises(SystemExit, match="the candidates are"):
        g8.trained(args)
    args = t8t.smoke_files(tmp_path / "s", t8t.summary(8, [0, 4, 8]))
    args.arm, args.run = "G4-9", "g4arm_G4-9_s8"
    g8.smoke8(args)
    assert json.loads((tmp_path / "s" / "smoke8.json").read_text())["protocol"] == "protocol-g4.md §22"


# ---------------------------------------------------------------------------------------------------- report8

NAMES9 = ["G4-9-s4", "G4-9-s8", "G4-9-s13"]
REF = "G4-8-s140"


def world9(tmp_path, monkeypatch, ref_decision_qe=1.7, ref_decision_share=1, comet=None):
    """test_en_nl_g4arm8t's world8 with G4-9's names and G4-8's choice beside: s4 holds (a1) on the search part with
    the largest (b); G4-8-s140 has QE ref_decision_qe and ref_decision_share of 4 classifier texts "original" on the
    decision part"""
    qe = {"G4-9-s4": {"search": 1.9, "decision": 1.9}, "G4-9-s8": {"search": 1.8, "decision": 1.0},
          "G4-9-s13": {"search": 3.0, "decision": 3.0}, "G4-0": {"search": 1.5, "decision": 1.5},
          "G4-4": {"search": 1.6, "decision": 1.6}, REF: {"search": 1.4, "decision": ref_decision_qe}}
    share = {"G4-9-s4": {"search": 4, "decision": 2}, "G4-9-s8": {"search": 3, "decision": 4},
             "G4-9-s13": {"search": 4, "decision": 4}, "G4-0": {"search": 1, "decision": 1},
             "G4-4": {"search": 1, "decision": 1}, REF: {"search": 2, "decision": ref_decision_share}}
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
    g8.write_parquet(rl, tmp_path / "g47-plan.parquet")
    lora, div = [], []
    for k, name in enumerate(NAMES9):
        (tmp_path / f"lora-{name}.json").write_text(json.dumps({"rank": 32, "alpha": 16.0, "modules": 410,
                                                               "relative_update_norm": 0.001 * (k + 1),
                                                               "elements_changed": 0.5}))
        lora.append(f"{name}={tmp_path / f'lora-{name}.json'}")
    for n, name in zip((6, 6, 3, 1), ["G4-0", *NAMES9]):
        rows = [t8d.sample(r["id"], j, r["en"], f"tekst {min(j, n - 1)}", f"w{k % 2}", tokens_out=20)
                for k, r in enumerate(rl) for j in range(8)]
        t8d.write_samples(tmp_path / "div" / name, rows)
        div.append(f"{name}={tmp_path / 'div' / name}")
    return Namespace(evalsets=a.evalsets, split=a.split, refqe=a.refqe, base=a.base, predictions=a.predictions,
                     predictions_c2=a.predictions_c2, candidate=[f"{n}={tmp_path / n}" for n in NAMES9],
                     reference=[f"G4-4={tmp_path / 'G4-4'}", f"{REF}={tmp_path / REF}"], pll=None,
                     summary=str(tmp_path / "summary.json"), manifest=str(tmp_path / "manifest.json"), smoke=None,
                     lora=lora, div=div, rlval=str(tmp_path / "g47-plan.parquet"), o6="R-nieuw", tally=None,
                     arm="G4-9", compare=REF, out=str(tmp_path / "report.md"))


def test_report8_of_g4_9_decides_against_g4_0_and_compares_with_g4_8s_choice(tmp_path, monkeypatch, capsys):
    args = world9(tmp_path, monkeypatch)
    r, md, line = t8t.run8(args, capsys)
    assert (r["arm"], r["protocol"], r["selected"]) == ("G4-9", "protocol-g4.md §22", "G4-9-s4")
    assert r["candidates"] == NAMES9 and r["references"] == ["G4-4", REF]
    # the rule against G4-0 as §19's: s4's C1 2 of 4 against G4-0's 1 of 4 on the decision part
    assert r["a1"]["holds"] and r["a3"]["holds"] and r["c1_guard"]["holds"] and r["blind_due"]
    assert r["b"]["delta"] == pytest.approx(0.25) and r["verdict"] == "wacht op de blinde beoordeling ((a2), (c))"
    # G4-9.11: against G4-8-s140 on the decision part, reported
    x = r["compare"]
    assert x["ref"] == REF and x["a1"] == {"model": True, "ref": True}
    for s in g4arm.HUMAN_QE:
        q = x["qe"][s]
        assert (q["model"], q["ref"]) == (pytest.approx(1.9), pytest.approx(1.7))
        assert q["ci"]["delta"] == q["ci"]["lo"] == q["ci"]["hi"] == pytest.approx(0.2) and q["ci"]["n"] == 2
    assert x["a3"]["holds"] and all(v["delta"] == pytest.approx(0.0) for v in x["a3"]["sets"].values())
    assert (x["c1"]["model"], x["c1"]["ref"]) == (0.5, 0.25) and x["c1"]["ci"]["delta"] == pytest.approx(0.25)
    assert x["c1"]["ci"]["lo"] <= 0.25 <= x["c1"]["ci"]["hi"] and x["c2"]["delta"] == pytest.approx(0.25)
    # G4-8-s140 is a reference, not a candidate: on both parts' tables, never chosen
    assert REF in r["parts"]["decision"]["models"] and REF not in r["training"]
    assert md.startswith("# Arm G4-9: uitslag volgens protocol-g4.md §22")
    assert "de N1-keuze van G4-8-data" in md and "Vergelijking met G4-8-s140 op het beslisdeel" in md
    assert "`$S/blind/G4-9/`" in md and "$S/blind/G4-8/" not in md
    assert line["selected"] == "G4-9-s4"


def test_report8_of_g4_9_refuses_without_its_comparison(tmp_path, monkeypatch):
    args = world9(tmp_path, monkeypatch)
    args.compare = None
    with pytest.raises(SystemExit, match="G4-9 compares with G4-8-s140"):
        t8t.run8(args)
    args = world9(tmp_path / "b", monkeypatch)
    args.reference = args.reference[:1]
    with pytest.raises(SystemExit, match="not among the references"):
        t8t.run8(args)
    args = world9(tmp_path / "c", monkeypatch)
    args.candidate = [c.replace("G4-9", "G4-8") for c in args.candidate]
    with pytest.raises(SystemExit, match="G4-8.4 takes"):
        t8t.run8(args)
    # G4-8 compares with nothing
    args = t8t.world8(tmp_path / "d", monkeypatch)
    args.compare = REF
    with pytest.raises(SystemExit, match="G4-8 compares with None"):
        t8t.run8(args)


# ---------------------------------------------------------------------------------------------------- run_g4arm8.sh

def test_run_g4arm8_with_arm8_g4_9_has_the_training_stages_only(tmp_path):
    run, body, put, shm, home, g = t8t.t8_world(tmp_path)
    for stage in ("fresh", "plan", "qe", "pairs", "judge", "select", "v3", "report", "n1pairs", "n1report"):
        out, calls = run(stage, ARM8="G4-9", CHUNK="c1")
        assert out.returncode != 0 and "is G4-8-data's (§18, §21): not with ARM8=G4-9" in out.stdout, (stage, out.stdout)
        assert not any(c.startswith(("SETSID", "gsutil")) for c in calls), stage
    out, _ = run("generate", ARM8="G4-9", JOB="c1")
    assert out.returncode != 0 and "generate JOB=div only" in out.stdout
    out, _ = run("n1targets")
    assert out.returncode != 0 and "n1targets is G4-9's" in out.stdout
    out, _ = run("sftset", ARM8="G4-10")
    assert out.returncode != 0 and "ARM8 is G4-8 or G4-9" in out.stdout


def test_run_g4arm8_n1targets_and_sftset_of_g4_9_render_their_jobs(tmp_path):
    run, body, put, shm, home, g = t8t.t8_world(tmp_path)
    out, _ = run("n1targets", ARM8="G4-9", TARGETS="n1", TAG="n1")
    assert out.returncode != 0 and "no gs://t5-dutch-english-usc2/mbert/translator/sft/g4arm/data/G4-8/n1/report.json" in out.stdout
    put("data/G4-8/n1/report.json", "{}")
    out, _ = run("n1targets", ARM8="G4-9", TARGETS="n1", TAG="n2")
    assert out.returncode == 0 and "started g4a9_n1targets_n2 on w0" in out.stdout, out.stdout + out.stderr
    b = body("g4a9_n1targets_n2")
    for x in ("g4arm8.py n1targets --plan", "/data/G4-8/plan/plan.parquet", "/data/G4-8/n1/report.json",
              "/data/G4-8/$c/select-n1/choice.parquet", "/data/G4-8/$c/select-n1/targets.parquet",
              "--chunk-dir c1=", "--chunk-dir c3=", "/data/G4-9/targets/targets.parquet",
              "/data/G4-9/targets/targets.json", "N1TARGETS_DONE"):
        assert x in b, x
    assert b.index("/data/G4-9/targets/targets.parquet") < b.index("/data/G4-9/targets/targets.json") and "rm " not in b
    assert f"{shm}/translator/g4arm9/n1targets-n2" in b
    out, _ = run("n1targets", host="1", ARM8="G4-9", TARGETS="n1", TAG="n3")
    assert out.returncode != 0 and "runs on w0 only" in out.stdout
    put("data/G4-9/targets/targets.json", json.dumps({"mode": "n1"}))
    out, _ = run("n1targets", ARM8="G4-9", TARGETS="n1", TAG="n4")
    assert "there before" in body("g4a9_n1targets_n4")
    # sftset reads G4-9's targets (G4-8-data's plan), writes $G9/sft/
    out, _ = run("sftset", ARM8="G4-9", TAG="s1")
    assert out.returncode != 0 and "data/G4-9/targets/targets.parquet" in out.stdout
    put("data/G4-9/targets/targets.parquet")
    out, _ = run("sftset", ARM8="G4-9", TAG="s2")
    assert out.returncode == 0 and "started g4a9_sftset_s2 on w0" in out.stdout, out.stdout + out.stderr
    b = body("g4a9_sftset_s2")
    for x in ("/data/G4-9/targets/targets.parquet", "/data/G4-8/plan/plan.parquet", "--arm=G4-9", "/data/G4-9/sft/smoke/",
              "/data/G4-9/sft/rlval/", "/data/G4-9/sft/manifest.json"):
        assert x in b, x
    assert "/data/G4-8/targets/" not in b and "/data/G4-8/sft" not in b
    put("data/G4-9/sft/manifest.json", "{}")
    out, _ = run("check", ARM8="G4-9", TAG="c1")
    b = body("g4a9_check_c1")
    assert out.returncode == 0 and "--arm G4-9" in b and "/data/G4-9/sft/check.txt" in b


def test_run_g4arm8_v3_with_set_n1_renders_the_g4_9_v3n1_job(tmp_path):
    run, body, put, shm, home, g = t8t.t8_world(tmp_path)
    out, calls = run("v3", SET="n1", TAG="v1")
    assert out.returncode != 0 and "no gs://t5-dutch-english-usc2/mbert/translator/sft/g4arm/data/G4-8/c1/select-n1/targets.parquet" in out.stdout
    put("data/G4-8/c1/select-n1/targets.parquet")
    out, calls = run("v3", SET="n1", TAG="v2")
    assert out.returncode != 0 and "data/G4-8/n1/report.json (§21's n1report)" in out.stdout
    assert not any(c.startswith("SETSID") for c in calls)
    put("data/G4-8/n1/report.json", "{}")
    out, _ = run("v3", SET="n1", TAG="v3")
    assert out.returncode == 0 and "started g4a8_v3n1_v3 on w0" in out.stdout, out.stdout + out.stderr
    b = body("g4a8_v3n1_v3")
    for x in ("g4arm8.py v3 --plan", "--set n1 --choice", "/data/G4-8/c1/select-n1/targets.parquet",
              "/data/G4-8/c1/select-n1/choice.parquet", "/data/G4-8/c1/select/choice.parquet", "/data/G4-8/n1/report.json",
              "/sft/blind/G4-9-V3N1/key.jsonl", "/sft/blind/G4-9-V3N1/", f"{shm}/translator/g4arm8/v3n1-v3"):
        assert x in b, x
    assert "G4-8-V3" not in b and "rm " not in b
    assert b.index("judge-input.jsonl") < b.index("/sft/blind/G4-9-V3N1/key.jsonl")
    out, _ = run("v3", host="2", SET="n1", TAG="v4")
    assert out.returncode != 0 and "runs on w0 only" in out.stdout
    out, _ = run("v3", SET="n2", TAG="v5")
    assert out.returncode != 0 and "SET is empty (G4-8-V3) or n1 (G4-9-V3N1), not n2" in out.stdout
    put("../blind/G4-9-V3N1/key.jsonl")
    out, calls = run("v3", SET="n1", TAG="v6")
    assert out.returncode != 0 and "G4-9-V3N1 has a key.jsonl: the package stays as it is" in out.stdout
    assert not any(c.startswith("SETSID") for c in calls)
    # without SET the stage is G4-8-V3's, as before
    put("data/G4-8/c1/select/targets.parquet")
    out, _ = run("v3", TAG="v7")
    assert out.returncode == 0, out.stdout + out.stderr
    b = body("g4a8_v3_v7")
    assert "--set" not in b and "/sft/blind/G4-8-V3/key.jsonl" in b and "G4-9-V3N1" not in b and "select-n1" not in b


def test_run_g4arm8_train_smoke8_adapter_div_and_report8_of_g4_9(tmp_path):
    run, body, put, shm, home, g = t8t.t8_world(tmp_path)
    for d in ("data/G4-9/sft", "data/G4-9/sft/smoke"):
        for f in ("train.parquet", "template.jinja"):
            put(f"{d}/{f}")
    put("data/G4-9/sft/smoke/manifest.json", "{}")
    put("data/G4-9/sft/manifest.json", json.dumps({"steps": 221, "eval_every": 37, "save_steps": [74, 148],
                                                   "candidates": ["G4-9-s74", "G4-9-s148", "G4-9-s221"]}))
    put("data/G4-9/sft/check.txt")
    for run_name in ("g4arm_G4-8_s8", "g4arm_G4-8"):
        out, _ = run("train", ARM8="G4-9", RUN=run_name, PHASE="prep")
        assert out.returncode != 0 and "no run of G4-9" in out.stdout, run_name
    out, calls = run("train", ARM8="G4-9", RUN="g4arm_G4-9_s8", PHASE="prep")
    assert out.returncode == 0 and out.stdout.strip().splitlines()[-1] == "prepared g4arm_G4-9_s8 on w0", out.stdout + out.stderr
    launch = (shm / "translator/g4arm9/data-g4arm_G4-9_s8/launch.sh").read_text().replace("\\", "")
    for x in ("NAME=g4arm_G4-9_s8", "STEPS=8", "--save-steps 4", "learning_rate=5e-05", "lora.lora_rank=32"):
        assert x in launch, x
    assert any("/data/G4-9/sft/smoke/train.parquet" in c for c in calls)
    out, _ = run("train", ARM8="G4-9", RUN="g4arm_G4-9", PHASE="prep")
    launch = (shm / "translator/g4arm9/data-g4arm_G4-9/launch.sh").read_text().replace("\\", "")
    assert out.returncode == 0 and "STEPS=221" in launch and "--save-steps 74,148" in launch
    out, _ = run("smoke8", ARM8="G4-9", RUN="g4arm_G4-8_s8", TAG="m1")
    assert out.returncode != 0 and "no smoke run of G4-9" in out.stdout
    out, _ = run("smoke8", ARM8="G4-9", RUN="g4arm_G4-9_s8", TAG="m2")
    b = body("g4a9_smoke8_m2")
    assert out.returncode == 0 and "runs/g4arm_G4-9_s8/adapter-step4.json" in b and "--arm=G4-9" in b
    assert "/data/G4-9/sft/smoke8-g4arm_G4-9_s8.json" in b
    out, _ = run("adapter", ARM8="G4-9", RUN="g4arm_G4-9", TAG="a1")
    b = body("g4a9_adapter_a1")
    for x in ("g4arm8.py trained --run g4arm_G4-9", "--adapter G4-9-s221=", "--adapter G4-9-s74=", "--arm=G4-9",
              "/G4-9-s$k/adapter.npz", "/G4-9-s221/adapter.npz", "/data/G4-9/sft/trained.json"):
        assert x in b, x
    assert "/G4-8-s" not in b
    # RL-val's samples with G4-9's seed
    put("data/G4-9/sft/rlval/segments-w1.parquet")
    out, _ = run("generate", host="1", ARM8="G4-9", JOB="div", MODEL="G4-0", TAG="d1")
    b = body("g4a9_gen_div_G4-0_d1")
    assert out.returncode == 0 and f"samples=1.0:{g4rl.seed_int('en-nl-g4arm/0/G4-9/div/w1')}" in b
    assert "/data/G4-9/sft/div/G4-0/{sample}/w1'" in b
    out, _ = run("generate", ARM8="G4-9", JOB="div", MODEL="G4-8-s140", TAG="d2")
    assert out.returncode != 0 and "MODEL is G4-0 or a candidate G4-9-s<step>" in out.stdout
    # report8: G4-8-s140 beside as a reference and compared
    cands = "G4-9-s74 G4-9-s148 G4-9-s221"
    for rel in ("naturalness-G4-9/predictions.parquet", "naturalness-G4-9/predictions-c2.parquet",
                "data/G4-5-split/split.json", "data/G4-5-split/refqe.parquet", "rl/G4-7/data/plan/plan.parquet"):
        put(rel)
    put("data/G4-9/sft/trained.json", json.dumps({"run": "g4arm_G4-9"}))
    for m in ("G4-0", *cands.split()):
        put(f"data/G4-9/sft/div/{m}/samples/w0/DONE")
    out, _ = run("report8", ARM8="G4-9", CANDIDATES="G4-9-s74 G4-8-s140", O6="R-nieuw", TAG="r1")
    assert out.returncode != 0 and "G4-8-s140 is no candidate of G4-9" in out.stdout
    out, _ = run("report8", ARM8="G4-9", CANDIDATES=cands, O6="R-nieuw", TAG="r2")
    assert out.returncode == 0, out.stdout + out.stderr
    b = body("g4a9_report8_r2")
    x9 = f"{shm}/translator/g4arm9/report8-r2"
    for x in (f"--candidate G4-9-s74={x9}/G4-9-s74", f"--reference G4-4={x9}/G4-4", f"--reference {REF}={x9}/{REF}",
              f"--compare {REF}", "--arm=G4-9", "for m in G4-9-s74 G4-9-s148 G4-9-s221 G4-0 G4-4 G4-8-s140; do",
              "runs/g4arm_G4-9/summary.json", "/naturalness-G4-9/pll.parquet", "/report-G4-9/tally.json",
              "/report-G4-9/", "REPORT8_TRAIN_DONE"):
        assert x in b, x
    assert "naturalness-G4-8" not in b and "report-G4-8" not in b


# ---------------------------------------------------------------------------------------------------- run_g4arm.sh

def g49_world(tmp_path):
    run, shm, home, g = t7.g47_world(tmp_path)
    files = {"G4-9-s74/adapter.npz": "x", "G4-9-s74/adapter.json": json.dumps({"rank": 32, "alpha": 16.0, "scale": 0.5}),
             "G4-9-s148/adapter.npz": "x", "G4-9-s148/adapter.json": json.dumps({"rank": 16, "alpha": 16.0, "scale": 1.0}),
             "naturalness-G4-9/texts.parquet": "x",
             "report-G4-9/report.json": json.dumps({"selected": "G4-9-s74", "blind_due": True})}
    for rel, text in files.items():
        (g / rel).parent.mkdir(parents=True, exist_ok=True)
        (g / rel).write_text(text)
    return run, shm, home, g


def test_g49_stages_of_run_g4arm(tmp_path):
    run, shm, home, g = g49_world(tmp_path)
    for stage in ("train", "adapter", "report7", "report", "build", "check"):
        out, calls = run(stage, ARM="G4-9")
        assert out.returncode != 0 and "is no stage of G4-9" in out.stdout, (stage, out.stdout + out.stderr)
        assert not any(c.startswith(("SETSID", "gsutil")) for c in calls), stage
    out, _ = run("translate", ARM="G4-9", MODEL="G4-9-s74", TAG="t1")
    assert out.returncode == 0 and "started g4a_translate_G4-9-s74_t1 on w0" in out.stdout, out.stdout + out.stderr
    b = (home / "job_body_g4a_translate_G4-9-s74_t1.sh").read_text()
    copy = f"{shm}/translator/g4arm/adapter-g4a_translate_G4-9-s74_t1"
    assert f"--lora {copy}" in b and "/eval/G4-9-s74/translations/w0/tiny" in b
    assert b.index("translate_class ") < b.index(f"rm -rf -- {copy}") < b.index("TRANSLATED")
    out, _ = run("translate", ARM="G4-9", MODEL="G4-9-s148", TAG="t2")
    assert out.returncode != 0 and "G4-9-s148/adapter.json is not rank 32, alpha 16" in out.stdout
    for model, arm, msg in (("G4-0", "G4-9", "ARM=G4-9: MODEL is a candidate G4-9-s<step>"),
                            ("G4-8-s140", "G4-9", "G4-8-s140 is G4-8's (ARM=G4-8)"),
                            ("G4-9-s74", "G4-8", "G4-9-s74 is G4-9's (ARM=G4-9)"),
                            ("G4-9-s74", "G4-7", "ARM=G4-7: MODEL is a candidate"),
                            ("G4-9-s74", "G4-6c", "G4-9-s74 is G4-9's"), ("G4-9-s0", "G4-9", "MODEL is G4-1")):
        for stage in ("translate", "score"):
            out, _ = run(stage, ARM=arm, MODEL=model, TAG="t3")
            assert out.returncode != 0 and msg in out.stdout, (model, arm, stage, out.stdout)
    out, _ = run("score", ARM="G4-9", MODEL="G4-9-s74", TAG="t4")
    assert out.returncode == 0 and "/eval/G4-9-s74/scores/metricx-w0.parquet" in (
        home / "job_body_g4a_score_G4-9-s74_t4.sh").read_text()
    out, _ = run("texts", ARM="G4-9", CANDIDATES="G4-9-s74 G4-8-s140", TAG="t5")
    assert out.returncode != 0 and "G4-8-s140 is no candidate of G4-9" in out.stdout
    out, _ = run("texts", ARM="G4-9", CANDIDATES="G4-9-s74 G4-9-s148 G4-9-s221", TAG="t6")
    assert out.returncode == 0, out.stdout + out.stderr
    b = (home / "job_body_g4a_texts_t6.sh").read_text()
    assert "echo 'G4-9-s74 G4-9-s148 G4-9-s221 G4-0 G4-4 G4-8-s140' |" in b and "naturalness-G4-9/texts.parquet" in b
    assert "/eval/G4-8-s140/pairs/pairs-w*.parquet" in b
    for stage, x in (("classify-c2", "naturalness-G4-9/predictions-c2.parquet"), ("pll", "naturalness-G4-9/pll.parquet")):
        out, _ = run(stage, ARM="G4-9", TAG=f"t7{stage}")
        name = {"classify-c2": "g4a_classify_c2", "pll": "g4a_pll"}[stage]
        b = (home / f"job_body_{name}_t7{stage}.sh").read_text()
        assert out.returncode == 0 and x in b and "naturalness-G4-8" not in b
    out, _ = run("classify", ARM="G4-9", TAG="t8")
    assert out.returncode == 0 and "naturalness-G4-9/predictions.parquet" in (home / "job_body_arms_classify_t8.sh").read_text()
    out, _ = run("blind", ARM="G4-9", SELECTED="G4-9-s74", TAG="t9")
    b = (home / "job_body_g4a_blind_t9.sh").read_text()
    assert out.returncode == 0 and "/eval/G4-9-s74/pairs/pairs-w*.parquet" in b
    assert "sft/blind/G4-9/" in b and "sft/blind/G4-9-tg/" in b
    for selected, msg in (("G4-9-s148", "not report8's choice"), ("G4-8-s140", "no candidate of G4-9")):
        out, _ = run("blind", ARM="G4-9", SELECTED=selected, TAG="t10")
        assert out.returncode != 0 and msg in out.stdout, (selected, out.stdout)


# ---------------------------------------------------------------------------------------------------- §22 O9

def section_22():
    text = (CODE / "protocol-g4.md").read_text(encoding="utf-8")
    start = text.index("## 22. G4-9 training")
    end = text.find("\n## 23.", start)
    return text[start:] if end < 0 else text[start:end]


def test_section_22_has_the_choice_o9_and_the_code_names_its_modes():
    section = section_22()
    o9 = re.findall(r"`O9 \[besluit Yeb[^]`]*\]: ([^`]*)`", section)
    if "CONCEPT, NIET GESTART" in section:
        # the draft names both variants; the chain refuses it
        assert sorted(set(o9)) == ["n1", "n1∩rubriek"]
    else:
        # the final version as g4arm9_train_run.sh reads it: one decision on O9
        assert len(set(o9)) == 1 and o9[0] in ("n1", "n1∩rubriek")
    assert {m: x["o9"] for m, x in g8.TARGET_MODES_9.items()} == {"n1": "n1", "intersect": "n1∩rubriek"}
    for x in ("9.126", "16.349", "`TARGETS=n1`", "`TARGETS=intersect`", "`n1targets --mode intersect`",
              "`--targets-info`", "`targets_mode`", "dezelfde `j`"):
        assert x in section, x
    assert g8.TARGET_MODES_9["n1"]["targets"] == g8.ARMS_T["G4-9"]["targets"]


def rubric_chunks(tmp_path, dirs, rep):
    """§18's select/ per chunk beside n1_chunks: for the N1 targets in turn the same sample, another sample (another
    text) or the anchor, the other sources as N1's; the choice's sha256 and §21's overlap 'zelfde sample' per part in
    the report"""
    rdirs, same = {}, collections.Counter()
    for c, d in dirs.items():
        n1_targets = {t["id"]: t for t in g8.read_rows(d / "targets.parquet")}
        choice, targets, k = [], [], 0
        for row in g8.read_rows(d / "choice.parquet"):
            row = dict(row)
            if row["reason"] == "sample":
                kind, k = k % 3, k + 1
                t = dict(n1_targets[row["id"]])
                if kind == 0:
                    same[row["part"]] += 1
                    targets.append(t)
                elif kind == 1:
                    row["j"] = t["j"] = (row["j"] + 1) % 8
                    targets.append({**t, "target": f"ander sample {row['id']}", "qe": 1.0})
                else:
                    row.update({"reason": "p", "j": -1, **dict.fromkeys(g8.TARGET_MEASURES)})
            choice.append(row)
        r = tmp_path / "rubric" / c
        g8.write_parquet(choice, r / "choice.parquet")
        g8.write_parquet(targets, r / "targets.parquet")
        rep["inputs"][c]["choice.parquet"] = g4rl.file_sha256(r / "choice.parquet")
        rdirs[c] = r
    rep["parts"]["samen"]["overlap"] = {g8.SAME_SAMPLE_N1: sum(same.values()), "sample/ander sample": 3}
    for p in g8.PARTS8:
        rep["parts"][p]["overlap"] = {g8.SAME_SAMPLE_N1: same[p], "sample/anker": 1}
    (tmp_path / "n1report.json").write_text(json.dumps(rep))
    return rdirs, same


def rubric_args(rdirs):
    return ["--mode", "intersect", *[x for c, d in rdirs.items() for x in ("--rubric-dir", f"{c}={d}")]]


def test_n1targets_intersect_keeps_the_sources_where_the_rubric_took_the_same_sample(tmp_path, capsys):
    dirs, rep = n1_chunks(tmp_path)
    rdirs, same = rubric_chunks(tmp_path, dirs, rep)
    g8.main(n1_args(tmp_path, dirs) + rubric_args(rdirs))
    rows = pq.read_table(tmp_path / "out" / "targets.parquet").to_pylist()
    rubric = {r["id"]: (r["reason"], r["j"]) for c in rdirs for r in g8.read_rows(rdirs[c] / "choice.parquet")}
    assert len(rows) == sum(same.values()) == 6 and [r["id"] for r in rows] == sorted(r["id"] for r in rows)
    assert all(rubric[r["id"]] == ("sample", r["j"]) for r in rows)
    # every N1 target whose source the rubric gave the same sample, none other
    n1 = [t for c in dirs for t in g8.read_rows(dirs[c] / "targets.parquet")]
    assert {r["id"] for r in rows} == {t["id"] for t in n1 if rubric[t["id"]] == ("sample", t["j"])}
    assert pq.read_table(tmp_path / "out" / "targets.parquet").schema.names == \
        pq.read_table(dirs["c1"] / "targets.parquet").schema.names
    info = json.loads((tmp_path / "out" / "targets.json").read_text())
    assert (info["mode"], info["o9"], info["targets"]) == ("intersect", "n1∩rubriek", 6)
    assert info["parts"] == dict(same) and info["n1_targets"] == {p: 4 for p in g8.PARTS8}
    assert info["inputs"]["c2"]["rubric_choice"] == g4rl.file_sha256(rdirs["c2"] / "choice.parquet")
    assert info["inputs"]["c2"]["rubric_targets"] == g4rl.file_sha256(rdirs["c2"] / "targets.parquet")
    assert info["targets_sha256"] == g4rl.file_sha256(tmp_path / "out" / "targets.parquet")
    assert '"mode": "intersect"' in capsys.readouterr().out
    # sftset takes them as G4-8's (§19's checks of a janchor target)
    assert g8.checked_targets(rows, g8.read_rows(tmp_path / "plan.parquet")) == info["parts"]
    # --mode n1, and no --mode: all N1 targets, mode n1
    for out, extra in (("out-n1", ["--mode", "n1"]), ("out-default", [])):
        g8.main(n1_args(tmp_path, dirs, out=out) + extra)
        info = json.loads((tmp_path / out / "targets.json").read_text())
        assert (info["mode"], info["o9"], info["targets"]) == ("n1", "n1", 12) and "n1_targets" not in info


@pytest.mark.parametrize("what, msg", [
    ("rubric_sha", "is not §18's choice.parquet that §21's n1report checked"),
    ("rubric_cover", "not the sources of chunk c2 in the plan"),
    ("rubric_j", "not one per source of §18's choice with a sample"),
    ("rubric_drop", "not one per source of §18's choice with a sample"),
    ("rubric_text", "its QE or text is not §18's target"), ("rubric_qe", "its QE or text is not §18's target"),
    ("count", "§21's report"), ("no_rubric", "--mode intersect: --rubric-dir"),
    ("rubric_with_n1", "--rubric-dir: only with --mode intersect"),
])
def test_n1targets_intersect_refuses_and_writes_nothing(tmp_path, what, msg):
    dirs, rep = n1_chunks(tmp_path)
    rdirs, same = rubric_chunks(tmp_path, dirs, rep)
    r2 = rdirs["c2"]
    kept = sorted(t["id"] for c in dirs for t in g8.read_rows(dirs[c] / "targets.parquet"))
    rubric = {r["id"]: (r["reason"], r["j"]) for r in g8.read_rows(r2 / "choice.parquet")}
    same2 = [i for i in kept if i in rubric and rubric[i][0] == "sample"
             and rubric[i][1] == next(t["j"] for t in g8.read_rows(dirs["c2"] / "targets.parquet") if t["id"] == i)]
    args = n1_args(tmp_path, dirs) + rubric_args(rdirs)
    if what == "rubric_sha":
        edit(r2 / "choice.parquet", lambda rows: rows[5].update(fails=1))
    elif what == "rubric_cover":
        edit(r2 / "choice.parquet", lambda rows: rows.pop())
        rep["inputs"]["c2"]["choice.parquet"] = g4rl.file_sha256(r2 / "choice.parquet")
    elif what == "rubric_j":
        edit(r2 / "targets.parquet", lambda rows: rows[0].update(j=(rows[0]["j"] + 3) % 8))
    elif what == "rubric_drop":
        edit(r2 / "targets.parquet", lambda rows: rows.pop())
    elif what in ("rubric_text", "rubric_qe"):
        change = {"target": "een andere tekst"} if what == "rubric_text" else {"qe": 0.5}
        edit(r2 / "targets.parquet", lambda rows: [x.update(change) for x in rows if x["id"] == same2[0]])
    elif what == "count":
        rep["parts"]["novels"]["overlap"][g8.SAME_SAMPLE_N1] += 1
    elif what == "no_rubric":
        args = n1_args(tmp_path, dirs) + ["--mode", "intersect"]
    elif what == "rubric_with_n1":
        args = n1_args(tmp_path, dirs) + rubric_args(rdirs)[2:] + ["--mode", "n1"]
    (tmp_path / "n1report.json").write_text(json.dumps(rep))
    with pytest.raises(SystemExit, match=re.escape(msg)):
        g8.main(args)
    assert not (tmp_path / "out").exists()


def targets_info(d, rows, change=None):
    """n1targets' targets.json of the targets.parquet in d, mode intersect (and the change)"""
    per = {p: sum(t["part"] == p for t in rows) for p in g8.PARTS8}
    info = {"protocol": "protocol-g4.md §22", "arm": "G4-9", "mode": "intersect", "o9": "n1∩rubriek",
            "targets": len(rows), "parts": per, "targets_sha256": g4rl.file_sha256(d / "targets.parquet"),
            **(change or {})}
    (d / "targets.json").write_text(json.dumps(info))
    return str(d / "targets.json")


def test_sftset_of_g4_9_takes_the_mode_of_n1targets_json(tmp_path, monkeypatch):
    monkeypatch.setattr(g8, "SEQ_T", 32)
    monkeypatch.setattr(g8, "SMOKE_LONGEST_T", 10)
    monkeypatch.setattr(g8, "SMOKE_RANDOM_T", 20)
    blind = tmp_path / "in" / "blind100.jsonl"
    monkeypatch.setattr(g4arm, "blind_sha256", lambda: g4rl.file_sha256(blind))
    d, plan, targets = t8t.sft_world(tmp_path)
    args = t8t.sft_args(d, tmp_path / "out")
    args.arm, args.targets_info = "G4-9", targets_info(d, targets)
    g8.sftset(args, counter=t8t.Counter())
    m = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert m["targets_mode"] == "intersect" and m["inputs"]["targets_info"] == g4rl.file_sha256(d / "targets.json")
    assert m["inputs"]["targets"] == g4rl.file_sha256(d / "targets.parquet")
    md = (tmp_path / "out" / "report.md").read_text()
    assert "Elk doel van de doorsnede van de N1-keuze (§21) en de rubriekkeuze (§18) van G4-8-data (O9 `n1∩rubriek`)" in md
    # refused before anything is written
    for k, (arm, change, msg) in enumerate((
            ("G4-8", {}, "--targets-info: G4-9's targets.json of n1targets, not with --arm G4-8"),
            ("G4-9", {"targets_sha256": "0" * 64}, "is not the targets.parquet of"),
            ("G4-9", {"targets": len(targets) + 1}, f"{len(targets) + 1} targets"),
            ("G4-9", {"parts": {"flan": 1, "novels": 2, "bt": 3}}, "targets {'flan': 1"),
            ("G4-9", {"mode": "all"}, "with a mode of ['n1', 'intersect']"),
            ("G4-9", {"arm": "G4-8"}, "not n1targets' targets.json of G4-9"))):
        args = t8t.sft_args(d, tmp_path / f"bad{k}")
        args.arm, args.targets_info = arm, targets_info(d, targets, change)
        with pytest.raises(SystemExit, match=re.escape(msg)):
            g8.sftset(args, counter=t8t.Counter())
        assert not (tmp_path / f"bad{k}").exists()


def test_report8_of_g4_9_names_the_mode_of_its_targets(tmp_path, monkeypatch, capsys):
    args = world9(tmp_path, monkeypatch)
    m = json.loads((tmp_path / "manifest.json").read_text())
    (tmp_path / "manifest.json").write_text(json.dumps({**m, "targets_mode": "intersect"}))
    r, md, _ = t8t.run8(args, capsys)
    assert r["targets_mode"] == "intersect" and r["selected"] == "G4-9-s4"
    assert ("SFT met LoRA vanaf G4-0 op de doelen van de doorsnede van de N1-keuze en de rubriekkeuze van G4-8-data (de "
            "bronnen waar N1 en de rubriek hetzelfde sample kozen; QE ≤ QE₀ als bewaker) (O9 `n1∩rubriek`)") in md
    # a mode is G4-9's, one of O9's
    (tmp_path / "manifest.json").write_text(json.dumps({**m, "targets_mode": "all"}))
    with pytest.raises(SystemExit, match=re.escape("targets mode 'all': G4-9's ['n1', 'intersect']")):
        t8t.run8(args)
    args8 = t8t.world8(tmp_path / "d", monkeypatch)
    m8 = json.loads(open(args8.manifest).read())
    with open(args8.manifest, "w") as f:
        json.dump({**m8, "targets_mode": "n1"}, f)
    with pytest.raises(SystemExit, match=re.escape("not G4-8's")):
        t8t.run8(args8)


def test_run_g4arm8_n1targets_takes_targets_and_sftset_the_json_of_g4_9(tmp_path):
    run, body, put, shm, home, g = t8t.t8_world(tmp_path)
    put("data/G4-8/n1/report.json", "{}")
    for env in ({}, {"TARGETS": "all"}, {"TARGETS": "n1∩rubriek"}):
        out, calls = run("n1targets", ARM8="G4-9", TAG="m0", **env)
        assert out.returncode != 0 and "TARGETS is n1 or intersect (§22 G4-9.0, O9)" in out.stdout, env
        assert not any(c.startswith(("SETSID", "gsutil")) for c in calls), env
    # intersect: §18's select/ of each chunk beside select-n1
    out, calls = run("n1targets", ARM8="G4-9", TARGETS="intersect", TAG="m1")
    assert out.returncode != 0 and "data/G4-8/c1/select/choice.parquet (§18's select)" in out.stdout
    assert not any(c.startswith("SETSID") for c in calls)
    for c in ("c1", "c2", "c3"):
        for f in ("choice", "targets"):
            put(f"data/G4-8/{c}/select/{f}.parquet")
    out, _ = run("n1targets", ARM8="G4-9", TARGETS="intersect", TAG="m2")
    assert out.returncode == 0 and "started g4a9_n1targets_m2 on w0" in out.stdout, out.stdout + out.stderr
    b = body("g4a9_n1targets_m2")
    x2 = f"{shm}/translator/g4arm9/n1targets-m2"
    for x in ("n1targets --plan", " --mode intersect --chunk-dir c1=", f"--rubric-dir c1={x2}/rubric/c1",
              f"--rubric-dir c3={x2}/rubric/c3", "/data/G4-8/$c/select/choice.parquet",
              "/data/G4-8/$c/select/targets.parquet", f"{x2}/rubric/$c/", "/data/G4-8/$c/select-n1/targets.parquet",
              "N1TARGETS_DONE", "(TARGETS intersect)"):
        assert x in b, x
    assert b.index("/select/targets.parquet") < b.index("g4arm8.py n1targets") and "rm " not in b
    out, _ = run("n1targets", ARM8="G4-9", TARGETS="n1", TAG="m3")
    b = body("g4a9_n1targets_m3")
    assert out.returncode == 0 and " --mode n1 --chunk-dir c1=" in b and "--rubric-dir" not in b and "/select/" not in b
    # a targets.json of the other mode stays as it is: refused, nothing started; of the same mode: there before
    put("data/G4-9/targets/targets.json", json.dumps({"mode": "n1"}, indent=1))
    out, calls = run("n1targets", ARM8="G4-9", TARGETS="intersect", TAG="m4")
    assert out.returncode != 0 and "targets.json is not of TARGETS intersect: it stays as it is" in out.stdout
    assert not any(c.startswith("SETSID") for c in calls)
    out, _ = run("n1targets", ARM8="G4-9", TARGETS="n1", TAG="m5")
    assert out.returncode == 0 and "there before, TARGETS n1" in body("g4a9_n1targets_m5")
    # sftset of G4-9 hands n1targets' json to g4arm8.py (its mode into the manifest); G4-8's has none
    put("data/G4-9/targets/targets.parquet")
    out, _ = run("sftset", ARM8="G4-9", TAG="s1")
    b = body("g4a9_sftset_s1")
    x1 = f"{shm}/translator/g4arm9/sftset-s1"
    assert out.returncode == 0 and f"/data/G4-9/targets/targets.json {x1}/in/targets.json" in b
    assert f"--arm=G4-9 --targets-info {x1}/in/targets.json --out" in b
    assert b.index(f"{x1}/in/targets.json") < b.index("g4arm8.py sftset")
    put("data/G4-8/targets/targets.parquet")
    out, _ = run("sftset", TAG="s2")
    b = body("g4a8_sftset_s2")
    assert out.returncode == 0 and "--targets-info" not in b and "targets.json" not in b
    # without n1targets' json G4-9's sftset does not start
    run2, body2, put2, *_ = t8t.t8_world(tmp_path / "b")
    put2("data/G4-9/targets/targets.parquet")
    out, calls = run2("sftset", ARM8="G4-9", TAG="s3")
    assert out.returncode != 0 and "data/G4-9/targets/targets.json (n1targets)" in out.stdout
    assert not any(c.startswith("SETSID") for c in calls)
