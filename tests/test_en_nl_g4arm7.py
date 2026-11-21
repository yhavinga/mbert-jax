"""Arm G4-7 (amendment G4-7, protocol-g4.md §14): what comes after the RL. g4arm.py's records7 and report7 (the choice
on the search part with select5, (a1), (a3) and (b) against G4-0 on the decision part, the blind review only after
those three, C2 beside C1, the training per iteration, the stop, c2cal7, smoke7, PLL) and run_g4arm.sh's stages with
ARM=G4-7 (the job scripts they write, with test_en_nl_g4arm5's fake gsutil), on small synthetic inputs (no model, no
TPU)."""

import json

import pyarrow.parquet as pq
import pytest
import test_en_nl_g4arm6 as t6

g4arm = t6.g4arm
ns, write = t6.ns, t6.write

import g4rl  # noqa: E402

CANDS = ["G4-7-i5", "G4-7-i10", "G4-7-i12"]
MODELS = [*CANDS, "G4-0", "G4-4"]
SETS = ("wmt24pp", "flores", "ntrex", "flan", "europarl", "blind")
FAILS = {"not_ended": 2, "decode_loop": 1, "empty": 0, "english_left": 1, "length": 4, "qe": 1, "c2_overflow": 1}


def iteration_files(d, i, last, stop_qe=0.2):
    """the records of iteration i as g4rl.py iter7, stop7 and sft_lora.py --grpo write them: RL-val's QE above the
    anchor by stop_qe in the last two validations (S2 then hits in the last one), an SFT eval every 5th iteration"""
    d.mkdir(parents=True, exist_ok=True)
    report = {"iteration": i, "sources": 512, "samples": 4096, "groups_without_signal": 3 + i, "fail": 10,
              "fail_share": 10 / 4096, "fails": FAILS, "reward_mean": 0.5 + 0.01 * i, "reward_mean_ok": 0.6,
              "logit_mean_ok": 0.9, "qe_pen_mean_ok": 0.2, "len_pen_mean_ok": 0.1, "qe_mean": 2.3, "p_c2_mean": 0.4,
              "origineel_share": 0.35, "ratio_median": 1.01, "adv_abs_mean": 0.5, "c2_tokens_max": 400,
              "train": {"samples": 4000, "rows": 400, "batches": 25}}
    val = {"sources": 384, "failed": 1, "fail_share": 1 / 384, "qe_sources": 384, "qe_mean": 2.5, "qe0_mean": 2.4,
           "qe_delta": stop_qe if i >= last - 1 else 0.05, "ratio_median": 1.02, "c2_overflow": 0, "p_c2_mean": 0.45,
           "c2_origineel_share": 0.4}
    previous = None
    if i:
        previous = json.loads((d.parent / f"iter-{i - 1}" / "stop.json").read_text())
    stop = g4rl.stop_rules7(val, report["fail_share"], 0.01 * i, previous)
    stop.update({"iteration": i, "sampler_gap": 0.02, "tis_next": False})
    steps = [{"step": 2 * i + 1, "tokens": 100, "loss": 0.1, "kl": 0.001, "clip_frac": 0.1},
             {"step": 2 * i + 2, "tokens": 300, "loss": 0.2, "kl": 0.003, "clip_frac": 0.2}]
    evals = []
    if (i + 1) % 5 == 0:
        evals = [{"step": 2 * i + 2, "set": n, "loss": 0.8 + 0.01 * i} for n in g4arm.SFT_SETS7]
    summary = {"eval": evals, "grpo": {"spec": g4rl.grpo_spec(), "passes": {"k1_old_ref": 0.01 * i, "sampler_gap": 0.02},
                                       "steps": steps}}
    for name, x in (("report", report), ("stop", stop), ("summary", summary)):
        (d / f"{name}.json").write_text(json.dumps(x))


CAL7 = {"ok": True, "conditions": {
    "1_test_accuracy": {"value": 0.93, "bound": ">= 0.75", "ok": True},
    "2_europarl_original": {"value": 0.81, "bound": ">= 0.75", "ok": True},
    "3_g40_greedy_share": {"value": 0.3, "bound": "in [0.02, 0.9]", "ok": True},
    "4_spread_wmt24pp": {"value": 0.6, "qe_penalty": 0.3, "bound": ">= 0.25 and >= the QE penalty's", "ok": True},
    "4_spread_flan": {"value": 0.7, "qe_penalty": 0.2, "bound": ">= 0.25 and >= the QE penalty's", "ok": True}},
    "calibration": {"g4-6c": {"segments": 11760, "spearman_c2_c1": 0.62}}, "high_correlation_with_c1": ["g4-6c"]}


def world7(tmp_path, monkeypatch, qe, share, share_c2=None, comet=None, last=11, stopped=("S2",), smoke=True):
    """report6c's world for G4-7: the evalsets of the rules' sets (2 search, 2 decision segments each), the candidates,
    G4-0 and G4-4 scored with a QE per (model, part) and COMET-22 0.85 unless comet[(model, part, set)] says otherwise;
    share (C1) and share_c2: per (model, part) how many of the part's 4 classifier texts (WMT24++ 2, FLAN 2) are
    "original"; the split, the references' QE; the records of iterations 0..last, the stop, c2cal7, smoke7"""
    import g4mix

    comet = comet or {}
    share_c2 = share_c2 or share
    monkeypatch.setattr(g4arm, "HUMAN_QE", {"wmt24pp": 2.0, "flores": 2.0, "ntrex": 2.0})
    evalsets, assign = [], {}
    for s in SETS:
        for k in range(4):
            i = f"{s}-{k}"
            ref = None if s in ("flan", "blind") else "een twee drie vier"
            evalsets.append({"id": i, "set": s, "source": "one two three four", "reference": ref, "meta": "{}"})
            if s not in ("europarl", "blind"):
                assign[i] = "search" if k < 2 else "decision"
    write(evalsets, tmp_path / "evalsets.parquet")
    digest = g4mix.digest(dict(sorted(assign.items())))
    monkeypatch.setattr(g4arm, "SPLIT_DIGEST6C", digest)
    (tmp_path / "split.json").write_text(json.dumps({"assign": dict(sorted(assign.items())), "digest": digest,
                                                     "seed": "x", "share": 0.5, "sets": {}}))
    write([{"id": r["id"], "qe_source_reference": 2.0} for r in evalsets], tmp_path / "refqe.parquet")
    c1, c2 = [], []
    for name in qe:
        d = tmp_path / name
        rows = []
        for r in evalsets:
            part = assign.get(r["id"], "search")
            # i12: 7 words (1.75 x the reference's); one MTLD factor either way (armeval.mtld: 7.0)
            hyp = "een twee een twee drie vier vijf" if name == "G4-7-i12" else "een twee drie vier"
            rows.append({"id": r["id"], "set": r["set"], "source": r["source"], "hypothesis": hyp,
                         "reference": r["reference"], "ended": not (name == "G4-7-i12" and r["id"].endswith("0")),
                         "decode_loop": False, "english_left": False, "empty": False, "qe": qe[name][part],
                         "comet": comet.get((name, part, r["set"]), 0.85)})
        write([{k: v for k, v in x.items() if k not in ("qe", "comet")} for x in rows], d / "pairs" / "pairs-w0.parquet")
        write([{"id": x["id"], "qe_source_hypothesis": x["qe"],
                "ref_source_hypothesis_reference": x["qe"] if x["reference"] else None} for x in rows],
              d / "scores" / "metricx-w0.parquet")
        write([{"id": x["id"], "comet22": x["comet"]} for x in rows if x["reference"]], d / "scores" / "comet-w0.parquet")
        for preds, n_original in ((c1, share), (c2, share_c2)):
            seen = {"search": 0, "decision": 0}
            for r in evalsets:
                if r["set"] in ("wmt24pp", "flan"):
                    part = assign[r["id"]]
                    preds.append({"model": name, "set": r["set"], "id": r["id"], "truncated": name == "G4-7-i12",
                                  "p_original": 0.9 if seen[part] < n_original[name][part] else 0.1})
                    seen[part] += 1
    # the control texts of the classify run (texts adds them): in both files, in no measure
    for preds in (c1, c2):
        preds += [{"model": "TG", "set": "flan", "id": "flan-0", "truncated": False, "p_original": 0.2}]
    write(c1, tmp_path / "pred.parquet")
    write(c2, tmp_path / "pred-c2.parquet")
    rec = tmp_path / "records"
    for i in range(last + 1):
        iteration_files(rec / f"iter-{i}", i, last)
    (tmp_path / "stopped.json").write_text(json.dumps({"last_iteration": last, "stop": list(stopped),
                                                       "reason": "S2 after two validations"}))
    (tmp_path / "cal7.json").write_text(json.dumps(CAL7))
    (tmp_path / "smoke7.json").write_text(json.dumps({"passed": True, "checks": {"judges": {"value": 256, "holds": True}},
                                                      "estimate": {"hours": 20.5}}))
    return ns(evalsets=str(tmp_path / "evalsets.parquet"), split=str(tmp_path / "split.json"),
              refqe=str(tmp_path / "refqe.parquet"), base=str(tmp_path / "G4-0"),
              predictions=str(tmp_path / "pred.parquet"), predictions_c2=str(tmp_path / "pred-c2.parquet"),
              candidate=[f"{n}={tmp_path / n}" for n in CANDS], reference=[f"G4-4={tmp_path / 'G4-4'}"],
              records=str(rec), stopped=str(tmp_path / "stopped.json"), cal7=str(tmp_path / "cal7.json"),
              smoke=str(tmp_path / "smoke7.json") if smoke else None, pll=None, tally=None,
              out=str(tmp_path / "report.md"))


# i5 holds (a1) on the search part with the larger (b) there; i10 holds (a1) with a smaller (b); i12 has the largest (b)
# but not (a1). On the decision part i10 would have been better: the choice must not look there.
QE = {"G4-7-i5": {"search": 1.9, "decision": 1.9}, "G4-7-i10": {"search": 1.8, "decision": 1.0},
      "G4-7-i12": {"search": 3.0, "decision": 3.0}, "G4-0": {"search": 1.5, "decision": 1.5},
      "G4-4": {"search": 1.6, "decision": 1.6}}
SHARE = {"G4-7-i5": {"search": 4, "decision": 2}, "G4-7-i10": {"search": 3, "decision": 4},
         "G4-7-i12": {"search": 4, "decision": 4}, "G4-0": {"search": 1, "decision": 1},
         "G4-4": {"search": 1, "decision": 1}}


def run7(tmp_path, monkeypatch, capsys=None, **kw):
    args = world7(tmp_path, monkeypatch, kw.pop("qe", QE), kw.pop("share", SHARE), **kw)
    g4arm.report7(args)
    r = json.loads((tmp_path / "report.json").read_text())
    line = None
    if capsys:
        line = json.loads(next(x for x in capsys.readouterr().out.splitlines() if x.startswith("REPORT7 "))[8:])
    return r, (tmp_path / "report.md").read_text(), line


def tally(arm, a2=True, c=True):
    return {"arm": arm, "passages": 100, "arm_better": 50, "base_better": 20, "equal": 30, "share_of_unequal": 0.71,
            "errors_arm": 10, "errors_base": 12, "doubt": 2, "a2": a2, "c": c, "per_set": {}}


# ---------------------------------------------------------------------------------------------------- report7

def test_report7_chooses_on_the_search_part_and_decides_a1_a3_b_on_the_decision_part(tmp_path, monkeypatch, capsys):
    # (a3) at its bound: i5's COMET on the decision part of every set exactly G4-0's - 0.005
    comet = {("G4-7-i5", "decision", s): 0.845 for s in g4arm.A3_SETS}
    r, md, line = run7(tmp_path, monkeypatch, capsys, comet=comet)
    assert r["selected"] == "G4-7-i5" and r["selection"].startswith("a1 on the search part")
    assert r["iterations_of"] == {"G4-7-i5": 5, "G4-7-i10": 10, "G4-7-i12": 12}
    assert r["parts"]["search"]["models"]["G4-7-i5"]["b_delta"] == pytest.approx(0.75)
    assert r["parts"]["decision"]["models"]["G4-7-i10"]["b_delta"] == pytest.approx(0.75)
    assert r["a1"]["holds"] and r["b"]["holds"] and r["b"]["delta"] == pytest.approx(0.25)
    assert r["a3"]["holds"] and all(v["delta"] == pytest.approx(-0.005) for v in r["a3"]["sets"].values())
    assert r["a3"]["sets"]["wmt24pp"]["n"] == 2
    assert r["blind_due"] and r["blind"] is None and r["verdict"] == "wacht op de blinde beoordeling ((a2), (c))"
    assert line == {"selected": "G4-7-i5", "a1": True, "a3": True, "b": pytest.approx(0.25), "b_holds": True,
                    "blind_due": True, "verdict": r["verdict"]}
    # what the chain reads
    assert {k: r[k]["holds"] for k in ("a1", "a3", "b")} == {"a1": True, "a3": True, "b": True}
    # the other measures on both parts, all models
    c = r["counts"]["search"]["G4-7-i12"]["wmt24pp"]
    assert (c["n"], c["not_ended"], c["out_per_ref"], c["over_1_5_ref"]) == (2, 1, 1.75, 1.0)
    assert c["comet22"] == pytest.approx(0.85) and c["metricx_ref"] == pytest.approx(3.0) and c["mtld"] == 7.0
    # four distinct words never reach a factor: MTLD undefined, left out of the mean
    assert r["counts"]["search"]["G4-0"]["wmt24pp"]["mtld"] is None
    assert set(r["counts"]["decision"]) == set(MODELS) and r["counts"]["decision"]["G4-7-i5"]["flan"]["metricx_ref"] is None
    assert r["truncated"]["c2"]["G4-7-i12"] == 8 and r["truncated"]["c1"]["G4-0"] == 0
    assert "Gekozen kandidaat: **G4-7-i5**" in md and "(a3) **gehaald**" in md
    assert "| zoek | G4-7-i12 | wmt24pp | 2 | 3,000 | 3,000 | 0,8500 | 1 | 0 | 0 | 0 | 1,750 | 100,0% | 7,000 |" in md
    assert "PLL: **niet gemeten**" in md


def test_a3_fails_by_a_hair_in_one_set_and_then_no_blind_review(tmp_path, monkeypatch, capsys):
    comet = {("G4-7-i5", "decision", "wmt24pp"): 0.845, ("G4-7-i5", "decision", "flores"): 0.845,
             ("G4-7-i5", "decision", "ntrex"): 0.8449}
    r, md, line = run7(tmp_path, monkeypatch, capsys, comet=comet)
    assert r["a1"]["holds"] and r["b"]["holds"] and not r["a3"]["holds"]
    assert [s for s, v in r["a3"]["sets"].items() if not v["holds"]] == ["ntrex"]
    assert not r["blind_due"] and r["verdict"] == "helpt niet"
    assert line["a3"] is False and line["blind_due"] is False
    assert "(a3) **niet gehaald**" in md and "alleen als (a1), (a3) en (b) gelden" in md
    # (a3) on the search part is reported and chooses nothing: a worse COMET there leaves the choice
    comet = {("G4-7-i5", "search", s): 0.5 for s in g4arm.A3_SETS}
    r, _, _ = run7(tmp_path / "b", monkeypatch, comet=comet)
    assert r["selected"] == "G4-7-i5" and not r["a3_all"]["search"]["G4-7-i5"]["holds"] and r["a3"]["holds"]


def test_the_blind_tally_counts_only_after_a1_a3_and_b(tmp_path, monkeypatch):
    args = world7(tmp_path, monkeypatch, QE, SHARE)
    (tmp_path / "tally.json").write_text(json.dumps(tally("G4-7-i5")))
    args.tally = str(tmp_path / "tally.json")
    g4arm.report7(args)
    r = json.loads((tmp_path / "report.json").read_text())
    assert r["a2"] and r["c"] and r["verdict"] == "helpt"
    (tmp_path / "tally.json").write_text(json.dumps(tally("G4-7-i5", c=False)))
    g4arm.report7(args)
    assert json.loads((tmp_path / "report.json").read_text())["verdict"] == "helpt niet"
    (tmp_path / "tally.json").write_text(json.dumps(tally("G4-7-i10")))
    with pytest.raises(SystemExit, match="the selected candidate is G4-7-i5"):
        g4arm.report7(args)
    # a tally while (a1), (a3) or (b) fails: the blind review was not due
    for over in ({"comet": {("G4-7-i5", "decision", "flores"): 0.8}},
                 {"qe": {**QE, "G4-7-i5": {"search": 1.9, "decision": 2.1}}},
                 {"share": {**SHARE, "G4-7-i5": {"search": 4, "decision": 1}}}):
        args = world7(tmp_path / "x", monkeypatch, over.get("qe", QE), over.get("share", SHARE), comet=over.get("comet"))
        (tmp_path / "tally.json").write_text(json.dumps(tally("G4-7-i5")))
        args.tally = str(tmp_path / "tally.json")
        with pytest.raises(SystemExit, match="the blind review only after them"):
            g4arm.report7(args)


def test_report7_without_a1_on_the_search_part_takes_the_smallest_excess(tmp_path, monkeypatch):
    qe = {**QE, "G4-7-i5": {"search": 2.3, "decision": 1.9}, "G4-7-i10": {"search": 2.2, "decision": 1.0}}
    r, _, _ = run7(tmp_path, monkeypatch, qe=qe)
    assert r["selected"] == "G4-7-i10" and r["selection"].startswith("no candidate holds (a1)")


def test_c2_beside_c1_and_the_flag_c2_rises_c1_not(tmp_path, monkeypatch):
    # i12: C2 says 4 of 4 "original" on both parts, C1 only as many as G4-0 on the search part
    share = {**SHARE, "G4-7-i12": {"search": 1, "decision": 4}}
    share_c2 = {**share, "G4-7-i12": {"search": 4, "decision": 4}, "G4-0": {"search": 1, "decision": 1}}
    r, md, _ = run7(tmp_path, monkeypatch, share=share, share_c2=share_c2)
    s = r["c2"]["search"]["G4-7-i12"]
    assert s["c1"]["pooled"] == 0.25 and s["c2"]["pooled"] == 1.0 and s["c2"]["wmt24pp"] == 1.0
    assert s["c2_delta"]["pooled"] == pytest.approx(0.75) and s["c1_delta"]["pooled"] == 0.0 and s["c2_up_c1_not"]
    # on the decision part C1 rose too: no flag; G4-0 against itself: none
    assert not r["c2"]["decision"]["G4-7-i12"]["c2_up_c1_not"] and not r["c2"]["search"]["G4-0"]["c2_up_c1_not"]
    assert "| zoek | G4-7-i12 | 50,0% | 0,0% | 25,0% | +0,000 | 100,0% | 100,0% | 100,0% | +0,750 | **ja** |" in md
    # C2 on other texts than C1's classify run: refused
    args = world7(tmp_path / "x", monkeypatch, QE, SHARE)
    rows = pq.read_table(args.predictions_c2).to_pylist()
    write(rows[:-2], tmp_path / "x" / "pred-c2.parquet")
    with pytest.raises(SystemExit, match="not C2 on the texts of"):
        g4arm.report7(args)


def test_report7_tables_every_iteration_and_the_stop(tmp_path, monkeypatch):
    r, md, _ = run7(tmp_path, monkeypatch)
    its = r["iterations"]
    assert [x["i"] for x in its] == list(range(12)) and [x["n"] for x in its][-1] == 12
    x = its[3]
    # token-weighted over the steps: (100 x 0.1 + 300 x 0.2) / 400
    assert x["clip_frac"] == pytest.approx(0.175) and x["kl"] == pytest.approx(0.0025)
    assert x["k1_old_ref"] == pytest.approx(0.03) and x["sampler_gap"] == 0.02 and not x["tis"]
    assert x["fails"]["c2_overflow"] == pytest.approx(1 / 4096) and x["groups_without_signal"] == 6
    assert (x["step_first"], x["step_last"], x["steps"]) == (7, 8, 2)
    assert x["val"]["qe_delta"] == 0.05 and set(x["rules"]) == set(g4arm.RULES7) and x["stop"] == []
    assert [y["i"] for y in its if y["sft"]] == [4, 9] and its[4]["sft"] == {k: pytest.approx(0.84) for k in g4arm.SFT_SETS7}
    assert its[-1]["stop"] == ["S2"] and its[-1]["rules"]["S2"]["hit"]
    assert "Gestopt na iteratie 11 (adapter G4-7-i12) op S2: S2 after two validations." in md
    assert "| 11 | 0,3% | +0,200 | 1,020 | 0,450 | 40,0% | 0 | 0,003 | 0,200* | 0,110 | 1,020 | 0,002 |" in md
    assert "| 3 | 7–8 (2) | 0,0300 | 0,0200 | uit | uit | 0,1750 | 0,0025 | 0,1750 | 6 van 512 | 4000 |" in md
    assert "| 4 | 10 | 0,840 | 0,840 | 0,840 |" in md
    # c2cal7 (Spearman with C1 0.62 >= 0.5: the sentence of G4-7.10) and smoke7
    assert r["cal7"]["high"] == ["g4-6c"] and g4arm.HIGH_SPEARMAN_TEXT7 in md
    assert "Kalibratie `g4-6c`: Spearman per segment C2~C1 0,6200 over 11760 segmenten." in md
    assert "Geslaagd: ja; controles 1, niet gehaald: geen; schatting {\"hours\": 20.5}." in md


def test_report7_holds_the_records_to_the_stop(tmp_path, monkeypatch):
    # an iteration before the last that stopped
    args = world7(tmp_path, monkeypatch, QE, SHARE)
    p = tmp_path / "records" / "iter-3" / "stop.json"
    s = json.loads(p.read_text())
    s["rules"]["S6"]["hit"], s["stop"] = True, ["S6"]
    p.write_text(json.dumps(s))
    with pytest.raises(SystemExit, match="iteration 3 stopped"):
        g4arm.report7(args)
    # the chain's stop is not the last iteration's
    args = world7(tmp_path / "b", monkeypatch, QE, SHARE, stopped=("S4",))
    with pytest.raises(SystemExit, match="the last iteration stopped on \\['S2'\\], the chain says \\['S4'\\]"):
        g4arm.report7(args)
    # a stop.json whose stop is not its rules' hits; a record of another iteration; a summary without GRPO
    args = world7(tmp_path / "c", monkeypatch, QE, SHARE)
    p = tmp_path / "c" / "records" / "iter-2" / "stop.json"
    p.write_text(json.dumps({**json.loads(p.read_text()), "stop": ["S1"]}))
    with pytest.raises(SystemExit, match="its rules hit"):
        g4arm.report7(args)
    args = world7(tmp_path / "d", monkeypatch, QE, SHARE)
    p = tmp_path / "d" / "records" / "iter-2" / "report.json"
    p.write_text(json.dumps({**json.loads(p.read_text()), "iteration": 1}))
    with pytest.raises(SystemExit, match="not 2"):
        g4arm.report7(args)
    args = world7(tmp_path / "e", monkeypatch, QE, SHARE)
    (tmp_path / "e" / "records" / "iter-0" / "summary.json").write_text(json.dumps({"eval": []}))
    with pytest.raises(SystemExit, match="no GRPO"):
        g4arm.report7(args)
    # the chain's stop names rules of G4-7.5 only
    args = world7(tmp_path / "f", monkeypatch, QE, SHARE, stopped=("S5",))
    with pytest.raises(SystemExit, match="stop \\['S5'\\]"):
        g4arm.report7(args)


def test_report7_takes_the_candidates_of_g4_7_7_only(tmp_path, monkeypatch):
    args = world7(tmp_path, monkeypatch, QE, SHARE)
    for cands, msg in ((["G4-7-i10", "G4-7-i5", "G4-7-i12"], "order of their iterations"),
                       (["G4-7-i5", "G4-7-i12"], "G4-7.7 takes the adapters after 5, 10 iterations"),
                       (["G4-7-i5", "G4-7-i10", "G4-7-i13"], "after 12 iterations"),
                       (["G4-7-i5", "G4-7-i10", "G4-7-i11", "G4-7-i12"], "G4-7.7 takes"),
                       (["G4-7-i5", "G4-7-i10", "G4-7-i31"], "no candidate of G4-7"),
                       (["G4-6c-s240", "G4-7-i5", "G4-7-i10"], "no candidate of G4-7")):
        args.candidate = [f"{n}={tmp_path / n}" for n in cands]
        with pytest.raises(SystemExit, match=msg):
            g4arm.report7(args)
    # the last before a stop can be the adapter after the stopping iteration or the one before it
    assert g4arm.check_candidates7(["G4-7-i5", "G4-7-i10", "G4-7-i11"], 11) == [5, 10, 11]
    assert g4arm.check_candidates7(["G4-7-i5", "G4-7-i10"], 11) == [5, 10]
    assert g4arm.check_candidates7([f"G4-7-i{k}" for k in range(5, 31, 5)], 29) == list(range(5, 31, 5))
    assert g4arm.check_candidates7(["G4-7-i3"], 2) == [3]


def test_report7_refuses_another_split_and_a_failed_c2cal7(tmp_path, monkeypatch):
    args = world7(tmp_path, monkeypatch, QE, SHARE)
    monkeypatch.setattr(g4arm, "SPLIT_DIGEST6C", "f953c026")
    with pytest.raises(SystemExit, match="not the split of G4-5.4"):
        g4arm.report7(args)
    args = world7(tmp_path / "b", monkeypatch, QE, SHARE)
    bad = json.loads(json.dumps(CAL7))
    bad["ok"], bad["conditions"]["3_g40_greedy_share"]["ok"] = False, False
    (tmp_path / "b" / "cal7.json").write_text(json.dumps(bad))
    with pytest.raises(SystemExit, match="c2cal7 did not pass \\(\\['3_g40_greedy_share'\\]\\)"):
        g4arm.report7(args)


def test_report7_low_spearman_no_smoke_and_pll_measured(tmp_path, monkeypatch):
    args = world7(tmp_path, monkeypatch, QE, SHARE, smoke=False)
    low = {**CAL7, "calibration": {"g4-6c": {"segments": 100, "spearman_c2_c1": 0.31}}}
    (tmp_path / "cal7.json").write_text(json.dumps(low))
    pll = [{"model": r["model"], "set": r["set"], "id": r["id"], "tokens": 10,
            "pll": None if r["id"] == "flan-3" else -20.0} for r in pq.read_table(args.predictions).to_pylist()]
    write(pll, tmp_path / "pll.parquet")
    args.pll = str(tmp_path / "pll.parquet")
    g4arm.report7(args)
    r = json.loads((tmp_path / "report.json").read_text())
    md = (tmp_path / "report.md").read_text()
    assert r["cal7"]["high"] == [] and g4arm.HIGH_SPEARMAN_TEXT7 not in md
    assert "Geen verslag van `smoke7`" in md and "niet gemeten" not in md
    v = r["pll"]["decision"]["G4-7-i5"]["flan"]
    assert (v["n"], v["skipped"], v["pll"], v["pll_per_token"]) == (1, 1, -20.0, -2.0)
    assert "| beslis | G4-7-i5 | flan | 1 | 1 | -20,000 | -2,000 |" in md
    write([x for x in pll if x["model"] != "G4-4"], tmp_path / "pll.parquet")
    with pytest.raises(SystemExit, match="--pll: no row of G4-4"):
        g4arm.report7(args)


def test_report7_through_the_command_line(tmp_path, monkeypatch, capsys):
    a = world7(tmp_path, monkeypatch, QE, SHARE)
    argv = ["report7", "--evalsets", a.evalsets, "--split", a.split, "--refqe", a.refqe, "--base", a.base,
            "--predictions", a.predictions, "--predictions-c2", a.predictions_c2, "--records", a.records,
            "--stopped", a.stopped, "--cal7", a.cal7, "--smoke", a.smoke, "--out", a.out]
    for c in a.candidate:
        argv += ["--candidate", c]
    g4arm.main([*argv, "--reference", a.reference[0]])
    assert capsys.readouterr().out.startswith('REPORT7 {"selected": "G4-7-i5"')


# ---------------------------------------------------------------------------------------------------- records7

def listing(n=3, **over):
    rl = "gs://b/mbert/translator/sft/g4arm/rl/G4-7"
    out = {"iterations": [{"i": i, "report": f"{rl}/iter/{i}/train/report.json", "stop": f"{rl}/iter/{i}/stop.json",
                           "summary": f"{rl}/runs/r{i}/summary.json"} for i in range(n)],
           "stopped": {"last_iteration": n - 1, "stop": [], "reason": "no stop"},
           "smoke": f"{rl}/smoke7/smoke7.json", "cal7": f"{rl}/c2/cal7/report.json"}
    out.update(over)
    return out


def test_records7_lists_the_copies_into_report7s_layout(tmp_path, capsys):
    pairs = g4arm.records7_list(listing(), tmp_path / "rec")
    assert len(pairs) == 3 * 3 + 2
    assert pairs[0] == ("gs://b/mbert/translator/sft/g4arm/rl/G4-7/iter/0/train/report.json",
                        tmp_path / "rec" / "iter-0" / "report.json")
    assert pairs[5] == ("gs://b/mbert/translator/sft/g4arm/rl/G4-7/runs/r1/summary.json",
                        tmp_path / "rec" / "iter-1" / "summary.json")
    assert [d.name for _, d in pairs[-2:]] == ["cal7.json", "smoke7.json"]
    assert all(d.parent.is_dir() for _, d in pairs)
    assert json.loads((tmp_path / "rec" / "stopped.json").read_text()) == listing()["stopped"]
    assert len(g4arm.records7_list(listing(smoke=None), tmp_path / "rec2")) == 3 * 3 + 1
    lst = tmp_path / "records.json"
    lst.write_text(json.dumps(listing(2)))
    g4arm.main(["records7", "--list", str(lst), "--out", str(tmp_path / "rec3")])
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 8 and lines[3] == ("gs://b/mbert/translator/sft/g4arm/rl/G4-7/iter/1/train/report.json "
                                            f"{tmp_path / 'rec3' / 'iter-1' / 'report.json'}")


def test_records7_refuses_a_broken_list(tmp_path):
    bad_its = listing()
    bad_its["iterations"][1]["i"] = 2
    for value, msg in ((listing(iterations=[]), "no iterations"), (bad_its, "not 0, 1"),
                       (listing(stopped={"last_iteration": 1, "stop": [], "reason": "x"}), "stopped"),
                       (listing(stopped={"last_iteration": 2, "stop": ["S5"], "reason": "x"}), "stopped"),
                       (listing(stopped={"last_iteration": 2, "stop": []}), "stopped"),
                       (listing(cal7=None), "cal7 is None"), (listing(smoke="/tmp/smoke7.json"), "not a gs:// path"),
                       (listing(31), "at most 30")):
        with pytest.raises(SystemExit, match=msg):
            g4arm.records7_list(value, tmp_path / "rec")
    one = listing()
    one["iterations"][2]["summary"] = "gs://b/x y.json"
    with pytest.raises(SystemExit, match="iteration 2's summary"):
        g4arm.records7_list(one, tmp_path / "rec")


# ---------------------------------------------------------------------------------------------------- run_g4arm.sh

def g47_world(tmp_path):
    """test_en_nl_g4arm6's world for ARM=G4-7: run_arms.sh beside run_g4arm.sh, the DUMB venv and a git that says its
    checkout is at ae11478 and clean, and in the fake bucket what the stages of G4-7 read"""
    import test_en_nl_g4arm5 as t5

    run, shm, home, g = t6.g46b_world(tmp_path)
    (tmp_path / "code" / "run_arms.sh").write_text((t5.CODE / "run_arms.sh").read_text().replace("/dev/shm/", f"{shm}/"))
    git = tmp_path / "bin" / "git"
    git.write_text('#!/bin/bash\ncase "$*" in *"rev-parse HEAD"*) echo ae114789bf2eae153e4e3bea579299a617dcd417 ;; '
                   '*"status --porcelain"*) ;; *) exit 1 ;; esac\n')
    git.chmod(0o755)
    dpy = home / "mbert-dumb/.venv/bin/python"
    dpy.parent.mkdir(parents=True)
    dpy.write_text("#!/bin/bash\nexit 0\n")
    dpy.chmod(0o755)
    files = {"G4-7-i5/adapter.npz": "x", "G4-7-i5/adapter.json": json.dumps({"rank": 32, "alpha": 16.0, "scale": 0.5}),
             "G4-7-i10/adapter.npz": "x", "G4-7-i10/adapter.json": json.dumps({"rank": 16, "alpha": 16.0, "scale": 1.0}),
             "data/eval/classes-w0.txt": "tiny small\n", "data/eval/segments-w0.parquet": "x",
             "rl/G4-7/c2/model/params.pkl": "x", "rl/G4-7/c2/model/config.json": "{}",
             "naturalness-G4-7/texts.parquet": "x", "naturalness-G4-7/predictions.parquet": "x",
             "naturalness-G4-7/predictions-c2.parquet": "x", "report-G4-7/records.json": json.dumps(listing()),
             "data/G4-5-split/split.json": "{}", "data/G4-5-split/refqe.parquet": "x",
             "report-G4-7/report.json": json.dumps({"selected": "G4-7-i10", "blind_due": True})}
    for rel, body in files.items():
        (g / rel).parent.mkdir(parents=True, exist_ok=True)
        (g / rel).write_text(body)
    return run, shm, home, g


def test_g47_refuses_the_stages_of_the_rl_and_other_arms_refuse_g47s(tmp_path):
    run, shm, home, g = g47_world(tmp_path)
    for stage in ("train", "adapter", "smoke-translate", "smoke-score", "smoke-report", "build", "evaldata", "build6",
                  "check6", "cpuproof6", "report", "report5"):
        out, calls = run(stage, ARM="G4-7")
        assert out.returncode != 0 and "is no stage of G4-7" in out.stdout, (stage, out.stdout + out.stderr)
        assert not any(c.startswith(("SETSID", "gsutil")) for c in calls), stage
    for stage in ("ends6c", "report6c"):
        out, _ = run(stage, ARM="G4-7")
        assert out.returncode != 0 and "is G4-6c's" in out.stdout
    for stage in ("classify-c2", "pll", "report7"):
        out, _ = run(stage, ARM="G4-6c")
        assert out.returncode != 0 and f"{stage} is G4-7's" in out.stdout
    out, _ = run("translate", ARM="G4-7", MODEL="G4-7-i5", SET="heldout")
    assert out.returncode != 0 and "SET=eval only" in out.stdout


def test_g47_translates_and_scores_its_candidates_only(tmp_path):
    run, shm, home, g = g47_world(tmp_path)
    out, calls = run("translate", ARM="G4-7", MODEL="G4-7-i5", TAG="t2")
    assert out.returncode == 0 and "started g4a_translate_G4-7-i5_t2 on w0" in out.stdout, out.stdout + out.stderr
    body = (home / "job_body_g4a_translate_G4-7-i5_t2.sh").read_text()
    copy = f"{shm}/translator/g4arm/adapter-g4a_translate_G4-7-i5_t2"
    assert f"--lora {copy}" in body and "/eval/G4-7-i5/translations/w0/tiny" in body and "/w0/small" in body
    assert body.index("translate_class ") < body.index(f"rm -rf -- {copy}") < body.index("TRANSLATED")
    assert any("sft/g4arm/G4-7-i5/adapter.npz" in c for c in calls)
    # a candidate's adapter is rank 32 / alpha 16 (nothing stacked); the refused copy is removed
    out, _ = run("translate", ARM="G4-7", MODEL="G4-7-i10", TAG="t3")
    assert out.returncode != 0 and "G4-7-i10/adapter.json is not rank 32, alpha 16" in out.stdout
    assert not (shm / "translator/g4arm/adapter-g4a_translate_G4-7-i10_t3").exists()
    for model, arm, msg in (("G4-4", "G4-7", "ARM=G4-7: MODEL is a candidate"), ("G4-0", "G4-7", "ARM=G4-7: MODEL"),
                            ("G4-7-i31", "G4-7", "MODEL is G4-1"), ("G4-7-i0", "G4-7", "MODEL is G4-1"),
                            ("G4-7-i5", "G4-6c", "G4-7-i5 is G4-7's")):
        for stage in ("translate", "score"):
            out, _ = run(stage, ARM=arm, MODEL=model, TAG="t4")
            assert out.returncode != 0 and msg in out.stdout, (model, arm, stage, out.stdout)
    out, _ = run("score", ARM="G4-7", MODEL="G4-7-i5", TAG="t5")
    assert out.returncode == 0, out.stdout + out.stderr
    body = (home / "job_body_g4a_score_G4-7-i5_t5.sh").read_text()
    assert "/eval/G4-7-i5/scores/metricx-w0.parquet" in body and "comet_score.py score" in body


def test_g47_texts_classify_and_c2_and_pll_on_its_own_texts(tmp_path):
    run, shm, home, g = g47_world(tmp_path)
    out, _ = run("texts", ARM="G4-7")
    assert out.returncode != 0 and "CANDIDATES" in (out.stdout + out.stderr)
    out, _ = run("texts", ARM="G4-7", CANDIDATES="G4-7-i5 G4-6c-s240", TAG="t1b")
    assert out.returncode != 0 and "G4-6c-s240 is no candidate of G4-7" in out.stdout
    out, _ = run("texts", ARM="G4-7", CANDIDATES="G4-7-i5 G4-7-i10 G4-7-i12", TAG="t2")
    assert out.returncode == 0, out.stdout + out.stderr
    body = (home / "job_body_g4a_texts_t2.sh").read_text()
    assert "echo 'G4-7-i5 G4-7-i10 G4-7-i12 G4-0 G4-4' |" in body and "naturalness-G4-7/texts.parquet" in body
    # C1: run_arms.sh's classify, unchanged, on G4-7's texts
    out, _ = run("classify", ARM="G4-7", TAG="t3")
    assert out.returncode == 0 and "started arms_classify_t3" in out.stdout, out.stdout + out.stderr
    body = (home / "job_body_arms_classify_t3.sh").read_text()
    assert "naturalness-G4-7/texts.parquet" in body and "naturalness-G4-7/predictions.parquet" in body
    assert "sft/arms/naturalness/classifier/params.pkl" in body and "rl/G4-7" not in body
    # C2 on the same texts, C1's venv and checkout, 1,024 tokens
    out, _ = run("classify-c2", ARM="G4-7", TAG="t4")
    assert out.returncode == 0 and "started g4a_classify_c2_t4 on w0" in out.stdout, out.stdout + out.stderr
    body = (home / "job_body_g4a_classify_c2_t4.sh").read_text()
    for x in ("rl/G4-7/c2/model/params.pkl", "rl/G4-7/c2/model/config.json", "naturalness-G4-7/texts.parquet",
              f"{home}/mbert-dumb/.venv/bin/python -u", "naturalness.py classify", "--max-length 1024",
              f"cd {home}/mbert-dumbx", "naturalness-G4-7/predictions-c2.parquet", "CLASSIFIED_C2:"):
        assert x in body, x
    assert "HF_HUB_OFFLINE" not in body
    out, _ = run("pll", ARM="G4-7", TAG="t5")
    assert out.returncode == 0, out.stdout + out.stderr
    body = (home / "job_body_g4a_pll_t5.sh").read_text()
    for x in ("naturalness.py pll", "culturax-nl-230m-chunk1024/ckpt_keep/653000", "naturalness-G4-7/texts.parquet",
              "naturalness-G4-7/pll.parquet", "PLL: "):
        assert x in body, x
    # without C2's model, or with a checkout elsewhere: refused before the job
    (g / "rl/G4-7/c2/model/params.pkl").unlink()
    out, calls = run("classify-c2", ARM="G4-7", TAG="t6")
    assert out.returncode != 0 and "rl/G4-7/c2/model/params.pkl" in out.stdout
    assert not any(c.startswith("SETSID") for c in calls)
    (tmp_path / "bin" / "git").write_text("#!/bin/bash\necho 0000\n")
    out, _ = run("pll", ARM="G4-7", TAG="t7")
    assert out.returncode != 0 and "is not at ae114789" in out.stdout


def test_g47_report7_and_blind(tmp_path):
    run, shm, home, g = g47_world(tmp_path)
    out, _ = run("report7", ARM="G4-7", CANDIDATES="G4-7-i5 G4-7-i10", TAG="t2")
    assert out.returncode == 0 and "started g4a_report7_t2 on w0" in out.stdout, out.stdout + out.stderr
    body = (home / "job_body_g4a_report7_t2.sh").read_text()
    x7 = f"{shm}/translator/g4arm/report7-t2"
    for x in (f"--candidate G4-7-i5={x7}/G4-7-i5", f"--candidate G4-7-i10={x7}/G4-7-i10", f"--reference G4-4={x7}/G4-4",
              "for m in G4-7-i5 G4-7-i10 G4-0 G4-4; do", "data/G4-5-split/split.json", "data/G4-5-split/refqe.parquet",
              "naturalness-G4-7/predictions.parquet", "naturalness-G4-7/predictions-c2.parquet",
              "report-G4-7/records.json", f"g4arm.py records7 --list {x7}/records.json --out {x7}/records",
              'gsutil -q cp "$src" "$dst"', f"--stopped {x7}/records/stopped.json", f"--cal7 {x7}/records/cal7.json",
              f"--smoke {x7}/records/smoke7.json", "naturalness-G4-7/pll.parquet", "report-G4-7/tally.json",
              "$EXTRA7 --out", "report-G4-7/", "REPORT7_DONE"):
        assert x in body, x
    out, _ = run("report7", ARM="G4-7", CANDIDATES="G4-7-i5 G4-6c-s240", TAG="t3")
    assert out.returncode != 0 and "G4-6c-s240 is no candidate of G4-7" in out.stdout
    (g / "report-G4-7/records.json").unlink()
    out, calls = run("report7", ARM="G4-7", CANDIDATES="G4-7-i5 G4-7-i10", TAG="t4")
    assert out.returncode != 0 and "report-G4-7/records.json" in out.stdout
    assert not any(c.startswith("SETSID") for c in calls)
    # the blind packages of report7's choice, only with (a1), (a3) and (b) holding
    out, _ = run("blind", ARM="G4-7", SELECTED="G4-7-i10", TAG="t5")
    assert out.returncode == 0, out.stdout + out.stderr
    body = (home / "job_body_g4a_blind_t5.sh").read_text()
    assert "/eval/G4-7-i10/pairs/pairs-w*.parquet" in body and "sft/blind/G4-7/" in body and "sft/blind/G4-7-tg/" in body
    for selected, report, msg in (("G4-7-i5", {"selected": "G4-7-i10", "blind_due": True}, "not report7's choice"),
                                  ("G4-7-i10", {"selected": "G4-7-i10", "blind_due": False}, "not report7's choice"),
                                  ("G4-6c-s240", {"selected": "G4-7-i10", "blind_due": True}, "no candidate of G4-7")):
        (g / "report-G4-7/report.json").write_text(json.dumps(report))
        out, _ = run("blind", ARM="G4-7", SELECTED=selected, TAG="t6")
        assert out.returncode != 0 and msg in out.stdout, (selected, out.stdout)
