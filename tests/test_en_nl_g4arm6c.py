"""Arm G4-6c (amendment G4-6c, a milder DPO from G4-4): the turn's end outside the log-ratio (g4dpo.end_mask,
check_ends, dpo_terms with end_tokens), the records' NLL over every target token, sft_lora.py's --save-steps and its
check of the packed files, g4arm.py's smoke6 checks of G4-6c, ends6c and report6c, and run_g4arm.sh's stages with
ARM=G4-6c, on small synthetic inputs (no tokenizer, no model, no TPU)."""

import json
import math
import random

import numpy as np
import pytest
import test_en_nl_g4arm6 as t6

g4arm, g4dpo, sft_lora = t6.g4arm, t6.g4dpo, t6.sft_lora
ns, write = t6.ns, t6.write
END = [106, 107]                                   # <turn|>, \n of Gemma 4 (checked on a worker by ends6c)
OPTS6C = g4dpo.spec(**g4arm.DPO6C)


def end_pairs(n, seed=0, end=END):
    """toy pairs whose every side ends in the turn's end, as build6's completions (g4lora.completion_text)"""
    rng = random.Random(seed)
    out = []
    for k in range(n):
        prompt = [2] + [rng.randint(10, 99) for _ in range(rng.randint(3, 8))]
        out.append({"id": f"p{k:03d}", "prompt": prompt,
                    "chosen": [rng.randint(10, 99) for _ in range(rng.randint(1, 5))] + list(end),
                    "rejected": [rng.randint(10, 99) for _ in range(rng.randint(1, 5))] + list(end)})
    return out


def batch_of(pairs, seq, rows):
    packed = g4dpo.pack_rows(pairs, seq, rows)
    return {k: np.asarray([r[k] for r in packed], np.int32) for k in g4dpo.KEYS}, packed


# ---------------------------------------------------------------------------------------------------- the loss

def test_end_mask_marks_the_last_n_targets_of_every_side_which_are_the_turns_end():
    batch, packed = batch_of(end_pairs(9, seed=1), seq=40, rows=3)
    seg, tgt = batch["targets_segmentation"], batch["targets"]
    mask = g4dpo.end_mask(seg, 2)
    assert mask.dtype == np.float32 and mask.shape == seg.shape
    sides = sum(2 * len(r["pairs"]) for r in packed)
    assert mask.sum() == 2 * sides
    for b in range(len(seg)):
        for s in set(seg[b][seg[b] > 0].tolist()):
            where = np.flatnonzero(seg[b] == s)
            assert tgt[b][where[-2:]].tolist() == END and mask[b][where[-2:]].tolist() == [1.0, 1.0]
            assert not mask[b][where[:-2]].any()
    assert (mask[seg == 0] == 0).all()
    assert not g4dpo.end_mask(seg, 0).any()


def test_dpo_terms_leaves_the_turns_end_out_of_the_log_ratio_and_keeps_it_in_the_nll():
    # amendment G4-6c.3: per pair h of the sums without the last two targets of each side (policy and reference); the
    # NLL term over every target token of the chosen sides
    pairs = end_pairs(9, seed=3)
    batch, packed = batch_of(pairs, seq=40, rows=3)
    seg = batch["targets_segmentation"]
    rng = np.random.default_rng(0)
    logp = np.where(seg != 0, -rng.uniform(0.1, 3.0, seg.shape), 0.0).astype(np.float32)
    ref = np.where(seg != 0, -rng.uniform(0.1, 3.0, seg.shape), 0.0).astype(np.float32)
    loss, sums = g4dpo.dpo_terms(logp, ref, seg, OPTS6C)
    beta, alpha = OPTS6C["beta"], OPTS6C["nll_alpha"]
    assert (beta, alpha, OPTS6C["end_tokens"]) == (0.5, 2.0, 2)
    losses, chosen_ratio, chosen_all, tok_ratio, tok_all = [], 0.0, 0.0, 0, 0
    for b, r in enumerate(packed):
        for j, _ in enumerate(r["pairs"]):
            c, x = np.flatnonzero(seg[b] == 2 * j + 1), np.flatnonzero(seg[b] == 2 * j + 2)
            cr, xr = c[:-2], x[:-2]                                   # the turn's end left out
            h = (logp[b][cr].sum() - ref[b][cr].sum()) - (logp[b][xr].sum() - ref[b][xr].sum())
            losses.append(-math.log(1.0 / (1.0 + math.exp(-beta * h))))
            chosen_ratio += logp[b][cr].sum()
            chosen_all += logp[b][c].sum()
            tok_ratio += len(cr)
            tok_all += len(c)
    want = np.mean(losses) + alpha * (-chosen_all / tok_all)
    assert float(loss) == pytest.approx(want, rel=1e-5)
    assert sums["pairs"] == 9 and sums["chosen_tokens"] == tok_ratio and sums["chosen_tokens_all"] == tok_all
    assert tok_all - tok_ratio == 2 * 9 and sums["rejected_tokens_all"] - sums["rejected_tokens"] == 2 * 9
    assert float(sums["chosen_logp_sum"]) == pytest.approx(chosen_ratio, rel=1e-5)
    assert float(sums["chosen_logp_all_sum"]) == pytest.approx(chosen_all, rel=1e-5)
    # the end tokens alone move neither the log-ratio nor the rewards, only the NLL term
    bump = logp + 5.0 * g4dpo.end_mask(seg, 2) * (seg % 2 == 0)        # the rejected sides' end more likely
    loss2, sums2 = g4dpo.dpo_terms(bump, ref, seg, OPTS6C)
    assert float(sums2["dpo_sum"]) == pytest.approx(float(sums["dpo_sum"]), rel=1e-6)
    assert float(sums2["margin_sum"]) == pytest.approx(float(sums["margin_sum"]), rel=1e-6)
    assert float(loss2) == pytest.approx(float(loss), rel=1e-6)
    bump = logp + 0.5 * g4dpo.end_mask(seg, 2) * (seg % 2 == 1)        # the chosen sides' end more likely
    loss3, sums3 = g4dpo.dpo_terms(bump, ref, seg, OPTS6C)
    assert float(sums3["dpo_sum"]) == pytest.approx(float(sums["dpo_sum"]), rel=1e-6)
    assert float(loss3) == pytest.approx(float(loss) - alpha * 0.5 * 2 * 9 / tok_all, rel=1e-5)


def test_dpo_terms_with_the_masked_references_sums_is_the_same_and_end_tokens_0_is_g4_6b():
    pairs = end_pairs(12, seed=5)
    data, _ = batch_of(pairs, seq=48, rows=4)
    seg = data["targets_segmentation"]
    rng = np.random.default_rng(5)
    logp = np.where(seg != 0, -rng.uniform(0.1, 2.0, seg.shape), 0.0).astype(np.float32)
    ref = np.where(seg != 0, -rng.uniform(0.1, 2.0, seg.shape), 0.0).astype(np.float32)
    # the precomputed reference with end_tokens: its sums without the turn's end (reference_sums masks as dpo_terms)
    ref_seg = g4dpo.segment_sums(ref * (1.0 - g4dpo.end_mask(seg, 2)), seg, OPTS6C["max_segments"])
    a, sa = g4dpo.dpo_terms(logp, ref, seg, OPTS6C)
    b, sb = g4dpo.dpo_terms(logp, None, seg, OPTS6C, ref_seg=ref_seg)
    assert float(b) == pytest.approx(float(a), rel=1e-6)
    for k in g4dpo.SUMS:
        assert float(sb[k]) == pytest.approx(float(sa[k]), rel=1e-6, abs=1e-6), k
    # end_tokens 0: the sums over every token are the log-ratio's, the loss G4-6b's
    old = g4dpo.spec(**g4arm.DPO6)
    assert old["end_tokens"] == 0
    c, sc = g4dpo.dpo_terms(logp, ref, seg, old)
    assert sc["chosen_logp_all_sum"] == sc["chosen_logp_sum"] and sc["chosen_tokens_all"] == sc["chosen_tokens"]
    want = sc["dpo_sum"] / sc["pairs"] + old["nll_alpha"] * (-sc["chosen_logp_sum"] / sc["chosen_tokens"])
    assert float(c) == pytest.approx(float(want), rel=1e-6)


def test_spec_takes_end_tokens_and_g4_6cs_spec():
    assert g4dpo.spec(**g4arm.DPO6C) == {**g4dpo.DEFAULTS, **g4arm.DPO6C}
    for bad in ({"end_tokens": -1}, {"end_tokens": 1.5}, {"end_tokens": True}, {"end_tokens": "2"}):
        with pytest.raises(ValueError, match="end_tokens"):
            g4dpo.spec(**bad)


def test_pref_and_step_records_give_the_nll_over_every_target_token():
    part = {k: 0.0 for k in g4dpo.SUMS} | {"pairs": 2, "dpo_sum": 1.0, "correct": 1, "chosen_logp_sum": -8.0,
                                            "chosen_tokens": 4, "rejected_logp_sum": -2.0, "rejected_tokens": 4,
                                            "ref_chosen_logp_sum": -8.0, "ref_rejected_logp_sum": -2.0,
                                            "chosen_logp_all_sum": -9.0, "chosen_tokens_all": 6,
                                            "rejected_logp_all_sum": -3.0, "rejected_tokens_all": 6}
    r = g4dpo.pref_record("pref_x", 0, [part, part], 2, 1.0)
    assert (r["chosen_nll"], r["chosen_nll_all"], r["rejected_nll"], r["rejected_nll_all"]) == (2.0, 1.5, 0.5, 0.5)
    assert (r["chosen_tokens"], r["chosen_tokens_all"], r["rejected_tokens_all"]) == (8, 12, 12)
    s = g4dpo.step_record(1, part)
    assert (s["chosen_nll"], s["chosen_nll_all"]) == (2.0, 1.5)


def test_check_ends_holds_a_packed_file_to_its_runs_and_end_ids(tmp_path):
    g4dpo.write_rows(g4dpo.pack_rows(end_pairs(30, seed=2), 48, 4), tmp_path / "ok.parquet")
    got = g4dpo.check_ends(tmp_path / "ok.parquet", 2, END)
    assert got["segments"] == 60 and got["rows"] % 4 == 0
    with pytest.raises(SystemExit, match=r"ends in \[106, 107\], not \[1, 107\]"):
        g4dpo.check_ends(tmp_path / "ok.parquet", 2, [1, 107])
    # a side of two tokens (only the turn's end) has nothing left for the log-ratio
    short = end_pairs(4, seed=3)
    short[1]["rejected"] = list(END)
    g4dpo.write_rows(g4dpo.pack_rows(short, 48, 2), tmp_path / "short.parquet")
    with pytest.raises(SystemExit, match="not more than 2"):
        g4dpo.check_ends(tmp_path / "short.parquet", 2)
    # a segment whose targets are not one run
    rows = g4dpo.pack_rows(end_pairs(2, seed=4), 48, 1)
    rows[0]["targets_segmentation"][rows[0]["targets_segmentation"].index(2) + 1] = 0
    g4dpo.write_rows(rows, tmp_path / "gap.parquet")
    with pytest.raises(SystemExit, match="not one run"):
        g4dpo.check_ends(tmp_path / "gap.parquet", 2)


# ---------------------------------------------------------------------------------------------------- sft_lora

def test_save_steps_are_distinct_steps_before_the_last():
    assert sft_lora.save_steps_arg(None, 717) == []
    assert sft_lora.save_steps_arg("480,240,240", 717) == [240, 480]
    for bad in ("0", "717", "a,b", ","):
        with pytest.raises(SystemExit, match="--save-steps"):
            sft_lora.save_steps_arg(bad, 717)


def test_sft_lora_with_end_tokens_checks_the_packed_files_before_jax(tmp_path, capsys):
    spec = tmp_path / "dpo.json"
    spec.write_text(json.dumps(OPTS6C))
    g4dpo.write_rows(g4dpo.pack_rows(end_pairs(8, seed=6), 48, 2), tmp_path / "train.parquet")
    short = end_pairs(8, seed=7)
    short[5]["chosen"] = list(END)
    g4dpo.write_rows(g4dpo.pack_rows(short, 48, 2), tmp_path / "pref.parquet")
    args = ns(dpo=str(spec), pref_train=str(tmp_path / "train.parquet"), pref_eval=[f"pref_x={tmp_path / 'pref.parquet'}"],
              base_adapter=None, combine=None, temperatures=None, resume_step=0, init_adapter=None)
    with pytest.raises(SystemExit, match="not more than 2"):
        sft_lora.dpo_args(args, {"books": "b.parquet"})
    assert 'ENDS {"set": "train", "end_tokens": 2' in capsys.readouterr().out
    g4dpo.write_rows(g4dpo.pack_rows(end_pairs(8, seed=8), 48, 2), tmp_path / "pref.parquet")
    opts, prefs = sft_lora.dpo_args(args, {"books": "b.parquet"})
    assert opts == OPTS6C and set(prefs) == {"pref_x"}


# ---------------------------------------------------------------------------------------------------- smoke6

def smoke6c_summary(**over):
    s = t6.smoke6b_summary(**over)
    s["dpo"]["spec"] = OPTS6C
    for e in s["eval"]:
        if e.get("kind") == "pref":
            e.update({"chosen_tokens": 10000, "chosen_tokens_all": 11000, "rejected_tokens": 9000,
                      "rejected_tokens_all": 10000, "chosen_nll_all": 1.9, "rejected_nll_all": 0.12})
    s["dpo"]["steps"] = [dict(x, chosen_nll_all=0.6) for x in s["dpo"]["steps"]]
    s["loss"] = [[k, 0.69 + 2.0 * 0.6] for k in range(1, 9)]
    s["saved_adapters"] = {"4": {"keys": 820, "sum_abs": 12.5, "peak_gib": 29.1}}
    s["adapter_sum_abs"] = 13.0
    s.update(over)
    return s


SAVED4 = {"rank": 32, "alpha": 16.0, "scale": 0.5, "steps": 4}


def test_smoke6_of_g4_6c_checks_its_spec_the_end_tokens_the_loss_and_the_kept_adapter():
    adapter = {"rank": 32, "alpha": 16.0, "scale": 0.5}
    build = {"packing": {"4096": {"train": {"steps": 717}}}}
    args = (adapter, t6.G44_RUN, 4096, t6.LORA, t6.smoke_rows(), t6.smoke_rows(qe=2.25), build)
    ok = g4arm.smoke6_verdict(smoke6c_summary(), *args, start="G4-4", arm="G4-6c", saved=SAVED4)
    assert ok["passed"], {k: c for k, c in ok["checks"].items() if not c["holds"]}
    assert {"end_tokens", "loss_composition", "saved_adapter"} <= set(ok["checks"])
    # four eval rounds of G4-6c (0, 240, 480, 717) against five of G4-6b's 200 (0, 200, 400, 600, 717)
    assert ok["estimate"]["rounds"] == 4
    assert g4arm.smoke6_verdict(t6.smoke6b_summary(), *args, start="G4-4")["estimate"]["rounds"] == 5
    # G4-6b's spec is not G4-6c's
    assert not g4arm.smoke6_verdict(t6.smoke6b_summary(), *args, start="G4-4", arm="G4-6c",
                                    saved=SAVED4)["checks"]["dpo_spec"]["holds"]
    bad = smoke6c_summary(adapter_sum_abs=12.5)
    bad["loss"][0][1] += 0.01
    for e in bad["eval"]:
        if e["set"] == "pref_btval":
            e["chosen_tokens_all"] = e["chosen_tokens"]
    v = g4arm.smoke6_verdict(bad, *args, start="G4-4", arm="G4-6c", saved=SAVED4)
    assert {k for k, c in v["checks"].items() if not c["holds"]} == {"end_tokens", "loss_composition", "saved_adapter"}
    v = g4arm.smoke6_verdict(smoke6c_summary(), *args, start="G4-4", arm="G4-6c", saved={**SAVED4, "steps": 8})
    assert {k for k, c in v["checks"].items() if not c["holds"]} == {"saved_adapter"}


# ---------------------------------------------------------------------------------------------------- report6c

def report6c_world(tmp_path, monkeypatch, qe, share):
    """evalsets of the four rules' sets (2 search, 2 decision segments each), G4-6c's checkpoints, G4-0 and G4-4
    scored with a QE per (model, part); share: per (model, part) how many of the part's 4 classifier texts (WMT24++ 2,
    FLAN 2) are "original"; the split, the references' QE"""
    import g4mix

    monkeypatch.setattr(g4arm, "HUMAN_QE", {"wmt24pp": 2.0, "flores": 2.0, "ntrex": 2.0})
    evalsets, assign = [], {}
    for s in ("wmt24pp", "flores", "ntrex", "flan", "europarl", "blind"):
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
    preds = []
    for name in qe:
        d = tmp_path / name
        rows = []
        for r in evalsets:
            part = assign.get(r["id"], "search")
            hyp = "een twee drie vier vijf zes zeven" if name == "G4-6c-s717" else "een twee drie vier"
            rows.append({"id": r["id"], "set": r["set"], "source": r["source"], "hypothesis": hyp,
                         "reference": r["reference"], "ended": not (name == "G4-6c-s717" and r["id"].endswith("0")),
                         "decode_loop": False, "english_left": False, "empty": False, "qe": qe[name][part]})
        write([{k: v for k, v in x.items() if k != "qe"} for x in rows], d / "pairs" / "pairs-w0.parquet")
        write([{"id": x["id"], "qe_source_hypothesis": x["qe"],
                "ref_source_hypothesis_reference": x["qe"] if x["reference"] else None} for x in rows],
              d / "scores" / "metricx-w0.parquet")
        write([{"id": x["id"], "comet22": 0.85} for x in rows if x["reference"]], d / "scores" / "comet-w0.parquet")
        seen = {"search": 0, "decision": 0}
        for r in evalsets:
            if r["set"] in ("wmt24pp", "flan"):
                part = assign[r["id"]]
                preds.append({"model": name, "set": r["set"], "id": r["id"], "truncated": False,
                              "p_original": 0.9 if seen[part] < share[name][part] else 0.1})
                seen[part] += 1
    write(preds, tmp_path / "pred.parquet")
    evals = []
    for step in (0, 240, 480, 717):
        evals += [{"step": step, "set": n, "loss": 0.5 + step / 1e4} for n in g4arm.SFT_SETS6C]
        evals += [{"step": step, "set": n, "kind": "pref", "pairs": 500, "loss": 0.6, "accuracy": 0.7, "margin": 1.0,
                   "chosen_nll": 0.8, "chosen_nll_all": 0.75, "rejected_nll": 0.3, "rejected_nll_all": 0.28,
                   "ref_chosen_nll": 0.8, "ref_rejected_nll": 0.2} for n in g4arm.PREF_SETS6]
    summary = {"steps_done": 717, "eval": evals, "dpo": {"spec": OPTS6C}, "base_adapter": {"name": "G4-4"},
               "saved_adapters": {"240": {"sum_abs": 1.0}, "480": {"sum_abs": 2.0}}}
    (tmp_path / "train.json").write_text(json.dumps(summary))
    beside = {"eval": [{"step": 717, "set": "books", "loss": 0.86},
                       {"step": 717, "set": "pref_books", "kind": "pref", "loss": 0.001, "accuracy": 1.0, "margin": 13.6,
                        "chosen_nll": 0.858, "rejected_nll": 0.746}], "dpo": {"spec": g4dpo.spec(**g4arm.DPO6)}}
    (tmp_path / "beside.json").write_text(json.dumps(beside))
    names = ["G4-6c-s240", "G4-6c-s480", "G4-6c-s717"]
    return ns(evalsets=str(tmp_path / "evalsets.parquet"), split=str(tmp_path / "split.json"),
              refqe=str(tmp_path / "refqe.parquet"), base=str(tmp_path / "G4-0"),
              predictions=str(tmp_path / "pred.parquet"), train_summary=str(tmp_path / "train.json"),
              candidate=[f"{n}={tmp_path / n}" for n in names], reference=[f"G4-4={tmp_path / 'G4-4'}"],
              beside_summary=[f"G4-6b={tmp_path / 'beside.json'}"], tally=None, out=str(tmp_path / "report.md"))


def test_report6c_chooses_on_the_search_part_and_decides_on_the_decision_part(tmp_path, monkeypatch):
    # s240 holds (a1) on the search part with the larger (b) there; s480 holds (a1) with a smaller (b); s717 has the
    # largest (b) but not (a1). On the decision part s480 would have been better: the choice must not look there.
    qe = {"G4-6c-s240": {"search": 1.9, "decision": 2.1}, "G4-6c-s480": {"search": 1.8, "decision": 1.0},
          "G4-6c-s717": {"search": 3.0, "decision": 3.0}, "G4-0": {"search": 1.5, "decision": 1.5},
          "G4-4": {"search": 1.6, "decision": 1.6}}
    share = {"G4-6c-s240": {"search": 4, "decision": 2}, "G4-6c-s480": {"search": 3, "decision": 4},
             "G4-6c-s717": {"search": 4, "decision": 4}, "G4-0": {"search": 1, "decision": 1},
             "G4-4": {"search": 1, "decision": 1}}
    args = report6c_world(tmp_path, monkeypatch, qe, share)
    g4arm.report6c(args)
    r = json.loads((tmp_path / "report.json").read_text())
    assert r["selected"] == "G4-6c-s240" and r["selection"].startswith("a1 on the search part")
    assert r["parts"]["search"]["models"]["G4-6c-s240"]["b_delta"] == pytest.approx(0.75)
    assert r["parts"]["decision"]["models"]["G4-6c-s480"]["b_delta"] == pytest.approx(0.75)
    # the rule on the decision part: s240's QE 2.1 > the human 2.0 -> (a1) fails; (b) +0.25 holds
    assert not r["a1"]["holds"] and r["b"]["holds"] and r["b"]["delta"] == pytest.approx(0.25)
    assert r["verdict"] == "helpt niet"
    c = r["counts"]["search"]["G4-6c-s717"]["wmt24pp"]
    assert (c["n"], c["not_ended"], c["out_per_ref"], c["over_1_5_ref"]) == (2, 1, 1.75, 1.0)
    assert r["counts"]["search"]["G4-6c-s240"]["flan"]["out_per_ref"] is None
    assert list(r["training"]) == ["0", "240", "480", "717"]
    assert r["training"]["480"]["pref"]["pref_btval"]["chosen_nll_all"] == 0.75
    assert r["beside"]["G4-6b"]["pref"]["pref_books"]["margin"] == 13.6
    md = (tmp_path / "report.md").read_text()
    assert "Gekozen checkpoint: **G4-6c-s240**" in md and "| zoek | G4-6c-s717 | wmt24pp | 2 | 3,000 | 1 |" in md
    # none holds (a1) on the search part: the smallest largest QE excess
    qe["G4-6c-s240"]["search"], qe["G4-6c-s480"]["search"] = 2.3, 2.2
    g4arm.report6c(report6c_world(tmp_path / "b", monkeypatch, qe, share))
    r = json.loads((tmp_path / "b" / "report.json").read_text())
    assert r["selected"] == "G4-6c-s480" and r["selection"].startswith("no candidate holds (a1)")


def test_report6c_refuses_another_split_or_checkpoints(tmp_path, monkeypatch):
    qe = {n: {"search": 1.0, "decision": 1.0} for n in ("G4-6c-s240", "G4-6c-s480", "G4-6c-s717", "G4-0", "G4-4")}
    share = {n: {"search": 2, "decision": 2} for n in qe}
    args = report6c_world(tmp_path, monkeypatch, qe, share)
    monkeypatch.setattr(g4arm, "SPLIT_DIGEST6C", "f953c026")
    with pytest.raises(SystemExit, match="not the split of G4-5.4"):
        g4arm.report6c(args)
    args = report6c_world(tmp_path, monkeypatch, qe, share)
    args.candidate = list(reversed(args.candidate))
    with pytest.raises(SystemExit, match="order of their steps"):
        g4arm.report6c(args)
    args = report6c_world(tmp_path, monkeypatch, qe, share)
    args.candidate = [args.candidate[0].replace("G4-6c-s240", "G4-6b")] + args.candidate[1:]
    with pytest.raises(SystemExit, match="no checkpoint of G4-6c"):
        g4arm.report6c(args)


# ---------------------------------------------------------------------------------------------------- run_g4arm.sh

def test_g46c_train_takes_its_spec_evals_at_the_checkpoints_and_keeps_their_adapters(tmp_path):
    run, shm, home, g = t6.g46b_world(tmp_path)
    out, calls = run("train", ARM="G4-6c", RUN="g4arm_G4-6c", PHASE="prep")
    assert out.returncode == 0 and out.stdout.strip().splitlines()[-1] == "prepared g4arm_G4-6c on w0", out.stdout + out.stderr
    d = shm / "translator/g4arm/data-g4arm_G4-6c"
    launch = (d / "launch.sh").read_text()
    # printf %q writes the comma as \, (bash reads it back as a comma)
    assert "--save-steps\\ 240\\,480\\ " in launch
    assert "--eval-every\\ 240" in launch or "--eval-every 240" in launch
    assert "own4=" in launch and "--base-adapter" in launch and "G4-4=" in launch
    assert json.loads((d / "dpo.json").read_text()) == OPTS6C
    # G4-6b's pairs, read only
    assert any("data/G4-6b/train-4096.parquet" in c for c in calls)
    out, _ = run("train", ARM="G4-6c", RUN="g4arm_G4-6c_s8", STEPS="8", PHASE="prep", TAG="t2")
    assert out.returncode == 0, out.stdout + out.stderr
    launch = (shm / "translator/g4arm/data-g4arm_G4-6c_s8/launch.sh").read_text()
    assert ("--save-steps\\ 4" in launch or "--save-steps 4" in launch) and "--eval-every\\ 4" in launch
    out, _ = run("train", ARM="G4-6c", RUN="g4arm_G4-6b", PHASE="prep", TAG="t3")
    assert out.returncode != 0 and "no run of G4-6c" in out.stdout


def test_g46c_stages_leave_g4_6bs_data_alone_and_take_the_checkpoints(tmp_path):
    run, shm, home, g = t6.g46b_world(tmp_path)
    for stage in ("build6", "check6", "own", "pairtexts6", "pairclass6", "cands6", "report"):
        out, _ = run(stage, ARM="G4-6c")
        assert out.returncode != 0 and ("not with ARM=G4-6c" in out.stdout or "report6c" in out.stdout), (stage, out.stdout)
    out, _ = run("translate", ARM="G4-6c", MODEL="G4-4", SET="dpo")
    assert out.returncode != 0 and "SET=dpo is G4-6's and G4-6b's" in out.stdout
    for stage in ("ends6c", "report6c"):
        out, _ = run(stage, ARM="G4-6b")
        assert out.returncode != 0 and "is G4-6c's" in out.stdout
    out, _ = run("translate", ARM="G4-6c", MODEL="G4-6c-s240", SET="heldout")
    assert out.returncode != 0 and "SET=eval only" in out.stdout
    out, _ = run("translate", ARM="G4-6c", MODEL="G4-6c")
    assert out.returncode != 0 and "smoke run's" in out.stdout
    # texts: the checkpoints, G4-0, G4-4 and G4-6b in one classify run
    out, _ = run("texts", ARM="G4-6c")
    assert out.returncode != 0 and "CANDIDATES" in (out.stdout + out.stderr)
    out, _ = run("texts", ARM="G4-6c", CANDIDATES="G4-6c-s240 G4-6c-s480 G4-6c-s717", TAG="t4")
    assert out.returncode == 0, out.stdout + out.stderr
    body = (home / "job_body_g4a_texts_t4.sh").read_text()
    assert "echo 'G4-6c-s240 G4-6c-s480 G4-6c-s717 G4-0 G4-4 G4-6b' |" in body and "naturalness-G4-6c/" in body
    out, _ = run("report6c", ARM="G4-6c", CANDIDATES="G4-6c-s240 G4-6c-s480 G4-6c-s717", TAG="t5")
    assert out.returncode == 0, out.stdout + out.stderr
    body = (home / "job_body_g4a_report6c_t5.sh").read_text()
    for x in ("--candidate G4-6c-s240=", "--reference G4-4=", "--reference G4-6b=", "data/G4-5-split/split.json",
              "data/G4-5-split/refqe.parquet", "naturalness-G4-6c/predictions.parquet", "runs/g4arm_G4-6c/summary.json",
              "report-G4-6c/"):
        assert x in body, x
    out, _ = run("blind", ARM="G4-6c", SELECTED="G4-6b", TAG="t6")
    assert out.returncode != 0 and "no checkpoint of G4-6c" in out.stdout


def test_g46c_translates_a_checkpoint_and_removes_its_own_adapter_copy_after(tmp_path):
    run, shm, home, g = t6.g46b_world(tmp_path)
    for rel, body in {"G4-6c-s240/adapter.npz": "x", "G4-6c-s240/adapter.json": "{}", "data/eval/classes-w0.txt": "tiny\n",
                      "data/eval/segments-w0.parquet": "x", "G4-6b/adapter.npz": "x", "G4-6b/adapter.json": "{}"}.items():
        (g / rel).parent.mkdir(parents=True, exist_ok=True)
        (g / rel).write_text(body)
    out, calls = run("translate", ARM="G4-6c", MODEL="G4-6c-s240", TAG="t7")
    assert out.returncode == 0 and "started g4a_translate_G4-6c-s240_t7 on w0" in out.stdout, out.stdout + out.stderr
    body = (home / "job_body_g4a_translate_G4-6c-s240_t7.sh").read_text()
    copy = f"{shm}/translator/g4arm/adapter-g4a_translate_G4-6c-s240_t7"
    assert f"--lora {copy}" in body and "/eval/G4-6c-s240/translations/w0/tiny" in body
    assert body.index("translate_class ") < body.index(f"rm -rf -- {copy}") < body.index("TRANSLATED")
    out, _ = run("translate", ARM="G4-6b", MODEL="G4-6b", TAG="t8")
    assert out.returncode == 0 and "rm -rf" not in (home / "job_body_g4a_translate_G4-6b_t8.sh").read_text()
