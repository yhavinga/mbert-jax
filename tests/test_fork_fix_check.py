"""fork_fix_check.py (the fork fix check of 14 Nov): its tests against the prefill check's and the trial's shapes, the
refusal of another MaxText, and plan, collect and report with the run without the fix on made-up data. No network,
model or TPU."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "experiments" / "en-nl-translator", ROOT / "experiments" / "flan-nl"):
    sys.path.insert(0, str(_p))

import bt_length as bl  # noqa: E402
import fork_fix_check as ffc  # noqa: E402
import prefill_check as pc  # noqa: E402
import translate as tr  # noqa: E402


def test_the_tests_are_the_runs_that_looped_with_only_the_build_and_slots_changed():
    a = tr.MODELS["gemma4-31b-doc"]["length_classes"]
    t6, t5 = pc.TESTS["T6"], pc.TESTS["T5"]
    v1, v2, v3 = ffc.TESTS["V1"], ffc.TESTS["V2"], ffc.TESTS["V3"]
    shape = ("part", "bin", "base", "prefill", "decode", "max_source")
    assert all(v1[k] == t6[k] for k in shape + ("slots", "shard", "reuse"))
    assert all(v3[k] == t5[k] for k in shape) and v3["slots"] == 4 < 10 <= v3["shard"] and v3["reuse"]
    assert (v2["prefill"], v2["decode"], v2["max_source"]) == (a["d2k"]["prefill_lengths"][-1], a["d2k"]["max_decode"],
                                                               a["d2k"]["max_source"])
    assert v2["slots"] == pc.TRIAL_SLOTS["A"]["d2k"] and v2["known"] == "d2k" and v2["reuse"]
    # every test decodes past the window, and its run without the fix is a test of the prefill check or the trial
    assert all(t["decode"] > pc.W for t in ffc.TESTS.values())
    assert {b for b in ffc.BEFORE.values() if b != "heel"} <= set(pc.TESTS) and set(ffc.BEFORE) == set(ffc.TESTS)


def test_main_leaves_the_prefill_checks_tests_as_they_were(tmp_path):
    tests, report_md = pc.TESTS, pc.report_md
    with pytest.raises(SystemExit):
        ffc.main(["plan", "--test", "T6", "--docs", "x", "--segments", "y", "--out", str(tmp_path)])
    assert pc.TESTS is tests and pc.report_md is report_md and "T6" in pc.TESTS


def test_translate_refuses_another_maxtext(monkeypatch, tmp_path):
    monkeypatch.setattr(tr, "maxtext_source", lambda: {"version": "0.2.5", "url": "x", "commit": "2249474b9"})
    with pytest.raises(SystemExit, match="not the fork"):
        ffc.main(["translate", "--test", "V1", "--segments", "s", "--out", "gs://b/o", "--tokenizer", "t", "--ckpt",
                  "c", "--work-dir", str(tmp_path)])


def _docs(n, bin_, tokens, prefix="d"):
    return [{"id": f"{prefix}{k:02d}", "source": "dpc", "doc_id": str(k), "bin": bin_, "tokens": tokens + k,
             "paragraphs": [f"Paragraph one of document {k} with some words in it.",
                            f"Paragraph two of document {k}, a little longer than the first one is."]}
            for k in range(n)]


def _measured(docs, text, cond, not_ended=()):
    rows, units = [], []
    for d in docs:
        row, us, _ = bl.measure_doc(d, text[d["id"]], d["id"] not in not_ended, "nl", "en")
        rows.append({"id": d["id"], "source": d["source"], "bin": d["bin"], "condition": cond, "test": cond,
                     "tokens_source": d["tokens"], "tokens_in": d["tokens"] + 90, "tokens_out": 40,
                     "translation": text[d["id"]], **row})
        units += [{"id": f"{d['id']}|{cond}|u{u['unit']:04d}", "doc": d["id"], "condition": cond,
                   "n_source": u["n_source"]} for u in us]
    return rows, units


def test_plan_collect_and_report_with_the_run_without_the_fix(tmp_path):
    docs = _docs(12, 1024, 800)
    bl.write_parquet(docs, tmp_path / "docs.parquet")
    bl.write_parquet([{"id": d["id"], "text": bl.join(d["paragraphs"]), "tokens": d["tokens"]} for d in docs],
                     tmp_path / "seg" / "segments-w0.parquet")
    ffc.main(["plan", "--test", "V1", "--docs", str(tmp_path / "docs.parquet"), "--segments",
              str(tmp_path / "seg" / "segments-w0.parquet"), "--workers", "2", "--out", str(tmp_path / "plan")])
    plan = json.loads((tmp_path / "plan" / "V1" / "plan.json").read_text())
    assert plan["test"] == "V1" and plan["workers"] == [6, 6] and plan["slots"] == 4
    good = {d["id"]: f"Paragraaf een van document {d['doc_id']}, met wat woorden erin.\n\nParagraaf twee van "
                     f"document {d['doc_id']}, iets langer dan de eerste." for d in docs}
    loop = " ".join(["de"] * 40)
    for k in range(2):  # with the fix every document is done
        rows = [{"id": s["id"], "text_en": good[s["id"]], "ended": True, "tokens_in": s["tokens"] + 90,
                 "tokens_out": 40} for s in bl.read_rows(tmp_path / "plan" / "V1" / f"segments-w{k}.parquet")]
        bl.write_parquet(rows, tmp_path / "out" / "V1" / f"w{k}" / "v1" / "shard-00000.parquet")
    for name in ("docs_a", "docs_b"):
        bl.write_parquet(docs, tmp_path / f"{name}.parquet")
    ffc.main(["collect", "--plan", str(tmp_path / "plan"), "--translations", str(tmp_path / "out"), "--docs-a",
              str(tmp_path / "docs_a.parquet"), "--docs-b", str(tmp_path / "docs_b.parquet"), "--tests", "V1",
              "--out", str(tmp_path / "collect")])
    units = bl.read_rows(tmp_path / "collect" / "units.parquet")
    bl.write_parquet([{"id": u["id"], "qe_src_cand": 2.0} for u in units], tmp_path / "qe.parquet")
    for part in ("a", "b"):  # the trial: every document done whole and in pieces
        ref = tmp_path / "ref" / part
        rows, unit_rows = [], []
        for cond in ("heel", "stukken"):
            rs, us = _measured(docs, good, cond)
            rows += rs
            unit_rows += us
        bl.write_parquet(rows, ref / "measured.parquet")
        bl.write_parquet(unit_rows, ref / "units.parquet")
        bl.write_parquet([{"id": u["id"], "qe_src_cand": 1.5} for u in unit_rows], ref / "qe-x-w0.parquet")
        bl.write_parquet([{"id": d["id"], "tokens": d["tokens"]} for d in docs], ref / "seg" / "segments-w0.parquet")
    # without the fix (T6): the documents after the first 4 of a worker started late, looped and ran to 1280 tokens
    collected = {m["id"]: m for m in bl.read_rows(tmp_path / "collect" / "measured.parquet")}
    late = {i for i, m in collected.items() if not m["first_inserted"]}
    text = {i: loop if i in late else good[i] for i in good}
    extra = {i: {"worker": m["worker"], "shard": m["shard"], "rank": m["rank"],
                 "first_inserted": m["first_inserted"], "tokens_out": 1280 if i in late else 40}
             for i, m in collected.items()}
    rows, unit_rows = _measured(docs, text, "T6", not_ended=late)
    rows = [dict(r, **extra[r["id"]]) for r in rows]
    before = tmp_path / "before"
    bl.write_parquet(rows, before / "measured.parquet")
    bl.write_parquet(unit_rows, before / "units.parquet")
    bl.write_parquet([{"id": u["id"], "qe_src_cand": 9.0 if u["doc"] in late else 2.0} for u in unit_rows],
                     before / "qe-pc-w0.parquet")
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "pc_V1_x-w0.log").write_text(
        f'MAXTEXT {{"version": "0.2.5", "url": "https://github.com/yhavinga/maxtext.git", "commit": "{ffc.FIX_COMMIT}"}}\n'
        'v1: 6 segments in 1 shards, 0 done before; shapes {"max_source": 1024, "prefill_lengths": [1152], '
        '"max_decode": 1280, "slots": 4}\nengine ready in 150 s\n'
        'SHARD {"segments": 6, "tokens_in": 5000, "tokens_out": 3000, "seconds": 100.0}\nMEM {"peak_gib_max": 19.1}\n')
    ffc.main(["report", "--plan", str(tmp_path / "plan"), "--measured", str(tmp_path / "collect" / "measured.parquet"),
              "--units", str(tmp_path / "collect" / "units.parquet"), "--qe", str(tmp_path / "qe.parquet"),
              "--ref-a", str(tmp_path / "ref" / "a"), "--ref-b", str(tmp_path / "ref" / "b"), "--before", str(before),
              "--logs", str(tmp_path / "logs"), "--out", str(tmp_path / "report" / "fork-fix-check.md")])
    r = json.loads((tmp_path / "report" / "fork-fix-check.json").read_text())
    e = r["tests"]["V1"]
    assert e["test"]["done"] == 12 and e["verdict"] == "goed" and e["test"]["qe_mean"] == pytest.approx(2.0)
    b = e["before"]
    assert b["source"] == "T6" and b["slots"] == 4 and b["stats"]["done"] == 8 and b["stats"]["docs"] == 12
    assert b["only_now_done"] == 4 and b["only_before_done"] == 0 and b["both_done"] == 8
    assert b["similarity"]["below_half"] == 4 and e["similarity_stukken"]["median"] == 1.0
    # the late documents wrapped without the fix (their 1280 tokens ran past the end of the cache) and not with it
    assert b["groups"]["later_cross"] == {"n": 4, "done": 0, "loop": 4, "not_ended": 4,
                                          "qe_mean": pytest.approx(9.0)}
    assert e["group_later_inside"]["n"] == 4 and e["group_later_inside"]["done"] == 4
    md = (tmp_path / "report" / "fork-fix-check.md").read_text()
    assert "| V1 |" in md and "T6: 8/12" in md and ffc.FIX_COMMIT[:9] in md and "19,10" in md and "| V4" not in md
