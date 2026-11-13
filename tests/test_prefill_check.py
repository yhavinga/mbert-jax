"""prefill_check.py (the prefill check of 13 Nov): the tests' models and shapes against the length trial's classes,
the order of decode_all, the plan's split without slot reuse, translate.py's arguments, the verdict, and collect and
report on made-up data. No network, model or TPU."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "experiments" / "en-nl-translator", ROOT / "experiments" / "flan-nl"):
    sys.path.insert(0, str(_p))

import bt_length as bl  # noqa: E402
import prefill_check as pc  # noqa: E402
import translate as tr  # noqa: E402


def test_every_test_has_one_class_of_its_shapes_on_its_base_model():
    models = pc.add_models(dict(tr.MODELS))
    for test, t in pc.TESTS.items():
        m = models[pc.model_name(test)]
        base = tr.MODELS[t["base"]]
        assert list(m["length_classes"]) == [pc.class_name(test)] and m["slots"] == {pc.class_name(test): t["slots"]}
        cls = m["length_classes"][pc.class_name(test)]
        assert cls == {"max_source": t["max_source"], "prefill_lengths": [t["prefill"]], "max_decode": t["decode"]}
        assert cls["prefill_lengths"][-1] > cls["max_source"]
        for key in ("maxtext_model", "hf_repo", "eos_ids", "prompt", "turn", "chips", "checkpoints"):
            assert m[key] == base[key], (test, key)


def test_the_tests_change_only_what_they_are_about():
    doc = {"B": tr.MODELS["gemma3-27b-doc"], "A": tr.MODELS["gemma4-31b-doc"]}
    t5, t6 = pc.TESTS["T5"], pc.TESTS["T6"]
    d1k_b, d1k_a = doc["B"]["length_classes"]["d1k"], doc["A"]["length_classes"]["d1k"]
    # T5 is part B's d1k on the other MaxText; T6 part A's d1k with fewer slots than documents a worker
    assert (t5["prefill"], t5["decode"], t5["slots"]) == (d1k_b["prefill_lengths"][-1], d1k_b["max_decode"],
                                                           doc["B"]["slots"]["d1k"])
    assert (t6["prefill"], t6["decode"]) == (d1k_a["prefill_lengths"][-1], d1k_a["max_decode"]) and t6["reuse"]
    assert t6["slots"] < 10 and t6["shard"] >= 10
    # the bin-512 tests keep part B's d512 decode limit; only the prefill moves across the window
    d512 = doc["B"]["length_classes"]["d512"]
    for test, prefill in (("T4a", pc.W), ("T4b", pc.W + 64), ("T1", d1k_b["prefill_lengths"][-1])):
        assert pc.TESTS[test]["decode"] == d512["max_decode"] and pc.TESTS[test]["prefill"] == prefill
    # T2 and T2b: prefill exactly W, decode above and at W
    assert (pc.TESTS["T2"]["prefill"], pc.TESTS["T2"]["decode"]) == (pc.W, d1k_b["max_decode"])
    assert (pc.TESTS["T2b"]["prefill"], pc.TESTS["T2b"]["decode"]) == (pc.W, pc.W)
    assert pc.TESTS["T3"]["prefill"] == doc["A"]["length_classes"]["d2k"]["prefill_lengths"][-1]
    assert pc.TESTS["T3"]["per_process"] and pc.TESTS["T3"]["shard"] <= pc.TESTS["T3"]["slots"]
    assert set(pc.TG_025) | set(pc.FORK) == set(pc.TESTS) and not set(pc.TG_025) & set(pc.FORK)
    assert {pc.TESTS[t]["maxtext"] for t in pc.TG_025} == {"0.2.5"} and {pc.TESTS[t]["maxtext"] for t in pc.FORK} == {"fork"}


def test_add_models_leaves_the_existing_models_as_they_are():
    models = dict(tr.MODELS)
    before = json.dumps(models, sort_keys=True)
    pc.add_models(models)
    assert json.dumps({k: v for k, v in models.items() if not k.startswith("prefill-check-")}, sort_keys=True) == before


class _Device:
    """decode_all's device: records the order of the prefills; every completion ends after `steps` tokens"""

    def __init__(self, steps):
        self.order, self.steps, self.slot_of, self.count = [], steps, {}, {}

    def prefill(self, tokens):
        self.order.append(tokens[0])
        return tokens[0], 5

    def insert(self, prefix, slot):
        self.slot_of[slot], self.count[slot] = prefix, 0

    def generate(self):
        out = []
        for s in range(max(self.slot_of) + 1 if self.slot_of else 0):
            self.count[s] = self.count.get(s, 0) + 1
            out.append(1 if self.count[s] >= self.steps.get(self.slot_of.get(s), 3) else 5)
        return out

    @staticmethod
    def fetch(arrays):
        return arrays


def test_insertion_ranks_follow_decode_all():
    lengths = [7, 9, 7, 3, 9, 5]
    prompts = [[k] + [0] * (n - 1) for k, n in enumerate(lengths)]
    device = _Device({})
    tr.decode_all(prompts, 2, device, max_decode=10, sync_every=2, eos_ids=[1])
    rows = [{"id": f"d{k}", "tokens_in": n} for k, n in enumerate(lengths)]
    ranks = pc.insertion_ranks(rows)
    assert [f"d{k}" for k in device.order] == sorted(ranks, key=ranks.get)
    assert ranks == {"d1": 0, "d4": 1, "d0": 2, "d2": 3, "d5": 4, "d3": 5}


def test_similarity_on_words():
    s = pc.similarity([("een twee drie", "een twee drie"), ("een twee drie vier", "een twee drie"),
                       ("x x x x x x", "een twee drie"), (None, "een")])
    assert s["n"] == 4 and s["identical"] == 1 and s["below_half"] == 2
    assert s["median"] == pytest.approx(6 / 7)
    assert pc.similarity([]) == {"n": 0, "identical": 0, "median": None, "below_half": 0}


def test_wrap_schedule_replays_decode_all():
    # one slot, A = 10, sync every 8: the first request (longest prompt) ends after 3 generate calls, seen at the
    # sync after 8; the second starts at call 8 and writes 5 tokens (8 + 5 > 10: it wraps)
    rows = [{"id": "b", "tokens_in": 3, "tokens_out": 6}, {"id": "a", "tokens_in": 5, "tokens_out": 4}]
    sched = pc.wrap_schedule(rows, slots=1, max_decode=10)
    assert sched["a"] == {"start": 0, "index": 0, "writes": 3, "crosses": False}
    assert sched["b"] == {"start": 8, "index": 8, "writes": 5, "crosses": True}
    assert not pc.wrap_schedule(rows, slots=1, max_decode=16)["b"]["crosses"]
    # two slots: both start at 0
    assert {k: v["start"] for k, v in pc.wrap_schedule(rows, slots=2, max_decode=10).items()} == {"a": 0, "b": 0}
    # a request that ran to the limit: its writes are max_decode - 1, no wrap from index 0
    full = pc.wrap_schedule([{"id": "c", "tokens_in": 2, "tokens_out": 10}], slots=1, max_decode=10)["c"]
    assert full["writes"] == 9 and not full["crosses"]


def test_verdict_thresholds():
    assert pc.verdict(9, 10) == "goed" and pc.verdict(36, 40) == "goed"
    assert pc.verdict(35, 40) == "gemengd" and pc.verdict(20, 40) == "gemengd"
    assert pc.verdict(19, 40) == "fout" and pc.verdict(0, 3) == "fout"
    assert pc.verdict(0, 0) == "niet gemeten"


def test_translate_argv_of_both_parts():
    b = pc.translate_argv("T1", "s.parquet", "gs://x/out", "tok", "gs://c", "/w")
    assert b[b.index("--model") + 1] == "prefill-check-T1" and b[b.index("--length-class") + 1] == "t1"
    assert "--prompt" not in b and "--max-shards" not in b and b[b.index("--shard-size") + 1] == "64"
    assert (b[b.index("--source-lang") + 1], b[b.index("--target-lang") + 1]) == ("en", "nl")
    a = pc.translate_argv("T3", "s.parquet", "gs://x/out", "tok", "gs://c", "/w", max_shards=1)
    assert a[a.index("--prompt") + 1] == "natural-v1" and a[a.index("--max-shards") + 1] == "1"
    assert (a[a.index("--source-lang") + 1], a[a.index("--target-lang") + 1]) == ("nl", "en")
    # translate.py takes them once the models are added
    pc.add_models(tr.MODELS)
    assert pc.model_name("T3") in tr.MODELS


def _docs(n, bin_, tokens, source="dpc", prefix="d"):
    return [{"id": f"{prefix}{k:02d}", "source": source, "doc_id": str(k), "bin": bin_, "tokens": tokens + k,
             "paragraphs": [f"Paragraph one of document {k} with some words in it.",
                            f"Paragraph two of document {k}, a little longer than the first one is."]}
            for k in range(n)]


def _plan(tmp_path, test, docs, workers=4):
    bl.write_parquet(docs, tmp_path / "docs.parquet")
    bl.write_parquet([{"id": d["id"], "text": bl.join(d["paragraphs"]), "tokens": d["tokens"]} for d in docs],
                     tmp_path / "seg" / "segments-w0.parquet")
    pc.main(["plan", "--test", test, "--docs", str(tmp_path / "docs.parquet"), "--segments",
             str(tmp_path / "seg" / "segments-w0.parquet"), "--workers", str(workers), "--out", str(tmp_path / "plan")])
    return json.loads((tmp_path / "plan" / test / "plan.json").read_text())


def test_plan_splits_one_bin_and_keeps_slots_unshared(tmp_path):
    docs = _docs(40, 1024, 800) + _docs(5, 512, 400, prefix="s")
    p = _plan(tmp_path, "T3", docs)
    assert p["bin_docs"] == 40 and len(p["kept"]) == 40 and p["workers"] == [10, 10, 10, 10] and not p["dropped"]
    rows = [r for k in range(4) for r in bl.read_rows(tmp_path / "plan" / "T3" / f"segments-w{k}.parquet")]
    assert sorted(r["id"] for r in rows) == sorted(p["kept"])
    # T3 runs one process per shard of 5 (6 slots): 2 processes for 10 documents
    pc.main(["processes", "--test", "T3", "--segments", str(tmp_path / "plan" / "T3" / "segments-w0.parquet")])
    # T6 keeps 10 a worker in 4 slots on purpose; T1 refuses more documents a worker than slots
    assert _plan(tmp_path, "T6", docs)["workers"] == [10, 10, 10, 10]
    with pytest.raises(SystemExit):
        _plan(tmp_path, "T1", _docs(70, 512, 400))


def test_plan_drops_a_source_over_the_class_and_wants_a_tokenizer_when_the_prompt_may_not_fit(tmp_path):
    p = _plan(tmp_path, "T5", _docs(8, 1024, 600) + [dict(_docs(1, 1024, 1100, prefix="x")[0])])
    assert len(p["kept"]) == 8 and [d["id"] for d in p["dropped"]] == ["x00"]
    with pytest.raises(SystemExit):  # 900 source tokens and a template may not fit a prefill of 1024
        _plan(tmp_path, "T2", _docs(8, 1024, 900))


def _shard(rows, path):
    bl.write_parquet(rows, path)


def test_collect_and_report_on_made_up_translations(tmp_path, capsys):
    docs = _docs(12, 1024, 800)
    p = _plan(tmp_path, "T6", docs, workers=2)
    assert p["workers"] == [6, 6]
    good = {d["id"]: "\n\n".join(f"Paragraaf {i} van document {d['doc_id']}, met wat woorden erin."
                                 if i == 0 else f"Paragraaf twee van document {d['doc_id']}, iets langer dan de eerste."
                                 for i in range(2)) for d in docs}
    loop = "de de de de de de de de de de de de de de de de de de de de de de de de de de de de de de de de de"
    for k in range(2):
        segs = bl.read_rows(tmp_path / "plan" / "T6" / f"segments-w{k}.parquet")
        rows = []
        for s in segs:
            rows.append({"id": s["id"], "text_en": good[s["id"]], "ended": True, "tokens_in": s["tokens"] + 90,
                         "tokens_out": 40})
        ranks = pc.insertion_ranks(rows)
        for r in rows:  # the documents after the first 4 of a worker loop and do not end
            if ranks[r["id"]] >= 4:
                r.update(text_en=loop, ended=False, tokens_out=1280)
        _shard(rows, tmp_path / "out" / "T6" / f"w{k}" / "t6" / "shard-00000.parquet")
    for name in ("docs_a", "docs_b"):
        bl.write_parquet(docs, tmp_path / f"{name}.parquet")
    pc.main(["collect", "--plan", str(tmp_path / "plan"), "--translations", str(tmp_path / "out"), "--docs-a",
             str(tmp_path / "docs_a.parquet"), "--docs-b", str(tmp_path / "docs_b.parquet"), "--tests", "T6",
             "--out", str(tmp_path / "collect")])
    measured = bl.read_rows(tmp_path / "collect" / "measured.parquet")
    assert len(measured) == 12 and sum(m["first_inserted"] for m in measured) == 8
    assert all(m["done"] for m in measured if m["first_inserted"])
    assert not any(m["done"] for m in measured if not m["first_inserted"])
    units = bl.read_rows(tmp_path / "collect" / "units.parquet")
    bl.write_parquet([{"id": u["id"], "qe_src_cand": 2.0} for u in units], tmp_path / "qe.parquet")
    # the trial's reference: every document done whole and in pieces
    for part in ("a", "b"):
        ref = tmp_path / "ref" / part
        rows, unit_rows = [], []
        for d in docs:
            for cond in ("heel", "stukken"):
                row, us, _ = bl.measure_doc(d, good[d["id"]], True, "nl", "en")
                rows.append({"id": d["id"], "source": d["source"], "bin": d["bin"], "condition": cond,
                             "tokens_source": d["tokens"], "tokens_in": d["tokens"] + 90, "tokens_out": 40,
                             "translation": good[d["id"]], **row})
                unit_rows += [{"id": f"{d['id']}|{cond}|u{u['unit']:04d}", "doc": d["id"], "condition": cond,
                               "n_source": u["n_source"]} for u in us]
        bl.write_parquet(rows, ref / "measured.parquet")
        bl.write_parquet(unit_rows, ref / "units.parquet")
        bl.write_parquet([{"id": u["id"], "qe_src_cand": 1.5} for u in unit_rows], ref / "qe-x-w0.parquet")
        bl.write_parquet([{"id": d["id"], "tokens": d["tokens"]} for d in docs], ref / "seg" / "segments-w0.parquet")
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "pc_T6_x-w0.log").write_text(
        'MAXTEXT {"version": "0.2.5", "url": "https://github.com/yhavinga/maxtext.git", "commit": "2249474b9"}\n'
        't6: 6 segments in 1 shards, 0 done before; shapes {"max_source": 1024, "prefill_lengths": [1152], '
        '"max_decode": 1280, "slots": 4}\nengine ready in 160 s\n'
        'SHARD {"segments": 6, "tokens_in": 5000, "tokens_out": 3000, "seconds": 100.0}\n'
        'MEM {"peak_gib_max": 25.1}\n')
    capsys.readouterr()
    pc.main(["report", "--plan", str(tmp_path / "plan"), "--measured", str(tmp_path / "collect" / "measured.parquet"),
             "--units", str(tmp_path / "collect" / "units.parquet"), "--qe", str(tmp_path / "qe.parquet"),
             "--ref-a", str(tmp_path / "ref" / "a"), "--ref-b", str(tmp_path / "ref" / "b"), "--logs",
             str(tmp_path / "logs"), "--out", str(tmp_path / "report" / "prefill-check.md")])
    r = json.loads((tmp_path / "report" / "prefill-check.json").read_text())
    e = r["tests"]["T6"]
    assert e["test"]["docs"] == 12 and e["test"]["done"] == 8 and e["ref_heel"]["done"] == 12
    assert e["group_first"]["verdict"] == "goed" and e["group_later"]["verdict"] == "fout"
    assert e["group_later"]["n"] == 4 and e["only_ref_done"] == 4 and e["same_prompts"] == 12
    assert e["ref_heel"]["qe_mean"] == pytest.approx(1.5) and e["test"]["qe_mean"] == pytest.approx(2.0)
    assert e["similarity"] == {"n": 12, "identical": 8, "median": 1.0, "below_half": 4}
    # the 4 loops started after the first 40-token round and ran to 1280 tokens: past the end of the cache
    assert e["group_later_cross"]["n"] == 4 and e["group_later_inside"]["n"] == 0 and e["group_first_cross"]["n"] == 0
    assert e["group_later_cross"]["similarity"]["below_half"] == 4
    md = (tmp_path / "report" / "prefill-check.md").read_text()
    assert "| T6 |" in md and "eerst ingevoegd" in md and "2249474b9" in md and "25,10" in md


def test_reuse_cpu_decodes_like_a_forward_pass_until_the_shared_cache_index_wraps(tmp_path):
    """The fork's decode with slot reuse (MaxText upstream main d1c0d0afe / fork 2249474, CPU, tiny Gemma 4): R2 in a
    fresh decode state matches a train-mode forward pass at every step; R2 inserted after R1 decoded REUSE_LEAD steps
    matches until the shared cache index wraps and not after (characterizes H-fork; fails once MaxText fixes it)."""
    pytest.importorskip("maxtext")
    fresh, reused = pc.reuse_cpu(type("Args", (), {"work_dir": str(tmp_path)})())
    assert fresh["first_step_over_tol"] is None
    wrap = pc.REUSE_DECODE - pc.REUSE_LEAD
    assert reused["first_step_over_tol"] == wrap + 1 and max(reused["rel"][:wrap + 1]) <= 1e-3
