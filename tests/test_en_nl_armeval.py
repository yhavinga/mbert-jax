"""experiments/en-nl-translator/armeval.py: the extra sets in evalsets.py's format, the classifier's data, the texts,
the blind package, MTLD, the rules and the report, on small tables (no model, no TPU; chrF is replaced where sacrebleu
is not installed)."""

import hashlib
import json
import math
import random
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "en-nl-translator"))

import armeval as ae  # noqa: E402


@pytest.fixture(autouse=True)
def plain_chrf(monkeypatch):
    # stand-ins for sacrebleu's chrF: 100 for the same text, 0 otherwise
    monkeypatch.setattr(ae, "chrf", lambda a, b: 100.0 if a == b else 0.0)
    monkeypatch.setattr(ae, "corpus_chrf", lambda hs, rs: 100.0 * sum(h == r for h, r in zip(hs, rs)) / len(hs))


def tokenizer_dir(tmp_path):
    from tokenizers import Tokenizer, models, pre_tokenizers

    words = sorted({f"w{i}" for i in range(200)} | {"the", "house", "agrees", "de", "kamer"})
    tok = Tokenizer(models.WordLevel({w: i for i, w in enumerate(["[UNK]"] + words)}, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    d = tmp_path / "tok"
    d.mkdir()
    tok.save(str(d / "tokenizer.json"))
    return d


def write(path, rows, schema=None):
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)
    return path


def evalsets_rows():
    rows = []
    for k in range(4):
        rows.append({"id": f"wmt24pp-{k:04d}", "set": "wmt24pp", "source": f"the house w{k}", "reference": f"de kamer {k}",
                     "tokens": 3, "over_512": False, "length_class": "tiny",
                     "meta": json.dumps({"is_bad_source": k == 3})})
    for name in ("flores", "ntrex", "europarl"):
        for k in range(2):
            rows.append({"id": f"{name}-{k:04d}", "set": name, "source": f"w{k} w{k}", "reference": f"r{k}", "tokens": 2,
                         "over_512": False, "length_class": "tiny", "meta": "{}"})
    return rows


@pytest.fixture
def small(monkeypatch, tmp_path):
    """Inputs of `data` with 3 FLAN rows, 3 blind passages (one of WMT24++), 2 in-domain passages, 4 WMT24++ segments
    of which 3 without is_bad_source."""
    monkeypatch.setattr(ae, "N_FLAN", 3)
    monkeypatch.setattr(ae, "N_BLIND", 3)
    monkeypatch.setattr(ae, "N_INDOMAIN", 2)
    monkeypatch.setattr(ae, "N_WMT", 4)
    monkeypatch.setattr(ae, "N_WMT_GOOD", 3)
    long_text = " ".join(f"w{i}" for i in range(40))  # 40 tokens: class small, the others tiny
    flan = [{"set": "classifier", "host": 0, "position": k, "source": f"task{k}", "instruction_tokens": 3,
             "inputs": f" w{k} the house " if k < 2 else long_text, "targets": "t", "segments": [k]} for k in range(3)]
    flan.append({"set": "yeb", "host": 1, "position": 9, "source": "taskY", "instruction_tokens": 3, "inputs": "w9",
                 "targets": "t", "segments": [9]})
    blind = [{"id": "flan:1-9", "set": "flan", "source": "w9", "meta": {"task": "taskY"}, "blind_id": "blind-001"},
             {"id": "wmt24pp-0001", "set": "wmt24pp", "source": "the house w1", "meta": {}, "blind_id": "blind-002"},
             {"id": "gv:1:2", "set": "idiom", "source": long_text, "meta": {"idiom": "w3"}, "blind_id": "blind-003"}]
    blind_path = tmp_path / "blind100.jsonl"
    ae.write_jsonl(blind_path, blind)
    monkeypatch.setattr(ae, "BLIND_SHA256", ae.sha256_file(blind_path))
    meta = [{"id": "handelingen:a:0", "kind": "bt", "source": "handelingen", "length_class": "short", "en": "the house agrees"},
            {"id": "raad:b:0", "kind": "bt_context", "source": "raad", "length_class": "short", "en": "w5 w6"}]
    messages = [{"messages": [{"role": "user", "content": "Translate: the house agrees"},
                              {"role": "assistant", "content": "De Kamer stemt in."}]},
                {"messages": [{"role": "user", "content": "<earlier_turns> w5 w6"},
                              {"role": "assistant", "content": "Raad."}]}]
    return {"flan": write(tmp_path / "eval-rows.parquet", flan), "blind": blind_path,
            "evalsets": write(tmp_path / "evalsets.parquet", evalsets_rows()),
            "meta": write(tmp_path / "eval-meta.parquet", meta), "eval": write(tmp_path / "eval.parquet", messages),
            "tok": tokenizer_dir(tmp_path)}


def run_data(small, out, kind, workers=2):
    argv = ["data", "--kind", kind, "--flan-eval", str(small["flan"]), "--evalsets", str(small["evalsets"]),
            "--tokenizer", str(small["tok"]), "--workers", str(workers), "--out", str(out)]
    if kind == "x":
        argv += ["--blind", str(small["blind"]), "--eval-meta", str(small["meta"]), "--eval", str(small["eval"])]
    ae.main(argv)
    return {r["id"]: r for r in pq.read_table(out / "evalsets.parquet").to_pylist()}


def test_data_x_has_the_three_sets_in_evalsets_format(small, tmp_path):
    rows = run_data(small, tmp_path / "x", "x")
    sets = {}
    for r in rows.values():
        sets.setdefault(r["set"], []).append(r)
    assert {k: len(v) for k, v in sets.items()} == {"flan": 3, "blind": 3, "indomain": 2}
    assert rows["flan:0-0"]["source"] == "w0 the house" and rows["flan:0-0"]["reference"] is None
    # the blind WMT24++ passage keeps its reference, the others have none
    assert rows["blind-002"]["reference"] == "de kamer 1"
    assert rows["blind-001"]["reference"] is None and rows["blind-003"]["reference"] is None
    assert json.loads(rows["blind-003"]["meta"])["blind_set"] == "idiom"
    # in-domain: the synthetic English in, the Dutch original as reference, without context
    assert rows["indomain:handelingen:a:0"]["source"] == "the house agrees"
    assert rows["indomain:handelingen:a:0"]["reference"] == "De Kamer stemt in."
    assert json.loads(rows["indomain:raad:b:0"]["meta"])["source"] == "raad"
    assert rows["blind-003"]["length_class"] == "small" and rows["blind-003"]["tokens"] == 40
    # every row once over the workers' segments, whole classes per worker
    seen = []
    for k in range(2):
        seg = pq.read_table(tmp_path / "x" / f"segments-w{k}.parquet").to_pylist()
        classes = (tmp_path / "x" / f"classes-w{k}.txt").read_text().split()
        assert {rows[s["id"]]["length_class"] for s in seg} == set(classes)
        seen += [s["id"] for s in seg]
    assert sorted(seen) == sorted(rows)
    manifest = json.loads((tmp_path / "x" / "manifest.json").read_text())
    assert manifest["sets"]["blind"]["with_reference"] == 1


def test_data_instr_is_wmt24pp_and_flan(small, tmp_path):
    rows = run_data(small, tmp_path / "i", "instr")
    assert sorted({r["set"] for r in rows.values()}) == ["flan", "wmt24pp"]
    assert sum(r["set"] == "wmt24pp" for r in rows.values()) == 4
    assert json.loads(rows["wmt24pp-0003"]["meta"])["is_bad_source"] is True


def test_data_refuses_a_changed_blind_list(small, tmp_path, monkeypatch):
    monkeypatch.setattr(ae, "BLIND_SHA256", "0" * 64)
    with pytest.raises(SystemExit, match="sha256"):
        run_data(small, tmp_path / "x", "x")


def test_data_refuses_a_worker_without_a_class(small, tmp_path):
    with pytest.raises(SystemExit, match="no length class"):
        run_data(small, tmp_path / "x", "x", workers=4)


def test_classifier_data_orders_by_sha256_and_splits_per_passage(tmp_path, monkeypatch):
    monkeypatch.setattr(ae, "N_CLASSIFIER", 4)
    monkeypatch.setattr(ae, "N_CLASSIFIER_TEST", 1)
    meta = [{"id": f"p{k}:0", "kind": "bt", "source": "raad"} for k in range(6)]
    meta.append({"id": "c:0", "kind": "bt_context", "source": "handelingen"})
    pairs = [{"id": f"p{k}:0", "pid": f"p{k}", "kind": "passage", "split": "eval", "nl": f" nl{k} ", "back": f"back{k}"}
             for k in range(6)]
    pairs += [{"id": "c:0", "pid": "c", "kind": "passage", "split": "eval", "nl": "nlc", "back": "backc"},
              {"id": "t:0", "pid": "t", "kind": "passage", "split": "train", "nl": "x", "back": "y"}]
    out = tmp_path / "c.parquet"
    ae.main(["classifier-data", "--eval-meta", str(write(tmp_path / "m.parquet", meta)),
             "--pairs", str(write(tmp_path / "p.parquet", pairs)), "--out", str(out)])
    rows = pq.read_table(out).to_pylist()
    expected = sorted((f"p{k}" for k in range(6)),
                      key=lambda p: hashlib.sha256(f"en-nl-sft/classifier/{p}".encode()).hexdigest())[:4]
    assert [r["pid"] for r in rows if r["label"] == 1] == expected
    assert [r["split"] for r in rows if r["label"] == 1] == ["train", "train", "train", "test"]
    # both versions of a passage on the same side, the original stripped
    by = {(r["pid"], r["label"]): r for r in rows}
    for pid in expected:
        assert by[(pid, 1)]["split"] == by[(pid, 0)]["split"]
        assert by[(pid, 1)]["text"] == f"nl{pid[1:]}" and by[(pid, 0)]["text"] == f"back{pid[1:]}"
    assert "c" not in {r["pid"] for r in rows}


def test_texts_takes_the_960_and_the_flan_rows(small, tmp_path):
    hyp = tmp_path / "hyp.parquet"
    write(hyp, [{"id": f"wmt24pp-{k:04d}", "set": "wmt24pp", "hypothesis": f"h{k}"} for k in range(4)]
          + [{"id": f"flan:0-{k}", "set": "flan", "hypothesis": f"f{k}"} for k in range(3)])
    out = tmp_path / "texts.parquet"
    ae.main(["texts", "--evalsets", str(small["evalsets"]), "--wmt", f"base={hyp}", "--flan", f"base={hyp}",
             "--out", str(out)])
    rows = pq.read_table(out).to_pylist()
    assert sorted(r["id"] for r in rows if r["set"] == "wmt24pp") == ["wmt24pp-0000", "wmt24pp-0001", "wmt24pp-0002"]
    assert len([r for r in rows if r["set"] == "flan"]) == 3


def test_mtld():
    assert ae.mtld("a a a a a a") == 2.0
    assert ae.mtld("a b a b") == 4.0
    assert math.isnan(ae.mtld("a b c d"))  # no factor ends, nothing falls: undefined


def test_ab_order_is_fixed_by_the_seed():
    ids = [f"blind-{k:03d}" for k in range(1, 101)]
    first = [ae.base_is_a("B", i) for i in ids]
    assert first == [random.Random(f"en-nl-sft/blind/ab/B/{i}").random() < 0.5 for i in ids]
    assert 30 < sum(first) < 70
    assert first != [ae.base_is_a("N", i) for i in ids]


def test_blind_package(small, tmp_path):
    blind = ae.read_jsonl(small["blind"])
    base = write(tmp_path / "base.parquet", [{"id": b["blind_id"], "set": "blind", "source": b["source"].strip(),
                                              "hypothesis": f"basis {b['blind_id']}"} for b in blind])
    arm = write(tmp_path / "arm.parquet", [{"id": b["blind_id"], "set": "blind", "source": b["source"].strip(),
                                             "hypothesis": f"arm {b['blind_id']}"} for b in blind])
    ae.main(["blind", "--blind", str(small["blind"]), "--base", str(base), "--arm", f"B={arm}", "--out", str(tmp_path / "o")])
    judge = ae.read_jsonl(tmp_path / "o" / "B" / "judge-input.jsonl")
    key = ae.read_jsonl(tmp_path / "o" / "B" / "key.jsonl")
    assert [set(j) for j in judge] == [{"blind_id", "source", "A", "B"}] * 3
    for j, k in zip(judge, key):
        assert j["blind_id"] == k["blind_id"]
        assert j["A"] == (f"basis {j['blind_id']}" if k["A"] == "base" else f"arm {j['blind_id']}")
        assert (k["A"] == "base") is ae.base_is_a("B", j["blind_id"])
        assert {k["A"], k["B"]} == {"base", "B"}


def test_rule_a_holds_on_the_bound():
    base = {s: [{"metricx_qe": 1.0}, {"metricx_qe": 2.0}] for s in ae.RULE_A_SETS}
    arm = {s: [{"metricx_qe": 1.2}, {"metricx_qe": 2.2}] for s in ae.RULE_A_SETS}
    assert ae.rule_a(base, arm)["holds"]
    arm["ntrex"] = [{"metricx_qe": 1.3}, {"metricx_qe": 2.2}]
    r = ae.rule_a(base, arm)
    assert not r["holds"] and not r["sets"]["ntrex"]["holds"] and r["sets"]["flores"]["holds"]


def test_classifier_shares_pool_both_sets():
    preds = ([{"model": "base", "set": "wmt24pp", "p_original": p} for p in (0.1, 0.6)]
             + [{"model": "base", "set": "flan", "p_original": p} for p in (0.2, 0.3, 0.5, 0.9)])
    s = ae.classifier_shares(preds)["base"]
    assert s["wmt24pp"] == 0.5 and s["flan"] == 0.5 and s["pooled"] == pytest.approx(3 / 6) and s["n"] == 6


def score_dir(d, rows, comet=True):
    """pairs/ and scores/ as run_arms.sh score writes them, on one worker."""
    (d / "pairs").mkdir(parents=True)
    (d / "scores").mkdir()
    write(d / "pairs" / "pairs-w0.parquet", [{k: r[k] for k in ("id", "set", "source", "hypothesis", "reference", "ended",
                                                                   "decode_loop", "english_left", "empty")} for r in rows],
          schema=pa.schema([("id", pa.string()), ("set", pa.string()), ("source", pa.string()),
                            ("hypothesis", pa.string()), ("reference", pa.string()), ("ended", pa.bool_()),
                            ("decode_loop", pa.bool_()), ("english_left", pa.bool_()), ("empty", pa.bool_())]))
    write(d / "scores" / "metricx-w0.parquet", [{"id": r["id"], "ref_source_hypothesis_reference": r["ref"],
                                                 "qe_source_hypothesis": r["qe"]} for r in rows],
          schema=pa.schema([("id", pa.string()), ("ref_source_hypothesis_reference", pa.float64()),
                            ("qe_source_hypothesis", pa.float64())]))
    if comet:
        write(d / "scores" / "comet-w0.parquet", [{"id": r["id"], "comet22": 0.8} for r in rows if r["reference"]])


def test_report_end_to_end(small, tmp_path, monkeypatch):
    monkeypatch.setattr(ae, "N_INDOMAIN", 2)
    ev = evalsets_rows()

    def scores(qe):
        return [{"id": r["id"], "set": r["set"], "metricx_ref": 1.0, "metricx_qe": qe, "comet22": 0.85, "ended": True,
                 "decode_loop": False, "english_left": False, "empty": False, "over_512": False} for r in ev]

    base4 = write(tmp_path / "base4.parquet", scores(1.0))
    b4 = write(tmp_path / "b4.parquet", scores(1.1))
    n4 = write(tmp_path / "n4.parquet", scores(1.25))
    x_rows = [{"id": "indomain:handelingen:a:0", "set": "indomain", "source": "s", "reference": "De Kamer.",
               "meta": json.dumps({"source": "handelingen"})},
              {"id": "indomain:raad:b:0", "set": "indomain", "source": "s", "reference": "Raad.",
               "meta": json.dumps({"source": "raad"})},
              {"id": "flan:0-0", "set": "flan", "source": "s", "reference": None, "meta": "{}"},
              {"id": "blind-001", "set": "blind", "source": "s", "reference": None, "meta": "{}"}]
    x_data = write(tmp_path / "x-evalsets.parquet", x_rows,
                   schema=pa.schema([("id", pa.string()), ("set", pa.string()), ("source", pa.string()),
                                     ("reference", pa.string()), ("meta", pa.string())]))

    def x_dir(name, hyp):
        rows = [{"id": r["id"], "set": r["set"], "source": r["source"], "hypothesis": hyp if r["reference"] else "f",
                 "reference": r["reference"], "ended": True, "decode_loop": False, "english_left": False,
                 "empty": False, "ref": 2.0 if r["reference"] else None, "qe": 1.5} for r in x_rows]
        score_dir(tmp_path / name, rows)
        return tmp_path / name

    instr_rows = [{"id": r["id"], "set": r["set"], "source": r["source"], "hypothesis": "h", "reference": r["reference"],
                   "ended": True, "decode_loop": False, "english_left": False, "empty": False, "ref": 1.0, "qe": 1.0}
                  for r in ev if r["set"] == "wmt24pp"]
    instr_rows.append({"id": "flan:0-0", "set": "flan", "source": "s", "hypothesis": "h", "reference": None,
                       "ended": True, "decode_loop": False, "english_left": False, "empty": False, "ref": None, "qe": 2.0})
    score_dir(tmp_path / "ninstr", instr_rows)
    (tmp_path / "results.json").write_text(json.dumps({"test_accuracy": 0.8, "test_examples": 400}))
    preds = []
    for model, share in (("base", 0.25), ("B", 0.5), ("N", 0.25), ("Ninstr", 0.25)):
        for s in ("wmt24pp", "flan"):
            preds += [{"model": model, "set": s, "id": f"{s}{k}", "p_original": 0.9 if k < share * 4 else 0.1}
                      for k in range(4)]
    write(tmp_path / "pred.parquet", preds)
    write(tmp_path / "pll.parquet", [{"model": m, "set": s, "id": "i", "tokens": 4, "pll": -8.0}
                                     for m in ("base", "B") for s in ("wmt24pp", "flan")]
          + [{"model": "N", "set": "flan", "id": "j", "tokens": 2000, "pll": None}])
    write(tmp_path / "texts.parquet", [{"model": "base", "set": "flan", "id": "i", "text": "a a a a a a"}])
    out = tmp_path / "report.md"
    ae.main(["report", "--evalsets", str(small["evalsets"]), "--baseline-scores", str(base4), "--four", f"B={b4}",
             f"N={n4}", "--x-data", str(x_data), "--x", f"base={x_dir('bx', 'De Kamer.')}", f"B={x_dir('b', 'x')}",
             f"N={x_dir('n', 'x')}", "--instr", f"Ninstr={tmp_path / 'ninstr'}", "--classifier",
             str(tmp_path / "results.json"), "--predictions", str(tmp_path / "pred.parquet"), "--pll",
             str(tmp_path / "pll.parquet"), "--texts", str(tmp_path / "texts.parquet"), "--commit", "abc",
             "--out", str(out), "--json", str(tmp_path / "report.json")])
    r = json.loads((tmp_path / "report.json").read_text())
    assert r["rule_a"]["B"]["holds"] and not r["rule_a"]["N"]["holds"]
    assert r["rule_a"]["B"]["sets"]["wmt24pp"]["n"] == 3  # the WMT24++ segments without is_bad_source
    assert r["rule_b"]["B"]["holds"] and not r["rule_b"]["N"]["holds"]
    assert r["indomain"]["base"]["handelingen"]["chrf"] == 100.0 and r["indomain"]["B"]["all"]["chrf"] == 0.0
    assert r["indomain"]["base"]["all"]["n"] == 2 and set(r["indomain"]["base"]) == {"handelingen", "raad", "all"}
    assert r["mode_check"]["Ninstr"]["wmt24pp"]["n"] == 3
    assert r["pll"]["N"]["flan"] == {"n": 0, "skipped": 1, "pll": None, "pll_per_token": None}
    assert r["pll"]["base"]["wmt24pp"]["pll_per_token"] == -2.0
    assert r["mtld"]["base"]["flan"] == 2.0
    md = out.read_text()
    assert "regel (a) gehaald" in md and "regel (a) NIET gehaald" in md and "(c) open" in md


def test_report_without_a_usable_classifier(small, tmp_path, monkeypatch):
    # below 0.75 the shares are not used: rule (b) cannot be measured
    monkeypatch.setattr(ae, "N_INDOMAIN", 0)
    ev = evalsets_rows()
    rows = [{"id": r["id"], "set": r["set"], "metricx_ref": 1.0, "metricx_qe": 1.0, "comet22": 0.85, "ended": True,
             "decode_loop": False, "english_left": False, "empty": False, "over_512": False} for r in ev]
    s4 = write(tmp_path / "s4.parquet", rows)
    x_data = write(tmp_path / "x.parquet", [{"id": "flan:0-0", "set": "flan", "meta": "{}"}])
    score_dir(tmp_path / "bx", [{"id": "flan:0-0", "set": "flan", "source": "s", "hypothesis": "h", "reference": None,
                                 "ended": True, "decode_loop": False, "english_left": False, "empty": False, "ref": None,
                                 "qe": 1.0}], comet=False)
    (tmp_path / "results.json").write_text(json.dumps({"test_accuracy": 0.7, "test_examples": 400}))
    ae.main(["report", "--evalsets", str(small["evalsets"]), "--baseline-scores", str(s4), "--four", f"B={s4}",
             "--x-data", str(x_data), "--x", f"base={tmp_path / 'bx'}", "--classifier", str(tmp_path / "results.json"),
             "--out", str(tmp_path / "r.md"), "--json", str(tmp_path / "r.json")])
    r = json.loads((tmp_path / "r.json").read_text())
    assert r["rule_b"] is None
    assert "NIET gehaald: regel (b) is niet te meten" in (tmp_path / "r.md").read_text()


def test_report_of_other_arms_without_x(small, tmp_path):
    # the amendment of 11 Nov (protocol §9): arms N50 and N25, rule (a) on the four sets; no extra sets, no classifier
    ev = evalsets_rows()

    def scores(qe):
        return [{"id": r["id"], "set": r["set"], "metricx_ref": 1.0, "metricx_qe": qe, "comet22": 0.85, "ended": True,
                 "decode_loop": False, "english_left": r["set"] == "flores", "empty": False, "over_512": False}
                for r in ev]

    base4 = write(tmp_path / "base4.parquet", scores(1.0))
    n50 = write(tmp_path / "n50.parquet", scores(1.3))
    n25 = write(tmp_path / "n25.parquet", scores(1.2))
    out = tmp_path / "r.md"
    ae.main(["report", "--title", "Armen N50 en N25", "--evalsets", str(small["evalsets"]), "--baseline-scores",
             str(base4), "--four", f"N50={n50}", f"N25={n25}", "--out", str(out), "--json", str(tmp_path / "r.json")])
    r = json.loads((tmp_path / "r.json").read_text())
    assert list(r["rule_a"]) == ["N50", "N25"]
    assert not r["rule_a"]["N50"]["holds"] and r["rule_a"]["N25"]["holds"]
    assert r["four"]["N25"]["flores"]["english_left"] == r["four"]["N25"]["flores"]["n"]
    assert r["rule_b"] is None and r["indomain"] == {}
    md = out.read_text()
    assert md.startswith("# Armen N50 en N25\n")
    assert "**N50**: (a) nee, (b) niet te meten" in md and "**N25**: (a) ja" in md


def test_report_x_needs_x_data(small, tmp_path):
    with pytest.raises(SystemExit):
        ae.main(["report", "--evalsets", str(small["evalsets"]), "--baseline-scores", "b", "--four", "N50=a",
                 "--x", f"base={tmp_path}", "--out", "o", "--json", "j"])
