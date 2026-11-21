"""Arm G4-7's reward classifier C2 (protocol-g4.md §14, G4-7.6): g4c2.py's data (build7: the filters, C1's documents and
texts out, one passage per document, whole texts only, 900/100 per source with both versions on one side, seeded and
reproducible) and its check c2cal7 (the four conditions and what is reported), on synthetic inputs with a stand-in
tokenizer (no model, no TPU)."""

import collections
import json
import math
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "en-nl-translator"))
sys.path.insert(0, str(ROOT / "experiments" / "flan-nl"))

import g4arm  # noqa: E402
import g4c2  # noqa: E402
import select_passages as sp  # noqa: E402

SOURCES = tuple(g4arm.BT_REPOS3)
BLOCKED_EN = "the quick brown fox jumps over the lazy dog near the old river bank today again"


def pid(source, d, k):
    if source == "handelingen":
        return f"handelingen:h-tk-{d}:t{k}:0"
    if source == "fineweb":
        return f"fineweb:<urn:uuid:{d:08d}-aaaa>:d:{k}"
    return f"{source}:{d:08d}-doc:d:{k}"


def bt_rows(n_docs=1150, passages=2):
    rows = []
    for s in SOURCES:
        for d in range(n_docs):
            for k in range(passages):
                r = {"pid": pid(s, d, k), "source": s, "nl": f"{s} document {d} stuk {k} in het Nederlands",
                     "nl_back": f"{s} document {d} stuk {k} terugvertaald", "en": f"{s} document {d} part {k}",
                     "hard_fails": [], "back_ended": True, "back_decode_loop": False, "back_english_left": False}
                if d % 50 == 7:  # every passage of the document fails a flag
                    r["hard_fails"] = ["structure"]
                if d % 50 == 9 and k == 0:
                    r["back_decode_loop"] = True
                if d % 50 == 11:  # too long for C2 in every passage
                    r["nl"] = " ".join(["woord"] * (g4c2.C2_MAX_TOKENS + 1))
                if d % 50 == 13 and k == 1:
                    r["nl_back"] = " ".join(["woord"] * (g4c2.C2_MAX_TOKENS + 1))
                if d == 21:
                    r["en"] = BLOCKED_EN
                rows.append(r)
    return rows


def count(texts):
    return [len(t.split()) for t in texts]


def c1_rows():
    out = []
    for s in SOURCES:
        for d in (3, 4):
            out.append({"pid": pid(s, d, 0), "label": 1, "text": "c1"})
        out.append({"pid": pid(s, 999999, 0), "label": 1, "text": f"{s} document 5 stuk 0 in het Nederlands"})
        out.append({"pid": pid(s, 999999, 0), "label": 0, "text": f"{s} document 6 stuk 0 terugvertaald"})
    return out


def test_doc_key():
    assert g4c2.doc_key("fineweb:<urn:uuid:0000600d-ca05-4aa2-98ab-ab2727c32d6b>:d:3") == \
        "fineweb:<urn:uuid:0000600d-ca05-4aa2-98ab-ab2727c32d6b>"
    assert g4c2.doc_key("handelingen:h-tk-19941995-100-6089-6103:t11:0") == "handelingen:h-tk-19941995-100-6089-6103"
    assert g4c2.doc_key("raad:0001a1a0-4250-5cf8-adf7-1b07dbd67baf:d:19") == "raad:0001a1a0-4250-5cf8-adf7-1b07dbd67baf"
    # a long passage of G4-3's revisions (paragraphs 4..9 of the document)
    assert g4c2.doc_key("rechtspraak:ECLI:NL:RBAMS:2019:1234:d:p4-9") == "rechtspraak:ECLI:NL:RBAMS:2019:1234"
    for bad in ("raad:abc", "raad:abc:x:1", "heldout:btlong:3", "raad:abc:d:p4", "raad:abc:d:p4-"):
        with pytest.raises(SystemExit):
            g4c2.doc_key(bad)


def test_build7_draws_one_whole_passage_per_document_and_splits_per_source():
    rows = bt_rows()
    blocked = sp.ngrams(BLOCKED_EN)
    out, docs, report = g4c2.build7(rows, c1_rows(), blocked, count)
    assert report["counts"] == {"test/0": 500, "test/1": 500, "train/0": 4500, "train/1": 4500}
    assert docs == sorted(set(docs)) and len(docs) == 5000
    assert {g4c2.doc_key(r["pid"]) for r in out} == set(docs)
    by_pid = collections.defaultdict(list)
    for r in out:
        by_pid[r["pid"]].append(r)
    assert all(len(v) == 2 and {x["label"] for x in v} == {0, 1} and len({x["split"] for x in v}) == 1
               for v in by_pid.values())
    per_doc = collections.Counter(g4c2.doc_key(p) for p in by_pid)
    assert max(per_doc.values()) == 1
    for s in SOURCES:
        tests = {r["pid"] for r in out if r["source"] == s and r["split"] == "test"}
        assert len(tests) == 100
    chosen = {g4c2.doc_key(p) for p in by_pid}
    for s in SOURCES:
        assert not chosen & {g4c2.doc_key(pid(s, d, 0)) for d in (3, 4)}         # C1's documents
        assert pid(s, 5, 0) not in by_pid                                      # C1's text (an original)
        assert pid(s, 21, 0) not in by_pid and pid(s, 21, 1) not in by_pid     # a 13-gram of an evaluation set
        for d in range(1150):
            if d % 50 in (7, 11):
                assert g4c2.doc_key(pid(s, d, 0)) not in chosen
            if d % 50 == 9 and g4c2.doc_key(pid(s, d, 0)) in chosen:
                assert pid(s, d, 1) in by_pid
            if d % 50 == 13 and g4c2.doc_key(pid(s, d, 0)) in chosen:
                assert pid(s, d, 0) in by_pid
    assert max(r["tokens"] for r in out) <= g4c2.C2_MAX_TOKENS
    assert all(r["tokens"] == len(r["text"].split()) for r in out)
    f = report["funnel"]["raad"]
    assert f["drawn"] == 1000 and f["doc_of_c1"] >= 0 and f["over_1022_tokens"] > 0 and f["hard_fails"] > 0
    again = g4c2.build7(list(reversed(rows)), c1_rows(), blocked, count)
    assert again[0] == out and again[1] == docs  # the seeds decide, not the input order


def test_build7_stops_when_a_source_has_too_few_documents():
    with pytest.raises(SystemExit, match="1,000 needed|1000 needed"):
        g4c2.build7(bt_rows(n_docs=1010), c1_rows(), set(), count)


def test_build7_never_takes_the_same_text_twice():
    rows = bt_rows()
    for r in rows:
        if r["source"] == "dpc" and r["pid"].endswith(":d:0"):
            r["nl"] = "dezelfde zin overal"
    out, _, report = g4c2.build7(rows, c1_rows(), set(), count)
    texts = [r["text"] for r in out]
    assert len(texts) == len(set(texts))
    assert report["funnel"]["dpc"]["text_drawn"] > 0


def test_spearman_and_logit():
    assert g4c2.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert g4c2.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    # average ranks [1, 2.5, 2.5, 4] and [1, 3, 2, 4]: Pearson of the ranks 4.5 / sqrt(4.5 * 5)
    assert g4c2.spearman([1, 2, 2, 3], [1, 3, 2, 4]) == pytest.approx(4.5 / math.sqrt(4.5 * 5))
    assert g4c2.spearman([1, 1, 1], [1, 2, 3]) is None
    assert g4c2.logit_of(0.5) == 0 and g4c2.logit_of(1.0) == 12 and g4c2.logit_of(0.0) == -12
    assert g4c2.logit_of(0.9) == pytest.approx(math.log(9))
    assert g4c2.logit_of(1 - 1e-9) == 12      # ln((1 - 1e-9) / 1e-9) ~ 20.7, clipped (G4-7.4e)


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)
    return path


class FakeTokenizer:
    def __call__(self, texts, add_special_tokens=True):
        return {"input_ids": [[1] * len(t.split()) for t in texts]}


def cal7_inputs(tmp_path):
    ev = []
    ev += [{"id": f"europarl-{i:04d}", "set": "europarl", "source": "x", "reference": f"origineel {i}", "meta": "{}"}
           for i in range(640)]
    ev += [{"id": f"wmt24pp-{i:04d}", "set": "wmt24pp", "source": "x", "reference": "r",
            "meta": json.dumps({"is_bad_source": i >= 960})} for i in range(998)]
    ev += [{"id": f"flan-{i:04d}", "set": "flan", "source": "x", "reference": "r", "meta": "{}"} for i in range(1000)]
    evalsets = write(tmp_path / "evalsets.parquet", ev)
    b = [r for r in ev if (r["set"] == "wmt24pp" and json.loads(r["meta"]).get("is_bad_source") is False)
         or r["set"] == "flan"]
    write(tmp_path / "g40" / "pairs-w0.parquet", [{"id": r["id"], "set": r["set"], "hypothesis": f"g40 {r['id']}"}
                                                  for r in b])
    write(tmp_path / "g40" / "metricx-w0.parquet", [{"id": r["id"], "qe_source_hypothesis": 2.0} for r in b])
    for m in ("G4-0", "G4-4"):
        for k in range(1, 9):
            s = f"t10s{k}"
            write(tmp_path / "sampling" / m / s / "pairs" / "pairs-w0.parquet",
                  [{"id": r["id"], "set": r["set"], "hypothesis": f"{m} {s} {r['id']} " + "w " * k, "tokens_out": 5 + k}
                   for r in b])
            write(tmp_path / "sampling" / m / s / "scores" / "metricx-w0.parquet",
                  [{"id": r["id"], "qe_source_hypothesis": 2.0 + 0.01 * k} for r in b])
        write(tmp_path / "sampling" / m / "best8" / "pairs" / "pairs-wall.parquet",
              [{"id": r["id"], "set": r["set"], "hypothesis": f"{m} t10s1 {r['id']} w ", "tokens_out": 6} for r in b])
    data = []
    for s in SOURCES:
        for i in range(4):
            for label in (1, 0):
                data.append({"pid": pid(s, i, 0), "id": pid(s, i, 0), "source": s, "rank": i,
                             "split": "test" if i < 2 else "train", "label": label,
                             "text": " ".join(["t"] * (100 if i == 0 else 300)), "tokens": 100 if i == 0 else 300})
    c2data = write(tmp_path / "c2" / "classifier.parquet", data)
    calib = write(tmp_path / "calib-texts.parquet", [{"model": m, "set": "flan", "id": f"flan-{i:04d}",
                                                      "text": f"{m} {i}"} for m in ("G4-0", "TG") for i in range(50)])
    c1 = write(tmp_path / "calib-pred.parquet", [{"model": m, "set": "flan", "id": f"flan-{i:04d}",
                                                  "p_original": (i + 1) / 52} for m in ("G4-0", "TG") for i in range(50)])
    (tmp_path / "results.json").write_text(json.dumps({"test_accuracy": 0.9, "test_accuracy_by_source": {}}))
    return evalsets, c2data, calib, c1


def predict(texts_path, spread=True):
    out = []
    for r in g4arm.read_rows(texts_path):
        m = r["model"]
        if m == "human":
            p = 0.9
        elif m == "G4-0":
            p = 0.8 if int(r["id"][-1]) % 2 else 0.2
        elif m.startswith("G4-0/t10s") or m.startswith("G4-4/t10s"):
            k = int(m[-1])
            logit = (k - 4.5) * 0.4 if spread else 0.3
            p = 1 / (1 + math.exp(-logit))
        elif m.endswith("/best8"):
            p = 0.4
        elif m == "C2-test":
            p = 0.9 if r["id"].endswith("#1") else 0.1
        elif m.startswith("calib:"):
            p = (int(r["id"][-4:]) + 1) / 52
        else:
            raise AssertionError(m)
        out.append({"model": m, "set": r["set"], "id": r["id"], "p_original": p, "truncated": False})
    return out


def test_cal7_passes_and_reports(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(g4c2, "tokenizer_of", lambda checkpoint: FakeTokenizer())
    evalsets, c2data, calib, c1 = cal7_inputs(tmp_path)
    texts = tmp_path / "texts.parquet"
    g4c2.main(["cal7-texts", "--evalsets", str(evalsets), "--g40", str(tmp_path / "g40" / "pairs-w*.parquet"),
               "--sampling", str(tmp_path / "sampling"), "--c2-data", str(c2data), "--checkpoint", "x",
               "--calib", f"G4-6c={calib}", "--out", str(texts)])
    counts = collections.Counter(r["model"] for r in g4arm.read_rows(texts))
    assert counts["human"] == 640 and counts["G4-0"] == 1960 and counts["G4-0/t10s3"] == 1960
    assert counts["G4-4/best8"] == 1960 and counts["C2-test"] == 20 and counts["calib:G4-6c:TG"] == 50
    preds = write(tmp_path / "predictions.parquet", predict(texts))
    args = ["cal7", "--results", str(tmp_path / "results.json"), "--texts", str(texts), "--predictions", str(preds),
            "--c2-data", str(c2data), "--sampling", str(tmp_path / "sampling"),
            "--g40-scores", str(tmp_path / "g40" / "metricx-w*.parquet"), "--c1", f"G4-6c={c1}",
            "--out", str(tmp_path / "cal7")]
    assert g4c2.main(args) == 0
    assert "C2CAL7_OK" in capsys.readouterr().out
    rep = json.loads((tmp_path / "cal7" / "report.json").read_text())
    c = rep["conditions"]
    assert c["2_europarl_original"]["value"] == 1.0 and c["3_g40_greedy_share"]["value"] == pytest.approx(0.5, abs=0.01)
    sd = statistics_pstdev([(k - 4.5) * 0.4 for k in range(1, 9)])
    assert c["4_spread_flan"]["value"] == pytest.approx(sd)
    assert c["4_spread_flan"]["qe_penalty"] == pytest.approx(statistics_pstdev([0.02 * k for k in range(1, 9)]))
    assert rep["test_by_length"]["<=128"]["accuracy"] == 1.0 and rep["test_by_length"][">256"]["n"] == 10
    assert rep["calibration"]["G4-6c"]["spearman_c2_c1"] == pytest.approx(1.0)
    assert rep["high_correlation_with_c1"] == ["G4-6c"]
    md = (tmp_path / "cal7" / "report.md").read_text()
    assert "alle vier gehaald" in md and "hack" in md


def statistics_pstdev(xs):
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def test_cal7_fails_without_spread_within_the_group(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(g4c2, "tokenizer_of", lambda checkpoint: FakeTokenizer())
    evalsets, c2data, calib, c1 = cal7_inputs(tmp_path)
    texts = tmp_path / "texts.parquet"
    g4c2.main(["cal7-texts", "--evalsets", str(evalsets), "--g40", str(tmp_path / "g40" / "pairs-w*.parquet"),
               "--sampling", str(tmp_path / "sampling"), "--c2-data", str(c2data), "--checkpoint", "x",
               "--out", str(texts)])
    preds = write(tmp_path / "predictions.parquet", predict(texts, spread=False))
    rc = g4c2.main(["cal7", "--results", str(tmp_path / "results.json"), "--texts", str(texts), "--predictions",
                    str(preds), "--c2-data", str(c2data), "--sampling", str(tmp_path / "sampling"),
                    "--g40-scores", str(tmp_path / "g40" / "metricx-w*.parquet"), "--out", str(tmp_path / "cal7")])
    out = capsys.readouterr().out
    assert rc == 1 and "C2CAL7_FAIL" in out and "4_spread_wmt24pp" in out and "4_spread_flan" in out
    assert "1_test_accuracy" not in out.split("C2CAL7_FAIL")[1].split("]")[0]
