"""experiments/en-nl-translator/evalsets.py (the evaluation sets: reading each source, length classes, the workers'
share, collecting the translations, the summary) and metricx_qe.py's reference mode, without a model or the network.
Run with --noconftest where the conftest's imports are not installed."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "en-nl-translator"))

import evalsets  # noqa: E402
import metricx_qe as mx  # noqa: E402

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

# ---------------------------------------------------------------- MetricX-24 with a reference


def test_metricx_input_with_and_without_reference():
    # metricx24/predict.py (fc4978e), _make_input: the QE form, and with is_qe false the reference after the candidate
    assert mx.make_input("A house.", "Een huis.") == "source: A house. candidate: Een huis."
    assert mx.make_input("A house.", "Een huis.", "Een woning.") == \
        "source: A house. candidate: Een huis. reference: Een woning."


def test_metricx_pairs_name_their_columns():
    assert mx.parse_pair("en:nl") == ("en", "nl", None, "qe_en_nl")
    assert mx.parse_pair("source:hypothesis:reference") == \
        ("source", "hypothesis", "reference", "ref_source_hypothesis_reference")
    with pytest.raises(SystemExit):
        mx.parse_pair("a:b:c:d")


def test_metricx_rows_to_score_keeps_an_empty_candidate_only_when_asked():
    table = {"s": ["x", "x", "", "x", "x"], "c": ["y", "", "y", None, "y"], "r": ["z", "z", "z", "z", ""]}
    assert mx.rows_to_score(table, "s", "c", None, False) == [0, 4]
    assert mx.rows_to_score(table, "s", "c", None, True) == [0, 1, 4]
    assert mx.rows_to_score(table, "s", "c", "r", True) == [0, 1]


def test_metricx_score_writes_a_column_per_pair(tmp_path, monkeypatch):
    seen = []

    class FakeScorer:
        def __init__(self, *a, **k):
            pass

        def score(self, sources, candidates, references=None):
            seen.append(references)
            return [float(len(c)) + (0.5 if references else 0.0) for c in candidates], [10] * len(sources)

    monkeypatch.setattr(mx, "Scorer", FakeScorer)
    pairs = tmp_path / "pairs.parquet"
    pq.write_table(pa.table({"id": ["a", "b"], "source": ["s1", "s2"], "hypothesis": ["hyp", ""],
                             "reference": ["r1", "r2"]}), pairs)
    out = tmp_path / "mx.parquet"
    mx.main(["score", "--pairs", str(pairs), "--pair", "source:hypothesis:reference", "--pair", "source:hypothesis",
             "--out", str(out), "--empty-candidates"])
    got = pq.read_table(out).to_pydict()
    assert got == {"id": ["a", "b"], "ref_source_hypothesis_reference": [3.5, 0.5], "qe_source_hypothesis": [3.0, 0.0]}
    assert seen == [["r1", "r2"], None]


# ---------------------------------------------------------------- reading the sources


def test_lines_strip_crlf_and_split_only_on_newlines(tmp_path):
    path = tmp_path / "x.txt"
    path.write_bytes("one\r\ntwo still two\x85\r\nthree\r\n".encode())
    assert evalsets.lines(path, 3) == ["one", "two still two\x85", "three"]
    with pytest.raises(SystemExit):
        evalsets.lines(path, 4)


def test_wmt24pp_takes_the_post_edit_as_reference(tmp_path):
    path = tmp_path / "en-nl_NL.jsonl"
    rec = {"lp": "en-nl_NL", "domain": "news", "document_id": "d", "segment_id": 3, "is_bad_source": False,
           "source": "A house.", "target": "Een woning.", "original_target": "Een huis."}
    path.write_text(json.dumps(rec) + "\n")
    [row] = evalsets.read_wmt24pp(path)
    assert row["source"] == "A house." and row["reference"] == "Een woning."
    assert row["meta"]["original_target"] == "Een huis." and row["meta"]["is_bad_source"] is False


def europarl_file(path, rows):
    header = "iid\tsrc\tnative_speaker\toriginal\tdest\ttext\tdirect\tlabel"
    path.write_text("\n".join([header] + ["\t".join(r) for r in rows]) + "\n", encoding="utf-8")


def test_europarl_takes_dutch_originals_with_their_english_translation(tmp_path):
    f = tmp_path / "multi8_dev.tsv"
    europarl_file(f, [["1:a", "nl", "1", "Het huis.", "en", "The house.", "1", "1"],
                      ["1:a", "nl", "1", "Het huis.", "nl", "Het huis.", "1", "0"],
                      ["2:b", "en", "1", "The car \"red\".", "nl", "De auto.", "1", "1"],
                      ["3:c", "nl", "0", "Een \"rode\" auto.", "en", "A red car.", "1", "1"]])
    rows = evalsets.read_europarl([("dev", f)])
    assert [(r["source"], r["reference"]) for r in rows] == [("The house.", "Het huis."),
                                                              ("A red car.", "Een \"rode\" auto.")]
    assert rows[1]["meta"] == {"iid": "3:c", "split": "dev", "native_speaker": "0"}


def test_europarl_refuses_a_row_with_a_tab_too_many_and_a_twice_seen_iid(tmp_path):
    f = tmp_path / "multi8_dev.tsv"
    europarl_file(f, [["1:a", "nl", "1", "Het\thuis.", "en", "The house.", "1", "1"]])
    with pytest.raises(SystemExit, match="fields"):
        evalsets.read_europarl([("dev", f)])
    europarl_file(f, [["1:a", "nl", "1", "Het huis.", "en", "The house.", "1", "1"]])
    with pytest.raises(SystemExit, match="twice"):
        evalsets.read_europarl([("dev", f), ("test", f)])


def test_checked_refuses_a_file_whose_sha256_is_not_pinned(tmp_path):
    path = tmp_path / "multi8_dev.tsv"
    path.write_text("not the file")
    with pytest.raises(SystemExit, match="sha256"):
        evalsets.checked(path)


# ---------------------------------------------------------------- length classes and workers


def test_length_classes_follow_translate_py():
    assert [evalsets.length_class(n) for n in (1, 32, 33, 128, 129, 384, 385, 512, 513, 768)] == \
        ["tiny", "tiny", "small", "small", "medium", "long", "xlong", "xlong", "xxlong", "xxlong"]
    with pytest.raises(SystemExit):
        evalsets.length_class(769)


def rows_of(spec):
    """[(class, tokens, count)] -> rows"""
    out = []
    for cls, tokens, count in spec:
        out += [{"id": f"{cls}-{k}", "tokens": tokens + k % 3, "length_class": cls} for k in range(count)]
    return out


def test_workers_get_whole_classes_largest_first():
    rows = rows_of([("tiny", 20, 2700), ("small", 60, 1700), ("medium", 170, 200), ("long", 300, 11),
                    ("xlong", 470, 1)])
    got = evalsets.assign_workers(rows, 4)
    classes = {k: sorted({r["length_class"] for r in v}) for k, v in got.items()}
    assert classes == {0: ["small"], 1: ["tiny"], 2: ["medium"], 3: ["long", "xlong"]}
    assert sum(len(v) for v in got.values()) == len(rows)
    for part in got.values():
        assert [(r["tokens"], r["id"]) for r in part] == sorted((r["tokens"], r["id"]) for r in part)


def test_more_workers_than_classes_leaves_some_empty():
    got = evalsets.assign_workers(rows_of([("tiny", 20, 5)]), 4)
    assert [len(v) for v in got.values()] == [5, 0, 0, 0]


# ---------------------------------------------------------------- collect and summary


def write_data(root, rows, assignment):
    root.mkdir(parents=True, exist_ok=True)
    cols = ["id", "set", "source", "reference", "tokens", "over_512", "length_class", "meta"]
    pq.write_table(pa.table({c: [r[c] for r in rows] for c in cols}), root / "evalsets.parquet")
    for k, ids in assignment.items():
        pq.write_table(pa.table({"id": pa.array(ids, pa.string()), "text": pa.array(["x"] * len(ids), pa.string()),
                                 "tokens": pa.array([1] * len(ids), pa.int64())}), root / f"segments-w{k}.parquet")
    (root / "manifest.json").write_text(json.dumps({"sets": {n: {**s, "segments": 0, "source_tokens": 0,
                                                                  "max_tokens": 0, "over_512": 0}
                                                              for n, s in evalsets.SETS.items()},
                                                    "files": {}}))


def eval_rows():
    rows = []
    for name in evalsets.SETS:
        for k in range(2):
            meta = {"is_bad_source": k == 1} if name == "wmt24pp" else {}
            rows.append({"id": f"{name}-{k:04d}", "set": name, "source": f"The house number {k} is red.",
                         "reference": f"Huis {k} is rood.", "tokens": 10, "over_512": False, "length_class": "tiny",
                         "meta": json.dumps(meta)})
    return rows


def write_shard(path, ids, texts, ended=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"id": ids, "text_nl": texts, "ended": ended or [True] * len(ids),
                             "tokens_in": [80] * len(ids), "tokens_out": [9] * len(ids)}), path)


def test_collect_joins_translations_and_flags_loops(tmp_path):
    rows = eval_rows()
    data = tmp_path / "data"
    write_data(data, rows, {0: [r["id"] for r in rows[:4]], 1: [r["id"] for r in rows[4:]]})
    loop = "Het huis is rood, rood, rood, rood, rood, rood, rood, rood, rood, rood, rood, rood, rood, rood."
    write_shard(tmp_path / "w0" / "tiny" / "shard-00000.parquet", [r["id"] for r in rows[:4]],
                ["Huis 0 is rood.", loop, "", "Huis 1 is rood."], [True, False, True, True])
    out = tmp_path / "pairs-w0.parquet"
    evalsets.main(["collect", "--data", str(data), "--worker", "0", "--shards", str(tmp_path / "w0"), "--out", str(out)])
    got = pq.read_table(out).to_pylist()
    assert [r["id"] for r in got] == [r["id"] for r in rows[:4]]
    assert [r["ended"] for r in got] == [True, False, True, True]
    assert [r["decode_loop"] for r in got] == [False, True, False, False]
    assert [r["empty"] for r in got] == [False, False, True, False]
    assert got[0]["reference"] == "Huis 0 is rood." and got[0]["source"] == "The house number 0 is red."


def test_collect_refuses_a_missing_translation(tmp_path):
    rows = eval_rows()
    data = tmp_path / "data"
    write_data(data, rows, {0: [r["id"] for r in rows]})
    write_shard(tmp_path / "w0" / "tiny" / "shard-00000.parquet", [r["id"] for r in rows[:-1]], ["x"] * (len(rows) - 1))
    with pytest.raises(SystemExit, match="without a translation"):
        evalsets.main(["collect", "--data", str(data), "--worker", "0", "--shards", str(tmp_path / "w0"),
                       "--out", str(tmp_path / "p.parquet")])


def pair_rows(rows):
    return [{"id": r["id"], "set": r["set"], "source": r["source"], "hypothesis": "Huis.", "reference": r["reference"],
             "ended": r["id"] != "flores-0001", "tokens_in": 80, "tokens_out": 3, "decode_loop": r["id"] == "ntrex-0000",
             "english_left": False, "empty": False, "tokens": 10, "over_512": False, "length_class": "tiny"}
            for r in rows]


def test_summary_per_set_with_the_wmt_subset_without_bad_sources(tmp_path):
    rows = eval_rows()
    data = tmp_path / "data"
    write_data(data, rows, {0: [r["id"] for r in rows]})
    pairs = pair_rows(rows)
    pq.write_table(pa.table({c: [r[c] for r in pairs] for c in pairs[0]}), tmp_path / "pairs-w0.parquet")
    ids = [r["id"] for r in rows]
    pq.write_table(pa.table({"id": ids, "ref_source_hypothesis_reference": [float(k) for k in range(len(ids))],
                             "qe_source_hypothesis": [1.0] * len(ids)}), tmp_path / "metricx-w0.parquet")
    pq.write_table(pa.table({"id": ids, "comet22": [0.8] * len(ids)}), tmp_path / "comet-w0.parquet")
    (tmp_path / "comet-timing-w0.json").write_text(json.dumps({"rows": 8, "threads": 4, "batch_size": 32, "first": 8,
                                                               "first_seconds": 1.0, "projected_seconds": 1}))
    out = tmp_path / "out"
    evalsets.main(["summary", "--data", str(data), "--pairs", str(tmp_path / "pairs-w0.parquet"),
                   "--metricx", str(tmp_path / "metricx-w0.parquet"), "--comet", str(tmp_path / "comet-w0.parquet"),
                   "--comet-timing", str(tmp_path / "comet-timing-w0.json"), "--model", "TG 27B",
                   "--setup", json.dumps({"decoding": "greedy"}), "--out", str(out)])
    s = json.loads((out / "summary.json").read_text())
    assert s["sets"]["wmt24pp"]["all"]["metricx_ref"] == {"mean": 0.5, "median": 0.5}
    assert s["sets"]["wmt24pp"]["without_bad_source"]["segments"] == 1
    assert "without_bad_source" not in s["sets"]["flores"]
    assert s["sets"]["flores"]["all"]["not_ended"] == 1 and s["sets"]["ntrex"]["all"]["decode_loop"] == 1
    assert s["sets"]["ntrex"]["all"]["not_ended_or_loop"] == 1
    assert s["comet_timing"]["w0"]["first"] == 8 and s["setup"] == {"decoding": "greedy"}
    md = (out / "summary.md").read_text()
    assert "| wmt24pp | without_bad_source | 1 |" in md and "TG 27B" in md
    scores = pq.read_table(out / "scores.parquet").to_pydict()
    assert scores["id"] == ids and scores["comet22"] == [0.8] * len(ids)
    assert pq.read_table(out / "translations.parquet").column("hypothesis").to_pylist() == ["Huis."] * len(ids)


def test_summary_refuses_a_missing_score(tmp_path):
    rows = eval_rows()
    data = tmp_path / "data"
    write_data(data, rows, {0: [r["id"] for r in rows]})
    pairs = pair_rows(rows)
    pq.write_table(pa.table({c: [r[c] for r in pairs] for c in pairs[0]}), tmp_path / "p.parquet")
    ids = [r["id"] for r in rows]
    pq.write_table(pa.table({"id": ids, "ref_source_hypothesis_reference": [1.0] * (len(ids) - 1) + [None],
                             "qe_source_hypothesis": [1.0] * len(ids)}), tmp_path / "m.parquet")
    pq.write_table(pa.table({"id": ids, "comet22": [0.8] * len(ids)}), tmp_path / "c.parquet")
    with pytest.raises(SystemExit, match="no metricx_ref"):
        evalsets.main(["summary", "--data", str(data), "--pairs", str(tmp_path / "p.parquet"), "--metricx",
                       str(tmp_path / "m.parquet"), "--comet", str(tmp_path / "c.parquet"), "--model", "x",
                       "--out", str(tmp_path / "o")])
    pq.write_table(pa.table({"id": ids[:-1], "comet22": [0.8] * (len(ids) - 1)}), tmp_path / "c.parquet")
    with pytest.raises(SystemExit, match="COMET scores have"):
        evalsets.main(["summary", "--data", str(data), "--pairs", str(tmp_path / "p.parquet"), "--metricx",
                       str(tmp_path / "m.parquet"), "--comet", str(tmp_path / "c.parquet"), "--model", "x",
                       "--out", str(tmp_path / "o")])
