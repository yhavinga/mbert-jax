"""experiments/en-nl-translator/fw2year.py on made-up rows: crawl years, the Wilson interval, the fixed rule, the
supplement's needs, row groups and draw, the classifier's passages left out, the report and the Gemma 4 sizes' texts
and rule (b). No download, no model. Run with --noconftest where the conftest's imports (jax, flax) are not
installed."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "en-nl-translator"))

import fw2year as fy  # noqa: E402


def test_crawl_year_takes_the_year_of_the_dump_and_refuses_anything_else():
    assert fy.crawl_year("CC-MAIN-2019-35") == 2019
    for bad in ("CC-MAIN-2019", "cc-main-2019-35", "", None, "CC-MAIN-2019-35 "):
        with pytest.raises(ValueError):
            fy.crawl_year(bad)


def test_wilson_matches_known_values():
    low, high = fy.wilson(50, 100)
    assert low == pytest.approx(0.4038, abs=1e-4) and high == pytest.approx(0.5962, abs=1e-4)
    low, high = fy.wilson(0, 10)
    assert low == 0.0 and high == pytest.approx(0.2775, abs=1e-4)
    low, high = fy.wilson(10, 10)
    assert high == pytest.approx(1.0) and low == pytest.approx(0.7225, abs=1e-4)
    with pytest.raises(ValueError):
        fy.wilson(0, 0)


def s(k, n):
    return fy.stat([True] * k + [False] * (n - k))


def test_a_decline_needs_both_more_than_five_points_and_separate_intervals():
    base = s(9000, 10000)  # 90.0%, Wilson ~[89.4, 90.6]
    assert fy.decline(s(800, 1000), base) == (True, True, True)       # 80%: 10 pp lower, apart
    assert fy.decline(s(860, 1000), base) == (False, False, True)     # 86%: 4 pp lower only, intervals apart
    assert fy.decline(s(16, 20), base) == (False, True, False)        # 80% of 20: lower, intervals overlap
    assert fy.decline(s(950, 1000), base) == (False, False, False)


def test_the_first_decline_is_the_earliest_year_it_holds_for():
    base = s(9000, 10000)
    years = {2024: s(700, 1000), 2022: s(900, 1000), 2023: s(800, 1000)}
    assert fy.first_decline(years, base) == 2023
    assert fy.first_decline({2022: s(900, 1000)}, base) is None


def test_supplement_needs_fill_thin_years_up_to_the_minimum():
    assert fy.supplement_needs({2013: 73, 2016: 379, 2017: 3152, 2018: 1000}, 1000) == {2013: 927, 2016: 621}
    assert fy.supplement_needs({2020: 5000}, 1000) == {}


def test_row_groups_touching_a_year_by_their_crawl_range():
    groups = [("a", 0, "CC-MAIN-2013-20", "CC-MAIN-2013-48"), ("a", 1, "CC-MAIN-2013-48", "CC-MAIN-2014-15"),
              ("a", 2, "CC-MAIN-2014-15", "CC-MAIN-2016-07"), ("a", 3, "CC-MAIN-2016-07", "CC-MAIN-2017-04")]
    assert [g[1] for g in fy.groups_touching(groups, 2013)] == [0, 1]
    assert [g[1] for g in fy.groups_touching(groups, 2015)] == [2]
    assert [g[1] for g in fy.groups_touching(groups, 2016)] == [2, 3]
    assert fy.groups_touching(groups, 2018) == []


def test_the_classifiers_passages_are_left_out_by_pid_or_text():
    bt = [{"id": "fineweb:x:d:1", "nl": "Een tekst."}, {"id": "fineweb:y:d:2", "nl": " Tweede tekst. "},
          {"id": "fineweb:z:d:3", "nl": "Derde."}]
    clf = [{"pid": "fineweb:x:d:1", "label": 1, "text": "anders"}, {"pid": "fineweb:x:d:1", "label": 0, "text": "terug"},
           {"pid": "raad:q:d:0", "label": 1, "text": "Tweede tekst."}, {"pid": "raad:q:d:0", "label": 0, "text": "Derde."}]
    ids, how = fy.classifier_exclusions(bt, clf)
    assert ids == {"fineweb:x:d:1", "fineweb:y:d:2"}  # a back-translation's text (label 0) does not count
    assert how == {"by_pid": 1, "by_text": 1, "either": 2}


def passages(n_docs, per_doc, tokens, prefix="d"):
    return [{"pid": f"fineweb:{prefix}{d}:d:{k}", "source": "fineweb", "doc_id": f"{prefix}{d}",
             "text": f"{prefix} tekst {d} {k} " * 3, "tokens": tokens, "fails": []}
            for d in range(n_docs) for k in range(per_doc)]


def test_the_supplement_draws_one_passage_per_document_a_quarter_long_and_reproducibly():
    pool = passages(40, 3, 100) + passages(20, 2, 500, prefix="L")
    pool.append({"pid": "fineweb:bad:d:0", "source": "fineweb", "doc_id": "bad", "text": "fout", "tokens": 100,
                 "fails": ["langid"]})
    drawn = fy.draw_supplement(pool, 20, 2015)
    assert len(drawn) == 20
    assert sum(p["length_class"] == "long" for p in drawn) == 5
    assert len({p["doc_id"] for p in drawn}) == 20
    assert "bad" not in {p["doc_id"] for p in drawn}
    again = fy.draw_supplement([dict(p, fails=list(p["fails"])) for p in pool], 20, 2015)
    assert [p["pid"] for p in again] == [p["pid"] for p in drawn]
    with pytest.raises(SystemExit, match="only 2 long passages"):
        fy.draw_supplement(passages(30, 1, 100) + passages(2, 1, 500, prefix="L"), 20, 2014)


def meta_and_predictions(spec):
    """spec: [(group, year, half, k original, n)]"""
    meta, pred = [], []
    for group, year, half, k, n in spec:
        for i in range(n):
            pid = f"{group}-{year}-{half}-{i}"
            meta.append({"id": pid, "group": group, "dump": f"CC-MAIN-{year}-10", "year": year, "half": half})
            pred.append({"model": group, "set": f"CC-MAIN-{year}-10", "id": pid,
                         "p_original": 0.9 if i < k else 0.1, "truncated": False})
    return meta, pred


def test_year_check_pools_the_baseline_and_keeps_groups_apart():
    meta, pred = meta_and_predictions([("bt", y, "random", 90, 100) for y in fy.BASELINE_YEARS]
                                      + [("bt", 2023, "idiom", 50, 300), ("extra", 2015, "random", 600, 1000),
                                         ("bt", 2015, "random", 5, 10)])
    per_year, per_dump, base, per_half = fy.year_check(meta, pred)
    assert base["n"] == 500 and base["k"] == 450
    assert per_year["bt"][2023]["share"] == pytest.approx(50 / 300)
    assert per_year["extra"][2015]["n"] == 1000 and per_year["bt"][2015]["n"] == 10
    assert set(per_dump["bt"]) == {"CC-MAIN-2023-10"}  # only crawls with n >= 200
    assert set(per_half) == {"random", "idiom"}
    with pytest.raises(SystemExit, match="disagree"):
        fy.year_check(meta, pred[:-1])


def write_parquet(path, rows):
    import pyarrow as pa
    import pyarrow.parquet as pq

    pq.write_table(pa.Table.from_pylist(rows), path)


def test_report_states_the_fixed_rule_and_the_first_declining_year(tmp_path, capsys):
    meta, pred = meta_and_predictions([("bt", y, "random", 900, 1000) for y in fy.BASELINE_YEARS]
                                      + [("bt", 2022, "random", 880, 1000), ("bt", 2023, "random", 780, 1000),
                                         ("bt", 2024, "random", 700, 1000), ("extra", 2014, "random", 900, 1000),
                                         ("bt", 2014, "random", 70, 79)])
    write_parquet(tmp_path / "meta.parquet", meta)
    write_parquet(tmp_path / "pred.parquet", pred)
    (tmp_path / "data.json").write_text(json.dumps({
        "rule": {"text": fy.RULE_NL, "fixed_at": fy.RULE_FIXED_AT},
        "bt": {"repo": "r", "revision": "0123456789abcdef", "rows": 59408, "kept": 59000,
               "left_out_classifier": {"by_pid": 400, "by_text": 401, "either": 408}},
        "fineweb": {"repo": "f", "revision": "fedcba9876543210", "min_per_year": 1000, "row_groups_per_year": 30,
                    "seed": "fw2year"}}))
    (tmp_path / "results.json").write_text(json.dumps({"test_accuracy": 0.9575}))
    (tmp_path / "sources.json").write_text(json.dumps({"targets": {"bt": {
        "all": {"share": 0.9525, "n": 2000}, "set/fineweb": {"share": 0.9191, "n": 544}}}}))
    args = ["report", "--meta", str(tmp_path / "meta.parquet"), "--predictions", str(tmp_path / "pred.parquet"),
            "--data-json", str(tmp_path / "data.json"), "--classifier-results", str(tmp_path / "results.json"),
            "--bt-sources", str(tmp_path / "sources.json"), "--commit", "abc1234",
            "--out", str(tmp_path / "r.md"), "--json", str(tmp_path / "r.json")]
    fy.main(args)
    md = (tmp_path / "r.md").read_text()
    assert fy.RULE_NL in md and fy.RULE_FIXED_AT in md
    assert "Eerste \"dalende\" crawljaar (BT-passages): 2023." in md
    assert "eerste jaar waarvoor de regel geldt: geen." in md
    assert "| fineweb | 544 | 91,9% |" in md
    out = json.loads((tmp_path / "r.json").read_text())
    assert out["first_decline"] == {"bt": 2023, "extra": None}
    assert "FW2_RULE base=0.9000" in capsys.readouterr().out
    # a rule changed after the data were made stops the report
    (tmp_path / "data.json").write_text(json.dumps({"rule": {"text": "anders", "fixed_at": fy.RULE_FIXED_AT}}))
    with pytest.raises(SystemExit, match="rule changed"):
        fy.main(args)


def texts(model, sets=(("wmt24pp", 3), ("flan", 2))):
    return [{"model": model, "set": st, "id": f"{st}-{i}", "text": f"{model} {st} {i}"} for st, n in sets for i in range(n)]


def test_g4_combine_checks_one_model_per_file_and_the_base_models_ids():
    base = texts("base") + texts("B")
    rows, models = fy.g4_combine(base, {"gemma4-e2b-en-nl": texts("G4-e2b"), "gemma4-12b-en-nl": texts("G4-12b")})
    assert models == {"gemma4-e2b-en-nl": "G4-e2b", "gemma4-12b-en-nl": "G4-12b"}
    assert len(rows) == 15 and {r["model"] for r in rows} == {"base", "G4-e2b", "G4-12b"}
    with pytest.raises(SystemExit, match="expected one model"):
        fy.g4_combine(base, {"x": texts("G4-e2b") + texts("G4-e4b")})
    with pytest.raises(SystemExit, match="expected one model"):
        fy.g4_combine(base, {"x": texts("base")})
    with pytest.raises(SystemExit, match="also another file's"):
        fy.g4_combine(base, {"x": texts("G4"), "y": texts("G4")})
    with pytest.raises(SystemExit, match="not the base model's"):
        fy.g4_combine(base, {"x": texts("G4", (("wmt24pp", 3), ("flan", 1)))})


def test_g4report_applies_rule_b_against_the_base_model(tmp_path, capsys):
    pred = []
    for model, k_wmt, k_flan in (("base", 2, 0), ("G4-e2b", 3, 2), ("G4-12b", 2, 0)):
        for st, n, k in (("wmt24pp", 10, k_wmt), ("flan", 10, k_flan)):
            pred += [{"model": model, "set": st, "id": f"{st}-{i}", "p_original": 0.9 if i < k else 0.1}
                     for i in range(n)]
    write_parquet(tmp_path / "pred.parquet", pred)
    (tmp_path / "names.json").write_text(json.dumps({"gemma4-e2b-en-nl": "G4-e2b", "gemma4-12b-en-nl": "G4-12b"}))
    (tmp_path / "results.json").write_text(json.dumps({"test_accuracy": 0.9575}))
    fy.main(["g4report", "--predictions", str(tmp_path / "pred.parquet"), "--names", str(tmp_path / "names.json"),
             "--classifier-results", str(tmp_path / "results.json"), "--out", str(tmp_path / "g4.md"),
             "--json", str(tmp_path / "g4.json")])
    out = json.loads((tmp_path / "g4.json").read_text())["models"]
    assert out["gemma4-e2b-en-nl"]["delta"] == pytest.approx(0.15) and out["gemma4-e2b-en-nl"]["holds"]
    assert out["gemma4-12b-en-nl"]["delta"] == pytest.approx(0.0) and not out["gemma4-12b-en-nl"]["holds"]
    md = (tmp_path / "g4.md").read_text()
    assert "| gemma4-e2b-en-nl | G4-e2b | 30,0% | 20,0% | 25,0% | 10,0% | +15,0 pp | gehaald |" in md
    assert capsys.readouterr().out.count("FW2_G4 ") == 2
