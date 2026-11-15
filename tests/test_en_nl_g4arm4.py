"""g4arm.py for arm G4-4 (protocol-g4.md §9, amendment G4-4) on small synthetic inputs (no tokenizer, no model, no
TPU): G4-3's set taken over pass by pass, the English novels drawn apart per pass (at most 20 a book, one Dutch edition,
one English id per work, near duplicates and 13-grams out), the replay that keeps 10% of every pass from G4-3's
replay order, the check, the smoke test's verdict and estimate, the report beside G4-3."""

import json
import random

import pyarrow.parquet as pq
import pytest

# test_en_nl_g4arm3 puts experiments/en-nl-translator on the path; its world fixture is G4-3's inputs
import test_en_nl_g4arm3 as t3

g4arm = t3.g4arm
ADAPTER, EN_WORDS, SMOKE_MANIFEST, WORDS = t3.ADAPTER, t3.EN_WORDS, t3.SMOKE_MANIFEST, t3.WORDS
FakeCounter, ns, run_build3, run_heldout3, sha = t3.FakeCounter, t3.ns, t3.run_build3, t3.run_heldout3, t3.sha
smoke3_summary, text, write = t3.smoke3_summary, t3.text, t3.write
world = t3.world

EVAL_REF = ("de commissie heeft na lang beraad besloten het voorstel over de nieuwe brug bij de rivier "
            "voorlopig niet in stemming te brengen")
LONG_EN = " ".join(EN_WORDS[k % len(EN_WORDS)] for k in range(2100))


def novel_row(rng, en_book, nl_book, k, **kw):
    """A row of b1-selected.parquet: group en_orig, every check passed, its own random passage"""
    en = text(rng, EN_WORDS, 30)
    row = {"id": f"{nl_book}-{en_book}-b1-{k}", "book": f"{nl_book}-{en_book}", "nl_book_id": nl_book,
           "en_book_id": en_book, "group": "en_orig", "author": f"author {en_book}", "en": en,
           "nl": text(rng, WORDS, 32), "en_raw": None, "nl_raw": None, "passes": True, "dup_id": False,
           "dup_nl": False, "dup_en": False, "original": "en", "published": "2010"}
    row.update(kw)
    return row


def novels_file(path):
    """English books: 101 (70 rows; a second Dutch edition 202 with 30), 102 (45 and five that fail a check), 103 (15,
    one with an evaluation 13-gram, one too long), 104 (50) and 105 (30, of which 12 with 104's English: one work under
    two ids), 106 (40, of which 3 begin as rows of 102: a preview); a row of another group"""
    rng = random.Random(11)
    rows = [novel_row(rng, "101", "201", k) for k in range(70)]
    rows += [novel_row(rng, "101", "202", k) for k in range(30)]
    b102 = [novel_row(rng, "102", "203", k) for k in range(45)]
    rows += b102
    rows += [novel_row(rng, "102", "203", 100, en="a **bold** start " + text(rng, EN_WORDS, 20)),
             novel_row(rng, "102", "203", 101, dup_nl=True), novel_row(rng, "102", "203", 102, passes=False),
             novel_row(rng, "102", "203", 103, en_raw="- " + text(rng, EN_WORDS, 20)),
             novel_row(rng, "102", "203", 104, dup_id=True)]
    rows += [novel_row(rng, "103", "204", k) for k in range(15)]
    rows += [novel_row(rng, "103", "204", 50, nl="en toen " + EVAL_REF + " zei hij"),
             novel_row(rng, "103", "204", 51, en=LONG_EN)]
    b104 = [novel_row(rng, "104", "205", k) for k in range(50)]
    rows += b104
    rows += [novel_row(rng, "105", "206", k, en=b104[k]["en"] + " again") for k in range(12)]
    rows += [novel_row(rng, "105", "206", k) for k in range(12, 30)]
    rows += [novel_row(rng, "106", "207", k, en=b102[k]["en"] + " later") for k in range(3)]
    rows += [novel_row(rng, "106", "207", k) for k in range(3, 40)]
    rows.append({**novel_row(rng, "137307", "69915", 0), "group": "nl_orig_nlbooks"})
    write(rows, path)
    return rows


@pytest.fixture
def world4(world):
    # an evaluation reference long enough for 13-grams, so that a novel row can share one
    write([{"id": "wmt-0", "set": "wmt24pp", "source": "a b c", "reference": EVAL_REF, "meta": "{}"}],
          world["tmp"] / "evalsets.parquet")
    set3 = run_build3(world, run_heldout3(world))
    novels_file(world["tmp"] / "b1-selected.parquet")
    return {**world, "set3": set3}


def run_build4(w, **kw):
    out = w["tmp"] / "set4"
    args = dict(g43=str(w["set3"] / "G4-3"), g43_report=str(w["set3"] / "report.json"),
                novels=str(w["tmp"] / "b1-selected.parquet"), replay=str(w["tmp"] / "replay.parquet"),
                replay_model=g4arm.REPLAY_MODEL2, replay_share=0.1, epochs=3,
                flan_eval=str(w["tmp"] / "eval-rows.parquet"), evalsets=str(w["tmp"] / "evalsets.parquet"),
                blind=str(w["blind"]), tokenizer="unused", out=str(out))
    args.update(kw)
    g4arm.build4(ns(**args), counter=FakeCounter())
    return out


def per_pass(metas, kind):
    return [[m for m in metas if m["epoch"] == e and m["kind"] == kind] for e in (1, 2, 3)]


def test_build4_is_g4_3_per_pass_plus_novels_apart_and_10_percent_replay(world4):
    out = run_build4(world4)
    m3 = pq.read_table(world4["set3"] / "G4-3" / "train-meta.parquet").to_pylist()
    x3 = pq.read_table(world4["set3"] / "G4-3" / "train.parquet").column("messages").to_pylist()
    m4 = pq.read_table(out / "G4-4" / "train-meta.parquet").to_pylist()
    x4 = pq.read_table(out / "G4-4" / "train.parquet").column("messages").to_pylist()
    report = json.loads((out / "report.json").read_text())
    assert len(m4) == len(x4) and [m["epoch"] for m in m4] == sorted(m["epoch"] for m in m4)
    # every example of G4-3's pass k is in G4-4's pass k, the same example, in another order
    for e in (1, 2, 3):
        three = {m["id"]: (x, {k: v for k, v in m.items()}) for x, m in zip(x3, m3) if m["epoch"] == e}
        four = {m["id"]: (x, m) for x, m in zip(x4, m4) if m["epoch"] == e}
        assert set(three) <= set(four)
        assert all(four[i] == three[i] for i in three)
        assert [i for i in four if i in three] != [m["id"] for m in m3 if m["epoch"] == e]
    # the novels: at most 20 of a book in a pass, a book's next ones in the next pass, no pair twice
    novels = per_pass(m4, "novel")
    by = [{} for _ in range(3)]
    for e, rows in enumerate(novels):
        for m in rows:
            book = m["id"].split(":")[1].rsplit("-b1-", 1)[0]
            by[e][book] = by[e].get(book, 0) + 1
    assert by[0] == {"201-101": 20, "203-102": 20, "204-103": 15, "205-104": 20, "207-106": 20}
    assert by[1] == {"201-101": 20, "203-102": 20, "205-104": 20, "207-106": 17}
    assert by[2] == {"201-101": 20, "203-102": 5, "205-104": 10}
    ids = [m["id"] for rows in novels for m in rows]
    assert len(ids) == len(set(ids)) == report["novels"]["distinct_pairs"] == 95 + 77 + 35
    # failing checks, the other Dutch edition, the other id of a work, a preview, a 13-gram, too long: never drawn
    assert not any(i.startswith(("novel:202-101", "novel:206-105")) for i in ids)
    assert not any(i in ids for i in ("novel:203-102-b1-100", "novel:203-102-b1-101", "novel:203-102-b1-102",
                                      "novel:203-102-b1-103", "novel:203-102-b1-104", "novel:204-103-b1-50",
                                      "novel:204-103-b1-51", "novel:207-106-b1-0", "novel:207-106-b1-1",
                                      "novel:207-106-b1-2"))
    w = report["novels"]["same_works"]
    assert w["books_dropped"] == 1 and w["dropped"][0]["en_book_id"] == "105"
    assert report["novels"]["near_duplicates"]["other_book"] == 3
    assert report["novels"]["draw"]["eval_13gram"] == 1 and report["novels"]["draw"]["too_long"] == 1
    assert report["novels"]["groups"] == {"en_orig": len(pq.read_table(world4["tmp"] / "b1-selected.parquet")) - 1,
                                          "nl_orig_nlbooks": 1}
    # the replay: G4-3's rows of the pass, plus the next rows of its order, 10% of every pass, apart over the passes
    r3 = json.loads((world4["set3"] / "report.json").read_text())["replay"]["per_pass"]
    rep = per_pass(m4, "replay")
    rep3 = per_pass(m3, "replay")
    seen = set()
    for e in range(3):
        n = sum(1 for m in m4 if m["epoch"] == e + 1)
        assert len(rep[e]) == round((n - len(rep[e])) / 9)
        assert {m["id"] for m in rep3[e]} <= {m["id"] for m in rep[e]} and len(rep3[e]) == r3
        added = {m["id"] for m in rep[e]} - {m["id"] for m in rep3[e]}
        assert not added & {m["id"] for rows in rep3 for m in rows}
        assert not seen & {m["id"] for m in rep[e]}
        seen |= {m["id"] for m in rep[e]}
    assert report["replay"]["distinct"] == len(seen) and report["replay"]["apart_in_every_pass"]
    # the estimate is the packing of the file's rows; the smoke set: pass 1's longest first
    lengths = [m["tokens_prompt"] + m["tokens_target"] for m in m4]
    assert report["estimate"]["batches_per_process"] == g4arm.pack_batches(lengths)
    manifest = json.loads((out / "G4-4" / "manifest.json").read_text())
    assert manifest["set"] == "G4-4" and manifest["seq"] == g4arm.SEQ3 and manifest["steps"] == report["estimate"]["steps"]
    sm = pq.read_table(out / "G4-4-smoke" / "train-meta.parquet").to_pylist()
    tok = [m["tokens_prompt"] + m["tokens_target"] for m in sm]
    n_long = min(g4arm.SMOKE3_LONGEST, sum(1 for m in m4 if m["epoch"] == 1))
    assert tok[:n_long] == sorted(tok[:n_long], reverse=True) and min(tok[:n_long]) >= max(tok[n_long:], default=0)
    assert json.loads((out / "G4-4-smoke" / "manifest.json").read_text())["batches_per_process"] == \
        g4arm.pack_batches(tok)
    text_md = (out / "report.md").read_text()
    assert "# Trainingsset G4-4" in text_md and "Verschillende romanparen over de drie doorgangen: 207" in text_md


def test_build4_writes_row_groups_of_at_most_row_group4_rows(world4, monkeypatch):
    monkeypatch.setattr(g4arm, "ROW_GROUP4", 700)
    out = run_build4(world4)
    for name in ("train.parquet", "train-meta.parquet"):
        f = pq.ParquetFile(out / "G4-4" / name)
        sizes = [f.metadata.row_group(k).num_rows for k in range(f.metadata.num_row_groups)]
        assert len(sizes) > 1 and max(sizes) <= 700 and sum(sizes) == f.metadata.num_rows
    # the rows in the file's order, as datasets reads them batch by batch
    import pyarrow.dataset as pads

    with open(out / "G4-4" / "train.parquet", "rb") as f:
        frag = pads.ParquetFileFormat().make_fragment(f)
        batched = [r for b in frag.to_batches(batch_size=frag.row_groups[0].num_rows) for r in b.to_pylist()]
    assert batched == pq.read_table(out / "G4-4" / "train.parquet").to_pylist()


def test_build4_is_the_same_twice(world4):
    a = run_build4(world4)
    first = sha(a / "G4-4" / "train.parquet"), sha(a / "G4-4" / "train-meta.parquet")
    b = run_build4(world4)
    assert (sha(b / "G4-4" / "train.parquet"), sha(b / "G4-4" / "train-meta.parquet")) == first


def test_build4_refuses_inputs_g4_3_was_not_built_from(world4):
    rng = random.Random(5)
    other = world4["tmp"] / "replay-other.parquet"
    write([{"id": 10_000 + k, "en": text(rng, EN_WORDS, 40), "nl": text(rng, WORDS, 40), "task": "flan/t/x",
            "model": g4arm.REPLAY_MODEL2} for k in range(3000)], other)
    with pytest.raises(SystemExit, match="not the replay file"):
        run_build4(world4, replay=str(other))
    rep = json.loads((world4["set3"] / "report.json").read_text())
    rep["passes"][1]["kinds"]["book"]["tokens"] += 1
    (world4["tmp"] / "report-bad.json").write_text(json.dumps(rep))
    with pytest.raises(SystemExit, match="not the pass"):
        run_build4(world4, g43_report=str(world4["tmp"] / "report-bad.json"))


def test_build4_refuses_a_g4_3_set_whose_replay_comes_back(world):
    rng = random.Random(4)
    write([{"id": 10_000 + k, "en": text(rng, EN_WORDS, 40), "nl": text(rng, WORDS, 40), "task": "flan/t/x",
            "model": g4arm.REPLAY_MODEL2} for k in range(300)], world["tmp"] / "replay.parquet")
    set3 = run_build3(world, run_heldout3(world))
    novels_file(world["tmp"] / "b1-selected.parquet")
    with pytest.raises(SystemExit, match="replay row in two passes"):
        run_build4({**world, "set3": set3})


def test_same_works_keeps_the_larger_id_and_near_duplicates_go():
    rng = random.Random(3)
    a = [novel_row(rng, "7", "1", k) for k in range(30)]
    b = [novel_row(rng, "8", "2", k, en=a[k]["en"].upper()) for k in range(g4arm.SAME_WORK_MIN_SHARED)]
    b += [novel_row(rng, "8", "2", k) for k in range(20, 25)]
    c = [novel_row(rng, "9", "3", k, nl=a[k]["nl"] + " extra") for k in range(g4arm.SAME_WORK_MIN_SHARED - 1)]
    c += [novel_row(rng, "9", "3", k) for k in range(20, 30)]
    rows = a + b + c
    chosen, _ = g4arm.b1_editions(rows)
    kept, info = g4arm.same_works(rows, chosen)
    assert set(kept) == {"7", "9"} and info["dropped"] == [{"en_book_id": "8", "rows": 15, "kept_instead": "7"}]
    left, near = g4arm.near_duplicates(kept)
    assert len(left["7"]) == 30 and len(left["9"]) == 10 and near["other_book"] == 9


def test_check_of_g4_4_is_g4_3s_check_of_distinct_examples(world4, monkeypatch):
    import g4lora

    d = run_build4(world4) / "G4-4"
    metas = pq.read_table(d / "train-meta.parquet").to_pylist()
    want = {m["id"]: m["tokens_prompt"] + m["tokens_target"] for m in metas}
    calls = []

    def fake(messages, ens, path, model, labels):
        calls.append(len(labels))
        return [want[i] for i in labels], sum(want[i] for i in labels), 0

    monkeypatch.setattr(g4lora, "check_examples", fake)
    g4arm.check(ns(data=str(d), tokenizer="unused", arm="G4-4", processes=1, chunk=500))
    assert sum(calls) == len(want) < len(metas)


def test_the_smoke4_estimate_is_three_passes_and_stops_above_five_days(tmp_path):
    report = {"arm": "G4-4", "estimate": {"steps": 9500, "tokens": 590_000_000, "fill": 0.95}}
    x = g4arm.smoke4_estimate(smoke3_summary(), report, SMOKE_MANIFEST)
    assert x["passes"] == 3 and x["steps"] == 9500 and x["hours"] == x["three"]["hours"] < 40
    slow = g4arm.smoke4_estimate(smoke3_summary(step_s=50.0), report, SMOKE_MANIFEST)
    assert slow["hours"] > 120 and slow["passes"] is None
    files = {}
    for name, value in (("summary", smoke3_summary(step_s=50.0)), ("adapter", ADAPTER),
                        ("smoke-manifest", SMOKE_MANIFEST), ("data-report", report)):
        files[name] = tmp_path / f"{name}.json"
        files[name].write_text(json.dumps(value))
    args = ns(summary=str(files["summary"]), adapter=str(files["adapter"]), smoke_manifest=str(files["smoke-manifest"]),
              data_report=str(files["data-report"]), out=str(tmp_path / "g4arm4-smoke.md"))
    with pytest.raises(SystemExit):
        g4arm.smoke4(args)
    assert json.loads((tmp_path / "g4arm4-smoke.json").read_text())["passed"]
    files["summary"].write_text(json.dumps(smoke3_summary()))
    g4arm.smoke4(args)
    assert "Smoke-test G4-4" in (tmp_path / "g4arm4-smoke.md").read_text()


def test_report_of_g4_4_puts_g4_3_beside_on_the_same_measures(tmp_path, monkeypatch):
    import armeval

    monkeypatch.setattr(g4arm, "N_WMT_GOOD", 2)
    monkeypatch.setattr(g4arm, "N_FLAN", 1)
    monkeypatch.setattr(armeval, "chrf", lambda h, r: 50.0 if h == r else 10.0)
    monkeypatch.setattr(armeval, "corpus_chrf",
                        lambda hs, rs: sum(50.0 if h == r else 10.0 for h, r in zip(hs, rs)) / len(hs))
    evalsets = [{"id": f"{s}-{k}", "set": s, "source": "a b c", "reference": None if s in ("flan", "blind") else "x y z",
                 "tokens": 3, "over_512": False, "length_class": "tiny", "meta": "{}"}
                for s in ("wmt24pp", "flores", "ntrex", "europarl", "flan", "blind") for k in range(2)]
    evalsets = [r for r in evalsets if r["id"] != "flan-1"]
    write(evalsets, tmp_path / "evalsets.parquet")
    for name, qe in (("G4-4", 1.0), ("G4-0", 2.0), ("G4-3", 1.25)):
        d = tmp_path / name
        write([{"id": r["id"], "set": r["set"], "source": r["source"], "hypothesis": "p q r", "reference": r["reference"],
                "ended": True, "decode_loop": False, "english_left": False, "empty": False} for r in evalsets],
              d / "pairs" / "pairs-w0.parquet")
        write([{"id": r["id"], "qe_source_hypothesis": qe,
                "ref_source_hypothesis_reference": qe if r["reference"] else None} for r in evalsets],
              d / "scores" / "metricx-w0.parquet")
        write([{"id": r["id"], "comet22": 0.85} for r in evalsets if r["reference"]], d / "scores" / "comet-w0.parquet")
    write([{"id": r["id"], "set": r["set"], "metricx_qe": 1.5, "metricx_ref": 1.4, "comet22": 0.86}
           for r in evalsets if r["set"] in g4arm.FOUR_SETS], tmp_path / "tg.parquet")
    preds = []
    for model, p in (("G4-4", 0.9), ("G4-0", 0.1), ("G4-3", None), ("TG", 0.1)):
        preds += [{"model": model, "set": s, "id": i, "truncated": False,
                   "p_original": p if p is not None else (0.9 if i == "wmt24pp-0" else 0.1)}
                  for s, i in (("wmt24pp", "wmt24pp-0"), ("wmt24pp", "wmt24pp-1"), ("flan", "flan-0"))]
    write(preds, tmp_path / "pred.parquet")
    held = [{"id": f"heldout:{s}:{k}", "set": s, "source": "e n", "reference": "o r", "tokens": 2, "over_512": False,
             "length_class": "tiny", "meta": json.dumps({"source": "raad" if s != "books" else "books:a-b"})}
            for s in ("books", "btval", "btlong") for k in range(2)]
    write(held, tmp_path / "held.parquet")
    for name, hyp in (("G4-4", "o r"), ("G4-0", "x"), ("G4-3", "x")):
        write([{"id": r["id"], "set": r["set"], "hypothesis": hyp, "reference": r["reference"], "ended": True,
                "decode_loop": False, "empty": False} for r in held], tmp_path / "h" / name / "pairs-w0.parquet")
    for name, last in (("summary", 1.4), ("summary3", 1.5)):
        (tmp_path / f"{name}.json").write_text(json.dumps({
            "steps": 300, "steps_done": 300, "loss": [[1, 2.3], [300, 0.8]], "lr_at": {"0": 0.0},
            "eval": [{"step": st, "set": s, "loss": x, "tokens": 9} for st, x in ((0, 2.2), (300, last))
                     for s in ("books", "btval", "own", "btlong")]}))
    out = tmp_path / "report.md"
    args = ns(evalsets=str(tmp_path / "evalsets.parquet"), arm=str(tmp_path / "G4-4"), base=str(tmp_path / "G4-0"),
              tg_scores=str(tmp_path / "tg.parquet"), predictions=str(tmp_path / "pred.parquet"), classifier_results=None,
              lora_line=[], train_summary=str(tmp_path / "summary.json"), tally=None, arm_name="G4-4",
              heldout=[str(tmp_path / "held.parquet")], heldout_arm=[str(tmp_path / "h" / "G4-4" / "pairs-w*.parquet")],
              heldout_base=[str(tmp_path / "h" / "G4-0" / "pairs-w*.parquet")], other=f"G4-3={tmp_path / 'G4-3'}",
              heldout_other=[str(tmp_path / "h" / "G4-3" / "pairs-w*.parquet")],
              other_summary=str(tmp_path / "summary3.json"), other_tally=None, out=str(out))
    g4arm.report(args)
    r = json.loads(out.with_suffix(".json").read_text())
    # the decision stays against G4-0
    assert r["base"] == "G4-0" and r["b"]["base"] == 0.0 and r["b"]["delta"] == pytest.approx(1.0) and r["b"]["holds"]
    o = r["other"]
    assert o["name"] == "G4-3" and o["b"]["base"] == pytest.approx(1 / 3) and o["b"]["delta"] == pytest.approx(2 / 3)
    assert o["a1"]["sets"]["flores"]["model"] == pytest.approx(1.25) and o["loss"]["books"]["300"] == 1.5
    assert r["sets"]["wmt24pp"]["G4-3"]["metricx_qe"] == pytest.approx(1.25)
    assert r["other_heldout"]["sets"]["books"]["G4-3"]["chrf_corpus"] == 10.0
    md = out.read_text()
    assert "## Vergelijking met G4-3 (rapporteren, beslist niet)" in md and "| books | 50,00 | 10,00 | ja | 1,400 | 1,500 |" in md
