"""bt_length.py (the length trial of 13 Nov): bins, stretches, pieces, the alignment and its omissions, the measures of
a translation, the quota per source, the decision rule exact on k/n, the units still to score, the document models of
translate_models.yml and the MEM line of bt_length_translate.py. Made-up data; no network, model or TPU."""

import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "experiments" / "en-nl-translator", ROOT / "experiments" / "flan-nl"):
    sys.path.insert(0, str(_p))

import bt_length as bl  # noqa: E402
import bt_length_translate as blt  # noqa: E402


def words_count(text):
    """A stand-in token count: one token per word, two per paragraph break (as the stretch search assumes)"""
    return len(text.split()) + 2 * text.count("\n\n")


def test_bin_of_takes_more_than_three_quarters_and_at_most_the_bin():
    assert bl.bin_of(512) == 512
    assert bl.bin_of(385) == 512
    assert bl.bin_of(384) is None  # 0.75 x 512: not more than it
    assert bl.bin_of(1024) == 1024
    assert bl.bin_of(769) == 1024
    assert bl.bin_of(16384) == 16384
    assert bl.bin_of(16385) is None


def test_paragraphs_of_a_translation_are_the_blocks_between_empty_lines():
    text = "First.\n\nSecond line one\nline two\n\n \n\nThird."
    assert bl.paragraphs_of_text(text) == ["First.", "Second line one\nline two", "Third."]
    assert bl.join(["a", "b"]) == "a\n\nb"


def test_pieces_are_whole_paragraphs_up_to_the_limit():
    paras = ["a " * 300, "b " * 150, "c " * 100, "d " * 600, "e " * 10]
    out = bl.pieces([p.strip() for p in paras], words_count, limit=512)
    assert [len(p) for p in out] == [2, 1, 1, 1]  # 300+2+150 fits; +100 does not; 600 stands alone; then 10
    assert all(words_count(bl.join(p)) <= 512 for p in out if len(p) > 1)
    assert [x for p in out for x in p] == [p.strip() for p in paras]


def test_find_stretch_keeps_whole_paragraphs_in_the_bin_and_skips_long_paragraphs():
    paras = [("w " * n).strip() for n in (100, 100, 600, 100, 100, 100, 100, 100)]
    counts = [len(p.split()) for p in paras]
    prose = [True] * len(paras)
    starts = [True] * len(paras)
    found = bl.find_stretch(paras, starts, counts, prose, words_count, 512, random.Random(0), lambda t: True)
    assert found is not None
    i, j, n = found
    assert i >= 3  # paragraph 2 (600 words) is longer than MAX_PARAGRAPH_TOKENS, nothing before it reaches 384
    assert 0.75 * 512 < n <= 512
    assert n == words_count(bl.join(paras[i:j]))


def test_find_stretch_needs_its_prose_share_and_accept():
    paras = [("w " * 100).strip()] * 5
    counts = [100] * 5
    assert bl.find_stretch(paras, [True] * 5, counts, [False] * 5, words_count, 512, random.Random(0),
                           lambda t: True) is None
    assert bl.find_stretch(paras, [True] * 5, counts, [True] * 5, words_count, 512, random.Random(0),
                           lambda t: False) is None
    # a stretch begins only where starts allows (the Handelingen: at a speaker's paragraph)
    found = bl.find_stretch(paras, [False, False, True, False, False], counts, [True] * 5, words_count, 512,
                            random.Random(0), lambda t: True)
    assert found is None  # from paragraph 2 only 3 x 100 + joins remain: not more than 384


def test_handelingen_stretch_paragraphs_begin_each_turn_with_its_speaker():
    doc = {"turns": [{"speaker": "De heer Klaver", "party": "GroenLinks", "text": "Voorzitter.\nIk heb een vraag."},
                     {"speaker": "De voorzitter", "party": None, "text": "Gaat uw gang."},
                     {"speaker": None, "party": None, "text": ""},
                     {"speaker": "Minister Rutte", "party": "VVD", "text": "Dank u."}]}
    paras, starts = bl.handelingen_paragraphs(doc)
    assert paras == ["De heer Klaver (GroenLinks):", "Voorzitter.", "Ik heb een vraag.", "De voorzitter:",
                     "Gaat uw gang.", "Minister Rutte (VVD):", "Dank u."]
    assert starts == [True, False, False, True, False, True, False]


def test_align_finds_a_paragraph_left_out_as_one():
    # lengths alone tell which paragraph is missing only when they differ (equal ones are a tie)
    src = ["x" * 200, "y" * 600, "z" * 400]
    tgt = ["X" * 210, "Z" * 390]
    assert bl.align(src, tgt, 1.0) == [([0], [0]), ([1], []), ([2], [1])]
    # long paragraphs: a missing one is skipped rather than merged with its neighbour
    src = ["x" * 1000, "y" * 1000, "z" * 1000]
    tgt = ["X" * 1000, "Z" * 1000]
    units = bl.align(src, tgt, 1.0)
    assert sum(1 for s, t in units if s and not t) == 1 and all(len(s) == 1 for s, t in units)


def test_align_merges_and_splits_and_adds():
    src = ["x" * 300, "y" * 300, "z" * 500]
    tgt = ["X" * 610, "Z" * 240, "Z" * 250]
    assert bl.align(src, tgt, 1.0) == [([0, 1], [0]), ([2], [1, 2])]
    # an added paragraph between two long ones
    assert bl.align(["x" * 1000, "z" * 2000], ["X" * 1000, "N" * 1500, "Z" * 2000], 1.0) == [
        ([0], [0]), ([], [1]), ([1], [2])]
    assert bl.align([], ["a"], 1.0) == [([], [0])]
    assert bl.align(["a"], [], 1.0) == [([0], [])]


def test_omissions_count_nothing_aligned_and_r2s_length_bound_by_direction():
    bounds = {"lo": 0.75, "hi": 1.34}
    src = ["a" * 100, "b" * 100, "c" * 100, "d" * 100]
    tgt = ["A" * 100, "B" * 50, "D" * 200]
    units = [([0], [0]), ([1], [1]), ([2], []), ([3], [2])]
    status, ratios = bl.omissions(units, src, tgt, bounds, "nl")
    # nl->en: NL/EN 2.0 is English too short, 0.5 too long
    assert status == [None, "short", "none", "long"]
    assert ratios == [1.0, 2.0, None, 0.5]
    # en->nl: source English, translation Dutch: NL/EN 0.5 is Dutch too short, 2.0 too long
    status, ratios = bl.omissions(units, src, tgt, bounds, "en")
    assert status == [None, "short", "none", "long"]
    assert ratios == [1.0, 0.5, None, 2.0]


def test_measure_doc_of_a_translation_cut_short_and_one_not_ended():
    doc = {"source": "fineweb", "paragraphs": ["Eerste alinea over de gemeente en haar plannen voor de wijk. " * 4,
                                               "Tweede alinea over het budget van volgend jaar en de kosten. " * 4,
                                               "Derde alinea over de planning en wat er daarna gebeurt. " * 8]}
    doc["paragraphs"] = [p.strip() for p in doc["paragraphs"]]
    full = "\n\n".join([
        ("First paragraph about the municipality and its plans for the neighbourhood. " * 4).strip(),
        ("Second paragraph about next year's budget and the costs. " * 4).strip(),
        ("Third paragraph about the schedule and what happens after that. " * 8).strip()])
    row, units, _ = bl.measure_doc(doc, full, True, "nl", "en")
    assert row["done"] and row["paragraphs_kept"] and row["omitted"] == 0 and not row["cut"]
    assert len(units) == 3 and all(u["n_source"] == 1 for u in units)
    cut = full.rsplit("\n\n", 1)[0]
    row, units, _ = bl.measure_doc(doc, cut, True, "nl", "en")
    assert row["cut"] and not row["done"] and row["omitted_none"] == 1 and not row["paragraphs_kept"]
    row, _, _ = bl.measure_doc(doc, full, False, "nl", "en")
    assert not row["done"] and not row["ended"]
    row, units, _ = bl.measure_doc(doc, "", False, "nl", "en")
    assert row["empty"] and row["omitted"] == 3 and units == []


def test_added_formatting_counts_only_what_the_translation_adds():
    assert bl.added_formatting("Een zin.", "*One* sentence.") == ["italic"]
    assert bl.added_formatting("- een\n- twee", "- one\n- two") == []
    assert bl.added_formatting("Kop\n\nTekst.", "## Heading\n\nText.") == ["heading"]


def test_quota_fills_a_short_source_from_the_others_reserves():
    q = bl.Quota(["a", "b"], (512, 1024), per_source=2, reserve=2)
    for k in range(4):
        q.add("a", 1024, {"id": f"a{k}", "source": "a", "bin": 1024})
    q.add("b", 1024, {"id": "b0", "source": "b", "bin": 1024})
    assert q.wanted("a") == [512]
    assert q.wanted("b") == [1024, 512]
    chosen = [r for r in q.chosen() if r["bin"] == 1024]
    assert [r["id"] for r in chosen] == ["a0", "a1", "b0", "a2"]
    assert chosen[-1]["filled_for"] == "shortfall"
    with pytest.raises(ValueError):
        q.add("a", 1024, {"id": "a4"})


def stat(docs, done, kept, paragraphs, omitted, qe):
    return {"docs": docs, "done": done, "paragraphs_kept": kept, "paragraphs": paragraphs, "omitted": omitted,
            "qe_mean": qe}


def test_rule_is_exact_on_k_over_n_and_takes_the_largest_bin_that_passes():
    stats = {
        512: {"heel": stat(50, 49, 48, 1000, 20, 3.0), "stukken": stat(50, 50, 50, 1000, 10, 2.8)},
        # 49/50 = 98% passes (i); 47/50 = 94% fails (ii)
        1024: {"heel": stat(50, 49, 47, 1000, 10, 3.0), "stukken": stat(50, 50, 50, 1000, 10, 3.0)},
        2048: {"heel": stat(40, 40, 40, 1000, 21, 3.2), "stukken": stat(40, 40, 40, 1000, 10, 3.0)},
        4096: {"heel": stat(40, 40, 39, 1000, 15, 3.21), "stukken": stat(40, 40, 40, 1000, 10, 3.0)},
    }
    verdict, length = bl.rule(stats)
    assert verdict[512]["pass"]  # +1.0 pp and +0.2 exactly: inside
    assert verdict[1024]["i"] and not verdict[1024]["ii"]
    assert not verdict[2048]["iii"] and verdict[2048]["iv"]  # +1.1 pp
    assert verdict[4096]["iii"] and not verdict[4096]["iv"]  # +0.21
    assert length == 512
    stats[8192] = {"heel": stat(40, 40, 40, 100, 1, 3.0), "stukken": None}
    verdict, length = bl.rule(stats)
    assert not verdict[8192]["measured"] and length == 512


def test_bin_stats_and_qe_per_paragraph():
    rows = [{"id": "d1", "bin": 512, "condition": "heel", "done": True, "ended": True, "decode_loop": False,
             "cut": False, "empty": False, "source_left": False, "chatter": False, "added_formatting": ["italic"],
             "paragraphs_kept": True, "paragraphs_source": 3, "omitted": 1, "omitted_none": 1, "omitted_short": 0,
             "omitted_long": 0, "added_paragraphs": 0, "word_ratio_nl_en": 1.0, "tokens_source": 400,
             "tokens_out": 380}]
    units = [{"id": "u0", "doc": "d1", "condition": "heel", "n_source": 2},
             {"id": "u1", "doc": "d1", "condition": "heel", "n_source": 1},
             {"id": "u2", "doc": "d1", "condition": "heel", "n_source": 1}]
    qe = bl.qe_per_paragraph(units, {"u0": 1.0, "u1": 4.0}, "qe")
    assert qe[("d1", "heel")] == [1.0, 1.0, 4.0]  # a merged unit counts once per source paragraph; u2 unscored
    s = bl.bin_stats(rows, qe, [512])[512]["heel"]
    assert s["qe_mean"] == pytest.approx(2.0) and s["qe_paragraphs"] == 3 and s["formatting"] == {"italic": 1}
    assert bl.bin_stats(rows, qe, [512])[512]["stukken"]["docs"] == 0


def test_balanced_splits_by_weight():
    parts = bl.balanced([10, 9, 1, 1, 1], 2, lambda x: x)
    assert sorted(map(sum, parts)) == [11, 11]


def test_pending_skips_scored_units_and_keeps_columns_of_an_empty_share(tmp_path):
    import pyarrow.parquet as pq

    bl.write_parquet([{"id": "a", "src": "x", "cand": "y"}, {"id": "b", "src": "x", "cand": "y"}],
                     tmp_path / "units.parquet")
    bl.write_parquet([{"id": "a", "qe_src_cand": 1.0}], tmp_path / "qe.parquet")
    bl.main(["pending", "--units", str(tmp_path / "units.parquet"), "--scored", str(tmp_path / "qe.parquet"),
             "--workers", "2", "--out", str(tmp_path / "p")])
    t0, t1 = (pq.read_table(tmp_path / "p" / f"pending-w{k}.parquet") for k in (0, 1))
    assert sorted(t0.column("id").to_pylist() + t1.column("id").to_pylist()) == ["b"]
    assert {t0.num_rows, t1.num_rows} == {0, 1}
    assert t0.column_names == t1.column_names == ["id", "src", "cand"]


def test_collect_joins_pieces_and_needs_all_of_them(tmp_path):
    doc = {"id": "s:1:p0-2", "source": "dpc", "doc_id": "1", "bin": 512, "tokens": 400,
           "paragraphs": ["Een korte alinea over het weer van vandaag.", "Nog een alinea over morgen en overmorgen."]}
    bl.write_parquet([doc], tmp_path / "docs.parquet")
    shard = {"id": ["s:1:p0-2#c000", "s:1:p0-2#c001"],
             "text_en": ["A short paragraph about today's weather.", "Another paragraph about tomorrow and after."],
             "ended": [True, True], "tokens_in": [100, 100], "tokens_out": [12, 10]}
    import pyarrow as pa
    import pyarrow.parquet as pq

    (tmp_path / "st" / "d512").mkdir(parents=True)
    pq.write_table(pa.table(shard), tmp_path / "st" / "d512" / "shard-00000.parquet")
    bl.write_parquet([{"id": i, "text": "x", "tokens": 1} for i in shard["id"]], tmp_path / "seg.parquet")
    bl.main(["collect", "--docs", str(tmp_path / "docs.parquet"), "--stukken", str(tmp_path / "st"),
             "--pieces", str(tmp_path / "seg.parquet"), "--source", "nl", "--target", "en", "--out",
             str(tmp_path / "out")])
    m = bl.read_rows(tmp_path / "out" / "measured.parquet")
    assert len(m) == 1 and m[0]["condition"] == "stukken" and m[0]["pieces"] == 2 and m[0]["paragraphs_kept"]
    assert m[0]["translation"] == "\n\n".join(shard["text_en"]) and m[0]["tokens_out"] == 22
    # one piece missing: the stretch has no condition stukken
    bl.write_parquet([{"id": i, "text": "x", "tokens": 1} for i in shard["id"] + ["s:1:p0-2#c002"]],
                     tmp_path / "seg3.parquet")
    bl.main(["collect", "--docs", str(tmp_path / "docs.parquet"), "--stukken", str(tmp_path / "st"),
             "--pieces", str(tmp_path / "seg3.parquet"), "--source", "nl", "--target", "en", "--out",
             str(tmp_path / "out3")])
    assert bl.read_rows(tmp_path / "out3" / "measured.parquet") == []


def test_throughput_reads_classes_shards_and_the_mem_line(tmp_path):
    (tmp_path / "half0.log").write_text(
        'd8k: 20 segments in 3 shards, 0 done before; shapes {"max_source": 8192, "prefill_lengths": [8320], '
        '"max_decode": 10240, "slots": 2}\n'
        "engine ready in 120 s\n"
        'SHARD {"class": "d8k", "shard": "shard-00000.parquet", "segments": 8, "tokens_in": 60000, '
        '"tokens_out": 50000, "seconds": 400.0, "out_per_s": 125, "all_per_s": 275, "not_ended": 0}\n'
        'MEM {"peak_gib_max": 27.5, "peak_gib": [27.5, 27.4, 27.4, 27.5], "limit_gib": 31.25}\n'
        'd16k: 20 segments in 3 shards, 0 done before; shapes {"max_source": 16384}\n'
        "jaxlib.xla_extension.XlaRuntimeError: RESOURCE_EXHAUSTED: Out of memory\n")
    t = bl.throughput(tmp_path)
    assert t[0]["class"] == "d8k" and t[0]["load_s"] == 120 and t[0]["tokens_out"] == 50000
    assert t[0]["peak_gib"] == 27.5 and not t[0]["oom"] and t[0]["chips"] == 4
    assert t[1]["class"] == "d16k" and t[1]["oom"] and t[1]["chips"] is None


def test_a_bin_that_could_not_be_measured_leaves_out_every_larger_bin():
    ok = {"heel": stat(40, 40, 40, 1000, 10, 3.0), "stukken": stat(40, 40, 40, 1000, 10, 3.0)}
    stats = {512: ok, 1024: ok, 2048: {"heel": stat(0, 0, 0, 0, 0, None), "stukken": stat(40, 40, 40, 1000, 10, 3.0)},
             4096: ok}
    verdict, length = bl.rule(stats)
    assert verdict[4096]["pass"] and not verdict[2048]["measured"]
    assert length == 1024  # bt-length-trial.md 1: "een grotere bak telt dan niet"
    del stats[2048]
    assert bl.rule(stats)[1] == 4096


def test_relative_rule_compares_heel_with_stukken_only():
    stats = {
        # one document fewer done, +1.0 pp and +0.2 exactly: inside; the paragraph count does not decide
        512: {"heel": stat(40, 37, 30, 1000, 20, 3.4), "stukken": stat(40, 38, 36, 1000, 10, 3.2)},
        # two documents fewer done
        1024: {"heel": stat(40, 36, 36, 1000, 10, 3.0), "stukken": stat(40, 38, 36, 1000, 10, 3.0)},
        2048: {"heel": stat(40, 39, 36, 1000, 21, 3.0), "stukken": stat(40, 39, 36, 1000, 10, 3.0)},
        4096: {"heel": stat(40, 40, 36, 1000, 10, 3.201), "stukken": stat(40, 39, 36, 1000, 10, 3.0)},
        8192: {"heel": stat(40, 39, 36, 1000, 10, 3.0), "stukken": stat(40, 39, 36, 1000, 10, 3.0)},
    }
    out, length = bl.relative(stats)
    assert out[512]["pass"] and out[512]["done_diff"] == -1 and out[512]["paragraphs_kept_diff"] == -6
    assert not out[1024]["done"] and out[1024]["omissions"] and out[1024]["qe"]
    assert not out[2048]["omissions"] and not out[4096]["qe"] and out[4096]["done_diff"] == 1
    assert length == 8192
    stats[16384] = {"heel": stat(0, 0, 0, 0, 0, None), "stukken": stat(40, 39, 36, 1000, 10, 3.0)}
    out, length = bl.relative(stats)
    assert not out[16384]["measured"] and length == 8192
    stats[4096]["heel"] = stat(0, 0, 0, 0, 0, None)
    assert bl.relative(stats)[1] == 512  # 4096 not measured: 8192 does not count


def test_merge_keeps_one_of_equal_rows_and_refuses_different_ones(tmp_path):
    def m(i, cond, text):
        return {"id": i, "condition": cond, "translation": text, "ended": True, "done": True, "omitted": 0, "bin": 512}

    def u(i, cand):
        return {"id": i, "src": "x", "cand": cand, "n_source": 1}

    bl.write_parquet([m("a", "heel", "A"), m("a", "stukken", "S")], tmp_path / "m1.parquet")
    bl.write_parquet([m("b", "heel", "B"), m("a", "stukken", "S")], tmp_path / "m2.parquet")
    bl.write_parquet([u("a|heel|u0000", "A"), u("a|stukken|u0000", "S")], tmp_path / "u1.parquet")
    bl.write_parquet([u("b|heel|u0000", "B"), u("a|stukken|u0000", "S")], tmp_path / "u2.parquet")
    bl.main(["merge", "--measured", str(tmp_path / "m1.parquet"), str(tmp_path / "m2.parquet"),
             "--units", str(tmp_path / "u1.parquet"), str(tmp_path / "u2.parquet"), "--out", str(tmp_path / "out")])
    rows = bl.read_rows(tmp_path / "out" / "measured.parquet")
    assert sorted((r["id"], r["condition"]) for r in rows) == [("a", "heel"), ("a", "stukken"), ("b", "heel")]
    assert len(bl.read_rows(tmp_path / "out" / "units.parquet")) == 3
    bl.write_parquet([m("a", "stukken", "other")], tmp_path / "m3.parquet")
    with pytest.raises(SystemExit, match="differs"):
        bl.main(["merge", "--measured", str(tmp_path / "m1.parquet"), str(tmp_path / "m3.parquet"),
                 "--units", str(tmp_path / "u1.parquet"), "--out", str(tmp_path / "out3")])


def test_report_with_relative_and_before_sections(tmp_path):
    import json

    rows, units, scores = [], [], []
    for cond, done in (("heel", True), ("stukken", True)):
        for k in range(2):
            i = f"d{k}"
            rows.append({"id": i, "source": "dpc", "bin": 512, "condition": cond, "done": done, "ended": True,
                         "decode_loop": False, "cut": False, "empty": False, "source_left": False, "chatter": False,
                         "added_formatting": [], "paragraphs_kept": cond == "stukken" or k == 0,
                         "paragraphs_source": 2, "omitted": 0,
                         "omitted_none": 0, "omitted_short": 0, "omitted_long": 0, "added_paragraphs": 0,
                         "word_ratio_nl_en": 1.0, "tokens_source": 400, "tokens_out": 380, "paragraphs_target": 2})
            units.append({"id": f"{i}|{cond}|u0000", "doc": i, "condition": cond, "n_source": 2})
            scores.append({"id": f"{i}|{cond}|u0000", "qe_src_cand": 3.0 if cond == "heel" else 2.9})
    bl.write_parquet(rows, tmp_path / "measured.parquet")
    bl.write_parquet(units, tmp_path / "units.parquet")
    bl.write_parquet(scores, tmp_path / "qe.parquet")
    before = {"title": "Deel A", "setup": {}, "stats": {"512": {"heel": {"docs": 2, "done": 1, "decode_loop": 1,
                                                                        "paragraphs_kept": 1, "omitted": 1,
                                                                        "paragraphs": 4, "qe_mean": 5.0}}}}
    (tmp_path / "a.json").write_text(json.dumps(before))
    bl.main(["report", "--measured", str(tmp_path / "measured.parquet"), "--units", str(tmp_path / "units.parquet"),
             "--qe", str(tmp_path / "qe.parquet"), "--source", "nl", "--target", "en", "--title", "A2", "--relative",
             "--before", str(tmp_path / "a.json"), "--out", str(tmp_path / "r.md")])
    md = (tmp_path / "r.md").read_text()
    r = json.loads((tmp_path / "r.json").read_text())
    # heel keeps the paragraph count in 1 of 2 documents: rule (ii) fails, the relative rule does not look at it
    assert r["L"] is None and not r["verdict"]["512"]["ii"] and r["L_relative"] == 512
    assert "geen deel van de beslisregel" in md and "| 512 | 2 / 2 (+0) | 1 / 2 (-1) |" in md
    assert "| 512 | 2/2 (100,0%) / 1/2 (50,0%) | 0 / 1 | 1/2 (50,0%) / 1/2 (50,0%) |" in md


def test_document_models_fit_their_sources_and_leave_the_others_as_they_were():
    import translate as tr

    for name, factor in (("gemma4-31b-doc", 1.25), ("gemma3-27b-doc", 1.5)):
        model = tr.MODELS[name]
        classes = model["length_classes"]
        assert [c["max_source"] for c in classes.values()] == list(bl.BINS)
        assert set(model["slots"]) == set(classes)
        for c in classes.values():
            assert c["prefill_lengths"][-1] == c["max_source"] + 128
            assert c["max_decode"] == int(factor * c["max_source"])
    assert tr.MODELS["gemma4-31b-doc"]["turn"] == tr.MODELS["gemma4-31b"]["turn"]
    assert tr.MODELS["gemma3-27b-doc"]["prompt"] == tr.MODELS["gemma3-27b"]["prompt"] == "translategemma"
    assert tr.CLASS_ORDER == ["tiny", "small", "medium", "long", "xlong", "xxlong"]


def test_memory_line_of_the_chips():
    class Device:
        def __init__(self, stats):
            self.stats = stats

        def memory_stats(self):
            return self.stats

    gib = 1024 ** 3
    line = blt.memory_line([Device({"peak_bytes_in_use": 27 * gib, "bytes_limit": 31 * gib}),
                            Device({"peak_bytes_in_use": 28 * gib, "bytes_limit": 31 * gib}), Device(None)])
    assert line.startswith("MEM ")
    import json

    d = json.loads(line[4:])
    assert d == {"peak_gib_max": 28.0, "peak_gib": [27.0, 28.0, None], "limit_gib": 31.0}
    assert json.loads(blt.memory_line([Device({})], "XlaRuntimeError: RESOURCE_EXHAUSTED")[4:])["error"].startswith(
        "XlaRuntimeError")


def test_docs_merge_chooses_over_the_sources_separate_runs(tmp_path):
    import json

    def run(d, source, quota_ids, reserve_ids):
        d.mkdir()
        rows = [{"id": i, "source": source, "bin": 512, "tokens": 450, "filled_for": "quota"} for i in quota_ids]
        rows += [{"id": i, "source": source, "bin": 512, "tokens": 450, "filled_for": "reserve"} for i in reserve_ids]
        bl.write_parquet(rows, d / "stretches.parquet")
        (d / "docs.json").write_text(json.dumps({"bins": [512], "per_source": 2, "sources": {source: {}}}))

    run(tmp_path / "a", "raad", ["r0", "r1"], ["r2", "r3"])
    run(tmp_path / "b", "dpc", ["d0"], [])
    bl.main(["docs-merge", "--inputs", str(tmp_path / "a"), str(tmp_path / "b"), "--out", str(tmp_path / "m")])
    chosen = bl.read_rows(tmp_path / "m" / "docs.parquet")
    assert [r["id"] for r in chosen] == ["r0", "r1", "d0", "r2"]
    assert json.loads((tmp_path / "m" / "docs.json").read_text())["chosen"] == {"512": {"raad": 3, "dpc": 1}}


def test_docs_reads_row_groups_and_writes_stretches_per_bin(tmp_path, monkeypatch):
    """docs() end to end on made-up documents: a whitespace tokenizer, a fake HfFileSystem, two row groups"""
    import json
    import types

    import select_passages as sp
    from tokenizers import Tokenizer, models, pre_tokenizers

    tok = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    (tmp_path / "tok").mkdir()
    tok.save(str(tmp_path / "tok" / "tokenizer.json"))
    sentence = "De gemeente heeft het plan voor de nieuwe wijk besproken en goedgekeurd."  # 12 words
    para = " ".join([sentence] * 3)  # 36 tokens
    words = {w.lower().strip(".") for w in sentence.split()}
    (tmp_path / "words.txt").write_text("\n".join(sorted(words)))
    docs_by_group = {0: [{"doc_id": "a", "text": "\n\n".join([para] * 14)}],     # ~530 tokens
                     1: [{"doc_id": "b", "text": "\n\n".join([para] * 30)},     # ~1130 tokens
                         {"doc_id": "c", "text": "kort"}]}
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(HfFileSystem=lambda: None))
    monkeypatch.setattr(sp, "row_groups", lambda source, n, seed, fs: [("f", 0, 1), ("f", 1, 2)])
    monkeypatch.setattr(sp, "read_docs", lambda source, groups, fs: (
        [dict(d, content_type=["article"]) for d in docs_by_group[groups[0][1]]], {"docs_read": 1}))
    bl.main(["docs", "--tokenizer", str(tmp_path / "tok"), "--wordlist", str(tmp_path / "words.txt"),
             "--sources", "raad", "--bins", "512,1024", "--per-source", "1", "--out", str(tmp_path / "out")])
    rows = bl.read_rows(tmp_path / "out" / "docs.parquet")
    assert sorted((r["bin"], r["doc_id"]) for r in rows) == [(512, "a"), (1024, "b")]
    for r in rows:
        assert 0.75 * r["bin"] < r["tokens"] <= r["bin"]
        assert r["text"] == "\n\n".join(r["paragraphs"]) and r["source"] == "raad"
    report = json.loads((tmp_path / "out" / "docs.json").read_text())
    assert report["chosen"] == {"512": {"raad": 1}, "1024": {"raad": 1}}
    assert bl.read_rows(tmp_path / "out" / "stretches.parquet")


def test_smoke_compares_ended_and_output_length_with_the_earlier_run(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    def shard(root, ended, tokens):
        (root / "d1k").mkdir(parents=True)
        pq.write_table(pa.table({"id": [f"d{k}" for k in range(4)], "ended": ended, "tokens_out": tokens}),
                       root / "d1k" / "shard-00000.parquet")

    shard(tmp_path / "before", [True] * 4, [100, 200, 300, 400])
    shard(tmp_path / "now", [True, True, True, False], [105, 190, 330, 1280])
    args = ["smoke", "--now", str(tmp_path / "now"), "--before", str(tmp_path / "before"), "--docs", "4"]
    bl.main(args + ["--ended", "3", "--ratio", "0.9,1.1"])  # ratios 1.05, 0.95, 1.1, 3.2: median 1.075
    with pytest.raises(SystemExit, match="smoke failed"):
        bl.main(args + ["--ended", "4", "--ratio", "0.9,1.1"])
    with pytest.raises(SystemExit, match="smoke failed"):
        bl.main(args + ["--ended", "3", "--ratio", "0.9,1.05"])


def test_complete_needs_every_segment_of_a_class_or_its_out_of_memory_mark(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    # d4k takes 2049-4096 tokens, d8k 4097-8192 (gemma4-31b-doc); the d1k segment is no part of either
    pq.write_table(pa.table({"id": ["a", "b", "c", "d", "e"], "tokens": [900, 3000, 4096, 4097, 8000]}),
                   tmp_path / "segments-w0.parquet")
    root = tmp_path / "all"

    def shard(cls, name, ids):
        (root / cls).mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table({"id": ids}), root / cls / name)

    args = ["complete", "--segments", str(tmp_path / "segments-w0.parquet"), "--model", "gemma4-31b-doc",
            "--root", str(root), "--classes"]
    shard("d4k", "shard-00000.parquet", ["b", "c"])
    bl.main(args + ["d4k"])
    # 14 Nov: d4k whole, d8k not run at all (the VM taken away while it ran)
    with pytest.raises(SystemExit, match=r"d8k: 2 of 2 segments not translated"):
        bl.main(args + ["d4k", "d8k"])
    shard("d8k", "shard-00000.parquet", ["d"])
    with pytest.raises(SystemExit, match=r"d8k: 1 of 2 segments not translated"):
        bl.main(args + ["d4k", "d8k"])
    shard("d8k", "shard-00001.parquet", ["e"])
    bl.main(args + ["d4k", "d8k"])
    (root / "d16k").mkdir()
    (root / "d16k" / "oom.txt").write_text("RESOURCE_EXHAUSTED")
    with pytest.raises(SystemExit, match=r"d16k: no segments"):  # a class asked for that has none is no pass either
        bl.main(args + ["d16k"])
    (root / "d8k" / "shard-00001.parquet").unlink()
    (root / "d8k" / "oom.txt").write_text("RESOURCE_EXHAUSTED")
    bl.main(args + ["d4k", "d8k"])  # out of memory with one slot: marked, the report has the bin as not measured
    with pytest.raises(SystemExit, match="not a class"):
        bl.main(args + ["d3k"])


def test_setup_writes_its_items(tmp_path):
    import json

    bl.main(["setup", "--out", str(tmp_path / "s.json"), "--item", "model", "Gemma 4", "--item", "VM (heel)", "v6e"])
    assert json.loads((tmp_path / "s.json").read_text()) == {"model": "Gemma 4", "VM (heel)": "v6e"}
