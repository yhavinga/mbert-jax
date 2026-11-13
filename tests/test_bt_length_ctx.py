"""bt_length_ctx.py (the length trial's condition ctx, 13 Nov): the pieces of a stretch, the context of a piece within the
prefill, the rounds on one engine (resumed after the last round stored), the copy of context, the terms and questions of
the consistency measure, the answers, the sign test, the rule of ctx against stukken, collect end to end and the blind
material. Made-up data and a stand-in tokenizer; the real Gemma 4 prompt where GEMMA4_TOKENIZER is set. No TPU."""

import collections
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "experiments" / "en-nl-translator", ROOT / "experiments" / "flan-nl"):
    sys.path.insert(0, str(_p))

import bt_length as bl  # noqa: E402
import bt_length_ctx as bc  # noqa: E402
import translate as tr  # noqa: E402

GEMMA4_PREFIX = "<bos><|turn>user\n"
GEMMA4_SUFFIX = "<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"


class WordTokenizer:
    """A token is a word (split on whitespace); Gemma 4's chat turn around a plain-text user turn"""

    def __call__(self, text, add_special_tokens):
        assert not add_special_tokens
        return {"input_ids": list(range(len(text.split())))}

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        text = GEMMA4_PREFIX + messages[0]["content"].strip() + GEMMA4_SUFFIX
        return self(text, False)["input_ids"] if tokenize else text


def codec():
    return bc.prompt_codec(WordTokenizer())


def test_the_ctx_model_and_prompt():
    model = tr.MODELS[bc.MODEL]
    assert model["prompt"] == model["context_prompt"] == "natural-v1-ctx"
    assert tr.PROMPTS["natural-v1-ctx"]["plain"] == tr.PROMPTS["natural-v1"]["plain"]
    assert model["turn"] == tr.MODELS["gemma4-31b-doc"]["turn"] and model["eos_ids"] == [1, 106, 50]
    c = model["length_classes"][bc.CLASS]
    # the prefill of part A's d1k, the decode limit of stukken (d512)
    assert c["prefill_lengths"][-1] == 1152 == tr.MODELS["gemma4-31b-doc"]["length_classes"]["d1k"]["prefill_lengths"][-1]
    assert c["max_decode"] == tr.MODELS["gemma4-31b-doc"]["length_classes"]["d512"]["max_decode"] == 640
    assert set(model["slots"]) == set(model["length_classes"])
    text = codec().with_context.text("Het stuk.", [{"source": "", "translation": "The piece before."}])
    assert text.startswith(GEMMA4_PREFIX + "Translate the following Dutch text into natural, idiomatic English")
    assert "<previous_translation>\nThe piece before.\n</previous_translation>" in text
    assert text.endswith("The Dutch text to translate:\n\n\nHet stuk." + GEMMA4_SUFFIX)
    # without context: natural-v1's prompt, as the condition stukken had it
    assert codec().plain.text("Het stuk.") == GEMMA4_PREFIX + tr.user_text(tr.PROMPTS["natural-v1"], "Het stuk.", "nl",
                                                                           "en") + GEMMA4_SUFFIX


@pytest.mark.skipif(not os.environ.get("GEMMA4_TOKENIZER"), reason="GEMMA4_TOKENIZER not set")
def test_real_gemma4_ctx_prompt_matches_its_chat_template():
    from transformers import AutoTokenizer

    c = bc.prompt_codec(AutoTokenizer.from_pretrained(os.environ["GEMMA4_TOKENIZER"]))
    c.check(["Een zin.", "Nog een zin."], [[{"source": "", "translation": "A sentence.\n\nTwo."}]] * 2)
    c.check(["Een zin."], None)
    q = bc.question_codec(AutoTokenizer.from_pretrained(os.environ["GEMMA4_TOKENIZER"]))
    q.check([bc.QUESTION.format(term="Tweede Kamer", dutch="De Tweede Kamer stemt.", english="The House votes.")])


def doc(i, paragraphs, bin_=2048, source="dpc"):
    return {"id": f"{source}:{i}:p0-{len(paragraphs)}", "source": source, "doc_id": str(i), "bin": bin_,
            "tokens": 1600, "paragraphs": paragraphs}


def test_stretches_are_the_pieces_in_order_and_must_join_to_the_stretch():
    d = doc(1, ["Eerste alinea.", "Tweede alinea.", "Derde alinea."])
    segs = [{"id": d["id"] + "#c001", "text": "Derde alinea.", "tokens": 2},
            {"id": d["id"] + "#c000", "text": "Eerste alinea.\n\nTweede alinea.", "tokens": 4},
            {"id": "other:9:p0-1#c000", "text": "x", "tokens": 1}]
    s = bc.stretches_of(segs, {d["id"]: d})
    assert [r["text"] for r in s[d["id"]]] == ["Eerste alinea.\n\nTweede alinea.", "Derde alinea."]
    with pytest.raises(SystemExit, match="not 0..n-1"):
        bc.stretches_of(segs[:1], {d["id"]: d})
    bad = [dict(segs[0], text="Iets anders."), segs[1]]
    with pytest.raises(SystemExit, match="joined are not the stretch"):
        bc.stretches_of(bad, {d["id"]: d})
    assert bc.piece_index("raad:x:p1-3#c012") == ("raad:x:p1-3", 12)


def test_context_is_whole_paragraphs_from_the_end_that_fit():
    c = codec()
    text = "een twee drie"
    base = len(c.encode(text, [{"source": "", "translation": ""}]))
    previous = "a b c d e.\n\nf g h.\n\ni j k l."
    full = bc.context_for(c.encode, text, previous, base + 100)
    assert full["context"][0]["translation"] == previous and full["full"] and full["paragraphs"] == 3
    assert full["tokens"] == 12
    two = bc.context_for(c.encode, text, previous, base + 7)  # "f g h." + "i j k l." = 7 words
    assert two["context"][0]["translation"] == "f g h.\n\ni j k l." and two["paragraphs"] == 2 and not two["full"]
    cut = bc.context_for(c.encode, text, previous, base + 2)  # not even the last paragraph: its last words
    assert cut["context"][0]["translation"] == "k l." and cut["cut"] and cut["paragraphs"] == 0
    assert bc.context_for(c.encode, text, "", base + 100)["context"] is None
    with pytest.raises(RuntimeError, match="no room"):
        bc.context_for(c.encode, text, previous, base - 1)


def plan_rows(stretches):
    return [{"id": f"{d}#c{k:03d}", "doc": d, "k": k, "pieces": len(ps), "bin": 2048, "source": "dpc", "text": t,
             "tokens": len(t.split())} for d, ps in stretches.items() for k, t in enumerate(ps)]


def test_rounds_translate_piece_k_of_every_stretch_with_the_english_of_k_minus_one():
    c = codec()
    stretches = bc.stretches_of_plan(plan_rows({"a": ["a nul", "a een", "a twee"], "b": ["b nul", "b een"]}))
    seen, stored = [], {}
    tok = WordTokenizer()

    def encode(text, context):
        seen.append((text, context[0]["translation"] if context else None))
        return c.encode(text, context)

    def decode(prompts):
        k = len(stored)
        texts = [f"EN {d} {k}" for d in ("a", "b") if k < len(stretches[d])]
        return [(t.split(), True) for t in texts]

    def store(k, rows, seconds):
        stored[k] = rows

    english = bc.run_rounds(stretches, 3, {}, encode, decode, " ".join, store, 10_000, log=lambda line: None)
    assert english == {"a": {0: "EN a 0", 1: "EN a 1", 2: "EN a 2"}, "b": {0: "EN b 0", 1: "EN b 1"}}
    assert stored[0]["id"] == ["a#c000", "b#c000"] and stored[2]["id"] == ["a#c002"]
    assert stored[1]["context_full"] == [True, True] and stored[0]["context_tokens"] == [0, 0]
    assert ("a een", "EN a 0") in seen and ("b een", "EN b 0") in seen and ("a nul", None) in seen
    assert tok  # the stand-in tokenizer is the codec's

    # resumed: rounds 0 and 1 stored before; round 2 takes the English of round 1 from them
    seen.clear()
    done = {k: [dict(zip(stored[k], v)) for v in zip(*stored[k].values())] for k in (0, 1)}
    stored2 = {}
    english = bc.run_rounds(stretches, 3, done, encode, lambda p: [("EN a again".split(), True)], " ".join,
                            lambda k, rows, s: stored2.__setitem__(k, rows), 10_000, log=lambda line: None)
    assert list(stored2) == [2] and english["a"][2] == "EN a again"
    assert any(text == "a twee" and ctx == "EN a 1" for text, ctx in seen)
    with pytest.raises(SystemExit, match="not a prefix"):
        bc.run_rounds(stretches, 3, {1: done[1]}, encode, decode, " ".join, store, 10_000, log=lambda line: None)


def test_a_prompt_over_the_prefill_is_an_error():
    c = codec()
    stretches = bc.stretches_of_plan(plan_rows({"a": ["een twee drie vier vijf zes"]}))
    with pytest.raises(RuntimeError, match="more than"):
        bc.run_rounds(stretches, 1, {}, c.encode, lambda p: [([], True)], " ".join, lambda *a: None, 10,
                      log=lambda line: None)


def test_the_first_stretches_of_two_pieces_or_more_for_the_smoke_test():
    s = bc.stretches_of_plan(plan_rows({"c": ["x", "y"], "a": ["x"], "b": ["x", "y", "z"], "d": ["x", "y"]}), 2)
    assert list(s) == ["b", "c"]


def test_copy_of_context_is_the_share_of_the_outputs_six_grams_in_the_piece_before():
    before = "The municipality granted the permit for the new building on the square last year."
    fresh = "The council then discussed the budget and the plans of the alderman for next year in detail."
    assert bc.copy_share(fresh, before) == 0.0
    copied = before + " " + fresh
    assert 0.3 < bc.copy_share(copied, before) < 0.6
    assert bc.copy_share("Too short.", before) == 0.0
    assert bc.tag_leak("<previous_translation>\nx") and bc.tag_leak("The Dutch text to translate: x")
    assert not bc.tag_leak("A normal translation.")
    rows = bc.piece_rows({"id": "d", "bin": 2048, "source": "dpc"}, "ctx",
                         [{"text": before, "ended": True, "tokens_out": 10, "context_tokens": 0},
                          {"text": copied, "ended": True, "tokens_out": 30, "context_tokens": 12}], ["nl0", "nl1"])
    assert rows[0]["copy_share"] is None and not rows[0]["copy"] and rows[1]["copy"]


def test_candidates_are_names_runs_abbreviations_and_lowercase_words():
    p = ("De Raad van State besliste dat de gemeente Amsterdam het bestemmingsplan moet wijzigen. "
         "Volgens de VVD en de heer Van der Laan is U niet bevoegd. In Utrecht gebeurde hetzelfde.")
    got = {(t, k) for t, k, _ in bc.candidates(p)}
    assert ("Raad van State", "name") in got
    assert ("Amsterdam", "name") in got and ("VVD", "name") in got and ("Van der Laan", "name") in got
    assert ("Utrecht", "name") in got  # "In" starts the sentence and is dropped
    assert ("bestemmingsplan", "term") in got
    assert not any(t in ("De", "Volgens", "U", "In") for t, _ in got)


def test_terms_are_in_two_pieces_or_more_and_asked_in_at_most_eight():
    pieces = [f"De gemeente Haaksbergen wijzigt het bestemmingsplan. Het bestemmingsplan blijft. Stuk {k}." for k in
              range(12)] + ["Alleen hier staat Enschede."]
    terms = bc.terms_of("d", pieces, collections.Counter({"bestemmingsplan": 1}), 100)
    by = {t["term"]: t for t in terms}
    assert set(by) >= {"Haaksbergen", "bestemmingsplan"} and "Enschede" not in by
    assert by["Haaksbergen"]["pieces"] == list(range(12)) and by["Haaksbergen"]["asked"][0] == 0
    assert len(by["bestemmingsplan"]["asked"]) == bc.MAX_QUERY_PIECES and by["bestemmingsplan"]["asked"][-1] == 11
    # a lowercase word in too many documents is no term
    assert "bestemmingsplan" not in {t["term"] for t in bc.terms_of("d", pieces, collections.Counter({"bestemmingsplan": 41}), 100)}
    assert bc.spread(list(range(10)), 3) == [0, 4, 9] and bc.spread([1, 2], 8) == [1, 2]
    assert bc.sentence_with(pieces[0], "Haaksbergen", "name") == (
        "De gemeente Haaksbergen wijzigt het bestemmingsplan.", "sentence")


def test_answers_are_compared_without_case_articles_or_quotes_and_must_be_in_the_english():
    english = "The Council of State ruled that the zoning plan must change."
    assert bc.parse_answer("the Council of State", english) == ("valid", "council of state")
    assert bc.parse_answer('"zoning plan".\nmore', english) == ("valid", "zoning plan")
    assert bc.parse_answer("NONE", english) == ("none", None)
    assert bc.parse_answer("land-use plan", english) == ("invalid", "land-use plan")
    assert bc.parse_answer("  ", english) == ("empty", None)
    assert bc.normalize("The Court's") == "court"


def test_sign_test_is_one_sided_and_exact():
    assert bc.sign_test(0, 0) == 1.0
    assert bc.sign_test(5, 0) == pytest.approx(1 / 32)
    assert bc.sign_test(6, 1) == pytest.approx(8 / 128)
    assert bc.sign_test(1, 6) == pytest.approx(127 / 128)


def consistency_fixture():
    terms = [{"doc": "d", "term_index": i, "bin": 2048, "kind": "name" if i < 2 else "term", "term": f"t{i}"}
             for i in range(4)]
    queries, answers = [], {}
    # term 0: stukken two renderings, ctx one; term 1: both consistent; term 2: ctx worse; term 3: one valid answer
    plan = {0: (["A", "B"], ["A", "A"]), 1: (["X", "X"], ["X", "X"]), 2: (["P", "P"], ["P", "Q"]),
            3: (["M", "NONE"], ["M", "NONE"])}
    for i, (st, cx) in plan.items():
        for cond, ans in (("stukken", st), ("ctx", cx)):
            for k, a in enumerate(ans):
                qid = f"d|{cond}|t{i:02d}|c{k:03d}"
                queries.append({"id": qid, "doc": "d", "condition": cond, "term_index": i, "english": "A B X P Q M"})
                answers[qid] = a
    return terms, queries, answers


def test_consistency_per_term_and_paired_over_the_conditions():
    terms, queries, answers = consistency_fixture()
    per, status = bc.consistency(terms, queries, answers)
    assert per[("d", 0)]["stukken"]["consistent"] is False and per[("d", 0)]["ctx"]["consistent"] is True
    assert not per[("d", 3)]["ctx"]["measurable"]
    assert status["ctx"]["none"] == 1 and status["ctx"]["valid"] == 7
    s = bc.consistency_stats(terms, per, [2048])[2048]
    assert (s["stukken"]["measurable"], s["stukken"]["consistent"]) == (3, 2)
    assert (s["ctx"]["measurable"], s["ctx"]["consistent"]) == (3, 2)
    assert s["paired"] == {"terms": 3, "ctx_better": 1, "ctx_worse": 1, "p": 0.75, "share_stukken": 2 / 3,
                           "share_ctx": 2 / 3}
    assert bc.consistency_stats(terms, per, [2048], ("name",))[2048]["ctx"]["consistent"] == 2


def bin_row(done, omitted, qe, docs=40, paragraphs=1000):
    return {"docs": docs, "done": done, "omitted": omitted, "paragraphs": paragraphs, "qe_mean": qe}


def cons_row(share_p, share_c, better, worse):
    return {"stukken": {"share": share_p}, "ctx": {"share": share_c},
            "paired": {"ctx_better": better, "ctx_worse": worse, "p": bc.sign_test(better, worse), "terms": 100}}


def copy_row(p, c, n=200):
    return {"stukken": {"copy": p, "pieces_after_first": n}, "ctx": {"copy": c, "pieces_after_first": n}}


def test_rule_needs_clearly_better_consistency_and_nothing_worse():
    stats = {2048: {"stukken": bin_row(38, 100, 3.30), "ctx": bin_row(37, 110, 3.50)}}
    good = bc.rule(stats, {2048: cons_row(0.60, 0.70, 12, 2)}, {2048: copy_row(2, 4)})[2048]
    # inconsistency 30% <= 0.75 x 40%; 12 against 2 (p < 0.01); 37 >= 38 - 1; +1.0 pp; +0.2; +1.0 pp: all at the edge
    assert good["pass"] and good["i_ratio"] and good["i_sign"] and good["ii"] and good["iii"] and good["iv"] and good["v"]
    assert not bc.rule(stats, {2048: cons_row(0.60, 0.65, 12, 2)}, {2048: copy_row(2, 4)})[2048]["i_ratio"]
    assert not bc.rule(stats, {2048: cons_row(0.60, 0.70, 6, 2)}, {2048: copy_row(2, 4)})[2048]["i_sign"]
    assert not bc.rule(stats, {2048: cons_row(0.60, 0.70, 12, 2)}, {2048: copy_row(2, 5)})[2048]["v"]
    worse = {2048: {"stukken": bin_row(38, 100, 3.30), "ctx": bin_row(36, 111, 3.51)}}
    v = bc.rule(worse, {2048: cons_row(0.60, 0.70, 12, 2)}, {2048: copy_row(2, 4)})[2048]
    assert not v["ii"] and not v["iii"] and not v["iv"] and not v["pass"]
    # nothing to improve: stukken fully consistent
    assert not bc.rule(stats, {2048: cons_row(1.0, 1.0, 0, 0)}, {2048: copy_row(0, 0)})[2048]["i"]
    assert not bc.rule({2048: {"stukken": bin_row(38, 100, 3.3), "ctx": None}}, {}, {})[2048]["measured"]


def test_costs_read_rounds_extract_and_the_mem_line(tmp_path):
    (tmp_path / "ctx_tr-w0.log").write_text(
        'c512: 40 stretches, 629 pieces, 39 rounds, 0 done before; shapes {"max_source": 512, "slots": 20}\n'
        "engine ready in 150 s\n"
        'ROUND {"round": 0, "prompts": 40, "tokens_in": 20000, "tokens_out": 15000, "seconds": 60.0}\n'
        'ROUND {"round": 1, "prompts": 40, "tokens_in": 40000, "tokens_out": 15000, "seconds": 40.0}\n'
        'MEM {"peak_gib_max": 27.9}\n')
    (tmp_path / "ctx_xq-w0.log").write_text(
        'xq: 900 questions; shapes {"slots": 24}\nengine ready in 90 s\n'
        'EXTRACT {"questions": 900, "asked": 899, "too_long": 1, "seconds": 300.0, "tokens_in": 600000, '
        '"not_ended": 0}\nMEM {"peak_gib_max": 25.0}\n')
    t = {x["log"]: x for x in bc.costs(tmp_path)}
    r = t["ctx_tr-w0.log"]
    assert (r["load_s"], r["rounds"], r["prompts"], r["seconds"], r["peak_gib"], r["slots"]) == (150, 2, 80, 100.0,
                                                                                                 27.9, 20)
    assert t["ctx_xq-w0.log"]["extract"]["asked"] == 899 and t["ctx_xq-w0.log"]["slots"] == 24


def test_collect_measures_both_conditions_and_asks_about_the_terms(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    nl = [["De gemeente Haaksbergen wijzigt het bestemmingsplan.", "Het bestemmingsplan wordt vastgesteld."],
          ["De gemeente Haaksbergen stelt het bestemmingsplan vast.", "Daarna volgt het bestemmingsplan opnieuw."]]
    d = doc(1, [p for piece in nl for p in piece], bin_=2048, source="raad")
    bl.write_parquet([d], tmp_path / "docs.parquet")
    ids = [d["id"] + "#c000", d["id"] + "#c001"]
    plan = [{"id": i, "doc": d["id"], "k": k, "pieces": 2, "bin": 2048, "source": "raad", "text": bl.join(nl[k]),
             "tokens": 10} for k, i in enumerate(ids)]
    bl.write_parquet(plan, tmp_path / "plan-w0.parquet")
    en_st = ["The municipality of Haaksbergen changes the zoning plan.\n\nThe zoning plan is adopted.",
             "The municipality of Haaksbergen adopts the land-use plan.\n\nThen the land-use plan follows again."]
    en_cx = [en_st[0], "The municipality of Haaksbergen adopts the zoning plan.\n\nThen the zoning plan follows again."]
    (tmp_path / "st" / "w0" / "d512").mkdir(parents=True)
    pq.write_table(pa.table({"id": ids, "text_en": en_st, "ended": [True, True], "tokens_in": [90, 90],
                             "tokens_out": [12, 13]}), tmp_path / "st" / "w0" / "d512" / "shard-00000.parquet")
    (tmp_path / "cx" / "w0").mkdir(parents=True)
    for k in (0, 1):
        pq.write_table(pa.table({"id": [ids[k]], "doc": [d["id"]], "k": [k], "text_en": [en_cx[k]], "ended": [True],
                                 "tokens_in": [90 + 20 * k], "tokens_out": [12], "context_tokens": [20 * k],
                                 "context_paragraphs": [2 * k], "context_full": [k == 1], "context_cut": [False]}),
                       tmp_path / "cx" / "w0" / bc.round_name(k))
    row, _, units = bl.measure_doc(d, bl.join(en_st), True, "nl", "en")
    bl.write_parquet([{"id": d["id"], "source": "raad", "bin": 2048, "condition": "stukken", "tokens_source": 1600,
                       "tokens_in": 180, "tokens_out": 25, "translation": bl.join(en_st), "pieces": 2,
                       "alignment": json.dumps(units), **row}], tmp_path / "measured-a.parquet")
    bc.main(["collect", "--docs", str(tmp_path / "docs.parquet"), "--plan", str(tmp_path / "plan-w0.parquet"),
             "--stukken", str(tmp_path / "st"), "--ctx", str(tmp_path / "cx"), "--measured-a",
             str(tmp_path / "measured-a.parquet"), "--workers", "2", "--out", str(tmp_path / "out")])
    out = tmp_path / "out"
    m = {r["condition"]: r for r in bl.read_rows(out / "measured.parquet")}
    assert set(m) == {"stukken", "ctx"} and m["ctx"]["translation"] == bl.join(en_cx)
    assert m["ctx"]["pieces_after_first"] == 1 and m["ctx"]["copy_pieces"] == 0
    assert {u["condition"] for u in bl.read_rows(out / "units.parquet")} == {"ctx"}
    terms = {t["term"] for t in bl.read_rows(out / "terms.parquet")}
    assert "Haaksbergen" in terms
    q = [r for w in (0, 1) for r in bl.read_rows(out / f"queries-w{w}.parquet")]
    assert {r["condition"] for r in q} == {"stukken", "ctx"} and all(r["english"] for r in q)
    # a piece without a translation stops collect
    (tmp_path / "cx" / "w0" / bc.round_name(1)).unlink()
    with pytest.raises(SystemExit, match="without a translation in ctx"):
        bc.main(["collect", "--docs", str(tmp_path / "docs.parquet"), "--plan", str(tmp_path / "plan-w0.parquet"),
                 "--stukken", str(tmp_path / "st"), "--ctx", str(tmp_path / "cx"), "--measured-a",
                 str(tmp_path / "measured-a.parquet"), "--out", str(tmp_path / "out2")])


def test_blind_material_is_stratified_with_a_key_apart(tmp_path):
    docs, pieces = [], []
    mark = {"stukken": "P", "ctx": "Q"}
    for i in range(30):
        source = bl.SOURCE_ORDER[i % 5]
        d = doc(i, ["x"], bin_=4096, source=source)
        docs.append(d)
        for cond in bc.CONDITIONS:
            for k in range(3):
                pieces.append({"doc": d["id"], "bin": 4096, "source": source, "condition": cond, "k": k,
                               "text_nl": f"NL {i} {k}", "text_en": f"EN {mark[cond]} {i} {k}"})
    bl.write_parquet(docs, tmp_path / "docs.parquet")
    (tmp_path / "c").mkdir()
    bl.write_parquet(pieces, tmp_path / "c" / "pieces.parquet")
    bc.main(["blind", "--collect", str(tmp_path / "c"), "--docs", str(tmp_path / "docs.parquet"), "--out",
             str(tmp_path / "blind")])
    key = json.loads((tmp_path / "blind-key-4096.json").read_text())
    assert len(key) == 20 and {v["source"] for v in key.values()} == set(bl.SOURCE_ORDER)
    assert all(sorted([v["A"], v["B"]]) == sorted(bc.CONDITIONS) for v in key.values())
    text = (tmp_path / "blind" / "doc-01.md").read_text()
    assert "— stukgrens 2 —" in text and "stukken" not in text and "ctx" not in text
    a = key["doc-01"]["A"]
    assert f"EN {mark[a]} " in text.split("## Vertaling A")[1].split("## Vertaling B")[0]
    assert (tmp_path / "blind" / "rubric.md").exists() and not (tmp_path / "blind" / "blind-key-4096.json").exists()
