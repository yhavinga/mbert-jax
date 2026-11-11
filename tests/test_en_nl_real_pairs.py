"""experiments/en-nl-translator/real_pairs.py: the real translation pairs of arm R2 (Global Voices, Europarl-direct after
2003): the alignment of an article's paragraphs, the speaker's role before a Europarl speech, the hard checks, the
held-out articles of the blind assessment, the candidates and the join of MetricX-QE. No network (gv-fetch runs on a
VM); run with --noconftest where jax and flax are not installed."""

import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "en-nl-translator"))

import real_pairs as rp  # noqa: E402


def test_a_json_line_may_hold_a_unicode_line_separator(tmp_path):
    path = tmp_path / "x.jsonl"
    path.write_text(json.dumps({"t": "a b"}, ensure_ascii=False) + "\n", encoding="utf-8")
    assert rp.read_jsonl(path) == [{"t": "a b"}]


def test_the_english_version_and_the_source_date_of_a_page():
    html = ("<span class='translation-language post-translation-it'><a href='https://it.globalvoices.org/x/'>It</a>"
            "</span>, <span class='translation-language post-translation-en'><a href='https://globalvoices.org/2026/10/"
            "23/a-post/' title='Lees dit artikel in het English'>English</a></span></div> <span class='post-date "
            "post-source-date'>Oorspronkelijk geplaatst 24 oktober 2026</span>")
    assert rp.EN_LINK.search(html)[1] == "https://globalvoices.org/2026/10/23/a-post/"
    assert rp.SOURCE_DATE.search(html)[1] == "Oorspronkelijk geplaatst 24 oktober 2026"
    assert rp.SOURCE_DATE.search("<span class='post-date'>Posted on</span>") is None


def para(text, links=()):
    return (text, list(links))


def test_the_alignment_skips_a_paragraph_the_translation_left_out_and_a_note_it_added():
    en = [para("The government announced new rules for the press on Monday, " * 2, ["https://a.org/x"]),
          para("Critics said the rules would silence independent journalists across the country. " * 2),
          para("In 2024, 35 journalists were arrested, according to the committee. " * 2, ["https://cpj.org/r"]),
          para("Read more: our coverage of the elections.")]
    nl = [para("De regering kondigde maandag nieuwe regels voor de pers aan, " * 2, ["http://www.a.org/x/"]),
          para("Noot van de vertaler: dit artikel verscheen eerder in het Engels en werd ingekort."),
          para("Volgens het comité werden in 2024 35 journalisten gearresteerd. " * 2, ["https://cpj.org/r#top"])]
    pairs = rp.align(en, nl)
    assert [(i, j) for i, j, _ in pairs] == [(0, 0), (2, 2)]
    assert all(score > 0 for *_, score in pairs)


def test_the_alignment_pairs_paragraphs_in_order_when_they_correspond():
    en = [para(f"Paragraph number {k} of the article, with some words to give it length {k}.") for k in range(6)]
    nl = [para(f"Alinea nummer {k} van het artikel, met wat woorden om het lengte te geven {k}.") for k in range(6)]
    assert [(i, j) for i, j, _ in rp.align(en, nl)] == [(k, k) for k in range(6)]
    assert rp.align([], nl) == [] and rp.align(en, []) == []


def test_the_speakers_role_before_a_speech_goes():
    assert rp.strip_role("lid van de Commissie . – Mijnheer de Voorzitter, ik dank u.") == \
        "Mijnheer de Voorzitter, ik dank u."
    assert rp.strip_role("namens de PSE-Fractie. – Mevrouw de Voorzitter, het verslag.") == \
        "Mevrouw de Voorzitter, het verslag."
    assert rp.strip_role("Draftsman of the opinion of the Committee on Legal Affairs . − Mr President, I agree.") == \
        "Mr President, I agree."
    # a speech without a role, and a dash inside a speech, stay as they are
    for text in ("Mijnheer de Voorzitter, ik dank u.", "Mr President, this is not so. And – as I said – no.",
                 "Ik zeg dit: ten eerste. – Ten tweede, de rest."):
        assert rp.strip_role(text) == text


def test_the_hard_checks_of_a_real_pair():
    en = "The Commission has accepted those amendments today in full."
    assert rp.real_fails(en, "De Commissie heeft die amendementen vandaag volledig aanvaard.") == []
    assert rp.real_fails("Thank you.", "Dank u.") == ["short"]
    assert rp.real_fails(en, en) == ["same"]
    assert "english_left" in rp.real_fails(
        "We should not forget that the people in the region have been waiting for the reforms for years.",
        "We moeten niet vergeten that the people in the region have been waiting for the reforms for years.")
    assert rp.real_fails(en, "De Commissie aanvaardt het.") == ["len_ratio"]


def write(path, rows):
    pq.write_table(pa.Table.from_pylist(rows), path)


def gv_row(en_post, k, en, nl, date="2019-03-01T10:00:00", link=None):
    return {"id": f"gv:{en_post}:9{en_post}:{k}:{k}", "source": "globalvoices", "en": en, "nl": nl, "date": date,
            "year": int(date[:4]),
            "meta": json.dumps({"en_link": link or f"https://globalvoices.org/2019/03/01/p{en_post}/",
                                "en_post": en_post})}


def test_candidates_hold_out_the_blind_articles_and_the_pool_and_split_for_the_workers(tmp_path):
    en = "The minister announced on Monday that the new law would enter into force next year."
    nl = "De minister kondigde maandag aan dat de nieuwe wet volgend jaar in werking treedt."
    gv = [gv_row(1, 0, en, nl), gv_row(2, 0, en + " A.", nl + " A."),
          gv_row(3, 0, en + " B.", nl + " B.", date="2025-05-05T00:00:00"),
          gv_row(4, 0, en + " C.", nl + " C.", link="https://globalvoices.org/2025/01/05/held/"),
          gv_row(5, 0, "Thank you.", "Dank u.")]
    ep = [{"id": f"europarl:2005:{k}", "source": "europarl", "en": en + f" {k}.", "nl": nl + f" {k}.",
           "date": "2005-01-01", "year": 2005, "meta": "{}"} for k in range(5)]
    write(tmp_path / "gv.parquet", gv)
    write(tmp_path / "ep.parquet", ep)
    (tmp_path / "blind.jsonl").write_text(json.dumps({"id": "gv:2:1", "set": "idiom", "source": "x",
                                                      "meta": {"url": "https://globalvoices.org/x/"}}) + "\n")
    (tmp_path / "idioms.jsonl").write_text(json.dumps({"id": "gv:77:4", "set": "idiom", "source": "y",
                                                       "meta": {"url": "https://globalvoices.org/2025/01/05/held/"}})
                                           + "\n")
    args = type("A", (), {"gv": str(tmp_path / "gv.parquet"), "europarl": str(tmp_path / "ep.parquet"),
                          "blind": str(tmp_path / "blind.jsonl"), "idioms": str(tmp_path / "idioms.jsonl"),
                          "workers": 2, "out": str(tmp_path / "out")})
    rp.candidates(args)
    rows = {r["id"]: r for r in pq.read_table(tmp_path / "out" / "real.parquet").to_pylist()}
    assert rows["gv:1:91:0:0"]["fails"] == []
    assert rows["gv:2:92:0:0"]["fails"] == ["blind_article"]       # its post id is in the blind set
    assert rows["gv:3:93:0:0"]["fails"] == ["blind_pool_year"]     # 2025: the third subagent's pool
    assert rows["gv:4:94:0:0"]["fails"] == ["blind_article"]       # its link is in the idiom set
    assert rows["gv:5:95:0:0"]["fails"] == ["short"]
    parts = [pq.read_table(tmp_path / "out" / f"metricx-in-w{w}.parquet").to_pylist() for w in range(2)]
    assert sorted(r["id"] for p in parts for r in p) == sorted(["gv:1:91:0:0"] + [r["id"] for r in ep])
    assert set(parts[0][0]) == {"id", "en", "nl"}
    report = json.loads((tmp_path / "out" / "candidates.json").read_text())
    assert report["to_score"] == 6 and report["stats"]["globalvoices/first/blind_article"] == 2

    # join-qe: every pair that passed gets its score; an unknown id or a missing score fails
    qe = [{"id": r["id"], "qe_en_nl": 1.5} for p in parts for r in p]
    write(tmp_path / "qe.parquet", qe)
    join = type("J", (), {"real": str(tmp_path / "out" / "real.parquet"), "qe": [str(tmp_path / "qe.parquet")],
                          "out": str(tmp_path / "real-qe.parquet")})
    rp.join_qe(join)
    joined = {r["id"]: r for r in pq.read_table(tmp_path / "real-qe.parquet").to_pylist()}
    assert joined["gv:1:91:0:0"]["qe_en_nl"] == 1.5 and joined["gv:5:95:0:0"]["qe_en_nl"] is None
    write(tmp_path / "qe.parquet", qe[1:])
    with pytest.raises(SystemExit, match="no MetricX-QE"):
        rp.join_qe(join)
    write(tmp_path / "qe.parquet", qe + [{"id": "gv:9:9:9:9", "qe_en_nl": 1.0}])
    with pytest.raises(SystemExit, match="not in"):
        rp.join_qe(join)


def test_europarl_keeps_the_years_after_the_evaluations_and_strips_the_role(tmp_path):
    rows = [{"id": f"{y}0101:1-1:{p}:2", "date": f"{y}-01-01", "year": y, "intervention_id": "1-1", "paragraph": p,
             "paragraphs": 2, "speaker": "A", "speaker_id": "1", "nationality": "x", "mode": "spoken", "role": "mep",
             "sl": "en", "en": "Mr President, we agree with the report.",
             "nl": ("lid van de Commissie . – " if p == 0 else "") + "Mijnheer de Voorzitter, wij steunen het verslag.",
             "en_words": 7, "nl_words": 7} for y in (2003, 2004, 2011) for p in (0, 1)]
    (tmp_path / "motra").mkdir()
    write(tmp_path / "motra" / "shard-00000.parquet", rows)
    args = type("A", (), {"motra": str(tmp_path / "motra"), "out": str(tmp_path / "ep.parquet")})
    rp.europarl(args)
    out = pq.read_table(tmp_path / "ep.parquet").to_pylist()
    assert sorted({r["year"] for r in out}) == [2004, 2011] and len(out) == 4
    assert all(r["nl"] == "Mijnheer de Voorzitter, wij steunen het verslag." for r in out)
    assert all(r["id"].startswith("europarl:") and r["source"] == "europarl" for r in out)


def test_paragraphs_with_their_links():
    pytest.importorskip("bs4")
    import blind

    html = ("<p>First <a href='https://a.org/x'>link</a> and <a href='https://b.org'>another</a>.</p>"
            "<blockquote><p>A quote.</p></blockquote><p>Second.</p>")
    assert blind.paragraphs(html) == ["First link and another.", "Second."]
    assert blind.paragraphs(html, links=True) == [("First link and another.", ["https://a.org/x", "https://b.org"]),
                                                  ("Second.", [])]


def test_global_voices_language_markers_go():
    assert rp.strip_lang_marks("Ze schreef [en] op haar blog [es, fr]:") == "Ze schreef op haar blog:"
    assert rp.strip_lang_marks("Zie het rapport [en – alle links] en de video [EN].") == "Zie het rapport en de video."
    assert rp.strip_lang_marks("Een symbool [Engels – alle links] van de strijd.") == "Een symbool van de strijd."
    assert rp.strip_lang_marks("He said [MKD] that…") == "He said that…"
    assert rp.strip_lang_marks("Het rapport [en, alle links] en [-en] meer.") == "Het rapport en meer."
    # an omission mark, a one-letter bracket and a longer bracket stay
    for text in ("Hij zei [...] dat [s] het [some sensitive tuna] was.", "Zie [1] en […]."):
        assert rp.strip_lang_marks(text) == text
