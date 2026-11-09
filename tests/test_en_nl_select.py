"""experiments/en-nl-translator/select_passages.py on made-up documents: paragraphs, prose, speaker labels, names the
PII filter swapped, idioms in any word order, the split per document, duplicates and the shares. No download, no
model: a token is a word here. Run with --noconftest where the conftest's imports (jax, flax) are not installed."""

import collections
import json
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "en-nl-translator"))

import select_passages as sp  # noqa: E402


def test_sentences_keep_their_quotes_and_do_not_end_at_abbreviations():
    text = 'Hij zei: "Dat is het paard achter de wagen spannen." Mr. Jansen knikte. Zie art. 7 van de wet. Klaar!'
    assert sp.split_sentences(text) == ['Hij zei: "Dat is het paard achter de wagen spannen."',
                                        "Mr. Jansen knikte.", "Zie art. 7 van de wet.", "Klaar!"]
    assert sp.split_sentences("J. de Vries kwam. Hij ging.") == ["J. de Vries kwam.", "Hij ging."]


@pytest.mark.parametrize("paragraph, prose", [
    ("Ja. Dat is precies wat ik de minister wil vragen.", True),
    ("Voorzitter. Ik dank de minister voor haar antwoorden.", True),
    ("- de motie over een bevolkingsonderzoek (26956, nr. 77);", False),
    ("a. de aanvraag is te laat ingediend door de verzoeker.", False),
    ("Onderwerp: Vissenbuurt in de wijk Malburgen", False),
    ("| kolom | kolom | kolom | kolom | kolom |.", False),
    ("GEMEENTERAAD ARNHEM BESLUIT OVER DE GYMZAAL NU.", False),
    ("Kijk op https://www.rijksoverheid.nl voor meer informatie hierover.", False),
    ("Te kort.", False),
])
def test_prose(paragraph, prose):
    assert sp.is_prose(paragraph) is prose


def test_council_paragraphs_are_unwrapped_and_hyphens_kept_only_in_compounds():
    text = "Wij waarderen uw inzet om in ge-\nsprek te gaan over sloop-\nnieuwbouw van de\nwoningen.\n\nTweede alinea."
    paras = sp.paragraphs_of("raad", text, {"gesprek"})
    assert paras == ["Wij waarderen uw inzet om in gesprek te gaan over sloop-nieuwbouw van de woningen.",
                     "Tweede alinea."]


def test_court_rulings_are_cut_at_their_numbered_points():
    text = ("Titel 1 Het verloop  1.1. Red Hat heeft hoger beroep ingesteld. 1.2. Het verloop blijkt uit de stukken. "
            "Titel 2 De kern  2.1. Op 1 mei 2022 is hij in dienst getreden.")
    paras = sp.paragraphs_of("rechtspraak", text, set())
    assert "Red Hat heeft hoger beroep ingesteld." in paras
    assert "Het verloop blijkt uit de stukken." in paras
    assert "Op 1 mei 2022 is hij in dienst getreden." in paras


@pytest.mark.parametrize("paragraph, label", [
    ("De heer Jansen (VVD): Wij steunen dit voorstel.", "De heer Jansen (VVD): "),
    ("Mevrouw Van der Berg: Dank u wel.", "Mevrouw Van der Berg: "),
    ("Wethouder Pietersen: Dat zeg ik toe.", "Wethouder Pietersen: "),
    ("De voorzitter: Ik open de vergadering.", "De voorzitter: "),
    ("Bakker (CDA): Een korte vraag.", "Bakker (CDA): "),
    ("Mevrouw Jansen vraagt het college het volgende: hoe staat het ermee?", None),
    ("Onderwerp: de begroting van 2024.", None),
])
def test_speaker_labels(paragraph, label):
    m = sp.SPEAKER_LABEL.match(paragraph)
    assert (m.group(0) if m else None) == label


@pytest.mark.parametrize("text, conflict", [
    ("Mevrouw Stook opent de vergadering. Hij heet iedereen welkom.", True),
    ("De heer Buijs sprak eerst. Daarna antwoordde mevrouw Buijs dat zij het eens was.", True),
    ("Mevrouw Van der Berg sprak. De heer Van Dijk antwoordde haar.", False),
    ("Mevrouw Stook opent de vergadering. Zij heet iedereen welkom.", False),
    ("De heer Jansen vroeg de wethouder of hij dat wilde toezeggen.", False),
    ("Mevrouw Jaceps vroeg erom en de heer Rose vraagt dat nu ook. Hij ziet een overeenkomst.", False),
])
def test_name_conflicts(text, conflict):
    assert sp.name_conflict(text) is conflict


KAIKKI = [
    {"lang_code": "nl", "word": "doorhakken", "pos": "verb", "senses": [{"glosses": ["to cut"]}],
     "forms": [{"form": f} for f in ("doorhakken", "hak door", "hakte door", "hakten door", "doorgehakt",
                                     "nl-conj-wk")] + [{"form": "separable weak", "tags": ["table-tags"]}]},
    {"lang_code": "nl", "word": "hakte door", "pos": "verb", "senses": [{"form_of": [{"word": "doorhakken"}]}]},
    {"lang_code": "nl", "word": "de knoop doorhakken", "pos": "verb", "senses": [{"glosses": ["to decide"]}]},
    {"lang_code": "nl", "word": "spannen", "pos": "verb", "senses": [{}],
     "forms": [{"form": f} for f in ("spannen", "span", "spant", "spande", "gespannen")]},
    {"lang_code": "nl", "word": "het paard achter de wagen spannen", "pos": "verb", "senses": [{"tags": ["idiomatic"]}]},
    {"lang_code": "nl", "word": "het naadje van de kous", "pos": "phrase", "senses": [{}]},
    {"lang_code": "nl", "word": "hem smeren", "pos": "verb", "senses": [{}]},
    {"lang_code": "nl", "word": "rode kool", "pos": "noun", "senses": [{}]},
    {"lang_code": "nl", "word": "jonge hond", "pos": "noun", "senses": [{"tags": ["idiomatic"]}]},
    {"lang_code": "nl", "word": "eene zaak", "pos": "phrase", "senses": [{"tags": ["archaic"]}]},
    {"lang_code": "en", "word": "cut the knot", "pos": "verb", "senses": [{}]},
]


@pytest.fixture(scope="module")
def idioms():
    return sp.build_idioms(json.dumps(e) for e in KAIKKI)


def test_the_idiom_list(idioms):
    by = {i["idiom"]: i for i in idioms}
    assert set(by) == {"de knoop doorhakken", "het paard achter de wagen spannen", "het naadje van de kous",
                       "jonge hond"}  # not: a form, a verb without a content word, a plain noun, an archaic one
    assert by["de knoop doorhakken"]["keys"] == ["knoop"]
    assert ["hakte", "door"] in by["de knoop doorhakken"]["verb_forms"]
    assert ["separable", "weak"] not in by["de knoop doorhakken"]["verb_forms"]
    assert by["het naadje van de kous"]["contiguous"] == ["het", "naadje", "van", "de", "kous"]


@pytest.mark.parametrize("text, found", [
    ("Het kabinet hakte vorige week na lang wikken de knoop door.", ["de knoop doorhakken"]),
    ("Ik vind dat wij nu de knoop moeten doorhakken.", ["de knoop doorhakken"]),
    ("De knoop is eindelijk doorgehakt.", ["de knoop doorhakken"]),
    ("De knoop in zijn das zat los. Hij hakte hout door de week.", []),
    ("Dat is het paard achter de wagen spannen.", ["het paard achter de wagen spannen"]),
    ("U spant het paard achter de wagen, voorzitter.", ["het paard achter de wagen spannen"]),
    ("Zij spanden het paard achter de wagen.", []),  # spanden is not among the made-up forms
    ("Hij wil het naadje van de kous weten. Een jonge hond.", ["het naadje van de kous", "jonge hond"]),
])
def test_idioms_are_found_in_any_word_order(idioms, text, found):
    assert sp.IdiomMatcher(idioms).find(text) == found


def test_the_split_is_per_document_and_stable():
    splits = [sp.split_of("raad", f"doc{k}", 100) for k in range(2000)]
    assert splits == [sp.split_of("raad", f"doc{k}", 100) for k in range(2000)]
    assert 150 < splits.count("eval") < 250
    assert sp.split_of("raad", "doc1", 0) == "train" and sp.split_of("raad", "doc1", 1000) == "eval"


def passage(pid, text, doc="d", source="raad", score=0.0, types=()):
    return {"pid": pid, "source": source, "doc_id": doc, "text": text, "idiom_score": score,
            "content_type": list(types)}


def test_the_chooser_skips_duplicates_overlaps_and_full_documents():
    long = " ".join(f"woord{k}" for k in range(20))
    pool = [passage("a", long, "d1"), passage("b", long.upper(), "d2"),  # exact duplicate, case aside
            passage("c", "x " + long[:-1] + "Z", "d3"),  # shares a 13-gram with a
            passage("d", "Een korte tekst.", "d1"), passage("e", "Nog een tekst.", "d1"),
            passage("f", "Een derde tekst.", "d4")]
    chooser = sp.Chooser(max_per_doc=2)
    chosen = chooser.take(pool, 10, random.Random(0), weight=lambda p: 1.0)
    assert {p["pid"] for p in chosen} <= {"a", "b", "c", "d", "e", "f"}
    assert sum(p["pid"] in "abc" for p in chosen) == 1
    assert sum(p["doc_id"] == "d1" for p in chosen) <= 2
    assert chooser.skipped["exact_duplicate"] + chooser.skipped["ngram_overlap"] == 2


def test_blocked_ngrams_keep_test_texts_out():
    text = " ".join(f"w{k}" for k in range(15))
    chooser = sp.Chooser(3, sp.ngrams("voor " + text))
    assert chooser.take([passage("a", text)], 1, random.Random(0)) == []


def test_quotas_add_up():
    assert sp.quotas(2000) == {"handelingen": 1000, "raad": 600, "dpc": 200, "rechtspraak": 200}
    assert sum(sp.quotas(4001).values()) == 4001


def test_half_from_the_highest_idiom_class():
    rng = random.Random(1)
    pool = [passage(f"p{k}", f"tekst nummer {k} met eigen woorden", f"d{k}", score=(k % 50) / 10)
            for k in range(1000)]
    chosen, info = sp.choose(pool, 100, sp.Chooser(3), rng, "dpc")
    halves = collections.Counter(p["half"] for p in chosen)
    assert halves == {"idiom": 50, "random": 50}
    assert all(p["idiom_score"] >= info["idiom_class_threshold"] for p in chosen if p["half"] == "idiom")
    assert info["idiom_class_size"] >= 150


def test_workers_get_similar_lengths():
    passages = [{"id": k, "tokens": 48 + k} for k in range(100)]
    parts = sp.balanced_workers(passages, 4)
    assert sorted(p["id"] for part in parts for p in part) == list(range(100))
    means = [sum(p["tokens"] for p in part) / len(part) for part in parts]
    assert max(means) - min(means) < 5


class WordTokenizer:
    class Encoding:
        def __init__(self, ids):
            self.ids = ids

    def encode_batch(self, texts, add_special_tokens):
        return [self.Encoding(t.split()) for t in texts]


def builder(idioms, words):
    import langid

    b = sp.Builder.__new__(sp.Builder)
    b.tokenizer, b.langid, b.wordset, b.matcher = WordTokenizer(), langid, words, sp.IdiomMatcher(idioms)
    return b


SPEECH = ("Voorzitter. Ik wil de minister vandaag een heel concrete vraag stellen over het beleid dat het kabinet de "
          "afgelopen jaren heeft gevoerd op het gebied van de woningbouw in de grote steden van ons land, want daar "
          "gaat het nog steeds mis. Het kabinet moet nu eindelijk de knoop doorhakken en kiezen voor meer woningen.")


def test_a_debate_gives_passages_of_its_speakers_only(idioms):
    words = set(sp.words_of(SPEECH.lower()))
    doc = {"id": "h-tk-1", "date": "2011-01-18", "turns": [
        {"speaker": "De voorzitter", "party": None, "text": SPEECH},
        {"speaker": None, "party": None, "text": SPEECH},
        {"speaker": "Vraag 1", "party": None, "text": SPEECH},
        {"speaker": "De heer Dibi", "party": "GroenLinks", "text": SPEECH + "\n" + SPEECH.replace("Voorzitter. ", "")},
    ]}
    passages, counters = builder(idioms, words).document("handelingen", doc)
    assert passages and all(p["speaker"] == "De heer Dibi" and p["party"] == "GroenLinks" for p in passages)
    assert all(not p["fails"] for p in passages), [p["fails"] for p in passages]
    assert all(p["idioms"] == ["de knoop doorhakken"] * p["text"].count("knoop") for p in passages)
    assert all(sp.MIN_TOKENS <= p["tokens"] <= sp.MAX_TOKENS for p in passages)


def test_council_passages_never_run_over_a_new_speaker(idioms):
    words = set(sp.words_of(SPEECH.lower()))
    text = f"De heer Jansen (VVD): {SPEECH}\n\nMevrouw Bakker (CDA): {SPEECH}\n\n{SPEECH}"
    passages, counters = builder(idioms, words).document("raad", {"doc_id": "r1", "text": text,
                                                                   "content_type": ["procedural"]})
    assert counters["speaker_labels"] == 2
    assert not any(p["text"].startswith(("De heer Jansen", "Mevrouw Bakker")) for p in passages)
    assert all(p["text"].count("Voorzitter.") <= 2 for p in passages)
    first = [p for p in passages if p["pid"].endswith(":0")][0]
    assert first["paragraphs"] == 1  # the second paragraph is another speaker's


def test_filters_catch_old_spelling_motions_and_other_languages(idioms):
    b = builder(idioms, set())
    old = "Wij wenschen den heer Minister geluk met zijn benoeming, want het is een goed oogenblik voor het land."
    assert "old_spelling" in b.checks("handelingen", old)[0]
    motion = "De Kamer,\ngehoord de beraadslaging,\nverzoekt de regering om dat te doen,\nen gaat over tot de orde."
    assert "motion" in b.checks("handelingen", motion)[0]
    english = "The government should finally make a decision and choose to build more houses in the big cities."
    assert "langid" in b.checks("raad", english)[0]
    fails, measures = b.checks("raad", "Er zijn 12345 67890 woningen gebouwd in 2023 en 2024 volgens 1234567 tellingen.")
    assert "digits" in fails and measures["oov"] == 1.0 and "oov" in fails


def test_lists_of_names_fail_on_capitals(idioms):
    b = builder(idioms, set())
    names = "Aanwezig: dhr. Balke (D66), dhr. Voortman (CKenG), mw. De Keijser (D66), dhr. Huijs (CDA), mw. Blom (SP)."
    assert "capitals" in b.checks("raad", names)[0]
    assert "capitals" not in b.checks("handelingen", SPEECH)[0]


def test_frequent_idioms_leave_the_score():
    passages = [{"idioms": ["met name"] * (k % 2) + (["de knoop doorhakken"] if k == 3 else []), "words": 50,
                 "fails": [], "idiom_score": 0.0} for k in range(100)]
    passages.append({"idioms": ["met name"], "words": 10, "fails": ["langid"], "idiom_score": 10.0})
    left_out, df = sp.drop_frequent_idioms(passages, 0.1)
    assert left_out == [("met name", 50)] and df["de knoop doorhakken"] == 1
    assert passages[3]["idioms"] == ["de knoop doorhakken"] and passages[3]["idiom_score"] == 2.0
    assert passages[1]["idioms"] == [] and passages[1]["idiom_score"] == 0.0 and passages[-1]["idiom_score"] == 0.0


def test_a_speaker_inside_a_council_paragraph_fails(idioms):
    b = builder(idioms, set())
    text = "Dit amendement is ondertekend door mij. Mevrouw Huisman (PvdA): Voorzitter! Mij bekruipt het gevoel."
    assert "speaker_change" in b.checks("raad", text)[0]
    assert "speaker_change" not in b.checks("raad", "Mevrouw Huisman sprak. Zij was het eens met de wethouder.")[0]
    procedure = "Deze motie is voorgesteld door de leden Ormel en Peters. Zij krijgt nr. 81 (33000-V)."
    assert "motion" in b.checks("handelingen", procedure)[0]
