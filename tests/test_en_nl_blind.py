"""experiments/en-nl-translator/blind.py: the idiom list from kaikki-like entries, the idioms' patterns and the matching
on lemma, the paragraphs of a post. spaCy is replaced by tokens with a given lemma; select runs on a VM."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "en-nl-translator"))

import blind  # noqa: E402

LEMMAS = {"beans": "bean", "spilled": "spill", "spills": "spill", "ropes": "rope", "was": "be", "is": "be",
          "kicked": "kick", "his": "his"}


def tok(text):
    return SimpleNamespace(text=text, lemma_=LEMMAS.get(text.lower(), text.lower()),
                           is_punct=bool(text) and all(not c.isalnum() for c in text) and text not in ("'s", "’s"),
                           is_space=False)


def doc(text):
    return [tok(t) for t in text.replace("'s", " 's").replace("-", " - ").split()]  # as spaCy splits them


def toks(text):
    return [(blind.norm(t.text), frozenset({blind.norm(t.text), blind.norm(t.lemma_)}),
             not t.is_punct and not t.is_space) for t in doc(text) if t.text not in blind.HYPHENS]


def entry(word, glosses_tags, cats=()):
    return json.dumps({"word": word, "lang_code": "en", "pos": "phrase", "categories": list(cats),
                       "senses": [{"glosses": [g], "tags": list(t)} for g, t in glosses_tags]})


def test_the_idiom_list_marks_a_literal_sense():
    found = blind.build_idioms([
        entry("spill the beans", [("To reveal a secret.", ["idiomatic"])]),
        entry("break the ice", [("Used other than figuratively or idiomatically: see break, ice.", []),
                                ("To start a conversation.", ["idiomatic"])]),
        entry("on the ropes", [("Near defeat.", [])], cats=["English idioms"]),
        entry("red car", [("A car that is red.", [])]),
        entry("old hat", [("Old-fashioned.", ["idiomatic", "obsolete"])]),
        entry("ice", [("Frozen water.", ["idiomatic"])]),
    ])
    assert sorted(found) == ["break the ice", "on the ropes", "spill the beans"]
    assert found["break the ice"]["literal"] and not found["spill the beans"]["literal"]
    assert found["break the ice"]["gloss"] == "To start a conversation."


def test_patterns_and_the_rule():
    assert blind.pattern_of(doc("spill the beans")) == [("lit", frozenset({"spill"})), ("lit", frozenset({"the"})),
                                                        ("lit", frozenset({"beans", "bean"}))]
    assert blind.pattern_of(doc("one's cup of tea"))[0] == ("poss",)
    assert blind.pattern_of(doc("someone's cup of tea"))[0] == ("poss",)
    assert blind.pattern_of(doc("give someone a hand"))[1] == ("gap",)
    assert blind.pattern_of(doc("pull something off")) == [("lit", frozenset({"pull"})), ("gap",),
                                                           ("lit", frozenset({"off"}))]
    assert blind.pattern_of(doc("something else")) == [("lit", frozenset({"else"}))]  # the leading argument goes
    assert blind.rule_ok(blind.pattern_of(doc("spill the beans")))
    assert blind.rule_ok(blind.pattern_of(doc("under the weather")))
    assert not blind.rule_ok(blind.pattern_of(doc("of course")))
    assert not blind.rule_ok(blind.pattern_of(doc("as well as")))


def test_matching_on_lemma_with_placeholders():
    patterns = [(w, blind.pattern_of(doc(w))) for w in
                ("spill the beans", "one's cup of tea", "pull something off", "on the ropes", "well-to-do")]
    m = blind.Matcher(patterns)
    found = m.find(toks("She spilled the beans , and it was not his cup of tea ."))
    assert [f[0] for f in found] == ["spill the beans", "one's cup of tea"]
    assert [f[0] for f in m.find(toks("Opera is not Maria 's cup of tea ."))] == ["one's cup of tea"]
    assert [f[0] for f in m.find(toks("They pulled the whole thing off ."))] == []  # 'pulled' has no lemma here
    assert [f[0] for f in m.find(toks("They pull the deal off ."))] == ["pull something off"]
    assert m.find(toks("A cup of tea .")) == []  # the possessive is part of the idiom
    assert [f[0] for f in m.find(toks("The champion was on the ropes ."))] == ["on the ropes"]
    assert [f[0] for f in m.find(toks("a well - to - do family"))] == ["well-to-do"]
    assert m.find(toks("the beans spilled")) == []


def test_paragraphs_of_a_post():
    pytest.importorskip("bs4")
    html = ("<p>Own <a href='x'>text</a>.</p><blockquote><p>A quote.</p></blockquote>"
            "<div class='wp-caption'><p class='wp-caption-text'>Photo by X.</p></div>"
            "<p><em>Editor's note: this is a note.</em></p><ul><li><p>In a list.</p></li></ul><p>Two<br>lines.</p>")
    assert blind.paragraphs(html) == ["Own text.", "Two lines."]
    assert blind.REPUBLISHED.search("This article was originally published on Meduza")
    assert blind.BOILERPLATE.match("Read more: the story")
