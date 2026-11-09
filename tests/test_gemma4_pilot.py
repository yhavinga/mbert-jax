"""experiments/en-nl-translator/gemma4_pilot.py without a model: the context trial's chains (Handelingen turns before
a passage, short Nemotron conversations) and its rounds, where every turn sees the earlier turns of its own chain with
the translations made with context before it. No TPU: run with --noconftest where the conftest's imports (jax, flax)
are not installed."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "en-nl-translator"))

import gemma4_pilot as gp  # noqa: E402


def words(text):
    return len(text.split())


def debate(n_turns, long_turn=None):
    turns = [{"speaker": "De voorzitter" if t % 2 else f"Spreker{t}", "party": None if t % 2 else "PvdA",
              "text": f"Beurt {t} regel een.\n\n  Beurt {t}   regel twee."} for t in range(n_turns)]
    if long_turn is not None:
        turns[long_turn]["text"] = "woord " * 500
    return {"id": "d1", "turns": turns}


def pair(t, k=0, nl=None):
    return {"id": 100 + t, "pid": f"handelingen:d1:t{t}:{k}", "source": "handelingen",
            "nl": nl or f"Beurt {t} regel een.\nBeurt {t} regel twee.", "en": f"Turn {t}."}


def test_handelingen_chains_take_two_turns_before_the_passage_when_they_fit():
    chains = gp.handelingen_chains([pair(3)], {"d1": debate(5)}, words)
    assert len(chains) == 1
    turns = chains[0]["turns"]
    assert [t["turn"] for t in turns] == [1, 2, 3] and [t["kind"] for t in turns] == ["context", "context", "passage"]
    assert turns[0]["label"] == "De voorzitter" and turns[1]["label"] == "Spreker2 (PvdA)"
    assert turns[0]["nl"] == "Beurt 1 regel een.\nBeurt 1 regel twee." and turns[0]["en"] is None
    assert turns[2]["en"] == "Turn 3." and chains[0]["pair_id"] == 103


def test_handelingen_chains_stop_at_a_turn_that_is_too_long_and_skip_a_first_turn():
    chains = gp.handelingen_chains([pair(2), pair(0)], {"d1": debate(4, long_turn=0)}, words)
    assert [[t["turn"] for t in c["turns"]] for c in chains] == [[1, 2]]


def test_nemotron_chains_take_short_english_lmarena_conversations_with_two_user_turns():
    pytest.importorskip("langid")
    english = "Could you explain how the tides work and why there are two of them every day?"
    ok = {"uuid": "a", "metadata": {"source": "lmarena-ai/arena-human-preference-140k"},
          "messages": [{"role": "user", "content": english, "reasoning_content": None},
                       {"role": "assistant", "content": "The Moon pulls the oceans toward it, and the Earth is "
                        "pulled away from the far side, so there are two bulges.", "reasoning_content": "hidden"},
                       {"role": "user", "content": "And what about the Sun? Does it matter as well?"},
                       {"role": "assistant", "content": "Yes, the Sun adds its own, smaller tide: spring and neap "
                        "tides come from the two together."}]}
    one_question = {**ok, "uuid": "b", "messages": ok["messages"][:2]}
    wildchat = {**ok, "uuid": "c", "metadata": {"source": "allenai/WildChat-1M"}}
    missing = {**ok, "uuid": "d", "messages": [{"role": "user", "content": None}] + ok["messages"][1:]}
    chains = gp.nemotron_chains([ok, one_question, wildchat, missing], words)
    assert [c["chain"] for c in chains] == ["a"]
    turns = chains[0]["turns"]
    assert [t["label"] for t in turns] == ["Gebruiker", "Assistent", "Gebruiker", "Assistent"]
    assert turns[1]["en"].startswith("The Moon") and all(t["nl"] is None for t in turns)
    assert "hidden" not in json.dumps(chains)


def test_nemotron_lines_of_a_byte_range_skip_the_cut_ones(tmp_path):
    path = tmp_path / "chunk.jsonl"
    path.write_text('ut": 1}\n{"uuid": "x", "messages": []}\n{"uuid": "y", "mess', encoding="utf-8")
    assert [r["uuid"] for r in gp.nemotron_conversations(path)] == ["x"]


def test_translate_chains_gives_every_turn_the_context_of_its_own_chain():
    chains = [
        {"chain": "h", "turns": [{"label": "A", "nl": "a-nl", "en": None}, {"label": "B", "nl": "b-nl", "en": "b-en"}]},
        {"chain": "n", "turns": [{"label": "Gebruiker", "nl": None, "en": "q1"}, {"label": "Assistent", "nl": None, "en": "r1"},
                                 {"label": "Gebruiker", "nl": None, "en": "q2"}]},
    ]
    calls = []

    def translate(name, direction, items):
        calls.append((name, direction, items))
        out = []
        for text, context, label in items:
            tag = "C" if context else "P"
            out.append((f"{tag}[{text}|{len(context or [])}|{label}]" if direction == "en-nl" else f"EN({text})",
                        True, 10, 5))
        return out

    gp.translate_chains(chains, translate)
    assert [c[0] for c in calls] == ["context-nl-en", "plain-en-nl", "context-en-nl-1", "context-en-nl-2"]
    h, n = chains[0]["turns"], chains[1]["turns"]
    assert h[0]["en"] == "EN(a-nl)" and h[0]["nl_plain"] == h[0]["nl_context"] == "P[EN(a-nl)|0|None]"
    assert h[1]["nl_plain"] == "P[b-en|0|None]" and h[1]["nl_context"] == "C[b-en|1|B]"
    # The third turn sees the first two, the second with its context translation
    _, _, items = calls[3]
    assert items == [("q2", [{"label": "Gebruiker", "source": "q1", "translation": "P[q1|0|None]"},
                             {"label": "Assistent", "source": "r1", "translation": "C[r1|1|Assistent]"}], "Gebruiker")]
    assert n[2]["nl_context"] == "C[q2|2|Gebruiker]" and n[2]["context_tokens_in"] == 10
