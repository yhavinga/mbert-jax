"""Nemotron Chat v3 (experiments/en-nl-translator/nemotron_chat.py, nemotron-chat-v3.md): the source rows and filters,
pieces that give the turn back character for character, the stratified draw, the context of a piece and its budget, the
waves with and without context (a failure breaks its conversation), the checks, natural-v1-chat, and collect, report
and blind end to end on small synthetic inputs (no model, no TPU, no langid)."""

import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "experiments" / "en-nl-translator"
for _p in (CODE, ROOT / "experiments" / "flan-nl", ROOT / "scripts"):
    sys.path.insert(0, str(_p))

import nemotron_chat as nc  # noqa: E402
import translate as tr  # noqa: E402


def words(text):
    return len(text.split())


def row(uuid="u1", seed=nc.LMARENA[0], messages=None, system=""):
    msgs = [{"role": "system", "content": system}] + (messages or [
        {"role": "user", "content": "How do I bake bread?"},
        {"role": "assistant", "content": "Mix flour, water and yeast.", "reasoning_content": "think"}])
    return {"uuid": uuid, "messages": msgs, "metadata": {"seed_dataset": seed, "model": "zai-org/GLM-5",
                                                         "train_turns": [False, False, True]}}


# ---------------------------------------------------------------------------------------------------- source rows

def test_an_lmarena_conversation_drops_its_empty_system_message_and_keeps_the_rest():
    conv, why = nc.conversation(row())
    assert why is None
    assert [t["role"] for t in conv["turns"]] == ["user", "assistant"]
    assert [t["message"] for t in conv["turns"]] == [1, 2]
    assert conv["reasoning_chars"] == 5 and json.loads(conv["train_turns"]) == [False, False, True]


@pytest.mark.parametrize("kwargs, reason", [
    ({"seed": "allenai/WildChat-1M"}, "seed_not_lmarena"),
    ({"system": "You are a pirate."}, "system_prompt"),
    ({"messages": [{"role": "user", "content": "hi"}]}, "roles"),
    ({"messages": [{"role": "assistant", "content": "hi"}, {"role": "user", "content": "x"}]}, "roles"),
    ({"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "  "}]}, "roles"),
    ({"messages": [{"role": "user", "content": None}, {"role": "assistant", "content": "x"}]}, "roles"),
])
def test_rows_that_are_left_out_say_why(kwargs, reason):
    assert nc.conversation(row(**kwargs)) == (None, reason)


def test_a_byte_range_gives_only_its_whole_lines(tmp_path):
    lines = [json.dumps({"uuid": f"u{i}"}) for i in range(4)]
    data = ("\n".join(lines) + "\n").encode()
    start = len(lines[0]) // 2
    end = len(data) - len(lines[3]) // 2
    path = tmp_path / f"part-{start}.jsonl"
    path.write_bytes(data[start:end])
    assert [r["uuid"] for r in nc.range_rows(path, nc.offset_of(path))] == ["u1", "u2"]
    whole = tmp_path / "part-0.jsonl"
    whole.write_bytes(data)
    assert [r["uuid"] for r in nc.range_rows(whole, 0)] == ["u0", "u1", "u2", "u3"]


# ---------------------------------------------------------------------------------------------------- markdown, pieces

TURN = ("## Plan\n\nFirst do this. Then do that.\n\n```\nline one\n\nline two\n```\n\n- a\n- b\n\n"
        "| x | y |\n|---|---|\n| 1 | 2 |\n")


def test_a_fenced_block_with_a_blank_line_stays_one_block_and_the_blocks_give_the_text_back():
    parts = nc.blocks(TURN.strip())
    assert "".join(b + s for b, s in parts) == TURN.strip()
    assert "```\nline one\n\nline two\n```" in [b for b, _ in parts]
    assert nc.fences(TURN) == ["```\nline one\n\nline two\n```"]
    assert "line one" not in nc.prose(TURN)


def test_split_turn_keeps_every_character_and_respects_the_limit():
    text = "\n" + " ".join(f"Sentence {i} is here." for i in range(40)) + "\n\nShort.\n\n" + TURN + "  "
    lead, pieces, trail = nc.split_turn(text, words, limit=30)
    assert lead + "".join(p + s for p, s in pieces) + trail == text
    assert all(words(p) <= 30 for p, _ in pieces)
    assert len(pieces) > 3


def test_sentences_keep_the_separator_of_an_empty_part():
    for block in ("First one. \nSecond one.", "A. \n\tB? C!\n", "x:\n\ny", "End. ", "\nStart"):
        parts = nc.sentences(block, "\n\n")
        assert "".join(p + s for p, s in parts) == block + "\n\n" and all(p for p, _ in parts)


def test_split_turn_gives_any_text_back_or_refuses_it():
    import random

    rng = random.Random(7)
    alphabet = ["word", "w", ".", "?", ":", " ", "  ", "\t", "\n", "\n\n", " \n", "\r", "```", "~~~", "`x`", "#", "- ",
                "\u00a0", "\u2028", "\x0b", "\x85", " ```py\n"]
    for _ in range(20000):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 60)))
        if not text.strip():
            continue
        try:
            lead, pieces, trail = nc.split_turn(text, words, limit=rng.choice([1, 3, 8, 40]))
        except nc.Unsplittable:
            continue
        assert lead + "".join(p + s for p, s in pieces) + trail == text


def test_a_fenced_block_longer_than_a_piece_cannot_be_split():
    with pytest.raises(nc.Unsplittable, match="fence"):
        nc.split_turn("Intro.\n\n```\n" + "x " * 50 + "\n```", words, limit=20)
    with pytest.raises(nc.Unsplittable, match="sentence"):
        nc.split_turn("word " * 50, words, limit=20)


def test_assemble_puts_the_sources_separators_back():
    assert nc.assemble("\n", ["A", "B", "C"], ["\n\n", "\n\n\n", ""], " ") == "\nA\n\nB\n\n\nC "


# ---------------------------------------------------------------------------------------------------- filters

def conv_of(*texts):
    return {"uuid": "c", "turns": [{"role": ("user", "assistant")[i % 2], "text": t, "message": i + 1}
                                   for i, t in enumerate(texts)], "reasoning_chars": 0, "train_turns": "null"}


def test_code_tasks_are_found_by_fence_tag_share_and_code_lines():
    assert nc.code_reason(conv_of("Write a sort.", "```python\nprint(1)\n```")) == "code_fence_lang"
    long_untagged = "Here:\n\n```\n" + "data " * 80 + "\n```\n\nDone."
    assert nc.code_reason(conv_of("Show it.", long_untagged)) == "code_share"
    lines = "Fix this:\ndef f(x):\nimport os\nconst a = 1\n}"
    assert nc.code_reason(conv_of(lines, "Sure, here is the fix.")) == "code_lines"
    small = "A table:\n\n```\nA B\n```\n\n" + "Plain prose sentence here. " * 20
    assert nc.code_reason(conv_of("Make a table.", small)) is None


def test_language_is_checked_on_the_first_turn_and_on_long_paragraphs():
    def identify(text):
        return "fr" if "bonjour" in text else "en"

    assert nc.language_reason(conv_of("bonjour, comment allez-vous aujourd'hui", "Hello."), identify) == "lang_first"
    assert nc.language_reason(conv_of("bonjour", "Hello."), identify) is None  # too short to tell
    para = "bonjour " * 20
    assert nc.language_reason(conv_of("Translate this text please", "Sure.\n\n" + para), identify) == "lang_paragraph"
    assert nc.language_reason(conv_of("Translate this text please", "Sure."), identify) is None


def test_prepare_gives_pieces_tokens_and_a_stratum():
    conv = conv_of("Tell me about tea. " * 3, "Tea is a drink. " * 10, "And coffee?", "Coffee is too.")
    out, why = nc.prepare(conv, words, lambda text: "en")
    assert why is None
    assert out["stratum"] == "a2s" and out["n_assistant"] == 2
    assert all(t["pieces"] and t["lead"] == "" for t in out["turns"])
    big = conv_of("Hi.", "word " * (nc.TURN_MAX_TOKENS + 1))
    assert nc.prepare(big, words, lambda text: "en") == (None, "turn_long")


def test_the_draw_is_even_over_the_strata_deterministic_and_takes_all_of_a_small_stratum():
    pool = [{"uuid": f"u{i}", "stratum": s, "tokens": i} for i, s in
            enumerate(["a1s"] * 50 + ["a2m"] * 50 + ["a4l"] * 3)]
    chosen = nc.pick(pool, 30)
    assert len(chosen) == 30 and chosen == nc.pick(list(reversed(pool)), 30)
    counts = {s: sum(c["stratum"] == s for c in chosen) for s in ("a1s", "a2m", "a4l")}
    assert counts["a4l"] == 3 and abs(counts["a1s"] - counts["a2m"]) <= 1


def test_workers_get_about_equal_tokens():
    convs = [{"uuid": f"u{i}", "tokens": t} for i, t in enumerate([900, 800, 700, 600, 100, 100, 100, 100])]
    w = nc.assign_workers(convs, 4)
    load = [sum(c["tokens"] for c in convs if w[c["uuid"]] == k) for k in range(4)]
    assert max(load) - min(load) <= 300


# ---------------------------------------------------------------------------------------------------- context

def test_history_gives_earlier_turns_whole_then_this_turns_earlier_pieces():
    turns = [{"role": "user"}, {"role": "assistant"}]
    pieces = {0: [{"pid": "c|t0|p0", "sep_after": ""}],
              1: [{"pid": "c|t1|p0", "sep_after": "\n\n"}, {"pid": "c|t1|p1", "sep_after": ""}]}
    results = {"c|t0|p0": {"text_nl": "Vraag"}, "c|t1|p0": {"text_nl": "Eerste deel"}}
    items = nc.history(turns, pieces, results, 1, 1)
    assert items == [{"label": "User", "translation": "Vraag"},
                     {"label": "Assistant, this turn so far", "translation": "Eerste deel"}]
    assert nc.history(turns, pieces, results, 0, 0) == []
    assert nc.current_label("assistant", 1) == "Assistant, continued"


def test_the_context_keeps_the_newest_turns_within_the_budget_and_cuts_a_long_newest_one():
    items = [{"label": "User", "translation": "een " * 10}, {"label": "Assistant", "translation": "twee " * 10},
             {"label": "User", "translation": "drie " * 10}]
    chosen, cut = nc.fit_context(items, words, budget=25)
    assert [i["translation"].split()[0] for i in chosen] == ["twee", "drie"] and not cut
    long_ = [{"label": "Assistant", "translation": "a " * 30 + "\n\n" + "b " * 5}]
    chosen, cut = nc.fit_context(long_, words, budget=12)
    assert cut and chosen[0]["translation"].startswith(nc.CUT_MARK) and "a a" not in chosen[0]["translation"]
    assert nc.fit_context([{"label": "User", "translation": "x " * 40}], words, budget=10) == ([], True)


def test_natural_v1_chat_without_context_is_natural_v1_and_with_context_names_the_speakers():
    p = tr.PROMPTS["natural-v1-chat"]
    assert tr.user_text(p, "Hello.") == tr.user_text(tr.PROMPTS["natural-v1"], "Hello.")
    text = tr.user_text(p, "Thanks!", context=[{"label": "User", "translation": "Hoi"},
                                                {"label": "Assistant", "translation": "Hallo"}], label="User")
    assert "<conversation_so_far>\n[User]\nHoi\n\n[Assistant]\nHallo\n\n</conversation_so_far>" in text
    assert text.endswith("The English text to translate (User):\n\n\nThanks!")
    assert text.startswith(tr.user_text(tr.PROMPTS["natural-v1"], "x").split("\n\n\n")[0] + "\n\nThe text is a turn")
    model = tr.MODELS["gemma4-31b-chat"]
    assert model["prompt"] == model["context_prompt"] == "natural-v1-chat"
    assert set(model["slots"]) == set(model["length_classes"]) == set(nc.CLASSES)
    assert all(c["max_source"] == nc.PIECE_MAX for k, c in model["length_classes"].items() if k != "n128")
    assert model["length_classes"]["c2k"]["prefill_lengths"][-1] == nc.PROMPT_MAX


def test_a_template_with_source_text_still_needs_the_source():
    with pytest.raises(KeyError):
        tr.user_text(tr.PROMPTS["chat-v1"], "x", context=[{"label": "a", "translation": "b"}])


# ---------------------------------------------------------------------------------------------------- waves

def units_of(convs):
    return {c: [{"pid": f"{c}|t{t}|p0", "conv": c, "turn": t, "piece": 0} for t in range(n)] for c, n in convs.items()}


def test_with_context_a_wave_takes_the_next_piece_of_every_conversation_and_a_failure_stops_it():
    units = units_of({"a": 3, "b": 2, "c": 1})
    seen = []

    def translate_wave(w, wave, results):
        seen.append([u["pid"] for u in wave])
        for u in wave:  # every earlier piece of its conversation is done before it
            assert all(f"{u['conv']}|t{t}|p0" in results for t in range(u["turn"]))
        return {u["pid"]: {"failed": u["pid"] == "b|t0|p0"} for u in wave}

    results, broken, waves = nc.run_waves(units, "ctx", translate_wave)
    assert seen == [["a|t0|p0", "b|t0|p0", "c|t0|p0"], ["a|t1|p0"], ["a|t2|p0"]]
    assert broken == {"b"} and waves == 3 and "b|t1|p0" not in results


def test_without_context_every_piece_is_one_wave_and_a_restart_goes_on():
    units = units_of({"a": 3, "b": 2})
    calls = []

    def translate_wave(w, wave, results):
        calls.append((w, len(wave)))
        return {u["pid"]: {"failed": False} for u in wave}

    results, _, waves = nc.run_waves(units, "noctx", translate_wave)
    assert calls == [(0, 5)] and waves == 1
    _, _, again = nc.run_waves(units, "noctx", translate_wave, results=results, first_wave=1)
    assert again == 0 and calls == [(0, 5)]
    part = {"a|t0|p0": {"failed": False}}
    nc.run_waves(units_of({"a": 3}), "ctx", translate_wave, results=part, first_wave=1)
    assert calls[1:] == [(1, 1), (2, 1)]


def test_a_wave_that_misses_a_unit_stops():
    with pytest.raises(RuntimeError, match="not the wave's"):
        nc.run_waves(units_of({"a": 1}), "ctx", lambda w, wave, r: {})


# ---------------------------------------------------------------------------------------------------- checks

def dutch(text):
    return "nl"


@pytest.mark.parametrize("nl, ended, flag", [
    ("", True, "empty"),
    ("Meng bloem, water en gist.", False, "truncated"),
    ("[Assistant]\nMeng bloem, water en gist.", True, "tag_leak"),
    ("Meng.", True, "length_ratio"),
])
def test_hard_checks_flag_empty_truncated_leaked_and_short_translations(nl, ended, flag):
    assert flag in nc.hard_flags("Mix the flour, the water and the yeast in a bowl.", nl, ended, dutch)


def test_a_good_translation_with_code_passes_and_its_code_is_compared_byte_for_byte():
    en = "Run this:\n\n```\nls -la\n```\n\nThen wait for the result to appear on the screen."
    nl = "Voer dit uit:\n\n```\nls -la\n```\n\nWacht dan tot het resultaat op het scherm verschijnt."
    assert nc.hard_flags(en, nl, True, dutch) == []
    st = nc.structure_of(en, nl)
    assert st["fences_same"] is True and st["diffs"] == []
    changed = nc.structure_of(en, nl.replace("ls -la", "ls -l"))
    assert changed["fences_same"] is False
    assert nc.structure_of("Plain.", "Gewoon.")["fences_same"] is None


# ---------------------------------------------------------------------------------------------------- end to end

def build_data(tmp_path, n_convs=50):
    convs, turns, pieces = [], [], []
    for c in range(n_convs):
        cid = f"conv{c:03d}"
        worker = c % nc.WORKERS
        convs.append({"conv": cid, "seed": nc.LMARENA[c % 2], "stratum": "a2m", "turns": 4, "assistant": 2,
                      "tokens": 100, "worker": worker, "train_turns": "[false, false, false, false, true]",
                      "reasoning_chars": 0})
        for t in range(4):
            role = ("user", "assistant")[t % 2]
            texts = [f"Question {c} {t} about the weather today?"] if role == "user" else \
                [f"Answer {c} {t} part one is here.", f"## Part two {c} {t}\n\n- item a\n- item b"]
            turns.append({"conv": cid, "turn": t, "role": role, "message": t + 1, "text": "\n\n".join(texts),
                          "lead": "", "trail": "", "pieces": len(texts), "tokens": 20, "worker": worker})
            for j, x in enumerate(texts):
                pieces.append({"pid": f"{cid}|t{t}|p{j}", "conv": cid, "turn": t, "piece": j, "n_pieces": len(texts),
                               "role": role, "text": x, "sep_after": "\n\n" if j < len(texts) - 1 else "",
                               "tokens": words(x), "worker": worker, "stratum": "a2m"})
    data = tmp_path / "data"
    nc.write_rows(convs, data / "conversations.parquet")
    nc.write_rows(turns, data / "turns.parquet")
    nc.write_rows(pieces, data / "pieces.parquet")
    nc.write_rows([{"uuid": c["conv"], "seed": c["seed"], "stratum": "a2m", "turns": 4, "assistant": 2, "tokens": 100,
                    "tokens_user": 30, "tokens_assistant": 70, "chars": 500, "pieces": 6, "pieces_short": 6,
                    "tokens_short": 100, "tokens_long": 0, "fences": 0, "reasoning_chars": 0} for c in convs],
                  data / "pool-meta.parquet")
    (data / "sample.json").write_text(json.dumps({
        "ranges": 2, "range_bytes": 1000, "file_bytes": nc.FILE_BYTES, "rows": 80,
        "seeds": {nc.LMARENA[0]: 40, nc.LMARENA[1]: 20, "allenai/WildChat-1M": 20},
        "left_out": {"seed_not_lmarena": 20, "code_fence_lang": 5, "lang_first": 5}, "pool": n_convs,
        "pool_strata": {"a2m": n_convs}, "chosen": n_convs, "chosen_strata": {"a2m": n_convs}, "turns": len(turns),
        "pieces": len(pieces), "pieces_short": len(pieces), "tokens": 100 * n_convs}))
    return data, pieces


def fake_translation(p, cond):
    text = p["text"].replace("Question", "Vraag").replace("about the weather today", "over het weer vandaag")
    text = text.replace("Answer", "Antwoord").replace("part one is here", "deel een is hier").replace("Part two", "Deel twee")
    return text if cond == "ctx" else text.replace("vandaag", "nu").replace("hier", "daar")


def build_gen(tmp_path, pieces, fail_pid=None):
    gen = tmp_path / "gen"
    for cond, cls in (("ctx", "c2k"), ("noctx", "n128"), ("noctx", "n512")):
        for w in range(nc.WORKERS):
            rows = []
            broken = False
            for p in sorted(pieces, key=lambda p: (p["conv"], p["turn"], p["piece"])):
                if p["worker"] != w or (cls == "n512" and p["tokens"] <= nc.SHORT_MAX) or \
                        (cls == "n128" and p["tokens"] > nc.SHORT_MAX):
                    continue
                failed = cond == "ctx" and p["pid"] == fail_pid
                if cond == "ctx" and broken and p["conv"] == fail_pid.split("|")[0]:
                    continue
                broken = broken or failed
                rows.append({"pid": p["pid"], "conv": p["conv"], "turn": p["turn"], "piece": p["piece"],
                             "role": p["role"], "cond": cond, "cls": cls, "worker": w, "wave": p["turn"], "attempt":
                             int(failed), "text_nl": "" if failed else fake_translation(p, cond), "ended": True,
                             "tokens_in": 50, "tokens_out": 12, "flags0": json.dumps(["empty"] if failed else []),
                             "flags": json.dumps(["empty"] if failed else []), "failed": failed,
                             "context_items": p["turn"], "context_tokens": 10 * p["turn"], "context_cut": False})
            d = gen / cond / cls / f"w{w}"
            d.mkdir(parents=True)
            if rows:
                nc.write_rows(rows, d / "wave-000.parquet")
            (d / "DONE.json").write_text(json.dumps({"cond": cond, "cls": cls, "worker": w, "units": len(rows),
                                                     "engine_seconds": 100, "seconds": 50.0, "tokens_in": 50 * len(rows),
                                                     "tokens_out": 12 * len(rows), "waves": 4, "retried": 0,
                                                     "retry_seconds": 0.0, "full_tokens_out": 12 * len(rows),
                                                     "full_seconds": 40.0}))
    return gen


def test_collect_report_and_blind_end_to_end(tmp_path):
    data, pieces = build_data(tmp_path)
    fail = "conv007|t2|p0"
    gen = build_gen(tmp_path, pieces, fail_pid=fail)
    col = tmp_path / "collect"
    info = nc.collect(Namespace(data=str(data), gen=str(gen), out=str(col), conds="ctx,noctx", leaf=r"w\d+"))
    s = info["summary"]
    assert s["noctx"]["complete"] == 50 and s["ctx"]["complete"] == 49 and s["ctx"]["broken"] == 1
    assert s["ctx"]["pieces"]["failed"] == 1 and s["ctx"]["flags"] == {"empty": 1}
    chat = [json.loads(x) for x in (col / "chat-nl-ctx.jsonl").read_text().splitlines()]
    assert len(chat) == 49 and all(len(c["messages"]) == 4 for c in chat)
    assert chat[0]["messages"][1]["content"].startswith("Antwoord") and "\n\n## Deel twee" in chat[0]["messages"][1]["content"]
    assert s["ctx"]["turns"]["turns_headings_same"] == s["ctx"]["turns"]["turns_headings"] > 0
    qe_rows = []
    for k in range(nc.WORKERS):
        rows = nc.read_rows(col / f"qe-in-w{k}.parquet")
        assert rows and all(r["hypothesis"] for r in rows)
        qe_rows += [{"id": r["id"], "qe_source_hypothesis": 1.0 if r["id"].startswith("ctx") else 1.5} for r in rows]
    nc.write_rows(qe_rows, tmp_path / "qe" / "qe-w0.parquet")
    out = tmp_path / "report.md"
    rep = nc.report(Namespace(data=str(data), collect=str(col), qe=str(tmp_path / "qe" / "qe-w*.parquet"), out=str(out)))
    text = out.read_text()
    assert "## Schatting voor 100.000 gesprekken" in text and "Vraag" not in text and "Antwoord" not in text
    assert rep["paired"]["ctx beter"] > 0 and rep["paired_mean"] == pytest.approx(-0.5)
    assert set(rep["extrapolation"]) >= {"ctx", "noctx"}
    key = tmp_path / "key" / "key.jsonl"
    key.parent.mkdir()
    b = nc.blind(Namespace(data=str(data), collect=str(col), out=str(tmp_path / "pkg"), key=str(key)))
    assert b["items"] == nc.N_BLIND and b["by_role"] == {"user": 20, "assistant": 20}
    items = [json.loads(x) for x in (tmp_path / "pkg" / "judge-input.jsonl").read_text().splitlines()]
    keys = [json.loads(x) for x in key.read_text().splitlines()]
    assert [i["blind_id"] for i in items] == [k["blind_id"] for k in keys]
    assert len({k["conv"] for k in keys}) == nc.N_BLIND and all(k["turn"] > 0 for k in keys)
    assert all({k["A"], k["B"]} == {"ctx", "base"} for k in keys)
    assert all(k["conv"] != "conv007" or k["turn"] < 2 for k in keys)
    reviewer = (tmp_path / "pkg" / "REVIEWER.md").read_text()
    assert "context" in reviewer and "ctx" not in reviewer and "noctx" not in reviewer and str(nc.N_BLIND) in reviewer


def test_collect_refuses_a_missing_translation_without_context(tmp_path):
    data, pieces = build_data(tmp_path, n_convs=8)
    gen = build_gen(tmp_path, pieces, fail_pid="conv001|t2|p0")
    victim = gen / "noctx" / "n128" / "w0" / "wave-000.parquet"
    rows = nc.read_rows(victim)
    nc.write_rows(rows[1:], victim)
    with pytest.raises(SystemExit, match="noctx: 1 pieces without a translation"):
        nc.collect(Namespace(data=str(data), gen=str(gen), out=str(tmp_path / "c"), conds="ctx,noctx", leaf=r"w\d+"))
