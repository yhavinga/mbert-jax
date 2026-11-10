"""experiments/en-nl-translator/sft_build.py: the query on the pairs, the context examples, the draw of the arms'
back-translation examples and the replay, the prompts of the tag, and a build end to end on small tables. No model and
no TPU (run with --noconftest where jax and flax are not installed); `check` runs MaxText and is run on a VM."""

import argparse
import re
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "en-nl-translator"))

import sft_build as sb  # noqa: E402

TG_HEAD = ("You are a professional English (en) to Dutch (nl) translator. Your goal is to accurately convey the meaning "
           "and nuances of the original English text while adhering to Dutch grammar, vocabulary, and cultural "
           "sensitivities.\nProduce only the Dutch translation, without any additional explanations or commentary. "
           "Please translate the following English text into Dutch:\n\n\n")
SPECIAL = re.compile(r"<unused\d+>|<start_of_turn>|<end_of_turn>|<bos>|.", re.S)


class CharTokenizer:
    """A token is a character, or one of Gemma's marks (<unused0> is id 6, as in Gemma 3's tokenizer)."""

    def convert_tokens_to_ids(self, token):
        m = re.fullmatch(r"<unused(\d+)>", token)
        return 6 + int(m.group(1)) if m else 100000 + len(token)

    def __call__(self, text, add_special_tokens=False):
        assert not add_special_tokens
        if isinstance(text, list):
            return {"input_ids": [self(t)["input_ids"] for t in text]}
        return {"input_ids": [self.convert_tokens_to_ids(t) if len(t) > 1 else 1000 + ord(t)
                              for t in SPECIAL.findall(text)]}


def pair(pid, source="handelingen", split="train", kind="passage", nl=None, en=None, qe=1.0, hard=(), chrf=60.0):
    return {"id": pid, "pid": pid, "kind": kind, "source": source, "split": split,
            "nl": nl or f"Nederlandse tekst van {pid}.", "en": en or f"English text of {pid}.", "tokens_nl": 10,
            "tokens_en": 8, "hard_fails": list(hard), "qe_en_nl": qe, "chrf": chrf}


def passage(pid, source="handelingen", split="train", score=0.9, origin=0.9, context_ids=(), speaker=None,
            party=None):
    return {"pid": pid, "source": source, "split": split, "source_score": score, "origin_score": origin,
            "context_ids": list(context_ids), "speaker": speaker, "party": party}


def turn(tid, speaker="De voorzitter", party=None, score=0.9):
    return {"id": tid, "speaker": speaker, "party": party, "source_score": score}


def test_query_fails_names_each_bound():
    p = passage("a")
    assert sb.query_fails(pair("a"), p, set()) == []
    assert sb.query_fails(pair("a", hard=["structure"]), p, set()) == ["hard_fails"]
    assert sb.query_fails(pair("a", qe=sb.QE_MAX), p, set()) == []
    assert sb.query_fails(pair("a", qe=sb.QE_MAX + 0.01), p, set()) == ["qe"]
    assert sb.query_fails(pair("a"), passage("a", score=0.49), set()) == ["source_score"]
    # the origin score counts for FineWeb-2 only
    assert sb.query_fails(pair("a"), passage("a", origin=0.1), set()) == []
    assert sb.query_fails(pair("a", source="fineweb"), passage("a", source="fineweb", origin=0.1), set()) == [
        "origin_score"]
    text = "one two three four five six seven eight nine ten eleven twelve thirteen fourteen"
    blocked = sb.sp.ngrams(text)
    assert sb.query_fails(pair("a", en=text), p, blocked) == ["eval_13gram"]
    assert sb.query_fails(pair("a", nl=text.upper()), p, blocked) == ["eval_13gram"]  # lowercased, as the selection
    # chrF has no bound
    assert sb.query_fails(pair("a", chrf=5.0), p, set()) == []


def test_context_keeps_the_turns_nearest_the_passage_up_to_one_that_fails():
    ids = ["h:d:t1", "h:d:t2", "h:d:t3"]
    pairs = {i: pair(i, kind="context", split=None) for i in ids}
    turns = {i: turn(i, speaker=f"De heer {i[-1]}", party="VVD") for i in ids}
    p = passage("h:d:t4:0", context_ids=ids)
    context, kept = sb.context_of(p, pairs, turns, set())
    assert kept == ids and [c["label"] for c in context] == ["De heer 1 (VVD)", "De heer 2 (VVD)", "De heer 3 (VVD)"]
    assert context[0] == {"label": "De heer 1 (VVD)", "source": "English text of h:d:t1.",
                          "translation": "Nederlandse tekst van h:d:t1."}
    pairs["h:d:t2"]["qe_en_nl"] = 99.0
    assert sb.context_of(p, pairs, turns, set())[1] == ["h:d:t3"]
    pairs["h:d:t2"]["qe_en_nl"] = 1.0
    turns["h:d:t3"]["source_score"] = 0.2
    assert sb.context_of(p, pairs, turns, set()) == ([], [])
    with pytest.raises(SystemExit):
        sb.context_of(passage("x", context_ids=["h:d:t9"]), pairs, turns, set())


def test_batched_counts_equal_single_ones():
    counter = sb.TokenCounter(CharTokenizer())
    users = [sb.user_turn("chat-v1-tag-token", "A text."), sb.user_turn("chat-v1", "Another one, longer.")]
    assert counter.prompt_lengths(users, batch=1) == [len(counter.prompt_ids(u)) for u in users]
    assert counter.target_lengths(["Een.", "Twee drie."]) == [len(counter.target_ids(t)) for t in ("Een.", "Twee drie.")]


def test_speaker_label():
    assert sb.speaker_label("De heer Thijssen", "PvdA") == "De heer Thijssen (PvdA)"
    assert sb.speaker_label("De voorzitter", None) == "De voorzitter"
    assert sb.speaker_label(None, "VVD") is None


@pytest.mark.parametrize("total, counts", [(10, {"a": 1, "b": 1, "c": 1}), (79, {"h": 889, "r": 468, "f": 580}),
                                           (0, {"a": 5})])
def test_allocate_sums_to_the_total_in_proportion(total, counts):
    out = sb.allocate(total, counts)
    assert sum(out.values()) == total
    whole = sum(counts.values())
    assert all(abs(out[k] - total * v / whole) < 1 for k, v in counts.items())


def test_the_draw_gives_context_its_share_and_the_sources_theirs():
    eligible = {"handelingen": [f"h{i}" for i in range(400)], "raad": [f"r{i}" for i in range(300)],
                "fineweb": [f"f{i}" for i in range(300)]}
    context = [f"h{i}" for i in range(0, 400, 4)]  # 100
    ctx, plain, info = sb.bt_draw(eligible, context, "auto", "s")
    assert info["bt_total"] == 500 and len(ctx) == 100 and info["context_share"] == 0.2
    assert len(ctx) + sum(len(v) for v in plain.values()) == 500
    assert {s: len(v) for s, v in plain.items()} == {"fineweb": 150, "handelingen": 100, "raad": 150}
    assert not set(ctx) & set(plain["handelingen"])
    ctx2, plain2, info2 = sb.bt_draw(eligible, context, "200", "s")
    assert len(ctx2) == 40 and info2["context_share"] == 0.2 and sum(map(len, plain2.values())) == 160
    assert (ctx2, plain2) == sb.bt_draw(eligible, context, "200", "s")[:2]  # seeded
    assert (ctx2, plain2) != sb.bt_draw(eligible, context, "200", "t")[:2]
    with pytest.raises(SystemExit):
        sb.bt_draw(eligible, context, "2000", "s")


def test_the_untagged_prompt_is_translategemmas_and_the_tags_change_one_place():
    en = "  The House agrees.  "
    plain = sb.user_turn("chat-v1", en)
    assert plain == TG_HEAD + "The House agrees."
    assert sb.user_turn("chat-v1-tag-token", en) == TG_HEAD + "<unused0>The House agrees."
    tagged = sb.user_turn("chat-v1-tag-instruction", en)
    assert tagged == plain.replace("or commentary. ", "or commentary. " + sb.TAG_SENTENCE + " ", 1)
    assert tagged.count(sb.TAG_SENTENCE) == 1 and tagged.index(sb.TAG_SENTENCE) < tagged.index("Please translate")
    context = [{"label": "De heer A (VVD)", "source": "Mr President.", "translation": "Voorzitter."}]
    with_context = sb.user_turn("chat-v1", en, context, "De voorzitter")
    assert '<current_turn speaker="De voorzitter">\nThe House agrees.\n</current_turn>' in with_context
    assert sb.user_turn("chat-v1-tag-token", en, context, "De voorzitter") == with_context.replace(
        '">\nThe House', '">\n<unused0>The House')
    assert sb.user_turn("chat-v1-tag-instruction", en, context, "De voorzitter") == with_context.replace(
        "or commentary.\n", "or commentary. " + sb.TAG_SENTENCE + "\n", 1)
    # without context the label plays no part
    assert sb.user_turn("chat-v1", en, [], "De voorzitter") == plain


def test_parse_sets():
    assert sb.parse_sets(["B=none", "N-token=token+replay"]) == {"B": ("none", False), "N-token": ("token", True)}
    for bad in ("B", "B=tag", "N=token+more", "=none"):
        with pytest.raises(SystemExit):
            sb.parse_sets([bad])


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def tables(tmp_path, tag_text=False):
    """A main run of 40 Handelingen and 20 FineWeb-2 passages (10 Handelingen with two context turns), a long run of
    10 raad passages, 4 eval passages, a replay of 100 pairs, two FLAN eval rows and one evaluation set."""
    run, long = tmp_path / "run", tmp_path / "long"
    pairs, passages, turns = [], [], []
    for i in range(40):
        pid = f"handelingen:d{i}:t5:0"
        ctx = [f"handelingen:d{i}:t3", f"handelingen:d{i}:t4"] if i < 10 else []
        pairs.append(pair(pid, qe=9.0 if i == 39 else 1.0))
        passages.append(passage(pid, context_ids=ctx, speaker="Mevrouw B", party="SP"))
        for t in ctx:
            pairs.append(pair(t, kind="context", split=None))
            turns.append(turn(t, speaker="De heer C", party="CDA"))
    for i in range(20):
        pid = f"fineweb:d{i}:0:0"
        pairs.append(pair(pid, source="fineweb", hard=["structure"] if i == 0 else ()))
        passages.append(passage(pid, source="fineweb"))
    for i in range(4):
        pid = f"handelingen:e{i}:t1:0"
        pairs.append(pair(pid, split="eval"))
        passages.append(passage(pid, split="eval"))
    if tag_text:
        pairs[0]["nl"] = "Een <unused0> in de tekst."
    write(run / "pairs.parquet", pairs)
    write(run / "passages.parquet", passages)
    write(run / "context-turns.parquet", turns)
    write(long / "pairs.parquet", [pair(f"raad:d{i}:0:0", source="raad") | {"length_class": "long"} for i in range(10)])
    write(long / "passages.parquet", [passage(f"raad:d{i}:0:0", source="raad") for i in range(10)])
    blocked = "the quick brown fox jumps over the lazy dog near the river bank today again"
    write(tmp_path / "replay.parquet",
          [{"id": i, "task": "flan/t", "en": blocked if i == 1 else f"Question {i}: what is it?",
            "nl": f"Vraag {i}: wat is het?", "tokens_en": 6, "tokens_nl": 6} for i in range(100)])
    write(tmp_path / "eval-rows.parquet", [{"set": "classifier", "host": 0, "position": 1, "source": "flan/t",
                                            "instruction_tokens": 60, "inputs": "Eval input.", "targets": "x",
                                            "segments": [0, 2]}])
    write(tmp_path / "evalsets.parquet", [{"id": "w1", "set": "wmt24pp", "source": blocked, "reference": "Ref.",
                                           "tokens": 14, "over_512": False, "length_class": "tiny", "meta": "{}"}])
    return argparse.Namespace(run=str(run), long=str(long), replay=str(tmp_path / "replay.parquet"),
                              flan_eval=str(tmp_path / "eval-rows.parquet"),
                              evalsets=str(tmp_path / "evalsets.parquet"), tokenizer="char", seed="t",
                              out=str(tmp_path / "out"))


def test_a_build_end_to_end(tmp_path, monkeypatch):
    args = tables(tmp_path)
    monkeypatch.setattr(sb, "CONTEXT_SHARE", 0.25)
    args.bt, args.sets = "auto", ["B=none", "N-token=token+replay", "N-instruction=instruction+replay"]
    sb.build(args, tokenizer=CharTokenizer())
    out = Path(args.out)
    meta = {s: pq.read_table(out / s / "train-meta.parquet").to_pylist() for s in ("B", "N-token", "N-instruction")}
    # 10 context examples at a share of 0.25: 40 back-translation examples; arm N adds a replay of 20%
    bt = {s: sorted(m["id"] for m in rows if m["kind"] != "replay") for s, rows in meta.items()}
    assert len(bt["B"]) == 40 and bt["B"] == bt["N-token"] == bt["N-instruction"]
    assert sum(m["kind"] == "bt_context" for m in meta["B"]) == 10
    replay = [m for m in meta["N-token"] if m["kind"] == "replay"]
    assert len(replay) == 10 and len(meta["N-token"]) == 50 and not any(m["kind"] == "replay" for m in meta["B"])
    # no eval segment of FLAN and no 13-gram of an evaluation set in the replay
    assert not {m["id"] for m in replay} & {"replay:0", "replay:1", "replay:2"}
    # the query: the hard fail and the QE outlier are out; eval stays eval
    ids = set(bt["B"])
    assert "fineweb:d0:0:0" not in ids and "handelingen:d39:t5:0" not in ids
    assert not any(i.startswith("handelingen:e") for i in ids)
    evals = pq.read_table(out / "B" / "eval-meta.parquet").to_pylist()
    assert sorted(m["id"] for m in evals) == [f"handelingen:e{i}:t1:0" for i in range(4)]
    # the sources keep their shares among the eligible: 39 Handelingen, 19 FineWeb-2, 10 raad of 68
    sources = {s: sum(m["source"] == s for m in meta["B"]) for s in ("handelingen", "fineweb", "raad")}
    assert sources == {"handelingen": 23, "fineweb": 11, "raad": 6}
    # the prompts: B untagged, N's back-translations tagged, its replay not; the messages are the meta's user turn
    for name, tag in (("B", "none"), ("N-token", "token"), ("N-instruction", "instruction")):
        msgs = pq.read_table(out / name / "train.parquet").column("messages").to_pylist()
        assert pq.read_table(out / name / "train.parquet").column_names == ["messages"]
        for m, row in zip(meta[name], msgs):
            assert row[0]["content"] == sb.user_turn(m["prompt"], m["en"], m["context"], m["label"])
            want = "none" if m["kind"] == "replay" else tag
            assert m["tag"] == want and m["prompt"] == sb.TAG_PROMPTS[want]
            assert (sb.TAG_TOKEN in row[0]["content"]) == (want == "token")
            assert (sb.TAG_SENTENCE in row[0]["content"]) == (want == "instruction")
            assert row[1]["content"].startswith(("Nederlandse tekst", "Vraag"))
            assert m["tokens_prompt"] + m["tokens_target"] <= sb.MAX_TOKENS
        ctx = [m for m in meta[name] if m["kind"] == "bt_context"]
        assert all(len(m["context"]) == 2 and m["label"] == "Mevrouw B (SP)" for m in ctx)
        assert (out / name / "template.jinja").read_text() == sb.TEMPLATE
    report = (out / "report.json").read_text()
    assert '"bt_total": 40' in report and (out / "report.md").exists()
    # the same seed, the same sets
    first = (out / "N-token" / "train.parquet").read_bytes()
    sb.build(args, tokenizer=CharTokenizer())
    assert (out / "N-token" / "train.parquet").read_bytes() == first


def test_a_build_refuses_data_with_the_tag_token_in_it(tmp_path):
    args = tables(tmp_path, tag_text=True)
    args.bt, args.sets = "auto", ["B=none"]
    with pytest.raises(SystemExit, match="tag token"):
        sb.build(args, tokenizer=CharTokenizer())


def test_a_context_that_makes_an_example_too_long_is_dropped_the_passage_stays(tmp_path, monkeypatch):
    args = tables(tmp_path)
    plain = len(TG_HEAD) + 2 + 40 + 60  # turn strings, the English and the Dutch with <end_of_turn>: well under
    monkeypatch.setattr(sb, "MAX_TOKENS", plain + 200)
    args.bt, args.sets = "20", ["B=none"]
    sb.build(args, tokenizer=CharTokenizer())
    report = (Path(args.out) / "report.json").read_text()
    assert '"train/context_too_long": 10' in report
    meta = pq.read_table(Path(args.out) / "B" / "train-meta.parquet").to_pylist()
    assert len(meta) == 20 and not any(m["kind"] == "bt_context" for m in meta)
