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


FLAN_TEMPLATE = "In this task you are given a question and you must answer it with one word only please"


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
          [{"id": i, "task": "flan/t", "en": blocked if i == 1 else FLAN_TEMPLATE if i == 3 else f"Question {i}: what is it?",
            "nl": f"Vraag {i}: wat is het?", "tokens_en": 6, "tokens_nl": 6} for i in range(100)])
    write(tmp_path / "eval-rows.parquet", [{"set": "classifier", "host": 0, "position": 1, "source": "flan/t",
                                            "instruction_tokens": 60, "inputs": FLAN_TEMPLATE + " Eval input.",
                                            "targets": "x",
                                            "segments": [0, 2]}])
    write(tmp_path / "evalsets.parquet", [{"id": "w1", "set": "wmt24pp", "source": blocked, "reference": "Ref.",
                                           "tokens": 14, "over_512": False, "length_class": "tiny", "meta": "{}"}])
    return argparse.Namespace(run=str(run), long=str(long), replay=str(tmp_path / "replay.parquet"),
                              flan_eval=str(tmp_path / "eval-rows.parquet"),
                              evalsets=str(tmp_path / "evalsets.parquet"), tokenizer="char", seed="t",
                              out=str(tmp_path / "out"), qe_max=sb.QE_MAX, replay_tag="none")


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
    # FLAN's evaluation rows keep the replay out by their segments; a template they share is counted, not blocked
    assert '"kept_sharing_a_13gram_with_flan_eval_rows": 1' in report and '"eval_13gram": 1' in report
    assert '"candidates": 97' in report
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


def test_the_qe_bound_is_a_parameter():
    p = passage("a")
    assert sb.query_fails(pair("a", qe=7.2), p, set(), qe_max=7.0) == ["qe"]
    assert sb.query_fails(pair("a", qe=7.2), p, set(), qe_max=7.5) == []
    ids = ["h:d:t1", "h:d:t2"]
    pairs = {i: pair(i, kind="context", split=None, qe=7.2 if i == "h:d:t1" else 1.0) for i in ids}
    turns = {i: turn(i) for i in ids}
    assert sb.context_of(passage("h:d:t3:0", context_ids=ids), pairs, turns, set(), qe_max=7.0)[1] == ["h:d:t2"]


def test_the_draw_of_all_takes_every_eligible_passage():
    eligible = {"handelingen": [f"h{i}" for i in range(400)], "raad": [f"r{i}" for i in range(300)]}
    context = [f"h{i}" for i in range(0, 400, 10)]  # 40
    ctx, plain, info = sb.bt_draw(eligible, context, "all", "s")
    assert info["bt_total"] == 700 and sorted(ctx) == sorted(context) and info["context_share"] == round(40 / 700, 4)
    assert sorted(ctx + plain["handelingen"] + plain["raad"]) == sorted(eligible["handelingen"] + eligible["raad"])


def test_the_conventional_tag_changes_one_place():
    en = "The House agrees."
    plain = sb.user_turn("chat-v1", en)
    tagged = sb.user_turn("chat-v1-tag-conventional", en)
    assert tagged == plain.replace("or commentary. ", "or commentary. " + sb.CONVENTIONAL_SENTENCE + " ", 1)
    assert sb.CONVENTIONAL_SENTENCE == "Translate closely, as a conventional translation."
    context = [{"label": "De heer A (VVD)", "source": "Mr President.", "translation": "Voorzitter."}]
    assert sb.user_turn("chat-v1-tag-conventional", en, context, "De voorzitter") == sb.user_turn(
        "chat-v1", en, context, "De voorzitter").replace("or commentary.\n", "or commentary. " + sb.CONVENTIONAL_SENTENCE
                                                         + "\n", 1)


def test_a_build_with_the_tag_turned_round(tmp_path):
    """Yeb, 10 Nov: every eligible passage, untagged; arm N adds a replay of 20% with the conventional sentence."""
    args = tables(tmp_path)
    args.bt, args.sets, args.replay_tag = "all", ["B=none", "N=none+replay"], "conventional"
    sb.build(args, tokenizer=CharTokenizer())
    out = Path(args.out)
    meta = {s: pq.read_table(out / s / "train-meta.parquet").to_pylist() for s in ("B", "N")}
    bt = {s: sorted(m["id"] for m in rows if m["kind"] != "replay") for s, rows in meta.items()}
    # 39 Handelingen (one QE outlier out), 19 FineWeb-2 (one hard fail out), 10 raad: all 68, 10 with context
    assert len(bt["B"]) == 68 and bt["B"] == bt["N"]
    assert sum(m["kind"] == "bt_context" for m in meta["B"]) == 10
    replay = [m for m in meta["N"] if m["kind"] == "replay"]
    assert len(replay) == 17 and len(meta["N"]) == 85  # 17 = 68 x 0.25: 20% of 85
    for name in ("B", "N"):
        msgs = pq.read_table(out / name / "train.parquet").column("messages").to_pylist()
        for m, row in zip(meta[name], msgs):
            want = "conventional" if m["kind"] == "replay" else "none"
            assert m["tag"] == want and m["prompt"] == sb.TAG_PROMPTS[want]
            assert (sb.CONVENTIONAL_SENTENCE in row[0]["content"]) == (want == "conventional")
            assert sb.TAG_SENTENCE not in row[0]["content"] and sb.TAG_TOKEN not in row[0]["content"]
            assert row[0]["content"] == sb.user_turn(m["prompt"], m["en"], m["context"], m["label"])
    report = __import__("json").loads((out / "report.json").read_text())
    assert report["query"]["qe_en_nl_max"] == sb.QE_MAX and report["replay_tag"] == "conventional"
    train = report["sets"]["N"]["train"]
    assert sum(train["tokens_by_source"].values()) == train["tokens"]
    assert sum(train["loss_tokens_by_source"].values()) == train["loss_tokens"]
    assert set(train["tokens_by_source"]) == {"handelingen", "fineweb", "raad", "replay"}
    assert "## Tokens per bron" in (out / "report.md").read_text()


def test_a_lower_qe_bound_gives_a_smaller_set(tmp_path):
    args = tables(tmp_path)
    args.bt, args.sets, args.qe_max = "all", ["B=none"], 8.0  # every pair of the tables has QE 1.0 or 9.0
    sb.build(args, tokenizer=CharTokenizer())
    assert len(pq.read_table(Path(args.out) / "B" / "train-meta.parquet")) == 68
    args.qe_max, args.out = 10.0, str(tmp_path / "out10")
    sb.build(args, tokenizer=CharTokenizer())
    assert len(pq.read_table(Path(args.out) / "B" / "train-meta.parquet")) == 69  # the outlier with QE 9.0 joins


# ---------------------------------------------------------------------------------------------------- arm R2

def test_the_length_bound_is_a_factor_around_the_sources_median():
    rows = [pair(f"h{i}", nl="x" * n, en="y" * 100) for i, n in enumerate((80, 90, 100, 110, 120))]
    bounds = sb.length_bounds(rows, 1.25)
    assert bounds == {"handelingen": {"median": 1.0, "lo": 0.8, "hi": 1.25, "n": 5}}
    p = passage("a")
    assert sb.query_fails(pair("a", nl="x" * 80, en="y" * 100), p, set(), bounds=bounds) == []
    assert sb.query_fails(pair("a", nl="x" * 79, en="y" * 100), p, set(), bounds=bounds) == ["length"]
    assert sb.query_fails(pair("a", nl="x" * 126, en="y" * 100), p, set(), bounds=bounds) == ["length"]
    assert sb.query_fails(pair("a", nl="x" * 126, en="y" * 100), p, set()) == []  # no bound without --len-factor
    with pytest.raises(SystemExit, match="no length bound"):
        sb.query_fails(pair("a", source="raad"), passage("a", source="raad"), set(), bounds=bounds)
    assert sb.word_ratio("een twee drie", "one two") == 1.5 and sb.char_ratio(" abcd ", "ab") == 2.0


def test_parse_sets_with_real_pairs():
    assert sb.parse_sets(["R2=none+real"]) == {"R2": ("none", "real")}


def test_the_draw_of_real_pairs_takes_every_global_voices_pair_first():
    eligible = [{"id": f"gv:{i}", "source": "globalvoices"} for i in range(5)] + \
        [{"id": f"europarl:{i}", "source": "europarl"} for i in range(50)]
    chosen, info = sb.real_draw(eligible, 12, "s")
    assert info["chosen"] == {"globalvoices": 5, "europarl": 7}
    assert {r["id"] for r in chosen} >= {f"gv:{i}" for i in range(5)}
    assert sb.real_draw(eligible, 100, "s")[1]["chosen"] == {"globalvoices": 5, "europarl": 50}  # fewer: all
    assert sb.real_draw(eligible, 3, "s")[1]["chosen"] == {"globalvoices": 3}
    assert [r["id"] for r in sb.real_draw(eligible, 12, "s")[0]] == [r["id"] for r in chosen]  # the seed


def real_pairs_table(path, with_qe=True):
    rows = []
    for i in range(30):
        source = "globalvoices" if i < 4 else "europarl"
        en, nl = f"The real English sentence number {i} here.", f"De echte Nederlandse zin nummer {i} hier."
        fails = []
        if i == 5:
            fails = ["short"]
        if i == 6:
            nl = nl * 3  # a length outlier
        if i == 7:  # an evaluation 13-gram (the Dutch as long, so that the length bound does not take it first)
            en = "the quick brown fox jumps over the lazy dog near the river bank today again"
            nl = "de snelle bruine vos springt over de luie hond bij de rivieroever vandaag weer"
        if i == 8:  # a blind passage's 13-gram
            en = "one two three four five six seven eight nine ten eleven twelve thirteen"
            nl = "een twee drie vier vijf zes zeven acht negen tien elf twaalf dertien"
        row = {"id": f"{source}:{i}", "source": source, "en": en, "nl": nl, "fails": fails}
        if with_qe:
            row["qe_en_nl"] = None if fails else (9.0 if i == 9 else 1.0)
        rows.append(row)
    write(path, rows)


def test_a_build_of_arm_r2(tmp_path):
    """BT with the length bound and a lower QE bound; real pairs at most 20%, Global Voices first, in TranslateGemma's
    own prompt; the blind passages' 13-grams blocked."""
    args = tables(tmp_path)
    real_pairs_table(tmp_path / "real.parquet")
    (tmp_path / "blind.jsonl").write_text(__import__("json").dumps(
        {"blind_id": "b1", "source": "one two three four five six seven eight nine ten eleven twelve thirteen"}) + "\n")
    args.bt, args.sets, args.qe_max = "all", ["R2=none+real"], 5.0
    args.len_factor, args.real, args.real_share = 1.25, str(tmp_path / "real.parquet"), 0.2
    args.block_jsonl, args.real_without_qe = str(tmp_path / "blind.jsonl"), False
    sb.build(args, tokenizer=CharTokenizer())
    out = Path(args.out)
    meta = pq.read_table(out / "R2" / "train-meta.parquet").to_pylist()
    msgs = pq.read_table(out / "R2" / "train.parquet").column("messages").to_pylist()
    bt = [m for m in meta if m["kind"] in ("bt", "bt_context")]
    real = [m for m in meta if m["kind"] == "real"]
    assert len(bt) == 68  # every pair of the tables has the same length: the bound costs nothing here
    # 17 = 68 x 0.25 at most; eligible: 4 Global Voices, 21 Europarl (short, length, 13-gram, blind, QE 9 out)
    assert len(real) == 17 and sum(m["source"] == "globalvoices" for m in real) == 4
    assert not {m["id"] for m in real} & {"europarl:5", "europarl:6", "europarl:7", "europarl:8", "europarl:9"}
    for m, row in zip(meta, msgs):
        if m["kind"] == "real":
            assert m["tag"] == "none" and m["prompt"] == "chat-v1" and m["context"] == []
            assert row[0]["content"] == sb.user_turn("chat-v1", m["en"]) and row[0]["content"].startswith(TG_HEAD)
            assert sb.CONVENTIONAL_SENTENCE not in row[0]["content"]
    report = __import__("json").loads((out / "report.json").read_text())
    real_cost = report["real"]["query_cost"]["first_failed"]
    assert real_cost == {"europarl/hard:short": 1, "europarl/length": 1, "europarl/eval_13gram": 2, "europarl/qe": 1}
    assert report["real"]["draw"]["chosen"] == {"globalvoices": 4, "europarl": 13}
    assert report["sets"]["R2"]["train"]["real_share"] == round(17 / 85, 4)
    lengths = report["sets"]["R2"]["train"]["length_ratio"]
    assert set(lengths) >= {"bt", "real", "all", "source/globalvoices", "source/handelingen"}
    assert lengths["real"]["n"] == 17 and lengths["all"]["n"] == 85
    assert report["query"]["len_factor"] == 1.25 and report["length_bounds_bt"]["handelingen"]["median"] > 0
    md = (out / "report.md").read_text()
    assert "## Echte vertaalparen" in md and "## Lengteverhouding" in md and "Proefbouw" not in md
    manifest = __import__("json").loads((out / "R2" / "manifest.json").read_text())
    assert manifest["real"] is True and manifest["real_without_qe"] is False
    # eval stays the back-translations'
    assert all(m["kind"].startswith("bt") for m in pq.read_table(out / "R2" / "eval-meta.parquet").to_pylist())


def test_real_pairs_without_qe_only_in_a_dry_run(tmp_path):
    args = tables(tmp_path)
    real_pairs_table(tmp_path / "real.parquet", with_qe=False)
    args.bt, args.sets, args.qe_max = "all", ["R2=none+real"], 5.0
    args.len_factor, args.real, args.real_share = 1.25, str(tmp_path / "real.parquet"), 0.2
    args.block_jsonl, args.real_without_qe = None, False
    with pytest.raises(SystemExit, match="no qe_en_nl"):
        sb.build(args, tokenizer=CharTokenizer())
    args.real_without_qe = True
    sb.build(args, tokenizer=CharTokenizer())
    assert "Proefbouw" in (Path(args.out) / "report.md").read_text()
    # a set with +real needs --real, and --real a set with +real
    args.sets = ["B=none"]
    with pytest.raises(SystemExit, match="--real goes with"):
        sb.build(args, tokenizer=CharTokenizer())


def test_a_build_of_arm_rb_takes_every_eligible_real_pair(tmp_path):
    """Arm RB: R2's query on BT and real pairs, but --real-all takes every eligible real pair (the Dutch books), above
    the 20% of R2"""
    args = tables(tmp_path)
    rows = []
    for i in range(40):
        en, nl = f"The English of book passage {i} here.", f"Het Nederlands van boekpassage {i} hier."
        fails = ["not_dutch_original"] if i == 0 else []
        qe = None if fails else (7.0 if i == 1 else 1.0)
        rows.append({"id": f"books:{i}", "source": "books", "en": en, "nl": nl, "fails": fails, "qe_en_nl": qe})
    write(tmp_path / "real.parquet", rows)
    args.bt, args.sets, args.qe_max = "all", ["RB=none+real"], 5.0
    args.len_factor, args.real, args.real_share, args.real_all = 1.25, str(tmp_path / "real.parquet"), 0.2, True
    args.block_jsonl, args.real_without_qe = None, False
    sb.build(args, tokenizer=CharTokenizer())
    out = Path(args.out)
    meta = pq.read_table(out / "RB" / "train-meta.parquet").to_pylist()
    real = [m for m in meta if m["kind"] == "real"]
    assert len([m for m in meta if m["kind"] in ("bt", "bt_context")]) == 68
    # 38 of 40: not a Dutch original, QE 7 out; more than R2's 17 (68 x 0.25)
    assert len(real) == 38 and {m["source"] for m in real} == {"books"}
    assert not {m["id"] for m in real} & {"books:0", "books:1"}
    report = __import__("json").loads((out / "report.json").read_text())
    assert report["real"]["share"] == "all" and report["real"]["draw"]["chosen"] == {"books": 38}
    assert report["real"]["query_cost"]["first_failed"] == {"books/hard:not_dutch_original": 1, "books/qe": 1}
    assert report["sets"]["RB"]["train"]["real_share"] == round(38 / 106, 4)
    assert "`--real-all`" in (out / "report.md").read_text()
    # --real-all and a --real-share are not both given
    with pytest.raises(SystemExit):
        sb.main(["build", "--run", "r", "--long", "l", "--replay", "x", "--flan-eval", "f", "--evalsets", "e",
                 "--tokenizer", "t", "--sets", "RB=none+real", "--real", "x", "--real-all", "--real-share", "0.3",
                 "--out", "o"])


def test_a_build_of_arm_rb2_draws_four_back_translations_per_book_pair(tmp_path):
    """Arm RB2: the books with their own QE bound (7.5, the back-translations at 5) and every eligible one of them; the
    back-translation examples drawn from every eligible one, 4 per book pair (80 : 20), the same draw for the same seed"""
    args = tables(tmp_path)
    rows = []
    for i in range(16):
        en, nl = f"The English of book passage {i} here.", f"Het Nederlands van boekpassage {i} hier."
        fails = ["not_dutch_original"] if i == 0 else []
        qe = None if fails else {1: 8.0, 2: 7.0, 3: 6.0}.get(i, 1.0)
        rows.append({"id": f"books:{i}", "source": "books", "en": en, "nl": nl, "fails": fails, "qe_en_nl": qe})
    write(tmp_path / "real.parquet", rows)
    args.bt, args.sets, args.qe_max = "all", ["RB2=none+real"], 5.0
    args.len_factor, args.real, args.real_share, args.real_all = 1.25, str(tmp_path / "real.parquet"), 0.2, True
    args.block_jsonl, args.real_without_qe, args.real_qe_max, args.bt_per_real = None, False, 7.5, 4.0
    sb.build(args, tokenizer=CharTokenizer())
    out = Path(args.out)
    meta = pq.read_table(out / "RB2" / "train-meta.parquet").to_pylist()
    real = [m for m in meta if m["kind"] == "real"]
    bt = [m for m in meta if m["kind"] in ("bt", "bt_context")]
    # 14 of 16 books: not a Dutch original, QE 8 out; QE 7 and 6 in (they were out at 5)
    assert len(real) == 14 and not {m["id"] for m in real} & {"books:0", "books:1"}
    assert {"books:2", "books:3"} <= {m["id"] for m in real}
    # 56 = 4 x 14 of the 68 eligible back-translations; the QE outlier and the hard fail stay out
    assert len(bt) == 56 and len({m["id"] for m in bt}) == 56
    assert not {m["id"] for m in bt} & {"fineweb:d0:0:0", "handelingen:d39:t5:0"}
    # a passage drawn with its context keeps it
    assert all(len(m["context"]) == 2 for m in bt if m["kind"] == "bt_context")
    assert all(m["id"].startswith("handelingen:d") and int(m["id"].split(":")[1][1:]) < 10
               for m in bt if m["kind"] == "bt_context")
    report = __import__("json").loads((out / "report.json").read_text())
    assert report["real"]["qe_max"] == 7.5 and report["query"]["qe_en_nl_max"] == 5.0
    assert report["query"]["real_qe_en_nl_max"] == 7.5
    assert report["real"]["query_cost"]["first_failed"] == {"books/hard:not_dutch_original": 1, "books/qe": 1}
    draw = report["bt"]["per_real_draw"]
    assert (draw["per_real"], draw["real"], draw["drawn"], draw["of"]) == (4.0, 14, 56, 68)
    assert draw["context"] == sum(m["kind"] == "bt_context" for m in bt) and sum(draw["drawn_by_source"].values()) == 56
    assert report["sets"]["RB2"]["train"]["real_share"] == 0.2
    assert "`--bt-per-real 4.0`" in (out / "report.md").read_text()
    # the same seed, the same draw; another seed, another draw of the same size
    first = sorted(m["id"] for m in bt)
    sb.build(args, tokenizer=CharTokenizer())
    assert sorted(m["id"] for m in pq.read_table(out / "RB2" / "train-meta.parquet").to_pylist()
                  if m["kind"] != "real") == first
    args.seed, args.out = "other", str(tmp_path / "out-other")
    sb.build(args, tokenizer=CharTokenizer())
    other = sorted(m["id"] for m in pq.read_table(Path(args.out) / "RB2" / "train-meta.parquet").to_pylist()
                   if m["kind"] != "real")
    assert len(other) == 56 and other != first


def test_bt_per_real_needs_enough_back_translations_and_every_real_pair(tmp_path):
    rows = [{"pid": f"p{i}", "source": "raad"} for i in range(10)]
    kept, info = sb.bt_per_real_draw(rows, 2, 4, "s")
    assert len(kept) == 8 and [r["pid"] for r in kept] == sorted(r["pid"] for r in kept)
    assert info["drawn_by_source"] == {"raad": 8}
    with pytest.raises(SystemExit, match="12 back-translation examples wanted for 3 real pairs, 10 eligible"):
        sb.bt_per_real_draw(rows, 3, 4, "s")
    base = ["build", "--run", "r", "--long", "l", "--replay", "x", "--flan-eval", "f", "--evalsets", "e",
            "--tokenizer", "t", "--sets", "RB2=none+real", "--real", "x", "--out", "o"]
    for extra in (["--bt", "all", "--bt-per-real", "4"], ["--bt", "auto", "--real-all", "--bt-per-real", "4"],
                  ["--bt", "all", "--real-all", "--bt-per-real", "0"],
                  ["--bt", "all", "--real-all", "--real-without-qe", "--real-qe-max", "7.5"]):
        with pytest.raises(SystemExit):
            sb.main(base + extra)
    # in build too (a set without +real would lose its back-translations to the draw)
    args = tables(tmp_path)
    write(tmp_path / "real.parquet", [{"id": "books:1", "source": "books", "en": "The English of a book passage.",
                                       "nl": "Het Nederlands van een boekpassage.", "fails": [], "qe_en_nl": 1.0}])
    args.bt, args.sets, args.qe_max = "all", ["B=none", "RB2=none+real"], 5.0
    args.len_factor, args.real, args.real_share, args.real_all = None, str(tmp_path / "real.parquet"), 0.2, True
    args.block_jsonl, args.real_without_qe, args.real_qe_max, args.bt_per_real = None, False, 7.5, 4.0
    with pytest.raises(SystemExit, match="sets with \\+real only"):
        sb.build(args, tokenizer=CharTokenizer())


def books_table(path, n=12):
    """n book pairs: books:0 not a Dutch original, books:1 at QE 8 (out at 7.5), the rest in"""
    rows = []
    for i in range(n):
        en, nl = f"The English of book passage {i} here.", f"Het Nederlands van boekpassage {i} hier."
        fails = ["not_dutch_original"] if i == 0 else []
        qe = None if fails else (8.0 if i == 1 else 6.0 if i == 2 else 1.0)
        rows.append({"id": f"books:{i}", "source": "books", "en": en, "nl": nl, "fails": fails, "qe_en_nl": qe})
    write(path, rows)


def test_a_build_of_arm_rbr_adds_r2s_real_pairs_to_the_books(tmp_path):
    """Arm RBR: the books as RB2 (their own QE bound 7.5, all of them), R2's real pairs as a second source (R2's query at
    the back-translations' QE bound 5; only the real examples of R2's set, drawn uniformly, 1 per book pair) and 3
    back-translations per book pair: 60 : 20 : 20"""
    import json

    args = tables(tmp_path)
    real_pairs_table(tmp_path / "r2-real.parquet")
    args.bt, args.sets, args.qe_max = "all", ["R2=none+real"], 5.0
    args.len_factor, args.real, args.real_share = 1.25, str(tmp_path / "r2-real.parquet"), 0.2
    args.block_jsonl, args.real_without_qe, args.out = None, False, str(tmp_path / "out-r2")
    sb.build(args, tokenizer=CharTokenizer())
    r2_meta = tmp_path / "out-r2" / "R2" / "train-meta.parquet"
    r2_real = sorted(m["id"] for m in pq.read_table(r2_meta).to_pylist() if m["kind"] == "real")
    assert len(r2_real) == 17  # 68 x 0.25: 4 Global Voices, 13 Europarl

    books_table(tmp_path / "books.parquet")
    args = tables(tmp_path)
    args.bt, args.sets, args.qe_max, args.out = "all", ["RBR=none+real"], 5.0, str(tmp_path / "out-rbr")
    args.len_factor, args.real, args.real_share, args.real_all = 1.25, str(tmp_path / "books.parquet"), 0.2, True
    args.block_jsonl, args.real_without_qe, args.real_qe_max, args.bt_per_real = None, False, 7.5, 3.0
    args.real2, args.real2_from, args.real2_per_real = str(tmp_path / "r2-real.parquet"), str(r2_meta), 1.0
    sb.build(args, tokenizer=CharTokenizer())
    out = Path(args.out)
    meta = pq.read_table(out / "RBR" / "train-meta.parquet").to_pylist()
    msgs = pq.read_table(out / "RBR" / "train.parquet").column("messages").to_pylist()
    books = [m for m in meta if m["kind"] == "real" and m["source"] == "books"]
    r2 = [m for m in meta if m["kind"] == "real" and m["source"] != "books"]
    bt = [m for m in meta if m["kind"] in ("bt", "bt_context")]
    # 10 of 12 books (not a Dutch original, QE 8 out; QE 6 in at 7.5); 10 of R2's 17; 30 = 3 x 10 back-translations
    assert len(books) == 10 and {"books:2"} <= {m["id"] for m in books}
    assert len(r2) == 10 and {m["id"] for m in r2} <= set(r2_real) and {m["source"] for m in r2} <= {"globalvoices", "europarl"}
    assert len(bt) == 30 and len(meta) == 50
    for m, row in zip(meta, msgs):
        if m["kind"] == "real":
            assert m["prompt"] == "chat-v1" and row[0]["content"] == sb.user_turn("chat-v1", m["en"])
    report = json.loads((out / "report.json").read_text())
    d = report["real2"]["draw"]
    assert (d["per_real"], d["real"], d["drawn"], d["of"]) == (1.0, 10, 10, 17)
    assert d["from"] == {"europarl": 13, "globalvoices": 4} and sum(d["chosen"].values()) == 10
    # R2's bound on its own pairs (QE 9 out at 5), the books' own bound
    assert report["real2"]["qe_max"] == 5.0 and report["real"]["qe_max"] == 7.5
    assert report["query"]["real2_qe_en_nl_max"] == 5.0
    assert report["real2"]["query_cost"]["first_failed"] == {"europarl/hard:short": 1, "europarl/length": 1,
                                                             "europarl/eval_13gram": 1, "europarl/qe": 1}
    assert report["bt"]["per_real_draw"]["drawn"] == 30 and report["sets"]["RBR"]["train"]["real_share"] == 0.4
    assert report["inputs"]["real2_from"]["path"] == str(r2_meta)
    assert "## Tweede bron van echte paren" in (out / "report.md").read_text()
    # the same seed, the same draw
    first = sorted(m["id"] for m in meta)
    sb.build(args, tokenizer=CharTokenizer())
    assert sorted(m["id"] for m in pq.read_table(out / "RBR" / "train-meta.parquet").to_pylist()) == first


def test_rbr_refuses_r2_pairs_that_fail_its_query_and_too_few_of_them(tmp_path):
    args = tables(tmp_path)
    real_pairs_table(tmp_path / "r2-real.parquet")
    books_table(tmp_path / "books.parquet")
    args.bt, args.sets, args.qe_max = "all", ["RBR=none+real"], 5.0
    args.len_factor, args.real, args.real_share, args.real_all = 1.25, str(tmp_path / "books.parquet"), 0.2, True
    args.block_jsonl, args.real_without_qe, args.real_qe_max, args.bt_per_real = None, False, 7.5, 3.0
    args.real2, args.real2_per_real = str(tmp_path / "r2-real.parquet"), 1.0
    # europarl:9 has QE 9: in a set built with another bound, not here
    write(tmp_path / "meta-bad.parquet", [{"id": i, "kind": "real"} for i in ("globalvoices:0", "europarl:9")])
    args.real2_from = str(tmp_path / "meta-bad.parquet")
    with pytest.raises(SystemExit, match="1 of its 2 real pairs do not pass the query here"):
        sb.build(args, tokenizer=CharTokenizer())
    # 10 wanted (1 per book pair), 3 in the set
    write(tmp_path / "meta-few.parquet", [{"id": f"globalvoices:{i}", "kind": "real"} for i in range(3)]
          + [{"id": "handelingen:d1:t5:0", "kind": "bt"}])
    args.real2_from = str(tmp_path / "meta-few.parquet")
    with pytest.raises(SystemExit, match="10 pairs wanted for 10 real pairs, 3 in --real2-from"):
        sb.build(args, tokenizer=CharTokenizer())
    # the same id in both sources: books:3 also in the second table, and drawn (all 16 of its set: 1.6 per book pair)
    rows = pq.read_table(tmp_path / "r2-real.parquet").to_pylist()
    write(tmp_path / "r2-books.parquet", rows + [{"id": "books:3", "source": "europarl", "en": "The English of book "
                                                  "passage 3 here.", "nl": "Het Nederlands van boekpassage 3 hier.",
                                                  "fails": [], "qe_en_nl": 1.0}])
    write(tmp_path / "meta-ok.parquet", [{"id": i, "kind": "real"} for i in
                                         ["books:3"] + [f"europarl:{i}" for i in range(10, 25)]])
    args.real2, args.real2_from, args.real2_per_real = str(tmp_path / "r2-books.parquet"), str(tmp_path / "meta-ok.parquet"), 1.6
    with pytest.raises(SystemExit, match="share an id"):
        sb.build(args, tokenizer=CharTokenizer())
    # the command line: --real2 with --real2-from and a positive --real2-per-real, --real and --real-all
    base = ["build", "--run", "r", "--long", "l", "--replay", "x", "--flan-eval", "f", "--evalsets", "e",
            "--tokenizer", "t", "--sets", "RBR=none+real", "--real", "x", "--bt", "all", "--out", "o"]
    for extra in (["--real-all", "--real2", "y", "--real2-per-real", "1"],
                  ["--real-all", "--real2", "y", "--real2-from", "m"],
                  ["--real-all", "--real2", "y", "--real2-from", "m", "--real2-per-real", "0"],
                  ["--real2", "y", "--real2-from", "m", "--real2-per-real", "1"],
                  ["--real-all", "--real-without-qe", "--real2", "y", "--real2-from", "m", "--real2-per-real", "1"],
                  ["--real-all", "--real2-from", "m", "--real2-per-real", "1"]):
        with pytest.raises(SystemExit):
            sb.main(base + extra)


# Arm RL (protocol §9, amendment of 12 Nov ~15:45): the books without a QE bound, with a check of their alignment, and
# the back-translations drawn to the word ratio of R2's real pairs

EN_SENTENCES = ["The sexton came out of the sacristy and asked something of someone in the front row.",
                "He turned around.",
                "Onno felt himself stiffening and immediately thought of Quinten, who was far away in the mountains of "
                "the south.",
                "Tell me what has happened, he said, and sat down on the bench near the door of the old church."]
NL_SENTENCES = ["De koster kwam uit de sacristie en vroeg iets aan iemand op de voorste rij van de kerk.",
                "Die draaide zich om.",
                "Onno voelde zich verstijven en dacht meteen aan Quinten, die ver weg was in de bergen van het zuiden van "
                "het land.",
                "Zeg het maar, wat is er gebeurd, zei hij, en ging zitten op de bank bij de deur van de oude kerk aan het "
                "plein."]
NL_EXTRA = "Hij zou zich natuurlijk weer helemaal moeten inwerken, ook de vakliteratuur had hij niet meer bijgehouden."


def test_sentences_end_at_a_full_stop_but_not_after_an_abbreviation_an_initial_or_a_note_number():
    text = ("If Mr. L. was prepared to wait, fine. 'No.' She left.\n5. Stuart Cloete, African Portraits.\n"
            "He said: \"Go!\" Then _silence._ And more… who knows? Mevr. Jansen bijv. niet.")
    assert sb.sentences(text) == ["If Mr. L. was prepared to wait, fine.", "'No.'", "She left.",
                                  "5. Stuart Cloete, African Portraits.", 'He said: "Go!"', "Then _silence._",
                                  "And more… who knows?", "Mevr. Jansen bijv. niet."]
    assert sb.sentences("\n\n  One line.  \n") == ["One line."]


def test_the_alignment_check_sees_a_passage_shifted_by_a_sentence():
    en = " ".join(EN_SENTENCES)
    aligned = sb.align_edges(en, " ".join(NL_SENTENCES), 1.033)
    # the Dutch begins with a sentence the English lacks and lacks the English's last: about as long in total
    shifted = sb.align_edges(en, " ".join([NL_EXTRA] + NL_SENTENCES[:3]), 1.033)
    assert max(aligned) < 1.0 and max(shifted) > 3.0
    beads = sb.sentence_alignment(sb.sentences(en), sb.sentences(" ".join(NL_SENTENCES)), 1.033)
    assert [(a, b) for a, b, _ in beads] == [(1, 1)] * 4
    # a sentence the translator split in two is a 1-2 bead, at a low cost
    split = NL_SENTENCES[2].replace(", die ver", ". Die was ver").replace("die ver weg was", "ver weg")
    beads = sb.sentence_alignment(sb.sentences(en), sb.sentences(" ".join(NL_SENTENCES[:2] + [split, NL_SENTENCES[3]])),
                                  1.033)
    assert [(a, b) for a, b, _ in beads] == [(1, 1), (1, 1), (1, 2), (1, 1)]
    # one sentence a side: the one bead is both edges
    start, end = sb.align_edges(EN_SENTENCES[0], NL_SENTENCES[0], 1.033)
    assert start == end < 1.0
    with pytest.raises(ValueError):
        sb.sentence_alignment([], ["Een zin."], 1.0)
    assert [sb.qe_band(q) for q in (1.0, 5.0, 5.01, 7.5, 9.0, 23.0, 25.0)] == \
        ["<=5", "<=5", "5-7.5", "5-7.5", "7.5-10", "20-23", ">23"]


def ratio_rows(spec):
    """back-translation rows with a given ratio of Dutch to English words: spec [(pid, words_nl, words_en)]"""
    return [{"pid": p, "source": "raad", "nl": " ".join(["nl"] * a), "en": " ".join(["en"] * b)} for p, a, b in spec]


def test_the_matching_draw_follows_the_target_and_takes_what_fits():
    target = [("europarl", 1.0)] * 10 + [("globalvoices", 1.5)] * 10
    pool = ratio_rows([(f"lo{i:02d}", 4, 4) for i in range(30)] + [(f"hi{i}", 6, 4) for i in range(6)])
    # two bins at the target's median; 8 wanted: 4 from each
    kept, info = sb.bt_match_draw(pool, target, 8, 2, "s")
    assert len(kept) == 8 and sum(r["pid"].startswith("hi") for r in kept) == 4 and not info["limited"]
    assert [b["drawn"] for b in info["per_bin"]] == [4, 4] and [b["pool"] for b in info["per_bin"]] == [30, 6]
    assert [r["pid"] for r in kept] == [r["pid"] for r in pool if r["pid"] in {k["pid"] for k in kept}]
    assert info["ratio"]["drawn"]["p50"] == 1.25 and info["ratio"]["target"]["n"] == 20
    assert info["target_by_source"] == {"europarl": 10, "globalvoices": 10}
    # 40 wanted: the upper bin holds 6, half of the target, so at most 12
    kept, info = sb.bt_match_draw(pool, target, 40, 2, "s")
    assert (len(kept), info["most"], info["drawn"], info["limited"]) == (12, 12, 12, True)
    assert sum(r["pid"].startswith("hi") for r in kept) == 6
    # the same seed, the same draw; another seed, another one
    first = [r["pid"] for r in sb.bt_match_draw(pool, target, 8, 2, "s")[0]]
    assert first == [r["pid"] for r in sb.bt_match_draw(pool, target, 8, 2, "s")[0]]
    assert first != [r["pid"] for r in sb.bt_match_draw(pool, target, 8, 2, "other")[0]]
    # a bin of the target without back-translations: no draw
    with pytest.raises(SystemExit, match="no back-translation in bin"):
        sb.bt_match_draw(ratio_rows([(f"lo{i}", 4, 4) for i in range(30)]), target, 8, 2, "s")
    # equal cuts merge: a target of one ratio is one bin
    kept, info = sb.bt_match_draw(pool, [("europarl", 1.0)] * 20, 8, 10, "s")
    assert info["bins"] == 2 and info["bins_asked"] == 10 and len(kept) == 8


def rl_tables(tmp_path):
    """tables() with back-translations of three word ratios (Handelingen d10-d15 at 1.5, d16-d19 at 0.75, the rest at
    1.0), books (aligned, shifted, high QE, not a Dutch original) and a set R2 of 20 real examples at 1.0 and 1.5"""
    args = tables(tmp_path)
    path = Path(args.run) / "pairs.parquet"
    rows = pq.read_table(path).to_pylist()
    for r in rows:
        if r["kind"] == "passage" and r["pid"].startswith("handelingen:d"):
            i = int(r["pid"].split(":")[1][1:])
            if 10 <= i <= 15:
                r["nl"] = f"Een twee drie vier vijf {r['pid']}."
            elif 16 <= i <= 19:
                r["nl"] = f"Nederlandse versie {r['pid']}."
    write(path, rows)
    books = []
    for i in range(12):
        nl = NL_SENTENCES[:3] + [NL_SENTENCES[3] + f" ({i})"]
        if i == 1:  # shifted
            nl = [NL_EXTRA] + NL_SENTENCES[:3]
        books.append({"id": f"books:{i}", "source": "books", "en": " ".join(EN_SENTENCES[:3] + [EN_SENTENCES[3] + f" ({i})"]),
                      "nl": " ".join(nl), "fails": ["not_dutch_original"] if i == 0 else [],
                      "qe_en_nl": None if i == 0 else 20.0 if i == 2 else 1.0})
    write(tmp_path / "books.parquet", books)
    real = [{"en": "one two three four", "nl": "een twee drie vier", "source": "europarl"}] * 10 + \
        [{"en": "one two three four", "nl": "een twee drie vier vijf zes", "source": "globalvoices"}] * 10
    write(tmp_path / "r2set" / "train.parquet",
          [{"messages": [{"role": "user", "content": "u"}, {"role": "assistant", "content": r["nl"]}]} for r in real]
          + [{"messages": [{"role": "user", "content": "u"}, {"role": "assistant", "content": "x"}]}])
    write(tmp_path / "r2set" / "train-meta.parquet",
          [{"id": f"r{k}", "kind": "real", "source": r["source"], "en": r["en"]} for k, r in enumerate(real)]
          + [{"id": "bt1", "kind": "bt", "source": "raad", "en": "one two three four five six seven eight"}])
    args.bt, args.sets, args.qe_max, args.len_factor = "all", ["RL=none+real"], 5.0, 1.25
    args.real, args.real_share, args.real_all, args.block_jsonl = str(tmp_path / "books.parquet"), 0.2, True, None
    args.real_without_qe, args.real_qe_max, args.real_qe_unbounded, args.real_align_max = False, None, True, 2.0
    args.bt_per_real, args.bt_match_set, args.bt_match_bins = 4.0, str(tmp_path / "r2set"), 2
    return args


def test_a_build_of_arm_rl(tmp_path):
    """Arm RL: the books without a QE bound but with the alignment check; the back-translations drawn to the word
    ratio of R2's real examples, 4 per book pair or as many as the bins hold"""
    import json

    args = rl_tables(tmp_path)
    assert [r for _, r in sb.match_target(args.bt_match_set)] == [1.0] * 10 + [1.5] * 10
    sb.build(args, tokenizer=CharTokenizer())
    out = Path(args.out)
    meta = pq.read_table(out / "RL" / "train-meta.parquet").to_pylist()
    books = [m for m in meta if m["kind"] == "real"]
    bt = [m for m in meta if m["kind"] in ("bt", "bt_context")]
    # 10 of 12 books: not a Dutch original, the shifted one out; QE 20 in (no bound)
    assert sorted(m["id"] for m in books) == sorted(f"books:{i}" for i in range(2, 12))
    # 40 wanted (4 x 10); the upper bin (ratio 1.5, half of the target) holds the 6 of d10-d15: 12 drawn
    assert len(bt) == 12 and {f"handelingen:d{i}:t5:0" for i in range(10, 16)} <= {m["id"] for m in bt}
    report = json.loads((out / "report.json").read_text())
    real = report["real"]
    assert real["qe_max"] is None and real["qe_unbounded"] and report["query"]["real_qe_en_nl_max"] is None
    assert real["query_cost"]["first_failed"] == {"books/hard:not_dutch_original": 1, "books/align": 1}
    assert (real["align"]["max"], real["align"]["checked"], real["align"]["failed"]) == (2.0, 11, 1)
    assert real["align"]["by_qe"] == {"<=5": {"pairs": 10, "failed": 1}, "15-20": {"pairs": 1, "failed": 0}}
    assert report["query"]["real_align_max"] == 2.0 and report["query"]["bt_match"] == {"bins": 2, "per_real": 4.0}
    d = report["bt"]["match_draw"]
    assert (d["want"], d["most"], d["drawn"], d["limited"]) == (40, 12, 12, True)
    assert d["context"] == sum(m["kind"] == "bt_context" for m in bt) and "per_real_draw" not in report["bt"]
    assert report["inputs"]["bt_match_set"]["path"] == args.bt_match_set
    md = (out / "report.md").read_text()
    assert "Uitlijncontrole" in md and "BT hersampled" in md and "Geen MetricX-24-QE-grens" in md
    # the same seed, the same set
    first = sorted(m["id"] for m in meta)
    sb.build(args, tokenizer=CharTokenizer())
    assert sorted(m["id"] for m in pq.read_table(out / "RL" / "train-meta.parquet").to_pylist()) == first
    # 1 per book pair fits: 10, 5 from each bin
    args.bt_per_real, args.out = 1.0, str(tmp_path / "out-1")
    sb.build(args, tokenizer=CharTokenizer())
    d = json.loads((Path(args.out) / "report.json").read_text())["bt"]["match_draw"]
    assert (d["want"], d["drawn"], d["limited"]) == (10, 10, False) and [b["drawn"] for b in d["per_bin"]] == [5, 5]


def test_rl_options_go_together(tmp_path):
    base = ["build", "--run", "r", "--long", "l", "--replay", "x", "--flan-eval", "f", "--evalsets", "e",
            "--tokenizer", "t", "--sets", "RL=none+real", "--bt", "all", "--real-all", "--out", "o"]
    for extra in (["--real", "x", "--real-qe-unbounded", "--real-qe-max", "7.5"],
                  ["--real", "x", "--real-qe-unbounded", "--real-without-qe"], ["--real-qe-unbounded"],
                  ["--real", "x", "--real-align-max", "4.39"],
                  ["--real", "x", "--len-factor", "1.3333", "--real-align-max", "0"],
                  ["--real", "x", "--bt-match-set", "s"], ["--real", "x", "--bt-per-real", "4", "--bt-match-bins", "5"],
                  ["--real", "x", "--bt-per-real", "4", "--bt-match-set", "s", "--bt-match-bins", "1"]):
        with pytest.raises(SystemExit):
            sb.main(base + extra)
    # without --len-factor in build too (no median to align at)
    write(tmp_path / "books.parquet", [{"id": "books:1", "source": "books", "en": EN_SENTENCES[0],
                                        "nl": NL_SENTENCES[0], "fails": [], "qe_en_nl": 1.0}])
    args = argparse.Namespace(real=str(tmp_path / "books.parquet"), real_align_max=2.0, len_factor=None,
                              real_without_qe=False, qe_max=5.0)
    with pytest.raises(SystemExit, match="needs --len-factor"):
        sb.candidates_real(args, set(), {})
