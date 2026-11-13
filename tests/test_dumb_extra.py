"""scripts/dumb/finetune_extra.py and its tasks on the CPU: SQuAD windows and post-processing, EM/F1, first-sub-token
labels, entity F1, POS accuracy, the learning-rate schedule, and tiny fine-tunes end to end."""
import json
import pickle
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "dumb"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import extra_metrics as em  # noqa: E402
import extra_tasks as et  # noqa: E402
import finetune_extra as fx  # noqa: E402
from dumbx_tokenizer import make_tokenizer  # noqa: E402

from mmbjax.configuration_mmbbert import MMBertConfig  # noqa: E402
from mmbjax.heads.classification import ModernBertForSequenceClassification  # noqa: E402
from mmbjax.heads.qa import ModernBertForQuestionAnswering  # noqa: E402

# ---------------------------------------------------------------- SQuAD metric

def test_squad_normalization_is_the_official_one():
    assert em.squad_normalize("The  Cat, the hat!") == "cat hat"
    # English articles only, as in the official script; Dutch ones stay
    assert em.squad_normalize("De kat (een) an  a") == "de kat een"
    assert em.squad_normalize(" 's-Hertogenbosch ") == "shertogenbosch"


def test_exact_match_and_f1_take_the_best_answer_per_question():
    assert em.squad_exact_match("de Panthers.", "De Panthers")
    assert em.squad_f1("de rode kat", "de kat") == pytest.approx(2 * (2 / 3) * 1 / (2 / 3 + 1))
    assert em.squad_f1("hond", "de kat") == 0.0
    scores = em.squad_scores({"a": "308 punten", "b": "Amsterdam"}, {"a": ["308", "308 punten"], "b": ["Rotterdam"]})
    assert scores == {"exact_match": 0.5, "f1": 0.5, "questions": 2}
    with pytest.raises(ValueError, match="without a prediction"):
        em.squad_scores({"a": "x"}, {"a": ["x"], "b": ["y"]})


# ---------------------------------------------------------------- entities and words

def test_entities_are_read_as_seqeval_reads_iob2():
    assert em.iob2_entities(["B-PER", "I-PER", "O", "B-LOC"]) == [("PER", 0, 1), ("LOC", 3, 3)]
    # I- without B- starts an entity, a new type ends one, B- after B- is a second one (conlleval)
    assert em.iob2_entities(["O", "I-ORG", "I-ORG", "I-LOC", "B-LOC", "B-LOC"]) == [("ORG", 1, 2), ("LOC", 3, 3), ("LOC", 4, 4), ("LOC", 5, 5)]
    with pytest.raises(ValueError):
        em.iob2_entities(["E-PER"])


def test_entity_scores_count_only_whole_entities_of_the_right_type():
    true = [["B-PER", "I-PER", "O", "B-LOC"], ["B-ORG", "O"]]
    predicted = [["B-PER", "O", "O", "B-LOC"], ["B-MISC", "O"]]
    scores = em.entity_scores(true, predicted)
    # True: PER(0-1), LOC, ORG; predicted: PER(0-0), LOC, MISC; only LOC is right
    assert (scores["true"], scores["predicted"], scores["correct"]) == (3, 3, 1)
    assert scores["precision"] == scores["recall"] == pytest.approx(1 / 3) and scores["f1"] == pytest.approx(1 / 3)
    assert scores["per_type"]["LOC"]["f1"] == 1.0 and scores["per_type"]["PER"]["f1"] == 0.0
    assert scores["per_type"]["MISC"] == {"precision": 0.0, "recall": 0.0, "f1": 0.0, "true": 0, "predicted": 1, "correct": 0}


def test_word_accuracy_counts_every_word():
    assert em.word_accuracy([["NOUN", "VERB"], ["DET"]], [["NOUN", "NOUN"], ["DET"]]) == {"accuracy": 2 / 3, "words": 3, "correct": 2}
    with pytest.raises(ValueError):
        em.word_accuracy([["NOUN"]], [["NOUN", "VERB"]])


# ---------------------------------------------------------------- first sub-token labels

def test_each_word_is_labelled_at_its_first_sub_token_only():
    tokenizer = make_tokenizer()
    rows = [{"tokens": ["Jan", "zag", "kalmte", "in", "Amsterdam", "."], "ner_tags": ["B-PER", "O", "O", "O", "B-LOC", "O"]}]
    example = et.token_examples(tokenizer, rows, et.NER_LABELS, "ner_tags", 64)[0]
    tokens = tokenizer.convert_ids_to_tokens(example["input_ids"])
    # "kalmte" is ▁kalm te and "." is ▁ . here: the label sits on the first piece of each word
    assert tokens == ["<s>", "▁Jan", "▁zag", "▁kalm", "te", "▁in", "▁Amsterdam", "▁", "."]
    assert example["first"] == [1, 2, 3, 5, 6, 7]
    labelled = {i: et.NER_LABELS[label] for i, label in enumerate(example["labels"]) if label != -100}
    assert labelled == {1: "B-PER", 2: "O", 3: "O", 5: "O", 6: "B-LOC", 7: "O"}
    assert example["word_labels"] == [et.NER_LABELS.index(t) for t in rows[0]["ner_tags"]]


@pytest.mark.parametrize("side", ["right", "left"])
def test_a_sentence_longer_than_max_length_is_cut_between_words_and_loses_none(side):
    tokenizer = make_tokenizer()
    # The real tokenizer truncates from the left; nothing here may depend on that
    tokenizer.truncation_side = side
    words = ["Jan", "zag", "kalmte", "in", "Amsterdam", "."] * 4  # 8 tokens per 6 words here
    rows = [{"tokens": ["de", "man"], "upos": ["DET", "NOUN"]}, {"tokens": words, "upos": ["X"] * len(words)}]
    examples = et.token_examples(tokenizer, rows, et.UPOS_LABELS, "upos", 10)
    assert examples[0]["sentence"] == 0 and [e["sentence"] for e in examples[1:]] == [1] * (len(examples) - 1) and len(examples) > 3
    joined = []
    for e in examples[1:]:
        assert len(e["input_ids"]) <= 10 and tokenizer.convert_ids_to_tokens(e["input_ids"][0]) == "<s>"
        pieces = tokenizer.convert_ids_to_tokens(e["input_ids"])
        # A row's first sub-tokens are its words, in order
        joined.extend(pieces[p] for p in e["first"])
    assert joined == [tokenizer.convert_ids_to_tokens(tokenizer([w], is_split_into_words=True)["input_ids"])[1] for w in words]
    with pytest.raises(ValueError, match="alone has"):
        et.encode_words(tokenizer, [["Hertogenbosch" * 3]], 10)


def test_the_chunks_of_a_sentence_are_scored_as_one_sentence():
    # An entity across the cut between two rows of one sentence is one entity
    examples = [{"sentence": 0, "first": [1, 2], "word_labels": [0, 1]}, {"sentence": 0, "first": [1], "word_labels": [2]},
                {"sentence": 1, "first": [1], "word_labels": [0]}]
    logits = [np.zeros((3, 9)), np.zeros((2, 9)), np.zeros((2, 9))]
    logits[0][1, 0] = logits[0][2, 1] = logits[1][1, 2] = logits[2][1, 0] = 5.0
    scores = fx.token_scores("conll2002", examples, logits)
    assert (scores["sentences"], scores["rows"], scores["words"]) == (2, 3, 4)
    assert (scores["entities"], scores["entity_f1"], scores["accuracy"]) == (1, 1.0, 1.0)


def test_token_scores_read_the_logits_at_the_first_sub_tokens():
    examples = [{"sentence": 0, "first": [1, 3], "word_labels": [1, 0]}, {"sentence": 1, "first": [1], "word_labels": [3]}]
    logits = [np.zeros((5, 9)), np.zeros((2, 9))]
    logits[0][1, 1] = logits[0][3, 0] = 5.0  # B-PER O: right
    logits[0][2, 4] = 9.0  # a later sub-token: ignored
    logits[1][1, 5] = 5.0  # B-LOC where I-ORG is true
    scores = fx.token_scores("conll2002", examples, logits)
    assert scores["accuracy"] == pytest.approx(2 / 3) and scores["words"] == 3
    assert (scores["entities"], scores["entities_predicted"]) == (2, 2) and scores["entity_f1"] == pytest.approx(0.5)
    assert scores["metric"] == scores["entity_f1"]
    pos = fx.token_scores("pos-alpino", examples, logits)
    assert pos["metric"] == pos["accuracy"] and "entity_f1" not in pos


# ---------------------------------------------------------------- SQuAD windows and answers

CONTEXT = "Jan woont in Amsterdam bij de man en de vrouw. Piet zit in de kamer bij Jan en de man."


def squad_rows():
    return [
        {"id": "q1", "question": "Waar woont Jan?", "context": CONTEXT, "answers": {"answer_start": [CONTEXT.index("Amsterdam")], "text": ["Amsterdam"]}},
        {"id": "q2", "question": "Wie zit in de kamer?", "context": CONTEXT,
         "answers": {"answer_start": [CONTEXT.index("Piet"), CONTEXT.index("Piet")], "text": ["Piet", "Piet zit"]}},
    ]


def test_answer_positions_map_characters_to_tokens():
    # Offsets as this tokenizer gives them: a ▁ piece includes the space before its word
    offsets = [None, (0, 3), (3, 9), (9, 12), (12, 22), (22, 23)]
    assert et.answer_positions(offsets, 13, 22) == (4, 4)
    assert et.answer_positions(offsets, 4, 12) == (2, 3)
    assert et.answer_positions(offsets, 20, 30) == (0, 0)  # not all inside this window


@pytest.mark.parametrize("side", ["right", "left"])
def test_windows_overlap_by_the_stride_and_label_the_answer_where_it_is_whole(side):
    tokenizer = make_tokenizer()
    # The real tokenizer truncates from the left, which only reverses the order of the windows
    tokenizer.truncation_side = side
    rows = squad_rows()
    features = et.qa_features(tokenizer, rows, max_length=24, doc_stride=6)
    assert len({f["example"] for f in features}) == 2 and len(features) > 2
    for f in features:
        sequence = [i for i, offset in enumerate(f["offsets"]) if offset is not None]
        # The context follows <s> question <s> and runs to the end of the window
        assert sequence == list(range(sequence[0], len(f["input_ids"])))
        assert tokenizer.convert_ids_to_tokens(f["input_ids"][sequence[0] - 1]) == "<s>"
        assert len(f["input_ids"]) <= 24
        row = rows[f["example"]]
        if f["start_position"] == 0:
            assert f["end_position"] == 0
            continue
        start, end = f["offsets"][f["start_position"]][0], f["offsets"][f["end_position"]][1]
        assert row["context"][start:end].strip() == row["answers"]["text"][0]
    for example in (0, 1):
        windows = [f for f in features if f["example"] == example]
        assert any(f["start_position"] > 0 for f in windows), "some window holds the whole answer"
        spans = sorted({f["offsets"][i] for f in windows for i in range(len(f["offsets"])) if f["offsets"][i] is not None})
        assert spans[0][0] == 0 and spans[-1][1] == len(CONTEXT), "the windows cover the context"
        # Consecutive windows share doc_stride context tokens
        contexts = sorted([[o for o in f["offsets"] if o is not None] for f in windows])
        for a, b in zip(contexts, contexts[1:]):
            assert len(set(a) & set(b)) == 6


def test_predictions_take_the_best_valid_span_over_all_windows():
    rows = [{"id": "x", "context": "abc def ghi", "answers": {"text": ["def"], "answer_start": [4]}}]
    features = [
        {"example": 0, "offsets": [None, (0, 3), (3, 7)]},
        {"example": 0, "offsets": [None, (3, 7), (7, 11)]},
    ]
    start = [np.array([9.0, 1.0, 0.0]), np.array([0.0, 3.0, 2.5])]
    end = [np.array([9.0, 0.0, 1.0]), np.array([0.0, 3.0, 0.0])]
    # Position 0 (<s>, outside the context) scores highest but is no answer; the best valid span is (3, 7) in window 2
    predictions, no_span = et.qa_predictions(rows, features, start, end)
    assert predictions == {"x": "def"} and no_span == 0
    # The highest sum, start 2 with end 1, ends before it starts: no span. The best valid one is start 1 with end 1
    predictions, _ = et.qa_predictions(rows, features[1:], [np.array([0.0, -8.0, 5.0])], [np.array([0.0, 5.0, -9.0])])
    assert predictions == {"x": "def"}
    # With n_best 1 only that invalid pair is considered
    _, no_span = et.qa_predictions(rows, features[1:], [np.array([0.0, -8.0, 5.0])], [np.array([0.0, 5.0, -9.0])], n_best=1)
    assert no_span == 1
    _, no_span = et.qa_predictions(rows, features[:1], [start[0]], [end[0]], max_answer_length=0)
    assert no_span == 1


def test_span_head_loss_is_the_mean_of_start_and_end_cross_entropy():
    config = MMBertConfig(vocab_size=16, hidden_size=8, intermediate_size=16, num_hidden_layers=1, num_attention_heads=2,
                          max_position_embeddings=32, local_attention=4, global_attn_every_n_layers=1, classifier_dropout=0.0,
                          num_labels=2)
    model = ModernBertForQuestionAnswering(config)
    rows = [{"input_ids": [1, 5, 6, 1, 7, 8, 9], "start_position": 4, "end_position": 5},
            {"input_ids": [1, 5, 1, 7], "start_position": 0, "end_position": 0}]
    batch = fx.collate(rows, "qa", 0, 16, 8)
    assert batch["input_ids"].shape == (2, 8) and batch["attention_mask"][1].tolist() == [1, 1, 1, 1, 0, 0, 0, 0]
    variables = model.init(jax.random.PRNGKey(0), jnp.asarray(batch["input_ids"]), deterministic=True)
    out = model.apply(variables, jnp.asarray(batch["input_ids"]), attention_mask=jnp.asarray(batch["attention_mask"]),
                      start_positions=jnp.asarray(batch["start_positions"]), end_positions=jnp.asarray(batch["end_positions"]),
                      deterministic=True)
    assert out.start_logits.shape == out.end_logits.shape == (2, 8)
    start = np.asarray(jax.nn.log_softmax(out.start_logits, axis=-1))
    end = np.asarray(jax.nn.log_softmax(out.end_logits, axis=-1))
    expected = -0.5 * (np.mean([start[0, 4], start[1, 0]]) + np.mean([end[0, 5], end[1, 0]]))
    assert float(out.loss) == pytest.approx(expected, rel=1e-5)


# ---------------------------------------------------------------- schedule and batches

def test_the_rate_falls_linearly_to_zero_after_the_warmup():
    decay = fx.learning_rate_schedule(5e-5, "linear", 100, 0)
    assert [float(decay(s)) for s in (0, 50, 100)] == pytest.approx([5e-5, 2.5e-5, 0.0])
    warm = fx.learning_rate_schedule(5e-5, "linear", 100, 10)
    assert [float(warm(s)) for s in (0, 5, 10, 55, 100)] == pytest.approx([0.0, 2.5e-5, 5e-5, 2.5e-5, 0.0])
    assert fx.learning_rate_schedule(5e-5, "constant", 100, 0) == 5e-5
    with pytest.raises(ValueError):
        fx.learning_rate_schedule(5e-5, "linear", 10, 10)


def test_batches_are_padded_to_a_multiple_and_labels_pad_with_ignore():
    rows = [{"input_ids": [1, 2, 3], "labels": [-100, 4, 5]}, {"input_ids": [1] * 130, "labels": [0] * 130}]
    batch = fx.collate(rows, "token", 0, 512, 128)
    assert batch["input_ids"].shape == (2, 256)
    assert batch["labels"][0, :4].tolist() == [-100, 4, 5, -100] and (batch["labels"][0, 3:] == -100).all()
    assert fx.collate(rows[:1], "token", 0, 512, 128)["input_ids"].shape == (1, 128)
    assert fx.collate(rows, "token", 0, 200, 128)["input_ids"].shape == (2, 200)
    with pytest.raises(ValueError):
        fx.collate(rows, "token", 0, 100, 128)


def test_prediction_covers_every_row_when_the_last_batch_is_incomplete():
    rows = [{"input_ids": list(range(1, n + 1))} for n in (3, 5, 2, 4, 1)]
    seen = []

    def forward(state, batch):
        seen.append(batch["input_ids"].shape)
        # logits = the input id, so each output can be traced to its row
        return np.asarray(batch["input_ids"], dtype=np.float32)[..., None]

    def collate_rows(chunk):
        return fx.collate([{**r, "labels": [0] * len(r["input_ids"])} for r in chunk], "token", 0, 16, 4)

    outputs = fx.predict(None, rows, collate_rows, 2 * fx.finetune.NUM_DEVICES, forward)
    assert [o[:, 0].tolist() for o in outputs] == [r["input_ids"] for r in rows]


# ---------------------------------------------------------------- tiny fine-tunes end to end

def _checkpoint(path):
    config = MMBertConfig(vocab_size=256, hidden_size=32, intermediate_size=32, num_hidden_layers=3, num_attention_heads=2,
                          max_position_embeddings=512, local_attention=8, global_attn_every_n_layers=3, pad_token_id=3,
                          classifier_pooling="cls")
    model = ModernBertForSequenceClassification(config)
    params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 8), dtype=jnp.int32), deterministic=True)["params"]
    path.mkdir()
    (path / "params.pkl").write_bytes(pickle.dumps({"model": jax.device_get(params["model"]), "prediction_head": jax.device_get(params["head"])}))
    (path / "config.json").write_text(json.dumps({**config.to_dict(), "tokenizer_name": "unused"}))
    return path


def _data_dir(path):
    """A staged directory in the layout of stage_data.py, a few rows per split."""
    rng = np.random.RandomState(0)
    words = ["Jan", "woont", "in", "Amsterdam", "de", "man", "zit", "kamer", "."]
    upos = {"Jan": "PROPN", "woont": "VERB", "in": "ADP", "Amsterdam": "PROPN", "de": "DET", "man": "NOUN", "zit": "VERB", "kamer": "NOUN", ".": "PUNCT"}
    ner = {"Jan": "B-PER", "Amsterdam": "B-LOC"}
    for task, files in (("conll2002-nl", ("train", "validation", "test")), ("ud-nl-alpino", ("train", "dev", "test"))):
        for file, n in zip(files, (40, 7, 9)):
            rows = []
            for i in range(n):
                tokens = [words[k] for k in rng.randint(0, len(words), size=rng.randint(2, 9))]
                rows.append({"id": i, "tokens": tokens, "ner_tags": [ner.get(t, "O") for t in tokens], "upos": [upos[t] for t in tokens]})
            (path / task).mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.Table.from_pylist(rows), path / task / f"{file}.parquet")
    (path / "squad-nl-v1.1").mkdir()
    for file, n in (("train", 24), ("dev", 3), ("test", 5)):
        rows = [{**row, "id": f"{row['id']}-{i}"} for i in range(n // 2 + 1) for row in squad_rows()][:n]
        pq.write_table(pa.Table.from_pylist(rows), path / "squad-nl-v1.1" / f"{file}.parquet")
    (path / "manifest.json").write_text(json.dumps({"datasets": {d: {"name": d, "revision": "test"} for d in ("conll2002-nl", "ud-nl-alpino", "squad-nl-v1.1")}}))
    return path


@pytest.fixture(scope="module")
def staged(tmp_path_factory):
    root = tmp_path_factory.mktemp("dumbx")
    return _checkpoint(root / "ckpt"), _data_dir(root / "data")


@pytest.mark.parametrize("task", ["conll2002", "pos-alpino", "squadnl"])
def test_a_tiny_fine_tune_reports_every_test_example(task, staged, tmp_path, monkeypatch):
    checkpoint, data_dir = staged
    monkeypatch.setattr(fx.AutoTokenizer, "from_pretrained", lambda name: make_tokenizer())
    extra = ["--max-length", "32", "--doc-stride", "8"] if task == "squadnl" else []
    results = fx.main(["--checkpoint", str(checkpoint), "--task", task, "--data-dir", str(data_dir), "--output", str(tmp_path),
                       "--epochs", "2", "--batch-size", "4", "--learning-rate", "1e-3", "--pad-to-multiple-of", "16", *extra])
    assert json.loads((tmp_path / "results.json").read_text()) == json.loads(json.dumps(results))
    for key in ("task", "checkpoint", "best_val_accuracy", "test_accuracy", "test_loss", "epochs", "learning_rate", "batch_size",
                "total_batch_size", "gradient_accumulation_steps", "num_devices", "max_length", "pad_to_multiple_of"):
        assert key in results, key
    assert results["metric"] == {"conll2002": "entity_f1", "pos-alpino": "accuracy", "squadnl": "f1"}[task]
    assert results["test_metric"] == results["test"]["metric"] and len(results["validation"]) == 2
    assert results["updates"] == results["train_examples"] // results["total_batch_size"] * 2
    test_rows = pq.read_table(data_dir / et.TASKS[task]["dir"] / "test.parquet").to_pylist()
    if task == "squadnl":
        assert results["test"]["questions"] == len(test_rows) and results["test"]["features"] > len(test_rows)
    else:
        # 9 test sentences on batches of 4: the last, incomplete one counts too
        assert results["test"]["sentences"] == 9 and results["test"]["words"] == sum(len(r["tokens"]) for r in test_rows)
    assert np.isfinite(results["test_loss"])


# ---------------------------------------------------------------- the protocol's table and comparison

def test_summary_compares_the_difference_of_means_with_the_larger_seed_range(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "dumb-extra"))
    import summarize

    numbers = {("S", "squadnl"): [0.80, 0.81, 0.82], ("mix", "squadnl"): [0.84, 0.845, 0.85],
               ("S", "conll2002"): [0.70, 0.71, 0.72], ("mix", "conll2002"): [0.69, 0.725, 0.73],
               ("mix-653000", "dbrd"): [0.95, 0.951, 0.952]}
    for (name, task), values in numbers.items():
        for seed, value in zip((42, 43, 44), values):
            key = "test_accuracy" if task == "dbrd" else "test_metric"
            (tmp_path / f"{name}--{task}--seed{seed}.json").write_text(json.dumps({key: value}))
    (tmp_path / "S--pos-alpino--seed42.json").write_text(json.dumps({"test_metric": 0.9}))
    out = summarize.main(["--results", str(tmp_path), "--reference", "S", "--compare", "mix"])
    assert out["tasks"]["mix-653000--dbrd"]["mean"] == pytest.approx(0.951)
    assert not out["tasks"]["S--pos-alpino"]["complete"] and out["tasks"]["S--squadnl"]["complete"]
    verdicts = {row["task"]: row["beats"] for row in out["comparison"]}
    # squadnl: +0.035 against a larger range of 0.02; conll2002: +0.005 against 0.04
    assert verdicts == {"squadnl": "mix", "conll2002": "neither"}
