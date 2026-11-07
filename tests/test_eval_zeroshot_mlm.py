"""scripts/eval_zeroshot_mlm.py on a tiny model on the CPU: prompt format, cutting, packing and scoring."""
import sys
import types
from pathlib import Path
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_multihost import _stub_wandb_if_missing  # noqa: E402

WORDS = ["positief", "negatief", "A", "B", "C", "D"] + [f"w{i}" for i in range(80)]


@pytest.fixture(scope="module")
def ez():
    _stub_wandb_if_missing()
    # Other test modules already started JAX, after which train_minimal's module-level initialize() can't run
    with mock.patch("jax.distributed.initialize", side_effect=ValueError):
        import eval_zeroshot_mlm

    return eval_zeroshot_mlm


def _tokenizer():
    """Like yhavinga/dutch-llama-tokenizer: no <mask> of its own; the verbalizers are words of the vocabulary."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    vocab = {token: i for i, token in enumerate(["<unk>", "<s>", "</s>", "<pad>"] + WORDS)}
    backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>", bos_token="<s>", eos_token="</s>",
                                   pad_token="<pad>", split_special_tokens=True)


def _words(rng, n):
    return " ".join(f"w{w}" for w in rng.randint(0, 80, size=n))


def _fake_load_dataset(name, config, split):
    import datasets

    assert split == "test"
    rng = np.random.RandomState(len(name))
    if name == "dbrd":
        # Some reviews longer than the window of the test
        rows = [{"text": _words(rng, rng.randint(1, 120)), "label": i % 2} for i in range(30)]
    elif name == "maximedb/sick_nl":
        rows = [{"sentence_A": _words(rng, 6), "sentence_B": _words(rng, 5), "entailment_label": label,
                 "label": {"CONTRADICTION": 0, "NEUTRAL": 1, "ENTAILMENT": 2}[label]}
                for label in ["NEUTRAL", "ENTAILMENT", "CONTRADICTION", "NEUTRAL"] * 6]
    else:
        assert (name, config) == ("facebook/belebele", "nld_Latn")
        rows = [{"flores_passage": _words(rng, 30), "question": _words(rng, 5), **{f"mc_answer{k}": _words(rng, 2) for k in range(1, 5)},
                 "correct_answer_num": str(1 + i % 4)} for i in range(16)]
    return datasets.Dataset.from_list(rows)


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory, ez):
    """A train_minimal.py checkpoint of a tiny model without <answer>, as S and W are."""
    import optax
    from flax.training import train_state

    tm = ez.tm
    tokenizer = _tokenizer()
    tokenizer.add_special_tokens({"mask_token": "<mask>"})
    args = types.SimpleNamespace(hidden_size=16, num_layers=2, num_heads=2, intermediate_size=32,
                                 global_attn_every_n_layers=3, compute_dtype="float32")
    config, model = ez.build_model(args, tokenizer)
    ones = jnp.ones((1, 8), dtype=jnp.int32)
    params = model.init(jax.random.PRNGKey(5), input_ids=ones, attention_mask=ones, deterministic=True)["params"]
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optax.adamw(1e-3))
    directory = tmp_path_factory.mktemp("ckpt")
    manager = tm.ocp.CheckpointManager(directory, options=tm.ocp.CheckpointManagerOptions(create=True))
    manager.save(7, args=tm.ocp.args.StandardSave(state))
    manager.wait_until_finished()
    return directory / "7", model, params


FLAGS = ["--tokenizer", "fake", "--hidden-size", "16", "--num-layers", "2", "--num-heads", "2", "--intermediate-size", "32",
         "--max-length", "64", "--rows-per-device", "2"]


def test_three_templates_per_task_with_their_options_and_one_token_verbalizers(ez, monkeypatch):
    tokenizer = _tokenizer()
    for task, templates in ez.TEMPLATES.items():
        assert len(templates) == 3
        for template in templates:
            # The options are in the prompt, in the order of the verbalizers
            options = ["positief", "negatief"] if task == "dbrd" else [f"{letter}" for letter in ez.VERBALIZERS[task]]
            assert all(option in template for option in options), template
        assert len(ez.verbalizer_ids(tokenizer, task)) == len(ez.VERBALIZERS[task])
    monkeypatch.setitem(ez.VERBALIZERS, "dbrd", ["neutraal"])
    with pytest.raises(ValueError, match="not one token"):
        ez.verbalizer_ids(tokenizer, "dbrd")


def test_examples_of_the_test_splits_and_their_labels(ez, monkeypatch):
    monkeypatch.setattr(ez, "load_dataset", _fake_load_dataset)
    sick = ez.task_examples("sicknl")
    # A is entailment in every prompt, whatever the integers of the dataset's label column
    assert [example["label"] for example in sick[:4]] == [1, 0, 2, 1]
    belebele = ez.task_examples("belebele")
    assert [example["label"] for example in belebele[:5]] == [0, 1, 2, 3, 0]
    assert set(belebele[0]) == {"passage", "question", "a", "b", "c", "d", "label"}
    assert len(ez.task_examples("dbrd", max_examples=7)) == 7


def test_a_long_review_is_cut_and_the_format_ends_in_answer_mask(ez):
    from mmbjax.training import add_answer_token

    tokenizer = _tokenizer()
    tokenizer.add_special_tokens({"mask_token": "<mask>"})
    answer_token_id = add_answer_token(tokenizer)
    review = " ".join(f"w{i % 80}" for i in range(500))
    tasks = {"dbrd": [{"text": review, "label": 1}, {"text": "w1 w2", "label": 0}]}
    sequences = ez.build_sequences(tokenizer, answer_token_id, tasks, 64)
    assert len(sequences) == 6
    for task, template, example, label, ids, mask_at, cut in sequences:
        assert len(ids) <= 64
        assert ids[0] == tokenizer.bos_token_id and list(ids[-3:]) == [answer_token_id, tokenizer.mask_token_id, tokenizer.eos_token_id]
        assert mask_at == len(ids) - 2
        assert cut == (example == 0)
        if example == 0:
            # Cut at a word, as long as it fits: one more word would not
            prompt = list(ids[1:-3])
            assert len(prompt) in (60, 59)
    # The question after the review is kept: the cut is in the review
    first = sequences[0][4]
    tail = tokenizer(ez.TEMPLATES["dbrd"][0].split("{text}")[1], add_special_tokens=False)["input_ids"]
    assert list(first[-3 - len(tail):-3]) == tail


def test_packed_predictions_are_those_of_each_prompt_alone(ez, tiny_checkpoint, tmp_path, monkeypatch):
    from mmbjax.training import add_answer_token

    step_dir, model, params = tiny_checkpoint
    monkeypatch.setattr(ez, "AutoTokenizer", types.SimpleNamespace(from_pretrained=lambda *_a, **_k: _tokenizer()))
    monkeypatch.setattr(ez, "load_dataset", _fake_load_dataset)
    result = ez.main(["--checkpoint", str(step_dir), *FLAGS, "--output", str(tmp_path / "zeroshot.json")])
    assert (tmp_path / "zeroshot.json").exists()
    assert result["answer_token_id"] == 91 and result["mask_token_id"] == 90 and result["vocab_size"] == 128
    assert result["format"] == "<s> {prompt} <answer> <mask> </s>"

    tokenizer = _tokenizer()
    tokenizer.add_special_tokens({"mask_token": "<mask>"})
    answer_token_id = add_answer_token(tokenizer)
    tasks = {task: ez.task_examples(task) for task in ez.DATASETS}
    for task, values in result["tasks"].items():
        ids = ez.verbalizer_ids(tokenizer, task)
        assert values["examples"] == len(tasks[task]) and values["chance"] == 1 / len(ids)
        accuracies = []
        for t, template in enumerate(values["templates"]):
            assert template["examples"] == len(tasks[task])
            correct = 0
            for task_, t_, e, label, sequence, mask_at, _ in ez.build_sequences(tokenizer, answer_token_id, {task: tasks[task]}, 64):
                if t_ != t:
                    continue
                logits = model.apply({"params": params}, input_ids=jnp.asarray(sequence[None]), deterministic=True).logits[0]
                correct += int(np.argmax(np.asarray(logits)[mask_at, ids])) == label
            assert template["accuracy"] == pytest.approx(correct / len(tasks[task])), (task, t)
            accuracies.append(template["accuracy"])
        assert values["mean_accuracy"] == pytest.approx(np.mean(accuracies))
    assert result["tasks"]["dbrd"]["templates"][0]["cut"] > 0


def test_a_vocabulary_without_room_for_answer_is_refused(ez, tiny_checkpoint, monkeypatch):
    step_dir = tiny_checkpoint[0]

    def full_tokenizer(*_a, **_k):
        # 128 tokens with <mask>: the vocabulary of 128 has no row left for <answer>
        tokenizer = _tokenizer()
        tokenizer.add_tokens([f"x{i}" for i in range(128 - 1 - len(tokenizer))])
        return tokenizer

    monkeypatch.setattr(ez, "AutoTokenizer", types.SimpleNamespace(from_pretrained=full_tokenizer))
    monkeypatch.setattr(ez, "load_dataset", _fake_load_dataset)
    with pytest.raises(ValueError, match="has no row in a vocabulary of 128"):
        ez.main(["--checkpoint", str(step_dir), *FLAGS])
