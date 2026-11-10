"""experiments/en-nl-translator/sft_data.py and sft_lora.py, and translate.py's LoRA merge, without a model: the split,
the messages of the two arms, the chat template against TranslateGemma's prompt, the argument split, the throughput,
and the algebra of the merge. MaxText's own SFT steps are checked on a worker (`sft_data.py check`), the merge against
qwix in tests/test_en_nl_lora_merge.py. Run with --noconftest where the conftest's imports are not installed."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "en-nl-translator"))
sys.path.insert(0, str(ROOT / "experiments" / "flan-nl"))

import sft_data  # noqa: E402
import sft_lora  # noqa: E402
import translate  # noqa: E402


def test_split_pairs_is_seeded_disjoint_and_complete():
    ids = list(range(2000))
    train, held = sft_data.split_pairs(ids, 10, 0)
    assert len(held) == 10 and len(train) == 1990
    assert not set(train) & set(held) and set(train) | set(held) == set(ids)
    assert sft_data.split_pairs(ids, 10, 0) == (train, held)
    assert sft_data.split_pairs(ids, 10, 1)[1] != held


def test_messages_tag_only_arm_n():
    tagged = sft_data.messages("Hello there.", "Hallo daar.", tagged=True)
    plain = sft_data.messages("Hello there.", "Hallo daar.", tagged=False)
    assert tagged[0] == {"role": "user", "content": f"{sft_data.TAG} Hello there."}
    assert plain[0] == {"role": "user", "content": "Hello there."}
    assert tagged[1] == plain[1] == {"role": "assistant", "content": "Hallo daar."}


def render(messages, add_generation_prompt):
    """As transformers renders a chat template: a sandboxed jinja2 environment with trim_blocks and lstrip_blocks."""
    jinja2 = pytest.importorskip("jinja2")
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    def raise_exception(message):
        raise jinja2.exceptions.TemplateError(message)

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    env.globals["raise_exception"] = raise_exception
    return env.from_string(sft_data.CHAT_TEMPLATE).render(messages=messages, bos_token="<bos>",
                                                          add_generation_prompt=add_generation_prompt)


def test_chat_template_gives_translategemmas_prompt_and_the_bare_completion():
    head = ("<start_of_turn>user\nYou are a professional English (en) to Dutch (nl) translator. Your goal is to "
            "accurately convey the meaning and nuances of the original English text while adhering to Dutch grammar, "
            "vocabulary, and cultural sensitivities.\nProduce only the Dutch translation, without any additional "
            "explanations or commentary. Please translate the following English text into Dutch:\n\n\n")
    msgs = sft_data.messages("  It rains.\n", "Het regent.", tagged=False)
    assert render(msgs[:1], True) == "<bos>" + head + "It rains.<end_of_turn>\n<start_of_turn>model\n"
    assert render(msgs, False) == ("<bos>" + head + "It rains.<end_of_turn>\n<start_of_turn>model\nHet regent."
                                   "<end_of_turn>\n")


def test_chat_template_refuses_a_conversation_that_starts_with_the_model():
    jinja2 = pytest.importorskip("jinja2")
    with pytest.raises(jinja2.exceptions.TemplateError):
        render([{"role": "assistant", "content": "x"}], False)


def test_split_argv():
    assert sft_lora.split_argv(["--out", "d", "--", "a=1", "b=2"]) == (["--out", "d"], ["a=1", "b=2"])
    assert sft_lora.split_argv(["--out", "d"]) == (["--out", "d"], [])


def test_throughput_leaves_out_the_first_step():
    records = [{"t": 100.0}, {"t": 102.0}, {"t": 103.0}, {"t": 104.0}]
    rates = sft_lora.throughput(records, tokens=[999, 10, 20, 30], loss_tokens=[999, 1, 2, 3])
    assert rates["steady_steps"] == 3 and rates["steps_per_s"] == pytest.approx(0.75)
    assert rates["tokens_per_s"] == pytest.approx(60 / 4) and rates["loss_tokens_per_s"] == pytest.approx(6 / 4)
    assert rates["step_s_median"] == pytest.approx(1.0)
    assert sft_lora.throughput(records[:1], [1], [1]) == {}


def adapter_for(tree_shapes, rank, rng):
    adapter = {}
    for module, (contract, remain) in tree_shapes.items():
        adapter[f"{module}/kernel_lora_a"] = rng.normal(size=contract + (rank,))
        adapter[f"{module}/kernel_lora_b"] = rng.normal(size=(rank,) + remain)
    return adapter


def test_lora_targets_match_by_path_suffix_and_refuse_ambiguity():
    rng = np.random.default_rng(0)
    tree = {"params": {"decoder": {"layers_0": {"self_attention": {"query": {"kernel": np.zeros((6, 2, 3))}},
                                                "mlp": {"wi_0": {"kernel": np.zeros((6, 5))}}}}}}
    adapter = adapter_for({"decoder/layers_0/self_attention/query": ((6,), (2, 3)),
                           "decoder/layers_0/mlp/wi_0": ((6,), (5,))}, 4, rng)
    targets = translate.lora_targets(adapter, tree)
    assert [t[0] for t in targets] == ["decoder/layers_0/mlp/wi_0", "decoder/layers_0/self_attention/query"]
    assert targets[1][1] is tree["params"]["decoder"]["layers_0"]["self_attention"]["query"]
    with pytest.raises(ValueError, match="0 kernels match"):
        translate.lora_targets(adapter_for({"decoder/layers_1/mlp/wi_0": ((6,), (5,))}, 4, rng), tree)
    with pytest.raises(ValueError, match="without both"):
        translate.lora_modules({"decoder/layers_0/mlp/wi_0/kernel_lora_a": np.zeros((6, 4))})
    with pytest.raises(ValueError, match="unexpected key"):
        translate.lora_modules({"decoder/layers_0/mlp/wi_0/kernel": np.zeros((6, 4))})


@pytest.mark.parametrize("contract, remain", [((6,), (2, 3)), ((2, 3), (6,)), ((6,), (5,))])
def test_lora_delta_is_the_forward_update(contract, remain):
    """x.(W + delta) = x.W + (x.A).B.scale, contracting x with the kernel's leading axes (MaxText's DenseGeneral)."""
    rng = np.random.default_rng(1)
    rank, scale = 4, 0.5
    w = rng.normal(size=contract + remain)
    a, b = rng.normal(size=contract + (rank,)), rng.normal(size=(rank,) + remain)
    x = rng.normal(size=(7,) + contract)
    axes = (tuple(range(1, 1 + len(contract))), tuple(range(len(contract))))
    expected = np.tensordot(x, w, axes) + np.tensordot(np.tensordot(x, a, axes), b, axes=1) * scale
    delta = translate.lora_delta(a, b, w.shape, scale)
    np.testing.assert_allclose(np.tensordot(x, w + delta, axes), expected, rtol=1e-5, atol=1e-5)
    # qwix's dot_general path stores A and B flattened
    flat = translate.lora_delta(a.reshape(-1, rank), b.reshape(rank, -1), w.shape, scale)
    np.testing.assert_allclose(flat, delta, rtol=1e-6)


def test_lora_delta_refuses_a_shape_it_cannot_split():
    rng = np.random.default_rng(2)
    with pytest.raises(ValueError, match="do not split"):
        translate.lora_delta(rng.normal(size=(5, 4)), rng.normal(size=(4, 6)), (6, 5), 1.0)
    with pytest.raises(ValueError, match="differ in rank"):
        translate.lora_delta(rng.normal(size=(6, 4)), rng.normal(size=(3, 5)), (6, 5), 1.0)


def test_smoke_report_marks_identical_and_unfinished_outputs():
    import smoke_report

    compare = [{"id": 0, "origin": "flan", "ref": "w0-1", "task": "t", "en": "One.\n\nTwo.", "nl_original": None},
               {"id": 1, "origin": "pilot", "ref": "p", "task": "dpc", "en": "Three.", "nl_original": "Drie."}]

    def row(text, ended=True):
        return {"text_nl": text, "ended": ended, "tokens_out": 3}

    outputs = {"base": {0: row("Een.\n\nTwee."), 1: row("Drie.")}, "N_tag": {0: row("Een.\n\nTwee."), 1: row("Drie!")},
               "N_plain": {0: row("Een."), 1: row("Drie.", ended=False)}, "B_plain": {0: row("Een."), 1: row("Drie.")}}
    text = smoke_report.render(compare, outputs)
    assert "| LoRA arm N, met tag | 2 | 0 | 1 | 6 |" in text
    assert "| LoRA arm N, zonder tag | 2 | 1 | 1 | 6 |" in text
    assert "> Een.\n>\n> Twee." in text
    assert "**LoRA arm N, met tag** (gelijk aan het basismodel)" in text
    assert "**LoRA arm N, zonder tag** (niet afgelopen, gelijk aan het basismodel)" in text
    assert "**Nederlands origineel (de doeltekst van de pilot)**\n\n> Drie." in text


def test_local_nonzero_counts_each_shard_once_on_the_host():
    import types

    import numpy as np

    from sft_lora import local_nonzero

    a, b = np.array([[1, 1, 0, 0]]), np.array([[2, 0, 0, 0]])
    shard = lambda data, row: types.SimpleNamespace(data=data, index=(slice(row, row + 1), slice(None)))  # noqa: E731
    # rows 0 and 1 of the batch, row 0 also held by a second device (a replicated axis)
    array = types.SimpleNamespace(addressable_shards=[shard(a, 0), shard(b, 1), shard(a, 0)])
    assert local_nonzero(array) == 3
