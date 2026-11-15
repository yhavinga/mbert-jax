"""experiments/final-v2/memtest.py on a tiny model on the CPU: fixed masks per document, one document per row, the
per-document metrics against a direct computation, and the paired statistics."""
import sys
import types
from pathlib import Path
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "experiments" / "final-v2"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_multihost import VOCAB_WORDS, _make_tokenizer, _stub_wandb_if_missing  # noqa: E402

MODEL_FLAGS = ["--hidden-size", "16", "--num-layers", "2", "--num-heads", "2", "--intermediate-size", "32"]


@pytest.fixture(scope="module")
def mt():
    _stub_wandb_if_missing()
    with mock.patch("jax.distributed.initialize", side_effect=ValueError):
        import memtest

    return memtest


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory, mt):
    import optax
    from flax.training import train_state

    tm = mt.tm
    tokenizer = _make_tokenizer()
    args = types.SimpleNamespace(hidden_size=16, num_layers=2, num_heads=2, intermediate_size=32,
                                 global_attn_every_n_layers=3, compute_dtype="float32")
    _, model = mt.ew.build_model(args, tokenizer)
    ones = jnp.ones((1, 8), dtype=jnp.int32)
    params = model.init(jax.random.PRNGKey(5), input_ids=ones, attention_mask=ones, deterministic=True)["params"]
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optax.adamw(1e-3))
    directory = tmp_path_factory.mktemp("ckpt")
    manager = tm.ocp.CheckpointManager(directory, options=tm.ocp.CheckpointManagerOptions(create=True))
    manager.save(7, args=tm.ocp.args.StandardSave(state))
    manager.wait_until_finished()
    return directory / "7", model, params, tokenizer


def _sample(path, n=6, lengths=(20, 40)):
    rng = np.random.RandomState(1)
    tokenizer = _make_tokenizer()
    texts = [" ".join(f"w{w}" for w in rng.randint(0, VOCAB_WORDS, size=rng.randint(*lengths))) for _ in range(2 * n)]
    tokens = [len(tokenizer(t, add_special_tokens=False)["input_ids"]) for t in texts]
    pq.write_table(pa.table({"text": texts, "split": ["test"] * n + ["train"] * n, "source": [str(k) for k in range(2 * n)],
                             "tokens": pa.array(tokens, pa.int32())}), path)
    return texts


def _run(mt, step_dir, sample, monkeypatch, out=None):
    monkeypatch.setattr(mt, "AutoTokenizer", types.SimpleNamespace(from_pretrained=lambda *_a, **_k: _make_tokenizer()))
    argv = ["--checkpoint", str(step_dir), "--sample", str(sample), "--tokenizer", "fake", "--window", "64",
            "--rows-per-device", "4", *MODEL_FLAGS]
    return mt.main(argv + (["--output", str(out)] if out else []))


def _direct(mt, model, params, tokenizer, tokens, inputs, labels):
    """One row alone, unpadded: the sum of the loss and the right argmaxes on its labels."""
    ids = np.r_[tokenizer.bos_token_id, inputs, tokenizer.eos_token_id]
    full_labels = np.r_[-100, labels, -100]
    logits = model.apply({"params": params}, input_ids=jnp.array(ids[None]), deterministic=True).logits[0]
    log_probs = np.asarray(jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1))
    keep = full_labels != -100
    nll = -log_probs[np.arange(ids.size), np.maximum(full_labels, 0)][keep].sum()
    correct = (log_probs.argmax(-1) == full_labels)[keep].sum()
    return nll, correct, keep.sum()


def test_each_documents_metrics_are_those_of_its_row_alone(mt, tiny_checkpoint, tmp_path, monkeypatch):
    step_dir, model, params, tokenizer = tiny_checkpoint
    texts = _sample(tmp_path / "s.parquet")
    result = _run(mt, step_dir, tmp_path / "s.parquet", monkeypatch, tmp_path / "out.json")
    assert (tmp_path / "out.json").exists()
    assert result["documents_per_split"] == 6 and set(result["splits"]) == {"test", "train"}
    special = np.asarray(tokenizer.all_special_ids)
    for index, text in enumerate(texts):
        tokens = tokenizer(text, add_special_tokens=False)["input_ids"]
        nll = correct = count = 0
        for seed in mt.SEEDS:
            a, b, c = _direct(mt, model, params, tokenizer, tokens,
                              *mt.ew.mask_document(tokens, index, seed, special, tokenizer.mask_token_id, tokenizer.vocab_size))
            nll, correct, count = nll + a, correct + b, count + c
        assert result["per_document"]["mlm_loss"][index] == pytest.approx(nll / count, rel=1e-4)
        assert result["per_document"]["mlm_accuracy"][index] == pytest.approx(correct / count, abs=1e-6)
        span_acc = exact = 0.0
        for seed in mt.SEEDS:
            _, b, c = _direct(mt, model, params, tokenizer, tokens, *mt.span_mask(tokens, index, seed, tokenizer.mask_token_id))
            assert c == mt.SPAN
            span_acc += b / mt.SPAN / len(mt.SEEDS)
            exact += float(b == mt.SPAN) / len(mt.SEEDS)
        assert result["per_document"]["span_accuracy"][index] == pytest.approx(span_acc, abs=1e-6)
        assert result["per_document"]["span_exact"][index] == pytest.approx(exact, abs=1e-6)
    means = np.array(result["per_document"]["mlm_loss"])
    assert result["splits"]["train"]["mlm_loss"] == pytest.approx(means[6:].mean(), rel=1e-6)
    assert result["train_minus_test"]["mlm_loss"]["difference"] == pytest.approx(means[6:].mean() - means[:6].mean(), rel=1e-5)
    # The masks belong to the documents: a second run gives the same numbers
    again = _run(mt, step_dir, tmp_path / "s.parquet", monkeypatch)
    assert again["per_document"] == result["per_document"]


def test_a_document_shorter_than_a_span_or_longer_than_a_row_is_refused(mt):
    with pytest.raises(ValueError, match="fewer than a span"):
        mt.span_mask(list(range(5, 15)), 0, 0, 4)
    with pytest.raises(ValueError, match="does not fit"):
        mt.row_of(np.arange(63), np.full(63, -100), 64, 2, 3, 0)


def test_a_sample_not_test_then_train_is_refused(mt, tmp_path):
    pq.write_table(pa.table({"text": ["a", "b"], "split": ["train", "test"], "tokens": pa.array([1, 1], pa.int32())}),
                   tmp_path / "bad.parquet")
    with pytest.raises(ValueError, match="not N test documents"):
        mt.read_sample(tmp_path / "bad.parquet")


def test_paired_statistics(mt):
    rng = np.random.default_rng(3)
    test = rng.normal(size=200)
    shifted = mt.paired(test + 1.0, test)
    assert shifted["difference"] == pytest.approx(1.0)
    assert shifted["ci95"][0] == pytest.approx(1.0) and shifted["p_sign_flip"] < 0.001
    noise = mt.paired(rng.normal(size=200), rng.normal(size=200))
    assert noise["ci95"][0] < noise["difference"] < noise["ci95"][1]
    assert noise["p_sign_flip"] > 0.01
