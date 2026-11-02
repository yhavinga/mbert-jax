"""scripts/eval_windows.py on a tiny model on the CPU: fixed masks, pieces, packing and the reported groups."""
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

from test_multihost import _make_tokenizer, _stub_wandb_if_missing  # noqa: E402

BOS, EOS, PAD, MASK = 2, 3, 0, 4
SPECIALS = np.array([0, 1, 2, 3, 4])


@pytest.fixture(scope="module")
def ew():
    _stub_wandb_if_missing()
    # Other test modules already started JAX, after which train_minimal's module-level initialize() can't run
    with mock.patch("jax.distributed.initialize", side_effect=ValueError):
        import eval_windows

    return eval_windows


def test_a_document_has_one_mask_whatever_the_window(ew):
    tokens = np.arange(5, 5 + 3000) % 120 + 5
    inputs, labels = ew.mask_document(tokens, 7, 0, SPECIALS, MASK, 128)
    again_inputs, again_labels = ew.mask_document(tokens, 7, 0, SPECIALS, MASK, 128)
    assert np.array_equal(inputs, again_inputs) and np.array_equal(labels, again_labels)
    other = ew.mask_document(tokens, 8, 0, SPECIALS, MASK, 128)[1]
    assert not np.array_equal(labels, other), "another document must get other masks"
    masked = labels != -100
    assert 0.13 < masked.mean() < 0.17
    assert np.array_equal(labels[masked], tokens[masked])
    # 80% <mask>, the rest random or kept
    assert 0.75 < (inputs[masked] == MASK).mean() < 0.85
    assert np.array_equal(inputs[~masked], tokens[~masked])

    for window in (16, 64, 1024):
        pieces = ew.document_pieces(inputs, labels, window, BOS, EOS)
        assert all(piece[0].size <= window for piece in pieces)
        assert all(piece[0][0] == BOS and piece[1][0] == -100 for piece in pieces)
        assert pieces[-1][0][-1] == EOS and all(piece[0][-1] != EOS for piece in pieces[:-1])
        # The bodies of the pieces, in order, are the document: the same masked tokens at every window
        body_inputs = np.concatenate([piece[0][1:] for piece in pieces])[:-1]
        body_labels = np.concatenate([piece[1][1:] for piece in pieces])[:-1]
        assert np.array_equal(body_inputs, inputs) and np.array_equal(body_labels, labels)


def test_specials_in_a_document_are_never_masked(ew):
    tokens = np.tile(np.array([1, 0, 9]), 2000)
    labels = ew.mask_document(tokens, 0, 0, SPECIALS, MASK, 128)[1]
    assert set(np.unique(labels)) == {-100, 9}


def test_pieces_are_split_where_the_trainer_splits(ew):
    from mmbjax.training.packing import split_into_pieces

    for n in (0, 1, 13, 14, 15, 16, 40):
        tokens = np.arange(n) + 5
        inputs, labels = tokens.copy(), np.full(n, -100)
        expected = split_into_pieces([int(t) for t in tokens], 16, BOS, EOS)
        assert [list(piece[0]) for piece in ew.document_pieces(inputs, labels, 16, BOS, EOS)] == expected


def test_packing_keeps_each_piece_whole_and_in_its_own_segment(ew):
    rng = np.random.RandomState(0)
    pieces = [(rng.randint(5, 100, size=n), rng.randint(-100, 100, size=n), n > 10) for n in rng.randint(1, 17, size=50)]
    rows = ew.pack_rows(pieces, 16, PAD)
    assert rows["input_ids"].shape[1] == 16
    assert rows["attention_mask"].sum() == sum(piece[0].size for piece in pieces)
    found = []
    for r in range(len(rows["input_ids"])):
        for segment in np.unique(rows["segment_ids"][r]):
            if segment == -1:
                assert (rows["input_ids"][r][rows["segment_ids"][r] == -1] == PAD).all()
                continue
            where = np.flatnonzero(rows["segment_ids"][r] == segment)
            assert np.array_equal(where, np.arange(where[0], where[-1] + 1)), "a segment must be contiguous"
            assert np.array_equal(rows["position_ids"][r][where], np.arange(where.size))
            found.append((tuple(rows["input_ids"][r][where]), tuple(rows["labels"][r][where]), bool(rows["long_document"][r][where[0]])))
    assert sorted(found) == sorted((tuple(p[0]), tuple(p[1]), p[2]) for p in pieces)
    again = ew.pack_rows(pieces, 16, PAD)
    assert all(np.array_equal(rows[key], again[key]) for key in rows)


def test_groups(ew):
    labels = jnp.array([[5, -100, 7, 8]])
    positions = jnp.array([[0, 1, 2, 3]])
    long_document = jnp.array([[0, 1, 1, 1]])
    groups = np.asarray(ew.group_masks(labels, positions, long_document, 3))
    by_name = {name: groups[0, :, g].tolist() for g, name in enumerate(ew.GROUPS)}
    assert by_name == {
        "all": [True, False, True, True],
        "long_documents": [False, False, True, True],
        "positions_low": [True, False, True, False],
        "positions_high": [False, False, False, True],
        "long_documents_positions_low": [False, False, True, False],
    }


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory, ew):
    """A train_minimal.py checkpoint of a tiny random model, saved the way the trainer saves it."""
    import optax
    from flax.training import train_state

    tm = ew.tm
    tokenizer = _make_tokenizer()
    args = types.SimpleNamespace(hidden_size=16, num_layers=2, num_heads=2, intermediate_size=32,
                                 global_attn_every_n_layers=3, compute_dtype="float32")
    config, model = ew.build_model(args, tokenizer)
    ones = jnp.ones((1, 8), dtype=jnp.int32)
    params = model.init(jax.random.PRNGKey(3), input_ids=ones, attention_mask=ones, deterministic=True)["params"]
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optax.adamw(1e-3))
    directory = tmp_path_factory.mktemp("ckpt")
    manager = tm.ocp.CheckpointManager(directory, options=tm.ocp.CheckpointManagerOptions(create=True))
    manager.save(5, args=tm.ocp.args.StandardSave(state))
    manager.wait_until_finished()
    return directory / "5", model, params, tokenizer


def _eval_file(path, n=60):
    import pyarrow as pa
    import pyarrow.parquet as pq

    rng = np.random.RandomState(4)
    # Lengths from 1 to 40 words: below, between and above windows of 8 and 16
    texts = [" ".join(f"w{w}" for w in rng.randint(0, 100, size=rng.randint(1, 41))) for _ in range(n)]
    pq.write_table(pa.table({"text": texts}), path)
    return texts


def _reference_loss(ew, model, params, tokenizer, texts, window, seed, boundary):
    """Each piece alone, unpacked: what the packed rows must sum to."""
    sums = {name: [0.0, 0] for name in ew.GROUPS}
    special_ids = np.asarray(tokenizer.all_special_ids)
    for index, text in enumerate(texts):
        tokens = tokenizer(text, add_special_tokens=False, split_special_tokens=True)["input_ids"]
        inputs, labels = ew.mask_document(tokens, index, seed, special_ids, tokenizer.mask_token_id, tokenizer.vocab_size)
        long_document = len(tokens) + 2 > boundary
        for piece_inputs, piece_labels in ew.document_pieces(inputs, labels, window, tokenizer.bos_token_id, tokenizer.eos_token_id):
            logits = model.apply({"params": params}, input_ids=jnp.array(piece_inputs[None]), deterministic=True).logits[0]
            log_probs = np.asarray(jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1))
            for position, label in enumerate(piece_labels):
                if label == -100:
                    continue
                nll = -float(log_probs[position, label])
                names = ["all"] + (["long_documents"] if long_document else [])
                names += ["positions_low"] if position < boundary else ["positions_high"]
                names += ["long_documents_positions_low"] if long_document and position < boundary else []
                for name in names:
                    sums[name][0] += nll
                    sums[name][1] += 1
    return sums


def test_window_eval_matches_unpacked_pieces_and_counts_the_same_masks(ew, tiny_checkpoint, tmp_path, monkeypatch):
    step_dir, model, params, tokenizer = tiny_checkpoint
    texts = _eval_file(tmp_path / "held_out.parquet")
    monkeypatch.setattr(ew, "AutoTokenizer", types.SimpleNamespace(from_pretrained=lambda *_a, **_k: _make_tokenizer()))
    result = ew.main([
        "--checkpoint", str(step_dir), "--eval-file", str(tmp_path / "held_out.parquet"), "--tokenizer", "fake",
        "--windows", "8", "16", "--tokens-per-device", "32", "--seed", "11", "--hidden-size", "16", "--num-layers", "2",
        "--num-heads", "2", "--intermediate-size", "32", "--output", str(tmp_path / "out.json"),
    ])
    assert (tmp_path / "out.json").exists()
    short, long = result["windows"]["8"]["groups"], result["windows"]["16"]["groups"]
    assert set(short) == {"all", "long_documents"}
    assert set(long) == set(ew.GROUPS)
    # The same masked tokens at both windows
    assert short["all"]["masked"] == long["all"]["masked"] > 0
    assert short["long_documents"]["masked"] == long["long_documents"]["masked"] > 0
    assert long["positions_low"]["masked"] + long["positions_high"]["masked"] == long["all"]["masked"]
    assert long["positions_high"]["masked"] > 0
    assert result["windows"]["16"]["complete"] and result["windows"]["16"]["long_documents"] == sum(
        len(tokenizer(t, add_special_tokens=False)["input_ids"]) + 2 > 8 for t in texts)

    for window, groups in (("8", short), ("16", long)):
        reference = _reference_loss(ew, model, params, tokenizer, texts, int(window), 11, 8)
        for name, values in groups.items():
            total, count = reference[name]
            assert values["masked"] == count, (window, name)
            assert values["loss"] == pytest.approx(total / count, rel=1e-5), (window, name)

    # A model is measured on these masks whatever the seed of anything else; another --seed gives others
    other = ew.main([
        "--checkpoint", str(step_dir), "--eval-file", str(tmp_path / "held_out.parquet"), "--tokenizer", "fake",
        "--windows", "8", "16", "--tokens-per-device", "32", "--seed", "12", "--hidden-size", "16", "--num-layers", "2",
        "--num-heads", "2", "--intermediate-size", "32",
    ])
    assert other["windows"]["16"]["groups"]["all"]["loss"] != long["all"]["loss"]


def test_max_batches_marks_the_result_incomplete(ew, tiny_checkpoint, tmp_path, monkeypatch):
    step_dir = tiny_checkpoint[0]
    _eval_file(tmp_path / "held_out.parquet")
    monkeypatch.setattr(ew, "AutoTokenizer", types.SimpleNamespace(from_pretrained=lambda *_a, **_k: _make_tokenizer()))
    result = ew.main([
        "--checkpoint", str(step_dir), "--eval-file", str(tmp_path / "held_out.parquet"), "--tokenizer", "fake",
        "--windows", "8", "16", "--tokens-per-device", "32", "--hidden-size", "16", "--num-layers", "2",
        "--num-heads", "2", "--intermediate-size", "32", "--max-batches", "1",
    ])
    assert result["windows"]["16"]["batches"] == 1
    assert not result["windows"]["16"]["complete"]


def test_a_checkpoint_of_another_model_is_refused(ew, tiny_checkpoint, tmp_path, monkeypatch):
    step_dir = tiny_checkpoint[0]
    _eval_file(tmp_path / "held_out.parquet")
    monkeypatch.setattr(ew, "AutoTokenizer", types.SimpleNamespace(from_pretrained=lambda *_a, **_k: _make_tokenizer()))
    with pytest.raises(ValueError, match="not those of this model"):
        ew.main(["--checkpoint", str(step_dir), "--eval-file", str(tmp_path / "held_out.parquet"), "--tokenizer", "fake",
                 "--windows", "8", "16", "--tokens-per-device", "32", "--hidden-size", "16", "--num-layers", "3",
                 "--num-heads", "2", "--intermediate-size", "32"])
