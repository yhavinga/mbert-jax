"""DBRD and SICK-NL through scripts/dumb/finetune.py: the same batches and results as at 03cd832.

The goldens were recorded at 03cd832 (origin/corpus-runs), before the tasks of experiments/dumb-extra were added,
with the VM's versions (jax 0.11.0, flax 0.12.8, optax 0.2.6, datasets 2.18.0). Only the downloads are replaced: the
datasets by small synthetic ones in the schema of `dbrd` and `maximedb/sick_nl`, the tokenizer by a fixed function;
data.py's loaders, the split, the shuffles, the collation, the steps and the evaluation are the real ones. A
difference here means the DBRD/SICK-NL numbers of the earlier rounds and of new runs are no longer comparable.
"""
import hashlib
import json
import pickle
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "dumb"))

import data  # noqa: E402
import finetune  # noqa: E402
from datasets import Dataset, DatasetDict  # noqa: E402

from mmbjax.configuration_mmbbert import MMBertConfig  # noqa: E402
from mmbjax.heads.classification import ModernBertForSequenceClassification  # noqa: E402

WORDS = ["de", "het", "een", "boek", "goed", "slecht", "man", "vrouw", "loopt", "zit", "niet", "wel", "mooi", "saai"]


class FixedTokenizer:
    """ids of words by their letters; a pair as the real template, <s> A <s> B. Truncation as in the real calls."""

    pad_token_id = 0

    @staticmethod
    def _ids(text):
        return [5 + sum(map(ord, word)) % 59 for word in text.split()]

    def __call__(self, first, second=None, truncation=False, max_length=None, padding=False):
        assert truncation is True and padding is False
        rows = []
        for i, text in enumerate(first):
            ids = [1] + self._ids(text)
            if second is not None:
                ids += [1] + self._ids(second[i])
            rows.append(ids[:max_length])
        return {"input_ids": rows, "attention_mask": [[1] * len(ids) for ids in rows]}


def sentence(rng, n):
    return " ".join(rng.choice(WORDS, size=n))


def fake_dbrd(rng):
    def split(n):
        return Dataset.from_dict({"text": [sentence(rng, rng.randint(3, 40)) for _ in range(n)],
                                  "label": [int(x) for x in rng.randint(0, 2, size=n)]})
    return DatasetDict({"train": split(700), "test": split(96), "unsupervised": split(4)})


def fake_sick(rng):
    columns = ["pair_ID", "sentence_A", "sentence_B", "entailment_label", "relatedness_score", "entailment_AB",
               "entailment_BA", "sentence_A_original", "sentence_B_original", "sentence_A_dataset",
               "sentence_B_dataset", "SemEval_set", "label_seq2seq", "label"]

    def split(n):
        rows = {c: [] for c in columns}
        for i in range(n):
            label = int(rng.randint(0, 3))
            values = {"pair_ID": i, "sentence_A": sentence(rng, rng.randint(3, 12)), "sentence_B": sentence(rng, rng.randint(3, 12)),
                      "entailment_label": ["CONTRADICTION", "NEUTRAL", "ENTAILMENT"][label], "relatedness_score": float(rng.rand() * 4 + 1),
                      "entailment_AB": "A_neutral_B", "entailment_BA": "B_neutral_A", "sentence_A_original": "a", "sentence_B_original": "b",
                      "sentence_A_dataset": "FLICKR", "sentence_B_dataset": "FLICKR", "SemEval_set": "TRAIN", "label_seq2seq": "x", "label": label}
            for c in columns:
                rows[c].append(values[c])
        return Dataset.from_dict(rows)
    return DatasetDict({"train": split(160), "validation": split(64), "test": split(96)})


def tiny_checkpoint(path):
    config = MMBertConfig(vocab_size=64, hidden_size=32, intermediate_size=32, num_hidden_layers=3, num_attention_heads=2,
                          max_position_embeddings=512, local_attention=4, global_attn_every_n_layers=3, pad_token_id=0,
                          classifier_pooling="cls")
    model = ModernBertForSequenceClassification(config)
    params = model.init(jax.random.PRNGKey(7), jnp.ones((1, 8), dtype=jnp.int32), deterministic=True)["params"]
    path.mkdir()
    (path / "params.pkl").write_bytes(pickle.dumps({"model": jax.device_get(params["model"]), "prediction_head": jax.device_get(params["head"])}))
    (path / "config.json").write_text(json.dumps({**config.to_dict(), "tokenizer_name": "unused"}))
    return path


def run(tmp_path, monkeypatch, task, extra_args):
    """results.json of finetune.main() and, per train step, a digest of its batch and its loss."""
    datasets = {"dbrd": fake_dbrd(np.random.RandomState(1)), "maximedb/sick_nl": fake_sick(np.random.RandomState(2))}
    monkeypatch.setattr(data, "load_dataset", lambda name: datasets[name])
    monkeypatch.setattr(finetune.AutoTokenizer, "from_pretrained", lambda name: FixedTokenizer())
    steps = []
    real_step = finetune.p_train_step

    def recording_step(state, batch):
        state, metrics = real_step(state, batch)
        digest = hashlib.sha256(b"".join(np.ascontiguousarray(batch[k]).tobytes() for k in ("input_ids", "attention_mask", "labels")))
        steps.append((digest.hexdigest()[:16], float(metrics["loss"][0])))
        return state, metrics

    monkeypatch.setattr(finetune, "p_train_step", recording_step)
    checkpoint = tiny_checkpoint(tmp_path / "ckpt")
    output = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["finetune.py", "--checkpoint", str(checkpoint), "--task", task, "--output", str(output), *extra_args])
    finetune.main()
    results = json.loads((output / "results.json").read_text())
    # How far the encoder moved from the checkpoint, in the params finetune.py saved (those of the best validation
    # epoch): more sensitive to the optimizer and its schedule than the losses of a model this small
    start = pickle.loads((checkpoint / "params.pkl").read_bytes())["model"]
    saved = pickle.loads((output / "params.pkl").read_bytes())["model"]
    moved = jax.tree_util.tree_map(lambda a, b: float(np.sum((np.asarray(a, np.float64) - np.asarray(b, np.float64)) ** 2)), saved, start)
    results["encoder_update_norm"] = float(np.sqrt(sum(jax.tree_util.tree_leaves(moved))))
    return results, steps


def batches_digest(steps):
    return hashlib.sha256("".join(d for d, _ in steps).encode()).hexdigest()[:16]


@pytest.mark.skipif(jax.local_device_count() != 1, reason="the goldens are of one CPU device")
def test_sicknl_with_every_default_is_the_computation_of_03cd832(tmp_path, monkeypatch):
    results, steps = run(tmp_path, monkeypatch, "sicknl", [])
    # The defaults: 3 epochs, 32 per device, lr 2e-5, length 512, no padding multiple, seed 42, no accumulation
    assert {k: results[k] for k in ("epochs", "learning_rate", "batch_size", "total_batch_size", "gradient_accumulation_steps",
                                    "max_length", "pad_to_multiple_of")} == {
        "epochs": 3, "learning_rate": 2e-5, "batch_size": 32, "total_batch_size": 32, "gradient_accumulation_steps": 1,
        "max_length": 512, "pad_to_multiple_of": 1}
    assert len(steps) == GOLDEN["sicknl"]["steps"]
    assert batches_digest(steps) == GOLDEN["sicknl"]["batches"]
    np.testing.assert_allclose([loss for _, loss in steps], GOLDEN["sicknl"]["losses"], rtol=1e-5)
    for key in ("best_val_accuracy", "test_accuracy", "test_loss"):
        assert results[key] == pytest.approx(GOLDEN["sicknl"][key], rel=1e-5), key
    assert results["encoder_update_norm"] == pytest.approx(GOLDEN["sicknl"]["encoder_update_norm"], rel=1e-4)


@pytest.mark.skipif(jax.local_device_count() != 1, reason="the goldens are of one CPU device")
def test_dbrd_with_accumulation_is_the_computation_of_03cd832(tmp_path, monkeypatch):
    results, steps = run(tmp_path, monkeypatch, "dbrd", ["--epochs", "2", "--batch-size", "24", "--gradient-accumulation-steps", "2",
                                                           "--learning-rate", "5e-5", "--max-length", "32", "--seed", "43"])
    assert len(steps) == GOLDEN["dbrd"]["steps"]
    assert batches_digest(steps) == GOLDEN["dbrd"]["batches"]
    np.testing.assert_allclose([loss for _, loss in steps], GOLDEN["dbrd"]["losses"], rtol=1e-5)
    for key in ("best_val_accuracy", "test_accuracy", "test_loss"):
        assert results[key] == pytest.approx(GOLDEN["dbrd"][key], rel=1e-5), key
    assert results["encoder_update_norm"] == pytest.approx(GOLDEN["dbrd"]["encoder_update_norm"], rel=1e-4)


GOLDEN = {
    "sicknl": {
        "steps": 15, "batches": "04567700ee45d68a",
        "losses": [1.1234406232833862, 1.120307445526123, 1.0605616569519043, 1.1063398122787476, 1.1060794591903687,
                   1.0872156620025635, 1.0923514366149902, 1.1039347648620605, 1.1053920984268188, 1.1229221820831299,
                   1.070871353149414, 1.1388459205627441, 1.121468424797058, 1.0783056020736694, 1.098593831062317],
        "best_val_accuracy": 0.328125, "test_accuracy": 0.3125, "test_loss": 1.1052872737248738,
        "encoder_update_norm": 0.00810131595657861,
    },
    "dbrd": {
        "steps": 16, "batches": "d76b4e281764180f",
        "losses": [0.7146642208099365, 0.6815716624259949, 0.7052521705627441, 0.7285706996917725, 0.6969965696334839,
                   0.6898248195648193, 0.7174663543701172, 0.6967031955718994, 0.7027593851089478, 0.6895670294761658,
                   0.7080458402633667, 0.7018599510192871, 0.6840364336967468, 0.7069970965385437, 0.6952719688415527,
                   0.7163724303245544],
        "best_val_accuracy": 0.5145833333333333, "test_accuracy": 0.5208333333333334, "test_loss": 0.6923959702253342,
        "encoder_update_norm": 0.022807823533285046,
    },
}

if __name__ == "__main__":
    # Prints the observed values, to record them as GOLDEN at a commit whose computation is the reference
    import tempfile

    class Patch:
        def __init__(self):
            self.undo = []

        def setattr(self, target, name, value):
            self.undo.append((target, name, getattr(target, name)))
            setattr(target, name, value)

    for task, args in (("sicknl", []), ("dbrd", ["--epochs", "2", "--batch-size", "24", "--gradient-accumulation-steps", "2",
                                                  "--learning-rate", "5e-5", "--max-length", "32", "--seed", "43"])):
        patch = Patch()
        with tempfile.TemporaryDirectory() as tmp:
            results, steps = run(Path(tmp), patch, task, args)
        for target, name, value in reversed(patch.undo):
            setattr(target, name, value)
        print(json.dumps({task: {"steps": len(steps), "batches": batches_digest(steps), "losses": [loss for _, loss in steps],
                                 **{k: results[k] for k in ("best_val_accuracy", "test_accuracy", "test_loss", "encoder_update_norm")}}}))
