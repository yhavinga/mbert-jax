"""experiments/en-nl-translator/comet_score.py with unbabel-comet itself, on a tiny randomly initialized COMET model
(a 2-layer XLM-R encoder and a word-level tokenizer, built here, nothing downloaded): the local model directory, loading
it without the network, the check that every parameter came from the checkpoint, and scoring in the order of the file
with the timing of the first rows. Runs only where unbabel-comet is installed (the venv of setup_comet.sh):
    <comet venv>/bin/python -m pytest -q --noconftest tests/test_en_nl_comet.py
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "en-nl-translator"))

import comet_score  # noqa: E402

WORDS = "the a cat dog sat on mat de het kat hond zat op een . , house huis".split()


def tiny_encoder(path):
    """config.json of a 2-layer XLM-R and a word-level tokenizer.json with XLM-R's special tokens"""
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import XLMRobertaConfig

    vocab = {t: k for k, t in enumerate(["<s>", "<pad>", "</s>", "<unk>"] + WORDS + ["<mask>"])}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.post_processor = processors.TemplateProcessing(single="<s> $A </s>", pair="<s> $A </s> </s> $B </s>",
                                                       special_tokens=[("<s>", 0), ("</s>", 2)])
    path.mkdir(parents=True)
    tok.save(str(path / "tokenizer.json"))
    (path / "tokenizer_config.json").write_text(json.dumps({"model_max_length": 512, "bos_token": "<s>",
                                                            "eos_token": "</s>", "sep_token": "</s>",
                                                            "cls_token": "<s>", "unk_token": "<unk>",
                                                            "pad_token": "<pad>", "mask_token": "<mask>"}))
    XLMRobertaConfig(vocab_size=len(vocab), hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                     intermediate_size=64, max_position_embeddings=514, pad_token_id=1, bos_token_id=0,
                     eos_token_id=2, type_vocab_size=1).save_pretrained(str(path))


def tiny_model(tmp_path):
    """A downloaded model's layout: <dir>/checkpoints/model.ckpt and <dir>/hparams.yaml naming the encoder by a name
    that does not exist (as xlm-roberta-large would be looked up on the Hub), and the local encoder directory"""
    import torch
    from comet.models import RegressionMetric

    encoder = tmp_path / "encoder"
    tiny_encoder(encoder)
    hparams = {"activations": "Tanh", "batch_size": 4, "class_identifier": "regression_metric", "dropout": 0.1,
               "encoder_learning_rate": 1e-06, "encoder_model": "XLM-RoBERTa", "final_activation": None,
               "hidden_sizes": [64, 32], "keep_embeddings_frozen": True, "layer": "mix", "layer_norm": False,
               "layer_transformation": "sparsemax", "layerwise_decay": 0.95, "learning_rate": 1.5e-05, "loss": "mse",
               "nr_frozen_epochs": 0.3, "optimizer": "AdamW", "pool": "avg", "train_data": ["data/1720-da.csv"],
               "validation_data": ["data/wmt-ende-newstest2021.csv", "data/wmt-enru-newstest2021.csv"]}
    torch.manual_seed(0)
    model = RegressionMetric(pretrained_model=str(encoder), load_pretrained_weights=False, local_files_only=True,
                             **{k: v for k, v in hparams.items() if k != "class_identifier"})
    downloaded = tmp_path / "download"
    (downloaded / "checkpoints").mkdir(parents=True)
    torch.save({"state_dict": model.state_dict(), "hyper_parameters": {**hparams, "pretrained_model": "no-such-model"},
                "pytorch-lightning_version": "2.6.6"}, downloaded / "checkpoints" / "model.ckpt")
    yaml = [f"{k}: {json.dumps(v) if v is not None else 'null'}" for k, v in hparams.items()]
    (downloaded / "hparams.yaml").write_text("\n".join(yaml + ["pretrained_model: no-such-model"]) + "\n")
    return downloaded, encoder, model


def test_local_hparams_points_the_encoder_at_a_directory():
    text = "class_identifier: regression_metric\npretrained_model: xlm-roberta-large\npool: avg\n"
    assert comet_score.local_hparams(text, "/x/enc") == \
        "class_identifier: regression_metric\npretrained_model: /x/enc\npool: avg\n"
    with pytest.raises(SystemExit):
        comet_score.local_hparams("pool: avg\n", "/x/enc")


def test_load_verify_and_score_a_tiny_model(tmp_path, monkeypatch):
    pytest.importorskip("comet")
    import pyarrow as pa
    import pyarrow.parquet as pq

    downloaded, encoder, original = tiny_model(tmp_path)
    root = tmp_path / "root"
    checkpoint = comet_score.prepare_local(root, downloaded, encoder)
    model = comet_score.load(checkpoint)
    comet_score.verify_parameters(model, checkpoint)
    # The scores of a sample do not depend on the others: predict in one go and one by one agree
    samples = [{"src": "the cat sat on the mat .", "mt": "de kat zat op de mat .", "ref": "de kat zat op het mat ."},
               {"src": "a dog", "mt": "", "ref": "een hond"},
               {"src": "the house , the house , the house", "mt": "het huis", "ref": "het huis , het huis"}]
    together = comet_score.predict(model, samples, batch_size=2)
    alone = [comet_score.predict(model, [s], batch_size=1)[0] for s in samples]
    assert together == pytest.approx(alone, abs=1e-5)

    pairs = tmp_path / "pairs.parquet"
    pq.write_table(pa.table({"id": ["a", "b", "c"], "source": [s["src"] for s in samples],
                             "hypothesis": [s["mt"] for s in samples], "reference": [s["ref"] for s in samples]}),
                   pairs)
    out, timing = tmp_path / "comet.parquet", tmp_path / "timing.json"
    comet_score.main(["score", "--root", str(root), "--pairs", str(pairs), "--out", str(out), "--time-first", "2",
                      "--timing", str(timing), "--batch-size", "2", "--threads", "2"])
    got = pq.read_table(out).to_pydict()
    assert got["id"] == ["a", "b", "c"]
    assert got["comet22"] == pytest.approx([round(s, 4) for s in together], abs=1e-4)
    t = json.loads(timing.read_text())
    assert t["rows"] == 3 and t["first"] == 2 and t["threads"] == 2 and "projected_seconds" in t


def test_verify_parameters_refuses_a_model_that_is_not_the_checkpoint(tmp_path):
    pytest.importorskip("comet")
    import torch

    downloaded, encoder, _ = tiny_model(tmp_path)
    checkpoint = comet_score.prepare_local(tmp_path / "root", downloaded, encoder)
    model = comet_score.load(checkpoint)
    with torch.no_grad():
        next(iter(model.parameters())).add_(1.0)
    with pytest.raises(SystemExit, match="does not match its checkpoint"):
        comet_score.verify_parameters(model, checkpoint)


def test_score_refuses_a_row_without_a_hypothesis(tmp_path):
    pytest.importorskip("pyarrow")
    import pyarrow as pa
    import pyarrow.parquet as pq

    pairs = tmp_path / "pairs.parquet"
    pq.write_table(pa.table({"id": ["a"], "source": ["x"], "hypothesis": pa.array([None], pa.string()),
                             "reference": ["y"]}), pairs)
    with pytest.raises(SystemExit, match="lacks"):
        comet_score.main(["score", "--root", str(tmp_path), "--pairs", str(pairs), "--out", str(tmp_path / "o")])
