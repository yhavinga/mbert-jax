#!/usr/bin/env python3
"""COMET-22 (Unbabel/wmt22-comet-da, Apache-2.0) with a reference, on the CPU, for the evaluation sets of the en-nl
translator (evalsets.py; hammond mbert/designs/en-nl-translator.md §4). It runs in a venv of its own
(setup_comet.sh: unbabel-comet 2.2.7 and torch for CPU, pinned in comet-requirements.txt), because unbabel-comet
needs numpy < 2 and transformers < 5, which the venvs of MetricX and MaxText do not have.

    python comet_score.py setup --root /dev/shm/translator/comet
    python comet_score.py score --root /dev/shm/translator/comet --pairs pairs-w0.parquet --out comet-w0.parquet \\
        --time-first 100 --timing comet-timing-w0.json

setup downloads the model at a fixed revision (and checks the sha256 of its checkpoint) and the config and tokenizer
of its encoder, xlm-roberta-large, at a fixed revision. COMET itself would fetch the encoder's files by name at
whatever revision the Hub has, so setup writes a model directory of its own: the checkpoint as a link, and the
model's hparams.yaml with pretrained_model pointing at the local encoder files. Loading then reads nothing from the
network. setup then loads the model, checks that every parameter came from the checkpoint (COMET loads with
strict=False, which would leave a parameter whose name does not match at its random initial value without a word),
and scores two sanity triples: a translation equal to its reference must score high, and clearly higher than an
unrelated sentence.

score gives one score per row of --pairs (columns id, source, hypothesis, reference; a hypothesis may be empty), in
the order of the file, in the column comet22. With --time-first n it first scores the first n rows on their own and
reports their duration and the projection for all rows (a COMET_TIMING line, and --timing as JSON), then the rest.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

MODEL = ("Unbabel/wmt22-comet-da", "2760a223ac957f30acfb18c8aa649b01cf1d75f2",
         "e213091cde220f97b89f8bdfa750c458cfea741ad62affb455b59900210ff2af")  # repo, revision, sha256 of model.ckpt
ENCODER = ("FacebookAI/xlm-roberta-large", "c23d21b0620b635a76227c604d44e43a9f0ee389")
ENCODER_FILES = ["config.json", "sentencepiece.bpe.model", "tokenizer.json", "tokenizer_config.json"]
# A translation equal to its reference scores near 1 with COMET-22; an unrelated sentence clearly lower. A model with
# a random or missing part scores both about the same. The bounds leave room on both sides.
SANITY_SOURCE = "The committee will meet again next week to discuss the budget."
SANITY_REFERENCE = "De commissie komt volgende week opnieuw bijeen om de begroting te bespreken."
SANITY_UNRELATED = "Mijn kat slaapt het liefst op de verwarming."
SANITY_COPY_LEAST = 0.8
SANITY_GAP_LEAST = 0.15


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def local_hparams(text, encoder_dir):
    """The model's hparams.yaml with pretrained_model set to the local encoder directory"""
    lines = text.splitlines()
    hits = [k for k, line in enumerate(lines) if line.startswith("pretrained_model:")]
    if len(hits) != 1:
        raise SystemExit(f"hparams.yaml has {len(hits)} lines 'pretrained_model:', expected 1")
    lines[hits[0]] = f"pretrained_model: {encoder_dir}"
    return "\n".join(lines) + "\n"


def prepare_local(root, model_dir, encoder_dir):
    """<root>/model: checkpoints/model.ckpt (a link to the downloaded checkpoint) and the edited hparams.yaml.
    Returns the checkpoint path to load."""
    local = Path(root) / "model"
    (local / "checkpoints").mkdir(parents=True, exist_ok=True)
    link = local / "checkpoints" / "model.ckpt"
    target = (Path(model_dir) / "checkpoints" / "model.ckpt").resolve()
    if link.is_symlink() or link.exists():
        if link.resolve() != target:
            raise SystemExit(f"{link} points to {link.resolve()}, not {target}")
    else:
        link.symlink_to(target)
    (local / "hparams.yaml").write_text(local_hparams((Path(model_dir) / "hparams.yaml").read_text(),
                                                      Path(encoder_dir).resolve()))
    return link


def load(checkpoint):
    """The COMET model of the prepared directory, on the CPU, without the network"""
    from comet import load_from_checkpoint

    os.environ["HF_HUB_OFFLINE"] = "1"
    # Lightning hands torch.load weights_only=None, which torch >= 2.6 reads as True; the checkpoint (a pinned sha256)
    # holds Lightning's own objects next to the tensors
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    return load_from_checkpoint(str(checkpoint), reload_hparams=True, local_files_only=True)


def verify_parameters(model, checkpoint):
    """Every tensor of the model's state dict equals the checkpoint's tensor of that name"""
    import torch

    saved = torch.load(str(checkpoint), map_location="cpu", weights_only=False)["state_dict"]
    missing, differ = [], []
    for name, value in model.state_dict().items():
        if name not in saved:
            missing.append(name)
        elif saved[name].shape != value.shape or not torch.equal(saved[name].to(value.dtype), value):
            differ.append(name)
    if missing or differ:
        raise SystemExit(f"the loaded model does not match its checkpoint: {len(missing)} tensors missing "
                         f"({missing[:5]}), {len(differ)} differ ({differ[:5]})")
    print(f"parameters: all {len(model.state_dict())} tensors of the model are those of the checkpoint", flush=True)


def predict(model, samples, batch_size):
    """COMET's segment scores, in the order of samples"""
    if not samples:
        return []
    out = model.predict(samples, batch_size=batch_size, gpus=0, progress_bar=False, accelerator="cpu")
    scores = list(out.scores)
    if len(scores) != len(samples):
        raise SystemExit(f"COMET returned {len(scores)} scores for {len(samples)} samples")
    return [float(s) for s in scores]


def sanity(model, batch_size):
    """The copy scores at least SANITY_COPY_LEAST and at least SANITY_GAP_LEAST above the unrelated sentence"""
    copy, unrelated = predict(model, [{"src": SANITY_SOURCE, "mt": mt, "ref": SANITY_REFERENCE}
                                      for mt in (SANITY_REFERENCE, SANITY_UNRELATED)], batch_size)
    print(f"SANITY {json.dumps({'copy': round(copy, 4), 'unrelated': round(unrelated, 4)})}", flush=True)
    if copy < SANITY_COPY_LEAST or copy - unrelated < SANITY_GAP_LEAST:
        raise SystemExit(f"COMET's sanity triples: a copy of the reference scores {copy:.4f} (at least "
                         f"{SANITY_COPY_LEAST}), an unrelated sentence {unrelated:.4f} (at least {SANITY_GAP_LEAST} "
                         "lower)")


def setup(args):
    from huggingface_hub import snapshot_download

    root = Path(args.root)
    model_dir, encoder_dir = root / "wmt22-comet-da", root / "xlm-roberta-large"
    snapshot_download(MODEL[0], revision=MODEL[1], local_dir=str(model_dir))
    digest = sha256(model_dir / "checkpoints" / "model.ckpt")
    if digest != MODEL[2]:
        raise SystemExit(f"{MODEL[0]}@{MODEL[1]}: model.ckpt has sha256 {digest}, expected {MODEL[2]}")
    snapshot_download(ENCODER[0], revision=ENCODER[1], local_dir=str(encoder_dir), allow_patterns=ENCODER_FILES)
    checkpoint = prepare_local(root, model_dir, encoder_dir)
    started = time.time()
    model = load(checkpoint)
    print(f"loaded in {time.time() - started:.0f} s", flush=True)
    verify_parameters(model, checkpoint)
    sanity(model, args.batch_size)
    print(f"COMET ready: {MODEL[0]}@{MODEL[1]} (sha256 {digest[:12]}), encoder {ENCODER[0]}@{ENCODER[1]}", flush=True)


def score(args):
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pq.read_table(args.pairs, columns=["id", "source", "hypothesis", "reference"]).to_pylist()
    for r in table:
        if not r["source"] or not r["reference"] or r["hypothesis"] is None:
            raise SystemExit(f"{args.pairs}: row {r['id']} lacks a source, reference or hypothesis")
    import torch

    if args.threads:
        torch.set_num_threads(args.threads)
    samples = [{"src": r["source"], "mt": r["hypothesis"], "ref": r["reference"]} for r in table]
    model = load(Path(args.root) / "model" / "checkpoints" / "model.ckpt")
    first = min(args.time_first, len(samples))
    timing = {"rows": len(samples), "threads": torch.get_num_threads(), "batch_size": args.batch_size}
    scores = []
    if first:
        started = time.time()
        scores += predict(model, samples[:first], args.batch_size)
        seconds = time.time() - started
        timing.update({"first": first, "first_seconds": round(seconds, 1),
                       "projected_seconds": round(seconds * len(samples) / first)})
        print(f"COMET_TIMING {json.dumps(timing)}", flush=True)
    started = time.time()
    scores += predict(model, samples[first:], args.batch_size)
    timing["rest_seconds"] = round(time.time() - started, 1)
    pq.write_table(pa.table({"id": [r["id"] for r in table], "comet22": [round(s, 4) for s in scores]}), args.out)
    if args.timing:
        Path(args.timing).write_text(json.dumps(timing, indent=1))
    print(f"COMET {json.dumps({**timing, 'mean': round(sum(scores) / max(len(scores), 1), 4)})}", flush=True)
    print(f"wrote {args.out}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("setup", "score"):
        p = sub.add_parser(name)
        p.add_argument("--root", default="/dev/shm/translator/comet", help="Where the model is downloaded")
        p.add_argument("--batch-size", type=int, default=32)
        if name == "score":
            p.add_argument("--pairs", required=True, help="parquet with id, source, hypothesis, reference")
            p.add_argument("--out", required=True)
            p.add_argument("--time-first", type=int, default=0, help="Score and time this many rows first")
            p.add_argument("--timing", default=None, help="Where to write the timing as JSON")
            p.add_argument("--threads", type=int, default=0, help="torch threads (default: torch's choice)")
    args = parser.parse_args(argv)
    {"setup": setup, "score": score}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
