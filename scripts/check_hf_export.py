#!/usr/bin/env python3
"""Checks of a Hugging Face export of a Dutch ModernBERT checkpoint before it is published.

Two steps, each in the environment it needs (the JAX environment has no torch, the export is checked with the
transformers version a user has):

  jax  REFERENCE: logits of the JAX model (params.pkl and config.json of convert_gcs_checkpoint.py) on fixed Dutch
       texts, tokenized with the tokenizer of the training run (config.json's tokenizer_name plus <mask>, as
       train_minimal.py adds it), into an .npz.
  hf   THE EXPORT (a directory, or a Hub repo with --revision): its tokenizer must give the reference's ids; its
       AutoModelForMaskedLM must give the reference's logits (largest absolute difference over every real position
       and the whole vocabulary at most --atol, the same top-1 token at every real position); plus fill-mask in
       three notations, the character offsets of the fast tokenizer, the special token ids of the config, and one
       fine-tuning step of AutoModelForSequenceClassification. A JSON report; exit 1 when a check fails.

  JAX_PLATFORMS=cpu python scripts/check_hf_export.py jax --flax DIR --out ref.npz
  python scripts/check_hf_export.py hf --model DIR_OR_REPO [--revision SHA] --reference ref.npz --out report.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

# Texts for the logits. The long one is ~300 tokens, so that the local layers' window of 128 tokens is exceeded.
LOGIT_TEXTS = [
    "Amsterdam is de hoofdstad van Nederland.",
    "De kat zit op de mat en kijkt naar buiten.",
    "Hij woont in 's-Hertogenbosch, toch?",
    "Eén café, twee euro (€2,00) en een paar woorden met tekens: ruïne, coördinatie, reünie.",
    (
        "Het Nederlands is een West-Germaanse taal die door ongeveer vierentwintig miljoen mensen als moedertaal "
        "wordt gesproken, vooral in Nederland, België en Suriname. De taal heeft een lange geschiedenis: de oudste "
        "bewaarde zinnen dateren uit de elfde eeuw, en in de zestiende en zeventiende eeuw groeide een "
        "standaardtaal die in drukwerk, bijbelvertalingen en bestuursstukken werd gebruikt. Tegenwoordig regelt de "
        "Nederlandse Taalunie de spelling voor Nederland, Vlaanderen en Suriname. Wie de taal leert, struikelt "
        "vaak over de lidwoorden de en het, over de plaats van het werkwoord in de bijzin en over woorden als er, "
        "die op veel verschillende manieren gebruikt worden. Ook de uitspraak van de g en de ui is voor veel "
        "leerders lastig. Toch is het Nederlands voor sprekers van het Duits of het Engels vaak goed te volgen, "
        "omdat veel woorden op elkaar lijken. In de digitale wereld is het Nederlands een middelgrote taal: er is "
        "veel tekst beschikbaar op het web, maar veel minder dan in het Engels, en taalmodellen die alleen op "
        "Nederlandse tekst getraind zijn, zijn daarom zeldzaam. Een encoder zoals deze leert de taal door woorden "
        "in zinnen te voorspellen die verborgen zijn, en kan daarna voor classificatie, het herkennen van namen of "
        "het beantwoorden van vragen worden bijgetraind."
    ),
]
# One row with a masked token at id level, as in the training: the first text with its word in [] masked
MASKED_ROW = "Amsterdam is de [hoofdstad] van Nederland."

# Fill-mask: the word in [] is one token; it is masked at id level (as in the training) and written as <mask> in
# three ways, to show which notation gives the training's ids
FILL_MASK = [
    "Amsterdam is de [hoofdstad] van Nederland.",
    "Ik drink elke ochtend een kopje [koffie].",
    "De [kat] zit op de mat.",
    "Utrecht ligt in het [midden] van het land.",
    "Het [weer] is vandaag erg mooi.",
]

# The card's pipeline example, and the same with spaces around <mask>
PIPELINE_PROMPTS = ["Amsterdam is de<mask>van Nederland.", "Amsterdam is de <mask> van Nederland."]

# Offsets: words, punctuation, accents, bytes outside the vocabulary, runs of whitespace, a literal <mask>
OFFSET_TEXTS = LOGIT_TEXTS + [
    "De hoofdstad is Amsterdam",
    " Begint met een spatie, eindigt met twee  ",
    "Twee\tregels:\nhier en daar.\n\nEn een lege regel.",
    "Emoji 😀 en tekens als „aanhalingstekens”, ‘enkele’ en — streepjes … zoals op https://www.example.nl/pad?q=1.",
    "Prijs: €12,50 (incl. 21% btw), 3½ kilo, 2×3 = 6.",
    "Amsterdam is de <mask> van Nederland.",
    "Amsterdam is de<mask>van Nederland.",
]
# A question and a context in windows, as the SQuAD-NL fine-tuning reads them (truncation of the context only)
QA_PAIR = (
    "Wat is de hoofdstad van Nederland?",
    LOGIT_TEXTS[4] + " De hoofdstad van Nederland is Amsterdam, de regering zetelt in Den Haag.",
)


def split_marked(sentence: str) -> tuple[str, str, str]:
    """'a [w] b' -> ('a ', 'w', ' b')."""
    m = re.fullmatch(r"(.*)\[([^\]]+)\](.*)", sentence)
    if m is None:
        raise ValueError(f"no [word] in {sentence!r}")
    return m.group(1), m.group(2), m.group(3)


def ids_with_word_masked(tokenizer, sentence: str) -> tuple[list[int], int]:
    """The ids of the sentence with its [word] replaced by <mask> at id level, and the word's token id.

    The word must be one token, ▁word, exactly once in the sentence: the training replaces single tokens.
    """
    before, word, after = split_marked(sentence)
    ids = list(tokenizer(before + word + after)["input_ids"])
    tokens = tokenizer.convert_ids_to_tokens(ids)
    hits = [k for k, t in enumerate(tokens) if t == "\u2581" + word]
    if len(hits) != 1:
        raise ValueError(f"{sentence!r}: {word!r} is not one token ▁{word} here: {tokens}")
    word_id = ids[hits[0]]
    ids[hits[0]] = tokenizer.mask_token_id
    return ids, word_id


def mask_notations(sentence: str, mask: str) -> dict[str, str]:
    before, _, after = split_marked(sentence)
    return {
        "spaces": before + mask + after,
        "no space before": before.rstrip(" ") + mask + after,
        "no space before or after": before.rstrip(" ") + mask + after.lstrip(" "),
    }


def logit_rows(tokenizer) -> tuple[list[str], list[list[int]]]:
    """The texts of the logits check and their ids (the tokenizer's default: <s> and the text), plus the masked row."""
    names = list(LOGIT_TEXTS)
    rows = [tokenizer(t)["input_ids"] for t in LOGIT_TEXTS]
    masked, _ = ids_with_word_masked(tokenizer, MASKED_ROW)
    names.append(MASKED_ROW + " (word masked at id level)")
    rows.append(masked)
    return names, rows


def pad_rows(rows: list[list[int]], pad_id: int) -> tuple[np.ndarray, np.ndarray]:
    width = max(len(r) for r in rows)
    ids = np.full((len(rows), width), pad_id, dtype=np.int32)
    mask = np.zeros((len(rows), width), dtype=np.int32)
    for i, r in enumerate(rows):
        ids[i, :len(r)] = r
        mask[i, :len(r)] = 1
    return ids, mask


def compare_logits(reference: np.ndarray, other: np.ndarray, attention_mask: np.ndarray) -> dict:
    """Largest absolute difference over the real positions (all of the vocabulary) and top-1 agreement there."""
    if reference.shape != other.shape:
        raise ValueError(f"logits of shape {reference.shape} and {other.shape}")
    real = attention_mask.astype(bool)
    diff = np.abs(reference.astype(np.float64) - other.astype(np.float64))[real]
    top_ref = reference.argmax(-1)[real]
    top_other = other.argmax(-1)[real]
    return {
        "positions": int(real.sum()),
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "max_abs_logit": float(np.abs(reference[real]).max()),
        "top1_equal": int((top_ref == top_other).sum()),
        "per_row_max_abs_diff": [float(np.abs(reference[i] - other[i])[real[i]].max()) for i in range(len(real))],
    }


# --- Offsets ---------------------------------------------------------------------------------------------------

BYTE_TOKEN = re.compile(r"<0x([0-9A-Fa-f]{2})>")


def offset_problems(text: str, tokens: list[str], offsets: list[tuple[int, int]], special: set[str],
                    first_is_segment_start: bool = True, prefix_space: str = "\u2581") -> tuple[list[dict], dict]:
    """Positions whose character span is not the text of their token, and some counts.

    A SentencePiece token stands for its piece with ▁ as a space; its span must hold exactly that text. Exceptions,
    each explained by the tokenizer: the first token of a segment (the start of a text, or the text after a special
    token) starts with the ▁ that the normalizer prepends, which has no character in the text; consecutive byte
    tokens (<0xNN>) of one character share its span and together are its UTF-8 bytes; a special token has an empty
    span (added by the template) or the span of its literal text. Spans follow each other without gap or overlap.
    first_is_segment_start: False for a window that starts inside a text.
    """
    problems = []
    counts = {"tokens": 0, "byte_groups": 0, "segment_starts": 0, "space_then_word": 0}
    i = 0
    segment_start = first_is_segment_start
    previous_end = None

    def follows(k, a, b):
        if previous_end is not None and a != previous_end:
            problems.append({"position": k, "token": tokens[k], "span": [a, b], "why": f"gap or overlap after {previous_end}"})

    while i < len(tokens):
        token, (a, b) = tokens[i], tuple(offsets[i])
        if token in special:
            if (a, b) != (0, 0):
                if text[a:b] != token:
                    problems.append({"position": i, "token": token, "span": [a, b], "span_text": text[a:b],
                                     "expected": token, "why": "special token whose span is not its text"})
                follows(i, a, b)
                previous_end = b
            segment_start = True
            i += 1
            continue
        if BYTE_TOKEN.fullmatch(token):
            j = i
            data = bytearray()
            while j < len(tokens) and BYTE_TOKEN.fullmatch(tokens[j]) and tuple(offsets[j]) == (a, b):
                data.append(int(BYTE_TOKEN.fullmatch(tokens[j]).group(1), 16))
                j += 1
            expected = data.decode("utf-8", errors="replace")
            counts["byte_groups"] += 1
        else:
            j = i + 1
            expected = token.replace(prefix_space, " ")
        if segment_start:
            counts["segment_starts"] += 1
            if not expected.startswith(" "):
                problems.append({"position": i, "token": token, "span": [a, b], "why": "segment start without ▁"})
            expected = expected[1:]
        elif expected.startswith(" ") and expected.strip():
            counts["space_then_word"] += 1
        counts["tokens"] += j - i
        if text[a:b] != expected:
            problems.append({"position": i, "token": tokens[i:j] if j > i + 1 else token, "span": [a, b],
                             "span_text": text[a:b], "expected": expected, "why": "span is not the token's text"})
        if expected:
            follows(i, a, b)
            previous_end = b
        segment_start = False
        i = j
    return problems, counts


def check_offsets(tokenizer) -> dict:
    """Offsets of return_offsets_mapping (and of the fast tokenizer itself) against the text of every token."""
    special = set(tokenizer.all_special_tokens)
    report = {"texts": [], "problems": 0}
    for text in OFFSET_TEXTS:
        enc = tokenizer(text, return_offsets_mapping=True)
        tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"])
        offsets = [tuple(o) for o in enc["offset_mapping"]]
        backend = tokenizer.backend_tokenizer.encode(text)
        same_backend = backend.ids == enc["input_ids"] and [tuple(o) for o in backend.offsets] == offsets
        problems, counts = offset_problems(text, tokens, offsets, special)
        spans = [o for o in offsets if o[1] > o[0]]
        if not spans or spans[0][0] != 0 or spans[-1][1] != len(text):
            problems.append({"why": f"the spans cover {spans[0][0] if spans else None}..{spans[-1][1] if spans else None}, "
                                    f"the text 0..{len(text)}"})
        report["texts"].append({"text": text, "tokens": len(tokens), "counts": counts, "same_as_backend": same_backend,
                                "problems": problems,
                                "sample": [[t, list(o), text[o[0]:o[1]]] for t, o in list(zip(tokens, offsets))[:12]]})
        report["problems"] += len(problems) + (0 if same_backend else 1)
    # Windows of a question and a context, as the SQuAD-NL fine-tuning makes them: in every window the offsets of
    # the context index into the context
    question, context = QA_PAIR
    enc = tokenizer(question, context, truncation="only_second", max_length=96, stride=32,
                    return_overflowing_tokens=True, return_offsets_mapping=True)
    windows = []
    for w in range(len(enc["input_ids"])):
        seq = enc.sequence_ids(w)
        tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][w])
        problems = []
        for s, part in ((0, question), (1, context)):
            idx = [k for k, x in enumerate(seq) if x == s]
            sub_offsets = [tuple(enc["offset_mapping"][w][k]) for k in idx]
            p, _ = offset_problems(part, [tokens[k] for k in idx], sub_offsets, special,
                                   first_is_segment_start=sub_offsets[0][0] == 0)
            problems += [dict(q, sequence=s) for q in p]
        windows.append({"window": w, "tokens": len(tokens), "context_span": [enc["offset_mapping"][w][k] for k in
                                                                              (seq.index(1), len(seq) - 1 - seq[::-1].index(1))],
                        "problems": problems})
        report["problems"] += len(problems)
    report["qa_windows"] = windows
    return report


# --- The two steps ---------------------------------------------------------------------------------------------

def training_tokenizer(name: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(name)
    if tokenizer.mask_token is None:
        tokenizer.add_special_tokens({"mask_token": "<mask>"})  # as train_minimal.py
    return tokenizer


def step_jax(args) -> None:
    import pickle

    import jax
    import jax.numpy as jnp

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from mmbjax.configuration_mmbbert import MMBertConfig
    from mmbjax.heads.masked_lm import ModernBertForMaskedLM

    if jax.default_backend() != "cpu":
        raise SystemExit(f"run with JAX_PLATFORMS=cpu (backend is {jax.default_backend()})")
    flax_dir = Path(args.flax)
    cfg = json.loads((flax_dir / "config.json").read_text())
    params = pickle.loads((flax_dir / "params.pkl").read_bytes())
    tokenizer = training_tokenizer(args.tokenizer or cfg["tokenizer_name"])
    names, rows = logit_rows(tokenizer)
    ids, mask = pad_rows(rows, tokenizer.pad_token_id)
    config = MMBertConfig(**{k: v for k, v in cfg.items() if k not in ("model_type", "tokenizer_name")})
    model = ModernBertForMaskedLM(config)
    logits = model.apply({"params": params}, input_ids=jnp.asarray(ids), attention_mask=jnp.asarray(mask),
                         deterministic=True).logits
    logits = np.asarray(jax.device_get(logits), dtype=np.float32)
    n_params = int(sum(np.asarray(x).size for x in jax.tree_util.tree_leaves(params)))
    meta = {"texts": names, "tokenizer": args.tokenizer or cfg["tokenizer_name"], "config": cfg,
            "mask_token_id": tokenizer.mask_token_id, "pad_token_id": tokenizer.pad_token_id,
            "parameters": n_params, "jax": jax.__version__}
    np.savez(args.out, input_ids=ids, attention_mask=mask, logits=logits, meta=json.dumps(meta))
    print(f"{args.out}: {len(rows)} rows, {int(mask.sum())} real positions, {n_params:,} parameters, "
          f"mask id {tokenizer.mask_token_id}")


def fill_mask(tokenizer, model, torch) -> list[dict]:
    out = []
    for sentence in FILL_MASK:
        ids_ref, word_id = ids_with_word_masked(tokenizer, sentence)
        variants = {"ids (as in training)": ids_ref}
        for name, text in mask_notations(sentence, tokenizer.mask_token).items():
            variants[name] = tokenizer(text)["input_ids"]
        entry = {"sentence": sentence, "word": tokenizer.convert_ids_to_tokens(word_id), "variants": []}
        for name, ids in variants.items():
            with torch.no_grad():
                logits = model(input_ids=torch.tensor([ids])).logits[0]
            pos = ids.index(tokenizer.mask_token_id)
            probs = torch.softmax(logits[pos].double(), -1)
            top = torch.topk(probs, 5)
            entry["variants"].append({
                "notation": name,
                "tokens": tokenizer.convert_ids_to_tokens(ids),
                "ids_as_in_training": ids == ids_ref,
                "top5": [[tokenizer.convert_ids_to_tokens(int(t)), round(float(p), 4)] for p, t in zip(top.values, top.indices)],
                "word_rank": int((probs > probs[word_id]).sum()) + 1,
            })
        out.append(entry)
    return out


def step_hf(args) -> int:
    import tokenizers
    import torch
    import transformers
    from transformers import AutoModelForMaskedLM, AutoModelForSequenceClassification, AutoTokenizer

    ref = np.load(args.reference)
    meta = json.loads(str(ref["meta"]))
    kw = {"revision": args.revision} if args.revision else {}
    tokenizer = AutoTokenizer.from_pretrained(args.model, **kw)
    model = AutoModelForMaskedLM.from_pretrained(args.model, **kw)
    model.eval()
    report = {"model": args.model, "revision": args.revision, "transformers": transformers.__version__,
              "torch": torch.__version__, "tokenizers": tokenizers.__version__,
              "model_class": type(model).__name__, "tokenizer_class": type(tokenizer).__name__,
              "is_fast": tokenizer.is_fast, "dtype": str(next(model.parameters()).dtype),
              "attn_implementation": getattr(model.config, "_attn_implementation", None), "checks": {}}
    if not Path(args.model).exists():
        from huggingface_hub import HfApi

        info = HfApi().model_info(args.model, revision=args.revision)
        report["resolved_revision"] = info.sha
        report["private"] = info.private
    checks = report["checks"]

    # Tokenizer and config
    names, rows = logit_rows(tokenizer)
    ids, mask = pad_rows(rows, tokenizer.pad_token_id)
    cfg = model.config
    checks["ids_equal_reference"] = bool(np.array_equal(ids, ref["input_ids"]) and np.array_equal(mask, ref["attention_mask"]))
    checks["special_ids"] = {
        "config": {k: getattr(cfg, f"{k}_token_id") for k in ("pad", "bos", "eos", "cls", "sep")},
        "tokenizer": {"pad": tokenizer.pad_token_id, "bos": tokenizer.bos_token_id, "eos": tokenizer.eos_token_id,
                      "cls": tokenizer.cls_token_id, "sep": tokenizer.sep_token_id, "mask": tokenizer.mask_token_id,
                      "unk": tokenizer.unk_token_id},
        "len_tokenizer": len(tokenizer), "vocab_size": cfg.vocab_size,
    }
    sp = checks["special_ids"]
    sp["ok"] = (sp["config"]["pad"] == tokenizer.pad_token_id and sp["config"]["bos"] == tokenizer.bos_token_id
                and sp["config"]["eos"] == tokenizer.eos_token_id and len(tokenizer) == cfg.vocab_size
                and tokenizer.mask_token_id == meta["mask_token_id"])
    report["parameters"] = {"unique": model.num_parameters(), "reference_jax": meta["parameters"],
                            "tied_decoder": bool(model.decoder.weight.data_ptr() == model.model.embeddings.tok_embeddings.weight.data_ptr())}

    # Logits against the JAX reference, batched with padding and each row alone
    with torch.no_grad():
        logits = model(input_ids=torch.tensor(ids, dtype=torch.long), attention_mask=torch.tensor(mask, dtype=torch.long)).logits.numpy()
        alone = np.zeros_like(logits)
        for i, r in enumerate(rows):
            alone[i, :len(r)] = model(input_ids=torch.tensor([r])).logits[0].numpy()
    lg = compare_logits(ref["logits"], logits, ref["attention_mask"])
    lg["rows"] = names
    lg["alone_vs_batched_max_abs_diff"] = compare_logits(logits, alone, mask)["max_abs_diff"]
    lg["atol"] = args.atol
    lg["ok"] = bool(checks["ids_equal_reference"] and lg["max_abs_diff"] <= args.atol and lg["top1_equal"] == lg["positions"])
    checks["logits"] = lg

    checks["fill_mask"] = fill_mask(tokenizer, model, torch)
    from transformers import pipeline

    fm = pipeline("fill-mask", model=model, tokenizer=tokenizer)
    checks["pipeline"] = {p: [[r["token_str"], round(float(r["score"]), 4)] for r in fm(p, top_k=5)] for p in PIPELINE_PROMPTS}
    checks["offsets"] = check_offsets(tokenizer)
    checks["offsets"]["ok"] = checks["offsets"]["problems"] == 0

    # The card's fine-tuning example: a classification head on the encoder, one step
    clf = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2, **kw)
    batch = tokenizer(["Een prachtig boek.", "Wat een saai verhaal, ik heb het niet uitgelezen."], padding=True, return_tensors="pt")
    out = clf(**batch, labels=torch.tensor([1, 0]))
    out.loss.backward()
    grads = [p.grad for p in clf.parameters() if p.requires_grad]
    checks["sequence_classification"] = {"class": type(clf).__name__, "loss": float(out.loss),
                                         "ok": bool(torch.isfinite(out.loss) and all(g is not None and torch.isfinite(g).all() for g in grads))}

    ok = {"ids_equal_reference": checks["ids_equal_reference"], "special_ids": sp["ok"], "logits": lg["ok"],
          "offsets": checks["offsets"]["ok"], "sequence_classification": checks["sequence_classification"]["ok"]}
    report["ok"] = ok
    Path(args.out).write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({"out": args.out, "ok": ok, "max_abs_diff": lg["max_abs_diff"], "top1": f"{lg['top1_equal']}/{lg['positions']}",
                      "offset_problems": checks["offsets"]["problems"], "transformers": transformers.__version__}, ensure_ascii=False))
    for p, top in checks["pipeline"].items():
        print(f"  pipeline {p!r}: {top}")
    for entry in checks["fill_mask"]:
        for v in entry["variants"]:
            print(f"  {entry['sentence']} [{v['notation']}] ids as in training: {v['ids_as_in_training']}, "
                  f"rank of the word {v['word_rank']}: {v['top5']}")
    return 0 if all(ok.values()) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="step", required=True)
    j = sub.add_parser("jax", help="logits of the JAX model into an .npz")
    j.add_argument("--flax", required=True, help="directory with params.pkl and config.json (convert_gcs_checkpoint.py)")
    j.add_argument("--tokenizer", help="tokenizer of the training run (default: config.json's tokenizer_name)")
    j.add_argument("--out", required=True)
    h = sub.add_parser("hf", help="check the export against the .npz")
    h.add_argument("--model", required=True, help="export directory or Hub repo id")
    h.add_argument("--revision")
    h.add_argument("--reference", required=True)
    h.add_argument("--atol", type=float, default=2e-3)
    h.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.step == "jax":
        step_jax(args)
        return 0
    return step_hf(args)


if __name__ == "__main__":
    sys.exit(main())
