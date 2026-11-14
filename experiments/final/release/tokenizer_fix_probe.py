"""Proposal, not applied: a tokenizer that gives the pretraining ids for "de <mask> van".

Change (tokenizer.json, and legacy=false plus <mask> lstrip in tokenizer_config.json): the normalizer (Prepend ▁ + Replace space) becomes the pre-tokenizer Metaspace(prepend_scheme="first",
split=False), so that the text after a special token gets no extra ▁; and <mask> gets lstrip=True, so that the space
before it belongs to the mask. Measured here:
  ids   (any env with pyarrow): legacy and fixed ids over the documents of a parquet file, tokenized as in the
        training (add_special_tokens=False, split_special_tokens=True), and the fill-mask notations;
  model (env with torch): fill-mask top-5 of the export with the fixed tokenizer.
  python tokenizer_fix_probe.py ids|model EXPORT_DIR FIXED_DIR [PARQUET] OUT.json
"""

import copy
import json
import shutil
import sys
from pathlib import Path

SENTENCES = [
    ("Amsterdam is de [hoofdstad] van Nederland.", "hoofdstad"),
    ("Ik drink elke ochtend een kopje [koffie].", "koffie"),
    ("De [kat] zit op de mat.", "kat"),
    ("Utrecht ligt in het [midden] van het land.", "midden"),
    ("Het [weer] is vandaag erg mooi.", "weer"),
]


def make_fixed(export: Path, fixed: Path) -> None:
    if fixed.exists():
        shutil.rmtree(fixed)
    fixed.mkdir(parents=True)
    for f in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja"]:
        shutil.copy(export / f, fixed / f)
    tj = json.loads((fixed / "tokenizer.json").read_text())
    tj = copy.deepcopy(tj)
    tj["normalizer"] = None
    tj["pre_tokenizer"] = {"type": "Metaspace", "replacement": "▁", "prepend_scheme": "first", "split": False}
    for a in tj["added_tokens"]:
        if a["content"] == "<mask>":
            a["lstrip"] = True
    (fixed / "tokenizer.json").write_text(json.dumps(tj, ensure_ascii=False, indent=2))
    tc = json.loads((fixed / "tokenizer_config.json").read_text())
    tc["added_tokens_decoder"]["32000"]["lstrip"] = True
    # transformers 5 rebuilds the pre-tokenizer of a Llama tokenizer from this flag (prepend "always" when legacy)
    tc["legacy"] = False
    (fixed / "tokenizer_config.json").write_text(json.dumps(tc, ensure_ascii=False, indent=2))
    sm = json.loads((fixed / "special_tokens_map.json").read_text())
    sm["mask_token"]["lstrip"] = True
    (fixed / "special_tokens_map.json").write_text(json.dumps(sm, ensure_ascii=False, indent=2))


def training_ids(sentence, tok):
    import re
    m = re.fullmatch(r"(.*)\[([^\]]+)\](.*)", sentence)
    before, word, after = m.groups()
    ids = tok(before + word + after)["input_ids"]
    toks = tok.convert_ids_to_tokens(ids)
    k = toks.index("▁" + word)
    ids = list(ids)
    ids[k] = tok.mask_token_id
    notations = {"spaces": before + "<mask>" + after, "no space before": before.rstrip() + "<mask>" + after,
                 "no space before or after": before.rstrip() + "<mask>" + after.lstrip()}
    return ids, notations


def main():
    mode, export, fixed = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    import transformers
    from transformers import AutoTokenizer

    out = Path(sys.argv[-1])
    if mode == "ids":
        import pyarrow.parquet as pq

        make_fixed(export, fixed)
        legacy = AutoTokenizer.from_pretrained(export)
        new = AutoTokenizer.from_pretrained(fixed)
        res = {"transformers": transformers.__version__, "parquet": sys.argv[4]}
        n = diff = diff_ws = ws = 0
        examples = []
        for batch in pq.ParquetFile(sys.argv[4]).iter_batches(batch_size=1000, columns=["text"]):
            texts = batch.column("text").to_pylist()
            a = legacy(texts, add_special_tokens=False, split_special_tokens=True)["input_ids"]
            b = new(texts, add_special_tokens=False, split_special_tokens=True)["input_ids"]
            for t, x, y in zip(texts, a, b):
                n += 1
                starts_ws = t[:1].isspace()
                ws += starts_ws
                if x != y:
                    diff += 1
                    diff_ws += starts_ws
                    if len(examples) < 5:
                        k = next(i for i, (p, q) in enumerate(zip(x, y)) if p != q) if any(p != q for p, q in zip(x, y)) else min(len(x), len(y))
                        examples.append({"text": t[:60], "legacy": legacy.convert_ids_to_tokens(x[max(0, k - 2):k + 3]),
                                         "fixed": new.convert_ids_to_tokens(y[max(0, k - 2):k + 3]), "at": k})
        res["documents"] = {"n": n, "different_ids": diff, "different_and_starting_with_whitespace": diff_ws,
                            "starting_with_whitespace": ws, "examples": examples}
        # With the default specials, single texts and pairs
        probes = ["De kat zit op de mat.", " Begint met een spatie.", "Twee\tregels:\nhier."]
        res["single"] = {p: legacy(p)["input_ids"] == new(p)["input_ids"] for p in probes}
        res["pair"] = legacy("Wat is de hoofdstad?", "Amsterdam is de hoofdstad.")["input_ids"] == new("Wat is de hoofdstad?", "Amsterdam is de hoofdstad.")["input_ids"]
        res["fill_mask_notations"] = []
        for s, _ in SENTENCES:
            ids, notations = training_ids(s, legacy)
            res["fill_mask_notations"].append({"sentence": s, **{
                k: {"legacy": legacy(v)["input_ids"] == ids, "fixed": new(v)["input_ids"] == ids} for k, v in notations.items()}})
        e = new("Amsterdam is de <mask> van Nederland.", return_offsets_mapping=True)
        res["fixed_offsets_mask_sentence"] = [[t, list(o)] for t, o in zip(new.convert_ids_to_tokens(e["input_ids"]), e["offset_mapping"])]
    else:
        import torch
        from transformers import AutoModelForMaskedLM, pipeline

        new = AutoTokenizer.from_pretrained(fixed)
        model = AutoModelForMaskedLM.from_pretrained(export)
        model.eval()
        fm = pipeline("fill-mask", model=model, tokenizer=new)
        res = {"transformers": transformers.__version__, "torch": torch.__version__, "pipeline": {}}
        for s, word in SENTENCES:
            _, notations = training_ids(s, new)
            for k, v in notations.items():
                top = fm(v, top_k=5)
                res["pipeline"][v] = [[r["token_str"], round(float(r["score"]), 4)] for r in top]
        # The ids of the fixed tokenizer for plain text in this transformers version equal the legacy ones
        legacy = AutoTokenizer.from_pretrained(export)
        res["plain_ids_equal"] = all(legacy(t)["input_ids"] == new(t)["input_ids"] for t in
                                     ["De kat zit op de mat.", "Eén café, twee euro (€2,00).", "Hij woont in 's-Hertogenbosch, toch?"])
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(res, ensure_ascii=False)[:3000])


if __name__ == "__main__":
    main()
