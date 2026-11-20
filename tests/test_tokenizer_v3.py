"""Tokenizer (a) of dmbert v3 (scripts/build_tokenizer_v3.py) against the tokenizer of dmbert v2's trainer: the same ids
for plain text in both modes, <mask> written with spaces gives the pretraining ids, its own <|pad|>, <s>/</s> as
CLS/SEP. Needs the base tokenizer in the local Hugging Face cache (python scripts/build_tokenizer_v3.py downloads it)."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import build_tokenizer_v3 as bt  # noqa: E402

TEXTS = [
    "De kat zit op de mat.", " Begint met een spatie.", "Twee\tregels:\nhier.", "  dubbel  spatie ", "\n\nNa twee regels",
    "Eén café, twee euro (€2,00). 😀 ñ ß Ω 東京", "Hij woont in 's-Hertogenbosch, toch?", "x" * 300, "",
    "https://www.example.nl/pad?a=1&b=2", "Getallen: 1.234.567,89 en 3e-4.", "Tekst met   harde spatie en ​",
]
LITERALS = "Een tekst met <s>, </s>, <unk>, <pad>, <mask>, <|pad|> en <unused5> als gewone tekens."
SENTENCES = [("Amsterdam is de [hoofdstad] van Nederland.", "hoofdstad"), ("Ik drink elke ochtend een kopje [koffie].", "koffie"),
             ("De [kat] zit op de mat.", "kat"), ("Utrecht ligt in het [midden] van het land.", "midden"),
             ("Het [weer] is vandaag erg mooi.", "weer"), ("[Amsterdam] is de hoofdstad.", "Amsterdam")]


@pytest.fixture(scope="module")
def tokenizers(tmp_path_factory):
    transformers = pytest.importorskip("transformers")
    from huggingface_hub import snapshot_download

    try:
        base = Path(snapshot_download(bt.BASE_REPO, revision=bt.BASE_REVISION, local_files_only=True,
                                      allow_patterns=["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]))
    except Exception:
        pytest.skip(f"{bt.BASE_REPO}@{bt.BASE_REVISION[:7]} is not in the local Hugging Face cache")
    out = tmp_path_factory.mktemp("v3a")
    bt.build(base, out)
    auto = transformers.AutoTokenizer
    # The trainer's tokenizer of dmbert v2: the base, special tokens split as text, <mask> added
    old_train = auto.from_pretrained(base, split_special_tokens=True)
    old_train.add_special_tokens({"mask_token": "<mask>"})
    old = auto.from_pretrained(base)
    old.add_special_tokens({"mask_token": "<mask>"})
    return {"old_train": old_train, "old": old, "new_train": auto.from_pretrained(out, split_special_tokens=True),
            "new": auto.from_pretrained(out), "dir": out}


def test_special_tokens_and_size(tokenizers):
    new = tokenizers["new"]
    assert type(new).__name__ == "PreTrainedTokenizerFast" and len(new) == 32_128 and new.vocab_size == 32_000
    assert (new.mask_token_id, new.pad_token_id, new.cls_token_id, new.sep_token_id, new.bos_token_id,
            new.eos_token_id, new.unk_token_id) == (32_000, 32_001, 1, 2, 1, 2, 0)
    assert new.convert_tokens_to_ids(["<unused1>", "<unused126>", "<pad>"]) == [32_002, 32_127, 20_277]
    assert new.model_input_names == ["input_ids", "attention_mask"] and new.truncation_side == "left"


@pytest.mark.parametrize("text", TEXTS + [LITERALS])
def test_plain_text_has_the_ids_of_v2_in_the_trainers_mode(tokenizers, text):
    kwargs = dict(add_special_tokens=False)
    assert tokenizers["new_train"](text, **kwargs)["input_ids"] == tokenizers["old_train"](text, **kwargs)["input_ids"]


@pytest.mark.parametrize("text", TEXTS)
def test_plain_text_has_the_ids_and_offsets_of_v2_in_the_default_mode(tokenizers, text):
    new = tokenizers["new"](text, add_special_tokens=False, return_offsets_mapping=True)
    old = tokenizers["old"](text, add_special_tokens=False, return_offsets_mapping=True)
    assert new["input_ids"] == old["input_ids"] and new["offset_mapping"] == old["offset_mapping"]
    assert tokenizers["new"].decode(new["input_ids"]) == tokenizers["old"].decode(old["input_ids"])


def _pretraining_ids(sentence, tok):
    before, rest = sentence.split("[")
    word, after = rest.split("]")
    ids = tok(before + word + after, add_special_tokens=False)["input_ids"]
    tokens = tok.convert_ids_to_tokens(ids)
    k = tokens.index("▁" + word)
    return ids[:k] + [tok.mask_token_id] + ids[k + 1:], before + "<mask>" + after


@pytest.mark.parametrize("sentence, word", SENTENCES)
def test_mask_written_with_spaces_gives_the_pretraining_ids(tokenizers, sentence, word):
    expected, notation = _pretraining_ids(sentence, tokenizers["old_train"])
    assert tokenizers["new"](notation, add_special_tokens=False)["input_ids"] == expected
    assert tokenizers["new"](notation)["input_ids"] == [1] + expected + [2]
    # v2's tokenizer did not: an extra ▁ before the mask or a ▁ in front of what follows it
    assert tokenizers["old"](notation, add_special_tokens=False)["input_ids"] != expected


def test_the_glued_notation_of_v2_no_longer_finds_the_mask(tokenizers):
    ids = tokenizers["new"]("Amsterdam is de<mask>van Nederland.")["input_ids"]
    assert tokenizers["new"].mask_token_id not in ids


def test_templates_and_padding(tokenizers):
    new = tokenizers["new"]
    single = new("Een zin.")["input_ids"]
    assert single[0] == 1 and single[-1] == 2 and 2 not in single[1:-1]
    a, b = new("Wat is de hoofdstad?", add_special_tokens=False)["input_ids"], new("Amsterdam.", add_special_tokens=False)["input_ids"]
    pair = new("Wat is de hoofdstad?", "Amsterdam.")
    assert pair["input_ids"] == [1] + a + [2] + b + [2] and "token_type_ids" not in pair
    batch = new(["kort", "een langere zin hier"], padding=True)
    assert batch["input_ids"][0][-1] == 32_001 and batch["attention_mask"][0][-1] == 0


def test_packed_pretraining_pieces_are_those_of_v2_up_to_the_padding_id(tokenizers):
    from mmbjax.training.packing import TokenizeTransform

    document = " ".join(TEXTS) + " " + LITERALS
    old = TokenizeTransform(tokenizers["old_train"], max_length=64).map({"text": document})
    new = TokenizeTransform(tokenizers["new_train"], max_length=64).map({"text": document})
    assert len(old) == len(new) > 2
    assert all(np.array_equal(o["input_ids"], n["input_ids"]) for o, n in zip(old, new))
    assert set(tokenizers["new_train"].all_special_ids) == {0, 1, 2, 32_000, 32_001}


def test_build_refuses_another_base(tmp_path, tokenizers):
    import json

    base = tmp_path / "base"
    base.mkdir()
    tj = json.loads((tokenizers["dir"] / "tokenizer.json").read_text())
    (base / "tokenizer.json").write_text(json.dumps(tj))
    (base / "tokenizer_config.json").write_text("{}")
    with pytest.raises(ValueError, match="added tokens"):
        bt.build(base, tmp_path / "out")
