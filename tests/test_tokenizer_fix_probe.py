"""Tests of experiments/final/release/tokenizer_fix_probe.py: the proposed tokenizer gives the pretraining ids for
"de <mask> van" and the same ids as the published tokenizer for plain text."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "final" / "release"))

from tokenizer_fix_probe import SENTENCES, make_fixed, training_ids  # noqa: E402

FILES = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja"]


@pytest.fixture
def tokenizers(tmp_path):
    transformers = pytest.importorskip("transformers")
    from huggingface_hub import snapshot_download

    try:
        export = Path(snapshot_download("yhavinga/dmbert-230m-culturax", allow_patterns=FILES, local_files_only=True))
    except Exception:
        pytest.skip("the tokenizer of yhavinga/dmbert-230m-culturax is not in the local Hugging Face cache")
    make_fixed(export, tmp_path / "fixed")
    return (transformers.AutoTokenizer.from_pretrained(export),
            transformers.AutoTokenizer.from_pretrained(tmp_path / "fixed"))


def test_fixed_tokenizer_gives_the_pretraining_ids_with_spaces_around_mask(tokenizers):
    legacy, fixed = tokenizers
    for sentence, _ in SENTENCES:
        ids, notations = training_ids(sentence, legacy)
        assert fixed(notations["spaces"])["input_ids"] == ids, sentence
        assert legacy(notations["spaces"])["input_ids"] != ids, sentence


def test_fixed_tokenizer_keeps_the_ids_of_plain_text_but_not_of_a_leading_space(tokenizers):
    legacy, fixed = tokenizers
    for text in ["De kat zit op de mat.", "Eén café, twee euro (€2,00).", "Twee\tregels:\nhier.", "Hij woont in 's-Hertogenbosch, toch?"]:
        assert fixed(text)["input_ids"] == legacy(text)["input_ids"], text
    pair = ("Wat is de hoofdstad?", "Amsterdam is de hoofdstad.")
    assert fixed(*pair)["input_ids"] == legacy(*pair)["input_ids"]
    assert fixed(" Begint met een spatie.")["input_ids"] != legacy(" Begint met een spatie.")["input_ids"]
