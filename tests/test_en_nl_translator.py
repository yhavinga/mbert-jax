"""experiments/flan-nl/translate.py with --source-lang/--target-lang (en-nl-translator): English to Dutch stays as it
was, Dutch to English gets its own prompt and column. No model and no TPU: run with --noconftest where the
conftest's imports (jax, flax) are not installed. The test of the real chat template runs where TranslateGemma's
tokenizer is (TRANSLATEGEMMA_TOKENIZER, default /dev/shm/flan_nl/tokenizer-27b on the workers of mbert-corpus)."""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "flan-nl"))

import translate as tr  # noqa: E402

NAMES = {"en": "English", "nl": "Dutch", "de": "German"}


class TemplateTokenizer:
    """TranslateGemma's chat template for a text turn, in the words of chat_template.jinja; a token is a character."""

    def __init__(self, fixed=None):
        self.fixed = fixed  # (source, target): a template that ignores the codes it is given

    def render(self, messages):
        content = messages[0]["content"][0]
        s, t = self.fixed or (content["source_lang_code"], content["target_lang_code"])
        source, target = NAMES[s], NAMES[t]
        return (f"<bos><start_of_turn>user\nYou are a professional {source} ({s}) to {target} ({t}) translator. "
                f"Your goal is to accurately convey the meaning and nuances of the original {source} text while "
                f"adhering to {target} grammar, vocabulary, and cultural sensitivities.\nProduce only the {target} "
                f"translation, without any additional explanations or commentary. Please translate the following "
                f"{source} text into {target}:\n\n\n{content['text'].strip()}<end_of_turn>\n<start_of_turn>model\n")

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert add_generation_prompt
        text = self.render(messages)
        return [ord(c) for c in text] if tokenize else text

    def __call__(self, text, add_special_tokens):
        assert not add_special_tokens
        return {"input_ids": [ord(c) for c in text]}

    @staticmethod
    def decode(ids, skip_special_tokens):
        return " ".join(str(i) for i in ids if i not in tr.EOS_IDS)


def test_messages_default_to_english_into_dutch_as_before():
    assert tr.messages("Hello.") == [{"role": "user", "content": [
        {"type": "text", "source_lang_code": "en", "target_lang_code": "nl", "text": "Hello."}]}]
    assert tr.messages("Hallo.", "nl", "en")[0]["content"][0]["source_lang_code"] == "nl"
    assert tr.messages("Hallo.", "nl", "en")[0]["content"][0]["target_lang_code"] == "en"


@pytest.mark.parametrize("source, target", [("en", "nl"), ("nl", "en")])
def test_prompt_codec_follows_the_template_in_both_directions(source, target):
    tok = TemplateTokenizer()
    codec = tr.PromptCodec(tok) if (source, target) == ("en", "nl") else tr.PromptCodec(tok, source, target)
    assert f"{NAMES[source]} ({source}) to {NAMES[target]} ({target}) translator" in codec.prefix
    assert codec.prefix.endswith(f"{NAMES[source]} text into {NAMES[target]}:\n\n\n")
    texts = ["De Kamer gaat akkoord.", "  The House agrees.  ", "Één regel\nen nog een."]
    codec.check([t.strip() for t in texts])
    assert codec.encode("Ja.") == tok.apply_chat_template(tr.messages("Ja.", source, target), tokenize=True,
                                                          add_generation_prompt=True)


def test_prompt_codec_refuses_a_template_of_another_language_pair():
    with pytest.raises(RuntimeError, match="unexpected chat template"):
        tr.PromptCodec(TemplateTokenizer(fixed=("en", "nl")), "nl", "en")
    with pytest.raises(RuntimeError, match="unexpected chat template"):
        tr.PromptCodec(TemplateTokenizer(fixed=("en", "de")))


class EchoDevice:
    """decode_all's device: every prompt continues with 3 tokens, then <end_of_turn>."""

    def __init__(self, slots):
        self.programs, self.positions = [None] * slots, [0] * slots

    def prefill(self, tokens):
        return list(tokens), np.int32(11)

    def insert(self, prefix, slot):
        self.programs[slot], self.positions[slot] = [11, 12, 13, tr.EOS_IDS[1]], 1

    def generate(self):
        out = []
        for s, program in enumerate(self.programs):
            p = self.positions[s]
            out.append(program[p] if program is not None and p < len(program) else 7)
            self.positions[s] += 1
        return np.asarray(out, np.int32)

    @staticmethod
    def fetch(arrays):
        return np.asarray(arrays)


@pytest.mark.parametrize("source, target, column", [("en", "nl", "text_nl"), ("nl", "en", "text_en")])
def test_a_shard_has_the_translation_in_the_column_of_its_target_language(source, target, column):
    tok = TemplateTokenizer()
    codec = tr.PromptCodec(tok, source, target)
    shapes = {"slots": 2, "max_decode": 16}
    rows, seconds = tr.translate_shard(EchoDevice(2), codec, tok, [5, 6, 7], ["a", "b", "c"], shapes, 2)
    assert list(rows) == ["id", column, "ended", "tokens_in", "tokens_out"]
    assert rows["id"] == [5, 6, 7] and rows[column] == ["11 12 13"] * 3 and rows["ended"] == [True] * 3
    assert rows["tokens_out"] == [4] * 3 and seconds >= 0


def test_main_refuses_one_language_on_both_sides():
    with pytest.raises(SystemExit):
        tr.main(["--segments", "s.parquet", "--out", "gs://x/y", "--ckpt", "c", "--tokenizer", "t",
                 "--length-class", "small", "--source-lang", "nl", "--target-lang", "nl"])


TOKENIZER = Path(os.environ.get("TRANSLATEGEMMA_TOKENIZER", "/dev/shm/flan_nl/tokenizer-27b"))


@pytest.mark.skipif(not (TOKENIZER / "tokenizer.json").exists(), reason="TranslateGemma's tokenizer is not here")
def test_the_real_template_in_both_directions():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(TOKENIZER))
    en_nl = tr.PromptCodec(tok)
    assert "English (en) to Dutch (nl)" in en_nl.prefix
    assert en_nl.prefix.endswith("Please translate the following English text into Dutch:\n\n\n")
    en_nl.check(["The House agrees with the motion.", "Two lines\nof text."])
    nl_en = tr.PromptCodec(tok, "nl", "en")
    assert "Dutch (nl) to English (en)" in nl_en.prefix
    assert nl_en.prefix.endswith("Please translate the following Dutch text into English:\n\n\n")
    nl_en.check(["Dat is het paard achter de wagen spannen.", "Voorzitter. Ik dank de minister voor zijn antwoord."])
    assert en_nl.suffix == nl_en.suffix
