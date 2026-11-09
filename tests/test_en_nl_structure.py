"""experiments/en-nl-translator/structure.py: the formatting of a text and of its translation compared. Run with
--noconftest where the conftest's imports (jax, flax) are not installed."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "en-nl-translator"))

import structure  # noqa: E402

MARKDOWN = """# Title

Some **bold** and *italic* and _also italic_ text with `code`.

- one
- two
1. first
2) second

| a | b |
|---|:--|
| 1 | 2 |

```
x = 1
```"""


def test_features_of_markdown():
    f = structure.features(MARKDOWN)
    assert f["heading"] == 1 and f["bold"] == 1 and f["italic"] == 2 and f["inline_code"] == 1
    assert f["bullet"] == 2 and f["number"] == 2
    assert f["table_row"] == 2 and f["table_rule"] == 1 and f["code_block"] == 2
    assert f["paragraph_break"] == 4 and f["line_break"] == 7


@pytest.mark.parametrize("text, absent", [
    ("snake_case_name and 2*3*4 are no italic", "italic"),
    ("* a bullet with *italic* inside", "bold"),
    ("A line - with a dash is no bullet", "bullet"),
    ("In 2023. Then more.", "number"),
])
def test_no_false_marks(text, absent):
    assert structure.features(text)[absent] == 0


def test_bullets_are_not_italic():
    f = structure.features("* one\n* two")
    assert f["bullet"] == 2 and f["italic"] == 0


def test_breaks():
    f = structure.features("a\nb\n\nc\n\n\n\nd\n")
    assert f["line_break"] == 1 and f["paragraph_break"] == 2


@pytest.mark.parametrize("source, target, diff", [
    ("Een zin.\nNog een.", "A sentence.\nAnother.", []),
    ("Een **vette** zin.", "A bold sentence.", ["bold"]),
    ("# Kop\n\nTekst.", "Heading\n\nText.", ["heading"]),
    ("- een\n- twee", "- one\n- two\n- three", ["bullet", "line_break"]),
    ("Een.\nTwee.\nDrie.", "One.\n\nTwo.\n\nThree.", [structure.NEWLINE_DOUBLED]),
    ("Een.\nTwee.", "One.\n\nTwo.\n\nThree.", ["paragraph_break", "line_break"]),
    ("Een.\n\nTwee.", "One.\nTwo.", ["paragraph_break", "line_break"]),
    ("Een *schuin* woord.\nRegel.", "An italic word.\n\nLine.", ["italic", "paragraph_break", "line_break"]),
    ("| a | b |\n|---|---|\n| 1 | 2 |", "| a | b |\n| 1 | 2 |", ["table_rule", "line_break"]),
])
def test_differences(source, target, diff):
    assert structure.differences(source, target) == diff
