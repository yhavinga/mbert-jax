#!/usr/bin/env python3
"""The structure of a text and of its translation compared (hammond mbert/designs/en-nl-translator.md §3 step 2,
Yeb 9 Nov: markdown and other formatting stay in both directions and are right).

features(text) counts per category what the formatting of a text consists of; differences(source, target) names the
categories whose counts differ. A pair with a difference drops out of the training data: otherwise the translator
learns to leave formatting out or to make it up.

Categories (CATEGORIES), counted per line or per pair of marks:
- heading: a line that starts with 1-6 # and a space (markdown)
- bullet: a line that starts with -, *, + or • and a space; number: a line that starts with 1-3 digits and . or )
- bold: **...** or __...__; italic: *...* or _..._ (one mark, not inside a word: snake_case is no italic)
- code_block: a line that starts with ```; inline_code: `...`
- table_row: a line that starts and ends with |; table_rule: a row of only |, -, : and spaces (|---|:--|)
- paragraph_break: an empty line (or more) between two lines of text; line_break: a newline between two lines of text

TranslateGemma's known deviation (the pilot, 9 Nov): an empty line where the original has a single newline. Where the
only difference is that some of the source's line breaks became paragraph breaks, differences() reports it as
newline_doubled, apart from the categories, so that it can be counted (and treated) on its own.
"""

import collections
import re

CATEGORIES = ("heading", "bullet", "number", "bold", "italic", "code_block", "inline_code", "table_row", "table_rule",
              "paragraph_break", "line_break")
NEWLINE_DOUBLED = "newline_doubled"

HEADING = re.compile(r"^\s{0,3}#{1,6}\s+\S", re.M)
BULLET = re.compile(r"^\s*[-*+•]\s+\S", re.M)
NUMBER = re.compile(r"^\s*\d{1,3}[.)]\s+\S", re.M)
FENCE = re.compile(r"^\s*```", re.M)
INLINE_CODE = re.compile(r"(?<!`)`(?!`)[^`\n]+`(?!`)")
BOLD = re.compile(r"(?<![*\w])\*\*(?!\s)[^*\n]+?(?<!\s)\*\*(?![*\w])|(?<![_\w])__(?!\s)[^_\n]+?(?<!\s)__(?![_\w])")
ITALIC = re.compile(r"(?<![*\w])\*(?![\s*])[^*\n]+?(?<![\s*])\*(?![*\w])|(?<![_\w])_(?![\s_])[^_\n]+?(?<![\s_])_(?![_\w])")
TABLE_RULE = re.compile(r"^\s*\|?\s*:?-+:?\s*(?:\|\s*:?-+:?\s*)+\|?\s*$", re.M)
TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$", re.M)


def features(text):
    """Counter of CATEGORIES in text"""
    text = text.strip("\n")
    c = collections.Counter()
    c["heading"] = len(HEADING.findall(text))
    # a list item that is in bold or italic is still a list item; a line of a table is no list item
    c["bullet"] = len(BULLET.findall(text))
    c["number"] = len(NUMBER.findall(text))
    c["code_block"] = len(FENCE.findall(text))
    c["inline_code"] = len(INLINE_CODE.findall(text))
    without_bullets = BULLET.sub(lambda m: m.group(0)[-1], text)  # "* item" is no italic mark
    c["bold"] = len(BOLD.findall(without_bullets))
    c["italic"] = len(ITALIC.findall(BOLD.sub(" ", without_bullets)))
    rules = len(TABLE_RULE.findall(text))
    c["table_rule"] = rules
    c["table_row"] = len(TABLE_ROW.findall(text)) - sum(1 for line in text.split("\n")
                                                        if TABLE_RULE.match(line) and TABLE_ROW.match(line))
    lines = text.split("\n")
    k = 0
    while k < len(lines) - 1:
        if lines[k + 1].strip():
            c["line_break"] += 1
            k += 1
            continue
        j = k + 1
        while j < len(lines) and not lines[j].strip():
            j += 1
        if j < len(lines):
            c["paragraph_break"] += 1
        k = j
    return +c


def differences(source, target):
    """The categories whose counts differ between source and target, in CATEGORIES's order; [NEWLINE_DOUBLED] when the
    only difference is line breaks of the source that are paragraph breaks in the target"""
    a, b = features(source), features(target)
    diff = [k for k in CATEGORIES if a[k] != b[k]]
    if diff == ["paragraph_break", "line_break"] or diff == ["line_break", "paragraph_break"]:
        extra = b["paragraph_break"] - a["paragraph_break"]
        if extra > 0 and a["line_break"] - b["line_break"] == extra:
            return [NEWLINE_DOUBLED]
    return diff
