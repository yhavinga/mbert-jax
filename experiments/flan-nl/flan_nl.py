#!/usr/bin/env python3
"""The Dutch FLAN of arm F-nl: which FLAN rows of arm F's mixes are translated, in which pieces, and how they come back.

Steps (commands.txt), each a subcommand here except the translation itself (translate.py) and the mix (build_nl_mix.py):
    extract   on every worker: the FLAN rows of this host's mix of F -> <root>/en/w<rank>.parquet
    plan      on one worker: task filters, the pieces of every example, the distinct segments -> <root>/plan/
    assemble  on one worker: the translations back into examples, the checks -> <root>/nl/w<rank>.parquet, report

A FLAN row of F is (instruction, answer); its id is (host, position), position being its place in the host's sequence
(the mix's `position` column). An example becomes a list of pieces per side: literals, kept as they are (labels such as
"(B).", separators, whitespace, text without words), and segments, translated once per distinct string. Multiple-choice
options are segments of their own, and an answer that is an option's text gets that option's translation, so the
example stays consistent; an answer that is an option's label stays the label.

Python 3.10 (the trainer's venv imports this for the mix) and the standard library only, so the tests run anywhere.
"""

import json
import re
import unicodedata
from collections import Counter

# Most source tokens (TranslateGemma's tokenizer) of one segment: its training data were "individual sentences and
# text blobs of up to 512 tokens" (TranslateGemma technical report, arXiv 2601.09012v2); longer text is split at
# paragraph, then line, then sentence boundaries.
MAX_SEGMENT_TOKENS = 512

# Tasks whose examples do not survive translation, by reason. Names are FLAN's _task_name (NIv2 with its number).
DROPPED_TASKS = {
    # Code, SQL, logical forms and other formal languages: the text is not English to translate
    "code": {
        "task076_splash_correcting_sql_mistake", "task077_splash_explanation_to_sql", "task107_splash_question_to_sql",
        "task868_cfq_mcd1_explanation_to_sql", "task869_cfq_mcd1_sql_to_explanation",
        "task921_code_x_glue_information_retreival", "task956_leetcode_420_strong_password_check",
        "task126_scan_structured_text_generation_command_action_all", "task127_scan_long_text_generation_action_command_all",
        "task128_scan_structured_text_generation_command_action_short",
        "task129_scan_long_text_generation_action_command_short",
        "task130_scan_structured_text_generation_command_action_long",
        "task131_scan_long_text_generation_action_command_long",
        "task110_logic2text_sentence_generation", "task210_logic2text_structured_text_generation",
        "task211_logic2text_classification", "task212_logic2text_classification",
    },
    # The label depends on the English form: grammar, acceptability, spelling, punctuation, case, fluency, word order
    "english_form": {
        "glue/cola:2.0.0", "task616_cola_classification", "task1346_glue_cola_grammatical_correctness_classification",
        "task1559_blimp_binary_classification", "task1560_blimp_binary_classification",
        "task021_mctaco_grammatical_logical", "task088_identify_typo_verification", "task089_swap_words_verification",
        "task138_detoxifying-lms_classification_fluency", "task132_dais_text_modification",
        "task428_senteval_inversion", "task515_senteval_odd_word_out", "task516_senteval_conjoints_inversion",
        "task673_google_wellformed_query_classification", "task674_google_wellformed_query_sentence_generation",
        "task675_google_wellformed_query_sentence_generation",
        "task1415_youtube_caption_corrections_grammar_correction",
        "task1416_youtube_caption_corrections_incorrect_grammar_classification",
        "task1557_jfleg_answer_generation", "task1558_jfleg_incorrect_answer_generation",
        "task1195_disflqa_disfluent_to_fluent_conversion", "task1622_disfl_qa_text_modication",
        "task1623_disfl_qa_disfluent_question_classification",
        "task1703_ljspeech_textmodification", "task1704_ljspeech_textmodification", "task1705_ljspeech_classification",
        "task1706_ljspeech_classification",
        "fix_punct", "true_case", "word_segment",
    },
    # Word games: rhymes, repetition, letters and strings, lists of random letter strings
    "word_game": {
        "task183_rhyme_generation", "task062_bigbench_repeat_copy_logic",
        "task113_count_frequency_of_letter", "task114_is_the_given_word_longest", "task157_count_vowels_and_consonants",
        "task158_count_frequency_of_words", "task159_check_frequency_of_words_in_sentence_pair",
        "task160_replace_letter_in_a_sentence", "task161_count_words_containing_letter",
        "task162_count_words_starting_with_letter", "task163_count_words_ending_with_letter",
        "task365_synthetic_remove_vowels", "task376_reverse_order_of_words", "task377_remove_words_of_given_length",
        "task378_reverse_words_of_given_length", "task850_synthetic_longest_palindrome", "task1188_count_max_freq_char",
        "task1189_check_char_in_string", "task1316_remove_duplicates_string", "task1148_maximum_ascii_value",
        "task1443_string_to_number", "task079_conala_concat_strings", "task600_find_the_longest_common_substring_in_two_strings",
        "task622_replace_alphabets_in_a_list_by_their_position_in_english_alphabet",
        "task063_first_i_elements", "task064_all_elements_except_first_i", "task078_all_elements_except_last_i",
        "task091_all_elements_from_index_i_to_j", "task099_reverse_elements_between_index_i_and_j",
        "task100_concatenate_all_elements_from_index_i_to_j",
        "task101_reverse_and_concatenate_all_elements_from_index_i_to_j",
        "task267_concatenate_and_reverse_all_elements_from_index_i_to_j",
        "task488_extract_all_alphabetical_elements_from_list_in_order", "task504_count_all_alphabetical_elements_in_list",
        "task506_position_of_all_alphabetical_elements_in_list",
        "task509_collate_of_all_alphabetical_and_numerical_elements_in_list_separately",
        "task523_find_if_numbers_or_alphabets_are_more_in_list", "task605_find_the_longest_common_subsequence_in_two_lists",
        "task636_extract_and_sort_unique_alphabets_in_a_list",
        "task755_find_longest_substring_and_replace_its_sorted_lowercase_version_in_both_lists",
        "task756_find_longert_substring_and_return_all_unique_alphabets_in_it",
        "task1542_every_ith_element_from_starting", "task1551_every_ith_element_from_kth_element",
    },
    # Labels per token, or the next or missing English word
    "token_level": {
        "task583_udeps_eng_coarse_pos_tagging", "task584_udeps_eng_fine_pos_tagging",
        "task1167_penn_treebank_coarse_pos_tagging", "task1168_brown_coarse_pos_tagging", "task155_count_nouns_verbs",
        "task585_preposition_classification", "task388_torque_token_classification", "task610_conllpp_ner",
        "task963_librispeech_asr_next_word_prediction", "task964_librispeech_asr_text_auto_completion",
        "task965_librispeech_asr_missing_word_prediction", "task603_wikitext-103_fill_in_the_blank", "lambada:1.0.0",
    },
    # Translation, tokenization, or text in other languages
    "translation": {
        "task547_alt_translation_entk_en", "task560_alt_translation_en_entk", "task874_opus_xhosanavy_sr",
        "task681_hope_edi_malayalam_text_classification",
    },
    # About English words: their senses, the language itself, overlapping words
    "english_words": {
        "super_glue/wic:1.0.2", "task625_xlwic_true_or_false_answer_generation",
        "task626_xlwic_sentence_based_on_given_word_sentence_generation",
        "task627_xlwic_word_with_same_meaning_sentence_generation",
        "task628_xlwic_word_with_different_meaning_sentence_generation",
        "task225_english_language_answer_generation", "task226_english_language_answer_relevance_classification",
        "task039_qasc_find_overlapping_words",
    },
}
# The six Dutch tasks of F's pool (experiments/flan-decay/niv2_languages.json): their text is Dutch already and only the
# task definition is English; they stay as in F
DUTCH_TASKS = {
    "task1133_xcsr_nl_commonsense_mc_classification", "task409_mickey_nl_sentence_perturbation_generation",
    "task1543_conll2002_parts_of_speech_tagging_answer_generation",
    "task1544_conll2002_named_entity_recognition_answer_generation",
    "task1545_conll2002_person_name_extraction_answer_generation",
    "task1546_conll2002_location_name_extraction_answer_generation",
}
_REASON = {task: reason for reason, tasks in DROPPED_TASKS.items() for task in tasks}
assert len(_REASON) == sum(map(len, DROPPED_TASKS.values())), "a task is in two drop lists"


def task_of(source):
    """'flan/<folder>/<task>' -> task"""
    parts = source.split("/", 2)
    if len(parts) != 3 or parts[0] != "flan":
        raise ValueError(f"not a FLAN source: {source!r}")
    return parts[2]


def task_drop_reason(task):
    return _REASON.get(task)


# ---------------------------------------------------------------------------------------------------------- options

_ROMAN = ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"]
_FAMILIES = {
    "upper": [chr(c) for c in range(ord("A"), ord("Z") + 1)],
    "lower": [chr(c) for c in range(ord("a"), ord("z") + 1)],
    "digit": [str(n) for n in range(1, 21)],
    "roman": _ROMAN,
    "ROMAN": [r.upper() for r in _ROMAN],
}
# How a label is written around its letter, number or numeral, as FLAN's templates write them
_STYLES = ["({}).", "({})", "[{}].", "[{}]", "{}).", "{})", "{}.", "Option{}:", "Option {}:"]
# The text right after these introduces a list of bullets rather than labels (FLAN's "OPTIONS:\n- yes\n- no")
_LIST_HEADER = re.compile(r"(?:OPTIONS|Options|Options are|Choices|Possible answers|Pick from|Choose your answer from|"
                          r"Available options|Available choices|Pick your answer from|Choose from|Select from|"
                          r"Answer options|Possible choices|[Pp]ick from the following\.|"
                          r"Select from the following\.)[:]?[ \t]*\n?[ \t]*")
_BULLETS = ["[+]", "[-]", "--", "-", "+", "*", "•"]
_MAX_OPTION_CHARS = 400


def _label_pattern(style, token):
    return re.escape(style.format(token))


def _labelled_run(text, family, style):
    """[(label start, label end), ...] of the longest run of this family's labels in order, or []."""
    tokens = _FAMILIES[family]
    best = []
    # Most instructions have no list of this kind: a plain substring test spares the regular expressions
    if style.format(tokens[0]) not in text or style.format(tokens[1]) not in text:
        return best
    first = re.compile(r"(?:(?<=\s)|(?<=^)|(?<=;))" + _label_pattern(style, tokens[0]) + r"(?=\s)")
    for match in first.finditer(text):
        run = [(match.start(), match.end())]
        for token in tokens[1:]:
            pattern = re.compile(r"(?<=\s)" + _label_pattern(style, token) + r"(?=\s)")
            following = pattern.search(text, run[-1][1])
            if following is None:
                break
            between = text[run[-1][1]:following.start()]
            # Labels of one list are close together, and a paragraph break ends the list
            if len(between) > _MAX_OPTION_CHARS or "\n\n" in between.strip("\n"):
                break
            run.append((following.start(), following.end()))
        if len(run) >= 2 and len(run) >= len(best):
            best = run
    return best


def _bullet_run(text):
    """[(bullet start, bullet end), ...] of a bullet list right after a list header, or []."""
    best = []
    for header in _LIST_HEADER.finditer(text):
        start = header.end()
        for bullet in _BULLETS:
            if not text.startswith(bullet, start):
                continue
            newline_list = "\n" in header.group(0)
            spans = [(start, start + len(bullet))]
            pattern = re.compile(r"(?<=\s)" + re.escape(bullet) + r"(?=\S| )")
            position = spans[-1][1]
            while True:
                end_of_line = text.find("\n", position)
                following = pattern.search(text, position)
                if following is None:
                    break
                if newline_list:
                    # One bullet per line: the next must start the next line, after its indentation
                    if end_of_line == -1 or text[end_of_line + 1:following.start()].strip(" \t") != "":
                        break
                elif end_of_line != -1 and following.start() > end_of_line:
                    break
                if following.start() - position > _MAX_OPTION_CHARS:
                    break
                spans.append((following.start(), following.end()))
                position = following.end()
            if len(spans) >= 2 and len(spans) > len(best):
                best = spans
    return best


def find_options(text):
    """The multiple-choice options of an instruction: [(label, label start, option start, option end)] or [].

    The option's text runs from its label to the next label; the last one to the end of its line. Trailing separators
    (";", ",", "." and whitespace) are not part of it, so "(A). negative." has the option "negative".
    """
    candidates = []
    for family in _FAMILIES:
        for style in _STYLES:
            run = _labelled_run(text, family, style)
            if run:
                candidates.append(run)
    bullets = _bullet_run(text)
    if bullets:
        candidates.append(bullets)
    if not candidates:
        return []
    # The most labels win; among equals the latest list, which is where the templates put the options
    run = max(candidates, key=lambda r: (len(r), r[0][0]))
    options = []
    for k, (label_start, label_end) in enumerate(run):
        if k + 1 < len(run):
            end = run[k + 1][0]
        else:
            end = text.find("\n", label_end)
            end = len(text) if end == -1 else end
        start = label_end
        while start < end and text[start] in " \t":
            start += 1
        stop = end
        while stop > start and text[stop - 1] in " \t\n;,.":
            stop -= 1
        if stop <= start:
            return []
        options.append((text[label_start:label_end], label_start, start, stop))
    return options


# --------------------------------------------------------------------------------------------------------- segments

_WORD = re.compile(r"[^\W\d_]{2,}")
# An answer such as "C", "(ii).", "[b].", "2)": a label, whatever the options of its instruction parsed to
_LABEL_ANSWER = re.compile(r"^[(\[]?(?:[A-Za-z]|\d{1,2}|[ivxIVX]{1,5})[)\]]?[.):]*$")


def needs_translation(text):
    """A word of two letters or more; numbers, labels and lists of numbers stay as they are."""
    return _WORD.search(text) is not None


class SegmentTooLong(ValueError):
    pass


def _split_long(text, count_tokens, limit):
    """text as parts of at most `limit` tokens, cut at paragraphs, then lines, then sentences: [(kind, str)], the
    separators between the parts as literals. Raises SegmentTooLong for a sentence longer than limit."""
    if count_tokens(text) <= limit:
        return [("S", text)]
    for boundary in (r"\n\s*\n", r"\n", r"(?<=[.!?])\s+"):
        parts = re.split(f"({boundary})", text)  # text, separator, text, ..., text
        if len(parts) == 1:
            continue
        out, current = [], parts[0]
        for i in range(1, len(parts), 2):
            separator, part = parts[i], parts[i + 1]
            if count_tokens(current + separator + part) <= limit:
                current += separator + part
            else:
                out += [("S", current), ("L", separator)]
                current = part
        out.append(("S", current))
        result = []
        for kind, piece in out:
            result.extend(_split_long(piece, count_tokens, limit) if kind == "S" else [(kind, piece)])
        return result
    raise SegmentTooLong(f"{count_tokens(text)} tokens without a sentence boundary: {text[:80]!r}")


def _wrap(text):
    """One part as pieces: the whitespace around it is literal, and so is the part without a word."""
    stripped = text.strip()
    lead = text[:len(text) - len(text.lstrip())]
    trail = text[len(text.rstrip()):] if stripped else ""
    return [("L", lead), ("S" if needs_translation(stripped) else "L", stripped), ("L", trail)]


def text_pieces(text, count_tokens, limit=MAX_SEGMENT_TOKENS, splits=None):
    """One stretch of text as pieces. splits, a list, gets the number of extra segments a long text was cut into."""
    if not needs_translation(text):
        return [("L", text)] if text else []
    stripped = text.strip()
    parts = _split_long(stripped, count_tokens, limit)
    if splits is not None and len(parts) > 1:
        splits.append(sum(kind == "S" for kind, _ in parts) - 1)
    pieces = [("L", text[:len(text) - len(text.lstrip())])]
    for kind, part in parts:
        pieces.extend(_wrap(part) if kind == "S" else [("L", part)])
    pieces.append(("L", text[len(text.rstrip()):]))
    return merge_literals(pieces)


def merge_literals(pieces):
    out = []
    for kind, value in pieces:
        if not value:
            continue
        if kind == "L" and out and out[-1][0] == "L":
            out[-1] = ("L", out[-1][1] + value)
        else:
            out.append((kind, value))
    return out


def _strip_end(text):
    return text.rstrip(" \t\n;,.")


def plan_example(inputs, targets, count_tokens, limit=MAX_SEGMENT_TOKENS):
    """(instruction pieces, answer pieces, info) of one FLAN example. info: options found, how the answer was mapped."""
    options = find_options(inputs)
    pieces, position, splits = [], 0, []
    for k, (label, label_start, start, stop) in enumerate(options):
        before = inputs[position:label_start]
        # Before the first label is the question; between two options only the separators the option left out
        pieces.extend(text_pieces(before, count_tokens, limit, splits) if k == 0 else [("L", before)])
        pieces.append(("L", inputs[label_start:start]))
        pieces.extend(text_pieces(inputs[start:stop], count_tokens, limit, splits))
        position = stop
    tail = inputs[position:]
    if options:
        # What closed the last option (".", ";") stays as it is; the text after it is translated
        separators = len(tail) - len(tail.lstrip(" \t\n;,."))
        pieces.append(("L", tail[:separators]))
        tail = tail[separators:]
    pieces.extend(text_pieces(tail, count_tokens, limit, splits))
    instruction = merge_literals(pieces)

    answer = targets.strip()
    lead = targets[:len(targets) - len(targets.lstrip())]
    trail = targets[len(targets.rstrip()):]
    labels = {_strip_end(label).strip("()[]") for label, _, _, _ in options}
    option_texts = {inputs[start:stop] for _, _, start, stop in options}
    if _LABEL_ANSWER.match(answer) or _strip_end(answer).strip("()[]") in labels:
        core, how = [("L", answer)], "label"
    elif _strip_end(answer) in option_texts and _strip_end(answer):
        option = _strip_end(answer)
        core, how = text_pieces(option, count_tokens, limit) + [("L", answer[len(option):])], "option"
    else:
        core, how = text_pieces(answer, count_tokens, limit, splits), "text"
    answer_pieces = merge_literals([("L", lead)] + core + [("L", trail)])
    return instruction, answer_pieces, {"options": len(options), "answer": how, "splits": sum(splits)}


def segments_of(pieces):
    return [value for kind, value in pieces if kind == "S"]


def render(pieces, translation):
    """The text of pieces with every segment replaced by translation[segment]."""
    return "".join(value if kind == "L" else translation[value] for kind, value in pieces)


# ---------------------------------------------------------------------------------------------------------- checks

def _normalize(text):
    text = unicodedata.normalize("NFKC", text).casefold()
    return re.sub(r"\s+", " ", text).strip()


def contains(haystack, needle):
    """needle as a whole-word stretch of haystack, case and runs of whitespace ignored; an empty needle is never in it."""
    needle = _strip_end(_normalize(needle))
    if not needle:
        return False
    haystack = _normalize(haystack)
    left = r"(?<!\w)" if needle[0].isalnum() else ""
    right = r"(?!\w)" if needle[-1].isalnum() else ""
    return re.search(left + re.escape(needle) + right, haystack) is not None


def answer_lost(inputs_en, targets_en, inputs_nl, targets_nl):
    """Extractive QA: the English answer is in the English instruction, the Dutch answer not in the Dutch one."""
    return contains(inputs_en, targets_en) and not contains(inputs_nl, targets_nl)


# Phrases with which a translator comments instead of translating; only counted when the source has no such phrase
_COMMENTARY = re.compile(
    r"(here is the translation|here's the translation|translation:|translated text|hier is de vertaling|"
    r"vertaling:|de vertaling is|opmerking:|\(opmerking|noot:|\(noot|note:|\(note|let op:|toelichting:|uitleg:|"
    r"explanation:|alternatie(?:f|ve)(?: vertaling)?:|als ai|as an ai|ik kan (?:dit|deze|de) .{0,30}niet vertalen|"
    r"i cannot translate|i can't translate|ik ben een vertaler)", re.IGNORECASE)
DUTCH_LANGUAGES = {"nl", "af"}  # langid takes short Dutch for Afrikaans now and then; a translator does not write it
LANGID_LANGUAGES = ["nl", "en", "de", "fr", "es", "it", "pt", "af"]
MIN_LANGID_WORDS = 5


def check_translation(source, output, ended, identify_language):
    """Why a segment's translation is unusable, or None. identify_language(text) -> language code."""
    if not ended:
        return "truncated"
    if not output.strip():
        return "empty"
    s, o = len(source), len(output)
    if (s >= 30 and not 0.5 <= o / s <= 2.2) or o > 3 * s + 30:
        return "length_ratio"
    match = _COMMENTARY.search(output)
    if match and not _COMMENTARY.search(source) and "translat" not in source.lower():
        return "commentary"
    if len(_WORD.findall(output)) >= MIN_LANGID_WORDS and identify_language(output) not in DUTCH_LANGUAGES:
        return "not_dutch"
    return None


# --------------------------------------------------------------------------------------------------------- the mix

def dutch_sequence(rows, dutch):
    """F's sequence of one host with every FLAN row replaced in place by its Dutch version; a row without one is gone.

    rows: (position, text, inputs, targets, source) of F's mix in position order. dutch: position -> (inputs, targets).
    Returns (sequence of ("web", i) and ("flan", j), web texts, FLAN pool [(inputs, targets, source, F's position)],
    counts). Web rows keep their text and their order.
    """
    sequence, web_texts, pool = [], [], []
    counts = Counter()
    expected = 0
    for position, text, inputs, targets, source in rows:
        if position != expected:
            raise ValueError(f"row {expected} of the sequence has position {position}: rows must come in position order")
        expected += 1
        if source == "culturax":
            if text is None or inputs is not None or targets is not None:
                raise ValueError(f"web row at position {position} is not a text row")
            sequence.append(("web", len(web_texts)))
            web_texts.append(text)
            counts["web"] += 1
            continue
        counts["flan_in"] += 1
        if position not in dutch:
            counts["flan_dropped"] += 1
            continue
        inputs_nl, targets_nl = dutch[position]
        sequence.append(("flan", len(pool)))
        pool.append((inputs_nl, targets_nl, source, position))
        counts["flan_kept"] += 1
    unknown = set(dutch) - {p for _, _, _, p in pool}
    if unknown:
        raise ValueError(f"{len(unknown)} Dutch rows have no FLAN row of F at their position, e.g. {sorted(unknown)[:5]}")
    return sequence, web_texts, pool, dict(counts)


# ---------------------------------------------------------------------------------------------------------- CLI

def _read_parquet(path, columns=None):
    import pyarrow.parquet as pq
    return pq.read_table(path, columns=columns)


def extract(args):
    """The FLAN rows of this host's mix of F, with their positions, in position order."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    mix = open(args.list).read().split()
    if len(mix) != 1:
        raise SystemExit(f"{args.list} lists {len(mix)} files, F has one mix per host")
    table = _read_parquet(mix[0], ["inputs", "targets", "source", "position"])
    build = pq.ParquetFile(mix[0]).schema_arrow.metadata[b"flan_decay_build"].decode()
    flan = table.filter(pc.not_equal(table.column("source"), "culturax"))
    flan = flan.sort_by("position")
    flan = flan.append_column("host", pa.array([args.rank] * flan.num_rows, pa.int32()))
    meta = {b"flan_nl_source_mix": mix[0].encode(), b"flan_decay_build": build.encode()}
    pq.write_table(flan.replace_schema_metadata(meta), args.out, compression="zstd")
    print(f"extract: host {args.rank}, mix {mix[0]} (build {build}): {table.num_rows:,} rows, "
          f"{flan.num_rows:,} FLAN rows -> {args.out}", flush=True)


_TOKENIZER = None


def _load_tokenizer(path):
    global _TOKENIZER
    from transformers import AutoTokenizer
    _TOKENIZER = AutoTokenizer.from_pretrained(path)


def _plan_rows(rows):
    """In a worker process: (status, instruction, answer, info, {segment: tokens}) per (source, inputs, targets)."""
    counted = {}

    def count_tokens(text):
        if text not in counted:
            counted[text] = len(_TOKENIZER(text, add_special_tokens=False)["input_ids"])
        return counted[text]

    out = []
    for source, inputs, targets in rows:
        task = task_of(source)
        reason = task_drop_reason(task)
        instruction, answer, info = [], [], {"options": 0, "answer": "", "splits": 0}
        if reason is not None:
            status = f"task:{reason}"
        elif task in DUTCH_TASKS:
            status = "dutch_task"
        else:
            try:
                instruction, answer, info = plan_example(inputs, targets, count_tokens)
                status = "translate"
            except SegmentTooLong:
                status = "segment_too_long"
        tokens = {v: count_tokens(v) for v in segments_of(instruction) + segments_of(answer)}
        out.append((status, instruction, answer, info, tokens))
    return out


def plan(args):
    """Task filters, the pieces of every example and the distinct segments, split over the workers by length."""
    import concurrent.futures

    import pyarrow as pa
    import pyarrow.parquet as pq

    examples = {k: [] for k in ("host", "position", "source", "status", "instruction", "answer", "options", "answer_kind",
                                "splits")}
    counted = {}
    report = Counter()
    chunk = 2000
    with concurrent.futures.ProcessPoolExecutor(args.processes, initializer=_load_tokenizer,
                                                initargs=(args.tokenizer,)) as pool:
        for path in args.english:
            table = _read_parquet(path)
            columns = {c: table.column(c).to_pylist() for c in ("host", "position", "source", "inputs", "targets")}
            rows = list(zip(columns["source"], columns["inputs"], columns["targets"]))
            planned = [r for part in pool.map(_plan_rows, [rows[k:k + chunk] for k in range(0, len(rows), chunk)])
                       for r in part]
            for h, p, s, (status, instruction, answer, info, tokens) in zip(columns["host"], columns["position"],
                                                                           columns["source"], planned):
                counted.update(tokens)
                for key, value in (("host", h), ("position", p), ("source", s), ("status", status),
                                   ("instruction", instruction), ("answer", answer), ("options", info["options"]),
                                   ("answer_kind", info["answer"]), ("splits", info["splits"])):
                    examples[key].append(value)
                report[status] += 1
            print(f"plan: {path}: {len(examples['host']):,} examples, {len(counted):,} distinct segments so far", flush=True)

    def count_tokens(text):
        return counted[text]

    # Shortest first, dealt round-robin: every worker gets the same mix of lengths, and a shard of consecutive
    # segments has similar lengths, which suits the engine's length buckets
    texts = sorted(counted, key=lambda t: (counted[t], t))
    segment_ids = {t: i for i, t in enumerate(texts)}
    encode = {"L": lambda v: v, "S": lambda v: segment_ids[v]}
    examples["instruction"] = [json.dumps([[k, encode[k](v)] for k, v in p], ensure_ascii=False) for p in examples["instruction"]]
    examples["answer"] = [json.dumps([[k, encode[k](v)] for k, v in p], ensure_ascii=False) for p in examples["answer"]]
    pq.write_table(pa.table(examples), f"{args.out}/examples.parquet", compression="zstd")
    tokens = [count_tokens(t) for t in texts]
    for worker in range(args.world):
        pq.write_table(pa.table({"id": list(range(worker, len(texts), args.world)), "text": texts[worker::args.world],
                                 "tokens": tokens[worker::args.world]}),
                       f"{args.out}/segments-w{worker}.parquet", compression="zstd")
    summary = {"examples": len(examples["host"]), "status": dict(report), "segments": len(texts),
               "segment_tokens": sum(tokens), "answer_kind": dict(Counter(examples["answer_kind"])),
               "with_options": sum(1 for n in examples["options"] if n),
               "examples_split": sum(1 for n in examples["splits"] if n), "extra_segments_from_splits": sum(examples["splits"]),
               "max_segment_tokens": MAX_SEGMENT_TOKENS, "tokenizer": args.tokenizer}
    json.dump(summary, open(f"{args.out}/plan.json", "w"), indent=1)
    print(f"PLAN {json.dumps(summary)}", flush=True)


def _identify(texts):
    import langid
    langid.set_languages(LANGID_LANGUAGES)
    return [langid.classify(t)[0] for t in texts]


def assemble(args):
    """Translations back into examples; per host the Dutch rows and what was dropped why."""
    import concurrent.futures
    import glob

    import pyarrow as pa
    import pyarrow.parquet as pq

    segments = {}
    for path in sorted(glob.glob(f"{args.plan}/segments-w*.parquet")):
        table = _read_parquet(path, ["id", "text"])
        segments.update(zip(table.column("id").to_pylist(), table.column("text").to_pylist()))
    translated = {}
    for path in sorted(glob.glob(f"{args.translations}/**/*.parquet", recursive=True)):
        table = _read_parquet(path, ["id", "text_nl", "ended"])
        for i, out, ended in zip(*(table.column(c).to_pylist() for c in ("id", "text_nl", "ended"))):
            if i in translated:
                raise ValueError(f"segment {i} translated twice ({path})")
            translated[i] = (out, ended)
    missing = set(segments) - set(translated)
    if missing:
        raise SystemExit(f"{len(missing):,} of {len(segments):,} segments have no translation, e.g. {sorted(missing)[:5]}")
    ids = sorted(segments)
    # Language ID is the slow part: in processes, in chunks
    with concurrent.futures.ProcessPoolExecutor(args.processes) as pool:
        chunk = 2000
        outputs = [translated[i][0] for i in ids]
        languages = [lang for part in pool.map(_identify, [outputs[k:k + chunk] for k in range(0, len(ids), chunk)])
                     for lang in part]
    language = dict(zip(ids, languages))
    failed = {}
    for i in ids:
        out, ended = translated[i]
        why = check_translation(segments[i], out, ended, lambda _: language[i])
        if why:
            failed[i] = why
    print(f"assemble: {len(ids):,} segments, {len(failed):,} unusable: {dict(Counter(failed.values()))}", flush=True)

    examples = _read_parquet(f"{args.plan}/examples.parquet")
    columns = {c: examples.column(c).to_pylist() for c in examples.column_names}
    english = {}
    for path in args.english:
        table = _read_parquet(path)
        for h, p, x, y in zip(*(table.column(c).to_pylist() for c in ("host", "position", "inputs", "targets"))):
            english[(h, p)] = (x, y)
    translation = {i: translated[i][0] for i in ids}
    rows = {h: {"position": [], "source": [], "inputs": [], "targets": []} for h in range(args.world)}
    report = {h: Counter() for h in range(args.world)}
    for k in range(len(columns["host"])):
        h, p, status = columns["host"][k], columns["position"][k], columns["status"][k]
        report[h]["flan_rows"] += 1
        x, y = english[(h, p)]
        if status == "dutch_task":
            report[h]["kept_dutch_task"] += 1
            out = (x, y)
        elif status != "translate":
            report[h][f"dropped:{status}"] += 1
            continue
        else:
            instruction = [(kind, value) for kind, value in json.loads(columns["instruction"][k])]
            answer = [(kind, value) for kind, value in json.loads(columns["answer"][k])]
            bad = [failed[v] for kind, v in instruction + answer if kind == "S" and v in failed]
            if bad:
                report[h][f"dropped:segment_{bad[0]}"] += 1
                continue
            out = (render(instruction, translation), render(answer, translation))
            if answer_lost(x, y, *out):
                report[h]["dropped:answer_not_in_instruction"] += 1
                continue
            if not out[0].strip() or not out[1].strip():
                report[h]["dropped:empty"] += 1
                continue
            report[h]["translated"] += 1
        for key, value in (("position", p), ("source", columns["source"][k]), ("inputs", out[0]), ("targets", out[1])):
            rows[h][key].append(value)
    for h in range(args.world):
        pq.write_table(pa.table(rows[h]), f"{args.out}/w{h}.parquet", compression="zstd")
    summary = {"segments": len(ids), "segments_unusable": dict(Counter(failed.values())),
               "hosts": {h: dict(sorted(report[h].items())) for h in range(args.world)}}
    json.dump(summary, open(f"{args.out}/report.json", "w"), indent=1)
    print(f"ASSEMBLE {json.dumps(summary)}", flush=True)


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="The Dutch FLAN of arm F-nl")
    sub = parser.add_subparsers(dest="step", required=True)
    p = sub.add_parser("extract", help="this host's FLAN rows of F")
    p.add_argument("--rank", type=int, required=True)
    p.add_argument("--list", required=True, help="~/flan_decay_files.txt: F's mix of this host")
    p.add_argument("--out", required=True)
    p = sub.add_parser("plan", help="filters, pieces and distinct segments")
    p.add_argument("--english", nargs="+", required=True, help="the extract files of all hosts")
    p.add_argument("--tokenizer", required=True, help="TranslateGemma's tokenizer directory")
    p.add_argument("--out", required=True)
    p.add_argument("--world", type=int, default=4)
    p.add_argument("--processes", type=int, default=64)
    p = sub.add_parser("assemble", help="the translations back into examples")
    p.add_argument("--english", nargs="+", required=True)
    p.add_argument("--plan", required=True)
    p.add_argument("--translations", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--world", type=int, default=4)
    p.add_argument("--processes", type=int, default=64)
    args = parser.parse_args(argv)
    {"extract": extract, "plan": plan, "assemble": assemble}[args.step](args)


if __name__ == "__main__":
    main()
