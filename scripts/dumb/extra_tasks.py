"""The tasks of finetune_extra.py: NER, POS and extractive QA, read from the parquet files of
experiments/dumb-extra/stage_data.py (gs://t5-dutch-english-usc2/mbert/dumb-extra/data/, copied to a local directory).

Token tasks: each word is labelled at its first sub-token, the other sub-tokens are not trained or scored (-100), as in
DUMB's token-classification trainer and Hugging Face's run_ner.py. The words go to the tokenizer pre-split
(is_split_into_words), so each word starts with ▁, punctuation too. A sentence of more than max_length tokens (a few
lists of sports results in CoNLL-2002) is not truncated but cut into consecutive chunks of whole words, each a row of
its own; its predictions are joined again before scoring. Truncation would drop words, and this tokenizer truncates
from the left (truncation_side "left" in its tokenizer_config.json).

QA: question and context as a pair, <s> question <s> context with this tokenizer; a context longer than the window is
cut into windows of max_length tokens that overlap by doc_stride tokens (Hugging Face's run_qa.py and DUMB). A window
without the whole answer is trained to point at position 0, <s>. Predictions: the best start + end logit over a
question's windows among the n_best starts and ends, within the context and at most max_answer_length tokens.
"""

from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

NER_LABELS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]
# The 17 UPOS tags of Universal Dependencies; PART does not occur in the Dutch treebanks but keeps the label set UD's
UPOS_LABELS = ["ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ",
               "SYM", "VERB", "X"]

# split role -> file of the staged directory; dev ("validation") is for logging only
TASKS = {
    "squadnl": {"kind": "qa", "dir": "squad-nl-v1.1", "files": {"train": "train", "validation": "dev", "test": "test"},
                "metric": "f1"},
    "conll2002": {"kind": "token", "dir": "conll2002-nl", "files": {"train": "train", "validation": "validation", "test": "test"},
                  "labels_column": "ner_tags", "labels": NER_LABELS, "metric": "entity_f1"},
    "pos-lassysmall": {"kind": "token", "dir": "ud-nl-lassysmall", "files": {"train": "train", "validation": "dev", "test": "test"},
                       "labels_column": "upos", "labels": UPOS_LABELS, "metric": "accuracy"},
    "pos-alpino": {"kind": "token", "dir": "ud-nl-alpino", "files": {"train": "train", "validation": "dev", "test": "test"},
                   "labels_column": "upos", "labels": UPOS_LABELS, "metric": "accuracy"},
}


def get_task(name):
    if name not in TASKS:
        raise ValueError(f"Unknown task: {name}. Available: {list(TASKS)}")
    return TASKS[name]


def read_split(data_dir, task, role, max_examples=None):
    path = Path(data_dir) / TASKS[task]["dir"] / f"{TASKS[task]['files'][role]}.parquet"
    rows = pq.read_table(path).to_pylist()
    return rows if max_examples is None else rows[:max_examples]


def encode_words(tokenizer, sentences, max_length):
    """Rows covering every word of every sentence: [(sentence index, index of its first word, input ids, the position
    of each of its words' first sub-token)]. A sentence that fits in max_length tokens is one row; a longer one is cut
    between words into rows that each fit. Pre-split words are tokenized one by one, so a row's tokens are its words'."""
    encoded = tokenizer(sentences, is_split_into_words=True)
    rows = []
    for i, words in enumerate(sentences):
        word_ids = encoded.word_ids(i)
        if len(word_ids) <= max_length:
            chunks = [(0, len(words))]
        else:
            sizes = np.bincount([w for w in word_ids if w is not None], minlength=len(words))
            specials = len(word_ids) - int(sizes.sum())
            chunks, start, used = [], 0, specials
            for w, size in enumerate(sizes):
                if specials + size > max_length:
                    raise ValueError(f"sentence {i}: word {w} alone has {size} tokens, more than --max-length {max_length} holds")
                if used + size > max_length:
                    chunks.append((start, w))
                    start, used = w, specials
                used += size
            chunks.append((start, len(words)))
        for start, end in chunks:
            chunk = encoded if len(chunks) == 1 else tokenizer([words[start:end]], is_split_into_words=True)
            index = i if len(chunks) == 1 else 0
            first, previous = [], None
            for position, word in enumerate(chunk.word_ids(index)):
                if word is not None and word != previous:
                    first.append(position)
                previous = word
            input_ids = chunk["input_ids"][index]
            if len(first) != end - start or len(input_ids) > max_length:
                raise ValueError(f"sentence {i}, words {start}-{end}: {len(first)} first sub-tokens in {len(input_ids)} tokens")
            rows.append((i, start, input_ids, first))
    return rows


def token_examples(tokenizer, rows, labels, labels_column, max_length):
    """[{sentence, input_ids, labels (a word's label id at its first sub-token, -100 elsewhere), first, word_labels}],
    one per row of encode_words, in the order of the sentences and their words."""
    label_id = {label: i for i, label in enumerate(labels)}
    examples = []
    for sentence, start, input_ids, first in encode_words(tokenizer, [row["tokens"] for row in rows], max_length):
        word_labels = [label_id[label] for label in rows[sentence][labels_column][start:start + len(first)]]
        position_labels = [-100] * len(input_ids)
        for position, label in zip(first, word_labels):
            position_labels[position] = label
        examples.append({"sentence": sentence, "input_ids": input_ids, "labels": position_labels, "first": first,
                         "word_labels": word_labels})
    return examples


def answer_positions(offsets, start_char, end_char):
    """(start, end) token positions of the characters [start_char, end_char) in a window, or (0, 0) when the window
    does not hold them all. offsets: per position (start, end) in the context, None outside it."""
    context = [i for i, offset in enumerate(offsets) if offset is not None]
    first, last = context[0], context[-1]
    if not (offsets[first][0] <= start_char and offsets[last][1] >= end_char):
        return 0, 0
    start = first
    while start <= last and offsets[start][0] <= start_char:
        start += 1
    end = last
    while end >= first and offsets[end][1] >= end_char:
        end -= 1
    return start - 1, end + 1


def qa_features(tokenizer, rows, max_length, doc_stride):
    """Windows of the questions: [{input_ids, example (index into rows), offsets, start_position, end_position}].
    The positions are those of the first answer. Questions lose their leading whitespace, as in run_qa.py: one train
    question of SQuAD-NL v1.1 is 1,609 tokens of it, longer than a window."""
    encoded = tokenizer([row["question"].lstrip() for row in rows], [row["context"] for row in rows], truncation="only_second",
                        max_length=max_length, stride=doc_stride, return_overflowing_tokens=True, return_offsets_mapping=True)
    features = []
    for i, input_ids in enumerate(encoded["input_ids"]):
        example = encoded["overflow_to_sample_mapping"][i]
        offsets = [tuple(offset) if sequence == 1 else None
                   for offset, sequence in zip(encoded["offset_mapping"][i], encoded.sequence_ids(i))]
        answers = rows[example]["answers"]
        start_char = answers["answer_start"][0]
        start, end = answer_positions(offsets, start_char, start_char + len(answers["text"][0]))
        features.append({"input_ids": input_ids, "example": example, "offsets": offsets, "start_position": start, "end_position": end})
    covered = {feature["example"] for feature in features}
    if len(covered) != len(rows):
        raise ValueError(f"{len(rows) - len(covered)} questions without a window")
    return features


def qa_predictions(rows, features, start_logits, end_logits, n_best=20, max_answer_length=30):
    """{id: answer text} and the number of questions without a valid span (their answer is "").

    start_logits, end_logits: per feature, an array at least as long as its input ids (positions past them are padding).
    """
    by_example = defaultdict(list)
    for f, feature in enumerate(features):
        by_example[feature["example"]].append(f)
    predictions = {}
    no_span = 0
    for e, row in enumerate(rows):
        best = None
        for f in by_example[e]:
            offsets = features[f]["offsets"]
            starts = np.asarray(start_logits[f][:len(offsets)])
            ends = np.asarray(end_logits[f][:len(offsets)])
            for s in np.argsort(starts)[::-1][:n_best]:
                for t in np.argsort(ends)[::-1][:n_best]:
                    if offsets[s] is None or offsets[t] is None or t < s or t - s + 1 > max_answer_length:
                        continue
                    score = float(starts[s] + ends[t])
                    if best is None or score > best[0]:
                        best = (score, offsets[s][0], offsets[t][1])
        if best is None:
            no_span += 1
            predictions[row["id"]] = ""
        else:
            predictions[row["id"]] = row["context"][best[1]:best[2]].strip()
    return predictions, no_span
