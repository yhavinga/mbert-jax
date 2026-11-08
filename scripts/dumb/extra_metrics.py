"""Metrics of the tasks of finetune_extra.py: SQuAD exact match and F1, entity-level F1 of IOB2 tags, word accuracy.

No dependency beyond the standard library: the VM's lock (vm.lock) has neither seqeval nor evaluate.
"""

import re
import string
from collections import Counter

PUNCTUATION = set(string.punctuation)


def squad_normalize(text):
    """normalize_answer of the official SQuAD v1.1 evaluation script: lower case, punctuation and the English articles
    a/an/the removed, whitespace collapsed. Kept as it is for Dutch, so the numbers mean what SQuAD numbers mean."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in PUNCTUATION)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def squad_exact_match(prediction, truth):
    return squad_normalize(prediction) == squad_normalize(truth)


def squad_f1(prediction, truth):
    """f1_score of the official script: the overlap of the normalized tokens as bags."""
    prediction_tokens = squad_normalize(prediction).split()
    truth_tokens = squad_normalize(truth).split()
    common = Counter(prediction_tokens) & Counter(truth_tokens)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision = same / len(prediction_tokens)
    recall = same / len(truth_tokens)
    return 2 * precision * recall / (precision + recall)


def squad_scores(predictions, answers):
    """Exact match and F1 as fractions, each the mean over questions of the best over the question's answers.

    predictions: {id: text}; answers: {id: [text, ...]}. Every question must have a prediction, as the official
    script counts a missing one as 0 and that would hide a lost feature.
    """
    missing = sorted(set(answers) - set(predictions))
    if missing:
        raise ValueError(f"{len(missing)} questions without a prediction, e.g. {missing[:3]}")
    exact = [max(squad_exact_match(predictions[i], truth) for truth in truths) for i, truths in answers.items()]
    f1 = [max(squad_f1(predictions[i], truth) for truth in truths) for i, truths in answers.items()]
    return {"exact_match": sum(exact) / len(exact), "f1": sum(f1) / len(f1), "questions": len(exact)}


def iob2_entities(tags):
    """(type, first, last) of the entities in one sentence's tags, as seqeval's default (conlleval) mode reads them.

    An entity starts at B-X, or at an I-X that does not continue an entity of type X; it ends before O, B or a tag of
    another type. So I-X after O counts as an entity, as in conlleval.
    """
    entities = []
    start, kind = None, None
    for i, tag in enumerate(list(tags) + ["O"]):
        prefix, _, tag_type = tag.partition("-")
        if prefix not in ("B", "I", "O") or (prefix == "O") != (tag_type == ""):
            raise ValueError(f"not an IOB2 tag: {tag!r}")
        if start is not None and (prefix in ("O", "B") or tag_type != kind):
            entities.append((kind, start, i - 1))
            start, kind = None, None
        if prefix == "B" or (prefix == "I" and start is None):
            start, kind = i, tag_type
    return entities


def entity_scores(true_sentences, predicted_sentences):
    """Micro precision, recall and F1 over all entities of all sentences, and per type, as seqeval's
    classification_report in its default mode. A predicted entity counts when its type and both ends are right."""
    if len(true_sentences) != len(predicted_sentences):
        raise ValueError(f"{len(true_sentences)} sentences of tags, {len(predicted_sentences)} of predictions")
    true, predicted = set(), set()
    for s, (true_tags, predicted_tags) in enumerate(zip(true_sentences, predicted_sentences)):
        if len(true_tags) != len(predicted_tags):
            raise ValueError(f"sentence {s}: {len(true_tags)} tags, {len(predicted_tags)} predictions")
        true.update((s,) + entity for entity in iob2_entities(true_tags))
        predicted.update((s,) + entity for entity in iob2_entities(predicted_tags))

    def score(true_set, predicted_set):
        correct = len(true_set & predicted_set)
        precision = correct / len(predicted_set) if predicted_set else 0.0
        recall = correct / len(true_set) if true_set else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {"precision": precision, "recall": recall, "f1": f1, "true": len(true_set), "predicted": len(predicted_set), "correct": correct}

    result = score(true, predicted)
    types = sorted({entity[1] for entity in true | predicted})
    result["per_type"] = {t: score({e for e in true if e[1] == t}, {e for e in predicted if e[1] == t}) for t in types}
    return result


def word_accuracy(true_sentences, predicted_sentences):
    """The share of words whose label is right, over all words of all sentences."""
    words = correct = 0
    for s, (true_labels, predicted_labels) in enumerate(zip(true_sentences, predicted_sentences, strict=True)):
        if len(true_labels) != len(predicted_labels):
            raise ValueError(f"sentence {s}: {len(true_labels)} labels, {len(predicted_labels)} predictions")
        words += len(true_labels)
        correct += sum(t == p for t, p in zip(true_labels, predicted_labels))
    return {"accuracy": correct / words, "words": words, "correct": correct}
