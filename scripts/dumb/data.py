"""Dataset loading for DUMB fine-tuning tasks.

Loads datasets directly from HuggingFace Hub.
"""

from datasets import load_dataset, DatasetDict
from typing import Callable


def load_dbrd(tokenizer: Callable, max_length: int = 512) -> DatasetDict:
    """Load DBRD (Dutch Book Reviews Dataset) for sentiment classification.

    Binary classification: positive (1) vs negative (0).
    ~20K train, ~2K test.
    """
    ds = load_dataset("dbrd")

    # Create validation split from train
    train_valid = ds["train"].train_test_split(test_size=500, seed=42)
    ds = DatasetDict({
        "train": train_valid["train"],
        "validation": train_valid["test"],
        "test": ds["test"],
    })

    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )

    ds = ds.map(tokenize, batched=True, remove_columns=["text"])
    return ds


def load_sicknl(tokenizer: Callable, max_length: int = 128) -> DatasetDict:
    """Load SICK-NL for NLI classification.

    3-class classification: entailment (0), neutral (1), contradiction (2).
    ~4.4K train, ~0.5K validation, ~4.9K test.
    """
    ds = load_dataset("maximedb/sick_nl")

    def process(examples):
        # Tokenize sentence pairs
        tokenized = tokenizer(
            examples["sentence_A"],
            examples["sentence_B"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        # Label column is already integer (0, 1, 2)
        tokenized["label"] = examples["label"]
        return tokenized

    # Keep only the columns we need
    columns_to_remove = [
        "pair_ID", "sentence_A", "sentence_B", "entailment_label",
        "relatedness_score", "entailment_AB", "entailment_BA",
        "sentence_A_original", "sentence_B_original",
        "sentence_A_dataset", "sentence_B_dataset", "SemEval_set",
        "label_seq2seq"
    ]

    ds = ds.map(process, batched=True, remove_columns=columns_to_remove)
    return ds


def load_sicknl_similarity(tokenizer: Callable, max_length: int = 128) -> DatasetDict:
    """Load SICK-NL for semantic similarity (embeddings evaluation).

    Regression task: predict relatedness score (1-5, normalized to 0-1).
    """
    ds = load_dataset("maximedb/sick_nl")

    def process(examples):
        tokenized = tokenizer(
            examples["sentence_A"],
            examples["sentence_B"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        # Normalize relatedness score to 0-1
        tokenized["score"] = [s / 5.0 for s in examples["relatedness_score"]]
        return tokenized

    columns_to_remove = [
        "pair_ID", "sentence_A", "sentence_B", "entailment_label",
        "relatedness_score", "entailment_AB", "entailment_BA",
        "sentence_A_original", "sentence_B_original",
        "sentence_A_dataset", "sentence_B_dataset", "SemEval_set",
        "label", "label_seq2seq"
    ]

    ds = ds.map(process, batched=True, remove_columns=columns_to_remove)
    return ds


# Task registry
TASKS = {
    "dbrd": {
        "loader": load_dbrd,
        "num_labels": 2,
        "task_type": "classification",
        "metric": "accuracy",
    },
    "sicknl": {
        "loader": load_sicknl,
        "num_labels": 3,
        "task_type": "classification",
        "metric": "accuracy",
    },
    "sicknl_sim": {
        "loader": load_sicknl_similarity,
        "num_labels": 1,
        "task_type": "regression",
        "metric": "spearman",
    },
}


def get_task(task_name: str) -> dict:
    """Get task configuration by name."""
    if task_name not in TASKS:
        raise ValueError(f"Unknown task: {task_name}. Available: {list(TASKS.keys())}")
    return TASKS[task_name]
