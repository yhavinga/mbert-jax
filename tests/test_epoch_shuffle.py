"""Passes over a host's files in the trainer's loader (EpochTrackingDataLoader with --data-files-list), for the final
run of two epochs: every pass has its own seed and order, no batch of pass 2 repeats one of pass 1, documents of
blocked hosts never reach a batch, and a data state of any batch, also the last one of a pass, resumes exactly.

The shuffle buffer is small against a file here, as in the real run (10,000 documents against ~917,000 per CulturaX
file): there the files' order and the buffer are what make a pass's order.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_multihost import MAX_LENGTH, VOCAB_WORDS, _make_tokenizer  # noqa: E402

from mmbjax.training import EpochTrackingDataLoader  # noqa: E402

ROWS = 4
BUFFER = 16
DOCUMENTS_PER_FILE = 150
FILES = 3
BLOCKED_EVERY = 7
# Only the documents of blocked hosts have this word: one in a batch is a blocked document
MARKER = f"w{VOCAB_WORDS - 1}"


def write_files(tmp_path):
    """FILES parquet files with text and url; every BLOCKED_EVERY-th document is a page of (www.)hebban.nl."""
    rng = np.random.RandomState(7)
    seen, paths, blocked, kept = set(), [], 0, set()
    for f in range(FILES):
        texts, urls = [], []
        while len(texts) < DOCUMENTS_PER_FILE:
            words = tuple(int(w) for w in rng.randint(0, VOCAB_WORDS - 1, size=rng.randint(3, 11)))
            if words in seen:
                continue
            seen.add(words)
            n = len(texts)
            if n % BLOCKED_EVERY == 3:
                texts.append(" ".join(f"w{w}" for w in words) + f" {MARKER}")
                urls.append(f"https://{'www.' if n % 2 else ''}hebban.nl/recensie/{f}-{n}")
                blocked += 1
            else:
                texts.append(" ".join(f"w{w}" for w in words))
                urls.append(f"https://example{n % 5}.nl/{f}/{n}")
                kept.add(words)
        path = tmp_path / f"nl_part_{f:05d}.parquet"
        pq.write_table(pa.table({"text": texts, "timestamp": ["2020"] * len(texts), "url": urls}), path, row_group_size=50)
        paths.append(str(path))
    return paths, blocked, kept


def loader_of(paths, blocklist=("hebban.nl",)):
    return EpochTrackingDataLoader(dataset_name="uonlp/CulturaX", tokenizer=_make_tokenizer(), max_length=MAX_LENGTH,
                                   batch_size=ROWS, seed=42, shuffle_buffer_size=BUFFER, data_files=paths,
                                   url_blocklist=blocklist)


def documents_of(batch, tokenizer):
    """The documents of a packed batch in row order, each as the tuple of its word numbers (one piece each here)."""
    first_word = 5  # [PAD] [UNK] [CLS] [SEP] [MASK], then w0, w1, ...
    docs = []
    for ids, positions in zip(batch["input_ids"], batch["input_ids_positions"]):
        starts = [i for i, p in enumerate(positions) if p == 0 and ids[i] != tokenizer.pad_token_id]
        for start, end in zip(starts, starts[1:] + [len(ids)]):
            piece = [int(t) for t in ids[start:end] if t not in (tokenizer.pad_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id)]
            docs.append(tuple(t - first_word for t in piece))
    return docs


def read_passes(loader, passes):
    """(pass number, batch, state after it) for every batch of the first `passes` passes."""
    out, iterator = [], iter(loader)
    while True:
        batch = next(iterator)
        if loader.epoch >= passes:
            return out
        out.append((loader.epoch, batch, loader.get_state()))


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    return write_files(tmp_path_factory.mktemp("passes"))


@pytest.fixture(scope="module")
def three_passes(files):
    paths, _, _ = files
    return read_passes(loader_of(paths), 3)


def test_each_pass_reads_every_kept_document_once_and_no_blocked_one(files, three_passes):
    _, blocked, kept = files
    tokenizer = _make_tokenizer()
    marker = tokenizer.convert_tokens_to_ids(MARKER)
    for p in range(3):
        batches = [batch for epoch, batch, _ in three_passes if epoch == p]
        assert not any((batch["input_ids"] == marker).any() for batch in batches), p
        docs = [doc for batch in batches for doc in documents_of(batch, tokenizer)]
        assert len(docs) == len(set(docs)), f"a document twice in pass {p + 1}"
        assert set(docs) <= kept
        # Packing drops the last incomplete batch of a pass, nothing else
        assert len(docs) > len(kept) - 4 * ROWS, (p, len(docs), len(kept))
    assert blocked == FILES * len(range(3, DOCUMENTS_PER_FILE, BLOCKED_EVERY)) and len(kept) == FILES * DOCUMENTS_PER_FILE - blocked


def test_pass_two_has_its_own_seed_and_repeats_no_batch_of_pass_one(files, three_passes, capsys):
    tokenizer = _make_tokenizer()
    batches = {p: [batch for epoch, batch, _ in three_passes if epoch == p] for p in range(3)}
    keys = {p: [batch["input_ids"].tobytes() for batch in batches[p]] for p in range(3)}
    for p in range(3):
        assert len(set(keys[p])) == len(keys[p])
    assert not set(keys[0]) & set(keys[1]), "a batch of pass 2 is one of pass 1"
    assert not set(keys[1]) & set(keys[2])
    # Not only other batches: another order of the documents. Few keep their place, few keep their neighbour.
    order = {p: [doc for batch in batches[p] for doc in documents_of(batch, tokenizer)] for p in range(2)}
    position = {doc: i for i, doc in enumerate(order[0])}
    same_place = sum(position.get(doc) == i for i, doc in enumerate(order[1]))
    pairs = set(zip(order[0], order[0][1:]))
    same_neighbour = sum(pair in pairs for pair in zip(order[1], order[1][1:]))
    assert same_place < 0.05 * len(order[1]) and same_neighbour < 0.05 * len(order[1]), (same_place, same_neighbour)
    # The pass and its seed, in the log of each pass; and the blocked documents each pass left out
    paths, blocked, _ = files
    read_passes(loader_of(paths), 2)
    log = capsys.readouterr().out
    assert "Pass 1 of shard 0/1 with seed 42" in log and "Pass 2 of shard 0/1 with seed 43" in log
    assert log.count(f"{blocked:,} documents of blocked hosts left out of this pass") == 2


def test_a_data_state_resumes_exactly_also_at_the_end_of_a_pass(files, three_passes):
    """From the state after the last batch of pass 1 and from one in the middle of pass 2, a new loader gives the
    very batches the first one gave next, on into pass 3."""
    paths, _, _ = files
    epochs = [epoch for epoch, _, _ in three_passes]
    last_of_pass_1 = epochs.index(1) - 1
    middle_of_pass_2 = (epochs.index(1) + epochs.index(2)) // 2
    for at in (last_of_pass_1, middle_of_pass_2):
        resumed = loader_of(paths)
        resumed.set_state(three_passes[at][2])
        iterator = iter(resumed)
        expected = [batch for _, batch, _ in three_passes[at + 1:]]
        assert len(expected) > 20
        for i, want in enumerate(expected):
            got = next(iterator)
            assert got["input_ids"].tobytes() == want["input_ids"].tobytes(), (at, i)
            assert got["input_ids_positions"].tobytes() == want["input_ids_positions"].tobytes(), (at, i)


def test_without_the_blocklist_the_blocked_documents_are_read(files):
    paths, blocked, _ = files
    tokenizer = _make_tokenizer()
    marker = tokenizer.convert_tokens_to_ids(MARKER)
    batches = [batch for _, batch, _ in read_passes(loader_of(paths, blocklist=None), 1)]
    assert sum(int((batch["input_ids"] == marker).sum()) for batch in batches) > blocked - 4 * ROWS


def test_blocklist_needs_local_files():
    with pytest.raises(ValueError, match="needs data_files"):
        EpochTrackingDataLoader(dataset_name="x", tokenizer=_make_tokenizer(), url_blocklist=("hebban.nl",))
