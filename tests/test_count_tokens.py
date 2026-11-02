"""scripts/count_tokens.py: its closed form of the pieces, and its length-only stand-in for the tokenizer."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import count_tokens  # noqa: E402

from src.mmbjax.training.packing import (  # noqa: E402
    GRAIN_AVAILABLE,
    create_packed_dataset_streaming,
    split_into_pieces,
)


@pytest.mark.parametrize("max_length", [3, 4, 7, 16, 1024])
def test_piece_lengths_matches_split_into_pieces(max_length):
    n = np.arange(0, 5 * max_length + 3)
    pieces, tokens = count_tokens.piece_lengths(n, max_length)
    for k in n:
        real = split_into_pieces(list(range(10, 10 + k)), max_length, bos_token_id=1, eos_token_id=2)
        assert pieces[k] == len(real), k
        assert tokens[k] == sum(len(p) for p in real), k


def test_file_order_keeps_the_stratum_out_of_the_random_order():
    files = [f"nl/nl_part_{i:05d}.parquet" for i in range(20)]
    stratum, order = count_tokens.file_order(files, {"nl_part_00000", "nl_part_00001"})
    assert stratum == files[:2]
    assert sorted(order) == files[2:] and order != files[2:]
    assert count_tokens.file_order(files, {"nl_part_00000", "nl_part_00001"})[1] == order


def test_ratio_estimate_of_the_whole_population_is_its_total():
    y, x = [10, 31, 25, 7], [2, 6, 5, 1]
    total, se, ratio, ratio_se = count_tokens.ratio_estimate(y, x, sum(x), len(x))
    assert total == pytest.approx(sum(y)) and se == 0.0
    total, se, _, _ = count_tokens.ratio_estimate(y[:3], x[:3], sum(x), len(x))
    assert total == pytest.approx(66 / 13 * 14) and se > 0


def test_weighted_percentiles_with_unit_weights_are_the_inverted_cdf():
    values = np.random.default_rng(1).integers(0, 5000, size=1001)
    expected = [int(v) for v in np.percentile(values, [50, 90, 99], method="inverted_cdf")]
    assert count_tokens.weighted_percentiles(values, np.ones(len(values)), (50, 90, 99)) == expected


class WordTokenizer:
    """One token per word, ids from 10 up: enough of a tokenizer for TokenizeTransform."""

    bos_token_id, eos_token_id, pad_token_id = 1, 2, 0

    def __call__(self, text, **kwargs):
        return {"input_ids": [10 + len(w) for w in text.split()]}


@pytest.mark.skipif(not GRAIN_AVAILABLE, reason="grain is not installed")
def test_length_tokenizer_packs_like_the_text():
    rng = np.random.default_rng(0)
    lengths = [int(n) for n in rng.choice([0, 1, 5, 40, 61, 62, 63, 64, 130, 300], size=400)]
    texts = [" ".join("w" * (1 + i % 3) for i in range(n)) for n in lengths]

    def batches(stream, tokenizer):
        return list(create_packed_dataset_streaming(stream, tokenizer=tokenizer, max_length=64, batch_size=8, seed=3))

    real = batches([{"text": t} for t in texts], WordTokenizer())
    stand_in = batches([{"text": n} for n in lengths], count_tokens.LengthTokenizer(1, 2, 0, token_id=100))
    assert len(real) == len(stand_in) > 0
    for a, b in zip(real, stand_in):
        assert a.keys() == b.keys()
        for key in a:
            if key == "input_ids":
                # Same layout: specials and padding where the text has them, the stand-in token elsewhere
                np.testing.assert_array_equal(np.where(a[key] > 2, 100, a[key]), b[key])
            else:
                np.testing.assert_array_equal(a[key], b[key])
