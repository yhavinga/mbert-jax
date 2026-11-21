"""Arm G4-7 (spec translator-sft/g4-rl-spec.md, online RL with GRPO): g4rl.py's pure parts on small synthetic inputs (no
tokenizer, no model, no TPU): the pool and RL-val, the plan after the anchor, the bucket protocol's readers, the reward,
the advantage, the packed rows with their per-token arrays, the loss's arithmetic in numpy, the stop rules; iter7 end
to end; sft_lora.py's --grpo argument checks; translate.py's token ids, log-probabilities, copies and DONE marker. The
JAX parts and the equality with Tunix's functions are in test_en_nl_g4rl_jax.py."""

import hashlib
import json
import math
import random
import sys
from pathlib import Path
from types import SimpleNamespace as ns

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "en-nl-translator"))
sys.path.insert(0, str(ROOT / "experiments" / "flan-nl"))
sys.path.insert(0, str(ROOT / "scripts"))

import g4dpo  # noqa: E402
import g4rl  # noqa: E402
import sft_lora  # noqa: E402
import translate as tr  # noqa: E402

EN = ("the", "house", "river", "green", "walks", "slowly", "and", "city", "old", "light", "water", "sees")
NL = ("het", "huis", "rivier", "groen", "loopt", "langzaam", "en", "stad", "oud", "licht", "water", "ziet")


class FakeTok:
    """tokenizers.Tokenizer's decode over FakeCounter7's vocabulary"""

    def __init__(self, vocab):
        self.vocab = vocab

    def decode(self, ids, skip_special_tokens=True):
        back = {v: k for k, v in self.vocab.items()}
        return " ".join(back[i] for i in ids if not (skip_special_tokens and i < 10))


class FakeCounter7:
    """g4arm.TokenCounter's interface: one id per word (ids from 10 up, in order of first sight), <pad> 0, <bos> 2"""

    def __init__(self):
        self.vocab = {}
        self.tok = FakeTok(self.vocab)

    def ids(self, texts):
        return [[self.vocab.setdefault(w, 10 + len(self.vocab)) for w in t.split()] for t in texts]

    def lengths(self, texts):
        return [len(x) for x in self.ids(texts)]

    def special_id(self, token):
        return {"<pad>": 0, "<bos>": 2}[token]


def sentence(rng, words, n):
    return " ".join(rng.choice(words) for _ in range(n))


# ---------------------------------------------------------------------------------------------------- reward, advantage

def sample(tokens_out=20, ended=True, **flags):
    return {"tokens_out": tokens_out, "ended": ended, "decode_loop": False, "empty": False, "english_left": False,
            **flags}


ANCHOR = {"l0": 20, "qe0": 2.0}
C2 = {"logit_c2": 1.5, "c2_overflow": False}


def test_reward_is_the_clipped_logit_minus_the_qe_penalty_against_the_anchor_and_the_length_penalty():
    t = g4rl.reward7(sample(), 1.0, C2, ANCHOR)          # better QE than the anchor: no penalty, no reward
    assert t == {"reward": 1.5, "fail": [], "logit": 1.5, "qe_pen": 0.0, "len_pen": 0.0, "ratio": 1.0}
    t = g4rl.reward7(sample(), 2.5, {**C2, "logit_c2": 9.0}, ANCHOR)
    assert t["logit"] == 6.0 and t["qe_pen"] == pytest.approx(1.0) and t["reward"] == pytest.approx(5.0)
    assert g4rl.reward7(sample(), 2.0, {**C2, "logit_c2": -8.0}, ANCHOR)["reward"] == -6.0
    # within +-15% of the anchor's length no penalty; beyond it 2 x (|ln ratio| - ln 1.15)
    assert g4rl.reward7(sample(tokens_out=23), 2.0, C2, ANCHOR)["len_pen"] == 0.0
    t = g4rl.reward7(sample(tokens_out=26), 2.0, C2, ANCHOR)
    assert t["len_pen"] == pytest.approx(2 * (math.log(1.3) - math.log(1.15)))
    t = g4rl.reward7(sample(tokens_out=15), 2.0, C2, ANCHOR)
    assert t["len_pen"] == pytest.approx(2 * (abs(math.log(0.75)) - math.log(1.15)))


@pytest.mark.parametrize("change, reason", [
    ({"ended": False}, "not_ended"), ({"decode_loop": True}, "decode_loop"), ({"empty": True}, "empty"),
    ({"english_left": True}, "english_left"), ({"tokens_out": 13}, "length"), ({"tokens_out": 31}, "length"),
    ({"qe": 7.6}, "qe"), ({"c2_overflow": True}, "c2_overflow")])
def test_every_failure_gives_minus_ten_without_classifier_or_qe(change, reason):
    s = sample(**{k: v for k, v in change.items() if k not in ("qe", "c2_overflow")})
    c2 = {**C2, "c2_overflow": change.get("c2_overflow", False)}
    t = g4rl.reward7(s, change.get("qe", 2.0), c2, ANCHOR)
    assert t["fail"] == [reason] and t["reward"] == -10.0
    # the band's ends are inside it: 0.67 and 1.5 of the anchor's length
    assert not g4rl.reward7(sample(tokens_out=30), 2.0, C2, ANCHOR)["fail"]
    assert not g4rl.reward7(sample(tokens_out=134), 2.0, C2, {"l0": 200, "qe0": 2.0})["fail"]


def test_the_advantage_is_dr_grpos_reward_minus_the_group_mean_and_a_flat_group_has_no_signal():
    adv, signal = g4rl.advantages7([1.0, 2.0, 3.0, -10.0, 0.0, 0.5, 1.5, 2.0])
    assert adv.sum() == pytest.approx(0.0) and signal
    assert adv[0] == pytest.approx(1.0 - np.mean([1.0, 2.0, 3.0, -10.0, 0.0, 0.5, 1.5, 2.0]))   # no division by the std
    assert g4rl.advantages7([1.0, 1.04, 1.02])[1] is False and g4rl.advantages7([1.0, 1.05])[1] is True
    with pytest.raises(SystemExit, match="exactly the samples 0..2"):
        g4rl.groups7([{"source_id": "a", "j": 0}, {"source_id": "a", "j": 2}], 3)


# ---------------------------------------------------------------------------------------------------- pool, plan

def test_part_counts_are_205_154_153_of_512():
    assert g4rl.part_counts(512) == {"flan": 205, "novels": 154, "bt": 153}
    assert sum(g4rl.part_counts(37).values()) == 37


def cands(n, part, strata=("a",), docs_per=1, rng=None):
    rng = rng or random.Random(0)
    return [{"id": f"{part}:{k}", "en": f"{sentence(rng, EN, rng.randint(3, 9))} {part} {k}",
             "doc": f"{part}-d{k // docs_per}", "stratum": strata[k % len(strata)], "tokens": 10} for k in range(n)]


def test_ordered_is_equal_per_stratum_and_fixed_by_its_seed():
    rows = cands(50, "bt", strata=("h", "r", "f"))
    a, b = g4rl.ordered(rows, "s"), g4rl.ordered(list(reversed(rows)), "s")
    assert [r["id"] for r in a] == [r["id"] for r in b]
    first = [r["stratum"] for r in a[:9]]
    assert sorted(first) == ["f"] * 3 + ["h"] * 3 + ["r"] * 3
    assert [r["id"] for r in g4rl.ordered(rows, "t")] != [r["id"] for r in a]


def test_candidate_filter_keeps_short_new_texts_of_allowed_documents():
    import select_passages as sp

    counter = FakeCounter7()
    rows = [{"id": "a", "en": "the house", "doc": "d1"}, {"id": "b", "en": " ".join(["the"] * 300), "doc": "d2"},
            {"id": "c", "en": "the river", "doc": "gone"}, {"id": "d", "en": "  ", "doc": "d3"},
            {"id": "e", "en": "the house", "doc": "d4"},
            {"id": "f", "en": "a b c d e f g h i j k l m n", "doc": "d5"}]
    blocked = sp.ngrams("a b c d e f g h i j k l m")
    keep, drops = g4rl.candidate_filter(rows, counter, blocked, {"gone"})
    assert [r["id"] for r in keep] == ["a"] and keep[0]["tokens"] == 2
    assert drops == {"empty": 1, "tokens": 1, "ngram": 1, "excluded_doc": 1, "duplicate": 1}


def test_draw_pool_takes_rlval_first_and_keeps_its_documents_out_of_the_pool():
    data = {"flan": cands(200, "flan"), "novels": cands(200, "novels", docs_per=4), "bt": cands(200, "bt", ("x", "y"))}
    rows = g4rl.draw_pool(data, iterations=3, per_iter=20, rlval=5, reserve=0.1)
    counts = g4rl.part_counts(20)
    for part in g4rl.PARTS7:
        val = [r for r in rows if r["part"] == part and r["role"] == "rlval"]
        pool = [r for r in rows if r["part"] == part and r["role"] == "pool"]
        assert len(val) == math.ceil(5 * 1.1) and len(pool) == math.ceil(3 * counts[part] * 1.1)
        assert not {r["doc"] for r in val} & {r["doc"] for r in pool}
        assert [r["rank"] for r in pool] == list(range(len(pool)))
    assert rows == g4rl.draw_pool(data, iterations=3, per_iter=20, rlval=5, reserve=0.1)
    with pytest.raises(SystemExit, match="flan has"):
        g4rl.draw_pool({**data, "flan": cands(20, "flan")}, iterations=3, per_iter=20, rlval=5)


def anchor(ok=True, **kw):
    return {"ended": ok, "decode_loop": False, "empty": False, "english_left": False, "l0": 20, "qe0": 2.0,
            "logit_c2_0": 0.1, "hypothesis0": "x", **kw}


def test_plan_drops_sources_whose_anchor_failed_and_fills_the_iterations_in_rank_order():
    data = {"flan": cands(100, "flan"), "novels": cands(100, "novels"), "bt": cands(100, "bt")}
    pool = g4rl.draw_pool(data, iterations=2, per_iter=10, rlval=3, reserve=0.5)
    anchors = {r["id"]: anchor() for r in pool}
    first_flan = next(r for r in pool if r["part"] == "flan" and r["role"] == "pool" and r["rank"] == 0)
    anchors[first_flan["id"]] = anchor(decode_loop=True)
    plan, report = g4rl.plan_sources(pool, anchors, iterations=2, per_iter=10, rlval=3)
    assert first_flan["id"] not in {r["id"] for r in plan}
    assert report["dropped"]["flan/pool"]["decode_loop"] == 1
    counts = g4rl.part_counts(10)
    for i in (0, 1):
        rows = [r for r in plan if r["iteration"] == i]
        assert {p: sum(r["part"] == p for r in rows) for p in g4rl.PARTS7} == counts
        assert sorted(r["position"] for r in rows) == list(range(10))
    assert sum(r["iteration"] == -1 for r in plan) == 9
    flan0 = [r["id"] for r in plan if r["part"] == "flan" and r["iteration"] == 0]
    want = [r["id"] for r in sorted(pool, key=lambda r: r["rank"]) if r["part"] == "flan" and r["role"] == "pool"][1:5]
    assert sorted(flan0) == sorted(want)
    with pytest.raises(SystemExit, match="without an anchor"):
        g4rl.plan_sources(pool, {}, iterations=2, per_iter=10, rlval=3)
    bad = {k: anchor(ok=False) for k in anchors}
    with pytest.raises(SystemExit, match="good anchor"):
        g4rl.plan_sources(pool, bad, iterations=2, per_iter=10, rlval=3)


def test_assign_workers_balances_tokens():
    rows = [{"id": str(k), "tokens": t} for k, t in enumerate([100, 90, 50, 40, 30, 20, 10, 5])]
    out = g4rl.assign_workers7(rows, 2)
    loads = [sum(r["tokens"] for r in out[k]) for k in (0, 1)]
    assert abs(loads[0] - loads[1]) <= 10 and sum(len(v) for v in out.values()) == 8


# ---------------------------------------------------------------------------------------------------- the bucket protocol

def write_samples(root, rows, shard_size=8):
    """Sample shards as translate.py --copies writes them, with DONE"""
    by = {}
    for r in rows:
        by.setdefault(r["worker"], []).append({k: v for k, v in r.items() if k != "worker"})
    for w, part in by.items():
        d = Path(root) / w
        d.mkdir(parents=True)
        digests = {}
        for k in range(0, len(part), shard_size):
            name = f"shard-{k // shard_size:05d}.parquet"
            pq.write_table(pa.Table.from_pylist(part[k:k + shard_size]), d / name)
            digests[name] = (len(part[k:k + shard_size]), hashlib.sha256((d / name).read_bytes()).hexdigest())
        (d / "DONE").write_text(json.dumps(tr.done_marker(list(digests), digests)))


def write_scores(root, samples, qe, c2, judges=("w0", "w2")):
    """Score shards of both roles: MetricX split over two judges by uid, C2 on one"""
    for role, names in (("metricx", judges), ("c2", ("w1",))):
        for m, judge in enumerate(names):
            d = Path(root) / role / judge
            d.mkdir(parents=True)
            per = {}
            for s in samples:
                if role == "metricx" and int(hashlib.md5(s["uid"].encode()).hexdigest(), 16) % len(names) != m:
                    continue
                k = s["worker"][1:]
                row = {"uid": s["uid"], "qe": qe[s["uid"]]} if role == "metricx" else {"uid": s["uid"], **c2[s["uid"]]}
                per.setdefault(k, []).append(row)
            for k, rows in per.items():
                pq.write_table(pa.Table.from_pylist(rows), d / f"shard-{k}-0.parquet")
                (d / f"DONE-{k}").write_text(json.dumps({"rows": len(rows)}))


def make_iteration(tmp_path, counter, plan, iteration, g=4, seed=0):
    """Samples of the plan's iteration (the bucket protocol), their scores; returns (samples, qe, c2)"""
    rng = random.Random(seed)
    samples, qe, c2 = [], {}, {}
    sources = [r for r in plan if r["iteration"] == iteration]
    for n, src in enumerate(sorted(sources, key=lambda r: r["id"])):
        for j in range(g):
            hyp = sentence(rng, NL, rng.randint(15, 25))
            ids = counter.ids([hyp])[0] + [1]                   # the stop token (id 1 < 10: a special token)
            u = g4rl.uid(src["id"], j)
            samples.append({"uid": u, "source_id": src["id"], "j": j, "source": src["en"], "hypothesis": hyp,
                            "token_ids": ids, "sampler_logp": [-rng.uniform(0.01, 3.0) for _ in ids], "ended": True,
                            "decode_loop": False, "english_left": False, "empty": False, "tokens_out": len(ids),
                            "tokens_in": len(counter.ids([g4rl_prompt(src["en"])])[0]), "worker": f"w{n % 2}"})
            qe[u] = rng.uniform(1.0, 4.0)
            logit = rng.uniform(-3, 3)
            c2[u] = {"p_c2": 1 / (1 + math.exp(-logit)), "logit_c2": logit, "c2_tokens": len(ids), "c2_overflow": False}
    root = tmp_path / f"iter{iteration}"
    write_samples(root / "samples", samples)
    write_scores(root / "scores", samples, qe, c2)
    return samples, qe, c2


def g4rl_prompt(en):
    import g4lora

    return g4lora.prompt_text(en)


def small_plan(n=6, iteration=0, l0=20):
    rng = random.Random(1)
    return [{"id": f"s{k}", "part": ("flan", "novels", "bt")[k % 3], "stratum": "", "doc": f"d{k}",
             "en": sentence(rng, EN, rng.randint(4, 8)), "tokens": 8, "iteration": iteration, "position": k,
             "qe0": 2.5, "l0": l0, "logit_c2_0": 0.0, "hypothesis0": "x"} for k in range(n)]


def test_read_samples_checks_the_marker_the_digests_and_the_uids(tmp_path):
    counter = FakeCounter7()
    plan = small_plan()
    samples, _, _ = make_iteration(tmp_path, counter, plan, 0)
    got = g4rl.read_samples(tmp_path / "iter0" / "samples")
    assert sorted(r["uid"] for r in got) == sorted(s["uid"] for s in samples)
    shard = next((tmp_path / "iter0" / "samples" / "w0").glob("shard-*.parquet"))
    data = shard.read_bytes()
    shard.write_bytes(data + b"x")
    with pytest.raises(SystemExit, match="sha256"):
        g4rl.read_samples(tmp_path / "iter0" / "samples")
    shard.write_bytes(data)
    (tmp_path / "iter0" / "samples" / "w1" / "DONE").unlink()
    with pytest.raises(SystemExit, match="no DONE"):
        g4rl.read_samples(tmp_path / "iter0" / "samples")


def test_read_scores_needs_every_samples_score_once_and_the_markers(tmp_path):
    counter = FakeCounter7()
    samples, _, _ = make_iteration(tmp_path, counter, small_plan(), 0)
    uids = {s["uid"] for s in samples}
    scores = g4rl.read_scores(tmp_path / "iter0" / "scores", uids)
    assert set(scores["metricx"]) == uids and set(scores["c2"]) == uids
    with pytest.raises(SystemExit, match="1 scores of no sample"):
        g4rl.read_scores(tmp_path / "iter0" / "scores", uids - {sorted(uids)[0]})
    with pytest.raises(SystemExit, match="1 samples without a score"):
        g4rl.read_scores(tmp_path / "iter0" / "scores", uids | {"s9#0"})
    marker = next((tmp_path / "iter0" / "scores" / "c2" / "w1").glob("DONE-*"))
    marker.write_text("1")
    with pytest.raises(SystemExit, match="markers"):
        g4rl.read_scores(tmp_path / "iter0" / "scores", uids)
    marker.unlink()
    with pytest.raises(SystemExit, match="markers"):
        g4rl.read_scores(tmp_path / "iter0" / "scores", uids)


# ---------------------------------------------------------------------------------------------------- packing

def test_row_arrays_put_each_tokens_advantage_and_logp_beside_its_target():
    a = ([2, 11, 12, 13], [21, 22, 1], 0.5, [-0.1, -0.2, -0.3])
    b = ([2, 14, 15], [23, 1], -1.5, [-1.0, -2.0])
    row = g4rl.row_arrays7([a, b], 16)
    want = g4dpo.row_arrays([(a[0], a[1]), (b[0], b[1])], 16)
    for k in g4dpo.KEYS:
        assert np.array_equal(row[k], want[k]), k
    on = row["targets_segmentation"] != 0
    # every target token of a sample carries its advantage and the sampler's log-probability of that token
    assert row["targets"][on].tolist() == [21, 22, 1, 23, 1]
    assert row["adv"][on].tolist() == [0.5, 0.5, 0.5, -1.5, -1.5]
    assert row["sampler_logp"][on].tolist() == pytest.approx([-0.1, -0.2, -0.3, -1.0, -2.0])
    assert not row["adv"][~on].any() and not row["sampler_logp"][~on].any()
    with pytest.raises(ValueError, match="log-probabilities"):
        g4rl.row_arrays7([([2, 3], [4, 5], 1.0, [-1.0])], 8)


def test_a_completion_token_that_is_the_pad_or_bos_id_is_masked_as_maxtexts_shift_does():
    row = g4rl.row_arrays7([([2, 11], [21, 0, 22, 2, 1], 1.0, [-0.1] * 5)], 16)
    on = row["targets_segmentation"] != 0
    assert row["targets"][on].tolist() == [21, 22, 1]


def test_pack_rows_writes_and_reads_back_with_the_batches_and_their_local_parts(tmp_path):
    rng = random.Random(3)
    units = [{"uid": f"u{k}", "prompt": [2] + [rng.randint(10, 99) for _ in range(rng.randint(3, 6))],
              "completion": [rng.randint(10, 99) for _ in range(rng.randint(2, 6))] + [1], "adv": rng.uniform(-2, 2)}
             for k in range(30)]
    for u in units:
        u["sampler_logp"] = [-rng.uniform(0, 2) for _ in u["completion"]]
    rows = g4rl.pack_rows7(units, 32, 4)
    g4rl.write_rows7(rows, tmp_path / "train.parquet")
    arrays, batch = g4rl.read_packed7(tmp_path / "train.parquet")
    assert arrays["adv"].dtype == np.float32 and arrays["inputs"].dtype == np.int32 and arrays["adv"].shape[1] == 32
    info = g4rl.packed_info7(tmp_path / "train.parquet", 4, 32)
    assert info["samples"] == 30 and info["loss_tokens"] == sum(len(u["completion"]) for u in units)
    assert info["rows"] % 4 == 0 and info["batches"] == info["rows"] // 4
    # adv summed over each sample's target tokens = its advantage x its completion length
    total = float(np.sum(arrays["adv"][arrays["targets_segmentation"] != 0]))
    assert total == pytest.approx(sum(u["adv"] * len(u["completion"]) for u in units), rel=1e-5)
    seq = 32
    passes = {k: np.full((info["rows"], seq), v, np.float32) for k, v in (("ref_logp", -1.0), ("old_logp", -2.0))}
    parts = list(g4rl.local_batches7(arrays, 1, 2, 4, info["batches"], padding=1, passes=passes, row_ids=True))
    assert len(parts) == info["batches"] + 1
    assert np.array_equal(parts[0]["inputs"], arrays["inputs"][2:4]) and parts[0][g4rl.ROW_KEY].tolist() == [2, 3]
    assert parts[0]["old_logp"].shape == (2, seq) and not parts[-1]["adv"].any() and not parts[-1]["ref_logp"].any()
    with pytest.raises(SystemExit, match="max_target_length"):
        g4rl.packed_info7(tmp_path / "train.parquet", 4, 64)


def test_tis_weights_are_the_ratio_old_over_sampler_capped_at_the_threshold():
    old = np.array([[-1.0, -1.0, -3.0]], np.float32)
    sampler = np.array([[-1.0, -2.0, -1.0]], np.float32)
    w = g4rl.tis_weights(old, sampler, np.array([[True, True, False]]), 2.0)
    assert w[0].tolist() == pytest.approx([1.0, 2.0, 0.0])


# ---------------------------------------------------------------------------------------------------- the loss in numpy

def test_grpo_terms_at_rho_1_and_the_reference_is_minus_the_token_mean_of_a():
    rng = np.random.default_rng(0)
    mask = rng.random((3, 10)) < 0.6
    logp = np.where(mask, -rng.uniform(0.1, 3, (3, 10)), 0.0).astype(np.float32)
    adv = np.where(mask, rng.normal(size=(3, 1)), 0.0).astype(np.float32)
    opts = g4rl.grpo_spec()
    loss, sums = g4rl.grpo_terms(logp, logp, logp, adv, mask, opts)
    assert loss == pytest.approx(-adv[mask].mean(), rel=1e-6)
    assert sums["kl_sum"] == 0.0 and sums["clip_sum"] == 0.0 and sums["tokens"] == mask.sum()
    # off-policy: the clip of Tunix's GRPO (symmetric) and the k2 KL weighted by the ratio
    old = logp - 0.5 * mask
    ref = logp + 0.3 * mask
    loss, sums = g4rl.grpo_terms(logp, old, ref, adv, mask, opts)
    rho = np.exp(0.5)
    pg = np.maximum(-adv * rho, -adv * np.clip(rho, 0.8, 1.2))
    want = (pg + 0.04 * rho * 0.5 * 0.09)[mask].mean()
    assert loss == pytest.approx(want, rel=1e-5)
    assert sums["clip_sum"] == pytest.approx(float((adv[mask] > 0).sum()))
    # TIS multiplies the policy term only
    tis = np.where(mask, 0.5, 0.0).astype(np.float32)
    loss_t, _ = g4rl.grpo_terms(logp, old, ref, adv, mask, opts, tis=tis)
    assert loss_t == pytest.approx((0.5 * pg + 0.04 * rho * 0.5 * 0.09)[mask].mean(), rel=1e-5)
    with pytest.raises(ValueError, match="unknown"):
        g4rl.grpo_spec(gamma=1)
    with pytest.raises(ValueError, match="epsilon"):
        g4rl.grpo_spec(epsilon=1.5)


def test_pass_record_measures_k1_and_the_sampler_gap_on_the_target_tokens():
    mask = np.array([[True, True, False]])
    ref = np.array([[-1.0, -2.0, 0.0]], np.float32)
    old = np.array([[-0.5, -2.5, 0.0]], np.float32)
    sampler = np.array([[-0.4, -2.7, 9.0]], np.float32)
    r = g4rl.pass_record(ref, old, sampler, mask)
    assert r["k1_old_ref"] == pytest.approx(0.0) and r["old_ref_abs_max"] == pytest.approx(0.5)
    assert r["sampler_gap"] == pytest.approx(0.15) and r["kl_sampler_old"] == pytest.approx(-0.05)
    assert r["old_is_ref"] is False and g4rl.pass_record(ref, ref, sampler, mask)["old_is_ref"] is True


def test_step_record_divides_by_the_tokens():
    sums = {k: 2.0 for k in g4rl.SUMS7} | {"tokens": 4.0, "log_ratio_abs_max": 0.3}
    r = g4rl.step_record7(5, sums)
    assert r["step"] == 5 and r["loss"] == 0.5 and r["log_ratio_abs_max"] == 0.3 and r["clip_frac"] == 0.5


# ---------------------------------------------------------------------------------------------------- iter7

def test_iter7_turns_samples_and_scores_into_rewards_advantages_and_a_packed_file(tmp_path, monkeypatch):
    import g4arm

    counter = FakeCounter7()
    plan = small_plan(n=6) + small_plan(n=3, iteration=1)
    for r in plan[6:]:
        r["id"] = "t" + r["id"]
    pq.write_table(pa.Table.from_pylist(plan), tmp_path / "plan.parquet")
    samples, qe, c2 = make_iteration(tmp_path, counter, plan, 0, g=8)
    # one group without signal: all eight fail
    flat = [s for s in samples if s["source_id"] == "s1"]
    monkeypatch.setattr(g4arm, "TokenCounter", lambda path: counter)
    args = ns(plan=str(tmp_path / "plan.parquet"), iteration=0, samples=str(tmp_path / "iter0" / "samples"),
              scores=str(tmp_path / "iter0" / "scores"), tokenizer="x", seq=256, rows=4, out=str(tmp_path / "out"))
    for s in flat:
        qe[s["uid"]] = 8.0
    import shutil

    shutil.rmtree(tmp_path / "iter0" / "scores")
    write_scores(tmp_path / "iter0" / "scores", samples, qe, c2)
    report = g4rl.iter7(args, counter)
    assert report["sources"] == 6 and report["samples"] == 48 and report["groups_without_signal"] == 1
    assert report["fails"]["qe"] == 8 and report["fail_share"] == pytest.approx(8 / 48)
    rows = pq.read_table(tmp_path / "out" / "samples.parquet").to_pylist()
    by_source = {}
    for r in rows:
        by_source.setdefault(r["source_id"], []).append(r)
    for sid, group in by_source.items():
        assert sum(r["adv"] for r in group) == pytest.approx(0.0, abs=1e-9)
        assert all(r["signal"] == (sid != "s1") for r in group)
    info = g4rl.packed_info7(tmp_path / "out" / "train.parquet", 4, 256)
    assert info["samples"] == 40 and report["train"]["samples"] == 40
    arrays, _ = g4rl.read_packed7(tmp_path / "out" / "train.parquet")
    on = arrays["targets_segmentation"] != 0
    adv_of = {r["uid"]: r["adv"] for r in rows}
    table = pq.read_table(tmp_path / "out" / "train.parquet", columns=["uids"]).column("uids").to_pylist()
    for r, uids in enumerate(table):
        # per row, the sampled ids follow each other in the targets with their own advantage and log-probabilities
        want_t, want_a, want_l = [], [], []
        for u in uids:
            s = next(x for x in samples if x["uid"] == u)
            want_t += s["token_ids"]
            want_a += [adv_of[u]] * len(s["token_ids"])
            want_l += s["sampler_logp"]
        assert arrays["targets"][r][on[r]].tolist() == want_t
        assert arrays["adv"][r][on[r]].tolist() == pytest.approx(want_a, rel=1e-6)
        assert arrays["sampler_logp"][r][on[r]].tolist() == pytest.approx(want_l, rel=1e-6)
    # a sample whose token ids are not its text stops the iteration
    bad_root = tmp_path / "bad"
    shutil.copytree(tmp_path / "iter0", bad_root)
    shard = sorted((bad_root / "samples" / "w0").glob("*.parquet"))[0]
    rows0 = pq.read_table(shard).to_pylist()
    rows0[0]["hypothesis"] = "iets anders"
    pq.write_table(pa.Table.from_pylist(rows0), shard)
    done = json.loads((bad_root / "samples" / "w0" / "DONE").read_text())
    done["sha256"][shard.name] = hashlib.sha256(shard.read_bytes()).hexdigest()
    (bad_root / "samples" / "w0" / "DONE").write_text(json.dumps(done))
    with pytest.raises(SystemExit, match="do not decode"):
        g4rl.iter7(ns(**{**vars(args), "samples": str(bad_root / "samples"), "out": str(tmp_path / "o2")}), counter)
    # samples of another iteration's sources are refused
    with pytest.raises(SystemExit, match="the plan has 3"):
        g4rl.iter7(ns(**{**vars(args), "iteration": 1, "out": str(tmp_path / "o3")}), counter)


# ---------------------------------------------------------------------------------------------------- stop rules

def val(fail_share=0.0, qe_delta=0.0, ratio=1.0):
    return {"fail_share": fail_share, "qe_delta": qe_delta, "ratio_median": ratio}


def test_stop_rules_with_s4s_band_of_yeb_and_s2_twice():
    assert g4rl.stop_rules7(val(), 0.05, 0.1)["stop"] == []
    assert g4rl.stop_rules7(val(fail_share=8 / 384), 0.0, 0.0)["stop"] == ["S1"]
    assert g4rl.stop_rules7(val(fail_share=7 / 384), 0.0, 0.0)["stop"] == []
    first = g4rl.stop_rules7(val(qe_delta=0.2), 0.0, 0.0)
    assert first["stop"] == [] and first["rules"]["S2"]["value"] == 0.2
    assert g4rl.stop_rules7(val(qe_delta=0.16), 0.0, 0.0, first)["stop"] == ["S2"]
    assert g4rl.stop_rules7(val(qe_delta=0.16), 0.0, 0.0, g4rl.stop_rules7(val(), 0.0, 0.0))["stop"] == []
    assert g4rl.stop_rules7(val(), 0.0, 0.31)["stop"] == ["S3"]
    for ratio, hit in ((0.85, False), (0.84, True), (1.15, False), (1.16, True), (0.89, False), (1.12, False)):
        assert (g4rl.stop_rules7(val(ratio=ratio), 0.0, 0.0)["stop"] == ["S4"]) is hit, ratio
    assert g4rl.stop_rules7(val(), 0.11, 0.0)["stop"] == ["S6"]
    assert "S5" not in g4rl.stop_rules7(val(), 0.0, 0.0)["rules"]


def test_validation_measures_against_the_anchor(tmp_path):
    counter = FakeCounter7()
    plan = small_plan(n=4, iteration=-1, l0=10)
    samples, qe, c2 = make_iteration(tmp_path, counter, plan, -1, g=1)
    got = g4rl.read_samples(tmp_path / "iter-1" / "samples")
    scores = g4rl.read_scores(tmp_path / "iter-1" / "scores", {s["uid"] for s in got})
    v = g4rl.validation_measures(got, scores, plan)
    assert v["sources"] == 4 and v["fail_share"] == 0.0
    assert v["qe_delta"] == pytest.approx(np.mean(list(qe.values())) - 2.5)
    assert v["ratio_median"] == pytest.approx(np.median([s["tokens_out"] / 10 for s in samples]))


# ---------------------------------------------------------------------------------------------------- sft_lora --grpo

def test_sft_lora_refuses_inconsistent_grpo_arguments_before_jax(tmp_path):
    spec = tmp_path / "grpo.json"
    spec.write_text(json.dumps(g4rl.GRPO_DEFAULTS))
    train = tmp_path / "train.parquet"
    train.write_bytes(b"x")
    base = ["--out", str(tmp_path / "o")]
    for argv, msg in (([*base, "--grpo-train", str(train)], "belong to --grpo"),
                      ([*base, "--passes-only"], "belong to --grpo"),
                      ([*base, "--grpo", str(spec)], "needs --grpo-train"),
                      ([*base, "--grpo", str(spec), "--grpo-train", str(train), "--dpo", str(spec)], "no --dpo"),
                      ([*base, "--grpo", str(spec), "--grpo-train", str(train), "--save-steps", "4"], "no --dpo"),
                      ([*base, "--grpo", str(spec), "--grpo-train", str(train), "--passes-only", "--eval", "a=b",
                        "--eval-every", "1"], "batches alone"),
                      ([*base, "--grpo", str(spec), "--grpo-train", str(train), "--passes-in", "nope"], "no such file")):
        with pytest.raises(SystemExit, match=msg):
            sft_lora.main(argv)
    spec.write_text(json.dumps({"beta": -1}))
    with pytest.raises(SystemExit, match="beta"):
        sft_lora.main([*base, "--grpo", str(spec), "--grpo-train", str(train)])
    # --base-adapter goes with --grpo too (the start G4-4 of the CPU proof's (d)); alone still refused
    args = ns(dpo=None, grpo=str(spec), pref_train=None, pref_eval=[], base_adapter="G4-4=x", ref_only=False, ref_in=None)
    assert sft_lora.dpo_args(args, {}) == (None, {})
    with pytest.raises(SystemExit, match="belong to --dpo"):
        sft_lora.dpo_args(ns(**{**vars(args), "grpo": None}), {})


def test_passes_identity_names_the_adapters_that_make_the_model(tmp_path):
    f = tmp_path / "train.parquet"
    f.write_bytes(b"rows")
    a = tmp_path / "adapter.npz"
    a.write_bytes(b"adapter")
    config = ns(model_name="gemma4-31b", max_target_length=1024, lora=ns(lora_rank=4, lora_alpha=2.0))
    one = sft_lora.passes_identity(str(f), 4, 4, None, None, 0, None, config, "re", {"count": 4})
    two = sft_lora.passes_identity(str(f), 4, 4, None, None, 4, str(a), config, "re", {"count": 4})
    three = sft_lora.passes_identity(str(f), 4, 4, None, str(a), 0, None, config, "re", {"count": 4})
    assert one["resume"] is None and two["resume"]["step"] == 4 and three["init_adapter"] == two["resume"]["adapter_sha256"]
    assert one != two != three


def test_passes_round_trip_and_refuse_another_runs(tmp_path):
    passes = {k: np.full((8, 16), -1.0, np.float32) for k in g4rl.PASS_KEYS}
    g4rl.write_passes(tmp_path, passes, {"a": 1}, {"tokens": 3})
    got, meta = g4rl.read_passes(tmp_path / g4rl.PASSES_FILE, {"a": 1}, 8, 16)
    assert np.array_equal(got["old_logp"], passes["old_logp"]) and meta["record"] == {"tokens": 3}
    with pytest.raises(SystemExit, match="differ in"):
        g4rl.read_passes(tmp_path / g4rl.PASSES_FILE, {"a": 2}, 8, 16)
    with pytest.raises(SystemExit, match="not finite float32"):
        g4rl.read_passes(tmp_path / g4rl.PASSES_FILE, {"a": 1}, 4, 16)


# ---------------------------------------------------------------------------------------------------- translate.py

class LogpDevice:
    """decode_all's device with log-probabilities: prompt [a, n] continues with n tokens 1000 a + k, then <turn|>
    (106), then noise; the log-probability of a token is -token / 1e4"""

    def __init__(self, slots):
        self.programs, self.positions, self.prefills = [None] * slots, [0] * slots, 0

    @staticmethod
    def completion(prompt):
        return [1000 * prompt[0] + k for k in range(prompt[1])] + [106]

    def prefill(self, tokens):
        self.prefills += 1
        first = self.completion(tokens)[0]
        return list(tokens), (np.int32(first), np.float32(-first / 1e4))

    def insert(self, prefix, slot):
        self.programs[slot], self.positions[slot] = self.completion(prefix), 1

    def generate(self):
        out = []
        for s, program in enumerate(self.programs):
            p = self.positions[s]
            out.append(program[p] if program is not None and p < len(program) else 7)
            self.positions[s] += 1
        out = np.asarray(out, np.int32)
        return out, (-out / 1e4).astype(np.float32)

    @staticmethod
    def fetch(arrays):
        return np.asarray(arrays)


@pytest.mark.parametrize("slots, sync_every", [(1, 1), (3, 4), (8, 16)])
def test_decode_all_with_logprobs_keeps_the_stop_token_and_each_tokens_logprob(slots, sync_every):
    rng = np.random.default_rng(slots)
    prompts = [[a, int(rng.integers(0, 20))] for a in range(1, 15)] + [[99, 60]]
    out = tr.decode_all(prompts, slots, LogpDevice(slots), max_decode=30, sync_every=sync_every, eos_ids=[1, 106, 50],
                        logprobs=True)
    for prompt, (tokens, ended, logps) in zip(prompts, out):
        want = LogpDevice.completion(prompt)
        if len(want) <= 30:
            assert (tokens, ended) == (want, True)
        else:
            assert (tokens, ended) == (want[:30], False)
        assert logps == pytest.approx([-t / 1e4 for t in tokens])


class WordTokenizer:
    """A tokenizer whose ids are words; 106 the stop token, special"""

    words = {11: "een", 12: "twee", 13: "drie", 106: "<turn|>"}

    def decode(self, ids, skip_special_tokens=True):
        return " " + " ".join(self.words[i] for i in ids if not (skip_special_tokens and i == 106)) + " "


class WordCodec:
    target = "nl"

    @staticmethod
    def encode(text):
        return [2, 5, 5]


class ThreeWordDevice(LogpDevice):
    @staticmethod
    def completion(prompt):
        return [11, 12, 13, 106]


def test_translate_shard_with_logprobs_gives_token_ids_that_decode_to_the_text():
    shapes = {"slots": 2, "max_decode": 16}
    rows, _ = tr.translate_shard(ThreeWordDevice(2), WordCodec(), WordTokenizer(), ["a#0", "a#1", "b#0"], ["x"] * 3,
                                 shapes, 2, eos_ids=[1, 106, 50], logprobs=True)
    tok = WordTokenizer()
    for ids, text, ended, logps, n in zip(rows["token_ids"], rows["text_nl"], rows["ended"], rows["sampler_logp"],
                                          rows["tokens_out"]):
        assert ids == [11, 12, 13, 106] and ended and len(logps) == len(ids) == n
        assert tok.decode(ids[:-1] if ended else ids, skip_special_tokens=True).strip() == text == "een twee drie"


def test_copies_and_protocol_rows_make_the_bucket_protocols_sample_shard():
    uids, texts = tr.copies_of(["s1", 7], ["the house", "the river"], 3)
    assert uids == ["s1#0", "s1#1", "s1#2", "7#0", "7#1", "7#2"] and texts == ["the house"] * 3 + ["the river"] * 3
    rows = {"id": ["s1#0", "7#2"], "text_nl": ["het huis", " "], "ended": [True, False], "tokens_in": [9, 9],
            "tokens_out": [3, 2], "token_ids": [[11, 12, 106], [5, 5]], "sampler_logp": [[-0.1, -0.2, -0.3], [-1, -2]]}
    out = tr.protocol_rows(rows, dict(zip(uids, texts)))
    assert tuple(out) == tr.PROTOCOL_COLUMNS and set(g4rl.SAMPLE_COLUMNS) <= set(out)
    assert out["source_id"] == ["s1", "7"] and out["j"] == [0, 2] and out["source"] == ["the house", "the river"]
    assert out["empty"] == [False, True] and out["decode_loop"] == [False, False]
    assert out["english_left"] == [False, False]
    marker = tr.done_marker(["b", "a"], {"a": (3, "x"), "b": (2, "y")})
    assert marker == {"rows": 5, "shards": 2, "sha256": {"a": "x", "b": "y"}}
    with pytest.raises(RuntimeError, match="no digest"):
        tr.done_marker(["c"], {})


def test_translate_refuses_greedy_copies():
    with pytest.raises(SystemExit):
        tr.main(["--segments", "s.parquet", "--out", "gs://x/y", "--tokenizer", "t", "--length-class", "medium",
                 "--copies", "8"])


def test_seeds_are_fixed_and_differ_per_iteration_and_worker():
    assert g4rl.sample_seed(3, 1) == g4rl.sample_seed(3, 1)
    assert len({g4rl.sample_seed(i, k) for i in range(4) for k in range(4)}) == 16
    assert 0 <= g4rl.sample_seed(0, 0) < 2 ** 63


GEMMA4 = Path.home() / ".cache/huggingface/hub/models--google--gemma-4-31B-it/snapshots/" \
    "842da3794eaa0b77d5f08bae87a17459d91ff475"


def test_gemma4s_token_ids_decode_to_the_text_in_both_tokenizer_libraries():
    """translate.py decodes with transformers' tokenizer, iter7 checks with tokenizers': the same text"""
    tokenizers = pytest.importorskip("tokenizers")
    if not (GEMMA4 / "tokenizer.json").is_file():
        pytest.skip("no Gemma 4 tokenizer here")
    tok = tokenizers.Tokenizer.from_file(str(GEMMA4 / "tokenizer.json"))
    counter = ns(tok=tok)
    for text in ("Het huis aan de rivier is oud, en het licht valt er zacht.", "Zij zei: “Nee!” — en liep weg.\n\nEinde."):
        ids = tok.encode(text, add_special_tokens=False).ids + [tok.token_to_id("<turn|>")]
        hyp = tok.decode(ids[:-1], skip_special_tokens=True).strip()
        assert hyp == text
        assert g4rl.decoded_ok(counter, {"token_ids": ids, "ended": True, "hypothesis": hyp})
        try:
            from transformers import AutoTokenizer
        except ImportError:
            continue
        hf = AutoTokenizer.from_pretrained(str(GEMMA4))
        assert hf.decode(ids[:-1], skip_special_tokens=True).strip() == hyp
