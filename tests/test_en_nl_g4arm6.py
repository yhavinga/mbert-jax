"""Arm G4-6 (draft amendment G4-6, preference training): g4dpo.py's pure parts (packing pairs into MaxText's SFT rows,
the per-process batches, the loss's arithmetic) and g4arm.py's cands6, build6, smoke6, pref_measure, pairtexts6 and
pairclass6 on small synthetic inputs (no tokenizer, no model, no TPU); sft_lora.py's --dpo argument checks. The JAX
parts of g4dpo.py are in test_en_nl_g4dpo_jax.py."""

import json
import math
import random

import numpy as np
import pyarrow.parquet as pq
import pytest

# test_en_nl_g4arm4 puts experiments/en-nl-translator on the path; its world4 fixture is G4-4's inputs
import test_en_nl_g4arm4 as t4

g4arm = t4.g4arm
ns, write, run_build4 = t4.ns, t4.write, t4.run_build4
world, world4 = t4.world, t4.world4

import g4dpo  # noqa: E402
import sft_lora  # noqa: E402

OPTS = g4dpo.spec()


# ---------------------------------------------------------------------------------------------------- packing

def test_pack_pairs_keeps_a_pair_in_one_row_first_fit_and_opens_a_batch_when_none_fits():
    # rows of 10 tokens, 2 rows per batch: (4+4) row 0, (3+3) row 1, (1+1) row 0 (room 2: full), (3+2) fits no row
    # (row 1 has room 4) -> batch 1, row 0; (2+2) into batch 1's row 0 (room 5)
    batches = g4dpo.pack_pairs([(4, 4), (3, 3), (1, 1), (3, 2), (2, 2)], seq=10, rows=2)
    assert batches == [[[0, 2], [1]], [[3, 4], []]]
    with pytest.raises(ValueError, match="more than a row"):
        g4dpo.pack_pairs([(6, 5)], seq=10, rows=2)


def test_pack_pairs_holds_the_segment_cap():
    batches = g4dpo.pack_pairs([(1, 1)] * 5, seq=100, rows=2, max_segments=4)
    assert batches == [[[0, 1], [2, 3]], [[4], []]]


def test_row_arrays_are_maxtexts_sft_row_shifted_as_shiftdata():
    prompt, chosen, rejected = [2, 11, 12], [21, 22], [31]
    row = g4dpo.row_arrays([(prompt, chosen), (prompt, rejected)], seq=12, pad_id=0, bos_id=2)
    assert row["inputs"].tolist() == [2, 11, 12, 21, 22, 2, 11, 12, 31, 0, 0, 0]
    assert row["inputs_segmentation"].tolist() == [1, 1, 1, 1, 1, 2, 2, 2, 2, 0, 0, 0]
    assert row["inputs_position"].tolist() == [0, 1, 2, 3, 4, 0, 1, 2, 3, 0, 0, 0]
    # the target of a place is the next token; only the completion's tokens count, each in its own segment
    assert row["targets"].tolist() == [0, 0, 21, 22, 0, 0, 0, 31, 0, 0, 0, 0]
    assert row["targets_segmentation"].tolist() == [0, 0, 1, 1, 0, 0, 0, 2, 0, 0, 0, 0]
    assert row["targets_position"].tolist() == row["inputs_position"].tolist()
    assert all(v.dtype == np.int32 and v.shape == (12,) for v in row.values())
    # the unshifted row is the packing's: the targets' segmentation that of the inputs
    raw = g4dpo.row_unshifted([(prompt, chosen)], seq=6, pad_id=0)
    assert raw["targets"].tolist() == [0, 0, 0, 21, 22, 0] and raw["targets_segmentation"].tolist() == [1] * 5 + [0]
    with pytest.raises(ValueError, match="more than"):
        g4dpo.row_arrays([(prompt, chosen)], seq=4)


def toy_pairs(n, seed=0):
    rng = random.Random(seed)
    out = []
    for k in range(n):
        prompt = [2] + [rng.randint(10, 99) for _ in range(rng.randint(3, 8))]
        out.append({"id": f"p{k:03d}", "prompt": prompt, "chosen": [rng.randint(10, 99) for _ in range(rng.randint(2, 6))],
                    "rejected": [rng.randint(10, 99) for _ in range(rng.randint(2, 6))]})
    return out


def test_pack_rows_every_pair_once_chosen_then_rejected_in_full_batches(tmp_path):
    pairs = toy_pairs(40)
    rows = g4dpo.pack_rows(pairs, seq=48, rows=4)
    assert len(rows) % 4 == 0 and [r["batch"] for r in rows] == sorted(r["batch"] for r in rows)
    assert sorted(i for r in rows for i in r["pairs"]) == [p["id"] for p in pairs]
    by = {p["id"]: p for p in pairs}
    for r in rows:
        seg = np.asarray(r["targets_segmentation"])
        for j, pid in enumerate(r["pairs"]):
            # segment 2j+1 holds the chosen side's tokens, 2j+2 the rejected side's, as targets
            assert np.asarray(r["targets"])[seg == 2 * j + 1].tolist() == by[pid]["chosen"]
            assert np.asarray(r["targets"])[seg == 2 * j + 2].tolist() == by[pid]["rejected"]
    g4dpo.write_rows(rows, tmp_path / "rows.parquet")
    arrays, batch = g4dpo.read_packed(tmp_path / "rows.parquet")
    assert arrays["inputs"].shape == (len(rows), 48) and g4dpo.batch_count(batch, 4) == len(rows) // 4
    info = g4dpo.packed_info(tmp_path / "rows.parquet", 4, 48)
    assert info["pairs"] == 40 and info["loss_tokens"] == sum(len(p["chosen"]) + len(p["rejected"]) for p in pairs)
    with pytest.raises(SystemExit, match="max_target_length"):
        g4dpo.packed_info(tmp_path / "rows.parquet", 4, 64)
    with pytest.raises(SystemExit, match="batches of 3"):
        g4dpo.batch_count(batch, 3)


def test_local_batches_give_each_process_its_rows_and_the_padding_batch():
    arrays = {k: np.arange(8 * 5, dtype=np.int32).reshape(8, 5) + 1 for k in g4dpo.KEYS}
    p0 = list(g4dpo.local_batches(arrays, 0, 2, rows=4, batches=2, padding=1))
    p1 = list(g4dpo.local_batches(arrays, 1, 2, rows=4, batches=2, padding=1))
    assert len(p0) == len(p1) == 3
    assert p0[0]["inputs"][:, 0].tolist() == [1, 6] and p1[0]["inputs"][:, 0].tolist() == [11, 16]
    assert p0[1]["inputs"][:, 0].tolist() == [21, 26] and p1[1]["inputs"][:, 0].tolist() == [31, 36]
    assert not p0[2]["inputs"].any() and p0[2]["inputs"].shape == (2, 5)
    with pytest.raises(SystemExit, match="over 3 processes"):
        list(g4dpo.local_batches(arrays, 0, 3, rows=4, batches=1))


# ---------------------------------------------------------------------------------------------------- the loss

def batch_of(pairs, seq, rows):
    packed = g4dpo.pack_rows(pairs, seq, rows)
    return {k: np.asarray([r[k] for r in packed], np.int32) for k in g4dpo.KEYS}, packed


def tunix_reference(chosen, rejected, ref_c, ref_r, beta, eps):
    """tunix/sft/dpo/dpo_trainer.py dpo_loss_fn (algorithm dpo) for one pair, written out"""
    delta = (chosen - ref_c) - (rejected - ref_r)

    def log_sigmoid(x):
        return -math.log1p(math.exp(-x)) if x >= 0 else x - math.log1p(math.exp(x))

    return -(log_sigmoid(beta * delta) * (1 - eps) + log_sigmoid(-beta * delta) * eps), beta * (chosen - ref_c), \
        beta * (rejected - ref_r)


def test_dpo_terms_is_tunixs_loss_per_pair_plus_the_nll_of_the_chosen_sides():
    pairs = toy_pairs(9, seed=3)
    batch, packed = batch_of(pairs, seq=40, rows=3)
    seg = batch["targets_segmentation"]
    rng = np.random.default_rng(0)
    logp = np.where(seg != 0, -rng.uniform(0.1, 3.0, seg.shape), 0.0).astype(np.float32)
    ref = np.where(seg != 0, -rng.uniform(0.1, 3.0, seg.shape), 0.0).astype(np.float32)
    opts = g4dpo.spec(beta=0.3, label_smoothing=0.1, nll_alpha=0.5)
    loss, sums = g4dpo.dpo_terms(logp, ref, seg, opts)
    losses, correct, chosen_logp, chosen_tok = [], 0, 0.0, 0
    for b, r in enumerate(packed):
        for j, _ in enumerate(r["pairs"]):
            c, x = seg[b] == 2 * j + 1, seg[b] == 2 * j + 2
            one, cr, rr = tunix_reference(logp[b][c].sum(), logp[b][x].sum(), ref[b][c].sum(), ref[b][x].sum(), 0.3, 0.1)
            losses.append(one)
            correct += cr > rr
            chosen_logp += logp[b][c].sum()
            chosen_tok += int(c.sum())
    assert sums["pairs"] == 9 and sums["correct"] == correct and sums["chosen_tokens"] == chosen_tok
    want = np.mean(losses) + 0.5 * (-chosen_logp / chosen_tok)
    assert float(loss) == pytest.approx(want, rel=1e-5)


def test_dpo_terms_at_the_reference_is_log_2_and_moves_with_the_chosen_and_rejected():
    batch, _ = batch_of(toy_pairs(6), seq=40, rows=2)
    seg = batch["targets_segmentation"]
    logp = np.where(seg != 0, -1.0, 0.0).astype(np.float32)
    opts = g4dpo.spec(nll_alpha=0.0)
    loss, sums = g4dpo.dpo_terms(logp, logp, seg, opts)
    assert float(loss) == pytest.approx(math.log(2), abs=1e-7) and sums["margin_sum"] == 0 and sums["correct"] == 0
    assert g4dpo.expected_step0(opts)["loss"] == pytest.approx(math.log(2))
    up = np.where(seg % 2 == 1, logp + 0.1, logp)        # the chosen sides more likely than under the reference
    down = np.where((seg % 2 == 0) & (seg > 0), logp - 0.1, logp)   # the rejected sides less likely
    assert g4dpo.dpo_terms(up, logp, seg, opts)[0] < loss and g4dpo.dpo_terms(down, logp, seg, opts)[0] < loss
    assert g4dpo.dpo_terms(up, logp, seg, opts)[1]["correct"] == 6


def test_dpo_terms_ignores_an_unpaired_segment_and_padding_rows():
    pairs = toy_pairs(2)
    rows = g4dpo.pack_rows(pairs, seq=60, rows=2)
    seg = np.asarray([r["targets_segmentation"] for r in rows], np.int32)
    seg = np.concatenate([seg, np.zeros_like(seg)])                       # an all-padding batch's rows
    seg[0][seg[0] == 4] = 0                                               # the second pair loses its rejected side
    logp = np.where(seg != 0, -1.0, 0.0).astype(np.float32)
    _, sums = g4dpo.dpo_terms(logp, logp, seg, OPTS)
    assert sums["pairs"] == 1
    loss, empty = g4dpo.dpo_terms(np.zeros((2, 8), np.float32), np.zeros((2, 8), np.float32), np.zeros((2, 8), np.int32),
                                  OPTS)
    assert empty["pairs"] == 0 and float(loss) == 0.0


def test_ipo_and_length_normalisation():
    batch, packed = batch_of(toy_pairs(4, seed=5), seq=40, rows=2)
    seg = batch["targets_segmentation"]
    rng = np.random.default_rng(1)
    logp = np.where(seg != 0, -rng.uniform(0.1, 3.0, seg.shape), 0.0).astype(np.float32)
    ref = np.where(seg != 0, -rng.uniform(0.1, 3.0, seg.shape), 0.0).astype(np.float32)
    opts = g4dpo.spec(loss="ipo", beta=0.5, nll_alpha=0.0, length_norm=True)
    loss, _ = g4dpo.dpo_terms(logp, ref, seg, opts)
    want = []
    for b, r in enumerate(packed):
        for j, _ in enumerate(r["pairs"]):
            c, x = seg[b] == 2 * j + 1, seg[b] == 2 * j + 2
            h = (logp[b][c].mean() - ref[b][c].mean()) - (logp[b][x].mean() - ref[b][x].mean())
            want.append((h - 1.0) ** 2)
    assert float(loss) == pytest.approx(np.mean(want), rel=1e-5)
    assert g4dpo.expected_step0(opts)["loss"] == pytest.approx(1.0)


def test_spec_refuses_what_it_cannot_compute():
    for bad in ({"loss": "orpo"}, {"beta": 0}, {"label_smoothing": 0.5}, {"nll_alpha": -1}, {"max_segments": 3},
                {"loss": "ipo", "label_smoothing": 0.1}, {"gamma": 1}):
        with pytest.raises(ValueError):
            g4dpo.spec(**bad)
    assert g4dpo.spec(**g4arm.DPO6) == {**g4dpo.DEFAULTS, **g4arm.DPO6}


def test_pref_and_step_records_sum_the_batches():
    parts = [{k: 0.0 for k in g4dpo.SUMS} | {"pairs": 2, "dpo_sum": 1.0, "correct": 1, "margin_sum": 0.4,
                                              "chosen_logp_sum": -10.0, "chosen_tokens": 5, "rejected_logp_sum": -2.0,
                                              "rejected_tokens": 4, "ref_chosen_logp_sum": -10.0,
                                              "ref_rejected_logp_sum": -2.0}] * 2
    r = g4dpo.pref_record("pref_x", 0, parts, 2, 1.0)
    assert (r["pairs"], r["loss"], r["accuracy"], r["margin"], r["chosen_nll"], r["rejected_nll"]) == (
        4, 0.5, 0.5, 0.2, 2.0, 0.5)
    with pytest.raises(SystemExit, match="planned"):
        g4dpo.pref_record("pref_x", 0, parts, 3, 1.0)
    s = g4dpo.step_record(3, parts[0])
    assert (s["step"], s["pairs"], s["loss"], s["accuracy"]) == (3, 2, 0.5, 0.5)


# ---------------------------------------------------------------------------------------------------- sft_lora --dpo

def test_sft_lora_refuses_inconsistent_dpo_arguments_before_jax(tmp_path):
    spec = tmp_path / "dpo.json"
    spec.write_text(json.dumps(g4arm.DPO6))
    base = ["--out", str(tmp_path / "o")]
    for argv, msg in (([*base, "--dpo", str(spec)], "needs --pref-train"),
                      ([*base, "--pref-train", "x"], "belong to --dpo"),
                      ([*base, "--dpo", str(spec), "--pref-train", "x", "--resume-step", "8", "--resume-adapter", "a"],
                       "no --combine"),
                      ([*base, "--dpo", str(spec), "--pref-train", "x", "--pref-eval", "a=x", "--eval", "a=y",
                        "--eval-every", "1"], "both name"),
                      ([*base, "--dpo", str(spec), "--pref-train", "x", "--pref-eval", "a=x"], "at least one --eval")):
        with pytest.raises(SystemExit, match=msg):
            sft_lora.main(argv)
    spec.write_text(json.dumps({"loss": "orpo"}))
    with pytest.raises(SystemExit, match="loss"):
        sft_lora.main([*base, "--dpo", str(spec), "--pref-train", "x"])


# ---------------------------------------------------------------------------------------------------- cands6, build6

class FakeCounter6(t4.FakeCounter):
    """FakeCounter with token ids: one id per word (10 + its letters' sum mod 997), <pad> 0, <bos> 2"""

    def ids(self, texts):
        return [[10 + sum(map(ord, w)) % 997 for w in t.split()] for t in texts]

    def special_id(self, token):
        return {"<pad>": 0, "<bos>": 2}.get(token)


@pytest.fixture
def world6(world4, monkeypatch):
    run_build4(world4)
    tmp = world4["tmp"]
    rng = random.Random(21)
    novels = [g4arm.messages_row(t4.text(rng, t4.EN_WORDS, 30), t4.text(rng, t4.WORDS, 31)) for _ in range(6)]
    write(novels, tmp / "eval-novels.parquet")
    write([{"pid": f"novel-eval-{k}", "en_book_id": "9", "book": "1-9", "author": "a", "tokens_prompt": 90,
            "tokens_target": 33} for k in range(6)], tmp / "eval-novels-meta.parquet")
    monkeypatch.setattr(g4arm, "WANT6", {"book": 1000, "novel": 30, **{f"bt:{s}": 8 for s in g4arm.BT_SOURCES6}})
    monkeypatch.setattr(g4arm, "SEQ6", 1024)
    monkeypatch.setattr(g4arm, "SEQ6_FALLBACK", 512)
    monkeypatch.setattr(g4arm, "PAIR_MAX6", 512)
    monkeypatch.setattr(g4arm, "ROWS6", 4)
    monkeypatch.setattr(g4arm, "SMOKE6_BATCHES", 2)
    return {**world4, "novels_eval": tmp / "eval-novels.parquet", "novels_meta": tmp / "eval-novels-meta.parquet"}


def run_cands6(w):
    out = w["tmp"] / "cands6"
    g4arm.cands6(ns(g44=str(w["tmp"] / "set4" / "G4-4"), g44_report=str(w["tmp"] / "set4" / "report.json"),
                    novels_eval=str(w["novels_eval"]), novels_eval_meta=str(w["novels_meta"]), tokenizer="unused",
                    factor=1.25, workers=4, out=str(out)), counter=FakeCounter6())
    return out


def test_cands6_draws_per_stratum_and_puts_every_class_on_every_worker(world6):
    out = run_cands6(world6)
    m = json.loads((out / "manifest.json").read_text())
    cands = pq.read_table(out / "cands.parquet").to_pylist()
    by = {}
    for r in cands:
        by.setdefault(r["stratum"], []).append(r)
    assert by.keys() == {"book", "novel", "pref_novels", *(f"bt:{s}" for s in g4arm.BT_SOURCES6)}
    assert len(by["novel"]) == math.ceil(30 * 1.25) and all(len(by[f"bt:{s}"]) == 10 for s in g4arm.BT_SOURCES6)
    assert len(by["book"]) == m["funnel"]["book"]["eligible"] and len(by["pref_novels"]) == 6
    # no long BT, no FLAN replay; the English of the eval novels is their user turn's
    assert {r["kind"] for r in cands} == {"book", "novel", "bt"}
    nov = pq.read_table(world6["novels_eval"]).to_pylist()
    assert sorted(g4arm.messages_row(r["en"], r["nl"])["messages"][0]["content"] for r in by["pref_novels"]) == sorted(
        x["messages"][0]["content"] for x in nov)
    segs = {}
    for k in range(4):
        part = pq.read_table(out / f"segments-w{k}.parquet").to_pylist()
        classes = (out / f"classes-w{k}.txt").read_text().split()
        segs.update({r["id"]: k for r in part})
        evals = {r["id"]: r for r in pq.read_table(out / "evalsets.parquet").to_pylist()}
        assert sorted({evals[r["id"]]["length_class"] for r in part}, key=g4arm.EVAL_CLASSES.index) == classes
    assert set(segs) == {r["id"] for r in cands}
    # the same draw again
    first = (out / "cands.parquet").read_bytes()
    run_cands6(world6)
    assert (out / "cands.parquet").read_bytes() == first


def g40_pairs(cands, path, special):
    """G4-0's collected pairs of the candidates: the chosen with its first word changed, unless special says otherwise"""
    rows = []
    for r in cands:
        hyp = "anders " + " ".join(r["nl"].split()[1:])
        flags = {"ended": True, "decode_loop": False, "english_left": False, "empty": False}
        kind = special.get(r["id"])
        if kind == "identical":
            hyp = r["nl"]
        elif kind == "low":
            hyp = "LOW " + hyp
        elif kind == "high":
            hyp = "HIGH " + hyp
        elif kind == "short":
            hyp = "kort"
        elif kind == "gram":
            hyp = t4.EVAL_REF + " " + " ".join(r["nl"].split()[:10])     # an evaluation 13-gram, about as long
        elif kind in ("not_ended", "decode_loop", "english_left", "empty"):
            flags = {**flags, "ended": kind != "not_ended", kind: True} if kind != "not_ended" else {**flags, "ended": False}
        rows.append({"id": r["id"], "set": r["set"], "source": r["en"], "hypothesis": hyp, "reference": r["nl"], **flags})
    write(rows, path)


def fake_chrf(hyp, ref):
    if hyp == ref:
        return 100.0
    return 10.0 if hyp.startswith("LOW") else 90.0 if hyp.startswith("HIGH") else 60.0


def run_build6(w, cands_dir, special):
    cands = pq.read_table(cands_dir / "cands.parquet").to_pylist()
    g40_pairs(cands, w["tmp"] / "g40" / "pairs-w0.parquet", special)
    held = [r for r in pq.read_table(w["tmp"] / "heldout" / "evalsets.parquet").to_pylist()]
    first_btval = min(r["id"] for r in held if r["set"] == "btval")
    write([{"id": r["id"], "hypothesis": "anders " + r["reference"], "ended": True, "decode_loop": False,
            "english_left": False, "empty": r["id"] == first_btval} for r in held],
          w["tmp"] / "g40held" / "pairs-w0.parquet")
    out = w["tmp"] / "build6"
    g4arm.build6(ns(cands=str(cands_dir / "cands.parquet"), rejected=[str(w["tmp"] / "g40" / "pairs-w*.parquet")],
                    heldout=str(w["tmp"] / "heldout" / "evalsets.parquet"),
                    heldout_g40=[str(w["tmp"] / "g40held" / "pairs-w*.parquet")],
                    evalsets=str(w["tmp"] / "evalsets.parquet"), blind=str(w["blind"]), tokenizer="unused", out=str(out)),
                 counter=FakeCounter6(), chrf=fake_chrf)
    return out


def test_build6_filters_draws_packs_and_writes_the_preference_eval_sets(world6):
    cands_dir = run_cands6(world6)
    cands = pq.read_table(cands_dir / "cands.parquet").to_pylist()
    novels = sorted(r["id"] for r in cands if r["stratum"] == "novel")
    special = dict(zip(novels, ("identical", "low", "high", "short", "gram", "not_ended", "decode_loop",
                                "english_left", "empty")))
    out = run_build6(world6, cands_dir, special)
    report = json.loads((out / "report.json").read_text())
    f = report["funnel"]["novel"]
    assert {k: f[k] for k in ("identical", "chrf_low", "chrf_high", "length_ratio", "eval_13gram", "not_ended",
                              "decode_loop", "english_left", "empty")} == dict.fromkeys(
        ("identical", "chrf_low", "chrf_high", "length_ratio", "eval_13gram", "not_ended", "decode_loop",
         "english_left", "empty"), 1)
    assert f["kept"] == len(novels) - 9 and f["drawn"] == min(30, f["kept"])
    pairs = pq.read_table(out / "pairs.parquet").to_pylist()
    train = sorted((p for p in pairs if p["set"] == "train"), key=lambda p: p["order"])
    assert not {p["id"] for p in train} & set(special)
    assert report["pairs"]["train"] == len(train) == sum(report["by_stratum"].values())
    assert all(report["by_stratum"][f"bt:{s}"] == 8 for s in g4arm.BT_SOURCES6)
    # the rejected side is G4-0's translation; every pair once in the packed file, in the training order
    for seq in (1024, 512):
        rows = pq.read_table(out / f"train-{seq}.parquet").to_pylist()
        # first-fit within a batch: a batch holds the next stretch of the training order, its rows in any order
        order = {p["id"]: p["order"] for p in train}
        by_batch = [sorted((i for r in rows if r["batch"] == b for i in r["pairs"]), key=order.get)
                    for b in range(1 + max(r["batch"] for r in rows))]
        assert [i for part in by_batch for i in part] == [p["id"] for p in train]
        assert all(len(r["inputs"]) == seq for r in rows) and len(rows) % 4 == 0
        smoke = pq.read_table(out / f"smoke-{seq}.parquet").to_pylist()
        assert smoke == [r for r in rows if r["batch"] < 2]
        assert report["packing"][str(seq)]["train"]["steps"] == 1 + max(r["batch"] for r in rows)
    # the preference eval sets: G4-0's held-out translations (an empty one out), the eval novels
    assert report["pairs"]["pref_btval"] == 19 and report["funnel"]["pref_btval"]["empty"] == 1
    assert report["pairs"]["pref_books"] == 1
    assert report["pairs"]["pref_novels"] == 6
    for name in g4arm.PREF_SETS6[1:]:
        assert pq.read_table(out / f"eval-{name}-1024.parquet").num_rows % 4 == 0
    # the SFT view of the chosen sides, in the training order
    chosen = pq.read_table(out / "train-chosen.parquet").column("messages").to_pylist()
    assert [m[1]["content"] for m in chosen] == [p["chosen"] for p in train]
    # deterministic
    first = (out / "train-1024.parquet").read_bytes()
    run_build6(world6, cands_dir, special)
    assert (out / "train-1024.parquet").read_bytes() == first


def test_build6_refuses_a_candidate_without_g4_0s_translation(world6):
    cands_dir = run_cands6(world6)
    cands = pq.read_table(cands_dir / "cands.parquet").to_pylist()
    write(cands[1:], cands_dir / "less.parquet")
    g40_pairs(cands[1:], world6["tmp"] / "g40" / "pairs-w0.parquet", {})
    with pytest.raises(SystemExit, match="without G4-0's translation"):
        g4arm.build6(ns(cands=str(cands_dir / "cands.parquet"), rejected=[str(world6["tmp"] / "g40" / "pairs-w*.parquet")],
                        heldout=str(world6["tmp"] / "heldout" / "evalsets.parquet"), heldout_g40=["unused"],
                        evalsets=str(world6["tmp"] / "evalsets.parquet"), blind=str(world6["blind"]),
                        tokenizer="unused", out=str(world6["tmp"] / "b")), counter=FakeCounter6(), chrf=fake_chrf)


def test_english_of_inverts_the_user_turn():
    en = "The house by the river.\n\nA second paragraph."
    assert g4arm.english_of(g4arm.g4lora.user_turn(en)) == en
    with pytest.raises(SystemExit, match="natural-v1"):
        g4arm.english_of("Translate this: x")


def test_assign_balanced_puts_every_class_on_every_worker():
    rows = [{"id": f"{c}{k}", "length_class": c, "tokens": k} for c in ("tiny", "small", "long") for k in range(8)]
    out = g4arm.assign_balanced(rows, 4)
    assert all({r["length_class"] for r in part} == {"tiny", "small", "long"} and len(part) == 6
               for part in out.values())
    with pytest.raises(SystemExit, match="beyond"):
        g4arm.assign_balanced([{"id": "x", "length_class": "xxlong", "tokens": 1}], 2)


# ---------------------------------------------------------------------------------------------------- smoke6, report

def smoke6_summary(seq=4096, **over):
    evals = []
    sft = [s for s in g4arm.EVAL_SETS4 if not (s == "btlong" and seq == 2048)]
    base = {"books": 2.5704, "btval": 1.8449, "own": 0.0747, "btlong": 1.2595, "novels": 2.2935}
    for step in (0, 4, 8):
        for s in sft:
            evals.append({"step": step, "set": s, "loss": base[s] - 0.01 * step, "seconds": 30.0})
        for s in g4arm.PREF_SETS6:
            evals.append({"step": step, "set": s, "kind": "pref", "pairs": 500, "loss": math.log(2) - 0.01 * step,
                          "accuracy": 0.0 if step == 0 else 0.6, "margin": 0.0 if step == 0 else 0.1,
                          "chosen_nll": 2.0, "rejected_nll": 0.1, "ref_chosen_nll": 2.0, "ref_rejected_nll": 0.1,
                          "seconds": 20.0})
    s = {"steps_done": 8, "loss": [[k, 3.0] for k in range(1, 9)], "peak_gib_max": 28.5, "eval": evals,
         "dpo": {"spec": g4dpo.spec(**g4arm.DPO6), "steps": [{"step": k, "pairs": 50, "loss": 0.69} for k in range(1, 9)]},
         "throughput": {"step_s_median": 14.0}, "times": {"python_to_setup_s": 300, "setup_s": 130, "compile_s_est": 420}}
    s.update(over)
    return s


def smoke_rows(n=50, ended=50, qe=2.2):
    return [{"id": f"s{k}", "ended": k < ended, "decode_loop": False, "empty": False, "metricx_qe": qe}
            for k in range(n)]


BUILD_REPORT = {"packing": {"4096": {"train": {"steps": 800}}, "2048": {"train": {"steps": 1600}}}}
G44_SUMMARY = {"eval": [{"step": 0, "set": s, "loss": v} for s, v in
                        {"books": 2.5705, "btval": 1.8449, "own": 0.0747, "btlong": 1.2595, "novels": 2.2935}.items()]}
LORA = {"modules": 410, "relative_update_norm": 1e-4}


def test_smoke6_passes_when_every_check_holds_and_names_each_failure():
    ok = g4arm.smoke6_verdict(smoke6_summary(), {"rank": 32, "alpha": 16.0, "scale": 0.5}, G44_SUMMARY, 4096, LORA,
                              smoke_rows(), smoke_rows(qe=2.25), BUILD_REPORT)
    assert ok["passed"], {k: c for k, c in ok["checks"].items() if not c["holds"]}
    # 800 steps x 14 s + 5 rounds of (5 x 30 + 3 x 20) s + 850 s
    assert ok["estimate"]["hours"] == pytest.approx((800 * 14 + 5 * 210 + 850) / 3600, abs=0.01)
    bad = smoke6_summary(peak_gib_max=30.1)
    bad["eval"] = [dict(e, margin=0.01) if e["set"] == "pref_books" and e["step"] == 0 else e for e in bad["eval"]]
    bad["eval"] = [dict(e, loss=e["loss"] + 0.01) if e["set"] == "own" and e["step"] == 0 else e for e in bad["eval"]]
    v = g4arm.smoke6_verdict(bad, {"rank": 32, "alpha": 16.0, "scale": 0.5}, G44_SUMMARY, 4096, LORA, smoke_rows(),
                             smoke_rows(), BUILD_REPORT)
    failed = {k for k, c in v["checks"].items() if not c["holds"]}
    assert failed == {"peak_gib", "policy_is_reference_at_step0", "base_losses_at_step0"}
    # the fallback at 2,048: no btlong, and its own step count
    fb = g4arm.smoke6_verdict(smoke6_summary(2048), {"rank": 32, "alpha": 16.0, "scale": 0.5}, G44_SUMMARY, 2048, LORA,
                              smoke_rows(), smoke_rows(), BUILD_REPORT)
    assert fb["passed"] and fb["estimate"]["steps"] == 1600


def test_smoke6_cli_exits_1_on_a_failed_check(tmp_path):
    files = {"summary": smoke6_summary(steps_done=7), "adapter": {"rank": 32, "alpha": 16.0, "scale": 0.5},
             "base-summary": G44_SUMMARY, "build-report": BUILD_REPORT}
    args = {}
    for name, value in files.items():
        (tmp_path / f"{name}.json").write_text(json.dumps(value))
        args[name.replace("-", "_")] = str(tmp_path / f"{name}.json")
    (tmp_path / "t.log").write_text("LORA " + json.dumps(LORA) + "\n")
    for name in ("arm", "base"):
        write(smoke_rows(), tmp_path / f"{name}-pairs.parquet")
        write([{"id": r["id"], "qe_source_hypothesis": 2.2} for r in smoke_rows()], tmp_path / f"{name}-metricx.parquet")
    with pytest.raises(SystemExit):
        g4arm.smoke6(ns(**args, translate_log=str(tmp_path / "t.log"), arm_pairs=str(tmp_path / "arm-pairs.parquet"),
                        arm_metricx=str(tmp_path / "arm-metricx.parquet"), base_pairs=str(tmp_path / "base-pairs.parquet"),
                        base_metricx=str(tmp_path / "base-metricx.parquet"), seq=4096, out=str(tmp_path / "s.md")))
    assert "Niet geslaagd" in (tmp_path / "s.md").read_text()


def test_pref_measure_reports_the_sets_own_and_the_training_windows():
    s = smoke6_summary()
    s["dpo"]["steps"] = [{"step": k, "pairs": 10 + k, "loss": 0.7, "accuracy": 0.5, "margin": 0.1, "chosen_nll": 1.0,
                          "rejected_nll": 0.2} for k in range(1, 121)]
    p = g4arm.pref_measure(s)
    assert set(p["sets"]) == set(g4arm.PREF_SETS6) and list(p["sets"]["pref_books"]) == [0, 4, 8]
    assert p["own"] == {0: pytest.approx(0.0747), 4: pytest.approx(0.0347), 8: pytest.approx(-0.0053)}
    assert [w["steps"] for w in p["train"]] == [[1, 50], [51, 100], [101, 120]]
    text = "\n".join(g4arm.pref_text(p))
    assert "Voorkeuren" in text and "pref_novels" in text and "0,075 → 0,339" in text


def test_pairtexts6_and_pairclass6(tmp_path):
    rows = [{"id": f"{k}-{i}", "set": "train", "kind": k, "chosen": f"c {i}", "rejected": f"r {i}"}
            for k in ("novel", "bt") for i in range(5)] + [{"id": "e", "set": "pref_books", "kind": "books",
                                                             "chosen": "c", "rejected": "r"}]
    write(rows, tmp_path / "pairs.parquet")
    g4arm.pairtexts6(ns(pairs=str(tmp_path / "pairs.parquet"), out=str(tmp_path / "texts.parquet")))
    texts = pq.read_table(tmp_path / "texts.parquet").to_pylist()
    assert len(texts) == 20 and {(t["model"], t["set"]) for t in texts} == {
        (m, k) for m in ("chosen", "rejected") for k in ("novel", "bt")}
    preds = [{**t, "p_original": 0.9 if t["model"] == "chosen" else 0.1} for t in texts]
    write(preds, tmp_path / "predictions.parquet")
    g4arm.pairclass6(ns(predictions=str(tmp_path / "predictions.parquet"), out=str(tmp_path / "pc.md")))
    r = json.loads((tmp_path / "pc.json").read_text())
    assert r["bt"]["chosen"]["original"] == 1.0 and r["bt"]["rejected"]["original"] == 0.0
