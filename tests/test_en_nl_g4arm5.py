"""Arm G4-5 (draft amendment G4-5): g4mix.py (stacked adapters = the weighted sum of their updates, the coordinate
search, the split), sft_lora.py's pure parts (the schedule of a resumed run, the argument checks), and g4arm.py's
build5, split5, smoke5 and report5 on small synthetic inputs (no tokenizer, no model, no TPU)."""

import json
import random

import numpy as np
import pyarrow.parquet as pq
import pytest

# test_en_nl_g4arm4 puts experiments/en-nl-translator on the path; its world4 fixture is G4-4's inputs
import test_en_nl_g4arm4 as t4

g4arm = t4.g4arm
FakeCounter, ns, write = t4.FakeCounter, t4.ns, t4.write
world, world4, run_build4 = t4.world, t4.world4, t4.run_build4

import g4mix  # noqa: E402
import sft_lora  # noqa: E402
import translate  # noqa: E402

SHAPES = {"decoder/layers_0/self_attention/query": ((8, 2, 4), (8, 3), (3, 2, 4)),
          "decoder/layers_0/mlp/wo": ((12, 8), (12, 3), (3, 8))}


def adapter(seed, rank=3):
    rng = np.random.default_rng(seed)
    out = {}
    for module, (_, a, b) in SHAPES.items():
        out[f"{module}/kernel_lora_a"] = rng.standard_normal(a[:-1] + (rank,)).astype(np.float32)
        out[f"{module}/kernel_lora_b"] = rng.standard_normal((rank,) + b[1:]).astype(np.float32)
    return out


def update(arrays, module, scale=0.5):
    shape = SHAPES[module][0]
    return translate.lora_delta(arrays[f"{module}/kernel_lora_a"], arrays[f"{module}/kernel_lora_b"], shape, scale)


# ---------------------------------------------------------------------------------------------------- stacking

def test_stacked_update_is_the_weighted_sum_of_the_deltas():
    shared, b1, b2 = adapter(0), adapter(1), adapter(2)
    weights, s = [0.7, 0.45], 1.0
    coefs = g4mix.coefficients(weights, s)
    assert coefs == pytest.approx([s - 0.7 - 0.45, 0.7, 0.45])
    stacked = g4mix.stack([shared, b1, b2], coefs)
    for module in SHAPES:
        assert stacked[f"{module}/kernel_lora_a"].shape[-1] == 9 and stacked[f"{module}/kernel_lora_b"].shape[0] == 9
        # W0 + Delta_s + sum_i w_i (Delta_i - Delta_s), the formula of the amendment
        want = update(shared, module) + sum(w * (update(b, module) - update(shared, module))
                                            for w, b in zip(weights, (b1, b2)))
        np.testing.assert_allclose(update(stacked, module), want, rtol=1e-5, atol=1e-6)


def test_one_branch_alone_is_that_branch_exactly():
    shared, b1, b2 = adapter(0), adapter(1), adapter(2)
    stacked = g4mix.stack([shared, b1, b2], g4mix.coefficients([1.0, 0.0]))
    for module in SHAPES:
        # coefficients 0, 1, 0: the stacked B holds zeros and b1's B unchanged (exactly); the product over the longer
        # rank sums the same nonzero terms, in another order of float32 additions
        b = stacked[f"{module}/kernel_lora_b"]
        assert not b[:3].any() and not b[6:].any() and np.array_equal(b[3:6], b1[f"{module}/kernel_lora_b"])
        np.testing.assert_allclose(update(stacked, module), update(b1, module), rtol=1e-6, atol=1e-7)
    warm = g4mix.stack([shared, b1, b2], g4mix.coefficients([0.0, 0.0]))
    for m in SHAPES:
        np.testing.assert_allclose(update(warm, m), update(shared, m), rtol=1e-6, atol=1e-7)


def test_stack_refuses_other_modules_or_shapes():
    a, b = adapter(0), adapter(1)
    with pytest.raises(ValueError, match="other modules"):
        g4mix.stack([a, {k: v for k, v in b.items() if "mlp" not in k}], [1.0, 0.0])
    with pytest.raises(ValueError, match="shapes"):
        g4mix.stack([a, adapter(1, rank=4)], [1.0, 0.0])
    with pytest.raises(ValueError, match="coefficients"):
        g4mix.stack([a, b], [1.0])


def write_adapter(directory, arrays, rank=3, alpha=1.5, **meta):
    directory.mkdir(parents=True, exist_ok=True)
    np.savez(directory / "adapter.npz", **arrays)
    (directory / "adapter.json").write_text(json.dumps({"rank": rank, "alpha": alpha, "scale": alpha / rank,
                                                        "steps": 7, **meta}))


def test_stack_cli_writes_an_adapter_that_translate_merges_to_the_sum(tmp_path, capsys):
    arrays = {"shared": adapter(0), "bt_raad": adapter(1), "novels": adapter(2)}
    for name, a in arrays.items():
        write_adapter(tmp_path / name, a)
    g4mix.main(["stack", "--component", f"shared={tmp_path / 'shared'}", "--component", f"bt_raad={tmp_path / 'bt_raad'}",
                "--component", f"novels={tmp_path / 'novels'}", "--weights", "0.5,0.25", "--out", str(tmp_path / "c1")])
    meta = json.loads((tmp_path / "c1" / "adapter.json").read_text())
    assert (meta["rank"], meta["alpha"], meta["scale"]) == (9, 4.5, 0.5)
    assert [c["coef"] for c in meta["components"]] == [0.25, 0.5, 0.25] and meta["weights"] == {"bt_raad": 0.5,
                                                                                                 "novels": 0.25}
    with np.load(tmp_path / "c1" / "adapter.npz") as z:
        got = {k: z[k] for k in z.files}
    for module, (shape, _, _) in SHAPES.items():
        a, b = got[f"{module}/kernel_lora_a"], got[f"{module}/kernel_lora_b"]
        want = 0.25 * update(arrays["shared"], module) + 0.5 * update(arrays["bt_raad"], module) + \
            0.25 * update(arrays["novels"], module)
        np.testing.assert_allclose(translate.lora_delta(a, b, shape, meta["scale"]), want, rtol=1e-5, atol=1e-6)
    assert '"rank": 9' in capsys.readouterr().out
    with pytest.raises(SystemExit, match="exists"):
        g4mix.main(["stack", "--component", f"shared={tmp_path / 'shared'}", "--component",
                    f"bt_raad={tmp_path / 'bt_raad'}", "--component", f"novels={tmp_path / 'novels'}",
                    "--weights", "0.5,0.25", "--out", str(tmp_path / "c1")])


def test_stack_cli_takes_a_candidate_of_the_search_and_refuses_other_components(tmp_path):
    for name, seed in (("shared", 0), ("flan", 1)):
        write_adapter(tmp_path / name, adapter(seed))
    summary = {"combine": {"spec": {"components": [{"name": "shared"}, {"name": "flan"}], "shared_coef": 1.0},
                           "search": {"top": [{"w": [0.75], "value": 1.0}, {"w": [0.25], "value": 1.1}]}}}
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    comps = ["--component", f"shared={tmp_path / 'shared'}", "--component", f"flan={tmp_path / 'flan'}"]
    g4mix.main(["stack", *comps, "--from-summary", str(tmp_path / "summary.json"), "--candidate", "2", "--out",
                str(tmp_path / "c2")])
    assert json.loads((tmp_path / "c2" / "adapter.json").read_text())["weights"] == {"flan": 0.25}
    with pytest.raises(SystemExit, match="not the search's"):
        g4mix.main(["stack", "--component", f"shared={tmp_path / 'shared'}", "--component",
                    f"books={tmp_path / 'flan'}", "--from-summary", str(tmp_path / "summary.json"), "--out",
                    str(tmp_path / "c3")])


def test_stacked_meta_refuses_other_scales_and_scaled_components():
    good = {"rank": 32, "alpha": 16.0, "scale": 0.5}
    with pytest.raises(ValueError, match="scale"):
        g4mix.stacked_meta([good, {"rank": 32, "alpha": 32.0, "scale": 1.0}], ["shared", "x"], ["a", "b"], [1, 0],
                           [0], 1.0)
    with pytest.raises(ValueError, match="lora_b_factor"):
        g4mix.stacked_meta([good, {**good, "lora_b_factor": 0.5}], ["shared", "x"], ["a", "b"], [1, 0], [0], 1.0)


# ---------------------------------------------------------------------------------------------------- the search

def quadratic(target):
    return lambda w: sum((a - b) ** 2 for a, b in zip(w, target)) + 1.0


def spec(n, **kw):
    comps = [("shared", "s.npz")] + [(f"b{i}", f"b{i}.npz") for i in range(n)]
    return g4mix.spec_dict(comps, {"wmt24pp_s": 1.0}, **kw)


def test_search_starts_with_the_anchors_and_finds_the_minimum():
    target = [0.375, 1.0, 0.0]
    r = g4mix.run_search(quadratic(target), spec(3, budget=200, step=0.25, min_step=0.0625, start=[0.0, 0.0, 0.0]))
    phases = [p["phase"] for p in r["points"]]
    assert phases[:5] == ["anchor/warmup", "anchor/alone/0", "anchor/alone/1", "anchor/alone/2", "anchor/uniform"]
    assert r["stopped"] == "converged" and r["best"]["w"] == pytest.approx(target)
    assert len({tuple(p["w"]) for p in r["points"]}) == len(r["points"])            # every point once
    assert r == g4mix.run_search(quadratic(target), spec(3, budget=200, step=0.25, min_step=0.0625,
                                                         start=[0.0, 0.0, 0.0]))  # deterministic
    # from the uniform start (1/3 each) the grid is 1/3 + k/16: the nearest grid point
    u = g4mix.run_search(quadratic(target), spec(3, budget=200, step=0.25, min_step=0.0625))
    assert u["best"]["w"] == pytest.approx(target, abs=0.0625 / 2)


def test_search_keeps_its_bounds_and_budget():
    r = g4mix.run_search(quadratic([2.0, -1.0]), spec(2, budget=12, upper=1.5))
    assert r["evaluations"] == 12 and r["stopped"] == "budget"
    assert all(0.0 <= x <= 1.5 for p in r["points"] for x in p["w"])
    with pytest.raises(ValueError, match="anchors"):
        g4mix.run_search(quadratic([0, 0]), spec(2, budget=3))


def test_top_candidates_are_the_best_and_apart():
    points = [{"order": k, "w": w, "value": v} for k, (w, v) in enumerate(
        [([0.5, 0.5], 1.0), ([0.5, 0.5625], 1.01), ([0.25, 0.5], 1.02), ([0.5, 0.25], 1.02), ([1.0, 1.0], 3.0)])]
    top = g4mix.top_candidates(points, 3, 0.125)
    assert [p["order"] for p in top] == [0, 2, 3]
    assert g4mix.objective_value({"a": 1.0, "b": 3.0}, {"a": 1.0, "b": 1.0}) == 2.0
    with pytest.raises(ValueError, match="no loss"):
        g4mix.objective_value({"a": 1.0}, {"a": 1.0, "b": 1.0})


def test_spec_wants_the_shared_warmup_first():
    with pytest.raises(ValueError, match="shared warmup"):
        g4mix.spec_dict([("b0", "x")], {"a": 1.0})
    s = spec(2)
    assert s["components"][0]["name"] == "shared" and s["search"]["budget"] == 80 and s["search"]["top_k"] == 4
    assert s["repeat_max_diff"] == 1e-4 and "warmup_check" not in s


def test_repeat_is_equal_within_1e_4_addendum_g4_5_7b():
    # the smoke test's pair (18 Nov): the first value broadcast in float32, the repeat in float64
    r = g4mix.repeat_result([0.0, 0.0], 2.4015402793884277, 2.401540332915216, g4mix.REPEAT_MAX_DIFF)
    assert r["equal"] and r["diff"] == pytest.approx(5.4e-8, abs=1e-8)
    # a rest of an earlier point after a failed restore of the base kernels (>= 1e-2)
    assert not g4mix.repeat_result([0.0], 2.40, 2.41, g4mix.REPEAT_MAX_DIFF)["equal"]


def warmup_summary(books=2.0, novels=1.5, tokens=130996):
    return {"steps": 1000, "eval": [{"step": s, "set": n, "loss": v + (1.0 if s == 0 else 0.0), "tokens": tokens}
                                    for s in (0, 500, 1000) for n, v in (("books", books), ("novels", novels))]}


def test_warmup_check_takes_the_warmups_last_eval_and_holds_within_its_bound(tmp_path):
    check = g4mix.warmup_check_of(warmup_summary(), ["books", "novels"], 0.02)
    assert check["step"] == 1000 and check["losses"] == {"books": 2.0, "novels": 1.5}
    assert check["tokens"] == {"books": 130996, "novels": 130996}
    tokens = {"books": 130996, "novels": 130996}
    ok = g4mix.warmup_check_result(check, {"books": 2.019, "novels": 1.49}, tokens)
    assert ok["holds"] and ok["diff"]["books"] == pytest.approx(0.019)
    # the smoke test's s8 merged (0.079 off) fails; other batches (tokens) fail
    assert not g4mix.warmup_check_result(check, {"books": 2.079, "novels": 1.5}, tokens)["holds"]
    assert not g4mix.warmup_check_result(check, {"books": 2.0, "novels": 1.5}, {**tokens, "novels": 1})["holds"]
    with pytest.raises(ValueError, match="no eval"):
        g4mix.warmup_check_of(warmup_summary(), ["books", "flanval"], 0.02)
    # the CLI puts it into the spec
    (tmp_path / "w.json").write_text(json.dumps(warmup_summary()))
    g4mix.main(["spec", "--component", "shared=s.npz", "--component", "b0=b.npz", "--objective", "wmt24pp_s=1",
                "--warmup-check", str(tmp_path / "w.json"), "--check-set", "books", "--check-set", "novels",
                "--check-max-diff", "0.02", "--out", str(tmp_path / "spec.json")])
    x = json.loads((tmp_path / "spec.json").read_text())
    assert x["warmup_check"]["losses"] == {"books": 2.0, "novels": 1.5} and x["warmup_check"]["max_diff"] == 0.02
    assert x["repeat_max_diff"] == 1e-4


# ---------------------------------------------------------------------------------------------------- the split

def eval_rows():
    """WMT24++: 4 domains of uneven size (literary: two big documents); FLORES: articles of 3 sentences, 2 domains"""
    rows = []
    for k in range(120):
        domain = ("literary", "news", "social", "speech")[min(k // 30, 3)] if k < 120 else "speech"
        rows.append({"id": f"wmt24pp-{k:04d}", "set": "wmt24pp",
                     "meta": json.dumps({"domain": domain, "document_id": f"{domain}{k // 60}", "is_bad_source": False})})
    for k in range(60):
        rows.append({"id": f"flores-{k:04d}", "set": "flores",
                     "meta": json.dumps({"domain": "wikinews" if k < 30 else "wikivoyage", "URL": f"u{k // 3}"})})
    return rows


def test_split_keeps_each_domains_share_and_articles_whole_and_the_blind_segment_out_of_the_search():
    rows = eval_rows()
    wmt = [r for r in rows if r["set"] == "wmt24pp"]
    forced = {"wmt24pp-0007"}
    part = g4mix.split_set(wmt, "seed/x", 0.5, forced)
    assert part == g4mix.split_set(list(reversed(wmt)), "seed/x", 0.5, forced)
    assert part != g4mix.split_set(wmt, "seed/y", 0.5, forced)
    assert part["wmt24pp-0007"] == "decision"
    for d in ("literary", "news", "social", "speech"):
        ids = [r["id"] for r in wmt if json.loads(r["meta"])["domain"] == d]
        assert sum(part[i] == "search" for i in ids) == 15                    # half of every domain
    flores = [r for r in rows if r["set"] == "flores"]
    fpart = g4mix.split_set(flores, "seed/f", 0.5)
    by_article = {}
    for r in flores:
        by_article.setdefault(g4mix.unit_of(r), set()).add(fpart[r["id"]])
    assert all(len(v) == 1 for v in by_article.values())
    for d in ("wikinews", "wikivoyage"):
        ids = [r["id"] for r in flores if json.loads(r["meta"])["domain"] == d]
        assert sum(fpart[i] == "search" for i in ids) == 15


def test_split5_writes_the_search_parts_reference_files(tmp_path, monkeypatch):
    monkeypatch.setattr(g4arm, "FOUR_SETS", {"wmt24pp": 40, "flores": 30, "ntrex": 20, "europarl": 4})
    monkeypatch.setattr(g4arm, "N_FLAN", 10)
    monkeypatch.setattr(g4arm, "N_BLIND", 3)
    rng = random.Random(3)
    rows = []
    for k in range(40):
        rows.append({"id": f"wmt24pp-{k:04d}", "set": "wmt24pp", "source": t4.text(rng, t4.EN_WORDS, 8),
                     "reference": t4.text(rng, t4.WORDS, 9),
                     "meta": json.dumps({"document_id": f"d{k // 4}", "is_bad_source": k == 0})})
    for k in range(30):
        rows.append({"id": f"flores-{k:04d}", "set": "flores", "source": "a b", "reference": "c d",
                     "meta": json.dumps({"URL": f"u{k // 3}"})})
    rows += [{"id": f"ntrex-{k:04d}", "set": "ntrex", "source": "a b c", "reference": "d e", "meta": "{}"}
             for k in range(20)]
    rows += [{"id": f"europarl-{k}", "set": "europarl", "source": "a", "reference": "b", "meta": "{}"} for k in range(4)]
    rows += [{"id": f"flan:{k}", "set": "flan", "source": "a", "reference": None, "meta": "{}"} for k in range(10)]
    rows += [{"id": "blind-001", "set": "blind", "source": "x", "reference": None,
              "meta": json.dumps({"blind_set": "wmt24pp", "item": "wmt24pp-0013"})},
             {"id": "blind-002", "set": "blind", "source": "x", "reference": None,
              "meta": json.dumps({"blind_set": "flan", "item": "flan:0-99"})},
             {"id": "blind-003", "set": "blind", "source": "x", "reference": None,
              "meta": json.dumps({"blind_set": "idiom", "item": "gv:1"})}]
    write(rows, tmp_path / "evalsets.parquet")
    out = tmp_path / "split"
    g4arm.split5(ns(evalsets=str(tmp_path / "evalsets.parquet"), share=0.5, tokenizer="unused", out=str(out)),
                 counter=FakeCounter())
    split = json.loads((out / "split.json").read_text())
    assign = split["assign"]
    assert {i.split("-")[0].split(":")[0] for i in assign} == {"wmt24pp", "flores", "ntrex", "flan"}
    assert assign["wmt24pp-0013"] == "decision"                                     # a blind passage's segment
    assert split["sets"]["wmt24pp"]["forced_decision"] == 1 and split["digest"] == g4mix.digest(assign)
    for s in g4arm.SEARCH_LOSS_SETS5:
        meta = pq.read_table(out / f"eval-{s}-s-meta.parquet").to_pylist()
        msgs = pq.read_table(out / f"eval-{s}-s.parquet").column("messages").to_pylist()
        want = [r for r in rows if r["set"] == s and assign[r["id"]] == "search" and r["id"] != "wmt24pp-0000"]
        assert [m["id"] for m in meta] == sorted(r["id"] for r in want) and len(msgs) == len(want)
        by_id = {r["id"]: r for r in want}
        for m, x in zip(meta, msgs):
            assert x[0]["content"] == g4arm.g4lora.user_turn(by_id[m["id"]]["source"])
            assert x[1]["content"] == by_id[m["id"]]["reference"]
    again = tmp_path / "again"
    g4arm.split5(ns(evalsets=str(tmp_path / "evalsets.parquet"), share=0.5, tokenizer="unused", out=str(again)),
                 counter=FakeCounter())
    assert (again / "split.json").read_bytes() == (out / "split.json").read_bytes()


# ---------------------------------------------------------------------------------------------------- build5

@pytest.fixture
def small_packing(monkeypatch):
    """After build4: the synthetic set is small (pass 1 without the replay: 9-12 batches per process at 4,096), so
    steps without the margin of sft_lora.step_limit"""

    def apply():
        monkeypatch.setattr(g4arm, "steps_of", lambda batches: min(batches))
        monkeypatch.setattr(g4arm, "SMOKE3_LONGEST", 8)
        monkeypatch.setattr(g4arm, "SMOKE3_RANDOM", 16)
        monkeypatch.setattr(g4arm, "SMOKE5_STEPS", 1)

    return apply


def run_build5(w, small_packing, **kw):
    set4 = run_build4(w)
    small_packing()
    write([{"pid": "999-1"}], w["tmp"] / "eval-novels-meta.parquet")
    out = w["tmp"] / "set5"
    args = dict(g44=str(set4 / "G4-4"), g44_report=str(set4 / "report.json"),
                novels_eval=str(w["tmp"] / "eval-novels-meta.parquet"), warmup_steps=6, branch_steps=5, steps=[],
                flanval=4, out=str(out))
    args.update(kw)
    g4arm.build5(ns(**args))
    return set4, out


def rows_of(d):
    return (pq.read_table(d / "train.parquet").column("messages").to_pylist(),
            pq.read_table(d / "train-meta.parquet").to_pylist())


def test_build5_takes_g4_4s_examples_per_source_with_a_fresh_order_per_epoch(world4, small_packing):
    set4, out = run_build5(world4, small_packing, steps=["books=12"])
    x4, m4 = rows_of(set4 / "G4-4")
    seen = {}
    for x, m in zip(x4, m4):
        seen[m["id"]] = x
    report = json.loads((out / "report.json").read_text())
    # the warmup: pass 1 without the replay, G4-4's examples byte for byte, the fewest rows that give its steps
    x, m = rows_of(out / "G4-5-warmup")
    assert {r["kind"] for r in m} <= {"bt", "book", "novel"} and all(seen[r["id"]] == e for r, e in zip(m, x))
    pass1 = {r["id"] for r in m4 if r["epoch"] == 1 and r["kind"] != "replay"}
    assert {r["id"] for r in m} <= pass1
    lengths = [r["tokens_prompt"] + r["tokens_target"] for r in m]
    assert g4arm.steps_of(g4arm.pack_batches(lengths)) >= 6 > g4arm.steps_of(g4arm.pack_batches(lengths[:-1]))
    # a BT branch: only its source; every epoch a permutation of the whole source, in another order than the last
    x, m = rows_of(out / "G4-5-bt_raad")
    source = sorted(r["id"] for r in m4 if r["epoch"] == 1 and r["kind"] == "bt" and r["source"] == "raad")
    assert {r["source"] for r in m} == {"raad"} and all(seen[r["id"]] == e for r, e in zip(m, x))
    # the books with 12 steps: several epochs, each the whole source once, the order new every epoch
    _, m = rows_of(out / "G4-5-books")
    books = sorted(r["id"] for r in m4 if r["epoch"] == 1 and r["kind"] == "book")
    epochs = sorted({r["epoch"] for r in m})
    assert len(epochs) >= 3 and report["sets"]["books"]["steps"] == 12
    full = [[r["id"] for r in m if r["epoch"] == e] for e in epochs[:-1]]
    assert all(sorted(f) == books for f in full) and full[0] != full[1]
    # the novels: G4-4's passes as epochs, a pair before any repeat
    _, m = rows_of(out / "G4-5-novels")
    for e in {r["epoch"] for r in m}:
        ids = {r["id"] for r in m if r["epoch"] == e}
        assert ids <= {r["id"] for r in m4 if r["epoch"] == (e - 1) % 3 + 1 and r["kind"] == "novel"}
    # FLAN: every pass's replay rows, without flanval's
    _, m = rows_of(out / "G4-5-flan")
    val = {r["id"] for r in pq.read_table(out / "eval-flanval-meta.parquet").to_pylist()}
    assert len(val) == 4 and not val & {r["id"] for r in m}
    assert {r["kind"] for r in m} == {"replay"} and len(source) > 0
    for name, s in report["sets"].items():
        man = json.loads((out / f"G4-5-{name}" / "manifest.json").read_text())
        assert man["step_limit"] >= man["steps"] == (6 if name == "warmup" else 12 if name == "books" else 5)
    assert set(report["smoke"]) == {"smoke-warmup", "smoke-branch"}


def test_build5_refuses_an_eval_novel_in_the_training(world4, small_packing):
    set4 = run_build4(world4)
    small_packing()
    novel = next(r["id"] for r in pq.read_table(set4 / "G4-4" / "train-meta.parquet").to_pylist()
                 if r["kind"] == "novel")
    write([{"pid": novel[len("novel:"):]}], world4["tmp"] / "eval-novels-meta.parquet")
    with pytest.raises(SystemExit, match="eval set are novel pairs"):
        g4arm.build5(ns(g44=str(set4 / "G4-4"), g44_report=str(set4 / "report.json"),
                        novels_eval=str(world4["tmp"] / "eval-novels-meta.parquet"), warmup_steps=6,
                        branch_steps=5, steps=[], flanval=4, out=str(world4["tmp"] / "x")))


# ---------------------------------------------------------------------------------------------------- sft_lora

@pytest.mark.parametrize("total", [8, 16, 1000, 1279, 2000, 9316, 10_001])
def test_schedule_overrides_give_the_exact_steps(total):
    for warmup, decay in ((2, 0), (279, 0), (279, 100), (0, 4), (min(279, total // 2), total // 10)):
        if warmup + decay > total:
            continue
        fw, fd, length = sft_lora.schedule_overrides(total, warmup, decay)
        w = float(fw.split("=", 1)[1])
        d = float(fd.split("=", 1)[1])
        assert int(total * w) == warmup and int(total * d) == decay and length == "learning_rate_schedule_steps=-1"


def test_lr_points_hold_the_resumed_steps_and_the_cooldown():
    pts = sft_lora.lr_points(16, 2.5 / 16, 4.5 / 16, extra=(8, 9))
    assert {0, 1, 2, 3, 8, 9, 12, 15} == set(pts)
    assert sft_lora.lr_points(16, 2.5 / 16, 4.5 / 16) == [0, 1, 2, 3, 8, 12, 15]


def test_sft_lora_refuses_inconsistent_arguments_before_jax(tmp_path):
    old = {k: v for k, v in spec(1).items() if k != "repeat_max_diff"}       # a spec from before addendum G4-5.7b
    (tmp_path / "old.json").write_text(json.dumps(old))
    checked = spec(1, warmup_check=g4mix.warmup_check_of(warmup_summary(), ["books", "novels"], 0.02))
    (tmp_path / "checked.json").write_text(json.dumps(checked))
    for argv, msg in ((["--resume-step", "8"], "go together"), (["--warmup-steps", "2"], "go together"),
                      (["--warmup-steps", "2", "--decay-fraction", "1.0"], r"\[0, 1\)"),
                      (["--combine", "x.json", "--steps", "8"], "measurement"),
                      (["--combine", str(tmp_path / "old.json"), "--steps", "1", "--eval", "wmt24pp_s=f"],
                       "repeat_max_diff"),
                      (["--combine", str(tmp_path / "checked.json"), "--steps", "1", "--eval", "wmt24pp_s=f",
                        "--eval", "books=g"], r"\['novels'\]"),
                      (["--temperatures", "1,2", "--eval", "wmt24pp_s=f", "--eval", "books=g", "--eval", "novels=h",
                        "--combine", str(tmp_path / "checked.json"), "--steps", "1"], "eval-only"),
                      (["--temperatures", "1,2"], "eval-only"),
                      (["--temperatures", "0.5,2", "--eval", "a=f"], "1.0 among them")):
        with pytest.raises(SystemExit, match=msg):
            sft_lora.main(["--out", str(tmp_path / "o"), *argv, "--", "steps=1"])


def test_temperature_grid_and_record():
    grid = sft_lora.temperature_grid("0.5:3.0:0.05")
    assert len(grid) == 51 and grid[0] == 0.5 and grid[-1] == 3.0 and 1.0 in grid and grid[10] == 1.0
    assert sft_lora.temperature_grid("2,1,1.5") == [1.0, 1.5, 2.0]
    for bad in ("0.5:3.0:0.07", "0:2:0.5", "1:2", "-1,1"):
        with pytest.raises(SystemExit):
            sft_lora.temperature_grid(bad)
    r = sft_lora.temperature_record([1.0, 2.0], [[3.0, 2.0], [1.0, 2.0]], 8.0, 2)
    assert r == {"t": [1.0, 2.0], "loss": [0.5, 0.5]}
    with pytest.raises(SystemExit, match="planned"):
        sft_lora.temperature_record([1.0, 2.0], [[3.0, 2.0]], 8.0, 2)


def test_temperature_losses_are_the_cross_entropy_of_the_scaled_logits_from_one_pass():
    jax = pytest.importorskip("jax")
    jnp = jax.numpy
    rng = np.random.default_rng(0)
    logits = rng.standard_normal((2, 5, 7)).astype(np.float32) * 3
    labels = rng.integers(0, 7, (2, 5))
    one_hot = np.eye(7, dtype=np.float32)[labels]
    temps = [0.5, 1.0, 2.0]

    def ce(z):
        lse = np.log(np.exp(z - z.max(-1, keepdims=True)).sum(-1)) + z.max(-1)
        return lse - np.take_along_axis(z, labels[..., None], -1)[..., 0]

    got = np.asarray(sft_lora.temperature_losses(jnp.asarray(logits), jnp.asarray(one_hot), temps))
    for k, t in enumerate(temps):
        np.testing.assert_allclose(got[k], ce(logits / t), rtol=1e-5, atol=1e-5)
    # greedy is the same at every T
    assert (np.argmax(logits / 0.5, -1) == np.argmax(logits / 2.0, -1)).all()

    # the wrapped eval loss: MaxText's own result unchanged, xent_temps the masked sums of the same pass
    class MaxUtils:
        @staticmethod
        def cross_entropy_with_logits(z, targets, z_loss=0.0):
            lse = jax.scipy.special.logsumexp(z, axis=-1)
            return lse - jnp.sum(targets * z, axis=-1), jnp.zeros(z.shape[:-1])

    module = MaxUtils()
    seg = np.ones((2, 5), np.int32)
    seg[1, 3:] = 0

    def maxtext_eval_loss(model, inputs, targets, targets_segmentation):
        xent, _ = module.cross_entropy_with_logits(model, jax.nn.one_hot(targets, 7))
        xent = xent * (targets_segmentation != 0)
        return jnp.sum(xent) / jnp.sum(targets_segmentation != 0), {"xent_sum": jnp.sum(xent),
                                                                     "total_weights": jnp.sum(targets_segmentation != 0)}

    wrapped = sft_lora.with_temperatures(maxtext_eval_loss, temps, 2, module)
    loss, aux = jax.jit(lambda z, y, s: wrapped(z, inputs=y, targets=y, targets_segmentation=s))(
        jnp.asarray(logits), jnp.asarray(labels), jnp.asarray(seg))
    want = [float((ce(logits / t) * (seg != 0)).sum()) for t in temps]
    np.testing.assert_allclose(np.asarray(aux["xent_temps"]), want, rtol=1e-5)
    np.testing.assert_allclose(float(aux["xent_sum"]), want[1], rtol=1e-5)


# ---------------------------------------------------------------------------------------------------- smoke5

def evals(steps, base=2.0, sets=None):
    return [{"step": s, "set": n, "loss": base - 0.01 * s, "tokens": 100, "seconds": 10.0, "batches": 1}
            for s in steps for n in (sets or g4arm.EVAL_SETS5)]


def smoke_inputs():
    batches = [21, 21, 21, 21]
    adapter_meta = {"rank": 32, "alpha": 16.0, "scale": 0.5}
    common = {"peak_gib_max": 29.0, "config": {"max_target_length": g4arm.SEQ3}, "epoch": {"batches_per_process":
                                                                                             batches},
              "throughput": {"step_s_median": 10.5}, "times": {"python_to_setup_s": 200, "setup_s": 130,
                                                                "compile_s_est": 420}}
    w = {**common, "steps_done": 8, "steps": 8, "loss": [[k, 1.5] for k in range(1, 9)], "eval": evals([0, 4, 8]),
         "lr_at": {"7": g4arm.LR5}}
    w["eval"] = [{**e, "loss": 0.05} if e["set"] == "own" and e["step"] == 0 else e for e in w["eval"]]
    b = {**common, "steps_done": 8, "steps": 16, "own_steps": 8, "loss": [[k, 1.2] for k in range(9, 17)],
         "eval": evals([8, 12, 16]), "lr_at": {"8": g4arm.LR5, "15": g4arm.LR5 * 0.1},
         "resume": {"step": 8, "counters": {"opt_state/0/count": 8}}}
    # the branch's eval at its start is the warmup's last: the same numbers
    b["eval"] = [e if e["step"] != 8 else dict(next(x for x in w["eval"] if x["step"] == 8 and x["set"] == e["set"]))
                 for e in b["eval"]]
    w0 = {e["set"]: e["loss"] for e in w["eval"] if e["step"] == 0}
    w8 = {e["set"]: e["loss"] for e in w["eval"] if e["step"] == 8}
    points = [{"order": 0, "phase": "anchor/warmup", "w": [0, 0], "value": 1.0,
               "losses": {"books": w8["books"] + 0.005, "novels": w8["novels"] - 0.004}},
              {"order": 1, "phase": "anchor/alone/0", "w": [1, 0], "value": 1.0, "losses": {"books": 1.9}},
              {"order": 2, "phase": "anchor/alone/1", "w": [0, 1], "value": 0.9,
               "losses": {"books": 0.88, "novels": 0.83}},
              {"order": 3, "phase": "anchor/uniform", "w": [0.5, 0.5], "value": 0.95, "losses": {}}]
    c = {**common, "eval": [{**e, "loss": w0[e["set"]]} for e in evals([0])] + evals([1]),
         "combine": {"spec": {"components": [{"name": "shared"}, {"name": "b8"}, {"name": "g43"}]},
                     "search": {"points": points, "evaluations": 4, "stopped": "budget", "best": points[2]},
                     "evals": [{"tag": f"search/{k}", "seconds": 60.0} for k in range(4)] +
                              [{"tag": "top/1", "seconds": 300.0}],
                     "repeat": {"first": 1.0, "again": 1.0, "equal": True}}}
    g43 = {"eval": [{"step": 7651, "set": "books", "loss": 0.878}]}
    g43n = {"eval": [{"step": 0, "set": "novels", "loss": 0.832}]}
    man = {"batches_per_process": batches, "tokens": 3_000_000}
    return w, adapter_meta, b, adapter_meta, c, man, man, g43, g43n


def test_smoke5_passes_when_every_check_holds_and_names_each_failure():
    args = smoke_inputs()
    r = g4arm.smoke5_verdict(*args)
    assert r["passed"], {k: v for k, v in r["checks"].items() if not v["holds"]}
    w, wa, b, ba, c, wm, bm, g43, g43n = smoke_inputs()
    b["eval"][0]["loss"] += 1e-3                                                    # the restore is not the warmup
    c["combine"]["search"]["points"][2]["losses"]["books"] = 0.95                   # the merge is not LoRA
    b["resume"] = None
    r = g4arm.smoke5_verdict(w, wa, b, ba, c, wm, bm, g43, g43n)
    failed = {k for k, v in r["checks"].items() if not v["holds"]}
    assert failed == {"branch_start_is_warmup_end", "merge_g43_is_g43", "branch_resumed"}


def test_smoke5_estimate_sums_warmup_branches_and_search(tmp_path):
    w, wa, b, ba, c, wm, bm, g43, g43n = smoke_inputs()
    sets = {name: {"steps": 1000, "tokens": 64_000_000} for name in (g4arm.WARMUP5, *g4arm.BRANCHES5)}
    x = g4arm.smoke5_estimate(w, b, c, {"sets": sets}, wm, bm)
    assert set(x["runs"]) == {g4arm.WARMUP5, *g4arm.BRANCHES5}
    assert x["search"]["points"] == g4arm.SEARCH_BUDGET5 and x["search"]["point_s"] == 60.0
    # 9 runs of 1,000 steps at 10.5 s: ~26 h of steps alone
    assert 26 < x["hours"] < 45


# ---------------------------------------------------------------------------------------------------- report5

def report_world(tmp_path, qe, share, human=2.0):
    """Eval sets of 8 segments per rule set (two documents of WMT24++), the split, the human QE, two candidates and
    G4-0 with the given mean QE per (model, part) and share of 'original' per (model, part)"""
    rows = []
    for s in ("wmt24pp", "flores", "ntrex", "flan"):
        for k in range(8):
            meta = {"document_id": f"d{k // 4}", "is_bad_source": False} if s == "wmt24pp" else (
                {"URL": f"u{k // 2}"} if s == "flores" else {})
            rows.append({"id": f"{s}-{k}", "set": s, "source": "a b", "reference": None if s == "flan" else "c d",
                         "tokens": 2, "over_512": False, "length_class": "tiny", "meta": json.dumps(meta)})
    write(rows, tmp_path / "evalsets.parquet")
    assign = {r["id"]: ("search" if int(r["id"].rsplit("-", 1)[1]) < 4 else "decision") for r in rows}
    (tmp_path / "split.json").write_text(json.dumps({"assign": assign, "digest": g4mix.digest(assign), "seed": "s",
                                                     "share": 0.5, "sets": {}}))
    write([{"id": r["id"], "qe_source_reference": human} for r in rows if r["set"] != "flan"],
          tmp_path / "refqe.parquet")
    preds = []
    for name in ("G4-5c1", "G4-5c2", "G4-0"):
        d = tmp_path / name
        write([{"id": r["id"], "set": r["set"], "source": r["source"], "hypothesis": "p", "reference": r["reference"],
                "ended": True, "decode_loop": False, "english_left": False, "empty": False} for r in rows],
              d / "pairs" / "pairs-w0.parquet")
        write([{"id": r["id"], "qe_source_hypothesis": qe[name][assign[r["id"]]],
                "ref_source_hypothesis_reference": 1.0 if r["reference"] else None} for r in rows],
              d / "scores" / "metricx-w0.parquet")
        write([{"id": r["id"], "comet22": 0.85} for r in rows if r["reference"]], d / "scores" / "comet-w0.parquet")
        for r in rows:
            if r["set"] in ("wmt24pp", "flan"):
                k = int(r["id"].rsplit("-", 1)[1])
                p = share[name][assign[r["id"]]]
                preds.append({"model": name, "set": r["set"], "id": r["id"], "p_original": 0.9 if k % 4 < p * 4 else 0.1})
    write(preds, tmp_path / "pred.parquet")
    top = [{"order": 0, "phase": "anchor/warmup", "w": [0.0], "value": 1.0, "losses": {"wmt24pp_s": 1.0}}]
    (tmp_path / "search.json").write_text(json.dumps({"eval": [{"step": 0, "set": "wmt24pp_s", "loss": 1.2}],
                                                      "combine": {"spec": {"objective": {"wmt24pp_s": 1.0}},
                                                                  "search": {"points": top, "top": top,
                                                                             "evaluations": 1, "stopped": "budget",
                                                                             "best": top[0]},
                                                                  "repeat": {"equal": True},
                                                                  "warmup_check": {"step": 1000, "holds": True,
                                                                                   "tokens_equal": True,
                                                                                   "want": {"books": 2.0},
                                                                                   "got": {"books": 2.01},
                                                                                   "diff": {"books": 0.01},
                                                                                   "max_diff": 0.02},
                                                                  "seconds": 1.0}}))
    for name, steps in (("warmup", (0, 1000)), ("books", (1000, 2000))):
        (tmp_path / f"{name}.json").write_text(json.dumps({"eval": [{"step": s, "set": "books", "loss": 2.0 - s / 1e4}
                                                                    for s in steps], "steps_done": 1000}))
    return ns(evalsets=str(tmp_path / "evalsets.parquet"), split=str(tmp_path / "split.json"),
              refqe=str(tmp_path / "refqe.parquet"), candidate=[f"G4-5c1={tmp_path / 'G4-5c1'}",
                                                                f"G4-5c2={tmp_path / 'G4-5c2'}"],
              base=str(tmp_path / "G4-0"), reference=[], predictions=str(tmp_path / "pred.parquet"),
              search=str(tmp_path / "search.json"),
              run=[f"warmup={tmp_path / 'warmup.json'}", f"books={tmp_path / 'books.json'}"], tally=None,
              smoke=None, out=str(tmp_path / "report.md"))


def test_report5_selects_on_the_search_part_and_decides_on_the_decision_part(tmp_path, monkeypatch):
    monkeypatch.setattr(g4arm, "HUMAN_QE", {"wmt24pp": 2.0, "flores": 2.0, "ntrex": 2.0})
    # c1 holds (a1) on the search part, c2 does not but has more 'original' there: c1 is chosen; on the decision
    # part c1 holds (a1) and is +0.25 against G4-0
    qe = {"G4-5c1": {"search": 1.9, "decision": 1.95}, "G4-5c2": {"search": 2.1, "decision": 1.0},
          "G4-0": {"search": 1.5, "decision": 1.5}}
    share = {"G4-5c1": {"search": 0.5, "decision": 0.5}, "G4-5c2": {"search": 0.75, "decision": 1.0},
             "G4-0": {"search": 0.25, "decision": 0.25}}
    args = report_world(tmp_path, qe, share)
    g4arm.report5(args)
    r = json.loads((tmp_path / "report.json").read_text())
    assert r["selected"] == "G4-5c1" and r["selection"].startswith("a1 on the search part")
    assert r["a1"]["holds"] and r["b"]["delta"] == pytest.approx(0.25) and r["b"]["holds"]
    assert r["verdict"].startswith("wacht op de blinde beoordeling")
    assert r["parts"]["decision"]["n"] == {"wmt24pp": 4, "flores": 4, "ntrex": 4, "flan": 4}
    assert r["training"]["books"]["vs_warmup"]["books"] == pytest.approx(-0.1)
    assert "Gekozen combinatie: **G4-5c1**" in (tmp_path / "report.md").read_text()
    tally = {"arm": "G4-5c1", "a2": True, "c": True, "arm_better": 40, "base_better": 20, "equal": 40,
             "errors_arm": 5, "errors_base": 6}
    (tmp_path / "tally.json").write_text(json.dumps(tally))
    args.tally = str(tmp_path / "tally.json")
    g4arm.report5(args)
    assert json.loads((tmp_path / "report.json").read_text())["verdict"] == "helpt"
    (tmp_path / "tally.json").write_text(json.dumps({**tally, "arm": "G4-5c2"}))
    with pytest.raises(SystemExit, match="selected candidate"):
        g4arm.report5(args)


def test_report5_wants_the_searchs_checks_of_addendum_g4_5_7b_and_reports_the_smoke_anchor(tmp_path, monkeypatch):
    monkeypatch.setattr(g4arm, "HUMAN_QE", {"wmt24pp": 2.0, "flores": 2.0, "ntrex": 2.0})
    qe = {k: {"search": 1.9, "decision": 1.9} for k in ("G4-5c1", "G4-5c2", "G4-0")}
    share = {k: {"search": 0.25, "decision": 0.25} for k in qe}
    args = report_world(tmp_path, qe, share)
    (tmp_path / "smoke.json").write_text(json.dumps({"checks": {"merge_g43_is_g43": {
        "value": {"books": [0.8787, 0.8777], "novels": [0.8991, 0.8897]}, "holds": True}}}))
    args.smoke = str(tmp_path / "smoke.json")
    g4arm.report5(args)
    text = (tmp_path / "report.md").read_text()
    assert "aanvulling G4-5.7b" in text and "gehaald" in text and "G4-3 alleen" in text
    search = json.loads((tmp_path / "search.json").read_text())
    for key, value in (("warmup_check", {**search["combine"]["warmup_check"], "holds": False}),
                       ("warmup_check", None), ("repeat", {"equal": False})):
        bad = json.loads(json.dumps(search))
        bad["combine"][key] = value
        (tmp_path / "search.json").write_text(json.dumps(bad))
        with pytest.raises(SystemExit, match="G4-5.7b"):
            g4arm.report5(args)


# ---------------------------------------------------------------------------------------------------- temp5

def temp_run(curves, tokens=100, steps_done=0, shift=0.0):
    """An eval-only temperature run: per set a curve over the grid, its own (stack) loss at T = 1 (+ shift)"""
    grid = sft_lora.temperature_grid(g4arm.TEMPERATURES5)
    out = []
    for name, f in curves.items():
        loss = [f(t) for t in grid]
        out.append({"step": 0, "set": name, "tokens": tokens, "loss": f(1.0) + shift,
                    "temperatures": {"t": grid, "loss": loss}})
    return {"steps_done": steps_done, "eval": out}


def test_temp5_fits_t_on_books_and_btval_and_reports_the_share_of_the_drop(tmp_path):
    sets = ("books", "btval", "own", "novels")
    # G4-0: books and btval lowest at T = 1.5 (2.5 -> 2.0), own lowest at the grid's lower end
    curve = {"books": lambda t: 2.0 + (t - 1.5) ** 2 * 2, "btval": lambda t: 1.5 + (t - 1.5) ** 2 * 2,
             "own": lambda t: 0.05 + 0.1 * t, "novels": lambda t: 2.0 + (t - 1.3) ** 2}
    (tmp_path / "base.json").write_text(json.dumps(temp_run(curve)))
    (tmp_path / "w.json").write_text(json.dumps(temp_run({s: (lambda t: 0.8 + (t - 1.0) ** 2) for s in sets})))
    warm = {"steps": 1000, "eval": [{"step": st, "set": s, "loss": curve[s](1.0) if st == 0 else 1.0, "tokens": 100}
                                    for st in (0, 1000) for s in sets]}
    (tmp_path / "warm.json").write_text(json.dumps(warm))
    g44 = {"heldout": {"loss": {s: {"0": curve[s](1.0), "9316": 0.5} for s in ("books", "btval", "own")}},
           "novels": {"arm_loss": 0.75, "base_loss": curve["novels"](1.0)}}
    (tmp_path / "g44.json").write_text(json.dumps(g44))
    args = ns(base=str(tmp_path / "base.json"), warmup_run=str(tmp_path / "w.json"), warmup=str(tmp_path / "warm.json"),
              g44_report=str(tmp_path / "g44.json"), out=str(tmp_path / "temp.md"))
    g4arm.temp5(args)
    r = json.loads((tmp_path / "temp.json").read_text())
    assert r["fit"]["sets"] == ["books", "btval"] and r["fit"]["t"] == 1.5 and not r["fit"]["edge"]
    books = r["sets"]["books"]
    # L1 2.5, L_T* 2.0, warmup 1.0: a third of the drop; G4-4 0.5: a quarter
    assert books["share_fit_warmup"] == pytest.approx(1 / 3) and books["share_fit_g44"] == pytest.approx(0.25)
    assert r["sets"]["own"]["best_t"] == 0.5 and r["sets"]["own"]["best_edge"]
    assert r["sets"]["novels"]["best_t"] == pytest.approx(1.3)
    assert r["fit_warmup"]["t"] == 1.0 and r["sets"]["books"]["warmup_best_t"] == 1.0
    text = (tmp_path / "temp.md").read_text()
    assert "T\\* = 1,50" in text and "≤ 0,50" in text and "33,3%" in text
    # T = 1 is not the round's own loss: the measurement is wrong
    (tmp_path / "base.json").write_text(json.dumps(temp_run(curve, shift=1e-3)))
    with pytest.raises(SystemExit, match="T = 1"):
        g4arm.temp5(args)
    # other target tokens than the warmup run's
    (tmp_path / "base.json").write_text(json.dumps(temp_run(curve, tokens=99)))
    with pytest.raises(SystemExit, match="target tokens"):
        g4arm.temp5(args)


def test_report5_without_a1_on_the_search_part_takes_the_smallest_excess_and_checks_the_human_qe(tmp_path, monkeypatch):
    monkeypatch.setattr(g4arm, "HUMAN_QE", {"wmt24pp": 2.0, "flores": 2.0, "ntrex": 2.0})
    qe = {"G4-5c1": {"search": 2.3, "decision": 2.3}, "G4-5c2": {"search": 2.1, "decision": 2.1},
          "G4-0": {"search": 1.5, "decision": 1.5}}
    share = {k: {"search": 0.25, "decision": 0.25} for k in qe}
    args = report_world(tmp_path, qe, share)
    g4arm.report5(args)
    r = json.loads((tmp_path / "report.json").read_text())
    assert r["selected"] == "G4-5c2" and r["selection"].startswith("no candidate holds")
    assert not r["a1"]["holds"] and r["verdict"] == "helpt niet"
    monkeypatch.setattr(g4arm, "HUMAN_QE", {"wmt24pp": 2.2, "flores": 2.0, "ntrex": 2.0})
    with pytest.raises(SystemExit, match="human references' QE of wmt24pp"):
        g4arm.report5(args)
