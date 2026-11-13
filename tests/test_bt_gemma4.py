"""experiments/en-nl-translator/bt_gemma4.py without a model or a bucket: the claims and the work loop on a store in
memory, the units of the plan, the probe's draw, the rule that chooses the prompt, the rows of the dataset and the
report. Run with --noconftest where the conftest's imports (jax, flax) are not installed. The test of the real prompt
lengths runs where Gemma 4's tokenizer is (GEMMA4_TOKENIZER)."""

import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "en-nl-translator"))
sys.path.insert(0, str(ROOT / "experiments" / "flan-nl"))

import bt_gemma4 as bg  # noqa: E402
import bt_natural as bn  # noqa: E402
import translate as tr  # noqa: E402

GEMMA4_TOKENIZER = Path(os.environ.get("GEMMA4_TOKENIZER", "/dev/shm/translate-generic/tokenizer-gemma4"))
ME = {"owner": "vm-a/h0", "run": "1"}


# ---------------------------------------------------------------------------------------------------- claims

def test_a_claim_is_made_once_and_refused_to_another_while_fresh():
    store = bg.MemoryStore()
    assert bg.claim(store, "claims/u", ME, lease=600, t=1000)
    assert not bg.claim(store, "claims/u", {"owner": "vm-b/h0", "run": "9"}, lease=600, t=1100)
    assert json.loads(store.read("claims/u")[0])["owner"] == "vm-a/h0"


def test_a_stale_claim_is_taken_over():
    store = bg.MemoryStore()
    assert bg.claim(store, "claims/u", ME, lease=600, t=1000)
    assert bg.claim(store, "claims/u", {"owner": "vm-b/h1", "run": "9"}, lease=600, t=1601)
    assert json.loads(store.read("claims/u")[0])["owner"] == "vm-b/h1"


def test_a_claim_of_an_earlier_run_on_the_same_vm_is_taken_over_at_once():
    store = bg.MemoryStore()
    assert bg.claim(store, "claims/u", {"owner": "vm-a/h1", "run": "1"}, lease=600, t=1000)
    # the VM came back after a preemption under the same name: a new run, either half
    assert bg.claim(store, "claims/u", {"owner": "vm-a/h0", "run": "2"}, lease=600, t=1001)
    # the other half of the same run does not take it from its sibling
    assert not bg.claim(store, "claims/u", {"owner": "vm-a/h1", "run": "2"}, lease=600, t=1002)


def test_a_lease_is_renewed_by_its_holder_and_free_once_released():
    store = bg.MemoryStore()
    assert bg.claim(store, "publish.lock", ME, lease=600, t=1000)
    assert bg.claim(store, "publish.lock", ME, lease=600, t=1500)
    assert json.loads(store.read("publish.lock")[0])["time"] == 1500
    other = {"owner": "vm-b/publish", "run": "5"}
    assert not bg.claim(store, "publish.lock", other, lease=600, t=1600)
    bg.release(store, "publish.lock", ME)
    assert bg.claim(store, "publish.lock", other, lease=600, t=1601)


def test_a_replace_with_an_old_generation_fails():
    store = bg.MemoryStore()
    store.create("k", "a")
    gen = store.read("k")[1]
    assert store.replace("k", "b", gen)
    assert not store.replace("k", "c", gen)
    assert store.read("k")[0] == "b"


# ---------------------------------------------------------------------------------------------------- units, work

def segment_rows(prefix, n, tokens=10):
    return [{"id": f"{prefix}:{i:05d}", "text": f"tekst {i}", "tokens": tokens} for i in range(n)]


def test_units_are_consecutive_ids_of_one_part_split_and_class_in_class_order():
    segs = {("raad", "train", "small"): ("segments/raad/train-small.parquet", segment_rows("raad", 5)),
            ("fineweb", "train", "medium"): ("segments/fineweb/train-medium.parquet", segment_rows("fw", 3)),
            ("fineweb", "validation", "small"): ("segments/fineweb/validation-small.parquet", segment_rows("fv", 2))}
    units = bg.make_units(segs, ["tiny", "small", "medium"], 2, list(bg.PARTS))
    assert [u["name"] for u in units] == ["fineweb/validation-small-0000", "raad/train-small-0000",
                                          "raad/train-small-0001", "raad/train-small-0002",
                                          "fineweb/train-medium-0000", "fineweb/train-medium-0001"]
    last = units[3]
    assert (last["start"], last["end"], last["n"], last["first"], last["last"]) == (4, 5, 1, "raad:00004", "raad:00004")
    assert bg.hf_path("raad/train-small-0002") == "raad/train-small-0002.parquet"
    assert bg.plan_hash(units) == bg.plan_hash([dict(u) for u in units])
    assert bg.plan_hash(units) != bg.plan_hash(units[:-1])


def fake_translate(calls):
    def translate(u):
        calls.append(u["name"])
        ids = [f"{u['name']}#{k}" for k in range(u["n"])]
        return {"id": ids, "text_en": ["x"] * u["n"], "ended": [True] * u["n"], "tokens_in": [5] * u["n"],
                "tokens_out": [3] * u["n"]}, 1.0
    return translate


def units_of(names, prompt="natural-v1-fmt"):
    return [{"name": n, "n": 2, "prompt": prompt, "class": "small"} for n in names]


def test_two_processes_share_the_units_without_doing_one_twice(tmp_path):
    store = bg.MemoryStore()
    names = [f"raad/train-small-{k:04d}" for k in range(4)]
    a, b = [], []
    me_a = dict(ME, work_dir=str(tmp_path / "a"), region="us-east1")
    me_b = {"owner": "vm-b/h0", "run": "7", "work_dir": str(tmp_path / "b"), "region": "us-east1"}
    # b already holds the second unit
    assert bg.claim(store, f"claims/{names[1]}", me_b, bg.LEASE_SECONDS)
    done_a = bg.work_units(units_of(names), store, me_a, fake_translate(a), log=lambda s: None)
    assert done_a == [names[0], names[2], names[3]] and a == done_a
    done_b = bg.work_units(units_of(names), store, me_b, fake_translate(b), progress=bg.Progress(store, every=0),
                           log=lambda s: None)
    assert done_b == [names[1]]
    assert store.list("shards/") == {f"{n}.parquet" for n in names}
    shard = tmp_path / "s.parquet"
    store.get(f"shards/{names[0]}.parquet", shard)
    row = bn.read_rows(shard)[0]
    assert (row["unit"], row["prompt"], row["owner"], row["region"]) == (names[0], "natural-v1-fmt", "vm-a/h0", "us-east1")


def test_units_done_in_the_other_region_are_skipped_and_count_for_the_status(tmp_path):
    store = bg.MemoryStore()
    names = [f"dpc/train-long-{k:04d}" for k in range(3)]
    store.create("other-done.txt", names[2] + "\n")
    calls = []
    done = bg.work_units(units_of(names), store, dict(ME, work_dir=str(tmp_path)), fake_translate(calls),
                         log=lambda s: None)
    assert done == names[:2]
    plan_info = {"units": [{"name": n, "class": "long"} for n in names + ["dpc/train-long-0003"]]}
    s = bg.region_status(plan_info, bg.Progress(store))
    assert s["classes"]["long"] == {"units": 4, "here": 2, "other": 1, "pending": 1} and not s["complete"]
    store.create("shards/dpc/train-long-0003.parquet", "x")
    assert bg.region_status(plan_info, bg.Progress(store))["complete"]


def test_backward_is_the_reverse_order():
    assert bg.ordered([1, 2, 3], "backward") == [3, 2, 1] and bg.ordered([1, 2, 3], "forward") == [1, 2, 3]


class WordCounter:
    def __init__(self, *_):
        pass

    def many(self, texts):
        return [len(t.split()) * 10 for t in texts]


def write_source(root, key, folder, split, rows):
    bn.write_parquet(rows, Path(root) / key / folder / f"{split}-00000-of-00001.parquet")


def fake_hf(root, n_per=3):
    for part, (key, folder) in bg.PARTS.items():
        for split in bg.SPLITS:
            rows = [{"id": f"{part}:{split}:{i}", "nl": " ".join(["woord"] * (i + 1)), "en": "word " * (i + 1)}
                    for i in range(n_per)]
            write_source(root, key, folder, split, rows)


def test_plan_puts_every_passage_in_one_unit_and_stops_at_a_passage_too_long(tmp_path, monkeypatch):
    monkeypatch.setattr(bn, "Counter", WordCounter)
    fake_hf(tmp_path / "hf")
    bg.main(["plan", "--data", str(tmp_path / "hf"), "--tokenizer", "-", "--out", str(tmp_path / "plan"),
             "--shard-size", "2", "--expect", str(len(bg.PARTS) * 2 * 3)])
    info = json.loads((tmp_path / "plan" / "units.json").read_text())
    covered = []
    for u in info["units"]:
        rows = bn.read_rows(tmp_path / "plan" / u["segments"])[u["start"]:u["end"]]
        assert [r["id"] for r in rows][0] == u["first"] and len(rows) == u["n"]
        covered += [r["id"] for r in rows]
    assert len(covered) == len(set(covered)) == len(bg.PARTS) * 2 * 3
    # 10, 20, 30 tokens: all tiny (max_source 32)
    assert {u["class"] for u in info["units"]} == {"tiny"}
    with pytest.raises(SystemExit, match="above 15 tokens"):
        bg.main(["plan", "--data", str(tmp_path / "hf"), "--tokenizer", "-", "--out", str(tmp_path / "p2"),
                 "--max-source", "15", "--expect", "0"])
    with pytest.raises(SystemExit, match="expected 5"):
        bg.main(["plan", "--data", str(tmp_path / "hf"), "--tokenizer", "-", "--out", str(tmp_path / "p3"),
                 "--expect", "5"])
    with pytest.raises(SystemExit, match="beyond the largest class"):
        bg.main(["plan", "--data", str(tmp_path / "hf"), "--tokenizer", "-", "--out", str(tmp_path / "p4"),
                 "--max-source", "4096"])


# ---------------------------------------------------------------------------------------------------- probe, rule

def probe_row(i, source, nl_words=10, en_words=10, length_class="short"):
    return {"id": f"{source}:{i:04d}", "nl": " ".join(["woord"] * nl_words), "en": " ".join(["word"] * en_words),
            "qe_en_nl": 1.0, "length_class": length_class, "hard_fails": [], "ended": True, "tokens_nl": 9,
            "tokens_en": 9}


def test_probe_draw_takes_fresh_rows_the_pilots_structure_cases_and_context_turns():
    rows = {s: [probe_row(i, s) for i in range(120)] + [probe_row(500 + i, s, length_class="long") for i in range(20)]
            + [probe_row(700 + i, s, nl_words=5) for i in range(30)] for s in ("dpc", "raad")}
    pilot = {"sample_ids": [f"dpc:{i:04d}" for i in range(50)],
             "structure_ids": [f"dpc:{i:04d}" for i in range(10)] + ["raad:0001"]}
    context = [dict(probe_row(i, "ctx"), length_class=None) for i in range(30)]
    out, info = bg.probe_draw(rows, context, pilot)
    by = {}
    for r in out:
        by.setdefault((r["source"], r["stratum"]), []).append(r)
    assert len(by[("dpc", "random")]) == 40 and len(by[("dpc", "long")]) == 7 and len(by[("dpc", "ratio_low")]) == 7
    assert not {r["id"] for s in ("random", "long", "ratio_low") for r in by[("dpc", s)]} & set(pilot["sample_ids"])
    assert len(by[("dpc", "structure")]) == 6 and {r["id"] for r in by[("dpc", "structure")]} <= set(pilot["structure_ids"])
    assert len(by[("raad", "structure")]) == 1 and info["raad"]["short"]["structure"] == [1, 6]
    assert len(by[("handelingen", "context")]) == 20
    assert all(r["length_class"] == "long" for r in by[("raad", "long")])


def stats(**prompts):
    return {p: {"n": 200, "hard": h, "median": m} for p, (h, m) in prompts.items()}


def test_the_rule_takes_the_eligible_prompt_closest_to_1_06():
    prompt, reason, table = bg.choose({"natural-v1": {"n": 200, "hard": 10, "median": 1.060},
                                       "natural-v1-fmt": {"n": 200, "hard": 3, "median": 1.030},
                                       "natural-v1-fmt-len": {"n": 200, "hard": 2, "median": 1.075}})
    # natural-v1 is closest but not eligible (5%); fmt-len (0.015 away) beats fmt (0.030 away) by more than 0.01
    assert prompt == "natural-v1-fmt-len" and reason.startswith("rule: eligible")
    assert [t["eligible"] for t in table] == [False, True, True]


def test_the_rule_takes_fewer_failures_when_the_distances_differ_by_less_than_0_01():
    prompt, reason, _ = bg.choose({"natural-v1-fmt": {"n": 200, "hard": 3, "median": 1.055},
                                   "natural-v1-fmt-len": {"n": 200, "hard": 1, "median": 1.047}})
    assert prompt == "natural-v1-fmt-len" and "fewest" in reason
    prompt, _, _ = bg.choose({"natural-v1-fmt": {"n": 200, "hard": 3, "median": 1.058},
                              "natural-v1-fmt-len": {"n": 200, "hard": 1, "median": 1.040}})
    assert prompt == "natural-v1-fmt"


def test_the_rule_bound_is_3_of_200_and_without_an_eligible_prompt_the_fewest_failures():
    _, _, table = bg.choose({"a": {"n": 200, "hard": 3, "median": 1.0}, "b": {"n": 200, "hard": 4, "median": 1.06}})
    assert [t["eligible"] for t in table] == [True, False]
    prompt, reason, _ = bg.choose({"a": {"n": 200, "hard": 9, "median": 1.06}, "b": {"n": 200, "hard": 5, "median": 0.9}})
    assert prompt == "b" and reason.startswith("no prompt")


# ---------------------------------------------------------------------------------------------------- dataset rows

def source_table_rows():
    return [{"id": "dpc:a", "nl": "De Kamer stemt in met de motie over de begroting.", "en": "The House agrees with the "
             "motion on the budget of the ministry.", "hard_fails": [], "structure": [], "qe_en_nl": 1.5},
            {"id": "dpc:b", "nl": "Activiteit.\n\nDe raad besluit.", "en": "Activity.\n\nThe council decides.",
             "hard_fails": ["structure"], "structure": ["italic"], "qe_en_nl": 2.0}]


def test_unit_files_keep_the_source_columns_and_one_schema_with_empty_lists(tmp_path):
    import pyarrow.parquet as pq

    write_source(tmp_path / "hf", "dpc", "data", "train", source_table_rows())
    sources = bg.SourceRows(tmp_path / "hf")
    unit = {"name": "dpc/train-small-0000", "part": "dpc", "split": "train"}
    shard = {"id": ["dpc:a", "dpc:b"], "text_en": ["The House agrees with the motion on the budget.",
                                                   "Here is the translation:\n\nActivity.\n\nThe council decides."],
             "ended": [True, False], "tokens_out": [12, 15], "prompt": ["natural-v1-fmt"] * 2,
             "owner": ["vm/h0"] * 2, "region": ["us-east1"] * 2, "time": ["t"] * 2}
    bn.write_parquet([dict(zip(shard, v)) for v in zip(*shard.values())], tmp_path / "shard.parquet")
    n = bg.unit_file(unit, tmp_path / "shard.parquet", sources, lambda u: {"dpc:a": 11, "dpc:b": 7},
                     tmp_path / "out" / "dpc" / "train-small-0000.parquet")
    table = pq.read_table(tmp_path / "out" / "dpc" / "train-small-0000.parquet")
    assert n == 2 and table.column_names[:6] == ["id", "nl", "en", "hard_fails", "structure", "qe_en_nl"]
    import pyarrow as pa

    for name in ("gemma4_numbers_missing", "gemma4_hard_fails", "gemma4_structure"):
        kind = table.schema.field(name).type
        assert pa.types.is_list(kind) and pa.types.is_string(kind.value_type), name
    rows = table.to_pylist()
    assert rows[0]["gemma4_hard_fails"] == [] and rows[0]["gemma4_word_ratio"] == round(10 / 9, 4)
    assert rows[0]["tg_word_ratio"] == round(10 / 12, 4) and rows[0]["gemma4_tokens_nl"] == 11
    assert rows[1]["gemma4_hard_fails"] == ["not_ended", "len_ratio", "structure", "chatter"] and rows[1]["gemma4_chatter"]
    assert rows[0]["gemma4_model"] == bg.MODEL_ID and "{text}" in rows[0]["gemma4_prompt_text"]
    assert rows[1]["hard_fails"] == ["structure"]  # TranslateGemma's own checks unchanged


def test_a_shard_with_a_passage_of_another_part_is_an_error(tmp_path):
    with pytest.raises(SystemExit, match="is not a passage"):
        bg.published_rows({"name": "u", "part": "dpc", "split": "train"},
                          {"id": ["x"], "text_en": ["y"], "ended": [True], "tokens_out": [1], "prompt": ["natural-v1"],
                           "owner": ["o"], "region": ["r"], "time": ["t"]}, {}, {"x": 1})


def test_card_has_one_config_per_part_and_the_prompt(tmp_path):
    import yaml

    bg.main(["card", "--prompt", "natural-v1-fmt", "--out", str(tmp_path / "README.md")])
    text = (tmp_path / "README.md").read_text()
    meta = yaml.safe_load(text.split("---\n")[1])
    assert [c["config_name"] for c in meta["configs"]] == list(bg.PARTS)
    assert meta["configs"][2]["data_files"][0] == {"split": "train", "path": "handelingen_context/train-*"}
    assert "Do not add line breaks" in text and "TPU Research Cloud" in text


def test_report_counts_rows_and_ratios_per_part(tmp_path):
    rows = []
    for k in range(10):
        rows.append({"id": f"dpc:{k}", "nl": "een twee drie", "en": "one two three four", "en_gemma4": "one two three",
                     "hard_fails": [], "gemma4_hard_fails": ["structure"] if k == 0 else [],
                     "gemma4_structure": ["italic"] if k == 0 else [], "gemma4_ended": True,
                     "gemma4_word_ratio": 1.0, "tg_word_ratio": 0.75, "gemma4_r2_length_out": False,
                     "gemma4_prompt": "natural-v1-fmt", "gemma4_region": "us-east1"})
    bn.write_parquet(rows, tmp_path / "ds" / "dpc" / "train-small-0000.parquet")
    bg.main(["report", "--data", str(tmp_path / "ds"), "--out", str(tmp_path / "r.md")])
    result = json.loads((tmp_path / "r.json").read_text())
    st = result["stats"]["dpc"]
    assert result["rows"] == result["unique_ids"] == 10 and st["g4_hard"] == 0.1 and st["g4"]["p50"] == 1.0
    assert st["tg"]["under_0_9"] == 1.0 and st["g4"]["within_human_p5_p95"] == 1.0
    assert "dpc" in (tmp_path / "r.md").read_text()


# ---------------------------------------------------------------------------------------------------- prompts

def test_the_new_prompts_are_natural_v1_with_one_sentence_each():
    base = tr.PROMPTS["natural-v1"]["plain"]
    fmt = tr.PROMPTS["natural-v1-fmt"]["plain"]
    fmtlen = tr.PROMPTS["natural-v1-fmt-len"]["plain"]
    added = "Do not add line breaks, paragraphs, lists, headings, emphasis, Markdown or quotation marks that the source does not have. "
    assert fmt.replace(added, "") == base
    assert fmtlen.replace("Make the {target} about as long as a professional human translation would be; do not add "
                          "words or explanations. ", "") == fmt
    assert tr.MODELS["gemma4-31b-bt"]["turn"] == tr.MODELS["gemma4-31b"]["turn"]
    assert set(bg.PROMPTS) <= set(tr.PROMPTS)


@pytest.mark.skipif(not (GEMMA4_TOKENIZER / "tokenizer.json").exists(), reason="Gemma 4's tokenizer is not here")
def test_the_probes_prompts_fit_every_class_of_the_bt_model():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(GEMMA4_TOKENIZER))
    classes = tr.MODELS[bg.MODEL]["length_classes"]
    headroom = min(c["prefill_lengths"][-1] - c["max_source"] for c in classes.values())
    for name in bg.PROMPTS:
        prompts = tr.ModelPrompts(tok, tr.with_prompt(tr.MODELS[bg.MODEL], name), "nl", "en")
        prompts.check(["Dit is een zin."])
        assert len(prompts.encode("x")) - 1 <= headroom
