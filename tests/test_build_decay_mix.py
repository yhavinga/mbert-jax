"""scripts/build_decay_mix.py on a made-up FLAN and made-up web shares: filters, caps, hosts, token share and order."""

import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import build_decay_mix as bdm  # noqa: E402

NIV2_LANGUAGES = {
    "task100_english_qa": {"input": ["English"], "output": ["English"], "instruction": ["English"]},
    "task1133_xcsr_nl_commonsense_mc_classification": {"input": ["Dutch"], "output": ["Dutch"], "instruction": ["English"]},
    "task200_spanish_translation": {"input": ["English"], "output": ["Spanish"], "instruction": ["English"]},
    "task1612_sick_label_classification": {"input": ["English"], "output": ["English"], "instruction": ["English"]},
    "task595_mocha_answer_generation": {"input": ["English"], "output": ["English"], "instruction": ["English"]},
}
# (folder, shards, tasks with their rows per shard)
FAKE_FLAN = [
    ("cot_zsopt_data", 1, {"cot_esnli": 40, "cot_gsm8k": 12}),
    ("dialog_zsopt_data", 2, {"wiki_dialog": 30}),
    ("flan_zsopt_data", 6, {"glue/sst2:2.0.0": 25, "super_glue/cb:1.0.2": 1, "wmt16_translate/de-en:1.0.0": 20, "hellaswag:1.1.0": 5}),
    ("flan_zsnoopt_data", 4, {"glue/sst2:2.0.0": 25, "para_crawl_enes": 5}),
    ("niv2_zsopt_data", 5, {task: 15 for task in NIV2_LANGUAGES}),
    ("t0_zsopt_data", 4, {"amazon_polarity_Is_this_review": 40, "wiki_qa_Jeopardy_style": 3}),
    ("t0_zsnoopt_data", 4, {"amazon_polarity_Is_this_review": 40}),
]
KEPT_TASKS = {"cot_esnli", "cot_gsm8k", "wiki_dialog", "glue/sst2:2.0.0", "super_glue/cb:1.0.2", "task100_english_qa",
              "task1133_xcsr_nl_commonsense_mc_classification", "amazon_polarity_Is_this_review", "wiki_qa_Jeopardy_style"}


def _tokenizer():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    from mmbjax.training import add_answer_token

    words = [f"w{i}" for i in range(95)] + [f"u{i}" for i in range(20000)]
    vocab = {token: i for i, token in enumerate(["<unk>", "<s>", "</s>", "<pad>"] + words)}
    backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>", bos_token="<s>", eos_token="</s>",
                                        pad_token="<pad>", split_special_tokens=True)
    tokenizer.add_special_tokens({"mask_token": "<mask>"})
    add_answer_token(tokenizer)
    return tokenizer


class FakeShards:
    """The source interface of HubShards over made-up shards; every row has a unique word u<n>."""

    def __init__(self, max_length):
        rng = np.random.RandomState(0)
        self.tables_of = {}
        self.read = Counter()
        unique = 0
        for folder, shards, tasks in FAKE_FLAN:
            template_type = "zs_opt" if "_zsopt_" in folder else "zs_noopt"
            for shard in range(shards):
                rows = []
                for task, n in tasks.items():
                    for i in range(n):
                        unique += 1
                        inputs = f"u{unique} " + " ".join(f"w{w}" for w in rng.randint(0, 95, size=rng.randint(1, 8)))
                        targets = " ".join(f"w{w}" for w in rng.randint(0, 95, size=rng.randint(1, 4)))
                        # A few of each kind the filters must drop
                        if i == 1:
                            targets += " ⁇"
                        elif i == 2:
                            targets = "  "
                        elif i == 3:
                            inputs += " w1" * max_length
                        rows.append((inputs, targets, task, template_type))
                order = rng.permutation(len(rows))
                rows = [rows[i] for i in order]
                self.tables_of[f"{folder}/part.{shard}.parquet"] = pa.table(
                    {name: [row[c] for row in rows] for c, name in enumerate(bdm.FLAN_COLUMNS)})

    def shards(self, folder):
        return sorted((name for name in self.tables_of if name.startswith(folder + "/")),
                      key=lambda name: int(name.split(".")[-2]))

    def tables(self, names):
        for name in names:
            self.read[name] += 1
            yield name, self.tables_of[name]


def _web_files(tmp_path, documents=600):
    rng = np.random.RandomState(1)
    paths = []
    for part in range(2):
        texts = [f"doc{part}_{i} " + " ".join(f"w{w}" for w in rng.randint(0, 95, size=rng.randint(3, 40))) for i in range(documents // 2)]
        path = tmp_path / f"nl_part_{part:05d}.parquet"
        pq.write_table(pa.table({"text": texts, "url": [f"https://example.org/{part}/{i}" for i in range(len(texts))]}), path)
        paths.append(str(path))
    listed = tmp_path / "culturax_nl_files.txt"
    listed.write_text("\n".join(paths) + "\n")
    return str(listed)


def _build(tmp_path, rank, web_files, monkeypatch, cap=8, steps=20, buffer=16):
    # The trainer's buffer of 10,000 rows would hold all of a test's rows: a small one shuffles like a long mix does
    monkeypatch.setattr(bdm, "TRAIN_SHUFFLE_BUFFER", buffer)
    languages = tmp_path / "niv2_languages.json"
    languages.write_text(json.dumps({"tasks": NIV2_LANGUAGES}))
    out = tmp_path / f"out{rank}"
    args = bdm.parse_args(["--rank", str(rank), "--web-files", web_files, "--out", str(out), "--list", str(out / "files.txt"),
                           "--niv2-languages", str(languages), "--steps", str(steps), "--rows-per-host", "2",
                           "--max-length", "32", "--cap", str(cap)])
    source = FakeShards(32)
    return bdm.build(args, _tokenizer(), source), source


def _rows(path):
    return pq.read_table(path).to_pylist()


def test_interleave_keeps_the_token_share_at_every_web_document():
    rng = np.random.RandomState(0)
    web = rng.randint(3, 1024, size=3000).tolist()
    flan = rng.randint(5, 300, size=5000).tolist()
    sequence, used = bdm.interleave(web, flan, 0.1)
    assert [i for kind, i in sequence if kind == "web"] == list(range(3000))
    assert [i for kind, i in sequence if kind == "flan"] == list(range(used))
    web_total = flan_total = 0
    for k, (kind, i) in enumerate(sequence):
        web_total += web[i] if kind == "web" else 0
        flan_total += flan[i] if kind == "flan" else 0
        if kind == "web":
            continue
        if k + 1 == len(sequence) or sequence[k + 1][0] == "web":
            # After the FLAN behind a web document: at least the share, and over it by less than that last example
            assert flan_total >= 0.1 * (web_total + flan_total)
            assert flan_total - flan[i] < 0.1 * (web_total + flan_total - flan[i])
    assert flan_total / (web_total + flan_total) == pytest.approx(0.1, abs=300 / (web_total + flan_total))
    with pytest.raises(RuntimeError, match="raise --cap"):
        bdm.interleave(web, flan[:100], 0.1)


def test_hosts_take_disjoint_units_of_every_folder():
    for n_shards in (1, 2, 3, 4, 7, 18, 131):
        units = [bdm.host_units(n_shards, rank, 4) for rank in range(4)]
        if n_shards >= 4:
            shards = [index for host in units for index, residue in host]
            assert sorted(shards) == list(range(n_shards)) and all(residue is None for host in units for _, residue in host)
        else:
            assert all(host == [(index, rank) for index in range(n_shards)] for rank, host in enumerate(units))


def test_task_exclusions():
    exclusion = lambda folder, task: bdm.task_exclusion(folder, task, NIV2_LANGUAGES)  # noqa: E731
    assert exclusion("niv2_zsopt_data", "task1612_sick_label_classification") == "dumb_leak"
    assert exclusion("niv2_zsopt_data", "task595_mocha_answer_generation") == "gpt_generated"
    assert exclusion("flan_zsopt_data", "hellaswag:1.1.0") == "gpt_generated"
    assert exclusion("niv2_zsopt_data", "task200_spanish_translation") == "language"
    assert exclusion("flan_zsopt_data", "wmt16_translate/de-en:1.0.0") == "language"
    assert exclusion("flan_zsnoopt_data", "para_crawl_enes") == "language"
    assert exclusion("niv2_zsopt_data", "task1133_xcsr_nl_commonsense_mc_classification") is None
    assert exclusion("t0_zsopt_data", "amazon_polarity_Is_this_review") is None
    with pytest.raises(KeyError, match="not in --niv2-languages"):
        exclusion("niv2_zsopt_data", "task999_unknown")


def test_niv2_languages_of_the_experiment_keep_the_dutch_tasks_and_drop_sick():
    path = Path(__file__).resolve().parents[1] / "experiments" / "flan-decay" / "niv2_languages.json"
    languages = json.loads(path.read_text())["tasks"]
    assert len(languages) == 1556
    kept = {task for task in languages if bdm.task_exclusion("niv2_zsopt_data", task, languages) is None}
    dutch = {task for task in kept if "Dutch" in languages[task]["input"] + languages[task]["output"]}
    # The Dutch NIv2 tasks of the design (section 2.2, item 6)
    assert {task.split("_")[0] for task in dutch} == {"task1133", "task1543", "task1544", "task1545", "task1546", "task409"}
    assert not any(task.startswith(("task1612_", "task1613_", "task1614_", "task1615_")) for task in kept)
    assert len(kept) == 979


def test_builder_mix_has_the_token_share_and_the_web_documents_of_arm_w_in_its_order(tmp_path, monkeypatch):
    web_files = _web_files(tmp_path)
    manifest, _ = _build(tmp_path, 0, web_files, monkeypatch)
    path = Path(manifest["file"])
    assert Path(tmp_path / "out0" / "files.txt").read_text() == f"{path}\n"
    assert pq.read_schema(path).metadata[bdm.METADATA_KEY].decode() == manifest["build"] == path.stem.removeprefix("mix-")

    # What the trainer reads of the mix, in its order: arm W's first documents, in W's order, with FLAN in between
    tokenizer = _tokenizer()
    mixed = list(bdm.trainer_stream([str(path)], tokenizer))
    assert [row["position"] for row in mixed] == list(range(len(mixed)))
    web = [row["text"] for row in mixed if row["text"] is not None]
    arm_w = [row["text"] for row, _ in zip(bdm.trainer_stream(Path(web_files).read_text().split(), tokenizer), web)]
    assert web == arm_w and len(web) == manifest["web"]["documents"]
    # And not just the files' order: the trainer's shuffle moved the documents
    assert web != [row["text"] for row in pq.read_table(Path(web_files).read_text().split()[0]).to_pylist()][:len(web)]

    # A second start finds its build and keeps it
    again, source = _build(tmp_path, 0, web_files, monkeypatch)
    assert again == json.loads(json.dumps(manifest)) and not source.read

    # The share in the trainer's own token counts
    from mmbjax.training import TokenizeTransform

    transform = TokenizeTransform(tokenizer, 32, answer_token_id=tokenizer.convert_tokens_to_ids("<answer>"))
    tokens = Counter()
    for row in mixed:
        tokens["web" if row["text"] is not None else "flan"] += sum(len(piece["input_ids"]) for piece in transform.map(row))
    assert transform.skipped_too_long == 0
    assert tokens["flan"] == manifest["flan"]["tokens"] and tokens["web"] == manifest["web"]["tokens"]
    assert tokens["flan"] / sum(tokens.values()) == pytest.approx(0.1, abs=0.01)
    assert manifest["web"]["tokens"] >= bdm.web_target_tokens(20, 2, 32, 0.1, 0.2)
    assert manifest["steps_at_full_rows"] >= 20 * 1.2


def test_builder_filters_caps_and_gives_each_host_its_own_part_of_flan(tmp_path, monkeypatch):
    web_files = _web_files(tmp_path)
    cap = 8
    per_host = math.ceil(cap / 4)
    flan_rows, dropped = {}, Counter()
    for rank in range(4):
        manifest, source = _build(tmp_path, rank, web_files, monkeypatch, cap=cap, steps=6)
        rows = [row for row in _rows(manifest["file"]) if row["inputs"] is not None]
        flan_rows[rank] = {row["inputs"].split()[0] for row in rows}
        tasks = {row["source"].split("/", 2)[2] for row in rows}
        assert tasks <= KEPT_TASKS
        for row in rows:
            assert "⁇" not in row["inputs"] + row["targets"] and row["targets"].strip()
        for folder, report in manifest["flan"]["folders"].items():
            assert all(n <= per_host for n in report["rows_per_task"].values()), folder
            assert set(report["excluded_tasks"]) == {task for f, _, tasks in FAKE_FLAN if f == folder for task in tasks} - KEPT_TASKS
            dropped.update(report["dropped"])
        # Shards of the other hosts are never read; a folder of fewer shards than hosts is read by all
        for name in source.read:
            folder, part = name.split("/")
            shards = len(source.shards(folder))
            assert shards < 4 or int(part.split(".")[1]) % 4 == rank, name
        # Read no further than needed: amazon_polarity fills its cap in the first shard of a t0 folder of this host
        assert manifest["flan"]["folders"]["t0_zsnoopt_data"]["shards_read"] == 1
    for a in range(4):
        for b in range(a + 1, 4):
            assert not flan_rows[a] & flan_rows[b], (a, b)
    # Rows past a task's cap are not looked at, so each reason shows up somewhere, not in every folder
    assert {"unknown_character", "empty", "too_long", "dumb_leak", "gpt_generated", "language"} <= set(dropped)


def test_a_file_in_another_order_fails_the_check_of_the_trainers_order(tmp_path, monkeypatch):
    tokenizer = _tokenizer()
    n = 12_000
    # The real buffer of 10,000 rows, and more rows than it holds
    order = bdm.trainer_shuffle_order(n)
    assert not np.array_equal(order, np.arange(n))
    sequence = [("web", i) for i in range(n)]
    texts = [f"w{i % 95}" for i in range(n)]
    good, bad = tmp_path / "good.parquet", tmp_path / "bad.parquet"
    bdm.write_mix(good, sequence, texts, [], order, "x")
    bdm.verify_order(good, tokenizer, n)
    bdm.write_mix(bad, sequence, texts, [], np.arange(n), "x")
    with pytest.raises(RuntimeError, match="not the one the order was made for"):
        bdm.verify_order(bad, tokenizer, n)
