"""experiments/final-v2/stage_optout.py on a made-up kept/ with the opt-out pipeline's schema: the plan from its
manifest, the staged shares, the comparison with kept/test, the held-out file, and the trainer's loader on the shares
(FineWeb-2's columns text and url, the URL blocklist)."""
from __future__ import annotations

import base64
import concurrent.futures as cf
import datetime
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "final-v2"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import stage_optout as so  # noqa: E402
from test_multihost import MAX_LENGTH, VOCAB_WORDS, _make_tokenizer  # noqa: E402

HOSTS = 4
SEGMENT = 5
# Rows per kept/train file: one empty, some smaller than a round of segments, none a multiple of 4
TRAIN_ROWS = [23, 0, 7, 41, 2, 18]
TEST_ROWS = 9
GROUP = 6  # row groups of the made-up files, so segments cross them
MARKER = f"w{VOCAB_WORDS - 1}"


def _md5(path: Path) -> str:
    return base64.b64encode(hashlib.md5(path.read_bytes()).digest()).decode()


def _write(path: Path, texts, ids, urls):
    """A file with the columns of the pipeline's output, including the ones staging leaves out."""
    n = len(texts)
    table = pa.table({
        "text": pa.array(texts, pa.string()), "id": pa.array(ids, pa.string()), "dump": pa.array(["CC-MAIN-2020-05"] * n),
        "url": pa.array(urls, pa.string()), "date": pa.array(["2020-01-01"] * n), "language_score": pa.array([0.9] * n),
        "minhash_cluster_size": pa.array([1] * n, pa.int64()), "source_file": pa.array(["x"] * n), "source_row": pa.array(range(n), pa.int64()),
        "optout": pa.array([False] * n), "optout_reasons": pa.array([[] for _ in range(n)], pa.list_(pa.string())),
        "host_status": pa.array(["ok"] * n), "pii_replaced": pa.array([0] * n, pa.int32()),
        # Null in a whole file: the reason staging keeps only text and url
        "optout_checked_at": pa.array([None] * n, pa.timestamp("ms", tz="UTC")),
    })
    pq.write_table(table, path, row_group_size=GROUP)


def _corpus(tmp_path, shared=()):
    """A source directory with kept/train, kept/test and manifest.json. `shared` (k, how): test row k also in train,
    by its id or by its text."""
    src = tmp_path / "src"
    (src / "kept" / "train").mkdir(parents=True)
    (src / "kept" / "test").mkdir(parents=True)
    test_ids = [f"<urn:uuid:test-{k}>" for k in range(TEST_ROWS)]
    test_texts = [f"w120 w121 w{k}" for k in range(TEST_ROWS)]
    _write(src / "kept" / "test" / "test-000_00000-r00000.parquet", test_texts, test_ids, [f"https://t{k}.nl/" for k in range(TEST_ROWS)])
    outputs, texts_of = [], {}
    shared = list(shared)
    for f, n in enumerate(TRAIN_ROWS):
        name = f"kept/train/train-00{f}_00000-r00000.parquet"
        # Words only, unique per row, the row number last; the pages of www.hebban.nl carry MARKER
        texts = [f"w{f + 10} w{r // 100} w{r % 100}" + (f" {MARKER}" if r % 5 == 2 else "") for r in range(n)]
        ids = [f"<urn:uuid:{f}-{r}>" for r in range(n)]
        urls = [f"https://{'www.hebban.nl' if r % 5 == 2 else f'site{r % 3}.nl'}/{f}/{r}" for r in range(n)]
        if f == 3:
            for r, (k, how) in zip((4, 30), shared):
                if how == "id":
                    ids[r] = test_ids[k]
                else:
                    texts[r] = test_texts[k]
        _write(src / name, texts, ids, urls)
        texts_of[name] = texts
        outputs.append({"file": name, "rows": n, "bytes": (src / name).stat().st_size, "md5": _md5(src / name),
                        "crc32c": "x", "code": "code-test"})
    test_name = "kept/test/test-000_00000-r00000.parquet"
    outputs.insert(2, {"file": test_name, "rows": TEST_ROWS, "bytes": (src / test_name).stat().st_size,
                       "md5": _md5(src / test_name), "code": "code-test"})
    manifest = {"code": {"report_commit": "abc"}, "source": {"repo": "HuggingFaceFW/fineweb-2", "revision": "af9c133",
                                                             "config": "nld_Latn", "files": []},
                "optout_snapshot_utc": {"attempt_1": ["a", "b"]},
                "counts": {"split_kept": {"train": sum(TRAIN_ROWS), "test": TEST_ROWS}},
                "outputs": {"kept": outputs[::-1], "excluded": []}}
    (src / "manifest.json").write_text(json.dumps(manifest))
    return src, texts_of, test_ids, test_texts


def _plan(tmp_path, src):
    plan = so.make_plan((src / "manifest.json").read_bytes(), str(src), HOSTS, SEGMENT)
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    return plan, path


def _stage(tmp_path, monkeypatch, plan_path, rank, extra=()):
    monkeypatch.setattr(so.cf, "ProcessPoolExecutor", cf.ThreadPoolExecutor)
    so.main(["stage", "--plan", str(plan_path), "--rank", str(rank), "--out", str(tmp_path / f"w{rank}"),
             "--list", str(tmp_path / f"w{rank}.txt"), "--download-dir", str(tmp_path / "shm"), "--workers", "2", *extra])
    return (tmp_path / f"w{rank}.txt").read_text().split()


def test_the_plan_lists_kept_train_in_name_order_and_splits_it_as_the_corpus_runs(tmp_path):
    src, *_ = _corpus(tmp_path)
    plan, _ = _plan(tmp_path, src)
    assert [f["file"] for f in plan["files"]] == sorted(f"kept/train/train-00{f}_00000-r00000.parquet" for f in range(6))
    assert plan["test"]["rows"] == TEST_ROWS and plan["documents"] == sum(TRAIN_ROWS)
    assert plan["manifest_sha256"] == hashlib.sha256((src / "manifest.json").read_bytes()).hexdigest()
    for f in plan["files"]:
        assert sum(f["host_rows"]) == f["rows"] and max(f["host_rows"]) - min(f["host_rows"]) <= 1
    docs = [s["documents"] for s in plan["shares"]]
    assert sum(docs) == sum(TRAIN_ROWS) and max(docs) - min(docs) <= 1
    expected = so.share_ranges(TRAIN_ROWS, HOSTS, SEGMENT)
    assert [[sum(b - a for a, b in host[i]) for host in expected] for i in range(len(TRAIN_ROWS))] == [f["host_rows"] for f in plan["files"]]


def test_a_manifest_whose_files_do_not_add_up_to_its_counts_is_refused(tmp_path):
    src, *_ = _corpus(tmp_path)
    manifest = json.loads((src / "manifest.json").read_text())
    manifest["counts"]["split_kept"]["train"] += 1
    with pytest.raises(ValueError, match="files hold"):
        so.make_plan(json.dumps(manifest).encode(), str(src), HOSTS, SEGMENT)


def test_every_row_of_kept_train_is_staged_once_with_text_and_url_in_the_files_order(tmp_path, monkeypatch):
    src, texts_of, *_ = _corpus(tmp_path)
    plan, plan_path = _plan(tmp_path, src)
    staged = []
    for rank in range(HOSTS):
        listed = _stage(tmp_path, monkeypatch, plan_path, rank)
        expected = [so.share_file_name(f["file"], rank, HOSTS) for f in plan["files"] if f["host_rows"][rank]]
        assert [Path(p).name for p in listed] == expected
        assert all("train-001_" not in p for p in listed), "the empty file has no share"
        assert all(pq.read_schema(p).names == ["text", "url"] for p in listed)
        rows = [t for p in listed for t in pq.read_table(p).column("text").to_pylist()]
        share = json.loads((tmp_path / f"w{rank}" / "share.json").read_text())
        assert share["documents"] == len(rows) == plan["shares"][rank]["documents"]
        assert share["plan_sha256"] == hashlib.sha256(plan_path.read_bytes()).hexdigest() and share["overlaps"] == 0
        for p in listed:
            numbers = [int(t.split()[2][1:]) for t in pq.read_table(p).column("text").to_pylist()]
            assert numbers == sorted(numbers), "a host's rows of a file stay in the file's order"
        assert not list((tmp_path / "shm").glob("*.download")), "a download was left behind"
        staged += rows
    assert sorted(staged) == sorted(t for texts in texts_of.values() for t in texts)


def test_a_second_start_keeps_what_is_staged_and_a_changed_source_is_refused(tmp_path, monkeypatch):
    src, *_ = _corpus(tmp_path)
    plan, plan_path = _plan(tmp_path, src)
    listed = _stage(tmp_path, monkeypatch, plan_path, 2)
    before = {p: Path(p).stat().st_mtime_ns for p in listed}
    assert _stage(tmp_path, monkeypatch, plan_path, 2) == listed
    assert {p: Path(p).stat().st_mtime_ns for p in listed} == before, "a staged file was written again"

    victim = Path(listed[0])
    victim.unlink()
    source = src / plan["files"][0]["file"]
    data = pq.read_table(source)
    pq.write_table(data, source, row_group_size=GROUP + 1)  # same rows, other bytes
    with pytest.raises(ValueError, match="md5"):
        _stage(tmp_path, monkeypatch, plan_path, 2)
    assert not victim.exists() and not victim.with_name(victim.name + ".partial").exists()


def test_documents_of_kept_test_in_kept_train_are_found_and_left_out_of_the_held_out_file(tmp_path, monkeypatch):
    src, _, test_ids, test_texts = _corpus(tmp_path, shared=[(1, "id"), (5, "text")])
    plan, plan_path = _plan(tmp_path, src)
    found = []
    for rank in range(HOSTS):
        _stage(tmp_path, monkeypatch, plan_path, rank)
        found += json.loads((tmp_path / f"w{rank}" / "overlap.json").read_text())
    assert sorted((o["test_id"], o["by"], o["row"]) for o in found) == [(test_ids[1], "id", 4), (test_ids[5], "text", 30)]
    assert {o["file"] for o in found} == {"kept/train/train-003_00000-r00000.parquet"}

    so.main(["heldout", "--plan", str(plan_path), "--overlap", *[str(tmp_path / f"w{r}" / "overlap.json") for r in range(HOSTS)],
             "--download-dir", str(tmp_path / "shm"), "--out", str(tmp_path / "heldout.parquet"), "--summary", str(tmp_path / "heldout.json")])
    held = pq.read_table(tmp_path / "heldout.parquet")
    assert held.schema.names == ["text", "url", "id"]
    assert held.column("id").to_pylist() == [i for k, i in enumerate(test_ids) if k not in (1, 5)]
    assert held.column("text").to_pylist() == [t for k, t in enumerate(test_texts) if k not in (1, 5)]
    summary = json.loads((tmp_path / "heldout.json").read_text())
    assert (summary["test_documents"], summary["in_train"], summary["by_id"], summary["by_text"], summary["documents"]) == (9, 2, 1, 1, 7)
    assert summary["md5"] == hashlib.md5((tmp_path / "heldout.parquet").read_bytes()).hexdigest()

    with pytest.raises(ValueError, match="overlap files, the plan has"):
        so.main(["heldout", "--plan", str(plan_path), "--overlap", str(tmp_path / "w0" / "overlap.json"),
                 "--download-dir", str(tmp_path / "shm"), "--out", str(tmp_path / "h2.parquet"), "--summary", str(tmp_path / "h2.json")])


def test_a_subset_of_files_stages_only_those_for_a_smoke_test(tmp_path, monkeypatch):
    src, *_ = _corpus(tmp_path)
    plan, plan_path = _plan(tmp_path, src)
    subset = tmp_path / "subset.txt"
    subset.write_text("kept/train/train-003_00000-r00000.parquet\nkept/train/train-005_00000-r00000.parquet\n")
    listed = _stage(tmp_path, monkeypatch, plan_path, 1, ["--files", str(subset)])
    assert [Path(p).name for p in listed] == ["train-003_00000-r00000-w1of4.parquet", "train-005_00000-r00000-w1of4.parquet"]
    assert json.loads((tmp_path / "w1" / "share.json").read_text())["subset"] is True


def test_the_trainers_loader_reads_the_shares_and_leaves_out_hebban_nl(tmp_path, monkeypatch, capsys):
    from mmbjax.training import EpochTrackingDataLoader

    src, *_ = _corpus(tmp_path)
    _, plan_path = _plan(tmp_path, src)
    listed = _stage(tmp_path, monkeypatch, plan_path, 0)
    table = pa.concat_tables(pq.read_table(p) for p in listed)
    texts, urls = table.column("text").to_pylist(), table.column("url").to_pylist()
    tokenizer = _make_tokenizer()
    allowed = {tuple(tokenizer(t, add_special_tokens=False)["input_ids"]) for t, u in zip(texts, urls) if "hebban.nl" not in u}
    blocked = sum("hebban.nl" in u for u in urls)
    assert allowed and blocked
    rows = 2
    loader = EpochTrackingDataLoader(dataset_name="fineweb2-nl-optout", dataset_config="kept-train", tokenizer=tokenizer,
                                     max_length=MAX_LENGTH, batch_size=rows, seed=42, shuffle_buffer_size=8, data_files=listed,
                                     url_blocklist=("hebban.nl",))
    marker = tokenizer.convert_tokens_to_ids(MARKER)
    docs, iterator = [], iter(loader)
    while True:
        batch = next(iterator)
        if loader.epoch >= 1:
            break
        assert not (batch["input_ids"] == marker).any(), "a page of hebban.nl reached a batch"
        for ids, positions in zip(batch["input_ids"], batch["input_ids_positions"]):
            starts = [i for i, p in enumerate(positions) if p == 0 and ids[i] != tokenizer.pad_token_id]
            for a, b in zip(starts, starts[1:] + [len(ids)]):
                docs.append(tuple(int(t) for t in ids[a:b] if t not in (tokenizer.pad_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id)))
    assert len(docs) == len(set(docs)) and set(docs) <= allowed
    # Packing drops the last incomplete batch of a pass, nothing else
    assert len(docs) > len(allowed) - 4 * rows, (len(docs), len(allowed))
    assert f"{blocked:,} documents of blocked hosts left out of this pass" in capsys.readouterr().out


def test_the_memorization_sample_matches_each_held_out_document_to_one_of_the_share_by_length(tmp_path, monkeypatch):
    rng = np.random.RandomState(0)
    held = [" ".join(f"w{w}" for w in rng.randint(0, VOCAB_WORDS, size=n)) for n in rng.randint(3, 40, size=30)]
    pq.write_table(pa.table({"text": held, "id": [f"t{k}" for k in range(30)]}), tmp_path / "heldout.parquet")
    share = [" ".join(f"w{w}" for w in rng.randint(0, VOCAB_WORDS, size=n)) for n in rng.randint(3, 40, size=400)]
    files = []
    for k in range(3):
        path = tmp_path / f"s{k}.parquet"
        pq.write_table(pa.table({"text": share[k::3], "url": ["u"] * len(share[k::3])}), path)
        files.append(str(path))
    (tmp_path / "files.txt").write_text("\n".join(files) + "\n")
    import transformers

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *_a, **_k: _make_tokenizer())
    args = ["memsample", "--heldout", str(tmp_path / "heldout.parquet"), "--files-list", str(tmp_path / "files.txt"),
            "--documents", "10", "--pool", "200", "--min-tokens", "5", "--max-tokens", "30", "--out", str(tmp_path / "m.parquet"),
            "--summary", str(tmp_path / "m.json")]
    so.main(args)
    sample = pq.read_table(tmp_path / "m.parquet").to_pylist()
    test = [r for r in sample if r["split"] == "test"]
    train = [r for r in sample if r["split"] == "train"]
    assert len(test) == len(train) == 10
    assert all(5 <= r["tokens"] <= 30 for r in sample)
    assert {r["text"] for r in test} <= set(held) and {r["text"] for r in train} <= set(share)
    assert len({r["source"] for r in train}) == 10
    assert all(abs(a["tokens"] - b["tokens"]) <= 2 for a, b in zip(test, train))
    first = (tmp_path / "m.parquet").read_bytes()
    so.main(args)
    assert (tmp_path / "m.parquet").read_bytes() == first, "the same seed draws the same sample"


def test_match_lengths_takes_the_nearest_unused_length():
    assert so.match_lengths([10, 10, 10], [9, 10, 12, 30]) == [1, 0, 2]
    with pytest.raises(ValueError, match="fewer documents"):
        so.match_lengths([1, 2], [5])


def test_share_file_names_carry_the_rank():
    assert so.share_file_name("kept/train/train-004_00008-r00900.parquet", 3, 4) == "train-004_00008-r00900-w3of4.parquet"
    assert datetime  # imported for the timestamp column's type above
