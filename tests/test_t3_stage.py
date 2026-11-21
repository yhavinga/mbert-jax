"""experiments/t3/stage_t3.py on made-up CulturaX files: the files against a reference, the held-out keys, the scan,
the copies without the held-out, the proof (a scan that must find nothing), the trainer's loader on the result, and
the sample of the memorization test."""
from __future__ import annotations

import concurrent.futures as cf
import hashlib
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "t3"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import stage_t3 as st  # noqa: E402
from test_multihost import MAX_LENGTH, VOCAB_WORDS, _make_tokenizer  # noqa: E402

MARKER = f"w{VOCAB_WORDS - 1}"  # carried by the pages of www.hebban.nl
HELD = 12


def _write(path: Path, texts, urls, group=7):
    """A file with CulturaX's columns: text, timestamp, url, source."""
    n = len(texts)
    table = pa.table({"text": pa.array(texts, pa.string()), "timestamp": pa.array(["2020/01/01 00:00:00"] * n),
                      "url": pa.array(urls, pa.string()), "source": pa.array(["mC4"] * n)})
    pq.write_table(table, path, row_group_size=group, compression="snappy")


def _corpus(tmp_path):
    """A home with culturax_nl/nl/: file 0 starts with the held-out documents (as nl_part_00097), file 2 holds a held-out
    text with other whitespace; hebban.nl pages in every file; T2's reference and list."""
    home = tmp_path / "home"
    nl = home / "culturax_nl" / "nl"
    nl.mkdir(parents=True)
    held = [f"w{40 + k} w{41 + k}  w{k}" for k in range(HELD)]
    held[3] = "w7 " + MARKER + " w8"  # a held-out page of hebban.nl, as 6 of T2's 40,000
    held_urls = [f"https://{'www.hebban.nl' if k == 3 else f'h{k}.nl'}/{k}" for k in range(HELD)]
    pq.write_table(pa.table({"text": held, "timestamp": ["t"] * HELD, "url": held_urls, "source": ["mC4"] * HELD}),
                   tmp_path / "heldout.parquet")
    paths, texts_of = [], {}
    for f, n in enumerate((30, 17, 25)):
        texts = [f"w{f + 10} w{r // 10} w{r % 10}" + (f" {MARKER}" if r % 6 == 4 else "") for r in range(n)]
        urls = [f"https://{'www.hebban.nl' if r % 6 == 4 else f'site{r % 3}.nl'}/{f}/{r}" for r in range(n)]
        if f == 0:
            texts = held + texts
            urls = held_urls + urls
        if f == 2:
            texts[9] = " " + held[5].replace("  ", "\n") + " "  # the same words, other whitespace
        path = nl / f"nl_part_{f:05d}.parquet"
        _write(path, texts, urls)
        paths.append(str(path))
        texts_of[str(path)] = texts
    order = [paths[1], paths[0], paths[2]]
    (tmp_path / "list.txt").write_text("".join(f"{p}\n" for p in order))
    (tmp_path / "ref.tsv").write_text("".join(f"{p}\t{Path(p).stat().st_size}\t{hashlib.sha256(Path(p).read_bytes()).hexdigest()}\n" for p in order))
    return home, order, texts_of, held


@pytest.fixture
def threads(monkeypatch):
    # Processes would need the module importable in a fresh interpreter; the work is the same in threads
    monkeypatch.setattr(st.cf, "ProcessPoolExecutor", cf.ThreadPoolExecutor)


def _keys(tmp_path):
    md5 = hashlib.md5((tmp_path / "heldout.parquet").read_bytes()).hexdigest()
    st.main(["keys", "--heldout", str(tmp_path / "heldout.parquet"), "--md5", md5, "--out", str(tmp_path / "keys.json")])
    return json.loads((tmp_path / "keys.json").read_text())


def _prepare(tmp_path, threads):
    home, order, texts_of, held = _corpus(tmp_path)
    keys = _keys(tmp_path)
    st.main(["scan", "--keys", str(tmp_path / "keys.json"), "--list", str(tmp_path / "list.txt"), "--processes", "2",
             "--out", str(tmp_path / "scan.json")])
    st.main(["write", "--scan", str(tmp_path / "scan.json"), "--out-dir", str(home / "culturax_nl" / "t3"),
             "--list-out", str(home / "t3_files.txt"), "--summary", str(tmp_path / "write.json")])
    return home, order, texts_of, held, keys


def test_files_are_checked_against_the_reference_and_a_wrong_one_stays(tmp_path):
    home, order, *_ = _corpus(tmp_path)
    args = ["files", "--reference", str(tmp_path / "ref.tsv"), "--list", str(tmp_path / "list.txt"), "--home", str(home),
            "--expect", "3", "--out", str(tmp_path / "files.json")]
    st.main(args)
    assert [f["path"] for f in json.loads((tmp_path / "files.json").read_text())["files"]] == order
    before = Path(order[2]).read_bytes()
    Path(order[2]).write_bytes(before[:-1] + bytes([before[-1] ^ 1]))
    with pytest.raises(SystemExit, match="not the reference"):
        st.main(args)
    assert Path(order[2]).exists(), "a wrong file is reported, not deleted"
    Path(order[2]).unlink()
    with pytest.raises(SystemExit, match="--fetch fetches them"):
        st.main(args)


def test_keys_are_exact_and_normalized_and_refuse_another_file(tmp_path):
    _corpus(tmp_path)
    keys = _keys(tmp_path)
    assert keys["rows"] == HELD and len(keys["exact"]) == len(keys["normalized"]) == HELD
    assert keys["exact"][0] == hashlib.sha256("w40 w41  w0".encode()).hexdigest()
    assert keys["normalized"][0] == hashlib.sha256("w40 w41 w0".encode()).hexdigest()
    assert st.normalized_key(" a b\n\tc ") == st.normalized_key("a b c")
    assert st.normalized_key("é") == st.normalized_key("é"), "NFC"
    with pytest.raises(SystemExit, match="md5"):
        st.main(["keys", "--heldout", str(tmp_path / "heldout.parquet"), "--md5", "0" * 32, "--out", str(tmp_path / "k.json")])


def test_the_scan_finds_the_held_out_exactly_and_with_other_whitespace_and_counts_hebban(tmp_path, threads):
    home, order, texts_of, held = _corpus(tmp_path)
    _keys(tmp_path)
    st.main(["scan", "--keys", str(tmp_path / "keys.json"), "--list", str(tmp_path / "list.txt"), "--processes", "2",
             "--out", str(tmp_path / "scan.json")])
    scan = json.loads((tmp_path / "scan.json").read_text())
    by_path = {f["path"]: f for f in scan["files"]}
    assert [m["row"] for m in by_path[order[1]]["matches"]] == list(range(HELD))
    assert {m["kind"] for m in by_path[order[1]]["matches"]} == {"exact"}
    assert by_path[order[2]]["matches"] == [{"row": 9, "heldout_row": 5, "kind": "normalized"}]
    assert by_path[order[0]]["matches"] == []
    assert scan["matching_rows"] == HELD + 1 and scan["heldout_rows_found"] == HELD
    blocked = sum(1 for p in order for t in texts_of[p] if MARKER in t)
    assert scan["blocked"] == blocked and scan["url_blocklist"] == ["hebban.nl"]
    assert by_path[order[1]]["sha256"] == hashlib.sha256(Path(order[1]).read_bytes()).hexdigest()


def test_write_leaves_the_held_out_out_of_copies_and_the_proof_finds_nothing(tmp_path, threads):
    home, order, texts_of, held, keys = _prepare(tmp_path, threads)
    listed = (home / "t3_files.txt").read_text().split()
    assert listed == [order[0], str(home / "culturax_nl" / "t3" / Path(order[1]).name), str(home / "culturax_nl" / "t3" / Path(order[2]).name)]
    # The originals are untouched, the copies keep every column and the other rows in their order
    assert pq.read_table(order[1]).num_rows == len(texts_of[order[1]])
    copy = pq.read_table(listed[1])
    assert copy.column_names == ["text", "timestamp", "url", "source"]
    assert copy.column("text").to_pylist() == texts_of[order[1]][HELD:]
    assert pq.ParquetFile(listed[1]).metadata.row_group(0).num_rows == 7
    assert pq.read_table(listed[2]).column("text").to_pylist() == texts_of[order[2]][:9] + texts_of[order[2]][10:]
    rows = int((home / "t3_files.rows").read_text())
    assert rows == sum(len(t) for t in texts_of.values()) - HELD - 1
    assert (home / "t3_files.md5").read_text().strip() == hashlib.md5((home / "t3_files.txt").read_bytes()).hexdigest()
    st.main(["scan", "--keys", str(tmp_path / "keys.json"), "--list", str(home / "t3_files.txt"), "--processes", "2",
             "--require-none", "--out", str(tmp_path / "proof.json")])
    proof = json.loads((tmp_path / "proof.json").read_text())
    assert proof["matching_rows"] == 0 and proof["rows"] == rows and proof["keys_sha256"] == keys["keys_sha256"]
    # Every training text hashed against every held-out key, independently of the scan
    trained = {hashlib.sha256(t.encode()).hexdigest() for p in listed for t in pq.read_table(p).column("text").to_pylist()}
    assert not trained & set(keys["exact"])


def test_the_proof_fails_on_a_list_that_still_holds_the_held_out(tmp_path, threads):
    _corpus(tmp_path)
    _keys(tmp_path)
    with pytest.raises(SystemExit, match="still holds 13 rows"):
        st.main(["scan", "--keys", str(tmp_path / "keys.json"), "--list", str(tmp_path / "list.txt"), "--processes", "2",
                 "--require-none", "--out", str(tmp_path / "proof.json")])
    assert json.loads((tmp_path / "proof.json").read_text())["matching_rows"] == HELD + 1


def test_write_refuses_a_file_changed_since_the_scan(tmp_path, threads):
    home, order, *_ = _corpus(tmp_path)
    _keys(tmp_path)
    st.main(["scan", "--keys", str(tmp_path / "keys.json"), "--list", str(tmp_path / "list.txt"), "--processes", "2",
             "--out", str(tmp_path / "scan.json")])
    _write(Path(order[1]), ["other"], ["https://x.nl/"])
    with pytest.raises(SystemExit, match="changed since the scan"):
        st.main(["write", "--scan", str(tmp_path / "scan.json"), "--out-dir", str(home / "culturax_nl" / "t3"),
                 "--list-out", str(home / "t3_files.txt"), "--summary", str(tmp_path / "write.json")])
    assert not (home / "t3_files.txt").exists()


def test_the_trainers_loader_on_the_list_reads_no_held_out_document(tmp_path, threads, capsys):
    from mmbjax.training import EpochTrackingDataLoader

    home, order, texts_of, held, _ = _prepare(tmp_path, threads)
    listed = (home / "t3_files.txt").read_text().split()
    tokenizer = _make_tokenizer()

    def ids(text):
        return tuple(tokenizer(text, add_special_tokens=False)["input_ids"])

    held_ids = {ids(t) for t in held} | {ids(" ".join(t.split())) for t in held}
    loader = EpochTrackingDataLoader(dataset_name="uonlp/CulturaX", dataset_config="nl", tokenizer=tokenizer,
                                     max_length=MAX_LENGTH, batch_size=2, seed=42, shuffle_buffer_size=8, data_files=listed,
                                     url_blocklist=("hebban.nl",))
    docs, iterator = [], iter(loader)
    while True:
        batch = next(iterator)
        if loader.epoch >= 1:
            break
        for row, positions in zip(batch["input_ids"], batch["input_ids_positions"]):
            starts = [i for i, p in enumerate(positions) if p == 0 and row[i] != tokenizer.pad_token_id]
            for a, b in zip(starts, starts[1:] + [len(row)]):
                docs.append(tuple(int(t) for t in row[a:b] if t not in (tokenizer.pad_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id)))
    assert docs and not set(docs) & held_ids
    assert "documents of blocked hosts left out of this pass" in capsys.readouterr().out


def test_the_memorization_sample_takes_trained_documents_only(tmp_path, threads, monkeypatch):
    home, order, texts_of, held, _ = _prepare(tmp_path, threads)
    import transformers

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *_a, **_k: _make_tokenizer())
    args = ["memsample", "--heldout", str(tmp_path / "heldout.parquet"), "--keys", str(tmp_path / "keys.json"),
            "--files-list", str(home / "t3_files.txt"), "--documents", "5", "--pool", "60", "--min-tokens", "2",
            "--max-tokens", "30", "--out", str(tmp_path / "m.parquet"), "--summary", str(tmp_path / "m.json")]
    st.main(args)
    sample = pq.read_table(tmp_path / "m.parquet").to_pylist()
    test = [r for r in sample if r["split"] == "test"]
    train = [r for r in sample if r["split"] == "train"]
    assert len(test) == len(train) == 5
    assert {r["text"] for r in test} <= set(held) and all(r["source"].startswith("heldout:") for r in test)
    assert not {r["text"] for r in train} & set(held)
    assert not any(MARKER in r["text"] for r in train), "a page of the blocklist is not trained on"
    first = (tmp_path / "m.parquet").read_bytes()
    st.main(args)
    assert (tmp_path / "m.parquet").read_bytes() == first, "the same seed draws the same sample"


def test_the_memorization_sample_refuses_a_list_with_held_out_documents(tmp_path, threads, monkeypatch):
    _corpus(tmp_path)
    _keys(tmp_path)
    import transformers

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *_a, **_k: _make_tokenizer())
    with pytest.raises(ValueError, match="is a held-out document"):
        st.main(["memsample", "--heldout", str(tmp_path / "heldout.parquet"), "--keys", str(tmp_path / "keys.json"),
                 "--files-list", str(tmp_path / "list.txt"), "--documents", "3", "--pool", "1000", "--min-tokens", "2",
                 "--max-tokens", "30", "--out", str(tmp_path / "m.parquet"), "--summary", str(tmp_path / "m.json")])

