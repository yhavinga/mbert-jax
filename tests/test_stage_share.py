"""scripts/stage_share.py: the split into per-host shares and the staging of a share, on made-up files."""
import gzip
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import stage_share  # noqa: E402

# Row counts like those of FineWeb-2 nl's files: none a multiple of 4, some smaller than a segment round
ROWS = [4004000, 908166, 3999999, 1, 2, 3, 5, 4013, 1000, 999, 3001]


def _cover(ranges, rows):
    """Per file, how often each row is in some share."""
    seen = [np.zeros(n, dtype=np.int64) for n in rows]
    for host in ranges:
        for f, file_ranges in enumerate(host):
            for start, stop in file_ranges:
                seen[f][start:stop] += 1
    return seen


@pytest.mark.parametrize("hosts", [1, 2, 3, 4, 7])
def test_every_row_is_in_exactly_one_share(hosts):
    ranges = stage_share.share_ranges(ROWS, hosts)
    assert len(ranges) == hosts and all(len(host) == len(ROWS) for host in ranges)
    for seen in _cover(ranges, ROWS):
        assert (seen == 1).all()


@pytest.mark.parametrize("hosts", [2, 3, 4, 7])
def test_shares_differ_by_at_most_one_document_in_total_and_per_file(hosts):
    ranges = stage_share.share_ranges(ROWS, hosts)
    totals = [sum(stop - start for r in host for start, stop in r) for host in ranges]
    assert sum(totals) == sum(ROWS)
    assert max(totals) - min(totals) <= 1
    for f, n in enumerate(ROWS):
        per_file = [sum(stop - start for start, stop in host[f]) for host in ranges]
        assert set(per_file) <= {n // hosts, n // hosts + 1}, (n, per_file)


def test_segments_are_sorted_disjoint_and_at_most_a_segment_long():
    for host in stage_share.share_ranges(ROWS, 4, segment=1000):
        for file_ranges in host:
            assert all(0 < stop - start <= 1000 for start, stop in file_ranges)
            assert all(a[1] < b[0] for a, b in zip(file_ranges, file_ranges[1:])), "a host's segments must not touch"


def test_hosts_take_turns_so_no_host_is_always_first_in_a_round():
    # One file of 4 rounds of 4 segments: host k's segments start in rotation, not always k segments in
    (file_ranges,) = stage_share.share_ranges([16], 4, segment=1)[0]
    assert file_ranges == [(0, 1), (7, 8), (10, 11), (13, 14)]


def test_a_trend_within_files_is_spread_over_the_hosts():
    """What a block per host gets wrong on FineWeb-2: documents grow longer towards the end of a file."""
    rng = np.random.RandomState(0)
    # Hundreds of rounds per file, as FineWeb-2's files of 0.9 to 4 million rows have in segments of 1,000
    files = [np.arange(n) + rng.randint(0, 50, size=n) for n in (400000, 333333, 250001)]

    def spread(segment):
        ranges = stage_share.share_ranges([len(f) for f in files], 4, segment=segment)
        tokens = [sum(int(f[start:stop].sum()) for f, r in zip(files, host) for start, stop in r) for host in ranges]
        return (max(tokens) - min(tokens)) / np.mean(tokens)

    block, segments = spread(10**9), spread(1000)
    assert block > 0.3, "a block per host should show the trend, or this tests nothing"
    assert segments < 0.01 and segments < block / 100, (block, segments)


def test_the_split_is_deterministic_and_matches_the_committed_plans():
    for corpus in ("fineweb2", "mc4"):
        plan = stage_share.load_plan(corpus)
        rows = [f["rows"] for f in plan["files"]]
        assert sum(rows) == plan["documents"]
        ranges = stage_share.share_ranges(rows, plan["hosts"])
        assert ranges == stage_share.share_ranges(rows, plan["hosts"])
        totals = [sum(stop - start for r in host for start, stop in r) for host in ranges]
        assert totals == [share["documents"] for share in plan["shares"]]
    assert stage_share.load_plan("fineweb2")["documents"] == 147_301_270
    assert stage_share.load_plan("mc4")["documents"] == 64_545_895


def test_take_rows_cuts_a_batch_at_the_range_edges():
    texts = list(range(100, 110))  # rows 100..109 of a file
    ranges = [(95, 102), (104, 105), (108, 120), (130, 140)]
    assert stage_share.take_rows(ranges, 100, texts) == [[100, 101], [104], [108, 109]]


def _sizes(n):
    """Cluster sizes of a made-up file: 1 to 200 in turn, so every weight class, its edges and sizes past the last key."""
    return [(i * 119) % 200 + 1 for i in range(n)]


def _lengths(n):
    """Tokens per document of a made-up file, as the lengths parquets of count_tokens.py give them."""
    return [i % 50 + 1 for i in range(n)]


def _fake_corpus(tmp_path, monkeypatch, corpus, files):
    """A plan of these files, their lengths parquets and an hf_hub_download that copies them; returns the per-file texts."""
    spec = stage_share.CORPORA[corpus]
    source = tmp_path / "hub"
    texts = {}
    for name, n in files.items():
        path = source / f"{spec['prefix']}{name}{spec['suffix']}"
        path.parent.mkdir(parents=True, exist_ok=True)
        texts[path.relative_to(source).as_posix()] = [f"{name} document {i}" for i in range(n)]
        if spec["suffix"] == ".parquet":
            # Row groups of 7 rows: segments of 1000 don't line up with them
            pq.write_table(pa.table({"text": texts[path.relative_to(source).as_posix()], "id": list(range(n)),
                                     "minhash_cluster_size": _sizes(n)}), path, row_group_size=7)
        else:
            with gzip.open(path, "wb") as f:
                for i, text in enumerate(texts[path.relative_to(source).as_posix()]):
                    f.write(json.dumps({"text": text, "url": f"u{i}", "timestamp": "t"}).encode() + b"\n")

    def fake_download(repo, filename, repo_type, revision, local_dir):
        assert (repo, repo_type, revision) == (spec["repo"], "dataset", spec["revision"])
        target = Path(local_dir) / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(source / filename, target)
        return str(target)

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    monkeypatch.setattr(stage_share, "PLAN_DIR", tmp_path / "plans")
    monkeypatch.setattr(stage_share, "SEGMENT_ROWS", 5)
    (tmp_path / "plans").mkdir()
    (tmp_path / "lengths").mkdir()
    for p, t in texts.items():
        pq.write_table(pa.table({"text": pa.array(_lengths(len(t)), pa.int32())}),
                       tmp_path / "lengths" / f"{stage_share.output_stem(p)}.parquet")
    rows = [len(t) for t in texts.values()]
    ranges = stage_share.share_ranges(rows, 4, segment=5)
    plan = {"corpus": corpus, "repo": spec["repo"], "revision": spec["revision"], "hosts": 4, "segment_rows": 5,
            "documents": sum(rows), "tokens_note": "made up",
            "files": [{"path": p, "rows": len(t), "tokens": sum(_lengths(len(t)))} for p, t in texts.items()],
            "shares": [{"rank": k, "documents": sum(b - a for r in host for a, b in r), "tokens": 0}
                       for k, host in enumerate(ranges)]}
    (tmp_path / "plans" / f"{corpus}_plan.json").write_text(json.dumps(plan))
    return texts


@pytest.mark.parametrize("corpus", ["fineweb2", "mc4"])
def test_staged_shares_hold_every_document_once_in_the_file_order(tmp_path, monkeypatch, corpus):
    texts = _fake_corpus(tmp_path, monkeypatch, corpus, {"a": 53, "b": 4, "c": 21})
    staged = []
    for rank in range(4):
        # One file at a time in this process: the fake download is patched here only
        monkeypatch.setattr(stage_share.cf, "ProcessPoolExecutor", stage_share.cf.ThreadPoolExecutor)
        stage_share.main(["stage", "--corpus", corpus, "--rank", str(rank), "--out", str(tmp_path / f"w{rank}"),
                          "--list", str(tmp_path / f"w{rank}.txt"), "--workers", "1", "--download-dir", str(tmp_path / "shm")])
        assert not list((tmp_path / "shm").iterdir()), "a download was left behind"
        listed = (tmp_path / f"w{rank}.txt").read_text().split()
        assert [Path(p).name for p in listed] == [f"{n}-w{rank}of4.parquet" for n in ("a", "b", "c")]
        assert pq.read_schema(listed[0]).names == ["text"]
        rows = [t for p in listed for t in pq.read_table(p).column("text").to_pylist()]
        summary = json.loads((tmp_path / f"w{rank}" / "share.json").read_text())
        assert summary["documents"] == len(rows)
        # A host's rows of a file stay in the file's order
        for name in ("a", "b", "c"):
            numbers = [int(t.split()[-1]) for t in rows if t.startswith(f"{name} ")]
            assert numbers == sorted(numbers)
        staged.append(rows)
        assert not list((tmp_path / f"w{rank}").glob("*.download")), "a download was left on disk"
    everything = [t for rows in staged for t in rows]
    assert sorted(everything) == sorted(t for file_texts in texts.values() for t in file_texts)
    assert max(map(len, staged)) - min(map(len, staged)) <= 1


def test_a_staged_file_is_kept_and_a_source_with_other_rows_is_refused(tmp_path, monkeypatch):
    texts = _fake_corpus(tmp_path, monkeypatch, "fineweb2", {"a": 30})
    monkeypatch.setattr(stage_share.cf, "ProcessPoolExecutor", stage_share.cf.ThreadPoolExecutor)
    args = ["stage", "--corpus", "fineweb2", "--rank", "1", "--out", str(tmp_path / "w1"), "--list", str(tmp_path / "w1.txt")]
    stage_share.main(args)
    staged = tmp_path / "w1" / "a-w1of4.parquet"
    before = staged.stat().st_mtime_ns
    stage_share.main(args)
    assert staged.stat().st_mtime_ns == before, "a second start staged a finished file again"

    plan_file = tmp_path / "plans" / "fineweb2_plan.json"
    plan = json.loads(plan_file.read_text())
    plan["files"][0]["rows"] = 31
    for k, host in enumerate(stage_share.share_ranges([31], 4, segment=5)):
        plan["shares"][k]["documents"] = sum(b - a for r in host for a, b in r)
    plan_file.write_text(json.dumps(plan))
    staged.unlink()
    with pytest.raises(ValueError, match="has 30 rows, the plan 31"):
        stage_share.main(args)
    assert len(texts["data/nld_Latn/train/a.parquet"]) == 30


# ------------------------------------------------------------------------------------------------ rehydrate

W3 = stage_share.REHYDRATION_WEIGHTS[3]


def test_a_cluster_size_takes_the_weight_of_the_largest_key_at_or_below_it():
    assert W3 == {1: 1, 4: 2, 10: 3, 45: 2, 124: 1}, "nld_Latn in weights/up_to_03_reps.json of FineWeb-2"
    table = stage_share.expand_weights(W3)
    assert len(table) == 125
    assert table[1:4] == [1] * 3 and table[4:10] == [2] * 6 and table[10:45] == [3] * 35
    assert table[45:124] == [2] * 79 and table[124] == 1
    sizes = np.array([1, 2, 3, 4, 9, 10, 44, 45, 123, 124, 125, 1000, 10**6])
    assert stage_share.cluster_weights(sizes, W3).tolist() == [1, 1, 1, 2, 2, 3, 3, 2, 2, 1, 1, 1, 1]
    with pytest.raises(ValueError, match="a cluster size of -1"):
        stage_share.cluster_weights(np.array([3, -1]), W3)


def test_a_documents_draw_is_the_same_on_whichever_host_and_on_a_second_run():
    rows = 12345
    weights = stage_share.cluster_weights(np.array(_sizes(rows)), W3)
    decided = []
    for hosts, segment in ((1, 1000), (2, 1000), (4, 1000), (4, 7), (7, 3)):
        keep = np.zeros(rows, dtype=bool)
        for (ranges,) in stage_share.share_ranges([rows], hosts, segment=segment):
            mine = stage_share.keep_rows(42, 5, rows, ranges, np.concatenate([weights[a:b] for a, b in ranges]), 3)
            for (a, b), part in zip(ranges, np.split(mine, np.cumsum([b - a for a, b in ranges])[:-1])):
                keep[a:b] = part
        decided.append(keep)
    for keep in decided[1:]:
        assert (keep == decided[0]).all()
    assert (stage_share.keep_rows(42, 5, rows, [(0, rows)], weights, 3) == decided[0]).all()
    # Another seed or another file draws again
    assert (stage_share.keep_rows(43, 5, rows, [(0, rows)], weights, 3) != decided[0]).any()
    assert (stage_share.keep_rows(42, 6, rows, [(0, rows)], weights, 3) != decided[0]).any()
    # Kept with probability w / 3: weight 3 always, 1 and 2 about a third and two thirds of the time
    assert decided[0][weights == 3].all()
    for w in (1, 2):
        assert abs(decided[0][weights == w].mean() - w / 3) < 0.03
    with pytest.raises(ValueError, match="more than --max-reps 2"):
        stage_share.keep_rows(42, 5, rows, [(0, rows)], weights, 2)


def test_a_hosts_cluster_sizes_are_read_in_the_order_of_its_staged_rows(tmp_path):
    rows = 9001
    path = tmp_path / "source.parquet"
    pq.write_table(pa.table({"text": [f"document {i}" for i in range(rows)],
                             "minhash_cluster_size": list(range(1, rows + 1))}), path, row_group_size=7)
    source = pq.ParquetFile(path)
    # The call of the stage step: the sizes must be those of the rows it staged, in that order
    for (ranges,) in stage_share.share_ranges([rows], 4):
        assert len(ranges) > 1, "a host should have several segments, or this tests little"
        sizes = stage_share._rows(source, ranges, "minhash_cluster_size").to_numpy()
        assert sizes.tolist() == [i + 1 for a, b in ranges for i in range(a, b)]


class _Response:
    def __init__(self, status_code, content):
        self.status_code, self.content = status_code, content

    def raise_for_status(self):
        pass


def test_a_source_is_read_by_one_range_request_per_row_group_of_a_column(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    path = tmp_path / "source.parquet"
    # Uncompressed text of 4 MB, so that reading it would show in the bytes fetched
    pq.write_table(pa.table({"text": [rng.bytes(2000).hex() for _ in range(1000)], "minhash_cluster_size": list(range(1000))}),
                   path, row_group_size=100, compression="none")
    data = path.read_bytes()
    url, spans = "https://cdn.example/file?Signature=x", []

    def fake_backoff(method, address, headers, retry_on_status_codes):
        assert (method, address) == ("GET", url) and 429 in retry_on_status_codes
        start, stop = map(int, headers["Range"].removeprefix("bytes=").split("-"))
        spans.append((start, stop))
        return _Response(206, data[start:stop + 1])

    import huggingface_hub.utils

    monkeypatch.setattr(huggingface_hub.utils, "http_backoff", fake_backoff)
    source = pq.ParquetFile(stage_share._RangeFile(url, len(data)))
    footer = len(spans)
    sizes = stage_share._rows(source, [(200, 300), (700, 1000)], "minhash_cluster_size")
    assert sizes.to_pylist() == list(range(200, 300)) + list(range(700, 1000))
    assert len(spans) - footer == 4, "one request per row group read: 2, 7, 8 and 9"
    assert sum(stop + 1 - start for start, stop in spans) < len(data) / 10

    monkeypatch.setattr(huggingface_hub.utils, "http_backoff", lambda *args, **kwargs: _Response(200, data))
    with pytest.raises(IOError, match="gave status 200"):
        stage_share._RangeFile(url, len(data)).read(10)


def _local_source(tmp_path):
    spec = stage_share.CORPORA["fineweb2"]

    def open_source(repo, revision, path):
        assert (repo, revision) == (spec["repo"], spec["revision"])
        return open(tmp_path / "hub" / path, "rb")

    return open_source


def _rehydrate(tmp_path, rank, out, *extra):
    stage_share.main(["rehydrate", "--corpus", "fineweb2", "--rank", str(rank), "--staged", str(tmp_path / f"w{rank}"),
                      "--out", str(tmp_path / out), "--list", str(tmp_path / f"{out}.txt"), "--lengths",
                      str(tmp_path / "lengths"), "--max-reps", "3", "--readers", "2", "--workers", "1", *extra])


def _stage(tmp_path, rank):
    stage_share.main(["stage", "--corpus", "fineweb2", "--rank", str(rank), "--out", str(tmp_path / f"w{rank}"),
                      "--list", str(tmp_path / f"w{rank}.txt"), "--workers", "1"])


def test_rehydrated_shares_keep_each_document_by_its_own_draw_whichever_host_has_it(tmp_path, monkeypatch):
    # a has several segments per host, b fewer rows than hosts
    texts = _fake_corpus(tmp_path, monkeypatch, "fineweb2", {"a": 9001, "b": 3, "c": 2100})
    monkeypatch.setattr(stage_share.cf, "ProcessPoolExecutor", stage_share.cf.ThreadPoolExecutor)
    monkeypatch.setattr(stage_share, "open_source", _local_source(tmp_path))
    plan = stage_share.load_plan("fineweb2")
    tokens = _lengths(9001)  # of a document by its row
    expected = {}
    for i, (path, file_texts) in enumerate(texts.items()):
        n = len(file_texts)
        keep = stage_share.keep_rows(42, i, n, [(0, n)], stage_share.cluster_weights(np.array(_sizes(n)), W3), 3)
        expected[path] = [t for t, k in zip(file_texts, keep) if k]
    kept = []
    for rank in range(4):
        _stage(tmp_path, rank)
        _rehydrate(tmp_path, rank, f"rh{rank}")
        listed = (tmp_path / f"rh{rank}.txt").read_text().split()
        assert listed == [str(tmp_path / f"rh{rank}" / f"{n}-w{rank}of4.parquet") for n in ("a", "b", "c")]
        assert pq.read_schema(listed[0]).names == ["text"]
        per_file = [pq.read_table(p).column("text").to_pylist() for p in listed]
        summary = json.loads((tmp_path / f"rh{rank}" / "share.json").read_text())
        assert summary["rows_in"] == plan["shares"][rank]["documents"]
        assert summary["rows_kept"] == sum(map(len, per_file))
        assert summary["tokens_kept"] == sum(tokens[int(t.split()[-1])] for f in per_file for t in f)
        assert sum(c["rows_kept"] for c in summary["by_weight"].values()) == summary["rows_kept"]
        assert sum(c["rows_in"] for c in summary["by_weight"].values()) == summary["rows_in"]
        assert all(f["segments_checked"] for f in summary["per_file"] if f["rows_in"]), "a file was not checked"
        kept.append(per_file)
    for f, path in enumerate(texts):
        # Each kept document on one host, in the file's order there, and only those of their own draw
        assert sorted(t for host in kept for t in host[f]) == sorted(expected[path])
        for host in kept:
            numbers = [int(t.split()[-1]) for t in host[f]]
            assert numbers == sorted(numbers)
    kept_share = sum(map(len, expected.values())) / sum(map(len, texts.values()))
    assert 0.45 < kept_share < 0.65, kept_share


def test_a_rehydrated_file_is_kept_and_a_staged_share_out_of_line_with_the_source_writes_nothing(tmp_path, monkeypatch):
    _fake_corpus(tmp_path, monkeypatch, "fineweb2", {"a": 9001, "c": 2100})
    monkeypatch.setattr(stage_share.cf, "ProcessPoolExecutor", stage_share.cf.ThreadPoolExecutor)
    monkeypatch.setattr(stage_share, "open_source", _local_source(tmp_path))
    _stage(tmp_path, 1)

    # A smoke test of the first file writes that file, but no list and no summary
    _rehydrate(tmp_path, 1, "rh1", "--max-files", "1")
    assert [p.name for p in (tmp_path / "rh1").iterdir()] == ["a-w1of4.parquet"]
    assert not (tmp_path / "rh1.txt").exists()
    written = tmp_path / "rh1" / "a-w1of4.parquet"
    before = written.stat().st_mtime_ns
    _rehydrate(tmp_path, 1, "rh1")
    assert written.stat().st_mtime_ns == before, "a second start wrote a finished file again"
    assert (tmp_path / "rh1.txt").read_text().split() == [str(written), str(tmp_path / "rh1" / "c-w1of4.parquet")]

    staged = tmp_path / "w1" / "a-w1of4.parquet"
    rows = pq.read_table(staged).column("text").to_pylist()
    pq.write_table(pa.table({"text": rows[1:2] + rows[:1] + rows[2:]}), staged)
    with pytest.raises(ValueError, match="rows 0-1000 of .*a-w1of4.parquet are not rows 1000-2000 of data/nld_Latn/train/a.parquet"):
        _rehydrate(tmp_path, 1, "rh1b")
    assert not list((tmp_path / "rh1b").iterdir()), "a file was written while a share did not line up"

    pq.write_table(pa.table({"text": rows[:-1]}), staged)
    with pytest.raises(ValueError, match=f"has {len(rows) - 1} rows, rank 1's share of data/nld_Latn/train/a.parquet {len(rows)}"):
        _rehydrate(tmp_path, 1, "rh1b")
    with pytest.raises(ValueError, match="needs paths of its own"):
        _rehydrate(tmp_path, 1, "w1")
