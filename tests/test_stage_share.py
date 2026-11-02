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


def _fake_corpus(tmp_path, monkeypatch, corpus, files):
    """A plan of these files and an hf_hub_download that copies them; returns the per-file texts."""
    spec = stage_share.CORPORA[corpus]
    source = tmp_path / "hub"
    texts = {}
    for name, n in files.items():
        path = source / f"{spec['prefix']}{name}{spec['suffix']}"
        path.parent.mkdir(parents=True, exist_ok=True)
        texts[path.relative_to(source).as_posix()] = [f"{name} document {i}" for i in range(n)]
        if spec["suffix"] == ".parquet":
            # Row groups of 7 rows: segments of 1000 don't line up with them
            pq.write_table(pa.table({"text": texts[path.relative_to(source).as_posix()], "id": list(range(n))}), path,
                           row_group_size=7)
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
    rows = [len(t) for t in texts.values()]
    ranges = stage_share.share_ranges(rows, 4, segment=5)
    plan = {"corpus": corpus, "repo": spec["repo"], "revision": spec["revision"], "hosts": 4, "segment_rows": 5,
            "documents": sum(rows), "files": [{"path": p, "rows": len(t)} for p, t in texts.items()],
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
