"""experiments/gptnl/gptnl_mix.py: GPT-NL counted, planned and mixed with CulturaX nl into per-host shares, on made-up
files and a made-up tokenizer (a token per word)."""
import concurrent.futures as cf
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import xxhash

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "gptnl"))

import gptnl_mix  # noqa: E402
import stage_share  # noqa: E402  (on the path through gptnl_mix)


class FakeTokenizer:
    """A token per word, with the trainer's arguments only."""

    def __init__(self):
        self.calls = []

    def __call__(self, texts, truncation, return_tensors, add_special_tokens, split_special_tokens):
        assert (truncation, return_tensors, add_special_tokens, split_special_tokens) == (False, None, False, True)
        self.calls.append(list(texts))
        return {"input_ids": [[7] * len(t.split()) for t in texts]}


class _InProcess(cf.ThreadPoolExecutor):
    """The pools in threads, as the fakes are patched in this process only."""

    def __init__(self, max_workers=None, mp_context=None, initializer=None, initargs=()):
        super().__init__(max_workers, initializer=initializer, initargs=initargs)


def _words(rng, n):
    return " ".join(f"w{rng.integers(1000)}" for _ in range(n))


def _world(tmp_path, monkeypatch):
    """GPT-NL with a file or two per included collection and one per excluded one, 8 CulturaX files, their counts of
    count_tokens.py, and the Hub, the bucket and the tokenizer faked."""
    rng = np.random.default_rng(0)
    hub = tmp_path / "hub"
    culturax = {}
    for f in range(8):
        # A first word of its own, so that no text is in two places by chance
        culturax[gptnl_mix.culturax_path(f)] = [f"c{f}.{i} " + _words(rng, (i * 7 + f) % 20) for i in range(60)]
    gptnl = {}
    for c, folder in enumerate(gptnl_mix.INCLUDED):
        for f in range(1 + c % 2):
            gptnl[f"{folder}/{folder}_{f + 1:04d}.parquet"] = [
                (f"g{c}.{f}.{i} " + _words(rng, (i * 5 + c) % 12), ("nl", "nl", "nl", "en", "fy", None)[(i + c) % 6])
                for i in range(10 + (c * 7 + f * 3) % 15)]
    first = [p for p in gptnl if p.startswith(gptnl_mix.INCLUDED[0] + "/")][0]
    second = [p for p in gptnl if p.startswith(gptnl_mix.INCLUDED[1] + "/")][0]
    third = [p for p in gptnl if p.startswith(gptnl_mix.INCLUDED[2] + "/")][0]
    fourth = [p for p in gptnl if p.startswith(gptnl_mix.INCLUDED[3] + "/")][0]
    # Rows the plan must leave out: a text twice, an empty one and one of CulturaX's
    gptnl[first][0] = ("de zelfde tekst twee keer", "nl")
    gptnl[second][0] = ("de zelfde tekst twee keer", "nl")
    gptnl[third][0] = ("", "nl")
    gptnl[fourth][0] = (culturax[gptnl_mix.culturax_path(5)][3], "nl")
    for folder in gptnl_mix.EXCLUDED:
        gptnl[f"{folder}/{folder}_0001.parquet"] = [("an English text", "en")] * 5
    for k, (path, rows) in enumerate(gptnl.items()):
        target = hub / "gptnl" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        columns = {"id": [f"{path}#{i}" for i in range(len(rows))], "text": [t for t, _ in rows],
                   "language": [lang for _, lang in rows], "license": ["cc-by"] * (len(rows) - 1) + ["public-domain"]}
        # The order of the columns differs between GPT-NL's files
        order = ["id", "text", "language", "license"] if k % 2 else ["language", "license", "id", "text"]
        pq.write_table(pa.table({name: columns[name] for name in order}), target)
    counts = tmp_path / "cx_counts"
    counts.mkdir()
    (tmp_path / "cx_lengths").mkdir()
    for path, texts in culturax.items():
        target = hub / "culturax" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        # Row groups of 7 rows, so a host's rows are in several
        pq.write_table(pa.table({"text": texts, "url": ["u"] * len(texts)}), target, row_group_size=7)
        n = np.array([len(t.split()) for t in texts], dtype=np.uint32)
        np.savez(counts / f"{stage_share.output_stem(path)}.npz", counts=n,
                 utf8_bytes=np.array([len(t.encode()) for t in texts], dtype=np.uint32),
                 xxh64=np.array([xxhash.xxh64_intdigest(t.encode()) for t in texts], dtype=np.uint64))
        pq.write_table(pa.table({"text": pa.array(n, pa.int32())}), tmp_path / "cx_lengths" / f"{stage_share.output_stem(path)}.parquet")

    def list_repo_tree(self, repo, repo_type, revision, recursive):
        assert (repo, repo_type, revision, recursive) == (gptnl_mix.GPTNL["repo"], "dataset", gptnl_mix.GPTNL["revision"], True)
        folders = sorted({p.split("/")[0] for p in gptnl})
        return [SimpleNamespace(path=f) for f in folders] + [SimpleNamespace(path="README.md", size=10)] + [
            SimpleNamespace(path=p, size=(hub / "gptnl" / p).stat().st_size) for p in gptnl]

    def open_source(repo, revision, path):
        where = {(gptnl_mix.GPTNL["repo"], gptnl_mix.GPTNL["revision"]): "gptnl",
                 (gptnl_mix.CULTURAX["repo"], gptnl_mix.CULTURAX["revision"]): "culturax"}[(repo, revision)]
        return open(hub / where / path, "rb")

    # The trainer's held-out file: documents of a CulturaX file of host 3
    heldout = culturax[gptnl_mix.culturax_path(3)][10:15]
    pq.write_table(pa.table({"text": heldout}), tmp_path / "heldout.parquet")
    bucket = tmp_path / "gcs"
    bucket.mkdir()

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub.HfApi, "list_repo_tree", list_repo_tree)
    monkeypatch.setattr(stage_share, "open_source", open_source)
    monkeypatch.setattr(gptnl_mix, "load_tokenizer", lambda name=gptnl_mix.TOKENIZER: FakeTokenizer())
    monkeypatch.setattr(gptnl_mix, "_gcs_stems", lambda gcs: {p.stem for p in bucket.glob("*.parquet")})
    monkeypatch.setattr(gptnl_mix, "_upload", lambda local, gcs: shutil.copy(local, bucket / Path(local).name))
    monkeypatch.setattr(gptnl_mix.cf, "ProcessPoolExecutor", _InProcess)
    monkeypatch.setattr(gptnl_mix, "CULTURAX", {**gptnl_mix.CULTURAX, "files": 8})
    monkeypatch.setattr(gptnl_mix, "CULTURAX_FILES", ((0, 4), (1, 5), (2, 6), (3, 7)))
    monkeypatch.setattr(gptnl_mix, "CULTURAX_HOST_DOCUMENTS", (120, 120, 120, 120))
    monkeypatch.setattr(gptnl_mix, "HOST_TOKENS", 800)
    monkeypatch.setattr(gptnl_mix, "SHARE_FILES", 3)
    monkeypatch.setattr(gptnl_mix, "PLAN_PATH", tmp_path / "plan.json")
    return SimpleNamespace(gptnl=gptnl, culturax=culturax, bucket=bucket, duplicate=second, empty=third,
                           in_culturax=fourth, heldout=heldout)


def _count(tmp_path, rank, *extra):
    gptnl_mix.main(["count", "--rank", str(rank), "--work", str(tmp_path / f"work{rank}"), "--gcs", "gs://b/gptnl/lengths",
                    "--check-lengths", str(tmp_path / "cx_lengths" / "nl_part_00000.parquet"), "--workers", "2", *extra])


def _plan(tmp_path):
    gptnl_mix.main(["plan", "--gptnl-lengths", str(tmp_path / "gcs"), "--culturax-counts", str(tmp_path / "cx_counts"),
                    "--heldout", str(tmp_path / "heldout.parquet"),
                    "--out-dir", str(tmp_path / "selections"), "--plan", str(tmp_path / "plan.json")])
    return json.loads((tmp_path / "plan.json").read_text())


def _mix(tmp_path, rank, out=None):
    out = out or f"share{rank}"
    gptnl_mix.main(["mix", "--rank", str(rank), "--selection", str(tmp_path / "selections" / f"cxgptnl-w{rank}.npz"),
                    "--spill", str(tmp_path / f"spill{rank}"), "--out", str(tmp_path / out),
                    "--list", str(tmp_path / f"{out}.txt"), "--readers", "2", "--workers", "2"])


def test_the_included_and_excluded_collections_are_the_repos_and_disjoint():
    assert len(gptnl_mix.INCLUDED) == 20 and len(gptnl_mix.EXCLUDED) == 12
    assert not set(gptnl_mix.INCLUDED) & set(gptnl_mix.EXCLUDED)
    assert {"youtube-commons", "wikidata", "common-crawl", "multi-eurlex"} <= set(gptnl_mix.EXCLUDED)


def test_the_culturax_split_is_the_long_runs():
    files = [f for host in gptnl_mix.CULTURAX_FILES for f in host]
    assert sorted(files) == list(range(128)) and all(len(host) == 32 for host in gptnl_mix.CULTURAX_FILES)
    assert all(list(host) == sorted(host) for host in gptnl_mix.CULTURAX_FILES)
    assert sum(gptnl_mix.CULTURAX_HOST_DOCUMENTS) == 117_392_666, "CulturaX nl's documents, token-counts/summary.json"
    assert gptnl_mix.HOST_TOKENS == 10_229_514_240


def test_a_repo_with_another_collection_is_refused(monkeypatch):
    import huggingface_hub

    folders = list(gptnl_mix.INCLUDED) + list(gptnl_mix.EXCLUDED)[1:] + ["new-collection"]
    monkeypatch.setattr(huggingface_hub.HfApi, "list_repo_tree", lambda self, *a, **k: [
        SimpleNamespace(path=f"{f}/{f}_0001.parquet", size=1) for f in folders])
    with pytest.raises(ValueError, match=r"not in INCLUDED or EXCLUDED \['new-collection'\], missing \['youtube-commons'\]"):
        gptnl_mix.gptnl_files()


def test_files_are_assigned_largest_first_to_the_least_loaded_host():
    sizes = [100, 1, 50, 50, 30, 20, 99, 2]
    host_of = gptnl_mix.assign_files(sizes, 4)
    assert host_of == gptnl_mix.assign_files(sizes, 4)
    load = [sum(s for s, k in zip(sizes, host_of) if k == h) for h in range(4)]
    assert sorted(load) == [73, 80, 99, 100]
    assert host_of[0] == 0 and host_of[6] == 1


def test_token_counts_split_the_calls_by_documents_and_characters(monkeypatch):
    monkeypatch.setattr(gptnl_mix, "CHUNK_DOCS", 3)
    monkeypatch.setattr(gptnl_mix, "CHUNK_CHARS", 20)
    tokenizer = FakeTokenizer()
    texts = ["a b", "c", "d e f", "x " * 30, "g", "h i", "", "j"]
    assert gptnl_mix.token_counts(tokenizer, texts).tolist() == [2, 1, 3, 30, 1, 2, 0, 1]
    assert [len(c) for c in tokenizer.calls] == [3, 1, 3, 1]
    assert all(len(c) == 1 or sum(map(len, c)) <= 20 for c in tokenizer.calls)


def test_tokens_with_specials_are_the_trainers_pieces():
    from src.mmbjax.training.packing import split_into_pieces

    for max_length in (8, 1024):
        for n in list(range(0, 3 * max_length + 3)) + [100_003]:
            pieces = split_into_pieces(list(range(10, 10 + n)), max_length, 1, 2)
            assert gptnl_mix.tokens_with_specials(np.array([n]), max_length) == sum(map(len, pieces)), (max_length, n)


def test_documents_are_eligible_when_dutch_not_empty_and_their_text_is_new():
    dutch = np.array([1, 1, 0, 1, 1, 1, 1], dtype=bool)
    tokens = np.array([5, 3, 4, 0, 5, 2, 6])
    hashes = np.array([10, 11, 12, 13, 10, 14, 99], dtype=np.uint64)
    mask, dropped = gptnl_mix.eligible_documents(dutch, tokens, hashes, np.array([99, 12], dtype=np.uint64))
    assert mask.tolist() == [1, 1, 0, 0, 0, 1, 0]
    assert dropped["not_dutch"].tolist() == [0, 0, 1, 0, 0, 0, 0]
    assert dropped["empty"].tolist() == [0, 0, 0, 1, 0, 0, 0]
    assert dropped["duplicate"].tolist() == [0, 0, 0, 0, 1, 0, 0]
    assert dropped["in_culturax"].tolist() == [0, 0, 0, 0, 0, 0, 1]


def test_gptnl_documents_are_drawn_once_and_each_host_stops_at_its_target():
    rng = np.random.default_rng(1)
    tokens = rng.integers(1, 2000, size=5000)
    picks = gptnl_mix.gptnl_selection(tokens, 42, 4, 500_000)
    assert [p.tolist() for p in picks] == [p.tolist() for p in gptnl_mix.gptnl_selection(tokens, 42, 4, 500_000)]
    everything = np.concatenate(picks)
    assert len(np.unique(everything)) == len(everything), "a document on two hosts"
    order = np.random.default_rng([42, 0]).permutation(len(tokens))
    for k, p in enumerate(picks):
        assert tokens[p].sum() >= 500_000 and tokens[p].sum() - tokens[order[k::4][len(p) - 1]] < 500_000
        assert sorted(p.tolist()) == sorted(order[k::4][:len(p)].tolist())
    assert not set(gptnl_mix.gptnl_selection(tokens, 43, 4, 500_000)[0].tolist()) == set(picks[0].tolist())
    with pytest.raises(ValueError, match="fewer than"):
        gptnl_mix.gptnl_selection(tokens, 42, 4, int(tokens.sum()))


def test_culturax_rows_follow_a_draw_per_row_of_the_whole_file():
    tokens = [np.full(1000, 3), np.full(500, 5)]
    files, rows = gptnl_mix.culturax_selection(42, (4, 9), tokens, 2000)
    assert set(files.tolist()) == {4, 9}
    taken = sum(tokens[(4, 9).index(f)][r] for f, r in zip(files, rows))
    assert 2000 <= taken < 2005
    assert len(set(zip(files.tolist(), rows.tolist()))) == len(rows)
    assert list(zip(files.tolist(), rows.tolist())) == sorted(zip(files.tolist(), rows.tolist()))
    # The draws of file 4 do not depend on the other files of the host
    alone, rows_alone = gptnl_mix.culturax_selection(42, (4,), tokens[:1], 999)
    draws = np.random.default_rng([42, 1, 4]).random(1000)
    assert sorted(rows_alone.tolist()) == sorted(np.argsort(draws, kind="stable")[:len(rows_alone)].tolist())
    with pytest.raises(ValueError, match="not sorted"):
        gptnl_mix.culturax_selection(42, (9, 4), tokens, 2000)
    # Rows not allowed are passed over, the others keep their order
    allowed = [np.arange(1000) % 2 == 0, np.ones(500, dtype=bool)]
    files_a, rows_a = gptnl_mix.culturax_selection(42, (4, 9), tokens, 2000, allowed)
    assert not ((files_a == 4) & (rows_a % 2 == 1)).any()
    draws = np.concatenate([np.random.default_rng([42, 1, f]).random(len(t)) for f, t in zip((4, 9), tokens)])
    order = [i for i in np.argsort(draws, kind="stable") if np.concatenate(allowed)[i]]
    chosen = sorted(order[:len(rows_a)])
    assert [(4, i) if i < 1000 else (9, i - 1000) for i in chosen] == list(zip(files_a.tolist(), rows_a.tolist()))


def test_the_tokenizer_check_refuses_other_counts(tmp_path, monkeypatch):
    _world(tmp_path, monkeypatch)
    lengths = tmp_path / "cx_lengths" / "nl_part_00000.parquet"
    gptnl_mix.check_tokenizer(FakeTokenizer(), str(lengths), documents=50)
    n = pq.read_table(lengths).column("text").to_numpy().copy()
    n[np.arange(0, len(n), 2)] += 1
    pq.write_table(pa.table({"text": pa.array(n, pa.int32())}), lengths)
    with pytest.raises(AssertionError, match="other token counts than count_tokens.py's"):
        gptnl_mix.check_tokenizer(FakeTokenizer(), str(lengths), documents=50)


def test_counted_plans_and_mixed_shares_hold_each_chosen_document_once(tmp_path, monkeypatch):
    world = _world(tmp_path, monkeypatch)
    for rank in range(4):
        _count(tmp_path, rank)
    counted = sorted(p.name for p in world.bucket.iterdir())
    included = [p for p in world.gptnl if p.split("/")[0] in gptnl_mix.INCLUDED]
    assert counted == sorted(f"{stage_share.output_stem(p)}.parquet" for p in included)
    for path in included:
        lengths = pq.read_table(world.bucket / f"{stage_share.output_stem(path)}.parquet").to_pylist()
        assert [r["text"] for r in lengths] == [len(t.split()) for t, _ in world.gptnl[path]]
        assert [r["language"] for r in lengths] == [lang for _, lang in world.gptnl[path]]
        assert [r["xxh64"] for r in lengths] == [xxhash.xxh64_intdigest(t.encode()) for t, _ in world.gptnl[path]]
        assert [r["utf8_bytes"] for r in lengths] == [len(t.encode()) for t, _ in world.gptnl[path]]

    plan = _plan(tmp_path)
    assert plan["gptnl"]["dropped"]["duplicate"]["documents"] == 1
    assert plan["gptnl"]["dropped"]["empty"]["documents"] == 1
    assert plan["gptnl"]["dropped"]["in_culturax"]["documents"] == 1
    nl = [t for p in included for t, lang in world.gptnl[p] if lang == "nl"]
    assert plan["gptnl"]["eligible"]["documents"] == len(nl) - 3
    assert {lang for c in plan["collections"].values() for lang in c["languages"]} == {"nl", "en", "fy", "null"}
    assert plan["dataset_num_rows"] == sum(s["documents"] for s in plan["shares"])

    eligible = set(nl) - {"", world.gptnl[world.in_culturax][0][0]}
    seen = []
    for rank in range(4):
        _mix(tmp_path, rank)
        listed = (tmp_path / f"share{rank}.txt").read_text().split()
        assert [Path(p).name for p in listed] == [f"cxgptnl_{j:03d}-w{rank}of4.parquet" for j in range(3)]
        assert all(pq.read_schema(p).names == ["text"] for p in listed)
        texts = [t for p in listed for t in pq.read_table(p).column("text").to_pylist()]
        share = plan["shares"][rank]
        summary = json.loads((tmp_path / f"share{rank}" / "share.json").read_text())
        assert len(texts) == share["documents"] == summary["documents"]
        gptnl = [t for t in texts if t in eligible]
        culturax = [t for t in texts if t not in eligible]
        assert len(gptnl) == share["gptnl"]["documents"] and len(culturax) == share["culturax"]["documents"]
        host_culturax = {t for f in gptnl_mix.CULTURAX_FILES[rank] for t in world.culturax[gptnl_mix.culturax_path(f)]}
        assert set(culturax) <= host_culturax, "a CulturaX document of another host's files"
        assert not set(texts) & set(world.heldout), "a document of the held-out file"
        assert share["culturax"]["heldout_passed_over"] == (5 if rank == 3 else 0)
        gptnl_tokens = sum(len(t.split()) for t in gptnl)
        culturax_tokens = sum(len(t.split()) for t in culturax)
        assert gptnl_tokens == share["gptnl"]["tokens"] >= 240 and culturax_tokens == share["culturax"]["tokens"]
        assert 29.0 < 100 * gptnl_tokens / (gptnl_tokens + culturax_tokens) < 31.0
        # In the order of the positions: the share interleaves the two
        s = np.load(tmp_path / "selections" / f"cxgptnl-w{rank}.npz")
        is_gptnl = np.zeros(len(texts), dtype=bool)
        is_gptnl[s["pos"][:len(s["gptnl_row"])]] = True
        assert [t in eligible for t in texts] == is_gptnl.tolist()
        assert 0 < is_gptnl[:len(texts) // 2].sum() < len(gptnl)
        assert sum(f["tokens"] for f in summary["per_file"]) == share["tokens"]
        seen.extend(gptnl)
    assert len(seen) == len(set(seen)), "a GPT-NL document in two shares"


def test_a_second_mix_keeps_what_is_written_and_a_source_out_of_line_with_the_plan_is_refused(tmp_path, monkeypatch):
    world = _world(tmp_path, monkeypatch)
    for rank in range(4):
        _count(tmp_path, rank)
    _plan(tmp_path)
    _mix(tmp_path, 1)
    written = tmp_path / "share1" / "cxgptnl_000-w1of4.parquet"
    spilled = sorted((tmp_path / "spill1").glob("*.parquet"))
    before = [p.stat().st_mtime_ns for p in [written] + spilled]
    _mix(tmp_path, 1)
    assert [p.stat().st_mtime_ns for p in [written] + spilled] == before, "a second start wrote a finished file again"

    # A selection that is not the plan's
    selection = tmp_path / "selections" / "cxgptnl-w2.npz"
    good = selection.read_bytes()
    s = dict(np.load(selection))
    s["pos"] = s["pos"].copy()
    s["pos"][[0, 1]] = s["pos"][[1, 0]]
    np.savez(selection, **s)
    with pytest.raises(ValueError, match="is not the selection of the plan"):
        _mix(tmp_path, 2)
    selection.write_bytes(good)

    # Sources whose rows are not the plan's: every text of GPT-NL changed length
    for path in world.gptnl:
        table = pq.read_table(tmp_path / "hub" / "gptnl" / path)
        texts = table.column("text").to_pylist()
        pq.write_table(table.set_column(table.schema.get_field_index("text"), "text", pa.array([t + " extra" for t in texts])),
                       tmp_path / "hub" / "gptnl" / path)
    with pytest.raises(ValueError, match="bytes, the plan"):
        _mix(tmp_path, 2, out="share2b")
    assert not list((tmp_path / "share2b").glob("*.parquet")), "a share file was written from a source out of line"
