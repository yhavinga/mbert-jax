"""Tests of experiments/final/release/hub_sync.py: the plan of a commit that makes a repo hold exactly a set of files."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "final" / "release"))

from hub_sync import compare, plan, sha256_file  # noqa: E402


def test_plan_replaces_everything_and_deletes_what_is_not_in_the_set():
    current = [".gitattributes", "README.md", "model.safetensors", "old_vocab.txt"]
    files = {"README.md": Path("a"), "model.safetensors": Path("b"), "tokenizer.json": Path("c")}
    adds, deletes = plan(current, files, keep={".gitattributes"})
    assert adds == ["README.md", "model.safetensors", "tokenizer.json"]
    assert deletes == ["old_vocab.txt"]
    # Without keep, a .gitattributes that the new set lacks goes too
    assert plan(current, files, keep=set())[1] == [".gitattributes", "old_vocab.txt"]


def test_compare_names_different_missing_and_extra_files():
    expected = {"a": "1", "b": "2", "c": "3"}
    actual = {"a": "1", "b": "9", "d": "4"}
    assert compare(expected, actual) == {"different": ["b"], "missing": ["c"], "extra": ["d"]}
    assert compare(expected, dict(expected)) == {"different": [], "missing": [], "extra": []}


def test_sha256_file(tmp_path):
    p = tmp_path / "x"
    p.write_bytes(b"abc")
    assert sha256_file(p) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
