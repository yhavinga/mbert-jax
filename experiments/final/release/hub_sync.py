"""Helpers of push_staging.py and replace_public.py: one commit that makes a Hub repo hold exactly a set of files, and
the sha256 of every file of a repo at a revision. The token comes from the existing login (HF_TOKEN or the cached
token); nothing here prints it."""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def remote_sha256(api, repo: str, revision: str, cache_dir: Path) -> tuple[str, bool, dict[str, str]]:
    """(commit sha, private, {path: sha256}) of every file of a model repo at a revision.

    An LFS/Xet file's sha256 is the Hub's own; a small file is downloaded at the resolved commit and hashed.
    """
    from huggingface_hub import hf_hub_download

    info = api.model_info(repo, revision=revision, files_metadata=True)
    out = {}
    for s in info.siblings:
        if s.lfs is not None:
            out[s.rfilename] = s.lfs.sha256
        else:
            out[s.rfilename] = sha256_file(Path(hf_hub_download(repo, s.rfilename, revision=info.sha, cache_dir=cache_dir)))
    return info.sha, bool(info.private), out


def plan(current: list[str], files: dict[str, Path], keep: set[str]) -> tuple[list[str], list[str]]:
    """(paths to add or replace, paths to delete) for a commit after which the repo holds `files` (plus `keep`)."""
    return sorted(files), sorted(set(current) - set(files) - keep)


def commit_exactly(api, repo: str, files: dict[str, Path], message: str, keep: set[str]) -> tuple[str, list[str]]:
    """One commit on main: every file of `files` added or replaced, every other file except `keep` deleted."""
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete

    adds, deletes = plan(api.list_repo_files(repo), files, keep)
    ops = [CommitOperationAdd(path_in_repo=p, path_or_fileobj=str(files[p])) for p in adds]
    ops += [CommitOperationDelete(path_in_repo=p) for p in deletes]
    info = api.create_commit(repo, operations=ops, commit_message=message)
    return info.oid, deletes


def compare(expected: dict[str, str], actual: dict[str, str]) -> dict:
    """Paths whose sha256 differ, and paths on one side only."""
    return {
        "different": sorted(p for p in expected.keys() & actual.keys() if expected[p] != actual[p]),
        "missing": sorted(expected.keys() - actual.keys()),
        "extra": sorted(actual.keys() - expected.keys()),
    }
