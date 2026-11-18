"""Tests of experiments/final/release/replace_public.py against an in-memory Hub: the replacement of an existing public
repo (T2's case) and, with --create, a public repo that does not exist yet."""

import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

huggingface_hub = pytest.importorskip("huggingface_hub")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "final" / "release"))

import replace_public  # noqa: E402

STAGING = "yhavinga/model-staging"
PUBLIC = "yhavinga/model"
LFS = {"model.safetensors"}


class FakeHub:
    """Repos as lists of commits, each a {path: bytes}; the sha of a commit is the hash of its position and files."""

    def __init__(self):
        self.repos = {}

    def add(self, repo, files, private):
        self.repos[repo] = {"private": private, "commits": []}
        self.commit(repo, files)

    def commit(self, repo, files):
        commits = self.repos[repo]["commits"]
        h = hashlib.sha1(repr((repo, len(commits), sorted(files.items()))).encode()).hexdigest()
        commits.append({"sha": h, "files": dict(files)})
        return h

    def at(self, repo, revision):
        commits = self.repos[repo]["commits"]
        if revision in (None, "main"):
            return commits[-1]
        return next(c for c in commits if c["sha"] == revision)


class FakeApi:
    def __init__(self, hub):
        self.hub = hub

    def repo_exists(self, repo, repo_type="model"):
        return repo in self.hub.repos

    def model_info(self, repo, revision=None, files_metadata=False):
        c = self.hub.at(repo, revision)
        siblings = [SimpleNamespace(rfilename=p, lfs=SimpleNamespace(sha256=hashlib.sha256(b).hexdigest()) if p in LFS else None)
                    for p, b in sorted(c["files"].items())]
        return SimpleNamespace(sha=c["sha"], private=self.hub.repos[repo]["private"], siblings=siblings)

    def create_repo(self, repo, repo_type="model", private=False):
        assert repo not in self.hub.repos
        self.hub.add(repo, {".gitattributes": b"*.safetensors filter=lfs\n"}, private)

    def list_repo_files(self, repo):
        return sorted(self.hub.at(repo, "main")["files"])

    def create_commit(self, repo, operations, commit_message):
        files = dict(self.hub.at(repo, "main")["files"])
        for op in operations:
            if isinstance(op, huggingface_hub.CommitOperationAdd):
                files[op.path_in_repo] = Path(op.path_or_fileobj).read_bytes()
            else:
                del files[op.path_in_repo]
        return SimpleNamespace(oid=self.hub.commit(repo, files))

    def super_squash_history(self, repo_id, branch, commit_message):
        r = self.hub.repos[repo_id]
        r["commits"] = r["commits"][-1:]

    def list_repo_commits(self, repo):
        return [SimpleNamespace(commit_id=c["sha"]) for c in reversed(self.hub.repos[repo]["commits"])]

    def delete_repo(self, repo_id, repo_type="model"):
        del self.hub.repos[repo_id]


@pytest.fixture
def hub(monkeypatch, tmp_path):
    hub = FakeHub()

    def download(repo, filename, revision=None, cache_dir=None, local_dir=None, **_):
        c = hub.at(repo, revision)
        target = Path(local_dir or cache_dir or tmp_path / "dl") / c["sha"][:8] / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(c["files"][filename])
        return str(target)

    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: FakeApi(hub))
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    staging_files = {".gitattributes": b"*.safetensors filter=lfs\n", "README.md": b"# card\n",
                     "config.json": b"{}\n", "model.safetensors": b"\x00" * 64}
    hub.add(STAGING, staging_files, private=True)
    return hub


def run(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["replace_public.py", "--staging", STAGING, "--public", PUBLIC, *args])
    return replace_public.main()


def test_create_a_new_private_repo_with_the_staging_files_in_one_commit(hub, monkeypatch, tmp_path):
    rev = hub.at(STAGING, "main")["sha"]
    # Plan only: nothing is created
    assert run(monkeypatch, "--staging-revision", rev, "--create", "private") == 0
    assert PUBLIC not in hub.repos
    assert run(monkeypatch, "--staging-revision", rev, "--create", "private", "--execute", "--delete-staging",
               "--workdir", str(tmp_path / "work")) == 0
    public = hub.repos[PUBLIC]
    assert public["private"] is True and len(public["commits"]) == 1
    assert public["commits"][0]["files"] == {".gitattributes": b"*.safetensors filter=lfs\n", "README.md": b"# card\n",
                                             "config.json": b"{}\n", "model.safetensors": b"\x00" * 64}
    assert STAGING not in hub.repos


def test_create_is_refused_for_an_existing_repo_and_needed_for_a_missing_one(hub, monkeypatch):
    rev = hub.at(STAGING, "main")["sha"]
    assert run(monkeypatch, "--staging-revision", rev, "--execute") == 1
    assert PUBLIC not in hub.repos
    hub.add(PUBLIC, {"README.md": b"old\n"}, private=False)
    assert run(monkeypatch, "--staging-revision", rev, "--create", "public", "--execute") == 1
    assert hub.at(PUBLIC, "main")["files"] == {"README.md": b"old\n"}


def test_an_existing_public_repo_is_replaced_and_keeps_its_visibility(hub, monkeypatch, tmp_path):
    hub.add(PUBLIC, {".gitattributes": b"old\n", "README.md": b"old\n", "old_vocab.txt": b"x\n"}, private=False)
    hub.commit(PUBLIC, {".gitattributes": b"old\n", "README.md": b"older\n", "old_vocab.txt": b"x\n"})
    rev = hub.at(STAGING, "main")["sha"]
    assert run(monkeypatch, "--staging-revision", rev, "--execute", "--workdir", str(tmp_path / "work")) == 0
    public = hub.repos[PUBLIC]
    assert public["private"] is False and len(public["commits"]) == 1
    assert public["commits"][0]["files"] == hub.at(STAGING, "main")["files"]
    assert STAGING in hub.repos
