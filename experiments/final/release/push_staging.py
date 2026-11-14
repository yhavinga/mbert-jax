#!/usr/bin/env python3
"""Put an export directory in a private staging repo on the Hub, in one commit, and read it back.

  python push_staging.py --folder DIR --repo yhavinga/dmbert-230m-culturax-staging --out push.json

The repo is created private when it does not exist, and must be private (otherwise nothing is uploaded). After the
commit the repo holds exactly the files of DIR plus its .gitattributes. Read back: the sha256 of every file at the
new commit equals the local file's. Exit 1 when the repo is not private or a file differs.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hub_sync import commit_exactly, compare, remote_sha256, sha256_file  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--folder", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--message", default="Model, tokenizer, config and model card")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if not args.repo.endswith("-staging"):
        sys.exit(f"{args.repo}: this script only writes a repo whose name ends in -staging")

    from huggingface_hub import HfApi

    api = HfApi()
    folder = Path(args.folder)
    files = {p.relative_to(folder).as_posix(): p for p in sorted(folder.rglob("*")) if p.is_file()}
    if not files or ".gitattributes" in files:
        sys.exit(f"{folder}: no files, or a .gitattributes of its own")
    api.create_repo(args.repo, repo_type="model", private=True, exist_ok=True)
    if not api.model_info(args.repo).private:
        sys.exit(f"{args.repo} is not private: nothing uploaded")
    oid, deleted = commit_exactly(api, args.repo, files, args.message, keep={".gitattributes"})
    local = {p: sha256_file(f) for p, f in files.items()}
    with tempfile.TemporaryDirectory() as cache:
        sha, private, remote = remote_sha256(api, args.repo, oid, Path(cache))
    remote_files = {p: s for p, s in remote.items() if p != ".gitattributes"}
    diff = compare(local, remote_files)
    ok = private and sha == oid and not any(diff.values())
    report = {"repo": args.repo, "commit": oid, "private": private, "deleted": deleted, "files": local,
              "readback": diff, "ok": ok}
    Path(args.out).write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: report[k] for k in ("repo", "commit", "private", "deleted", "readback", "ok")}))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
