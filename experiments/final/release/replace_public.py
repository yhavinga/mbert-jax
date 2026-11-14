#!/usr/bin/env python3
"""Replace the contents of the public model repo by those of the private staging repo, after Yeb's approval.

  # plan only (reads both repos, changes nothing):
  python replace_public.py --staging-revision <sha>
  # the replacement:
  python replace_public.py --staging-revision <sha> --execute [--delete-staging]

With --execute, in this order, each step only when the previous one succeeded:
  1. staging (private) at --staging-revision: every file is downloaded to --workdir and its sha256 checked against
     the Hub's;
  2. one commit on the public repo's main: every staging file added or replaced, every public file that staging does
     not have deleted (in the same commit);
  3. super_squash_history of the public repo's main into one commit;
  4. read back: the set of files of the public repo's main and the sha256 of each equal staging's;
  5. with --delete-staging, and only when 4 passed: the staging repo is deleted.
The public repo keeps its visibility. Exit 1 at the first failing step. The token comes from the existing login.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hub_sync import commit_exactly, compare, plan, remote_sha256, sha256_file  # noqa: E402

STAGING = "yhavinga/dmbert-230m-culturax-staging"
PUBLIC = "yhavinga/dmbert-230m-culturax"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--staging", default=STAGING)
    parser.add_argument("--staging-revision", required=True, help="the staging commit Yeb approved (full sha)")
    parser.add_argument("--public", default=PUBLIC)
    parser.add_argument("--message", default="Dutch ModernBERT 230M trained on CulturaX (nl): model, tokenizer, config and model card")
    parser.add_argument("--workdir", default=None, help="for the downloaded staging files (default: a temporary directory)")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--delete-staging", action="store_true")
    args = parser.parse_args()
    if len(args.staging_revision) != 40:
        sys.exit("--staging-revision must be a full commit sha")

    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(args.workdir) if args.workdir else Path(tmp)
        cache = work / "readback-cache"
        sha, private, staging = remote_sha256(api, args.staging, args.staging_revision, cache)
        if sha != args.staging_revision or not private:
            print(f"FAILED: {args.staging} at {args.staging_revision}: resolved {sha}, private {private}")
            return 1
        public_sha, public_private, public = remote_sha256(api, args.public, "main", cache)
        adds, deletes = plan(list(public), {p: Path(p) for p in staging}, keep=set())
        print(f"staging {args.staging} @ {sha} (private): {len(staging)} files")
        print(f"public  {args.public} @ {public_sha} (private={public_private}): {len(public)} files")
        for p in adds:
            state = "new" if p not in public else ("same" if public[p] == staging[p] else "replaced")
            print(f"  add {p}  sha256 {staging[p][:16]}…  ({state})")
        for p in deletes:
            print(f"  delete {p}")
        print(f"then super_squash_history of {args.public} main; read back; "
              + (f"delete {args.staging}" if args.delete_staging else f"{args.staging} stays"))
        if not args.execute:
            print("plan only: nothing changed (add --execute)")
            return 0

        # 1. the staging files, at the approved commit, checked
        local_dir = work / f"staging-{sha[:12]}"
        files = {}
        for p in sorted(staging):
            f = Path(hf_hub_download(args.staging, p, revision=sha, local_dir=local_dir))
            if sha256_file(f) != staging[p]:
                print(f"FAILED step 1: {p} downloaded with another sha256")
                return 1
            files[p] = f
        print(f"step 1 done: {len(files)} files of {args.staging} @ {sha} in {local_dir}")
        # 2. one commit with all of them, and the deletions
        oid, deleted = commit_exactly(api, args.public, files, args.message, keep=set())
        print(f"step 2 done: commit {oid} on {args.public}, deleted {deleted}")
        # 3. one commit in the history
        api.super_squash_history(repo_id=args.public, branch="main", commit_message=args.message)
        commits = api.list_repo_commits(args.public)
        print(f"step 3 done: {len(commits)} commit(s) on {args.public} main, head {commits[0].commit_id}")
        # 4. read back
        new_sha, new_private, now = remote_sha256(api, args.public, "main", work / "readback-public")
        diff = compare(staging, now)
        print(json.dumps({"public": args.public, "head": new_sha, "private": new_private, "readback": diff}))
        if any(diff.values()) or new_private != public_private or len(commits) != 1:
            print("FAILED step 4: the public repo is not staging's contents in one commit; staging is kept")
            return 1
        print(f"step 4 done: {len(now)} files, each sha256 equal to staging's")
        # 5. staging away, only on request
        if args.delete_staging:
            print(f"deleting {args.staging}")
            api.delete_repo(repo_id=args.staging, repo_type="model")
            print(f"step 5 done: {args.staging} deleted")
    return 0


if __name__ == "__main__":
    sys.exit(main())
