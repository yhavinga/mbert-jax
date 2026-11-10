#!/bin/bash
# On every worker of a second v4-32 (mbert-mix): variant culturax's data exactly as on mbert-corpus, so that tak 1 (or
# another leg) can run there while the main run goes on (protocol.md "Het schema"; mbert-ops HANDOVER-final-mix.md).
#   bash stage_culturax.sh <vm>
# Worker N gets S's 32 CulturaX nl files of mbert-corpus's worker N on the same paths (~/culturax_nl/nl/nl_part_*.parquet)
# and ~/culturax_nl_files.txt byte for byte that worker's list: the trainer's data state names the files and their order,
# so a data state of mbert-corpus continues here only with the same list (restore_data_state in train_minimal.py).
# The reference comes from the bucket, made on co1 from mbert-corpus (read only there: its lists, stat, the sha256 that
# Hugging Face reported at its download, and the sha256 of every file as it is on its disk):
#   $F/staging/culturax-w<N>.tsv            per file in list order: path, size, sha256
#   $F/staging/culturax_nl_files-w<N>.txt   mbert-corpus's ~/culturax_nl_files.txt of worker N
# The files come from Hugging Face, uonlp/CulturaX at S's revision 6a8734bc, with the worker's ~/.cache/huggingface/token:
# first Hugging Face's size and LFS sha256 of each file must be the reference's, then the download (a file already there
# with the reference's size is not fetched again), then every file's size and sha256 against the reference, and only
# then ~/culturax_nl_files.txt (an existing one must already be the reference). CPU, disk and network only: nothing
# imports JAX or opens the TPU, at nice 19 and ionice idle, so it may run next to other work. Nothing is deleted: a
# wrong file is reported with its path and stays (HANDOVER-final-mix.md says how to replace it by hand).
# Detached: the log is ~/logs/final_stage_culturax.log, ending in "EXIT <code>", copied to
# $F/logs/final_stage_culturax-<vm>-w<N>.log; per file here its path, size, sha256 and mtime (seconds) to
# $F/staging/<vm>-w<N>.tsv, written also when a file is wrong.
# Environment: PY (default ~/mbert-jax/.venv/bin/python), WORKERS (parallel downloads, default 4); for a dry run F, RANK.
set -euo pipefail
VM_NAME=${1:?usage: stage_culturax.sh <vm>}
RANK=${RANK:-$(hostname | sed 's/.*-w-//')}
PY=${PY:-$HOME/mbert-jax/.venv/bin/python}
F=${F:-gs://t5-dutch-english-usc2/mbert/final}
LOG=$HOME/logs/final_stage_culturax.log
[[ $RANK =~ ^[0-3]$ ]] || { echo "$(hostname): worker number '$RANK' is not one of 0 to 3"; exit 1; }
[[ $VM_NAME =~ ^[a-z0-9-]+$ ]] || { echo "$(hostname): not a VM name: $VM_NAME"; exit 1; }
# mbert-corpus has the originals and trains on them: nothing of this runs there
[ "$VM_NAME" != mbert-corpus ] || { echo "$(hostname): not on mbert-corpus, whose files are the reference"; exit 1; }
if [ -z "${FINAL_DETACHED:-}" ]; then
  # The detached wrapper's command line; the ssh shell that runs this script has another
  pgrep -f "logs/final_stage_culturax[.]log 2>&1; echo EXIT" > /dev/null && { echo "$(hostname): stage_culturax.sh runs already"; exit 1; }
  mkdir -p ~/logs
  FINAL_DETACHED=1 setsid nohup nice -n 19 ionice -c3 bash -c "bash $0 $VM_NAME > $LOG 2>&1; echo EXIT \$? >> $LOG; gsutil -q cp $LOG $F/logs/final_stage_culturax-$VM_NAME-w$RANK.log" > /dev/null 2>&1 < /dev/null &
  echo "$(hostname): started, $LOG"
  exit 0
fi

cd "$HOME"
[ -x "$PY" ] || { echo "$(hostname): no $PY"; exit 1; }
[ -s ~/.cache/huggingface/token ] || { echo "$(hostname): no ~/.cache/huggingface/token (put it there through stdin)"; exit 1; }
WORK=$(mktemp -d)
gsutil -q cp "$F/staging/culturax-w$RANK.tsv" "$F/staging/culturax_nl_files-w$RANK.txt" "$WORK/"
echo "$(hostname): worker $RANK of $VM_NAME, $(df -h ~ | tail -1 | awk '{print $4}') free on the disk"
export HF_HUB_DISABLE_PROGRESS_BARS=1 HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_ENABLE_HF_TRANSFER=0 PYTHONDONTWRITEBYTECODE=1
rc=0
"$PY" -I -u - "$WORK/culturax-w$RANK.tsv" "$WORK/culturax_nl_files-w$RANK.txt" "$HOME" "${WORKERS:-4}" "$WORK/result.tsv" <<'EOF' || rc=$?
import hashlib, os, shutil, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

REPO, REVISION = "uonlp/CulturaX", "6a8734bc69fefcbb7735f4f9250f43e4cd7a442e"
ref_path, list_path, home, workers, result_path = sys.argv[1], sys.argv[2], Path(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
target = home / "culturax_nl"
ref = [line.split("\t") for line in Path(ref_path).read_text().splitlines()]
listed = Path(list_path).read_text()
paths = [r[0] for r in ref]
if len(ref) != 32 or any(len(r) != 3 for r in ref) or listed != "".join(p + "\n" for p in paths):
    sys.exit(f"the reference {ref_path} is not 32 lines of path, size and sha256 in the order of {list_path}")
names = [Path(p).name for p in paths]
wrong_place = [p for p, n in zip(paths, names) if p != str(target / "nl" / n)]
if wrong_place:
    sys.exit(f"the reference's paths are not under {target / 'nl'} here (e.g. {wrong_place[0]}): the data state would not match")
size_of = {n: int(r[1]) for n, r in zip(names, ref)}
sha_of = {n: r[2] for n, r in zip(names, ref)}

# Hugging Face's own size and sha256 at S's revision, before a byte is fetched
info = {i.path: i for i in HfApi().get_paths_info(REPO, [f"nl/{n}" for n in names], repo_type="dataset", revision=REVISION)}
for n in names:
    i = info.get(f"nl/{n}")
    if i is None or i.lfs is None or i.size != size_of[n] or i.lfs.sha256 != sha_of[n]:
        sys.exit(f"nl/{n} at {REVISION[:8]} on Hugging Face is not the reference's: {i}")
print(f"Hugging Face at {REVISION[:8]}: the 32 files have the reference's sizes and sha256", flush=True)

present = [n for n in names if (target / "nl" / n).is_file() and (target / "nl" / n).stat().st_size == size_of[n]]
missing = [n for n in names if n not in present]
need, free = sum(size_of[n] for n in missing), shutil.disk_usage(home).free
print(f"{len(present)} files there already, {len(missing)} to fetch: {need / 1e9:.1f} GB, {free / 1e9:.1f} GB free", flush=True)
if need + 3 * 2**30 > free:
    sys.exit(f"too little disk: {need / 1e9:.1f} GB to fetch and 3 GiB to spare, {free / 1e9:.1f} GB free")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(16 * 2**20):
            h.update(chunk)
    return h.hexdigest()


def fetch(n):
    t = time.time()
    got = hf_hub_download(REPO, f"nl/{n}", repo_type="dataset", revision=REVISION, local_dir=target)
    if Path(got) != target / "nl" / n:
        raise RuntimeError(f"nl/{n} went to {got}, not {target / 'nl' / n}")
    seconds = time.time() - t
    print(f"fetched {n}: {size_of[n] / 1e9:.2f} GB in {seconds:.0f} s ({size_of[n] / 1e6 / max(seconds, 1e-3):.0f} MB/s)", flush=True)


t0 = time.time()
with ThreadPoolExecutor(workers) as pool:
    list(pool.map(fetch, missing))
print(f"fetched {len(missing)} files in {time.time() - t0:.0f} s; now the sha256 of all 32", flush=True)

with ThreadPoolExecutor(workers) as pool:
    sums = dict(zip(names, pool.map(lambda n: sha256(target / "nl" / n), names)))
stats = {n: (target / "nl" / n).stat() for n in names}
sizes = {n: stats[n].st_size for n in names}
# The mtime lets a later check (data_check of final_lib.sh on co1) see by stat alone that a file is still the hashed one
Path(result_path).write_text("".join(f"{p}\t{sizes[n]}\t{sums[n]}\t{int(stats[n].st_mtime)}\n" for p, n in zip(paths, names)))
bad = [f"{target / 'nl' / n}: size {sizes[n]} sha256 {sums[n]}, the reference {size_of[n]} {sha_of[n]}"
       for n in names if sizes[n] != size_of[n] or sums[n] != sha_of[n]]
if bad:
    sys.exit(f"{len(bad)} of 32 files are not the reference's (they stay; replace them by hand):\n" + "\n".join(bad))
print(f"all 32 files have the reference's size and sha256 ({sum(sizes.values()) / 1e9:.1f} GB)", flush=True)
EOF
[ ! -s "$WORK/result.tsv" ] || gsutil -q cp "$WORK/result.tsv" "$F/staging/$VM_NAME-w$RANK.tsv"
[ "$rc" = 0 ] || exit "$rc"
if [ -e ~/culturax_nl_files.txt ]; then
  cmp -s ~/culturax_nl_files.txt "$WORK/culturax_nl_files-w$RANK.txt" \
    || { echo "$(hostname): ~/culturax_nl_files.txt is there and is not mbert-corpus's list of worker $RANK"; exit 1; }
  echo "$(hostname): ~/culturax_nl_files.txt was there already and is mbert-corpus's list of worker $RANK"
else
  cp "$WORK/culturax_nl_files-w$RANK.txt" ~/culturax_nl_files.txt.new && mv ~/culturax_nl_files.txt.new ~/culturax_nl_files.txt
  echo "$(hostname): ~/culturax_nl_files.txt written, mbert-corpus's list of worker $RANK"
fi
echo "$(hostname): ready, $(df -h ~ | tail -1 | awk '{print $4}') free on the disk"
