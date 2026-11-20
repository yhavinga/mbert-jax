#!/bin/bash
# On a worker of a v4-32: CulturaX nl for the shape ablation of dmbert v3 (protocol.md part 2), detached.
#   NFILES=8 bash stage_ablation.sh
# Worker N takes the first NFILES files of its list in T2's staging (gs://.../mbert/final/staging/
# culturax_nl_files-w<N>.txt, the list of the final run on that worker) and fetches each from Hugging Face, uonlp/CulturaX
# at T2's revision 6a8734bc, to the same path (~/culturax_nl/nl/nl_part_*.parquet; a file already there with the
# reference size is not fetched again), checks size and sha256 against T2's reference (culturax-w<N>.tsv), and only then
# writes ~/v3abl_files.txt (those paths, in that order) and ~/v3abl_files.rows (their rows). 4 workers x NFILES files of
# the 128; CPU, disk and network only, at nice 19 and ionice idle. Deletes nothing. The log is ~/logs/v3abl_stage.log,
# ending in "EXIT <code>". Environment: NFILES (default 8), PY (default ~/mbert-jax/.venv/bin/python), RANK, F.
set -euo pipefail
NFILES=${NFILES:-8}
RANK=${RANK:-$(hostname | sed 's/.*-w-//')}
PY=${PY:-$HOME/mbert-jax/.venv/bin/python}
F=${F:-gs://t5-dutch-english-usc2/mbert/final}
LOG=$HOME/logs/v3abl_stage.log
[[ $RANK =~ ^[0-3]$ ]] || { echo "$(hostname): worker number '$RANK' is not one of 0 to 3"; exit 1; }
[[ $NFILES =~ ^[0-9]+$ ]] && [ "$NFILES" -ge 1 ] && [ "$NFILES" -le 32 ] || { echo "NFILES must be 1 to 32"; exit 1; }
if [ -z "${V3ABL_DETACHED:-}" ]; then
  pgrep -f "logs/v3abl_stage[.]log 2>&1; echo EXIT" > /dev/null && { echo "$(hostname): stage_ablation.sh runs already"; exit 1; }
  mkdir -p ~/logs
  V3ABL_DETACHED=1 setsid nohup nice -n 19 ionice -c3 bash -c "bash $0 > $LOG 2>&1; echo EXIT \$? >> $LOG" > /dev/null 2>&1 < /dev/null &
  echo "$(hostname): started, $LOG"
  exit 0
fi
[ -s ~/.cache/huggingface/token ] || { echo "$(hostname): no ~/.cache/huggingface/token"; exit 1; }
gsutil -q cp "$F/staging/culturax-w$RANK.tsv" /tmp/v3abl_ref.tsv
gsutil -q cp "$F/staging/culturax_nl_files-w$RANK.txt" /tmp/v3abl_list.txt
"$PY" - /tmp/v3abl_ref.tsv /tmp/v3abl_list.txt "$NFILES" <<'PY'
import hashlib, os, sys
from pathlib import Path
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq

REPO, REVISION = "uonlp/CulturaX", "6a8734bc69fefcbb7735f4f9250f43e4cd7a442e"
ref = {line.split("\t")[0]: (int(line.split("\t")[1]), line.split("\t")[2].strip()) for line in open(sys.argv[1]) if line.strip()}
listed = [line.strip() for line in open(sys.argv[2]) if line.strip()][: int(sys.argv[3])]
home = str(Path.home())
rows = 0
for path in listed:
    if path not in ref:
        sys.exit(f"{path} is not in the reference")
    if not path.startswith(f"{home}/culturax_nl/nl/"):
        sys.exit(f"{path} is not under {home}/culturax_nl/nl/")
    size, sha = ref[path]
    p = Path(path)
    if not (p.exists() and p.stat().st_size == size):
        free = os.statvfs(home).f_bavail * os.statvfs(home).f_frsize
        if free < size + 2**30:
            sys.exit(f"{path}: {free / 2**30:.1f} GiB free on the disk, {size / 2**30:.1f} GiB + 1 needed; nothing deleted")
        hf_hub_download(REPO, f"nl/{p.name}", repo_type="dataset", revision=REVISION, local_dir=f"{home}/culturax_nl")
    if p.stat().st_size != size:
        sys.exit(f"{path}: {p.stat().st_size} bytes, the reference {size}")
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    if h.hexdigest() != sha:
        sys.exit(f"{path}: sha256 {h.hexdigest()}, the reference {sha}")
    n = pq.ParquetFile(p).metadata.num_rows
    rows += n
    print(f"{path} {size} {sha[:12]} rows {n}", flush=True)
tmp = Path(home, "v3abl_files.txt.tmp")
tmp.write_text("".join(f"{path}\n" for path in listed))
os.replace(tmp, Path(home, "v3abl_files.txt"))
Path(home, "v3abl_files.rows").write_text(f"{rows}\n")
print(f"{len(listed)} files, {rows} rows: ~/v3abl_files.txt")
PY
echo "$(hostname): staged, $(df -h ~ | tail -1 | awk '{print $4}') free"
