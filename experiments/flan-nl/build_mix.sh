#!/bin/bash
# On every worker of mbert-corpus (after flan_nl.py assemble): this host's Dutch mix of arm F-nl, on the CPU only.
#   PY=$HOME/mbert-dumb/.venv/bin/python bash build_mix.sh
# F's mix of this host (~/flan_decay_files.txt) with every FLAN row replaced in place by its Dutch version
# (build_nl_mix.py), written by the builder of F's commit (~/mbert-flan at F_COMMIT, as setup_flan.sh left it) with
# the venv F's builder and trainer ran with, so the trainer's shuffle is the one the order was made for.
# Writes ~/flan_decay_nl/mix-<build>.parquet (~1.4 GB), its manifest and id map, ~/flan_decay_nl_files.txt for
# --data-files-list, and copies of the manifest and the id map to $ROOT/mix/. A second start keeps a finished mix.
set -euo pipefail
ROOT=gs://t5-dutch-english-usc2/mbert/flan-nl
F_COMMIT=bd6ddca909168bd4fb1dc9cd1cc748bb861a53c5
PY=${PY:-$HOME/mbert-dumb/.venv/bin/python}
F_CODE=$HOME/mbert-flan
CODE=$(cd "$(dirname "$0")" && pwd)
RANK=$(hostname | sed 's/.*-w-//')
[[ $RANK =~ ^[0-3]$ ]] || { echo "$(hostname): worker number '$RANK' is not one of 0 to 3"; exit 1; }
[ -x "$PY" ] && [ -s ~/flan_decay_files.txt ] || { echo "$(hostname): needs $PY and ~/flan_decay_files.txt (F's setup_flan.sh)"; exit 1; }
# The checkout may be a later commit of flan-decay (the zero-shot fix); what the mix depends on must be F's
git -C "$F_CODE" diff --quiet "$F_COMMIT" -- scripts/build_decay_mix.py src \
  || { echo "$(hostname): the builder or src/ of $F_CODE differ from F's commit $F_COMMIT"; exit 1; }
# F's mix is ~1.4 GB; the Dutch one is a little smaller, and the trainer writes logs here too
free_gb=$(df -BG --output=avail "$HOME" | tail -1 | tr -dc 0-9)
[ "$free_gb" -ge 4 ] || { echo "$(hostname): $free_gb GB free on the boot disk, the Dutch mix needs ~1.4 GB and a margin"; exit 1; }
mkdir -p ~/logs ~/flan_decay_nl /dev/shm/flan_nl/nl
gsutil -q cp "$ROOT/nl/w$RANK.parquet" /dev/shm/flan_nl/nl/w$RANK.parquet
echo "$(hostname): rank $RANK, F's code $(git -C "$F_CODE" rev-parse --short HEAD), $free_gb GB free, $(date +%H:%M:%S)"
HF_HUB_DISABLE_PROGRESS_BARS=1 PYTHONPATH=$F_CODE/src:$F_CODE/scripts "$PY" -u "$CODE/build_nl_mix.py" --rank "$RANK" \
  --english-list ~/flan_decay_files.txt --dutch /dev/shm/flan_nl/nl/w$RANK.parquet --out ~/flan_decay_nl \
  --list ~/flan_decay_nl_files.txt
mix=$(cat ~/flan_decay_nl_files.txt)
[ -s "$mix" ] || { echo "$(hostname): $mix of ~/flan_decay_nl_files.txt is missing"; exit 1; }
build=$(basename "$mix" .parquet); build=${build#mix-}
gsutil -q cp ~/flan_decay_nl/manifest-$build.json "$ROOT/mix/w$RANK.json"
gsutil -q cp ~/flan_decay_nl/ids-$build.parquet "$ROOT/mix/w$RANK-ids.parquet"
echo "$(hostname): ready at $(date +%H:%M:%S), $(du -h "$mix" | cut -f1) in $mix, $(df -h ~ | tail -1 | awk '{print $4}') free"
