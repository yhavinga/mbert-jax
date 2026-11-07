#!/bin/bash
# On every worker of mbert-corpus (v4-32, 4 hosts), once: the code of arm F in a clone of its own (~/mbert-flan at
# COMMIT) and this host's decay mix (scripts/build_decay_mix.py): its CulturaX share in ~/culturax_nl_files.txt, the
# list of arm W, and its part of FLAN. On the CPU only, so it can run while a trainer has the TPU.
#   COMMIT=<commit> PY=$HOME/mbert-dumb/.venv/bin/python bash setup_flan.sh
# Writes ~/flan_decay/mix-<build>.parquet (~2.5 GB), its manifest, ~/flan_decay_files.txt for --data-files-list, and a
# copy of the manifest to $RUN_ROOT/mix/w<N>.json. FLAN shards pass through /dev/shm/flan_decay (~600 MB at a time).
# Needs ~/.cache/huggingface/token (commands.txt). A second start keeps a finished mix.
# Environment overrides, for the smoke test only: RUN_ROOT RANK EXTRA (more flags for the builder).
set -euo pipefail
COMMIT=${COMMIT:?set COMMIT to the commit of commands.txt}
PY=${PY:-$HOME/mbert-corpus/.venv/bin/python}
CODE=$HOME/mbert-flan
RUN_ROOT=${RUN_ROOT:-gs://t5-dutch-english-usc2/mbert/flan-decay}
RANK=${RANK:-$(hostname | sed 's/.*-w-//')}
EXTRA=${EXTRA:-}
cd "$HOME"
[ -s ~/.cache/huggingface/token ] || { echo "$(hostname): no ~/.cache/huggingface/token, put it there first (commands.txt)"; exit 1; }
[[ $RANK =~ ^[0-3]$ ]] || { echo "$(hostname): worker number '$RANK' is not one of 0 to 3"; exit 1; }
[ -f ~/culturax_nl_files.txt ] && [ -x "$PY" ] || { echo "$(hostname): needs ~/culturax_nl_files.txt (arm W's share) and $PY"; exit 1; }
while read -r f; do [ -s "$f" ] || { echo "$(hostname): missing $f of ~/culturax_nl_files.txt"; exit 1; }; done < ~/culturax_nl_files.txt
# The mix is ~2.5 GB; a trainer on this host also writes logs and its trackio database here
free_gb=$(df -BG --output=avail "$HOME" | tail -1 | tr -dc 0-9)
[ "$free_gb" -ge 6 ] || { echo "$(hostname): $free_gb GB free on the boot disk, the mix needs ~2.5 GB and a margin"; exit 1; }
[ -d "$CODE/.git" ] || git clone -q https://github.com/yhavinga/mbert-jax.git "$CODE"
git -C "$CODE" fetch -q origin flan-decay
git -C "$CODE" checkout -q "$COMMIT"
mkdir -p ~/logs ~/flan_decay
cd "$CODE"
echo "$(hostname): rank $RANK, code $(git rev-parse --short HEAD), $free_gb GB free, $(date +%H:%M:%S)"
HF_HUB_DISABLE_PROGRESS_BARS=1 PYTHONPATH=src "$PY" -u scripts/build_decay_mix.py --rank "$RANK" --web-files ~/culturax_nl_files.txt \
  --out ~/flan_decay --list ~/flan_decay_files.txt --niv2-languages experiments/flan-decay/niv2_languages.json \
  --download-dir /dev/shm/flan_decay $EXTRA
rm -rf /dev/shm/flan_decay
mix=$(cat ~/flan_decay_files.txt)
[ -s "$mix" ] || { echo "$(hostname): $mix of ~/flan_decay_files.txt is missing"; exit 1; }
build=$(basename "$mix" .parquet); build=${build#mix-}
gsutil -q cp ~/flan_decay/manifest-$build.json "$RUN_ROOT/mix/w$RANK.json"
echo "$(hostname): ready at $(date +%H:%M:%S), $(du -h "$mix" | cut -f1) in $mix, $(df -h ~ | tail -1 | awk '{print $4}') free"
