#!/bin/bash
# On every worker of mbert-mix, once, after setup_mbert_mask.sh: the code of the 2048 phase in a clone of its own.
# ~/mbert-jax stays at the commit of the mix run, which start_mix.sh resumes; the venv of that setup (vm.lock)
# serves this code too, as it needs no other package.
set -euo pipefail
COMMIT=${COMMIT:?set COMMIT to the commit of commands.txt}
SCRIPTS=${SCRIPTS:-gs://t5-dutch-english-usc2/mbert/phase2048/scripts}
cd "$HOME"
[ -x mbert-jax/.venv/bin/python ] && [ -f culturax_nl_files.txt ] && [ -f heldout.parquet ] || { echo "$(hostname): run setup_mbert_mask.sh first"; exit 1; }
[ "$(md5sum heldout.parquet | cut -d' ' -f1)" = b2f579a876d06eef40c94270537bb7af ] || { echo "$(hostname): held-out file is not the long run's"; exit 1; }
[ -d mbert-p2048/.git ] || git clone -q https://github.com/yhavinga/mbert-jax.git mbert-p2048
git -C mbert-p2048 fetch -q origin phase2048
git -C mbert-p2048 checkout -q "$COMMIT"
gsutil -q cp "$SCRIPTS/start_phase.sh" "$SCRIPTS/eval_phase.sh" "$SCRIPTS/dumb_phase.sh" "$SCRIPTS/verify_ckpt.py" "$HOME/"
mkdir -p ~/logs
echo "$(hostname): ready at $(git -C mbert-p2048 rev-parse --short HEAD)"
