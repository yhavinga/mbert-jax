#!/bin/bash
# On every worker of mbert-corpus (v4-32, us-central2-b, 4 hosts), once: the code, the venv of the CulturaX runs
# (vm.lock), the held-out file and this host's share of FineWeb-2 nl (scripts/stage_share.py with
# experiments/corpus-runs/fineweb2_plan.json): 36,825,317 or 36,825,318 documents, every file of the corpus in
# segments of 1,000 rows. Each host downloads all 45 files from Hugging Face (190 GB, 6 at a time into /dev/shm, up
# to 29 GB of its 200) and keeps its own rows, ~37 GB, listed in ~/fineweb2_nl_files.txt in the order of the plan.
# The boot disk writes 27 MB/s, so that takes ~25 minutes.
# Needs ~/.cache/huggingface/token on the worker (commands.txt). A second start keeps what is staged.
# Environment overrides, for the smoke test only: SCRIPTS RANK.
set -euo pipefail
COMMIT=${COMMIT:?set COMMIT to the commit of commands.txt}
SCRIPTS=${SCRIPTS:-gs://t5-dutch-english-usc2/mbert/corpus-runs/scripts}
RANK=${RANK:-$(hostname | sed 's/.*-w-//')}
cd "$HOME"
[ -s ~/.cache/huggingface/token ] || { echo "$(hostname): no ~/.cache/huggingface/token, put it there first (commands.txt)"; exit 1; }
[[ $RANK =~ ^[0-3]$ ]] || { echo "$(hostname): worker number '$RANK' is not one of 0 to 3"; exit 1; }
[ -d mbert-jax/.git ] || git clone -q https://github.com/yhavinga/mbert-jax.git mbert-jax
git -C mbert-jax fetch -q origin corpus-runs
git -C mbert-jax checkout -q "$COMMIT"
gsutil -q cp "$SCRIPTS/vm.lock" "$SCRIPTS/start_fineweb2.sh" "$HOME/"
cd mbert-jax
if ! command -v uv > /dev/null; then curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; fi
[ -x .venv/bin/python ] || uv venv -q --python 3.12 .venv
uv pip install -q --python .venv/bin/python -r "$HOME/vm.lock"
mkdir -p ~/logs
.venv/bin/python -c "import importlib.metadata as m; print('$(hostname)', {p: m.version(p) for p in ['jax','jaxlib','libtpu','flax','optax','orbax-checkpoint','datasets','transformers','numpy','grain','gcsfs','trackio']})"
gsutil -q cp gs://t5-dutch-english-usc2/mbert/heldout/nl_part_00097_first40000.parquet ~/heldout.parquet
[ "$(md5sum ~/heldout.parquet | cut -d' ' -f1)" = b2f579a876d06eef40c94270537bb7af ] || { echo "$(hostname): held-out file is not the long run's"; exit 1; }
HF_HUB_DISABLE_PROGRESS_BARS=1 PYTHONPATH=src .venv/bin/python -u scripts/stage_share.py stage --corpus fineweb2 --rank "$RANK" --out ~/fineweb2_nl --list ~/fineweb2_nl_files.txt --download-dir /dev/shm/stage_fineweb2
rmdir /dev/shm/stage_fineweb2
[ "$(wc -l < ~/fineweb2_nl_files.txt)" = 45 ] || { echo "$(hostname): ~/fineweb2_nl_files.txt does not list 45 files"; exit 1; }
while read -r f; do [ -s "$f" ] || { echo "$(hostname): missing $f"; exit 1; }; done < ~/fineweb2_nl_files.txt
gsutil -q cp ~/fineweb2_nl/share.json "${SCRIPTS%/scripts}/shares/fineweb2-w$RANK.json"
echo "$(hostname): ready at $(git rev-parse --short HEAD), $(du -sh ~/fineweb2_nl | cut -f1) of data, $(df -h ~ | tail -1 | awk '{print $4}') free"
