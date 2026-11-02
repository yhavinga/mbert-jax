#!/bin/bash
# On every worker of the v4-32 of the mC4 run (mbert-mix, us-central2-b, 4 hosts), once: the code in a clone of its own,
# ~/mbert-corpus with its own venv of vm.lock (~/mbert-jax stays at the commit of the mix run, which start_mix.sh resumes),
# the held-out file and this host's share of mC4 nl cleaned (scripts/stage_share.py with experiments/corpus-runs/mc4_plan.json):
# 16,136,473 or 16,136,474 documents, every file in segments of 1,000 rows. Each host downloads all 1,024 json.gz files
# from Hugging Face (62 GB, 16 at a time into /dev/shm) and keeps its own rows as parquet, ~13 GB, listed in
# ~/mc4_nl_files.txt in the order of the plan. The CulturaX share of mbert-mix (~/culturax_nl) is not touched.
# It can run while the mix run trains; start_mc4.sh waits for the TPU to be free.
# Needs ~/.cache/huggingface/token on the worker (commands.txt). A second start keeps what is staged.
# Environment overrides, for the smoke test only: SCRIPTS RANK.
set -euo pipefail
COMMIT=${COMMIT:?set COMMIT to the commit of commands.txt}
SCRIPTS=${SCRIPTS:-gs://t5-dutch-english-usc2/mbert/corpus-runs/scripts}
RANK=${RANK:-$(hostname | sed 's/.*-w-//')}
cd "$HOME"
[ -s ~/.cache/huggingface/token ] || { echo "$(hostname): no ~/.cache/huggingface/token, put it there first (commands.txt)"; exit 1; }
[[ $RANK =~ ^[0-3]$ ]] || { echo "$(hostname): worker number '$RANK' is not one of 0 to 3"; exit 1; }
[ -d mbert-corpus/.git ] || git clone -q https://github.com/yhavinga/mbert-jax.git mbert-corpus
git -C mbert-corpus fetch -q origin corpus-runs
git -C mbert-corpus checkout -q "$COMMIT"
gsutil -q cp "$SCRIPTS/vm.lock" "$SCRIPTS/start_mc4.sh" "$HOME/"
cd mbert-corpus
if ! command -v uv > /dev/null; then curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; fi
[ -x .venv/bin/python ] || uv venv -q --python 3.12 .venv
uv pip install -q --python .venv/bin/python -r "$HOME/vm.lock"
mkdir -p ~/logs
.venv/bin/python -c "import importlib.metadata as m; print('$(hostname)', {p: m.version(p) for p in ['jax','jaxlib','libtpu','flax','optax','orbax-checkpoint','datasets','transformers','numpy','grain','gcsfs','trackio']})"
gsutil -q cp gs://t5-dutch-english-usc2/mbert/heldout/nl_part_00097_first40000.parquet ~/heldout.parquet
[ "$(md5sum ~/heldout.parquet | cut -d' ' -f1)" = b2f579a876d06eef40c94270537bb7af ] || { echo "$(hostname): held-out file is not the long run's"; exit 1; }
HF_HUB_DISABLE_PROGRESS_BARS=1 PYTHONPATH=src .venv/bin/python -u scripts/stage_share.py stage --corpus mc4 --rank "$RANK" --out ~/mc4_nl --list ~/mc4_nl_files.txt --download-dir /dev/shm/stage_mc4 --workers 16
rmdir /dev/shm/stage_mc4
[ "$(wc -l < ~/mc4_nl_files.txt)" = 1024 ] || { echo "$(hostname): ~/mc4_nl_files.txt does not list 1024 files"; exit 1; }
while read -r f; do [ -s "$f" ] || { echo "$(hostname): missing $f"; exit 1; }; done < ~/mc4_nl_files.txt
gsutil -q cp ~/mc4_nl/share.json "${SCRIPTS%/scripts}/shares/mc4-w$RANK.json"
echo "$(hostname): ready at $(git rev-parse --short HEAD), $(du -sh ~/mc4_nl | cut -f1) of data, $(df -h ~ | tail -1 | awk '{print $4}') free"
