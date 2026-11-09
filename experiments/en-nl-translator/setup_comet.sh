#!/bin/bash
# On a worker: the venv of comet_score.py and COMET-22 itself, both in /dev/shm/translator (the boot disk of mbert-mix
# has ~12 GB free). The venv is comet-requirements.txt (unbabel-comet 2.2.7, torch 2.9.1 for CPU, numpy < 2 and
# transformers < 5, which is why it is a venv of its own); the model is Unbabel/wmt22-comet-da and the files of its
# encoder at fixed revisions (comet_score.py setup: sha256 of the checkpoint, every parameter from the checkpoint, and
# two sanity triples).
#   bash setup_comet.sh
# A second start keeps what is there and checks it again.
set -euo pipefail
SHM=${SHM:-/dev/shm/translator}
VENV=$SHM/comet-venv
CODE=$(cd "$(dirname "$0")" && pwd)
export PATH=$HOME/.local/bin:$PATH UV_CACHE_DIR=$SHM/uv-cache UV_PYTHON_INSTALL_DIR=$SHM/python
export HF_HUB_DISABLE_PROGRESS_BARS=1 HF_XET_CACHE=$SHM/xet-cache
command -v uv > /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
command -v uv > /dev/null || { echo "$(hostname): uv is not on PATH after installing it"; exit 1; }
[ -x "$VENV/bin/python" ] || uv venv -q --python 3.12 "$VENV"
VIRTUAL_ENV=$VENV UV_TORCH_BACKEND=cpu uv pip install -q -r "$CODE/comet-requirements.txt"
"$VENV/bin/python" -c "import importlib.metadata as m; print('$(hostname):', {p: m.version(p) for p in ('unbabel-comet', 'torch', 'pytorch-lightning', 'transformers', 'numpy', 'setuptools')})"
"$VENV/bin/python" -I -u "$CODE/comet_score.py" setup --root "$SHM/comet"
echo "$(hostname): COMET ready in $SHM/comet, $(df -h /dev/shm | tail -1 | awk '{print $4}') free in /dev/shm"
