#!/bin/bash
# On a worker of mbert-corpus: the environment of the LoRA smoke test (MaxText train_sft on Tunix) in
# /dev/shm/translator/smoke-lora, and on one worker the conversion of TranslateGemma 12B into a MaxText checkpoint.
#   bash setup_sft.sh            the venv: MaxText 0.2.5 as FLAN-nl pins it, plus Tunix at the commit MaxText 0.2.5 pins
#   bash setup_sft.sh convert    also: TranslateGemma 12B from Hugging Face through /dev/shm into $CKPT, done.txt last
# FLAN-nl's venv (/dev/shm/flan_nl/venv) has no Tunix and stays as it is, so this is a venv of its own. Needs
# ~/.cache/huggingface/token for the conversion (the model is gated); the token is never printed.
set -euo pipefail
SHM=/dev/shm/translator/smoke-lora
VENV=$SHM/venv
PY=$VENV/bin/python
MAXTEXT_PIN=0.2.5
# maxtext 0.2.5: dependencies/extra_deps/post_train_github_deps.txt
TUNIX=https://github.com/google/tunix/archive/a8d70582f1e2f1fb65973210989e0e148b5ef7ad.zip
REVISION=d1b225e1caa17f1ddc7e62065d8637d0923f34e2  # google/translategemma-12b-it, as experiments/flan-nl/setup.sh
CKPT=${CKPT:-gs://t5-dutch-english-usc2/mbert/translator/smoke-lora/translategemma-12b-maxtext}
CODE=$(cd "$(dirname "$0")" && pwd)
export PATH=$HOME/.local/bin:$PATH UV_CACHE_DIR=/dev/shm/translator/uv-cache UV_PYTHON_INSTALL_DIR=/dev/shm/translator/python
mkdir -p "$SHM" ~/logs
command -v uv > /dev/null || { echo "$(hostname): no uv (experiments/en-nl-translator/setup.sh installs it)"; exit 1; }

[ -x "$PY" ] || uv venv -q --python 3.12 "$VENV"
if ! VIRTUAL_ENV=$VENV uv pip show maxtext 2> /dev/null | grep -q "^Version: $MAXTEXT_PIN$"; then
  # --resolution=lowest is how MaxText pins its extras (experiments/flan-nl/setup.sh)
  VIRTUAL_ENV=$VENV uv pip install -q "maxtext[tpu]==$MAXTEXT_PIN" --resolution=lowest
fi
# Without --upgrade uv keeps every installed version that satisfies Tunix, so JAX, Flax and qwix stay MaxText's
VIRTUAL_ENV=$VENV uv pip show google-tunix > /dev/null 2>&1 || VIRTUAL_ENV=$VENV uv pip install -q "google-tunix @ $TUNIX"
"$PY" -c "import importlib.metadata as m; print('$(hostname):', {p: m.version(p) for p in ('maxtext', 'google-tunix', 'jax', 'jaxlib', 'libtpu', 'flax', 'qwix', 'optax', 'orbax-checkpoint', 'grain', 'datasets', 'transformers', 'tokenizers')})"
JAX_PLATFORMS=cpu "$PY" -c "import tunix.sft.peft_trainer, maxtext.trainers.post_train.sft.train_sft; print('imports ok')"

[ "${1:-}" = convert ] || { echo "$(hostname): environment ready in $VENV, $(df -h /dev/shm | tail -1 | awk '{print $4}') free in /dev/shm"; exit 0; }

if gsutil -q stat "$CKPT/done.txt"; then
  echo "$(hostname): $CKPT is done before"
  exit 0
fi
[ -s ~/.cache/huggingface/token ] || { echo "$(hostname): no ~/.cache/huggingface/token"; exit 1; }
# Only the conversion reads bf16 safetensors, which numpy cannot
VIRTUAL_ENV=$VENV UV_TORCH_BACKEND=cpu uv pip install -q torch==2.9.1
HF=$SHM/hf-12b
echo "$(hostname): downloading google/translategemma-12b-it@$REVISION to $HF at $(date +%H:%M:%S)"
HF_HUB_DISABLE_PROGRESS_BARS=1 "$PY" -c "from huggingface_hub import snapshot_download as s; s('google/translategemma-12b-it', revision='$REVISION', local_dir='$HF')"
echo "$(hostname): converting at $(date +%H:%M:%S), $(du -sh "$HF" | cut -f1) of weights"
gsutil -q ls "$CKPT/" > /dev/null 2>&1 && { echo "$(hostname): $CKPT exists without done.txt; remove it first"; exit 1; }
# FLAN-nl's converter: MaxText's own, unscanned, with the tensor names of TranslateGemma's files
JAX_PLATFORMS=cpu "$PY" -u "$CODE/convert.py" --hf-dir "$HF" --model gemma3-12b --out "$CKPT"
gsutil -q stat "$CKPT/0/commit_success.txt" || { echo "$(hostname): no $CKPT/0/commit_success.txt after the conversion"; exit 1; }
echo "google/translategemma-12b-it@$REVISION converted by MaxText $MAXTEXT_PIN (experiments/flan-nl/convert.py) on $(hostname) at $(date -u +%FT%TZ)" | gsutil -q cp - "$CKPT/done.txt"
echo "$(hostname): $CKPT done at $(date +%H:%M:%S)"
