#!/bin/bash
# On a worker of mbert-corpus: the environment of the en-nl translator in /dev/shm/translator, and the word lists of the
# passage selection. TranslateGemma itself runs in FLAN-nl's venv (/dev/shm/flan_nl/venv, built by
# experiments/flan-nl/setup.sh), which this script reads and never changes.
#   bash setup.sh
# venv: the selection's CPU tools, JAX for TPU (MetricX-24 on the TPU) and CPU torch (only for the reference check of
# the MetricX port). resources/: OpenTaal's word list at a fixed commit, and kaikki.org's extract of the English
# Wiktionary's Dutch entries (the source of the idiom list; not versioned upstream, so its Last-Modified and sha256 go
# into resources/manifest.txt). Needs ~/.cache/huggingface/token; the token is never printed.
# RESOURCES=0 builds only the venv (the evaluation of run_baseline.sh needs no word lists).
set -euo pipefail
SHM=/dev/shm/translator
VENV=$SHM/venv
PY=$VENV/bin/python
RES=$SHM/resources
OPENTAAL_COMMIT=b250510dda431785f962019167d1415198ff3905
KAIKKI_URL=https://kaikki.org/dictionary/Dutch/kaikki.org-dictionary-Dutch.jsonl
export PATH=$HOME/.local/bin:$PATH UV_CACHE_DIR=$SHM/uv-cache UV_PYTHON_INSTALL_DIR=$SHM/python
[ -s ~/.cache/huggingface/token ] || { echo "$(hostname): no ~/.cache/huggingface/token"; exit 1; }
mkdir -p "$RES" ~/logs
command -v uv > /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -x "$PY" ] || uv venv -q --python 3.12 "$VENV"
# jax 0.11.1 with its libtpu is what MaxText runs on these v4 hosts (FLAN-nl's venv)
VIRTUAL_ENV=$VENV uv pip install -q "jax[tpu]==0.11.1" numpy==2.1.3 pyarrow==25.0.1 "huggingface_hub>=1.0" \
  tokenizers==0.23.2 transformers==5.17.0 sentencepiece==0.2.2 safetensors==0.8.0 langid==1.1.6 sacrebleu==2.5.1 \
  protobuf pytest
VIRTUAL_ENV=$VENV UV_TORCH_BACKEND=cpu uv pip install -q torch==2.9.1
"$PY" -c "import importlib.metadata as m; print('$(hostname):', {p: m.version(p) for p in ('jax', 'libtpu', 'transformers', 'tokenizers', 'pyarrow', 'langid', 'sacrebleu', 'torch')})"

if [ "${RESOURCES:-1}" = 0 ]; then
  echo "$(hostname): venv ready in $VENV (no word lists: RESOURCES=0), $(df -h /dev/shm | tail -1 | awk '{print $4}') free in /dev/shm"
  exit 0
fi
if [ ! -s "$RES/opentaal-wordlist.txt" ]; then
  curl -fsSL "https://raw.githubusercontent.com/OpenTaal/opentaal-wordlist/$OPENTAAL_COMMIT/wordlist.txt" -o "$RES/opentaal-wordlist.txt.part"
  mv "$RES/opentaal-wordlist.txt.part" "$RES/opentaal-wordlist.txt"
  echo "opentaal-wordlist.txt OpenTaal/opentaal-wordlist@$OPENTAAL_COMMIT sha256 $(sha256sum "$RES/opentaal-wordlist.txt" | cut -c1-64)" >> "$RES/manifest.txt"
fi
if [ ! -s "$RES/kaikki-dutch.jsonl" ]; then
  modified=$(curl -fsSI "$KAIKKI_URL" | grep -i '^last-modified:' | cut -d' ' -f2- | tr -d '\r')
  curl -fsSL "$KAIKKI_URL" -o "$RES/kaikki-dutch.jsonl.part"
  mv "$RES/kaikki-dutch.jsonl.part" "$RES/kaikki-dutch.jsonl"
  echo "kaikki-dutch.jsonl $KAIKKI_URL last-modified '$modified' sha256 $(sha256sum "$RES/kaikki-dutch.jsonl" | cut -c1-64)" >> "$RES/manifest.txt"
fi
cat "$RES/manifest.txt"
echo "$(hostname): ready in $SHM, $(df -h /dev/shm | tail -1 | awk '{print $4}') free in /dev/shm"
