#!/bin/bash
# On the v6e-8 of the Gemma 4 pilot (translate.py --model gemma4-31b): the environment and the data, in
# /dev/shm/translate-generic. Each step is skipped when its result is there.
#   bash setup_generic.sh            venv-fork, tokenizers, data
#   bash setup_generic.sh release    also venv-025: MaxText 0.2.5 from PyPI, for translate_checks.py decode only
#   bash setup_generic.sh venv       venv-fork and tokenizers only, without the pilot's data (bt_natural.py)
# venv-fork: MaxText from the fork yhavinga/maxtext, branch fix-nnx-decode-cache-abstract-init (upstream main d1c0d0afe
# of 9 Nov with fix 1: the NNX decode cache made from shapes inside the init jit; upstream main has since 8 Oct the
# sliding-window decode mask measured in true positions, commit 2bbe8006c), installed from git at a fixed commit with
# the fork's tpu-requirements.txt, as the session of notes/maxtext-fixes.md did on a v6e-8; plus what the measures and
# MetricX need. Without JetStream: translate.py registers the stub ResultTokens as a pytree where it is not one yet.
# Data, each from its own source to this VM: the pilot pairs once from us-central2 (1.4 MB, by permission), the
# Handelingen from Hugging Face (yhavinga/dutch-handelingen-modern @ abdba14, 8 parquet files, 1.84 GB), three byte
# ranges of 8 MB of Nemotron's chat.jsonl (HTTP range, pinned revision), the tokenizers of Gemma 4 and TranslateGemma
# 27B. Needs ~/.cache/huggingface/token (put there through stdin; never printed).
set -euo pipefail
SHM=/dev/shm/translate-generic
FORK_COMMIT=2249474b93b971d8248520e3a09b94b4ce626ce4
GEMMA4=google/gemma-4-31B-it
GEMMA4_REV=842da3794eaa0b77d5f08bae87a17459d91ff475
TG27=google/translategemma-27b-it
TG27_REV=7d10f0b72f89a2d0f268cea30727d8b77c0d25c2
HANDELINGEN=yhavinga/dutch-handelingen-modern
HANDELINGEN_REV=abdba14ac9271b6d43ccefe6143202b7bdd4e900
NEMOTRON=nvidia/Nemotron-SFT-Instruction-Following-Chat-v3
PILOT=gs://t5-dutch-english-usc2/mbert/translator/pilot/pairs.parquet
export PATH=$HOME/.local/bin:$PATH UV_CACHE_DIR=$SHM/uv-cache UV_PYTHON_INSTALL_DIR=$SHM/python HF_HUB_DISABLE_PROGRESS_BARS=1
[ -s ~/.cache/huggingface/token ] || { echo "$(hostname): no ~/.cache/huggingface/token"; exit 1; }
ZONE=$(curl -fs -H Metadata-Flavor:Google http://metadata.google.internal/computeMetadata/v1/instance/zone | awk -F/ '{print $NF}')
case ${ZONE%-*} in
  us-east1) BUCKET=gs://t5-dutch-english-use1 ;;
  europe-west4) BUCKET=gs://t5-dutch-english ;;
  *) echo "no bucket for zone '$ZONE'"; exit 1 ;;
esac
mkdir -p "$SHM/data" ~/logs
command -v uv > /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
command -v uv > /dev/null || { echo "$(hostname): uv is not on PATH after installing it"; exit 1; }

venv_report() {
  "$1/bin/python" -c "import importlib.metadata as m, json; d = m.distribution('maxtext'); u = json.loads(d.read_text('direct_url.json') or '{}'); print('$(basename "$1"):', {p: m.version(p) for p in ('maxtext', 'jax', 'jaxlib', 'libtpu', 'flax', 'orbax-checkpoint', 'transformers', 'tokenizers', 'pyarrow')}, u.get('vcs_info', {}).get('commit_id'))"
}

if [ "${1:-}" = release ]; then
  V=$SHM/venv-025
  [ -x "$V/bin/python" ] || uv venv -q --python 3.12 "$V"
  if ! VIRTUAL_ENV=$V uv pip show maxtext 2> /dev/null | grep -q "^Version: 0.2.5$"; then
    VIRTUAL_ENV=$V uv pip install -q "maxtext[tpu]==0.2.5" --resolution=lowest
  fi
  VIRTUAL_ENV=$V uv pip install -q pyyaml
  venv_report "$V"
  exit 0
fi

V=$SHM/venv-fork
if [ ! -s "$V/done.txt" ]; then
  [ -x "$V/bin/python" ] || uv venv -q --python 3.12 "$V"
  SRC=$SHM/maxtext-src
  [ -d "$SRC/.git" ] || git clone -q https://github.com/yhavinga/maxtext.git "$SRC"
  git -C "$SRC" fetch -q origin "$FORK_COMMIT"
  git -C "$SRC" checkout -q "$FORK_COMMIT"
  VIRTUAL_ENV=$V uv pip install -q -r "$SRC/src/dependencies/requirements/generated_requirements/tpu-requirements.txt"
  VIRTUAL_ENV=$V uv pip install -q --no-deps "maxtext @ git+https://github.com/yhavinga/maxtext.git@$FORK_COMMIT"
  VIRTUAL_ENV=$V uv pip install -q sacrebleu==2.5.1 langid==1.1.6 pytest
  venv_report "$V" | tee "$V/done.txt"
fi
cat "$V/done.txt"
PY=$V/bin/python
# MetricX-24's bfloat16 releases have only pytorch_model*.bin (no safetensors): metricx_qe.py reads them with torch on
# the CPU. Outside the done.txt step, so a venv-fork made before this line gets it too
"$PY" -c "import torch" 2> /dev/null || VIRTUAL_ENV=$V UV_TORCH_BACKEND=cpu uv pip install -q torch==2.9.1

snapshot() {  # repo revision dir [patterns...]
  local repo=$1 rev=$2 dir=$3; shift 3
  local patterns
  patterns=$(printf "'%s'," "$@")
  [ -s "$dir/tokenizer.json" ] || "$PY" -c "from huggingface_hub import snapshot_download as s; s('$repo', revision='$rev', local_dir='$dir', allow_patterns=[$patterns])"
}
snapshot "$GEMMA4" "$GEMMA4_REV" "$SHM/tokenizer-gemma4" 'tokenizer*' chat_template.jinja config.json generation_config.json
snapshot "$TG27" "$TG27_REV" "$SHM/tokenizer-tg27b" 'tokenizer*' special_tokens_map.json added_tokens.json chat_template.jinja config.json generation_config.json
if [ "${1:-}" = venv ]; then
  echo "$(hostname): venv-fork and tokenizers ready in $SHM"
  exit 0
fi

D=$SHM/data
[ -s "$D/pairs.parquet" ] || gsutil -q cp "$PILOT" "$D/pairs.parquet"
if [ ! -s "$D/handelingen/done.txt" ]; then
  "$PY" -c "from huggingface_hub import snapshot_download as s; s('$HANDELINGEN', repo_type='dataset', revision='$HANDELINGEN_REV', local_dir='$D/handelingen-repo', allow_patterns=['data/train-*.parquet'])"
  mkdir -p "$D/handelingen" && mv "$D"/handelingen-repo/data/train-*.parquet "$D/handelingen/"
  echo "$HANDELINGEN@$HANDELINGEN_REV $(ls "$D/handelingen" | wc -l) files $(du -sh "$D/handelingen" | cut -f1)" > "$D/handelingen/done.txt"
fi
cat "$D/handelingen/done.txt"
if [ ! -s "$D/nemotron/done.txt" ]; then
  mkdir -p "$D/nemotron"
  rev=$(curl -fsS -H @<(printf 'Authorization: Bearer %s\n' "$(cat ~/.cache/huggingface/token)") \
    "https://huggingface.co/api/datasets/$NEMOTRON" | "$PY" -c "import json, sys; print(json.load(sys.stdin)['sha'])")
  url="https://huggingface.co/datasets/$NEMOTRON/resolve/$rev/data/chat.jsonl"
  : > "$D/nemotron/chat-ranges.jsonl"
  for start in 0 4000000000 10000000000; do
    curl -fsSL -H @<(printf 'Authorization: Bearer %s\n' "$(cat ~/.cache/huggingface/token)") \
      -r "$start-$((start + 7999999))" "$url" -o "$D/nemotron/part-$start.jsonl"
    # A range starts and ends inside a line: a newline between ranges keeps the cut pieces apart
    cat "$D/nemotron/part-$start.jsonl" >> "$D/nemotron/chat-ranges.jsonl"
    echo >> "$D/nemotron/chat-ranges.jsonl"
  done
  echo "$NEMOTRON@$rev data/chat.jsonl bytes 0, 4e9, 1e10 (+8 MB each) $(du -sh "$D/nemotron/chat-ranges.jsonl" | cut -f1)" > "$D/nemotron/done.txt"
fi
cat "$D/nemotron/done.txt"
gsutil -q cp "$BUCKET/mbert/translator/scripts/pairs30-pids.txt" "$D/pairs30-pids.txt"
"$PY" -c "import jax; d = jax.devices(); print('$(hostname):', len(d), d[0].platform, d[0].device_kind)" 2>&1 | tail -1
echo "$(hostname): ready in $SHM, $(df -h /dev/shm | tail -1 | awk '{print $4}') free in /dev/shm"
