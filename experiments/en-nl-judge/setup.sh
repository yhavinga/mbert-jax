#!/bin/bash
# On the v6e-8 host of the en-nl judge: the environment, and once per region Gemma 4 31B-it as a MaxText checkpoint in
# the bucket of that region.
#   bash setup.sh             the venv (MaxText 0.2.5, JAX for TPU), the tokenizer, a check of the 8 chips
#   bash setup.sh convert     also: Gemma 4 31B-it from Hugging Face through /dev/shm into $CKPT, marked done last
#   bash setup.sh reference   the venv of hf_reference.py (transformers and torch on the CPU) and the HF weights
# Everything lives in /dev/shm/judge: a new spot VM starts empty, and the checkpoint, data and scores are in the
# bucket. The bucket is that of the VM's own region, read from the metadata server, so nothing crosses regions.
# Needs ~/.cache/huggingface/token (put there through stdin; it is never printed).
set -euo pipefail
REPO=google/gemma-4-31B-it
REVISION=842da3794eaa0b77d5f08bae87a17459d91ff475
ZONE=$(curl -fs -H Metadata-Flavor:Google http://metadata.google.internal/computeMetadata/v1/instance/zone | awk -F/ '{print $NF}')
case ${ZONE%-*} in
  us-east1) BUCKET=gs://t5-dutch-english-use1 ;;
  europe-west4) BUCKET=gs://t5-dutch-english ;;
  us-central2) BUCKET=gs://t5-dutch-english-usc2 ;;
  *) echo "no judge bucket for zone '$ZONE'"; exit 1 ;;
esac
JUDGE=$BUCKET/mbert/translator/judge
CKPT=${CKPT:-$JUDGE/gemma-4-31b-it-maxtext}
SHM=/dev/shm/judge
VENV=$SHM/venv
PY=$VENV/bin/python
MAXTEXT_PIN=0.2.5
HF=$SHM/hf-31b
export PATH=$HOME/.local/bin:$PATH UV_CACHE_DIR=$SHM/uv-cache UV_PYTHON_INSTALL_DIR=$SHM/python
# The boot disks of mbert-corpus (us-central2) have a few GB free: Hugging Face's chunk cache and MaxText's HLO dump
# (which it writes to /tmp/xla_dump whenever XLA_FLAGS is empty) stay off them
export HF_XET_CACHE=$SHM/xet-cache XLA_FLAGS=${XLA_FLAGS:---xla_dump_hlo_as_text=false}
[ -s ~/.cache/huggingface/token ] || { echo "$(hostname): no ~/.cache/huggingface/token"; exit 1; }
mkdir -p "$SHM" ~/logs
command -v uv > /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
command -v uv > /dev/null || { echo "$(hostname): uv is not on PATH after installing it"; exit 1; }

download() {  # the full snapshot of the model, ~62 GB, from Hugging Face straight to this VM
  local py=$1
  echo "$(hostname): downloading $REPO@$REVISION to $HF at $(date +%H:%M:%S)"
  HF_HUB_DISABLE_PROGRESS_BARS=1 "$py" -c "from huggingface_hub import snapshot_download as s; s('$REPO', revision='$REVISION', local_dir='$HF')"
  echo "$(hostname): $(du -sh "$HF" | cut -f1) in $HF at $(date +%H:%M:%S)"
}

if [ "${1:-}" = reference ]; then
  RVENV=$SHM/hfvenv
  [ -x "$RVENV/bin/python" ] || uv venv -q --python 3.12 "$RVENV"
  VIRTUAL_ENV=$RVENV UV_TORCH_BACKEND=cpu uv pip install -q torch==2.15.0 transformers==5.22.2 pyarrow numpy
  "$RVENV/bin/python" -c "import importlib.metadata as m; print('$(hostname):', {p: m.version(p) for p in ('torch', 'transformers', 'tokenizers', 'huggingface-hub')})"
  [ -s "$HF/model.safetensors.index.json" ] || download "$RVENV/bin/python"
  exit 0
fi

# MaxText 0.2.5 needs Python 3.12; --resolution=lowest is how MaxText pins its extras (the floors are its tested
# lock), as in FLAN-nl's setup.sh
[ -x "$PY" ] || uv venv -q --python 3.12 "$VENV"
if ! VIRTUAL_ENV=$VENV uv pip show maxtext 2> /dev/null | grep -q "^Version: $MAXTEXT_PIN$"; then
  VIRTUAL_ENV=$VENV uv pip install -q "maxtext[tpu]==$MAXTEXT_PIN" --resolution=lowest
fi
"$PY" -c "import importlib.metadata as m; print('$(hostname):', {p: m.version(p) for p in ('maxtext', 'jax', 'jaxlib', 'libtpu', 'flax', 'orbax-checkpoint', 'transformers', 'tokenizers', 'pyarrow')})"

# The tokenizer and the template of the model, small
TOK=$SHM/tokenizer
if [ ! -s "$TOK/tokenizer.json" ]; then
  "$PY" -c "from huggingface_hub import snapshot_download as s; s('$REPO', revision='$REVISION', local_dir='$TOK', allow_patterns=['tokenizer*', 'chat_template.jinja', 'config.json', 'generation_config.json'])"
fi

[ "${1:-}" = convert ] || {
  if [ -n "$(sudo lsof -t /dev/accel* /dev/vfio/* 2> /dev/null)" ]; then
    echo "$(hostname): the TPU is in use; the device check is skipped"
  else
    CHIPS=8
    if [ "${ZONE%-*}" = us-central2 ]; then
      # A host of the v4-32 mbert-corpus as a TPU of its own 4 chips (as FLAN-nl's translate.sh): without these
      # variables JAX takes the slice from the metadata and waits for the other hosts
      export TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 TPU_PROCESS_BOUNDS=1,1,1 TPU_VISIBLE_CHIPS=0,1,2,3 TPU_PROCESS_PORT=8476
      export TPU_PROCESS_ADDRESSES=localhost:8476 CLOUD_TPU_TASK_ID=0 ALLOW_MULTIPLE_LIBTPU_LOAD=1
      CHIPS=4
    fi
    timeout 600 "$PY" -c "import jax; d = jax.devices(); print('$(hostname):', len(d), d[0].platform, d[0].device_kind); assert len(d) == $CHIPS and d[0].platform == 'tpu'"
  fi
  echo "$(hostname): environment ready in $VENV, $(df -h /dev/shm | tail -1 | awk '{print $4}') free in /dev/shm, bucket $JUDGE"
  exit 0
}

if gsutil -q stat "$CKPT/done.txt"; then
  echo "$(hostname): $CKPT is done before"
  exit 0
fi
# Only the conversion reads bf16 safetensors, which numpy cannot; the judge itself has no torch
VIRTUAL_ENV=$VENV UV_TORCH_BACKEND=cpu uv pip install -q torch==2.9.1
[ -s "$HF/model.safetensors.index.json" ] || download "$PY"
# Written to /dev/shm first and then copied: a preempted VM leaves at most a partial copy in the bucket, which the next
# rsync completes, and done.txt is written only after a complete one (orbax renames its temporary directory to 0/ on
# a local file system when the save is complete, so 0/items exists only for a whole checkpoint)
LOCAL_CKPT=$SHM/ckpt
if [ ! -d "$LOCAL_CKPT/0/items" ]; then
  echo "$(hostname): converting at $(date +%H:%M:%S)"
  # MaxText's own converter: lazy loading, unscanned layers, no vision tower. The tensor names of this snapshot
  # (model.language_model.layers.N...) are the ones its Gemma 4 mapping asks for.
  JAX_PLATFORMS=cpu "$PY" -u -m maxtext.checkpoint_conversion.to_maxtext model_name=gemma4-31b scan_layers=false \
    use_multimodal=false skip_jax_distributed_system=True hardware=cpu "base_output_directory=$LOCAL_CKPT" \
    run_name=gemma4_31b_conversion "--hf_model_path=$HF" --lazy_load_tensors=True --save_dtype=bfloat16
fi
[ -d "$LOCAL_CKPT/0/items" ] || { echo "$(hostname): no $LOCAL_CKPT/0/items after the conversion"; exit 1; }
echo "$(hostname): $(du -sh "$LOCAL_CKPT" | cut -f1) converted; copying to $CKPT at $(date +%H:%M:%S)"
gsutil -m -q rsync -r "$LOCAL_CKPT" "$CKPT"
echo "$REPO@$REVISION converted by MaxText $MAXTEXT_PIN on $(hostname) at $(date -u +%FT%TZ)" | gsutil -q cp - "$CKPT/done.txt"
echo "$(hostname): $CKPT done at $(date +%H:%M:%S)"
