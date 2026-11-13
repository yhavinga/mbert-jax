#!/bin/bash
# The memory and step test of LoRA SFT on Gemma 4 26B-A4B-it (MoE) and Gemma 4 31B-it (Yeb, 13 Nov ~12:50: "een test met
# lora op de MoE"), on mbert-mix with the translator's SFT stack (MaxText 0.2.5, Tunix, qwix: run_sft.sh, sft_lora.py),
# one stage per call on a worker; translator_g4lora_run.sh (co1) starts the stages and waits for their EXIT lines. Every
# stage that takes longer than a few seconds runs detached from ssh, with a fresh log ~/logs/<job>.log that ends in
# "EXIT <code>". A measurement: 8 steps per variant, no arm is trained.
#   MODEL=gemma4-26b bash run_g4lora.sh convert  a worker (CPU, nice/ionice): the Hugging Face snapshot at its pinned
#                                                revision to /dev/shm, MaxText 0.2.5's to_maxtext (unscanned, text only,
#                                                bf16) straight into $CK/<dir>/, done.txt last
#                                                                            (job g4l_convert_<model>_$TAG)
# MODEL: gemma4-26b (google/gemma-4-26B-A4B-it) or gemma4-31b (google/gemma-4-31B-it). TAG names the jobs of one chain.
set -euo pipefail
STAGE=${1:?a stage: convert}
S=gs://t5-dutch-english-usc2/mbert/translator/sft
CK=gs://t5-dutch-english-usc2/mbert/translator/gemma4-ckpt   # us-central2, beside mbert-mix; nothing crosses regions
SHM=/dev/shm/translator
G_SHM=$SHM/g4lora
MAXTEXT_PIN=0.2.5                     # the SFT venv's (setup_sft.sh): the checkpoint's tree is the one its trainer loads
CODE=$(cd "$(dirname "$0")" && pwd)
W=$(hostname | sed 's/.*-w-//')
TAG=${TAG:-manual}
export PATH=$HOME/.local/bin:$PATH UV_CACHE_DIR=$SHM/uv-cache UV_PYTHON_INSTALL_DIR=$SHM/python
mkdir -p "$G_SHM" ~/logs

model_info() {  # MODEL: REPO REVISION DIR (the revisions of translate_models.yml and setup_generic.sh, 13 Nov)
  case $1 in
    gemma4-26b) echo "google/gemma-4-26B-A4B-it 4d7ae4984b7db7de8f8457170b3f1a419ee76d52 gemma-4-26b-a4b-it" ;;
    gemma4-31b) echo "google/gemma-4-31B-it 842da3794eaa0b77d5f08bae87a17459d91ff475 gemma-4-31b-it" ;;
    *) echo "MODEL is gemma4-26b or gemma4-31b, not '$1'" >&2; return 1 ;;
  esac
}

detach() {  # detach NAME COMMAND...: the command in the background, log ~/logs/NAME.log ending in EXIT <code>
  local name=$1
  shift
  [ -e ~/logs/$name.log ] && { echo "w$W: ~/logs/$name.log exists; a new start takes a new TAG"; exit 1; }
  printf '%q ' "$@" > ~/job_$name.sh
  setsid nohup bash -c "bash ~/job_$name.sh > ~/logs/$name.log 2>&1; echo EXIT \$? >> ~/logs/$name.log" \
    > /dev/null 2>&1 < /dev/null &
  echo "started $name on w$W: ~/logs/$name.log"
}

case $STAGE in
convert)
  MODEL=${MODEL:?gemma4-26b or gemma4-31b}
  read -r REPO REV DIR < <(model_info "$MODEL")
  OUT=$CK/$DIR
  if gsutil -q stat "$OUT/done.txt"; then echo "w$W: $OUT is done before"; exit 0; fi
  ! gsutil -q ls "$OUT/" > /dev/null 2>&1 || { echo "w$W: $OUT exists without done.txt; move it aside first"; exit 1; }
  [ -s ~/.cache/huggingface/token ] || { echo "w$W: no ~/.cache/huggingface/token"; exit 1; }
  command -v uv > /dev/null || { echo "w$W: no uv (experiments/en-nl-translator/setup.sh installs it)"; exit 1; }
  V=$G_SHM/venv-convert
  HF=$G_SHM/hf-$MODEL
  NAME=g4l_convert_${MODEL#gemma4-}_$TAG
  # A venv of its own: the SFT venv (setup_sft.sh) is in use by other chains and has no torch, which the conversion
  # needs to read bf16 safetensors (as en-nl-judge's setup.sh). --resolution=lowest pins MaxText's extras as there.
  cat > ~/job_body_$NAME.sh <<EOF
set -euo pipefail
export PATH=$PATH UV_CACHE_DIR=$UV_CACHE_DIR UV_PYTHON_INSTALL_DIR=$UV_PYTHON_INSTALL_DIR HF_HUB_DISABLE_PROGRESS_BARS=1
echo "\$(date -u +%T) $MODEL: venv $V"
[ -x $V/bin/python ] || uv venv -q --python 3.12 $V
VIRTUAL_ENV=$V uv pip show maxtext 2> /dev/null | grep -q '^Version: $MAXTEXT_PIN\$' || VIRTUAL_ENV=$V uv pip install -q "maxtext[tpu]==$MAXTEXT_PIN" --resolution=lowest
$V/bin/python -c "import torch" 2> /dev/null || VIRTUAL_ENV=$V UV_TORCH_BACKEND=cpu uv pip install -q torch==2.9.1
$V/bin/python -c "import importlib.metadata as m; print({p: m.version(p) for p in ('maxtext', 'jax', 'flax', 'orbax-checkpoint', 'torch', 'safetensors', 'huggingface-hub')})"
if [ ! -s $HF/done.txt ]; then
  echo "\$(date -u +%T) $MODEL: downloading $REPO@$REV to $HF"
  $V/bin/python -c "from huggingface_hub import snapshot_download as s; s('$REPO', revision='$REV', local_dir='$HF')"
  echo "$REPO@$REV \$(du -sh $HF | cut -f1)" > $HF/done.txt
fi
cat $HF/done.txt
echo "\$(date -u +%T) $MODEL: converting to $OUT"
# MaxText's own converter, as en-nl-judge's setup.sh converted the 31B: lazy loading, unscanned layers (run_sft.sh
# trains with scan_layers=false), no vision tower, bf16 (MaxText casts to its weight_dtype when it loads)
JAX_PLATFORMS=cpu $V/bin/python -u -m maxtext.checkpoint_conversion.to_maxtext model_name=$MODEL scan_layers=false \
  use_multimodal=false skip_jax_distributed_system=True hardware=cpu base_output_directory=$OUT \
  run_name=${MODEL}_conversion --hf_model_path=$HF --lazy_load_tensors=True --save_dtype=bfloat16
gsutil -q stat $OUT/0/commit_success.txt || { echo "no $OUT/0/commit_success.txt after the conversion"; exit 1; }
echo "\$(date -u +%T) $MODEL: \$(gsutil du -s $OUT | awk '{printf "%.1f GB", \$1 / 1e9}') in $OUT"
echo "$REPO@$REV converted by MaxText $MAXTEXT_PIN (run_g4lora.sh convert, mbert-jax $(cat "$CODE/COMMIT" 2> /dev/null || echo unknown)) on \$(hostname) at \$(date -u +%FT%TZ)" | gsutil -q cp - $OUT/done.txt
echo "\$(date -u +%T) $MODEL: $OUT done"
EOF
  detach "$NAME" nice -n 19 ionice -c3 bash ~/job_body_$NAME.sh
  ;;
*)
  echo "unknown stage $STAGE"
  exit 1
  ;;
esac
