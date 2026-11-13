#!/bin/bash
# The memory and step test of LoRA SFT on Gemma 4 26B-A4B-it (MoE) and Gemma 4 31B-it (Yeb, 13 Nov ~12:50: "een test met
# lora op de MoE"), on mbert-mix with the translator's SFT stack (MaxText 0.2.5, Tunix, qwix: run_sft.sh, sft_lora.py),
# one stage per call on a worker; translator_g4lora_run.sh (co1) starts the stages and waits for their EXIT lines. Every
# stage that takes longer than a few seconds runs detached from ssh, with a fresh log ~/logs/<job>.log that ends in
# "EXIT <code>". A measurement: 8 steps per variant, no arm is trained.
#   MODEL=gemma4-26b bash run_g4lora.sh convert  a worker (CPU, nice/ionice): the Hugging Face snapshot at its pinned
#                                                revision to /dev/shm, MaxText 0.2.5's to_maxtext (unscanned, text only,
#                                                bf16) straight into $CK/<dir>/, done.txt last
#                                                                            (job g4l_convert_<26b|31b>_$TAG)
#   bash run_g4lora.sh setup                     every worker: the SFT venv (setup_sft.sh) and both checkpoints there,
#                                                the tokenizers of both models at their revisions; READY
#   bash run_g4lora.sh data                      w3 (CPU, nice/ionice): g4lora.py data, to $G/data/  (job g4l_data_$TAG)
#   VARIANT=v1 [LEN=2048] [FALLBACK=0] bash run_g4lora.sh spec    anywhere: the variant's JSON (model, rank, alpha, regex,
#                                                length, extra, run name), nothing else
#   VARIANT=v1 [LEN=2048] [FALLBACK=0] bash run_g4lora.sh train   every worker at once: run_sft.sh, 8 steps over the 16
#                                                chips (run g4l_<variant>[_<len>][f], log ~/logs/<run>.log)
#   VARIANT=v1 [LEN=..] [FALLBACK=..] [LABEL=v1] bash run_g4lora.sh result   a worker: the run's logs and summary from
#                                                $G/runs/<run>/, g4lora.py result -> one line RESULT {json}
#   bash run_g4lora.sh best                      a worker: the variant to repeat at 4096 from the chain's markers
#                                                $G/stages/run_<run>.done (RESULT lines), or none
#   bash run_g4lora.sh report                    a worker: g4lora.py report, to $G/report/
# MODEL: gemma4-26b (google/gemma-4-26B-A4B-it) or gemma4-31b (google/gemma-4-31B-it). TAG names the jobs of one chain.
# The variants (rank 64, alpha 32, LR 1e-5, max_target_length 2048, 16 chips, 8 steps, as run_arms.sh's memory test):
#   v1   26B, LoRA on attention only (query, key, value, out)
#   v2   26B, attention + the routed experts (the MoE block's own weights wi_0, wi_1, wo; its router `gate` is a module of
#        its own and stays frozen)
#   v2b  26B, attention + the dense MLP beside the experts (MaxText's shared_experts): TranslateGemma's modules on the 26B
#   v3   31B, TranslateGemma's modules (run_sft.sh's regex: attention + MLP)
#   v3b  31B, the same at rank 32, alpha 16 (the chain runs it only when v3 does not fit)
# FALLBACK=1: once, after a variant ended in an error that is not out of memory: attention=dot_product, and for the 26B
# megablox=false (jax.lax.ragged_dot instead of the Pallas kernel), in case a TPU kernel does not compile on the v4.
set -euo pipefail
STAGE=${1:?a stage: convert, setup, data, spec, train, result, best, report}
S=gs://t5-dutch-english-usc2/mbert/translator/sft
G=$S/g4lora                           # the test's data, runs, markers and report
CK=gs://t5-dutch-english-usc2/mbert/translator/gemma4-ckpt   # us-central2, beside mbert-mix; nothing crosses regions
SHM=/dev/shm/translator
G_SHM=$SHM/g4lora
SFT_PY=$SHM/smoke-lora/venv/bin/python  # setup_sft.sh: MaxText 0.2.5 and Tunix
MAXTEXT_PIN=0.2.5                     # the SFT venv's: the checkpoint's tree is the one its trainer loads
RANK_DEFAULT=64
ALPHA_DEFAULT=32
LR=1e-5
STEPS=8
CODE=$(cd "$(dirname "$0")" && pwd)
W=$(hostname | sed 's/.*-w-//')
TAG=${TAG:-manual}
export PATH=$HOME/.local/bin:$PATH UV_CACHE_DIR=$SHM/uv-cache UV_PYTHON_INSTALL_DIR=$SHM/python
ATTN='self_attention/(query|key|value|out)'
TG_REGEX='decoder/layers_[0-9]+/.*(self_attention/(query|key|value|out)|mlp/(wi_0|wi_1|wo))'  # run_sft.sh's default

model_info() {  # MODEL: REPO REVISION DIR (the revisions of translate_models.yml and setup_generic.sh, 13 Nov)
  case $1 in
    gemma4-26b) echo "google/gemma-4-26B-A4B-it 4d7ae4984b7db7de8f8457170b3f1a419ee76d52 gemma-4-26b-a4b-it" ;;
    gemma4-31b) echo "google/gemma-4-31B-it 842da3794eaa0b77d5f08bae87a17459d91ff475 gemma-4-31b-it" ;;
    *) echo "MODEL is gemma4-26b or gemma4-31b, not '$1'" >&2; return 1 ;;
  esac
}

variant() {  # VARIANT LEN FALLBACK: sets MODEL RANK ALPHA REGEX EXTRA NAME
  local v=$1 len=$2 fallback=$3
  RANK=$RANK_DEFAULT
  ALPHA=$ALPHA_DEFAULT
  case $v in
    v1) MODEL=gemma4-26b; REGEX="decoder/layers_[0-9]+/$ATTN" ;;
    v2) MODEL=gemma4-26b; REGEX="decoder/layers_[0-9]+/($ATTN|mlp/moe_block/MoeBlock_0)" ;;
    v2b) MODEL=gemma4-26b; REGEX="decoder/layers_[0-9]+/($ATTN|mlp/moe_block/shared_experts/(wi_0|wi_1|wo))" ;;
    v3) MODEL=gemma4-31b; REGEX=$TG_REGEX ;;
    v3b) MODEL=gemma4-31b; REGEX=$TG_REGEX; RANK=32; ALPHA=16 ;;
    *) echo "VARIANT is v1, v2, v2b, v3 or v3b, not '$v'" >&2; return 1 ;;
  esac
  case $len in 2048 | 4096) ;; *) echo "LEN is 2048 or 4096, not '$len'" >&2; return 1 ;; esac
  EXTRA=""
  case $fallback in
    0) ;;
    1) EXTRA="attention=dot_product"; [ "$MODEL" = gemma4-26b ] && EXTRA="$EXTRA megablox=false" ;;
    *) echo "FALLBACK is 0 or 1, not '$fallback'" >&2; return 1 ;;
  esac
  NAME=g4l_$v
  [ "$len" = 2048 ] || NAME=${NAME}_$len
  [ "$fallback" = 0 ] || NAME=${NAME}f
}

spec_json() {  # the variant as JSON (g4lora.py result's spec)
  LABEL=${LABEL:-$VARIANT} V=$VARIANT RUN=$NAME M=$MODEL R=$RANK A=$ALPHA L=$LEN X=$EXTRA RE=$REGEX F=$FALLBACK \
    python3 -c 'import json, os; e = os.environ; print(json.dumps({"variant": e["LABEL"], "base": e["V"], "run": e["RUN"],
"model": e["M"], "rank": int(e["R"]), "alpha": int(e["A"]), "length": int(e["L"]), "extra": e["X"], "regex": e["RE"],
"fallback": int(e["F"])}))'
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

tpu_free() {
  [ -z "$(sudo lsof -t /dev/accel* /dev/vfio/* 2> /dev/null)" ] || { echo "w$W: the TPU is in use"; exit 1; }
}

case $STAGE in
convert)
  MODEL=${MODEL:?gemma4-26b or gemma4-31b}
  read -r REPO REV DIR < <(model_info "$MODEL")
  mkdir -p "$G_SHM" ~/logs
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
setup)
  mkdir -p "$G_SHM" ~/logs
  [ "$(loginctl show-user "$(id -un)" -p Linger --value 2> /dev/null)" = yes ] || sudo loginctl enable-linger "$(id -un)"
  [ -x "$SFT_PY" ] || { echo "w$W: no SFT venv (setup_sft.sh)"; exit 1; }
  "$SFT_PY" -c "import importlib.metadata as m; v = m.version('maxtext'); assert v == '$MAXTEXT_PIN', v" \
    || { echo "w$W: the SFT venv's MaxText is not $MAXTEXT_PIN"; exit 1; }
  [ -s ~/.cache/huggingface/token ] || { echo "w$W: no ~/.cache/huggingface/token"; exit 1; }
  for MODEL in gemma4-26b gemma4-31b; do
    read -r REPO REV DIR < <(model_info "$MODEL")
    gsutil -q stat "$CK/$DIR/done.txt" || { echo "w$W: $CK/$DIR has no done.txt (run_g4lora.sh convert)"; exit 1; }
    TOK=$G_SHM/tok-$MODEL
    [ -s "$TOK/tokenizer.json" ] && [ -s "$TOK/chat_template.jinja" ] || HF_HUB_DISABLE_PROGRESS_BARS=1 "$SFT_PY" -c \
      "from huggingface_hub import snapshot_download as s; s('$REPO', revision='$REV', local_dir='$TOK', allow_patterns=['tokenizer*', 'chat_template.jinja', 'config.json', 'generation_config.json', 'special_tokens_map.json'])"
    [ -s "$TOK/tokenizer.json" ] || { echo "w$W: no tokenizer in $TOK"; exit 1; }
  done
  free=$(df -BG --output=avail "$HOME" | tail -1 | tr -dc 0-9)
  [ "$free" -ge 3 ] || { echo "w$W: only $free GB free on the boot disk"; exit 1; }
  echo "READY w$W: boot disk $free GB free, /dev/shm $(df -h /dev/shm | tail -1 | awk '{print $4}') free"
  ;;
data)
  ! gsutil -q stat "$G/data/data.json" || { echo "$G/data has a data.json: the data stay as they are"; exit 1; }
  mkdir -p "$G_SHM" ~/logs
  IN=$G_SHM/in-$TAG
  OUT=$G_SHM/data-$TAG
  NAME=g4l_data_$TAG
  cat > ~/job_body_$NAME.sh <<EOF
set -euo pipefail
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=true HF_HUB_OFFLINE=1 JAX_PLATFORMS=cpu
mkdir -p $IN/rl $IN/r2
gsutil -m -q cp $S/data/rl-b5/RL/train.parquet $S/data/rl-b5/RL/train-meta.parquet $IN/rl/
gsutil -m -q cp $S/data/r2/R2/train.parquet $S/data/r2/R2/train-meta.parquet $IN/r2/
$SFT_PY -u $CODE/g4lora.py data --rl $IN/rl --r2 $IN/r2 --tokenizer gemma4-26b=$G_SHM/tok-gemma4-26b \
  --tokenizer gemma4-31b=$G_SHM/tok-gemma4-31b --out $OUT
gsutil -m -q cp $OUT/train.parquet $OUT/train-meta.parquet $OUT/template.jinja $OUT/data.json $G/data/
echo "DATA_DONE $G/data/"
EOF
  detach "$NAME" nice -n 19 ionice -c3 bash ~/job_body_$NAME.sh
  ;;
spec)
  VARIANT=${VARIANT:?v1, v2, v2b, v3 or v3b}
  LEN=${LEN:-2048}
  FALLBACK=${FALLBACK:-0}
  variant "$VARIANT" "$LEN" "$FALLBACK"
  spec_json
  ;;
train)
  VARIANT=${VARIANT:?v1, v2, v2b, v3 or v3b}
  LEN=${LEN:-2048}
  FALLBACK=${FALLBACK:-0}
  variant "$VARIANT" "$LEN" "$FALLBACK"
  tpu_free
  D=$G_SHM/data
  mkdir -p "$D"
  gsutil -m -q rsync -r "$G/data" "$D"
  # One job over the 16 chips as run_arms.sh's memory test: sequences of LEN, one per chip, FSDP over the 16 chips
  # (MaxText's default mesh), the LoRA parameters replicated (MaxText's apply_lora_to_model), 8 steps
  # shellcheck disable=SC2086
  MULTIHOST=1 NAME=$NAME MODEL=$MODEL DATA=$D/train.parquet TEMPLATE=$D/template.jinja ROOT=$G LORA_MODULES=$REGEX \
    STEPS=$STEPS bash "$CODE/run_sft.sh" per_device_batch_size=1 max_target_length=$LEN lora.lora_rank=$RANK \
    lora.lora_alpha=$ALPHA learning_rate=$LR $EXTRA
  ;;
result)
  VARIANT=${VARIANT:?v1, v2, v2b, v3 or v3b}
  LEN=${LEN:-2048}
  FALLBACK=${FALLBACK:-0}
  variant "$VARIANT" "$LEN" "$FALLBACK"
  RUN=$G/runs/$NAME
  X=$G_SHM/result-$NAME
  mkdir -p "$X"
  # run_sft.sh copies a worker's log after its EXIT line, and worker 0 the summary after that: up to 3 minutes
  for _ in $(seq 1 36); do
    n=$(gsutil ls "$RUN/$NAME-w*.log" 2> /dev/null | wc -l)
    [ "$n" = 4 ] && break
    sleep 5
  done
  [ "$n" = 4 ] || { echo "$RUN: $n of 4 worker logs"; exit 1; }
  gsutil -m -q cp "$RUN/$NAME-w*.log" "$X/"
  if grep -q '^EXIT 0$' "$X/$NAME-w0.log"; then
    for _ in $(seq 1 24); do gsutil -q stat "$RUN/summary.json" && break; sleep 5; done
    gsutil -q cp "$RUN/summary.json" "$X/" 2> /dev/null || true
  fi
  "$SFT_PY" -I "$CODE/g4lora.py" result --run-dir "$X" --run "$NAME" --spec "$(spec_json)"
  ;;
best)
  X=$G_SHM/best-$TAG-$$
  mkdir -p "$X"
  gsutil -m -q cp "$G/stages/run_*.done" "$X/"
  "$SFT_PY" -I "$CODE/g4lora.py" best "$X"/run_*.done
  ;;
report)
  X=$G_SHM/report-$TAG
  mkdir -p "$X/stages"
  gsutil -m -q cp "$G/stages/run_*.done" "$X/stages/"
  gsutil -q cp "$G/data/data.json" "$X/"
  gsutil -q cp "$S/rl/RL/summary.json" "$X/tg-rl.json"
  gsutil -q cp "$S/r2/R2/summary.json" "$X/tg-r2.json"
  "$SFT_PY" -I "$CODE/g4lora.py" report --results "$X"/stages/run_*.done --data "$X/data.json" \
    --tg "RL=$X/tg-rl.json" --tg "R2=$X/tg-r2.json" --head "$CODE/g4lora-report-head.md" \
    --out "$X/gemma4-moe-lora-test.md"
  gsutil -m -q cp "$X/gemma4-moe-lora-test.md" "$X/gemma4-moe-lora-test.json" "$G/report/"
  echo "REPORT_DONE $G/report/gemma4-moe-lora-test.md"
  ;;
*)
  echo "unknown stage $STAGE"
  exit 1
  ;;
esac
