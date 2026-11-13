#!/bin/bash
# Arm G4-1 of protocol-g4.md (Gemma 4 31B-it + LoRA rank 32 / alpha 16; Yeb, 14 Nov ~00:40) on a v4-32 (mbert-mix or
# mbert-corpus, us-central2-b), one stage per call on a worker; g4arm_run.sh (co1, mbert-ops) starts the stages and
# waits for their EXIT lines. Every stage that takes longer than a few seconds runs detached from ssh, with a fresh log
# ~/logs/g4a_<stage>_$TAG.log that ends in "EXIT <code>" (also copied to $G/logs/).
# The data (CPU, nice/ionice; before any TPU is free):
#   bash run_g4arm.sh fetchbt     w3: the five Gemma 4 BT datasets at their pinned revisions (g4arm.py fetch) from
#                                 Hugging Face, once, to $G/input/bt/ (a second start takes them from there)
#   bash run_g4arm.sh build       w3: g4arm.py build, to $G/data/ (set G4-1, report)
#   bash run_g4arm.sh check       w2: g4arm.py check (MaxText's own SFT steps on every example), $G/data/check.txt
#   bash run_g4arm.sh evaldata    w0: g4arm.py evaldata, to $G/data/eval/
# On the TPU:
#   bash run_g4arm.sh setup       every worker: the SFT venv, the fork's venv (@ f4d8615), Gemma 4's tokenizer, MetricX-24 XXL;
#                                 with FULL=1 also COMET-22 and (w0) the classifier's venv; READY
#   RUN=g4arm_G4-1_s8 STEPS=8 bash run_g4arm.sh train    every worker at once: run_sft.sh over the 16 chips (RUN
#                                 g4arm_G4-1 with STEPS=auto: one epoch), to $G/runs/$RUN/
#   bash run_g4arm.sh adapter     a worker: the run's adapter.npz/json and summary.json to $G/G4-1/ (rank, alpha checked)
#   MODEL=G4-1|G4-0 bash run_g4arm.sh translate    every worker: its classes of $G/data/eval/ (translate.py on the fork @ f4d8615,
#                                 natural-v1, G4-1 with the adapter merged), to $G/eval/$MODEL/translations/w<k>/<class>/
#   MODEL=G4-1|G4-0 bash run_g4arm.sh score        every worker: collect, MetricX-24 (reference and QE), COMET-22 on the
#                                 rows with a reference, to $G/eval/$MODEL/{pairs,scores}/
#   bash run_g4arm.sh texts       w0 (CPU): g4arm.py texts, to $G/naturalness/texts.parquet
#   bash run_g4arm.sh classify    w0 (its TPU): run_arms.sh classify with NAT=$G/naturalness (the classifier of 10 Nov)
#   bash run_g4arm.sh blind       w0 (CPU): armeval.py blind, G4-1 against G4-0 ($S/blind/G4-1/) and against TG
#                                 ($S/blind/G4-1-tg/)
#   bash run_g4arm.sh report      w0 (CPU): g4arm.py report, to $G/report/
# The smoke test (protocol-g4.md §6), after train with RUN=g4arm_G4-1_s8 STEPS=8:
#   MODEL=G4-1|G4-0 bash run_g4arm.sh smoke-translate   w0: the 50 smoke segments, class small, to $G/smoke/$TAG/$MODEL/
#   MODEL=G4-1|G4-0 bash run_g4arm.sh smoke-score       w0: collect and MetricX-24 of those 50
#   bash run_g4arm.sh smoke-report                      w0 (CPU): g4arm.py smoke, to $G/smoke/$TAG/
# The code is beside this script (a copy of $G/scripts/). TAG names the jobs of one chain.
set -euo pipefail
STAGE=${1:?a stage: fetchbt, build, check, evaldata, setup, train, adapter, translate, score, texts, classify, blind, report, smoke-translate, smoke-score, smoke-report}
R=gs://t5-dutch-english-usc2/mbert/translator
S=$R/sft
E=$R/eval
G=$S/g4arm
ARM=G4-1
SMOKE_RUN=g4arm_G4-1_s8
FULL_RUN=g4arm_G4-1
TRAIN_CKPT=$R/gemma4-ckpt/gemma-4-31b-it                       # run_sft.sh's gemma4-31b (MaxText 0.2.5 conversion)
DECODE_CKPT=$R/judge/gemma-4-31b-it-maxtext/0/items            # the same conversion's twin, as every Gemma 4 decode
G4_REPO=google/gemma-4-31B-it
G4_REV=842da3794eaa0b77d5f08bae87a17459d91ff475
SHM=/dev/shm/translator
G_SHM=$SHM/g4arm
SFT_PY=$SHM/smoke-lora/venv/bin/python                         # setup_sft.sh: MaxText 0.2.5, Tunix
GEN=/dev/shm/translate-generic
# protocol-g4.md §9, amendment of 14 Nov ~00:49: the fork with the fix of the shared decode-cache index (on 2249474),
# setup_generic.sh's venv-fork-<first 9 of the commit>
FORK_COMMIT=f4d8615f478b6c18de9df8f2dff1f57f775a2521
FPY=$GEN/venv-fork-${FORK_COMMIT:0:9}/bin/python
TOK_SFT=$SHM/g4lora/tok-gemma4-31b                             # run_sft.sh's tokenizer of gemma4-31b
TOK_DEC=$GEN/tokenizer-gemma4                                  # setup_generic.sh's, the same revision
MX_ROOT=$SHM/metricx
CODE=$(cd "$(dirname "$0")" && pwd)
W=$(hostname | sed 's/.*-w-//')
TAG=${TAG:-manual}
export PATH=$HOME/.local/bin:$PATH UV_CACHE_DIR=$SHM/uv-cache UV_PYTHON_INSTALL_DIR=$SHM/python
mkdir -p "$G_SHM" ~/logs

detach() {  # detach NAME COMMAND...: the command in the background, log ~/logs/NAME.log ending in EXIT <code>
  local name=$1
  shift
  [ -e ~/logs/$name.log ] && { echo "w$W: ~/logs/$name.log exists; a new start takes a new TAG"; exit 1; }
  printf '%q ' "$@" > ~/job_$name.sh
  setsid nohup bash -c "bash ~/job_$name.sh > ~/logs/$name.log 2>&1; echo EXIT \$? >> ~/logs/$name.log; gsutil -q cp ~/logs/$name.log $G/logs/$name-w$W.log" \
    > /dev/null 2>&1 < /dev/null &
  echo "started $name on w$W"
}

tpu_free() {
  [ -z "$(sudo lsof -t /dev/accel* /dev/vfio/* 2> /dev/null)" ] || { echo "w$W: the TPU is in use"; exit 1; }
}

only() {  # only WORKER: the stage runs on that worker alone
  [ "$W" = "$1" ] || { echo "$STAGE runs on w$1 only"; exit 1; }
}

# A worker of the v4-32 as a TPU of its own 4 chips (run_arms.sh, run_bt_length_v4.sh)
ONE_HOST="export TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 TPU_PROCESS_BOUNDS=1,1,1 TPU_VISIBLE_CHIPS=0,1,2,3 TPU_PROCESS_PORT=8476
export TPU_PROCESS_ADDRESSES=localhost:8476 CLOUD_TPU_TASK_ID=0 ALLOW_MULTIPLE_LIBTPU_LOAD=1
export XLA_FLAGS=--xla_dump_hlo_as_text=false TPU_LOG_DIR=$SHM/tpu_logs TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 HF_HUB_DISABLE_PROGRESS_BARS=1"
PREFLIGHT="timeout 600 $FPY -c \"import jax; assert (jax.device_count(), jax.process_count()) == (4, 1), (jax.device_count(), jax.process_count()); print('devices', jax.devices())\" || { echo 'preflight failed: not 4 chips of one process'; exit 2; }"

# translate_class SEGMENTS CLASS OUT LOG [EXTRA...]: one class through translate.py; out of memory: again with half
# the class's slots of translate_models.yml (sized on a v6e chip of 31.24 GiB; a v4 chip has 30.75), down to 4; every
# attempt its own log. G4-1 and G4-0 hold the same memory (the merged kernels replace the base ones), so both take the
# same slots.
TRANSLATE_CLASS='translate_class() {
  local seg=$1 cls=$2 out=$3 log=$4 code slots try
  shift 4
  slots=$($FPY -c "import sys; sys.path.insert(0, \"$CODE\"); import translate; print(translate.MODELS[\"gemma4-31b\"][\"slots\"][\"$cls\"])")
  try=$log
  while true; do
    echo "== $cls slots $slots $(date -u +%T)"
    $FPY -u $CODE/translate.py --segments $seg --out $out --ckpt $DECODE_CKPT --tokenizer $TOK_DEC --model gemma4-31b \
      --prompt natural-v1 --length-class $cls --source-lang en --target-lang nl --work-dir $G_SHM/work --slots $slots \
      "$@" > $try 2>&1 && code=0 || code=$?
    echo "EXIT $code" >> $try
    gsutil -q cp $try $G/logs/$(basename $try .log)-w$W.log || true
    grep -E "^(LORA|SHARD|DONE|engine)" $try | tail -n 4 | cut -c1-300
    [ $code = 0 ] && return 0
    if grep -q -E "RESOURCE_EXHAUSTED|Ran out of memory|[Oo]ut of memory" $try && [ $slots -gt 4 ]; then
      slots=$((slots / 2)); try=${log%.log}-s$slots.log
      echo "$cls: out of memory, again with $slots slots"
      continue
    fi
    tail -n 30 $try
    return $code
  done
}'
VARS="FPY=$FPY CODE=$CODE DECODE_CKPT=$DECODE_CKPT TOK_DEC=$TOK_DEC G_SHM=$G_SHM G=$G W=$W"

adapter_local() {  # adapter_local RUN_DIR LOCAL: the adapter of a run (or $G/G4-1) copied for translate.py --lora
  mkdir -p "$2"
  gsutil -q cp "$1/adapter.npz" "$1/adapter.json" "$2/"
}

case $STAGE in
fetchbt)
  only 3
  B=$G_SHM/bt-$TAG
  cat > ~/job_body_g4a_fetchbt_$TAG.sh <<EOF
set -euo pipefail
export HF_HUB_DISABLE_PROGRESS_BARS=1 HF_XET_CACHE=$SHM/xet-cache
if gsutil -q stat $G/input/bt/revisions.json; then
  echo "the BT input is in $G/input/bt/ before"
else
  [ -s ~/.cache/huggingface/token ] || { echo "w$W: no ~/.cache/huggingface/token"; exit 1; }
  $SFT_PY -u $CODE/g4arm.py fetch --out $B
  rm -rf $B/*/.cache
  gsutil -m -q rsync -r $B $G/input/bt
fi
echo "FETCHED: \$(gsutil cat $G/input/bt/revisions.json | tr -d '\n ' | cut -c1-400)"
EOF
  detach "g4a_fetchbt_$TAG" nice -n 19 ionice -c3 bash ~/job_body_g4a_fetchbt_$TAG.sh
  ;;
build)
  only 3
  ! gsutil -q stat "$G/data/$ARM/manifest.json" || { echo "$G/data/$ARM has a manifest.json: the set stays as it is"; exit 1; }
  [ -s "$TOK_SFT/tokenizer.json" ] || { echo "w$W: no tokenizer in $TOK_SFT (setup)"; exit 1; }
  IN=$G_SHM/in-$TAG
  OUT=$G_SHM/data-$TAG
  # BUILD_EXTRA: the build's options of an amendment (e.g. --fineweb-max-crawl 2023, --replay-share 0.1 --replay ...)
  cat > ~/job_body_g4a_build_$TAG.sh <<EOF
set -euo pipefail
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=true HF_HUB_OFFLINE=1 JAX_PLATFORMS=cpu
mkdir -p $IN/bt $IN/rl $IN/r2
gsutil -m -q rsync -r $G/input/bt $IN/bt
gsutil -m -q cp $S/data/rl-b5/RL/train.parquet $S/data/rl-b5/RL/train-meta.parquet $IN/rl/
gsutil -m -q cp $S/data/r2/R2/train.parquet $S/data/r2/R2/train-meta.parquet $IN/r2/
gsutil -q cp $E/data/evalsets.parquet $S/blind/fixed-2026-11-10/blind100.jsonl $R/mainset/flan-eval/eval-rows.parquet $IN/
$SFT_PY -u $CODE/g4arm.py build --bt $IN/bt --books $IN/rl --match-set $IN/r2 --evalsets $IN/evalsets.parquet \
  --blind $IN/blind100.jsonl --flan-eval $IN/eval-rows.parquet --tokenizer $TOK_SFT --out $OUT ${BUILD_EXTRA:-}
gsutil -m -q cp $OUT/$ARM/* $G/data/$ARM/
gsutil -m -q cp $OUT/report.json $OUT/report.md $G/data/
echo "BUILT: $G/data/$ARM/"
EOF
  detach "g4a_build_$TAG" nice -n 19 ionice -c3 bash ~/job_body_g4a_build_$TAG.sh
  ;;
check)
  only 2
  [ -s "$TOK_SFT/tokenizer.json" ] || { echo "w$W: no tokenizer in $TOK_SFT (setup)"; exit 1; }
  D=$G_SHM/check-$TAG
  cat > ~/job_body_g4a_check_$TAG.sh <<EOF
set -euo pipefail
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=true HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 JAX_PLATFORMS=cpu
mkdir -p $D
gsutil -m -q cp $G/data/$ARM/train.parquet $G/data/$ARM/train-meta.parquet $G/data/$ARM/template.jinja $D/
$SFT_PY -u $CODE/g4arm.py check --data $D --tokenizer $TOK_SFT | tee $D/check.txt
gsutil -q cp $D/check.txt $G/data/check.txt
EOF
  detach "g4a_check_$TAG" nice -n 19 ionice -c3 bash ~/job_body_g4a_check_$TAG.sh
  ;;
evaldata)
  only 0
  ! gsutil -q stat "$G/data/eval/manifest.json" || { echo "$G/data/eval has a manifest.json: it stays as it is"; exit 1; }
  [ -s "$TOK_SFT/tokenizer.json" ] || { echo "w$W: no tokenizer in $TOK_SFT (setup)"; exit 1; }
  IN=$G_SHM/evalin-$TAG
  cat > ~/job_body_g4a_evaldata_$TAG.sh <<EOF
set -euo pipefail
export TOKENIZERS_PARALLELISM=true HF_HUB_OFFLINE=1 JAX_PLATFORMS=cpu
mkdir -p $IN
gsutil -q cp $E/data/evalsets.parquet $IN/four.parquet
gsutil -q cp $S/arms/data/x/data/evalsets.parquet $IN/x.parquet
$SFT_PY -u $CODE/g4arm.py evaldata --four $IN/four.parquet --x $IN/x.parquet --tokenizer $TOK_SFT --out $IN/eval
gsutil -m -q cp $IN/eval/* $G/data/eval/
echo "EVALDATA_DONE $G/data/eval/"
EOF
  detach "g4a_evaldata_$TAG" nice -n 19 ionice -c3 bash ~/job_body_g4a_evaldata_$TAG.sh
  ;;
setup)
  # The data stages need only the SFT venv and the tokenizer (SETUP=data); the TPU stages everything
  [ "$(loginctl show-user "$(id -un)" -p Linger --value 2> /dev/null)" = yes ] || sudo loginctl enable-linger "$(id -un)"
  [ -s ~/.cache/huggingface/token ] || { echo "w$W: no ~/.cache/huggingface/token"; exit 1; }
  free=$(df -BG --output=avail "$HOME" | tail -1 | tr -dc 0-9)
  [ "$free" -ge 3 ] || { echo "w$W: only $free GB free on the boot disk"; exit 1; }
  cat > ~/job_body_g4a_setup_$TAG.sh <<EOF
set -euo pipefail
export PATH=$PATH UV_CACHE_DIR=$UV_CACHE_DIR UV_PYTHON_INSTALL_DIR=$UV_PYTHON_INSTALL_DIR HF_HUB_DISABLE_PROGRESS_BARS=1
[ -x $SFT_PY ] || bash $CODE/setup_sft.sh
$SFT_PY -c "import importlib.metadata as m; v = m.version('maxtext'); assert v == '0.2.5', v"
[ -s $TOK_SFT/tokenizer.json ] && [ -s $TOK_SFT/chat_template.jinja ] || $SFT_PY -c "from huggingface_hub import snapshot_download as s; s('$G4_REPO', revision='$G4_REV', local_dir='$TOK_SFT', allow_patterns=['tokenizer*', 'chat_template.jinja', 'config.json', 'generation_config.json', 'special_tokens_map.json'])"
[ -s $TOK_SFT/tokenizer.json ] || { echo "no tokenizer in $TOK_SFT"; exit 1; }
if [ "${SETUP:-tpu}" = data ]; then
  echo "READY w$W (data): SFT venv, tokenizer"
  exit 0
fi
gsutil -q stat $TRAIN_CKPT/done.txt || { echo "$TRAIN_CKPT has no done.txt"; exit 1; }
gsutil -q stat ${DECODE_CKPT%/0/items}/0/commit_success.txt || { echo "no ${DECODE_CKPT%/0/items}/0/commit_success.txt"; exit 1; }
FORK_COMMIT=$FORK_COMMIT bash $CODE/setup_generic.sh venv
[ -s $TOK_DEC/tokenizer.json ] || { echo "no $TOK_DEC/tokenizer.json"; exit 1; }
cmp -s $TOK_DEC/tokenizer.json $TOK_SFT/tokenizer.json || { echo "the two Gemma 4 tokenizers differ"; exit 1; }
if [ ! -d $MX_ROOT/metricx-24-hybrid-xxl-v2p6-bfloat16 ]; then
  HF_HUB_OFFLINE=0 $FPY -c "import sys; sys.path.insert(0, '$CODE'); import metricx_qe as m; print('MXDL w$W', m.download('xxl', '$MX_ROOT'))"
fi
if [ "${FULL:-0}" = 1 ]; then
  bash $CODE/setup_comet.sh
  if [ "$W" = 0 ]; then
    [ -x \$HOME/mbert-dumb/.venv/bin/python ] || { echo "w0: no DUMB venv (the classifier)"; exit 1; }
    [ "\$(git -C \$HOME/mbert-dumbx rev-parse HEAD 2> /dev/null)" = ae114789bf2eae153e4e3bea579299a617dcd417 ] || { echo "w0: \$HOME/mbert-dumbx is not at ae11478"; exit 1; }
  fi
fi
echo "READY w$W: /dev/shm \$(df -h /dev/shm | tail -1 | awk '{print \$4}') free, boot disk $free GB free"
EOF
  detach "g4a_setup_$TAG" bash ~/job_body_g4a_setup_$TAG.sh
  ;;
train)
  RUN=${RUN:?$SMOKE_RUN (with STEPS=8) or $FULL_RUN}
  STEPS=${STEPS:-auto}
  case "$RUN/$STEPS" in "$SMOKE_RUN/8"|"$FULL_RUN/auto") ;; *) echo "RUN/STEPS is $SMOKE_RUN/8 or $FULL_RUN/auto, not $RUN/$STEPS"; exit 1 ;; esac
  tpu_free
  gsutil -q stat "$G/data/check.txt" || { echo "no $G/data/check.txt: the set is not checked"; exit 1; }
  D=$G_SHM/data
  mkdir -p "$D"
  gsutil -m -q cp "$G/data/$ARM/train.parquet" "$G/data/$ARM/template.jinja" "$G/data/$ARM/manifest.json" "$D/"
  # protocol-g4.md §3: one job over the 16 chips, sequences of 2,048, one per chip, LoRA rank 32 / alpha 16 on
  # run_sft.sh's modules (attention + MLP), LR 1e-5 cosine with 10% warmup, one epoch (STEPS=auto) or 8 steps
  MULTIHOST=1 NAME=$RUN MODEL=gemma4-31b DATA=$D/train.parquet TEMPLATE=$D/template.jinja ROOT=$G STEPS=$STEPS \
    bash "$CODE/run_sft.sh" per_device_batch_size=1 max_target_length=2048 lora.lora_rank=32 lora.lora_alpha=16.0 \
    learning_rate=1e-5
  ;;
adapter)
  RUN_DIR=$G/runs/$FULL_RUN
  for _ in $(seq 1 36); do
    gsutil -q stat "$RUN_DIR/adapter.npz" && gsutil -q stat "$RUN_DIR/adapter.json" && gsutil -q stat "$RUN_DIR/summary.json" && break
    sleep 5
  done
  gsutil -q cp "$RUN_DIR/adapter.json" "$G_SHM/adapter-$TAG.json"
  python3 -c "import json, sys; a = json.load(open(sys.argv[1])); assert (a['rank'], a['alpha'], a['scale']) == (32, 16.0, 0.5), a" \
    "$G_SHM/adapter-$TAG.json" || { echo "the adapter is not rank 32, alpha 16"; exit 1; }
  gsutil -q cp "$RUN_DIR/adapter.npz" "$RUN_DIR/adapter.json" "$RUN_DIR/summary.json" "$G/$ARM/"
  echo "ADAPTER $ARM: $G/$ARM/ $(tr -d '\n ' < "$G_SHM/adapter-$TAG.json" | cut -c1-300)"
  ;;
translate|smoke-translate)
  MODEL=${MODEL:?G4-1 or G4-0}
  case $MODEL in G4-1|G4-0) ;; *) echo "MODEL is G4-1 or G4-0, not $MODEL"; exit 1 ;; esac
  tpu_free
  if [ "$STAGE" = smoke-translate ]; then
    only 0
    OUT=$G/smoke/$TAG/$MODEL/translations
    ADAPTER_DIR=$G/runs/$SMOKE_RUN
    SEG=segments-smoke.parquet
    CLASSES=small
    NAME=g4a_smoke_translate_${MODEL}_$TAG
  else
    OUT=$G/eval/$MODEL/translations/w$W
    ADAPTER_DIR=$G/$ARM
    SEG=segments-w$W.parquet
    CLASSES=$(gsutil cat "$G/data/eval/classes-w$W.txt")
    NAME=g4a_translate_${MODEL}_$TAG
  fi
  [ -n "$CLASSES" ] || { echo "w$W: no classes"; exit 1; }
  EXTRA=
  if [ "$MODEL" = G4-1 ]; then
    adapter_local "$ADAPTER_DIR" "$G_SHM/adapter-$NAME"
    EXTRA="--lora $G_SHM/adapter-$NAME"
  fi
  {
    echo "set -uo pipefail"
    echo "$VARS"
    echo "$ONE_HOST"
    echo "export JAX_COMPILATION_CACHE_DIR=$G_SHM/jax_cache"
    echo "mkdir -p $G_SHM/seg-$TAG $G_SHM/work"
    echo "gsutil -q cp $G/data/eval/$SEG $G_SHM/seg-$TAG/$SEG || exit 1"
    echo "$PREFLIGHT"
    echo "$TRANSLATE_CLASS"
    for cls in $CLASSES; do
      echo "translate_class $G_SHM/seg-$TAG/$SEG $cls $OUT/$cls ~/logs/${NAME}_$cls.log $EXTRA || exit 1"
    done
    echo "echo \"TRANSLATED w$W $MODEL: $CLASSES\""
  } > ~/job_body_$NAME.sh
  detach "$NAME" bash ~/job_body_$NAME.sh
  ;;
score|smoke-score)
  MODEL=${MODEL:?G4-1 or G4-0}
  case $MODEL in G4-1|G4-0) ;; *) echo "MODEL is G4-1 or G4-0, not $MODEL"; exit 1 ;; esac
  tpu_free
  if [ "$STAGE" = smoke-score ]; then
    only 0
    O=$G/smoke/$TAG/$MODEL
    SRC=$O/translations
    SEG=segments-smoke.parquet
    COMET=0
    NAME=g4a_smoke_score_${MODEL}_$TAG
  else
    O=$G/eval/$MODEL
    SRC=$O/translations/w$W
    SEG=segments-w$W.parquet
    COMET=1
    NAME=g4a_score_${MODEL}_$TAG
  fi
  L=$G_SHM/score-$NAME
  cat > ~/job_body_$NAME.sh <<EOF
set -euo pipefail
$ONE_HOST
export JAX_COMPILATION_CACHE_DIR=$G_SHM/jax_cache_metricx
mkdir -p $L/data $L/translations $L/out
gsutil -q cp $G/data/eval/evalsets.parquet $L/data/
# evalsets.py collect reads this worker's segments as segments-w<k>.parquet
gsutil -q cp $G/data/eval/$SEG $L/data/segments-w$W.parquet
gsutil -m -q rsync -r $SRC $L/translations
$FPY -I $CODE/evalsets.py collect --data $L/data --worker $W --shards $L/translations --out $L/out/pairs-w$W.parquet
gsutil -q cp $L/out/pairs-w$W.parquet $O/pairs/pairs-w$W.parquet
if gsutil -q stat $O/scores/metricx-w$W.parquet; then
  echo "w$W: MetricX scores are in $O/scores/ before"
else
  timeout 600 $FPY -c "import jax; assert (jax.device_count(), jax.process_count()) == (4, 1), (jax.device_count(), jax.process_count())"
  $FPY -I -u $CODE/metricx_qe.py score --pairs $L/out/pairs-w$W.parquet --pair source:hypothesis:reference --pair source:hypothesis --model xxl --root $MX_ROOT --empty-candidates --out $L/out/metricx-w$W.parquet
  gsutil -q cp $L/out/metricx-w$W.parquet $O/scores/metricx-w$W.parquet
fi
if [ $COMET = 1 ]; then
  n=\$($FPY -I -c "import pyarrow.parquet as pq, pyarrow.compute as pc, sys; t = pq.read_table(sys.argv[1]); t = t.filter(pc.is_valid(t['reference'])); pq.write_table(t, sys.argv[2]); print(t.num_rows)" $L/out/pairs-w$W.parquet $L/out/pairs-ref-w$W.parquet)
  if [ "\$n" = 0 ]; then
    echo "w$W: no row with a reference, no COMET-22"
  elif gsutil -q stat $O/scores/comet-w$W.parquet; then
    echo "w$W: COMET scores are in $O/scores/ before"
  else
    $SHM/comet-venv/bin/python -I -u $CODE/comet_score.py score --root $SHM/comet --pairs $L/out/pairs-ref-w$W.parquet --out $L/out/comet-w$W.parquet
    gsutil -q cp $L/out/comet-w$W.parquet $O/scores/comet-w$W.parquet
  fi
fi
echo "SCORED w$W $MODEL: $O/scores/"
EOF
  detach "$NAME" bash ~/job_body_$NAME.sh
  ;;
texts)
  only 0
  T=$G_SHM/texts-$TAG
  cat > ~/job_body_g4a_texts_$TAG.sh <<EOF
set -euo pipefail
mkdir -p $T/G4-1 $T/G4-0
gsutil -m -q cp '$G/eval/G4-1/pairs/pairs-w*.parquet' $T/G4-1/
gsutil -m -q cp '$G/eval/G4-0/pairs/pairs-w*.parquet' $T/G4-0/
gsutil -q cp $G/data/eval/evalsets.parquet $S/re/classify/g4-base-texts.parquet $T/
$FPY -I -u $CODE/g4arm.py texts --evalsets $T/evalsets.parquet --pairs 'G4-1=$T/G4-1/pairs-w*.parquet' \
  --pairs 'G4-0=$T/G4-0/pairs-w*.parquet' --control $T/g4-base-texts.parquet --out $T/texts.parquet
gsutil -q cp $T/texts.parquet $G/naturalness/texts.parquet
echo "TEXTS: $G/naturalness/texts.parquet"
EOF
  detach "g4a_texts_$TAG" bash ~/job_body_g4a_texts_$TAG.sh
  ;;
classify)
  only 0
  # run_arms.sh's stage, unchanged: the classifier of 10 Nov ($S/arms/naturalness/classifier/) on $NAT/texts.parquet,
  # to $NAT/predictions.parquet; its job is arms_classify_$TAG
  NAT=$G/naturalness TAG=$TAG bash "$CODE/run_arms.sh" classify
  ;;
blind)
  only 0
  for d in "$S/blind/$ARM" "$S/blind/$ARM-tg"; do
    ! gsutil -q stat "$d/key.jsonl" || { echo "$d has a key.jsonl: the package stays as it is"; exit 1; }
  done
  X=$G_SHM/blind-$TAG
  cat > ~/job_body_g4a_blind_$TAG.sh <<EOF
set -euo pipefail
mkdir -p $X/G4-1 $X/G4-0 $X/tg $X/out
gsutil -q cp $S/blind/fixed-2026-11-10/blind100.jsonl $X/
gsutil -m -q cp '$G/eval/G4-1/pairs/pairs-w*.parquet' $X/G4-1/
gsutil -m -q cp '$G/eval/G4-0/pairs/pairs-w*.parquet' $X/G4-0/
gsutil -m -q cp '$S/arms/eval/base-x/pairs/pairs-w*.parquet' $X/tg/
# G4-1 against its starting point G4-0 (rules (a2), (c)); against TranslateGemma 27B (reported); seeds
# en-nl-sft/blind/ab/<package>/<blind_id> (armeval.base_is_a), key apart
$FPY -I -u $CODE/armeval.py blind --blind $X/blind100.jsonl --base '$X/G4-0/pairs-w*.parquet' --arm 'G4-1=$X/G4-1/pairs-w*.parquet' --out $X/out
$FPY -I -u $CODE/armeval.py blind --blind $X/blind100.jsonl --base '$X/tg/pairs-w*.parquet' --arm 'G4-1-tg=$X/G4-1/pairs-w*.parquet' --out $X/out
gsutil -m -q cp $X/out/G4-1/* $S/blind/G4-1/
gsutil -m -q cp $X/out/G4-1-tg/* $S/blind/G4-1-tg/
echo "BLIND: $S/blind/G4-1/ $S/blind/G4-1-tg/"
EOF
  detach "g4a_blind_$TAG" bash ~/job_body_g4a_blind_$TAG.sh
  ;;
report)
  only 0
  X=$G_SHM/report-$TAG
  cat > ~/job_body_g4a_report_$TAG.sh <<EOF
set -euo pipefail
mkdir -p $X/G4-1 $X/G4-0 $X/logs
gsutil -m -q rsync -r -x 'translations/.*' $G/eval/G4-1 $X/G4-1
gsutil -m -q rsync -r -x 'translations/.*' $G/eval/G4-0 $X/G4-0
gsutil -q cp $G/data/eval/evalsets.parquet $E/baseline-tg27b/scores.parquet $G/naturalness/predictions.parquet $X/
gsutil -q cp $S/arms/naturalness/classifier/results.json $X/classifier-results.json
gsutil -q cp $G/$ARM/summary.json $X/train-summary.json
gsutil -m -q cp '$G/logs/g4a_translate_G4-1_*' $X/logs/
lora=""
for f in $X/logs/*.log; do
  line=\$(grep -m1 '^LORA ' \$f | cut -c6- || true)
  [ -n "\$line" ] && { echo "\$line" > \$f.lora.json; lora="\$lora --lora-line \$f.lora.json"; }
done
TALLY=""
if gsutil -q stat $G/report/tally.json; then gsutil -q cp $G/report/tally.json $X/; TALLY="--tally $X/tally.json"; fi
$FPY -I -u $CODE/g4arm.py report --evalsets $X/evalsets.parquet --arm $X/G4-1 --base $X/G4-0 --tg-scores $X/scores.parquet \
  --predictions $X/predictions.parquet --classifier-results $X/classifier-results.json --train-summary $X/train-summary.json \
  \$lora \$TALLY --out $X/report.md
gsutil -m -q cp $X/report.md $X/report.json $G/report/
echo "REPORT_DONE $G/report/report.md"
EOF
  detach "g4a_report_$TAG" bash ~/job_body_g4a_report_$TAG.sh
  ;;
smoke-report)
  only 0
  X=$G_SHM/smoke-$TAG
  O=$G/smoke/$TAG
  cat > ~/job_body_g4a_smoke_report_$TAG.sh <<EOF
set -euo pipefail
mkdir -p $X
gsutil -q cp $G/runs/$SMOKE_RUN/summary.json $G/runs/$SMOKE_RUN/adapter.json $X/
gsutil -q cp $O/G4-1/pairs/pairs-w0.parquet $X/arm-pairs.parquet
gsutil -q cp $O/G4-1/scores/metricx-w0.parquet $X/arm-metricx.parquet
gsutil -q cp $O/G4-0/pairs/pairs-w0.parquet $X/base-pairs.parquet
gsutil -q cp $O/G4-0/scores/metricx-w0.parquet $X/base-metricx.parquet
gsutil -q cp $G/logs/g4a_smoke_translate_G4-1_${TAG}_small-w0.log $X/translate-arm.log
CK=""
gsutil -q stat $G/runs/$SMOKE_RUN/maxtext/$SMOKE_RUN/checkpoints/8/commit_success.txt && CK=--checkpoint-ok
$FPY -I -u $CODE/g4arm.py smoke --summary $X/summary.json --adapter $X/adapter.json --translate-log $X/translate-arm.log \
  --arm-pairs $X/arm-pairs.parquet --arm-metricx $X/arm-metricx.parquet --base-pairs $X/base-pairs.parquet \
  --base-metricx $X/base-metricx.parquet \$CK --out $X/g4arm-smoke.md && code=0 || code=\$?
gsutil -m -q cp $X/g4arm-smoke.md $X/g4arm-smoke.json $O/
exit \$code
EOF
  detach "g4a_smoke_report_$TAG" bash ~/job_body_g4a_smoke_report_$TAG.sh
  ;;
*)
  echo "unknown stage $STAGE"
  exit 2
  ;;
esac
