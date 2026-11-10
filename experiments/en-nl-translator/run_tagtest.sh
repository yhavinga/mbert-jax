#!/bin/bash
# The fine-tune's training sets and the tag test on TranslateGemma 12B (protocol.md), one stage per call on a worker of
# mbert-mix; translator_sft_tagtest.sh (co1) starts the stages and waits for their EXIT lines. Every stage that takes
# longer than a few seconds runs detached from ssh, with a fresh log ~/logs/<job>.log that ends in "EXIT <code>".
#   bash run_tagtest.sh setup                  every worker: linger, the SFT venv (setup_sft.sh), both tokenizers
#   bash run_tagtest.sh build                  w3 (CPU): sft_build.py for the 27B arms and the tag test, tagtest.py
#                                              segments, all to $S/data/ and $S/tagtest/segments/   (job sft_build_$TAG)
#   SET=arms/B bash run_tagtest.sh check       a worker (CPU): sft_build.py check of a set in the SFT venv
#                                              (job sft_check_<set>_$TAG)
#   ARM=token bash run_tagtest.sh train        every worker at once (one job over the 16 chips): run_sft.sh, 12B,
#                                              one epoch of tagtest-$ARM (run tag12b_${ARM}_mh, log
#                                              ~/logs/tag12b_${ARM}_mh.log). On one worker 12B with a sequence of
#                                              2,048 per chip needs 38.5 GB of HLO temporaries (10 Nov), over 16
#                                              chips the smoke test's 12B took 9.65 GiB: the arms' layout on 27B
#   VARIANT=token_tag bash run_tagtest.sh translate   a worker (its TPU): run_translate.sh, 12B, the 600 segments
#                                              (log ~/logs/tagtr_${VARIANT}_$TAG.log)
#   bash run_tagtest.sh score                  a worker (its TPU): tagtest.py collect, MetricX-24-QE, report, to
#                                              $S/tagtest/report/                                    (job sft_score_$TAG)
# The code is beside this script (a copy of $S/scripts/). TAG names the jobs of one start of the chain.
set -euo pipefail
STAGE=${1:?setup, build, check, train, translate or score}
S=gs://t5-dutch-english-usc2/mbert/translator/sft
M=gs://t5-dutch-english-usc2/mbert/translator/mainset
E=gs://t5-dutch-english-usc2/mbert/translator/eval
T=$S/tagtest
SHM=/dev/shm/translator
W_DIR=$SHM/sft
PY=$SHM/venv/bin/python               # the translator's venv: pyarrow, transformers, sacrebleu
SFT_PY=$SHM/smoke-lora/venv/bin/python  # setup_sft.sh: MaxText 0.2.5 and Tunix
CODE=$(cd "$(dirname "$0")" && pwd)
W=$(hostname | sed 's/.*-w-//')
TAG=${TAG:-manual}
BT_TAGTEST=${BT_TAGTEST:-20000}
mkdir -p "$W_DIR" ~/logs

detach() {  # detach NAME COMMAND...: the command in the background, log ~/logs/NAME.log ending in EXIT <code>
  local name=$1
  shift
  [ -e ~/logs/$name.log ] && { echo "w$W: ~/logs/$name.log exists; a new start takes a new TAG"; exit 1; }
  printf '%q ' "$@" > ~/job_$name.sh
  setsid nohup bash -c "bash ~/job_$name.sh > ~/logs/$name.log 2>&1; echo EXIT \$? >> ~/logs/$name.log" \
    > /dev/null 2>&1 < /dev/null &
  echo "started $name on w$W"
}

tpu_free() {
  [ -z "$(sudo lsof -t /dev/accel* /dev/vfio/* 2> /dev/null)" ] || { echo "w$W: the TPU is in use"; exit 1; }
}

case $STAGE in
setup)
  [ "$(loginctl show-user "$(id -un)" -p Linger --value 2> /dev/null)" = yes ] || sudo loginctl enable-linger "$(id -un)"
  [ -x "$PY" ] || { echo "w$W: no $PY (the baseline's setup.sh)"; exit 1; }
  "$PY" -c "import sacrebleu, pyarrow, transformers, yaml" || { echo "w$W: $PY lacks a package"; exit 1; }
  [ -d $SHM/metricx ] || echo "w$W: no $SHM/metricx (metricx_qe.py fetches MetricX-24 XXL from HF, ~25 GB)"
  for model in 12b 27b; do
    [ -s /dev/shm/flan_nl/tokenizer-$model/tokenizer.json ] || MODEL=$model bash "$CODE/setup_flan.sh"
  done
  bash "$CODE/setup_sft.sh"
  free=$(df -BG --output=avail "$HOME" | tail -1 | tr -dc 0-9)
  [ "$free" -ge 3 ] || { echo "w$W: only $free GB free on the boot disk"; exit 1; }
  echo "READY w$W: boot disk $free GB free, /dev/shm $(df -h /dev/shm | tail -1 | awk '{print $4}') free"
  ;;
build)
  IN=$W_DIR/in
  mkdir -p "$IN/run" "$IN/long" "$IN/replay" "$IN/flan-eval" "$IN/eval"
  cat > ~/job_body_build_$TAG.sh <<EOF
set -euo pipefail
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=true
[ -s $IN/run/pairs.parquet ] || gsutil -q cp $M/run/pairs/pairs.parquet $IN/run/
[ -s $IN/run/passages.parquet ] || gsutil -q cp $M/run/select/passages.parquet $IN/run/
[ -s $IN/run/context-turns.parquet ] || gsutil -q cp $M/run/select/context-turns.parquet $IN/run/
[ -s $IN/long/pairs.parquet ] || gsutil -q cp $M/run-long/pairs/pairs.parquet $IN/long/
[ -s $IN/long/passages.parquet ] || gsutil -q cp $M/run-long/select/passages.parquet $IN/long/
[ -s $IN/replay/pairs.parquet ] || gsutil -q cp $M/replay/pairs.parquet $IN/replay/
[ -s $IN/flan-eval/eval-rows.parquet ] || gsutil -q cp $M/flan-eval/eval-rows.parquet $IN/flan-eval/
[ -s $IN/eval/evalsets.parquet ] || gsutil -q cp $E/data/evalsets.parquet $IN/eval/
common="--run $IN/run --long $IN/long --replay $IN/replay/pairs.parquet --flan-eval $IN/flan-eval/eval-rows.parquet --evalsets $IN/eval/evalsets.parquet --tokenizer /dev/shm/flan_nl/tokenizer-27b"
O=$W_DIR/out
[ ! -e \$O ] || { echo "\$O exists"; exit 1; }
$PY -I -u $CODE/sft_build.py build \$common --bt auto --sets B=none N-token=token+replay N-instruction=instruction+replay --out \$O/arms
$PY -I -u $CODE/sft_build.py build \$common --bt $BT_TAGTEST --sets tagtest-token=token+replay tagtest-instruction=instruction+replay --out \$O/tagtest
$PY -I -u $CODE/tagtest.py segments --flan-eval $IN/flan-eval/eval-rows.parquet --evalsets $IN/eval/evalsets.parquet --tokenizer /dev/shm/flan_nl/tokenizer-12b --out \$O/segments
gsutil -m -q rsync -r \$O/arms $S/data/arms
gsutil -m -q rsync -r \$O/tagtest $S/data/tagtest
gsutil -m -q rsync -r \$O/segments $T/segments
echo "BUILT: $S/data/arms, $S/data/tagtest, $T/segments"
EOF
  detach "sft_build_$TAG" bash ~/job_body_build_$TAG.sh
  ;;
check)
  SET=${SET:?arms/<set> or tagtest/<set>}
  D=$W_DIR/check/$SET
  name="sft_check_$(echo "$SET" | tr '/' '_')_$TAG"
  cat > ~/job_body_$name.sh <<EOF
set -euo pipefail
mkdir -p $D
gsutil -m -q rsync -r $S/data/$SET $D
JAX_PLATFORMS=cpu TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 $SFT_PY -u $CODE/sft_build.py check --data $D --tokenizer /dev/shm/flan_nl/tokenizer-27b
EOF
  detach "$name" bash ~/job_body_$name.sh
  ;;
train)
  ARM=${ARM:?token or instruction}
  tpu_free
  D=$W_DIR/data/tagtest-$ARM
  mkdir -p "$D"
  gsutil -m -q rsync -r "$S/data/tagtest/tagtest-$ARM" "$D"
  # One epoch at the smoke test's LoRA settings (run_sft.sh); sequences of 2048 (design §7 point 7), one per chip
  MULTIHOST=1 NAME=tag12b_${ARM}_mh MODEL=12b DATA=$D/train.parquet TEMPLATE=$D/template.jinja ROOT=$T \
    bash "$CODE/run_sft.sh" per_device_batch_size=1 max_target_length=2048
  ;;
translate)
  VARIANT=${VARIANT:?base, token_tag, token_plain, instruction_tag or instruction_plain}
  tpu_free
  case $VARIANT in
    base) ARM=; PROMPT= ;;
    token_tag) ARM=token; PROMPT=chat-v1-tag-token ;;
    token_plain) ARM=token; PROMPT= ;;
    instruction_tag) ARM=instruction; PROMPT=chat-v1-tag-instruction ;;
    instruction_plain) ARM=instruction; PROMPT= ;;
    *) echo "unknown VARIANT $VARIANT"; exit 1 ;;
  esac
  EXTRA=${PROMPT:+--prompt $PROMPT}
  if [ -n "$ARM" ]; then
    A=$W_DIR/adapters/tag12b_${ARM}_mh
    mkdir -p "$A"
    # run_sft.sh copies the adapter to the bucket after its log's EXIT line: give that copy up to 2 minutes
    for _ in $(seq 1 24); do
      gsutil -q stat "$T/runs/tag12b_${ARM}_mh/adapter.npz" && gsutil -q stat "$T/runs/tag12b_${ARM}_mh/adapter.json" && break
      sleep 5
    done
    gsutil -q cp "$T/runs/tag12b_${ARM}_mh/adapter.npz" "$T/runs/tag12b_${ARM}_mh/adapter.json" "$A/"
    EXTRA="$EXTRA --lora $A"
  fi
  MODEL=12b SOURCE=en TARGET=nl SEGMENTS="$T/segments/segments.parquet" OUT="$T/translations/$VARIANT" \
    NAME="tagtr_${VARIANT}_$TAG" LOGS="$T/logs" CLASSES="tiny small medium long xlong" EXTRA="$EXTRA" \
    bash "$CODE/run_translate.sh"
  ;;
score)
  tpu_free
  R=$W_DIR/report
  cat > ~/job_body_score_$TAG.sh <<EOF
set -euo pipefail
mkdir -p $R/shards
gsutil -m -q rsync -r $T/translations $R/shards
gsutil -q cp $T/segments/items.parquet $R/
$PY -I -u $CODE/tagtest.py collect --items $R/items.parquet --shards $R/shards --out $R/pairs.parquet
(
  export TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 TPU_PROCESS_BOUNDS=1,1,1 TPU_VISIBLE_CHIPS=0,1,2,3 TPU_PROCESS_PORT=8476
  export TPU_PROCESS_ADDRESSES=localhost:8476 CLOUD_TPU_TASK_ID=0 ALLOW_MULTIPLE_LIBTPU_LOAD=1
  export JAX_COMPILATION_CACHE_DIR=$SHM/jax_cache_metricx XLA_FLAGS=--xla_dump_hlo_as_text=false TPU_LOG_DIR=$SHM/tpu_logs
  $PY -I -u $CODE/metricx_qe.py score --pairs $R/pairs.parquet --pair source:hypothesis --model xxl --root $SHM/metricx --empty-candidates --out $R/qe.parquet
)
$PY -I -u $CODE/tagtest.py report --pairs $R/pairs.parquet --qe $R/qe.parquet --out $R/tag-test-table.md --json $R/tag-test.json --flagged $R/flagged.md
gsutil -m -q cp $R/pairs.parquet $R/qe.parquet $R/tag-test-table.md $R/tag-test.json $R/flagged.md $T/report/
echo "SCORED: $T/report/"
EOF
  detach "sft_score_$TAG" bash ~/job_body_score_$TAG.sh
  ;;
*)
  echo "unknown stage $STAGE"
  exit 2
  ;;
esac
