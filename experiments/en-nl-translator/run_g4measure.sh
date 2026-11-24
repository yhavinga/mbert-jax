#!/bin/bash
# G4-maat (protocol-g4.md §23): which automatic signal picks the same side as Yeb, on mbert-mix (4 x v4-8,
# us-central2-b), one stage per call on a worker; the chain g4measure_run.sh (co1, mbert-ops) starts the stages and waits
# for their EXIT lines. Every stage runs detached from ssh with a fresh log ~/logs/gm_<stage>_$TAG.log that ends in
# "EXIT <code>" (also copied to $GM/logs/). The venvs, the tokenizers, MetricX-24, COMET-22 and the classifier's venv come
# from run_g4arm.sh setup (FULL=1 on w0 and w2).
#   bash run_g4measure.sh data         w0 (CPU): g4measure.py data (the pairs of Yeb's two samples and the 14 blind
#                                      packages, the mixed sample checked against its file), to $GM/data/ and the
#                                      classifier texts to $GM/naturalness/texts.parquet
#   bash run_g4measure.sh judge        every worker (its TPU): pairjudge.py score of its prompts (rubric, N1, N2, both
#                                      orders), to $GM/judge/scores/w<k>/
#   bash run_g4measure.sh qe           w1 (its TPU): MetricX-24 XXL QE of every side, to $GM/scores/metricx.parquet
#   bash run_g4measure.sh classify     w0 (its TPU): C1, run_arms.sh classify unchanged with NAT=$GM/naturalness (job
#                                      arms_classify_<TAG>c), to $GM/naturalness/predictions.parquet
#   bash run_g4measure.sh classify-c2  w0 (its TPU): C2 as run_g4arm.sh classify-c2, to $GM/naturalness/predictions-c2.parquet
#   bash run_g4measure.sh comet        w2 (CPU): COMET-22 of the news sides with a reference, to $GM/scores/comet.parquet
# A stage whose output is in the bucket already prints that and ends in EXIT 0; nothing is removed or overwritten. The
# code is beside this script (a copy of $G/scripts-g4m-<commit>/). TAG names the jobs of one chain.
set -euo pipefail
STAGE=${1:?a stage: data judge qe classify classify-c2 comet}
R=gs://t5-dutch-english-usc2/mbert/translator
S=$R/sft
G=$S/g4arm
GM=$G/measure                                                   # this measurement's data, scores, report
GJ=$G/judge
RL=$G/rl/G4-7
MS=$G/mixsample
B=$S/blind
PACKAGES="G4-7 G4-7-tg G4-bon-i15c2q G4-judge-i15 G4-judge-G4-0 G4-judge-i15-vs-qe G4-judge-anchor-i15 G4-8 G4-8-tg G4-8-replica G4-8-romans-bt G4-8-V3 G4-9-V3N1 RB3"
DECODE_CKPT=$R/judge/gemma-4-31b-it-maxtext/0/items            # the conversion every Gemma 4 decode uses
SHM=/dev/shm/translator
G_SHM=$SHM/g4arm
WX=$SHM/g4measure                                               # this script's own work directories
GEN=/dev/shm/translate-generic
FORK_COMMIT=f4d8615f478b6c18de9df8f2dff1f57f775a2521             # protocol-g4.md §9: every Gemma 4 decode
FPY=$GEN/venv-fork-${FORK_COMMIT:0:9}/bin/python
SFT_PY=$SHM/smoke-lora/venv/bin/python                          # setup_sft.sh (run_g4arm.sh setup)
TOK_DEC=$GEN/tokenizer-gemma4
TOK_SFT=$SHM/g4lora/tok-gemma4-31b                              # the same tokenizer.json (run_g4arm.sh setup checks)
MX_ROOT=$SHM/metricx
WORK=$G_SHM/judge-work                                          # MaxText's output and the compilation cache
DPY=$HOME/mbert-dumb/.venv/bin/python                           # C1's and C2's venv and checkout (run_g4arm.sh)
DUMB=$HOME/mbert-dumbx
DUMB_COMMIT=ae114789bf2eae153e4e3bea579299a617dcd417
CODE=$(cd "$(dirname "$0")" && pwd)
W=$(hostname | sed 's/.*-w-//')
TAG=${TAG:-manual}
mkdir -p "$G_SHM" "$WX" ~/logs

detach() {  # detach NAME COMMAND...: the command in the background, log ~/logs/NAME.log ending in EXIT <code>
  local name=$1
  shift
  [ -e ~/logs/$name.log ] && { echo "w$W: ~/logs/$name.log exists; a new start takes a new TAG"; exit 1; }
  printf '%q ' "$@" > ~/job_$name.sh
  setsid nohup bash -c "bash ~/job_$name.sh > ~/logs/$name.log 2>&1; echo EXIT \$? >> ~/logs/$name.log; gsutil -q cp ~/logs/$name.log $GM/logs/$name-w$W.log" \
    > /dev/null 2>&1 < /dev/null &
  echo "started $name on w$W"
}

body() {  # body NAME: the job's script from stdin, returns its path
  local f=~/job_body_$1.sh
  cat > "$f"
  echo "$f"
}

there_before() {  # there_before NAME WHAT: a job that only says its output is in the bucket already
  local f
  f=$(body "$1" <<EOF
echo "$2"
EOF
)
  detach "$1" bash "$f"
  exit 0
}

tpu_free() {
  [ -z "$(sudo lsof -t /dev/accel* /dev/vfio/* 2> /dev/null)" ] || { echo "w$W: the TPU is in use"; exit 1; }
}

only() {  # only WORKER: the stage runs on that worker alone
  [ "$W" = "$1" ] || { echo "$STAGE runs on w$1 only"; exit 1; }
}

ready() {
  [ -x "$FPY" ] && [ -x "$SFT_PY" ] && [ -s "$TOK_DEC/tokenizer.json" ] && [ -s "$TOK_SFT/tokenizer.json" ] \
    || { echo "w$W: no fork venv, SFT venv or tokenizer (run_g4arm.sh setup)"; exit 1; }
}

dumb_ready() {
  [ -x "$DPY" ] || { echo "w$W: no $DPY"; exit 1; }
  [ "$(git -C "$DUMB" rev-parse HEAD 2> /dev/null)" = "$DUMB_COMMIT" ] || { echo "w$W: $DUMB is not at $DUMB_COMMIT"; exit 1; }
  [ -z "$(git -C "$DUMB" status --porcelain --untracked-files=no)" ] || { echo "w$W: $DUMB has changes"; exit 1; }
}

# A worker of the v4-32 as a TPU of its own 4 chips (run_g4arm.sh)
ONE_HOST="export TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 TPU_PROCESS_BOUNDS=1,1,1 TPU_VISIBLE_CHIPS=0,1,2,3 TPU_PROCESS_PORT=8476
export TPU_PROCESS_ADDRESSES=localhost:8476 CLOUD_TPU_TASK_ID=0 ALLOW_MULTIPLE_LIBTPU_LOAD=1
export XLA_FLAGS=--xla_dump_hlo_as_text=false TPU_LOG_DIR=$SHM/tpu_logs TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 HF_HUB_DISABLE_PROGRESS_BARS=1"
ONE_HOST_DUMB="export TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 TPU_PROCESS_BOUNDS=1,1,1 TPU_VISIBLE_CHIPS=0,1,2,3 TPU_PROCESS_PORT=8476
export TPU_PROCESS_ADDRESSES=localhost:8476 CLOUD_TPU_TASK_ID=0 ALLOW_MULTIPLE_LIBTPU_LOAD=1
export XLA_FLAGS=--xla_dump_hlo_as_text=false TPU_LOG_DIR=$SHM/tpu_logs TOKENIZERS_PARALLELISM=false"
PREFLIGHT="timeout 600 $FPY -c \"import jax; assert (jax.device_count(), jax.process_count()) == (4, 1), (jax.device_count(), jax.process_count()); print('devices', jax.devices())\" || { echo 'preflight failed: not 4 chips of one process'; exit 2; }"
CPU="export JAX_PLATFORMS=cpu OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1"

case $STAGE in
data)
  only 0
  ready
  NAME=gm_data_$TAG
  gsutil -q stat "$GM/data/data.json" && there_before "$NAME" "MEASUREDATA_DONE $GM/data/ (there before)"
  X=$WX/data-$TAG
  {
    echo "set -euo pipefail"
    echo "$CPU"
    echo "mkdir -p $X/blind $X/in/eval/base $X/in/eval/G4-8-s140 $X/in/eval/G4-7-i15 $X/in/val $X/out"
    for p in $PACKAGES steekproef-yeb steekproef-yeb-g4-8-mix; do echo "gsutil -m -q cp -r $B/$p $X/blind/"; done
    echo "gsutil -m -q cp '$G/eval/G4-0/pairs/pairs-w*.parquet' $X/in/eval/base/"
    echo "gsutil -m -q cp '$G/eval/G4-8-s140/pairs/pairs-w*.parquet' $X/in/eval/G4-8-s140/"
    echo "gsutil -m -q cp '$G/eval/G4-7-i15/pairs/pairs-w*.parquet' $X/in/eval/G4-7-i15/"
    echo "gsutil -m -q cp $MS/pairs/G4-0.parquet $MS/pairs/G4-8-s140.parquet $X/in/"
    echo "gsutil -q cp $G/data/eval/evalsets.parquet $X/in/evalsets.parquet"
    echo "gsutil -m -q cp '$GJ/val/data/prompts-w*.parquet' $X/in/val/"
    echo "nice -n 19 ionice -c3 $SFT_PY -I -u $CODE/g4measure.py data --blind $X/blind --eval 'base=$X/in/eval/base/pairs-w*.parquet' --eval 'G4-8-s140=$X/in/eval/G4-8-s140/pairs-w*.parquet' --eval 'G4-7-i15=$X/in/eval/G4-7-i15/pairs-w*.parquet' --mix base=$X/in/G4-0.parquet --mix G4-8-s140=$X/in/G4-8-s140.parquet --evalsets $X/in/evalsets.parquet --val-prompts '$X/in/val/prompts-w*.parquet' --tokenizer $TOK_SFT --out $X/out"
    echo "gsutil -m -q cp $X/out/prompts-w*.parquet $X/out/pairs.parquet $X/out/refs.parquet $X/out/qe-in.parquet $X/out/comet-in.parquet $GM/data/"
    echo "gsutil -q cp $X/out/texts.parquet $GM/naturalness/texts.parquet"
    echo "gsutil -q cp $X/out/data.json $GM/data/data.json"
    echo "echo \"MEASUREDATA_DONE $GM/data/\""
  } > ~/job_body_$NAME.sh
  detach "$NAME" bash ~/job_body_$NAME.sh
  ;;
judge)
  ready
  NAME=gm_judge_$TAG
  gsutil -q stat "$GM/data/prompts-w$W.parquet" || { echo "no $GM/data/prompts-w$W.parquet"; exit 1; }
  tpu_free
  X=$WX/judge-$TAG
  # pairjudge.py score skips the shards in the bucket (a restart goes on where it stopped)
  cat > ~/job_body_$NAME.sh <<EOF
set -euo pipefail
$ONE_HOST
mkdir -p $X $WORK
gsutil -q cp $GM/data/prompts-w$W.parquet $X/prompts.parquet
$PREFLIGHT
$FPY -u $CODE/pairjudge.py score --prompts $X/prompts.parquet --out $GM/judge/scores/w$W --ckpt $DECODE_CKPT --tokenizer $TOK_DEC --work-dir $WORK
echo "MEASUREJUDGE_DONE w$W $GM/judge/scores/w$W/"
EOF
  detach "$NAME" bash ~/job_body_$NAME.sh
  ;;
qe)
  only 1
  ready
  NAME=gm_qe_$TAG
  gsutil -q stat "$GM/scores/metricx.parquet" && there_before "$NAME" "MEASUREQE_DONE $GM/scores/metricx.parquet (there before)"
  gsutil -q stat "$GM/data/qe-in.parquet" || { echo "no $GM/data/qe-in.parquet"; exit 1; }
  [ -d "$MX_ROOT/metricx-24-hybrid-xxl-v2p6-bfloat16" ] || { echo "w$W: no MetricX-24 XXL in $MX_ROOT (run_g4arm.sh setup)"; exit 1; }
  tpu_free
  X=$WX/qe-$TAG
  cat > ~/job_body_$NAME.sh <<EOF
set -euo pipefail
$ONE_HOST
export JAX_COMPILATION_CACHE_DIR=$G_SHM/jax_cache_metricx
mkdir -p $X
gsutil -q cp $GM/data/qe-in.parquet $X/qe-in.parquet
$PREFLIGHT
$FPY -I -u $CODE/metricx_qe.py score --pairs $X/qe-in.parquet --pair source:hypothesis --model xxl --root $MX_ROOT --empty-candidates --out $X/metricx.parquet
gsutil -q cp $X/metricx.parquet $GM/scores/metricx.parquet
echo "MEASUREQE_DONE $GM/scores/metricx.parquet"
EOF
  detach "$NAME" bash ~/job_body_$NAME.sh
  ;;
classify)
  only 0
  gsutil -q stat "$GM/naturalness/texts.parquet" || { echo "no $GM/naturalness/texts.parquet (data)"; exit 1; }
  ! gsutil -q stat "$GM/naturalness/predictions.parquet" || { echo "$GM/naturalness/predictions.parquet exists: it stays as it is"; exit 1; }
  # C1: run_arms.sh's classify, unchanged, on this measurement's texts; its job is arms_classify_<TAG>c
  NAT=$GM/naturalness TAG=${TAG}c bash "$CODE/run_arms.sh" classify
  ;;
classify-c2)
  only 0
  NAME=gm_classify_c2_$TAG
  gsutil -q stat "$GM/naturalness/predictions-c2.parquet" && there_before "$NAME" "CLASSIFIED_C2 $GM/naturalness/predictions-c2.parquet (there before)"
  for f in "$RL/c2/model/params.pkl" "$RL/c2/model/config.json" "$GM/naturalness/texts.parquet"; do
    gsutil -q stat "$f" || { echo "no $f"; exit 1; }
  done
  tpu_free
  dumb_ready
  X=$WX/classify-c2-$TAG
  # as run_g4arm.sh classify-c2 (amendment G4-7.8): C2 in C1's venv and checkout, --max-length 1024
  cat > ~/job_body_$NAME.sh <<EOF
set -euo pipefail
$ONE_HOST_DUMB
export JAX_COMPILATION_CACHE_DIR=$G_SHM/jax_cache_dumb PYTHONPATH=src
cd $DUMB
mkdir -p $X/model $G_SHM/jax_cache_dumb
gsutil -m -q cp $RL/c2/model/params.pkl $RL/c2/model/config.json $X/model/
gsutil -q cp $GM/naturalness/texts.parquet $X/texts.parquet
$DPY -u $CODE/naturalness.py classify --dumb-code $DUMB --model $X/model --texts $X/texts.parquet --max-length 1024 --out $X/predictions-c2.parquet
gsutil -q cp $X/predictions-c2.parquet $GM/naturalness/predictions-c2.parquet
echo "CLASSIFIED_C2 $GM/naturalness/predictions-c2.parquet"
EOF
  detach "$NAME" bash ~/job_body_$NAME.sh
  ;;
comet)
  only 2
  NAME=gm_comet_$TAG
  gsutil -q stat "$GM/scores/comet.parquet" && there_before "$NAME" "MEASURECOMET_DONE $GM/scores/comet.parquet (there before)"
  gsutil -q stat "$GM/data/comet-in.parquet" || { echo "no $GM/data/comet-in.parquet"; exit 1; }
  [ -x "$SHM/comet-venv/bin/python" ] || { echo "w$W: no COMET venv (run_g4arm.sh setup with FULL=1)"; exit 1; }
  X=$WX/comet-$TAG
  cat > ~/job_body_$NAME.sh <<EOF
set -euo pipefail
mkdir -p $X
gsutil -q cp $GM/data/comet-in.parquet $X/comet-in.parquet
nice -n 10 $SHM/comet-venv/bin/python -I -u $CODE/comet_score.py score --root $SHM/comet --pairs $X/comet-in.parquet --out $X/comet.parquet
gsutil -q cp $X/comet.parquet $GM/scores/comet.parquet
echo "MEASURECOMET_DONE $GM/scores/comet.parquet"
EOF
  detach "$NAME" bash ~/job_body_$NAME.sh
  ;;
*)
  echo "unknown stage $STAGE (data judge qe classify classify-c2 comet)"
  exit 1
  ;;
esac
