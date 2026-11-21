#!/bin/bash
# The judge side of arm G4-7 (protocol-g4.md §14, G4-7.4a and G4-7.6) on a worker of mbert-mix, one stage per call; the
# chain (g4arm7_run.sh, co1) starts the stages and waits for their EXIT lines. A stage runs detached from ssh with a
# fresh log ~/logs/<job>.log that ends in "EXIT <code>" (as run_arms.sh); DRY=1 prints the job instead of starting it.
#   bash run_g4arm7_judges.sh convert    w1 (CPU): dmbert v2 @ 1306000 converted for finetune.py as C1's S @ 653k was
#                                        (convert_gcs_checkpoint.py --params-only, head 0), to $C2_W/v2   (job c2_convert_$TAG)
#   bash run_g4arm7_judges.sh data       w1 (CPU, nice 19): g4c2.py data7 on G4-3's BT train split ($G/input/bt3), to
#                                        $RL/c2/data/ (classifier.parquet, docs.txt, report.json)     (job c2_data_$TAG)
#   bash run_g4arm7_judges.sh train      w1 (its TPU): naturalness.py classifier on v2, 3 epochs, LR 5e-5, seed 42,
#                                        --max-length 1024, to $RL/c2/model/                          (job c2_train_$TAG)
#   bash run_g4arm7_judges.sh texts      w1 (CPU, nice 19): g4c2.py cal7-texts, to $RL/c2/cal7/texts.parquet
#   bash run_g4arm7_judges.sh classify   w1 (its TPU): naturalness.py classify with C2 on those texts, to
#                                        $RL/c2/cal7/predictions.parquet                          (job c2_classify_$TAG)
#   bash run_g4arm7_judges.sh cal        w1 (CPU): g4c2.py cal7, to $RL/c2/cal7/; the log has C2CAL7_OK or C2CAL7_FAIL
#   ROLE=metricx PART=0 bash run_g4arm7_judges.sh serve   w0 (PART=1: w2): g4judge.py serve --role metricx in the
#                                        fork's venv, MetricX-24 hybrid XXL from $MX_ROOT    (job judge_metricx_w<W>_$TAG)
#   ROLE=c2 bash run_g4arm7_judges.sh serve               w1: g4judge.py serve --role c2 in the DUMB venv, C2 from
#                                        $RL/c2/model/                                         (job judge_c2_w1_$TAG)
# The judges stop on $RL/judges/STOP (written by the chain) or after MAX_IDLE s without work (default 6 h).
# CALIB: "NAME=gs://.../texts.parquet:gs://.../predictions.parquet ..." (C1's existing classify runs, for the reported
# Spearman with C1; spec §5.3.5).
set -euo pipefail

R=gs://t5-dutch-english-usc2/mbert/translator
S=$R/sft
E=$R/eval
G=$S/g4arm
RL=${RL:-$G/rl/G4-7}
V2=gs://t5-dutch-english-usc2/mbert/v2-fw2optout-230m/ckpt_keep/1306000   # Yeb 21 Nov: C2's encoder
C1_DATA=$S/arms/data/classifier.parquet
BLIND=$S/blind/fixed-2026-11-10/blind100.jsonl
FLAN_EVAL=$R/mainset/flan-eval/eval-rows.parquet
SHM=/dev/shm/translator
C2_W=$SHM/g4arm7
DPY=$HOME/mbert-dumb/.venv/bin/python   # the venv of the DUMB rounds (jax, flax, optax, transformers, pyarrow, GCS)
DUMB=$HOME/mbert-dumbx                  # a checkout of mbert-jax branch dumb-extra
DUMB_COMMIT=ae114789bf2eae153e4e3bea579299a617dcd417   # as C1 (protocol-g4.md G4-7.6)
GEN=/dev/shm/translate-generic
FORK_COMMIT=f4d8615f478b6c18de9df8f2dff1f57f775a2521
FPY=$GEN/venv-fork-${FORK_COMMIT:0:9}/bin/python
MX_ROOT=$SHM/metricx
CODE=$(cd "$(dirname "$0")" && pwd)
W=$(hostname | sed 's/.*-w-//')
TAG=${TAG:-manual}
MAX_IDLE=${MAX_IDLE:-21600}
mkdir -p ~/logs

detach() {  # detach NAME COMMAND...: the command in the background, log ~/logs/NAME.log ending in EXIT <code>
  local name=$1
  shift
  if [ -n "${DRY:-}" ]; then
    echo "DRY $name:"; cat "$2"; return 0
  fi
  [ -e ~/logs/$name.log ] && { echo "w$W: ~/logs/$name.log exists; a new start takes a new TAG"; exit 1; }
  printf '%q ' "$@" > ~/job_$name.sh
  setsid nohup bash -c "bash ~/job_$name.sh > ~/logs/$name.log 2>&1; echo EXIT \$? >> ~/logs/$name.log" \
    > /dev/null 2>&1 < /dev/null &
  echo "started $name on w$W"
}

tpu_free() {
  [ -n "${DRY:-}" ] && return 0
  [ -z "$(sudo lsof -t /dev/accel* /dev/vfio/* 2> /dev/null)" ] || { echo "w$W: the TPU is in use"; exit 1; }
}

dumb_ready() {
  [ -n "${DRY:-}" ] && return 0
  [ -x "$DPY" ] || { echo "w$W: no $DPY"; exit 1; }
  [ "$(git -C "$DUMB" rev-parse HEAD 2> /dev/null)" = "$DUMB_COMMIT" ] || { echo "w$W: $DUMB is not at $DUMB_COMMIT"; exit 1; }
  [ -z "$(git -C "$DUMB" status --porcelain --untracked-files=no)" ] || { echo "w$W: $DUMB has changes"; exit 1; }
}

absent() {  # absent GS_OBJECT: stop when a write-once output is there
  [ -n "${DRY:-}" ] && return 0
  ! gsutil -q stat "$1" || { echo "$1 exists: G4-7's outputs are written once"; exit 1; }
}

# A host of mbert-mix as a TPU of its own 4 chips (as run_arms.sh)
ONE_HOST="export TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 TPU_PROCESS_BOUNDS=1,1,1 TPU_VISIBLE_CHIPS=0,1,2,3 TPU_PROCESS_PORT=8476
export TPU_PROCESS_ADDRESSES=localhost:8476 CLOUD_TPU_TASK_ID=0 ALLOW_MULTIPLE_LIBTPU_LOAD=1
export XLA_FLAGS=--xla_dump_hlo_as_text=false TPU_LOG_DIR=$SHM/tpu_logs TOKENIZERS_PARALLELISM=false"

body() {  # body NAME: the job's script from stdin, returns its path
  local f=~/job_body_$1.sh
  cat > "$f"
  echo "$f"
}

stage=${1:?stage: convert data train texts classify cal serve}
case "$stage" in
convert)
  dumb_ready
  f=$(body c2_convert_$TAG <<EOF
set -euo pipefail
cd $DUMB
export PYTHONPATH=src JAX_PLATFORMS=cpu
mkdir -p $C2_W
gsutil -q stat $V2/commit_success.txt
# as run_arms.sh converted S @ 653k for C1 (dumb-extra's dumbx.sh): params only, the tokenizer of the dmbert runs
nice -n 19 $DPY scripts/convert_gcs_checkpoint.py --checkpoint $V2 --params-only --output $C2_W/v2 --tokenizer yhavinga/dutch-llama-tokenizer --global-attn-every-n-layers 3 < /dev/null
rm -f $C2_W/v2/model.safetensors
nice -n 19 $DPY - $C2_W/v2/params.pkl <<'PY'
import pickle, re, sys
path = sys.argv[1]
p = pickle.load(open(path, "rb"))
extra = [n for n in p if re.fullmatch(r"(prediction_head|decoder)_[0-9]+", n)]
if extra:
    for n in extra:
        del p[n]
    pickle.dump(p, open(path, "wb"))
    print(f"head 0 kept as prediction_head/decoder; left out: {sorted(extra)}")
PY
ls -la $C2_W/v2
echo "CONVERTED: $C2_W/v2"
EOF
)
  detach "c2_convert_$TAG" bash "$f"
  ;;
data)
  dumb_ready
  absent $RL/c2/data/classifier.parquet
  X=$C2_W/data-$TAG
  f=$(body c2_data_$TAG <<EOF
set -euo pipefail
[ -f $C2_W/v2/config.json ] || { echo "no $C2_W/v2: run the stage convert first"; exit 1; }
mkdir -p $X/in
gsutil -m -q rsync -r $G/input/bt3 $X/in/bt3
gsutil -m -q rsync -r $G/input/bt3-val $X/in/bt3-val
gsutil -q cp $C1_DATA $E/data/evalsets.parquet $BLIND $FLAN_EVAL $X/in/
nice -n 19 $DPY -I -u $CODE/g4c2.py data7 --bt $X/in/bt3 --bt-validation $X/in/bt3-val --c1-data $X/in/classifier.parquet --evalsets $X/in/evalsets.parquet --blind $X/in/blind100.jsonl --flan-eval $X/in/eval-rows.parquet --checkpoint $C2_W/v2 --out $X/out
gsutil -q cp $X/out/classifier.parquet $X/out/docs.txt $X/out/report.json $RL/c2/data/
echo "C2_DATA: $RL/c2/data/"
EOF
)
  detach "c2_data_$TAG" bash "$f"
  ;;
train)
  tpu_free
  dumb_ready
  absent $RL/c2/model/results.json
  X=$C2_W/train-$TAG
  f=$(body c2_train_$TAG <<EOF
set -euo pipefail
$ONE_HOST
export JAX_COMPILATION_CACHE_DIR=$C2_W/jax_cache PYTHONPATH=src
cd $DUMB
[ -f $C2_W/v2/params.pkl ] || { echo "no $C2_W/v2: run the stage convert first"; exit 1; }
timeout 600 $DPY -c "import jax; assert (jax.device_count(), jax.process_count()) == (4, 1), (jax.device_count(), jax.process_count())"
mkdir -p $X
gsutil -q cp $RL/c2/data/classifier.parquet $X/classifier.parquet
# C1's command (run_arms.sh classifier) with v2 as the encoder: 3 epochs, LR 5e-5, seed 42, 8 per chip, 1,024 tokens
$DPY -u $CODE/naturalness.py classifier --dumb-code $DUMB --checkpoint $C2_W/v2 --data $X/classifier.parquet --max-length 1024 --epochs 3 --learning-rate 5e-5 --seed 42 --batch-size 8 --out $X/model
gsutil -m -q cp $X/model/results.json $X/model/params.pkl $X/model/config.json $RL/c2/model/
echo "C2_MODEL: $RL/c2/model/"
EOF
)
  detach "c2_train_$TAG" bash "$f"
  ;;
texts)
  dumb_ready
  absent $RL/c2/cal7/texts.parquet
  X=$C2_W/texts-$TAG
  calib_args=""
  calib_fetch=""
  for c in ${CALIB:-}; do
    name=${c%%=*}; rest=${c#*=}; texts=${rest%%:gs://*}; preds=gs://${rest#*:gs://}
    calib_fetch="$calib_fetch
gsutil -q cp $texts $X/calib/$name-texts.parquet
gsutil -q cp $preds $X/calib/$name-c1.parquet"
    calib_args="$calib_args --calib $name=$X/calib/$name-texts.parquet"
  done
  f=$(body c2_texts_$TAG <<EOF
set -euo pipefail
mkdir -p $X/sampling $X/g40 $X/calib $X/c2-config
gsutil -q cp $E/data/evalsets.parquet $X/evalsets.parquet
gsutil -m -q cp '$G/eval/G4-0/pairs/pairs-w*.parquet' '$G/eval/G4-0/scores/metricx-w*.parquet' $X/g40/
gsutil -m -q rsync -r -x '.*comet.*' $G/sampling/G4-0 $X/sampling/G4-0
gsutil -m -q rsync -r -x '.*comet.*' $G/sampling/G4-4 $X/sampling/G4-4
gsutil -q cp $RL/c2/data/classifier.parquet $X/c2-data.parquet
gsutil -q cp $RL/c2/model/config.json $X/c2-config/config.json
$calib_fetch
nice -n 19 $DPY -I -u $CODE/g4c2.py cal7-texts --evalsets $X/evalsets.parquet --g40 '$X/g40/pairs-w*.parquet' --sampling $X/sampling --c2-data $X/c2-data.parquet --checkpoint $X/c2-config $calib_args --out $X/texts.parquet
gsutil -q cp $X/texts.parquet $RL/c2/cal7/texts.parquet
echo "C2_TEXTS: $RL/c2/cal7/texts.parquet"
EOF
)
  detach "c2_texts_$TAG" bash "$f"
  ;;
classify)
  tpu_free
  dumb_ready
  absent $RL/c2/cal7/predictions.parquet
  X=$C2_W/classify-$TAG
  f=$(body c2_classify_$TAG <<EOF
set -euo pipefail
$ONE_HOST
export JAX_COMPILATION_CACHE_DIR=$C2_W/jax_cache PYTHONPATH=src
cd $DUMB
mkdir -p $X/model
gsutil -m -q cp $RL/c2/model/params.pkl $RL/c2/model/config.json $X/model/
gsutil -q cp $RL/c2/cal7/texts.parquet $X/texts.parquet
$DPY -u $CODE/naturalness.py classify --dumb-code $DUMB --model $X/model --texts $X/texts.parquet --max-length 1024 --out $X/predictions.parquet
gsutil -q cp $X/predictions.parquet $RL/c2/cal7/predictions.parquet
echo "C2_PREDICTIONS: $RL/c2/cal7/predictions.parquet"
EOF
)
  detach "c2_classify_$TAG" bash "$f"
  ;;
cal)
  dumb_ready
  absent $RL/c2/cal7/report.json
  X=$C2_W/texts-${TEXTS_TAG:-$TAG}
  c1_args=""
  for c in ${CALIB:-}; do
    name=${c%%=*}
    c1_args="$c1_args --c1 $name=$X/calib/$name-c1.parquet"
  done
  f=$(body c2_cal_$TAG <<EOF
set -euo pipefail
[ -f $X/texts.parquet ] || { echo "no $X/texts.parquet (TEXTS_TAG: the tag of the stage texts)"; exit 1; }
mkdir -p $X/cal7
gsutil -q cp $RL/c2/model/results.json $X/results.json
gsutil -q cp $RL/c2/cal7/predictions.parquet $X/predictions.parquet
set +e
nice -n 19 $DPY -I -u $CODE/g4c2.py cal7 --results $X/results.json --texts $X/texts.parquet --predictions $X/predictions.parquet --c2-data $X/c2-data.parquet --sampling $X/sampling --g40-scores '$X/g40/metricx-w*.parquet' $c1_args --out $X/cal7
code=\$?
set -e
gsutil -q cp $X/cal7/report.json $X/cal7/report.md $RL/c2/cal7/
exit \$code
EOF
)
  detach "c2_cal_$TAG" bash "$f"
  ;;
serve)
  ROLE=${ROLE:?ROLE=metricx (PART=0 or 1) or ROLE=c2}
  tpu_free
  if [ "$ROLE" = metricx ]; then
    PART=${PART:?PART=0 (w0) or 1 (w2)}
    [ -n "${DRY:-}" ] || [ -x "$FPY" ] || { echo "w$W: no $FPY (setup_generic.sh venv)"; exit 1; }
    [ -n "${DRY:-}" ] || [ -d $MX_ROOT/metricx-24-hybrid-xxl-v2p6-bfloat16 ] || { echo "w$W: no MetricX-24 in $MX_ROOT"; exit 1; }
    f=$(body judge_metricx_w${W}_$TAG <<EOF
set -euo pipefail
$ONE_HOST
export HF_HUB_OFFLINE=1
$FPY -I -u $CODE/g4judge.py serve --rl $RL --role metricx --worker w$W --part $PART --root $MX_ROOT --writers iter=0,1,2,3 --writers val=3 --writers anchor=0,1,2,3 --max-idle $MAX_IDLE
EOF
)
    detach "judge_metricx_w${W}_$TAG" bash "$f"
  else
    dumb_ready
    X=$C2_W/judge-$TAG
    f=$(body judge_c2_w${W}_$TAG <<EOF
set -euo pipefail
$ONE_HOST
export JAX_COMPILATION_CACHE_DIR=$C2_W/jax_cache PYTHONPATH=src
cd $DUMB
mkdir -p $X/model
gsutil -m -q cp $RL/c2/model/params.pkl $RL/c2/model/config.json $X/model/
$DPY -u $CODE/g4judge.py serve --rl $RL --role c2 --worker w$W --dumb-code $DUMB --model $X/model --writers iter=0,1,2,3 --writers val=3 --writers anchor=0,1,2,3 --max-idle $MAX_IDLE
EOF
)
    detach "judge_c2_w${W}_$TAG" bash "$f"
  fi
  ;;
*)
  echo "unknown stage $stage"; exit 1 ;;
esac
