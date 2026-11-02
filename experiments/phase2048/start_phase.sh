#!/bin/bash
# On every worker of mbert-mix (after setup_phase.sh): start or resume one arm of the 2048 phase, detached from ssh.
#   bash start_phase.sh ARM START_CKPT MASKING START_NAME
# ARM C: --max-length 1024, 8 per chip. ARM L: --max-length 2048, 4 per chip. On the 16 chips of the v4-32 both are
#   128 x 1024 = 64 x 2048 = 131,072 slots per optimizer step.
# START_CKPT: step directory of the start model S, e.g. gs://t5-dutch-english-usc2/mbert/culturax-nl-230m-chunk1024/ckpt_keep/653000.
# MASKING: the masking S was trained with (uniform15, uniform30 or mix); checked against S's data state.
# START_NAME: a short name of S for the run names, e.g. u15 or mix: runs p2048-<START_NAME>-C1024 and -L2048.
# Both arms: fresh AdamW, linear warmup over WARMUP steps to PEAK_LR, cosine to MIN_LR at STEPS, CulturaX-nl from a fresh
# data state, seeds and model config of the long run. A checkpoint of the arm in the bucket makes it a resume, which
# wins over --init-from; a resume with another START_CKPT or MASKING is refused.
# Environment overrides, for the smoke test only: RUN_ROOT DATA_FILES CODE PY STOP_AT CHECKPOINT_EVERY EVAL_EVERY
# EVAL_BATCHES SYSTEM_EVERY PER_DEVICE ACCUMULATION.
set -euo pipefail
[ $# = 4 ] || { echo "usage: bash start_phase.sh C|L START_CKPT MASKING START_NAME"; exit 1; }
ARM=$1; START_CKPT=${2%/}; MASKING=$3; START_NAME=$4

# The schedule of the design: the same in both arms
STEPS=20000
WARMUP=1000
PEAK_LR=1e-4
MIN_LR=1e-6
case $ARM in
  C) MAX_LENGTH=1024; PER_DEVICE=${PER_DEVICE:-8} ;;
  L) MAX_LENGTH=2048; PER_DEVICE=${PER_DEVICE:-4} ;;
  *) echo "ARM must be C or L, not $ARM"; exit 1 ;;
esac
ACCUMULATION=${ACCUMULATION:-1}
RUN_ROOT=${RUN_ROOT:-gs://t5-dutch-english-usc2/mbert/phase2048}
DATA_FILES=${DATA_FILES:-$HOME/culturax_nl_files.txt}
CODE=${CODE:-$HOME/mbert-p2048}
PY=${PY:-$HOME/mbert-jax/.venv/bin/python}
CHECKPOINT_EVERY=${CHECKPOINT_EVERY:-2000}
EVAL_EVERY=${EVAL_EVERY:-5000}
EVAL_BATCHES=${EVAL_BATCHES:-50}
SYSTEM_EVERY=${SYSTEM_EVERY:-1000}
RUN=p2048-$START_NAME-$ARM$MAX_LENGTH
OUT=$RUN_ROOT/$RUN

pgrep -f '[t]rain_minimal.py' > /dev/null && { echo "$(hostname): a trainer is running"; exit 1; }
[ -f "$DATA_FILES" ] && [ -f ~/heldout.parquet ] && [ -x "$PY" ] && [ -f "$CODE/scripts/train_minimal.py" ] || { echo "$(hostname): run setup_phase.sh first"; exit 1; }

# S's masking is in the identity of its data state, next to its checkpoints; absent means uniform15
STATE="$(dirname "$(dirname "$START_CKPT")")/ckpt_data_state/$(basename "$START_CKPT")/process_0.json"
TRAINED=$(gsutil cat "$STATE" | "$PY" -c "import json, sys; print(json.load(sys.stdin)['identity'].get('masking', 'uniform15'))")
[ "$TRAINED" = "$MASKING" ] || { echo "$(hostname): S at $START_CKPT was trained with --masking $TRAINED, not $MASKING"; exit 1; }
gsutil -q stat "$START_CKPT/commit_success.txt" || { echo "$(hostname): $START_CKPT has no commit_success.txt"; exit 1; }

# The first start records what this run starts from; a resume must match it
START="$START_CKPT $MASKING"
if gsutil -q stat "$OUT/start.txt"; then
  RECORDED=$(gsutil cat "$OUT/start.txt")
  [ "$RECORDED" = "$START" ] || { echo "$(hostname): $RUN started from '$RECORDED', not '$START'"; exit 1; }
else
  echo "$START" | gsutil -q cp - "$OUT/start.txt"
fi

pkill -f "[s]ync_$RUN.sh" || true
mkdir -p ~/logs ~/trackio
rm -f ~/logs/$RUN.exit
if gsutil -q stat "$OUT/tracking/$RUN.db"; then
  rm -f ~/trackio/$RUN.db ~/trackio/$RUN.db-wal ~/trackio/$RUN.db-shm
  gsutil -q cp "$OUT/tracking/$RUN.db" ~/trackio/$RUN.db
fi
cat > ~/run_$RUN.sh <<EOF
cd $CODE
TRACKIO_DIR=\$HOME/trackio TOKENIZERS_PARALLELISM=false PYTHONPATH=src $PY -u scripts/train_minimal.py --dataset uonlp/CulturaX --dataset-config nl --tokenizer yhavinga/dutch-llama-tokenizer --use-packing --max-length $MAX_LENGTH --hidden-size 1024 --num-layers 22 --num-heads 16 --intermediate-size 1536 --global-attn-every-n-layers 3 --lr-schedule cosine --min-lr $MIN_LR --weight-decay 0.01 --layer-norm-lr-factor 1.0 --log-every 10 --dataset-num-rows 117392666 --per-device-batch-size $PER_DEVICE --gradient-accumulation-steps $ACCUMULATION --learning-rate $PEAK_LR --warmup-steps $WARMUP --max-steps $STEPS ${STOP_AT:+--stop-at-step $STOP_AT} --masking $MASKING --init-from $START_CKPT --checkpoint-every $CHECKPOINT_EVERY --checkpoint-dir $OUT/ckpt --wandb-project $RUN --wandb-run-name $RUN --tracking-sync-dir $OUT/tracking --system-metrics-every $SYSTEM_EVERY --eval-file \$HOME/heldout.parquet --eval-every $EVAL_EVERY --eval-batches $EVAL_BATCHES --data-files-list $DATA_FILES >> ~/logs/$RUN.log 2>&1
echo EXIT \$? > ~/logs/$RUN.exit
EOF
cat > ~/sync_$RUN.sh <<EOF
w=\$(hostname | sed 's/.*-w-/w/')
while [ ! -f ~/logs/$RUN.exit ]; do gsutil -q cp ~/logs/$RUN.log $OUT/logs/\$w.log; sleep 300; done
gsutil -q cp ~/logs/$RUN.log $OUT/logs/\$w.log; gsutil -q cp ~/logs/$RUN.exit $OUT/logs/\$w.exit
EOF
setsid nohup bash ~/run_$RUN.sh > /dev/null 2>&1 < /dev/null &
setsid nohup bash ~/sync_$RUN.sh > /dev/null 2>&1 < /dev/null &
echo "started $RUN on $(hostname) from $START: ~/logs/$RUN.log, $OUT; process_0 may be any host: read the log with Step lines"
