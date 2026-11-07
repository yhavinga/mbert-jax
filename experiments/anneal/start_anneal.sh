#!/bin/bash
# On every worker of mbert-mix: start or resume one arm of the anneal test, detached from ssh.
#   COMMIT=<commit> bash start_anneal.sh START_CKPT START_MASKING ANNEAL_MASKING START_NAME
# The arms: S + 20,000 steps of CulturaX-nl with single-token masking at a low rate, from the parameters of S only
# (--init-from), with the schedule and data of the 2048 phase's arm C (experiments/phase2048/start_phase.sh): fresh
# AdamW, linear warmup over 1,000 steps to 1e-4, cosine to 1e-6 at step 20,000, 128 x 1024 per step, a fresh data state.
# Only the start model and its masking differ between arms, so u15 + anneal is the control for mix + anneal.
# START_CKPT: step directory of S. START_MASKING: the masking S was trained with, checked against S's data state.
# ANNEAL_MASKING: uniform5 or uniform10. START_NAME: a short name of S, e.g. u15 or mix: run anneal-<START_NAME>-<ANNEAL_MASKING>.
# The code is a clone of its own (~/mbert-anneal, at COMMIT), with the venv of the corpus runs; PY overrides it on a VM
# whose venv of that lock is elsewhere (mbert-corpus: PY=$HOME/mbert-dumb/.venv/bin/python).
set -euo pipefail
[ $# = 4 ] || { echo "usage: COMMIT=<commit> bash start_anneal.sh START_CKPT START_MASKING ANNEAL_MASKING START_NAME"; exit 1; }
START_CKPT=${1%/}; START_MASKING=$2; ANNEAL_MASKING=$3; START_NAME=$4
COMMIT=${COMMIT:?set COMMIT to the commit of the anneal branch}
case $ANNEAL_MASKING in uniform5|uniform10) ;; *) echo "ANNEAL_MASKING must be uniform5 or uniform10, not $ANNEAL_MASKING"; exit 1 ;; esac

STEPS=20000
WARMUP=1000
PEAK_LR=1e-4
MIN_LR=1e-6
RUN_ROOT=gs://t5-dutch-english-usc2/mbert/anneal
DATA_FILES=$HOME/culturax_nl_files.txt
CODE=$HOME/mbert-anneal
PY=${PY:-$HOME/mbert-corpus/.venv/bin/python}
RUN=anneal-$START_NAME-$ANNEAL_MASKING
OUT=$RUN_ROOT/$RUN

pgrep -f '[t]rain_minimal.py' > /dev/null && { echo "$(hostname): a trainer is running"; exit 1; }
[ -f "$DATA_FILES" ] && [ -f ~/heldout.parquet ] && [ -x "$PY" ] || { echo "$(hostname): needs the CulturaX share, ~/heldout.parquet and $PY"; exit 1; }
[ -d "$CODE/.git" ] || git clone -q https://github.com/yhavinga/mbert-jax.git "$CODE"
git -C "$CODE" fetch -q origin anneal
git -C "$CODE" checkout -q "$COMMIT"

# S's masking is in the identity of its data state, next to its checkpoints; absent means uniform15
STATE="$(dirname "$(dirname "$START_CKPT")")/ckpt_data_state/$(basename "$START_CKPT")/process_0.json"
TRAINED=$(gsutil cat "$STATE" | "$PY" -c "import json, sys; print(json.load(sys.stdin)['identity'].get('masking', 'uniform15'))")
[ "$TRAINED" = "$START_MASKING" ] || { echo "$(hostname): S at $START_CKPT was trained with --masking $TRAINED, not $START_MASKING"; exit 1; }
gsutil -q stat "$START_CKPT/commit_success.txt" || { echo "$(hostname): $START_CKPT has no commit_success.txt"; exit 1; }

# The first start records what this run starts from; a resume must match it
START="$START_CKPT $START_MASKING $ANNEAL_MASKING $COMMIT"
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
TRACKIO_DIR=\$HOME/trackio TOKENIZERS_PARALLELISM=false PYTHONPATH=src $PY -u scripts/train_minimal.py --dataset uonlp/CulturaX --dataset-config nl --tokenizer yhavinga/dutch-llama-tokenizer --use-packing --max-length 1024 --hidden-size 1024 --num-layers 22 --num-heads 16 --intermediate-size 1536 --global-attn-every-n-layers 3 --lr-schedule cosine --min-lr $MIN_LR --weight-decay 0.01 --layer-norm-lr-factor 1.0 --log-every 10 --dataset-num-rows 117392666 --per-device-batch-size 8 --learning-rate $PEAK_LR --warmup-steps $WARMUP --max-steps $STEPS --masking $ANNEAL_MASKING --init-from $START_CKPT --checkpoint-every 2000 --checkpoint-dir $OUT/ckpt --wandb-project $RUN --wandb-run-name $RUN --tracking-sync-dir $OUT/tracking --system-metrics-every 1000 --eval-file \$HOME/heldout.parquet --eval-every 5000 --eval-batches 50 --data-files-list $DATA_FILES >> ~/logs/$RUN.log 2>&1
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
