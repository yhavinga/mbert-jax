#!/bin/bash
# On every worker of mbert-corpus (after setup_flan.sh): start or resume arm F of the FLAN-decay proof, detached from ssh.
#   COMMIT=<commit> PY=$HOME/mbert-dumb/.venv/bin/python bash start_flan.sh START_CKPT START_MASKING
# Arm F is arm W (anneal-u15-uniform5: experiments/anneal/start_anneal.sh S uniform15 uniform5 u15) with other data
# and another collator, and nothing else: S + 20,000 steps from the parameters of S only (--init-from), fresh AdamW,
# linear warmup over 1,000 steps to 1e-4, cosine to 1e-6 at step 20,000, 128 x 1024 per step, the same held-out eval,
# seeds and model. The data: this host's decay mix (~/flan_decay_files.txt, scripts/build_decay_mix.py), W's first
# CulturaX documents in W's order with FLAN at 10% of the tokens in between. The masking: --masking decay, web text at
# 5% single tokens with 80/10/10 as W's uniform5, instructions and <answer> never, answers of up to 16 tokens whole as
# <mask>, longer ones with span_mu8_r50 (DecayCollator in src/mmbjax/training/packing.py).
# START_CKPT: step directory of S. START_MASKING: the masking S was trained with, checked against S's data state.
# The code is a clone of its own (~/mbert-flan, at COMMIT); PY overrides the venv (on mbert-corpus
# PY=$HOME/mbert-dumb/.venv/bin/python, as for W). A checkpoint of the arm in the bucket makes it a resume, which wins
# over --init-from; a resume with another START_CKPT, START_MASKING or COMMIT is refused.
# Environment overrides, for the smoke test only: RUN_ROOT RUN STOP_AT CHECKPOINT_EVERY EVAL_EVERY EVAL_BATCHES SYSTEM_EVERY.
set -euo pipefail
[ $# = 2 ] || { echo "usage: COMMIT=<commit> bash start_flan.sh START_CKPT START_MASKING"; exit 1; }
START_CKPT=${1%/}; START_MASKING=$2
COMMIT=${COMMIT:?set COMMIT to the commit of the flan-decay branch}

# The schedule of arm W
STEPS=20000
WARMUP=1000
PEAK_LR=1e-4
MIN_LR=1e-6
# The masking of arm F
WEB_MASK_RATE=0.05
ANSWER_FULL_MAX=16
RUN_ROOT=${RUN_ROOT:-gs://t5-dutch-english-usc2/mbert/flan-decay}
RUN=${RUN:-flan-F}
DATA_FILES=$HOME/flan_decay_files.txt
CODE=$HOME/mbert-flan
PY=${PY:-$HOME/mbert-corpus/.venv/bin/python}
CHECKPOINT_EVERY=${CHECKPOINT_EVERY:-2000}
EVAL_EVERY=${EVAL_EVERY:-5000}
EVAL_BATCHES=${EVAL_BATCHES:-50}
SYSTEM_EVERY=${SYSTEM_EVERY:-1000}
OUT=$RUN_ROOT/$RUN

pgrep -f '[t]rain_minimal.py|[e]val_windows.py|[e]val_zeroshot_mlm.py|[f]inetune.py' > /dev/null && { echo "$(hostname): the TPU is in use"; exit 1; }
[ -f "$DATA_FILES" ] && [ -f ~/heldout.parquet ] && [ -x "$PY" ] || { echo "$(hostname): needs $DATA_FILES (setup_flan.sh), ~/heldout.parquet and $PY"; exit 1; }
while read -r f; do [ -s "$f" ] || { echo "$(hostname): missing $f of $DATA_FILES"; exit 1; }; done < "$DATA_FILES"
[ "$(md5sum ~/heldout.parquet | cut -d' ' -f1)" = b2f579a876d06eef40c94270537bb7af ] || { echo "$(hostname): held-out file is not the long run's"; exit 1; }
[ -d "$CODE/.git" ] || git clone -q https://github.com/yhavinga/mbert-jax.git "$CODE"
git -C "$CODE" fetch -q origin flan-decay
git -C "$CODE" checkout -q "$COMMIT"

# S's masking is in the identity of its data state, next to its checkpoints; absent means uniform15
STATE="$(dirname "$(dirname "$START_CKPT")")/ckpt_data_state/$(basename "$START_CKPT")/process_0.json"
TRAINED=$(gsutil cat "$STATE" | "$PY" -c "import json, sys; print(json.load(sys.stdin)['identity'].get('masking', 'uniform15'))")
[ "$TRAINED" = "$START_MASKING" ] || { echo "$(hostname): S at $START_CKPT was trained with --masking $TRAINED, not $START_MASKING"; exit 1; }
gsutil -q stat "$START_CKPT/commit_success.txt" || { echo "$(hostname): $START_CKPT has no commit_success.txt"; exit 1; }

# The first start records what this run starts from; a resume must match it. The mix of each host is in its data state.
START="$START_CKPT $START_MASKING decay $WEB_MASK_RATE $ANSWER_FULL_MAX $COMMIT"
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
# W's command (experiments/anneal/start_anneal.sh), but for --masking and --data-files-list. --dataset and
# --dataset-num-rows are W's too: with --data-files-list the first only names the data in the identity, the second
# only scales a logged fraction of a pass.
cat > ~/run_$RUN.sh <<EOF
cd $CODE
TRACKIO_DIR=\$HOME/trackio TOKENIZERS_PARALLELISM=false PYTHONPATH=src $PY -u scripts/train_minimal.py --dataset uonlp/CulturaX --dataset-config nl --tokenizer yhavinga/dutch-llama-tokenizer --use-packing --max-length 1024 --hidden-size 1024 --num-layers 22 --num-heads 16 --intermediate-size 1536 --global-attn-every-n-layers 3 --lr-schedule cosine --min-lr $MIN_LR --weight-decay 0.01 --layer-norm-lr-factor 1.0 --log-every 10 --dataset-num-rows 117392666 --per-device-batch-size 8 --learning-rate $PEAK_LR --warmup-steps $WARMUP --max-steps $STEPS ${STOP_AT:+--stop-at-step $STOP_AT} --masking decay --web-mask-rate $WEB_MASK_RATE --answer-full-max $ANSWER_FULL_MAX --init-from $START_CKPT --checkpoint-every $CHECKPOINT_EVERY --checkpoint-dir $OUT/ckpt --wandb-project $RUN --wandb-run-name $RUN --tracking-sync-dir $OUT/tracking --system-metrics-every $SYSTEM_EVERY --eval-file \$HOME/heldout.parquet --eval-every $EVAL_EVERY --eval-batches $EVAL_BATCHES --data-files-list $DATA_FILES >> ~/logs/$RUN.log 2>&1
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
