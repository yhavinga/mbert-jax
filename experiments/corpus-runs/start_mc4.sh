#!/bin/bash
# On every worker of the v4-32 of the mC4 run (after setup_mc4.sh): start or resume mc4-nl-230m-chunk1024-mix, detached from ssh.
# The command of culturax-nl-230m-chunk1024-mix (start_mix.sh in gs://t5-dutch-english-usc2/mbert/masking/scripts, without
# its --stop-at-step) on mC4 nl cleaned: only --dataset, --dataset-config, --dataset-num-rows, --data-files-list, the code
# directory (~/mbert-corpus, with its own venv) and the names and paths in the bucket differ. A checkpoint in the bucket
# makes it a resume; one of another data setup is refused.
# 653,000 steps are 2.21 passes over mC4 nl (295,506 steps per pass, gs://t5-dutch-english-usc2/mbert/token-counts/summary.json).
# Each host starts its next pass on its own when its share runs out, with seed 42 + the pass number for the order of its
# files, the shuffle buffer and the window of pieces; the hosts' passes end at other steps, as their shares differ by a
# document and in tokens by 0.13%.
# MASKING: mix (default) or uniform15, the masking of culturax-nl-230m-chunk1024; with uniform15 the run is
# mc4-nl-230m-chunk1024, without the -mix suffix. Mix lost to 15% on SICK-NL at step 653,000 (80.76 against 84.90).
# Environment overrides, for the smoke test only: RUN_ROOT RUN STOP_AT CHECKPOINT_EVERY EVAL_EVERY.
# Past a NaN loss in bf16, which a resume from the same checkpoint repeats at the same step (mask30, 2026-11-02):
# COMPUTE_DTYPE=float32 with STOP_AT the next checkpoint step, then a resume without both.
set -euo pipefail
MASKING=${MASKING:-mix}
case $MASKING in mix) SUFFIX=-mix ;; uniform15) SUFFIX= ;; *) echo "MASKING must be mix or uniform15, not $MASKING"; exit 1 ;; esac
RUN_ROOT=${RUN_ROOT:-gs://t5-dutch-english-usc2/mbert}
RUN=${RUN:-mc4-nl-230m-chunk1024$SUFFIX}
CHECKPOINT_EVERY=${CHECKPOINT_EVERY:-5000}
EVAL_EVERY=${EVAL_EVERY:-25000}
STOP_AT=${STOP_AT:-}
COMPUTE_DTYPE=${COMPUTE_DTYPE:-}
OUT=$RUN_ROOT/$RUN
pgrep -f '[t]rain_minimal.py|[f]inetune.py|[e]val_windows.py' > /dev/null && { echo "$(hostname): the TPU is in use"; exit 1; }
[ -f ~/mc4_nl_files.txt ] && [ -f ~/heldout.parquet ] && [ -x ~/mbert-corpus/.venv/bin/python ] || { echo "$(hostname): run setup_mc4.sh first"; exit 1; }
pkill -f "[s]ync_$RUN.sh" || true
mkdir -p ~/logs ~/trackio
rm -f ~/logs/$RUN.exit
if gsutil -q stat "$OUT/tracking/$RUN.db"; then
  rm -f ~/trackio/$RUN.db ~/trackio/$RUN.db-wal ~/trackio/$RUN.db-shm
  gsutil -q cp "$OUT/tracking/$RUN.db" ~/trackio/$RUN.db
fi
cat > ~/run_$RUN.sh <<EOF
cd ~/mbert-corpus
TRACKIO_DIR=\$HOME/trackio TOKENIZERS_PARALLELISM=false PYTHONPATH=src .venv/bin/python -u scripts/train_minimal.py --dataset yhavinga/mc4_nl_cleaned --dataset-config full --tokenizer yhavinga/dutch-llama-tokenizer --use-packing --max-length 1024 --hidden-size 1024 --num-layers 22 --num-heads 16 --intermediate-size 1536 --global-attn-every-n-layers 3 --lr-schedule cosine --min-lr 1e-6 --weight-decay 0.01 --layer-norm-lr-factor 1.0 --log-every 10 --dataset-num-rows 64545895 --per-device-batch-size 8 --learning-rate 4.8e-4 --warmup-steps 1250 --max-steps 653000 ${STOP_AT:+--stop-at-step $STOP_AT} ${COMPUTE_DTYPE:+--compute-dtype $COMPUTE_DTYPE} --masking $MASKING --checkpoint-every $CHECKPOINT_EVERY --checkpoint-dir $OUT/ckpt --wandb-project $RUN --wandb-run-name $RUN --tracking-sync-dir $OUT/tracking --system-metrics-every 1000 --eval-file \$HOME/heldout.parquet --eval-every $EVAL_EVERY --eval-batches 50 --data-files-list \$HOME/mc4_nl_files.txt >> ~/logs/$RUN.log 2>&1
echo EXIT \$? > ~/logs/$RUN.exit
EOF
cat > ~/sync_$RUN.sh <<EOF
w=\$(hostname | sed 's/.*-w-/w/')
while [ ! -f ~/logs/$RUN.exit ]; do gsutil -q cp ~/logs/$RUN.log $OUT/logs/\$w.log; sleep 300; done
gsutil -q cp ~/logs/$RUN.log $OUT/logs/\$w.log; gsutil -q cp ~/logs/$RUN.exit $OUT/logs/\$w.exit
EOF
setsid nohup bash ~/run_$RUN.sh > /dev/null 2>&1 < /dev/null &
setsid nohup bash ~/sync_$RUN.sh > /dev/null 2>&1 < /dev/null &
echo "started $RUN on $(hostname): ~/logs/$RUN.log, $OUT; process_0 may be any host: read the log of the one with Step lines"
