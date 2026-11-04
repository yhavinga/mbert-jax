#!/bin/bash
# On every worker of a v4-32 whose CulturaX share is staged by setup_mbert_mask.sh (gs://t5-dutch-english-usc2/mbert/masking/scripts),
# as on mbert-mix: start or resume culturax-nl-230m-chunk1024-mixheads, detached from ssh.
#   COMMIT=<commit of branch mix-heads> bash start_mixheads.sh
# The command of culturax-nl-230m-chunk1024-mix (start_mix.sh there, without its --stop-at-step 100000), except --mix-heads
# rate2, a checkout of its own (~/mbert-mixheads at COMMIT, with the venv of that setup) and the names and paths in the bucket.
# A checkpoint in the bucket makes it a resume; a resume with another --masking or --mix-heads is refused by the data state.
# The first start records COMMIT in the bucket; a start at another commit is refused.
set -euo pipefail
COMMIT=${COMMIT:?set COMMIT to the commit of branch mix-heads}
RUN=culturax-nl-230m-chunk1024-mixheads
OUT=gs://t5-dutch-english-usc2/mbert/$RUN
CODE=$HOME/mbert-mixheads
PY=$HOME/mbert-jax/.venv/bin/python

pgrep -f '[t]rain_minimal.py' > /dev/null && { echo "$(hostname): already running"; exit 1; }
[ -f ~/culturax_nl_files.txt ] && [ -f ~/heldout.parquet ] && [ -x "$PY" ] || { echo "$(hostname): run setup_mbert_mask.sh first"; exit 1; }
[ -d "$CODE/.git" ] || git clone -q https://github.com/yhavinga/mbert-jax.git "$CODE"
git -C "$CODE" fetch -q origin mix-heads
git -C "$CODE" checkout -q "$COMMIT"
if gsutil -q stat "$OUT/start.txt"; then
  RECORDED=$(gsutil cat "$OUT/start.txt")
  [ "$RECORDED" = "$COMMIT" ] || { echo "$(hostname): $RUN started at commit $RECORDED, not $COMMIT"; exit 1; }
else
  echo "$COMMIT" | gsutil -q cp - "$OUT/start.txt"
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
TRACKIO_DIR=\$HOME/trackio TOKENIZERS_PARALLELISM=false PYTHONPATH=src $PY -u scripts/train_minimal.py --dataset uonlp/CulturaX --dataset-config nl --tokenizer yhavinga/dutch-llama-tokenizer --use-packing --max-length 1024 --hidden-size 1024 --num-layers 22 --num-heads 16 --intermediate-size 1536 --global-attn-every-n-layers 3 --lr-schedule cosine --min-lr 1e-6 --weight-decay 0.01 --layer-norm-lr-factor 1.0 --log-every 10 --dataset-num-rows 117392666 --per-device-batch-size 8 --learning-rate 4.8e-4 --warmup-steps 1250 --max-steps 653000 --masking mix --mix-heads rate2 --checkpoint-every 5000 --checkpoint-dir $OUT/ckpt --wandb-project $RUN --wandb-run-name $RUN --tracking-sync-dir $OUT/tracking --system-metrics-every 1000 --eval-file \$HOME/heldout.parquet --eval-every 25000 --eval-batches 50 --data-files-list \$HOME/culturax_nl_files.txt >> ~/logs/$RUN.log 2>&1
echo EXIT \$? > ~/logs/$RUN.exit
EOF
cat > ~/sync_$RUN.sh <<EOF
w=\$(hostname | sed 's/.*-w-/w/')
while [ ! -f ~/logs/$RUN.exit ]; do gsutil -q cp ~/logs/$RUN.log $OUT/logs/\$w.log; sleep 300; done
gsutil -q cp ~/logs/$RUN.log $OUT/logs/\$w.log; gsutil -q cp ~/logs/$RUN.exit $OUT/logs/\$w.exit
EOF
setsid nohup bash ~/run_$RUN.sh > /dev/null 2>&1 < /dev/null &
setsid nohup bash ~/sync_$RUN.sh > /dev/null 2>&1 < /dev/null &
echo "started $RUN on $(hostname) at $COMMIT: ~/logs/$RUN.log, $OUT; process_0 may be any host: read the log of the one with Step lines"
