#!/bin/bash
# On every worker of a v4-32: start or resume one arm of the shape ablation of dmbert v3 (protocol.md part 2), detached.
#   bash start_ablation.sh <arm>
# arm  a   22 x 1024, GeGLU 1536, 16 heads, global attention every 3rd layer (the shape of T2 and v2)
#      b   29 x 768, GeGLU 2048, 12 heads, every 3rd layer
#      c   16 x 1024, GeGLU 2560, 16 heads, every 3rd layer
#      d   22 x 1024, GeGLU 1536, 16 heads, every 2nd layer
#      a2  a with --seed 43 (another initialization, data order and masking: the pretraining noise of a)
#      smoke-<arm>-<tag>  that arm's command to STOP_AT (default 200) under its own name, a checkpoint at the end only
# T2's recipe (experiments/final/start_final.sh culturax stable at f958d01: CulturaX nl, 15% single tokens, 1,024 tokens,
# packing, --url-blocklist hebban.nl, batch 128, AdamW, peak 4.8e-4 after 1,250 warmup steps, WSD with a 1-sqrt cooldown
# to 1e-6, held-out eval on ~/heldout.parquet) over 100,000 steps with a cooldown of 20,000, on the files of
# stage_ablation.sh (~/v3abl_files.txt), with tokenizer (a) (~/mbert-v3-tokenizer, build_tokenizer_v3.py) and v3's
# stability flags (--attention-float32 --z-loss 1e-4 --nonfinite-diagnostics, protocol.md part 1); PRECISION=float32
# instead runs the whole model in float32 without those flags (only on Yeb's word, if the replay test did not pass).
# --dataset-num-rows is ~/v3abl_files.total, the documents of the 4 workers' files (for the logged passes only).
# Run directory gs://t5-dutch-english-usc2/mbert/v3-ablation/<run>/. PRINT_RUN_SCRIPT=1 prints the trainer command.
# Environment: COMMIT (required, ~/mbert-v3 must be at it), STOP_AT, PRECISION (fix|float32, default fix), PY.
set -euo pipefail
ARM=${1:?usage: start_ablation.sh a|b|c|d|a2|smoke-<arm>-<tag>}
COMMIT=${COMMIT:?set COMMIT to the commit of ~/mbert-v3}
PY=${PY:-\$HOME/mbert-jax/.venv/bin/python}
PRECISION=${PRECISION:-fix}
M=gs://t5-dutch-english-usc2/mbert
shape() {
  case $1 in
    a|a2|d) echo "--hidden-size 1024 --num-layers 22 --num-heads 16 --intermediate-size 1536 --global-attn-every-n-layers $([ "$1" = d ] && echo 2 || echo 3)" ;;
    b) echo "--hidden-size 768 --num-layers 29 --num-heads 12 --intermediate-size 2048 --global-attn-every-n-layers 3" ;;
    c) echo "--hidden-size 1024 --num-layers 16 --num-heads 16 --intermediate-size 2560 --global-attn-every-n-layers 3" ;;
    *) return 1 ;;
  esac
}
CHECKPOINT_EVERY=10000; EVAL_EVERY=25000; STOP=${STOP_AT:-}
case $ARM in
  a|b|c|d|a2) BASE_ARM=$ARM; RUN=v3-abl-$ARM ;;
  smoke-*-*) BASE_ARM=$(echo "$ARM" | cut -d- -f2); RUN=v3-abl-$ARM; STOP=${STOP_AT:-200}; CHECKPOINT_EVERY=100000; EVAL_EVERY=200 ;;
  *) echo "arm must be a, b, c, d, a2 or smoke-<arm>-<tag>, not $ARM"; exit 1 ;;
esac
SHAPE=$(shape "$BASE_ARM") || { echo "no shape for $BASE_ARM"; exit 1; }
SEED=; [ "$BASE_ARM" = a2 ] && SEED=" --seed 43"
case $PRECISION in
  fix) FLAGS="--attention-float32 --z-loss 1e-4 --nonfinite-diagnostics" ;;
  float32) FLAGS="--compute-dtype float32 --nonfinite-diagnostics" ;;
  *) echo "PRECISION must be fix or float32, not $PRECISION"; exit 1 ;;
esac
OUT=$M/v3-ablation/$RUN
ROWS=$(cat ~/v3abl_files.total 2>/dev/null || echo NO_ROWS)

run_script() {
  cat <<SCRIPT
cd ~/mbert-v3
TRACKIO_DIR=\$HOME/trackio TOKENIZERS_PARALLELISM=false PYTHONPATH=src $PY -u scripts/train_minimal.py --dataset uonlp/CulturaX --dataset-config nl --tokenizer \$HOME/mbert-v3-tokenizer --use-packing --max-length 1024 $SHAPE --lr-schedule wsd --decay-steps 20000 --decay-shape 1-sqrt --min-lr 1e-6 --weight-decay 0.01 --layer-norm-lr-factor 1.0 --log-every 10 --dataset-num-rows $ROWS --per-device-batch-size 8 --learning-rate 4.8e-4 --warmup-steps 1250 --max-steps 100000${STOP:+ --stop-at-step $STOP} --skip-nonfinite-updates $FLAGS$SEED --url-blocklist hebban.nl --checkpoint-every $CHECKPOINT_EVERY --checkpoint-dir $OUT/ckpt --wandb-project $RUN --wandb-run-name $RUN --tracking-sync-dir $OUT/tracking --system-metrics-every 1000 --eval-file \$HOME/heldout.parquet --eval-every $EVAL_EVERY --eval-batches 50 --data-files-list \$HOME/v3abl_files.txt >> ~/logs/$RUN.log 2>&1
echo EXIT \$? > ~/logs/$RUN.exit
SCRIPT
}

if [ -n "${PRINT_RUN_SCRIPT:-}" ]; then
  run_script
  exit 0
fi
pgrep -f '[t]rain_minimal.py|[f]inetune.py|[f]inetune_extra.py|[e]val_windows.py|[e]val_blimp_nl.py|[m]emtest.py|[v]erify_ckpt.py|[c]onvert_gcs_checkpoint.py' > /dev/null && { echo "$(hostname): the TPU is in use"; exit 1; }
pgrep -f '[s]tage_ablation.sh' > /dev/null && { echo "$(hostname): staging still runs"; exit 1; }
[ "$(git -C ~/mbert-v3 rev-parse HEAD)" = "$COMMIT" ] || { echo "$(hostname): ~/mbert-v3 is not at $COMMIT"; exit 1; }
[ -x "$(eval echo "$PY")" ] || { echo "$(hostname): no $PY"; exit 1; }
[ -s ~/mbert-v3-tokenizer/tokenizer.json ] || { echo "$(hostname): no ~/mbert-v3-tokenizer"; exit 1; }
[ -s ~/v3abl_files.txt ] || { echo "$(hostname): no ~/v3abl_files.txt (stage_ablation.sh)"; exit 1; }
while read -r f; do [ -s "$f" ] || { echo "$(hostname): ~/v3abl_files.txt names $f, which is missing or empty"; exit 1; }; done < ~/v3abl_files.txt
[[ $ROWS =~ ^[0-9]+$ ]] || { echo "$(hostname): no row count in ~/v3abl_files.total (the chain writes the sum over the workers)"; exit 1; }
[ "$(md5sum ~/heldout.parquet | cut -d' ' -f1)" = b2f579a876d06eef40c94270537bb7af ] || { echo "$(hostname): ~/heldout.parquet is not T2's"; exit 1; }
pkill -f "[s]ync_$RUN.sh" || true
mkdir -p ~/logs ~/trackio
rm -f ~/logs/$RUN.exit
if gsutil -q stat "$OUT/tracking/$RUN.db"; then
  rm -f ~/trackio/$RUN.db ~/trackio/$RUN.db-wal ~/trackio/$RUN.db-shm
  gsutil -q cp "$OUT/tracking/$RUN.db" ~/trackio/$RUN.db
fi
run_script > ~/run_$RUN.sh
cat > ~/sync_$RUN.sh <<SYNC
w=\$(hostname | sed 's/.*-w-/w/')
while [ ! -f ~/logs/$RUN.exit ]; do gsutil -q cp ~/logs/$RUN.log $OUT/logs/\$w.log; sleep 300; done
gsutil -q cp ~/logs/$RUN.log $OUT/logs/\$w.log; gsutil -q cp ~/logs/$RUN.exit $OUT/logs/\$w.exit
SYNC
setsid nohup bash ~/run_$RUN.sh > /dev/null 2>&1 < /dev/null &
setsid nohup bash ~/sync_$RUN.sh > /dev/null 2>&1 < /dev/null &
echo "started $RUN on $(hostname): ~/logs/$RUN.log, $OUT${STOP:+ (to $STOP)}"
