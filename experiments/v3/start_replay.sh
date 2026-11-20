#!/bin/bash
# On every worker of mbert-mix: start or resume one arm of the replay test of dmbert v3 (protocol.md part 1), detached.
#   bash start_replay.sh <arm>
# arm  zf     v3-replay-zf: --attention-float32 --z-loss 1e-4 --nonfinite-diagnostics
#      bf16   v3-replay-bf16 (control): --nonfinite-diagnostics only, the bf16 path of dmbert v2
#      smoke  v3-replay-smoke-LEG_TAG: as zf
# Every arm is the command of experiments/final-v2/start_v2.sh main (final-v2 @ 43ec0ba: v2's data, schedule and
# settings, bf16) with these flags, --stop-at-step STOP_AT and its own names; its checkpoint directory
# gs://t5-dutch-english-usc2/mbert/v3-replay/<run>/ckpt must hold the checkpoint to resume from (the chain copies v2's
# 250000 there first). PRINT_RUN_SCRIPT=1 prints the trainer command and starts nothing. Environment: COMMIT (required,
# ~/mbert-v3 must be at it), STOP_AT (required), LEG_TAG (smoke only), PY.
set -euo pipefail
ARM=${1:?usage: start_replay.sh zf|bf16|smoke}
COMMIT=${COMMIT:?set COMMIT to the commit of ~/mbert-v3}
STOP=${STOP_AT:?set STOP_AT}
PY=${PY:-\$HOME/mbert-jax/.venv/bin/python}
M=gs://t5-dutch-english-usc2/mbert
case $ARM in
  zf) RUN=v3-replay-zf; FLAGS="--attention-float32 --z-loss 1e-4 --nonfinite-diagnostics" ;;
  bf16) RUN=v3-replay-bf16; FLAGS="--nonfinite-diagnostics" ;;
  smoke) RUN=v3-replay-smoke-${LEG_TAG:?set LEG_TAG}; FLAGS="--attention-float32 --z-loss 1e-4 --nonfinite-diagnostics" ;;
  *) echo "arm must be zf, bf16 or smoke, not $ARM"; exit 1 ;;
esac
OUT=$M/v3-replay/$RUN
LIST=fw2optout_files.txt; EVALF=fw2optout_heldout.parquet
ROWS=$(cat ~/${LIST%.txt}.rows 2>/dev/null || echo NO_ROWS)

run_script() {
  cat <<SCRIPT
cd ~/mbert-v3
TRACKIO_DIR=\$HOME/trackio TOKENIZERS_PARALLELISM=false PYTHONPATH=src $PY -u scripts/train_minimal.py --dataset fineweb2-nl-optout --dataset-config kept-train --tokenizer yhavinga/dutch-llama-tokenizer --use-packing --max-length 1024 --hidden-size 1024 --num-layers 22 --num-heads 16 --intermediate-size 1536 --global-attn-every-n-layers 3 --lr-schedule wsd --decay-steps 266000 --decay-shape 1-sqrt --min-lr 1e-6 --weight-decay 0.01 --layer-norm-lr-factor 1.0 --log-every 10 --dataset-num-rows $ROWS --per-device-batch-size 8 --learning-rate 4.8e-4 --warmup-steps 1250 --max-steps 1306000 --stop-at-step $STOP --skip-nonfinite-updates $FLAGS --url-blocklist hebban.nl --checkpoint-every 10000 --checkpoint-dir $OUT/ckpt --wandb-project $RUN --wandb-run-name $RUN --tracking-sync-dir $OUT/tracking --system-metrics-every 1000 --eval-file \$HOME/$EVALF --eval-every 25000 --eval-batches 50 --data-files-list \$HOME/$LIST >> ~/logs/$RUN.log 2>&1
echo EXIT \$? > ~/logs/$RUN.exit
SCRIPT
}

if [ -n "${PRINT_RUN_SCRIPT:-}" ]; then
  run_script
  exit 0
fi
# Each alternative in brackets, and none whose text holds another's (a pgrep pattern can find itself)
pgrep -f '[t]rain_minimal.py|[f]inetune.py|[f]inetune_extra.py|[e]val_windows.py|[e]val_blimp_nl.py|[m]emtest.py|[v]erify_ckpt.py|[c]onvert_gcs_checkpoint.py' > /dev/null && { echo "$(hostname): the TPU is in use"; exit 1; }
[ "$(git -C ~/mbert-v3 rev-parse HEAD)" = "$COMMIT" ] || { echo "$(hostname): ~/mbert-v3 is not at $COMMIT"; exit 1; }
[ -x "$(eval echo "$PY")" ] || { echo "$(hostname): no $PY"; exit 1; }
[ -s ~/$LIST ] || { echo "$(hostname): no ~/$LIST"; exit 1; }
[[ $ROWS =~ ^[0-9]+$ ]] || { echo "$(hostname): no row count in ~/${LIST%.txt}.rows"; exit 1; }
[ -s ~/$EVALF.md5 ] && [ "$(md5sum ~/$EVALF | cut -d' ' -f1)" = "$(cat ~/$EVALF.md5)" ] || { echo "$(hostname): ~/$EVALF is not the one of ~/$EVALF.md5"; exit 1; }
gsutil -q ls "$OUT/ckpt/*/commit_success.txt" > /dev/null 2>&1 || { echo "$(hostname): no committed checkpoint in $OUT/ckpt to resume from"; exit 1; }
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
echo "started $RUN on $(hostname): ~/logs/$RUN.log, $OUT (to $STOP)"
