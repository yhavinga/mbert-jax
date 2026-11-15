#!/bin/bash
# On every worker of the v4-32 (mbert-mix): start or resume one leg of dmbert v2, detached from ssh (protocol.md).
#   bash start_v2.sh <leg>
# leg  main       v2-fw2optout-230m: the WSD schedule of T2 in one leg, 1,306,000 steps with a 1-sqrt cooldown over
#                 the last 266,000 (from 1,040,000), to the end; STOP_AT only for a float32 piece past a self-stop
#      smoke      the main leg's command as v2-fw2optout-230m-smoke[-LEG_TAG] on a few staged files (~/fw2optout_smoke_files.txt,
#                 held-out ~/fw2optout_smoke_heldout.parquet) to STOP_AT (default 300), a checkpoint every
#                 CHECKPOINT_EVERY (default 100), eval at 200, system metrics every 100; a second start resumes it
#      preflight  the main leg's command as v2-fw2optout-230m-preflight[-LEG_TAG] on the final staged data to step 200, a
#                 checkpoint at the end only, eval at 200, system metrics every 100
# Every leg is the command of T2's main run (experiments/final/start_final.sh culturax stable at 1c8f2c7) with this
# data: --data-files-list ~/fw2optout_files.txt (the share of kept/train of stage_optout.py, columns text and url) with
# --url-blocklist hebban.nl, --eval-file ~/fw2optout_heldout.parquet (kept/test less what is also in kept/train),
# --dataset-num-rows of ~/fw2optout_files.rows (kept/train less hebban.nl, for the metric data/passes only), the
# dataset's name, --max-steps 1306000 without --stop-at-step, and its names; PRINT_RUN_SCRIPT=1 prints it and starts
# nothing. A checkpoint in the run's bucket directory makes it a resume (also after a preemption); one of another data
# setup is refused by the data state. Environment: COMMIT (required, ~/mbert-v2 must be at it), PY, STOP_AT, LEG_TAG
# (smoke and preflight only: a name of their own per start of the chain, so none resumes an earlier one's checkpoints),
# COMPUTE_DTYPE (float32 or bfloat16; past a self-stop on non-finite gradients in bf16: float32 to the next checkpoint
# step, then a resume without it, protocol.md "Precisie").
set -euo pipefail
LEG=${1:?usage: start_v2.sh main|smoke|preflight}
COMMIT=${COMMIT:?set COMMIT to the commit of experiments/final-v2}
PY=${PY:-\$HOME/mbert-jax/.venv/bin/python}
COMPUTE_DTYPE=${COMPUTE_DTYPE:-}
case $COMPUTE_DTYPE in ""|float32|bfloat16) ;; *) echo "COMPUTE_DTYPE must be float32 or bfloat16, not $COMPUTE_DTYPE"; exit 1 ;; esac
M=gs://t5-dutch-english-usc2/mbert
BASE=v2-fw2optout-230m
LIST=fw2optout_files.txt; EVALF=fw2optout_heldout.parquet
MAX=1306000; DECAY=266000; CHECKPOINT_EVERY_LEG=10000; EVAL_EVERY=25000; SYSTEM_EVERY=1000
case $LEG in
  main) RUN=$BASE; STOP=${STOP_AT:-} ;;
  smoke) RUN=$BASE-smoke${LEG_TAG:+-$LEG_TAG}; LIST=fw2optout_smoke_files.txt; EVALF=fw2optout_smoke_heldout.parquet; STOP=${STOP_AT:-300}
    CHECKPOINT_EVERY_LEG=${CHECKPOINT_EVERY:-100}; EVAL_EVERY=200; SYSTEM_EVERY=100 ;;
  preflight) RUN=$BASE-preflight${LEG_TAG:+-$LEG_TAG}; STOP=200; CHECKPOINT_EVERY_LEG=100000; EVAL_EVERY=200; SYSTEM_EVERY=100 ;;
  *) echo "leg must be main, smoke or preflight, not $LEG"; exit 1 ;;
esac
OUT=$M/$RUN
ROWS=$(cat ~/${LIST%.txt}.rows 2>/dev/null || echo NO_ROWS)

run_script() {
  cat <<EOF
cd ~/mbert-v2
TRACKIO_DIR=\$HOME/trackio TOKENIZERS_PARALLELISM=false PYTHONPATH=src $PY -u scripts/train_minimal.py --dataset fineweb2-nl-optout --dataset-config kept-train --tokenizer yhavinga/dutch-llama-tokenizer --use-packing --max-length 1024 --hidden-size 1024 --num-layers 22 --num-heads 16 --intermediate-size 1536 --global-attn-every-n-layers 3 --lr-schedule wsd --decay-steps $DECAY --decay-shape 1-sqrt --min-lr 1e-6 --weight-decay 0.01 --layer-norm-lr-factor 1.0 --log-every 10 --dataset-num-rows $ROWS --per-device-batch-size 8 --learning-rate 4.8e-4 --warmup-steps 1250 --max-steps $MAX${STOP:+ --stop-at-step $STOP}${COMPUTE_DTYPE:+ --compute-dtype $COMPUTE_DTYPE} --skip-nonfinite-updates --url-blocklist hebban.nl --checkpoint-every $CHECKPOINT_EVERY_LEG --checkpoint-dir $OUT/ckpt --wandb-project $RUN --wandb-run-name $RUN --tracking-sync-dir $OUT/tracking --system-metrics-every $SYSTEM_EVERY --eval-file \$HOME/$EVALF --eval-every $EVAL_EVERY --eval-batches 50 --data-files-list \$HOME/$LIST >> ~/logs/$RUN.log 2>&1
echo EXIT \$? > ~/logs/$RUN.exit
EOF
}

if [ -n "${PRINT_RUN_SCRIPT:-}" ]; then
  run_script
  exit 0
fi
# Each alternative in brackets, and none whose text holds another's (LEARNINGS, 14 Nov: a pgrep pattern can find itself)
pgrep -f '[t]rain_minimal.py|[f]inetune.py|[f]inetune_extra.py|[e]val_windows.py|[e]val_blimp_nl.py|[m]emtest.py|[v]erify_ckpt.py|[c]onvert_gcs_checkpoint.py' > /dev/null && { echo "$(hostname): the TPU is in use"; exit 1; }
pgrep -f '[s]tage_optout.py|[b]locklist_scan.py' > /dev/null && { echo "$(hostname): a data step still runs"; exit 1; }
[ "$(git -C ~/mbert-v2 rev-parse HEAD)" = "$COMMIT" ] || { echo "$(hostname): ~/mbert-v2 is not at $COMMIT"; exit 1; }
[ -x "$(eval echo "$PY")" ] || { echo "$(hostname): no $PY"; exit 1; }
[ -s ~/$LIST ] || { echo "$(hostname): no ~/$LIST"; exit 1; }
while read -r f; do [ -s "$f" ] || { echo "$(hostname): ~/$LIST names $f, which is missing or empty"; exit 1; }; done < ~/$LIST
[[ $ROWS =~ ^[0-9]+$ ]] || { echo "$(hostname): no row count in ~/${LIST%.txt}.rows"; exit 1; }
[ -s ~/$EVALF.md5 ] && [ "$(md5sum ~/$EVALF | cut -d' ' -f1)" = "$(cat ~/$EVALF.md5)" ] || { echo "$(hostname): ~/$EVALF is not the one of ~/$EVALF.md5"; exit 1; }
pkill -f "[s]ync_$RUN.sh" || true
mkdir -p ~/logs ~/trackio
rm -f ~/logs/$RUN.exit
if gsutil -q stat "$OUT/tracking/$RUN.db"; then
  rm -f ~/trackio/$RUN.db ~/trackio/$RUN.db-wal ~/trackio/$RUN.db-shm
  gsutil -q cp "$OUT/tracking/$RUN.db" ~/trackio/$RUN.db
fi
run_script > ~/run_$RUN.sh
cat > ~/sync_$RUN.sh <<EOF
w=\$(hostname | sed 's/.*-w-/w/')
while [ ! -f ~/logs/$RUN.exit ]; do gsutil -q cp ~/logs/$RUN.log $OUT/logs/\$w.log; sleep 300; done
gsutil -q cp ~/logs/$RUN.log $OUT/logs/\$w.log; gsutil -q cp ~/logs/$RUN.exit $OUT/logs/\$w.exit
EOF
setsid nohup bash ~/run_$RUN.sh > /dev/null 2>&1 < /dev/null &
setsid nohup bash ~/sync_$RUN.sh > /dev/null 2>&1 < /dev/null &
echo "started $RUN on $(hostname): ~/logs/$RUN.log, $OUT; process_0 may be any host: read the log of the one with Step lines"
