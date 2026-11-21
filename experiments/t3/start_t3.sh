#!/bin/bash
# On every worker of the v4-32 of T3 (mbert-corpus): start or resume one leg of T3, detached from ssh
# (experiments/t3/protocol.md, sections 6 to 8).
#   bash start_t3.sh <leg>
# leg  main    t3-culturax-317m: the WSD schedule of the main leg, 1,959,000 steps with a 1-sqrt cooldown over the last
#              399,000 (from 1,560,000) to 1e-6; STOP_AT the step to stop at (1,040,000 for the branch, or the end of a
#              float32 piece), none to the end
#      branch  t3-culturax-317m-epoch2: the cooldown branch after two epochs, from a copy of the main run's checkpoint
#              1,040,000 and its data state (t3_run.sh copies them into the branch's ckpt/ and ckpt_data_state/) to
#              1,306,000 with T2's tak 2 schedule (1-sqrt over the last 266,000 steps); STOP_AT only for a float32 piece
#      smoke   the main leg's command as t3-culturax-317m-smoke-LEG_TAG to STOP_AT (default 300), a checkpoint every
#              CHECKPOINT_EVERY (default 100), eval at 200, system metrics every 100; a second start resumes it
# The command is T2's (experiments/final/start_final.sh culturax stable at 1c8f2c7: CulturaX nl, 15% single tokens,
# 1,024 tokens with packing, --url-blocklist hebban.nl, batch 8 per chip, AdamW, peak 4.8e-4 after 1,250 warmup steps,
# WSD 1-sqrt to 1e-6, eval on ~/heldout.parquet every 25,000 steps) with T3's shape (18 x 1152, 18 heads, GeGLU 2944,
# global attention every 3rd layer), tokenizer (a) (~/mbert-t3-tokenizer), v3's fix (--attention-float32 --z-loss 1e-4,
# and --nonfinite-diagnostics), T3's data (~/t3_files.txt of prepare_t3.sh data: T2's files without the held-out) and
# its names. PRINT_RUN_SCRIPT=1 prints the trainer's script and starts nothing. A checkpoint in the run's bucket
# directory makes it a resume (also after a preemption); one of another data setup is refused by the data state.
# Environment: COMMIT (required, ~/mbert-t3 must be at it), PY, STOP_AT, LEG_TAG (smoke), CHECKPOINT_EVERY (smoke),
# COMPUTE_DTYPE (float32 for a piece after a burst, protocol.md section 7; the fix and the z-loss stay), M.
set -euo pipefail
LEG=${1:?usage: start_t3.sh main or branch or smoke}
COMMIT=${COMMIT:?set COMMIT to the commit of branch t3}
PY=${PY:-\$HOME/mbert-jax/.venv/bin/python}
COMPUTE_DTYPE=${COMPUTE_DTYPE:-}
case $COMPUTE_DTYPE in ""|float32|bfloat16) ;; *) echo "COMPUTE_DTYPE must be float32 or bfloat16, not $COMPUTE_DTYPE"; exit 1 ;; esac
M=${M:-gs://t5-dutch-english-usc2/mbert}
T=$M/t3
BASE=t3-culturax-317m
SHAPE="--hidden-size 1152 --num-layers 18 --num-heads 18 --intermediate-size 2944 --global-attn-every-n-layers 3"
FIX="--attention-float32 --z-loss 1e-4 --nonfinite-diagnostics"
# The held-out of T3 (protocol.md section 5); HELDOUT_MD5 is set only by the dry run
HELDOUT_MD5=${HELDOUT_MD5:-b2f579a876d06eef40c94270537bb7af}
# sha256 of tokenizer.json of tokenizer (a) (scripts/build_tokenizer_v3.py, experiments/v3/protocol.md appendix)
TOKENIZER_SHA256=860167d4618eac7a7ce74c1646bda60f348c40efcb528c1cb000f4f3c200ed02
MAX=1959000; DECAY=399000; CHECKPOINT_EVERY_LEG=10000; EVAL_EVERY=25000; SYSTEM_EVERY=1000; BRANCH_STEP=
case $LEG in
  main) RUN=$BASE; STOP=${STOP_AT:-} ;;
  branch) RUN=$BASE-epoch2; MAX=1306000; DECAY=266000; STOP=${STOP_AT:-}; BRANCH_STEP=1040000 ;;
  smoke) RUN=$BASE-smoke-${LEG_TAG:?set LEG_TAG for a smoke run}; STOP=${STOP_AT:-300}
    CHECKPOINT_EVERY_LEG=${CHECKPOINT_EVERY:-100}; EVAL_EVERY=200; SYSTEM_EVERY=100 ;;
  *) echo "leg must be main, branch or smoke, not $LEG"; exit 1 ;;
esac
OUT=$T/$RUN
ROWS=$(cat ~/t3_files.total 2>/dev/null || echo NO_ROWS)

run_script() {
  cat <<EOF
cd ~/mbert-t3
TRACKIO_DIR=\$HOME/trackio TOKENIZERS_PARALLELISM=false PYTHONPATH=src $PY -u scripts/train_minimal.py --dataset uonlp/CulturaX --dataset-config nl --tokenizer \$HOME/mbert-t3-tokenizer --use-packing --max-length 1024 $SHAPE --lr-schedule wsd --decay-steps $DECAY --decay-shape 1-sqrt --min-lr 1e-6 --weight-decay 0.01 --layer-norm-lr-factor 1.0 --log-every 10 --dataset-num-rows $ROWS --per-device-batch-size 8 --learning-rate 4.8e-4 --warmup-steps 1250 --max-steps $MAX${STOP:+ --stop-at-step $STOP}${COMPUTE_DTYPE:+ --compute-dtype $COMPUTE_DTYPE} --skip-nonfinite-updates $FIX --url-blocklist hebban.nl --checkpoint-every $CHECKPOINT_EVERY_LEG --checkpoint-dir $OUT/ckpt --wandb-project $RUN --wandb-run-name $RUN --tracking-sync-dir $OUT/tracking --system-metrics-every $SYSTEM_EVERY --eval-file \$HOME/heldout.parquet --eval-every $EVAL_EVERY --eval-batches 50 --data-files-list \$HOME/t3_files.txt >> ~/logs/$RUN.log 2>&1
echo EXIT \$? > ~/logs/$RUN.exit
EOF
}

if [ -n "${PRINT_RUN_SCRIPT:-}" ]; then
  run_script
  exit 0
fi
# Each alternative in brackets, and none whose text holds another's (LEARNINGS, 14 Nov: a pgrep pattern can find itself)
pgrep -f '[t]rain_minimal.py|[f]inetune.py|[f]inetune_extra.py|[e]val_windows.py|[e]val_blimp_nl.py|[m]emtest.py|[v]erify_ckpt.py|[c]onvert_gcs_checkpoint.py|[t]ranslate.py|[s]ft_lora.py' > /dev/null && { echo "$(hostname): the TPU is in use"; exit 1; }
holders=$(sudo lsof -t /dev/accel* /dev/vfio/* 2>/dev/null | sort -u | tr '\n' ' ' || true)
[ -z "${holders// /}" ] || { echo "$(hostname): the TPU is held by: $holders"; exit 1; }
pgrep -f 'logs/t3_data[.]log 2>&1; echo EXIT' > /dev/null && { echo "$(hostname): prepare_t3.sh data still runs"; exit 1; }
[ "$(git -C ~/mbert-t3 rev-parse HEAD)" = "$COMMIT" ] || { echo "$(hostname): ~/mbert-t3 is not at $COMMIT"; exit 1; }
[ -x "$(eval echo "$PY")" ] || { echo "$(hostname): no $PY"; exit 1; }
[ "$(sha256sum ~/mbert-t3-tokenizer/tokenizer.json 2>/dev/null | cut -d' ' -f1)" = $TOKENIZER_SHA256 ] || { echo "$(hostname): ~/mbert-t3-tokenizer is not tokenizer (a)"; exit 1; }
[ -s ~/t3_files.txt ] && [ "$(wc -l < ~/t3_files.txt)" = 32 ] || { echo "$(hostname): ~/t3_files.txt does not list 32 files"; exit 1; }
[ "$(md5sum ~/t3_files.txt | cut -d' ' -f1)" = "$(cat ~/t3_files.md5 2>/dev/null)" ] || { echo "$(hostname): ~/t3_files.txt is not the list of the data step (~/t3_files.md5)"; exit 1; }
while read -r f; do [ -s "$f" ] || { echo "$(hostname): ~/t3_files.txt names $f, which is missing or empty"; exit 1; }; done < ~/t3_files.txt
[[ $ROWS =~ ^[0-9]+$ ]] || { echo "$(hostname): no row count in ~/t3_files.total (the chain writes it)"; exit 1; }
[ "$(md5sum ~/heldout.parquet 2>/dev/null | cut -d' ' -f1)" = $HELDOUT_MD5 ] || { echo "$(hostname): ~/heldout.parquet is not the held-out of T3"; exit 1; }
if [ -n "$BRANCH_STEP" ]; then
  # Without the copied checkpoint this would train from scratch under the branch's name
  gsutil -q stat "$OUT/ckpt_data_state/$BRANCH_STEP/process_0.json" && gsutil ls "$OUT/ckpt/" | grep -q '/[0-9]*/$' \
    || { echo "$(hostname): no checkpoint $BRANCH_STEP with data state in $OUT (t3_run.sh copies it)"; exit 1; }
fi
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
echo "started $RUN on $(hostname): ~/logs/$RUN.log, $OUT${STOP:+ (to $STOP)}${COMPUTE_DTYPE:+ in $COMPUTE_DTYPE}"
