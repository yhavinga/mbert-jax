#!/bin/bash
# On every worker of mbert-corpus: start or resume one leg of the final run, detached from ssh (protocol.md).
#   bash start_final.sh <variant> <leg>
# variant  culturax        CulturaX nl, S's staged files (~/culturax_nl_files.txt), --url-blocklist hebban.nl
#          culturax+gptnl  70% CulturaX nl + 30% GPT-NL in tokens (~/final_cxgptnl_files.txt of prepare_final.sh mix,
#                          CulturaX's hebban.nl rows left out by the plan, so no --url-blocklist: the files have no url)
# leg      stable  the main run <run>: the WSD schedule of tak 2 (1,306,000 steps, cooldown of 266,000), stopped at
#                  STOP_AT, by default 1,040,000 (the start of tak 2's cooldown; the chain starts it with 520,000)
#          smoke   the stable leg's command as <run>-smoke to STOP_AT=200, checkpoint at the end only, eval at 200
#          epoch1  tak 1, <run>-epoch1: cooldown of 133,000 steps to 653,000 (one epoch), from the main run's
#                  checkpoint 520,000, which start_branch.sh copied into its ckpt/ with its data state
#          epoch2  tak 2, <run>-epoch2: cooldown of 266,000 steps to 1,306,000 (two epochs), from 1,040,000
# Every leg is the command of culturax-nl-230m-chunk1024 (S, run_culturax-nl-230m-chunk1024.sh in its scripts/) with
# --lr-schedule wsd --decay-steps --decay-shape 1-sqrt instead of cosine, --skip-nonfinite-updates, the data above, its
# --max-steps and --stop-at-step, and its names; PRINT_RUN_SCRIPT=1 prints it without starting anything.
# A checkpoint in the run's bucket directory makes it a resume (also after a preemption); one of another data setup
# is refused by the data state. Environment: COMMIT (required, ~/mbert-final must be at it), PY, STOP_AT, COMPUTE_DTYPE.
# Past a self-stop on non-finite gradients in bf16 (RuntimeError: n steps in a row ...), which a resume from the same
# checkpoint meets again on the same data: COMPUTE_DTYPE=float32 with STOP_AT the next checkpoint step, then a resume
# without it (as start_mc4.sh).
set -euo pipefail
VARIANT=${1:?usage: start_final.sh culturax|culturax+gptnl stable|smoke|epoch1|epoch2}
LEG=${2:?usage: start_final.sh culturax|culturax+gptnl stable|smoke|epoch1|epoch2}
COMMIT=${COMMIT:?set COMMIT to the commit of commands.txt}
PY=${PY:-\$HOME/mbert-jax/.venv/bin/python}
COMPUTE_DTYPE=${COMPUTE_DTYPE:-}
case $COMPUTE_DTYPE in ""|float32|bfloat16) ;; *) echo "COMPUTE_DTYPE must be float32 or bfloat16, not $COMPUTE_DTYPE"; exit 1 ;; esac
M=gs://t5-dutch-english-usc2/mbert
case $VARIANT in
  culturax) BASE=final-culturax-230m; LIST=culturax_nl_files.txt; NFILES=32; BLOCK=" --url-blocklist hebban.nl"
    # 117,392,666 documents of CulturaX nl less the 45,781 of www.hebban.nl (the blocklist scans of 9 Nov)
    ROWS=117346885 ;;
  culturax+gptnl) BASE=final-cxgptnl-230m; LIST=final_cxgptnl_files.txt; NFILES=128; BLOCK=
    PLAN=$HOME/mbert-final/experiments/final/final_plan.json
    ROWS=$( [ -f "$PLAN" ] && python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['dataset_num_rows'])" "$PLAN" || echo PLAN_ROWS) ;;
  *) echo "variant must be culturax or culturax+gptnl, not $VARIANT"; exit 1 ;;
esac
CHECKPOINT_EVERY=10000; EVAL_EVERY=25000; BRANCH_STEP=
case $LEG in
  stable) RUN=$BASE; MAX=1306000; DECAY=266000; STOP=${STOP_AT:-1040000} ;;
  smoke) RUN=$BASE-smoke; MAX=1306000; DECAY=266000; STOP=${STOP_AT:-200}; CHECKPOINT_EVERY=100000; EVAL_EVERY=200 ;;
  epoch1) RUN=$BASE-epoch1; MAX=653000; DECAY=133000; STOP=; BRANCH_STEP=520000 ;;
  epoch2) RUN=$BASE-epoch2; MAX=1306000; DECAY=266000; STOP=; BRANCH_STEP=1040000 ;;
  *) echo "leg must be stable, smoke, epoch1 or epoch2, not $LEG"; exit 1 ;;
esac
OUT=$M/$RUN

run_script() {
  cat <<EOF
cd ~/mbert-final
TRACKIO_DIR=\$HOME/trackio TOKENIZERS_PARALLELISM=false PYTHONPATH=src $PY -u scripts/train_minimal.py --dataset uonlp/CulturaX --dataset-config nl --tokenizer yhavinga/dutch-llama-tokenizer --use-packing --max-length 1024 --hidden-size 1024 --num-layers 22 --num-heads 16 --intermediate-size 1536 --global-attn-every-n-layers 3 --lr-schedule wsd --decay-steps $DECAY --decay-shape 1-sqrt --min-lr 1e-6 --weight-decay 0.01 --layer-norm-lr-factor 1.0 --log-every 10 --dataset-num-rows $ROWS --per-device-batch-size 8 --learning-rate 4.8e-4 --warmup-steps 1250 --max-steps $MAX${STOP:+ --stop-at-step $STOP}${COMPUTE_DTYPE:+ --compute-dtype $COMPUTE_DTYPE} --skip-nonfinite-updates${BLOCK} --checkpoint-every $CHECKPOINT_EVERY --checkpoint-dir $OUT/ckpt --wandb-project $RUN --wandb-run-name $RUN --tracking-sync-dir $OUT/tracking --system-metrics-every 1000 --eval-file \$HOME/heldout.parquet --eval-every $EVAL_EVERY --eval-batches 50 --data-files-list \$HOME/$LIST >> ~/logs/$RUN.log 2>&1
echo EXIT \$? > ~/logs/$RUN.exit
EOF
}

if [ -n "${PRINT_RUN_SCRIPT:-}" ]; then
  run_script
  exit 0
fi
pgrep -f '[t]rain_minimal.py|[f]inetune.py|[e]val_windows.py|[v]erify_ckpt.py|[c]onvert_gcs_checkpoint.py' > /dev/null && { echo "$(hostname): the TPU is in use"; exit 1; }
pgrep -f '[g]ptnl_mix.py|[b]locklist_scan.py' > /dev/null && { echo "$(hostname): a data step still runs"; exit 1; }
[ "$(git -C ~/mbert-final rev-parse HEAD)" = "$COMMIT" ] || { echo "$(hostname): ~/mbert-final is not at $COMMIT"; exit 1; }
[ -x "$(eval echo "$PY")" ] || { echo "$(hostname): no $PY"; exit 1; }
[ -f ~/$LIST ] && [ "$(wc -l < ~/$LIST)" = $NFILES ] || { echo "$(hostname): ~/$LIST does not list $NFILES files"; exit 1; }
[[ $ROWS =~ ^[0-9]+$ ]] || { echo "$(hostname): no dataset_num_rows in the plan of $VARIANT"; exit 1; }
[ "$(md5sum ~/heldout.parquet | cut -d' ' -f1)" = b2f579a876d06eef40c94270537bb7af ] || { echo "$(hostname): held-out file is not the long run's"; exit 1; }
if [ -n "$BRANCH_STEP" ]; then
  # Without the copied checkpoint this would train from scratch under a branch's name. The copy itself rotates out of
  # ckpt/ after three newer checkpoints, so a resume needs its data state and any checkpoint
  gsutil -q stat "$OUT/ckpt_data_state/$BRANCH_STEP/process_0.json" && gsutil ls "$OUT/ckpt/" | grep -q '/[0-9]*/$' \
    || { echo "$(hostname): no checkpoint $BRANCH_STEP with data state in $OUT; run start_branch.sh"; exit 1; }
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
echo "started $RUN on $(hostname): ~/logs/$RUN.log, $OUT; process_0 may be any host: read the log of the one with Step lines"
