#!/bin/bash
# On every worker of mbert-mix (after prepare_gptnl.sh mix, commands.txt): start or resume
# culturax-gptnl-nl-230m-chunk1024, arm G of protocol.md, detached from ssh. A checkpoint in the bucket makes it a resume;
# one of another data setup is refused by the data state.
# The command of culturax-nl-230m-chunk1024 (arm X), run_culturax-nl-230m-chunk1024.sh in
# gs://t5-dutch-english-usc2/mbert/culturax-nl-230m-chunk1024/scripts/, with only these differences:
#   --data-files-list   $HOME/cxgptnl_nl_files.txt instead of $HOME/culturax_nl_files.txt: per host 64 files of 30%
#                       GPT-NL and 70% CulturaX nl in tokens (prepare_gptnl.sh mix, experiments/gptnl/cxgptnl_plan.json)
#   --dataset-num-rows  ROWS, the documents of the 4 shares (dataset_num_rows of cxgptnl_plan.json), instead of
#                       117392666; it feeds only the data/passes metric and the logged config
#   --stop-at-step 300000, after --max-steps 653000: the run stops at step 300,000 on the LR schedule of 653,000 steps
#   the run name culturax-gptnl-nl-230m-chunk1024 instead of culturax-nl-230m-chunk1024 in --checkpoint-dir,
#                       --wandb-project, --wandb-run-name, --tracking-sync-dir, the log and the exit file
# The same as X's, also where it may look otherwise: --dataset uonlp/CulturaX --dataset-config nl (with
# --data-files-list nothing is read through them; they are logged and part of the data state's identity),
# --checkpoint-every 10000, and no --masking, so the default of the trainer, uniform15: X's 15% single-token masking.
# Trainer: ~/mbert-jax at 0d92db4, the code of fineweb2-nl-230m-chunk1024 (F) and -rh3 (R) on this VM; X ran on the code
# of its time (code.tgz in its scripts directory), as F and R do against X too.
# Checked on 2026-11-08 (commands.txt, step 4): PRINT_RUN_SCRIPT=1 prints the run script without starting anything; with
# the 4 differences above undone it is byte-identical to run_culturax-nl-230m-chunk1024.sh.
# Environment overrides, for a smoke test only: RUN STOP_AT CHECKPOINT_EVERY EVAL_EVERY.
set -euo pipefail
RUN=${RUN:-culturax-gptnl-nl-230m-chunk1024}
STOP_AT=${STOP_AT:-300000}
CHECKPOINT_EVERY=${CHECKPOINT_EVERY:-10000}
EVAL_EVERY=${EVAL_EVERY:-25000}
ROWS=47092713
OUT=gs://t5-dutch-english-usc2/mbert/$RUN

run_script() {
  cat <<EOF
cd ~/mbert-jax
TRACKIO_DIR=\$HOME/trackio TOKENIZERS_PARALLELISM=false PYTHONPATH=src .venv/bin/python -u scripts/train_minimal.py --dataset uonlp/CulturaX --dataset-config nl --tokenizer yhavinga/dutch-llama-tokenizer --use-packing --max-length 1024 --hidden-size 1024 --num-layers 22 --num-heads 16 --intermediate-size 1536 --global-attn-every-n-layers 3 --lr-schedule cosine --min-lr 1e-6 --weight-decay 0.01 --layer-norm-lr-factor 1.0 --log-every 10 --dataset-num-rows $ROWS --per-device-batch-size 8 --learning-rate 4.8e-4 --warmup-steps 1250 --max-steps 653000 --stop-at-step $STOP_AT --checkpoint-every $CHECKPOINT_EVERY --checkpoint-dir $OUT/ckpt --wandb-project $RUN --wandb-run-name $RUN --tracking-sync-dir $OUT/tracking --system-metrics-every 1000 --eval-file \$HOME/heldout.parquet --eval-every $EVAL_EVERY --eval-batches 50 --data-files-list \$HOME/cxgptnl_nl_files.txt >> ~/logs/$RUN.log 2>&1
echo EXIT \$? > ~/logs/$RUN.exit
EOF
}

if [ -n "${PRINT_RUN_SCRIPT:-}" ]; then
  run_script
  exit 0
fi
pgrep -f '[t]rain_minimal.py|[f]inetune.py|[e]val_windows.py' > /dev/null && { echo "$(hostname): the TPU is in use"; exit 1; }
pgrep -f '[g]ptnl_mix.py' > /dev/null && { echo "$(hostname): gptnl_mix.py still runs"; exit 1; }
[ "$(git -C ~/mbert-jax rev-parse --short=7 HEAD)" = 0d92db4 ] || { echo "$(hostname): ~/mbert-jax is not at 0d92db4"; exit 1; }
[ -f ~/cxgptnl_nl_files.txt ] && [ "$(wc -l < ~/cxgptnl_nl_files.txt)" = 64 ] && [ -f ~/heldout.parquet ] \
  && [ -x ~/mbert-jax/.venv/bin/python ] || { echo "$(hostname): run prepare_gptnl.sh mix first"; exit 1; }
[ "$(md5sum ~/heldout.parquet | cut -d' ' -f1)" = b2f579a876d06eef40c94270537bb7af ] || { echo "$(hostname): held-out file is not the long run's"; exit 1; }
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
