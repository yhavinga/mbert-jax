#!/bin/bash
# On every worker of mbert-mix: the MLM window eval of each model, one model after the other, detached from ssh.
#   bash eval_phase.sh NAME=STEP_DIR ...
#   e.g. bash eval_phase.sh u15-S=gs://.../culturax-nl-230m-chunk1024/ckpt_keep/653000 u15-C=$R/p2048-u15-C1024/ckpt/20000 u15-L=$R/p2048-u15-L2048/ckpt/20000
# scripts/eval_windows.py on all of ~/heldout.parquet at windows 1024 and 2048, 15% single tokens with mask seed 0 for
# every model, whatever its training masking. Results: $RUN_ROOT/eval/NAME.json; a model with a result is skipped.
# Environment overrides, for the smoke test only: RUN_ROOT CODE PY EXTRA (e.g. EXTRA="--max-batches 5").
set -euo pipefail
[ $# -ge 1 ] || { echo "usage: bash eval_phase.sh NAME=STEP_DIR ..."; exit 1; }
RUN_ROOT=${RUN_ROOT:-gs://t5-dutch-english-usc2/mbert/phase2048}
CODE=${CODE:-$HOME/mbert-p2048}
PY=${PY:-$HOME/mbert-jax/.venv/bin/python}
EXTRA=${EXTRA:-}
pgrep -f '[t]rain_minimal.py|[e]val_windows.py|[f]inetune.py' > /dev/null && { echo "$(hostname): the TPU is in use"; exit 1; }
for spec in "$@"; do
  gsutil -q stat "${spec#*=}/commit_success.txt" || { echo "$(hostname): ${spec#*=} has no commit_success.txt"; exit 1; }
done
mkdir -p ~/logs
rm -f ~/logs/eval_phase.exit
cat > ~/run_eval_phase.sh <<EOF
set -uo pipefail
cd $CODE
for spec in $*; do
  name=\${spec%%=*}; ckpt=\${spec#*=}
  gsutil -q stat $RUN_ROOT/eval/\$name.json && { echo "\$name: done before"; continue; }
  echo "\$(date +%H:%M:%S) \$name: \$ckpt"
  TOKENIZERS_PARALLELISM=false PYTHONPATH=src $PY -u scripts/eval_windows.py --checkpoint \$ckpt --eval-file \$HOME/heldout.parquet --tokenizer yhavinga/dutch-llama-tokenizer --hidden-size 1024 --num-layers 22 --num-heads 16 --intermediate-size 1536 --global-attn-every-n-layers 3 --windows 1024 2048 --seed 0 $EXTRA --output $RUN_ROOT/eval/\$name.json || exit \$?
done
EOF
cat > ~/sync_eval_phase.sh <<EOF
w=\$(hostname | sed 's/.*-w-/w/')
while [ ! -f ~/logs/eval_phase.exit ]; do gsutil -q cp ~/logs/eval_phase.log $RUN_ROOT/eval/logs/\$w.log; sleep 120; done
gsutil -q cp ~/logs/eval_phase.log $RUN_ROOT/eval/logs/\$w.log; gsutil -q cp ~/logs/eval_phase.exit $RUN_ROOT/eval/logs/\$w.exit
EOF
setsid nohup bash -c "bash ~/run_eval_phase.sh >> ~/logs/eval_phase.log 2>&1; echo EXIT \$? > ~/logs/eval_phase.exit" > /dev/null 2>&1 < /dev/null &
setsid nohup bash ~/sync_eval_phase.sh > /dev/null 2>&1 < /dev/null &
echo "started the window eval of $# models on $(hostname): ~/logs/eval_phase.log, $RUN_ROOT/eval"
