#!/bin/bash
# On one worker (a single-host TPU of its own 4 chips): BLiMP-NL of a list of checkpoints, one after the other, detached.
#   COMMIT=<code commit of commands.txt> bash blimp.sh NAME=STEP_DIR ...
# scripts/eval_blimp_nl.py: the 8,400 pairs of BLiMP-NL large, PLL-word-l2r with the MLM head, float32, head 0 of a
# --mix-heads checkpoint; the params are read in place from the step directory (same region). Results:
# $OUT/NAME.json; a model with a result is skipped. Logs: $OUT/logs/NAME.log (ends in EXIT <code>), the queue's
# ~/logs/blimp_<TAG>.log and at the end $OUT/logs/blimp_<TAG>.{log,exit}.
# Environment overrides, for a smoke test only: OUT CODE PY TAG EXTRA (e.g. EXTRA="--max-pairs 200").
set -euo pipefail
[ $# -ge 1 ] || { echo "usage: COMMIT=<commit> bash blimp.sh NAME=STEP_DIR ..."; exit 1; }
COMMIT=${COMMIT:?set COMMIT to the code commit of commands.txt}
OUT=${OUT:-gs://t5-dutch-english-usc2/mbert/blimp-nl}
CODE=${CODE:-$HOME/mbert-dumbx}
PY=${PY:-$HOME/mbert-dumb/.venv/bin/python}
TAG=${TAG:-$(hostname | sed 's/.*-w-/w/')}
EXTRA=${EXTRA:-}
DATA=gs://t5-dutch-english-usc2/mbert/dumb-extra/data
pgrep -f '[t]rain_minimal.py|[f]inetune.py|[f]inetune_extra.py|[e]val_windows.py|[e]val_zeroshot_mlm.py|[e]val_blimp_nl.py' > /dev/null && { echo "$(hostname): the TPU is in use"; exit 1; }
[ -x "$PY" ] || { echo "$(hostname): no $PY"; exit 1; }
for spec in "$@"; do
  [[ "$spec" == *=gs://* ]] || { echo "not NAME=gs://STEP_DIR: $spec"; exit 1; }
  gsutil -q stat "${spec#*=}/commit_success.txt" || { echo "$(hostname): ${spec#*=} has no commit_success.txt"; exit 1; }
done
mkdir -p ~/logs
rm -f ~/logs/blimp_$TAG.exit
cat > ~/run_blimp_$TAG.sh <<EOF
set -uo pipefail
export TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 TPU_PROCESS_BOUNDS=1,1,1 TPU_VISIBLE_CHIPS=0,1,2,3 TPU_PROCESS_PORT=8476
export TPU_PROCESS_ADDRESSES=localhost:8476 CLOUD_TPU_TASK_ID=0 ALLOW_MULTIPLE_LIBTPU_LOAD=1
export TOKENIZERS_PARALLELISM=false PYTHONPATH=src
cd \$HOME
[ -d $CODE/.git ] || git clone -q https://github.com/yhavinga/mbert-jax.git $CODE || exit 2
git -C $CODE fetch -q origin dumb-extra && git -C $CODE checkout -q $COMMIT || exit 2
cd $CODE
mkdir -p \$HOME/dumbx/data/blimp-nl && gsutil -q cp $DATA/blimp-nl/large.parquet \$HOME/dumbx/data/blimp-nl/large.parquet || exit 2
# Fail fast, before any compile, when this host is not a TPU of its own 4 chips: the eval would wait for the other hosts
timeout 600 $PY -c "import jax; assert (jax.device_count(), jax.process_count()) == (4, 1), (jax.device_count(), jax.process_count()); print('devices', jax.devices())" || { echo "preflight failed: not 4 chips of one process"; exit 3; }
failed=0
for spec in $*; do
  name=\${spec%%=*}; ckpt=\${spec#*=}
  gsutil -q stat $OUT/\$name.json && { echo "\$name: a result exists, skipped"; continue; }
  echo "\$(date +%H:%M:%S) \$name: \$ckpt"
  $PY -u scripts/eval_blimp_nl.py --checkpoint \$ckpt --data \$HOME/dumbx/data/blimp-nl/large.parquet --tokenizer yhavinga/dutch-llama-tokenizer --hidden-size 1024 --num-layers 22 --num-heads 16 --intermediate-size 1536 --global-attn-every-n-layers 3 --export-head 0 $EXTRA --output $OUT/\$name.json > \$HOME/logs/blimp_\$name.log 2>&1 < /dev/null
  code=\$?
  echo "EXIT \$code" >> \$HOME/logs/blimp_\$name.log
  gsutil -q cp \$HOME/logs/blimp_\$name.log $OUT/logs/\$name.log
  if [ \$code = 0 ]; then grep -h "^BLIMP accuracy" \$HOME/logs/blimp_\$name.log; else echo "FAILED \$name with exit \$code"; failed=\$((failed + 1)); fi
done
echo "\$failed failed"
[ \$failed = 0 ]
EOF
setsid nohup bash -c "bash ~/run_blimp_$TAG.sh > ~/logs/blimp_$TAG.log 2>&1; echo EXIT \$? > ~/logs/blimp_$TAG.exit; gsutil -q cp ~/logs/blimp_$TAG.log ~/logs/blimp_$TAG.exit $OUT/logs/" > /dev/null 2>&1 < /dev/null &
echo "started BLiMP-NL of $# models on $(hostname): ~/logs/blimp_$TAG.log, results in $OUT"
