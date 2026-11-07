#!/bin/bash
# On every worker of mbert-corpus (after setup_flan.sh): the zero-shot eval of one model per worker, detached from ssh.
# Worker w takes the w-th NAME=STEP_DIR argument as a single-host TPU of its own 4 chips (the variables of
# experiments/phase2048/dumb_phase.sh); a worker without one ends at once.
#   COMMIT=<commit> PY=$HOME/mbert-dumb/.venv/bin/python bash eval_zeroshot.sh S=<step dir> W=<step dir> F=<step dir> C=<step dir>
# scripts/eval_zeroshot_mlm.py: DBRD test, SICK-NL test and Belebele nld_Latn, 3 Dutch templates each, the verbalizer
# with the highest logit at <answer> <mask>. Results: $RUN_ROOT/zeroshot/NAME.json; a model with a result is skipped.
# Logs: $RUN_ROOT/zeroshot/logs/NAME.log and .exit.
# Environment overrides, for the smoke test only: RUN_ROOT EXTRA (e.g. EXTRA="--max-examples 50").
set -euo pipefail
[ $# -ge 1 ] || { echo "usage: COMMIT=<commit> bash eval_zeroshot.sh NAME=STEP_DIR ..."; exit 1; }
COMMIT=${COMMIT:?set COMMIT to the commit of commands.txt}
RUN_ROOT=${RUN_ROOT:-gs://t5-dutch-english-usc2/mbert/flan-decay}
CODE=$HOME/mbert-flan
PY=${PY:-$HOME/mbert-corpus/.venv/bin/python}
EXTRA=${EXTRA:-}
WORKER=$(hostname | sed 's/.*-w-//')
specs=("$@")
[ "$WORKER" -lt ${#specs[@]} ] || { echo "w$WORKER: no model for this worker"; exit 0; }
spec=${specs[$WORKER]}; name=${spec%%=*}; ckpt=${spec#*=}
pgrep -f '[t]rain_minimal.py|[e]val_windows.py|[e]val_zeroshot_mlm.py|[f]inetune.py' > /dev/null && { echo "$(hostname): the TPU is in use"; exit 1; }
gsutil -q stat "$ckpt/commit_success.txt" || { echo "$(hostname): $ckpt has no commit_success.txt"; exit 1; }
[ -x "$PY" ] || { echo "$(hostname): no $PY"; exit 1; }
[ -d "$CODE/.git" ] || git clone -q https://github.com/yhavinga/mbert-jax.git "$CODE"
git -C "$CODE" fetch -q origin flan-decay
git -C "$CODE" checkout -q "$COMMIT"
mkdir -p ~/logs
rm -f ~/logs/zeroshot_$name.exit
# With these variables eval_zeroshot_mlm.py sets up a runtime of one process before it imports train_minimal, whose
# jax.distributed.initialize() would otherwise take the slice from the TPU metadata and abort after waiting for the
# other hosts (exit 134 on 7 Nov). eval_zeroshot_mlm.py refuses more than one process.
cat > ~/run_zeroshot_$name.sh <<EOF
set -uo pipefail
export TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 TPU_PROCESS_BOUNDS=1,1,1 TPU_VISIBLE_CHIPS=0,1,2,3 TPU_PROCESS_PORT=8476
export TPU_PROCESS_ADDRESSES=localhost:8476 CLOUD_TPU_TASK_ID=0 ALLOW_MULTIPLE_LIBTPU_LOAD=1
export HF_DATASETS_TRUST_REMOTE_CODE=1 TOKENIZERS_PARALLELISM=false PYTHONPATH=src
cd $CODE
gsutil -q stat $RUN_ROOT/zeroshot/$name.json && { echo "$name: done before"; exit 0; }
# Fail fast when this host is not a TPU of its own 4 chips: the eval would wait for the other hosts
timeout 600 $PY -c "import jax; assert (jax.device_count(), jax.process_count()) == (4, 1), (jax.device_count(), jax.process_count()); print('devices', jax.devices())" || { echo "preflight failed: not 4 chips of one process"; exit 2; }
echo "\$(date +%H:%M:%S) $name: $ckpt"
$PY -u scripts/eval_zeroshot_mlm.py --checkpoint $ckpt --tokenizer yhavinga/dutch-llama-tokenizer --hidden-size 1024 --num-layers 22 --num-heads 16 --intermediate-size 1536 --global-attn-every-n-layers 3 $EXTRA --output $RUN_ROOT/zeroshot/$name.json
EOF
setsid nohup bash -c "bash ~/run_zeroshot_$name.sh > ~/logs/zeroshot_$name.log 2>&1; echo EXIT \$? > ~/logs/zeroshot_$name.exit; gsutil -q cp ~/logs/zeroshot_$name.log $RUN_ROOT/zeroshot/logs/$name.log; gsutil -q cp ~/logs/zeroshot_$name.exit $RUN_ROOT/zeroshot/logs/$name.exit" > /dev/null 2>&1 < /dev/null &
echo "started the zero-shot eval of $name on $(hostname): ~/logs/zeroshot_$name.log, $RUN_ROOT/zeroshot/$name.json"
