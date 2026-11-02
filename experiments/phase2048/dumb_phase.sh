#!/bin/bash
# On every worker of mbert-mix: the DUMB fine-tunes of the 2048 phase, detached from ssh. Each worker is a
# single-host TPU of its own 4 chips, as the v4-8 of the earlier rounds, and runs its share of the jobs.
#   bash dumb_phase.sh NAME=STEP_DIR ...
#   e.g. bash dumb_phase.sh u15-S=gs://.../culturax-nl-230m-chunk1024/ckpt_keep/653000 u15-C=$R/p2048-u15-C1024/ckpt/20000 u15-L=$R/p2048-u15-L2048/ckpt/20000
# Per model: DBRD at --max-length 512, 1024 and 2048 and SICK-NL at 512, seeds 42 43 44: the protocol of
# gs://t5-dutch-english-usc2/mbert/masking/dumb (scripts/dumb/finetune.py, 3 epochs, lr 2e-5, an update of 192 for
# DBRD and 256 for SICK-NL on 4 chips), only --max-length differs. At 512 both tasks run exactly as before. At 1024
# and 2048 DBRD takes fewer examples per chip and more accumulation, for the same update of 192 on the same examples,
# and pads to a multiple of 128: padding is masked and the classifier reads <s>, so that changes no result, and it
# saves a compile per batch length (tests/test_dumb_finetune.py). The per-chip batch at 1024 and 2048 is the largest
# candidate whose worst case (every row at that length) fits a chip of this VM (scripts/dumb/probe_memory.py), measured
# once per worker; all workers must come to the same table ($RUN_ROOT/dumb/tables/), or the queue stops.
# Results: $RUN_ROOT/dumb/<NAME>--<task><length>--seed<seed>.json and logs/; a job with a result is skipped, so a
# second start continues. Every worker converts every model first (the held-out check of the earlier rounds included).
# Environment overrides, for the smoke test only: RUN_ROOT CODE PY WORKERS WORKER CHIPS CHIP_BOUNDS PORT ONLY_JOBS.
set -euo pipefail
[ $# -ge 1 ] || { echo "usage: bash dumb_phase.sh NAME=STEP_DIR ..."; exit 1; }
RUN_ROOT=${RUN_ROOT:-gs://t5-dutch-english-usc2/mbert/phase2048}
CODE=${CODE:-$HOME/mbert-p2048}
PY=${PY:-$HOME/mbert-jax/.venv/bin/python}
WORKERS=${WORKERS:-4}
WORKER=${WORKER:-$(hostname | sed 's/.*-w-//')}
# One host of the slice as a TPU of its own (the variables of jax/_src/test_multiprocess.py for 4 chips per process)
CHIPS=${CHIPS:-0,1,2,3}
CHIP_BOUNDS=${CHIP_BOUNDS:-2,2,1}
PORT=${PORT:-8476}
ONLY_JOBS=${ONLY_JOBS:-}

# Per-chip batches to try at 1024 and 2048, largest first; each divides 48, the examples per chip of an update of 192
CANDIDATES_1024="12 8 6 4 3 2 1"
CANDIDATES_2048="4 3 2 1"

pgrep -f '[t]rain_minimal.py|[e]val_windows.py|[f]inetune.py' > /dev/null && { echo "$(hostname): the TPU is in use"; exit 1; }
for spec in "$@"; do
  gsutil -q stat "${spec#*=}/commit_success.txt" || { echo "$(hostname): ${spec#*=} has no commit_success.txt"; exit 1; }
done
mkdir -p ~/logs ~/dumb
rm -f ~/logs/dumb_phase_w$WORKER.exit
cat > ~/run_dumb_phase.sh <<EOF
set -uo pipefail
export TPU_CHIPS_PER_PROCESS_BOUNDS=$CHIP_BOUNDS TPU_PROCESS_BOUNDS=1,1,1 TPU_VISIBLE_CHIPS=$CHIPS TPU_PROCESS_PORT=$PORT
export TPU_PROCESS_ADDRESSES=localhost:$PORT CLOUD_TPU_TASK_ID=0 ALLOW_MULTIPLE_LIBTPU_LOAD=1
export HF_DATASETS_TRUST_REMOTE_CODE=1 TOKENIZERS_PARALLELISM=false JAX_COMPILATION_CACHE_DIR=\$HOME/dumb/jaxcache PYTHONPATH=src
cd $CODE
# Fail fast when this host is not a TPU of its own 4 chips: a fine-tune would wait for the other hosts
timeout 600 $PY -c "import jax, jax.numpy as jnp; n = jax.device_count(); assert (n, jax.process_count()) == (4, 1), (n, jax.process_count()); print('devices', jax.devices(), float(jnp.arange(4.0).sum()))" || { echo "preflight failed: not 4 chips of one process"; exit 2; }
for spec in $*; do
  name=\${spec%%=*}; ckpt=\${spec#*=}
  [ -f \$HOME/dumb/ckpt\$name/params.pkl ] && continue
  rm -rf \$HOME/dumb/orbax/\$name; mkdir -p \$HOME/dumb/orbax/\$name
  gsutil -q -m cp -r "\$ckpt/*" \$HOME/dumb/orbax/\$name/ || exit 3
  $PY scripts/convert_gcs_checkpoint.py --checkpoint \$HOME/dumb/orbax/\$name --output \$HOME/dumb/ckpt\$name --tokenizer yhavinga/dutch-llama-tokenizer --global-attn-every-n-layers 3 || exit 3
  rm -rf \$HOME/dumb/orbax/\$name \$HOME/dumb/ckpt\$name/model.safetensors
  $PY \$HOME/verify_ckpt.py \$HOME/dumb/ckpt\$name || exit 4
done
# task length per-chip-batch accumulation pad-to-multiple-of, with 4 chips: 24 x 4 x 2 = 192, 64 x 4 x 1 = 256
first=${1%%=*}
pick() {
  length=\$1; shift
  for b in "\$@"; do
    $PY scripts/dumb/probe_memory.py --checkpoint \$HOME/dumb/ckpt\$first --max-length \$length --batch-size \$b --gradient-accumulation-steps \$((48 / b)) >> \$HOME/logs/dumb_probes.log 2>&1
    code=\$?
    [ \$code = 0 ] && { echo \$b; return 0; }
    [ \$code = 3 ] || { echo "probe failed with exit \$code, see ~/logs/dumb_probes.log" >&2; return 1; }
  done
  echo "no candidate fits at \$length" >&2; return 1
}
TABLE_FILE=$RUN_ROOT/dumb/tables/w$WORKER.txt
if gsutil -q stat \$TABLE_FILE; then
  table=\$(gsutil cat \$TABLE_FILE)
else
  b1024=\$(pick 1024 $CANDIDATES_1024) || exit 5
  b2048=\$(pick 2048 $CANDIDATES_2048) || exit 5
  gsutil -q cp \$HOME/logs/dumb_probes.log $RUN_ROOT/dumb/logs/probes_w$WORKER.log
  table="dbrd 512 24 2 1
dbrd 1024 \$b1024 \$((48 / b1024)) 128
dbrd 2048 \$b2048 \$((48 / b2048)) 128
sicknl 512 64 1 1"
  echo "\$table" | gsutil -q cp - \$TABLE_FILE
fi
echo "jobs table of this worker:"; echo "\$table"
for w in \$(seq 0 $((WORKERS - 1))); do
  waited=0
  until gsutil -q stat $RUN_ROOT/dumb/tables/w\$w.txt; do
    [ \$waited -gt 5400 ] && { echo "no table of worker \$w after 90 minutes"; exit 6; }
    sleep 60; waited=\$((waited + 60))
  done
  [ "\$(gsutil cat $RUN_ROOT/dumb/tables/w\$w.txt)" = "\$table" ] || { echo "worker \$w came to another table"; exit 6; }
done
i=0
failed=0
while read -r task length batch accumulation pad; do
  for spec in $*; do
    name=\${spec%%=*}
    for seed in 42 43 44; do
      job=\$name--\$task\$length--seed\$seed
      mine=\$(( i % $WORKERS == $WORKER )); i=\$((i + 1))
      [ \$mine = 1 ] || continue
      [ -z "$ONLY_JOBS" ] || [[ " $ONLY_JOBS " == *" \$job "* ]] || continue
      gsutil -q stat $RUN_ROOT/dumb/\$job.json && continue
      echo "\$(date +%H:%M:%S) \$job: batch \$batch x 4 chips x \$accumulation, pad to a multiple of \$pad"
      $PY -u scripts/dumb/finetune.py --checkpoint \$HOME/dumb/ckpt\$name --task \$task --output \$HOME/dumb/runs/\$job --epochs 3 --batch-size \$batch --gradient-accumulation-steps \$accumulation --learning-rate 2e-5 --max-length \$length --pad-to-multiple-of \$pad --seed \$seed > \$HOME/logs/\$job.log 2>&1
      code=\$?
      echo "EXIT \$code" >> \$HOME/logs/\$job.log
      gsutil -q cp \$HOME/logs/\$job.log $RUN_ROOT/dumb/logs/\$job.log
      if [ \$code = 0 ] && [ -f \$HOME/dumb/runs/\$job/results.json ]; then
        gsutil -q cp \$HOME/dumb/runs/\$job/results.json $RUN_ROOT/dumb/\$job.json
        grep -h "Peak memory\|Test Accuracy" \$HOME/logs/\$job.log | tail -2
      else
        echo "FAILED \$job with exit \$code"; failed=\$((failed + 1))
      fi
      rm -f \$HOME/dumb/runs/\$job/params.pkl
    done
  done
done <<< "\$table"
echo "\$failed failed"
[ \$failed = 0 ]
EOF
setsid nohup bash -c "bash ~/run_dumb_phase.sh >> ~/logs/dumb_phase_w$WORKER.log 2>&1; echo EXIT \$? > ~/logs/dumb_phase_w$WORKER.exit; gsutil -q cp ~/logs/dumb_phase_w$WORKER.log ~/logs/dumb_phase_w$WORKER.exit $RUN_ROOT/dumb/logs/" > /dev/null 2>&1 < /dev/null &
echo "started worker $WORKER of $WORKERS of the DUMB queue on $(hostname): ~/logs/dumb_phase_w$WORKER.log, $RUN_ROOT/dumb"
