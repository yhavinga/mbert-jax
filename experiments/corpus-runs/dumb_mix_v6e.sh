#!/bin/bash
# On a spot v6e-8 (one host of 8 chips, us-east1-d): DUMB of the mix run's checkpoint at STEP, detached from ssh.
#   COMMIT=<commit of commands.txt> bash dumb_mix_v6e.sh [STEP]      (STEP 300000 by default)
# The protocol of gs://t5-dutch-english-usc2/mbert/masking/dumb (run_one.sh and run_set.sh in its setup/ and setup2/):
# scripts/dumb/finetune.py, DBRD and SICK-NL at --max-length 512, seeds 42 43 44, 3 epochs, lr 2e-5, DBRD 24 per chip x
# 4 chips x 2 accumulation = 192, SICK-NL 64 x 4 x 1 = 256; finetune.py at this commit runs those as the one of the
# earlier rounds (it only gained --pad-to-multiple-of, default 1, and a memory print). Two fine-tunes at a time, each a
# TPU of its own 4 chips (0-3 and 4-7), with the variables dumb_phase.sh sets per worker of a v4-32. The earlier rounds
# ran on a v4-8; this one runs on v6e chips.
# Results as those of mix-100000: $OUT/mix-<STEP>--<task>--seed<seed>.json and $OUT/logs/; a job with a result is
# skipped, so a second start (after a preemption) continues and nothing of another round is overwritten.
# Across regions per start: the params of the checkpoint, 0.92 GB, read in place with --params-only (its optimizer
# state stays behind); the held-out file, 68 MB; scripts and results, under 1 MB.
# Needs ~/.cache/huggingface/token on the VM (commands.txt). QUEUES=1: all jobs one after the other on chips 0-3, for
# when chips 4-7 fail the preflight next to them. Environment overrides, for a smoke test only: OUT CKPT ONLY_JOBS.
# For a fine-tune LR sweep: LR (default 2e-5, the protocol's) and NAME (default mix-<STEP>), with CKPT, OUT and ONLY_JOBS.
# On a worker of a v4-32, as a single-host TPU of its own 4 chips (as dumb_phase.sh of phase2048): QUEUES=1, CODE a
# checkout of its own (the training's ~/mbert-jax may be at another commit there), LOGTAG a name per worker, so the
# workers' logs in $OUT/logs/ do not overwrite each other.
set -euo pipefail
COMMIT=${COMMIT:?set COMMIT to the commit of commands.txt}
QUEUES=${QUEUES:-2}
[ "$QUEUES" = 1 ] || [ "$QUEUES" = 2 ] || { echo "QUEUES must be 1 or 2, not $QUEUES"; exit 1; }
STEP=${1:-300000}
NAME=${NAME:-mix-$STEP}
LR=${LR:-2e-5}
CKPT=${CKPT:-gs://t5-dutch-english-usc2/mbert/culturax-nl-230m-chunk1024-mix/ckpt_keep/$STEP}
OUT=${OUT:-gs://t5-dutch-english-usc2/mbert/masking/dumb}
ONLY_JOBS=${ONLY_JOBS:-}
CODE=${CODE:-mbert-jax}
LOGTAG=${LOGTAG:-v6e}
[ -s ~/.cache/huggingface/token ] || { echo "$(hostname): no ~/.cache/huggingface/token, put it there first (commands.txt)"; exit 1; }
pgrep -f '[t]rain_minimal.py|[f]inetune.py|[e]val_windows.py' > /dev/null && { echo "$(hostname): the TPU is in use"; exit 1; }
gsutil -q stat "$CKPT/commit_success.txt" || { echo "$(hostname): $CKPT has no commit_success.txt (yet)"; exit 1; }
mkdir -p ~/logs ~/dumb
rm -f ~/logs/dumb_$NAME.exit
cat > ~/run_dumb_$NAME.sh <<EOF
set -uo pipefail
cd \$HOME
# Code, venv and held-out file, as setup_fineweb2.sh does
[ -d $CODE/.git ] || git clone -q https://github.com/yhavinga/mbert-jax.git $CODE || exit 2
git -C $CODE fetch -q origin corpus-runs && git -C $CODE checkout -q $COMMIT || exit 2
gsutil -q cp gs://t5-dutch-english-usc2/mbert/corpus-runs/scripts/vm.lock gs://t5-dutch-english-usc2/mbert/phase2048/scripts/verify_ckpt.py \$HOME/ || exit 2
cd $CODE
if ! command -v uv > /dev/null; then curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="\$HOME/.local/bin:\$PATH"; fi
[ -x .venv/bin/python ] || uv venv -q --python 3.12 .venv || exit 2
uv pip install -q --python .venv/bin/python -r \$HOME/vm.lock || exit 2
[ "\$(md5sum \$HOME/heldout.parquet 2>/dev/null | cut -d' ' -f1)" = b2f579a876d06eef40c94270537bb7af ] || gsutil -q cp gs://t5-dutch-english-usc2/mbert/heldout/nl_part_00097_first40000.parquet \$HOME/heldout.parquet
[ "\$(md5sum \$HOME/heldout.parquet | cut -d' ' -f1)" = b2f579a876d06eef40c94270537bb7af ] || { echo "held-out file is not the long run's"; exit 2; }
export HF_DATASETS_TRUST_REMOTE_CODE=1 TOKENIZERS_PARALLELISM=false JAX_COMPILATION_CACHE_DIR=\$HOME/dumb/jaxcache PYTHONPATH=src
# The params only, read where they are; then the held-out check of the earlier rounds
if [ ! -f \$HOME/dumb/ckpt$NAME/params.pkl ]; then
  .venv/bin/python scripts/convert_gcs_checkpoint.py --checkpoint $CKPT --params-only --output \$HOME/dumb/ckpt$NAME --tokenizer yhavinga/dutch-llama-tokenizer --global-attn-every-n-layers 3 || exit 3
  rm -f \$HOME/dumb/ckpt$NAME/model.safetensors
fi
.venv/bin/python \$HOME/verify_ckpt.py \$HOME/dumb/ckpt$NAME || exit 4
# task per-chip-batch accumulation seed; queue w takes the jobs i with i % QUEUES == w
JOBS="dbrd 24 2 42
dbrd 24 2 43
dbrd 24 2 44
sicknl 64 1 42
sicknl 64 1 43
sicknl 64 1 44"
queue() {
  w=\$1; chips=\$2; port=\$3
  export TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 TPU_PROCESS_BOUNDS=1,1,1 TPU_VISIBLE_CHIPS=\$chips TPU_PROCESS_PORT=\$port
  export TPU_PROCESS_ADDRESSES=localhost:\$port CLOUD_TPU_TASK_ID=0 ALLOW_MULTIPLE_LIBTPU_LOAD=1
  # Fail fast when these chips are not a TPU of their own 4 chips
  timeout 600 .venv/bin/python -c "import jax, jax.numpy as jnp; n = jax.device_count(); assert (n, jax.process_count()) == (4, 1), (n, jax.process_count()); print('devices', jax.devices(), float(jnp.arange(4.0).sum()))" || { echo "preflight failed: chips \$chips are not 4 chips of one process"; return 2; }
  i=0; failed=0
  while read -r task batch accumulation seed; do
    mine=\$(( i % $QUEUES == w )); i=\$((i + 1))
    [ \$mine = 1 ] || continue
    job=$NAME--\$task--seed\$seed
    [ -z "$ONLY_JOBS" ] || [[ " $ONLY_JOBS " == *" \$job "* ]] || continue
    gsutil -q stat $OUT/\$job.json && { echo "\$job: a result exists, skipped"; continue; }
    echo "\$(date +%H:%M:%S) \$job: batch \$batch x 4 chips x \$accumulation on chips \$chips"
    .venv/bin/python -u scripts/dumb/finetune.py --checkpoint \$HOME/dumb/ckpt$NAME --task \$task --output \$HOME/dumb/runs/\$job --epochs 3 --batch-size \$batch --gradient-accumulation-steps \$accumulation --learning-rate $LR --max-length 512 --seed \$seed > \$HOME/logs/\$job.log 2>&1
    code=\$?
    echo "EXIT \$code" >> \$HOME/logs/\$job.log
    gsutil -q cp \$HOME/logs/\$job.log $OUT/logs/\$job.log
    if [ \$code = 0 ] && [ -f \$HOME/dumb/runs/\$job/results.json ]; then
      gsutil -q cp \$HOME/dumb/runs/\$job/results.json $OUT/\$job.json
      grep -h "Peak memory\|Test Accuracy" \$HOME/logs/\$job.log | tail -2
    else
      echo "FAILED \$job with exit \$code"; failed=\$((failed + 1))
    fi
    rm -f \$HOME/dumb/runs/\$job/params.pkl
  done <<< "\$JOBS"
  echo "worker \$w: \$failed failed"
  [ \$failed = 0 ]
}
queue 0 0,1,2,3 8476 > \$HOME/logs/dumb_${NAME}_w0.log 2>&1 &
first=\$!
code1=0
if [ $QUEUES = 2 ]; then
  queue 1 4,5,6,7 8477 > \$HOME/logs/dumb_${NAME}_w1.log 2>&1 &
  second=\$!
fi
wait \$first; code0=\$?
[ $QUEUES = 2 ] && { wait \$second; code1=\$?; }
cat \$HOME/logs/dumb_${NAME}_w*.log
echo "queues: worker 0 exit \$code0, worker 1 exit \$code1"
[ \$code0 = 0 ] && [ \$code1 = 0 ]
EOF
setsid nohup bash -c "bash ~/run_dumb_$NAME.sh > ~/logs/dumb_$NAME.log 2>&1; echo EXIT \$? > ~/logs/dumb_$NAME.exit; gsutil -q cp ~/logs/dumb_$NAME.log $OUT/logs/dumb_${NAME}_$LOGTAG.log; gsutil -q cp ~/logs/dumb_$NAME.exit $OUT/logs/dumb_${NAME}_$LOGTAG.exit" > /dev/null 2>&1 < /dev/null &
echo "started the DUMB queue of $NAME on $(hostname): ~/logs/dumb_$NAME.log, results in $OUT"
