#!/bin/bash
# On every worker of a v4-32 (or on one host with WORKER=0): the fine-tunes of experiments/dumb-extra, detached from ssh.
#   COMMIT=<code commit of commands.txt> bash dumbx.sh SPEC
# SPEC (a local file or gs://): lines "WORKER NAME CKPT TASK LR OUT", # starts a comment. This worker runs, one after the
# other, the lines whose WORKER is its number (the hostname's -w-N), each for seeds 42 43 44: scripts/dumb/finetune_extra.py
# with the protocol's settings (protocol.md): 3 epochs, LR of the line, linear decay to 0 without warmup, 8 per chip x 4
# chips x 1 = an update of 32, padding to a multiple of 128; squadnl at 384 tokens with a doc stride of 128, the token
# tasks at 512. Results: OUT/NAME--TASK--seed<n>.json, logs OUT/logs/NAME--TASK--seed<n>.log (each ends in EXIT <code>);
# a job with a result is skipped, so a second start (after a preemption or a failure) continues.
# Each worker is a single-host TPU of its own 4 chips (the variables of dumb_phase.sh), with a checkout of its own at
# COMMIT ($CODE, default ~/mbert-dumbx) and the venv of the earlier DUMB rounds ($PY, default ~/mbert-dumb/.venv; no
# package is installed, the code needs none beyond vm.lock). Per model, once per worker: convert_gcs_checkpoint.py
# --params-only reads the params in place (0.92 GB, same region), a --mix-heads checkpoint keeps head 0 (as
# EXPORT_HEAD=0 of corpus-runs/dumb_mix_v6e.sh), and verify_ckpt.py checks the tree and prints the held-out accuracy.
# The data: gs://t5-dutch-english-usc2/mbert/dumb-extra/data/ (26 MB) to ~/dumbx/data. Worker log: ~/logs/dumbx_w<N>.log,
# at the end also $LOGS/dumbx_w<N>.{log,exit}.
# Environment overrides, for a smoke test only: WORKER CODE PY DATA LOGS SEEDS EXTRA (extra finetune_extra.py arguments,
# e.g. EXTRA="--epochs 1 --max-train-examples 256 --max-eval-examples 64", with OUT of the spec in a smoke directory).
set -euo pipefail
SPEC=${1:?usage: COMMIT=<commit> bash dumbx.sh SPEC}
COMMIT=${COMMIT:?set COMMIT to the code commit of commands.txt}
CODE=${CODE:-$HOME/mbert-dumbx}
PY=${PY:-$HOME/mbert-dumb/.venv/bin/python}
WORKER=${WORKER:-$(hostname | sed 's/.*-w-//')}
DATA=${DATA:-gs://t5-dutch-english-usc2/mbert/dumb-extra/data}
LOGS=${LOGS:-gs://t5-dutch-english-usc2/mbert/dumb-extra/logs}
SEEDS=${SEEDS:-42 43 44}
EXTRA=${EXTRA:-}
[[ "$WORKER" =~ ^[0-9]+$ ]] || { echo "$(hostname): no worker number in the hostname; set WORKER"; exit 1; }
pgrep -f '[t]rain_minimal.py|[f]inetune.py|[f]inetune_extra.py|[e]val_windows.py|[e]val_zeroshot_mlm.py|[e]val_blimp_nl.py' > /dev/null && { echo "$(hostname): the TPU is in use"; exit 1; }
[ -x "$PY" ] || { echo "$(hostname): no $PY"; exit 1; }
mkdir -p ~/logs ~/dumbx
case "$SPEC" in gs://*) gsutil -q cp "$SPEC" ~/dumbx/spec_w$WORKER.txt ;; *) cp "$SPEC" ~/dumbx/spec_w$WORKER.txt ;; esac
# This worker's lines, checked before anything starts
mine=$(grep -v '^\s*#' ~/dumbx/spec_w$WORKER.txt | awk -v w="$WORKER" 'NF && $1 == w')
bad=$(grep -v '^\s*#' ~/dumbx/spec_w$WORKER.txt | awk 'NF && (NF != 6 || $1 !~ /^[0-9]+$/ || $4 !~ /^(squadnl|conll2002|pos-lassysmall|pos-alpino)$/)')
[ -z "$bad" ] || { echo "spec lines that are not 'WORKER NAME CKPT TASK LR OUT' with a known task:"; echo "$bad"; exit 1; }
[ -n "$mine" ] || { echo "w$WORKER: no lines for this worker in $SPEC"; exit 0; }
while read -r _ name ckpt _ _ _; do
  gsutil -q stat "$ckpt/commit_success.txt" || { echo "$(hostname): $ckpt ($name) has no commit_success.txt"; exit 1; }
done <<< "$mine"
rm -f ~/logs/dumbx_w$WORKER.exit
cat > ~/run_dumbx_w$WORKER.sh <<EOF
set -uo pipefail
export TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 TPU_PROCESS_BOUNDS=1,1,1 TPU_VISIBLE_CHIPS=0,1,2,3 TPU_PROCESS_PORT=8476
export TPU_PROCESS_ADDRESSES=localhost:8476 CLOUD_TPU_TASK_ID=0 ALLOW_MULTIPLE_LIBTPU_LOAD=1
export HF_DATASETS_TRUST_REMOTE_CODE=1 TOKENIZERS_PARALLELISM=false JAX_COMPILATION_CACHE_DIR=\$HOME/dumbx/jaxcache PYTHONPATH=src
cd \$HOME
[ -d $CODE/.git ] || git clone -q https://github.com/yhavinga/mbert-jax.git $CODE || exit 2
git -C $CODE fetch -q origin dumb-extra && git -C $CODE checkout -q $COMMIT || exit 2
cd $CODE
$PY -c "import jax, flax, optax, orbax.checkpoint, transformers, pyarrow, tqdm" || { echo "$PY lacks a package of vm.lock"; exit 2; }
mkdir -p \$HOME/dumbx/data && gsutil -q -m cp -r "$DATA/*" \$HOME/dumbx/data/ || exit 2
gsutil -q cp gs://t5-dutch-english-usc2/mbert/phase2048/scripts/verify_ckpt.py \$HOME/verify_ckpt.py || exit 2
[ "\$(md5sum \$HOME/heldout.parquet 2>/dev/null | cut -d' ' -f1)" = b2f579a876d06eef40c94270537bb7af ] || gsutil -q cp gs://t5-dutch-english-usc2/mbert/heldout/nl_part_00097_first40000.parquet \$HOME/heldout.parquet
[ "\$(md5sum \$HOME/heldout.parquet | cut -d' ' -f1)" = b2f579a876d06eef40c94270537bb7af ] || { echo "held-out file is not the long run's"; exit 2; }
# Fail fast, before any compile, when this host is not a TPU of its own 4 chips: a fine-tune would wait for the other hosts
timeout 600 $PY -c "import jax, jax.numpy as jnp; n = jax.device_count(); assert (n, jax.process_count()) == (4, 1), (n, jax.process_count()); print('devices', jax.devices(), float(jnp.arange(4.0).sum()))" || { echo "preflight failed: not 4 chips of one process"; exit 3; }
failed=0
while read -r _ name ckpt task lr out; do
  if [ ! -f \$HOME/dumbx/ckpt\$name/params.pkl ]; then
    $PY scripts/convert_gcs_checkpoint.py --checkpoint \$ckpt --params-only --output \$HOME/dumbx/ckpt\$name --tokenizer yhavinga/dutch-llama-tokenizer --global-attn-every-n-layers 3 < /dev/null || { echo "conversion of \$name failed"; exit 4; }
    rm -f \$HOME/dumbx/ckpt\$name/model.safetensors
    # A --mix-heads checkpoint: head 0 under the names of the single head, the other heads left out
    $PY - \$HOME/dumbx/ckpt\$name/params.pkl <<'PY' || exit 4
import pickle, re, sys
path = sys.argv[1]
p = pickle.load(open(path, "rb"))
extra = [n for n in p if re.fullmatch(r"(prediction_head|decoder)_[0-9]+", n)]
if extra:
    for n in extra:
        del p[n]
    pickle.dump(p, open(path, "wb"))
    print(f"head 0 kept as prediction_head/decoder; left out: {sorted(extra)}")
PY
    $PY \$HOME/verify_ckpt.py \$HOME/dumbx/ckpt\$name < /dev/null || { echo "verify_ckpt.py failed on \$name"; rm -rf \$HOME/dumbx/ckpt\$name; exit 4; }
  fi
  case \$task in squadnl) length="--max-length 384 --doc-stride 128" ;; *) length="--max-length 512" ;; esac
  for seed in $SEEDS; do
    job=\$name--\$task--seed\$seed
    gsutil -q stat \$out/\$job.json < /dev/null && { echo "\$job: a result exists, skipped"; continue; }
    echo "\$(date +%H:%M:%S) \$job: \$ckpt, lr \$lr, 8 x 4 chips x 1"
    $PY -u scripts/dumb/finetune_extra.py --checkpoint \$HOME/dumbx/ckpt\$name --task \$task --data-dir \$HOME/dumbx/data --output \$HOME/dumbx/runs/\$job --epochs 3 --batch-size 8 --gradient-accumulation-steps 1 --learning-rate \$lr --lr-schedule linear --warmup-steps 0 --pad-to-multiple-of 128 \$length --seed \$seed $EXTRA > \$HOME/logs/\$job.log 2>&1 < /dev/null
    code=\$?
    echo "EXIT \$code" >> \$HOME/logs/\$job.log
    gsutil -q cp \$HOME/logs/\$job.log \$out/logs/\$job.log
    if [ \$code = 0 ] && [ -f \$HOME/dumbx/runs/\$job/results.json ]; then
      gsutil -q cp \$HOME/dumbx/runs/\$job/results.json \$out/\$job.json
      grep -h "Peak memory\|^Test" \$HOME/logs/\$job.log | tail -2
    else
      echo "FAILED \$job with exit \$code"; failed=\$((failed + 1))
    fi
  done
done < \$HOME/dumbx/mine_w$WORKER.txt
echo "w$WORKER: \$failed failed"
[ \$failed = 0 ]
EOF
echo "$mine" > ~/dumbx/mine_w$WORKER.txt
setsid nohup bash -c "bash ~/run_dumbx_w$WORKER.sh > ~/logs/dumbx_w$WORKER.log 2>&1; echo EXIT \$? > ~/logs/dumbx_w$WORKER.exit; gsutil -q cp ~/logs/dumbx_w$WORKER.log ~/logs/dumbx_w$WORKER.exit $LOGS/" > /dev/null 2>&1 < /dev/null &
echo "started the dumb-extra queue of w$WORKER on $(hostname): $(echo "$mine" | wc -l | tr -d ' ') lines, ~/logs/dumbx_w$WORKER.log"
