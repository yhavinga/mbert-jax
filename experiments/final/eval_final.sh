#!/bin/bash
# On one worker of mbert-mix (v4-32, us-central2-b), started by final_eval.sh on co1: this worker's share of the
# evaluations of one arm of protocol.md ("Evaluatie"), detached from ssh. The worker is a single-host TPU of its own 4
# chips (the variables of dumb_phase.sh and dumbx.sh) and runs its jobs one after the other.
#   COMMIT=<dumb-extra code commit> bash eval_final.sh NAME CKPT JOB...
# JOB is <task>:<seed> with task dbrd1024 or sicknl (scripts/dumb/finetune.py), squadnl, conll2002, pos-lassysmall or
# pos-alpino (scripts/dumb/finetune_extra.py), or "blimp" (scripts/eval_blimp_nl.py, one per model). The commands are
# those of the earlier rounds; only the fine-tune LR of DBRD and SICK-NL is the protocol's 5e-5:
#   dbrd1024  finetune.py --task dbrd --max-length 1024, 12 per chip x 4 chips x 4 accumulation = 192, padding to a
#             multiple of 128: the table that dumb_phase.sh measured on these workers (gs://t5-dutch-english-usc2/
#             mbert/dbrd1024/dumb/tables/w0-3.txt, "dbrd 1024 12 4 128"), as G's DBRD@1024
#   sicknl    finetune.py --task sicknl --max-length 512, 64 x 4 x 1 = 256, no padding (dumb_mix_v6e.sh)
#             both --epochs 3, constant LR (finetune.py has no schedule), AdamW with weight decay 0.01
#   extra     the command of dumbx.sh at ae11478: 8 x 4 x 1 = 32, LR 5e-5 falling linearly to 0 without warmup,
#             3 epochs, padding to 128; squadnl --max-length 384 --doc-stride 128, the token tasks --max-length 512
#   blimp     the command of blimp.sh at ae11478: PLL-word-l2r over the 8,400 pairs, float32, --export-head 0
# Code: a checkout of mbert-jax at COMMIT in $CODE (~/mbert-dumbx, as dumbx.sh); the venv of the DUMB rounds ($PY,
# ~/mbert-dumb/.venv, vm.lock; nothing is installed). The model is converted once per worker (--params-only, read in
# place) and checked with verify_ckpt.py, into $ROOT/ckpt<NAME>.
# Results (a job with a result is skipped, so a second start continues):
#   $OUT/dumb/NAME--{dbrd1024,sicknl}--seed<n>.json    $OUT/dumb-extra/<task>/NAME--<task>--seed<n>.json
#   $OUT/blimp-nl/NAME.json                            each job's log in logs/ next to them, ending in EXIT <code>
# Nothing is deleted: converted model, run directories and data go to $ROOT (/dev/shm/final_eval, RAM; the workers have
# 8-10 GB of free disk), the compile caches are those of the earlier rounds (~/dumb/jaxcache for finetune.py,
# ~/dumbx/jaxcache for finetune_extra.py), and an earlier exit file is renamed to .exit.<epoch seconds>.
# Queue files: $LOGDIR/NAME_w<N>.{log,pid,exit} (~/logs/final_eval/), the log and exit file at the end also in
# $OUT/eval/logs/NAME/. Prints "started <pid>", or "running <pid>" when this NAME's queue already runs here (nothing
# started), or a refusal and exit code 1.
# Environment overrides, for the dry run only: WORKER CODE PY ROOT OUT LOGDIR DATA M.
set -euo pipefail
[ $# -ge 3 ] || { echo "usage: COMMIT=<commit> bash eval_final.sh NAME CKPT JOB..."; exit 1; }
NAME=$1; CKPT=$2; shift 2
COMMIT=${COMMIT:?set COMMIT to the dumb-extra code commit}
M=${M:-gs://t5-dutch-english-usc2/mbert}
OUT=${OUT:-$M/final}
DATA=${DATA:-$M/dumb-extra/data}
CODE=${CODE:-$HOME/mbert-dumbx}
PY=${PY:-$HOME/mbert-dumb/.venv/bin/python}
ROOT=${ROOT:-/dev/shm/final_eval}
LOGDIR=${LOGDIR:-$HOME/logs/final_eval}
WORKER=${WORKER:-$(hostname | sed 's/.*-w-//')}
BUSY='[t]rain_minimal.py|[f]inetune.py|[f]inetune_extra.py|[e]val_windows.py|[e]val_zeroshot_mlm.py|[e]val_blimp_nl.py|[v]erify_ckpt.py|[c]onvert_gcs_checkpoint.py|[p]robe_memory.py|[t]ranslate.py|[j]udge.py|[m]etricx_qe.py|[c]omet_score.py|[n]aturalness.py|[s]ft_lora.py'
[[ "$WORKER" =~ ^[0-9]+$ ]] || { echo "$(hostname): no worker number in the hostname; set WORKER"; exit 1; }
[[ "$NAME" =~ ^[a-z0-9][a-z0-9-]*$ ]] || { echo "w$WORKER: not a name: $NAME"; exit 1; }
[[ "$CKPT" == gs://* ]] || { echo "w$WORKER: not a gs:// step directory: $CKPT"; exit 1; }
for job in "$@"; do
  [[ "$job" =~ ^(dbrd1024|sicknl|squadnl|conll2002|pos-lassysmall|pos-alpino):(42|43|44)$ || "$job" == blimp ]] || {
    echo "w$WORKER: not a job (<task>:<seed> or blimp): $job"; exit 1; }
done
mkdir -p "$LOGDIR"
Q=$LOGDIR/${NAME}_w$WORKER
RUN=$HOME/run_final_eval_${NAME}_w$WORKER.sh
# This name's queue may still run here (a restart of the chain): then nothing is started
if [ -s "$Q.pid" ] && [ ! -e "$Q.exit" ]; then
  pid=$(cat "$Q.pid")
  if [ -r "/proc/$pid/cmdline" ] && tr '\0' ' ' < "/proc/$pid/cmdline" | grep -q "run_final_eval_${NAME}_w$WORKER"; then
    echo "running $pid"; exit 0
  fi
fi
busy=$(pgrep -af "$BUSY" | cut -c1-160 || true)
[ -z "$busy" ] || { echo "w$WORKER: the TPU is in use: $(echo "$busy" | tr '\n' ' ')"; exit 1; }
holders=$(sudo lsof -t /dev/accel* /dev/vfio/* 2>/dev/null | sort -u | tr '\n' ' ' || true)
[ -z "${holders// /}" ] || { echo "w$WORKER: the TPU is held by: $holders"; exit 1; }
[ -x "$PY" ] || { echo "w$WORKER: no $PY"; exit 1; }
# finetune.py reads DBRD and SICK-NL from the Hugging Face hub (dumb_mix_v6e.sh); the token is checked, never printed
[ -s "$HOME/.cache/huggingface/token" ] || { echo "w$WORKER: no ~/.cache/huggingface/token (put it there through stdin)"; exit 1; }
gsutil -q stat "$CKPT/commit_success.txt" || { echo "w$WORKER: $CKPT has no commit_success.txt"; exit 1; }
# Per arm: ~1.8 GB of converted model and up to ~0.9 GB per DBRD/SICK-NL job in $ROOT; compile caches on the disk
mkdir -p "$ROOT"
shm=$(df -BG --output=avail "$ROOT" | tail -1 | tr -dc 0-9); disk=$(df -BG --output=avail "$HOME" | tail -1 | tr -dc 0-9)
[ "$shm" -ge 8 ] && [ "$disk" -ge 2 ] || { echo "w$WORKER: too little space: $shm GB free for $ROOT (8 needed), $disk GB on the disk (2 needed)"; exit 1; }
[ -e "$Q.exit" ] && mv "$Q.exit" "$Q.exit.$(date +%s)"

cat > "$RUN" <<EOF
set -uo pipefail
export TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 TPU_PROCESS_BOUNDS=1,1,1 TPU_VISIBLE_CHIPS=0,1,2,3 TPU_PROCESS_PORT=8476
export TPU_PROCESS_ADDRESSES=localhost:8476 CLOUD_TPU_TASK_ID=0 ALLOW_MULTIPLE_LIBTPU_LOAD=1
export HF_DATASETS_TRUST_REMOTE_CODE=1 TOKENIZERS_PARALLELISM=false PYTHONPATH=src
echo "\$(date '+%F %T') w$WORKER $NAME from $CKPT: $*"
cd \$HOME
[ -d $CODE/.git ] || git clone -q https://github.com/yhavinga/mbert-jax.git $CODE || exit 2
if [ "\$(git -C $CODE rev-parse HEAD)" != $COMMIT ]; then git -C $CODE fetch -q origin dumb-extra && git -C $CODE checkout -q $COMMIT || exit 2; fi
[ "\$(git -C $CODE rev-parse HEAD)" = $COMMIT ] || { echo "$CODE is not at $COMMIT"; exit 2; }
cd $CODE
$PY -c "import jax, flax, optax, orbax.checkpoint, transformers, datasets, pyarrow, tqdm" || { echo "$PY lacks a package of vm.lock"; exit 2; }
mkdir -p $ROOT/data $ROOT/runs \$HOME/dumb/jaxcache \$HOME/dumbx/jaxcache || exit 2
gsutil -q -m cp -r "$DATA/*" $ROOT/data/ || exit 2
$PY - $ROOT/data <<'PY' || exit 2
import hashlib, json, sys
from pathlib import Path
root = Path(sys.argv[1])
manifest = json.loads((root / "manifest.json").read_text())
files = [(root / folder / split["file"], split["sha256"]) for folder, d in manifest["datasets"].items() for split in d["splits"].values()]
bad = [str(path) for path, sha in files if hashlib.sha256(path.read_bytes()).hexdigest() != sha]
if bad:
    sys.exit(f"sha256 not that of manifest.json: {bad}")
print(f"data: {len(files)} files with the sha256 of manifest.json")
PY
gsutil -q cp $M/phase2048/scripts/verify_ckpt.py $ROOT/verify_ckpt.py || exit 2
[ "\$(md5sum \$HOME/heldout.parquet 2>/dev/null | cut -d' ' -f1)" = b2f579a876d06eef40c94270537bb7af ] || gsutil -q cp $M/heldout/nl_part_00097_first40000.parquet \$HOME/heldout.parquet
[ "\$(md5sum \$HOME/heldout.parquet | cut -d' ' -f1)" = b2f579a876d06eef40c94270537bb7af ] || { echo "held-out file is not the long run's"; exit 2; }
# Fail fast, before any compile, when this host is not a TPU of its own 4 chips: a job would wait for the other hosts
timeout 600 $PY -c "import jax, jax.numpy as jnp; n = jax.device_count(); assert (n, jax.process_count()) == (4, 1), (n, jax.process_count()); print('devices', jax.devices(), float(jnp.arange(4.0).sum()))" || { echo "preflight failed: not 4 chips of one process"; exit 3; }
CK=$ROOT/ckpt$NAME
if [ "$*" != blimp ] && [ ! -f \$CK/config.json ]; then
  # Converted and checked in a directory of its own, renamed to \$CK only when both passed
  tmp=$ROOT/convert-$NAME-\$(date +%s)
  $PY scripts/convert_gcs_checkpoint.py --checkpoint $CKPT --params-only --output \$tmp --tokenizer yhavinga/dutch-llama-tokenizer --global-attn-every-n-layers 3 < /dev/null || { echo "conversion of $NAME failed"; exit 4; }
  $PY - \$tmp/params.pkl <<'PY' || exit 4
import pickle, re, sys
extra = sorted(n for n in pickle.load(open(sys.argv[1], "rb")) if re.fullmatch(r"(prediction_head|decoder)_[0-9]+", n))
if extra:
    sys.exit(f"{sys.argv[1]}: more than one MLM head ({extra}); the arms of the final run have one")
PY
  $PY $ROOT/verify_ckpt.py \$tmp < /dev/null || { echo "verify_ckpt.py failed on $NAME (\$tmp)"; exit 4; }
  mv \$tmp \$CK || exit 4
fi
failed=0
for spec in $*; do
  task=\${spec%%:*}; seed=\${spec#*:}
  case \$task in
    dbrd1024|sicknl) job=$NAME--\$task--seed\$seed; result=$OUT/dumb/\$job.json; logs=$OUT/dumb/logs ;;
    blimp) job=blimp_$NAME; result=$OUT/blimp-nl/$NAME.json; logs=$OUT/blimp-nl/logs ;;
    *) job=$NAME--\$task--seed\$seed; result=$OUT/dumb-extra/\$task/\$job.json; logs=$OUT/dumb-extra/\$task/logs ;;
  esac
  gsutil -q stat \$result < /dev/null && { echo "\$job: a result exists, skipped"; continue; }
  echo "\$(date +%H:%M:%S) \$job"
  case \$task in
    dbrd1024) JAX_COMPILATION_CACHE_DIR=\$HOME/dumb/jaxcache $PY -u scripts/dumb/finetune.py --checkpoint \$CK --task dbrd --output $ROOT/runs/\$job --epochs 3 --batch-size 12 --gradient-accumulation-steps 4 --learning-rate 5e-5 --max-length 1024 --pad-to-multiple-of 128 --seed \$seed ;;
    sicknl) JAX_COMPILATION_CACHE_DIR=\$HOME/dumb/jaxcache $PY -u scripts/dumb/finetune.py --checkpoint \$CK --task sicknl --output $ROOT/runs/\$job --epochs 3 --batch-size 64 --gradient-accumulation-steps 1 --learning-rate 5e-5 --max-length 512 --pad-to-multiple-of 1 --seed \$seed ;;
    squadnl) JAX_COMPILATION_CACHE_DIR=\$HOME/dumbx/jaxcache $PY -u scripts/dumb/finetune_extra.py --checkpoint \$CK --task squadnl --data-dir $ROOT/data --output $ROOT/runs/\$job --epochs 3 --batch-size 8 --gradient-accumulation-steps 1 --learning-rate 5e-5 --lr-schedule linear --warmup-steps 0 --pad-to-multiple-of 128 --max-length 384 --doc-stride 128 --seed \$seed ;;
    blimp) $PY -u scripts/eval_blimp_nl.py --checkpoint $CKPT --data $ROOT/data/blimp-nl/large.parquet --tokenizer yhavinga/dutch-llama-tokenizer --hidden-size 1024 --num-layers 22 --num-heads 16 --intermediate-size 1536 --global-attn-every-n-layers 3 --export-head 0 --output \$result ;;
    *) JAX_COMPILATION_CACHE_DIR=\$HOME/dumbx/jaxcache $PY -u scripts/dumb/finetune_extra.py --checkpoint \$CK --task \$task --data-dir $ROOT/data --output $ROOT/runs/\$job --epochs 3 --batch-size 8 --gradient-accumulation-steps 1 --learning-rate 5e-5 --lr-schedule linear --warmup-steps 0 --pad-to-multiple-of 128 --max-length 512 --seed \$seed ;;
  esac > $LOGDIR/\$job.log 2>&1 < /dev/null
  code=\$?
  echo "EXIT \$code" >> $LOGDIR/\$job.log
  gsutil -q cp $LOGDIR/\$job.log \$logs/\${job#blimp_}.log < /dev/null
  if [ \$task = blimp ]; then
    if [ \$code = 0 ] && gsutil -q stat \$result < /dev/null; then grep -ah "^BLIMP accuracy" $LOGDIR/\$job.log; else echo "FAILED \$job with exit \$code"; failed=\$((failed + 1)); fi
  elif [ \$code = 0 ] && [ -f $ROOT/runs/\$job/results.json ] && gsutil -q cp $ROOT/runs/\$job/results.json \$result < /dev/null; then
    grep -ah "Peak memory\|^Test" $LOGDIR/\$job.log | tail -2
  else
    echo "FAILED \$job with exit \$code"; failed=\$((failed + 1))
  fi
done
echo "w$WORKER $NAME: \$failed failed"
[ \$failed = 0 ]
EOF
old_pid=$(cat "$Q.pid" 2>/dev/null || true)
setsid nohup bash -c "echo \$\$ > $Q.pid; bash $RUN >> $Q.log 2>&1; echo EXIT \$? > $Q.exit; gsutil -q cp $Q.log $Q.exit $OUT/eval/logs/$NAME/" > /dev/null 2>&1 < /dev/null &
for _ in $(seq 20); do [ -s "$Q.pid" ] && [ "$(cat "$Q.pid")" != "$old_pid" ] && break; sleep 0.5; done
echo "started $(cat "$Q.pid" 2>/dev/null) on $(hostname): $*, log $Q.log"
