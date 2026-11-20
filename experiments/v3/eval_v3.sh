#!/bin/bash
# On one worker of a v4-32, started by v3_ablation_run.sh on co1: this worker's share of the evaluations of one arm of
# the shape ablation of dmbert v3 (experiments/v3/protocol.md, part 2), detached from ssh. experiments/final-v2/eval_v2.sh
# (the evaluation of v2, which repeats T2's) with the same commands, code, venv, data and result layout, but:
#   - the model's shape from SHAPE (the trainer's flags --hidden-size --num-layers --num-heads --intermediate-size
#     --global-attn-every-n-layers): the conversion takes its global-attention frequency (the rest it reads from the
#     parameters) and BLiMP-NL all of them; v2's script had A's shape written in;
#   - no memtest and no heldout jobs (v2's own), so no ~/mbert-v2;
#   - results under OUT (default gs://t5-dutch-english-usc2/mbert/v3-ablation).
#   COMMIT=<dumb-extra code commit> SHAPE="--hidden-size H ..." bash eval_v3.sh NAME CKPT JOB...
# JOB is <task>:<seed> with task dbrd1024, dbrd512 or sicknl (scripts/dumb/finetune.py), squadnl, conll2002,
# pos-lassysmall or pos-alpino (scripts/dumb/finetune_extra.py), or "blimp" (scripts/eval_blimp_nl.py); settings as in
# eval_v2.sh. The fine-tunes and BLiMP-NL use v2's tokenizer (yhavinga/dutch-llama-tokenizer): for text its ids are
# those of tokenizer (a), so the evaluation measures the shape alone, the same way for every arm.
# Results: $OUT/dumb/NAME--{dbrd1024,dbrd512,sicknl}--seed<n>.json, $OUT/dumb-extra/<task>/NAME--<task>--seed<n>.json,
# $OUT/blimp-nl/NAME.json; queue files $LOGDIR/NAME_w<N>.{log,pid,exit}. Prints "started <pid>", "running <pid>" or a
# refusal (exit code 1). Environment overrides, for the dry run only: WORKER CODE PY ROOT OUT LOGDIR DATA M.
set -euo pipefail
[ $# -ge 3 ] || { echo "usage: COMMIT=<commit> SHAPE=\"--hidden-size H ...\" bash eval_v3.sh NAME CKPT JOB..."; exit 1; }
NAME=$1; CKPT=$2; shift 2
COMMIT=${COMMIT:?set COMMIT to the dumb-extra code commit}
M=${M:-gs://t5-dutch-english-usc2/mbert}
OUT=${OUT:-$M/v3-ablation}
SHAPE=${SHAPE:?set SHAPE to the trainer's shape flags}
[[ "$SHAPE" =~ ^--hidden-size\ [0-9]+\ --num-layers\ [0-9]+\ --num-heads\ [0-9]+\ --intermediate-size\ [0-9]+\ --global-attn-every-n-layers\ [0-9]+$ ]] || { echo "SHAPE is not the five shape flags in order: $SHAPE"; exit 1; }
GLOBAL=${SHAPE##* }
DATA=${DATA:-$M/dumb-extra/data}
CODE=${CODE:-$HOME/mbert-dumbx}
PY=${PY:-$HOME/mbert-dumb/.venv/bin/python}
ROOT=${ROOT:-/dev/shm/final_eval}
LOGDIR=${LOGDIR:-$HOME/logs/final_eval}
WORKER=${WORKER:-$(hostname | sed 's/.*-w-//')}
BUSY='[t]rain_minimal.py|[f]inetune.py|[f]inetune_extra.py|[e]val_windows.py|[e]val_zeroshot_mlm.py|[e]val_blimp_nl.py|[m]emtest.py|[v]erify_ckpt.py|[c]onvert_gcs_checkpoint.py|[p]robe_memory.py|[t]ranslate.py|[j]udge.py|[m]etricx_qe.py|[c]omet_score.py|[n]aturalness.py|[s]ft_lora.py'
[[ "$WORKER" =~ ^[0-9]+$ ]] || { echo "$(hostname): no worker number in the hostname; set WORKER"; exit 1; }
[[ "$NAME" =~ ^[a-z0-9][a-z0-9-]*$ ]] || { echo "w$WORKER: not a name: $NAME"; exit 1; }
[[ "$CKPT" == gs://* ]] || { echo "w$WORKER: not a gs:// step directory: $CKPT"; exit 1; }
for job in "$@"; do
  [[ "$job" =~ ^(dbrd1024|dbrd512|sicknl|squadnl|conll2002|pos-lassysmall|pos-alpino):(42|43|44)$ || "$job" =~ ^blimp$ ]] || {
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
FINETUNES=$(printf '%s\n' "$@" | grep -c ':' || true)
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
if [ $FINETUNES -gt 0 ] && [ ! -f \$CK/config.json ]; then
  # Converted and checked in a directory of its own, renamed to \$CK only when both passed
  tmp=$ROOT/convert-$NAME-\$(date +%s)
  $PY scripts/convert_gcs_checkpoint.py --checkpoint $CKPT --params-only --output \$tmp --tokenizer yhavinga/dutch-llama-tokenizer --global-attn-every-n-layers $GLOBAL < /dev/null || { echo "conversion of $NAME failed"; exit 4; }
  $PY - \$tmp/params.pkl <<'PY' || exit 4
import pickle, re, sys
extra = sorted(n for n in pickle.load(open(sys.argv[1], "rb")) if re.fullmatch(r"(prediction_head|decoder)_[0-9]+", n))
if extra:
    sys.exit(f"{sys.argv[1]}: more than one MLM head ({extra}); the arms of the ablation have one")
PY
  $PY $ROOT/verify_ckpt.py \$tmp < /dev/null || { echo "verify_ckpt.py failed on $NAME (\$tmp)"; exit 4; }
  mv \$tmp \$CK || exit 4
fi
failed=0
for spec in $*; do
  task=\${spec%%:*}; seed=\${spec#*:}
  case \$task in
    dbrd1024|dbrd512|sicknl) job=$NAME--\$task--seed\$seed; result=$OUT/dumb/\$job.json; logs=$OUT/dumb/logs ;;
    blimp) job=blimp_$NAME; result=$OUT/blimp-nl/$NAME.json; logs=$OUT/blimp-nl/logs ;;
    *) job=$NAME--\$task--seed\$seed; result=$OUT/dumb-extra/\$task/\$job.json; logs=$OUT/dumb-extra/\$task/logs ;;
  esac
  gsutil -q stat \$result < /dev/null && { echo "\$job: a result exists, skipped"; continue; }
  echo "\$(date +%H:%M:%S) \$job"
  case \$task in
    dbrd1024) JAX_COMPILATION_CACHE_DIR=\$HOME/dumb/jaxcache $PY -u scripts/dumb/finetune.py --checkpoint \$CK --task dbrd --output $ROOT/runs/\$job --epochs 3 --batch-size 12 --gradient-accumulation-steps 4 --learning-rate 5e-5 --max-length 1024 --pad-to-multiple-of 128 --seed \$seed ;;
    dbrd512) JAX_COMPILATION_CACHE_DIR=\$HOME/dumb/jaxcache $PY -u scripts/dumb/finetune.py --checkpoint \$CK --task dbrd --output $ROOT/runs/\$job --epochs 3 --batch-size 12 --gradient-accumulation-steps 4 --learning-rate 5e-5 --max-length 512 --pad-to-multiple-of 128 --seed \$seed ;;
    sicknl) JAX_COMPILATION_CACHE_DIR=\$HOME/dumb/jaxcache $PY -u scripts/dumb/finetune.py --checkpoint \$CK --task sicknl --output $ROOT/runs/\$job --epochs 3 --batch-size 64 --gradient-accumulation-steps 1 --learning-rate 5e-5 --max-length 512 --pad-to-multiple-of 1 --seed \$seed ;;
    squadnl) JAX_COMPILATION_CACHE_DIR=\$HOME/dumbx/jaxcache $PY -u scripts/dumb/finetune_extra.py --checkpoint \$CK --task squadnl --data-dir $ROOT/data --output $ROOT/runs/\$job --epochs 3 --batch-size 8 --gradient-accumulation-steps 1 --learning-rate 5e-5 --lr-schedule linear --warmup-steps 0 --pad-to-multiple-of 128 --max-length 384 --doc-stride 128 --seed \$seed ;;
    blimp) $PY -u scripts/eval_blimp_nl.py --checkpoint $CKPT --data $ROOT/data/blimp-nl/large.parquet --tokenizer yhavinga/dutch-llama-tokenizer $SHAPE --export-head 0 --output \$result ;;
    *) JAX_COMPILATION_CACHE_DIR=\$HOME/dumbx/jaxcache $PY -u scripts/dumb/finetune_extra.py --checkpoint \$CK --task \$task --data-dir $ROOT/data --output $ROOT/runs/\$job --epochs 3 --batch-size 8 --gradient-accumulation-steps 1 --learning-rate 5e-5 --lr-schedule linear --warmup-steps 0 --pad-to-multiple-of 128 --max-length 512 --seed \$seed ;;
  esac > $LOGDIR/\$job.log 2>&1 < /dev/null
  code=\$?
  echo "EXIT \$code" >> $LOGDIR/\$job.log
  gsutil -q cp $LOGDIR/\$job.log \$logs/\$job.log < /dev/null
  if [ \$task = blimp ]; then
    if [ \$code = 0 ] && gsutil -q stat \$result < /dev/null; then grep -ah "^BLIMP accuracy" $LOGDIR/\$job.log | cut -c1-300; else echo "FAILED \$job with exit \$code"; failed=\$((failed + 1)); fi
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
