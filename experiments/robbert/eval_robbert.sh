#!/bin/bash
# On one worker of mbert-mix (v4-32, us-central2-b), started by mbert-ops/scripts/robbert_eval_run.sh on co1: this
# worker's share of the comparison with RobBERT (experiments/robbert/protocol.md), detached from ssh, one job after the
# other on this worker's own TPU of 4 chips (the variables of eval_final.sh).
#   COMMIT=<commit of this branch> bash eval_robbert.sh JOB...
# JOB is <model>:<task>:<seed> or <model>:blimp, with
#   model  robbert-2023   DTAI-KULeuven/robbert-2023-dutch-base @ 1164b2df (RoBERTa), scripts/convert_hf_encoder.py
#          robbert-2026   pdelobelle/robbert-2026 @ a406b1a3 (NeoBERT), scripts/convert_hf_encoder.py
#          robbert-2023-rawoffsets  RobBERT-2023 with its tokenizer's shifted offsets left as they are (squadnl:42 only,
#                         a diagnostic: what they cost; hf_encoders.load_tokenizer)
#          epoch2-1306000 T2, the final run's model (DBRD@512 only), converted as eval_final.sh does and fine-tuned with
#                         the code of all its other evaluations (ae11478 in ~/mbert-dumbx), not this branch's
#   task   dbrd512, sicknl, squadnl, conll2002, pos-lassysmall, pos-alpino; seed 42, 43 or 44
# The commands are eval_final.sh's (protocol.md "Evaluatie" of the final run, extra evals of dumb-extra) but DBRD at
# --max-length 512, the most RoBERTa's learned positions hold, with the update of DBRD@1024 (12 x 4 chips x 4 = 192):
#   dbrd512   finetune.py --task dbrd --max-length 512, 12 x 4 x 4 = 192, padding to a multiple of 128, LR 5e-5
#   sicknl    finetune.py --task sicknl --max-length 512, 64 x 4 x 1 = 256, no padding, LR 5e-5
#             both 3 epochs, constant LR, AdamW with weight decay 0.01, test_accuracy after epoch 3
#   extra     finetune_extra.py, 8 x 4 x 1 = 32, LR 5e-5 falling linearly to 0 without warmup, 3 epochs, padding to
#             128; squadnl --max-length 384 --doc-stride 128, the token tasks --max-length 512
#   blimp     scripts/eval_blimp_nl_hf.py, PLL-word-l2r over the 8,400 pairs, float32
# Code: this branch at COMMIT in ~/mbert-robbert, T2's in ~/mbert-dumbx at ae11478; venv ~/mbert-dumb/.venv (nothing is
# installed). Converted models, data and run directories in $ROOT (/dev/shm/robbert_eval); compile caches
# ~/robbert/jaxcache (RobBERT) and ~/dumb/jaxcache (T2, as its DBRD@1024).
# Results (a job with a result is skipped, so a second start continues), under $OUT = gs://.../mbert/final/robbert:
#   $OUT/dumb/<model>--{dbrd512,sicknl}--seed<n>.json   $OUT/dumb-extra/<task>/<model>--<task>--seed<n>.json
#   $OUT/blimp-nl/<model>.json                          each job's log in logs/ next to them, ending in EXIT <code>
# Queue files: $LOGDIR/robbert_w<N>.{log,pid,exit} (~/logs/robbert_eval/), the log and exit file at the end also in
# $OUT/eval/logs/. Prints "started <pid>", "running <pid>" when the queue already runs here, or a refusal (exit 1).
# Environment overrides, for a smoke test only: WORKER OUT ROOT LOGDIR EXTRA_FT (e.g. "--max-train-examples 64
# --max-eval-examples 64", finetune_extra.py only) EXTRA_BLIMP (e.g. "--max-pairs 100").
set -euo pipefail
[ $# -ge 1 ] || { echo "usage: COMMIT=<commit> bash eval_robbert.sh JOB..."; exit 1; }
COMMIT=${COMMIT:?set COMMIT to the commit of branch robbert-compare}
T2_COMMIT=ae114789bf2eae153e4e3bea579299a617dcd417
M=gs://t5-dutch-english-usc2/mbert
OUT=${OUT:-$M/final/robbert}
DATA=$M/dumb-extra/data
T2_CKPT=$M/final-culturax-230m-epoch2/ckpt_keep/1306000
CODE=$HOME/mbert-robbert
T2_CODE=$HOME/mbert-dumbx
PY=$HOME/mbert-dumb/.venv/bin/python
ROOT=${ROOT:-/dev/shm/robbert_eval}
LOGDIR=${LOGDIR:-$HOME/logs/robbert_eval}
WORKER=${WORKER:-$(hostname | sed 's/.*-w-//')}
EXTRA_FT=${EXTRA_FT:-}
EXTRA_BLIMP=${EXTRA_BLIMP:-}
BUSY='[t]rain_minimal.py|[f]inetune.py|[f]inetune_extra.py|[e]val_windows.py|[e]val_zeroshot_mlm.py|[e]val_blimp_nl.py|[e]val_blimp_nl_hf.py|[v]erify_ckpt.py|[c]onvert_gcs_checkpoint.py|[c]onvert_hf_encoder.py|[p]robe_memory.py|[t]ranslate.py|[j]udge.py|[m]etricx_qe.py|[c]omet_score.py|[n]aturalness.py|[s]ft_lora.py'
[[ "$WORKER" =~ ^[0-9]+$ ]] || { echo "$(hostname): no worker number in the hostname; set WORKER"; exit 1; }
for job in "$@"; do
  [[ "$job" =~ ^robbert-202[36]:((dbrd512|sicknl|squadnl|conll2002|pos-lassysmall|pos-alpino):(42|43|44)|blimp)$ || "$job" =~ ^epoch2-1306000:dbrd512:(42|43|44)$
     || "$job" == robbert-2023-rawoffsets:squadnl:42 ]] || {
    echo "w$WORKER: not a job (<model>:<task>:<seed> or <model>:blimp): $job"; exit 1; }
done
mkdir -p "$LOGDIR"
Q=$LOGDIR/robbert_w$WORKER
RUN=$HOME/run_robbert_eval_w$WORKER.sh
if [ -s "$Q.pid" ] && [ ! -e "$Q.exit" ]; then
  pid=$(cat "$Q.pid")
  if [ -r "/proc/$pid/cmdline" ] && tr '\0' ' ' < "/proc/$pid/cmdline" | grep -q "run_robbert_eval_w$WORKER"; then
    echo "running $pid"; exit 0
  fi
fi
busy=$(pgrep -af "$BUSY" | cut -c1-160 || true)
[ -z "$busy" ] || { echo "w$WORKER: the TPU is in use: $(echo "$busy" | tr '\n' ' ')"; exit 1; }
holders=$(sudo lsof -t /dev/accel* /dev/vfio/* 2>/dev/null | sort -u | tr '\n' ' ' || true)
[ -z "${holders// /}" ] || { echo "w$WORKER: the TPU is held by: $holders"; exit 1; }
[ -x "$PY" ] || { echo "w$WORKER: no $PY"; exit 1; }
[ -s "$HOME/.cache/huggingface/token" ] || { echo "w$WORKER: no ~/.cache/huggingface/token (put it there through stdin)"; exit 1; }
mkdir -p "$ROOT"
shm=$(df -BG --output=avail "$ROOT" | tail -1 | tr -dc 0-9); disk=$(df -BG --output=avail "$HOME" | tail -1 | tr -dc 0-9)
[ "$shm" -ge 16 ] && [ "$disk" -ge 2 ] || { echo "w$WORKER: too little space: $shm GB free for $ROOT (16 needed), $disk GB on the disk (2 needed)"; exit 1; }
[ -e "$Q.exit" ] && mv "$Q.exit" "$Q.exit.$(date +%s)"

cat > "$RUN" <<EOF
set -uo pipefail
export TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 TPU_PROCESS_BOUNDS=1,1,1 TPU_VISIBLE_CHIPS=0,1,2,3 TPU_PROCESS_PORT=8476
export TPU_PROCESS_ADDRESSES=localhost:8476 CLOUD_TPU_TASK_ID=0 ALLOW_MULTIPLE_LIBTPU_LOAD=1
export HF_DATASETS_TRUST_REMOTE_CODE=1 TOKENIZERS_PARALLELISM=false PYTHONPATH=src
echo "\$(date '+%F %T') w$WORKER robbert eval, code $COMMIT: $*"
cd \$HOME
checkout() {  # checkout <dir> <branch> <commit>
  [ -d \$1/.git ] || git clone -q https://github.com/yhavinga/mbert-jax.git \$1 || return 1
  if [ "\$(git -C \$1 rev-parse HEAD)" != \$3 ]; then git -C \$1 fetch -q origin \$2 && git -C \$1 checkout -q \$3 || return 1; fi
  [ "\$(git -C \$1 rev-parse HEAD)" = \$3 ] || { echo "\$1 is not at \$3"; return 1; }
}
checkout $CODE robbert-compare $COMMIT || exit 2
case "$*" in *epoch2-1306000*) checkout $T2_CODE dumb-extra $T2_COMMIT || exit 2 ;; esac
$PY -c "import jax, flax, optax, transformers, datasets, safetensors, huggingface_hub, pyarrow, tqdm" || { echo "$PY lacks a package"; exit 2; }
mkdir -p $ROOT/data $ROOT/runs $ROOT/hf \$HOME/robbert/jaxcache \$HOME/dumb/jaxcache || exit 2
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
# Fail fast, before any compile, when this host is not a TPU of its own 4 chips: a job would wait for the other hosts
timeout 600 $PY -c "import jax, jax.numpy as jnp; n = jax.device_count(); assert (n, jax.process_count()) == (4, 1), (n, jax.process_count()); print('devices', jax.devices(), float(jnp.arange(4.0).sum()))" || { echo "preflight failed: not 4 chips of one process"; exit 3; }
convert() {  # convert <model>: the converted model in $ROOT/ckpt-<model>, once per worker
  local ck=$ROOT/ckpt-\$1 tmp=$ROOT/convert-\$1-\$(date +%s)
  [ -f \$ck/config.json ] && return 0
  case \$1 in
    robbert-2023) (cd $CODE && $PY scripts/convert_hf_encoder.py --model DTAI-KULeuven/robbert-2023-dutch-base --revision 1164b2df6d3ceb1735900c839da8de9bad1a99f9 --cache-dir $ROOT/hf --output \$tmp < /dev/null) || return 1 ;;
    robbert-2023-rawoffsets) (cd $CODE && $PY scripts/convert_hf_encoder.py --model DTAI-KULeuven/robbert-2023-dutch-base --revision 1164b2df6d3ceb1735900c839da8de9bad1a99f9 --keep-tokenizer-offsets --cache-dir $ROOT/hf --output \$tmp < /dev/null) || return 1 ;;
    robbert-2026) (cd $CODE && $PY scripts/convert_hf_encoder.py --model pdelobelle/robbert-2026 --revision a406b1a3eabf41401293e2a096167729bbd4d479 --cache-dir $ROOT/hf --output \$tmp < /dev/null) || return 1 ;;
    epoch2-1306000)
      if [ -f /dev/shm/final_eval/ckptepoch2-1306000/config.json ]; then
        # eval_final.sh's conversion for T2's own evaluations (the same command), verified there: read in place
        ln -s /dev/shm/final_eval/ckptepoch2-1306000 \$ck; return
      fi
      gsutil -q cp $M/phase2048/scripts/verify_ckpt.py $ROOT/verify_ckpt.py || return 1
      [ "\$(md5sum \$HOME/heldout.parquet 2>/dev/null | cut -d' ' -f1)" = b2f579a876d06eef40c94270537bb7af ] || gsutil -q cp $M/heldout/nl_part_00097_first40000.parquet \$HOME/heldout.parquet || return 1
      (cd $T2_CODE && $PY scripts/convert_gcs_checkpoint.py --checkpoint $T2_CKPT --params-only --output \$tmp --tokenizer yhavinga/dutch-llama-tokenizer --global-attn-every-n-layers 3 < /dev/null) || return 1
      (cd $T2_CODE && $PY $ROOT/verify_ckpt.py \$tmp < /dev/null) || return 1 ;;
  esac
  mv \$tmp \$ck
}
failed=0
for spec in $*; do
  model=\${spec%%:*}; rest=\${spec#*:}; task=\${rest%%:*}; seed=\${rest#*:}
  case \$task in
    dbrd512|sicknl) job=\$model--\$task--seed\$seed; result=$OUT/dumb/\$job.json; logs=$OUT/dumb/logs ;;
    blimp) job=\$model--blimp; result=$OUT/blimp-nl/\$model.json; logs=$OUT/blimp-nl/logs ;;
    *) job=\$model--\$task--seed\$seed; result=$OUT/dumb-extra/\$task/\$job.json; logs=$OUT/dumb-extra/\$task/logs ;;
  esac
  gsutil -q stat \$result < /dev/null && { echo "\$job: a result exists, skipped"; continue; }
  if ! convert \$model >> $LOGDIR/convert-\$model.log 2>&1; then
    echo "FAILED \$job: conversion of \$model (last lines of $LOGDIR/convert-\$model.log: \$(tail -3 $LOGDIR/convert-\$model.log | tr '\n' ' '))"; failed=\$((failed + 1)); continue
  fi
  CK=$ROOT/ckpt-\$model
  echo "\$(date +%H:%M:%S) \$job"
  if [ \$model = epoch2-1306000 ]; then cd $T2_CODE; cache=\$HOME/dumb/jaxcache; else cd $CODE; cache=\$HOME/robbert/jaxcache; fi
  case \$task in
    dbrd512) JAX_COMPILATION_CACHE_DIR=\$cache $PY -u scripts/dumb/finetune.py --checkpoint \$CK --task dbrd --output $ROOT/runs/\$job --epochs 3 --batch-size 12 --gradient-accumulation-steps 4 --learning-rate 5e-5 --max-length 512 --pad-to-multiple-of 128 --seed \$seed ;;
    sicknl) JAX_COMPILATION_CACHE_DIR=\$cache $PY -u scripts/dumb/finetune.py --checkpoint \$CK --task sicknl --output $ROOT/runs/\$job --epochs 3 --batch-size 64 --gradient-accumulation-steps 1 --learning-rate 5e-5 --max-length 512 --pad-to-multiple-of 1 --seed \$seed ;;
    squadnl) JAX_COMPILATION_CACHE_DIR=\$cache $PY -u scripts/dumb/finetune_extra.py --checkpoint \$CK --task squadnl --data-dir $ROOT/data --output $ROOT/runs/\$job --epochs 3 --batch-size 8 --gradient-accumulation-steps 1 --learning-rate 5e-5 --lr-schedule linear --warmup-steps 0 --pad-to-multiple-of 128 --max-length 384 --doc-stride 128 --seed \$seed $EXTRA_FT ;;
    blimp) mkdir -p $ROOT/runs/\$job && $PY -u scripts/eval_blimp_nl_hf.py --checkpoint \$CK --data $ROOT/data/blimp-nl/large.parquet --output $ROOT/runs/\$job/results.json $EXTRA_BLIMP ;;
    *) JAX_COMPILATION_CACHE_DIR=\$cache $PY -u scripts/dumb/finetune_extra.py --checkpoint \$CK --task \$task --data-dir $ROOT/data --output $ROOT/runs/\$job --epochs 3 --batch-size 8 --gradient-accumulation-steps 1 --learning-rate 5e-5 --lr-schedule linear --warmup-steps 0 --pad-to-multiple-of 128 --max-length 512 --seed \$seed $EXTRA_FT ;;
  esac > $LOGDIR/\$job.log 2>&1 < /dev/null
  code=\$?
  cd \$HOME
  echo "EXIT \$code" >> $LOGDIR/\$job.log
  gsutil -q cp $LOGDIR/\$job.log \$logs/\$job.log < /dev/null
  if [ \$code = 0 ] && [ -f $ROOT/runs/\$job/results.json ] && gsutil -q cp $ROOT/runs/\$job/results.json \$result < /dev/null; then
    grep -ah "^Test\|^BLIMP accuracy" $LOGDIR/\$job.log | tail -2
  else
    echo "FAILED \$job with exit \$code"; failed=\$((failed + 1))
  fi
done
echo "w$WORKER robbert eval: \$failed failed"
[ \$failed = 0 ]
EOF
old_pid=$(cat "$Q.pid" 2>/dev/null || true)
setsid nohup bash -c "echo \$\$ > $Q.pid; bash $RUN >> $Q.log 2>&1; echo EXIT \$? > $Q.exit; gsutil -q cp $Q.log $Q.exit $OUT/eval/logs/" > /dev/null 2>&1 < /dev/null &
for _ in $(seq 20); do [ -s "$Q.pid" ] && [ "$(cat "$Q.pid")" != "$old_pid" ] && break; sleep 0.5; done
echo "started $(cat "$Q.pid" 2>/dev/null) on $(hostname): $*, log $Q.log"
