#!/bin/bash
# The policy side of arm G4-7 (protocol-g4.md §14, amendment G4-7 and its addenda) on a worker, one stage per call;
# the chain (g4arm7_run.sh, co1) starts the stages and waits for their EXIT lines. The judge side (C2's data, training
# and check, the judges) is run_g4arm7_judges.sh; the candidates' evaluation is run_g4arm.sh with ARM=G4-7.
# A stage that takes more than a few seconds runs detached from ssh with a fresh log ~/logs/<job>.log that ends in
# "EXIT <code>" (copied to $RL/logs/<job>-w<k>.log); the jobs are named g4a7_<what>_$TAG.
# mbert-corpus (the policy: 4 workers, one multi-host job over 16 chips for the training):
#   bash run_g4arm7.sh cpuproof7       w1 (CPU, nice 19): cpuproof7.sh (spec §8.2), CPU_PROOF7_OK
#   bash run_g4arm7.sh cands7          w1 (CPU, nice 19): g4rl.py cands7 -> $RL/data/cands/
#   bash run_g4arm7.sh pool7           w1 (CPU, nice 19): g4rl.py pool7 -> $RL/data/pool/
#   JOB=anchor|iter/<i>|val/<i> [PLAN=gs://...] bash run_g4arm7.sh segments   (any worker, CPU, at once): g4rl.py
#                                      segments7 -> $RL/<job>/segments/ (anchor: every source of the pool)
#   JOB=... [ADAPTER=gs://dir] bash run_g4arm7.sh generate   each writer: translate.py --copies on its segments, G4-0
#                                      with the adapter merged (none: G4-0), class medium; anchor and val greedy (1 copy),
#                                      iter/<i> 8 samples at T = 1 (seed g4rl.sample_seed(i, k)) -> $RL/<job>/samples/w<k>/
#                                      (val: mbert-mix w3, its segments-w0.parquet)
#   JOB=... WRITERS=0,1,2,3 bash run_g4arm7.sh wait      w1 (CPU): g4judge.py wait --full (SCORES_OK)
#   bash run_g4arm7.sh alive           w1 (CPU, at once): g4judge.py alive (JUDGES_ALIVE)
#   bash run_g4arm7.sh plan7           w1 (CPU): g4rl.py plan7 on the scored anchor -> $RL/data/plan/
#   I=<i> [PLAN=gs://...] bash run_g4arm7.sh iter7       w1 (CPU): g4rl.py iter7 -> $RL/iter/<i>/train/
#   I=<i> RUN=<run> PREV=gs://.../runs/<prev run> bash run_g4arm7.sh resume7   w0: the previous run's Tunix checkpoint
#                                      of its last step into RUN's checkpoint directory (Tunix restores it)
#   I=<i> RUN=<run> [PREV=...] [TIS=1] [EVALS=1] PHASE=prep|launch bash run_g4arm7.sh train7   every worker:
#                                      sft_lora.py --grpo over the 16 chips (run_sft.sh MULTIHOST=1, ROOT=$RL/iter/<i>)
#   I=<i> RUN=<run> bash run_g4arm7.sh trained7          every worker (w0 first): the run's files in the bucket
#                                      checked, then this worker's own copies of them removed (paths in the log)
#   I=<i> [PREV_STOP=gs://...] SUMMARY=gs://... bash run_g4arm7.sh stop7   w1 (CPU): g4rl.py stop7 -> $RL/iter/<i>/stop.json
# The smoke test (spec §8.3, amendment G4-7.9; STAG names its directory $RL/smoke7/$STAG/):
#   STAG=... BASE=<N> bash run_g4arm7.sh smokeplan7      w1 (CPU, at once): g4rl.py smokeplan7 -> $RL/smoke7/$STAG/plan.parquet
#   STAG=... MODEL=G4-0|S [ADAPTER=gs://dir] bash run_g4arm7.sh smoke-translate | smoke-score   w0: the 50 WMT24++
#                                      smoke segments, class small, greedy; collect and MetricX-24 -> $RL/smoke7/$STAG/translate/<M>/
#   STAG=... I=<N> bash run_g4arm7.sh smoke7-loose       mbert-mix w3: MetricX-24-QE and C2 apart on iteration N's samples
#   STAG=... I=<N> AB=A|B bash run_g4arm7.sh check7      w1 (CPU, the SFT venv): g4rl.py check7 on the iteration's file
#   STAG=... NA=<N> RUN1=... RUN2=... LOGS_A=... LOGS_B=... bash run_g4arm7.sh smoke7-report   w1 (CPU): g4rl.py smoke7
# The code is beside this script (a copy of $G/scripts-<c7>/); TAG names the jobs of one chain.
set -euo pipefail

R=gs://t5-dutch-english-usc2/mbert/translator
S=$R/sft
E=$R/eval
G=$S/g4arm
H=$G/data/heldout
RL=${RL:-$G/rl/G4-7}
SHM=/dev/shm/translator
G_SHM=$SHM/g4arm
W7=$SHM/g4arm7-policy                                          # this script's own work directories
SFT_PY=$SHM/smoke-lora/venv/bin/python                         # setup_sft.sh: MaxText 0.2.5, Tunix
GEN=/dev/shm/translate-generic
FORK_COMMIT=f4d8615f478b6c18de9df8f2dff1f57f775a2521           # protocol-g4.md §9: every Gemma 4 decode
FPY=$GEN/venv-fork-${FORK_COMMIT:0:9}/bin/python
TOK_SFT=$SHM/g4lora/tok-gemma4-31b
TOK_DEC=$GEN/tokenizer-gemma4
DECODE_CKPT=$R/judge/gemma-4-31b-it-maxtext/0/items
MX_ROOT=$SHM/metricx
DPY=$HOME/mbert-dumb/.venv/bin/python
DUMB=$HOME/mbert-dumbx
DUMB_COMMIT=ae114789bf2eae153e4e3bea579299a617dcd417
C2_W=$SHM/g4arm7
CODE=$(cd "$(dirname "$0")" && pwd)
W=$(hostname | sed 's/.*-w-//')
TAG=${TAG:-manual}
JUDGES="--judge metricx:w0:0 --judge metricx:w2:1 --judge c2:w1"
export PATH=$HOME/.local/bin:$PATH
mkdir -p "$G_SHM" "$W7" ~/logs

detach() {  # detach NAME COMMAND...: the command in the background, log ~/logs/NAME.log ending in EXIT <code>
  local name=$1
  shift
  [ -e ~/logs/$name.log ] && { echo "w$W: ~/logs/$name.log exists; a new start takes a new TAG"; exit 1; }
  printf '%q ' "$@" > ~/job_$name.sh
  setsid nohup bash -c "bash ~/job_$name.sh > ~/logs/$name.log 2>&1; echo EXIT \$? >> ~/logs/$name.log; gsutil -q cp ~/logs/$name.log $RL/logs/$name-w$W.log" \
    > /dev/null 2>&1 < /dev/null &
  echo "started $name on w$W"
}

tpu_free() {
  [ -z "$(sudo lsof -t /dev/accel* /dev/vfio/* 2> /dev/null)" ] || { echo "w$W: the TPU is in use"; exit 1; }
}

only() {  # only WORKER...: the stage runs on these workers alone
  local k
  for k in "$@"; do [ "$W" = "$k" ] && return 0; done
  echo "$STAGE runs on w$* only"; exit 1
}

absent() {  # absent GS_OBJECT: a write-once output is not there yet
  ! gsutil -q stat "$1" || { echo "$1 exists: G4-7's outputs are written once"; exit 1; }
}

g7() {  # g7 NAME: a constant of g4rl.py, so that the stages and the code agree
  $SFT_PY -I -c "import sys; sys.path.insert(0, '$CODE'); import g4rl; print(g4rl.$1)"
}

body() {  # body NAME: the job's script from stdin, returns its path
  local f=~/job_body_$1.sh
  cat > "$f"
  echo "$f"
}

# A worker as a TPU of its own 4 chips (run_g4arm.sh, run_arms.sh)
ONE_HOST="export TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 TPU_PROCESS_BOUNDS=1,1,1 TPU_VISIBLE_CHIPS=0,1,2,3 TPU_PROCESS_PORT=8476
export TPU_PROCESS_ADDRESSES=localhost:8476 CLOUD_TPU_TASK_ID=0 ALLOW_MULTIPLE_LIBTPU_LOAD=1
export XLA_FLAGS=--xla_dump_hlo_as_text=false TPU_LOG_DIR=$SHM/tpu_logs TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 HF_HUB_DISABLE_PROGRESS_BARS=1"
PREFLIGHT="timeout 600 $FPY -c \"import jax; assert (jax.device_count(), jax.process_count()) == (4, 1), (jax.device_count(), jax.process_count()); print('devices', jax.devices())\" || { echo 'preflight failed: not 4 chips of one process'; exit 2; }"
CPU="export JAX_PLATFORMS=cpu OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1"

job_tag() {  # job_tag JOB: anchor, i<i> or v<i>, for the job names
  case $1 in anchor) echo anchor ;; iter/*) echo "i${1#iter/}" ;; val/*) echo "v${1#val/}" ;; esac
}

check_job() {  # check_job JOB
  case $1 in anchor|iter/[0-9]*|val/[0-9]*) ;; *) echo "JOB is anchor, iter/<i> or val/<i>, not $1"; exit 1 ;; esac
  case $1 in iter/*|val/*) [[ ${1#*/} =~ ^[0-9]+$ ]] || { echo "JOB $1: <i> is a number"; exit 1; } ;; esac
}

STAGE=${1:?a stage: cpuproof7 cands7 pool7 segments generate wait alive plan7 iter7 resume7 train7 trained7 stop7 smokeplan7 smoke-translate smoke-score smoke7-loose check7 smoke7-report}
case $STAGE in
cpuproof7)
  X=$G_SHM/cpuproof7-$TAG
  [ ! -e "$X" ] || { echo "w$W: $X exists"; exit 1; }
  f=$(body g4a7_cpuproof7_$TAG <<EOF
set -euo pipefail
# the CPU alone (JAX_PLATFORMS=cpu): the TPU is never opened
env PY=$SFT_PY TOK=$TOK_SFT CODE=$CODE X=$X JAX_PLATFORMS=cpu nice -n 19 ionice -c3 bash $CODE/cpuproof7.sh
EOF
)
  detach "g4a7_cpuproof7_$TAG" bash "$f"
  ;;
cands7)
  absent "$RL/data/cands/cands.json"
  X=$W7/cands7-$TAG
  f=$(body g4a7_cands7_$TAG <<EOF
set -euo pipefail
$CPU
mkdir -p $X/in $X/out
CANDS7_FETCH
nice -n 19 ionice -c3 $SFT_PY -I -u $CODE/g4rl.py cands7 CANDS7_ARGS --out $X/out
gsutil -m -q cp $X/out/cands-flan.parquet $X/out/cands-novels.parquet $X/out/cands-bt.parquet $X/out/cands.json $RL/data/cands/
echo "CANDS7_DONE $RL/data/cands/"
EOF
)
  detach "g4a7_cands7_$TAG" bash "$f"
  ;;
pool7)
  absent "$RL/data/pool/pool.parquet"
  for f in cands-flan.parquet cands-novels.parquet cands-bt.parquet; do
    gsutil -q stat "$RL/data/cands/$f" || { echo "no $RL/data/cands/$f (cands7 first)"; exit 1; }
  done
  gsutil -q stat "$RL/c2/data/docs.txt" || { echo "no $RL/c2/data/docs.txt (g4c2.py data7 before pool7, G4-7.2)"; exit 1; }
  X=$W7/pool7-$TAG
  f=$(body g4a7_pool7_$TAG <<EOF
set -euo pipefail
$CPU
mkdir -p $X/in $X/cands $X/out
gsutil -m -q cp $RL/data/cands/cands-flan.parquet $RL/data/cands/cands-novels.parquet $RL/data/cands/cands-bt.parquet $X/cands/
gsutil -m -q cp $E/data/evalsets.parquet $S/blind/fixed-2026-11-10/blind100.jsonl $R/mainset/flan-eval/eval-rows.parquet $RL/c2/data/docs.txt $X/in/
nice -n 19 ionice -c3 $SFT_PY -I -u $CODE/g4rl.py pool7 --cands $X/cands --evalsets $X/in/evalsets.parquet --blind $X/in/blind100.jsonl --c2-docs $X/in/docs.txt --flan-eval $X/in/eval-rows.parquet --tokenizer $TOK_SFT --out $X/out
gsutil -m -q cp $X/out/pool.parquet $X/out/pool.json $RL/data/pool/
echo "POOL7_DONE $RL/data/pool/"
EOF
)
  detach "g4a7_pool7_$TAG" bash "$f"
  ;;
segments)
  JOB=${JOB:?anchor, iter/<i> or val/<i>}
  check_job "$JOB"
  if gsutil -q stat "$RL/$JOB/segments/segments-w0.parquet"; then
    # segments7 is deterministic: the segments of an earlier start stay
    echo "SEGMENTS_DONE $JOB (there before)"
    exit 0
  fi
  X=$W7/segments-$(job_tag "$JOB")-$TAG
  mkdir -p "$X/out"
  case $JOB in
    anchor) gsutil -q cp "$RL/data/pool/pool.parquet" "$X/pool.parquet"; ARGS="--pool $X/pool.parquet"; N=4 ;;
    iter/*) gsutil -q cp "${PLAN:-$RL/data/plan/plan.parquet}" "$X/plan.parquet"; ARGS="--plan $X/plan.parquet --iteration ${JOB#iter/}"; N=4 ;;
    val/*) gsutil -q cp "${PLAN:-$RL/data/plan/plan.parquet}" "$X/plan.parquet"; ARGS="--plan $X/plan.parquet --iteration -1"; N=1 ;;
  esac
  # shellcheck disable=SC2086
  JAX_PLATFORMS=cpu nice -n 19 $SFT_PY -I -u "$CODE/g4rl.py" segments7 $ARGS --workers $N --out "$X/out"
  gsutil -m -q cp "$X"/out/segments-w*.parquet "$RL/$JOB/segments/"
  echo "SEGMENTS_DONE $JOB"
  ;;
generate)
  JOB=${JOB:?anchor, iter/<i> or val/<i>}
  check_job "$JOB"
  ADAPTER=${ADAPTER:-none}
  tpu_free
  T=$(job_tag "$JOB")
  NAME=g4a7_gen_${T}_$TAG
  case $JOB in
    # the anchor (G4-7.4a): G4-0's greedy translation of every source of the pool and RL-val
    anchor) [ "$ADAPTER" = none ] || { echo "the anchor is G4-0's (no adapter)"; exit 1; }
            K=$W; SEG=segments-w$W.parquet; OUT=$RL/anchor/samples/w$K; EXTRA="--copies 1"; SS=512 ;;
    # an iteration: G = 8 samples per source at T = 1, the worker's own seed (addendum G4-7.4c), the shards of 32
    # sources (a multiple of 8: a source's samples in one shard)
    iter/*) i=${JOB#iter/}
            K=$W; SEG=segments-w$W.parquet; OUT="$RL/$JOB/{sample}/w$K"
            SEED=$($SFT_PY -I -c "import sys; sys.path.insert(0, '$CODE'); import g4rl; print(g4rl.sample_seed($i, $W))")
            EXTRA="--copies $(g7 G7) --sample samples=1.0:$SEED"; SS=256 ;;
    # RL-val after an iteration: greedy with the new adapter, on mbert-mix w3 (the writer 3 of G4-7.4a)
    val/*) only 3
           K=3; SEG=segments-w0.parquet; OUT=$RL/$JOB/samples/w3; EXTRA="--copies 1"; SS=128 ;;
  esac
  ! gsutil -q stat "${OUT//\{sample\}/samples}/DONE" || { echo "${OUT//\{sample\}/samples}/DONE exists: generated before"; exit 1; }
  gsutil -q stat "$RL/$JOB/segments/$SEG" || { echo "no $RL/$JOB/segments/$SEG (segments first)"; exit 1; }
  LORA=""
  if [ "$ADAPTER" != none ]; then
    for f in adapter.npz adapter.json; do gsutil -q stat "$ADAPTER/$f" || { echo "no $ADAPTER/$f"; exit 1; }; done
    LORA="--lora $G_SHM/adapter-$NAME"
  fi
  f=$(body "$NAME" <<EOF
set -uo pipefail
$ONE_HOST
export JAX_COMPILATION_CACHE_DIR=$G_SHM/jax_cache
mkdir -p $G_SHM/seg-$NAME $G_SHM/work
gsutil -q cp $RL/$JOB/segments/$SEG $G_SHM/seg-$NAME/$SEG || exit 1
$PREFLIGHT
if [ -n "$LORA" ]; then
  mkdir -p $G_SHM/adapter-$NAME
  gsutil -q cp $ADAPTER/adapter.npz $ADAPTER/adapter.json $G_SHM/adapter-$NAME/ || exit 1
  python3 -c "import json, sys; a = json.load(open(sys.argv[1])); assert (a['rank'], a['alpha'], a['scale']) == (32, 16.0, 0.5), a" $G_SHM/adapter-$NAME/adapter.json || { echo "the adapter is not rank 32, alpha 16"; exit 1; }
fi
echo "== $JOB w$K $(date -u +%T) adapter $ADAPTER"
# no fallback on memory (amendment G4-7.5: an OOM is FAILED): the class's own slots
$FPY -u $CODE/translate.py --segments $G_SHM/seg-$NAME/$SEG --out '$OUT' --ckpt $DECODE_CKPT --tokenizer $TOK_DEC --model gemma4-31b --prompt natural-v1 --length-class $(g7 LENGTH_CLASS7) --source-lang en --target-lang nl --work-dir $G_SHM/work --shard-size $SS $EXTRA $LORA
code=\$?
if [ -n "$LORA" ]; then
  # this job's own copy of the adapter (~1 GB; the bucket has it)
  echo "removing $G_SHM/adapter-$NAME"
  rm -rf -- $G_SHM/adapter-$NAME
fi
[ \$code = 0 ] || exit \$code
gsutil -q stat '${OUT//\{sample\}/samples}/DONE' || { echo "no DONE in ${OUT//\{sample\}/samples}"; exit 1; }
echo "GENERATED w$K $JOB"
EOF
)
  detach "$NAME" bash "$f"
  ;;
wait)
  JOB=${JOB:?anchor, iter/<i> or val/<i>}
  check_job "$JOB"
  WRITERS=${WRITERS:?0,1,2,3 (val: 3)}
  f=$(body "g4a7_wait_$(job_tag "$JOB")_$TAG" <<EOF
set -euo pipefail
$CPU
# --full: the uids of the scores against the samples' (reads every file, in the bucket's region)
nice -n 19 $FPY -I -u $CODE/g4judge.py wait --rl $RL --job $JOB --writers $WRITERS $JUDGES --full
EOF
)
  detach "g4a7_wait_$(job_tag "$JOB")_$TAG" bash "$f"
  ;;
alive)
  # shellcheck disable=SC2086
  JAX_PLATFORMS=cpu timeout 300 $FPY -I -u "$CODE/g4judge.py" alive --rl "$RL" $JUDGES
  ;;
plan7)
  absent "$RL/data/plan/plan.parquet"
  X=$W7/plan7-$TAG
  f=$(body g4a7_plan7_$TAG <<EOF
set -euo pipefail
$CPU
mkdir -p $X/out
gsutil -q cp $RL/data/pool/pool.parquet $X/pool.parquet
gsutil -m -q rsync -r $RL/anchor/samples $X/samples
gsutil -m -q rsync -r $RL/anchor/scores $X/scores
mkdir -p $X/pool && mv $X/pool.parquet $X/pool/pool.parquet
nice -n 19 $SFT_PY -I -u $CODE/g4rl.py plan7 --pool $X/pool --anchor-samples $X/samples --anchor-scores $X/scores --out $X/out
gsutil -m -q cp $X/out/plan.parquet $X/out/plan.json $RL/data/plan/
echo "PLAN7_DONE $RL/data/plan/"
EOF
)
  detach "g4a7_plan7_$TAG" bash "$f"
  ;;
iter7)
  I=${I:?the iteration}
  PLAN=${PLAN:-$RL/data/plan/plan.parquet}
  absent "$RL/iter/$I/train/report.json"
  X=$W7/iter7-$I-$TAG
  f=$(body g4a7_iter7_${I}_$TAG <<EOF
set -euo pipefail
$CPU
mkdir -p $X/out
gsutil -q cp $PLAN $X/plan.parquet
gsutil -m -q rsync -r $RL/iter/$I/samples $X/samples
gsutil -m -q rsync -r $RL/iter/$I/scores $X/scores
nice -n 19 $SFT_PY -I -u $CODE/g4rl.py iter7 --plan $X/plan.parquet --iteration $I --samples $X/samples --scores $X/scores --tokenizer $TOK_SFT --out $X/out
gsutil -m -q cp $X/out/train.parquet $X/out/samples.parquet $X/out/report.json $RL/iter/$I/train/
echo "ITER7_DONE $RL/iter/$I/train/"
EOF
)
  detach "g4a7_iter7_${I}_$TAG" bash "$f"
  ;;
resume7)
  only 0
  I=${I:?the iteration}; RUN=${RUN:?the run}; PREV=${PREV:?gs://.../runs/<the previous run>}
  P=$(basename "$PREV")
  gsutil -q cp "$PREV/summary.json" "$W7/resume7-$RUN.json"
  step=$(python3 -c "import json, sys; print(json.load(open(sys.argv[1]))['steps'])" "$W7/resume7-$RUN.json")
  SRC=$PREV/maxtext/$P/checkpoints/$step
  DST=$RL/iter/$I/runs/$RUN/maxtext/$RUN/checkpoints
  gsutil -q stat "$SRC/commit_success.txt" || { echo "no $SRC/commit_success.txt"; exit 1; }
  [ -z "$(gsutil ls "$RL/iter/$I/runs/$RUN/" 2> /dev/null)" ] || { echo "$RL/iter/$I/runs/$RUN/ exists: a new run takes a new name"; exit 1; }
  gsutil -m -q cp -r "$SRC" "$DST/"
  gsutil -q stat "$DST/$step/commit_success.txt" || { echo "the copy of $SRC is incomplete"; exit 1; }
  echo "RESUME7 $RUN: $P step $step, $(gsutil du -s "$DST/$step" | awk '{print $1}') bytes"
  ;;
train7)
  I=${I:?the iteration}; RUN=${RUN:?the run}
  TRAIN=${TRAIN:-$RL/iter/$I/train/train.parquet}
  ROOT=$RL/iter/$I
  D=$G_SHM/data-$RUN
  case ${PHASE:-} in
    launch) [ -s "$D/launch.sh" ] || { echo "w$W: no $D/launch.sh (PHASE=prep first)"; exit 1; }
            bash "$D/launch.sh"; exit 0 ;;
    prep) ;;
    *) echo "PHASE is prep or launch, not ${PHASE:-}"; exit 1 ;;
  esac
  tpu_free
  # a launch never takes the arguments of an earlier or a refused preparation
  rm -f -- "$D/launch.sh"
  mkdir -p "$D"
  gsutil -q stat "$TRAIN" || { echo "no $TRAIN (iter7 first)"; exit 1; }
  ! gsutil -q stat "$ROOT/runs/$RUN/summary.json" || { echo "$ROOT/runs/$RUN/ has a summary.json: a new run takes a new name"; exit 1; }
  # MaxText's data hooks are made from DATA and never read (sft_lora.py --grpo replaces the iterator): G4-6b's chosen
  # sides and template, as G4-6c's multi-host runs had them; the SFT evals' template is the same (sft_build.py)
  gsutil -m -q cp "$TRAIN" "$D/train.parquet"
  gsutil -m -q cp "$G/data/G4-6b/train-chosen.parquet" "$G/data/G4-6b/template.jinja" "$D/"
  $SFT_PY -I -c "import json, sys; sys.path.insert(0, '$CODE'); import g4rl; json.dump(g4rl.grpo_spec(tis=bool(int(sys.argv[2]))), open(sys.argv[1], 'w'))" "$D/grpo.json" "${TIS:-0}"
  ARGS="--grpo $D/grpo.json --grpo-train $D/train.parquet --warmup-steps $(g7 WARMUP7) --decay-fraction 0"
  if [ -n "${PREV:-}" ]; then
    # the optimizer state continued (G4-5): the previous run's last step, which resume7 copied into this run's place
    gsutil -q cp "$PREV/summary.json" "$D/prev-summary.json"
    gsutil -q cp "$PREV/adapter.npz" "$PREV/adapter.json" "$D/"
    mv "$D/adapter.npz" "$D/resume-adapter.npz"; mv "$D/adapter.json" "$D/resume-adapter.json"
    python3 -c "import json, sys; a = json.load(open(sys.argv[1])); assert (a['rank'], a['alpha'], a['scale']) == (32, 16.0, 0.5), a" "$D/resume-adapter.json" \
      || { echo "the previous adapter is not rank 32, alpha 16"; exit 1; }
    resume=$(python3 -c "import json, sys; print(json.load(open(sys.argv[1]))['steps'])" "$D/prev-summary.json")
    gsutil -q stat "$ROOT/runs/$RUN/maxtext/$RUN/checkpoints/$resume/commit_success.txt" \
      || { echo "no checkpoint of step $resume in $ROOT/runs/$RUN/maxtext/$RUN/checkpoints/ (resume7 first)"; exit 1; }
    ARGS="$ARGS --resume-step $resume --resume-adapter $D/resume-adapter.npz"
  fi
  if [ "${EVALS:-0}" = 1 ]; then
    # spec §6.3, G4-7.10: books, btval and own every 5th iteration, before its first and after its last step
    for f in eval-books.parquet eval-btval.parquet eval-own.parquet; do gsutil -q stat "$H/$f" || { echo "no $H/$f"; exit 1; }; done
    gsutil -m -q cp "$H/eval-books.parquet" "$H/eval-btval.parquet" "$H/eval-own.parquet" "$D/"
    ARGS="$ARGS --eval books=$D/eval-books.parquet --eval btval=$D/eval-btval.parquet --eval own=$D/eval-own.parquet --eval-every 1000000"
  fi
  # PHASE=prep stops here and PHASE=launch starts run_sft.sh from this file: the processes of the multi-host job meet in
  # jax.distributed.initialize within a timeout, so a chain prepares every worker first (G4-5, G4-6)
  printf '%q ' env SFT_ARGS="$ARGS" MULTIHOST=1 NAME="$RUN" MODEL=gemma4-31b DATA="$D/train-chosen.parquet" \
    TEMPLATE="$D/template.jinja" ROOT="$ROOT" STEPS=auto bash "$CODE/run_sft.sh" per_device_batch_size=1 \
    max_target_length="$(g7 SEQ7)" lora.lora_rank=32 lora.lora_alpha=16.0 learning_rate="$(g7 LR7)" \
    lr_schedule_type=wsd wsd_decay_style=linear learning_rate_final_fraction=0.1 > "$D/launch.sh.new"
  mv "$D/launch.sh.new" "$D/launch.sh"
  echo "prepared $RUN on w$W"
  ;;
trained7)
  I=${I:?the iteration}; RUN=${RUN:?the run}
  OUT=$SHM/smoke-lora/runs/$RUN
  D=$G_SHM/data-$RUN
  if [ "$W" = 0 ]; then
    # worker 0 wrote the run's files: each in the bucket with its own size before this worker's copy goes
    for f in summary.json adapter.npz adapter.json grpo-passes.npz grpo-passes.json; do
      [ -s "$OUT/$f" ] || { echo "w0: no $OUT/$f"; exit 1; }
      here=$(stat -c %s "$OUT/$f")
      there=$(gsutil du "$RL/iter/$I/runs/$RUN/$f" 2> /dev/null | awk '{print $1}')
      [ "$here" = "$there" ] || { echo "$RL/iter/$I/runs/$RUN/$f: $there bytes in the bucket, $here here"; exit 1; }
    done
    echo "removing $OUT (its files are in $RL/iter/$I/runs/$RUN/)"
    rm -rf -- "$OUT"
  fi
  if [ -e "$D/resume-adapter.npz" ]; then
    echo "removing $D/resume-adapter.npz (the previous run's adapter in the bucket)"
    rm -f -- "$D/resume-adapter.npz"
  fi
  echo "TRAINED7 w$W $RUN"
  ;;
stop7)
  I=${I:?the iteration}
  SUMMARY=${SUMMARY:?gs://.../runs/<run>/summary.json}
  PLAN=${PLAN:-$RL/data/plan/plan.parquet}
  OUTP=${STOP_OUT:-$RL/iter/$I/stop.json}
  VAL=${VAL:-val/$I}
  ITER=${ITER:-$I}
  absent "$OUTP"
  X=$W7/stop7-$I-$TAG
  prev=""
  [ -z "${PREV_STOP:-}" ] || prev="--previous $X/previous.json"
  f=$(body g4a7_stop7_${I}_$TAG <<EOF
set -euo pipefail
$CPU
mkdir -p $X
gsutil -q cp $PLAN $X/plan.parquet
gsutil -m -q rsync -r $RL/$VAL/samples $X/val-samples
gsutil -m -q rsync -r $RL/$VAL/scores $X/val-scores
gsutil -q cp $RL/iter/$ITER/train/report.json $X/iter-report.json
gsutil -q cp $SUMMARY $X/summary.json
if [ -n "${PREV_STOP:-}" ]; then gsutil -q cp ${PREV_STOP:-} $X/previous.json; fi
nice -n 19 $SFT_PY -I -u $CODE/g4rl.py stop7 --plan $X/plan.parquet --iteration $ITER --val-samples $X/val-samples --val-scores $X/val-scores --iter-report $X/iter-report.json --train-summary $X/summary.json $prev --out $X/stop.json
gsutil -q cp $X/stop.json $OUTP
echo "STOP7_DONE $OUTP"
EOF
)
  detach "g4a7_stop7_${I}_$TAG" bash "$f"
  ;;
smokeplan7)
  STAG=${STAG:?the tag of the smoke test}; BASE=${BASE:?the first iteration number of the smoke test, at least 900}
  absent "$RL/smoke7/$STAG/plan.parquet"
  X=$W7/smokeplan7-$STAG
  mkdir -p "$X/out"
  gsutil -q cp "$RL/data/plan/plan.parquet" "$X/plan.parquet"
  JAX_PLATFORMS=cpu nice -n 19 $SFT_PY -I -u "$CODE/g4rl.py" smokeplan7 --plan "$X/plan.parquet" --base "$BASE" --sources 32 --out "$X/out"
  gsutil -q cp "$X/out/plan.parquet" "$RL/smoke7/$STAG/plan.parquet"
  echo "SMOKEPLAN7_DONE $RL/smoke7/$STAG/plan.parquet"
  ;;
smoke-translate)
  only 0
  tpu_free
  STAG=${STAG:?the tag of the smoke test}; MODEL=${MODEL:?G4-0 or S}
  case $MODEL in G4-0|S) ;; *) echo "MODEL is G4-0 or S, not $MODEL"; exit 1 ;; esac
  NAME=g4a7_smoke_translate_${MODEL}_$TAG
  O=$RL/smoke7/$STAG/translate/$MODEL
  LORA=""
  if [ "$MODEL" = S ]; then
    ADAPTER=${ADAPTER:?gs://.../runs/<the second run of the smoke test>}
    LORA="--lora $G_SHM/adapter-$NAME"
  fi
  f=$(body "$NAME" <<EOF
set -uo pipefail
$ONE_HOST
export JAX_COMPILATION_CACHE_DIR=$G_SHM/jax_cache
mkdir -p $G_SHM/seg-$NAME $G_SHM/work
gsutil -q cp $G/data/eval/segments-smoke.parquet $G_SHM/seg-$NAME/segments-smoke.parquet || exit 1
$PREFLIGHT
if [ -n "$LORA" ]; then
  mkdir -p $G_SHM/adapter-$NAME
  gsutil -q cp ${ADAPTER:-}/adapter.npz ${ADAPTER:-}/adapter.json $G_SHM/adapter-$NAME/ || exit 1
fi
$FPY -u $CODE/translate.py --segments $G_SHM/seg-$NAME/segments-smoke.parquet --out $O/translations/small --ckpt $DECODE_CKPT --tokenizer $TOK_DEC --model gemma4-31b --prompt natural-v1 --length-class small --source-lang en --target-lang nl --work-dir $G_SHM/work $LORA
code=\$?
if [ -n "$LORA" ]; then echo "removing $G_SHM/adapter-$NAME"; rm -rf -- $G_SHM/adapter-$NAME; fi
[ \$code = 0 ] || exit \$code
echo "SMOKE_TRANSLATED $MODEL"
EOF
)
  detach "$NAME" bash "$f"
  ;;
smoke-score)
  only 0
  tpu_free
  STAG=${STAG:?the tag of the smoke test}; MODEL=${MODEL:?G4-0 or S}
  NAME=g4a7_smoke_score_${MODEL}_$TAG
  O=$RL/smoke7/$STAG/translate/$MODEL
  L=$W7/score-$NAME
  f=$(body "$NAME" <<EOF
set -euo pipefail
$ONE_HOST
export JAX_COMPILATION_CACHE_DIR=$G_SHM/jax_cache_metricx
mkdir -p $L/data $L/translations $L/out
gsutil -q cp $G/data/eval/evalsets.parquet $L/data/
gsutil -q cp $G/data/eval/segments-smoke.parquet $L/data/segments-w0.parquet
gsutil -m -q rsync -r $O/translations $L/translations
$FPY -I $CODE/evalsets.py collect --data $L/data --worker 0 --shards $L/translations --out $L/out/pairs-w0.parquet
timeout 600 $FPY -c "import jax; assert (jax.device_count(), jax.process_count()) == (4, 1), (jax.device_count(), jax.process_count())"
$FPY -I -u $CODE/metricx_qe.py score --pairs $L/out/pairs-w0.parquet --pair source:hypothesis:reference --pair source:hypothesis --model xxl --root $MX_ROOT --empty-candidates --out $L/out/metricx-w0.parquet
gsutil -m -q cp $L/out/pairs-w0.parquet $L/out/metricx-w0.parquet $O/
echo "SMOKE_SCORED $MODEL"
EOF
)
  detach "$NAME" bash "$f"
  ;;
smoke7-loose)
  only 3
  tpu_free
  STAG=${STAG:?the tag of the smoke test}; I=${I:?the first iteration of the smoke test}
  [ -x "$DPY" ] && [ "$(git -C "$DUMB" rev-parse HEAD 2> /dev/null)" = "$DUMB_COMMIT" ] || { echo "w$W: no DUMB venv or $DUMB not at $DUMB_COMMIT"; exit 1; }
  X=$W7/loose-$STAG
  f=$(body g4a7_smoke7_loose_$TAG <<EOF
set -euo pipefail
$ONE_HOST
mkdir -p $X/out $X/c2
gsutil -m -q rsync -r $RL/iter/$I/samples $X/samples
gsutil -m -q cp $RL/c2/model/params.pkl $RL/c2/model/config.json $X/c2/
JAX_PLATFORMS=cpu nice -n 19 $SFT_PY -I -u $CODE/g4rl.py smoke7-loose --samples $X/samples --out $X/in
SMOKE7_LOOSE_METRICX
SMOKE7_LOOSE_C2
gsutil -m -q cp $X/out/* $RL/smoke7/$STAG/loose/
echo "SMOKE7_LOOSE_DONE $RL/smoke7/$STAG/loose/"
EOF
)
  detach "g4a7_smoke7_loose_$TAG" bash "$f"
  ;;
check7)
  STAG=${STAG:?the tag of the smoke test}; I=${I:?the iteration}; AB=${AB:?A or B}
  X=$W7/check7-$STAG-$AB
  f=$(body g4a7_check7_${AB}_$TAG <<EOF
set -euo pipefail
$CPU
mkdir -p $X
gsutil -q cp $RL/iter/$I/train/train.parquet $X/train.parquet
nice -n 19 $SFT_PY -I -u $CODE/g4rl.py check7 --train $X/train.parquet --tokenizer $TOK_SFT --out $X/check.json
gsutil -q cp $X/check.json $RL/smoke7/$STAG/check7-$AB.json
echo "CHECK7_DONE $AB"
EOF
)
  detach "g4a7_check7_${AB}_$TAG" bash "$f"
  ;;
smoke7-report)
  STAG=${STAG:?the tag of the smoke test}; NA=${NA:?the first iteration of the smoke test}; RUN1=${RUN1:?}; RUN2=${RUN2:?}
  NB=$((NA + 1))
  SM=$RL/smoke7/$STAG
  X=$W7/smoke7-$STAG
  D=$X/d
  f=$(body g4a7_smoke7_report_$TAG <<EOF
set -euo pipefail
$CPU
mkdir -p $D/iter-A $D/iter-B $D/train-1 $D/train-2 $D/logs $D/translate $D/val $D/heartbeats $D/loose $X/out
gsutil -q cp $SM/plan.parquet $D/plan.parquet
for ab in A:$NA B:$NB; do
  d=$D/iter-\${ab%%:*}; i=\${ab#*:}
  gsutil -m -q rsync -r $RL/iter/\$i/samples \$d/samples
  gsutil -m -q rsync -r $RL/iter/\$i/scores \$d/scores
  gsutil -m -q rsync -r $RL/iter/\$i/train \$d/train
done
gsutil -q cp $SM/check7-A.json $SM/check7-B.json $D/
gsutil -m -q cp "$SM/loose/*" $D/loose/
for k in 1:$NA:$RUN1 2:$NB:$RUN2; do
  n=\${k%%:*}; rest=\${k#*:}; i=\${rest%%:*}; run=\${rest#*:}
  r=$RL/iter/\$i/runs/\$run
  gsutil -q cp \$r/summary.json $D/train-\$n/summary.json
  step=\$(python3 -c "import json, sys; print(json.load(open(sys.argv[1]))['steps'])" $D/train-\$n/summary.json)
  ck=\$r/maxtext/\$run/checkpoints/\$step
  a=false; c=false
  gsutil -q stat \$r/adapter.npz && a=true
  gsutil -q stat \$ck/commit_success.txt && c=true
  echo "{\"adapter.npz\": \$a, \"checkpoint\": \"\$ck\", \"commit_success\": \$c}" > $D/train-\$n/files.json
done
k=0; for l in ${LOGS_A:-}; do gsutil -q cp \$l $D/logs/gen-A-w\$k.log; k=\$((k + 1)); done
k=0; for l in ${LOGS_B:-}; do gsutil -q cp \$l $D/logs/gen-B-w\$k.log; k=\$((k + 1)); done
for m in G4-0 S; do mkdir -p $D/translate/\$m; gsutil -m -q cp $SM/translate/\$m/pairs-w0.parquet $SM/translate/\$m/metricx-w0.parquet $D/translate/\$m/; done
gsutil -q cp $SM/val-stop.json $D/val/stop.json
gsutil -m -q cp "$RL/judges/*.json" $D/heartbeats/
python3 -c "import json, time; print(json.dumps({'epoch': time.time()}))" > $D/now.json
gsutil -q cp $SM/timings.json $D/timings.json
set +e
nice -n 19 $SFT_PY -I -u $CODE/g4rl.py smoke7 --dir $D --out $X/out
code=\$?
set -e
gsutil -m -q cp $X/out/smoke7.json $X/out/smoke7.md $SM/
exit \$code
EOF
)
  detach "g4a7_smoke7_report_$TAG" bash "$f"
  ;;
*)
  echo "unknown stage $STAGE"
  exit 2
  ;;
esac
