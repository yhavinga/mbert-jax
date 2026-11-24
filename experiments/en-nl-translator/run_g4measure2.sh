#!/bin/bash
# G4-maat2 and the evaluation pool (protocol-g4.md §24) on mbert-mix (4 x v4-8, us-central2-b), one stage per call on a
# worker; the chain g4measure2_run.sh (co1, mbert-ops) starts the stages and waits for their EXIT lines. Every stage runs
# detached from ssh with a fresh log ~/logs/<name>.log that ends in "EXIT <code>" (also copied to $GM2/logs/ or $P/logs/). The
# venvs, the tokenizers, MetricX-24, COMET-22 and the classifier's venv come from run_g4arm.sh setup (FULL=1 on w0, w2).
# Part 1 (G4-maat2):
#   bash run_g4measure2.sh data1          w0 (CPU): g4measure2.py data1 (A1 and J-nat of §20's and §23's pairs, the
#                                         reproduction's prompts) -> $GM2/data/
#   SET=m2|p1|p2 bash run_g4measure2.sh judge   every worker (its TPU): pairjudge.py score of its prompts (m2: $GM2,
#                                         p1: $P/judge1, p2: $P/judge2) -> <dir>/scores/w<k>/ (m2: $GM2/judge/scores)
# Part 2 (the pool):
#   bash run_g4measure2.sh pooldata       w0 (CPU): g4pool.py data -> $P/data/
#   JOB=samples bash run_g4measure2.sh generate              w0 (its TPU): G4-0's 8 samples per source at T = 1
#   JOB=greedy MODEL=G4-0|G4-8-s140|G4-7-i15 bash ... generate   a worker (its TPU): the model's greedy translation
#   bash run_g4measure2.sh texts          w0 (CPU): g4pool.py texts -> $P/texts/, $P/naturalness/texts.parquet
#   bash run_g4measure2.sh qe             every worker (its TPU): MetricX-24 XXL QE of its rows -> $P/scores/qe-w<k>
#   bash run_g4measure2.sh classify       w0 (its TPU): C1, run_arms.sh classify unchanged with NAT=$P/naturalness
#   bash run_g4measure2.sh classify-c2    w0 (its TPU): C2 as run_g4arm.sh classify-c2
#   bash run_g4measure2.sh comet          w2 (CPU): COMET-22 of the news texts -> $P/scores/comet.parquet
#   bash run_g4measure2.sh cands          w0 (CPU): g4pool.py cands -> $P/choose/cands.*, $P/judge1/data/
#   bash run_g4measure2.sh choose         w0 (CPU): g4pool.py choose -> $P/choose/, $P/judge2/data/
#   bash run_g4measure2.sh poolreport     w0 (CPU): g4pool.py report from the files without text -> $P/report.md/.json
# A stage whose output is in the bucket already prints that and ends in EXIT 0; nothing is removed or overwritten. The
# code is beside this script (a copy of $G/scripts-g4m2-<commit>/). TAG names the jobs of one chain.
set -euo pipefail
STAGE=${1:?a stage: data1 judge pooldata generate texts qe classify classify-c2 comet cands choose poolreport}
R=gs://t5-dutch-english-usc2/mbert/translator
S=$R/sft
G=$S/g4arm
GM=$G/measure                                                   # §23
GM2=$G/measure2                                                 # §24 part 1
P=$G/pool                                                       # §24 part 2
GJ=$G/judge
RL=$G/rl/G4-7
G8=$G/data/G4-8
MS=$G/mixsample
B=$S/blind
DECODE_CKPT=$R/judge/gemma-4-31b-it-maxtext/0/items            # the conversion every Gemma 4 decode uses
SHM=/dev/shm/translator
G_SHM=$SHM/g4arm
WX=$SHM/g4measure2                                              # this script's own work directories
GEN=/dev/shm/translate-generic
FORK_COMMIT=f4d8615f478b6c18de9df8f2dff1f57f775a2521             # protocol-g4.md §9: every Gemma 4 decode
FPY=$GEN/venv-fork-${FORK_COMMIT:0:9}/bin/python
SFT_PY=$SHM/smoke-lora/venv/bin/python                          # setup_sft.sh (run_g4arm.sh setup)
TOK_DEC=$GEN/tokenizer-gemma4
TOK_SFT=$SHM/g4lora/tok-gemma4-31b                              # the same tokenizer.json (run_g4arm.sh setup checks)
MX_ROOT=$SHM/metricx
WORK=$G_SHM/judge-work                                          # MaxText's output and the compilation cache
DPY=$HOME/mbert-dumb/.venv/bin/python                           # C1's and C2's venv and checkout (run_g4arm.sh)
DUMB=$HOME/mbert-dumbx
DUMB_COMMIT=ae114789bf2eae153e4e3bea579299a617dcd417
MODELS="G4-0 G4-8-s140 G4-7-i15"
CODE=$(cd "$(dirname "$0")" && pwd)
W=$(hostname | sed 's/.*-w-//')
TAG=${TAG:-manual}
mkdir -p "$G_SHM" "$WX" ~/logs
LOGDIR=$P/logs                                                  # part 1's stages set $GM2/logs

detach() {  # detach NAME COMMAND...: the command in the background, log ~/logs/NAME.log ending in EXIT <code>
  local name=$1
  shift
  [ -e ~/logs/$name.log ] && { echo "w$W: ~/logs/$name.log exists; a new start takes a new TAG"; exit 1; }
  printf '%q ' "$@" > ~/job_$name.sh
  setsid nohup bash -c "bash ~/job_$name.sh > ~/logs/$name.log 2>&1; echo EXIT \$? >> ~/logs/$name.log; gsutil -q cp ~/logs/$name.log $LOGDIR/$name-w$W.log" \
    > /dev/null 2>&1 < /dev/null &
  echo "started $name on w$W"
}

body() {  # body NAME: the job's script from stdin, returns its path
  local f=~/job_body_$1.sh
  cat > "$f"
  echo "$f"
}

there_before() {  # there_before NAME WHAT: a job that only says its output is in the bucket already
  local f
  f=$(body "$1" <<EOF
echo "$2"
EOF
)
  detach "$1" bash "$f"
  exit 0
}

tpu_free() {
  [ -z "$(sudo lsof -t /dev/accel* /dev/vfio/* 2> /dev/null)" ] || { echo "w$W: the TPU is in use"; exit 1; }
}

only() {  # only WORKER: the stage runs on that worker alone
  [ "$W" = "$1" ] || { echo "$STAGE runs on w$1 only"; exit 1; }
}

need() {  # need <bucket path>...: the data a stage reads
  local p
  for p in "$@"; do gsutil -q stat "$p" || { echo "no $p"; exit 1; }; done
}

ready() {
  [ -x "$FPY" ] && [ -x "$SFT_PY" ] && [ -s "$TOK_DEC/tokenizer.json" ] && [ -s "$TOK_SFT/tokenizer.json" ] \
    || { echo "w$W: no fork venv, SFT venv or tokenizer (run_g4arm.sh setup)"; exit 1; }
}

dumb_ready() {
  [ -x "$DPY" ] || { echo "w$W: no $DPY"; exit 1; }
  [ "$(git -C "$DUMB" rev-parse HEAD 2> /dev/null)" = "$DUMB_COMMIT" ] || { echo "w$W: $DUMB is not at $DUMB_COMMIT"; exit 1; }
  [ -z "$(git -C "$DUMB" status --porcelain --untracked-files=no)" ] || { echo "w$W: $DUMB has changes"; exit 1; }
}

# A worker of the v4-32 as a TPU of its own 4 chips (run_g4arm.sh)
ONE_HOST="export TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 TPU_PROCESS_BOUNDS=1,1,1 TPU_VISIBLE_CHIPS=0,1,2,3 TPU_PROCESS_PORT=8476
export TPU_PROCESS_ADDRESSES=localhost:8476 CLOUD_TPU_TASK_ID=0 ALLOW_MULTIPLE_LIBTPU_LOAD=1
export XLA_FLAGS=--xla_dump_hlo_as_text=false TPU_LOG_DIR=$SHM/tpu_logs TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 HF_HUB_DISABLE_PROGRESS_BARS=1"
ONE_HOST_DUMB="export TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 TPU_PROCESS_BOUNDS=1,1,1 TPU_VISIBLE_CHIPS=0,1,2,3 TPU_PROCESS_PORT=8476
export TPU_PROCESS_ADDRESSES=localhost:8476 CLOUD_TPU_TASK_ID=0 ALLOW_MULTIPLE_LIBTPU_LOAD=1
export XLA_FLAGS=--xla_dump_hlo_as_text=false TPU_LOG_DIR=$SHM/tpu_logs TOKENIZERS_PARALLELISM=false"
PREFLIGHT="timeout 600 $FPY -c \"import jax; assert (jax.device_count(), jax.process_count()) == (4, 1), (jax.device_count(), jax.process_count()); print('devices', jax.devices())\" || { echo 'preflight failed: not 4 chips of one process'; exit 2; }"
CPU="export JAX_PLATFORMS=cpu OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1"

case $STAGE in
data1)
  only 0
  ready
  LOGDIR=$GM2/logs
  NAME=g4m2_data1_$TAG
  gsutil -q stat "$GM2/data/data.json" && there_before "$NAME" "MEASURE2DATA_DONE $GM2/data/ (there before)"
  need "$GJ/fb/data/refs.parquet" "$GM/data/pairs.parquet"
  X=$WX/data1-$TAG
  f=$(body "$NAME" <<EOF
set -euo pipefail
$CPU
mkdir -p $X/fb $X/fbn1 $X/fbn2 $X/m23 $X/out
gsutil -m -q cp '$GJ/fb/data/prompts-w*.parquet' $GJ/fb/data/refs.parquet $X/fb/
gsutil -m -q cp '$GJ/fbn1/data/prompts-w*.parquet' $X/fbn1/
gsutil -m -q cp '$GJ/fbn2/data/prompts-w*.parquet' $X/fbn2/
gsutil -m -q cp '$GM/data/prompts-w*.parquet' $GM/data/pairs.parquet $X/m23/
nice -n 19 $SFT_PY -I -u $CODE/g4measure2.py data1 --fb-refs $X/fb/refs.parquet --fb-prompts '$X/fb/prompts-w*.parquet' --fb-n1-prompts '$X/fbn1/prompts-w*.parquet' --fb-n2-prompts '$X/fbn2/prompts-w*.parquet' --pairs $X/m23/pairs.parquet --m23-prompts '$X/m23/prompts-w*.parquet' --out $X/out
gsutil -m -q cp $X/out/prompts-w*.parquet $GM2/data/
gsutil -q cp $X/out/data.json $GM2/data/data.json
echo "MEASURE2DATA_DONE $GM2/data/"
EOF
)
  detach "$NAME" bash "$f"
  ;;
judge)
  SET=${SET:?SET=m2, p1 or p2}
  case $SET in
    m2) DATA=$GM2/data; OUT=$GM2/judge/scores; LOGDIR=$GM2/logs ;;
    p1) DATA=$P/judge1/data; OUT=$P/judge1/scores ;;
    p2) DATA=$P/judge2/data; OUT=$P/judge2/scores ;;
    *) echo "SET is m2, p1 or p2, not $SET"; exit 1 ;;
  esac
  ready
  NAME=g4m2_judge_${SET}_$TAG
  need "$DATA/prompts-w$W.parquet"
  tpu_free
  X=$WX/judge-$SET-$TAG
  # pairjudge.py score unchanged; it skips the shards in the bucket (a restart goes on where it stopped)
  f=$(body "$NAME" <<EOF
set -euo pipefail
$ONE_HOST
mkdir -p $X $WORK
gsutil -q cp $DATA/prompts-w$W.parquet $X/prompts.parquet
$PREFLIGHT
$FPY -u $CODE/pairjudge.py score --prompts $X/prompts.parquet --out $OUT/w$W --ckpt $DECODE_CKPT --tokenizer $TOK_DEC --work-dir $WORK
echo "JUDGE2_DONE w$W $SET $OUT/w$W/"
EOF
)
  detach "$NAME" bash "$f"
  ;;
pooldata)
  only 0
  ready
  NAME=g4p_data_$TAG
  gsutil -q stat "$P/data/data.json" && there_before "$NAME" "POOLDATA_DONE $P/data/ (there before)"
  X=$WX/pooldata-$TAG
  f=$(body "$NAME" <<EOF
set -euo pipefail
$CPU
mkdir -p $X/in/bt3 $X/in/keys $X/out
gsutil -m -q cp $G/data/heldout4/eval-novels-meta.parquet $S/data/books-b1/b1-selected.parquet $RL/data/pool/pool.parquet $RL/c2/data/classifier.parquet $X/in/
gsutil -q cp $G8/fresh/fresh.parquet $X/in/g48-fresh.parquet
gsutil -q cp $G8/plan/plan.parquet $X/in/g48-plan.parquet
gsutil -q cp $G8/targets/targets.parquet $X/in/g48-targets.parquet
gsutil -m -q rsync -r $G/input/bt3 $X/in/bt3
gsutil -m -q cp $G/data/eval/evalsets.parquet $G/data/G4-5-split/split.json $X/in/
gsutil -q cp $MS/data/data.json $X/in/mix-data.json
gsutil -q cp $B/steekproef-yeb-g4-8-mix/key.json $X/in/keys/mix-key.json
gsutil -q cp $B/steekproef-yeb/key-map.json $X/in/keys/yeb1-key.json
gsutil -q cp $B/G4-judge-i15/key.jsonl $X/in/keys/G4-judge-i15.jsonl
gsutil -q cp $B/G4-judge-anchor-i15/key.jsonl $X/in/keys/G4-judge-anchor-i15.jsonl
gsutil -q cp $B/steekproef-yeb-g4-8/key-map.json $X/in/keys/sample2-key.json
nice -n 19 ionice -c3 $SFT_PY -I -u $CODE/g4pool.py data --novels-meta $X/in/eval-novels-meta.parquet --b1 $X/in/b1-selected.parquet --g47-pool $X/in/pool.parquet --g48-fresh $X/in/g48-fresh.parquet --g48-plan $X/in/g48-plan.parquet --g48-targets $X/in/g48-targets.parquet --c2-data $X/in/classifier.parquet --bt $X/in/bt3 --evalsets $X/in/evalsets.parquet --split $X/in/split.json --mix-data $X/in/mix-data.json --mix-key $X/in/keys/mix-key.json --yeb1-key $X/in/keys/yeb1-key.json --package-key G4-judge-i15=$X/in/keys/G4-judge-i15.jsonl --package-key G4-judge-anchor-i15=$X/in/keys/G4-judge-anchor-i15.jsonl --sample2-key $X/in/keys/sample2-key.json --tokenizer $TOK_SFT --out $X/out
gsutil -m -q cp $X/out/sources.parquet $X/out/sources-meta.parquet $X/out/segments.parquet $P/data/
gsutil -q cp $X/out/data.json $P/data/data.json
echo "POOLDATA_DONE $P/data/"
EOF
)
  detach "$NAME" bash "$f"
  ;;
generate)
  JOB=${JOB:?JOB=samples or greedy}
  ready
  case $JOB in
    # Pool.3: G4-0's 8 samples per source at T = 1 (G4-8-data.3), shards of 32 sources
    samples) only 0; MODEL=G4-0; OUT="$P/{sample}/w$W"; DONE_AT=$P/samples/w$W/DONE; SS=256
      SEED=$(JAX_PLATFORMS=cpu $SFT_PY -I -c "import sys; sys.path.insert(0, '$CODE'); import g4rl; print(g4rl.seed_int('en-nl-g4arm/0/G4-maat2/pool/sample/w$W'))")
      EXTRA="--copies 8 --sample samples=1.0:$SEED" ;;
    # Pool.3: a model's greedy translation (as G4-8-data.2's anchor)
    greedy) MODEL=${MODEL:?G4-0, G4-8-s140 or G4-7-i15}
      case " $MODELS " in *" $MODEL "*) ;; *) echo "MODEL is one of $MODELS, not $MODEL"; exit 1 ;; esac
      OUT=$P/gen/$MODEL/w$W; DONE_AT=$OUT/DONE; SS=512; EXTRA="--copies 1" ;;
    *) echo "JOB is samples or greedy, not $JOB"; exit 1 ;;
  esac
  NAME=g4p_gen_${JOB}_${MODEL}_$TAG
  gsutil -q stat "$DONE_AT" && there_before "$NAME" "GENERATED w$W $JOB $MODEL (DONE there before)"
  tpu_free
  need "$P/data/segments.parquet"
  LORA=""
  if [ "$MODEL" != G4-0 ]; then
    # the model's LoRA, rank 32 / alpha 16 from G4-0 (nothing stacked); its copy stays (nothing removed)
    LORA=$WX/adapter-$NAME
    mkdir -p "$LORA"
    gsutil -q cp "$G/$MODEL/adapter.npz" "$G/$MODEL/adapter.json" "$LORA/"
    python3 -c "import json, sys; a = json.load(open(sys.argv[1])); assert (a['rank'], a['alpha'], a['scale']) == (32, 16.0, 0.5), a" "$LORA/adapter.json" \
      || { echo "$G/$MODEL/adapter.json is not rank 32, alpha 16"; exit 1; }
    EXTRA="$EXTRA --lora $LORA"
  fi
  f=$(body "$NAME" <<EOF
set -uo pipefail
$ONE_HOST
export JAX_COMPILATION_CACHE_DIR=$G_SHM/jax_cache
mkdir -p $WX/seg-$NAME $G_SHM/work
gsutil -q cp $P/data/segments.parquet $WX/seg-$NAME/segments.parquet || exit 1
$PREFLIGHT
echo "== $JOB $MODEL w$W \$(date -u +%T) ${LORA:+with its LoRA}${LORA:-(no adapter)}"
# no fallback on memory (as G4-8-data): the class's own slots; translate.py skips the shards already in the bucket
$FPY -u $CODE/translate.py --segments $WX/seg-$NAME/segments.parquet --out '$OUT' --ckpt $DECODE_CKPT --tokenizer $TOK_DEC --model gemma4-31b --prompt natural-v1 --length-class medium --source-lang en --target-lang nl --work-dir $G_SHM/work --shard-size $SS $EXTRA || exit \$?
gsutil -q stat '$DONE_AT' || { echo "no $DONE_AT"; exit 1; }
echo "GENERATED w$W $JOB $MODEL"
EOF
)
  detach "$NAME" bash "$f"
  ;;
texts)
  only 0
  ready
  NAME=g4p_texts_$TAG
  gsutil -q stat "$P/texts/texts.json" && there_before "$NAME" "POOLTEXTS_DONE $P/texts/ (there before)"
  need "$P/samples/w0/DONE" "$P/gen/G4-0/w1/DONE" "$P/gen/G4-8-s140/w2/DONE" "$P/gen/G4-7-i15/w3/DONE" "$GM/data/qe-in.parquet"
  X=$WX/texts-$TAG
  f=$(body "$NAME" <<EOF
set -euo pipefail
$CPU
mkdir -p $X/data $X/samples $X/out
gsutil -q cp $P/data/sources.parquet $X/data/
gsutil -m -q rsync -r $P/samples $X/samples
for m in $MODELS; do mkdir -p $X/gen/\$m $X/eval/\$m; gsutil -m -q rsync -r $P/gen/\$m $X/gen/\$m; gsutil -m -q cp "$G/eval/\$m/pairs/pairs-w*.parquet" $X/eval/\$m/; done
gsutil -q cp $GM/data/qe-in.parquet $X/m23-qe-in.parquet
nice -n 19 $SFT_PY -I -u $CODE/g4pool.py texts --data $X/data --samples $X/samples --greedy G4-0=$X/gen/G4-0 --greedy G4-8-s140=$X/gen/G4-8-s140 --greedy G4-7-i15=$X/gen/G4-7-i15 --eval 'G4-0=$X/eval/G4-0/pairs-w*.parquet' --eval 'G4-8-s140=$X/eval/G4-8-s140/pairs-w*.parquet' --eval 'G4-7-i15=$X/eval/G4-7-i15/pairs-w*.parquet' --m23-qe $X/m23-qe-in.parquet --out $X/out
gsutil -m -q cp $X/out/texts.parquet $X/out/texts-meta.parquet $X/out/comet-in.parquet $X/out/qe-in-w*.parquet $P/texts/
gsutil -q cp $X/out/nat-texts.parquet $P/naturalness/texts.parquet
gsutil -q cp $X/out/texts.json $P/texts/texts.json
echo "POOLTEXTS_DONE $P/texts/"
EOF
)
  detach "$NAME" bash "$f"
  ;;
qe)
  ready
  NAME=g4p_qe_$TAG
  gsutil -q stat "$P/scores/qe-w$W.parquet" && there_before "$NAME" "POOLQE_DONE w$W (there before)"
  need "$P/texts/qe-in-w$W.parquet"
  [ -d "$MX_ROOT/metricx-24-hybrid-xxl-v2p6-bfloat16" ] || { echo "w$W: no MetricX-24 XXL in $MX_ROOT (run_g4arm.sh setup)"; exit 1; }
  tpu_free
  X=$WX/qe-$TAG
  # MetricX-24 hybrid XXL in QE mode (no reference; an empty hypothesis without a score, as G4-8-data.2)
  f=$(body "$NAME" <<EOF
set -euo pipefail
$ONE_HOST
export JAX_COMPILATION_CACHE_DIR=$G_SHM/jax_cache_metricx
mkdir -p $X
gsutil -q cp $P/texts/qe-in-w$W.parquet $X/qe-in.parquet
$PREFLIGHT
$FPY -I -u $CODE/metricx_qe.py score --pairs $X/qe-in.parquet --pair source:hypothesis --model xxl --root $MX_ROOT --out $X/qe.parquet
gsutil -q cp $X/qe.parquet $P/scores/qe-w$W.parquet
echo "POOLQE_DONE w$W"
EOF
)
  detach "$NAME" bash "$f"
  ;;
classify)
  only 0
  need "$P/naturalness/texts.parquet"
  ! gsutil -q stat "$P/naturalness/predictions.parquet" || { echo "$P/naturalness/predictions.parquet exists: it stays as it is"; exit 1; }
  # C1: run_arms.sh's classify, unchanged, on the pool's texts; its job is arms_classify_<TAG>c
  NAT=$P/naturalness TAG=${TAG}c bash "$CODE/run_arms.sh" classify
  ;;
classify-c2)
  only 0
  NAME=g4p_classify_c2_$TAG
  gsutil -q stat "$P/naturalness/predictions-c2.parquet" && there_before "$NAME" "CLASSIFIED_C2 $P/naturalness/predictions-c2.parquet (there before)"
  need "$RL/c2/model/params.pkl" "$RL/c2/model/config.json" "$P/naturalness/texts.parquet"
  tpu_free
  dumb_ready
  X=$WX/classify-c2-$TAG
  # as run_g4arm.sh classify-c2 (amendment G4-7.8): C2 in C1's venv and checkout, --max-length 1024
  f=$(body "$NAME" <<EOF
set -euo pipefail
$ONE_HOST_DUMB
export JAX_COMPILATION_CACHE_DIR=$G_SHM/jax_cache_dumb PYTHONPATH=src
cd $DUMB
mkdir -p $X/model $G_SHM/jax_cache_dumb
gsutil -m -q cp $RL/c2/model/params.pkl $RL/c2/model/config.json $X/model/
gsutil -q cp $P/naturalness/texts.parquet $X/texts.parquet
$DPY -u $CODE/naturalness.py classify --dumb-code $DUMB --model $X/model --texts $X/texts.parquet --max-length 1024 --out $X/predictions-c2.parquet
gsutil -q cp $X/predictions-c2.parquet $P/naturalness/predictions-c2.parquet
echo "CLASSIFIED_C2 $P/naturalness/predictions-c2.parquet"
EOF
)
  detach "$NAME" bash "$f"
  ;;
comet)
  only 2
  NAME=g4p_comet_$TAG
  gsutil -q stat "$P/scores/comet.parquet" && there_before "$NAME" "POOLCOMET_DONE $P/scores/comet.parquet (there before)"
  need "$P/texts/comet-in.parquet"
  [ -x "$SHM/comet-venv/bin/python" ] || { echo "w$W: no COMET venv (run_g4arm.sh setup with FULL=1)"; exit 1; }
  X=$WX/comet-$TAG
  f=$(body "$NAME" <<EOF
set -euo pipefail
mkdir -p $X
gsutil -q cp $P/texts/comet-in.parquet $X/comet-in.parquet
nice -n 10 $SHM/comet-venv/bin/python -I -u $CODE/comet_score.py score --root $SHM/comet --pairs $X/comet-in.parquet --out $X/comet.parquet
gsutil -q cp $X/comet.parquet $P/scores/comet.parquet
echo "POOLCOMET_DONE $P/scores/comet.parquet"
EOF
)
  detach "$NAME" bash "$f"
  ;;
cands)
  only 0
  ready
  NAME=g4p_cands_$TAG
  gsutil -q stat "$P/choose/cands.json" && there_before "$NAME" "POOLCANDS_DONE $P/choose/ (there before)"
  need "$P/texts/texts.parquet" "$P/scores/qe-w0.parquet" "$P/scores/qe-w1.parquet" "$P/scores/qe-w2.parquet" "$P/scores/qe-w3.parquet"
  X=$WX/cands-$TAG
  f=$(body "$NAME" <<EOF
set -euo pipefail
$CPU
mkdir -p $X/data $X/qe $X/m23 $X/out $X/prompts
gsutil -q cp $P/data/sources.parquet $X/data/
gsutil -q cp $P/texts/texts.parquet $X/texts.parquet
gsutil -m -q cp '$P/scores/qe-w*.parquet' $X/qe/
gsutil -m -q cp '$GM/data/prompts-w*.parquet' $X/m23/
nice -n 19 $SFT_PY -I -u $CODE/g4pool.py cands --data $X/data --texts $X/texts.parquet --qe '$X/qe/qe-w*.parquet' --m23-prompts '$X/m23/prompts-w*.parquet' --out $X/out --prompts-out $X/prompts
gsutil -m -q cp $X/prompts/prompts-w*.parquet $P/judge1/data/
gsutil -q cp $X/out/cands.parquet $P/choose/cands.parquet
gsutil -q cp $X/out/cands.json $P/choose/cands.json
echo "POOLCANDS_DONE $P/choose/ $P/judge1/data/"
EOF
)
  detach "$NAME" bash "$f"
  ;;
choose)
  only 0
  ready
  NAME=g4p_choose_$TAG
  gsutil -q stat "$P/choose/choose.json" && there_before "$NAME" "POOLCHOOSE_DONE $P/choose/ (there before)"
  need "$P/choose/cands.parquet"
  X=$WX/choose-$TAG
  f=$(body "$NAME" <<EOF
set -euo pipefail
$CPU
mkdir -p $X/data $X/out $X/prompts
gsutil -q cp $P/data/sources.parquet $X/data/
gsutil -q cp $P/texts/texts.parquet $X/texts.parquet
gsutil -q cp $P/choose/cands.parquet $X/cands.parquet
for k in 0 1 2 3; do mkdir -p $X/scores/w\$k; gsutil -m -q cp "$P/judge1/scores/w\$k/shard-*.parquet" $X/scores/w\$k/; done
nice -n 19 $SFT_PY -I -u $CODE/g4pool.py choose --data $X/data --texts $X/texts.parquet --cands $X/cands.parquet --scores '$X/scores/w*/shard-*.parquet' --out $X/out --prompts-out $X/prompts
gsutil -m -q cp $X/prompts/prompts-w*.parquet $P/judge2/data/
gsutil -m -q cp $X/out/choice.parquet $X/out/outputs.parquet $X/out/outputs-meta.parquet $X/out/pairs-meta.parquet $X/out/pairs.parquet $P/choose/
gsutil -q cp $X/out/choose.json $P/choose/choose.json
echo "POOLCHOOSE_DONE $P/choose/ $P/judge2/data/"
EOF
)
  detach "$NAME" bash "$f"
  ;;
poolreport)
  only 0
  ready
  NAME=g4p_report_$TAG
  gsutil -q stat "$P/report.json" && there_before "$NAME" "POOLREPORT_DONE $P/report.md (there before)"
  need "$P/choose/choose.json" "$P/scores/comet.parquet" "$P/naturalness/predictions.parquet" "$P/naturalness/predictions-c2.parquet"
  X=$WX/report-$TAG
  # only files without text
  f=$(body "$NAME" <<EOF
set -euo pipefail
$CPU
mkdir -p $X/data $X/choose $X/qe $X/report
gsutil -q cp $P/data/sources-meta.parquet $P/data/data.json $X/data/
gsutil -q cp $P/texts/texts-meta.parquet $P/texts/texts.json $X/
gsutil -m -q cp $P/choose/cands.json $P/choose/choose.json $P/choose/choice.parquet $P/choose/outputs-meta.parquet $P/choose/pairs-meta.parquet $X/choose/
gsutil -m -q cp '$P/scores/qe-w*.parquet' $X/qe/
gsutil -q cp $P/scores/comet.parquet $X/comet.parquet
gsutil -m -q cp $P/naturalness/predictions.parquet $P/naturalness/predictions-c2.parquet $X/
gsutil -q cp $GM/scores/metricx.parquet $X/m23-metricx.parquet
for s in judge1 judge2; do for k in 0 1 2 3; do mkdir -p $X/\$s/w\$k; gsutil -m -q cp "$P/\$s/scores/w\$k/shard-*.parquet" $X/\$s/w\$k/; done; done
for k in 0 1 2 3; do mkdir -p $X/m23j/w\$k; gsutil -m -q cp "$GM/judge/scores/w\$k/shard-*.parquet" $X/m23j/w\$k/; done
nice -n 19 $SFT_PY -I -u $CODE/g4pool.py report --data $X/data --texts-meta $X/texts-meta.parquet --texts-json $X/texts.json --cands-json $X/choose/cands.json --choose $X/choose --qe '$X/qe/qe-w*.parquet' --comet $X/comet.parquet --c1 $X/predictions.parquet --c2 $X/predictions-c2.parquet --judge1 '$X/judge1/w*/shard-*.parquet' --judge2 '$X/judge2/w*/shard-*.parquet' --m23-metricx $X/m23-metricx.parquet --m23-judge '$X/m23j/w*/shard-*.parquet' --out $X/report/report.md
gsutil -q cp $X/report/report.md $P/report.md
gsutil -q cp $X/report/report.json $P/report.json
echo "POOLREPORT_DONE $P/report.md"
EOF
)
  detach "$NAME" bash "$f"
  ;;
*)
  echo "unknown stage $STAGE (data1 judge pooldata generate texts qe classify classify-c2 comet cands choose poolreport)"
  exit 1
  ;;
esac
