#!/bin/bash
# G4-8-data (protocol-g4.md §18) on mbert-mix (4 x v4-8, us-central2-b), one stage per call on a worker; the chain
# g4arm8_data_run.sh (co1, mbert-ops) starts the stages and waits for their EXIT lines. Every stage runs detached from
# ssh with a fresh log ~/logs/g4a8_<stage>_$TAG.log that ends in "EXIT <code>" (also copied to $G8/logs/). The venvs,
# the tokenizers and MetricX-24 come from run_g4arm.sh setup. G8=$G/data/G4-8.
#   bash run_g4arm8.sh fresh                    w0 (CPU): g4arm8.py fresh (cands7's inputs, G4-7's plan and pool) ->
#                                               $G8/fresh/ (fresh.parquet, fresh.json, segments-w<k>.parquet)
#   JOB=anchor|c1|c2|c3 bash run_g4arm8.sh generate   every worker (its TPU): translate.py --copies on its segments,
#                                               G4-0 (no adapter), natural-v1, class medium; anchor greedy (1 copy) ->
#                                               $G8/anchor/samples/w<k>/; a chunk 8 samples at T = 1 -> $G8/<c>/samples/w<k>/
#   bash run_g4arm8.sh plan                     w0 (CPU): g4arm8.py plan -> $G8/plan/ and $G8/<c>/segments/
#   CHUNK=c<n> bash run_g4arm8.sh qe            every worker (its TPU): g4arm8.py qepairs of its samples, then
#                                               metricx_qe.py score (QE, XXL) -> $G8/<c>/qe/qe-w<k>.parquet
#   CHUNK=c<n> bash run_g4arm8.sh pairs         w0 (CPU): g4arm8.py pairs -> $G8/<c>/judge/data/
#   CHUNK=c<n> bash run_g4arm8.sh judge         every worker (its TPU): pairjudge.py score -> $G8/<c>/judge/scores/w<k>/
#   CHUNK=c<n> bash run_g4arm8.sh select        w0 (CPU): g4arm8.py select -> $G8/<c>/select/
#   bash run_g4arm8.sh v3                       w0 (CPU): g4arm8.py v3 on c1's targets -> $S/blind/G4-8-V3/
#   bash run_g4arm8.sh report                   w0 (CPU): g4arm8.py report -> $G8/report/, $G8/targets/targets.parquet
# A stage whose output is in the bucket already (its last file) prints that and ends in EXIT 0 (the outputs are
# deterministic or resumed by shard); nothing is removed. The code is beside this script (a copy of
# $G/scripts-g4a8-<commit>/). TAG names the jobs of one chain.
set -euo pipefail
STAGE=${1:?a stage: fresh generate plan qe pairs judge select v3 report}
R=gs://t5-dutch-english-usc2/mbert/translator
S=$R/sft
E=$R/eval
G=$S/g4arm
RL=$G/rl/G4-7
G8=$G/data/G4-8
V3_OUT=$S/blind/G4-8-V3
FNL=gs://t5-dutch-english-usc2/mbert/flan-nl
DECODE_CKPT=$R/judge/gemma-4-31b-it-maxtext/0/items            # the conversion every Gemma 4 decode uses
SHM=/dev/shm/translator
G_SHM=$SHM/g4arm
W8=$SHM/g4arm8                                                  # this script's own work directories
GEN=/dev/shm/translate-generic
FORK_COMMIT=f4d8615f478b6c18de9df8f2dff1f57f775a2521             # protocol-g4.md §9: every Gemma 4 decode
FPY=$GEN/venv-fork-${FORK_COMMIT:0:9}/bin/python
SFT_PY=$SHM/smoke-lora/venv/bin/python                          # setup_sft.sh (run_g4arm.sh setup)
TOK_DEC=$GEN/tokenizer-gemma4
TOK_SFT=$SHM/g4lora/tok-gemma4-31b                              # the same tokenizer.json (run_g4arm.sh setup checks)
MX_ROOT=$SHM/metricx
WORK=$G_SHM/judge-work                                          # pairjudge's MaxText output and compilation cache
CHUNKS="c1 c2 c3"
CODE=$(cd "$(dirname "$0")" && pwd)
W=$(hostname | sed 's/.*-w-//')
TAG=${TAG:-manual}
mkdir -p "$G_SHM" "$W8" ~/logs

detach() {  # detach NAME COMMAND...: the command in the background, log ~/logs/NAME.log ending in EXIT <code>
  local name=$1
  shift
  [ -e ~/logs/$name.log ] && { echo "w$W: ~/logs/$name.log exists; a new start takes a new TAG"; exit 1; }
  printf '%q ' "$@" > ~/job_$name.sh
  setsid nohup bash -c "bash ~/job_$name.sh > ~/logs/$name.log 2>&1; echo EXIT \$? >> ~/logs/$name.log; gsutil -q cp ~/logs/$name.log $G8/logs/$name-w$W.log" \
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

ready() {
  [ -x "$FPY" ] && [ -x "$SFT_PY" ] && [ -s "$TOK_DEC/tokenizer.json" ] && [ -s "$TOK_SFT/tokenizer.json" ] \
    || { echo "w$W: no fork venv, SFT venv or tokenizer (run_g4arm.sh setup)"; exit 1; }
}

chunk_of() {
  local c=${CHUNK:?CHUNK=c1, c2 or c3}
  case " $CHUNKS " in *" $c "*) echo "$c" ;; *) echo "CHUNK is one of $CHUNKS, not $c" >&2; exit 1 ;; esac
}

# A worker of the v4-32 as a TPU of its own 4 chips (run_g4arm.sh)
ONE_HOST="export TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 TPU_PROCESS_BOUNDS=1,1,1 TPU_VISIBLE_CHIPS=0,1,2,3 TPU_PROCESS_PORT=8476
export TPU_PROCESS_ADDRESSES=localhost:8476 CLOUD_TPU_TASK_ID=0 ALLOW_MULTIPLE_LIBTPU_LOAD=1
export XLA_FLAGS=--xla_dump_hlo_as_text=false TPU_LOG_DIR=$SHM/tpu_logs TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 HF_HUB_DISABLE_PROGRESS_BARS=1"
PREFLIGHT="timeout 600 $FPY -c \"import jax; assert (jax.device_count(), jax.process_count()) == (4, 1), (jax.device_count(), jax.process_count()); print('devices', jax.devices())\" || { echo 'preflight failed: not 4 chips of one process'; exit 2; }"
CPU="export JAX_PLATFORMS=cpu OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1"

case $STAGE in
fresh)
  only 0
  ready
  NAME=g4a8_fresh_$TAG
  gsutil -q stat "$G8/fresh/fresh.json" && there_before "$NAME" "FRESH8_DONE $G8/fresh/ (there before)"
  X=$W8/fresh-$TAG
  f=$(body "$NAME" <<EOF
set -euo pipefail
$CPU
mkdir -p $X/in/flan/plan2 $X/in/flan/nl-g4b $X/in/novels $X/in/bt3 $X/in/g47 $X/out
# cands7's inputs (~2.4 GB, all from the bucket of the region), G4-7's plan and pool, pool7's filters' inputs
gsutil -m -q cp $FNL/plan2/examples.parquet $FNL/plan2/plan.json '$FNL/plan2/segments-w[0-3].parquet' $X/in/flan/plan2/
gsutil -m -q cp '$FNL/nl-g4b/w[0-3].parquet' $X/in/flan/nl-g4b/
gsutil -m -q cp $FNL/nl-g4b/checks/build/build_report.json $R/mainset/flan-eval/eval-rows.parquet $X/in/
gsutil -m -q cp $S/data/books-b1/b1-selected.parquet $G/data/report-G4-4.json $X/in/novels/
gsutil -q cp $G/data/heldout4/manifest.json $X/in/novels/heldout4-manifest.json
gsutil -m -q rsync -r $G/input/bt3 $X/in/bt3
gsutil -m -q cp $RL/data/plan/plan.parquet $RL/data/pool/pool.parquet $X/in/g47/
gsutil -m -q cp $E/data/evalsets.parquet $S/blind/fixed-2026-11-10/blind100.jsonl $RL/c2/data/docs.txt $X/in/
nice -n 19 ionice -c3 $SFT_PY -I -u $CODE/g4arm8.py fresh --g47-plan $X/in/g47/plan.parquet --g47-pool $X/in/g47/pool.parquet --flan-plan $X/in/flan/plan2 --flan-kept $X/in/flan/nl-g4b --flan-build-report $X/in/build_report.json --flan-eval $X/in/eval-rows.parquet --novels $X/in/novels/b1-selected.parquet --novels-report $X/in/novels/report-G4-4.json --novels-eval $X/in/novels/heldout4-manifest.json --bt $X/in/bt3 --evalsets $X/in/evalsets.parquet --blind $X/in/blind100.jsonl --c2-docs $X/in/docs.txt --tokenizer $TOK_SFT --out $X/out
gsutil -m -q cp $X/out/fresh.parquet '$X/out/segments-w*.parquet' $G8/fresh/
gsutil -q cp $X/out/fresh.json $G8/fresh/fresh.json
echo "FRESH8_DONE $G8/fresh/"
EOF
)
  detach "$NAME" bash "$f"
  ;;
generate)
  JOB=${JOB:?anchor, c1, c2 or c3}
  ready
  case $JOB in
    # the anchor of the fresh sources: G4-0's greedy translation (G4-7.4a), shards of 512
    anchor) SEG=$G8/fresh/segments-w$W.parquet; OUT=$G8/anchor/samples/w$W; DONE_AT=$OUT/DONE; EXTRA="--copies 1"; SS=512 ;;
    # a chunk: 8 samples per source at T = 1 with the worker's own seed (G4-8-data.3), shards of 32 sources
    c1|c2|c3) SEG=$G8/$JOB/segments/segments-w$W.parquet; OUT="$G8/$JOB/{sample}/w$W"; DONE_AT=$G8/$JOB/samples/w$W/DONE
       SEED=$(JAX_PLATFORMS=cpu $SFT_PY -I -c "import sys; sys.path.insert(0, '$CODE'); import g4rl; print(g4rl.seed_int('en-nl-g4arm/0/G4-8/sample/$JOB/w$W'))")
       EXTRA="--copies 8 --sample samples=1.0:$SEED"; SS=256 ;;
    *) echo "JOB is anchor, c1, c2 or c3, not $JOB"; exit 1 ;;
  esac
  NAME=g4a8_gen_${JOB}_$TAG
  gsutil -q stat "$DONE_AT" && there_before "$NAME" "GENERATED w$W $JOB (DONE there before)"
  tpu_free
  gsutil -q stat "$SEG" || { echo "no $SEG"; exit 1; }
  f=$(body "$NAME" <<EOF
set -uo pipefail
$ONE_HOST
export JAX_COMPILATION_CACHE_DIR=$G_SHM/jax_cache
mkdir -p $W8/seg-$NAME $G_SHM/work
gsutil -q cp $SEG $W8/seg-$NAME/segments.parquet || exit 1
$PREFLIGHT
echo "== $JOB w$W \$(date -u +%T) G4-0, no adapter"
# no fallback on memory (as G4-7.5): the class's own slots; translate.py skips the shards already in the bucket
$FPY -u $CODE/translate.py --segments $W8/seg-$NAME/segments.parquet --out '$OUT' --ckpt $DECODE_CKPT --tokenizer $TOK_DEC --model gemma4-31b --prompt natural-v1 --length-class medium --source-lang en --target-lang nl --work-dir $G_SHM/work --shard-size $SS $EXTRA || exit \$?
gsutil -q stat '$DONE_AT' || { echo "no $DONE_AT"; exit 1; }
echo "GENERATED w$W $JOB"
EOF
)
  detach "$NAME" bash "$f"
  ;;
plan)
  only 0
  ready
  NAME=g4a8_plan_$TAG
  gsutil -q stat "$G8/plan/plan.json" && there_before "$NAME" "PLAN8_DONE $G8/plan/ (there before)"
  X=$W8/plan-$TAG
  f=$(body "$NAME" <<EOF
set -euo pipefail
$CPU
mkdir -p $X/fresh $X/anchor $X/out
gsutil -q cp $RL/data/plan/plan.parquet $X/g47-plan.parquet
gsutil -q cp $G8/fresh/fresh.parquet $X/fresh/
gsutil -m -q rsync -r $G8/anchor/samples $X/anchor
nice -n 19 $SFT_PY -I -u $CODE/g4arm8.py plan --g47-plan $X/g47-plan.parquet --fresh $X/fresh --anchor-samples $X/anchor --out $X/out
for c in $CHUNKS; do gsutil -m -q cp "$X/out/\$c/segments-w*.parquet" $G8/\$c/segments/; done
gsutil -q cp $X/out/plan.parquet $G8/plan/plan.parquet
gsutil -q cp $X/out/plan.json $G8/plan/plan.json
echo "PLAN8_DONE $G8/plan/"
EOF
)
  detach "$NAME" bash "$f"
  ;;
qe)
  C=$(chunk_of)
  ready
  NAME=g4a8_qe_${C}_$TAG
  gsutil -q stat "$G8/$C/qe/qe-w$W.parquet" && there_before "$NAME" "QE8_DONE w$W $C (there before)"
  tpu_free
  [ -d "$MX_ROOT/metricx-24-hybrid-xxl-v2p6-bfloat16" ] || { echo "w$W: no MetricX-24 in $MX_ROOT"; exit 1; }
  gsutil -q stat "$G8/$C/samples/w$W/DONE" || { echo "no $G8/$C/samples/w$W/DONE"; exit 1; }
  X=$W8/qe-$C-$TAG
  f=$(body "$NAME" <<EOF
set -euo pipefail
mkdir -p $X/samples/w$W
gsutil -q cp $G8/plan/plan.parquet $X/plan.parquet
gsutil -m -q rsync -r $G8/$C/samples/w$W $X/samples/w$W
(
$CPU
nice -n 19 $SFT_PY -I -u $CODE/g4arm8.py qepairs --plan $X/plan.parquet --chunk $C --samples $X/samples --out $X/pairs.parquet
)
# MetricX-24 hybrid XXL in QE mode (no reference, an empty hypothesis without a score, as G4-7's judges)
(
$ONE_HOST
export JAX_COMPILATION_CACHE_DIR=$G_SHM/jax_cache_metricx
$FPY -I -u $CODE/metricx_qe.py score --pairs $X/pairs.parquet --pair source:hypothesis --model xxl --root $MX_ROOT --out $X/qe.parquet
)
gsutil -q cp $X/qe.parquet $G8/$C/qe/qe-w$W.parquet
echo "QE8_DONE w$W $C"
EOF
)
  detach "$NAME" bash "$f"
  ;;
pairs)
  C=$(chunk_of)
  only 0
  ready
  NAME=g4a8_pairs_${C}_$TAG
  gsutil -q stat "$G8/$C/judge/data/pairs.json" && there_before "$NAME" "PAIRS8_DONE $C (there before)"
  X=$W8/pairs-$C-$TAG
  f=$(body "$NAME" <<EOF
set -euo pipefail
$CPU
mkdir -p $X/samples $X/qe $X/out
gsutil -q cp $G8/plan/plan.parquet $X/plan.parquet
gsutil -m -q rsync -r $G8/$C/samples $X/samples
gsutil -m -q cp '$G8/$C/qe/qe-w*.parquet' $X/qe/
nice -n 19 $SFT_PY -I -u $CODE/g4arm8.py pairs --plan $X/plan.parquet --chunk $C --samples $X/samples --qe '$X/qe/qe-w*.parquet' --out $X/out
gsutil -m -q cp '$X/out/prompts-w*.parquet' $X/out/cands.parquet $G8/$C/judge/data/
gsutil -q cp $X/out/pairs.json $G8/$C/judge/data/pairs.json
echo "PAIRS8_DONE $C"
EOF
)
  detach "$NAME" bash "$f"
  ;;
judge)
  C=$(chunk_of)
  ready
  NAME=g4a8_judge_${C}_$TAG
  tpu_free
  gsutil -q stat "$G8/$C/judge/data/prompts-w$W.parquet" || { echo "no $G8/$C/judge/data/prompts-w$W.parquet"; exit 1; }
  X=$W8/judge-$C-$TAG
  # pairjudge.py score unchanged (G4-judge.2-4); it skips the shards already in the bucket
  f=$(body "$NAME" <<EOF
set -euo pipefail
$ONE_HOST
mkdir -p $X $WORK
gsutil -q cp $G8/$C/judge/data/prompts-w$W.parquet $X/prompts.parquet
$PREFLIGHT
$FPY -u $CODE/pairjudge.py score --prompts $X/prompts.parquet --out $G8/$C/judge/scores/w$W --ckpt $DECODE_CKPT --tokenizer $TOK_DEC --work-dir $WORK
EOF
)
  detach "$NAME" bash "$f"
  ;;
select)
  C=$(chunk_of)
  only 0
  ready
  NAME=g4a8_select_${C}_$TAG
  gsutil -q stat "$G8/$C/select/select.json" && there_before "$NAME" "SELECT8_DONE $C (there before)"
  X=$W8/select-$C-$TAG
  f=$(body "$NAME" <<EOF
set -euo pipefail
$CPU
mkdir -p $X/samples $X/out
gsutil -q cp $G8/plan/plan.parquet $X/plan.parquet
gsutil -m -q rsync -r $G8/$C/samples $X/samples
gsutil -q cp $G8/$C/judge/data/cands.parquet $X/cands.parquet
for k in 0 1 2 3; do mkdir -p $X/scores/w\$k && gsutil -m -q cp "$G8/$C/judge/scores/w\$k/shard-*.parquet" $X/scores/w\$k/; done
nice -n 19 $SFT_PY -I -u $CODE/g4arm8.py select --plan $X/plan.parquet --chunk $C --samples $X/samples --cands $X/cands.parquet --scores '$X/scores/w*/shard-*.parquet' --out $X/out
gsutil -m -q cp $X/out/choice.parquet $X/out/targets.parquet $G8/$C/select/
gsutil -q cp $X/out/select.json $G8/$C/select/select.json
echo "SELECT8_DONE $C"
EOF
)
  detach "$NAME" bash "$f"
  ;;
v3)
  only 0
  ready
  NAME=g4a8_v3_$TAG
  gsutil -q stat "$V3_OUT/key.jsonl" && { echo "$V3_OUT has a key.jsonl: the package stays as it is"; exit 1; }
  gsutil -q stat "$G8/c1/select/targets.parquet" || { echo "no $G8/c1/select/targets.parquet"; exit 1; }
  X=$W8/v3-$TAG
  f=$(body "$NAME" <<EOF
set -euo pipefail
$CPU
mkdir -p $X/out
gsutil -q cp $G8/plan/plan.parquet $X/plan.parquet
gsutil -q cp $G8/c1/select/targets.parquet $X/targets.parquet
nice -n 19 $SFT_PY -I -u $CODE/g4arm8.py v3 --plan $X/plan.parquet --targets $X/targets.parquet --out $X/out
gsutil -m -q cp $X/out/judge-input.jsonl $X/out/select-report.json $V3_OUT/
gsutil -q cp $X/out/key.jsonl $V3_OUT/key.jsonl
echo "V3_DONE $V3_OUT/"
EOF
)
  detach "$NAME" bash "$f"
  ;;
report)
  only 0
  ready
  NAME=g4a8_report_$TAG
  X=$W8/report-$TAG
  ARGS=""
  for c in $CHUNKS; do ARGS="$ARGS --chunk-dir $c=$X/$c"; done
  f=$(body "$NAME" <<EOF
set -euo pipefail
$CPU
mkdir -p $X/out
gsutil -q cp $G8/plan/plan.parquet $X/plan.parquet
for c in $CHUNKS; do
  mkdir -p $X/\$c
  gsutil -m -q cp $G8/\$c/select/choice.parquet $G8/\$c/select/targets.parquet $G8/\$c/judge/data/cands.parquet $G8/\$c/judge/data/pairs.json $X/\$c/
done
nice -n 19 $SFT_PY -I -u $CODE/g4arm8.py report --plan $X/plan.parquet$ARGS --out $X/out
gsutil -q cp $X/out/targets.parquet $G8/targets/targets.parquet
gsutil -m -q cp $X/out/report.json $X/out/report.md $G8/report/
echo "REPORT8_DONE $G8/report/report.md"
EOF
)
  detach "$NAME" bash "$f"
  ;;
*)
  echo "unknown stage $STAGE"
  exit 2
  ;;
esac
