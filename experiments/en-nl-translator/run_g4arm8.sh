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
# §21 (G4-8-data N1-scores, a preparation that decides nothing; the chain g4arm8_n1_run.sh): the same pairs with FB.8's N1
#   CHUNK=c<n> bash run_g4arm8.sh n1pairs       w0 (CPU): g4arm8.py n1pairs (checked against §18's prompts) -> $G8/<c>/judge-n1/data/
#   CHUNK=c<n> SET=n1 bash run_g4arm8.sh judge  every worker (its TPU): pairjudge.py score -> $G8/<c>/judge-n1/scores/w<k>/
#   CHUNK=c<n> SET=n1 bash run_g4arm8.sh select w0 (CPU): g4arm8.py select on N1's scores -> $G8/<c>/select-n1/
#   bash run_g4arm8.sh n1report                 w0 (CPU): g4arm8.py n1report -> $G8/n1/
#   SET=n1 bash run_g4arm8.sh v3                w0 (CPU): §22 G4-9.V3N1, g4arm8.py v3 --set n1 on c1's N1 targets (checked
#                                               against §21's report) -> $S/blind/G4-9-V3N1/
# G4-8's training (protocol-g4.md §19; the chain g4arm8_train_run.sh; SFT=$G8/sft; translate, score, texts, classify,
# classify-c2, pll and blind are run_g4arm.sh's with ARM=G4-8):
#   bash run_g4arm8.sh sftset                   w0 (CPU): g4arm8.py sftset (the targets as SFT rows, 2 passes, checked
#                                               against the plan and the evaluation; the smoke set; RL-val's segments) -> $SFT/
#   bash run_g4arm8.sh check                    w0 (CPU): g4arm.py check --arm G4-8 (MaxText's own SFT steps) -> $SFT/check.txt
#   RUN=g4arm_G4-8_s8|g4arm_G4-8 [PHASE=prep|launch] bash run_g4arm8.sh train   every worker at once: run_sft.sh over the
#                                               16 chips (the smoke run: 8 steps on $SFT/smoke/, the adapter of step 4 kept;
#                                               the full run: sftset's steps, the adapters of its save steps kept)
#   RUN=g4arm_G4-8_s8 bash run_g4arm8.sh smoke8 w0 (CPU): g4arm8.py smoke8 -> $SFT/smoke8-<run>.json (exit 1: a check failed)
#   RUN=g4arm_G4-8 bash run_g4arm8.sh adapter   w0 (CPU): g4arm8.py trained, then the candidates' adapters -> $G/G4-8-s<step>/
#   JOB=div MODEL=G4-0|G4-8-s<step> bash run_g4arm8.sh generate   every worker (its TPU): 8 samples at T = 1 of its RL-val
#                                               sources (the candidate's LoRA merged) -> $SFT/div/<MODEL>/samples/w<k>/
#   CANDIDATES="G4-8-s<k1> ..." O6=R-oud|R-nieuw|open [SMOKE_RUN=<run>] bash run_g4arm8.sh report8   w0 (CPU):
#                                               g4arm8.py report8 -> $G/report-G4-8/
# G4-9's training (protocol-g4.md §22, a draft: §19's training on the N1 choice of §21; the chain g4arm9_train_run.sh):
# ARM8=G4-9 with the stages of §19 above (sftset ... report8; generate JOB=div only), its names (runs g4arm_G4-9_s8 and
# g4arm_G4-9, candidates G4-9-s<step>, jobs g4a9_*) and places (G9=$G/data/G4-9, SFT=$G9/sft, $G/G4-9-s<step>/,
# $G/naturalness-G4-9/, $G/report-G4-9/), its seed en-nl-g4arm/0/G4-9; report8 with G4-8-s140 beside (G4-9.11), and
#   ARM8=G4-9 TARGETS=n1|intersect bash run_g4arm8.sh n1targets   w0 (CPU): g4arm8.py n1targets --mode $TARGETS (§22 O9:
#                                               §21's select-n1 per chunk, checked; intersect: only where §18's select took
#                                               the same sample) -> $G9/targets/targets.parquet, targets.json (its mode)
# A stage whose output is in the bucket already (its last file) prints that and ends in EXIT 0 (the outputs are
# deterministic or resumed by shard); nothing is removed. The code is beside this script (a copy of
# $G/scripts-g4a8-<commit>/). TAG names the jobs of one chain.
set -euo pipefail
STAGE=${1:?a stage: fresh generate plan qe pairs judge select v3 report n1pairs n1report n1targets sftset check train smoke8 adapter report8}
R=gs://t5-dutch-english-usc2/mbert/translator
S=$R/sft
E=$R/eval
G=$S/g4arm
RL=$G/rl/G4-7
G8=$G/data/G4-8
COMPARE9=G4-8-s140                                              # g4arm8.ARMS_T["G4-9"]["compare"] (§22 G4-9.11)
V3_OUT=$S/blind/G4-8-V3
V3N1_OUT=$S/blind/G4-9-V3N1                                     # §22 G4-9.V3N1
ARM8=${ARM8:-G4-8}
case $ARM8 in
  G4-8) TG=$G8; J=g4a8; REFS="G4-4" ;;                         # §19: G4-8-data's targets
  G4-9) TG=$G/data/G4-9; J=g4a9; REFS="G4-4 $COMPARE9" ;;      # §22: the N1 choice of §21, merged by n1targets
  *) echo "ARM8 is G4-8 or G4-9, not $ARM8"; exit 1 ;;
esac
SFT=$TG/sft                                                     # §19 (§22): the arm's SFT set, its checks, RL-val's samples
H=$G/data/heldout                                               # G4-2.5's SFT evals books, btval, own
FNL=gs://t5-dutch-english-usc2/mbert/flan-nl
DECODE_CKPT=$R/judge/gemma-4-31b-it-maxtext/0/items            # the conversion every Gemma 4 decode uses
SHM=/dev/shm/translator
G_SHM=$SHM/g4arm
W8=$SHM/g4arm${ARM8#G4-}                                        # this script's own work directories
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
  setsid nohup bash -c "bash ~/job_$name.sh > ~/logs/$name.log 2>&1; echo EXIT \$? >> ~/logs/$name.log; gsutil -q cp ~/logs/$name.log $TG/logs/$name-w$W.log" \
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

judge_set() {  # the judge's directories of a chunk: §18's rubric (SET empty) or §21's N1 (SET=n1)
  case ${SET:-} in
    "") JD=$G8/$1/judge; SD=$G8/$1/select; SN="" ;;
    n1) JD=$G8/$1/judge-n1; SD=$G8/$1/select-n1; SN=n1 ;;
    *) echo "SET is empty (§18) or n1 (§21), not $SET"; exit 1 ;;
  esac
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

t8() {  # t8 NAME: a constant of g4arm8.py (§19), so that the stages and the code agree
  JAX_PLATFORMS=cpu $SFT_PY -I -c "import sys; sys.path.insert(0, '$CODE'); import g4arm8; print(g4arm8.$1)"
}

json_key() {  # json_key FILE KEY: a top-level value of a json file (a list joined by commas)
  python3 -c "import json, sys; v = json.load(open(sys.argv[1]))[sys.argv[2]]; print(','.join(map(str, v)) if isinstance(v, list) else v)" "$1" "$2"
}

smoke_run() { [[ $1 =~ ^g4arm_${ARM8}_s8[a-z]?$ ]]; }           # g4arm8.SMOKE_RUN_T (G4-8)
full_run() { [[ $1 =~ ^g4arm_${ARM8}[a-z]?$ ]]; }               # g4arm8.RUN_T (G4-8)
cand8() { [[ $1 =~ ^${ARM8}-s[1-9][0-9]{0,3}$ ]]; }             # g4arm8.CAND_T of the arm (run_g4arm.sh's cand8)

# §22: G4-9 has the training's stages only; the data are §18's and §21's
case $ARM8/$STAGE in
  G4-8/n1targets) echo "n1targets is G4-9's (ARM8=G4-9)"; exit 1 ;;
  G4-9/n1targets|G4-9/sftset|G4-9/check|G4-9/train|G4-9/smoke8|G4-9/adapter|G4-9/report8) ;;
  G4-9/generate) [ "${JOB:-}" = div ] || { echo "ARM8=G4-9: generate JOB=div only"; exit 1; } ;;
  G4-9/*) echo "$STAGE is G4-8-data's (§18, §21): not with ARM8=G4-9"; exit 1 ;;
esac

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
  JOB=${JOB:?anchor, c1, c2, c3 or div}
  ready
  NAME=g4a8_gen_${JOB}_$TAG
  LORA=""
  case $JOB in
    # the anchor of the fresh sources: G4-0's greedy translation (G4-7.4a), shards of 512
    anchor) SEG=$G8/fresh/segments-w$W.parquet; OUT=$G8/anchor/samples/w$W; DONE_AT=$OUT/DONE; EXTRA="--copies 1"; SS=512 ;;
    # a chunk: 8 samples per source at T = 1 with the worker's own seed (G4-8-data.3), shards of 32 sources
    c1|c2|c3) SEG=$G8/$JOB/segments/segments-w$W.parquet; OUT="$G8/$JOB/{sample}/w$W"; DONE_AT=$G8/$JOB/samples/w$W/DONE
       SEED=$(JAX_PLATFORMS=cpu $SFT_PY -I -c "import sys; sys.path.insert(0, '$CODE'); import g4rl; print(g4rl.seed_int('en-nl-g4arm/0/G4-8/sample/$JOB/w$W'))")
       EXTRA="--copies 8 --sample samples=1.0:$SEED"; SS=256 ;;
    # §19 G4-8.6: RL-val's 8 samples at T = 1 of G4-0 or a candidate, the same seed per worker for every model
    div) MODEL=${MODEL:?G4-0 or a candidate $ARM8-s<step>}
       { [ "$MODEL" = G4-0 ] || cand8 "$MODEL"; } || { echo "MODEL is G4-0 or a candidate $ARM8-s<step>, not $MODEL"; exit 1; }
       SEG=$SFT/rlval/segments-w$W.parquet; OUT="$SFT/div/$MODEL/{sample}/w$W"; DONE_AT=$SFT/div/$MODEL/samples/w$W/DONE
       SEED=$(JAX_PLATFORMS=cpu $SFT_PY -I -c "import sys; sys.path.insert(0, '$CODE'); import g4rl; print(g4rl.seed_int('en-nl-g4arm/0/$ARM8/div/w$W'))")
       EXTRA="--copies 8 --sample samples=1.0:$SEED"; SS=256; NAME=${J}_gen_div_${MODEL}_$TAG ;;
    *) echo "JOB is anchor, c1, c2, c3 or div, not $JOB"; exit 1 ;;
  esac
  gsutil -q stat "$DONE_AT" && there_before "$NAME" "GENERATED w$W $JOB (DONE there before)"
  tpu_free
  gsutil -q stat "$SEG" || { echo "no $SEG"; exit 1; }
  if [ "$JOB" = div ] && [ "$MODEL" != G4-0 ]; then
    # G4-8.4 (G4-9.5): the candidate's LoRA, rank 32 / alpha 16 from G4-0 (nothing stacked); the job removes its own copy
    LORA=$W8/adapter-$NAME
    mkdir -p "$LORA"
    gsutil -q cp "$G/$MODEL/adapter.npz" "$G/$MODEL/adapter.json" "$LORA/"
    if ! python3 -c "import json, sys; a = json.load(open(sys.argv[1])); assert (a['rank'], a['alpha'], a['scale']) == (32, 16.0, 0.5), a" "$LORA/adapter.json"; then
      rm -rf -- "$LORA"
      echo "$G/$MODEL/adapter.json is not rank 32, alpha 16"
      exit 1
    fi
    EXTRA="$EXTRA --lora $LORA"
  fi
  f=$(body "$NAME" <<EOF
set -uo pipefail
$ONE_HOST
export JAX_COMPILATION_CACHE_DIR=$G_SHM/jax_cache
mkdir -p $W8/seg-$NAME $G_SHM/work
gsutil -q cp $SEG $W8/seg-$NAME/segments.parquet || exit 1
$PREFLIGHT
echo "== $JOB w$W \$(date -u +%T) ${MODEL:-G4-0}${LORA:+ with its LoRA}${LORA:- (no adapter)}"
# no fallback on memory (as G4-7.5): the class's own slots; translate.py skips the shards already in the bucket
$FPY -u $CODE/translate.py --segments $W8/seg-$NAME/segments.parquet --out '$OUT' --ckpt $DECODE_CKPT --tokenizer $TOK_DEC --model gemma4-31b --prompt natural-v1 --length-class medium --source-lang en --target-lang nl --work-dir $G_SHM/work --shard-size $SS $EXTRA || exit \$?
${LORA:+rm -rf -- $LORA}
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
  judge_set "$C"
  ready
  NAME=g4a8_judge${SN}_${C}_$TAG
  tpu_free
  [ -z "$SN" ] || gsutil -q stat "$JD/data/n1pairs.json" || { echo "no $JD/data/n1pairs.json"; exit 1; }
  gsutil -q stat "$JD/data/prompts-w$W.parquet" || { echo "no $JD/data/prompts-w$W.parquet"; exit 1; }
  X=$W8/judge$SN-$C-$TAG
  # pairjudge.py score unchanged (G4-judge.2-4); it skips the shards already in the bucket
  f=$(body "$NAME" <<EOF
set -euo pipefail
$ONE_HOST
mkdir -p $X $WORK
gsutil -q cp $JD/data/prompts-w$W.parquet $X/prompts.parquet
$PREFLIGHT
$FPY -u $CODE/pairjudge.py score --prompts $X/prompts.parquet --out $JD/scores/w$W --ckpt $DECODE_CKPT --tokenizer $TOK_DEC --work-dir $WORK
EOF
)
  detach "$NAME" bash "$f"
  ;;
select)
  C=$(chunk_of)
  judge_set "$C"
  only 0
  ready
  NAME=g4a8_select${SN}_${C}_$TAG
  gsutil -q stat "$SD/select.json" && there_before "$NAME" "SELECT8_DONE $C${SN:+ $SN} (there before)"
  X=$W8/select$SN-$C-$TAG
  # the candidates are always §18's (judge/data/cands.parquet); SET only changes whose scores choose
  f=$(body "$NAME" <<EOF
set -euo pipefail
$CPU
mkdir -p $X/samples $X/out
gsutil -q cp $G8/plan/plan.parquet $X/plan.parquet
gsutil -m -q rsync -r $G8/$C/samples $X/samples
gsutil -q cp $G8/$C/judge/data/cands.parquet $X/cands.parquet
for k in 0 1 2 3; do mkdir -p $X/scores/w\$k && gsutil -m -q cp "$JD/scores/w\$k/shard-*.parquet" $X/scores/w\$k/; done
nice -n 19 $SFT_PY -I -u $CODE/g4arm8.py select --plan $X/plan.parquet --chunk $C --samples $X/samples --cands $X/cands.parquet --scores '$X/scores/w*/shard-*.parquet' --out $X/out
gsutil -m -q cp $X/out/choice.parquet $X/out/targets.parquet $SD/
gsutil -q cp $X/out/select.json $SD/select.json
echo "SELECT8_DONE $C${SN:+ $SN}"
EOF
)
  detach "$NAME" bash "$f"
  ;;
v3)
  # G4-8-data.8 (SET empty): the rubric's targets of c1; §22 G4-9.V3N1 (SET=n1): §21's N1 targets of c1, checked by
  # g4arm8.py against §21's report, with §18's choice for the key
  only 0
  ready
  case ${SET:-} in
    "") OUT=$V3_OUT; NAME=g4a8_v3_$TAG; X=$W8/v3-$TAG; SRC=$G8/c1/select; EXTRA=""; N1IN="" ;;
    n1) OUT=$V3N1_OUT; NAME=g4a8_v3n1_$TAG; X=$W8/v3n1-$TAG; SRC=$G8/c1/select-n1
        EXTRA=" --set n1 --choice $X/choice.parquet --rubric $X/rubric-choice.parquet --n1report $X/n1report.json"
        N1IN="gsutil -q cp $SRC/choice.parquet $X/choice.parquet
gsutil -q cp $G8/c1/select/choice.parquet $X/rubric-choice.parquet
gsutil -q cp $G8/n1/report.json $X/n1report.json" ;;
    *) echo "SET is empty (G4-8-V3) or n1 (G4-9-V3N1), not $SET"; exit 1 ;;
  esac
  gsutil -q stat "$OUT/key.jsonl" && { echo "$OUT has a key.jsonl: the package stays as it is"; exit 1; }
  gsutil -q stat "$SRC/targets.parquet" || { echo "no $SRC/targets.parquet"; exit 1; }
  [ -z "${SET:-}" ] || gsutil -q stat "$G8/n1/report.json" || { echo "no $G8/n1/report.json (§21's n1report)"; exit 1; }
  f=$(body "$NAME" <<EOF
set -euo pipefail
$CPU
mkdir -p $X/out
gsutil -q cp $G8/plan/plan.parquet $X/plan.parquet
gsutil -q cp $SRC/targets.parquet $X/targets.parquet
$N1IN
nice -n 19 $SFT_PY -I -u $CODE/g4arm8.py v3 --plan $X/plan.parquet --targets $X/targets.parquet$EXTRA --out $X/out
gsutil -m -q cp $X/out/judge-input.jsonl $X/out/select-report.json $OUT/
gsutil -q cp $X/out/key.jsonl $OUT/key.jsonl
echo "V3_DONE $OUT/"
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
n1pairs)
  # §21 G4-8-N1.2-3: N1's prompts of §18's judged keys; n1pairs stops unless its rubric prompts are §18's per uid
  C=$(chunk_of)
  only 0
  ready
  NAME=g4a8_n1pairs_${C}_$TAG
  gsutil -q stat "$G8/$C/judge-n1/data/n1pairs.json" && there_before "$NAME" "N1PAIRS_DONE $C (there before)"
  gsutil -q stat "$G8/$C/judge/data/pairs.json" || { echo "no $G8/$C/judge/data/pairs.json (§18's pairs)"; exit 1; }
  X=$W8/n1pairs-$C-$TAG
  f=$(body "$NAME" <<EOF
set -euo pipefail
$CPU
mkdir -p $X/samples $X/rubric $X/out
gsutil -q cp $G8/plan/plan.parquet $X/plan.parquet
gsutil -m -q rsync -r $G8/$C/samples $X/samples
gsutil -m -q cp $G8/$C/judge/data/cands.parquet '$G8/$C/judge/data/prompts-w*.parquet' $X/rubric/
nice -n 19 $SFT_PY -I -u $CODE/g4arm8.py n1pairs --plan $X/plan.parquet --chunk $C --samples $X/samples --cands $X/rubric/cands.parquet --rubric $X/rubric --out $X/out
gsutil -m -q cp '$X/out/prompts-w*.parquet' $G8/$C/judge-n1/data/
gsutil -q cp $X/out/n1pairs.json $G8/$C/judge-n1/data/n1pairs.json
echo "N1PAIRS_DONE $C"
EOF
)
  detach "$NAME" bash "$f"
  ;;
n1report)
  # §21 G4-8-N1.7: N1's choice beside §18's rubric choice, both recomputed from their scores and checked
  only 0
  ready
  NAME=g4a8_n1report_$TAG
  gsutil -q stat "$G8/n1/report.json" && there_before "$NAME" "N1REPORT_DONE $G8/n1/report.md (there before)"
  X=$W8/n1report-$TAG
  ARGS=""
  for c in $CHUNKS; do ARGS="$ARGS --chunk-dir $c=$X/$c"; done
  f=$(body "$NAME" <<EOF
set -euo pipefail
$CPU
mkdir -p $X/out
gsutil -q cp $G8/plan/plan.parquet $X/plan.parquet
for c in $CHUNKS; do
  mkdir -p $X/\$c
  gsutil -m -q cp $G8/\$c/judge/data/cands.parquet $G8/\$c/judge-n1/data/n1pairs.json $G8/\$c/select/choice.parquet $X/\$c/
  gsutil -q cp $G8/\$c/select-n1/choice.parquet $X/\$c/choice-n1.parquet
  for k in 0 1 2 3; do
    mkdir -p $X/\$c/scores/w\$k $X/\$c/scores-n1/w\$k
    gsutil -m -q cp "$G8/\$c/judge/scores/w\$k/shard-*.parquet" $X/\$c/scores/w\$k/
    gsutil -m -q cp "$G8/\$c/judge-n1/scores/w\$k/shard-*.parquet" $X/\$c/scores-n1/w\$k/
  done
done
nice -n 19 $SFT_PY -I -u $CODE/g4arm8.py n1report --plan $X/plan.parquet$ARGS --out $X/out
gsutil -q cp $X/out/report.md $G8/n1/report.md
gsutil -q cp $X/out/report.json $G8/n1/report.json
echo "N1REPORT_DONE $G8/n1/report.md"
EOF
)
  detach "$NAME" bash "$f"
  ;;
n1targets)
  # §22 G4-9.2: the N1 choice of §21 (select-n1 per chunk) as one targets file in §18's form, checked against §21's
  # report; TARGETS is O9 of G4-9.0: n1 (all of them) or intersect (only where §18's rubric took the same sample,
  # select/ of each chunk beside); the json last: its presence says the stage is done, a json of the other mode stays
  only 0
  ready
  case ${TARGETS:-} in
    n1|intersect) ;;
    *) echo "TARGETS is n1 or intersect (§22 G4-9.0, O9), not '${TARGETS:-}'"; exit 1 ;;
  esac
  NAME=${J}_n1targets_$TAG
  if gsutil -q stat "$TG/targets/targets.json"; then
    gsutil cat "$TG/targets/targets.json" | $SFT_PY -I -c "import json, sys; sys.exit(json.load(sys.stdin).get('mode') != '$TARGETS')" \
      || { echo "$TG/targets/targets.json is not of TARGETS $TARGETS: it stays as it is"; exit 1; }
    there_before "$NAME" "N1TARGETS_DONE $TG/targets/ (there before, TARGETS $TARGETS)"
  fi
  gsutil -q stat "$G8/n1/report.json" || { echo "no $G8/n1/report.json (§21's n1report)"; exit 1; }
  X=$W8/n1targets-$TAG
  ARGS=" --mode $TARGETS"
  for c in $CHUNKS; do ARGS="$ARGS --chunk-dir $c=$X/$c"; done
  RUBRIC=""
  if [ "$TARGETS" = intersect ]; then
    for c in $CHUNKS; do
      for x in choice targets; do
        gsutil -q stat "$G8/$c/select/$x.parquet" || { echo "no $G8/$c/select/$x.parquet (§18's select)"; exit 1; }
      done
      ARGS="$ARGS --rubric-dir $c=$X/rubric/$c"
    done
    RUBRIC="for c in $CHUNKS; do mkdir -p $X/rubric/\$c && gsutil -m -q cp $G8/\$c/select/choice.parquet $G8/\$c/select/targets.parquet $X/rubric/\$c/; done"
  fi
  f=$(body "$NAME" <<EOF
set -euo pipefail
$CPU
mkdir -p $X/out
gsutil -q cp $G8/plan/plan.parquet $X/plan.parquet
gsutil -q cp $G8/n1/report.json $X/n1report.json
for c in $CHUNKS; do mkdir -p $X/\$c && gsutil -m -q cp $G8/\$c/select-n1/choice.parquet $G8/\$c/select-n1/targets.parquet $X/\$c/; done
$RUBRIC
nice -n 19 $SFT_PY -I -u $CODE/g4arm8.py n1targets --plan $X/plan.parquet --n1report $X/n1report.json$ARGS --out $X/out
gsutil -q cp $X/out/targets.parquet $TG/targets/targets.parquet
gsutil -q cp $X/out/targets.json $TG/targets/targets.json
echo "N1TARGETS_DONE $TG/targets/ (TARGETS $TARGETS)"
EOF
)
  detach "$NAME" bash "$f"
  ;;
sftset)
  # §19 G4-8.2, G4-8.5, G4-8.6 (§22 G4-9.3): the SFT set of the targets, the smoke set, RL-val's segments
  only 0
  ready
  NAME=${J}_sftset_$TAG
  gsutil -q stat "$SFT/manifest.json" && there_before "$NAME" "SFTSET8_DONE $SFT/ (there before)"
  gsutil -q stat "$TG/targets/targets.parquet" || { echo "no $TG/targets/targets.parquet (G4-8-data's report; G4-9: n1targets)"; exit 1; }
  X=$W8/sftset-$TAG
  # §22 G4-9.3: G4-9's set names its targets' mode (O9) from n1targets' json; G4-8's targets have none
  INFO=""; INFO_CP=""
  if [ "$ARM8" = G4-9 ]; then
    gsutil -q stat "$TG/targets/targets.json" || { echo "no $TG/targets/targets.json (n1targets)"; exit 1; }
    INFO=" --targets-info $X/in/targets.json"
    INFO_CP="gsutil -q cp $TG/targets/targets.json $X/in/targets.json"
  fi
  f=$(body "$NAME" <<EOF
set -euo pipefail
$CPU
mkdir -p $X/in $X/out
gsutil -m -q cp $TG/targets/targets.parquet $G8/plan/plan.parquet $E/data/evalsets.parquet $S/blind/fixed-2026-11-10/blind100.jsonl $R/mainset/flan-eval/eval-rows.parquet $X/in/
gsutil -q cp $RL/data/pool/pool.parquet $X/in/g47-pool.parquet
gsutil -q cp $RL/data/plan/plan.parquet $X/in/g47-plan.parquet
gsutil -m -q cp $H/eval-books.parquet $H/eval-btval.parquet $H/eval-own.parquet $X/in/
$INFO_CP
nice -n 19 ionice -c3 $SFT_PY -I -u $CODE/g4arm8.py sftset --targets $X/in/targets.parquet --plan $X/in/plan.parquet --evalsets $X/in/evalsets.parquet --blind $X/in/blind100.jsonl --flan-eval $X/in/eval-rows.parquet --g47-pool $X/in/g47-pool.parquet --g47-plan $X/in/g47-plan.parquet --sft-eval books=$X/in/eval-books.parquet --sft-eval btval=$X/in/eval-btval.parquet --sft-eval own=$X/in/eval-own.parquet --tokenizer $TOK_SFT --arm=$ARM8$INFO --out $X/out
gsutil -m -q cp $X/out/train.parquet $X/out/train-meta.parquet $X/out/template.jinja $X/out/report.json $X/out/report.md $SFT/
gsutil -m -q cp $X/out/smoke/train.parquet $X/out/smoke/train-meta.parquet $X/out/smoke/template.jinja $X/out/smoke/manifest.json $SFT/smoke/
gsutil -m -q cp '$X/out/rlval/segments-w*.parquet' $SFT/rlval/
# the manifest last: its presence says the stage is done
gsutil -q cp $X/out/manifest.json $SFT/manifest.json
echo "SFTSET8_DONE $SFT/"
EOF
)
  detach "$NAME" bash "$f"
  ;;
check)
  # §19 G4-8.2: MaxText's own SFT steps on every distinct example (g4arm.py check, as G4-3 and G4-4)
  only 0
  ready
  NAME=${J}_check_$TAG
  gsutil -q stat "$SFT/check.txt" && there_before "$NAME" "CHECK8_DONE $SFT/check.txt (there before)"
  gsutil -q stat "$SFT/manifest.json" || { echo "no $SFT/manifest.json (sftset)"; exit 1; }
  X=$W8/check-$TAG
  f=$(body "$NAME" <<EOF
set -euo pipefail
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=true HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 JAX_PLATFORMS=cpu
mkdir -p $X
gsutil -m -q cp $SFT/train.parquet $SFT/train-meta.parquet $SFT/template.jinja $X/
nice -n 19 ionice -c3 $SFT_PY -u $CODE/g4arm.py check --data $X --tokenizer $TOK_SFT --arm $ARM8 --processes ${CHECK_PROCESSES:-8} | tee $X/check.txt
grep -q '^CHECK ' $X/check.txt
gsutil -q cp $X/check.txt $SFT/check.txt
echo "CHECK8_DONE $SFT/check.txt"
EOF
)
  detach "$NAME" bash "$f"
  ;;
train)
  # §19 G4-8.3/G4-8.5: one job over the 16 chips; PHASE=prep on every worker first, then PHASE=launch on every worker
  # (the processes meet in jax.distributed.initialize within a timeout, as G4-5's and G4-7's runs)
  RUN=${RUN:?g4arm_${ARM8}_s8 (the smoke run) or g4arm_$ARM8 (the full run; a new start of either takes a new letter)}
  D=$W8/data-$RUN
  case ${PHASE:-} in
    launch) [ -s "$D/launch.sh" ] || { echo "w$W: no $D/launch.sh (PHASE=prep first)"; exit 1; }
            bash "$D/launch.sh"
            exit 0 ;;
    prep|"") ;;
    *) echo "PHASE is prep, launch or empty (both), not $PHASE"; exit 1 ;;
  esac
  ready
  tpu_free
  # a launch never takes the arguments of an earlier or a refused preparation
  rm -f -- "$D/launch.sh"
  if smoke_run "$RUN"; then
    SETDIR=$SFT/smoke
  elif full_run "$RUN"; then
    SETDIR=$SFT
  else
    echo "RUN $RUN is no run of $ARM8 (g4arm_${ARM8}_s8[a-z] or g4arm_$ARM8[a-z])"; exit 1
  fi
  gsutil -q stat "$SFT/check.txt" || { echo "no $SFT/check.txt: the set is not checked (check)"; exit 1; }
  ! gsutil -q stat "$G/runs/$RUN/summary.json" || { echo "$G/runs/$RUN has a summary.json: a new run takes a new RUN"; exit 1; }
  for f in eval-books.parquet eval-btval.parquet eval-own.parquet; do gsutil -q stat "$H/$f" || { echo "no $H/$f"; exit 1; }; done
  mkdir -p "$D"
  gsutil -m -q cp "$SETDIR/train.parquet" "$SETDIR/template.jinja" "$SETDIR/manifest.json" "$H/eval-books.parquet" \
    "$H/eval-btval.parquet" "$H/eval-own.parquet" "$D/"
  if smoke_run "$RUN"; then
    STEPS=$(t8 SMOKE_STEPS_T); EVERY=$(t8 SMOKE_EVERY_T); SAVE=$(t8 SMOKE_SAVE_T)
  else
    gsutil -q cp "$SFT/manifest.json" "$D/manifest-sft.json"
    STEPS=$(json_key "$D/manifest-sft.json" steps); EVERY=$(json_key "$D/manifest-sft.json" eval_every)
    SAVE=$(json_key "$D/manifest-sft.json" save_steps)
  fi
  ARGS="--eval books=$D/eval-books.parquet --eval btval=$D/eval-btval.parquet --eval own=$D/eval-own.parquet"
  ARGS="$ARGS --eval-every $EVERY --save-steps $SAVE --warmup-steps $(t8 WARMUP_T) --decay-fraction 0"
  # G4-8.3: LoRA rank 32 / alpha 16 on run_sft.sh's modules, rows of 4,096, LR 5e-5 constant after the warmup (MaxText's
  # WSD without decay: --decay-fraction 0)
  printf '%q ' env SFT_ARGS="$ARGS" MULTIHOST=1 NAME="$RUN" MODEL=gemma4-31b DATA="$D/train.parquet" \
    TEMPLATE="$D/template.jinja" ROOT="$G" STEPS="$STEPS" bash "$CODE/run_sft.sh" per_device_batch_size=1 \
    max_target_length="$(t8 SEQ_T)" lora.lora_rank=32 lora.lora_alpha=16.0 learning_rate="$(t8 LR_T)" \
    lr_schedule_type=wsd wsd_decay_style=linear learning_rate_final_fraction=0.1 > "$D/launch.sh.new"
  mv "$D/launch.sh.new" "$D/launch.sh"
  if [ "${PHASE:-}" = prep ]; then
    echo "prepared $RUN on w$W"
    exit 0
  fi
  bash "$D/launch.sh"
  ;;
smoke8)
  # §19 G4-8.5: the smoke run's checks; the job ends in EXIT 1 if one fails
  only 0
  ready
  RUN=${RUN:?the smoke run, g4arm_${ARM8}_s8}
  smoke_run "$RUN" || { echo "RUN $RUN is no smoke run of $ARM8"; exit 1; }
  NAME=${J}_smoke8_$TAG
  RD=$G/runs/$RUN
  X=$W8/smoke8-$TAG
  f=$(body "$NAME" <<EOF
set -euo pipefail
$CPU
mkdir -p $X
# run_sft.sh copies the run's files after its EXIT line: up to 10 min
for _ in \$(seq 1 60); do
  gsutil -q stat $RD/summary.json && gsutil -q stat $RD/adapter.json && gsutil -q stat $RD/adapter-step4.json && break
  sleep 10
done
gsutil -m -q cp $RD/summary.json $RD/adapter.json $RD/adapter-step4.json $X/
gsutil -q cp $SFT/smoke/manifest.json $X/smoke-manifest.json
gsutil -q cp $SFT/manifest.json $X/manifest.json
code=0
$SFT_PY -I -u $CODE/g4arm8.py smoke8 --run $RUN --summary $X/summary.json --adapter $X/adapter.json --adapter-step $X/adapter-step4.json --smoke-manifest $X/smoke-manifest.json --manifest $X/manifest.json --arm=$ARM8 --out $X/smoke8.json || code=\$?
if [ -s $X/smoke8.json ]; then gsutil -q cp $X/smoke8.json $SFT/smoke8-$RUN.json; fi
[ \$code = 0 ] || exit \$code
echo "SMOKE8_DONE $SFT/smoke8-$RUN.json"
EOF
)
  detach "$NAME" bash "$f"
  ;;
adapter)
  # §19 G4-8.4 (§22 G4-9.5): the full run checked against sftset (g4arm8.py trained), then each candidate's adapter to
  # $G/<arm>-s<step>/
  only 0
  ready
  RUN=${RUN:?the full run, g4arm_$ARM8}
  full_run "$RUN" || { echo "RUN $RUN is no full run of $ARM8"; exit 1; }
  NAME=${J}_adapter_$TAG
  gsutil -q stat "$SFT/trained.json" && there_before "$NAME" "TRAINED8_DONE $SFT/trained.json (there before)"
  gsutil -q stat "$SFT/manifest.json" || { echo "no $SFT/manifest.json (sftset)"; exit 1; }
  X=$W8/adapter-$TAG
  mkdir -p "$X"
  gsutil -q cp "$SFT/manifest.json" "$X/manifest.json"
  SAVE=$(json_key "$X/manifest.json" save_steps | tr ',' ' ')
  LAST=$(json_key "$X/manifest.json" steps)
  for k in $SAVE $LAST; do
    ! gsutil -q stat "$G/$ARM8-s$k/adapter.json" || { echo "$G/$ARM8-s$k has an adapter.json: it stays as it is"; exit 1; }
  done
  RD=$G/runs/$RUN
  ADAPTERS="--adapter $ARM8-s$LAST=$X/adapter.json"
  WANT="$RD/summary.json $RD/adapter.json"
  for k in $SAVE; do ADAPTERS="$ADAPTERS --adapter $ARM8-s$k=$X/adapter-step$k.json"; WANT="$WANT $RD/adapter-step$k.json"; done
  f=$(body "$NAME" <<EOF
set -euo pipefail
$CPU
# run_sft.sh copies the run's files after its EXIT line: up to 10 min
for _ in \$(seq 1 60); do
  n=0; for p in $WANT; do gsutil -q stat \$p && n=\$((n + 1)); done
  [ \$n = $(echo $WANT | wc -w) ] && break
  sleep 10
done
gsutil -m -q cp $WANT $X/
code=0
$SFT_PY -I -u $CODE/g4arm8.py trained --run $RUN --summary $X/summary.json --manifest $X/manifest.json $ADAPTERS --arm=$ARM8 --out $X/trained.json || code=\$?
if [ \$code != 0 ]; then
  if [ -s $X/trained.json ]; then gsutil -q cp $X/trained.json $SFT/trained-$RUN-failed.json; fi
  exit \$code
fi
for k in $SAVE; do
  gsutil -q cp $RD/adapter-step\$k.npz $G/$ARM8-s\$k/adapter.npz
  gsutil -q cp $RD/adapter-step\$k.json $G/$ARM8-s\$k/adapter.json
  gsutil -q cp $RD/summary.json $G/$ARM8-s\$k/summary.json
  echo "ADAPTER8 $ARM8-s\$k: $G/$ARM8-s\$k/ (adapter-step\$k of $RUN)"
done
gsutil -q cp $RD/adapter.npz $G/$ARM8-s$LAST/adapter.npz
gsutil -q cp $RD/adapter.json $G/$ARM8-s$LAST/adapter.json
gsutil -q cp $RD/summary.json $G/$ARM8-s$LAST/summary.json
echo "ADAPTER8 $ARM8-s$LAST: $G/$ARM8-s$LAST/ (the last step of $RUN)"
gsutil -q cp $X/trained.json $SFT/trained.json
echo "TRAINED8_DONE $SFT/trained.json"
EOF
)
  detach "$NAME" bash "$f"
  ;;
report8)
  # §19 G4-8.10: the choice on the search part, both variants of the rule on the decision part, G4-8.6 per candidate;
  # §22 G4-9.11: G4-9's with G4-8's choice beside it (a reference) and compared on the decision part
  only 0
  ready
  CANDS=${CANDIDATES:?the candidates, $ARM8-s<step> ...}
  O6=${O6:?R-oud, R-nieuw or open (G4-8.0)}
  case $O6 in R-oud|R-nieuw|open) ;; *) echo "O6 is R-oud, R-nieuw or open, not $O6"; exit 1 ;; esac
  for m in $CANDS; do cand8 "$m" || { echo "CANDIDATES: $m is no candidate of $ARM8"; exit 1; }; done
  NAT=$G/naturalness-$ARM8
  REP=$G/report-$ARM8
  G5S=$G/data/G4-5-split
  for f in "$NAT/predictions.parquet" "$NAT/predictions-c2.parquet" "$G5S/split.json" "$G5S/refqe.parquet" \
    "$SFT/manifest.json" "$SFT/trained.json" "$RL/data/plan/plan.parquet"; do
    gsutil -q stat "$f" || { echo "no $f (classify; classify-c2; split5; refqe5; sftset; adapter)"; exit 1; }
  done
  for m in G4-0 $CANDS; do gsutil -q stat "$SFT/div/$m/samples/w0/DONE" || { echo "no $SFT/div/$m/samples/w0/DONE (generate JOB=div)"; exit 1; }; done
  RUN=$(gsutil cat "$SFT/trained.json" | python3 -c "import json, sys; print(json.load(sys.stdin)['run'])")
  NAME=${J}_report8_$TAG
  X=$W8/report8-$TAG
  ARGS=""
  for m in $CANDS; do ARGS="$ARGS --candidate $m=$X/$m --lora $m=$X/lora-$m.json --div $m=$X/div/$m"; done
  ARGS="$ARGS --div G4-0=$X/div/G4-0"
  for m in $REFS; do ARGS="$ARGS --reference $m=$X/$m"; done
  [ "$ARM8" != G4-9 ] || ARGS="$ARGS --compare $COMPARE9"
  SMOKE=""
  if [ -n "${SMOKE_RUN:-}" ]; then
    gsutil -q stat "$SFT/smoke8-$SMOKE_RUN.json" || { echo "no $SFT/smoke8-$SMOKE_RUN.json"; exit 1; }
    SMOKE="--smoke $X/smoke8.json"
  fi
  f=$(body "$NAME" <<EOF
set -euo pipefail
$CPU
mkdir -p $X/logs
for m in $CANDS G4-0 $REFS; do mkdir -p $X/\$m; gsutil -m -q rsync -r -x 'translations/.*' $G/eval/\$m $X/\$m; done
for m in $CANDS G4-0; do mkdir -p $X/div/\$m; gsutil -m -q rsync -r $SFT/div/\$m/samples $X/div/\$m; done
gsutil -q cp $G/data/eval/evalsets.parquet $G5S/split.json $G5S/refqe.parquet $NAT/predictions.parquet $NAT/predictions-c2.parquet $X/
gsutil -q cp $RL/data/plan/plan.parquet $X/g47-plan.parquet
gsutil -q cp $G/runs/$RUN/summary.json $X/summary.json
gsutil -q cp $SFT/manifest.json $X/manifest.json
if [ -n "$SMOKE" ]; then gsutil -q cp $SFT/smoke8-${SMOKE_RUN:-}.json $X/smoke8.json; fi
# G4-8.6: each candidate's update norm from translate.py's LORA line in its translate logs
for m in $CANDS; do
  gsutil -m -q cp "$G/logs/g4a_translate_\${m}_*" $X/logs/
  $FPY -I -u $CODE/g4arm8.py loraline --logs "$X/logs/g4a_translate_\${m}_*" --out $X/lora-\$m.json
done
EXTRA=""
if gsutil -q stat $NAT/pll.parquet; then gsutil -q cp $NAT/pll.parquet $X/; EXTRA="--pll $X/pll.parquet"; fi
if gsutil -q stat $REP/tally.json; then gsutil -q cp $REP/tally.json $X/; EXTRA="\$EXTRA --tally $X/tally.json"; fi
$FPY -I -u $CODE/g4arm8.py report8 --evalsets $X/evalsets.parquet --split $X/split.json --refqe $X/refqe.parquet --base $X/G4-0 --predictions $X/predictions.parquet --predictions-c2 $X/predictions-c2.parquet --summary $X/summary.json --manifest $X/manifest.json --rlval $X/g47-plan.parquet --o6 $O6 --arm=$ARM8$ARGS $SMOKE \$EXTRA --out $X/report.md
gsutil -m -q cp $X/report.md $X/report.json $REP/
echo "REPORT8_TRAIN_DONE $REP/report.md"
EOF
)
  detach "$NAME" bash "$f"
  ;;
*)
  echo "unknown stage $STAGE"
  exit 2
  ;;
esac
