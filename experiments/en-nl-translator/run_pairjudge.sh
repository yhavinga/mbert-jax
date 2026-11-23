#!/bin/bash
# The measurement G4-judge (protocol-g4.md §17: Gemma 4 31B-it as a judge of two translations) on mbert-mix (v4-32,
# us-central2-b), one stage per call on a worker; g4judge_run.sh (co1, mbert-ops) starts the stages and waits for their
# EXIT lines. Every stage runs detached from ssh, with a fresh log ~/logs/pj_<stage>_$TAG.log that ends in "EXIT <code>"
# (also copied to $GJ/logs/). The venvs, the tokenizer and the classifier's venv come from run_g4arm.sh setup.
# Step 1, the validation:
#   bash run_pairjudge.sh valdata      w0 (CPU): pairjudge.py valdata, the five blind packages in both orders, to
#                                      $GJ/val/data/ (prompts-w<k>.parquet, refs.parquet without text, valdata.json)
#   SET=val|bon bash run_pairjudge.sh score   every worker (its TPU): pairjudge.py score of its prompts (one prefill
#                                      each), to $GJ/<SET>/scores/w<k>/
#   bash run_pairjudge.sh valreport    w0 (CPU): pairjudge.py valreport, D1-D3, to $GJ/val/report.md/.json (GATE line)
# Step 2, only after GATE pass:
#   bash run_pairjudge.sh bonpairs     w0 (CPU): pairjudge.py bonpairs (G4-0, G4-7-i15 against G4-0 greedy), to $GJ/bon/data/
#   MODEL=G4-0|G4-7-i15 bash run_pairjudge.sh select   w0 (CPU): pairjudge.py select, to $GJ/<MODEL>/select/
#   bash run_pairjudge.sh texts        w0 (CPU): g4bon.py texts of the 12 outputs, to $GJ/naturalness/texts.parquet
#   bash run_pairjudge.sh classify     w0 (its TPU): C1, run_arms.sh classify with NAT=$GJ/naturalness (job
#                                      arms_classify_<TAG>c), unchanged
#   bash run_pairjudge.sh bonreport    w0 (CPU): pairjudge.py bonreport, to $GJ/bon/report/
#   bash run_pairjudge.sh blind        w0 (CPU): pairjudge.py blind (G4-7-i15, judgeq against G4-0 greedy), to
#                                      $S/blind/G4-judge-i15/ (refused if a key.jsonl is there)
# A, judge or anchor (G4-judge.12), from step 2's scores:
#   MODEL=G4-0|G4-7-i15 bash run_pairjudge.sh aselect   w0 (CPU): pairjudge.py anchorselect, to $GJ/<MODEL>/select-anchor/
#   bash run_pairjudge.sh atexts       w0 (CPU): g4bon.py texts of the 14 outputs, to $GJ/anchor/naturalness/texts.parquet
#   bash run_pairjudge.sh aclassify    w0 (its TPU): C1, run_arms.sh classify with NAT=$GJ/anchor/naturalness (job
#                                      arms_classify_<TAG>c), unchanged
#   bash run_pairjudge.sh areport      w0 (CPU): pairjudge.py bonreport with janchor, to $GJ/anchor/report/
#   bash run_pairjudge.sh ablind       w0 (CPU): pairjudge.py blind --kind janchor (G4-7-i15), to
#                                      $S/blind/G4-judge-anchor-i15/ (refused if a key.jsonl is there)
# B, a judge that reasons first (G4-judge.13), validation and time only:
#   bash run_pairjudge.sh rvaldata     w0 (CPU): pairjudge.py valdata --prompt reason, to $GJ/reason/val/data/
#   bash run_pairjudge.sh reason       every worker (its TPU): pairjudge.py reason of its prompts, to
#                                      $GJ/reason/val/scores/w<k>/ (shards with the reasoning, timing-*.json)
#   bash run_pairjudge.sh rreport      w0 (CPU): pairjudge.py reasonreport, to $GJ/reason/val/report.md/.json
# The code is beside this script (a copy of $G/scripts-g4j-<commit>/). TAG names the jobs of one chain.
set -euo pipefail
STAGE=${1:?a stage: valdata, score, valreport, bonpairs, select, texts, classify, bonreport, blind, aselect, atexts, aclassify, areport, ablind, rvaldata, reason, rreport}
R=gs://t5-dutch-english-usc2/mbert/translator
S=$R/sft
G=$S/g4arm
GJ=$G/judge
GB=$G/bon
GS=$G/sampling
G5S=$G/data/G4-5-split
DECODE_CKPT=$R/judge/gemma-4-31b-it-maxtext/0/items            # the conversion every Gemma 4 decode uses
SHM=/dev/shm/translator
G_SHM=$SHM/g4arm
GEN=/dev/shm/translate-generic
FORK_COMMIT=f4d8615f478b6c18de9df8f2dff1f57f775a2521             # the fork of every Gemma 4 decode since 14 Nov
FPY=$GEN/venv-fork-${FORK_COMMIT:0:9}/bin/python
TOK_DEC=$GEN/tokenizer-gemma4
WORK=$G_SHM/judge-work                                          # MaxText's output and the compilation cache
BLIND_OUT=$S/blind/G4-judge-i15
ABLIND_OUT=$S/blind/G4-judge-anchor-i15
GA=$GJ/anchor
GR=$GJ/reason/val
PACKAGES="G4-7 G4-7-tg G4-bon-i15c2q RB3 RB2"
MODELS="G4-0 G4-7-i15"
CODE=$(cd "$(dirname "$0")" && pwd)
W=$(hostname | sed 's/.*-w-//')
TAG=${TAG:-manual}
mkdir -p "$G_SHM" ~/logs

detach() {  # detach NAME COMMAND...: the command in the background, log ~/logs/NAME.log ending in EXIT <code>
  local name=$1
  shift
  [ -e ~/logs/$name.log ] && { echo "w$W: ~/logs/$name.log exists; a new start takes a new TAG"; exit 1; }
  printf '%q ' "$@" > ~/job_$name.sh
  setsid nohup bash -c "bash ~/job_$name.sh > ~/logs/$name.log 2>&1; echo EXIT \$? >> ~/logs/$name.log; gsutil -q cp ~/logs/$name.log $GJ/logs/$name-w$W.log" \
    > /dev/null 2>&1 < /dev/null &
  echo "started $name on w$W"
}

tpu_free() {
  [ -z "$(sudo lsof -t /dev/accel* /dev/vfio/* 2> /dev/null)" ] || { echo "w$W: the TPU is in use"; exit 1; }
}

only() {  # only WORKER: the stage runs on that worker alone
  [ "$W" = "$1" ] || { echo "$STAGE runs on w$1 only"; exit 1; }
}

fork_ready() {
  [ -x "$FPY" ] && [ -s "$TOK_DEC/tokenizer.json" ] || { echo "w$W: no fork venv or tokenizer (run_g4arm.sh setup)"; exit 1; }
}

samples_of() {  # samples_of MODEL: where its 8 samples are (G4-0: the sampling probe's, as G4-bon reuses them)
  case $1 in G4-0) echo "$GS/G4-0" ;; G4-7-i15) echo "$GB/G4-7-i15" ;; *) echo "no model $1" >&2; exit 1 ;; esac
}

# A worker of the v4-32 as a TPU of its own 4 chips (run_g4arm.sh)
ONE_HOST="export TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 TPU_PROCESS_BOUNDS=1,1,1 TPU_VISIBLE_CHIPS=0,1,2,3 TPU_PROCESS_PORT=8476
export TPU_PROCESS_ADDRESSES=localhost:8476 CLOUD_TPU_TASK_ID=0 ALLOW_MULTIPLE_LIBTPU_LOAD=1
export XLA_FLAGS=--xla_dump_hlo_as_text=false TPU_LOG_DIR=$SHM/tpu_logs TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 HF_HUB_DISABLE_PROGRESS_BARS=1"
PREFLIGHT="timeout 600 $FPY -c \"import jax; assert (jax.device_count(), jax.process_count()) == (4, 1), (jax.device_count(), jax.process_count()); print('devices', jax.devices())\" || { echo 'preflight failed: not 4 chips of one process'; exit 2; }"
INPUTS="gsutil -q cp $GB/data/evalsets.parquet $G5S/split.json"   # + the destination: the search part's inputs
NOTEXT="-x 'translations/.*'"                                    # an evaluation's or a sample's pairs and scores
PAIRSONLY="-x 'translations/.*|scores/.*'"

case $STAGE in
valdata)
  only 0
  fork_ready
  X=$G_SHM/judge-valdata-$TAG
  {
    echo "set -euo pipefail"
    echo "mkdir -p $X/blind"
    for p in $PACKAGES; do echo "gsutil -m -q cp -r $S/blind/$p $X/blind/"; done
    echo "HF_HUB_OFFLINE=1 JAX_PLATFORMS=cpu $FPY -I -u $CODE/pairjudge.py valdata --blind $X/blind --out $X/out"
    echo "gsutil -m -q cp $X/out/* $GJ/val/data/"
    echo "echo \"VALDATA_DONE $GJ/val/data/\""
  } > ~/job_body_pj_valdata_$TAG.sh
  detach "pj_valdata_$TAG" bash ~/job_body_pj_valdata_$TAG.sh
  ;;
score)
  SET=${SET:?SET=val or SET=bon}
  case $SET in val|bon) ;; *) echo "SET is val or bon, not $SET"; exit 1 ;; esac
  tpu_free
  fork_ready
  X=$G_SHM/judge-score-$SET-$TAG
  cat > ~/job_body_pj_score_${SET}_$TAG.sh <<EOF
set -euo pipefail
$ONE_HOST
mkdir -p $X $WORK
gsutil -q cp $GJ/$SET/data/prompts-w$W.parquet $X/prompts.parquet
$PREFLIGHT
$FPY -u $CODE/pairjudge.py score --prompts $X/prompts.parquet --out $GJ/$SET/scores/w$W --ckpt $DECODE_CKPT --tokenizer $TOK_DEC --work-dir $WORK
EOF
  detach "pj_score_${SET}_$TAG" bash ~/job_body_pj_score_${SET}_$TAG.sh
  ;;
valreport)
  only 0
  fork_ready
  X=$G_SHM/judge-valreport-$TAG
  {
    echo "set -euo pipefail"
    echo "mkdir -p $X"
    echo "gsutil -q cp $GJ/val/data/refs.parquet $X/"
    for k in 0 1 2 3; do echo "mkdir -p $X/scores/w$k && gsutil -m -q cp '$GJ/val/scores/w$k/shard-*.parquet' $X/scores/w$k/"; done
    echo "JAX_PLATFORMS=cpu $FPY -I -u $CODE/pairjudge.py valreport --refs $X/refs.parquet --scores '$X/scores/w*/shard-*.parquet' --out $X/report.md"
    echo "gsutil -m -q cp $X/report.md $X/report.json $GJ/val/"
    echo "echo \"VALREPORT_DONE $GJ/val/report.md\""
  } > ~/job_body_pj_valreport_$TAG.sh
  detach "pj_valreport_$TAG" bash ~/job_body_pj_valreport_$TAG.sh
  ;;
bonpairs)
  only 0
  fork_ready
  X=$G_SHM/judge-bonpairs-$TAG
  ARGS=""
  {
    echo "set -euo pipefail"
    echo "mkdir -p $X/anchor"
    echo "$INPUTS $X/"
    echo "gsutil -m -q rsync -r $PAIRSONLY $G/eval/G4-0 $X/anchor"
    for m in $MODELS; do
      src=$(samples_of "$m")
      for k in 1 2 3 4 5 6 7 8; do echo "mkdir -p $X/$m/t10s$k && gsutil -m -q rsync -r $PAIRSONLY $src/t10s$k $X/$m/t10s$k"; done
      echo "gsutil -q cp $GB/$m/select/choice.parquet $X/$m-choice.parquet"
      ARGS="$ARGS --model $m=$X/$m,$X/$m-choice.parquet"
    done
    echo "JAX_PLATFORMS=cpu $FPY -I -u $CODE/pairjudge.py bonpairs$ARGS --anchor $X/anchor --evalsets $X/evalsets.parquet --split $X/split.json --out $X/out"
    echo "gsutil -m -q cp $X/out/* $GJ/bon/data/"
    echo "echo \"BONPAIRS_DONE $GJ/bon/data/\""
  } > ~/job_body_pj_bonpairs_$TAG.sh
  detach "pj_bonpairs_$TAG" bash ~/job_body_pj_bonpairs_$TAG.sh
  ;;
select)
  only 0
  fork_ready
  MODEL=${MODEL:?G4-0 or G4-7-i15}
  src=$(samples_of "$MODEL")
  X=$G_SHM/judge-select-$MODEL-$TAG
  {
    echo "set -euo pipefail"
    echo "mkdir -p $X"
    echo "$INPUTS $X/"
    for k in 1 2 3 4 5 6 7 8; do echo "mkdir -p $X/t10s$k && gsutil -m -q rsync -r $NOTEXT $src/t10s$k $X/t10s$k"; done
    echo "gsutil -q cp $GB/$MODEL/select/choice.parquet $X/choice.parquet"
    for k in 0 1 2 3; do echo "mkdir -p $X/scores/w$k && gsutil -m -q cp '$GJ/bon/scores/w$k/shard-*.parquet' $X/scores/w$k/"; done
    echo "JAX_PLATFORMS=cpu $FPY -I -u $CODE/pairjudge.py select --model $MODEL --samples $X --choice $X/choice.parquet --scores '$X/scores/w*/shard-*.parquet' --evalsets $X/evalsets.parquet --split $X/split.json --out $X/out"
    echo "gsutil -m -q rsync -r $X/out $GJ/$MODEL/select"
    echo "echo \"JSELECT_DONE $GJ/$MODEL/select/\""
  } > ~/job_body_pj_select_${MODEL}_$TAG.sh
  detach "pj_select_${MODEL}_$TAG" bash ~/job_body_pj_select_${MODEL}_$TAG.sh
  ;;
texts|bonreport|atexts|areport)
  only 0
  fork_ready
  X=$G_SHM/judge-$STAGE-$TAG
  # A (atexts, areport): janchor beside the 12 outputs of step 2, C1 of A's own classify run
  case $STAGE in a*) NATDIR=$GA/naturalness ;; *) NATDIR=$GJ/naturalness ;; esac
  PAIRS=""
  OUTS=""
  {
    echo "set -euo pipefail"
    echo "mkdir -p $X"
    echo "$INPUTS $G5S/refqe.parquet $X/"
    for m in $MODELS; do
      src=$(samples_of "$m")
      echo "mkdir -p $X/$m/greedy $X/$m/t10s1 $X/$m/bon $X/$m/judge"
      echo "gsutil -m -q rsync -r $NOTEXT $G/eval/$m $X/$m/greedy"
      echo "gsutil -m -q rsync -r $NOTEXT $src/t10s1 $X/$m/t10s1"
      echo "gsutil -m -q rsync -r $GB/$m/select $X/$m/bon"
      echo "gsutil -m -q rsync -r $GJ/$m/select $X/$m/judge"
      PAIRS="$PAIRS --pairs '$m=$X/$m/greedy/pairs/pairs-w*.parquet' --pairs '$m/t10s1=$X/$m/t10s1/pairs/pairs-w*.parquet'"
      PAIRS="$PAIRS --pairs '$m/qe=$X/$m/bon/qe/pairs/pairs-w*.parquet' --pairs '$m/c2q=$X/$m/bon/c2q/pairs/pairs-w*.parquet'"
      PAIRS="$PAIRS --pairs '$m/judge=$X/$m/judge/judge/pairs/pairs-w*.parquet' --pairs '$m/judgeq=$X/$m/judge/judgeq/pairs/pairs-w*.parquet'"
      OUTS="$OUTS --model $m=$X/$m/greedy,$X/$m/t10s1,$X/$m/bon,$X/$m/judge"
      case $STAGE in
      a*)
        echo "mkdir -p $X/$m/anchor && gsutil -m -q rsync -r $GJ/$m/select-anchor $X/$m/anchor"
        PAIRS="$PAIRS --pairs '$m/janchor=$X/$m/anchor/janchor/pairs/pairs-w*.parquet'"
        OUTS="$OUTS,$X/$m/anchor"
        ;;
      esac
    done
    case $STAGE in
    texts|atexts)
      echo "JAX_PLATFORMS=cpu $FPY -I -u $CODE/g4bon.py texts --evalsets $X/evalsets.parquet --split $X/split.json$PAIRS --out $X/texts.parquet"
      echo "gsutil -q cp $X/texts.parquet $NATDIR/texts.parquet"
      echo "echo '$MODELS' | gsutil -q cp - $NATDIR/models.txt"
      echo "echo \"JTEXTS_DONE $NATDIR/texts.parquet\""
      ;;
    *)
      REPORT=$GJ/bon/report
      JUDGEC1=""
      if [ "$STAGE" = areport ]; then
        REPORT=$GA/report
        echo "gsutil -q cp $GJ/naturalness/predictions.parquet $X/judge-predictions.parquet"
        JUDGEC1=" --judge-predictions $X/judge-predictions.parquet"
      fi
      echo "gsutil -q cp $NATDIR/predictions.parquet $X/predictions.parquet"
      echo "gsutil -q cp $GS/naturalness/predictions.parquet $X/probe-predictions.parquet"
      echo "gsutil -q cp $GB/naturalness/G4-0_G4-7-i15/predictions.parquet $X/bon-predictions.parquet"
      echo "JAX_PLATFORMS=cpu $FPY -I -u $CODE/pairjudge.py bonreport --evalsets $X/evalsets.parquet --split $X/split.json --refqe $X/refqe.parquet --anchor $X/G4-0/greedy --predictions $X/predictions.parquet --probe-predictions $X/probe-predictions.parquet --bon-predictions $X/bon-predictions.parquet$JUDGEC1$OUTS --out $X/report.md"
      echo "gsutil -m -q cp $X/report.md $X/report.json $REPORT/"
      echo "echo \"JBONREPORT_DONE $REPORT/report.md\""
      ;;
    esac
  } > ~/job_body_pj_${STAGE}_$TAG.sh
  detach "pj_${STAGE}_$TAG" bash ~/job_body_pj_${STAGE}_$TAG.sh
  ;;
aselect)
  only 0
  fork_ready
  MODEL=${MODEL:?G4-0 or G4-7-i15}
  src=$(samples_of "$MODEL")
  X=$G_SHM/judge-aselect-$MODEL-$TAG
  {
    echo "set -euo pipefail"
    echo "mkdir -p $X/select $X/anchor"
    echo "$INPUTS $X/"
    for k in 1 2 3 4 5 6 7 8; do echo "mkdir -p $X/t10s$k && gsutil -m -q rsync -r $NOTEXT $src/t10s$k $X/t10s$k"; done
    echo "gsutil -m -q rsync -r $NOTEXT $G/eval/G4-0 $X/anchor"
    echo "gsutil -q cp $GJ/$MODEL/select/choice.parquet $X/select/choice.parquet"
    echo "JAX_PLATFORMS=cpu $FPY -I -u $CODE/pairjudge.py anchorselect --model $MODEL --select $X/select --samples $X --anchor $X/anchor --evalsets $X/evalsets.parquet --split $X/split.json --out $X/out"
    echo "gsutil -m -q rsync -r $X/out $GJ/$MODEL/select-anchor"
    echo "echo \"ASELECT_DONE $GJ/$MODEL/select-anchor/\""
  } > ~/job_body_pj_aselect_${MODEL}_$TAG.sh
  detach "pj_aselect_${MODEL}_$TAG" bash ~/job_body_pj_aselect_${MODEL}_$TAG.sh
  ;;
aclassify)
  only 0
  gsutil -q stat "$GA/naturalness/texts.parquet" || { echo "no $GA/naturalness/texts.parquet (atexts)"; exit 1; }
  # C1: run_arms.sh's classify, unchanged, on A's texts; its job is arms_classify_<TAG>c
  NAT=$GA/naturalness TAG=${TAG}c bash "$CODE/run_arms.sh" classify
  ;;
ablind)
  only 0
  fork_ready
  ! gsutil -q stat "$ABLIND_OUT/key.jsonl" || { echo "$ABLIND_OUT has a key.jsonl: the package stays as it is"; exit 1; }
  X=$G_SHM/judge-ablind-$TAG
  {
    echo "set -euo pipefail"
    echo "mkdir -p $X/select $X/anchor"
    echo "$INPUTS $X/"
    echo "gsutil -m -q rsync -r $GJ/G4-7-i15/select-anchor $X/select"
    echo "gsutil -m -q rsync -r $PAIRSONLY $G/eval/G4-0 $X/anchor"
    echo "gsutil -q cp $BLIND_OUT/key.jsonl $X/earlier-key.jsonl"
    echo "JAX_PLATFORMS=cpu $FPY -I -u $CODE/pairjudge.py blind --kind janchor --model G4-7-i15 --select $X/select --anchor $X/anchor --evalsets $X/evalsets.parquet --split $X/split.json --overlap $X/earlier-key.jsonl --overlap-name G4-judge-i15 --out $X/out"
    echo "gsutil -m -q cp $X/out/judge-input.jsonl $X/out/key.jsonl $X/out/select-report.json $ABLIND_OUT/"
    echo "echo \"JBLIND_DONE $ABLIND_OUT/\""
  } > ~/job_body_pj_ablind_$TAG.sh
  detach "pj_ablind_$TAG" bash ~/job_body_pj_ablind_$TAG.sh
  ;;
rvaldata)
  only 0
  fork_ready
  X=$G_SHM/judge-rvaldata-$TAG
  {
    echo "set -euo pipefail"
    echo "mkdir -p $X/blind"
    for p in $PACKAGES; do echo "gsutil -m -q cp -r $S/blind/$p $X/blind/"; done
    echo "HF_HUB_OFFLINE=1 JAX_PLATFORMS=cpu $FPY -I -u $CODE/pairjudge.py valdata --prompt reason --blind $X/blind --out $X/out"
    echo "gsutil -m -q cp $X/out/* $GR/data/"
    echo "echo \"VALDATA_DONE $GR/data/\""
  } > ~/job_body_pj_rvaldata_$TAG.sh
  detach "pj_rvaldata_$TAG" bash ~/job_body_pj_rvaldata_$TAG.sh
  ;;
reason)
  tpu_free
  fork_ready
  X=$G_SHM/judge-reason-$TAG
  cat > ~/job_body_pj_reason_$TAG.sh <<EOF
set -euo pipefail
$ONE_HOST
mkdir -p $X $WORK
gsutil -q cp $GR/data/prompts-w$W.parquet $X/prompts.parquet
$PREFLIGHT
$FPY -u $CODE/pairjudge.py reason --prompts $X/prompts.parquet --out $GR/scores/w$W --ckpt $DECODE_CKPT --tokenizer $TOK_DEC --work-dir $WORK
EOF
  detach "pj_reason_$TAG" bash ~/job_body_pj_reason_$TAG.sh
  ;;
rreport)
  only 0
  fork_ready
  X=$G_SHM/judge-rreport-$TAG
  {
    echo "set -euo pipefail"
    echo "mkdir -p $X/logs"
    echo "gsutil -q cp $GR/data/refs.parquet $X/"
    for k in 0 1 2 3; do
      echo "mkdir -p $X/scores/w$k $X/base/w$k && gsutil -m -q cp '$GR/scores/w$k/shard-*.parquet' '$GR/scores/w$k/timing-*.json' $X/scores/w$k/"
      echo "gsutil -m -q cp '$GJ/val/scores/w$k/shard-*.parquet' $X/base/w$k/"
    done
    echo "gsutil -m -q cp '$GJ/logs/pj_score_bon_*-w*.log' $X/logs/"
    echo "JAX_PLATFORMS=cpu $FPY -I -u $CODE/pairjudge.py reasonreport --refs $X/refs.parquet --scores '$X/scores/w*/shard-*.parquet' --base-scores '$X/base/w*/shard-*.parquet' --timing '$X/scores/w*/timing-*.json' --base-logs '$X/logs/pj_score_bon_*-w*.log' --out $X/report.md"
    echo "gsutil -m -q cp $X/report.md $X/report.json $GR/"
    echo "echo \"RREPORT_DONE $GR/report.md\""
  } > ~/job_body_pj_rreport_$TAG.sh
  detach "pj_rreport_$TAG" bash ~/job_body_pj_rreport_$TAG.sh
  ;;
classify)
  only 0
  gsutil -q stat "$GJ/naturalness/texts.parquet" || { echo "no $GJ/naturalness/texts.parquet (texts)"; exit 1; }
  # C1: run_arms.sh's classify, unchanged, on this measurement's texts; its job is arms_classify_<TAG>c
  NAT=$GJ/naturalness TAG=${TAG}c bash "$CODE/run_arms.sh" classify
  ;;
blind)
  only 0
  fork_ready
  ! gsutil -q stat "$BLIND_OUT/key.jsonl" || { echo "$BLIND_OUT has a key.jsonl: the package stays as it is"; exit 1; }
  X=$G_SHM/judge-blind-$TAG
  {
    echo "set -euo pipefail"
    echo "mkdir -p $X/select $X/anchor"
    echo "$INPUTS $X/"
    echo "gsutil -m -q rsync -r $GJ/G4-7-i15/select $X/select"
    echo "gsutil -m -q rsync -r $PAIRSONLY $G/eval/G4-0 $X/anchor"
    echo "JAX_PLATFORMS=cpu $FPY -I -u $CODE/pairjudge.py blind --model G4-7-i15 --select $X/select --anchor $X/anchor --evalsets $X/evalsets.parquet --split $X/split.json --out $X/out"
    echo "gsutil -m -q cp $X/out/judge-input.jsonl $X/out/key.jsonl $X/out/select-report.json $BLIND_OUT/"
    echo "echo \"JBLIND_DONE $BLIND_OUT/\""
  } > ~/job_body_pj_blind_$TAG.sh
  detach "pj_blind_$TAG" bash ~/job_body_pj_blind_$TAG.sh
  ;;
*)
  echo "unknown stage $STAGE"
  exit 2
  ;;
esac
