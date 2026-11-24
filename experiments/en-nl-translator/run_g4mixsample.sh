#!/bin/bash
# G4-8.15 (protocol-g4.md §19): a human sample of G4-8-s140 against G4-0 from all sources, on mbert-mix (4 x v4-8,
# us-central2-b), one stage per call on a worker; the chain g4mixsample_run.sh (co1, mbert-ops) starts the stages and
# waits for their EXIT lines. Every stage runs detached from ssh with a fresh log ~/logs/g4mx_<stage>_$TAG.log that ends
# in "EXIT <code>" (also copied to $MS/logs/). The venvs, the tokenizers and the checkpoint come from run_g4arm.sh setup.
#   bash run_g4mixsample.sh data            w0 (CPU): g4mixsample.py data (60 novel and 60 BT sources, checked against
#                                           G4-7's pool and G4-8's fresh, plan and targets) -> $MS/data/
#   MODEL=G4-0|G4-8-s140 bash run_g4mixsample.sh translate   a worker (its TPU): the 120 sources through the
#                                           evaluation's own translate_class of run_g4arm.sh (translate.py on the fork,
#                                           natural-v1, greedy, per class; G4-8-s140's LoRA merged), then evalsets.py
#                                           collect -> $MS/translations/<MODEL>/<class>/, $MS/pairs/<MODEL>.parquet
#   bash run_g4mixsample.sh pick            w0 (CPU): g4mixsample.py pick -> $MS/pick/, the sample file and its key to
#                                           $S/blind/steekproef-yeb-g4-8-mix/, the blind package to $S/blind/G4-8-romans-bt/
# A stage whose output is in the bucket already prints that and ends in EXIT 0; nothing is removed and no package is
# overwritten. The code is beside this script (a copy of $G/scripts-g4mx-<commit>/). TAG names the jobs of one chain.
set -euo pipefail
STAGE=${1:?a stage: data translate pick}
R=gs://t5-dutch-english-usc2/mbert/translator
S=$R/sft
E=$R/eval
G=$S/g4arm
RL=$G/rl/G4-7
G8=$G/data/G4-8
MS=$G/mixsample                                                 # this sample's data, translations, pairs, counts
OUT_SAMPLE=$S/blind/steekproef-yeb-g4-8-mix
OUT_JUDGE=$S/blind/G4-8-romans-bt
MODEL8=G4-8-s140
DECODE_CKPT=$R/judge/gemma-4-31b-it-maxtext/0/items            # the conversion every Gemma 4 decode uses
SHM=/dev/shm/translator
G_SHM=$SHM/g4arm
WX=$SHM/g4mix                                                   # this script's own work directories
GEN=/dev/shm/translate-generic
FORK_COMMIT=f4d8615f478b6c18de9df8f2dff1f57f775a2521             # protocol-g4.md §9: every Gemma 4 decode
FPY=$GEN/venv-fork-${FORK_COMMIT:0:9}/bin/python
SFT_PY=$SHM/smoke-lora/venv/bin/python                          # setup_sft.sh (run_g4arm.sh setup)
TOK_DEC=$GEN/tokenizer-gemma4
TOK_SFT=$SHM/g4lora/tok-gemma4-31b                              # the same tokenizer.json (run_g4arm.sh setup checks)
TMODEL=gemma4-31b
CODE=$(cd "$(dirname "$0")" && pwd)
W=$(hostname | sed 's/.*-w-//')
TAG=${TAG:-manual}
mkdir -p "$G_SHM" "$WX" ~/logs

detach() {  # detach NAME COMMAND...: the command in the background, log ~/logs/NAME.log ending in EXIT <code>
  local name=$1
  shift
  [ -e ~/logs/$name.log ] && { echo "w$W: ~/logs/$name.log exists; a new start takes a new TAG"; exit 1; }
  printf '%q ' "$@" > ~/job_$name.sh
  setsid nohup bash -c "bash ~/job_$name.sh > ~/logs/$name.log 2>&1; echo EXIT \$? >> ~/logs/$name.log; gsutil -q cp ~/logs/$name.log $MS/logs/$name-w$W.log" \
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

# A worker of the v4-32 as a TPU of its own 4 chips (run_g4arm.sh)
ONE_HOST="export TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 TPU_PROCESS_BOUNDS=1,1,1 TPU_VISIBLE_CHIPS=0,1,2,3 TPU_PROCESS_PORT=8476
export TPU_PROCESS_ADDRESSES=localhost:8476 CLOUD_TPU_TASK_ID=0 ALLOW_MULTIPLE_LIBTPU_LOAD=1
export XLA_FLAGS=--xla_dump_hlo_as_text=false TPU_LOG_DIR=$SHM/tpu_logs TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 HF_HUB_DISABLE_PROGRESS_BARS=1"
PREFLIGHT="timeout 600 $FPY -c \"import jax; assert (jax.device_count(), jax.process_count()) == (4, 1), (jax.device_count(), jax.process_count()); print('devices', jax.devices())\" || { echo 'preflight failed: not 4 chips of one process'; exit 2; }"
CPU="export JAX_PLATFORMS=cpu OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1"
# the evaluation's own translate_class (run_g4arm.sh, G4-8.7: translate.py with the class's slots of
# translate_models.yml, natural-v1, greedy, half the slots on running out of memory), so that the settings are its own
eval "$(sed -n "/^TRANSLATE_CLASS='translate_class() {\$/,/^}'\$/p" "$CODE/run_g4arm.sh")"
[ -n "${TRANSLATE_CLASS:-}" ] || { echo "no TRANSLATE_CLASS in $CODE/run_g4arm.sh"; exit 1; }
VARS="FPY=$FPY CODE=$CODE DECODE_CKPT=$DECODE_CKPT TOK_DEC=$TOK_DEC G_SHM=$G_SHM G=$G W=$W TMODEL=$TMODEL"

case $STAGE in
data)
  only 0
  ready
  NAME=g4mx_data_$TAG
  gsutil -q stat "$MS/data/data.json" && there_before "$NAME" "MIXDATA_DONE $MS/data/ (there before)"
  X=$WX/data-$TAG
  f=$(body "$NAME" <<EOF
set -euo pipefail
$CPU
mkdir -p $X/in/bt3 $X/out
gsutil -m -q cp $G/data/heldout4/eval-novels-meta.parquet $S/data/books-b1/b1-selected.parquet $RL/data/pool/pool.parquet $RL/c2/data/classifier.parquet $X/in/
gsutil -m -q cp $G8/fresh/fresh.parquet $X/in/g48-fresh.parquet
gsutil -m -q cp $G8/plan/plan.parquet $X/in/g48-plan.parquet
gsutil -m -q cp $G8/targets/targets.parquet $X/in/g48-targets.parquet
gsutil -m -q rsync -r $G/input/bt3 $X/in/bt3
nice -n 19 ionice -c3 $SFT_PY -I -u $CODE/g4mixsample.py data --novels-meta $X/in/eval-novels-meta.parquet --b1 $X/in/b1-selected.parquet --g47-pool $X/in/pool.parquet --g48-fresh $X/in/g48-fresh.parquet --g48-plan $X/in/g48-plan.parquet --g48-targets $X/in/g48-targets.parquet --c2-data $X/in/classifier.parquet --bt $X/in/bt3 --tokenizer $TOK_SFT --out $X/out
gsutil -m -q cp $X/out/evalsets.parquet $X/out/segments.parquet $X/out/classes.txt $MS/data/
gsutil -q cp $X/out/data.json $MS/data/data.json
echo "MIXDATA_DONE $MS/data/"
EOF
)
  detach "$NAME" bash "$f"
  ;;
translate)
  MODEL=${MODEL:?G4-0 or $MODEL8}
  case $MODEL in G4-0|"$MODEL8") ;; *) echo "MODEL is G4-0 or $MODEL8, not $MODEL"; exit 1 ;; esac
  ready
  NAME=g4mx_translate_${MODEL}_$TAG
  gsutil -q stat "$MS/pairs/$MODEL.parquet" && there_before "$NAME" "TRANSLATED w$W $MODEL ($MS/pairs/$MODEL.parquet there before)"
  tpu_free
  gsutil -q stat "$MS/data/data.json" || { echo "no $MS/data/data.json"; exit 1; }
  CLASSES=$(echo $(gsutil cat "$MS/data/classes.txt"))
  [ -n "$CLASSES" ] || { echo "no classes in $MS/data/classes.txt"; exit 1; }
  L=$WX/translate-$NAME
  EXTRA=
  WHAT="(no adapter)"
  if [ "$MODEL" = "$MODEL8" ]; then
    # G4-8.4: the candidate's LoRA, rank 32 / alpha 16 from G4-0 (nothing stacked); its copy stays (nothing removed)
    mkdir -p "$L/adapter"
    gsutil -q cp "$G/$MODEL/adapter.npz" "$G/$MODEL/adapter.json" "$L/adapter/"
    python3 -c "import json, sys; a = json.load(open(sys.argv[1])); assert (a['rank'], a['alpha'], a['scale']) == (32, 16.0, 0.5), a" "$L/adapter/adapter.json" \
      || { echo "$G/$MODEL/adapter.json is not rank 32, alpha 16"; exit 1; }
    EXTRA="--lora $L/adapter"
    WHAT="with its LoRA"
  fi
  {
    echo "set -uo pipefail"
    echo "$VARS"
    echo "$ONE_HOST"
    echo "export JAX_COMPILATION_CACHE_DIR=$G_SHM/jax_cache"
    echo "mkdir -p $L/data $L/translations $G_SHM/work"
    echo "gsutil -q cp $MS/data/evalsets.parquet $L/data/evalsets.parquet || exit 1"
    # evalsets.py collect reads a worker's segments as segments-w<k>.parquet
    echo "gsutil -q cp $MS/data/segments.parquet $L/data/segments-w$W.parquet || exit 1"
    echo "$PREFLIGHT"
    echo "$TRANSLATE_CLASS"
    echo "echo \"== $MODEL w$W \$(date -u +%T) $WHAT\""
    for cls in $CLASSES; do
      echo "translate_class $L/data/segments-w$W.parquet $cls $MS/translations/$MODEL/$cls ~/logs/${NAME}_$cls.log $EXTRA || exit 1"
    done
    echo "gsutil -m -q rsync -r $MS/translations/$MODEL $L/translations || exit 1"
    echo "$FPY -I $CODE/evalsets.py collect --data $L/data --worker $W --shards $L/translations --out $L/pairs.parquet || exit 1"
    echo "gsutil -q cp $L/pairs.parquet $MS/pairs/$MODEL.parquet || exit 1"
    echo "echo \"TRANSLATED w$W $MODEL: $CLASSES\""
  } > ~/job_body_$NAME.sh
  detach "$NAME" bash ~/job_body_$NAME.sh
  ;;
pick)
  only 0
  ready
  NAME=g4mx_pick_$TAG
  gsutil -q stat "$MS/pick/pick.json" && there_before "$NAME" "MIXPICK_DONE $MS/pick/ (there before)"
  # a package or a key once written stays as it is
  for p in "$OUT_SAMPLE/key.json" "$OUT_JUDGE/key.jsonl"; do
    ! gsutil -q stat "$p" || { echo "$p exists: it stays as it is"; exit 1; }
  done
  for m in G4-0 "$MODEL8"; do gsutil -q stat "$MS/pairs/$m.parquet" || { echo "no $MS/pairs/$m.parquet"; exit 1; }; done
  X=$WX/pick-$TAG
  B=$S/blind
  f=$(body "$NAME" <<EOF
set -euo pipefail
$CPU
mkdir -p $X/in/data $X/in/G4-0 $X/in/$MODEL8 $X/in/keys $X/out
gsutil -m -q cp $MS/data/evalsets.parquet $X/in/data/
gsutil -m -q cp $MS/pairs/G4-0.parquet $MS/pairs/$MODEL8.parquet $X/in/
gsutil -m -q cp '$G/eval/G4-0/pairs/pairs-w*.parquet' $X/in/G4-0/
gsutil -m -q cp '$G/eval/$MODEL8/pairs/pairs-w*.parquet' $X/in/$MODEL8/
gsutil -m -q cp $G/data/eval/evalsets.parquet $G/data/G4-5-split/split.json $X/in/
gsutil -q cp $B/G4-8-replica/key.jsonl $X/in/keys/replica.jsonl
gsutil -q cp $B/G4-judge-i15/key.jsonl $X/in/keys/judge-i15.jsonl
gsutil -q cp $B/G4-judge-anchor-i15/key.jsonl $X/in/keys/anchor-i15.jsonl
gsutil -q cp $B/steekproef-yeb-g4-8/key-map.json $X/in/keys/sample2.json
$SFT_PY -I -u $CODE/g4mixsample.py pick --data $X/in/data --mix base=$X/in/G4-0.parquet --mix $MODEL8=$X/in/$MODEL8.parquet --eval 'base=$X/in/G4-0/pairs-w*.parquet' --eval '$MODEL8=$X/in/$MODEL8/pairs-w*.parquet' --evalsets $X/in/evalsets.parquet --split $X/in/split.json --exclude-items $X/in/keys/replica.jsonl --exclude-items $X/in/keys/judge-i15.jsonl --exclude-items $X/in/keys/anchor-i15.jsonl --used-blind $X/in/keys/sample2.json --out $X/out
gsutil -q cp $X/out/judge-input.jsonl $OUT_JUDGE/judge-input.jsonl
gsutil -q cp $X/out/select-report.json $OUT_JUDGE/select-report.json
gsutil -q cp $X/out/judge-key.jsonl $OUT_JUDGE/key.jsonl
gsutil -q cp $X/out/steekproef-yeb-g4-8-mix.md $OUT_SAMPLE/steekproef-yeb-g4-8-mix.md
gsutil -q cp $X/out/key.json $OUT_SAMPLE/key.json
gsutil -q cp $X/out/pick.json $MS/pick/pick.json
echo "MIXPICK_DONE $MS/pick/ $OUT_SAMPLE/ $OUT_JUDGE/"
EOF
)
  detach "$NAME" bash "$f"
  ;;
*)
  echo "unknown stage $STAGE (data translate pick)"
  exit 1
  ;;
esac
