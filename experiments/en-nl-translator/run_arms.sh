#!/bin/bash
# The fine-tune's 27B arms B and N and their evaluation (protocol.md §3-§5), one stage per call on a worker of mbert-mix;
# translator_sft_arms_run.sh (co1) starts the stages and waits for their EXIT lines. Every stage that takes longer than
# a few seconds runs detached from ssh, with a fresh log ~/logs/<job>.log that ends in "EXIT <code>".
#   bash run_arms.sh setup                       every worker: linger, the venvs and tokenizers it needs; READY
#   bash run_arms.sh data                        w3 (CPU): armeval.py data (kinds x and instr) and classifier-data, to
#                                                $A/data/                                   (job arms_data_$TAG)
#   ARM=B bash run_arms.sh train                 every worker at once: run_sft.sh, 27B, one epoch of $SETS/$ARM over
#                                                the 16 chips (run arm27b_v2_$ARM, log ~/logs/arm27b_v2_$ARM.log)
#   ARM=B bash run_arms.sh adapter               a worker: the run's adapter.npz, adapter.json, summary.json to $A/$ARM/
#   MODEL=B DATA=four bash run_arms.sh translate every worker: run_baseline.sh translate (its own detaching; log
#                                                ~/logs/tbl_translate_${TAG}_${MODEL}_$DATA.log), to $A/eval/$MODEL-$DATA/
#   MODEL=B DATA=four bash run_arms.sh score     every worker: four: run_baseline.sh score; x and instr: collect,
#                                                MetricX-24 (reference where there is one, and QE), COMET-22 on the rows
#                                                with a reference                 (job arms_score_${TAG}_${MODEL}_$DATA)
#   MODEL=B bash run_arms.sh summary             a worker: run_baseline.sh summary of the four sets (job arms_summary_...)
#   bash run_arms.sh texts                       a worker (CPU): armeval.py texts, to $A/naturalness/texts.parquet
#   bash run_arms.sh classifier                  a worker (its TPU): S @ 653k converted, naturalness.py classifier, to
#                                                $A/naturalness/classifier/                 (job arms_classifier_$TAG)
#   bash run_arms.sh classify                    a worker (its TPU): naturalness.py classify, to $A/naturalness/
#   bash run_arms.sh pll                         a worker (its TPU): naturalness.py pll, to $A/naturalness/
#   bash run_arms.sh blind                       a worker (CPU): armeval.py blind, to $S/blind/{B,N}/
#   bash run_arms.sh report                      a worker (CPU): armeval.py report, to $A/report/
# MODEL: base (TranslateGemma 27B), B, N, or Ninstr (arm N with the conventional sentence, chat-v1-tag-conventional).
# DATA: four (the four sets of the baseline, $E/data), x (FLAN 1,000 + blind 100 + in-domain 4,519) or instr (WMT24++
# 998 + FLAN 1,000). The code is beside this script (a copy of $A/scripts/). TAG names the jobs of one chain.
set -euo pipefail
STAGE=${1:?a stage: setup, data, train, adapter, translate, score, summary, texts, classifier, classify, pll, blind, report}
S=gs://t5-dutch-english-usc2/mbert/translator/sft
A=$S/arms
E=gs://t5-dutch-english-usc2/mbert/translator/eval
M=gs://t5-dutch-english-usc2/mbert/translator/mainset
SETS=$S/data/arms-v2
BLIND=$S/blind/fixed-2026-11-10/blind100.jsonl
S653=gs://t5-dutch-english-usc2/mbert/culturax-nl-230m-chunk1024/ckpt_keep/653000
SHM=/dev/shm/translator
W_DIR=$SHM/sft/arms
PY=$SHM/venv/bin/python                 # the translator's venv: pyarrow, tokenizers, transformers, sacrebleu
SFT_PY=$SHM/smoke-lora/venv/bin/python  # setup_sft.sh: MaxText 0.2.5 and Tunix
DPY=$HOME/mbert-dumb/.venv/bin/python   # the venv of the DUMB rounds (jax, flax, optax, transformers, pyarrow)
DUMB=$HOME/mbert-dumbx                  # a checkout of mbert-jax branch dumb-extra
# dumb-extra @ ae11478, as G's evaluations used it; 3c8bcb9 (the branch's head) changes only experiments/dumb-extra
DUMB_COMMIT=ae114789bf2eae153e4e3bea579299a617dcd417
CODE=$(cd "$(dirname "$0")" && pwd)
W=$(hostname | sed 's/.*-w-//')
TAG=${TAG:-manual}
COMMIT=${COMMIT:-unknown}
mkdir -p "$W_DIR" ~/logs

detach() {  # detach NAME COMMAND...: the command in the background, log ~/logs/NAME.log ending in EXIT <code>
  local name=$1
  shift
  [ -e ~/logs/$name.log ] && { echo "w$W: ~/logs/$name.log exists; a new start takes a new TAG"; exit 1; }
  printf '%q ' "$@" > ~/job_$name.sh
  setsid nohup bash -c "bash ~/job_$name.sh > ~/logs/$name.log 2>&1; echo EXIT \$? >> ~/logs/$name.log" \
    > /dev/null 2>&1 < /dev/null &
  echo "started $name on w$W"
}

tpu_free() {
  [ -z "$(sudo lsof -t /dev/accel* /dev/vfio/* 2> /dev/null)" ] || { echo "w$W: the TPU is in use"; exit 1; }
}

# A host of the v4-32 as a TPU of its own 4 chips (as run_baseline.sh score); XLA_FLAGS set and the TPU logs in
# /dev/shm keep the boot disk free
ONE_HOST="export TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 TPU_PROCESS_BOUNDS=1,1,1 TPU_VISIBLE_CHIPS=0,1,2,3 TPU_PROCESS_PORT=8476
export TPU_PROCESS_ADDRESSES=localhost:8476 CLOUD_TPU_TASK_ID=0 ALLOW_MULTIPLE_LIBTPU_LOAD=1
export XLA_FLAGS=--xla_dump_hlo_as_text=false TPU_LOG_DIR=$SHM/tpu_logs TOKENIZERS_PARALLELISM=false"

dumb_ready() {
  [ -x "$DPY" ] || { echo "w$W: no $DPY"; exit 1; }
  [ "$(git -C "$DUMB" rev-parse HEAD 2> /dev/null)" = "$DUMB_COMMIT" ] || { echo "w$W: $DUMB is not at $DUMB_COMMIT"; exit 1; }
  [ -z "$(git -C "$DUMB" status --porcelain --untracked-files=no)" ] || { echo "w$W: $DUMB has changes"; exit 1; }
}

model_dir() {  # model_dir MODEL DATA: the bucket directory of a model's output on a data kind. Its basename must be
  # unique: run_baseline.sh keeps a run's local files in $SHM/eval/<basename>
  echo "$A/eval/$1-$2"
}

data_dir() {  # data_dir DATA: the E of run_baseline.sh (its data in $E/data/)
  case $1 in
    four) echo "$E" ;;
    x|instr) echo "$A/data/$1" ;;
    *) echo "unknown DATA $1" >&2; exit 1 ;;
  esac
}

check_model_data() {
  case "$MODEL/$DATA" in
    base/x|B/four|B/x|N/four|N/x|Ninstr/instr) ;;
    *) echo "MODEL/DATA $MODEL/$DATA is not one of the chain's: base/x, B/four, B/x, N/four, N/x, Ninstr/instr"; exit 1 ;;
  esac
}

case $STAGE in
setup)
  [ "$(loginctl show-user "$(id -un)" -p Linger --value 2> /dev/null)" = yes ] || sudo loginctl enable-linger "$(id -un)"
  "$PY" -c "import sacrebleu, pyarrow, transformers, tokenizers, yaml" || { echo "w$W: $PY lacks a package"; exit 1; }
  [ -x $SHM/comet-venv/bin/python ] && [ -d $SHM/comet ] || { echo "w$W: no COMET venv or model (setup_comet.sh)"; exit 1; }
  [ -x /dev/shm/flan_nl/venv/bin/python ] || { echo "w$W: no FLAN-nl venv (setup_flan.sh)"; exit 1; }
  [ -s /dev/shm/flan_nl/tokenizer-27b/tokenizer.json ] || { echo "w$W: no tokenizer of 27B"; exit 1; }
  [ -x "$SFT_PY" ] || { echo "w$W: no SFT venv (setup_sft.sh)"; exit 1; }
  [ -d $SHM/metricx ] || echo "w$W: no $SHM/metricx (metricx_qe.py fetches MetricX-24 XXL from HF, ~25 GB)"
  if [ "$W" = 0 ] || [ "$W" = 1 ]; then
    dumb_ready
    "$DPY" -c "import jax, flax, optax, orbax.checkpoint, transformers, pyarrow, etils" || { echo "w$W: $DPY lacks a package"; exit 1; }
  fi
  free=$(df -BG --output=avail "$HOME" | tail -1 | tr -dc 0-9)
  [ "$free" -ge 3 ] || { echo "w$W: only $free GB free on the boot disk"; exit 1; }
  echo "READY w$W: boot disk $free GB free, /dev/shm $(df -h /dev/shm | tail -1 | awk '{print $4}') free"
  ;;
data)
  ! gsutil -q stat "$A/data/x/data/manifest.json" || { echo "$A/data/x has a manifest.json: the data stay as they are"; exit 1; }
  IN=$SHM/sft/in
  D=$W_DIR/data-$TAG
  cat > ~/job_body_arms_data_$TAG.sh <<EOF
set -euo pipefail
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=true
mkdir -p $IN/run $IN/long $IN/flan-eval $IN/eval $D/in
[ -s $IN/run/pairs.parquet ] || gsutil -q cp $M/run/pairs/pairs.parquet $IN/run/
[ -s $IN/long/pairs.parquet ] || gsutil -q cp $M/run-long/pairs/pairs.parquet $IN/long/
gsutil -q cp $M/flan-eval/eval-rows.parquet $D/in/
gsutil -q cp $E/data/evalsets.parquet $D/in/
gsutil -q cp $BLIND $D/in/
gsutil -q cp $SETS/B/eval-meta.parquet $SETS/B/eval.parquet $D/in/
$PY -I -u $CODE/armeval.py data --kind x --flan-eval $D/in/eval-rows.parquet --blind $D/in/blind100.jsonl --evalsets $D/in/evalsets.parquet --eval-meta $D/in/eval-meta.parquet --eval $D/in/eval.parquet --tokenizer /dev/shm/flan_nl/tokenizer-27b --out $D/x
$PY -I -u $CODE/armeval.py data --kind instr --flan-eval $D/in/eval-rows.parquet --evalsets $D/in/evalsets.parquet --tokenizer /dev/shm/flan_nl/tokenizer-27b --out $D/instr
$PY -I -u $CODE/armeval.py classifier-data --eval-meta $D/in/eval-meta.parquet --pairs $IN/run/pairs.parquet $IN/long/pairs.parquet --out $D/classifier.parquet
gsutil -m -q rsync -r $D/x $A/data/x/data
gsutil -m -q rsync -r $D/instr $A/data/instr/data
gsutil -q cp $D/classifier.parquet $A/data/classifier.parquet
echo "DATA: $A/data/"
EOF
  detach "arms_data_$TAG" bash ~/job_body_arms_data_$TAG.sh
  ;;
train)
  ARM=${ARM:?B or N}
  tpu_free
  D=$SHM/sft/data/arms-v2-$ARM
  mkdir -p "$D"
  gsutil -m -q rsync -r "$SETS/$ARM" "$D"
  # One epoch (sft_lora.py's step_limit), sequences of 2,048, one per chip, over the 16 chips (protocol §3). TRAIN_EXTRA:
  # MaxText overrides of an amendment (protocol §9), after these
  # shellcheck disable=SC2086
  MULTIHOST=1 NAME=arm27b_v2_$ARM MODEL=27b DATA=$D/train.parquet TEMPLATE=$D/template.jinja ROOT=$A \
    bash "$CODE/run_sft.sh" per_device_batch_size=1 max_target_length=2048 ${TRAIN_EXTRA:-}
  ;;
adapter)
  ARM=${ARM:?B or N}
  RUN=$A/runs/arm27b_v2_$ARM
  # run_sft.sh copies the run's files after its log's EXIT line: give that copy up to 3 minutes
  for _ in $(seq 1 36); do
    gsutil -q stat "$RUN/adapter.npz" && gsutil -q stat "$RUN/adapter.json" && gsutil -q stat "$RUN/summary.json" && break
    sleep 5
  done
  gsutil -q cp "$RUN/adapter.npz" "$RUN/adapter.json" "$RUN/summary.json" "$A/$ARM/"
  echo "ADAPTER $ARM: $A/$ARM/ $(gsutil cat "$A/$ARM/adapter.json" | tr -d '\n ' | cut -c1-300)"
  ;;
translate)
  MODEL=${MODEL:?base, B, N or Ninstr}
  DATA=${DATA:?four, x or instr}
  check_model_data
  tpu_free
  case $MODEL in
    base) ADAPTER=; PROMPT= ;;
    B|N) ADAPTER=$A/$MODEL; PROMPT= ;;
    Ninstr) ADAPTER=$A/N; PROMPT=chat-v1-tag-conventional ;;
  esac
  if [ -n "$ADAPTER" ]; then
    gsutil -q stat "$ADAPTER/adapter.npz" || { echo "w$W: no $ADAPTER/adapter.npz"; exit 1; }
  fi
  out=$(model_dir "$MODEL" "$DATA")
  xe=$(data_dir "$DATA")
  name=${TAG}_${MODEL}_$DATA
  # MODEL here is the translator of this chain (base, B, N, Ninstr); run_translate.sh reads MODEL as TranslateGemma's
  # size, so it is set to 27b for the call
  MODEL=27b E=$xe B=$out ADAPTER=$ADAPTER PROMPT=$PROMPT TAG=$name bash "$CODE/run_baseline.sh" translate
  ;;
score)
  MODEL=${MODEL:?base, B, N or Ninstr}
  DATA=${DATA:?four, x or instr}
  check_model_data
  tpu_free
  OUT=$(model_dir "$MODEL" "$DATA")
  XE=$(data_dir "$DATA")
  name=arms_score_${TAG}_${MODEL}_$DATA
  if [ "$DATA" = four ]; then
    # the baseline's score stage, unchanged (it skips scores already in $OUT/scores/)
    detach "$name" env B="$OUT" TAG="${TAG}_${MODEL}_$DATA" bash "$CODE/run_baseline.sh" score
    exit 0
  fi
  L=$W_DIR/eval/$MODEL-$DATA
  cat > ~/job_body_$name.sh <<EOF
set -euo pipefail
mkdir -p $L/data $L/translations $L/out
gsutil -q cp $XE/data/evalsets.parquet $XE/data/manifest.json $XE/data/segments-w$W.parquet $L/data/
gsutil -m -q rsync -r $OUT/translations/w$W $L/translations
$PY -I $CODE/evalsets.py collect --data $L/data --worker $W --shards $L/translations --out $L/out/pairs-w$W.parquet
gsutil -q cp $L/out/pairs-w$W.parquet $OUT/pairs/pairs-w$W.parquet
if gsutil -q stat $OUT/scores/metricx-w$W.parquet; then
  echo "w$W: MetricX scores are in $OUT/scores/ before"
else
  (
    $ONE_HOST
    export JAX_COMPILATION_CACHE_DIR=$SHM/jax_cache_metricx
    timeout 600 $PY -c "import jax; assert (jax.device_count(), jax.process_count()) == (4, 1), (jax.device_count(), jax.process_count())"
    # a row without a reference gets no reference score (metricx_qe.py rows_to_score), only QE
    $PY -I -u $CODE/metricx_qe.py score --pairs $L/out/pairs-w$W.parquet --pair source:hypothesis:reference --pair source:hypothesis --model xxl --root $SHM/metricx --empty-candidates --out $L/out/metricx-w$W.parquet
  )
  gsutil -q cp $L/out/metricx-w$W.parquet $OUT/scores/metricx-w$W.parquet
fi
n=\$($PY -I -c "import pyarrow.parquet as pq, pyarrow.compute as pc, sys; t = pq.read_table(sys.argv[1]); t = t.filter(pc.is_valid(t['reference'])); pq.write_table(t, sys.argv[2]); print(t.num_rows)" $L/out/pairs-w$W.parquet $L/out/pairs-ref-w$W.parquet)
if [ "\$n" = 0 ]; then
  echo "w$W: no row with a reference, no COMET-22"
elif gsutil -q stat $OUT/scores/comet-w$W.parquet; then
  echo "w$W: COMET scores are in $OUT/scores/ before"
else
  $SHM/comet-venv/bin/python -I -u $CODE/comet_score.py score --root $SHM/comet --pairs $L/out/pairs-ref-w$W.parquet --out $L/out/comet-w$W.parquet
  gsutil -q cp $L/out/comet-w$W.parquet $OUT/scores/comet-w$W.parquet
fi
echo "SCORED w$W: $OUT/scores/ (\$n rows with a reference)"
EOF
  detach "$name" bash ~/job_body_$name.sh
  ;;
summary)
  MODEL=${MODEL:?B or N}
  label="TranslateGemma 27B + LoRA $MODEL (arms-v2, QE ≤ 7,5; protocol en-nl-sft)"
  detach "arms_summary_${TAG}_$MODEL" env B="$(model_dir "$MODEL" four)" ADAPTER="$A/$MODEL" LABEL="$label" \
    TAG="${TAG}_${MODEL}_four" COMMIT="$COMMIT" bash "$CODE/run_baseline.sh" summary
  ;;
texts)
  # MODELS: the models whose translations exist (a test before the arms: base)
  MODELS=${MODELS:-base B N Ninstr}
  T=$W_DIR/texts-$TAG
  wmt=""; flan=""; fetch=""
  for m in $MODELS; do
    case $m in
      base) fetch="$fetch
gsutil -q cp $E/baseline-tg27b/translations.parquet $T/base-four.parquet
gsutil -m -q cp '$(model_dir base x)/pairs/pairs-w*.parquet' $T/base-x/"
            wmt="$wmt 'base=$T/base-four.parquet'"; flan="$flan 'base=$T/base-x/pairs-w*.parquet'" ;;
      B|N) fetch="$fetch
gsutil -q cp $(model_dir "$m" four)/translations.parquet $T/$m-four.parquet
gsutil -m -q cp '$(model_dir "$m" x)/pairs/pairs-w*.parquet' $T/$m-x/"
           wmt="$wmt '$m=$T/$m-four.parquet'"; flan="$flan '$m=$T/$m-x/pairs-w*.parquet'" ;;
      Ninstr) fetch="$fetch
gsutil -m -q cp '$(model_dir Ninstr instr)/pairs/pairs-w*.parquet' $T/Ninstr-instr/"
              wmt="$wmt 'Ninstr=$T/Ninstr-instr/pairs-w*.parquet'"; flan="$flan 'Ninstr=$T/Ninstr-instr/pairs-w*.parquet'" ;;
      *) echo "unknown model $m"; exit 1 ;;
    esac
  done
  cat > ~/job_body_arms_texts_$TAG.sh <<EOF
set -euo pipefail
mkdir -p $T/base-x $T/B-x $T/N-x $T/Ninstr-instr
$fetch
gsutil -q cp $E/data/evalsets.parquet $T/evalsets.parquet
$PY -I -u $CODE/armeval.py texts --evalsets $T/evalsets.parquet --wmt $wmt --flan $flan --out $T/texts.parquet
gsutil -q cp $T/texts.parquet $A/naturalness/texts.parquet
echo "TEXTS: $A/naturalness/texts.parquet"
EOF
  detach "arms_texts_$TAG" bash ~/job_body_arms_texts_$TAG.sh
  ;;
classifier)
  tpu_free
  dumb_ready
  ! gsutil -q stat "$A/naturalness/classifier/results.json" || { echo "$A/naturalness/classifier has a results.json"; exit 1; }
  CK=$W_DIR/s653
  X=$W_DIR/classifier-$TAG
  cat > ~/job_body_arms_classifier_$TAG.sh <<EOF
set -euo pipefail
$ONE_HOST
export JAX_COMPILATION_CACHE_DIR=$W_DIR/jax_cache_s PYTHONPATH=src
cd $DUMB
timeout 600 $DPY -c "import jax; assert (jax.device_count(), jax.process_count()) == (4, 1), (jax.device_count(), jax.process_count())"
# S @ 653k as dumb-extra's dumbx.sh converts a checkpoint for scripts/dumb/finetune.py
if [ ! -f $CK/params.pkl ]; then
  gsutil -q stat $S653/commit_success.txt
  $DPY scripts/convert_gcs_checkpoint.py --checkpoint $S653 --params-only --output $CK --tokenizer yhavinga/dutch-llama-tokenizer --global-attn-every-n-layers 3 < /dev/null
  rm -f $CK/model.safetensors
  $DPY - $CK/params.pkl <<'PY'
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
  gsutil -q cp gs://t5-dutch-english-usc2/mbert/phase2048/scripts/verify_ckpt.py \$HOME/verify_ckpt.py
  [ "\$(md5sum \$HOME/heldout.parquet 2>/dev/null | cut -d' ' -f1)" = b2f579a876d06eef40c94270537bb7af ] || gsutil -q cp gs://t5-dutch-english-usc2/mbert/heldout/nl_part_00097_first40000.parquet \$HOME/heldout.parquet
  $DPY \$HOME/verify_ckpt.py $CK < /dev/null
fi
mkdir -p $X
gsutil -q cp $A/data/classifier.parquet $X/classifier.parquet
$DPY -u $CODE/naturalness.py classifier --dumb-code $DUMB --checkpoint $CK --data $X/classifier.parquet --out $X/model
gsutil -m -q cp $X/model/results.json $X/model/params.pkl $X/model/config.json $A/naturalness/classifier/
echo "CLASSIFIER: $A/naturalness/classifier/"
EOF
  detach "arms_classifier_$TAG" bash ~/job_body_arms_classifier_$TAG.sh
  ;;
classify)
  tpu_free
  dumb_ready
  X=$W_DIR/classify-$TAG
  cat > ~/job_body_arms_classify_$TAG.sh <<EOF
set -euo pipefail
$ONE_HOST
export JAX_COMPILATION_CACHE_DIR=$W_DIR/jax_cache_s PYTHONPATH=src
cd $DUMB
mkdir -p $X/model
gsutil -m -q cp $A/naturalness/classifier/params.pkl $A/naturalness/classifier/config.json $X/model/
gsutil -q cp $A/naturalness/texts.parquet $X/texts.parquet
$DPY -u $CODE/naturalness.py classify --dumb-code $DUMB --model $X/model --texts $X/texts.parquet --out $X/predictions.parquet
gsutil -q cp $X/predictions.parquet $A/naturalness/predictions.parquet
echo "CLASSIFIED: $A/naturalness/predictions.parquet"
EOF
  detach "arms_classify_$TAG" bash ~/job_body_arms_classify_$TAG.sh
  ;;
pll)
  tpu_free
  dumb_ready
  X=$W_DIR/pll-$TAG
  cat > ~/job_body_arms_pll_$TAG.sh <<EOF
set -euo pipefail
$ONE_HOST
export JAX_COMPILATION_CACHE_DIR=$W_DIR/jax_cache_s PYTHONPATH=src
cd $DUMB
mkdir -p $X
gsutil -q cp $A/naturalness/texts.parquet $X/texts.parquet
$DPY -u $CODE/naturalness.py pll --dumb-code $DUMB --checkpoint $S653 --texts $X/texts.parquet --out $X/pll.parquet
gsutil -q cp $X/pll.parquet $A/naturalness/pll.parquet
echo "PLL: $A/naturalness/pll.parquet"
EOF
  detach "arms_pll_$TAG" bash ~/job_body_arms_pll_$TAG.sh
  ;;
blind)
  ! gsutil -q stat "$S/blind/B/key.jsonl" || { echo "$S/blind/B has a key.jsonl: the package stays as it is"; exit 1; }
  X=$W_DIR/blind-$TAG
  cat > ~/job_body_arms_blind_$TAG.sh <<EOF
set -euo pipefail
mkdir -p $X/base $X/B $X/N
gsutil -q cp $BLIND $X/blind100.jsonl
for m in base B N; do gsutil -m -q cp "$A/eval/\$m-x/pairs/pairs-w*.parquet" $X/\$m/; done
$PY -I -u $CODE/armeval.py blind --blind $X/blind100.jsonl --base "$X/base/pairs-w*.parquet" --arm "B=$X/B/pairs-w*.parquet" "N=$X/N/pairs-w*.parquet" --out $X/out
for arm in B N; do gsutil -q cp $X/out/\$arm/judge-input.jsonl $X/out/\$arm/key.jsonl $S/blind/\$arm/; done
echo "BLIND: $S/blind/B/, $S/blind/N/ (judge-input.jsonl; key.jsonl)"
EOF
  detach "arms_blind_$TAG" bash ~/job_body_arms_blind_$TAG.sh
  ;;
report)
  X=$W_DIR/report-$TAG
  cat > ~/job_body_arms_report_$TAG.sh <<EOF
set -euo pipefail
mkdir -p $X
gsutil -q cp $E/data/evalsets.parquet $X/evalsets.parquet
gsutil -q cp $E/baseline-tg27b/scores.parquet $X/base-four-scores.parquet
gsutil -q cp $A/data/x/data/evalsets.parquet $X/x-evalsets.parquet
for m in B N; do gsutil -q cp $A/eval/\$m-four/scores.parquet $X/\$m-four-scores.parquet; done
for d in base-x B-x N-x Ninstr-instr; do
  mkdir -p $X/\$d/pairs $X/\$d/scores
  gsutil -m -q cp "$A/eval/\$d/pairs/pairs-w*.parquet" $X/\$d/pairs/
  gsutil -m -q cp "$A/eval/\$d/scores/*.parquet" $X/\$d/scores/
done
gsutil -m -q cp $A/naturalness/classifier/results.json $A/naturalness/predictions.parquet $A/naturalness/pll.parquet $A/naturalness/texts.parquet $X/
$PY -I -u $CODE/armeval.py report --evalsets $X/evalsets.parquet --baseline-scores $X/base-four-scores.parquet --four B=$X/B-four-scores.parquet N=$X/N-four-scores.parquet --x-data $X/x-evalsets.parquet --x base=$X/base-x B=$X/B-x N=$X/N-x --instr Ninstr=$X/Ninstr-instr --classifier $X/results.json --predictions $X/predictions.parquet --pll $X/pll.parquet --texts $X/texts.parquet --commit $COMMIT --out $X/report.md --json $X/report.json
gsutil -q cp $X/report.md $X/report.json $A/report/
echo "REPORT: $A/report/report.md"
EOF
  detach "arms_report_$TAG" bash ~/job_body_arms_report_$TAG.sh
  ;;
*)
  echo "unknown stage $STAGE"
  exit 2
  ;;
esac
