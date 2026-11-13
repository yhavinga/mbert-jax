#!/bin/bash
# Part B of the length trial of 13 Nov (bt_length.py) on a worker of mbert-mix: TranslateGemma 27B (without LoRA, its
# own chat template) translates the English of part A (the condition stukken of Gemma 4, docs-en.parquet) back into
# Dutch, whole (heel) and in pieces of whole paragraphs up to ~512 tokens (stukken); MetricX-24-QE en->nl per aligned
# paragraph; the report. One stage per call; every stage that takes longer than a few seconds runs detached from ssh,
# with a fresh log ~/logs/lentg_<stage>_$TAG.log that ends in "EXIT <code>" (translator_length_run.sh waits for it).
#   bash run_bt_length_tg.sh check               every worker: the venvs, tokenizer, MetricX and a free TPU; READY
#   bash run_bt_length_tg.sh data                w0 (CPU): bt_length.py segments of $OB/data/docs-en.parquet for 4
#                                                workers, to $OB/seg/
#   CLASSES="d512 d1k" bash run_bt_length_tg.sh translate
#                                                every worker (its TPU): stukken (class d512) and the heel classes of
#                                                CLASSES of its segments, to $OB/{heel,stukken}/w<k>/<class>/; a class
#                                                that runs out of memory is marked (oom.txt) and not fatal
#   bash run_bt_length_tg.sh collect             w0 (CPU): collect and the units not scored, split for 4 workers
#   bash run_bt_length_tg.sh metricx             every worker (its TPU): MetricX-24-QE en->nl of its pending units
#   bash run_bt_length_tg.sh report              w0 (CPU): the report of part B, to $OB/report/
# TAG names the jobs of one chain. The code is beside this script (a copy of $OB/scripts/).
set -euo pipefail
STAGE=${1:?a stage: check, data, translate, collect, metricx, report}
OB=${OB:-gs://t5-dutch-english-usc2/mbert/translator/sft/length}
SHM=/dev/shm/translator
L=$SHM/length
PY=$SHM/venv/bin/python                # the translator's venv (CPU): pyarrow, tokenizers, transformers, sacrebleu
TPY=/dev/shm/flan_nl/venv/bin/python   # FLAN-nl's venv: MaxText 0.2.5, translate.py
TOK=/dev/shm/flan_nl/tokenizer-27b
CKPT=gs://t5-dutch-english-usc2/mbert/flan-nl/translategemma-27b-maxtext/0/items
CODE=$(cd "$(dirname "$0")" && pwd)
W=$(hostname | sed 's/.*-w-//')
TAG=${TAG:-manual}
mkdir -p "$L" ~/logs

detach() {  # detach NAME COMMAND...: the command in the background, log ~/logs/NAME.log ending in EXIT <code>
  local name=$1
  shift
  [ -e ~/logs/$name.log ] && { echo "w$W: ~/logs/$name.log exists; a new start takes a new TAG"; exit 1; }
  printf '%q ' "$@" > ~/job_$name.sh
  setsid nohup bash -c "bash ~/job_$name.sh > ~/logs/$name.log 2>&1; echo EXIT \$? >> ~/logs/$name.log; gsutil -q cp ~/logs/$name.log $OB/logs/$name-w$W.log" \
    > /dev/null 2>&1 < /dev/null &
  echo "started $name on w$W"
}

tpu_free() {
  [ -z "$(sudo lsof -t /dev/accel* /dev/vfio/* 2> /dev/null)" ] || { echo "w$W: the TPU is in use"; exit 1; }
}

# A host of the v4-32 as a TPU of its own 4 chips (run_arms.sh ONE_HOST)
ONE_HOST="export TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 TPU_PROCESS_BOUNDS=1,1,1 TPU_VISIBLE_CHIPS=0,1,2,3 TPU_PROCESS_PORT=8476
export TPU_PROCESS_ADDRESSES=localhost:8476 CLOUD_TPU_TASK_ID=0 ALLOW_MULTIPLE_LIBTPU_LOAD=1
export XLA_FLAGS=--xla_dump_hlo_as_text=false TPU_LOG_DIR=$SHM/tpu_logs TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1"

case $STAGE in
check)
  "$PY" -c "import pyarrow, transformers, tokenizers, sacrebleu, yaml" || { echo "w$W: $PY lacks a package"; exit 1; }
  [ -x "$TPY" ] || { echo "w$W: no $TPY"; exit 1; }
  [ -s "$TOK/tokenizer.json" ] || { echo "w$W: no tokenizer in $TOK"; exit 1; }
  [ -d "$SHM/metricx" ] || { echo "w$W: no $SHM/metricx"; exit 1; }
  gsutil -q stat "${CKPT%/0/items}/done.txt" || { echo "w$W: ${CKPT%/0/items} has no done.txt"; exit 1; }
  tpu_free
  free=$(df -BG --output=avail "$HOME" | tail -1 | tr -dc 0-9)
  [ "$free" -ge 3 ] || { echo "w$W: only $free GB free on the boot disk"; exit 1; }
  echo "READY w$W: boot disk $free GB free, /dev/shm $(df -h /dev/shm | tail -1 | awk '{print $4}') free"
  ;;
data)
  cat > ~/job_body_lentg_data_$TAG.sh <<EOF
set -euo pipefail
mkdir -p $L/data
gsutil -q cp $OB/data/docs-en.parquet $L/data/
$PY -I -u $CODE/bt_length.py segments --docs $L/data/docs-en.parquet --tokenizer $TOK --model gemma3-27b-doc \
  --doc-model gemma3-27b-doc --source en --target nl --workers 4 --out $L/seg
gsutil -m -q rsync -r $L/seg $OB/seg
echo "DATA: \$(cat $L/seg/segments.json | tr -d '\n ')"
EOF
  detach "lentg_data_$TAG" bash ~/job_body_lentg_data_$TAG.sh
  ;;
translate)
  CLASSES=${CLASSES:?the heel classes, e.g. "d512 d1k d2k d4k d8k"}
  tpu_free
  cat > ~/job_body_lentg_translate_${TAG}.sh <<EOF
set -uo pipefail
$ONE_HOST
export JAX_COMPILATION_CACHE_DIR=$SHM/jax_cache
mkdir -p $L/seg
gsutil -m -q rsync -r $OB/seg $L/seg || exit 1
timeout 600 $TPY -c "import jax; assert (jax.device_count(), jax.process_count()) == (4, 1), (jax.device_count(), jax.process_count())" || { echo "preflight failed"; exit 2; }
run_class() {  # cond class: one class; out of memory is marked in the bucket and not fatal
  local cond=\$1 cls=\$2 out=$OB/\$1/w$W/\$2 log=~/logs/lentg_\$1_\$2_${TAG}.log shard code
  if gsutil -q stat \$out/oom.txt; then echo "\$cond \$cls: out of memory before"; return 0; fi
  if [ \$cond = stukken ]; then shard=256; else case \$cls in d512) shard=112 ;; d1k) shard=64 ;; d2k) shard=32 ;; d4k) shard=16 ;; d8k) shard=8 ;; *) shard=2 ;; esac; fi
  echo "== \$cond \$cls \$(date -u +%T)"
  $TPY -u $CODE/bt_length_translate.py --segments $L/seg/\$cond/segments-w$W.parquet --out \$out --tokenizer $TOK \
    --model gemma3-27b-doc --length-class \$cls --source-lang en --target-lang nl --ckpt $CKPT --work-dir $SHM/work-len \
    --shard-size \$shard > \$log 2>&1 && code=0 || code=\$?
  echo "EXIT \$code" >> \$log
  gsutil -q cp \$log $OB/logs/lentg_\$1_\$2_${TAG}-w$W.log
  grep -E '^(SHARD|DONE|MEM|engine)' \$log | tail -n 3 | cut -c1-300
  if [ \$code != 0 ]; then
    if grep -q -E 'RESOURCE_EXHAUSTED|Ran out of memory|[Oo]ut of memory' \$log; then
      grep -m1 -E 'RESOURCE_EXHAUSTED|Ran out of memory|[Oo]ut of memory' \$log | cut -c1-500 | gsutil -q cp - \$out/oom.txt
      echo "\$cond \$cls: OUT OF MEMORY (marked)"
      return 0
    fi
    tail -n 30 \$log
    return \$code
  fi
}
if grep -qw d512 $L/seg/stukken/classes.txt; then run_class stukken d512 || exit 1; fi
for cls in $CLASSES; do
  if grep -qw \$cls $L/seg/heel/classes.txt; then run_class heel \$cls || exit 1; fi
done
echo "TRANSLATED w$W: $CLASSES"
EOF
  detach "lentg_translate_$TAG" bash ~/job_body_lentg_translate_${TAG}.sh
  ;;
collect)
  cat > ~/job_body_lentg_collect_$TAG.sh <<EOF
set -euo pipefail
mkdir -p $L/heel $L/stukken $L/scores $L/data
gsutil -q cp $OB/data/docs-en.parquet $L/data/
gsutil -m -q rsync -r $OB/heel $L/heel
gsutil -m -q rsync -r $OB/stukken $L/stukken
gsutil -m -q rsync -r $OB/scores $L/scores 2> /dev/null || true
gsutil -m -q rsync -r $OB/seg $L/seg
# part B's source is the English: its paragraphs are the documents' paragraphs
$PY -I -u $CODE/bt_length.py collect --docs $L/data/docs-en.parquet --heel $L/heel --stukken $L/stukken \
  --pieces $L/seg/stukken/segments-w*.parquet --source en --target nl --out $L/collect
$PY -I -u $CODE/bt_length.py pending --units $L/collect/units.parquet --scored $L/scores/qe-*.parquet --workers 4 \
  --out $L/pending
gsutil -m -q cp $L/collect/measured.parquet $L/collect/units.parquet $OB/collect/
gsutil -m -q cp $L/pending/pending-w*.parquet $OB/pending-$TAG/
echo "COLLECTED"
EOF
  detach "lentg_collect_$TAG" bash ~/job_body_lentg_collect_$TAG.sh
  ;;
metricx)
  tpu_free
  cat > ~/job_body_lentg_metricx_$TAG.sh <<EOF
set -euo pipefail
$ONE_HOST
export JAX_COMPILATION_CACHE_DIR=$SHM/jax_cache_metricx
mkdir -p $L/pending $L/scores
gsutil -q cp $OB/pending-$TAG/pending-w$W.parquet $L/pending/
n=\$($PY -I -c "import pyarrow.parquet as pq, sys; print(pq.read_metadata(sys.argv[1]).num_rows)" $L/pending/pending-w$W.parquet)
if [ "\$n" = 0 ]; then echo "SCORED w$W: no unit"; exit 0; fi
timeout 600 $PY -c "import jax; assert (jax.device_count(), jax.process_count()) == (4, 1), (jax.device_count(), jax.process_count())"
$PY -I -u $CODE/metricx_qe.py score --pairs $L/pending/pending-w$W.parquet --pair src:cand --model xxl --root $SHM/metricx \
  --out $L/scores/qe-$TAG-w$W.parquet
gsutil -q cp $L/scores/qe-$TAG-w$W.parquet $OB/scores/
echo "SCORED w$W: \$n units"
EOF
  detach "lentg_metricx_$TAG" bash ~/job_body_lentg_metricx_$TAG.sh
  ;;
report)
  cat > ~/job_body_lentg_report_$TAG.sh <<EOF
set -euo pipefail
mkdir -p $L/scores $L/logs $L/report $L/collect
gsutil -m -q rsync -r $OB/scores $L/scores
gsutil -m -q cp $OB/collect/measured.parquet $OB/collect/units.parquet $L/collect/
gsutil -m -q cp "$OB/logs/lentg_heel_*" "$OB/logs/lentg_stukken_*" $L/logs/
echo '{"model": "TranslateGemma 27B (google/translategemma-27b-it @ 7d10f0b, MaxText 0.2.5), zonder LoRA, eigen chat-template, greedy; heel en in stukken: translate_models.yml gemma3-27b-doc (één klasse per bak, decodeerlimiet 1,5 x de bron; alle stukken in d512)", "bron": "het Engels van deel A, conditie in stukken (Gemma 4 natural-v1), met zijn alinea-indeling", "MetricX": "metricx-24-hybrid-xxl-v2p6-bfloat16, QE en->nl per uitgelijnde eenheid", "VM": "mbert-mix (v4-32), elke worker als TPU van 4 chips"}' > $L/report/setup.json
$PY -I -u $CODE/bt_length.py report --measured $L/collect/measured.parquet --units $L/collect/units.parquet \
  --qe $L/scores/qe-*.parquet --logs $L/logs --setup $L/report/setup.json --source en --target nl \
  --title "Deel B: TranslateGemma 27B en→nl, heel tegen in stukken" --out $L/report/bt-length-b.md > /dev/null
gsutil -m -q cp $L/report/bt-length-b.md $L/report/bt-length-b.json $OB/report/
echo "REPORT: $OB/report/bt-length-b.md"
EOF
  detach "lentg_report_$TAG" bash ~/job_body_lentg_report_$TAG.sh
  ;;
*)
  echo "unknown stage $STAGE"; exit 1 ;;
esac
