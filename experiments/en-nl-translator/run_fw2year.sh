#!/bin/bash
# The FineWeb-2 year check (fw2year.py) on a worker of mbert-mix, one stage per call; translator_fw2year_run.sh (co1)
# starts the stages and waits for their EXIT lines. The classifier itself runs as run_arms.sh classify with NAT set to
# this check's directory (the same job, code and classifier as the arms'). Every stage that takes longer than a few
# seconds runs detached from ssh, with a fresh log ~/logs/<job>.log that ends in "EXIT <code>".
#   bash run_fw2year.sh check                    READY when this worker has what the stages need
#   bash run_fw2year.sh data                     CPU (nice, ionice): fw2year.py data from Hugging Face, to $F/data/
#                                                (meta.parquet, data.json) and $F/bt/texts.parquet  (job fw2_data_$TAG)
#   NAT=$F/bt TAG=<t>bt bash run_arms.sh classify    the classifier on $F/bt/texts.parquet, to $F/bt/predictions.parquet
#   bash run_fw2year.sh report                   CPU: fw2year.py report, to $F/report/      (job fw2_report_$TAG)
#   G4_FILES='NAME=gs://... ...' bash run_fw2year.sh g4texts   CPU: the Gemma 4 sizes' texts beside the base model's,
#                                                to $F/g4/ (texts.parquet, names.json)       (job fw2_g4texts_$TAG)
#   NAT=$F/g4 TAG=<t>g4 bash run_arms.sh classify    the classifier on $F/g4/texts.parquet
#   bash run_fw2year.sh g4report                 CPU: fw2year.py g4report, to $F/report/    (job fw2_g4report_$TAG)
# The code is beside this script (a copy of $F/scripts/). TAG names the jobs of one chain.
set -euo pipefail
STAGE=${1:?a stage: check, data, report, g4texts, g4report}
S=gs://t5-dutch-english-usc2/mbert/translator/sft
F=${FW2_ROOT:-$S/fw2year}
BASE_A=$S/arms                          # the classifier and the base model's texts (the arms' chain)
RE_SUMMARY=$S/re/classify/summary.json  # the five BT sources (RE's reclassify, 13 Nov)
RES=gs://t5-dutch-english-usc2/mbert/translator
SHM=/dev/shm/translator
W_DIR=$SHM/sft/fw2year
PY=$SHM/venv/bin/python                 # the translator's venv: pyarrow, tokenizers, huggingface_hub, langid
TOKENIZER=/dev/shm/flan_nl/tokenizer-27b
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

case $STAGE in
check)
  "$PY" -c "import pyarrow, tokenizers, huggingface_hub, langid" || { echo "w$W: $PY lacks a package"; exit 1; }
  [ -s $TOKENIZER/tokenizer.json ] || { echo "w$W: no $TOKENIZER"; exit 1; }
  for f in fw2year.py select_passages.py armeval.py naturalness.py run_arms.sh COMMIT; do
    [ -s "$CODE/$f" ] || { echo "w$W: no $CODE/$f"; exit 1; }
  done
  free=$(df -BG --output=avail "$HOME" | tail -1 | tr -dc 0-9)
  [ "$free" -ge 3 ] || { echo "w$W: only $free GB free on the boot disk"; exit 1; }
  echo "READY w$W: code $(cat "$CODE/COMMIT"), boot disk $free GB free"
  ;;
data)
  ! gsutil -q stat "$F/data/data.json" || { echo "$F/data/data.json exists: the data stay as they are"; exit 1; }
  D=$W_DIR/data-$TAG
  cat > ~/job_body_fw2_data_$TAG.sh <<EOF
set -euo pipefail
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1
mkdir -p $D/in
gsutil -q cp $RES/mainset/resources/opentaal-wordlist.txt $RES/pilot/resources/idioms.json $D/in/
gsutil -q cp $BASE_A/data/classifier.parquet $D/in/
$PY -I -u $CODE/fw2year.py data --classifier-data $D/in/classifier.parquet --tokenizer $TOKENIZER --wordlist $D/in/opentaal-wordlist.txt --idioms $D/in/idioms.json --out $D/out --processes ${PROCESSES:-48} --commit $COMMIT
gsutil -q cp $D/out/texts.parquet $F/bt/texts.parquet
gsutil -q cp $D/out/meta.parquet $F/data/meta.parquet
gsutil -q cp $D/out/data.json $F/data/data.json
echo "FW2_DATA_DONE: $F/data/ and $F/bt/texts.parquet"
EOF
  # CPU only, beside the TPU jobs of other chains: the lowest CPU and I/O priority
  detach "fw2_data_$TAG" nice -n 19 ionice -c3 bash ~/job_body_fw2_data_$TAG.sh
  ;;
report)
  X=$W_DIR/report-$TAG
  cat > ~/job_body_fw2_report_$TAG.sh <<EOF
set -euo pipefail
mkdir -p $X
gsutil -q cp $F/bt/predictions.parquet $F/data/meta.parquet $F/data/data.json $X/
gsutil -q cp $BASE_A/naturalness/classifier/results.json $X/
gsutil -q cp $RE_SUMMARY $X/re-summary.json
$PY -I -u $CODE/fw2year.py report --meta $X/meta.parquet --predictions $X/predictions.parquet --data-json $X/data.json --classifier-results $X/results.json --bt-sources $X/re-summary.json --commit $COMMIT --out $X/fineweb2-year-check.md --json $X/fineweb2-year-check.json
gsutil -q cp $X/fineweb2-year-check.md $X/fineweb2-year-check.json $F/report/
echo "FW2_REPORT: $F/report/fineweb2-year-check.md"
EOF
  detach "fw2_report_$TAG" nice -n 19 bash ~/job_body_fw2_report_$TAG.sh
  ;;
g4texts)
  [ -n "${G4_FILES:-}" ] || { echo "g4texts: G4_FILES is empty"; exit 1; }
  X=$W_DIR/g4texts-$TAG
  args=()
  copies=""
  for item in $G4_FILES; do
    name=${item%%=*}
    copies+="gsutil -q cp ${item#*=} $X/in/$name.parquet"$'\n'
    args+=("$name=$X/in/$name.parquet")
  done
  cat > ~/job_body_fw2_g4texts_$TAG.sh <<EOF
set -euo pipefail
mkdir -p $X/in
$copies
gsutil -q cp $BASE_A/naturalness/texts.parquet $X/arms-texts.parquet
$PY -I -u $CODE/fw2year.py g4texts --base-texts $X/arms-texts.parquet --texts ${args[*]} --out $X/texts.parquet --names $X/names.json
gsutil -q cp $X/texts.parquet $X/names.json $F/g4/
echo "FW2_G4TEXTS_DONE: $F/g4/"
EOF
  detach "fw2_g4texts_$TAG" nice -n 19 bash ~/job_body_fw2_g4texts_$TAG.sh
  ;;
g4report)
  X=$W_DIR/g4report-$TAG
  cat > ~/job_body_fw2_g4report_$TAG.sh <<EOF
set -euo pipefail
mkdir -p $X
gsutil -q cp $F/g4/predictions.parquet $F/g4/names.json $X/
gsutil -q cp $BASE_A/naturalness/classifier/results.json $X/
$PY -I -u $CODE/fw2year.py g4report --predictions $X/predictions.parquet --names $X/names.json --classifier-results $X/results.json --out $X/gemma4-sizes-classify.md --json $X/gemma4-sizes-classify.json
gsutil -q cp $X/gemma4-sizes-classify.md $X/gemma4-sizes-classify.json $F/report/
echo "FW2_G4REPORT: $F/report/gemma4-sizes-classify.md"
EOF
  detach "fw2_g4report_$TAG" nice -n 19 bash ~/job_body_fw2_g4report_$TAG.sh
  ;;
*)
  echo "unknown stage $STAGE"
  exit 1
  ;;
esac
