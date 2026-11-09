#!/bin/bash
# On worker 0 of mbert-corpus after both translation runs of the pilot: the step-2 measures of every pair (CPU), then
# MetricX-24 XXL QE in both directions on this host's TPU (4 chips), then the report for Yeb. Everything to $R/pilot/.
#   bash qe_job.sh      (started detached by the main session; log ~/logs/translator_qe.log)
set -euo pipefail
C=/dev/shm/translator/code
R=gs://t5-dutch-english-usc2/mbert/translator
PY=/dev/shm/translator/venv/bin/python
P=/dev/shm/translator/pilot
NOTE="Controle (9 nov): op de CPU in float32 gaf deze code dezelfde scores als transformers' MT5ForConditionalGeneration (6 paren, large en xxl: grootste verschil < 0,00005); op de TPU (bfloat16-producten) week xxl op die 6 paren hoogstens 0,006 af van de float32-score."
mkdir -p $P/en-nl $P/logs $P/report
gsutil -m -q rsync -r $R/pilot/en-nl $P/en-nl
$PY -I $C/pairs.py measures --select $P/select --nl-en $P/nl-en --en-nl $P/en-nl --out $P/pairs.parquet
gsutil -q cp $P/pairs.parquet $R/pilot/pairs.parquet
[ -z "$(sudo lsof -t /dev/accel* /dev/vfio/* 2> /dev/null)" ] || { echo "the TPU is in use"; exit 1; }
# This host as a TPU of its own (as in run_translate.sh)
export TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 TPU_PROCESS_BOUNDS=1,1,1 TPU_VISIBLE_CHIPS=0,1,2,3 TPU_PROCESS_PORT=8476
export TPU_PROCESS_ADDRESSES=localhost:8476 CLOUD_TPU_TASK_ID=0 ALLOW_MULTIPLE_LIBTPU_LOAD=1 HF_HUB_DISABLE_PROGRESS_BARS=1
export JAX_COMPILATION_CACHE_DIR=/dev/shm/translator/jax_cache_metricx
# The scores do not depend on the measures above: a rerun for the measures or the report keeps them
if [ ! -s $P/qe.parquet ]; then
  $PY -I -u $C/metricx_qe.py score --pairs $P/pairs.parquet --pair en:nl --pair nl:en --model xxl --out $P/qe.parquet
  gsutil -q cp $P/qe.parquet $R/pilot/qe.parquet
fi
gsutil -m -q cp "$R/pilot/logs/translator_*-w*.log" $P/logs/
$PY -I $C/report.py --pairs $P/pairs.parquet --qe $P/qe.parquet --select-report $P/select/report.json --logs $P/logs \
  --metricx-model google/metricx-24-hybrid-xxl-v2p6-bfloat16 --out $P/report \
  --wordlist /dev/shm/translator/resources/opentaal-wordlist.txt --metricx-note "$NOTE"
gsutil -m -q cp $P/report/* $R/pilot/
