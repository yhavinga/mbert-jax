#!/bin/bash
# On the judge's VM after every (re)start: prep.sh, then at the same time
#   on the CPU, once per region: transformers' reference of the first sanity rows (hf_reference.py) -> $JUDGE/reference/
#   on the TPU: the sanity set, then the MoTra pairs -> $JUDGE/scores/{sanity,motra}/ (shards that are there skipped)
# Start it detached, with its log ending in an EXIT line:
#   setsid nohup bash -c 'bash /dev/shm/judge/code/run.sh > ~/logs/judge_run.log 2>&1; echo EXIT $? >> ~/logs/judge_run.log' > /dev/null 2>&1 < /dev/null &
set -euo pipefail
ZONE=$(curl -fs -H Metadata-Flavor:Google http://metadata.google.internal/computeMetadata/v1/instance/zone | awk -F/ '{print $NF}')
case ${ZONE%-*} in
  us-east1) BUCKET=gs://t5-dutch-english-use1 ;;
  europe-west4) BUCKET=gs://t5-dutch-english ;;
  *) echo "no judge bucket for zone '$ZONE'"; exit 1 ;;
esac
JUDGE=$BUCKET/mbert/translator/judge
SHM=/dev/shm/judge
CODE=$SHM/code
PY=$SHM/venv/bin/python

bash "$CODE/prep.sh"

REF_PID=
if ! gsutil -q stat "$JUDGE/reference/ref.json"; then
  echo "== reference (CPU) in the background, log ~/logs/judge_reference.log $(date -u +%T)"
  (bash "$CODE/setup.sh" reference \
    && "$SHM/hfvenv/bin/python" -u "$CODE/hf_reference.py" --model "$SHM/hf-31b" --sanity "$SHM/sanity" --out "$SHM/ref.json" \
    && gsutil -q cp "$SHM/ref.json" "$JUDGE/reference/ref.json") > ~/logs/judge_reference.log 2>&1 &
  REF_PID=$!
fi

if [ -n "$(sudo lsof -t /dev/accel* /dev/vfio/* 2> /dev/null)" ]; then
  echo "the TPU is in use: $(sudo lsof -t /dev/accel* /dev/vfio/* | tr '\n' ' ')"
  exit 1
fi
CKPT_ITEMS=$JUDGE/gemma-4-31b-it-maxtext/0/items
[ -d "$SHM/ckpt/0/items" ] && CKPT_ITEMS=$SHM/ckpt/0/items
# The compilation cache of earlier VMs (the 60 unscanned layers compile for minutes per shape)
mkdir -p "$SHM/jax_cache"
if gsutil -q ls "$JUDGE/jax_cache/" > /dev/null 2>&1; then gsutil -m -q rsync -r "$JUDGE/jax_cache" "$SHM/jax_cache"; fi
echo "== score $(date -u +%T), parameters $CKPT_ITEMS"
JUDGE_JAX_CACHE=$SHM/jax_cache JUDGE_JAX_CACHE_GCS=$JUDGE/jax_cache "$PY" -u "$CODE/judge.py" score \
  --ckpt "$CKPT_ITEMS" --tokenizer "$SHM/tokenizer" \
  --pairs "$SHM/sanity" --out "$JUDGE/scores/sanity" --pairs "$SHM/motra" --out "$JUDGE/scores/motra"
if [ -n "$REF_PID" ]; then
  wait "$REF_PID" || { echo "the reference failed: ~/logs/judge_reference.log"; exit 1; }
fi
echo "== reports $(date -u +%T)"
bash "$CODE/reports.sh"
echo "== run done $(date -u +%T)"
