#!/bin/bash
# On a worker of mbert-corpus: the compared passages of the LoRA smoke test (sft_data.py's segments-*.parquet) through
# translate.py with TranslateGemma 12B, without an adapter or with one of sft_lora.py merged into the kernels, detached
# from ssh, the worker's 4 chips as a TPU of their own (as run_translate.sh).
#   VARIANT=base|N_tag|N_plain|B_plain bash run_compare.sh
# base: the base model, no tag; N_tag / N_plain: arm N's adapter with and without the tag in front of the source;
# B_plain: arm B's adapter, no tag. Shards land in $ROOT/compare/$VARIANT/<class>/. Log: ~/logs/compare_$VARIANT.log,
# a fresh file per start that ends in "EXIT <code>", copied to $ROOT/compare/logs/.
set -euo pipefail
ROOT=gs://t5-dutch-english-usc2/mbert/translator/smoke-lora
SHM=/dev/shm/translator/smoke-lora
# FLAN-nl's venv, only read: the inference path of FLAN-nl and the pilot
PY=/dev/shm/flan_nl/venv/bin/python
TOK=/dev/shm/flan_nl/tokenizer-12b
CKPT=$ROOT/translategemma-12b-maxtext
CODE=$(cd "$(dirname "$0")" && pwd)
VARIANT=${VARIANT:?base, N_tag, N_plain or B_plain}
case $VARIANT in
  base) SEGMENTS=plain; RUN= ;;
  N_tag) SEGMENTS=tag; RUN=sft12b_N ;;
  N_plain) SEGMENTS=plain; RUN=sft12b_N ;;
  B_plain) SEGMENTS=plain; RUN=sft12b_B ;;
  *) echo "unknown VARIANT $VARIANT"; exit 1 ;;
esac
NAME=compare_$VARIANT
LORA=
[ -x "$PY" ] || { echo "$(hostname): no $PY (experiments/flan-nl/setup.sh)"; exit 1; }
[ -s "$SHM/data/segments-$SEGMENTS.parquet" ] || { echo "$(hostname): no $SHM/data/segments-$SEGMENTS.parquet"; exit 1; }
[ -z "$(sudo lsof -t /dev/accel* /dev/vfio/* 2> /dev/null)" ] || { echo "$(hostname): the TPU is in use"; exit 1; }
gsutil -q stat "$CKPT/done.txt" || { echo "$(hostname): $CKPT has no done.txt"; exit 1; }
if [ -n "$RUN" ]; then
  LORA=$SHM/adapters/$RUN
  mkdir -p "$LORA"
  gsutil -q cp "$ROOT/runs/$RUN/adapter.npz" "$ROOT/runs/$RUN/adapter.json" "$LORA/"
fi
mkdir -p ~/logs

cat > ~/run_$NAME.sh <<EOF
set -uo pipefail
export TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 TPU_PROCESS_BOUNDS=1,1,1 TPU_VISIBLE_CHIPS=0,1,2,3 TPU_PROCESS_PORT=8476
export TPU_PROCESS_ADDRESSES=localhost:8476 CLOUD_TPU_TASK_ID=0 ALLOW_MULTIPLE_LIBTPU_LOAD=1
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
timeout 600 $PY -c "import jax; assert (jax.device_count(), jax.process_count()) == (4, 1), (jax.device_count(), jax.process_count()); print('devices', jax.devices())" || { echo "preflight failed: not 4 chips of one process"; exit 2; }
echo "\$(date +%H:%M:%S) $(hostname): $VARIANT, segments-$SEGMENTS, adapter ${LORA:-none}"
for class in tiny small medium long xlong; do
  $PY -u $CODE/translate.py --segments $SHM/data/segments-$SEGMENTS.parquet --out $ROOT/compare/$VARIANT/\$class \
    --ckpt $CKPT/0/items --tokenizer $TOK --model gemma3-12b --length-class \$class --source-lang en --target-lang nl \
    --work-dir $SHM/work-$VARIANT ${LORA:+--lora $LORA} || exit \$?
done
EOF
setsid nohup bash -c "bash ~/run_$NAME.sh > ~/logs/$NAME.log 2>&1; echo EXIT \$? >> ~/logs/$NAME.log; gsutil -q cp ~/logs/$NAME.log $ROOT/compare/logs/$NAME.log" > /dev/null 2>&1 < /dev/null &
echo "started $NAME on $(hostname): ~/logs/$NAME.log, shards in $ROOT/compare/$VARIANT"
