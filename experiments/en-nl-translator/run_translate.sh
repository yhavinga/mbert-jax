#!/bin/bash
# On every worker of mbert-corpus: this worker's segments through TranslateGemma 27B in one direction, length class
# after length class (experiments/flan-nl/translate.py, copied beside this script), detached from ssh, each worker a TPU
# of its own 4 chips. Like experiments/flan-nl/translate.sh, but with the languages as parameters and everything local
# in /dev/shm/translator; it runs in FLAN-nl's venv (MaxText 0.2.5, /dev/shm/flan_nl/venv), which it does not change.
#   SOURCE=nl TARGET=en SEGMENTS=$R/pilot/select/segments-w{w}.parquet OUT=$R/pilot/nl-en/w{w} NAME=translator_nl_en \
#     bash run_translate.sh
# {w} in SEGMENTS and OUT is this worker's number. Log: ~/logs/$NAME.log, a fresh file per start that ends in
# "EXIT <code>"; a copy goes to $ROOT/pilot/logs/. Shards land in $OUT/<class>/; a second start skips every shard there.
set -euo pipefail
ROOT=gs://t5-dutch-english-usc2/mbert/translator
CKPT=gs://t5-dutch-english-usc2/mbert/flan-nl/translategemma-27b-maxtext
SHM=/dev/shm/translator
PY=/dev/shm/flan_nl/venv/bin/python
TOK=/dev/shm/flan_nl/tokenizer-27b
CODE=$(cd "$(dirname "$0")" && pwd)
WORKER=$(hostname | sed 's/.*-w-//')
SOURCE=${SOURCE:?nl or en}
TARGET=${TARGET:?en or nl}
SEGMENTS=${SEGMENTS:?the segments of a worker, w in braces for its number}
OUT=${OUT:?the bucket directory of its shards, w in braces for its number}
NAME=${NAME:?the name of the log}
CLASSES=${CLASSES:-tiny small medium long xlong}
EXTRA=${EXTRA:-}
SEGMENTS=${SEGMENTS//\{w\}/$WORKER}
OUT=${OUT//\{w\}/$WORKER}

[ -x "$PY" ] || { echo "$(hostname): no $PY (experiments/flan-nl/setup.sh)"; exit 1; }
[ -s "$TOK/tokenizer.json" ] || { echo "$(hostname): no tokenizer in $TOK (experiments/flan-nl/setup.sh)"; exit 1; }
[ -s "$CODE/translate.py" ] || { echo "$(hostname): no translate.py beside $0"; exit 1; }
[ -z "$(sudo lsof -t /dev/accel* /dev/vfio/* 2> /dev/null)" ] || { echo "$(hostname): the TPU is in use"; exit 1; }
gsutil -q stat "$CKPT/done.txt" || { echo "$(hostname): $CKPT has no done.txt"; exit 1; }
mkdir -p ~/logs "$SHM/segments"
local_segments=$SHM/segments/$NAME-$(basename "$SEGMENTS")
gsutil -q cp "$SEGMENTS" "$local_segments"

# A host of the v4-32 as a TPU of its own: without these variables JAX takes the slice from the TPU metadata and waits
# for the other hosts. MaxText runs with skip_jax_distributed_system, so nothing initializes a multi-process runtime.
cat > ~/run_$NAME.sh <<EOF
set -uo pipefail
export TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 TPU_PROCESS_BOUNDS=1,1,1 TPU_VISIBLE_CHIPS=0,1,2,3 TPU_PROCESS_PORT=8476
export TPU_PROCESS_ADDRESSES=localhost:8476 CLOUD_TPU_TASK_ID=0 ALLOW_MULTIPLE_LIBTPU_LOAD=1
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
export JAX_COMPILATION_CACHE_DIR=$SHM/jax_cache
timeout 600 $PY -c "import jax; assert (jax.device_count(), jax.process_count()) == (4, 1), (jax.device_count(), jax.process_count()); print('devices', jax.devices())" || { echo "preflight failed: not 4 chips of one process"; exit 2; }
echo "\$(date +%H:%M:%S) $(hostname): $SOURCE -> $TARGET, $SEGMENTS -> $OUT with $CKPT"
for class in $CLASSES; do
  $PY -u $CODE/translate.py --segments $local_segments --out $OUT/\$class --ckpt $CKPT/0/items \
    --tokenizer $TOK --model gemma3-27b --length-class \$class --source-lang $SOURCE --target-lang $TARGET \
    --work-dir $SHM $EXTRA || exit \$?
done
EOF
setsid nohup bash -c "bash ~/run_$NAME.sh > ~/logs/$NAME.log 2>&1; echo EXIT \$? >> ~/logs/$NAME.log; gsutil -q cp ~/logs/$NAME.log $ROOT/pilot/logs/$NAME-w$WORKER.log" > /dev/null 2>&1 < /dev/null &
echo "started $NAME on $(hostname): ~/logs/$NAME.log, shards in $OUT"
