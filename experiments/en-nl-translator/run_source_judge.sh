#!/bin/bash
# On every worker of mbert-corpus: the source check of Gemma 4 31B-it (experiments/en-nl-judge/judge.py score --task
# source, copied beside this script) on this worker's shards, detached from ssh, each worker a TPU of its own 4 chips
# (as run_translate.sh). The judge's environment comes from experiments/en-nl-judge/setup.sh (/dev/shm/judge/venv and
# tokenizer); the MaxText checkpoint is the one setup.sh convert wrote to the bucket of us-central2.
#   SHARDS=$R/mainset/judge-in/w{w} OUT=$R/mainset/source-scores/w{w} NAME=translator_source_1 bash run_source_judge.sh
# {w} in SHARDS and OUT is this worker's number. SHARDS holds shard-NNNNN.parquet with id, nl and en (en empty; the
# source task reads only nl). Log: ~/logs/$NAME.log, a fresh file per start that ends in "EXIT <code>"; a copy goes to
# $ROOT/mainset/logs/. A score shard lands in OUT when it is complete; a second start skips the shards there.
set -euo pipefail
ROOT=gs://t5-dutch-english-usc2/mbert/translator
CKPT=$ROOT/judge/gemma-4-31b-it-maxtext
SHM=/dev/shm/judge
PY=$SHM/venv/bin/python
CODE=$(cd "$(dirname "$0")" && pwd)
WORKER=$(hostname | sed 's/.*-w-//')
SHARDS=${SHARDS:?the bucket directory of the input shards of a worker, w in braces for its number}
OUT=${OUT:?the bucket directory of the score shards, w in braces for its number}
NAME=${NAME:?the name of the log, new for every start}
TOKEN_BUDGET=${TOKEN_BUDGET:-32768}
SHARDS=${SHARDS//\{w\}/$WORKER}
OUT=${OUT//\{w\}/$WORKER}

[ -x "$PY" ] || { echo "$(hostname): no $PY (experiments/en-nl-judge/setup.sh)"; exit 1; }
[ -s "$SHM/tokenizer/tokenizer.json" ] || { echo "$(hostname): no $SHM/tokenizer (experiments/en-nl-judge/setup.sh)"; exit 1; }
[ -s "$CODE/judge.py" ] || { echo "$(hostname): no judge.py beside $0"; exit 1; }
[ ! -e ~/logs/$NAME.log ] || { echo "$(hostname): ~/logs/$NAME.log exists; a new start takes a new NAME"; exit 1; }
[ -z "$(sudo lsof -t /dev/accel* /dev/vfio/* 2> /dev/null)" ] || { echo "$(hostname): the TPU is in use"; exit 1; }
gsutil -q stat "$CKPT/done.txt" || { echo "$(hostname): $CKPT has no done.txt (setup.sh convert)"; exit 1; }
LOCAL=$SHM/in/$NAME
mkdir -p ~/logs "$LOCAL"
gsutil -m -q cp "$SHARDS/shard-*.parquet" "$LOCAL/"
ITEMS=$CKPT/0/items
[ -d "$SHM/ckpt/0/items" ] && ITEMS=$SHM/ckpt/0/items  # the worker that converted it reads its local copy

# A host of the v4-32 as a TPU of its own: without these variables JAX takes the slice from the TPU metadata and waits
# for the other hosts. judge.py runs MaxText with skip_jax_distributed_system. XLA_FLAGS is set because MaxText 0.2.5
# dumps every program to /tmp/xla_dump when it is empty; libtpu's logs and the compilation cache stay off the boot disk.
cat > ~/run_$NAME.sh <<EOF
set -uo pipefail
export TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 TPU_PROCESS_BOUNDS=1,1,1 TPU_VISIBLE_CHIPS=0,1,2,3 TPU_PROCESS_PORT=8476
export TPU_PROCESS_ADDRESSES=localhost:8476 CLOUD_TPU_TASK_ID=0 ALLOW_MULTIPLE_LIBTPU_LOAD=1
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 XLA_FLAGS=--xla_dump_hlo_as_text=false
export TPU_LOG_DIR=$SHM/tpu_logs JUDGE_JAX_CACHE=$SHM/jax_cache
timeout 600 $PY -c "import jax; assert (jax.device_count(), jax.process_count()) == (4, 1), (jax.device_count(), jax.process_count()); print('devices', jax.devices())" || { echo "preflight failed: not 4 chips of one process"; exit 2; }
echo "\$(date +%H:%M:%S) $(hostname): $SHARDS -> $OUT with \$(ls $LOCAL | wc -l) shards, parameters $ITEMS"
$PY -u $CODE/judge.py score --ckpt $ITEMS --tokenizer $SHM/tokenizer --fsdp 4 --token-budget $TOKEN_BUDGET \
  --task source --pairs $LOCAL --out $OUT
EOF
setsid nohup bash -c "bash ~/run_$NAME.sh > ~/logs/$NAME.log 2>&1; echo EXIT \$? >> ~/logs/$NAME.log; gsutil -q cp ~/logs/$NAME.log $ROOT/mainset/logs/$NAME-w$WORKER.log" > /dev/null 2>&1 < /dev/null &
echo "started $NAME on $(hostname): ~/logs/$NAME.log, scores in $OUT"
