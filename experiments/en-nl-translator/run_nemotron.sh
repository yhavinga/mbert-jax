#!/bin/bash
# The Nemotron Chat v3 trial (nemotron-chat-v3.md) on mbert-mix (4 x v4-8, us-central2-b), one stage per call on a
# worker; the chain nemotron_trial_run.sh (co1, mbert-ops) starts the stages and waits for their EXIT lines. Every stage
# runs detached from ssh with a fresh log ~/logs/<name>.log that ends in "EXIT <code>" (also copied to $T/logs/). The fork
# venv, the Gemma 4 tokenizer and MetricX-24 come from run_g4arm.sh setup.
#   bash run_nemotron.sh data          w0 (CPU): byte ranges of data/chat.jsonl from HF at the pinned revision (once, to
#                                      $T/source/), nemotron_chat.py sample -> $T/data/
#   COND=ctx|noctx bash run_nemotron.sh gen   every worker (its TPU): nemotron_chat.py translate, G4-8-s140's LoRA,
#                                      the condition's classes (ctx: c2k; noctx: n128, then n512) -> $T/gen/<cond>/<cls>/w<k>/
#   bash run_nemotron.sh collect       w0 (CPU): nemotron_chat.py collect -> $T/collect/
#   bash run_nemotron.sh qe            every worker (its TPU): MetricX-24 hybrid XXL QE of its pieces -> $T/scores/qe-w<k>
#   bash run_nemotron.sh report        w0 (CPU): nemotron_chat.py report (no text) -> $T/report.md, $T/report.json
#   bash run_nemotron.sh blind         w0 (CPU): nemotron_chat.py blind -> $T/blind/package/ (judge-input.jsonl,
#                                      REVIEWER.md), the key $T/blind/key.jsonl
# The full run (Yeb 24 Nov ~14:55: the whole chat split; nemotron_full_run.sh, after mbert-ops/nemotron.full-go), under
# $F, every stage per shard (a shard = the lines that start in 128 MiB of chat.jsonl; shard r on worker r mod 4), a shard
# with its marker skipped, files read and written in the bucket from Python (nothing left on a worker):
#   bash run_nemotron.sh fwildchat     w1 (CPU): WildChat-1M's first prompts by sha256, straight from HF -> $F/wildchat/
#   bash run_nemotron.sh fprepare      every worker (CPU): its shards from HF, filters, pieces -> $F/data/shard-<r>/
#   COND=ctx|noctx bash run_nemotron.sh fgen    every worker (its TPU): translate --jobs over its shards, one engine
#                                      per class -> $F/gen/<cond>/<cls>/shard-<r>/
#   COND=... bash run_nemotron.sh fcollect      every worker (CPU): collect per shard -> $F/collect/shard-<r>/
#   bash run_nemotron.sh fqe           every worker (its TPU): MetricX-24 QE, its shards in groups of 8 -> $F/scores/
#   COND=... QE_MAX=... bash run_nemotron.sh ffinal   every worker (CPU): -> $F/out/chat-nl-<r>.jsonl, final-<r>.json
#   bash run_nemotron.sh fsummary      w0 (CPU): the counts of every stage -> $F/summary.json
# A stage whose output is in the bucket already prints that and ends in EXIT 0; nothing is removed or overwritten
# (translate goes on from the waves in the bucket). The code is beside this script. TAG names the jobs of one chain.
set -euo pipefail
STAGE=${1:?a stage: data gen collect qe report blind, or fwildchat fprepare fgen fcollect fqe ffinal fsummary}
R=gs://t5-dutch-english-usc2/mbert/translator
N=$R/nemotron
T=${T:-$N/trial}
F=${F:-$N/full}
QE_GROUP=8                                                      # shards per MetricX run in the full run
G=$R/sft/g4arm
ADAPTER=$G/G4-8-s140                                            # the translator (Yeb, 24 Nov ~11:15)
DECODE_CKPT=$R/judge/gemma-4-31b-it-maxtext/0/items            # the conversion every Gemma 4 decode uses
SHM=/dev/shm/translator
G_SHM=$SHM/g4arm
WX=$SHM/nemotron
GEN=/dev/shm/translate-generic
FORK_COMMIT=f4d8615f478b6c18de9df8f2dff1f57f775a2521             # every Gemma 4 decode above 1,024 tokens (14 Nov)
FPY=$GEN/venv-fork-${FORK_COMMIT:0:9}/bin/python
TOK_DEC=$GEN/tokenizer-gemma4
MX_ROOT=$SHM/metricx
DATASET=nvidia/Nemotron-SFT-Instruction-Following-Chat-v3
REVISION=be3b3e04ef605ac9d3f8f35b9d5a632f4a3a3402
FILE_BYTES=17042669218
RANGES=32                                                       # byte ranges of RANGE_BYTES, evenly over the file
RANGE_BYTES=8000000
CODE=$(cd "$(dirname "$0")" && pwd)
W=$(hostname | sed 's/.*-w-//')
TAG=${TAG:-manual}
mkdir -p "$WX" ~/logs
LOGDIR=$T/logs

detach() {  # detach NAME COMMAND...: the command in the background, log ~/logs/NAME.log ending in EXIT <code>
  local name=$1
  shift
  [ -e ~/logs/$name.log ] && { echo "w$W: ~/logs/$name.log exists; a new start takes a new TAG"; exit 1; }
  printf '%q ' "$@" > ~/job_$name.sh
  setsid nohup bash -c "bash ~/job_$name.sh > ~/logs/$name.log 2>&1; echo EXIT \$? >> ~/logs/$name.log; gsutil -q cp ~/logs/$name.log $LOGDIR/$name-w$W.log" \
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

need() {  # need <bucket path>...: the data a stage reads
  local p
  for p in "$@"; do gsutil -q stat "$p" || { echo "no $p"; exit 1; }; done
}

ready() {
  [ -x "$FPY" ] && [ -s "$TOK_DEC/tokenizer.json" ] || { echo "w$W: no fork venv or tokenizer (run_g4arm.sh setup)"; exit 1; }
}

shards_of_worker() {  # the full run's shards of this worker, as 5 digits (nemotron_chat.shards_of)
  JAX_PLATFORMS=cpu $FPY -I -c "import sys; sys.path.insert(0, '$CODE'); import nemotron_chat as nc; print(' '.join('%05d' % r for r in nc.shards_of($W)))"
}

# A worker of the v4-32 as a TPU of its own 4 chips (run_g4arm.sh)
ONE_HOST="export TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 TPU_PROCESS_BOUNDS=1,1,1 TPU_VISIBLE_CHIPS=0,1,2,3 TPU_PROCESS_PORT=8476
export TPU_PROCESS_ADDRESSES=localhost:8476 CLOUD_TPU_TASK_ID=0 ALLOW_MULTIPLE_LIBTPU_LOAD=1
export XLA_FLAGS=--xla_dump_hlo_as_text=false TPU_LOG_DIR=$SHM/tpu_logs TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 HF_HUB_DISABLE_PROGRESS_BARS=1"
PREFLIGHT="timeout 600 $FPY -c \"import jax; assert (jax.device_count(), jax.process_count()) == (4, 1), (jax.device_count(), jax.process_count()); print('devices', jax.devices())\" || { echo 'preflight failed: not 4 chips of one process'; exit 2; }"
CPU="export JAX_PLATFORMS=cpu OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1"

case $STAGE in
data)
  only 0
  ready
  NAME=nemo_data_$TAG
  gsutil -q stat "$T/data/sample.json" && there_before "$NAME" "NEMODATA_DONE $T/data/ (there before)"
  [ -s ~/.cache/huggingface/token ] || { echo "w$W: no ~/.cache/huggingface/token"; exit 1; }
  X=$WX/data-$TAG
  f=$(body "$NAME" <<EOF
set -euo pipefail
$CPU
mkdir -p $X/ranges $X/out
if gsutil -q stat $T/source/DONE; then
  gsutil -m -q cp '$T/source/ranges/part-*.jsonl' $X/ranges/
else
  url=https://huggingface.co/datasets/$DATASET/resolve/$REVISION/data/chat.jsonl
  for i in \$(seq 0 $((RANGES - 1))); do
    start=\$(( i * ($FILE_BYTES - $RANGE_BYTES) / ($RANGES - 1) ))
    # the token as a header from a file descriptor: never on a command line or in a log
    curl -fsSL --retry 5 -H @<(printf 'Authorization: Bearer %s\n' "\$(cat ~/.cache/huggingface/token)") \
      -r "\$start-\$((start + $RANGE_BYTES - 1))" "\$url" -o $X/ranges/part-\$start.jsonl
    [ "\$(stat -c %s $X/ranges/part-\$start.jsonl)" = $RANGE_BYTES ] || { echo "range \$start: not $RANGE_BYTES bytes"; exit 1; }
  done
  gsutil -m -q cp $X/ranges/part-*.jsonl $T/source/ranges/
  echo "$DATASET@$REVISION data/chat.jsonl: $RANGES ranges of $RANGE_BYTES bytes, \$(date -u +%FT%TZ)" | gsutil -q cp - $T/source/DONE
fi
echo "ranges: \$(ls $X/ranges | wc -l), \$(du -sh $X/ranges | cut -f1)"
nice -n 19 $FPY -I -u $CODE/nemotron_chat.py sample --ranges-dir $X/ranges --tokenizer $TOK_DEC --out $X/out
gsutil -m -q cp $X/out/*.parquet $X/out/sample.json $T/data/
echo "NEMODATA_DONE $T/data/"
EOF
)
  detach "$NAME" bash "$f"
  ;;
gen)
  COND=${COND:?COND=ctx or noctx}
  case $COND in ctx) CLASSES="c2k" ;; noctx) CLASSES="n128 n512" ;; *) echo "COND is ctx or noctx, not $COND"; exit 1 ;; esac
  ready
  NAME=nemo_gen_${COND}_$TAG
  all=1
  for c in $CLASSES; do gsutil -q stat "$T/gen/$COND/$c/w$W/DONE.json" || all=0; done
  [ $all = 1 ] && there_before "$NAME" "NEMOGEN_DONE w$W $COND (DONE there before)"
  tpu_free
  need "$T/data/pieces.parquet" "$ADAPTER/adapter.npz" "$ADAPTER/adapter.json" "${DECODE_CKPT%/0/items}/0/commit_success.txt"
  X=$WX/gen-$COND-$TAG
  LORA=$X/adapter
  mkdir -p "$LORA"
  # G4-8-s140's LoRA, rank 32 / alpha 16 from G4-0 (as run_g4measure2.sh generate); its copy stays (nothing removed)
  gsutil -q cp "$ADAPTER/adapter.npz" "$ADAPTER/adapter.json" "$LORA/"
  python3 -c "import json, sys; a = json.load(open(sys.argv[1])); assert (a['rank'], a['alpha'], a['scale']) == (32, 16.0, 0.5), a" "$LORA/adapter.json" \
    || { echo "$ADAPTER/adapter.json is not rank 32, alpha 16"; exit 1; }
  f=$(body "$NAME" <<EOF
set -uo pipefail
$ONE_HOST
export JAX_COMPILATION_CACHE_DIR=$G_SHM/jax_cache
mkdir -p $X $G_SHM/work
gsutil -q cp $T/data/pieces.parquet $X/pieces.parquet || exit 1
$PREFLIGHT
for c in $CLASSES; do
  echo "== $COND \$c w$W \$(date -u +%T) with the LoRA of $ADAPTER"
  $FPY -u $CODE/nemotron_chat.py translate --pieces $X/pieces.parquet --cond $COND --cls \$c --worker $W --out $T/gen/$COND/\$c/w$W --ckpt $DECODE_CKPT --tokenizer $TOK_DEC --lora $LORA --work-dir $G_SHM/work || exit \$?
  gsutil -q stat $T/gen/$COND/\$c/w$W/DONE.json || { echo "no $T/gen/$COND/\$c/w$W/DONE.json"; exit 1; }
done
echo "NEMOGEN_DONE w$W $COND"
EOF
)
  detach "$NAME" bash "$f"
  ;;
collect)
  only 0
  ready
  NAME=nemo_collect_$TAG
  gsutil -q stat "$T/collect/collect.json" && there_before "$NAME" "NEMOCOLLECT_DONE $T/collect/ (there before)"
  for c in ctx/c2k noctx/n128 noctx/n512; do for k in 0 1 2 3; do need "$T/gen/$c/w$k/DONE.json"; done; done
  X=$WX/collect-$TAG
  f=$(body "$NAME" <<EOF
set -euo pipefail
$CPU
mkdir -p $X/data $X/gen $X/out
gsutil -m -q cp $T/data/pieces.parquet $T/data/turns.parquet $T/data/conversations.parquet $X/data/
gsutil -m -q rsync -r $T/gen $X/gen
nice -n 19 $FPY -I -u $CODE/nemotron_chat.py collect --data $X/data --gen $X/gen --out $X/out
gsutil -m -q cp $X/out/* $T/collect/
echo "NEMOCOLLECT_DONE $T/collect/"
EOF
)
  detach "$NAME" bash "$f"
  ;;
qe)
  ready
  NAME=nemo_qe_$TAG
  gsutil -q stat "$T/scores/qe-w$W.parquet" && there_before "$NAME" "NEMOQE_DONE w$W (there before)"
  need "$T/collect/qe-in-w$W.parquet"
  [ -d "$MX_ROOT/metricx-24-hybrid-xxl-v2p6-bfloat16" ] || { echo "w$W: no MetricX-24 XXL in $MX_ROOT (run_g4arm.sh setup)"; exit 1; }
  tpu_free
  X=$WX/qe-$TAG
  # MetricX-24 hybrid XXL in QE mode (no reference), per piece (at most 512 English tokens: within its 1,536)
  f=$(body "$NAME" <<EOF
set -euo pipefail
$ONE_HOST
export JAX_COMPILATION_CACHE_DIR=$G_SHM/jax_cache_metricx
mkdir -p $X
gsutil -q cp $T/collect/qe-in-w$W.parquet $X/qe-in.parquet
$PREFLIGHT
$FPY -I -u $CODE/metricx_qe.py score --pairs $X/qe-in.parquet --pair source:hypothesis --model xxl --root $MX_ROOT --out $X/qe.parquet
gsutil -q cp $X/qe.parquet $T/scores/qe-w$W.parquet
echo "NEMOQE_DONE w$W"
EOF
)
  detach "$NAME" bash "$f"
  ;;
report)
  only 0
  ready
  NAME=nemo_report_$TAG
  gsutil -q stat "$T/report.md" && there_before "$NAME" "NEMOREPORT_DONE $T/report.md (there before)"
  for k in 0 1 2 3; do need "$T/scores/qe-w$k.parquet"; done
  X=$WX/report-$TAG
  f=$(body "$NAME" <<EOF
set -euo pipefail
$CPU
mkdir -p $X/data $X/collect $X/qe
gsutil -m -q cp $T/data/sample.json $T/data/pool-meta.parquet $X/data/
gsutil -m -q cp $T/collect/collect.json $T/collect/pieces-result-meta.parquet $X/collect/
gsutil -m -q cp '$T/scores/qe-w*.parquet' $X/qe/
nice -n 19 $FPY -I -u $CODE/nemotron_chat.py report --data $X/data --collect $X/collect --qe '$X/qe/qe-w*.parquet' --out $X/report.md
gsutil -q cp $X/report.md $T/report.md
gsutil -q cp $X/report.json $T/report.json
echo "NEMOREPORT_DONE $T/report.md"
EOF
)
  detach "$NAME" bash "$f"
  ;;
blind)
  only 0
  ready
  NAME=nemo_blind_$TAG
  gsutil -q stat "$T/blind/key.jsonl" && there_before "$NAME" "NEMOBLIND_DONE $T/blind/ (there before)"
  need "$T/collect/turns-nl.parquet"
  X=$WX/blind-$TAG
  f=$(body "$NAME" <<EOF
set -euo pipefail
$CPU
mkdir -p $X/data $X/collect $X/key
gsutil -m -q cp $T/data/turns.parquet $X/data/
gsutil -m -q cp $T/collect/turns-nl.parquet $X/collect/
nice -n 19 $FPY -I -u $CODE/nemotron_chat.py blind --data $X/data --collect $X/collect --out $X/package --key $X/key/key.jsonl
gsutil -m -q cp $X/package/judge-input.jsonl $X/package/REVIEWER.md $T/blind/package/
gsutil -q cp $X/key/blind.json $T/blind/blind.json
gsutil -q cp $X/key/key.jsonl $T/blind/key.jsonl
echo "NEMOBLIND_DONE $T/blind/"
EOF
)
  detach "$NAME" bash "$f"
  ;;
fwildchat)
  only 1
  ready
  NAME=nemo_fwildchat_$TAG
  gsutil -q stat "$F/wildchat/wildchat.json" && there_before "$NAME" "NEMOFWILDCHAT_DONE $F/wildchat/ (there before)"
  [ -s ~/.cache/huggingface/token ] || { echo "w$W: no ~/.cache/huggingface/token"; exit 1; }
  f=$(body "$NAME" <<EOF
set -euo pipefail
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=8 HF_HUB_DISABLE_PROGRESS_BARS=1
nice -n 19 $FPY -I -u $CODE/nemotron_chat.py wildchat --out $F/wildchat
echo "NEMOFWILDCHAT_DONE $F/wildchat/"
EOF
)
  detach "$NAME" bash "$f"
  ;;
fprepare)
  ready
  NAME=nemo_fprepare_$TAG
  need "$F/wildchat/wildchat.json"
  [ -s ~/.cache/huggingface/token ] || { echo "w$W: no ~/.cache/huggingface/token"; exit 1; }
  f=$(body "$NAME" <<EOF
set -euo pipefail
$CPU
nice -n 19 $FPY -I -u $CODE/nemotron_chat.py prepare --worker $W --wildchat $F/wildchat --tokenizer $TOK_DEC --out $F/data
echo "NEMOFPREPARE_DONE w$W"
EOF
)
  detach "$NAME" bash "$f"
  ;;
fgen)
  COND=${COND:?COND=ctx or noctx}
  case $COND in ctx) CLASSES="c2k" ;; noctx) CLASSES="n128 n512" ;; *) echo "COND is ctx or noctx, not $COND"; exit 1 ;; esac
  ready
  NAME=nemo_fgen_${COND}_$TAG
  tpu_free
  need "$ADAPTER/adapter.npz" "$ADAPTER/adapter.json" "${DECODE_CKPT%/0/items}/0/commit_success.txt"
  X=$WX/fgen-$COND-$TAG
  LORA=$X/adapter
  mkdir -p "$LORA"
  gsutil -q cp "$ADAPTER/adapter.npz" "$ADAPTER/adapter.json" "$LORA/"
  python3 -c "import json, sys; a = json.load(open(sys.argv[1])); assert (a['rank'], a['alpha'], a['scale']) == (32, 16.0, 0.5), a" "$LORA/adapter.json" \
    || { echo "$ADAPTER/adapter.json is not rank 32, alpha 16"; exit 1; }
  # one job line per shard of this worker; every shard has its prepare.json (fprepare ran)
  SHARDS=$(shards_of_worker)
  for r in $SHARDS; do gsutil -q stat "$F/data/shard-$r/prepare.json" || { echo "no $F/data/shard-$r/prepare.json"; exit 1; }; done
  for c in $CLASSES; do
    for r in $SHARDS; do echo "$F/data/shard-$r/pieces.parquet $F/gen/$COND/$c/shard-$r"; done > "$X/jobs-$c.txt"
  done
  f=$(body "$NAME" <<EOF
set -uo pipefail
$ONE_HOST
export JAX_COMPILATION_CACHE_DIR=$G_SHM/jax_cache
mkdir -p $G_SHM/work
$PREFLIGHT
for c in $CLASSES; do
  echo "== $COND \$c w$W \$(date -u +%T) \$(wc -l < $X/jobs-\$c.txt) shards, the LoRA of $ADAPTER"
  $FPY -u $CODE/nemotron_chat.py translate --jobs $X/jobs-\$c.txt --cond $COND --cls \$c --worker $W --ckpt $DECODE_CKPT --tokenizer $TOK_DEC --lora $LORA --work-dir $G_SHM/work || exit \$?
  for r in $SHARDS; do gsutil -q stat $F/gen/$COND/\$c/shard-\$r/DONE.json || { echo "no $F/gen/$COND/\$c/shard-\$r/DONE.json"; exit 1; }; done
done
echo "NEMOFGEN_DONE w$W $COND"
EOF
)
  detach "$NAME" bash "$f"
  ;;
fcollect)
  COND=${COND:?COND=ctx or noctx}
  ready
  NAME=nemo_fcollect_$TAG
  SHARDS=$(shards_of_worker)
  f=$(body "$NAME" <<EOF
set -euo pipefail
$CPU
for r in $SHARDS; do
  gsutil -q stat $F/collect/shard-\$r/collect.json && continue
  nice -n 19 $FPY -I -u $CODE/nemotron_chat.py collect --data $F/data/shard-\$r --gen $F/gen --leaf shard-\$r --conds $COND --out $F/collect/shard-\$r
done
echo "NEMOFCOLLECT_DONE w$W"
EOF
)
  detach "$NAME" bash "$f"
  ;;
fqe)
  ready
  NAME=nemo_fqe_$TAG
  [ -d "$MX_ROOT/metricx-24-hybrid-xxl-v2p6-bfloat16" ] || { echo "w$W: no MetricX-24 XXL in $MX_ROOT (run_g4arm.sh setup)"; exit 1; }
  tpu_free
  X=$WX/fqe-$TAG
  mkdir -p "$X"
  # the shards without scores, in groups of QE_GROUP: one MetricX run per group, its input and output files overwritten
  TODO=""
  for r in $(shards_of_worker); do gsutil -q stat "$F/scores/qe-$r.parquet" || TODO="$TODO $r"; done
  QGROUPS=$(echo $TODO | xargs -r -n $QE_GROUP | tr ' ' ',')
  f=$(body "$NAME" <<EOF
set -euo pipefail
$ONE_HOST
export JAX_COMPILATION_CACHE_DIR=$G_SHM/jax_cache_metricx
$PREFLIGHT
for g in $QGROUPS; do
  $FPY -I -u $CODE/nemotron_chat.py qe-join --collect $F/collect --shards \$g --worker $W --out $X/qe-in.parquet
  $FPY -I -u $CODE/metricx_qe.py score --pairs $X/qe-in.parquet --pair source:hypothesis --model xxl --root $MX_ROOT --out $X/qe.parquet
  $FPY -I -u $CODE/nemotron_chat.py qe-split --scores $X/qe.parquet --shards \$g --out $F/scores
done
echo "NEMOFQE_DONE w$W"
EOF
)
  detach "$NAME" bash "$f"
  ;;
ffinal)
  COND=${COND:?COND=ctx or noctx}
  QE_MAX=${QE_MAX:?QE_MAX=<number> or none}
  ready
  NAME=nemo_ffinal_$TAG
  f=$(body "$NAME" <<EOF
set -euo pipefail
$CPU
nice -n 19 $FPY -I -u $CODE/nemotron_chat.py final --worker $W --cond $COND --collect $F/collect --scores $F/scores --out $F/out --qe-max $QE_MAX
echo "NEMOFFINAL_DONE w$W"
EOF
)
  detach "$NAME" bash "$f"
  ;;
fsummary)
  only 0
  ready
  NAME=nemo_fsummary_$TAG
  f=$(body "$NAME" <<EOF
set -euo pipefail
$CPU
nice -n 19 $FPY -I -u $CODE/nemotron_chat.py summary --data $F/data --gen $F/gen --out $F/out --json $F/summary.json
echo "NEMOFSUMMARY_DONE $F/summary.json"
EOF
)
  detach "$NAME" bash "$f"
  ;;
*)
  echo "unknown stage $STAGE"
  exit 1
  ;;
esac
