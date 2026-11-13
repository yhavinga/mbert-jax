#!/bin/bash
# The length trial's condition ctx (bt_length_ctx.py; rule in mbert-ops/translator-sft/bt-length-trial.md, "Deel A,
# conditie ctx") on a v4-32 (mbert-mix, us-central2-b), each worker a TPU of its own 4 chips (one process; as
# run_bt_length_v4.sh, whose setup and mxdl stages the chain runs first). Part A's documents, pieces and stukken are
# read from $O (gs://t5-dutch-english-usc2/mbert/translator/bt-length), everything of ctx goes to $OC = $O/ctx. One
# stage per call, from co1 (g4_ctx_run.sh); every stage runs detached from ssh with a fresh log
# ~/logs/ctx_<stage>_$TAG.log that ends in "EXIT <code>" (also in $OC/logs/).
#   bash run_bt_length_ctx.sh plan        w0 (CPU): the plan of 4 workers to $OC/plan/ (skipped when plan.json is there)
#   bash run_bt_length_ctx.sh smoke       every worker: two rounds of its first 3 stretches to $OC/smoke-$TAG/; SMOKE
#                                         with the rounds' lines, or EXIT 1
#   bash run_bt_length_ctx.sh translate   every worker: all rounds of its stretches to $OC/tr/w<k>/ (resumes after the
#                                         last round there); out of memory: again with half the slots, down to one
#   bash run_bt_length_ctx.sh collect     w0 (CPU): measures, copy, terms, questions; the ctx units MetricX has not
#                                         scored, split for 4 workers ($OC/pending/)
#   bash run_bt_length_ctx.sh metricx     every worker: MetricX-24-QE nl->en of its pending units
#                                         ($OC/scores/qe-ctx-$TAG-w<k>.parquet)
#   bash run_bt_length_ctx.sh extract     every worker: Gemma's answers to its questions ($OC/extract/; skipped when
#                                         there); out of memory: half the slots
#   bash run_bt_length_ctx.sh report      w0 (CPU): $OC/report/bt-length-ctx.md (.json), the blind material of bin
#                                         4096 ($OC/report/blind-4096/, the key apart); VERDICT
# The code is beside this script (a copy of $OC/scripts/ at the commit in $OC/scripts/COMMIT). TAG names a chain's jobs.
set -euo pipefail
STAGE=${1:?a stage: plan, smoke, translate, collect, metricx, extract, report}
SHM=${SHM:-/dev/shm/translate-generic}  # another root only for a dry run on CPU
C=$(cd "$(dirname "$0")" && pwd)
D=$SHM/bt-length-ctx
PY=$SHM/venv-fork/bin/python
O=${O:-gs://t5-dutch-english-usc2/mbert/translator/bt-length}
OC=${OC:-$O/ctx}
CKPT=${CKPT:-gs://t5-dutch-english-usc2/mbert/translator/judge/gemma-4-31b-it-maxtext/0/items}
TOK=$SHM/tokenizer-gemma4
W=$(hostname | sed 's/.*-w-//')
TAG=${TAG:-manual}
WORKERS=4
SMOKE_DOCS=${SMOKE_DOCS:-3}
mkdir -p ~/logs "$D"

detach() {  # detach NAME COMMAND...: the command in the background, log ~/logs/NAME.log ending in EXIT <code>
  local name=$1
  shift
  [ -e ~/logs/$name.log ] && { echo "w$W: ~/logs/$name.log exists; a new start takes a new TAG"; exit 1; }
  printf '%q ' "$@" > ~/job_$name.sh
  setsid nohup bash -c "bash ~/job_$name.sh > ~/logs/$name.log 2>&1; echo EXIT \$? >> ~/logs/$name.log; gsutil -q cp ~/logs/$name.log $OC/logs/$name-w$W.log" \
    > /dev/null 2>&1 < /dev/null &
  echo "started $name on w$W"
}

tpu_free() {
  [ -z "$(sudo lsof -t /dev/accel* /dev/vfio/* 2> /dev/null)" ] || { echo "w$W: the TPU is in use"; exit 1; }
}

ONE_HOST="export TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 TPU_PROCESS_BOUNDS=1,1,1 TPU_VISIBLE_CHIPS=0,1,2,3 TPU_PROCESS_PORT=8476
export TPU_PROCESS_ADDRESSES=localhost:8476 CLOUD_TPU_TASK_ID=0 ALLOW_MULTIPLE_LIBTPU_LOAD=1
export XLA_FLAGS=--xla_dump_hlo_as_text=false TPU_LOG_DIR=$SHM/tpu_logs TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 HF_HUB_DISABLE_PROGRESS_BARS=1
export JAX_COMPILATION_CACHE_DIR=$SHM/jax_cache"
PREFLIGHT="timeout 600 $PY -c \"import jax; assert (jax.device_count(), jax.process_count()) == (4, 1), (jax.device_count(), jax.process_count()); print('devices', jax.devices())\" || { echo 'preflight failed: not 4 chips of one process'; exit 2; }"

# on_tpu NAME SLOTS LOG COMMAND...: the command with --slots; out of memory: again with half the slots, down to one
# (the slots of gemma4-31b-ctx are an estimate from part A's d1k on a v4 chip); each attempt its own log
# (…-s<slots>.log after the first), copied to $OC/logs/; LAST_LOG is the log of the attempt that succeeded
ON_TPU='on_tpu() {
  local name=$1 slots=$2 log=$3 code try
  shift 3
  try=$log
  while true; do
    echo "== $name slots $slots $(date -u +%T)"
    "$@" --slots $slots > $try 2>&1 && code=0 || code=$?
    echo "EXIT $code" >> $try
    gsutil -q cp $try '"$OC"'/logs/$(basename $try .log)-w'"$W"'.log || true
    grep -E "^(ROUND|DONE|EXTRACT|MEM|engine)" $try | tail -n 4 | cut -c1-300
    [ $code = 0 ] && { LAST_LOG=$try; return 0; }
    if grep -q -E "RESOURCE_EXHAUSTED|Ran out of memory|[Oo]ut of memory" $try && [ $slots -gt 1 ]; then
      slots=$((slots / 2)); try=${log%.log}-s$slots.log
      echo "$name: out of memory, again with $slots slots"
      continue
    fi
    tail -n 30 $try
    return $code
  done
}'

case $STAGE in
plan)
  [ "$W" = 0 ] || { echo "plan runs on w0 only"; exit 1; }
  cat > ~/job_body_ctx_plan_$TAG.sh <<EOF
set -euo pipefail
mkdir -p $D/data $D/seg $D/plan
if gsutil -q stat $OC/plan/plan.json; then
  echo "plan there before"
else
  gsutil -m -q cp $O/data/docs.parquet $D/data/
  gsutil -m -q cp "$O/seg/stukken/segments-w*.parquet" $D/seg/
  $PY -u $C/bt_length_ctx.py plan --docs $D/data/docs.parquet --segments $D/seg/segments-w*.parquet --tokenizer $TOK \
    --workers $WORKERS --out $D/plan
  gsutil -m -q cp $D/plan/plan.json $D/plan/plan-w*.parquet $OC/plan/
fi
echo "PLAN: \$(gsutil cat $OC/plan/plan.json | $PY -c "import json, sys; p = json.load(sys.stdin); print(json.dumps({k: p[k] for k in ('stretches', 'pieces', 'plain', 'context_room')}), [w['rounds'] for w in p['workers']])")"
EOF
  detach "ctx_plan_$TAG" bash ~/job_body_ctx_plan_$TAG.sh
  ;;
smoke)
  tpu_free
  cat > ~/job_body_ctx_smoke_$TAG.sh <<EOF
set -uo pipefail
$ONE_HOST
mkdir -p $D/plan
gsutil -q cp $OC/plan/plan-w$W.parquet $D/plan/ || exit 1
$PREFLIGHT
$ON_TPU
log=~/logs/ctx_smoke_tr_$TAG.log
on_tpu smoke 20 \$log $PY -u $C/bt_length_ctx.py translate --plan $D/plan/plan-w$W.parquet --out $OC/smoke-$TAG/w$W \
  --tokenizer $TOK --ckpt $CKPT --max-docs $SMOKE_DOCS --max-rounds 2 || exit 1
log=\$LAST_LOG
n=\$(grep -c '^ROUND ' \$log)
[ "\$n" = 2 ] || { echo "SMOKE FAILED w$W: \$n ROUND lines, not 2"; exit 1; }
ctx=\$(grep '^ROUND {"round": 1,' \$log | $PY -c "import json, sys; print(json.loads(sys.stdin.read()[6:])['context_tokens_mean'])")
$PY -c "import sys; sys.exit(0 if float(sys.argv[1]) > 0 else 1)" "\$ctx" || { echo "SMOKE FAILED w$W: round 1 without context"; exit 1; }
echo "SMOKE w$W \$(grep '^ROUND ' \$log | tr '\n' ' ') \$(grep -m1 '^engine ready' \$log) \$(grep -m1 '^MEM ' \$log)"
EOF
  detach "ctx_smoke_$TAG" bash ~/job_body_ctx_smoke_$TAG.sh
  ;;
translate)
  tpu_free
  cat > ~/job_body_ctx_translate_$TAG.sh <<EOF
set -uo pipefail
$ONE_HOST
mkdir -p $D/plan
gsutil -q cp $OC/plan/plan-w$W.parquet $D/plan/ || exit 1
$PREFLIGHT
$ON_TPU
on_tpu translate 20 ~/logs/ctx_tr_$TAG.log $PY -u $C/bt_length_ctx.py translate --plan $D/plan/plan-w$W.parquet \
  --out $OC/tr/w$W --tokenizer $TOK --ckpt $CKPT || exit 1
echo "TRANSLATED w$W: \$(gsutil ls $OC/tr/w$W/ | grep -c 'shard-r') rounds in $OC/tr/w$W"
EOF
  detach "ctx_translate_$TAG" bash ~/job_body_ctx_translate_$TAG.sh
  ;;
collect)
  [ "$W" = 0 ] || { echo "collect runs on w0 only"; exit 1; }
  cat > ~/job_body_ctx_collect_$TAG.sh <<EOF
set -euo pipefail
rm -rf $D/collect $D/pending
mkdir -p $D/data $D/plan $D/stukken $D/tr $D/scores $D/a $D/collect $D/pending
gsutil -m -q cp $O/data/docs.parquet $D/data/
gsutil -m -q cp $O/collect/measured.parquet $D/a/
gsutil -m -q cp "$OC/plan/plan-w*.parquet" $D/plan/
gsutil -m -q rsync -r $O/stukken $D/stukken
gsutil -m -q rsync -r $OC/tr $D/tr
gsutil -m -q rsync -r $OC/scores $D/scores 2> /dev/null || true
$PY -u $C/bt_length_ctx.py collect --docs $D/data/docs.parquet --plan $D/plan/plan-w*.parquet --stukken $D/stukken \
  --ctx $D/tr --measured-a $D/a/measured.parquet --workers $WORKERS --out $D/collect
$PY -u $C/bt_length.py pending --units $D/collect/units.parquet --scored $D/scores/qe-ctx-*.parquet --workers $WORKERS \
  --out $D/pending
gsutil -m -q cp $D/collect/* $OC/collect/
gsutil -m -q cp $D/pending/pending-w*.parquet $OC/pending/
echo "COLLECTED: \$(cat $D/collect/collect.json)"
EOF
  detach "ctx_collect_$TAG" bash ~/job_body_ctx_collect_$TAG.sh
  ;;
metricx)
  tpu_free
  cat > ~/job_body_ctx_metricx_$TAG.sh <<EOF
set -euo pipefail
$ONE_HOST
export JAX_COMPILATION_CACHE_DIR=$SHM/jax_cache_metricx
mkdir -p $D/pending $D/scores
gsutil -q cp $OC/pending/pending-w$W.parquet $D/pending/
n=\$($PY -I -c "import pyarrow.parquet as pq, sys; print(pq.read_metadata(sys.argv[1]).num_rows)" $D/pending/pending-w$W.parquet)
if [ "\$n" = 0 ]; then echo "SCORED w$W: no unit"; exit 0; fi
# one file per chain and worker: a start again scores only what collect still finds pending
if gsutil -q stat $OC/scores/qe-ctx-$TAG-w$W.parquet; then echo "SCORED w$W: scored before"; exit 0; fi
$PREFLIGHT
$PY -u $C/metricx_qe.py score --pairs $D/pending/pending-w$W.parquet --pair src:cand --model xxl \
  --root $SHM/metricx --out $D/scores/qe-ctx-$TAG-w$W.parquet
gsutil -q cp $D/scores/qe-ctx-$TAG-w$W.parquet $OC/scores/
echo "SCORED w$W: \$n units"
EOF
  detach "ctx_metricx_$TAG" bash ~/job_body_ctx_metricx_$TAG.sh
  ;;
extract)
  tpu_free
  cat > ~/job_body_ctx_extract_$TAG.sh <<EOF
set -uo pipefail
$ONE_HOST
mkdir -p $D/collect $D/extract
if gsutil -q stat $OC/extract/answers-w$W.parquet; then echo "EXTRACTED w$W: answered before"; exit 0; fi
gsutil -q cp $OC/collect/queries-w$W.parquet $D/collect/ || exit 1
$PREFLIGHT
$ON_TPU
on_tpu extract 24 ~/logs/ctx_xq_$TAG.log $PY -u $C/bt_length_ctx.py extract --queries $D/collect/queries-w$W.parquet \
  --out $D/extract/answers-w$W.parquet --tokenizer $TOK --ckpt $CKPT || exit 1
gsutil -q cp $D/extract/answers-w$W.parquet $OC/extract/ || exit 1
echo "EXTRACTED w$W: \$(grep -h '^EXTRACT' ~/logs/ctx_xq_$TAG*.log | tail -n 1)"
EOF
  detach "ctx_extract_$TAG" bash ~/job_body_ctx_extract_$TAG.sh
  ;;
report)
  [ "$W" = 0 ] || { echo "report runs on w0 only"; exit 1; }
  cat > ~/job_body_ctx_report_$TAG.sh <<EOF
set -euo pipefail
rm -rf $D/report $D/logs $D/rcollect
mkdir -p $D/scores $D/logs $D/report $D/rcollect $D/extract $D/data $D/a
gsutil -m -q rsync -r $OC/collect $D/rcollect
gsutil -m -q cp "$O/scores/qe-a-*.parquet" "$OC/scores/qe-ctx-*.parquet" $D/scores/
gsutil -m -q cp "$OC/extract/answers-w*.parquet" $D/extract/
gsutil -m -q cp $O/collect/units.parquet $D/a/
gsutil -m -q cp $O/data/docs.parquet $D/data/
gsutil -m -q cp "$OC/logs/ctx_tr_*" "$OC/logs/ctx_xq_*" $D/logs/
n=\$(ls $D/extract/answers-w*.parquet | wc -l)
[ "\$n" = $WORKERS ] || { echo "answers of \$n workers, not $WORKERS"; exit 1; }
cat > $D/report/setup.json <<'JEOF'
{"model": "Gemma 4 31B-it (google/gemma-4-31B-it @ 842da37, MaxText 0.2.5-checkpoint van en-nl-judge), greedy, eigen chat-template met lege denkruimte; ctx: translate_models.yml gemma4-31b-ctx, klasse c512 (prefill 640 of 1.152, decodeerlimiet 640), prompt natural-v1-ctx (stuk 0: natural-v1); in stukken: de vertalingen van deel A (gemma4-31b-doc d512, natural-v1)", "context": "het eind van de Engelse vertaling van het vorige stuk (hele alinea's van achteren, zoveel als in een prompt van 1.152 tokens passen; de Nederlandse bron van het vorige stuk niet)", "consistentie": "termen van de Nederlandse bron (namen en kleine-letter-termen in >= 2 stukken), hun weergave per stuk door Gemma 4 31B-it (greedy) uit het Engels gehaald (bt_length_ctx.py extract), per term gelijk of niet", "MetricX": "google/metricx-24-hybrid-xxl-v2p6-bfloat16 @ 3d9c6f7, metricx_qe.py (JAX), QE nl->en per uitgelijnde eenheid; in stukken: de scores van deel A", "VM": "mbert-mix (v4-32, us-central2-b), elke worker als TPU van 4 chips (4 processen)", "code": "mbert-jax en-nl-length @ $(cat $C/COMMIT)"}
JEOF
$PY -u $C/bt_length_ctx.py report --collect $D/rcollect --units-a $D/a/units.parquet --qe $D/scores/qe-*.parquet \
  --answers $D/extract/answers-w*.parquet --logs $D/logs --setup $D/report/setup.json --out $D/report/bt-length-ctx.md \
  > $D/report/stdout.txt
$PY -u $C/bt_length_ctx.py blind --collect $D/rcollect --docs $D/data/docs.parquet --out $D/report/blind-4096
gsutil -m -q cp $D/report/bt-length-ctx.md $D/report/bt-length-ctx.json $D/report/setup.json \
  $D/report/blind-key-4096.json $OC/report/
gsutil -m -q rsync -r $D/report/blind-4096 $OC/report/blind-4096
grep '^VERDICT' $D/report/stdout.txt
echo "REPORT: $OC/report/bt-length-ctx.md"
EOF
  detach "ctx_report_$TAG" bash ~/job_body_ctx_report_$TAG.sh
  ;;
*)
  echo "unknown stage $STAGE"; exit 1 ;;
esac
