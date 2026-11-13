#!/bin/bash
# The prefill check of 13 Nov (prefill_check.py) on a worker of mbert-mix (v4-32, each worker a TPU of its own 4
# chips): TranslateGemma 27B on MaxText 0.2.5 (FLAN-nl's venv, as part B) and on the fork yhavinga/maxtext @ 2249474
# (venv-fork of setup_generic.sh, as part A), Gemma 4 31B-it on the fork; the documents, prompts and checkpoints of
# parts A and B of the length trial. One stage per call; every stage runs detached from ssh with a fresh log
# ~/logs/pc_<stage>_$TAG.log ending in "EXIT <code>" (also in $O/logs/).
#   bash run_prefill_check.sh check       every worker: FLAN-nl's venv and TG tokenizer, the translator's venv and
#                                         MetricX, a free TPU, room on disk; READY (not detached)
#   bash run_prefill_check.sh setup       every worker (CPU): setup_generic.sh venv (venv-fork, tokenizers of Gemma 4
#                                         and TG 27B from Hugging Face); its ~/.cache/huggingface/token is there
#   bash run_prefill_check.sh plan        w0 (CPU): the documents of every test, split over 4 workers, to $O/plan/
#   bash run_prefill_check.sh translate_tg     every worker: the tests on MaxText 0.2.5 (T4a T1 T4b T2b T2)
#   bash run_prefill_check.sh translate_fork   every worker: the tests on the fork (T5 T3 T6); needs setup
#   bash run_prefill_check.sh collect     w0 (CPU): measures of every test, the units for MetricX split for 4 workers
#   bash run_prefill_check.sh metricx     every worker: MetricX-24-QE of its units (the translator's venv, as part B)
#   bash run_prefill_check.sh report      w0 (CPU): $O/prefill-check.md and .json
# Out of memory in a test is logged and marked ($O/out/<test>/w<k>/oom.txt), not fatal. The code is beside this
# script (a copy of $O/scripts/). Deletes nothing.
# Another check with the same stages (fork_fix_check.py, 14 Nov) sets, all passed to every stage: CHECK (the script of
# the tests), FORK_COMMIT (the fork's commit: venv-fork-<first 9> of setup_generic.sh), O and D (its bucket and local
# directories), PLAN_A and PLAN_B (the tests of part A, Gemma 4, and of part B, TranslateGemma), TESTS_TG and
# TESTS_FORK, REPORT (the report's name), LOG_PREFIX (the test logs for the report) and BEFORE (an earlier check's
# bucket directory, given to the report as --before). Unset, each is what the prefill check of 13 Nov used.
set -euo pipefail
STAGE=${1:?a stage: check, setup, plan, translate_tg, translate_fork, collect, metricx, report}
O=${O:-gs://t5-dutch-english-usc2/mbert/translator/prefill-check}
CHECK=${CHECK:-prefill_check}
DEFAULT_FORK_COMMIT=2249474b93b971d8248520e3a09b94b4ce626ce4   # setup_generic.sh's
FORK_COMMIT=${FORK_COMMIT:-$DEFAULT_FORK_COMMIT}
PLAN_A=${PLAN_A-T3 T6}
PLAN_B=${PLAN_B-T4a T1 T4b T2b T2 T5}
TESTS_TG=${TESTS_TG-T4a T1 T4b T2b T2}
TESTS_FORK=${TESTS_FORK-T5 T3 T6}
REPORT=${REPORT:-prefill-check}
LOG_PREFIX=${LOG_PREFIX:-pc_T}
BEFORE=${BEFORE:-}
OA=gs://t5-dutch-english-usc2/mbert/translator/bt-length       # part A of the length trial
OB=gs://t5-dutch-english-usc2/mbert/translator/sft/length      # part B
SHM=/dev/shm/translator
TGEN=/dev/shm/translate-generic
D=${D:-$SHM/prefill-check}
PY=$SHM/venv/bin/python                 # the translator's venv (CPU): pyarrow, transformers, tokenizers; MetricX
TPY=/dev/shm/flan_nl/venv/bin/python    # FLAN-nl's venv: MaxText 0.2.5 from PyPI (part B)
FV=$TGEN/venv-fork                      # the fork yhavinga/maxtext @ 2249474 (part A)
[ "$FORK_COMMIT" = "$DEFAULT_FORK_COMMIT" ] || FV=$TGEN/venv-fork-${FORK_COMMIT:0:9}
FPY=$FV/bin/python
TOK_TG=/dev/shm/flan_nl/tokenizer-27b   # part B's tokenizer, for both builds (the same prompts)
TOK_G4=$TGEN/tokenizer-gemma4
CKPT_TG=gs://t5-dutch-english-usc2/mbert/flan-nl/translategemma-27b-maxtext/0/items
CKPT_G4=gs://t5-dutch-english-usc2/mbert/translator/judge/gemma-4-31b-it-maxtext/0/items
C=$(cd "$(dirname "$0")" && pwd)
W=$(hostname | sed 's/.*-w-//')
TAG=${TAG:-manual}
mkdir -p ~/logs "$D"

detach() {  # detach NAME COMMAND...: the command in the background, log ~/logs/NAME.log ending in EXIT <code>
  local name=$1
  shift
  [ -e ~/logs/$name.log ] && { echo "w$W: ~/logs/$name.log exists; a new start takes a new TAG"; exit 1; }
  printf '%q ' "$@" > ~/job_$name.sh
  setsid nohup bash -c "bash ~/job_$name.sh > ~/logs/$name.log 2>&1; echo EXIT \$? >> ~/logs/$name.log; gsutil -q cp ~/logs/$name.log $O/logs/$name-w$W.log" \
    > /dev/null 2>&1 < /dev/null &
  echo "started $name on w$W"
}

tpu_free() {
  [ -z "$(sudo lsof -t /dev/accel* /dev/vfio/* 2> /dev/null)" ] || { echo "w$W: the TPU is in use"; exit 1; }
}

# A worker of the v4-32 as a TPU of its own 4 chips (run_bt_length_tg.sh, run_bt_length_v4.sh)
ONE_HOST="export TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 TPU_PROCESS_BOUNDS=1,1,1 TPU_VISIBLE_CHIPS=0,1,2,3 TPU_PROCESS_PORT=8476
export TPU_PROCESS_ADDRESSES=localhost:8476 CLOUD_TPU_TASK_ID=0 ALLOW_MULTIPLE_LIBTPU_LOAD=1
export XLA_FLAGS=--xla_dump_hlo_as_text=false TPU_LOG_DIR=$SHM/tpu_logs TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 HF_HUB_DISABLE_PROGRESS_BARS=1"

# run_test PY TEST TOKENIZER CKPT: one test on this worker's share, one translate.py process (or one per shard),
# each with its own log ~/logs/pc_<test>_$TAG[-p<k>].log (to $O/logs/); out of memory marked, not fatal
RUN_TEST='run_test() {
  local py=$1 test=$2 tok=$3 ckpt=$4 seg out n k log code
  seg=$D/plan/$test/segments-w$W.parquet
  out=$O/out/$test/w$W
  n=$($py -I $C/$CHECK.py processes --test $test --segments $seg)
  echo "== $test $(date -u +%T): $n process(es)"
  for k in $(seq 1 $n); do
    log=~/logs/pc_${test}_$TAG.log
    [ "$n" -gt 1 ] && log=~/logs/pc_${test}_$TAG-p$k.log
    $py -u $C/$CHECK.py translate --test $test --segments $seg --out $out --tokenizer $tok --ckpt $ckpt \
      --work-dir $SHM/work-pc $([ "$n" -gt 1 ] && echo --max-shards 1) > $log 2>&1 && code=0 || code=$?
    echo "EXIT $code" >> $log
    gsutil -q cp $log $O/logs/$(basename $log .log)-w$W.log || true
    grep -E "^(SHARD|DONE|MEM|engine|MAXTEXT)" $log | tail -n 5 | cut -c1-300
    if [ $code != 0 ]; then
      if grep -q -E "RESOURCE_EXHAUSTED|Ran out of memory|[Oo]ut of memory" $log; then
        grep -m1 -E "RESOURCE_EXHAUSTED|Ran out of memory|[Oo]ut of memory" $log | cut -c1-500 | gsutil -q cp - $out/oom.txt
        echo "$test: OUT OF MEMORY (marked)"
        return 0
      fi
      tail -n 30 $log
      return $code
    fi
  done
}'

translate_body() {  # translate_body NAME PY TESTS...: the job of one family of tests
  local name=$1 py=$2
  shift 2
  {
    echo "set -uo pipefail"
    echo "C=$C D=$D O=$O W=$W TAG=$TAG SHM=$SHM CHECK=$CHECK"
    echo "$ONE_HOST"
    echo "export JAX_COMPILATION_CACHE_DIR=$SHM/jax_cache_pc"
    echo "mkdir -p $D/plan"
    echo "gsutil -m -q rsync -r $O/plan $D/plan || exit 1"
    echo "timeout 600 $py -c \"import jax; assert (jax.device_count(), jax.process_count()) == (4, 1), (jax.device_count(), jax.process_count())\" || { echo 'preflight failed: not 4 chips of one process'; exit 2; }"
    echo "$RUN_TEST"
    for test in "$@"; do
      case " $PLAN_A " in *" $test "*) echo "run_test $py $test $TOK_G4 $CKPT_G4 || exit 1" ;;
        *) echo "run_test $py $test $TOK_TG $CKPT_TG || exit 1" ;;
      esac
    done
    echo "echo \"TRANSLATED w$W: $*\""
  } > ~/job_body_${name}.sh
  detach "$name" bash ~/job_body_${name}.sh
}

case $STAGE in
check)
  "$PY" -c "import pyarrow, transformers, tokenizers, yaml" || { echo "w$W: $PY lacks a package"; exit 1; }
  "$TPY" -c "import maxtext, yaml" || { echo "w$W: $TPY has no maxtext"; exit 1; }
  [ -s "$TOK_TG/tokenizer.json" ] || { echo "w$W: no tokenizer in $TOK_TG"; exit 1; }
  [ -d "$SHM/metricx" ] || { echo "w$W: no $SHM/metricx"; exit 1; }
  [ -s ~/.cache/huggingface/token ] || { echo "w$W: no ~/.cache/huggingface/token"; exit 1; }
  tpu_free
  shm=$(df -BG --output=avail /dev/shm | tail -1 | tr -dc 0-9)
  [ "$shm" -ge 40 ] || { echo "w$W: only $shm GB free in /dev/shm"; exit 1; }
  free=$(df -BG --output=avail "$HOME" | tail -1 | tr -dc 0-9)
  [ "$free" -ge 3 ] || { echo "w$W: only $free GB free on the boot disk"; exit 1; }
  echo "READY w$W: /dev/shm $shm GB free, boot disk $free GB free"
  ;;
setup)
  cat > ~/job_body_pc_setup_$TAG.sh <<EOF
set -euo pipefail
FORK_COMMIT=$FORK_COMMIT HF_HUB_OFFLINE=0 bash $C/setup_generic.sh venv
[ -s $FV/done.txt ] && [ -s $TOK_G4/tokenizer.json ]
echo "SETUP w$W: \$(cat $FV/done.txt | cut -c1-300)"
EOF
  detach "pc_setup_$TAG" bash ~/job_body_pc_setup_$TAG.sh
  ;;
plan)
  [ "$W" = 0 ] || { echo "plan runs on w0 only"; exit 1; }
  cat > ~/job_body_pc_plan_$TAG.sh <<EOF
set -euo pipefail
mkdir -p $D/ref/a/seg $D/ref/b/seg $D/plan
gsutil -m -q cp $OA/data/docs.parquet $D/ref/a/
gsutil -m -q cp "$OA/seg/heel/segments-w*.parquet" $D/ref/a/seg/
gsutil -m -q cp $OB/data/docs-en.parquet $D/ref/b/
gsutil -m -q cp "$OB/seg/heel/segments-w*.parquet" $D/ref/b/seg/
for t in $PLAN_B; do
  $PY -I -u $C/$CHECK.py plan --test \$t --docs $D/ref/b/docs-en.parquet --segments $D/ref/b/seg/segments-w*.parquet \
    --tokenizer $TOK_TG --workers 4 --out $D/plan
done
for t in $PLAN_A; do
  $PY -I -u $C/$CHECK.py plan --test \$t --docs $D/ref/a/docs.parquet --segments $D/ref/a/seg/segments-w*.parquet \
    --workers 4 --out $D/plan
done
gsutil -m -q rsync -r $D/plan $O/plan
echo "PLANNED"
EOF
  detach "pc_plan_$TAG" bash ~/job_body_pc_plan_$TAG.sh
  ;;
translate_tg)
  tpu_free
  translate_body "pc_translate_tg_$TAG" "$TPY" $TESTS_TG
  ;;
translate_fork)
  tpu_free
  [ -x "$FPY" ] && [ -s "$FV/done.txt" ] || { echo "w$W: no $FV (stage setup)"; exit 1; }
  translate_body "pc_translate_fork_$TAG" "$FPY" $TESTS_FORK
  ;;
collect)
  [ "$W" = 0 ] || { echo "collect runs on w0 only"; exit 1; }
  cat > ~/job_body_pc_collect_$TAG.sh <<EOF
set -euo pipefail
mkdir -p $D/out $D/collect $D/pending $D/plan
gsutil -m -q rsync -r $O/plan $D/plan
gsutil -m -q rsync -r $O/out $D/out
$FPY -I -u $C/$CHECK.py collect --plan $D/plan --translations $D/out --docs-a $D/ref/a/docs.parquet \
  --docs-b $D/ref/b/docs-en.parquet --out $D/collect
$FPY -I -u $C/bt_length.py pending --units $D/collect/units.parquet --workers 4 --out $D/pending
gsutil -m -q cp $D/collect/measured.parquet $D/collect/units.parquet $O/collect/
gsutil -m -q cp $D/pending/pending-w*.parquet $O/pending-$TAG/
echo "COLLECTED"
EOF
  detach "pc_collect_$TAG" bash ~/job_body_pc_collect_$TAG.sh
  ;;
metricx)
  tpu_free
  cat > ~/job_body_pc_metricx_$TAG.sh <<EOF
set -euo pipefail
$ONE_HOST
export JAX_COMPILATION_CACHE_DIR=$SHM/jax_cache_metricx
mkdir -p $D/pending $D/scores
gsutil -q cp $O/pending-$TAG/pending-w$W.parquet $D/pending/
n=\$($PY -I -c "import pyarrow.parquet as pq, sys; print(pq.read_metadata(sys.argv[1]).num_rows)" $D/pending/pending-w$W.parquet)
if [ "\$n" = 0 ]; then echo "SCORED w$W: no unit"; exit 0; fi
timeout 600 $PY -c "import jax; assert (jax.device_count(), jax.process_count()) == (4, 1), (jax.device_count(), jax.process_count())"
$PY -I -u $C/metricx_qe.py score --pairs $D/pending/pending-w$W.parquet --pair src:cand --model xxl --root $SHM/metricx \
  --out $D/scores/qe-$TAG-w$W.parquet
gsutil -q cp $D/scores/qe-$TAG-w$W.parquet $O/scores/
echo "SCORED w$W: \$n units"
EOF
  detach "pc_metricx_$TAG" bash ~/job_body_pc_metricx_$TAG.sh
  ;;
report)
  [ "$W" = 0 ] || { echo "report runs on w0 only"; exit 1; }
  cat > ~/job_body_pc_report_$TAG.sh <<EOF
set -euo pipefail
mkdir -p $D/scores $D/logs $D/report
for p in a b; do mkdir -p $D/ref/\$p; done
gsutil -m -q cp $OA/collect/measured.parquet $OA/collect/units.parquet "$OA/scores/qe-*.parquet" $D/ref/a/
gsutil -m -q cp $OB/collect/measured.parquet $OB/collect/units.parquet "$OB/scores/qe-*.parquet" $D/ref/b/
gsutil -m -q rsync -r $O/scores $D/scores
gsutil -m -q cp $O/collect/measured.parquet $O/collect/units.parquet $D/collect/
gsutil -m -q cp "$O/logs/$LOG_PREFIX*" $D/logs/
before=()
if [ -n "$BEFORE" ]; then
  mkdir -p $D/before
  gsutil -m -q cp $BEFORE/collect/measured.parquet $BEFORE/collect/units.parquet "$BEFORE/scores/qe-*.parquet" $D/before/
  before=(--before $D/before)
fi
$FPY -I -u $C/$CHECK.py report --plan $D/plan --measured $D/collect/measured.parquet \
  --units $D/collect/units.parquet --qe $D/scores/qe-*.parquet --ref-a $D/ref/a --ref-b $D/ref/b --logs $D/logs \
  \${before[@]+"\${before[@]}"} --out $D/report/$REPORT.md
gsutil -m -q cp $D/report/$REPORT.md $D/report/$REPORT.json $O/
echo "REPORT: $O/$REPORT.md"
EOF
  detach "pc_report_$TAG" bash ~/job_body_pc_report_$TAG.sh
  ;;
*)
  echo "unknown stage $STAGE"; exit 1 ;;
esac
