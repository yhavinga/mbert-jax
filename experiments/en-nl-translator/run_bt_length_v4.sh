#!/bin/bash
# Part A of the length trial of 13 Nov (bt_length.py) on the on-demand v4-32 mbert-corpus (us-central2-b), instead of
# a spot v6e-8 (run_bt_length.sh, which stays the route for v6e): Gemma 4 31B-it (natural-v1) translates stretches of
# five length bins of real Dutch documents into English, whole (heel) and in pieces of whole paragraphs up to ~512
# tokens (stukken); MetricX-24-QE nl->en per aligned paragraph; the report and the English for part B. Each worker of
# the v4-32 is a TPU of its own 4 chips (one process; run_bt_length_tg.sh's ONE_HOST), so the segments are split over
# 4 workers instead of 2 halves. One stage per call, from co1 (g4_corpus_run.sh) on the workers it names; every stage
# runs detached from ssh with a fresh log ~/logs/lenv4_<stage>_$TAG.log that ends in "EXIT <code>" (also in $O/logs/).
#   bash run_bt_length_v4.sh setup        every worker: setup_generic.sh venv (venv-fork, tokenizers), the TPU free,
#                                         4 chips in one process, room in /dev/shm; READY
#   bash run_bt_length_v4.sh mxdl         every worker: MetricX-24 XXL from Hugging Face into $SHM/metricx
#   bash run_bt_length_v4.sh data         w0 (CPU): the segments of $O/data/docs.parquet for 4 workers, to $O/seg/
#   bash run_bt_length_v4.sh smoke        every worker: one shard of 32 of its pieces (class d512) to $O/smoke-$TAG/;
#                                         SMOKE with the shard's line, or EXIT 1
#   CLASSES="d512 d1k" bash run_bt_length_v4.sh translate
#                                         every worker: its stukken (all their classes) and the heel classes of CLASSES,
#                                         to $O/{heel,stukken}/w<k>/<class>/; a class out of memory is marked (oom.txt)
#   bash run_bt_length_v4.sh collect S    w0 (CPU): collect, the units not scored, split for 4 workers ($O/pending-S/)
#   bash run_bt_length_v4.sh metricx S    every worker: MetricX-24-QE nl->en of its pending units ($O/scores/qe-S-w<k>)
#   bash run_bt_length_v4.sh report       w0 (CPU): report A ($O/report/bt-length-a.md, .json), the English for part
#                                         B ($O/report/docs-en.parquet); VERDICT8192 pass|fail
# Translate skips every shard already in the bucket (translate.py), so a stage started again resumes.
# The repeat of 14 Nov on mbert-mix (heel d512 d1k d2k on the fork's fix of the decode cache, beside part A's stukken)
# sets, for every stage: FORK_COMMIT (the fork's commit: venv-fork-<first 9> of setup_generic.sh), O and D (its bucket
# and local directories; part A's data, segments and stukken copied there beforehand, so translate skips the stukken,
# and the scores of part A's stukken as $O/scores/qe-a-stukken.parquet), REPORT (the report's name), TITLE, BEFORE
# (part A's report .json in the bucket: report --before, and --relative), ENGLISH=0 (no English for part B) and VMNAME.
# Unset, each is what part A used.
# The code is beside this script (a copy of $O/scripts/, stage.sh). TAG names the jobs of one chain.
set -euo pipefail
STAGE=${1:?a stage: setup, mxdl, data, smoke, translate, collect, metricx, report}
SHM=${SHM:-/dev/shm/translate-generic}   # another root only for a dry run on a CPU host
C=$(cd "$(dirname "$0")" && pwd)
D=${D:-$SHM/bt-length}
FORK_COMMIT=${FORK_COMMIT:-}
V=$SHM/venv-fork
[ -n "$FORK_COMMIT" ] && V=$SHM/venv-fork-${FORK_COMMIT:0:9}
PY=$V/bin/python
O=${O:-gs://t5-dutch-english-usc2/mbert/translator/bt-length}
REPORT=${REPORT:-bt-length-a}
TITLE=${TITLE:-Deel A: Gemma 4 31B-it nl→en, heel tegen in stukken}
BEFORE=${BEFORE:-}
ENGLISH=${ENGLISH:-1}
VMNAME=${VMNAME:-mbert-corpus}
CKPT=${CKPT:-gs://t5-dutch-english-usc2/mbert/translator/judge/gemma-4-31b-it-maxtext/0/items}
TOK=$SHM/tokenizer-gemma4
TOK_TG=$SHM/tokenizer-tg27b
W=$(hostname | sed 's/.*-w-//')
TAG=${TAG:-manual}
WORKERS=4
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

# A worker of the v4-32 as a TPU of its own 4 chips (run_bt_length_tg.sh, run_arms.sh ONE_HOST)
ONE_HOST="export TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 TPU_PROCESS_BOUNDS=1,1,1 TPU_VISIBLE_CHIPS=0,1,2,3 TPU_PROCESS_PORT=8476
export TPU_PROCESS_ADDRESSES=localhost:8476 CLOUD_TPU_TASK_ID=0 ALLOW_MULTIPLE_LIBTPU_LOAD=1
export XLA_FLAGS=--xla_dump_hlo_as_text=false TPU_LOG_DIR=$SHM/tpu_logs TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 HF_HUB_DISABLE_PROGRESS_BARS=1"
PREFLIGHT="timeout 600 $PY -c \"import jax; assert (jax.device_count(), jax.process_count()) == (4, 1), (jax.device_count(), jax.process_count()); print('devices', jax.devices())\" || { echo 'preflight failed: not 4 chips of one process'; exit 2; }"

# The values translate_class reads, for the job bodies that define it
VARS="PY=$PY C=$C D=$D W=$W TOK=$TOK CKPT=$CKPT SHM=$SHM O=$O"
# translate_class COND CLASS SHARD OUT LOG [EXTRA...]: one class of this worker's segments. Out of memory: again with
# half the slots, down to one (the slots of gemma4-31b-doc were sized on a v6e chip, 31.24 GiB; a v4 chip has 30.75,
# and the decision rule counts a bin that could not be measured as failed); out of memory with one slot is marked in
# the bucket and not fatal (as run_bt_length.sh). Each attempt has its own log (…-s<slots>.log after the first).
TRANSLATE_CLASS='translate_class() {
  local cond=$1 cls=$2 shard=$3 out=$4 log=$5 code slots try
  shift 5
  if gsutil -q stat $out/oom.txt; then echo "$cond $cls: out of memory before ($out/oom.txt)"; return 0; fi
  case $cls in d512) slots=32 ;; d1k) slots=16 ;; d2k) slots=8 ;; d4k) slots=4 ;; d8k) slots=2 ;; *) slots=1 ;; esac
  try=$log
  while true; do
    echo "== $cond $cls slots $slots $(date -u +%T)"
    $PY -u $C/bt_length_translate.py --segments $D/seg/$cond/segments-w$W.parquet --out $out --tokenizer $TOK \
      --model gemma4-31b-doc --prompt natural-v1 --length-class $cls --source-lang nl --target-lang en --ckpt $CKPT \
      --work-dir $SHM/work-len --shard-size $shard --slots $slots "$@" > $try 2>&1 && code=0 || code=$?
    echo "EXIT $code" >> $try
    gsutil -q cp $try $O/logs/$(basename $try .log)-w$W.log || true
    grep -E "^(SHARD|DONE|MEM|engine)" $try | tail -n 4 | cut -c1-300
    [ $code = 0 ] && return 0
    if grep -q -E "RESOURCE_EXHAUSTED|Ran out of memory|[Oo]ut of memory" $try; then
      if [ $slots -gt 1 ]; then
        slots=$((slots / 2)); try=${log%.log}-s$slots.log
        echo "$cond $cls: out of memory, again with $slots slots"
        continue
      fi
      grep -m1 -E "RESOURCE_EXHAUSTED|Ran out of memory|[Oo]ut of memory" $try | cut -c1-500 | gsutil -q cp - $out/oom.txt
      echo "$cond $cls: OUT OF MEMORY with one slot, marked ($out/oom.txt)"
      return 0
    fi
    tail -n 30 $try
    return $code
  done
}'

shard_size() {  # cond class: as run_bt_length.sh (256 pieces; whole stretches twice the class's slots)
  [ "$1" = stukken ] && { echo 256; return; }
  case $2 in d512) echo 64 ;; d1k) echo 32 ;; d2k) echo 16 ;; d4k) echo 8 ;; d8k) echo 4 ;; d16k) echo 2 ;; *) echo 256 ;; esac
}

case $STAGE in
setup)
  tpu_free
  [ -s ~/.cache/huggingface/token ] || { echo "w$W: no ~/.cache/huggingface/token"; exit 1; }
  shm=$(df -BG --output=avail /dev/shm | tail -1 | tr -dc 0-9)
  [ "$shm" -ge 60 ] || { echo "w$W: only $shm GB free in /dev/shm"; exit 1; }
  free=$(df -BG --output=avail "$HOME" | tail -1 | tr -dc 0-9)
  [ "$free" -ge 3 ] || { echo "w$W: only $free GB free on the boot disk"; exit 1; }
  cat > ~/job_body_lenv4_setup_$TAG.sh <<EOF
set -euo pipefail
sudo -n loginctl enable-linger "\$(id -un)"
FORK_COMMIT=$FORK_COMMIT HF_HUB_OFFLINE=0 bash $C/setup_generic.sh venv
[ -z "$FORK_COMMIT" ] || grep -q "$FORK_COMMIT" $V/done.txt || { echo "$V is not at $FORK_COMMIT"; exit 1; }
$ONE_HOST
$PREFLIGHT
echo "READY w$W: /dev/shm \$(df -h /dev/shm | tail -1 | awk '{print \$4}') free"
EOF
  detach "lenv4_setup_$TAG" bash ~/job_body_lenv4_setup_$TAG.sh
  ;;
mxdl)
  cat > ~/job_body_lenv4_mxdl_$TAG.sh <<EOF
set -euo pipefail
HF_HUB_OFFLINE=0 HF_HUB_DISABLE_PROGRESS_BARS=1 $PY -c "import sys; sys.path.insert(0, '$C'); import metricx_qe as m; print('MXDL w$W', m.download('xxl', '$SHM/metricx'))"
EOF
  detach "lenv4_mxdl_$TAG" bash ~/job_body_lenv4_mxdl_$TAG.sh
  ;;
data)
  [ "$W" = 0 ] || { echo "data runs on w0 only"; exit 1; }
  cat > ~/job_body_lenv4_data_$TAG.sh <<EOF
set -euo pipefail
mkdir -p $D/data $D/seg
if gsutil -q stat $O/seg/segments.json; then
  echo "segments there before"
else
  gsutil -m -q cp $O/data/docs.parquet $O/data/docs.json $D/data/
  $PY -u $C/bt_length.py segments --docs $D/data/docs.parquet --tokenizer $TOK --model gemma4-31b-doc \
    --doc-model gemma4-31b-doc --prompt natural-v1 --source nl --target en --workers $WORKERS --out $D/seg
  gsutil -m -q rsync -r $D/seg $O/seg
fi
echo "DATA: \$(gsutil cat $O/seg/segments.json | tr -d '\n ')"
EOF
  detach "lenv4_data_$TAG" bash ~/job_body_lenv4_data_$TAG.sh
  ;;
smoke)
  tpu_free
  cat > ~/job_body_lenv4_smoke_$TAG.sh <<EOF
set -uo pipefail
$VARS
$ONE_HOST
export JAX_COMPILATION_CACHE_DIR=$SHM/jax_cache
mkdir -p $D/seg
gsutil -m -q rsync -r $O/seg $D/seg || exit 1
$PREFLIGHT
$TRANSLATE_CLASS
log=~/logs/lenv4_smoke_d512_$TAG.log
translate_class stukken d512 32 $O/smoke-$TAG/w$W/d512 \$log --max-shards 1 || exit 1
gsutil -q stat $O/smoke-$TAG/w$W/d512/oom.txt && { echo "SMOKE FAILED w$W: out of memory at d512"; exit 1; }
line=\$(grep -m1 '^SHARD ' \$log) || { echo "SMOKE FAILED w$W: no SHARD line"; exit 1; }
echo "SMOKE w$W \$line \$(grep -m1 '^engine ready' \$log) \$(grep -m1 '^MEM ' \$log)"
EOF
  detach "lenv4_smoke_$TAG" bash ~/job_body_lenv4_smoke_$TAG.sh
  ;;
translate)
  CLASSES=${CLASSES:?the heel classes, e.g. "d512 d1k d2k d4k d8k"}
  tpu_free
  {
    echo "set -uo pipefail"
    echo "$VARS"
    echo "$ONE_HOST"
    echo "export JAX_COMPILATION_CACHE_DIR=$SHM/jax_cache"
    echo "mkdir -p $D/seg"
    echo "gsutil -m -q rsync -r $O/seg $D/seg || exit 1"
    echo "$PREFLIGHT"
    echo "$TRANSLATE_CLASS"
    echo "for cls in \$(cat $D/seg/stukken/classes.txt); do"
    echo "  translate_class stukken \$cls 256 $O/stukken/w$W/\$cls ~/logs/len_stukken_\${cls}_$TAG.log || exit 1"
    echo "done"
    for cls in $CLASSES; do
      echo "if grep -qw $cls $D/seg/heel/classes.txt; then"
      echo "  translate_class heel $cls $(shard_size heel "$cls") $O/heel/w$W/$cls ~/logs/len_heel_${cls}_$TAG.log || exit 1"
      echo "fi"
    done
    echo "echo \"TRANSLATED w$W: $CLASSES\""
  } > ~/job_body_lenv4_translate_$TAG.sh
  detach "lenv4_translate_$TAG" bash ~/job_body_lenv4_translate_$TAG.sh
  ;;
collect)
  S=${2:?the scoring stage, a or b}
  [ "$W" = 0 ] || { echo "collect runs on w0 only"; exit 1; }
  cat > ~/job_body_lenv4_collect_$TAG.sh <<EOF
set -euo pipefail
mkdir -p $D/heel $D/stukken $D/scores $D/data $D/seg $D/pending-$S
gsutil -m -q cp $O/data/docs.parquet $O/data/docs.json $D/data/
gsutil -m -q rsync -r $O/seg $D/seg
gsutil -m -q rsync -r $O/heel $D/heel
gsutil -m -q rsync -r $O/stukken $D/stukken
gsutil -m -q rsync -r $O/scores $D/scores 2> /dev/null || true
$PY -u $C/bt_length.py collect --docs $D/data/docs.parquet --heel $D/heel --stukken $D/stukken \
  --pieces $D/seg/stukken/segments-w*.parquet --source nl --target en --out $D/collect
$PY -u $C/bt_length.py pending --units $D/collect/units.parquet --scored $D/scores/qe-*.parquet --workers $WORKERS \
  --out $D/pending-$S
gsutil -m -q cp $D/collect/measured.parquet $D/collect/units.parquet $O/collect/
gsutil -m -q cp $D/pending-$S/pending-w*.parquet $O/pending-$S/
echo "COLLECTED $S"
EOF
  detach "lenv4_collect_$TAG" bash ~/job_body_lenv4_collect_$TAG.sh
  ;;
metricx)
  S=${2:?the scoring stage, a or b}
  tpu_free
  cat > ~/job_body_lenv4_metricx_$TAG.sh <<EOF
set -euo pipefail
$ONE_HOST
export JAX_COMPILATION_CACHE_DIR=$SHM/jax_cache_metricx
mkdir -p $D/pending-$S $D/scores
gsutil -q cp $O/pending-$S/pending-w$W.parquet $D/pending-$S/
n=\$($PY -I -c "import pyarrow.parquet as pq, sys; print(pq.read_metadata(sys.argv[1]).num_rows)" $D/pending-$S/pending-w$W.parquet)
if [ "\$n" = 0 ]; then echo "SCORED w$W: no unit"; exit 0; fi
if gsutil -q stat $O/scores/qe-$S-w$W.parquet; then echo "SCORED w$W: qe-$S scored before"; exit 0; fi
$PREFLIGHT
$PY -u $C/metricx_qe.py score --pairs $D/pending-$S/pending-w$W.parquet --pair src:cand --model xxl \
  --root $SHM/metricx --out $D/scores/qe-$S-w$W.parquet
gsutil -q cp $D/scores/qe-$S-w$W.parquet $O/scores/
echo "SCORED w$W: \$n units"
EOF
  detach "lenv4_metricx_$TAG" bash ~/job_body_lenv4_metricx_$TAG.sh
  ;;
report)
  [ "$W" = 0 ] || { echo "report runs on w0 only"; exit 1; }
  cat > ~/job_body_lenv4_report_$TAG.sh <<EOF
set -euo pipefail
mkdir -p $D/scores $D/logs $D/report $D/collect $D/data
gsutil -m -q rsync -r $O/scores $D/scores
gsutil -m -q cp $O/collect/measured.parquet $O/collect/units.parquet $D/collect/
gsutil -m -q cp $O/data/docs.parquet $O/data/docs.json $D/data/
gsutil -m -q cp "$O/logs/len_heel_*" "$O/logs/len_stukken_*" $D/logs/
if [ -n "$BEFORE" ]; then stukken="deel A (13 nov, fork 2249474, alle stukken in klasse d512)"; else stukken="dezelfde, alle stukken in klasse d512"; fi
$PY $C/bt_length.py setup --out $D/report/setup.json \
  --item model "Gemma 4 31B-it (google/gemma-4-31B-it @ 842da37, MaxText 0.2.5-checkpoint van en-nl-judge), prompt natural-v1, greedy, eigen chat-template met lege denkruimte; heel: translate_models.yml gemma4-31b-doc (één klasse per bak, decodeerlimiet 1,25 x de bron); in stukken: \$stukken" \
  --item MaxText "fork yhavinga/maxtext; pakketten: \$(cat $V/done.txt)" \
  --item MetricX "google/metricx-24-hybrid-xxl-v2p6-bfloat16 @ 3d9c6f7, metricx_qe.py (JAX), QE nl->en per uitgelijnde eenheid (bron: de Nederlandse alinea's, kandidaat: het Engels), invoer afgekapt op 1.536 tokens" \
  --item VM "$VMNAME (v4-32 on-demand, us-central2-b), elke worker als TPU van 4 chips (4 processen)"
extra=()
if [ -n "$BEFORE" ]; then
  gsutil -q cp $BEFORE $D/report/before.json
  extra=(--relative --before $D/report/before.json)
fi
$PY -u $C/bt_length.py report --measured $D/collect/measured.parquet --units $D/collect/units.parquet \
  --qe $D/scores/qe-*.parquet --logs $D/logs --setup $D/report/setup.json --source nl --target en \
  \${extra[@]+"\${extra[@]}"} --title "$TITLE" --out $D/report/$REPORT.md > /dev/null
gsutil -m -q cp $D/report/$REPORT.md $D/report/$REPORT.json $D/report/setup.json $O/report/
if [ "$ENGLISH" = 1 ]; then
  $PY -u $C/bt_length.py english --docs $D/data/docs.parquet --measured $D/collect/measured.parquet \
    --tokenizer $TOK_TG --out $D/report/docs-en.parquet
  gsutil -q cp $D/report/docs-en.parquet $O/report/
fi
v=\$($PY -I -c "import json, sys; v = json.load(open(sys.argv[1]))['verdict'].get('8192', {}); print('pass' if v.get('pass') else 'fail')" $D/report/$REPORT.json)
echo "VERDICT8192 \$v"
echo "REPORT: $O/report/$REPORT.md"
EOF
  detach "lenv4_report_$TAG" bash ~/job_body_lenv4_report_$TAG.sh
  ;;
*)
  echo "unknown stage $STAGE"; exit 1 ;;
esac
