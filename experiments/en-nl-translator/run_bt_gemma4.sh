#!/bin/bash
# On a spot v6e-8 in us-east1-d or europe-west4-a: the new back-translation with Gemma 4 31B-it (bt_gemma4.py).
#   bash run_bt_gemma4.sh all
# 1. setup: the fork's venv and Gemma 4's tokenizer, the five datasets from Hugging Face (pinned revisions), the full
#    plan; its hash goes to $O/full/plan-hash.txt, and a VM whose plan differs stops (every VM of both regions makes the
#    same plan from the same data)
# 2. probe, only in the region whose bucket has $O/probe/run-here: both halves translate the probe plan (three prompts),
#    select applies the rule, MetricX-QE of the probe for the report
# 3. full, once $O/full/prompt.txt is there (put there after the probe's selection): both halves take units of the full
#    plan, us-east1 forwards and europe-west4 backwards; the publisher puts this region's shards on Hugging Face
# 4. final, once every unit is on Hugging Face: the report (bt_gemma4.py final, under a lease, once a region)
# A step done is in the bucket or on Hugging Face and is skipped there: a preempted VM loses at most a unit a half.
# The v6e-8 runs as two TPUs of 4 chips (Gemma 4 31B allows at most 4-way tensor parallelism).
set -euo pipefail
# systemd-logind (RemoveIPC=yes) deletes a user's files in /dev/shm when the user's last session ends (9 Nov)
sudo -n loginctl enable-linger "$(id -un)"
SHM=/dev/shm/translate-generic
C=$SHM/code4
D=$SHM/bt4
PY=$SHM/venv-fork/bin/python
TOK=$SHM/tokenizer-gemma4
ZONE=$(curl -fs -H Metadata-Flavor:Google http://metadata.google.internal/computeMetadata/v1/instance/zone | awk -F/ '{print $NF}')
REGION=${ZONE%-*}
case $REGION in
  us-east1) BUCKET=gs://t5-dutch-english-use1; DIRECTION=forward ;;
  europe-west4) BUCKET=gs://t5-dutch-english; DIRECTION=backward ;;
  *) echo "no bucket for zone '$ZONE'"; exit 1 ;;
esac
O=$BUCKET/mbert/translator/bt-gemma4
REPO=${REPO:-yhavinga/nl-backtranslated-en-gemma4}
RUN_ID=$(date +%s)
# MaxText dumps HLO to /tmp/xla_dump whenever XLA_FLAGS is empty (run_sft.sh)
export XLA_FLAGS=--xla_dump_hlo_as_text=false TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 HF_HUB_DISABLE_PROGRESS_BARS=1
mkdir -p ~/logs "$D"
log() { echo "== $* $(date -u +%T)"; }

half() {  # as run_bt_natural.sh: the 4 chips behind the first (0) or second (1) PCI root as a TPU of their own
  local chips port
  chips=$(for g in /dev/vfio/[0-9]*; do n=${g##*/}; dev=$(ls "/sys/kernel/iommu_groups/$n/devices"); echo "${dev%:*} $n"; done \
    | LC_ALL=C sort -k1,1 -k2,2n | awk -v h="$1" '$1 != root {k++; root = $1} k == h + 1 {printf "%s%s", sep, $2; sep = ","}')
  [ "$(echo "$chips" | tr , '\n' | grep -c .)" = 4 ] || { echo "half $1: vfio groups '$chips', not 4 behind one PCI root"; exit 2; }
  echo "half $1: TPU_VISIBLE_CHIPS=$chips"
  if [ "$1" = 0 ]; then port=8476; else port=8477; fi
  export H=$1 TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 TPU_PROCESS_BOUNDS=1,1,1 TPU_VISIBLE_CHIPS=$chips TPU_PROCESS_PORT=$port
  export TPU_PROCESS_ADDRESSES=localhost:$port CLOUD_TPU_TASK_ID=0 ALLOW_MULTIPLE_LIBTPU_LOAD=1
}

preflight() {
  timeout 600 "$PY" -c "import jax; assert (jax.device_count(), jax.process_count()) == (4, 1), (jax.device_count(), jax.process_count()); print('devices', jax.devices())" || { echo "preflight failed: not 4 chips of one process"; exit 2; }
}

job() {  # half log command...: the command in the environment of a half, its log ending in an EXIT line and in the bucket
  local h=$1 log=$2 code pid
  shift 2
  # Detached and then waited for: a subshell whose status a && or || tests runs with set -e ignored
  ( half "$h"; preflight; "$@" ) > ~/logs/$log.log 2>&1 &
  pid=$!
  wait $pid && code=0 || code=$?
  echo "EXIT $code" >> ~/logs/$log.log
  gsutil -q cp ~/logs/$log.log "$O/logs/$log-$(hostname)-$RUN_ID.log" || true
  return $code
}

classes() {  # plan: the plan's classes in this region's direction
  "$PY" -c "
import json, sys
seen = []
for u in json.load(open(sys.argv[1]))['units']:
    if u['class'] not in seen:
        seen.append(u['class'])
print(' '.join(seen if sys.argv[2] == 'forward' else seen[::-1]))" "$1/units.json" "$DIRECTION"
}

work_loop() {  # plan root prompt: this half takes units of each class until none is left in the region
  local plan=$1 root=$2 prompt=$3 class st
  while true; do
    for class in $(classes "$plan"); do
      "$PY" -u "$C/bt_gemma4.py" work --plan "$plan" --root "$root" --length-class "$class" ${prompt:+--prompt "$prompt"} \
        --direction "$DIRECTION" --owner "$(hostname)/h$H" --run "$RUN_ID" --region "$REGION" --tokenizer "$TOK" \
        --work-dir "$SHM/work4-h$H"
    done
    st=$("$PY" -u "$C/bt_gemma4.py" status --plan "$plan" --root "$root")
    echo "$st"
    if [ "$(echo "$st" | tail -n 1)" = COMPLETE ]; then break; fi
    sleep 60
  done
}

probe_qe() {  # MetricX-24-QE of the probe in both directions (for the report only), then the selection again with it
  local pairs=() k
  for k in tg v1 fmt fmtlen; do pairs+=(--pair "nl:en_$k" --pair "en_$k:nl"); done
  JAX_COMPILATION_CACHE_DIR=$SHM/jax_cache_metricx "$PY" -u "$C/metricx_qe.py" score --pairs "$D/probe/report/compare.parquet" \
    "${pairs[@]}" --model xxl --root "$SHM/metricx" --empty-candidates --out "$D/probe/report/qe.parquet"
  "$PY" -u "$C/bt_gemma4.py" select --plan "$D/probe" --shards "$D/probe/shards" --qe "$D/probe/report/qe.parquet" \
    --out "$D/probe/report"
}

step=${1:-}
case $step in
all)
  log "setup $(hostname) $REGION $DIRECTION"
  if [ ! -s "$SHM/venv-fork/done.txt" ] || [ ! -s "$TOK/tokenizer.json" ]; then
    HF_HUB_OFFLINE=0 bash "$C/setup_generic.sh" venv
  fi
  if [ ! -s "$D/hf/done.txt" ]; then
    HF_HUB_OFFLINE=0 "$PY" -u "$C/bt_natural.py" fetch --out "$D/hf"
    date -u +%FT%TZ > "$D/hf/done.txt"
  fi
  if [ ! -s "$D/plan/units.json" ]; then
    "$PY" -u "$C/bt_gemma4.py" plan --data "$D/hf" --tokenizer "$TOK" --out "$D/plan"
  fi
  hash=$("$PY" -c "import json, sys; print(json.load(open(sys.argv[1]))['hash'])" "$D/plan/units.json")
  if gsutil -q stat "$O/full/plan-hash.txt"; then
    there=$(gsutil cat "$O/full/plan-hash.txt")
    [ "$there" = "$hash" ] || { echo "this VM's plan $hash differs from $O/full/plan-hash.txt ($there)"; exit 1; }
  else
    echo "$hash" | gsutil -q cp - "$O/full/plan-hash.txt"
  fi
  log "plan $hash"
  if gsutil -q stat "$O/probe/run-here"; then
    if ! gsutil -q stat "$O/probe/report/selection.json"; then
      log "probe"
      gsutil -q cp "$O/scripts/probe-ids.json" "$D/probe-ids.json"
      [ -s "$D/probe/units.json" ] || "$PY" -u "$C/bt_gemma4.py" probeplan --data "$D/hf" --pilot "$D/probe-ids.json" \
        --tokenizer "$TOK" --out "$D/probe"
      job 0 bt4_probe_0 work_loop "$D/probe" "$O/probe" "" & a=$!
      job 1 bt4_probe_1 work_loop "$D/probe" "$O/probe" "" & b=$!
      wait $a && ca=0 || ca=$?
      wait $b && cb=0 || cb=$?
      if [ $ca != 0 ] || [ $cb != 0 ]; then echo "probe failed: half 0 $ca, half 1 $cb"; exit 1; fi
    fi
    # The selection and MetricX once, on the VM that creates the lock first (a generation precondition, atomic)
    if ! gsutil -q stat "$O/probe/report/qe.parquet" \
        && hostname | gsutil -q -h "x-goog-if-generation-match:0" cp - "$O/probe/select.lock" 2> /dev/null; then
      log "selection on this VM"
      ( HF_HUB_OFFLINE=0 "$PY" -c "import sys; sys.path.insert(0, '$C'); import metricx_qe as m; print(m.download('xxl', '$SHM/metricx'))" ) \
        > ~/logs/bt4_metricx_dl.log 2>&1 &
      dl=$!
      mkdir -p "$D/probe/shards" "$D/probe/report"
      gsutil -m -q rsync -r "$O/probe/shards" "$D/probe/shards"
      "$PY" -u "$C/bt_gemma4.py" select --plan "$D/probe" --shards "$D/probe/shards" --out "$D/probe/report"
      gsutil -m -q cp "$D/probe/report/*" "$O/probe/report/"
      log "selection in $O/probe/report/selection.md; MetricX-QE of the probe"
      wait $dl || true
      HF_HUB_OFFLINE=0 "$PY" -c "import sys; sys.path.insert(0, '$C'); import metricx_qe as m; print(m.download('xxl', '$SHM/metricx'))" \
        >> ~/logs/bt4_metricx_dl.log 2>&1 || echo "MetricX download failed: ~/logs/bt4_metricx_dl.log"
      if job 0 bt4_probe_qe probe_qe; then
        gsutil -m -q cp "$D/probe/report/*" "$O/probe/report/"
      else
        echo "MetricX-QE of the probe failed (report only; the run goes on): ~/logs/bt4_probe_qe.log"
      fi
    fi
  fi
  log "full"
  if ! gsutil -q stat "$O/full/prompt.txt"; then
    echo "WAITING for $O/full/prompt.txt"
    until gsutil -q stat "$O/full/prompt.txt"; do sleep 60; done
  fi
  prompt=$(gsutil cat "$O/full/prompt.txt")
  log "prompt $prompt"
  ( HF_HUB_OFFLINE=0 "$PY" -u "$C/bt_gemma4.py" publish --plan "$D/plan" --root "$O/full" --data "$D/hf" --repo "$REPO" \
      --region "$REGION" --owner "$(hostname)/publish" --run "$RUN_ID" --work-dir "$SHM/pub4" --interval 300 \
      > ~/logs/bt4_publish.log 2>&1; code=$?; echo "EXIT $code" >> ~/logs/bt4_publish.log; exit $code ) & pub=$!
  job 0 bt4_full_0 work_loop "$D/plan" "$O/full" "$prompt" & a=$!
  job 1 bt4_full_1 work_loop "$D/plan" "$O/full" "$prompt" & b=$!
  wait $a && ca=0 || ca=$?
  wait $b && cb=0 || cb=$?
  if [ $ca != 0 ] || [ $cb != 0 ]; then
    kill $pub 2> /dev/null || true
    echo "full run failed: half 0 $ca, half 1 $cb"; exit 1
  fi
  log "region complete; waiting for the publisher"
  wait $pub && cp=0 || cp=$?
  gsutil -q cp ~/logs/bt4_publish.log "$O/logs/bt4_publish-$(hostname)-$RUN_ID.log" || true
  [ $cp = 0 ] || { echo "publisher failed: ~/logs/bt4_publish.log"; exit 1; }
  log "final"
  HF_HUB_OFFLINE=0 "$PY" -u "$C/bt_gemma4.py" final --plan "$D/plan" --root "$O/full" --repo "$REPO" \
    --owner "$(hostname)/final" --run "$RUN_ID" --work-dir "$SHM/final4"
  log "done"
  ;;
*)
  echo "usage: run_bt_gemma4.sh all"; exit 1 ;;
esac
