#!/bin/bash
# On a spot v6e-8 in europe-west4 or us-east1: the pilot of bt_natural.py, and Gemma 4 as an en->nl translator on the
# translator's evaluation sets. Gemma 4 31B's MaxText checkpoint is in europe-west4's bucket; in a region without one,
# en-nl-judge's setup.sh convert makes it (from Hugging Face, beside the other setup).
#   bash run_bt_natural.sh all   setup, data, MetricX-QE of the four sets' human references (refqe, the calibration of
#                                rule (a)), the probe (a handful of passages per prompt, shown in ~/logs/bt_probe_show.txt
#                                and $O/probe/show.txt); then, only once $O/probe/approved is in the bucket (put there
#                                after reading the probe), translate, score and report
# The v6e-8 runs as two TPUs of 4 chips (Gemma 4 31B allows at most 4-way tensor parallelism):
#   half 0: nl->en natural-v1 (the 5,000 passages), then MetricX of TranslateGemma's and natural-v1's English
#   half 1: en->nl natural-v1 on the four sets and FLAN and their MetricX, then nl->en bare-v1 and its MetricX;
#   COMET-22 of the four sets on the CPU beside it
# Everything done is in the bucket, and every step skips it there: a preempted spot VM loses at most a shard. Each run's
# logs go to $O/logs/ under their own name.
set -euo pipefail
# systemd-logind (RemoveIPC=yes) deletes a user's files in /dev/shm when the user's last session ends (9 Nov)
sudo -n loginctl enable-linger "$(id -un)"
SHM=/dev/shm/translate-generic
C=$SHM/code
D=$SHM/bt
PY=$SHM/venv-fork/bin/python
CPY=$SHM/comet-venv/bin/python
ZONE=$(curl -fs -H Metadata-Flavor:Google http://metadata.google.internal/computeMetadata/v1/instance/zone | awk -F/ '{print $NF}')
case ${ZONE%-*} in
  europe-west4) BUCKET=gs://t5-dutch-english ;;
  us-east1) BUCKET=gs://t5-dutch-english-use1 ;;
  *) echo "no bucket for zone '$ZONE'"; exit 1 ;;
esac
B=$BUCKET/mbert/translator
O=$B/bt-natural
TOK=$SHM/tokenizer-gemma4
CKPT=$B/judge/gemma-4-31b-it-maxtext/0/items
SHARD_SIZE=256
RUN_ID=$(hostname | sed 's/.*-\([^-]*\)-w-0$/\1/')-$(date +%s)
# MaxText dumps HLO to /tmp/xla_dump whenever XLA_FLAGS is empty (run_sft.sh)
export XLA_FLAGS=--xla_dump_hlo_as_text=false TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 HF_HUB_DISABLE_PROGRESS_BARS=1
mkdir -p ~/logs "$D"

half() {  # as run_gemma4_pilot.sh: the 4 chips behind the first (0) or second (1) PCI root as a TPU of their own
  local chips port
  chips=$(for g in /dev/vfio/[0-9]*; do n=${g##*/}; dev=$(ls "/sys/kernel/iommu_groups/$n/devices"); echo "${dev%:*} $n"; done \
    | LC_ALL=C sort -k1,1 -k2,2n | awk -v h="$1" '$1 != root {k++; root = $1} k == h + 1 {printf "%s%s", sep, $2; sep = ","}')
  [ "$(echo "$chips" | tr , '\n' | grep -c .)" = 4 ] || { echo "half $1: vfio groups '$chips', not 4 behind one PCI root"; exit 2; }
  echo "half $1: TPU_VISIBLE_CHIPS=$chips"
  if [ "$1" = 0 ]; then port=8476; else port=8477; fi
  export TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 TPU_PROCESS_BOUNDS=1,1,1 TPU_VISIBLE_CHIPS=$chips TPU_PROCESS_PORT=$port
  export TPU_PROCESS_ADDRESSES=localhost:$port CLOUD_TPU_TASK_ID=0 ALLOW_MULTIPLE_LIBTPU_LOAD=1
}

preflight() {
  timeout 600 "$PY" -c "import jax; assert (jax.device_count(), jax.process_count()) == (4, 1), (jax.device_count(), jax.process_count()); print('devices', jax.devices())" || { echo "preflight failed: not 4 chips of one process"; exit 2; }
}

translate() {  # name prompt source target segments out classes: every class of the segments, skipping stored shards
  local name=$1 prompt=$2 source=$3 target=$4 segments=$5 out=$6 classes=$7 class
  if gsutil -q stat "$out/done.txt"; then echo "$name: done before"; return 0; fi
  for class in $classes; do
    "$PY" -u "$C/translate.py" --segments "$segments" --out "$out/w0/$class" --tokenizer "$TOK" --model gemma4-31b \
      --prompt "$prompt" --length-class "$class" --source-lang "$source" --target-lang "$target" --ckpt "$CKPT" \
      --work-dir "$SHM/work-$name" --shard-size $SHARD_SIZE
  done
  echo "$name $prompt $source->$target $(date -u +%FT%TZ)" | gsutil -q cp - "$out/done.txt"
}

job() {  # half log command...: the command in the environment of a half, its log ending in an EXIT line and in the bucket
  local h=$1 log=$2 code pid
  shift 2
  # Detached and then waited for: a subshell whose status a && or || tests runs with set -e ignored
  ( half "$h"; preflight; "$@" ) > ~/logs/$log.log 2>&1 &
  pid=$!
  wait $pid && code=0 || code=$?
  echo "EXIT $code" >> ~/logs/$log.log
  gsutil -q cp ~/logs/$log.log "$O/logs/$log-$RUN_ID.log" || true
  return $code
}

metricx() {  # pairs out pair...: MetricX-24 hybrid XXL, unless out is in the bucket
  local pairs=$1 out=$2
  shift 2
  if gsutil -q stat "$O/scores/$(basename "$out")"; then gsutil -q cp "$O/scores/$(basename "$out")" "$out"; return 0; fi
  local args=()
  for p in "$@"; do args+=(--pair "$p"); done
  JAX_COMPILATION_CACHE_DIR=$SHM/jax_cache_metricx "$PY" -u "$C/metricx_qe.py" score --pairs "$pairs" "${args[@]}" \
    --model xxl --root "$SHM/metricx" --empty-candidates --out "$out"
  gsutil -q cp "$out" "$O/scores/"
}

shards() {  # out dir: a local copy of translate.py's shards
  mkdir -p "$2"
  gsutil -m -q rsync -r "$1/w0" "$2"
}

half0() {
  translate natural natural-v1 nl en "$D/segments-nl.parquet" "$O/nl-en/natural" "$(cat "$D/classes.txt")"
  shards "$O/nl-en/natural" "$D/shards/natural"
  "$PY" -u "$C/bt_natural.py" merge --sample "$D/sample.parquet" --out "$D/compare-tg.parquet"
  "$PY" -u "$C/bt_natural.py" merge --sample "$D/sample.parquet" --shards "natural=$D/shards/natural" --out "$D/compare-natural.parquet"
  metricx "$D/compare-tg.parquet" "$D/qe-tg.parquet" nl:en_tg en_tg:nl
  metricx "$D/compare-natural.parquet" "$D/qe-natural.parquet" nl:en_natural en_natural:nl
}

half1() {
  local part
  for part in four flan; do
    translate "eval-$part" natural-v1 en nl "$D/eval/$part/segments-w0.parquet" "$O/en-nl/$part" "$(cat "$D/eval/$part/classes.txt")"
    shards "$O/en-nl/$part" "$D/eval/shards-$part"
    "$PY" -u "$C/evalsets.py" collect --data "$D/eval/$part" --worker 0 --shards "$D/eval/shards-$part" \
      --out "$D/eval/pairs-$part.parquet"
    gsutil -q cp "$D/eval/pairs-$part.parquet" "$O/eval/"
  done
  metricx "$D/eval/pairs-four.parquet" "$D/eval/metricx-four.parquet" source:hypothesis:reference source:hypothesis
  metricx "$D/eval/pairs-flan.parquet" "$D/eval/metricx-flan.parquet" source:hypothesis
  translate bare bare-v1 nl en "$D/segments-nl.parquet" "$O/nl-en/bare" "$(cat "$D/classes.txt")"
  shards "$O/nl-en/bare" "$D/shards/bare"
  "$PY" -u "$C/bt_natural.py" merge --sample "$D/sample.parquet" --shards "bare=$D/shards/bare" --out "$D/compare-bare.parquet"
  metricx "$D/compare-bare.parquet" "$D/qe-bare.parquet" nl:en_bare en_bare:nl
}

probe0() {
  translate probe-natural natural-v1 nl en "$D/probe-nl.parquet" "$O/probe/natural" medium
  translate probe-en-nl natural-v1 en nl "$D/eval/probe-en.parquet" "$O/probe/en-nl" small
}

probe1() {
  translate probe-bare bare-v1 nl en "$D/probe-nl.parquet" "$O/probe/bare" medium
}

refqe() {  # the calibration of rule (a): the human references of the four sets scored as an arm's translations
  mkdir -p "$D/report"
  metricx "$D/eval/four/evalsets.parquet" "$D/eval/refqe.parquet" source:reference
  "$PY" -u "$C/bt_natural.py" refqe --evalsets "$D/eval/four/evalsets.parquet" --qe "$D/eval/refqe.parquet" \
    --base "$D/eval-input/tg27b" --out "$D/report/refqe.md"
  gsutil -q cp "$D/report/refqe.md" "$D/report/refqe.json" "$O/report/"
}

comet() {  # pid: COMET-22 on the CPU, once the en->nl pairs of the four sets are in the bucket (while pid, half 1, runs)
  until gsutil -q stat "$O/eval/pairs-four.parquet"; do
    kill -0 "$1" 2> /dev/null || { echo "half 1 ended without the pairs of the four sets"; return 1; }
    sleep 60
  done
  if gsutil -q stat "$O/scores/comet-four.parquet"; then return 0; fi
  gsutil -q cp "$O/eval/pairs-four.parquet" "$D/eval/comet-input.parquet"
  "$CPY" -I -u "$C/comet_score.py" score --root "$SHM/comet" --pairs "$D/eval/comet-input.parquet" \
    --out "$D/eval/comet-four.parquet" --time-first 100 --timing "$D/eval/comet-timing.json"
  gsutil -q cp "$D/eval/comet-four.parquet" "$D/eval/comet-timing.json" "$O/scores/"
}

step=${1:-}
case $step in
all)
  # Setup: the fork's venv and the tokenizers; COMET's venv and model and MetricX's weights beside the data and probe
  if [ ! -s "$SHM/venv-fork/done.txt" ] || [ ! -s "$TOK/tokenizer.json" ]; then
    HF_HUB_OFFLINE=0 bash "$C/setup_generic.sh" venv
  fi
  ( SHM=$SHM bash "$C/setup_comet.sh" && HF_HUB_OFFLINE=0 "$PY" -c "import sys; sys.path.insert(0, '$C'); import metricx_qe as m; print(m.download('xxl', '$SHM/metricx'))" ) \
    > ~/logs/bt_setup_bg.log 2>&1 & setup_bg=$!
  if gsutil -q stat "$B/judge/gemma-4-31b-it-maxtext/done.txt"; then
    convert_bg=
  else
    mkdir -p /dev/shm/judge/code && gsutil -q cp "$B/judge/scripts/setup.sh" /dev/shm/judge/code/setup.sh
    HF_HUB_OFFLINE=0 bash /dev/shm/judge/code/setup.sh convert > ~/logs/bt_convert.log 2>&1 & convert_bg=$!
  fi
  echo "== data $(date -u +%T)"
  if gsutil -q stat "$O/data/done.txt"; then
    gsutil -m -q rsync -r "$O/data" "$D"
  else
    HF_HUB_OFFLINE=0 "$PY" -u "$C/bt_natural.py" fetch --out "$D/hf"
    "$PY" -u "$C/bt_natural.py" sample --data "$D/hf" --tokenizer "$TOK" --out "$D"
    mkdir -p "$D/eval-input"
    gsutil -m -q rsync -r "$O/eval-input" "$D/eval-input"
    "$PY" -u "$C/bt_natural.py" evalsegments --four "$D/eval-input/four/evalsets.parquet" \
      --x "$D/eval-input/x/evalsets.parquet" --tokenizer "$TOK" --out "$D/eval"
    gsutil -m -q rsync -r -x '^(hf|eval-input|shards)/.*' "$D" "$O/data"
    date -u +%FT%TZ | gsutil -q cp - "$O/data/done.txt"
  fi
  [ -d "$D/eval-input" ] || { mkdir -p "$D/eval-input" && gsutil -m -q rsync -r "$O/eval-input" "$D/eval-input"; }
  # head on the file, not after cat: with pipefail cat's SIGPIPE would end the script
  head -c 3000 "$D/sample.json"; echo
  wait $setup_bg || { echo "background setup failed: ~/logs/bt_setup_bg.log"; exit 1; }
  echo "== MetricX-QE of the human references $(date -u +%T)"
  if ! gsutil -q stat "$O/report/refqe.md"; then job 1 bt_refqe refqe; fi
  gsutil -q cp "$O/report/refqe.md" -
  if [ -n "$convert_bg" ]; then
    echo "== waiting for the conversion of Gemma 4 $(date -u +%T)"
    wait $convert_bg || { echo "conversion failed: ~/logs/bt_convert.log"; exit 1; }
  fi
  echo "== probe $(date -u +%T)"
  if ! gsutil -q stat "$O/probe/show.txt"; then
    job 0 bt_probe_0 probe0 & a=$!
    job 1 bt_probe_1 probe1 & b=$!
    wait $a && ca=0 || ca=$?
    wait $b && cb=0 || cb=$?
    if [ $ca != 0 ] || [ $cb != 0 ]; then echo "probe failed: half 0 $ca, half 1 $cb"; exit 1; fi
    for p in natural bare en-nl; do shards "$O/probe/$p" "$D/probe/$p"; done
    {
      "$PY" -u "$C/bt_natural.py" show --segments "$D/probe-nl.parquet" --tokenizer "$TOK" \
        --shards "natural=natural-v1=$D/probe/natural" "bare=bare-v1=$D/probe/bare"
      "$PY" -u "$C/bt_natural.py" show --segments "$D/eval/probe-en.parquet" --tokenizer "$TOK" --source en --target nl \
        --shards "en-nl=natural-v1=$D/probe/en-nl"
    } > ~/logs/bt_probe_show.txt
    gsutil -q cp ~/logs/bt_probe_show.txt "$O/probe/show.txt"
  fi
  if ! gsutil -q stat "$O/probe/approved"; then
    echo "WAITING: read $O/probe/show.txt, then put $O/probe/approved and start again"
    exit 0
  fi
  echo "== translate and score $(date -u +%T)"
  job 0 bt_half0 half0 & a=$!
  job 1 bt_half1 half1 & b=$!
  comet $b > ~/logs/bt_comet.log 2>&1 &
  c=$!
  wait $c && cc=0 || cc=$?
  echo "EXIT $cc" >> ~/logs/bt_comet.log
  gsutil -q cp ~/logs/bt_comet.log "$O/logs/bt_comet-$RUN_ID.log" || true
  wait $a && ca=0 || ca=$?
  wait $b && cb=0 || cb=$?
  echo "== half 0 $ca, half 1 $cb, COMET $cc $(date -u +%T)"
  # Not "[ ... ] && [ ... ]": set -e ignores a failure left of && (9 Nov)
  if [ $ca != 0 ] || [ $cb != 0 ] || [ $cc != 0 ]; then echo "stops before the report"; exit 1; fi
  bash "$0" report
  ;;
report)
  mkdir -p "$D/logs" "$D/report"
  gsutil -m -q cp "$O/logs/bt_half*.log" "$D/logs/"
  # bt_natural.py throughput reads translate_<setup>-*.log: the translate.py lines of each half's logs per setup
  for f in "$D"/logs/bt_half0-*.log; do cp "$f" "$D/logs/translate_natural-$(basename "$f" .log | cut -d- -f2-).log"; done
  for f in "$D"/logs/bt_half1-*.log; do
    awk '/^MAXTEXT /{p = index($0, "\"prompt\": \"bare-v1\"") ? "bare" : "eval"} {print > (FILENAME "." p)}' "$f"
    [ -f "$f.bare" ] && mv "$f.bare" "$D/logs/translate_bare-$(basename "$f" .log | cut -d- -f2-).log"
    [ -f "$f.eval" ] && mv "$f.eval" "$D/logs/translate_eval-$(basename "$f" .log | cut -d- -f2-).log"
  done
  "$PY" -u "$C/bt_natural.py" merge --sample "$D/sample.parquet" --shards "natural=$D/shards/natural" \
    "bare=$D/shards/bare" --out "$D/compare.parquet"
  "$PY" - "$D/report/setup.json" "$ZONE" <<'EOF'
import json, os, sys
from pathlib import Path
fork = Path("/dev/shm/translate-generic/venv-fork/done.txt").read_text().strip()
Path(sys.argv[1]).write_text(json.dumps({
    "passages": "5.000 uit de vijf gepubliceerde BT-datasets (collectie yhavinga/dutch-back-translation-nlen, revisies van "
                "11 nov; alleen train): per bron 500 uniform (stratum random), 150 lang (385–511 TG-tokens), 175 met "
                "QE en→nl > 5, 175 uit het laagste deciel NL/EN-woorden; volgorde van een hash van de id",
    "TranslateGemma 27B": "het gepubliceerde Engels (kolom en) en zijn controles; MetricX opnieuw op deze VM",
    "Gemma 4 31B-it": "google/gemma-4-31B-it @ 842da37, omgezet naar MaxText 0.2.5 (en-nl-judge), translate.py "
                      "--model gemma4-31b --prompt natural-v1 of bare-v1 (translate_models.yml), greedy, eigen "
                      "chat-template met lege denkruimte, zonder context",
    "MaxText": "de fork yhavinga/maxtext, branch fix-nnx-decode-cache-abstract-init @ 2249474b9 (met de "
               "sliding-window-fix); pakketten: " + fork,
    "MetricX": "google/metricx-24-hybrid-xxl-v2p6-bfloat16 @ 3d9c6f7, metricx_qe.py (JAX-port), QE-modus, op 4 chips",
    "VM": f"{os.uname().nodename}, v6e-8 spot in {sys.argv[2]}, als twee TPU's van 4 chips"}, ensure_ascii=False))
EOF
  "$PY" -u "$C/bt_natural.py" report --compare "$D/compare.parquet" --qe "$D/qe-tg.parquet" "$D/qe-natural.parquet" \
    "$D/qe-bare.parquet" --logs "$D/logs" --fullset "$D/fullset.json" --setup "$D/report/setup.json" \
    --out "$D/report/bt-natural.md"
  "$PY" -u "$C/bt_natural.py" evalcompare --data "$D/eval/four" --pairs "$D/eval/pairs-four.parquet" \
    --metricx "$D/eval/metricx-four.parquet" --comet "$D/eval/comet-four.parquet" --flan-pairs "$D/eval/pairs-flan.parquet" \
    --flan-metricx "$D/eval/metricx-flan.parquet" --base "$D/eval-input/tg27b" --logs "$D/logs" \
    --out "$D/report/eval-gemma4.md"
  # The texts of the classifier (rule (b)), as armeval.py texts writes them: WMT24++ (960) and FLAN, model G4
  "$PY" -u "$C/armeval.py" texts --evalsets "$D/eval/four/evalsets.parquet" --wmt "G4=$D/eval/pairs-four.parquet" \
    --flan "G4=$D/eval/pairs-flan.parquet" --out "$D/report/texts.parquet"
  gsutil -m -q cp "$D/compare.parquet" "$D"/report/* "$O/report/"
  echo "== report done $(date -u +%T)"
  ;;
*)
  echo "usage: run_bt_natural.sh all|report"; exit 1 ;;
esac
