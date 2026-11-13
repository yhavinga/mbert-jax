#!/bin/bash
# On a spot v6e-8 in us-east1 or europe-west4: part A of the length trial of 13 Nov (bt_length.py): Gemma 4 31B-it
# (natural-v1) translates stretches of five length bins of real Dutch documents into English, whole (heel) and in
# pieces of whole paragraphs up to ~512 tokens (stukken); MetricX-24-QE nl->en per aligned paragraph; the report.
#   bash run_bt_length.sh all    setup, the stretches (from Hugging Face, one process per source), the segments, then
#                                on both halves (4 chips each) stukken and heel up to d8k, collect, MetricX, report A
#                                (bins up to 8192) and the English for part B
#   bash run_bt_length.sh d16k   the bin 16384 whole on both halves, then collect, MetricX and the report of all bins
#   bash run_bt_length.sh report collect, MetricX of what is not scored, the report (what is in the bucket)
# Everything done is in the bucket ($O) and every step skips it there: a preempted spot VM loses at most a shard.
# Logs: ~/logs/len_*.log (one per half and class), copied to $O/logs/.
set -euo pipefail
# systemd-logind (RemoveIPC=yes) deletes a user's files in /dev/shm when the user's last session ends (9 Nov)
sudo -n loginctl enable-linger "$(id -un)"
SHM=/dev/shm/translate-generic
C=$SHM/code-len
D=$SHM/bt-length
PY=$SHM/venv-fork/bin/python
ZONE=$(curl -fs -H Metadata-Flavor:Google http://metadata.google.internal/computeMetadata/v1/instance/zone | awk -F/ '{print $NF}')
case ${ZONE%-*} in
  europe-west4) BUCKET=gs://t5-dutch-english ;;
  us-east1) BUCKET=gs://t5-dutch-english-use1 ;;
  *) echo "no bucket for zone '$ZONE'"; exit 1 ;;
esac
B=$BUCKET/mbert/translator
O=${O:-$B/bt-length}
TOK=$SHM/tokenizer-gemma4
TOK_TG=$SHM/tokenizer-tg27b
CKPT=$B/judge/gemma-4-31b-it-maxtext/0/items
RES=$SHM/resources
OPENTAAL_COMMIT=b250510dda431785f962019167d1415198ff3905
SOURCES="handelingen raad dpc rechtspraak fineweb"
RUN_ID=$(hostname | sed 's/.*-\([^-]*\)-w-0$/\1/')-$(date +%s)
# MaxText dumps HLO to /tmp/xla_dump whenever XLA_FLAGS is empty (run_sft.sh)
export XLA_FLAGS=--xla_dump_hlo_as_text=false TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 HF_HUB_DISABLE_PROGRESS_BARS=1
mkdir -p ~/logs "$D"

half() {  # as run_bt_natural.sh: the 4 chips behind the first (0) or second (1) PCI root as a TPU of their own
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

shard_size() {  # cond class: a shard of 256 pieces, or of twice the class's slots for whole stretches
  [ "$1" = stukken ] && { echo 256; return; }
  case $2 in d512) echo 64 ;; d1k) echo 32 ;; d2k) echo 16 ;; d4k) echo 8 ;; d8k) echo 4 ;; d16k) echo 2 ;; *) echo 256 ;; esac
}

translate_class() {  # k cond class: one class of a half's segments; a run out of memory is marked and not fatal.
  # Both conditions take gemma4-31b-doc: every piece (at most 512 tokens) is in its class d512, one engine load
  local k=$1 cond=$2 cls=$3 model=gemma4-31b-doc log out code
  out=$O/$cond/w$k/$cls
  log=~/logs/len_${cond}_${cls}_w$k.log
  if gsutil -q stat "$out/oom.txt"; then echo "w$k $cond $cls: out of memory before ($out/oom.txt)"; return 0; fi
  echo "== w$k $cond $cls $(date -u +%T)"
  "$PY" -u "$C/bt_length_translate.py" --segments "$D/seg/$cond/segments-w$k.parquet" --out "$out" --tokenizer "$TOK" \
    --model "$model" --prompt natural-v1 --length-class "$cls" --source-lang nl --target-lang en --ckpt "$CKPT" \
    --work-dir "$SHM/work-len-$k" --shard-size "$(shard_size "$cond" "$cls")" > "$log" 2>&1 && code=0 || code=$?
  echo "EXIT $code" >> "$log"
  gsutil -q cp "$log" "$O/logs/len_${cond}_${cls}_w$k-$RUN_ID.log" || true
  grep -E '^(SHARD|DONE|MEM|engine)' "$log" | tail -n 4 | cut -c1-300
  if [ "$code" != 0 ]; then
    if grep -q -E 'RESOURCE_EXHAUSTED|Ran out of memory|[Oo]ut of memory' "$log"; then
      grep -m1 -E 'RESOURCE_EXHAUSTED|Ran out of memory|[Oo]ut of memory' "$log" | cut -c1-500 | gsutil -q cp - "$out/oom.txt"
      echo "w$k $cond $cls: OUT OF MEMORY, marked ($out/oom.txt)"
      return 0
    fi
    echo "w$k $cond $cls: EXIT $code"; tail -n 30 "$log"
    return "$code"
  fi
}

translate_half() {  # k classes...: the half's stukken (all their classes) and the named heel classes
  local k=$1 cls
  shift
  for cls in $(cat "$D/seg/stukken/classes.txt"); do translate_class "$k" stukken "$cls"; done
  for cls in "$@"; do
    grep -qw "$cls" "$D/seg/heel/classes.txt" && translate_class "$k" heel "$cls"
  done
  return 0
}

job() {  # half log command...: the command in the environment of a half, its log ending in an EXIT line and in the bucket
  local h=$1 log=$2 code pid
  shift 2
  ( half "$h"; preflight; "$@" ) > ~/logs/$log.log 2>&1 &
  pid=$!
  wait $pid && code=0 || code=$?
  echo "EXIT $code" >> ~/logs/$log.log
  gsutil -q cp ~/logs/$log.log "$O/logs/$log-$RUN_ID.log" || true
  return $code
}

both() {  # name classes...: translate_half on both halves at once
  local name=$1 a b ca cb
  shift
  job 0 "len_${name}_h0" translate_half 0 "$@" & a=$!
  job 1 "len_${name}_h1" translate_half 1 "$@" & b=$!
  wait $a && ca=0 || ca=$?
  wait $b && cb=0 || cb=$?
  echo "== $name: half 0 EXIT $ca, half 1 EXIT $cb $(date -u +%T)"
  if [ $ca != 0 ] || [ $cb != 0 ]; then tail -n 20 ~/logs/len_${name}_h0.log ~/logs/len_${name}_h1.log; exit 1; fi
}

metricx_half() {  # k stage: MetricX-24-QE nl->en of the half's pending units, unless they are scored or there are none
  local k=$1 stage=$2 n
  n=$("$PY" -c "import pyarrow.parquet as pq, sys; print(pq.read_metadata(sys.argv[1]).num_rows)" "$D/pending/pending-w$k.parquet")
  if [ "$n" = 0 ]; then echo "w$k: no unit to score"; return 0; fi
  if gsutil -q stat "$O/scores/qe-$stage-w$k.parquet"; then echo "w$k: qe-$stage scored before"; return 0; fi
  JAX_COMPILATION_CACHE_DIR=$SHM/jax_cache_metricx "$PY" -u "$C/metricx_qe.py" score --pairs "$D/pending/pending-w$k.parquet" \
    --pair src:cand --model xxl --root "$SHM/metricx" --out "$D/scores/qe-$stage-w$k.parquet"
  gsutil -q cp "$D/scores/qe-$stage-w$k.parquet" "$O/scores/"
}

score_and_report() {  # stage: collect, MetricX of the units not scored, report A, the English for part B
  local stage=$1 a b ca cb
  mkdir -p "$D/heel" "$D/stukken" "$D/scores" "$D/logs" "$D/report"
  gsutil -m -q rsync -r "$O/heel" "$D/heel" || true
  gsutil -m -q rsync -r "$O/stukken" "$D/stukken"
  gsutil -m -q rsync -r "$O/scores" "$D/scores" 2> /dev/null || true
  "$PY" -u "$C/bt_length.py" collect --docs "$D/data/docs.parquet" --heel "$D/heel" --stukken "$D/stukken" \
    --pieces "$D"/seg/stukken/segments-w*.parquet --source nl --target en --out "$D/collect"
  "$PY" -u "$C/bt_length.py" pending --units "$D/collect/units.parquet" --scored "$D"/scores/qe-*.parquet --workers 2 \
    --out "$D/pending"
  job 0 "len_metricx_${stage}_h0" metricx_half 0 "$stage" & a=$!
  job 1 "len_metricx_${stage}_h1" metricx_half 1 "$stage" & b=$!
  wait $a && ca=0 || ca=$?
  wait $b && cb=0 || cb=$?
  if [ $ca != 0 ] || [ $cb != 0 ]; then echo "MetricX failed: half 0 $ca, half 1 $cb"; tail -n 20 ~/logs/len_metricx_${stage}_h*.log; exit 1; fi
  gsutil -m -q cp "$O/logs/len_heel_*" "$O/logs/len_stukken_*" "$D/logs/"
  "$PY" - "$D/report/setup.json" "$ZONE" <<'EOF'
import json, os, sys
from pathlib import Path
fork = Path("/dev/shm/translate-generic/venv-fork/done.txt").read_text().strip()
Path(sys.argv[1]).write_text(json.dumps({
    "model": "Gemma 4 31B-it (google/gemma-4-31B-it @ 842da37, MaxText 0.2.5-checkpoint van en-nl-judge), prompt "
             "natural-v1, greedy, eigen chat-template met lege denkruimte; heel: translate_models.yml gemma4-31b-doc "
             "(één klasse per bak, decodeerlimiet 1,25 x de bron); in stukken: dezelfde, alle stukken in klasse d512",
    "MaxText": "fork yhavinga/maxtext, branch fix-nnx-decode-cache-abstract-init @ 2249474b9; pakketten: " + fork,
    "MetricX": "google/metricx-24-hybrid-xxl-v2p6-bfloat16 @ 3d9c6f7, metricx_qe.py (JAX), QE nl->en per uitgelijnde "
               "eenheid (bron: de Nederlandse alinea's, kandidaat: het Engels), invoer afgekapt op 1.536 tokens",
    "VM": f"{os.uname().nodename}, v6e-8 spot in {sys.argv[2]}, als twee TPU's van 4 chips"}, ensure_ascii=False))
EOF
  "$PY" -u "$C/bt_length.py" report --measured "$D/collect/measured.parquet" --units "$D/collect/units.parquet" \
    --qe "$D"/scores/qe-*.parquet --logs "$D/logs" --setup "$D/report/setup.json" --source nl --target en \
    --title "Deel A: Gemma 4 31B-it nl→en, heel tegen in stukken" --out "$D/report/bt-length-a.md" > /dev/null
  "$PY" -u "$C/bt_length.py" english --docs "$D/data/docs.parquet" --measured "$D/collect/measured.parquet" \
    --tokenizer "$TOK_TG" --out "$D/report/docs-en.parquet"
  gsutil -m -q cp "$D/report/bt-length-a.md" "$D/report/bt-length-a.json" "$D/report/setup.json" \
    "$D/report/docs-en.parquet" "$D/collect/measured.parquet" "$D/collect/units.parquet" "$O/report/"
  head -c 6000 "$D/report/bt-length-a.md"; echo
  echo "== report $stage done $(date -u +%T)"
}

step=${1:-}
case $step in
all)
  if [ ! -s "$SHM/venv-fork/done.txt" ] || [ ! -s "$TOK/tokenizer.json" ] || [ ! -s "$TOK_TG/tokenizer.json" ]; then
    HF_HUB_OFFLINE=0 bash "$C/setup_generic.sh" venv
  fi
  ( HF_HUB_OFFLINE=0 "$PY" -c "import sys; sys.path.insert(0, '$C'); import metricx_qe as m; print(m.download('xxl', '$SHM/metricx'))" ) \
    > ~/logs/len_metricx_download.log 2>&1 & dl=$!
  mkdir -p "$RES"
  if [ ! -s "$RES/opentaal-wordlist.txt" ]; then
    curl -fsSL "https://raw.githubusercontent.com/OpenTaal/opentaal-wordlist/$OPENTAAL_COMMIT/wordlist.txt" -o "$RES/opentaal-wordlist.txt.part"
    mv "$RES/opentaal-wordlist.txt.part" "$RES/opentaal-wordlist.txt"
  fi
  echo "== stretches $(date -u +%T)"
  if gsutil -q stat "$O/data/docs.parquet"; then
    mkdir -p "$D/data" && gsutil -m -q cp "$O/data/docs.parquet" "$O/data/docs.json" "$D/data/"
  else
    pids=""
    for s in $SOURCES; do
      HF_HUB_OFFLINE=0 "$PY" -u "$C/bt_length.py" docs --tokenizer "$TOK" --wordlist "$RES/opentaal-wordlist.txt" \
        --sources "$s" --out "$D/docs-$s" > ~/logs/len_docs_$s.log 2>&1 &
      pids="$pids $!"
    done
    for p in $pids; do wait "$p" || { echo "a docs run failed:"; tail -n 5 ~/logs/len_docs_*.log; exit 1; }; done
    for s in $SOURCES; do tail -n 1 ~/logs/len_docs_$s.log | cut -c1-300; done
    "$PY" -u "$C/bt_length.py" docs-merge --inputs $(for s in $SOURCES; do echo "$D/docs-$s"; done) --out "$D/data"
    gsutil -m -q cp "$D/data/docs.parquet" "$D/data/docs.json" "$O/data/"
    for s in $SOURCES; do gsutil -q cp ~/logs/len_docs_$s.log "$O/logs/len_docs_$s-$RUN_ID.log"; done
  fi
  echo "== segments $(date -u +%T)"
  if gsutil -q stat "$O/seg/segments.json"; then
    gsutil -m -q rsync -r "$O/seg" "$D/seg"
  else
    "$PY" -u "$C/bt_length.py" segments --docs "$D/data/docs.parquet" --tokenizer "$TOK" --model gemma4-31b-doc \
      --doc-model gemma4-31b-doc --prompt natural-v1 --source nl --target en --workers 2 --out "$D/seg"
    gsutil -m -q rsync -r "$D/seg" "$O/seg"
  fi
  cat "$D/seg/segments.json"; echo
  wait $dl || { echo "MetricX download failed: ~/logs/len_metricx_download.log"; exit 1; }
  echo "== translate: stukken, heel up to d8k $(date -u +%T)"
  both a d512 d1k d2k d4k d8k
  score_and_report a
  ;;
d16k)
  [ -s "$D/data/docs.parquet" ] && [ -s "$D/seg/segments.json" ] || { echo "run 'all' first"; exit 1; }
  echo "== translate: heel d16k $(date -u +%T)"
  both b d16k
  score_and_report b
  ;;
report)
  score_and_report "${2:-r$(date +%s)}"
  ;;
*)
  echo "usage: run_bt_length.sh all|d16k|report"; exit 1 ;;
esac
