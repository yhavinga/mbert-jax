#!/bin/bash
# On a spot v6e-8 in europe-west4 or us-east1: the smaller Gemma 4 instruction models (E2B, E4B, 12B, 26B-A4B) as en->nl
# translators, with the setup of the 31B's run (run_bt_natural.sh half1 and report, run_g4_blind.sh): prompt
# natural-v1, each model's own chat turn, greedy, the classes and decode limits of translate_models.yml, shards of 256;
# the four sets and FLAN (1,000) scored with MetricX-24 hybrid XXL (reference and QE) and COMET-22 on the CPU,
# evalsets.py collect's checks, armeval's length measures, the classifier's texts (armeval.py texts, model G4-<size>),
# and the 100 passages of the blind assessment translated (evalsets.py collect, packed on co1).
#   bash run_g4_sizes.sh all      everything; each step skips what is in the bucket, so a run started again after a
#                                 preemption goes on where the last stopped (at most a shard per class is lost)
#   bash run_g4_sizes.sh chips    the chip check only: one process per chip group, each asserting its device count
# Chips: a process takes as many chips as its model's KV heads allow (translate_models.yml chips): E2B, E4B and 12B
# one each (ordinals 0, 1, 2 of the chips sorted by PCI root, as run_bt_natural.sh's halves), 26B two (ordinals 4, 5);
# MetricX the four chips of ordinals 0-3 once E2B, E4B and 12B are done. The conversion of each model (g4_sizes.py
# convert) runs on the CPU from Hugging Face into this region's bucket, beside the setup; check (g4_sizes.py check)
# compares a few of each model's translations with the Hugging Face model on the CPU and stops that model's report
# when they disagree.
# Output in this region's bucket only, $O = $B/gemma4-sizes: ckpt/, data/<size>/, en-nl/<size>/<part>/w0/<class>/,
# eval/<size>/pairs-<part>.parquet, scores/<size>/, check/<size>.json, report/<size>/, report/sizes.md, logs/.
set -euo pipefail
# systemd-logind (RemoveIPC=yes) deletes a user's files in /dev/shm when the user's last session ends (9 Nov)
sudo -n loginctl enable-linger "$(id -un)"
SHM=/dev/shm/translate-generic
C=$SHM/code
D=$SHM/g4s
PY=$SHM/venv-fork/bin/python
CPY=$SHM/comet-venv/bin/python
ZONE=$(curl -fs -H Metadata-Flavor:Google http://metadata.google.internal/computeMetadata/v1/instance/zone | awk -F/ '{print $NF}')
case ${ZONE%-*} in
  europe-west4) BUCKET=gs://t5-dutch-english ;;
  us-east1) BUCKET=gs://t5-dutch-english-use1 ;;
  *) echo "no bucket for zone '$ZONE'"; exit 1 ;;
esac
B=$BUCKET/mbert/translator
O=$B/gemma4-sizes
IN=$B/bt-natural/eval-input
SIZES=(e2b e4b 12b 26b)
SHARD_SIZE=256
RUN_ID=$(hostname | sed 's/.*-\([^-]*\)-w-0$/\1/')-$(date +%s)
# MaxText dumps HLO to /tmp/xla_dump whenever XLA_FLAGS is empty (run_sft.sh)
export XLA_FLAGS=--xla_dump_hlo_as_text=false TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 HF_HUB_DISABLE_PROGRESS_BARS=1
export HF_XET_CACHE=$SHM/xet-cache
mkdir -p ~/logs "$D"

model_of() { echo "gemma4-$1"; }
chips_of() { case $1 in e2b) echo 0 ;; e4b) echo 1 ;; 12b) echo 2 ;; 26b) echo 4 5 ;; metricx) echo 0 1 2 3 ;; esac; }
ckpt_of() { echo "$O/ckpt/gemma-4-$1-it-maxtext"; }

groups() {  # the chips' vfio groups, ordered by PCI root and group number (run_bt_natural.sh's half takes 4 of them)
  for g in /dev/vfio/[0-9]*; do n=${g##*/}; dev=$(ls "/sys/kernel/iommu_groups/$n/devices"); echo "${dev%:*} $n"; done \
    | LC_ALL=C sort -k1,1 -k2,2n | awk '{print $2}'
}

on_chips() {  # ordinals...: the TPU environment of one process on these chips (1, 2 or 4 of them)
  local all sel=() k bounds port
  mapfile -t all < <(groups)
  [ "${#all[@]}" = 8 ] || { echo "vfio groups '${all[*]}', not 8"; exit 2; }
  for k in "$@"; do sel+=("${all[$k]}"); done
  case ${#sel[@]} in
    1) bounds=1,1,1 ;;
    2) bounds=${G4S_BOUNDS_2:-1,2,1} ;;
    4) bounds=2,2,1 ;;
    *) echo "no process bounds for ${#sel[@]} chips"; exit 2 ;;
  esac
  port=$((8480 + $1))
  export TPU_CHIPS_PER_PROCESS_BOUNDS=$bounds TPU_PROCESS_BOUNDS=1,1,1 TPU_VISIBLE_CHIPS=$(IFS=,; echo "${sel[*]}")
  export TPU_PROCESS_PORT=$port TPU_PROCESS_ADDRESSES=localhost:$port CLOUD_TPU_TASK_ID=0 ALLOW_MULTIPLE_LIBTPU_LOAD=1
  echo "chips $*: TPU_VISIBLE_CHIPS=$TPU_VISIBLE_CHIPS bounds $bounds port $port"
}

preflight() {  # n: one process with n chips
  timeout 600 "$PY" -c "import jax; assert (jax.device_count(), jax.process_count()) == ($1, 1), (jax.device_count(), jax.process_count()); print('devices', jax.devices())" || { echo "preflight failed: not $1 chips of one process"; exit 2; }
}

job() {  # name ordinals log command...: the command on these chips, its log ending in an EXIT line and in the bucket
  local name=$1 ordinals=$2 log=$3 code pid
  shift 3
  # Detached and then waited for: a subshell whose status a && or || tests runs with set -e ignored
  # shellcheck disable=SC2086
  ( on_chips $ordinals; preflight "$(echo $ordinals | wc -w)"; "$@" ) > ~/logs/$log.log 2>&1 &
  pid=$!
  wait $pid && code=0 || code=$?
  echo "EXIT $code" >> ~/logs/$log.log
  gsutil -q cp ~/logs/$log.log "$O/logs/$log-$RUN_ID.log" || true
  echo "$name: EXIT $code $(date -u +%T)"
  return $code
}

tokenizer() {  # size: the model's tokenizer, chat template and configs at its pinned revision
  local size=$1 repo rev
  read -r repo rev < <("$PY" -c "import sys; sys.path.insert(0, '$C'); import translate as t; m = t.MODELS['$(model_of "$size")']; print(m['hf_repo'], m['hf_revision'])")
  [ -s "$SHM/tok-$size/tokenizer.json" ] || HF_HUB_OFFLINE=0 "$PY" -c "from huggingface_hub import snapshot_download as s; s('$repo', revision='$rev', local_dir='$SHM/tok-$size', allow_patterns=['tokenizer*', 'chat_template.jinja', 'config.json', 'generation_config.json'])"
}

snapshot() {  # size: the whole snapshot (for the conversion and the check)
  local size=$1 repo rev
  read -r repo rev < <("$PY" -c "import sys; sys.path.insert(0, '$C'); import translate as t; m = t.MODELS['$(model_of "$size")']; print(m['hf_repo'], m['hf_revision'])")
  if [ ! -s "$SHM/hf-$size/done.txt" ]; then
    echo "$size: downloading $repo@$rev $(date -u +%T)"
    HF_HUB_OFFLINE=0 "$PY" -c "from huggingface_hub import snapshot_download as s; s('$repo', revision='$rev', local_dir='$SHM/hf-$size')"
    echo "$repo@$rev $(du -sh "$SHM/hf-$size" | cut -f1)" > "$SHM/hf-$size/done.txt"
  fi
  cat "$SHM/hf-$size/done.txt"
}

convert() {  # size: the MaxText checkpoint in this region's bucket, unless it is there
  local size=$1 ck
  ck=$(ckpt_of "$size")
  if gsutil -q stat "$ck/done.txt"; then echo "$size: $ck done before"; return 0; fi
  snapshot "$size"
  if [ ! -d "$SHM/ckpt-$size/0/items" ]; then
    echo "$size: converting $(date -u +%T)"
    JAX_PLATFORMS=cpu "$PY" -u "$C/g4_sizes.py" convert --model "$(model_of "$size")" --hf "$SHM/hf-$size" \
      --out "$SHM/ckpt-$size"
  fi
  [ -d "$SHM/ckpt-$size/0/items" ] || { echo "$size: no $SHM/ckpt-$size/0/items after the conversion"; return 1; }
  echo "$size: $(du -sh "$SHM/ckpt-$size" | cut -f1) converted; copying to $ck $(date -u +%T)"
  gsutil -m -q rsync -r "$SHM/ckpt-$size" "$ck"
  echo "$(cat "$SHM/hf-$size/done.txt") converted by g4_sizes.py on $(hostname) at $(date -u +%FT%TZ)" | gsutil -q cp - "$ck/done.txt"
  echo "$size: $ck done $(date -u +%T)"
}

data() {  # size: the segments of the four sets, FLAN and the blind passages in this model's tokens
  local size=$1 d=$D/$1/eval
  if gsutil -q stat "$O/data/$size/done.txt"; then
    mkdir -p "$d" && gsutil -m -q rsync -r "$O/data/$size" "$d"
    return 0
  fi
  "$PY" -u "$C/bt_natural.py" evalsegments --four "$D/in/four/evalsets.parquet" --x "$D/in/x/evalsets.parquet" \
    --tokenizer "$SHM/tok-$size" --model "$(model_of "$size")" --blind --out "$d"
  gsutil -m -q rsync -r "$d" "$O/data/$size"
  date -u +%FT%TZ | gsutil -q cp - "$O/data/$size/done.txt"
}

translate_part() {  # size part: every class of the part, skipping stored shards; then collect's pairs in the bucket
  local size=$1 part=$2 class out=$O/en-nl/$1/$2 d=$D/$1/eval
  if ! gsutil -q stat "$out/done.txt"; then
    for class in $(cat "$d/$part/classes.txt"); do
      "$PY" -u "$C/translate.py" --segments "$d/$part/segments-w0.parquet" --out "$out/w0/$class" \
        --tokenizer "$SHM/tok-$size" --model "$(model_of "$size")" --prompt natural-v1 --length-class "$class" \
        --source-lang en --target-lang nl --work-dir "$SHM/work-$size-$part" --shard-size $SHARD_SIZE
    done
    echo "$size $part natural-v1 en->nl $(date -u +%FT%TZ)" | gsutil -q cp - "$out/done.txt"
  fi
  mkdir -p "$d/shards-$part"
  gsutil -m -q rsync -r "$out/w0" "$d/shards-$part"
  "$PY" -u "$C/evalsets.py" collect --data "$d/$part" --worker 0 --shards "$d/shards-$part" --out "$d/pairs-$part.parquet"
  gsutil -q cp "$d/pairs-$part.parquet" "$O/eval/$size/"
}

translate_model() {  # size: the four sets and FLAN (log translate_eval, read for the throughput), then the blind passages
  local size=$1 part log
  for part in four flan blind; do
    if [ $part = blind ]; then log=translate_blind-$RUN_ID; else log=translate_eval-$RUN_ID; fi
    translate_part "$size" $part 2>&1 | tee -a ~/logs/$size-$log.log
    gsutil -q cp ~/logs/$size-$log.log "$O/logs/$size/$log.log"
  done
}

wait_for() {  # url pid...: until url is in the bucket; an error once none of the pids runs any more
  local url=$1
  shift
  until gsutil -q stat "$url"; do
    local alive=no p
    for p in "$@"; do kill -0 "$p" 2> /dev/null && alive=yes; done
    [ $alive = yes ] || { echo "$url never came"; return 1; }
    sleep 60
  done
}

check() {  # size pid: once the four sets' pairs are there, a few of the small class against Hugging Face's model
  local size=$1 d=$D/$1/eval
  if gsutil -q stat "$O/check/$size.json"; then return 0; fi
  wait_for "$O/eval/$size/pairs-four.parquet" "$2"
  snapshot "$size"
  mkdir -p "$d/check-shards"
  gsutil -m -q rsync -r "$O/en-nl/$size/four/w0/small" "$d/check-shards"
  "$PY" -u "$C/g4_sizes.py" check --model "$(model_of "$size")" --hf "$SHM/hf-$size" --tokenizer "$SHM/tok-$size" \
    --segments "$d/four/segments-w0.parquet" --shards "$d/check-shards" --out "$d/check-$size.json"
  gsutil -q cp "$d/check-$size.json" "$O/check/$size.json"
}

comet() {  # size:pid...: COMET-22 of the four sets on the CPU, one model after the other as their pairs come
  local item size pid d
  for item in "$@"; do
    size=${item%%:*} pid=${item#*:} d=$D/$size/eval
    gsutil -q stat "$O/scores/$size/comet-four.parquet" && continue
    wait_for "$O/eval/$size/pairs-four.parquet" "$pid"
    gsutil -q cp "$O/eval/$size/pairs-four.parquet" "$d/comet-input.parquet"
    "$CPY" -I -u "$C/comet_score.py" score --root "$SHM/comet" --pairs "$d/comet-input.parquet" \
      --out "$d/comet-four.parquet" --time-first 100 --timing "$d/comet-timing.json" --threads "${COMET_THREADS:-88}"
    gsutil -q cp "$d/comet-four.parquet" "$d/comet-timing.json" "$O/scores/$size/"
  done
}

metricx() {  # size:pid...: MetricX-24 of the four sets (reference and QE) and of FLAN (QE), as their pairs come
  local item size pid d part args
  for item in "$@"; do
    size=${item%%:*} pid=${item#*:} d=$D/$size/eval
    wait_for "$O/eval/$size/pairs-flan.parquet" "$pid"
    for part in four flan; do
      if gsutil -q stat "$O/scores/$size/metricx-$part.parquet"; then continue; fi
      gsutil -q cp "$O/eval/$size/pairs-$part.parquet" "$d/pairs-$part.parquet"
      if [ $part = four ]; then args=(--pair source:hypothesis:reference --pair source:hypothesis)
      else args=(--pair source:hypothesis); fi
      JAX_COMPILATION_CACHE_DIR=$SHM/jax_cache_metricx "$PY" -u "$C/metricx_qe.py" score --pairs "$d/pairs-$part.parquet" \
        "${args[@]}" --model xxl --root "$SHM/metricx" --empty-candidates --out "$d/metricx-$part.parquet"
      gsutil -q cp "$d/metricx-$part.parquet" "$O/scores/$size/"
    done
  done
}

report_model() {  # size: evalcompare against TranslateGemma, the texts of the classifier, the counts of the checks
  local size=$1 d=$D/$1/eval r=$D/$1/report
  mkdir -p "$r" "$D/$1/logs"
  for f in pairs-four pairs-flan pairs-blind; do gsutil -q cp "$O/eval/$size/$f.parquet" "$d/"; done
  for f in metricx-four metricx-flan comet-four; do gsutil -q cp "$O/scores/$size/$f.parquet" "$d/"; done
  gsutil -m -q cp "$O/logs/$size/translate_eval-*.log" "$D/$1/logs/"
  [ -n "$(ls "$D/$1/logs")" ] || { echo "$size: no translate_eval logs in $O/logs/$size"; return 1; }
  "$PY" -u "$C/bt_natural.py" evalcompare --data "$d/four" --pairs "$d/pairs-four.parquet" \
    --metricx "$d/metricx-four.parquet" --comet "$d/comet-four.parquet" --flan-pairs "$d/pairs-flan.parquet" \
    --flan-metricx "$d/metricx-flan.parquet" --base "$D/in/tg27b" --logs "$D/$1/logs" \
    --label "$("$PY" -c "import sys; sys.path.insert(0, '$C'); import g4_sizes as g; print(g.LABELS['$size'])")" \
    --out "$r/eval-$size.md"
  "$PY" -u "$C/g4_sizes.py" extra --size "$size" --pairs-four "$d/pairs-four.parquet" --pairs-flan "$d/pairs-flan.parquet" \
    --out "$r/extra-$size.json"
  # The texts of the classifier (rule (b)), as armeval.py texts writes them: WMT24++ (960) and FLAN, model G4-<size>
  "$PY" -u "$C/armeval.py" texts --evalsets "$d/four/evalsets.parquet" --wmt "G4-$size=$d/pairs-four.parquet" \
    --flan "G4-$size=$d/pairs-flan.parquet" --out "$r/texts.parquet"
  gsutil -m -q cp "$r"/* "$O/report/$size/"
}

step=${1:-}
case $step in
chips)
  pids=()
  # shellcheck disable=SC2046
  for size in "${SIZES[@]}"; do ( on_chips $(chips_of "$size"); preflight "$(chips_of "$size" | wc -w)" ) & pids+=($!); done
  code=0
  for p in "${pids[@]}"; do wait "$p" || code=1; done
  exit $code
  ;;
all)
  echo "== $(hostname) $ZONE $O $(date -u +%T)"
  if [ ! -s "$SHM/venv-fork/done.txt" ] || [ ! -s "$SHM/tokenizer-gemma4/tokenizer.json" ]; then
    HF_HUB_OFFLINE=0 bash "$C/setup_generic.sh" venv
  fi
  cat "$SHM/venv-fork/done.txt"
  # Before anything runs in the background: every chip group must come up as a process of its own
  echo "== chips $(date -u +%T)"
  bash "$0" chips
  ( HF_HUB_OFFLINE=0 SHM=$SHM bash "$C/setup_comet.sh" && HF_HUB_OFFLINE=0 "$PY" -c "import sys; sys.path.insert(0, '$C'); import metricx_qe as m; print(m.download('xxl', '$SHM/metricx'))" ) \
    > ~/logs/g4s_setup_bg.log 2>&1 & setup_bg=$!
  for size in "${SIZES[@]}"; do tokenizer "$size"; done
  declare -A conv
  for size in "${SIZES[@]}"; do
    convert "$size" > ~/logs/g4s_convert_$size.log 2>&1 & conv[$size]=$!
  done
  echo "== data $(date -u +%T)"
  mkdir -p "$D/in"
  for p in four x tg27b; do mkdir -p "$D/in/$p" && gsutil -m -q rsync -r "$IN/$p" "$D/in/$p"; done
  for size in "${SIZES[@]}"; do data "$size"; done
  echo "== translate $(date -u +%T)"
  declare -A tr
  for size in "${SIZES[@]}"; do
    (
      wait_conv() { while kill -0 "$1" 2> /dev/null; do sleep 20; done; }
      wait_conv "${conv[$size]}"
      gsutil -q stat "$(ckpt_of "$size")/done.txt" || { echo "$size: no checkpoint: ~/logs/g4s_convert_$size.log"; exit 1; }
      job "$size" "$(chips_of "$size")" "g4s_translate_$size" translate_model "$size"
    ) & tr[$size]=$!
  done
  wait $setup_bg || { echo "background setup failed: ~/logs/g4s_setup_bg.log"; exit 1; }
  ( comet "e2b:${tr[e2b]}" "12b:${tr[12b]}" ) > ~/logs/g4s_comet_a.log 2>&1 & ca=$!
  ( comet "e4b:${tr[e4b]}" "26b:${tr[26b]}" ) > ~/logs/g4s_comet_b.log 2>&1 & cb=$!
  declare -A ck
  for size in "${SIZES[@]}"; do check "$size" "${tr[$size]}" > ~/logs/g4s_check_$size.log 2>&1 & ck[$size]=$!; done
  # MetricX on chips 0-3 once E2B, E4B and 12B are done with them
  failed=()
  for size in e2b e4b 12b; do wait "${tr[$size]}" || failed+=("translate $size"); done
  ok=(); for size in "${SIZES[@]}"; do [[ " ${failed[*]} " == *" translate $size "* ]] || ok+=("$size:${tr[$size]}"); done
  job metricx "$(chips_of metricx)" g4s_metricx metricx "${ok[@]}" || failed+=("metricx")
  wait "${tr[26b]}" || failed+=("translate 26b")
  wait $ca || failed+=("comet a")
  wait $cb || failed+=("comet b")
  for size in "${SIZES[@]}"; do wait "${ck[$size]}" || failed+=("check $size"); done
  echo "== reports $(date -u +%T); failed: ${failed[*]:-none}"
  evals=()
  for size in "${SIZES[@]}"; do
    if gsutil -q stat "$O/scores/$size/comet-four.parquet" && gsutil -q stat "$O/scores/$size/metricx-flan.parquet" \
        && gsutil -q stat "$O/eval/$size/pairs-blind.parquet"; then
      report_model "$size" > ~/logs/g4s_report_$size.log 2>&1 || failed+=("report $size")
      [ -s "$D/$size/report/eval-$size.json" ] && evals+=("$size=$D/$size/report/eval-$size.json")
    fi
  done
  gsutil -q cp "$B/bt-natural/report/eval-gemma4.json" "$B/bt-natural/report/refqe.json" "$D/"
  if [ ${#evals[@]} -gt 0 ]; then
    "$PY" -u "$C/g4_sizes.py" report --eval "${evals[@]}" --g4 "$D/eval-gemma4.json" --refqe "$D/refqe.json" \
      --out "$D/sizes.md"
    gsutil -q cp "$D/sizes.md" "$D/sizes.json" "$O/report/"
  fi
  for f in ~/logs/g4s_*.log; do gsutil -q cp "$f" "$O/logs/$(basename "$f" .log)-$RUN_ID.log" || true; done
  echo "== done $(date -u +%T); failed: ${failed[*]:-none}"
  [ ${#failed[@]} = 0 ]
  ;;
*)
  echo "usage: run_g4_sizes.sh all|chips"; exit 1 ;;
esac
