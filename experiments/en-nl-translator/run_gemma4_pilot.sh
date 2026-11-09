#!/bin/bash
# On the v6e-8 of the Gemma 4 pilot, after setup_generic.sh, one step at a time (each started detached, its log ending
# in an EXIT line):
#   bash run_gemma4_pilot.sh checks      CPU: Gemma 4 decode against a forward pass (fork, and 0.2.5 if venv-025 is
#                                        there), TranslateGemma's prompt ids before and after translate_models.yml, the
#                                        unit tests with the real tokenizers
#   bash run_gemma4_pilot.sh translate   en->nl on chips 0-3 and nl->en on chips 4-7 at the same time
#   bash run_gemma4_pilot.sh post        merge; the context trial (chips 0-3) and MetricX-24-QE (chips 4-7) at the same
#                                        time; the report
# The v6e-8 is used as two TPUs of 4 chips: Gemma 4 31B allows at most 4-way tensor parallelism (4 KV heads in its
# global layers), as translate_models.yml says. Results go to $B/gemma4-pilot/; a second start of translate skips the
# shards that are there.
set -euo pipefail
SHM=/dev/shm/translate-generic
C=$SHM/code
D=$SHM/data
PY=$SHM/venv-fork/bin/python
B=gs://t5-dutch-english/mbert/translator
O=$B/gemma4-pilot
TOK=$SHM/tokenizer-gemma4
CKPT=$B/judge/gemma-4-31b-it-maxtext/0/items
CLASSES="tiny small medium long xlong"
# MaxText sets XLA_FLAGS to dump HLO to /tmp/xla_dump whenever XLA_FLAGS is empty (run_sft.sh); a flag that changes
# nothing keeps it from doing so
export XLA_FLAGS=--xla_dump_hlo_as_text=false TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
mkdir -p ~/logs "$SHM/out"

half() {  # the environment of one half of the v6e-8 as a TPU of its own: 0 = chips 0-3, 1 = chips 4-7
  local chips port
  if [ "$1" = 0 ]; then chips=0,1,2,3; port=8476; else chips=4,5,6,7; port=8477; fi
  export TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 TPU_PROCESS_BOUNDS=1,1,1 TPU_VISIBLE_CHIPS=$chips TPU_PROCESS_PORT=$port
  export TPU_PROCESS_ADDRESSES=localhost:$port CLOUD_TPU_TASK_ID=0 ALLOW_MULTIPLE_LIBTPU_LOAD=1
}

direction() {  # half source target segments log
  local h=$1 source=$2 target=$3 segments=$4 log=$5 code
  (
    half "$h"
    timeout 600 "$PY" -c "import jax; assert (jax.device_count(), jax.process_count()) == (4, 1), (jax.device_count(), jax.process_count()); print('devices', jax.devices())" || { echo "preflight failed: not 4 chips of one process"; exit 2; }
    for class in $CLASSES; do
      "$PY" -u "$C/translate.py" --segments "$segments" --out "$O/$source-$target/w0/$class" --tokenizer "$TOK" \
        --model gemma4-31b --length-class "$class" --source-lang "$source" --target-lang "$target" \
        --ckpt "$CKPT" --work-dir "$SHM/work-$source-$target" || exit $?
    done
  ) > ~/logs/$log.log 2>&1 && code=0 || code=$?
  echo "EXIT $code" >> ~/logs/$log.log
  gsutil -q cp ~/logs/$log.log "$O/logs/$log.log"
  return $code
}

case ${1:-} in
checks)
  [ -z "$(sudo lsof -t /dev/accel* /dev/vfio/* 2> /dev/null)" ] || echo "the TPU is in use: the checks run on the CPU anyway"
  echo "== decode, fork $(date -u +%T)"
  JAX_PLATFORMS=cpu "$PY" -u "$C/translate_checks.py" decode --work-dir "$SHM/checks-fork" 2>&1 | grep -E '^(MAXTEXT|CASE|DECODE)|Error' || true
  if [ -x "$SHM/venv-025/bin/python" ]; then
    echo "== decode, 0.2.5 $(date -u +%T)"
    JAX_PLATFORMS=cpu "$SHM/venv-025/bin/python" -u "$C/translate_checks.py" decode --work-dir "$SHM/checks-025" 2>&1 | grep -E '^(MAXTEXT|CASE|DECODE)|Error' || true
  fi
  echo "== prompts $(date -u +%T)"
  JAX_PLATFORMS=cpu "$PY" -u "$C/translate_checks.py" prompts --old "$C/translate_old.py" --pairs "$D/pairs.parquet" \
    --tokenizer "$SHM/tokenizer-tg27b"
  echo "== tests $(date -u +%T)"
  cd "$C" && JAX_PLATFORMS=cpu TRANSLATEGEMMA_TOKENIZER=$SHM/tokenizer-tg27b GEMMA4_TOKENIZER=$TOK \
    "$PY" -m pytest -q --noconftest -p no:cacheprovider tests/test_translate_models.py tests/test_gemma4_pilot.py \
    tests/test_en_nl_translator.py 2>&1 | tail -3
  ;;
translate)
  [ -z "$(sudo lsof -t /dev/accel* /dev/vfio/* 2> /dev/null)" ] || { echo "the TPU is in use"; exit 1; }
  [ -s "$D/segments-en.parquet" ] || JAX_PLATFORMS=cpu "$PY" -u "$C/gemma4_pilot.py" segments --pairs "$D/pairs.parquet" --tokenizer "$TOK" --out "$D"
  gsutil -q cp "$D/segments-en.parquet" "$D/segments-nl.parquet" "$O/data/"
  echo "== translate $(date -u +%T): en->nl on chips 0-3, nl->en on chips 4-7"
  direction 0 en nl "$D/segments-en.parquet" gemma4_en_nl & a=$!
  direction 1 nl en "$D/segments-nl.parquet" gemma4_nl_en & b=$!
  wait $a && ca=0 || ca=$?
  wait $b && cb=0 || cb=$?
  grep -h '^DONE\|^MAXTEXT' ~/logs/gemma4_en_nl.log ~/logs/gemma4_nl_en.log || true
  echo "== translate done $(date -u +%T): en->nl $ca, nl->en $cb"
  [ $ca = 0 ] && [ $cb = 0 ]
  ;;
post)
  [ -z "$(sudo lsof -t /dev/accel* /dev/vfio/* 2> /dev/null)" ] || { echo "the TPU is in use"; exit 1; }
  mkdir -p "$SHM/shards/en-nl" "$SHM/shards/nl-en"
  gsutil -m -q rsync -r "$O/en-nl" "$SHM/shards/en-nl"
  gsutil -m -q rsync -r "$O/nl-en" "$SHM/shards/nl-en"
  echo "== merge $(date -u +%T)"
  JAX_PLATFORMS=cpu "$PY" -u "$C/gemma4_pilot.py" merge --pairs "$D/pairs.parquet" --en-nl "$SHM/shards/en-nl" \
    --nl-en "$SHM/shards/nl-en" --out "$SHM/out/compare.parquet"
  gsutil -q cp "$SHM/out/compare.parquet" "$O/compare.parquet"
  [ -s "$D/context/chains.json" ] || JAX_PLATFORMS=cpu "$PY" -u "$C/gemma4_pilot.py" context-data --pairs "$D/pairs.parquet" \
    --handelingen "$D/handelingen" --nemotron "$D/nemotron/chat-ranges.jsonl" --tokenizer "$TOK" --out "$D/context"
  gsutil -q cp "$D/context/chains.json" "$O/context/chains.json"
  echo "== context trial (chips 0-3) and MetricX (chips 4-7) $(date -u +%T)"
  (
    half 0
    if [ ! -s "$SHM/out/context.parquet" ]; then
      "$PY" -u "$C/gemma4_pilot.py" context-run --data "$D/context" --tokenizer "$TOK" --ckpt "$CKPT" --out "$SHM/out"
    fi
  ) > ~/logs/gemma4_context.log 2>&1 & a=$!
  (
    half 1
    if [ ! -s "$SHM/out/qe.parquet" ]; then
      HF_HUB_OFFLINE=0 JAX_COMPILATION_CACHE_DIR=$SHM/jax_cache_metricx "$PY" -u "$C/metricx_qe.py" score \
        --pairs "$SHM/out/compare.parquet" --pair en:back --pair en:nl_g4 --pair nl:en --pair nl:en_g4 --model xxl \
        --root "$SHM/metricx" --out "$SHM/out/qe.parquet"
    fi
  ) > ~/logs/gemma4_qe.log 2>&1 & b=$!
  wait $a && ca=0 || ca=$?
  wait $b && cb=0 || cb=$?
  echo "EXIT $ca" >> ~/logs/gemma4_context.log
  echo "EXIT $cb" >> ~/logs/gemma4_qe.log
  gsutil -q cp ~/logs/gemma4_context.log ~/logs/gemma4_qe.log "$O/logs/"
  echo "== context $ca, MetricX $cb $(date -u +%T)"
  [ $ca = 0 ] && [ $cb = 0 ]
  gsutil -q cp "$SHM/out/context.parquet" "$SHM/out/qe.parquet" "$O/"
  mkdir -p "$SHM/out/logs" && cp ~/logs/gemma4_en_nl.log ~/logs/gemma4_nl_en.log "$SHM/out/logs/"
  "$PY" - "$SHM/out/setup.json" <<'EOF'
import json, sys, yaml
from pathlib import Path
fork = Path("/dev/shm/translate-generic/venv-fork/done.txt").read_text().strip()
slots = yaml.safe_load(Path("/dev/shm/translate-generic/code/translate_models.yml").read_text())["models"]["gemma4-31b"]["slots"]
Path(sys.argv[1]).write_text(json.dumps({
    "maxtext": "de fork yhavinga/maxtext, branch fix-nnx-decode-cache-abstract-init @ 2249474b9 (upstream main "
               "d1c0d0afe van 9 nov plus fix 1), zonder JetStream; omweg make_result_tokens_a_pytree actief, "
               "make_decode_cache_sharded niet nodig. Pakketten: " + fork,
    "vm": "translate-generic, v6e-8 spot in europe-west4-a, als twee TPU's van 4 chips (TPU_VISIBLE_CHIPS)",
    "slots": slots}, ensure_ascii=False))
EOF
  echo "== report $(date -u +%T)"
  "$PY" -u "$C/gemma4_pilot.py" report --compare "$SHM/out/compare.parquet" --qe "$SHM/out/qe.parquet" \
    --context "$SHM/out/context.parquet" --pids "$D/pairs30-pids.txt" --logs "$SHM/out/logs" \
    --setup "$SHM/out/setup.json" --out "$SHM/out/gemma4-vs-tg27b.md"
  gsutil -q cp "$SHM/out/gemma4-vs-tg27b.md" "$SHM/out/gemma4-vs-tg27b.json" "$SHM/out/setup.json" "$O/"
  echo "== post done $(date -u +%T)"
  ;;
*)
  echo "usage: run_gemma4_pilot.sh checks|translate|post"; exit 1 ;;
esac
