#!/bin/bash
# The starting point of the en->nl translator (hammond mbert/designs/en-nl-translator.md §4): TranslateGemma 27B, the
# base model, translates the four evaluation sets of evalsets.py; MetricX-24 hybrid XXL (with the reference and as QE,
# on the TPU) and COMET-22 (with the reference, on the CPU) score the translations. Nothing is trained. One stage per
# call, on the workers of a v4-32, each host a TPU of its own 4 chips; translator_baseline.sh (co1) starts the stages
# detached and waits for their EXIT lines.
#   bash run_baseline.sh setup      every worker: the venvs (FLAN-nl's for TranslateGemma, the translator's for MetricX
#                                   and the sets, COMET's) and the COMET model; all in /dev/shm
#   bash run_baseline.sh data       worker 0: evalsets.py fetch (from the sources, pinned), to $E/data/
#   bash run_baseline.sh translate  every worker: its length classes through run_translate.sh, which detaches (log
#                                   ~/logs/tbl_translate_$TAG.log ending in EXIT), shards to $B/translations/w<k>/
#   bash run_baseline.sh score      every worker: its pairs (evalsets.py collect), MetricX-24 (reference and QE),
#                                   COMET-22 (the first 100 segments timed on their own first), to $B/pairs/, $B/scores/
#   bash run_baseline.sh summary    worker 0: evalsets.py summary, to $B/ (translations, scores, summary.json/.md)
# The code is beside this script (a copy of $E/scripts/). A rerun of score skips the scores in $B/scores/.
# Environment: TAG (log names), COMMIT (named in the summary); for a dry run also SHM, W, E, B, TOK.
set -euo pipefail
STAGE=${1:?setup, data, translate, score or summary}
E=${E:-gs://t5-dutch-english-usc2/mbert/translator/eval}
B=${B:-$E/baseline-tg27b}
CKPT=gs://t5-dutch-english-usc2/mbert/flan-nl/translategemma-27b-maxtext
SHM=${SHM:-/dev/shm/translator}
L=$SHM/eval/run
PY=$SHM/venv/bin/python
CPY=$SHM/comet-venv/bin/python
TOK=${TOK:-/dev/shm/flan_nl/tokenizer-27b}
CODE=$(cd "$(dirname "$0")" && pwd)
W=${W:-$(hostname | sed 's/.*-w-//')}
TAG=${TAG:-manual}
COMMIT=${COMMIT:-unknown}
export HF_HUB_DISABLE_PROGRESS_BARS=1 HF_XET_CACHE=$SHM/xet-cache
mkdir -p "$L" ~/logs

case $STAGE in
setup)
  # systemd-logind may empty a user's /dev/shm at the last logout (seen on v6e, 9 Nov); lingering prevents that
  [ "$(loginctl show-user "$(id -un)" -p Linger --value 2> /dev/null)" = yes ] || sudo loginctl enable-linger "$(id -un)"
  bash "$CODE/setup_flan.sh"
  RESOURCES=0 bash "$CODE/setup.sh"
  bash "$CODE/setup_comet.sh"
  gsutil -q stat "$CKPT/done.txt" || { echo "w$W: $CKPT has no done.txt"; exit 1; }
  free=$(df -BG --output=avail "$HOME" | tail -1 | tr -dc 0-9)
  [ "$free" -ge 3 ] || { echo "w$W: only $free GB free on the boot disk"; exit 1; }
  echo "READY w$W: boot disk $free GB free, /dev/shm $(df -h /dev/shm | tail -1 | awk '{print $4}') free"
  ;;
data)
  "$PY" -I "$CODE/evalsets.py" fetch --out "$L/data" --tokenizer "$TOK"
  # hf_hub_download keeps its metadata in .cache/ beside the file; that stays here
  gsutil -m -q rsync -r -x '.*\.cache/.*' "$L/data" "$E/data"
  echo "DATA w$W: $E/data"
  ;;
translate)
  classes=$(gsutil cat "$E/data/classes-w$W.txt")
  [ -n "$classes" ] || { echo "w$W: no length classes in $E/data/classes-w$W.txt"; exit 1; }
  CLASSES="$classes" SOURCE=en TARGET=nl SEGMENTS="$E/data/segments-w{w}.parquet" OUT="$B/translations/w{w}" \
    NAME="tbl_translate_$TAG" LOGS="$B/logs" bash "$CODE/run_translate.sh"
  ;;
score)
  D=$L/data
  O=$L/out
  mkdir -p "$D" "$L/translations" "$O"
  gsutil -q cp "$E/data/evalsets.parquet" "$E/data/manifest.json" "$E/data/segments-w$W.parquet" "$D/"
  gsutil -m -q rsync -r "$B/translations/w$W" "$L/translations"
  "$PY" -I "$CODE/evalsets.py" collect --data "$D" --worker "$W" --shards "$L/translations" --out "$O/pairs-w$W.parquet"
  gsutil -q cp "$O/pairs-w$W.parquet" "$B/pairs/pairs-w$W.parquet"
  if gsutil -q stat "$B/scores/metricx-w$W.parquet"; then
    echo "w$W: MetricX scores are in $B/scores/ before"
  else
    [ -z "$(sudo lsof -t /dev/accel* /dev/vfio/* 2> /dev/null)" ] || { echo "w$W: the TPU is in use"; exit 1; }
    (
      # This host as a TPU of its own (as in run_translate.sh): without these JAX takes the slice from the metadata
      # and waits for the other hosts. XLA_FLAGS set and the TPU logs in /dev/shm keep the boot disk free.
      export TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 TPU_PROCESS_BOUNDS=1,1,1 TPU_VISIBLE_CHIPS=0,1,2,3 TPU_PROCESS_PORT=8476
      export TPU_PROCESS_ADDRESSES=localhost:8476 CLOUD_TPU_TASK_ID=0 ALLOW_MULTIPLE_LIBTPU_LOAD=1
      export JAX_COMPILATION_CACHE_DIR=$SHM/jax_cache_metricx XLA_FLAGS=--xla_dump_hlo_as_text=false TPU_LOG_DIR=$SHM/tpu_logs
      timeout 600 "$PY" -c "import jax; assert (jax.device_count(), jax.process_count()) == (4, 1), (jax.device_count(), jax.process_count())"
      "$PY" -I -u "$CODE/metricx_qe.py" score --pairs "$O/pairs-w$W.parquet" --pair source:hypothesis:reference \
        --pair source:hypothesis --model xxl --root "$SHM/metricx" --empty-candidates --out "$O/metricx-w$W.parquet"
    )
    gsutil -q cp "$O/metricx-w$W.parquet" "$B/scores/metricx-w$W.parquet"
  fi
  if gsutil -q stat "$B/scores/comet-w$W.parquet"; then
    echo "w$W: COMET scores are in $B/scores/ before"
  else
    "$CPY" -I -u "$CODE/comet_score.py" score --root "$SHM/comet" --pairs "$O/pairs-w$W.parquet" \
      --out "$O/comet-w$W.parquet" --time-first 100 --timing "$O/comet-timing-w$W.json"
    gsutil -q cp "$O/comet-timing-w$W.json" "$B/scores/comet-timing-w$W.json"
    gsutil -q cp "$O/comet-w$W.parquet" "$B/scores/comet-w$W.parquet"
  fi
  echo "SCORED w$W"
  ;;
summary)
  A=$L/all
  mkdir -p "$A" "$L/data"
  gsutil -m -q cp "$B/pairs/pairs-w*.parquet" "$B/scores/*" "$E/data/evalsets.parquet" "$E/data/manifest.json" "$A/"
  cp "$A/evalsets.parquet" "$A/manifest.json" "$L/data/"
  converted=$(gsutil cat "$CKPT/done.txt")
  cat > "$L/setup.json" <<EOF
{"vertaler": "TranslateGemma 27B, het basismodel, zonder fine-tune: $converted; checkpoint $CKPT",
 "decodering": "greedy, de prompt van het chat-template van het model (en→nl), translate.py met zijn lengteklassen (MaxText 0.2.5, venv van FLAN-nl); per worker hele lengteklassen; segmenten boven 512 brontokens gemarkeerd (over_512)",
 "MetricX-24": "google/metricx-24-hybrid-xxl-v2p6-bfloat16 @ 3d9c6f7fd11b58d3eceeb68ee39fe7d2c761ee20, eigen JAX-port (metricx_qe.py), bf16-producten op de TPU; met referentie 'source: … candidate: … reference: …', QE 'source: … candidate: …', invoer tot 1.536 mT5-tokens; een lege vertaling krijgt ook een score",
 "COMET-22": "Unbabel/wmt22-comet-da @ 2760a223ac957f30acfb18c8aa649b01cf1d75f2 (encoder FacebookAI/xlm-roberta-large @ c23d21b0620b635a76227c604d44e43a9f0ee389), unbabel-comet 2.2.7, torch 2.9.1 op de CPU, met referentie",
 "niet gebruikt": "CometKiwi en xCOMET (CC BY-NC-SA)",
 "code": "github.com/yhavinga/mbert-jax, branch en-nl-eval @ $COMMIT; kopie in $E/scripts/",
 "data": "$E/data/ (manifest.json: bronnen, revisies, sha256, licenties)",
 "uitvoer": "$B/"}
EOF
  "$PY" -I "$CODE/evalsets.py" summary --data "$L/data" --pairs "$A"/pairs-w*.parquet --metricx "$A"/metricx-w*.parquet \
    --comet "$A"/comet-w*.parquet --comet-timing "$A"/comet-timing-w*.json --setup "$(cat "$L/setup.json")" \
    --model "TranslateGemma 27B (basismodel)" --out "$L/summary"
  gsutil -m -q cp "$L/summary/*" "$B/"
  echo "SUMMARY w$W: $B/summary.md"
  ;;
*)
  echo "unknown stage $STAGE"
  exit 2
  ;;
esac
