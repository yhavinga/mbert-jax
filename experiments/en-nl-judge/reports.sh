#!/bin/bash
# On the judge's VM (CPU), after the scores are complete: the sanity report, the comparison with transformers and
# europarl-scan.md, from the scores in the bucket of the VM's region -> $JUDGE/reports/. run.sh ends with it.
set -euo pipefail
ZONE=$(curl -fs -H Metadata-Flavor:Google http://metadata.google.internal/computeMetadata/v1/instance/zone | awk -F/ '{print $NF}')
case ${ZONE%-*} in
  us-east1) BUCKET=gs://t5-dutch-english-use1 ;;
  europe-west4) BUCKET=gs://t5-dutch-english ;;
  *) echo "no judge bucket for zone '$ZONE'"; exit 1 ;;
esac
JUDGE=$BUCKET/mbert/translator/judge
SHM=/dev/shm/judge
CODE=$SHM/code
PY=$SHM/venv/bin/python
R=$SHM/reports
mkdir -p "$R" "$SHM/scores/sanity" "$SHM/scores/motra"
gsutil -m -q rsync -r "$JUDGE/scores/sanity" "$SHM/scores/sanity"
gsutil -m -q rsync -r "$JUDGE/scores/motra" "$SHM/scores/motra"
"$PY" "$CODE/report.py" sanity --scores "$SHM/scores/sanity" --out "$R"
COMPARE=()
if gsutil -q stat "$JUDGE/reference/ref.json"; then
  gsutil -q cp "$JUDGE/reference/ref.json" "$SHM/ref.json"
  "$PY" "$CODE/report.py" compare --reference "$SHM/ref.json" --scores "$SHM/scores/sanity" --out "$R/compare.md"
  COMPARE=(--compare-md "$R/compare.md")
fi
"$PY" "$CODE/report.py" scan --pairs "$SHM/motra" --scores "$SHM/scores/motra" --manifest "$SHM/motra/manifest.json" \
  --sanity-md "$R/sanity.md" "${COMPARE[@]}" --out "$R/europarl-scan.md"
if [ -d "$SHM/pilot" ] && gsutil -q stat "$JUDGE/data/pilot-source/pairs30_ids.txt"; then
  mkdir -p "$SHM/scores/pilot"
  gsutil -m -q rsync -r "$JUDGE/scores/pilot" "$SHM/scores/pilot"
  gsutil -q cp "$JUDGE/data/pilot-source/pairs30_ids.txt" "$SHM/pairs30_ids.txt"
  "$PY" "$CODE/report.py" pilot --pairs "$SHM/pilot" --scores "$SHM/scores/pilot" --ids "$SHM/pairs30_ids.txt" \
    --out "$R/pilot-scores.md"
  if gsutil -q ls "$JUDGE/scores/pilot-source/" > /dev/null 2>&1; then
    mkdir -p "$SHM/scores/pilot-source"
    gsutil -m -q rsync -r "$JUDGE/scores/pilot-source" "$SHM/scores/pilot-source"
    "$PY" "$CODE/report.py" source --pairs "$SHM/pilot" --scores "$SHM/scores/pilot-source" \
      --pair-scores "$SHM/scores/pilot" --ids "$SHM/pairs30_ids.txt" --out "$R/source-quality.md" \
      --case "raad:e9ec1e4c-c4f5-522a-8af1-285a6fc2c680:d:34"
  fi
fi
gsutil -m -q cp "$R/*" "$JUDGE/reports/"
echo "reports in $JUDGE/reports/: $(ls "$R" | tr '\n' ' ')"
