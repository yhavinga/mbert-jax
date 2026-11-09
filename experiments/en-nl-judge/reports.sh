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
gsutil -m -q cp "$R/*" "$JUDGE/reports/"
echo "reports in $JUDGE/reports/: $(ls "$R" | tr '\n' ' ')"
