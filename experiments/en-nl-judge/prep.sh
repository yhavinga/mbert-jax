#!/bin/bash
# On the judge's VM, everything before TPU work, each step skipped when its result is in the bucket of the VM's region:
# the environment (setup.sh), the MoTra en->nl pairs (data.py motra), the sanity set (data.py sanity), the MaxText
# checkpoint (setup.sh convert). The pairs and the sanity set go to the bucket at once, so a new spot VM copies them.
# Start it detached, with its log ending in an EXIT line:
#   setsid nohup bash -c 'bash /dev/shm/judge/code/prep.sh > ~/logs/judge_prep.log 2>&1; echo EXIT $? >> ~/logs/judge_prep.log' > /dev/null 2>&1 < /dev/null &
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
MOTRA_REPO=https://github.com/UDS-SFB-B6-Datasets/Multilingual-Parallel-Direct-Europarl

echo "== setup $(date -u +%T)"
bash "$CODE/setup.sh"

echo "== motra $(date -u +%T)"
if gsutil -q stat "$JUDGE/data/motra/manifest.json"; then
  mkdir -p "$SHM/motra" && gsutil -m -q cp "$JUDGE/data/motra/*" "$SHM/motra/"
else
  # Straight from GitHub to this VM: only the English and Dutch sittings and the scripts
  if [ ! -d "$SHM/motra-repo/.git" ]; then
    git clone -q --depth 1 --filter=blob:none --sparse "$MOTRA_REPO" "$SHM/motra-repo"
    git -C "$SHM/motra-repo" sparse-checkout set proceedings/xml/en proceedings/xml/nl scripts
  fi
  "$PY" -u "$CODE/data.py" motra --repo "$SHM/motra-repo" --out "$SHM/motra" --processes 64 \
    --commit "$(git -C "$SHM/motra-repo" rev-parse HEAD)"
  gsutil -m -q cp "$SHM/motra/shard-*.parquet" "$JUDGE/data/motra/"
  gsutil -q cp "$SHM/motra/manifest.json" "$JUDGE/data/motra/manifest.json"
fi
ls "$SHM/motra" | wc -l

echo "== sanity $(date -u +%T)"
if gsutil -q stat "$JUDGE/data/sanity/manifest.json"; then
  mkdir -p "$SHM/sanity" && gsutil -m -q cp "$JUDGE/data/sanity/*" "$SHM/sanity/"
else
  "$PY" -u "$CODE/data.py" sanity --pairs "$SHM/motra" --out "$SHM/sanity"
  gsutil -q cp "$SHM/sanity/shard-00000.parquet" "$JUDGE/data/sanity/shard-00000.parquet"
  gsutil -q cp "$SHM/sanity/manifest.json" "$JUDGE/data/sanity/manifest.json"
fi

echo "== checkpoint $(date -u +%T)"
bash "$CODE/setup.sh" convert
echo "== prep done $(date -u +%T)"
