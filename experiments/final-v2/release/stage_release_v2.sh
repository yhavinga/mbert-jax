#!/bin/bash
# The Hugging Face export of the Dutch ModernBERT 230M trained on the opt-out-filtered FineWeb-2-nl (run
# v2-fw2optout-230m, checkpoint 1,306,000) on one worker of a v4-32 in us-central2-b, on its CPU only, by the route of
# T2's export (experiments/final/release/stage_release.sh); only this run's values are set here. Started detached from
# co1, with the code at COMMIT in $W/code (W defaults to ~/hf_release_v2):
#   COMMIT=<hf-release-v2 commit> PHASE=facts bash experiments/final-v2/release/stage_release_v2.sh
#   COMMIT=<hf-release-v2 commit> PHASE=local bash experiments/final-v2/release/stage_release_v2.sh
#   COMMIT=<commit with the filled-in card> PHASE=stage bash experiments/final-v2/release/stage_release_v2.sh
# facts  the training facts of the model card (card_facts.py), from the data state of every process at 1,306,000 and
#        the trainer logs of the run in the bucket (same region): $W/reports/facts.json and $OUT/reports/
# local  stage_release.sh local on CKPT: the params read in place, JAX reference logits, convert_flax_to_hf.py, the
#        checks of check_hf_export.py; then every tokenizer and config file of the export must equal that of PUBLIC
#        at PUBLIC_REV, the published model with the same tokenizer and architecture, converted by the same code
# stage  stage_release.sh stage: CARD (no TODO( left) as README.md, push_staging.py into the private REPO (created
#        when it does not exist: run this only on Yeb's word), the checks from the Hub with transformers 4.57.1 and 5.23.0
# Nothing is deleted outside $W; $W is removed by hand afterwards with its literal path.
set -euo pipefail
PHASE=${PHASE:?set PHASE to facts, local or stage}
COMMIT=${COMMIT:?set COMMIT to the hf-release-v2 commit}
M=gs://t5-dutch-english-usc2/mbert
RUN=v2-fw2optout-230m
LAST_STEP=1306000
PARAMETERS=230110592
export CKPT=${CKPT:-$M/$RUN/ckpt_keep/$LAST_STEP}
export OUT=${OUT:-$M/final-v2/hf-release}
export REPO=${REPO:-yhavinga/dmbert-230m-fineweb2-optout-staging}
export PUBLIC=yhavinga/dmbert-230m-culturax
export PUBLIC_REV=6f9372044114c3a99815442bd2c253a70db85a79
export CARD=${CARD:-experiments/final-v2/release/MODEL_CARD.md}
export W=${W:-$HOME/hf_release_v2}
export JAXPY=${JAXPY:-$HOME/mbert-dumb/.venv/bin/python}
export UV=${UV:-$HOME/.local/bin/uv}
C=$W/code
cd "$C"
[ "$(git rev-parse HEAD)" = "$COMMIT" ] || { echo "$C is not at $COMMIT"; exit 2; }
mkdir -p "$W/reports"
log() { echo "$(date '+%F %T') $*"; }
need() { [ -e "$1" ] || { echo "missing: $1 ($2)"; exit 2; }; }

gsutil -q stat "$M/$RUN/ckpt_keep/$LAST_STEP/commit_success.txt" || { echo "no committed $M/$RUN/ckpt_keep/$LAST_STEP: the run is not done"; exit 2; }
if [ "$PHASE" != facts ]; then
  need "$JAXPY" "the JAX venv of the evaluations"
  need "$UV" "uv, for the torch venvs"
  # PUBLIC is gated: its files need a token with access (the owner's); never printed
  [ -s "$HOME/.cache/huggingface/token" ] || [ -n "${HF_TOKEN:-}" ] || { echo "no Hugging Face token on $(hostname): put it via stdin into ~/.cache/huggingface/token"; exit 2; }
  free_gb=$(df -BG --output=avail "$(dirname "$W")" | tail -1 | tr -dc 0-9)
  [ "$free_gb" -ge 10 ] || { echo "only $free_gb GB free for $W (needs ~6 GB, 10 asked)"; exit 2; }
fi

case $PHASE in
facts)
  d=$W/facts
  mkdir -p "$d"
  log "data state at $LAST_STEP and the trainer logs of $RUN"
  gsutil -q cp "$M/$RUN/ckpt_data_state/$LAST_STEP/process_*.json" "$d/"
  for w in 0 1 2 3; do gsutil -q cp "$M/$RUN/logs/w$w.log" "$d/w$w.log"; done
  nice -n 19 python3 -I experiments/final-v2/release/card_facts.py --data-state "$d"/process_*.json \
    --trainer-logs "$d"/w0.log "$d"/w1.log "$d"/w2.log "$d"/w3.log --last-step $LAST_STEP --parameters $PARAMETERS \
    --out "$W/reports/facts.json"
  gsutil -q cp "$W/reports/facts.json" "$OUT/reports/facts.json"
  log "facts done: $OUT/reports/facts.json"
  ;;
local)
  rc=0
  bash experiments/final/release/stage_release.sh || rc=$?
  log "tokenizer and config files of the export against $PUBLIC @ $PUBLIC_REV: all must be equal"
  python3 -I - "$W/reports/files-vs-public.json" <<'PY' || rc=1
import json, sys
res = json.load(open(sys.argv[1]))
different = sorted(f for f, r in res.items() if not r["equal"])
print("all equal: " + ", ".join(sorted(res)) if not different else f"FAILED: different from the published files: {different}")
sys.exit(1 if different else 0)
PY
  log "local done, exit $rc"
  exit $rc
  ;;
stage)
  bash experiments/final/release/stage_release.sh
  ;;
*)
  echo "PHASE must be facts, local or stage"; exit 2
  ;;
esac
