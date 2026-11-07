#!/bin/bash
# On every worker of mbert-mix, once, also while fineweb2-nl-230m-chunk1024 trains there: this host's share of FineWeb-2
# nl rehydrated for fineweb2-nl-230m-chunk1024-rh3 (protocol.md). scripts/stage_share.py rehydrate reads the share that
# setup_fineweb2.sh staged (~/fineweb2_nl), the cluster sizes of its rows from the Hugging Face files and the token
# counts of count_tokens.py, and writes the kept rows to ~/fineweb2_nl_rh3, listed in ~/fineweb2_nl_rh3_files.txt.
# CPU only, at nice 19 and in the idle I/O class: the code is a checkout of its own, ~/mbert-rh at COMMIT, run with the
# venv of ~/mbert-jax (the trainer's, which it leaves alone); it imports no jax, so it opens no TPU.
# Detached: the log is ~/logs/rehydrate_fineweb2.log, which ends in "EXIT <code>". A second start keeps the files written.
# Needs ~/.cache/huggingface/token on the worker. Environment overrides: READERS and WORKERS, source files read and
# kept rows written at a time; MAX_FILES, for the smoke test only: the first files of the plan, log
# ~/logs/rehydrate_fineweb2_smoke.log, and no list, no share.json and no upload.
set -euo pipefail
COMMIT=${COMMIT:?set COMMIT to the commit of commands.txt}
RANK=${RANK:-$(hostname | sed 's/.*-w-//')}
READERS=${READERS:-16}
WORKERS=${WORKERS:-4}
MAX_FILES=${MAX_FILES:-}
LOG=$HOME/logs/rehydrate_fineweb2${MAX_FILES:+_smoke}.log
SHARES=gs://t5-dutch-english-usc2/mbert/rehydration/shares
[[ $RANK =~ ^[0-3]$ ]] || { echo "$(hostname): worker number '$RANK' is not one of 0 to 3"; exit 1; }
[ -s ~/.cache/huggingface/token ] || { echo "$(hostname): no ~/.cache/huggingface/token, put it there first"; exit 1; }
[ -f ~/fineweb2_nl_files.txt ] && [ -x ~/mbert-jax/.venv/bin/python ] || { echo "$(hostname): run setup_fineweb2.sh first"; exit 1; }
if [ -z "${REHYDRATE_DETACHED:-}" ]; then
  pgrep -f '[s]tage_share.py rehydrate' > /dev/null && { echo "$(hostname): a rehydration runs already"; exit 1; }
  mkdir -p ~/logs
  REHYDRATE_DETACHED=1 setsid nohup nice -n 19 ionice -c3 bash -c "bash $0 > $LOG 2>&1; echo EXIT \$? >> $LOG" > /dev/null 2>&1 < /dev/null &
  echo "$(hostname): started, $LOG"
  exit 0
fi

cd "$HOME"
[ -d mbert-rh/.git ] || git clone -q https://github.com/yhavinga/mbert-jax.git mbert-rh
git -C mbert-rh fetch -q origin rehydration
git -C mbert-rh checkout -q "$COMMIT"
# The tokens per document the plan was made of, 244 MB, from the bucket of this region
mkdir -p tc_fineweb2/lengths
gsutil -m -q rsync gs://t5-dutch-english-usc2/mbert/token-counts/fineweb2/lengths tc_fineweb2/lengths
cd mbert-rh
HF_HUB_DISABLE_PROGRESS_BARS=1 ~/mbert-jax/.venv/bin/python -u scripts/stage_share.py rehydrate --corpus fineweb2 \
  --rank "$RANK" --staged ~/fineweb2_nl --out ~/fineweb2_nl_rh3 --list ~/fineweb2_nl_rh3_files.txt \
  --lengths ~/tc_fineweb2/lengths --max-reps 3 --seed 42 --readers "$READERS" --workers "$WORKERS" ${MAX_FILES:+--max-files $MAX_FILES}
if [ -n "$MAX_FILES" ]; then
  echo "$(hostname): smoke test of $MAX_FILES files done, $(df -h ~ | tail -1 | awk '{print $4}') free"
  exit 0
fi
[ "$(wc -l < ~/fineweb2_nl_rh3_files.txt)" = 45 ] || { echo "$(hostname): ~/fineweb2_nl_rh3_files.txt does not list 45 files"; exit 1; }
while read -r f; do [ -s "$f" ] || { echo "$(hostname): missing $f"; exit 1; }; done < ~/fineweb2_nl_rh3_files.txt
gsutil -q cp ~/fineweb2_nl_rh3/share.json "$SHARES/fineweb2-rh3-w$RANK.json"
echo "$(hostname): ready at $(git rev-parse --short HEAD), $(du -sh ~/fineweb2_nl_rh3 | cut -f1) of data, $(df -h ~ | tail -1 | awk '{print $4}') free"
