#!/bin/bash
# On a worker of mbert-mix, also while another run trains there: one step of the data of
# culturax-gptnl-nl-230m-chunk1024 (protocol.md, commands.txt), with experiments/gptnl/gptnl_mix.py of a checkout of
# its own, ~/mbert-gptnl at COMMIT, run with the venv of ~/mbert-jax (the trainer's, which it leaves alone). CPU only,
# at nice 19 and in the idle I/O class; it imports no jax, so it opens no TPU. Its temporary files are in
# /dev/shm/gptnl_work.
#   count  every worker: its part of the included GPT-NL files counted, the lengths parquets to $G/lengths
#   plan   worker 0: the plan from all lengths and CulturaX's counts of token-counts; the selections and the plan to
#          $G/plan, the plan to be committed as experiments/gptnl/cxgptnl_plan.json
#   mix    every worker, at a COMMIT with that plan: this host's share in ~/cxgptnl_nl, listed in
#          ~/cxgptnl_nl_files.txt; its summary to $G/shares/cxgptnl-w<N>.json
# Detached: the log is ~/logs/gptnl_<step>.log, which ends in "EXIT <code>" and is copied to $G/logs/. A second start
# continues. Needs ~/.cache/huggingface/token. Environment overrides: WORKERS (count: files counted at a time; mix:
# share files written at a time), READERS (mix: source files read at a time), MAX_FILES (count, for a smoke test: only
# the first files to do, log ~/logs/gptnl_count_smoke.log).
set -euo pipefail
STEP=${1:?usage: prepare_gptnl.sh count, plan or mix}
COMMIT=${COMMIT:?set COMMIT to the commit of commands.txt}
RANK=${RANK:-$(hostname | sed 's/.*-w-//')}
MAX_FILES=${MAX_FILES:-}
G=gs://t5-dutch-english-usc2/mbert/gptnl
TC=gs://t5-dutch-english-usc2/mbert/token-counts
WORK=/dev/shm/gptnl_work
LOG=$HOME/logs/gptnl_$STEP${MAX_FILES:+_smoke}.log
case $STEP in
  count) WORKERS=${WORKERS:-16} ;;
  plan) ;;
  mix) WORKERS=${WORKERS:-4}; READERS=${READERS:-8} ;;
  *) echo "the step must be count, plan or mix, not $STEP"; exit 1 ;;
esac
[[ $RANK =~ ^[0-3]$ ]] || { echo "$(hostname): worker number '$RANK' is not one of 0 to 3"; exit 1; }
[ "$STEP" != plan ] || [ "$RANK" = 0 ] || { echo "$(hostname): the plan is made on worker 0"; exit 1; }
[ -s ~/.cache/huggingface/token ] || { echo "$(hostname): no ~/.cache/huggingface/token, put it there first"; exit 1; }
[ -x ~/mbert-jax/.venv/bin/python ] || { echo "$(hostname): no venv in ~/mbert-jax"; exit 1; }
if [ -z "${GPTNL_DETACHED:-}" ]; then
  pgrep -f "[g]ptnl_mix.py $STEP" > /dev/null && { echo "$(hostname): gptnl_mix.py $STEP runs already"; exit 1; }
  mkdir -p ~/logs
  GPTNL_DETACHED=1 setsid nohup nice -n 19 ionice -c3 bash -c "bash $0 $STEP > $LOG 2>&1; echo EXIT \$? >> $LOG; gsutil -q cp $LOG $G/logs/$(basename "$LOG" .log)-w$RANK.log" > /dev/null 2>&1 < /dev/null &
  echo "$(hostname): started, $LOG"
  exit 0
fi

cd "$HOME"
[ -d mbert-gptnl/.git ] || git clone -q https://github.com/yhavinga/mbert-jax.git mbert-gptnl
git -C mbert-gptnl fetch -q origin gptnl-mix
git -C mbert-gptnl checkout -q "$COMMIT"
cd mbert-gptnl
mkdir -p $WORK
# No .pyc files in the trainer's venv; the tokenizer on one core per process
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false RAYON_NUM_THREADS=1 HF_HUB_DISABLE_PROGRESS_BARS=1
PY=("$HOME/mbert-jax/.venv/bin/python" -u experiments/gptnl/gptnl_mix.py)
echo "$(hostname): $STEP at $(git rev-parse --short HEAD), $(df -h ~ | tail -1 | awk '{print $4}') free on the disk, $(df -h /dev/shm | tail -1 | awk '{print $4}') in /dev/shm"
case $STEP in
  count)
    # The token counts of count_tokens.py of one CulturaX file, against which the tokenization is checked
    gsutil -q cp $TC/culturax/lengths/nl_part_00000.parquet $WORK/
    "${PY[@]}" count --rank "$RANK" --work $WORK/lengths --gcs $G/lengths --check-lengths $WORK/nl_part_00000.parquet \
      --workers "$WORKERS" ${MAX_FILES:+--max-files $MAX_FILES}
    ;;
  plan)
    mkdir -p $WORK/gptnl_lengths $WORK/culturax_counts
    gsutil -m -q rsync $G/lengths $WORK/gptnl_lengths
    gsutil -m -q rsync $TC/culturax/counts $WORK/culturax_counts
    # The trainer's held-out file, as setup_fineweb2.sh checked it: its documents are left out of the shares
    [ "$(md5sum ~/heldout.parquet | cut -d' ' -f1)" = b2f579a876d06eef40c94270537bb7af ] || { echo "$(hostname): ~/heldout.parquet is not the long run's"; exit 1; }
    "${PY[@]}" plan --gptnl-lengths $WORK/gptnl_lengths --culturax-counts $WORK/culturax_counts --heldout ~/heldout.parquet \
      --out-dir $WORK/plan --plan $WORK/plan/cxgptnl_plan.json --seed 42
    gsutil -q cp $WORK/plan/cxgptnl-w0.npz $WORK/plan/cxgptnl-w1.npz $WORK/plan/cxgptnl-w2.npz $WORK/plan/cxgptnl-w3.npz \
      $WORK/plan/cxgptnl_plan.json $G/plan/
    ;;
  mix)
    gsutil -q cp $G/plan/cxgptnl-w$RANK.npz $WORK/
    "${PY[@]}" mix --rank "$RANK" --selection $WORK/cxgptnl-w$RANK.npz --spill $WORK/spill --out ~/cxgptnl_nl \
      --list ~/cxgptnl_nl_files.txt --readers "$READERS" --workers "$WORKERS"
    [ "$(wc -l < ~/cxgptnl_nl_files.txt)" = 64 ] || { echo "$(hostname): ~/cxgptnl_nl_files.txt does not list 64 files"; exit 1; }
    while read -r f; do [ -s "$f" ] || { echo "$(hostname): missing $f"; exit 1; }; done < ~/cxgptnl_nl_files.txt
    gsutil -q cp ~/cxgptnl_nl/share.json $G/shares/cxgptnl-w$RANK.json
    echo "$(hostname): ready, $(du -sh ~/cxgptnl_nl | cut -f1) of data, $(df -h ~ | tail -1 | awk '{print $4}') free"
    ;;
esac
