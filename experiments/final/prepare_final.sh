#!/bin/bash
# On a worker of mbert-corpus: one step of the data and code of the final run (protocol.md, commands.txt). CPU only:
# nothing here imports JAX or opens the TPU, so setup, scan and plan may run next to other work at nice 19.
#   setup  every worker: ~/mbert-final at COMMIT; the venv PY (~/mbert-jax/.venv) checked against vm.lock, the
#          held-out file's md5, ~/culturax_nl_files.txt with S's 32 files of this worker
#   scan   every worker: blocklist_scan.py over ~/culturax_nl_files.txt, the rows of www.hebban.nl and other
#          subdomains of hebban.nl, to $F/blocklist/culturax-w<N>.json (done 9 Nov 23:36 at 25def26)
#   check  every worker, variant culturax: the staged files are those of the scan, with their rows and a column url;
#          the trainer leaves the scanned rows out itself (--url-blocklist hebban.nl)
#   plan   worker 0, variant culturax+gptnl: gptnl_mix.py plan for 653,000 steps, CulturaX's blocked rows passed
#          over; selections and plan to $F/plan/, the plan to be committed as experiments/final/final_plan.json
#   mix    every worker, variant culturax+gptnl, at a COMMIT with that plan: this host's share in ~/final_cxgptnl
#          (128 files), listed in ~/final_cxgptnl_files.txt, its summary to $F/shares/; reads from Hugging Face
# Detached: the log is ~/logs/final_<step>.log, ending in "EXIT <code>", copied to $F/logs/. Environment: COMMIT
# (required), PY (default ~/mbert-jax/.venv/bin/python), READERS and WORKERS (mix).
set -euo pipefail
STEP=${1:?usage: prepare_final.sh setup, scan, check, plan or mix}
COMMIT=${COMMIT:?set COMMIT to the commit of commands.txt}
RANK=${RANK:-$(hostname | sed 's/.*-w-//')}
PY=${PY:-$HOME/mbert-jax/.venv/bin/python}
F=gs://t5-dutch-english-usc2/mbert/final
G=gs://t5-dutch-english-usc2/mbert/gptnl
TC=gs://t5-dutch-english-usc2/mbert/token-counts
LOCK=gs://t5-dutch-english-usc2/mbert/corpus-runs/scripts/vm.lock
WORK=/dev/shm/final_work
LOG=$HOME/logs/final_$STEP.log
case $STEP in setup|scan|check|plan|mix) ;; *) echo "the step must be setup, scan, check, plan or mix, not $STEP"; exit 1 ;; esac
[[ $RANK =~ ^[0-3]$ ]] || { echo "$(hostname): worker number '$RANK' is not one of 0 to 3"; exit 1; }
[ "$STEP" != plan ] || [ "$RANK" = 0 ] || { echo "$(hostname): the plan is made on worker 0"; exit 1; }
if [ -z "${FINAL_DETACHED:-}" ]; then
  # The detached wrapper's command line; the ssh shell that runs this script has another
  pgrep -f "logs/final_${STEP}[.]log 2>&1; echo EXIT" > /dev/null && { echo "$(hostname): prepare_final.sh $STEP runs already"; exit 1; }
  mkdir -p ~/logs
  FINAL_DETACHED=1 setsid nohup nice -n 19 ionice -c3 bash -c "bash $0 $STEP > $LOG 2>&1; echo EXIT \$? >> $LOG; gsutil -q cp $LOG $F/logs/final_$STEP-w$RANK.log" > /dev/null 2>&1 < /dev/null &
  echo "$(hostname): started, $LOG"
  exit 0
fi

cd "$HOME"
[ -d mbert-final/.git ] || git clone -q https://github.com/yhavinga/mbert-jax.git mbert-final
git -C mbert-final fetch -q origin final-run
git -C mbert-final checkout -q "$COMMIT"
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false RAYON_NUM_THREADS=1 HF_HUB_DISABLE_PROGRESS_BARS=1
echo "$(hostname): $STEP at $(git -C mbert-final rev-parse --short HEAD), $(df -h ~ | tail -1 | awk '{print $4}') free on the disk, $(df -h /dev/shm | tail -1 | awk '{print $4}') in /dev/shm"
scan_json=$HOME/final_blocklist_w$RANK.json
case $STEP in
  setup)
    [ -x "$PY" ] || { echo "$(hostname): no $PY"; exit 1; }
    gsutil -q cp $LOCK /tmp/final_vm.lock
    # The packages that decide what the trainer computes, at the pins of the corpus runs' lock
    "$PY" - /tmp/final_vm.lock <<'EOF'
import importlib.metadata as m, sys
pins = dict(line.strip().split("==", 1) for line in open(sys.argv[1]) if "==" in line and not line.startswith("#"))
names = ["jax", "jaxlib", "libtpu", "flax", "optax", "orbax-checkpoint", "grain", "datasets", "transformers", "tokenizers",
         "numpy", "pyarrow", "xxhash"]
wrong = {n: (m.version(n), pins.get(n)) for n in names if m.version(n) != pins.get(n)}
print({n: m.version(n) for n in names})
if wrong:
    sys.exit(f"the venv differs from vm.lock: {wrong}")
EOF
    [ "$(md5sum ~/heldout.parquet | cut -d' ' -f1)" = b2f579a876d06eef40c94270537bb7af ] || { echo "$(hostname): ~/heldout.parquet is not the long run's"; exit 1; }
    [ "$(wc -l < ~/culturax_nl_files.txt)" = 32 ] || { echo "$(hostname): ~/culturax_nl_files.txt does not list 32 files"; exit 1; }
    [ -s ~/.cache/huggingface/token ] || { echo "$(hostname): no ~/.cache/huggingface/token (needed by mix)"; exit 1; }
    echo "$(hostname): setup ready"
    ;;
  scan)
    "$PY" -u mbert-final/experiments/final/blocklist_scan.py --files-list ~/culturax_nl_files.txt --blocklist hebban.nl --out "$scan_json"
    gsutil -q cp "$scan_json" $F/blocklist/culturax-w$RANK.json
    ;;
  check)
    gsutil -q cp $F/blocklist/culturax-w$RANK.json "$scan_json"
    "$PY" - "$scan_json" ~/culturax_nl_files.txt <<'EOF'
import json, sys
import pyarrow.parquet as pq
scan, files = json.load(open(sys.argv[1])), open(sys.argv[2]).read().split()
if [f["path"] for f in scan["files"]] != files:
    sys.exit("the staged files are not those of the blocklist scan, or in another order")
for f in scan["files"]:
    meta = pq.ParquetFile(f["path"])
    if meta.metadata.num_rows != f["rows"] or "url" not in meta.schema_arrow.names:
        sys.exit(f"{f['path']}: {meta.metadata.num_rows} rows and columns {meta.schema_arrow.names}, the scan {f['rows']} rows")
print(f"{len(files)} files, {scan['documents']:,} documents, {scan['blocked']:,} of {scan['blocklist']} left out by the trainer")
EOF
    ;;
  plan)
    mkdir -p $WORK/gptnl_lengths $WORK/culturax_counts $WORK/scans $WORK/plan
    gsutil -m -q rsync $G/lengths $WORK/gptnl_lengths
    gsutil -m -q rsync $TC/culturax/counts $WORK/culturax_counts
    gsutil -m -q cp "$F/blocklist/culturax-w*.json" $WORK/scans/
    [ "$(md5sum ~/heldout.parquet | cut -d' ' -f1)" = b2f579a876d06eef40c94270537bb7af ] || { echo "$(hostname): ~/heldout.parquet is not the long run's"; exit 1; }
    cd mbert-final
    "$PY" -u experiments/gptnl/gptnl_mix.py plan --gptnl-lengths $WORK/gptnl_lengths --culturax-counts $WORK/culturax_counts \
      --heldout ~/heldout.parquet --out-dir $WORK/plan --plan $WORK/plan/final_plan.json --seed 42 --steps 653000 \
      --share-files 128 --run final-cxgptnl-230m --name final-cxgptnl --exclude-culturax $WORK/scans/culturax-w*.json
    gsutil -q cp $WORK/plan/final-cxgptnl-w0.npz $WORK/plan/final-cxgptnl-w1.npz $WORK/plan/final-cxgptnl-w2.npz \
      $WORK/plan/final-cxgptnl-w3.npz $WORK/plan/final_plan.json $F/plan/
    ;;
  mix)
    [ -s ~/.cache/huggingface/token ] || { echo "$(hostname): no ~/.cache/huggingface/token"; exit 1; }
    [ -f mbert-final/experiments/final/final_plan.json ] || { echo "$(hostname): COMMIT $COMMIT has no final_plan.json"; exit 1; }
    mkdir -p $WORK
    gsutil -q cp $F/plan/final-cxgptnl-w$RANK.npz $WORK/
    cd mbert-final
    "$PY" -u experiments/gptnl/gptnl_mix.py mix --rank "$RANK" --selection $WORK/final-cxgptnl-w$RANK.npz \
      --plan experiments/final/final_plan.json --spill $WORK/spill --out ~/final_cxgptnl --list ~/final_cxgptnl_files.txt \
      --readers "${READERS:-8}" --workers "${WORKERS:-4}"
    [ "$(wc -l < ~/final_cxgptnl_files.txt)" = 128 ] || { echo "$(hostname): ~/final_cxgptnl_files.txt does not list 128 files"; exit 1; }
    while read -r f; do [ -s "$f" ] || { echo "$(hostname): missing $f"; exit 1; }; done < ~/final_cxgptnl_files.txt
    gsutil -q cp ~/final_cxgptnl/share.json $F/shares/final-cxgptnl-w$RANK.json
    echo "$(hostname): ready, $(du -sh ~/final_cxgptnl | cut -f1) of data, $(df -h ~ | tail -1 | awk '{print $4}') free"
    ;;
esac
