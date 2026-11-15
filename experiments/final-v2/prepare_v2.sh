#!/bin/bash
# On a worker of mbert-mix: one step of the data and code of dmbert v2 (protocol.md), started by final_v2_run.sh on co1.
# CPU, disk and network only: nothing here imports JAX or opens the TPU, at nice 19 and ionice idle.
#   setup      ~/mbert-v2 at COMMIT; the venv PY (~/mbert-jax/.venv) against vm.lock and the fingerprint VENV_FP of its
#              package list; the HF token there (for the tokenizer); free disk, /dev/shm and RAM
#   stage      this host's share of kept/train by the plan $D/plan.json (its sha256 PLAN_SHA256), stage_optout.py stage:
#              ~/fw2optout{,_smoke}/ and the list ~/fw2optout{,_smoke}_files.txt; share.json and overlap.json to
#              $D/shares/ and $D/overlap/. KIND=smoke: only the files of SMOKE_FILES (a file in the checkout)
#   scan       experiments/final/blocklist_scan.py over the list: the documents of hebban.nl, to $D/blocklist/w<N>.json
#   heldout    worker 0: stage_optout.py heldout from the overlap.json of every host, to $D/heldout/
#              (KIND=smoke: kept/test itself, unchanged)
#   fetch      the held-out file of $D/heldout/ to ~/fw2optout{,_smoke}_heldout.parquet, its md5 checked against
#              heldout.json and written next to it; ROWS (the documents of kept/train less hebban.nl over all hosts)
#              to ~/fw2optout{,_smoke}_files.rows
#   memsample  worker 0: the sample of the memorization test, from the held-out file and this host's share, to
#              MEMTEST_OUT (default $V/memtest/; the smoke test's own elsewhere)
# Detached: the log is ~/logs/v2_<step>_<kind>.log, ending in "EXIT <code>", copied to $D/logs/. Environment: COMMIT
# (required), KIND (final or smoke, default final), PLAN_SHA256 (stage, heldout), ROWS (fetch), VENV_FP (setup), PY.
# Deletes only its own downloads in /dev/shm/fw2optout_stage_<kind>.
set -euo pipefail
STEP=${1:?usage: prepare_v2.sh setup|stage|scan|heldout|fetch|memsample}
COMMIT=${COMMIT:?set COMMIT to the commit of experiments/final-v2}
KIND=${KIND:-final}
RANK=${RANK:-$(hostname | sed 's/.*-w-//')}
PY=${PY:-$HOME/mbert-jax/.venv/bin/python}
V=${V:-gs://t5-dutch-english-usc2/mbert/final-v2}
LOCK=gs://t5-dutch-english-usc2/mbert/corpus-runs/scripts/vm.lock
case $STEP in setup|stage|scan|heldout|fetch|memsample) ;; *) echo "the step must be setup, stage, scan, heldout, fetch or memsample, not $STEP"; exit 1 ;; esac
case $KIND in
  final) D=$V/data; SUFFIX= ;;
  smoke) D=$V/smoke; SUFFIX=_smoke ;;
  *) echo "KIND must be final or smoke, not $KIND"; exit 1 ;;
esac
[[ $RANK =~ ^[0-3]$ ]] || { echo "$(hostname): worker number '$RANK' is not one of 0 to 3"; exit 1; }
case $STEP in heldout|memsample) [ "$RANK" = 0 ] || { echo "$(hostname): $STEP runs on worker 0"; exit 1; } ;; esac
LOG=$HOME/logs/v2_${STEP}_$KIND.log
if [ -z "${V2_DETACHED:-}" ]; then
  # The detached wrapper's command line; the ssh shell that runs this script has another
  pgrep -f "logs/v2_${STEP}_${KIND}[.]log 2>&1; echo EXIT" > /dev/null && { echo "$(hostname): prepare_v2.sh $STEP $KIND runs already"; exit 1; }
  mkdir -p ~/logs
  V2_DETACHED=1 setsid nohup nice -n 19 ionice -c3 bash -c "bash $0 $STEP > $LOG 2>&1; echo EXIT \$? >> $LOG; gsutil -q cp $LOG $D/logs/v2_${STEP}-w$RANK.log" > /dev/null 2>&1 < /dev/null &
  echo "$(hostname): started, $LOG"
  exit 0
fi

cd "$HOME"
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1
CODE=$HOME/mbert-v2
DATA=$HOME/fw2optout$SUFFIX
LIST=$HOME/fw2optout${SUFFIX}_files.txt
HELDOUT=$HOME/fw2optout${SUFFIX}_heldout.parquet
SHM=/dev/shm/fw2optout_stage_$KIND
if [ "$STEP" != setup ]; then
  [ "$(git -C "$CODE" rev-parse HEAD 2>/dev/null)" = "$COMMIT" ] || { echo "$(hostname): $CODE is not at $COMMIT (run setup)"; exit 1; }
fi
echo "$(hostname): $STEP $KIND at ${COMMIT:0:7}, $(df -h ~ | tail -1 | awk '{print $4}') free on the disk, $(df -h /dev/shm | tail -1 | awk '{print $4}') in /dev/shm, $(free -g | awk '/Mem:/ {print $7}') GB RAM available"
plan_here() {  # the plan of $D, checked against PLAN_SHA256, at $SHM/plan.json
  mkdir -p "$SHM"
  gsutil -q cp "$D/plan.json" "$SHM/plan.json"
  [ "$(sha256sum "$SHM/plan.json" | cut -d' ' -f1)" = "${PLAN_SHA256:?set PLAN_SHA256}" ] || { echo "$(hostname): $D/plan.json is not the plan of sha256 $PLAN_SHA256"; exit 1; }
}
case $STEP in
  setup)
    [ -d "$CODE/.git" ] || git clone -q https://github.com/yhavinga/mbert-jax.git "$CODE"
    git -C "$CODE" fetch -q origin final-v2
    git -C "$CODE" checkout -q "$COMMIT"
    [ "$(git -C "$CODE" rev-parse HEAD)" = "$COMMIT" ] || { echo "$(hostname): $CODE is not at $COMMIT"; exit 1; }
    [ -x "$PY" ] || { echo "$(hostname): no $PY"; exit 1; }
    gsutil -q cp $LOCK /tmp/v2_vm.lock
    "$PY" - /tmp/v2_vm.lock "${VENV_FP:?set VENV_FP}" <<'EOF'
import hashlib, importlib.metadata as m, sys
pins = dict(line.strip().split("==", 1) for line in open(sys.argv[1]) if "==" in line and not line.startswith("#"))
names = ["jax", "jaxlib", "libtpu", "flax", "optax", "orbax-checkpoint", "grain", "datasets", "transformers", "tokenizers",
         "numpy", "pyarrow", "xxhash"]
wrong = {n: (m.version(n), pins.get(n)) for n in names if m.version(n) != pins.get(n)}
listing = "\n".join(sorted(f"{d.metadata['Name'].lower()}=={d.version}" for d in m.distributions())) + "\n"
fingerprint = hashlib.md5(listing.encode()).hexdigest()[:12]
print({n: m.version(n) for n in names}, "fingerprint", fingerprint)
if wrong:
    sys.exit(f"the venv differs from vm.lock: {wrong}")
if fingerprint != sys.argv[2]:
    sys.exit(f"the venv's package list has fingerprint {fingerprint}, not {sys.argv[2]}")
EOF
    [ -s ~/.cache/huggingface/token ] || { echo "$(hostname): no ~/.cache/huggingface/token"; exit 1; }
    echo "$(hostname): setup ready"
    ;;
  stage)
    plan_here
    files=()
    if [ "$KIND" = smoke ]; then files=(--files "$CODE/experiments/final-v2/smoke_files.txt"); fi
    "$PY" -u "$CODE/experiments/final-v2/stage_optout.py" stage --plan "$SHM/plan.json" --rank "$RANK" --out "$DATA" \
      --list "$LIST" --download-dir "$SHM" --workers "${WORKERS:-8}" "${files[@]}"
    gsutil -q cp "$DATA/share.json" "$D/shares/w$RANK.json"
    gsutil -q cp "$DATA/overlap.json" "$D/overlap/w$RANK.json"
    echo "$(hostname): staged $(wc -l < "$LIST") files, $(du -sh "$DATA" | cut -f1), $(df -h ~ | tail -1 | awk '{print $4}') free"
    ;;
  scan)
    "$PY" -u "$CODE/experiments/final/blocklist_scan.py" --files-list "$LIST" --blocklist hebban.nl --out "$DATA/blocklist.json"
    gsutil -q cp "$DATA/blocklist.json" "$D/blocklist/w$RANK.json"
    ;;
  heldout)
    plan_here
    if [ "$KIND" = smoke ]; then
      # The smoke test's held-out set is kept/test as it is (the comparison with all of kept/train is the final one's)
      "$PY" - "$SHM/plan.json" "$SHM" <<'EOF'
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path.home() / "mbert-v2" / "experiments" / "final-v2"))
import stage_optout as so
plan = json.load(open(sys.argv[1]))
local = so.fetch_test(plan, Path(sys.argv[2]))
import hashlib
md5 = hashlib.md5(local.read_bytes()).hexdigest()
Path(sys.argv[2], "heldout.json").write_text(json.dumps({"file": "heldout.parquet", "md5": md5, "test_file": plan["test"]["file"],
                                                         "note": "kept/test unchanged, for the smoke test"}, indent=1) + "\n")
local.rename(Path(sys.argv[2], "heldout.parquet"))
print("HELDOUT smoke", md5)
EOF
    else
      mkdir -p "$SHM/overlap"
      gsutil -q cp "$D/overlap/w0.json" "$D/overlap/w1.json" "$D/overlap/w2.json" "$D/overlap/w3.json" "$SHM/overlap/"
      "$PY" -u "$CODE/experiments/final-v2/stage_optout.py" heldout --plan "$SHM/plan.json" \
        --overlap "$SHM/overlap/w0.json" "$SHM/overlap/w1.json" "$SHM/overlap/w2.json" "$SHM/overlap/w3.json" \
        --download-dir "$SHM" --out "$SHM/heldout.parquet" --summary "$SHM/heldout.json"
    fi
    gsutil -q cp "$SHM/heldout.parquet" "$SHM/heldout.json" "$D/heldout/"
    ;;
  fetch)
    [[ ${ROWS:-} =~ ^[0-9]+$ ]] || { echo "$(hostname): ROWS is not a number: '${ROWS:-}'"; exit 1; }
    mkdir -p "$SHM"
    gsutil -q cp "$D/heldout/heldout.parquet" "$D/heldout/heldout.json" "$SHM/"
    md5=$("$PY" -c "import json, sys; print(json.load(open(sys.argv[1]))['md5'])" "$SHM/heldout.json")
    [ "$(md5sum "$SHM/heldout.parquet" | cut -d' ' -f1)" = "$md5" ] || { echo "$(hostname): the held-out file is not the one of heldout.json"; exit 1; }
    cp "$SHM/heldout.parquet" "$HELDOUT.new" && mv "$HELDOUT.new" "$HELDOUT"
    echo "$md5" > "$HELDOUT.md5"
    echo "$ROWS" > "${LIST%.txt}.rows"
    echo "$(hostname): $HELDOUT ($md5), ${LIST%.txt}.rows = $ROWS"
    ;;
  memsample)
    mkdir -p "$SHM"
    "$PY" -u "$CODE/experiments/final-v2/stage_optout.py" memsample --heldout "$HELDOUT" --files-list "$LIST" \
      --out "$SHM/memsample.parquet" --summary "$SHM/memsample.json"
    gsutil -q cp "$SHM/memsample.parquet" "$SHM/memsample.json" "${MEMTEST_OUT:-$V/memtest}/"
    ;;
esac
