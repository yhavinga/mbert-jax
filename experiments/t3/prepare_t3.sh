#!/bin/bash
# On a worker of the v4-32 of T3 (mbert-corpus; for the evaluation also mbert-mix), started by t3_run.sh on co1
# (experiments/t3/protocol.md, sections 5 and 9).
#   COMMIT=<t3 commit> VENV_FP=<fingerprint> bash prepare_t3.sh setup
#       ~/mbert-t3 at COMMIT (branch t3 of mbert-jax), the venv PY (~/mbert-jax/.venv) against vm.lock and the
#       fingerprint of its package list, tokenizer (a) built to ~/mbert-t3-tokenizer; prints "TOK <sha256>  <file>"
#       per tokenizer file and "<host>: setup ready". In the foreground, CPU and network only. With TOKENIZER=0 no
#       tokenizer (the evaluation needs none).
#   COMMIT=<t3 commit> bash prepare_t3.sh data
#       Detached (log ~/logs/t3_data.log, ending in "EXIT <code>"): stage_t3.py files (T2's 32 files of this worker
#       by size and sha256, a missing one fetched from Hugging Face), keys (of ~/heldout.parquet), scan (T2's list),
#       write (~/culturax_nl/t3/ and ~/t3_files.txt), the proof (scan --require-none of ~/t3_files.txt), and on
#       worker 0 memsample; every summary to $T/data/w<N>/ (keys.json from worker 0 only), the sample to $T/memtest/.
# Deletes nothing; the originals in ~/culturax_nl/nl/ are not changed. Environment: COMMIT (required), VENV_FP (setup),
# PY, M (the bucket's mbert directory), RANK; F (T2's staging, default $M/final); HELDOUT_MD5 only for the dry run.
set -euo pipefail
STEP=${1:?usage: prepare_t3.sh setup or data}
COMMIT=${COMMIT:?set COMMIT to the commit of branch t3}
PY=${PY:-$HOME/mbert-jax/.venv/bin/python}
M=${M:-gs://t5-dutch-english-usc2/mbert}
T=$M/t3
F=${F:-$M/final}
CODE=$HOME/mbert-t3
RANK=${RANK:-$(hostname | sed 's/.*-w-//')}
LOG=$HOME/logs/t3_data.log
HELDOUT_MD5=${HELDOUT_MD5:-b2f579a876d06eef40c94270537bb7af}
BUSY='[t]rain_minimal.py|[f]inetune.py|[f]inetune_extra.py|[e]val_windows.py|[e]val_blimp_nl.py|[m]emtest.py|[v]erify_ckpt.py|[c]onvert_gcs_checkpoint.py'
[[ $RANK =~ ^[0-3]$ ]] || { echo "$(hostname): worker number $RANK is not one of 0 to 3"; exit 1; }

case $STEP in
  setup)
    VENV_FP=${VENV_FP:?set VENV_FP to the fingerprint of the venv}
    [ -d "$CODE/.git" ] || git clone -q https://github.com/yhavinga/mbert-jax.git "$CODE"
    if [ "$(git -C "$CODE" rev-parse HEAD)" != "$COMMIT" ]; then
      pgrep -f "$BUSY" > /dev/null && { echo "$(hostname): $CODE is at another commit and a TPU job runs: not changed"; exit 1; }
      git -C "$CODE" fetch -q origin t3
      git -C "$CODE" checkout -q "$COMMIT"
    fi
    [ "$(git -C "$CODE" rev-parse HEAD)" = "$COMMIT" ] || { echo "$(hostname): $CODE is not at $COMMIT"; exit 1; }
    [ -x "$PY" ] || { echo "$(hostname): no $PY"; exit 1; }
    gsutil -q cp "$M/corpus-runs/scripts/vm.lock" /tmp/t3_vm.lock
    "$PY" - /tmp/t3_vm.lock "$VENV_FP" <<'PY'
import hashlib, importlib.metadata as m, sys
pins = dict(line.strip().split("==", 1) for line in open(sys.argv[1]) if "==" in line and not line.startswith("#"))
names = ["jax", "jaxlib", "libtpu", "flax", "optax", "orbax-checkpoint", "grain", "datasets", "transformers", "tokenizers",
         "numpy", "pyarrow", "xxhash"]
wrong = {n: (m.version(n), pins.get(n)) for n in names if m.version(n) != pins.get(n)}
listing = "\n".join(sorted(f"{d.metadata['Name'].lower()}=={d.version}" for d in m.distributions())) + "\n"
fingerprint = hashlib.md5(listing.encode()).hexdigest()[:12]
if wrong:
    sys.exit(f"the venv differs from vm.lock: {wrong}")
if fingerprint != sys.argv[2]:
    sys.exit(f"the venv's package list has fingerprint {fingerprint}, not {sys.argv[2]}")
PY
    if [ "${TOKENIZER:-1}" = 1 ]; then
      (cd "$CODE" && "$PY" -I scripts/build_tokenizer_v3.py --out "$HOME/mbert-t3-tokenizer" 2> /dev/null) | sed 's/^/TOK /'
      [ -s "$HOME/mbert-t3-tokenizer/tokenizer.json" ] || { echo "$(hostname): no tokenizer (a) in ~/mbert-t3-tokenizer"; exit 1; }
    fi
    echo "$(hostname): setup ready"
    ;;
  data)
    if [ -z "${T3_DATA_DETACHED:-}" ]; then
      pgrep -f "logs/t3_data[.]log 2>&1; echo EXIT" > /dev/null && { echo "$(hostname): prepare_t3.sh data runs already"; exit 1; }
      pgrep -f "$BUSY" > /dev/null && { echo "$(hostname): a TPU job runs; the data of T3 is made before any"; exit 1; }
      [ "$(git -C "$CODE" rev-parse HEAD 2>/dev/null)" = "$COMMIT" ] || { echo "$(hostname): $CODE is not at $COMMIT (setup first)"; exit 1; }
      mkdir -p ~/logs
      T3_DATA_DETACHED=1 setsid nohup nice -n 19 ionice -c3 bash -c "bash $0 data > $LOG 2>&1; echo EXIT \$? >> $LOG; gsutil -q cp $LOG $T/data/logs/t3_data-w$RANK.log" > /dev/null 2>&1 < /dev/null &
      echo "$(hostname): started, $LOG"
      exit 0
    fi
    cd "$CODE"
    W=$HOME/t3_data
    mkdir -p "$W" "$HOME/culturax_nl/t3"
    echo "$(date '+%F %T') worker $RANK, code $COMMIT, $(df -h ~ | tail -1 | awk '{print $4}') free"
    gsutil -q cp "$F/staging/culturax-w$RANK.tsv" "$F/staging/culturax_nl_files-w$RANK.txt" "$W/"
    if [ "$(md5sum ~/heldout.parquet 2>/dev/null | cut -d' ' -f1)" != $HELDOUT_MD5 ]; then
      gsutil -q cp "$M/heldout/nl_part_00097_first40000.parquet" ~/heldout.parquet.new && mv ~/heldout.parquet.new ~/heldout.parquet
    fi
    S="$PY -u experiments/t3/stage_t3.py"
    $S files --reference "$W/culturax-w$RANK.tsv" --list "$W/culturax_nl_files-w$RANK.txt" --fetch --out "$W/files.json"
    $S keys --heldout ~/heldout.parquet --md5 "$HELDOUT_MD5" --out "$W/keys.json"
    $S scan --keys "$W/keys.json" --list "$W/culturax_nl_files-w$RANK.txt" --out "$W/scan.json"
    $S write --scan "$W/scan.json" --out-dir "$HOME/culturax_nl/t3" --list-out "$HOME/t3_files.txt" --summary "$W/write.json"
    $S scan --keys "$W/keys.json" --list "$HOME/t3_files.txt" --out "$W/proof.json" --require-none
    "$PY" -c "import json, sys; k = json.load(open(sys.argv[1])); print(json.dumps({x: k[x] for x in k if x not in ('exact', 'normalized')}))" "$W/keys.json" > "$W/keys-summary.json"
    gsutil -q cp "$W/files.json" "$W/scan.json" "$W/write.json" "$W/proof.json" "$W/keys-summary.json" "$T/data/w$RANK/"
    if [ "$RANK" = 0 ]; then
      gsutil -q cp "$W/keys.json" "$T/data/keys.json"
      $S memsample --heldout ~/heldout.parquet --keys "$W/keys.json" --files-list "$HOME/t3_files.txt" \
        --out "$W/memsample.parquet" --summary "$W/memsample.json"
      gsutil -q cp "$W/memsample.parquet" "$W/memsample.json" "$T/memtest/"
    fi
    echo "$(date '+%F %T') $(hostname): data ready, $(df -h ~ | tail -1 | awk '{print $4}') free"
    ;;
  *) echo "the step must be setup or data, not $STEP"; exit 1 ;;
esac
