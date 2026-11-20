#!/bin/bash
# On a worker of mbert-mix: ~/mbert-v3 at COMMIT (branch v3 of mbert-jax), and the venv PY (~/mbert-jax/.venv, that of
# dmbert v2) checked against vm.lock and the fingerprint VENV_FP of its package list, as prepare_v2.sh setup does.
# CPU and network only; prints "<host>: setup ready" at the end. Environment: COMMIT (required), VENV_FP (required), PY.
set -euo pipefail
COMMIT=${COMMIT:?set COMMIT}
PY=${PY:-$HOME/mbert-jax/.venv/bin/python}
CODE=$HOME/mbert-v3
LOCK=gs://t5-dutch-english-usc2/mbert/corpus-runs/scripts/vm.lock
[ -d "$CODE/.git" ] || git clone -q https://github.com/yhavinga/mbert-jax.git "$CODE"
git -C "$CODE" fetch -q origin v3
git -C "$CODE" checkout -q "$COMMIT"
[ "$(git -C "$CODE" rev-parse HEAD)" = "$COMMIT" ] || { echo "$(hostname): $CODE is not at $COMMIT"; exit 1; }
[ -x "$PY" ] || { echo "$(hostname): no $PY"; exit 1; }
gsutil -q cp $LOCK /tmp/v3_vm.lock
"$PY" - /tmp/v3_vm.lock "${VENV_FP:?set VENV_FP}" <<'PY'
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
echo "$(hostname): setup ready"
