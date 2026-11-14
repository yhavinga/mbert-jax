#!/bin/bash
# The Hugging Face export of the final model (T2, decide.py of 13 Nov) on one worker of a v4-32 in us-central2-b, on
# its CPU only: the TPU stays with whatever runs there. Started detached from co1, with the code at COMMIT in $W/code:
#   COMMIT=<hf-release commit> PHASE=local bash $W/code/experiments/final/release/stage_release.sh
#   COMMIT=<hf-release commit> PHASE=stage bash $W/code/experiments/final/release/stage_release.sh
# local  T2's params read in place from gs:// (convert_gcs_checkpoint.py --params-only, same region); JAX reference
#        logits (scripts/check_hf_export.py jax, in the venv of the evaluations, $JAXPY, JAX on CPU); venv-a (torch CPU, transformers 4.57.1 as the evaluations);
#        convert_flax_to_hf.py into $W/hf; check_hf_export.py hf on $W/hf; the tokenizer and config files of $W/hf
#        against those of the public repo (read only). Reports to $OUT.
# stage  the model card into $W/hf/README.md; push_staging.py into the private $REPO; check_hf_export.py hf from the
#        Hub with venv-a and with venv-b (the newest transformers); reports and the export directory to $OUT.
# Every Python step runs at nice 19, ionice idle, on $CPUS, with JAX_PLATFORMS=cpu; process names match no busy
# pattern of the chains (convert_gcs_checkpoint.py is run from stdin). Disk: ~6 GB in $W (no /dev/shm). Nothing is
# deleted; $W is removed by hand afterwards with its literal path.
set -euo pipefail
PHASE=${PHASE:?set PHASE to local or stage}
COMMIT=${COMMIT:?set COMMIT to the hf-release commit}
CKPT=${CKPT:-gs://t5-dutch-english-usc2/mbert/final-culturax-230m-epoch2/ckpt_keep/1306000}
OUT=${OUT:-gs://t5-dutch-english-usc2/mbert/final/hf-release}
REPO=${REPO:-yhavinga/dmbert-230m-culturax-staging}
PUBLIC=yhavinga/dmbert-230m-culturax
PUBLIC_REV=cf67c63e62d2cdcccf4a13eb0eca2cab332af296
TOKENIZER=yhavinga/dutch-llama-tokenizer
W=${W:-$HOME/hf_release}
JAXPY=${JAXPY:-$HOME/mbert-dumb/.venv/bin/python}
UV=${UV:-$HOME/.local/bin/uv}
CPUS=${CPUS:-200-231}
TORCH=2.15.0
TF_B=5.23.0
export JAX_PLATFORMS=cpu TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=32 MKL_NUM_THREADS=32 PYTHONNOUSERSITE=1
export HF_HUB_CACHE=$W/hfcache UV_CACHE_DIR=$W/uvcache HF_HUB_DISABLE_PROGRESS_BARS=1
LOW="nice -n 19 ionice -c 3 taskset -c $CPUS"
C=$W/code
cd "$C"
[ "$(git rev-parse HEAD)" = "$COMMIT" ] || { echo "$C is not at $COMMIT"; exit 2; }
mkdir -p "$W/reports"
log() { echo "$(date '+%F %T') $*"; }

venv() {  # venv NAME TRANSFORMERS [HUB]: torch CPU and that transformers (and huggingface_hub, else what it needs)
  local py=$W/$1/bin/python
  if [ ! -x "$py" ] || ! "$py" -c "import torch, transformers; assert transformers.__version__ == '$2'" 2>/dev/null; then
    $UV venv -q --python 3.12 "$W/$1"
    $UV pip install -q --python "$py" --index-url https://download.pytorch.org/whl/cpu "torch==$TORCH"
    $UV pip install -q --python "$py" "transformers==$2" ${3:+"huggingface_hub==$3"} numpy safetensors
  fi
  "$py" -c "import torch, transformers, tokenizers, huggingface_hub; print('$1: torch', torch.__version__, 'transformers', transformers.__version__, 'tokenizers', tokenizers.__version__, 'hub', huggingface_hub.__version__)"
}

if [ "$PHASE" = local ]; then
  log "params of $CKPT"
  if [ ! -f "$W/flax/params.pkl" ]; then
    rm -rf "$W/flax.tmp"
    $LOW "$JAXPY" - "$CKPT" "$W/flax.tmp" "$TOKENIZER" <<'PY'
import runpy, sys
ckpt, out, tok = sys.argv[1:4]
sys.argv = ["convert_gcs_checkpoint.py", "--checkpoint", ckpt, "--params-only", "--output", out, "--tokenizer", tok,
            "--global-attn-every-n-layers", "3"]
runpy.run_path("scripts/convert_gcs_checkpoint.py", run_name="__main__")
PY
    mv "$W/flax.tmp" "$W/flax"
  fi
  log "JAX reference logits"
  [ -f "$W/jax_ref.npz" ] || $LOW "$JAXPY" scripts/check_hf_export.py jax --flax "$W/flax" --out "$W/jax_ref.npz"
  log "venv-a"
  venv venv-a 4.57.1 0.36.2
  log "convert_flax_to_hf.py"
  rm -rf "$W/hf.tmp"
  $LOW "$W/venv-a/bin/python" scripts/convert_flax_to_hf.py --input "$W/flax" --output "$W/hf.tmp" --tokenizer "$TOKENIZER"
  rm -rf "$W/hf"; mv "$W/hf.tmp" "$W/hf"
  ls -la "$W/hf"
  log "checks of the export directory"
  rc=0
  $LOW "$W/venv-a/bin/python" scripts/check_hf_export.py hf --model "$W/hf" --reference "$W/jax_ref.npz" --out "$W/reports/check-local-4.57.1.json" || rc=1
  log "tokenizer and config files against $PUBLIC @ $PUBLIC_REV"
  "$W/venv-a/bin/python" - "$W/hf" "$PUBLIC" "$PUBLIC_REV" "$W/reports/files-vs-public.json" <<'PY' || rc=1
import hashlib, json, sys
from pathlib import Path
from huggingface_hub import hf_hub_download
hf, repo, rev, out = sys.argv[1:5]
res = {}
for f in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja", "config.json"]:
    mine = Path(hf) / f
    theirs = Path(hf_hub_download(repo, f, revision=rev))
    h = lambda p: hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None
    res[f] = {"export": h(mine), "public": h(theirs), "equal": h(mine) == h(theirs)}
json.dump(res, open(out, "w"), indent=1)
print(json.dumps({f: r["equal"] for f, r in res.items()}))
PY
  gsutil -q cp "$W"/reports/*.json "$OUT/reports/"
  log "local done, checks exit $rc"
  exit $rc
fi

if [ "$PHASE" = stage ]; then
  [ -f "$W/hf/model.safetensors" ] && [ -f "$W/reports/check-local-4.57.1.json" ] || { echo "run PHASE=local first"; exit 2; }
  cp experiments/final/release/MODEL_CARD.md "$W/hf/README.md"
  venv venv-a 4.57.1 0.36.2
  log "push to $REPO (private)"
  "$W/venv-a/bin/python" experiments/final/release/push_staging.py --folder "$W/hf" --repo "$REPO" --out "$W/reports/push.json"
  rev=$("$W/venv-a/bin/python" -c "import json; print(json.load(open('$W/reports/push.json'))['commit'])")
  rc=0
  log "checks from the Hub at $rev, transformers 4.57.1"
  $LOW "$W/venv-a/bin/python" scripts/check_hf_export.py hf --model "$REPO" --revision "$rev" --reference "$W/jax_ref.npz" --out "$W/reports/check-hub-4.57.1.json" || rc=1
  venv venv-b "$TF_B"
  log "checks from the Hub at $rev, transformers $TF_B"
  $LOW "$W/venv-b/bin/python" scripts/check_hf_export.py hf --model "$REPO" --revision "$rev" --reference "$W/jax_ref.npz" --out "$W/reports/check-hub-$TF_B.json" || rc=1
  gsutil -q cp "$W"/reports/*.json "$OUT/reports/"
  gsutil -q -m cp "$W"/hf/* "$OUT/export/"
  log "stage done at $rev, checks exit $rc"
  exit $rc
fi
echo "PHASE must be local or stage"; exit 2
