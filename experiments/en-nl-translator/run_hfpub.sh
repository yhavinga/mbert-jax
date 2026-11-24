#!/bin/bash
# G4-8-s140 for Hugging Face (protocol-g4.md §26) on ONE worker of mbert-mix, CPU only; one stage per call, detached
# from ssh, log ~/logs/hfp_<stage>_$TAG.log ending in "EXIT <code>". publish_g48_run.sh (co1) runs the stages in order
# and holds mbert-ops/final_eval_mix.lock. Prints ids, counts, sizes and scores only. Deletes nothing: the chain on co1
# deletes intermediate files with literal paths, visibly.
#   TAG=<tag> bash run_hfpub.sh <setup|llama|download|convert|select|check|example|merge|mergecheck|gguf|quant|smoke|stage>
set -euo pipefail
STAGE=${1:?a stage}
TAG=${TAG:?TAG}
D=/dev/shm/hfpub                      # big files (base, adapter, merged, GGUF)
HOMEP=$HOME/hfpub                     # venvs and llama.cpp on the boot disk
V=$HOMEP/venv
GV=$HOMEP/venv-gguf
LL=$HOMEP/llama.cpp
CODE=$(cd "$(dirname "$0")" && pwd)
G=gs://t5-dutch-english-usc2/mbert/translator/sft/g4arm
H=$G/hf
BASE_REPO=google/gemma-4-31B-it
BASE_REV=842da3794eaa0b77d5f08bae87a17459d91ff475
LLAMA_TAG=v1.0.0
LLAMA_COMMIT=b1b770d90d94cad9e9ea5b993bdbc383a522fb28
MAXTEXT_NORM=0.0017412984739521172    # translate.py's LORA line for G4-8-s140 (g4t8c)
NAME=hfp_${STAGE}_$TAG
THREADS=${THREADS:-96}
export PATH=$HOME/.local/bin:$PATH UV_CACHE_DIR=$HOMEP/uv-cache HF_HUB_DISABLE_PROGRESS_BARS=1 HF_XET_HIGH_PERFORMANCE=1
mkdir -p ~/logs "$D" "$HOMEP"
[ -e ~/logs/$NAME.log ] && { echo "~/logs/$NAME.log exists; a new start takes a new TAG"; exit 1; }

body() {
case $STAGE in
setup)
  [ -x $V/bin/python ] || uv venv -q --python 3.12 $V
  VIRTUAL_ENV=$V UV_TORCH_BACKEND=cpu uv pip install -q torch==2.9.1
  VIRTUAL_ENV=$V uv pip install -q transformers==5.23.0 peft==0.22.0 accelerate==1.16.0 safetensors numpy pyarrow \
    sacrebleu langid pyyaml huggingface_hub==2.4.0 hf_xet
  $V/bin/python -c "import importlib.metadata as m, json; print('VERSIONS', json.dumps({p: m.version(p) for p in ('torch', 'transformers', 'peft', 'accelerate', 'safetensors', 'huggingface_hub', 'hf_xet', 'sacrebleu', 'langid')}))"
  [ -s ~/.cache/huggingface/token ] || { echo "no ~/.cache/huggingface/token"; exit 1; }
  echo READY ;;
llama)
  [ -d $LL/.git ] || git clone -q --depth 1 --branch $LLAMA_TAG https://github.com/ggml-org/llama.cpp $LL
  [ "$(git -C $LL rev-parse HEAD)" = $LLAMA_COMMIT ] || { echo "llama.cpp $LLAMA_TAG is not $LLAMA_COMMIT"; exit 1; }
  [ -x $GV/bin/python ] || uv venv -q --python 3.12 $GV
  VIRTUAL_ENV=$GV uv pip install -q -r $LL/requirements/requirements-convert_hf_to_gguf.txt --index-strategy unsafe-best-match
  VIRTUAL_ENV=$GV uv pip install -q cmake
  (cd $LL && $GV/bin/cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DLLAMA_CURL=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF > $HOMEP/cmake.log 2>&1 \
    || { tail -30 $HOMEP/cmake.log; exit 1; })
  $GV/bin/cmake --build $LL/build --config Release -j 64 --target llama-quantize llama-server llama-cli > $HOMEP/build.log 2>&1 \
    || { tail -40 $HOMEP/build.log; exit 1; }
  echo "LLAMA {\"tag\": \"$LLAMA_TAG\", \"commit\": \"$(git -C $LL rev-parse HEAD)\"}" ;;
download)
  if [ ! -s $D/base/done.txt ]; then
    $V/bin/hf download $BASE_REPO --revision $BASE_REV --local-dir $D/base > /dev/null
    echo "$BASE_REPO@$BASE_REV" > $D/base/done.txt
  fi
  mkdir -p $D/mt $D/eval/arm $D/eval/base $D/out
  gsutil -m -q cp $G/G4-8-s140/adapter.npz $G/G4-8-s140/adapter.json $G/G4-8-s140/summary.json $D/mt/
  gsutil -m -q cp -r $G/eval/G4-8-s140/pairs $D/eval/arm/
  gsutil -m -q cp -r $G/eval/G4-0/pairs $D/eval/base/
  gsutil -m -q cp $H/cards/hf-card-lora.md $H/cards/hf-card-gguf.md $D/out/
  echo "DOWNLOAD {\"base_gb\": $(du -sb $D/base | awk '{printf "%.2f", $1 / 1e9}'), \"adapter_gb\": $(du -sb $D/mt | awk '{printf "%.2f", $1 / 1e9}')}" ;;
convert)
  $V/bin/python -u $CODE/g4peft.py convert --npz $D/mt/adapter.npz --meta $D/mt/adapter.json \
    --config $D/base/config.json --base-dir $D/base --out $D/lora
  gsutil -m -q cp $D/lora/* $H/lora/ ;;
select)
  $V/bin/python -u $CODE/g4hfcheck.py select --arm $D/eval/arm --base $D/eval/base --per-set 10 --smoke 8 --out $D/out/ids.json
  gsutil -q cp $D/out/ids.json $H/out/ ;;
check)
  rc=0
  $V/bin/python -u $CODE/g4hfcheck.py check --model-dir $D/base --adapter-dir $D/lora --arm $D/eval/arm \
    --base $D/eval/base --ids $D/out/ids.json --runs adapter:arm,base:base,base:arm --require 'adapter:arm>=0.99' \
    --require 'base:base>=0.99' --threads $THREADS --out $D/out/check.json || rc=$?
  gsutil -q cp $D/out/check.json $H/out/ || true
  exit $rc ;;
example)
  rc=0
  $V/bin/python -u $CODE/g4hfcheck.py example --card $D/out/hf-card-lora.md --replace "google/gemma-4-31B-it=$D/base" \
    --replace "yhavinga/gemma-4-31b-it-en-nl-lora=$D/lora" --arm $D/eval/arm --ids $D/out/ids.json --n 4 \
    --out $D/out/example.json || rc=$?
  gsutil -q cp $D/out/example.json $H/out/ || true
  exit $rc ;;
merge)
  $V/bin/python -u $CODE/g4peft.py merge --base-dir $D/base --adapter-dir $D/lora --out $D/merged \
    --maxtext-norm $MAXTEXT_NORM
  gsutil -q cp $D/merged/merge.json $H/out/ ;;
mergecheck)
  rc=0
  $V/bin/python -u $CODE/g4hfcheck.py check --model-dir $D/merged --arm $D/eval/arm --base $D/eval/base \
    --ids $D/out/ids.json --runs model:arm --require 'model:arm>=0.99' --threads $THREADS --out $D/out/mergecheck.json || rc=$?
  gsutil -q cp $D/out/mergecheck.json $H/out/ || true
  exit $rc ;;
gguf)
  mkdir -p $D/gguf
  # llama.cpp pins transformers 4.57.6; should its AutoConfig not know gemma4, once more with the release that does
  if ! $GV/bin/python -u $LL/convert_hf_to_gguf.py $D/merged --outtype bf16 --outfile $D/gguf/gemma-4-31b-it-en-nl-BF16.gguf; then
    echo "GGUF_RETRY with transformers 5.23.0"
    VIRTUAL_ENV=$GV uv pip install -q transformers==5.23.0
    $GV/bin/python -u $LL/convert_hf_to_gguf.py $D/merged --outtype bf16 --outfile $D/gguf/gemma-4-31b-it-en-nl-BF16.gguf
  fi
  echo "GGUF {\"bf16_bytes\": $(stat -c %s $D/gguf/gemma-4-31b-it-en-nl-BF16.gguf)}" ;;
quant)
  mkdir -p $D/upload-gguf
  for Q in Q8_0 Q4_K_M; do
    F=$D/upload-gguf/gemma-4-31b-it-en-nl-$Q.gguf
    [ -s $F ] || $LL/build/bin/llama-quantize $D/gguf/gemma-4-31b-it-en-nl-BF16.gguf $F $Q $THREADS > $D/out/quant-$Q.log 2>&1 \
      || { tail -20 $D/out/quant-$Q.log; exit 1; }
    echo "QUANT {\"quant\": \"$Q\", \"bytes\": $(stat -c %s $F), \"sha256\": \"$(sha256sum $F | cut -d' ' -f1)\"}"
  done ;;
smoke)
  JINJA=""; $LL/build/bin/llama-server --help 2>&1 | grep -q -- '--jinja' && JINJA=--jinja
  k=0
  for Q in Q4_K_M Q8_0; do
    PORT=$((8091 + k)); k=$((k + 1))
    $LL/build/bin/llama-server -m $D/upload-gguf/gemma-4-31b-it-en-nl-$Q.gguf --host 127.0.0.1 --port $PORT -c 4096 \
      -t $THREADS $JINJA > $D/out/server-$Q.log 2>&1 &
    PID=$!
    for _ in $(seq 180); do curl -sf http://127.0.0.1:$PORT/health > /dev/null && break; sleep 5; done
    curl -sf http://127.0.0.1:$PORT/health > /dev/null || { kill $PID; tail -20 $D/out/server-$Q.log | cut -c1-300; exit 1; }
    $V/bin/python -u $CODE/g4hfcheck.py smoke --server http://127.0.0.1:$PORT --quant $Q --arm $D/eval/arm \
      --ids $D/out/ids.json --out $D/out/smoke-$Q.json || { kill $PID; exit 1; }
    kill $PID; wait $PID 2> /dev/null || true
    gsutil -q cp $D/out/smoke-$Q.json $H/out/
  done
  $V/bin/python -c "
import json, sys
for q in ('Q4_K_M', 'Q8_0'):
    s = json.load(open('$D/out/smoke-' + q + '.json'))
    if not s['nonempty'] == s['dutch'] == s['n']:
        sys.exit(f'{q}: {s[\"nonempty\"]} non-empty, {s[\"dutch\"]} Dutch of {s[\"n\"]}')
print('SMOKE_OK')" ;;
stage)
  # the upload folders: the adapter with its card, the quants with theirs (sizes, llama.cpp tag, smoke chrF filled in)
  mkdir -p $D/upload-lora
  cp $D/lora/adapter_model.safetensors $D/lora/adapter_config.json $D/upload-lora/
  cp $D/out/hf-card-lora.md $D/upload-lora/README.md
  $V/bin/python - <<EOF
import json, os
d = "$D"
s = {q: json.load(open(f"{d}/out/smoke-{q}.json")) for q in ("Q4_K_M", "Q8_0")}
card = open(f"{d}/out/hf-card-gguf.md").read()
for q in ("Q4_K_M", "Q8_0"):
    size = os.path.getsize(f"{d}/upload-gguf/gemma-4-31b-it-en-nl-{q}.gguf")
    card = card.replace("{{SIZE_" + q + "}}", f"{size / 1e9:.1f} GB").replace("{{CHRF_" + q + "}}", f"{s[q]['chrf_corpus']:.1f}")
card = card.replace("{{LLAMA_TAG}}", "$LLAMA_TAG").replace("{{SMOKE_N}}", str(s["Q4_K_M"]["n"]))
assert "{{" not in card, "a placeholder left in the GGUF card"
open(f"{d}/upload-gguf/README.md", "w").write(card)
print("STAGED", json.dumps({"lora": sorted(os.listdir(f"{d}/upload-lora")), "gguf": sorted(os.listdir(f"{d}/upload-gguf"))}))
EOF
  gsutil -q cp $D/upload-gguf/README.md $H/cards/hf-card-gguf.filled.md ;;
*) echo "unknown stage $STAGE"; exit 1 ;;
esac
}

export -f body 2> /dev/null || true
declare -f body > ~/job_$NAME.sh
cat >> ~/job_$NAME.sh <<EOF
set -euo pipefail
$(declare -p STAGE TAG D HOMEP V GV LL CODE G H BASE_REPO BASE_REV LLAMA_TAG LLAMA_COMMIT MAXTEXT_NORM NAME THREADS | sed 's/^declare -[a-z-]* //;s/^/export /')
export PATH=$PATH UV_CACHE_DIR=$UV_CACHE_DIR HF_HUB_DISABLE_PROGRESS_BARS=1 HF_XET_HIGH_PERFORMANCE=1 TOKENIZERS_PARALLELISM=false
echo "\$(date -u +%T) \$(hostname): $NAME"
body
echo "\$(date -u +%T) $NAME done"
EOF
setsid nohup bash -c "nice -n 5 bash ~/job_$NAME.sh > ~/logs/$NAME.log 2>&1; echo EXIT \$? >> ~/logs/$NAME.log" > /dev/null 2>&1 < /dev/null &
echo "started $NAME on w$(hostname | sed 's/.*-w-//')"
