#!/bin/bash
# On a worker of mbert-corpus: sft_lora.py (MaxText train_sft with LoRA) detached from ssh, either as a TPU of the
# worker's own 4 chips (default) or, with MULTIHOST=1 started on all four workers at once, as one job over the 16 chips.
#   NAME=sft12b_N MODEL=12b [STEPS=auto] DATA=/dev/shm/translator/smoke-lora/data/train-N.parquet [ROOT=gs://...]
#     [TEMPLATE=<chat template>] bash run_sft.sh [key=value ...]
# The key=value arguments are MaxText overrides after this script's own (later ones win). Log: ~/logs/$NAME.log, a fresh
# file per start that ends in "EXIT <code>"; the log and, from worker 0, the run's summary and adapter go to
# $ROOT/runs/$NAME/.
set -euo pipefail
# ROOT: where the run's log, summary and adapter go (the smoke test's place by default; the fine-tune's arms and tag
# test: gs://t5-dutch-english-usc2/mbert/translator/sft). The venv, the runs and the compilation cache stay in SHM.
ROOT=${ROOT:-gs://t5-dutch-english-usc2/mbert/translator/smoke-lora}
SHM=/dev/shm/translator/smoke-lora
PY=$SHM/venv/bin/python
CODE=$(cd "$(dirname "$0")" && pwd)
WORKER=$(hostname | sed 's/.*-w-//')
NAME=${NAME:?the name of the run}
MODEL=${MODEL:?12b or 27b}
DATA=${DATA:?the training parquet}
MULTIHOST=${MULTIHOST:-0}
STEPS=${STEPS:-auto}  # auto: one epoch, counted by sft_lora.py
case $MODEL in
  12b) CKPT=gs://t5-dutch-english-usc2/mbert/translator/smoke-lora/translategemma-12b-maxtext ;;
  27b) CKPT=gs://t5-dutch-english-usc2/mbert/flan-nl/translategemma-27b-maxtext ;;
  *) echo "MODEL is 12b or 27b, not $MODEL"; exit 1 ;;
esac
TOK=/dev/shm/flan_nl/tokenizer-$MODEL
# sft_data.py's template (the smoke test) or sft_build.py's (template.jinja: the user turn as built, in Gemma 3's turn)
TEMPLATE=${TEMPLATE:-$SHM/data/translategemma_en_nl.jinja}
OUT=$SHM/runs/$NAME

[ -x "$PY" ] || { echo "$(hostname): no $PY (setup_sft.sh)"; exit 1; }
[ -s "$TOK/tokenizer.json" ] || { echo "$(hostname): no tokenizer in $TOK (experiments/flan-nl/setup.sh)"; exit 1; }
[ -s "$DATA" ] && [ -s "$TEMPLATE" ] || { echo "$(hostname): no $DATA or $TEMPLATE (sft_data.py build)"; exit 1; }
[ -z "$(sudo lsof -t /dev/accel* /dev/vfio/* 2> /dev/null)" ] || { echo "$(hostname): the TPU is in use"; exit 1; }
gsutil -q stat "$CKPT/done.txt" || { echo "$(hostname): $CKPT has no done.txt"; exit 1; }
[ -e "$OUT" ] && { echo "$(hostname): $OUT exists; a new run takes a new NAME"; exit 1; }
mkdir -p ~/logs "$OUT"

# EnSiTa's TranslateGemma recipe (arXiv 2609.29511, Table 3) where MaxText has the knob: rank 16, alpha 16, attention
# and MLP, lr 1e-5, cosine to 0 with 10% warmup, weight decay 0.01, clipping 1.0, sequences of 1024. Not available:
# LoRA dropout (MaxText passes 0.0 to qwix) and 8-bit AdamW (adamw in float32 state; the LoRA parameters are few).
# jax_cache_dir: MaxText's default is ~/jax_cache on the nearly full boot disk.
# scan_layers=false: the converted checkpoints are unscanned, as translate.py reads them. num_vocab_tiling=8: the loss
# in 8 chunks of the batch-sequence axis (Gemma's 262,144-token vocabulary); without it 12B with float32 weights and 3
# sequences per chip needs 46.65 GB of HLO temporaries per chip, with it the run fits (9 Nov).
# lora_module_path: MaxText's gemma3 regex (configs/post_train/lora_module_path.yml) names decoder/(scanned_blocks|
# layers_remainder|layers)/, which no unscanned layer (decoder/layers_<i>) matches, and _get_lora_module_path's rewrite
# of "layers/" does not reach it inside that group; qwix then finds no module (9 Nov). Same modules, unscanned names.
LORA_MODULES='decoder/layers_[0-9]+/.*(self_attention/(query|key|value|out)|mlp/(wi_0|wi_1|wo))'
ARGS=(run_name=$NAME model_name=gemma3-$MODEL load_parameters_path=$CKPT/0/items tokenizer_type=huggingface
  tokenizer_path=$TOK hf_path=parquet hf_train_files=$DATA train_split=train chat_template_path=$TEMPLATE
  enable_data_shuffling=false max_target_length=1024 packing=true scan_layers=false
  lora.enable_lora=true lora.lora_rank=16 lora.lora_alpha=16.0 "lora.lora_module_path=$LORA_MODULES"
  learning_rate=1e-5 learning_rate_final_fraction=0.0 warmup_steps_fraction=0.1 adam_b2=0.999
  adam_weight_decay=0.01 gradient_clipping_threshold=1.0 eval_interval=-1 checkpoint_period=100000
  enable_goodput_recording=false monitor_goodput=false log_period=1 skip_jax_distributed_system=true
  num_vocab_tiling=8 jax_cache_dir=$SHM/jax_cache "$@")
if [ "$MULTIHOST" = 1 ]; then
  ENV="export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_DATASETS_CACHE=$SHM/hf-datasets"
  PRE="echo multi-host: jax.distributed.initialize in sft_lora.py"
  # Orbax's final save of a multi-host job waits for every process's commit in one directory: a local /dev/shm per
  # host never completes (9 Nov), so MaxText's output goes to the bucket
  FLAGS="--distributed --maxtext-output $ROOT/runs/$NAME/maxtext"
else
  # A host of the v4-32 as a TPU of its own (translate.sh): without these JAX takes the slice from the TPU metadata
  ENV="export TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 TPU_PROCESS_BOUNDS=1,1,1 TPU_VISIBLE_CHIPS=0,1,2,3 TPU_PROCESS_PORT=8476
export TPU_PROCESS_ADDRESSES=localhost:8476 CLOUD_TPU_TASK_ID=0 ALLOW_MULTIPLE_LIBTPU_LOAD=1
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_DATASETS_CACHE=$SHM/hf-datasets"
  PRE="timeout 600 $PY -c \"import jax; assert (jax.device_count(), jax.process_count()) == (4, 1), (jax.device_count(), jax.process_count()); print('devices', jax.devices())\" || { echo 'preflight failed: not 4 chips of one process'; exit 2; }"
  FLAGS=""
fi
# MaxText 0.2.5 sets XLA_FLAGS to dump every jit_train_step to /tmp/xla_dump whenever XLA_FLAGS is empty, also with
# dump_hlo false (configs/types.py, validate_and_set_hlo_dump_defaults): 3-5 GB per worker on the boot disk, which
# filled it on 9 Nov. A flag that changes nothing keeps it from doing so.
NODUMP="export XLA_FLAGS=--xla_dump_hlo_as_text=false"
cat > ~/run_$NAME.sh <<EOS
set -uo pipefail
$ENV
$NODUMP
$PRE
echo "\$(date +%H:%M:%S) $(hostname): $NAME, $MODEL, $DATA"
$PY -u $CODE/sft_lora.py --out $OUT --steps $STEPS $FLAGS -- $(printf '%q ' "${ARGS[@]}")
EOS
COPY="gsutil -q cp ~/logs/$NAME.log $ROOT/runs/$NAME/$NAME-w$WORKER.log"
# sft_lora.py writes its files on the job's process 0: this worker alone, or worker 0 of a multi-host job
if [ "$MULTIHOST" != 1 ] || [ "$WORKER" = 0 ]; then
  COPY="$COPY; gsutil -m -q cp $OUT/summary.json $OUT/adapter.json $OUT/adapter.npz $OUT/metrics.txt $ROOT/runs/$NAME/"
fi
setsid nohup bash -c "bash ~/run_$NAME.sh > ~/logs/$NAME.log 2>&1; echo EXIT \$? >> ~/logs/$NAME.log; $COPY" > /dev/null 2>&1 < /dev/null &
echo "started $NAME on $(hostname): ~/logs/$NAME.log, output in $OUT"
