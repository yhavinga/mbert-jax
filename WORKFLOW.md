# ModernBERT Training Workflow

## 1. Train (on TPU)

### Single-host (v4-8)
```bash
python scripts/train_minimal.py \
    --checkpoint-dir checkpoints/my_run \
    --wandb-project mbert_dutch
```

### Multi-host (v4-32)
```bash
# Deploy code to all workers
./tpu.sh --tpu-type v4-32 deploy

# Create GCS metadata dir (first time only)
echo "" | gcloud storage cp - gs://your-bucket/my_run/metadata/.placeholder

# Run training
./tpu.sh --tpu-type v4-32 run 'python scripts/train_minimal.py \
    --checkpoint-dir gs://your-bucket/my_run \
    --wandb-project mbert_dutch'

# Monitor
./tpu.sh --tpu-type v4-32 status
```

**Notes:** 
- Multi-host requires GCS (`gs://`) for checkpoints. Local paths don't work.
- Tokenizer handling: Train script adds `<mask>` token (ID=vocab_base) and pads to next 128 multiple.

## 2. Convert

### Option A: On TPU (single-host v4-8 checkpoints)
```bash
# Find latest step
ls checkpoints/my_run/

# Convert
python scripts/convert_checkpoint_on_tpu.py \
    --checkpoint checkpoints/my_run/2000000 \
    --output experiments/converted/my_run
```

### Option B: Locally (multi-host v4-32 GCS checkpoints)
```bash
# Download from GCS
gcloud storage cp -r gs://bucket/checkpoints/STEP experiments/gcs/STEP

# Convert locally (CPU-only, no TPU needed)
# IMPORTANT: --tokenizer is REQUIRED (adds <mask> token and pads vocab for TPU)
python scripts/convert_gcs_checkpoint.py \
    --checkpoint experiments/gcs/STEP \
    --output experiments/converted/my_run \
    --num-layers 22 --hidden-size 512 --vocab-size 32128 \
    --tokenizer yhavinga/dutch-llama-tokenizer
```

## 3. Download (local)
```bash
# Update SSH config
./tpu.sh status

# Copy converted checkpoint
rsync -avz --progress \
    -e "ssh -F ssh_config" \
    "tpu-v4-modernbert:~/mbert-jax/experiments/converted/my_run" \
    experiments/converted/
```

## 4. Evaluate (local)
```bash
python scripts/eval.py \
    --checkpoint experiments/converted/my_run \
    --wandb-name WANDB_RUN_ID
```

## 5. Sync W&B (local)
```bash
# Get run ID from: https://wandb.ai/yepster/mbert_dutch
python scripts/analyze.py wandb sync \
    --run-id WANDB_RUN_ID \
    --project mbert_dutch \
    --checkpoint experiments/converted/my_run
```

## 6. Compare
```bash
python scripts/analyze.py best --top 5
python scripts/analyze.py compare red_1m yellow_2m cosine_2m
```

## 7. Export to HuggingFace (optional)
```bash
# Tokenizer info read from config.json (saved by convert_gcs_checkpoint.py)
# Optional: override with --tokenizer yhavinga/dutch-llama-tokenizer
python scripts/convert_flax_to_hf.py \
    --input experiments/converted/my_run \
    --output experiments/hf_modernbert/my_run

# Upload to HuggingFace Hub
python -c "from huggingface_hub import HfApi; \
HfApi().upload_folder('experiments/hf_modernbert/my_run', '<username>/<model-name>')"
```

---

## Analysis Tools

```bash
# Checkpoint info
python scripts/analyze.py checkpoint experiments/converted/my_run

# Stability analysis (parameter evolution between checkpoints)
python scripts/analyze.py stability experiments/converted/cp1 experiments/converted/cp2

# Spectral analysis (conditioning, eigenvalues)
python scripts/analyze.py spectral experiments/converted/my_run

# Full report
python scripts/analyze.py report experiments/converted/my_run --output report.txt
```

## Database
Results stored in `evaluations.db`. Query directly:
```bash
sqlite3 evaluations.db "SELECT * FROM training_runs;"
sqlite3 evaluations.db "SELECT checkpoint_name, perplexity FROM evaluations ORDER BY perplexity;"
```
