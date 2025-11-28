# ModernBERT Training Workflow

## 1. Train (on TPU)
```bash
python scripts/train_minimal.py \
    --checkpoint-dir checkpoints/my_run \
    --wandb-project mbert_dutch
```

## 2. Convert (on TPU)
```bash
# Find latest step
ls checkpoints/my_run/

# Convert
python scripts/convert_checkpoint_on_tpu.py \
    --checkpoint checkpoints/my_run/2000000 \
    --output experiments/converted/my_run
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
python scripts/convert_flax_to_hf.py \
    --input experiments/converted/my_run \
    --output experiments/hf_modernbert/my_run

# Load with transformers:
# ModernBertForMaskedLM.from_pretrained("experiments/hf_modernbert/my_run")
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
