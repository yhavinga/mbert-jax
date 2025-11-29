# ModernBERT in JAX/Flax

Pure JAX port of [ModernBERT](https://huggingface.co/answerdotai/ModernBERT-base) (Answer.AI/LightOn). No PyTorch dependency at inference. Matches HuggingFace logits to ~1e-5.

## Why

Training on Dutch text requires full control over the model implementation. PyTorch→JAX gives us:
- XLA compilation for TPU efficiency
- Direct access to attention mechanics (alternating global/local windows, dual RoPE scales)
- Clean unpadding paths for long sequences (>8k tokens)
- JAX's functional paradigm for cleaner training loops

## Status

Core architecture complete. Masked LM head validated against PyTorch reference. Classification/QA heads implemented but need testing. See `PLAN_MMBERT_JAX.md` for detailed roadmap.

**Validated:**
- Token embeddings + LayerNorm
- Alternating global/local attention (sliding window)
- Dual RoPE theta (global: 160k, local: 10k)
- GeGLU MLP with proper bias/dropout flags
- Weight conversion from HuggingFace checkpoints
- Masked language modeling head
- MLM training with AdamW optimizer
- TPU support with async checkpointing

**TODO:**
- Unpadded/FlashAttention execution path
- Multi-device data parallelism (pmap/pjit)
- Mixed precision training (bf16/fp16)

## Quick Start

```bash
# Setup
python -m venv venv
. venv/bin/activate
pip install -r requirements.txt

# Run inference (downloads jhu-clsp/mmBERT-small first time)
python scripts/run_mmbert_inference.py \
  --texts "The cat sat on the <mask>." \
  --remote-files-ok

# Compare JAX vs PyTorch outputs
python scripts/run_mmbert_inference.py \
  --texts "Hello <mask>!" \
  --top-k 3
```

Output shows side-by-side predictions with logit differences (typically <0.0001).

## Structure

```
src/mmbjax/
  configuration_mmbbert.py      # Config matching HF ModernBertConfig
  modeling_mmbbert.py            # Core transformer layers
  modules/                       # Embeddings, RoPE, masks, attention, MLP
  heads/                         # Masked LM, classification, QA
  utils/                         # Weight conversion, initializers

scripts/
  py_reference_dump.py           # Extract PyTorch golden tensors
  run_mmbert_inference.py        # Interactive masked LM demo
  debug_*.py                     # Layer-by-layer numerical validation

tests/                           # Unit tests mirroring src structure
vendor/pytorch_reference/        # Frozen HF transformers for diffing
```

## Design Notes

**No magic.** Every config flag from PyTorch is preserved:
- `attention_bias`, `mlp_bias`, `norm_bias`, `classifier_bias`, `decoder_bias`
- `global_attn_every_n_layers` (layer 0, N, 2N... are global)
- `local_attention` window size
- Per-component dropout rates

**Layer 0 quirk:** Skips attention norm (outputs raw attention). Matches PyTorch exactly.

**RoPE scales:** Global layers use `global_rope_theta=160000`, local layers use `local_rope_theta=10000`. Both are full-dimensional (not partial RoPE).

**GeGLU:** MLP projects to 2×intermediate, splits for gating. Not the same as standard BERT FFN.

**Unpadding:** PyTorch ModernBERT unpads sequences for FlashAttention efficiency. JAX port currently uses padded attention; unpadded path is next milestone.

## Weight Conversion

```python
from mmbjax.utils import convert_pytorch_state_dict_to_flax
import torch

pt_state = torch.load("pytorch_model.bin", map_location="cpu")
flax_params = convert_pytorch_state_dict_to_flax(
    {k: v.numpy() for k, v in pt_state.items()},
    config
)
```

Handles:
- QKV fusion/unfusion
- Weight matrix transposes
- Embedding table sharing (decoder ties to embeddings)
- All bias flags

## Testing

```bash
# Unit tests
pytest tests/

# Numerical validation against PyTorch
python scripts/debug_attention_layer0.py
python scripts/debug_mlp.py
```

Golden tensors in `artifacts/reference-configs/` provide deterministic regression targets.

## Training

Masked language modeling (MLM) training is implemented in `scripts/train_minimal.py`. Includes AdamW optimization, W&B logging, and TPU support.

```bash
# Train on CPU/GPU
python scripts/train_minimal.py \
  --batch_size 16 \
  --num_epochs 3 \
  --learning_rate 1e-4

# Train on TPU with Dutch data
python scripts/train_minimal.py \
  --dataset yhavinga/mc4_nl_cleaned \
  --batch-size 16 \
  --wandb-project mbert_dutch

# Resume from checkpoint
python scripts/train_minimal.py \
  --checkpoint_dir checkpoints/run_123 \
  --resume
```

Features: gradient clipping, linear warmup/decay, comprehensive metrics (perplexity, gradient norms, update ratio), async checkpointing for spot instances.

Model architecture is configurable via `--hidden-size`, `--num-layers`, `--num-heads`, `--intermediate-size`. Run with `--help` for all options.

## License

Code follows ModernBERT's Apache 2.0 license. This is a clean-room port (no copied PyTorch code except for reference comparison).

## Performance Notes

JAX compilation adds ~10-30s startup latency on first run. Subsequent inference is fast. For production, serialize compiled artifacts with `jax.jit` + `static_argnums`.

Long sequences (>2k tokens) currently hit quadratic attention cost. Unpadded path + block-sparse attention will unlock 8k+ context efficiently.

## Contact

Questions about Dutch pre-training setup or architectural choices: see `PLAN_MMBERT_JAX.md` or open an issue.
