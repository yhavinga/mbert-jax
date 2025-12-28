#!/usr/bin/env python3
"""Upload a converted HuggingFace model to the HuggingFace Hub.

Usage:
    python scripts/upload_to_hf.py \
        --model-path experiments/hf_modernbert/dutchl-1024h-22l-2-2000000 \
        --repo-id yhavinga/modernbert-dutch-base-wide
"""

import argparse
from pathlib import Path
from huggingface_hub import HfApi, create_repo


MODEL_CARD_TEMPLATE = """---
license: apache-2.0
base_model: ModernBERT
tags:
- modernbert
- dutch
- masked-language-model
- mc4
- pretrained
- jax
- flax
library_name: transformers
---

# ModernBERT Dutch Base Wide

A ModernBERT model pretrained on Dutch mc4_nl_cleaned dataset. This model has 22 layers (like ModernBERT-base) but with wider hidden dimensions (1024 instead of 768), placing it between base and large in terms of parameters (230M).

## Model Details

- **Architecture**: ModernBERT (Answer.AI/LightOn)
- **Layers**: 22
- **Hidden size**: 1024
- **Attention heads**: 16
- **Intermediate size**: 1536
- **Vocab size**: 32,128
- **Parameters**: 230M
- **Tokenizer**: `yhavinga/dutch-llama-tokenizer` (SentencePiece, Dutch-optimized)

## Training

- **Dataset**: `yhavinga/mc4_nl_cleaned` (full config)
- **Steps**: 2,000,000
- **Batch size**: 8 per device (multi-host TPU v4)
- **Learning rate**: 3e-5 with cosine decay to 1e-6
- **Warmup steps**: 20,000
- **Weight decay**: 0.01
- **Sequence length**: 1024
- **Precision**: bfloat16

## Usage

```python
from transformers import AutoTokenizer, ModernBertForMaskedLM

model = ModernBertForMaskedLM.from_pretrained("yhavinga/modernbert-dutch-base-wide")
tokenizer = AutoTokenizer.from_pretrained("yhavinga/modernbert-dutch-base-wide")

# Masked language modeling
inputs = tokenizer("Amsterdam is de<mask> van Nederland.", return_tensors="pt")
outputs = model(**inputs)
predictions = tokenizer.decode(outputs.logits[0, 4].topk(5).indices[0])
# Expected: "hoofdstad" (capital)
```

## Model Architecture Differences

This model (`1024h-22L-2`) differs from the earlier `1024h-22L` variant:

| Parameter | 1024h-22L | 1024h-22L-2 (this model) |
|-----------|-----------|--------------------------|
| `intermediate_size` | 4096 | **1536** |
| `tokenizer` | `jhu-clsp/mmBERT-small` | **`yhavinga/dutch-llama-tokenizer`** |
| `vocab_size` | 256,000 | **32,128** |

The smaller intermediate MLP size and Dutch-specific tokenizer make this model more efficient while maintaining strong Dutch language understanding.

## Citation

If you use this model, please cite:

```bibtex
@model{modernbert_dutch_base_wide,
  title={ModernBERT Dutch Base Wide},
  author={havinga, y havinga},
  year={2025},
  publisher={Hugging Face},
  url={https://huggingface.co/yhavinga/modernbert-dutch-base-wide}
}
```
"""


def upload_model(model_path: str, repo_id: str, create: bool = True):
    """Upload model to HuggingFace Hub."""
    api = HfApi()
    model_path = Path(model_path)

    print(f"Uploading {model_path} to {repo_id}")

    # Create repo if requested
    if create:
        try:
            create_repo(repo_id, exist_ok=True, repo_type="model")
            print(f"✓ Created/found repo {repo_id}")
        except Exception as e:
            print(f"Note: {e}")

    # Create model card
    readme_path = model_path / "README.md"
    if not readme_path.exists():
        with open(readme_path, "w") as f:
            f.write(MODEL_CARD_TEMPLATE)
        print(f"✓ Created README.md at {readme_path}")

    # Upload folder
    api.upload_folder(
        folder_path=str(model_path),
        repo_id=repo_id,
        repo_type="model",
        commit_message="Upload ModernBERT Dutch Base Wide (2M steps, mc4_nl_cleaned)"
    )
    print(f"✓ Uploaded to https://huggingface.co/{repo_id}")


def main():
    parser = argparse.ArgumentParser(description="Upload model to HuggingFace Hub")
    parser.add_argument("--model-path", "-m", required=True,
                        help="Path to converted HuggingFace model")
    parser.add_argument("--repo-id", "-r", required=True,
                        help="HuggingFace repo ID (e.g., username/model-name)")
    parser.add_argument("--no-create", action="store_true",
                        help="Don't create repo if it doesn't exist")

    args = parser.parse_args()

    upload_model(args.model_path, args.repo_id, create=not args.no_create)


if __name__ == "__main__":
    main()
