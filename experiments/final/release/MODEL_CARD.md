---
language:
- nl
license: apache-2.0
library_name: transformers
pipeline_tag: fill-mask
datasets:
- uonlp/CulturaX
widget:
- text: "Amsterdam is de<mask>van Nederland."
tags:
- modernbert
- dutch
- masked-language-model
- encoder
- culturax
- jax
---

# dmbert-230m-culturax

A Dutch [ModernBERT](https://arxiv.org/abs/2412.13663) encoder with 230M parameters, pretrained from scratch with
masked language modelling on the Dutch part of [CulturaX](https://huggingface.co/datasets/uonlp/CulturaX): about 161
billion tokens, two passes over the corpus, in sequences of up to 1,024 tokens. Trained with JAX/Flax on a Cloud TPU
v4-32. This is a base model: fine-tune it for classification, token classification, extractive question answering or
retrieval.

## Usage

### Fill-mask

```python
from transformers import pipeline

fill = pipeline("fill-mask", model="yhavinga/dmbert-230m-culturax")
for r in fill("Amsterdam is de<mask>van Nederland.", top_k=5):
    print(r["token_str"], round(r["score"], 3))
```

This prints `hoofdstad` (0.996), `stad` (0.004), `grootste`, `hoofdplaats` and `gemeente`.

**How to write `<mask>`.** In this tokenizer a word carries its leading space (`▁hoofdstad`), and the text after a
special token such as `<mask>` is tokenized as if it starts a new text, with a `▁` in front. Written without spaces
around it, as above (`de<mask>van`), the input ids are exactly those of the pretraining, where `<mask>` replaced the
token `▁hoofdstad` of "de hoofdstad van". With spaces (`de <mask> van`) the ids contain an extra lone `▁` before the
mask and `▁▁` + `van` after it, which the model did not see in pretraining: the pipeline then ranks digits first and
`hoofdstad` 31st. When punctuation follows the mask (`kopje<mask>.`), the tokenizer still puts a `▁`
before the punctuation; to get exactly the pretraining form, mask a token at id level:

```python
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("yhavinga/dmbert-230m-culturax")
model = AutoModelForMaskedLM.from_pretrained("yhavinga/dmbert-230m-culturax")
ids = tok("Ik drink elke ochtend een kopje koffie.")["input_ids"]
ids[tok.convert_ids_to_tokens(ids).index("▁koffie")] = tok.mask_token_id
with torch.no_grad():
    logits = model(input_ids=torch.tensor([ids])).logits[0, ids.index(tok.mask_token_id)]
print(tok.convert_ids_to_tokens(logits.topk(5).indices))
```

### Fine-tuning

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tok = AutoTokenizer.from_pretrained("yhavinga/dmbert-230m-culturax")
model = AutoModelForSequenceClassification.from_pretrained("yhavinga/dmbert-230m-culturax", num_labels=2)
batch = tok(["Een prachtig boek.", "Wat een saai verhaal, ik heb het niet uitgelezen."],
            padding=True, truncation=True, max_length=1024, return_tensors="pt")
loss = model(**batch, labels=torch.tensor([1, 0])).loss
loss.backward()
```

The tokenizer truncates from the left (`truncation_side="left"`): with `truncation=True`, the end of a long text is
kept. Our evaluation below used this setting. Set `tok.truncation_side = "right"` to keep the beginning instead.
Pass `max_length` (at most 1,024, see Limitations): the tokenizer itself sets no maximum length.

## Model details

| | |
|---|---|
| Architecture | ModernBERT: pre-norm Transformer encoder, rotary position embeddings, GeGLU, no bias terms in the encoder, alternating local and global attention |
| Layers / hidden size / heads / intermediate size | 22 / 1,024 / 16 (head size 64) / 1,536 |
| Attention | global in every third layer (layers 0, 3, …, 21), the other layers a sliding window of 128 tokens |
| RoPE base | 160,000 (global layers), 10,000 (local layers) |
| Parameters | 230,110,592. The output layer is tied to the input embeddings; the file stores that matrix a second time, so the Hub's parameter count shows ~263M. |
| Masked-LM head | dense layer, GELU and LayerNorm, then the tied decoder with a bias |
| Pretraining sequence length | 1,024 tokens. `max_position_embeddings` (8,192) is the architecture's limit, not a trained length. |
| Weights | float32, safetensors |

## Tokenizer

[`yhavinga/dutch-llama-tokenizer`](https://huggingface.co/yhavinga/dutch-llama-tokenizer): a SentencePiece BPE
tokenizer (Llama type, `LlamaTokenizerFast`) with byte fallback and 32,000 tokens, extended with `<mask>` (id 32,000)
and 127 unused tokens to 32,128 entries, the size of the embedding matrix.

| token | id |
|---|---|
| `<unk>` | 0 |
| `<s>` (BOS, CLS) | 1 |
| `</s>` (EOS, SEP) | 2 |
| `<pad>` | 20,277 |
| `<mask>` | 32,000 |

- A single text is encoded as `<s> text` (no `</s>`), a pair as `<s> A <s> B`. Our sentence-pair evaluation
  (SICK-NL) used this default.
- In pretraining every document was tokenized without special tokens and cut into pieces of at most 1,024 tokens;
  every piece starts with `<s>`, and the piece with the end of the document ends with `</s>`.
- **Character offsets** (`return_offsets_mapping=True`) are exact: for every token, `text[start:end]` is the token's text with `▁` read as a space.
  A word after a space therefore has a span that begins at that space (`▁hoofdstad` in "De hoofdstad" covers
  " hoofdstad"); the first word of a text, and of each sequence of a pair, begins at the word itself. Byte-fallback
  tokens of one character share that character's span. Strip the space when you need the span of the word alone.
  Checked on 12 texts (455 tokens, with punctuation, accents, emoji, runs of whitespace and a literal `<mask>`) and on
  the overlapping windows of a question–context pair (`return_overflowing_tokens`).

## Training data

The Dutch part (config `nl`) of [`uonlp/CulturaX`](https://huggingface.co/datasets/uonlp/CulturaX), revision
`6a8734bc69fefcbb7735f4f9250f43e4cd7a442e`: 117,392,666 documents, 80.7 billion tokens with this tokenizer. CulturaX
combines mC4 (version 3.1.0) and the OSCAR corpora 20.19 to 23.01, cleaned and deduplicated by its authors
([Nguyen et al., 2024](https://arxiv.org/abs/2309.09400)) with "language identification, URL-based filtering,
metric-based cleaning, document refinement, and data deduplication" (from the dataset card).

One filter of our own: every document whose URL host is `hebban.nl` or a subdomain of it was removed before
shuffling, because the book reviews of the DBRD benchmark come from that site. That removed 45,781 documents (all
from `www.hebban.nl`, 35.6M tokens), leaving 117,346,885 documents. There was no other filtering or deduplication.

## Training

| | |
|---|---|
| Hardware | Cloud TPU v4-32 (16 chips, 4 hosts), data parallel |
| Objective | masked language modelling: 15% of the non-special tokens, of which 80% replaced by `<mask>`, 10% by a random token, 10% unchanged |
| Batch | 128 sequences of 1,024 positions (131,072 per step), packed: several documents per sequence with block-diagonal attention (no attention between documents) and positions restarting at each document; ~94.6% of the positions are real tokens |
| Steps | 1,306,000, about 161 billion tokens: two passes over the corpus, each document twice and in a different order in the second pass (the last ~760 steps begin a third pass) |
| Optimizer | AdamW (β1 0.9, β2 0.999, ε 1e-8), weight decay 0.01 (not on norms and biases), gradient clipping at global norm 1.0 |
| Learning rate | warmup-stable-decay: linear warmup over 1,250 steps to 4.8e-4, constant, then a cooldown of the form 1 − √(fraction of the cooldown) to 1e-6 over the last 266,000 steps (from step 1,040,000) ([Hägele et al., 2024](https://arxiv.org/abs/2405.18392)) |
| Precision | bfloat16 compute with float32 parameters, optimizer state and logits. Steps 1,010,000 to 1,020,000 were computed in float32: at step 1,012,690 bfloat16 gave non-finite gradients for 57 steps in a row, and training resumed from step 1,010,000 in float32, then continued in bfloat16 |
| Non-finite gradients | a step whose gradients were not finite left parameters and optimizer state unchanged (`optax.apply_if_finite`): 1,188 of the 1,306,000 steps (0.09%), all between steps 173,540 and 829,350 |
| Sequence length | 1,024 tokens; longer documents were split into pieces, not truncated |
| Code | [github.com/yhavinga/mbert-jax](https://github.com/yhavinga/mbert-jax), branch `final-run` (`scripts/train_minimal.py`; protocol in `experiments/final/protocol.md`) |

## Evaluation

All models below went through the same fine-tuning and evaluation code with the same data splits, seeds and
hyperparameters; nothing was tuned per model. [RobBERT-2023-base](https://huggingface.co/DTAI-KULeuven/robbert-2023-dutch-base)
(124.5M parameters, revision `1164b2d`) and [RobBERT-2026](https://huggingface.co/pdelobelle/robbert-2026) (275.1M
parameters, revision `a406b1a`) were run as JAX re-implementations whose masked-LM logits match their PyTorch code.
Each fine-tuned number is the mean over three seeds (42, 43, 44) with the range (max − min) in brackets, in
percentage points, on the test split after the last epoch (no early stopping, no checkpoint selection).

| Task | Metric | dmbert-230m-culturax (this model) | RobBERT-2023-base | RobBERT-2026 |
|---|---|---:|---:|---:|
| DBRD, 512 tokens | accuracy | 96.05 [0.52] | 94.93 [0.80] | 93.15 [3.03] |
| DBRD, 1,024 tokens | accuracy | 96.12 [0.28] | — | — |
| SICK-NL | accuracy | **87.23** [0.76] | 83.95 [0.72] | 78.32 [0.95] |
| SQuAD-NL v1.1 | F1 | 84.74 [0.60] | 84.43 [0.64] | 76.99 [0.91] |
| SQuAD-NL v1.1 | exact match | 72.70 [1.18] | **73.94** [0.59] | 64.92 [2.03] |
| POS, UD Dutch LassySmall | UPOS accuracy | 97.90 [0.26] | 97.91 [0.09] | 97.62 [0.24] |
| POS, UD Dutch Alpino | UPOS accuracy | 97.71 [0.52] | 97.50 [0.40] | 97.31 [0.43] |
| BLiMP-NL (zero-shot) | accuracy | **94.94** | 83.74 | 82.23 |

Bold: higher than each of the other two models by more than the larger seed range of the pair (BLiMP-NL: by more
than 0.5 points). DBRD at 1,024 tokens only for this model: RobBERT-2023 accepts at most 512 tokens, RobBERT-2026 was
pretrained on 512 tokens.

**Tasks and data.**

- **DBRD** ([van der Burgh & Verberne, 2019](https://arxiv.org/abs/1910.00896)): sentiment of book reviews, binary.
  Texts truncated from the left to 512 or 1,024 tokens (the end of a review is kept). 2,112 of the 2,224 test
  reviews are scored (whole evaluation batches).
- **SICK-NL** ([Wijnholds & Moortgat, 2021](https://aclanthology.org/2021.eacl-main.126/)): natural language
  inference, three classes, as a sentence pair. 4,864 of the 4,906 test pairs are scored.
- **SQuAD-NL v1.1** ([`GroNLP/squad-nl-v1.1`](https://huggingface.co/datasets/GroNLP/squad-nl-v1.1)): extractive
  question answering, answerable questions only, 1,183 test questions; windows of 384 tokens with a stride of 128;
  EM and F1 with the official SQuAD v1.1 normalization. This is not DUMB's SQuAD-NL v2.0.
- **POS**: UPOS tagging on [UD Dutch LassySmall](https://github.com/UniversalDependencies/UD_Dutch-LassySmall) and
  [UD Dutch Alpino](https://github.com/UniversalDependencies/UD_Dutch-Alpino) (release 2.18), accuracy over all words
  of the test split, label at each word's first token.
- **BLiMP-NL** ([Suijkerbuijk et al., 2025](https://doi.org/10.34973/tj4p-y007)): 8,400 minimal pairs of 84
  paradigms, scored without fine-tuning with the masked-LM head: a pair counts as right when the grammatical sentence
  has the higher pseudo-log-likelihood (PLL-word-l2r of Kauf & Ivanova, 2023, the metric of the BLiMP-NL paper).

**Fine-tuning hyperparameters**, the same for every model:

| | DBRD, SICK-NL | SQuAD-NL, POS |
|---|---|---|
| Epochs | 3 | 3 |
| Learning rate | 5e-5, constant (no warmup, no decay) | 5e-5, linear decay to 0, no warmup |
| Batch (examples per update) | DBRD 192, SICK-NL 256 | 32 |
| Optimizer | AdamW, weight decay 0.01 | AdamW, weight decay 0.01 |
| Max length | DBRD 512 or 1,024, SICK-NL 512 | SQuAD-NL 384 (stride 128), POS 512 |
| Precision | float32 | float32 |
| Head (this model) | the pretrained head transformation (dense, GELU, LayerNorm) on `<s>`, then a new linear classifier | the pretrained head transformation per token, then a new linear layer (start and end for SQuAD-NL) |

The RobBERT models get the task heads of their own Hugging Face classes. These hyperparameters were not tuned for
any model, and they are not the settings the RobBERT authors used for their
own published results, so the RobBERT numbers here differ from theirs.

<details>
<summary>RobBERT-2023 needed an offset correction for SQuAD-NL</summary>

Its fast tokenizer returns word offsets that start one character late (`ĠDe` → "e"). Our evaluation turns off
`trim_offsets` in its post-processor: same token ids, correct offsets. Without the correction RobBERT-2023 scores
F1 68.21 / EM 14.12 on SQuAD-NL (seed 42) instead of 84.08 / 73.63. RobBERT-2026 and the tokenizer of this model are
not affected.

</details>

## Limitations

- Pretrained on sequences of at most 1,024 tokens. The rotary embeddings accept longer inputs, but the model has not
  been trained or evaluated on them.
- Trained on web text only (CulturaX: mC4 and OSCAR, both from Common Crawl). It reflects the content, opinions and
  biases of the Dutch web, and the corpus can contain personal information.
- Dutch only; other languages were not part of the training data.
- `hebban.nl` was excluded to keep DBRD out of the training data. We did not check the other evaluation sets for
  overlap with CulturaX.
- No evaluation of social bias or safety.

## License

Apache 2.0, for the model weights, configuration and tokenizer files in this repository. The tokenizer
`yhavinga/dutch-llama-tokenizer` is also Apache 2.0. The training data, CulturaX, follows the licences of mC4
([ODC-BY](https://opendatacommons.org/licenses/by/1-0/), with the [Common Crawl terms of
use](https://commoncrawl.org/terms-of-use)) and OSCAR (CC0 for the packaging; the texts belong to their authors);
the model does not contain or redistribute the corpus.

## Acknowledgements

Research supported with Cloud TPUs from Google's [TPU Research Cloud (TRC)](https://sites.research.google/trc/about/).

## Citation

```bibtex
@misc{dmbert_230m_culturax,
  title  = {dmbert-230m-culturax: a Dutch ModernBERT encoder},
  author = {Yeb Havinga},
  year   = {2026},
  publisher = {Hugging Face},
  url    = {https://huggingface.co/yhavinga/dmbert-230m-culturax}
}
```
