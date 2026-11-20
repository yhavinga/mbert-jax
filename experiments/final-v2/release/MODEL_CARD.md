---
language:
- nl
license: apache-2.0
library_name: transformers
pipeline_tag: fill-mask
datasets:
- HuggingFaceFW/fineweb-2
widget:
- text: "Amsterdam is de<mask>van Nederland."
tags:
- modernbert
- dutch
- masked-language-model
- encoder
- fineweb-2
- jax
---

# dmbert-230m-fineweb2-optout

A Dutch [ModernBERT](https://arxiv.org/abs/2412.13663) encoder with 230M parameters, pretrained from scratch with
masked language modelling on the Dutch part of [FineWeb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2),
after removing the documents of sites whose machine-readable reservation of text and data mining our check found in
November 2026, and after replacing more kinds of personal data by placeholders: about 163.3 billion tokens, about 2.2
passes over the corpus, in sequences of up to 1,024 tokens. Trained with JAX/Flax on a Cloud TPU v4-32. This is a
base model: fine-tune it for classification, token classification, extractive question answering or retrieval.

## Related model

[dmbert-230m-culturax](https://huggingface.co/yhavinga/dmbert-230m-culturax) has the same architecture, tokenizer and
training recipe, and was pretrained on the Dutch part of CulturaX, which was not filtered on reservations of text and
data mining. Both models are in the evaluation and the memorization test below.

## Usage

### Fill-mask

```python
from transformers import pipeline

fill = pipeline("fill-mask", model="yhavinga/dmbert-230m-fineweb2-optout")
for r in fill("Amsterdam is de<mask>van Nederland.", top_k=5):
    print(r["token_str"], round(r["score"], 3))
```

This prints `hoofdstad` (0.998), `stad` (0.001), `grootste`, `bekendste` and `belangrijkste`.

**How to write `<mask>`.** In this tokenizer a word carries its leading space (`▁hoofdstad`), and the text after a
special token such as `<mask>` is tokenized as if it starts a new text, with a `▁` in front. Written without spaces
around it, as above (`de<mask>van`), the input ids are exactly those of the pretraining, where `<mask>` replaced the
token `▁hoofdstad` of "de hoofdstad van". With spaces (`de <mask> van`) the ids contain an extra lone `▁` before the
mask and `▁▁` + `van` after it, which the model did not see in pretraining: the pipeline then ranks digits first and
`hoofdstad` 32nd. When punctuation follows the mask (`kopje<mask>.`), the tokenizer still puts a `▁` before the
punctuation; to get exactly the pretraining form, mask a token at id level:

```python
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("yhavinga/dmbert-230m-fineweb2-optout")
model = AutoModelForMaskedLM.from_pretrained("yhavinga/dmbert-230m-fineweb2-optout")
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

tok = AutoTokenizer.from_pretrained("yhavinga/dmbert-230m-fineweb2-optout")
model = AutoModelForSequenceClassification.from_pretrained("yhavinga/dmbert-230m-fineweb2-optout", num_labels=2)
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
- **Character offsets** (`return_offsets_mapping=True`) are exact: for every token, `text[start:end]` is the token's
  text with `▁` read as a space. A word after a space therefore has a span that begins at that space (`▁hoofdstad` in
  "De hoofdstad" covers " hoofdstad"); the first word of a text, and of each sequence of a pair, begins at the word
  itself. Byte-fallback tokens of one character share that character's span. Strip the space when you need the span
  of the word alone. Checked on 12 texts (455 tokens, with punctuation, accents, emoji, runs of whitespace and a
  literal `<mask>`) and on the overlapping windows of a question–context pair (`return_overflowing_tokens`).

## Training data

The Dutch part (config `nld_Latn`) of [`HuggingFaceFW/fineweb-2`](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2),
revision `af9c13333eb981300149d5ca60a8e9d659b276b9`, train split, limited to the Common Crawl snapshots up to and
including CC-MAIN-2023-23 (the earliest is CC-MAIN-2013-20). FineWeb-2 is built from Common Crawl by its authors with
language identification, filtering and deduplication; in their words, "data is deduplicated per language, globally".
Of its train split, 126,113,527 documents are from these snapshots. The filters below left **100,687,582 documents,
73.4 billion tokens** with this tokenizer, the training data of this model. FineWeb-2's test split of `nld_Latn`,
filtered the same way (22,591 documents, none of them also in the training data), was not trained on.

The filters below, with their tests, are in
[github.com/yhavinga/fineweb2-nl-optout](https://github.com/yhavinga/fineweb2-nl-optout), commit
[`ecb4b68`](https://github.com/yhavinga/fineweb2-nl-optout/tree/ecb4b685581ba702762d527cae01f97730d8b9a3).

### Opt-outs

A document was left out when, at the time of our check:

- its URL path (with query) is disallowed in the site's robots.txt (RFC 9309) for `*` or for any of 198 user agents of AI
  crawlers and assistants: 31 named in our rule (among them GPTBot, ChatGPT-User, CCBot, Google-Extended, ClaudeBot,
  anthropic-ai, PerplexityBot, Bytespider, Applebot-Extended, meta-externalagent, cohere-ai and Amazonbot) and the
  other agents of the [ai.robots.txt](https://github.com/ai-robots-txt/ai.robots.txt) list at commit `795e897`;
- the robots.txt group for `*` or for one of these agents says `Content-Signal: ai-train=no`, or has a
  `Content-Usage` line that withholds training for its path;
- the site's `/.well-known/tdmrep.json` (W3C TDMRep) reserves its path (`tdm-reservation` 1), or the site's `/ai.txt`
  disallows its path, or text, for `*` or one of these agents;
- its host is in the Apertus robots.txt blocklist
  ([`swiss-ai/robots-txt-blocked-domains-multilingual`](https://huggingface.co/datasets/swiss-ai/robots-txt-blocked-domains-multilingual))
  or in the [Common Crawl Opt-Out Ledger](https://commoncrawl.org/blog/common-crawl-foundation-opt-out-registry);
- its host is on a list of 725 domains (and 24 name labels such as `thepiratebay.*`) of infringing sites from 33
  public sources: the EU Counterfeit and Piracy Watch List, the USTR Review of Notorious Markets, Dutch and Belgian
  blocking orders and blocking lists, and court documents.

**When.** robots.txt, `tdmrep.json` and `ai.txt` were fetched once per host (3,199,045 hosts) between 2026-11-14
15:49 and 2026-11-15 02:44 UTC from Google Cloud; hosts without a usable answer were asked again at least 30 minutes
later, until 2026-11-15 12:47 UTC. Hosts whose robots.txt then still gave an HTTP error other than 404, 410 and 429
(mostly 403, often from bot protection that blocks the addresses of cloud providers) were asked for robots.txt a third
time, from a server outside Google Cloud, on 2026-11-15 between 14:04 and 17:45 UTC; 45,489 of these hosts (87,903
documents) were not asked again. The blocklists are snapshots of 2026-11-14.

**What it removed.** Of the documents from the snapshots above, 79.8% of the documents and 82.3% of the tokens were
kept (both splits). Most documents were removed by robots.txt rules for AI crawlers; among the sites with the most
documents removed are Dutch and Flemish news publishers.

**What it did not remove.** A host whose robots.txt could not be read (no usable answer in two attempts, or an HTTP
error other than 404 and 410, also on the third attempt) counts as having no known reservation; its documents were
removed only when another reason held. 22,828,343 of the kept documents (22.7%, both splits) are from such hosts:
18,629,857 from unreachable hosts (838,013 of the 994,068 unreachable hosts gave no DNS answer) and 4,198,486 from
hosts whose robots.txt answered an error. A host whose robots.txt answered 404 or 410 has no robots.txt and no
restriction. Not checked: reservations in the HTTP headers or HTML meta tags of individual pages, reservations in
natural language (terms of use), and reservations made after our check. The rule works per host: text of a
publisher with a reservation that another site copied stays in when that site has no reservation.

### Personal data

FineWeb-2 replaces e-mail addresses and public IP addresses ("We anonymize email addresses and public IP addresses",
from its card). In addition, in the text of every document, we replaced IBANs (length and mod-97 check), payment card
numbers (scheme prefix, length and Luhn check), Dutch and Belgian phone numbers, Dutch citizen service numbers (BSN,
with its check digit) and Dutch postcodes with a house number by fixed placeholders that are invalid by construction:
`NL00 BANK 0000 0000 00`, `0000 0000 0000 0000`, `000-0000000`, `000000000`, and `0000 XX` with house number `0`. In
the kept documents (both splits) that replaced 5,910,083 phone numbers, 2,940,230 postcodes with a house number,
220,125 IBANs, 52,707 BSNs and 14,521 card numbers; 5,431,005 documents (5.4%) had at least one replacement. Names
and other personal data in running text were not replaced, and the patterns can also replace numbers that are not
personal data.

### Benchmark data

Documents whose URL host is `hebban.nl` or a subdomain of it, the site of the book reviews of the DBRD benchmark,
were blocked by URL during training; none of them remained after the filters above. We did not check the other
evaluation sets for overlap with the training data.

## Training

| | |
|---|---|
| Hardware | Cloud TPU v4-32 (16 chips, 4 hosts), data parallel |
| Objective | masked language modelling: 15% of the non-special tokens, of which 80% replaced by `<mask>`, 10% by a random token, 10% unchanged |
| Batch | 128 sequences of 1,024 positions (131,072 per step), packed: several documents per sequence with block-diagonal attention (no attention between documents) and positions restarting at each document; ~95.4% of the positions are real tokens |
| Steps | 1,306,000, about 163.3 billion tokens: about 2.2 passes over the corpus, each pass in a different order |
| Optimizer | AdamW (β1 0.9, β2 0.999, ε 1e-8), weight decay 0.01 (not on norms and biases), gradient clipping at global norm 1.0 |
| Learning rate | warmup-stable-decay: linear warmup over 1,250 steps to 4.8e-4, constant, then a cooldown of the form 1 − √(fraction of the cooldown) to 1e-6 over the last 266,000 steps (from step 1,040,000) ([Hägele et al., 2024](https://arxiv.org/abs/2405.18392)) |
| Precision | float32 parameters, optimizer state and logits throughout. Compute in bfloat16 for steps 1–250,000, 270,001–280,000 and 320,001–340,000, and in float32 for steps 250,001–270,000, 280,001–320,000 and 340,001–1,306,000. In bfloat16, stretches of steps with non-finite gradients began at about steps 252,700, 281,800 and 344,300; each time, training resumed from the last checkpoint before the stretch (250,000, 280,000, 340,000) in float32, which computed the same data without a non-finite step; after the third stretch it stayed in float32 to the end. |
| Non-finite gradients | a step whose gradients were not finite left parameters and optimizer state unchanged (`optax.apply_if_finite`): 34 of the 1,306,000 steps that made this model (0.0026%), all computed in bfloat16, between steps 196,680 and 275,570 |
| Sequence length | 1,024 tokens; longer documents were split into pieces, not truncated |
| Code | [github.com/yhavinga/mbert-jax](https://github.com/yhavinga/mbert-jax), commit [`494d4d9`](https://github.com/yhavinga/mbert-jax/commit/494d4d94b4df9c45d92e9b6ce6f77dfe7d4130a5) (`scripts/train_minimal.py`; the protocol of this run is in `experiments/` at that commit) |

Estimated training compute: about 2.3·10^20 FLOP (6 × parameters × tokens), well below the indicative threshold of
10^23 FLOP in the European Commission's guidelines on general-purpose AI models.

## Evaluation

All models below went through the same fine-tuning and evaluation code with the same data splits, seeds and
hyperparameters; nothing was tuned per model. dmbert-230m-culturax is the related model above.
[RobBERT-2023-base](https://huggingface.co/DTAI-KULeuven/robbert-2023-dutch-base) (124.5M parameters, revision
`1164b2d`) was run as a JAX re-implementation whose masked-LM logits match its PyTorch code. Each fine-tuned number
is the mean over three seeds (42, 43, 44) with the range (max − min) in brackets, in percentage points, on the test
split after the last epoch (no early stopping, no checkpoint selection).

| Task | Metric | dmbert-230m-culturax | dmbert-230m-fineweb2-optout (this model) | RobBERT-2023-base |
|---|---|---:|---:|---:|
| DBRD, 1,024 tokens | accuracy | 96.12 [0.28] | 95.86 [1.14] | — |
| DBRD, 512 tokens | accuracy | 96.05 [0.52] | 96.02 [0.90] | 94.93 [0.80] |
| SICK-NL | accuracy | **87.23** [0.76] | 84.19 [1.64] | 83.95 [0.72] |
| SQuAD-NL v1.1 | F1 | 84.74 [0.60] | 82.79 [1.33] | 84.43 [0.64] |
| SQuAD-NL v1.1 | exact match | 72.70 [1.18] | 69.85 [0.42] | **73.94** [0.59] |
| CoNLL-2002 | entity F1 | **91.86** [0.49] | 90.12 [0.29] | 90.47 [0.71] |
| POS, UD Dutch LassySmall | UPOS accuracy | 97.90 [0.26] | 97.58 [0.08] | 97.91 [0.09] |
| POS, UD Dutch Alpino | UPOS accuracy | 97.71 [0.52] | 97.38 [0.16] | 97.50 [0.40] |
| BLiMP-NL (zero-shot) | accuracy | 94.94 | 94.76 | 83.74 |

Bold: higher than each of the other models in its row by more than the larger seed range of the pair (BLiMP-NL: by
more than 0.5 points). DBRD at 1,024 tokens only for the dmbert models: RobBERT-2023 accepts at most 512 tokens.

**Tasks and data.**

- **DBRD** ([van der Burgh & Verberne, 2019](https://arxiv.org/abs/1910.00896)): sentiment of book reviews, binary.
  Texts truncated from the left to 1,024 or 512 tokens (the end of a review is kept). 2,112 of the 2,224 test
  reviews are scored (whole evaluation batches).
- **SICK-NL** ([Wijnholds & Moortgat, 2021](https://aclanthology.org/2021.eacl-main.126/)): natural language
  inference, three classes, as a sentence pair. 4,864 of the 4,906 test pairs are scored.
- **SQuAD-NL v1.1** ([`GroNLP/squad-nl-v1.1`](https://huggingface.co/datasets/GroNLP/squad-nl-v1.1)): extractive
  question answering, answerable questions only, 1,183 test questions; windows of 384 tokens with a stride of 128;
  EM and F1 with the official SQuAD v1.1 normalization. This is not DUMB's SQuAD-NL v2.0.
- **CoNLL-2002** ([Tjong Kim Sang, 2002](https://aclanthology.org/W02-2024/)): named entity recognition on the Dutch
  data (persons, organizations, locations and miscellaneous names; IOB2 tags), trained on `ned.train` and tested on
  `ned.testb` (5,195 sentences, 3,941 entities); micro-averaged F1 over entities, where a predicted entity counts when
  its type and both boundaries are right; label at each word's first token.
- **POS**: UPOS tagging on [UD Dutch LassySmall](https://github.com/UniversalDependencies/UD_Dutch-LassySmall) and
  [UD Dutch Alpino](https://github.com/UniversalDependencies/UD_Dutch-Alpino) (release 2.18), accuracy over all words
  of the test split, label at each word's first token.
- **BLiMP-NL** ([Suijkerbuijk et al., 2025](https://doi.org/10.34973/tj4p-y007)): 8,400 minimal pairs of 84
  paradigms, scored without fine-tuning with the masked-LM head: a pair counts as right when the grammatical sentence
  has the higher pseudo-log-likelihood (PLL-word-l2r of Kauf & Ivanova, 2023, the metric of the BLiMP-NL paper).

**Fine-tuning hyperparameters**, the same for every model:

| | DBRD, SICK-NL | SQuAD-NL, CoNLL-2002, POS |
|---|---|---|
| Epochs | 3 | 3 |
| Learning rate | 5e-5, constant (no warmup, no decay) | 5e-5, linear decay to 0, no warmup |
| Batch (examples per update) | DBRD 192, SICK-NL 256 | 32 |
| Optimizer | AdamW, weight decay 0.01 | AdamW, weight decay 0.01 |
| Max length | DBRD 1,024 or 512, SICK-NL 512 | SQuAD-NL 384 (stride 128), CoNLL-2002 and POS 512 |
| Precision | float32 | float32 |
| Head (dmbert models) | the pretrained head transformation (dense, GELU, LayerNorm) on `<s>`, then a new linear classifier | the pretrained head transformation per token, then a new linear layer (start and end for SQuAD-NL) |

RobBERT-2023 gets the task heads of its own Hugging Face classes. These hyperparameters were not tuned for any model,
and they are not the settings the RobBERT authors used for their own published results, so the RobBERT numbers here
differ from theirs.

<details>
<summary>RobBERT-2023 needed an offset correction for SQuAD-NL</summary>

Its fast tokenizer returns word offsets that start one character late (`ĠDe` → "e"). Our evaluation turns off
`trim_offsets` in its post-processor: same token ids, correct offsets. Without the correction RobBERT-2023 scores
F1 68.21 / EM 14.12 on SQuAD-NL (seed 42) instead of 84.08 / 73.63. The tokenizer of the dmbert models is not
affected.

</details>

## Memorization test

Does a model reconstruct documents it was trained on better than comparable documents it never saw? The sample:
1,000 documents of the filtered FineWeb-2 test split described under Training data (drawn at random among those of 64
to 1,022 tokens), and for each of them a document of the training data of dmbert-230m-fineweb2-optout with the same
number of tokens (from 50,000 training documents drawn at random); 383 tokens per document on average. Each document
is scored alone, in float32, with masks that depend only on the document, so both models see the same masks:

- **MLM**: 15% of the tokens masked as in pretraining, three mask seeds; the loss and the accuracy on the masked
  tokens;
- **span**: 16 consecutive tokens replaced by `<mask>` at a random position, three positions per document; the share
  of right tokens, and the share of spans whose 16 tokens are all right.

The difference training − held-out is the mean over the 1,000 matched pairs, with a 95% bootstrap interval (10,000
resamples) in brackets and a two-sided sign-flip permutation p-value (10,000 permutations).

| Model | Documents | MLM loss | MLM accuracy | span accuracy | spans all right |
|---|---|---:|---:|---:|---:|
| dmbert-230m-culturax | training data of dmbert-230m-fineweb2-optout | 0.7497 | 0.8153 | 0.1749 | 1 of 3,000 |
| | held-out | 0.7706 | 0.8122 | 0.1787 | 0 of 3,000 |
| | difference | −0.0208 [−0.0509, +0.0087], p 0.18 | +0.0032 [−0.0029, +0.0091], p 0.31 | −0.0039 [−0.0105, +0.0030], p 0.28 | +0.0003 [0, +0.0010], p 1.00 |
| dmbert-230m-fineweb2-optout (this model) | its training data | 0.7331 | 0.8198 | 0.1769 | 2 of 3,000 |
| | held-out | 0.7319 | 0.8193 | 0.1831 | 0 of 3,000 |
| | difference | +0.0012 [−0.0266, +0.0291], p 0.93 | +0.0004 [−0.0054, +0.0064], p 0.88 | −0.0062 [−0.0130, +0.0009], p 0.09 | +0.0007 [0, +0.0017], p 0.50 |

For this model none of the four differences is distinguishable from chance: every 95% interval contains 0, and no
p-value is below 0.05. dmbert-230m-culturax was not trained on these documents as a set, so its row is a control; its
own training data, CulturaX, was not tested this way. Both corpora come from Common Crawl, and we did not measure how
many of these documents also occur in CulturaX. The test covers 1,000 documents and these two kinds of
reconstruction; it does not show that no training text can be recovered from a model.

## Limitations

- Pretrained on sequences of at most 1,024 tokens. The rotary embeddings accept longer inputs, but the model has not
  been trained or evaluated on them.
- Trained on web text only (FineWeb-2, from Common Crawl snapshots of 2013 to 2023). It reflects the content,
  opinions and biases of the Dutch web of those years. Personal data in running text, such as names, was not removed.
- The opt-out filter reflects what sites said in November 2026, per host, in robots.txt, `tdmrep.json`, `ai.txt` and
  the blocklists above. It did not see reservations in page headers, meta tags or terms of use, reservations of hosts
  whose robots.txt could not be read (22.7% of the kept documents), or text that a site copied from another site with
  a reservation. The blocklists cover only what their sources list.
- Dutch only; other languages were not part of the training data.
- `hebban.nl` was excluded to keep DBRD out of the training data. We did not check the other evaluation sets for
  overlap with the training data.
- No evaluation of social bias or safety.

## License

Apache 2.0, for the model weights, configuration and tokenizer files in this repository. The tokenizer
`yhavinga/dutch-llama-tokenizer` is also Apache 2.0. The training data, FineWeb-2, is released under the
[ODC-By 1.0](https://opendatacommons.org/licenses/by/1-0/) license, and its use is also subject to the
[Common Crawl terms of use](https://commoncrawl.org/terms-of-use); the texts belong to their authors. The model does
not generate text and does not redistribute the corpus.

## Acknowledgements

Research supported with Cloud TPUs from Google's [TPU Research Cloud (TRC)](https://sites.research.google/trc/about/).

## Citation

```bibtex
@misc{dmbert_230m_fineweb2_optout,
  title  = {dmbert-230m-fineweb2-optout: a Dutch ModernBERT encoder},
  author = {Yeb Havinga},
  year   = {2026},
  publisher = {Hugging Face},
  url    = {https://huggingface.co/yhavinga/dmbert-230m-fineweb2-optout}
}
```
