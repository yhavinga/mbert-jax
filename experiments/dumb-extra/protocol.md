# Pre-registered protocol: span-, token- and grammar-level evaluations of Dutch ModernBERT

Status: written on 2026-11-08, before any run of these evaluations (smoke tests aside). Changes after a real run has
started are added below under "Amendments", dated, and never replace the text above them.

Code: branch `dumb-extra` of mbert-jax, commit `ae114789bf2eae153e4e3bea579299a617dcd417` (the `COMMIT` of
`commands.txt`). Scripts: `gs://t5-dutch-english-usc2/mbert/dumb-extra/scripts/` (= `experiments/dumb-extra/`).
Data: `gs://t5-dutch-english-usc2/mbert/dumb-extra/data/` with `manifest.json` (written by `stage_data.py`).

## Question

Every model comparison so far used two DUMB tasks, DBRD and SICK-NL, both sentence classification, and the masking
decision ("15% single tokens beats a UL2-style mix of span denoisers") rests on SICK-NL. Span masking is known to
help extractive QA and span-level tasks. Do the models differ on span-level (QA), token-level (NER, POS) and
grammar-level (minimal pairs) tests, and in particular: does the mix (or mixheads) model beat the 15% model on QA or
NER by more than the larger seed range?

## Models

Fine-tune tasks: the six 653,000-step models, named as their DBRD/SICK-NL results in
`gs://t5-dutch-english-usc2/mbert/masking/dumb`. BLiMP-NL: these six and nine more. All are step directories in
`gs://t5-dutch-english-usc2/mbert/` with `commit_success.txt` (all 15 checked on 2026-11-08).

| name | step directory | fine-tunes | BLiMP-NL |
|---|---|---|---|
| `653000` (S, 15%) | `culturax-nl-230m-chunk1024/ckpt_keep/653000` | yes | yes |
| `mix-653000` | `culturax-nl-230m-chunk1024-mix/ckpt_keep/653000` | yes | yes |
| `mixheads-653000` | `culturax-nl-230m-chunk1024-mixheads/ckpt_keep/653000` (2 heads; head 0 is used) | yes | yes |
| `mc4-653000` | `mc4-nl-230m-chunk1024/ckpt_keep/653000` | yes | yes |
| `fw2-653000` (FineWeb-2, 15%) | `fineweb2-nl-230m-chunk1024/ckpt_keep/653000` | yes | yes |
| `fw2mix-653000` | `fineweb2-nl-230m-chunk1024-mix/ckpt_keep/653000` | yes | yes |
| `anneal-u15-uniform10` | `anneal/anneal-u15-uniform10/ckpt/20000` | | yes |
| `anneal-u15-uniform5` | `anneal/anneal-u15-uniform5/ckpt/20000` | | yes |
| `anneal-u15-uniform15-65300` | `anneal/anneal-u15-uniform15-65300/ckpt/65300` | | yes |
| `anneal-u15-uniform10-65300` | `anneal/anneal-u15-uniform10-65300/ckpt/65300` | | yes |
| `anneal-u15-uniform5-65300` | `anneal/anneal-u15-uniform5-65300/ckpt/65300` | | yes |
| `p2048-u15-C1024` | `phase2048/p2048-u15-C1024/ckpt/20000` | | yes |
| `flan-F` | `flan-decay/flan-F/ckpt/20000` | | yes |
| `300000` | `culturax-nl-230m-chunk1024/ckpt_keep/300000` | | yes |
| `fw2-300000` | `fineweb2-nl-230m-chunk1024/ckpt_keep/300000` | | yes |

A `--mix-heads` checkpoint is used with head 0 (the head of the 15% denoisers, which predicts the held-out masks), as
`EXPORT_HEAD=0` of `corpus-runs/dumb_mix_v6e.sh`: `dumbx.sh` drops the other heads after the conversion,
`eval_blimp_nl.py --export-head 0` reads them and leaves them out.

## Tasks and data

Evaluation data only; none of it may become pretraining data. Licences as their sources state them
(`manifest.json` quotes them).

| task | data (staged revision) | licence | train / dev / test | metric (test) |
|---|---|---|---|---|
| `squadnl` | SQuAD-NL v1.1, `GroNLP/squad-nl-v1.1` @ `7e1167c6` | CC BY-SA 4.0 | 87,599 / 9,380 / 1,183 questions | F1 (and EM) |
| `conll2002` | CoNLL-2002 NER, Dutch (teropa/nlp @ `f89bc5a1`, = the archived official files) | none stated; README: research use in the context of the shared task only, no commercial use | 15,806 / 2,895 / 5,195 sentences | entity F1 |
| `pos-lassysmall` | UD_Dutch-LassySmall, tag r2.18 | CC BY-SA 4.0 | 13,817 / 1,542 / 1,761 sentences | UPOS accuracy |
| `pos-alpino` | UD_Dutch-Alpino, tag r2.18 | CC BY-SA 4.0 | 12,289 / 718 / 596 sentences | UPOS accuracy |
| BLiMP-NL | BLiMP-NL large, DOI 10.34973/tj4p-y007 v1 | CC BY-SA 4.0 | 8,400 pairs, 84 paradigms, 22 phenomena | accuracy |

Not run: WiC-NL (DUMB's version is built from DutchSemCor's annotations, which hold token ids only; the sentences come
from SoNaR-500, IvdNT licence, manual download), and, as decided beforehand, COPA-NL, DPR, DALC and the IvdNT
corpora Lassy Small and SoNaR. DUMB's `squadnl` is SQuAD-NL v2.0 (with unanswerable questions); this protocol uses
v1.1 (answerable only), so its numbers are not DUMB's.

## Fine-tune tasks

`scripts/dumb/finetune_extra.py`, run by `dumbx.sh` with `spec_653k.txt`, per model and task:

- Seeds 42, 43, 44 (the seed of the head initialization, dropout and the NumPy shuffles).
- Learning rate 5e-5 (the best measured for SICK-NL), AdamW, weight decay 0.01, float32, as `finetune.py`. The rate
  falls linearly to 0 over all updates, without warmup: Hugging Face's Trainer as DUMB runs it (`warmup_ratio` 0.0).
  `finetune.py` itself has no schedule (a constant rate) and is not changed; DBRD and SICK-NL keep their computation
  (`tests/test_dumb_pinned.py`).
- 3 epochs. The update is 32 examples (DUMB's `train_batch_size`), as 8 per chip x 4 chips x 1 accumulation step.
  DUMB splits the 32 into per-device batch x accumulation per task where memory requires (`trainers/run.sh`: 16 x 2
  for these tasks with large models); 8 per chip fits all four tasks here without accumulation. Not DBRD's 192 or
  SICK-NL's 256: with 256, POS on Alpino would get 3 x 12,289 / 256 = 144 updates.
- Length: token tasks at most 512 tokens per row; a longer sentence (CoNLL-2002: 4 in train, 1 of 409 words in test,
  none in dev; none in UD) is cut between words into rows that fit and joined again before scoring, so no word is
  dropped. Truncation would drop words, and this tokenizer truncates from the left (`truncation_side: left`). `squadnl`: windows of 384
  tokens, `<s> question <s> context`, overlapping by a doc stride of 128 tokens; a window without the whole answer is
  trained on position 0; questions lose their leading whitespace, as in Hugging Face's `run_qa.py` (one train question
  is 1,609 tokens of it). Rows are padded to a multiple of 128.
- Heads: a new linear classifier per token (token tasks, label at a word's first sub-token, `-100` elsewhere) or per
  token for start and end (`squadnl`), on the encoder and the MLM prediction head of the checkpoint.
- QA predictions: the best start + end logit over the question's windows among the 20 best starts and ends, end not
  before start, at most 30 tokens. EM and F1 with the official SQuAD v1.1 normalization (lower case, punctuation and
  English articles removed), best over the reference answers.
- NER: entity-level micro F1 of the IOB2 tags, as seqeval's default (conlleval) mode; checked equal to seqeval 1.2.2.
  POS: accuracy over all words (UPOS), each treebank on its own.
- Reported on the test split, with the state after the last epoch. The dev split is evaluated after each epoch for
  the log only: no checkpoint selection, no early stopping, no tuning on it.
- Result per job: `gs://t5-dutch-english-usc2/mbert/dumb-extra/<task>/<name>--<task>--seed<n>.json`: the fields of
  `finetune.py`'s results (`test_accuracy` = word accuracy, for `squadnl` exact match) plus `metric`, `test_metric`,
  `test` (all test numbers), `validation` (per epoch), `seed`, `updates`, `lr_schedule` and the data revision.
- Summary per model and task: mean and [min-max] of `test_metric` over the three seeds (`summarize.py`).

## BLiMP-NL

`scripts/eval_blimp_nl.py`, run by `blimp.sh`; no fine-tuning, deterministic, one score per model. Per sentence the
PLL-word-l2r of Kauf and Ivanova (2023), the metric of the BLiMP-NL paper (Suijkerbuijk et al. 2025, section 5.2): per
token the log-probability with that token and the later tokens of its word masked, summed. Words: a new word at a
token with `▁`, at a token that starts with punctuation, after a token that ends with punctuation (BERT's basic
tokenization, which the paper's masked models had; this tokenizer has no pre-tokenizer). A sentence is `<s> tokens
</s>`; the specials are not scored. A pair is right when the grammatical sentence scores strictly higher. float32.
Reported: accuracy over the 8,400 pairs, per phenomenon (22), per paradigm (84), and for the 840 hand-crafted pairs
(the small version's items) and the 7,560 pairs generated with ChatGPT (GPT-3.5 Turbo) and checked by hand, plus the
per-pair scores. Result: `gs://t5-dutch-english-usc2/mbert/blimp-nl/<name>.json`.

## What it decides

The numbers are reported next to DBRD and SICK-NL of the same models. The question of this protocol: does
`mix-653000` (or `mixheads-653000`) beat `653000` on `squadnl` (F1) or `conll2002` (entity F1) by more than the larger
seed range, that is mean(mix) - mean(S) > max(range(mix), range(S)) with range = max - min over the three seeds (and
the same the other way round)? `summarize.py` prints this per task. No automatic consequence; Yeb decides.

## Amendments

2026-11-09, before step 4 started (09:07 CET), after the fine-tunes of step 3 had started: BLiMP-NL also runs on
`fw2rh3-300000` (FineWeb-2 with rehydration, `fineweb2-nl-230m-chunk1024-rh3/ckpt_keep/300000`) and `flan-Fnl` (the
decay with Dutch FLAN, `flan-decay/flan-Fnl/ckpt/20000`), 17 models in all. They do not enter the question above.

2026-11-09: step 5 ran with the JSON results (not the logs) in the session's scratchpad instead of `/tmp/dx`, because
macOS removes files in `/tmp` after 3 days. Same `summarize.py` arguments.

2026-11-09, after the results of step 3 were known and before these runs: the four fine-tune tasks, with the same
settings and seeds, also for three 300,000-step models, `300000` (CulturaX, 15%), `fw2-300000` and `fw2rh3-300000`
(`spec_300k.txt`). They are the references of the GPT-NL proof G (`cxgptnl-300000`, 300,000 steps), which gets the same
fine-tunes and BLiMP-NL once it is done. Reported with the same comparison (difference of means against the larger
seed range), against `300000`; descriptive, no consequence, and outside the question above, which the 653,000-step
models answer.

## Result (2026-11-09)

All 72 fine-tunes and 17 BLiMP-NL runs complete (`summary.json` next to the results). The question: `mix-653000` beats
`653000` on `squadnl` (F1 84.58 against 83.18, +1.40 against a larger seed range of 1.26) and on `conll2002` (entity
F1 92.50 against 91.86, +0.64 against 0.41); `mixheads-653000` beats it on `squadnl` (85.13, +1.95 against 1.17), not
on `conll2002` (+0.33 against 0.72). POS: no model differs from `653000` by more than the larger range, except
`fw2-653000` on LassySmall (-0.22 against 0.20, computed on word counts). BLiMP-NL: `653000` 94.60, `mix-653000`
93.74, `mixheads-653000` 93.52 (of 8,400 pairs).

The 300k references (amendment above; 36 fine-tunes, done 2026-11-09 10:01, `summary-300k.json`): against `300000`
(SQuAD-NL 79.00, CoNLL-2002 85.66, POS 97.02 / 97.14), neither `fw2-300000` nor `fw2rh3-300000` differs by more than
the larger seed range on `squadnl` or `conll2002`; `fw2rh3-300000` is below it on both POS treebanks (-0.23 against
0.18, -0.68 against 0.25) and `fw2-300000` on Alpino (-0.28 against 0.25), all on word counts.
