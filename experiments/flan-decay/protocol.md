# Pre-registered protocol: FLAN with masked answers in the decay of Dutch ModernBERT 230M (arm F)

Status: written on 2026-11-07, before any run of arm F (smoke tests aside). Changes after the real run of F has
started are added below under "Amendments", dated, and never replace the text above them.

Design: `mbert/designs/flan-decay.md` in the hammond repo (2026-11-03), sections 2, 3, 4 and 6. This protocol builds
only what arm F needs; the deviations from the design are listed at the end, with the reason for each.
Code: branch `flan-decay` of mbert-jax, at the commit `COMMIT` of `commands.txt` (the commit `setup_flan.sh`,
`start_flan.sh` and `eval_zeroshot.sh` check out).
Scripts: `gs://t5-dutch-english-usc2/mbert/flan-decay/scripts/` (= `experiments/flan-decay/` of that commit).
Outputs: `gs://t5-dutch-english-usc2/mbert/flan-decay/`.

## Question

Does a decay phase of web text at a low single-token rate plus a share of FLAN examples with masked answers give a
better Dutch encoder than the same decay with web text only?

- H1 (fine-tuning): no harm on DUMB (DBRD, SICK-NL). A gain is possible; the literature expects little (EuroBERT
  measured harm with 1.2% instruction data in its annealing).
- H2 (zero-shot): the MLM head classifies Dutch tasks zero-shot ("generative classifier", as ModernBERT-Large-Instruct);
  without FLAN it is close to chance.

## Models

| | what | run | step directory |
|---|---|---|---|
| S | the start: 653,000 steps of CulturaX-nl at 15% single tokens | `culturax-nl-230m-chunk1024` | `gs://t5-dutch-english-usc2/mbert/culturax-nl-230m-chunk1024/ckpt_keep/653000` |
| W | S + 20,000 steps of web text at 5% single tokens | `anneal-u15-uniform5` | `gs://t5-dutch-english-usc2/mbert/anneal/anneal-u15-uniform5/ckpt/20000` |
| F | S + 20,000 steps of web text at 5% and FLAN at 10% of the tokens, answers masked | `flan-F` | `gs://t5-dutch-english-usc2/mbert/flan-decay/flan-F/ckpt/20000` |
| C | S + 20,000 steps at 15% single tokens (arm C of the 2048 phase); a free extra | `p2048-u15-C1024` | `gs://t5-dutch-english-usc2/mbert/phase2048/p2048-u15-C1024/ckpt/20000` |

W already exists: `experiments/anneal/start_anneal.sh S uniform15 uniform5 u15` (branch `anneal`, commit 9bdde82,
recorded in its `start.txt`), on mbert-corpus. F is W with other data and another collator, and nothing else
(`start_flan.sh` holds W's command but for `--masking` and `--data-files-list`):

- Start: `--init-from S`, the parameters of S only; the optimizer, the step and the data start fresh. A preempted F
  resumes from its own last checkpoint (that wins over `--init-from`).
- Optimizer: fresh AdamW, weight decay 0.01, gradient clipping 1.0, `--layer-norm-lr-factor 1.0`; LR linear from 0 to
  1e-4 over 1,000 steps, then cosine to 1e-6 at step 20,000.
- Batch: 8 per chip on 16 chips, `--max-length 1024`: 128 x 1024 = 131,072 slots per step, packed with segment ids.
- Seeds: data 42, masking 42 + host, as W.
- Held-out loss: `--eval-file ~/heldout.parquet --eval-every 5000 --eval-batches 50`, 15% single tokens in every arm.
- Model: 22 layers, hidden 1024, 16 heads, intermediate 1536, global attention every 3 layers; `vocab_size` 32,128.

### Data of F

Per host one parquet file of `scripts/build_decay_mix.py` (`setup_flan.sh`), in `--data-files-list ~/flan_decay_files.txt`:

- Web: the host's CulturaX share of W (`~/culturax_nl_files.txt`), read through the trainer's own route
  (`load_dataset` parquet, shuffle seed 42, buffer 10,000), from the start, until the web tokens are what 20,000 steps
  need at 90% of the slots plus 20% (707,788,800 tokens per host). These are W's first documents, in W's order.
- FLAN: `Open-Orca/FLAN` at revision `6845b1b3b53c6d5c5b1e49767ed759df3fc246cc`, the zero-shot folders only
  (`cot_zsopt_data`, `dialog_zsopt_data`, `flan_zsopt_data`, `flan_zsnoopt_data`, `niv2_zsopt_data`, `t0_zsopt_data`,
  `t0_zsnoopt_data`; niv2, cot and dialog have no zsnoopt folder there). English, plus the six Dutch NIv2 tasks.
  Host r reads shards r, r + 4, ... of a folder (every shard is a random sample of its folder: measured, all tasks in
  every shard); a folder of fewer than 4 shards is split by row index. The hosts' parts are disjoint.
- Filters, per task: SICK (`task1612`-`task1615`, any task name with `sick` or `dbrd`), `belebele`/`flores` in the name,
  HellaSwag (Flan 2021 and NIv2 `task1389`) and MOCHA (`task595`, `task596`) for their GPT-generated parts, WMT and
  `para_crawl_enes` (other languages), NIv2 tasks with an input, output or instruction language other than English and
  Dutch (`experiments/flan-decay/niv2_languages.json`, from the task files of natural-instructions at commit 55a3656:
  979 NIv2 tasks stay, the Dutch ones `task1133`, `task409`, `task1543`-`task1546` among them). Per row: U+2047, an
  empty instruction or answer, more than 1,024 tokens as `<s> inputs <answer> targets </s>`.
- Cap: at most 2,000 rows per (folder, task) over the 4 hosts, 500 per host, taken in shard order.
- Mix: after each web document, FLAN examples from the host's shuffled pool until FLAN is 10% of the tokens so far,
  counted as the trainer counts them. The file's rows are in the order that the trainer's shuffle turns into exactly
  this sequence; the builder reads the file back through the trainer's route and refuses it otherwise.
- How close F's web order stays to W's: the web documents reach F's tokenizer in W's order, document for document, with
  FLAN rows between them. After tokenization both arms shuffle pieces in windows of 5,000 (seed 42); in F the FLAN
  pieces shift the window boundaries, so within about one window (5,000 pieces) the order of F's web pieces differs
  from W's. F has 10% fewer web tokens per step, so at step t it has read the web documents W read by about 0.9 t;
  over 20,000 steps F reads about the first 90% of W's web documents.
- Expected per host (estimates from samples, the builder's manifest has the counts): ~1.03 M web documents,
  ~0.6 M FLAN examples in the pool (~120 M tokens) of which ~0.4 M (~79 M tokens, ~200 tokens each) are in the mix;
  ~1.4 M rows, ~2.5 GB. In 20,000 steps F trains on about 0.31 M FLAN examples per host, ~1.25 M in all.

### Masking of F

`--masking decay --web-mask-rate 0.05 --answer-full-max 16` (`DecayCollator` in `src/mmbjax/training/packing.py`):

- A FLAN example is one segment `<s> inputs <answer> targets </s>`, never split; `token_role` marks web (0),
  instruction (1), `<answer>` (2) and answer (3) and passes through packing.
- Web tokens: single tokens at 5% with 80/10/10, the draws of W's `PackedMLMDataCollator(mlm_probability=0.05)`; a
  batch without FLAN collates to the same bytes (`tests/test_packing_collator.py`).
- Instruction tokens and `<answer>`: never masked.
- An answer of at most 16 tokens: masked whole, every token `<mask>` (no 80/10/10), as at inference.
- A longer answer: the spans of `span_mu8_r50` (mean span 8, half of the answer), with 80/10/10.
- `<answer>` is a special token added after `<mask>`: `<mask>` stays 32000, `<answer>` is 32001, `vocab_size` stays
  32,128. S, W and C never trained row 32001.

## Evaluations

### 1. DUMB fine-tunes (decide)

The protocol of `gs://t5-dutch-english-usc2/mbert/masking/dumb/` (`scripts/dumb/finetune.py`, run by the existing
driver `dumb_v4_vm.sh` of `gs://t5-dutch-english-usc2/mbert/corpus-runs/scripts/`): DBRD and SICK-NL at
`--max-length 512`, 3 epochs, LR 2e-5, 4 chips per fine-tune, an update of 192 (DBRD) and 256 (SICK-NL), seeds
42, 43, 44. Cells: dbrd512 and sicknl512. Metric: `test_accuracy` of `results.json`.

- W: `gs://t5-dutch-english-usc2/mbert/masking/dumb/anneal-u15-u5--{dbrd,sicknl}--seed{42,43,44}.json` (the name the
  driver of the anneal arms gives it; whatever name it gets, it is the model of W's row above).
- F: the same driver with `flan-F=gs://t5-dutch-english-usc2/mbert/flan-decay/flan-F/ckpt/20000`, results
  `gs://t5-dutch-english-usc2/mbert/masking/dumb/flan-F--{dbrd,sicknl}--seed{42,43,44}.json`.
- S: `653000--{dbrd,sicknl}--seed{42,43,44}.json` there; C: `gs://t5-dutch-english-usc2/mbert/phase2048/dumb/u15-C--{dbrd512,sicknl512}--seed{42,43,44}.json`.

### 2. Zero-shot via the MLM head (decides "helps zero-shot")

`scripts/eval_zeroshot_mlm.py` (`eval_zeroshot.sh`), the same for S, W, F and C, one model per worker, each worker a
TPU of its own 4 chips, bfloat16. Each example is `<s> {prompt} <answer> <mask> </s>`; the prediction is the
verbalizer with the highest logit at the `<mask>`, among the task's verbalizers only. Accuracy per template and the
mean over the 3 templates. A prompt longer than 1,024 tokens loses words at the end of its long field (the review, the
passage); the count is reported.

| task | data | verbalizers (id) | labels |
|---|---|---|---|
| DBRD | `dbrd`, test, 2,224 | `▁negatief` 19080, `▁positief` 17456 | neg, pos |
| SICK-NL | `maximedb/sick_nl`, test, 4,906 | `▁A` 330, `▁B` 365, `▁C` 334 | A entailment, B neutral, C contradiction (from `entailment_label`; 56.9% neutral) |
| Belebele | `facebook/belebele` `nld_Latn`, test, 900 | `▁A` 330, `▁B` 365, `▁C` 334, `▁D` 384 | `correct_answer_num` 1-4 |

The templates, fixed here (`TEMPLATES` in `scripts/eval_zeroshot_mlm.py`):

- DBRD 1: `{text}\n\nIs deze boekrecensie positief of negatief?`
- DBRD 2: `Recensie: {text}\nIs het oordeel van de recensent over het boek positief of negatief?\nAntwoord:`
- DBRD 3: `Lees de volgende recensie van een boek en zeg of die positief of negatief is.\n\n{text}`
- SICK-NL 1: `Zin 1: {sentence_a}\nZin 2: {sentence_b}\nWat is de relatie tussen de twee zinnen?\nA: zin 2 volgt uit zin 1\nB: neutraal\nC: zin 2 spreekt zin 1 tegen`
- SICK-NL 2: `Premisse: {sentence_a}\nHypothese: {sentence_b}\nVolgt de hypothese uit de premisse?\nA: ja\nB: misschien\nC: nee\nAntwoord:`
- SICK-NL 3: `{sentence_a}\nVraag: is het volgende dan waar, onwaar of geen van beide? {sentence_b}\nA: waar\nB: geen van beide\nC: onwaar`
- Belebele 1: `{passage}\n\nVraag: {question}\nA: {a}\nB: {b}\nC: {c}\nD: {d}\nAntwoord:`
- Belebele 2: `Lees de tekst en beantwoord de vraag.\n\nTekst: {passage}\nVraag: {question}\n\nOpties:\nA. {a}\nB. {b}\nC. {c}\nD. {d}`
- Belebele 3: `{passage}\n{question}\nKies het juiste antwoord: A) {a} B) {b} C) {c} D) {d}`

Reported with each task: chance, the majority class, the predicted label counts per template, and how often the
head's first choice over the whole vocabulary is a verbalizer at all.

### 3. Supporting evidence (decides nothing)

- The `EVAL` lines of W and F at step 20,000: held-out MLM loss on 50 batches of `heldout.parquet` (15% single tokens).
- The `DATA` line at the end of each arm: per host the documents completed and the tokens other than padding. In F a
  FLAN example ends with `</s>` and counts as a document.
- The `MASKING` lines of F (coordinator host, every 1,000 steps): FLAN's share of the tokens, the masked web rate,
  answers masked whole and in spans, the answers' share of the masked tokens (`answer_loss_share`), and
  `instruction_masked`, which must be 0.
- The builder's manifests, `gs://t5-dutch-english-usc2/mbert/flan-decay/mix/w{0,1,2,3}.json`.

## Estimates on mbert-corpus (v4-32)

- Builder: per host ~15 GB of FLAN shards from Hugging Face through `/dev/shm`, ~1 M web documents tokenized; about
  20-30 minutes, on the CPU, the 4 hosts at once. It can run while W trains.
- F: W ran at 0.21 s per step on this VM (its log at step 4,500): 20,000 steps ~70 minutes, with start, evaluations and
  checkpoints about 1.5 hours. The collator's loop over FLAN answers adds a few milliseconds per step.
- Zero-shot: ~3.6 M tokens per model in ~115 batches of 32 x 1,024, a few minutes per model; S, W, F and C on the 4
  workers at once, about 10 minutes with loading and compiling.
- DUMB of F: 6 fine-tunes, DBRD@512 3.5 minutes each and SICK-NL 2-4 on a v4 host (earlier rounds).

## Decision rule

Per cell (dbrd512, sicknl512): the mean of `test_accuracy` over seeds 42, 43, 44, and the seed range, max minus min.
"More than" is strict. Checked in this order:

1. "FLAN harms" if F's mean on dbrd512 or on sicknl512 is below W's by more than W's seed range in that cell.
2. "FLAN helps fine-tuning" if not 1, and F's mean on dbrd512 or on sicknl512 is above W's by more than the larger of
   the two seed ranges (F's and W's) in that cell.
3. "FLAN helps zero-shot" if not 1, and F's zero-shot accuracy (mean over the 3 templates) is at least 5 points above
   W's on at least 2 of the 3 tasks.
4. Otherwise "no measurable difference": FLAN is not in the final run.

2 and 3 can both hold; both are then reported.

To be decided by Yeb before F starts (a business choice, design section 6), and written here as amendment 1: does
"FLAN helps zero-shot" alone, with no gain in fine-tuning, count as a reason to put FLAN in the final run? Until it is
decided, that outcome is reported without a consequence.

The free extra: W against C with the same three-seed statistics shows the effect of the lower masking (5% against 15%)
in a decay of 20,000 steps. It is reported and decides nothing about FLAN. S against W and F shows the effect of the
20,000 steps themselves and decides nothing.

Known head starts, noted with the outcome and not corrected: English review sentiment (amazon_polarity, imdb, yelp,
sst2) is in FLAN, a task relative of DBRD; English NLI (MNLI, SNLI, ANLI, RTE, CB) is a task relative of SICK-NL. No
copy of DBRD or of SICK-NL is in the training data. Belebele's FLORES passages were not checked against FLAN beyond the
name filter; the NIv2 FLORES tasks are translation tasks and are out by language as well.

## Deviations from the design

1. Cap 2,000 per (folder, task) over all hosts, not ~1,000 (design 2.2, item 5). Estimated from 3,000 rows per folder
   and the per-task counts of the design: at a cap of 1,000 the pool is ~1.31 M examples and ~262 M tokens, the need of
   20,000 steps without any margin; at 2,000 it is ~2.4 M examples and ~480 M tokens, 1.5 times the need with the 20%
   margin. A pool larger than the need is harmless: the mix takes a random part of it, so every task keeps its share.
2. DUMB with seeds 42-44 (3 seeds, like every other measurement of this project), not 42-46 (design 6).
3. Not built: WSD and `--branch-from` (design 3.6 and 4), F-plain, F-nl, and the separate FLAN count in
   `EpochTrackingDataLoader` (design 4: the `MASKING` lines count FLAN segments instead).
4. Answers longer than 16 tokens keep the 80/10/10 of the span denoiser; the design excludes 80/10/10 only for the
   answers of at most 16 tokens.
5. Filters beyond design 2.2: `para_crawl_enes` (English-Spanish) with the WMT tasks; HellaSwag in every submix (also
   NIv2 `task1389`); tasks named after Belebele or FLORES; rows with an empty instruction or answer. The NIv2 languages
   come from the task files of natural-instructions, as listed in `niv2_languages.json`.
6. `token_role` gives `</s>` the role of the answer it closes and `<s>` that of the instruction; both are special
   tokens and never masked, so this decides nothing.

## Amendments

(none)
