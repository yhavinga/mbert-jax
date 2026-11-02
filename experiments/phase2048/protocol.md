# Pre-registered protocol: a 2048-token phase for Dutch ModernBERT

Status: written on 2026-11-02, before any run of this experiment (smoke tests aside). Changes after a real run has
started are added below under "Amendments", dated, and never replace the text above them.

Code: branch `phase2048` of mbert-jax, commit `3de74276658f504aaa91929fd7f31d343e3237df` (the commit `setup_phase.sh` checks out).
Scripts: `gs://t5-dutch-english-usc2/mbert/phase2048/scripts/` (= `experiments/phase2048/` of that branch).
Outputs: `gs://t5-dutch-english-usc2/mbert/phase2048/`.

## Question

Does continued pretraining at 2048 tokens per sequence give a better encoder than the same amount of continued
pretraining at 1024, the length of the pretraining so far? If not, the retrain skips the 2048 phase.

## Models

| | what | steps | `--max-length` | per chip | slots per step on 16 chips |
|---|---|---|---|---|---|
| S | the start checkpoint | - | (1024) | - | - |
| C | S + 20,000 steps | 20,000 | 1024 | 8 | 128 x 1024 = 131,072 |
| L | S + 20,000 steps | 20,000 | 2048 | 4 | 64 x 2048 = 131,072 |

S is the better of the two final 653,000-step models, chosen before C and L start and recorded in
`<run>/start.txt` by `start_phase.sh`:

- `gs://t5-dutch-english-usc2/mbert/culturax-nl-230m-chunk1024/ckpt_keep/653000`, masking `uniform15`, name `u15`;
- `gs://t5-dutch-english-usc2/mbert/culturax-nl-230m-chunk1024-mix/ckpt/653000`, masking `mix`, name `mix`.

C and L differ only in `--max-length` and the per-chip batch. Shared, as set in `start_phase.sh`:

- Start: `--init-from S`, the parameters of S only. The optimizer, the step and the data start fresh. A preempted
  arm resumes from its own last checkpoint (that wins over `--init-from`).
- Optimizer: fresh AdamW, weight decay 0.01, gradient clipping 1.0, `--layer-norm-lr-factor 1.0`; LR linear from 0
  to 1e-4 over 1,000 steps, then cosine to 1e-6 at step 20,000 (`--lr-schedule cosine --warmup-steps 1000
  --learning-rate 1e-4 --min-lr 1e-6 --max-steps 20000`).
- Masking: that of S (`--masking`; `start_phase.sh` checks it against the data state of S).
- Data: CulturaX-nl, the 32 local parquet files per host of the long run (`--data-files-list`), from a fresh data
  state: shuffle seed 42, buffer 10,000, piece window 5,000, packing on. A document longer than `--max-length`
  is cut into pieces of at most `--max-length` tokens (`split_into_pieces` in `src/mmbjax/training/packing.py`,
  called with the trainer's `--max-length` by `TokenizeTransform`), each piece with `<s>`, the last with `</s>`.
  The long run read CulturaX-nl once in 653,000 steps; a fresh data state reads the same documents in the same
  order as the long run's first steps. C re-reads about the documents of the long run's first 20,000 steps, L
  about as many (accepted: a small part of a second epoch).
- Seeds: as in the long run (model init 42, unused with `--init-from`; data 42; masking 42 + host).
- Model: 22 layers, hidden 1024, 16 heads, intermediate 1536, global attention every 3 layers. RoPE and the
  local window are not changed: `global_rope_theta` 160000.0, `local_rope_theta` 10000.0, `local_attention` 128
  (a window of +-64 tokens), `max_position_embeddings` 8192. These are the defaults of `MMBertConfig` in
  `src/mmbjax/configuration_mmbbert.py` (lines 41, 45, 46 and 31); `scripts/train_minimal.py` does not pass them
  (it sets `max_position_embeddings=8192` itself), and `scripts/convert_gcs_checkpoint.py` writes
  `local_attention: 128` and `max_position_embeddings: 8192` into the `config.json` of the fine-tunes.

Check after both arms: the `DATA` line at the end of each arm's coordinator log gives per host the documents whose
last piece was trained on (`documents_completed`) and the tokens other than padding (`tokens_read`). Both arms read
the same stream; with equal slots per step they consume about the same documents. L packs with less padding and
cuts fewer pieces, so it may consume slightly more; the counts make the difference explicit.

## Evaluations

### DUMB fine-tunes (decide)

The protocol of `gs://t5-dutch-english-usc2/mbert/masking/dumb/` (`scripts/dumb/finetune.py`): 3 epochs, LR 2e-5,
AdamW weight decay 0.01, 4 chips per fine-tune, seeds 42, 43, 44; DBRD with an update of 192 examples, SICK-NL
with 256. Only `--max-length` changes. Cells, for each of S, C and L:

| cell | task | `--max-length` | per chip x 4 chips x accumulation | `--pad-to-multiple-of` |
|---|---|---|---|---|
| dbrd512 | DBRD | 512 | 24 x 4 x 2 = 192 | 1 |
| dbrd1024 | DBRD | 1024 | b x 4 x 48/b = 192, b the largest of 12, 8, 6, 4, 3, 2, 1 that fits | 128 |
| dbrd2048 | DBRD | 2048 | b x 4 x 48/b = 192, b the largest of 4, 3, 2, 1 that fits | 128 |
| sicknl512 | SICK-NL | 512 (its usual length) | 64 x 4 x 1 = 256 | 1 |

At 512 both tasks run exactly as before. At 1024 and 2048 the update is the same 192 examples, accumulated from
smaller micro-batches because a chip does not hold 24 sequences of that length. "Fits" is measured on the v4 chips
themselves before the queue starts: `dumb_phase.sh` runs the train and eval step of `finetune.py` on a batch with
every row at the full length (`scripts/dumb/probe_memory.py`) and writes the table to
`gs://t5-dutch-english-usc2/mbert/phase2048/dumb/tables/`; every worker must come to the same table. The same b holds
for S, C and L. Measured on a v6e chip (31.24 GB available), XLA temporaries per chip: 23.72 GB at 24 x 512 (the
configuration of the earlier rounds, which fits a v4 chip), 23.81 GB at 12 x 1024, 23.52 GB at 3 x 2048; 4 x 2048
does not fit (31.55 GB), nor 6 x 2048 (47.32 GB). A v4 chip needs more: 48 x 512 took 33.98 GB there and 27.66 GB
on v6e. So on v4 b is at most 3 at 2048, and 12 at 1024 and 3 at 2048 are borderline; the probes decide.

Every batch at 1024 and 2048 is padded to a multiple of 128. Padding is masked and the classifier reads `<s>`, so
this changes no result (`tests/test_dumb_finetune.py`). It avoids a compile for every distinct batch length:
without it DBRD@1024 has 418 and DBRD@2048 1,361 distinct batch shapes in a run of 3 epochs, with it 10 and 31.

Known side effect of accumulation, also present at 512: an epoch of 19,528 examples does not end on an update
boundary, and `optax.MultiSteps` carries the micro-batches of the last, incomplete update into the next epoch. With
other micro-batch sizes the examples grouped into an update after the first epoch differ slightly between the
lengths. Within one cell all models use the same grouping.

Metric: `test_accuracy` of `results.json`, the accuracy on the test split after the third epoch, as reported by
`finetune.py` in all earlier rounds.

How much DBRD can show (tokens of `yhavinga/dutch-llama-tokenizer`, `<s>` included, `scripts/dumb/lengths.py`):

| split | examples | > 512 | > 1024 | > 2048 |
|---|---|---|---|---|
| train | 19,528 | 4,601 (23.6%) | 518 (2.65%) | 36 (0.18%) |
| validation | 500 | 114 (22.8%) | 14 (2.8%) | 0 |
| test | 2,224 | 527 (23.7%) | 58 (2.61%) | 4 (0.18%) |

So DBRD at 2048 and at 1024 see different input for 58 of 2,224 test examples (2.6%); at 1024 and at 512 for 527
(23.7%). A difference between dbrd2048 and dbrd1024 of a model can come only from those 58 examples, at most 2.6
points, and is likely far smaller. DBRD@2048 can show little; the decision below can therefore easily come out as
"no measurable difference" even if 2048 helps documents that are long.

### MLM loss per window (supporting evidence, decides nothing)

`scripts/eval_windows.py` on all 40,000 documents of `heldout.parquet`
(`gs://t5-dutch-english-usc2/mbert/heldout/nl_part_00097_first40000.parquet`, md5
b2f579a876d06eef40c94270537bb7af): each document masked once with 15% single tokens (80/10/10) from a generator
seeded with (0, document index), the same masked tokens for every model and both windows, whatever the training
masking. Windows 1024 and 2048, each document cut into the trainer's pieces for that window. Reported: the loss
(a) on all masked tokens, (b) on the masked tokens of the 6,747 documents longer than 1024 tokens (with `<s>` and
`</s>`), and (c) at window 2048 per position in the piece, 0-1023 and 1024-2047.

## Estimates on mbert-mix (v4-32)

- C: 20,000 x 0.237 s, the mean step time of the long run on this VM type with the same command, is 79 minutes;
  with start, evaluations and checkpoints about 1.5 hours.
- L: on a v6e-8 the steady step time of L was 1.52 times that of C (0.207 s against 0.136 s for 65,536 slots), as
  the attention is computed in full also in the local layers and costs B x L^2. Hence about 0.36 s per step, 2.0
  hours, with overhead about 2.2 hours.
- Memory of the arms per chip (XLA's analysis on v6e, 31.24 GB available): C 13.49 GB of temporaries + 2.76 GB of
  arguments + 2.76 GB of outputs; L 19.46 + 2.76 + 2.76 GB. On v4 C peaked at 18.07 GiB in the long run, about what
  v6e gives, so L should fit at 4 per chip without accumulation, about 5 GB below the limit.
- Window eval: 27.4 million tokens per window and model, about 213 global batches per window: a few minutes per
  model, about 15 minutes for S, C and L.
- DUMB: 9 fine-tunes per worker, the 4 workers in parallel, after per worker 3 conversions and the memory probes.
  DBRD@512 took 3.5 minutes per run and SICK-NL 2 to 4 (24 for the first, which compiles all its shapes) on a v4-8.
  DBRD@2048 on 4 v6e chips at 3 x 4 x 16: the first epoch took 13:48, about 9.5 minutes of it compiling batch
  shapes, then 11.8 micro-steps per second, 2.3 minutes per epoch of 1,628 micro-steps. A v4 chip ran the trainer
  1.74 times slower than a v6e chip, so about 4 minutes per epoch on v4: some 30 minutes for a worker's first
  DBRD@2048 run and 15 for later ones (the compile cache is per worker). DBRD@1024 somewhat less. About 2.5 to 3
  hours for the queue, the conversions and memory probes included.

## Decision rule

Per cell: the mean of `test_accuracy` over seeds 42, 43, 44, and the seed range, max minus min.

- "2048 helps" if the mean of L at dbrd2048 exceeds the best mean of C over {dbrd512, dbrd1024, dbrd2048} by more
  than the larger seed range of the two compared cells (L at dbrd2048 and that best C cell), AND L is not below C
  on sicknl512 or on dbrd512 by more than C's seed range for that cell.
- "2048 harms" if L is below C on sicknl512 or on dbrd512 by more than C's seed range for that cell.
- Otherwise "no measurable difference"; the retrain then skips the 2048 phase.

"More than" is strict. "2048 harms" is checked first: a gain on DBRD with a loss on SICK-NL or DBRD@512 beyond
C's range is "2048 harms". S against C shows the effect of the extra 20,000 steps themselves and decides nothing.
The window eval is supporting evidence and decides nothing. `experiments/phase2048/decide.py` applies this rule to
the result files (`tests/test_phase2048_decide.py`).

## Amendments

### 1. A paired window test can also decide "2048 helps" (2026-11-02, before any real run)

Why: DBRD has 2.6% of its test reviews over 1024 tokens and 0.18% over 2048, so DBRD@2048 can differ from
DBRD@1024 on at most 58 of 2,224 examples. The rule above would then almost always end in "no measurable
difference", also when the 2048 phase works. The question is whether L does no harm on the short tasks and uses the
longer context; DUMB measures the first, the window eval measures the second. Chosen by Yeb on 2026-11-02.

The window eval now decides along with DUMB. With the results of `scripts/eval_windows.py` for S, C and L
(`$RUN_ROOT/eval/<start>-{S,C,L}.json`, both windows over all documents of `heldout.parquet`):

- "L uses the longer context" if L's loss on the long documents (`long_documents`, documents over 1024 tokens) at
  window 2048 is lower than L's own loss on the same group at window 1024, AND L's loss on positions 1024-2047
  (`positions_high` at window 2048) is lower than C's. The test is paired: both windows and all three models score
  the same masked tokens, which `decide.py` checks on the counts of masked tokens; a difference is an error.
- The decision: "2048 harms" as above, checked first. Otherwise "2048 helps" if L exceeds C on DBRD as above OR L
  uses the longer context. Otherwise "no measurable difference".

This replaces "The window eval is supporting evidence and decides nothing". Its other groups (`all`,
`positions_low`, `long_documents_positions_low`) stay reported and decide nothing. Positions 0-1023 against
1024-2047 is not used as a test: positions 0-1023 contain the starts of documents and, in `positions_low`, all the
short documents, so their loss differs for reasons other than context. Code: `decide.py` (`uses_context`), tests
in `tests/test_phase2048_decide.py`.
