# Pre-registered protocol: FineWeb-2 nl with rehydration for Dutch ModernBERT

Status: written on 2026-11-07, before the training of this experiment starts. Changes after its training has
started are added below under "Amendments", dated, and never replace the text above them.

Data code: branch `rehydration` of mbert-jax, commit `71e4c49421c4de94d3a8f681c2079eab1b4da4a5`
(`scripts/stage_share.py rehydrate`, run by `rehydrate_fineweb2.sh`). Trainer: `~/mbert-jax` on mbert-mix at
`0d92db4`, the code of `fineweb2-nl-230m-chunk1024`.
Scripts: `gs://t5-dutch-english-usc2/mbert/rehydration/scripts/` (= `experiments/rehydration/` of that branch, and
its `experiments/corpus-runs/start_fineweb2.sh`). Shares: `gs://t5-dutch-english-usc2/mbert/rehydration/shares/`.
Run: `gs://t5-dutch-english-usc2/mbert/fineweb2-nl-230m-chunk1024-rh3/`.

## Question

Does FineWeb-2's rehydration (deduplication-informed upsampling, at most 3 repetitions) give a better Dutch encoder
than plain FineWeb-2 nl, at the same number of steps? Whether a full rehydrated FineWeb-2 run follows is Yeb's
decision.

## Arms

| arm | run | checkpoint | DBRD@512 mean [min-max] | SICK-NL@512 mean [min-max] |
|---|---|---|---|---|
| R | `fineweb2-nl-230m-chunk1024-rh3` | step 300,000 | to be measured | to be measured |
| F | `fineweb2-nl-230m-chunk1024` | `ckpt_keep/300000` | 91.84 [91.52-92.33] | 75.05 [74.69-75.39] |
| X | `culturax-nl-230m-chunk1024` (15%) | `ckpt_keep/300000` | 93.09 [92.71-93.42] | 76.85 [76.09-77.61] |

Accuracy in % (`test_accuracy` x 100). F: `gs://t5-dutch-english-usc2/mbert/masking/dumb/fw2-300000--{dbrd,sicknl}--seed{42,43,44}.json`
(DBRD 92.330, 91.525, 91.667; SICK-NL 75.391, 75.062, 74.692 for seeds 42, 43, 44). X:
`gs://t5-dutch-english-usc2/mbert/masking/dumb/300000--{dbrd,sicknl}--seed{42,43,44}.json` (DBRD 93.134, 93.419,
92.708; SICK-NL 76.090, 76.850, 77.611). Seed ranges: F 0.805 (DBRD) and 0.699 (SICK-NL); X 0.710 and 1.521.

### What differs between R and F

Only the data: `--data-files-list` (`$HOME/fineweb2_nl_rh3_files.txt` instead of `$HOME/fineweb2_nl_files.txt`) and
`--dataset-num-rows` (the kept documents of all 4 shares instead of 147,301,270; it feeds only the `data/passes`
metric and the logged config). Besides those, R has `--stop-at-step 300000` and its own name and bucket paths. The
same: trainer commit `0d92db4`, the command of `start_fineweb2.sh` with `MASKING=uniform15` (15% single-token
masking), model, optimizer, `--max-steps 653000` with the cosine schedule of 653,000 steps (so R's learning rate at
every step up to 300,000 is F's), seeds (model 42, data 42, masking 42 + host), 4 hosts with 32 sequences of 1024
tokens per step. `--stop-at-step` stops the run and keeps the schedule. Checked on 2026-11-07: the run script that
`start_fineweb2.sh` (this branch) writes with its defaults is byte-identical to `~/run_fineweb2-nl-230m-chunk1024.sh`
on worker 0 of mbert-mix, and with the overrides of `commands.txt` it differs only in the items above.

The trainer reads a host's files one after the other in an order drawn from seed 42, through a shuffle buffer of
10,000 documents (`load_dataset("parquet", data_files=..., streaming=True).shuffle(...)`,
`src/mmbjax/training/packing.py`). R's files hold about half the rows of F's, so by step 300,000 R has read through
more of its files than F; which documents a run has seen at a step follows from that order, for both arms alike.

## Rehydration

FineWeb-2 deduplicates each language with MinHash and keeps one document per duplicate cluster, with the size of
its cluster in `minhash_cluster_size`. Rehydration repeats a kept document by a weight of its cluster size,
computed from the filtering rates per cluster size (`misc/rehydration/README.md` of github.com/huggingface/fineweb-2,
commit 995a1ca, the last to change its weights): "for cluster sizes where the filtering rate was close to or higher
than the global filtering rate, we do not upsample the documents at all; for the cluster size with the lowest
filtering rate we repeat these document the maximum number of times [...] and for anything in between we linearly
interpolate between these 2 extremes." Weights of nld_Latn for at most 3 repetitions
(`misc/rehydration/weights/up_to_03_reps.json`): `{1: 1, 4: 2, 10: 3, 45: 2, 124: 1}`. As in the README's
`Rehydrater`, a cluster size takes the weight of the largest key at or below it:

| cluster size | 1-3 | 4-9 | 10-44 | 45-123 | 124 and more |
|---|---|---|---|---|---|
| weight w | 1 | 2 | 3 | 2 | 1 |

Fully rehydrated, nld_Latn grows from 147,301,270 to 233,693,423 documents (`weights/resulting_ds_sizes.json`, key
"3"); the post-filtering document counts per cluster size of `misc/rehydration/distributions/nld_Latn.json` (same
commit; 147,301,270 documents) give exactly that sum with this table, which confirms the reading of the weights.
Expected from those counts: 84,434,461 documents of weight 1, 39,341,465 of weight 2 and 23,525,344 of weight 3, and
77,897,808 documents kept (52.9%).

R does not repeat documents. Its budget, 300,000 steps, is below one pass over the rehydrated set, so rehydration is
done as a weighted selection without repeats: every document of FineWeb-2 nl is kept with probability w / 3
(`stage_share.py rehydrate --max-reps 3 --seed 42`). The draw is `numpy.random.default_rng([42, file index]).random(rows
of the file)` for every row of a source file, indexed by the row, so a document's draw does not depend on the host
that has it. Expected exposure: R reads a fraction c = T / K of its kept tokens K, where T is the tokens of 300,000
steps; a document of weight w is then seen (w / 3) x c times on average. Sampling T tokens from the fully rehydrated
set, of H tokens, sees it w x T / H times. These agree as K is H / 3 in expectation. Measured values below.

How the shares were made (`rehydrate_fineweb2.sh` on each worker of mbert-mix): from the share each host staged for
`fineweb2-nl-230m-chunk1024` (`~/fineweb2_nl`, every source file in segments of 1,000 rows,
`experiments/corpus-runs/fineweb2_plan.json`) and the column `minhash_cluster_size` of the source files of
HuggingFaceFW/fineweb-2 at revision af9c13333eb981300149d5ca60a8e9d659b276b9, read with range requests for this
host's row groups only. Before anything was written, per file and host the first, the last and one random segment
of the staged file were compared row by row with the `text` of the source. Tokens are those of the lengths parquets
of `count_tokens.py` (`gs://t5-dutch-english-usc2/mbert/token-counts/fineweb2/lengths/`), without `<s>` and `</s>`.

### The shares, as made on 2026-11-07

`rehydrate_fineweb2.sh` at commit 71e4c49 on the 4 workers of mbert-mix, 13:19-13:48 UTC, next to the training of
`fineweb2-nl-230m-chunk1024` (summaries: `gs://t5-dutch-english-usc2/mbert/rehydration/shares/fineweb2-rh3-w{0,1,2,3}.json`):

| host | documents in | kept | tokens in | tokens kept | on disk |
|---|---|---|---|---|---|
| 0 | 36,825,318 | 19,474,870 (52.88%) | 25,845,959,633 | 13,062,762,662 (50.54%) | 18.48 GB |
| 1 | 36,825,318 | 19,470,675 (52.87%) | 25,841,348,570 | 13,053,412,931 (50.51%) | 18.46 GB |
| 2 | 36,825,317 | 19,468,722 (52.87%) | 25,832,926,080 | 13,047,280,959 (50.51%) | 18.45 GB |
| 3 | 36,825,317 | 19,478,402 (52.89%) | 25,824,724,143 | 13,056,194,189 (50.56%) | 18.47 GB |
| all | 147,301,270 | 77,892,669 (52.88%) | 103,344,958,426 | 52,219,650,741 (50.53%) | 73.86 GB |

| weight | documents in | kept | tokens in | tokens kept |
|---|---|---|---|---|
| 1 | 84,434,461 | 28,139,284 (33.33%) | 64,854,686,115 | 21,624,647,743 |
| 2 | 39,341,465 | 26,228,041 (66.67%) | 23,687,499,163 | 15,792,229,850 |
| 3 | 23,525,344 | 23,525,344 (100%) | 14,802,773,148 | 14,802,773,148 |

- The documents per weight are exactly those of `distributions/nld_Latn.json`, and their sum weighted by w is
  233,693,423, FineWeb-2's own size of rehydrated nld_Latn: the cluster sizes read line up with FineWeb-2's. Kept
  77,892,669 against 77,897,808 expected (5,139 fewer; the binomial standard deviation is about 5,200).
- Alignment: on every host 135 segments (3 per file) of the staged share were identical to the source, 540 in all.
- Exposure at step 300,000, T = 300,000 x 124,267 = 37.28 B tokens (see below): the rehydrated set has
  H = sum of w x tokens = 156,638,003,885 tokens and K = 52,219,650,741 = 1.00013 x H / 3, so a document of weight w
  is seen 0.2380 w times on average in R, and 0.2380 w times when T tokens are drawn from the fully rehydrated set.
  Documents of weight 2 and 3 are shorter on average than those of weight 1 (602 and 629 tokens against 768), so
  rehydration raises the share of documents (52.9% kept) more than that of tokens (50.5%).
- The source files differ: the kept share falls with the file name, from 72.3% of the rows of `000_00000` to 33.4%
  of `004_00008` (`per_file` in the summaries), as the files with the lowest names hold most of the documents of
  weight 2 and 3 (on host 0, 41% of the rows of `000_00000` have weight 3). The trainer reads a host's files one after the other in a seeded order, so which documents R
  (and F) have seen by a step depends on that order; the expected exposure above holds over the whole share.
- The trainer of `fineweb2-nl-230m-chunk1024` on the same VM, steps per second without the 50-step intervals that
  hold a checkpoint: 4.41 in the 4 minutes before, 4.39 while the source was read (range requests, 18 minutes),
  4.28 while the kept rows were written (11 minutes, 4 processes per host; 2 to 3% slower), 4.38 in the 90 minutes
  after. Per host 1,605 to 1,650 s of `stage_share.py`; 31.3 to 31.6 GB of the boot disk free afterwards.

### Step 653,000 would need true repetition

At 300,000 steps a host reads about 9.32 B tokens (FineWeb-2 nl: 831,635 steps per pass of 103,344,958,426 tokens,
`gs://t5-dutch-english-usc2/mbert/token-counts/summary.json`, so 124,267 tokens per step over 4 hosts; F's data
state at 300,000 has 9,359,648,886 `tokens_read` on host 0, `<s>` and `</s>` included), about 0.71 of its kept
13.05 B. Continued to 653,000 steps a host needs about 20.29 B tokens: its kept share runs out near step 420,000,
and the trainer then starts a second pass over the same kept documents. That repeats the selection of weight w / 3
twice, which is not FineWeb-2's rehydration (w copies of each document). A run to 653,000 steps or longer on
rehydrated FineWeb-2 needs true repetition, e.g. w copies per document in the shares; this experiment does not test
that.

## Evaluation

DUMB, the fixed protocol of `gs://t5-dutch-english-usc2/mbert/masking/dumb/` (`scripts/dumb/finetune.py`): DBRD and
SICK-NL at `--max-length` 512, 3 epochs, LR 2e-5, AdamW weight decay 0.01, 4 chips per fine-tune, seeds 42, 43, 44;
DBRD 24 per chip x 4 chips x accumulation 2 (192 per update), SICK-NL 64 x 4 x 1 (256), as in F's result files.
Metric: `test_accuracy`. R's results named as F's: `fw2rh3-300000--{dbrd,sicknl}--seed{42,43,44}.json` in the same
directory.

Secondary, decides nothing: DBRD at 1024, as the results in `gs://t5-dutch-english-usc2/mbert/dbrd1024/dumb/` (e.g.
12 per chip x 4 chips x accumulation 4 = 192 per update, `--pad-to-multiple-of 128`). On 2026-11-07 that directory
has no results of F or X at step 300,000, so for this comparison all three arms need DBRD@1024 at 300,000; it is
reported as the decision rule below reports, without consequence.

## Decision rule

Per arm and task: the mean of `test_accuracy` over seeds 42, 43, 44, and the seed range, max minus min. Per task
the threshold is the larger of R's and F's seed range.

1. "schaadt" (harms) if R's mean is below F's by more than the threshold on DBRD or on SICK-NL. Checked first.
2. Otherwise "helpt" (helps) if R's mean is above F's by more than the threshold on SICK-NL.
3. Otherwise "geen verschil" (no difference).

"More than" is strict. A gain on DBRD alone is "geen verschil". R against X is reported the same way (threshold per
task the larger of R's and X's seed range) and has no consequence by itself. Whether a full rehydrated FineWeb-2 run
follows is Yeb's decision.

## Amendments
