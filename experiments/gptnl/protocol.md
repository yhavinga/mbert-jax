# Pre-registered protocol: GPT-NL Public Corpus in the training data of Dutch ModernBERT

Status: written on 2026-11-08, before the training of this experiment starts. Changes after its training has
started are added below under "Amendments", dated, and never replace the text above them.

Data code: branch `gptnl-mix` of mbert-jax (`experiments/gptnl/gptnl_mix.py`, run by `prepare_gptnl.sh`): the token
counts at `f855872`, the plan at `d3a85ee` and the shares at `7fb2392f17968411afce2a1dd367ca34532f056e`, whose
`experiments/gptnl/cxgptnl_plan.json` is that plan. Trainer: `~/mbert-jax` on mbert-mix at `0d92db4`. Scripts: `gs://t5-dutch-english-usc2/mbert/gptnl/scripts/`
(= `experiments/gptnl/` of that branch). Token counts: `gs://t5-dutch-english-usc2/mbert/gptnl/lengths/`. Plan and
selections: `gs://t5-dutch-english-usc2/mbert/gptnl/plan/`. Shares: `gs://t5-dutch-english-usc2/mbert/gptnl/shares/`.
Run: `gs://t5-dutch-english-usc2/mbert/culturax-gptnl-nl-230m-chunk1024/`. Commands: `commands.txt`.

## Question

Does a training mix of 70% CulturaX nl and 30% GPT-NL Public Corpus (in tokens; only its collections that are
originally Dutch) give a better Dutch encoder than CulturaX nl alone, at the same number of steps? Whether GPT-NL
goes into the final run is Yeb's decision.

## Arms

| arm | run | checkpoint | DBRD@512 mean [min-max] | SICK-NL@512 mean [min-max] |
|---|---|---|---|---|
| G | `culturax-gptnl-nl-230m-chunk1024` | step 300,000 | to be measured | to be measured |
| X | `culturax-nl-230m-chunk1024` (15%) | `ckpt_keep/300000` | 93.09 [92.71-93.42] | 76.85 [76.09-77.61] |
| F (context) | `fineweb2-nl-230m-chunk1024` (15%) | `ckpt_keep/300000` | 91.84 [91.52-92.33] | 75.05 [74.69-75.39] |
| R (context) | `fineweb2-nl-230m-chunk1024-rh3` | step 300,000 | training (`experiments/rehydration/protocol.md`) | training |

Accuracy in % (`test_accuracy` x 100). X: `gs://t5-dutch-english-usc2/mbert/masking/dumb/300000--{dbrd,sicknl}--seed{42,43,44}.json`;
F: `.../masking/dumb/fw2-300000--{dbrd,sicknl}--seed{42,43,44}.json`. `test_accuracy` is k / n with n = 2,112 for
DBRD (the full batches of its 2,224 test examples) and 4,864 for SICK-NL; k for seeds 42, 43, 44:

| arm | DBRD k of 2,112 | SICK-NL k of 4,864 | seed range DBRD | seed range SICK-NL |
|---|---|---|---|---|
| X | 1967, 1973, 1958 | 3701, 3738, 3775 | 15 (0.710) | 74 (1.521) |
| F | 1950, 1933, 1936 | 3667, 3651, 3633 | 17 (0.805) | 34 (0.699) |

### What differs between G and X

- The data: `--data-files-list` is `$HOME/cxgptnl_nl_files.txt` instead of `$HOME/culturax_nl_files.txt` (64 files
  per host of 30% GPT-NL and 70% CulturaX nl, below), and `--dataset-num-rows` is 47,092,713, the documents of the 4
  shares, instead of 117,392,666 (it feeds only the `data/passes` metric and the logged config).
- `--stop-at-step 300000`, after the same `--max-steps 653000`: G stops at step 300,000 on the cosine schedule of
  653,000 steps, so its learning rate at every step up to 300,000 is X's.
- The run name `culturax-gptnl-nl-230m-chunk1024` in `--checkpoint-dir`, `--wandb-project`, `--wandb-run-name`,
  `--tracking-sync-dir` and the log and exit files.
- The trainer code: G runs at `0d92db4` (as F and R); X ran on the code of its time (`code.tgz` in
  `gs://t5-dutch-english-usc2/mbert/culturax-nl-230m-chunk1024/scripts/`). F and R are compared with X across the
  same difference.

The same as X's: model, optimizer, learning rate and warmup, 4 hosts with 32 sequences of 1024 tokens per step,
packing with long documents split into pieces, seeds, `--checkpoint-every 10000`, the held-out eval
(`--eval-file $HOME/heldout.parquet`, every 25,000 steps, 50 batches), and `--dataset uonlp/CulturaX
--dataset-config nl` (with `--data-files-list` nothing is read through them; they are logged and part of the data
state's identity). No `--masking`, as X had none: the default of `0d92db4` is `uniform15`, X's 15% single-token
masking (F and R pass `--masking uniform15` explicitly, which is the same). Checked on 2026-11-08 (`commands.txt`,
step 4): the run script that `start_cxgptnl.sh` writes, with the 4 differences above undone, is byte-identical to
`run_culturax-nl-230m-chunk1024.sh` in X's scripts directory. The diff of the two:

```
$ diff <(tr ' ' '\n' < run_culturax-nl-230m-chunk1024.sh) <(PRINT_RUN_SCRIPT=1 bash start_cxgptnl.sh | tr ' ' '\n')
39c39
< 117392666
---
> 47092713
47a48,49
> --stop-at-step
> 300000
51c53
< gs://t5-dutch-english-usc2/mbert/culturax-nl-230m-chunk1024/ckpt
---
> gs://t5-dutch-english-usc2/mbert/culturax-gptnl-nl-230m-chunk1024/ckpt
53c55
< culturax-nl-230m-chunk1024
---
> culturax-gptnl-nl-230m-chunk1024
55c57
< culturax-nl-230m-chunk1024
---
> culturax-gptnl-nl-230m-chunk1024
57c59
< gs://t5-dutch-english-usc2/mbert/culturax-nl-230m-chunk1024/tracking
---
> gs://t5-dutch-english-usc2/mbert/culturax-gptnl-nl-230m-chunk1024/tracking
67c69
< $HOME/culturax_nl_files.txt
---
> $HOME/cxgptnl_nl_files.txt
69c71
< ~/logs/culturax-nl-230m-chunk1024.log
---
> ~/logs/culturax-gptnl-nl-230m-chunk1024.log
75c77
< ~/logs/culturax-nl-230m-chunk1024.exit
---
> ~/logs/culturax-gptnl-nl-230m-chunk1024.exit
```

The trainer reads a host's files one after the other in an order drawn from seed 42, through a shuffle buffer of
10,000 documents (`src/mmbjax/training/packing.py`). Each of G's files is a random interleaving of its host's GPT-NL
and CulturaX documents, so GPT-NL's share of what G reads is about 30% of the tokens from the start of training to
its end (per file 27.95 to 32.51% of the tokens, the summaries of the shares).

The held-out eval says little here: its documents are CulturaX's, which favours X, and they are training documents
of X (`nl_part_00097` is one of X's files on worker 0), while G leaves them out (below). Its loss is a monitor of
training, not a held-out comparison of G and X.

## The data

### GPT-NL Public Corpus

`GPT-NL/GPT-NL_Public_Corpus` on Hugging Face at revision `117b13ee7246aa5c83a42ff77e15d4d50b4156b9` (the current
commit on 2026-11-08, last modified 2026-11-04). License CC-BY-4.0 for the corpus, a `license` field per document.
The dataset card's own table (README.md, "Source Data") sorts the collections by how they were made: "Collected Data",
"Selected Data", "Synthetic Data" and "Filtered web-crawl Data".

Included, 20 collections that are originally Dutch by provenance (the folder names at that revision; texts before
1950 included): openraadsinformatie-part1, openraadsinformatie-part2, officiele-bekendmakingen, woogle, tweedekamer,
dpc, pbl, auditdienstrijk, rechtspraak, nationaal-archief, utrechts-archief, noordhollandsarchief, zeeuws-archief,
kb, kb-pd-books, kb-open-kranten, dansknaw, naturalis, wikiwijs, belgian-journal.

Excluded, the other 12: youtube-commons (transcripts translated into Dutch, "Synthetic Data"), wikidata (triples
turned into text, "Synthetic Data"), european-parliament, multi-eurlex and cc_eurovoc (EU documents, translations),
common-crawl (web text of unknown provenance), cc_openalex, cc_english-pd, cc_german-pd, cc_loc-pd-books,
american-stories (English or German, by the card's table) and cc_github_open_source (code). `gptnl_mix.py` refuses a revision whose
folders are not exactly these 32.

Within the included collections only documents whose `language` is `nl`; the values found are `nl`, `en`, `de`,
`fy`, `sv`, `af` and `da` (no document without a language). Left out besides, so that no text is in the shares twice: empty documents, every repetition of a text (exact, by xxh64
of its UTF-8 bytes, the first in path and row order kept) and texts that are also a document of CulturaX nl (any of
its 128 files, by the xxh64 of count_tokens.py).

Tokens are counted as the trainer tokenizes (`yhavinga/dutch-llama-tokenizer`, `split_special_tokens`, no special
tokens; `TokenizeTransform._tokenize` of `src/mmbjax/training/packing.py`), without `<s>` and `</s>`, as the lengths
parquets of `count_tokens.py` (branch token-counts) count them. Before counting, each worker compared its counts
with `count_tokens.py`'s on a seeded sample of 2,000 documents of CulturaX `nl_part_00000`: identical, 1,392,862
tokens. The lengths parquets (`gs://t5-dutch-english-usc2/mbert/gptnl/lengths/<file>.parquet`, one row per document
in the file's order: `text` = tokens, `language`, `license`, `utf8_bytes`, `xxh64`) cover all 530 files.

Documents and tokens per included collection, by language (`cxgptnl_plan.json`, "collections"):

| collection | files | documents | tokens | nl documents | nl tokens | other languages dropped: documents / tokens |
|---|---|---|---|---|---|---|
| openraadsinformatie-part1 | 120 | 1,425,477 | 9,396,390,182 | 1,423,800 | 9,376,868,748 | en 919 / 14,759,578; de 349 / 4,183,158; fy 409 / 578,698 |
| openraadsinformatie-part2 | 104 | 1,292,830 | 7,823,456,323 | 1,288,733 | 7,799,548,808 | en 935 / 12,582,186; fy 2,900 / 9,154,372; de 262 / 2,170,957 |
| officiele-bekendmakingen | 54 | 1,826,487 | 3,278,609,406 | 1,822,093 | 3,263,871,054 | en 3,801 / 13,532,281; de 562 / 1,063,934; fy 25 / 131,141; af 4 / 8,775; sv 2 / 2,221 |
| woogle | 54 | 4,358,740 | 3,182,763,283 | 4,088,931 | 3,017,656,269 | en 262,730 / 158,717,780; de 6,735 / 6,042,491; fy 266 / 271,528; sv 59 / 53,751; da 13 / 14,549; af 6 / 6,915 |
| tweedekamer | 28 | 233,676 | 1,592,404,293 | 229,600 | 1,511,424,639 | en 4,047 / 80,547,255; fy 17 / 324,078; de 12 / 108,321 |
| dpc | 3 | 127,715 | 82,649,375 | 127,715 | 82,649,375 | none |
| pbl | 1 | 489 | 23,264,639 | 341 | 16,602,732 | en 148 / 6,661,907 |
| auditdienstrijk | 1 | 560 | 6,068,512 | 555 | 6,040,648 | en 5 / 27,864 |
| rechtspraak | 40 | 918,799 | 2,514,310,191 | 918,634 | 2,512,355,088 | en 163 / 1,950,934; de 1 / 2,687; fy 1 / 1,482 |
| nationaal-archief | 13 | 1,932,515 | 1,186,608,465 | 1,925,582 | 1,178,311,558 | en 6,689 / 8,108,528; de 244 / 188,379 |
| utrechts-archief | 4 | 526,029 | 232,958,415 | 525,886 | 232,758,122 | de 137 / 196,575; en 5 / 3,690; sv 1 / 28 |
| noordhollandsarchief | 1 | 38,755 | 24,731,487 | 38,755 | 24,731,487 | none |
| zeeuws-archief | 1 | 33,897 | 39,528,324 | 33,897 | 39,528,324 | none |
| kb | 20 | 6,705 | 526,917,549 | 6,704 | 526,894,439 | de 1 / 23,110 |
| kb-pd-books | 15 | 2,250 | 393,706,640 | 2,245 | 393,226,439 | de 4 / 414,362; en 1 / 65,839 |
| kb-open-kranten | 50 | 1,573,792 | 1,748,380,190 | 1,565,896 | 1,740,570,443 | en 7,124 / 7,025,293; de 772 / 784,454 |
| dansknaw | 1 | 57,104 | 23,545,850 | 52,880 | 22,597,443 | en 4,220 / 947,341; de 4 / 1,066 |
| naturalis | 4 | 12,430 | 196,629,949 | 2,652 | 19,335,140 | en 9,036 / 158,045,093; de 742 / 19,249,716 |
| wikiwijs | 1 | 124,501 | 28,613,023 | 119,187 | 27,289,978 | en 4,606 / 1,048,450; de 697 / 273,233; fy 11 / 1,362 |
| belgian-journal | 15 | 209,088 | 782,211,420 | 208,242 | 778,148,498 | de 835 / 3,998,063; en 11 / 64,859 |
| all | 530 | 14,701,839 | 33,083,747,516 | 14,382,328 | 32,570,409,232 | en 304,440 / 464,088,878; de 11,357 / 38,700,506; fy 3,629 / 10,462,661; sv 62 / 56,000; af 10 / 15,690; da 13 / 14,549 |

Of the documents in `nl`, left out and drawn (eligible: `nl`, not empty, the first of its text, not in CulturaX):

| collection | nl documents left out: empty / repeated / in CulturaX | eligible documents | eligible tokens | selected documents | selected tokens | share of the selected tokens | share of the eligible tokens |
|---|---|---|---|---|---|---|---|
| openraadsinformatie-part1 | 0 / 0 / 0 | 1,423,800 | 9,376,868,748 | 537,818 | 3,539,403,259 | 28.83% | 28.84% |
| openraadsinformatie-part2 | 0 / 23,844 / 0 | 1,264,889 | 7,737,344,181 | 477,733 | 2,915,622,056 | 23.75% | 23.80% |
| officiele-bekendmakingen | 0 / 0 / 0 | 1,822,093 | 3,263,871,054 | 688,166 | 1,229,260,712 | 10.01% | 10.04% |
| woogle | 0 / 0 / 0 | 4,088,931 | 3,017,656,269 | 1,544,657 | 1,140,186,155 | 9.29% | 9.28% |
| tweedekamer | 0 / 0 / 0 | 229,600 | 1,511,424,639 | 87,157 | 573,251,468 | 4.67% | 4.65% |
| dpc | 0 / 0 / 113 | 127,602 | 82,633,442 | 48,056 | 30,949,523 | 0.25% | 0.25% |
| pbl | 0 / 0 / 0 | 341 | 16,602,732 | 116 | 4,698,901 | 0.04% | 0.05% |
| auditdienstrijk | 0 / 0 / 0 | 555 | 6,040,648 | 202 | 2,276,277 | 0.02% | 0.02% |
| rechtspraak | 0 / 0 / 0 | 918,634 | 2,512,355,088 | 347,115 | 948,594,489 | 7.73% | 7.73% |
| nationaal-archief | 1,469 / 6 / 0 | 1,924,107 | 1,178,311,464 | 725,558 | 444,433,031 | 3.62% | 3.62% |
| utrechts-archief | 0 / 0 / 0 | 525,886 | 232,758,122 | 199,159 | 88,201,968 | 0.72% | 0.72% |
| noordhollandsarchief | 0 / 0 / 0 | 38,755 | 24,731,487 | 14,657 | 9,314,215 | 0.08% | 0.08% |
| zeeuws-archief | 0 / 0 / 0 | 33,897 | 39,528,324 | 12,909 | 15,045,568 | 0.12% | 0.12% |
| kb | 0 / 0 / 0 | 6,704 | 526,894,439 | 2,491 | 199,126,873 | 1.62% | 1.62% |
| kb-pd-books | 0 / 0 / 0 | 2,245 | 393,226,439 | 868 | 156,979,817 | 1.28% | 1.21% |
| kb-open-kranten | 2,954 / 253 / 0 | 1,562,689 | 1,740,551,613 | 590,094 | 656,775,251 | 5.35% | 5.35% |
| dansknaw | 0 / 0 / 0 | 52,880 | 22,597,443 | 19,903 | 8,570,403 | 0.07% | 0.07% |
| naturalis | 0 / 0 / 0 | 2,652 | 19,335,140 | 1,037 | 7,603,859 | 0.06% | 0.06% |
| wikiwijs | 0 / 0 / 0 | 119,187 | 27,289,978 | 44,927 | 10,284,829 | 0.08% | 0.08% |
| belgian-journal | 0 / 0 / 0 | 208,242 | 778,148,498 | 79,057 | 294,979,077 | 2.40% | 2.39% |
| all | 4,423 / 24,103 / 113 | 14,353,689 | 32,508,169,748 | 5,421,680 | 12,275,557,731 | 100% | 100% |

Repetitions are texts that occur earlier in path and row order (23,844 in openraadsinformatie-part2, 62,223,551
tokens in all); the empty documents have 0 tokens. Of the 33.08 B tokens of the included collections, 513 M are in
other languages than `nl` (464 M English, most in woogle, naturalis and tweedekamer) and 62 M are repetitions; 32.51 B
are eligible, of which the shares take 12.28 B (37.8%). A GPT-NL document has 2,264 tokens on average, a CulturaX
one of the shares 687, so GPT-NL is 11.5% of the documents of a share and 30.0% of its tokens.

The sample: every eligible document is in a permutation drawn with `numpy.random.default_rng([42, 0])`, dealt to the
4 hosts in turn; each host takes its documents in that order until it has 3,068,854,272 tokens (30% of its 10.23 B,
below). No reweighting between collections: a collection's share of the sample is its share of the eligible tokens,
up to sampling noise (table above). No document is on two hosts.

The `license` field of the drawn documents, as the corpus writes it: `Public domain` 2,874,086, `public-domain`
2,354,498, `CC-0 1.0` 147,016, `CC-BY` 46,080.

DBRD (book reviews of Hebban) and SICK-NL (a translation of SICK, sentence pairs from image and video captions) are
not in any included collection by provenance: government, parliament, courts, archives, the KB's books and
newspapers, archaeology (DANS-KNAW), biology (Naturalis), school material (Wikiwijs) and Belgian company bylaws.
No text in the shares was written by Claude or GPT.

### CulturaX nl

`uonlp/CulturaX`, config nl, revision `6a8734bc69fefcbb7735f4f9250f43e4cd7a442e` (that of X's staging and of
`count_tokens.py`). Each host takes its documents from its own 32 files of X, by worker number: the split that
`stage.py` (`~/stage.py` on mbert-mix) made with `split_dataset_by_node` by the worker number in the hostname,
recomputed on 2026-11-08 with the trainer's venv as `count_tokens.py`'s `long_run_file_order` does (`CULTURAX_FILES`
in `gptnl_mix.py`). Their documents per worker are those of `long_run_hosts` in
`gs://t5-dutch-english-usc2/mbert/token-counts/summary.json` (29,348,172, 29,348,135, 29,348,224, 29,348,135), which
the plan checks. They are also the 4 sets of `data_files` in X's data states at step 300,000
(`gs://t5-dutch-english-usc2/mbert/culturax-nl-230m-chunk1024/ckpt_data_state/300000/process_<k>.json`), but those are
per process index, which by then was not the worker number: `process_<k>.json` has the files of worker (k + 2) mod 4.
So G's CulturaX documents on a worker are some of those X had on that worker. The selection: every row of a file has
a draw `numpy.random.default_rng([42, 1, <file number>]).random(<rows of the file>)`; a host takes the rows of its files in
the order of their draws until it has 7/3 of its GPT-NL tokens (rounded up), so that GPT-NL is 30.000% of its
tokens. The 40,000 documents of the trainer's held-out file (`~/heldout.parquet`, the first 40,000 of
`nl_part_00097`, md5 `b2f579a8...`) are passed over, by their xxh64. Tokens are those of `count_tokens.py`
(`gs://t5-dutch-english-usc2/mbert/token-counts/culturax/counts/`).

### The shares

Per host 330,000 x 128 x 1024 x 0.946 / 4 = 10,229,514,240 tokens (300,000 steps and 10% to spare; 0.946 is
CulturaX nl's packing fill, 0.9464 in the token-counts summary), 30% GPT-NL. The 300,000 steps need at most
300,000 x 128 x 1024 / 4 = 9,830,400,000 tokens per host with `<s>` and `</s>` even at a packing fill of 1, so no host
starts a second pass over its share before step 300,000.

`prepare_gptnl.sh mix` on each worker read the host's rows from the Hugging Face files by range requests, checked
each document's UTF-8 length against the counts (and the language of GPT-NL's), and wrote 64 parquet files with the
column `text` (zstd 9, row groups of 50,000), in the order of a permutation of the host's documents drawn with
`numpy.random.default_rng([42, 2, <host>])` (GPT-NL's documents first, then CulturaX's, each in file and row order,
get positions in that permutation). Summaries: `gs://t5-dutch-english-usc2/mbert/gptnl/shares/cxgptnl-w{0,1,2,3}.json`.

| host | GPT-NL documents | GPT-NL tokens | CulturaX documents | CulturaX tokens | of the host's CulturaX tokens | tokens | with `<s>`, `</s>` | GPT-NL | files | GB on disk |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 1,352,001 | 3,068,856,272 | 10,418,423 | 7,160,665,242 | 35.5% | 10,229,521,514 | 10,258,409,062 | 30.0000% | 64 | 13.478 |
| 1 | 1,360,074 | 3,068,854,702 | 10,418,562 | 7,160,660,998 | 35.5% | 10,229,515,700 | 10,258,414,720 | 30.0000% | 64 | 13.496 |
| 2 | 1,340,440 | 3,068,856,667 | 10,413,496 | 7,160,665,723 | 35.5% | 10,229,522,390 | 10,258,382,956 | 30.0000% | 64 | 13.480 |
| 3 | 1,369,165 | 3,068,990,090 | 10,420,552 | 7,160,978,441 | 35.5% | 10,229,968,531 | 10,258,885,221 | 30.0000% | 64 | 13.505 |

All 4: 47,092,713 documents, 40,918,528,135 tokens (spread between hosts 0.0044%). Of each host's own CulturaX
documents 35.5% are in its share; the 40,000 held-out documents were passed over on host 0, which has
`nl_part_00097`. With `<s>` and `</s>` a share has 10.258 B tokens, more than the 9.83 B that 300,000 steps can read
at a packing fill of 1; at CulturaX's fill of 0.946 it lasts ~331,000 steps. GPT-NL's longer documents fill more of
the rows with full pieces, so G's fill will be somewhat higher than X's and a share lasts somewhat fewer steps, at
least 313,000.

Checked afterwards, independently of `gptnl_mix.py` (a script of this session, in
`gs://t5-dutch-english-usc2/mbert/gptnl/verify/` with its outputs): on every host the xxh64 of each document of the 64 files, in list order, is that of the document the plan puts at
that position (GPT-NL's from the lengths parquets, CulturaX's from `count_tokens.py`), and so are the UTF-8 lengths;
no document of the held-out file is in a share; the 4 shares hold 47,092,713 documents with 47,092,713 distinct
texts. Every file has the single column `text` (string), zstd, row groups of at most 50,000.

How it went, on 2026-11-08 next to the training of `fineweb2-nl-230m-chunk1024-rh3` (R) on the same VM
(`commands.txt`):

- count (`f855872`), 10:45-11:17 UTC: started with 16 processes per host and stopped by me at 10:48, as R's step
  time rose 5-8% (4.13 steps per second against 4.41 before); continued with 8 processes per host, the 124 files
  already in the bucket skipped. 1,605-1,691 s per host for the second start. Each host first found its token counts
  of 2,000 CulturaX documents equal to `count_tokens.py`'s.
- plan (worker 0): a first start, at `e8fdfa7`, stopped on its check against `long_run_hosts`, as it should: its
  split of CulturaX was taken from X's data states, whose process indices were not the worker numbers by step
  300,000. Fixed in `d3a85ee`; the second start, at `d3a85ee`, ran 11:19-11:22.
- mix (`7fb2392`), 11:23-11:35: 324-330 s reading (8 files at a time per host), then 64 files written per host (4 at
  a time, ~23 s each); 712-744 s per host. 15 to 16 GB of the boot disk free afterwards (27 to 29 GB before).
- R's speed (steps per second between samples, checkpoints included; the trainer's own mean `Step:` time per 1,000
  steps): before 4.41 (22,300-24,500) and 0.2252 s (18,000-24,999); counting with 16 processes 4.13; with 8
  processes 4.30 and 0.2304 s (+2.3%); plan and mix 4.31 and 0.2285 s (+1.5%).

## Evaluation

DUMB, the fixed protocol of `gs://t5-dutch-english-usc2/mbert/masking/dumb/` (`scripts/dumb/finetune.py`): DBRD and
SICK-NL at `--max-length` 512, 3 epochs, LR 2e-5, AdamW weight decay 0.01, 4 chips per fine-tune, seeds 42, 43, 44;
DBRD 24 per chip x 4 chips x accumulation 2, SICK-NL 64 x 4 x 1, as in X's result files. Metric: `test_accuracy`.
G's results: `cxgptnl-300000--{dbrd,sicknl}--seed{42,43,44}.json` in the same directory.

Secondary, decides nothing: DBRD at 1024 (as the results in `gs://t5-dutch-english-usc2/mbert/dbrd1024/dumb/`) and
DBRD and SICK-NL at fine-tune LR 5e-5 (as `gs://t5-dutch-english-usc2/mbert/masking/dumb/lrsweep-v4/`), each against
X at 300,000 measured the same way. They are reported as the decision rule below reports, without consequence.

## Decision rule

Per arm and task: the mean of `test_accuracy` over seeds 42, 43, 44, and the seed range, max minus min. Per task the
threshold is the larger of G's and X's seed range. All of it is computed exactly, with the accuracies as fractions
k / n (`fractions.Fraction`; k = round(test_accuracy x n), which must give back `test_accuracy`), so that no rounding
decides.

1. "GPT-NL schaadt" (harms) if G's mean is below X's by more than the threshold on DBRD or on SICK-NL. Checked first.
2. Otherwise "GPT-NL helpt" (helps) if G's mean is above X's by more than the threshold on SICK-NL.
3. Otherwise "geen verschil" (no difference).

"More than" is strict. A gain on DBRD alone is "geen verschil". G against F and R is reported the same way and has no
consequence by itself. Whether GPT-NL goes into the final run is Yeb's decision.

## Amendments
