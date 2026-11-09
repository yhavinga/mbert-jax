# Handover: the Dutch FLAN of arm F-nl (state of 8 Nov 2026, 14:50 UTC by the VMs' clock)

Branch `flan-nl` of github.com/yhavinga/mbert-jax; everything of the sub-project is in `experiments/flan-nl/` and
`tests/test_flan_nl.py`. Scripts in the bucket: `gs://t5-dutch-english-usc2/mbert/flan-nl/scripts/` (copies of
`experiments/flan-nl/`). `R=gs://t5-dutch-english-usc2/mbert/flan-nl` below. The steps are numbered as in
`commands.txt`; the protocol is `protocol.md`.

## What runs where

- mbert-corpus (v4-32, us-central2-b), all 4 workers: the 27B translation, started 14:31 UTC on 8 Nov, one process
  per worker, each a TPU of its own 4 chips:
  `/dev/shm/flan_nl/venv/bin/python -u /dev/shm/flan_nl/code/translate.py --segments /dev/shm/flan_nl/segments/segments-w<N>.parquet --out $R/translations/27b/w<N>/<class> ...`
  started by `~/run_flan_nl_translate.sh` (written by `translate.sh`), class after class: tiny, small, medium, long,
  xlong. Log: `~/logs/flan_nl_translate.log` (one `SHARD {...}` line per stored shard, a `DONE {...}` line per class,
  `EXIT <code>` at the end); at the end a copy goes to `$R/logs/flan_nl_translate-w<N>.log`.
  Plan: `$R/plan2/` (1,811,525 distinct segments, 158.9 M source tokens; the 31 template strings protected).
  Shards: `$R/translations/27b/w<N>/<class>/shard-NNNNN.parquet` (id, text_nl, ended, tokens_in, tokens_out).
- The 12B run is stopped (14:30 UTC, `pkill -9`), at w0 75, w1 75, w2 61, w3 76 shards (all tiny and part of small, of
  its plan `$R/plan/`). Its shards stay in `$R/translations/12b/`, its logs in `$R/logs/flan_nl_translate12b-w<N>.log`.
  Not used further.
- mbert-mix: not touched.
- Nothing else of this sub-project runs. (Pilots, benchmarks and the 12B run have ended.)

## Progress and ETA

```
SSH='gcloud compute tpus tpu-vm ssh mbert-corpus --zone us-central2-b --project t5-dutch-english --quiet'
$SSH --worker=all --command "grep -ac '^SHARD' ~/logs/flan_nl_translate.log; grep -a '^SHARD' ~/logs/flan_nl_translate.log | tail -n 1 | cut -c1-200; grep -a '^DONE\|^EXIT\|Traceback' ~/logs/flan_nl_translate.log" < /dev/null
for w in 0 1 2 3; do echo w$w $(for c in tiny small medium long xlong; do gsutil ls $R/translations/27b/w$w/$c/ 2>/dev/null | wc -l; done); done
```

Shards per worker when done: tiny 57, small 31, medium 12, long 6, xlong 8 (114 in all; 4,096 segments a shard, the
last of a class fewer). Shards run in a fixed random order within a class, so the mean time of the first ones is a fair
sample of the class.

Measured (8 Nov, v4 host, 27B): tiny 134-141 s per warm shard, 644-676 output tokens/s and ~3,300-3,400
prompt+output tokens/s per host (tiny is prefill-bound: ~32 ms per prompt). Decode steps of the five class shapes
(bench, `$R/dev/bench3.py`): tiny 384 slots x 224 64.7 ms, small 224 x 512 61.0 ms, medium 136 x 832 55.9 ms, long
104 x 1,152 48.8 ms, xlong 80 x 1,472 48.5 ms, peak 28.9 GiB of 30.7 per chip; prefill+insert 38.6 ms at 256 tokens,
48.8 at 384, 61.1 at 512. Projection per worker: tiny ~2.1 h, small ~2.1 h, medium ~2.0 h, long ~1.6-1.9 h, xlong
~4.5-4.8 h, plus ~5 minutes of loading and compiling per class: ~12 hours, so the end around 02:30-03:30 UTC on 9 Nov
(xlong decides; its first warm shard gives the best estimate).

## What "done" looks like

On every worker `~/logs/flan_nl_translate.log` has five `DONE` lines (one per class) and ends in `EXIT 0`;
`$R/logs/flan_nl_translate-w<N>.log` exists; the counts above are 57 31 12 6 8 for w0..w3; no process holds the TPU
(`sudo lsof -t /dev/accel* /dev/vfio/*` empty).

## If a worker fails or reboots

- `EXIT` other than 0, or a hung process: read the traceback in the log; `pkill -9 -f '[t]ranslate.py'` (the brackets
  keep pkill from matching its own ssh shell); then on that worker `bash /dev/shm/flan_nl/code/translate.sh`. Stored
  shards are skipped (by name, in GCS); at most the shard in progress is lost.
- After a reboot /dev/shm is empty: first step 1 of `commands.txt` on that worker (`bash /dev/shm/flan_nl/code/setup.sh`
  after copying `$R/scripts/*` to `/dev/shm/flan_nl/code/`; ~2 minutes; MODEL=27b is the default and fetches the 27B
  tokenizer), then `translate.sh` as above. The checkpoint is read from `$R/translategemma-27b-maxtext/0/items`. The
  compilation cache in `/dev/shm/flan_nl/jax_cache` is lost then too (first shard of each class ~2 minutes slower).
- Before any start: the TPU must be free (`translate.sh` refuses otherwise).

## Remaining steps (after "done")

1. Assembly on worker 0 (step 8 of `commands.txt`, ~15 minutes, CPU). Needs `/dev/shm/flan_nl/en/w{0,1,2,3}.parquet`
   and `/dev/shm/flan_nl/plan2/` on worker 0 (there now; after a reboot copy them from `$R/en/` and `$R/plan2/`) and
   `$CODE/templates_nl.json` (Yeb's table, 31 keys, in the branch and in `$R/scripts/`):
   ```
   CODE=/dev/shm/flan_nl/code; VENV=/dev/shm/flan_nl/venv/bin/python
   $SSH --worker=0 --command "gsutil -q cp '$R/scripts/*' $CODE/ && mkdir -p /dev/shm/flan_nl/out27b /dev/shm/flan_nl/nl && gsutil -m -q rsync -r $R/translations/27b /dev/shm/flan_nl/out27b && $VENV -u $CODE/flan_nl.py assemble --english /dev/shm/flan_nl/en/w0.parquet /dev/shm/flan_nl/en/w1.parquet /dev/shm/flan_nl/en/w2.parquet /dev/shm/flan_nl/en/w3.parquet --plan /dev/shm/flan_nl/plan2 --translations /dev/shm/flan_nl/out27b --templates-nl $CODE/templates_nl.json --out /dev/shm/flan_nl/nl --processes 96 > ~/logs/flan_nl_assemble.log 2>&1; echo EXIT \$? >> ~/logs/flan_nl_assemble.log; gsutil -m -q cp /dev/shm/flan_nl/nl/* $R/nl/" < /dev/null
   gsutil cat $R/nl/report.json
   ```
   It refuses when a segment has no translation, or without the table. It applies: the task filters again (lists in
   `flan_nl.py`, `DROPPED_TASKS`; the 6 Dutch NIv2 tasks kept as in F), the two repairs (danda, a final period the
   source lacks), the segment checks (truncated, empty, length ratio, commentary, alternatives, foreign script, not
   Dutch), the substring rule (extractive answers), and the templates: per example the register (u/je from its MT, else
   by hash), the variant its MT names, else seeded; Head:/Tail: by the pair its MT names (beginzin/eindzin, inleidende
   zin/afsluitende zin, kop/staart, Head/Tail, begin/einde), else Beginzin/Eindzin or Kop/Staart by hash.
   `nl/report.json` has per host the rows in, translated and dropped per reason, `segments_repaired`,
   `segments_unusable`, `template_variants` (rows per variant; the row count of "Tail:" is the sum of its entry),
   `template_registers`, `head_tail_pairs`, `variant_decisions` (lexical against seeded). Output: `$R/nl/w<N>.parquet`.
2. The Dutch mixes (step 9 of `commands.txt`, all 4 workers at once, CPU, ~5 minutes each in the dry run):
   ```
   PY='$HOME/mbert-dumb/.venv/bin/python'
   $SSH --worker=all --command "gsutil -q cp '$R/scripts/*' /dev/shm/flan_nl/code/ && (PY=$PY setsid nohup bash -c 'bash /dev/shm/flan_nl/code/build_mix.sh > ~/logs/flan_nl_mix.log 2>&1; echo EXIT \$? >> ~/logs/flan_nl_mix.log' > /dev/null 2>&1 < /dev/null &)" < /dev/null
   $SSH --worker=all --command 'tail -n 3 ~/logs/flan_nl_mix.log' < /dev/null
   for w in 0 1 2 3; do gsutil cat $R/mix/w$w.json | python3 -c "import json, sys; m = json.load(sys.stdin); f = m['flan']; print('w$w', m['rows'], f['f_rows'], f['rows'], round(m['flan_token_share'], 4), round(m['f_flan_token_share'], 4))"; done
   ```
   `build_mix.sh` checks the free disk (>= 4 GB; 8.0-9.5 GB free now) and that `~/mbert-flan`'s builder and `src/` are
   those of F's commit bd6ddca (the checkout is at 2411eb8, which only changed the zero-shot eval). It writes
   `~/flan_decay_nl/mix-<build>.parquet` (~1.4 GB), `manifest-<build>.json`, `ids-<build>.parquet` (Dutch position ->
   F's position), `~/flan_decay_nl_files.txt`, and copies the manifest and the id map to `$R/mix/w<N>.json` and
   `$R/mix/w<N>-ids.parquet`. Verification inside: the order is made by F's own `trainer_shuffle_order` and read back
   through the trainer's route (`verify_order`), the web rows are compared with F's byte for byte, the first 2,000
   FLAN rows' token counts with the trainer's `TokenizeTransform`; Dutch examples over 1,024 tokens are dropped
   (`dropped_too_long`). The manifest has the Dutch FLAN token share against F's 0.1000.
3. `start_flan_nl.sh`: `experiments/flan-nl/start_flan_nl.sh` and `$R/scripts/start_flan_nl.sh`; step 10 of
   `commands.txt`. Not to be started by this sub-project (Yeb decides).
4. Before any run of F-nl: put the final commit in `COMMIT_NL` of `commands.txt`, copy `protocol.md` to
   `$R/protocol.md` (step 0).
5. Publish FLAN-nl on Hugging Face after step 1: section "Publish" below, step 11 of `commands.txt`.

## Publish: FLAN-nl on Hugging Face

Yeb's consent (8 Nov, evening): the Dutch rows go public as the dataset `yhavinga/flan-nl`, with the ids of their source
rows so that every row leads back to Open-Orca/FLAN. Open-Orca/FLAN has no id column; (revision, file, row) is the id.
Code: `experiments/flan-nl/publish.py`, tests `tests/test_flan_nl_publish.py`. Copies: `$R/publish/` (`publish.py`,
and what it imports: `flan_nl.py` and F's `scripts/build_decay_mix.py`, both of this branch); on worker 0 in
`/dev/shm/flan_nl/publish/` (not in `code/`, not in `$R/scripts/`). `yhavinga/flan-nl` did not exist on 8 Nov 20:11 UTC
(neither as dataset nor as model; checked with Yeb's token on worker 0).

1. `link`: done, 8 Nov 20:21-20:29 UTC on worker 0 (CPU, `nice -n 19`, one process). All 1,357,022 FLAN rows of F
   (w0 340,016, w1 339,792, w2 338,986, w3 338,228) have their `flan_file` and `flan_row` in Open-Orca/FLAN at
   `6845b1b3b53c6d5c5b1e49767ed759df3fc246cc`, plus `_template_idx`, `_task_source`, `_task_name`, `_template_type`
   of that row. Per host it read the shards F's `select_flan` read (`host_units`, F's order, `shards_read` of F's
   manifests `gs://t5-dutch-english-usc2/mbert/flan-decay/mix/w<N>.json`: every unit of every folder) and matched on
   (task, inputs, targets); 1,229 rows share their content with another row of their host, and such twins took the
   equal source rows one by one in shard order. Each shard was downloaded to `/dev/shm/flan_nl/oo-flan/` and deleted
   after reading (the directory holds only empty folders now). Output: `/dev/shm/flan_nl/links/` and `$R/links/`
   (`w<N>.parquet`: host, position, flan_file, flan_row, template_idx, task_source, task_name, template_type;
   `report.json`); log `$R/logs/flan_nl_link.log`. "Done" = the log ends in `EXIT 0` and `$R/links/report.json` has
   `rows` = `linked` = 1,357,022.
   Checked after (`$R/publish/dev/verify_links.py`, `spot_check.py`): every source row once; each in a shard (for
   `cot_zsopt_data` a row residue) of its own host that F read; folder, task and template type as in the extract; no
   (folder, task) over F's 500 per host; 15 random links, fetched the way the card says, equal the English extract.
   Effect on the translation: w0's medium `shard-00010`, which ran during `link`, took 800.3 s against 789.2, 791.7 and
   794.3 s on w1-w3 (+1.1%; the shard before, `shard-00002`, +0.4%): within the spread between workers.
2. `build`, after the assembly (step 8) ends in `EXIT 0`: step 11b of `commands.txt` (worker 0, CPU, `nice -n 19`,
   32 processes, ~2 minutes). Reads `/dev/shm/flan_nl/nl/w<N>.parquet` and `nl/report.json`, `links/`,
   `plan2/examples.parquet` (`translated` = status is not `dutch_task`) and the English extracts
   `/dev/shm/flan_nl/en/w<N>.parquet` (`--english`, for the checks of this publication only); writes `--out`
   (`data/train-0000k-of-0000n.parquet`, zstd, each under 400 MB, sorted by (flan_file, flan_row); `README.md`, the
   dataset card with every number from the data and the reports; `build_report.json`, the rows the checks left out
   per check, by task and by id, not uploaded). Columns: id (`w<N>-<position>`), inputs, targets, translated,
   task_name, task_source, folder, template_type, template_idx, flan_revision, flan_file, flan_row. No English text in
   the output. It refuses, before writing anything, a Dutch row without a link, a link of another task, a row whose
   plan status is neither `translate` nor `dutch_task`, a translated row without its English row (same host,
   position and source), a report that does not add up to the data (FLAN rows of F = links; translated rows = kept +
   left out by the checks; FLAN rows minus dropped minus left out = rows), and a drop reason the card cannot explain;
   and it refuses an `--out` that is not empty (a rerun goes to a new directory).
   Dry run, done: the 27B pilot rows (filtered to plan2's tasks, their report padded to add up) with the real links:
   `/dev/shm/flan_nl/publish/dry/hf/`, card `$R/publish/dev/dry-README.md` (its counts are the pilot's, not real).
   Builds (9 Nov, the VMs' clock):
   - `hf` (03:17, step 11b without checks): `/dev/shm/flan_nl/hf/`, `$R/hf/`, 1,031,764 rows. Not to be uploaded: the
     main session's sample found decode loops and untranslated English in it.
   - `hf2` (04:24) and `hf3` (04:29): the same data and `build_report.json` (md5 equal), 1,025,582 rows (1,019,417
     translated, 6,165 Dutch-task), 4 files, 392 MB. Only the card differs: hf2's says the 6,182 rows left out were in
     the training data, but 663 of them are not in F-nl's mixes (`$R/mix/w<N>-ids.parquet`: the mix builder dropped
     them as longer than 1,024 tokens; 5,519 are in the mixes). hf3's card says "Dutch rows of the experiment".
     **Upload hf3**: `/dev/shm/flan_nl/hf3/`, `$R/hf3/`; log `$R/logs/flan_nl_publish_build3.log`.
   The checks of this publication (`publish.py`: `decode_loop`, `english_left`, `PUBLICATION_CHECKS`). Only for this
   dataset: `flan_nl.py` and F-nl's training data stay as the protocol fixed them (translation errors are noted
   there, not corrected). Each field (inputs, targets) of a translated row against the same field of its English row;
   a row that fails both is counted under `decode_loop`.
   - `decode_loop`: a sequence of 1-4 lowercase tokens (`\w+|[^\w\s]`) with a word in it, 5 or more times back to back,
     or of 5-16 tokens 3 or more times, also counted over the words alone (loops vary their punctuation); excused when
     the English has a repeat of at least 3 and half as many copies (periods up to 32, also over its words alone and
     inside one token, e.g. emoji written out as `loudlycryingface` x5). Rows: 4,957.
   - `english_left`: a run of words of the English row, none in `DUTCH_WORDS`, with at least 3 words of
     `ENGLISH_WORDS` and at least 5 other lowercase words of 3+ letters not in `SHARED_WORDS`; capitalized words
     (names, titles) count for nothing but do not end the run; URLs, e-mail, @names, #tags removed first. Rows failing
     it: 1,519, of which 1,225 not already under `decode_loop`.
   - Together 6,182 rows (0.60% of the 1,025,599 translated rows); 294 fail both; 238 tasks, most duorc (plots),
     cnn_dailymail, wiki_hop. The known examples w2-497773, w3-976648, w3-1242456, w0-51286 are left out, w1-1353724
     (samsum) is in.
   - How the thresholds were chosen (on the hf rows with their English, `/dev/shm/flan_nl/publish/filter/` on worker
     0, scripts also in `gs://t5-dutch-english-usc2/hammond/scratch-scripts/flan-nl-filter/`; samples read by the
     agent of 9 Nov 03:20-04:30). `decode_loop`, random rows flagged: 30/30 loops at 6 copies (two samples); 29/30
     with 5 copies while sequences of punctuation alone still counted (the miss, w2-1058347, 6 dots, `Bess...` /
     `...is` for the English `bess .` / `.`; of the 29 rows flagged by punctuation alone ~2 were loops, the rest
     tables the translation drew and emoji, so only sequences with a word count now); rows only caught by the
     words-alone count 25/25; long periods beyond the short rule 27/30 (two of the three misses,
     `C'mon........C'MON` and emoji written out, are now excused; the third, a dialogue whose English has a typo,
     `no deawl`, remains). Just below: 5 copies 20/20 loops (so 5, not 6), 4 copies ~10/20 (`... .`, `_ .`), 2 copies
     of 5-16 tokens would add 12,120 rows. `english_left`: 29/30 with the final rule (the miss: quoted song lyrics);
     earlier variants (4 other words, no minimum of letters) 28/30 and 26/30, their misses quoted lyrics, song and
     book titles; just below the final rule (4 other words) 15/20, the misses song titles (Gilbert and Sullivan, "May
     the good lord bless and keep you"). Estimated precision: decode_loop ~97-100%, english_left ~95%. The remaining
     false positives are lyrics, quotations and titles that the translation kept in English; failed rows below the
     thresholds remain (the card says so).
3. Read `$R/hf3/README.md`, then `upload` of hf3 (step 11c): `create_repo(repo_type="dataset", private=False,
   exist_ok=True)` and `upload_folder` of README.md and data/ (not build_report.json); it prints the URL. Built and
   tested with a fake hub, not run. `yhavinga/flan-nl` not created yet.

## For the final report

Pilot (300 random examples, seed 0, plan `$R/plan/`): 27B 249 translated (`$R/pilot/nl/`), 12B 232 (`$R/pilot/nl12b/`);
the 15 pairs were printed from `pilot/nl12b` (12B) by the scratch script; 27B pairs can be printed the same way from
`$R/pilot/nl/w*.parquet` joined with `$R/en/w*.parquet` on (host, position). Plan counts: `$R/plan2/plan.json` (task
filters: word_game 42,926, english_form 28,701, code 16,242, token_level 15,468, word_relations 14,081, english_words
4,208, translation 1,754 rows; 6,165 Dutch-task rows kept; 69 with a sentence over 512 tokens; 49,417 examples split;
1,032,546 examples with a template string). Template table: `$R/templates/top30.md` = `templates_top30.md`.

## Open TODOs and risks

- Assembly, mixes, report: above. Disk: fine for one mix per host (1.4 GB of 8-9.5 GB free); nothing else may grow there.
- A reboot or maintenance event of mbert-corpus empties /dev/shm (venv, code, segments, plan2/en on worker 0); everything
  is in the bucket and `setup.sh` rebuilds the environment.
- Expected losses (pilot): ~15% of examples by the substring rule (extractive answers whose translation is not a stretch
  of the translated passage), ~1-3% by segment checks.
- Not done: packed prefill for the tiny class (would save ~0.85 h per worker; MaxText's packed path needs more repairs).

## Commits and tests

Main commits on `flan-nl` (from 2411eb8): c57f726 setup/convert, 29ae713 library, 997e86e translate, 3ddb17b tests,
3d28cae mix builder, 6060062 start_flan_nl.sh, 051814d protocol/commands, 4c683f0 three MaxEngine fixes, f0df603 own
continuous batching, 57841a4 position order past 2 GB, b167a5d templates top 30 + mechanism, 85252d8 / 0c407a8
phrasing table (31 keys, variants by Yeb drafted with Gemini), register, Head/Tail and lexical rules, 1b52081 protocol
for 27B. Tests: `tests/test_flan_nl.py`, 45 passed (local venv with pytest, numpy, pyarrow, langid; `--noconftest`,
since the conftest imports jax and flax); ruff clean. The publish step: `publish.py` and `tests/test_flan_nl_publish.py`
(39 tests with the checks of this publication; with `test_flan_nl.py` 84 passed, ruff clean). The mix builder was also dry-run on worker 0 with pilot rows
(passed `verify_order` and the byte check); `assemble` with templates was dry-run on a plan2 sample.
