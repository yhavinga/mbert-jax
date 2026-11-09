# Handover: the en-nl judge (Gemma 4 31B-it as a yes/no judge of translations)

Branch `en-nl-judge` of github.com/yhavinga/mbert-jax (from 7acefa2); everything in `experiments/en-nl-judge/` and
`tests/test_en_nl_judge.py`. Design context: `hammond/mbert/designs/en-nl-translator.md` §3 step 2, §4, §7 point 6.

## What it does

- `judge.py`: Gemma 4 31B-it (google/gemma-4-31B-it @ 842da37) converted to MaxText 0.2.5, on a v6e-8 (fsdp 8),
  train mode, left-padded batches, only the last position through the output head. Prompt: the model's chat template
  with one user turn (`QUESTION` in judge.py: English text, Dutch text, "Is the Dutch a correct and complete
  translation of the English? ... Answer with one word: yes or no."), generation prompt with the empty thought
  channel. Score = P(yes) / (P(yes) + P(no)) of the first token, summed over yes/Yes/YES/... and no/No/NO/...
  No threshold is chosen (Yeb, after calibration).
- `data.py motra`: Multilingual Parallel Direct Europarl (MoTra 2021), commit 4e284f1, paired as its
  `scripts/utils.py` does (sl="en" paragraphs, key date:intervention:index:count). 126,362 pairs: 1999-2003 38,617,
  2004-2011 87,745; nothing from mid-2011 on (both files hold the untranslated verbatim speeches there).
- `data.py sanity`: 500 paragraphs (4 strata of 125) + deterministic errors: mismatch 500, sentence_dropped 433,
  number_changed 166, negation_removed 189, english_kept 500.
- `data.py pilot`: the translator pilot's 2,000 pairs (`gs://t5-dutch-english-usc2/mbert/translator/pilot/`),
  English = synthetic English, Dutch = original; id = pid; chrF and MetricX-QE carried along.
- `hf_reference.py`: transformers on the CPU (float32) for 9 sanity rows, as a check of the conversion and of the
  prompt ids; `report.py compare` sets it beside the TPU's scores.
- `report.py`: sanity (AUC per error), pilot (30 pairs of pairs30.md, Spearman, 10 lowest), scan (europarl-scan.md).
- `setup.sh`, `prep.sh`, `run.sh`, `reports.sh`: the chain on the VM, each step skipped when its result is in the
  bucket. `keeper.sh` (co1): re-requests a preempted VM through `hammond/training/tpu.py up --replace` and starts
  run.sh on every new VM.

## Where things are (region of the VM; europe-west4 -> gs://t5-dutch-english, us-east1 -> gs://t5-dutch-english-use1)

`J=gs://t5-dutch-english/mbert/translator/judge`:
- `$J/scripts/` copies of the scripts; `$J/gemma-4-31b-it-maxtext/0/items` + `done.txt` (converted 9 Nov 12:01 UTC)
- `$J/data/{motra,sanity,pilot,pilot-source}/`, `$J/reference/ref.json`
- `$J/scores/{sanity,pilot,motra}/shard-NNNNN.parquet` (id, metadata, score, logp_yes, logp_no, answer_mass,
  top_token, top_logp, prompt_tokens, skipped, lengths), `$J/jax_cache/`, `$J/reports/`

## Commands

```
SSH='timeout 900 gcloud compute tpus tpu-vm ssh en-nl-judge --zone europe-west4-a --project t5-dutch-english --quiet --ssh-flag="-o ConnectTimeout=30 -o ServerAliveInterval=30 -o ServerAliveCountMax=4"'
tail -5 ~/logs/judge_keeper_euw4.log                      # co1: the keeper
$SSH --command 'grep -a "^==\|COMPILE\|SHARD\|DONE\|EXIT\|Traceback" ~/logs/judge_run.log | tail' < /dev/null
gsutil ls $J/scores/motra/ | wc -l                        # 13 shards when done
```

"Done": `~/logs/judge_run.log` ends in `EXIT 0`, `$J/reports/` has sanity.md, compare.md, pilot-scores.md and
europarl-scan.md; then copy pilot-scores.md and europarl-scan.md (small) to
`~/Developer/yhavinga/mbert-ops/translator-judge/` on co1, and a small copy of `$J/scores/pilot/` there.

## State (9 Nov, 12:46 UTC)

- VM `en-nl-judge`, v6e-8 spot, europe-west4-a (third VM of the day; us-east1-d VMs were preempted after 5-8 minutes,
  europe-west4-a once during the first compile). Keeper on co1: `NAME=en-nl-judge ZONE=europe-west4-a bash keeper.sh`,
  log `~/logs/judge_keeper_euw4.log`; it ends when run.sh ends.
- Conversion: works (MaxText's to_maxtext, 4.75 min on the host CPU, peak 101 GB RAM). Loading from the bucket 87 s;
  compiling 53-58 s per shape (fsdp 8, unscanned).
- Check against transformers (CPU, float32), 9 sanity rows: score difference at most 0.0002, log probabilities within
  0.37 (bf16), same top token (`$J/reports/compare.md`). Prompt ids equal to apply_chat_template.
- Sanity (`$J/reports/sanity.md`), AUC against all 500 originals / against the same paragraphs: english_kept
  1.000/1.000, mismatch 1.000/1.000, sentence_dropped 0.973/0.971, number_changed 0.932/0.948, negation_removed
  0.920/0.927. 119 of the 500 originals score below 0.1: the judge is strict (it rejects free but correct renderings,
  a speaker prefix such as "namens de ECR-Fractie . -", and also catches real errors). Scores are nearly binary (0 or
  1); logp_yes - logp_no in the score shards ranks more finely.
- Pilot (`$J/reports/pilot-scores.md`, copy on co1): Spearman with chrF 0.198, with MetricX-QE en-nl -0.486, nl-en
  -0.190 (n 2,000); 652 of 2,000 below 0.1.
- Tempo: MoTra 64 pairs/s (20k prompt tokens/s); sanity and pilot ~55-70 pairs/s net of compiling.
