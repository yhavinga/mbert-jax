# Pre-registered protocol: arm F-nl of the FLAN-decay proof, FLAN in Dutch

Status: written on 2026-11-08, before any run of arm F-nl (the translation and its pilot aside; no training of F-nl has
started). Changes after the run of F-nl has started are added below under "Amendments", dated, and never replace the
text above them.

Builds on `experiments/flan-decay/protocol.md` (arm F, branch `flan-decay`), whose models, evaluations and estimates
hold here unless this file says otherwise.
Code: branch `flan-nl` of mbert-jax at the commit `COMMIT_NL` of `commands.txt` (translation and mix), the trainer at
F's commit `bd6ddca` (`start_flan_nl.sh` checks it out, as `start_flan.sh` does).
Scripts: `gs://t5-dutch-english-usc2/mbert/flan-nl/scripts/` (= `experiments/flan-nl/` of `COMMIT_NL`).
Outputs: translation in `gs://t5-dutch-english-usc2/mbert/flan-nl/`, the run in
`gs://t5-dutch-english-usc2/mbert/flan-decay/flan-Fnl/`.

## Question

F (English FLAN with masked answers in the decay) helps zero-shot via the MLM head with Dutch prompts (DBRD 89.84
against 60.5 for W, Belebele 37.33 against 25.0) but not on SICK-NL (44.15, below the majority class 56.9), and not
fine-tuning. Does the same data in Dutch do better?

## Models

| | what | run | step directory |
|---|---|---|---|
| W | S + 20,000 steps of web text at 5% single tokens | `anneal-u15-uniform5` | `gs://t5-dutch-english-usc2/mbert/anneal/anneal-u15-uniform5/ckpt/20000` |
| F | W's schedule, web text at 5% and English FLAN at 10% of the tokens, answers masked | `flan-F` | `gs://t5-dutch-english-usc2/mbert/flan-decay/flan-F/ckpt/20000` |
| F-nl | F with the FLAN rows in Dutch | `flan-Fnl` | `gs://t5-dutch-english-usc2/mbert/flan-decay/flan-Fnl/ckpt/20000` |

F-nl differs from F only in the language of the FLAN rows: the same web rows in the same order, the same seeds, the
same trainer commit (`bd6ddca`), the same command (`start_flan_nl.sh` is `start_flan.sh` with only the mix path and
the run name changed), the same masking (`--masking decay --web-mask-rate 0.05 --answer-full-max 16`).

### Data of F-nl

Per host the mix of F (`~/flan_decay/mix-<build>.parquet`, 1.36 M rows of which ~340,000 FLAN rows) with every FLAN
row replaced in place by its Dutch version (`build_nl_mix.py`); a row without one is left out. The sequence the
trainer reads is F's with the Dutch rows at the places of the English ones; the file is written by F's builder
(`write_mix`, `trainer_shuffle_order`, `verify_order` of `scripts/build_decay_mix.py` at `bd6ddca`) and read back
through the trainer's route, and its web rows are checked to be F's byte for byte.

The translation (`flan_nl.py`, `translate.py`):

- Model: TranslateGemma 12B (`google/translategemma-12b-it` at revision `d1b225e1caa17f1ddc7e62065d8637d0923f34e2`),
  its language model converted to MaxText 0.2.5 (`convert.py`), run by MaxText's `MaxEngine` (prefill, insert,
  generate) with continuous batching of our own (`decode_all` in `translate.py`) on the 4 chips of each host, greedy
  decoding, a stop at `<end_of_turn>` or `<eos>`, four length classes of segments with their own cache shapes.
  27B (`google/translategemma-27b-it` at `7d10f0b72f89a2d0f268cea30727d8b77c0d25c2`, en-nl MetricX 1.67 against 2.01
  for 12B in the TranslateGemma report) was the first choice, to be used if its projected run on the 4 workers was at
  most 10 hours. Measured on a v4 host on 8 Nov: 27B 53 ms per decode step at 160 slots x 576 and 38 ms per prefill of
  256 tokens; 12B 38 ms at 256 x 512, 34 ms at 96 x 1,472, 18.5 ms per prefill of 256 tokens. Over the 1.81 M distinct
  segments (220 M source tokens, ~55 M per worker) that projects to ~18 hours for 27B and ~9 hours for 12B, so 12B.
  The 27B pilot of 300 examples (quality) and its measurements stay in `gs://t5-dutch-english-usc2/mbert/flan-nl/pilot/`
  and `out/27b/`; the 12B pilot is in `pilot/out/12b/` and `pilot/nl12b/`.
- Prompt: the model's chat template, a user turn `{type: text, source_lang_code: en, target_lang_code: nl, text}`.
- Segments: the instruction and the answer are translated, each distinct string once (yes/no and the like repeat).
  Multiple-choice options (labelled lists A/B/C, a/b/c, 1/2/3, i/ii/iii, `Option1:`, and bullet lists after a header
  such as `OPTIONS:`) are segments of their own; labels and separators stay as they are; an answer that is an option's
  text gets that option's translation; an answer that is a label stays the label. A segment of more than 512 source
  tokens (TranslateGemma's training data: "individual sentences and text blobs of up to 512 tokens") is cut at
  paragraphs, then lines, then sentences, the parts translated and joined; the plan counts the examples cut.
- Tasks left out (the lists `DROPPED_TASKS` in `flan_nl.py`): code, SQL and formal languages; English grammar,
  acceptability, spelling, punctuation and case (CoLA, BLiMP, JFLEG, `fix_punct`, `true_case`, `word_segment`, ...);
  word games, letters and strings (rhymes, character counts, lists of random letter strings, ...); POS tagging, NER
  and other token-level tasks (and next or missing English words, LAMBADA); translation and tokenization; tasks about
  English words (WiC, XL-WiC, the English-language Q&A, overlapping words); relations between bare words (odd man out,
  the BARD analogies, hypernyms, meronyms, antonyms, semantic relations of word pairs: a word translated out of
  context loses the sense the answer rests on). The assembly applies the lists again, so a task added after the plan
  is left out too.
- The six Dutch NIv2 tasks of F's pool (`task1133`, `task409`, `task1543`-`task1546`) have Dutch text with an English
  task definition; they stay as in F.
- Examples left out: an answer that is a whole-word stretch of the English instruction (case and spacing aside) while
  its translation is not one of the Dutch instruction (extractive QA; typographic quotes and dashes count as plain
  ones); a segment whose translation did not end (decode limit), is empty, has an implausible length (outside
  0.5-2.2 times the source's characters from 30 characters on, or more than 3 times plus 30), carries a translator's
  commentary ("Hier is de vertaling", "Vertaling:", "(Opmerking: ...", ...), offers alternatives (" / ", or ";" and
  "/" after a source of at most 4 words), has letters of a script the source does not have (Devanagari, CJK, Cyrillic,
  ...), or is not Dutch (`langid` 1.1.6 over nl, en, de, fr, es, it, pt, af says neither nl nor af, from 5 words on,
  and the output has English function words in it: names and short Dutch alone fooled `langid` in the pilot); a
  sentence of more than 512 tokens; a Dutch example of more than 1,024 tokens as the trainer counts them. One repair,
  counted: the Devanagari danda that TranslateGemma now and then writes for a period becomes the period (or nothing
  where the source ends without one). Every count is in the plan (`plan/plan.json`), the assembly (`nl/report.json`)
  and the mix manifests (`mix/w<N>.json`).
- No DBRD or SICK-NL data: the pool is F's, which excludes the SICK tasks and any task named after SICK, DBRD,
  Belebele or FLORES.

Not the same as F, as a consequence of the language and noted with the outcome, not corrected:

- The FLAN token share. The rows are F's, but Dutch text is shorter than English text in the Dutch tokenizer, so
  FLAN is less than F's 10.0% of the tokens; F-nl reads more web text per step and its 20,000 steps reach further into
  the web documents. The share per host is in the mix manifests.
- The dropped rows: F-nl has fewer FLAN examples, and none of the dropped tasks.
- The translation itself: TranslateGemma's errors, and Dutch answers that are translations rather than Dutch originals.

## Evaluations

The same as F's (`experiments/flan-decay/protocol.md`, "Evaluations"), with F-nl in F's place:

1. DUMB fine-tunes: the driver `dumb_v4_vm.sh` with `flan-Fnl=gs://t5-dutch-english-usc2/mbert/flan-decay/flan-Fnl/ckpt/20000`,
   DBRD and SICK-NL at `--max-length 512`, 3 epochs, LR 2e-5, seeds 42, 43, 44; results
   `gs://t5-dutch-english-usc2/mbert/masking/dumb/flan-Fnl--{dbrd,sicknl}--seed{42,43,44}.json`, metric `test_accuracy`.
2. Zero-shot via the MLM head: `experiments/flan-decay/eval_zeroshot.sh` with `Fnl=<step dir>`, the same tasks,
   templates and verbalizers; result `gs://t5-dutch-english-usc2/mbert/flan-decay/zeroshot/Fnl.json`.
3. Supporting evidence, deciding nothing: the `EVAL`, `DATA` and `MASKING` lines of F-nl next to F's.

The numbers of W and F this protocol compares with (DUMB: mean over seeds 42-44 [min-max]):

| | dbrd512 | sicknl512 | zero-shot DBRD | zero-shot SICK-NL | zero-shot Belebele |
|---|---|---|---|---|---|
| W | 95.69 [95.64-95.79] | 84.91 [84.17-85.63] | 60.52 | 47.22 | 25.00 |
| F | 95.55 [95.41-95.79] | 85.62 [85.16-85.92] | 89.84 | 44.15 | 37.33 |

The zero-shot numbers are the means over the 3 templates in `gs://t5-dutch-english-usc2/mbert/flan-decay/zeroshot/{W,F}.json`.

## Decision rule

Per cell (dbrd512, sicknl512): the mean of `test_accuracy` over seeds 42, 43, 44, and the seed range, max minus min.
"More than" is strict. Accuracies are k/n, so means, ranges and differences are compared exactly (as fractions of the
counts in `results.json`, not as rounded percentages). Checked in this order:

1. "Dutch harms" if F-nl's mean on dbrd512 or on sicknl512 is below F's by more than F's seed range in that cell.
2. "Dutch helps fine-tuning" if not 1, and F-nl's mean on dbrd512 or on sicknl512 is above F's by more than the larger
   of the two seed ranges (F's and F-nl's) in that cell.
3. "Dutch helps zero-shot" if not 1, and F-nl's zero-shot accuracy (mean over the 3 templates) is at least 5 points
   above F's on at least 2 of the 3 tasks (DBRD, SICK-NL, Belebele).
4. Otherwise "no measurable difference".

2 and 3 can both hold; both are then reported.

F-nl against W is reported with the rule of the FLAN protocol (`experiments/flan-decay/protocol.md`, "Decision rule",
with F-nl in F's place): "FLAN harms", "FLAN helps fine-tuning", "FLAN helps zero-shot", "no measurable difference".

Consequence: the outcome counts only for a possible separate instruct model; whether one is built, and with which
FLAN, Yeb decides. It does not change the FLAN protocol's decision about the final run (its amendment 1).

## Amendments
