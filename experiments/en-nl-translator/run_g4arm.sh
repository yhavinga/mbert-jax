#!/bin/bash
# Arm G4-1 of protocol-g4.md (Gemma 4 31B-it + LoRA rank 32 / alpha 16; Yeb, 14 Nov ~00:40) on a v4-32 (mbert-mix or
# mbert-corpus, us-central2-b), one stage per call on a worker; g4arm_run.sh (co1, mbert-ops) starts the stages and
# waits for their EXIT lines. Every stage that takes longer than a few seconds runs detached from ssh, with a fresh log
# ~/logs/g4a_<stage>_$TAG.log that ends in "EXIT <code>" (also copied to $G/logs/).
# The data (CPU, nice/ionice; before any TPU is free):
#   bash run_g4arm.sh fetchbt     w3: the five Gemma 4 BT datasets at their pinned revisions (g4arm.py fetch) from
#                                 Hugging Face, once, to $G/input/bt/ (a second start takes them from there)
#   bash run_g4arm.sh build       w3: g4arm.py build, to $G/data/ (set G4-1, report)
#   bash run_g4arm.sh check       w2: g4arm.py check (MaxText's own SFT steps on every example), $G/data/check.txt
#   bash run_g4arm.sh evaldata    w0: g4arm.py evaldata, to $G/data/eval/
# On the TPU:
#   bash run_g4arm.sh setup       every worker: the SFT venv, the fork's venv (@ f4d8615), Gemma 4's tokenizer, MetricX-24 XXL;
#                                 with FULL=1 also COMET-22 and (w0) the classifier's venv; READY
#   RUN=g4arm_G4-1_s8 STEPS=8 bash run_g4arm.sh train    every worker at once: run_sft.sh over the 16 chips (RUN
#                                 g4arm_G4-1 with STEPS=auto: one epoch), to $G/runs/$RUN/
#   bash run_g4arm.sh adapter     a worker: the run's adapter.npz/json and summary.json to $G/G4-1/ (rank, alpha checked)
#   MODEL=G4-1|G4-0 bash run_g4arm.sh translate    every worker: its classes of $G/data/eval/ (translate.py on the fork @ f4d8615,
#                                 natural-v1, G4-1 with the adapter merged), to $G/eval/$MODEL/translations/w<k>/<class>/
#   MODEL=G4-1|G4-0 bash run_g4arm.sh score        every worker: collect, MetricX-24 (reference and QE), COMET-22 on the
#                                 rows with a reference, to $G/eval/$MODEL/{pairs,scores}/
#   bash run_g4arm.sh texts       w0 (CPU): g4arm.py texts, to $G/naturalness/texts.parquet
#   bash run_g4arm.sh classify    w0 (its TPU): run_arms.sh classify with NAT=$G/naturalness (the classifier of 10 Nov)
#   bash run_g4arm.sh blind       w0 (CPU): armeval.py blind, G4-1 against G4-0 ($S/blind/G4-1/) and against TG
#                                 ($S/blind/G4-1-tg/)
#   bash run_g4arm.sh report      w0 (CPU): g4arm.py report, to $G/report/
# The smoke test (protocol-g4.md §6), after train with RUN=g4arm_G4-1_s8 STEPS=8:
#   MODEL=G4-1|G4-0 bash run_g4arm.sh smoke-translate   w0: the 50 smoke segments, class small, to $G/smoke/$TAG/$MODEL/
#   MODEL=G4-1|G4-0 bash run_g4arm.sh smoke-score       w0: collect and MetricX-24 of those 50
#   bash run_g4arm.sh smoke-report                      w0 (CPU): g4arm.py smoke, to $G/smoke/$TAG/
# Arm G4-2 (protocol-g4.md §9, amendment G4-2), with ARM=G4-2 (default ARM=G4-1, every stage as before):
#   bash run_g4arm.sh fetchval    w3: the five datasets' validation split at the pinned revisions, to $G/input/bt-val/
#   bash run_g4arm.sh heldout     w3: g4arm.py heldout (held-out books of G4-1's set, BT validation), to $G/data/heldout/
#   MODEL=G4-0|G4-2 SET=heldout bash run_g4arm.sh translate|score    the held-out segments (score: collect only), to
#                                 $G/heldout/$MODEL/
#   bash run_g4arm.sh own         w0 (CPU): g4arm.py own (G4-0's held-out outputs as an eval file), to $G/data/heldout/
#   ARM=G4-2 RUN=g4arm_G4-2_s8 STEPS=8 bash run_g4arm.sh train, then ARM=G4-2 bash run_g4arm.sh smoke2-report (w0):
#                                 the smoke test with the held-out evals on G4-1's set, verdict to $G/smoke2/$TAG/
#   bash run_g4arm.sh replay      w3: g4arm.py replay from the Gemma 4 FLAN-nl after its post step, to $G/data/replay/
#   bash run_g4arm.sh build2      w3: g4arm.py build2, to $G/data/G4-2/; ARM=G4-2 check: $G/data/check-G4-2.txt
#   ARM=G4-2 RUN=g4arm_G4-2 bash run_g4arm.sh train (LR 1e-4, warmup 3%, constant, linear decay over the last 10% to
#                                 1e-5; three passes; held-out evals every 150 steps), adapter, translate, score,
#                                 texts, classify, blind, report as for G4-1 with G4-2's places
# Arm G4-3 (protocol-g4.md §9, amendment G4-3), with ARM=G4-3:
#   bash run_g4arm.sh fetchbt|fetchval   w3: the five datasets (train, validation) at the revisions with the long rows
#                                 (g4arm.BT_REPOS3), to $G/input/bt3/ and $G/input/bt3-val/
#   bash run_g4arm.sh heldout3    w3: g4arm.py heldout3 (the long validation rows), to $G/data/heldout3/
#   bash run_g4arm.sh replay      w3: the replay pool of G4-3 (g4arm.REPLAY_POOL3), to $G/data/replay-G4-3/
#   bash run_g4arm.sh build3      w3: g4arm.py build3, to $G/data/G4-3/ and the smoke set $G/data/G4-3-smoke/
#   bash run_g4arm.sh check       w2: check --arm G4-3 in processes, $G/data/check-G4-3.txt
#   RUN=g4arm_G4-3_s8 STEPS=8 bash run_g4arm.sh train, then bash run_g4arm.sh smoke3-report (w0): the smoke test on
#                                 the smoke set at g4arm.SEQ3 with the four held-out evals, verdict and estimate to
#                                 $G/smoke3/$TAG/ (exit 1: a check failed); its json names the passes of the full run
#                                 (3, or 2 if three would take more than g4arm.MAX_HOURS3)
#   MODEL=G4-0|G4-3 SET=long bash run_g4arm.sh translate|score   the long validation rows (gemma4-31b-long; score:
#                                 collect only), to $G/long/$MODEL/
#   RUN=g4arm_G4-3 [PASSES=2] bash run_g4arm.sh train (LR 1e-5 WSD as G4-2, the evals every g4arm.EVAL_EVERY3
#                                 steps; PASSES=2: the set's first two passes), adapter,
#                                 translate, score (SET=eval|heldout|long), texts (G4-3, G4-0, G4-2), classify,
#                                 blind, report with G4-3's places
# Arm G4-4 (protocol-g4.md §9, amendment G4-4), with ARM=G4-4:
#   bash run_g4arm.sh build4      w1: g4arm.py build4 (G4-3's set per pass, the English novels of -b1- apart per pass,
#                                 the replay at 10% of every pass), to $G/data/G4-4/ and $G/data/G4-4-smoke/
#   bash run_g4arm.sh check       w2: check --arm G4-4 in CHECK_PROCESSES (8) processes, $G/data/check-G4-4.txt
#   RUN=g4arm_G4-4_s8 STEPS=8 bash run_g4arm.sh train, then bash run_g4arm.sh smoke4-report (w0): G4-3's smoke test on
#                                 G4-4's smoke set, verdict and estimate to $G/smoke4/$TAG/ (exit 1: a check failed or
#                                 the estimate is above g4arm.MAX_HOURS3)
#   RUN=g4arm_G4-4 bash run_g4arm.sh train (as G4-3, three passes), adapter, translate and score (SET=eval|heldout|long),
#                                 texts (G4-4, G4-0, G4-3), classify, blind, report with G4-3 beside G4-0 (reported)
#   Amendment of 16 Nov (G4-4.9-G4-4.11): the fifth eval set novels (unseen passages of seen novels), reported only:
#   bash run_g4arm.sh novels4     w1: g4arm.py novels4, to $G/data/heldout4/ (eval-novels.parquet, its meta, manifest)
#   RUN=g4arm_G4-3_novels STEPS=1 bash run_g4arm.sh train   G4-3's final adapter ($G/G4-3/) loaded before the first
#                                 step, one step on G4-4's smoke set, the novels evaluated at step 0 (G4-3's loss)
#                                 and after the step; G4-4's smoke test and training evaluate the novels too
# Arm G4-5 (draft amendment G4-5, Yeb 17 Nov: one LoRA per source, branched from a shared warmup with its optimizer
# state, combined by weights searched on the search part of the rules' sets), with ARM=G4-5:
#   bash run_g4arm.sh build5      w1: g4arm.py build5 (from G4-4's set: the warmup's set, one set per branch, flanval, the
#                                 smoke sets), to $G/data/G4-5/
#   bash run_g4arm.sh split5      w0: g4arm.py split5 (search and decision parts, the search part's reference files), to
#                                 $G/data/G4-5-split/
#   bash run_g4arm.sh refqe5      w0 (its TPU): MetricX-24-QE of the human references of WMT24++, FLORES, NTREX (the
#                                 human side of (a1) per part), to $G/data/G4-5-split/refqe.parquet
#   The smoke test: RUN=g4arm_G4-5_s8 train (the warmup's smoke set, 8 steps); RUN=g4arm_G4-5_b8 resume5 (w0: the s8
#     checkpoint into b8's checkpoint directory), RUN=g4arm_G4-5_b8 train (the branch's smoke set from step 8, 8 steps
#     with a cooldown); RUN=g4arm_G4-5_c8 train (the search on s8, b8 and G4-3's final adapter); MODEL=G4-5c1
#     SEARCH_RUN=g4arm_G4-5_c8 smoke-translate and smoke-score (w0: the stacked adapter decodes); smoke5-report (w0)
#   RUN=g4arm_G4-5_warmup train; for every branch B: RUN=g4arm_G4-5_B resume5, then RUN=g4arm_G4-5_B train
#   RUN=g4arm_G4-5_search train (STEPS=1: the search at step 0; SEARCH_BUDGET, SEARCH_TOP_K, SEARCH_STEP, ... override)
#   PHASE=prep, then PHASE=launch with train (every run of G4-5 and G4-6): the copies and run_sft.sh's arguments on
#     every worker first ("prepared <run> on wK", $G_SHM/data-<run>/launch.sh), then run_sft.sh on every worker at once
#   MODEL=G4-5c<k> translate and score (the k-th of the search's top, stacked on every worker from the runs' adapters);
#   CANDIDATES="G4-5c1 ..." [REFS=G4-4] texts, classify, report5 (w0); SELECTED=<report5's choice> blind
#   Addendum G4-5.7b (Yeb, 18 Nov): RUN=g4arm_G4-5_temp0 train and RUN=g4arm_G4-5_tempw train (eval only, the
#     temperature probe: G4-0 and the warmup's adapter, the loss at logits / T on every eval set, num_vocab_tiling=1);
#     the search's spec checks the merged warmup alone against the warmup's own eval at its last step first, and
#     takes its objective (T = 1 or calibrated) from the probe's rule ($REP/temperature.json); the search runs eval only
#     with --temperatures (both objectives per point);
#     report5 takes SMOKE5=<gs:// smoke5 json> (its anchor G4-3 alone, reported)
#   Addendum G4-5.7b B, the sampling probe (reported, decides nothing; its own places under $G/sampling/):
#   bash run_g4arm.sh sample-data          w0 (CPU): g4arm.py sampledata5, WMT24++ (960) and FLAN (1,000) of the
#                                          evaluation's segments per worker, to $G/sampling/data/
#   MODEL=G4-0|G4-4 SAMPLES="t07=0.7:1 t10s1=1.0:1 ..." bash run_g4arm.sh sample-translate|sample-score   every worker:
#                                          translate.py --sample (G4-4's adapter merged), collect, MetricX-24 (ref, QE),
#                                          COMET-22, to $G/sampling/<MODEL>/<sample>/
#   bash run_g4arm.sh sample-best8         w0 (CPU): best-of-8 of t10s1-t10s8 per model, $G/sampling/<MODEL>/best8/
#   SETTINGS="t07 t10s1 [tstar] best8" bash run_g4arm.sh sample-texts | sample-classify (TAG<TAG>s) | sample-report
#                                          w0: texts with the greedy G4-0 and G4-4 as anchors, the classifier,
#                                          g4arm.py sample5, to $G/sampling/naturalness/ and $G/sampling/report/
# Arm G4-6 (draft amendment G4-6: preference training, DPO with LoRA against G4-0; g4dpo.py), with ARM=G4-6:
#   bash run_g4arm.sh cands6      w1 (CPU): g4arm.py cands6 (the candidates from G4-4's set and the eval novels), the
#                                 rejected run's input to $G/data/G4-6/rejected-in/
#   MODEL=G4-0 SET=dpo bash run_g4arm.sh translate|score   every worker: G4-0 translates the candidates (the fork, as
#                                 the evaluation; score: collect only), to $G/dpo/G4-0/
#   bash run_g4arm.sh build6      w1 (CPU): g4arm.py build6 (the pairs, packed at 4,096 and 2,048, the smoke sets, the
#                                 preference eval sets), to $G/data/G4-6/
#   bash run_g4arm.sh check6      w2 (CPU): g4arm.py check6 (MaxText's own SFT steps and shift), $G/data/check-G4-6.txt
#   bash run_g4arm.sh cpuproof6   w1 (CPU): cpuproof6.sh (sft_lora.py --dpo with a tiny Gemma 4 on 4 simulated devices)
#   bash run_g4arm.sh pairtexts6, pairclassify6, pairclass6   w0: the classifier on chosen and rejected (reported)
#   RUN=g4arm_G4-6_s8 train (8 steps on the smoke set at 4,096; SEQ6=2048 and RUN=g4arm_G4-6_s8_2048: the fallback),
#     MODEL=G4-6|G4-0 smoke-translate and smoke-score (w0), smoke6-report (w0): the smoke test and the estimate
#   [SEQ6=2048] RUN=g4arm_G4-6 train (one epoch of the packed pairs), adapter, translate and score (SET=eval|heldout|
#     long), texts (G4-6, G4-0, G4-4, G4-5c2), classify, blind, report (G4-4 beside G4-0, the preference measures)
# Arm G4-6b (amendment G4-6b: G4-6 from G4-4, whose final adapter is merged into the base kernels as the start model and
# the reference; sft_lora.py --base-adapter), with ARM=G4-6b, the stages of G4-6 on G4-6's candidates:
#   MODEL=G4-4 SET=dpo translate|score (the rejected run, to $G/dpo/G4-4/); build6 (rejected G4-4, compared with G4-6's
#   pairs; to $G/data/G4-6b/); own (w0: G4-4's held-out translations of the BT validation as $G/data/G4-6b/eval-own4);
#   check6; cpuproof6 (also the merged adapter: BAKE=1); pairtexts6, pairclassify6, pairclass6; RUN=g4arm_G4-6b_s8
#   train (SEQ6=2048: g4arm_G4-6b_s8_2048), MODEL=G4-6b|G4-4 smoke-translate and smoke-score, smoke6-report;
#   RUN=g4arm_G4-6b train; adapter (G4-4's adapter and the run's stacked, to $G/G4-6b/; the run's alone in
#   $G/G4-6b/delta/); MODEL=G4-6b translate and score; texts (G4-6b, G4-0, G4-4, G4-5c2, and G4-6 once its report is
#   there; the list in $NAT/models.txt), classify, blind, report (G4-4 as the other arm, the rest beside)
# Arm G4-6c (amendment G4-6c: a milder DPO from G4-4 on G4-6b's pairs: beta 0.5, NLL weight 2.0, the turn's end outside
# the log-ratio, the adapter kept at steps 240 and 480), with ARM=G4-6c, G4-6b's pairs read only ($G/data/G4-6b/):
#   bash run_g4arm.sh ends6c      w2 (CPU): g4arm.py ends6c (every packed file: each segment ends in <turn|>\n), to
#                                 $G/data/ends-G4-6c.txt
#   cpuproof6 (BAKE=1, G4-6c's spec, the saved adapter and the end tokens, REFIN=1: the reference read in);
#   RUN=g4arm_G4-6c_s8_ref train, then RUN=g4arm_G4-6c_s8 train (SEQ6=2048: g4arm_G4-6c_s8_2048[_ref]; the adapter of
#   step 4 kept), MODEL=G4-6c|G4-4 smoke-translate and smoke-score, smoke6-report; RUN=g4arm_G4-6c_ref train, then
#   RUN=g4arm_G4-6c train (the adapters of steps 240 and 480 kept). Addendum G4-6c.6e: a run <run>_ref is the
#   reference of <run>'s training batches alone (sft_lora.py --ref-only), which <run> reads (--ref-in); [CKPT=<step>] adapter (G4-4's adapter and each
#   checkpoint's stacked, to $G/G4-6c-s240/, $G/G4-6c-s480/, $G/G4-6c-s717/, the checkpoint's alone in delta/); MODEL=G4-6c-s<k>
#   translate and score (SET=eval); texts (the three, G4-0, G4-4, G4-6b), classify, report6c (w0: the checkpoint
#   chosen on the search part of G4-5's split, the rule on its decision part); SELECTED=<report6c's choice> blind
# Arm G4-7 (amendment G4-7, protocol-g4.md §14: online RL, GRPO with LoRA from G4-0), with ARM=G4-7, what comes after
# the RL (g4arm7_run.sh and run_g4arm7.sh train; the chain puts the adapter after n iterations in $G/G4-7-i<n>/); only
# setup and these stages, SET=eval:
#   MODEL=G4-7-i<n> bash run_g4arm.sh translate|score   every worker: a candidate (n = 1..30), its LoRA merged (rank 32 /
#                                 alpha 16, checked; the job's own copy removed after), to $G/eval/G4-7-i<n>/
#   CANDIDATES="G4-7-i5 ..." bash run_g4arm.sh texts   w0 (CPU): the candidates, G4-0 and G4-4 (reported), one classify run
#   bash run_g4arm.sh classify     w0 (its TPU): C1, run_arms.sh classify with NAT=$G/naturalness-G4-7 (unchanged)
#   bash run_g4arm.sh classify-c2  w0 (its TPU): C2 ($RL/c2/model/) on the same texts, naturalness.py classify as C1 and
#                                 as the judge side runs it (--max-length 1024), to $NAT/predictions-c2.parquet
#   bash run_g4arm.sh pll          w0 (its TPU): naturalness.py pll (S @ 653k, as run_arms.sh pll) on the same texts, to
#                                 $NAT/pll.parquet (reported; without it report7 says "niet gemeten")
#   CANDIDATES="G4-7-i5 ..." bash run_g4arm.sh report7  w0 (CPU): g4arm.py report7 (the choice on the search part, (a1),
#                                 (a3), (b) on the decision part, C2 beside C1, the training per iteration from the
#                                 chain's $REP/records.json), to $G/report-G4-7/
#   SELECTED=<report7's choice> bash run_g4arm.sh blind   w0 (CPU): only if (a1), (a3) and (b) hold (G4-7.8)
# The measurement G4-bon (protocol-g4.md §15: best-of-8 with the choice on C2 + QE, the search part of G4-5.4 only;
# g4bon.py; mbert-ops/scripts/g4bon_run.sh on mbert-mix), with ARM=G4-bon, its places under $GB = $G/bon/:
#   bash run_g4arm.sh bon-data     w0 (CPU): g4bon.py bondata, the sampling probe's segments of the search part on their
#                                 workers, to $GB/data/
#   MODEL=G4-7-i<n> bash run_g4arm.sh bon-translate|bon-score   every worker: the candidate's 8 samples of the search
#                                 part as the probe made G4-0's (translate.py --sample t10s<k>=1:<k>, the candidate's
#                                 LoRA merged); collect, MetricX-24 (ref, QE), COMET-22, to $GB/<MODEL>/t10s<k>/
#   MODEL=G4-7-i<n> bash run_g4arm.sh bon-c2texts | bon-c2     w0: C2's texts (the samples, the candidate's and G4-0's
#                                 greedy translations; g4bon.py c2texts) and C2 on them, to $GB/<MODEL>/c2/
#   MODEL=G4-0|G4-4|G4-7-i<n> bash run_g4arm.sh bon-select   w0 (CPU): g4bon.py select (G4-0 and G4-4: the probe's
#                                 samples and C2 of c2cal7, G4-bon.10), to $GB/<MODEL>/select/
#   KEY=<key> MODELS="G4-0 [G4-4] [G4-7-i<n>]" bash run_g4arm.sh bon-texts | bon-classify | bon-report   w0: C1's texts of
#                                 every output, C1 (run_arms.sh classify, job arms_classify_<TAG>b), g4bon.py report, to
#                                 $GB/naturalness/<key>/ and $GB/report/<key>/
# The code is beside this script (a copy of $G/scripts/). TAG names the jobs of one chain.
set -euo pipefail
STAGE=${1:?a stage: fetchbt, build, check, evaldata, setup, train, adapter, translate, score, texts, classify, blind, report, smoke-translate, smoke-score, smoke-report, fetchval, heldout, own, smoke2-report, replay, build2, heldout3, build3, smoke3-report, build4, smoke4-report, novels4, build5, split5, refqe5, resume5, report5, smoke5-report, cands6, build6, check6, cpuproof6, smoke6-report, pairtexts6, pairclassify6, pairclass6, ends6c, report6c, classify-c2, pll, report7, bon-data, bon-translate, bon-score, bon-c2texts, bon-c2, bon-select, bon-texts, bon-classify, bon-report}
R=gs://t5-dutch-english-usc2/mbert/translator
S=$R/sft
E=$R/eval
G=$S/g4arm
ARM=${ARM:-G4-1}
case $ARM in G4-1|G4-2|G4-3|G4-4|G4-5|G4-6|G4-6b|G4-6c|G4-7|G4-bon) ;; *) echo "ARM is G4-1, G4-2, G4-3, G4-4, G4-5, G4-6, G4-6b, G4-6c, G4-7 or G4-bon, not $ARM"; exit 1 ;; esac
SMOKE_RUN=g4arm_${ARM}_s8
FULL_RUN=g4arm_$ARM
# G4-6 (draft amendment G4-6.4): rows of 4,096, or of 2,048 if the smoke test at 4,096 does not fit (its own smoke run)
SEQ6=${SEQ6:-4096}
case $SEQ6 in 4096|2048) ;; *) echo "SEQ6 is 4096 or 2048, not $SEQ6"; exit 1 ;; esac
A6=0
case $ARM in G4-6|G4-6b|G4-6c) A6=1 ;; esac
[ $A6 = 1 ] && [ "$SEQ6" = 2048 ] && SMOKE_RUN=g4arm_${ARM}_s8_2048
SET=${SET:-eval}
case $SET in eval|heldout|long|dpo) ;; *) echo "SET is eval, heldout, long or dpo, not $SET"; exit 1 ;; esac
[ "$SET" != dpo ] || { [ $A6 = 1 ] && [ "$ARM" != G4-6c ]; } || { echo "SET=dpo is G4-6's and G4-6b's (the rejected run)"; exit 1; }
# G4-2's places (amendment G4-2.9)
H=$G/data/heldout
NAT=$G/naturalness
REP=$G/report
CHECK_TXT=$G/data/check.txt
if [ "$ARM" = G4-2 ]; then NAT=$G/naturalness-G4-2; REP=$G/report-G4-2; CHECK_TXT=$G/data/check-G4-2.txt; fi
# G4-3's places (amendment G4-3.8): its inputs at the new revisions, the long validation rows, its set and replay
H3=$G/data/heldout3
BT_IN=$G/input/bt
BTVAL_IN=$G/input/bt-val
REPLAY_DIR=$G/data/replay
if [ "$ARM" = G4-3 ]; then
  NAT=$G/naturalness-G4-3; REP=$G/report-G4-3; CHECK_TXT=$G/data/check-G4-3.txt
  BT_IN=$G/input/bt3; BTVAL_IN=$G/input/bt3-val; REPLAY_DIR=$G/data/replay-G4-3
fi
# G4-4's places (amendment G4-4.8): its set on G4-3's (with G4-3's replay pool), the novels of the -b1- inventory;
# the novels' eval set (amendment G4-4.11)
B1=$S/data/books-b1
H4=$G/data/heldout4
NOVELS_RUN=g4arm_G4-3_novels
if [ "$ARM" = G4-4 ]; then
  NAT=$G/naturalness-G4-4; REP=$G/report-G4-4; CHECK_TXT=$G/data/check-G4-4.txt
  BT_IN=$G/input/bt3; BTVAL_IN=$G/input/bt3-val; REPLAY_DIR=$G/data/replay-G4-3
fi
# G4-5's places (draft amendment G4-5): its sets on G4-4's, the split of the rules' sets, its runs
G5=$G/data/G4-5
G5S=$G/data/G4-5-split
G5_WARMUP=g4arm_G4-5_warmup
G5_SEARCH=${SEARCH_RUN:-g4arm_G4-5_search}
G5_BRANCHES="bt_handelingen bt_raad bt_fineweb bt_rechtspraak bt_dpc books novels flan"
if [ "$ARM" = G4-5 ]; then
  NAT=$G/naturalness-G4-5; REP=$G/report-G4-5
fi
GS=$G/sampling                                                 # addendum G4-5.7b B: the sampling probe
# G4-6's places (draft amendment G4-6): the candidates and the pairs, G4-0's rejected run, the pairs' classifier run
G6=$G/data/G4-6
G6IN=$G6/rejected-in
G6R=$G/dpo
G6NAT=$G/naturalness-G4-6-pairs
REJ6=G4-0                                                      # whose translations are the rejected side
if [ "$ARM" = G4-6 ]; then
  NAT=$G/naturalness-G4-6; REP=$G/report-G4-6; CHECK_TXT=$G/data/check-G4-6.txt
fi
# G4-6b's places (amendment G4-6b): G4-6's candidates ($G6IN), G4-4's rejected run, its own pairs, runs and reports
if [ "$ARM" = G4-6b ]; then
  G6=$G/data/G4-6b; G6NAT=$G/naturalness-G4-6b-pairs; REJ6=G4-4
  NAT=$G/naturalness-G4-6b; REP=$G/report-G4-6b; CHECK_TXT=$G/data/check-G4-6b.txt
fi
# G4-6c's places (amendment G4-6c): G4-6b's pairs, their check and own4 (read only), its own runs, checkpoints, reports
if [ "$ARM" = G4-6c ]; then
  G6=$G/data/G4-6b; REJ6=G4-4; CHECK_TXT=$G/data/check-G4-6b.txt
  NAT=$G/naturalness-G4-6c; REP=$G/report-G4-6c
  # the stages that make or rewrite G4-6b's data are G4-6b's
  case $STAGE in cands6|build6|check6|own|pairtexts6|pairclassify6|pairclass6) echo "$STAGE writes G4-6b's data: not with ARM=G4-6c"; exit 1 ;; esac
  [ "$STAGE" != report ] || { echo "G4-6c's report is report6c (amendment G4-6c.7)"; exit 1; }
fi
[ "$ARM" = G4-6c ] || case $STAGE in ends6c|report6c) echo "$STAGE is G4-6c's"; exit 1 ;; esac
# G4-7's places (amendment G4-7.11): its RL directory (C2's model), its candidates' classify run and report. The RL
# itself (the samples, the judges, the training, the adapters $G/G4-7-i<n>/) is g4arm7_run.sh's and run_g4arm7.sh's
RL=$G/rl/G4-7
cand7() {  # cand7 NAME: a candidate of G4-7, the adapter after n = 1..30 iterations (g4rl.ITERATIONS7)
  case $1 in G4-7-i[1-9]|G4-7-i[12][0-9]|G4-7-i30) return 0 ;; *) return 1 ;; esac
}
if [ "$ARM" = G4-7 ]; then
  NAT=$G/naturalness-G4-7; REP=$G/report-G4-7
  case $STAGE in
    setup|translate|score|texts|classify|classify-c2|pll|blind|report7) ;;
    *) echo "$STAGE is no stage of G4-7 (setup, translate, score, texts, classify, classify-c2, pll, blind, report7)"; exit 1 ;;
  esac
  [ "$SET" = eval ] || { echo "ARM=G4-7: SET=eval only"; exit 1; }
fi
[ "$ARM" = G4-7 ] || case $STAGE in classify-c2|pll|report7) echo "$STAGE is G4-7's"; exit 1 ;; esac
# The measurement G4-bon (protocol-g4.md §15): its own stages and places; it trains nothing and reads G4-7's adapters
GB=$G/bon
if [ "$ARM" = G4-bon ]; then
  case $STAGE in
    setup|bon-*) ;;
    *) echo "$STAGE is no stage of G4-bon (setup, bon-data, bon-translate, bon-score, bon-c2texts, bon-c2, bon-select, bon-texts, bon-classify, bon-report)"; exit 1 ;;
  esac
fi
[ "$ARM" = G4-bon ] || case $STAGE in bon-*) echo "$STAGE is G4-bon's (ARM=G4-bon)"; exit 1 ;; esac
# The Gemma 4 FLAN-nl after its post step (flan_g4_full_run.sh), the replay's only source (amendment G4-2.3)
FNL=gs://t5-dutch-english-usc2/mbert/flan-nl
TRAIN_CKPT=$R/gemma4-ckpt/gemma-4-31b-it                       # run_sft.sh's gemma4-31b (MaxText 0.2.5 conversion)
DECODE_CKPT=$R/judge/gemma-4-31b-it-maxtext/0/items            # the same conversion's twin, as every Gemma 4 decode
G4_REPO=google/gemma-4-31B-it
G4_REV=842da3794eaa0b77d5f08bae87a17459d91ff475
SHM=/dev/shm/translator
G_SHM=$SHM/g4arm
SFT_PY=$SHM/smoke-lora/venv/bin/python                         # setup_sft.sh: MaxText 0.2.5, Tunix
GEN=/dev/shm/translate-generic
# protocol-g4.md §9, amendment of 14 Nov ~00:49: the fork with the fix of the shared decode-cache index (on 2249474),
# setup_generic.sh's venv-fork-<first 9 of the commit>
FORK_COMMIT=f4d8615f478b6c18de9df8f2dff1f57f775a2521
FPY=$GEN/venv-fork-${FORK_COMMIT:0:9}/bin/python
TOK_SFT=$SHM/g4lora/tok-gemma4-31b                             # run_sft.sh's tokenizer of gemma4-31b
TOK_DEC=$GEN/tokenizer-gemma4                                  # setup_generic.sh's, the same revision
MX_ROOT=$SHM/metricx
CODE=$(cd "$(dirname "$0")" && pwd)
W=$(hostname | sed 's/.*-w-//')
TAG=${TAG:-manual}
export PATH=$HOME/.local/bin:$PATH UV_CACHE_DIR=$SHM/uv-cache UV_PYTHON_INSTALL_DIR=$SHM/python
mkdir -p "$G_SHM" ~/logs

detach() {  # detach NAME COMMAND...: the command in the background, log ~/logs/NAME.log ending in EXIT <code>
  local name=$1
  shift
  [ -e ~/logs/$name.log ] && { echo "w$W: ~/logs/$name.log exists; a new start takes a new TAG"; exit 1; }
  printf '%q ' "$@" > ~/job_$name.sh
  setsid nohup bash -c "bash ~/job_$name.sh > ~/logs/$name.log 2>&1; echo EXIT \$? >> ~/logs/$name.log; gsutil -q cp ~/logs/$name.log $G/logs/$name-w$W.log" \
    > /dev/null 2>&1 < /dev/null &
  echo "started $name on w$W"
}

tpu_free() {
  [ -z "$(sudo lsof -t /dev/accel* /dev/vfio/* 2> /dev/null)" ] || { echo "w$W: the TPU is in use"; exit 1; }
}

only() {  # only WORKER: the stage runs on that worker alone
  [ "$W" = "$1" ] || { echo "$STAGE runs on w$1 only"; exit 1; }
}

# A worker of the v4-32 as a TPU of its own 4 chips (run_arms.sh, run_bt_length_v4.sh)
ONE_HOST="export TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 TPU_PROCESS_BOUNDS=1,1,1 TPU_VISIBLE_CHIPS=0,1,2,3 TPU_PROCESS_PORT=8476
export TPU_PROCESS_ADDRESSES=localhost:8476 CLOUD_TPU_TASK_ID=0 ALLOW_MULTIPLE_LIBTPU_LOAD=1
export XLA_FLAGS=--xla_dump_hlo_as_text=false TPU_LOG_DIR=$SHM/tpu_logs TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 HF_HUB_DISABLE_PROGRESS_BARS=1"
# G4-7's C2 and PLL: the DUMB rounds' venv and checkout as run_arms.sh's classify and pll (C1), with its host variables
DPY=$HOME/mbert-dumb/.venv/bin/python
DUMB=$HOME/mbert-dumbx
DUMB_COMMIT=ae114789bf2eae153e4e3bea579299a617dcd417
S653=gs://t5-dutch-english-usc2/mbert/culturax-nl-230m-chunk1024/ckpt_keep/653000
ONE_HOST_DUMB="export TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 TPU_PROCESS_BOUNDS=1,1,1 TPU_VISIBLE_CHIPS=0,1,2,3 TPU_PROCESS_PORT=8476
export TPU_PROCESS_ADDRESSES=localhost:8476 CLOUD_TPU_TASK_ID=0 ALLOW_MULTIPLE_LIBTPU_LOAD=1
export XLA_FLAGS=--xla_dump_hlo_as_text=false TPU_LOG_DIR=$SHM/tpu_logs TOKENIZERS_PARALLELISM=false"
dumb_ready() {
  [ -x "$DPY" ] || { echo "w$W: no $DPY"; exit 1; }
  [ "$(git -C "$DUMB" rev-parse HEAD 2> /dev/null)" = "$DUMB_COMMIT" ] || { echo "w$W: $DUMB is not at $DUMB_COMMIT"; exit 1; }
  [ -z "$(git -C "$DUMB" status --porcelain --untracked-files=no)" ] || { echo "w$W: $DUMB has changes"; exit 1; }
}
PREFLIGHT="timeout 600 $FPY -c \"import jax; assert (jax.device_count(), jax.process_count()) == (4, 1), (jax.device_count(), jax.process_count()); print('devices', jax.devices())\" || { echo 'preflight failed: not 4 chips of one process'; exit 2; }"

# translate_class SEGMENTS CLASS OUT LOG [EXTRA...]: one class through translate.py; out of memory: again with half
# the class's slots of translate_models.yml (sized on a v6e chip of 31.24 GiB; a v4 chip has 30.75), down to 4; every
# attempt its own log. G4-1 and G4-0 hold the same memory (the merged kernels replace the base ones), so both take the
# same slots.
TRANSLATE_CLASS='translate_class() {
  local seg=$1 cls=$2 out=$3 log=$4 code slots try
  shift 4
  slots=$($FPY -c "import sys; sys.path.insert(0, \"$CODE\"); import translate; print(translate.MODELS[\"$TMODEL\"][\"slots\"][\"$cls\"])")
  try=$log
  while true; do
    echo "== $cls slots $slots $(date -u +%T)"
    $FPY -u $CODE/translate.py --segments $seg --out $out --ckpt $DECODE_CKPT --tokenizer $TOK_DEC --model $TMODEL \
      --prompt natural-v1 --length-class $cls --source-lang en --target-lang nl --work-dir $G_SHM/work --slots $slots \
      "$@" > $try 2>&1 && code=0 || code=$?
    echo "EXIT $code" >> $try
    gsutil -q cp $try $G/logs/$(basename $try .log)-w$W.log || true
    grep -E "^(LORA|SHARD|DONE|engine)" $try | tail -n 4 | cut -c1-300
    [ $code = 0 ] && return 0
    if grep -q -E "RESOURCE_EXHAUSTED|Ran out of memory|[Oo]ut of memory" $try && [ $slots -gt 4 ]; then
      slots=$((slots / 2)); try=${log%.log}-s$slots.log
      echo "$cls: out of memory, again with $slots slots"
      continue
    fi
    tail -n 30 $try
    return $code
  done
}'
TMODEL=gemma4-31b
[ "$SET" = long ] && TMODEL=gemma4-31b-long                    # amendment G4-3.6: the long validation rows' classes
VARS="FPY=$FPY CODE=$CODE DECODE_CKPT=$DECODE_CKPT TOK_DEC=$TOK_DEC G_SHM=$G_SHM G=$G W=$W TMODEL=$TMODEL"
g4a_const() {  # g4a_const NAME: a constant of g4arm.py (SEQ3, LR3, EVAL_EVERY3), so that the stages and the code agree
  $SFT_PY -c "import sys; sys.path.insert(0, '$CODE'); import g4arm; print(g4arm.$1)"
}

adapter_local() {  # adapter_local RUN_DIR LOCAL: the adapter of a run (or $G/G4-1) copied for translate.py --lora
  mkdir -p "$2"
  gsutil -q cp "$1/adapter.npz" "$1/adapter.json" "$2/"
}

json_key() {  # json_key FILE KEY: a top-level value of a json file
  python3 -c "import json, sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])" "$1" "$2"
}

g45_component_dir() {  # g45_component_dir NAME: where a component of G4-5's search has its adapter
  case $1 in
    shared) if [ "$G5_SEARCH" = g4arm_G4-5_c8 ]; then echo "$G/runs/g4arm_G4-5_s8"; else echo "$G/runs/$G5_WARMUP"; fi ;;
    g43) echo "$G/G4-3" ;;
    b8) echo "$G/runs/g4arm_G4-5_b8" ;;
    *) echo "$G/runs/g4arm_G4-5_$1" ;;
  esac
}

g45_check_adapter() {  # g45_check_adapter adapter.json: rank 32, alpha 16 (every component of G4-5)
  python3 -c "import json, sys; a = json.load(open(sys.argv[1])); assert (a['rank'], a['alpha'], a['scale']) == (32, 16.0, 0.5), a" \
    "$1" || { echo "$1 is not rank 32, alpha 16"; exit 1; }
}

g45_stack() {  # g45_stack MODEL OUT: G4-5c<k>, the k-th of the search's top, stacked here from the components' adapters
  local k=${1#G4-5c} out=$2 sum=$G_SHM/g45-summary-$G5_SEARCH.json comps="" n d l
  gsutil -q cp "$G/runs/$G5_SEARCH/summary.json" "$sum"
  for n in $(python3 -c "import json, sys; print(' '.join(c['name'] for c in json.load(open(sys.argv[1]))['combine']['spec']['components']))" "$sum"); do
    d=$(g45_component_dir "$n")
    l=$G_SHM/g45-components-$G5_SEARCH/$n
    [ -s "$l/adapter.npz" ] || adapter_local "$d" "$l"
    g45_check_adapter "$l/adapter.json"
    comps="$comps --component $n=$l"
  done
  [ -s "$out/adapter.npz" ] || $FPY -I "$CODE/g4mix.py" stack $comps --from-summary "$sum" --candidate "$k" --out "$out"
}

g46b_stack() {  # g46b_stack RUN_DIR OUT: G4-4's adapter and a run's of G4-6b side by side, both with coefficient 1
  local run_dir=$1 out=$2 g44=$G_SHM/g46b-start-$TAG d=$G_SHM/g46b-delta-$TAG-$(basename "$1")
  [ -s "$g44/adapter.npz" ] || adapter_local "$G/G4-4" "$g44"
  [ -s "$d/adapter.npz" ] || adapter_local "$run_dir" "$d"
  g45_check_adapter "$g44/adapter.json"
  g45_check_adapter "$d/adapter.json"
  [ -s "$out/adapter.npz" ] || $FPY -I "$CODE/g4mix.py" stack --component shared="$g44" --component dpo="$d" \
    --weights 1 --shared-coef 2 --out "$out"
}

g46c_stack() {  # g46c_stack DELTA_LOCAL OUT: G4-4's adapter and a G4-6c checkpoint's (a local copy), both with coefficient 1
  local delta=$1 out=$2 g44=$G_SHM/g46b-start-$TAG
  [ -s "$g44/adapter.npz" ] || adapter_local "$G/G4-4" "$g44"
  g45_check_adapter "$g44/adapter.json"
  g45_check_adapter "$delta/adapter.json"
  [ -s "$out/adapter.npz" ] || $FPY -I "$CODE/g4mix.py" stack --component shared="$g44" --component dpo="$delta" \
    --weights 1 --shared-coef 2 --out "$out"
}

g45_train() {  # the training runs of G4-5 (draft amendment G4-5): the warmup, the branches, the search, the smoke runs
  RUN=${RUN:?g4arm_G4-5_warmup, g4arm_G4-5_<branch>, $G5_SEARCH or a smoke run (g4arm_G4-5_s8, _b8, _c8)}
  tpu_free
  local setdir steps every lrw decay from="" resume=0 combine="" temps="" init="" b
  every=$(g4a_const EVAL_EVERY5)
  lrw=$(g4a_const LR_WARMUP_STEPS5)
  decay=$(g4a_const DECAY_FRACTION5)
  D=$G_SHM/data-$RUN
  mkdir -p "$D"
  # a launch never takes the arguments of an earlier or a broken preparation
  rm -f -- "$D/launch.sh"
  case $RUN in
    g4arm_G4-5_s8) setdir=$G5/G4-5-smoke-warmup; steps=8; every=4; lrw=$(g4a_const SMOKE5_LR_WARMUP); decay=0 ;;
    g4arm_G4-5_b8) setdir=$G5/G4-5-smoke-branch; steps=8; every=4; lrw=$(g4a_const SMOKE5_LR_WARMUP)
                   decay=$(g4a_const SMOKE5_DECAY); from=g4arm_G4-5_s8 ;;
    g4arm_G4-5_c8|"$G5_SEARCH") setdir=$G5/G4-5-smoke-warmup; steps=1; every=1; lrw=0; decay=0; combine=1
                                # addendum G4-5.7b A: both objectives per point, eval only (the search, no step)
                                if [ "$RUN" = "$G5_SEARCH" ]; then temps=1; fi ;;
    # addendum G4-5.7b: the temperature probe, eval only (no step runs; the set is only the stack's training input)
    g4arm_G4-5_temp0) setdir=$G5/G4-5-smoke-warmup; steps=1; every=1; lrw=0; decay=0; temps=1 ;;
    g4arm_G4-5_tempw) setdir=$G5/G4-5-smoke-warmup; steps=1; every=1; lrw=0; decay=0; temps=1; init=$G5_WARMUP ;;
    "$G5_WARMUP") setdir=$G5/G4-5-warmup; decay=0 ;;
    g4arm_G4-5_*) b=${RUN#g4arm_G4-5_}
                  case " $G5_BRANCHES " in *" $b "*) ;; *) echo "RUN $RUN: $b is no branch of G4-5 ($G5_BRANCHES)"; exit 1 ;; esac
                  setdir=$G5/G4-5-$b; from=$G5_WARMUP ;;
    *) echo "RUN $RUN is no run of G4-5"; exit 1 ;;
  esac
  gsutil -q cp "$setdir/manifest.json" "$D/manifest.json"
  [ -n "${steps:-}" ] || steps=$(json_key "$D/manifest.json" steps)
  [ "${STEPS:-$steps}" = "$steps" ] || { echo "$RUN takes $steps steps (its set), not STEPS=$STEPS"; exit 1; }
  local EVAL_FILES="$H/eval-books.parquet $H/eval-btval.parquet $H/eval-own.parquet $H3/eval-btlong.parquet $H4/eval-novels.parquet $G5/eval-flanval.parquet $G5S/eval-wmt24pp-s.parquet $G5S/eval-flores-s.parquet $G5S/eval-ntrex-s.parquet"
  for f in $EVAL_FILES; do gsutil -q stat "$f" || { echo "no $f (heldout, heldout3, novels4, build5, split5)"; exit 1; }; done
  gsutil -m -q cp "$setdir/train.parquet" "$setdir/train-meta.parquet" "$setdir/template.jinja" $EVAL_FILES "$D/"
  local ARGS="--eval books=$D/eval-books.parquet --eval btval=$D/eval-btval.parquet --eval own=$D/eval-own.parquet"
  ARGS="$ARGS --eval btlong=$D/eval-btlong.parquet --eval novels=$D/eval-novels.parquet --eval flanval=$D/eval-flanval.parquet"
  ARGS="$ARGS --eval wmt24pp_s=$D/eval-wmt24pp-s.parquet --eval flores_s=$D/eval-flores-s.parquet --eval ntrex_s=$D/eval-ntrex-s.parquet"
  ARGS="$ARGS --eval-every $every --warmup-steps $lrw --decay-fraction $decay"
  if [ -n "$from" ]; then
    # the restored state is the last step of the run it comes from; resume5 copied its checkpoint into this run's place
    gsutil -q cp "$G/runs/$from/summary.json" "$D/from-summary.json"
    gsutil -q cp "$G/runs/$from/adapter.npz" "$D/resume-adapter.npz"
    gsutil -q cp "$G/runs/$from/adapter.json" "$D/resume-adapter.json"
    g45_check_adapter "$D/resume-adapter.json"
    resume=$(json_key "$D/from-summary.json" steps)
    gsutil -q stat "$G/runs/$RUN/maxtext/$RUN/checkpoints/$resume/commit_success.txt" || \
      { echo "no checkpoint of step $resume in $G/runs/$RUN/maxtext/$RUN/checkpoints/ (resume5 first)"; exit 1; }
    ARGS="$ARGS --resume-step $resume --resume-adapter $D/resume-adapter.npz"
  fi
  if [ -n "$init" ]; then
    # the adapter of the run's last step as the LoRA parameters before the eval-only round
    gsutil -q cp "$G/runs/$init/adapter.npz" "$D/init-adapter.npz"
    gsutil -q cp "$G/runs/$init/adapter.json" "$D/init-adapter.json"
    g45_check_adapter "$D/init-adapter.json"
    ARGS="$ARGS --init-adapter $D/init-adapter.npz"
  fi
  [ -z "$temps" ] || ARGS="$ARGS --temperatures $(g4a_const TEMPERATURES5)"
  if [ -n "$combine" ]; then
    local comps="" n names objective check=""
    if [ "$RUN" = g4arm_G4-5_c8 ]; then
      names="shared b8 g43"
      objective="--objective books=1 --objective novels=1"
      SEARCH_BUDGET=${SEARCH_BUDGET:-8}; SEARCH_TOP_K=${SEARCH_TOP_K:-2}; SEARCH_STEP=${SEARCH_STEP:-0.5}
      SEARCH_MIN_STEP=${SEARCH_MIN_STEP:-0.5}
    else
      names="shared $G5_BRANCHES"
      objective=$($SFT_PY -c "import sys; sys.path.insert(0, '$CODE'); import g4arm; print(' '.join(f'--objective {k}={v}' for k, v in g4arm.OBJECTIVE5.items()))")
      # addendum G4-5.7b: the merged warmup alone against the warmup's own eval at its last step, before the search
      gsutil -q cp "$G/runs/$G5_WARMUP/summary.json" "$D/warmup-summary.json"
      check=$($SFT_PY -c "import sys; sys.path.insert(0, '$CODE'); import g4arm; print(' '.join(f'--check-set {s}' for s in g4arm.WARMUP_CHECK_SETS5), '--check-max-diff', g4arm.MERGE_MAX_DIFF5)")
      check="--warmup-check $D/warmup-summary.json $check"
      # addendum G4-5.7b A: the rule fixed before the search (temp5's search_rule) chooses its objective
      gsutil -q cp "$REP/temperature.json" "$D/temperature.json" || { echo "no $REP/temperature.json (the temperature probe first)"; exit 1; }
      check="$check --objective-rule $D/temperature.json"
    fi
    for n in $names; do
      mkdir -p "$D/components/$n"
      gsutil -q cp "$(g45_component_dir "$n")/adapter.npz" "$(g45_component_dir "$n")/adapter.json" "$D/components/$n/"
      g45_check_adapter "$D/components/$n/adapter.json"
      comps="$comps --component $n=$D/components/$n/adapter.npz"
    done
    $SFT_PY "$CODE/g4mix.py" spec $comps $objective --budget "${SEARCH_BUDGET:-$(g4a_const SEARCH_BUDGET5)}" \
      --top-k "${SEARCH_TOP_K:-$(g4a_const SEARCH_TOP_K5)}" --step "${SEARCH_STEP:-0.25}" --min-step "${SEARCH_MIN_STEP:-0.0625}" \
      --lower "${SEARCH_LOWER:-0.0}" --upper "${SEARCH_UPPER:-1.5}" --start "${SEARCH_START:-uniform}" \
      --shared-coef "${SEARCH_SHARED:-1.0}" --scale 0.5 $check --out "$D/spec.json" > /dev/null
    ARGS="$ARGS --combine $D/spec.json"
  fi
  # PHASE=prep stops here and PHASE=launch starts run_sft.sh from this file: the processes of the multi-host job meet in
  # jax.distributed.initialize within a timeout, and the search's copies (~8 GB of components) take ~2 min per worker,
  # so a chain prepares every worker before it launches any (19 Nov: four sequential starts 6 min apart failed)
  printf '%q ' env SFT_ARGS="$ARGS" MULTIHOST=1 NAME="$RUN" MODEL=gemma4-31b DATA="$D/train.parquet" \
    TEMPLATE="$D/template.jinja" ROOT="$G" STEPS="$steps" bash "$CODE/run_sft.sh" per_device_batch_size=1 \
    max_target_length="$(g4a_const SEQ3)" lora.lora_rank=32 lora.lora_alpha=16.0 learning_rate="$(g4a_const LR5)" \
    lr_schedule_type=wsd wsd_decay_style=linear learning_rate_final_fraction=0.1 ${temps:+num_vocab_tiling=1} \
    > "$D/launch.sh.new"
  mv "$D/launch.sh.new" "$D/launch.sh"
  if [ "${PHASE:-}" = prep ]; then
    echo "prepared $RUN on w$W"
    return 0
  fi
  bash "$D/launch.sh"
}

g46_train() {  # the runs of G4-6, G4-6b and G4-6c (amendments G4-6, G4-6b, G4-6c): the smoke runs at 4,096 and 2,048, the full run
  RUN=${RUN:?g4arm_${ARM}_s8, g4arm_${ARM}_s8_2048 or g4arm_$ARM}
  tpu_free
  local seq setfile steps every n f spec=DPO6 save="" ref_only="" train_run=$RUN
  # addendum G4-6c.6e: G4-6c's reference of a run's training batches is a run of its own, <run>_ref (sft_lora.py
  # --ref-only, the same model and config), which the run then reads (--ref-in)
  if [ "$ARM" = G4-6c ] && [[ $RUN == *_ref ]]; then ref_only=1; train_run=${RUN%_ref}; fi
  case $train_run in
    "g4arm_${ARM}_s8") seq=4096; setfile=smoke-4096.parquet; steps=$(g4a_const SMOKE6_STEPS); every=4 ;;
    "g4arm_${ARM}_s8_2048") seq=2048; setfile=smoke-2048.parquet; steps=$(g4a_const SMOKE6_STEPS); every=4 ;;
    "g4arm_$ARM") seq=$SEQ6; setfile=train-$SEQ6.parquet; steps=auto; every=$(g4a_const EVAL_EVERY6) ;;
    *) echo "RUN $RUN is no run of $ARM"; exit 1 ;;
  esac
  if [ "$ARM" = G4-6c ]; then
    # amendment G4-6c: its loss, evals at the checkpoints, the adapters of the checkpoints kept (the smoke runs: step 4)
    spec=DPO6C
    if [ "$train_run" = "g4arm_$ARM" ]; then every=$(g4a_const EVAL_EVERY6C); save=$(g4a_const SAVE_ARG6C); else save=$(g4a_const SMOKE_SAVE6C); fi
    [ -z "$ref_only" ] || save=""
  fi
  D=$G_SHM/data-$RUN
  # a launch never takes the arguments of an earlier or a refused preparation
  rm -f -- "$D/launch.sh"
  [ "${STEPS:-$steps}" = "$steps" ] || { echo "$RUN takes $steps steps, not STEPS=$STEPS"; exit 1; }
  gsutil -q stat "$G6/report.json" || { echo "no $G6/report.json (build6)"; exit 1; }
  gsutil -q stat "$CHECK_TXT" || { echo "no $CHECK_TXT: the pairs are not checked (check6)"; exit 1; }
  mkdir -p "$D"
  local sft="$H/eval-books.parquet $H/eval-btval.parquet $H/eval-own.parquet $H4/eval-novels.parquet"
  # amendment G4-6.4: the long validation rows only in rows of 4,096 (MaxText would cut them at 2,048)
  [ "$seq" = 4096 ] && sft="$sft $H3/eval-btlong.parquet"
  # amendment G4-6b: also G4-4's own greedy held-out translations of the BT validation (own4); G4-6c: G4-6b's
  case $ARM in G4-6b|G4-6c) sft="$sft $G6/eval-own4.parquet" ;; esac
  local prefs=""
  for n in pref_books pref_btval pref_novels; do prefs="$prefs $G6/eval-$n-$seq.parquet"; done
  for f in $sft $prefs; do gsutil -q stat "$f" || { echo "no $f"; exit 1; }; done
  # a reference run reads only the training file (no evals)
  [ -n "$ref_only" ] && sft="" prefs=""
  gsutil -m -q cp "$G6/$setfile" "$G6/train-chosen.parquet" "$G6/template.jinja" $sft $prefs "$D/"
  $SFT_PY -c "import json, sys; sys.path.insert(0, '$CODE'); import g4arm, g4dpo; print(json.dumps(g4dpo.spec(**g4arm.$spec)))" > "$D/dpo.json"
  # MaxText's data hooks are made from DATA (the chosen sides as SFT examples) and never read: the batches are the
  # packed pairs of --pref-train
  local ARGS="--dpo $D/dpo.json --pref-train $D/$setfile"
  [ -n "$save" ] && ARGS="$ARGS --save-steps $save"
  if [ "$ARM" = G4-6b ] || [ "$ARM" = G4-6c ]; then
    # amendment G4-6b.2: G4-4's final adapter merged into the base kernels before the first step (start and reference);
    # a reference run of G4-6c takes the copy of its run's data (one copy of ~1 GB on w0's /dev/shm per run)
    local start=$G_SHM/data-$train_run/start
    mkdir -p "$start"
    gsutil -q cp "$G/G4-4/adapter.npz" "$G/G4-4/adapter.json" "$start/"
    g45_check_adapter "$start/adapter.json"
    ARGS="$ARGS --base-adapter G4-4=$start/adapter.npz"
  fi
  local evals="--eval-every $every"
  if [ -n "$ref_only" ]; then
    # addendum G4-6c.6e: the reference alone (after G4-4's merge): no eval, no kept adapter
    ARGS="$ARGS --ref-only"
    evals=""
  else
    for n in pref_books pref_btval pref_novels; do ARGS="$ARGS --pref-eval $n=$D/eval-$n-$seq.parquet"; done
    for f in $sft; do n=$(basename "$f" .parquet); ARGS="$ARGS --eval ${n#eval-}=$D/$(basename "$f")"; done
  fi
  if [ "$ARM" = G4-6c ] && [ -z "$ref_only" ]; then
    # addendum G4-6c.6e: the training batches' reference from the run <run>_ref (sft_lora.py checks it is this run's)
    for f in ref-train.npy ref-train.json; do
      gsutil -q stat "$G/runs/${RUN}_ref/$f" || { echo "no $G/runs/${RUN}_ref/$f: the reference run ${RUN}_ref first (addendum G4-6c.6e)"; exit 1; }
    done
    mkdir -p "$D/ref"
    gsutil -m -q cp "$G/runs/${RUN}_ref/ref-train.npy" "$G/runs/${RUN}_ref/ref-train.json" "$D/ref/"
    ARGS="$ARGS --ref-in $D/ref/ref-train.npy"
  fi
  # PHASE=prep stops here and PHASE=launch starts run_sft.sh from this file, as G4-5's runs: the processes of the
  # multi-host job meet in jax.distributed.initialize within a timeout, so a chain prepares every worker first
  printf '%q ' env SFT_ARGS="$ARGS${evals:+ $evals}" MULTIHOST=1 NAME="$RUN" MODEL=gemma4-31b \
    DATA="$D/train-chosen.parquet" TEMPLATE="$D/template.jinja" ROOT="$G" STEPS="$steps" bash "$CODE/run_sft.sh" \
    per_device_batch_size=1 max_target_length="$seq" lora.lora_rank=32 lora.lora_alpha=16.0 \
    learning_rate="$(g4a_const LR6)" lr_schedule_type=wsd warmup_steps_fraction=0.03 wsd_decay_steps_fraction=0.1 \
    wsd_decay_style=linear learning_rate_final_fraction=0.1 > "$D/launch.sh.new"
  mv "$D/launch.sh.new" "$D/launch.sh"
  if [ "${PHASE:-}" = prep ]; then
    echo "prepared $RUN on w$W"
    return 0
  fi
  bash "$D/launch.sh"
}

case $STAGE in
fetchbt)
  only 3
  B=$G_SHM/bt-$TAG
  cat > ~/job_body_g4a_fetchbt_$TAG.sh <<EOF
set -euo pipefail
export HF_HUB_DISABLE_PROGRESS_BARS=1 HF_XET_CACHE=$SHM/xet-cache
if gsutil -q stat $BT_IN/revisions.json; then
  echo "the BT input is in $BT_IN/ before"
else
  [ -s ~/.cache/huggingface/token ] || { echo "w$W: no ~/.cache/huggingface/token"; exit 1; }
  $SFT_PY -u $CODE/g4arm.py fetch --arm $ARM --out $B
  rm -rf $B/*/.cache
  gsutil -m -q rsync -r $B $BT_IN
fi
echo "FETCHED: \$(gsutil cat $BT_IN/revisions.json | tr -d '\n ' | cut -c1-400)"
EOF
  detach "g4a_fetchbt_$TAG" nice -n 19 ionice -c3 bash ~/job_body_g4a_fetchbt_$TAG.sh
  ;;
build)
  only 3
  [ "$ARM" = G4-1 ] || { echo "build makes G4-1's set; G4-2's is build2"; exit 1; }
  ! gsutil -q stat "$G/data/$ARM/manifest.json" || { echo "$G/data/$ARM has a manifest.json: the set stays as it is"; exit 1; }
  [ -s "$TOK_SFT/tokenizer.json" ] || { echo "w$W: no tokenizer in $TOK_SFT (setup)"; exit 1; }
  IN=$G_SHM/in-$TAG
  OUT=$G_SHM/data-$TAG
  # BUILD_EXTRA: the build's options of an amendment (e.g. --fineweb-max-crawl 2023, --replay-share 0.1 --replay ...)
  cat > ~/job_body_g4a_build_$TAG.sh <<EOF
set -euo pipefail
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=true HF_HUB_OFFLINE=1 JAX_PLATFORMS=cpu
mkdir -p $IN/bt $IN/rl $IN/r2
gsutil -m -q rsync -r $G/input/bt $IN/bt
gsutil -m -q cp $S/data/rl-b5/RL/train.parquet $S/data/rl-b5/RL/train-meta.parquet $IN/rl/
gsutil -m -q cp $S/data/r2/R2/train.parquet $S/data/r2/R2/train-meta.parquet $IN/r2/
gsutil -q cp $E/data/evalsets.parquet $S/blind/fixed-2026-11-10/blind100.jsonl $R/mainset/flan-eval/eval-rows.parquet $IN/
$SFT_PY -u $CODE/g4arm.py build --bt $IN/bt --books $IN/rl --match-set $IN/r2 --evalsets $IN/evalsets.parquet \
  --blind $IN/blind100.jsonl --flan-eval $IN/eval-rows.parquet --tokenizer $TOK_SFT --out $OUT ${BUILD_EXTRA:-}
gsutil -m -q cp $OUT/$ARM/* $G/data/$ARM/
gsutil -m -q cp $OUT/report.json $OUT/report.md $G/data/
echo "BUILT: $G/data/$ARM/"
EOF
  detach "g4a_build_$TAG" nice -n 19 ionice -c3 bash ~/job_body_g4a_build_$TAG.sh
  ;;
check)
  only 2
  [ -s "$TOK_SFT/tokenizer.json" ] || { echo "w$W: no tokenizer in $TOK_SFT (setup)"; exit 1; }
  D=$G_SHM/check-$TAG
  CHECK_ARGS=""
  [ "$ARM" = G4-3 ] && CHECK_ARGS="--arm G4-3 --processes 16"
  [ "$ARM" = G4-4 ] && CHECK_ARGS="--arm G4-4 --processes ${CHECK_PROCESSES:-8}"
  cat > ~/job_body_g4a_check_$TAG.sh <<EOF
set -euo pipefail
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=true HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 JAX_PLATFORMS=cpu
mkdir -p $D
gsutil -m -q cp $G/data/$ARM/train.parquet $G/data/$ARM/train-meta.parquet $G/data/$ARM/template.jinja $D/
$SFT_PY -u $CODE/g4arm.py check --data $D --tokenizer $TOK_SFT $CHECK_ARGS | tee $D/check.txt
gsutil -q cp $D/check.txt $CHECK_TXT
EOF
  detach "g4a_check_$TAG" nice -n 19 ionice -c3 bash ~/job_body_g4a_check_$TAG.sh
  ;;
evaldata)
  only 0
  ! gsutil -q stat "$G/data/eval/manifest.json" || { echo "$G/data/eval has a manifest.json: it stays as it is"; exit 1; }
  [ -s "$TOK_SFT/tokenizer.json" ] || { echo "w$W: no tokenizer in $TOK_SFT (setup)"; exit 1; }
  IN=$G_SHM/evalin-$TAG
  cat > ~/job_body_g4a_evaldata_$TAG.sh <<EOF
set -euo pipefail
export TOKENIZERS_PARALLELISM=true HF_HUB_OFFLINE=1 JAX_PLATFORMS=cpu
mkdir -p $IN
gsutil -q cp $E/data/evalsets.parquet $IN/four.parquet
gsutil -q cp $S/arms/data/x/data/evalsets.parquet $IN/x.parquet
$SFT_PY -u $CODE/g4arm.py evaldata --four $IN/four.parquet --x $IN/x.parquet --tokenizer $TOK_SFT --out $IN/eval
gsutil -m -q cp $IN/eval/* $G/data/eval/
echo "EVALDATA_DONE $G/data/eval/"
EOF
  detach "g4a_evaldata_$TAG" nice -n 19 ionice -c3 bash ~/job_body_g4a_evaldata_$TAG.sh
  ;;
setup)
  # The data stages need only the SFT venv and the tokenizer (SETUP=data); the TPU stages everything
  [ "$(loginctl show-user "$(id -un)" -p Linger --value 2> /dev/null)" = yes ] || sudo loginctl enable-linger "$(id -un)"
  [ -s ~/.cache/huggingface/token ] || { echo "w$W: no ~/.cache/huggingface/token"; exit 1; }
  free=$(df -BG --output=avail "$HOME" | tail -1 | tr -dc 0-9)
  [ "$free" -ge 3 ] || { echo "w$W: only $free GB free on the boot disk"; exit 1; }
  cat > ~/job_body_g4a_setup_$TAG.sh <<EOF
set -euo pipefail
export PATH=$PATH UV_CACHE_DIR=$UV_CACHE_DIR UV_PYTHON_INSTALL_DIR=$UV_PYTHON_INSTALL_DIR HF_HUB_DISABLE_PROGRESS_BARS=1
[ -x $SFT_PY ] || bash $CODE/setup_sft.sh
$SFT_PY -c "import importlib.metadata as m; v = m.version('maxtext'); assert v == '0.2.5', v"
[ -s $TOK_SFT/tokenizer.json ] && [ -s $TOK_SFT/chat_template.jinja ] || $SFT_PY -c "from huggingface_hub import snapshot_download as s; s('$G4_REPO', revision='$G4_REV', local_dir='$TOK_SFT', allow_patterns=['tokenizer*', 'chat_template.jinja', 'config.json', 'generation_config.json', 'special_tokens_map.json'])"
[ -s $TOK_SFT/tokenizer.json ] || { echo "no tokenizer in $TOK_SFT"; exit 1; }
if [ "${SETUP:-tpu}" = data ]; then
  echo "READY w$W (data): SFT venv, tokenizer"
  exit 0
fi
gsutil -q stat $TRAIN_CKPT/done.txt || { echo "$TRAIN_CKPT has no done.txt"; exit 1; }
gsutil -q stat ${DECODE_CKPT%/0/items}/0/commit_success.txt || { echo "no ${DECODE_CKPT%/0/items}/0/commit_success.txt"; exit 1; }
FORK_COMMIT=$FORK_COMMIT bash $CODE/setup_generic.sh venv
[ -s $TOK_DEC/tokenizer.json ] || { echo "no $TOK_DEC/tokenizer.json"; exit 1; }
cmp -s $TOK_DEC/tokenizer.json $TOK_SFT/tokenizer.json || { echo "the two Gemma 4 tokenizers differ"; exit 1; }
if [ ! -d $MX_ROOT/metricx-24-hybrid-xxl-v2p6-bfloat16 ]; then
  HF_HUB_OFFLINE=0 $FPY -c "import sys; sys.path.insert(0, '$CODE'); import metricx_qe as m; print('MXDL w$W', m.download('xxl', '$MX_ROOT'))"
fi
if [ "${FULL:-0}" = 1 ]; then
  bash $CODE/setup_comet.sh
  if [ "$W" = 0 ]; then
    [ -x \$HOME/mbert-dumb/.venv/bin/python ] || { echo "w0: no DUMB venv (the classifier)"; exit 1; }
    [ "\$(git -C \$HOME/mbert-dumbx rev-parse HEAD 2> /dev/null)" = ae114789bf2eae153e4e3bea579299a617dcd417 ] || { echo "w0: \$HOME/mbert-dumbx is not at ae11478"; exit 1; }
  fi
fi
echo "READY w$W: /dev/shm \$(df -h /dev/shm | tail -1 | awk '{print \$4}') free, boot disk $free GB free"
EOF
  detach "g4a_setup_$TAG" bash ~/job_body_g4a_setup_$TAG.sh
  ;;
train)
  if [ "$ARM" = G4-5 ] || [ $A6 = 1 ]; then
    case ${PHASE:-} in
      ""|prep) if [ "$ARM" = G4-5 ]; then g45_train; else g46_train; fi ;;
      launch) RUN=${RUN:?the run that PHASE=prep prepared}
              [ -s "$G_SHM/data-$RUN/launch.sh" ] || { echo "w$W: no $G_SHM/data-$RUN/launch.sh (PHASE=prep first)"; exit 1; }
              bash "$G_SHM/data-$RUN/launch.sh" ;;
      *) echo "PHASE is prep, launch or empty (both), not $PHASE"; exit 1 ;;
    esac
    exit 0
  fi
  RUN=${RUN:?$SMOKE_RUN (with STEPS=8) or $FULL_RUN}
  STEPS=${STEPS:-auto}
  case "$ARM/$RUN/$STEPS" in "$ARM/$SMOKE_RUN/8"|"$ARM/$FULL_RUN/auto"|"G4-4/$NOVELS_RUN/1") ;; *) echo "RUN/STEPS is $SMOKE_RUN/8 or $FULL_RUN/auto (G4-4 also $NOVELS_RUN/1), not $RUN/$STEPS"; exit 1 ;; esac
  tpu_free
  if [ "$ARM" = G4-1 ]; then
    gsutil -q stat "$G/data/check.txt" || { echo "no $G/data/check.txt: the set is not checked"; exit 1; }
    D=$G_SHM/data
    mkdir -p "$D"
    gsutil -m -q cp "$G/data/$ARM/train.parquet" "$G/data/$ARM/template.jinja" "$G/data/$ARM/manifest.json" "$D/"
    # protocol-g4.md §3: one job over the 16 chips, sequences of 2,048, one per chip, LoRA rank 32 / alpha 16 on
    # run_sft.sh's modules (attention + MLP), LR 1e-5 cosine with 10% warmup, one epoch (STEPS=auto) or 8 steps
    MULTIHOST=1 NAME=$RUN MODEL=gemma4-31b DATA=$D/train.parquet TEMPLATE=$D/template.jinja ROOT=$G STEPS=$STEPS \
      bash "$CODE/run_sft.sh" per_device_batch_size=1 max_target_length=2048 lora.lora_rank=32 lora.lora_alpha=16.0 \
      learning_rate=1e-5
  elif [ "$ARM" = G4-3 ] || [ "$ARM" = G4-4 ]; then
    # amendment G4-3.4/G4-3.5: the smoke test on the smoke set (pass 1's longest examples first), the full run on G4-3's
    # set (PASSES=2: its first two passes, amendment G4-3.5's rule above 5 days); both at SEQ3 with the four held-out
    # evals (G4-2's three and the long validation rows). Amendment G4-4.4: G4-4 the same on its own set, three passes
    PASSES=${PASSES:-3}
    case $ARM/$PASSES in G4-3/2|G4-3/3|G4-4/3) ;; *) echo "PASSES is 2 or 3 (G4-4: 3), not $PASSES"; exit 1 ;; esac
    if [ "$RUN" = "$SMOKE_RUN" ]; then SETDIR=$G/data/$ARM-smoke; EVERY=4; else SETDIR=$G/data/$ARM; EVERY=$(g4a_const EVAL_EVERY3); fi
    # amendment G4-4.11: G4-3's final model on the novels, one step on G4-4's smoke set (its step-0 eval is the measure)
    if [ "$RUN" = "$NOVELS_RUN" ]; then SETDIR=$G/data/G4-4-smoke; EVERY=1; fi
    SEQ=$(g4a_const SEQ3)
    LR=$(g4a_const LR3)
    gsutil -q stat "$CHECK_TXT" || { echo "no $CHECK_TXT: the set is not checked"; exit 1; }
    EVAL_FILES="$H/eval-books.parquet $H/eval-btval.parquet $H/eval-own.parquet $H3/eval-btlong.parquet"
    [ "$ARM" = G4-4 ] && EVAL_FILES="$EVAL_FILES $H4/eval-novels.parquet"
    for f in $EVAL_FILES; do
      gsutil -q stat "$f" || { echo "no $f (heldout, own, heldout3, heldout4)"; exit 1; }
    done
    D=$G_SHM/data-$RUN
    mkdir -p "$D"
    gsutil -m -q cp "$SETDIR/train.parquet" "$SETDIR/train-meta.parquet" "$SETDIR/template.jinja" "$SETDIR/manifest.json" \
      $EVAL_FILES "$D/"
    EVAL_ARGS="--eval books=$D/eval-books.parquet --eval btval=$D/eval-btval.parquet --eval own=$D/eval-own.parquet --eval btlong=$D/eval-btlong.parquet"
    [ "$ARM" = G4-4 ] && EVAL_ARGS="$EVAL_ARGS --eval novels=$D/eval-novels.parquet"
    if [ "$RUN" = "$NOVELS_RUN" ]; then
      gsutil -q cp "$G/G4-3/adapter.npz" "$G/G4-3/adapter.json" "$D/"
      python3 -c "import json, sys; a = json.load(open(sys.argv[1])); assert (a['rank'], a['alpha'], a['scale']) == (32, 16.0, 0.5), a" \
        "$D/adapter.json" || { echo "G4-3's adapter is not rank 32, alpha 16"; exit 1; }
      EVAL_ARGS="--init-adapter $D/adapter.npz --eval novels=$D/eval-novels.parquet"
    fi
    if [ "$RUN" = "$FULL_RUN" ] && [ "$PASSES" = 2 ]; then
      # the passes follow each other in train.parquet: the first two are its first rows
      $SFT_PY -c "import pyarrow.parquet as pq, pyarrow.compute as pc, sys; d = sys.argv[1]; e = pq.read_table(d + '/train-meta.parquet', columns=['epoch']).column('epoch'); n = int(pc.sum(pc.less_equal(e, 2)).as_py()); assert e.slice(0, n).to_pylist() == sorted(e.slice(0, n).to_pylist()) and pc.min(e.slice(n)).as_py() == 3; pq.write_table(pq.read_table(d + '/train.parquet').slice(0, n), d + '/train-2p.parquet'); print('PASSES 2:', n, 'rows')" "$D"
      DATA_FILE=$D/train-2p.parquet
    else
      DATA_FILE=$D/train.parquet
    fi
    SFT_ARGS="$EVAL_ARGS --eval-every $EVERY" \
      MULTIHOST=1 NAME=$RUN MODEL=gemma4-31b DATA=$DATA_FILE TEMPLATE=$D/template.jinja ROOT=$G STEPS=$STEPS \
      bash "$CODE/run_sft.sh" per_device_batch_size=1 max_target_length=$SEQ lora.lora_rank=32 lora.lora_alpha=16.0 \
      learning_rate=$LR lr_schedule_type=wsd warmup_steps_fraction=0.03 wsd_decay_steps_fraction=0.1 \
      wsd_decay_style=linear learning_rate_final_fraction=0.1
  else
    # amendment G4-2.4: the smoke test on G4-1's (checked) set, the full run on G4-2's; both with the held-out evals
    if [ "$RUN" = "$SMOKE_RUN" ]; then SETDIR=$G/data/G4-1; CK=$G/data/check.txt; EVERY=4; else SETDIR=$G/data/G4-2; CK=$CHECK_TXT; EVERY=150; fi
    gsutil -q stat "$CK" || { echo "no $CK: the set is not checked"; exit 1; }
    for f in eval-books.parquet eval-btval.parquet eval-own.parquet; do
      gsutil -q stat "$H/$f" || { echo "no $H/$f (heldout, own)"; exit 1; }
    done
    D=$G_SHM/data-$RUN
    mkdir -p "$D"
    gsutil -m -q cp "$SETDIR/train.parquet" "$SETDIR/template.jinja" "$SETDIR/manifest.json" "$H/eval-books.parquet" \
      "$H/eval-btval.parquet" "$H/eval-own.parquet" "$D/"
    # amendment G4-2.4: LR 1e-4, warmup 3%, constant, the last 10% linear to 1e-5 (MaxText's wsd); the epochs are the
    # set's three passes (STEPS=auto counts them)
    SFT_ARGS="--eval books=$D/eval-books.parquet --eval btval=$D/eval-btval.parquet --eval own=$D/eval-own.parquet --eval-every $EVERY" \
      MULTIHOST=1 NAME=$RUN MODEL=gemma4-31b DATA=$D/train.parquet TEMPLATE=$D/template.jinja ROOT=$G STEPS=$STEPS \
      bash "$CODE/run_sft.sh" per_device_batch_size=1 max_target_length=2048 lora.lora_rank=32 lora.lora_alpha=16.0 \
      learning_rate=1e-4 lr_schedule_type=wsd warmup_steps_fraction=0.03 wsd_decay_steps_fraction=0.1 \
      wsd_decay_style=linear learning_rate_final_fraction=0.1
  fi
  ;;
adapter)
  RUN_DIR=$G/runs/$FULL_RUN
  for _ in $(seq 1 36); do
    gsutil -q stat "$RUN_DIR/adapter.npz" && gsutil -q stat "$RUN_DIR/adapter.json" && gsutil -q stat "$RUN_DIR/summary.json" && break
    sleep 5
  done
  gsutil -q cp "$RUN_DIR/adapter.json" "$G_SHM/adapter-$TAG.json"
  python3 -c "import json, sys; a = json.load(open(sys.argv[1])); assert (a['rank'], a['alpha'], a['scale']) == (32, 16.0, 0.5), a" \
    "$G_SHM/adapter-$TAG.json" || { echo "the adapter is not rank 32, alpha 16"; exit 1; }
  if [ "$ARM" = G4-6c ]; then
    # amendment G4-6c.5: every checkpoint (the steps of --save-steps and the last) as G4-4's adapter and the
    # checkpoint's LoRA side by side, as G4-6b.7, to $G/G4-6c-s<step>/ (the checkpoint's alone in delta/)
    gsutil -q cp "$RUN_DIR/summary.json" "$G_SHM/summary-$TAG.json"
    last=$(json_key "$G_SHM/adapter-$TAG.json" steps)
    for k in $(g4a_const SAVE_ARG6C | tr ',' ' ') "$last"; do
      # CKPT=<step>: that checkpoint alone (each takes a few minutes: ~1 GB down, 2 GB stacked and up)
      [ -z "${CKPT:-}" ] || [ "$k" = "$CKPT" ] || continue
      m=G4-6c-s$k
      d=$G_SHM/g46c-delta-$TAG-s$k
      src=adapter-step$k
      [ "$k" = "$last" ] && src=adapter
      mkdir -p "$d"
      gsutil -q cp "$RUN_DIR/$src.npz" "$d/adapter.npz"
      gsutil -q cp "$RUN_DIR/$src.json" "$d/adapter.json"
      [ "$(json_key "$d/adapter.json" steps)" = "$k" ] || { echo "$RUN_DIR/$src.json is not the adapter of step $k"; exit 1; }
      g46c_stack "$d" "$G_SHM/adapter-stack-$TAG-$m"
      gsutil -q cp "$d/adapter.npz" "$d/adapter.json" "$G/$m/delta/"
      gsutil -q cp "$G_SHM/adapter-stack-$TAG-$m/adapter.npz" "$G_SHM/adapter-stack-$TAG-$m/adapter.json" "$G/$m/"
      gsutil -q cp "$G_SHM/summary-$TAG.json" "$G/$m/summary.json"
      echo "ADAPTER $m: $G/$m/ $(tr -d '\n ' < "$G_SHM/adapter-stack-$TAG-$m/adapter.json" | cut -c1-200)"
      # this stage's own copies (3 GB per checkpoint; the bucket has them): w0's /dev/shm is short (29 GB, 20 Nov)
      rm -rf -- "$d" "$G_SHM/adapter-stack-$TAG-$m"
    done
    exit 0
  elif [ "$ARM" = G4-6b ]; then
    # amendment G4-6b.7: the run's LoRA was trained on G4-4 merged; translate.py --lora takes G4-4's adapter and the
    # run's side by side (g4mix.py stack, both with coefficient 1: shared 2 - weight 1, and weight 1; rank 64, scale 0.5)
    gsutil -q cp "$RUN_DIR/adapter.npz" "$RUN_DIR/adapter.json" "$RUN_DIR/summary.json" "$G/$ARM/delta/"
    g46b_stack "$RUN_DIR" "$G_SHM/adapter-stack-$TAG"
    gsutil -q cp "$G_SHM/adapter-stack-$TAG/adapter.npz" "$G_SHM/adapter-stack-$TAG/adapter.json" "$RUN_DIR/summary.json" "$G/$ARM/"
  else
    gsutil -q cp "$RUN_DIR/adapter.npz" "$RUN_DIR/adapter.json" "$RUN_DIR/summary.json" "$G/$ARM/"
  fi
  echo "ADAPTER $ARM: $G/$ARM/ $(tr -d '\n ' < "$G_SHM/adapter-$TAG.json" | cut -c1-300)"
  ;;
translate|smoke-translate)
  MODEL=${MODEL:?G4-1, G4-2, G4-3, G4-4, G4-5c<k>, G4-6, G4-6b, G4-6c, G4-6c-s<step>, G4-7-i<n> or G4-0}
  case $MODEL in G4-1|G4-2|G4-3|G4-4|G4-6|G4-6b|G4-6c|G4-0|G4-5c[1-9]|G4-6c-s[1-9]*) ;; *) cand7 "$MODEL" || { echo "MODEL is G4-1, G4-2, G4-3, G4-4, G4-5c<k>, G4-6, G4-6b, G4-6c, G4-6c-s<step>, G4-7-i<n> (n = 1..30) or G4-0, not $MODEL"; exit 1; } ;; esac
  # amendment G4-7: its candidates only with ARM=G4-7, and ARM=G4-7 translates and scores its candidates only
  if cand7 "$MODEL"; then [ "$ARM" = G4-7 ] || { echo "$MODEL is G4-7's (ARM=G4-7)"; exit 1; }; fi
  [ "$ARM" != G4-7 ] || cand7 "$MODEL" || { echo "ARM=G4-7: MODEL is a candidate G4-7-i<n> (n = 1..30), not $MODEL"; exit 1; }
  [ "$SET" != dpo ] || [ "$MODEL" = "$REJ6" ] || { echo "SET=dpo with ARM=$ARM: $REJ6 only (the rejected side of its pairs)"; exit 1; }
  case $MODEL in G4-5c*) [ "$SET" = eval ] || { echo "$MODEL: SET=eval only"; exit 1; } ;; esac
  case $MODEL in G4-6c-s*) [ "$SET" = eval ] && [ "$STAGE" = translate ] || { echo "$MODEL: SET=eval only"; exit 1; } ;; esac
  # G4-6c: its smoke run's model only in the smoke test
  [ "$MODEL" != G4-6c ] || [ "$STAGE" = smoke-translate ] || { echo "MODEL=G4-6c is the smoke run's (smoke-translate)"; exit 1; }
  tpu_free
  EVALDIR=$G/data/eval
  if [ "$SET" = dpo ]; then
    # draft amendment G4-6.2: G4-0's greedy translation of the candidates (the rejected side), as the evaluation's
    OUT=$G6R/$MODEL/translations/w$W
    ADAPTER_DIR=none
    # amendment G4-6b.3: G4-4 with its final adapter merged, as its evaluation
    [ "$MODEL" = G4-4 ] && ADAPTER_DIR=$G/G4-4
    SEG=segments-w$W.parquet
    EVALDIR=$G6IN
    CLASSES=$(gsutil cat "$G6IN/classes-w$W.txt")
    NAME=g4a_dtranslate_${MODEL}_$TAG
  elif [ "$SET" = long ]; then
    OUT=$G/long/$MODEL/translations/w$W
    ADAPTER_DIR=$G/$MODEL
    SEG=segments-w$W.parquet
    EVALDIR=$H3
    CLASSES=$(gsutil cat "$H3/classes-w$W.txt")
    NAME=g4a_ltranslate_${MODEL}_$TAG
  elif [ "$SET" = heldout ]; then
    OUT=$G/heldout/$MODEL/translations/w$W
    ADAPTER_DIR=$G/$MODEL
    SEG=segments-w$W.parquet
    EVALDIR=$H
    CLASSES=$(gsutil cat "$H/classes-w$W.txt")
    NAME=g4a_htranslate_${MODEL}_$TAG
  elif [ "$STAGE" = smoke-translate ]; then
    only 0
    OUT=$G/smoke/$TAG/$MODEL/translations
    ADAPTER_DIR=$G/runs/$SMOKE_RUN
    # amendment G4-6b.6: the smoke translations' base is G4-4 with its own adapter (G4-6c: as G4-6b)
    case $ARM in G4-6b|G4-6c) [ "$MODEL" = G4-4 ] && ADAPTER_DIR=$G/G4-4 ;; esac
    SEG=segments-smoke.parquet
    CLASSES=small
    NAME=g4a_smoke_translate_${MODEL}_$TAG
  else
    OUT=$G/eval/$MODEL/translations/w$W
    ADAPTER_DIR=$G/$MODEL
    SEG=segments-w$W.parquet
    CLASSES=$(gsutil cat "$G/data/eval/classes-w$W.txt")
    NAME=g4a_translate_${MODEL}_$TAG
  fi
  CLASSES=$(echo $CLASSES)
  if [ -z "$CLASSES" ]; then
    # the held-out segments fill fewer classes than there are workers
    [ "$SET" != eval ] || { echo "w$W: no classes"; exit 1; }
    printf 'echo "TRANSLATED w%s %s: no class on this worker"\n' "$W" "$MODEL" > ~/job_body_$NAME.sh
    detach "$NAME" bash ~/job_body_$NAME.sh
    exit 0
  fi
  EXTRA=
  if [[ $MODEL == G4-5c* ]]; then
    # draft amendment G4-5: the k-th of the search's top, the components' adapters side by side (g4mix.py stack)
    g45_stack "$MODEL" "$G_SHM/adapter-$NAME"
    if [ "$STAGE" = smoke-translate ]; then META=$G/smoke/$TAG/$MODEL/adapter.json; else META=$G/eval/$MODEL/adapter.json; fi
    gsutil -q cp "$G_SHM/adapter-$NAME/adapter.json" "$META"
    EXTRA="--lora $G_SHM/adapter-$NAME"
  elif { [ "$MODEL" = G4-6b ] || [ "$MODEL" = G4-6c ]; } && [ "$STAGE" = smoke-translate ]; then
    # amendment G4-6b.6: the smoke run's LoRA on G4-4, stacked as the full run's adapter stage does (G4-6c.6: the same)
    g46b_stack "$ADAPTER_DIR" "$G_SHM/adapter-$NAME"
    gsutil -q cp "$G_SHM/adapter-$NAME/adapter.json" "$G/smoke/$TAG/$MODEL/adapter.json"
    EXTRA="--lora $G_SHM/adapter-$NAME"
  elif [ "$MODEL" != G4-0 ]; then
    adapter_local "$ADAPTER_DIR" "$G_SHM/adapter-$NAME"
    EXTRA="--lora $G_SHM/adapter-$NAME"
    # amendment G4-7.4: a candidate's LoRA is rank 32 / alpha 16 from G4-0 (nothing stacked)
    if cand7 "$MODEL" && ! python3 -c "import json, sys; a = json.load(open(sys.argv[1])); assert (a['rank'], a['alpha'], a['scale']) == (32, 16.0, 0.5), a" \
        "$G_SHM/adapter-$NAME/adapter.json"; then
      rm -rf -- "$G_SHM/adapter-$NAME"
      echo "$ADAPTER_DIR/adapter.json is not rank 32, alpha 16"
      exit 1
    fi
  fi
  {
    echo "set -uo pipefail"
    echo "$VARS"
    echo "$ONE_HOST"
    echo "export JAX_COMPILATION_CACHE_DIR=$G_SHM/jax_cache"
    echo "mkdir -p $G_SHM/seg-$TAG $G_SHM/work"
    echo "mkdir -p $G_SHM/seg-$TAG/$SET"
    echo "gsutil -q cp $EVALDIR/$SEG $G_SHM/seg-$TAG/$SET/$SEG || exit 1"
    echo "$PREFLIGHT"
    echo "$TRANSLATE_CLASS"
    for cls in $CLASSES; do
      echo "translate_class $G_SHM/seg-$TAG/$SET/$SEG $cls $OUT/$cls ~/logs/${NAME}_$cls.log $EXTRA || exit 1"
    done
    # G4-6c: the job's own copy of a checkpoint's adapter (2 GB on every worker; the bucket has it); G4-7: a candidate's
    case $MODEL in G4-6c-s*|G4-7-i*) echo "rm -rf -- $G_SHM/adapter-$NAME" ;; esac
    echo "echo \"TRANSLATED w$W $MODEL: $CLASSES\""
  } > ~/job_body_$NAME.sh
  detach "$NAME" bash ~/job_body_$NAME.sh
  ;;
score|smoke-score)
  MODEL=${MODEL:?G4-1, G4-2, G4-3, G4-4, G4-5c<k>, G4-6, G4-6b, G4-6c, G4-6c-s<step>, G4-7-i<n> or G4-0}
  case $MODEL in G4-1|G4-2|G4-3|G4-4|G4-6|G4-6b|G4-6c|G4-0|G4-5c[1-9]|G4-6c-s[1-9]*) ;; *) cand7 "$MODEL" || { echo "MODEL is G4-1, G4-2, G4-3, G4-4, G4-5c<k>, G4-6, G4-6b, G4-6c, G4-6c-s<step>, G4-7-i<n> (n = 1..30) or G4-0, not $MODEL"; exit 1; } ;; esac
  # amendment G4-7: its candidates only with ARM=G4-7, and ARM=G4-7 translates and scores its candidates only
  if cand7 "$MODEL"; then [ "$ARM" = G4-7 ] || { echo "$MODEL is G4-7's (ARM=G4-7)"; exit 1; }; fi
  [ "$ARM" != G4-7 ] || cand7 "$MODEL" || { echo "ARM=G4-7: MODEL is a candidate G4-7-i<n> (n = 1..30), not $MODEL"; exit 1; }
  case $MODEL in G4-6c-s*) [ "$SET" = eval ] && [ "$STAGE" = score ] || { echo "$MODEL: SET=eval only"; exit 1; } ;; esac
  [ "$MODEL" != G4-6c ] || [ "$STAGE" = smoke-score ] || { echo "MODEL=G4-6c is the smoke run's (smoke-score)"; exit 1; }
  if [ "$SET" != eval ]; then
    # the held-out and the long segments, and G4-6's rejected run: collect only, on the CPU
    if [ "$SET" = long ]; then
      O=$G/long/$MODEL; NAME=g4a_lscore_${MODEL}_$TAG; H=$H3
    elif [ "$SET" = dpo ]; then
      O=$G6R/$MODEL; NAME=g4a_dscore_${MODEL}_$TAG; H=$G6IN
    else
      O=$G/heldout/$MODEL; NAME=g4a_hscore_${MODEL}_$TAG
    fi
    L=$G_SHM/score-$NAME
    if [ -z "$(gsutil cat "$H/classes-w$W.txt" | tr -d ' \n')" ]; then
      printf 'echo "SCORED w%s %s: no class on this worker"\n' "$W" "$MODEL" > ~/job_body_$NAME.sh
    else
      cat > ~/job_body_$NAME.sh <<EOF
set -euo pipefail
mkdir -p $L/data $L/translations $L/out
gsutil -q cp $H/evalsets.parquet $H/segments-w$W.parquet $L/data/
gsutil -m -q rsync -r $O/translations/w$W $L/translations
$FPY -I $CODE/evalsets.py collect --data $L/data --worker $W --shards $L/translations --out $L/out/pairs-w$W.parquet
gsutil -q cp $L/out/pairs-w$W.parquet $O/pairs/pairs-w$W.parquet
echo "SCORED w$W $MODEL: $O/pairs/"
EOF
    fi
    detach "$NAME" bash ~/job_body_$NAME.sh
    exit 0
  fi
  tpu_free
  if [ "$STAGE" = smoke-score ]; then
    only 0
    O=$G/smoke/$TAG/$MODEL
    SRC=$O/translations
    SEG=segments-smoke.parquet
    COMET=0
    NAME=g4a_smoke_score_${MODEL}_$TAG
  else
    O=$G/eval/$MODEL
    SRC=$O/translations/w$W
    SEG=segments-w$W.parquet
    COMET=1
    NAME=g4a_score_${MODEL}_$TAG
  fi
  L=$G_SHM/score-$NAME
  cat > ~/job_body_$NAME.sh <<EOF
set -euo pipefail
$ONE_HOST
export JAX_COMPILATION_CACHE_DIR=$G_SHM/jax_cache_metricx
mkdir -p $L/data $L/translations $L/out
gsutil -q cp $G/data/eval/evalsets.parquet $L/data/
# evalsets.py collect reads this worker's segments as segments-w<k>.parquet
gsutil -q cp $G/data/eval/$SEG $L/data/segments-w$W.parquet
gsutil -m -q rsync -r $SRC $L/translations
$FPY -I $CODE/evalsets.py collect --data $L/data --worker $W --shards $L/translations --out $L/out/pairs-w$W.parquet
gsutil -q cp $L/out/pairs-w$W.parquet $O/pairs/pairs-w$W.parquet
if gsutil -q stat $O/scores/metricx-w$W.parquet; then
  echo "w$W: MetricX scores are in $O/scores/ before"
else
  timeout 600 $FPY -c "import jax; assert (jax.device_count(), jax.process_count()) == (4, 1), (jax.device_count(), jax.process_count())"
  $FPY -I -u $CODE/metricx_qe.py score --pairs $L/out/pairs-w$W.parquet --pair source:hypothesis:reference --pair source:hypothesis --model xxl --root $MX_ROOT --empty-candidates --out $L/out/metricx-w$W.parquet
  gsutil -q cp $L/out/metricx-w$W.parquet $O/scores/metricx-w$W.parquet
fi
if [ $COMET = 1 ]; then
  n=\$($FPY -I -c "import pyarrow.parquet as pq, pyarrow.compute as pc, sys; t = pq.read_table(sys.argv[1]); t = t.filter(pc.is_valid(t['reference'])); pq.write_table(t, sys.argv[2]); print(t.num_rows)" $L/out/pairs-w$W.parquet $L/out/pairs-ref-w$W.parquet)
  if [ "\$n" = 0 ]; then
    echo "w$W: no row with a reference, no COMET-22"
  elif gsutil -q stat $O/scores/comet-w$W.parquet; then
    echo "w$W: COMET scores are in $O/scores/ before"
  else
    $SHM/comet-venv/bin/python -I -u $CODE/comet_score.py score --root $SHM/comet --pairs $L/out/pairs-ref-w$W.parquet --out $L/out/comet-w$W.parquet
    gsutil -q cp $L/out/comet-w$W.parquet $O/scores/comet-w$W.parquet
  fi
fi
echo "SCORED w$W $MODEL: $O/scores/"
EOF
  detach "$NAME" bash ~/job_body_$NAME.sh
  ;;
texts)
  only 0
  T=$G_SHM/texts-$TAG
  # G4-2: G4-2, G4-0 and (to report) G4-1 in one classify run (amendment G4-2.7)
  MODELS="G4-1 G4-0"
  [ "$ARM" = G4-2 ] && MODELS="G4-2 G4-0 G4-1"
  [ "$ARM" = G4-3 ] && MODELS="G4-3 G4-0 G4-2"                 # amendment G4-3.7
  [ "$ARM" = G4-4 ] && MODELS="G4-4 G4-0 G4-3"                 # amendment G4-4.7
  [ "$ARM" = G4-5 ] && MODELS="${CANDIDATES:?the candidates of the search, G4-5c1 ...} G4-0 ${REFS:-}"   # draft G4-5
  [ "$ARM" = G4-6 ] && MODELS="G4-6 G4-0 G4-4 G4-5c2"          # amendment G4-6.5: G4-4 and G4-5c2 reported
  if [ "$ARM" = G4-6b ]; then
    # amendment G4-6b.7: G4-4, G4-5c2 and, once its report is there (its scores complete), G4-6 reported
    MODELS="G4-6b G4-0 G4-4 G4-5c2"
    if gsutil -q stat "$G/report-G4-6/report.json"; then MODELS="$MODELS G4-6"; fi
  fi
  if [ "$ARM" = G4-6c ]; then
    # amendment G4-6c.7: the checkpoints, G4-0, and G4-4 and G4-6b reported, in one classify run
    for m in ${CANDIDATES:?the checkpoints, G4-6c-s240 ...}; do
      case $m in G4-6c-s[1-9]*) ;; *) echo "CANDIDATES: $m is no checkpoint of G4-6c"; exit 1 ;; esac
    done
    MODELS="$CANDIDATES G4-0 G4-4 G4-6b"
  fi
  if [ "$ARM" = G4-7 ]; then
    # amendment G4-7.7 (spec §6.5): the candidates, G4-0, and G4-4 reported, in one classify run of C1; classify-c2 runs
    # C2 on the same texts
    for m in ${CANDIDATES:?the candidates, G4-7-i5 ...}; do
      cand7 "$m" || { echo "CANDIDATES: $m is no candidate of G4-7"; exit 1; }
    done
    MODELS="$CANDIDATES G4-0 G4-4"
  fi
  PAIRS=""
  for m in $MODELS; do PAIRS="$PAIRS --pairs '$m=$T/$m/pairs-w*.parquet'"; done
  {
    echo "set -euo pipefail"
    for m in $MODELS; do echo "mkdir -p $T/$m && gsutil -m -q cp '$G/eval/$m/pairs/pairs-w*.parquet' $T/$m/"; done
    echo "gsutil -q cp $G/data/eval/evalsets.parquet $S/re/classify/g4-base-texts.parquet $T/"
    echo "$FPY -I -u $CODE/g4arm.py texts --evalsets $T/evalsets.parquet $PAIRS --control $T/g4-base-texts.parquet --out $T/texts.parquet"
    echo "gsutil -q cp $T/texts.parquet $NAT/texts.parquet"
    # the report takes the same models (the classify run has each one's texts)
    echo "echo '$MODELS' | gsutil -q cp - $NAT/models.txt"
    echo "echo \"TEXTS: $NAT/texts.parquet ($MODELS)\""
  } > ~/job_body_g4a_texts_$TAG.sh
  detach "g4a_texts_$TAG" bash ~/job_body_g4a_texts_$TAG.sh
  ;;
classify)
  only 0
  # run_arms.sh's stage, unchanged: the classifier of 10 Nov ($S/arms/naturalness/classifier/) on $NAT/texts.parquet,
  # to $NAT/predictions.parquet; its job is arms_classify_$TAG
  NAT=$NAT TAG=$TAG bash "$CODE/run_arms.sh" classify
  ;;
blind)
  only 0
  # G4-5: the candidate report5 selected, under its own name (the key's and the tally's arm)
  BARM=$ARM
  [ "$ARM" = G4-5 ] && BARM=${SELECTED:?the candidate report5 selected, G4-5c<k>}
  # G4-6c: the checkpoint report6c selected (amendment G4-6c.8)
  if [ "$ARM" = G4-6c ]; then
    BARM=${SELECTED:?the checkpoint report6c selected, G4-6c-s<step>}
    case $BARM in G4-6c-s[1-9]*) ;; *) echo "SELECTED $BARM is no checkpoint of G4-6c"; exit 1 ;; esac
  fi
  # G4-7: the candidate report7 selected (amendment G4-7.8: only if (a1), (a3) and (b) hold)
  if [ "$ARM" = G4-7 ]; then
    BARM=${SELECTED:?the candidate report7 selected, G4-7-i<n>}
    cand7 "$BARM" || { echo "SELECTED $BARM is no candidate of G4-7"; exit 1; }
    gsutil cat "$REP/report.json" | python3 -c "import json, sys; r = json.load(sys.stdin); assert r['selected'] == sys.argv[1] and r['blind_due'], (r['selected'], r['blind_due'])" "$BARM" \
      || { echo "$REP/report.json: $BARM is not report7's choice with (a1), (a3) and (b) holding"; exit 1; }
  fi
  for d in "$S/blind/$ARM" "$S/blind/$ARM-tg"; do
    ! gsutil -q stat "$d/key.jsonl" || { echo "$d has a key.jsonl: the package stays as it is"; exit 1; }
  done
  X=$G_SHM/blind-$TAG
  cat > ~/job_body_g4a_blind_$TAG.sh <<EOF
set -euo pipefail
mkdir -p $X/$BARM $X/G4-0 $X/tg $X/out
gsutil -q cp $S/blind/fixed-2026-11-10/blind100.jsonl $X/
gsutil -m -q cp '$G/eval/$BARM/pairs/pairs-w*.parquet' $X/$BARM/
gsutil -m -q cp '$G/eval/G4-0/pairs/pairs-w*.parquet' $X/G4-0/
gsutil -m -q cp '$S/arms/eval/base-x/pairs/pairs-w*.parquet' $X/tg/
# the arm against its starting point G4-0 (rules (a2), (c)); against TranslateGemma 27B (reported); seeds
# en-nl-sft/blind/ab/<package>/<blind_id> (armeval.base_is_a), key apart
$FPY -I -u $CODE/armeval.py blind --blind $X/blind100.jsonl --base '$X/G4-0/pairs-w*.parquet' --arm '$BARM=$X/$BARM/pairs-w*.parquet' --out $X/out
$FPY -I -u $CODE/armeval.py blind --blind $X/blind100.jsonl --base '$X/tg/pairs-w*.parquet' --arm '$BARM-tg=$X/$BARM/pairs-w*.parquet' --out $X/out
gsutil -m -q cp $X/out/$BARM/* $S/blind/$ARM/
gsutil -m -q cp $X/out/$BARM-tg/* $S/blind/$ARM-tg/
echo "BLIND: $S/blind/$ARM/ $S/blind/$ARM-tg/"
EOF
  detach "g4a_blind_$TAG" bash ~/job_body_g4a_blind_$TAG.sh
  ;;
report)
  only 0
  X=$G_SHM/report-$TAG
  # G4-2: the direct measure from the held-out pairs of G4-2 and G4-0 and the training's evals (amendment G4-2.6)
  HELD=""
  [ "$ARM" = G4-2 ] && HELD="--heldout $X/heldout/evalsets.parquet --heldout-arm '$X/heldout/G4-2/pairs-w*.parquet' --heldout-base '$X/heldout/G4-0/pairs-w*.parquet'"
  # G4-3: G4-2's held-out sets and the long validation rows (amendment G4-3.6)
  [ "$ARM" = G4-3 ] && HELD="--heldout $X/heldout/evalsets.parquet --heldout $X/long/evalsets.parquet --heldout-arm '$X/heldout/G4-3/pairs-w*.parquet' --heldout-arm '$X/long/G4-3/pairs-w*.parquet' --heldout-base '$X/heldout/G4-0/pairs-w*.parquet' --heldout-base '$X/long/G4-0/pairs-w*.parquet'"
  # G4-4: the same held-out sets, and G4-3 beside G4-0 on the same measures (amendment G4-4.7, reported)
  [ "$ARM" = G4-4 ] && HELD="--heldout $X/heldout/evalsets.parquet --heldout $X/long/evalsets.parquet --heldout-arm '$X/heldout/G4-4/pairs-w*.parquet' --heldout-arm '$X/long/G4-4/pairs-w*.parquet' --heldout-base '$X/heldout/G4-0/pairs-w*.parquet' --heldout-base '$X/long/G4-0/pairs-w*.parquet' --other G4-3=$X/G4-3 --heldout-other '$X/heldout/G4-3/pairs-w*.parquet' --heldout-other '$X/long/G4-3/pairs-w*.parquet' --other-summary $X/g4-3-summary.json"
  # G4-6: the same held-out sets, G4-4 beside G4-0 on the same measures (draft amendment G4-6.7, reported); the
  # preference measures come with the training's summary
  [ $A6 = 1 ] && HELD="--heldout $X/heldout/evalsets.parquet --heldout $X/long/evalsets.parquet --heldout-arm '$X/heldout/$ARM/pairs-w*.parquet' --heldout-arm '$X/long/$ARM/pairs-w*.parquet' --heldout-base '$X/heldout/G4-0/pairs-w*.parquet' --heldout-base '$X/long/G4-0/pairs-w*.parquet' --other G4-4=$X/G4-4 --heldout-other '$X/heldout/G4-4/pairs-w*.parquet' --heldout-other '$X/long/G4-4/pairs-w*.parquet' --other-summary $X/g4-4-summary.json"
  # G4-6b: the models of its classify run beyond the arm, G4-0 and G4-4 (texts wrote them), beside on the same measures
  BESIDE=""
  if [ "$ARM" = G4-6b ]; then
    MODELS=$(gsutil cat "$NAT/models.txt") || { echo "no $NAT/models.txt (texts)"; exit 1; }
    for m in $MODELS; do
      case $m in G4-6b|G4-0|G4-4) ;; *) BESIDE="$BESIDE $m" ;; esac
    done
  fi
  cat > ~/job_body_g4a_report_$TAG.sh <<EOF
set -euo pipefail
mkdir -p $X/$ARM $X/G4-0 $X/logs $X/heldout/$ARM $X/heldout/G4-0 $X/long/$ARM $X/long/G4-0
gsutil -m -q rsync -r -x 'translations/.*' $G/eval/$ARM $X/$ARM
gsutil -m -q rsync -r -x 'translations/.*' $G/eval/G4-0 $X/G4-0
gsutil -q cp $G/data/eval/evalsets.parquet $E/baseline-tg27b/scores.parquet $NAT/predictions.parquet $X/
gsutil -q cp $S/arms/naturalness/classifier/results.json $X/classifier-results.json
gsutil -q cp $G/$ARM/summary.json $X/train-summary.json
gsutil -m -q cp '$G/logs/g4a_translate_${ARM}_*' $X/logs/
if [ "$ARM" != G4-1 ]; then
  gsutil -q cp $H/evalsets.parquet $X/heldout/
  gsutil -m -q cp '$G/heldout/$ARM/pairs/pairs-w*.parquet' $X/heldout/$ARM/
  gsutil -m -q cp '$G/heldout/G4-0/pairs/pairs-w*.parquet' $X/heldout/G4-0/
fi
if [ "$ARM" = G4-3 ] || [ "$ARM" = G4-4 ] || [ $A6 = 1 ]; then
  gsutil -q cp $H3/evalsets.parquet $X/long/
  gsutil -m -q cp '$G/long/$ARM/pairs/pairs-w*.parquet' $X/long/$ARM/
  gsutil -m -q cp '$G/long/G4-0/pairs/pairs-w*.parquet' $X/long/G4-0/
fi
OTHER_TALLY=""
if [ "$ARM" = G4-4 ]; then
  mkdir -p $X/G4-3 $X/heldout/G4-3 $X/long/G4-3
  gsutil -m -q rsync -r -x 'translations/.*' $G/eval/G4-3 $X/G4-3
  gsutil -m -q cp '$G/heldout/G4-3/pairs/pairs-w*.parquet' $X/heldout/G4-3/
  gsutil -m -q cp '$G/long/G4-3/pairs/pairs-w*.parquet' $X/long/G4-3/
  gsutil -q cp $G/G4-3/summary.json $X/g4-3-summary.json
  if gsutil -q stat $G/report-G4-3/tally.json; then gsutil -q cp $G/report-G4-3/tally.json $X/g4-3-tally.json; OTHER_TALLY="--other-tally $X/g4-3-tally.json"; fi
  # amendment G4-4.11: G4-3's final model on the novels
  gsutil -q cp $G/runs/$NOVELS_RUN/summary.json $X/g4-3-novels-summary.json
  OTHER_TALLY="\$OTHER_TALLY --novels-other $X/g4-3-novels-summary.json"
fi
if [ $A6 = 1 ]; then
  mkdir -p $X/G4-4 $X/heldout/G4-4 $X/long/G4-4
  gsutil -m -q rsync -r -x 'translations/.*' $G/eval/G4-4 $X/G4-4
  gsutil -m -q cp '$G/heldout/G4-4/pairs/pairs-w*.parquet' $X/heldout/G4-4/
  gsutil -m -q cp '$G/long/G4-4/pairs/pairs-w*.parquet' $X/long/G4-4/
  gsutil -q cp $G/G4-4/summary.json $X/g4-4-summary.json
fi
for m in $BESIDE; do
  mkdir -p $X/\$m
  gsutil -m -q rsync -r -x 'translations/.*' $G/eval/\$m $X/\$m
  OTHER_TALLY="\$OTHER_TALLY --beside \$m=$X/\$m"
done
lora=""
for f in $X/logs/*.log; do
  line=\$(grep -m1 '^LORA ' \$f | cut -c6- || true)
  [ -n "\$line" ] && { echo "\$line" > \$f.lora.json; lora="\$lora --lora-line \$f.lora.json"; }
done
TALLY=""
if gsutil -q stat $REP/tally.json; then gsutil -q cp $REP/tally.json $X/; TALLY="--tally $X/tally.json"; fi
$FPY -I -u $CODE/g4arm.py report --arm-name $ARM --evalsets $X/evalsets.parquet --arm $X/$ARM --base $X/G4-0 --tg-scores $X/scores.parquet \
  --predictions $X/predictions.parquet --classifier-results $X/classifier-results.json --train-summary $X/train-summary.json \
  \$lora \$TALLY \$OTHER_TALLY $HELD --out $X/report.md
gsutil -m -q cp $X/report.md $X/report.json $REP/
echo "REPORT_DONE $REP/report.md"
EOF
  detach "g4a_report_$TAG" bash ~/job_body_g4a_report_$TAG.sh
  ;;
smoke-report)
  only 0
  X=$G_SHM/smoke-$TAG
  O=$G/smoke/$TAG
  cat > ~/job_body_g4a_smoke_report_$TAG.sh <<EOF
set -euo pipefail
mkdir -p $X
gsutil -q cp $G/runs/$SMOKE_RUN/summary.json $G/runs/$SMOKE_RUN/adapter.json $X/
gsutil -q cp $O/G4-1/pairs/pairs-w0.parquet $X/arm-pairs.parquet
gsutil -q cp $O/G4-1/scores/metricx-w0.parquet $X/arm-metricx.parquet
gsutil -q cp $O/G4-0/pairs/pairs-w0.parquet $X/base-pairs.parquet
gsutil -q cp $O/G4-0/scores/metricx-w0.parquet $X/base-metricx.parquet
gsutil -q cp $G/logs/g4a_smoke_translate_G4-1_${TAG}_small-w0.log $X/translate-arm.log
CK=""
gsutil -q stat $G/runs/$SMOKE_RUN/maxtext/$SMOKE_RUN/checkpoints/8/commit_success.txt && CK=--checkpoint-ok
$FPY -I -u $CODE/g4arm.py smoke --summary $X/summary.json --adapter $X/adapter.json --translate-log $X/translate-arm.log \
  --arm-pairs $X/arm-pairs.parquet --arm-metricx $X/arm-metricx.parquet --base-pairs $X/base-pairs.parquet \
  --base-metricx $X/base-metricx.parquet \$CK --out $X/g4arm-smoke.md && code=0 || code=\$?
gsutil -m -q cp $X/g4arm-smoke.md $X/g4arm-smoke.json $O/
exit \$code
EOF
  detach "g4a_smoke_report_$TAG" bash ~/job_body_g4a_smoke_report_$TAG.sh
  ;;
fetchval)
  only 3
  B=$G_SHM/btval-$TAG
  cat > ~/job_body_g4a_fetchval_$TAG.sh <<EOF
set -euo pipefail
export HF_HUB_DISABLE_PROGRESS_BARS=1 HF_XET_CACHE=$SHM/xet-cache
if gsutil -q stat $BTVAL_IN/revisions.json; then
  echo "the BT validation is in $BTVAL_IN/ before"
else
  [ -s ~/.cache/huggingface/token ] || { echo "w$W: no ~/.cache/huggingface/token"; exit 1; }
  $SFT_PY -u $CODE/g4arm.py fetch --arm $ARM --split validation --out $B
  rm -rf $B/*/.cache
  gsutil -m -q rsync -r $B $BTVAL_IN
fi
echo "FETCHED: \$(gsutil cat $BTVAL_IN/revisions.json | tr -d '\n ' | cut -c1-400)"
EOF
  detach "g4a_fetchval_$TAG" nice -n 19 ionice -c3 bash ~/job_body_g4a_fetchval_$TAG.sh
  ;;
heldout)
  only 3
  ! gsutil -q stat "$H/manifest.json" || { echo "$H has a manifest.json: the held-out data stay as they are"; exit 1; }
  [ -s "$TOK_SFT/tokenizer.json" ] || { echo "w$W: no tokenizer in $TOK_SFT (setup)"; exit 1; }
  IN=$G_SHM/hin-$TAG
  OUT=$G_SHM/heldout-$TAG
  cat > ~/job_body_g4a_heldout_$TAG.sh <<EOF
set -euo pipefail
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=true HF_HUB_OFFLINE=1 JAX_PLATFORMS=cpu
mkdir -p $IN/g41 $IN/btval
gsutil -m -q cp $G/data/G4-1/* $IN/g41/
gsutil -m -q rsync -r $G/input/bt-val $IN/btval
gsutil -q cp $G/data/report.json $IN/data-report.json
gsutil -q cp $E/data/evalsets.parquet $S/blind/fixed-2026-11-10/blind100.jsonl $IN/
$SFT_PY -u $CODE/g4arm.py heldout --g41 $IN/g41 --btval $IN/btval --data-report $IN/data-report.json \
  --evalsets $IN/evalsets.parquet --blind $IN/blind100.jsonl --tokenizer $TOK_SFT --out $OUT
gsutil -m -q cp $OUT/* $H/
echo "HELDOUT_DONE: $H/"
EOF
  detach "g4a_heldout_$TAG" nice -n 19 ionice -c3 bash ~/job_body_g4a_heldout_$TAG.sh
  ;;
own)
  only 0
  # amendment G4-6b.4: with ARM=G4-6b G4-4's held-out translations, as $G6/eval-own4.parquet
  OWN_DIR=$H; OWN_NAME=eval-own; OWN_MODEL=G4-0
  if [ "$ARM" = G4-6b ]; then OWN_DIR=$G6; OWN_NAME=eval-own4; OWN_MODEL=G4-4; fi
  ! gsutil -q stat "$OWN_DIR/$OWN_NAME.parquet" || { echo "$OWN_DIR has an $OWN_NAME.parquet: it stays as it is"; exit 1; }
  X=$G_SHM/own-$TAG
  cat > ~/job_body_g4a_own_$TAG.sh <<EOF
set -euo pipefail
export TOKENIZERS_PARALLELISM=true HF_HUB_OFFLINE=1 JAX_PLATFORMS=cpu
mkdir -p $X/pairs
gsutil -q cp $H/evalsets.parquet $X/
gsutil -m -q cp '$G/heldout/$OWN_MODEL/pairs/pairs-w*.parquet' $X/pairs/
$SFT_PY -u $CODE/g4arm.py own --heldout $X/evalsets.parquet --pairs '$X/pairs/pairs-w*.parquet' --tokenizer $TOK_SFT --out $X/$OWN_NAME.parquet
gsutil -q cp $X/$OWN_NAME.parquet $X/$OWN_NAME.json $OWN_DIR/
echo "OWN_DONE: $OWN_DIR/$OWN_NAME.parquet"
EOF
  detach "g4a_own_$TAG" nice -n 19 ionice -c3 bash ~/job_body_g4a_own_$TAG.sh
  ;;
smoke2-report)
  only 0
  [ "$ARM" = G4-2 ] || { echo "smoke2-report is G4-2's"; exit 1; }
  X=$G_SHM/smoke2-$TAG
  O=$G/smoke2/$TAG
  cat > ~/job_body_g4a_smoke2_report_$TAG.sh <<EOF
set -euo pipefail
mkdir -p $X
gsutil -q cp $G/runs/$SMOKE_RUN/summary.json $G/runs/$SMOKE_RUN/adapter.json $X/
$FPY -I -u $CODE/g4arm.py smoke2 --summary $X/summary.json --adapter $X/adapter.json --out $X/g4arm2-smoke.md && code=0 || code=\$?
gsutil -m -q cp $X/g4arm2-smoke.md $X/g4arm2-smoke.json $O/
exit \$code
EOF
  detach "g4a_smoke2_report_$TAG" bash ~/job_body_g4a_smoke2_report_$TAG.sh
  ;;
replay)
  only 3
  ! gsutil -q stat "$REPLAY_DIR/replay.parquet" || { echo "$REPLAY_DIR has a replay.parquet: it stays as it is"; exit 1; }
  X=$G_SHM/replay-$TAG
  cat > ~/job_body_g4a_replay_$TAG.sh <<EOF
set -euo pipefail
mkdir -p $X/plan2 $X/shards $X/nl $X/report
gsutil -m -q cp $FNL/plan2/examples.parquet $FNL/plan2/plan.json '$FNL/plan2/segments-w*.parquet' $X/plan2/
gsutil -m -q rsync -r $FNL/translations/g4 $X/shards
gsutil -m -q cp '$FNL/nl-g4/w*.parquet' $X/nl/
gsutil -q cp $FNL/g4-full/report/measures.parquet $FNL/g4-full/report/setup.json $X/report/
$SFT_PY -u $CODE/g4arm.py replay --plan $X/plan2 --shards $X/shards --measures $X/report/measures.parquet --nl $X/nl \
  --setup $X/report/setup.json --out $X/replay.parquet --uri plan=$FNL/plan2 --uri shards=$FNL/translations/g4 \
  --uri measures=$FNL/g4-full/report/measures.parquet --uri nl=$FNL/nl-g4 --uri setup=$FNL/g4-full/report/setup.json \
  $([ "$ARM" = G4-3 ] && echo "--arm G4-3")
gsutil -m -q cp $X/replay.parquet $X/replay.json $REPLAY_DIR/
echo "REPLAY_DONE: $REPLAY_DIR/"
EOF
  detach "g4a_replay_$TAG" nice -n 19 ionice -c3 bash ~/job_body_g4a_replay_$TAG.sh
  ;;
build2)
  only 3
  ! gsutil -q stat "$G/data/G4-2/manifest.json" || { echo "$G/data/G4-2 has a manifest.json: the set stays as it is"; exit 1; }
  [ -s "$TOK_SFT/tokenizer.json" ] || { echo "w$W: no tokenizer in $TOK_SFT (setup)"; exit 1; }
  IN=$G_SHM/in2-$TAG
  OUT=$G_SHM/data2-$TAG
  cat > ~/job_body_g4a_build2_$TAG.sh <<EOF
set -euo pipefail
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=true HF_HUB_OFFLINE=1 JAX_PLATFORMS=cpu
mkdir -p $IN/g41
gsutil -m -q cp $G/data/G4-1/* $IN/g41/
gsutil -q cp $H/books.json $G/data/replay/replay.parquet $IN/
gsutil -q cp $E/data/evalsets.parquet $S/blind/fixed-2026-11-10/blind100.jsonl $R/mainset/flan-eval/eval-rows.parquet $IN/
$SFT_PY -u $CODE/g4arm.py build2 --base-set $IN/g41 --heldout-books $IN/books.json --replay $IN/replay.parquet \
  --flan-eval $IN/eval-rows.parquet --evalsets $IN/evalsets.parquet --blind $IN/blind100.jsonl --tokenizer $TOK_SFT --out $OUT
gsutil -m -q cp $OUT/G4-2/* $G/data/G4-2/
gsutil -q cp $OUT/report.json $G/data/report-G4-2.json
gsutil -q cp $OUT/report.md $G/data/report-G4-2.md
echo "BUILT2: $G/data/G4-2/"
EOF
  detach "g4a_build2_$TAG" nice -n 19 ionice -c3 bash ~/job_body_g4a_build2_$TAG.sh
  ;;
heldout3)
  only 3
  [ "$ARM" = G4-3 ] || { echo "heldout3 is G4-3's"; exit 1; }
  ! gsutil -q stat "$H3/manifest.json" || { echo "$H3 has a manifest.json: the long validation rows stay as they are"; exit 1; }
  [ -s "$TOK_SFT/tokenizer.json" ] || { echo "w$W: no tokenizer in $TOK_SFT (setup)"; exit 1; }
  IN=$G_SHM/h3in-$TAG
  OUT=$G_SHM/heldout3-$TAG
  cat > ~/job_body_g4a_heldout3_$TAG.sh <<EOF
set -euo pipefail
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=true HF_HUB_OFFLINE=1 JAX_PLATFORMS=cpu
mkdir -p $IN/btval
gsutil -m -q rsync -r $BTVAL_IN $IN/btval
gsutil -q cp $H/evalsets.parquet $IN/g42-heldout.parquet
gsutil -q cp $E/data/evalsets.parquet $S/blind/fixed-2026-11-10/blind100.jsonl $IN/
$SFT_PY -u $CODE/g4arm.py heldout3 --btval $IN/btval --heldout $IN/g42-heldout.parquet --evalsets $IN/evalsets.parquet \
  --blind $IN/blind100.jsonl --tokenizer $TOK_SFT --out $OUT
gsutil -m -q cp $OUT/* $H3/
echo "HELDOUT3_DONE: $H3/"
EOF
  detach "g4a_heldout3_$TAG" nice -n 19 ionice -c3 bash ~/job_body_g4a_heldout3_$TAG.sh
  ;;
build3)
  only 3
  [ "$ARM" = G4-3 ] || { echo "build3 is G4-3's"; exit 1; }
  ! gsutil -q stat "$G/data/G4-3/manifest.json" || { echo "$G/data/G4-3 has a manifest.json: the set stays as it is"; exit 1; }
  [ -s "$TOK_SFT/tokenizer.json" ] || { echo "w$W: no tokenizer in $TOK_SFT (setup)"; exit 1; }
  IN=$G_SHM/in3-$TAG
  OUT=$G_SHM/data3-$TAG
  cat > ~/job_body_g4a_build3_$TAG.sh <<EOF
set -euo pipefail
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=true HF_HUB_OFFLINE=1 JAX_PLATFORMS=cpu
mkdir -p $IN/g41 $IN/bt
gsutil -m -q cp $G/data/G4-1/* $IN/g41/
gsutil -m -q rsync -r $BT_IN $IN/bt
gsutil -q cp $H/books.json $REPLAY_DIR/replay.parquet $IN/
gsutil -q cp $H/evalsets.parquet $IN/heldout.parquet
gsutil -q cp $H3/evalsets.parquet $IN/heldout3.parquet
gsutil -q cp $E/data/evalsets.parquet $S/blind/fixed-2026-11-10/blind100.jsonl $R/mainset/flan-eval/eval-rows.parquet $IN/
$SFT_PY -u $CODE/g4arm.py build3 --bt $IN/bt --base-set $IN/g41 --heldout-books $IN/books.json --heldout $IN/heldout.parquet \
  --heldout $IN/heldout3.parquet --replay $IN/replay.parquet --flan-eval $IN/eval-rows.parquet --evalsets $IN/evalsets.parquet \
  --blind $IN/blind100.jsonl --tokenizer $TOK_SFT --out $OUT
gsutil -m -q cp $OUT/G4-3/* $G/data/G4-3/
gsutil -m -q cp $OUT/G4-3-smoke/* $G/data/G4-3-smoke/
gsutil -q cp $OUT/report.json $G/data/report-G4-3.json
gsutil -q cp $OUT/report.md $G/data/report-G4-3.md
echo "BUILT3: $G/data/G4-3/ $G/data/G4-3-smoke/"
EOF
  detach "g4a_build3_$TAG" nice -n 19 ionice -c3 bash ~/job_body_g4a_build3_$TAG.sh
  ;;
smoke3-report)
  only 0
  [ "$ARM" = G4-3 ] || { echo "smoke3-report is G4-3's"; exit 1; }
  X=$G_SHM/smoke3-$TAG
  O=$G/smoke3/$TAG
  cat > ~/job_body_g4a_smoke3_report_$TAG.sh <<EOF
set -uo pipefail
mkdir -p $X
gsutil -q cp $G/runs/$SMOKE_RUN/summary.json $G/runs/$SMOKE_RUN/adapter.json $X/ || exit 1
gsutil -q cp $G/data/G4-3-smoke/manifest.json $X/smoke-manifest.json || exit 1
gsutil -q cp $G/data/report-G4-3.json $X/data-report.json || exit 1
$FPY -I -u $CODE/g4arm.py smoke3 --summary $X/summary.json --adapter $X/adapter.json --smoke-manifest $X/smoke-manifest.json \
  --data-report $X/data-report.json --out $X/g4arm3-smoke.md
code=\$?
gsutil -m -q cp $X/g4arm3-smoke.md $X/g4arm3-smoke.json $O/ || exit 1
exit \$code
EOF
  detach "g4a_smoke3_report_$TAG" bash ~/job_body_g4a_smoke3_report_$TAG.sh
  ;;
build4)
  only 1
  [ "$ARM" = G4-4 ] || { echo "build4 is G4-4's"; exit 1; }
  ! gsutil -q stat "$G/data/G4-4/manifest.json" || { echo "$G/data/G4-4 has a manifest.json: the set stays as it is"; exit 1; }
  [ -s "$TOK_SFT/tokenizer.json" ] || { echo "w$W: no tokenizer in $TOK_SFT (setup)"; exit 1; }
  IN=$G_SHM/in4-$TAG
  OUT=$G_SHM/data4-$TAG
  # few threads: the CPUs serve a TPU training and other work at the same time
  cat > ~/job_body_g4a_build4_$TAG.sh <<EOF
set -euo pipefail
export OMP_NUM_THREADS=4 RAYON_NUM_THREADS=4 TOKENIZERS_PARALLELISM=true HF_HUB_OFFLINE=1 JAX_PLATFORMS=cpu
mkdir -p $IN/g43
gsutil -m -q cp $G/data/G4-3/train.parquet $G/data/G4-3/train-meta.parquet $G/data/G4-3/manifest.json $G/data/G4-3/template.jinja $IN/g43/
gsutil -q cp $G/data/report-G4-3.json $REPLAY_DIR/replay.parquet $B1/b1-selected.parquet $IN/
gsutil -q cp $E/data/evalsets.parquet $S/blind/fixed-2026-11-10/blind100.jsonl $R/mainset/flan-eval/eval-rows.parquet $IN/
$SFT_PY -u $CODE/g4arm.py build4 --g43 $IN/g43 --g43-report $IN/report-G4-3.json --novels $IN/b1-selected.parquet \
  --replay $IN/replay.parquet --flan-eval $IN/eval-rows.parquet --evalsets $IN/evalsets.parquet --blind $IN/blind100.jsonl \
  --tokenizer $TOK_SFT --out $OUT
gsutil -m -q cp $OUT/G4-4/* $G/data/G4-4/
gsutil -m -q cp $OUT/G4-4-smoke/* $G/data/G4-4-smoke/
gsutil -q cp $OUT/report.json $G/data/report-G4-4.json
gsutil -q cp $OUT/report.md $G/data/report-G4-4.md
echo "BUILT4: $G/data/G4-4/ $G/data/G4-4-smoke/"
EOF
  detach "g4a_build4_$TAG" nice -n 19 ionice -c3 bash ~/job_body_g4a_build4_$TAG.sh
  ;;
novels4)
  only 1
  [ "$ARM" = G4-4 ] || { echo "novels4 is G4-4's"; exit 1; }
  ! gsutil -q stat "$H4/manifest.json" || { echo "$H4 has a manifest.json: the set stays as it is"; exit 1; }
  [ -s "$TOK_SFT/tokenizer.json" ] || { echo "w$W: no tokenizer in $TOK_SFT (setup)"; exit 1; }
  IN=$G_SHM/in-novels4-$TAG
  OUT=$G_SHM/novels4-$TAG
  # few threads: the CPUs serve a TPU training at the same time; the manifest goes last (it marks the set complete)
  cat > ~/job_body_g4a_novels4_$TAG.sh <<EOF
set -euo pipefail
export OMP_NUM_THREADS=4 RAYON_NUM_THREADS=4 TOKENIZERS_PARALLELISM=true HF_HUB_OFFLINE=1 JAX_PLATFORMS=cpu
mkdir -p $IN/g44
gsutil -m -q cp $G/data/G4-4/train.parquet $G/data/G4-4/train-meta.parquet $G/data/G4-4/manifest.json $IN/g44/
gsutil -q cp $G/data/report-G4-4.json $B1/b1-selected.parquet $E/data/evalsets.parquet $S/blind/fixed-2026-11-10/blind100.jsonl $IN/
$SFT_PY -u $CODE/g4arm.py novels4 --g44 $IN/g44 --data-report $IN/report-G4-4.json --novels $IN/b1-selected.parquet \
  --evalsets $IN/evalsets.parquet --blind $IN/blind100.jsonl --tokenizer $TOK_SFT --out $OUT
gsutil -m -q cp $OUT/eval-novels.parquet $OUT/eval-novels-meta.parquet $H4/
gsutil -q cp $OUT/manifest.json $H4/
rm -f $IN/g44/train.parquet $IN/g44/train-meta.parquet $IN/b1-selected.parquet
echo "NOVELS4_DONE: $H4/"
EOF
  detach "g4a_novels4_$TAG" nice -n 19 ionice -c3 bash ~/job_body_g4a_novels4_$TAG.sh
  ;;
smoke4-report)
  only 0
  [ "$ARM" = G4-4 ] || { echo "smoke4-report is G4-4's"; exit 1; }
  X=$G_SHM/smoke4-$TAG
  O=$G/smoke4/$TAG
  cat > ~/job_body_g4a_smoke4_report_$TAG.sh <<EOF
set -uo pipefail
mkdir -p $X
gsutil -q cp $G/runs/$SMOKE_RUN/summary.json $G/runs/$SMOKE_RUN/adapter.json $X/ || exit 1
gsutil -q cp $G/data/G4-4-smoke/manifest.json $X/smoke-manifest.json || exit 1
gsutil -q cp $G/data/report-G4-4.json $X/data-report.json || exit 1
$FPY -I -u $CODE/g4arm.py smoke4 --summary $X/summary.json --adapter $X/adapter.json --smoke-manifest $X/smoke-manifest.json \
  --data-report $X/data-report.json --out $X/g4arm4-smoke.md
code=\$?
gsutil -m -q cp $X/g4arm4-smoke.md $X/g4arm4-smoke.json $O/ || exit 1
exit \$code
EOF
  detach "g4a_smoke4_report_$TAG" bash ~/job_body_g4a_smoke4_report_$TAG.sh
  ;;
build5)
  only 1
  [ "$ARM" = G4-5 ] || { echo "build5 is G4-5's"; exit 1; }
  ! gsutil -q stat "$G5/report.json" || { echo "$G5 has a report.json: the sets stay as they are"; exit 1; }
  IN=$G_SHM/in5-$TAG
  OUT=$G_SHM/data5-$TAG
  # BUILD5_EXTRA: other numbers of steps (--warmup-steps N, --branch-steps N, --steps BRANCH=N); report.json goes
  # last (it marks the sets complete)
  cat > ~/job_body_g4a_build5_$TAG.sh <<EOF
set -euo pipefail
export OMP_NUM_THREADS=4 JAX_PLATFORMS=cpu
mkdir -p $IN/g44
gsutil -m -q cp $G/data/G4-4/train.parquet $G/data/G4-4/train-meta.parquet $G/data/G4-4/manifest.json $G/data/G4-4/template.jinja $IN/g44/
gsutil -q cp $G/data/report-G4-4.json $H4/eval-novels-meta.parquet $IN/
$SFT_PY -u $CODE/g4arm.py build5 --g44 $IN/g44 --g44-report $IN/report-G4-4.json --novels-eval $IN/eval-novels-meta.parquet \
  ${BUILD5_EXTRA:-} --out $OUT
for d in $OUT/G4-5-*; do gsutil -m -q cp \$d/* $G5/\$(basename \$d)/; done
gsutil -m -q cp $OUT/eval-flanval.parquet $OUT/eval-flanval-meta.parquet $OUT/report.md $G5/
gsutil -q cp $OUT/report.json $G5/
gsutil -q cp $OUT/report.json $G/data/report-G4-5.json
gsutil -q cp $OUT/report.md $G/data/report-G4-5.md
rm -f $IN/g44/train.parquet $IN/g44/train-meta.parquet
echo "BUILT5: $G5/"
EOF
  detach "g4a_build5_$TAG" nice -n 19 ionice -c3 bash ~/job_body_g4a_build5_$TAG.sh
  ;;
split5)
  only 0
  [ "$ARM" = G4-5 ] || { echo "split5 is G4-5's"; exit 1; }
  ! gsutil -q stat "$G5S/split.json" || { echo "$G5S has a split.json: the split stays as it is"; exit 1; }
  [ -s "$TOK_SFT/tokenizer.json" ] || { echo "w$W: no tokenizer in $TOK_SFT (setup)"; exit 1; }
  X=$G_SHM/split5-$TAG
  cat > ~/job_body_g4a_split5_$TAG.sh <<EOF
set -euo pipefail
export TOKENIZERS_PARALLELISM=true HF_HUB_OFFLINE=1 JAX_PLATFORMS=cpu
mkdir -p $X
gsutil -q cp $G/data/eval/evalsets.parquet $X/
$SFT_PY -u $CODE/g4arm.py split5 --evalsets $X/evalsets.parquet --tokenizer $TOK_SFT --out $X/out
gsutil -m -q cp $X/out/eval-*.parquet $G5S/
gsutil -q cp $X/out/split.json $G5S/
echo "SPLIT5_DONE: $G5S/"
EOF
  detach "g4a_split5_$TAG" bash ~/job_body_g4a_split5_$TAG.sh
  ;;
refqe5)
  only 0
  [ "$ARM" = G4-5 ] || { echo "refqe5 is G4-5's"; exit 1; }
  ! gsutil -q stat "$G5S/refqe.parquet" || { echo "$G5S has a refqe.parquet"; exit 1; }
  tpu_free
  X=$G_SHM/refqe5-$TAG
  # draft amendment G4-5: the human side of (a1) per part: the references scored as an arm's translations (as
  # bt_natural.py refqe did for the protocol's numbers), with this chain's MetricX
  cat > ~/job_body_g4a_refqe5_$TAG.sh <<EOF
set -euo pipefail
$ONE_HOST
export JAX_COMPILATION_CACHE_DIR=$G_SHM/jax_cache_metricx
mkdir -p $X
gsutil -q cp $G/data/eval/evalsets.parquet $X/
$FPY -I -c "import pyarrow as pa, pyarrow.parquet as pq, pyarrow.compute as pc, sys; t = pq.read_table(sys.argv[1], columns=['id', 'set', 'source', 'reference']); t = t.filter(pc.is_in(t['set'], value_set=pa.array(['wmt24pp', 'flores', 'ntrex']))); pq.write_table(t, sys.argv[2]); print('REFS', t.num_rows)" $X/evalsets.parquet $X/refs.parquet
timeout 600 $FPY -c "import jax; assert (jax.device_count(), jax.process_count()) == (4, 1), (jax.device_count(), jax.process_count())"
$FPY -I -u $CODE/metricx_qe.py score --pairs $X/refs.parquet --pair source:reference --model xxl --root $MX_ROOT --empty-candidates --out $X/refqe.parquet
gsutil -q cp $X/refqe.parquet $G5S/refqe.parquet
echo "REFQE5_DONE: $G5S/refqe.parquet"
EOF
  detach "g4a_refqe5_$TAG" bash ~/job_body_g4a_refqe5_$TAG.sh
  ;;
resume5)
  only 0
  [ "$ARM" = G4-5 ] || { echo "resume5 is G4-5's"; exit 1; }
  RUN=${RUN:?g4arm_G4-5_<branch> or g4arm_G4-5_b8}
  case $RUN in
    g4arm_G4-5_b8) FROM=g4arm_G4-5_s8 ;;
    g4arm_G4-5_*) case " $G5_BRANCHES " in *" ${RUN#g4arm_G4-5_} "*) FROM=$G5_WARMUP ;; *) echo "$RUN is no branch"; exit 1 ;; esac ;;
    *) echo "$RUN is no run of G4-5"; exit 1 ;;
  esac
  # the training state of FROM's last step (Tunix's checkpoint: the LoRA parameters and the optimizer state) into
  # RUN's checkpoint directory, where Tunix restores it when RUN starts (sft_lora.py --resume-step checks it)
  gsutil -q cp "$G/runs/$FROM/summary.json" "$G_SHM/resume5-$RUN.json"
  step=$(json_key "$G_SHM/resume5-$RUN.json" steps)
  SRC=$G/runs/$FROM/maxtext/$FROM/checkpoints/$step
  DST=$G/runs/$RUN/maxtext/$RUN/checkpoints
  gsutil -q stat "$SRC/commit_success.txt" || { echo "no $SRC/commit_success.txt"; exit 1; }
  [ -z "$(gsutil ls "$G/runs/$RUN/" 2> /dev/null)" ] || { echo "$G/runs/$RUN/ exists: a new run takes a new name"; exit 1; }
  gsutil -m -q cp -r "$SRC" "$DST/"
  gsutil -q stat "$DST/$step/commit_success.txt" || { echo "the copy of $SRC is incomplete"; exit 1; }
  echo "RESUME5 $RUN: $FROM step $step, $(gsutil du -s "$DST/$step" | awk '{print $1}') bytes"
  ;;
report5)
  only 0
  [ "$ARM" = G4-5 ] || { echo "report5 is G4-5's"; exit 1; }
  X=$G_SHM/report5-$TAG
  MODELS5="${CANDIDATES:?the candidates of the search, G4-5c1 ...} G4-0 ${REFS:-}"
  ARGS5=""
  for m in $CANDIDATES; do ARGS5="$ARGS5 --candidate $m=$X/$m"; done
  for m in ${REFS:-}; do ARGS5="$ARGS5 --reference $m=$X/$m"; done
  RUNS5="$G5_WARMUP"
  ARGS5="$ARGS5 --run warmup=$X/runs/$G5_WARMUP.json"
  for b in $G5_BRANCHES; do RUNS5="$RUNS5 g4arm_G4-5_$b"; ARGS5="$ARGS5 --run $b=$X/runs/g4arm_G4-5_$b.json"; done
  cat > ~/job_body_g4a_report5_$TAG.sh <<EOF
set -euo pipefail
mkdir -p $X/runs
for m in $MODELS5; do mkdir -p $X/\$m; gsutil -m -q rsync -r -x 'translations/.*' $G/eval/\$m $X/\$m; done
gsutil -q cp $G/data/eval/evalsets.parquet $G5S/split.json $G5S/refqe.parquet $NAT/predictions.parquet $X/
gsutil -q cp $G/runs/$G5_SEARCH/summary.json $X/search.json
for r in $RUNS5; do gsutil -q cp $G/runs/\$r/summary.json $X/runs/\$r.json; done
${SMOKE5:+gsutil -q cp $SMOKE5 $X/smoke5.json}
TALLY=""
if gsutil -q stat $REP/tally.json; then gsutil -q cp $REP/tally.json $X/; TALLY="--tally $X/tally.json"; fi
$FPY -I -u $CODE/g4arm.py report5 --evalsets $X/evalsets.parquet --split $X/split.json --refqe $X/refqe.parquet \
  --base $X/G4-0 --predictions $X/predictions.parquet --search $X/search.json $ARGS5 \$TALLY ${SMOKE5:+--smoke $X/smoke5.json} --out $X/report.md
gsutil -m -q cp $X/report.md $X/report.json $REP/
echo "REPORT5_DONE $REP/report.md"
EOF
  detach "g4a_report5_$TAG" bash ~/job_body_g4a_report5_$TAG.sh
  ;;
sample-data)
  only 0
  X=$G_SHM/sampledata-$TAG
  cat > ~/job_body_g4s_data_$TAG.sh <<EOF
set -euo pipefail
mkdir -p $X/in $X/out
gsutil -m -q cp $G/data/eval/evalsets.parquet '$G/data/eval/segments-w*.parquet' '$G/data/eval/classes-w*.txt' $X/in/
$FPY -I -u $CODE/g4arm.py sampledata5 --evalsets $X/in/evalsets.parquet --segments $X/in --out $X/out
gsutil -m -q cp $X/out/* $X/in/evalsets.parquet $GS/data/
echo "SAMPLEDATA_DONE $GS/data/"
EOF
  detach "g4s_data_$TAG" bash ~/job_body_g4s_data_$TAG.sh
  ;;
sample-translate|sample-score)
  MODEL=${MODEL:?G4-0 or G4-4}
  case $MODEL in G4-0|G4-4) ;; *) echo "the sampling probe takes G4-0 or G4-4, not $MODEL"; exit 1 ;; esac
  SAMPLES=${SAMPLES:?NAME=T:SEED ...}
  tpu_free
  if [ "$STAGE" = sample-translate ]; then
    NAME=g4s_translate_${MODEL}_$TAG
    CLASSES=$(echo $(gsutil cat "$GS/data/classes-w$W.txt"))
    EXTRA=""
    for x in $SAMPLES; do EXTRA="$EXTRA --sample $x"; done
    if [ "$MODEL" = G4-4 ]; then
      # G4-4's final adapter, merged into the kernels as translate.py --lora does for every arm
      adapter_local "$G/G4-4" "$G_SHM/adapter-$NAME"
      EXTRA="$EXTRA --lora $G_SHM/adapter-$NAME"
    fi
    {
      echo "set -uo pipefail"
      echo "$VARS"
      echo "$ONE_HOST"
      echo "export JAX_COMPILATION_CACHE_DIR=$G_SHM/jax_cache"
      echo "mkdir -p $G_SHM/sseg-$TAG $G_SHM/work"
      echo "gsutil -q cp $GS/data/segments-w$W.parquet $G_SHM/sseg-$TAG/segments-w$W.parquet || exit 1"
      echo "$PREFLIGHT"
      echo "$TRANSLATE_CLASS"
      for cls in $CLASSES; do
        # {sample}: translate.py writes each sample's shards under $GS/$MODEL/<sample>/
        echo "translate_class $G_SHM/sseg-$TAG/segments-w$W.parquet $cls '$GS/$MODEL/{sample}/translations/w$W/$cls' ~/logs/${NAME}_$cls.log $EXTRA || exit 1"
      done
      echo "echo \"TRANSLATED w$W $MODEL: $CLASSES ($SAMPLES)\""
    } > ~/job_body_$NAME.sh
  else
    NAME=g4s_score_${MODEL}_$TAG
    L=$G_SHM/score-$NAME
    NAMES=""
    for x in $SAMPLES; do NAMES="$NAMES ${x%%=*}"; done
    {
      echo "set -euo pipefail"
      echo "$ONE_HOST"
      echo "export JAX_COMPILATION_CACHE_DIR=$G_SHM/jax_cache_metricx"
      echo "mkdir -p $L/data"
      echo "gsutil -q cp $GS/data/evalsets.parquet $GS/data/segments-w$W.parquet $L/data/"
      echo "n=\$($FPY -I -c 'import pyarrow.parquet as pq, sys; print(pq.read_metadata(sys.argv[1]).num_rows)' $L/data/segments-w$W.parquet)"
      echo "if [ \"\$n\" = 0 ]; then echo \"SCORED w$W $MODEL: no probe segment on this worker\"; exit 0; fi"
      JOIN=""
      for n in $NAMES; do
        O=$GS/$MODEL/$n
        echo "mkdir -p $L/$n/translations"
        echo "gsutil -m -q rsync -r $O/translations/w$W $L/$n/translations"
        echo "$FPY -I $CODE/evalsets.py collect --data $L/data --worker $W --shards $L/$n/translations --out $L/$n/pairs-w$W.parquet"
        echo "gsutil -q cp $L/$n/pairs-w$W.parquet $O/pairs/pairs-w$W.parquet"
        JOIN="$JOIN --pairs $n=$L/$n/pairs-w$W.parquet"
      done
      # one MetricX and one COMET pass over all samples of this worker (ids NAME|ID), split back per sample
      echo "$FPY -I -u $CODE/g4arm.py samplejoin $JOIN --out $L/all-pairs.parquet"
      echo "timeout 600 $FPY -c \"import jax; assert (jax.device_count(), jax.process_count()) == (4, 1)\""
      echo "$FPY -I -u $CODE/metricx_qe.py score --pairs $L/all-pairs.parquet --pair source:hypothesis:reference --pair source:hypothesis --model xxl --root $MX_ROOT --empty-candidates --out $L/all-metricx.parquet"
      echo "$FPY -I -u $CODE/g4arm.py samplesplit --scores $L/all-metricx.parquet --out '$L/{sample}/metricx-w$W.parquet'"
      echo "r=\$($FPY -I -c \"import pyarrow.parquet as pq, pyarrow.compute as pc, sys; t = pq.read_table(sys.argv[1]); t = t.filter(pc.is_valid(t['reference'])); pq.write_table(t, sys.argv[2]); print(t.num_rows)\" $L/all-pairs.parquet $L/all-pairs-ref.parquet)"
      echo "if [ \"\$r\" != 0 ]; then"
      echo "  $SHM/comet-venv/bin/python -I -u $CODE/comet_score.py score --root $SHM/comet --pairs $L/all-pairs-ref.parquet --out $L/all-comet.parquet"
      echo "  $FPY -I -u $CODE/g4arm.py samplesplit --scores $L/all-comet.parquet --out '$L/{sample}/comet-w$W.parquet'"
      echo "fi"
      for n in $NAMES; do
        echo "gsutil -q cp $L/$n/metricx-w$W.parquet $GS/$MODEL/$n/scores/metricx-w$W.parquet"
        echo "if [ -e $L/$n/comet-w$W.parquet ]; then gsutil -q cp $L/$n/comet-w$W.parquet $GS/$MODEL/$n/scores/comet-w$W.parquet; fi"
      done
      echo "echo \"SCORED w$W $MODEL: $NAMES\""
    } > ~/job_body_$NAME.sh
  fi
  detach "$NAME" bash ~/job_body_$NAME.sh
  ;;
sample-best8)
  only 0
  X=$G_SHM/sbest8-$TAG
  {
    echo "set -euo pipefail"
    for m in G4-0 G4-4; do
      RUNS=""
      for k in $(seq 1 "$(g4a_const SAMPLE_BEST_OF5)"); do
        echo "mkdir -p $X/$m/t10s$k && gsutil -m -q rsync -r -x 'translations/.*' $GS/$m/t10s$k $X/$m/t10s$k"
        RUNS="$RUNS --run t10s$k=$X/$m/t10s$k"
      done
      echo "$FPY -I -u $CODE/g4arm.py best8 $RUNS --out $X/$m/best8"
      echo "gsutil -m -q rsync -r $X/$m/best8 $GS/$m/best8"
    done
    echo "echo \"BEST8_DONE $GS/<model>/best8/\""
  } > ~/job_body_g4s_best8_$TAG.sh
  detach "g4s_best8_$TAG" bash ~/job_body_g4s_best8_$TAG.sh
  ;;
sample-texts|sample-report)
  only 0
  SETTINGS=${SETTINGS:?t07 t10s1 [tstar] best8}
  X=$G_SHM/${STAGE}-$TAG
  PAIRS=""
  OUTS=""
  {
    echo "set -euo pipefail"
    for m in G4-0 G4-4; do
      echo "mkdir -p $X/$m && gsutil -m -q rsync -r -x 'translations/.*' $G/eval/$m $X/$m"
      PAIRS="$PAIRS --pairs '$m=$X/$m/pairs/pairs-w*.parquet'"
      OUTS="$OUTS --output $m=$X/$m"
      for s in $SETTINGS; do
        echo "mkdir -p $X/$m-$s && gsutil -m -q rsync -r -x 'translations/.*' $GS/$m/$s $X/$m-$s"
        PAIRS="$PAIRS --pairs '$m/$s=$X/$m-$s/pairs/pairs-w*.parquet'"
        OUTS="$OUTS --output $m/$s=$X/$m-$s"
      done
    done
    echo "gsutil -q cp $G/data/eval/evalsets.parquet $S/re/classify/g4-base-texts.parquet $X/"
    if [ "$STAGE" = sample-texts ]; then
      echo "$FPY -I -u $CODE/g4arm.py texts --evalsets $X/evalsets.parquet $PAIRS --control $X/g4-base-texts.parquet --out $X/texts.parquet"
      echo "gsutil -q cp $X/texts.parquet $GS/naturalness/texts.parquet"
      echo "echo \"TEXTS: $GS/naturalness/texts.parquet\""
    else
      echo "gsutil -q cp $GS/naturalness/predictions.parquet $G/report-G4-5/temperature.json $X/"
      echo "$FPY -I -u $CODE/g4arm.py sample5 --evalsets $X/evalsets.parquet $OUTS --predictions $X/predictions.parquet --temperature $X/temperature.json --out $X/report.md"
      echo "gsutil -m -q cp $X/report.md $X/report.json $GS/report/"
      echo "echo \"SAMPLE5_DONE $GS/report/report.md\""
    fi
  } > ~/job_body_g4s_${STAGE#sample-}_$TAG.sh
  detach "g4s_${STAGE#sample-}_$TAG" bash ~/job_body_g4s_${STAGE#sample-}_$TAG.sh
  ;;
sample-classify)
  only 0
  # run_arms.sh's classifier on the probe's texts, a job of its own (arms_classify_<TAG>s)
  NAT=$GS/naturalness TAG=${TAG}s bash "$CODE/run_arms.sh" classify
  ;;
smoke5-report)
  only 0
  [ "$ARM" = G4-5 ] || { echo "smoke5-report is G4-5's"; exit 1; }
  X=$G_SHM/smoke5-$TAG
  O=$G/smoke5/$TAG
  cat > ~/job_body_g4a_smoke5_report_$TAG.sh <<EOF
set -uo pipefail
mkdir -p $X
for r in g4arm_G4-5_s8 g4arm_G4-5_b8 g4arm_G4-5_c8; do
  gsutil -q cp $G/runs/\$r/summary.json $X/\$r.json || exit 1
done
gsutil -q cp $G/runs/g4arm_G4-5_s8/adapter.json $X/s8-adapter.json || exit 1
gsutil -q cp $G/runs/g4arm_G4-5_b8/adapter.json $X/b8-adapter.json || exit 1
gsutil -q cp $G5/G4-5-smoke-warmup/manifest.json $X/warmup-manifest.json || exit 1
gsutil -q cp $G5/G4-5-smoke-branch/manifest.json $X/branch-manifest.json || exit 1
gsutil -q cp $G5/report.json $X/data-report.json || exit 1
gsutil -q cp $G/G4-3/summary.json $X/g43-summary.json || exit 1
gsutil -q cp $G/runs/$NOVELS_RUN/summary.json $X/g43-novels-summary.json || exit 1
$FPY -I -u $CODE/g4arm.py smoke5 --warmup-summary $X/g4arm_G4-5_s8.json --warmup-adapter $X/s8-adapter.json \
  --branch-summary $X/g4arm_G4-5_b8.json --branch-adapter $X/b8-adapter.json --combine-summary $X/g4arm_G4-5_c8.json \
  --warmup-manifest $X/warmup-manifest.json --branch-manifest $X/branch-manifest.json --data-report $X/data-report.json \
  --g43-summary $X/g43-summary.json --g43-novels-summary $X/g43-novels-summary.json --out $X/g4arm5-smoke.md
code=\$?
gsutil -m -q cp $X/g4arm5-smoke.md $X/g4arm5-smoke.json $O/ || exit 1
exit \$code
EOF
  detach "g4a_smoke5_report_$TAG" bash ~/job_body_g4a_smoke5_report_$TAG.sh
  ;;
cands6)
  only 1
  [ "$ARM" = G4-6 ] || { echo "cands6 is G4-6's"; exit 1; }
  ! gsutil -q stat "$G6IN/manifest.json" || { echo "$G6IN has a manifest.json: the candidates stay as they are"; exit 1; }
  [ -s "$TOK_SFT/tokenizer.json" ] || { echo "w$W: no tokenizer in $TOK_SFT (setup)"; exit 1; }
  IN=$G_SHM/in6-$TAG
  OUT=$G_SHM/cands6-$TAG
  # few threads: the CPUs may serve a TPU job at the same time; the manifest goes last (it marks the input complete)
  cat > ~/job_body_g4a_cands6_$TAG.sh <<EOF
set -euo pipefail
export OMP_NUM_THREADS=4 RAYON_NUM_THREADS=4 TOKENIZERS_PARALLELISM=true HF_HUB_OFFLINE=1 JAX_PLATFORMS=cpu
mkdir -p $IN/g44
gsutil -m -q cp $G/data/G4-4/train.parquet $G/data/G4-4/train-meta.parquet $G/data/G4-4/manifest.json $G/data/G4-4/template.jinja $IN/g44/
gsutil -q cp $G/data/report-G4-4.json $H4/eval-novels.parquet $H4/eval-novels-meta.parquet $IN/
$SFT_PY -u $CODE/g4arm.py cands6 --g44 $IN/g44 --g44-report $IN/report-G4-4.json --novels-eval $IN/eval-novels.parquet \
  --novels-eval-meta $IN/eval-novels-meta.parquet --tokenizer $TOK_SFT --out $OUT
gsutil -m -q cp $OUT/cands.parquet $OUT/evalsets.parquet $OUT/segments-w*.parquet $OUT/classes-w*.txt $G6IN/
gsutil -q cp $OUT/manifest.json $G6IN/
rm -f $IN/g44/train.parquet $IN/g44/train-meta.parquet
echo "CANDS6_DONE: $G6IN/"
EOF
  detach "g4a_cands6_$TAG" nice -n 19 ionice -c3 bash ~/job_body_g4a_cands6_$TAG.sh
  ;;
build6)
  only 1
  [ $A6 = 1 ] || { echo "build6 is G4-6's and G4-6b's"; exit 1; }
  ! gsutil -q stat "$G6/report.json" || { echo "$G6 has a report.json: the pairs stay as they are"; exit 1; }
  gsutil -q stat "$G6IN/manifest.json" || { echo "no $G6IN/manifest.json (cands6)"; exit 1; }
  [ -s "$TOK_SFT/tokenizer.json" ] || { echo "w$W: no tokenizer in $TOK_SFT (setup)"; exit 1; }
  IN=$G_SHM/in6b-$TAG
  OUT=$G_SHM/build6-$TAG
  # amendment G4-6b.4: G4-6b's pairs reported beside G4-6's (the same candidates, rejected G4-0)
  COMPARE=""
  [ "$ARM" = G4-6b ] && COMPARE="--compare $IN/compare"
  # the fork's venv: sacrebleu (chrF, as the reports), tokenizers; report.json goes last (it marks the pairs complete)
  cat > ~/job_body_g4a_build6_$TAG.sh <<EOF
set -euo pipefail
export OMP_NUM_THREADS=4 RAYON_NUM_THREADS=4 TOKENIZERS_PARALLELISM=true HF_HUB_OFFLINE=1 JAX_PLATFORMS=cpu
mkdir -p $IN/g40 $IN/held $IN/compare
gsutil -q cp $G6IN/cands.parquet $S/blind/fixed-2026-11-10/blind100.jsonl $IN/
gsutil -q cp $H/evalsets.parquet $IN/heldout-evalsets.parquet
gsutil -q cp $E/data/evalsets.parquet $IN/evalsets.parquet
gsutil -m -q cp '$G6R/$REJ6/pairs/pairs-w*.parquet' $IN/g40/
gsutil -m -q cp '$G/heldout/$REJ6/pairs/pairs-w*.parquet' $IN/held/
if [ -n "$COMPARE" ]; then gsutil -q cp $G/data/G4-6/report.json $G/data/G4-6/pairs.parquet $IN/compare/; fi
$FPY -I -u $CODE/g4arm.py build6 --cands $IN/cands.parquet --rejected '$IN/g40/pairs-w*.parquet' \
  --heldout $IN/heldout-evalsets.parquet --heldout-g40 '$IN/held/pairs-w*.parquet' --evalsets $IN/evalsets.parquet \
  --blind $IN/blind100.jsonl --tokenizer $TOK_SFT --out $OUT --arm $ARM --rejected-model $REJ6 $COMPARE
gsutil -m -q cp $OUT/*.parquet $OUT/template.jinja $OUT/report.md $G6/
gsutil -q cp $OUT/report.json $G6/
gsutil -q cp $OUT/report.json $G/data/report-$ARM.json
gsutil -q cp $OUT/report.md $G/data/report-$ARM.md
echo "BUILT6: $G6/"
EOF
  detach "g4a_build6_$TAG" nice -n 19 ionice -c3 bash ~/job_body_g4a_build6_$TAG.sh
  ;;
check6)
  only 2
  [ $A6 = 1 ] || { echo "check6 is G4-6's and G4-6b's"; exit 1; }
  gsutil -q stat "$G6/report.json" || { echo "no $G6/report.json (build6)"; exit 1; }
  [ -s "$TOK_SFT/tokenizer.json" ] || { echo "w$W: no tokenizer in $TOK_SFT (setup)"; exit 1; }
  D=$G_SHM/check6-$TAG
  cat > ~/job_body_g4a_check6_$TAG.sh <<EOF
set -euo pipefail
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=true HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 JAX_PLATFORMS=cpu
mkdir -p $D
gsutil -m -q cp '$G6/*.parquet' $G6/report.json $D/
$SFT_PY -u $CODE/g4arm.py check6 --data $D --tokenizer $TOK_SFT --processes ${CHECK_PROCESSES:-8} | tee $D/check.txt
gsutil -q cp $D/check.txt $CHECK_TXT
rm -f $D/*.parquet
EOF
  detach "g4a_check6_$TAG" nice -n 19 ionice -c3 bash ~/job_body_g4a_check6_$TAG.sh
  ;;
cpuproof6)
  only 1
  [ $A6 = 1 ] || { echo "cpuproof6 is G4-6's and G4-6b's"; exit 1; }
  gsutil -q stat "$G6/report.json" || { echo "no $G6/report.json (build6)"; exit 1; }
  [ -x "$SFT_PY" ] && [ -s "$TOK_SFT/tokenizer.json" ] || { echo "w$W: no SFT venv or tokenizer (SETUP=data setup)"; exit 1; }
  # on the worker's CPU, the TPU untouched; nothing written to the bucket but the log
  # amendment G4-6b.6: G4-6b's proof also merges an adapter into the base kernels (BAKE=1)
  BAKE=0 SPEC=DPO6 C6=0
  [ "$ARM" = G4-6b ] && BAKE=1
  # amendment G4-6c.6: G4-6c's spec, the adapter kept after a step and the end tokens, besides G4-6b's merge;
  # addendum G4-6c.6e: the reference of a --ref-only run read in (REFIN=1)
  REFIN=0
  [ "$ARM" = G4-6c ] && BAKE=1 SPEC=DPO6C C6=1 REFIN=1
  detach "g4a_cpuproof6_$TAG" env G=$G G6=$G6 H=$H PY=$SFT_PY TOK=$TOK_SFT CODE=$CODE X=$G_SHM/cpuproof6-$TAG BAKE=$BAKE \
    SPEC=$SPEC C6=$C6 REFIN=$REFIN nice -n 19 ionice -c3 bash "$CODE/cpuproof6.sh"
  ;;
smoke6-report)
  only 0
  [ $A6 = 1 ] || { echo "smoke6-report is G4-6's and G4-6b's"; exit 1; }
  # amendment G4-6b.6: G4-6b starts from G4-4 merged; its smoke translations' base is G4-4
  START=""
  case $ARM in G4-6b|G4-6c) START="--start G4-4" ;; esac
  # amendment G4-6c.6: the adapter the smoke run kept after its step SMOKE_SAVE6C
  SAVED=""
  [ "$ARM" = G4-6c ] && SAVED="adapter-step$(g4a_const SMOKE_SAVE6C).json"
  X=$G_SHM/smoke6-$TAG
  O=$G/smoke6/$TAG
  cat > ~/job_body_g4a_smoke6_report_$TAG.sh <<EOF
set -uo pipefail
mkdir -p $X
gsutil -q cp $G/runs/$SMOKE_RUN/summary.json $G/runs/$SMOKE_RUN/adapter.json $X/ || exit 1
${SAVED:+gsutil -q cp $G/runs/$SMOKE_RUN/$SAVED $X/saved-adapter.json || exit 1}
gsutil -q cp $G/G4-4/summary.json $X/g4-4-summary.json || exit 1
gsutil -q cp $G6/report.json $X/build-report.json || exit 1
gsutil -q cp $G/smoke/$TAG/$ARM/pairs/pairs-w0.parquet $X/arm-pairs.parquet || exit 1
gsutil -q cp $G/smoke/$TAG/$ARM/scores/metricx-w0.parquet $X/arm-metricx.parquet || exit 1
gsutil -q cp $G/smoke/$TAG/$REJ6/pairs/pairs-w0.parquet $X/base-pairs.parquet || exit 1
gsutil -q cp $G/smoke/$TAG/$REJ6/scores/metricx-w0.parquet $X/base-metricx.parquet || exit 1
gsutil -q cp $G/logs/g4a_smoke_translate_${ARM}_${TAG}_small-w0.log $X/translate-arm.log || exit 1
$FPY -I -u $CODE/g4arm.py smoke6 --summary $X/summary.json --adapter $X/adapter.json --base-summary $X/g4-4-summary.json \
  --build-report $X/build-report.json --translate-log $X/translate-arm.log --arm-pairs $X/arm-pairs.parquet \
  --arm-metricx $X/arm-metricx.parquet --base-pairs $X/base-pairs.parquet --base-metricx $X/base-metricx.parquet \
  --seq $SEQ6 --arm $ARM $START ${SAVED:+--saved-adapter $X/saved-adapter.json} --out $X/g4arm6-smoke.md
code=\$?
gsutil -m -q cp $X/g4arm6-smoke.md $X/g4arm6-smoke.json $O/ || exit 1
exit \$code
EOF
  detach "g4a_smoke6_report_$TAG" bash ~/job_body_g4a_smoke6_report_$TAG.sh
  ;;
pairtexts6)
  only 0
  [ $A6 = 1 ] || { echo "pairtexts6 is G4-6's and G4-6b's"; exit 1; }
  X=$G_SHM/pairtexts6-$TAG
  cat > ~/job_body_g4a_pairtexts6_$TAG.sh <<EOF
set -euo pipefail
mkdir -p $X
gsutil -q cp $G6/pairs.parquet $X/
$FPY -I -u $CODE/g4arm.py pairtexts6 --pairs $X/pairs.parquet --out $X/texts.parquet
gsutil -q cp $X/texts.parquet $G6NAT/texts.parquet
echo "PAIRTEXTS6: $G6NAT/texts.parquet"
EOF
  detach "g4a_pairtexts6_$TAG" bash ~/job_body_g4a_pairtexts6_$TAG.sh
  ;;
pairclassify6)
  only 0
  [ $A6 = 1 ] || { echo "pairclassify6 is G4-6's and G4-6b's"; exit 1; }
  # run_arms.sh's classify on the pairs' texts; its job is arms_classify_<TAG>p
  NAT=$G6NAT TAG=${TAG}p bash "$CODE/run_arms.sh" classify
  ;;
pairclass6)
  only 0
  [ $A6 = 1 ] || { echo "pairclass6 is G4-6's and G4-6b's"; exit 1; }
  X=$G_SHM/pairclass6-$TAG
  cat > ~/job_body_g4a_pairclass6_$TAG.sh <<EOF
set -euo pipefail
mkdir -p $X
gsutil -q cp $G6NAT/predictions.parquet $X/
$FPY -I -u $CODE/g4arm.py pairclass6 --predictions $X/predictions.parquet --out $X/pairclass.md --arm $ARM --rejected-model $REJ6
gsutil -m -q cp $X/pairclass.md $X/pairclass.json $REP/
echo "PAIRCLASS6: $REP/pairclass.md"
EOF
  detach "g4a_pairclass6_$TAG" bash ~/job_body_g4a_pairclass6_$TAG.sh
  ;;
ends6c)
  only 2
  gsutil -q stat "$G6/report.json" || { echo "no $G6/report.json (G4-6b's build6)"; exit 1; }
  [ -s "$TOK_SFT/tokenizer.json" ] || { echo "w$W: no tokenizer in $TOK_SFT (setup)"; exit 1; }
  D=$G_SHM/ends6c-$TAG
  # amendment G4-6c.3: every packed file of G4-6b's pairs, each segment ending in <turn|>\n (the ids of the tokenizer)
  cat > ~/job_body_g4a_ends6c_$TAG.sh <<EOF
set -euo pipefail
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=true HF_HUB_OFFLINE=1 JAX_PLATFORMS=cpu
mkdir -p $D/in
gsutil -m -q cp '$G6/*.parquet' $D/in/
$SFT_PY -u $CODE/g4arm.py ends6c --data $D/in --tokenizer $TOK_SFT | tee $D/ends.txt
gsutil -q cp $D/ends.txt $G/data/ends-G4-6c.txt
rm -rf -- $D/in
EOF
  detach "g4a_ends6c_$TAG" nice -n 19 ionice -c3 bash ~/job_body_g4a_ends6c_$TAG.sh
  ;;
report6c)
  only 0
  X=$G_SHM/report6c-$TAG
  for m in ${CANDIDATES:?the checkpoints, G4-6c-s240 ...}; do
    case $m in G4-6c-s[1-9]*) ;; *) echo "CANDIDATES: $m is no checkpoint of G4-6c"; exit 1 ;; esac
  done
  # amendment G4-6c.7: G4-4 and G4-6b reported beside (they are in the same classify run)
  REFS6C="G4-4 G4-6b"
  ARGS6C=""
  for m in $CANDIDATES; do ARGS6C="$ARGS6C --candidate $m=$X/$m"; done
  for m in $REFS6C; do ARGS6C="$ARGS6C --reference $m=$X/$m"; done
  cat > ~/job_body_g4a_report6c_$TAG.sh <<EOF
set -euo pipefail
for m in $CANDIDATES G4-0 $REFS6C; do mkdir -p $X/\$m; gsutil -m -q rsync -r -x 'translations/.*' $G/eval/\$m $X/\$m; done
gsutil -q cp $G/data/eval/evalsets.parquet $G5S/split.json $G5S/refqe.parquet $NAT/predictions.parquet $X/
gsutil -q cp $G/runs/$FULL_RUN/summary.json $X/train-summary.json
gsutil -q cp $G/G4-6b/summary.json $X/g4-6b-summary.json
TALLY=""
if gsutil -q stat $REP/tally.json; then gsutil -q cp $REP/tally.json $X/; TALLY="--tally $X/tally.json"; fi
$FPY -I -u $CODE/g4arm.py report6c --evalsets $X/evalsets.parquet --split $X/split.json --refqe $X/refqe.parquet \
  --base $X/G4-0 --predictions $X/predictions.parquet --train-summary $X/train-summary.json \
  --beside-summary G4-6b=$X/g4-6b-summary.json $ARGS6C \$TALLY --out $X/report.md
gsutil -m -q cp $X/report.md $X/report.json $REP/
echo "REPORT6C_DONE $REP/report.md"
EOF
  detach "g4a_report6c_$TAG" bash ~/job_body_g4a_report6c_$TAG.sh
  ;;
classify-c2)
  only 0
  tpu_free
  dumb_ready
  # amendment G4-7.8 and G4-7.10: C2, the reward classifier ($RL/c2/model/), on the texts of C1's classify run, in C1's
  # venv and checkout (run_arms.sh classify) and as the judge side runs C2 (run_g4arm7_judges.sh classify: --max-length
  # 1024); reported beside C1, decides nothing
  for f in "$RL/c2/model/params.pkl" "$RL/c2/model/config.json" "$NAT/texts.parquet"; do
    gsutil -q stat "$f" || { echo "no $f (C2's model: run_g4arm7_judges.sh train; the texts: texts)"; exit 1; }
  done
  X=$G_SHM/classify-c2-$TAG
  cat > ~/job_body_g4a_classify_c2_$TAG.sh <<EOF
set -euo pipefail
$ONE_HOST_DUMB
export JAX_COMPILATION_CACHE_DIR=$G_SHM/jax_cache_dumb PYTHONPATH=src
cd $DUMB
mkdir -p $X/model $G_SHM/jax_cache_dumb
gsutil -m -q cp $RL/c2/model/params.pkl $RL/c2/model/config.json $X/model/
gsutil -q cp $NAT/texts.parquet $X/texts.parquet
$DPY -u $CODE/naturalness.py classify --dumb-code $DUMB --model $X/model --texts $X/texts.parquet --max-length 1024 --out $X/predictions-c2.parquet
gsutil -q cp $X/predictions-c2.parquet $NAT/predictions-c2.parquet
echo "CLASSIFIED_C2: $NAT/predictions-c2.parquet"
EOF
  detach "g4a_classify_c2_$TAG" bash ~/job_body_g4a_classify_c2_$TAG.sh
  ;;
pll)
  only 0
  tpu_free
  dumb_ready
  # amendment G4-7.10 (spec §5.4, protocol.md §4.2.3): PLL-word-l2r under S @ 653k on the classify run's texts, as
  # run_arms.sh pll (whose places are the arms'); reported, decides nothing
  gsutil -q stat "$NAT/texts.parquet" || { echo "no $NAT/texts.parquet (texts)"; exit 1; }
  X=$G_SHM/pll-$TAG
  cat > ~/job_body_g4a_pll_$TAG.sh <<EOF
set -euo pipefail
$ONE_HOST_DUMB
export JAX_COMPILATION_CACHE_DIR=$G_SHM/jax_cache_dumb PYTHONPATH=src
cd $DUMB
mkdir -p $X $G_SHM/jax_cache_dumb
gsutil -q cp $NAT/texts.parquet $X/texts.parquet
$DPY -u $CODE/naturalness.py pll --dumb-code $DUMB --checkpoint $S653 --texts $X/texts.parquet --out $X/pll.parquet
gsutil -q cp $X/pll.parquet $NAT/pll.parquet
echo "PLL: $NAT/pll.parquet"
EOF
  detach "g4a_pll_$TAG" bash ~/job_body_g4a_pll_$TAG.sh
  ;;
report7)
  only 0
  for m in ${CANDIDATES:?the candidates, G4-7-i5 ...}; do
    cand7 "$m" || { echo "CANDIDATES: $m is no candidate of G4-7"; exit 1; }
  done
  # records.json: the chain's list of the RL run's records (gs:// paths; g4arm.py records7)
  for f in "$REP/records.json" "$NAT/predictions.parquet" "$NAT/predictions-c2.parquet" "$G5S/split.json" "$G5S/refqe.parquet"; do
    gsutil -q stat "$f" || { echo "no $f (records.json: the chain; classify; classify-c2; split5; refqe5)"; exit 1; }
  done
  X=$G_SHM/report7-$TAG
  # amendment G4-7.7: G4-4 reported beside (in the same classify run)
  ARGS7=""
  for m in $CANDIDATES; do ARGS7="$ARGS7 --candidate $m=$X/$m"; done
  ARGS7="$ARGS7 --reference G4-4=$X/G4-4"
  cat > ~/job_body_g4a_report7_$TAG.sh <<EOF
set -euo pipefail
for m in $CANDIDATES G4-0 G4-4; do mkdir -p $X/\$m; gsutil -m -q rsync -r -x 'translations/.*' $G/eval/\$m $X/\$m; done
gsutil -q cp $G/data/eval/evalsets.parquet $G5S/split.json $G5S/refqe.parquet $NAT/predictions.parquet $NAT/predictions-c2.parquet $REP/records.json $X/
# the RL run's records in report7's layout: every iteration's report.json, stop.json and summary.json, cal7.json,
# smoke7.json, and stopped.json (written by records7 from the list)
$FPY -I -u $CODE/g4arm.py records7 --list $X/records.json --out $X/records > $X/records-copies.txt
while read -r src dst; do gsutil -q cp "\$src" "\$dst"; done < $X/records-copies.txt
EXTRA7=""
if [ -f $X/records/smoke7.json ]; then EXTRA7="--smoke $X/records/smoke7.json"; fi
if gsutil -q stat $NAT/pll.parquet; then gsutil -q cp $NAT/pll.parquet $X/; EXTRA7="\$EXTRA7 --pll $X/pll.parquet"; fi
if gsutil -q stat $REP/tally.json; then gsutil -q cp $REP/tally.json $X/; EXTRA7="\$EXTRA7 --tally $X/tally.json"; fi
$FPY -I -u $CODE/g4arm.py report7 --evalsets $X/evalsets.parquet --split $X/split.json --refqe $X/refqe.parquet \
  --base $X/G4-0 --predictions $X/predictions.parquet --predictions-c2 $X/predictions-c2.parquet --records $X/records \
  --stopped $X/records/stopped.json --cal7 $X/records/cal7.json$ARGS7 \$EXTRA7 --out $X/report.md
gsutil -m -q cp $X/report.md $X/report.json $REP/
echo "REPORT7_DONE $REP/report.md"
EOF
  detach "g4a_report7_$TAG" bash ~/job_body_g4a_report7_$TAG.sh
  ;;
bon-data)
  only 0
  # protocol-g4.md §15: the sampling probe's segments (G4-5.7b B, $GS/data/) kept to G4-5.4's search part
  X=$G_SHM/bondata-$TAG
  cat > ~/job_body_g4b_data_$TAG.sh <<EOF
set -euo pipefail
mkdir -p $X/in $X/out
gsutil -m -q cp $GS/data/evalsets.parquet '$GS/data/segments-w*.parquet' '$GS/data/classes-w*.txt' $G5S/split.json $X/in/
$FPY -I -u $CODE/g4bon.py bondata --evalsets $X/in/evalsets.parquet --split $X/in/split.json --segments $X/in --out $X/out
gsutil -m -q cp $X/out/* $X/in/evalsets.parquet $GB/data/
echo "BONDATA_DONE $GB/data/"
EOF
  detach "g4b_data_$TAG" bash ~/job_body_g4b_data_$TAG.sh
  ;;
bon-translate|bon-score)
  MODEL=${MODEL:?a candidate of G4-7, G4-7-i<n>}
  # G4-0's 8 samples are the sampling probe's ($GS/G4-0/); new samples only for G4-7's candidate (§15)
  cand7 "$MODEL" || { echo "$STAGE: new samples for a candidate of G4-7 (G4-7-i<n>) only, not $MODEL"; exit 1; }
  tpu_free
  # the probe's T = 1 settings (g4bon.SAMPLES): t10s<k> at T 1 with seed k, written as the probe's chain wrote them
  BON_SAMPLES=""
  for k in 1 2 3 4 5 6 7 8; do BON_SAMPLES="$BON_SAMPLES t10s$k=1:$k"; done
  if [ "$STAGE" = bon-translate ]; then
    NAME=g4b_translate_${MODEL}_$TAG
    CLASSES=$(echo $(gsutil cat "$GB/data/classes-w$W.txt"))
    adapter_local "$G/$MODEL" "$G_SHM/adapter-$NAME"
    # amendment G4-7.4: a candidate's LoRA is rank 32 / alpha 16 from G4-0 (nothing stacked)
    if ! python3 -c "import json, sys; a = json.load(open(sys.argv[1])); assert (a['rank'], a['alpha'], a['scale']) == (32, 16.0, 0.5), a" \
        "$G_SHM/adapter-$NAME/adapter.json"; then
      rm -rf -- "$G_SHM/adapter-$NAME"
      echo "$G/$MODEL/adapter.json is not rank 32, alpha 16"
      exit 1
    fi
    EXTRA="--lora $G_SHM/adapter-$NAME"
    for x in $BON_SAMPLES; do EXTRA="$EXTRA --sample $x"; done
    {
      echo "set -uo pipefail"
      echo "$VARS"
      echo "$ONE_HOST"
      echo "export JAX_COMPILATION_CACHE_DIR=$G_SHM/jax_cache"
      echo "mkdir -p $G_SHM/bseg-$TAG $G_SHM/work"
      echo "gsutil -q cp $GB/data/segments-w$W.parquet $G_SHM/bseg-$TAG/segments-w$W.parquet || exit 1"
      echo "$PREFLIGHT"
      echo "$TRANSLATE_CLASS"
      for cls in $CLASSES; do
        # {sample}: translate.py writes each sample's shards under $GB/$MODEL/<sample>/
        echo "translate_class $G_SHM/bseg-$TAG/segments-w$W.parquet $cls '$GB/$MODEL/{sample}/translations/w$W/$cls' ~/logs/${NAME}_$cls.log $EXTRA || exit 1"
      done
      echo "rm -rf -- $G_SHM/adapter-$NAME"
      echo "echo \"TRANSLATED w$W $MODEL: $CLASSES ($BON_SAMPLES)\""
    } > ~/job_body_$NAME.sh
  else
    NAME=g4b_score_${MODEL}_$TAG
    L=$G_SHM/score-$NAME
    NAMES=""
    for x in $BON_SAMPLES; do NAMES="$NAMES ${x%%=*}"; done
    {
      echo "set -euo pipefail"
      echo "$ONE_HOST"
      echo "export JAX_COMPILATION_CACHE_DIR=$G_SHM/jax_cache_metricx"
      echo "mkdir -p $L/data"
      echo "gsutil -q cp $GB/data/evalsets.parquet $GB/data/segments-w$W.parquet $L/data/"
      echo "n=\$($FPY -I -c 'import pyarrow.parquet as pq, sys; print(pq.read_metadata(sys.argv[1]).num_rows)' $L/data/segments-w$W.parquet)"
      echo "if [ \"\$n\" = 0 ]; then echo \"SCORED w$W $MODEL: no search segment on this worker\"; exit 0; fi"
      JOIN=""
      for n in $NAMES; do
        O=$GB/$MODEL/$n
        echo "mkdir -p $L/$n/translations"
        echo "gsutil -m -q rsync -r $O/translations/w$W $L/$n/translations"
        echo "$FPY -I $CODE/evalsets.py collect --data $L/data --worker $W --shards $L/$n/translations --out $L/$n/pairs-w$W.parquet"
        echo "gsutil -q cp $L/$n/pairs-w$W.parquet $O/pairs/pairs-w$W.parquet"
        JOIN="$JOIN --pairs $n=$L/$n/pairs-w$W.parquet"
      done
      # one MetricX and one COMET pass over all samples of this worker (ids NAME|ID), split back per sample (as the probe)
      echo "$FPY -I -u $CODE/g4arm.py samplejoin $JOIN --out $L/all-pairs.parquet"
      echo "timeout 600 $FPY -c \"import jax; assert (jax.device_count(), jax.process_count()) == (4, 1)\""
      echo "$FPY -I -u $CODE/metricx_qe.py score --pairs $L/all-pairs.parquet --pair source:hypothesis:reference --pair source:hypothesis --model xxl --root $MX_ROOT --empty-candidates --out $L/all-metricx.parquet"
      echo "$FPY -I -u $CODE/g4arm.py samplesplit --scores $L/all-metricx.parquet --out '$L/{sample}/metricx-w$W.parquet'"
      echo "r=\$($FPY -I -c \"import pyarrow.parquet as pq, pyarrow.compute as pc, sys; t = pq.read_table(sys.argv[1]); t = t.filter(pc.is_valid(t['reference'])); pq.write_table(t, sys.argv[2]); print(t.num_rows)\" $L/all-pairs.parquet $L/all-pairs-ref.parquet)"
      echo "if [ \"\$r\" != 0 ]; then"
      echo "  $SHM/comet-venv/bin/python -I -u $CODE/comet_score.py score --root $SHM/comet --pairs $L/all-pairs-ref.parquet --out $L/all-comet.parquet"
      echo "  $FPY -I -u $CODE/g4arm.py samplesplit --scores $L/all-comet.parquet --out '$L/{sample}/comet-w$W.parquet'"
      echo "fi"
      for n in $NAMES; do
        echo "gsutil -q cp $L/$n/metricx-w$W.parquet $GB/$MODEL/$n/scores/metricx-w$W.parquet"
        echo "if [ -e $L/$n/comet-w$W.parquet ]; then gsutil -q cp $L/$n/comet-w$W.parquet $GB/$MODEL/$n/scores/comet-w$W.parquet; fi"
      done
      echo "echo \"SCORED w$W $MODEL: $NAMES\""
    } > ~/job_body_$NAME.sh
  fi
  detach "$NAME" bash ~/job_body_$NAME.sh
  ;;
bon-c2texts|bon-c2)
  only 0
  MODEL=${MODEL:?a candidate of G4-7, G4-7-i<n>}
  # G4-0's samples have C2's scores of c2cal7 ($RL/c2/cal7/); a candidate's samples get theirs here (§15)
  cand7 "$MODEL" || { echo "$STAGE: C2 of a candidate of G4-7 (G4-7-i<n>) only, not $MODEL"; exit 1; }
  dumb_ready
  for f in "$RL/c2/model/params.pkl" "$RL/c2/model/config.json"; do
    gsutil -q stat "$f" || { echo "no $f (C2's model)"; exit 1; }
  done
  if [ "$STAGE" = bon-c2texts ]; then
    X=$G_SHM/bonc2texts-${MODEL}-$TAG
    SAMPLE_ARGS=""
    for k in 1 2 3 4 5 6 7 8; do SAMPLE_ARGS="$SAMPLE_ARGS --sample t10s$k='$X/t10s$k/pairs-w*.parquet'"; done
    {
      echo "set -euo pipefail"
      echo "mkdir -p $X/c2-config $X/greedy $X/g40"
      for k in 1 2 3 4 5 6 7 8; do echo "mkdir -p $X/t10s$k && gsutil -m -q cp '$GB/$MODEL/t10s$k/pairs/pairs-w*.parquet' $X/t10s$k/"; done
      echo "gsutil -m -q cp '$G/eval/$MODEL/pairs/pairs-w*.parquet' $X/greedy/"
      echo "gsutil -m -q cp '$G/eval/G4-0/pairs/pairs-w*.parquet' $X/g40/"
      echo "gsutil -q cp $GB/data/evalsets.parquet $G5S/split.json $X/"
      echo "gsutil -q cp $RL/c2/model/config.json $X/c2-config/config.json"
      echo "nice -n 19 $DPY -I -u $CODE/g4bon.py c2texts --model $MODEL $SAMPLE_ARGS --greedy '$X/greedy/pairs-w*.parquet' --g40 '$X/g40/pairs-w*.parquet' --evalsets $X/evalsets.parquet --split $X/split.json --checkpoint $X/c2-config --out $X/texts.parquet"
      echo "gsutil -q cp $X/texts.parquet $GB/$MODEL/c2/texts.parquet"
      echo "echo \"BON_C2TEXTS: $GB/$MODEL/c2/texts.parquet\""
    } > ~/job_body_g4b_c2texts_${MODEL}_$TAG.sh
    detach "g4b_c2texts_${MODEL}_$TAG" bash ~/job_body_g4b_c2texts_${MODEL}_$TAG.sh
  else
    tpu_free
    gsutil -q stat "$GB/$MODEL/c2/texts.parquet" || { echo "no $GB/$MODEL/c2/texts.parquet (bon-c2texts)"; exit 1; }
    X=$G_SHM/bonc2-${MODEL}-$TAG
    # C2 as classify-c2 and the judge side run it: C1's venv and checkout, --max-length 1024
    cat > ~/job_body_g4b_c2_${MODEL}_$TAG.sh <<EOF
set -euo pipefail
$ONE_HOST_DUMB
export JAX_COMPILATION_CACHE_DIR=$G_SHM/jax_cache_dumb PYTHONPATH=src
cd $DUMB
mkdir -p $X/model $G_SHM/jax_cache_dumb
gsutil -m -q cp $RL/c2/model/params.pkl $RL/c2/model/config.json $X/model/
gsutil -q cp $GB/$MODEL/c2/texts.parquet $X/texts.parquet
$DPY -u $CODE/naturalness.py classify --dumb-code $DUMB --model $X/model --texts $X/texts.parquet --max-length 1024 --out $X/predictions.parquet
gsutil -q cp $X/predictions.parquet $GB/$MODEL/c2/predictions.parquet
echo "BON_C2: $GB/$MODEL/c2/predictions.parquet"
EOF
    detach "g4b_c2_${MODEL}_$TAG" bash ~/job_body_g4b_c2_${MODEL}_$TAG.sh
  fi
  ;;
bon-select)
  only 0
  MODEL=${MODEL:?G4-0, G4-4 or a candidate of G4-7, G4-7-i<n>}
  X=$G_SHM/bonselect-${MODEL}-$TAG
  RUNS=""
  for k in 1 2 3 4 5 6 7 8; do RUNS="$RUNS --sample t10s$k=$X/t10s$k"; done
  if [ "$MODEL" = G4-0 ] || [ "$MODEL" = G4-4 ]; then
    # the sampling probe's samples and best8, C2 of c2cal7 (G4-7.6, the same model as the reward's)
    SRC=$GS/$MODEL
    C2P=$RL/c2/cal7/predictions.parquet
    C2T=$RL/c2/cal7/texts.parquet
    CHECKS="--best8 $X/best8-choice.parquet --probe-report $X/probe-report.json"
    FETCH="gsutil -q cp $GS/$MODEL/best8/choice.parquet $X/best8-choice.parquet && gsutil -q cp $GS/report/report.json $X/probe-report.json"
    if [ "$MODEL" = G4-4 ]; then
      # G4-bon.10: c2cal7 has G4-4's greedy translation only among its calibration rows (the probe's classify texts,
      # from $G/eval/G4-4/); select checks those texts against G4-4's evaluation
      CHECKS="$CHECKS --c2-greedy calib:sampling:G4-4 --greedy $X/greedy"
      FETCH="$FETCH && mkdir -p $X/greedy && gsutil -m -q rsync -r -x 'translations/.*' $G/eval/G4-4 $X/greedy"
    fi
  elif cand7 "$MODEL"; then
    SRC=$GB/$MODEL
    C2P=$GB/$MODEL/c2/predictions.parquet
    C2T=$GB/$MODEL/c2/texts.parquet
    CHECKS=""
    FETCH="true"
  else
    echo "bon-select: G4-0, G4-4 or a candidate of G4-7 (G4-7-i<n>), not $MODEL"; exit 1
  fi
  for f in "$C2P" "$C2T"; do gsutil -q stat "$f" || { echo "no $f (C2's scores of $MODEL)"; exit 1; }; done
  {
    echo "set -euo pipefail"
    for k in 1 2 3 4 5 6 7 8; do echo "mkdir -p $X/t10s$k && gsutil -m -q rsync -r -x 'translations/.*' $SRC/t10s$k $X/t10s$k"; done
    echo "mkdir -p $X/anchor && gsutil -m -q rsync -r -x 'translations/.*' $G/eval/G4-0 $X/anchor"
    echo "gsutil -q cp $GB/data/evalsets.parquet $G5S/split.json $X/"
    echo "gsutil -q cp $C2P $X/c2-predictions.parquet && gsutil -q cp $C2T $X/c2-texts.parquet"
    echo "$FETCH"
    echo "$FPY -I -u $CODE/g4bon.py select --model $MODEL $RUNS --anchor $X/anchor --c2-predictions $X/c2-predictions.parquet --c2-texts $X/c2-texts.parquet --evalsets $X/evalsets.parquet --split $X/split.json $CHECKS --out $X/out"
    echo "gsutil -m -q rsync -r $X/out $GB/$MODEL/select"
    echo "echo \"BONSELECT_DONE $GB/$MODEL/select/\""
  } > ~/job_body_g4b_select_${MODEL}_$TAG.sh
  detach "g4b_select_${MODEL}_$TAG" bash ~/job_body_g4b_select_${MODEL}_$TAG.sh
  ;;
bon-texts|bon-classify|bon-report)
  only 0
  KEY=${KEY:?the key of the classify run, e.g. G4-0 or G4-0_G4-7-i15}
  MODELS=${MODELS:?G4-0 first, then G4-4 or the candidate of G4-7 if there is one}
  [ "${MODELS%% *}" = G4-0 ] || { echo "MODELS: G4-0 first (the deltas are against its greedy translation), not '$MODELS'"; exit 1; }
  for m in ${MODELS#G4-0}; do [ "$m" = G4-4 ] || cand7 "$m" || { echo "MODELS: $m is neither G4-4 nor a candidate of G4-7"; exit 1; }; done
  BNAT=$GB/naturalness/$KEY
  if [ "$STAGE" = bon-classify ]; then
    gsutil -q stat "$BNAT/texts.parquet" || { echo "no $BNAT/texts.parquet (bon-texts)"; exit 1; }
    # C1: run_arms.sh's classify, unchanged, on this measurement's texts; its job is arms_classify_<TAG>b
    NAT=$BNAT TAG=${TAG}b bash "$CODE/run_arms.sh" classify
    exit 0
  fi
  X=$G_SHM/${STAGE}-$TAG
  PAIRS=""
  OUTS=""
  {
    echo "set -euo pipefail"
    echo "mkdir -p $X"
    echo "gsutil -q cp $GB/data/evalsets.parquet $G5S/split.json $G5S/refqe.parquet $X/"
    for m in $MODELS; do
      if [ "$m" = G4-0 ] || [ "$m" = G4-4 ]; then T1=$GS/$m/t10s1; else T1=$GB/$m/t10s1; fi
      echo "mkdir -p $X/$m/greedy $X/$m/t10s1 $X/$m/select"
      echo "gsutil -m -q rsync -r -x 'translations/.*' $G/eval/$m $X/$m/greedy"
      echo "gsutil -m -q rsync -r -x 'translations/.*' $T1 $X/$m/t10s1"
      echo "gsutil -m -q rsync -r $GB/$m/select $X/$m/select"
      PAIRS="$PAIRS --pairs '$m=$X/$m/greedy/pairs/pairs-w*.parquet' --pairs '$m/t10s1=$X/$m/t10s1/pairs/pairs-w*.parquet'"
      for sel in qe r7 c2q c2; do PAIRS="$PAIRS --pairs '$m/$sel=$X/$m/select/$sel/pairs/pairs-w*.parquet'"; done
      OUTS="$OUTS --model $m=$X/$m/greedy,$X/$m/t10s1,$X/$m/select"
      if cand7 "$m"; then
        echo "gsutil -q cp $GB/$m/c2/predictions.parquet $X/$m/c2-predictions.parquet"
        OUTS="$OUTS --c2-check $m=$X/$m/c2-predictions.parquet,$X/cal7-predictions.parquet"
      fi
    done
    if [ "$STAGE" = bon-texts ]; then
      echo "$FPY -I -u $CODE/g4bon.py texts --evalsets $X/evalsets.parquet --split $X/split.json $PAIRS --out $X/texts.parquet"
      echo "gsutil -q cp $X/texts.parquet $BNAT/texts.parquet"
      echo "echo '$MODELS' | gsutil -q cp - $BNAT/models.txt"
      echo "echo \"BONTEXTS_DONE $BNAT/texts.parquet ($MODELS)\""
    else
      echo "gsutil -q cp $BNAT/predictions.parquet $X/predictions.parquet"
      echo "gsutil -q cp $GS/naturalness/predictions.parquet $X/probe-predictions.parquet"
      echo "gsutil -q cp $RL/c2/cal7/predictions.parquet $X/cal7-predictions.parquet"
      echo "$FPY -I -u $CODE/g4bon.py report --evalsets $X/evalsets.parquet --split $X/split.json --refqe $X/refqe.parquet --anchor $X/G4-0/greedy --predictions $X/predictions.parquet --probe-predictions $X/probe-predictions.parquet$OUTS --out $X/report.md"
      echo "gsutil -m -q cp $X/report.md $X/report.json $GB/report/$KEY/"
      echo "echo \"BONREPORT_DONE $GB/report/$KEY/report.md\""
    fi
  } > ~/job_body_g4b_${STAGE#bon-}_$TAG.sh
  detach "g4b_${STAGE#bon-}_$TAG" bash ~/job_body_g4b_${STAGE#bon-}_$TAG.sh
  ;;
*)
  echo "unknown stage $STAGE"
  exit 2
  ;;
esac
