# Protocol: dmbert v2, Dutch ModernBERT 230M op de opt-out-gefilterde FineWeb-2-nl

Status: **vastgelegd** op 15 nov 2026 door de subagent die v2 klaarzet, in opdracht van de hoofdsessie, vóór de start
van de training. Grondslag: het plan `mbert-ops/notes/dmbert-v2-plan.md` (15 nov 07:30–08:40), door Yeb op 15 nov
~08:00 goedgekeurd met alle aanbevelingen (`mbert-ops/HANDOVER.md` §0: "Plan dmbert v2 AKKOORD (Yeb) … op mbert-mix …
naam `yhavinga/dmbert-230m-fineweb2-optout`, beslisregel vóór de start committen"). Wijzigingen na de start komen onder
"Amendementen", gedateerd, en vervangen de tekst erboven niet. De keten verwijdert zelf niets.

Markering waar het ertoe doet: **[citaat]** letterlijk uit de bron, **[gemeten]** een telling of uitslag met bron,
**[eigen keuze]** een technische keuze van deze subagent binnen het goedgekeurde plan.

## Startregels (Yeb, 15 nov ~08:15, vooraf)

[citaat, `HANDOVER.md` §0] "dmbert v2 zonder nieuwe vraag starten (rapport + beslisregel gecommit + smoke-test OK),
ook bij ≥ 40 mld tokens, < 40 mld wachten." Uitgewerkt in de opdracht van de hoofdsessie (15 nov ~14:45): de keten
`mbert-ops/scripts/final_v2_run.sh` start de training pas als

1. de definitieve `kept/train` er is: de marker `gs://t5-dutch-english-usc2/mbert/data/fineweb2-nl-optout/work/final-fwo1115a.done`
   bestaat (na de herhaling van robots.txt voor de `robots_error`-hosts en het opnieuw filteren);
2. dit protocol gecommit en gepusht is (de keten controleert dat zijn code-commit ervan afstamt);
3. de smoke-test op mbert-mix geslaagd is (hieronder);
4. `kept/train` volgens de marker minstens 40.000.000.000 tokens heeft. Daaronder start de keten niet en eindigt hij
   op `FAILED` met de getallen.

## Code

mbert-jax, branch `final-v2`, vanaf `final-run` `fe96902` (T2's code: de trainer van `1c8f2c7`, ongewijzigd). Nieuw,
alleen in `experiments/final-v2/`:

| bestand | wat |
|---|---|
| `stage_optout.py` | plan (uit `manifest.json`), staging per host, held-out-bestand, steekproef van de memorisatietest |
| `prepare_v2.sh` | de stappen op een worker (setup, staging, scan, held-out, controle), losgekoppeld, met log en `EXIT` |
| `start_v2.sh` | het trainercommando van T2 met de data van v2 (`main`, `smoke`, `preflight`) |
| `final_v2_plan.json` | het vaste deel van het plan: stappen, schema, seeds, data, checkpoints |
| `eval_v2.sh` | het deel van de evaluatie op één worker (als `experiments/final/eval_final.sh`, plus DBRD@512, memorisatietest, held-out-evals) |
| `memtest.py` | de memorisatietest |
| `single_host.py` | start de runtime voor één proces, zodat `memtest.py` en `scripts/eval_windows.py` op één worker van de v4-32 draaien (de import van `train_minimal.py` wacht anders op de andere hosts: gemeten op w0, 15 nov) |
| `fw2_reference.json`, `smoke_files.txt` | de referentie van de smoke-test (verlies en staptijd van `fineweb2-nl-230m-chunk1024` tot stap 400) en zijn 8 bronbestanden |
| `decide_v2.py` | de beslisregel hieronder, exact op tellingen (rekent met `experiments/final/decide.py`) |

Tests: `tests/test_final_v2_stage.py`, `tests/test_final_v2_memtest.py`, `tests/test_final_v2_decide.py`. Keten op co1:
`mbert-ops/scripts/final_v2_run.sh`, log `mbert-ops/logs/final_v2_run.log`, overdracht `mbert-ops/HANDOVER-dmbert-v2.md`.

## Het model en het recept: exact T2

| onderdeel | waarde (gelijk aan T2, `experiments/final/protocol.md` en `start_final.sh` @ `1c8f2c7`) |
|---|---|
| model | ModernBERT 230M: 22 lagen, hidden 1.024, 16 heads, intermediate 1.536, globale attention elke 3e laag, lokaal venster 128, RoPE 160k/10k; 230.110.592 parameters |
| tokenizer | `yhavinga/dutch-llama-tokenizer` (vocab 32.128 met `<mask>`) |
| objective | 15% losse tokens, 80/10/10 |
| batch | 4 hosts × 32 rijen × 1.024 tokens = 131.072 plekken per stap; packing, lange documenten opgeknipt in stukken van 1.024 |
| optimizer | AdamW, weight decay 0,01, `--layer-norm-lr-factor 1.0`, clipping 1,0, `--skip-nonfinite-updates` (`optax.apply_if_finite`, zelfstop na 50 op rij) |
| schema | WSD: warmup 1.250 stappen lineair naar 4,8e-4, constant 4,8e-4, cooldown 1-sqrt naar 1e-6 over de laatste 266.000 stappen (vanaf 1.040.000); `--max-steps 1306000 --decay-steps 266000` |
| precisie | bf16 (parameters float32); de float32-regel hieronder |
| seeds | init 42, masking 42 + procesindex, data 42 + pass |
| checkpoints | elke 10.000 stappen (de trainer houdt de laatste 3); `ckpt_keep/` van elke 100.000e stap, 1.040.000 (begin cooldown) en 1.306.000 |
| held-out-eval | elke 25.000 stappen, 50 batches, vaste maskers |

**Eén doorlopende run**, `v2-fw2optout-230m`, van stap 0 tot 1.306.000, zonder tak na één epoch. Het schema is per
stap gelijk aan dat van T2: T2 trainde hetzelfde schema in delen (hoofdrun tot 1.040.000, tak 2 vanaf een kopie),
v2 in één keer. Een hervatting gebeurt alleen na een preemptie of reboot, of volgens de float32-regel. Na een
hervatting seedt de trainer de maskering opnieuw (42 + procesindex), zoals bij T2 op 520.000.

Het runscript van `start_v2.sh main` verschilt van dat van `start_final.sh culturax stable` alleen in (gecontroleerd
15 nov met `PRINT_RUN_SCRIPT=1`, woord voor woord):
- code-map `~/mbert-v2` in plaats van `~/mbert-final`;
- `--dataset fineweb2-nl-optout --dataset-config kept-train` in plaats van `uonlp/CulturaX nl` (alleen een naam: met
  `--data-files-list` leest de trainer de lokale bestanden);
- `--dataset-num-rows`: de documenten van `kept/train` min die van `hebban.nl` (alleen voor de metriek `data/passes`);
- geen `--stop-at-step 1040000`;
- `--eval-file ~/fw2optout_heldout.parquet` en `--data-files-list ~/fw2optout_files.txt`;
- de namen (`v2-fw2optout-230m`, bucketmap `gs://t5-dutch-english-usc2/mbert/v2-fw2optout-230m/`).

`--url-blocklist hebban.nl` en alle andere opties zijn gelijk.

## Data

**Training: `kept/train`** van `gs://t5-dutch-english-usc2/mbert/data/fineweb2-nl-optout/` (pipeline
`github.com/yhavinga/fineweb2-nl-optout`; FineWeb-2 `nld_Latn` @ `af9c133`, crawls t/m CC-MAIN-2023-23, zonder
documenten met een machineleesbaar voorbehoud, PII-patronen vervangen). Niet `kept/test` en niet `excluded/`. De
versie is die van de marker `final-fwo1115a.done`; de keten leest de documenten en tokens uit de marker en
controleert de documenten tegen `manifest.json`.

- Vóór de herhaling van de `robots_error`-hosts [gemeten, rapport `notes/fineweb2-nl-optout.md` @ `f13ff2a`]:
  `kept/` 100.830.825 documenten en 73.455.296.003 tokens, waarvan `kept/train` 100.808.212 documenten. De
  herhaling kan daar alleen documenten af halen.
- Stappen per pass ≈ tokens × 1,0033 (`<s>`/`</s>` per stuk) / (131.072 × 0,9512, de vulling van FineWeb-2). Bij
  73,4 mld tokens is dat ~590.700 stappen per pass, dus ~2,21 passes in 1.306.000 stappen; bij 70 mld ~2,32. Het
  aantal stappen blijft 1.306.000, ongeacht de omvang ("hetzelfde aantal stappen als T2", plan §1.2).

**Staging** (`stage_optout.py plan` en `stage`, op de 4 workers van mbert-mix, CPU, uit de bucket in dezelfde regio):
- het plan: de `kept/train`-bestanden van `manifest.json` in naamvolgorde; per bestand de rijen per host zoals
  `scripts/stage_share.py` een corpus verdeelt (segmenten van 1.000 rijen om de beurt, de overgebleven rijen per beurt
  doorgegeven); het plan met de sha256 van het manifest gaat naar `gs://…/mbert/final-v2/plan/`;
- per host per bronbestand één bestand `~/fw2optout/<stem>-w<rank>of4.parquet` met alleen `text` en `url` (zstd 9),
  in de volgorde van het plan, gecontroleerd op grootte, md5 en rijen tegen het manifest; de lijst
  `~/fw2optout_files.txt` is de `--data-files-list` en ligt daarmee vast (de datastand van de trainer noemt de
  bestanden en hun volgorde);
- de trainer leest de bestanden zelf (`load_dataset("parquet", streaming=True)`, de tekstkolom `text`) en laat met
  `--url-blocklist hebban.nl` op de kolom `url` de documenten van `hebban.nl` en zijn subdomeinen weg, vóór de shuffle,
  zoals bij T2. De trainercode verandert daarvoor niet: de gestagede bestanden hebben de kolommen van T2's staging
  die de trainer gebruikt (getest: `tests/test_final_v2_stage.py`, de loader van de trainer op gestagede bestanden).
- `experiments/final/blocklist_scan.py` telt vóór de start per host de documenten die de blocklist weglaat
  (`gs://…/final-v2/blocklist/`).

**Held-out: `kept/test`** (22.613 documenten vóór de herhaling) min elk document dat ook in `kept/train` staat: zelfde
`id`, of zelfde sha256 van de tekst. De staging vergelijkt elke rij van elke host met `kept/test`; `stage_optout.py
heldout` schrijft het resultaat (`gs://…/final-v2/heldout/`, met md5). Dit bestand is de `--eval-file` van de training
en de "niet geziene" helft van de memorisatietest. De EVAL-regels beslissen niet mee.

## Precisie en niet-eindige stappen (regel vooraf)

Zoals T2's amendement 2, nu vooraf vastgelegd:
1. Een losse niet-eindige stap slaat `apply_if_finite` over; geen ingreep.
2. Stopt de trainer zichzelf na 50 niet-eindige stappen op rij (`RuntimeError: … steps in a row …`), dan hervat de
   keten vanaf het laatste checkpoint met `COMPUTE_DTYPE=float32` tot het volgende veelvoud van 10.000 na de stap
   van de zelfstop, en daarna weer in bf16 tot het eind. Geen andere wijziging. De keten doet dit zelf, ten hoogste
   5 keer; elke keer staat in het log en in het rapport. Een zesde zelfstop is `FAILED`.
3. Twee opeenvolgende gelogde stappen met `Loss: nan` (dus niet overgeslagen) zijn `FAILED`: de keten stopt de
   trainer met `pkill -9` op zijn eigen runnaam vóór de rotatie de gezonde checkpoints weggooit.
4. Trainers weg zonder exitbestand (reboot of preemptie van een worker): hervatten met dezelfde opties vanaf het
   laatste checkpoint, ten hoogste 3 keer.

## Smoke-test vóór de start

Op mbert-mix, met het commando van de hoofdrun als `v2-fw2optout-230m-smoke-<TAG>`, op een klein stuk van de `kept/train`
van vóór de herhaling (8 bronbestanden, op elke host zijn deel; het schema is gelijk) en `kept/test` als held-out:
1. 300 stappen met een checkpoint elke 100 en EVAL op 200; daarna een hervatting tot 400 vanaf checkpoint 300 met zijn
   datastand.
2. Criteria [eigen keuze, naar T2's smoke-test]: verlies op 100, 150 en 200 binnen 10% van dat van de FineWeb-2-run
   `fineweb2-nl-230m-chunk1024` (uit zijn Trackio-database); de mediane staptijd op 100/150/200 ten hoogste 1,25 ×
   T2's 0,24 s; geen overgeslagen stap; de regel van de URL-blocklist; `EVAL step=200`; de hervatting leest het
   checkpoint en de datastand van 300 en logt stap 400; geheugen per chip en piek-RSS per host gerapporteerd.
   Daarna draaien de nieuwe evaluatietaken (memorisatietest en held-out-evals) één keer op checkpoint 400.
3. Vlak vóór de training draait de keten 200 stappen op de definitieve data (`v2-fw2optout-230m-preflight-<TAG>`):
   dezelfde criteria voor verlies (tegen dezelfde referentie), staptijd, overgeslagen stappen, blocklist en EVAL.

De bucketmappen van de smoke-test en de preflight blijven staan (verwijderen alleen met de hand).

## Evaluatie

### Armen

| arm | model | checkpoint | uitslagen |
|---|---|---|---|
| V | v2, `v2-1306000` | `gs://…/mbert/v2-fw2optout-230m/ckpt_keep/1306000` | nieuw: 22, in `gs://…/mbert/final-v2/{dumb,dumb-extra,blimp-nl}/` |
| T2 | `epoch2-1306000` (`yhavinga/dmbert-230m-culturax`) | `gs://…/mbert/final-culturax-230m-epoch2/ckpt_keep/1306000` | bestaan: `gs://…/mbert/final/{dumb,dumb-extra,blimp-nl}/`, DBRD@512 in `gs://…/mbert/final/robbert/dumb/` |
| R23 | RobBERT-2023-base (`DTAI-KULeuven/robbert-2023-dutch-base`) | — | bestaan: `gs://…/mbert/final/robbert/` |

De uitslagen van T2 en R23 worden hergebruikt, omdat V met dezelfde code, venv en hardware draait: fine-tune-code
`ae11478` (`~/mbert-dumbx`), venv `~/mbert-dumb/.venv` (JAX 0.11.0, libtpu 0.0.44; vingerafdruk van de pakketlijst
`91a3c3a5f544` op alle 4 workers van mbert-mix, 15 nov, gelijk aan die van `~/mbert-jax/.venv`), mbert-mix met per
worker een TPU van 4 chips. De keten controleert code en vingerafdruk vóór de evaluatie; wijken ze af, dan `FAILED`
(dan eerst T2 opnieuw, ~30 min, als amendement).

### Taken en instellingen

Gelijk aan T2's protocol (amendement 1) en aan `experiments/robbert/protocol.md`:
- **DBRD@1024** en **SICK-NL@512**: `scripts/dumb/finetune.py`, LR 5e-5 constant, 3 epochs, AdamW wd 0,01, update 192
  (12 × 4 chips × 4) resp. 256 (64 × 4 × 1), DBRD opgevuld tot een veelvoud van 128; seeds 42, 43, 44;
  `test_accuracy` na epoch 3; n = 2.112 resp. 4.864.
- **DBRD@512**: dezelfde instellingen als DBRD@1024 met `--max-length 512` (het commando van `eval_robbert.sh` @
  `5398bb9`, dat T2's DBRD@512 maakte met code `ae11478`).
- **SQuAD-NL v1.1** (F1), **CoNLL-2002** (entity-F1), **POS LassySmall** en **POS Alpino** (UPOS): `finetune_extra.py`,
  LR 5e-5 lineair naar 0 zonder warmup, update 32, 3 epochs, seeds 42, 43, 44.
- **BLiMP-NL**: zero-shot, PLL-word-l2r, 8.400 paren, float32, één getal.

### Toetsen

Zoals `experiments/final/decide.py` (42 tests): per arm en taak het gemiddelde over de seeds en de seed-range (max −
min), alles exact (`test_accuracy` als k/n, POS op woordaantallen, F1 als `Fraction` van de opgeslagen float, BLiMP-NL
op paren); elke uitslag op de instellingen hierboven gecontroleerd; een ontbrekende of afwijkende uitslag is FOUT. De
drempel per taak is de grootste van de twee seed-ranges; BLiMP-NL: 0,5 punt, dus meer dan 42 van de 8.400 paren.
"Meer dan" is strikt. "Slechter"/"beter": het verschil van de gemiddelden is groter dan de drempel.

## Beslisregel

**Stap 1, V tegen T2** (de regel van T2's protocol):
1. "**schaadt**" als V's gemiddelde meer dan de drempel onder dat van T2 ligt op DBRD@1024 of op SICK-NL. Eerst
   getoetst.
2. Anders "**helpt**" als V meer dan de drempel boven T2 ligt op SICK-NL.
3. Anders "**geen verschil**". Winst op DBRD@1024 alleen is "geen verschil".

Daarnaast de extra toetsen van V tegen T2: DBRD@512, SQuAD-NL F1, CoNLL-2002 F1, POS LassySmall, POS Alpino, BLiMP-NL.

**Stap 2, V tegen RobBERT-2023** op DBRD@512, SICK-NL en BLiMP-NL, met dezelfde drempelvorm: de drie taken waarop T2
duidelijk beter was dan RobBERT-2023. SQuAD-NL F1 en POS worden gerapporteerd; CoNLL-2002 alleen intern.

| uitkomst | voorwaarde | gevolg |
|---|---|---|
| **A, vervangt T2** | stap 1 is "helpt" of "geen verschil", **en** geen enkele extra toets van V tegen T2 is "slechter" | v2 wordt het publieke dmbert-230m-model; wat er met T2 gebeurt, beslist Yeb |
| **B, naast T2** | niet A, **en** in stap 2 is V op geen van DBRD@512, SICK-NL en BLiMP-NL "slechter" dan RobBERT-2023 | v2 onder een eigen naam (`yhavinga/dmbert-230m-fineweb2-optout`); T2 blijft zoals het is |
| **C, Yeb beslist opnieuw** | niet A en niet B | niet publiceren zonder nieuw besluit van Yeb |

Publiceren gebeurt in geen geval door de keten; de uitkomst is een advies voor Yeb.

**Geldigheid van de run:** 1.306.000 stappen gehaald (`ckpt_keep/1306000` met `commit_success.txt`); niet-eindige
stappen alleen afgehandeld volgens de regel hierboven; elke andere afwijking als amendement vóór de evaluatie.

De vergelijkingswaarden [gemeten, `decide_v2.py` op de bestaande uitslagen, 15 nov]: T2 DBRD@1024 96,12 [0,28],
SICK-NL 87,23 [0,76], DBRD@512 96,05 [0,52], SQuAD-NL 84,74 [0,60], CoNLL-2002 91,86 [0,49], POS LassySmall 97,90
[0,26], POS Alpino 97,71 [0,52], BLiMP-NL 94,94; RobBERT-2023 DBRD@512 94,93 [0,80], SICK-NL 83,95 [0,72], BLiMP-NL
83,74, SQuAD-NL 84,43 [0,64], POS 97,91 [0,09] en 97,50 [0,40]. Met T2's uitslagen als V geeft `decide_v2.py` uitkomst
A, met overal "geen verschil" in stap 1 en "beter" tegen RobBERT-2023 op de drie beslissende taken (de controle van
de code op echte bestanden).

## Wat niet meebeslist

- **Memorisatietest** (`memtest.py`, plan §5.3 risico 8): 1.000 documenten uit `kept/train` (getrokken uit de share
  van host 0, seed 42) tegen 1.000 van het held-out-bestand, één op één gekoppeld op tokenaantal, elk 64–1.022 tokens
  (één stuk). Per document: MLM-verlies en -accuratesse op 15% losse tokens (maskerseeds 0, 1, 2) en span-reconstructie
  (16 opeenvolgende tokens `<mask>`, 3 posities per document): accuratesse en het aandeel spans dat helemaal goed is;
  float32. Gerapporteerd: per split het gemiddelde en het verschil train − test over de paren met een
  bootstrap-interval van 95% en een sign-flip-p-waarde. Gemeten op V en, als controle, op T2 (dat deze documenten niet
  als zodanig zag; CulturaX en FineWeb-2 overlappen wel, dus T2's verschil is niet per se nul).
- **Held-out-evals**: V en T2 op beide held-out-sets (die van v2 en CulturaX `nl_part_00097_first40000`),
  `scripts/eval_windows.py --windows 1024`.
- de EVAL-regels tijdens de training, zero-shot, en winst op DBRD@1024 alleen.

## Verwachting vooraf

Overgenomen uit het plan §4.5 [eigen gevolgtrekking van de planschrijver]: bij 653k en cosine lag FineWeb-2 tegen
CulturaX op DBRD@1024 −0,84 en op DBRD@512 −1,18, buiten de ranges, en op SICK-NL −1,57 nominaal. Blijft dat zo, dan
is V tegen T2 "schaadt" en wordt de uitkomst waarschijnlijk **B**. De verwachting staat hier zodat de uitkomst geen
verrassing is; ze verandert de regel niet.

## Amendementen

(geen)
