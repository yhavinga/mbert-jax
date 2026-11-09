# Protocol (concept): de eindrun van Dutch ModernBERT 230M, WSD met cooldown-takken na 1 en 2 epochs

Status: **vastgelegd** op 10 nov 2026 ~00:50 CET door de hoofdsessie, na lezing van het concept (geschreven 00:00–01:00
door een subagent), vóór de start van de training. Wijzigingen na de start komen onder "Amendementen", gedateerd,
en vervangen de tekst erboven niet. Eén wijziging bij het vastleggen: de keten verwijdert zelf niets meer (Yebs regel:
verwijderen alleen met de hand, met letterlijke paden in een zichtbaar commando); de bucketmap van de smoke-test
blijft staan, en bij variant (b) verwijdert de hoofdsessie `~/culturax_nl` met de hand vóór de mix.

Besluiten waarop dit rust (LEARNINGS, "Besluiten van Yeb", 9 nov): 15% losse tokens; `hebban.nl` eruit met een
URL-blocklist op alle webcorpora; een WSD-schema met een decay-tak na 1 en na 2 epochs uit dezelfde run, waarbij de
1-epoch-tak ook met S wordt vergeleken en welke tak het eindmodel wordt uit dit protocol volgt; of GPT-NL meegaat,
beslist de hoofdsessie met de vooraf vastgelegde regel in LEARNINGS "Open". Yebs eisen van 9 nov ~23:30:
`optax.apply_if_finite` om de optimizer, epoch 2 anders geshuffled met een eigen seed, exacte hervatting ook over de
epochgrens, checkpoints elke 100.000 stappen en aan het begin van elke cooldown.

Code: branch `final-run` van mbert-jax, vanaf `gptnl-mix` `a7c2a57` (de nieuwste trainer: `gptnl-mix` bevat
`corpus-runs` volledig, met G's code `0d92db4`). Commits: `25def26` (blocklist en scan), `6a0ef8a` (trainer: WSD,
`--skip-nonfinite-updates`, `--url-blocklist`), `501534f` (`gptnl_mix.py` voor het plan van variant (b)), `8f1b1e5`
(scripts), `79f2a35` (het plan van (b)). Scripts: `experiments/final/` (ook in
`gs://t5-dutch-english-usc2/mbert/final/scripts/`); keten op co1: `mbert-ops/scripts/final_run.sh`,
`start_branch.sh`, `final_lib.sh`. Blocklistscans: `gs://t5-dutch-english-usc2/mbert/final/blocklist/`. Plan van (b):
`gs://t5-dutch-english-usc2/mbert/final/plan/` en `experiments/final/final_plan.json`.

## Vragen

1. **WSD tegen cosine.** Is tak 1 (WSD, cooldown eindigend na 1 epoch, 653.000 stappen) minstens zo goed als S
   (`culturax-nl-230m-chunk1024`, cosine 4,8e-4 → 1e-6 over 653.000 stappen)?
2. **Een tweede epoch.** Is tak 2 (cooldown eindigend na 2 epochs, 1.306.000 stappen) beter dan tak 1?
3. **Het eindmodel.** Tak 1 of tak 2 (regel hieronder). S is geen kandidaat: S trainde op de pagina's van
   `hebban.nl`, de bron van DBRD.

## Armen

| arm | run | checkpoint | rol |
|---|---|---|---|
| T1 | `<run>-epoch1` | stap 653.000 | tak 1 |
| T2 | `<run>-epoch2` | stap 1.306.000 | tak 2 |
| S | `culturax-nl-230m-chunk1024` | `ckpt_keep/653000` | referentie voor vraag 1 |

`<run>` is `final-culturax-230m` bij variant (a), `final-cxgptnl-230m` bij variant (b); de hoofdrun (stabiele fase)
heet `<run>`. Alle runs in `gs://t5-dutch-english-usc2/mbert/<naam>/`.

## Het schema

Warmup 1.250 stappen lineair van 0 naar 4,8e-4 (stap voor stap die van S, `optax.warmup_cosine_decay_schedule`),
daarna constant 4,8e-4, daarna per tak een cooldown in de vorm 1-sqrt naar 1e-6
(`scripts/train_minimal.py`, `wsd_schedule`; `--lr-schedule wsd --decay-steps <n> --decay-shape 1-sqrt`). De
cooldown eindigt op `--max-steps`. Tijdens de cooldown is de rate 4,8e-4 − (4,8e-4 − 1e-6) · √f, met f het deel van
de cooldown dat voorbij is; na een kwart (f = 0,25) is hij precies halverwege.

| deel | run | stappen | LR | duur bij 0,235 s/stap |
|---|---|---|---|---|
| warmup | `<run>` | 0 – 1.250 | 0 → 4,8e-4 | 5 min |
| stabiel, eerste deel | `<run>` | 1.250 – 520.000 | 4,8e-4 | ~34 u |
| tak 1, cooldown | `<run>-epoch1`, vanaf `<run>` 520.000 | 520.000 – 653.000 (133.000) | 4,8e-4 → 1e-6 | ~8,7 u |
| stabiel, tweede deel | `<run>` | 520.000 – 1.040.000 | 4,8e-4 | ~34 u |
| tak 2, cooldown | `<run>-epoch2`, vanaf `<run>` 1.040.000 | 1.040.000 – 1.306.000 (266.000) | 4,8e-4 → 1e-6 | ~17,4 u |

Samen 1.439.000 stappen, ~94 uur rekentijd. Elke tak heeft een cooldown van 20,37% van zijn eigen stappen
(133.000/653.000 = 266.000/1.306.000). De hoofdrun traint op het schema van tak 2 (`--max-steps 1306000
--decay-steps 266000`) en stopt bij 1.040.000; tak 1 en tak 2 beginnen elk met een kopie van het checkpoint en de
datastand van de hoofdrun op hun aftakpunt (`start_branch.sh`), met Adams momenten en de stap van de hoofdrun, en
lezen daarna dezelfde batches die de hoofdrun daar las. Tot 520.000 zijn de schema's van hoofdrun en tak 1 bit voor
bit gelijk (test `test_branches_share_every_step_before_the_first_cooldown`). Volgorde: hoofdrun tot 520.000
(`final_run.sh`), dan tak 1, dan hoofdrun tot 1.040.000, dan tak 2; tak 1 mag ook op een andere v4-32 lopen terwijl
de hoofdrun doorgaat (HANDOVER-final.md).

### Keuze van vorm en lengte van de cooldown

Bron: Hägele, Bakouch, Kosson, Ben Allal, Von Werra, Jaggi (2024), "Scaling Laws and Compute-Optimal Training Beyond
Fixed Training Durations", arXiv:2405.18392 (v3, 17 okt 2024), https://arxiv.org/abs/2405.18392. Citaten zijn
letterlijk uit de HTML-versie; de rest is parafrase of eigen gevolgtrekking, zo gemarkeerd.

- Vorm. §3.2 definieert de cooldown (1-sqrt) als f(n, N, N_decay) = 1 − √((n − (N − N_decay)) / N_decay)
  (parafrase van de formule). Takeaway 2, citaat: "We find a cooldown form (1-sqrt) that consistently performs better
  than linear decay." §3.2, citaat: "the longer the training duration, the more linear cooldown is outperformed by the
  (1-sqrt) cooldown". **Keuze: 1-sqrt.** Afwijking van het artikel (eigen keuze): naar 1e-6 in plaats van naar 0,
  zodat het eindpunt gelijk is aan dat van S.
- Lengte. §3.2, citaat: "the benefits of extended cooldown periods plateau at around 20%", waarna de auteurs 20%
  gebruiken. Takeaway 3 (parafrase): een constante LR met een cooldown van minder dan 20% van de stappen bereikt
  hetzelfde eindverlies als cosine, en voor lange runs mag de cooldown korter zijn dan 20% zolang hij in absolute
  stappen lang genoeg is (Figuur 6: bij een run van 200k stappen evenaren 10k cooldown-stappen cosine bijna). §3.2,
  citaat: "using only 5% decay with the (1-sqrt) cooldown can nearly match the performance of the cosine schedule"
  (een run van 20B tokens). Tabel 6 (parafrase): een langere cooldown verbetert de metrieken niet altijd (5% → 20%).
- **Keuze: ~20%, precies 20,37%** (133.000 en 266.000 stappen), zodat de aftakpunten 520.000 en 1.040.000
  veelvouden van 10.000 zijn en gewone checkpoints van de hoofdrun (`--checkpoint-every 10000`). Waarom niet 10%
  (eigen afweging): met 10% (65.300 en 130.600 stappen) zijn het samen 1.371.300 stappen in plaats van 1.439.000,
  67.700 stappen of ~4,4 uur minder op ~94; vraag 1 vergelijkt met cosine, en de uitspraak dat WSD cosine evenaart is voor 20% het best gedekt; de
  bewijzen voor korter dan 20% gaan over het verlies van decoders, niet over fine-tunen van een encoder. Geen goede
  reden voor 10% gevonden.

## Wat verschilt tussen T1 en S

- Het LR-schema: WSD (hierboven) in plaats van cosine. Warmup, piek en eind-LR zijn gelijk.
- `--skip-nonfinite-updates`: `optax.apply_if_finite` om de hele optimizer (clipping en AdamW). Een stap met een
  niet-eindige gradiënt verandert parameters noch optimizerstand; bij eindige gradiënten rekent de optimizer
  hetzelfde. Gemeten op de CPU (`tests/test_final_run.py`): na één stap byte voor byte gelijk; na 30 stappen wijkt 1 van
  de 52 getallen 1 ulp af (de test laat ten hoogste 2 ulp toe), omdat optax de update binnen een `lax.cond` compileert en XLA die tak in de
  laatste bit anders kan afronden; Adams stapteller en dus de LR blijven gelijk. De trainer schrijft elke overgeslagen
  stap als regel `NONFINITE step=… skipped=… in_a_row=…` en `| Skipped: <n>` op de `Step`-regels, en stopt zichzelf na
  50 overgeslagen stappen op rij, met gezonde parameters. Een overgeslagen stap verhoogt Adams teller niet: het
  schema loopt dan in de cooldown die stappen achter op de stapteller (gerapporteerd, niet gecorrigeerd).
- De data: zie "Data". Variant (a): S's eigen bestanden, zonder de 45.781 documenten van `www.hebban.nl` (0,039%
  van de documenten, 35.580.456 tokens, 0,044%). Variant (b): 70% CulturaX + 30% GPT-NL. Bij (b) meet T1 tegen S de
  combinatie van WSD en GPT-NL; vraag 1 gaat dan over het recept, niet over het schema alleen.
- De trainercode: `final-run` (`0d92db4` plus de wijzigingen hierboven); S draaide op de code van zijn tijd
  (`code.tgz` in S's `scripts/`), zoals bij G, F en R tegen X.
- Namen, `--dataset-num-rows` (alleen voor de metriek `data/passes`), `--max-steps 1306000 --stop-at-step
  1040000` van de hoofdrun.

Gelijk aan S: model (230M, 22 lagen, hidden 1024), optimizer (AdamW, weight decay 0,01, `--layer-norm-lr-factor
1.0`, clipping 1,0), piek-LR en warmup, bf16, 4 hosts × 32 rijen × 1024 tokens per stap, packing met opgeknipte lange
documenten, 15% losse tokens, seeds (init 42, masking 42 + proces, data 42 + pass), `--checkpoint-every 10000`, de
held-out-eval (`~/heldout.parquet`, elke 25.000 stappen, 50 batches). Het runscript van `start_final.sh culturax
stable` verschilt van `run_culturax-nl-230m-chunk1024.sh` alleen in: code-map en venv-pad, `wsd --decay-steps 266000
--decay-shape 1-sqrt` voor `cosine`, `--dataset-num-rows 117346885`, `--max-steps 1306000 --stop-at-step 1040000`,
`--skip-nonfinite-updates --url-blocklist hebban.nl`, en de namen (gecontroleerd 10 nov met `PRINT_RUN_SCRIPT=1`).

## Wat verschilt tussen T2 en T1

Tot stap 520.000 is het dezelfde run. Daarna: T1 koelt af tot 653.000; T2 traint door op 4,8e-4 tot 1.040.000 en
koelt af tot 1.306.000. T2 ziet dus elk document van zijn host twee keer (de tweede pass in een andere volgorde,
hieronder), en heeft een twee keer zo lange cooldown in stappen.

## Data

### Variant (a): alleen CulturaX nl

`uonlp/CulturaX`, config nl, revisie `6a8734bc69fefcbb7735f4f9250f43e4cd7a442e`: de staging van S op mbert-corpus
(`~/culturax_nl/nl/nl_part_*.parquet`, de originele HF-bestanden met de kolommen text, timestamp, url en source;
per worker S's 32 bestanden, `CULTURAX_FILES` in `gptnl_mix.py`), niet opnieuw gestaged. De trainer laat met
`--url-blocklist hebban.nl` elk document weg waarvan de host van de url `hebban.nl` is of op `.hebban.nl` eindigt
(`src/mmbjax/training/url_blocklist.py`), vóór de shuffle: de documentstroom is die van bestanden zonder die
documenten. De scan van 9 nov 23:36 (`blocklist_scan.py`, dezelfde functie, op alle 128 bestanden; `prepare_final.sh
check` controleerde 10 nov dat de gestagede bestanden die van de scan zijn): 

| worker | documenten | weg (alle `www.hebban.nl`) |
|---|---|---|
| 0 | 29.348.172 | 11.313 |
| 1 | 29.348.135 | 11.461 |
| 2 | 29.348.224 | 11.597 |
| 3 | 29.348.135 | 11.410 |
| samen | 117.392.666 | 45.781 (35.580.456 tokens) |

Andere hosts dan `www.hebban.nl` kwamen niet voor. Voorbeelden: `/recensie/…`, `/recensies/…`, `/boeken/…`,
`/artikelen/…`. Eén pass duurt na de blocklist ≈ 652.620 stappen (S: 652.905 volgens de tokentelling); T1 eindigt
daardoor ~380 stappen in pass 2, T2 ~760 stappen in pass 3 (elk ~0,06%), zoals S 95 stappen in pass 2 eindigde. De
held-out-documenten (de eerste 40.000 van `nl_part_00097`) zijn in (a) trainingsdata, zoals bij S; zes ervan zijn
pagina's van `www.hebban.nl` en vallen nu weg.

### Variant (b): 70% CulturaX nl + 30% GPT-NL

Als bij G (`experiments/gptnl/protocol.md`): GPT-NL Public Corpus revisie `117b13ee7246aa5c83a42ff77e15d4d50b4156b9`,
dezelfde 20 collecties, alleen `language == nl`, zonder lege documenten, herhalingen en teksten die ook in CulturaX
staan, CulturaX zonder de held-out-documenten; dezelfde trekkingen met seed 42. Anders dan bij G: het plan is voor
653.000 stappen (`--steps 653000`, per host 20.242.038.784 tokens), 128 bestanden per host, en de 45.781 rijen van
`www.hebban.nl` worden net als de held-out-rijen overgeslagen (`--exclude-culturax`). GPT-NL heeft geen url en zijn
collecties zijn geen webcrawls (common-crawl is uitgesloten): de blocklist geldt daar niet. De gestagede bestanden
hebben alleen `text`, dus de trainer krijgt geen `--url-blocklist`.

| host | GPT-NL documenten / tokens | CulturaX documenten / tokens | GPT-NL | stappen per pass bij vulling 0,9484 |
|---|---|---|---|---|
| 0 | 2.690.681 / 6.072.621.682 | 20.611.137 / 14.169.452.079 | 30,0% | 653.188 |
| 1 | 2.685.054 / 6.072.612.985 | 20.614.525 / 14.169.430.710 | 30,0% | 653.187 |
| 2 | 2.667.186 / 6.072.611.737 | 20.599.696 / 14.169.427.527 | 30,0% | 653.186 |
| 3 | 2.691.644 / 6.072.612.421 | 20.615.956 / 14.169.429.412 | 30,0% | 653.188 |

De vulling 0,9484 is die van G (tokens per stap uit `Tok/s × Step` van G's log, stap 259.600–279.550; CulturaX alleen:
0,9464). Eén pass duurt dus ≈ 653.190 stappen: T1 eindigt net vóór het eind van pass 1, T2 net vóór het eind van
pass 2.

**Herhaling van GPT-NL in (b).** Getrokken worden 10.734.565 van de 14.353.689 toegelaten GPT-NL-documenten,
24.290.458.825 van de 32.508.169.748 toegelaten tokens (74,72%).

- Over 1 epoch (T1): elk getrokken GPT-NL-document 1×, geen herhaling; 0,75× de toegelaten GPT-NL-tokens.
- Over 2 epochs (T2): elk getrokken GPT-NL-document 2× (pass 2 in een andere volgorde); 1,49× de toegelaten
  GPT-NL-tokens, waarvan 25,3% nooit gezien.
- CulturaX in (b): 82.441.314 van 117.392.666 documenten (70,2%), 1× resp. 2×. In (a) elk CulturaX-document 1×
  resp. 2×.

Staging van (b): `prepare_final.sh mix` per worker (leest van Hugging Face met range requests, spill in
`/dev/shm/final_work`, 128 bestanden in `~/final_cxgptnl`, ~27 GB per host, ~25 min geschat uit G's 12 min voor
13,5 GB). Daarvoor moet `~/culturax_nl` (S's staging, 49 GB) weg: de hoofdsessie verwijdert die met de hand op alle
workers; de keten stopt tot dat gebeurd is.

## Epochs: volgorde en hervatting

Hoe de trainer over een epoch heen gaat (onderzocht 9–10 nov in `src/mmbjax/training/packing.py`,
`EpochTrackingDataLoader`): elke host leest zijn eigen bestanden; is zijn pass op, dan bouwt hij zelf een nieuwe
pijplijn met seed 42 + pass voor (1) de volgorde van zijn bestanden en de shuffle-buffer van 10.000 documenten
(`IterableDataset.shuffle`), en (2) het venster van 5.000 stukken vóór de packing. De hosts beginnen hun volgende
pass op verschillende stappen (zo liep mC4 2,21 epochs; `test_shares_of_other_sizes_…`). De masking trekt uit de
globale NumPy-RNG, eenmaal geseed met 42 + proces, en loopt over de passen door. Alleen het schudden van de bins in
Grain's packing heeft een vaste seed (0); de rijen die erin komen verschillen per pass. Een datastand bewaart pass en
Grain-stand; een hervatting bouwt de pijplijn van die pass met zijn seed opnieuw en leest tot de index; een stand op
de laatste batch van een pass gaat verder met de volgende pass (`65fa609`). Nieuw: elke pass logt
`Pass <n> of shard <h>/<H> with seed <42+n-1>` en bij het eind het aantal weggelaten documenten.

Aangetoond in tests (branch `final-run`):
- `tests/test_epoch_shuffle.py`: met een buffer klein tegen een bestand, zoals in de echte run, heeft pass 2 seed
  43 en pass 3 seed 44, herhaalt geen batch van pass 2 een batch van pass 1 (en pass 3 geen van pass 2), staan
  minder dan 5% van de documenten op dezelfde plek en minder dan 5% naast dezelfde buur; elke pass leest elk
  toegelaten document één keer en geen enkel geblokkeerd; een datastand na de laatste batch van pass 1 en een midden
  in pass 2 geven in een nieuwe loader exact de volgende batches, tot in pass 3.
- `tests/test_multihost.py::test_final_run_resumes_exactly_in_its_second_pass_…`: de echte trainer met de opties
  van de eindrun op twee JAX-processen, gestopt midden in pass 2 van host 0 en hervat: dezelfde batches als de run
  zonder onderbreking, seed 43 in pass 2, de geblokkeerde documenten per pass gelogd.
- `test_a_cooldown_branch_continues_the_main_run_from_a_copy_of_its_checkpoint`: een tak vanaf een kopie van een
  checkpoint van de hoofdrun leest de batches die de hoofdrun daarna las, en zijn optimizer telt door.

## Checkpoints en bewaking

Elke 10.000 stappen een checkpoint (de trainer houdt de laatste 3). De keten kopieert naar `ckpt_keep/`: van de
hoofdrun elke 100.000e stap plus 520.000 en 1.040.000; van tak 1 600.000 en 653.000; van tak 2 1.100.000,
1.200.000, 1.300.000 en 1.306.000. Datastanden (`ckpt_data_state/<stap>/`) worden niet weggegooid. NaN-bewaking op
co1 (elke 180 s): staan in de laatste twee `Step`-regels (50 stappen uit elkaar) van een worker allebei `Loss:
nan`, dan stopt de keten de trainer met `pkill -9` (ook `verify_ckpt.py` en `convert_gcs_checkpoint.py`) en meldt
FAILED; één niet-eindige stap wordt door `apply_if_finite` al overgeslagen en is geen reden om te stoppen.

## Smoke-test vóór de start

200 stappen met het commando van de hoofdrun als `<run>-smoke` (eval op stap 200), vergeleken met S
(`experiments/final/s_reference.json`, uit S's Trackio-database): het verlies op 100, 150 en 200 binnen 10% van S's
(7,4894 / 6,8017 / 6,2256), de mediane staptijd op die stappen ten hoogste 1,25 × S's mediaan van 110–300 (0,2095
s), geen overgeslagen stap, bij (a) de regel van de blocklist. De bucketmap van de smoke-test blijft
staan.

## Evaluatie

Primair, per arm (T1, T2, en S opnieuw op dezelfde manier):

- DBRD met `--max-length 1024` en SICK-NL met `--max-length 512`, `scripts/dumb/finetune.py`, fine-tune-LR 5e-5,
  verder het vaste protocol: 3 epochs, AdamW met weight decay 0,01, constante LR, update 192 (DBRD) en 256
  (SICK-NL), seeds 42, 43, 44, `test_accuracy` na epoch 3. DBRD@1024 met 5e-5 is voor S nog niet gemeten (LEARNINGS:
  "de combinatie is niet gemeten"); S wordt daarom voor beide taken opnieuw gefinetuned, met hetzelfde script en
  dezelfde hardware als T1 en T2. Uitslagen: `gs://t5-dutch-english-usc2/mbert/final/dumb/<naam>--{dbrd1024,sicknl}--seed{42,43,44}.json`,
  met naam `epoch1-653000` (T1), `epoch2-1306000` (T2) en `s-653000` (S). n = 2.112 voor DBRD@1024 (k = 2.026 geeft
  precies de 0,959280… van een bestaande uitslag) en 4.864 voor SICK-NL.

Extra evals, zoals `experiments/dumb-extra/protocol.md` (branch `dumb-extra`), per arm:

- SQuAD-NL v1.1 (F1), CoNLL-2002 NER (entity-F1), POS LassySmall en POS Alpino (UPOS-accuratesse): fine-tune-LR
  5e-5, update 32, lineaire decay zonder warmup, 3 epochs, seeds 42, 43, 44, test na de laatste epoch.
- BLiMP-NL: zero-shot, PLL-word-l2r, 8.400 paren, één getal per model.

Geen DBRD- of SICK-NL-data (en geen data van de extra evals) in de training: DBRD komt van `hebban.nl`, dat de
blocklist weglaat; SICK-NL is een vertaling van beschrijvingen bij beelden en video's, en zit in geen enkele
opgenomen GPT-NL-collectie (G's protocol). CoNLL-2002-getallen niet publiceren zonder Yeb.

## Beslisregel

Per arm en primaire taak: het gemiddelde van `test_accuracy` over de seeds 42, 43, 44 en de seed-range (max − min).
Alles exact, met breuken k/n (`fractions.Fraction`; k = round(test_accuracy × n), dat `test_accuracy` precies terug
moet geven, anders FOUT), zodat geen afronding beslist. De drempel per taak is de grootste van de twee seed-ranges.
"Meer dan" is strikt.

Vergelijking van A (kandidaat) met B (referentie):

1. "**schaadt**" als A's gemiddelde meer dan de drempel onder B's ligt op DBRD@1024 of op SICK-NL. Eerst getoetst.
2. Anders "**helpt**" als A's gemiddelde meer dan de drempel boven B's ligt op SICK-NL.
3. Anders "**geen verschil**". Winst op DBRD@1024 alleen is "geen verschil".

Toetsen voor de extra evals (zoals de regel voor GPT-NL in LEARNINGS "Open"): SQuAD-NL F1, CoNLL-2002 F1, POS
LassySmall en POS Alpino zijn "slechter" (of "beter") als het verschil van de gemiddelden groter is dan de grootste
seed-range van de twee; POS exact op woordaantallen (k/n), F1 exact op de opgeslagen getallen (`Fraction` van de
float). BLiMP-NL is "slechter" bij meer dan 0,5 punt lager (meer dan 42 van de 8.400 paren) en "beter" bij meer dan
0,5 punt hoger.

**Vraag 1** (A = T1, B = S): de uitkomst van 1–3 en de extra evals worden gerapporteerd. Ze beslissen niet over het
eindmodel (S is geen kandidaat), wel over het schema van een latere run: bij "schaadt" is WSD hier slechter dan
cosine.

**Vraag 2 en 3** (A = T2, B = T1):

1. T2 "schaadt" → eindmodel **T1**.
2. T2 "helpt" → **T2**, tenzij een extra eval van T2 met de toetsen hierboven "slechter" is dan T1; dan T1.
3. "geen verschil" → **T2** alleen als minstens één extra eval "beter" is en geen enkele "slechter"; anders **T1**
   (een tweede pass over dezelfde data zonder gemeten winst; T1 is ook het model dat met S vergeleken is).

De held-out-eval (`EVAL`-regels) beslist niet mee: in (a) zijn de held-out-documenten trainingsdata, en hij
bevoordeelt 15%-masking en CulturaX.

## Amendementen

(geen)
