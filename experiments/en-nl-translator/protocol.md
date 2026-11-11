# Protocol: fine-tune van de en→nl-vertaler, armen N en B op TranslateGemma 27B

Status: **vastgelegd 10 nov 2026**, vóór de eerste trainingsstap op 27B (ontwerp §4: "vóór de fine-tune vast te leggen
in `protocol.md`"), met Yebs akkoord van 10 nov ~10:15 (§8). Geschreven door subagents voor de hoofdsessie en Yeb;
bijgewerkt 10 nov ~10:30 met Yebs besluiten van ~09:00 en ~09:20, en bij het vastleggen met de besluiten van ~10:15 en de
aanvulling van de hoofdsessie bij §4.2.2 (in-domein). Wat na het vastleggen verandert, staat als gedateerd amendement in
§9. Branch `en-nl-sft` van `github.com/yhavinga/mbert-jax`.

Bronnen: hammond `mbert/designs/en-nl-translator.md` (hierna "ontwerp"), §3 stap 3, §4, §7 punt 6–8 en §9;
hammond `mbert/LEARNINGS.md`, de vertaler-items; `experiments/en-nl-translator/HANDOVER.md` (smoke-test, hoofdrun,
aanvullende run); `mbert-ops/HANDOVER-baseline.md` en `mbert-ops/translator-eval/baseline-tg27b.md` (uitgangspunt).

Labels: **[besluit]** = besluit van Yeb (met datum); **[ontwerp]** = staat zo in het ontwerp; **[gemeten]** = op
10 nov gemeten op de data (`sft_build.py`, rapport in `gs://t5-dutch-english-usc2/mbert/translator/sft/data/arms-v2/report.md`;
de sets van vóór de besluiten van 10 nov in `…/sft/data/arms/report.md`);
**[keuze]** = keuze van dit protocol (technisch, door de subagent die het schreef); **[aanvulling]** = toegevoegd door de
hoofdsessie vóór de training.

`R=gs://t5-dutch-english-usc2/mbert/translator`, `S=$R/sft`, `E=$R/eval`.

## 1. Vraag en armen

Verliest de vertaler na de fine-tune betekenis, en schrijft hij natuurlijker Nederlands? [ontwerp §4]

| arm | trainingsdata | bij het vertalen |
|---|---|---|
| N | alle BT-paren zonder tag + 20% FLAN-nl-replay mét een vaste instructiezin | TranslateGemma's eigen prompt, niets erbij |
| B | dezelfde BT-paren zonder tag, zonder replay | TranslateGemma's eigen prompt |

- **De tag is omgedraaid** [besluit 10 nov ~09:00]: natuurlijk Nederlands is de standaard. De BT-paren (synthetisch
  Engels → origineel Nederlands) krijgen geen tag; de FLAN-nl-replay (echt Engels → het Nederlands van TranslateGemma
  27B) krijgt een vaste instructiezin die de gewone, letterlijke vertaalmodus markeert (§2.4). Bij gebruik zet je niets
  aan. Tot 10 nov: BT-paren met tag, replay zonder (sets `$S/data/arms/`, de tagproef van §6).
- Het andere deel van N is de FLAN-nl-replay [besluit 9 nov, ontwerp §7 punt 6]; echte vertaalparen (Europarl)
  niet, zolang er geen scan met Gemma 4 31B is gedaan.
- C (controle-arm) alleen bij rekentijd over [besluit 9 nov]; niet in dit protocol.
- Basismodel en uitgangspunt: TranslateGemma 27B zonder fine-tune (`$E/baseline-tg27b/`, 10 nov) [gemeten].

## 2. Trainingsdata: een query op de pairs-tabellen

Code: `experiments/en-nl-translator/sft_build.py` (`build`, `check`); tests `tests/test_en_nl_sft_build.py`.
Invoer: `$R/mainset/run/pairs/pairs.parquet` (240.633 paren: 202.993 passages en 37.640 contextbeurten) en
`$R/mainset/run-long/pairs/pairs.parquet` (31.077 passages van 385–511 tokens), met de selecties ernaast
(`…/select/passages.parquet`: bronscore, herkomst, contextbeurten, spreker; `context-turns.parquet`). Er gaat geen
paar weg uit de tabellen; een set is het resultaat van de query [besluit 9 nov, "alles bewaren"].

### 2.1 De query (standaard) [besluit 9 nov: streng op de bron, mild op het paar]

Een BT-paar (passage) komt in aanmerking als:

1. `hard_fails` leeg is: niet afgelopen, `decode_loop`, Nederlandse resten in het Engels, lengteverhouding > 1,5,
   structuurverschil (inclusief `newline_doubled`) (`pairs.py hard_fails`);
2. de bron door de selectie kwam: Gemma-bronscore ≥ 0,5, bij FineWeb-2 ook herkomst ≥ 0,5. Dat geldt al voor elke
   passage in de selectie; de query controleert het en het kost 0 paren [gemeten];
3. MetricX-24-QE en→nl ≤ **7,5**: een ruime grens tegen uitschieters, net boven p99 (7,416) van de trainpassages die
   door de harde controles komen [besluit 10 nov ~10:15: de grens blijft 7,5, tegen uitschieters]; geen grens op
   chrF en geen Gemma-paarscore, geen zinnen- of getallenfilter [besluit 9 nov]. De grens is een parameter
   (`sft_build.py --qe-max`): een andere grens geeft met één commando een nieuwe, gecontroleerde set,
   `QE_MAX=<x> bash mbert-ops/scripts/translator_sft_arms.sh` → `$S/data/arms-v2-qe<x>/` (startregel in de kop van
   het script). Voorbeelden per QE-band en de kosten van 6, 7, 7,5, 8, 9 en 10:
   `mbert-ops/translator-sft/examples-qe-cutoff.md` (`$S/examples/`);
4. het geen 13-gram (woorden, kleine letters, als `select_passages.ngrams`) deelt met een evaluatieset: bron en
   referentie van WMT24++, FLORES-200 devtest, NTREX-128 en Europarl-direct (`$E/data/evalsets.parquet`) [keuze; de
   selectie blokkeerde al op de Nederlandse kant van WMT24++, FLORES en NTREX, niet op de Engelse kant en niet op
   Europarl-direct];
5. het hele voorbeeld (prompt in Gemma 3's beurt, Nederlands, `<end_of_turn>\n`) in **2.048 tokens** past, in elke
   prompt van de build (ontwerp §7 punt 7: de fine-tune draait met sequenties van 2.048).

Train en eval volgen de splitsing van de selectie (per document) [ontwerp §3 stap 0].

**Wat elke grens kost** [gemeten, 10 nov]:

| grens | train: valt af (eerste grens die faalt) | eval |
|---|---:|---:|
| passages vóór de query | 229.472 | 4.598 |
| 1. `hard_fails` niet leeg | 2.113 | 37 |
| 2. bronscore < 0,5 / herkomst < 0,5 (FineWeb-2) | 0 / 0 | 0 / 0 |
| 3. MetricX-QE > 7,5 | 2.117 (alleen deze grens: 2.177) | 42 |
| 4. 13-gram met een evaluatieset | 0 | 0 |
| 5. langer dan 2.048 tokens | 0 (wel 261 contexten weg, zie 2.2) | 0 (9 contexten) |
| **komt in aanmerking** | **225.242** | **4.519** |

Per bron in aanmerking (train): Handelingen 88.040, raad 46.227, FineWeb-2 57.556, dpc 17.139, rechtspraak 16.280.

Wat de QE-grens zou kosten bij andere waarden (trainpassages die door de harde controles komen, n = 227.359;
p50 2,139, p90 4,723, p95 5,572, p99 7,416, p99,5 8,332, p99,9 10,584, max 19,71):

| QE > | 6 | 7 | **7,5** | 8 | 9 | 10 | 12 | 15 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| paren | 7.902 | 3.309 | **2.117** | 1.441 | 662 | 326 | 88 | 11 |

Bij 7,5 vallen ook 480 contextbeurten af (die door de andere grenzen komen), en verliezen 196 contextvoorbeelden
daardoor hun hele context (zij blijven als gewoon voorbeeld) [gemeten, `examples-qe-cutoff.md`].

### 2.2 Contextvoorbeelden [besluit 9 nov, ontwerp §9]

- Een trainpassage die de selectie als contextvoorbeeld markeerde (Handelingen, 1–3 voorgaande spreekbeurten,
  samen ≤ 1.024 tokens), houdt haar beurten, van de dichtstbijzijnde terug, zolang elke beurt door dezelfde query
  komt (harde controles, QE ≤ 7,5, 13-gram, bronscore ≥ 0,5); de eerste die faalt, beëindigt de context. Zonder
  beurt over is het een gewoon voorbeeld [keuze].
- Elke beurt staat in de context als (synthetisch Engels, origineel Nederlands) met het sprekerslabel van de
  Handelingen ("De heer Korthals (VVD)", "De voorzitter"); de huidige beurt krijgt het label van haar spreker.
  Verlies alleen op de huidige beurt.
- **Omvang en aandeel: alle 225.242 trainpassages die door de query komen, in N en B gelijk** [besluit 10 nov
  ~09:00] (`sft_build.py --bt all`). Van de 15.903 gemarkeerde trainpassages die door de query komen, hielden 15.583
  minstens één beurt (320 geen; 1.304 beurten vielen weg op de query), en bij 256 maakte de context het
  voorbeeld langer dan 2.048 tokens (zij blijven als gewoon voorbeeld): **15.327 trainvoorbeelden met context, 6,8%**
  van de BT-voorbeelden [gemeten]. Elke bron houdt vanzelf haar aandeel.
- Tot 10 nov (sets `$S/data/arms/`): 20% context, 76.610 BT-voorbeelden (15.322 met context, 61.288 gewone, per
  bron getrokken), ~34% van de bruikbare passages. Daar telde de tokengrens over drie prompts (zonder tag, token,
  instructie), dus viel bij 261 passages de context weg; nu alleen over de prompt zonder tag.

### 2.3 FLAN-nl-replay (alleen arm N) [besluit 9 nov]

- Bron: `$R/mainset/replay/pairs.parquet`, 521.329 segmentparen (Engels FLAN-segment → vertaling door TranslateGemma
  27B), al gefilterd op `ended`, `decode_loop`, `english_left`, structuur en 48–384 Nederlandse tokens, en zonder
  de segmenten van FLAN's 1.035 evaluatierijen.
- De query sluit daarnaast paren uit die een 13-gram delen met een evaluatieset (zoals 2.1 punt 4) [keuze]:
  0 paren [gemeten].
- FLAN's evaluatierijen blijven buiten de replay via hun segmenten (ontwerp §7 punt 6), niet via 13-grammen. Een
  13-gramblokkade op die rijen nam 131.273 van de 521.329 replayparen weg, omdat de vaste zinnen van de FLAN-templates
  in een kwart van de replay terugkomen: dan verdwijnen hele taken [gemeten]. Het rapport telt de overlap
  (131.273 van de 521.329 replayparen delen een 13-gram met een evaluatierij; niet weggehaald). Een passage die onder een ander template ook in de replay zit, kan dus in de training komen; dat
  maakt de uitvoer van N op die rij eerder gelijker aan het basismodel, dus werkt tegen N bij (b) en (c) [keuze].
- **Aandeel: 20% van de voorbeelden van N** [besluit 10 nov ~09:00], dus replay = BT × 0,25 = **56.310**,
  willekeurig getrokken (seed `en-nl-sft/0/replay`) uit de 521.329. Zonder context, met de instructiezin van §2.4
  (`--replay-tag conventional`). Tot 10 nov: 19.152, zonder tag.

### 2.4 Prompt en tag [besluit 10 nov ~09:00: de tag omgedraaid]

- **BT-voorbeelden, beide armen: zonder tag.** Zonder context `chat-v1` zonder context: woord voor woord TranslateGemma's
  eigen prompt (`translate.user_text`; `sft_build.py check` toetst de token-id's tegen TranslateGemma's eigen
  chat-template). Met context `chat-v1` met context (`<earlier_turns>`, `<current_turn speaker="…">`) [besluit 9 nov,
  ontwerp §9: eigen prompt alleen voor de voorbeelden met context].
- **FLAN-nl-replay, alleen N: met de instructiezin** "Translate closely, as a conventional translation." (Yebs
  voorbeeld, 10 nov), in de instructie na "… or commentary." en vóór "Please translate …", dus op de plek van de
  instructiezin uit de proef van §6; de bron blijft zoals hij is. Preset `chat-v1-tag-conventional` in
  `experiments/flan-nl/translate_models.yml` (`sft_build.py`: `CONVENTIONAL_SENTENCE`, `--replay-tag
  conventional`). `sft_build.py check` eist de zin precies één keer in elk replayvoorbeeld en nergens anders.
- **Bij gebruik zet je niets aan**: beide armen vertalen met TranslateGemma's eigen prompt (§4.1). De bedoeling
  [besluit 10 nov]: zonder zin vertaalt het model in de modus van de BT-paren, natuurlijk Nederlands; of dat zo
  uitpakt, meet §4.
- Achtergrond: de proef van §6 vergeleek twee vormen van een tag op de BT-paren (token `<unused0>`, of de zin
  "Translate into natural, original-sounding Dutch."); geen van beide lekte. Met de omgedraaide tag hoeft bij gebruik
  niets mee en kan er dus ook niets lekken [eigen gevolgtrekking]. De presets van de proef (`chat-v1-tag-token`,
  `chat-v1-tag-instruction`) blijven bestaan.

### 2.5 De sets [gemeten, 10 nov; `$S/data/arms-v2/`]

| set | split | voorbeelden | BT zonder context | BT met context | replay | lang (385–511) | tokens | doeltokens | max |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B | train | 225.242 | 209.915 | 15.327 | 0 | 29.329 | 109.010.933 | 45.364.520 | 2.048 |
| N | train | 281.552 | 209.915 | 15.327 | 56.310 | 29.329 | 126.915.299 | 52.982.459 | 2.048 |
| B en N | eval | 4.519 | 4.223 | 296 | 0 | 602 | 2.189.542 | 917.827 | 2.006 |

Tokens per bron (train; tokens = prompt + doel, doeltokens = het Nederlands + `<end_of_turn>\n`, het verlies):

| bron | voorbeelden | aandeel BT | tokens | doeltokens |
|---|---:|---:|---:|---:|
| Handelingen | 88.040 | 39,1% | 49.995.466 | 18.092.396 |
| FineWeb-2 | 57.556 | 25,6% | 24.649.223 | 11.202.852 |
| raad | 46.227 | 20,5% | 19.759.235 | 9.219.202 |
| dpc | 17.139 | 7,6% | 7.030.210 | 3.217.967 |
| rechtspraak | 16.280 | 7,2% | 7.576.799 | 3.632.103 |
| replay (alleen N) | 56.310 | – | 17.904.366 | 7.617.939 |

- De bronverdeling is die van de passages die door de query komen, dicht bij 40/20/7,5/7,5/25 van het ontwerp.
- B en N hebben dezelfde BT-voorbeelden met dezelfde prompt; N heeft daarnaast de replay met de instructiezin. De
  eval-sets zijn alle eval-passages die door de query komen (met context waar die er is), zonder replay, in beide
  sets gelijk.
- Gecontroleerd met `sft_build.py check` (MaxText's eigen SFT-stappen op elk voorbeeld): B en N beide EXIT 0 (10 nov 09:47 en 09:49): train 225.242 en 281.552 voorbeelden, 109.010.933 en 126.915.299 tokens, max 2.048; eval 4.519, max 2.006 (`$S/data/arms-v2/check.txt`).
- Gebouwd met `en-nl-sft` `e8979e1` door `mbert-ops/scripts/translator_sft_arms.sh` (tag `11100817`); rapport
  `$S/data/arms-v2/report.md` (kopie `mbert-ops/translator-sft/data-arms-v2-report.md`), `check.txt` ernaast.
- De sets van vóór de besluiten van 10 nov (`$S/data/arms/`: B 76.610, N-token en N-instruction 95.762) blijven
  staan, maar worden niet gebruikt.

Bestanden per set in `$S/data/arms-v2/<set>/`: `train.parquet` en `eval.parquet` (kolom `messages`, voor MaxText; train
geschud met seed `en-nl-sft/0/shuffle`), `train-meta.parquet` en `eval-meta.parquet` (per rij id, soort, bron, tag,
prompt, Engels, context, label, tokens, QE, chrF), `template.jinja` (de gebruikersbeurt zoals gebouwd in Gemma 3's
beurt) en `manifest.json` (query, aantallen, sha256). `report.md`/`report.json` in `$S/data/arms-v2/`. `sft_build.py
check` draait MaxText's eigen SFT-stappen (chat-template, tokenisering, maskering van de prompt) op elk voorbeeld en
vergelijkt de id's met `translate.py`'s prompt.

## 3. Training

[ontwerp §3 stap 3; instellingen van de smoke-test van 9 nov, `run_sft.sh` en `sft_lora.py`]

- MaxText 0.2.5 `train_sft` met Tunix (`a8d70582`) en qwix-LoRA, venv `setup_sft.sh`.
- Model: TranslateGemma 27B, `gs://t5-dutch-english-usc2/mbert/flan-nl/translategemma-27b-maxtext` (ongescand),
  gewichten in fp32, als één multi-host-job over de 16 chips van mbert-mix (`MULTIHOST=1`, FSDP).
- LoRA (naar EnSiTa): rank 16, α 16, op attentie (`query`, `key`, `value`, `out`) en MLP (`wi_0`, `wi_1`, `wo`) van
  alle lagen; embeddings bevroren; LoRA-dropout 0 (MaxText geeft qwix 0,0).
- Optimalisatie: AdamW (β2 0,999, weight decay 0,01), LR 1e-5, cosine naar 0 met 10% warmup, clipping 1,0, één
  epoch, `num_vocab_tiling=8`. Het aantal stappen telt `sft_lora.py` eerst: elk proces telt de batches van zijn
  eigen deel van de data, en alle processen nemen het kleinste aantal min een marge van max(8, 1%) batches
  (`step_limit`; in de 12B-proef 431, 433, 438 en 437 → 423 stappen, ~3% van de batches ongebruikt). De marge is
  nodig: Tunix laadt de volgende batch vóór de controle op `max_steps`, dus bij precies het kleinste aantal liep het
  proces met dat aantal leeg terwijl de andere nog een batch over de 16 chips verdeelden, en de job brak na de
  laatste stap af (10 nov) [gemeten]. Een opgegeven `STEPS` boven die grens weigert `sft_lora.py` in multi-host.
- Data: packing, sequenties van **2.048** tokens (de smoke-test gebruikte 1.024), 1 sequentie per chip
  (`per_device_batch_size=1`, 16 × 2.048 = 32.768 tokens per stap; de smoke-test: 2 × 1.024), geen shuffling in
  MaxText (de build schudt), verlies alleen op het Nederlands en `<end_of_turn>` (`sft_train_on_completion_only`).
- Duur [schatting, 10 nov, met alle paren]: bij het tempo van de smoke-test (4.018 tokens/s over 16 chips,
  2 × 1.024 per chip) B 109,0 M tokens ≈ 7,5 uur, N 126,9 M tokens ≈ 8,8 uur, plus ~15 min laden, tellen en
  compileren; samen ~16,5 uur. Bij 32.768 tokenplaatsen per stap is dat bij volledig gevulde sequenties ~3.330
  (B) en ~3.870 (N) stappen. Bij 2.048 per sequentie niet gemeten. Eerst een korte run (`STEPS=8`) om geheugen en
  tempo bij 2.048 te meten [keuze].
- Uitvoer: `$S/arms/runs/<naam>/` (`adapter.npz`, `adapter.json`, `summary.json`, logs); de adapter van elke arm
  daarna ook in `$S/arms/B/` en `$S/arms/N/` (`adapter.npz`, `adapter.json`, `summary.json`).

## 4. Evaluatie

### 4.1 Betekenis (beslist, regel (a))

De keten van `en-nl-eval` (`run_baseline.sh`: `translate`, `score`, `summary`; met `ADAPTER`, `PROMPT`, `LABEL` en een
eigen `B` per arm), op dezelfde sets en met dezelfde code als het uitgangspunt:

| set | n | rol |
|---|---:|---|
| WMT24++ en→nl_NL | 960 zonder `is_bad_source`, zoals het artikel [besluit 10 nov ~09:00]; alle 998 gerapporteerd | beslist (a) |
| FLORES-200 devtest eng→nld | 1.012 | beslist (a) |
| NTREX-128 newstest2019 eng→nld | 1.997 | beslist (a) |
| Europarl-direct (nl-origineel als referentie) | 640 | gerapporteerd |

- Maten: MetricX-24 hybrid XXL in QE-modus (**beslist**), MetricX-24 met referentie en COMET-22 met referentie
  (gerapporteerd). MetricX bevoordeelt TranslateGemma (RL op MetricX-QE); COMET-22 staat er daarom naast [ontwerp].
  CometKiwi en xCOMET niet (CC BY-NC-SA).
- Decodering zoals het uitgangspunt: greedy, `translate.py`, lengteklassen, MaxText 0.2.5; de adapter samengevoegd
  in de kernels (`--lora`). Beide armen met TranslateGemma's eigen prompt, zonder instructiezin (geen `PROMPT`)
  [besluit 10 nov ~09:00, §2.4].
- Ook geteld: niet af, `decode_loop`, Engels over, leeg (zoals het uitgangspunt: daar alle 0).
- **N met de instructiezin, als controle van de modus** [besluit 10 nov ~10:15, keuze van de hoofdsessie bij het
  vroegere open punt 5 van §8]: arm N vertaalt ook mét "Translate closely, as a conventional translation." (preset `chat-v1-tag-conventional`)
  de WMT24++-bron (998, de 960 apart) en de 1.000 FLAN-evaluatierijen van §4.2.1. Gerapporteerd naast N zonder zin
  (MetricX-24 ref en QE en COMET-22 op WMT24++, het aandeel "origineel" van §4.2.1, PLL en MTLD van §4.2.3); het
  beslist niet mee.

### 4.2 Natuurlijkheid (beslist, regel (b) en (c))

1. **Classifier "origineel of vertaald"** [ontwerp §4.2, regel (b)].
   - Model: onze Dutch ModernBERT S @ 653k (`gs://t5-dutch-english-usc2/mbert/culturax-nl-230m-chunk1024/ckpt_keep/653000`)
     [besluit 10 nov ~09:00], binaire classificatie met de fine-tune-code van DUMB (`finetune.py`), LR 5e-5
     (LEARNINGS: elk model beter bij 5e-5), 3 epochs, `--max-length 1024`, seed 42 [keuze].
   - Data: 2.000 originelen uit de eval-kant van de selectie (BT-eval-passages die door de query komen, ander
     document dan elke trainpassage) tegen hun terugvertaling door het basismodel (kolom `back`, dezelfde inhoud
     in het Nederlands van TranslateGemma 27B). 90/10 gesplitst per passage (de twee versies van een passage aan
     dezelfde kant). Keuze van de 2.000: de eval-passages zonder context, geordend op sha256 van `pid` met
     `en-nl-sft/classifier`, de eerste 2.000 [keuze].
   - Uitvoering [keuze, vastgelegd vóór de training]: de eval-passages zonder context zijn de rijen van soort `bt`
     in `$S/data/arms-v2/B/eval-meta.parquet`; hun `pid`, `nl` en `back` komen uit de pairs-tabellen (`run` en
     `run-long`); volgorde op `sha256("en-nl-sft/classifier/" + pid)`, oplopend; de eerste 1.800 trainen, de laatste
     200 testen (400 teksten). Label 1 = origineel (`nl`), 0 = vertaald (`back`). De lus en het model van
     `scripts/dumb/finetune.py` (branch `dumb-extra`: `create_model_and_state`, AdamW met constante LR en weight decay
     0,01, `collate_fn`), 8 per chip × 4 chips = 32 per update, opvullen tot een veelvoud van 128 (zoals dumb-extra);
     geen validatieset en geen keuze van een "beste" epoch: de toestand na 3 epochs wordt getest en gebruikt. De
     nauwkeurigheid telt alle 400 testteksten. "Origineel" = softmax-kans van label 1 ≥ 0,5.
   - Voorwaarde: nauwkeurigheid op de 10% ≥ 0,75 [besluit 10 nov ~09:00]; anders meldt het verslag dat (b) niet te
     meten is, en beslist Yeb.
   - Meting: het aandeel van de uitvoer dat de classifier "origineel" noemt (kans ≥ 0,5), op de vertalingen van de
     **WMT24++-bron (de 960, als §4.1) en de 1.000 FLAN-evaluatierijen** (set `classifier` van
     `$R/mainset/flan-eval/eval-rows.parquet`), samen genomen; per set gerapporteerd.
   - Voorbehoud [ontwerp]: de classifier leert de stijl van dit basismodel herkennen; een arm kan daaraan
     ontsnappen zonder natuurlijker te worden. Daarom ook 3 en 4.
2. **In-domein** [aanvulling hoofdsessie, 10 nov, vóór de training; ontwerp §4, natuurlijkheid punt 1]: de hele
   eval-split van de sets, **4.519 passages** (`$S/data/arms-v2/B/eval-meta.parquet`, uit andere documenten dan de
   training): het synthetische Engels (`en`) als invoer, het Nederlandse origineel als referentie. Vertaald door het
   basismodel en door elke arm, met TranslateGemma's eigen prompt en zonder context (ook bij de 296 passages die in de
   set een context hebben) [keuze]. Het basismodel vertaalt opnieuw, in dezelfde keten als de armen [keuze: de kolom
   `back` van de pairs-tabellen is ook een vertaling door het basismodel, maar uit de hoofdrun, met andere
   lengteklassen en bij contextbeurten met context]. Maten: MetricX-24 met referentie en QE, COMET-22 en chrF
   (sacrebleu, per passage en over het geheel) tegen het origineel, **per bron** (Handelingen, raad, dpc, rechtspraak,
   FineWeb-2) en samen. Een natuurlijkheidsmaat naast Europarl-direct; beslist niet mee. Dit vervangt "de volgende
   2.000 eval-passages" van het concept. De 2.000 passages van de classifier (1) zitten erin; de classifier wordt op
   deze uitvoer niet toegepast, dus dat raakt (b) niet.
3. **PLL en MTLD**: pseudo-log-likelihood onder S @ 653k (de PLL-word-l2r van BLiMP-NL in `dumb-extra`) per
   vertaling, en MTLD (woordenschatvariatie), op dezelfde uitvoer als 1. Gerapporteerd.
   - Uitvoering [keuze, vastgelegd vóór de training]: PLL met de code van `scripts/eval_blimp_nl.py` (branch
     `dumb-extra`: woorden zoals daar, `<s>`/`</s>` niet gescoord, float32, kop 0); per vertaling de som en de som
     gedeeld door het aantal tokens; vertalingen boven 1.022 tokens van S vallen weg en worden geteld. MTLD naar McCarthy
     en Jarvis (2010): woorden = `\w+` in kleine letters, drempel 0,72, het gemiddelde van de voorwaartse en de
     achterwaartse richting, per set over de aaneengesloten uitvoer in de volgorde van de id's (voor elk model
     dezelfde volgorde).
4. **Blinde beoordeling** [ontwerp §4.4, besluit 9 nov: 100 passages; besluit 10 nov ~09:20: een Claude-subagent
   beoordeelt, twijfelgevallen gaan naar Yeb; regel (c)].
   - **De 100 passages**, gekozen vóór de fine-tune op 10 nov met `blind.py` en vastgelegd in
     `$S/blind/blind100.jsonl` (sha256 `4dfe0675648a56caa872764a3208de7d22f6a6a6ded48130dd2e10567b1e0add`; per regel `blind_id`, `id`, `set`, de Engelse bron
     en herkomst); verslag `mbert-ops/translator-sft/blind-idiom-30.md`:
     - 35 FLAN: de 35 rijen van set `yeb` in `$R/mainset/flan-eval/eval-rows.parquet` (liggen vast sinds 9 nov);
     - 35 WMT24++: seed `en-nl-sft/blind/wmt`, uit de 960 zonder `is_bad_source`, zonder de 300 van de tagproef
       (een deel daarvan staat met vertalingen in `examples-qe-rule.md`), dus uit 660;
     - 30 met een Engels idioom (`$S/blind/idiom30.jsonl`, sha256 `b26d16ede8a757d7f7ad3d6c040ec4e53eb2762056bc2f9d7234cb4bd54c0033`) [voorstel hoofdsessie
       10 nov, uitgevoerd]: eigen alinea's van 40–200 woorden uit Engelse artikelen van Global Voices
       (`globalvoices.org`, CC BY 3.0 volgens de voettekst van elke pagina; buiten het domein van de trainingsdata),
       opgehaald via de WordPress-API van de site (de 1.006 berichten van 2025; 203 weggelaten die zeggen elders eerst
       te zijn verschenen of onder een samenwerking zijn overgenomen); niet de OPUS-versie, die geen URL en auteur per
       artikel heeft en tot 2018 loopt. Elk met minstens één uitdrukking uit de Engelse idiomen van Wiktionary (via
       kaikki.org, extract van 30 okt 2026), gematcht op lemma (spaCy 3.8.7, `en_core_web_sm` 3.8.0). Letterlijk
       gebruik is waar mogelijk uitgesloten: uitdrukkingen met een letterlijke of niet als idioom gemarkeerde betekenis
       in Wiktionary vallen weg, en de subagent die koos, las de gekozen passages na en sloot er 12 uit (letterlijk
       gebruik of een andere betekenis) en twee uitdrukkingen die telkens letterlijk bleken ("what someone said",
       "leave home"). Eerst de uitdrukking (seed `en-nl-sft/blind/idiom`), dan de passage (seed
       `en-nl-sft/blind/idiom/<uitdrukking>`), één per artikel. Per passage staan URL, auteur en uitdrukking in
       `idiom30.jsonl`. Opnieuw te maken met `blind.py` @ `b750bd0` (`run_blind.sh select` met de `EXCLUDE` en
       `EXCLUDE_IDIOMS` uit `mbert-ops/HANDOVER-sft.md`).
     - Geen van de 30 idioompassages en de 35 WMT24++-segmenten deelt een 13-gram met de trainingsdata van
       `arms-v2` [gemeten, `$S/blind/select-report.json`].
   - **Wie beoordeelt** [besluit 10 nov ~09:20]: per arm een verse Claude-subagent (zonder andere context dan de
     rubriek hieronder en de passages). Per passage krijgt hij alleen de Engelse bron en twee vertalingen, als A en B:
     die van het basismodel en die van de arm, in een volgorde die per passage met een vaste seed is gehusseld (seed
     `en-nl-sft/blind/ab/<arm>/<blind_id>`: `random.Random(seed).random() < 0,5` → A is het basismodel, anders de
     arm [keuze]). Het basismodel en de armen vertalen de 100 passages in de keten van de evaluatie, met
     TranslateGemma's eigen prompt. Hij weet niet welke vertaling van welk model is; de sleutel staat apart
     in de bucket (`$S/blind/<arm>/key.jsonl`), zijn invoer in `$S/blind/<arm>/judge-input.jsonl`.
   - **Rubriek** (letterlijk zo aan de beoordelaar gegeven; vastgelegd vóór de fine-tune):

     ```
     Je beoordeelt vertalingen uit het Engels in het Nederlands. Per passage krijg je de Engelse bron en twee
     Nederlandse vertalingen, A en B. Je weet niet hoe of door wie ze zijn gemaakt; probeer dat niet te raden.

     Beoordeel elke passage in deze volgorde:

     1. Betekenisfout, apart voor A en voor B: "ja" of "nee". Een betekenisfout is:
        - een weglating: iets uit de bron ontbreekt in de vertaling;
        - een toevoeging: de vertaling zegt iets wat niet in de bron staat;
        - een verkeerde betekenis: de vertaling zegt iets anders dan de bron;
        - een misplaatste uitdrukking: een idioom of vaste uitdrukking uit de bron is letterlijk vertaald waar dat in
          het Nederlands niet werkt, of vertaald met een Nederlandse uitdrukking die iets anders betekent.
        Woordkeus, stijl en zinsbouw zijn geen betekenisfout zolang de betekenis klopt.
     2. Oordeel: "A beter", "B beter" of "gelijk". Eerst op juistheid: de vertaling zonder betekenisfout, of met de
        minst ernstige, is beter. Zijn ze even juist, dan op natuurlijk, oorspronkelijk klinkend Nederlands: de
        vertaling die het meest leest als tekst die in het Nederlands is geschreven en niet als vertaling. "Gelijk"
        als er op beide punten geen duidelijk verschil is.
     3. Zekerheid: "zeker" of "twijfel", over je hele oordeel over deze passage (de betekenisfouten en de keuze).
        Twijfel je aan één van die drie onderdelen, kies dan "twijfel".
     4. Uitleg: één zin die je oordeel onderbouwt.

     Geef per passage één JSON-regel, en verder niets:
     {"blind_id": "...", "fout_A": "ja|nee", "fout_B": "ja|nee", "oordeel": "A beter|B beter|gelijk", "zekerheid": "zeker|twijfel", "uitleg": "..."}
     ```

   - **Twijfelgevallen** [besluit 10 nov ~09:20]: elk oordeel met "twijfel" gaat met de bron, A, B (nog steeds
     zonder sleutel) en de uitleg van de beoordelaar naar Yeb. Yeb geeft dezelfde vier onderdelen; zijn oordeel
     vervangt dat van de beoordelaar. Daarna pas wordt de sleutel toegepast.
   - Beoordeeld worden alleen de armen die (a) en (b) halen: een arm die daarop faalt, helpt volgens de regel toch
     niet [keuze].
   - **Voorbehoud** [besluit 10 nov ~09:20]: een taalmodel als beoordelaar kan een eigen stijlvoorkeur hebben (bijvoorbeeld
     voor vertalingen die op zijn eigen uitvoer lijken). Daarom staan in het verslag naast (c) de twijfelgevallen met
     Yebs oordeel, het aantal oordelen dat Yeb veranderde, en MetricX-24 en COMET-22 op dezelfde passages waar een
     referentie is (WMT24++).

## 5. Beslisregel (exact)

Per arm X ∈ {N, B}, tegen het basismodel TranslateGemma 27B (`$E/baseline-tg27b/summary.json`):

- **(a) betekenis**: voor elk van WMT24++ (de 960 zonder `is_bad_source` [besluit 10 nov ~09:00]), FLORES-200 en
  NTREX-128 geldt gemiddelde MetricX-24-QE(X) − gemiddelde MetricX-24-QE(basis) ≤ 0,20, over alle segmenten van de
  set (onafgeronde gemiddelden). Grenzen: WMT24++ 1,547 + 0,20 = 1,747, FLORES 0,822 + 0,20 = 1,022, NTREX
  1,082 + 0,20 = 1,282 (afgerond weergegeven). De drempel 0,20 blijft [besluit 10 nov ~10:15, na de voorbeelden in
  `mbert-ops/translator-sft/examples-qe-rule.md`]. Het uitgangspunt per set wordt uit de scores per segment
  (`$E/baseline-tg27b/scores.parquet`) berekend, niet uit de afgeronde waarden van `summary.json` [keuze].
- **(b) natuurlijkheid volgens de classifier**: aandeel "origineel"(X) − aandeel "origineel"(basis) ≥ 0,10, op de
  samengenomen uitvoer van WMT24++ (960) en de 1.000 FLAN-rijen (§4.2.1).
- **(c) blinde beoordeling** [besluit 10 nov ~09:20]: na vervanging van de twijfelgevallen door Yebs oordeel
  (§4.2.4) wint X in minstens 60% van de niet-gelijke oordelen: k / n ≥ 0,60, met n het aantal passages met "A
  beter" of "B beter" en k het aantal waarin X beter is; én het aantal vertalingen van X met een betekenisfout ≤ dat
  van het basismodel, over dezelfde 100 passages.

X **helpt** als (a), (b) en (c) alle drie gelden; anders **helpt X niet**. Europarl-direct, het in-domeindeel,
MetricX met referentie, COMET-22, PLL en MTLD worden gerapporteerd, maar beslissen niet [ontwerp].
Ter informatie bij (c): bij n = 60 is k = 36 (60%) eenzijdig binomiaal p = 0,078, bij n = 100 is k = 60 p = 0,028
[ontwerp].

Helpen N en B allebei, dan wordt het **B** [besluit 10 nov ~09:00].

## 6. Proef met de tag op TranslateGemma 12B (vóór de armen)

Doel: kiezen tussen de twee vormen van de tag (§2.4), omdat `<BT> ` als platte tekst in de smoke-test lekte
(eenmaal letterlijk, eenmaal als inleidende zin) [gemeten 9 nov].

- Twee LoRA-adapters op TranslateGemma 12B (`run_sft.sh`, instellingen van §3): elk 20.000 BT-voorbeelden uit de
  query van §2.1 (4.000 met context, 20%) plus 5.000 replay (20%), 12,9–13,0 M tokens, dezelfde paren voor beide;
  alleen de vorm van de tag verschilt (`$S/data/tagtest/`). Elk als één job over de 16 chips van mbert-mix, dus dezelfde
  indeling als de 27B-armen (16 × 1 × 2.048 tokens per stap): op één worker
  vroeg 12B bij 1 × 2.048 per chip 38,5 GB aan tijdelijke buffers (> 30,75) [gemeten 10 nov].
- Daarna vertaalt elke adapter met en zonder tag 300 FLAN-evaluatierijen (set `classifier`, seed
  `en-nl-sft/tagtest/0/flan`) en 300 WMT24++-segmenten zonder `is_bad_source` (seed `…/wmt`); het basismodel 12B
  ernaast.
- Geteld per variant en modus (`tagtest.py`): tag of instructie letterlijk in de uitvoer; een inleiding ("Hieronder
  volgt …"); afwijkende lengte tegen het basismodel (> 1,4× of < 0,6× en chrF < 50 tegen de basisvertaling: een
  teken van een beantwoorde vraag of een weggelaten stuk; met de hand nagelezen); lussen (`decode_loop`); niet
  afgemaakt; Engels over; MetricX-24-QE.
- Uitslag: `mbert-ops/translator-sft/tag-test.md` en `$S/tagtest/report/`.

**Stand bij het schrijven** (10 nov ~07:00 CET): de eerste training van de adapter met het token (431 stappen)
brak na de laatste stap af op het leeglopen van de data op één proces (§3); opnieuw met de stapgrens van §3, daarna
die met de instructie, dan de vertalingen en MetricX. De uitslag komt in
`mbert-ops/translator-sft/tag-test.md` (automatisch, met tabel en 10 voorbeelden per adapter) en
`$S/tagtest/report/`; de keuze van de vorm hoort daarna hier, als besluit van Yeb.

**Uitslag** (10 nov 08:18, nagelezen door de hoofdsessie; 600 vertalingen per variant):

| variant | tag letterlijk | inleiding | langer / korter | lus | QE WMT24++ gem. | QE alles gem. |
|---|---:|---:|---:|---:|---:|---:|
| basis 12B | 0 | 0 | 0 / 0 | 0 | 1,715 | 2,089 |
| token, met tag | 0 | 0 | 3 / 6 | 0 | 1,925 | 2,208 |
| token, zonder tag | 0 | 0 | 0 / 3 | 0 | 1,830 | 2,107 |
| instructie, met tag | 0 | 0 | 0 / 3 | 0 | 1,923 | 2,156 |
| instructie, zonder tag | 0 | 0 | 0 / 4 | 0 | 1,847 | 2,131 |

- Geen van beide vormen lekt: geen tag of instructie in de uitvoer, geen inleidingen, geen lussen. Het lekken van
  `<BT> ` in de smoke-test was dus de platte-tekstvorm.
- Nagelezen gevallen:
  - De FLAN-rij `flan:0-1173495` staat bij elke variant als "korter", omdat het basismodel daar een lange lijst
    verzon. Daar zijn de varianten juist beter.
  - De variant token met tag liet twee keer een @-vermelding weg ("@user33 wow!" → "Wauw!"), liet "hehehe" weg, en
    maakte van "Finding an alternative shortly…" "Zoek een alternatief...". Dat is betekenisverlies op korte
    sociale-mediazinnen.
  - De variant instructie met tag maakte "And this is why" korter tot "En daarom."; verder niets.
- MetricX-QE op WMT24++ ligt met tag bij beide vormen ~0,21 boven het basismodel, precies de drempel van regel (a).
  Een vrijere, natuurlijkere vertaling scoort op QE vermoedelijk slechter (§4), maar dit is wel een waarschuwing
  voor de armen.
- **Voorstel (hoofdsessie): de instructiezin.** Hij geeft minder afwijkingen (3 tegen 9), laat geen @-vermeldingen
  weg en heeft een iets betere QE over alles (2,156 tegen 2,208). Het besluit is aan Yeb.
- **Besluit Yeb (10 nov ~09:00)**: de tag wordt omgedraaid (§1, §2.4): de BT-paren zonder tag, de replay met een
  instructiezin voor de gewone, letterlijke vertaalmodus. De proef blijft hier als achtergrond staan. Het verschil van
  ~0,21 op WMT24++ is de aanleiding voor de voorbeelden bij regel (a) (`examples-qe-rule.md`): het gemiddelde komt
  vooral uit een staart van segmenten met een groot verschil (mediaan 0,001; de 10 grootste verschillen dragen 63%
  van de som) [gemeten].

## 7. Stappen na akkoord

1. Yeb beslist de open punten (§8); daarna wordt dit protocol vastgelegd (commit, datum) vóór de eerste stap op 27B.
   **Gedaan**: vastgelegd 10 nov (Yebs akkoord ~10:15); de QE-grens blijft 7,5, dus de sets `arms-v2` blijven.
2. De 100 blinde passages: vastgelegd op 10 nov (§4.2.4).
3. De classifier trainen en zijn nauwkeurigheid melden (§4.2.1); de uitvoer van het basismodel op de 1.000
   FLAN-rijen, het in-domeindeel en de blinde passages maken.
4. Per arm: korte run (`STEPS=8`, één keer, op de set van B), dan de volledige epoch, dan de evaluatie van §4.
   Keten: `mbert-ops/scripts/translator_sft_arms_run.sh`; commando's en stand in `mbert-ops/HANDOVER-sft.md`.
5. Verslag per arm met de regel van §5.

## 8. Besluiten en open punten

**Beslist door Yeb, 10 nov ~09:00:**

- De tag omgedraaid: BT-paren zonder tag, FLAN-nl-replay met een vaste instructiezin; bij gebruik niets erbij (§1,
  §2.4). Arm N: BT zonder tag plus replay met de zin; arm B: BT zonder tag, zonder replay.
- WMT24++ volgens het artikel: de 960 segmenten zonder `is_bad_source` (§4.1, §5).
- Alle BT-paren die door de query komen: 225.242 train, 6,8% met context; de replay in N blijft 20% van het totaal
  (§2.2, §2.3).
- De classifier op S @ 653k met nauwkeurigheid ≥ 0,75 (§4.2.1); als N en B allebei helpen, wordt het B (§5); de
  27B-armen op mbert-mix.

**Beslist door Yeb, 10 nov ~09:20:** de blinde beoordeling doet een verse Claude-subagent met een vaste rubriek;
twijfelgevallen gaan naar Yeb, wiens oordeel dat van de beoordelaar vervangt; regel (c) zoals in §5; met het
voorbehoud over de stijlvoorkeur van een taalmodel (§4.2.4).

**Beslist door Yeb, 10 nov ~10:15 (akkoord om te starten):**

1. De drempel 0,20 van regel (a) blijft: een arm mag op WMT24++ (960), FLORES en NTREX met MetricX-QE gemiddeld
   hoogstens 0,2 slechter zijn dan het uitgangspunt (§5).
2. De grens MetricX-QE ≤ 7,5 in de trainingsdata blijft, tegen uitschieters (§2.1).
3. ~19 uur rekentijd op mbert-mix voor beide armen met training en evaluatie is akkoord; mbert-corpus blijft bij de
   eindrun.
4. N wordt ook mét de instructiezin vertaald en gerapporteerd, als controle van de modus; het beslist niet mee
   (§4.1). Keuze van de hoofdsessie bij het vroegere open punt 5.

**Aanvulling van de hoofdsessie, 10 nov, vóór de training:** het in-domeindeel is de hele eval-split (4.519
passages), met het basismodel ernaast, gescoord met MetricX-24 (ref en QE), COMET-22 en chrF, per bron (§4.2.2).

**Ter kennisname, zonder bezwaar vóór het vastleggen:** de 30 idioompassages (§4.2.4).

**Nog te meten, niet te beslissen:** de sequentielengte van 2.048 met 1 sequentie per chip op 27B. De korte run
meet geheugen en staptijd; past het niet, dan kiest de subagent de kleinste wijziging die past (bijvoorbeeld 1.024
voor de voorbeelden zonder context, of remat) en legt die vast als amendement in §9, vóór de volledige epoch.

## 9. Amendementen na het vastleggen

Geen amendement (stand 10 nov ~10:50).

**Meting, 10 nov ~10:50: de korte run past, de instellingen van §3 blijven.** Run `arm27b_v2_B_s8` (8 stappen, set B,
`per_device_batch_size=1 max_target_length=2048`, `remat_policy` full zoals MaxText's standaard, multi-host over de 16
chips van mbert-mix; code `e8979e1`, `sft_lora.py` en `run_sft.sh` gelijk aan `c7c600e`) [gemeten, `summary.json` in
`$S/arms/runs/arm27b_v2_B_s8/`]:
- piekgeheugen 21,23 GiB per chip op alle 16 (grens 30,75 GiB);
- staptijd mediaan 7,11 s, gemiddeld 7,53 s over stap 2–8; 3.982 tokens/s over de 16 chips (de smoke-test bij
  2 × 1.024: 4.018); ~29.900 van de 32.768 tokenplaatsen per stap gevuld;
- `EPOCH`: batches per proces [3.638, 3.625, 3.649, 3.629], `step_limit` 3.588: de volledige epoch van B wordt
  3.588 stappen, ~7,1–7,5 uur, plus ~12 min tellen, ~2 min laden en ~6 min compileren.

**Amendement 11 nov 2026 ~09:30 CET: armen N50 en N25, arm N met een afgezwakte adapter** [besluit Yeb 11 nov,
vastgelegd vóór de eerste vertaling; niets wordt getraind]. Aanleiding: de uitslag van 11 nov
(`$S/arms/report/report.md`, code `998b34e`): N haalt (a) niet (MetricX-24-QE +1,177 / +0,688 / +0,746 op WMT24++
(960) / FLORES / NTREX, grens +0,20) en (b) wel (+0,127). Yeb: meet N met de LoRA op 50% en 25% sterkte, als
tussenpunt tussen natuurlijk en nauwkeurig.
- **Armen.** N50 en N25 = de adapter van N (`$S/arms/N/`, 4.143 stappen) met de bijdrage van de LoRA × 0,5 en
  × 0,25: elke `kernel_lora_b` vermenigvuldigd met de factor s, `kernel_lora_a`, rank, alpha en `scale` ongewijzigd,
  zodat de update van elke kernel (A·B·scale, `translate.py --lora`) s × die van N is [keuze: B × s; een factor 0,5 of
  0,25 is in float32 exact]. Adapters in `$S/arms/N50/` en `$S/arms/N25/`; hun `adapter.json` noemt de bron en s.
- **Vertalen en meten zoals §4.1**: dezelfde vier sets (WMT24++, FLORES-200, NTREX-128, Europarl-direct), dezelfde
  code, TranslateGemma's eigen prompt zonder instructiezin, greedy; MetricX-24 met referentie en QE, COMET-22; niet af,
  lus, Engels over, leeg.
- **Beslisregel zoals §5, ongewijzigd.** (a): tegen hetzelfde uitgangspunt (`$E/baseline-tg27b/scores.parquet`, grenzen
  1,747 / 1,022 / 1,282) op dezelfde drie sets, drempel 0,20. (b): dezelfde classifier (S @ 653k,
  `$S/arms/naturalness/classifier/`, nauwkeurigheid 0,9575; niet opnieuw getraind) op de WMT24++ (960) en de 1.000
  FLAN-rijen (uit de vertaling van soort `x`, zoals bij N), aandeel "origineel" min dat van het basismodel ≥ 0,10
  [keuze: de teksten van het basismodel gaan opnieuw door dezelfde classify-run, naast N50 en N25; ter controle tegen
  de 19,1% van 11 nov]. (c): alleen voor een arm die (a) en (b) haalt, zoals §5 en §4.2.4 (dezelfde 100 passages, A/B
  met seed `en-nl-sft/blind/ab/<arm>/<blind_id>`, `<arm>` = N50 of N25).
- **Halen N50 en N25 allebei (a) en (b), dan gaat alleen N50 (de sterkere bijdrage) door naar (c).**
- Volgorde [keuze, door de deadline: mbert-mix moet om 12:45 CET vrij zijn voor tak 1 van de eindrun]: eerst de vier
  sets (regel (a)), dan FLAN en classify (regel (b)); de scores van het in-domeindeel worden gerapporteerd als ze
  passen en beslissen niet mee. Niet gemeten: PLL, MTLD (behalve wat het rapport uit de teksten berekent), N50/N25 met
  de instructiezin. Wat niet past, staat als niet gemeten in het rapport.

**Amendement 11 nov 2026: arm R2, een tweede trainingsronde** [besluit Yeb 11 nov ~09:15, met de keuzes van de
hoofdsessie; vastgelegd vóór de eerste trainingsstap van R2; los van het amendement N50/N25 hierboven]. Aanleiding: B
en N halen (a) ruim niet (`$S/arms/report/report.md`): de uitvoer is ~10% korter, er vallen stukken weg, er staan
letterlijke calques in en Engels blijft staan; echt Engels zag het model alleen in de replay van N, en alleen mét de
instructiezin. Eén nieuwe arm, **R2**; B, N, N50 en N25 blijven zoals ze zijn.

- **Trainingsdata** (`$S/data/r2/R2/`; `sft_build.py build --bt all --qe-max 5 --len-factor 1.3333 --block-jsonl
  blind100.jsonl --real real-qe.parquet --sets R2=none+real`, keten `run_arms.sh r2data`, `r2qe`, `r2build`, `r2check`):
  1. **BT-paren**: de query van §2.1 en de contextvoorbeelden van §2.2, met drie wijzigingen: (i) MetricX-24-QE
     en→nl ≤ **5** in plaats van 7,5 [besluit Yeb: niet 3, omdat de en→nl-QE ook vrije, idiomatische formuleringen
     afstraft]; (ii) een **lengtegrens** [besluit Yeb; vorm: keuze]: de verhouding tekens Nederlands / tekens Engels ligt
     binnen een factor 4/3 van de mediaan van de bron (mediaan / 1,3333 tot mediaan × 1,3333; de medianen over de
     trainpassages van die bron die door `hard_fails` komen), aan beide kanten; een grens tegen uitschieters, geen
     correctie van het niveau van een bron; (iii) geen 13-gram gedeeld met de bronnen van de 100 blinde passages
     (`$S/blind/fixed-2026-11-10/blind100.jsonl`; voor B en N gold dat al als gemeten, niet als grens). Zonder tag;
     contextvoorbeelden met `chat-v1` met context, zoals §2.4.
  2. **Echt Engels** [besluit Yeb]: Engels-originele tekst met een menselijke Nederlandse vertaling, met
     TranslateGemma's eigen prompt (`chat-v1` zonder context) en **zonder** instructiezin; alles in train (`real_pairs.py`):
     - **Global Voices en→nl** (CC BY 3.0): alle berichten van `nl.globalvoices.org` (WordPress-API) waarvan de pagina
       een Engelse versie noemt; die Engelse versie is zelf geen vertaling (geen regel "post-source-date" op haar
       pagina), geen overgenomen partnertekst (`blind.REPUBLISHED`, zoals bij de blinde passages), en beide pagina's
       noemen CC BY 3.0. Per artikel de eigen alinea's van beide kanten (`blind.paragraphs`: geen citaten, bijschriften,
       embeds, tabellen, lijsten of cursieve noten), monotoon gekoppeld (`real_pairs.align`: lengte, gedeelde links en
       getallen, plaats; een alinea zonder tegenhanger valt weg); geen boilerplate ("Noot van de vertaler", "Lees
       ook"). Uitgesloten: de artikelen van de blinde beoordeling en de idioomset, en alle Engelse berichten van 2025
       (de pool van de derde subagent) [keuze].
     - **Europarl-direct en→nl**: de XML van de MoTra-repo (`UDS-SFB-B6-Datasets/Multilingual-Parallel-Direct-Europarl`
       @ `4e284f1`), gekoppeld zoals `en-nl-judge/data.py motra` (alinea's met `sl="en"`), **2004–2011** [keuze:
       "na 2003"; de evaluatie is 1999–2003, ontwerp §6]; de rol van de spreker vóór de eerste alinea ("lid van de
       Commissie . – ") weggehaald, aan beide kanten (2004–2011: aan de Nederlandse kant in 4.893 eerste alinea's, aan
       de Engelse in 81) [keuze: metadata, geen tekst].
     - **Harde controles** (`real_pairs.real_fails`): minstens 5 Engelse woorden, Nederlands ≠ Engels, geen Engels in het
       Nederlands (`publish.english_left`), lengteverhouding ≤ 1,5 in tekens (als `pairs.py`).
     - **Dezelfde query als de BT-paren**: MetricX-24-QE en→nl ≤ 5 (gemeten in de keten vóór de build, stap `r2qe`,
       MetricX-24 hybrid XXL), de lengtegrens (factor 4/3 rond de mediaan van de bron), geen 13-gram met een
       evaluatieset of een blinde passage, ≤ 2.048 tokens.
     - **Omvang** [keuze hoofdsessie]: hoogstens **20% van de voorbeelden** (echt ≤ BT × 0,25): alle bruikbare Global
       Voices-paren, Europarl als aanvulling, willekeurig getrokken (seed `en-nl-sft/0/real/rest`; Global Voices met
       `en-nl-sft/0/real/globalvoices` als het er meer zijn dan de grens). Is er minder, dan alles.
  3. **Geen FLAN-replay** en nergens een instructiezin [besluit Yeb].
- **Wat elke grens kost** en de verhoudingen: zie de tabellen hieronder (gemeten 11 nov); de kosten van de QE-grens op
  de echte paren meet de keten vannacht (stap `r2qe`) en staan dan in `$S/data/r2/report.md` (kopie
  `mbert-ops/translator-sft/data-r2-report.md`).

  **BT-paren, train** [gemeten 11 nov, proefbouw `$S/data/r2-dry/report.md`; eerste grens die faalt, tussen haakjes
  alle paren die de grens niet halen]:

  | grens | R2 | (arms-v2, §2.1) |
  |---|---:|---:|
  | passages vóór de query | 229.472 | 229.472 |
  | `hard_fails` niet leeg | 2.113 | 2.113 |
  | bronscore / herkomst | 0 / 0 | 0 / 0 |
  | MetricX-QE > 5 (> 7,5) | 18.270 (18.604) | 2.117 |
  | lengtegrens (factor 4/3) | 363 (528) | – |
  | 13-gram met een evaluatieset of blinde passage | 0 | 0 |
  | langer dan 2.048 tokens | 0 (186 contexten weg) | 0 (261) |
  | **komt in aanmerking** | **208.726** (6,0% met context) | 225.242 |
  | eval | 4.167 | 4.519 |

  Lengtegrens per bron (tekens NL / EN; medianen over de trainpassages zonder `hard_fails`): dpc 0,984 (0,738–1,311),
  FineWeb-2 1,003 (0,752–1,337), Handelingen 1,006 (0,755–1,341), raad 0,993 (0,745–1,324), rechtspraak 1,029
  (0,772–1,372). Kosten bij andere factoren (BT met QE ≤ 5, onder / boven): 1,2: 3.281 / 1.390 (2,2%); 1,25: 1.346 /
  450 (0,9%); **4/3: 309 / 54 (0,17%)**; 1,5: 2 / 0. Voorbeelden op de grens: `mbert-ops/translator-sft/examples-r2-length.md`
  (`$S/examples/examples-r2-length.md`). Daar blijkt: de BT-paren net buiten de grens zijn meestal correcte
  vertalingen waarin het synthetische Engels een afkorting uitschrijft ("Works Council Act (WOR)", "mavo-havo-vwo") of
  het Nederlands compact is; een strengere factor zou vooral dat compacte Nederlands wegsnijden. Daarom 4/3: een grens
  tegen uitschieters, die het niveau van een bron laat staan [keuze, met de aanvulling van de hoofdsessie van 11 nov].

  **Echte paren** [gemeten 11 nov; `$S/data/r2-real/`; QE-grens nog niet gemeten]:

  | | Global Voices | Europarl-direct |
  |---|---:|---:|
  | bron | 2.282 Nederlandse berichten; 1.245 met Engelse versie; 224 artikelen overgenomen partnertekst, 4 Engels zelf vertaling, 3 niet in de API: 1.014 artikelen | 126.362 en→nl-alinea's; 2004–2011 |
  | gekoppelde paren | 13.604 | 87.745 |
  | blinde artikelen / pool 2025 | 32 / 1.082 | – |
  | harde controles: kort / lengte > 1,5 / gelijk / Engels over | 348 / 123 / 9 / 3 | 71 / 719 / 15 / 22 |
  | door de harde controles | 12.007 | 86.918 |
  | lengtegrens (factor 4/3 rond mediaan 1,110 / 1,147) | 176 | 406 |
  | 13-gram | 0 | 2 |
  | MetricX-QE > 5 | vannacht (`r2qe`) | vannacht |
  | in aanmerking zonder QE-grens | 11.831 | 86.510 |

  De Global Voices-teksten verliezen ook de taalmarkering van links ("[en]", "[en – alle links]"): aan de Nederlandse
  kant in 2.150 van de 13.604 paren, aan de Engelse in 226 [keuze: metadata, geen tekst]. Bevatten de bronnen zelf
  weglatingen? Ja, in de staart: onder de ondergrens (0,75 × de mediaan) liggen 151 Global Voices-paren (1,26%) en 406
  Europarl-paren (0,47%), onder 0,7 × de mediaan 68 (0,57%) en 148 (0,17%); daar laat de vertaler een zin of bijzin weg
  of vat samen ("Some industries are already showing signs of a major slow-down." ontbreekt; "not only to you, Madam
  President, for delighting us with your presence" ontbreekt). De mediaan is 1,05 (Global Voices) en 1,07 (Europarl)
  Nederlandse woorden per Engels woord.

  **Proefbouw** (`$S/data/r2-dry/`, `--real-without-qe`, niet voor de training): 260.908 trainvoorbeelden (BT 208.726,
  waarvan 12.540 met context; Global Voices 11.831; Europarl 40.351; echt 20,0%), 112,6 M tokens, 47,6 M doeltokens;
  eval 4.167. Mediane verhouding woorden NL / EN (tekens): hele set 0,980 (1,020), BT 0,967 (1,004), echt 1,062
  (1,139). Gecontroleerd met `sft_build.py check` (MaxText's eigen SFT-stappen op elk voorbeeld; de echte voorbeelden
  met de token-id's van TranslateGemma's eigen prompt): EXIT 0, train 260.908, eval 4.167, max 2.048 tokens
  (`$S/data/r2-dry/check.txt`). De set voor de training wordt vannacht gebouwd met de QE-grens op de echte paren;
  Europarl vult dan aan tot 20%.

- **Training** zoals arm N (§3), behalve wat hieronder open staat: LoRA op attentie en MLP, cosine naar 0 met 10%
  warmup, één epoch (`step_limit`), sequenties van 2.048, 1 per chip, 16 chips. Uitvoer `$S/r2/runs/arm27b_r2_R2/`,
  adapter `$S/r2/R2/`.
  - **Open parameter, vóór de start vast te leggen**: LoRA-rank, alpha en LR. De standaard is die van N (rank 16,
    alpha 16, LR 1e-5); Yeb oppert een grotere rank en minder hard ingrijpen; de hoofdsessie beslist na de meting van
    N50/N25. Regel: vóór de start vervangt de hoofdsessie `open` in de regel hieronder door de waarden, in de vorm
    `R2-instellingen: lora_rank=<r> lora_alpha=<a> learning_rate=<lr>` (bijv. `lora_rank=16 lora_alpha=16
    learning_rate=1e-5`), in een gepushte commit;
    de keten (`translator_sft_r2_run.sh`, `LORA_RANK`, `LORA_ALPHA`, `LR`) weigert te starten als die regel niet
    letterlijk gelijk is aan zijn startwaarden, en controleert na de training de rank in `adapter.json`.

R2-instellingen: lora_rank=16 lora_alpha=16 learning_rate=1e-5

  - Geheugen bij een hogere rank [schatting, niet gemeten]: de LoRA-parameters groeien lineair met de rank
    (gemeten bij rank 16: 113.516.544, `summary.json` van B): rank 32 227,0 M, rank 64 454,1 M. In float32, met
    gradiënt en de twee momenten van AdamW, op elke chip gerepliceerd (`sft_lora.py` eist gerepliceerde LoRA-parameters):
    16 bytes per parameter, dus 1,69 GiB per chip bij rank 16 (zit in de gemeten piek van 21,23 GiB), 3,38 GiB bij 32
    en 6,77 GiB bij 64. Verwachte piek: rank 32 ~22,9 GiB, rank 64 ~26,3 GiB per chip (grens 30,75); als de stap van de
    optimizer die toestand tijdelijk verdubbelt, ~24,6 en ~31,4 GiB. Rank 32 past dus; rank 64 waarschijnlijk, maar in
    het slechtste geval niet. Een te hoge piek faalt in de eerste ~20 minuten (tellen ~12, compileren ~6) en stopt de
    keten met `FAILED`. De extra rekentijd per stap is klein: ~0,5% (rank 32) en ~2% (rank 64) meer matrixproducten.
- **Evaluatie, gelijk aan die van B en N** (§4; keten `translator_sft_r2_run.sh`, rapport `$S/r2/report/report.md`):
  - de **vier sets** (WMT24++ 998 en de 960, FLORES-200, NTREX-128, Europarl-direct) met dezelfde code: MetricX-24 met
    referentie en QE, COMET-22; niet af, lus, Engels over, leeg;
  - de **extra sets** van soort `x` (FLAN 1.000, blind 100, in-domein 4.519): dezelfde 4.519 passages als bij B en N
    (`$S/arms/data/x/`), niet de kleinere eval-split van R2 [keuze: vergelijkbaar met B, N en het basismodel]; de
    vertalingen van het basismodel van 10 nov (`$S/arms/eval/base-x/`) [keuze: dezelfde code, greedy];
  - de **classifier** van 10 nov (S @ 653k, `$S/arms/naturalness/classifier/`, nauwkeurigheid 0,9575; niet opnieuw
    getraind), op de WMT24++ (960) en de 1.000 FLAN-rijen; de teksten van basismodel en R2 in één classify-run;
  - **PLL en MTLD** zoals §4.2.3;
  - het **blinde pakket** `$S/blind/R2/` (`judge-input.jsonl`, sleutel `key.jsonl` alleen in de bucket; A/B met seed
    `en-nl-sft/blind/ab/R2/<blind_id>`), dezelfde 100 passages;
  - het **rapport** met regel (a), (b) en de stand van (c), zoals voor B en N.
  - **Gerapporteerd, beslist niet mee** [aanvulling hoofdsessie, 11 nov]: de lengte van de uitvoer tegenover de
    menselijke referentie op de vier sets (`armeval.py report --four-texts`): per segment Nederlandse woorden per
    Engels woord van referentie en uitvoer (mediaan), woorden van de uitvoer per woord van de referentie (mediaan) en
    het aandeel onder 0,7 × de referentie, voor het basismodel en R2. Ook in het rapport van de set: de mediane
    verhouding van de hele set, van het BT-deel en van het echte-Engelsdeel.
  - Niet: R2 met de instructiezin (R2 heeft er geen).
- **Beslisregel zoals §5, ongewijzigd, tegen hetzelfde uitgangspunt** (`$E/baseline-tg27b/scores.parquet`): (a)
  MetricX-24-QE(R2) − QE(basis) ≤ 0,20 op WMT24++ (960), FLORES-200 en NTREX-128 (grenzen 1,747 / 1,022 / 1,282);
  (b) aandeel "origineel" ≥ basis + 0,10 met dezelfde classifier; (c) de blinde beoordeling, alleen als (a) en (b)
  gelden. R2 helpt als (a), (b) en (c) alle drie gelden.
- **Aanvulling 11 nov ~10:30 CET: arm N75** (N met de bijdrage van de LoRA × 0,75; elke `kernel_lora_b` × 0,75, in
  float32 met één afronding per element), verkennend toegevoegd na het zien van N50 en N25, op 11 nov ~10:25 op verzoek
  van Yeb; vastgelegd vóór de eerste vertaling van N75. Gezien op dat moment [gemeten, keten tag `s11110816`]: N50
  haalt (a) (MetricX-24-QE +0,029 / +0,033 / +0,034) en niet (b) (aandeel "origineel" 20,20% tegen 19,13%, +1,1 pp;
  WMT24++ 34,3% tegen 32,3%, FLAN 6,7% tegen 6,5%); N25 haalt (a) (−0,012 / +0,007 / +0,003) en niet (b) (+0,5 pp).
  Zelfde regels (a) en (b), zelfde uitgangspunt, sets en classifier als N50 en N25. Haalt N75 (a) en (b), dan gaat N75
  door naar (c) (zoals hierboven; N50 en N25 halen (b) niet). Volgorde vóór 12:45: N75 op de vier sets (vertalen,
  scoren), dan classify; de scores van soort `x` (in-domein, alleen gerapporteerd) van N75 vervallen als ze niet passen.
