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

R2-instellingen: lora_rank=die van RB2 (geheugentest) lora_alpha=rank/2 learning_rate=1e-5

  [Aangepast 11 nov ~21:45 CET, vóór de start van R2, door het amendement "R2 met de rank van RB2" onderaan §9; was
  `lora_rank=16 lora_alpha=16 learning_rate=1e-5` (`9082ee6`).]

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

**Amendement 11 nov 2026 ~12:00 CET: arm RB, de BT-paren plus Nederlandse boeken** [besluit Yeb 11 nov ~12:00;
vastgelegd vóór de eerste trainingsstap van RB; los van R2, waarvan data en instellingen ongewijzigd blijven].
Aanleiding: arm N comprimeert op echt Engels: hij schrijft ~5% korter dan menselijke vertalers en laat stukken weg
[meting van de hoofdsessie, 11 nov]. Het synthetische Engels van de BT-paren is omslachtiger dan het
Nederlandse origineel, dus het model leert comprimeren. Een set Nederlandse boeken met hun gepubliceerde Engelse
vertaling geeft echt, idiomatisch Engels met een **origineel** Nederlands doel. Eén nieuwe arm, **RB**.

- **Trainingsdata** (`$S/data/rb/RB/`; `sft_build.py build --bt all --qe-max 5 --len-factor 1.3333 --block-jsonl
  blind100.jsonl --real real-qe.parquet --real-all --sets RB=none+real`; keten `run_arms.sh rbdata`, `rbqe`, `rbedges`,
  `rbbuild`, `rbcheck`):
  1. **Alle BT-paren met de query van R2** [besluit Yeb]: punt 1 van het amendement R2 (§2.1 met MetricX-24-QE en→nl
     ≤ 5, de lengtegrens factor 4/3 rond de mediaan van de bron, geen 13-gram met een evaluatieset of een blinde
     passage) en de contextvoorbeelden zoals R2. Hetzelfde BT-deel als R2 (proefbouw R2: 208.726 voorbeelden, 12.540 met
     context).
  2. **Alle boekparen** [besluit Yeb] uit `yhavinga/nedd_x_dataset_nlbooks_b1` (privé HF-dataset, revisie `0270fc9`,
     15.004 rijen van een paar alinea's, kolom `translation{nl,en}`; rechtstreeks van HF naar w3 van mbert-mix met het
     token van de worker). Richting: Engels (bron) → Nederlands origineel (doel), met TranslateGemma's eigen prompt
     (`chat-v1` zonder context) en **zonder** instructiezin; de regeleinden binnen een rij blijven staan. **Alles in
     train, geen held-out** [besluit Yeb]; wel geen 13-gram gedeeld met een evaluatieset of een blinde passage, en
     ≤ 2.048 tokens. `real_pairs.py books-fetch`, `books`, `candidates --books`.
     - **Boeken die niet in het Nederlands geschreven zijn, eruit** [keuze, ter bevestiging aan Yeb]: vijf auteurs van
       de set schreven niet in het Nederlands: Eric Van Lustbader (956 rijen), Matthew Pearl (626) en Morgan Rice (167)
       in het Engels, Paulo Coelho (187) in het Portugees, Giovanni Guareschi (89) in het Italiaans; samen 2.025 rijen.
       Hun Nederlands is een vertaling, niet het originele doel waarvoor RB er is (de README van de set kiest boeken op
       id, niet op taal). Harde fout `not_dutch_original` (`real_pairs.NOT_DUTCH_ORIGINAL`). Over: 12.979 rijen van 19
       auteursnamen (o.a. Wieringa, Koch, Mulisch, Hermans, Couperus, Nescio, Bossenbroek, Brokken, Van Iperen;
       Gerbrand Bakker en Simone van der Vlugt staan er onder meer dan één spelling in).
     - **Harde controles** zoals de echte paren van R2 (`real_pairs.real_fails`): minstens 5 Engelse woorden,
       Nederlands ≠ Engels, geen Engels in het Nederlands (`publish.english_left`), lengteverhouding ≤ 1,5 in tekens.
     - **Uitlijnfouten eruit** [besluit Yeb; in een steekproef begon een Engelse alinea twee zinnen eerder dan de
       Nederlandse]: dezelfde query als de echte paren van R2: MetricX-24-QE en→nl ≤ **5** (stap `rbqe`, MetricX-24
       hybrid XXL) en de **lengtegrens** (tekens NL / tekens EN binnen een factor 4/3 van de mediaan van de boeken,
       over de paren die door de harde controles komen).
     - **Omvang**: alle paren die in aanmerking komen (`--real-all`), geen aandeel [besluit Yeb].
  3. **Geen** Europarl, **geen** Global Voices, **geen** FLAN-replay, nergens een instructiezin [besluit Yeb].
- **Wat elke grens kost bij de boeken** [gemeten 11 nov ~12:10, `$S/data/rb-real/`, code `21239f0`; volgorde zoals
  `sft_build.py`, eerste grens die faalt; voorbeelden op beide grenzen en per auteur:
  `mbert-ops/translator-sft/examples-rb-books.md` (`$S/examples/examples-rb-books.md`)]:

  | grens | rijen |
  |---|---:|
  | rijen in de set | 15.004 |
  | niet in het Nederlands geschreven (`not_dutch_original`) | 2.025 |
  | harde controles: Engels over / lengte > 1,5 / kort | 80 / 15 / 1 |
  | door de harde controles | 12.883 |
  | MetricX-QE > 5 | 4.033 (31,3%) |
  | lengtegrens (mediaan 1,033: 0,774–1,377) | 13 (alle 13 boven de grens) |
  | over vóór 13-grammen en tokens | 8.837 |

  MetricX-24-QE van de 12.883 paren: p10 1,81, p25 2,73, mediaan 4,00, p75 5,38, p90 6,96, p99 12,40. Kosten bij
  andere QE-grenzen (paren boven de grens): 3: 9.002; 4: 6.443; **5: 4.033**; 6: 2.287; 7,5: 941; 10: 300. De
  lengtegrens alleen (zonder QE) zou 13 paren eronder en 32 erboven kosten; de QE-grens neemt de meeste daarvan al.
  **Wat de grenzen raken.** De hoogste QE (23–25) zijn verschoven passages: het Nederlands begint met een of twee
  zinnen die in het Engels ontbreken (Mulisch `books:74768-137379-b1-950`, Hermans `…-b1-217`, Koch `…-b1-223`). De
  lengtegrens vangt die niet, omdat de lengte van een verschoven passage nauwelijks verandert. Net boven QE 5 staan
  vooral correcte, vrije literaire vertalingen (Gerritsen 5,02, Wieringa 5,33), soms met een weggevallen zin aan de
  rand ("Okay." bij Bakker 5,63). De QE-grens kost dus ook goede paren, en het meest bij ouder of literair Nederlands:
  over blijven van Couperus 18,7%, Nescio 21,6%, Mortier 36,1%, Slauerhoff 43,8%, tegen Koch 90,1%, Brokken 89,7% en
  Van der Vlugt (2017) 89,9%. De grens van 5 blijft zoals Yeb besliste.

  **De set** [gemeten 11 nov 12:14, `$S/data/rb/report.md`, code `21239f0`]: 217.563 trainvoorbeelden: BT 208.726
  (12.540 met context, 6,0%; gelijk aan het BT-deel van de proefbouw van R2) en boeken 8.837 (4,1% van de voorbeelden);
  103.897.723 tokens, 44.144.213 doeltokens (boeken 4.666.527 tokens, 249 doeltokens per voorbeeld; BT 201); eval 4.167
  BT-passages. Mediane verhouding woorden NL / EN (tekens): hele set 0,968 (1,006), BT 0,967 (1,004), boeken 0,990
  (1,037). Gecontroleerd met `sft_build.py check` (MaxText's eigen SFT-stappen op elk voorbeeld; de boekvoorbeelden met
  de token-id's van TranslateGemma's eigen prompt): EXIT 0, train 217.563, eval 4.167, max 2.048 tokens
  (`$S/data/rb/check.txt`, 11 nov 12:31).
- **Training** gelijk aan N en R2 [besluit Yeb]: LoRA rank 16, alpha 16, op attentie en MLP, LR 1e-5 cosine naar 0 met
  10% warmup, één epoch (`step_limit`), sequenties van 2.048, 1 per chip, 16 chips. Uitvoer `$S/rb/runs/arm27b_rb_RB/`,
  adapter `$S/rb/RB/`. De keten (`translator_sft_rb_run.sh`, `LORA_RANK`, `LORA_ALPHA`, `LR`) weigert te starten als de
  regel hieronder niet letterlijk gelijk is aan zijn startwaarden, en controleert na de training de rank in
  `adapter.json`.

RB-instellingen: lora_rank=16 lora_alpha=16 learning_rate=1e-5

- **Evaluatie en beslisregel gelijk aan die van B, N en R2** (§4, §5; het amendement R2), tegen hetzelfde uitgangspunt
  (`$E/baseline-tg27b/scores.parquet`): de vier sets met dezelfde code (MetricX-24 met referentie en QE, COMET-22; niet
  af, lus, Engels over, leeg); de extra sets van soort `x` (FLAN 1.000, blind 100, in-domein 4.519 van `$S/arms/data/x/`,
  het basismodel uit `$S/arms/eval/base-x/`); de classifier van 10 nov (`$S/arms/naturalness/classifier/`, 0,9575; niet
  opnieuw getraind) op de WMT24++ (960) en de 1.000 FLAN-rijen, basismodel en RB in één classify-run; PLL en MTLD; het
  blinde pakket `$S/blind/RB/` (A/B met seed `en-nl-sft/blind/ab/RB/<blind_id>`, sleutel alleen in de bucket); het
  rapport `$S/rb/report/report.md` met regel (a), (b), de stand van (c) en, gerapporteerd zonder mee te beslissen, de
  lengte van de uitvoer tegenover de menselijke referentie op de vier sets (`armeval.py report --four-texts`). (a)
  MetricX-24-QE(RB) − QE(basis) ≤ 0,20 op WMT24++ (960), FLORES-200 en NTREX-128; (b) aandeel "origineel" ≥ basis +
  0,10; (c) de blinde beoordeling, alleen als (a) en (b) gelden. RB helpt als (a), (b) en (c) alle drie gelden.

**Amendement 11 nov 2026 ~21:45 CET: arm RB2, BT-paren en boeken 80 : 20, met de grootste LoRA-rank die past**
[besluit Yeb 11 nov ~21:15; vastgelegd vóór de geheugentest en vóór de eerste trainingsstap van RB2; RB blijft zoals
hij is]. Aanleiding: RB haalt (a) niet (MetricX-24-QE +1,62 / +0,75 / +1,03 op WMT24++ (960) / FLORES / NTREX,
`$S/rb/report/report.md`). De uitvoer van RB heeft per Engels bronwoord mediaan 1,000 Nederlandse woorden, terwijl
de menselijke referenties er 1,05–1,10 hebben; 8.837 boekparen (4,1% van de voorbeelden) waren te weinig om de
compressie van de BT-paren te corrigeren. RB2 geeft de boeken een vijfde van de voorbeelden en traint met een grotere
rank. Eén nieuwe arm, **RB2**.

- **Trainingsdata** (`$S/data/rb2/RB2/`; `sft_build.py build --bt all --qe-max 5 --len-factor 1.3333 --block-jsonl
  blind100.jsonl --real real-qe.parquet --real-all --real-qe-max 7.5 --bt-per-real 4 --sets RB2=none+real`; keten
  `run_arms.sh rb2build`, `rb2check`, code `ae66022`):
  1. **Boeken** [besluit Yeb]: alle boekparen van `yhavinga/nedd_x_dataset_nlbooks_b1` zoals bij RB (punt 2 van het
     amendement RB: zonder de vijf auteurs die niet in het Nederlands schreven, de harde controles, de lengtegrens
     factor 4/3 rond de mediaan van de boeken, geen 13-gram met een evaluatieset of blinde passage, ≤ 2.048 tokens,
     TranslateGemma's eigen prompt zonder instructiezin), maar met **MetricX-24-QE en→nl ≤ 7,5** in plaats van 5. De
     kandidaten en hun QE-scores zijn die van RB (`$S/data/rb-real/candidates/real.parquet`,
     `$S/data/rb-real/qe/metricx-w{0..3}.parquet`); niets wordt opnieuw opgehaald of gescoord. Verwacht volgens de
     kosten van RB [schatting]: boven 7,5 liggen 941 van de 12.883 paren die door de harde controles komen (boven 5:
     4.033), dus ~3.100 paren meer dan bij RB, ~11.900 samen.
  2. **BT-paren** [besluit Yeb]: de BT-voorbeelden die bij RB en R2 in aanmerking komen (§2.1 met QE ≤ 5, de lengtegrens,
     geen 13-gram met een evaluatieset of blinde passage, en de contextvoorbeelden volgens §2.2 zoals bij `--bt all`),
     daaruit willekeurig getrokken met een vaste seed (`en-nl-sft/0/bt-per-real`) **4 × het aantal boekparen**, zodat
     BT : boeken = **80 : 20** in voorbeelden. De trekking is uniform over alle BT-voorbeelden die in aanmerking komen
     [keuze: niet per bron gestratificeerd; elke bron houdt in verwachting haar aandeel; een passage die met haar context
     getrokken wordt, blijft een contextvoorbeeld, zodat het aandeel contextvoorbeelden in verwachting dat van RB is,
     6,0%].
  3. **Geen** Europarl, **geen** Global Voices, **geen** FLAN-replay, nergens een instructiezin [besluit Yeb].
- **De set** [gemeten 11 nov ~22:00, `$S/data/rb2/report.md`, code `8fed60b`; vóór de geheugentest en de eerste
  trainingsstap]: boeken 15.004 rijen → niet in het Nederlands geschreven 2.025 → harde controles (Engels over 80,
  lengte > 1,5 15, kort 1) → 12.883 → **MetricX-QE > 7,5: 941** → **lengtegrens 19** (mediaan 1,033: 0,774–1,377) →
  13-gram 0, te lang 0 → **11.923 boekparen**. BT: **47.692** = 4 × 11.923, getrokken uit de 208.726 van RB (2.937 met
  context, 6,2%). Samen **59.615 trainvoorbeelden** (boeken 20,0%), 29.092.196 tokens (boeken 6.359.249, 21,9%),
  12.588.433 doeltokens (boeken 2.998.433, 23,8%); eval 4.167 BT-passages.

  | bron | voorbeelden | tokens | doeltokens | doeltokens per voorbeeld |
  |---|---:|---:|---:|---:|
  | boeken | 11.923 | 6.359.249 | 2.998.433 | 251 |
  | Handelingen | 18.344 | 10.192.443 | 3.798.828 | 207 |
  | FineWeb-2 | 12.366 | 5.261.986 | 2.388.377 | 193 |
  | raad | 9.578 | 4.048.636 | 1.887.878 | 197 |
  | dpc | 3.708 | 1.519.316 | 695.659 | 188 |
  | rechtspraak | 3.696 | 1.710.566 | 819.258 | 222 |

  Mediane verhouding woorden NL / EN (tekens): hele set 0,972 (1,010), BT 0,967 (1,004), boeken 0,988 (1,034).
  Gecontroleerd met `sft_build.py check` (MaxText's eigen SFT-stappen op elk voorbeeld): EXIT 0, train 59.615, eval
  4.167, max 2.038 tokens (`$S/data/rb2/check.txt`). Verwachte epoch [schatting, tokens per stap als bij RB]: ~950
  stappen.
- **Training** zoals RB (§3 en het amendement RB: LoRA op attentie en MLP, LR 1e-5 cosine naar 0 met 10% warmup, één
  epoch (`step_limit`), sequenties van 2.048, 1 per chip, 16 chips), met een andere rank en alpha. Uitvoer
  `$S/rb2/runs/arm27b_rb2_RB2/`, adapter `$S/rb2/RB2/`.
  - **Keuzeregel voor de rank** [besluit Yeb, vastgelegd vóór de test]: een geheugentest van 8 stappen op de set RB2,
    eerst met rank 128 en alpha 64, dan met rank 64 en alpha 32, verder gelijk aan de training (multi-host over de 16
    chips, `per_device_batch_size=1`, `max_target_length=2048`; runs `$S/rb2/runs/arm27b_rb2m128_RB2/` en
    `…/arm27b_rb2m64_RB2/`). Maat: de piek per chip, `peak_bytes_in_use` van `memory_stats` na de 8 stappen
    (`peak_gib_max` in de regel `SUMMARY`), het maximum over de 16 chips. **Gekozen wordt de grootste rank waarvan die
    piek ≤ 29,5 GiB is** (marge onder de grens van 30,75 GiB). Een test die faalt op geheugen (`RESOURCE_EXHAUSTED` of
    "out of memory" in het log van een worker) telt als "past niet". **Past geen van beide, dan rank 32 met alpha
    16.** Faalt een test om een andere reden, dan stopt de keten zonder keuze (`FAILED`).
  - **Alpha = rank/2** [besluit Yeb]. Yeb motiveerde dat met rsLoRA: alpha/√rank gelijk aan die van rank 16 met alpha
    16 (16/√16 = 4). Dat geldt precies voor rank 64 (32/√64 = 4); bij rank 128 is 64/√128 ≈ 5,66 en bij rank 32 is
    16/√32 ≈ 2,83 [eigen berekening]. MaxText/qwix schaalt de LoRA-bijdrage met alpha/rank (`adapter.json`, `scale`):
    met alpha = rank/2 is die schaal bij elke rank 0,5, tegen 1,0 bij rank 16 met alpha 16. Dit protocol volgt de
    letterlijke waarden van het besluit (alpha = rank/2; terugval rank 32 met alpha 16).
  - De keten (`mbert-ops/scripts/translator_sft_night2_run.sh`) voert de test uit, kiest de rank volgens deze regel,
    logt beide pieken en de keuze (regel `RANK`), traint RB2 en daarna R2 met die rank, en controleert na elke
    training rank en alpha in `adapter.json`. Hij weigert te starten als de regel hieronder niet letterlijk gelijk is
    aan de zijne.

RB2-instellingen: lora_rank=geheugentest(128,64; piek per chip <= 29.5 GiB; anders 32) lora_alpha=rank/2 learning_rate=1e-5

- **Evaluatie en beslisregel gelijk aan die van B, N, RB en R2** (§4, §5; de amendementen R2 en RB), tegen hetzelfde
  uitgangspunt (`$E/baseline-tg27b/scores.parquet`): de vier sets met dezelfde code (MetricX-24 met referentie en QE,
  COMET-22; niet af, lus, Engels over, leeg); de extra sets van soort `x` (FLAN 1.000, blind 100, in-domein 4.519 van
  `$S/arms/data/x/`, het basismodel uit `$S/arms/eval/base-x/`); de classifier van 10 nov
  (`$S/arms/naturalness/classifier/`, 0,9575; niet opnieuw getraind) op de WMT24++ (960) en de 1.000 FLAN-rijen,
  basismodel en RB2 in één classify-run; PLL en MTLD; het blinde pakket `$S/blind/RB2/` (A/B met seed
  `en-nl-sft/blind/ab/RB2/<blind_id>`, sleutel alleen in de bucket); het rapport `$S/rb2/report/report.md` met regel
  (a), (b), de stand van (c) en, gerapporteerd zonder mee te beslissen, de lengte van de uitvoer tegenover de menselijke
  referentie op de vier sets. (a) MetricX-24-QE(RB2) − QE(basis) ≤ 0,20 op WMT24++ (960), FLORES-200 en NTREX-128
  (grenzen 1,747 / 1,022 / 1,282); (b) aandeel "origineel" ≥ basis + 0,10; (c) de blinde beoordeling, alleen als (a)
  en (b) gelden. RB2 helpt als (a), (b) en (c) alle drie gelden.

**Amendement 11 nov 2026 ~21:45 CET: R2 met de rank van RB2** [besluit Yeb 11 nov ~21:15; vastgelegd vóór de start van
R2]. R2 met rank 16 (regel `R2-instellingen` van `9082ee6`, 11 nov ~10:45) vervalt. R2 traint na RB2, in dezelfde
keten, met de rank die de geheugentest van RB2 kiest (amendement RB2 hierboven) en **alpha = rank/2**, LR 1e-5; data,
grenzen, training verder, evaluatie en beslisregel zoals het amendement R2. De regel `R2-instellingen` in het amendement
R2 is daarom aangepast: de waarden volgen uit de geheugentest; de keten vult ze in, weigert te starten als die regel
niet letterlijk gelijk is aan de zijne, en controleert na de training rank en alpha in `adapter.json` van
`$S/r2/R2/`.

**Amendement 12 nov 2026 ~13:10 CET: armen RB3 en RBR, meer boeken en de echte paren van R2 erbij** [besluit van de
hoofdsessie namens Yeb, 12 nov ~13:10, genomen terwijl Yeb afwezig was en zonder vragen aan hem; vastgelegd vóór de
bouw van de sets en vóór de eerste trainingsstap van RB3 en RBR; RB2 en R2 blijven zoals ze zijn]. Aanleiding: RB2
haalt (a) niet (MetricX-24-QE +1,131 / +0,555 / +0,684 op WMT24++ (960) / FLORES / NTREX, `$S/rb2/report/report.md`)
en (b) wel (+15,5 pp). Tot nu toe hielp een groter aandeel boeken het meest (RB met 4% boeken +1,62 / +0,75 / +1,03,
RB2 met 20% +1,13 / +0,55 / +0,68), maar de uitvoer heeft per Engels bronwoord nog steeds mediaan ~1,00 Nederlands
woord, tegen 1,05–1,10 bij de menselijke referenties. Twee nieuwe armen, in deze volgorde, na de nachtketen van 11/12
nov (RB2 en R2) en op dezelfde VM (mbert-mix): **RB3** en daarna, alleen onder de voorwaarde hieronder, **RBR**.

- **Arm RB3: BT : boeken = 50 : 50** (`$S/data/rb3/RB3/`; `sft_build.py build --bt all --qe-max 5 --len-factor 1.3333
  --block-jsonl blind100.jsonl --real real-qe.parquet --real-all --real-qe-max 7.5 --bt-per-real 1 --sets RB3=none+real`;
  keten `run_arms.sh rb3build`, `rb3check`):
  1. **Boeken**: dezelfde boekparen als RB2 (punt 1 van het amendement RB2: de kandidaten en QE-scores van RB in
     `$S/data/rb-real/`, MetricX-24-QE en→nl ≤ 7,5, de lengtegrens, geen 13-gram met een evaluatieset of blinde
     passage, ≤ 2.048 tokens; verwacht dezelfde 11.923).
  2. **BT-paren**: **1 per boekpaar**, uniform getrokken met de vaste seed `en-nl-sft/0/bt-per-real` uit de
     BT-voorbeelden die bij RB2 in aanmerking kwamen (de 208.726: §2.1 met QE ≤ 5, de lengtegrens, contextvoorbeelden
     volgens §2.2 zoals bij `--bt all`; een passage die met haar context getrokken wordt, blijft een contextvoorbeeld).
     `random.sample` met een andere omvang geeft een andere trekking: het BT-deel van RB3 is geen deelverzameling van
     dat van RB2 [gevolg van `--bt-per-real 1`, geen aparte keuze].
  3. Geen Europarl, Global Voices, FLAN-replay of instructiezin.
- **Arm RBR: BT 60% + boeken 20% + Europarl/Global Voices 20%** (`$S/data/rbr/RBR/`; dezelfde build met
  `--bt-per-real 3 --real2 r2-real-qe.parquet --real2-from r2-train-meta.parquet --real2-per-real 1 --sets
  RBR=none+real`; keten `run_arms.sh rbrbuild`, `rbrcheck`; nieuwe code: `sft_build.py --real2`, `--real2-from`,
  `--real2-per-real`, met tests):
  1. **Boeken** als RB2 en RB3 (11.923 verwacht).
  2. **De echte paren van R2**: de echte voorbeelden van de set R2 zoals die getraind is (`$S/data/r2/R2/train-meta.parquet`,
     soort `real`: 11.412 Global Voices, 40.770 Europarl-direct 2004–2011; `$S/data/r2/report.md`). In de build van
     RBR gaan de kandidaten van R2 (`$S/data/r2-real/candidates/real.parquet` met hun QE uit `$S/data/r2-real/qe/`)
     opnieuw door de query van R2 (harde controles, MetricX-24-QE ≤ 5, de lengtegrens factor 4/3 rond de mediaan van
     elke bron, geen 13-gram met een evaluatieset of blinde passage, ≤ 2.048 tokens); elk van de 52.182 moet daar
     doorheen komen, anders stopt de build (dan verschillen de grenzen van die van R2). Daaruit **zoveel dat ze 20% van
     de voorbeelden zijn: 1 per boekpaar**, **uniform** getrokken met de seed `en-nl-sft/0/real2` [keuze: uniform uit
     de echte paren van R2, zodat Global Voices en Europarl in verwachting hun aandeel in R2 houden (21,9% en 78,1%);
     niet de regel van R2 "eerst alle Global Voices, Europarl vult aan", want die zou bij ~11.900 paren ~96% Global
     Voices geven, een menging die R2 niet getest heeft, terwijl de voorwaarde hieronder op de uitslag van R2 steunt].
  3. **BT-paren**: **3 per boekpaar**, getrokken zoals bij RB2 en RB3, zodat BT : boeken : R2-echt = 60 : 20 : 20 in
     voorbeelden (in tokens niet: boekpassages zijn lang, de echte paren van R2 kort; het rapport van de set geeft de
     tokens per bron).
  4. Geen FLAN-replay of instructiezin.
- **Voorwaarde voor RBR** [besluit hoofdsessie]: RBR traint alleen als R2 op regel (a), **gemiddeld over de drie sets**
  (WMT24++ (960), FLORES-200, NTREX-128), **minstens zo goed** is als RB2: het gemiddelde van de drie verschillen
  MetricX-24-QE(R2) − QE(basis) ≤ dat van RB2 (lager is beter; gelijk telt als "minstens zo goed"). De verschillen
  komen onafgerond uit `report.json` van beide rapporten (`$S/r2/report/report.json` en `$S/rb2/report/report.json`,
  veld `rule_a.<arm>.sets.<set>.delta`); voor RB2 is het gemiddelde (1,1315 + 0,5547 + 0,6843) / 3 = 0,7902 [berekend
  uit `report.json`]. De keten leest dat na RB3 en schrijft de vergelijking in zijn log; anders slaat hij RBR over met
  een logregel `RBR overgeslagen: …`. De set RBR wordt wel vooraf gebouwd en gecontroleerd (CPU), zodat de keten bij
  een geldige voorwaarde meteen kan trainen.

RBR-voorwaarde: gemiddelde regel (a) R2 <= gemiddelde regel (a) RB2 (wmt24pp, flores, ntrex; report.json, onafgerond)

- **De sets** [gemeten 12 nov ~13:40, `$S/data/rb3/report.md` en `$S/data/rbr/report.md`, code `8d8946d`; vóór de
  eerste trainingsstap; gebouwd op w3 en gecontroleerd op w2 van mbert-mix, alleen CPU, terwijl R2 trainde]:
  - **Boeken** (beide sets, gelijk aan RB2): 15.004 rijen → niet in het Nederlands geschreven 2.025 → harde controles
    96 → 12.883 → MetricX-QE > 7,5: 941 → lengtegrens 19 → 13-gram 0, te lang 0 → **11.923**.
  - **RB3**: BT **11.923** = 1 × 11.923, getrokken uit de 208.726 van RB2 (733 met context, 6,1%). Samen **23.846
    trainvoorbeelden** (boeken 50,0%), 12.024.479 tokens (boeken 6.359.249, 52,9%), 5.393.084 doeltokens (boeken
    2.998.433, 55,6%); eval 4.167 BT-passages. Mediane verhouding woorden NL / EN (tekens): hele set 0,978 (1,020), BT
    0,967 (1,004), boeken 0,988 (1,034). Gecontroleerd met `sft_build.py check` (MaxText's eigen SFT-stappen op elk
    voorbeeld): EXIT 0, train 23.846 (bt 11.190, bt_context 733, real 11.923), 12.024.479 tokens, max 2.038; eval
    4.167, max 2.006 (`$S/data/rb3/check.txt`).
  - **RBR**: van de kandidaten van R2 komen er 96.284 door de query (Global Voices 11.412, Europarl 84.872, gelijk aan de
    build van R2), en **alle 52.182 echte voorbeelden van R2 halen hem opnieuw**; daaruit **11.923** uniform: Europarl
    9.329 (78,2%), Global Voices 2.594 (21,8%) (in R2 78,1% / 21,9%). BT **35.769** = 3 × 11.923 (2.186 met context,
    6,1%). Samen **59.615 trainvoorbeelden**: BT 60,0%, boeken 20,0%, R2-echt 20,0%; 26.435.452 tokens (BT 17.019.275,
    64,4%; boeken 6.359.249, 24,1%; R2-echt 3.056.928, 11,6%), 11.463.319 doeltokens (62,6 / 26,2 / 11,2%); eval 4.167.
    Mediane verhouding woorden NL / EN (tekens): hele set 0,985 (1,028), BT 0,967 (1,004), boeken 0,988 (1,033), R2-echt
    1,062 (1,138) (in R2 1,062 (1,139)). Gecontroleerd met `sft_build.py check`: EXIT 0, train 59.615 (bt 33.583, bt_context
    2.186, real 23.846), 26.435.452 tokens, max 2.038; eval 4.167, max 2.006 (`$S/data/rbr/check.txt`).
  - Verwachte epoch [schatting, tokens per stap als bij RB2: 29,09 M tokens in 950 stappen]: RB3 ~390, RBR ~860
    stappen.
- **Training** van beide armen zoals RB2 en R2: LoRA op attentie en MLP, **rank 64, alpha 32**, LR 1e-5 cosine naar 0
  met 10% warmup, **één epoch** (`step_limit`), sequenties van 2.048, 1 per chip, **16 chips** [besluit hoofdsessie;
  rank 64 is de rank die de geheugentest van RB2 koos: piek 27,16 GiB per chip, rank 128 `oom`]. Geen nieuwe
  geheugentest: een stap heeft dezelfde vorm als in de test van RB2 (gepakte sequenties van 2.048, 1 per chip) en
  dezelfde rank.
  Uitvoer `$S/rb3/runs/arm27b_rb3_RB3/` en `$S/rbr/runs/arm27b_rbr_RBR/`, adapters `$S/rb3/RB3/` en `$S/rbr/RBR/`. De
  keten (`mbert-ops/scripts/translator_sft_followup_run.sh`) weigert te starten als de regels hieronder niet letterlijk
  gelijk zijn aan de zijne, en controleert na elke training rank en alpha in `adapter.json`.

RB3-instellingen: lora_rank=64 lora_alpha=32 learning_rate=1e-5
RBR-instellingen: lora_rank=64 lora_alpha=32 learning_rate=1e-5

- **Evaluatie en beslisregel gelijk aan die van B, N, RB, RB2 en R2** (§4, §5; de amendementen R2 en RB), tegen
  hetzelfde uitgangspunt (`$E/baseline-tg27b/scores.parquet`), per arm: de vier sets met dezelfde code (MetricX-24 met
  referentie en QE, COMET-22; niet af, lus, Engels over, leeg); de extra sets van soort `x` (FLAN 1.000, blind 100,
  in-domein 4.519 van `$S/arms/data/x/`, het basismodel uit `$S/arms/eval/base-x/`); de classifier van 10 nov
  (`$S/arms/naturalness/classifier/`, 0,9575; niet opnieuw getraind) op de WMT24++ (960) en de 1.000 FLAN-rijen,
  basismodel en arm in één classify-run; PLL en MTLD; het blinde pakket `$S/blind/RB3/` en `$S/blind/RBR/` (A/B met
  seed `en-nl-sft/blind/ab/<arm>/<blind_id>`, sleutel alleen in de bucket); het rapport `$S/rb3/report/report.md` en
  `$S/rbr/report/report.md` met regel (a), (b), de stand van (c) en, gerapporteerd zonder mee te beslissen, de lengte
  van de uitvoer tegenover de menselijke referentie op de vier sets. (a) MetricX-24-QE(arm) − QE(basis) ≤ 0,20 op
  WMT24++ (960), FLORES-200 en NTREX-128 (grenzen 1,747 / 1,022 / 1,282); (b) aandeel "origineel" ≥ basis + 0,10; (c)
  de blinde beoordeling, alleen als (a) en (b) gelden. Een arm helpt als (a), (b) en (c) alle drie gelden.
- **Tijd** [keuze, voor de evaluatie van T2 op 13 nov ~17:00 op dezelfde VM]: de keten laat mbert-mix vóór 13 nov
  ~17:00 vrij. Hij start de training van een arm alleen als die arm met evaluatie naar schatting vóór 13 nov 16:30 CET
  klaar is (RB3 ~2,5 uur, RBR ~3,5 uur gerekend vanaf de start van de training); anders slaat hij de arm over met een
  logregel. Verwacht: de nachtketen klaar ~19:15 op 12 nov, RB3 klaar ~21:30, RBR ~00:30.

**Amendement 12 nov 2026 ~15:20 CET: arm RL, BT-paren hersampled naar de lengteverhouding van menselijke vertalingen,
en boeken zonder QE-grens** [besluit Yeb 12 nov ~14:45; de uitwerking (methode en bakken van de trekking, de
uitlijncontrole en haar grens) door de subagent RL; vastgelegd vóór de bouw van de set en vóór de eerste trainingsstap
van RL; RB2, R2, RB3 en RBR blijven zoals ze zijn]. Aanleiding: de BT-paren hebben synthetisch Engels dat omslachtiger is
dan het Nederlandse origineel: mediaan 0,967 Nederlandse woorden per Engels woord in de 208.726 BT-voorbeelden van RB en
RB2, tegen 1,0625 in de echte voorbeelden van de trainingsset van R2 (Europarl-direct 1,065, Global Voices 1,050)
[gemeten 12 nov]. Elke arm leert daardoor comprimeren. RB2 is de beste arm tot nu toe (MetricX-24-QE +1,131 / +0,555 /
+0,684 op WMT24++ (960) / FLORES / NTREX), maar zijn uitvoer heeft nog steeds ~1,00 woord per bronwoord. Eén nieuwe arm,
**RL**, op mbert-mix na de vervolgketen van 12/13 nov (RB3, RBR).

- **Trainingsdata** (`$S/data/rl/RL/`; `sft_build.py build --bt all --qe-max 5 --len-factor 1.3333 --block-jsonl
  blind100.jsonl --real real-qe.parquet --real-all --real-qe-unbounded --real-align-max 4.39 --bt-per-real 4
  --bt-match-set r2-set --bt-match-bins 10 --sets RL=none+real`; keten `run_arms.sh rlbuild`, `rlcheck`; nieuwe code
  `1db9957`, met tests):
  1. **Boeken zonder QE-grens** [besluit Yeb: "move towards no QE filter on the books"]: alle boekparen van RB (kandidaten
     en QE in `$S/data/rb-real/`, zonder de vijf auteurs die niet in het Nederlands schreven), met de harde controles,
     de lengtegrens van R2 (factor 4/3 rond de mediane tekenverhouding van de boeken, 1,0326: 0,7745–1,3768), geen 13-gram
     met een evaluatieset of blinde passage, ≤ 2.048 tokens en TranslateGemma's eigen prompt; **geen MetricX-24-QE-grens**
     (de QE wordt gelezen en gerapporteerd, niet gebruikt), en in plaats daarvan **de uitlijncontrole** van punt 2.
  2. **Uitlijncontrole** [keuze, binnen het besluit "vang de uitlijnfouten op met een structurele controle"]. De
     uitlijnfouten bij QE 23–25 zijn verschoven passages: het Nederlands begint met een of twee zinnen die in het Engels
     ontbreken, en vaak eindigt het Engels met een zin die in het Nederlands ontbreekt; de lengteverhouding van het geheel
     blijft gewoon [gelezen: de zes paren met QE 23–25].
     - **Zinnen**: per regel geknipt na `.`, `!`, `?` of `…` (met afsluitende aanhalingstekens, haakjes, `_` of `*`),
       gevolgd door witruimte en een hoofdletter, een cijfer of een openend teken; niet na een afkorting (Mr, Mrs, Ms, Dr,
       St, Prof, Jr, Sr, Lt, Col, Gen, Capt, Rev, vs, mevr, dhr, drs, ir, ing, bijv, nr, ca, jhr, mgr), een initiaal of
       een los getal (het nummer van een noot).
     - **Uitlijning naar lengte in tekens** volgens Gale en Church (1993, "A Program for Aligning Sentences in Bilingual
       Corpora", *Computational Linguistics* 19(1)): paren van 1-1, 1-0, 0-1, 2-1, 1-2 en 2-2 zinnen met hun
       a-priorikansen 0,89 / 0,0099 (1-0 en 0-1 samen, gelijk verdeeld) / 0,089 (2-1 en 1-2 samen, gelijk verdeeld) /
       0,011 en de variantie 6,8 per teken; de verwachte verhouding is de mediane tekenverhouding van de bron (boeken
       1,0326) in plaats van hun 1. Kosten van een paar zinnen = −log a-priorikans − log P(lengteverschil); de
       goedkoopste uitlijning door dynamisch programmeren (`sft_build.py`: `sentences`, `sentence_alignment`,
       `align_edges`).
     - **Maat**: het gemiddelde van de kosten van de eerste twee paren zinnen (begin) en van de laatste twee (eind).
       **Een boekpaar valt af als begin of eind boven 4,39 ligt.**
     - **Grens uit de verdeling** [keuze]: 4,39 is p98 (4,388) van die maat over de 11.942 boekparen met QE ≤ 7,5 die
       door de harde controles komen (de boeken van RB2 vóór de lengtegrens): de controle neemt 2% van die paren weg.
       Gekozen na het lezen van paren rond p95–p98 (met een eerdere versie van de zinsknipper, nog zonder de
       afkortingen): daarboven vooral verschoven passages, eronder steeds meer correcte vertalingen die de vertaler
       anders in zinnen knipte.
     - **Getoetst** [gemeten 12 nov ~15:00 op w3 van mbert-mix, met de code van `1db9957`; tegengehouden / paren per
       QE-band, van de 12.883 boekparen die door de harde controles komen]:

       | MetricX-QE | ≤ 5 | 5–7,5 | 7,5–10 | 10–15 | 15–20 | 20–23 | 23–25 |
       |---|---:|---:|---:|---:|---:|---:|---:|
       | tegengehouden | 158 / 8.850 (1,8%) | 81 / 3.092 (2,6%) | 41 / 641 (6,4%) | 39 / 242 (16%) | 15 / 38 (39%) | 6 / 14 | **4 / 6** |

       Van de zes met QE 23–25 houdt de controle de vier verschoven passages tegen (Mulisch `books:74768-137379-b1-950`,
       Hermans `books:59396-137374-b1-217`, Koch `books:42430-137372-b1-223` en `…-533`); ze laat er twee door: Van Iperen
       `books:74080-137392-b1-427` (begin en eind kloppen; in het Engels staat een zin twee keer) en Wieringa
       `books:46782-137377-b1-212` (het Nederlands mist aan het begin "She had smiled and said" en heeft een losse regel
       "cover: eve lesage back in the limelight."). Van tien willekeurig gelezen tegengehouden paren met QE 5–7,5 (seed
       `en-nl-sft/rl/align-read`) zijn er vijf verschoven of aan een rand anders (het Nederlands begint of eindigt met een
       zin die in het Engels ontbreekt, of omgekeerd; één daarvan een notenlijst met andere nummers), drie correct maar
       anders in zinnen geknipt (een samengevoegde zin, een beletselteken gevolgd door een hoofdletter, dialoog met
       gedachtestreepjes) en twee onduidelijk. De controle kost op de "meestal correcte" paren met QE 5–7,5 dus 2,6%,
       waarvan ongeveer de helft zelf uitlijnfouten zijn.
     - **Het aantal zinnen** (een van Yebs voorbeelden) gebruikt de controle niet: het verschil scheidt niet (≥ 2 zinnen
       verschil bij QE ≤ 5 15,6%, bij 5–7,5 16,2%, bij 23–25 1 van de 6) [gemeten, met de eerdere zinsknipper]; de
       uitlijning vangt samengevoegde en gesplitste zinnen op met 2-1 en 1-2. De lengtegrens van R2 blijft (punt 1).
       **Opnieuw uitlijnen** op zinsniveau niet [keuze]: de uitlijning naar lengte ziet een verschoven rand, maar is in
       literaire tekst (dialoog, samengevoegde zinnen) niet betrouwbaar genoeg om zinnen weg te knippen.
     - Verwacht [schatting uit de meting]: 12.883 → lengtegrens 45 → uitlijning ~340 → **~12.520 boekparen** (RB2:
       11.923).
  3. **BT-paren hersampled naar de lengteverhouding van menselijke vertalingen** [besluit Yeb]: uit de BT-voorbeelden die
     bij RB2 in aanmerking kwamen (de 208.726: §2.1 met QE ≤ 5, de lengtegrens van R2 en de contextvoorbeelden volgens
     §2.2 zoals bij `--bt all`; een passage die met haar context getrokken wordt, blijft een contextvoorbeeld) een
     steekproef waarvan de verdeling van het aantal Nederlandse woorden per Engels woord (split op witruimte, zoals in de
     rapporten van de sets) die van de **echte voorbeelden van de trainingsset van R2** volgt
     (`$S/data/r2/R2/train.parquet` en `train-meta.parquet`, soort `real`: 52.182, Europarl-direct 2004–2011 40.770 en
     Global Voices 11.412; mediaan 1,0625), **niet** die van de evaluatiesets.
     - **Methode** [keuze binnen het besluit]: histogram matching op log(woorden NL / woorden EN), met **10 bakken**
       begrensd door de decielen van het doel (de buitenste twee open; gelijke grenzen vallen samen); bak b krijgt n × zijn
       aandeel in het doel (grootste rest), **zonder teruglegging** uniform getrokken met de vaste seed
       `en-nl-sft/0/bt-match/<b>` (`sft_build.py`: `match_target`, `bt_match_draw`). Een bak van het doel zonder
       BT-voorbeelden is een fout.
     - **Aantal** [besluit Yeb]: n = 4 × het aantal boekparen als elke bak dat toelaat; anders de grootste n waarvoor
       elke bak zijn deel bevat.
     - **Wat de verdeling toelaat** [gemeten: een proef met deze code op de 208.726 BT-voorbeelden van RB, met 4 × 12.520
       = 50.080 gewenst]: **hoogstens 7.267** (0,58 per boekpaar). De bovenste deciel van het doel (meer dan 1,222 woord
       per woord) heeft onder de BT-voorbeelden maar 728 voorbeelden (0,35%), en die begrenzen n. Ter vergelijking: met
       20 bakken 4.486, met 5 bakken 12.763 (dan ligt de mediaan van de trekking op 1,057). Dus **zoveel als past: ~7.270
       BT-voorbeelden**, BT : boeken ≈ **37 : 63** in voorbeelden, in plaats van 80 : 20 bij RB2 [gevolg van de regel
       "anders zoveel als past"; hierbij gemeld].
     - De getrokken verdeling in die proef (woorden NL / EN, p5 / p10 / p25 / p50 / p75 / p90 / p95): **0,886 / 0,927 /
       1,000 / 1,0625 / 1,138 / 1,222 / 1,250**; doel 0,889 / 0,927 / 1,000 / 1,0625 / 1,141 / 1,222 / 1,278; alle
       BT-voorbeelden 0,833 / 0,867 / 0,917 / 0,967 / 1,017 / 1,065 / 1,097. Bronnen in de trekking: Handelingen 46,7%,
       FineWeb-2 25,7%, raad 14,6%, rechtspraak 7,8%, dpc 5,2% (bij RB2 38,5 / 25,9 / 20,1 / 7,7 / 7,8%): bronnen met
       een hogere verhouding wegen zwaarder [gevolg; niet per bron gestratificeerd, want per bron gematcht past er
       hoogstens ~1.900]. Contextvoorbeelden 621 (8,5%).
     - Voorbehoud [eigen gevolgtrekking]: de echte voorbeelden van R2 zijn korter (mediaan 57 Engelse woorden, de
       BT-voorbeelden 107), dus hun verhouding spreidt meer; het doel volgen in alle decielen neemt daardoor ook
       BT-voorbeelden aan beide uiteinden (onderste deciel onder 0,927 woord per woord). In vijf gelezen BT-voorbeelden
       uit de bovenste bak was het Engels beknopt maar volledig, op één ingekorte bijzin na.
  4. Geen Europarl, Global Voices, FLAN-replay of instructiezin.
- **De set** [gemeten 12 nov ~15:20, `$S/data/rl/report.md` (kopie `mbert-ops/translator-sft/data-rl-report.md`), code
  `9fb6879`; vóór de eerste trainingsstap; gebouwd op w3 en gecontroleerd op w2 van mbert-mix, alleen CPU (`nice -n 19`,
  `ionice -c3`), terwijl R2 op de TPU trainde]:
  - **Boeken**: 15.004 rijen → niet in het Nederlands geschreven 2.025 → harde controles 96 (Engels over 80, lengte > 1,5
    15, kort 1) → 12.883 → lengtegrens 45 → **uitlijning 317** (343 als de 26 meetellen die ook de lengtegrens niet
    halen) → 13-gram 0, te lang 0 → **12.521 boekparen**, zonder QE-grens (QE van de 12.883: p50 4,00, p90 6,96, p99
    12,40). Tegengehouden door de uitlijning per QE-band: ≤ 5 158 / 8.850, 5–7,5 80 / 3.092, 7,5–10 41 / 641, 10–15
    39 / 242, 15–20 15 / 38, 20–23 6 / 14, 23–25 4 / 6 (bij 5–7,5 één paar minder dan in de meting hierboven, die
    p98 = 4,388 als grens nam in plaats van 4,39).
  - **BT**: gewenst 4 × 12.521 = 50.084, **hoogstens en getrokken 7.267** (0,58 per boekpaar), uit de 208.726; 621 met
    context (8,5%). Verdeling woorden NL / EN vóór (de 208.726) en na (de 7.267) de trekking, p10 / p50 / p90:

    | deel | n vóór | vóór | n na | na |
    |---|---:|---|---:|---|
    | doel: echte voorbeelden R2 | | | 52.182 | 0,927 / 1,062 / 1,222 (Europarl 0,932 / 1,065 / 1,221, Global Voices 0,909 / 1,050 / 1,222) |
    | BT samen | 208.726 | 0,867 / 0,967 / 1,065 | 7.267 | 0,927 / 1,062 / 1,222 |
    | Handelingen | 79.734 | 0,900 / 0,988 / 1,081 | 3.396 | 0,955 / 1,073 / 1,226 |
    | FineWeb-2 | 54.475 | 0,869 / 0,966 / 1,062 | 1.866 | 0,926 / 1,065 / 1,222 |
    | raad | 42.144 | 0,837 / 0,938 / 1,036 | 1.062 | 0,875 / 1,018 / 1,200 |
    | dpc | 16.478 | 0,829 / 0,931 / 1,027 | 376 | 0,863 / 1,000 / 1,149 |
    | rechtspraak | 15.895 | 0,875 / 0,970 / 1,067 | 567 | 0,933 / 1,065 / 1,227 |
    | boeken (niet getrokken) | | | 12.521 | 0,907 / 0,988 / 1,082 |

  - Samen **19.788 trainvoorbeelden** (boeken 63,3%), 9.899.934 tokens (boeken 6.687.192, 67,5%), 4.428.244 doeltokens
    (boeken 3.154.121, 71,2%); eval 4.167 BT-passages.

    | bron | voorbeelden | tokens | doeltokens | doeltokens per voorbeeld |
    |---|---:|---:|---:|---:|
    | boeken | 12.521 | 6.687.192 | 3.154.121 | 252 |
    | Handelingen | 3.396 | 1.744.997 | 604.350 | 178 |
    | FineWeb-2 | 1.866 | 693.237 | 310.572 | 166 |
    | raad | 1.062 | 393.810 | 180.424 | 170 |
    | rechtspraak | 567 | 237.434 | 113.349 | 200 |
    | dpc | 376 | 143.264 | 65.428 | 174 |

    De getrokken BT-voorbeelden zijn korter dan bij RB2 (175 doeltokens per voorbeeld, RB2 201) [gemeten]; vermoedelijk
    omdat kortere passages meer spreiden in hun verhouding en zo vaker in de buitenste bakken vallen [eigen
    gevolgtrekking].
  - Mediane verhouding woorden NL / EN (tekens): hele set 1,007 (1,042), BT 1,062 (1,063), boeken 0,988 (1,032); RB2
    0,972 (1,010).
  - Gecontroleerd met `sft_build.py check` (MaxText's eigen SFT-stappen op elk voorbeeld): EXIT 0, train 19.788 (bt
    6.646, bt_context 621, real 12.521), 9.899.934 tokens, max 2.034; eval 4.167, max 2.006 (`$S/data/rl/check.txt`).
  - Verwachte epoch [schatting, tokens per stap als bij RB2: 29,09 M tokens in 950 stappen]: ~325 stappen.
- **Training** zoals RB2, RB3 en RBR: LoRA op attentie en MLP, **rank 64, alpha 32**, LR 1e-5 cosine naar 0 met 10%
  warmup, **één epoch** (`step_limit`), sequenties van 2.048, 1 per chip, **16 chips** [besluit Yeb]. Uitvoer
  `$S/rl/runs/arm27b_rl_RL/`, adapter `$S/rl/RL/`. De keten (`mbert-ops/scripts/translator_sft_rl_run.sh`) weigert te
  starten als de regel hieronder niet letterlijk gelijk is aan de zijne, en controleert na de training rank en alpha in
  `adapter.json`. Verwachte epoch [schatting, tokens per stap als bij RB2]: ~10 M tokens, ~330 stappen.

RL-instellingen: lora_rank=64 lora_alpha=32 learning_rate=1e-5

- **Evaluatie en beslisregel gelijk aan die van B, N, RB, RB2, R2, RB3 en RBR** (§4, §5; de amendementen R2 en RB),
  tegen hetzelfde uitgangspunt (`$E/baseline-tg27b/scores.parquet`): de vier sets met dezelfde code (MetricX-24 met
  referentie en QE, COMET-22; niet af, lus, Engels over, leeg); de extra sets van soort `x` (FLAN 1.000, blind 100,
  in-domein 4.519 van `$S/arms/data/x/`, het basismodel uit `$S/arms/eval/base-x/`); de classifier van 10 nov
  (`$S/arms/naturalness/classifier/`, 0,9575; niet opnieuw getraind) op de WMT24++ (960) en de 1.000 FLAN-rijen,
  basismodel en RL in één classify-run; PLL en MTLD; het blinde pakket `$S/blind/RL/` (A/B met seed
  `en-nl-sft/blind/ab/RL/<blind_id>`, sleutel alleen in de bucket); het rapport `$S/rl/report/report.md` met regel (a),
  (b), de stand van (c) en, gerapporteerd zonder mee te beslissen, **de lengte van de uitvoer tegenover de menselijke
  referentie** op de vier sets. (a) MetricX-24-QE(RL) − QE(basis) ≤ 0,20 op WMT24++ (960), FLORES-200 en NTREX-128
  (grenzen 1,747 / 1,022 / 1,282); (b) aandeel "origineel" ≥ basis + 0,10; (c) de blinde beoordeling, alleen als (a) en
  (b) gelden. RL helpt als (a), (b) en (c) alle drie gelden.
- **Tijd** [keuze, voor de evaluatie van T2 op 13 nov ~17:00 op dezelfde VM]: de keten start pas na `EXIT 0` van de
  vervolgketen (`translator_sft_followup_run.sh`; een andere `EXIT` is `FAILED`), neemt dan `final_eval_mix.lock`, en
  start de training alleen als RL met evaluatie naar schatting vóór 13 nov 16:30 CET klaar is (150 min gerekend vanaf de
  start van de training); anders slaat hij RL over met een logregel. Verwacht: de vervolgketen klaar ~00:15 op 13 nov
  (met RBR) of ~21:20 (zonder), RL ~2 uur later.

**Aanvulling 12 nov 2026 ~15:35 CET op het amendement RL: 5 bakken in plaats van 10** [besluit van de hoofdsessie
namens Yeb (afwezig), op advies van een Fable-adviseur, 12 nov ~16:00 volgens de hoofdsessie (de klok van co1 gaf bij
het vastleggen 15:35); vastgelegd vóór de nieuwe bouw van de set en vóór de eerste trainingsstap van RL; de set met 10
bakken (`$S/data/rl/`) is niet getraind en blijft staan; de keten die op die set wachtte is gestopt]. Aanleiding: met
10 bakken liet het hersamplen maar 7.267 BT-voorbeelden toe in plaats van 4 × 12.521 = 50.084, zodat BT : boeken 37 : 63
werd in voorbeelden (32 : 68 in tokens). RL verschilde daardoor van RB3 ook in het aandeel boeken. Met 5 bakken (de
kwintielen van het doel) pasten er in de proef van het amendement 12.763 (mediaan 1,057), ongeveer evenveel als er
boekparen zijn: BT : boeken ≈ 50 : 50, zoals bij RB3. RL verschilt dan van RB3 (BT : boeken 50 : 50, BT uniform
getrokken, boeken met MetricX-QE ≤ 7,5) alleen in de lengteverdeling van de BT en in het filter op de boeken (geen
QE-grens, wel de uitlijncontrole).

- **Wat verandert**: in punt 3 van het amendement RL ("Methode") **5 bakken**, begrensd door de kwintielen van het doel,
  in plaats van 10 op de decielen. De rest van de methode blijft: log(woorden NL / woorden EN), het doel (de echte
  voorbeelden van de trainingsset van R2), de verdeling over de bakken naar hun aandeel in het doel (grootste rest),
  zonder teruglegging, de seeds `en-nl-sft/0/bt-match/<b>`, een bak van het doel zonder BT-voorbeelden is een fout.
  "Aantal" blijft: 4 × het aantal boekparen als elke bak dat toelaat, anders de grootste n waarvoor elke bak zijn deel
  bevat; verwacht ~12.760 [schatting uit de proef, die 12.520 boekparen nam].
- **Wat gelijk blijft**: de boeken (punt 1 en 2: zonder QE-grens, de uitlijncontrole met grens 4,39), geen Europarl,
  Global Voices, FLAN-replay of instructiezin; de training (rank 64, alpha 32, LR 1e-5, één epoch, 16 chips; de regel
  met de RL-instellingen hierboven verandert niet), de evaluatie, de beslisregel en de tijdgrens (13 nov 16:30 CET).
- **Code en plaats**: `b258cc4` (`run_arms.sh rlbuild` met `--bt-match-bins 5`, met tests); de set in
  **`$S/data/rl-b5/`**, een nieuwe plaats, zodat de eerste bouw in `$S/data/rl/` niet wordt overschreven. Het aantal
  bakken is een parameter (`RL_MATCH_BINS` in `run_arms.sh`, standaard 5; `sft_build.py --bt-match-bins`), zodat de
  versie met 10 bakken later kan worden gebouwd voor een arm "RL met QE-filter" [besluit van de hoofdsessie]; die arm
  is hiermee niet vastgelegd.
- De keten (`mbert-ops/scripts/translator_sft_rl_run.sh`) start opnieuw met een nieuwe tag, op de set in
  `$S/data/rl-b5/`, en weigert te starten zonder deze aanvulling en zonder `b258cc4`. De arm heet verder RL (paden
  `$S/rl/`, `$S/blind/RL/`).
- **De set met 5 bakken** [gemeten 12 nov ~15:40, `$S/data/rl-b5/report.md` (kopie
  `mbert-ops/translator-sft/data-rl-report.md`; die van de eerste bouw: `data-rl-b10-report.md`), code `3da9b70`;
  vóór de eerste trainingsstap; gebouwd op w3 (tag `rl11121433`) en gecontroleerd op w2 van mbert-mix, alleen CPU
  (`nice -n 19`, `ionice -c3`), terwijl R2 op de TPU trainde]:
  - **Boeken**: gelijk aan de eerste bouw, **12.521** (uitlijning 317, verder als hierboven).
  - **BT**: gewenst 4 × 12.521 = 50.084, **hoogstens en getrokken 12.763** (1,02 per boekpaar) uit de 208.726; 1.021
    met context (8,0%). De bovenste bak (boven 1,1613 woord per woord, 20,0% van het doel) heeft onder de BT-voorbeelden
    2.557 voorbeelden, en die zitten er alle 2.557 in. Grenzen van de bakken (kwintielen van het doel): 0,9737 /
    1,0351 / 1,0909 / 1,1613; getrokken per bak 2.547 / 2.556 / 2.537 / 2.566 / 2.557.
  - Verdeling woorden NL / EN (p5 / p10 / p25 / p50 / p75 / p90 / p95; vóór = de 208.726, na = de getrokken):

    | deel | n | p5 | p10 | p25 | p50 | p75 | p90 | p95 |
    |---|---:|---:|---:|---:|---:|---:|---:|---:|
    | doel: echte voorbeelden R2 | 52.182 | 0,889 | 0,927 | 1,000 | 1,062 | 1,141 | 1,222 | 1,278 |
    | BT samen vóór | 208.726 | 0,833 | 0,867 | 0,917 | 0,967 | 1,017 | 1,065 | 1,097 |
    | BT samen na (5 bakken) | 12.763 | 0,880 | 0,921 | 0,986 | 1,057 | 1,132 | 1,194 | 1,227 |
    | BT samen na (10 bakken, eerste bouw) | 7.267 | 0,886 | 0,927 | 1,000 | 1,062 | 1,138 | 1,222 | 1,250 |
    | Handelingen vóór / na | 79.734 / 5.933 | 0,871 / 0,919 | 0,900 / 0,958 | 0,943 / 1,009 | 0,988 / 1,073 | 1,034 / 1,146 | 1,081 / 1,196 | 1,112 / 1,231 |
    | FineWeb-2 vóór / na | 54.475 / 3.269 | 0,837 / 0,875 | 0,869 / 0,918 | 0,917 / 0,984 | 0,966 / 1,058 | 1,015 / 1,134 | 1,062 / 1,195 | 1,095 / 1,229 |
    | raad vóór / na | 42.144 / 1.908 | 0,803 / 0,836 | 0,837 / 0,871 | 0,887 / 0,936 | 0,938 / 1,018 | 0,987 / 1,105 | 1,036 / 1,182 | 1,068 / 1,214 |
    | dpc vóór / na | 16.478 / 661 | 0,795 / 0,839 | 0,829 / 0,872 | 0,880 / 0,931 | 0,931 / 1,000 | 0,980 / 1,079 | 1,027 / 1,163 | 1,058 / 1,196 |
    | rechtspraak vóór / na | 15.895 / 992 | 0,845 / 0,891 | 0,875 / 0,930 | 0,921 / 0,990 | 0,970 / 1,059 | 1,020 / 1,131 | 1,067 / 1,194 | 1,100 / 1,235 |
    | boeken (niet getrokken) | 12.521 | 0,882 | 0,907 | 0,945 | 0,988 | 1,034 | 1,082 | 1,114 |
    | hele set | 25.284 | 0,881 | 0,911 | 0,960 | 1,015 | 1,087 | 1,167 | 1,199 |

    Met 5 bakken volgt de trekking het doel minder precies aan de bovenkant (p90 1,194 tegen 1,222 in het doel; met 10
    bakken 1,222), omdat de bovenste bak breder is en de BT-voorbeelden daarin naar zijn ondergrens liggen [eigen
    gevolgtrekking]; de mediaan ligt op 1,057 (doel 1,062, BT vóór 0,967).
  - Samen **25.284 trainvoorbeelden**: BT 12.763 (50,5%), boeken 12.521 (49,5%); 12.411.900 tokens (BT 5.724.708,
    46,1%; boeken 6.687.192, 53,9%), 5.460.048 doeltokens (BT 2.305.927, 42,2%; boeken 3.154.121, 57,8%); eval 4.167
    BT-passages.

    | bron | voorbeelden | tokens | doeltokens | doeltokens per voorbeeld |
    |---|---:|---:|---:|---:|
    | boeken | 12.521 | 6.687.192 | 3.154.121 | 252 |
    | Handelingen | 5.933 | 3.060.063 | 1.082.251 | 182 |
    | FineWeb-2 | 3.269 | 1.258.360 | 567.777 | 174 |
    | raad | 1.908 | 719.397 | 331.202 | 174 |
    | rechtspraak | 992 | 435.729 | 210.328 | 212 |
    | dpc | 661 | 251.159 | 114.369 | 173 |

    Bronnen in de BT-trekking: Handelingen 46,5%, FineWeb-2 25,6%, raad 14,9%, rechtspraak 7,8%, dpc 5,2% (eerste bouw
    46,7 / 25,7 / 14,6 / 7,8 / 5,2%; RB2 38,5 / 25,9 / 20,1 / 7,7 / 7,8%). Ter vergelijking RB3: 23.846
    trainvoorbeelden, 11.923 boeken + 11.923 BT, 12,0 M tokens.
  - Mediane verhouding woorden NL / EN (tekens): hele set 1,015 (1,045), BT 1,057 (1,060), boeken 0,988 (1,032).
  - Gecontroleerd met `sft_build.py check`: EXIT 0, train 25.284 (bt 11.742, bt_context 1.021, real 12.521),
    12.411.900 tokens, max 2.047; eval 4.167, max 2.006 (`$S/data/rl-b5/check.txt`).
  - Verwachte epoch [schatting, tokens per stap als bij RB2: 29,09 M tokens in 950 stappen]: ~405 stappen.

**Amendement 12 nov 2026 ~18:20 CET: arm RE, de set van R2 plus Engelse romans als derde echt-Engelse bron**
[besluit Yeb 12 nov ~18:15: "ja eens doe die arm ook", op het voorstel in `mbert-ops/translator-sft/books-b1-inventory.md`
§8 punt 2; de opzet (optie A, K = 20, de tijdregel en de classifier-stap vóór de training) door de hoofdsessie namens
Yeb (afwezig), op advies van een Fable-adviseur, 12 nov ~18:20; de uitwerking (de filters, de keuze van de editie, de
trekking en de volgorde) door de subagent RE; vastgelegd vóór de bouw van de set en vóór de eerste trainingsstap van RE
(de klok van co1 gaf bij het vastleggen ~18:30); R2, RB2, RB3, RBR en RL blijven zoals ze zijn]. Aanleiding: het echte
deel van R2 (Europarl-direct 2004–2011 en Global Voices, 52.182 voorbeelden, 11,9% van de tokens) is parlementaire en
journalistieke tekst. Het `-b1-`-deel van `yhavinga/nedd_x_dataset` heeft 1.546.907 paren uit 2.967 Engelse romans
met hun gepubliceerde Nederlandse vertaling die door de controles van de inventaris komen; hun woordverhouding NL / EN
(mediaan 1,072, p5–p95 0,937–1,224) ligt dicht bij die van de echte voorbeelden van R2 (1,062; 0,889–1,278) en ver van
de BT (0,967) [gemeten, inventaris §5.2]. Ze brengen literaire registers en dialoog [eigen gevolgtrekking, inventaris
§8]. Eén nieuwe arm, **RE**: R2's set ongewijzigd plus een begrensde trekking uit die romans. Alleen de romans
toevoegen houdt de vergelijking met R2 zuiver [advies Fable, optie A].

- **Trainingsdata** (`$S/data/re-k20/RE/` en `$S/data/re-k10/RE/`; `sft_build.py extend --base R2 --books
  b1-selected.parquet --per-book K --evalsets evalsets.parquet --block-jsonl blind100.jsonl --name RE`; keten
  `run_arms.sh rebuild`, `recheck`; code op branch `en-nl-sft-re`, met tests):
  1. **De set van R2 ongewijzigd** (`$S/data/r2/R2/`): alle 260.908 trainvoorbeelden byte voor byte en in hun eigen
     volgorde, met hun meta; de eval-split (4.167) en `template.jinja` gekopieerd. `extend` controleert dat na het
     schrijven (de voorbeelden die geen roman zijn, zijn die van R2 in dezelfde volgorde; sha256 over alle voorbeelden,
     de bestanden van de eval-split en de template); `check-base` doet het in de controlestap opnieuw, los daarvan
     (sha256 per voorbeeld en over alle voorbeelden).
  2. **De romans** uit `$S/data/books-b1/b1-selected.parquet` (de inventaris van 12 nov: `yhavinga/nedd_x_dataset` @
     `708f34de8ec1ff4172f22b30e5890a5ee01d2ca7`, de rijen met `-b1-` in het id, tekst na `clean_line` van de loader),
     een rij als:
     - de groep `en_orig` is (Engels origineel, bepaald per auteur; inventaris §3);
     - `passes` waar is: de controles van de inventaris, dus de harde controles van `real_pairs.real_fails`, de
       lengtegrens van R2 (factor 4/3 rond de mediane tekenverhouding van de groep), de uitlijncontrole van RL (4,39 bij
       die mediaan), geen herhaald id en geen herhaald Engels;
     - de Nederlandse tekst niet eerder voorkomt (`dup_nl`) [keuze: twee Engelse passages met dezelfde Nederlandse tekst
       wijzen op een uitlijnfout; 1.006 rijen van de 1.546.907, gemeten];
     - er aan geen van beide kanten een Markdown-spoor in staat, met de maat van de inventaris (§5.4): de reguliere
       expressie `(^|\s)_[^_\s][^_]*_($|[\s.,;:!?])|\*\*`;
     - de schoonmaak van de loader niet maar aan één kant iets veranderde: `en_raw` en `nl_raw` beide leeg of beide
       gevuld [letterlijk volgens de opdracht; ook een verschil in alleen zachte afbreekstreepjes of witruimte telt;
       gemeten: dat kost ~313 paren en 8 boeken meer dan een maat die alleen zichtbare veranderingen telt];
     - **geen MetricX-QE-grens** [richting Yeb: de data zo goed dat filteren niet nodig is].
     Van een Engels boek met meer Nederlandse edities telt **één editie**: die met de meeste rijen die door de controles
     hierboven komen; bij gelijkspel het laagste Nederlandse id [keuze binnen de opdracht].
  3. **Trekking, hoogstens K per Engels boek**: de rijen van de gekozen editie in een volgorde getrokken met de seed
     `en-nl-sft/0/re-books/<Engels id>`; daarvan de eerste K die geen 13-gram delen met een evaluatieset (bron en
     referentie van WMT24++, FLORES-200, NTREX-128 en Europarl-direct) of een blinde passage (`blind100.jsonl`) en in
     2.048 tokens passen, zoals bij de echte paren van R2. Rijen na de K-de worden niet bekeken, dus de trekking met
     K = 10 is het begin van die met K = 20. **K = 20** (`$S/data/re-k20/`) en **K = 10** (`$S/data/re-k10/`) worden
     allebei gebouwd en gecontroleerd. Verwacht [gemeten 12 nov ~18:20 op w1 van mbert-mix, met de rijcontroles
     hierboven zonder `dup_nl`, 13-gram en tokengrens]: 2.953 Engelse boeken met rijen, K = 20 **~58.650** paren, K = 10
     **~29.390** (de inventaris: 59.324 en 29.664 zonder de Markdown- en schoonmaakcontrole).
  4. **Prompt en vorm**: TranslateGemma's eigen prompt (`chat-v1`), zonder context of tag, soort `real`, bron
     `books_en`, zoals de echte paren van R2.
  5. **Volgorde** [keuze, het meest conservatieve dat R2 ongewijzigd laat]: MaxText leest de trainingsdata in de volgorde
     van het bestand (`enable_data_shuffling=false`). De voorbeelden van R2 houden hun eigen volgorde; de
     romanvoorbeelden, eerst onderling geschud (`en-nl-sft/0/re-books-order`), komen op plaatsen die uniform over de hele
     set getrokken zijn (`en-nl-sft/0/re-merge`). Ze staan dus verspreid, niet achter elkaar.
  6. Geen FLAN-replay, instructiezin of nieuwe contextvoorbeelden; de BT en de echte paren van R2 zoals ze zijn.
- **Verwacht** [schatting, tokens per voorbeeld als de boeken van RL (534) en tokens per stap als R2 (112,64 M tokens in
  3.653 stappen)]: K = 20 ~31 M tokens erbij, ~144 M samen, echt Engels ~30% van de tokens (R2 11,9%), ~4.650 stappen;
  K = 10 ~16 M erbij, ~128 M, ~4.160 stappen. De set zoals gebouwd en gecontroleerd volgt in een aanvulling vóór de
  eerste trainingsstap.
- **Training** zoals R2, RB2, RB3, RBR en RL: LoRA op attentie en MLP, **rank 64, alpha 32**, LR 1e-5 cosine naar 0 met
  10% warmup, **één epoch** (`step_limit`), sequenties van 2.048, 1 per chip, **16 chips**. Uitvoer
  `$S/re/runs/arm27b_re_RE/`, adapter `$S/re/RE/`. De keten (`mbert-ops/scripts/translator_sft_re_run.sh`) weigert te
  starten als de regel hieronder niet letterlijk gelijk is aan de zijne, en controleert na de training rank en alpha in
  `adapter.json`.

RE-instellingen: lora_rank=64 lora_alpha=32 learning_rate=1e-5

- **Vóór de training: de classifier op de doelteksten** [voorwaarde Fable; een interpretatie van regel (b), geen poort].
  De classifier van 10 nov (`$S/arms/naturalness/classifier/`, nauwkeurigheid 0,9575; niet opnieuw getraind) op de
  Nederlandse doelteksten van drie groepen: 2.000 romandoelen (uit de trekking met K = 10, dus in beide sets), 2.000
  doelen van Europarl-direct en Global Voices en 2.000 BT-doelen uit de set van R2, elk getrokken met de seed
  `en-nl-sft/0/re-classify/<groep>` (`classify-targets.parquet` naast de set). Het aandeel "origineel" (kans ≥ 0,5) per
  groep en per bron komt in het log en in het rapport. Risico [Fable]: de romandoelen zijn vertaald Nederlands; regel (b)
  kan voor RE daardoor dalen, en dan weegt (c) zwaarder. In dezelfde stap dezelfde classifier op de vertalingen van
  Gemma 4 31B als en→nl-vertaler (`$S/gemma4-en-nl/texts.parquet`: WMT24++ 960 en FLAN 1.000) naast die van het
  basismodel (`$S/arms/naturalness/texts.parquet`, dezelfde id's), in één run; het verschil in het samengenomen aandeel
  "origineel" zoals regel (b) het rekent, in het log (regel (b) voor Gemma 4 ontbrak nog). Beide beslissen niets over RE.
  Uitvoer `$S/re/classify/` (`run_arms.sh reclassify`).
- **Tijd** [voorwaarde Fable]: de keten start pas na `EXIT 0` van de keten van RL (`translator_sft_rl_run.sh`; een andere
  `EXIT` is `FAILED`) en neemt dan `final_eval_mix.lock`. De training start uiterlijk **13 nov 05:00 CET met K = 20**;
  start ze tussen 05:00 en 07:00, dan **met K = 10**; later wordt RE **niet getraind** (logregel "RE overgeslagen: …",
  daarna `DONE`). Daarnaast moeten training en evaluatie naar schatting vóór **13 nov 16:30 CET** klaar zijn (zoals bij
  RL; mbert-mix is daarna voor de evaluatie van T2); anders ook overgeslagen. De schatting (minuten van de start van de
  training tot het rapport) volgt met de aanvulling.
- **Evaluatie en beslisregel gelijk aan die van B, N, RB, RB2, R2, RB3, RBR en RL** (§4, §5; de amendementen R2 en RB),
  tegen hetzelfde uitgangspunt (`$E/baseline-tg27b/scores.parquet`): de vier sets met dezelfde code (MetricX-24 met
  referentie en QE, COMET-22; niet af, lus, Engels over, leeg); de extra sets van soort `x` (FLAN 1.000, blind 100,
  in-domein 4.519 van `$S/arms/data/x/`, het basismodel uit `$S/arms/eval/base-x/`); de classifier van 10 nov op de
  WMT24++ (960) en de 1.000 FLAN-rijen, basismodel en RE in één classify-run; PLL en MTLD; het blinde pakket
  `$S/blind/RE/` (A/B met seed `en-nl-sft/blind/ab/RE/<blind_id>`, sleutel alleen in de bucket); het rapport
  `$S/re/report/report.md` met regel (a), (b), de stand van (c) en, gerapporteerd zonder mee te beslissen, de lengte van
  de uitvoer tegenover de menselijke referentie op de vier sets. RE helpt als (a), (b) en (c) alle drie gelden.
- **Licentie en privacy**: de romans zijn gepubliceerde boeken en vertalingen onder auteursrecht, uit een privé
  dataset; de sets, de adapter en alle afgeleiden blijven privé (bucket); niets wordt gepubliceerd zonder Yeb.
