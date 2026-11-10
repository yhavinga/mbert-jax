# Protocol (concept): fine-tune van de en→nl-vertaler, armen N en B op TranslateGemma 27B

Status: **concept**, 10 nov 2026, geschreven door een subagent voor de hoofdsessie en Yeb. Nog niet vastgelegd en
nog niets op 27B gestart. Vast te leggen vóór de fine-tune (ontwerp §4: "vóór de fine-tune vast te leggen in
`protocol.md`"). Branch `en-nl-sft` van `github.com/yhavinga/mbert-jax`.

Bronnen: hammond `mbert/designs/en-nl-translator.md` (hierna "ontwerp"), §3 stap 3, §4, §7 punt 6–8 en §9;
hammond `mbert/LEARNINGS.md`, de vertaler-items; `experiments/en-nl-translator/HANDOVER.md` (smoke-test, hoofdrun,
aanvullende run); `mbert-ops/HANDOVER-baseline.md` en `mbert-ops/translator-eval/baseline-tg27b.md` (uitgangspunt).

Labels: **[besluit]** = besluit van Yeb (met datum); **[ontwerp]** = staat zo in het ontwerp; **[gemeten]** = op
10 nov gemeten op de data (`sft_build.py`, rapport in `gs://t5-dutch-english-usc2/mbert/translator/sft/data/arms/report.md`);
**[keuze]** = keuze van dit concept, door Yeb te bevestigen; **[open]** = beslist Yeb (§8).

`R=gs://t5-dutch-english-usc2/mbert/translator`, `S=$R/sft`, `E=$R/eval`.

## 1. Vraag en armen

Verliest de vertaler na de fine-tune betekenis, en schrijft hij natuurlijker Nederlands? [ontwerp §4]

| arm | trainingsdata | bij het vertalen |
|---|---|---|
| N | BT-paren met tag + 20% FLAN-nl-replay zonder tag | met tag (ook zonder tag gemeten, alleen gerapporteerd) |
| B | dezelfde BT-paren, zonder tag | zonder tag |

- Het ongetagde deel van N is de FLAN-nl-replay [besluit 9 nov, ontwerp §7 punt 6]; echte vertaalparen (Europarl)
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
   door de harde controles komen [keuze]; geen grens op chrF en geen Gemma-paarscore, geen zinnen- of getallenfilter
   [besluit 9 nov];
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

### 2.2 Contextvoorbeelden [besluit 9 nov, ontwerp §9]

- Een trainpassage die de selectie als contextvoorbeeld markeerde (Handelingen, 1–3 voorgaande spreekbeurten,
  samen ≤ 1.024 tokens), houdt haar beurten, van de dichtstbijzijnde terug, zolang elke beurt door dezelfde query
  komt (harde controles, QE ≤ 7,5, 13-gram, bronscore ≥ 0,5); de eerste die faalt, beëindigt de context. Zonder
  beurt over is het een gewoon voorbeeld [keuze].
- Elke beurt staat in de context als (synthetisch Engels, origineel Nederlands) met het sprekerslabel van de
  Handelingen ("De heer Korthals (VVD)", "De voorzitter"); de huidige beurt krijgt het label van haar spreker.
  Verlies alleen op de huidige beurt.
- **Aandeel: 20% van de BT-voorbeelden, in N en B gelijk.** Van de 15.903 gemarkeerde trainpassages die door de
  query komen, hielden 15.583 minstens één beurt (320 geen; 1.304 beurten vielen weg op de query), en bij 261 maakte
  de context het voorbeeld langer dan 2.048 tokens (zij blijven als gewoon voorbeeld): er zijn 15.322
  trainvoorbeelden met context [gemeten]. Bij 20% zijn dat 76.610 BT-voorbeelden in totaal (`--bt auto`). De overige 61.288 gewone voorbeelden
  worden per bron getrokken in verhouding tot de passages die door de query komen, zodat elke bron haar aandeel
  houdt [keuze]. Gevolg: van de 225.242 bruikbare trainpassages gaat ~34% de training in. Alternatief: alle
  passages met ~7% context [open, §8 punt 3].

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
- **Aandeel: 20% van de voorbeelden van N**, dus replay = BT × 0,25 = 19.152, willekeurig getrokken (seed
  `en-nl-sft/0/replay`). Zonder tag en zonder context, in TranslateGemma's eigen prompt.

### 2.4 Prompt en tag

- Een voorbeeld zonder context krijgt `chat-v1` zonder context: woord voor woord TranslateGemma's eigen prompt
  (`translate.user_text`; `sft_build.py check` toetst de token-id's tegen TranslateGemma's eigen chat-template). Een
  voorbeeld met context krijgt `chat-v1` met context (`<earlier_turns>`, `<current_turn speaker="…">`) [besluit
  9 nov, ontwerp §9: eigen prompt alleen voor de voorbeelden met context].
- De tag van arm N heeft twee vormen; welke het wordt, volgt uit de proef van §6 [open, §8 punt 2]:
  - **token**: Gemma 3's ongebruikte token `<unused0>` (id 6) direct vóór de te vertalen tekst (`chat-v1-tag-token`;
    met context vóór de tekst van de huidige beurt). Eén token, geen speciaal token voor de detokenizer (een
    gelekte tag blijft dus zichtbaar), inbedding met een gewone norm (0,964 tegen p50 0,951 van gebruikte tokens);
    komt in geen enkele tekst van de pairs-tabellen of de replay voor (de build stopt anders) [gemeten];
  - **instructie**: de zin "Translate into natural, original-sounding Dutch." in de instructie, na "… or
    commentary." en vóór "Please translate …" (`chat-v1-tag-instruction`; met context op dezelfde plek), niet vóór de
    bron.
- De replay en arm B zijn altijd zonder tag.

### 2.5 De sets [gemeten]

| set | split | voorbeelden | BT zonder context | BT met context | replay | lang (385–511) | tokens | verliestokens | max |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B | train | 76.610 | 61.288 | 15.322 | 0 | 8.671 | 43.346.465 | 14.992.546 | 2.039 |
| N-token | train | 95.762 | 61.288 | 15.322 | 19.152 | 8.671 | 49.358.568 | 17.583.060 | 2.040 |
| N-instruction | train | 95.762 | 61.288 | 15.322 | 19.152 | 8.671 | 49.971.448 | 17.583.060 | 2.048 |
| alle drie | eval | 4.519 | 4.223 | 296 | 0 | 602 | 2,19–2,23 M | 917.827 | ≤ 2.015 |

- Bronnen van de BT-voorbeelden (train, in elke set gelijk): Handelingen 29.945 (39,1%), FineWeb-2 19.576 (25,6%),
  raad 15.723 (20,5%), dpc 5.829 (7,6%), rechtspraak 5.537 (7,2%): de verhouding van de passages die door de query
  komen, en dicht bij de verdeling 40/20/7,5/7,5/25 van het ontwerp.
- N-token en N-instruction hebben dezelfde paren in dezelfde volgorde; ze verschillen alleen in de prompt van de
  BT-voorbeelden. B heeft dezelfde BT-paren zonder tag en zonder replay. De eval-sets zijn alle 4.519 eval-passages
  die door de query komen (met context waar die er is), in de prompt van de arm, zonder replay.
- Tokens: alle niet-opvultokens van prompt en doel, zoals `sft_lora.py` ze telt; verliestokens: het Nederlands plus
  `<end_of_turn>\n`. Een contextvoorbeeld is gemiddeld ~1.000 tokens, een gewoon voorbeeld ~450 [gemeten, afgeleid].

Bestanden per set in `$S/data/arms/<set>/`: `train.parquet` en `eval.parquet` (kolom `messages`, voor MaxText; train
geschud met seed `en-nl-sft/0/shuffle`), `train-meta.parquet` en `eval-meta.parquet` (per rij id, soort, bron, tag,
prompt, Engels, context, label, tokens, QE, chrF), `template.jinja` (de gebruikersbeurt zoals gebouwd in Gemma 3's
beurt) en `manifest.json` (query, aantallen, sha256). `report.md`/`report.json` in `$S/data/arms/`. `sft_build.py
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
  eigen deel van de data, en alle processen nemen het kleinste aantal (in de 12B-proef 431, 433, 438 en 437 → 431
  stappen: ~1% van de batches van drie processen blijft ongebruikt) [gemeten].
- Data: packing, sequenties van **2.048** tokens (de smoke-test gebruikte 1.024), 1 sequentie per chip
  (`per_device_batch_size=1`, 16 × 2.048 = 32.768 tokens per stap; de smoke-test: 2 × 1.024), geen shuffling in
  MaxText (de build schudt), verlies alleen op het Nederlands en `<end_of_turn>` (`sft_train_on_completion_only`).
- Duur [schatting]: bij het tempo van de smoke-test (4.018 tokens/s over 16 chips, 2 × 1.024 per chip) B 43,3 M tokens ≈ 3,0 uur, N 49,4–50,0 M tokens ≈ 3,4–3,5 uur, plus ~10 min laden, tellen en compileren; bij 2.048 per sequentie niet gemeten. Eerst een korte run (`STEPS=8`) om geheugen en tempo bij 2.048 te meten [keuze].
- Uitvoer: `$S/arms/runs/<naam>/` (`adapter.npz`, `adapter.json`, `summary.json`, logs).

## 4. Evaluatie

### 4.1 Betekenis (beslist, regel (a))

De keten van `en-nl-eval` (`run_baseline.sh`: `translate`, `score`, `summary`; met `ADAPTER`, `PROMPT`, `LABEL` en een
eigen `B` per arm), op dezelfde sets en met dezelfde code als het uitgangspunt:

| set | n | rol |
|---|---:|---|
| WMT24++ en→nl_NL | 998 (of 960 zonder `is_bad_source`) [open, §8 punt 1] | beslist (a) |
| FLORES-200 devtest eng→nld | 1.012 | beslist (a) |
| NTREX-128 newstest2019 eng→nld | 1.997 | beslist (a) |
| Europarl-direct (nl-origineel als referentie) | 640 | gerapporteerd |

- Maten: MetricX-24 hybrid XXL in QE-modus (**beslist**), MetricX-24 met referentie en COMET-22 met referentie
  (gerapporteerd). MetricX bevoordeelt TranslateGemma (RL op MetricX-QE); COMET-22 staat er daarom naast [ontwerp].
  CometKiwi en xCOMET niet (CC BY-NC-SA).
- Decodering zoals het uitgangspunt: greedy, `translate.py`, lengteklassen, MaxText 0.2.5; de adapter samengevoegd
  in de kernels (`--lora`). Arm N met tag (`--prompt chat-v1-tag-…`) en daarnaast zonder tag; arm B zonder tag.
- Ook geteld: niet af, `decode_loop`, Engels over, leeg (zoals het uitgangspunt: daar alle 0).

### 4.2 Natuurlijkheid (beslist, regel (b) en (c))

1. **Classifier "origineel of vertaald"** [ontwerp §4.2, regel (b)].
   - Model: onze Dutch ModernBERT S @ 653k (`gs://t5-dutch-english-usc2/mbert/culturax-nl-230m-chunk1024/ckpt_keep/653000`),
     binaire classificatie met de fine-tune-code van DUMB (`finetune.py`), LR 5e-5 (LEARNINGS: elk model beter bij
     5e-5), 3 epochs, `--max-length 1024`, seed 42 [keuze].
   - Data: 2.000 originelen uit de eval-kant van de selectie (BT-eval-passages die door de query komen, ander
     document dan elke trainpassage) tegen hun terugvertaling door het basismodel (kolom `back`, dezelfde inhoud
     in het Nederlands van TranslateGemma 27B). 90/10 gesplitst per passage (de twee versies van een passage aan
     dezelfde kant). Keuze van de 2.000: de eval-passages zonder context, geordend op sha256 van `pid` met
     `en-nl-sft/classifier`, de eerste 2.000 [keuze].
   - Voorwaarde: nauwkeurigheid op de 10% ≥ 0,75; anders meldt het verslag dat (b) niet te meten is, en beslist
     Yeb [keuze].
   - Meting: het aandeel van de uitvoer dat de classifier "origineel" noemt (kans ≥ 0,5), op de vertalingen van de
     **WMT24++-bron (998 of 960, als §4.1) en de 1.000 FLAN-evaluatierijen** (set `classifier` van
     `$R/mainset/flan-eval/eval-rows.parquet`), samen genomen; per set gerapporteerd.
   - Voorbehoud [ontwerp]: de classifier leert de stijl van dit basismodel herkennen; een arm kan daaraan
     ontsnappen zonder natuurlijker te worden. Daarom ook 3 en 4.
2. **In-domein**: de volgende 2.000 eval-passages in dezelfde volgorde (dus niet die van de classifier), Engels →
   arm, met het originele Nederlands als referentie: MetricX-24 ref, COMET-22, chrF. Gerapporteerd [ontwerp §4.1:
   "de 2.000 achtergehouden GPT-NL-paren"].
3. **PLL en MTLD**: pseudo-log-likelihood onder S @ 653k (de PLL-word-l2r van BLiMP-NL in `dumb-extra`) per
   vertaling, en MTLD (woordenschatvariatie), op dezelfde uitvoer als 1. Gerapporteerd.
4. **Blinde beoordeling door Yeb** [ontwerp §4.4, besluit 9 nov: 100 passages; regel (c)].
   - 100 Engelse passages, **gekozen vóór de fine-tune** en vastgelegd in `$S/blind/passages.parquet` (met sha256 in
     dit protocol):
     - 35 FLAN: de 35 rijen van set `yeb` in `$R/mainset/flan-eval/eval-rows.parquet` (liggen vast sinds 9 nov);
     - 35 WMT24++: seed `en-nl-sft/blind/wmt`, uit de 960 zonder `is_bad_source`, zonder de 300 van de tagproef;
     - 30 met uitdrukkingen en phrasal verbs: kandidaten uit de Engelse bronnen van NTREX-128 en de overige
       WMT24++-segmenten met een treffer uit een Engelse lijst van idiomen en phrasal verbs (Wiktionary via
       kaikki.org, dezelfde bron als de Nederlandse idioomlijst), seed `en-nl-sft/blind/idiom`; Yeb kan de
       kandidaten bekijken voor de keuze vastligt [keuze; open, §8 punt 6].
   - Per passage de vertaling van het basismodel en van de arm (N met tag, B zonder), in willekeurige volgorde (seed
     per passage), zonder naam. Oordeel "A beter", "B beter" of "gelijk", en per vertaling betekenisfout ja/nee; een
     misplaatste uitdrukking telt als betekenisfout.
   - Yeb beoordeelt alleen de armen die (a) en (b) halen: een arm die daarop faalt, helpt volgens de regel toch niet
     [keuze].

## 5. Beslisregel (exact)

Per arm X ∈ {N, B}, tegen het basismodel TranslateGemma 27B (`$E/baseline-tg27b/summary.json`):

- **(a) betekenis**: voor elk van WMT24++, FLORES-200 en NTREX-128 geldt
  gemiddelde MetricX-24-QE(X) − gemiddelde MetricX-24-QE(basis) ≤ 0,20, over alle segmenten van de set
  (onafgeronde gemiddelden). Grenzen: WMT24++ 1,623 + 0,20 = 1,823 (alle 998) of 1,547 + 0,20 = 1,747 (960),
  FLORES 0,822 + 0,20 = 1,022, NTREX 1,082 + 0,20 = 1,282 (afgerond weergegeven).
- **(b) natuurlijkheid volgens de classifier**: aandeel "origineel"(X) − aandeel "origineel"(basis) ≥ 0,10, op de
  samengenomen uitvoer van WMT24++ en de 1.000 FLAN-rijen (§4.2.1).
- **(c) Yeb**: k / n ≥ 0,60, met n het aantal passages met een oordeel "A beter" of "B beter" en k het aantal waarin
  X beter is; én het aantal vertalingen van X met een betekenisfout ≤ dat van het basismodel, over dezelfde 100
  passages.

X **helpt** als (a), (b) en (c) alle drie gelden; anders **helpt X niet**. Europarl-direct, het in-domeindeel,
MetricX met referentie, COMET-22, PLL en MTLD, en N zonder tag worden gerapporteerd, maar beslissen niet [ontwerp].
Ter informatie bij (c): bij n = 60 is k = 36 (60%) eenzijdig binomiaal p = 0,078, bij n = 100 is k = 60 p = 0,028
[ontwerp].

Helpen N en B allebei: [open, §8 punt 9].

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

**Stand bij het schrijven** (10 nov ~05:55 CET): de adapter met het token traint (431 stappen van 3,6 s; verlies
2,16 bij stap 6, 1,71 bij stap 19), daarna die met de instructie, dan de vertalingen en MetricX. De uitslag komt in
`mbert-ops/translator-sft/tag-test.md` (automatisch, met tabel en 10 voorbeelden per adapter) en
`$S/tagtest/report/`; de keuze van de vorm hoort daarna hier, als besluit van Yeb.

## 7. Stappen na akkoord

1. Yeb beslist de open punten (§8); daarna wordt dit protocol vastgelegd (commit, datum) vóór de eerste stap op 27B.
2. De 100 blinde passages kiezen en vastleggen (§4.2.4).
3. De classifier trainen en zijn nauwkeurigheid melden (§4.2.1); de uitvoer van het basismodel op de 1.000
   FLAN-rijen, het in-domeindeel en de blinde passages maken.
4. Per arm: korte run (`STEPS=8`), dan de volledige epoch, dan de evaluatie van §4. Commando's in
   `mbert-ops/HANDOVER-sft.md`.
5. Verslag per arm met de regel van §5.

## 8. Open punten voor Yeb

1. **WMT24++: 998 of 960?** Telt (a) met alle 998 segmenten of met de 960 zonder `is_bad_source` (zoals het artikel
   en de datasetkaart)? Uitgangspunt 1,623 tegen 1,547. Ook voor (b).
2. **Vorm van de tag**: token of instructie (§6).
3. **Omvang**: 76.610 BT-voorbeelden met 20% context (de opdracht; ALMA's optimum 10K–100K), of alle 225.242
   bruikbare passages met ~7% context (meer data, langere run, ander contextaandeel).
4. **QE-grens 7,5** (≈ p99): kost 2.117 trainpassages. Strenger (bijv. 6: 7.902) of geen grens?
5. **Sequentielengte 2.048** met 1 sequentie per chip: niet gemeten op 27B; de korte run meet het. Bij geheugengebrek
   terug naar 1.024 en de langste voorbeelden (context) weglaten?
6. **De 30 idioompassages**: bron en keuze (§4.2.4); kiest Yeb zelf uit kandidaten?
7. **Classifier**: S @ 653k, of het eindmodel als dat er is? Grens 0,75 voor bruikbaarheid?
8. **N zonder tag**: alleen rapporteren (zo in dit concept).
9. **Als N en B allebei helpen**: N (de literatuur: getagde BT), of de arm met de hoogste k/n bij (c)?
10. **Rekentijd**: twee armen samen ~6,5–7 uur training plus ~1 uur evaluatie per arm (4.1); mbert-mix is daarvoor nodig, mbert-corpus blijft bij de eindrun.
