# Protocol: arm G4-1, de eerste vertalerarm op Gemma 4 31B-it

Status: **vastgelegd 14 nov 2026** (tijdstip: de commit van dit bestand op branch `en-nl-g4arm` van
`github.com/yhavinga/mbert-jax`), vóór de bouw van de trainingsset, vóór de eerste trainingsstap en vóór elke uitslag
van G4-1. Geschreven door een subagent voor de hoofdsessie en Yeb. Wat na het vastleggen verandert, staat als gedateerd
amendement in §9. Het protocol van de armen op TranslateGemma (`protocol.md`, hierna "TG-protocol") blijft gelden waar
dit document ernaar verwijst; waar dit document iets anders zegt, geldt dit document voor G4-1.

Labels: **[besluit]** = besluit van Yeb (met datum); **[keuze]** = keuze van dit protocol (technisch, door de subagent);
**[gemeten]** = gemeten, met bron; **[schatting]**; **[gevolgtrekking]** = eigen gevolgtrekking, niet gemeten.

`R=gs://t5-dutch-english-usc2/mbert/translator`, `S=$R/sft`, `E=$R/eval`, `G=$S/g4arm` (alles in us-central2).

## 1. Vraag, arm en uitgangspunt

**Vraag.** Maakt een LoRA-fine-tune van Gemma 4 31B-it op terugvertalingen van Gemma 4 en op de Nederlandse boeken een
betere en→nl-vertaler dan Gemma 4 31B-it zelf, zonder fine-tune, met de prompt `natural-v1`?

- **Besluit van Yeb (14 nov ~00:40)**: de eerste arm op Gemma 4 bouwen, **zonder FLAN-replay** zolang FLAN-nl niet
  hersteld is. De gepubliceerde FLAN-nl is met TranslateGemma op MaxText 0.2.5 vertaald, met de fout in het
  sliding-window-masker (LEARNINGS, "Lange teksten falen door twee fouten in ons MaxText-decodeerpad"). FLAN-replay komt
  pas in de mix als er een Gemma 4-vertaling van de replay-voorbeelden is; **TranslateGemma-FLAN-nl wordt nooit
  gebruikt** (§2.5).
- **Arm G4-1** = Gemma 4 31B-it (`google/gemma-4-31B-it` @ `842da3794eaa0b77d5f08bae87a17459d91ff475`, Apache 2.0) +
  LoRA **rank 32, alpha 16**, op attention (`query`, `key`, `value`, `out`) en MLP (`wi_0`, `wi_1`, `wo`) van alle 60
  lagen; de 10 globale lagen hebben geen `value` (410 modules, 244.858.880 LoRA-parameters) [gemeten 13 nov, run
  `g4l_v3b`, `$S/g4lora/runs/g4l_v3b/summary.json`]. Waarom rank 32: het advies van de geheugentest van 13 nov volgens de
  daar vooraf vastgelegde regel (rank 64: piek 29,6 GiB per chip, boven de grens 29,5; rank 32: 24,6 GiB, 8,2 s per stap;
  `mbert-ops/translator-sft/gemma4-moe-lora-test.md`). Alpha 16 geeft dezelfde schaal alpha / rank = 0,5 als rank 64 /
  alpha 32 van RB2, RB3, RBR, RL en RE.
- **Uitgangspunt G4-0** = hetzelfde Gemma 4 31B-it zonder fine-tune, prompt `natural-v1`, **opnieuw vertaald in
  dezelfde keten als G4-1** (zelfde VM-type v4, code, MaxText-fork, lengteklassen, slots, greedy) [keuze]. Reden: het
  verschil tussen arm en uitgangspunt moet van de adapter komen, niet van een ander decodeerpad: greedy uitvoer hangt af
  van de batch-samenstelling en de klasse-indeling (bij de blinde set van de 31B waren 32 van 35 WMT-passages gelijk aan
  de eerdere vertaling; `gemma4-sizes.md` §7). De eerdere uitkomsten van Gemma 4 31B op een v6e-8 (QE 2,230 / 1,160 /
  1,554 op WMT24++ (960) / FLORES / NTREX, "origineel" 30,46%, blind 56 / 22 / 22 tegen TG) worden naast G4-0
  gerapporteerd als controle; ze beslissen niet.
- **Tweede referentie: TranslateGemma 27B (TG)** zonder fine-tune, de bestaande uitkomsten van het TG-protocol
  (`$E/baseline-tg27b/`, de basisvertalingen van FLAN en de blinde set in `$S/arms/eval/base-x/`). Gerapporteerd, beslist
  niet (§5).

## 2. Trainingsdata

Code: `experiments/en-nl-translator/g4arm.py build` en `check`; tests `tests/test_en_nl_g4arm.py`. De set komt in
`$G/data/G4-1/` (`train.parquet` met kolom `messages`, `train-meta.parquet`, `template.jinja`, `manifest.json`), het
verslag in `$G/data/report.md` en `report.json` (kopie `mbert-ops/translator-sft/data-g4-1-report.md`). Vaste seed-prefix
`en-nl-g4arm/0`.

### 2.1 Terugvertalingen van Gemma 4 (BT)

- **Bron** [besluit 13 nov: alleen de Gemma 4-BT telt]: de vijf publieke datasets `yhavinga/<bron>-nl-en-gemma4`, split
  `train` (de split `validation` blijft de evaluatiekant, per document gesplitst), op deze revisies (14 nov 00:17 CET,
  met de kolommen van de rondgang):

  | dataset | revisie | bron in de rapporten |
  |---|---|---|
  | `yhavinga/handelingen-tweede-kamer-nl-en-gemma4` | `3a129f586544d57eb7d6135b3c7b154153f22ee8` | handelingen |
  | `yhavinga/openraadsinformatie-nl-en-gemma4` | `6fb33363cf38a329beb15a1ed72e986ad220ccf5` | raad |
  | `yhavinga/rijksoverheid-websites-nl-en-gemma4` | `0683859b68e6ee594b60e800d4d0110ff4c86276` | dpc |
  | `yhavinga/rechtspraak-nl-en-gemma4` | `0eaca05493c9293f377ec3bf6b2e2c0d02963b82` | rechtspraak |
  | `yhavinga/fineweb-2-nl-en-gemma4` | `f7538a912cb3b24c810e2032d8dca108e216b2e0` | fineweb |

  Een squash op Hugging Face kan een revisie later laten verdwijnen; daarom gaat een kopie van de gelezen bestanden naar
  `$G/input/bt/` met hun sha256 in `manifest.json` [keuze]. Alleen de configuratie `default` (`data/`); de
  contextbeurten van de Handelingen (`context_turns`) niet (§2.6).
- **Pool** (een rij komt in aanmerking als alles geldt) [keuze, zoals RL waar het kan]:
  1. `hard_fails` leeg (de harde controles van de dataset: niet af, lus, Nederlands over, lengteverhouding > 1,5,
     opmaak, verdubbelde regelafbreking, inleiding of noot);
  2. de lengtegrens van R2: de tekenverhouding Nederlands / Engels ligt binnen een factor 4/3 van de mediaan van de
     bron (mediaan / 1,3333 tot mediaan × 1,3333; medianen over de train-rijen van die bron die door punt 1 komen)
     (TG-protocol §9, amendement R2, punt 1 (ii); `sft_build.length_bounds`);
  3. geen 13-gram van woorden gedeeld met een evaluatieset (WMT24++, FLORES-200, NTREX-128, Europarl-direct: bron en
     referentie, `$E/data/evalsets.parquet`) of met de bronnen van de 100 blinde passages
     (`$S/blind/fixed-2026-11-10/blind100.jsonl`), aan de Nederlandse en de Engelse kant;
  4. het hele voorbeeld (prompt in het chat-template + Nederlands + `<turn|>\n`) telt hoogstens **2.048** tokens van
     Gemma 4;
  5. **FineWeb-2-crawljaar: een parameter, standaard geen grens** (`--fineweb-max-crawl`, inclusief; een jaar `2023`
     of een crawl `CC-MAIN-2023-23`; leest de kolom `dump`). De meting van 13 nov
     (`mbert-ops/translator-sft/fineweb2-year-check.md`) vond de eerste daling in crawljaar 2024 en een knik bij
     CC-MAIN-2023-40; welke grens geldt, heeft Yeb nog niet gekozen. G4-1 traint **zonder grens**; een grens maakt een
     andere arm (amendement).
- **Geen QE-grens**: er is geen MetricX-score van het Gemma 4-Engels, en Yebs richting (12 nov) is data zo goed dat
  filteren niet nodig is. De chrF van de rondgang (kolom `chrf`) is **geen grens**; het verslag geeft haar verdeling in
  de pool en in de trekking [keuze].
- **Hersampling naar de menselijke lengteverhouding** (de methode van RL, TG-protocol §9, amendement RL punt 3 met de
  aanvulling "5 bakken"; `sft_build.match_target` en `bt_match_draw`, ongewijzigd): histogram matching op
  log(woorden NL / woorden EN) naar de **echte voorbeelden van de trainingsset van R2** (`$S/data/r2/R2`, soort `real`:
  52.182, Europarl-direct en Global Voices, mediaan 1,0625), **5 bakken** begrensd door de kwintielen van het doel
  (0,9737 / 1,0351 / 1,0909 / 1,1613), elke bak gevuld naar zijn aandeel in het doel (grootste rest), zonder
  teruglegging, uniform met seed `en-nl-g4arm/0/bt-match/<b>`. Een bak van het doel zonder BT-voorbeelden is een fout.
  Volgens een schatting van de hoofdsessie houdt de hersampling hoogstens ~60.145 paren; G4-1 trekt er veel minder
  (§2.4), dus geen bak begrenst het aantal [schatting, de build meet het].

### 2.2 Nederlandse boeken

De **12.521 boekparen van RL** (`$S/data/rl-b5/RL`, soort `real`: `yhavinga/nedd_x_dataset_nlbooks_b1` zonder de vijf
auteurs die niet in het Nederlands schreven, de harde controles, de lengtegrens, de uitlijncontrole met grens 4,39, geen
QE-grens, geen 13-gram met een evaluatieset of blinde passage; TG-protocol §9, amendement RL punten 1–2): het Engels uit
`train-meta.parquet`, het Nederlands uit `train.parquet`, ongewijzigd, met de prompt van §2.6. Opnieuw geteld met Gemma 4
(punt 4 van §2.1); het verslag noemt wat daarbij wegvalt [keuze: dezelfde boekparen als RL, zodat het verschil met RL
niet in de boeken zit].

### 2.3 Wat er niet in zit

Geen contextvoorbeelden (§2.6), geen Europarl of Global Voices (RBR en R2: echt Engels uit Europarl/GV voegde niets toe
boven de boeken en drukte regel (b)), geen romans van RE, geen FLAN-replay (§2.5), geen instructiezin.

### 2.4 Verhouding BT : boeken = 1 : 1 [keuze, onderbouwing]

n_BT = het aantal boekparen (`--bt-per-book 1`), dus ~50 : 50 in voorbeelden, zoals RB3 (11.923 + 11.923) en RL
(12.763 + 12.521). Onderbouwing, uit de uitslagen van de TG-armen (LEARNINGS):
- RB3 (50 : 50) lag op regel (a) dichter bij TG dan RB2 (80 : 20) en RBR (60 : 20 : 20) en haalde (b) (+12,2 pp);
  meer BT (B, RB, RB2, R2) trok de uitvoer verder van het uitgangspunt en comprimeerde haar.
- RL (50 : 50, BT hersampled naar de menselijke lengte, boeken zonder QE-grens) was even goed als RB3 op (a1) en (a2) en
  het hoogst op (b) (+13,4 pp), met uitvoer even lang als de referentie op alle drie de sets.
- G4-1 verschilt daardoor van RL alleen in: het basismodel (Gemma 4 31B-it in plaats van TranslateGemma 27B), de bron
  van het synthetische Engels (Gemma 4 in plaats van TranslateGemma), de prompt en het template (§2.6), de rank (32 / 16
  in plaats van 64 / 32, zelfde schaal) en het ontbreken van contextvoorbeelden (RL: 1.021, 8%).

### 2.5 FLAN-replay: een parameter, nu 0

- `--replay-share` (aandeel van de voorbeelden; replay = round((BT + boeken) × aandeel / (1 − aandeel))), **standaard
  0**. G4-1 heeft geen replay [besluit 14 nov ~00:40].
- Met een aandeel > 0 eist de build `--replay` (een parquet met kolommen `id`, `en`, `nl`, eventueel `task` en `model`)
  en `--replay-model`; hij **weigert** een bron die TranslateGemma-FLAN-nl is: een pad of naam met `yhavinga/flan-nl`,
  `mbert/flan-nl/` (behalve de map van de Gemma 4-pilot `mbert/flan-nl/g4-pilot`) of `mainset/replay`, een
  `--replay-model` zonder `gemma-4`, of een kolom `model` met een waarde zonder `gemma-4` [keuze].
- Getrokken uit de replay-rijen die geen segment van de FLAN-evaluatierijen zijn (`$R/mainset/flan-eval/eval-rows.parquet`,
  kolom `segments`, sets `classifier` en `yeb`), geen 13-gram delen met een evaluatieset of blinde passage, en binnen
  2.048 tokens passen; uniform met seed `en-nl-g4arm/0/replay`. Prompt: `natural-v1` **zonder** instructiezin, zoals de
  rest [keuze: de replay moet de standaardmodus leren een opdracht te vertalen in plaats van uit te voeren, de faalwijze
  van de TG-armen op FLAN; de instructiezin van arm N was een modusschakelaar naar TG's letterlijke stijl].
- Een arm met replay is een **nieuwe arm (G4-2)** met een eigen amendement vóór zijn training; de beslisregel en de
  evaluatie van dit protocol gelden dan ongewijzigd, met G4-0 als uitgangspunt.

### 2.6 Prompt, template, verlies

- User-beurt: `natural-v1` uit `experiments/flan-nl/translate_models.yml`, letterlijk, met het Engels als `{text}`
  (`translate.user_text`): dezelfde prompt als het uitgangspunt van de blinde beoordeling en als G4-0. `natural-v1` heeft
  geen contextvorm; daarom geen contextvoorbeelden [keuze; een arm met context is een latere arm].
- Template: het chat-template van Gemma 4 met één user-beurt en **thinking uit** (het lege denkkanaal gesloten:
  `<bos><|turn>user\n…<turn|>\n<|turn>model\n<|channel>thought\n<channel|>`), het Nederlands erachter, afgesloten met
  `<turn|>\n` (`g4lora.TEMPLATE`). `check` legt elk voorbeeld langs MaxTexts eigen SFT-stappen en het eigen chat-template
  van het model (zoals `g4lora.py data` op 13 nov).
- Verlies alleen op het Nederlands en `<turn|>\n` (`sft_train_on_completion_only`).
- De volgorde van de voorbeelden: geschud met seed `en-nl-g4arm/0/shuffle` (MaxText schudt niet).

## 3. Training

- Stack van de geheugentest van 13 nov: MaxText 0.2.5 `train_sft` met Tunix en qwix-LoRA (`run_sft.sh`, `sft_lora.py`,
  SFT-venv van `setup_sft.sh`), checkpoint `$R/gemma4-ckpt/gemma-4-31b-it` (ongescand, bf16 op schijf, gewichten in fp32
  tijdens de training), als één multi-host-job over de 16 chips van een v4-32 (`MULTIHOST=1`, FSDP).
- LoRA rank 32, alpha 16, de modules van §1 (de regex van `run_sft.sh`), embeddings bevroren, dropout 0.
- Optimalisatie als RL: AdamW (β2 0,999, weight decay 0,01), LR 1e-5, cosine naar 0 met 10% warmup, clipping 1,0,
  **één epoch** (`step_limit` van `sft_lora.py`), `num_vocab_tiling=8`.
- **Sequenties van 2.048 tokens**, packing, 1 per chip (16 × 2.048 = 32.768 tokenplaatsen per stap) [keuze].
  Onderbouwing: elk voorbeeld past (punt 4 van §2.1); de tokens per stap zijn die van RL en de andere armen, dus dezelfde
  effectieve batch; gemeten 24,6 GiB per chip en 8,2 s per stap (13 nov). Bij 4.096 (25,0 GiB, 10,6 s per stap) zou een
  epoch ~35% sneller zijn, maar met de helft van de stappen en een dubbele batch: een tweede verschil met RL. 4.096 is
  bedoeld voor een latere set met langere voorbeelden.
- **Eén epoch** [keuze]: zoals alle TG-armen; de boeken komen uit 31 boeken, en een tweede epoch op dezelfde 12.521
  boekparen leert eerder die boeken dan Nederlands. Verwacht [schatting]: ~12,8 M tokens (RL 12,4 M TG-tokens × 1,03) →
  ~430 stappen × 8,2 s ≈ 1,0 uur, plus ~15 min tellen, laden en compileren; de build meet de tokens.
- Uitvoer: `$G/runs/g4arm_G4-1/` (`adapter.npz`, `adapter.json`, `summary.json`, `metrics.txt`, logs per worker,
  `maxtext/`: de checkpoint van Tunix met de LoRA-parameters, alleen aan het eind), daarna `$G/G4-1/` (`adapter.npz`,
  `adapter.json`, `summary.json`). Na de training controleert de keten rank, alpha en schaal in `adapter.json`.
- **Smoke-test vooraf** (§6): 8 stappen op dezelfde set, dezelfde instellingen, run `g4arm_G4-1_s8`.

## 4. Evaluatie

### 4.1 Vertalen

- `translate.py` in de venv van de MaxText-fork @ `2249474` (`setup_generic.sh venv`, `/dev/shm/translate-generic/venv-fork`; **@ `f4d8615` volgens het amendement van 14 nov ~00:49, §9**),
  model `gemma4-31b` van `translate_models.yml`, `--prompt natural-v1`, checkpoint `$R/judge/gemma-4-31b-it-maxtext`
  (dezelfde omzetting als die van de training: `_METADATA` byte-gelijk, `HANDOVER-g4lora.md` §1), greedy, elke worker
  een TPU van zijn eigen 4 chips.
- **Lengteklassen tiny tot en met xlong** (prefill ≤ 640, decodeerlimiet ≤ 832 ≤ 1.024 tokens). Daarmee raakt geen van
  de twee fouten van het decodeerpad de evaluatie: fout 2 (de gedeelde `cache_ar_index` van de fork @ `2249474`) speelt
  alleen bij een decodeerlimiet boven het venster van 1.024, fout 1 (het masker in cache-slots) alleen in MaxText 0.2.5.
  Klasse xxlong (limiet 1.152) wordt niet gebruikt: een segment boven 512 Gemma 4-tokens is een fout van de
  datastap [keuze]. De evaluatie van Gemma 4 op 12/13 nov had ook alleen tiny–xlong [gemeten:
  `gs://t5-dutch-english/mbert/translator/bt-natural/data/eval/{four,flan}/classes.txt`].
- **De adapter samengevoegd in de gewichten bij het laden** (`translate.py --lora`: per module A·B·schaal in float32
  opgeteld bij de kernel en terug naar bf16), niet LoRA tijdens het decoderen [keuze]. Redenen:
  1. hetzelfde pad als alle TG-armen, dus vergelijkbaar met hun uitkomsten;
  2. de samenvoeging is getest tegen qwix' eigen LoRA-forward op MaxText's `DenseGeneral`
     (`tests/test_en_nl_lora_merge.py`), het laagtype van de attention en MLP van Gemma 4;
  3. de decode-engine van de fork kent geen LoRA; qwix in `prefill`/`insert`/`generate` van `maxengine` inbouwen is
     nieuwe, ongeteste code in precies het pad waar op 13 nov twee fouten zaten;
  4. samengevoegd vertaalt even snel als het basismodel.
  Prijs: de update wordt in bf16 afgerond (bij N bleef 45% van de kernelelementen ongewijzigd); het geëvalueerde model is
  een afgeronde versie van de getrainde LoRA, zoals bij alle TG-armen. `translate.py` meldt per run het aantal modules
  (verwacht 410), de relatieve norm van de update en het aandeel gewijzigde elementen (regel `LORA`); het verslag neemt
  die over.
- G4-0 vertaalt dezelfde segmenten in dezelfde keten, zonder `--lora`.

### 4.2 Sets

| set | n | rol |
|---|---:|---|
| WMT24++ en→nl_NL | 960 zonder `is_bad_source` (alle 998 gerapporteerd) | (a1), (b) |
| FLORES-200 devtest | 1.012 | (a1) |
| NTREX-128 | 1.997 | (a1) |
| Europarl-direct | 640 | gerapporteerd |
| FLAN (set `classifier` van `$R/mainset/flan-eval/eval-rows.parquet`) | 1.000 | (b); QE gerapporteerd |
| blind (`blind100.jsonl`: 35 WMT24++, 30 idioom, 35 FLAN) | 100 | (a2), (c) |

Bron, referentie en ids als in het TG-protocol (`$E/data/evalsets.parquet` en de soort `x` van `$S/arms/data/x/data/`);
de segmenten opnieuw geteld met de tokenizer van Gemma 4 voor de lengteklassen. **Niet**: het in-domeindeel (TG-protocol
§4.2.2; met het Engels van Gemma 4 zou het een andere set zijn dan bij de TG-armen, en het beslist niet), PLL, MTLD
[keuze].

### 4.3 Maten

- MetricX-24 hybrid XXL (`metricx_qe.py`, JAX-port, bf16 op de TPU): QE (**beslist (a1)**) en met referentie;
  COMET-22 met referentie (CPU); niet af, lus, Engels over, leeg; de lengte van de uitvoer tegenover de referentie
  (`armeval.length_ratios`).
- **Classifier "origineel of vertaald"** van 10 nov (S @ 653k, `$S/arms/naturalness/classifier/`, nauwkeurigheid
  0,9575; niet opnieuw getraind; TG-protocol §4.2.1) op de vertalingen van WMT24++ (960) en de 1.000 FLAN-rijen. In
  één classify-run: G4-1, G4-0 en, als controle, de bestaande teksten van TG en van Gemma 4 op de v6e
  (`$S/re/classify/g4-base-texts.parquet`, modellen `base` 19,13% en `G4` 30,46% op 13 nov).
- **Blinde beoordeling** (TG-protocol §4.2.4: dezelfde 100 passages, dezelfde rubriek, letterlijk): A/B **G4-1 tegen
  G4-0**, volgorde per passage met seed `en-nl-sft/blind/ab/G4-1/<blind_id>` (`armeval.base_is_a`), pakket in
  `$S/blind/G4-1/` (`judge-input.jsonl`; de sleutel `key.jsonl` alleen in de bucket). Eén verse Claude-subagent met
  alleen de rubriek en de passages, geen tweede ronde; twijfelgevallen blijven gemarkeerd en Yeb kan ze vervangen
  [besluit 13 nov ~07:50: geen Fable]. Daarnaast een pakket **G4-1 tegen TG** (`$S/blind/G4-1-tg/`, seed
  `en-nl-sft/blind/ab/G4-1-tg/<blind_id>`), alleen gerapporteerd; of het beoordeeld wordt, beslist de hoofdsessie.

## 5. Beslisregel (exact)

G4-1 **helpt** als (a1), (a2), (b) en (c) alle vier gelden; anders **helpt G4-1 niet**. Het uitgangspunt van (a2), (b)
en (c) is **G4-0, Gemma 4 31B-it zonder fine-tune**, niet TranslateGemma: de arm moet Gemma 4 zelf verbeteren. TG
verslaan is niet genoeg, want G4-0 verslaat TG al (13 nov: (b) +11,3 pp, blind 56 / 22 / 22, fouten 17 tegen 41).

- **(a1) niet slechter dan een mens op MetricX-24-QE** (ongewijzigd uit het TG-protocol §9, amendement 13 nov): voor elk
  van WMT24++ (de 960 zonder `is_bad_source`), FLORES-200 en NTREX-128 geldt gemiddelde QE(G4-1) ≤ gemiddelde QE van de
  menselijke referentie als kandidaat: **2,3917784375 / 1,4938481225 / 1,8419510766** (onafgerond;
  `gs://t5-dutch-english-use1/mbert/translator/bt-natural/report/refqe.json`, `ref_mean`).
- **(a2) niet meer betekenisfouten dan G4-0**: in de blinde beoordeling G4-1 tegen G4-0 (§4.3) is het aantal passages
  met "fout = ja" bij G4-1 ≤ dat bij G4-0, over dezelfde 100 passages.
- **(b) natuurlijker volgens de classifier dan G4-0**: aandeel "origineel"(G4-1) − aandeel "origineel"(G4-0) ≥ **0,10**,
  op de samengenomen uitvoer van WMT24++ (960) en de 1.000 FLAN-rijen, G4-1 en G4-0 uit dezelfde classify-run.
  Bij G4-0 ≈ 30,5% betekent dat ≈ 40,5% voor G4-1, ~21 pp boven TG [gevolgtrekking uit de v6e-meting].
- **(c) beter dan G4-0 in de blinde beoordeling**: G4-1 wint in minstens 60% van de niet-gelijke oordelen: k / n ≥ 0,60,
  met n het aantal passages met "A beter" of "B beter" en k het aantal waarin G4-1 beter is.
- De blinde beoordeling gebeurt alleen als (a1) en (b) gelden (TG-protocol §4.2.4: "een arm die daarop faalt, helpt
  volgens de regel toch niet"); het pakket wordt altijd gemaakt.
- **Gerapporteerd, beslist niet**: MetricX-QE, MetricX-ref en COMET-22 tegen G4-0 en tegen TG (verschillen per set), (b)
  tegen TG, de lengte van de uitvoer, niet af / lus / Engels over / leeg, Europarl-direct, FLAN-QE, de v6e-uitkomsten van
  Gemma 4 naast G4-0, de LoRA-samenvoeging (§4.1), en de blinde beoordeling tegen TG als die gedaan wordt.
- Vergelijkingen op onafgeronde gemiddelden; een verschil gelijk aan de grens op afronding na (1e-9) telt als op de
  grens (`armeval.TOLERANCE`).

## 6. Smoke-test op de TPU (vóór de volle training)

Op de eerste vrije v4-32 (mbert-mix na `EXIT` van `logs/fork_fix_check_run.log`, of mbert-corpus na `EXIT` van
`logs/flan_g4_pilot_run.log`), met de locks en controles van die VM. Een technische controle, geen uitslag van G4-1:

1. **8 trainingsstappen** op de set van §2 met de instellingen van §3 (`STEPS=8`, run `g4arm_G4-1_s8`), en de
   checkpoint van stap 8. Geslaagd als: alle 4 workers `EXIT 0`, 8 eindige verliezen, piek ≤ 29,5 GiB per chip,
   `adapter.npz`, `adapter.json` (rank 32, alpha 16, schaal 0,5) en `maxtext/g4arm_G4-1_s8/checkpoints/8/commit_success.txt`
   in `$G/runs/g4arm_G4-1_s8/`.
2. **Decoderen van 50 segmenten**: de eerste 50 WMT24++-segmenten (uit de 960) van klasse `small` in volgorde van id,
   op w0, met de adapter van stap 8 (`--lora`) en zonder (G4-0), elk met de code en klassen van §4.1. Geslaagd als: de
   regel `LORA` noemt 410 modules en een relatieve norm > 0; van beide runs zijn ≥ 48 van de 50 vertalingen af, geen lus,
   niet leeg.
3. **MetricX-24 (QE en met referentie) op die 50**, op w0. Geslaagd als alle scores eindig zijn en |gemiddelde QE(G4-1,
   stap 8) − gemiddelde QE(G4-0)| ≤ 0,5 (na 8 stappen in de warmup hoort de adapter de vertaling nauwelijks te veranderen;
   een fout in de samenvoeging gaf onzin).

Verslag `$G/smoke/<tag>/smoke.md` en `.json` (kopie `mbert-ops/translator-sft/g4arm-smoke.md`).

## 7. Volgorde van de volle run

Training (§3) → adapter → vertalen G4-1 en G4-0 (§4.1) → MetricX en COMET → teksten → classify → blinde pakketten →
verslag `$G/report/report.md` met (a1), (b) en de stand van (a2) en (c) (kopie `mbert-ops/translator-sft/g4-1-report.md`).
De blinde beoordeling zelf doet de hoofdsessie (een verse subagent), daarna (a2) en (c) met `g4arm.py tally`.

## 8. Bronnen

TG-protocol (`protocol.md`, §2–§5 en §9); hammond `mbert/LEARNINGS.md`, de vertaler-items vanaf "27B-armen B en N";
`mbert-ops/translator-sft/gemma4-moe-lora-test.md` (geheugentest), `rl-report.md`, `rb3-report.md`, `gemma4-sizes.md`,
`prefill-check.md`, `fineweb2-year-check.md`; `mbert-ops/HANDOVER-g4lora.md`.

## 9. Amendementen na het vastleggen

**Amendement 14 nov 2026 ~00:49 CET: vertalen op de fork met de fix van de gedeelde cache-index** [instructie van de
hoofdsessie 00:50; vastgelegd vóór de bouw van de set, de eerste trainingsstap en elke uitslag]. Aanleiding: de
fork-fix-controle van 14 nov (`mbert-ops/translator-sft/fork-fix-check.md`, `EXIT 0` om 00:43): branch
`fix-decode-cache-shared-index` @ `f4d8615f478b6c18de9df8f2dff1f57f775a2521` (op `2249474`) laat Gemma 4 ook met
slot-hergebruik en een decodeerlimiet boven 1.024 goed decoderen.
- **Wat verandert** (§4.1): `translate.py` draait voor G4-1 en G4-0 in de venv van de fork @ `f4d8615`
  (`FORK_COMMIT=f4d8615… setup_generic.sh venv`, `/dev/shm/translate-generic/venv-fork-f4d8615f4`) in plaats van
  @ `2249474`. Ook de smoke-test (§6).
- **Wat gelijk blijft**: de lengteklassen tiny–xlong (decodeerlimiet ≤ 832): alle evaluatiesegmenten passen erin, dus
  het grotere bereik van de fix is niet nodig, en de klassen blijven die van de eerdere Gemma 4-evaluatie; de rest van
  §4–§6.
- **De training blijft op MaxText 0.2.5** (§3) [keuze]. De fix verandert alleen `_decode_cache_positions` in
  `attention_op.py` (18 regels; de posities in de AR-cache bij het decoderen); een trainingsstap draait geen
  decode-cache en roept die functie niet aan. Ook fout 1 van 0.2.5 (het masker in cache-slots) zit in het decodeerpad.
  Trainen op de fork zou een andere SFT-stack betekenen (de Tunix en qwix die bij de fork horen), waarvan geheugen en
  staptijd op Gemma 4 niet gemeten zijn; de stack van 0.2.5 is gemeten (24,6 GiB, 8,2 s per stap, 13 nov).

**Amendement 14 nov 2026 (tijdstip: de commit van deze sectie op branch `en-nl-g4arm2`): arm G4-2** [besluiten van Yeb
14 nov; uitwerking door een subagent, labels als in de kop]. Vastgelegd vóór de bouw van de set van G4-2, vóór de
eerste trainingsstap en vóór elke uitslag van G4-2. Waar deze sectie niets zegt, gelden §1–§8 voor G4-2 met "G4-2" in
plaats van "G4-1".

*G4-2.1 Aanleiding.* G4-1 hielp niet ((b) −1,4 pp tegen G4-0, `mbert-ops/translator-sft/g4-1-report.md`) en was
volgens Yeb te zwak getraind [besluit 14 nov]: LR 1e-5 met cosine (stap 1: 2,6e-7), verlies 2,31 (stap 1) → 1,07
(gemiddeld stap 41–80) → 0,81 (stap 301–390) [gemeten: `$G/runs/g4arm_G4-1/summary.json`], relatieve update-norm
0,0009. Besluit Yeb: **haalt G4-2 de regel van §5 ook niet, dan stoppen we met fine-tunen en wordt G4-0 de vertaler.**

*G4-2.2 Arm.* Gemma 4 31B-it met LoRA rank 32 / alpha 16 op dezelfde 410 modules als G4-1 (§1). Ongewijzigd: stack
(MaxText 0.2.5 + Tunix + qwix, `run_sft.sh`, 16 chips, één multi-host-job), sequenties van 2.048 met packing, één per
chip, AdamW (β2 0,999, weight decay 0,01), clipping 1,0, `num_vocab_tiling=8`, verlies alleen op het Nederlands en
`<turn|>\n`, prompt `natural-v1` in het template van §2.6.

*G4-2.3 Data* [besluit Yeb: dezelfde data als G4-1 plus FLAN-replay uit de nieuwe Gemma 4-vertaling van FLAN-nl, nooit
de TranslateGemma-versie]. Code `g4arm.py build2`; set `$G/data/G4-2/`.
1. **De voorbeelden van G4-1**, letterlijk: `$G/data/G4-1/train.parquet` en `train-meta.parquet` (sha256 in het
   manifest van G4-2), dus dezelfde 12.521 BT-voorbeelden en 12.521 boekparen, **zonder de held-out boeken** van
   G4-2.5 [keuze: de BT niet opnieuw trekken, zodat de BT-kant van G4-2 die van G4-1 is; BT : boeken wordt daardoor
   ~51 : 49].
2. **FLAN-replay: 20% van de voorbeelden** van G4-2 (replay = round(n × 0,25), n = de voorbeelden van punt 1)
   [keuze]. Onderbouwing: 20% is het aandeel van arm N (TG-protocol §2.3), de enige eerdere arm met replay; N had de
   faalwijze op FLAN (de instructie in de bron uitvoeren in plaats van vertalen) minder dan de armen zonder replay
   (LEARNINGS, armen RB3 en RE: 5 keer uitgevoerd bij RE; G4-1 liet op FLAN vaker Engels staan, 52 tegen 40, en
   zakte op FLAN-"origineel" van 13,5% naar 10,7%). Een kleiner aandeel is niet gemeten. Prijs [gevolgtrekking]: het
   Nederlands van de replay is Gemma 4's eigen vertaling (prompt `flan-v1`), dus vertaald Nederlands; de replay trekt
   de uitvoer op FLAN-achtige invoer naar G4-0 en kan (b) op FLAN drukken.
3. **Bron van de replay**: de volle Gemma 4-vertaling van FLAN-nl (keten `flan_g4_full_run.sh`, flan-nl `fd5e585`,
   fork `f4d8615`), na de nabewerking: de segmenten van plan 2 (`mbert/flan-nl/plan2/`), hun vertalingen
   (`mbert/flan-nl/translations/g4/`), de maten per segment (`mbert/flan-nl/g4-full/report/measures.parquet`) en de
   voorbeelden die `flan_nl.py assemble` hield (`mbert/flan-nl/nl-g4/w*.parquet`). `g4arm.py replay` eist dat
   `g4-full/report/setup.json` Gemma 4 31B-it en `flan-v1` noemt, en weigert elk ander pad; de build weigert zoals in
   §2.5 elke rij waarvan `model` niet Gemma 4 is.
4. **Welke replayvoorbeelden** (paren Engels segment → Nederlands segment, zoals bij N) [keuze]: een segment komt in
   aanmerking als (i) het een S-stuk is (instructie of antwoord) van een voorbeeld dat assemble hield, en geen
   templatesegment (`plan.json`, `template_segments`); (ii) de maten van de nabewerking zeggen: `check` leeg (de
   controle van assemble), af, geen lus, niet `executed`, geen `chatter`, geen Engels over, niet gelijk aan de bron, en
   geen reparatie nodig (`repairs` leeg: dan is de opgeslagen tekst de tekst die assemble gebruikt); (iii) 48–384
   Gemma 4-tokens Nederlands (`tokens_out`; het venster van N, TG-protocol §2.3: geen losse woorden of
   templatestukken); daarna, zoals §2.5: geen segment van de FLAN-evaluatierijen (`eval-rows.parquet`, sets
   `classifier` en `yeb`), geen 13-gram met een evaluatieset of blinde passage, hoogstens 2.048 tokens. `g4arm.py
   replay` trekt uit de kandidaten van (i)–(iii) eerst uniform 30.000 (seed `en-nl-g4arm/0/G4-2/replay-pool`), de
   build daaruit na de rest uniform het aantal van punt 2 (seed `en-nl-g4arm/0/replay`). Taak (`task`): de `source` van
   het eerste gehouden voorbeeld met dat segment.
5. Prompt van de replay: `natural-v1` **zonder** instructiezin (§2.5).
6. **Drie epochs** [besluit Yeb]: `train.parquet` bevat de set drie keer, elke doorgang apart geschud (seeds
   `en-nl-g4arm/0/G4-2/shuffle/<1|2|3>`); MaxText leest het bestand in volgorde (geen eigen schudding,
   `num_epoch=1`), elke host elk vierde voorbeeld [gemeten in de code van 0.2.5: `split_dataset_by_node` op één
   bestand], dus de doorgangen volgen elkaar op. `check` legt alle rijen langs MaxText's eigen SFT-stappen.
7. FineWeb-2-crawlgrens: geen (zoals G4-1).

*G4-2.4 Training* [besluit Yeb: LR 1e-4, korte warmup, daarna constant of langzaam dalend, 3 epochs].
- **LR 1e-4, schema warmup–constant–afbouw** (MaxText `lr_schedule_type=wsd`) [keuze]: lineaire warmup van 0 over de
  eerste **3%** van de stappen, dan **constant 1e-4**, dan in de laatste **10%** lineair naar **1e-5**
  (`learning_rate_final_fraction=0.1`, `wsd_decay_steps_fraction=0.1`, `wsd_decay_style=linear`). Onderbouwing:
  (1) constant na een warmup van 3% bij LR 1e-4 is het recept van QLoRA voor de 33B- en 65B-modellen
  (`github.com/artidoro/qlora`, `scripts/finetune_guanaco_33b.sh` en `_65b.sh`: `--learning_rate 0.0001
  --lr_scheduler_type constant --warmup_ratio 0.03`; ook LoRA rank 64 / alpha 16); (2) het probleem van G4-1 was een
  te kleine update; een constante LR houdt de stap groot over alle drie de epochs (gemiddelde LR ~0,95e-4 tegen
  ~0,5e-5 bij G4-1 [berekening: cosine gemiddeld de helft van de piek]); (3) de afbouw aan het eind omdat het
  geëvalueerde model de laatste stap is (geen keuze van een checkpoint): Hägele e.a. 2024 (arXiv 2405.18392,
  "Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations") vinden dat een constante LR met een
  afkoeling aan het eind zich gedraagt als cosine; 10% is de standaard van MaxText. Warmup 3%: ~40 stappen
  [schatting], de helft van de 80 waarin het verlies van G4-1 viel.
- **Drie epochs**: `STEPS=auto` telt de drie doorgangen (`sft_lora.step_limit`: de kleinste telling van de processen
  min de marge, ~1%). Verwacht [schatting]: ~13,7 M tokens per epoch (G4-1 12,1 M − ~0,4 M held-out boeken + ~2 M
  replay) → ~1.350 stappen × 8,2 s ≈ 3,1 uur, plus ~15 min evaluatie tijdens de training en ~12 min laden en compileren.
  Geheugen als G4-1 (24,3 GiB per chip; LR en schema veranderen het niet).
- **MaxText 0.2.5 voor de training, de fork @ `f4d8615` voor het vertalen**, zoals G4-1 (amendement van 00:49): de fix
  zit alleen in `_decode_cache_positions` van het decodeerpad.
- **Evaluatie tijdens de training** (de directe maat, G4-2.6): op stap 0 (vóór de eerste update; LoRA-B is nul, dus
  het basismodel = G4-0 in de trainingsstack), elke 150 stappen en na de laatste stap: het verlies op drie held-out
  sets (G4-2.5), elk volledig (`generate_padding_batch_eval`: een host zonder data rekent een lege batch mee, zodat
  alle processen even veel stappen doen), token-gewogen (`xent_sum` / `total_weights` van MaxText's `loss_fn`).
- **Smoke-test vooraf** (op mbert-mix, binnen de lock, terwijl de keten op de FLAN-nabewerking wacht): 8 stappen met
  de instellingen van G4-2 op de set van G4-1 (de set van G4-2 bestaat dan nog niet; technisch, geen uitslag),
  evaluatie op stap 0, 4 en 8, run `g4arm_G4-2_s8`. Geslaagd als: alle 4 workers `EXIT 0`, 8 eindige verliezen, piek
  ≤ 29,5 GiB per chip, adapter rank 32 / alpha 16 / schaal 0,5, van elke held-out set drie eindige verliezen met
  ≥ 1 token, en het **verlies van het basismodel (stap 0) op zijn eigen greedy vertalingen ("eigen") ≤ 1,0**
  [keuze]. Die laatste eis is een controle van template en stack: de greedy uitvoer van G4-0 is per token de meest
  waarschijnlijke van het model in de decodeerstack; geeft de trainingsstack daar meer dan 1,0 nat per token, dan
  ziet de training een ander model of een ander formaat dan het vertalen, en heeft trainen geen zin. Faalt de
  smoke-test, dan stopt de keten vóór de training.

*G4-2.5 Held-out data* [besluit Yeb: een held-out stuk van de boeken en de BT-validatie]. Code `g4arm.py heldout`;
`$G/data/heldout/`.
- **Boeken: hele boeken** [keuze; een boek is het paar boek-id's in het id van een boekpaar, `books:<nl>-<en>-b1-<k>`;
  31 boeken met 59–1.425 paren in de set van G4-1 (gemeten)]: de boeken met ≤ 600 paren in volgorde van seed
  `en-nl-g4arm/0/G4-2/heldout-books`, genomen tot er ≥ 500 paren en ≥ 2 boeken zijn. Al hun paren gaan uit de
  training van G4-2. Waarom hele boeken: losse passages uit een boek dat in de training zit, delen namen, personages
  en stijl met de trainingspassages; dan meet de maat ook het onthouden van dat boek. Waarom ≤ 600: geen boek met een
  groot deel van de boekparen (de drie grootste hebben 813–1.425). Voor de maat: hun paren die in de decodeerklassen
  passen (Engels ≤ 512 Gemma 4-tokens, Nederlands ≤ 0,8 × de decodeerlimiet van de klasse van het Engels), hoogstens
  500 (seed `en-nl-g4arm/0/G4-2/heldout-books/sample`).
- **BT-validatie**: de split `validation` van de vijf datasets van §2.1 op dezelfde revisies (sha256 in
  `$G/input/bt-val/revisions.json`), door de regels 1–4 van de pool van §2.1 (lengtegrenzen: die van de train-rijen in
  het verslag van G4-1) en de decodeerpassing van hierboven; daaruit 800 uniform (seed
  `en-nl-g4arm/0/G4-2/heldout-btval`).
- **Eigen**: de greedy vertalingen van G4-0 (dezelfde keten als §4.1) van de 800 Engelse bronnen van de BT-validatie
  (alleen af, geen lus, niet leeg), als doel. Geen maat van kwaliteit: de controle van G4-2.4 en, na de training, hoe
  ver G4-2 van G4-0 af is gegaan.

*G4-2.6 De directe maat: rapporteren, beslist niet* [keuze; voorstel van Yeb was rapporteren of een extra eis].
- Maten, per held-out set (boeken, BT-validatie; BT ook per bron): (i) het **verlies** van G4-2.4 op het originele
  Nederlands, stap 0 (G4-0) tegen de laatste stap (G4-2), met het verloop; (ii) **chrF** (sacrebleu chrF2, over het
  corpus en gemiddeld per segment) van de vertalingen van G4-2 en G4-0 (§4.1, dezelfde klassen en slots) tegen het
  originele Nederlands; daarbij niet af / lus / leeg en de lengte tegen het origineel.
- Vooraf vastgelegde lezing: **"dichter bij het originele Nederlands"** als op beide sets het verlies van de laatste
  stap lager is dan op stap 0 én chrF(G4-2) > chrF(G4-0) (corpus); **"niet dichter"** als op beide sets geen van de
  twee geldt; anders **"gemengd"**, met de getallen.
- Waarom niet beslissen: (1) Yeb legde voor G4-2 dezelfde regels (a1), (a2), (b), (c) tegen G4-0 vast als voor G4-1;
  een extra eis maakt G4-2 en G4-1 onvergelijkbaar; (2) chrF tegen één referentie meet gelijkenis met die tekst, geen
  kwaliteit: bij de TG-armen B en N steeg chrF tegen het origineel op het in-domeindeel (64,3 → 70,8) terwijl MetricX
  slechter werd en beide armen faalden (LEARNINGS, "27B-armen B en N"); op de BT-validatie is het Engels bovendien
  Gemma 4's eigen letterlijke terugvertaling; (3) het verlies op held-out origineel Nederlands is het trainingsdoel
  op ongeziene data: het laat zien of de training generaliseert of onthoudt, en of de uitvoer dichter bij het
  originele Nederlands komt, wat Yeb wil zien; de oordelen over kwaliteit blijven bij MetricX-QE (a1), de blinde
  beoordeling (a2, c) en de classifier (b).

*G4-2.7 Evaluatie en beslisregel* (§4, §5): ongewijzigd, met G4-2 als arm en G4-0 als uitgangspunt: (a1) QE ≤ de
mens (2,3917784375 / 1,4938481225 / 1,8419510766), (a2) fouten(G4-2) ≤ fouten(G4-0), (b) "origineel"(G4-2) −
"origineel"(G4-0) ≥ 0,10 op WMT24++ (960) + FLAN (1.000), (c) ≥ 60% van de niet-gelijke oordelen; blind alleen als (a1)
en (b) gelden; pakket G4-2 tegen G4-0 in `$S/blind/G4-2/` (seed `en-nl-sft/blind/ab/G4-2/<blind_id>`) en tegen TG in
`$S/blind/G4-2-tg/`. **G4-0 = de vertalingen en scores van G4-0 uit de run van G4-1** (`$G/eval/G4-0/`, 14 nov,
tag `g4f1114a`), niet opnieuw vertaald [keuze]: dezelfde `translate.py`, fork `f4d8615`, segmenten
(`$G/data/eval/`), klassen en slots op hetzelfde VM-type, dus wat §1 met "dezelfde keten" bedoelt; zo is het
uitgangspunt van G4-1 en G4-2 hetzelfde. De classifier draait opnieuw in één run over G4-2, G4-0 en (ter informatie)
G4-1, zodat (b) G4-2 en G4-0 uit dezelfde run neemt.

*G4-2.8 Template* (opdracht Yeb: controleren of het template van de trainingsvoorbeelden gelijk is aan dat van het
vertalen). Gemeten 14 nov op de CPU van mbert-mix w3 (tokenizer `google/gemma-4-31B-it` @ `842da37`, code
`43c1e2d`): de prompt van de training (`g4lora.prompt_text`, ook het renderen van `g4lora.TEMPLATE` met jinja2) is
tekst voor tekst gelijk aan die van `translate.py` (`turn.prefix` + `user_text(natural-v1)` + `turn.suffix`) en aan
het eigen chat-template van het model met `add_generation_prompt` (standaard en `enable_thinking=False`: het lege
denkkanaal `<|channel>thought\n<channel|>`); de completion is het Nederlands + `<turn|>\n`, en `<turn|>` (106) is een
stoptoken van het vertalen (`eos_ids` 1, 106, 50). `check` legde dat voor alle 25.042 voorbeelden van G4-1 op
token-niveau vast. Wel anders: het eigen template zet een eerdere modelbeurt (geschiedenis) zónder het lege
denkkanaal (`<|turn>model\nDe kat…<turn|>\n`); dat raakt het vertalen niet, dat altijd met de generatieprompt
begint. De snelle daling van het verlies is niet eigen aan Gemma 4: RL (TranslateGemma 27B, zelfde soort data, LR
1e-5) ging van 2,27 (stap 1–5) naar 0,95 (stap 21–40) en 0,83 (stap 41–80) [gemeten:
`$S/rl/runs/arm27b_rl_RL/summary.json`]. Of die daling een formaat is of iets dat generaliseert, laten de held-out
verliezen en de eis "eigen ≤ 1,0" van G4-2.4 zien.

*G4-2.9 Uitvoer.* Run `$G/runs/g4arm_G4-2/`, adapter `$G/G4-2/`, vertalingen en scores `$G/eval/G4-2/`, held-out
`$G/heldout/{G4-2,G4-0}/`, classifier `$G/naturalness-G4-2/`, blind `$S/blind/G4-2/` en `$S/blind/G4-2-tg/`, verslag
`$G/report-G4-2/report.md` (kopie `mbert-ops/translator-sft/g4-2-report.md`), stappen in `$G/stages/g4-2/`. Keten
`mbert-ops/scripts/g4arm2_run.sh` op mbert-mix.

**Amendement 15 nov 2026 (tijdstip: de commit van deze sectie op branch `en-nl-g4arm3`): arm G4-3** [besluiten van Yeb
15 nov ~08:00 (`mbert-ops/translator-sft/g4-3-spec.md`) en ~08:05 (looptijd); uitwerking door een subagent, labels als
in de kop]. Vastgelegd vóór de bouw van de set van G4-3, vóór de eerste trainingsstap en vóór elke uitslag van G4-3.
Waar deze sectie niets zegt, gelden §1–§8 en amendement G4-2 voor G4-3 met "G4-3" in plaats van "G4-2".

*G4-3.1 Aanleiding.* G4-2 (LR 1e-4) haalde (a1) en (b) niet ((b) +1,0 pp tegen G4-0,
`mbert-ops/translator-sft/g4-2-report.md`). Yeb besliste daarna toch een arm G4-3: alle BT-rijen, ook de lange
(513–2.048 tokens), LR 1e-5, 10% FLAN-replay, drie epochs [besluit 15 nov ~08:00]. Engels vakjargon in de uitvoer is
volgens Yeb geen reden voor een filter [besluit].

*G4-3.2 Bron.* De vijf datasets van §2.1 op de revisies na de publicatie van de lange rijen (15 nov; per dataset één
commit; de kolommen `region`, `vm` en `time` zijn eruit; `mbert-ops/bt-long/publish-report.md`), vastgelegd in
`g4arm.BT_REPOS3`:
`yhavinga/handelingen-tweede-kamer-nl-en-gemma4` @ `e4010d9534789927f14f364d3ee6aa3850dd66b8`,
`yhavinga/openraadsinformatie-nl-en-gemma4` @ `926d94adc3c67a0aec59d13f7bcc78dd94d4601a`,
`yhavinga/rijksoverheid-websites-nl-en-gemma4` @ `958b56a33b65c2a21d081ae5e4ef503b2c8913eb`,
`yhavinga/rechtspraak-nl-en-gemma4` @ `09a8f8c811c99ab34b311d3d03147eb638d6e28c`,
`yhavinga/fineweb-2-nl-en-gemma4` @ `15dc526d3359af2fa8f94b17320c91420b66f1fa`.
Handelingen kreeg om ~08:05 nog een commit die alleen `README.md` veranderde, en een squash; de datafiles zijn byte
voor byte die van de publicatie (`d61e32a`) [gemeten: LFS-sha256 van alle `data/`-bestanden gelijk]. `g4arm.py fetch
--arm G4-3` eist dat elk gedownload bestand de LFS-sha256 van die revisie op de Hub heeft en dat er geen bestand van de
split ontbreekt.

*G4-3.3 Data per doorgang* [besluit Yeb; uitwerking `g4arm.py build3`, set `$G/data/G4-3/`].
1. **Alle BT-train-rijen** van G4-3.2, korte (≤ 512 tokens Nederlands) en lange (513–2.048), zonder hersampling naar
   lengte en zonder chrF-filter [besluit Yeb]. Eruit blijven: rijen met een niet-lege `hard_fails` (zoals G4-1/G4-2)
   [technisch, besluit hoofdsessie], en rijen met een 13-gram van een evaluatieset of blinde passage (§2.1 regel 3)
   [keuze: die regel beschermt de evaluatie, geen kwaliteitsfilter; bij G4-1 kostte hij 1 rij]. Niet toegepast: de
   lengtegrens van R2 (§2.1, bij G4-1 108 rijen; het verslag telt wat hij zou weghalen) en een crawlgrens. Gemeten op
   15 nov (alle rijen, Gemma 4-tokenizer, prompt + antwoord zoals de training): 250.840 train-rijen, 3.118 met
   `hard_fails`, dus 247.722, waarvan 20.180 lang; 144,2 M tokens per doorgang, waarvan 42,9 M in de lange rijen.
2. **Alle boeken**: de boekparen van G4-1 zonder de held-out boeken van G4-2.5 (11.760 paren), letterlijk uit
   `$G/data/G4-1/`, zoals G4-2.
3. **FLAN-replay: 10% van de voorbeelden** (replay = round(n / 9), n = de voorbeelden van 1 en 2) [besluit Yeb], uit
   de Gemma 4-FLAN-nl zoals G4-2.3 punt 3–5 (de plaatsen in de bucket, met `nl-g4/w*.parquet` van 14 nov; dezelfde
   eisen en het venster van 48–384 tokens Nederlands). Elke doorgang krijgt andere replayrijen [besluit hoofdsessie,
   "waar de pool dat toelaat"]: `g4arm.py replay --arm G4-3` trekt uit de kandidaten (G4-2: 534.456) uniform een pool
   van 120.000 (seed `en-nl-g4arm/0/G4-3/replay-pool`); build3 schudt die (seed `en-nl-g4arm/0/G4-3/replay`) en geeft
   doorgang k de rijen (k−1)·r tot k·r; is de pool kleiner dan 3·r, dan komen rijen terug (het verslag zegt het). De
   69 voorbeelden die FLAN-69 op 15 nov vertaalt, zitten niet in `nl-g4` van 14 nov [gevolgtrekking: hun lange
   segmenten vielen buiten het venster; het effect op 1,03 M voorbeelden is verwaarloosbaar]. De sha256 van de
   gelezen invoer staat in `replay.json`.
4. **Drie doorgangen**, elk apart geschud (seeds `en-nl-g4arm/0/G4-3/shuffle/<1|2|3>`), achter elkaar in
   `train.parquet` (zoals G4-2.3 punt 6).
5. **Sequentielengte 4.096** (`max_target_length`) [keuze]. Gemeten: het langste voorbeeld is 3.972 tokens
   (FineWeb-2: prompt met 1.929 tokens Engels, Nederlands 2.048 tokens en `<turn|>\n`); de prompt rond het Engels
   kost 89 tokens. 4.096 is het kleinste veelvoud van 512 dat elk voorbeeld houdt; MaxText's `SFTPromptMasking`
   kapt zonder melding af wat langer is, dus build3 en check stoppen bij een voorbeeld boven 4.096. Packing,
   `per_device_batch_size=1` en 16 sequenties per stap als G4-2; build3 rekent de batches per proces na zoals MaxText
   0.2.5 ze maakt (elk proces elk vierde voorbeeld, grain's first-fit over de 4 rijen van een batch) en daaruit de
   stappen (`sft_lora.step_limit`).
6. `check --arm G4-3`: MaxText's eigen SFT-stappen op elk voorbeeld dat in de set staat, één keer per id (een id moet
   in elke doorgang hetzelfde voorbeeld zijn), en elke rij zijn lengte van de build.

*G4-3.4 Training.* Als G4-2.2/G4-2.4 (rank 32 / alpha 16 op de 410 modules, AdamW, clipping 1,0,
`num_vocab_tiling=8`, MaxText 0.2.5, één multi-host-job over de 16 chips van mbert-corpus), behalve:
- **LR 1e-5** [besluit Yeb], WSD als G4-2: warmup 3%, constant, de laatste 10% lineair naar 1e-6 (LR/10).
- **`max_target_length` 4.096** (G4-3.3 punt 5). Gemeten op 13 nov (`translator-sft/gemma4-moe-lora-test.md`, variant
  v4): Gemma 4 31B, rank 32, 4.096, 16 chips: piek 25,04 GiB per chip, 10,60 s per stap.
- **Evaluatie tijdens de training**: stap 0, elke 750 stappen [keuze: ~10 rondes bij de ~8.000 stappen van de
  schatting] en na de laatste, op vier held-out sets: `books`, `btval` en `own` (de bestanden van G4-2.5) en `btlong`
  (G4-3.6).

*G4-3.5 Smoke-test en schatting* (op mbert-corpus, binnen `final_run.lock`, vóór de volle training). 8 stappen met de
instellingen van G4-3.4 op een smoke-set uit doorgang 1: eerst de 256 langste voorbeelden, dan 512 andere (seed
`en-nl-g4arm/0/G4-3/smoke`), evaluatie op stap 0, 4 en 8; run `g4arm_G4-3_s8`. Geslaagd als: alle 4 workers `EXIT 0`,
8 eindige verliezen, piek ≤ 29,5 GiB per chip, adapter rank 32 / alpha 16 / schaal 0,5, `max_target_length` 4.096,
LR-piek 1e-5, elke stap bevat 16 van de langste voorbeelden (tokens per stap ≥ 16 × het kortste daarvan; geheugen en
stapduur gelden dus voor de langste rijen), de batches per proces van de pipeline zijn die van de naberekening van
build3 op de smoke-set, van elke held-out set drie eindige verliezen met ≥ 1 token, en het verlies van het basismodel
op zijn eigen greedy vertalingen (stap 0) ≤ 1,0 (G4-2.4). Faalt een controle, dan stopt de keten vóór de training.
**Schatting** [schatting]: het tellen van de data vóór de eerste stap (de tijd van de smoke-test plus 6,6 s per miljoen
tokens extra, uit G4-2: 212,5 s voor 12,1 M en 404,2 s voor 41,2 M), laden en compileren (uit de smoke-test), de
stappen van build3 × de mediane stapduur van de smoke-test, en de evaluatierondes × de langste ronde na stap 0.
**Regel** [besluit Yeb 15 nov ~08:05]: is de schatting voor drie doorgangen ≤ 120 uur (~5 dagen), dan traint G4-3 drie
doorgangen; daarboven de eerste twee doorgangen van `train.parquet` (de keten stopt niet om het voor te leggen).

*G4-3.6 De lange validation-rijen: rapporteren, beslist niet* [besluit Yeb: chrF op de lange validation-rijen tegen
G4-0, rapporteren]. Code `g4arm.py heldout3`, `$G/data/heldout3/`. De rijen van de split `validation` van G4-3.2 met
`tokens_nl` > 512, zonder `hard_fails`, zonder 13-gram van een evaluatieset of blinde passage, als voorbeeld
≤ 4.096 tokens, en passend in de klassen van `gemma4-31b-long` (`translate_models.yml`: klassen e512–e2k op het aantal
tokens Engels, het Nederlands ≤ 0,8 × de decodeerlimiet van de klasse, zoals G4-2.5). Alle zulke rijen, zonder
steekproef. Elke rij wordt **heel** vertaald (documentniveau), door G4-3 en door G4-0, met `natural-v1` op de fork
@ `f4d8615` (decodeerlimieten boven 1.024 vragen die fix). Maten: chrF2 tegen het originele Nederlands (corpus en
gemiddeld per rij), totaal en per bron; niet af, lus, leeg en de lengte tegen het origineel; het verlies op
`eval-btlong` op stap 0 tegen de laatste stap. Ze staan in het verslag naast de directe maat van G4-2.6 en tellen niet
mee in de vooraf vastgelegde lezing (die blijft over `books` en `btval`). G4-0's vertalingen van deze rijen maakt
de keten van G4-3 (ze bestonden nog niet). heldout3 controleert ook dat de 800 rijen van de BT-validatie van G4-2.5
in de revisies van G4-3.2 dezelfde rijen zijn (Engels en Nederlands gelijk), en build3 dat geen held-out rij een
train-rij is.

*G4-3.7 Evaluatie en beslisregel* (§4, §5): ongewijzigd, met G4-3 als arm en G4-0 als uitgangspunt: (a1) QE ≤ de mens
(2,3917784375 / 1,4938481225 / 1,8419510766), (a2) fouten(G4-3) ≤ fouten(G4-0), (b) "origineel"(G4-3) −
"origineel"(G4-0) ≥ 0,10 op WMT24++ (960) + FLAN (1.000), (c) ≥ 60% van de niet-gelijke oordelen; blind alleen als (a1)
en (b) gelden, door een verse Claude-subagent; pakket G4-3 tegen G4-0 in `$S/blind/G4-3/` (seed
`en-nl-sft/blind/ab/G4-3/<blind_id>`) en tegen TG in `$S/blind/G4-3-tg/`. G4-0 = de vertalingen en scores uit de run
van G4-1 (`$G/eval/G4-0/`) en de held-out vertalingen uit de run van G4-2 (`$G/heldout/G4-0/`), zoals G4-2.7; `own`
is het bestand van G4-2. De classifier draait opnieuw in één run over G4-3, G4-0 en (ter informatie) G4-2.

*G4-3.8 Uitvoer.* Invoer `$G/input/bt3/`, `$G/input/bt3-val/`, `$G/data/replay-G4-3/`; lange validation-rijen
`$G/data/heldout3/`; set `$G/data/G4-3/`, smoke-set `$G/data/G4-3-smoke/`, verslag `$G/data/report-G4-3.md`; smoke
`$G/smoke3/<tag>/`; run `$G/runs/g4arm_G4-3/`, adapter `$G/G4-3/`, vertalingen en scores `$G/eval/G4-3/`, held-out
`$G/heldout/G4-3/`, lange rijen `$G/long/{G4-3,G4-0}/`, classifier `$G/naturalness-G4-3/`, blind `$S/blind/G4-3/` en
`$S/blind/G4-3-tg/`, verslag `$G/report-G4-3/report.md` (kopie `mbert-ops/translator-sft/g4-3-report.md`), stappen in
`$G/stages/g4-3/`. Keten `mbert-ops/scripts/g4arm3_run.sh` op mbert-corpus.

**Amendement 15 nov 2026 (tijdstip: de commit van deze sectie op branch `en-nl-g4arm4`): arm G4-4** [besluit van Yeb
15 nov ~13:45; uitwerking door een subagent, labels als in de kop]. Vastgelegd vóór de bouw van de set van G4-4, vóór
de eerste trainingsstap en vóór elke uitslag van G4-4; ook vóór de uitslag van G4-3 (die traint nog). Waar deze sectie
niets zegt, gelden §1–§8 en de amendementen G4-2 en G4-3 voor G4-4 met "G4-4" in plaats van "G4-3".

*G4-4.1 Aanleiding.* Yeb besliste een arm G4-4: precies de mix van G4-3 plus Engelse originelen met hun menselijke
Nederlandse vertaling (populaire fictie, de groep "Engels origineel → menselijk Nederlands" van het `-b1-`-deel van
`yhavinga/nedd_x_dataset`), elke doorgang andere romanparen; de keten start direct na G4-3 [besluit]. Bij TranslateGemma
tilde zo'n romandeel arm RE over de grens van (b) (R2 plus 29.394 romanparen: "origineel" 31,7%, +0,126 tegen het
basismodel; `mbert-ops/translator-sft/re-report.md`) [gemeten, daar].

*G4-4.2 Bron.*
- **De set van G4-3** (`$G/data/G4-3/`, amendement G4-3.3) zoals hij is, met zijn verslag `$G/data/report-G4-3.json`
  en zijn replaypool `$G/data/replay-G4-3/replay.parquet`. `build4` legt de sha256 van `train.parquet` en
  `train-meta.parquet` vast en eist: de manifest van G4-3 (set, sequentielengte, seed, drie doorgangen), per doorgang
  de voorbeelden en tokens per soort van `report-G4-3.json`, de doorgangen op volgorde, elk voorbeeld buiten de replay
  in elke doorgang hetzelfde, geen replayrij in twee doorgangen, en dezelfde evaluatie-, blinde en replaybestanden
  (sha256 uit de manifest van G4-3).
- **De romans**: `$S/data/books-b1/b1-selected.parquet` (inventaris van 12 nov, `mbert-ops/translator-sft/books-b1-inventory.md`;
  `yhavinga/nedd_x_dataset` @ `708f34de8ec1ff4172f22b30e5890a5ee01d2ca7`), groep `en_orig` (1.628.217 rijen, 3.046
  Engelse boeken) [besluit: deze groep]. `build4` legt de sha256 vast. De data zijn romans onder auteursrecht; de set
  blijft privé.

*G4-4.3 Data per doorgang* [besluit Yeb; uitwerking `g4arm.py build4`, set `$G/data/G4-4/`].
1. **De delen van G4-3 blijven gelijk** [besluit]: doorgang k van G4-4 bevat alle rijen van doorgang k van G4-3 (BT
   ≤ 512 en BT lang, elke doorgang dezelfde; de 11.760 boekparen; de replayrijen van G4-3 in die doorgang), byte voor
   byte dezelfde voorbeelden.
2. **Romanparen** [besluit: "dezelfde controles, zonder QE, zoals RL"]: de rijcontroles van arm RE (mbert-jax
   `en-nl-sft-re` @ `97c44b6`, `sft_build.py`: `b1_fails`), letterlijk overgenomen: `passes` van de inventaris
   (`real_fails`, zijn lengtegrens en uitlijncontrole, geen herhaald id of Engels), geen herhaald Nederlands, geen
   Markdown-spoor, niet aan één kant schoongemaakt; geen QE-grens. **Eén editie per boek** [besluit]: van een Engels boek
   met meer Nederlandse edities de editie met de meeste rijen die de controles halen, bij gelijkstand het laagste
   Nederlandse id (RE's `b1_editions`). Daarbij twee keuzes voor hetzelfde boek onder een ander id [keuze]:
   (i) **één Engels id per werk**: twee Engelse boek-id's met ten minste 10 gelijke passagebegins (de eerste 80 letters
   en cijfers, kleine letters, van het Engels of van het Nederlands, over alle `en_orig`-rijen zonder herhaald id) zijn
   één werk; het id met de meeste rijen na de controles blijft, bij gelijkstand het laagste. Gemeten op 15 nov (deze
   bron, vóór de bouw): 19 paren id's met ≥ 10 gelijke Engelse begins, bv. Larry Niven `92870`/`127882` met 525, Ken
   Follett `101895`/`128985` met 393; daaronder 1–9 (voorpublicaties en bundels) [gemeten, verkenning op mbert-corpus
   w1]. (ii) **Bijna-dubbele passages**: een rij waarvan het Engelse of Nederlandse begin (zelfde sleutel) al bij een
   eerdere rij hoort (boeken op id, rijen op id; ander boek of hetzelfde) gaat eruit; de inventaris haalde alleen
   exacte herhalingen weg. Verder, zoals G4-3.3 punt 1: geen 13-gram met een evaluatieset of blinde passage; en de
   tokengrens van RE: prompt plus antwoord ≤ 2.048 tokens (natural-v1 in Gemma 4's template).
3. **Per doorgang hoogstens 20 paren per boek, over drie doorgangen verschillende paren, tot 60 per boek** [besluit].
   Per Engels boek de rijen op id, geschud met seed `en-nl-g4arm/0/G4-4/novels/<Engels boek-id>`; daarvan de eerste 60
   die de 13-gram- en tokengrens halen; het r-de (vanaf 0) gaat naar doorgang ⌊r / 20⌋ + 1. Een boek met minder paren
   levert wat het heeft, aan de eerste doorgangen eerst [besluit: "levert wat het heeft"; de volgorde is een keuze].
   Gemeten vóór de 13-gram-, token- en bijna-dubbelgrens: 2.953 boeken, 58.646 / 57.882 / 56.899 paren per doorgang
   (doorgang 1 is RE's trekking met K = 20 in aantal) [gemeten, verkenning].
4. **FLAN-replay: 10% van elke doorgang, berekend over het totaal zoals G4-3** [besluit; uitwerking keuze]: replay_k =
   round(n_k / 9), met n_k = BT + boeken + romanparen van doorgang k (de regel `replay_count` van G4-3.3 punt 3, per
   doorgang, omdat de romanparen per doorgang verschillen). De 28.831 replayrijen van G4-3 in doorgang k blijven; wat
   erbij moet, komt uit G4-3's pool in G4-3's volgorde (seed `en-nl-g4arm/0/G4-3/replay`): de rijen na de 3 × 28.831
   van G4-3, achter elkaar, eerst voor doorgang 1, dan 2, dan 3. `build4` maakt G4-3's trekking opnieuw en eist dat
   die precies G4-3's replayrijen per doorgang geeft. Geen replayrij in twee doorgangen; is de pool te klein, dan
   stopt `build4` (geen herhaling) [keuze].
5. **Drie doorgangen, elk opnieuw geschud** (seeds `en-nl-g4arm/0/G4-4/shuffle/<1|2|3>`), achter elkaar in
   `train.parquet` [besluit: "elk apart geschud"].
6. **Sequentielengte 4.096** zoals G4-3.3 punt 5; packing, `per_device_batch_size=1`, 16 sequenties per stap, de
   batches per proces nagerekend zoals MaxText 0.2.5 ze maakt, en daaruit de stappen. De romanparen blijven onder 2.048
   tokens; het langste voorbeeld is dat van G4-3 (3.972 tokens) [gevolgtrekking uit punt 2].
7. `check --arm G4-4` = G4-3.3 punt 6 (MaxText's eigen SFT-stappen op elk voorbeeld, één keer per id).

*G4-4.4 Training.* Als G4-3.4: rank 32 / alpha 16 op de 410 modules, AdamW, clipping 1,0, `num_vocab_tiling=8`,
MaxText 0.2.5, één multi-host-job over de 16 chips van mbert-corpus; **LR 1e-5, WSD** (warmup 3%, constant, de laatste
10% lineair naar 1e-6); `max_target_length` 4.096; evaluatie op stap 0, elke 750 stappen en na de laatste, op `books`,
`btval`, `own` en `btlong` [besluit: "LR 1e-5 met WSD zoals G4-3, LoRA zoals G4-3"]. **Drie doorgangen** [besluit].
Run `g4arm_G4-4`. [schatting vóór de bouw: ~590 M tokens, ~9.400 stappen, ~29 uur.]

*G4-4.5 Smoke-test en schatting* (op mbert-corpus, binnen `final_run.lock`, vóór de volle training). Als G4-3.5 op de
smoke-set van G4-4 (de 256 langste voorbeelden van doorgang 1, dan 512 andere, seed `en-nl-g4arm/0/G4-4/smoke`), run
`g4arm_G4-4_s8`, met dezelfde controles (o.a. piek ≤ 29,5 GiB per chip, elke stap 16 van de langste voorbeelden, de
batches per proces van de pipeline gelijk aan de naberekening, de vier held-out sets, het verlies op `own` op stap 0
≤ 1,0). Faalt een controle, dan stopt de keten vóór de training. Schatting van drie doorgangen zoals G4-3.5. **Is die
schatting boven 120 uur, dan stopt de keten vóór de training** en beslist de hoofdsessie [keuze: de regel "dan twee
doorgangen" was Yebs regel voor G4-3; voor G4-4 besliste hij drie doorgangen].

*G4-4.6 De lange validation-rijen*: als G4-3.6 (rapporteren, beslist niet). G4-4 vertaalt ze; G4-0's vertalingen zijn
die uit de keten van G4-3 (`$G/long/G4-0/`).

*G4-4.7 Evaluatie en beslisregel* (§4, §5): ongewijzigd, als G4-3.7, met G4-4 als arm en **G4-0 als uitgangspunt**:
(a1) QE ≤ de mens (2,3917784375 / 1,4938481225 / 1,8419510766), (a2) fouten(G4-4) ≤ fouten(G4-0), (b) "origineel"(G4-4)
− "origineel"(G4-0) ≥ 0,10 op WMT24++ (960) + FLAN (1.000), (c) ≥ 60% van de niet-gelijke oordelen; de blinde
beoordeling alleen als (a1) en (b) gelden, door de hoofdsessie (een verse Claude-subagent); pakket G4-4 tegen G4-0 in
`$S/blind/G4-4/` (seed `en-nl-sft/blind/ab/G4-4/<blind_id>`) en tegen TG in `$S/blind/G4-4-tg/`. G4-0 = de vertalingen
en scores van G4-3.7. De classifier draait opnieuw in één run over G4-4, G4-0 en G4-3.
**Daarnaast G4-3, op dezelfde maten, alleen rapporteren** [besluit]: MetricX-QE per set (de getallen van (a1)),
MetricX-ref, COMET-22, de lengte en niet af / lus / Engels over / leeg per set, het aandeel "origineel" uit dezelfde
classify-run en het verschil G4-4 − G4-3, chrF op de held-out sets en de lange validation-rijen, het verlies op de
held-out sets na de laatste stap van beide trainingen, en de blinde uitslag van G4-3 tegen G4-0 als die er is. Dit
beslist niets. De uitvoer van G4-3 komt uit zijn keten (`$G/eval/G4-3/`, `$G/heldout/G4-3/`, `$G/long/G4-3/`,
`$G/G4-3/summary.json`); daarom start de keten van G4-4 pas als die van G4-3 op `DONE` eindigde.

*G4-4.8 Uitvoer.* Set `$G/data/G4-4/`, smoke-set `$G/data/G4-4-smoke/`, verslag `$G/data/report-G4-4.md`, check
`$G/data/check-G4-4.txt`; smoke `$G/smoke4/<tag>/`; run `$G/runs/g4arm_G4-4/`, adapter `$G/G4-4/`, vertalingen en
scores `$G/eval/G4-4/`, held-out `$G/heldout/G4-4/`, lange rijen `$G/long/G4-4/`, classifier `$G/naturalness-G4-4/`,
blind `$S/blind/G4-4/` en `$S/blind/G4-4-tg/`, verslag `$G/report-G4-4/report.md` (kopie
`mbert-ops/translator-sft/g4-4-report.md`), stappen in `$G/stages/g4-4/`. Keten `mbert-ops/scripts/g4arm4_run.sh` op
mbert-corpus, na de keten van G4-3 (`final_run.lock`).

**Amendement 16 nov 2026 (tijdstip: de commit van deze sectie op branch `en-nl-g4arm4`): romanpassages als vijfde
eval-set van G4-4** [besluit van Yeb 16 nov ~07:25 en ~07:35; uitwerking door een subagent, labels als in de kop].
Vastgelegd vóór de bouw van deze set, vóór de eerste trainingsstap van G4-4 (ook vóór zijn smoke-test) en vóór elke
uitslag van G4-4 en G4-3. De trainingsset van G4-4 (`$G/data/G4-4/`, G4-4.3) verandert niet.

*G4-4.9 Aanleiding.* Yeb wilde een held-out set van Engelse originele romans met hun menselijke Nederlandse vertaling.
Hele romans buiten de training zijn er niet: alle 2.931 werken van `en_orig` die de controles van G4-4.3 halen, zitten
in de training van G4-4 (elk met 1–60 paren); de overige 115 boeken van de groep zijn 79 andere Nederlandse edities van
een Engels boek, 22 boeken onder een tweede Engels id van een werk dat al meedoet, en de boeken van 14 Engelse id's
waarvan geen rij de controles haalt [gemeten: `data-g4-4-report.json`, `books.tsv`, `books-groups.tsv`]. Yeb koos
daarom **nieuwe passages uit romans die al in de training zitten** [besluit]. De set heet in protocol en verslag
**"unseen passages of seen novels"**: geen held-out romans. De boeken, personages, namen en vertalers zijn in de training
gezien; het verlies op deze set is daardoor optimistischer dan het op onbekende boeken zou zijn [gevolgtrekking].

*G4-4.10 De set `novels`* [besluit: ~10 boeken × ~50 paren, dezelfde controles en 13-gram-filter als build4, geen
overlap of bijna-dubbel met de training, boeken verspreid over auteurs met een vaste seed; uitwerking `g4arm.py
novels4`, `$G/data/heldout4/`].
1. **Kandidaten**: dezelfde bron (`b1-selected.parquet`, sha256 vastgelegd en gelijk aan die van de set van G4-4) door
   dezelfde stappen als G4-4.3 punt 2 (RE's rijcontroles, één editie per boek, één Engels id per werk, bijna-dubbele
   passages eruit); `novels4` eist dat dat de getallen van `report-G4-4.json` geeft (2.931 boeken, 1.239.634 rijen).
   Daarvan de rijen die **geen romanpaar van G4-4** zijn (de 172.168 ids `novel:<pid>` van `train-meta.parquet`).
2. **Per rij, zoals build4**: geen 13-gram met een evaluatieset of blinde passage, prompt plus antwoord ≤ 2.048 tokens.
   **Daarbovenop** [besluit]: geen rij waarvan het Engelse of Nederlandse begin (de sleutel van G4-4.3 punt 2 (ii): de
   eerste 80 letters en cijfers, kleine letters) gelijk is aan dat van een trainingsvoorbeeld van G4-4 (alle
   1.056.235 rijen: het Engels uit `train-meta.parquet`, het Nederlands uit de assistent-beurt van `train.parquet`),
   en geen rij waarvan het voorbeeld (`messages_row`) in `train.parquet` staat; `novels4` telt beide en eist 0 in de
   uitvoer.
3. **Boeken** [keuze]: de auteur genormaliseerd (kleine letters, alleen letters, de naamdelen gesorteerd, zodat
   "Rand, Ayn" en "Ayn Rand" één auteur zijn); een boek telt mee als het ten minste 100 kandidaatrijen heeft en een
   auteursnaam. Uit de gesorteerde auteurs worden er 10 getrokken (seed `en-nl-g4arm/0/G4-4/novels-eval/authors`), per
   auteur één boek (seed `en-nl-g4arm/0/G4-4/novels-eval/book/<auteur>`): 10 boeken van 10 verschillende auteurs.
4. **Paren**: per boek de kandidaatrijen op id, geschud met seed `en-nl-g4arm/0/G4-4/novels-eval/rows/<Engels
   boek-id>`; de eerste 50 die punt 2 halen. Samen 500 paren, in één volgorde geschud (seed
   `en-nl-g4arm/0/G4-4/novels-eval/order`), als `eval-novels.parquet` in het formaat van de andere eval-bestanden
   (`messages_row`: natural-v1 in Gemma 4's template, het Nederlands als antwoord), met `eval-novels-meta.parquet`
   (pid, boek-id's, auteur, tokens) en `manifest.json` (boeken, aantallen, seeds, sha256 van elke invoer).

*G4-4.11 Gebruik.*
1. **G4-4**: `novels` als vijfde eval-set naast `books`, `btval`, `own` en `btlong` (G4-4.4): verlies op stap 0 (= G4-0
   in de trainingsstack), elke 750 stappen en na de laatste; in de smoke-test (G4-4.5) op stap 0, 4 en 8, met dezelfde
   controle als de andere sets (evaluaties op die stappen, eindig verlies, ten minste één doeltoken). Zoals alle
   eval-sets draait elk proces evenveel batches (het kleinste aantal, zonder padding-batches; `sft_lora.py` @ `a3c9e62`).
2. **Het eindmodel van G4-3** op dezelfde set, alleen het verlies [besluit]: run `g4arm_G4-3_novels` in de keten van
   G4-4, binnen `final_run.lock`, na `setup-tpu` en vóór de smoke-test: G4-3's adapter (`$G/G4-3/adapter.npz`, rank 32 /
   alpha 16) wordt vóór de eerste stap in de LoRA-parameters geladen (`sft_lora.py --init-adapter`), dezelfde
   `max_target_length` (4.096) en dezelfde eval-batches als G4-4; de evaluatie op stap 0 is G4-3's verlies. Om de
   trainingsstack te gebruiken zoals hij is, doet de run één trainingsstap op de smoke-set van G4-4; die stap en de
   evaluatie erna worden niet gebruikt [keuze].
3. **Verslag**: een sectie "Romanpassages (unseen passages of seen novels)" met het verlies van G4-0 (stap 0), van G4-4
   (per eval-stap en na de laatste) en van G4-3 (eindmodel), en de waarschuwing van G4-4.9. **Rapporteren, beslist
   niets**: de set telt niet mee in (a1), (a2), (b), (c), de directe maat (G4-2.6) of de smoke-test-schatting, en is
   geen reden om de training te stoppen of te verlengen.
4. Uitvoer: `$G/data/heldout4/`, run `$G/runs/g4arm_G4-3_novels/`, stappen `novels4` en `g43novels` in
   `$G/stages/g4-4/`.
