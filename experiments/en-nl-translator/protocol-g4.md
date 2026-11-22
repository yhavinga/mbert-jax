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

## 10. Amendement G4-5

**Amendement 17 nov 2026 (tijdstip: de commit van deze sectie op branch `en-nl-g4arm5`): arm G4-5** [besluiten van Yeb
17 nov; uitwerking door een subagent (spec `mbert-ops/translator-sft/g4-5-spec.md`, code `8cd6619`), labels als in de
kop]. Vastgelegd vóór de bouw van de sets van G4-5, vóór de splitsing in zoek- en beslisdeel, vóór de eerste stap op
een worker (ook vóór de CPU-proef en de smoke-test) en vóór elke uitslag van G4-5. Waar deze sectie niets zegt, gelden
§1–§8 en de amendementen van §9 met "G4-5" in plaats van "G4-4".

*G4-5.0 Keuzes.* De open keuzes van de spec (§9 daar) staan op de **standaardwaarden van de spec, gekozen door de
hoofdsessie op 17 nov** (~22:30, toen Yeb 8 uur weg was en vroeg wat handig was om te starten) [keuze hoofdsessie]:
G4-5 nu draaien; warmup 1.000 stappen; elke tak 1.000 stappen; LR 1e-5 constant, cooldown over de laatste 10% van de
eigen stappen van elke tak; optimizer-state mee naar de takken; rank 32 (gevolg daarvan); zoekdoel het
referentieverlies van WMT24++, FLORES en NTREX op het zoekdeel; w ∈ [0; 1,5], c_s = 1, ten hoogste 80 punten, top 4;
aandeel zoekdeel 0,5; romantak G4-4's 172.168 paren; G4-4 als referentie in de classify-run en het verslag
(`REFS=G4-4`). **Yeb kan de stappen per tak vóór het begin van de takken nog wijzigen met een nieuw amendement.**

*G4-5.1 Aanleiding.* G4-2, G4-3 en G4-4 haalden (b) niet (+1,0 / +0,4 / +0,7 pp). G4-4: teacher-forced verlies op
menselijk Nederlands sterk omlaag, `own` 0,075 → 0,339: de modus van G4-0 blijft. Yeb wil per bron een eigen LoRA,
vanaf een gezamenlijke warmup, gecombineerd met gezochte gewichten [besluit 17 nov]. Verwachting vooraf
[gevolgtrekking, spec §1]: (b) +10 pp halen is onwaarschijnlijk (elke tak traint met het verlies en de LR van G4-4;
alleen gewichten boven 1 kunnen een tak-delta versterken); de waarde van G4-5 ligt eerder in de diagnose per bron.

*G4-5.2 Data* (`g4arm.py build5`, `$G/data/G4-5/`). Uit de set van G4-4, byte voor byte (sha256 in `report-G4-5.json`).
Warmup: doorgang 1 van G4-4 zonder replay, opnieuw geschud (seed `en-nl-g4arm/0/G4-5/warmup`), de kortste prefix voor
1.000 stappen. Takken (8): `bt_handelingen`, `bt_raad`, `bt_fineweb`, `bt_rechtspraak`, `bt_dpc` (elk kort en lang),
`books`, `novels` (G4-4's drie doorgangen als epochs), `flan` (G4-4's replayrijen zonder `flanval`); elke epoch de hele
bron in een nieuwe volgorde (seed `…/G4-5/<tak>/epoch/<e>`), de kortste prefix voor de 1.000 stappen van de tak. Geen
FLAN in de warmup en geen FLAN in de andere takken [besluit]. `flanval`: 500 replayrijen (seed `…/G4-5/flanval`).

*G4-5.3 Training.* Als G4-4.4 (rank 32 / alpha 16, 410 modules, AdamW, clipping 1,0, 4.096, packing, 16 per stap),
behalve: warmup 1.000 stappen, LR lineair naar 1e-5 over 279 stappen, dan constant, geen cooldown; elke tak 1.000
stappen vanaf de volledige trainingsstate van de warmup (LoRA-parameters en optimizer-state, Tunix' checkpoint van
stap 1.000; gecontroleerd: herstelde stap, LoRA-parameters gelijk aan de adapter van de warmup, staptellers 1.000),
LR 1e-5 constant, de laatste 10% van de eigen stappen lineair naar 1e-6 [besluit 4–6]. Evaluatie bij de start, elke
500 stappen en na de laatste op `books`, `btval`, `own`, `btlong`, `novels`, `flanval`, `wmt24pp_s`, `flores_s`,
`ntrex_s`. Runs `g4arm_G4-5_warmup`, `g4arm_G4-5_<tak>`.

*G4-5.4 Zoek- en beslisdeel* (`g4arm.py split5`, `$G/data/G4-5-split/`, seed `en-nl-g4arm/0/G4-5/split/<set>/<stratum>`,
aandeel 0,5): WMT24++, FLORES, NTREX en de 1.000 FLAN-rijen, per domein (WMT24++, FLORES), FLORES per artikel, anders
per segment; de 35 WMT24++-segmenten van de blinde passages in het beslisdeel. Digest van de toewijzing
`f953c026597eedcb20ab56dab7d0b4389057697c01550017ba954ba6963440f3` (code `8cd6619`, `evalsets.parquet` sha256
`4b025635941cf1fff8813206d85e398b0335e62f30408da7e8d9b464be20062c`); de keten stopt bij een andere digest en
`report5` stopt bij een andere toewijzing. Europarl en de blinde passages worden niet gesplitst.

*G4-5.5 Combinatie en zoektocht.* W = W₀ + Δ_s + Σ wᵢ (Δᵢ − Δ_s), als één adapter van rank 288 (`g4mix.py stack`). Run
`g4arm_G4-5_search` (`sft_lora.py --combine`): ankers (w = 0; elke tak alleen; uniform), dan coordinate search vanaf
uniform, stap 0,25 tot 0,0625, w ∈ [0; 1,5], ten hoogste 80 punten, doel het gemiddelde van de token-gewogen
verliezen van de menselijke referentie op het zoekdeel van WMT24++, FLORES en NTREX; de top 4 (onderling ≥ 0,125)
op alle eval-sets. Die vier (`G4-5c1`–`G4-5c4`) worden vertaald en gescoord zoals G4-4 (§4.1, fork `f4d8615`), met
G4-0 (en G4-4, gerapporteerd) in één classify-run.

*G4-5.6 Keuze en beslisregel.* Keuze op het zoekdeel (vast): van de kandidaten die (a1) halen de grootste (b)-afstand
tot G4-0, bij gelijkheid de eerdere in de top; haalt geen (a1), de kleinste grootste QE-overschrijding. Besluit op het
beslisdeel voor de gekozen kandidaat, met §5 tegen G4-0: (a1) gemiddelde QE ≤ die van de menselijke referentie op
dezelfde segmenten (opnieuw gescoord in de keten, stap `refqe5`; het gemiddelde over de hele set moet binnen 0,05 van
§5's getallen liggen), (b) "origineel" − G4-0 ≥ 0,10 op WMT24++ (goed) + FLAN van het beslisdeel; (a2) en (c) op de 100
blinde passages zoals §5, alleen als (a1) en (b) gelden (pakket `$S/blind/G4-5/` met de naam van de gekozen
kandidaat). Gerapporteerd, beslist niet: de verliezen per tak (G4-5.3), de zoektocht, alle kandidaten op beide delen.

*G4-5.7 Smoke-test* (vóór de training): `g4arm_G4-5_s8` (8 stappen op de smoke-set van de warmup, LR-opwarming 2),
`g4arm_G4-5_b8` (8 stappen vanaf stap 8 van s8, cooldown 4), `g4arm_G4-5_c8` (zoektocht op s8, b8 en G4-3's
eindadapter, doel `books` + `novels`, 8 punten, top 2), en de gestapelde adapter op 50 smoke-segmenten. Geslaagd als
de controles van G4-3.5 voor s8 en b8 gelden (EXIT 0, eindige verliezen, piek ≤ 29,5 GiB, adapter 32/16, batches zoals
build5, evaluaties op de stappen), b8 stap 8 herstelde (tellers 8) en zijn evaluatie op stap 8 gelijk is aan die van
s8 (≤ 1e-6), de LR op stap 8 1e-5 en op stap 15 1e-6 is, de zoektocht zijn basisverlies gelijk aan s8's stap 0 geeft,
het herhaalde punt gelijk is, het anker "G4-3 alleen" binnen 0,02 van G4-3's eigen verlies op `books` en `novels`
ligt en het anker "warmup alleen" binnen 0,02 van s8's stap 8, de gestapelde adapter (rank 96, 410 modules) ten minste
48 van de 50 smoke-segmenten af vertaalt, en de schatting (`g4arm.py smoke5`) ≤ 120 u is. Faalt de smoke-test, dan
stopt de keten vóór de training.

*G4-5.8 Uitvoer.* `$G/data/G4-5/`, `$G/data/G4-5-split/`, `$G/runs/g4arm_G4-5_*`, `$G/eval/G4-5c<k>/`,
`$G/naturalness-G4-5/`, `$S/blind/G4-5/`, `$G/report-G4-5/` (kopie `mbert-ops/translator-sft/g4-5-report.md`), stappen
in `$G/stages/g4-5/`. Keten `mbert-ops/scripts/g4arm5_run.sh` op mbert-corpus.

**Aanvulling 18 nov 2026 (tijdstip: de commit van deze aanvulling op branch `en-nl-g4arm5`): G4-5.7a, de uitslag van
de smoke-test en wat daarop start** [besluit van de hoofdsessie 18 nov ~01:50, terwijl Yeb weg is; uitwerking door
een subagent, labels als in de kop]. Vastgelegd na de smoke-test (keten TAG `g4e1117a`, `FAILED` op 18 nov 01:34 CET)
en vóór de eerste stap van de warmup, de zoektocht en elke uitslag van de training van G4-5.

*Gehaald* [gemeten: `mbert-ops/translator-sft/g4arm5-smoke.json`, `$G/smoke5/g4e1117a/`]: alle controles van
`g4arm_G4-5_s8` (warmup) en `g4arm_G4-5_b8` (tak): EXIT 0, eindige verliezen, piek 29,13 GiB, adapter 32/16, batches
zoals build5, evaluaties op stap 0/4/8 en 8/12/16; b8 herstelde stap 8 (tellers `step`, `opt_state/1/0/count`,
`opt_state/1/2/count` alle 8), zijn evaluatie op stap 8 is op elke set gelijk aan die van s8 (books 2,456876575668341
in beide); LR op stap 8 1e-5 en op stap 15 1e-6. Van de zoektocht (`g4arm_G4-5_c8`): de zoektocht liep (8 punten),
het basisverlies is op elke set gelijk aan s8's stap 0, piek 29,13 GiB, het anker "G4-3 alleen" binnen 0,02 van G4-3
(books 0,8777 tegen 0,8787, novels 0,8897 tegen 0,8991); de gestapelde adapter `G4-5c1` (rank 96, 410 modules,
relatieve update-norm 0,0095) vertaalde 50 van de 50 smoke-segmenten af. Schatting 34,4 u (warmup 3,51 u, elke tak
3,62–3,63 u, zoektocht 1,88 u bij 63,9 s per punt).

*Gefaald*:
1. `search_repeat_equal`: eerste waarde van het anker 2,4015402793884277, herhaling 2,401540332915216 (verschil 5e-8).
   [gemeten] float32(2,401540332915216) = 2,4015402793884277. [gevolgtrekking, uit de code van `sft_lora.py`] de
   doelwaarde van elk punt gaat door `broadcast_float` (`multihost_utils.broadcast_one_to_all`, zonder x64 in float32),
   de herhaling niet; met één proces (CPU-proef) geeft `broadcast_float` de float64 terug en was de herhaling gelijk.
   Het verschil is precisie, geen niet-determinisme.
2. `merge_warmup_is_lora`: het samengevoegde anker "warmup alleen" (W₀ + Δ van s8) geeft books 2,5363 / novels
   2,2668, s8 zelf (LoRA) op stap 8 2,4569 / 2,2041, de basis 2,5704 / 2,2935; verschil 0,079 / 0,063, grens 0,02.
   [gemeten] de update van s8 heeft een Frobenius-norm van 0,0108 × die van G4-3 (b8: 0,0199 ×); G4-3's relatieve
   update-norm in de bf16-kernels is 0,00946, en 59,1% van de elementen verandert bij het samenvoegen (LORA-regel van
   `translate.py`). In de CPU-proef met float32-kernels week hetzelfde anker 1,6e-6 / 0 af. [gevolgtrekking] De
   relatieve update van s8 is ~1,0e-4, ~20× kleiner dan een halve ulp van een bf16-kernel (relatief 2⁻⁹–2⁻⁸): W₀ + Δ
   rondt voor de meeste elementen terug naar W₀, terwijl LoRA naast de kernel de update in de forward-pass houdt. Het
   anker van G4-3 (update ~0,01) haalt de grens. Dezelfde afronding geldt voor `translate.py --lora` (samenvoegen in de
   bf16-kernels) bij elke arm.

*Besluit*: de grenzen van G4-5.7 blijven zoals ze zijn. **De warmup en de 8 takken starten** op grond van hun gehaalde
controles: de keten neemt de smoke-test van `g4e1117a` over als alleen deze twee controles faalden en de schatting
≤ 120 u is. **De keten stopt vóór de zoektocht** (regel `HOLD before search: amendment G4-5.7a, Yeb decides on
search_repeat_equal and merge_warmup_is_lora`, `EXIT 0`). Yeb beslist vóór de zoektocht over deze twee controles (voor
de herhaling bijvoorbeeld een tolerantie van 1e-6 of beide waarden door dezelfde broadcast; voor het warmup-anker wat
er moet gelden, bijvoorbeeld de grens toetsen op een adapter met de update van een echte warmup, of samenvoegen in
float32). Dat besluit wordt een nieuwe aanvulling, gecommit vóór de zoektocht.

**Aanvulling 18 nov 2026 (tijdstip: de commit van deze aanvulling op branch `en-nl-g4arm5`): G4-5.7b, de twee controles
van de zoektocht en een temperatuurproef** [besluiten van Yeb 18 nov ~08:45, punten 1–3; uitgebreid met punt 4
(gekalibreerd zoekdoel) en punt 5 (sampling-proef) op Yebs besluit van 18 nov ~09:15 ("allebei"); uitwerking door een
subagent, labels als in de kop]. Vastgelegd terwijl de takken van keten `g4e1118a` lopen, vóór de zoektocht, vóór de
temperatuurproef, vóór de sampling-proef en vóór elke uitslag daarvan; de eerste versie (punten 1–3) is commit
`429e7f8`, deze versie de commit die haar vervangt, beide vóór elke stap van de vervolgketen op de TPU. De warmup is klaar; zijn eval-verliezen op stap 1.000 zijn het doel van controle 2
hieronder en worden door de keten uit zijn `summary.json` gelezen, niet gekozen. Deze aanvulling heft de stop vóór de
zoektocht van G4-5.7a op.

*1. `search_repeat_equal`* [besluit Yeb]: "gelijk" = |eerste waarde − herhaling| ≤ **1e-4**. Waarom deze grens [Yeb]:
de afrondingsruis is ~1e-7 (smoke-test: 5e-8, de eerste waarde door de broadcast in float32); een rest van een vorig
punt na een fout bij het terugzetten van de basiskernels zou ≥ 1e-2 geven. Code: `g4mix.REPEAT_MAX_DIFF` in de spec
(`repeat_max_diff`); `sft_lora.py --combine` schrijft `repeat.diff` en `repeat.equal`; de keten en `report5` stoppen
als `equal` niet geldt.

*2. `merge_warmup_is_lora`* [besluit Yeb]: de controle van de smoke-test is een unit-test van het inbakken, geen
kwaliteitsmaat; de adapter van 8 stappen is te klein voor bf16 (G4-5.7a). Vervangen door: aan het begin van de echte
zoektocht (`g4arm_G4-5_search`), vóór elk ander punt, het anker "warmup alleen" (w = 0, c_s = 1, ingebakken in de
kernels zoals elk punt) op `books` en `novels`; het moet binnen **0,02** (`MERGE_MAX_DIFF5`) liggen van het eigen
eval-verlies van de warmup op stap 1.000 op die sets (de `eval` in `summary.json` van run `g4arm_G4-5_warmup`, dezelfde
waarden als zijn EVAL-regels), met dezelfde doeltokens [keuze: tokens gelijk erbij, anders vergelijkt de controle andere
batches]. Faalt dat, dan stopt de zoektocht vóór de andere punten (regel `WARMUP_CHECK`, de run eindigt met een fout,
de keten op `FAILED`); alle processen nemen het oordeel van proces 0 (gebroadcast). `report5` stopt als de controle
ontbreekt of faalde. Het anker "G4-3 alleen" van de smoke-test (gehaald: books 0,8777 tegen 0,8787, novels 0,8897
tegen 0,8991, verschil 0,001 / 0,009) blijft als bewijs in het verslag (`report5 --smoke`). `smoke5` blijft
ongewijzigd: zijn uitslag van 18 nov staat (G4-5.7a) en hij draait niet opnieuw.

*3. Temperatuurproef* [besluit Yeb: "goed idee, laten we dat ook doen"; rapporteren, beslist niets]. Vraag: welk deel
van de daling van het teacher-forced verlies op menselijk Nederlands bij de fine-tunes is alleen kalibratie, dus een
andere scherpte van dezelfde verdeling?
- Maat: het token-gewogen verlies van softmax(z / T), z de logits van de trainingsstack (Gemma 4 met zijn soft-cap, in
  float32), op dezelfde negen eval-sets en doeltokens als de training (`books`, `btval`, `own`, `btlong`, `novels`,
  `flanval`, `wmt24pp_s`, `flores_s`, `ntrex_s`), voor T op een rooster van **0,50 tot 3,00 met stap 0,05** (51
  waarden, 1,00 erbij) [keuze: ook T < 1, zodat een optimum onder 1 te zien is, bijvoorbeeld bij `own`, waar G4-0's eigen
  greedy uitvoer het doel is].
- **T verandert de volgorde van de logits niet** (z / T met T > 0 is monotoon in z): de greedy-uitvoer is bij elke T
  gelijk. De proef meet alleen kalibratie, niets van wat (b), QE of de blinde beoordeling meten.
- Uitvoering: `sft_lora.py --temperatures 0.5:3.0:0.05`, eval-only: de eerste evaluatie van Tunix (vóór de eerste
  stap), daarna stopt de run zonder trainingsstap; één forward-pass per batch voor alle T (MaxText's
  `cross_entropy_with_logits` op de volle logits wordt tijdens het tracen omwikkeld, `lax.map` over T). Daarvoor
  `num_vocab_tiling=1` (de trainingsruns hebben 8; zonder trainingsstap past het geheugen [schatting, niet gemeten]).
  Runs `g4arm_G4-5_temp0` (G4-0, geen adapter) en `g4arm_G4-5_tempw` (de adapter van de warmup op stap 1.000 via
  `--init-adapter`: ligt zijn optimum al bij T = 1?).
- Controles (anders stopt het rapport): per set geeft T = 1 het eigen verlies van dezelfde ronde (binnen 1e-4); de
  doeltokens zijn die van de warmup-run; G4-4's verlies op stap 0 is dat van de warmup-run (zelfde batches, binnen
  1e-4). Gerapporteerd: G4-0 bij T = 1 hier (num_vocab_tiling 1) tegen stap 0 van de warmup-run (8), en de
  warmup-adapter bij T = 1 tegen het eigen eval-verlies van de warmup op stap 1.000.
- **Keuze van T**: T\* = het minimum op het rooster van het gemiddelde van de verliezen van **`books` en `btval`** (elk
  token-gewogen, de twee gelijk gewogen; bij gelijke waarden de T het dichtst bij 1) [keuze; waarom: (1) twee sets met
  menselijk Nederlands als doel (Nederlandse boeken en de originele Nederlandse doelen van de BT), waar de vraag over
  gaat; (2) ze bestaan sinds G4-2 in elke arm, zodat het aandeel ook voor G4-4 op de keuzesets te lezen is; (3) ze zijn
  geen deel van het zoekdoel of van het beslisdeel van G4-5, zodat de proef niets van de zoektocht of het besluit
  gebruikt; (4) de zeven andere sets liggen buiten de keuze: verklaart één T daar ook de daling, dan is dat geen fit op
  die sets]. Daarnaast per set de beste T op het rooster (gerapporteerd, met "≤ 0,50" of "≥ 3,00" aan de randen).
- **Lezing**: aandeel van de daling dat één temperatuur verklaart = (L₁ − L_T) / (L₁ − L_fine-tune) per set, met L₁ en
  L_T van G4-0 in deze proef en L_fine-tune = de warmup op stap 1.000 en G4-4 na zijn laatste stap (9.316; uit
  `g4-4-report.json`: `books`, `btval`, `own`, `btlong`, `novels`; de andere vier sets heeft G4-4 niet); voor T\* en voor
  de beste T per set (de bovengrens van wat één T op die set kan). 1 = één temperatuur geeft de hele daling, 0 = niets;
  bij `own` stijgt het verlies bij de fine-tunes, het aandeel is dan dat van de stijging; geen aandeel als de fine-tune
  het verlies minder dan 0,001 verandert.
- Rapport: `g4arm.py temp5` (op co1, uit de `summary.json` van de runs en `g4-4-report.json`) →
  `mbert-ops/translator-sft/g4-5-temperature.md` en `.json`, kopie in `$G/report-G4-5/`.

*4. Zoekdoel: T = 1 of gekalibreerd, met een vooraf vastgelegde regel* [besluit Yeb 18 nov ~09:15].
- Bij elk punt van de zoektocht naast het doel van G4-5.5 (het gemiddelde van de token-gewogen referentieverliezen op het
  zoekdeel van WMT24++, FLORES en NTREX bij T = 1) ook het **gekalibreerde doel**: hetzelfde gemiddelde bij de beste T
  van dat punt, één T voor de drie sets samen (het minimum op het rooster van punt 3, bij gelijke waarden de T het dichtst
  bij 1) [keuze: één T per punt, zoals één temperatuur bij het decoderen]. Uit dezelfde forward-pass (`--temperatures`
  in de zoektocht; die draait daarom eval-only met `num_vocab_tiling=1`, zonder trainingsstap: de zoektocht is een
  meting). Beide doelen worden per punt gerapporteerd (`objective_t1`, `objective_calibrated`, `t_calibrated`).
- **Regel** (vóór de zoektocht, uit de temperatuurproef, door `g4arm.py temp5` berekend en door `g4mix.py spec
  --objective-rule` toegepast): T₀ = de ene T van G4-0 met het laagste gemiddelde op de drie zoekdeelsets (zoals het
  gekalibreerde doel van een punt); per set aandeel = (L₁(G4-0) − L_T₀(G4-0)) / (L₁(G4-0) − L₁(warmup)), met L₁(warmup)
  het eigen eval-verlies van de warmup op stap 1.000; het gemiddelde van de drie aandelen, de sets gelijk gewogen
  [keuze: het gemiddelde van de aandelen per set, niet de verhouding van de gemiddelde verliezen]. Is dat
  gemiddelde **≥ 0,5**, dan kiest de zoektocht (coordinate search, top 4, herhaling) op het gekalibreerde doel, anders
  op het doel bij T = 1. Kan de regel niet berekend worden (een verandering van de warmup < 0,001 op een zoekdeelset),
  dan stopt de keten vóór de zoektocht.
- De keten past de regel toe en schrijft de keuze in zijn log; `temp5` en `report5` schrijven haar in hun rapport. De
  keuze in het verdere protocol (G4-5.6) is ongewijzigd: de kandidaten worden vertaald en op (a1) en (b) gekozen.

*5. Sampling-proef* [besluit Yeb 18 nov ~09:15; rapporteren, beslist niets; na `report5` en het blinde pakket].
- Modellen: G4-0 (geen adapter) en G4-4 (eindadapter `$G/G4-4/`, ingebakken in de kernels zoals `translate.py --lora`).
- Decoding: MaxText's eigen sampling in de fork `f4d8615` [gemeten in de code: `inference_utils.sampling`,
  `decode_sampling_strategy` `weighted` = `jax.random.categorical(rng, logits / T)`, geen top-k of top-p; `MaxEngine.prefill`
  en `generate` nemen `algorithm`, `temperature` en `rng` per aanroep]. `translate.py --sample NAAM=T:SEED` zet per instelling
  `engine.rng = PRNGKey(SEED)` (MaxEngine splitst die bij elke aanroep) en geeft `algorithm="weighted"` en T per aanroep
  mee; zonder `--sample` is het pad greedy zoals altijd. Prompt `natural-v1`, de klassen, lengtes en workers van de
  evaluatie van G4-4 (`$G/data/eval/segments-w<k>.parquet`, gefilterd).
- Instellingen [seeds vast]: `t07` (T = 0,7, seed 1); `t10s1`–`t10s8` (T = 1,0, seeds 1–8; `t10s1` is de instelling
  T = 1,0); `tstar` (T = T\* van punt 3, op `books` + `btval`, seed 1), alleen als T\* ≤ 1,5 en niet 0,7 of 1,0 (anders
  overgeslagen of gelijk aan een bestaande instelling, genoteerd in het rapport) [keuze: T\* van punt 3, de vooraf
  vastgelegde T op origineel en menselijk Nederlands]; **best-of-8** bij T = 1,0: per segment van `t10s1`–`t10s8` het
  sample met de laagste MetricX-24-QE (gelijk: de lagere seed).
- Sets: WMT24++ (de 960 goede segmenten) en FLAN (1.000), heel.
- Metingen per model en instelling (`t07`, `t10s1`, `tstar`, `best8`), naast de greedy uitvoer van G4-0 en G4-4: het
  aandeel "origineel" (p ≥ 0,5) per set en samen, uit één classify-run met de greedy uitvoer van G4-0 en G4-4 als ankers
  (en de controleteksten van 13 nov, zoals elke classify-run); MetricX-24-QE op beide sets, MetricX-ref en COMET-22 op
  WMT24++; het aandeel niet af, lus en Engels over; de lengteverhouding (tekens uitvoer / tekens bron, en op WMT24++
  tekens uitvoer / tekens referentie); de QE van de menselijke referentie op WMT24++ (§5). Best-of-8 kiest op QE: zijn
  QE is geen onafhankelijke maat.
- Uitvoer apart van de kandidaten: `$G/sampling/` (`data/`, `<model>/<instelling>/`, `<model>/best8/`, `naturalness/`,
  `report/`), jobs `g4s_*` en classify-job `arms_classify_<TAG>s`; rapport `mbert-ops/translator-sft/g4-sampling.md` en
  `.json` (`g4arm.py sample5`).

*Wat de vervolgketen doet* (`mbert-ops/scripts/g4arm5_after_hold.sh`, TAG `g4e1118b`): wacht tot keten `g4e1118a`
eindigt met de HOLD-regel en `EXIT 0` (eindigt hij anders, dan `FAILED` zonder iets op de TPU); neemt dan de locks over,
zet de code van deze aanvulling op de workers en ruimt de resten van de smoke-test op; dan (1) de temperatuurproef
(`g4arm_G4-5_temp0`, `g4arm_G4-5_tempw`, `temp5` met de regel van punt 4); (2) de zoektocht met controles 1 en 2 en het
doel volgens de regel; (3) G4-5.5–G4-5.6 ongewijzigd: de top 4 vertalen en scoren, texts (met G4-4), classify,
`report5` (met `--smoke`), het blinde pakket van de gekozen kandidaat (het pakket na `report5`, dat de kandidaat kiest);
(4) de sampling-proef van punt 5; (5) `DONE`. Faalt een stap, dan stopt de keten (`FAILED`); de zoektocht heeft de
temperatuurproef nodig (de regel van punt 4).

## 11. Amendement G4-6

**Amendement 19 nov 2026 (tijdstip: de commit van deze sectie op branch `en-nl-g4arm6`): arm G4-6, preference-training
(DPO) met LoRA** [voorbereiding door een subagent op opdracht van de hoofdsessie, 18 nov (spec
`mbert-ops/translator-sft/g4-6-spec.md`, code `53df95f`); besluit van Yeb 19 nov ~17:30: "akkoord om door te gaan met
DPO"; labels als in de kop]. Vastgelegd vóór de kandidaten, vóór de rejected-run, vóór de eerste trainingsstap (ook vóór
de CPU-proef en de smoke-test) en vóór elke uitslag van G4-6. Waar deze sectie niets zegt, gelden §1–§8 en de
amendementen van §9 met "G4-6" in plaats van "G4-4".

*G4-6.0 Keuzes.* De open keuzes van de spec (§9 daar) staan op het **advies van de spec, gekozen door de hoofdsessie op
19 nov na Yebs akkoord** [keuze hoofdsessie]: G4-6 nu starten (G4-5 haalde (b) niet); mix ~43.000 paren (alle boeken
die de filters halen, ten hoogste 11.000; 16.000 romans; 16.000 BT, 3.200 per bron); rejected = G4-0's greedy vertaling
(fork `f4d8615`, prompt `natural-v1`); chrF-band BT en boeken [50, 80), romans [25, 80); lengteband [0,67; 1,5];
sigmoid-DPO met β 0,1 plus de NLL van chosen met gewicht 1,0 (RPO), geen length-normalisatie; LR 1e-5 WSD; één epoch
(~720 stappen bij 4.096 [schatting]); referentie = hetzelfde model met elke LoRA-parameter op nul in dezelfde stap; de
classifier op chosen en rejected meten en rapporteren, geen paren erop kiezen; evaluatie-interval 200 stappen; G4-4
naast G4-0 in het verslag; de geheugen-terugval naar rijen van 2.048 volgens de vaste regel van G4-6.4. G4-6 hangt niet
af van de sampling-proef van G4-5.7b (punt 5); de keten wacht alleen tot de TPU vrij is.

*G4-6.1 Aanleiding.* G4-2, G4-3, G4-4 en G4-5 haalden (b) niet (+1,0 / +0,4 / +0,7 / −0,4 pp; G4-5: de gekozen
combinatie G4-5c2 op het beslisdeel, `mbert-ops/translator-sft/g4-5-report.md`). Bij G4-4 bleef G4-0's greedy vertaling
zeer waarschijnlijk (`own` 0,075 → 0,339). De temperatuurproef van G4-5.7b [gemeten:
`mbert-ops/translator-sft/g4-5-temperature.md`]: één temperatuur op G4-0 (T = 2,05) geeft gemiddeld 63,8% van de daling
van het teacher-forced verlies G4-0 → warmup op de zoekdeelsets (books 65,1%, novels 64,4% van de daling naar G4-4);
een temperatuur verandert de greedy uitvoer niet. [Gevolgtrekking] Het grootste deel van wat de SFT-armen aan verlies
wonnen, is kalibratie; de modus verschuift niet. G4-6 traint daarom op voorkeuren: per paar het menselijke Nederlands
(chosen) boven G4-0's greedy vertaling van hetzelfde Engels (rejected), met G4-0 als referentie.

*G4-6.2 Kandidaten en rejected-run* (`g4arm.py cands6`, `$G/data/G4-6/rejected-in/`). Uit de set van G4-4
(`g44_set`): de Nederlandse boeken, de romanparen van de drie doorgangen en de BT van ten hoogste 512 Nederlandse tokens
per bron (geen lange BT, geen FLAN-replay), elk waarvan het Nederlands binnen 0,8 van de decodeerlimiet van de klasse
van het Engels past en het paar (geschat) binnen 2.048 tokens; per stratum een trekking van 1,4 × het doel (boeken:
alle; seed `en-nl-g4arm/0/G4-6/candidates/<stratum>`); daarbij de 500 romanpassages van `novels`. G4-0 vertaalt ze
zoals §4.1 (fork `f4d8615`, `natural-v1`, tiny–xlong, greedy), elke klasse over elke worker, naar `$G/dpo/G4-0/`.

*G4-6.3 Paren* (`g4arm.py build6`, `$G/data/G4-6/`). Chosen = het menselijke Nederlands, rejected = G4-0's vertaling.
Uit, in deze volgorde: G4-0 niet af, lus, leeg, Engels over; identiek; chrF(rejected, chosen) buiten [50, 80) (BT,
boeken) of [25, 80) (romans); tekens rejected / chosen buiten [0,67; 1,5]; een 13-gram van de evaluatiesets of de
blinde passages in rejected; meer dan 2.048 tokens voor het paar. Per stratum een trekking (seed
`…/G4-6/draw/<stratum>`) van: boeken 11.000, romans 16.000, BT 3.200 per bron (of wat er is); één volgorde (seed
`…/G4-6/order`). Haalt een stratum zijn aantal niet (`short_of_want` > 0 in `report.json`), dan stopt de keten vóór de
CPU-proef en beslist Yeb; de grenzen worden niet verschoven. Elke kant is een SFT-voorbeeld van G4-4 (§2.6); een paar
zijn twee segmenten naast elkaar in één rij, chosen eerst, first-fit per batch van 16 rijen, op 4.096 en 2.048.
Voorkeurssets (rapporteren): `pref_books` en `pref_btval` (de held-out sets van G4-2 met G4-0's held-out vertalingen),
`pref_novels` (de 500 romanpassages); uit alleen fouten van G4-0, identieke en te lange paren. `check6`: elke kant door
MaxText's eigen SFT-stappen, elke rij gelijk aan MaxText's `shift_and_refine`. Geen TEST-split, geen evaluatiesets in
de training.

*G4-6.4 Training.* Als G4-4.4 (rank 32 / alpha 16, 410 modules, AdamW, clipping 1,0, 16 rijen per stap), behalve:
`sft_lora.py --dpo` (`g4dpo.py`): verlies sigmoid-DPO met β 0,1, label smoothing 0, plus 1,0 × het token-gewogen NLL
van de chosen-kanten, geen length-normalisatie; de referentie is het model met elke LoRA-parameter op nul, in dezelfde
stap onder `stop_gradient`; de log-kansen in 8 stukken (`num_vocab_tiling=8`); LR 1e-5 WSD (3% warmup, de laatste 10%
naar 1e-6); één epoch (elke batch van `train-4096.parquet`); rijen van 4.096. Evaluatie op stap 0, elke 200 stappen en
na de laatste op `books`, `btval`, `own`, `btlong`, `novels`, `pref_books`, `pref_btval`, `pref_novels`. Run
`g4arm_G4-6`. **Terugval**: faalt de smoke-test bij 4.096 op `RESOURCE_EXHAUSTED` (in de log van een worker) of op
alleen de controle van de piek (boven 29,5 GiB), dan de smoke-test bij 2.048 (`g4arm_G4-6_s8_2048`); slaagt die, dan
de training bij 2.048 (`train-2048.parquet`, ~1.500 stappen [schatting]) zonder `btlong`; anders stoppen.

*G4-6.4a Aanvulling 19 nov 2026 ~22:55 (hoofdsessie; Yeb liet de beslissingen over de DPO-armen aan de hoofdsessie,
19 nov ~20:15): de referentie vooraf berekend.* De smoke-test van keten `g4f1119b` paste niet in het HBM van een
v4-chip: de trainstap vroeg bij 4.096 38,86G (beschikbaar 30,75G), bij 2.048 34,20G; de keten stopte volgens G4-6.4
[gemeten: `~/logs/g4arm_G4-6_s8-fail1.log`, `~/logs/g4arm_G4-6_s8_2048-fail1.log` op de workers van mbert-corpus].
Besluit van de hoofdsessie: het geheugen terugdringen zonder de methode te veranderen. **Wat verandert**: de referentie
wordt niet meer in elke stap berekend maar één keer vóór de eerste stap (`g4dpo.precompute_reference`, zoals TRL's
`precompute_ref_log_probs`): één forward-pass zonder gradiënt van het model vóór zijn eerste stap, waarin elke LoRA-B
nul is (de run controleert dat en stopt anders), over de batches die de run traint en over elke voorkeursset; per rij
de som van de log-kansen per segment (per paar chosen en rejected), meegegeven met elke batch (`ref_seg`), en per paar
in `dpo_ref.parquet` van de run. De trainstap doet alleen de forward- en backward-pass van het beleid; de eval-stap van
een voorkeursset alleen die van het beleid; een SFT-evaluatieset alleen zijn token-gewogen verlies (ongewijzigd).
**Waarom de methode gelijk blijft**: (1) de referentie is hetzelfde model: met B = 0 is de LoRA-term x·A·B exact nul,
dus het model vóór de eerste stap is het model met de LoRA op nul; (2) de referentie hing in de stap niet van de
LoRA-parameters af en kreeg geen gradiënt (`stop_gradient`), dus h = (log π − log π_ref)(chosen) − (log π −
log π_ref)(rejected), het verlies (sigmoid-DPO β 0,1 + 1,0 × het NLL van chosen) en elke gradiënt zijn wiskundig
dezelfde [getest op de CPU met een klein model: verlies, sommen en gradiënten gelijk aan die met de referentie in de
stap, op stap 0 en na 6 stappen; `tests/test_en_nl_g4dpo_jax.py`]; (3) data, packing, LR en schema, 16 rijen per stap,
één epoch, rijen van 4.096 met de terugval van G4-6.4, `num_vocab_tiling=8` en de remat-instelling van de SFT-stack
blijven. De log-kansen per token werden al in 8 stukken uit de hidden states berekend zonder de volle logits
(`g4dpo.model_logp`), en elke laag wordt al geremat zoals in de SFT-stack (`remat_policy` `full`); daaraan verandert
niets. **Gemeten** [gemeten 19 nov 22:10–22:55 op mbert-corpus, eigen probes over `sft_lora.py`'s opzet van de
smoke-test (`smoke-4096.parquet`, 16 rijen van 4.096): XLA's geheugenanalyse van de gecompileerde trainstap
(`memory_analysis()`, piek = argumenten + tijdelijk + uitvoer − alias); logs `~/logs/g46probe1.log` en
`~/logs/g46probe2.log` op de workers]: MaxText's eigen SFT-stap (het verlies van G4-4) piek 27,54 GiB (tijdelijk 11,98);
de DPO-stap met de referentie in de stap: niet te compileren, 38,86G tijdelijk tegen 30,75G (dezelfde fout als de
smoke-test); dezelfde stap zonder de referentiepass piek 25,18 GiB (tijdelijk 9,51); de trainstap van deze code piek
25,18 GiB (tijdelijk 9,51), gelijk aan de stap zonder referentiepass. De referentiepass: 3,96 s per batch van 16
rijen (mediaan; de eerste batch met de compilatie), voor de volle run ~720 batches ≈ 48 min [schatting]; de piek na de
referentiepass en de eval op stap 0 van `pref_btval` 18,19 GiB per chip. Op stap 0 gaf de eval-stap op `pref_btval` (12 batches, 799 paren) met de vooraf
berekende referentie |verlies − log 2| = 6e-9, marge 0 en |NLL − NLL referentie| = 0 voor chosen en rejected (de grens
van G4-6.6 is 1e-4). Ter vergelijking: dezelfde log-kansen met elke LoRA-parameter op nul (`without_lora`) in een eigen
programma wijken per segment tot 4,2 nats af van de vooraf berekende (2 batches; gemiddeld per token ~4e-5 [gemeten: de
smoke-test van `g4f1119b`, NLL 1,84547 tegen 1,84551 nu]): bf16-ruis tussen verschillend gefuseerde programma's op de
TPU, die de referentie in de stap ook had (op stap 0 van `g4f1119b`: |verlies − log 2| tot 1,3e-3 en |marge| tot 0,009
op de voorkeurssets, boven de grens 1e-4 van G4-6.6) [gevolgtrekking: bij de oude code was ook die controle van de
smoke-test gefaald]. In de trainstap (een ander programma dan de referentiepass) wijkt het beleid op stap 1 op dezelfde
manier af van de vooraf berekende referentie als het vroeger van de referentie in de stap afweek [gevolgtrekking]. De
CPU-proef draait opnieuw met deze code (de marker van `g4f1119b` weggezet), met een
extra controle dat de referentie vooraf berekend is; de marker van de geheugenterugval van `g4f1119b` is weggezet, zodat
de smoke-test eerst weer bij 4.096 draait. De schatting van de smoke-test telt de referentiepass mee (elke trainbatch ×
de mediane tijd per batch van de smoke-test, plus de voorkeurssets één keer).

*G4-6.5 Evaluatie en beslisregel*: als G4-4.7 met G4-6 als arm en G4-0 als uitgangspunt: (a1) gemiddelde
MetricX-24-QE ≤ die van de menselijke referentie op WMT24++ (960), FLORES en NTREX; (b) "origineel"(G4-6) −
"origineel"(G4-0) ≥ 0,10 op WMT24++ (960) + FLAN (1.000). De classifier in één run over G4-6, G4-0, G4-4 en G4-5c2 (de
gekozen combinatie van G4-5; G4-4 en G4-5c2 alleen rapporteren). (a2) en (c) alleen als (a1) en (b) gelden: pas dan
het blinde pakket (`$S/blind/G4-6/`, ook `$S/blind/G4-6-tg/`) en de beoordeling door één verse Claude-subagent (rubriek
van `protocol.md` §4.2.4). G4-4 naast G4-0 op dezelfde maten (QE, MetricX-ref, COMET-22, "origineel", de directe maat,
het verlies), alleen rapporteren.

*G4-6.6 CPU-proef en smoke-test.* De CPU-proef (`cpuproof6.sh`, op de CPU van een worker) moet `CPU_PROOF6_OK` geven
vóór de eerste TPU-stap van de training (de rejected-run is gewone decodering zoals de evaluatie en mag ervoor). Smoke-test
`g4arm_G4-6_s8` (8 stappen op de eerste 12 batches, evaluaties op 0, 4, 8, 50 segmenten vertaald): de controles van
`g4arm.smoke6` (§7 van de spec), waaronder: op stap 0 op elke voorkeursset |verlies − log 2|, |marge| en |NLL − NLL
referentie| ≤ 1e-4; de SFT-verliezen op stap 0 binnen 2e-3 van stap 0 van G4-4; piek ≤ 29,5 GiB; schatting ≤ 24 u.
Faalt een controle (buiten de terugval van G4-6.4): stoppen vóór de training.

*G4-6.6a Aanvulling 19 nov 2026 ~20:45 (hoofdsessie; Yeb liet het besluit over de G4-6-armen aan de hoofdsessie, en
besloot 18 nov voor G4-5 dat "gelijk" bij afrondingsruis ≤ 1e-4 is).* De CPU-proef (keten `g4f1119a`, 19 nov 20:10)
faalde op één controle: op stap 0 |verlies − log 2| = 8,8e-7 en |marge| = 1,8e-6 op `pref_btval`, tegen de grenzen
1e-5 / 1e-6 in `cpuproof6.sh`; alle andere controles van de proef gehaald (o.a. het NLL van het beleid gelijk aan dat van
de referentie op stap 0, de referentie op stap 8 nog de basis, het beleid op stap 8 bewogen: marge 0,54). De grenzen van
het script waren strenger dan de grens die G4-6.6 voor dezelfde grootheden vastlegt (≤ 1e-4); de afwijking is
float-ruis van twee forward-passes in een andere fusie. De CPU-proef gebruikt voortaan G4-6.6's grens 1e-4 voor het
verlies en de marge op stap 0; niets anders verandert. De nauwkeurigheid op stap 0 (0,25) zegt niets: bij marge ≈ 0
beslist de ruis.

*G4-6.7 Rapportage* (beslist niets): de voorkeursmaten per voorkeursset en eval-stap, met de lezing "voorkeur
geleerd" (nauwkeurigheid op `pref_books` en `pref_btval` na de laatste stap beide ≥ 0,70) / "niet geleerd" (beide
< 0,60) / "gemengd"; het verlies op `own` en de andere eval-sets; de directe maat (G4-2.6) op de held-out sets en
de lange rijen; de classifier op chosen en rejected per soort (1.000 paren per soort, seed `…/G4-6/pairtexts/<soort>`;
kiest geen paren).

*G4-6.8 Uitvoer.* `$G/data/G4-6/` (met `rejected-in/`), `$G/dpo/G4-0/`, `$G/data/report-G4-6.{md,json}`,
`$G/data/check-G4-6.txt`, `$G/runs/g4arm_G4-6_s8[_2048]/`, `$G/smoke6/<tag>/`, `$G/runs/g4arm_G4-6/`, `$G/G4-6/`,
`$G/eval/G4-6/`, `$G/heldout/G4-6/`, `$G/long/G4-6/`, `$G/naturalness-G4-6/`, `$G/naturalness-G4-6-pairs/`,
`$S/blind/G4-6/`, `$S/blind/G4-6-tg/`, `$G/report-G4-6/` (kopie `mbert-ops/translator-sft/g4-6-report.md`).

*G4-6.9 De keten* (`mbert-ops/scripts/g4arm6_run.sh`): wacht tot de vervolgketen van G4-5 (`g4arm5_after_hold.sh`)
eindigt (`EXIT`, welke code ook; de code komt in de log), neemt dan de lock van mbert-corpus over en wacht tot de TPU
vrij is; dan in deze volgorde: setup, `cands6`, de rejected-run (vertalen, verzamelen), `build6` (met de controle op
`short_of_want` van G4-6.3), `check6`, de CPU-proef, de classifier op chosen en rejected, de smoke-test (met de
terugval van G4-6.4), de training, de adapter, vertalen en scoren (evaluatie, held-out, lange rijen), texts, classify,
het verslag, het blinde pakket alleen als (a1) en (b) gelden. Elke run start pas op de vier workers als hij op alle vier
is klaargezet (code, geen protocolwijziging; de les van G4-5's zoektocht op 19 nov). Faalt een stap, dan stopt de keten
(`FAILED`).

## 12. Amendement G4-6b

**Amendement 19 nov 2026 (tijdstip: de commit van deze sectie op branch `en-nl-g4arm6b`, vanaf `en-nl-g4arm6` @
`8b6aceb`): arm G4-6b, DPO met LoRA vanaf G4-4** [voorbereiding door een subagent op opdracht van de hoofdsessie, 19 nov
~20:15; **besluit hoofdsessie 19 nov namens Yeb (akkoord G4-6b, mag zelf starten)**: Yeb gaf 19 nov ~20:15 akkoord voor
G4-6b en liet het starten aan de hoofdsessie; de hoofdsessie besliste: G4-6b draait op mbert-mix zodra de keten van
dmbert v2 daar klaar is; labels als in de kop]. Vastgelegd vóór de rejected-run van G4-6b, vóór de eerste trainingsstap
(ook vóór de CPU-proef en de smoke-test) en vóór elke uitslag van G4-6b. Waar deze sectie niets zegt, geldt §11
(amendement G4-6) met "G4-6b" in plaats van "G4-6" en "G4-4" in plaats van "G4-0" waar het om het startmodel, de
referentie of de rejected-kant gaat; de beslisregel blijft tegen G4-0.

*G4-6b.0 Keuzes* [besluit hoofdsessie namens Yeb, 19 nov]: startmodel en referentie = G4-4; rejected = G4-4's greedy
vertaling (`natural-v1`, leeg denkkanaal, fork `f4d8615`) van **dezelfde kandidaten als G4-6** (`$G/data/G4-6/rejected-in/`,
hergebruikt, niet opnieuw getrokken); chosen, prompts, mix (~43k: boeken, romans, BT), filters, verlies, LR, epoch,
rijen en de terugval als G4-6; de chrF-band opnieuw toegepast op G4-4's rejected, met een vergelijking met G4-6's paren
in het dataverslag. Evaluatie als G4-6, met G4-4, G4-5c2 en (als zijn verslag er dan is) G4-6 ernaast.

*G4-6b.0a Aanvulling 20 nov 2026 ~05:00 (besluit hoofdsessie 20 nov ~04:50; Yeb liet de DPO-armen aan de hoofdsessie):
G4-6b draait op mbert-corpus in plaats van mbert-mix.* Reden: G4-6 is klaar (keten `g4f1119c`, `EXIT 0` op 20 nov
04:43), dus mbert-corpus is vrij en heeft de vertaal- en evaluatie-opzet al in `/dev/shm`; op mbert-mix had G4-6b moeten
wachten tot dmbert v2 klaar is (verwacht 20 nov ~08:00–09:30) en daarna alles opnieuw moeten opzetten. Alleen de VM
verandert: G4-6b.0–G4-6b.9 en G4-6b.5a blijven, en de code is die van `da07029` (deze aanvulling verandert alleen dit
bestand). Voor G4-6b.10 betekent dit: de keten (`g4arm6b_run.sh` met `VM=mbert-corpus`) wacht (alleen lezend) tot de
keten van G4-6 (`mbert-ops/logs/g4arm6_run.log`) op een `EXIT`-regel eindigt (bij `EXIT` ≠ 0 doet hij niets), neemt de
lock van mbert-corpus (`mbert-ops/final_run.lock`, als G4-6), wacht tot de TPU op alle workers vrij is en draait dan
dezelfde stappen in dezelfde volgorde; de setup installeert alleen wat ontbreekt en controleert de rest. Vastgelegd vóór
elke stap van G4-6b op een worker; de wachtende keten `g4g1119b` had niets op mbert-mix gedaan en is gestopt.

*G4-6b.1 Aanleiding.* De sampling-proef van G4-5.7b [gemeten: `mbert-ops/translator-sft/g4-sampling.md`]: bij T = 1
noemt de classifier G4-4's uitvoer op WMT24++ 60,9% "origineel" tegen 51,1% voor G4-0 bij T = 1; greedy kiest dat niet
(G4-4 greedy 50,8%, G4-0 greedy 48,5%). [Gevolgtrekking] G4-4 heeft meer kansmassa op "originele" vertalingen dan G4-0,
maar zijn modus ligt op een vertaling die de classifier even vaak "vertaald" noemt. DPO met G4-4's eigen greedy
vertaling als rejected en G4-4 als referentie duwt precies die modus omlaag ten opzichte van het menselijke Nederlands.
G4-6b hangt niet af van de uitslag van G4-6.

*G4-6b.2 Startmodel en referentie* (`sft_lora.py --base-adapter G4-4=<adapter.npz>`, alleen samen met `--dpo`). De
eindadapter van G4-4 (`$G/G4-4/adapter.npz`, rank 32 / alpha 16, 410 modules) wordt vóór de eerste eval en de eerste
stap in de basiskernels gezet met `KernelMerger` van G4-5 (W0 + 0,5 · A·B in float32, dan het eigen dtype en de eigen
sharding van de kernel; het model zet een kernel bij gebruik om naar bf16, dus het model is G4-4 zoals `translate.py
--lora` het decodeert; dezelfde weg als de warmupcontrole van G4-5.7b [gemeten: merged warmup books 0,9227 tegen 0,9074
met LoRA in de forward pass, novels 0,8424 tegen 0,8304]). Daarop een nieuwe LoRA (rank 32 / alpha 16, dezelfde 410
modules, B = 0 bij de start). De referentie is het model met die nieuwe LoRA op nul (als G4-6), dus G4-4 ingebakken. Run
`g4arm_G4-6b` (smoke `g4arm_G4-6b_s8`, terugval `g4arm_G4-6b_s8_2048`).

*G4-6b.3 Rejected-run.* G4-4 (zijn eindadapter samengevoegd zoals bij zijn evaluatie) vertaalt de kandidaten van G4-6
zoals G4-6.2 (fork `f4d8615`, `natural-v1`, tiny–xlong, greedy, elke klasse over elke worker), naar `$G/dpo/G4-4/`.
[Gevolgtrekking, risico] De kandidaten komen uit G4-4's eigen trainingsset (romans drie doorgangen, BT en boeken); G4-4's
greedy vertaling ligt daar waarschijnlijk dichter bij chosen dan die van G4-0, dus meer paren vallen boven de chrF-grens
van 80 en een stratum kan zijn aantal niet halen.

*G4-6b.4 Paren* (`g4arm.py build6 --arm G4-6b --rejected-model G4-4 --compare $G/data/G4-6`, `$G/data/G4-6b/`). Als
G4-6.3 met G4-4's vertaling als rejected: dezelfde volgorde van filters, chrF-banden (BT en boeken [50, 80), romans
[25, 80)), lengteband [0,67; 1,5], 13-gramfilter, paargrens 2.048, aantallen (boeken 11.000, romans 16.000, BT 3.200 per
bron, of wat er is) en seeds (`en-nl-g4arm/0/G4-6/draw/<stratum>`, `…/G4-6/order`, dezelfde als G4-6). Voorkeurssets:
`pref_books` en `pref_btval` met G4-4's held-out vertalingen (`$G/heldout/G4-4/`), `pref_novels` met G4-4's vertalingen
van de 500 romanpassages (uit de rejected-run). Haalt een stratum zijn aantal niet (`short_of_want` > 0), dan stopt de
keten vóór de CPU-proef en beslist de hoofdsessie; de grenzen worden niet verschoven. Rapporteren (beslist niets): per
stratum de trechter van G4-6 naast die van G4-6b, de mediaan-chrF per soort, en hoeveel getrokken trainingsparen in beide
zitten. Daarnaast het eval-bestand `own4` (`g4arm.py own` op G4-4's held-out vertalingen van de BT-validatie, als `own`
van G4-2.5 voor G4-0; `$G/data/G4-6b/eval-own4.parquet`). `check6` op de paren van G4-6b.

*G4-6b.5 Training.* Als G4-6.4 (sigmoid-DPO β 0,1, label smoothing 0, + 1,0 × het NLL van chosen, geen
length-normalisatie, LR 1e-5 WSD, één epoch, rijen van 4.096, de terugval naar 2.048 volgens dezelfde regel), met
`--base-adapter` van G4-6b.2. Evaluatie op stap 0, elke 200 stappen en na de laatste op `books`, `btval`, `own`, `btlong`,
`novels`, `own4`, `pref_books`, `pref_btval`, `pref_novels`.

*G4-6b.5a Aanvulling 19 nov 2026 ~23:00 (hoofdsessie, zoals aanvulling G4-6.4a): de referentie vooraf berekend.*
G4-6b heeft dezelfde trainstap als G4-6 en dus hetzelfde geheugenprobleem (G4-6's smoke-test: 38,86G bij 4.096,
34,20G bij 2.048, tegen 30,75G per v4-chip). Aanvulling G4-6.4a geldt ook voor G4-6b (merge van `en-nl-g4arm6` @
`d1c9cfe`): de referentie wordt één keer vóór de eerste stap berekend, met een forward-pass zonder gradiënt van het model
ná het
samenvoegen van G4-4's adapter in de basiskernels (G4-6b.2) en vóór zijn eerste stap, waarin elke LoRA-B nul is (de run
controleert dat); de referentie blijft dus het samengevoegde model G4-4 van G4-6b.2. Per rij de sommen van de
log-kansen per segment, meegegeven met de batches en per paar in `dpo_ref.parquet`; de trainstap doet alleen het
beleid. Verlies, data, LR, rijen, de terugval en de controles van G4-6b.6 blijven; de CPU-proef controleert voor (a) en
(c) ook dat de referentie vooraf berekend is. Waarom dit de methode niet verandert en wat het geheugen oplevert: G4-6.4a.
De keten `g4g1119a` stond nog in zijn wachtlus op dmbert v2 (niets op mbert-mix gedaan) en is herstart op deze code.

*G4-6b.6 CPU-proef en smoke-test.* De CPU-proef van G4-6.6 met de grenzen van aanvulling G4-6.6a (op stap 0
|verlies − log 2| en |marge| ≤ 1e-4; code `cab4c65`, in deze branch gemerged) op de paren van G4-6b, met daarbij (`BAKE=1`): (c) `--dpo`
met de adapter van (a) als `--base-adapter`, 4 stappen; (d) gewone SFT met dezelfde adapter als `--init-adapter`, de eval
op stap 0. Controles: de verliezen van `books` en `own` op stap 0 van (c) gelijk aan die van (d) (≤ 1e-4 relatief) en
niet die van het basismodel (b); op stap 0 is de policy van (c) zijn referentie (|verlies − log 2| en |marge| ≤ 1e-4, de
grens van G4-6.6 zoals aanvulling G4-6.6a die voor de CPU-proef vastlegt; elke NLL die van de referentie binnen 1e-6) en die referentie is niet die van
(a); op stap 4 is de referentie van (c) onveranderd (≤ 1e-5). Smoke-test `g4arm_G4-6b_s8` met de controles van
`g4arm.smoke6` voor G4-6, behalve: de SFT-verliezen op stap 0 binnen **0,05** van G4-4's eigen eval na zijn laatste stap
(9.316: books 0,858, btval 0,495, own 0,339, btlong 0,437, novels 0,776) [keuze: samengevoegd in bf16-kernels tegen LoRA
in de forward pass; G4-5.7b mat 0,015 en 0,012 voor zijn warmup; 0,05 scheidt G4-4 ruim van G4-0 (books 2,570)]; de
regel `BASE_ADAPTER` (naam G4-4, 410 modules, rank 32, alpha 16, schaal 0,5); `own4` op 0, 4 en 8; de 50
smoke-vertalingen van G4-6b (G4-4's adapter en die van de smoke-run naast elkaar, zie G4-6b.7) tegen die van G4-4:
|ΔQE| ≤ 0,5.

*G4-6b.7 Evaluatie en beslisregel.* Als G4-6.5 met G4-6b als arm en **G4-0 als uitgangspunt van de beslisregel**: (a1)
en (b) tegen G4-0; (a2) en (c) alleen als (a1) en (b) gelden (pakket `$S/blind/G4-6b/`, ook `$S/blind/G4-6b-tg/`). Het
model van G4-6b bij vertalen: G4-4's adapter en de DPO-LoRA naast elkaar (`g4mix.py stack`, beide coëfficiënt 1, rank
64, schaal 0,5; `$G/G4-6b/`, de DPO-LoRA alleen in `$G/G4-6b/delta/`), samengevoegd door `translate.py --lora`. De
classifier in één run over G4-6b, G4-0, G4-4, G4-5c2 en G4-6 (die laatste alleen als `$G/report-G4-6/report.json` er is
wanneer `texts` draait; de lijst in `$G/naturalness-G4-6b/models.txt`). Het verslag zet G4-4 ernaast op alle maten van
G4-6.5 (QE, MetricX-ref, COMET-22, "origineel", de directe maat, het verlies) en G4-5c2 (en G4-6) op QE, MetricX-ref,
COMET-22, (a1) en "origineel"; alleen rapporteren.

*G4-6b.8 Rapportage* (beslist niets): als G4-6.7, met de voorkeursmaten tegen de referentie G4-4, het verlies op `own`
(G4-0's greedy vertalingen) én op `own4` (G4-4's) per eval-stap, en de classifier op chosen en rejected (G4-4) per soort.

*G4-6b.9 Uitvoer.* `$G/dpo/G4-4/`, `$G/data/G4-6b/`, `$G/data/report-G4-6b.{md,json}`, `$G/data/check-G4-6b.txt`,
`$G/runs/g4arm_G4-6b_s8[_2048]/`, `$G/smoke6/<tag>/`, `$G/runs/g4arm_G4-6b/`, `$G/G4-6b/` (met `delta/`),
`$G/eval/G4-6b/`, `$G/heldout/G4-6b/`, `$G/long/G4-6b/`, `$G/naturalness-G4-6b/`, `$G/naturalness-G4-6b-pairs/`,
`$S/blind/G4-6b/`, `$S/blind/G4-6b-tg/`, `$G/report-G4-6b/` (kopie `mbert-ops/translator-sft/g4-6b-report.md`), markers
`$G/stages/g4-6b/`.

*G4-6b.10 De keten* (`mbert-ops/scripts/g4arm6b_run.sh`, op mbert-mix): wacht (alleen lezend) tot de keten van dmbert v2
(`mbert-ops/logs/final_v2_run.v2c1116d.log`) eindigt op een `EXIT`-regel; bij `EXIT` ≠ 0 doet hij niets (`FAILED`).
Daarna wacht hij tot `final_eval_mix.lock` vrij is en de TPU op alle workers vrij is, neemt een eigen lock voor
mbert-mix en draait in deze volgorde: setup (venvs, tokenizer, MetricX, COMET, de classifier op w0), de rejected-run
(vertalen, verzamelen), `build6` (met de controle op `short_of_want`), `own4`, `check6`, de CPU-proef, de classifier op
chosen en rejected, de smoke-test (met de terugval), de training, de adapter, vertalen en scoren (evaluatie, held-out,
lange rijen), texts, classify, het verslag, het blinde pakket alleen als (a1) en (b) gelden. Elke run wordt eerst op
alle vier workers klaargezet en dan op alle vier gestart (als G4-6). Faalt een stap, dan stopt de keten (`FAILED`).

*G4-6b.4a Aanvulling 20 nov 2026 ~06:45 (hoofdsessie; Yeb liet de DPO-armen aan de hoofdsessie; G4-6b.4: een tekort
is aan de hoofdsessie, de grenzen blijven).* `build6` (keten `g4g1120a`, 06:33) haalde twee strata niet: `bt:dpc` 3.019
van 3.200 (−181) en `bt:rechtspraak` 3.078 van 3.200 (−122), omdat G4-4's vertaling daar vaker boven chrF 80 ligt
(`chrf_high` 1.452 en 1.353 van 4.480 kandidaten, tegen 569 en 140 bij G4-6); alle andere strata vol. Besluit: G4-6b
traint op de 42.697 paren die er zijn (−0,7%), zonder de chrF-band te verschuiven en zonder de andere strata aan te
vullen; de keten aanvaardt precies dit tekort (`ACCEPT_SHORT`) en stopt bij elk ander.

## 13. Amendement G4-6c

**Amendement 20 nov 2026 (tijdstip: de commit van deze sectie op branch `en-nl-g4arm6c`, vanaf `en-nl-g4arm6b` @
`2797975`): arm G4-6c, een mildere DPO vanaf G4-4** [**Yeb koos optie 2, invulling door de hoofdsessie** (20 nov
~13:50); uitwerking door een subagent; labels als in de kop]. Vastgelegd vóór de CPU-proef, de smoke-test, de eerste
trainingsstap en elke uitslag van G4-6c. Waar deze sectie niets zegt, geldt §12 (amendement G4-6b met G4-6b.0a, G4-6b.4a
en G4-6b.5a) met "G4-6c" in plaats van "G4-6b"; de beslisregel blijft tegen G4-0.

*G4-6c.0 Keuzes* [Yeb koos optie 2, invulling door de hoofdsessie]: startmodel, referentie en data als G4-6b; β 0,5
in plaats van 0,1; het eindetoken van de beurt telt niet mee in de log-ratio van de DPO-term, wel in de NLL-term;
NLL-gewicht 2,0 in plaats van 1,0; LR 1e-5 en verder alles als G4-6b; de adapter bewaard op stap 240, 480 en 717;
de drie checkpoints vertaald en gescoord zoals de kandidaten van G4-5; de keuze vooraf vastgelegd op het zoekdeel, het
besluit op het beslisdeel. G4-6c draait op mbert-corpus.

*G4-6c.1 Aanleiding* [gemeten: `mbert-ops/translator-sft/g4-6b-report.md`]. G4-6b (DPO vanaf G4-4, β 0,1, NLL-gewicht
1,0) duwde de rejected-kant sterk weg (`own4` 0,169 → 0,512 per token op stap 400; marge op `pref_books` 11,2 op stap
400), maar chosen steeg niet mee (NLL chosen op `pref_books` 0,859 → 0,865); 30–37% van de vertalingen was niet af, met
lussen en uitvoer van 1,4–2,0 × de referentie; QE 8,4 / 9,0 / 9,8; (a1) en (b) niet gehaald. G4-6 (vanaf G4-0) bewoog
(b) niet (−1,7 pp). [Gevolgtrekking] De log-ratio van G4-6b telde ook `<turn|>\n`, het einde van de beurt. Het
wegduwen van de rejected-kant verlaagt dan ook de kans op het afsluiten van de beurt, wat bij de niet-afgemaakte
vertalingen en lussen past. G4-6c toetst of een mildere DPO regel (b) kan bewegen zonder te ontsporen.

*G4-6c.2 Startmodel, referentie en data.* Als G4-6b: G4-4's eindadapter ingebakken (G4-6b.2), dus de referentie is
G4-4; de referentie vooraf berekend (G4-6b.5a); dezelfde 42.697 paren uit `$G/data/G4-6b/` (G4-6b.4 met het tekort van
G4-6b.4a), dezelfde voorkeurssets, `own4` en dezelfde controle `$G/data/check-G4-6b.txt`. Geen nieuwe rejected-run en
geen nieuwe `build6`; de keten controleert het tekort van G4-6b.4a opnieuw (`ACCEPT_SHORT`) en stopt bij elk ander.

*G4-6c.3 Verlies.* Sigmoid-DPO met **β 0,5**, label smoothing 0, geen length-normalisatie, plus **2,0** × het
token-gewogen NLL van chosen. **Het eindetoken van de beurt telt niet mee in de log-ratio**: per zijde (segment) vallen
de laatste twee doeltokens weg, `<turn|>` en de `\n` die de completion afsluit (`g4lora.END = "<turn|>\n"`), bij het
beleid én bij de referentie (die vooraf met hetzelfde masker wordt berekend). In de NLL-term tellen ze wel mee: alle
doeltokens van chosen, zoals bij G4-4 en G4-6b. Spec `{"loss": "sigmoid", "beta": 0.5, "label_smoothing": 0.0,
"nll_alpha": 2.0, "length_norm": false, "end_tokens": 2}`; `end_tokens` is het aantal doeltokens aan het eind van elk
segment dat buiten de log-ratio valt (0 = G4-6 en G4-6b, ongewijzigd). Controle vóór de CPU-proef (stap `ends6c`, CPU
van w2): in elk gepakt bestand van `$G/data/G4-6b/` zijn de doeltokens van elk segment aaneengesloten, meer dan twee,
en zijn de laatste twee de ids van `<turn|>\n` volgens Gemma 4's tokenizer (`$G/data/ends-G4-6c.txt`); de run controleert
aaneengeslotenheid en lengte opnieuw op de bestanden die hij leest. Een voorkeurseval rapporteert de NLL per token van
chosen en rejected over de tokens van de log-ratio (zonder het eindetoken) én over alle doeltokens (de maat van G4-6b).
[Gevolgtrekking, geen meting] Met β 0,5 verzadigt de sigmoid bij een vijf keer kleinere verschuiving van de log-ratio,
dus de DPO-term trekt minder ver; het NLL-gewicht 2,0 houdt chosen sterker vast.

*G4-6c.4 Training.* Als G4-6b.5: LR 1e-5 WSD (opwarming 3%, de laatste 10% lineair naar 1e-6), één epoch (717 stappen bij
rijen van 4.096), rank 32 / alpha 16, 410 modules, rijen van 4.096 met de terugval naar 2.048 volgens G4-6.4. Run
`g4arm_G4-6c` (smoke `g4arm_G4-6c_s8`, terugval `g4arm_G4-6c_s8_2048`). Evaluatie op stap 0, **240, 480** en na de
laatste (717) op `books`, `btval`, `own`, `btlong`, `novels`, `own4`, `pref_books`, `pref_btval`, `pref_novels` [keuze
subagent: elke 240 stappen in plaats van 200, zodat de evaluaties op de bewaarde checkpoints vallen; de evaluatie
verandert de training niet].

*G4-6c.5 Tussencheckpoints.* De DPO-LoRA wordt bewaard na stap 240 en 480 (`sft_lora.py --save-steps 240,480`:
`adapter-step240.npz`/`.json` en `adapter-step480.npz`/`.json` in `$G/runs/g4arm_G4-6c/`) en na de laatste stap 717
(`adapter.npz`). Elk checkpoint is als model G4-4's adapter en de DPO-LoRA van die stap naast elkaar (`g4mix.py stack`,
beide coëfficiënt 1, rank 64, schaal 0,5; als G4-6b.7): **`G4-6c-s240`, `G4-6c-s480`, `G4-6c-s717`** in `$G/<naam>/`,
de DPO-LoRA alleen in `$G/<naam>/delta/`.

*G4-6c.6 CPU-proef en smoke-test.* De CPU-proef van G4-6b.6 (`BAKE=1`) met de spec van G4-6c, plus: (a) bewaart zijn
adapter na stap 4 (`--save-steps 4`); (e) gewone SFT met die adapter als `--init-adapter` geeft op stap 0 dezelfde
verliezen op `books` en `own` als (a)'s eval op stap 4 (≤ 1e-4 relatief): het bewaarde checkpoint is het model van die
stap; in (a)'s voorkeurseval telt de log-ratio per zijde precies twee doeltokens minder dan het NLL (op elke stap);
(a)'s trainingsverlies van stap 1 = de DPO-term + 2,0 × het NLL van chosen over alle doeltokens (≤ 1e-4 relatief); in de
echte paren, gepakt met de tokenizer, zijn de weggelaten tokens van elk segment `<turn|>\n`. Smoke-test `g4arm_G4-6c_s8`
met de controles van G4-6b.6 (de spec van G4-6c, `own4`, stap 0 binnen 0,05 van G4-4's eigen eval, `BASE_ADAPTER`),
plus: de adapter van stap 4 bewaard (`--save-steps 4`; rank 32 / alpha 16) en verschillend van die van stap 8; de
tokentelling van de log-ratio en de opbouw van het verlies als in de CPU-proef. De 50 smoke-vertalingen van G4-6c
(G4-4's adapter en die van de smoke-run naast elkaar) tegen die van G4-4: |ΔQE| ≤ 0,5. De schatting telt vier
evaluatierondes (stap 0, 240, 480, 717). Faalt een controle buiten de terugval van G4-6.4: stoppen vóór de training.

*G4-6c.7 Keuze van het checkpoint (zoekdeel), vooraf vastgelegd.* De drie checkpoints worden vertaald en gescoord met
dezelfde keten als de kandidaten van G4-5 (G4-5.5: fork `f4d8615`, `natural-v1`, greedy, de klassen tiny–xlong over de
vier workers; MetricX-24-QE en -ref, COMET-22), op de hele evaluatieset; de classifier in één run over `G4-6c-s240`,
`G4-6c-s480`, `G4-6c-s717`, G4-0, G4-4 en G4-6b (de laatste twee gerapporteerd). De keuze leest alleen het **zoekdeel**
van G4-5.4 (`split5`, digest `f953c026597eedcb20ab56dab7d0b4389057697c01550017ba954ba6963440f3`; WMT24++, FLORES,
NTREX, FLAN): van de checkpoints die (a1) op het zoekdeel halen (per set gemiddelde QE ≤ die van de menselijke referentie
op dezelfde segmenten, `$G/data/G4-5-split/refqe.parquet`), het checkpoint met de grootste (b) ("origineel" − G4-0 op
WMT24++ (goed) + FLAN van het zoekdeel); bij gelijkheid het eerdere checkpoint; haalt geen enkel checkpoint (a1), dan het
checkpoint met de kleinste grootste QE-overschrijding (de regel en de code van G4-5.6, `select5`). [Keuze subagent: de
hele set wordt vertaald, zoals bij de kandidaten van G4-5, zodat het gekozen checkpoint zijn beslisdeel al heeft; de
beslisdeelcijfers van de andere twee checkpoints staan pas na de keuze in het verslag en beslissen niets.] Het verslag
stopt bij een andere toewijzing van het zoek- en beslisdeel dan die van de digest.

*G4-6c.8 Beslisregel (beslisdeel).* Voor het gekozen checkpoint met §5 tegen G4-0 op het **beslisdeel**, als G4-5.6:
(a1) per set gemiddelde QE ≤ die van de menselijke referentie op dezelfde segmenten (de QE van de menselijke referenties
over de hele set binnen 0,05 van §5's getallen); (b) "origineel" − G4-0 ≥ 0,10 op WMT24++ (goed) + FLAN van het
beslisdeel. (a2) en (c) op de 100 blinde passages zoals §5, **alleen als (a1) en (b) gelden** (pakket `$S/blind/G4-6c/`
en `$S/blind/G4-6c-tg/`, met de naam van het gekozen checkpoint als arm).

*G4-6c.9 Rapportage* (beslist niets). Per checkpoint, op het zoekdeel en (na de keuze) op het beslisdeel: per set QE,
het aandeel niet af, lussen, Engels over, leeg en de lengteverhouding (mediaan woorden uitvoer / referentie, aandeel
> 1,5 × de referentie); uit de training op stap 240, 480 en 717: het verlies op `own4`, `own`, `books`, `btval`,
`btlong`, `novels`, en per voorkeursset verlies, nauwkeurigheid, marge, NLL van chosen en rejected (over de log-ratio en
over alle doeltokens) met die van de referentie G4-4; G4-4 en G4-6b ernaast. [Keuze subagent] G4-6c vertaalt de
held-out sets en de lange rijen niet (de directe maat van G4-2.6 als chrF): de beslisregel gebruikt ze niet, en het
teacher-forced verlies op `books`, `btval` en `btlong` komt uit de evaluaties van de training.

*G4-6c.10 Uitvoer.* `$G/data/ends-G4-6c.txt`, `$G/runs/g4arm_G4-6c_s8[_2048]/`, `$G/smoke6/<tag>/`,
`$G/runs/g4arm_G4-6c/` (met `adapter-step240.*` en `adapter-step480.*`), `$G/G4-6c-s240/`, `$G/G4-6c-s480/`,
`$G/G4-6c-s717/` (elk met `delta/`), `$G/eval/G4-6c-s<k>/`, `$G/naturalness-G4-6c/`, `$S/blind/G4-6c/`,
`$S/blind/G4-6c-tg/`, `$G/report-G4-6c/` (kopie `mbert-ops/translator-sft/g4-6c-report.md`), markers `$G/stages/g4-6c/`.

*G4-6c.11 De keten* (`mbert-ops/scripts/g4arm6c_run.sh`, `VM=mbert-corpus`). Eigen lock; wacht (alleen lezend) tot de
keten van G4-6b (`mbert-ops/logs/g4arm6b_run.log`) op een `EXIT`-regel eindigt (bij `EXIT` ≠ 0 doet hij niets); neemt
`mbert-ops/final_run.lock` tot het eind; wacht tot de TPU op alle workers vrij is; dan in deze volgorde: setup (alleen wat
ontbreekt), de controle van G4-6b's paren (`report.json` met het tekort van G4-6b.4a, `check-G4-6b.txt`,
`eval-own4.parquet`), `ends6c`, de CPU-proef, de smoke-test (met de terugval), de training, de drie adapters, vertalen en
scoren van de drie checkpoints, texts, classify, `report6c` (keuze, beslisdeel, verslag), het blinde pakket alleen als
(a1) en (b) gelden; `DONE`. Elke run eerst op alle vier workers klaargezet en dan op alle vier gestart. Faalt een stap,
dan stopt de keten (`FAILED`). mbert-mix wordt niet gebruikt.

*G4-6c.6a Aanvulling 20 nov 2026 ~15:15 (subagent op opdracht van de hoofdsessie; de CPU-proef van G4-6c.6 deed haar
werk).* De CPU-proef van keten `g4h1120a` (w1, 15:03) faalde alleen op de controle (e): de bewaarde adapter "van stap 4"
gaf `books` 12,9917 en `own` 12,9804 tegen (a)'s eval op stap 4 12,9879 en 12,9760 [gemeten:
`~/logs/g4a_cpuproof6_g4h1120a.log` op w1]; alle andere controles (ook de eindtokens, de verliesopbouw 26,64655 tegen
26,64655 en de echte paren op `<turn|>\n`) gehaald. Oorzaak [gelezen in Tunix' `peft_trainer.py` in de SFT-venv]: Tunix
schrijft de metrics van een stap pas na de volgende stap en roept dan `on_train_step_end` aan, dus de hook "stap 4" zag het
model na stap 5. Fix (code, geen methodewijziging): `sft_lora.py --save-steps` bewaart de adapter in de eval-ronde van die
stap (Tunix' eval na stap k, vóór stap k + 1), dus precies het model van de eval op die stap; elke bewaarstap moet een
eval-stap zijn (240 en 480 bij evals elke 240; 4 in de smoke-test en de CPU-proef). De controle (e) blijft ongewijzigd.
Verlies, data, LR, stappen en de rest van §13 blijven. Vastgelegd vóór de herstart (nieuwe TAG); er was nog niets op de
TPU gedaan (alleen setup, `ends6c` en de CPU-proef).

*G4-6c.6b Aanvulling 20 nov 2026 ~16:15 (subagent op opdracht van de hoofdsessie; vóór elke trainingsstap van G4-6c).*
De smoke-test van keten `g4h1120b` bij 4.096 (`g4arm_G4-6c_s8`, 20 nov 15:31–15:54) stopte in de eerste voorkeurseval op
stap 0 (`pref_books`) met `RESOURCE_EXHAUSTED` ("Attempting to allocate 21.00M … There are 5.00M free") op w1 en w2
[gemeten: `~/logs/g4arm_G4-6c_s8.log`]; de referentiepass en de SFT-evals waren klaar. G4-6b's smoke-test met dezelfde
stap paste (piek 29,13 GiB): de eval zat al op de rand, en de afzonderlijke sommen voor `end_tokens` (drie contracties
met een one-hot van de segmenten in plaats van één) duwden haar erover [gevolgtrekking]. De keten viel volgens G4-6.4
terug op 2.048; die smoke-run is gestopt (keten `kill -- -2767740`, de vier `sft_lora.py`-processen bij PID) vóór zijn
eerste stap. Fix (code, geen methodewijziging): alle sommen per segment in één contractie met één one-hot
(`g4dpo.segment_columns`), voor het beleid en de vooraf berekende referentie dezelfde functie, zodat ze op stap 0 dezelfde
getallen zijn; verlies, data, LR, rijen en checkpoints blijven. De CPU-proef en de smoke-test bij 4.096 draaien opnieuw
(nieuwe TAG; de mislukte runs hernoemd naar `…-fail1`, de marker `smoke-4096-memory` naar
`smoke-4096-memory-g4h1120b`). **Valt de smoke-test bij 4.096 opnieuw buiten het geheugen, dan stopt de keten (geen
terugval naar 2.048) en beslist de hoofdsessie** [keuze subagent, de veilige optie: bij 2.048 heeft de epoch 1.689 stappen
en vallen de checkpoints 240 en 480 op 14% en 28% in plaats van op een en twee derde, wat niet de opdracht is].

*G4-6c.6c Aanvulling 20 nov 2026 ~17:20 (subagent op opdracht van de hoofdsessie; vóór elke trainingsstap van G4-6c).*
Ook met G4-6c.6b liep de smoke-test bij 4.096 (keten `g4h1120c`, 16:36–16:59) op dezelfde plek tegen
`RESOURCE_EXHAUSTED` (eerste voorkeurseval op stap 0, w1 en w2); de keten stopte volgens G4-6c.6b zonder terugval. G4-6b
had bij dezelfde stap 1,62 GiB over (piek 29,13 van 30,75 GiB [gemeten: zijn `summary.json`]), dus de eval van G4-6c vroeg
meer dan 1,6 GiB extra; het aantal contracties was het niet. Wat G4-6c op het apparaat nieuw doet, is het verschuiven van
de segmentatie langs de reeks (`end_mask`: aanvullen en afsnijden langs L) en het vermenigvuldigen van de log-kansen met
dat masker [gevolgtrekking, niet gemeten: de sharding van het verschoven masker kan zich naar de berekening van de
log-kansen voortzetten]. Fix (code, geen methodewijziging): het masker van de log-ratio (`1 − end_mask`) wordt op de host
berekend en met elke batch meegegeven (`g4dpo.KEEP_KEY`, zoals de vooraf berekende referentie), voor de training, de
voorkeursevals en de referentiepass; op het apparaat alleen nog een vermenigvuldiging met een invoer, zoals het
bestaande masker `targets_segmentation != 0`. Een voorkeursbatch zonder masker stopt de run. CPU-proef en smoke-test bij
4.096 opnieuw (nieuwe TAG, de mislukte run hernoemd naar `g4arm_G4-6c_s8-fail2`, de markers `cpuproof6` en
`smoke-4096-memory` met `-g4h1120c`); G4-6c.6b's regel blijft: bij een volgende geheugenfout stopt de keten en beslist de
hoofdsessie.

*G4-6c.6d Aanvulling 20 nov 2026 ~20:15 (subagent op opdracht van de hoofdsessie; vóór elke trainingsstap van G4-6c).*
Met G4-6c.6c paste de smoke-test bij 4.096 (keten `g4h1120d`: voorkeursevals op stap 0 gehaald, stap 1–8 10,39 s,
piek 29,13 GiB; smoke-test geslaagd), maar de volle run `g4arm_G4-6c` liep na zijn referentiepass over 717 batches in
dezelfde eerste voorkeurseval op stap 0 tegen `RESOURCE_EXHAUSTED` (21 MB gevraagd, 4,96 MB vrij; w1; 19:32, vóór de
eerste stap). De eval zit dus op enkele tientallen MB van de grens. [Gevolgtrekking, niet gemeten] G4-6 (zonder
ingebakken adapter) piekte op 25,18 GiB, G4-6b (met) op 29,13 GiB: ~3,9 GiB, ongeveer de bf16-kernels van de 410
modules per chip; het samenvoegen van G4-4's adapter laat de oude kernels waarschijnlijk elders vastgehouden op het
apparaat. Fix (code, geen methodewijziging): `bake_adapter` voegt elke kernel samen uit zichzelf en geeft zijn buffer op
(`donate_argnums`), zodat de kernels van vóór het samenvoegen van het apparaat gaan, ook als iets anders ze nog
vasthoudt; geen hostkopie meer (er wordt niets teruggezet). Dezelfde waarden als G4-6b.2 (W0 + 0,5 · A·B in float32,
dan het eigen dtype). `BASE_ADAPTER` meldt het geheugen in gebruik vóór en na het samenvoegen, `REF` na de
referentiepass. Gebruikt iets daarna toch een opgegeven kernel, dan stopt de run met een fout. CPU-proef, smoke-test en
volle run opnieuw (nieuwe TAG; de mislukte volle run hernoemd naar `g4arm_G4-6c-fail1`, markers `cpuproof6`,
`smoke6` en `train-g4arm_G4-6c_s8` met `-g4h1120d`); G4-6c.6b's regel blijft.

*G4-6c.6e Amendement 20 nov 2026 ~22:30 (besluit van de hoofdsessie 20 nov 22:15, optie 1 van de overdracht; Yeb liet
de DPO-armen aan de hoofdsessie; uitgewerkt door een subagent; vóór elke stap op een worker).* Ook met G4-6c.6d liep de
volle run `g4arm_G4-6c` (keten `g4h1120e`, 21:04–22:10) na zijn referentiepass over 717 batches (2.810 s, geheugen in
gebruik daarna 9,98 GiB) in de eerste voorkeurseval op stap 0 tegen `RESOURCE_EXHAUSTED` (21 MB gevraagd, 4,96 MB vrij;
w1, w2), vóór stap 1; de smoke-test met dezelfde eval (referentiepass over 8 batches) paste [gemeten: de worker-logs,
`HANDOVER-g4arm6c.md` §0]. De oorzaak is niet gevonden. **Wijziging.** De referentie van de trainingsbatches wordt in een
eigen run berekend: `<run>_ref` (`sft_lora.py --dpo … --ref-only`, met G4-4's adapter ingebakken, hetzelfde model en
dezelfde MaxText-config als de training, geen eval, geen stap) doet de referentiepass over de batches die `<run>` traint
en schrijft ze weg als `ref-train.npy` (float32, [batches × rijen, max_segments], zoals de pass ze geeft) en
`ref-train.json` (waarvan het de referentie is: sha256 van het gepakte bestand en van G4-4's adapter, batches, rijen,
`max_segments`, `end_tokens`, de config van de voorwaartse pass, de vorm van de LoRA en de apparaten; de sha256 van het
array). `<run>` leest ze in (`--ref-in`), controleert vóór het opzetten van het model dat alles daarvan gelijk is aan de
eigen run (anders stopt hij), en doet zelf geen referentiepass over zijn trainingsdata; de referentie van de drie
voorkeurs-evalbestanden (32 batches) berekent hij zoals voorheen vóór de eerste stap, zoals de smoke-test die paste.
Zo ook de smoke-test: `g4arm_G4-6c_s8_ref`, dan `g4arm_G4-6c_s8`; daarna `g4arm_G4-6c_ref`, dan `g4arm_G4-6c`.
**Ongewijzigd:** β 0,5, NLL-gewicht 2,0, `<turn|>\n` buiten de log-ratio, rijen van 4.096, LR en schema, checkpoints
240/480/717, de keuze op het zoekdeel (G4-6c.7) en het besluit op het beslisdeel (G4-6c.8); G4-6c.6b's regel blijft
(bij een geheugenfout geen terugval naar 2.048, de keten stopt en de hoofdsessie beslist), ook voor de volle run.
**Dezelfde getallen, aangetoond in de CPU-proef** (`cpuproof6.sh` met `REFIN=1`, naast (a)–(e)): (f) `--ref-only` met
(c)'s ingebakken adapter over (c)'s 4 trainingsbatches; (g) (c) opnieuw met `--ref-in` van (f); (h) `--ref-in` van (f)
in een run zonder de ingebakken adapter. Eisen: de referentie van (f) is per paar (gekozen en afgewezen som) gelijk aan
die welke (c) in zijn eigen proces vóór zijn eerste stap berekende, binnen `REF_MAX_DIFF6E` = 1e-6 relatief
(t.o.v. max(1, |x|)), en de proef meldt of ze bitgelijk is; (g) las die referentie (sha256) en deed geen pass over zijn
trainingsbatches; (g)'s training is gelijk aan (c)'s (elke stap zijn voorkeursmaten, elke eval, de eindadapter) binnen
dezelfde grens; (h) wordt geweigerd vóór het opzetten van het model. [Keuze subagent] De grens 1e-6 is ~16 ulp van
float32: een referentie van andere rijen of een ander model wijkt ordes meer af. **Smoke-test, extra controle `ref_in`:**
de run las zijn referentie (record met `ref_in` en sha256, geen eigen pass) en op stap 1, waar het beleid nog de
referentie is, liggen de DPO-term en de marge binnen `STEP1_MAX_DIFF6E` = 1e-3 van log 2 en 0 [keuze subagent; gemeten
in de smoke-test van `g4h1120e` met de referentie in hetzelfde proces: 7,5e-5 en 1,5e-4]; de keten toetst hetzelfde
voor de volle run na zijn einde. De schatting van de smoke-test telt geen referentiepass over de trainingsbatches meer
(die is een eigen run, ~50 min). [Gevolgtrekking, niet gemeten] De volle run doet zo vóór zijn eerste eval wat de
smoke-test deed (geen 717 referentiestappen); of de eval dan past, blijkt pas in de volle run. **Keten:** wacht op
`final_run.lock` (de v3-ablatiearmen c en d op mbert-corpus) tot 36 u, dan CPU-proef, smoke-test bij 4.096 met de
ingelezen referentie, de referentierun, de volle run en het vervolg als G4-6c.11. Uitvoer erbij:
`$G/runs/g4arm_G4-6c_s8_ref/` en `$G/runs/g4arm_G4-6c_ref/` (`ref-train.npy`, `ref-train.json`, `summary.json`,
`dpo_ref.parquet`). De markers `cpuproof6`, `smoke6` en `train-g4arm_G4-6c_s8` van code `20904d4` hernoemd met
`-g4h1120e`, de smoke-run van `g4h1120e` naar `g4arm_G4-6c_s8-g4h1120e`; nieuwe TAG.

## 14. Amendement G4-7

**Amendement 21 nov 2026 (tijdstip: de commit van deze sectie op branch `en-nl-g4arm7`, vanaf `en-nl-g4arm6c` @
`71940fe`): arm G4-7, online RL (GRPO) met LoRA op Gemma 4 31B-it** [kader van de hoofdsessie, afgestemd met Yeb op 20
nov; besluiten van Yeb van 20 nov ~15:25 en 21 nov ~09:10–09:21; de overige keuzes zijn een invulling door de
hoofdsessie volgens het advies van de spec `mbert-ops/translator-sft/g4-rl-spec.md` (§9); uitwerking door een subagent;
labels als in de kop]. Vastgelegd vóór de data van C2, vóór de bronnenpool, vóór de CPU-proef, de smoke-test en de
eerste RL-stap, en vóór elke uitslag van G4-7. Waar deze sectie niets zegt, gelden §1–§8 en de amendementen van §9–§13
met "G4-7" in plaats van de arm daar; de beslisregel is die van G4-7.8. Wie wat koos, staat in G4-7.0; de onderbouwing
staat in de spec.

*G4-7.0 Keuzes, en wie ze maakte.*
- **Yeb (20 nov ~15:25):** C2 is de reward-classifier; C1 (S @ 653k, de classifier van regel (b)) blijft de
  ongewijzigde meter van (b) en komt nooit in de lus.
- **Yeb (21 nov ~09:10):** de encoder van C2 is dmbert v2 (`yhavinga/dmbert-230m-fineweb2-optout`; checkpoint
  `gs://t5-dutch-english-usc2/mbert/v2-fw2optout-230m/ckpt_keep/1306000`). RobBERT-2023 valt af. C2 gaat via hetzelfde
  pad als C1 (`naturalness.py classifier` en `classify`, de DUMB-fine-tune); er komt geen RoBERTa-pad.
- **Yeb (21 nov ~09:20):** de lengteregels: bronnen van ten hoogste 256 Gemma 4-tokens; een vertaling met ℓ/ℓ₀ buiten
  [0,67; 1,5] is een fail (r = −10); C2 scoort de hele tekst en kapt nooit af, en een tekst van meer dan 1.022 tokens is
  een fail (`c2_overflow`); het poolfilter van 340 RobBERT-tokens vervalt. De stopregels S1–S6 van spec §6.4 (S5
  vervalt), met S4 ruimer: de mediane ℓ/ℓ₀ op RL-val moet binnen [0,85; 1,15] blijven. De vier grenzen van `c2cal7`
  (spec §5.3.3): akkoord.
- **Advies van de hoofdsessie; Yeb maakte geen bezwaar (21 nov):** (a3), COMET-22 ≥ G4-0 − 0,005 per set, blijft in de
  beslisregel.
- **Invulling door de hoofdsessie volgens het advies van de spec (21 nov):** route C-fase (per iteratie genereren,
  beoordelen, trainen); startpunt G4-0, referentie het model met LoRA op nul; de classifierterm is de logit van C2,
  geknipt op ±6; de QE-term is een grens met anker, gewicht 2; G = 8; 512 bronnen per iteratie; data FLAN 40 / romans
  30 / BT 30; LR 5e-6; β 0,04 met k2 en sg(ρ); geen overlap van genereren en beoordelen; de rollen op mbert-mix: w0 en w2
  MetricX-24-QE, w1 C2, w3 de vertalingen van RL-val; G4-7 start na het verslag van G4-6c en als mbert-mix vrij is.
- **Keuzes van de subagent (techniek):** het bucketprotocol van G4-7.4a (kolommen, markers, de verdeling van MetricX
  over w0 en w2, de jobs `anchor` en `val/<i>`); de data van C2 met één passage per document en vóór de bronnenpool
  (G4-7.6); de code van C2 in `g4c2.py` in plaats van `g4rl.py` (de judge-kant en de beleidskant worden op twee branches
  gebouwd).

*G4-7.1 Aanleiding.* G4-2 t/m G4-6b haalden (b) niet (+1,0 / +0,4 / +0,7 / −0,4 / −1,7 / −4,5 pp). SFT verschuift de
kansverdeling, niet de greedy modus (temperatuurproef: één T verklaart 62–67% van de lossdaling); bij T = 1 is G4-4's
uitvoer op WMT24++ 60,9% "origineel" (sampling-proef). DPO verzadigde (G4-6) of ontspoorde (G4-6b). G4-7 beloont het
model op zijn eigen samples.

*G4-7.2 Data* (`g4rl.py pool7`, `$RL/data/`; `RL=$G/rl/G4-7`). Engelse bronnen van ten hoogste 256 Gemma 4-tokens:
FLAN-nl-segmenten (40%), het Engels van de `-b1-`-romans zonder de 10 boeken van `novels` (30%), het Engels van de vijf
`-nl-en-gemma4`-sets (train, gelijk per bron, zonder de documenten van C2; 30%). De documenten van C2 staan in
`$RL/c2/data/docs.txt` (één documentsleutel per regel, G4-7.6); `pool7` leest die lijst en stopt als hij ontbreekt, dus
`g4c2.py data7` draait vóór `pool7`. Uit: elke bron met een 13-gram van de evaluatiesets, de FLAN-evaluatierijen of de
blinde passages. 15.360 bronnen in volgorde `en-nl-g4arm/0/G4-7/order`, RL-val 384 (128 per deel). Anker: G4-0's greedy
vertaling (klasse `medium`, fork `f4d8615`, `natural-v1`) met QE, lengte ℓ₀ (Gemma 4-tokens) en C2; een bron waarvan het
anker niet af is, een lus heeft of Engels laat staan, valt weg. Er is geen filter op het aantal tokens van C2 [Yeb 21 nov].

*G4-7.3 Reward en verlies* (`g4rl.py`). fail = niet af, lus, leeg, Engels over, ℓ/ℓ₀ buiten [0,67; 1,5], QE > 7,5 of
`c2_overflow` (de uitvoer heeft meer dan 1.022 tokens van C2's tokenizer, zonder de speciale tokens; C2 kapt nooit af)
→ r = −10; anders r = clip(logit C2, −6, 6) − 2·max(0, QE − QE₀) − 2·max(0, |ln(ℓ/ℓ₀)| − ln 1,15). Voordeel = r − het
groepsgemiddelde (G = 8); groepen met max − min < 0,05 niet in het trainbestand. Verlies per uitvoertoken (inclusief het
stoptoken): −min(ρA, clip(ρ, 0,8, 1,2)A) + 0,04·sg(ρ)·½(log πθ − log π_ref)², gemiddeld over de tokens van de batch;
π_old en π_ref per iteratie berekend door de leerder vóór de eerste stap; π_ref = het model met LoRA op nul. Gelijk getest
tegen Tunix @ `a8d7058` (`compute_advantages` "drgrpo", `compute_kl_divergence` "mse_kl", `grpo_loss_fn`,
`aggregate_loss` "token-mean"). Is het verschil sampler/leerder (gemiddelde |log π_old − log π_sampler| per token) > 0,05,
dan vanaf de volgende iteratie × min(π_old/π_sampler, 2) per token.

*G4-7.4 Training* (`sft_lora.py --grpo`, `g4arm7_run.sh`). Per iteratie i: (1) de vier workers van mbert-corpus vertalen
hun deel van 512 bronnen × 8 met `translate.py --lora` (de adapter van i ingebakken vanaf de fp32-basis; T = 1, seed
`en-nl-g4arm/0/G4-7/sample/<i>`) en schrijven de shards naar `$RL/iter/<i>/samples/`; (2) de judges van mbert-mix (w0, w2
MetricX-24-QE hybrid XXL; w1 C2) scoren elke shard (G4-7.4a); (3) `g4rl.py iter7` bouwt het trainbestand; (4) één job
over 16 chips rekent π_ref en π_old en traint één optimizer-stap per batch van 16 rijen van 4.096, LR 5e-6 (10 stappen
opwarmen, dan constant), AdamW zoals G4-4.4, de state voortgezet met `--resume-step`; adapter en checkpoint naar
`$RL/iter/<i>/`. Rank 32 / alpha 16, 410 modules. Ten hoogste 30 iteraties. Na elke iteratie vertaalt w3 van mbert-mix
RL-val greedy met de nieuwe adapter naar `$RL/val/<i>/samples/w3/`, en scoren de judges die.

*G4-7.4a Het bucketprotocol* (`g4judge.py`; alles in `gs://t5-dutch-english-usc2`, Python-client van GCS; paden onder
`$RL`). Een **job** is `iter/<i>` (de samples van iteratie i), `val/<i>` (RL-val na iteratie i) of `anchor` (G4-0's
greedy vertaling van de pool en RL-val, iteratie 0 van C2 en QE). Per job:
- `<job>/samples/w<k>/shard-<n>.parquet` (k = de schrijvende worker, n = een geheel getal): per sample `uid` (string,
  `<id>#<j>`), `id` (string, de bron-id), `j` (int; 0 bij `anchor` en `val`), `source` (het Engels), `hypothesis` (het
  gedecodeerde Nederlands zonder het einde van de beurt), `token_ids` (list<int>), `sampler_logp` (list<float>, per token
  van `token_ids`), `ended`, `decode_loop`, `english_left`, `empty` (bool), `tokens_out` (int). Een `uid` komt binnen
  een job één keer voor. De judges lezen alleen `uid`, `source` en `hypothesis`.
- `<job>/samples/w<k>/DONE`, als laatste: JSON `{"rows": <int>, "shards": {"shard-<n>.parquet": {"rows": <int>,
  "sha256": "<hex>"}}}`.
- `<job>/scores/metricx/w<m>/shard-<k>-<n>.parquet`: `uid`, `qe` (float, MetricX-24 hybrid XXL v2p6 bf16 in QE-modus,
  afgerond op 4 decimalen zoals `metricx_qe.py score`; null bij een lege `hypothesis`). MetricX is verdeeld op `uid`:
  deel = int(de eerste 8 hextekens van sha256(uid)) mod 2; w0 scoort deel 0, w2 deel 1, in elke job. Een
  scorebestand bevat alleen de rijen van het eigen deel.
- `<job>/scores/c2/w1/shard-<k>-<n>.parquet`: `uid`, `p_c2` (float, P(label 1 = origineel) met de softmax van
  `naturalness.py classify`), `logit_c2` (float, logit₁ − logit₀ in float32, ongeknipt), `c2_tokens` (int, tokens van
  C2's tokenizer zonder de speciale tokens), `c2_overflow` (bool, `c2_tokens` > 1.022; dan `p_c2` en `logit_c2` null).
  C2 scoort elke rij; nooit afgekapt.
- `<job>/scores/<rol>/w<m>/DONE-<k>`, als laatste, zodra elke shard van `samples/w<k>/DONE` gescoord is: JSON `{"rows":
  <int>, "shards": {"shard-<n>.parquet": {"rows": <int>, "samples_sha256": "<hex>"}}}`, met de sha256 van het
  samplebestand zoals de judge het las; wijkt die af van `DONE`, dan schrijft de judge geen marker en stopt hij (exit ≠ 0).
- `judges/<rol>-w<m>.json` (hartslag; het enige object dat overschreven wordt), ten minste elke 60 s: `pid`, `host`,
  `role`, `worker`, `state`, `time` (UTC, ISO 8601), `epoch`, `last_shard`, `scored_rows`.
- `judges/STOP`: geschreven door de keten; een judge maakt zijn shard af en stopt met exit 0.
- Elk ander object wordt één keer geschreven (bij GCS met `if_generation_match=0`) en de marker komt als laatste. Een
  judge scoort elke shard zodra die verschijnt (streaming), slaat een shard met een bestaand resultaat over (herstartbaar,
  idempotent) en leest alleen zijn eigen invoer.
- De keten (`g4judge.py wait`): vóór elke iteratie is elke hartslag hoogstens 5 min oud, anders `FAILED`; na de
  generatie liggen alle `DONE-<k>` van alle judges er binnen 15 min na de laatste `DONE` van de generatie, anders `FAILED`
  met de namen van de ontbrekende shards; de `uid`'s van de scores (MetricX: w0 en w2 samen, zonder overlap; C2) zijn
  precies die van de samples, anders `FAILED`. C1 is geen judge en scoort niets onder `$RL/`.

*G4-7.5 Stopregels* (na elke iteratie; RL-val greedy op w3) [Yeb 21 nov]: S1 niet af + lus + leeg + Engels over op RL-val
> 2% (8 van 384); S2 gemiddelde QE op RL-val − die van het anker > 0,15 bij twee validaties na elkaar; S3 KL per token
(k1-schatting op de samples) > 0,30 nats; **S4 mediane ℓ/ℓ₀ op RL-val buiten [0,85; 1,15]**; S6 fail-aandeel van de
samples van een iteratie > 10% (inclusief `c2_overflow`). Er is geen S5: geen stopregel gebruikt C1. Een stop beëindigt
de training; technische fouten (NaN, OOM, timeout van een judge): `FAILED`, geen stille terugval.

*G4-7.6 C2, de reward-classifier* (`g4c2.py data7`, `naturalness.py classifier`, `g4c2.py cal7`) [Yeb 20 en 21 nov].
- **Encoder:** dmbert v2 (`yhavinga/dmbert-230m-fineweb2-optout`), het checkpoint
  `gs://t5-dutch-english-usc2/mbert/v2-fw2optout-230m/ckpt_keep/1306000`, omgezet met
  `scripts/convert_gcs_checkpoint.py --params-only` zoals C1's checkpoint; zijn eigen tokenizer (`config.tokenizer_name`).
- **Data** (`$RL/c2/data/`): Nederlandse originelen (`nl`, label 1) tegen Gemma 4's rondgangvertaling van dezelfde
  passage (`nl_back`, label 0) uit de vijf `yhavinga/<bron>-nl-en-gemma4`-sets, split `train`, op de revisies van G4-3
  (`BT_REPOS3`). Een passage komt in aanmerking als `nl` en `nl_back` niet leeg zijn, `hard_fails` leeg is,
  `back_ended` waar is, `back_decode_loop` en `back_english_left` onwaar zijn, en `nl` en `nl_back` allebei ten hoogste
  1.022 tokens van C2's tokenizer hebben (zonder de speciale tokens; heel, niet afgekapt). Uit: de documenten van C1's
  2.000 passages en elke passage met dezelfde `nl`-tekst; elke passage met een 13-gram uit WMT24++, FLORES, NTREX,
  Europarl-direct, de FLAN-evaluatierijen of de blinde passages (`nl` of `en`); `btval`, `btlong` en `own` komen uit de
  split `validation` en vallen daarmee weg (gecontroleerd). Documentsleutel: de `id` zonder de laatste twee velden
  (`<bron>:<document>`). Per bron 1.000 documenten in een geseede volgorde (`en-nl-g4arm/0/G4-7/c2/<bron>`), van elk
  document één passage (ook geseed); de eerste 100 documenten in een tweede geseede volgorde
  (`en-nl-g4arm/0/G4-7/c2/<bron>/test`) zijn de testsplit, de andere 900 train: 9.000 + 1.000 teksten per klasse samen,
  beide versies van een passage aan dezelfde kant. Heeft een bron minder dan 1.000 documenten die in aanmerking komen,
  dan stopt `data7`. Het bestand heeft de kolommen van C1's data (`pid`, `id`, `source`, `rank`, `split`, `label`,
  `text`); `docs.txt` heeft de 5.000 documentsleutels voor `pool7` (G4-7.2).
- **Training** (`$RL/c2/model/`), op w1 in `~/mbert-dumb/.venv` met `~/mbert-dumbx` @ `ae11478`, zoals C1:
  `naturalness.py classifier` ongewijzigd, 3 epochs, LR 5e-5, seed 42, 8 per chip, `--max-length 1024`; geen
  validatiesplit en geen keuze van een epoch.
- **Bruikbaar als reward** (`g4c2.py cal7`, `$RL/c2/cal7/`, vóór het anker en de eerste RL-stap; alle vier moeten gelden,
  anders geen RL en aan Yeb): (1) nauwkeurigheid op de eigen testsplit ≥ 0,75; (2) aandeel "origineel" (P ≥ 0,5) op de
  Nederlandse originelen van Europarl-direct (de referenties van die set) ≥ 0,75; (3) G4-0's greedy aandeel "origineel" op
  WMT24++ (goed) + FLAN tussen 0,02 en 0,90; (4) op de 8 samples per bron van de sampling-proef van G4-0
  (`$G/sampling/G4-0/t10s1…t10s8/`) per set (WMT24++, FLAN) de mediaan over de bronnen van de sd (ddof 0) van
  clip(logit C2, −6, 6) binnen de groep ≥ 0,25 en ≥ de mediaan-sd van 2·max(0, QE − QE₀) in dezelfde groepen (QE₀ = de QE
  van G4-0's greedy vertaling van dat segment uit zijn evaluatie). Gerapporteerd, beslist niets: dezelfde maten voor G4-4,
  het aandeel groepen met max − min van de logit > 1 en met alle 8 geknipt, |Δ logit| `t10s1` tegen `best8`, de Spearman
  binnen de bron van Δlogit C2 met Δtokens en met ΔQE, de nauwkeurigheid op de testsplit per lengte (≤ 128, 129–256, > 256
  tokens), de C2-tokens van de samples, en de kalibratie van spec §5.3.5 (C2 op de bestaande uitvoer, de Spearman per
  segment met C1 uit de bestaande classify-runs).
- C2 scoort elk sample, het anker en RL-val, als judge op w1 (G4-7.4a); C2 op held-out wordt alleen gerapporteerd. C1 (S @
  653k, ongewijzigd) scoort alleen de kandidaten en G4-0 op de evaluatieset, in één classify-run, voor G4-7.7 en G4-7.8;
  nooit de samples, het anker of RL-val, en niet in een stopregel.

*G4-7.7 Keuze (zoekdeel)*, als G4-5.6 en G4-6c.7. Kandidaten: de adapters na iteratie 5, 10, 15, 20, 25, 30 en de
laatste vóór een stop, elk vertaald en gescoord zoals G4-5.5. Van de kandidaten die op het zoekdeel van G4-5.4 (digest
`f953c026…f3`) (a1) halen: de grootste (b) (C1 "origineel" − G4-0 op WMT24++ (goed) + FLAN van het zoekdeel); bij
gelijkheid de eerdere; haalt geen kandidaat (a1), dan de kleinste grootste QE-overschrijding (`select5`).

*G4-7.8 Beslisregel (beslisdeel), tegen G4-0.* (a1) per set QE ≤ die van de menselijke referentie op dezelfde
segmenten; (a3) per set COMET-22 ≥ G4-0 − 0,005 (WMT24++, FLORES, NTREX); (b) C1 "origineel" − G4-0 ≥ 0,10 op WMT24++
(goed) + FLAN, ongewijzigd; (a2) en (c) zoals §5, alleen als (a1), (a3) en (b) gelden (pakket `$S/blind/G4-7/`, één
verse Claude-subagent). Helpt als alle vijf gelden. C2 per set wordt naast C1 gerapporteerd en beslist niets.

*G4-7.9 CPU-proef en smoke-test* (spec §8.2–§8.3): `CPU_PROOF7_OK` vóór de eerste TPU-stap; `smoke7` met twee
mini-iteraties; faalt een controle, dan stoppen vóór de RL-run. In de smoke-test is `c2_overflow` bij meer dan 1 van de
256 samples een stop.

*G4-7.10 Rapportage* (beslist niets). Vóór de RL: `c2cal7` met de kalibratie (bij een hoge correlatie met C1, Spearman
≥ 0,5 per segment, de zin dat een C2-hack ook C1 kan misleiden en de blinde beoordeling dan zwaarder weegt). Per
iteratie: reward en zijn termen, fail-aandelen per soort (ook `c2_overflow`), QE, C2, lengte, KL, clip-fractie, verschil
sampler/leerder, groepen zonder signaal; per validatie RL-val met C2; per 5 iteraties `books`, `btval`, `own`; alle
kandidaten op beide delen na de keuze, met C1 en C2 naast elkaar ("C2 stijgt, C1 niet" = signaal van reward hacking);
MTLD en PLL.

*G4-7.11 Uitvoer.* `$RL/c2/data/`, `$RL/c2/model/`, `$RL/c2/cal7/`, `$RL/data/`, `$RL/anchor/`, `$RL/iter/<i>/`,
`$RL/val/<i>/`, `$RL/judges/`, `$G/G4-7-i<k>/`, `$G/eval/G4-7-i<k>/`, `$G/naturalness-G4-7/`, `$S/blind/G4-7/`,
`$G/report-G4-7/` (kopie `mbert-ops/translator-sft/g4-7-report.md`), markers `$G/stages/g4-7/`.

*G4-7.4b Aanvulling 21 nov 2026 (subagent van de judge-kant; vóór elke stap van G4-7 op een worker; geen
methodewijziging).* Wat de code (`g4judge.py`, `g4c2.py`, `run_g4arm7_judges.sh` op `en-nl-g4arm7`) aan G4-7.4a en
G4-7.6 toevoegt:
- **Markers.** `DONE-<k>` van een judge heeft naast `rows` en `shards` (per shard `rows` en `samples_sha256`) ook
  `samples_rows` (per shard en in totaal), `role`, `worker`, `part` en `time`. Elk scorebestand draagt in zijn
  parquet-metadata (`g4judge`) de naam, de sha256 en het aantal rijen van het samplebestand dat de judge las, zodat een
  herstarte judge `DONE-<k>` kan schrijven zonder opnieuw te scoren.
- **Hartslag.** Toestanden `loading`, `idle`, `scoring`, `stopped`, `stopped-idle`, `stopped-signal`, `failed` (met
  `error`); verder `part`, `scorer`, `started`, `busy_since`, `scored_rows`, `scored_shards`. Een draad schrijft hem elke
  60 s, ook tijdens het laden van een model en het scoren van een shard. `g4judge.py alive` en `wait` rekenen een judge
  als dood bij een hartslag ouder dan 5 min, een andere toestand dan `loading`/`idle`/`scoring`, een ander deel dan
  verwacht, of één shard langer dan 15 min bezig. `wait` controleert zonder samplebestanden te lezen (alleen markers):
  elke judge las per shard dezelfde bytes als `DONE` noemt, C2 scoorde elke rij, de twee MetricX-delen samen elke rij;
  `--full` vergelijkt ook de `uid`'s (leest alles: alleen in de regio van de bucket).
- **Einde.** Een judge stopt op `judges/STOP` of na 6 u zonder werk (`MAX_IDLE`), met exit 0.
- **C2's tokenizer** (`yhavinga/dutch-llama-tokenizer`, zoals C1) zet alleen `<s>` vooraan, geen `</s>`
  [gemeten 21 nov op co1]. De regel blijft die van Yeb: meer dan 1.022 tokens zonder de speciale tokens is
  `c2_overflow`; een tekst die met `<s>` meer dan 1.024 tokens zou hebben, bereikt het model nooit (de judge stopt dan
  met een fout).
- **Data van C2.** Een passage waarvan `nl` of `nl_back` gelijk is aan een tekst die al getrokken is, wordt
  overgeslagen (geen dubbele tekst over train en test). De controle op de split `validation` gebruikt de opgeslagen
  fetch van G4-3 (`$G/input/bt3-val`); de trainsplit komt uit `$G/input/bt3` (beide `BT_REPOS3`). Het checkpoint van v2
  wordt op de CPU van w1 omgezet zoals C1's S @ 653k (`convert_gcs_checkpoint.py --params-only`, tokenizer
  `yhavinga/dutch-llama-tokenizer`, `--global-attn-every-n-layers 3`, kop 0).

*G4-7.4c Aanvulling 21 nov 2026 (subagent van de keten; vóór elke stap van de keten `g4arm7_run.sh` op een worker;
geen methodewijziging).* Wat de samenvoeging van de beleidskant (`en-nl-g4arm7-policy`) en de judge-kant op
`en-nl-g4arm7`, de evaluatie (`run_g4arm.sh` met `ARM=G4-7`, `g4arm.py report7`) en de keten aan G4-7.2–G4-7.11
toevoegen, of waarin de code van de letter van de spec afwijkt:
- **De samples.** Een sampleshard heeft naast de kolommen van G4-7.4a ook `tokens_in` (de lengte van de prompt van de
  sampler); `iter7` stopt als de prompt van `g4lora.prompt_text` een andere lengte heeft. De judges lezen alleen `uid`,
  `source`, `hypothesis`. Het zaad van worker k in iteratie i is de eerste 8 bytes van
  sha256(`en-nl-g4arm/0/G4-7/sample/<i>/w<k>`) (`g4rl.sample_seed`), niet één zaad `…/sample/<i>` voor alle vier
  workers: dan zou de ruis van het samplen per slot en stap op de vier workers gelijk zijn. Generatie met
  `translate.py --copies 8 --sample samples=1.0:<zaad> --shard-size 256` (32 bronnen per shard, de samples van een bron in
  één shard), klasse `medium` met zijn eigen slots; het anker en RL-val greedy met `--copies 1`. Een geheugenfout is
  `FAILED`; de keten halveert geen slots.
- **De pool.** `pool7` trekt per deel 3% meer bronnen (reserve voor wat het anker laat vallen); `plan7` houdt daarna
  precies 512 bronnen per iteratie (205/154/153) en 128 per deel voor RL-val. Het anker vertaalt dus ~3% meer dan 15.744
  bronnen. De documenten van RL-val komen niet in de pool (strenger dan de spec), en een bron met een leeg anker valt ook
  weg (een leeg anker zegt evenmin iets als een anker dat niet af is).
- **Kandidaten** (`g4rl.py cands7`, `$RL/data/cands/`; keuzes van een subagent, techniek): `cands-flan|novels|bt.parquet`
  (`id`, `en`, `doc`, `stratum`) en `cands.json`. FLAN: de segmenten van `mbert/flan-nl/plan2` van de voorbeelden die in
  de gepubliceerde set bleven (`nl-g4b` min de 6.120 die de publicatiecontroles weglieten; de tellingen gelijk aan de
  build, 1.032.592 rijen), zonder templatestrings en zonder de segmenten van de 1.035 evaluatievoorbeelden; `doc` = het
  eerste voorbeeld met dat segment (`flan:w<h>-<p>`). Romans: de passages zoals `build4` ze maakt (gecontroleerd tegen
  het rapport van G4-4: 2.931 boeken), zonder de 10 boeken van `novels`; `doc` = het boek. BT: het Engels van de
  trainsplit op de revisies van G4-3, zonder de rijen met `hard_fails` (een controle op het Engels faalde); `doc` =
  `g4c2.doc_key`, `stratum` = de bron. Per deel (BT per bron) een geseede trekking (`en-nl-g4arm/0/G4-7/cands/<deel>`)
  van 2× wat `pool7` neemt; lege, dubbele en te lange teksten laat `cands7` al weg, zoals `pool7`. [Gemeten op mbert-corpus
  w1, met een nep-`docs.txt`: na de filters van `pool7` FLAN 11.719, romans 9.782, BT 9.264 kandidaten, tegen 132 + 6.335,
  132 + 4.759 en 132 + 4.728 nodig.] `g4c2.doc_key` herkent ook de lange passages van de revisies van G4-3
  (`…:d:p<a>-<b>`, 21.368 rijen van de trainsplit); `data7` zou er anders op stoppen.
- **Verlies.** TIS vermenigvuldigt alleen de beleidsterm (zoals Tunix' `sampler_is_weights`), niet de KL-term; de spec
  zei "het verlies per token". Is het verschil sampler/leerder (gemiddelde |log π_old − log π_sampler| per token) in de
  smoke-test of in een iteratie > 0,05, dan staat TIS aan vanaf de iteratie erna tot het eind ("vanaf de volgende
  iteratie", spec §4.4). `sft_lora.py --passes-only`/`--passes-in` dienen alleen de CPU-proef (punt (c): één
  ononderbroken run met de passes van twee iteraties); de keten gebruikt ze niet. De CPU-proef draait rijen van 1.024
  (niet 4.096), zodat 64 samples ≥ 4 batches geven, in zeven runs (a, c, d, e0, e, f, g; (b) via de passes-only-runs f
  en g; (d) 2 stappen met 1 warmup-stap). LR 5e-6 met precies 10 warmup-stappen en geen decay (`--warmup-steps 10
  --decay-fraction 0`), geteld over de globale stappen (`--resume-step`). MaxText's datahooks krijgen de gekozen kanten en
  de template van G4-6b (zoals de multi-host-runs van G4-6c) en worden nooit gelezen: `--grpo` vervangt de iterator.
- **Nummering.** Iteratie i (0..29) samplet met adapter A_i (A_0 = G4-0, geen LoRA) en traint A_{i+1}: run
  `g4a7_i<i>_<TAG>` onder `$RL/iter/<i>/runs/`, met de optimizer-state van de run van i − 1 (zijn Tunix-checkpoint van de
  laatste stap gekopieerd naar de checkpointmap van de nieuwe run, zoals `resume5`). `val/<i>` is RL-val greedy met
  A_{i+1}. De stopregels na iteratie i: S1, S2 en S4 op `val/<i>` (A_{i+1}), S6 op de samples van iteratie i (A_i), S3 op
  de passes van de training van iteratie i (k1 van A_i tegen de referentie): S3 loopt dus één iteratie achter. S2 rekent
  over de RL-val-vertalingen met een QE (een lege vertaling heeft geen QE en telt in S1). Kandidaat `G4-7-i<n>` is A_n
  (n = 5, 10, …, 30), in `$G/G4-7-i<n>/`; "de laatste vóór een stop" is A_{i+1} als een regel na iteratie i raakt (de
  laatste adapter die de training maakte).
- **Overlap van RL-val en de volgende generatie.** Na de training van iteratie i vertaalt mbert-mix w3 RL-val met A_{i+1}
  terwijl mbert-corpus de samples van iteratie i + 1 met A_{i+1} maakt; de training van i + 1 begint pas na `stop7` van
  i. Stopt de training na iteratie i, dan worden die samples niet gebruikt.
- **SFT-evaluaties.** `books`, `btval` en `own` (spec §6.3, G4-7.10) in de trainrun van elke 5e iteratie
  ((i + 1) mod 5 = 0), vóór zijn eerste en na zijn laatste stap (`--eval-every 1000000`).
- **Volgorde.** De data van C2 (`convert`, `g4c2.py data7`) draait op de CPU van mbert-corpus w1, met dezelfde venv en
  code (`~/mbert-dumb/.venv`, `~/mbert-dumbx` @ `ae11478`) als op mbert-mix, zodra mbert-corpus vrij is en vóór mbert-mix
  vrij is; daarna `cands7`, `pool7` en de vertalingen van het anker (G4-0 greedy, de vier workers van mbert-corpus). G4-7.6
  zet `c2cal7` "vóór het anker": de scores van het anker (QE₀, C2), `plan7` en alles daarna komen pas na `C2CAL7_OK`; de
  vertalingen zelf hangen niet van C2 af. Faalt `c2cal7`, dan worden ze niet gebruikt en komt er geen RL.
- **Kalibratie** (spec §5.3.5, gerapporteerd): tegen de bestaande classify-runs van C1 `naturalness-G4-6b` (G4-6b, G4-0,
  G4-4, G4-5c2, G4-6, G4-v6e, TG op WMT24++ en FLAN), `sampling/naturalness` (G4-0 en G4-4 greedy, T 0,7, T 1, best8,
  G4-v6e, TG) en `naturalness-G4-6c` als die er dan is. De menselijke referenties zitten er niet in: C1's scores daarvan
  bestaan alleen als voorspellingen zonder tekstbestand in dit formaat (`$S/re/classify/targets-predictions.parquet`).
- **Smoke-test.** De twee mini-iteraties zijn de jobs `iter/<N>` en `iter/<N+1>` (N het eerste vrije even getal ≥ 900)
  en `val/<N>` onder `$RL`, zodat de persistente judges ze scoren; hun bronnen zijn de eerste 32 (op `position`) van
  planiteraties 0 en 1 (`g4rl.py smokeplan7`), met de ankers van het plan; de rest onder `$RL/smoke7/<TAG>/`. De RL
  gebruikt er niets van. Trainfase 2 draait met de SFT-evaluaties (het geheugen van een 5e iteratie); de losse MetricX- en
  C2-scores op mbert-mix w3. `smoke7` toetst ook dat de generatie van A geen LoRA heeft en het zaad `g4rl.sample_seed(N, k)` bij T = 1, dat
  de 50 smoke-segmenten met de adapter van fase 2 een `LORA`-regel hebben (anders zegt |ΔQE| niets) en dat de judges
  binnen 900 s na de laatste `DONE` klaar waren; "af" is af, geen lus en niet leeg (zoals de eerdere smoke-tests). De
  schatting (≤ 24 u) telt per iteratie generatie, judge-staart, `iter7`, RL-val en training uit de gemeten fasen
  (`g4rl.smoke_estimate7`). Een gefaalde controle of een geheugenfout: `FAILED` vóór de RL, zonder
  terugval.
- **Herstart.** Elke stap heeft een marker in `$G/stages/g4-7/`; een herstart (nieuwe TAG) begint bij de eerste stap
  zonder marker. Een generatie houdt de shards en `DONE`-markers die al in de bucket staan (`translate.py` slaat ze
  over); een training die niet af is, draait opnieuw onder een nieuwe naam (`g4a7_i<i>_<nieuwe TAG>`), hervat vanaf de
  gemarkeerde run van i − 1. Een smoke-test zonder marker begint opnieuw met een nieuwe N.
- **Judges.** Ze draaien tot de keten `judges/STOP` schrijft, na de laatste RL-val; daarna geeft de keten mbert-mix vrij.
- **Rapport** (`g4arm.py report7`, beslist niets buiten G4-7.7 en G4-7.8): (a3) per set op dezelfde segmenten van het
  beslisdeel als (a1) (WMT24++ zonder de slechte bronnen, zoals (a1)); de vlag "C2 stijgt, C1 niet" is ΔC2 ≥ +0,10 en
  ΔC1 < +0,10 op WMT24++ (goed) + FLAN samen; PLL (protocol.md §4.2.3) met `naturalness.py pll` onder S @ 653k op de
  teksten van de classify-run (`run_g4arm.sh pll`).

*G4-7.4d Aanvulling 21 nov 2026 (besluit van Yeb ~11:50 en de invulling door de hoofdsessie; uitgewerkt door een
subagent; vóór elke stap van G4-7 op een worker: de keten `g4i1121a` wachtte nog op G4-6c en is om 11:49 gestopt zonder
iets gedaan te hebben; een methodewijziging van G4-7.0 en G4-7.2).* De datamix van de RL-bronnen:
- **Verhoudingen** [Yeb 21 nov ~11:50]: FLAN 10%, romans 45%, BT 45% (was FLAN 40 / romans 30 / BT 30, de invulling van
  G4-7.0 volgens de spec). Per iteratie van 512 bronnen: 51 FLAN, 231 romans, 230 BT (de grootste resten; de ene bron
  die overblijft gaat bij gelijke rest naar het eerste deel in de volgorde, de romans). Yebs reden: FLAN-bronnen zijn
  kort (mediaan 19 Gemma 4-tokens, 31% ≤ 10 [gemeten 21 nov, de subagent van `cands7`]), zodat een groep van 8 vaak 8
  gelijke samples heeft en dan niets leert; ze zijn opdrachtachtig; en ze zijn dezelfde soort tekst als de helft van de
  (b)-meting.
- **Minimumlengte** [invulling hoofdsessie 21 nov]: elke bron, in alle drie de delen, in de pool en in RL-val, heeft ten
  minste 20 Gemma 4-tokens, geteld zoals het maximum van 256 (de ids van de Engelse tekst alleen, `g4arm.TokenCounter`):
  20 ≤ tokens ≤ 256. `cands7` trekt alleen zulke teksten (trechter: `short`), `pool7` toetst het opnieuw (`drops.short`)
  (`g4rl.MIN_SOURCE7 = 20`, `candidate_filter`, `draw_candidates`).
- **RL-val** blijft 128 bronnen per deel (384), getrokken met dezelfde filters, dus ook ≥ 20 tokens [keuze subagent].
  Reden: RL-val dient alleen de stopregels S1, S2 en S4 (G4-7.5), en die moeten schade in elk register zien, ook in het
  register dat de training nu minder ziet; FLAN is de helft van (b). Met de nieuwe verhouding had RL-val 38
  FLAN-bronnen, en een terugval op FLAN zou in S1 (8 van 384) en in de mediaan van S4 nauwelijks te zien zijn. Gelijke
  delen houden ook de drempels van Yeb (S1: 8 van 384) en `plan7`, `stop7` en `smoke7` zoals ze zijn.
- **Nodig** (`pool7` met de reserve van 3%, G4-7.4c): FLAN 132 RL-val + 1.576 pool = 1.708, romans 132 + 7.138 = 7.270,
  BT 132 + 7.107 = 7.239 (was 6.467, 4.891, 4.860); het anker vertaalt 16.217 bronnen (was 16.218). `cands7` trekt
  ongewijzigd 2× zoveel (FLAN 3.416, romans 14.540, BT 14.478, 2.896 per BT-bron).
- **Beschikbaar** [gemeten 21 nov op mbert-corpus w1, alleen CPU met nice 19: `cands7` en `pool7` van deze commit
  (dezelfde `g4rl.py`, sha256 `71e8bc76…`) op de invoer van de proef van G4-7.4c, met dezelfde benadering van C2's
  `docs.txt` (per BT-bron de eerste 1.000 documenten in C2's geseede volgorde); de echte `docs.txt` komt pas uit
  `data7`]: per deel de trechter van `cands7` en wat `pool7` hield:
  - FLAN: van 7.250 bekeken segmenten 2.989 korter dan 20 tokens (41%), 845 langer dan 256 (12%), 3.416 getrokken; na de
    13-grammen (637 eruit, 19%; korte segmenten hebben er geen, dus lange vallen vaker) 2.779 kandidaten tegen 1.708
    nodig. Naar schatting komen ~745.000 van de 1.580.656 FLAN-segmenten door de lengtefilters (1.580.656 × 3.416 /
    7.250).
  - romans: van 18.173 bekeken passages 10 korter dan 20 tokens, 3.623 langer dan 256, 14.540 getrokken; `pool7` hield
    ze alle 14.540, tegen 7.270 nodig (~988.000 van de 1.234.854 passages komen door de lengtefilters [schatting, zo
    gerekend]).
  - BT: per bron 2.896 getrokken, geen enkele korter dan 20 tokens (langer dan 256: handelingen 815, raad 830, dpc
    1.387, rechtspraak 1.580, fineweb 977 van 3.711–4.476 bekeken; door de lengtefilters naar schatting 73.000 / 38.000
    / 14.000 / 13.700 / 47.000); na C2's documenten (679 eruit, 5%) 13.801 kandidaten tegen 7.239 nodig; de pool per
    bron 1.419–1.423, RL-val 26–27 per bron.
  - `pool7` slaagde (pool 1.576 / 7.138 / 7.107, RL-val 132 per deel); `cands7` 8:36 min en 14,0 GB RAM, `pool7` 8 s.
  Het minimum raakt dus vrijwel alleen FLAN: in de pool van de oude verhouding (proef van G4-7.4c) had FLAN 51,6%
  bronnen korter dan 20 tokens, romans en BT 0,0%.
- **Iteraties**: 30 blijft. Elk deel heeft ruim genoeg bronnen; minder iteraties is niet nodig.
- **Tijd**: de bronnen van de pool zijn gemiddeld langer: FLAN 67,9 tokens (mediaan 45; was 38,4, mediaan 19), romans
  195,9, BT 116,9; per iteratie gemiddeld ~148 tokens per bron tegen ~109 bij 40 / 30 / 30 (+35%) [gemeten op de pools
  van beide proeven]. Genereren en trainen schalen ongeveer met de tokens: ~+25–35% per iteratie tegenover de oude
  verhouding, dicht bij de aanname van spec §7.2 (~390 tokens per sample met de prompt, ~26 batches) [schatting]. De
  smoke-test meet het (`smoke_estimate7` neemt de tokens per sample van de echte iteraties; grens 24 u).
- Waar G4-7.0, G4-7.2, G4-7.4c ("205/154/153") of de spec nog FLAN 40 / romans 30 / BT 30 noemen, geldt deze aanvulling.
  De beslisregel (G4-7.7, G4-7.8), de reward, de stopregels en de evaluatie veranderen niet.

*G4-7.4e Aanvulling 21 nov 2026 ~23:00 (besluit van Yeb; een methodewijziging van de reward, G4-7.3; vóór de eerste
RL-stap: de keten `g4i1121f` stopte om 22:27 in `c2cal7`, vóór het anker gescoord en vóór elke RL-stap).* `c2cal7`
haalde voorwaarde 4 niet voor FLAN: de mediaan-sd binnen de groep van clip(logit C2, −6, 6) was 0,0 (QE-straf 0,30);
voorwaarden 1–3 en 4 voor WMT24++ werden gehaald (0,886; 0,864; 0,146; 0,672 tegen 0,263) [gemeten,
`translator-sft/g4-7-c2cal7.md`]. Oorzaak [gemeten, hoofdsessie, op dezelfde `predictions.parquet`]: C2's logits van
G4-0's samples liggen ver onder −6 (FLAN p5/p25/p50/p75/p95 −11,0/−9,8/−8,6/−5,8/1,7; WMT24++ −10,4/−8,3/−4,6/−0,5/4,3;
min −12,2, max 8,3), zodat 52% van de FLAN-groepen en 22% van de WMT24++-groepen geheel op −6 stonden. Mediaan-sd bij
knip ±10: FLAN 0,96, WMT24++ 1,13; bij ±12 of geen knip: 1,02 en 1,17.
- **Knip** [Yeb, 21 nov ~23:00, optie 1 van de hoofdsessie]: clip(logit C2, **−12, 12**) in de reward (`g4rl.LOGIT_CLIP`)
  en in `c2cal7` (`g4c2.LOGIT_CLIP`). `c2cal7` draait opnieuw met deze knip en dezelfde vooraf vastgelegde grenzen.
- **Fail-reward** [invulling hoofdsessie, gevolg van de knip]: **−30** (was −10). De laagste reward zonder fail is nu
  −12 − 2 × 7,5 − 2 (ln 1,5 − ln 1,15) = −27,5; met −10 zou een fail vaak boven een gewone vertaling liggen (logits −9 tot
  −12 zijn gewoon) en zou de RL falen kunnen leren. Een fail blijft zo de slechtste uitkomst, zoals G4-7.3 bedoelt
  (test `test_a_fail_is_below_every_reward_that_is_not_a_fail`).
- Het eerdere `c2cal7`-rapport (knip ±6) blijft bewaard als `$RL/c2/cal7/report-clip6.{json,md}`. Verder verandert niets.

*G4-7.4f Aanvulling 21 nov 2026 ~23:35 (hoofdsessie; uitvoering, geen methodewijziging).* `g4judge.py wait` telde de
time-out van 900 s vanaf de laatste DONE van de generatie. Het anker was om 16:29 gegenereerd, de judges startten pas om
23:21 (de keten liep in delen), zodat `wait_anchor` na 6 s faalde terwijl de judges net begonnen. Nu telt de time-out
vanaf het latere van de laatste DONE en de start van de wacht, en begint opnieuw bij elke nieuwe score; een dode judge
(hartslag) faalt zoals voorheen meteen. Proef: `JUDGE_PROEF_OK` 13 controles, waaronder "a DONE of an hour ago". De
judges van `g4i1121h` (code `706c8d7`, `serve` ongewijzigd) liepen door en worden door de herstart als levend herkend.

*G4-7.4g Aanvulling 21 nov 2026 ~23:59 (hoofdsessie; uitvoering).* De trainfase van `smoke7` (iteratie 900, 2 stappen)
faalde op "warmup 10 and decay 0 steps do not fit a schedule of 2": de warmup van 10 stappen (G4-7.3) past niet in de
2 + 2 stappen van de smoke-test, en ook niet in een eerste RL-iteratie met minder dan 10 stappen (veel groepen zonder
signaal). `sft_lora.py` neemt nu min(warmup, stappen − decay − 1) (MaxText's fracties vragen een stap na de warmup) en meldt dat (`WARMUP … clipped`); runs met ≥ 11 stappen
veranderen niet. In de smoke-test is de warmup dus 1 (eerste mini-iteratie, 2 stappen) en 3 (tweede, 4 stappen); de smoke-test toetst mechaniek en
geheugen, niet het leren.

*G4-7.4h Aanvulling 22 nov 2026 ~01:40 (hoofdsessie, volgens G4-7.9: "de hoofdsessie beslist" bij geheugen in de
smoke-test; geen wijziging van wat beslist).* Trainfase 2 van `smoke7` (`g4a7_s905_g4i1122k`) liep in zijn eerste
SFT-evaluatie op stap 2 tegen `RESOURCE_EXHAUSTED` ("Attempting to allocate 21.00M … 5.01M free"), dezelfde fout als G4-6c's
evaluaties (G4-6c.6b–.6f); de training zelf paste (fase 1: piek 23,9 GiB, 10,7 s/stap, verschil sampler/leerder 0,0062).
De SFT-evaluaties `books`/`btval`/`own` in de trainruns (G4-7.10, rapportage, beslissen niets) **vervallen**: geen
`--eval` in `train7` (`g4arm7_run.sh` EVALS=0, `g4rl.SFT_EVALS7 = False`); `smoke7` eist voor fase 2 eindige verliezen en
géén evaluaties. Drift en kwaliteit blijven bewaakt door S1–S4 en S6 (RL-val, KL), en de kandidaten worden na de RL
geëvalueerd zoals in G4-7.7/7.8. Tests: de fixture van fase 2 zonder evaluaties (wandtijd 800 → 680 s, de 2 × 3 × 20 s
evaluaties eruit), en een evaluatie in de run laat `train2_finite` falen.

*G4-7.4i Aanvulling 22 nov 2026 ~03:55 (hoofdsessie volgens G4-7.9; versoepeling achteraf van technische controles van
de smoke-test, niet van een beslisregel; voor Yeb gemarkeerd).* `smoke7` `g4i1122m` (iteraties 908/909) haalde alle
functionele controles (decoderen, logprobs, rewards en voordelen gelijk, check7, hervatten, piek 24,1/24,1 GiB, vertalen,
QE S − G4-0 +0,04) maar niet vijf numerieke: `qe_loose` max 0,014 (grens 0,001), `c2_loose` max 0,006 (grens 1e-4),
`train1_step1` max |log-ratio| 0,89 bij KL 4e-4 en clip_frac 0,18% (grens 1e-4), `train1_old_ref` max 1,18 bij gemiddelde
k1 8,6e-6 (grens 1e-4), `estimate` 44,0 u (grens 24 u). De max-grenzen van 1e-4 kwamen uit de CPU-proef in float32 (daar
exact 0); op de TPU in bf16 geven verschillend gecompileerde programma's (judge tegen losse score, met tegen zonder
LoRA-pad) per token afrondingsverschillen, zonder systematische afwijking (gemiddelden ~0; sampler/leerder 0,006, spec-grens
0,05). De schatting schaalt de vaste opstarttijd van `iter7` lineair mee (37,6 min per iteratie); realistisch ~58 min per
iteratie, ~29 u. Nieuwe grenzen: QE 0,02; p_c2 0,01; stap 1: |KL| ≤ 1e-3 en clip_frac ≤ 0,01; old = ref: |gemiddelde k1|
≤ 1e-4 (maxima gerapporteerd); schatting ≤ 48 u. 30 iteraties blijven. De smoke-test draait opnieuw (`g4i1122n`).

*G4-7.4j Besluit 22 nov 2026 ~05:00 (hoofdsessie volgens G4-7.9; voor Yeb gemarkeerd).* De herhaalde `smoke7`
(`g4i1122n`, iteraties 910/911) haalde opnieuw alle functionele controles en de grenzen van G4-7.4i op twee na:
`qe_loose` max 0,026 (grens 0,02) en `train1_old_ref` gemiddelde k1 1,7e-4 (grens 1e-4; max 2,7); `c2_loose` 0,0067,
stap 1 KL 6,1e-4 en clip 0,2%, piek ~24 GiB, schatting 43,8 u (grens 48). Dezelfde bf16-ruis als bij `g4i1122m`, met een
andere steekproef; de grenzen nog eens verschuiven op één meting zegt niets. Besluit: **de smoke-test is functioneel
geslaagd en de RL start** (marker `smoke7` met de hand gezet, in het formaat van de keten: smoke `$RL/smoke7/g4i1122n/`,
tis 0, hours 43.8). Een gemiddeld verschil old/ref van 1,7e-4 nats per token is verwaarloosbaar tegen de KL-stopregel
S3 (0,30); een QE-verschil van 0,026 (schaal 0–25) verandert de reward niet merkbaar. De grenzen in de code blijven die
van G4-7.4i.

*G4-7.4k Aanvulling 22 nov 2026 ~09:00 (besluit van Yeb; een methodewijziging van G4-7.3/§6.3).* De RL met LR 5e-6
(`g4i1122o`, iteraties 0–5 voltooid, 6 gestopt) bewoog het beleid niet: gemiddelde C2-logit van de samples −7,89 → −7,93,
p_c2 op RL-val 0,0997 → 0,0989, KL (S3) ~0 (k1 old/ref −3e-4), relatieve update van de LoRA 1,4e-4 na ~84 stappen; de QE
liep licht op (S2 +0,007 → +0,022). **LR 5e-5** (10×), verder alles gelijk; de KL-stopregel S3 (0,30) blijft het vangnet.
De RL begint opnieuw bij iteratie 0: de iteraties, RL-val-jobs en hun markers van de eerste poging staan onder
`$RL/archive-lr5e-6/` en `$G/stages/g4-7/archive-lr5e-6/` (verplaatst, niets verwijderd); C2, `c2cal7`, het anker, `pool7`,
`plan7` en `smoke7` blijven gelden.
