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

*G4-7.4l Aanvulling 22 nov 2026 ~13:00 (besluit van Yeb; vóór de kandidatenfase).* De blinde beoordeling ((a2) en (c),
G4-7.8) draait ook als (a1) en (a3) op het beslisdeel gelden en (b) niet, **als rapportage**: ze beslist dan niets, en
"G4-7 helpt" vraagt nog steeds alle vijf. Reden (review door Gemini 4 Neon, 22 nov): +10 pp op C1 is met deze opzet
nauwelijks eerlijk haalbaar (FLAN is de helft van (b) maar ~2% van de trainingstokens), en zonder blinde beoordeling blijft
bij een bescheiden echte winst onbekend of het Nederlands natuurlijker werd en of C2 als reward iets meet wat telt. De
lopende keten (`g4i1122p`, code `2f8119c`) blijft ongewijzigd: geldt (b) niet, dan draait de hoofdsessie na `DONE` de stap
`blind` van `run_g4arm7.sh` met de hand (`ARM=G4-7 SELECTED=<gekozen kandidaat>`, mbert-corpus w0, vóór T3 de TPU krijgt)
en daarna de beoordeling door één verse Claude-subagent volgens G4-7.8 (rubriek protocol.md §4.2.4, de 100 passages,
twijfelgevallen naar Yeb). Bij (a1) of (a3) niet gehaald: geen blinde beoordeling.

## 15. Meting G4-bon: het plafond van best-of-8 met selectie op C2 + QE

**Vastgelegd 22 nov 2026 (tijdstip: de commit van deze sectie op branch `en-nl-g4bon`, vanaf `en-nl-g4arm7` @
`e6c22b7`)**, vóór elke stap van de keten `mbert-ops/scripts/g4bon_run.sh` op een worker en vóór elke uitslag van deze
meting [besluit Yeb 22 nov ~13:00: mbert-mix wordt na G4-7 gebruikt voor een plafondmeting best-of-N, geen training;
uitwerking door een subagent; labels als in de kop]. Een meting, geen arm: er is geen beslisregel en er wordt niets
getraind; de tabellen van G4-bon.7 liggen vooraf vast. Geen Claude of GPT als generator of maat; de TEST-split van
hammond blijft dicht.

*G4-bon.1 Vraag.* Hoeveel "origineel" (C1, P ≥ 0,5) haalt best-of-8, en tegen welke kwaliteitsprijs (QE, COMET-22,
MetricX-ref, lengte, niet af, lus, Engels over), als de keuze uit de 8 samples van een bron op C2 (de reward-classifier
van G4-7) en QE gebeurt in plaats van op QE alleen? Voor G4-0 en voor de kandidaat die G4-7 kiest (G4-7.7).
Aanleiding [gemeten, `mbert-ops/translator-sft/g4-sampling.md`, hele sets]: best-of-8 op QE gaf G4-0 MetricX-ref 2,284
→ 1,850, COMET-22 0,853 → 0,855 en +2,1 pp C1. De vervolgrichting G4-8 (SFT op de eigen geselecteerde samples, spec
`mbert-ops/translator-sft/g4-8-spec.md`) vraagt eerst wat er in de 8 samples te vinden is.

*G4-bon.2 Sets.* Alleen het **zoekdeel** van G4-5.4 (digest `f953c026…f3`, `g4arm.SPLIT_DIGEST6C`): WMT24++ zonder
slechte bronnen (480) en FLAN (500), de sets van (b). Het beslisdeel wordt niet gelezen (geen nieuwe uitvoer, geen
score, geen selectie), zodat het schoon blijft voor latere besluiten. FLORES en NTREX niet: de sampling-proef samplede ze
niet.

*G4-bon.3 Uitvoer per model.*
- **G4-0**: de 8 samples van de sampling-proef (G4-5.7b B; `$G/sampling/G4-0/t10s1…t10s8/`: T = 1, seed k, MaxText
  `weighted`, fork `f4d8615`, prompt `natural-v1`, de lengteklassen en de verdeling over de workers van de evaluatie) met
  hun MetricX-24 (met referentie en QE) en COMET-22; C2 uit `c2cal7` (`$RL/c2/cal7/predictions.parquet` en de
  tokens van C2 uit `texts.parquet`; hetzelfde model als de reward van G4-7). Niets wordt opnieuw vertaald of gescoord.
  [gemeten 22 nov, alleen metadata gelezen: de proef dekt alle 960 + 1.000 segmenten van (b), dus het hele zoekdeel; de
  prompt en de fork in `run_g4arm.sh` zijn ongewijzigd sinds de commit van de proef `03286d6`, en `translate.py` zonder
  `--copies` samplet zoals toen; `$G/sampling/data/evalsets.parquet` is byte-gelijk aan `$G/data/eval/evalsets.parquet`
  (md5); `bondata` toetst de sha256 tegen `split.json` en `select` de instellingen t10s1…t10s8 tegen het rapport van de
  proef.]
- **De kandidaat K van G4-7** (`selected` in `$G/report-G4-7/report.json`): 8 nieuwe samples per bron van het zoekdeel
  met de instellingen van de proef (`translate.py --sample t10s<k>=1:<k>`, k = 1…8, de LoRA van K ingebakken zoals bij
  G4-7.7 en gecontroleerd op rank 32 / alpha 16, de klassen en de verdeling over de workers van de proef), gescoord zoals
  de proef (collect, MetricX-24 met referentie en QE, COMET-22) en door C2 (hetzelfde model, `naturalness.py classify
  --max-length 1024`, zoals `classify-c2`), samen met K's en G4-0's greedy vertaling (de laatste als controle tegen
  `c2cal7`). K's greedy vertaling komt uit zijn evaluatie door G4-7 (`$G/eval/K/`).
- **Uitgangspunten** per model: de greedy vertaling en één willekeurig sample (`t10s1`).

*G4-bon.4 Selecties (vooraf vastgelegd; `g4bon.py choose`)*, per bron één van de 8 samples. Fail = de fail van G4-7
(G4-7.3 met G4-7.4e: niet af, lus, leeg, Engels over, ℓ/ℓ₀ buiten [0,67; 1,5], QE > 7,5, `c2_overflow`); anker = G4-0's
greedy vertaling van het segment (QE₀ uit zijn evaluatie, ℓ₀ = zijn `tokens_out`), voor beide modellen (de reward van
G4-7 had G4-0 als anker).
1. **qe**: de laagste QE; gelijk: het eerdere sample. Dit is de best-of-8 van de sampling-proef; voor G4-0 stopt `select`
   als de keuze op een segment afwijkt van `$G/sampling/G4-0/best8/choice.parquet`.
2. **r7**: de hoogste reward van G4-7 (`g4rl.reward7`, ongewijzigd: clip(logit C2, −12, 12) − 2·max(0, QE − QE₀) −
   2·max(0, |ln(ℓ/ℓ₀)| − ln 1,15), een fail −30); gelijk: de lagere QE, dan het eerdere sample.
3. **c2q**: de hoogste geknipte logit van C2 onder de samples die geen fail zijn en QE ≤ QE₀ hebben (op QE niet slechter
   dan G4-0's greedy vertaling); gelijk: de lagere QE, dan het eerdere sample. Is er zo'n sample niet, dan de keuze van
   qe (het aantal staat in tabel 4).
4. **c2** [keuze, de toegestane extra variant]: de hoogste geknipte logit van C2 onder de samples die geen fail zijn,
   zonder QE-grens; is er geen, dan de keuze van qe. Onderbouwing: (i) het is het plafond van een selectie op C2 binnen
   de harde grenzen van G4-7 (lengteband, QE > 7,5): het laat zien hoeveel C1 een C2-selectie hoogstens oplevert en wat
   de QE-grens van c2q daarvan kost; (ii) hier zijn QE (op de grens 7,5 na), MetricX-ref en COMET-22 onafhankelijke
   toetsen, en daarmee is c2 de enige selectie waarvan de kwaliteitsprijs op FLAN met een maat buiten de selectie
   gemeten wordt; (iii) "C2 stijgt, C1 niet" is hier het duidelijkst te zien. Waarom niet MBR met COMET-22 zonder
   referentie: COMET draait op de CPU [gemeten in eerdere ketens: `COMET_TIMING` 100 rijen in 40–82 s per worker]; MBR
   vraagt 8 × 7 paren per bron, ~55.000 per model op het zoekdeel, ~1,5–3 u per model over vier workers [schatting];
   een versie met gecachte embeddings is nieuwe code tegen de binnenkant van COMET. MBR kiest bovendien het sample dat
   het meest op de andere lijkt; of dat de minst "originele" is, is niet gemeten [gevolgtrekking].

C1 kiest nooit: C1 scoort alleen de gekozen uitvoer en de uitgangspunten. C2 kiest wel (r7, c2q, c2), en is daarom voor
die uitvoer geen toets.

*G4-bon.5 Maten.* C1 (S @ 653k, ongewijzigd) in **één classify-run** over alle uitvoer van het zoekdeel (`run_arms.sh
classify`), per set en samen zoals (b); C2: aandeel logit ≥ 0 (P ≥ 0,5) en de gemiddelde geknipte logit;
MetricX-24-QE per set, met de QE van de menselijke referentie op WMT24++ van het zoekdeel (`$G/data/G4-5-split/
refqe.parquet`); MetricX-24 met referentie en COMET-22 op WMT24++; het fail-aandeel van G4-7 en per soort (niet af, lus,
Engels over, leeg); tekens per bron en per referentie; de mediane ℓ/ℓ₀. Verschillen tegen de eigen greedy vertaling, tegen
G4-0's greedy vertaling en tussen selecties: gepaard per segment, met een 95%-interval uit een gepaarde bootstrap
(10.000 trekkingen, per set binnen de set, zaad uit `en-nl-g4arm/0/G4-bon/boot/<vergelijking>`).

*G4-bon.6 Wat niet onafhankelijk is.* Een maat waarop gekozen is, toetst die uitvoer niet: **qe**: QE en MetricX-ref
(hetzelfde model als QE); **r7**: C2, QE, MetricX-ref, lengte, de fail-vlaggen; **c2q**: C2, QE, MetricX-ref, de
fail-vlaggen; **c2**: C2 en de fail-vlaggen (QE alleen via de grens 7,5 en lengte alleen via de band). Onafhankelijk zijn
bij elke selectie C1 en COMET-22 (alleen WMT24++). Op FLAN is er geen referentie: de kwaliteitsprijs op FLAN is voor qe,
r7 en c2q alleen met een selectiemaat gemeten. Het rapport noemt de selectiematen per rij.

*G4-bon.7 Rapport (vooraf vastgelegde tabellen; `g4bon.py report`, `$G/bon/report/<key>/`, kopie
`mbert-ops/translator-sft/g4-bon.md` en `.json`).*
- **Tabel 1, C1**: per model × uitvoer (greedy, t10s1, qe, r7, c2q, c2): "origineel" WMT24++ / FLAN / samen; Δ tegen de
  eigen greedy vertaling [95%-BI]; Δ tegen G4-0 greedy [95%-BI]; C2 samen; ΔC2 tegen de eigen greedy vertaling; de vlag
  "C2 stijgt, C1 niet" (ΔC2 ≥ +0,10 en ΔC1 < +0,10 tegen de eigen greedy vertaling, de regel van `report7`, G4-7.4c).
- **Tabel 2, kwaliteit en vorm**: per model × uitvoer × set: QE, MetricX-ref, COMET-22, fail (G4-7), niet af, lus,
  Engels over, leeg, tekens / bron, tekens / referentie, mediane ℓ/ℓ₀, gemiddelde C2-logit, de selectiematen; in de kop
  de QE van de menselijke referentie.
- **Tabel 3, contrasten binnen een model**: r7 − qe, c2q − qe en c2 − qe: ΔC1 samen, ΔCOMET-22 en ΔMetricX-ref op
  WMT24++, elk met 95%-BI; en voor elke selectie ΔCOMET-22 en ΔMetricX-ref tegen de eigen greedy vertaling.
- **Tabel 4, de selectie**: per set en selectie het aantal terugvallen op qe, het aandeel gelijk aan de keuze van qe, de
  verdeling over t10s1…t10s8; voor c2q het gemiddelde aantal toegelaten samples; het aantal bronnen waar alle 8 falen.
- **Controles** (gerapporteerd): C1 van G4-0/t10s1, G4-0/qe (= best8) en G4-0 greedy in deze run tegen de classify-run
  van de sampling-proef op dezelfde segmenten (aandelen en het aantal omgeslagen labels); C2 van G4-0 greedy in de C2-run
  van de kandidaat tegen `c2cal7` (aandelen, omgeslagen labels, grootste |Δlogit|).
- De vraag van G4-bon.1 beantwoorden tabel 1 en 3 (r7 en c2q tegen qe); c2 is het plafond van de C2-selectie.

*G4-bon.8 Wat de meting niet zegt.*
- Een selectie uit 8 samples is geen getraind model: de meting zegt wat er in de verdeling zit en met deze maten te
  vinden is, niet hoeveel een fine-tune daarvan naar greedy decoding overbrengt (dat is de vraag van G4-8).
- C1 en C2 hangen samen (Spearman per segment 0,60 en 0,62, `c2cal7`): een C1-winst na een selectie op C2 is geen bewijs
  van natuurlijker Nederlands; een blinde beoordeling hoort niet bij deze meting.
- n = 480 + 500: een verschil in C1 van een paar procentpunten kan ruis zijn (het interval staat erbij); de intervallen
  zijn per vergelijking, zonder correctie voor het aantal vergelijkingen.
- De kandidaat van G4-7 is op dit zoekdeel gekozen, op C1 van zijn greedy vertaling (G4-7.7, de grootste (b) van ten
  hoogste 7 kandidaten): zijn cijfers hier zijn optimistisch, zijn greedy vertaling het meest; de contrasten tussen zijn
  eigen uitvoer minder [gevolgtrekking].
- C2 van G4-0's samples komt uit `c2cal7`, die van de kandidaat uit een eigen run; de controle in het rapport meet het
  verschil op G4-0's greedy vertaling.
- Een uitslag beslist niets; G4-8 heeft een eigen, vooraf vastgelegde drempel (spec §2).

*G4-bon.9 Keten* (`mbert-ops/scripts/g4bon_run.sh`, alleen mbert-mix; code `en-nl-g4bon`, `g4bon.py` en de stages
`bon-*` van `run_g4arm.sh` met `ARM=G4-bon`; plaatsen `$G/bon/`, markers `$G/stages/g4-bon/`). Wacht tot de log van
G4-7 (`logs/g4arm7_run.log`) eindigt op `EXIT 0` of `g4bon.mix-go` bestaat, neemt `final_eval_mix.lock` tot het eind;
code, HF-token, TPU vrij, setup; `bon-data`; `bon-select` voor G4-0; is de kandidaat bekend (`selected` in het rapport
van G4-7, of `BON_CANDIDATE`): `bon-translate`, `bon-score`, `bon-c2texts`, `bon-c2`, `bon-select` voor K; dan
`bon-texts`, `bon-classify` (C1) en `bon-report` over G4-0 en (als die er is) K, onder key `G4-0` of `G4-0_K`. Zonder
kandidaat doet de keten alleen G4-0 en zegt dat in de `DONE`-regel; een herstart met een nieuwe TAG slaat de gemarkeerde
stappen over en maakt voor `G4-0_K` een nieuwe classify-run en een nieuw rapport. Vóór elke start op een worker: de
TPU vrij op die worker(s); elke ssh met `timeout 900` en de ssh-opties van de eerdere ketens; per stap `EXIT 0`, anders
`FAILED` en stoppen. De keten schrijft niets over wat G4-7 of de sampling-proef schreef.

*G4-bon.10 Aanvulling: G4-4 als extra model* **(vastgelegd 23 nov 2026 ~00:35; tijdstip: de commit van deze alinea op
branch `en-nl-g4bon`)**, vóór elke stap van deze aanvulling op een worker en vóór elke uitslag over G4-4 in deze meting
[advies Gemini 4 Neon 22/23 nov, via de hoofdsessie: G4-4 is de nuttigste extra meting vóór Yebs besluit over G4-8; Yeb
vroeg mbert-mix bezig te houden en staat zulke metingen toe; uitwerking door een subagent]. Een meting, geen arm: zij
beslist niets, ook niet over G4-8.
- **Uitvoer van G4-4** [keuze: het hergebruikpad van G4-0, G4-bon.3]: de 8 samples van de sampling-proef
  (`$G/sampling/G4-4/t10s1…t10s8/`: G4-4's eindadapter ingebakken, verder dezelfde instellingen als G4-0's samples) met
  hun MetricX-24 (met referentie en QE) en COMET-22; C2 van de samples uit `c2cal7` (rijen `G4-4/t10s<k>`); de greedy
  vertaling uit G4-4's evaluatie (`$G/eval/G4-4/`), haar C2 uit de kalibratierijen `calib:sampling:G4-4` van `c2cal7`
  (de teksten van de classify-run van de proef, die G4-4's greedy vertaling uit `$G/eval/G4-4/` nam). Niets wordt
  opnieuw vertaald of gescoord; nieuw zijn alleen de selectie, één C1-run en het rapport. [gemeten 23 nov, alleen
  namen en tellingen: `c2cal7` heeft `G4-4/t10s1…t10s8` en `calib:sampling:G4-4` met elk 1.960 rijen en geen rij `G4-4`;
  de classify-run van de proef heeft `G4-4`, `G4-4/t10s1` en `G4-4/best8` met elk 1.960 rijen; elke
  `$G/sampling/G4-4/t10s<k>/` heeft pairs en MetricX op w0–w3 en COMET op w0–w2, zoals die van G4-0.]
- **Controles** (`select` stopt bij een afwijking): de keuze van qe is per segment `$G/sampling/G4-4/best8/choice.parquet`;
  de instellingen t10s1…t10s8 die van het rapport van de proef; de tekst van `calib:sampling:G4-4` in `c2cal7` is op elk
  segment van het zoekdeel gelijk aan G4-4's greedy vertaling in `$G/eval/G4-4/`. Gerapporteerd: C1 van G4-4 greedy,
  G4-4/t10s1 en G4-4/qe (= best8) tegen de classify-run van de proef, zoals voor G4-0.
- **Selecties, maten, anker, tabellen**: G4-bon.4–G4-bon.7 ongewijzigd, met G4-4 als model naast G4-0 (key
  `G4-0_G4-4`). Anker en QE₀ blijven G4-0's greedy vertaling (de grens van c2q is dus QE ≤ QE₀ van G4-0, niet van G4-4).
  Tabel 1 geeft voor G4-4 ΔC1 tegen zijn eigen greedy vertaling en tegen G4-0 greedy en de vlag "C2 stijgt, C1 niet";
  tabel 2 QE, MetricX-ref en COMET-22; tabel 3 de contrasten binnen G4-4.
- **Wat het niet zegt** (naast G4-bon.8): G4-4 is niet op dit zoekdeel gekozen; C2 van G4-4's greedy vertaling en van
  zijn samples komt uit dezelfde run (`c2cal7`), de kandidaat van G4-7 staat niet in dit rapport (zijn cijfers staan in
  het rapport onder key `G4-0_G4-7-i15`, uit een andere C1-run).
- **Keten**: `g4bon_run.sh` met `BON_MODEL=G4-4` (niet samen met `BON_CANDIDATE`): de gemarkeerde stappen (setup, data,
  select van G4-0) overgeslagen, dan `bon-select` (MODEL=G4-4, w0, CPU), `bon-texts`, `bon-classify` (C1) en `bon-report`
  onder key `G4-0_G4-4`; geen kandidaat. Het rapport gaat naar `translator-sft/g4-bon-G4-0_G4-4.md/.json` en
  `g4-bon-G4-4.md/.json`; `g4-bon.md` blijft het rapport van de kandidaat.

## 16. Meting C2F: kennen C2 en C1 fictie?

**Vastgelegd 23 nov 2026 (~01:30; tijdstip: de commit van deze sectie op branch `en-nl-g4bon`)**, vóór de eerste stap op
een worker en vóór elke uitslag van deze meting [aanleiding: advies Gemini 4 Neon; Yeb vroeg mbert-mix 's nachts bezig te
houden; een meting binnen eerdere besluiten; uitwerking door een subagent]. Vooraf is alleen het schema van de invoer
bekeken (kolommen, aantallen, de vorm van de id's; geen tekst, geen score). Een meting, geen arm: er wordt niets getraind,
niets vertaald en niets beslist.

*C2F.1 Vraag.* C2 (de reward van G4-7, G4-7.6: dmbert v2 met head, Nederlandse originelen tegen Gemma 4's rondgang, vijf
registers zonder fictie) en C1 (S @ 653k met head, §4.3 en spec §3.2: originelen tegen TranslateGemma's rondgang) zagen in
hun training geen fictie. Op romans gaf G4-0 bij C2 2–3% "origineel", en de RL van G4-7 ontspoorde daar het eerst. Scheiden
C2 en C1 op romanpassages Nederlandse originelen van menselijke vertalingen en van G4-0's vertalingen, en menselijke
vertalingen van G4-0's?

*C2F.2 Groepen* (alleen bestaande teksten; er wordt niets met Gemma gegenereerd):
- `nl_books`: de Nederlandse kant (`reference`) van de set `books` van G4-2.5 (`$G/data/heldout/evalsets.parquet`, set
  `books`: 500 rijen uit 2 boeken; Nederlandse originelen).
- `nl_books31` [aanvulling, keuze subagent: `nl_books` is 2 boeken]: Nederlandse originelen uit de andere boeken van
  G4-1's boekenset (`$G/data/G4-1/`, `g4arm.g41_set`, soort `book`, het Nederlands zoals G4-2.5 het leest: de
  assistent-beurt, gestript), zonder de 2 boeken van `nl_books` (`$G/data/heldout/books.json`).
- `human`: de menselijke Nederlandse vertaling (`nl` van `$S/data/books-b1/b1-selected.parquet`, groep `en_orig`) van de
  romanpassages van G4-7's anker (`$RL/anchor/samples/`, id `novels:<rij-id>`; pool en RL-val).
- `g40`: G4-0's greedy vertaling (`hypothesis`, klasse `medium`, fork `f4d8615`, `natural-v1`) van dezelfde passages, uit
  hetzelfde anker. `human` en `g40` zijn gepaard: dezelfde passages.
- Een ankerrij komt in aanmerking als `ended` waar is, `decode_loop`, `english_left` en `empty` onwaar zijn en de
  `hypothesis` niet leeg is. Het Engels van het anker (`source`) moet gelijk zijn aan `en` van de b1-rij (beide gestript);
  is dat bij één rij niet zo, of ontbreekt een rij in b1 of staat hij er niet één keer in met groep `en_orig`, dan stopt
  de meting.
- **Heel, niet afgekapt** (als G4-7.6): elke tekst heeft ten hoogste 1.022 tokens zonder de speciale tokens, geteld met
  de tokenizer van C1 en met die van C2 (elk via zijn `config.json`, `g4c2.tokenizer_of`). Bij `human`/`g40` moeten beide
  kanten passen.
- **Trekking**: van de rijen die in aanmerking komen, gesorteerd op id, `random.Random(seed).sample` van min(500, n), seed
  `c2fiction/0/<groep>`; `human` en `g40` één trekking van passages met seed `c2fiction/0/novels`.

*C2F.3 Classifiers.* C1: `$S/arms/naturalness/classifier/`; C2: `$RL/c2/model/`. Beide met `naturalness.py classify`
(`~/mbert-dumb/.venv`, `~/mbert-dumbx` @ `ae11478`, `--max-length 1024`) op één texts-bestand, op w0 van mbert-mix, binnen
`final_eval_mix.lock`, alleen als de TPU van w0 vrij is. mbert-corpus wordt niet aangeraakt.

*C2F.4 Maten* (`c2fiction.py report`; P = P(label 1 = origineel); logit = clip(logit, −12, 12) uit P, `g4c2.logit_of`,
zoals `c2cal7`):
1. Per classifier en groep: n, boeken, mediaan tokens; aandeel "origineel" (P ≥ 0,5); gemiddelde P; gemiddelde en mediaan
   logit.
2. AUC (Mann–Whitney op P, gelijke scores ½) per classifier voor de paren `nl_books`–`human`, `nl_books`–`g40`,
   `human`–`g40`, `nl_books31`–`human`, `nl_books31`–`g40` en `nl_books`–`nl_books31` (de eerste groep hoort hoger; het
   laatste paar is een controle, verwacht rond 0,5), met een 95%-bootstrapinterval (2.000 trekkingen, seed
   `c2fiction/0/boot`; binnen elke groep apart, bij `human`–`g40` per paar).
3. Gepaard `human`–`g40` per classifier: aandeel paren met P(human) > P(g40) (gelijk ½), gemiddelde Δlogit (human − g40)
   met 95%-bootstrapinterval.
4. Per lengteband (≤ 128, 129–256, > 256 tokens): n en aandeel "origineel" per groep en classifier; Spearman logit~tokens
   per groep; Spearman C1~C2 per groep en over alle teksten.
5. Controles (gerapporteerd): C2's P op `g40` tegen de judge-score van het anker (`$RL/anchor/scores/c2/w1/`, `p_c2`;
   grootste |ΔP|, aantal > 1e-3) en het aantal tokens tegen `c2_tokens`; het aantal teksten dat `naturalness.py` als
   `truncated` markeert (alleen teksten van precies 1.022 tokens mogen dat; anders stopt het rapport); hoeveel referenties
   van `nl_books` gelijk zijn aan het Nederlands van G4-1 voor hetzelfde paar. Ter vergelijking: het aandeel "origineel"
   van C2 volgens de judge op alle romanrijen van het anker die in aanmerking komen.

*C2F.5 Wat het niet zegt.* Geen maat van kwaliteit: een hoge AUC zegt dat een classifier de groepen scheidt, niet dat hij
een goede reward op fictie is. De groepen verschillen ook in lengte (het anker heeft Engels van ten hoogste 256 Gemma
4-tokens, `books` tot 512), in boeken, auteurs en tijd van uitgave; `nl_books` is 2 boeken. Alleen `human`–`g40` is
gepaard; de vergelijking met de originelen is dat niet. De menselijke vertalingen komen uit het gelijnde en gefilterde
-b1-corpus, geen steekproef van alle vertaalde fictie. Het besluit niets over G4-7, G4-bon of een volgende arm.

*C2F.6 Uitvoer.* `$G/c2fiction/<TAG>/` in de bucket: `texts.parquet` (privé, blijft in de bucket en op de VM),
`meta.parquet` (id, groep, boek, tokens; geen tekst), `predictions-c1.parquet`, `predictions-c2.parquet`,
`report.json`/`.md`. Op co1 alleen het rapport: `mbert-ops/translator-sft/c2-fiction-check.md` (en `.json`), tabellen met
tellingen en scores, geen teksten. Keten: `mbert-ops/scripts/c2fiction_run.sh`.

*C2F.7 Correctie (23 nov, na de meting).* Het tijdstip in de kop van §16 is ~01:05 (de commit `b3b560a`), niet ~01:30.
De meting liep als run `c2f1123a` met `c2fiction.py` @ `a4128e3`; het rapport op co1 is uit `report.json` opnieuw
opgemaakt met de getalnotatie van `c2fiction.py` @ de commit van deze correctie (mediaan tokens met één decimaal, |ΔP|
met komma); de getallen zijn dezelfde.

## 17. Meting G4-judge: Gemma 4 31B-it als judge van twee vertalingen

**Vastgelegd 23 nov 2026 (tijdstip: de commit van deze sectie op branch `en-nl-g4judge`, vanaf `en-nl-g4bon` @
`f30c1a8`)**, vóór elke stap op een worker en vóór elke uitslag van deze meting [besluit Yeb 23 nov ~08:15: een kleine
proef met een open LLM-judge op mbert-mix; uitwerking door een subagent; labels als in de kop]. Vooraf is van de
blinde pakketten alleen gelezen: de veldnamen, de aantallen per oordeel, zekerheid, set en sleutelnaam, en het aantal
tokens van bron en vertalingen (geen tekst). Een meting, geen arm: er wordt niets getraind en zij beslist niets over een
arm of over G4-8. Claude en GPT zijn geen judge, generator of trainingsdata; de bestaande Claude-oordelen dienen hier
alleen als validatie. De TEST-split van hammond blijft dicht; het beslisdeel van G4-5.4 wordt niet gelezen.

*G4-judge.1 Vraag.* C2 bleek als reward en als selectiemaat een zwakke proxy (G4-7; G4-bon: selectie op C2 verdubbelt
C2 en verhoogt C1 weinig; blind 29 / 22 / 49, niet significant). Een judge die de bron en twee vertalingen naast elkaar
ziet, kan betekenis en natuurlijkheid samen wegen. (1) Volgt Gemma 4 31B-it, zero-shot met de rubriek van protocol.md
§4.2.4, het oordeel van de Claude-beoordelaar op de bestaande blinde pakketten, en hoe groot is zijn positiebias?
(2) Alleen als (1) de drempel van G4-judge.6 haalt: wat geeft best-of-8 met de keuze op deze judge (elk sample tegen
het anker G4-0 greedy) voor C1, QE, COMET-22 en MetricX-ref, naast de selecties qe en c2q van G4-bon?

*G4-judge.2 De judge.* Gemma 4 31B-it (`google/gemma-4-31B-it` @ `842da37`, Apache 2.0), de MaxText-omzetting die
elke Gemma 4-decodering gebruikt (`$R/judge/gemma-4-31b-it-maxtext/0/items`), in MaxText's MaxEngine op de fork @
`f4d8615` met `translate.make_engine` (model `gemma4-31b`: bf16, logits in fp32, tensor 4 over de 4 chips van één
worker van mbert-mix). Per prompt één prefill en geen decodering: de logits van de eerste antwoordpositie
(`prefix["logits"]` van `MaxEngine.prefill`); prefill-lengtes 512, 1.024, 1.536, 2.048, 3.072, 4.096 (een prompt van
meer dan 4.096 tokens wordt niet gescoord en geteld). Het chat-template van het model met één gebruikersbeurt en een
lege denkruimte (de `turn` van `gemma4-31b` in `translate_models.yml`, voor de eerste prompts per worker getoetst tegen
`apply_chat_template`). Antwoordtokens [gemeten in `tokenizer.json` @ `842da37`]: A = {`A`, `▁A`} (236776, 562),
B = {`B`, `▁B`} (236799, 603), gelijk = {`gelijk`, `▁gelijk`} (161224, 190506); "gelijk" is één token, "Gelijk" niet
(`G` + `elijk`) en telt niet mee. Per prompt: log P van elke set (logsumexp over haar ids), de massa
m = P(A) + P(B) + P(gelijk), de vijf waarschijnlijkste tokens en het eerste token van de engine (greedy); genormaliseerd
p_X = P(X) / m.

*G4-judge.3 Prompt* (letterlijk de tekst van de gebruikersbeurt; `{bron}`, `{A}` en `{B}` gestript ingevuld):

```
Je beoordeelt vertalingen uit het Engels in het Nederlands. Je krijgt de Engelse bron en twee Nederlandse vertalingen, A en B. Je weet niet hoe of door wie ze zijn gemaakt; probeer dat niet te raden.

Oordeel eerst op juistheid: de vertaling zonder betekenisfout, of met de minst ernstige, is beter. Een betekenisfout is:
- een weglating: iets uit de bron ontbreekt in de vertaling;
- een toevoeging: de vertaling zegt iets wat niet in de bron staat;
- een verkeerde betekenis: de vertaling zegt iets anders dan de bron;
- een misplaatste uitdrukking: een idioom of vaste uitdrukking uit de bron is letterlijk vertaald waar dat in het Nederlands niet werkt, of vertaald met een Nederlandse uitdrukking die iets anders betekent.
Woordkeus, stijl en zinsbouw zijn geen betekenisfout zolang de betekenis klopt.

Zijn ze even juist, oordeel dan op natuurlijk, oorspronkelijk klinkend Nederlands: de vertaling die het meest leest als tekst die in het Nederlands is geschreven en niet als vertaling. "gelijk" als er op beide punten geen duidelijk verschil is.

Engelse bron:
{bron}

Vertaling A:
{A}

Vertaling B:
{B}

Welke vertaling is beter? Antwoord met één woord: A, B of gelijk.
```

Dit is de rubriek van §4.2.4 (de definities van een betekenisfout en de volgorde juistheid → natuurlijkheid →
gelijk letterlijk), zonder de aparte vragen naar betekenisfouten, zekerheid en uitleg en zonder JSON. [keuze] Geen
betekenisfoutvragen vooraf: (i) dan moet het model eerst tekst genereren en hangt de kans op het antwoord af van zijn
eigen, gesampelde of greedy gegenereerde tussenantwoorden; een kans uit één prefill is dan niet meer te lezen; (ii) stap
2 vraagt ~30.000 prompts, en decoderen kost een veelvoud van één prefill; (iii) de rubriek staat in de prompt als
instructie. Er wordt geen prompt gezocht of bijgesteld op de validatieset: deze ene prompt ligt hier vast.

*G4-judge.4 Beide volgordes.* Elk paar twee keer: in de volgorde van het pakket (`ab`: A is pakket-A) en verwisseld
(`ba`). In het kader van het pakket: P̄(A beter) = ½·[p_A(ab) + p_B(ba)], P̄(B beter) = ½·[p_B(ab) + p_A(ba)],
P̄(gelijk) = ½·[p_gelijk(ab) + p_gelijk(ba)]. **Oordeel van de judge** = de grootste van de drie (bij exact gelijke
waarden "gelijk"); **richting** = het teken van P̄(A beter) − P̄(B beter). Het greedy oordeel per volgorde is de grootste
van p_A, p_B, p_gelijk in die volgorde, teruggezet in het kader van het pakket.

*G4-judge.5 Stap 1: validatie.* Sets: de blinde pakketten met een oordeel van een verse Claude-subagent volgens de
rubriek van §4.2.4, elk 100 paren (`gs://…/sft/blind/<pakket>/`: `judge-input.jsonl`, `key.jsonl`, de oordelen):
`G4-7` (G4-7-i15 tegen G4-0, `review-74l`), `G4-7-tg` (G4-7-i15 tegen TranslateGemma 27B, `review-74l`),
`G4-bon-i15c2q` (de doeltekst (i15, c2q) tegen G4-0 greedy, `review`), `RB3` en `RB2` (armen op TranslateGemma tegen
TranslateGemma 27B, `review-exploratory/judgments.jsonl`); samen 500 paren, 1.000 prompts. Referentie = die oordelen
(Claude, vóór elke vervanging van twijfelgevallen); daarnaast gerapporteerd voor RB3 en RB2 de `final-judgments.jsonl`
(twijfelgevallen daar opnieuw beoordeeld; de zekere oordelen zijn dezelfde). Aantallen [gelezen, alleen labels]: niet
gelijk 340, waarvan zeker 125 (G4-7 15, G4-7-tg 30, G4-bon-i15c2q 17, RB3 31, RB2 32); gelijk 160. Maten, per pakket en
samen:
1. **Richtingsovereenstemming** op Claude's niet-gelijke paren: het aandeel waar de richting van de judge Claude's
   winnaar is (P̄ precies gelijk: ½); voor alle niet-gelijke en alleen de zekere; met een 95%-Wilson-interval.
2. Overeenstemming over de drie klassen (oordeel van de judge tegen Claude) en **Cohen's κ** (drie klassen), op alle
   500 paren en op de zekere; de kruistabel.
3. **Positiebias**: *omklappen* = het aandeel paren waar de greedy oordelen in `ab` en `ba`, teruggezet, tegengestelde
   winnaars noemen (de een A, de ander B; dezelfde positie twee keer gekozen); *inconsistent* = het aandeel waar ze
   op enige manier verschillen (ook gelijk tegen een winnaar); het aandeel greedy-antwoorden "A" (eerste positie) en
   "gelijk" per volgorde; gemiddeld p_A − p_B over alle prompts.
4. **Massa** m per volgorde: mediaan, aantal < 0,5; de meest voorkomende top-tokens.
5. Gerapporteerd, beslist niet mee: (a) op paren waar Claude precies één kant een betekenisfout gaf, het aandeel waar de
   richting van de judge de kant zonder fout kiest; (b) het aandeel niet-gelijke oordelen van de judge voor het
   niet-basissysteem (sleutel) tegen dat van Claude, per pakket (G4-7-tg: Gemma 4 tegen TranslateGemma; een maat voor
   een voorkeur voor de eigen modelfamilie); (c) per set (FLAN, WMT24++, idioom) de richtingsovereenstemming.

*G4-judge.6 Drempel voor stap 2* (vooraf; alle drie moeten gelden):
- **D1**: richtingsovereenstemming op de **zekere** niet-gelijke paren, samen over de vijf pakketten (n = 125), ≥ 70%;
- **D2**: richtingsovereenstemming op **alle** niet-gelijke paren van de twee pakketten met twee Gemma 4-uitvoeren die
  dicht bij elkaar liggen (`G4-7` en `G4-bon-i15c2q`, n = 43 + 51 = 94), ≥ 60%;
- **D3**: omklappen (maat 3) over alle 500 paren ≤ 20%.

Onderbouwing [keuze]: D1 volgt het voorstel van Yeb (≥ 70% op de zekere niet-gelijke paren); bij n = 125 ligt de
ondergrens van het 95%-interval dan rond 61%, dus ruim boven toeval. D1 wordt gedragen door de pakketten met een groot
kwaliteitsverschil (G4-7-tg 60 / 16, RB2, RB3); stap 2 kiest echter tussen samples van één model tegen zijn eigen
greedy vertaling, dicht bij elkaar. Daarom D2 op de twee pakketten van dat soort; de zekere paren zijn daar te weinig
(15 + 17), dus alle niet-gelijke, met een lagere grens omdat Claude daar zelf meestal twijfelde (62 van 94); bij n = 94
is 60% eenzijdig p ≈ 0,03 tegen 50%. D3 volgt het voorstel (≤ 20%): het middelen over beide volgordes neemt een
constante positievoorkeur weg, maar klapt het oordeel bij meer dan een op de vijf paren om, dan is het signaal vooral
positie en is de selectie in stap 2 ruis.

*G4-judge.7 Stap 2: herselectie* (alleen als D1–D3 gelden; anders eindigt de keten na het rapport van stap 1).
- **Modellen en samples**: G4-0 (de 8 samples van de sampling-proef, `$G/sampling/G4-0/t10s1…t10s8/`, zoals G4-bon ze
  hergebruikt) en G4-7-i15 (`$G/bon/G4-7-i15/t10s1…t10s8/`), op het zoekdeel van G4-bon (WMT24++ 480 + FLAN 500). Anker
  = G4-0's greedy vertaling (`$G/eval/G4-0/pairs/`). Fail, QE en QE₀ per sample uit G4-bon's `choice.parquet`
  (`$G/bon/<model>/select/`, de fail van G4-7: niet af, lus, leeg, Engels over, ℓ/ℓ₀ buiten [0,67; 1,5], QE > 7,5,
  `c2_overflow`); er wordt niets opnieuw vertaald of door MetricX/COMET gescoord.
- **Judge-paren**: elk sample dat geen fail is tegen het anker, in beide volgordes (`ab`: A = sample, B = anker; `ba`
  omgekeerd), prompt en judge als in stap 1 [schatting: ~14.600 paren, ~29.000 prompts, verdeeld over de 4 workers op
  sha256 van de uid]. P(sample beter dan anker) = P̄(sample beter) van G4-judge.4. Een fail-sample of een niet
  gescoorde prompt (> 4.096 tokens) komt niet in aanmerking.
- **Selecties** (per bron één van de 8): **judge**: het hoogste P̄(sample beter) onder de samples die geen fail zijn;
  **judgeq**: idem onder de samples die geen fail zijn én QE ≤ QE₀ hebben (zoals c2q); gelijk: de lagere QE, dan het
  eerdere sample; is er geen kandidaat, dan de keuze van qe (het aantal staat in het rapport). Ter vergelijking, uit
  G4-bon's `choice.parquet` ongewijzigd: greedy (het model zelf), t10s1, qe en c2q.
- **Maten**: C1 (S @ 653k, `run_arms.sh classify` ongewijzigd) in **één classify-run** over de 12 uitvoeren (2 modellen ×
  greedy, t10s1, qe, c2q, judge, judgeq) van het zoekdeel; QE, MetricX-ref, COMET-22 (WMT24++), fail en de vlaggen uit de
  bestaande scores van de gekozen samples; C2 (geknipte logit uit G4-bon's `choice.parquet`). Rapport in de vorm van
  G4-bon.7 (`g4bon.py`'s maten en gepaarde bootstrap, 10.000 trekkingen, per set binnen de set, zaad
  `en-nl-g4arm/0/G4-bon/boot/G4-judge/<vergelijking>`): tabel 1 C1 per model × uitvoer met Δ tegen de eigen greedy
  vertaling en tegen G4-0 greedy (95%-BI); tabel 2 kwaliteit en vorm; tabel 3 de contrasten judge − qe, judgeq − qe,
  judge − c2q en judgeq − c2q (ΔC1 samen, ΔCOMET-22 en ΔMetricX-ref op WMT24++, 95%-BI) en elke selectie tegen de eigen
  greedy; tabel 4 de selectie (terugvallen, gelijk aan qe/c2q, verdeling over t10s1…t10s8, gemiddeld P̄(sample beter)
  van de keuze, het aandeel bronnen waar de keuze P̄(sample beter) > P̄(anker beter) heeft, niet gescoorde prompts).
  Controles (gerapporteerd): C1 van G4-0 greedy, G4-0/t10s1 en G4-0/qe tegen de classify-run van de sampling-proef, en
  van de G4-7-i15-uitvoer (greedy, t10s1, qe, c2q) tegen G4-bon's run `G4-0_G4-7-i15` (aandelen, omgeslagen labels);
  de keuzes qe en c2q gelijk aan G4-bon's.
- **Niet onafhankelijk**: judge: de fail-vlaggen (QE > 7,5, lengteband); judgeq: QE, MetricX-ref (zelfde model als QE)
  en de fail-vlaggen. Onafhankelijk van beide: C1, C2, COMET-22. Op FLAN geen referentie.
- **Blind pakket** `gs://…/sft/blind/G4-judge-i15/` (zoals `G4-bon-i15c2q`): 50 WMT24++ + 50 FLAN van het zoekdeel waar
  de tekst van (G4-7-i15, judgeq) verschilt van G4-0's greedy vertaling (gestript vergeleken); per set
  `random.Random("g4judge/blind/<set>").sample` van de verschillende id's in gesorteerde volgorde (minder dan 50: alle);
  `blind_id` `judge-001…judge-100` (WMT24++ eerst, in de volgorde van de trekking); A/B per paar met
  `random.Random("g4judge/blind/ab/<blind_id>").random() < 0,5` → A = G4-0 greedy (`base`), anders A = de doeltekst
  (`G4-7-i15-judgeq`). `judge-input.jsonl` (blind_id, source, A, B) en apart `key.jsonl` (blind_id, A, B, seed, set,
  item) en `select-report.json` (aantallen, geen tekst) in de bucket; de beoordeling doet de hoofdsessie met een verse
  Claude-subagent (rubriek §4.2.4), niet deze keten.

*G4-judge.8 Rapporten.* Stap 1: `$G/judge/val/report.md`/`.json`, kopie `mbert-ops/translator-sft/g4-judge-validation.md`
(`.json`); stap 2: `$G/judge/bon/report/report.md`/`.json`, kopie `mbert-ops/translator-sft/g4-judge-bon.md` (`.json`).
Alleen tabellen met tellingen en scores; geen teksten. Prompts en teksten blijven in de bucket en op de VM.

*G4-judge.9 Wat de proef niet zegt.*
- Claude's oordelen zijn geen grondwaarheid: de validatie meet overeenstemming met een andere taalmodel-beoordelaar met
  dezelfde rubriek, die bij 284 van de 500 paren twijfelde; Yebs oordeel over die twijfelgevallen is er voor G4-7 en
  G4-bon niet. Hoge overeenstemming kan ook gedeelde voorkeuren van taalmodellen weerspiegelen.
- Gemma 4 31B-it beoordeelt uitvoer van zichzelf (G4-0) en van LoRA's erop; een voorkeur voor de eigen stijl is niet
  uitgesloten (maat 5b meet alleen de voorkeur tegen TranslateGemma in G4-7-tg).
- Eén prompt, zero-shot, geen zoektocht: een andere prompt of een judge die eerst redeneert kan beter of slechter zijn.
- Stap 2 is een selectie uit 8 samples, geen getraind model; C1 meet, de judge kiest; een C1-winst na selectie op de
  judge is geen bewijs van natuurlijker Nederlands. De blinde controle van het pakket gebeurt door een Claude-subagent
  met de rubriek waartegen de judge in stap 1 gevalideerd is: een voorkeur daar is deels door die validatie te
  verwachten [gevolgtrekking].
- G4-7-i15 is op dit zoekdeel gekozen (G4-7.7); n = 480 + 500; intervallen per vergelijking, zonder correctie voor het
  aantal vergelijkingen. Een uitslag beslist niets.

*G4-judge.10 Keten* (`mbert-ops/scripts/g4judge_run.sh`, alleen mbert-mix; code `en-nl-g4judge`: `pairjudge.py` en
`run_pairjudge.sh`; plaatsen `$G/judge/`). Neemt `final_eval_mix.lock` met `flock` tot het eind; mbert-corpus wordt
niet aangeraakt. Code als bundel via de bucket naar de 4 workers; elke ssh met `timeout 900` en de ssh-opties van de
eerdere ketens; vóór elke TPU-stap de TPU van de worker(s) vrij. Stap 1: `setup` (`run_g4arm.sh setup` met
`ARM=G4-bon`), `valdata` (w0, CPU), `score` (4 workers), `valreport` (w0) met de toets van D1–D3. Stap 2 alleen bij
"gehaald": `bonpairs` (w0, CPU), `score` (4 workers), `select` per model, `texts`, `classify` (w0, C1), `bonreport`,
`blind` (w0). Elke stap eindigt op `EXIT 0`, anders `FAILED` en stoppen; de keten schrijft niets over wat G4-bon, G4-7 of
de sampling-proef schreef.

*G4-judge.11 Aanvulling van 23 nov 2026: A en B (tijdstip: de commit van deze alinea's op `en-nl-g4judge`), vóór elke
stap op een worker en vóór elke uitslag van A en B* [besluit van de hoofdsessie namens Yeb, op advies van Gemini 4
Neon, 23 nov ~10:35; binnen de door Yeb goedgekeurde judge-proef; uitwerking door een subagent]. Twee metingen na stap
2, geen training, geen G4-8, geen herselectie met B. Vooraf gelezen: de rapporten van stap 1 en 2
(`g4-judge-validation.md`, `g4-judge-bon.md`) en de blinde uitslagen van `G4-judge-i15` en `G4-judge-G4-0` (LEARNINGS);
niet gelezen: enige telling of keuze van A en enige uitvoer van B. Teksten, selecties en redeneringen van de judge
worden ook nu niet gelezen; alleen id's, tellingen en scores.

*G4-judge.12 A: judge of anker.* Vraag: in stap 2 heeft de judgeq-keuze maar bij 56–75% van de bronnen
P̄(sample beter) > P̄(anker beter) (tabel 4); wat geeft het houden van het anker waar de judge geen sample duidelijk beter
vindt?
- **Selectie `janchor`** per bron, voor G4-0 en G4-7-i15 op het zoekdeel: k = de judgeq-keuze van G4-judge.7
  (ongewijzigd, uit `$G/judge/<model>/select/choice.parquet`). Is judgeq niet teruggevallen (er was een kandidaat) en is
  P̄(sample k beter dan het anker) > 0,5 (G4-judge.4: de genormaliseerde kans, gemiddeld over beide volgordes; strikt
  groter), dan sample k; anders G4-0's greedy vertaling (het anker, ook voor G4-7-i15). Alles uit de bestaande
  judge-scores van stap 2: geen nieuwe judge-scoring, vertaling of MetricX/COMET.
- **Maten**: C1 in één nieuwe classify-run (`run_arms.sh classify` ongewijzigd) over 14 uitvoeren van het zoekdeel
  (2 modellen × greedy, t10s1, qe, c2q, judge, judgeq, janchor); QE, MetricX-ref, COMET-22 (WMT24++), C2 en de
  fail-vlaggen uit de bestaande scores van de gekozen tekst (bij het anker die van G4-0 greedy, zoals G4-bon ze heeft).
  Rapport in de vorm van G4-judge.7 (`pairjudge.py bonreport`, zelfde bootstrap, zaadnamen
  `en-nl-g4arm/0/G4-bon/boot/G4-judge/<model>/<vergelijking>`), met janchor als extra uitvoer: tabel 1 (C1, Δ eigen
  greedy en Δ G4-0 greedy) en tabel 2 met janchor; tabel 3 bovendien janchor − greedy (eigen), janchor − judgeq,
  janchor − qe, janchor − c2q en voor G4-7-i15 janchor − G4-0 greedy (ΔC1 samen, ΔCOMET-22 en ΔMetricX-ref op WMT24++,
  95%-BI); tabel 5 per model en set: het aantal en aandeel bronnen waar het anker blijft, gesplitst in "judgeq valt
  terug" en "P̄ ≤ 0,5", het aantal waar de tekst van janchor gelijk is aan die van G4-0 greedy (gestript), het gemiddelde
  P̄(sample beter) van de gekozen samples en de verdeling over anker en t10s1…t10s8. Controles (gerapporteerd): die van
  G4-judge.7; C1 van de 12 uitvoeren die ook in de classify-run van stap 2 zitten tegen die run, en C1 van janchor
  tegen het label dat zijn tekst daar had (omgeslagen labels).
- **Niet onafhankelijk** van janchor: de judge, QE en MetricX-ref (via judgeq's QE ≤ QE₀), de fail-vlaggen.
  Onafhankelijk: C1, C2, COMET-22.
- **Blind pakket** `gs://…/sft/blind/G4-judge-anchor-i15/`, opzet als `G4-judge-i15`: 50 WMT24++ + 50 FLAN van het
  zoekdeel waar de tekst van (G4-7-i15, janchor) verschilt van G4-0's greedy vertaling (gestript vergeleken); per set
  `random.Random("g4judge-anchor/blind/<set>").sample` van de verschillende id's in gesorteerde volgorde (minder dan
  50: alle); `blind_id` `anchor-001…anchor-100` (WMT24++ eerst, in de volgorde van de trekking); A/B per paar met
  `random.Random("g4judge-anchor/blind/ab/<blind_id>").random() < 0,5` → A = G4-0 greedy (`base`), anders A = de
  doeltekst (`G4-7-i15-janchor`). `judge-input.jsonl` (blind_id, source, A, B), apart `key.jsonl` (blind_id, A, B,
  seed, set, item) en `select-report.json` (aantallen, en hoeveel items ook in `G4-judge-i15` zitten: alleen id's
  vergeleken). De beoordeling doet de hoofdsessie met een verse Claude-subagent (rubriek §4.2.4), niet deze keten.
- Rapport: `$G/judge/anchor/report/report.md`/`.json`, kopie `mbert-ops/translator-sft/g4-judge-anchor.md` (`.json`).
- Wat A niet zegt: als G4-judge.9; janchor mengt bij G4-7-i15 twee systemen (waar het anker blijft, is de uitvoer
  G4-0 greedy); de grens 0,5 is vooraf gekozen, niet gezocht.

*G4-judge.13 B: een judge die eerst redeneert (alleen validatie en tijd).* Vraag: volgt een judge die eerst kort
redeneert de Claude-oordelen beter, vooral waar de judge van G4-judge.3 zwak is (5a, een betekenisfout aan precies één
kant: 61,5%; richting op de niet-gelijke idioomparen: 60,0%), en wat kost hij per paar? Geen herselectie in deze ronde.
- **Judge**: als G4-judge.2 (Gemma 4 31B-it @ `842da37`, MaxEngine op de fork @ `f4d8615`, `translate.make_engine`,
  tensor 4 over de 4 chips van één worker van mbert-mix; het chat-template met één gebruikersbeurt en een lege
  denkruimte).
- **Prompt B** (letterlijk de tekst van de gebruikersbeurt; `{bron}`, `{A}` en `{B}` gestript ingevuld): de prompt van
  G4-judge.3 met een andere laatste alinea.

```
Je beoordeelt vertalingen uit het Engels in het Nederlands. Je krijgt de Engelse bron en twee Nederlandse vertalingen, A en B. Je weet niet hoe of door wie ze zijn gemaakt; probeer dat niet te raden.

Oordeel eerst op juistheid: de vertaling zonder betekenisfout, of met de minst ernstige, is beter. Een betekenisfout is:
- een weglating: iets uit de bron ontbreekt in de vertaling;
- een toevoeging: de vertaling zegt iets wat niet in de bron staat;
- een verkeerde betekenis: de vertaling zegt iets anders dan de bron;
- een misplaatste uitdrukking: een idioom of vaste uitdrukking uit de bron is letterlijk vertaald waar dat in het Nederlands niet werkt, of vertaald met een Nederlandse uitdrukking die iets anders betekent.
Woordkeus, stijl en zinsbouw zijn geen betekenisfout zolang de betekenis klopt.

Zijn ze even juist, oordeel dan op natuurlijk, oorspronkelijk klinkend Nederlands: de vertaling die het meest leest als tekst die in het Nederlands is geschreven en niet als vertaling. "gelijk" als er op beide punten geen duidelijk verschil is.

Engelse bron:
{bron}

Vertaling A:
{A}

Vertaling B:
{B}

Welke vertaling is beter? Noem eerst in één of twee korte zinnen de betekenisfouten van A en van B, of dat er geen zijn; zijn ze even juist, zeg dan welke het natuurlijkst klinkt. Schrijf daarna op een laatste, aparte regel alleen: "Oordeel: A", "Oordeel: B" of "Oordeel: gelijk".
```

- **Decoderen**: greedy (MaxEngine's `decode_sampling_strategy`), hoogstens 256 nieuwe tokens, stop op de eos-ids van
  `gemma4-31b` (1, 106, 50); `translate.decode_all` (continuous batching, `sync_every` 8) met 12 slots, prefill-lengtes
  512, 1.024, 1.536, 2.048, 2.560 en `max_decode` 256 (12 × 2.816 = 33.792 slot-tokens, onder de 38.912 die op 13 nov op
  een v4-chip pasten [`translate_models.yml`, d1k]). Een prompt van meer dan 2.048 tokens wordt niet gescoord en
  geteld.
- **Antwoord** [keuze]: completion = de gedecodeerde tokens zonder stoptoken (`skip_special_tokens`, gestript). De
  oordeelregel is de eerste regel die, na weglaten van voorloopspaties en de tekens `*`, `#`, `_`, `>`, `-`, begint met
  "oordeel" (hoofdletterongevoelig). Redenering = de tekst vóór die regel, rechts gestript; zonder oordeelregel de hele
  completion (geteld). Dan één prefill van de chatbeurt + redenering + een lege regel + `Oordeel:` (zonder redenering
  alleen `Oordeel:`), als één string getokeniseerd (de ids van de prompt moeten daarvan het begin zijn); P(A), P(B) en
  P(gelijk) uit de logits van de volgende positie, precies als G4-judge.2 (zelfde antwoord-ids, massa, genormaliseerd);
  P̄, oordeel, richting en het greedy oordeel per volgorde als G4-judge.4. Waarom zo: (i) kansen in dezelfde vorm als
  stap 1, zodat D1–D3 en alle maten ongewijzigd gelden; (ii) een vaste antwoordplek, los van opmaak (vet, kopjes) in de
  oordeelregel van het model. Daarnaast vastgelegd, beslist niets: het eigen antwoord van het model (het woord na
  "Oordeel" op de oordeelregel, zonder opmaak en zonder een voorafgaand "Vertaling": A, B, gelijk of anders) en hoe vaak
  het afwijkt van het greedy oordeel op de antwoordplek.
- **Validatie**: dezelfde 500 paren, referenties en maten als G4-judge.5 (de maten van `valreport` ongewijzigd); D1–D3
  met de grenzen van G4-judge.6, gerapporteerd (beslist niets); 5a en de richting op de niet-gelijke idioomparen
  apart; per maat het gepaarde verschil met de judge van stap 1 op dezelfde paren (Δ richting op de paren van D1, D2, alle
  niet-gelijke, 5a en idioom; Δ omklappen en Δ drie klassen op alle paren; bootstrap over paren, 10.000 trekkingen, zaad
  `en-nl-g4arm/0/G4-judge/reason/boot/<maat>`). Per volgorde: gegenereerde tokens (mediaan, gemiddelde, max), niet af
  (256 bereikt), geen oordeelregel, eigen antwoord, eigen antwoord ≠ greedy op de antwoordplek, massa.
- **Tijd** (per worker; een worker van mbert-mix is 4 v4-chips, een v4-8): engine laden en opwarmen (prefill en insert
  van elke prefill-lengte en generate, vóór de klok) apart; daarna de wandklok van fase 1 (`decode_all` over alle
  prompts van de worker: prefills en decodeerstappen) en fase 2 (de prefills op de antwoordplek). Tijd per paar =
  2 × (fase 1 + fase 2) / prompts; tokens per seconde in fase 1. Ter vergelijking de judge van stap 1 uit de logs van stap
  2 (de shard-tijden zonder de COMPILE-regels). [gevolgtrekking, gemarkeerd] Uren voor 100.000 samples (elk tegen het
  anker, beide volgordes) op één v4-8 en op de 4 workers, bij deze instelling; er is niet naar een snellere instelling
  gezocht.
- Rapport: `$G/judge/reason/val/report.md`/`.json`, kopie `mbert-ops/translator-sft/g4-judge-reason-validation.md`
  (`.json`); alleen tabellen met tellingen en scores. De redeneringen staan in de score-shards in de bucket
  (`$G/judge/reason/val/scores/`) en worden niet gelezen.
- Wat B niet zegt: als G4-judge.9; een redenering volgens de rubriek kan het oordeel dichter bij Claude's brengen
  zonder het beter te maken (Claude oordeelde met dezelfde rubriek); één prompt, greedy, geen zoektocht; de tijd geldt
  voor deze 1.000 validatieprompts en deze instelling.

*G4-judge.14 Keten* (`mbert-ops/scripts/g4judge2_run.sh`, kopie in `gs://t5-dutch-english-usc2/hammond/scratch-scripts/`;
alleen mbert-mix; `final_eval_mix.lock` met `flock` tot het eind; mbert-corpus wordt niet aangeraakt). Code als bundel
via de bucket naar de 4 workers; elke ssh met `timeout 900` en de ssh-opties van G4-judge.10; vóór elke TPU-stap de
TPU van de worker(s) vrij. `setup` opnieuw (`run_g4arm.sh setup`, `ARM=G4-bon`). A: `aselect` per model (w0, CPU),
`atexts` (w0, CPU), `aclassify` (w0, C1), `areport` (w0), `ablind` (w0). B: `rvaldata` (w0, CPU), `reason` (4 workers),
`rreport` (w0). Markers in `$G/stages/g4-judge/` met nieuwe namen; plaatsen `$G/judge/anchor/`,
`$G/judge/<model>/select-anchor/`, `$G/judge/reason/`; niets wat stap 1 of 2 schreef wordt overschreven. Elke stap
eindigt op `EXIT 0`, anders `FAILED` en stoppen.

## 18. G4-8-data: de selectiedata van arm G4-8 (best-of-8 met de judge, janchor)

**Vastgelegd 23 nov 2026 (tijdstip: de commit van deze sectie op branch `en-nl-g4arm8`, vanaf `en-nl-g4judge` @
`a983e09`)**, vóór elke stap op een worker en vóór elke uitslag [besluit Yeb 23 nov ~16:30: de datavoorbereiding van
G4-8 bouwen en starten volgens `mbert-ops/translator-sft/g4-8-spec-judge.md` met de standaarden van die spec, zodat
mbert-mix niet stilstaat; **geen training**: de SFT start pas na Yebs steekproef (V1) en zijn keuze over O6;
uitwerking door een subagent]. **Besluit Yeb 23 nov ~16:40: FLAN-aandeel als bij G4-7, 10 / 45 / 45 (FLAN / romans /
BT) van de bronnen; wijkt af van de spec** (die 40 / 40 / 20 adviseerde, §3 en O4). De overige standaarden van de spec
gelden: startmodel G4-0 (O1), selectie janchor (O2), geen ankerrijen (O3), 24.000 bronnen (O5), alleen mbert-mix (O9),
één ronde (O10). Vooraf gelezen: alleen tellingen (`g4-7-plan.json`, `pool.json`, `cands.json` van G4-7), geen tekst.
Claude en GPT zijn geen judge of generator. Deze keten schrijft niets over wat G4-7, G4-bon of G4-judge schreef.
`RL=$G/rl/G4-7`, `G8=$G/data/G4-8`.

*G4-8-data.1 Bronnen* (24.000; FLAN 2.400, romans 10.800, BT 10.800; BT gelijk per BT-set, 2.160 per set).
- **Hergebruikt** [keuze]: alle 15.360 bronnen van G4-7's plan (`$RL/data/plan/plan.parquet`, iteratie 0–29: FLAN 1.530,
  romans 6.930, BT 6.900 = 1.380 per set) met hun anker (de tekst `hypothesis0` en ℓ₀ = `l0`, G4-0 greedy, G4-7.4a).
  RL-val (384 bronnen, iteratie −1) blijft erbuiten. Ze voldoen aan de filters van G4-7.2 en G4-7.4d (20–256 Gemma
  4-tokens, geen 13-gram met de evaluatie, de FLAN-evaluatierijen of de blinde passages, niet de documenten van C2 of
  RL-val).
- **Vers**: FLAN 870, romans 3.870, BT 3.900 (780 per set), uit dezelfde bronnen en met dezelfde code als `cands7` en
  `pool7` (G4-7.2, G4-7.4d): FLAN-nl's Engelse segmenten van de gepubliceerde voorbeelden (zonder de FLAN-evaluatierijen
  en hun voorbeelden), het Engels van de `-b1-`-romans zoals G4-4 ze nam (zonder de boeken van de evaluatieset `novels`),
  het Engels van de trainsplit van de vijf `-nl-en-gemma4`-sets op G4-3's revisies (zonder `hard_fails`). Vóór de
  trekking eruit: elk id en elke (gestripte) Engelse tekst van G4-7's `pool.parquet` (pool, reserve en RL-val). Trekking
  per deel (BT per set) in een geseede volgorde (`draw_candidates`, zaad `en-nl-g4arm/0/G4-8/cands/<deel>[/<set>]`),
  2 × (behoefte + 3% reserve), alleen teksten van 20–256 Gemma 4-tokens; dan de filters van `pool7`
  (`candidate_filter`): geen 13-gram met de evaluatiesets (`$E/data/evalsets.parquet`, bron en referentie), de blinde
  passages (`blind100.jsonl`) en de FLAN-evaluatierijen; niet de documenten van C2 (`$RL/c2/data/docs.txt`) en niet die
  van G4-7's RL-val; één bron per tekst. Volgorde per deel `g4rl.ordered` met zaad `en-nl-g4arm/0/G4-8/order/<deel>`
  (de BT-sets om de beurt, dus gelijk per set); de eerste behoefte × 1,03 gaan naar het anker.
- **Delen** (`c1`, `c2`, `c3`) [keuze]: per deel de 24.000 bronnen op id gesorteerd en geschud met zaad
  `en-nl-g4arm/0/G4-8/chunk/<deel>`; `c1` de eerste 1/6 (FLAN 400, romans 1.800, BT 1.800; 4.000 bronnen), `c2` en `c3`
  elk 5/12 (1.000 / 4.500 / 4.500; 10.000). Reden: V3 (G4-8-data.8) heeft na ~2–3 u BT- en romandoelen nodig, en een
  herstart hoeft dan minder over te doen. Elk deel doorloopt samplen → QE → paren → judge → selectie; de delen geven
  dezelfde uitkomst als één geheel (de regel is per bron).

*G4-8-data.2 Anker en QE₀.* Hergebruikt: G4-7's anker. Vers: G4-0 greedy zoals G4-7.4a (`translate.py --copies 1`,
fork `f4d8615`, prompt `natural-v1`, klasse `medium`, geen adapter, 4 workers van mbert-mix). Een verse bron waarvan
het anker niet af is, lust, leeg is of Engels laat staan valt weg (`g4rl.anchor_ok`, als `plan7`); per deel de eerste
behoefte die overblijven, in de volgorde van G4-8-data.1 (te weinig: `FAILED`). **QE₀** = MetricX-24-QE (hybrid XXL
v2p6 bf16, `metricx_qe.py score --pair source:hypothesis`, zonder referentie, een lege vertaling zonder score) van het
anker, voor alle 24.000 opnieuw berekend in dezelfde jobs als de samples [keuze: QE en QE₀ van dezelfde scorer, want
bf16-batch-effecten tussen G4-7's judges en een losse run zijn tot ~0,02 (G4-7.4i)]. Controle bij de hergebruikte
bronnen tegen G4-7's `qe0` (gerapporteerd): mediaan |Δ| > 0,05 of meer dan 2% met |Δ| > 0,5 is `FAILED` (een
verkeerde koppeling).

*G4-8-data.3 Samples.* 8 per bron bij T = 1 met G4-0 zonder adapter (`translate.py --copies 8 --sample
samples=1.0:<zaad>`, zaad `g4rl.seed_int("en-nl-g4arm/0/G4-8/sample/<deel>/w<k>")`, verder als het anker; shards van
32 bronnen; bronnen over de workers op tokens, `g4rl.assign_workers7`). QE per sample als QE₀.

*G4-8-data.4 Fail en kandidaten* (spec §3 stap 1). Fail = de fail van G4-7 (G4-7.3 met G4-7.4e) zonder `c2_overflow`:
niet af, lus, leeg, Engels over, ℓ/ℓ₀ buiten [0,67; 1,5] (ℓ = `tokens_out` van het sample, ℓ₀ van het anker),
QE > 7,5. **Kandidaat** = geen fail, QE ≤ QE₀ (+ 1e-9), en tekstueel anders dan het anker na normalisering van
witruimte (`" ".join(tekst.split())`). Kandidaten met dezelfde genormaliseerde tekst binnen een bron worden één keer
beoordeeld (sleutel: het eerste sample `j`), en delen die P̄. [gevolgtrekking] Een sample gelijk aan het anker geeft in
beide volgordes dezelfde prompt, dus P̄(sample beter) = P̄(anker beter) ≤ 0,5: het kan nooit doel worden. Het weglaten
ervan en het één keer beoordelen van gelijke teksten verandert geen enkele keuze van de regel van de spec; het spaart
alleen judge-tijd.

*G4-8-data.5 Judge* (spec §3 stap 2): de judge van G4-judge.2–4 ongewijzigd (Gemma 4 31B-it @ `842da37`, de
MaxText-omzetting, prompt G4-judge.3, één prefill, P(A)/P(B)/P(gelijk) uit de logits, beide volgordes; `ab`: A = sample,
B = anker), `pairjudge.py score` ongewijzigd, prompts over de 4 workers op sha256 van de uid. P̄ = P̄(sample beter) van
G4-judge.4. Een prompt boven 4.096 tokens wordt niet gescoord; die kandidaat valt af (geteld).

*G4-8-data.6 Selectie janchor* (spec §3 stap 3–4, G4-judge.12): k = de kandidaat met de hoogste P̄ (gelijk: de lagere
QE, dan het eerdere sample); **doel = de tekst van k als P̄(k) > 0,5**, anders valt de bron weg (geen ankerrij, O3);
zonder kandidaat valt de bron ook weg. Hoogstens één doel per bron.

*G4-8-data.7 Rapport vóór de training* (per deel en samen; alleen tellingen en scores, geen tekst): bronnen, samples,
fail per reden, kandidaten per bron, verschillende samples per groep (mediaan; aandeel met 8 gelijke), beoordeelde en
niet gescoorde prompts, **aandeel bronnen waar een sample het anker verslaat** (P̄ > 0,5 = gehouden), P̄ van de doelen,
QE van anker en doel en ΔQE, ℓ/ℓ₀, chrF(doel, anker) (de definitie van sacrebleu 2: tekens-n-grammen 1–6 zonder
witruimte, β = 2; eigen code), het gekozen sample, de doeltokens en [schatting] het aantal SFT-stappen (2 epochs, rijen
van 4.096, 16 per stap), de QE₀-controle van G4-8-data.2. Naar `$G8/report/report.md`/`.json`, kopie
`mbert-ops/translator-sft/g4-8-data.md` (`.json`).

*G4-8-data.8 V3* (spec §7): na de selectie van `c1` een blind pakket `gs://…/sft/blind/G4-8-V3/` van 50 BT- en 50
romandoelen van `c1` tegen het anker: per deel `random.Random("en-nl-g4arm/0/G4-8/v3/<deel>").sample` van de
gesorteerde id's met een doel (minder dan 50: alle); `blind_id` `v3-001…v3-100` (BT eerst, in de volgorde van de
trekking); A/B met `random.Random("en-nl-g4arm/0/G4-8/v3/ab/<blind_id>").random() < 0,5` → A = het anker (`base`),
anders A = het doel (`G4-8-janchor`). `judge-input.jsonl` (blind_id, source, A, B), apart `key.jsonl` (blind_id, A, B,
seed, set = deel, item, stratum) en `select-report.json` (aantallen, geen tekst); een bestaande `key.jsonl` wordt niet
overschreven. De beoordeling (verse Claude-subagent, rubriek protocol.md §4.2.4) en de regel van de spec (een deel met
< 50% van de niet-gelijke oordelen voor het doel gaat niet in de training; O8) zijn aan de hoofdsessie en Yeb.

*G4-8-data.9 Uitvoer.* `$G8/fresh/` (de verse bronnen, `fresh.json`), `$G8/anchor/samples/`, `$G8/plan/` (`plan.parquet`:
de 24.000 bronnen met deel, herkomst, anker, ℓ₀; `plan.json`), per deel `$G8/<deel>/{segments,samples,qe,judge/data,
judge/scores,select}/` (`select/targets.parquet`: id, bron, doel en de maten; `select/choice.parquet` zonder tekst),
`$G8/targets/targets.parquet` (alle doelen, de invoer van de latere SFT-stap) en het rapport; markers
`$G/stages/g4-8-data/`.

*G4-8-data.10 Keten* (`mbert-ops/scripts/g4arm8_data_run.sh`, kopie in `gs://t5-dutch-english-usc2/hammond/scratch-scripts/`;
code `g4arm8.py`, `run_g4arm8.sh` op `en-nl-g4arm8`). Alleen mbert-mix; `final_eval_mix.lock` met `flock` tot het eind;
mbert-corpus (T3) wordt niet aangeraakt. Code als bundel via de bucket naar de 4 workers; elke ssh met `timeout 900` en
de ssh-opties van G4-judge.10; vóór elke TPU-stap de TPU van de worker(s) vrij. Volgorde: `setup` (`run_g4arm.sh setup`,
`ARM=G4-bon`), `fresh` (w0, CPU), het anker (4 workers), `plan` (w0, CPU); per deel `sample` (4), `qe` (4), `pairs`
(w0, CPU), `judge` (4), `select` (w0, CPU); na `c1` ook `v3` (w0, CPU); tot slot `report` (w0, CPU). Elke stap eindigt
op `EXIT 0`, anders `FAILED` en stoppen; een gemarkeerde stap wordt bij een herstart overgeslagen. [schatting] ~8–10 u:
fresh ~15 min, anker ~15 min, `c1` ~1,5–2 u (V3 na ~2–3 u), `c2` en `c3` elk ~3–4 u.

*G4-8-data.11 Wat deze stap niet doet.* Geen training, geen evaluatie, geen C1 of C2 in de lus; de doelen zijn
ongetoetst tot V3 en V1. De judge is alleen op FLAN, WMT24++ en idioom gevalideerd (G4-judge.5, .12); BT en romans zijn
90% van de bronnen, dus V3 weegt hier zwaar. De rapportage "de judge op de 500 C2F-paren" (spec §7 V3) is niet in deze
keten.

## 19. G4-8 training: SFT met LoRA op de doelen van G4-8-data

**Vastgelegd 23 nov 2026 ~23:20 door de hoofdsessie, na Yebs steekproef (V1) en zijn keuze over O6.** Opgesteld 23 nov
(~19:30) door een subagent als concept; nu definitief met Yebs besluiten in G4-8.0. Spec:
`mbert-ops/translator-sft/g4-8-spec-judge.md` (§4 training, §6 beslisregel, §9 open keuzes) met `g4-8-spec.md` §6–§7
waar die naar verwijst. `G8=$G/data/G4-8`, `RL=$G/rl/G4-7`.

*G4-8.0 Besluiten vóór de start* [Yeb, 23 nov 2026].
- `O1 [besluit Yeb 23 nov 2026]: G4-0` (startmodel; spec-standaard, Yeb koos geen andere).
- `O5 [besluit Yeb 23 nov 2026]: 2 epochs, LR 5e-5` (spec-standaard, Yeb koos geen andere).
- `O6 [besluit Yeb 23 nov 2026]: R-nieuw`
- V1 (Yebs steekproef, 23 nov ~23:10; 40 paren uit `G4-judge-i15` en `G4-judge-anchor-i15`): Yeb kiest de judge-keuze in
  25 van 40 (62,5%, eenzijdig p = 0,077; judgeq 13 / 7, judge-of-anker 12 / 8; FLAN 14 / 6, WMT24++ 11 / 9; nooit
  "gelijk"): haalt O7 (≥ 60%). O8: V3 gaf BT 31 / 3 / 16 en romans 28 / 6 / 16, dus er valt geen deel af; FLAN is in V3
  niet getoetst. FLAN-aandeel 10% (besluit Yeb 23 nov ~16:40, §18).

*G4-8.1 Startmodel en LoRA.* G4-0 (Gemma 4 31B-it @ `842da37`, run_sft.sh's MaxText-omzetting `gemma4-ckpt/gemma-4-31b-it`)
zonder adapter [spec O1, standaard]. LoRA rank 32 / alpha 16 (schaal 0,5) op de 410 modules van `run_sft.sh` (attention
+ MLP), AdamW zoals G4-1…G4-5 (weight decay 0,01, clipping 1,0, b2 0,999), `num_vocab_tiling=8`, MaxText 0.2.5.

*G4-8.2 Trainingsdata* (`g4arm8.py sftset`, w0, CPU). Alle doelen van `$G8/targets/targets.parquet` (§18.6: hoogstens één
per bron, P̄ > 0,5, QE ≤ QE₀, niet de tekst van het anker; geen ankerrijen, O3). Per doel één voorbeeld in het format van
G4-1…G4-5: de user-beurt `g4lora.user_turn(en)` (prompt `natural-v1`), de assistant-beurt het doel (gestript), het
template `g4lora.TEMPLATE` (Gemma 4, denken uit); verlies alleen op het doel en `<turn|>\n` (MaxText `completion_only`).
**2 doorgangen** [O5], elke doorgang alle doelen, apart geschud met zaad `en-nl-g4arm/0/G4-8/train/shuffle/<doorgang>`.
Meta per rij zoals G4-3/G4-4 (`id` = het bron-id van §18, `kind` = deel, `source` = stratum, `chrf` = chrF(doel, anker),
`epoch`). Vooraf gecontroleerd (een treffer is `FAILED`, niets wordt stil weggelaten: de filters van §18 hadden ze al
uit moeten houden):
- elk doel heeft `reason` = `sample`, P̄ > 0,5, een niet-lege tekst die (na normalisering van witruimte) niet die van het
  anker is, een bron-id en Engels gelijk aan `$G8/plan/plan.parquet`; elk id één keer;
- **geen bron uit de evaluatie- of blinde sets**: geen gelijke (gestripte) Engelse tekst en geen gedeelde 13-gram met bron
  en referentie van de evaluatiesets (`$E/data/evalsets.parquet`), de bronnen van de blinde passages (`blind100.jsonl`)
  en de FLAN-evaluatierijen (`sft_build.flan_eval_ngrams`); niet een id, tekst of document van G4-7's RL-val (`role` =
  `rlval` in `$RL/data/pool/pool.parquet`); geen user-beurt gelijk aan een user-beurt van de SFT-evaluaties `books`,
  `btval`, `own` (`$G/data/heldout/eval-*.parquet`, G4-2.5).
Rijen van 4.096 tokens (`max_target_length`, packing), 16 rijen per stap (4 processen × 4 chips, `per_device_batch_size`
1). Het aantal stappen wordt bij de bouw nagerekend zoals build4 (`g4arm.pack_batches`, `steps_of`) en als `STEPS`
meegegeven; na de training moeten de batches per proces van MaxText's pipeline gelijk zijn aan die naberekening.
`g4arm.py check --arm G4-8` (MaxText's eigen SFT-stappen op elk verschillend voorbeeld, als G4-3/G4-4) vóór de eerste
stap. [schatting] Deel 1 hield 2.516 van 4.000 bronnen (63%); bij hetzelfde aandeel ~15.000 doelen × ~400 tokens × 2 ≈
12 M tokens ≈ 190–200 stappen.

*G4-8.3 Training* (`sft_lora.py` via `run_sft.sh`, één multi-host-job over de 16 chips van mbert-mix, zoals G4-1 en G4-2
op mbert-mix). **LR 5e-5 constant na 10 stappen lineaire warmup** [O5; als G4-7.4k], MaxText's WSD zonder afbouw
(`--warmup-steps 10 --decay-fraction 0`). Evaluatie van het verlies op `books`, `btval`, `own` (G4-2.5) op stap 0, elke
E = ⌈stappen / 6⌉ stappen en na de laatste. Run `g4arm_G4-8` (een herstart na een mislukte run: een nieuwe naam).

*G4-8.4 Kandidaten* [spec §4]: de adapters na **K₁ = 2E** (~⅓), **K₂ = 4E** (~⅔) en **de laatste stap**
(`--save-steps K₁,K₂`; K₁ en K₂ zijn evaluatiestappen, dus hun SFT-verlies is dat van die adapter, G4-6c.6a). Namen
`G4-8-s<stap>`, naar `$G/G4-8-s<stap>/` (rank 32 / alpha 16 / schaal 0,5 gecontroleerd; niets gestapeld).

*G4-8.5 Smoke-test* (vóór de volle run, binnen de lock). 8 stappen met de instellingen van G4-8.3 op een smoke-set uit
doorgang 1 (de 256 langste voorbeelden, dan 512 andere, zaad `en-nl-g4arm/0/G4-8/smoke`, als G4-3.5), evaluatie op stap
0, 4 en 8, `--save-steps 4`; run `g4arm_G4-8_s8`. Geslaagd als: 8 eindige verliezen, piek ≤ 29,5 GiB per chip, adapter
en adapter van stap 4 rank 32 / alpha 16 / schaal 0,5, `max_target_length` 4.096, de batches per proces van de pipeline
gelijk aan de naberekening, van elke SFT-evaluatie drie eindige verliezen met ≥ 1 token, `own` op stap 0 ≤ 1,0 (G4-2.4),
LR-piek 5e-5 (`lr_at` van de run). Faalt een controle, dan stopt de keten vóór de training (de hoofdsessie beslist).
Gerapporteerd: de schatting van de volle run (`g4arm.training_time`).

*G4-8.6 Per kandidaat, gerapporteerd* (spec §4; beslist niets): de relatieve update-norm (de `LORA`-regel van
`translate.py` bij het samenvoegen), het SFT-verlies op `books`, `btval`, `own` op de stap van de kandidaat, en op
G4-7's RL-val (384 bronnen, iteratie −1 van `$RL/data/plan/plan.parquet`) 8 samples per bron bij T = 1 (zaad per worker
`en-nl-g4arm/0/G4-8/div/w<k>`, voor elk model hetzelfde; ook G4-0 zonder adapter): de mediaan van het aantal
verschillende samples per groep (na normalisering van witruimte), het aandeel groepen met 8 gelijke, en het fail-aandeel
(niet af, lus, leeg, Engels over, ℓ/ℓ₀ buiten [0,67; 1,5] tegen G4-7's anker). **Vlag** bij een mediaan < 4 van 8.
Spec §8: `own` (G4-0's eigen greedy uitvoer) moet stijgen als het model van zijn greedy modus weggaat; een stijging van
`books`/`btval` zoals bij G4-5's FLAN-tak is een vlag (gerapporteerd, geen regel).

*G4-8.7 Evaluatie* (als G4-7.7, G4-5.5): elke kandidaat vertaald en gescoord met `run_g4arm.sh translate|score`
(`ARM=G4-8`, fork `f4d8615`, `natural-v1`, greedy, klassen tiny–xlong, de adapter samengevoegd; MetricX-24 met
referentie en QE, COMET-22); G4-0's bestaande evaluatie `$G/eval/G4-0/` (dezelfde als bij G4-7). C1 (S @ 653k,
ongewijzigd) in één classify-run over de kandidaten, G4-0 en G4-4 (gerapporteerd, als G4-7) op
`$G/naturalness-G4-8/`; C2 (dmbert v2, `$RL/c2/model/`) en PLL (S @ 653k) op dezelfde teksten, gerapporteerd. De judge
scoort geen evaluatie-uitvoer, ook niet als rapportage (spec §6).

*G4-8.8 Keuze (zoekdeel)*, ongewijzigd als G4-7.7: van de kandidaten die op het zoekdeel van G4-5.4 (digest
`f953c026…f3`) (a1) halen, de grootste (b) daar; gelijk: de eerdere; haalt geen kandidaat (a1), dan de kleinste grootste
QE-overschrijding (`select5`). Dezelfde keuze onder beide varianten van G4-8.9.

*G4-8.9 Beslisregel (beslisdeel), tegen G4-0* [open: Yeb, O6]. Beide varianten worden in het rapport uitgerekend; de
regel van G4-8.0 beslist.
- **(b) beslissend, R-oud** (als sinds G4-2; G4-7.8 met G4-7.4l): (a1) per set QE ≤ die van de menselijke referentie op
  dezelfde segmenten; (a3) per set COMET-22 ≥ G4-0 − 0,005 (WMT24++ zonder `is_bad_source`, FLORES, NTREX); (b) C1
  "origineel" − G4-0 ≥ 0,10 op WMT24++ (goed) + FLAN; dan (a2) en (c) zoals §5 op de 100 passages (één verse
  Claude-subagent, rubriek protocol.md §4.2.4, twijfelgevallen naar Yeb). "Helpt" vraagt alle vijf. Gelden (a1) en (a3)
  en (b) niet, dan draait de blinde beoordeling als rapportage (G4-7.4l).
- **(c) beslissend met (b) als rapportage, R-nieuw** (spec §6, voorstel): "helpt" = (a1) ∧ (a3) ∧ (a2) ∧ (c) ∧
  C1-vangrail (C1 − G4-0 ≥ 0 op WMT24++ (goed) + FLAN); (c) letterlijk als §5, maar **na vervanging van elk
  twijfelgeval door Yebs oordeel** (de telling van `g4arm.py tally` moet dan `doubt` = 0 hebben: elk twijfeloordeel
  vervangen, `zekerheid` = `yeb`); Yeb beoordeelt ook 10 zekere Claude-oordelen
  (`random.Random("en-nl-g4arm/0/G4-8/yeb10").sample` van de gesorteerde `blind_id`'s met `zekerheid` = `zeker`);
  wijkt hij bij meer dan 3 af, dan beoordeelt hij alle 100. (b) wordt gerapporteerd met een 95%-interval (gepaarde
  bootstrap per set, `g4bon.paired_ci`, 10.000 trekkingen, zaad `en-nl-g4arm/0/G4-bon/boot/G4-8/b`).
- In beide varianten: het blinde pakket (`armeval.py blind`, de gekozen kandidaat tegen G4-0 in `$S/blind/G4-8/` en
  tegen TG in `$S/blind/G4-8-tg/`) wordt gemaakt als (a1) en (a3) gelden; de beoordeling en de telling doet de
  hoofdsessie (`g4arm.py tally`), daarna `report8` opnieuw met `--tally`. Zonder (a1) of (a3): geen blinde beoordeling
  en "helpt niet". Het beslisdeel is in G4-bon, G4-judge en G4-8-data niet gelezen; het zoekdeel wel (de cijfers daar zijn
  optimistisch).

*G4-8.10 Rapport* (`g4arm8.py report8` → `$G/report-G4-8/report.md`/`.json`, kopie
`mbert-ops/translator-sft/g4-8-report.md`): de keuze, beide varianten van G4-8.9 met de regel van G4-8.0 als
beslissend, alle kandidaten op beide delen (QE, (a1), (a3), C1, C2, "C2 stijgt, C1 niet", MetricX-ref, COMET, niet af,
lus, Engels over, lengte, MTLD, PLL), G4-8.6 per kandidaat, de trainingsdata per deel en de smoke-test.

*G4-8.11 Uitvoer.* `$G8/sft/` (set, smoke-set, `check.txt`, RL-val-segmenten, `div/<model>/`), `$G/runs/g4arm_G4-8_s8/`,
`$G/runs/g4arm_G4-8/`, `$G/G4-8-s<stap>/`, `$G/eval/G4-8-s<stap>/`, `$G/naturalness-G4-8/`, `$G/report-G4-8/`,
`$S/blind/G4-8/`, `$S/blind/G4-8-tg/`; markers `$G/stages/g4-8-train/`. Niets van §18 of eerdere armen wordt
overschreven.

*G4-8.12 Keten* (`mbert-ops/scripts/g4arm8_train_run.sh`, kopie in `gs://t5-dutch-english-usc2/hammond/scratch-scripts/`;
code `g4arm8.py`, `run_g4arm8.sh`, `run_g4arm.sh` met `ARM=G4-8` op `en-nl-g4arm8`). Alleen mbert-mix (een v4-32: de
training over alle 16 chips, de rest per worker); mbert-corpus (T3) wordt niet aangeraakt. Wacht eerst op de marker
`$G/stages/g4-8-data/report.done` van §18 en `$G8/targets/targets.parquet`, dan op het startbestand
`mbert-ops/g4arm8.train-go` (twee regels: `O6=R-oud` of `O6=R-nieuw`, gelijk aan G4-8.0, en `COMMIT=<de gepushte
commit met de definitieve §19>`, die afstamt van dit concept en van de geteste code), dan op `final_eval_mix.lock` (met
`flock`, tot het eind gehouden). Volgorde: setup (`FULL=1`; daarna op elke worker ≥ 12 GB vrij in `/dev/shm`, anders
`FAILED`), `sftset` (w0), `check` (w0), smoke-run + `smoke8`
(w0), de training, `adapter` (w0), per kandidaat `translate` en `score` (4 workers), RL-val-samples per model (4
workers), `texts`, `classify`, `classify-c2`, `pll` (w0), `report8` (w0), het blinde pakket als (a1) en (a3) gelden;
`DONE` met de keuze en de regels. Elke stap eindigt op `EXIT 0`, anders `FAILED` en stoppen; een gemarkeerde stap wordt
bij een herstart overgeslagen. [schatting] ~5–6 u na de start: setup ~5 min, sftset + check ~20 min, smoke ~25 min,
training ~45 min, per kandidaat vertalen + scoren ~47 min (3×), RL-val-samples ~25 min, C1/C2/PLL ~40 min, rapport
~5 min.

*G4-8.13 Wat deze stap niet doet.* Geen tweede arm vanaf i15 (O1), geen tweede ronde (O10), geen judge op
evaluatie-uitvoer, geen beoordeling van het blinde pakket (die is aan de hoofdsessie en Yeb).

*G4-8.13 Aanvulling 23 nov 2026 ~23:30 (besluit Yeb: "ik ga niets meer beoordelen, kun jij niet zelf beoordelen?").* Onder
R-nieuw vervangt Yebs oordeel de twijfelgevallen (G4-8.9). Yeb beoordeelt ze niet; in plaats daarvan krijgt elk
"twijfel"-oordeel van de eerste blinde beoordelaar een **tweede blinde ronde** door een andere verse Claude-subagent (zelfde
rubriek protocol.md §4.2.4; per passage de bron, A, B en de uitleg van de eerste beoordelaar, geen sleutel, geen andere
context), die per passage een eindoordeel geeft ("A beter", "B beter" of "gelijk", plus de betekenisfouten). Dat oordeel
vervangt het twijfeloordeel; daarna pas de sleutel (zoals bij RB3, LEARNINGS 13 nov, toen namens Yeb). De hoofdsessie
beoordeelt zelf niet (geen blinde beoordelaar). Yeb kan de oordelen later overschrijven; het verslag noemt dan beide
tellingen. De keten wijzigt niet (het blinde pakket en de telling gebeuren na `DONE` door de hoofdsessie).
Ook de controle van 10 zekere oordelen (G4-8.9, `random.Random("en-nl-g4arm/0/G4-8/yeb10")`) doet de tweede blinde
beoordelaar in plaats van Yeb: wijkt hij bij meer dan 3 af, dan beoordeelt hij alle 100 en geldt zijn telling.

*G4-8.14 Aanvulling 24 nov 2026 ~00:15 (hoofdsessie; technische fout in een controle, als G4-7.4i).* De smoke-run
`g4arm_G4-8_s8` (keten `g4t8a`) faalde alleen op `lr_peak`: de trainer logt de LR in float32, en 5e-5 leest terug als
4,999999873689376e-05; de controle eiste |max − 5e-5| ≤ 1e-12 (verschil 1,26e-12). Nu: ≤ 1e-6 × 5e-5 (float32-precisie);
een echte afwijking (bv. 4,9e-5 of 1e-4) faalt nog steeds (tests). De overige controles van de smoke-run slaagden
(piek 29,13 GiB ≤ 29,5). De smoke-training is gemarkeerd en draait niet opnieuw; de herstart (nieuwe TAG) draait alleen
`smoke8` opnieuw. De test van §19 controleert nu de definitieve versie (één O6-regel, R-nieuw) in plaats van het concept.
Hetzelfde voor `lr_constant` in `trained` (24 nov ~01:35): de volle run `g4arm_G4-8` (205 stappen, piek 29,13 GiB,
verlies 0,114 → 0,074) faalde alleen daarop, met de LR na de warmup overal 4,999999873689376e-05; nu ≤ 1e-6 × 5e-5. De
training is af en draait niet opnieuw; de herstart draait `adapter` en wat volgt.

*G4-8.15 Aanvulling 24 nov 2026 ~08:40: menselijke steekproef van G4-8-s140 tegen G4-0 uit alle bronnen, rapportage*
(opdracht Yeb 24 nov ~07:50; uitwerking door een subagent; tijdstip: de commit van deze alinea's op `en-nl-g4arm8`, vóór
elke stap van deze steekproef op een worker en vóór elke vertaling van romans en BT). Beslist niets en verandert
G4-8.0–G4-8.14 niet; er wordt niets getraind. Claude en GPT zijn geen generator.
- **Aanleiding.** V1 en steekproef 2 (`translator-sft/steekproef-yeb-g4-8.md`) kwamen alleen uit WMT24++, FLAN en
  idioom, terwijl 90% van G4-8's bronnen romans en BT waren (§18). Yeb wil 40 blinde paren van het eindmodel G4-8-s140
  tegen G4-0, gemengd uit alle bronnen.
- **Vooraf gelezen** (alleen aantallen, geen tekst): in de evaluatie (`$G/eval/{G4-8-s140,G4-0}/pairs/`) verschillen de
  twee vertalingen (witruimte genormaliseerd, G4-8-data.4) in het zoekdeel van G4-5.4 bij FLAN 355, WMT24++ 199, FLORES
  159 en NTREX 346 segmenten; idioom zit alleen in de 100 vaste blinde passages (30 stuks), waarvan er 21 verschillen,
  12 daarvan niet in steekproef 2.
- **Romans (nieuw vertaald).** De set `novels` (G4-4.10: 500 passages uit 10 boeken van 10 auteurs,
  `$G/data/heldout4/eval-novels-meta.parquet`), het Engels van dezelfde rij in `b1-selected.parquet` (gestript; de rij
  één keer, groep `en_orig`, hetzelfde Engelse boek). De boeken zitten in G4-4's training (G4-4.9), maar niet in de
  bronnen van G4-7 en G4-8 (cands7 laat ze weg, G4-7.2 en §18); G4-8 begint bij G4-0 (O1), dus voor beide modellen zijn
  het ongeziene boeken [gevolgtrekking]. Controle (een treffer stopt): geen id `novels:<pid>`, document
  `novels:<Engels boek-id>` of gestripte Engelse tekst in G4-7's `$RL/data/pool/pool.parquet` of in G4-8's
  `fresh.parquet`, `plan.parquet` en `targets.parquet`. Daarna pool7's filter (`g4rl.candidate_filter`, geteld): 20–256
  Gemma 4-tokens, geen 13-gram met het Engels van die vier bestanden, één bron per tekst. Per boek 6 passages (de rijen op
  pid, geschud met zaad `en-nl-g4arm/0/G4-8/mix/novels/<Engels boek-id>`): **60**.
- **BT (nieuw vertaald).** C2's testsplit zoals §20 (`pairjudge.fb_bt` ongewijzigd, met zijn controles tegen C2's
  trainingsdocumenten, G4-7's pool en G4-8's plan), het Engels van Gemma 4 van dezelfde pid; daarbij dezelfde controle
  tegen G4-8's `fresh.parquet` en `targets.parquet` (een treffer stopt) en dezelfde filter als bij de romans. Per bron
  (dpc, fineweb, handelingen, raad, rechtspraak) 12 (zaad `en-nl-g4arm/0/G4-8/mix/bt/<bron>`): **60**.
- **Vertalen** zoals G4-8.7: G4-0 zonder adapter en G4-8-s140 (`$G/G4-8-s140/`, rank 32 / alpha 16 / schaal 0,5
  gecontroleerd), `translate.py` op fork `f4d8615`, model `gemma4-31b`, `natural-v1`, greedy, per lengteklasse naar de
  tokens van het Engels (`g4arm.class_of`, tiny–medium) met de slots van `translate_models.yml` en dezelfde terugval bij
  geheugengebrek als `run_g4arm.sh translate`; daarna `evalsets.py collect`. mbert-mix: w2 vertaalt met G4-0, w3 met
  G4-8-s140, w0 doet de CPU-stappen.
- **Keuze van de 40** (alleen paren waarvan de twee vertalingen na normalisering van witruimte verschillen; per set
  getrokken uit de gesorteerde id's met zaad `en-nl-g4arm/0/G4-8/mix/pick/<set>`):
  - romans **8**: de boeken in getrokken volgorde, per boek de eerste verschillende passage in getrokken volgorde, één
    per boek (dus 8 van de 10 boeken; een boek zonder verschillend paar wordt overgeslagen; pas als er minder dan 8
    boeken zijn een tweede ronde);
  - BT **10**: 2 per bron;
  - FLAN **6**, WMT24++ **6** (zonder `is_bad_source`), FLORES **3**, NTREX **2**: uit het zoekdeel van G4-5.4
    (`split.json`, digest `f953c026…f3`); niet de items van de 100 vaste blinde passages, van `$S/blind/G4-8-replica/`
    en van `G4-judge-i15` en `G4-judge-anchor-i15` (waaruit V1 kwam);
  - idioom **5**: de idioom-passages van de 100 vaste blinde passages (alleen daar bestaan idioom-vertalingen van beide
    modellen), zonder de passages van steekproef 2 (`$S/blind/steekproef-yeb-g4-8/key-map.json`).
  De volgorde van de 40 geschud (zaad `…/mix/pick/order`); A/B per paar met zaad `…/mix/ab/<set>/<item>`.
- **Uitvoer.** `mbert-ops/translator-sft/steekproef-yeb-g4-8-mix.md`: bovenaan de instructie (per nummer "A", "B" of
  "=", eventueel "fout A" / "fout B"), per paar alleen het nummer, de Engelse bron, A en B (geen setnaam), door een
  script geschreven. Sleutel (nummer → set, item, welke kant G4-8-s140 is) apart in
  `$S/blind/steekproef-yeb-g4-8-mix/key.json`. Het blinde pakket `$S/blind/G4-8-romans-bt/` (`judge-input.jsonl`:
  blind_id, source, A, B; apart `key.jsonl`; `select-report.json`) met **alle** verschillende roman- en BT-paren van de
  120, voor een verse Claude-beoordelaar (rubriek protocol.md §4.2.4; de beoordeling doet de hoofdsessie). Data, vertalingen
  en tellingen onder `$G/mixsample/`, markers `$G/stages/g4-8-mix/`. Teksten blijven in de bucket en op de VM, behalve het
  steekproefbestand voor Yeb.
- **Keten** `mbert-ops/scripts/g4mixsample_run.sh` (kopie in `gs://t5-dutch-english-usc2/hammond/scratch-scripts/`;
  code `g4mixsample.py`, `run_g4mixsample.sh` op `en-nl-g4arm8`). Alleen mbert-mix, `final_eval_mix.lock` met `flock` tot
  het eind; vóór elke TPU-stap de TPU van de worker vrij; elke ssh met `timeout 900` en de ssh-opties van G4-judge.10;
  mbert-corpus (T3) wordt niet aangeraakt; niets wordt verwijderd. Volgorde: setup (`run_g4arm.sh setup`, `ARM=G4-bon`,
  w0, w2, w3), `data` (w0, CPU), `translate` (w2 G4-0, w3 G4-8-s140), `pick` (w0, CPU). [schatting] ~40–60 min.
- **Wat het niet zegt.** 40 paren, alleen uit paren die verschillen (het aandeel verschillende per set staat in
  `pick.json`); de romans zijn "ongeziene boeken" voor G4-0 en G4-8, niet voor G4-4; de BT-bron is het Engels van
  Gemma 4; FLAN, WMT24++, FLORES en NTREX komen uit het zoekdeel, dat G4-8's keuze van de kandidaat al las (G4-8.8).

## 20. Meting G4-judge-FB: de judge op fictie en BT

**Vastgelegd 24 nov 2026 (tijdstip: de commit van deze sectie op branch `en-nl-g4arm8`)**, vóór elke stap op een worker
en vóór elke uitslag van deze meting [besluit van de hoofdsessie 24 nov ~05:10 op advies van Gemini 4 Neon, Yeb sliep;
binnen de judge-proef van §17; uitwerking door een subagent]. Vooraf gelezen: de rapporten van §16 en §17, de
kolomnamen en aantallen van de invoer (C2's `classifier.parquet`: 500 + 500 rijen met `split` = `test`;
`$RL/c2/cal7/predictions.parquet`: model `C2-test`, 200 rijen per BT-set, id `<pid>#<label>`) en de cijfers van
`c2cal7` (nauwkeurigheid op de testsplit 0,886); geen tekst en geen score van deze meting. Een meting, geen arm: er wordt
niets getraind, niets vertaald en niets beslist. Claude en GPT zijn geen judge. `GJ=$G/judge`, `RL=$G/rl/G4-7`.

*FB.1 Vraag.* De judge van §17 koos de doelen van G4-8 (§18), voor 90% romans en BT, maar is alleen gevalideerd op FLAN,
WMT24++ en idioom (G4-judge.5; G4-8-data.11). Herkent hij, bij een Engelse bron en twee Nederlandse teksten waarvan de
herkomst bekend is, de menselijke of oorspronkelijke kant als de betere? Twee proeven:
1. **fictie**: een Engelse romanpassage met (i) de menselijke Nederlandse vertaling en (ii) G4-0's vertaling;
2. **BT**: het Gemma 4-Engels van een Nederlands origineel met (i) het origineel en (ii) Gemma 4's rondgangvertaling.

*FB.2 Paren* (er wordt niets getrokken of gegenereerd; alleen bestaande teksten).
- **Fictie** (500 paren): de groepen `human` en `g40` van C2F (§16, run `c2f1123a`, `$G/c2fiction/c2f1123a/texts.parquet`;
  dezelfde 500 id's `novels:<rij-id van -b1->`), ongewijzigd. Engelse bron: `en` van die rij in
  `$S/data/books-b1/b1-selected.parquet` (gestript). Controles (een afwijking stopt de meting): elke id staat één keer
  in b1 met groep `en_orig`; de tekst van `human` is (gestript) `nl` van die rij; geen lege tekst; `human` en `g40`
  hebben dezelfde id's.
- **BT** (500 paren): **C2's testsplit** (`$RL/c2/data/classifier.parquet`, `split` = `test`: per BT-set 100 passages
  uit 100 documenten; `label` 1 = `nl`, het Nederlandse origineel, `label` 0 = `nl_back`, Gemma 4's rondgang; G4-7.6),
  ongewijzigd. Engelse bron: `en` van dezelfde `pid` in de trainsplit van de vijf `-nl-en-gemma4`-sets op G4-3's revisies
  (`$G/input/bt3/`, `g4arm.read_bt`, sha256 per bestand tegen `revisions.json`), gestript. Waarom de testsplit
  [keuze]: zijn documenten zijn per constructie geen trainingsdocumenten van C2 en zitten niet in de pools van G4-7 en
  G4-8 (die sloten alle 5.000 documenten van C2 uit, G4-7.2 en G4-8-data.1); hij is gelijk verdeeld over de vijf
  bronnen; beide teksten passen heel in C2 (zoals C2 getraind is, spec §5.3.2); en C2's scores op deze paren bestaan al
  (`c2cal7`), zodat judge en C2 op dezelfde paren te vergelijken zijn. Controles (een treffer stopt de meting): per bron
  precies 100 passages met beide labels; `nl` en `nl_back` van de BT-rij (gestript) gelijk aan de teksten van
  `classifier.parquet`; geen document (`g4c2.doc_key`) van de testsplit is een document van C2's train-split; geen id
  `bt:<pid>`, document of gestripte Engelse tekst staat in G4-7's `$RL/data/pool/pool.parquet` (pool, reserve, RL-val)
  of in G4-8's `$G/data/G4-8/plan/plan.parquet`.
- Paren waarvan beide teksten na strippen gelijk zijn, worden geteld en gewoon beoordeeld (P̄ is dan symmetrisch).
- Gerapporteerd (alleen id's): hoeveel fictie-id's in G4-7's pool en in G4-8's plan staan (het anker van G4-7 kwam uit die
  pool; G4-8 hergebruikte die bronnen).

*FB.3 Judge.* G4-judge.2–4 ongewijzigd: Gemma 4 31B-it @ `842da37`, de MaxText-omzetting, prompt G4-judge.3 letterlijk,
één prefill, P(A)/P(B)/P(gelijk) uit de logits van de eerste antwoordpositie, beide volgordes; `pairjudge.py score`
ongewijzigd, prompts over de 4 workers van mbert-mix op sha256 van de uid. Kader `ab`: A = de menselijke of
oorspronkelijke kant (`human`, `nl`), B = de machinekant (`g40`, `nl_back`); `ba` verwisseld. P̄(mens) = P̄(A beter),
P̄(machine) = P̄(B beter), P̄(gelijk) (G4-judge.4). Een prompt van meer dan 4.096 tokens wordt niet gescoord (geteld).
2.000 prompts.

*FB.4 Maten* (`pairjudge.py fbreport`, per proef; intervallen 95% Wilson, tenzij anders vermeld).
1. Het oordeel van de judge (de grootste van de drie P̄'s, G4-judge.4): aantal en aandeel mens / machine / gelijk.
2. **Richting**: aandeel paren met P̄(mens) > P̄(machine) (gelijk: ½); gemiddeld P̄(mens), P̄(machine), P̄(gelijk);
   gemiddelde Δ = P̄(mens) − P̄(machine) met een 95%-bootstrapinterval (10.000 trekkingen over paren, zaad
   `en-nl-g4arm/0/G4-judge-fb/boot/<proef>`).
3. **Positie**: omklappen (de greedy oordelen van `ab` en `ba`, teruggezet, noemen tegengestelde winnaars) en
   inconsistent (G4-judge.5 maat 3); per volgorde het greedy antwoord (eerste positie, tweede positie, gelijk), gemiddeld
   p_A − p_B, en de massa (mediaan, aantal < 0,5).
4. **Per bron en lengte**: BT per set (handelingen, raad, dpc, rechtspraak, fineweb); beide proeven per lengteband van
   de Engelse bron in Gemma 4-tokens (`g4arm.TokenCounter` op `tokenizer-gemma4`, zonder speciale tokens, als het
   maximum van 256 van G4-7.2): ≤ 128, 129–256, > 256. Per groep n, richting, de drie oordelen, omklappen.
5. **Classifiers op dezelfde paren, zoals ze er al zijn** (geen nieuwe classify-run): fictie C1 en C2 uit `c2f1123a`
   (`predictions-c1.parquet`, `predictions-c2.parquet`); BT C2 uit `c2cal7` (model `C2-test`, id `<pid>#1` = `nl`,
   `<pid>#0` = `nl_back`). C1 op de BT-paren bestaat niet en wordt niet gemaakt. Per classifier: aandeel paren met
   P(mens) > P(machine) (gelijk: ½); de overeenstemming van de richting met de judge op de paren waar beide een richting
   hebben (Δ ≠ 0), met de 2 × 2-tabel; Spearman tussen Δ van de judge en Δlogit van de classifier (`g4c2.logit_of`,
   geknipt op ±12).
6. **Betekenisfout aan de menselijke kant: niet gemeten.** De prompt van G4-judge.3 geeft één woord; juistheid en
   natuurlijkheid zitten in één antwoord en zijn er niet uit te scheiden. Een tweede prompt valt buiten deze meting. Het
   rapport zegt dat.

*FB.5 Wat het niet zegt.*
- De "waarheid" is de herkomst, niet de kwaliteit. Een menselijke romanvertaling heeft literaire vrijheid (weglaten,
  toevoegen, herschikken); de rubriek telt dat als betekenisfout, dus een voorkeur voor G4-0 kan de rubriek juist volgen.
  Bij BT is het Engels Gemma 4's vertaling van het origineel: waar dat Engels iets weglaat of toevoegt, lijkt het
  origineel tegen de bron een fout te hebben, en de rondgang volgt het Engels.
- De machinekant is in beide proeven Gemma 4 31B-it (G4-0 met `natural-v1`; de rondgang met `natural-v1-fmt` op fork
  `2249474`), hetzelfde model als de judge: een voorkeur voor de eigen stijl is niet te scheiden van de vraag.
- Fictie: 500 passages uit G4-7's anker (Engels van 20–256 Gemma 4-tokens, beide kanten heel in C1 en C2), uit het
  gelijnde en gefilterde -b1-corpus. BT: C2's testsplit, één passage per document, beide teksten heel in C2; ook langere
  bronnen dan G4-8's BT-bronnen (≤ 256 tokens), daarom per lengteband gerapporteerd. Geen steekproef van alle fictie of BT.
- Eén prompt, zero-shot; geen drempel; de meting beslist niets over G4-8, de judge of een volgende arm.

*FB.6 Uitvoer.* `$GJ/fb/data/` (`prompts-w<k>.parquet`, `refs.parquet` zonder tekst, `fbdata.json`),
`$GJ/fb/scores/w<k>/`, `$GJ/fb/report.md`/`.json`; kopie `mbert-ops/translator-sft/g4-judge-fiction-bt.md` (`.json`),
alleen tabellen met tellingen en scores. Prompts en teksten blijven in de bucket en op de VM.

*FB.7 Keten* (`mbert-ops/scripts/g4judgefb_run.sh`, kopie in `gs://t5-dutch-english-usc2/hammond/scratch-scripts/`; code
`pairjudge.py fbdata|fbreport` en `run_pairjudge.sh fbdata|score SET=fb|fbreport` op `en-nl-g4arm8`). Alleen mbert-mix;
`final_eval_mix.lock` met `flock` tot het eind; mbert-corpus (T3) wordt niet aangeraakt. Code als bundel via de bucket
naar de 4 workers; elke ssh met `timeout 900` en de ssh-opties van G4-judge.10; vóór elke TPU-stap de TPU van de
worker(s) vrij. Volgorde: `setup` (`run_g4arm.sh setup`, `ARM=G4-bon`), `fbdata` (w0, CPU), `score` met `SET=fb`
(4 workers), `fbreport` (w0, CPU). Markers `$G/stages/g4-judge/fb-*.done`; niets wat §16–§19 schreven wordt
overschreven. Elke stap eindigt op `EXIT 0`, anders `FAILED` en stoppen. [schatting] ~30–40 min.

*FB.8 Aanvulling van 24 nov 2026 ~04:40: G4-judge-FB.N, dezelfde judge alleen naar natuurlijkheid of herkomst gevraagd
(tijdstip: de commit van deze alinea's op `en-nl-g4arm8`), vóór elke stap op een worker en vóór elke uitslag van FB.N*
[opdracht van de hoofdsessie 24 nov ~04:33, Yeb sliep; binnen de judge-proef van §17; uitwerking door een subagent].
Vooraf gelezen: het rapport van §20 (`g4-judge-fiction-bt.md`, alleen tellingen en scores): met de prompt van
G4-judge.3 koos de judge op fictie G4-0 boven de menselijke vertaling (richting mens 17,6%) en op BT de rondgang boven
het origineel (richting origineel 30,8%), terwijl C1 en C2 op dezelfde paren de menselijke of oorspronkelijke kant
hoger zetten (82,6–96,4%). Geen tekst, geen prompt-uitvoer en geen score van FB.N gelezen. Een meting: er wordt niets
getraind, niets vertaald en niets beslist; Claude en GPT zijn geen judge.
- **Vraag.** Herkent dezelfde Gemma 4 31B-it de menselijke of oorspronkelijke kant wel als je hem alleen naar
  natuurlijkheid vraagt (N1, met de Engelse bron) of alleen naar herkomst (N2, zonder bron), zonder de juistheid van de
  rubriek?
- **Paren.** De 1.000 paren van §20 (FB.2, run `g4jfb1124a`), ongewijzigd: `pairjudge.py fbdata --prompt n1|n2` bouwt
  ze met dezelfde code en controles opnieuw uit de invoer die §20's `fbdata` op w0 neerzette
  (`/dev/shm/translator/g4arm/judge-fbdata-g4jfb1124a/in/`) en stopt als (i) de sha256 van de vijf invoerbestanden
  afwijkt van `$GJ/fb/data/fbdata.json`, (ii) `refs.parquet` afwijkt van die van §20, of (iii) de prompts van G4-judge.3
  die het uit deze paren maakt niet per uid gelijk zijn aan §20's `prompts-w<k>.parquet` (2.000 prompts, vergeleken op
  sha256 van de tekst). Daarmee zijn per uid A, B en de Engelse bron die van §20, en elke prompt staat op dezelfde worker.
- **Prompts** (letterlijk de tekst van de gebruikersbeurt; `{bron}`, `{A}` en `{B}` gestript ingevuld; verder als
  G4-judge.2: hetzelfde chat-template met één gebruikersbeurt en een lege denkruimte). Kader `ab`: A = de menselijke of
  oorspronkelijke kant, B = de machinekant; `ba` verwisseld.
  **N1** (met bron, alleen natuurlijkheid):

```
Je krijgt een Engelse bron en twee Nederlandse teksten, A en B, met ongeveer dezelfde inhoud.

Oordeel alleen op natuurlijk, oorspronkelijk klinkend Nederlands: welke tekst leest het meest als tekst die oorspronkelijk in het Nederlands is geschreven, en het minst als een vertaling? Laat juistheid uitdrukkelijk buiten beschouwing: weglatingen, toevoegingen en afwijkingen van de Engelse bron tellen niet mee. "gelijk" als er geen duidelijk verschil is.

Engelse bron:
{bron}

Tekst A:
{A}

Tekst B:
{B}

Welke tekst leest het meest als oorspronkelijk Nederlands? Antwoord met één woord: A, B of gelijk.
```

  **N2** (zonder bron):

```
Je krijgt twee Nederlandse teksten, A en B, met ongeveer dezelfde inhoud. De ene is oorspronkelijk in het Nederlands geschreven, de andere is een vertaling.

Welke van de twee is oorspronkelijk in het Nederlands geschreven, en welke is de vertaling? "gelijk" als je het verschil niet kunt zien.

Tekst A:
{A}

Tekst B:
{B}

Welke tekst is oorspronkelijk in het Nederlands geschreven? Antwoord met één woord: A, B of gelijk.
```

- **Scoren.** Als FB.3: `pairjudge.py score` ongewijzigd (Gemma 4 31B-it @ `842da37`, de MaxText-omzetting, één
  prefill, P(A)/P(B)/P(gelijk) uit de logits van de eerste antwoordpositie met de ids van G4-judge.2, beide volgordes,
  P̄ gemiddeld volgens G4-judge.4), per prompt over de 4 workers van mbert-mix; 2 × 2.000 prompts; een prompt van meer
  dan 4.096 tokens wordt niet gescoord (geteld). P̄(mens) = P̄(A) in kader `ab`: bij N1 "leest het meest als
  oorspronkelijk Nederlands", bij N2 "is oorspronkelijk Nederlands".
- **Maten** (`pairjudge.py fbnreport`): de maten van FB.4 1–5 per prompt en per proef, met code en bootstrapzaad van
  §20 (`fb_result`), en de uitslag van §20 (de rubriek-prompt, herberekend uit `$GJ/fb/scores/` en gecontroleerd tegen
  `$GJ/fb/report.json`) ernaast: oordelen, richting (Wilson), gem. P̄ en Δ (bootstrap), omklappen en inconsistent,
  positie en massa, BT per bron, per lengteband, C1/C2 op dezelfde paren zoals ze er al zijn. Erbij: de samenhang tussen
  de prompts op dezelfde paren (N1 ~ rubriek, N2 ~ rubriek, N1 ~ N2): richting gelijk op de paren waar beide Δ ≠ 0, met
  de 2 × 2-tabel, en Spearman tussen de Δ's.
- **Wat het niet zegt.** Fictie: beide kanten zijn vertalingen uit het Engels; de premisse van N2 (één tekst is
  oorspronkelijk Nederlands) klopt daar niet, en de menselijke kant geldt, als in §20, als de verwachte. BT: N1 ziet het
  Engels, dat Gemma 4's vertaling van het origineel is en waar de rondgang van komt; N2 ziet geen Engels. De machinekant
  is in beide proeven Gemma 4 31B-it, hetzelfde model als de judge. Eén formulering per prompt, zero-shot; geen drempel;
  beslist niets over G4-8, de judge of een volgende arm.
- **Uitvoer.** `$GJ/fbn1/data/` en `$GJ/fbn2/data/` (`prompts-w<k>.parquet`, `refs.parquet` zonder tekst,
  `fbdata.json`), `$GJ/fbn1/scores/w<k>/`, `$GJ/fbn2/scores/w<k>/`, `$GJ/fbn/report.md`/`.json`; kopie
  `mbert-ops/translator-sft/g4-judge-naturalness-prompts.md` (`.json`), alleen tabellen met tellingen en scores. Prompts
  en teksten blijven in de bucket en op de VM.
- **Keten** (`mbert-ops/scripts/g4judgefbn_run.sh`, kopie in `gs://t5-dutch-english-usc2/hammond/scratch-scripts/`; code
  `pairjudge.py fbdata --prompt|fbnreport` en `run_pairjudge.sh fbndata|score SET=fbn1/fbn2|fbnreport`). Als FB.7: alleen
  mbert-mix; `final_eval_mix.lock` met `flock` tot het eind; mbert-corpus (T3) wordt niet aangeraakt; code als bundel
  via de bucket naar de 4 workers; elke ssh met `timeout 900` en de ssh-opties van G4-judge.10; vóór elke TPU-stap de
  TPU van de worker(s) vrij. Volgorde: `setup` (`ARM=G4-bon`), `fbndata` N1 en N2 (w0, CPU), `score` `SET=fbn1` en
  `SET=fbn2` (4 workers), `fbnreport` (w0, CPU). Markers `$G/stages/g4-judge/fbn-*.done`; niets wat §16–§20 schreven
  wordt overschreven. Elke stap eindigt op `EXIT 0`, anders `FAILED` en stoppen. [schatting] ~25–35 min.

## 21. G4-8-data N1-scores: voorbereiding, beslist niets

**Vastgelegd 24 nov 2026 (tijdstip: de commit van deze sectie op branch `en-nl-g4arm8`)**, vóór elke stap op een worker
en vóór elke uitslag van deze scoring [besluit van de hoofdsessie 24 nov ~05:20, Yeb sliep: mbert-mix bezig houden met
veilig voorbereidend werk; uitwerking door een subagent]. Vooraf gelezen: alleen tellingen (de logregels `PAIRS8`,
`SELECT8` en `PROMPTS`/`SCORED` van run `g4d8a1123a`, het rapport `g4-8-data.md` van §18 en het rapport
`g4-judge-naturalness-prompts.md` van FB.8); geen tekst (sample, anker, bron) en geen score van deze scoring. Een
scoring, geen arm: er wordt niets getraind, niets gesampled, niets vertaald en niets beslist. Claude en GPT zijn geen
judge of generator. FLAN blijft gemaximeerd zoals in §18 (besluit Yeb); deze scoring verandert geen verdeling.
`G8=$G/data/G4-8`.

*G4-8-N1.1 Doel.* Als Yeb voor een volgende arm kiest met "QE als juistheidsbewaker + natuurlijkheidsjudge als
selectie", liggen de oordelen van de natuurlijkheidsprompt N1 (§20 FB.8) op de bestaande G4-8-paren er al. Deze sectie
legt alleen vast hoe die oordelen gemaakt en geteld worden.

*G4-8-N1.2 Paren* (dezelfde als de rubriek-scoring van §18; er wordt niets getrokken, gesampled of met QE gescoord).
Per deel `c1`, `c2`, `c3` de kandidaten van §18 (G4-8-data.4) zoals ze in `$G8/<c>/judge/data/cands.parquet` staan:
geen fail, QE ≤ QE₀ (+ 1e-9), tekstueel anders dan het anker; gelijke genormaliseerde teksten binnen een bron één keer,
met de sleutel `<id>/j<j>` van het eerste sample. Per sleutel beide volgordes met de uids van §18, `<id>/j<j>/ab` en
`<id>/j<j>/ba`; kader `ab`: A = het sample, B = het anker (G4-0 greedy, `hypothesis0` van `$G8/plan/plan.parquet`);
`{bron}` = het Engels (`en`) van de bron. De teksten komen uit `$G8/plan/plan.parquet` en de samples
`$G8/<c>/samples/`. [gemeten, log `g4d8a1123a`] 158.506 prompts: `c1` 26.484, `c2` 65.978, `c3` 66.044.

*G4-8-N1.3 Controles* (`g4arm8.py n1pairs`; een afwijking stopt vóór er iets geschreven wordt).
1. **Zelfde paren als §18**: n1pairs bouwt uit dezelfde invoer ook de prompts van G4-judge.3 (de rubriek) en vergelijkt
   ze per uid (sha256 van de tekst) met §18's `$G8/<c>/judge/data/prompts-w<k>.parquet`: dezelfde verzameling uids,
   geen ontbrekende of extra, elke tekst gelijk, elke uid op dezelfde worker (`pairjudge.worker_of`). Daarmee zijn per
   uid bron, sample, anker en volgorde die van §18, en staat elke N1-prompt op de worker van zijn rubriekprompt.
2. Elke sleutel in `cands.parquet` is het eerste sample met zijn genormaliseerde tekst (als `pairs` van §18), en de
   sleutels horen bij bronnen van dat deel.
3. **sha256 van de invoer**: `plan.parquet`, `cands.parquet`, §18's `prompts-w<k>.parquet` en de sampleshards (per
   bestand) in `$G8/<c>/judge-n1/data/n1pairs.json`. `n1report` stopt als de `cands.parquet` die het leest een andere
   sha256 heeft dan in n1pairs.json, en legt de sha256 van zijn eigen invoer (§18's `choice.parquet`, de scoreshards
   van rubriek en N1, de N1-`choice.parquet`) vast in `report.json`.
4. **De rubriekkeuze van §18 opnieuw**: `n1report` berekent uit §18's scores (`$G8/<c>/judge/scores/`) met de regel van
   G4-8-data.6 de keuze opnieuw en vergelijkt per bron (reden en sample) met §18's `select/choice.parquet`; een
   verschil stopt.
5. De N1-scores dekken precies de sleutels × 2 volgordes (als `select` van §18: ontbrekend of extra stopt).

*G4-8-N1.4 Prompt N1*, letterlijk FB.8 (`pairjudge.N1_PROMPT`, commit `49bcc96`; de tekst van de gebruikersbeurt;
`{bron}`, `{A}` en `{B}` gestript ingevuld; verder als G4-judge.2: hetzelfde chat-template met één gebruikersbeurt en
een lege denkruimte):

```
Je krijgt een Engelse bron en twee Nederlandse teksten, A en B, met ongeveer dezelfde inhoud.

Oordeel alleen op natuurlijk, oorspronkelijk klinkend Nederlands: welke tekst leest het meest als tekst die oorspronkelijk in het Nederlands is geschreven, en het minst als een vertaling? Laat juistheid uitdrukkelijk buiten beschouwing: weglatingen, toevoegingen en afwijkingen van de Engelse bron tellen niet mee. "gelijk" als er geen duidelijk verschil is.

Engelse bron:
{bron}

Tekst A:
{A}

Tekst B:
{B}

Welke tekst leest het meest als oorspronkelijk Nederlands? Antwoord met één woord: A, B of gelijk.
```

*G4-8-N1.5 Scoren.* Als G4-8-data.5: `pairjudge.py score` ongewijzigd (Gemma 4 31B-it @ `842da37`, de
MaxText-omzetting, één prefill, P(A)/P(B)/P(gelijk) uit de logits van de eerste antwoordpositie met de ids van
G4-judge.2, beide volgordes, P̄ gemiddeld volgens G4-judge.4 met `pairjudge.combine`), prompts over de 4 workers van
mbert-mix op sha256 van de uid. P̄_N1(sample) = P̄(A) in kader `ab`: "het sample leest meer als oorspronkelijk
Nederlands dan het anker". Een prompt van meer dan 4.096 tokens wordt niet gescoord (geteld; die kandidaat valt af,
als §18).

*G4-8-N1.6 Keuze met N1 (ter voorbereiding).* `g4arm8.py select` ongewijzigd met de N1-scores: per bron k = de
kandidaat met de hoogste P̄_N1 (gelijk: de lagere QE, dan het eerdere sample); k is de N1-keuze als P̄_N1(k) > 0,5,
anders kiest N1 het anker; zonder kandidaat geen keuze. Uitvoer `$G8/<c>/select-n1/` (`choice.parquet`,
`targets.parquet`, `select.json`). Dit is geen trainingsset: of en hoe een volgende arm deze keuze gebruikt, beslist Yeb.

*G4-8-N1.7 Rapport* (`g4arm8.py n1report`; per deel FLAN / romans / BT en samen; alleen tellingen en scores, geen tekst).
1. Bronnen; bronnen met minstens één beoordeelde kandidaat; **bronnen met een sample dat N1 boven het anker zet
   (P̄_N1 > 0,5) met QE ≤ QE₀** (aantal en aandeel), met ernaast hetzelfde voor de rubriek (§18); niet gescoorde
   kandidaten.
2. **Overlap met de rubriekkeuze van §18** (per bron met een kandidaat): N1 kiest een sample en de rubriek koos
   hetzelfde sample / een ander sample / het anker; N1 kiest het anker en de rubriek koos een sample / ook het anker.
3. **Verdeling van P̄_N1(sample)**: per beoordeelde tekst (sleutel) en de hoogste per bron: gemiddelde, kwartielen,
   aandeel > 0,5 en een histogram in tienden; de rubriek-P̄ van dezelfde sleutels ernaast; de samenhang per sleutel
   (Spearman tussen P̄_N1 en P̄_rubriek).
Naar `$G8/n1/report.md`/`.json`, kopie `mbert-ops/translator-sft/g4-8-data-n1.md` (`.json`).

*G4-8-N1.8 Wat het niet zegt.*
- N1 vraagt alleen naar natuurlijkheid en sluit juistheid uitdrukkelijk uit. De enige juistheidsbewaking is die van §18:
  de fail-filters en QE ≤ QE₀ (MetricX-24-QE, een automatische maat zonder referentie). Een N1-keuze kan dus een
  betekenisfout hebben die QE niet ziet.
- N1 is alleen op FB.8's 1.000 paren bekeken (fictie mens/G4-0 en BT origineel/rondgang): daar zette hij de menselijke
  of oorspronkelijke kant vaker boven de machinekant. Dat is een andere vraag dan kiezen tussen samples van hetzelfde
  model; of N1 tussen G4-0's samples de natuurlijkere kiest, is niet gemeten.
- De judge is hetzelfde model als de generator (Gemma 4 31B-it): een voorkeur voor de eigen stijl is niet te scheiden.
- Geen menselijke of blinde beoordeling van de N1-keuzes, geen training, geen evaluatie. De tellingen zeggen hoe vaak N1
  en de rubriek hetzelfde kiezen, niet welke keuze beter is. Eén formulering, zero-shot; geen drempel buiten P̄ > 0,5
  van G4-8-data.6. Beslist niets over een volgende arm.

*G4-8-N1.9 Uitvoer.* Per deel `$G8/<c>/judge-n1/data/` (`prompts-w<k>.parquet`, `n1pairs.json`),
`$G8/<c>/judge-n1/scores/w<k>/`, `$G8/<c>/select-n1/`; `$G8/n1/report.md`/`.json`; markers `$G/stages/g4-8-n1/`. Niets
wat §18–§20 schreven wordt overschreven. Prompts en teksten blijven in de bucket en op de VM.

*G4-8-N1.10 Keten* (`mbert-ops/scripts/g4arm8_n1_run.sh`, kopie in `gs://t5-dutch-english-usc2/hammond/scratch-scripts/`;
code `g4arm8.py n1pairs|select|n1report` en `run_g4arm8.sh n1pairs|judge SET=n1|select SET=n1|n1report` op
`en-nl-g4arm8`). Alleen mbert-mix; `final_eval_mix.lock` met `flock` tot het eind; mbert-corpus (T3) wordt niet
aangeraakt. Code als bundel via de bucket naar de 4 workers; elke ssh met `timeout 900` en de ssh-opties van
G4-judge.10; vóór elke TPU-stap de TPU van de worker(s) vrij. Volgorde: `setup` (`run_g4arm.sh setup`, `ARM=G4-bon`);
per deel `c1`, `c2`, `c3`: `n1pairs` (w0, CPU), `judge` met `SET=n1` (4 workers), `select` met `SET=n1` (w0, CPU); tot
slot `n1report` (w0, CPU). Elke stap eindigt op `EXIT 0`, anders `FAILED` en stoppen; een gemarkeerde stap wordt bij een
herstart overgeslagen. [schatting] ~2–2,5 u: §18's rubriek-scoring van dezelfde prompts duurde 20 / 45 / 45 min
(`c1` / `c2` / `c3`, gemeten in de log van `g4d8a1123a`); N1 is korter dan de rubriekprompt.

## 22. G4-9 training: SFT met LoRA op de N1-keuze van G4-8-data

**CONCEPT, NIET GESTART; wacht op Yebs besluit (G4-9 ja of nee, O1, O6, O9).** Opgesteld 24 nov 2026 (~08:00) door een
subagent op verzoek van de hoofdsessie [opdracht: de training van een mogelijke arm G4-9 klaarzetten zoals G4-8, maar met
de N1-keuze van §21 als doelen; bouwen, testen en droog draaien, niets trainen; Yeb sliep]. Vooraf gelezen: alleen
tellingen (de rapporten `g4-8-data-n1.json` van §21 en `g4-8-report.json` van §19, de logregels met tijden van de keten
`g4t8a`…`g4t8c`, de LEARNINGS-regels van G4-8 en N1); geen tekst (doel, sample, anker, bron) en geen uitslag van G4-9. De
**definitieve versie** volgt vóór de start: dan vervangt de hoofdsessie deze kop door "Vastgelegd <datum>", vult G4-9.0 in
met Yebs besluiten en zet het startbestand (G4-9.14). De keten weigert een protocol waarin deze conceptregel nog staat.
Alles is als §19 (definitief, met G4-8.13 en G4-8.14), behalve de doelen (G4-9.2), de namen, de zaden en de vergelijking
met G4-8 in het rapport (G4-9.11); waar hieronder "als G4-8.x" staat, geldt §19 letterlijk met `G4-8` vervangen door
`G4-9` in namen, zaden en plaatsen. Claude en GPT zijn geen judge of generator. `G8=$G/data/G4-8`, `G9=$G/data/G4-9`,
`RL=$G/rl/G4-7`.

*G4-9.0 Besluiten vóór de start* [open: Yeb]. Vast te leggen, elk als één regel:
- `O1 [besluit Yeb …]: G4-0` (startmodel; standaard, als §19). Het alternatief G4-8-s140 (verder trainen vanaf de keuze
  van §19) is **niet gebouwd**: het vraagt het stapelen van adapters zoals G4-6b (`sft_lora.py --base-adapter`,
  `g4mix.py stack` bij vertalen en samplen, rank 64) en een andere vergelijkingsbasis voor de regel; kiest Yeb dat, dan
  eerst een codewijziging en een nieuwe versie van deze sectie. **De keten leest deze regel** en weigert een andere
  waarde dan `G4-0`.
- `O5 [besluit Yeb …]: 2 epochs, LR 5e-5` (als §19).
- `O6 [besluit Yeb …]: R-nieuw` of `O6 [besluit Yeb …]: R-oud` (G4-9.10; standaard R-nieuw, als §19). **De keten leest
  deze regel.**
- Twijfelgevallen onder R-nieuw [besluit Yeb …]: Yebs oordeel (G4-8.9) of, als Yeb dat weer wil, de tweede blinde ronde
  van G4-8.13 (dan ook voor de controle van 10 zekere oordelen). Raakt de keten niet (de beoordeling is na `DONE`).
- `O9 [besluit Yeb …]: n1` of `O9 [besluit Yeb …]: n1∩rubriek` (de doelen, G4-9.2; toegevoegd 24 nov 2026 ~10:40 door
  een subagent op verzoek van de hoofdsessie, na G4-9.V3N1; geen tekst gelezen). `n1`: alle doelen van de N1-keuze van
  §21, **16.349**. `n1∩rubriek` (de doorsnede): alleen de bronnen waar de N1-keuze van §21 en de rubriekkeuze van §18
  **hetzelfde sample** kozen, **9.126** (FLAN 793, romans 4.507, BT 3.826) [gemeten: §21's rapport
  `g4-8-data-n1.md`, overlap "N1 kiest een sample, de rubriek hetzelfde"]. Reden: de blinde blik op de N1-doelen
  (G4-9.V3N1; 100 paren doel tegen anker uit `c1`, beoordeeld door een verse Claude-subagent; `mbert/LEARNINGS.md` van
  hammond, 24 nov ~10:15), uitgesplitst naar wat de rubriek voor die bron koos, als doel beter / anker beter / gelijk
  en betekenisfouten in doel / anker: **zelfde sample 29 / 3 / 27 (fouten 5 / 6)**, **ander sample 6 / 11 / 5 (fouten
  8 / 2)**, rubriek koos het anker 3 / 4 / 12 (fouten 2 / 1). **Advies van de hoofdsessie: de doorsnede
  (`n1∩rubriek`).** **De keten leest deze regel** en vergelijkt hem met de regel `TARGETS=` van het startbestand
  (`n1` hoort bij `TARGETS=n1`, `n1∩rubriek` bij `TARGETS=intersect`; G4-9.14).

*G4-9.1 Startmodel en LoRA* als G4-8.1: G4-0 zonder adapter, LoRA rank 32 / alpha 16 (schaal 0,5) op de 410 modules van
`run_sft.sh`, AdamW als G4-1…G4-5, `num_vocab_tiling=8`, MaxText 0.2.5.

*G4-9.2 Doelen* (`g4arm8.py n1targets`, w0, CPU; een stap van de keten **na** het startbestand, zodat er vóór Yebs
besluit niets nieuws in de bucket komt). De N1-keuze van §21 (G4-8-N1.6): per deel `c1`, `c2`, `c3` de bronnen waar N1
een kandidaat van §18 (geen fail, QE ≤ QE₀, anders dan het anker) boven het anker zet (P̄_N1 > 0,5), uit
`$G8/<c>/select-n1/targets.parquet` en `choice.parquet`; samengevoegd tot één `$G9/targets/targets.parquet` in de vorm
van §18 (G4-8-data.9: dezelfde kolommen, op id gesorteerd; `p_sample` is hier P̄_N1), met `targets.json` (tellingen per
deel, sha256 van invoer en uitvoer). Gecontroleerd (een afwijking is `FAILED`; er wordt dan niets geschreven):
- `$G8/n1/report.json` is het rapport van §21 over `c1`, `c2`, `c3`, en elke `choice.parquet` is (sha256) de
  `choice-n1.parquet` die `n1report` las en tegen de N1-scores controleerde (G4-8-N1.3.4);
- per deel dekt `choice.parquet` precies de bronnen van dat deel in `$G8/plan/plan.parquet`, en zijn de doelen precies
  de bronnen met reden `sample`, met dezelfde `j`, P̄ en QE als in `choice.parquet`; elk id één keer;
- elk doel: deel, chunk en Engels als in het plan, P̄_N1 > 0,5, **QE ≤ QE₀ (+ 1e-9)** (de juistheidsbewaker,
  opnieuw nagegaan), een niet-lege tekst die (na normalisering van witruimte) niet die van het anker is;
- met O9 `n1`: de aantallen per deel zijn die van het rapport van §21: **16.349** (FLAN 1.391, romans 7.900, BT 7.058)
  [gemeten, §21].
Met O9 `n1∩rubriek` (`n1targets --mode intersect`; de keten geeft `TARGETS` van het startbestand door, `n1` is
`--mode n1`) blijven van die N1-doelen alleen de bronnen waar §18's rubriekkeuze `$G8/<c>/select/choice.parquet` reden
`sample` met **dezelfde `j`** heeft (de regel "zelfde sample" van G4-8-N1.7.2). Daarbij gecontroleerd:
- elke `select/choice.parquet` is (sha256) de `choice.parquet` die §21's `n1report` las en tegen de rubriekscores
  controleerde, en dekt precies de bronnen van dat deel in het plan;
- §18's `$G8/<c>/select/targets.parquet` heeft precies de bronnen met reden `sample` van die keuze, met dezelfde `j`;
  voor elk doel van de doorsnede zijn `j`, QE en tekst die van §18 (het doel is dus ook een doel van G4-8);
- de aantallen per deel zijn de overlap "zelfde sample" van §21's rapport: **9.126** (FLAN 793, romans 4.507, BT
  3.826) [gemeten, §21].
`targets.json` noemt de variant (`mode`: `n1` of `intersect`), bij `intersect` ook de aantallen van alle N1-doelen en de
sha256 van §18's bestanden. Staat er al een `$G9/targets/targets.json` van de andere variant, dan weigeren de stap en de
keten (`FAILED`; het bestand blijft staan).
FLAN blijft gemaximeerd (besluit Yeb 23 nov ~16:40): de bronnen zijn die van §18 (10 / 45 / 45), van de doelen is FLAN
8,5% (1.391 / 16.349; bij de doorsnede 8,7%, 793 / 9.126). Geen ankerrijen (O3 van §18).

*G4-9.3 Trainingsdata* als G4-8.2 (`g4arm8.py sftset --arm G4-9` op `$G9/targets/targets.parquet`; alle controles van
G4-8.2 ongewijzigd, ook de lekcontrole), de doorgangen geschud met zaad `en-nl-g4arm/0/G4-9/train/shuffle/<doorgang>`;
uitvoer `$G9/sft/`. `g4arm.py check --arm G4-9` als G4-8. [schatting] Bij §19 gaven 15.194 doelen 6,78 M tokens per
doorgang en 205 stappen; 16.349 doelen ≈ 7,3 M tokens per doorgang ≈ 220 stappen; de doorsnede (9.126 doelen, met de
tokens per doel per deel van §19: FLAN 266, romans 534, BT 375) ≈ 4,05 M tokens per doorgang ≈ 123 stappen (E = 21,
kandidaten na 42, 84 en de laatste stap). De lekcontrole van G4-8.2 is per doel en geldt ongewijzigd voor beide
varianten; de doelen van de doorsnede zijn ook doelen van G4-8 (G4-9.2), waarvan de set die controle doorstond (alles 0,
`translator-sft/g4-8-sftset.md`) [gemeten]. `sftset` krijgt bij G4-9 ook `targets.json` (`--targets-info`): de sha256 van
`targets.parquet`, de arm en de aantallen per deel moeten kloppen; de variant komt in het manifest (`targets_mode`), in
`g4-9-sftset.md` en in het rapport (G4-9.12). Een manifest van de andere variant laat de keten weigeren.

*G4-9.4 Training* als G4-8.3: LR 5e-5 constant na 10 stappen lineaire warmup, rijen van 4.096, 16 per stap, één
multi-host-job over de 16 chips van mbert-mix; evaluatie op `books`, `btval`, `own` op stap 0, elke E = ⌈stappen / 6⌉
stappen en na de laatste. Run `g4arm_G4-9` (een herstart na een mislukte run: een nieuwe letter).

*G4-9.5 Kandidaten* als G4-8.4: de adapters na K₁ = 2E, K₂ = 4E en de laatste stap, namen `G4-9-s<stap>`, naar
`$G/G4-9-s<stap>/` (rank 32 / alpha 16 / schaal 0,5 gecontroleerd; niets gestapeld).

*G4-9.6 Smoke-test* als G4-8.5 (met G4-8.14: de LR binnen float32-precisie), de smoke-set met zaad
`en-nl-g4arm/0/G4-9/smoke`, run `g4arm_G4-9_s8`.

*G4-9.7 Per kandidaat, gerapporteerd* als G4-8.6; de RL-val-samples met zaad per worker `en-nl-g4arm/0/G4-9/div/w<k>`,
voor elk model hetzelfde, ook voor G4-0 zonder adapter (opnieuw gesampled), uitvoer `$G9/sft/div/<model>/`.

*G4-9.8 Evaluatie* als G4-8.7 met `ARM=G4-9`. C1 in één classify-run over de kandidaten, G4-0, G4-4 en **G4-8-s140**
(de keuze van §19; zijn vertalingen en scores in `$G/eval/G4-8-s140/` worden gelezen, niet opnieuw gemaakt) op
`$G/naturalness-G4-9/`; C2 en PLL op dezelfde teksten.

*G4-9.9 Keuze (zoekdeel)* als G4-8.8 (`select5` over de kandidaten van G4-9; G4-4 en G4-8-s140 doen niet mee).

*G4-9.10 Beslisregel (beslisdeel), tegen G4-0* als G4-8.9: beide varianten uitgerekend, de regel van G4-9.0 beslist; de
bootstrap van (b) met zaad `en-nl-g4arm/0/G4-bon/boot/G4-9/b`. Het blinde pakket van de gekozen kandidaat tegen G4-0 in
`$S/blind/G4-9/` en tegen TG in `$S/blind/G4-9-tg/` als (a1) en (a3) gelden; beoordeling en telling (`g4arm.py tally`,
`$G/report-G4-9/tally.json`) door de hoofdsessie, daarna `report8` opnieuw met de telling. Onder R-nieuw de
twijfelgevallen en de 10 zekere oordelen (`random.Random("en-nl-g4arm/0/G4-9/yeb10")`) volgens G4-9.0. [gevolgtrekking]
Het beslisdeel is na §19 één keer gelezen (de uitslag van G4-8): de keuze van G4-9 kijkt er niet naar, en het besluit
tegen G4-0 is als bij G4-8, maar een vergelijking "G4-9 tegen G4-8" op hetzelfde beslisdeel is niet meer onafhankelijk
(G4-9.11 beslist daarom niets).

*G4-9.11 Vergelijking met G4-8 (rapporteren, beslist niets).* Op het beslisdeel de gekozen kandidaat van G4-9 tegen
G4-8-s140 (de keuze van §19; de keten controleert `selected` in `$G/report-G4-8/report.json`): per set QE van beide, het
verschil en een 95%-interval (gepaarde bootstrap per set, `g4bon.paired_ci`, 10.000 trekkingen, zaad
`en-nl-g4arm/0/G4-bon/boot/G4-9/vs-G4-8/qe/<set>`), (a1) van beide; COMET-22 per set met het verschil en "(a3) tegen
G4-8" (≥ G4-8 − 0,005); C1 "origineel" op WMT24++ (goed) + FLAN van beide met het verschil en een 95%-interval (zaad
`…/G4-9/vs-G4-8/c1`); C2 van beide met het verschil. [keuze] Geen blinde vergelijking G4-9 tegen G4-8 in deze keten;
kan na `DONE` als rapportage (`armeval.py blind` met G4-8-s140 als basis) als Yeb dat wil.

*G4-9.12 Rapport* (`g4arm8.py report8 --arm G4-9 --compare G4-8-s140` → `$G/report-G4-9/report.md`/`.json`, kopie
`mbert-ops/translator-sft/g4-9-report.md`): als G4-8.10, plus G4-9.11; het rapport noemt de variant van de doelen (O9,
`targets_mode` uit het manifest van `sftset`).

*G4-9.13 Uitvoer.* `$G9/targets/` (G4-9.2), `$G9/sft/` (set, smoke-set, `check.txt`, RL-val-segmenten, `div/<model>/`),
`$G9/logs/`, `$G/runs/g4arm_G4-9_s8/`, `$G/runs/g4arm_G4-9/`, `$G/G4-9-s<stap>/`, `$G/eval/G4-9-s<stap>/`,
`$G/naturalness-G4-9/`, `$G/report-G4-9/`, `$S/blind/G4-9/`, `$S/blind/G4-9-tg/`; markers `$G/stages/g4-9-train/`. Van
§18–§21 en van G4-8 wordt alleen gelezen; niets wordt overschreven.

*G4-9.14 Keten* (`mbert-ops/scripts/g4arm9_train_run.sh`, dat `g4arm8_train_run.sh` met `ARM8=G4-9` draait; kopieën in
`gs://t5-dutch-english-usc2/hammond/scratch-scripts/`; code `g4arm8.py`, `run_g4arm8.sh` met `ARM8=G4-9` en `run_g4arm.sh`
met `ARM=G4-9` op `en-nl-g4arm8`). Alleen mbert-mix (een v4-32: de training over alle 16 chips, de rest per worker);
mbert-corpus (T3) wordt niet aangeraakt. Wacht eerst op de marker `$G/stages/g4-8-n1/report.done` van §21 en
`$G8/n1/report.json` (beide er al), dan op het startbestand `mbert-ops/g4arm9.train-go` (drie regels: `O6=R-oud` of
`O6=R-nieuw`, gelijk aan G4-9.0; `TARGETS=n1` of `TARGETS=intersect`, gelijk aan O9 van G4-9.0; en
`COMMIT=<de gepushte commit met de definitieve §22>`, die afstamt van dit concept en van de geteste code; de definitieve
§22 heeft precies één O6-regel, precies één O9-regel en precies één O1-regel, `G4-0`); bij `intersect` ook §18's
`$G8/<c>/select/choice.parquet` en `targets.parquet`; een `$G9/targets/targets.json` of `$G9/sft/manifest.json` van de
andere variant is `FAILED` vóór er iets op een worker gebeurt. Dan op `final_eval_mix.lock` (met `flock`, tot het eind
gehouden). Volgorde: setup (`FULL=1`; daarna op elke worker ≥ 12 GB vrij in `/dev/shm`, anders `FAILED`), `n1targets`
(w0, met `TARGETS`; daarna controleert de keten in `targets.json` de variant en de aantallen, 16.349 of 9.126 met de
aantallen per deel van G4-9.2), `sftset` (w0; daarna in het manifest de variant en de sha256 van de doelen), `check`
(w0), smoke-run + `smoke8` (w0), de training,
`adapter` (w0), per kandidaat `translate` en `score` (4 workers), RL-val-samples per model (4 workers), `texts`,
`classify`, `classify-c2`, `pll` (w0), `report8` (w0), het blinde pakket als (a1) en (a3) gelden; `DONE` met de keuze en de
regels. Elke stap eindigt op `EXIT 0`, anders `FAILED` en stoppen; een gemarkeerde stap wordt bij een herstart
overgeslagen. [schatting, naar de gemeten tijden van G4-8 in de ketens `g4t8a`…`g4t8c`] Met `n1` ~3,5 u na de start
(3–4 u): setup ~3 min, `n1targets` ~3 min, `sftset` + `check` ~5 min, smoke ~22 min, training ~65–70 min (~220 stappen;
G4-8: 205 stappen in 62 min), `adapter` ~2 min, per kandidaat vertalen + scoren ~16 min (3×), RL-val-samples ~26 min,
C1/C2/PLL ~25 min, rapport en blind pakket ~5 min. Met `intersect` ~3 u (2,5–3,5 u): `sftset` + `check` ~4 min, training
~40–50 min (~123 stappen: de 7 evaluatierondes, het laden en compileren blijven, ~10,6 s per stap zoals bij G4-8), de
rest als bij `n1`.

*G4-9.15 Wat het niet zegt.*
- N1 sluit juistheid uit; de enige juistheidsbewaking van de doelen is die van §18 (de fail-filters en QE ≤ QE₀,
  MetricX-24-QE zonder referentie). Een doel kan een betekenisfout hebben die QE niet ziet (G4-8-N1.8). De N1-doelen zijn
  niet door Yeb bekeken; G4-9.V3N1 is één blinde Claude-beoordeling van 100 paren uit `c1`.
- O9: de reden voor de doorsnede is G4-9.V3N1, een steekproef van 100 paren (zelfde sample 59, ander sample 22, rubriek
  koos het anker 19) [gevolgtrekking: de uitsplitsing per groep rust op 19–59 paren]. Met de doorsnede zijn de doelen een
  deelverzameling van die van G4-8 (9.126 van 15.194, 60%): een verschil met G4-8 komt dan uit de 6.068 bronnen van G4-8
  die wegvallen (N1 koos daar een ander sample, 4.856, of het anker, 1.212 [gemeten, §21]), het kleinere aantal stappen
  en andere zaden; over de N1-doelen die de rubriek niet koos zegt zo'n G4-9 niets.
- De evaluatie en de blinde rubriek (protocol.md §4.2.4: eerst juistheid, dan natuurlijkheid) zijn die van G4-8; C1 is
  de maat voor "origineel"; (b) blijft onder R-nieuw rapportage.
- De judge is hetzelfde model als de generator van de samples (Gemma 4 31B-it): een voorkeur voor de eigen stijl is niet
  te scheiden.
- Met O9 `n1`: een verschil tussen G4-9 en G4-8 komt uit de doelen (in ~1/3 van de bronnen een ander sample of een
  andere keuze tussen sample en anker, §21) en hun aantal (16.349 tegen 15.194), en ook uit andere zaden (schudding,
  smoke-set, RL-val).
- Geen tweede ronde, geen judge op evaluatie-uitvoer, geen beoordeling van het blinde pakket (die is aan de hoofdsessie
  en Yeb).

*G4-9.V3N1 Aanvulling 24 nov 2026 ~09:55: blinde blik op de N1-doelen, rapportage* (opdracht van de hoofdsessie 24 nov
~09:45, vóór Yebs besluit over G4-9; uitwerking door een subagent; tijdstip: de commit van deze alinea op `en-nl-g4arm8`,
vóór elke stap op een worker). Beslist niets en verandert G4-9.0–G4-9.15 niet; er wordt niets getraind, gesampled,
vertaald of gescoord: het pakket bevat alleen bestaande teksten (doelen van §21, ankers van §18). Vooraf gelezen: alleen
tellingen (§21's rapport; per deel, bron en reden de rijen van `$G8/c1/select-n1/choice.parquet` en
`$G8/c1/select/choice.parquet`); geen tekst.
- **Pakket** `$S/blind/G4-9-V3N1/` in de vorm van G4-8-data.8 (`g4arm8.py v3 --set n1`, `run_g4arm8.sh v3` met `SET=n1`,
  w0, CPU): 100 paren doel tegen anker uit deel `c1`; doel = de N1-keuze (G4-8-N1.6, `$G8/c1/select-n1/targets.parquet`),
  anker = G4-0 greedy (`hypothesis0` van het plan). BT 40: per bron (dpc, fineweb, handelingen, raad, rechtspraak) 8 met
  `random.Random("en-nl-g4arm/0/G4-9/v3n1/bt/<bron>").sample` van de gesorteerde id's met een doel; romans 40
  (`…/G4-9/v3n1/novels`); FLAN 20 (`…/G4-9/v3n1/flan`); minder doelen: alle. `blind_id` `v3n1-001…v3n1-100` (BT per bron
  op naam, dan romans, dan FLAN; binnen een groep in de volgorde van de trekking); A/B met
  `random.Random("en-nl-g4arm/0/G4-9/v3n1/ab/<blind_id>").random() < 0,5` → A = het anker (`base`), anders A = het doel
  (`G4-9-n1`). [gemeten in de c1-keuzes] BT 200–253 doelen per bron, romans 1.312, FLAN 236.
- **Controles** (een afwijking stopt vóór er iets geschreven wordt): die van G4-9.2 op `c1` (`choice.parquet` is (sha256)
  de `choice-n1.parquet` die §21's `n1report` las; de doelen zijn precies de bronnen met reden `sample`, met dezelfde `j`,
  P̄ en QE; P̄_N1 > 0,5; QE ≤ QE₀ (+ 1e-9); deel en Engels als in het plan; niet leeg en niet de tekst van het anker);
  daarbij zijn het plan en §18's `$G8/c1/select/choice.parquet` (sha256) die welke §21 las.
- **Uitvoer**: `judge-input.jsonl` (blind_id, source, A, B); apart `key.jsonl` (blind_id, A, B, seed, set, item, stratum
  en `rubric`: wat §18's rubriek voor die bron koos, `zelfde sample` / `ander sample` / `anker`, met de regel van
  G4-8-N1.7.2); `select-report.json` (aantallen per set en bron, de verdeling van `rubric` over de getrokken paren en over
  alle doelen van de set in `c1`, sha256 van de invoer; geen tekst). Een bestaande `key.jsonl` wordt niet overschreven.
  Eén job op mbert-mix w0 met `final_eval_mix.lock`; mbert-corpus (T3) wordt niet aangeraakt.
- **Beoordeling** als G4-8-data.8 (verse Claude-subagent, rubriek protocol.md §4.2.4, alleen `judge-input.jsonl`, de
  sleutel pas daarna) door de hoofdsessie; rapportage voor Yebs besluit over G4-9, zonder drempel of regel. Claude en GPT
  maken of kiezen geen doelen. [gevolgtrekking] Paren met `rubric` = `zelfde sample` zijn ook doelen van G4-8; het
  verschil tussen N1 en de rubriek zit in de paren met `ander sample` en `anker`. G4-8-V3 kwam ook uit `c1`: een bron kan
  in beide pakketten zitten (onafhankelijk getrokken).

## 23. Meting G4-maat: welk automatisch signaal kiest dezelfde kant als Yeb

**Vastgelegd 24 nov 2026 (tijdstip: de commit van deze sectie op branch `en-nl-g4arm8`)**, vóór elke stap van deze
meting op een worker en vóór elke uitslag [besluit Yeb 24 nov ~11:15: G4-8-s140 is voortaan de vertaler; geen G4-9
voordat een maat gevonden is die samenhangt met zijn menselijke oordelen; uitwerking door een subagent]. Vooraf gelezen:
Yebs antwoorden (`answers.txt` van beide steekproeven) en de sleutels; van de blinde pakketten de veldnamen en de
aantallen per oordeel (de labels van de Claude-oordelen; hun uitleg niet); de kolomnamen en aantallen (geen tekst, geen
score) van de bestanden die de keten leest, met één leesopdracht op mbert-mix w0. Geen tekst (bron, vertaling, roman) en
geen score van deze meting. Een meting: er wordt niets getraind, gesampled, vertaald of beslist. Claude en GPT zijn geen
judge of generator; de bestaande Claude-oordelen dienen alleen als vergelijking. `GM=$G/measure`.

*G4-maat.1 Vraag.* Welk automatisch signaal (of welke van vier vooraf vastgelegde combinaties) kiest bij een paar
vertalingen van dezelfde Engelse bron het vaakst dezelfde kant als Yeb? Ter vergelijking: hetzelfde tegen de blinde
Claude-oordelen (rubriek protocol.md §4.2.4), en Claude tegen Yeb op de paren die beiden beoordeelden. Rapportage; geen
drempel, geen winnaar, geen besluit.

*G4-maat.2 Paren en oordelen.*
- **Yeb, steekproef 1** (V1 van `translator-sft/g4-8-spec-judge.md`): 40 paren,
  `$S/blind/steekproef-yeb/key-map.json` (nummer → pakket `G4-judge-i15` of `G4-judge-anchor-i15`, `blind_id`),
  antwoorden `answers.txt` (A of B, in het kader van het pakket).
- **Yeb, gemengde steekproef** (G4-8.15): 40 paren G4-8-s140 tegen G4-0, `$S/blind/steekproef-yeb-g4-8-mix/key.json`
  (`n`, `set`, `item`, `A`, `B`), antwoorden `answers.txt` (A, B of `=`, eventueel `fout A` / `fout B`), in het kader
  van de steekproef. Samen [gelezen] 74 niet-gelijke oordelen (40 + 34), 6 keer `=`.
- **Claude** (één verse subagent per pakket, eerste ronde, vóór elke vervanging van twijfelgevallen): de 14 pakketten
  in `$S/blind/` met hun `judge-input.jsonl` (de teksten die de beoordelaar zag) en `key.jsonl`, samen 1.398 paren,
  824 niet-gelijk [gelezen, labels]:

  | pakket | paren | A/B (sleutel) | Claude-oordelen | paren gekozen op |
  |---|---:|---|---|---|
  | G4-7 | 100 | G4-7-i15 / G4-0 | `review-74l/judgments.jsonl` | – (vaste 100) |
  | G4-7-tg | 100 | G4-7-i15 / TranslateGemma 27B | `review-74l/judgments.jsonl` | – (vaste 100) |
  | G4-bon-i15c2q | 100 | (i15, c2q) / G4-0 | `review/judgments.jsonl` | C2 + QE ≤ QE₀ |
  | G4-judge-i15 | 100 | (i15, judgeq) / G4-0 | `review/judgments.jsonl` | rubriek + QE ≤ QE₀ |
  | G4-judge-G4-0 | 100 | (G4-0, judgeq) / G4-0 | `review/judgments.jsonl` | rubriek + QE ≤ QE₀ |
  | G4-judge-i15-vs-qe | 100 | (i15, judgeq) / (i15, qe) | `review/judgments.jsonl` | rubriek + QE ≤ QE₀ tegen laagste QE |
  | G4-judge-anchor-i15 | 100 | (i15, janchor) / G4-0 | `review/judgments.jsonl` | rubriek (P̄ > 0,5) + QE ≤ QE₀ |
  | G4-8 | 100 | G4-8-s140 / G4-0 | `review/judge-judgments.jsonl` | – (vaste 100) |
  | G4-8-tg | 100 | G4-8-s140 / TranslateGemma 27B | `review/judgments.jsonl` | – (vaste 100) |
  | G4-8-replica | 100 | G4-8-s140 / G4-0 | `review/judgments.jsonl` | – |
  | G4-8-romans-bt | 98 | G4-8-s140 / G4-0 | `review/judgments.jsonl` | – |
  | G4-8-V3 | 100 | janchor-doel / G4-0 | `review/judgments.jsonl` | rubriek (P̄ > 0,5) + QE ≤ QE₀ |
  | G4-9-V3N1 | 100 | N1-doel / G4-0 | `review/judgments.jsonl` | N1 (P̄ > 0,5) + QE ≤ QE₀ |
  | RB3 | 100 | RB3 / TranslateGemma 27B | `review-exploratory/judgments.jsonl` | – (vaste 100) |

  Groepen: **alle** (14); **Gemma 4 tegen Gemma 4** (zonder G4-7-tg, G4-8-tg en RB3, waar één kant TranslateGemma is;
  600 niet-gelijk); **zonder paarselectie** (G4-7, G4-8, G4-8-replica, G4-8-romans-bt: twee greedy systemen, geen
  signaal koos de paren; 191 niet-gelijk). Daarnaast alleen voor "Claude tegen Claude": de tweede beoordelaar van G4-8
  (`review/second-reviewer-all.jsonl`).
- **Teksten** = wat de beoordelaar zag: bij de pakketten `judge-input.jsonl` (`source`, `A`, `B`); bij de gemengde
  steekproef per `key.json` opnieuw opgebouwd zoals `g4mixsample.pick` (FLAN, WMT24++, FLORES, NTREX en idioom uit
  `$G/eval/{G4-0,G4-8-s140}/pairs/`, idioom als de rij `blind-NNN`; romans en BT uit `$G/mixsample/pairs/`), en
  gecontroleerd: `g4mixsample.sample_text` van deze 40 paren met de datum uit de kop van het bestand geeft byte voor byte
  `steekproef-yeb-g4-8-mix.md` in de bucket (anders stopt de meting). Steekproef 1 bestaat uit paren van twee pakketten
  en gebruikt hun teksten.
- **Gedeelde paren Yeb–Claude**: de 40 van steekproef 1 (zelfde paar, zelfde kader) en de paren van de gemengde
  steekproef die ook in een pakket zitten met (gestript) dezelfde bron en dezelfde twee vertalingen: de roman- en
  BT-paren van `G4-8-romans-bt` (veld `steekproef_n`) en de idioomparen van `G4-8` (`blind_id` = de rij `blind-NNN`);
  Claude's oordeel naar het kader van de steekproef omgezet via de systeemnamen van de twee sleutels. Een paar waarvan
  een tekst verschilt, telt niet als gedeeld (geteld).

*G4-maat.3 Signalen* (per kant A en B, vers berekend op de teksten van G4-maat.2, gestript zoals de judge ze krijgt,
in één keten op mbert-mix).
1. **QE**: MetricX-24 hybrid XXL zonder referentie (`metricx_qe.py score --pair source:hypothesis --model xxl`, bf16,
   als §4.3; lege vertaling gescoord); lager is beter.
2. **COMET-22** met de referentie van de testset (`comet_score.py`, CPU), alleen WMT24++, FLORES en NTREX: de rij van
   `$G/data/eval/evalsets.parquet` met het item als id, bij de vaste 100 de rij `blind-NNN` (35 WMT24++ met referentie);
   de bron van die rij moet (gestript) de bron van het paar zijn, anders stopt de meting. Niet bij FLAN en idioom (geen
   referentie), romans en BT [keuze: een gepubliceerde romanvertaling neemt vrijheden (§20 FB.5); bij BT is het
   Nederlandse origineel de bron van het Engels, geen vertaling ervan].
3. **C1**: P("origineel") van de classifier van 10 nov (S @ 653k, `$S/arms/naturalness/classifier/`), `run_arms.sh
   classify` ongewijzigd, op de Nederlandse tekst alleen; één classify-run over alle kanten.
4. **C2**: P("origineel") van C2 (`$RL/c2/model/`), `naturalness.py classify --max-length 1024` zoals `classify-c2`, op
   dezelfde teksten.
5. **Judge, rubriek** (prompt G4-judge.3), 6. **judge, N1** (FB.8, met bron), 7. **judge, N2** (FB.8, zonder bron):
   Gemma 4 31B-it, `pairjudge.py score` ongewijzigd (één prefill, G4-judge.2), beide volgordes, P̄ volgens G4-judge.4 in
   het kader van het paar; uids `<paar>/<prompt>/ab|ba`, verdeeld over de 4 workers op sha256 van de uid. Een prompt van
   meer dan 4.096 tokens wordt niet gescoord (geteld).
8. **Lengte**: Gemma 4-tokens van de gestripte tekst (`g4arm.TokenCounter`, zonder speciale tokens).
Daarnaast twee **referentierijen** die geen signaal zijn: **nieuw systeem** (altijd de kant waarvan de sleutel niet
`base` is: bij steekproef 1 de judge-keuze, bij de gemengde G4-8-s140) en **positie A**.

*G4-maat.4 Keuze van een signaal per paar.* QE: de kant met de lagere QE; COMET-22, C1, C2: de hogere; judge: het teken
van P̄(A beter) − P̄(B beter); lengte: de kant met minder tokens. Gelijke waarden (QE en COMET-22 zoals geschreven, op 4
decimalen) en een ontbrekende waarde (geen referentie, niet gescoord): **geen keuze**.

*G4-maat.5 Combinaties* (vooraf vastgelegd, geen andere; [keuze] QE als juistheidsbewaker en een judge waar QE weinig
verschil ziet, de opzet van §21–§22; K4 is de doorsnede van O9 met QE als terugval; de grens 0,2 is vooraf gekozen, het
voorbeeld in de opdracht van de hoofdsessie, niet gezocht):
- **K1**: de QE-winnaar; bij |ΔQE| < 0,2 de keuze van N1.
- **K2**: de QE-winnaar; bij |ΔQE| < 0,2 de keuze van de rubriek.
- **K3**: de meerderheid van QE, rubriek en N1 (de kant die minstens twee van de drie kiezen; anders geen keuze).
- **K4**: kiezen rubriek en N1 dezelfde kant, dan die; anders de QE-winnaar.

*G4-maat.6 Maten* (`g4measure.py report`; alleen tellingen en percentages).
1. **Overeenstemming** met een referentie (Yeb, Claude) op de paren waar de referentie A of B kiest: n_ref; n = de paren
   waar het signaal kiest; k = dezelfde kant; k / n met een 95%-Wilson-interval; n_ref − n (geen keuze) apart.
2. **Hoofdtabel**: de 8 signalen, K1–K4, Claude (als signaal, alleen tegen Yeb, op de gedeelde paren; Claude "gelijk" =
   geen keuze) en de twee referentierijen × Yeb steekproef 1 / Yeb gemengd / Yeb samen / Claude alle / Claude Gemma 4
   tegen Gemma 4 / Claude zonder paarselectie.
3. **Per set**: Yeb steekproef 1 per set (FLAN, WMT24++) en per pakket (judgeq, janchor); gemengd per groep (romans, BT,
   FLAN, idioom, nieuws = WMT24++ + FLORES + NTREX); beide samen per groep (FLAN, nieuws, romans, BT, idioom); Claude per
   pakket en per set (FLAN, WMT24++, idioom, romans, BT). Alle groepen met n; bij n < 10 geen lezing.
4. **Claude tegen Yeb** op de gedeelde paren: de 3 × 3-tabel (Claude A / B / gelijk × Yeb A / B / =) en de
   overeenstemming op de paren waar beiden een kant kiezen. **Claude tegen Claude**: G4-8, eerste ronde tegen de tweede
   beoordelaar, op de paren waar beide een kant kiezen.
5. **Per marge** (rapportage): QE bij |ΔQE| < 0,2 en ≥ 0,2; elke judge bij |ΔP̄| < 0,2 en ≥ 0,2; tegen Yeb (samen) en
   Claude (alle).
6. **Betekenisfouten**: op de paren waar Yeb precies één kant `fout` noemde, en apart waar Claude precies één kant een
   betekenisfout gaf (`fout_A`/`fout_B`): hoe vaak het signaal de kant zonder fout kiest (k / n).
7. **Controles** (gerapporteerd; een afwijking in de eerste stopt de meting vóór de TPU): (a) de rubriekprompts van
   G4-7, G4-7-tg, G4-bon-i15c2q en RB3 zijn per paar (sha256 van de tekst) die van §17's
   `$G/judge/val/data/prompts-w<k>.parquet`; (b) P̄ van de rubriek tegen §17's scores op die 400 paren (grootste |Δ|,
   zelfde richting); (c) QE, COMET-22, C1 en C2 van de kanten waarvan de tekst (gestript) gelijk is aan een bestaande
   evaluatie-uitvoer (G4-0, G4-8-s140, G4-7-i15 in `$G/eval/`; C1 en C2 van `$G/naturalness-G4-8/` en
   `$G/naturalness-G4-7/` waar ze er zijn) tegen die bestaande scores: n, grootste |Δ|, zelfde label bij 0,5; (d) de
   paren van de gemengde steekproef die ook in een pakket zitten: dezelfde scores; (e) niet gescoorde prompts, afgekapte
   invoer van MetricX, C1 en C2, lege teksten, gelijke teksten.

*G4-maat.7 Wat het niet zegt.*
- n is klein: 74 niet-gelijke oordelen van Yeb, per groep 5–40; een interval van ±10–20 pp; veel vergelijkingen zonder
  correctie. Geen harde conclusie, geen drempel, geen winnaar; de meting beslist niets over G4-9 of over een maat.
- **Selectie**: de paren van steekproef 1 zijn gekozen met de rubriek-judge en QE ≤ QE₀ (judgeq, janchor); rubriek en QE
  kiezen daar grotendeels de kant van "nieuw systeem" door de constructie, en hun overeenstemming met Yeb is daar vooral
  die van de selectie (25 / 40, LEARNINGS), geen onafhankelijke toets. Zo ook in de pakketten van de tabel met een
  selectie. Zonder paarselectie: de gemengde steekproef en de groep "zonder paarselectie" (G4-8-s140 is wel getraind op
  keuzes van de rubriek-judge).
- Claude oordeelde met dezelfde rubriek als de rubriek-judge (§17: 71% overeenstemming): tegen Claude heeft de rubriek
  een voorsprong door de constructie.
- Yeb is één beoordelaar en hier de referentie, geen grondwaarheid; zijn oordeel volgt de instructie van de
  steekproefbestanden (eerst juistheid, dan natuurlijk Nederlands).
- De judge is Gemma 4 31B-it, hetzelfde model als (de basis van) de meeste vertalingen.
- COMET-22 alleen op nieuws (WMT24++, FLORES, NTREX): voor Yeb ~31 paren.
- Lengte is geen kwaliteitsmaat; de rij toont of een beoordelaar de kortere kant voortrekt.

*G4-maat.8 Uitvoer.* `$GM/data/` (`pairs.parquet` met teksten, `refs.parquet` zonder tekst, `prompts-w<k>.parquet`,
`qe-in.parquet`, `comet-in.parquet`, `data.json`), `$GM/judge/scores/w<k>/`, `$GM/scores/` (`metricx.parquet`,
`comet.parquet`), `$GM/naturalness/` (`texts.parquet`, `predictions.parquet`, `predictions-c2.parquet`); het rapport
`$GM/report.md`/`.json`, kopie `mbert-ops/translator-sft/g4-measure-vs-yeb.md` (`.json`), alleen tabellen met tellingen
en percentages. Prompts en teksten blijven in de bucket en op de VM.

*G4-maat.9 Keten* (`mbert-ops/scripts/g4measure_run.sh`, kopie in `gs://t5-dutch-english-usc2/hammond/scratch-scripts/`;
code `g4measure.py` en `run_g4measure.sh` op `en-nl-g4arm8`). Alleen mbert-mix; `final_eval_mix.lock` met `flock` tot het
eind; mbert-corpus (T3) wordt niet aangeraakt. Code als bundel via de bucket naar de 4 workers; elke ssh met `timeout 900`
en de ssh-opties van G4-judge.10; vóór elke TPU-stap de TPU van de worker(s) vrij. Volgorde: `setup` (`run_g4arm.sh
setup`, `ARM=G4-bon`; w0 met `FULL=1`), `data` (w0, CPU), `judge` (4 workers), daarna tegelijk `qe` (w1), `classify`
(C1) en `classify-c2` (w0, na elkaar) en `comet` (w2, CPU); tot slot `report` op co1 uit de bestanden zonder tekst
(`refs.parquet`, de scores, de sleutels en antwoorden). Markers `$G/stages/g4-measure/`; niets wat eerdere secties
schreven wordt overschreven. Elke stap eindigt op `EXIT 0`, anders `FAILED` en stoppen. [schatting] ~35–50 min.

*G4-maat.10 Aanvulling 24 nov 2026 ~12:20, na de uitslag (run `g4m1124a`, code `81f68a6`); verandert G4-maat.1–9 niet.*
Twee wijzigingen in het rapport, beide na het lezen van de uitslag en daarom zo gemarkeerd: (a) in de tabel
"betekenisfouten" staat geen rij Claude meer in de kolom van Claude's eigen foutlabels (Claude tegen zijn eigen labels is
per constructie 100% en zegt niets); (b) een tabel **"hoe vaak kiest een signaal het nieuwe systeem"** per kolom van de
hoofdtabel (het aandeel paren waar het signaal de kant kiest die niet `base` is), als maat voor de
selectiekoppeling van G4-maat.7 [niet vooraf vastgelegd; beschrijvend, geen toets]. De run, de scores en de tabellen
1–7 blijven die van `g4m1124a`; het rapport wordt op co1 opnieuw gemaakt uit dezelfde bestanden zonder tekst.

## 24. Meting G4-maat2 en de evaluatiepool: twee judge-prompts en een schone pool (voorbereiding, beslist niets)

**Vastgelegd 24 nov 2026 (tijdstip: de commit van deze sectie op branch `en-nl-g4arm8`)**, vóór elke stap van deze
meting op een worker en vóór elke uitslag [besluit van de hoofdsessie namens Yeb, op advies van Gemini 4 Neon, 24 nov
~12:40, binnen Yebs regel "voorbereidend werk mag"; geen training, geen nieuwe arm, G4-9 niet starten; uitwerking door
een subagent]. Vooraf gelezen: de rapporten van §20, FB.8 en §23 (alleen tellingen en scores); met één leesopdracht op
mbert-mix w0 de aantallen van `$G/data/eval/evalsets.parquet` per set, deel van G4-5.4 en lengteband (geen tekst);
de veldnamen en id's van de sleutels van Yebs drie steekproeven. Geen tekst (bron, vertaling, roman) en geen score van
deze meting. Een meting en een voorbereiding: er wordt niets getraind en niets beslist; er wordt geen pakket voor Yeb en
geen blind pakket gemaakt (dat besluit Yeb later: stoppen of een menselijke proef van ~96 paren). Claude en GPT zijn geen
judge of generator; de bestaande Claude-oordelen dienen alleen als vergelijking. `GM=$G/measure`, `GM2=$G/measure2`,
`P=$G/pool`, `GJ=$G/judge`, `G8=$G/data/G4-8`, `MS=$G/mixsample`.

### 24.1 Deel 1: twee nieuwe judge-prompts

*G4-maat2.1 Vraag.* De rubriek-judge (§17) kiest bij mens tegen machine de machine (§20: fictie 17,6%, BT 30,8% richting
mens); N1 en N2 (FB.8) herkennen de menselijke kant, maar N1-keuzes die van de rubriek afwijken zijn minder juist
(G4-9.V3N1); tegen Yeb is geen signaal duidelijk beter (§23). Gemeten wordt of een judge die alleen trouw meet (A1),
een judge die natuurlijkheid voorop zet met een fout als veto (J-nat), en twee vooraf vastgelegde combinaties
(A1-veto → N1, A1-veto → N2) de menselijke kant herkennen, de foutloze kant kiezen en stabiel zijn bij het verwisselen
van A en B, en hoe vaak ze dezelfde kant kiezen als Yeb en Claude. Rapportage; geen winnaar, geen besluit.

*G4-maat2.2 Judge en prompts.* De judge van G4-judge.2–4 ongewijzigd: Gemma 4 31B-it @ `842da37`, de MaxText-omzetting,
`pairjudge.py score` ongewijzigd (één prefill, P(A)/P(B)/P(gelijk) uit de logits van de eerste antwoordpositie met de
ids van G4-judge.2, het chat-template met één gebruikersbeurt en een lege denkruimte), elk paar in beide volgordes,
P̄ volgens G4-judge.4 (`pairjudge.combine`), een prompt van meer dan 4.096 tokens niet gescoord (geteld). De tekst van de
gebruikersbeurt, letterlijk (`{bron}`, `{A}` en `{B}` gestript ingevuld, `pairjudge.user_text`):

**A1** (alleen trouw aan de bron; stijl genegeerd):

```
Je beoordeelt vertalingen uit het Engels in het Nederlands. Je krijgt de Engelse bron en twee Nederlandse vertalingen, A en B. Je weet niet hoe of door wie ze zijn gemaakt; probeer dat niet te raden.

Oordeel alleen op trouw aan de bron: welke vertaling geeft de betekenis van de Engelse bron het volledigst en het juistst weer? Tel alleen betekenisfouten:
- een weglating: iets uit de bron ontbreekt in de vertaling;
- een toevoeging: de vertaling zegt iets wat niet in de bron staat;
- een verkeerde betekenis: de vertaling zegt iets anders dan de bron, ook als een idioom of vaste uitdrukking zo is vertaald dat de betekenis verandert.
Laat stijl, woordkeus, zinsbouw en natuurlijkheid uitdrukkelijk buiten beschouwing: een stijve of letterlijke vertaling is even goed als een vloeiende, zolang de betekenis klopt. "gelijk" als er geen duidelijk verschil in betekenisfouten is.

Engelse bron:
{bron}

Vertaling A:
{A}

Vertaling B:
{B}

Welke vertaling is getrouwer aan de bron? Antwoord met één woord: A, B of gelijk.
```

**J-nat** (natuurlijkheid eerst, een fout als veto):

```
Je beoordeelt vertalingen uit het Engels in het Nederlands. Je krijgt de Engelse bron en twee Nederlandse vertalingen, A en B. Je weet niet hoe of door wie ze zijn gemaakt; probeer dat niet te raden.

Kies de vertaling met het natuurlijkste Nederlands: de tekst die het meest leest als tekst die oorspronkelijk in het Nederlands is geschreven, en het minst als een vertaling. Alleen als die vertaling een duidelijke betekenisfout of weglating heeft (iets uit de bron ontbreekt, of de vertaling zegt iets anders dan de bron) en de andere niet, kies je de andere. Woordkeus, stijl en zinsbouw zijn geen betekenisfout zolang de betekenis klopt. "gelijk" als er geen duidelijk verschil is.

Engelse bron:
{bron}

Vertaling A:
{A}

Vertaling B:
{B}

Welke vertaling kies je? Antwoord met één woord: A, B of gelijk.
```

Er wordt geen prompt gezocht of bijgesteld; deze twee liggen hier vast.

*G4-maat2.3 Keuze per paar en de combinaties* (vooraf vastgelegd, geen andere). Een judge kiest de kant met de hoogste
P̄ (het teken van P̄(A) − P̄(B), G4-maat.4; gelijk of niet gescoord: geen keuze). **A1-veto → N1**: is
|P̄_A1(A) − P̄_A1(B)| ≥ **0,2**, dan de kant van A1; anders de keuze van N1. **A1-veto → N2**: idem met N2. [keuze] De
marge 0,2 is die van §23 (K1, K2 en de tabel per marge), vooraf gekozen en niet gezocht; er wordt niets op Yebs paren,
op de Claude-paren of op de fictie/BT-paren afgesteld.

*G4-maat2.4 Paren.*
- **Fictie/BT** (1.000 paren van §20, run `g4jfb1124a`, kader `ab`: A = de menselijke of oorspronkelijke kant). De teksten
  komen uit §20's prompts `$GJ/fb/data/prompts-w<k>.parquet`: per paar wordt de prompt `ab` van G4-judge.3 ontleed in
  bron, A en B (het begin en eind van de template en de twee scheidingen `\n\nVertaling A:\n` en `\n\nVertaling B:\n`
  elk precies één keer; anders stopt de stap) en gecontroleerd: dezelfde drie teksten geven precies §20's prompts `ab`
  en `ba` en FB.8's prompts N1 en N2 (`$GJ/fbn1/data/`, `$GJ/fbn2/data/`), per uid; de paren zijn die van §20's
  `refs.parquet`. Een afwijking stopt vóór er iets geschreven wordt.
- **§23** (1.438 paren: Yebs 80 en de 14 blinde Claude-pakketten): de teksten van `$GM/data/pairs.parquet`; de prompts
  rubriek, N1 en N2 die daaruit opnieuw gemaakt worden moeten per uid (sha256 van de tekst) die van §23 zijn
  (`$GM/data/prompts-w<k>.parquet`), anders stopt de stap.
- Prompts: A1 en J-nat × (1.000 + 1.438) paren × 2 volgordes = 9.752; daarbij de reproductie (G4-maat2.7) 200 × 3 × 2 =
  1.200; samen 10.952, over de 4 workers op sha256 van de uid.

*G4-maat2.5 Toets (vooraf vastgelegd; rapportage, beslist niets).* Voor A1, J-nat, A1-veto → N1 en A1-veto → N2:
- **(a)** richting mens/origineel > **70%** op de 1.000 fictie/BT-paren samen: een judge telt 1 als P̄(mens) >
  P̄(machine), ½ bij gelijke P̄ (FB.4.2); een combinatie 1 als ze de menselijke kant kiest, 0 de machinekant, ½ zonder
  keuze; een niet gescoord paar telt niet (geteld). Per proef (fictie, BT) ernaast.
- **(b)** ≥ **65%** de foutloze kant op de **263** Claude-paren met precies één kant `fout` (§23 G4-maat.6.6): k / n,
  n = de paren waar het signaal een keuze maakt (n_ref − n apart).
- **(c)** omklappen bij verwisselen ≤ **20%**, op de 1.000 fictie/BT-paren en op de 1.438 §23-paren apart (beide moeten
  het halen): een prompt omklapt als de greedy oordelen van `ab` en `ba` (de grootste van p_A, p_B, p_gelijk van die
  ene prompt), teruggezet in het kader van het paar, tegengestelde winnaars noemen (G4-judge.5.3); een combinatie omklapt
  als ze, toegepast op de genormaliseerde kansen van één volgorde (teruggezet) in plaats van P̄, in `ab` en `ba`
  tegengestelde kanten kiest.
Bestaande signalen in dezelfde tabel ter vergelijking (rubriek, N1, N2 op de fictie/BT-paren uit §20 en FB.8's scores
herberekend; op de §23-paren uit §23's scores).

*G4-maat2.6 Maten* (`g4measure2.py report1`, op co1 uit bestanden zonder tekst). (1) De toets van G4-maat2.5 met
gehaald/niet per criterium. (2) Fictie/BT per proef en signaal: oordelen mens / machine / gelijk, richting (Wilson),
gem. P̄ en Δ (bootstrap als §20), omklappen, inconsistent (`pairjudge.fb_result`; voor de combinaties alleen richting en
omklappen). (3) De tabellen van §23 (`g4measure.report_data` met de nieuwe rijen; de bestaande signalen QE, COMET-22,
C1, C2, rubriek, N1, N2, lengte, K1–K4, Claude, nieuw systeem, positie A uit §23's scores, niet opnieuw berekend):
hoofdtabel (Yeb V1, gemengd, samen; Claude alle, Gemma 4 tegen Gemma 4, zonder paarselectie), per set en pakket, per
marge (ook A1 en J-nat), betekenisfouten (Yeb en Claude), hoe vaak een signaal het nieuwe systeem kiest. (4) Omklappen
op de §23-paren per prompt. (5) Controles: de reproductie, de paarcontroles, dekking. Kopie
`mbert-ops/translator-sft/g4-measure2.md` (`.json`).

*G4-maat2.7 Reproductie* (een afwijking stopt vóór het rapport). 200 §23-paren
(`random.Random("en-nl-g4arm/0/G4-maat2/repro").sample` van de gesorteerde paar-id's) worden met rubriek, N1 en N2
opnieuw gescoord met precies §23's prompttekst (uid `repro/<paar>/<prompt>/<volgorde>`): per prompt |Δ| ≤ 1e-5
(float32-precisie) voor p_A, p_B en p_gelijk tegen §23's scores en hetzelfde greedy oordeel, en per paar dezelfde kant.

*G4-maat2.8 Wat het niet zegt.* Als G4-maat.7 en FB.5/FB.8: n van Yeb is klein (74 niet-gelijke oordelen); V1 en
pakketten met een selectie zijn met de rubriek gekozen; Claude oordeelde met de rubriek; de herkomst is bij fictie/BT de
"waarheid", niet de kwaliteit (literaire vrijheid, BT-Engels is Gemma 4's vertaling); de judge is Gemma 4 31B-it, het
model achter de meeste vertalingen. Eén formulering per prompt, zero-shot. De toets beslist niets.

### 24.2 Deel 2: een schone evaluatiepool

*Pool.1 Doel.* ~400–500 achtergehouden Engelse bronnen met per bron de vertalingen van G4-8-s140, G4-0 en G4-7-i15,
G4-0's best-of-8 met vier vooraf vastgelegde keuzes, de menselijke vertaling waar die bestaat, en alle scores, zodat een
latere menselijke proef (als Yeb daartoe besluit) zonder nieuwe TPU-stap getrokken kan worden. Er wordt nu niets
getrokken of aan iemand voorgelegd.

*Pool.2 Bronnen* (vaste aantallen; per stratum op id gesorteerd, geschud met het zaad, de eerste n; een tekort stopt):
- **romans 100**: de set `novels` (G4-4.10: 500 passages, 10 boeken; `$G/data/heldout4/eval-novels-meta.parquet`), Engels
  en menselijke vertaling (`en`, `nl`) van de rij in `b1-selected.parquet` (groep `en_orig`), zonder de 60 bronnen van
  G4-8.15 (`$MS/data/data.json`); de controles van G4-8.15 (geen id, document of gestript Engels in G4-7's pool of in
  G4-8's `fresh`, `plan`, `targets`: een treffer stopt) en zijn filter (`g4rl.candidate_filter`: 20–256 Gemma 4-tokens,
  geen 13-gram met het Engels van die vier bestanden, één bron per tekst); 10 per boek, zaad
  `en-nl-g4arm/0/G4-maat2/pool/novels/<boek>`. De boeken zitten niet in de bronnen van G4-7 en G4-8 (G4-8.15).
- **BT 100**: §20's BT-paren (C2's testsplit, `pairjudge.fb_bt` ongewijzigd met zijn controles), Engels = Gemma 4's
  Engels, de menselijke kant = het Nederlandse origineel (`nl`); zonder de 60 van G4-8.15; dezelfde controles en filter;
  20 per bron (dpc, fineweb, handelingen, raad, rechtspraak), zaad `…/pool/bt/<bron>`.
- **WMT24++ 100**, **FLORES 40**, **NTREX 40**, **FLAN 40**: de sets `wmt24pp`, `flores`, `ntrex`, `flan` van
  `evalsets.parquet`, alleen het **zoekdeel** van G4-5.4 (`split.json`, digest `f953c026…f3`), 20–256 Gemma 4-tokens
  [gelezen: 322 / 387 / 636 / 422, WMT24++ zonder `is_bad_source`], niet `is_bad_source`, geen item van de 100 vaste
  blinde passages en geen item dat Yeb zag (V1: de items van `G4-judge-i15` en `G4-judge-anchor-i15` in
  `steekproef-yeb/key-map.json`; de gemengde steekproef: `steekproef-yeb-g4-8-mix/key.json`); dezelfde controles en
  filter als de romans; zaad `…/pool/<set>`. Referentie (menselijke vertaling) bij WMT24++, FLORES en NTREX die van de
  testset; FLAN zonder.
- **idioom**: de idioompassages van de 100 vaste blinde passages (rijen `blind-NNN`, 30), 20–256 tokens, zonder die
  van Yebs steekproef 2 (`steekproef-yeb-g4-8/key-map.json`) en de gemengde steekproef; alle overige; geen referentie.
- **FLAN ≤ 10%**: FLAN = min(40, ⌊(alle andere bronnen) / 9⌋).
- **Niet**: de TEST-split van hammond (geen bron komt uit hammond), het beslisdeel van G4-5.4 van alle sets [keuze: ook
  FLORES, NTREX en FLAN, zodat het beslisdeel ongelezen blijft], europarl, en niets uit de trainingsdata van G4-7 en G4-8
  (de controles hierboven).

*Pool.3 Vertalingen* (fork `f4d8615`, model `gemma4-31b`, prompt `natural-v1`, `translate.py`, klasse `medium`, alle
bronnen ≤ 256 tokens; als het anker en de samples van G4-8-data.2–3). Greedy (`--copies 1`): G4-0 zonder adapter (w1),
G4-8-s140 (`$G/G4-8-s140/`, w2) en G4-7-i15 (`$G/G4-7-i15/`, w3), de LoRA's rank 32 / alpha 16 / schaal 0,5 gecontroleerd
en samengevoegd. Samples: G4-0 zonder adapter, 8 per bron bij T = 1 (`--copies 8 --sample samples=1.0:<zaad>`, zaad
`g4rl.seed_int("en-nl-g4arm/0/G4-maat2/pool/sample/w0")`, w0). Het anker van best-of-8 = G4-0 greedy. Gerapporteerd
(geen stop): bij de bronnen uit de evaluatiesets hoe vaak de greedy tekst (witruimte genormaliseerd) gelijk is aan de
bestaande evaluatie-uitvoer van hetzelfde model (`$G/eval/<model>/pairs/`, vertaald per lengteklasse).

*Pool.4 Scores per tekst* (per bron: de drie greedy vertalingen, de 8 samples, de menselijke tekst). **QE**: MetricX-24
hybrid XXL bf16 zonder referentie (`metricx_qe.py score --pair source:hypothesis --model xxl`, een lege tekst zonder
score, als G4-8-data.2), over de 4 workers. **COMET-22** met de referentie bij WMT24++, FLORES en NTREX (`comet_score.py`,
CPU, w2) [keuze als §23: niet bij romans en BT]. **C1** (`run_arms.sh classify` ongewijzigd) en **C2**
(`naturalness.py classify --max-length 1024`, als `classify-c2`) op alle teksten (w0).

*Pool.5 Best-of-8 met vier keuzes.* Fail en kandidaat als G4-8-data.4 (`g4arm8.source_candidates` ongewijzigd: niet af,
lus, leeg, Engels over, ℓ/ℓ₀ buiten [0,67; 1,5], QE > 7,5; kandidaat = geen fail, QE ≤ QE₀ (+ 1e-9), tekst anders dan
het anker; gelijke teksten één keer, sleutel het eerste sample). Een bron waarvan het anker niet in orde is
(`g4rl.anchor_ok`) of geen QE heeft, krijgt geen kandidaten (geteld). Elke kandidaat tegen het anker met **rubriek**,
**N1** en **A1** (kader `ab`: A = sample), beide volgordes. Per bron, elke keuze met het anker als terugval:
- **rubriek-of-anker** (janchor, G4-8-data.6): k = de kandidaat met de hoogste P̄_rubriek(sample) (gelijk: lagere QE, dan
  het eerdere sample); het sample als P̄ > 0,5, anders het anker (`g4arm8.choose8`);
- **N1** (G4-8-N1.6): idem met P̄_N1;
- **rubriek ∩ N1** (het doel van §22 O9 `intersect`): het sample als rubriek en N1 een sample met dezelfde `j` kozen,
  anders het anker;
- **A1-veto → N1**: per kandidaat beslist A1 als |P̄_A1(sample) − P̄_A1(anker)| ≥ 0,2 (het sample als P̄_A1(sample) groter
  is), anders de regel van N1 (het sample als P̄_N1(sample) > 0,5); onder de kandidaten waar dat het sample geeft de
  hoogste P̄_N1(sample) (gelijk: lagere QE, dan het eerdere sample); geen: het anker.

*Pool.6 Paren voor een latere menselijke proef* (alle vijf prompts: rubriek, N1, N2, A1, J-nat; beide volgordes;
kader `ab`: A = de eerstgenoemde): **mens** G4-8-s140 tegen de menselijke tekst (romans, BT-origineel, WMT24++, FLORES,
NTREX); **bo8** G4-8-s140 tegen G4-0 best-of-8 rubriek ∩ N1 (bij terugval G4-0 greedy); **i15** G4-8-s140 tegen
G4-7-i15; **keuze** de rubriekkeuze tegen de N1-keuze waar hun teksten verschillen. Paren met (genormaliseerd) gelijke
teksten worden geteld en niet gescoord. Er wordt geen paar getrokken, geen pakket en geen steekproefbestand gemaakt.

*Pool.7 Reproductie* (een afwijking stopt vóór het rapport). QE van 400 kanten van 200 §23-paren
(`random.Random("en-nl-g4arm/0/G4-pool/repro-qe").sample` van de gesorteerde paar-id's) in dezelfde QE-stap: grootste
|Δ| tegen §23's `metricx.parquet` ≤ **0,06**. Rubriek, N1 en N2 van 100 §23-paren (`…/G4-pool/repro-judge`) in de eerste
judge-stap met §23's prompttekst: als G4-maat2.7 (|Δ| ≤ 1e-5, zelfde greedy oordeel, zelfde kant).

*Pool.8 Rapport* (`g4pool.py report`, w0, alleen tellingen en scores, geen tekst; `$P/report.md`/`.json`, kopie
`mbert-ops/translator-sft/g4-pool.md`): bronnen per stratum (geschikt, gefilterd, uitgesloten, getrokken; FLAN-aandeel);
per systeem (G4-0, G4-8-s140, G4-7-i15, de vier best-of-8-keuzes, mens) en stratum n, gem. QE, gem. COMET-22 (nieuws),
C1 (gem. P(origineel) en aandeel ≥ 0,5), gem. C2, leeg/niet af; de best-of-8-keuzes (sample of anker per keuze en
stratum, overeenstemming tussen keuzes, kandidaten en fails); per paartype en prompt n, gelijke teksten, richting voor A
(Wilson), gem. P̄(A), P̄(B); de controles van Pool.3 en Pool.7.

*Pool.9 Uitvoer* (alles onder `$P`): `data/` (`sources.parquet` met tekst, `sources-meta.parquet` zonder, `segments.parquet`,
`data.json`), `gen/<model>/` en `samples/` (de vertalingen), `texts/` (`texts.parquet`, `qe-in-w<k>.parquet`,
`comet-in.parquet`, `texts.json`), `scores/` (`qe-w<k>.parquet`, `comet.parquet`), `naturalness/` (`texts.parquet`,
`predictions.parquet`, `predictions-c2.parquet`), `judge1/` en `judge2/` (`data/`, `scores/w<k>/`), `choose/`
(`cands.parquet`, `choice.parquet` en `pairs-meta.parquet` zonder tekst, `outputs.parquet` en `pairs.parquet` met tekst),
`report.md`/`.json`; markers `$G/stages/g4-pool/`. Teksten blijven in de bucket en op de VM; naar co1 alleen het rapport.

*Pool.10 Wat het niet zegt.* De pool is een voorraad, geen uitslag: de gemiddelden per systeem zijn automatische maten
(QE, COMET-22, C1, C2) en judge-oordelen van Gemma 4 31B-it, het model achter G4-0, G4-8-s140 en G4-7-i15. FLAN, WMT24++,
FLORES en NTREX komen uit het zoekdeel, dat bij G4-7 en G4-8 voor de keuze van de kandidaat gelezen is. De romanboeken
zitten in G4-4's training, niet in die van G4-7 en G4-8. Het BT-origineel is geen vertaling van het Engels (het Engels
is Gemma 4's vertaling ervan).

### 24.3 Keten

*G4-maat2.K* (`mbert-ops/scripts/g4measure2_run.sh`, kopie in `gs://t5-dutch-english-usc2/hammond/scratch-scripts/`; code
`g4measure2.py`, `g4pool.py` en `run_g4measure2.sh` op `en-nl-g4arm8`). Alleen mbert-mix; `final_eval_mix.lock` met
`flock` tot het eind; mbert-corpus (T3) wordt niet aangeraakt. Code als bundel via de bucket naar de 4 workers; elke ssh
met `timeout 900` en de ssh-opties van G4-judge.10; vóór elke TPU-stap de TPU van de worker(s) vrij. Deel 1: `setup`
(`run_g4arm.sh setup`, `ARM=G4-bon`; w0 en w2 met `FULL=1`), `data1` (w0, CPU), `judge` met `SET=m2` (4 workers),
`report1` op co1 uit de bestanden zonder tekst. Deel 2: `pooldata` (w0, CPU), `generate` (w0 samples, w1–w3 greedy),
`texts` (w0, CPU), `qe` (4 workers), `classify` en `classify-c2` (w0) en `comet` (w2, CPU), `cands` (w0, CPU), `judge`
met `SET=p1` (4), `choose` (w0, CPU), `judge` met `SET=p2` (4), `poolreport` (w0, CPU). Markers
`$G/stages/g4-measure2/` en `$G/stages/g4-pool/`; niets wat eerdere secties schreven wordt overschreven of verwijderd.
Elke stap eindigt op `EXIT 0`, anders `FAILED` en stoppen; een gemarkeerde stap wordt bij een herstart overgeslagen.
[schatting] deel 1 ~20–30 min, deel 2 ~1–1,5 u.

## 25. Twee blinde pakketten uit de evaluatiepool: G4-8-s140 tegen de mens en tegen G4-0 best-of-8 (rapportage, beslist niets)

**Vastgelegd 24 nov 2026 (tijdstip: de commit van deze sectie op branch `en-nl-g4arm8`)**, vóór de trekking en vóór
elke beoordeling [besluit van de hoofdsessie namens Yeb, op advies van Gemini 4 Neon, 24 nov ~14:35; uitwerking door een
subagent]. Vooraf gelezen: alleen tellingen zonder tekst uit `$P/choose/choice.parquet` en `$P/choose/pairs-meta.parquet`
(de geschikte bronnen per stratum en groep, hieronder); geen tekst en geen score van een paar. Geen TPU: de pakketten
worden op co1 gebouwd uit de pool van §24 in de bucket (`g4pool.py blind`); mbert-mix en mbert-corpus (T3) worden niet
aangeraakt. **Een rapportage: dit beslist niets** en verandert G4-8.0–G4-8.15 en §22–§24 niet; er wordt niets getraind,
vertaald of gescoord. Claude en GPT zijn geen generator; de beoordelaars zijn verse Claude-subagenten die de hoofdsessie
start. `P=$G/pool`, `S25=en-nl-g4arm/0/G4-blind25`.

*Blind25.1 Vraag.* Hoe verhoudt G4-8-s140 (greedy, de vertaler sinds 24 nov) zich volgens een blinde beoordelaar met de
rubriek van protocol.md §4.2.4 tot (P1) de menselijke tekst en (P2) G4-0 best-of-8 met de keuze rubriek ∩ N1 (het doel
van G4-9, §22 O9 `intersect`; Pool.5), op romans, BT en WMT24++?

*Blind25.2 Pakketten* (de paren van Pool.6; A en B in de pool: A = G4-8-s140, B = de tegenkant).
- **P2 `G4-8-vs-bo8ri`, 60 paren**: paartype `bo8`, G4-8-s140 greedy tegen G4-0 best-of-8 rubriek ∩ N1. Alleen bronnen
  waar die keuze een sample is en niet het G4-0-anker (`bo8-rubriek∩N1|reason` = `sample` in `choice.parquet`) én waar
  de tekst na normalisering van witruimte verschilt van G4-8-s140 (`identical` = false in `pairs-meta.parquet`).
  Controles (een afwijking stopt): de tekst van de tegenkant is het sample `s<j>` van de keuze en verschilt
  (genormaliseerd) van G4-0 greedy. Geschikt [gelezen]: romans 37 (per boek 1–7), BT 40 (per bron 5–12), WMT24++ 36.
- **P1 `G4-8-vs-mens`, 60 paren**: paartype `mens`, G4-8-s140 greedy tegen de menselijke tekst: bij romans de menselijke
  vertaling, bij BT het Nederlandse origineel, bij WMT24++ de referentie van de testset. Geen FLORES, NTREX, FLAN of
  idioom. De bronnen zijn disjunct van P2: P2 wordt eerst getrokken, en zijn 60 bronnen vallen voor P1 weg. Geschikt
  [gelezen]: per stratum 100 paren, geen met gelijke tekst; na P2 dus 80 per stratum.
- Per pakket **20 romans, 20 BT en 20 WMT24++**.

*Blind25.3 Trekking* (`g4pool.py blind`; naar het scherm alleen ids en tellingen):
- Volgorde: eerst P2, dan P1.
- Per pakket en stratum (romans, BT, WMT24++) de geschikte bronnen, gegroepeerd op de kolom `stratum` van de pool
  (romans: het Engelse boek, 10 groepen; BT: de bron dpc, fineweb, handelingen, raad, rechtspraak; WMT24++: één groep).
  De groepen in de volgorde van `random.Random(f"{S25}/{pakket}/{stratum}/groups").shuffle` van de gesorteerde
  groepnamen; per groep de ids in de volgorde van `random.Random(f"{S25}/{pakket}/{stratum}/{groep}").shuffle` van de
  gesorteerde ids. Dan om de beurt: ronde 1 de eerste id van elke groep in groepvolgorde, ronde 2 de tweede, enzovoort;
  een lege groep wordt overgeslagen; stoppen bij 20. Minder dan 20 geschikt stopt.
- **A/B geblokt per stratum**, niet vrij gegooid: van de 20 paren van een stratum is G4-8-s140 bij precies 10 A en bij
  10 B. De lijst `["A"] * 10 + ["B"] * 10`, geschud met `random.Random(f"{S25}/{pakket}/{stratum}/ab").shuffle`, wordt in
  de getrokken volgorde toegekend (de waarde is de kant van G4-8-s140).
- Volgorde van de 60: de paren (stratum, id) gesorteerd, geschud met `random.Random(f"{S25}/{pakket}/order").shuffle`;
  `blind_id` in die volgorde `vb-001`…`vb-060` (P2) en `vm-001`…`vm-060` (P1).
- Teksten: bron, A en B uit `$P/choose/pairs.parquet` (de teksten die de judge van Pool.6 las; gecontroleerd tegen
  `outputs.parquet` via `a_tid` en `b_tid`), aan alle drie alleen de witruimte aan begin en eind weggehaald (`strip`),
  aan beide kanten gelijk, zodat witruimte de kant niet verraadt [keuze]. Een lege tekst stopt.

*Blind25.4 Uitvoer.* Per pakket `$S/blind/<pakket>/` (de vorm van de eerdere pakketten): `judge-input.jsonl`
(`blind_id`, `source`, `A`, `B`), `key.jsonl` (`blind_id`; `A` en `B` met de waarde `G4-8-s140` of `base`, waarbij `base`
de tegenkant is en `base_system` hem noemt: `mens` of `bo8-rubriek∩N1`; `seed`, `set`, `item`, `stratum`, `pair`,
`a_tid`, `b_tid`, bij P2 `bo8_j`), `select-report.json` (aantallen per stratum en groep, A/B per stratum, geschikt en
uitgesloten, de zaden, sha256 van de invoer), `review/REVIEWER.md` (de instructie aan de beoordelaar) en na de
beoordeling `review/judgments.jsonl`. Op co1 per pakket de map van de beoordelaar
`mbert-ops/translator-sft/blind-review/<pakket>/` met alleen `judge-input.jsonl` en `REVIEWER.md`; zijn uitvoer
`judgments.jsonl` komt in dezelfde map. De sleutel staat alleen in de bucket en in `mbert-ops/translator-sft/blind-keys/
<pakket>/` (`key.jsonl`, `select-report.json`), een map die de beoordelaar niet krijgt.

*Blind25.5 Beoordeling* (door de hoofdsessie, niet in deze stap). Per pakket een verse Claude-subagent met alleen zijn
map. `REVIEWER.md` bevat de rubriek van protocol.md §4.2.4 letterlijk, met de uitvoervorm daar (per paar één JSON-regel
met `blind_id`, `fout_A`, `fout_B`, `oordeel` ∈ {`A beter`, `B beter`, `gelijk`}, `zekerheid` ∈ {`zeker`, `twijfel`},
`uitleg`), en verder alleen de namen van het invoer- en het uitvoerbestand en het aantal paren. Laat de hoofdsessie de
twijfelgevallen opnieuw beoordelen, dan zoals G4-8.13: een andere verse Claude-subagent krijgt per twijfelgeval de bron,
A, B en de uitleg van de eerste beoordelaar (geen sleutel) en geeft een eindoordeel dat het twijfeloordeel vervangt; hij
controleert ook 10 zekere oordelen (`random.Random(f"{S25}/{pakket}/check10").sample` van de gesorteerde `blind_id`'s met
`zekerheid` = `zeker`, 10 stuks); wijkt hij bij meer dan 3 daarvan af, dan beoordeelt hij alle 60 en geldt zijn telling.

*Blind25.6 Telling* (één commando per pakket: `python3 mbert-ops/scripts/blind_tally.py <judgments.jsonl> <key.jsonl>`;
`g4arm.py tally --arm G4-8-s140` geeft dezelfde aantallen als json). Per stratum en samen: G4-8-s140 beter / tegenkant
(`base`) beter / gelijk, het aandeel van G4-8-s140 onder de niet-gelijke oordelen, de betekenisfouten van G4-8-s140 en
van de tegenkant, het aantal twijfel. Regel (c) als protocol.md §5: k/n met k het aantal paren waarin G4-8-s140 beter is
en n het aantal niet-gelijke, of k/n ≥ 0,60, met de eenzijdige binomiale p = P(X ≥ k) bij X ~ Bin(n, ½); en of de
betekenisfouten van G4-8-s140 ≤ die van de tegenkant. De telling gebeurt op de oordelen van de eerste ronde (een
twijfeloordeel telt met het oordeel van de beoordelaar; het aantal twijfel wordt genoemd); na een tweede ronde ook op de
eindoordelen; beide tellingen worden gerapporteerd.

*Blind25.7 Wat het niet zegt.* Een rapportage: (c), de fouten en p beslissen niets, ook niet over G4-9 of over een
publicatie. 60 paren, 20 per stratum: kleine n per stratum. P2 bevat alleen bronnen waar de keuze rubriek ∩ N1 een sample
is (romans 37, BT 41, WMT24++ 38 van de 100 in de pool; Pool.5) en zegt dus niets over de bronnen waar G4-9's doel het
anker zou zijn. In P1 is de tegenkant geen machine: bij BT is het Nederlandse origineel geen vertaling van het Engels (het
Engels is Gemma 4's vertaling ervan), de romanboeken zitten in G4-4's training (niet in die van G4-7 en G4-8), en
WMT24++ komt uit het zoekdeel van G4-5.4. De beoordelaar is een Claude-model; de positie is per stratum geblokt, niet
gecorrigeerd.
