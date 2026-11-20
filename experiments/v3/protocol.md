# dmbert v3: protocol

Opgesteld 20 november 2026 (subagent, opdracht van de hoofdsessie na Yebs besluiten van 20 nov ~10:45). Deel 1 (de
replay-toets van de stabiliteitsfix) is gecommit en gepusht vóór de start van de replay; de keten pint die commit.
Deel 2 (de vormablatie) volgt in een latere commit en wordt niet gestart zonder Yebs akkoord.

Markering: **[citaat]** letterlijk uit de bron, **[parafrase]** samengevat uit de bron, **[gemeten]** door mij gemeten
in deze opdracht, **[afgeleid]** eigen gevolgtrekking.

## Deel 1: replay-toets van de stabiliteitsfix (vastgelegd vóór de start)

### Vraag

Blijft het recept van dmbert v2 (T2's recept op de opt-out-FineWeb-2) in bf16 stabiel over het stuk 250.000–350.000,
waar v2 in bf16 drie bursts van niet-eindige stappen had, als (1) query, key, RoPE en de attention-logits in float32
rekenen en (2) een z-loss van 1e-4 op de MLM-logits meetelt? En kost dat MLM-verlies?

### Achtergrond: de bursts van v2 [gemeten, trainerlog van v2 `gs://…/mbert/v2-fw2optout-230m/logs/w0.log`]

| stuk van v2 | precisie | gestart vanaf | niet-eindige stappen |
|---|---|---|---|
| 0 → 255.200 | bf16 | 0 | 31 tot 250.000 (losse en clusters bij 196.680–196.960 en 238.440–238.860); burst vanaf 251.220/252.620, tot 780 in 1.000 stappen, 30 op rij |
| 250.000 → 270.000 | float32 | checkpoint 250.000 | 0 |
| 270.000 → 286.200 | bf16 | checkpoint 270.000 | cluster bij 275.560; burst vanaf 281.530, tot 694 in 1.000 stappen, 38 op rij |
| 280.000 → 320.000 | float32 | checkpoint 280.000 | 0 |
| 320.000 → 345.000 | bf16 | checkpoint 320.000 | 12 in 70 stappen bij 340.220; burst vanaf 344.170, tot 567 in 1.000 stappen |
| 340.000 → 1.306.000 | float32 | 340.000, daarna 350.000 | 0 |

Staptijd (mediaan van de Step-regels, 250k–350k): bf16 0,220 s, float32 0,301–0,308 s.

### Wat de fix verandert [afgeleid uit de code van `final-v2` @ `494d4d9`]

In v2's bf16-pad was al float32: de LayerNorms (en dus de residual stream), de softmax van de attention (de logits
werden na de einsum naar float32 gecast, de maskers zijn float32), de som van de values, de MLM-logits en de loss.
In bf16 rekenden: de qkv-projectie, de RoPE-cos/sin (bf16 bewaard, de rotatie zelf in float32 en daarna terug naar
bf16), de einsum query·key met bf16-uitvoer, de schaal 1/√d, de out-projectie, de MLP en de dense-laag van de head.

- `--attention-float32` (`MMBertConfig.attention_float32`): query en key gaan direct na de qkv-projectie naar
  float32, de RoPE-cache is float32, de einsum query·key en de schaal rekenen in float32. De qkv-projectie zelf, de
  out-projectie en de MLP blijven bf16. De matmul-precisie blijft de standaard van JAX op TPU (`DEFAULT`, bf16-passes
  met float32-accumulatie), net als in v2's float32-stukken, die stabiel waren. **[gemeten, CPU]** Op één
  attention-laag met exacte bf16-query en -key en logits tot 266 is de grootste fout in de kansen 0,067 zonder en
  2,4·10⁻⁶ met de vlag (gelijk aan het float32-model); in het hele model (4 lagen, query/key-gewichten ×10, ×40, ×100)
  daalt de gemiddelde fout van de kansen per laag met 14–59% (`tests/test_v3_flags.py`).
- `--z-loss 1e-4`: het doel wordt MLM-verlies + 1e-4 · gemiddelde over de gemaskeerde posities van log²Z, met Z de
  noemer van de softmax over de 32.128 MLM-logits. De gelogde `Loss` blijft het MLM-verlies; de Step-regel krijgt
  `| Z: <gemiddelde log²Z>` erachter.
  - **Waarom 1e-4.** PaLM, Chowdhery et al. 2022, *PaLM: Scaling Language Modeling with Pathways*, arXiv:2204.02311,
    §5 "Training Setup", punt "Loss function" **[citaat]**: "We additionally use an auxiliary loss of z_loss =
    10−4 · log2 Z to encourage the softmax normalizer log(Z) to be close to 0, which we found increases the stability
    of training." Wortsman et al. 2023, *Small-scale proxies for large-scale Transformer training instabilities*,
    arXiv:2309.14322, figuur 3 en 4 **[parafrase]**: met z-loss-coëfficiënt 1e-4 verdwijnt de instabiliteit door
    "output logit divergence" in kleine modellen bij hoge LR. **[afgeleid]** De term kost het MLM-verlies vrijwel
    niets: de decoder-bias kan alle logits samen verschuiven, wat log Z verandert zonder de softmax te veranderen.
    Is log²Z bij stap 250.000 ~100, dan is de term ~0,01 (1% van het verlies) en zijn gradiënt ~0,3% van die van de
    cross-entropy. Een grotere coëfficiënt is niet nodig om het effect te zien en zou meer afwijken van het recept.
- `--nonfinite-diagnostics` (alleen diagnose): na elke `NONFINITE`-regel een regel `NONFINITE_GROUPS` met de
  laagtypen (lm_head, final_layer_norm, `encoder_<i>_ffn`, `encoder_<i>_attention`, embeddings) waarvan een gradiënt
  NaN/inf was, van de loss terug geordend. **[gemeten, CPU]** Verandert geen update (bit voor bit gelijk aan v2).
- Zonder deze vlaggen zijn model en trainstap bit voor bit die van v2 (fixture gemaakt met de code van `final-v2`
  @ `494d4d9`, `tests/fixtures/v2_reference.npz`; logits, metrics en parameters na twee stappen, in float32 en bf16).

### Startpunt

v2's checkpoint 250.000 stond niet meer in `ckpt/` (rotatie van de laatste drie, 16 nov 14:33 UTC) en niet in
`ckpt_keep/` (alleen elke 100.000e). Het stond in de soft delete van de bucket (bewaartermijn 7 dagen, tot 23 nov
14:33 UTC). **[gemeten]** Hersteld op 20 nov met `gcloud storage restore`: van elk van de 29 objecten de laatste
generatie (die bij de rotatie werd verwijderd), op zijn oude plek `gs://…/mbert/v2-fw2optout-230m/ckpt/250000/`
(daar stond niets meer, dus niets overschreven); grootte en md5 gelijk aan de verwijderde generaties. De datastand
`ckpt_data_state/250000/` stond er nog. De keten kopieert beide naar elke run-map van de replay en verandert niets in
de map van v2.

### Armen (alle op mbert-mix, met v2's data, staging en trainercommando)

Elke arm is het commando van `experiments/final-v2/start_v2.sh main` (bf16, WSD tot 1.306.000, constante LR 4,8e-4
over dit hele stuk, `--skip-nonfinite-updates`, dezelfde bestanden per host, `--url-blocklist hebban.nl`, EVAL elke
25.000 stappen op `kept/test`), met alleen de genoemde vlaggen, een `--stop-at-step` en eigen namen. Run-mappen onder
`gs://t5-dutch-english-usc2/mbert/v3-replay/<run>/`.

1. **smoke** (`v3-replay-smoke-<TAG>`): fix aan, 250.000 → 250.300. Eisen: hervat op 250.000 met de datastand op elk
   proces; Step-regels 250.050–250.300 eindig, geen `NONFINITE`; per Step-regel het verlies binnen 0,02 en gemiddeld
   binnen 0,01 van v2's float32-stuk op dezelfde stap (zelfde batches, zie hieronder); `Z` gelogd; mediane staptijd
   ≤ 0,35 s; eindcheckpoint 250.300 geschreven. Faalt een eis, dan stopt de keten (`FAILED`) en start niets.
2. **zf** (`v3-replay-zf`): fix aan (`--attention-float32 --z-loss 1e-4 --nonfinite-diagnostics`), in drie stukken:
   250.000 → 270.000, 270.000 → 320.000, 320.000 → 350.000.
   - **Waarom hervatten op 270.000 en 320.000.** De trainer seedt de maskering bij elke start met 42 + procesindex
     (`np.random.seed(42 + process_idx)`), en de datastand legt vast welke documenten volgen. **[afgeleid]** Een start
     vanaf dezelfde stap geeft dus dezelfde batches (documenten én maskers) zolang de koppeling host → procesindex
     gelijk is. Daarmee ziet zf: op 250k–270k exact de batches van v2's float32-stuk; op 270k–320k exact die van
     v2's bf16-stuk vanaf 270.000 (burst 2 bij 281.530); op 320k–350k exact die van v2's bf16-stuk vanaf 320.000
     (cluster 340.220, burst 3 bij 344.170). Burst 1 (252.620) kwam in een doorlopende run met andere maskers; die
     batches zijn niet terug te halen. De koppeling bij v2 was w0 → 0, w1 → 3, w2 → 1, w3 → 2 (de VM is sinds
     2 nov niet herstart); de smoke-test controleert haar, een afwijking wordt gemeld en geldt als verstoring.
3. **bf16** (controle, `v3-replay-bf16`): zonder fix (wel `--nonfinite-diagnostics`), 250.000 → 256.000, na zf. Laat
   zien of gewoon bf16 op deze batches (die van v2's float32-stuk) bij ~252.600 ontspoort, en in welke laagtypen de
   eerste niet-eindige gradiënten zitten. Beslist niet mee.

### Criteria (vooraf vastgelegd; `experiments/v3/replay_verdict.py`)

Geslaagd = S1 én S2 én S3.

- **S1 stabiliteit**, over de optimizer-stappen 250.001–350.000 van zf: geen zelfstop (50 op rij), hoogstens 2
  overgeslagen stappen op rij, minder dan 10 in elk venster van 1.000 stappen, hoogstens 10 in totaal, en stap
  350.000 bereikt. (Ter vergelijking: v2's clusters vóór de bursts hadden 10 in 280 stappen en 12 in 70; de bursts
  567–780 per 1.000.)
- **S2 verlies**, per blok van 10.000 stappen dat v2 in float32 draaide (250–260k, 260–270k, 280–290k, 290–300k,
  300–310k, 310–320k, 340–350k): het gemiddelde gelogde MLM-verlies van zf (Step-regels, elk het gemiddelde van 10
  stappen, elke 50 stappen) hoogstens **0,010** boven dat van v2 op dezelfde stappen. Eenzijdig: lager is geen
  bezwaar. 0,010 is ~1% van het verlies; bf16 tegen float32 op dezelfde batches scheelde bij v2 0,0006.
- **S3 held-out**, EVAL-verlies op 275.000, 300.000, 325.000 en 350.000: zf hoogstens **0,010** boven v2 op dezelfde
  stap (v2 was daar in bf16, float32, bf16, float32).

v2's referentie **[gemeten]**: S2-blokken 0,9741 / 0,9585 / 0,9519 / 0,9468 / 0,9405 / 0,9534 / 0,9389; EVAL 0,94703 /
0,94305 / 0,93619 / 0,93193.

Gerapporteerd, beslist niet: zf tegen v2's bf16 op de exact gelijke batches (270.000–281.700 en 320.000–344.200),
de gemiddelde log²Z per 10.000 stappen, de staptijd tegen v2's bf16 (0,220 s) en float32 (~0,30 s), de
`NONFINITE_GROUPS`-regels, en de controle.

### Uitvoering

Keten `mbert-ops/scripts/v3_replay_run.sh` op co1 (lock `final_eval_mix.lock` tot het eind; log
`mbert-ops/logs/v3_replay.log`, laatste regels `DONE …` of `FAILED: …` en `EXIT n`). Volgorde: setup van `~/mbert-v3`
op de gepinde commit, checkpoints kopiëren, smoke, zf (drie stukken), controle, uitslag. De bewaking stopt zf zodra S1
geschonden is (de trainers met `kill -9` van hun eigen pids), draait dan nog de controle en de uitslag, en eindigt op
`DONE … niet geslaagd`. Een reboot of preemptie: hervatten van hetzelfde stuk, hoogstens 3 keer. Een `FAILED` is een
fout van de keten of een gezakte smoke-test, geen uitslag. De keten verwijdert niets.

### Als de toets niet slaagt

QK-norm valt af (het model moet een gewone `ModernBertForMaskedLM` blijven). Dan beslist Yeb; de uitslag en de
`NONFINITE_GROUPS` van zf en van de controle zeggen in welk laagtype de niet-eindige gradiënten ontstaan. Opties die
binnen een gewone ModernBERT blijven (genoteerd, niet gekozen): het hele model in float32 (v2's stukken: stabiel,
+38% staptijd), alleen de MLP of de head in float32, een lagere piek-LR, of een grotere z-loss.

### Wat deze toets niet zegt

Eén run per arm, vanaf één checkpoint van v2: geen variantie over seeds. Of de fix ook vanaf stap 0 en op CulturaX
stabiel is, blijkt pas in de ablatie (deel 2). Het model van zf is na 250.000 een ander model dan v2; "zelfde
batches" betekent zelfde invoer, niet zelfde toestand.

### Aantekening na de start (20 nov 11:23; verandert geen criterium)

**[gemeten]** In de smoke-test (250.000 → 250.300, zelfde batches als v2's float32-stuk) was de gemiddelde log²Z
317 → 290 (|log Z| ≈ 17), niet de ~100 van het rekenvoorbeeld hierboven: de z-loss-term is bij de start dus ~0,03
(~3% van het verlies). Verlies per Step-regel 0,0014–0,0034 boven v2's float32, staptijd 0,260–0,273 s (v2 bf16
0,220 s, float32 ~0,30 s). Koppeling host → procesindex gelijk aan v2's.

## Deel 2: vormablatie (vastgelegd vóór de start; niet starten zonder Yebs akkoord)

### Vraag

Welke vorm van ~230M parameters geeft, met het recept van T2 en v3's stabiliteitsfix, na 100.000 stappen op
CulturaX de beste fine-tune-uitslagen? De uitkomst kiest de vorm van dmbert v3; ze zegt niets over v3 tegen T2.

### Vormen [gemeten: parametertelling met de modelcode, `jax.eval_shape` van `ModernBertForMaskedLM`]

Inclusief de embedding (32.128 × hidden, gekoppeld aan de decoder), de norms, de MLM-head (dense + norm) en de
decoder-bias. Head-dimensie overal 64.

| arm | lagen × hidden | heads | GeGLU (intermediate) | globaal | globale lagen | parameters | t.o.v. A | waarvan embedding |
|---|---|---:|---:|---|---:|---:|---:|---:|
| A | 22 × 1024 | 16 | 1536 | elke 3e | 8 | 230.110.592 | 0 | 32.899.072 |
| B | 29 × 768 | 12 | 2048 | elke 3e | 10 | 230.601.088 | +0,21% | 24.674.304 |
| C | 16 × 1024 | 16 | 2560 | elke 3e | 6 | 226.952.576 | −1,37% | 32.899.072 |
| D | 22 × 1024 | 16 | 1536 | elke 2e | 11 | 230.110.592 | 0 | 32.899.072 |
| A2 | als A, `--seed 43` | | | | | 230.110.592 | 0 | |

**Afstemming.** Yebs schets: B ~30 × 768 met GeGLU ~2048, C ~18 × 1024 met GeGLU ~2560. Regel: de MLP-breedte
van de schets houden (die maakt het verschil tussen de vormen) en het aantal lagen kiezen dat het dichtst bij A's
parameters komt. Bekeken en niet gekozen: B 30 × 768 × 2048 (+3,29%), B 30 × 768 × 1920 (−0,56%), C 18 × 1024 × 2560
(+9,11%, buiten ±5%), C 18 × 1024 × 2176 (−0,12%, maar MLP-verhouding 2,1 in plaats van 2,5), C 17 × 1024 × 2560
(+3,87%). Alle gekozen vormen liggen binnen ±1,5% van A.

**A2** (voorstel van mij, niet van Yeb): A met een andere seed voor initialisatie, datavolgorde en maskering
(`--seed`, nieuw in deze branch; standaard 42 zoals elke eerdere run). Eén pretraining per vorm laat niet zien
hoeveel van een verschil ruis van de pretraining is; |A − A2| maakt die ruis zichtbaar en gaat in de drempel. Zonder
A2 (op Yebs woord) is de drempel alleen de seed-range van de fine-tunes (`decide_v3.py --without-a2`).

**[afgeleid uit de code]** De lokale attention rekent in deze code de volle 1.024 × 1.024 logits en maskeert daarna;
D kost dus evenveel rekenwerk per stap als A (in een implementatie met een echte sliding-window-kernel is D duurder).

### Data, tokens, stappen, schema

- **Data:** CulturaX-nl op T2's revisie `6a8734bc`, `--url-blocklist hebban.nl`. Per worker de eerste 8 van zijn 32
  bestanden van de eindrun (`gs://…/mbert/final/staging/culturax_nl_files-w<N>.txt`, sha256 tegen
  `culturax-w<N>.tsv`): 32 van de 128 bestanden, 12,2 GiB per worker, ~20 mld tokens **[afgeleid: 25% van 80,7
  mld]**. Zelfde bestanden op dezelfde workers en seed 42 voor de data: A, B, C en D zien exact dezelfde batches; A2
  een andere volgorde van dezelfde bestanden. `nl_part_00097` (T2's held-out-slice voor de EVAL in de trainer) zit niet
  in deze 32: hier is die EVAL echt held-out (bij T2 niet).
- **Tokens:** 100.000 stappen × 128 × 1.024 = 13,1 mld posities, ~12,4 mld echte tokens (94,6%); ~0,6 doorgang over
  de 32 bestanden, geen document twee keer.
- **Schema (WSD):** warmup 1.250 stappen naar 4,8e-4, constant tot 80.000, 1-sqrt-cooldown over 20.000 stappen (20%,
  T2: 20,37%) naar 1e-6.
- **Verder als T2:** 15% losse tokens, packing, 1.024 tokens, batch 128 (8 per chip × 16), AdamW, weight decay 0,01,
  layer-norm-LR-factor 1,0, `--skip-nonfinite-updates`; bf16 met `--attention-float32 --z-loss 1e-4`
  (`--nonfinite-diagnostics` alleen diagnose). Dezelfde LR voor elke vorm (niet per vorm afgesteld).
- **Tokenizer (a)** in de pretraining (`scripts/build_tokenizer_v3.py`, `--tokenizer ~/mbert-v3-tokenizer`). Voor
  tekst geeft hij in de modus van de trainer dezelfde ids als v2's tokenizer (getest, zie hieronder); alleen het
  padding-id verschilt (32.001 in plaats van 20.277), en padding is gemaskeerd. De pretraining is daarmee die van v2's
  tokenizer.
- **Smoke-test per vorm** (niet voor A2): 200 stappen onder een eigen naam: eindig verlies op 200, geen `NONFINITE`,
  EVAL op 200, staptijd gelogd.

### Evaluaties

Per arm op checkpoint 100.000, `experiments/v3/eval_v3.sh` (= v2's `eval_v2.sh` met de vorm als parameter), seeds
42, 43 en 44, de instellingen van v2's evaluatie:

| taak | script | instellingen |
|---|---|---|
| SICK-NL | finetune.py | max-length 512, update 256, LR 5e-5 constant, 3 epochs |
| DBRD@512 | finetune.py | max-length 512, update 192, padding tot 128, LR 5e-5, 3 epochs |
| SQuAD-NL v1.1 F1 | finetune_extra.py | 384, doc-stride 128, update 32, LR 5e-5 lineair naar 0, 3 epochs |
| CoNLL-2002 entity-F1, POS LassySmall, POS Alpino | finetune_extra.py | 512, update 32, LR 5e-5 lineair, 3 epochs |
| BLiMP-NL | eval_blimp_nl.py | PLL-word-l2r over 8.400 paren, float32, één meting |

18 fine-tunes en BLiMP-NL per arm. Code mbert-jax `dumb-extra` @ `ae11478`, de venv's van v2's evaluatie
(vingerafdruk `91a3c3a5f544`), alleen op mbert-mix. De fine-tunes gebruiken v2's tokenizer
(`yhavinga/dutch-llama-tokenizer`; voor tekst dezelfde ids als (a)): de evaluatie meet de vorm, voor elke arm gelijk.
Wat tokenizer (a)'s sjablonen (`<s> A </s> B </s>`) bij fine-tunen doen, meet deze ablatie niet.

### Beslisregel (`experiments/v3/decide_v3.py`, exact op tellingen)

Elke kandidaat X ∈ {B, C, D} tegen A, per taak:

1. **Drempel:** de grootste seed-range van X en A (zoals `decide_v2`/dumb-extra); met A2 ook |gemiddelde A −
   gemiddelde A2|; BLiMP-NL 0,5 punt (of |A − A2| als dat groter is).
2. **Uitslag per taak:** "beter" of "slechter" als het verschil van de gemiddelden groter is dan de drempel, anders
   "gelijk".
3. **Oordeel** (T2's `primary_rule`) op SICK-NL en DBRD@512: "schaadt" als X op een van beide slechter is, anders
   "helpt" als X op SICK-NL beter is, anders "geen verschil".
4. **X voldoet** als X "helpt" en op geen extra taak (SQuAD-NL F1, CoNLL-2002, POS LassySmall, POS Alpino, BLiMP-NL)
   slechter is, of als X "geen verschil" geeft, op minstens één extra taak beter is en op geen enkele slechter.
5. **Uitkomst:** voldoet geen kandidaat, dan blijft A (de vorm van T2 en v2). Voldoet er één, dan die. Voldoen er
   meer, dan die met de meeste "beter" over alle zeven taken; staan die gelijk, dan beslist Yeb.

Gerapporteerd, beslist niet: staptijd per vorm, EVAL-curve (held-out CulturaX), niet-eindige stappen per arm.

### Rekentijd

- **Per run [gemeten voor A, geschat voor B en C]:** A met de fix 0,260–0,273 s per stap (smoke-test van deel 1):
  100.000 stappen ≈ 7,3 u, met checkpoints en EVAL ~7,5 u, plus ~10 min smoke-test. D als A (zie boven). B en C
  hebben evenveel parameters en dus ongeveer evenveel FLOPs per token **[afgeleid]**; B's smallere matmuls (768)
  benutten de MXU slechter, C heeft minder lagen: geschat binnen ±20% van A, gemeten in hun smoke-test.
- **Evaluatie per arm:** ~45–90 min (v2: 22 taken in ~50 min; een nieuwe vorm compileert opnieuw).
- **Staging:** ~15–30 min per VM (8 × 1,6 GB per worker van Hugging Face).
- **Eén VM (mbert-mix), vijf armen na elkaar:** 5 × ~7,7 u + 5 × ~1 u + 0,5 u ≈ **44 u** (zonder A2 ~35 u).
- **Twee VM's:** mbert-mix A, B, A2 en de evaluatie van alle vijf (3 × 7,7 + 5 × 1 ≈ 28 u), mbert-corpus C en D
  (2 × 7,7 + staging ≈ 16 u): **~28 u** doorlooptijd (zonder A2 ~20 u). mbert-corpus pas na G4-6b; of zijn venv de
  vingerafdruk van de trainer heeft en of daar nog CulturaX-bestanden staan, is niet bekeken (die VM is niet
  aangeraakt); de keten controleert het en stopt anders.

### Voorwaarden voor de start

1. De replay-toets van deel 1 is geslaagd; de keten leest het oordeel in `gs://…/mbert/v3-replay/verdict/`. Is hij niet
   geslaagd, dan valt QK-norm af en beslist Yeb: bijvoorbeeld `PRECISION=float32` (de armen helemaal in float32 zonder
   de fix, ~0,30 s per stap, +15% tijd) of een andere fix (dan eerst een nieuw protocol).
2. Yebs akkoord op dit deel: vormen, A2, de 32 bestanden, de regel.
3. Schijf: per worker 12,2 GiB + 1 GiB. Op mbert-mix had w2 op 20 nov 17 GB vrij naast v2's `~/fw2optout` (~25 GB):
   dat past. De staging stopt zelf bij te weinig ruimte en verwijdert niets.

### Ketenscript

`mbert-ops/scripts/v3_ablation_run.sh` (co1): lock van de VM, code-controle, oordeel van de replay, setup (code,
venv, tokenizer (a) met de sha256 van co1), staging, per arm smoke-test en 100.000 stappen (een zelfstop of fout is
`FAILED`: dan beslist Yeb), evaluatie, `decide_v3.py`. Startcommando's in de kop van het script en in
`mbert-ops/HANDOVER-dmbert-v3.md`.

### Wat de ablatie niet zegt

- 100.000 stappen (~12 mld tokens) is ~8% van een volledige run van 1.306.000 stappen: de volgorde van de vormen kan
  bij langer trainen veranderen.
- Eén pretraining per vorm; A2 meet de ruis alleen voor A.
- Eén LR voor alle vormen; een diepe, smalle vorm kan een andere LR willen.
