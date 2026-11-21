# T3: de eindrun van dmbert v3, protocol

Opgesteld 21 en 22 november 2026 door een subagent in opdracht van de hoofdsessie, na Yebs besluit van 21 nov ~23:10.
Gecommit en gepusht vóór elke stap op een worker; de keten `mbert-ops/scripts/t3_run.sh` pint deze commit
(`PROTOCOL_COMMIT`) en start niets op een worker voordat hij mbert-corpus heeft. Wijzigingen na de start komen onder
"Amendementen", gedateerd, en vervangen de tekst erboven niet. De keten verwijdert zelf niets en publiceert niets.

Markering:
- **[Yeb]** een besluit van Yeb (opdracht van de hoofdsessie, 21 nov ~23:10, en de besluitentabel in
  `hammond/mbert/LEARNINGS.md`);
- **[eigen keuze]** een keuze van de opsteller van dit protocol, binnen Yebs besluit;
- **[citaat]** letterlijk uit de bron, **[parafrase]** samengevat uit de bron, **[gemeten]** door mij gemeten of geteld,
  met bron, **[afgeleid]** eigen gevolgtrekking of schatting.

## 1. Besluit van Yeb (21 nov ~23:10) [Yeb]

- De eindrun van dmbert v3 heet **T3**.
- **Vorm:** 18 lagen × hidden 1.152, 18 heads (head-dimensie 64), GeGLU intermediate 2.944, globale attentie elke 3e
  laag; lokaal venster en RoPE zoals T2. Een gewoon `ModernBertForMaskedLM`, aangetoond met een exporttest.
- **Data:** gelijk aan T2 ("de CulturaX-mix"), 1,5× zoveel: 1.959.000 stappen bij dezelfde batch, ~3 epochs
  CulturaX-nl, ~243 mld tokens.
- **Stabiliteit en tokenizer:** z-loss 1e-4 en `--attention-float32`, in bf16 (de fix van de replay); tokenizer (a).
- **Schema WSD:** warmup 1.250, daarna constant; een cooldown-tak na 2 epochs op dezelfde plek als bij T2, met dezelfde
  vorm als T2's tak; de hoofdtak loopt constant door en koelt in de laatste ~20% af naar 1.959.000.
- **LR:** 4,8e-4 zoals T2, tenzij er een concrete reden is voor een andere (dan als open vraag; de standaard blijft
  4,8e-4).
- **Held-out:** vooraf uit de trainingsdata gehaald, met hashes bewezen afwezig (bij T2 zat de held-out-slice per
  ongeluk in de training: LEARNINGS, correctie 20 nov).
- **Evaluatie:** zoals v2 en de ablatie, met de eval-venvs (vingerafdruk `91a3c3a5f544`, `mbert-dumbx` @ `ae11478`),
  plus de memorisatietest en de held-out-evals zoals bij v2.
- **Beslisregel** vooraf, naar het model van v2's regel: T3 (hoofdtak) tegen T2, met de drempelvorm van de ablatie;
  ook wat de tak tegen T2 en wat de hoofdtak tegen de tak betekent. Uitkomsten: vervangt T2 / naast T2 / niet
  publiceren. Publiceren alleen op Yebs woord.
- **Planning:** T3 draait op mbert-corpus (`final_run.lock`) zodra de keten van G4-7 klaar is; de evaluatie op mbert-mix
  (`final_eval_mix.lock`) of op mbert-corpus. Geen TPU vóór dat moment.

## 2. Code

mbert-jax, branch `t3`, vanaf `v3` @ `356f8f2` (de trainer met `--attention-float32`, `--z-loss`, `--seed`,
`--nonfinite-diagnostics` en tokenizer (a); de trainercode zelf is in deze branch niet veranderd). Nieuw, alleen in
`experiments/t3/`:

| bestand | wat |
|---|---|
| `protocol.md` | dit protocol |
| `start_t3.sh` | het trainercommando van T3 per worker: `main`, `branch` (de cooldown-tak), `smoke` |
| `stage_t3.py` | de data: bestanden tegen T2's referentie (`files`), de sleutels van de held-out (`keys`), de scan van de trainingsbestanden (`scan`), het weglaten van de held-out (`write`), het bewijs (`scan --require-none`), de steekproef van de memorisatietest (`memsample`) |
| `prepare_t3.sh` | de stappen op een worker: `setup` (code, venv, tokenizer (a)) en `data` (alle datastappen, losgekoppeld, met log en `EXIT`) |
| `eval_t3.sh` | het deel van de evaluatie op één worker: `eval_v2.sh` met de vorm als parameter (zoals `eval_v3.sh`), plus memorisatietest en held-out-evals met die vorm |
| `decide_t3.py` | de beslisregel van §11, exact op tellingen (rekent met `experiments/final/decide.py`, `final-v2/decide_v2.py`, `v3/decide_v3.py`) |
| `smoke_reference.json` | de referentie en grenzen van de smoke-test (§8) |

Tests: `tests/test_t3_stage.py`, `tests/test_t3_decide.py`, `tests/test_t3_start.py` (commando's, schema, scripts),
`tests/test_t3_export.py` (export naar HF). Keten op co1: `mbert-ops/scripts/t3_run.sh` (bron van de gedeelde functies:
`final_v2_lib.sh`), log `mbert-ops/logs/t3_run.log`, overdracht `mbert-ops/HANDOVER-t3.md`.

## 3. Het model

| | T2 | T3 |
|---|---|---|
| lagen × hidden | 22 × 1.024 | **18 × 1.152** [Yeb] |
| heads (head-dimensie) | 16 (64) | **18 (64)** [Yeb] |
| GeGLU intermediate | 1.536 | **2.944** [Yeb] |
| globale attentie | elke 3e laag (8 van 22) | elke 3e laag (**6 van 18**: laag 0, 3, …, 15) |
| lokaal venster, RoPE | 128; 160k globaal / 10k lokaal | gelijk (standaard van `MMBertConfig`, zoals T2) |
| parameters | 230.110.592 | **317.106.304** [gemeten] |
| waarvan embedding (gekoppeld aan de decoder) | 32.899.072 | 37.011.456 |
| zonder embedding | 197.211.520 | 280.094.848 (1,42×) |

**[gemeten]** De telling is die van de modelcode: `jax.eval_shape` van `ModernBertForMaskedLM` (`mmbjax.heads.masked_lm`)
met vocab 32.128, inclusief de norms, de MLM-head (dense + norm) en de decoder-bias; dezelfde code geeft voor A
(T2's vorm) 230.110.592 en voor C 226.952.576, de getallen van de ablatie. Yebs "~316M" is dus 317,1M. Per laag
15.482.880 gewichten in de matrices (qkv 3.981.312, out 1.327.104, Wi 6.782.976, Wo 3.391.488) plus twee norms.

**HF-export [gemeten, `tests/test_t3_export.py`]:** een model met T3's breedte (hidden 1.152, 18 heads, intermediate
2.944, globaal elke 3e laag; 4 lagen om het op de CPU van co1 te houden) en tokenizer (a) exporteert met
`scripts/convert_flax_to_hf.py` naar een gewoon `ModernBertForMaskedLM` met de logits van het JAX-model (≤ 1e-3
relatief, top-1 overal gelijk, voorbij het lokale venster); en `transformers.ModernBertConfig` met T3's volle vorm geeft
op het meta-device precies 317.106.304 parameters. De conversie voor de fine-tunes (`convert_gcs_checkpoint.py` @
`ae11478`) leidt het aantal heads af als hidden/64 = 18; dat klopt alleen omdat de head-dimensie 64 is.

## 4. Data: die van T2 [Yeb]

**Wat T2 trainde [parafrase, `experiments/final/protocol.md` en `start_final.sh` @ `1c8f2c7`; gemeten, bucket]:**
`uonlp/CulturaX`, config nl, revisie `6a8734bc69fefcbb7735f4f9250f43e4cd7a442e`, alle 128 bestanden (117.392.666
documenten), 32 per worker in de lijst `~/culturax_nl_files.txt` van die worker
(`gs://t5-dutch-english-usc2/mbert/final/staging/culturax_nl_files-w<N>.txt`, met grootte en sha256 per bestand in
`culturax-w<N>.tsv`); `--url-blocklist hebban.nl` (45.781 documenten, alle `www.hebban.nl`, weggelaten vóór de shuffle);
`--dataset-num-rows 117346885`; packing met lange documenten opgeknipt in stukken van 1.024; 15% losse tokens
(80/10/10); 4 hosts × 32 rijen × 1.024 = 131.072 plekken per stap; seeds: init 42, maskering 42 + procesindex, data
42 + pass (per pass een andere volgorde); shuffle-buffer 10.000.

**T3 neemt dat over**, met drie verschillen:
1. de held-out (§5) is uit de trainingsdata gehaald: in de lijst van worker 0 vervangt een kopie van `nl_part_00097`
   zonder de held-out-rijen het origineel (en zo ook elk ander bestand waarin een held-out-tekst staat);
2. `--dataset-num-rows` is het aantal documenten van T3's lijsten min dat van `hebban.nl` (alleen voor de metriek
   `data/passes`);
3. tokenizer (a) (§1): voor tekst geeft hij in de modus van de trainer dezelfde ids als T2's tokenizer (getest op
   120.000 CulturaX-documenten, `experiments/v3/protocol.md`, bijlage); alleen het padding-id verschilt (32.001 in plaats
   van 20.277), en padding is gemaskeerd.

Dezelfde lijsten op dezelfde workers met dezelfde seeds: per host leest T3 dezelfde documentstroom als T2, op de
weggelaten held-out-documenten na (worker 0 vanaf het punt waar `nl_part_00097` in zijn volgorde komt). De koppeling
host → procesindex (en dus de maskerseed per host) legt de runtime vast; die wordt gelogd, niet afgedwongen.

**[gemeten, 21 nov ~23:20, alleen lezen]** Op elke worker van mbert-corpus staan de 32 bestanden van zijn lijst,
`~/culturax_nl_files.txt` (32 regels) en `~/heldout.parquet` (md5 `b2f579a8…`); 17–18 GB vrij op de schijf. De staging
controleert grootte en sha256 van elk bestand tegen T2's referentie en haalt een ontbrekend of afwijkend bestand
opnieuw van Hugging Face op T2's revisie (een afwijkend bestand blijft staan en is dan `FAILED`).

**Tokens en passes [afgeleid].** Eén pass van T2 na de blocklist ≈ 652.620 stappen (T2's protocol). De held-out is
~40.000 documenten van gemiddeld ~690 tokens (81 mld tokens / 117,4 mln documenten), ~28 mln tokens (0,034%): één pass
van T3 ≈ 652.400 stappen. De hoofdtak (1.959.000 stappen) eindigt dus ~0,3% in pass 4, de tak (1.306.000) ~0,2% in pass
3, zoals T2 ~0,06% in pass 3. Posities: 1.959.000 × 131.072 = 256,8 mld, waarvan 94,6% echte tokens (de vulling van
CulturaX): **~243 mld tokens**; de tak 1.306.000 stappen, ~162 mld tokens, gelijk aan T2.

## 5. Held-out: vooraf uit de training, bewezen met hashes [Yeb; uitwerking eigen keuze]

**Welke documenten [eigen keuze].** De held-out van T3 is T2's held-out-bestand
`gs://t5-dutch-english-usc2/mbert/heldout/nl_part_00097_first40000.parquet` (`~/heldout.parquet`, md5
`b2f579a876d06eef40c94270537bb7af`, 40.000 documenten, volgens de naam de eerste 40.000 rijen van `nl_part_00097`).
Redenen: de EVAL-regels van de trainer (elke 25.000 stappen, 50 batches, vaste maskers) zijn dan voor T3 een echte
held-out op dezelfde documenten als T2's EVAL-curve, die geziene tekst meet; het bestand staat al op beide VM's en de
startscripts controleren zijn md5; en voor de memorisatietest is er een zuivere vergelijking: T3 zag deze documenten
nooit, T2 zag ze twee keer.

**Sleutels.** Per held-out-document twee sha256-sleutels: `exact` = sha256 van de tekst in UTF-8, zoals de trainer hem
leest; `normalized` = sha256 van de tekst na Unicode-NFC en het samenvoegen van alle witruimte tot één spatie (zonder
witruimte aan begin en eind). Een trainingsdocument "is" een held-out-document als zijn `exact`- of zijn
`normalized`-sleutel in de held-out voorkomt. Bijna-duplicaten met andere woorden vangt dit niet (zie §13).

**Weglaten (`stage_t3.py`, op elke worker, CPU).**
1. `files`: de 32 bestanden van de worker tegen grootte en sha256 van `culturax-w<N>.tsv` (ontbrekend: ophalen van HF
   op T2's revisie; afwijkend: `FAILED`, niets verwijderd).
2. `keys`: de sleutels van `~/heldout.parquet` (md5 gecontroleerd); elke worker rekent ze zelf uit, de keten eist dat
   de sha256 van de sleutellijst op de 4 workers gelijk is.
3. `scan`: elke rij van elk van de 32 bestanden (kolommen `text` en `url`): beide sleutels, de documenten van
   `hebban.nl` (dezelfde functie als de trainer, `url_blocklist.py`) en elke rij met een held-out-sleutel (rij, welke
   held-out-rij, `exact` of `normalized`), plus de sha256 van het bestand.
4. `write`: van elk bestand met zulke rijen een kopie zonder die rijen naar `~/culturax_nl/t3/<naam>` (alle kolommen,
   dezelfde compressie en rijgroepgrootte), en de lijst `~/t3_files.txt` = T2's lijst met die paden vervangen. De
   originelen blijven ongewijzigd.
5. **Bewijs** (`scan --require-none` op `~/t3_files.txt`): opnieuw elke rij van elk bestand van de definitieve lijst
   gehasht; het aantal rijen met een held-out-sleutel moet **0** zijn, anders `FAILED`. Per worker gaat het bewijs
   (`proof-w<N>.json`: per bestand pad, sha256, rijen, `hebban.nl`-rijen, overeenkomsten; de sha256 van de sleutellijst;
   de md5 van de held-out) naar `gs://…/mbert/t3/data/`, en in het rapport. Iedereen kan het met dezelfde bestanden
   (sha256) en dezelfde sleutels herhalen.

Verwachting [afgeleid]: op worker 0 de rijen 0–39.999 van `nl_part_00097` (`exact`); CulturaX is per taal ontdubbeld,
dus elders weinig of geen. De scan stelt het vast en het rapport noemt het aantal.

**Wat de trainer leest.** Alleen `~/t3_files.txt` (32 paden per worker, `--data-files-list`), met
`--url-blocklist hebban.nl`; de held-out zit er volgens het bewijs niet in. De trainer is niet veranderd. De
`--dataset-num-rows` is Σ rijen van de 4 lijsten − Σ `hebban.nl`-rijen van de bewijzen.

## 6. Schema en LR

**[Yeb]** WSD: warmup 1.250 stappen lineair naar de piek, daarna constant, cooldown in de vorm van T2's tak.
**[gemeten, T2's protocol en `start_final.sh`]** T2's tak 2 liep van **1.040.000** (niet 1.045.000) tot 1.306.000: een
**1-sqrt-cooldown** over 266.000 stappen (20,37% van 1.306.000) naar 1e-6 (`--lr-schedule wsd --decay-steps 266000
--decay-shape 1-sqrt --min-lr 1e-6 --max-steps 1306000`), vanaf een kopie van checkpoint 1.040.000 van de hoofdrun.
T3 doet precies hetzelfde voor de tak. Voor de hoofdtak dezelfde verhouding [eigen keuze]: 399.000 stappen = 20,37%
van 1.959.000, zodat de cooldown op 1.560.000 begint, een veelvoud van 10.000 (een gewoon checkpoint).

| deel | run | stappen | LR | tokens [afgeleid] |
|---|---|---|---|---|
| warmup | `t3-culturax-317m` | 0 – 1.250 | 0 → 4,8e-4 | |
| constant | `t3-culturax-317m` | 1.250 – 1.040.000 | 4,8e-4 | ~129 mld |
| **tak** (T3-tak) | `t3-culturax-317m-epoch2`, vanaf een kopie van 1.040.000 | 1.040.000 – 1.306.000 (266.000) | 4,8e-4 → 1e-6, 1-sqrt | samen ~162 mld |
| constant | `t3-culturax-317m` (hervat op 1.040.000) | 1.040.000 – 1.560.000 | 4,8e-4 | |
| **cooldown hoofdtak** (T3-hoofd) | `t3-culturax-317m` | 1.560.000 – 1.959.000 (399.000) | 4,8e-4 → 1e-6, 1-sqrt | samen ~243 mld |

- De hoofdrun traint op het schema `--max-steps 1959000 --decay-steps 399000` en stopt eerst op 1.040.000
  (`--stop-at-step`); de tak begint met een kopie van checkpoint én datastand 1.040.000 (Adams momenten en de stap van
  de hoofdrun) en leest daarna dezelfde batches die de hoofdrun daarna leest. Tot 1.040.000 is de LR van hoofdrun en
  tak bit voor bit gelijk (en gelijk aan T2's tak 2), en daarna is de hoofdrun constant tot 1.560.000 (test
  `test_t3_schedules_share_every_step_before_the_branch_and_the_main_leg_holds_the_peak_to_1560000` in
  `tests/test_t3_start.py`, met `wsd_schedule` van de trainer).
- Tijdens de cooldown is de LR 4,8e-4 − (4,8e-4 − 1e-6) · √f, met f het deel van de cooldown dat voorbij is
  ([parafrase] Hägele et al. 2024, *Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations*,
  arXiv:2405.18392, §3.2, zoals in T2's protocol uitgewerkt).
- Volgorde [eigen keuze]: hoofdrun tot 1.040.000 → tak → evaluatie van de tak → hoofdrun verder tot 1.959.000 →
  evaluatie. Zo is na ~5 dagen bekend wat vorm en grootte bij T2's data doen (§11, vraag 2), terwijl de hoofdrun nog
  ~3,5 dag loopt; Yeb kan dan beslissen of hij doorloopt. De keten loopt zonder ingreep door.
- Na een hervatting seedt de trainer de maskering opnieuw (42 + procesindex): dezelfde documenten, andere maskers dan
  een doorlopende run (zoals bij T2 op 520.000 en 1.040.000).

**LR [eigen keuze, Yebs standaard]: 4,8e-4, gelijk aan T2.** Geen concrete reden voor een andere gevonden. Bekeken:
ModernBERT (Warner et al. 2024, *Smarter, Better, Faster, Longer*, arXiv:2412.13663, §2.2.2 "Training Settings")
**[citaat]** "We trained ModernBERT-base at a constant LR of 8e-4 for 1.7 trillion tokens following a 3 billion token
warmup" en "After a 2 billion token warmup, we trained ModernBERT-large at a LR of 5e-4 for 900 billion tokens"
(149M resp. 395M parameters). **[afgeleid]** T3 (317M) zit tussen die twee in; 4,8e-4 ligt al onder de LR van
ModernBERT-large, en T2 (230M) trainde ermee. Een LR die met de breedte meeschaalt (×1.024/1.152 = 4,27e-4) zou een
afwijking zijn zonder meting bij ons; de stabiliteit bij 4,8e-4 is met de fix getoetst (replay en ablatie: 0
overgeslagen stappen) en blijft bewaakt (§7). Geen open vraag.

**Overige instellingen, gelijk aan T2:** AdamW, weight decay 0,01, `--layer-norm-lr-factor 1.0`, clipping 1,0,
`--skip-nonfinite-updates` (`optax.apply_if_finite`, zelfstop na 50 op rij), bf16 met parameters en optimizerstand in
float32, checkpoint elke 10.000 stappen (de trainer houdt de laatste 3), EVAL elke 25.000 stappen op `~/heldout.parquet`
(50 batches, vaste maskers), `--log-every 10`, systeemmetrieken elke 1.000 stappen. Plus de fix **[Yeb]**:
`--attention-float32 --z-loss 1e-4`, en `--nonfinite-diagnostics` (alleen diagnose; verandert geen update).

**ckpt_keep** (de keten kopieert): hoofdrun elke 100.000e stap, 1.040.000 (aftakpunt), 1.560.000 (begin cooldown) en
1.959.000; tak 1.100.000, 1.200.000, 1.300.000 en 1.306.000. Datastanden (`ckpt_data_state/<stap>/`) worden niet
weggegooid.

## 7. Precisie, bursts en de andere stopregels (vooraf vastgelegd) [eigen keuze, naar Yebs regels van 10 en 16 nov]

Uitgangspunt: bf16 met de fix. De replay (v2 250k–350k, waar v2 drie bursts had) en de ablatie (vijf armen, 100.000
stappen vanaf 0) hadden 0 overgeslagen stappen [gemeten, LEARNINGS 20 en 21 nov]. T3 is groter en loopt ~20× langer;
de bewaking gaat ervan uit dat een burst kan komen. Een overgeslagen stap verandert parameters noch optimizerstand.

1. **Losse overgeslagen stappen en kleine clusters:** geen ingreep (gelogd).
2. **Burst** = een van: (a) een `NONFINITE`-regel met `in_a_row` ≥ 10 (Yeb, 10 nov: float32 "pas bij ≥10 op rij");
   (b) ≥ 500 overgeslagen stappen binnen 1.000 opeenvolgende stappen, gemeten aan `Skipped` op de Step-regels (v2's
   les: een burst van > 50% met hoogstens 5 op rij gaf geen zelfstop); (c) de zelfstop van de trainer (50 op rij).
   Gevolg: de keten stopt de trainers (alleen hun eigen pids, `kill -9`) en hervat vanaf het laatste gecommitte
   checkpoint in **float32** (`COMPUTE_DTYPE=float32`; de fix en de z-loss blijven, dus hetzelfde doel) tot
   **P = (⌊b / 10.000⌋ + 3) × 10.000**, met b de stap waarop de burst gezien is (20.000–30.000 stappen erna; v2's
   bursts kwamen 12–24k stappen na de terugkeer naar bf16), daarna weer bf16. Komt P voorbij het einde van een deel
   (1.040.000, 1.306.000, 1.959.000), dan stopt het stuk daar en gaat het volgende deel in float32 verder tot P. P
   geldt voor de stappen, niet voor één run: hoofdrun en tak lezen na 1.040.000 dezelfde batches, dus een burst in de
   tak bij b maakt ook de stappen 1.040.000–P van de hoofdrun float32 (en omgekeerd). Een burst wordt gezien bij de
   volgende poll (elke 180 s, ~560 stappen); is er sinds de burst al een nieuw checkpoint, dan hervat het stuk daar
   (de overgeslagen stappen ervoor blijven overgeslagen; de parameters zijn gaaf).
3. **Derde burst** (geteld over de hele run, hoofdrun en tak samen): de rest van alle training in float32, zoals Yeb
   op 16 nov voor v2 besloot. Kost +38% staptijd over de rest (v2: 0,30 tegen 0,22 s per stap).
4. **Een burst of zelfstop in float32**: `FAILED`, Yeb beslist (v2 had in ~1 mln stappen float32 geen enkele).
5. **Echte NaN**: de laatste twee Step-regels (50 stappen uit elkaar) hebben `Loss: nan` terwijl `Skipped` tussen
   beide niet steeg (een uitgevoerde update met NaN): trainers stoppen, `FAILED`. Een NaN met stijgende `Skipped` is
   een overgeslagen stap (regel 1–2), niet dit.
6. **Trainers weg zonder exitbestand** (reboot of preemptie): hervatten met dezelfde opties vanaf het laatste
   checkpoint, hoogstens 3 keer per deel.
7. **Elke andere afsluiting met een foutcode** (ook een geheugenfout tijdens de training): `FAILED`, trainers
   gestopt.
8. **EVAL-verlies**: elke EVAL-regel komt in het ketenlog; ligt er een meer dan 0,02 boven de laagste tot dan, dan
   een `WARN`-regel. Geen automatische ingreep (Yeb, 10 nov: achteruitgang van loss of EVAL is een reden om te kijken).

Elke ingreep staat met tijd in `mbert-ops/dx/t3/events.txt` en in het rapport. Een float32-stuk verandert geen data,
geen schema en geen volgorde (een hervatting met de datastand van het checkpoint gaat exact verder).

## 8. Smoke-test en preflight

**Smoke-test [eigen keuze, naar v2's smoke-test]**, na de staging, op de definitieve data, onder een eigen naam
(`t3-culturax-317m-smoke-<TAG>`, eigen bucketmap; het commando van de hoofdrun met hetzelfde schema):
1. 300 stappen, checkpoint elke 100, EVAL op 200, systeemmetrieken elke 100; daarna een hervatting vanaf checkpoint
   300 met zijn datastand tot 400.
2. Criteria (`smoke_reference.json`):
   - **verlies** op 100, 150 en 200 binnen 10% van S's (7,4894 / 6,8017 / 6,2256; T2's smoke-referentie). De
     ablatie-armen lagen op 200 binnen 0,3% (A 6,2441, C 6,2535) en B op +4,3% [gemeten, `logs/v3_ablation_*.log`];
   - **staptijd**: de mediaan op 100/150/200 ten hoogste **0,42 s** (1,3 × de schatting van 0,32 s, §12); daarboven
     duurt de run > 10 dagen en beslist Yeb;
   - **geheugen**: de piek per chip (`system/host<h>_chip<c>_peak_bytes_in_use` uit de `SYSTEM`-regels van de trainer),
     ten hoogste **30,0 GiB** van de 32 GiB HBM van een v4-chip; een geheugenfout (`RESOURCE_EXHAUSTED`) of een piek erboven
     is `FAILED`, en er start niets;
   - geen overgeslagen stap, geen `NONFINITE`; de regel van de URL-blocklist; `EVAL step=200`; de hervatting leest
     checkpoint en datastand van 300 en logt stap 400.
3. Daarna draaien memorisatietest en held-out-eval één keer op checkpoint 400 (alleen of ze lopen; de getallen zeggen
   niets).

Bij een geheugenfout zijn de opties (genoteerd, niet gekozen; Yeb beslist): 4 rijen per chip met
`--gradient-accumulation-steps 2` (zelfde batch per update; eerst te toetsen, want de stapnummering van checkpoints en
schema loopt dan in micro-stappen), of rematerialisatie in de trainer (nieuwe code).

**Preflight vóór elke start of hervatting** (in `start_t3.sh` op elke worker, en in de keten): de TPU vrij (`lsof`
op `/dev/accel*`, geen trainer of eval van welke keten dan ook), `~/mbert-t3` op de gepinde commit, de venv, tokenizer
(a) met de sha256 van de bouw op co1, `~/t3_files.txt` met 32 bestaande bestanden en de md5 van de data-stap,
`~/t3_files.total` een getal, `~/heldout.parquet` met zijn md5; voor de tak de gekopieerde checkpoint en datastand
1.040.000; vrije schijf en RAM.

## 9. Uitvoering

Keten `mbert-ops/scripts/t3_run.sh` op co1 (TAG per start, log `logs/t3_run.log`, pid- en pgid-bestand
`t3_run.pid`/`t3_run.pgid`). Volgorde; elke gedane stap staat als marker in `gs://…/mbert/t3/chain/` en wordt bij een
herstart overgeslagen:

1. **Code** (alleen lezen): de worktree `mbert-jax-wt/t3` schoon, de commit gepusht en afstammend van dit protocol.
   De commit wordt bij de eerste start vastgelegd in `mbert-ops/dx/t3/commit.txt`; een herstart neemt die, zodat een
   latere commit (een amendement) de lopende code niet verandert.
2. **Wachten** [Yeb]: tot mbert-corpus vrij is. Voorwaarde: de keten van G4-7 (`logs/g4arm7_run.log`, pidbestand
   `g4arm7_run.pid`) is geëindigd (het proces leeft niet meer en de laatste regel is `EXIT <n>`), of het overridebestand
   `mbert-ops/t3.corpus-go` bestaat; daarna `final_run.lock` (zonder wachten op een andere houder; anders wacht de
   keten verder). Ten hoogste 72 u. Een `EXIT` anders dan 0 [eigen keuze]: de keten wacht nog 60 min en neemt
   mbert-corpus niet zolang `mbert-ops/t3.corpus-hold` bestaat (de hoofdsessie wil G4-7 dan eerst herstellen; daarna
   `t3.corpus-go`, of het holdbestand weg en weer 60 min). Tijdens het wachten doet de keten niets op een worker of in
   de bucket (in de droge run gemeten: 0 aanroepen van `gcloud` en `gsutil` tot het slot genomen was).
3. **Setup** (mbert-corpus): `~/mbert-t3` op de commit, de venv tegen `vm.lock` en vingerafdruk `91a3c3a5f544`,
   tokenizer (a) naar `~/mbert-t3-tokenizer` met de sha256 van co1's bouw.
4. **Data** (§5): `prepare_t3.sh data` op de 4 workers, het bewijs en de steekproef naar de bucket.
5. **Smoke-test** (§8).
6. **Hoofdrun tot 1.040.000**, met de bewaking van §7.
7. **Tak**: kopie van checkpoint en datastand 1.040.000, tak tot 1.306.000.
8. **Evaluatie van de tak** en `decide_t3.py --branch-only` (vraag 2, voorlopig rapport; beslist niets).
9. **Hoofdrun verder tot 1.959.000.**
10. **Evaluatie van de hoofdtak** (en T2's memorisatietest en held-out-evals op T3's steekproef), `decide_t3.py`,
    rapport `mbert-ops/dx/t3/report.md`. `DONE`.

**VM's [Yeb].** Training op mbert-corpus (`final_run.lock`, tot het eind vastgehouden). Evaluatie op `EVAL_VM`:
standaard mbert-corpus (die de keten al heeft); met `EVAL_VM=mbert-mix` neemt de keten daarvoor `final_eval_mix.lock`
(ten hoogste 12 u wachten) en zet hij `~/mbert-t3` daar op dezelfde commit. Op beide dezelfde controle vóór de
evaluatie: de venvs `~/mbert-dumb/.venv` en `~/mbert-jax/.venv` met vingerafdruk `91a3c3a5f544` en `~/mbert-dumbx` op
`ae11478`, op alle 4 workers.

Elke `gcloud … tpu-vm ssh` met `timeout 900` en `ConnectTimeout`/`ServerAlive`; de keten stopt alleen zijn eigen
trainers (op hun runnaam), verwijdert niets en eindigt op `DONE …` of `FAILED: …`, gevolgd door de `EXIT`-regel.
Elke start draait een eigen kopie van het script (`dx/t3/chain-scripts/t3_run.<TAG>.sh`): bash leest een script
tijdens het lopen, en een latere bewerking van `t3_run.sh` mag een lopende keten niet veranderen. Een herstart (nieuwe
TAG) slaat de gemarkeerde stappen over en volgt trainers die al lopen (hun eigen commando zegt tot welke stap en in
welke precisie). Zolang de keten niet loopt, kopieert niemand checkpoints naar `ckpt_keep/` (de trainer houdt de laatste
3, dus 30.000 stappen of ~2,7 u): een gestopte keten moet binnen die tijd weer lopen.

**Droge run [gemeten, 22 nov, co1, nep-`gcloud`/`gsutil` met nep-workers en een nep-bucket; de datascripts en de
tokenizerbouw echt, trainer en TPU-jobs nep met T2's echte uitslagen als basis]:** wachten op G4-7 (proces leeft, geen
`EXIT`; override terwijl het slot bezet is: blijft wachten; `EXIT 0`: neemt het slot); `EXIT 1` met holdbestand: blijft
wachten, zonder holdbestand na de wachttijd verder; setup, data (held-out op w0 gevonden, een exacte kopie op w2 en een
witruimtevariant op w3 ook; bewijs 0), smoke-test met hervatting en geheugen; hoofdrun met burst 1 (> 50% in 1.000
stappen) → float32-stuk → bf16; keten gestopt en herstart; tak met burst 2 (10 op rij) → float32-stuk; evaluatie
met een mislukte fine-tune → `FAILED`, herstart doet alleen die taak opnieuw; hoofdrun verder (eerst float32 tot P van
burst 2), keten gestopt tijdens het trainen en herstart (volgt de lopende trainers), een reboot (trainers weg zonder
exitbestand → hervat), burst 3 (zelfstop) → de rest float32; evaluatie, `decide_t3.py`, rapport, `DONE`. Apart: een
geheugenfout in de smoke-test → `FAILED`, niets getraind. De nep-`gsutil rsync` faalt naar een lokale map die niet
bestaat, zoals de echte.

## 10. Evaluatie

**Armen.**

| arm | model | checkpoint | uitslagen |
|---|---|---|---|
| T3-hoofd | `t3-1959000` | `gs://…/mbert/t3/t3-culturax-317m/ckpt_keep/1959000` | nieuw, `gs://…/mbert/t3/{dumb,dumb-extra,blimp-nl,memtest,heldout-eval}/` |
| T3-tak | `t3-epoch2-1306000` | `gs://…/mbert/t3/t3-culturax-317m-epoch2/ckpt_keep/1306000` | nieuw, idem |
| T2 | `epoch2-1306000` (`yhavinga/dmbert-230m-culturax`) | `gs://…/mbert/final-culturax-230m-epoch2/ckpt_keep/1306000` | fine-tunes bestaan (`gs://…/mbert/final/…`, DBRD@512 in `gs://…/mbert/final/robbert/dumb/`); memorisatietest en held-out-evals nieuw (T3's steekproef) |
| A, A2 (ablatie) | `abl-a-100000`, `abl-a2-100000` | | bestaan (`gs://…/mbert/v3-ablation/…`): alleen voor de ruis van de pretraining |

T2's fine-tune-uitslagen worden hergebruikt: dezelfde code (`ae11478`), venvs (vingerafdruk `91a3c3a5f544`) en
hardware (v4-32, per worker een TPU van 4 chips). De keten controleert code en vingerafdruk vóór elke evaluatie; wijken
ze af, dan `FAILED`.

**Taken en instellingen** (gelijk aan v2's evaluatie en de ablatie; `eval_t3.sh` = `eval_v2.sh` met de vorm als
parameter): DBRD@1024 en DBRD@512 (`finetune.py`, update 192, opgevuld tot 128), SICK-NL@512 (update 256), alle drie LR
5e-5 constant, 3 epochs; SQuAD-NL v1.1 F1 (384, doc-stride 128), CoNLL-2002 entity-F1, POS LassySmall en POS Alpino
(`finetune_extra.py`, update 32, LR 5e-5 lineair naar 0, 3 epochs); seeds 42, 43, 44; BLiMP-NL zero-shot
(PLL-word-l2r, 8.400 paren, float32). De fine-tunes, BLiMP-NL, de memorisatietest en de held-out-evals gebruiken
T2's tokenizer (`yhavinga/dutch-llama-tokenizer`; voor tekst dezelfde ids als (a)), zoals de ablatie: de evaluatie is
voor T2 en T3 gelijk. 22 uitslagen per T3-arm.

**Niet beslissend, gerapporteerd:**
- **Memorisatietest** (`experiments/final-v2/memtest.py`, ongewijzigd): 1.000 documenten uit de trainingsdata van T3
  (getrokken uit de lijst van worker 0, seed 42, zonder `hebban.nl`-documenten) tegen 1.000 van de held-out, één op één
  gekoppeld op tokenaantal, elk 64–1.022 tokens; MLM-verlies en -accuratesse (15%, maskerseeds 0–2) en
  span-reconstructie (16 tokens), verschil train − test met bootstrap-interval en sign-flip-p. Op T3-hoofd, T3-tak en
  T2. **[afgeleid]** T2 zag beide helften (de held-out twee keer), dus bij T2 verwacht ~0; bij T3 meet train − test
  wat T3 van geziene tekst onthoudt (de tak 2×, de hoofdtak 3× gezien).
- **Held-out-evals** (`scripts/eval_windows.py --windows 1024`): T3-hoofd, T3-tak en T2 op T3's held-out
  (`~/heldout.parquet`; voor T2 geziene tekst) en op v2's held-out (opt-out-FineWeb-2 `kept/test` min `kept/train`,
  `gs://…/mbert/final-v2/data/heldout/heldout.parquet`, md5 `34a7f646…`; CulturaX en FineWeb-2 overlappen, dus niet
  per se ongezien).
- De EVAL-curve van de trainer (T3's eigen held-out), staptijden, overgeslagen stappen en float32-stukken.

## 11. Beslisregel (vooraf vastgelegd; `experiments/t3/decide_t3.py`, exact op tellingen)

**Statistiek** (als `experiments/final/decide.py`): per arm en taak het gemiddelde over de seeds en de seed-range
(max − min), exact (`test_accuracy` als k/n, POS op woordaantallen, F1 als `Fraction` van de opgeslagen float, BLiMP-NL
op paren); elke uitslag op de instellingen van §10 gecontroleerd; een ontbrekende of afwijkende uitslag is `FOUT`.

**Drempel per taak [Yeb: de drempelvorm van de ablatie]** = de grootste van: de seed-range van de ene arm, die van de
andere, en de ruis van de pretraining |gemiddelde A − gemiddelde A2| uit de ablatie (`decide_v3.py`: `max`, niet de
som). BLiMP-NL: de grootste van 0,5 punt en |A − A2|. "Meer dan" is strikt; "beter"/"slechter" als het verschil van de
gemiddelden groter is dan de drempel, anders "gelijk".

| taak | ruis |A − A2| [gemeten, `dx/v3-ablation/decision.json`] |
|---|---:|
| SICK-NL | 0,32 punt (47/14.592) |
| DBRD@512 | 0,016 (1/6.336) |
| DBRD@1024 | niet gemeten in de ablatie; **[eigen keuze]** die van DBRD@512 |
| SQuAD-NL F1 | 1,02 |
| CoNLL-2002 F1 | 0,67 |
| POS LassySmall | 0,11 (94/86.985) |
| POS Alpino | 0,11 (19/16.569) |
| BLiMP-NL | 0,05 (1/2.100; dus 0,5 punt geldt) |

`decide_t3.py` rekent de ruis zelf uit de uitslagen van A en A2.

**Oordeel van X tegen Y** (T2's regel, `decide.primary_rule`) op DBRD@1024 en SICK-NL: "**schaadt**" als X op een van
beide slechter is (eerst getoetst); anders "**helpt**" als X op SICK-NL beter is; anders "**geen verschil**" (winst op
DBRD@1024 alleen is geen verschil). **Extra taken:** DBRD@512, SQuAD-NL F1, CoNLL-2002 F1, POS LassySmall, POS Alpino,
BLiMP-NL.

**Vraag 1, beslissend: T3-hoofd tegen T2.**

| uitkomst | voorwaarde | gevolg (advies; publiceren alleen op Yebs woord) |
|---|---|---|
| **vervangt T2** | "helpt" en geen extra taak slechter; of "geen verschil", minstens één extra taak beter en geen slechter | T3 wordt het dmbert-model van CulturaX in plaats van T2 |
| **naast T2** | niet "vervangt", geen "schaadt", en op minstens één van de acht taken beter | T3 onder een eigen naam naast T2 |
| **niet publiceren** | anders: "schaadt", of nergens beter | T3 niet publiceren; T2 blijft |

**[eigen keuze]** Anders dan v2's regel (waar "geen verschil" zonder verlies al "vervangt" was, omdat v2 om de
opt-out-data te verkiezen was) vraagt "vervangt" hier een gemeten winst: T3 is 1,38× zo groot en dus trager en
duurder in gebruik; bij gelijke uitslagen is er geen reden het kleinere T2 te vervangen. Die drempel is dezelfde als
T2's regel 2/3 (T2 tegen T1) en "voldoet" van de ablatie.

**Vraag 2, gerapporteerd: T3-tak tegen T2.** Zelfde data, zelfde aantal tokens, zelfde schema en cooldown-tak
(1.040.000 → 1.306.000, 1-sqrt). Wat verschilt [afgeleid]: vorm en grootte (18 × 1.152 tegen 22 × 1.024; 317M tegen
230M), de stabiliteitsfix (z-loss, attentie in float32; T2 rekende in bf16 met één float32-stuk 1.010k–1.020k, en sloeg 590 stappen
over tot 520.000 en 1.251 tussen 520.000 en zijn zelfstop op 1.012.690 [LEARNINGS]), tokenizer (a) (tekst-ids gelijk), de 40.000 weggelaten held-out-documenten (0,034%) en de
initialisatie (andere vorm). Betekenis: "helpt" → de grotere vorm met de fix is bij gelijke data beter; "geen
verschil" → vorm en grootte geven bij twee epochs CulturaX geen meetbare winst; "schaadt" → slechter. Beslist niets;
het komt na ~5 dagen als voorlopig rapport (`decision-branch.md`), zodat Yeb kan beslissen of de hoofdrun doorloopt.

**Vraag 3, gerapporteerd: T3-hoofd tegen T3-tak.** Wat verschilt: de helft meer training op dezelfde data (een derde
epoch) en daarmee een langere cooldown (399.000 tegen 266.000 stappen) op een later punt. Betekenis: "helpt" → het
derde epoch helpt; "geen verschil" → geen meetbare winst van het derde epoch; "schaadt" → het derde epoch schaadt
(herhaling). **[eigen keuze]** Bij "schaadt" geeft het rapport ook de uitkomst van de regel van vraag 1 voor T3-tak
tegen T2, als alternatief: Yeb kiest dan of de tak in plaats van de hoofdtak de kandidaat wordt. De uitkomst van de
keten blijft die van vraag 1.

**Geldigheid van de run:** 1.959.000 en 1.306.000 gehaald (`ckpt_keep/…` met `commit_success.txt`); niet-eindige
stappen alleen afgehandeld volgens §7; elke andere afwijking als amendement vóór de evaluatie.

**Controle van de code [gemeten, 22 nov, `tests/test_t3_decide.py`]:** met T2's uitslagen als beide T3-armen geeft
`decide_t3.py` op vraag 1 "geen verschil", nergens beter → **niet publiceren** (een even goed, groter model vervangt
T2 niet). In de droge run (T2's echte uitslagen + 60/4.864 op SICK-NL en +50/8.400 op BLiMP-NL voor de hoofdtak, de
echte ruis van A en A2) gaf hij "helpt" → **vervangt T2**, en voor de tak (+40/4.864 op SICK-NL, 0,82 > drempel 0,76)
"helpt".

## 12. Rekentijd en geheugen (schatting; de smoke-test meet ze) [afgeleid]

**Staptijd.** Gemeten met de fix: A (22 × 1.024) 0,26 s per stap, C (16 × 1.024) 0,225 s (smoke-tests van de ablatie);
C trainde 100.000 stappen in 6 u 35 min inclusief checkpoints en EVAL (0,237 s per stap). Rekenwerk per token
(vooruit, matrices inclusief de decoder plus de attentie over 1.024 posities, die de code ook voor lokale lagen volledig
rekent): A ~552, C ~521, T3 ~719 MFLOP; het elementgewijze deel van de attentie schaalt met heads × lagen (A 352, C 256,
T3 324). Een fit van tijd = a · FLOPs + b · (heads × lagen) op A en C geeft voor T3 **~0,30 s**, met checkpoints en
EVAL **~0,32 s per stap** (Yebs schatting: ~1,4× het rekenwerk van C). Bandbreedte 0,30–0,34 s.

| deel | stappen | bij 0,32 s/stap |
|---|---:|---:|
| setup, data, smoke-test | | ~1–1,5 u |
| hoofdrun 0 → 1.040.000 | 1.040.000 | 92,4 u (3,85 d) |
| tak 1.040.000 → 1.306.000 | 266.000 | 23,6 u |
| evaluatie tak (22 fine-tunes + memtest + held-out) | | ~1,5–2 u |
| hoofdrun 1.040.000 → 1.959.000 | 919.000 | 81,7 u (3,4 d) |
| evaluatie hoofdtak + T2's memtest/held-out | | ~2 u |
| **samen** | 2.225.000 | **~203 u ≈ 8,5 dagen** (0,30 s: ~7,9 d; 0,34 s: ~9,1 d) |

Zonder de tak zou de hoofdtak alleen ~7,4 dagen zijn (Yebs 7,5–8). Elk float32-stuk van 20.000–30.000 stappen kost
+38% over dat stuk (~+0,7–1 u); de rest van de run in float32 vanaf stap s kost +0,38 × (de rest).

**Geheugen.** v2 (230M, bf16) had 17,99 GiB piek per chip [gemeten, v2's smoke-test]. Voor T3: parameters,
gradiënten en Adam (16 B per parameter) +1,4 GiB; de activaties van de attentie schalen met heads × lagen (−8%), die van
de MLP met intermediate × lagen (+57%), de rest met hidden × lagen (−8%). Schatting **~20 GiB per chip (19–23)**, ruim
onder de 32 GiB van een v4-chip. De smoke-test meet het; grens 30 GiB (§8).

## 13. Wat T3 niet zegt

- Eén pretraining per arm; de ruis van de pretraining is die van A tegen A2 bij 100.000 stappen en 230M, niet gemeten
  voor T3 zelf.
- Vraag 2 scheidt vorm, grootte en de fix niet van elkaar.
- De held-out is vrij van exacte en witruimte-varianten van zijn documenten; bijna-duplicaten (andere woorden, andere
  opmaak) en dezelfde tekst in een andere bron (FineWeb-2) kan hij niet uitsluiten.
- 15% losse tokens, 1.024 tokens, CulturaX: geen uitspraak over andere recepten.

## 14. Open vragen voor Yeb

1. **Na de tak:** als T3-tak tegen T2 "schaadt" (vraag 2), loopt de hoofdrun volgens dit protocol toch door (~3,5 dag).
   Wil Yeb dat de keten dan stopt? (Standaard: doorlopen.)
2. **Derde burst → de rest in float32** (§7.3, naar zijn besluit voor v2): akkoord, ook als dat dagen kost? Of dan
   eerst vragen?
3. **Een andere `EXIT` dan 0 van G4-7** (§9.2): de keten neemt mbert-corpus na 60 minuten, tenzij `t3.corpus-hold`
   bestaat. Liever alleen bij `EXIT 0` of het overridebestand?
5. **Vervangen vraagt winst** (§11): anders dan v2's regel is "geen verschil" zonder extra winst hier "niet
   publiceren", omdat T3 groter is. Akkoord?
4. **DBRD@1024** heeft geen ruismaat uit de ablatie; ik neem die van DBRD@512 (0,016 punt, in de praktijk telt dan de
   seed-range).

## Amendementen

(geen)
