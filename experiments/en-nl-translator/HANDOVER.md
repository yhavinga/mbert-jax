# Overdracht: pilot, LoRA-smoke-test en hoofdset van de en→nl-vertaler (stand 9 nov 2026)

## Hoofdset tot vlak vóór het vertalen (9 nov, 13:45–15:30 UTC; opdracht "hoofdset voorbereiden")

Geen vertaling, geen drempels, geen fine-tune. `R=gs://t5-dutch-english-usc2/mbert/translator`, `M=$R/mainset`.
Overzicht voor Yeb: `$M/mainset-prep.md` (kopie op co1: `~/Developer/yhavinga/mbert-ops/translator-pilot/mainset-prep.md`;
nu zonder sectie 3, die vult `finish_mainset.sh` na de bronscore). Besluiten waar dit op rust: ontwerp §3 stap 0, §7
punt 6 en 7, §9 (hammond `mbert/designs/en-nl-translator.md`, `63eaf66`) en de richtlijn "streng op de bron, mild op
het paar".

### Wat loopt (stand 15:30 UTC)

- mbert-corpus, alle 4 workers, elk één los proces (single-host, mesh tensor 4): de bronscore van Gemma 4 31B op de pool,
  run 1 `translator_source_main_1` (start 15:12 UTC; 37 shards van 2.000 rijen per worker; ~3,5 uur bij ~3.700
  prompttokens/s per worker). Log `~/logs/translator_source_main_1.log` (`SHARD`-regel per shard, `EXIT` aan het eind,
  kopie naar `$M/logs/`). Scores per shard in `$M/source-scores/w<k>/`.
- Op elke worker een keten (`~/translator_source_chain.sh`, log `~/logs/translator_source_chain_1.log`): wacht op `EXIT 0`
  van run 1 en op `$M/judge-in/fineweb-ready.txt`, en start dan run 2 `translator_source_main_2` (dezelfde taak, dezelfde
  mappen; scoort alleen de FineWeb-2-shards `shard-1NNNN`). Eindigt run 1 niet op `EXIT 0`, dan start run 2 niet.
- FineWeb-2 (`HuggingFaceFW/fineweb-2` @ `af9c133`, nld_Latn, ODC-By): selectie klaar (`$M/pool-fineweb/`, 93.000 +
  1.860 eval uit 240 row groups; 53.868 van 240.000 documenten zonder .nl/.be-domein, 10.440 met een taalpad); de
  shards staan als `shard-10000`–`shard-10011` in `$M/judge-in/w<k>/` en `fineweb-ready.txt` is gezet, dus run 2 start
  vanzelf na run 1. Verwacht einde run 1 ~17:50 UTC, run 2 ~18:50 UTC (19:50 CET), ruim vóór 00:30 CET.
- Niets anders van deze opdracht. mbert-mix en `en-nl-judge` (europe-west4) niet aangeraakt.

"Klaar" betekent:
- op alle 4 workers eindigen `translator_source_main_1.log` en `translator_source_main_2.log` op `EXIT 0`, en
  `$M/source-scores/w<k>/` heeft evenveel shards als `$M/judge-in/w<k>/`;
- `$M/mainset-prep.md` heeft sectie 3 (verdeling per bron, herkomst, onderwerpen, 5+5 voorbeelden per bron) en
  `$M/minipilot-long/` bestaat; kopie van `mainset-prep.md` op co1;
- mbert-corpus vrij: `sudo lsof -t /dev/accel* /dev/vfio/*` leeg op alle workers, geen `judge.py`.

Hoofdsessie: `setsid nohup bash ~/Developer/yhavinga/mbert-ops/translator-mainset/finish_mainset.sh >
~/logs/finish_mainset.log 2>&1 < /dev/null &` (pollt elke 180 s; maakt na afloop de minipilot en het verslag; verwijdert
niets, start niets op de TPU's). Daarna opruimen, met zichtbare commando's (alles van deze opdracht, niets anders):
`rm -rf /dev/shm/judge` op alle workers (venv, tokenizer, logs; op w3 ook de lokale checkpointkopie van 45 GB) en
`rm -rf /dev/shm/translator/mainset` op alle workers. Het checkpoint blijft in `$R/judge/gemma-4-31b-it-maxtext/`.

### Stappen en uitkomsten

1. **Stap 0 v2** (`select_passages.py`). Regels, elk met unit-tests (`tests/test_en_nl_select.py`, ook voor goede tekst
   die moet blijven staan): `encoding`, `soft_hyphen`, `hyphenation` (nu ook met hoofdletter, de lettergrepen
   be/ge/ver/ont/her, en "enthousias- merend"), `sentence_start`, `cut_off`, `lower_after_end`, `glued` (camelCase met
   kleine beginletter of aanspreektitel, leesteken zonder spatie, cijfers aan een woord), `heading_glued`,
   `mid_capital`, `page_mark`, `footnote`, `repeated`, `split_word`, en de contextregels `open_before` en
   `continues_after` (de alinea vóór en na de passage). Op de 2.000 pilotpassages (opnieuw opgebouwd met `annotate`):
   **188 van de 288** door Gemma afgekeurde gevangen, **99 van de 1.692** goedgekeurde tegengehouden (46 daarvan door
   `heading_glued`, 25 door `sentence_start`: echte zinsbeginnen midden in een zin die Gemma doorliet). Per regel en
   per bron: `mainset-prep.md` §1. Bewust niet: een beletselteken midden in de tekst (zo schrijven de Handelingen een
   interruptie), en een regel ervoor zonder punt als zodanig (dat waren vooral koppen en tabelrijen).
2. **Pool** (`$M/pool/`: `passages.parquet`, `eval.parquet`, `context-turns.parquet`, `report.json`). Verdeling
   40/20/7,5/7,5 van 200.000 (+ FineWeb-2 25%), met 30% marge op de afkeur van de pilot: Handelingen 106.667, raad 78.000,
   dpc 21.196, rechtspraak 25.491; eval 2.134/1.560/424/510. Passages tot 768 tokens, per bron 25% lang (> 384), kolom
   `length_class`; `half` (idioom/willekeurig), `idiom_class`, `doc_id`, `source`. `--max-per-doc handelingen=15,raad=5`
   (anders 3): met 3 haalden de Handelingen het niet (alle 51.104 debatten gelezen; capaciteit staat nu in
   `report.json`). Uitgesloten via 13-grammen: DBRD-test, SICK-NL-test, FLORES-200 devtest (uit Meta's tarball:
   `facebook/flores` en `flores_plus` zijn afgeschermd voor Yebs token), WMT24++ en→nl_NL, NTREX-128 (`$M/testsets/`).
3. **Context (§9)**. Elke Handelingen-passage heeft `context_turns`/`context_tokens`: de 1–3 beurten ervoor (ook de
   voorzitter), elk door de bronfilters en ≤ 768 tokens, samen ≤ 1.024. 21.760 passages (20%, seed vast) zijn
   `context_example`; hun 52.564 contextbeurten (`context-turns.parquet`) kosten **7.340.633 extra nl→en-tokens**
   (+11,9% op de 61,5 M tokens van de vier bronnen) en krijgen dezelfde bronscore (split `context`).
4. **Bronscore** (`judge.py` van `en-nl-judge`, hier gemerged en uitgebreid). Omgezet uit `google/gemma-4-31B-it` @
   `842da37` naar `$R/judge/gemma-4-31b-it-maxtext/` (4,8 min op w3). Op v4: FSDP 4 vraagt 31,3 GB HBM aan tijdelijke
   buffers (> 30,75): tensor 4 werkt (fsdp 2 × tensor 2 ook, trager). Taak `source_topic_origin`: één forward pass met
   de bronvraag van `source-quality.md` (onveranderd, door causale attentie), daarna "leest als oorspronkelijk
   Nederlands?" en een onderwerp uit 15 (A–O). Op de pilot: zelfde bronscore als de v6e-8 (mediaan log-odds-verschil
   0,15; 5 van 2.000 wisselen rond 0,1). Tempo op de pilot: 10,7 passages/s per worker met alleen de bronvraag, 7,7 met
   beide vervolgvragen (~3.700 prompttokens/s).
5. **FLAN-nl-replay** (`$M/flan-eval/`, `$M/replay/`). Eerst 1.000 + 35 evaluatierijen uit F (status translate,
   instructie 48–384 tokens, seed `en-nl-translator/flan-eval/1`; 2.626 segmenten). Daarna van 1.811.525 segmenten:
   −2.626 evaluatierij, −1.267 niet afgelopen, −1.121.337 buiten 48–384 tokens Nederlands, −13 `decode_loop`, −120
   `english_left`, −164.833 structuur (69.719 alleen `newline_doubled`) → **521.329 paren**.
6. **Structuurcontrole** (`structure.py`, `pairs.py measures` schrijft `structure` en `back_structure`). Pilot: nl→en
   326 paren met verschil, waarvan 287 alleen `newline_doubled`; en→nl 20 (15).

### Commando's (elk ssh zoals hieronder in "Commando's"; scripts in `$R/scripts/mainset/`, op de workers in
`/dev/shm/translator/mainset/code/`; de co1-kant in `~/Developer/yhavinga/mbert-ops/translator-mainset/remote/`)

- Judge-omgeving per worker: `bash /dev/shm/judge/code/setup.sh` (us-central2 wordt herkend); omzetten:
  `bash setup.sh convert` (staat al in de bucket).
- Pool: `select_passages.py select --out <dir> --n 231354 --counts handelingen=106667,raad=78000,dpc=21196,rechtspraak=25491
  --n-eval 4628 --eval-counts handelingen=2134,raad=1560,dpc=424,rechtspraak=510 --max-per-doc handelingen=15,raad=5
  --row-groups handelingen=100,raad=100,dpc=32,rechtspraak=24 --block-texts <5 testsets> --processes 200` (w0, 12 min).
- FineWeb-2: dezelfde met `--row-groups handelingen=0,raad=0,dpc=0,rechtspraak=0,fineweb=240 --max-docs fineweb=400000
  --counts fineweb=93000 --eval-counts fineweb=1860`.
- Shards: `mainset.py judge-shards --select <pool> --out <dir> [--first-shard 10000]`, naar `$M/judge-in/w<k>/`.
- Bronscore per worker: `TASK=source_topic_origin FSDP=1 TENSOR=4 TOKEN_BUDGET=16384 SHARDS=$M/judge-in/w{w}
  OUT=$M/source-scores/w{w} NAME=<nieuw> bash run_source_judge.sh` (hervat per shard).
- Pilottoets: `select_passages.py annotate --pids <pilot-pids> --row-groups handelingen=8,raad=12,dpc=8,rechtspraak=6`,
  daarna `mainset.py stage0-eval` (op co1, kleine bestanden).
- Verslag: `finish_mainset.sh` (of `mainset.py minipilot` en `mainset.py report`, zie het script).

### Open punten

1. **Drempels** (Yeb): bronscore, herkomst (FineWeb-2) en eventueel onderwerp; er is er geen gekozen. De log-odds
   staan erbij (`logp_yes`, `logp_no`, `origin_logp_*`, `topic_logps`).
2. **`heading_glued`** houdt 46 van de 1.692 door Gemma goedgekeurde pilotpassages tegen (koppen die aan de alinea
   vastzitten, vooral raad en dpc). Streng gekozen; Yeb kan de regel laten vallen (één regel in `extraction_errors`).
3. **Alinea-scheiding**: passages zetten alinea's met één regelovergang onder elkaar; TranslateGemma maakt er bij nl→en
   in 287 van de 2.000 pilotparen een lege regel van (`newline_doubled`). Vóór stap 1 beslissen: alinea's met een lege
   regel scheiden (markdown), of `newline_doubled` als toegestaan verschil behandelen.
4. **FineWeb-2** heeft geen pilot: de 30% afkeur is aangenomen. De 13-gramontdubbeling liep per selectie; tussen de
   FineWeb-pool en de pool van de vier bronnen is niet ontdubbeld (overlap is onwaarschijnlijk, niet gemeten).
5. **Context-voorbeelden** zijn gekozen uit passages die bij de eerste of tweede alinea van hun beurt beginnen; een
   eerder deel van de eigen beurt is geen context. De contextbeurten zijn niet idioomgeselecteerd (zoals bedoeld).
6. **Replay**: de structuurcontrole laat 521.329 van 686.162 paren over (24% af); ruim genoeg voor het aandeel van 20%.
   De cursiefregel telt ook ontsnapte underscores (`\_\_\_`) mee.
7. **Evaluatierijen**: de band 48–384 tokens voor de 1.000 + 35 FLAN-rijen is een eigen keuze (zelfde band als de
   pilotpassages), zodat de classifier en Yeb geen piepkleine rijen krijgen.
8. **Niet van deze opdracht, wel op mbert-corpus**: op alle workers draaien verweesde processen uit
   `/dev/shm/translator/smoke-lora/venv` (`multiprocessing.resource_tracker` en `spawn_main`, PPID 1, gestart tussen
   ~13:49 en ~15:06 UTC op alle workers tegelijk). Ze houden de TPU niet vast. Niet gestopt: herkomst onbekend.
   Vóór 00:30 CET moet mbert-corpus zonder processen zijn; de hoofdsessie beslist.
9. 9 van de 2.000 pilotteksten wijken bij `annotate` af door commit `9853d3c` (afkortingen in de zinsplitsing, van na
   de pilotselectie); hun Gemma-score hoort bij de oude tekst.
10. GPT-NL's verzonnen adressen ("Liamstraat 0", "DorMellering 8") vangt alleen `glued` als er cijfers aan vastzitten.

Verkeer: tussen regio's niets (geen bucket-naar-bucket). Naar co1, eenmalig en klein: de bronscores van de pilot uit
`gs://t5-dutch-english` (134 kB), `passages.parquet` en `pairs.parquet` van de pilot (0,5 en 1,4 MB), vier versies
van `annotated.parquet` (~0,7 MB elk), vier pilot-scoreshards van v4 (~0,15 MB elk), `report.json` (22 kB) en
`mainset-prep.md` (13 kB; `finish_mainset.sh` haalt hem nog één keer). Van co1 naar de bucket: de OpenTaal-lijst
(5,1 MB, die co1 van GitHub haalde) en `stage0-eval.json`. Van HF naar de VM's: Gemma 4 31B (59 GB, w3; daarna van
/dev/shm verwijderd), de testsets, de row groups van de bronnen en van FineWeb-2; van Meta naar w1: het FLORES-200-tarball.



## LoRA-smoke-test (9 nov, 12:45–14:30 CET): klaar

Vraag: draait MaxText `train_sft` (op Tunix) met LoRA voor TranslateGemma op v4, en vertaalt het model daarna nog?
Antwoord: ja. Het verslag met metingen en de 10 passages naast elkaar staat in `$R/smoke-lora/smoke-lora.md` (kopie op
co1: `~/Developer/yhavinga/mbert-ops/translator-pilot/smoke-lora.md`). `S=$R/smoke-lora` hieronder.

"Klaar" betekent:
- `$S/runs/sft12b_{N,B}/` bevat `summary.json`, `adapter.npz` en `adapter.json`, en de logs eindigen op `EXIT 0`.
- `$S/compare/{base,N_tag,N_plain,B_plain}/` bevat de shards en `$S/passages.md`; `$S/smoke-lora.md` bestaat.
- mbert-corpus is vrij.

Code (alles in `experiments/en-nl-translator/`, kopieën in `$R/scripts/`, op de workers in
`/dev/shm/translator/smoke-lora/code/`):

- `setup_sft.sh`: eigen venv `/dev/shm/translator/smoke-lora/venv`, met MaxText 0.2.5 en Tunix `a8d70582` (de pin
  van MaxText); met `convert` zet het ook TranslateGemma 12B om naar `$S/translategemma-12b-maxtext`.
- `sft_data.py build|check`: de paren als arm N (tag `<BT> `) en arm B, 10 achtergehouden, de vergelijkingsset en het
  chat-template; `check` draait MaxText's eigen SFT-stappen op elk voorbeeld.
- `sft_lora.py` (MaxText's `setup_trainer_state`/`train_model` met metingen, epoch geteld, adapter als npz) en
  `run_sft.sh` (single-host of `MULTIHOST=1`).
- `translate.py --lora <map>` (in `experiments/flan-nl/`) voegt de adapter na het laden in de kernels samen; `run_compare.sh`
  draait de vier varianten; `smoke_report.py` zet ze naast elkaar.
- `hf_loss_check.py` geeft het referentieverlies in transformers.
- Tests: `tests/test_en_nl_sft.py`, plus `tests/test_en_nl_lora_merge.py` (qwix, op een worker met
  `JAX_PLATFORMS=cpu`).

Commando's (zelfde ssh-vorm als hieronder; `C=/dev/shm/translator/smoke-lora/code`, `D=/dev/shm/translator/smoke-lora/data`):
1. Alle workers: `bash $C/setup_sft.sh` (~1 min), op één worker `bash $C/setup_sft.sh convert` (~5 min).
2. Worker 0: `sft_data.py build --pairs $D/pairs.parquet --flan-en $D/flan-en-w0.parquet --flan-examples
   $D/flan-plan2-examples.parquet --tokenizer /dev/shm/flan_nl/tokenizer-12b --out $D`, daarna `sft_data.py check`
   (beide met `JAX_PLATFORMS=cpu`). Daarna `$D` naar `$S/data/` en naar de andere workers.
3. Training, per worker: `NAME=sft12b_N MODEL=12b DATA=$D/train-N.parquet bash $C/run_sft.sh per_device_batch_size=3`.
   Duurt ~8 min: ~1 min laden, ~4 min compileren, 59 × 2,2 s.
4. Vergelijking: `VARIANT=base|N_tag|N_plain|B_plain bash $C/run_compare.sh`, ~5 min per variant. Daarna
   `smoke_report.py --data $D --compare <kopie van $S/compare> --out passages.md`.
5. 27B over 4 workers: op alle workers tegelijk `MULTIHOST=1 NAME=<naam> MODEL=27b STEPS=8 DATA=$D/train-N.parquet
   bash $C/run_sft.sh per_device_batch_size=1`; MaxText's uitvoer gaat dan naar `$S/runs/<naam>/maxtext`.

Uitkomsten: zie `smoke-lora.md`. Kort:
- De route werkt op 12B en 27B.
- 12B, één worker: 4.755 tokens/s, piek 30,7 GiB per chip, ~4 min compileren. Over 16 chips: 10.135 tokens/s,
  piek 9,65 GiB.
- 27B past alleen goed als één multi-host-job over de 4 workers, in fp32 (gewichten en LoRA): piek 19,9 GiB (1 per
  chip) en 21,2 GiB (2 per chip), 2.072 en 4.018 tokens/s, ~7 min compileren. Op één worker past het alleen met
  bf16 en 1 per chip, zonder marge (30,69 van 30,75 GiB).
- Verlies 3,03 → ~0,9 in één epoch. Het beginverlies klopt met transformers (3,072).
- Inferentie met de samengevoegde adapter werkt (toets tegen qwix). De vertalingen gaan niet stuk, maar arm N met tag
  laat gedrag zien dat bewaakt moet worden: de tag lekt één keer, één keer een meta-zin, één keer een weggelaten zin,
  en bij een FLAN-vraag een toegevoegd antwoord.

Open punten van de smoke-test:
1. Geheugen: 12B met fp32-gewichten en 3 sequenties per chip zit op 30,7 van 30,75 GiB. Met bf16-gewichten (16,9 GiB,
   ~20% sneller) worden de LoRA-parameters ook bf16 (qwix neemt het dtype van de kernel over); bij LR 1×10⁻⁵ is dat
   te grof. Een cast van de LoRA-parameters naar fp32 na `apply_lora_to_model` is niet geprobeerd.
2. 27B met 2 sequenties per chip vraagt op één worker 43 GB per chip (over 16 chips: 21,2 GiB), ook met vocab-tiling en `dot_product`; de oorzaak is niet
   gevonden. Rematerialisatie per ongescande laag staat aan (`prevent_cse`).
3. `scan_layers=true` (sneller compileren, mogelijk minder geheugen) vraagt een gescand checkpoint en een omzetting van
   de gestapelde LoRA naar `translate.py`'s ongescande lagen; niet gedaan.
4. De tag `<BT>` als platte tekst (keuze van deze test) en de ongefilterde paren: zie `smoke-lora.md`.
5. Tensor-parallel (`ici_tensor_parallelism=4`) brak op een `IndivisibleError`; niet verder onderzocht.
6. `/dev/shm/translator/smoke-lora/` op de workers bevat de venv, de data, de runs en op w3 `hf-12b` (23 GB
   safetensors). Niets daarvan is opgeruimd.
7. Bootdisk:
   - MaxText 0.2.5 dumpt bij een lege `XLA_FLAGS` elke `jit_train_step` naar `/tmp/xla_dump`, ook bij
     `dump_hlo: false`. Dat vulde op 9 nov ~13:20 UTC de bootdisk van w0; `/tmp/xla_dump` (alleen van deze runs) is
     verwijderd.
   - `run_sft.sh` zet nu een neutrale `XLA_FLAGS` en `jax_cache_dir` in `/dev/shm`. Bestanden in `~/jax_cache` van
     vóór die fix (van `sft_lora.py` en van `translate.py` via `run_compare.sh`, ~0,1–0,3 GB per worker) staan er
     nog.
   - `translate.py` zet MaxText's `jax_cache_dir` niet, dus ook FLAN-nl's runs schrijven daar.
8. Multi-host:
   - Proces 0 schrijft de adapter nadat de andere processen klaar zijn. `sft_lora.py` laat ze nu op elkaar wachten
     (`sync_global_devices`); de run zonder die wachtstap eindigde op w0 met EXIT 134, met de bestanden wel
     geschreven.
   - Het geheugen van één worker (FSDP over 4 chips) is veel hoger dan over 16 chips (12B: 30,7 tegen 9,65 GiB);
     niet verklaard.
9. De 27B-adapter van `mh27b_fp32_b1` staat in `$S/runs/`; die is niet door `translate.py` gehaald.

Branch `en-nl-translator` van github.com/yhavinga/mbert-jax (vanaf `flan-nl` @ `7acefa2`). Ontwerp:
hammond `mbert/designs/en-nl-translator.md` (besluiten §7). Code in `experiments/en-nl-translator/`, plus
`experiments/flan-nl/translate.py` (nu met `--source-lang/--target-lang/--work-dir`). Tests:
`tests/test_en_nl_translator.py`, `tests/test_en_nl_select.py`, `tests/test_en_nl_pairs.py`.
`R=gs://t5-dutch-english-usc2/mbert/translator` hieronder.

## Status: de pilot is klaar

Gedaan: stap 0 (2.000 passages), stap 1 (nl→en) en stap 2 (en→nl terug, filters, MetricX-24-QE) met TranslateGemma
27B op de 4 workers van mbert-corpus, en het rapport voor Yeb. Niets draait meer; mbert-corpus is vrij (9 nov 11:42
UTC: `sudo lsof -t /dev/accel* /dev/vfio/*` leeg op alle workers). mbert-mix niet aangeraakt.

"Klaar" betekent: `$R/pilot/report.md`, `$R/pilot/pairs30.md`, `$R/pilot/summary.json`, `$R/pilot/pairs.parquet`
(2.000 paren met alle metingen) en `$R/pilot/qe.parquet` bestaan; de logs eindigen op `EXIT 0`
(`$R/pilot/logs/translator_{nl_en,en_nl}-w<N>.log`, `translator_qe3-w0.log`). Kopie van `pairs30.md` op co1:
`~/Developer/yhavinga/mbert-ops/translator-pilot/pairs30.md`.

Wat er nu op Yeb wacht: de drempels van chrF, MetricX-QE en de ja/nee-filters, op basis van `report.md` en
`pairs30.md`. Er is geen drempel vastgesteld.

## Bucket

- `$R/scripts/`: alle scripts die op de VM's draaiden (ook `translate.py`, `publish.py`, `flan_nl.py`,
  `build_decay_mix.py`, die `pairs.py` importeert), en `tests/`.
- `$R/pilot/select/`: `passages.parquet` (de 2.000, met bron, document, helft idioom/willekeurig, idioomtreffers,
  metingen), `segments-w<k>.parquet` (invoer van `translate.py`), `rejected.parquet` (steekproef per filter),
  `report.json` (tellingen, verdelingen, gelezen row groups, sha256 van de woordenlijsten).
- `$R/pilot/resources/`: `idioms.json` (1.526 uitdrukkingen) en `manifest.txt` (herkomst en sha256 van OpenTaal
  @ `b250510` en de kaikki.org-extractie van 30 okt 2026).
- `$R/pilot/nl-en/w<k>/<klasse>/shard-*.parquet` (`text_en`), `$R/pilot/back/` (Engelse segmenten),
  `$R/pilot/en-nl/w<k>/<klasse>/shard-*.parquet` (`text_nl`).
- `$R/pilot/pairs.parquet`, `qe.parquet`, `report.md`, `pairs30.md`, `summary.json`, `logs/`.

## Commando's (zoals de pilot ze draaide)

Elke ssh: `gcloud compute tpus tpu-vm ssh mbert-corpus --zone us-central2-b --project t5-dutch-english --quiet
--ssh-flag="-o ConnectTimeout=30 -o ServerAliveInterval=30 -o ServerAliveCountMax=4" --worker=<w|all> --command
"..." < /dev/null`, binnen `timeout 900`. Op co1 doen `~/Developer/yhavinga/mbert-ops/translator-pilot/vm.sh` (een
scriptbestand als commando) en `poll.sh` precies dat; de gebruikte scriptjes staan in `.../translator-pilot/remote/`.

1. Omgeving, alle workers (~1 min): `setup.sh` naar `/dev/shm/translator/code/` en
   `bash /dev/shm/translator/code/setup.sh` (venv met JAX/TPU, transformers, torch-CPU; OpenTaal en kaikki in
   `/dev/shm/translator/resources/`). TranslateGemma draait in FLAN-nl's venv `/dev/shm/flan_nl/venv` (alleen
   gelezen); na een reboot eerst `experiments/flan-nl/setup.sh` (zonder `convert`).
2. Idioomlijst en selectie, worker 0 (~90 s, CPU):
   ```
   PY=/dev/shm/translator/venv/bin/python; C=/dev/shm/translator/code; RES=/dev/shm/translator/resources
   $PY -I $C/select_passages.py idioms --kaikki $RES/kaikki-dutch.jsonl --out $RES/idioms.json
   OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
     $PY -I -u $C/select_passages.py select --out /dev/shm/translator/pilot/select --n 2000 --n-eval 0 \
     --idioms $RES/idioms.json --wordlist $RES/opentaal-wordlist.txt --tokenizer /dev/shm/flan_nl/tokenizer-27b \
     --row-groups handelingen=8,raad=12,dpc=8,rechtspraak=6 --processes 96
   ```
   Zonder de `*_NUM_THREADS=1` start elk proces een volle BLAS-pool (langid gebruikt numpy): load 2.000 op worker 0.
   Daarna `gsutil -m cp /dev/shm/translator/pilot/select/* $R/pilot/select/`.
3. nl→en, alle workers (~11 min, waarvan ~2 min laden per lengteklasse):
   `SOURCE=nl TARGET=en SEGMENTS="$R/pilot/select/segments-w{w}.parquet" OUT="$R/pilot/nl-en/w{w}"
   NAME=translator_nl_en bash /dev/shm/translator/code/run_translate.sh` (weigert als de TPU bezet is; log
   `~/logs/translator_nl_en.log`, eindigt op `EXIT <code>`; hervat per shard).
4. Engelse segmenten, worker 0: `gsutil -m rsync -r $R/pilot/nl-en /dev/shm/translator/pilot/nl-en` en
   `$PY -I $C/pairs.py back-segments --select /dev/shm/translator/pilot/select --nl-en /dev/shm/translator/pilot/nl-en
   --tokenizer /dev/shm/flan_nl/tokenizer-27b --out /dev/shm/translator/pilot/back`, dan naar `$R/pilot/back/`.
5. en→nl, alle workers (~12 min): als 3 met `SOURCE=en TARGET=nl SEGMENTS="$R/pilot/back/segments-w{w}.parquet"
   OUT="$R/pilot/en-nl/w{w}" NAME=translator_en_nl`.
6. Metingen, MetricX-24 XXL (TPU van worker 0) en rapport: `bash /dev/shm/translator/code/qe_job.sh` op de
   achtergrond (log `~/logs/translator_qe*.log`; ~3 min, waarvan 25 GB download van HF en 80 s scoren). Bestaat
   `/dev/shm/translator/pilot/qe.parquet` al, dan slaat het de scores over. De XXL-gewichten zijn na de pilot uit
   `/dev/shm/translator/metricx` op worker 0 en 1 verwijderd; `metricx_qe.py` haalt ze opnieuw op.
7. Controle van MetricX tegen transformers (CPU): `JAX_PLATFORMS=cpu $PY -I $C/metricx_qe.py check --pairs <parquet>
   --pair en:nl --model xxl --n 6`.

## Uitkomsten (details in `$R/pilot/report.md`)

- Selectie: 520.677 kandidaatpassages (Handelingen 320.443, raad 105.188, dpc 22.563, rechtspraak 72.483), door de
  filters 490.544, waarvan 480.265 aan de train-kant van de documentsplitsing; gekozen 1.000/600/200/200, elk de helft uit de hoogste idioomklasse.
- Tempo per worker (één v4-host, 4 chips; alleen decodeertijd, ~500 segmenten per worker, dus de laatste ronde
  vult de slots niet): nl→en 504–535 uitvoertokens/s (1.474–1.558 prompt+uitvoer), en→nl 543–559 (1.227–1.263).
  Laden plus compileren ~2 min per lengteklasse.
- Geen enkel paar met `ended = false`, `decode_loop`, Nederlandse resten in het Engels of lengteverhouding > 1,5.
  Ander aantal zinnen 9,8% (verschil ≥ 2: 1,3%), ontbrekend getal 1,8%; samen 10,9%.
- chrF terug/origineel mediaan 62,0 (p5 47,5); MetricX-24-QE en→nl mediaan 2,37 (p95 6,80), nl→en mediaan 3,22.
- MetricX: eigen JAX-port, op de CPU gelijk aan transformers (large en xxl, 6 paren, verschil < 0,00005); bf16 op
  de TPU wijkt ≤ 0,006 af.

## Keuzes in de code die Yeb kan willen terugzien [keuze]

- Passages: het aantal alinea's (1–4) wordt per passage geloot; alinea's > 384 tokens worden in zinsreeksen
  geknipt (`piece`), in de pilot 311 van de 2.000.
- Raad: documenten met `conversational`/`opinion_editorial` wegen 3× in de loting (`RAAD_PREFERRED_WEIGHT`);
  de raadsverslagen zelf zijn meestal `legal_document`/`procedural`.
- Idioomscore: meerwoordige lemma's van de Engelse Wiktionary (kaikki.org); uitdrukkingen in meer dan 0,1% van de
  passages (100 stuks: "sprake zijn van", "dank u wel", "met name") tellen niet mee. Valse treffers komen voor
  ("met smart" in "smart repair"). Hoogste klasse = top 3× het benodigde aantal met score > 0.
- Stage-0-drempels: OOV > 0,15, cijfers > 0,05, leestekens > 0,10, hoofdletterwoorden > 0,30 (presentielijsten).
  129 van de 1.000 Handelingenpassages beginnen met "Voorzitter"; 243 van de 600 raadspassages komen uit
  voorkeursdocumenten.
- Na de selectie van de pilot toegevoegd (dus niet op de pilot toegepast): filter `hyphenation` (woorden die een
  regelafbreking in de bron liet staan, "voor-ge-legd"); in de pilot 5 passages, waarvan 2 echte.

## Open punten

1. Drempels (Yeb). `zinnen` en `getallen` zijn ruis-gevoelig: in steekproeven waren zinsverschillen vaak
   samengevoegde of gesplitste zinnen, en ontbrekende getallen deels omzettingen ("9 ton" → 900,000; "20.00 uur" →
   8:00 PM; "plek 1" → first) en deels ruis in de bron (paginanummers).
2. DBRD/SICK-NL-ontsmetting (§3): `select_passages.py --block-texts` bestaat, maar is niet gedraaid: de opdracht
   verbiedt het gebruik van die testdata; Yeb beslist of het voor de hoofdset mag (alleen als blokkeerlijst).
3. TranslateGemma zet tussen alinea's vaak een lege regel waar het origineel één regelovergang heeft.
4. GPT-NL's verzonnen namen worden alleen herkend bij titel/naam-tegenstrijdigheden (`name_conflict`, 19 in raad);
   "Jet Stook-Bouwhuisen ... Hij" zonder titel wordt niet gevangen (geen voornaam-geslachtslijst).
5. Libtpu-logs in `/tmp/tpu_logs` op de bootdisk (~230 MB per worker, deels van deze runs); niet opgeruimd.

## Volgende stappen (na Yebs drempels; niet in deze opdracht)

- Hoofdset: `select --n 200000 --n-eval 4000` met meer row groups (Handelingen alle ~38, raad ~100, dpc alle 128,
  rechtspraak ~40) en `--max-per-doc` hoger dan 3 voor de Handelingen (51k debatten × 3 = 153k plafond);
  geheugen in het hoofdproces enkele GB. Vertalen: bij het tempo van de pilot ~2,9 uur nl→en en ~3,7 uur en→nl per
  worker voor 50k segmenten, plus laden (volle shards zijn sneller). MetricX: `qe_job.sh` scoort op één host
  (~80 s per 2.000 paren); voor 200k paren per worker een deel scoren (~35 min op 4 hosts).
- Smoke-test van `train_sft` met LoRA op 12B, licentie van echte vertaalparen: aparte opdrachten.
