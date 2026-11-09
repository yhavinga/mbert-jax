# Overdracht: pilot en LoRA-smoke-test van de en→nl-vertaler (stand 9 nov 2026)

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
