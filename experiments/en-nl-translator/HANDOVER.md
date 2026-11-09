# Overdracht: pilot, LoRA-smoke-test en hoofdset van de en→nl-vertaler (stand 9 nov 2026)

## Aanvullende run: passages van 385–511 tokens (9 nov, nacht; Yebs akkoord ~23:15 CET)

`M=gs://t5-dutch-english-usc2/mbert/translator/mainset`, `L=$M/run-long`. In de minipilot liep 384–511 tokens 0 van
248 keer vast; de hoofdrun ging alleen tot 384. Deze run vertaalt de passages van 385–511 Gemma-tokens (na de twee
bewerkingen) die door dezelfde drempels komen, op mbert-corpus, pas na `EXIT 0` van de hoofdrun.

### Wat loopt

Stand 9 nov 22:35 UTC: de keten `translator_longrun.sh` draait losgekoppeld op co1 (tag `11092234`), log
`~/Developer/yhavinga/mbert-ops/logs/translator_longrun.log`. Hij wacht: de regel
`wait: translator_mainrun.log has no EXIT yet` komt elke 180 s terug tot `translator_mainrun.log` op `EXIT <code>`
eindigt. Bij `EXIT 0` (en `$M/run/stages/summary.done`) zet hij `$L/stages/wait.done`, controleert of de TPU van alle
4 workers vrij is en zet de code klaar; bij een andere `EXIT` stopt hij met `FAILED`. Daarna:
1. proefrun op 16 passages (`$M/run-long-dry/`, de 8 langste en 8 andere, 392–511 tokens, ook 1 eval): `nl_en` →
   `back` → `en_nl` → `measures` → `qe` → `summary`, poll 60 s;
2. dezelfde stappen voor de echte run (`$L/`), poll 180 s. Logs per worker `~/logs/tlr_<dry|long>_<stap>_11092234.log`
   (vertaallogs ook in `<run>/logs/`); markers in `<run>/stages/*.done`.

Afwijkingen van de hoofdrun: geen contextbeurten; `back-segments --max-tokens 512` (Engels boven 512 tokens zou in
klasse `xxlong` vallen, die in de minipilot ontspoorde; zo'n paar blijft zonder terugvertaling en chrF en staat in
`back/back_segments.json` onder `too_long`); code op de workers in `/dev/shm/translator/longrun/code/` (kopie van
`gs://t5-dutch-english-usc2/mbert/translator/scripts/longrun/`, commit `e378f1e`), niet die van de hoofdrun.

### "Klaar" betekent

- `translator_longrun.log` eindigt op `DONE translator long run: ...` en `EXIT 0`; `$L/stages/` heeft `wait nl_en back
  en_nl measures qe summary` en `$M/run-long-dry/stages/` `nl_en back en_nl measures qe summary` (`.done`).
- `$L/pairs/pairs.parquet`: dezelfde 26 kolommen als `$M/run/pairs/pairs.parquet`, plus `length_class` (`long`; in de
  tabel van de hoofdrun ontbreekt de kolom, daar is alles ≤ 384 tokens, dus `short`). Samenvoegen:
  `pa.concat_tables([hoofdrun, lang], promote_options="default")` (getoetst op de proefrun van de hoofdrun: de oude
  kolommen gelijk, `length_class` erbij). Verder `$L/pairs/summary.md`, `summary.json`; kopieën op co1 in
  `~/Developer/yhavinga/mbert-ops/translator-longrun/` (`summary.md`, `selection.md`).
- mbert-corpus vrij: `sudo lsof -t /dev/accel* /dev/vfio/*` leeg op alle workers, geen `translate.py` of `metricx_qe.py`.

### ETA

Per worker 7.770 passages, ~3,43 M brontokens, alle in klasse `xlong` (80 slots). Geschat op de hoofdrun (klasse
`long` ~1.260 uitvoertokens/s per worker) en ~3,5 min laden per klasse: wachten ≤ 3 min, proefrun ~30 min, `nl_en`
~45 min, `back` ~5 min, `en_nl` ~55 min, `measures` ~10 min, `qe` ~15–30 min, `summary` ~5 min: samen ~3 uur na
`EXIT 0` van de hoofdrun. De hoofdrun stond om 22:34 UTC in `en_nl`, klasse `medium`; zijn `DONE` verwacht ~00:30–01:00
UTC, deze run dus `DONE` ~03:30–04:15 UTC (04:30–05:15 CET).

### Bij een fout

De keten stopt bij de eerste fout met `FAILED: <wat>` en daarna `EXIT 1`; processen op de workers lopen door.
1. Lees de regel `FAILED` en het log van die stap op de worker (`~/logs/tlr_<dry|long>_<stap>_<tag>.log`).
2. Eindigde de hoofdrun niet op `EXIT 0`, dan start deze run niet: eerst de hoofdrun herstellen (zijn overdracht
   hieronder), daarna deze keten opnieuw starten.
3. Wacht tot geen worker nog rekent; stop een hangend proces alleen met `pkill -9 -f '[t]ranslate.py'` (of
   `[m]etricx_qe.py`) op die worker.
4. Herstart met hetzelfde commando (hieronder). Afgeronde stappen slaat hij over (`<run>/stages/*.done`), `translate.py`
   zijn afgeronde shards. `back` weigert als `/dev/shm/translator/longrun/run-long/back` (of `.../run-long-dry/back`)
   op w0 al bestaat: die map eerst met een zichtbaar `rm -rf` van precies dat pad weghalen.
5. Ontspoort de proefrun (`decode_loop`, niet afgelopen) of `en_nl` op `xlong`: niet doorzetten zonder Yeb; de
   minipilot testte alleen nl→en.

### Commando's

- Keten (co1): `setsid nohup bash -c 'bash ~/Developer/yhavinga/mbert-ops/scripts/translator_longrun.sh >
  ~/Developer/yhavinga/mbert-ops/logs/translator_longrun.log 2>&1; echo EXIT $? >>
  ~/Developer/yhavinga/mbert-ops/logs/translator_longrun.log' > /dev/null 2>&1 < /dev/null &` (flock op
  `~/Developer/yhavinga/mbert-ops/translator_longrun.lock`). Kopie in
  `gs://t5-dutch-english-usc2/hammond/scratch-scripts/translator_longrun.sh`. Omgeving: `FULL`, `DRY`, `MAINLOG`,
  `POLL`, `POLL_DRY`, `TAG`. De `EXIT`-regel schrijft de wrapper, niet het script.
- Elke ssh via `~/Developer/yhavinga/mbert-ops/translator-longrun/vm.sh` (`timeout 900`, de ssh-vlaggen,
  `< /dev/null`); de stukjes voor de workers in `.../translator-longrun/remote/chain/`.
- Selectie (al gedaan, w3, CPU, ~1 min): `mainset.py final --select pool pool-fineweb repaired --scores ... --hosts
  ... --block-texts <testsets> --n 66000 --n-eval 1320 --min-tokens 384 --max-tokens 511 --exclude <de passages.parquet
  van $M/run/select> --no-context --seed en-nl-translator/mainset-long/0`, dan `mainset.py subset --n 16`
  (`.../translator-longrun/remote/select_body.sh`, log `~/logs/tlr_select_11092231.log` op w3).

### Selectie (`$L/select/`: `passages.parquet`, `segments-w<k>.parquet`, `report.json`, `selection.md`)

Gevraagd hoogstens ~25% van het totaal: `--n 66000` (train, verdeeld 40/20/7,5/7,5/25) en `--n-eval 1320` (dezelfde
verhouding als de 4.000 van de hoofdrun). De pool heeft bij elke bron minder: elke bron is (vrijwel) leeg gehaald.

| bron | train | eval | kandidaten train/eval | gevraagd train | idioomrijk (train) |
|---|---:|---:|---:|---:|---:|
| Handelingen | 9.885 | 216 | 9.886 / 216 | 26.400 | 5.878 |
| raad | 8.395 | 165 | 9.377 / 180 | 13.200 | 1.480 |
| dpc | 2.346 | 44 | 2.351 / 45 | 4.950 | 447 |
| rechtspraak | 1.578 | 40 | 1.582 / 40 | 4.950 | 729 |
| FineWeb-2 | 8.252 | 156 | 8.716 / 184 | 16.500 | 2.558 |
| **samen** | **30.456** | **621** | | 66.000 | |

31.077 passages, 13,73 M tokens (gemiddeld 442, 385–511). Met de hoofdrun is 13,3% van de train-passages lang
(30.456 van 229.472), niet 25%: de pool had per bron 25% lang tussen 384 en 768, en het deel boven 511 blijft buiten
(`too_long`: Handelingen 16.490, raad 7.292, dpc 2.344, rechtspraak 3.757, FineWeb-2 7.011).

Regels, zoals de hoofdrun: bronscore ≥ 0,5, herkomst ≥ 0,5 (FineWeb-2), de regels van stap 0, de herstelde koppen
(raad: 141 passages uit `repaired`, 239 met een herstelde kop) en de lege regel tussen alinea's, de 13-grammen van de
testsets, het documentmaximum (Handelingen 15, raad 5, anders 3). Nieuw (`mainset.py final`):
- `--exclude`: de 202.993 passages van de hoofdrun (train en eval) zijn geen kandidaat en gaan eerst door de chooser,
  zodat duplicaten, 13-grammen en het documentmaximum voor beide runs samen gelden (overgeslagen: 874 documentmaximum,
  139 13-gram, 21 exact). Een document houdt zijn kant: de pool-splitsing is per document, en de code stopt als een
  poolpassage een andere kant heeft dan haar document in de hoofdrun.
- FineWeb-2: onderwerp- en hostplafond zijn een aandeel van wat de pool kan geven (min(gevraagd, kandidaten)), niet
  van het gevraagde aantal: host ≤ 43, onderwerp ≤ 1.307 (ronde 5 liet het onderwerp los voor 1.077 passages). Samen
  met de hoofdrun: nrc.nl 293 van 58.252 (0,50%), ad.nl 0,46%, nu.nl 0,45%.
- Geen contextvoorbeelden, seed `en-nl-translator/mainset-long/0`.

Gecontroleerd op w3 tegen `$M/run/select/`: 0 gedeelde pids, 0 gelijke teksten (ook niet met de 37.640
contextbeurten), 0 documenten met een andere kant; 18.183 passages komen uit documenten die de hoofdrun ook gebruikte
(binnen het documentmaximum).

### Opruiming

- **De opruiming na de hoofdrun (hieronder) moet wachten tot deze run `EXIT 0` heeft**: `run_translate.sh` leest per
  klasse `/dev/shm/translator/segments/`, en `qe` gebruikt `/dev/shm/translator/metricx` (anders haalt hij de 25 GB
  opnieuw van HF).
- Daarna, op alle workers: `rm -rf /dev/shm/translator/longrun/run-long /dev/shm/translator/longrun/run-long-dry`;
  op w3 ook `rm -rf /dev/shm/translator/longrun/sel` (de invoer van de selectie, ~0,4 GB); `/dev/shm/translator/longrun/code`
  mag blijven tot de fine-tune.
- In de bucket weggehaald (9 nov 22:31 UTC, met zichtbaar commando): de eerste selectie in `$L/select` en
  `$M/run-long-dry/select`, gemaakt met het hostplafond op het gevraagde aantal (FineWeb-2 8.417, host tot 82); nog door
  niets gebruikt. Opnieuw gemaakt met commit `e378f1e`.

### Verkeer

Tussen regio's niets; alles in `gs://t5-dutch-english-usc2` en op mbert-corpus. Naar w3: de pool, de bronscores,
`repaired`, `fineweb-hosts.parquet`, de testsets en `passages.parquet` en `context-turns.parquet` van de hoofdrun
(~0,4 GB, binnen us-central2). Naar co1, klein: `selection.md` en `report.json` van de selectie (6 en 16 kB) en de
proefrun van de hoofdrun (`$M/run-dry/`, ~0,2 MB) om `pairs.py` te toetsen. Van co1 naar de bucket: de code
(`$R/scripts/longrun/`) en de keten.

### Open punten (aanvullende run)

1. **Aandeel lang** 13,3% in plaats van ~25%: de pool heeft niet meer passages van 385–511 tokens. Meer vraagt een
   nieuwe pool (meer row groups) of de passages van 512–768 (`xxlong` eerst repareren).
2. **FineWeb-2-onderwerp K** (cultuur, sport, media) is 28,9% van de lange FineWeb-2-passages (de pool); samen met de
   hoofdrun ~23,7%.
3. **en→nl op `xlong`** is niet eerder getest (de minipilot deed alleen nl→en); de proefrun is de eerste toets.
4. **Engels boven 512 tokens** krijgt geen terugvertaling (zie boven); het aantal staat na `back` in het log.

## Hoofdrun: selectie, minipilot en de keten van stap 1 en 2 (9 nov, avond; Yebs akkoord ~17:00 CET)

`R=gs://t5-dutch-english-usc2/mbert/translator`, `M=$R/mainset`, `RUN=$M/run`. Besluiten: ontwerp §3 stap 0–2, §7 punt 6
en 7 (bronscore ≥ 0,5, lege regel tussen alinea's, `heading_glued` met `_…_` herstellen), §9 (context), plus de
aanvulling van de coördinator na overleg met Yeb: FineWeb-2 hoogstens 0,5% per host.

### Wat loopt

Stand 9 nov 19:25 UTC: de keten `translator_mainrun.sh` draait losgekoppeld op co1 (gestart 19:08 UTC, tag
`11091908`), log `~/Developer/yhavinga/mbert-ops/logs/translator_mainrun.log`. Afgerond (`$RUN/stages/*.done`):
`wait`, `prep`, `pilot`, `decide`, `select` (door dezelfde keten met `STOP_AFTER=select`, log
`translator_mainrun_pre.log`, tag `11091630`). Nu stap 1 (`nl_en`) op alle 4 workers van mbert-corpus: log per worker
`~/logs/tmr_nl_en_11091908.log` (kopie na afloop in `$RUN/logs/`), shards in `$RUN/nl-en/w<k>/<klasse>/`. De eerste
shards kwamen goed binnen: `tiny` (contextbeurten, 1.798–1.799 segmenten per worker, 0 niet afgelopen, Engels in orde)
en `small` shard-00006 (1.662 segmenten in ~134 s, ~1.280 uitvoertokens/s, 0 niet afgelopen).

Daarna doet de keten vanzelf: `back` (w0) → `en_nl` (alle workers, `~/logs/tmr_en_nl_11091908.log`, `$RUN/en-nl/`) →
`measures` (w0) → `qe` (alle workers, `~/logs/tmr_qe_11091908.log`, `$RUN/qe/`) → `summary` (w0, `$RUN/pairs/`).
Een proefrun van al deze stappen (40 + 4 passages tot 128 tokens, `$M/run-dry/`, log `translator_mainrun_dry.log`)
liep vóór de start foutloos door (33 min, `DONE`, `EXIT 0`).

mbert-mix niet aangeraakt; de andere agent (worktree `translate-generic`, us-east1-d) niet aangeraakt.

### "Klaar" betekent

- `~/Developer/yhavinga/mbert-ops/logs/translator_mainrun.log` op co1 eindigt op `DONE translator main run: ...` en
  `EXIT 0`; `$RUN/stages/` heeft `wait prep pilot decide select nl_en back en_nl measures qe summary` (`.done`).
- `$RUN/pairs/pairs.parquet` (elk paar: `nl` origineel na de bewerkingen, `en` synthetisch, `back` terugvertaling,
  `hard_fails`, `structure`, `chrf`, `qe_en_nl`, `kind` passage/context, `split` train/eval), `$RUN/pairs/summary.md`
  en `summary.json`; kopieën op co1: `~/Developer/yhavinga/mbert-ops/translator-mainrun/summary.md` en `selection.md`.
- mbert-corpus vrij: `sudo lsof -t /dev/accel* /dev/vfio/*` leeg op alle workers, geen `translate.py`, `judge.py`,
  `metricx_qe.py`.

### ETA's

Geschat op de eerste shards, de stapgroottes (per worker 60.158 segmenten, 9,38 M brontokens: tiny 1.798, small
26.238, medium 22.556, long 9.566) en de decodeerstappen van FLAN-nl; laden plus compileren ~3,5 min per klasse:
- stap 1 (`nl_en`) klaar ~21:15–22:15 UTC;
- `back` ~5 min; stap 2 (`en_nl`) klaar ~00:00–01:15 UTC (10 nov);
- `measures` ~10 min, `qe` ~30 min (de XXL-gewichten staan al in `/dev/shm/translator/metricx` van de proefrun),
  `summary` ~5 min: `DONE` ~01:00–02:15 UTC (02:00–03:15 CET).
Let op: in de oude planning moest mbert-corpus om 00:30 CET vrij zijn voor G's extra evals; deze keten houdt hem tot
~02:15 UTC bezet (Yebs akkoord van ~17:00 CET voor vannacht).

### Bij een fout

De keten stopt bij de eerste fout met `FAILED: <wat>` en `EXIT 1`; de processen op de andere workers lopen door
(losgekoppeld). Dan:
1. Lees de regel `FAILED` en het log van die stap op de worker (`~/logs/tmr_<stap>_<tag>.log`; de vertaal- en
   judge-logs ook in `$RUN/logs/` en `$M/logs/`).
2. Wacht tot geen worker nog rekent (`sudo lsof -t /dev/accel* /dev/vfio/*` leeg); stop een hangend proces alleen met
   `pkill -9 -f '[t]ranslate.py'` (of `[m]etricx_qe.py`, `[j]udge.py`) op die worker.
3. Herstart de keten met hetzelfde commando (hieronder). Afgeronde stappen slaat hij over (`$RUN/stages/*.done`);
   `translate.py`, `judge.py` en de QE-stap slaan hun afgeronde shards over. Een CPU-stap (`select`, `back`) weigert
   als zijn lokale map al bestaat (`/dev/shm/translator/mainrun/run/select` of `.../back`): die map eerst met een
   zichtbaar `rm -rf` van precies dat pad weghalen.
4. Gaat een stap op de TPU herhaald mis op `xxlong` (512–768 tokens, 56 slots, alleen in de minipilot getest): de
   selectie met `--max-tokens 512` opnieuw maken is een besluit voor Yeb, niet voor de keten.

### Commando's

- Keten (co1): `setsid nohup bash ~/Developer/yhavinga/mbert-ops/scripts/translator_mainrun.sh >
  ~/Developer/yhavinga/mbert-ops/logs/translator_mainrun.log 2>&1 < /dev/null &` (flock op
  `~/Developer/yhavinga/mbert-ops/translator_mainrun.lock`; een tweede start stopt meteen). Kopie in
  `gs://t5-dutch-english-usc2/hammond/scratch-scripts/translator_mainrun.sh`. Omgeving: `RUN`, `N`, `N_EVAL`,
  `MAX_TOKENS`, `STOP_AFTER`, `POLL` (s, standaard 300).
- Elke ssh gaat via `~/Developer/yhavinga/mbert-ops/translator-mainrun/vm.sh` (`timeout 900`, de ssh-vlaggen,
  `< /dev/null`); de stukjes die de keten op de workers draait, schrijft hij naar
  `~/Developer/yhavinga/mbert-ops/translator-mainrun/remote/chain/`.
- Code op de workers: `/dev/shm/translator/mainrun/code/` (kopie van `$R/scripts/mainrun/`, deze branch).
  TranslateGemma draait in FLAN-nl's venv `/dev/shm/flan_nl/venv`, de rest in `/dev/shm/translator/venv`, de
  bronscore in `/dev/shm/judge/venv`.

### Opruiming (met zichtbare commando's, letterlijke paden)

Al gedaan op 9 nov 19:23 UTC op alle workers, toen niets ze meer nodig had (`pilot` en `finish_mainset.sh` klaar):
`rm -rf /dev/shm/judge` (op w3 met de lokale kopie van Gemma 4 31B; het checkpoint blijft in
`$R/judge/gemma-4-31b-it-maxtext/`), `rm -rf /dev/shm/translator/mainset`, en de eigen werkmappen
`/dev/shm/translator/mainrun/{test,run-dry,repaired,pool,pool-fineweb,hosts}`.

Na `EXIT 0` van de keten én van de aanvullende run (`translator_longrun.sh`, zie boven; die gebruikt
`/dev/shm/translator/segments` en `/dev/shm/translator/metricx` ook), op alle workers (alles staat in de bucket):
- `rm -rf /dev/shm/translator/mainrun/run` (lokale kopieën van selectie, shards, paren, QE).
- `rm -rf /dev/shm/translator/metricx` (MetricX-24-XXL, ~25 GB per worker; `metricx_qe.py` haalt hem opnieuw van HF).
- `rm -rf /dev/shm/translator/segments /dev/shm/translator/tpu_logs` (segmentkopieën van `run_translate.sh`, libtpu-logs).
- `/dev/shm/translator/mainrun/code` en `/dev/shm/translator/jax_cache` mogen blijven tot de fine-tune.
- Niet: `/dev/shm/flan_nl` (TranslateGemma, FLAN-nl), `/dev/shm/translator/venv`, `/dev/shm/translator/resources`.
- In de bucket staat de proefrun `gs://t5-dutch-english-usc2/mbert/translator/mainset/run-dry/` (klein); weghalen
  alleen met Yebs akkoord: `gsutil -m rm -r gs://t5-dutch-english-usc2/mbert/translator/mainset/run-dry`.

### Verkeer

- Tussen regio's: niets. Alles in `gs://t5-dutch-english-usc2` en op mbert-corpus (us-central2).
- Van HF naar de VM's: voor het herstel van `heading_glued` de row groups van de pool opnieuw (w1: Handelingen 40,
  raad 100, dpc 32, rechtspraak 24, FineWeb-2 240, alleen de nodige kolommen); de kolommen `id` en `url` van de 240
  FineWeb-2-row-groups (w2); MetricX-24-XXL en de mT5-tokenizer (~25 GB per worker, alle 4, in de proefrun).
- Naar co1, eenmalig en klein: `rejected.parquet` van beide pools (2,0 en 0,5 MB, om de vorm van `heading_glued` te
  zien), beide `report.json` (41 en 33 kB), drie scriptkopieën van w0 (~0,1 MB), `selection.md` (6 kB), de
  `summary.md` van de proefrun (2 kB), en via `gsutil cat` `report.json` van de selectie en `minipilot-check.json`.
- Van co1 naar de bucket: de code (`$R/scripts/mainrun/`) en de ketenscripts
  (`gs://t5-dutch-english-usc2/hammond/scratch-scripts/`).

### Uitkomsten tot de start (details: `$RUN/select/selection.md`, `report.json`; kopie op co1 in `translator-mainrun/`)

1. **`heading_glued`** (`select_passages.py repair`, w1, CPU, 634 s; `$RUN/repaired/`). Uit dezelfde row groups en met
   dezelfde seed opnieuw opgebouwd (zelfde pids), alleen documenten met een gemarkeerde kop (`_…_` aan het begin van
   een regel, gevolgd door een hoofdletter): raad 5.063 documenten, dpc, rechtspraak en Handelingen 0, FineWeb-2 1.
   Van de raadspassages met een herstelde kop slagen er 8.147 daarna voor alle regels van stap 0 (8.018 train, 129
   eval): 6.323 vielen eerder op `heading_glued`, de rest alleen op `mid_capital`, omdat de kop een leesteken bevat
   ("_1._ _Grondexploitaties_ Een …"). Een kop zonder markering blijft afgekeurd (unit-tests). Gemma 4 31B scoorde ze
   op w2–w3 (`$RUN/source-scores-repaired/`). Ze kregen de idioomklasse van de pool (drempels uit `report.json` van de
   pool). In de selectie: 2.238 train en 34 eval uit het herstel (het documentmaximum van de raad, 5, sloeg in de hele selectie 1.766
   passages over),
   plus 610 + 11 poolpassages met een gemarkeerde kop vóór een gewoon woord met hoofdletter, die de regel niet ving
   en die de bewerking nu ook op een eigen regel zet.
2. **FineWeb-2-herkomst** (train, bronscore ≥ 0,5: 82.099 passages): bij drempel 0,1 vallen er 14.367 af (17,5%),
   bij 0,5 15.060 (18,3%), bij 0,9 16.127 (19,6%); de score is bijna binair. Drempel 0,5.
3. **Minipilot** (501 lange passages, na beide bewerkingen; w0–w1): 166 met `decode_loop` (33%), waarvan 27 niet
   afgelopen, 0 met Nederlandse resten. Besluit volgens de regel (> 2%): alleen passages tot 384 tokens. Maar:
   384–511 brontokens 0 van 248, 512–767 166 van 253 (512–575: 54/78, 576–639: 22/73, 640–703: 53/59, 704–767: 37/43).
   De fout begint precies bij de grens van de nieuwe klasse `xxlong` (prefill 768/896, 56 slots, cache 2.048); in
   `xlong` gingen vertalingen met prompt plus uitvoer tot 1.216 tokens goed, dus het schuifvenster van 1.024 van
   Gemma 3 is niet de grens. Het lijkt een fout in de motorinstelling van `xxlong`, niet het gedrag van het model op
   lange tekst ("youyouyou…", "The. The. The."). Niet uitgezocht. Gevolg: 384–512 had waarschijnlijk mee gekund;
   dat is een besluit voor Yeb (een aanvullende run voor 384–512 kost ~1–2 uur).
4. **Selectie** (`mainset.py final`, `--max-tokens 384`, seed `en-nl-translator/mainset-final/0`): 199.016 train
   (Handelingen 79.095, raad 40.000, dpc 14.921, rechtspraak 15.000, FineWeb-2 50.000) en 3.977 eval
   (1.582/800/295/300/1.000). Handelingen en dpc zijn te kort (905 en 79), omdat hun pool zonder lange passages op
   is; niet aangevuld uit andere bronnen. Idioomrijk: Handelingen 41.023 (52%), raad 5.325 (13%), dpc 2.342 (16%),
   rechtspraak 1.683 (11%), FineWeb-2 21.064 (42%); de pool had niet meer. Context: 16.082 Handelingen-passages
   (20,3%) met 37.640 contextbeurten (een beurt met bronscore < 0,5 of > 384 tokens beëindigt de context; 1.269
   voorbeelden hielden geen beurt over). FineWeb-2: onderwerp K (cultuur, sport, media) 22,9% en O (dagelijks leven)
   15,4%, boven de 15%: zonder lange passages is de pool te klein (51.312 kandidaten voor 50.000; ronde 5 liet de
   onderwerpgrens los voor 4.153 passages). Grootste hosts: nu.nl, nrc.nl, ad.nl elk 250 (0,5%), nos.nl 230,
   volkskrant.nl 208, hln.be 186, gelderlander.nl 172, trouw.nl 154, telegraaf.nl 151, bndestem.nl 149. Ontdubbeld
   over alles (ook FineWeb-2 tegen de andere bronnen) en tegen de 13-grammen van de testsets: 368 13-gram-overlap,
   220 exacte duplicaten. Stap 1: 240.633 segmenten, 33,1 M tokens in passages en 4,4 M in contextbeurten.

### Open punten (hoofdrun)

1. **`xxlong`** (512–768 tokens) gaf in de minipilot 166 van 253 ontspoorde vertalingen; 384–512 (`xlong`) 0 van 248.
   Oorzaak niet gezocht (de motorinstelling van de nieuwe klasse, niet het schuifvenster). Of de 384–512-passages
   alsnog mee moeten (aanvullende run), is aan Yeb; de regel van de opdracht hield alles boven 384 buiten de run.
2. **Aantallen**: 199.016 train in plaats van 200.000 (Handelingen −905, dpc −79), niet uit andere bronnen aangevuld,
   zodat de verdeling klopt.
3. **Idioomrijk** bij raad, dpc en rechtspraak maar 11–16% (de pool had niet meer); FineWeb-2 42%.
4. **FineWeb-2-onderwerpen** K 22,9% en O 15,4% (> 15%), omdat de pool zonder lange passages te klein was.
5. **Herstelde koppen**: 1.824 van de 8.147 herstelde raadspassages vielen eerder alleen op `mid_capital` (een
   gemarkeerde kop met een leesteken erin, "_1._ _Kop_ Een …"); ze zijn meegenomen als dezelfde fout. De 610 + 11
   poolpassages die de bewerking nu ook herstelt, houden de bronscore van hun onherstelde tekst. Gemma scoorde alle
   passages met één regelovergang tussen alinea's, niet met de lege regel van de bewerking.
6. **Paarfilters**: `newline_doubled` is een harde afval (zoals bij de replay), apart geteld; door de lege regels in de
   bron kwam het in de proefrun niet voor. MetricX-QE alleen en→nl (de richting van het trainingspaar);
   invoer boven 1.536 mT5-tokens wordt afgekapt en per worker geteld in `tmr_qe_*.log` (`truncated`).
7. `selection.md` van deze run noemt nog "~25% lang (384–384 …)"; de tekst is na de selectie in de code verbeterd,
   de cijfers kloppen.
8. In de oude planning moest mbert-corpus om 00:30 CET vrij zijn voor G's extra evals; zie de ETA's.


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
