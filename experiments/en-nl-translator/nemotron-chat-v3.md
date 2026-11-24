# Spec: Nemotron Chat v3 naar het Nederlands met G4-8-s140

Vastgelegd 24 nov 2026, vóór de eerste stap van de proef. Besluit Yeb (24 nov ~14:50): eerst deze spec en een kleine
proef op mbert-mix; de volle run pas op zijn woord. Code: mbert-jax, branch `en-nl-nemotron` (vanaf `en-nl-g4arm8`),
`experiments/en-nl-translator/nemotron_chat.py`, `run_nemotron.sh`; keten `mbert-ops/scripts/nemotron_trial_run.sh`.
Kopie van deze spec in de repo: `experiments/en-nl-translator/nemotron-chat-v3.md`. Proefrapport:
`mbert-ops/translator-sft/nemotron-chat-v3-trial.md`.

Markering: **[gemeten]** = nagekeken of geteld, met bron; **[keuze]** = beslissing van deze spec, te herzien door Yeb;
**[open]** = Yeb beslist.

## 1. De bron en welke delen meegaan

`nvidia/Nemotron-SFT-Instruction-Following-Chat-v3`, revisie `be3b3e04ef605ac9d3f8f35b9d5a632f4a3a3402` (HF, laatst
gewijzigd 2026-06-04). Volgens de kaart (sectie "Dataset Quantification"): chat 637K gesprekken (16G), instruction
following 249K (3G). `data/chat.jsonl` is 17.042.669.218 bytes (HTTP `x-linked-size`) **[gemeten]**.

Velden per rij **[gemeten op 505 rijen, drie bereiken van 4 MB, 24 nov]**: `messages` (`role`, `content`, bij de
assistent ook `reasoning_content`), `metadata` (`seed_dataset`, `seed_prompt_sha256`, `model`, `reward_model`,
`train_turns`), `used_in`, `uuid`. `metadata.model` is in alle 505 `zai-org/GLM-5`, `reward_model`
`nvidia/Qwen3-Nemotron-235B-A22B-GenRM-2603`. `seed_dataset`: `lmarena-ai/arena-human-preference-100k` 182,
`lmarena-ai/arena-human-preference-55k` 165, `allenai/WildChat-1M` 158. (De notitie van 9 nov noemde "-140k"; de
metadata zegt `-100k`.) Elk gesprek begint met een systeembericht; bij WildChat zijn systeembericht en eerste vraag
`null` (158 van 158).

Wat de kaart zegt over de herkomst (sectie "Dataset Description", citaat): "The chat subset uses human written prompts
from sources like lmarena, lmsys, and wildchat as seed prompts. Responses are generated with GLM-5. [...] Each
conversation is further extended to multiple turns by user simulation with GLM-5. [...] Only the last assistant turn in
each sample should hence be used for training." En: "rows from these sources [lmsys-chat-1m, WildChat-1M] have the
initial system message content and first user message content set to `null`".

| deel | gaat mee | waarom |
|---|---|---|
| chat, seed lmarena (100k + 55k) | **ja, fase 1: 100.000 gesprekken** (besluit Yeb 9 nov) | eerste vraag menselijk en aanwezig; rest GLM-5 |
| chat, seed WildChat-1M | **nee in fase 1** [keuze] | eerste vraag en systeembericht `null`; terug te zetten met `prepare_chat_prompts.py` uit WildChat-1M (ODC-BY). Fase 2 als Yeb dat wil [open] |
| chat, seed lmsys-chat-1m | **nooit** | de licentie verbiedt doorgeven (zie §2) |
| `instruction_following` | **nee in fase 1** [keuze, open voor Yeb] | antwoorden GPT-OSS-120B (mag), maar een deel van de vragen komt uit `allenai/tulu-3-sft-personas-instruction-following`, die met GPT-4o gemaakt zijn (§2); en de controleerbare eisen ("in minder dan 100 woorden", "alles in hoofdletters", letterlijke Engelse woorden) overleven een vertaling meestal niet |
| `reasoning_content` | **niet vertaald, niet in de uitvoer** [keuze] | de kaart staat toe het weg te laten ("for non-reasoning/thinking training by removing the reasoning_content fields"); het is ~11.000 tekens per gesprek (notitie 9 nov) |
| codetaken | **nee** (besluit Yeb 9 nov) | filter §3 |

## 2. Licenties

Nagekeken op HF op 24 nov via `https://huggingface.co/api/...` (`cardData.license`) en de README's; het
WebFetch-resultaat over Tulu 3 en de Gemma 4-licentiepagina staat erbij. "Trainen op uitvoer" volgens de regel in
memory `generator-rule-license`: open modellen waarvan de licentie dat toestaat mogen; Claude en GPT via de API niet.

| bron | revisie | licentie | wat er staat | gevolg |
|---|---|---|---|---|
| `nvidia/Nemotron-SFT-Instruction-Following-Chat-v3` | `be3b3e0` | CC BY 4.0 en ODC-By (README "License/Terms of Use"; front matter noemt ook apache-2.0 en other) | "The data may be freely used to train and evaluate." (README "Intended Usage") | mag; naamsvermelding bij publicatie |
| `zai-org/GLM-5` (generator chat) | `c183ef8` | MIT (front matter `license: mit`; de repo heeft geen LICENSE-bestand, alleen de README) | MIT stelt niets over uitvoer | mag |
| `nvidia/Qwen3-Nemotron-235B-A22B-GenRM-2603` | – | niet nagekeken | alleen beoordelaar, geen tekst in de data | geen gevolg |
| `lmarena-ai/arena-human-preference-100k` | `72e85b3` | geen `license` in de metadata; README "License": "User prompts are licensed under CC-BY-4.0, and model outputs are governed by the terms of use set by the respective model providers." | Nemotron neemt alleen de vragen over | mag; naamsvermelding |
| `lmarena-ai/arena-human-preference-55k` | `18c2983` | Apache-2.0 (front matter) | – | mag |
| `allenai/WildChat-1M` | `7d6490e` | ODC-BY (README "Licensing Information"; sinds 2024-06-26, met terugwerkende kracht) | – | mag (fase 2) |
| `lmsys/lmsys-chat-1m` | `200748d` | LMSYS-Chat-1M Dataset License Agreement (gated) | "Prohibited Transfers: You should not distribute, copy, disclose, assign, sublicense, embed, host, or otherwise transfer the dataset to any third party." | **niet**: de rijen staan in het bestand met `null`, ze gaan nergens mee |
| `openai/gpt-oss-120b` (generator IF) | `b5c939d` | Apache-2.0 | `USAGE_POLICY` (hele tekst): "We aim for our tools to be used safely, responsibly, and democratically, while maximizing your control over how you use them. By using OpenAI gpt-oss-120b, you agree to comply with all applicable law." | antwoorden mogen |
| `allenai/tulu-3-sft-personas-instruction-following` (vragen in IF) | `fe0c7d3` | ODC-BY | kaart noemt geen generator; Tulu 3-rapport (Lambert et al. 2024, arXiv 2411.15124, §3.1.2 "Precise Instruction Following", via WebFetch): "We then generate new instructions using GPT-4o"; voetnoot: "We use GPT-4o-2024-08-06 for all our persona-driven data synthesis, unless otherwise stated." | vragen zijn GPT-uitvoer via de API: **niet** volgens onze regel. Of de IF-rijen hun herkomst in de metadata dragen is niet nagekeken |
| `google/gemma-4-31B-it` (vertaler, basis) | `842da37` | Apache-2.0 (front matter; `license_link` https://ai.google.dev/gemma/docs/gemma_4_license verwijst door naar https://gemma.ai/resources/apache-2/, de tekst van de Apache License 2.0) | de pagina zegt niets over uitvoer; de navigatie noemt een aparte pagina "Prohibited use", niet nagekeken | vertaling mag |
| G4-8-s140 (LoRA, `gs://t5-dutch-english-usc2/mbert/translator/sft/g4arm/G4-8-s140/`) | – | eigen werk, getraind op data binnen de regel (Gemma 4-doelen, Gemma 4-judge) | – | mag |
| `google/metricx-24-hybrid-xxl-v2p6-bfloat16` | `3d9c6f7` | Apache-2.0 | alleen scoren | geen gevolg |

Conclusie: de Nederlandse lmarena-gesprekken (vragen van mensen, rest GLM-5, vertaling Gemma 4 + eigen LoRA) mogen
volgens de regel als trainingsdata. Publiceren alleen op Yebs woord, met naamsvermelding (CC BY 4.0, ODC-By, lmarena
CC-BY-4.0) en de TRC-zin.

## 3. Filters

Per gesprek, in deze volgorde; de reden van uitval wordt geteld (`nemotron_chat.py`, functies `conversation`,
`code_reason`, `language_reason`, `prepare`):

1. **Herkomst**: `metadata.seed_dataset` is een van de twee lmarena-sets; anders `seed_not_lmarena`.
2. **Rollen**: het eerste bericht is een systeembericht met lege inhoud (een niet-leeg: `system_prompt`, eruit), daarna
   om en om `user` en `assistant`, beginnend met `user`, eindigend met `assistant`, elke inhoud een niet-lege string;
   anders `roles`. Ook de door GLM-5 gesimuleerde gebruikersbeurten worden vertaald.
3. **Code** (een codetaak, eruit): een omheind blok (```/~~~) met een programmeertaal als label (`CODE_LANGS`, ook
   `json`, `yaml`, `sql`, `bash`, `latex`, `html`), of meer dan 10% van de tekens van het gesprek in omheinde blokken,
   of meer dan 2 coderegels buiten blokken (`def …(`, `import …`, `#include`, `function …(`, `SELECT … FROM`, `}`
   alleen, …). Omheinde blokken zonder programmeertaal (tekst, ascii-tabellen) blijven, mits ≤ 10%.
4. **Taal**: langid 1.1.6 (alle talen) op de eerste vraag (vanaf 20 letters proza, zonder code) en op elke alinea van
   100 letters of meer: alles moet `en` zijn; anders `lang_first` / `lang_paragraph`. Daarmee vallen ook
   vertaalopdrachten met een anderstalige tekst af.
5. **Lengte** (Gemma 4-tokens van de Engelse tekst, zonder `reasoning_content`): een beurt > 4.096 (`turn_long`) of
   het gesprek > 12.288 (`conv_long`) eruit.
6. **Opknippen** (§4): een omheind blok of een zin van meer dan 512 tokens kan niet in een stuk: eruit
   (`unsplittable_fence`, `unsplittable_sentence`).

De proef meet de uitval per reden op 32 bytebereiken van 8 MB, gelijk verdeeld over het bestand. Een NSFW- of
veiligheidsfilter zit er niet in: NVIDIA's set is de bron zoals hij is [open].

**Markdown en codeblokken in antwoorden**: de hele beurt gaat naar de vertaler, met opmaak en eventuele omheinde
blokken; de prompt vraagt de opmaak te houden ("Keep the paragraphs, line breaks and formatting"). Daarna wordt per
beurt geteld of de omheinde blokken byte voor byte gelijk zijn en of het aantal koppen, lijstitems, tabelrijen en
inline-codestukken klopt (`structure.py`, het bestaande schema van de vertaler). Beslisregel voor de volle run
[keuze]: zijn in de proef ≥ 99% van de beurten met een omheind blok byte voor byte gelijk, dan blijft het zo en geldt
een afwijkend blok als harde fout (gesprek eruit); anders worden de blokken vóór vertalen vervangen door een
plaatshouder en daarna teruggezet (een nieuwe proef). Rollen: `user` → Gebruiker, `assistant` → Assistent alleen in
het blinde pakket; in de uitvoer blijven de rollen `user` / `assistant`. Systeemberichten (bij lmarena leeg) gaan
niet mee.

## 4. Stukken, prompt en context

- **Stukken** [keuze]: een beurt wordt geknipt in stukken van hoogstens 512 Engelse Gemma 4-tokens (de lengte waarop
  G4-8-s140 getraind en geëvalueerd is: klassen tiny–xlong ≤ 512). Knippen gebeurt eerst tussen blokken (lege regels,
  buiten omheinde blokken; een omheind blok met lege regels erin blijft heel), een te lang blok tussen zinnen of op
  regeleinden. Opeenvolgende blokken worden samengenomen zolang het stuk ≤ 512 tokens blijft. De scheidingstekens
  tussen de stukken komen uit de bron: de Nederlandse beurt is `lead + nl₀ + sep₀ + nl₁ + … + trail`, teken voor teken
  de vorm van de bron (getest).
- **Model**: Gemma 4 31B-it in MaxText (conversie `gs://…usc2/mbert/translator/judge/gemma-4-31b-it-maxtext`), fork
  `f4d8615` (prompt plus decode boven 1.024 tokens), LoRA van G4-8-s140 samengevoegd (rang 32, alpha 16, zoals
  `run_g4measure2.sh generate`), greedy, één proces per worker van 4 chips.
- **Prompt zonder context**: `natural-v1`, letterlijk, de prompt waarop G4-8-s140 is gefinetuned (protocol-g4.md §2.6).
- **Prompt met context**: `natural-v1-chat` (`translate_models.yml`): de woorden van `natural-v1`, dan: "The text is a
  turn, or part of a turn, of a conversation between a user and an AI assistant. For context only, here is the Dutch
  translation already made of the conversation so far (or of its last turns). Keep names, terms and the form of
  address consistent with it, but do not translate it again and do not repeat it. Do not answer the text and do not
  continue the conversation: only translate the text.", het blok `<conversation_so_far>` met per eerdere beurt
  `[User]` of `[Assistant]` en de Nederlandse tekst, en als laatste "The English text to translate (User):" (of
  `Assistant`, `Assistant, continued` voor een vervolgstuk) met de Engelse tekst. Het eerste stuk van een gesprek
  heeft geen context en krijgt dus `natural-v1`.
  - Alleen het Nederlands van de eerdere beurten, niet het Engels [keuze]: de helft van de tokens, en wat de vertaler
    voor consistentie nodig heeft (je/u, termen, namen) staat in het Nederlands; zo deed ook `natural-v1-ctx` in de
    lengteproef (13 nov). Ontwerp §9 (9 nov) had bron en vertaling, maar dat was voor TranslateGemma's fine-tune met
    `chat-v1`. De huidige tekst staat niet tussen tags: `chat-v1`'s `<current_turn>` lekte in de pilot van 9 nov in 16
    beurten in de uitvoer.
  - G4-8-s140 heeft nooit voorbeelden met context gezien (protocol-g4.md §2.6: "natural-v1 heeft geen contextvorm;
    daarom geen contextvoorbeelden"). De proef meet of context dan helpt of schaadt.
- **Contextbudget**: de nieuwste eerdere beurten eerst, heel, samen hoogstens 1.408 tokens (`CTX_TOKENS`); de eerdere
  stukken van dezelfde beurt tellen als één item "this turn so far". Past alleen al het nieuwste item niet, dan zijn
  laatste blokken na de markering `(…)`; past ook het laatste blok niet, dan geen context. De prompt blijft zo ≤ 2.176
  tokens (de prefill die met de fork goed decodeerde, 14 nov). Maximale uitvoer per stuk: 832 tokens (1,6 × 512).
- **Klassen** (`gemma4-31b-chat`): zonder context `n128` (stukken ≤ 128 tokens, prefill 256, decode 256, 64 slots) en
  `n512` (129–512, prefill tot 640, decode 832, 28 slots); met context `c2k` (alle stukken van een gesprek, prefill tot
  2.176, decode 832, 12 slots = 36k slot-tokens).
- **Volgorde met context**: een golf neemt van elk gesprek het volgende stuk; de engine blijft geladen over alle golven.
  Zonder context zijn alle stukken onafhankelijk (één golf per klasse).

## 5. Uitvoerformaat

Eén JSON-regel per gesprek (`chat-nl.jsonl`, in de proef `chat-nl-ctx.jsonl` en `chat-nl-noctx.jsonl` in
`gs://…usc2/mbert/translator/nemotron/trial/collect/`):

```json
{"uuid": "<uuid van Nemotron>", "seed_dataset": "lmarena-ai/arena-human-preference-100k",
 "messages": [{"role": "user", "content": "<Nederlands>"}, {"role": "assistant", "content": "<Nederlands>"}, ...],
 "train_turns": [<metadata.train_turns van Nemotron, ongewijzigd>],
 "metadata": {"source": "nvidia/Nemotron-SFT-Instruction-Following-Chat-v3@be3b3e0…", "generator": "zai-org/GLM-5",
              "translator": "G4-8-s140 (Gemma 4 31B-it + LoRA)", "prompt": "natural-v1-chat", "context": true}}
```

Geen systeembericht, geen `reasoning_content`. Voor de volle run komen daarbij [keuze]: per bericht `qe` (MetricX-24 QE,
gewogen over de stukken) en `pieces`; de code-commit van de vertaling. De Engelse tekst staat niet in de uitvoer; die
is via `uuid` uit de bron te halen. Daarnaast in de bucket per stuk een tabel met bron, vertaling, alle controles en
scores (`pieces-result-meta.parquet` zonder tekst, `turns-nl.parquet` met).

## 6. Kwaliteitscontroles

Per stuk, direct na de vertaling (`hard_flags`):

- **hard** (het stuk telt als mislukt): niet afgekapt binnen 832 tokens (`truncated`), leeg (`empty`), de controles van
  FLAN-nl (`flan_nl.check_translation` op de proza zonder code: lengteverhouding buiten 0,5–2,2 × de tekens van de bron
  bij bronnen ≥ 30 tekens of > 3 × + 30, commentaar van een vertaler, alternatieven, vreemd schrift, niet Nederlands
  volgens langid over nl/en/de/fr/es/it/pt/af met Engelse woorden), de detectoren `decode_loop` en `english_left` van
  FLAN-nl, en een markering van de prompt in de uitvoer die de bron niet heeft (`tag_leak`: `<conversation_so_far`,
  `[User]`, `[Assistant`, `(…)`, "text to translate (").
- **Herkansing**: een mislukt stuk wordt één keer opnieuw vertaald, door te samplen (T = 0,7, vaste seed per golf), met
  dezelfde prompt. Mislukt het weer, dan is het **gesprek kapot**: met context worden zijn latere stukken niet meer
  vertaald, en het gesprek gaat niet in de uitvoer. **Een gesprek gaat nooit door met een gat.**
- **zacht** (geteld, per beurt): omheinde blokken byte voor byte gelijk, inline code gelijk, koppen, lijstitems,
  tabelrijen, vet/cursief, alinea's (`structure.differences`).
- **MetricX-24 QE** (hybrid XXL, QE-modus, per stuk: ≤ 512 + ~700 tokens past in zijn 1.536): verdeling per conditie
  en rol, en gepaard per stuk met en zonder context. Drempel voor de volle run [open, na de proef]: voorstel een gesprek
  eruit als een stuk QE > 10 heeft.
- **Faalpercentage**: per controle, eerste poging en na herkansing, per rol; aandeel hele gesprekken.

## 7. Wat de proef meet (vastgelegd vooraf)

~200 lmarena-gesprekken uit de pool van de 32 bereiken, gelijk verdeeld over 12 lagen: assistentbeurten 1 / 2 / 3 / ≥ 4
× lengte ≤ 1.024 / ≤ 3.072 / > 3.072 tokens (een laag met te weinig geeft alles; volgorde sha1 van
`nemotron-chat-v3/trial/<uuid>`). Per worker (verdeeld op tokens) eerst met context (`c2k`), dan zonder (`n128`,
`n512`). Gemeten: tokens/s per klasse en de tijd per gesprek; QE met en zonder context; het faalpercentage per
controle (leeg, Engels gebleven, afgekapt, lengteverhouding, lek, …); codeblokken en markdown als tellingen; de
contextlengtes. Daarna een blind pakket van 40 beurten (20 gebruiker, 20 assistent, elk uit een ander gesprek, met
minstens één eerdere beurt, beide condities heel en verschillend) met de eerdere beurten als context, A/B willekeurig;
alleen klaargezet, niet beoordeeld. Niemand leest in deze stap teksten: rapport en pakket worden door scripts gemaakt.

Niets in de proef beslist de volle run: dat doet Yeb. Wel volgt uit de proef [keuze]: (a) met of zonder context (QE,
faalpercentage, en later het blinde pakket), (b) de code-regel van §3, (c) de QE-drempel van §6.

## 8. Schatting voor 100.000 gesprekken

Uit de proef (`nemotron_chat.py report`, sectie "Schatting"): de gemiddelde Engelse tokens per gesprek van de pool, per
klasse, × de gemeten verhouding uitvoer/invoer, gedeeld door de gemeten uitvoertokens per seconde per worker (met
context alleen de golven die minstens alle slots vullen, want bij 100K is de staart verwaarloosbaar). Opslag uit de
bytes per heel gesprek van de proefuitvoer. De getallen staan in het proefrapport; de notitie van 9 nov schatte op het
tempo van FLAN-nl ~14 uur voor 100K gesprekken, maar dat was TranslateGemma 27B zonder context.

## 9. Open voor Yeb

1. Met of zonder context (na de proef en eventueel het blinde pakket).
2. WildChat in fase 2 (vragen terugzetten uit WildChat-1M), en `instruction_following` (alleen rijen waarvan de vraag
   niet van GPT-4o komt, en alleen als de controleerbare eisen het vertalen overleven).
3. QE-drempel, code-regel, veiligheidsfilter.
4. De volle run (100K) op mbert-mix, of op beide VM's na T3 (~2 dec).

## 10. Amendement 1: de volle run over het hele chatdeel (24 nov ~15:15, vóór enige uitslag van de proef)

Besluit Yeb (24 nov ~14:55, via de hoofdsessie): Nemotron Chat v3 helemaal vertalen, het hele chatdeel, met de
standaarden van deze spec; de volle run start pas met het startbestand van de hoofdsessie (mbert-mix doet eerst een
andere klus). Dit amendement vervangt §1 "fase 1: 100.000 lmarena" voor de volle run; de proef blijft zoals in §7.

**Standaarden (Yeb kan elk omdraaien)** [keuze]:

| vraag | standaard | waarom | omdraaien |
|---|---|---|---|
| WildChat-gesprekken | **ja**, met de eerste vraag teruggezet uit `allenai/WildChat-1M` @ `7d6490e` (ODC-BY), op sha256 zoals `prepare_chat_prompts.py` van NVIDIA; een rij waarvan de hash niet gevonden wordt valt af (`wildchat_not_restored`) | "het hele chatdeel"; ODC-BY staat het toe; de eerste vraag is van een mens, de rest GLM-5 | `FULL_SEEDS` in `nemotron_chat.py` zonder WildChat |
| lmsys-gesprekken | **nee, nooit** | licentie verbiedt doorgeven (§2) | – |
| `instruction_following` | **nee** | deels GPT-4o-vragen (tulu-3-personas), controleerbare eisen overleven vertaling niet | nieuwe spec nodig |
| codegesprekken | **eruit** (filter §3) | besluit Yeb 9 nov (codetaken niet); codeblokken vertalen vraagt maskeren | filter weg + maskeren, nieuwe proef |
| conditie | **regel hieronder** | proef | `COND=` in het startbestand |
| QE-drempel | **10** (gesprek eruit als een stuk QE > 10 heeft, of als een stuk geen QE heeft) | MetricX-schaal 0–25; > 10 is bij de vertaler zeldzaam (p95 mainset 5,6) | `QE_MAX=` in het startbestand (`none` = alles houden) |
| `reasoning_content` | niet vertaald, niet in de uitvoer | §1 | – |

**Conditie** (vastgelegd vóór de uitslag): de volle run gebruikt **met context** (`ctx`) als in de proef alle drie
gelden, anders **zonder** (`noctx`): (a) mislukte stukken na herkansing bij ctx ≤ die bij noctx + 1,0 procentpunt;
(b) gemiddelde gepaarde QE-verschil ctx − noctx ≤ +0,10; (c) `tag_leak` na herkansing ≤ 0,5% van de ctx-stukken.
Het blinde pakket (40 beurten) beslist hier niet; een latere beoordeling kan de keuze herzien (Yeb).

**Aantallen** [gemeten op 32 bereiken van 8 MB, 9.106 rijen, proef 24 nov]: `seed_dataset` lmarena (100k + 55k) en
WildChat en lmsys in de verhouding van `sample.json` (zie het proefrapport); van de lmarena-rijen haalde 63% de filters.
Extrapolatie naar het hele bestand (637K rijen volgens de kaart) in het proefrapport; de echte aantallen geeft
`fprepare` per shard (`prepare.json`) vóór er een TPU draait.

**Indeling**: een shard = de regels van `data/chat.jsonl` waarvan de eerste byte in een blok van 128 MiB ligt: 127
shards, elk ~5.000 rijen, elke regel precies één keer (getest). Shard r op worker r mod 4 (32/32/32/31). Alles wordt
rechtstreeks van HF gelezen (HTTP-bereiken op revisie `be3b3e0`) en rechtstreeks in de bucket geschreven vanuit Python:
niets blijft achter op een worker, niets via co1.

**Uitvoer** in `gs://t5-dutch-english-usc2/mbert/translator/nemotron/full/`:

| map | inhoud | marker |
|---|---|---|
| `wildchat/` | `prompts-<0-f>.parquet` (sha256, system, user), `wildchat.json` | `wildchat.json` |
| `data/shard-<r>/` | `conversations`, `turns`, `pieces`, `pieces-meta`, `pool-meta` (.parquet), `prepare.json` | `prepare.json` |
| `gen/<cond>/<cls>/shard-<r>/` | `wave-<w>.parquet` (vertalingen, controles), `DONE.json` (tempo) | `DONE.json` |
| `collect/shard-<r>/` | `chat-nl-<cond>.jsonl`, `turns-nl.parquet`, `pieces-result-meta.parquet`, `qe-in-w<k>.parquet`, `collect.json` | `collect.json` |
| `scores/` | `qe-<r>.parquet` (MetricX-24 QE per stuk) | het bestand |
| `out/` | **`chat-nl-<r>.jsonl`** (het eindformaat van §5, met `qe` per bericht), `final-<r>.json` | `final-<r>.json` |
| `stages/` | een marker per stap van de keten | – |
| `summary.json` | de tellingen van alle stappen, zonder tekst | – |

**Keten**: `mbert-ops/scripts/nemotron_full_run.sh` (back-up in `gs://…usc2/hammond/scratch-scripts/`). Wacht op
`mbert-ops/nemotron.full-go` (regels `COND=ctx|noctx`, `COMMIT=<sha van en-nl-nemotron>`, `QE_MAX=<getal>|none`),
neemt pas dan `flock` op `final_eval_mix.lock`, controleert de TPU vóór elke TPU-stap, en loopt
fwildchat → fprepare → fgen → fcollect → fqe → ffinal → fsummary. Elke stap per shard hervatbaar: een shard met zijn
marker wordt overgeslagen, een vertaling gaat verder na de laatste golf in de bucket; de keten zelf slaat stappen met een
marker in `stages/` over. Een herstart = dezelfde opdracht met een nieuwe TAG. mbert-corpus (T3) wordt niet aangeraakt.

### Amendement 1a (24 nov ~15:25): de herkomsten zoals ze in het bestand staan

Een test van `prepare` op echte shards (0, 63, 126; uitvoer alleen onder `…/nemotron/fulltest/`, met een lege
WildChat-tabel) liet zien dat `metadata.seed_dataset` meer waarden heeft dan de 505 rijen van §1. Op de 9.106 rijen van
de proefbereiken [gemeten, `trial/data/sample.json`]:

| `seed_dataset` | rijen | aandeel | volle run |
|---|---:|---:|---|
| `allenai/WildChat-1M` | 3.872 | 42,5% | ja (vraag teruggezet) |
| `lmarena-ai/arena-human-preference-55k` | 1.930 | 21,2% | ja |
| `lmarena-ai/arena-human-preference-100k` | 1.166 | 12,8% | ja |
| `lmarena-ai/arena-human-preference-140k` | 692 | 7,6% | **ja** (CC-BY-4.0, §2) |
| `lmsys/lmsys-chat-1m` | 895 | 9,8% | nooit |
| `nvidia/HelpSteer2` | 403 | 4,4% | nee [keuze: licentie niet nagekeken] |
| `lmarena-ai/repochat-arena-preference-4k` | 148 | 1,6% | nee [keuze: gesprekken over code-repo's, licentie niet nagekeken] |

De proef trekt alleen uit 100k en 55k (`LMARENA`); 140k valt daar onder `seed_not_lmarena`. Het bestand is niet
gemengd: shard 0 is 89% WildChat, shard 63 94% lmarena-140k, shard 126 100% WildChat. Van de lmarena-rijen (100k + 55k)
haalde 63% de filters; over 637K rijen geeft dat voor lmarena (41,6%) ~167K gesprekken, voor WildChat (42,5%, ~271K
rijen) een onbekend deel (meer niet-Engels en code). De echte aantallen geeft `fprepare`.
