# LoRA op Gemma 4 26B-A4B-it (MoE) en Gemma 4 31B-it: geheugen- en stappentest op mbert-mix

Opdracht van Yeb, 13 nov 2026 ~12:50 CET [citaat]: "een test met lora op de MoE: akkoord, dit is interessant". De vraag:
kunnen we Gemma 4 31B-it en 26B-A4B-it met LoRA bijtrainen zoals TranslateGemma 27B, en werkt LoRA op een MoE? Dit is
een meting van 8 stappen per variant. Er is geen arm getraind en geen vertaling gemaakt.

Het vaste deel (§1–§2) is geschreven op 13 nov, vóór de meting. Het deel "Uitkomst" erna maakt `g4lora.py report` uit de
logs van de runs. Labels: [citaat] = letterlijk, [code gelezen] = wat de broncode zegt (bestand en regel), [eigen
berekening], [gevolgtrekking] = mijn interpretatie, [schatting].

## 1. Stack, checkpoints en data

**Stack** (de SFT-venv van de armen op mbert-mix, versies gelezen op w3 op 13 nov): MaxText 0.2.5, google-tunix 0.1.8,
qwix 0.1.8, jax/jaxlib 0.11.0, libtpu 0.0.44.1, flax 0.12.8. Training met `run_sft.sh` en `sft_lora.py` zoals bij de
armen: één job over de 16 chips van de v4-32, `per_device_batch_size=1`, packing, `scan_layers=false`, remat `full`,
gewichten in float32 (`weight_dtype` van MaxText), LR 1e-5, cosine met 10% warmup, AdamW. Code: mbert-jax branch
`en-nl-g4lora`; nieuw is alleen dat `run_sft.sh` de twee Gemma 4-modellen en een eigen LoRA-regex kent, en dat
`sft_lora.py` vóór de eerste stap een regel `LORA_MODULES` schrijft (LoRA-parameters per moduletype, en de modules die de
regex wel raakt maar die geen LoRA kregen).

**Checkpoints** (13 nov 12:03–12:11 UTC, op de CPU van w2 en w3 van mbert-mix, `nice -n 19 ionice -c3`, rechtstreeks van
Hugging Face, met MaxText 0.2.5's eigen `to_maxtext`: ongescand, alleen tekst, bf16):

| model | Hugging Face-revisie | checkpoint | grootte | controle |
|---|---|---|---:|---|
| `google/gemma-4-26B-A4B-it` | `4d7ae4984b7db7de8f8457170b3f1a419ee76d52` | `gs://t5-dutch-english-usc2/mbert/translator/gemma4-ckpt/gemma-4-26b-a4b-it/` | 39,5 GB | `commit_success.txt`, `done.txt`; boom met per laag `self_attention/{query,key,value,out}`, `mlp/moe_block/MoeBlock_0/{gate,wi_0,wi_1,wo}` (128 experts × 2816 × 704) en `mlp/moe_block/shared_experts/{wi_0,wi_1,wo}`; 5 globale lagen (5, 11, 17, 23, 29) zonder `value` |
| `google/gemma-4-31B-it` | `842da3794eaa0b77d5f08bae87a17459d91ff475` | `gs://t5-dutch-english-usc2/mbert/translator/gemma4-ckpt/gemma-4-31b-it/` | 47,8 GB | `_METADATA` byte-gelijk aan die van de bestaande 0.2.5-omzetting `…/translator/judge/gemma-4-31b-it-maxtext/` (9 nov, zelfde revisie); 10 globale lagen zonder `value` |

**Data** (`gs://t5-dutch-english-usc2/mbert/translator/sft/g4lora/data/`): de eerste 3.000 voorbeelden zonder context van
de set van arm RL (`…/sft/data/rl-b5/RL`, in de volgorde van die set), alleen voor deze test. De gebruikersbeurt is de
en→nl-instructie `natural-v1` van de pilot met de Engelse bron (`translate.py` `user_text`); de doeltekst is het
Nederlands van RL. Chat-template van Gemma 4: `<bos><|turn>user\n{instructie + bron}<turn|>\n<|turn>model\n<|channel>thought\n<channel|>`
als prompt (het eigen template van het model met denken uit sluit een leeg denkkanaal), `{Nederlands}<turn|>\n` als
doel. `g4lora.py data` controleert elk voorbeeld met MaxTexts eigen SFT-stappen (`apply_chat_template`, tokenisatie,
`SFTPromptMasking`) tegen de tokenizer en het eigen chat-template van elk van de twee modellen. Contextvoorbeelden
vallen weg omdat `natural-v1` geen contextvorm heeft.

## 2. Varianten en beslisregel (vastgelegd vóór de meting)

### 2.1 Varianten

Rank 64, alpha 32, LR 1e-5, `max_target_length` 2.048, 16 chips, 8 stappen, zoals de geheugentest van `run_arms.sh`.
qwix past de regex toe met `re.fullmatch` op het pad van een module [code gelezen: `qwix/_src/qconfig.py` r. 213].

| variant | model | doelmodules | regex (`lora.lora_module_path`) |
|---|---|---|---|
| v1 | 26B-A4B | attention (q, k, v, o) | `decoder/layers_[0-9]+/self_attention/(query\|key\|value\|out)` |
| v2 | 26B-A4B | attention + routed experts; de router (`MoeBlock_0/gate`, een eigen module) blijft bevroren | `decoder/layers_[0-9]+/(self_attention/(query\|key\|value\|out)\|mlp/moe_block/MoeBlock_0)` |
| v2b | 26B-A4B | attention + de dichte MLP naast de experts (`shared_experts`): de modules van TranslateGemma op de 26B | `decoder/layers_[0-9]+/(self_attention/(query\|key\|value\|out)\|mlp/moe_block/shared_experts/(wi_0\|wi_1\|wo))` |
| v3 | 31B | de modules van de TranslateGemma-armen: attention + MLP (`run_sft.sh`) | `decoder/layers_[0-9]+/.*(self_attention/(query\|key\|value\|out)\|mlp/(wi_0\|wi_1\|wo))` |
| v3b | 31B | als v3, rank 32 / alpha 16; alleen als v3 niet past | als v3 |
| v4 | de beste variant volgens §2.4 | bij `max_target_length` 4.096 | als die variant |

v2b staat niet in de opdracht. Ik voegde hem toe omdat hij de modules van de TranslateGemma-armen op de 26B zet, en dus
de directe tegenhanger is van v3 [eigen keuze].

**Fallback** [eigen keuze, technisch]: eindigt een variant in een fout die geen geheugentekort is, dan draait de keten hem
één keer opnieuw met `attention=dot_product` en voor de 26B ook `megablox=false` (`jax.lax.ragged_dot` in plaats van de
Pallas-kernel), voor het geval een TPU-kernel niet compileert op de v4. Verdere varianten van dat model krijgen dezelfde
instellingen als de fallback werkte. De tabel noemt de fallback per run.

### 2.2 Wat ik vooraf in de code las

- **qwix 0.1.8 legt LoRA alleen op onderschepte matmuls.** `LoraProvider` overschrijft `dot_general`, `einsum` en
  `conv_general_dilated` [code gelezen: `qwix/_src/providers/lora.py`; de onderscheppingslijst in `ptq.py` r. 413–424:
  `jax.lax.conv_general_dilated`, `jax.lax.dot_general`, `jax.numpy.asarray`, `jax.numpy.dot`, `jax.numpy.einsum` en de
  flax-functies]. `jax.lax.ragged_dot` staat er niet in. LoRA komt er alleen als `find_param` de rechteroperand herkent
  als een parameter van de huidige module (`lora.py` r. 263 en 337).
- **De experts van MaxText 0.2.5 rekenen standaard niet via zo'n matmul.** `RoutedMoE` (`MoeBlock_0`) houdt
  `wi_0`, `wi_1` en `wo` als eigen parameters van vorm (experts, in, uit) [code gelezen: `layers/moe.py` r. 1057–1077].
  `base.yml` heeft `sparse_matmul: true` en `megablox: true` (r. 249–250). Daarmee gaan de experts door `mblx.gmm`, een
  Pallas-kernel, of met `megablox: false` door `jax.lax.ragged_dot` (`moe.py` r. 2262–2340). Alleen `sparse_matmul:
  false` rekent met `jnp.einsum` (`"BSM,EMH -> BSEH"`, `moe.py` r. 4248), en dan voor elk token alle 128 experts.
- **MaxText zelf sluit de routed experts uit bij een ander MoE-model** [citaat, `configs/post_train/lora_module_path.yml`
  van de fork-basis `d1c0d0afe`, r. 23, bij `deepseek4`]: "Routed experts (MoeBlock) are excluded."
- **[gevolgtrekking, te toetsen]** Verwachting voor v2: de regex raakt `MoeBlock_0`, maar qwix maakt er geen LoRA, dus v2
  traint hetzelfde als v1. Het bewijs is de regel `LORA_MODULES` (moduletype `…/MoeBlock_0` onder "regex raakt de
  module, geen LoRA"). Er wordt niet omheen gebouwd: `sparse_matmul=false` is niet geprobeerd.
- **[eigen berekening]** Zou LoRA de experts per expert raken zoals qwix dat bij een `dot_general` met batch-as doet (A:
  experts × in × r, B: experts × r × uit), dan is dat bij rank 64 2.595.225.600 parameters. Met 16 bytes per parameter
  (float32-waarde, gradiënt en twee AdamW-momenten) en LoRA gerepliceerd op elke chip (`lora_utils.apply_lora_to_model`)
  is dat 38,7 GiB per chip, meer dan de 32 GiB van een v4-chip. Via de einsum-route (A gedeeld, B per expert) is het
  535.265.280 parameters, 8,0 GiB per chip, maar met de rekenkosten van alle experts per token.

### 2.3 Verwachte LoRA-grootte [eigen berekening, rank 64; gemeten wordt het in `LORA_MODULES`]

| variant | LoRA-parameters | toestand per chip (16 B per parameter, gerepliceerd) |
|---|---:|---:|
| v1 | 45.957.120 | 0,68 GiB |
| v2b | 74.342.400 | 1,11 GiB |
| v3 | 489.717.760 | 7,30 GiB |
| v3b (rank 32) | 244.858.880 | 3,65 GiB |
| TranslateGemma 27B, rank 64 (gemeten, arm RL) | 454.066.176 | 6,77 GiB |

Basisgewichten in float32 over 16 chips: 26B ~5,9 GiB, 31B ~7,2 GiB per chip; TranslateGemma 27B ~6,3 GiB
[eigen berekening]. TranslateGemma 27B rank 64 piekte bij 2.048 op 27,16 GiB in de geheugentest van 11 nov en op
25,82 GiB in de training van RL. **[gevolgtrekking]** Voor de 31B bij rank 64 verwacht ik daarom een piek dicht bij of
boven de grens van 29,5 GiB; de 26B heeft veel ruimte.

### 2.4 Beslisregel voor het advies

Een variant is **bruikbaar** als alle vier de workers op `EXIT 0` eindigen, de piek per chip ≤ 29,5 GiB is en de 8
verliezen eindig zijn. De **beste** variant is de eerste bruikbare in deze volgorde: v3, v3b, v2 (alleen als LoRA de
experts echt bereikte), v2b, v1. De 31B gaat voor, omdat hij iets beter was dan de 26B: in de blinde beoordeling tegen
TranslateGemma 56/22/22 met 17 betekenisfouten tegen 41 (31B) en 49/25/26 met 17 tegen 46 (26B) (opdracht van 13 nov),
en op MetricX-QE WMT24++ 2,230 tegen 2,289 en FLAN 3,07 tegen 3,15 (`gemma4-sizes.md` §5). Binnen een model telt: zoveel mogelijk LoRA die past. De beste variant gaat daarna door naar
4.096 (v4).

**Duur van één epoch** [schatting]: het aantal stappen van TranslateGemma 27B op die set (RL: 403, R2: 3.653, uit hun
`summary.json`) maal de tokenverhouding Gemma 4 : TranslateGemma op die set (`natural-v1` en de beurt van Gemma 4
tegenover de prompt en beurt van TranslateGemma; RL helemaal, R2 een steekproef van 20.000 voorbeelden zonder context;
contextvoorbeelden geschat met dezelfde verhouding), maal de mediane staptijd van stap 2–8 uit deze test, plus het laden
en compileren uit deze test. Zonder de evaluatie.

