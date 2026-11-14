# RobBERT door onze evaluatie

Doel: RobBERT-2023-base en RobBERT-2026 met precies de evaluatie van de eindrun (`experiments/final/protocol.md`,
"Evaluatie", met de extra evals van `experiments/dumb-extra/protocol.md`) naast het eindmodel T2 (`epoch2-1306000`)
en S (`s-653000`) zetten. Vastgelegd op 14 nov 2026, vóór de run.

## Modellen

| naam | Hugging Face | revisie | architectuur |
|---|---|---|---|
| `robbert-2023` | `DTAI-KULeuven/robbert-2023-dutch-base` | `1164b2df6d3ceb1735900c839da8de9bad1a99f9` | RoBERTa-base, 12 lagen, 768, geleerde posities (max 512 tokens) |
| `robbert-2026` | `pdelobelle/robbert-2026` | `a406b1a3eabf41401293e2a096167729bbd4d479` | NeoBERT, 28 lagen, 768, RoPE, SwiGLU, RMSNorm; voorgetraind op 512 tokens |
| `epoch2-1306000` | T2, `gs://t5-dutch-english-usc2/mbert/final-culturax-230m-epoch2/ckpt_keep/1306000` | | onze ModernBERT 230M |

RoBERTa en NeoBERT staan in `src/mmbjax/hf_encoders.py` (Flax, gewichten uit `model.safetensors`; transformers 5 heeft
geen Flax meer). `tests/test_hf_encoders.py` vergelijkt de MLM-logits met de PyTorch-code (transformers'
`RobertaForMaskedLM`, en `model.py` van `pdelobelle/robbert-2026` op de revisie hierboven): kleine willekeurige modellen
altijd, de echte checkpoints met `HF_ENCODER_REAL=1`.

## Wat gelijk blijft

Alles van `experiments/final/eval_final.sh` (`f510292`) en de code `ae11478`: de data en splits (DBRD en SICK-NL van
de hub zoals `data.py`; de extra evals uit `gs://t5-dutch-english-usc2/mbert/dumb-extra/data/` met de sha256 van
`manifest.json`), `finetune.py` en `finetune_extra.py` met dezelfde stappen, optimizer (AdamW, weight decay 0,01),
LR 5e-5, schema's (constant voor DBRD/SICK-NL, lineair naar 0 zonder warmup voor de extra evals), 3 epochs, updates
(DBRD 192, SICK-NL 256, extra 32), seeds 42, 43, 44, float32, evaluatie op test na de laatste epoch, dezelfde metrics
(DBRD/SICK-NL `test_accuracy` op 2.112 resp. 4.864 voorbeelden; SQuAD-NL v1.1 F1 en EM met de officiële normalisatie;
CoNLL-2002 entity-F1; UPOS-accuratesse) en BLiMP-NL met PLL-word-l2r over 8.400 paren. Hardware: mbert-mix, per worker
een TPU van 4 chips, venv `~/mbert-dumb/.venv`.

## Wat niet gelijk kan, en hoe

1. **DBRD@512 in plaats van DBRD@1024.** RoBERTa heeft 514 geleerde posities (512 tokens); RobBERT-2026 is op 512
   voorgetraind (de RoPE-tabel loopt tot 8.192, maar daarboven is niet getraind). Alle drie (ook T2) met
   `--max-length 512`, de update van DBRD@1024 (12 per chip × 4 chips × 4 accumulatie = 192) en opvullen tot een
   veelvoud van 128. T2's DBRD@512 draait met de code van al zijn andere evals (`ae11478` in `~/mbert-dumbx`) en het
   omgezette model van `eval_final.sh`.
2. **Tokenizer.** Elk model met zijn eigen tokenizer op zijn revisie. Afkappen van links, zoals onze tokenizer
   (`truncation_side: left`), zodat een DBRD-recensie langer dan 512 tokens bij elk model hetzelfde einde houdt.
   Voor de woordtaken (`is_split_into_words`) `add_prefix_space=True`, zoals `run_ner.py` voor RoBERTa doet; dan
   begint elk woord met een spatie, zoals bij ons met ▁.
3. **Offsets van RobBERT-2023.** Zijn tokenizer (pre-tokenizers Whitespace, ByteLevel met `add_prefix_space`,
   Punctuation) haalt van elk woord een spatie af die niet in de tekst staat: elk woord begint één teken te laat
   ("ĠDe" van "De kat" krijgt (1, 2)). SQuAD-NL haalt de antwoordposities en de antwoordtekst uit de offsets.
   `load_tokenizer` zet daarom `trim_offsets` van de post-processor uit als een probe dat laat zien: dezelfde ids,
   offsets die bij het woord beginnen. Ter diagnose één extra run, `robbert-2023-rawoffsets` (SQuAD-NL, seed 42),
   met de offsets zoals de tokenizer ze geeft. RobBERT-2026's tokenizer (alleen ByteLevel) heeft het probleem niet.
4. **Taakkoppen.** Elk model met de koppen van zijn eigen Hugging Face-klassen, nieuw geïnitialiseerd (normaal, std
   0,02; bias 0): zinnen `<s>` → dropout → dense → tanh → dropout → classifier (RobertaClassificationHead en
   NeoBERTForSequenceClassification hebben die vorm); tokens dropout → classifier (RobertaForTokenClassification;
   NeoBERT levert geen tokenkop, dus dezelfde vorm); spans één lineaire laag (`qa_outputs`). Onze ModernBERT houdt
   zijn kop (de MLM-transformatie als begin, dan een nieuwe classifier), zoals HF's ModernBERT-klassen. Dropout zoals
   in elke config: RoBERTa 0,1 in embeddings, attention en de residuele takken, classifier-dropout 0,1; NeoBERT geen
   dropout in de encoder, classifier-dropout 0,1 (de standaard van zijn `NeoBERTForSequenceClassification`).
5. **BLiMP-NL.** `scripts/eval_blimp_nl_hf.py` met de maskers en paarscore van `eval_blimp_nl.py`; alleen de woorden
   worden anders uit de tokens gelezen (byte-level BPE: een spatie aan het begin van de bytes; een token zonder hele
   UTF-8-tekst zet het woord voort; het losse spatietoken dat RobBERT-2023 vóór elk leesteken zet, hoort bij het
   leesteken). Geen packing: elke gemaskeerde kopie is een eigen rij.

## Uitslagen en rapport

- `gs://t5-dutch-english-usc2/mbert/final/robbert/dumb/<model>--{dbrd512,sicknl}--seed{42,43,44}.json`
- `gs://t5-dutch-english-usc2/mbert/final/robbert/dumb-extra/<taak>/<model>--<taak>--seed{42,43,44}.json`
- `gs://t5-dutch-english-usc2/mbert/final/robbert/blimp-nl/<model>.json`
- `experiments/robbert/report.py`: per taak gemiddelde en range over de 3 seeds, en per paar modellen de vorm van
  `decide.py`: "beter"/"slechter" als het verschil van de gemiddelden groter is dan de grootste seed-range van de twee
  (BLiMP-NL: meer dan 0,5 punt). T2 en S komen uit `gs://t5-dutch-english-usc2/mbert/final/{dumb,dumb-extra,blimp-nl}/`.
  Rapport `robbert-compare.md` en `.json` in `gs://t5-dutch-english-usc2/mbert/final/robbert/` en
  `mbert-ops/dx/final/`; intern (CoNLL-2002-getallen niet publiceren).

Geen DBRD- of SICK-NL-testdata in de training: alleen fine-tunen op train, testen zoals het protocol.
