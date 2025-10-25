# MMBERT → JAX/Flax Porting Plan

## 0. Goal & Success Criteria
- Reproduce Hugging Face `ModernBertModel` (a.k.a. MMBERT) in pure JAX/Flax with parity for forward passes, masked-LM head, and attention behavior (global/local alternation + dual RoPE scales).
- Load at least one published checkpoint (`answerdotai/ModernBERT-small` or `ModernBERT-base`) via a conversion script and match PyTorch logits within 1e-5 relative tolerance for a deterministic test batch.
- Support both padded (standard) and unpadded (FlashAttention-style) execution paths so we can scale to ≥8k tokens without quadratic blowups.
- Enable Dutch-domain pre-training by exposing the same configuration knobs as PyTorch (rope scales, local window, dropout knobs, sparse prediction flags, classifier pooling choices).

## 1. Reference Baseline & Recon
1. Pin the reference version: clone or vendor `transformers==4.57.1` and freeze `src/transformers/models/modernbert/*` under `vendor/pytorch_reference/` for diffable inspection.
2. Smoke test PyTorch: load `answerdotai/ModernBERT-small` with `ModernBertForMaskedLM.from_pretrained`, run a dummy forward, dump the config json, and archive it under `artifacts/reference-configs/modernbert-small.json`.
3. Capture key tensors for regression: serialize one batch of `input_ids`, `attention_mask`, `position_ids`, logits, and hidden states to `.npz`; these serve as golden outputs for the Flax port.
4. Document any config deviations (e.g., `decoder_bias=True` even though upstream blocks are bias-free, `classifier_bias`, `sparse_prediction`) so the Flax config mirrors every flag.

## 2. Architecture Facts to Preserve
- **Embeddings**: token embeddings only (no token-type or position embeddings) + LayerNorm (ε=1e-5, bias controlled by `norm_bias`) + dropout (`embedding_dropout`).
- **Norm policy**: Pre-LN everywhere. Layer 0 skips the attention norm (identity); all other attn/MLP norms share `norm_bias` and `norm_eps`.
- **Attention**:
  - Single fused `Wqkv` linear (bias governed by `attention_bias`) projecting to `[batch, seq, 3 * heads * head_dim]`.
  - Alternating global/local layers defined by `(layer_idx % global_attn_every_n_layers == 0)` ⇒ layer 0, N, 2N… are global; others use local window `[−local/2, +local/2]`.
  - Dual RoPE scales: global layers use `global_rope_theta`, locals use `local_rope_theta` (fall back to global if local is None); RoPE dim equals head_dim.
  - Supports attention implementations: baseline (padded) softmax, FlashAttention-style varlen path (requires unpadding & packing), and SDPA fallback. JAX plan: start with padded attention + mask, then add optional block-sparse/flash path once correctness is proven.
- **Sliding-window mask**: For non-global layers, replace the 4D attention mask with one that masks out tokens beyond ±`local_attention/2`. Needs to compose with padding mask (i.e., `-inf` outside window *or* padded positions).
- **FFN/MLP**: GeGLU variant — `Wi` outputs 2×intermediate → split into `(act_input, gate)` with `ACT2FN[hidden_activation]` applied to the first chunk before elementwise product with the second chunk; `Wo` projects back to hidden. Bias toggled by `mlp_bias`, dropout by `mlp_dropout`.
- **Dropouts**: `attention_dropout`, `embedding_dropout`, `mlp_dropout`, `classifier_dropout` must all be configurable.
- **Prediction head**: dense (bias controlled by `classifier_bias`) → activation (`classifier_activation`) → LayerNorm → decoder (weights tied to embeddings, bias flag `decoder_bias`).
- **Unpadding utilities**: PyTorch version unpads inputs when using FlashAttention, producing `(indices, cu_seqlens, max_seqlen)` and later repads outputs. We need analogous JAX helpers (likely with `jax.vmap`/`segment_sum`) to unlock efficient long-context inference.
- **Init scheme**: truncated normal with layer-specific stds (`initializer_range`, `initializer_range/√(2L)` for outputs, `hidden_size^-0.5` for classifier); preserve to ease weight conversion parity.

## 3. Work Phases

### Phase 0 – Environment & Tooling (1 day)
- Ensure Python env has `jax`, `flax`, `optax`, `transformers==4.57.1` (for reference) plus testing deps (`pytest`, `numpy`).
- Create `scripts/py_reference_dump.py` that loads a ModernBERT checkpoint, logs config/params, and exports the golden batch mentioned above.

### Phase 1 – Configuration & Shared Types (0.5–1 day)
- Implement `mmbjax/configuration_mmbbert.py`:
  - Dataclass or HF-style config replicating every field from PyTorch `ModernBertConfig` (vocab_size, hidden_size, intermediate_size, max_position_embeddings, rope thetas, attention & MLP bias flags, window size, dropout knobs, classifier settings, sparse prediction toggles, `deterministic_flash_attn`, `repad_logits_with_grad`).
  - Provide serialization helpers (`to_dict`, `from_pretrained`) plus validation (e.g., ensure local window is even, `hidden_size % num_heads == 0`).
  - Add convenience properties such as `head_dim`, `num_global_layers = ceil(num_layers / global_attn_every_n_layers)`.

### Phase 2 – Core Modules (3–4 days)
1. **Embeddings module** (`mmbjax/modules/embeddings.py`): token embedding table, optional dropout, LN with `use_bias=config.norm_bias`. Support `deterministic` flag.
2. **RoPE utilities** (`mmbjax/modules/rope.py`):
   - `build_rotary_cache(max_seq, dim, theta)` returning `(cos, sin)` arrays.
   - `apply_rotary(q, k, cos, sin)` using `rotate_half` helper.
   - Provide cache manager to reuse per-device/per-theta tables; handle both padded and var-len inputs (i.e., support gathering by `position_ids`).
3. **Local attention mask helper** (`mmbjax/modules/masks.py`): function that, given `attention_mask` `[batch, seq]` and `window_size`, emits `[batch, 1, seq, seq]` mask with `-inf` outside window + padded spots. Optimize via Toeplitz trick or `jax.lax.dynamic_slice` to avoid explicit `seq x seq` for first version (precompute `[seq, seq]` mask once and broadcast; later optimize with block-sparse attention).
4. **Attention module** (`mmbjax/modules/attention.py`):
   - Fused dense for QKV (bias flag), reshape to `[batch, heads, seq, head_dim]`.
   - Choose theta per layer; call RoPE helper.
   - For local layers, pass sliding-window mask; for global, fall back to standard mask.
   - Provide two execution paths: (a) padded softmax attention using `jax.nn.softmax` and mask addition; (b) optional experimental FlashAttention path using `jax.experimental.pallas` or `jax.experimental.shard_map` once ready. Keep API that accepts `unpadded_qkv` + `indices` for future use.
5. **MLP/GeGLU** (`mmbjax/modules/mlp.py`): implement Wi split + gating exactly like PyTorch (activate first half, multiply by second, dropout, Wo). Honor `mlp_bias`, `mlp_dropout`.
6. **Transformer layer** (`mmbjax/modules/layer.py`): pre-LN pipeline, identity norm for layer 0, gradient checkpoint option via `flax.linen.remat`. Accept/propagate `attention_mask`, `sliding_window_mask`, `position_ids`, `unpadded_metadata`.

### Phase 3 – Encoder & Model Wrappers (2 days)
- **Encoder**: stack `num_hidden_layers` layers, manage alternating attention by passing `layer_idx` so attention modules can compute `is_global = (layer_idx % config.global_attn_every_n_layers == 0)`.
- **Final Norm**: apply same LN as embeddings with shared `norm_bias` semantics.
- **Model module** (`mmbjax/modeling_mmbbert.py`): combine embeddings + encoder, expose `__call__(input_ids=None, inputs_embeds=None, attention_mask=None, position_ids=None, sliding_window_mask=None, deterministic=True, unpadded_metadata=None)`.
- **Heads**: MaskedLM (prediction head + decoder, tied weights), SequenceClassification/TokenClassification/QA/MultipleChoice mirroring PyTorch API including pooling choice (`cls` vs `mean`) and dropout. Respect `classifier_bias`, `classifier_dropout`, `classifier_activation`.
- **Init**: implement truncated normal initializers that match PyTorch stds for embeddings/attention/MLP/out layers to minimize conversion drift.

### Phase 4 – Weight Conversion Tooling (1–2 days)
1. Build `scripts/convert_mmbbert_pt_to_flax.py`:
   - Load PyTorch state dict, map parameter names to Flax tree (`embeddings/tok_embeddings/embedding`, `encoder/layers/0/attn/Wqkv/kernel`, …).
   - Transpose dense kernels (`torch [out, in] → flax [in, out]`), reshape fused `Wqkv` to match JAX layout.
   - Handle LayerNorm weights (bias may be absent depending on config) and final decoder tie.
   - Support optional removal of biases if Flax config disables them; warn if PyTorch checkpoint contains bias tensors but config expects none.
2. Verification script `scripts/compare_pt_flax.py`: loads PyTorch + Flax models, runs identical batch, reports max |diff| on hidden states & logits. Target: <1e-5 relative.

### Phase 5 – Testing & Validation (2 days)
- **Unit tests** (`tests/`):
  - Embeddings shapes & dropout determinism.
  - RoPE correctness vs NumPy reference for multiple theta values.
  - Attention windowing mask ensures tokens outside ±window are suppressed (check by comparing softmax weights).
  - GeGLU gating equivalence to PyTorch for random inputs.
  - Encoder parity on tiny configs (e.g., 2 layers, 2 heads) to keep tests fast.
- **Integration tests**:
  - Load converted `ModernBERT-small` weights, run golden batch, assert closeness to saved PyTorch outputs.
  - Verify unpadded->padded round trip matches padded path when FlashAttention shim disabled.
  - Optional TPU/GPU multi-device smoke test using `pmap`/`pjit` with gradient checkpointing.
- **Parity coverage status (Jan 2025)**: ModernBERT-small and ModernBERT-base golden fixtures now backstop logits + per-layer hidden-state parity (CPU, deterministic). Next milestones: add a mixed-precision parity sweep (bf16/fp16) and duplicate the checks once the Flash/unpadded attention path lands.

### Phase 6 – Training & Dutch Adaptation (ongoing)
1. **Data pipeline**: adapt existing Dutch corpora loaders (mC4-nl, OSCAR-nl) using HF datasets + streaming; implement MLM masking strategy (15% with 80/10/10 rule).
2. **Training loop** (`train.py`): `flax.training.train_state`, Optax AdamW with linear warmup + cosine decay, gradient accumulation support, logging (TensorBoard/W&B).
3. **Resume from English weights**: replace embedding matrix / decoder with Dutch tokenizer, optionally average-case reinit for new vocab tokens.
4. **Long-context validation**: run sequences up to 8k tokens, profile memory/latency, benchmark padded vs (future) unpadded path.

### Phase 7 – Performance & Extras (optional)
- Implement JAX FlashAttention kernel or integrate with `jax_flash_attention` project; wire in deterministic flag similar to PyTorch’s `deterministic_flash_attn`.
- Add pad-unpad utilities using `jax.experimental.sparse` or `segment_sum` to avoid O(N²) cost on long sequences when not using flash attn.
- Expose `reference_compile`-style toggle (e.g., `jit_compile` flag) for experiments with `jax.jit` vs `pjit`.
- Implement sparse prediction mode if needed (mirrors PyTorch’s `sparse_prediction` branch in MLM loss).

## 4. Milestone Timeline (aggressive)
- Week 1: Phase 0–1 (reference dump + config) complete.
- Week 2: Phase 2 modules coded with baseline masked attention path.
- Week 3: Phase 3 assembly + MLM/CLS heads; run first forward pass with random weights.
- Week 4: Phase 4 conversion + parity test on `ModernBERT-small`.
- Week 5: Phase 5 testing suite stabilized; begin Dutch data plumbing.
- Week 6+: Train Dutch-adapted model, iterate on performance (Phase 6/7) as resources allow.

## 5. Outstanding Questions / TODOs Before Coding
- Confirm which checkpoints (e.g., `answerdotai/mmbert-small` vs `ModernBERT-small`) we target first and whether any use non-default bias flags.
- Decide on initial hardware target (TPU vs GPU) to choose between `pmap`/`pjit` scaffolding.
- Evaluate availability of JAX FlashAttention kernels; if none are production ready, schedule time to contribute or plan on efficient block-sparse masking instead.
