"""Debug layer 0 step-by-step against PyTorch."""
import os
os.environ["JAX_PLATFORMS"] = "cpu"

import sys
sys.path.insert(0, ".")

import numpy as np
import torch
import jax
import jax.numpy as jnp
from transformers import AutoModel

from src.mmbjax.configuration_mmbbert import MMBertConfig
from src.mmbjax.modeling_mmbbert import ModernBertModel
from src.mmbjax.utils.checkpoints import convert_pytorch_state_dict_to_flax, load_hf_config_as_mmbert

# Load reference data
golden_batch = np.load("artifacts/reference-configs/jhu-clsp_mmBERT-small_batch.npz")
input_ids = golden_batch["input_ids"]
attention_mask = golden_batch["attention_mask"]
batch, seq_len = input_ids.shape

print("="*80)
print("LOADING MODELS")
print("="*80)

# Load PyTorch model
pt_model = AutoModel.from_pretrained("jhu-clsp/mmBERT-small")
pt_model.eval()

# Load JAX model
config = load_hf_config_as_mmbert("artifacts/reference-configs/jhu-clsp_mmBERT-small_config.json")
jax_model = ModernBertModel(config=config, dtype=jnp.float32)

# Convert weights
pt_state_dict = {f"model.{k}" if not k.startswith("model.") else k: v 
                 for k, v in pt_model.state_dict().items()}
jax_params_wrapped = convert_pytorch_state_dict_to_flax(pt_state_dict, config)
jax_params = jax_params_wrapped["params"]["model"]

# Initialize
rng = jax.random.PRNGKey(0)
dummy_input = jnp.ones((1, 1), dtype=jnp.int32)
dummy_mask = jnp.ones((1, 1), dtype=jnp.float32)
variables = jax_model.init(rng, dummy_input, attention_mask=dummy_mask, deterministic=True)
variables = {"params": jax_params}

print("\n" + "="*80)
print("STEP 1: EMBEDDINGS")
print("="*80)

# PyTorch embeddings
with torch.no_grad():
    pt_emb = pt_model.embeddings.tok_embeddings(torch.from_numpy(input_ids))
    pt_emb = pt_model.embeddings.norm(pt_emb)
    print(f"PyTorch embeddings: {pt_emb.shape}")
    print(f"Sample [0,0,:5]: {pt_emb[0,0,:5]}")

# JAX embeddings
jax_emb = jax_model.apply(variables, jnp.array(input_ids), deterministic=True, 
                          method=lambda self, ids, **kw: self.embeddings(input_ids=ids, **kw))
print(f"JAX embeddings: {jax_emb.shape}")
print(f"Sample [0,0,:5]: {jax_emb[0,0,:5]}")

emb_diff = np.abs(np.array(jax_emb) - pt_emb.numpy()).max()
print(f"\nEmbeddings diff: {emb_diff:.2e} {'✅' if emb_diff < 1e-4 else '❌'}")

if emb_diff > 1e-4:
    print("\n⚠️  Embeddings already diverge! Debugging...")
    print(f"PT mean: {pt_emb.mean():.6f}, std: {pt_emb.std():.6f}")
    print(f"JAX mean: {np.array(jax_emb).mean():.6f}, std: {np.array(jax_emb).std():.6f}")
    
    # Check token embeddings only
    with torch.no_grad():
        pt_tok_only = pt_model.embeddings.tok_embeddings(torch.from_numpy(input_ids))
    jax_tok_emb = variables["params"]["embeddings"]["word_embeddings"]["embedding"]
    jax_tok_only = jnp.take(jax_tok_emb, input_ids, axis=0)
    
    tok_diff = np.abs(np.array(jax_tok_only) - pt_tok_only.numpy()).max()
    print(f"Token embedding diff (before LN): {tok_diff:.2e}")
    
    # Check LN weights
    pt_ln_weight = pt_model.embeddings.norm.weight.numpy()
    jax_ln_weight = np.array(variables["params"]["embeddings"]["layer_norm"]["scale"])
    ln_weight_diff = np.abs(jax_ln_weight - pt_ln_weight).max()
    print(f"LN weight diff: {ln_weight_diff:.2e}")
    
    print("\nDone with embedding debug")
    sys.exit(1)

print("\n" + "="*80)
print("STEP 2: LAYER 0 (NO ATTN NORM)")
print("="*80)

# Get layer 0 from PyTorch
pt_layer0 = pt_model.layers[0]
print(f"Layer 0 is global: {config.is_global_layer(0)}")
print(f"Has attn_norm in PT: {hasattr(pt_layer0, 'attn_norm')}")

# Run PyTorch layer 0
with torch.no_grad():
    pt_input = pt_emb  # No norm!
    
    # Get attention mask (4D)
    from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask
    pt_attn_mask_4d = _prepare_4d_attention_mask(
        torch.from_numpy(attention_mask), 
        dtype=pt_emb.dtype, 
        tgt_len=seq_len
    )
    print(f"\nPyTorch 4D mask shape: {pt_attn_mask_4d.shape if pt_attn_mask_4d is not None else None}")
    if pt_attn_mask_4d is not None:
        print(f"Unique values: {torch.unique(pt_attn_mask_4d)}")
        print(f"Mask [0,0,0,:]: {pt_attn_mask_4d[0,0,0,:]}")
    
    # Run attention
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(batch, -1)
    pt_layer0_result = pt_layer0(pt_input, attention_mask=pt_attn_mask_4d, position_ids=position_ids)
    if isinstance(pt_layer0_result, tuple):
        pt_attn_out = pt_layer0_result[0]
    else:
        pt_attn_out = pt_layer0_result
    print(f"\nPyTorch layer 0 output: {pt_attn_out.shape}")
    print(f"Sample [0,0,:5]: {pt_attn_out[0,0,:5]}")

# Run JAX encoder (full)
jax_enc_out = jax_model.apply(
    variables, 
    jnp.array(input_ids),
    attention_mask=jnp.array(attention_mask),
    deterministic=True
)
jax_final = jax_enc_out.last_hidden_state
print(f"\nJAX full encoder output: {jax_final.shape}")
print(f"Sample [0,0,:5]: {jax_final[0,0,:5]}")

layer0_diff = np.abs(np.array(jax_final) - pt_attn_out.numpy()).max()
print(f"\nLayer 0 output diff: {layer0_diff:.2e} {'✅' if layer0_diff < 1e-4 else '❌'}")

if layer0_diff > 1e-4:
    print("\n⚠️  Layer 0 diverges!")
    print("This suggests attention or MLP computation is wrong")

print("\n" + "="*80)
print("CHECKING ATTENTION MASK CONVERSION")
print("="*80)

# Check how we convert attention mask
print(f"\nInput attention_mask:\n{attention_mask}")
print(f"PyTorch 4D mask [0,0,:,:]:")
if pt_attn_mask_4d is not None:
    print(pt_attn_mask_4d[0,0,:,:])

# Check JAX mask handling
from src.mmbjax.modules.attention import ModernBertAttention, _ensure_attention_mask
jax_attn_mask_2d = _ensure_attention_mask(jnp.array(attention_mask), batch, seq_len, jnp.float32)
print(f"\nJAX attention_mask (2D, ensured): {jax_attn_mask_2d.shape}")
print(jax_attn_mask_2d)

# Check padding mask computation
mask_value = jnp.array(-1e9, dtype=jnp.float32)
padding_mask = (1.0 - jax_attn_mask_2d) * mask_value
padding_mask_4d = padding_mask[:, None, None, :]
print(f"\nJAX padding mask (4D): {padding_mask_4d.shape}")
print(f"Mask [0,0,0,:]: {padding_mask_4d[0,0,0,:]}")

# Compare
if pt_attn_mask_4d is not None:
    pt_mask_np = pt_attn_mask_4d.numpy()
    jax_mask_np = np.array(padding_mask_4d)
    mask_diff = np.abs(pt_mask_np - jax_mask_np).max()
    print(f"\nPadding mask diff: {mask_diff:.2e}")
    
    if mask_diff > 1e-6:
        print("\n⚠️  ATTENTION MASKS DIFFER!")
        print("This is likely the root cause!")
        print(f"\nPT mask [0,0,0,:]:  {pt_mask_np[0,0,0,:]}")
        print(f"JAX mask [0,0,0,:]: {jax_mask_np[0,0,0,:]}")
