"""Debug first transformer layer attention against PyTorch."""
import os
os.environ["JAX_PLATFORMS"] = "cpu"

import sys
import numpy as np
import torch
import jax
import jax.numpy as jnp
from transformers import AutoModel

# Load reference data
artifacts_dir = "artifacts/reference-configs"
golden_batch = np.load(f"{artifacts_dir}/jhu-clsp_mmBERT-small_batch.npz")

input_ids = golden_batch["input_ids"]
attention_mask = golden_batch["attention_mask"]
batch, seq_len = input_ids.shape

print(f"Batch size: {batch}, Seq len: {seq_len}")
print(f"Input IDs shape: {input_ids.shape}")
print(f"Attention mask shape: {attention_mask.shape}")
print(f"Attention mask:\n{attention_mask}")

# Load PyTorch model
print("\nLoading PyTorch model...")
pt_model = AutoModel.from_pretrained("jhu-clsp/mmBERT-small")
pt_model.eval()

# Run PyTorch forward pass with output_hidden_states
with torch.no_grad():
    pt_inputs = {
        "input_ids": torch.from_numpy(input_ids),
        "attention_mask": torch.from_numpy(attention_mask),
    }
    pt_outputs = pt_model(**pt_inputs, output_hidden_states=True)
    
# Get embeddings and first hidden state
pt_embeddings = pt_outputs.hidden_states[0]  # After embedding layer
pt_layer0_out = pt_outputs.hidden_states[1]  # After first transformer layer

print(f"\nPyTorch embeddings shape: {pt_embeddings.shape}")
print(f"PyTorch layer 0 output shape: {pt_layer0_out.shape}")
print(f"PyTorch embeddings sample [0,0,:5]: {pt_embeddings[0,0,:5]}")
print(f"PyTorch layer 0 sample [0,0,:5]: {pt_layer0_out[0,0,:5]}")

# Now load our JAX model
print("\nLoading JAX model...")
sys.path.insert(0, ".")
from src.mmbjax.configuration_mmbbert import MMBertConfig
from src.mmbjax.modeling_mmbbert import ModernBertModel
from src.mmbjax.utils.checkpoints import convert_pytorch_state_dict_to_flax, load_hf_config_as_mmbert

# Load config
config_path = f"{artifacts_dir}/jhu-clsp_mmBERT-small_config.json"
jax_config = load_hf_config_as_mmbert(config_path)

# Create model
jax_model = ModernBertModel(config=jax_config, dtype=jnp.float32)

# Convert weights
print("Converting weights...")
pt_state_dict = pt_model.state_dict()
# Add "model." prefix to keys if not present
pt_state_dict_prefixed = {f"model.{k}" if not k.startswith("model.") else k: v 
                          for k, v in pt_state_dict.items()}
jax_params_wrapped = convert_pytorch_state_dict_to_flax(pt_state_dict_prefixed, jax_config)
# Unwrap the "params" and "model" layers for ModernBertModel (not ForMaskedLM)
jax_params = jax_params_wrapped["params"]["model"]

# Initialize model
rng = jax.random.PRNGKey(0)
dummy_input = jnp.ones((1, 1), dtype=jnp.int32)
dummy_mask = jnp.ones((1, 1), dtype=jnp.float32)
variables = jax_model.init(rng, dummy_input, attention_mask=dummy_mask, deterministic=True)

# Replace params with converted weights
variables = {"params": jax_params}

# Run JAX forward
print("\nRunning JAX forward pass...")
jax_output = jax_model.apply(
    variables,
    jnp.array(input_ids),
    attention_mask=jnp.array(attention_mask),
    deterministic=True,
)

jax_last_hidden = jax_output.last_hidden_state
print(f"JAX output shape: {jax_last_hidden.shape}")
print(f"JAX output sample [0,0,:5]: {jax_last_hidden[0,0,:5]}")

# Compare layer 0
diff = np.abs(np.array(jax_last_hidden) - pt_layer0_out.numpy()).max()
print(f"\n❌ Max diff vs layer 0: {diff:.2e}")

# Now let's manually compute layer 0 step-by-step
print("\n" + "="*80)
print("MANUAL LAYER 0 COMPUTATION")
print("="*80)

# Get embeddings from both
jax_emb = jax_model.apply(
    variables,
    jnp.array(input_ids),
    attention_mask=jnp.array(attention_mask),
    deterministic=True,
    method=lambda module, *args, **kwargs: module.embeddings(*args, **kwargs)
)

print(f"\nJAX embeddings shape: {jax_emb.shape}")
print(f"JAX embeddings sample [0,0,:5]: {jax_emb[0,0,:5]}")

emb_diff = np.abs(np.array(jax_emb) - pt_embeddings.numpy()).max()
print(f"Embeddings diff: {emb_diff:.2e} {'✅' if emb_diff < 1e-5 else '❌'}")

# Get first encoder layer
pt_layer = pt_model.encoder.layers[0]
jax_layer_params = jax_params["encoder"]["layers"]["0"]

# PyTorch layer 0 forward (manual)
print("\n" + "-"*80)
print("Layer 0 components:")
print("-"*80)

# Note: Layer 0 is special - no attn_norm!
print("Layer 0 is GLOBAL and has NO attention norm")

# PyTorch attention forward
with torch.no_grad():
    # 1. QKV projection
    pt_hidden = pt_embeddings  # No norm for layer 0
    pt_qkv = pt_layer.attn.Wqkv(pt_hidden)  # [batch, seq, 3*hidden]
    print(f"\nPT QKV shape: {pt_qkv.shape}")
    print(f"PT QKV sample [0,0,:5]: {pt_qkv[0,0,:5]}")
    
    # 2. Reshape to [batch, seq, 3, heads, head_dim]
    num_heads = 6
    head_dim = 64
    pt_qkv = pt_qkv.reshape(batch, seq_len, num_heads, 3, head_dim)
    pt_qkv = pt_qkv.transpose(1, 3)  # [batch, 3, heads, seq, head_dim] ???
    
    # Actually, let me check the PyTorch code more carefully
    print("\nChecking PyTorch attention implementation...")

# Let me trace through PyTorch attention more carefully
print("\nDebugging PyTorch attention mask...")
pt_attn_mask = torch.from_numpy(attention_mask)
print(f"Input attention_mask shape: {pt_attn_mask.shape}")
print(f"Input attention_mask:\n{pt_attn_mask}")

# Check if PyTorch expands the mask
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask
pt_4d_mask = _prepare_4d_attention_mask(pt_attn_mask, dtype=torch.float32, tgt_len=seq_len)
print(f"\nPyTorch 4D mask shape: {pt_4d_mask.shape if pt_4d_mask is not None else None}")
if pt_4d_mask is not None:
    print(f"4D mask [0,0,0,:]: {pt_4d_mask[0,0,0,:]}")
    print(f"4D mask unique values: {torch.unique(pt_4d_mask)}")

print("\n" + "="*80)
print("HYPOTHESIS: Attention mask preprocessing differs")
print("="*80)
