"""Ultra-detailed debugging of attention computation."""
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
from src.mmbjax.modules.attention import ModernBertAttention
from src.mmbjax.utils.checkpoints import convert_pytorch_state_dict_to_flax, load_hf_config_as_mmbert

# Load reference data
golden_batch = np.load("artifacts/reference-configs/jhu-clsp_mmBERT-small_batch.npz")
input_ids = golden_batch["input_ids"]
attention_mask = golden_batch["attention_mask"]
batch, seq_len = input_ids.shape

print("="*80)
print("ULTRA-DETAILED ATTENTION DEBUG")
print("="*80)

# Load models
pt_model = AutoModel.from_pretrained("jhu-clsp/mmBERT-small")
pt_model.eval()

config = load_hf_config_as_mmbert("artifacts/reference-configs/jhu-clsp_mmBERT-small_config.json")

# Get embeddings from PyTorch
with torch.no_grad():
    pt_emb = pt_model.embeddings.tok_embeddings(torch.from_numpy(input_ids))
    pt_emb = pt_model.embeddings.norm(pt_emb)

embeddings_np = pt_emb.numpy()
print(f"\nEmbeddings shape: {embeddings_np.shape}")
print(f"Sample [0,0,:5]: {embeddings_np[0,0,:5]}")

# Load JAX model
jax_model = ModernBertModel(config=config, dtype=jnp.float32)
pt_state_dict = {f"model.{k}" if not k.startswith("model.") else k: v 
                 for k, v in pt_model.state_dict().items()}
jax_params_wrapped = convert_pytorch_state_dict_to_flax(pt_state_dict, config)
jax_params = jax_params_wrapped["params"]["model"]
variables = {"params": jax_params}

print("\n" + "="*80)
print("STEP 1: QKV PROJECTION")
print("="*80)

# PyTorch QKV
pt_layer0 = pt_model.layers[0]
with torch.no_grad():
    pt_qkv = pt_layer0.attn.Wqkv(pt_emb)  # [batch, seq, 3*hidden]
    
print(f"PyTorch QKV shape: {pt_qkv.shape}")
print(f"Sample [0,0,:5]: {pt_qkv[0,0,:5]}")

# JAX QKV
jax_attn = ModernBertAttention(config=config, layer_idx=0, dtype=jnp.float32)
rng = jax.random.PRNGKey(0)
attn_variables = jax_attn.init(
    rng, 
    jnp.ones((1, 1, config.hidden_size)), 
    attention_mask=jnp.ones((1, 1)),
    deterministic=True
)
attn_variables = {"params": jax_params["encoder"]["layers_0"]["attention"]}

# Forward through QKV projection only
jax_emb = jnp.array(embeddings_np)
jax_qkv = jax_attn.apply(
    attn_variables,
    jax_emb,
    method=lambda self, h: self.qkv(h)
)

print(f"JAX QKV shape: {jax_qkv.shape}")
print(f"Sample [0,0,:5]: {jax_qkv[0,0,:5]}")

qkv_diff = np.abs(np.array(jax_qkv) - pt_qkv.numpy()).max()
print(f"\nQKV diff: {qkv_diff:.2e} {'✅' if qkv_diff < 1e-4 else '❌'}")

if qkv_diff > 1e-4:
    print("\n⚠️  QKV projection differs!")
    print("This means weights weren't loaded correctly")
    sys.exit(1)

print("\n" + "="*80)
print("STEP 2: RESHAPE & TRANSPOSE QKV")
print("="*80)

# PyTorch reshape
num_heads = 6
head_dim = 64
with torch.no_grad():
    pt_qkv_reshaped = pt_qkv.reshape(batch, seq_len, num_heads, 3, head_dim)
    pt_qkv_transposed = pt_qkv_reshaped.transpose(1, 3)  # [batch, 3, heads, seq, head_dim]
    pt_q = pt_qkv_transposed[:, 0]  # [batch, heads, seq, head_dim]
    pt_k = pt_qkv_transposed[:, 1]
    pt_v = pt_qkv_transposed[:, 2]

print(f"PyTorch Q shape: {pt_q.shape}")
print(f"PyTorch K shape: {pt_k.shape}")
print(f"PyTorch V shape: {pt_v.shape}")
print(f"PT Q sample [0,0,0,:5]: {pt_q[0,0,0,:5]}")

# JAX reshape
jax_qkv_reshaped = jax_qkv.reshape(batch, seq_len, num_heads, 3, head_dim)
jax_qkv_transposed = jnp.transpose(jax_qkv_reshaped, (0, 2, 1, 3, 4))  # [batch, heads, seq, 3, head_dim]
jax_q = jax_qkv_transposed[:, :, :, 0, :]
jax_k = jax_qkv_transposed[:, :, :, 1, :]
jax_v = jax_qkv_transposed[:, :, :, 2, :]

print(f"JAX Q shape: {jax_q.shape}")
print(f"JAX K shape: {jax_k.shape}")
print(f"JAX V shape: {jax_v.shape}")
print(f"JAX Q sample [0,0,0,:5]: {jax_q[0,0,0,:5]}")

q_diff = np.abs(np.array(jax_q) - pt_q.numpy()).max()
print(f"\nQ diff before RoPE: {q_diff:.2e} {'✅' if q_diff < 1e-4 else '❌'}")

print("\n" + "="*80)
print("STEP 3: APPLY RoPE")
print("="*80)

# PyTorch RoPE
with torch.no_grad():
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(batch, -1)
    cos, sin = pt_layer0.attn.rotary_emb(pt_qkv, position_ids=position_ids)
    
    print(f"PT cos/sin before unsqueeze: {cos.shape}")
    print(f"PT cos[0,0,:5]: {cos[0,0,:5]}")
    print(f"PT cos[0,1,:5]: {cos[0,1,:5]}")
    
    # Apply rotation
    from transformers.models.modernbert.modeling_modernbert import apply_rotary_pos_emb
    cos_expanded = cos.unsqueeze(1)  # [batch, 1, seq, dim]
    sin_expanded = sin.unsqueeze(1)
    print(f"PT cos after unsqueeze: {cos_expanded.shape}")
    print(f"PT cos sample [0,0,0,:5]: {cos_expanded[0,0,0,:5]}")
    
    # Manual application to debug
    def pt_rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    
    pt_q_manual = (pt_q * cos_expanded) + (pt_rotate_half(pt_q) * sin_expanded)
    pt_k_manual = (pt_k * cos_expanded) + (pt_rotate_half(pt_k) * sin_expanded)
    
    pt_q_rot, pt_k_rot = apply_rotary_pos_emb(pt_q, pt_k, cos, sin)

print(f"PyTorch cos shape: {cos.shape}")
print(f"PyTorch sin shape: {sin.shape}")
print(f"PT Q before RoPE [0,0,0,:10]: {pt_q[0,0,0,:10]}")
print(f"PT Q after RoPE [0,0,0,:10]: {pt_q_rot[0,0,0,:10]}")
print(f"PT Q before RoPE [0,0,1,:10]: {pt_q[0,0,1,:10]}")
print(f"PT Q after RoPE [0,0,1,:10]: {pt_q_rot[0,0,1,:10]}")
print(f"PT K before RoPE [0,0,0,:10]: {pt_k[0,0,0,:10]}")
print(f"PT K after RoPE [0,0,0,:10]: {pt_k_rot[0,0,0,:10]}")

# JAX RoPE
from src.mmbjax.modules.rope import get_rotary_cache, apply_rotary_emb as jax_apply_rotary
cache = get_rotary_cache(head_dim, config.global_rope_theta, dtype=jnp.float32)
jax_cos, jax_sin = cache.get(seq_len)
jax_q_rot, jax_k_rot = jax_apply_rotary(jax_q, jax_k, jax_cos, jax_sin, position_ids=None)

print(f"JAX cos shape: {jax_cos.shape}")
print(f"JAX sin shape: {jax_sin.shape}")
print(f"JAX Q before RoPE [0,0,0,:10]: {jax_q[0,0,0,:10]}")
print(f"JAX Q after RoPE [0,0,0,:10]: {jax_q_rot[0,0,0,:10]}")
print(f"JAX Q before RoPE [0,0,1,:10]: {jax_q[0,0,1,:10]}")
print(f"JAX Q after RoPE [0,0,1,:10]: {jax_q_rot[0,0,1,:10]}")
print(f"JAX K before RoPE [0,0,0,:10]: {jax_k[0,0,0,:10]}")
print(f"JAX K after RoPE [0,0,0,:10]: {jax_k_rot[0,0,0,:10]}")

q_rot_diff = np.abs(np.array(jax_q_rot) - pt_q_rot.numpy()).max()
print(f"\nQ after RoPE diff: {q_rot_diff:.2e} {'✅' if q_rot_diff < 1e-4 else '❌'}")

if q_rot_diff > 1e-4:
    print("\n⚠️  RoPE application differs!")
    sys.exit(1)

print("\n" + "="*80)
print("STEP 4: ATTENTION SCORES")
print("="*80)

# PyTorch attention scores
with torch.no_grad():
    scale = 1.0 / np.sqrt(head_dim)
    pt_scores = torch.einsum("bhqd,bhkd->bhqk", pt_q_rot, pt_k_rot) * scale

print(f"PyTorch scores shape: {pt_scores.shape}")
print(f"PT scores [0,0,0,:5]: {pt_scores[0,0,0,:5]}")
print(f"PT scores [0,0,0,-5:]: {pt_scores[0,0,0,-5:]}")

# JAX attention scores
scale = 1.0 / np.sqrt(head_dim)
jax_scores = jnp.einsum("bhqd,bhkd->bhqk", jax_q_rot, jax_k_rot) * scale

print(f"JAX scores shape: {jax_scores.shape}")
print(f"JAX scores [0,0,0,:5]: {jax_scores[0,0,0,:5]}")
print(f"JAX scores [0,0,0,-5:]: {jax_scores[0,0,0,-5:]}")

scores_diff = np.abs(np.array(jax_scores) - pt_scores.numpy()).max()
print(f"\nScores (before mask) diff: {scores_diff:.2e} {'✅' if scores_diff < 1e-4 else '❌'}")

print("\n" + "="*80)
print("STEP 5: APPLY ATTENTION MASK")
print("="*80)

# PyTorch mask
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask
pt_mask_4d = _prepare_4d_attention_mask(
    torch.from_numpy(attention_mask),
    dtype=pt_scores.dtype,
    tgt_len=seq_len
)
print(f"PyTorch mask shape: {pt_mask_4d.shape}")
print(f"PT mask [0,0,0,-5:]: {pt_mask_4d[0,0,0,-5:]}")

with torch.no_grad():
    pt_scores_masked = pt_scores + pt_mask_4d

print(f"PT scores (masked) [0,0,0,-5:]: {pt_scores_masked[0,0,0,-5:]}")

# JAX mask
mask_value = jnp.finfo(jnp.float32).min
jax_mask_2d = jnp.array(attention_mask, dtype=jnp.float32)
jax_padding = (1.0 - jax_mask_2d) * mask_value
jax_padding_4d = jax_padding[:, None, None, :]

print(f"JAX mask shape: {jax_padding_4d.shape}")
print(f"JAX mask [0,0,0,-5:]: {jax_padding_4d[0,0,0,-5:]}")

jax_scores_masked = jax_scores + jax_padding_4d
print(f"JAX scores (masked) [0,0,0,-5:]: {jax_scores_masked[0,0,0,-5:]}")

masked_scores_diff = np.abs(np.array(jax_scores_masked) - pt_scores_masked.numpy()).max()
print(f"\nMasked scores diff: {masked_scores_diff:.2e} {'✅' if masked_scores_diff < 1e-3 else '❌'}")

print("\n" + "="*80)
print("STEP 6: SOFTMAX")
print("="*80)

with torch.no_grad():
    pt_probs = torch.nn.functional.softmax(pt_scores_masked, dim=-1)

print(f"PT probs [0,0,0,:5]: {pt_probs[0,0,0,:5]}")
print(f"PT probs [0,0,0,-5:]: {pt_probs[0,0,0,-5:]}")
print(f"PT probs sum [0,0,0]: {pt_probs[0,0,0].sum()}")

jax_probs = jax.nn.softmax(jax_scores_masked, axis=-1)

print(f"JAX probs [0,0,0,:5]: {jax_probs[0,0,0,:5]}")
print(f"JAX probs [0,0,0,-5:]: {jax_probs[0,0,0,-5:]}")
print(f"JAX probs sum [0,0,0]: {jax_probs[0,0,0].sum()}")

probs_diff = np.abs(np.array(jax_probs) - pt_probs.numpy()).max()
print(f"\nProbs diff: {probs_diff:.2e} {'✅' if probs_diff < 1e-5 else '❌'}")
