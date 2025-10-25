"""Debug RoPE implementation against PyTorch."""
import os
os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np
import torch
import jax.numpy as jnp

from src.mmbjax.modules.rope import build_rotary_cache, apply_rotary_emb

def test_rope_cache():
    """Compare RoPE cache generation."""
    print("=" * 80)
    print("Testing RoPE cache generation")
    print("=" * 80)
    
    max_seq = 128
    dim = 64
    theta = 160000.0
    
    # JAX version
    jax_cos, jax_sin = build_rotary_cache(max_seq, dim, theta, jnp.float32)
    
    # PyTorch version (manual calculation - matching actual ModernBERT implementation)
    positions = torch.arange(max_seq, dtype=torch.float32)
    idx = torch.arange(0, dim, 2, dtype=torch.float32)
    inv_freq = 1.0 / (theta ** (idx / dim))
    angles = positions[:, None] * inv_freq[None, :]
    # Stack (concatenate) to match ModernBERT implementation
    emb = torch.cat([angles, angles], dim=-1)
    pt_cos = torch.cos(emb)
    pt_sin = torch.sin(emb)
    
    # Compare
    cos_diff = np.abs(np.array(jax_cos) - pt_cos.numpy()).max()
    sin_diff = np.abs(np.array(jax_sin) - pt_sin.numpy()).max()
    
    print(f"JAX cos shape: {jax_cos.shape}, PyTorch cos shape: {pt_cos.shape}")
    print(f"JAX sin shape: {jax_sin.shape}, PyTorch sin shape: {pt_sin.shape}")
    print(f"Cos max diff: {cos_diff:.2e}")
    print(f"Sin max diff: {sin_diff:.2e}")
    
    if cos_diff < 1e-6 and sin_diff < 1e-6:
        print("✅ RoPE cache matches!")
    else:
        print("❌ RoPE cache MISMATCH!")
        print(f"First 5 cos values (JAX): {jax_cos[0, :5]}")
        print(f"First 5 cos values (PT):  {pt_cos[0, :5].numpy()}")
    
    return cos_diff < 1e-6 and sin_diff < 1e-6


def test_rotate_half():
    """Compare rotate_half implementation."""
    print("\n" + "=" * 80)
    print("Testing rotate_half")
    print("=" * 80)
    
    from src.mmbjax.modules.rope import rotate_half as jax_rotate_half
    
    # Test input
    x = np.random.randn(2, 4, 8, 64).astype(np.float32)
    x_jax = jnp.array(x)
    x_pt = torch.from_numpy(x)
    
    # JAX version
    jax_out = jax_rotate_half(x_jax)
    
    # PyTorch version
    x1 = x_pt[..., : x_pt.shape[-1] // 2]
    x2 = x_pt[..., x_pt.shape[-1] // 2 :]
    pt_out = torch.cat((-x2, x1), dim=-1)
    
    # Compare
    diff = np.abs(np.array(jax_out) - pt_out.numpy()).max()
    print(f"Max diff: {diff:.2e}")
    
    if diff < 1e-7:
        print("✅ rotate_half matches!")
    else:
        print("❌ rotate_half MISMATCH!")
    
    return diff < 1e-7


def test_apply_rotary():
    """Compare full rotary embedding application."""
    print("\n" + "=" * 80)
    print("Testing apply_rotary_emb")
    print("=" * 80)
    
    batch, heads, seq_len, head_dim = 2, 6, 128, 64
    theta = 160000.0
    
    # Create test Q, K
    q = np.random.randn(batch, heads, seq_len, head_dim).astype(np.float32)
    k = np.random.randn(batch, heads, seq_len, head_dim).astype(np.float32)
    
    # Get RoPE cache
    jax_cos, jax_sin = build_rotary_cache(seq_len, head_dim, theta, jnp.float32)
    pt_positions = torch.arange(seq_len, dtype=torch.float32)
    pt_idx = torch.arange(0, head_dim, 2, dtype=torch.float32)
    pt_inv_freq = 1.0 / (theta ** (pt_idx / head_dim))
    pt_angles = pt_positions[:, None] * pt_inv_freq[None, :]
    # Stack to match ModernBERT
    pt_emb = torch.cat([pt_angles, pt_angles], dim=-1)
    pt_cos = torch.cos(pt_emb)
    pt_sin = torch.sin(pt_emb)
    
    # JAX application
    q_jax = jnp.array(q)
    k_jax = jnp.array(k)
    q_rot_jax, k_rot_jax = apply_rotary_emb(q_jax, k_jax, jax_cos, jax_sin, position_ids=None)
    
    # PyTorch application (manual)
    q_pt = torch.from_numpy(q)
    k_pt = torch.from_numpy(k)
    # Broadcast cos/sin: [seq, dim] -> [1, 1, seq, dim]
    cos_expanded = pt_cos[None, None, :, :]
    sin_expanded = pt_sin[None, None, :, :]
    
    # Rotate half helper
    def pt_rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    
    q_rot_pt = (q_pt * cos_expanded) + (pt_rotate_half(q_pt) * sin_expanded)
    k_rot_pt = (k_pt * cos_expanded) + (pt_rotate_half(k_pt) * sin_expanded)
    
    # Compare
    q_diff = np.abs(np.array(q_rot_jax) - q_rot_pt.numpy()).max()
    k_diff = np.abs(np.array(k_rot_jax) - k_rot_pt.numpy()).max()
    
    print(f"Q max diff: {q_diff:.2e}")
    print(f"K max diff: {k_diff:.2e}")
    
    if q_diff < 1e-6 and k_diff < 1e-6:
        print("✅ apply_rotary_emb matches!")
    else:
        print("❌ apply_rotary_emb MISMATCH!")
        print(f"Q sample (JAX): {np.array(q_rot_jax)[0, 0, 0, :5]}")
        print(f"Q sample (PT):  {q_rot_pt[0, 0, 0, :5].numpy()}")
    
    return q_diff < 1e-6 and k_diff < 1e-6


if __name__ == "__main__":
    results = []
    results.append(("RoPE cache", test_rope_cache()))
    results.append(("rotate_half", test_rotate_half()))
    results.append(("apply_rotary", test_apply_rotary()))
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for name, passed in results:
        status = "✅" if passed else "❌"
        print(f"{status} {name}")
    
    all_passed = all(r[1] for r in results)
    if all_passed:
        print("\n🎉 All RoPE tests passed!")
    else:
        print("\n⚠️  Some RoPE tests failed!")
