"""Debug MLP (GeGLU) implementation."""
import os
os.environ["JAX_PLATFORMS"] = "cpu"

import sys
sys.path.insert(0, ".")

import numpy as np
import torch
import jax.numpy as jnp
from transformers import AutoModel

from src.mmbjax.modules.mlp import ModernBertMLP
from src.mmbjax.utils.checkpoints import convert_pytorch_state_dict_to_flax, load_hf_config_as_mmbert

# Simple test input
batch, seq, hidden = 2, 16, 384
np.random.seed(42)
test_input = np.random.randn(batch, seq, hidden).astype(np.float32)

print("="*80)
print("MLP (GeGLU) DEBUG")
print("="*80)

# Load models
pt_model = AutoModel.from_pretrained("jhu-clsp/mmBERT-small")
pt_model.eval()

config = load_hf_config_as_mmbert("artifacts/reference-configs/jhu-clsp_mmBERT-small_config.json")

# Convert weights
pt_state_dict = {f"model.{k}" if not k.startswith("model.") else k: v 
                 for k, v in pt_model.state_dict().items()}
jax_params_wrapped = convert_pytorch_state_dict_to_flax(pt_state_dict, config)
jax_params = jax_params_wrapped["params"]["model"]

# Get layer 0 MLP
pt_mlp = pt_model.layers[0].mlp
jax_mlp = ModernBertMLP(config=config, dtype=jnp.float32)

# Initialize JAX MLP
import jax
rng = jax.random.PRNGKey(0)
jax_mlp_vars = jax_mlp.init(rng, jnp.ones((1, 1, hidden)), deterministic=True)
# Replace with converted weights
jax_mlp_vars = {"params": jax_params["encoder"]["layers_0"]["mlp"]}

print("\nStep 1: Wi projection (gate + value)")
print("-" * 80)

# PyTorch
with torch.no_grad():
    pt_input = torch.from_numpy(test_input)
    pt_wi_out = pt_mlp.Wi(pt_input)  # [batch, seq, 2*intermediate]
    
print(f"PT Wi output shape: {pt_wi_out.shape}")
print(f"PT Wi sample [0,0,:5]: {pt_wi_out[0,0,:5]}")

# JAX
jax_input = jnp.array(test_input)
jax_wi_out = jax_mlp.apply(jax_mlp_vars, jax_input, deterministic=True, method=lambda self, x, **kw: self.wi(x))

print(f"JAX Wi output shape: {jax_wi_out.shape}")
print(f"JAX Wi sample [0,0,:5]: {jax_wi_out[0,0,:5]}")

wi_diff = np.abs(np.array(jax_wi_out) - pt_wi_out.numpy()).max()
print(f"\nWi diff: {wi_diff:.2e} {'✅' if wi_diff < 1e-4 else '❌'}")

if wi_diff > 1e-4:
    print("\n⚠️  Wi projection differs!")
    sys.exit(1)

print("\nStep 2: Split and apply activation (GeGLU)")
print("-" * 80)

# PyTorch
intermediate = config.intermediate_size
with torch.no_grad():
    pt_gate = pt_wi_out[..., :intermediate]
    pt_value = pt_wi_out[..., intermediate:]
    
    print(f"PT gate shape: {pt_gate.shape}")
    print(f"PT value shape: {pt_value.shape}")
    print(f"PT gate sample [0,0,:5]: {pt_gate[0,0,:5]}")
    print(f"PT value sample [0,0,:5]: {pt_value[0,0,:5]}")
    
    # Apply GELU to gate
    pt_gate_act = torch.nn.functional.gelu(pt_gate, approximate="tanh")
    print(f"PT gate after GELU [0,0,:5]: {pt_gate_act[0,0,:5]}")
    
    # Element-wise multiply
    pt_hidden = pt_gate_act * pt_value
    print(f"PT hidden after multiply [0,0,:5]: {pt_hidden[0,0,:5]}")

# JAX
jax_gate = jax_wi_out[..., :intermediate]
jax_value = jax_wi_out[..., intermediate:]

print(f"\nJAX gate shape: {jax_gate.shape}")
print(f"JAX value shape: {jax_value.shape}")
print(f"JAX gate sample [0,0,:5]: {jax_gate[0,0,:5]}")
print(f"JAX value sample [0,0,:5]: {jax_value[0,0,:5]}")

# Apply activation (should be gelu_new which is tanh approximation)
from src.mmbjax.utils.activations import ACT2FN
act_fn = ACT2FN[config.hidden_activation]
jax_gate_act = act_fn(jax_gate)
print(f"JAX gate after activation [0,0,:5]: {jax_gate_act[0,0,:5]}")

jax_hidden = jax_gate_act * jax_value
print(f"JAX hidden after multiply [0,0,:5]: {jax_hidden[0,0,:5]}")

hidden_diff = np.abs(np.array(jax_hidden) - pt_hidden.numpy()).max()
print(f"\nHidden (after GeGLU) diff: {hidden_diff:.2e} {'✅' if hidden_diff < 1e-4 else '❌'}")

if hidden_diff > 1e-4:
    print("\n⚠️  GeGLU differs!")
    print(f"Config activation: {config.hidden_activation}")
    sys.exit(1)

print("\nStep 3: Wo projection")
print("-" * 80)

# PyTorch
with torch.no_grad():
    pt_output = pt_mlp.Wo(pt_hidden)

print(f"PT output shape: {pt_output.shape}")
print(f"PT output sample [0,0,:5]: {pt_output[0,0,:5]}")

# JAX
jax_output = jax_mlp.apply(jax_mlp_vars, jax_input, deterministic=True)

print(f"JAX output shape: {jax_output.shape}")
print(f"JAX output sample [0,0,:5]: {jax_output[0,0,:5]}")

output_diff = np.abs(np.array(jax_output) - pt_output.numpy()).max()
print(f"\nFinal MLP output diff: {output_diff:.2e} {'✅' if output_diff < 1e-4 else '❌'}")

if output_diff < 1e-4:
    print("\n🎉 MLP matches perfectly!")
else:
    print("\n❌ MLP differs!")
