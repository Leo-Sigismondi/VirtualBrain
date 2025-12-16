import numpy as np
import os

print("=== LATENT NORMALIZATION STATS ===")

cond_path = 'checkpoints/gru/latent_norm_stats_conditional.npy'
multi_path = 'checkpoints/gru/latent_norm_stats_multistep.npy'

if os.path.exists(cond_path):
    cond = np.load(cond_path, allow_pickle=True).item()
    print(f"Conditional: mean={cond['mean']:.6f}, std={cond['std']:.6f}")
else:
    print(f"Conditional: NOT FOUND")

if os.path.exists(multi_path):
    multi = np.load(multi_path, allow_pickle=True).item()
    print(f"Multistep:   mean={multi['mean']:.6f}, std={multi['std']:.6f}")
else:
    print(f"Multistep: NOT FOUND")

print("\n=== EXPLANATION ===")
print("If both have similar stats, then the loss scale difference is not from normalization.")
print("The issue might be in how the training loop computes/reports loss.")
