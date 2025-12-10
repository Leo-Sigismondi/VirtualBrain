"""Debug why full eval shows negative eigenvalues"""
import sys
sys.path.insert(0, '.')
import torch
import numpy as np
from src.models.diffusion import TangentDiffusion
from src.preprocessing.geometry_utils import exp_euclidean_map, validate_spd, vec_to_sym_matrix

# Load
stats = np.load('data/processed/normalization_stats.npy', allow_pickle=True).item()
m = TangentDiffusion(253, 512, 128, 1000).cuda()
m.load_state_dict(torch.load('checkpoints/diffusion/tangent_diffusion_best.pth'))
m.eval()

# Generate samples with SAME shape as evaluation
with torch.no_grad():
    samples = m.sample_ddim((100, 64, 253), device='cuda', steps=50)  # 100 samples x 64 timesteps

# Denormalize
samples = samples * stats['std'] + stats['mean']
samples = samples.cpu()
print(f"Sample range: min={samples.min().item():.2f}, max={samples.max().item():.2f}")

# Check ALL samples
failures = []
total_checked = 0

for i in range(100):
    for t in range(64):
        vec = samples[i, t, :]
        matrix = vec_to_sym_matrix(vec, 22)
        spd = exp_euclidean_map(matrix.unsqueeze(0))[0]
        eigvals = torch.linalg.eigvalsh(spd)
        
        min_eig = eigvals.min().item()
        total_checked += 1
        
        if min_eig <= 0:
            failures.append((i, t, min_eig, vec.min().item(), vec.max().item()))

print(f"\nFailures: {len(failures)}/{total_checked} = {100*len(failures)/total_checked:.2f}%")
print(f"Success: {total_checked - len(failures)}/{total_checked} = {100*(total_checked - len(failures))/total_checked:.2f}%")

if failures:
    print(f"\nFirst 5 failures:")
    for f in failures[:5]:
        print(f"  Sample {f[0]}, time {f[1]}: min_eigval={f[2]:.2e}, vec range=[{f[3]:.2f}, {f[4]:.2f}]")
