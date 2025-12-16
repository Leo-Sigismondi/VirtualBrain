"""
Trace through the evaluate_diffusion.py pipeline step-by-step
to find where values become invalid.
"""
import sys
sys.path.insert(0, '.')

import torch
import numpy as np
from src.data.data_utils import load_normalized_dataset, encode_to_latent, INPUT_DIM
from src.models.vae import ImprovedVAE
from src.models.gru import TemporalGRU

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LATENT_DIM = 32

print("=== STEP BY STEP PIPELINE DEBUG ===\n")

# Step 1: Load data
print("1. Loading normalized data...")
normalized_data, norm_stats = load_normalized_dataset()
print(f"   Data shape: {normalized_data.shape}")
print(f"   Norm stats: mean={norm_stats['mean']:.4f}, std={norm_stats['std']:.4f}")

# Step 2: Sample some data
np.random.seed(42)
indices = np.random.choice(len(normalized_data), 10, replace=False)
data = torch.from_numpy(normalized_data[indices].copy()).float().to(DEVICE)
print(f"\n2. Sample data:")
print(f"   Shape: {data.shape}")
print(f"   Min: {data.min():.4f}, Max: {data.max():.4f}, Mean: {data.mean():.4f}")

# Step 3: Load VAE
print("\n3. Loading VAE...")
vae = ImprovedVAE(input_dim=INPUT_DIM, latent_dim=LATENT_DIM, hidden_dims=[256, 128, 64]).to(DEVICE)
vae.load_state_dict(torch.load('checkpoints/vae/vae_dynamics_latent32.pth', map_location=DEVICE))
vae.eval()
print("   VAE loaded")

# Step 4: Encode to latent
print("\n4. Encoding to latent...")
with torch.no_grad():
    latent = encode_to_latent(vae, data, DEVICE)
print(f"   Latent shape: {latent.shape}")
print(f"   Latent min: {latent.min():.4f}, max: {latent.max():.4f}, mean: {latent.mean():.4f}")

# Step 5: Decode back
print("\n5. Decoding back to tangent space...")
with torch.no_grad():
    flat_latent = latent.view(-1, LATENT_DIM).to(DEVICE)
    decoded = vae.decode(flat_latent)
    decoded = decoded.view(10, 64, -1)
print(f"   Decoded shape: {decoded.shape}")
print(f"   Decoded min: {decoded.min():.4f}, max: {decoded.max():.4f}, mean: {decoded.mean():.4f}")

# Step 6: Denormalize
print("\n6. Denormalizing...")
denorm = decoded.cpu() * norm_stats['std'] + norm_stats['mean']
print(f"   Denorm min: {denorm.min():.4f}, max: {denorm.max():.4f}, mean: {denorm.mean():.4f}")

# Step 7: Convert to SPD
print("\n7. Converting to SPD matrix...")
from src.utils.geometry_utils import vec_to_sym_matrix, vector_to_spd
try:
    sample_vec = denorm[0, 0, :]  # First sample, first timestep
    print(f"   Vector shape: {sample_vec.shape}")
    print(f"   Vector min: {sample_vec.min():.4f}, max: {sample_vec.max():.4f}")
    
    # Convert to symmetric matrix
    sym = vec_to_sym_matrix(sample_vec.numpy())
    print(f"   Symmetric matrix shape: {sym.shape}")
    print(f"   Symmetric min: {sym.min():.4f}, max: {sym.max():.4f}")
    
    # Check eigenvalues
    eigvals = np.linalg.eigvalsh(sym)
    print(f"   Eigenvalues: min={eigvals.min():.4f}, max={eigvals.max():.4f}")
    
except Exception as e:
    print(f"   ERROR: {e}")

print("\n=== DONE ===")
