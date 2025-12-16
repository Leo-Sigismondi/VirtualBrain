"""
Quick evaluation of base GRU to compare with conditional GRU.
"""
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
import numpy as np
from src.models.gru import TemporalGRU
from src.models.vae import ImprovedVAE
from src.data.data_utils import load_normalized_dataset, encode_to_latent, INPUT_DIM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LATENT_DIM = 32
HIDDEN_DIM = 256
NUM_LAYERS = 3

VAE_PATH = "checkpoints/vae/vae_dynamics_latent32_best.pth"
GRU_PATH = "checkpoints/gru/gru_multistep_L32_H256_3L_best.pth"
LATENT_STATS_PATH = "checkpoints/gru/latent_norm_stats_multistep.npy"

print("=" * 60)
print("BASE GRU EVALUATION")
print("=" * 60)

# Load VAE
print("\n1. Loading VAE...")
vae = ImprovedVAE(input_dim=INPUT_DIM, latent_dim=LATENT_DIM, hidden_dims=[256, 128, 64]).to(DEVICE)
vae.load_state_dict(torch.load(VAE_PATH, map_location=DEVICE))
vae.eval()

# Load GRU
print("2. Loading GRU...")
gru = TemporalGRU(latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS).to(DEVICE)
gru.load_state_dict(torch.load(GRU_PATH, map_location=DEVICE))
gru.eval()

# Load latent stats
latent_stats = np.load(LATENT_STATS_PATH, allow_pickle=True).item()
print(f"   Latent stats: mean={latent_stats['mean']:.4f}, std={latent_stats['std']:.4f}")

# Load data
print("\n3. Loading data...")
normalized_data, _ = load_normalized_dataset()
print(f"   Data shape: {normalized_data.shape}")

# Encode to latent
print("\n4. Encoding to latent...")
latent_sequences = encode_to_latent(vae, normalized_data[:5000], DEVICE)  # Subset
latent_sequences = latent_sequences.numpy()

# Normalize
latent_norm = (latent_sequences - latent_stats['mean']) / (latent_stats['std'] + 1e-8)

# Pick some samples and generate
print("\n5. Generating trajectories...")
SEED_STEPS = 8
n_samples = 100

real_samples = latent_norm[:n_samples]
seed = torch.from_numpy(real_samples[:, :SEED_STEPS, :]).float().to(DEVICE)

# Generate with base GRU (no class labels)
gru.eval()
with torch.no_grad():
    # Build hidden state
    hidden = None
    for t in range(SEED_STEPS):
        current = seed[:, t:t+1, :]
        _, hidden = gru.predict_next(current, hidden)
    
    # Generate autoregressively
    predictions = []
    current = seed[:, -1:, :]
    for _ in range(64 - SEED_STEPS):
        next_state, hidden = gru.predict_next(current, hidden)
        predictions.append(next_state)
        current = next_state
    
    generated = torch.cat(predictions, dim=1).cpu().numpy()

# Combine seed + generated
full_generated = np.concatenate([real_samples[:, :SEED_STEPS, :], generated], axis=1)

# Compute metrics
print("\n6. Computing metrics...")

# Velocity ratio
real_vel = np.diff(real_samples, axis=1)
gen_vel = np.diff(full_generated, axis=1)
real_vel_std = np.std(real_vel)
gen_vel_std = np.std(gen_vel)
vel_ratio = gen_vel_std / (real_vel_std + 1e-8)

# MSE between generated and real (post-seed)
real_post_seed = real_samples[:, SEED_STEPS:, :]
gen_post_seed = generated
mse = np.mean((gen_post_seed - real_post_seed) ** 2)

print(f"\n=== BASE GRU RESULTS ===")
print(f"MSE (post-seed): {mse:.6f}")
print(f"Velocity Ratio: {vel_ratio:.2f} (1.0 = ideal)")
print(f"Real velocity std: {real_vel_std:.4f}")
print(f"Generated velocity std: {gen_vel_std:.4f}")

# Compare with conditional
print(f"\n=== COMPARISON ===")
print(f"BASE GRU:        MSE={mse:.6f}, VelRatio={vel_ratio:.2f}")
print("(Run evaluate_conditional_gru.py to compare with conditional)")
