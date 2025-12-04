"""
Diagnose why GRU is acting as lazy predictor
Analyzes latent space temporal dynamics
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
import numpy as np
import matplotlib.pyplot as plt
from src.data.dataset import BCIDataset
from torch.utils.data import DataLoader
from src.models.vae import VAE

# Config
VAE_PATH = "checkpoints/vae/vae_latent32_best.pth"
LATENT_DIM = 32
INPUT_DIM = 325

device = "cuda" if torch.cuda.is_available() else "cpu"

print("="*60)
print("Latent Space Temporal Dynamics Diagnosis")
print("="*60 + "\n")

# Load VAE
print("Loading VAE...")
vae = VAE(input_dim=INPUT_DIM, latent_dim=LATENT_DIM).to(device)
vae.load_state_dict(torch.load(VAE_PATH, map_location=device))
vae.eval()

# Load data
print("Loading dataset...")
dataset = BCIDataset("data/processed/train")
dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

# Load normalization stats
vae_norm_stats = np.load("checkpoints/vae/vae_norm_stats_latent32.npy", allow_pickle=True).item()
vae_mean = torch.tensor(vae_norm_stats['mean']).to(device)
vae_std = torch.tensor(vae_norm_stats['std']).to(device)

# Encode all sequences
print("Encoding to latent space...")
all_latents = []
with torch.no_grad():
    for batch_seq, _ in dataloader:
        batch_seq = batch_seq.to(device)
        batch_size, seq_len, feat_dim = batch_seq.shape
        
        # Flatten
        x = batch_seq.view(-1, feat_dim)
        
        # Normalize
        x = (x - vae_mean) / vae_std
        
        # Encode
        mu, _ = vae.encode(x)
        
        # Reshape
        latent_seq = mu.view(batch_size, seq_len, -1)
        all_latents.append(latent_seq.cpu())

latent_sequences = torch.cat(all_latents, dim=0)
print(f"Latent sequences shape: {latent_sequences.shape}\n")

# Normalize latent space
latent_mean = latent_sequences.mean()
latent_std = latent_sequences.std()
latent_normalized = (latent_sequences - latent_mean) / (latent_std + 1e-8)

print("="*60)
print("TEMPORAL DYNAMICS ANALYSIS")
print("="*60 + "\n")

# 1. Compute temporal derivatives (velocity)
velocities = latent_normalized[:, 1:, :] - latent_normalized[:, :-1, :]
velocity_magnitude = torch.norm(velocities, dim=2)  # (N, T-1)

print("1. TEMPORAL VELOCITY")
print(f"   Mean velocity magnitude: {velocity_magnitude.mean():.6f}")
print(f"   Std velocity magnitude: {velocity_magnitude.std():.6f}")
print(f"   Max velocity: {velocity_magnitude.max():.6f}")
print(f"   Min velocity: {velocity_magnitude.min():.6f}\n")

# If velocity is very low, it means states barely change -> persistence is optimal!
if velocity_magnitude.mean() < 0.1:
    print("   ⚠️  WARNING: Very low temporal velocity!")
    print("   → States change very little over time")
    print("   → Predicting z[t+1] = z[t] is nearly optimal\n")

# 2. Autocorrelation analysis
print("2. TEMPORAL AUTOCORRELATION")
# Compute autocorrelation for lag 1
states_t = latent_normalized[:, :-1, :]
states_t1 = latent_normalized[:, 1:, :]

# Correlation coefficient
correlation = torch.cosine_similarity(
    states_t.reshape(-1, LATENT_DIM),
    states_t1.reshape(-1, LATENT_DIM),
    dim=1
).mean()

print(f"   Lag-1 autocorrelation: {correlation:.6f}")

if correlation > 0.95:
    print("   ⚠️  WARNING: Very high autocorrelation!")
    print("   → Consecutive states are almost identical")
    print("   → Temporal prediction is extremely hard\n")
else:
    print("   ✓ Reasonable autocorrelation\n")

# 3. Persistence baseline error
print("3. PERSISTENCE BASELINE QUALITY")
# How good is z[t+1] = z[t]?
persistence_error = torch.abs(states_t - states_t1).mean()
print(f"   Persistence MAE: {persistence_error:.6f}")

if persistence_error < 0.5:
    print("   ⚠️  WARNING: Persistence is very accurate!")
    print("   → The latent space is nearly static")
    print("   → Models will struggle to beat persistence\n")
else:
    print("   ✓ Persistence makes significant errors\n")

# 4. Predictability measure
# Compare: variance within sequence vs variance between sequences
within_seq_var = []
for seq in latent_normalized:
    within_seq_var.append(seq.var(dim=0).mean())
within_seq_var = torch.tensor(within_seq_var).mean()

between_seq_var = latent_normalized.mean(dim=1).var(dim=0).mean()

print("4. TEMPORAL vs STATIC VARIANCE")
print(f"   Within-sequence variance: {within_seq_var:.6f}")
print(f"   Between-sequence variance: {between_seq_var:.6f}")
print(f"   Ratio (within/between): {(within_seq_var / (between_seq_var + 1e-8)):.6f}")

if within_seq_var < between_seq_var:
    print("   ⚠️  WARNING: More variance BETWEEN than WITHIN sequences!")
    print("   → Sequences are mostly static over time")
    print("   → Most information is in initial state, not dynamics\n")
else:
    print("   ✓ Good temporal dynamics\n")

# 5. Visualization
print("Creating diagnostic visualizations...\n")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Plot 1: Sample trajectories
ax = axes[0, 0]
for i in range(min(10, len(latent_normalized))):
    seq = latent_normalized[i, :, :3]  # First 3 dims
    for dim in range(3):
        ax.plot(seq[:, dim].numpy(), alpha=0.5)
ax.set_xlabel('Timestep')
ax.set_ylabel('Latent Value')
ax.set_title('Sample Trajectories (First 3 Dims)')
ax.grid(True, alpha=0.3)

# Plot 2: Velocity distribution
ax = axes[0, 1]
ax.hist(velocity_magnitude.flatten().numpy(), bins=50, alpha=0.7, edgecolor='black')
ax.set_xlabel('Velocity Magnitude')
ax.set_ylabel('Frequency')
ax.set_title(f'Temporal Velocity Distribution\n(Mean: {velocity_magnitude.mean():.4f})')
ax.grid(True, alpha=0.3, axis='y')

# Plot 3: Persistence error distribution
ax = axes[0, 2]
persistence_errors = torch.abs(states_t - states_t1).mean(dim=2).flatten()
ax.hist(persistence_errors.numpy(), bins=50, alpha=0.7, color='coral', edgecolor='black')
ax.set_xlabel('Persistence Error')
ax.set_ylabel('Frequency')
ax.set_title(f'Persistence Baseline Errors\n(Mean: {persistence_error:.4f})')
ax.grid(True, alpha=0.3, axis='y')

# Plot 4: Autocorrelation by lag
ax = axes[1, 0]
lags = range(1, min(13, latent_normalized.shape[1]))
autocorrs = []
for lag in lags:
    states_t = latent_normalized[:, :-lag, :]
    states_tlag = latent_normalized[:, lag:, :]
    corr = torch.cosine_similarity(
        states_t.reshape(-1, LATENT_DIM),
        states_tlag.reshape(-1, LATENT_DIM),
        dim=1
    ).mean()
    autocorrs.append(corr.item())

ax.plot(lags, autocorrs, 'o-', linewidth=2, markersize=8)
ax.set_xlabel('Lag')
ax.set_ylabel('Autocorrelation')
ax.set_title('Temporal Autocorrelation by Lag')
ax.grid(True, alpha=0.3)
ax.axhline(y=0.95, color='red', linestyle='--', label='High threshold (0.95)')
ax.legend()

# Plot 5: Variance decomposition
ax = axes[1, 1]
dim_variances = []
for dim in range(LATENT_DIM):
    within_dim = []
    for seq in latent_normalized:
        within_dim.append(seq[:, dim].var().item())
    dim_variances.append(np.mean(within_dim))

ax.bar(range(LATENT_DIM), dim_variances, alpha=0.7, color='skyblue', edgecolor='black')
ax.set_xlabel('Latent Dimension')
ax.set_ylabel('Within-Sequence Variance')
ax.set_title('Temporal Variance by Dimension')
ax.grid(True, alpha=0.3, axis='y')

# Plot 6: Delta magnitudes over time
ax = axes[1, 2]
delta_per_timestep = velocity_magnitude.mean(dim=0).numpy()
timesteps = range(1, len(delta_per_timestep) + 1)
ax.plot(timesteps, delta_per_timestep, 'o-', linewidth=2, markersize=8, color='green')
ax.set_xlabel('Timestep')
ax.set_ylabel('Mean Velocity Magnitude')
ax.set_title('Temporal Change Across Sequence')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('checkpoints/gru/latent_dynamics_diagnosis.png', dpi=150, bbox_inches='tight')
print("Saved: checkpoints/gru/latent_dynamics_diagnosis.png")

print("\n" + "="*60)
print("DIAGNOSIS SUMMARY")
print("="*60 + "\n")

issues = []
if velocity_magnitude.mean() < 0.1:
    issues.append("Very low temporal velocity - states barely change")
if correlation > 0.95:
    issues.append("Very high autocorrelation - consecutive states nearly identical")
if persistence_error < 0.5:
    issues.append("Persistence baseline is very strong")
if within_seq_var < between_seq_var:
    issues.append("More variance between than within sequences - mostly static")

if issues:
    print("🔴 ISSUES FOUND:")
    for i, issue in enumerate(issues, 1):
        print(f"   {i}. {issue}")
    print("\n💡 LIKELY ROOT CAUSE:")
    print("   The VAE latent space encodes STATIC features, not TEMPORAL dynamics")
    print("   → Brain states change very little over 13 timesteps (3.25 seconds)")
    print("   → GRU has nothing to learn beyond 'copy the previous state'")
    print("\n🔧 POTENTIAL SOLUTIONS:")
    print("   1. Use VAE to encode DIFFERENCES/DELTAS instead of absolute states")
    print("   2. Add temporal loss to VAE training to encourage dynamic encoding") 
    print("   3. Use LONGER sequences to capture meaningful temporal changes")
    print("   4. Predict FUTURE states (t+k) instead of immediate next (t+1)")
    print("   5. Add explicit 'anti-lazy' penalty during GRU training")
else:
    print("✅ Latent space has good temporal dynamics")
    print("   The issue likely lies elsewhere (model architecture, training, etc.)")
