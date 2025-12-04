"""
Diagnostic script to check if GRU is truly learning or just doing lazy prediction
"""
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
import numpy as np
from src.data.dataset import BCIDataset
from torch.utils.data import DataLoader
from src.models.vae import VAE
from src.models.gru import TemporalGRU

VAE_PATH = "checkpoints/vae/vae_latent32_best.pth"
GRU_PATH = "checkpoints/gru/gru_L32_H128_Lay1_best.pth"
LATENT_DIM = 32
HIDDEN_DIM = 128
INPUT_DIM = 325
NUM_LAYERS = 1
SKIP_STEPS = 4

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load models
vae = VAE(input_dim=INPUT_DIM, latent_dim=LATENT_DIM).to(device)
vae.load_state_dict(torch.load(VAE_PATH, map_location=device))
vae.eval()

gru = TemporalGRU(latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS).to(device)
gru.load_state_dict(torch.load(GRU_PATH, map_location=device))
gru.eval()

# Load data
dataset = BCIDataset("data/processed/train")
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
seq_vectors, _ = next(iter(dataloader))
seq_vectors = seq_vectors[0].to(device)

# Load VAE normalization
vae_norm_stats = np.load("checkpoints/vae/vae_norm_stats_latent32.npy", allow_pickle=True).item()
vae_mean = torch.tensor(vae_norm_stats['mean']).to(device)
vae_std = torch.tensor(vae_norm_stats['std']).to(device)

# Encode
with torch.no_grad():
    latent_seq = []
    for t in range(seq_vectors.shape[0]):
        frame = seq_vectors[t:t+1]
        frame = (frame - vae_mean) / vae_std
        mu, _ = vae.encode(frame)
        latent_seq.append(mu)
    latent_seq = torch.cat(latent_seq, dim=0).unsqueeze(0)

# Load GRU normalization
stats = np.load("checkpoints/gru/latent_norm_stats.npy", allow_pickle=True).item()
latent_mean = torch.tensor(stats['mean']).to(device)
latent_std = torch.tensor(stats['std']).to(device)
latent_seq_norm = (latent_seq - latent_mean) / (latent_std + 1e-8)

# Make predictions
input_seq = latent_seq_norm[:, :-SKIP_STEPS, :]
with torch.no_grad():
    pred_states, _ = gru(input_seq)  # Direct state prediction, not delta

# Convert to numpy
input_np = input_seq[0].cpu().numpy()
pred_np = pred_states[0].cpu().numpy()  # These are predicted states
ground_truth_np = latent_seq_norm[0].cpu().numpy()

# Calculate deltas for analysis
delta_np = pred_np - input_np  # Derived delta from state predictions

print("="*60)
print("LAZY PREDICTOR DIAGNOSTIC")
print("="*60)

print(f"\nSequence length: {ground_truth_np.shape[0]}")
print(f"Input shape: {input_np.shape}")
print(f"Predictions for timesteps: {SKIP_STEPS} to {ground_truth_np.shape[0]-1}")

# Check delta magnitudes
print(f"\n--- Delta Statistics ---")
print(f"Mean absolute delta predicted: {np.abs(delta_np).mean():.6f}")
print(f"Max absolute delta: {np.abs(delta_np).max():.6f}")
print(f"Std of delta: {np.std(delta_np):.6f}")

# Check actual ground truth changes
actual_changes = ground_truth_np[SKIP_STEPS:] - input_np
print(f"\n--- Actual Changes (Ground Truth) ---")
print(f"Mean absolute actual change: {np.abs(actual_changes).mean():.6f}")
print(f"Max absolute actual change: {np.abs(actual_changes).max():.6f}")
print(f"Std of actual change: {np.std(actual_changes):.6f}")

# Check if predictions are close to zero (lazy predictor)
zero_threshold = 0.01
pct_near_zero = (np.abs(delta_np) < zero_threshold).mean() * 100
print(f"\n--- Lazy Predictor Check ---")
print(f"Percentage of predictions < {zero_threshold}: {pct_near_zero:.1f}%")

if pct_near_zero > 80:
    print("⚠️  WARNING: Model is predicting mostly zeros (LAZY PREDICTOR)")
else:
    print("✓ Model is predicting non-zero changes")

# Compare prediction accuracy: GRU vs just copying input
gru_error = np.abs(pred_np - ground_truth_np[SKIP_STEPS:]).mean()
copy_error = np.abs(input_np - ground_truth_np[SKIP_STEPS:]).mean()

print(f"\n--- Error Comparison ---")
print(f"GRU MAE: {gru_error:.6f}")
print(f"Copy (lazy) MAE: {copy_error:.6f}")
print(f"Improvement: {((copy_error - gru_error) / copy_error * 100):.2f}%")

if gru_error >= copy_error * 0.95:
    print("\n❌ FAILED: GRU is not better than lazy prediction!")
else:
    print("\n✓ SUCCESS: GRU beats lazy prediction")

# Visual check: print first few timesteps
print(f"\n--- Sample Predictions (first 3 timesteps) ---")
for t in range(min(3, input_np.shape[0])):
    print(f"\nTimestep {t} -> {t+SKIP_STEPS}:")
    print(f"  Input[0]: {input_np[t, 0]:.4f}")
    print(f"  Delta[0]: {delta_np[t, 0]:.4f}")
    print(f"  Predicted[0]: {pred_np[t, 0]:.4f}")
    print(f"  Ground Truth[0]: {ground_truth_np[t+SKIP_STEPS, 0]:.4f}")
    print(f"  Error: {abs(pred_np[t, 0] - ground_truth_np[t+SKIP_STEPS, 0]):.4f}")
