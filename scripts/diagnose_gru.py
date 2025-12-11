"""
Diagnose GRU Latent Prediction Problem

This script compares:
1. Real VAE-encoded latents (what diffusion was trained on)
2. GRU-predicted latents (what we want to use for generation)

Goal: Understand why GRU predictions don't match the real latent distribution.
"""

import sys
import os
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.vae import ImprovedVAE
from src.models.gru import TemporalGRU
from src.data.data_utils import load_normalized_dataset, encode_to_latent

# Config
VAE_PATH = "checkpoints/vae/vae_dynamics_latent32.pth"
GRU_PATH = "checkpoints/gru/gru_multistep_L32_H128_2L_best.pth"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    print("\n" + "="*60)
    print("GRU Latent Prediction Diagnosis")
    print("="*60)
    
    # Load models
    vae = ImprovedVAE(input_dim=253, latent_dim=32).to(DEVICE)
    vae.load_state_dict(torch.load(VAE_PATH, map_location=DEVICE))
    vae.eval()
    
    gru = TemporalGRU(latent_dim=32, hidden_dim=128, num_layers=2, dropout=0.2).to(DEVICE)
    gru.load_state_dict(torch.load(GRU_PATH, map_location=DEVICE))
    gru.eval()
    
    print("[OK] Models loaded")
    
    # Load data
    normalized_data, stats = load_normalized_dataset()
    np.random.seed(42)
    indices = np.random.choice(len(normalized_data), 100, replace=False)
    data = torch.from_numpy(normalized_data[indices].copy()).float().to(DEVICE)
    
    print(f"[OK] Loaded {len(data)} sequences")
    
    # Encode to latent space
    with torch.no_grad():
        real_latents = encode_to_latent(vae, data, DEVICE).to(DEVICE)
    
    print(f"\n{'='*60}")
    print("1. Real VAE Latent Statistics")
    print(f"{'='*60}")
    print(f"   Shape: {real_latents.shape}")
    print(f"   Mean:  {real_latents.mean().item():.6f}")
    print(f"   Std:   {real_latents.std().item():.6f}")
    print(f"   Min:   {real_latents.min().item():.6f}")
    print(f"   Max:   {real_latents.max().item():.6f}")
    
    # Per-dimension statistics
    per_dim_mean = real_latents.mean(dim=(0,1))
    per_dim_std = real_latents.std(dim=(0,1))
    print(f"\n   Per-dim mean range: [{per_dim_mean.min():.4f}, {per_dim_mean.max():.4f}]")
    print(f"   Per-dim std range: [{per_dim_std.min():.4f}, {per_dim_std.max():.4f}]")
    
    # GRU predictions with different seed steps
    print(f"\n{'='*60}")
    print("2. GRU Predictions vs Ground Truth")
    print(f"{'='*60}")
    
    # Load latent stats
    latent_stats_path = "checkpoints/gru/latent_norm_stats_multistep.npy"
    if os.path.exists(latent_stats_path):
        stats = np.load(latent_stats_path, allow_pickle=True).item()
        l_mean = torch.tensor(stats['mean']).to(DEVICE)
        l_std = torch.tensor(stats['std']).to(DEVICE)
    else:
        print("[WARNING] No latent stats found!")
        l_mean = torch.zeros(1).to(DEVICE)
        l_std = torch.ones(1).to(DEVICE)

    for seed_steps in [5, 10, 20, 32]:
        with torch.no_grad():
            # Use first seed_steps as context
            z_start = real_latents[:, seed_steps-1:seed_steps, :]
            
            # Normalize for GRU
            z_start_norm = (z_start - l_mean) / (l_std + 1e-8)
            
            # Predict remaining steps (output is normalized)
            num_predict = 64 - seed_steps
            predicted_norm = gru.generate_sequence(z_start_norm, num_predict)
            
            # Denormalize
            predicted = predicted_norm * (l_std + 1e-8) + l_mean
            
            # Ground truth for comparison
            gt = real_latents[:, seed_steps:, :]
            
            # Calculate error
            mse = ((predicted - gt) ** 2).mean().item()
            mae = (predicted - gt).abs().mean().item()
            
            # Statistics of predictions
            pred_mean = predicted.mean().item()
            pred_std = predicted.std().item()
            gt_mean = gt.mean().item()
            gt_std = gt.std().item()
        
        print(f"\n   Seed steps: {seed_steps}")
        print(f"   Predict: {num_predict} steps")
        print(f"   MSE: {mse:.6f}, MAE: {mae:.6f}")
        print(f"   GT stats:   mean={gt_mean:.4f}, std={gt_std:.4f}")
        print(f"   Pred stats: mean={pred_mean:.4f}, std={pred_std:.4f}")
        print(f"   Std ratio: {pred_std/gt_std:.4f}")
    
    # Check temporal dynamics
    print(f"\n{'='*60}")
    print("3. Temporal Dynamics Analysis")
    print(f"{'='*60}")
    
    with torch.no_grad():
        # Velocity (changes between timesteps)
        real_velocity = (real_latents[:, 1:, :] - real_latents[:, :-1, :]).abs().mean()
        
        # GRU predictions velocity
        z_start = real_latents[:, 4:5, :]
        z_start_norm = (z_start - l_mean) / (l_std + 1e-8)
        predicted_norm = gru.generate_sequence(z_start_norm, 59)
        predicted = predicted_norm * (l_std + 1e-8) + l_mean
        pred_velocity = (predicted[:, 1:, :] - predicted[:, :-1, :]).abs().mean()
    
    print(f"   Real latent velocity: {real_velocity:.6f}")
    print(f"   GRU predicted velocity: {pred_velocity:.6f}")
    print(f"   Ratio: {pred_velocity/real_velocity:.4f}")
    
    # Check if GRU is just predicting constants
    print(f"\n{'='*60}")
    print("4. Check for Lazy Prediction (Constant Output)")
    print(f"{'='*60}")
    
    with torch.no_grad():
        z_start = real_latents[:, 4:5, :]
        z_start_norm = (z_start - l_mean) / (l_std + 1e-8)
        predicted_norm = gru.generate_sequence(z_start_norm, 10)
        predicted = predicted_norm * (l_std + 1e-8) + l_mean
        
        # Check variation over time
        time_var = predicted.var(dim=1).mean().item()
        # Check variation across batch
        batch_var = predicted.var(dim=0).mean().item()
        # Check variation across dimensions
        dim_var = predicted.var(dim=2).mean().item()
    
    print(f"   Variance over time: {time_var:.6f}")
    print(f"   Variance over batch: {batch_var:.6f}")
    print(f"   Variance over dims: {dim_var:.6f}")
    
    # Compute what condition the diffusion expects
    print(f"\n{'='*60}")
    print("5. Condition Statistics (what diffusion expects)")
    print(f"{'='*60}")
    
    with torch.no_grad():
        # Real condition (averaged over time)
        real_condition = real_latents.mean(dim=1)
        
        # GRU-predicted condition
        z_start = real_latents[:, 4:5, :]
        z_start_norm = (z_start - l_mean) / (l_std + 1e-8)
        predicted_norm = gru.generate_sequence(z_start_norm, 59)
        predicted = predicted_norm * (l_std + 1e-8) + l_mean
        
        full_seq = torch.cat([real_latents[:, :5, :], predicted], dim=1)
        gru_condition = full_seq.mean(dim=1)
    
    print(f"   Real condition:  mean={real_condition.mean():.4f}, std={real_condition.std():.4f}")
    print(f"   GRU condition:   mean={gru_condition.mean():.4f}, std={gru_condition.std():.4f}")
    print(f"   Std ratio: {gru_condition.std()/real_condition.std():.4f}")
    
    # Correlation between real and GRU conditions
    corr = torch.corrcoef(torch.stack([
        real_condition.flatten(),
        gru_condition.flatten()
    ]))[0,1].item()
    print(f"   Correlation: {corr:.4f}")
    
    print("\n" + "="*60)
    print("Diagnosis Complete!")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
