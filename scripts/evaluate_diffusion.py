"""
Evaluate Conditional Diffusion Model

This script evaluates the conditional diffusion model with two modes:
1. 'vae' mode: Use real VAE-encoded latents (tests if diffusion works)
2. 'gru' mode: Use GRU-predicted latents (tests full pipeline)

Usage:
    python scripts/evaluate_diffusion.py --mode vae
    python scripts/evaluate_diffusion.py --mode gru
"""

import argparse
import os
import sys
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.diffusion import TangentDiffusion
from src.models.vae import ImprovedVAE
from src.models.gru import TemporalGRU
from src.data.data_utils import (
    load_normalized_dataset, get_normalization_stats, encode_to_latent,
    INPUT_DIM, N_CHANNELS, SEQUENCE_LENGTH as SEQ_LEN
)
from src.preprocessing.geometry_utils import (
    validate_spd, exp_euclidean_map, vec_to_sym_matrix
)

# ============================================================================
# Configuration
# ============================================================================

# Data dimensions imported from data_utils (INPUT_DIM=253, N_CHANNELS=22, SEQ_LEN=64)
VAE_LATENT_DIM = 32
GRU_HIDDEN_DIM = 128

# Paths
DIFFUSION_PATH = "checkpoints/diffusion/tangent_diffusion_best.pth"
VAE_PATH = "checkpoints/vae/vae_dynamics_latent32.pth"
GRU_PATH = "checkpoints/gru/gru_multistep_L32_H128_2L_best.pth"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================================
# Model Loading
# ============================================================================

def load_models(mode='vae'):
    """Load required models based on evaluation mode."""
    models = {}
    
    # Always load diffusion
    print(f"Loading Diffusion from {DIFFUSION_PATH}...")
    diffusion = TangentDiffusion(
        tangent_dim=INPUT_DIM,
        hidden_dim=512,
        condition_dim=VAE_LATENT_DIM,
        n_steps=1000
    ).to(DEVICE)
    diffusion.load_state_dict(torch.load(DIFFUSION_PATH, map_location=DEVICE))
    diffusion.eval()
    models['diffusion'] = diffusion
    print("[OK] Diffusion loaded")
    
    # Always load VAE (needed for conditioning)
    print(f"Loading VAE from {VAE_PATH}...")
    vae = ImprovedVAE(input_dim=INPUT_DIM, latent_dim=VAE_LATENT_DIM).to(DEVICE)
    vae.load_state_dict(torch.load(VAE_PATH, map_location=DEVICE))
    vae.eval()
    models['vae'] = vae
    print("[OK] VAE loaded")
    
    # Load GRU only for gru mode
    if mode == 'gru':
        print(f"Loading GRU from {GRU_PATH}...")
        gru = TemporalGRU(
            latent_dim=VAE_LATENT_DIM,
            hidden_dim=GRU_HIDDEN_DIM,
            num_layers=2,
            dropout=0.2
        ).to(DEVICE)
        gru.load_state_dict(torch.load(GRU_PATH, map_location=DEVICE))
        gru.eval()
        models['gru'] = gru
        print("[OK] GRU loaded")
    
    return models


def load_data(num_samples=100):
    """Load normalized data and normalization stats using shared utilities."""
    # Load data and stats from shared utilities
    normalized_data, norm_stats = load_normalized_dataset()
    
    # Sample random indices
    np.random.seed(42)
    indices = np.random.choice(len(normalized_data), num_samples, replace=False)
    data = torch.from_numpy(normalized_data[indices].copy()).float().to(DEVICE)
    
    return data, norm_stats


# ============================================================================
# Condition Generation
# ============================================================================

def get_vae_conditions(models, data):
    """Encode real data with VAE to get latent conditions."""
    vae = models['vae']
    batch_size, seq_len, input_dim = data.shape
    
    with torch.no_grad():
        flat = data.view(-1, input_dim)
        latent_mu, _ = vae.encode(flat)
        # Average over time for sequence-level condition
        conditions = latent_mu.view(batch_size, seq_len, -1).mean(dim=1)
    
    return conditions


def get_gru_conditions(models, data, seed_steps=5):
    """
    Use GRU to predict latent trajectories, then use as conditions.
    
    This tests if GRU learned correct dynamics:
    - If GRU predictions are good → diffusion generates realistic samples
    - If GRU predictions are bad → diffusion output will be poor
    """
    vae = models['vae']
    gru = models['gru']
    batch_size, seq_len, input_dim = data.shape
    
    # Load latent normalization stats (GRU was trained on normalized latents)
    latent_stats_path = "checkpoints/gru/latent_norm_stats_multistep.npy"
    if os.path.exists(latent_stats_path):
        stats = np.load(latent_stats_path, allow_pickle=True).item()
        l_mean = torch.tensor(stats['mean']).to(data.device)
        l_std = torch.tensor(stats['std']).to(data.device)
    else:
        print(f"[WARNING] No latent stats found at {latent_stats_path}. Using identity.")
        l_mean = torch.zeros(1).to(data.device)
        l_std = torch.ones(1).to(data.device)

    with torch.no_grad():
        # Encode to latent space
        flat = data.view(-1, input_dim)
        latent_mu = encode_to_latent(vae, data, data.device).to(data.device)
        latent_seq = latent_mu.view(batch_size, seq_len, -1)
        
        # Use last seed step as starting point for prediction
        z_start = latent_seq[:, seed_steps-1:seed_steps, :]  # (B, 1, 32)
        
        # Normalize z_start for GRU
        z_start_norm = (z_start - l_mean) / (l_std + 1e-8)
        
        # GRU autoregressive prediction (in normalized space)
        num_predict = seq_len - seed_steps
        predicted_norm = gru.generate_sequence(z_start_norm, num_predict)  # (B, num_predict, 32)
        
        # Denormalize predictions
        predicted = predicted_norm * (l_std + 1e-8) + l_mean
        
        # Combine seed + predicted (both in original latent space)
        full_seq = torch.cat([latent_seq[:, :seed_steps, :], predicted], dim=1)  # (B, seq_len, 32)
        
        # Use predicted latent as condition (average)
        conditions = full_seq.mean(dim=1)
    
    return conditions


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_spd(samples, norm_stats):
    """Validate SPD constraints on generated samples."""
    data_mean = norm_stats['mean']
    data_std = norm_stats['std']
    
    # Denormalize
    samples_denorm = samples * data_std + data_mean
    samples_denorm = samples_denorm.cpu()
    
    batch_size, seq_len, _ = samples.shape
    all_valid = []
    all_min_eigvals = []
    
    for t in range(seq_len):
        tangent_vecs = samples_denorm[:, t, :]
        matrices = vec_to_sym_matrix(tangent_vecs, N_CHANNELS)
        spd_matrices = exp_euclidean_map(matrices)
        
        try:
            eigvals = torch.linalg.eigvalsh(spd_matrices.double())
            min_eigval = eigvals.min(dim=-1).values
            is_valid = min_eigval > -1e-10
            all_valid.append(is_valid.float().mean().item())
            all_min_eigvals.append(min_eigval.min().item())
        except:
            all_valid.append(0.0)
            all_min_eigvals.append(float('nan'))
    
    spd_rate = np.mean(all_valid)
    min_eigval = np.nanmin(all_min_eigvals)
    
    return spd_rate, min_eigval


def compare_statistics(real_data, generated, norm_stats):
    """Compare statistics between real and generated samples."""
    data_mean = norm_stats['mean']
    data_std = norm_stats['std']
    
    # Denormalize
    real_denorm = (real_data * data_std + data_mean).cpu()
    gen_denorm = (generated * data_std + data_mean).cpu()
    
    stats = {
        'real_mean': real_denorm.mean().item(),
        'gen_mean': gen_denorm.mean().item(),
        'real_std': real_denorm.std().item(),
        'gen_std': gen_denorm.std().item(),
        'real_min': real_denorm.min().item(),
        'gen_min': gen_denorm.min().item(),
        'real_max': real_denorm.max().item(),
        'gen_max': gen_denorm.max().item(),
    }
    
    return stats


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate Conditional Diffusion")
    parser.add_argument('--mode', choices=['vae', 'gru'], default='vae',
                        help="Condition source: 'vae' (real latents) or 'gru' (predicted)")
    parser.add_argument('--num_samples', type=int, default=100,
                        help="Number of samples to generate")
    parser.add_argument('--seed_steps', type=int, default=5,
                        help="Seed steps for GRU prediction (only for gru mode)")
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print(f"Conditional Diffusion Evaluation (mode: {args.mode.upper()})")
    print("="*60)
    
    # Load models
    models = load_models(args.mode)
    
    # Load data
    print(f"\nLoading {args.num_samples} samples...")
    data, norm_stats = load_data(args.num_samples)
    print(f"[OK] Data loaded: shape {data.shape}")
    
    # Get conditions based on mode
    print(f"\nGenerating conditions ({args.mode} mode)...")
    if args.mode == 'vae':
        conditions = get_vae_conditions(models, data)
        print("[OK] Using real VAE-encoded latents as conditions")
    else:
        conditions = get_gru_conditions(models, data, args.seed_steps)
        print(f"[OK] Using GRU-predicted latents (seed={args.seed_steps} steps)")
    
    print(f"    Condition stats: mean={conditions.mean().item():.4f}, std={conditions.std().item():.4f}")
    
    # Generate samples
    print(f"\nGenerating {args.num_samples} samples with diffusion...")
    with torch.no_grad():
        generated = models['diffusion'].sample_ddim(
            (args.num_samples, SEQ_LEN, INPUT_DIM),
            condition=conditions,
            device=DEVICE,
            steps=50
        )
    print("[OK] Samples generated")
    
    # Evaluate SPD
    print("\nValidating SPD constraints...")
    spd_rate, min_eigval = evaluate_spd(generated, norm_stats)
    
    # Compare to real data
    stats = compare_statistics(data, generated, norm_stats)
    
    # Print results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    print(f"\n📊 SPD Validation:")
    print(f"   Valid Rate: {spd_rate*100:.2f}%")
    print(f"   Min Eigenvalue: {min_eigval:.6f}")
    
    if spd_rate == 1.0:
        print("   ✅ All matrices are valid SPD!")
    else:
        print(f"   ⚠️ {(1-spd_rate)*100:.2f}% failed validation")
    
    print(f"\n📈 Statistics Comparison:")
    print(f"   {'Metric':<15} {'Real':<15} {'Generated':<15} {'Diff':<10}")
    print("   " + "-"*55)
    print(f"   {'Mean':<15} {stats['real_mean']:<15.4f} {stats['gen_mean']:<15.4f} {abs(stats['real_mean']-stats['gen_mean']):<10.4f}")
    print(f"   {'Std':<15} {stats['real_std']:<15.4f} {stats['gen_std']:<15.4f} {abs(stats['real_std']-stats['gen_std']):<10.4f}")
    print(f"   {'Min':<15} {stats['real_min']:<15.4f} {stats['gen_min']:<15.4f} {abs(stats['real_min']-stats['gen_min']):<10.4f}")
    print(f"   {'Max':<15} {stats['real_max']:<15.4f} {stats['gen_max']:<15.4f} {abs(stats['real_max']-stats['gen_max']):<10.4f}")
    
    print("\n" + "="*60)
    print("Done!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
