"""
Evaluation script for Tangent Space Diffusion Model.

Validates that generated matrices are valid SPD and compares
their statistics to real data.
"""

import os
import sys
import torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.diffusion import TangentDiffusion
from src.data.dataset import BCIDataset
from src.preprocessing.geometry_utils import (
    validate_spd, exp_euclidean_map, vec_to_sym_matrix, riemannian_distance
)

# Config
INPUT_DIM = 253
N_CHANNELS = 23
DIFFUSION_PATH = "checkpoints/diffusion/tangent_diffusion_best.pth"
NORM_STATS_PATH = "checkpoints/diffusion/diffusion_norm_stats.npy"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_norm_stats():
    """Load normalization statistics for denormalization."""
    if os.path.exists(NORM_STATS_PATH):
        stats = np.load(NORM_STATS_PATH, allow_pickle=True).item()
        print(f"[OK] Loaded norm stats: mean={stats['mean']:.4f}, std={stats['std']:.4f}")
        return stats
    else:
        print("⚠️ No normalization stats found, using identity")
        return {'mean': 0.0, 'std': 1.0}


def evaluate_spd_constraints(diffusion, norm_stats, num_samples=100, seq_len=64):
    """
    Generate samples and validate SPD constraints.
    
    Returns detailed diagnostics about eigenvalues and condition numbers.
    """
    print("\n" + "="*60)
    print("SPD Constraint Validation")
    print("="*60)
    
    diffusion.eval()
    
    # Generate samples
    print(f"\nGenerating {num_samples} samples...")
    samples = diffusion.sample_ddim(
        (num_samples, seq_len, INPUT_DIM),
        condition=None,
        device=DEVICE,
        steps=50
    )
    
    # Denormalize samples to original scale
    data_mean = norm_stats['mean']
    data_std = norm_stats['std']
    samples = samples * data_std + data_mean
    print(f"Denormalized samples: mean={samples.mean().item():.4f}, std={samples.std().item():.4f}")
    
    # Validate each timestep
    all_min_eigvals = []
    all_max_eigvals = []
    all_condition_numbers = []
    all_valid = []
    
    print("Validating SPD constraints...")
    for t in tqdm(range(seq_len), desc="Timesteps"):
        tangent_vecs = samples[:, t, :]  # (B, 253)
        
        # Reconstruct symmetric matrix from strict lower triangle (no diagonal)
        # The data format is 23*22/2 = 253 elements (strict lower tri)
        batch_size = tangent_vecs.shape[0]
        matrices = torch.zeros(batch_size, N_CHANNELS, N_CHANNELS, device=DEVICE)
        
        # Get strict lower triangle indices (below diagonal, k=-1)
        rows, cols = torch.tril_indices(N_CHANNELS, N_CHANNELS, offset=-1)
        matrices[:, rows, cols] = tangent_vecs
        matrices[:, cols, rows] = tangent_vecs  # Symmetry
        # Diagonal stays 0 (log of 1 in normalized covariance)
        
        spd_matrices = exp_euclidean_map(matrices)
        
        try:
            is_valid, details = validate_spd(spd_matrices, return_details=True)
            all_valid.append(is_valid.float().mean().item())
            all_min_eigvals.append(details['min_eigenvalue'].min().item())
            all_max_eigvals.append(details['max_eigenvalue'].max().item())
            all_condition_numbers.append(details['condition_number'].mean().item())
        except Exception as e:
            print(f"\n⚠️ Timestep {t}: Numerical issue - {e}")
            all_valid.append(0.0)
            all_min_eigvals.append(float('nan'))
            all_max_eigvals.append(float('nan'))
            all_condition_numbers.append(float('inf'))
    
    # Summary
    print(f"\n{'─'*40}")
    print("RESULTS")
    print(f"{'─'*40}")
    print(f"✓ SPD Valid Rate: {np.mean(all_valid)*100:.2f}%")
    print(f"  Min Eigenvalue: {np.min(all_min_eigvals):.6f}")
    print(f"  Max Eigenvalue: {np.max(all_max_eigvals):.2f}")
    print(f"  Avg Condition Number: {np.mean(all_condition_numbers):.2f}")
    
    if np.mean(all_valid) == 1.0:
        print("\n✅ ALL GENERATED MATRICES ARE VALID SPD!")
    else:
        print(f"\n⚠️ Warning: {(1-np.mean(all_valid))*100:.2f}% matrices failed validation")
    
    return {
        'valid_rate': np.mean(all_valid),
        'min_eigenvalue': np.min(all_min_eigvals),
        'max_eigenvalue': np.max(all_max_eigvals),
        'avg_condition': np.mean(all_condition_numbers)
    }


def compare_to_real_data(diffusion, norm_stats, real_data_path="data/processed/train"):
    """
    Compare generated sample statistics to real data.
    """
    print("\n" + "="*60)
    print("Comparison to Real Data")
    print("="*60)
    
    # Load real data
    dataset = BCIDataset(real_data_path, sequence_length=64)
    real_sample = dataset[0][0]  # (T, D)
    
    # Generate samples
    generated = diffusion.sample_ddim(
        (10, 64, INPUT_DIM), condition=None, device=DEVICE, steps=50
    )
    
    # Denormalize generated samples
    data_mean = norm_stats['mean']
    data_std = norm_stats['std']
    generated = (generated * data_std + data_mean).cpu()
    
    # Compare statistics
    print(f"\n{'Statistic':<25} {'Real Data':<15} {'Generated':<15}")
    print("─" * 55)
    
    real_mean = real_sample.mean().item()
    gen_mean = generated.mean().item()
    print(f"{'Mean':<25} {real_mean:<15.4f} {gen_mean:<15.4f}")
    
    real_std = real_sample.std().item()
    gen_std = generated.std().item()
    print(f"{'Std':<25} {real_std:<15.4f} {gen_std:<15.4f}")
    
    real_min = real_sample.min().item()
    gen_min = generated.min().item()
    print(f"{'Min':<25} {real_min:<15.4f} {gen_min:<15.4f}")
    
    real_max = real_sample.max().item()
    gen_max = generated.max().item()
    print(f"{'Max':<25} {real_max:<15.4f} {gen_max:<15.4f}")


def main():
    print("\n🔬 Tangent Space Diffusion Model Evaluation")
    print("="*60)
    
    # Load normalization stats
    norm_stats = load_norm_stats()
    
    # Load model
    print(f"\nLoading model from {DIFFUSION_PATH}...")
    diffusion = TangentDiffusion(
        tangent_dim=INPUT_DIM,
        hidden_dim=512,
        condition_dim=128,
        n_steps=1000
    ).to(DEVICE)
    
    diffusion.load_state_dict(torch.load(DIFFUSION_PATH, map_location=DEVICE))
    diffusion.eval()
    print("✓ Model loaded")
    
    # Run evaluations
    with torch.no_grad():
        spd_results = evaluate_spd_constraints(diffusion, norm_stats)
        compare_to_real_data(diffusion, norm_stats)
    
    print("\n" + "="*60)
    print("Evaluation Complete!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
