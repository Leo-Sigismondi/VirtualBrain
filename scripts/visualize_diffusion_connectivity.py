"""
Diffusion Model Connectivity Visualization using MNE-Connectivity

This script evaluates the diffusion model quality by visualizing:
1. Connectivity matrix heatmaps (real vs generated)
2. Statistical comparison metrics
3. Optional MNE 3D sensor connectivity (if mne_connectivity installed)

Two modes:
- 'vae': Use real VAE-encoded latents as conditions (tests diffusion quality)
- 'gru': Use GRU-predicted latents as conditions (tests full pipeline)

Dataset: BCI Competition IV 2a (22 EEG channels, 10-20 system)

Usage:
    python scripts/visualize_diffusion_connectivity.py --mode vae
    python scripts/visualize_diffusion_connectivity.py --mode gru --seed_steps 8
"""

import argparse
import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
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
    validate_spd, exp_euclidean_map, vec_to_sym_matrix, riemannian_distance,
    log_euclidean_map
)

# ============================================================================
# Configuration - MATCHES TRAINING EXACTLY
# ============================================================================

# Diffusion config (from train_diffusion.py)
DIFFUSION_HIDDEN_DIM = 512
DIFFUSION_STEPS = 1000
DIFFUSION_SCHEDULE = 'cosine'

# Model dimensions
VAE_LATENT_DIM = 32
GRU_HIDDEN_DIM = 128
GRU_NUM_LAYERS = 2

# Paths
DIFFUSION_PATH = "checkpoints/diffusion/tangent_diffusion_best.pth"
VAE_PATH = "checkpoints/vae/vae_dynamics_latent32.pth"
GRU_PATH = "checkpoints/gru/gru_multistep_L32_H128_2L_best.pth"
GRU_LATENT_STATS_PATH = "checkpoints/gru/latent_norm_stats_multistep.npy"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# BCI Competition IV 2a - 22 EEG channels (10-20 system)
# Channel order as typically used in this dataset
BCI_IV_2A_CHANNELS = [
    'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4',
    'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
    'CP3', 'CP1', 'CPz', 'CP2', 'CP4',
    'P1', 'Pz', 'P2', 'POz'
]


# ============================================================================
# Model Loading
# ============================================================================

def load_models(mode='vae'):
    """Load required models with EXACT same config as training."""
    models = {}
    
    # Load diffusion with EXACT same config as train_diffusion.py
    print(f"Loading Diffusion from {DIFFUSION_PATH}...")
    print(f"  Config: hidden_dim={DIFFUSION_HIDDEN_DIM}, steps={DIFFUSION_STEPS}, schedule={DIFFUSION_SCHEDULE}")
    diffusion = TangentDiffusion(
        tangent_dim=INPUT_DIM,           # 253
        hidden_dim=DIFFUSION_HIDDEN_DIM, # 512
        condition_dim=VAE_LATENT_DIM,    # 32
        n_steps=DIFFUSION_STEPS,         # 1000
        schedule=DIFFUSION_SCHEDULE      # 'cosine'
    ).to(DEVICE)
    diffusion.load_state_dict(torch.load(DIFFUSION_PATH, map_location=DEVICE))
    diffusion.eval()
    models['diffusion'] = diffusion
    print("[OK] Diffusion loaded")
    
    # Load VAE
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
            num_layers=GRU_NUM_LAYERS,
            dropout=0.2
        ).to(DEVICE)
        gru.load_state_dict(torch.load(GRU_PATH, map_location=DEVICE))
        gru.eval()
        models['gru'] = gru
        print("[OK] GRU loaded")
    
    return models


def load_data(num_samples=50):
    """Load normalized data samples."""
    normalized_data, norm_stats = load_normalized_dataset()
    
    np.random.seed(42)
    indices = np.random.choice(len(normalized_data), num_samples, replace=False)
    data = torch.from_numpy(normalized_data[indices].copy()).float().to(DEVICE)
    
    return data, norm_stats


# ============================================================================
# Condition Generation
# ============================================================================

def get_vae_conditions(models, data):
    """Encode real data with VAE to get per-frame latent conditions."""
    vae = models['vae']
    batch_size, seq_len, input_dim = data.shape
    
    with torch.no_grad():
        flat = data.view(-1, input_dim)
        latent_mu, _ = vae.encode(flat)
        # Per-frame conditions preserve temporal dynamics
        conditions = latent_mu.view(batch_size, seq_len, -1)  # (B, T, 32)
    
    return conditions


def get_gru_conditions(models, data, seed_steps=8):
    """Use GRU to predict latent trajectories as conditions."""
    vae = models['vae']
    gru = models['gru']
    batch_size, seq_len, input_dim = data.shape
    
    # Load GRU latent normalization stats
    if os.path.exists(GRU_LATENT_STATS_PATH):
        stats = np.load(GRU_LATENT_STATS_PATH, allow_pickle=True).item()
        l_mean = torch.tensor(stats['mean']).to(DEVICE)
        l_std = torch.tensor(stats['std']).to(DEVICE)
    else:
        print(f"[WARNING] No latent stats found at {GRU_LATENT_STATS_PATH}")
        l_mean = torch.zeros(1).to(DEVICE)
        l_std = torch.ones(1).to(DEVICE)

    with torch.no_grad():
        # Encode to latent space
        latent_mu = encode_to_latent(vae, data, DEVICE).to(DEVICE)
        latent_seq = latent_mu.view(batch_size, seq_len, -1)
        
        # Use seed steps to warm up hidden state
        hidden = None
        for t in range(seed_steps):
            current = (latent_seq[:, t:t+1, :] - l_mean) / (l_std + 1e-8)
            _, hidden = gru.predict_next(current, hidden)
        
        # Predict remaining steps
        z_start_norm = (latent_seq[:, seed_steps-1:seed_steps, :] - l_mean) / (l_std + 1e-8)
        num_predict = seq_len - seed_steps
        predicted_norm = gru.generate_sequence(z_start_norm, num_predict)
        predicted = predicted_norm * (l_std + 1e-8) + l_mean
        
        # Combine seed + predicted
        full_seq = torch.cat([latent_seq[:, :seed_steps, :], predicted], dim=1)
        # Return per-frame conditions (B, T, 32)
        conditions = full_seq
    
    return conditions


# ============================================================================
# SPD Conversion and Connectivity Extraction
# ============================================================================

def tangent_to_spd(tangent_vecs, norm_stats):
    """Convert normalized tangent vectors to SPD matrices."""
    # Denormalize
    denorm = tangent_vecs * norm_stats['std'] + norm_stats['mean']
    
    # Reshape to symmetric matrix and apply exp map
    matrices = vec_to_sym_matrix(denorm, N_CHANNELS)
    spd_matrices = exp_euclidean_map(matrices)
    
    return spd_matrices


def extract_connectivity(spd_matrices):
    """Extract connectivity strengths from SPD covariance matrices.
    
    Uses the correlation form: C_ij = Cov_ij / sqrt(Cov_ii * Cov_jj)
    """
    # spd_matrices: (B, N, N) in float64
    diag = spd_matrices.diagonal(dim1=-2, dim2=-1)  # (B, N)
    diag_sqrt = torch.sqrt(diag.unsqueeze(-1) * diag.unsqueeze(-2))  # (B, N, N)
    
    # Correlation matrix
    corr = spd_matrices / (diag_sqrt + 1e-10)
    
    # Set diagonal to 1 (self-connection)
    eye = torch.eye(N_CHANNELS, device=corr.device, dtype=corr.dtype)
    corr = corr * (1 - eye) + eye
    
    return corr.float()


# ============================================================================
# Visualization
# ============================================================================

def plot_connectivity_comparison(real_conn, gen_conn, channel_names, output_path, mode):
    """Create side-by-side connectivity matrix comparison."""
    # Average over batch and time
    real_avg = real_conn.mean(dim=(0, 1)).cpu().numpy()
    gen_avg = gen_conn.mean(dim=(0, 1)).cpu().numpy()
    
    fig = plt.figure(figsize=(18, 6))
    gs = GridSpec(1, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.3)
    
    # Common colorbar limits
    vmin = min(real_avg.min(), gen_avg.min())
    vmax = max(real_avg.max(), gen_avg.max())
    
    # Real connectivity
    ax1 = fig.add_subplot(gs[0])
    im1 = ax1.imshow(real_avg, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')
    ax1.set_title('Real Data\nConnectivity', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(0, N_CHANNELS, 3))
    ax1.set_yticks(range(0, N_CHANNELS, 3))
    ax1.set_xticklabels([channel_names[i] for i in range(0, N_CHANNELS, 3)], rotation=45, ha='right')
    ax1.set_yticklabels([channel_names[i] for i in range(0, N_CHANNELS, 3)])
    
    # Generated connectivity
    ax2 = fig.add_subplot(gs[1])
    im2 = ax2.imshow(gen_avg, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')
    ax2.set_title(f'Generated ({mode.upper()} mode)\nConnectivity', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(0, N_CHANNELS, 3))
    ax2.set_yticks(range(0, N_CHANNELS, 3))
    ax2.set_xticklabels([channel_names[i] for i in range(0, N_CHANNELS, 3)], rotation=45, ha='right')
    ax2.set_yticklabels([channel_names[i] for i in range(0, N_CHANNELS, 3)])
    
    # Difference
    diff = gen_avg - real_avg
    ax3 = fig.add_subplot(gs[2])
    max_diff = max(abs(diff.min()), abs(diff.max()))
    im3 = ax3.imshow(diff, cmap='PiYG', vmin=-max_diff, vmax=max_diff, aspect='equal')
    ax3.set_title(f'Difference\n(MAE={np.abs(diff).mean():.4f})', fontsize=14, fontweight='bold')
    ax3.set_xticks(range(0, N_CHANNELS, 3))
    ax3.set_yticks(range(0, N_CHANNELS, 3))
    ax3.set_xticklabels([channel_names[i] for i in range(0, N_CHANNELS, 3)], rotation=45, ha='right')
    ax3.set_yticklabels([channel_names[i] for i in range(0, N_CHANNELS, 3)])
    
    # Colorbars
    cax = fig.add_subplot(gs[3])
    fig.colorbar(im2, cax=cax, label='Correlation')
    
    plt.suptitle(f'Diffusion Connectivity Evaluation ({mode.upper()} conditioning)', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[SAVED] {output_path}")
    plt.close()
    
    return real_avg, gen_avg


def plot_edge_distribution(real_conn, gen_conn, output_path, mode):
    """Compare distribution of edge weights."""
    # Get upper triangle (exclude diagonal)
    mask = torch.triu(torch.ones(N_CHANNELS, N_CHANNELS), diagonal=1).bool()
    
    real_edges = real_conn[:, :, mask].cpu().numpy().flatten()
    gen_edges = gen_conn[:, :, mask].cpu().numpy().flatten()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histograms
    axes[0].hist(real_edges, bins=50, alpha=0.7, label='Real', density=True, color='steelblue')
    axes[0].hist(gen_edges, bins=50, alpha=0.7, label='Generated', density=True, color='coral')
    axes[0].set_xlabel('Correlation Strength')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Edge Weight Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Q-Q plot
    real_sorted = np.sort(real_edges)
    gen_sorted = np.sort(gen_edges)
    # Resample to same length
    n_points = min(len(real_sorted), len(gen_sorted), 5000)
    real_q = np.percentile(real_sorted, np.linspace(0, 100, n_points))
    gen_q = np.percentile(gen_sorted, np.linspace(0, 100, n_points))
    
    axes[1].scatter(real_q, gen_q, alpha=0.3, s=1)
    lims = [min(real_q.min(), gen_q.min()), max(real_q.max(), gen_q.max())]
    axes[1].plot(lims, lims, 'r--', linewidth=2, label='Perfect match')
    axes[1].set_xlabel('Real Quantiles')
    axes[1].set_ylabel('Generated Quantiles')
    axes[1].set_title('Q-Q Plot')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(f'Edge Weight Comparison ({mode.upper()} mode)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[SAVED] {output_path}")
    plt.close()


def compute_metrics(real_spd, gen_spd, real_conn, gen_conn):
    """Compute evaluation metrics."""
    metrics = {}
    
    # 1. SPD Validity (use lenient threshold since exp() guarantees SPD)
    is_valid, details = validate_spd(gen_spd.reshape(-1, N_CHANNELS, N_CHANNELS), eps=1e-10, return_details=True)
    metrics['spd_valid_rate'] = is_valid.float().mean().item()
    metrics['min_eigenvalue'] = details['min_eigenvalue'].min().item()
    
    # 2. Mean Riemannian Distance
    real_flat = real_spd.reshape(-1, N_CHANNELS, N_CHANNELS)
    gen_flat = gen_spd.reshape(-1, N_CHANNELS, N_CHANNELS)
    n_samples = min(500, real_flat.shape[0])  # Limit for speed
    dists = riemannian_distance(
        real_flat[:n_samples], 
        gen_flat[:n_samples], 
        metric='log_euclidean'
    )
    metrics['mean_riemannian_dist'] = dists.mean().item()
    metrics['std_riemannian_dist'] = dists.std().item()
    
    # 3. Connectivity Correlation
    real_avg = real_conn.mean(dim=(0, 1)).cpu().numpy()
    gen_avg = gen_conn.mean(dim=(0, 1)).cpu().numpy()
    # Upper triangle only
    mask = np.triu(np.ones((N_CHANNELS, N_CHANNELS)), k=1).astype(bool)
    corr = np.corrcoef(real_avg[mask], gen_avg[mask])[0, 1]
    metrics['connectivity_correlation'] = corr
    
    # 4. MAE of connectivity
    metrics['connectivity_mae'] = np.abs(real_avg - gen_avg).mean()
    
    return metrics


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Visualize Diffusion Connectivity")
    parser.add_argument('--mode', choices=['vae', 'gru'], default='vae',
                        help="Condition source: 'vae' (real latents) or 'gru' (predicted)")
    parser.add_argument('--num_samples', type=int, default=50,
                        help="Number of samples to generate")
    parser.add_argument('--seed_steps', type=int, default=8,
                        help="Seed steps for GRU (only for gru mode)")
    parser.add_argument('--output_dir', type=str, default='checkpoints/diffusion/vis',
                        help="Output directory for visualizations")
    parser.add_argument('--ddim_steps', type=int, default=50,
                        help="DDIM sampling steps (faster than full 1000)")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print(f"Diffusion Connectivity Visualization ({args.mode.upper()} mode)")
    print("="*70)
    print(f"Diffusion Config: hidden={DIFFUSION_HIDDEN_DIM}, steps={DIFFUSION_STEPS}, schedule={DIFFUSION_SCHEDULE}")
    print(f"VAE Latent: {VAE_LATENT_DIM}, GRU Hidden: {GRU_HIDDEN_DIM}")
    print(f"Device: {DEVICE}")
    print("="*70 + "\n")
    
    # Load models
    models = load_models(args.mode)
    
    # Load data
    print(f"\nLoading {args.num_samples} samples...")
    data, norm_stats = load_data(args.num_samples)
    print(f"[OK] Data shape: {data.shape}")
    
    # Get conditions
    print(f"\nGenerating conditions ({args.mode} mode)...")
    if args.mode == 'vae':
        conditions = get_vae_conditions(models, data)
        print("[OK] Using real VAE-encoded latents as conditions")
    else:
        conditions = get_gru_conditions(models, data, args.seed_steps)
        print(f"[OK] Using GRU-predicted latents (seed={args.seed_steps} steps)")
    
    print(f"    Condition stats: mean={conditions.mean().item():.4f}, std={conditions.std().item():.4f}")
    
    # Generate samples
    print(f"\nGenerating samples with DDIM ({args.ddim_steps} steps)...")
    with torch.no_grad():
        generated = models['diffusion'].sample_ddim(
            (args.num_samples, SEQ_LEN, INPUT_DIM),
            condition=conditions,
            device=DEVICE,
            steps=args.ddim_steps
        )
    print("[OK] Samples generated")
    
    # Convert to SPD and extract connectivity
    print("\nConverting to SPD and extracting connectivity...")
    real_spd = tangent_to_spd(data.reshape(-1, INPUT_DIM), norm_stats)
    gen_spd = tangent_to_spd(generated.reshape(-1, INPUT_DIM), norm_stats)
    
    real_spd = real_spd.reshape(args.num_samples, SEQ_LEN, N_CHANNELS, N_CHANNELS)
    gen_spd = gen_spd.reshape(args.num_samples, SEQ_LEN, N_CHANNELS, N_CHANNELS)
    
    real_conn = extract_connectivity(real_spd)
    gen_conn = extract_connectivity(gen_spd)
    print("[OK] Connectivity extracted")
    
    # Compute metrics
    print("\nComputing metrics...")
    metrics = compute_metrics(real_spd, gen_spd, real_conn, gen_conn)
    
    # Print results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    
    print(f"\n📊 SPD Validation:")
    print(f"   Valid Rate: {metrics['spd_valid_rate']*100:.2f}%")
    print(f"   Min Eigenvalue: {metrics['min_eigenvalue']:.6f}")
    if metrics['spd_valid_rate'] == 1.0:
        print("   ✅ All matrices are valid SPD!")
    else:
        print(f"   ⚠️ {(1-metrics['spd_valid_rate'])*100:.2f}% failed validation")
    
    print(f"\n📐 Riemannian Distance (Log-Euclidean):")
    print(f"   Mean: {metrics['mean_riemannian_dist']:.4f} ± {metrics['std_riemannian_dist']:.4f}")
    
    print(f"\n🔗 Connectivity Metrics:")
    print(f"   Structure Correlation: {metrics['connectivity_correlation']:.4f}")
    print(f"   Edge MAE: {metrics['connectivity_mae']:.4f}")
    
    if metrics['connectivity_correlation'] > 0.8:
        print("   ✅ Excellent connectivity preservation!")
    elif metrics['connectivity_correlation'] > 0.5:
        print("   ⚠️ Moderate connectivity preservation")
    else:
        print("   ❌ Poor connectivity preservation")
    
    # Generate visualizations
    print("\n" + "="*70)
    print("Generating visualizations...")
    print("="*70)
    
    # Connectivity comparison
    conn_path = os.path.join(args.output_dir, f'connectivity_comparison_{args.mode}.png')
    plot_connectivity_comparison(real_conn, gen_conn, BCI_IV_2A_CHANNELS, conn_path, args.mode)
    
    # Edge distribution
    edge_path = os.path.join(args.output_dir, f'edge_distribution_{args.mode}.png')
    plot_edge_distribution(real_conn, gen_conn, edge_path, args.mode)
    
    # Save metrics
    metrics_path = os.path.join(args.output_dir, f'metrics_{args.mode}.npy')
    np.save(metrics_path, metrics)
    print(f"[SAVED] {metrics_path}")
    
    print("\n" + "="*70)
    print("Done!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
