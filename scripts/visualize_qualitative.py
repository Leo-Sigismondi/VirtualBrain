"""
Qualitative Evaluation Script for Brain Connectivity
====================================================
Generates neuroscience-grade visualizations to assess generation quality:
1. Connectivity Fingerprints (Correlation Heatmaps)
2. Topological Connectograms (Brain plots)
3. Temporal Dynamics Analysis (Trajectory tracking)
"""

import argparse
import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as patches

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.diffusion import TangentDiffusion
from src.models.vae import ImprovedVAE
from src.models.gru import TemporalGRU
from src.data.data_utils import (
    load_normalized_dataset, encode_to_latent,
    INPUT_DIM, N_CHANNELS, SEQUENCE_LENGTH
)
from src.preprocessing.geometry_utils import (
    exp_euclidean_map, vec_to_sym_matrix
)

# --- CONFIG ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Standard 10-20 System Channel Names (BCI IV 2a)
CHANNEL_NAMES = [
    'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4',
    'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
    'CP3', 'CP1', 'CPz', 'CP2', 'CP4',
    'P1', 'Pz', 'P2', 'POz'
]

# Approximate 2D positions for plotting (Top view, Nose up)
# Normalized coordinates [-1, 1]
CHANNEL_POS = {
    'Fz':  (0.0, 0.7),
    'FC3': (-0.5, 0.5), 'FC1': (-0.2, 0.5), 'FCz': (0.0, 0.5), 'FC2': (0.2, 0.5), 'FC4': (0.5, 0.5),
    'C5':  (-0.7, 0.0), 'C3':  (-0.5, 0.0), 'C1': (-0.2, 0.0), 'Cz': (0.0, 0.0), 'C2': (0.2, 0.0), 'C4': (0.5, 0.0), 'C6': (0.7, 0.0),
    'CP3': (-0.5, -0.5), 'CP1': (-0.2, -0.5), 'CPz': (0.0, -0.5), 'CP2': (0.2, -0.5), 'CP4': (0.5, -0.5),
    'P1':  (-0.2, -0.7), 'Pz':  (0.0, -0.7), 'P2': (0.2, -0.7),
    'POz': (0.0, -0.9)
}

def cov_to_corr(cov_matrix):
    """
    Converts Covariance Matrix to Correlation Matrix.
    Corr_ij = Cov_ij / sqrt(Cov_ii * Cov_jj)
    """
    diag = torch.diagonal(cov_matrix, dim1=-2, dim2=-1) # (B, N)
    std = torch.sqrt(diag) # (B, N)
    outer_std = torch.bmm(std.unsqueeze(2), std.unsqueeze(1)) # (B, N, N)
    
    corr = cov_matrix / (outer_std + 1e-8)
    # Clamp to handle numerical errors
    corr = torch.clamp(corr, -1.0, 1.0)
    
    # Fill diagonal with 1.0 explicitly (sometimes numerical noise makes it 0.999)
    eye = torch.eye(cov_matrix.shape[-1], device=cov_matrix.device)
    corr = corr * (1 - eye) + eye
    
    return corr

class QualitativeVisualizer:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def plot_fingerprint(self, real_seq, gen_seq, sample_idx=0):
        """
        Plot 1: Connectivity Fingerprint (Correlation Heatmaps)
        Visualizes the mean correlation matrix over time for a single sequence.
        """
        # Convert to correlation
        real_corr = cov_to_corr(real_seq) # (T, N, N)
        gen_corr = cov_to_corr(gen_seq)   # (T, N, N)
        
        # Average over time to get the "static functional connectivity" of this sequence
        real_mean = real_corr.mean(dim=0).cpu().numpy()
        gen_mean = gen_corr.mean(dim=0).cpu().numpy()
        
        # Mask the diagonal (always 1.0) to focus on off-diagonal patterns
        np.fill_diagonal(real_mean, np.nan)
        np.fill_diagonal(gen_mean, np.nan)
        
        diff = np.abs(np.nan_to_num(real_mean) - np.nan_to_num(gen_mean))
        
        fig = plt.figure(figsize=(20, 6))
        gs = GridSpec(1, 4, width_ratios=[1, 1, 1, 0.05])
        
        # Use data-driven range to show actual variation
        # Real brain correlations are typically very small (near 0)
        # Get percentiles to determine good color range
        all_vals = np.concatenate([real_mean[~np.isnan(real_mean)], gen_mean[~np.isnan(gen_mean)]])
        vmin = np.percentile(all_vals, 2)
        vmax = np.percentile(all_vals, 98)
        # Ensure symmetric around 0 for diverging colormap
        abs_max = max(abs(vmin), abs(vmax))
        vmin, vmax = -abs_max, abs_max
        
        cmap = 'coolwarm'
        
        # 1. Real
        ax1 = fig.add_subplot(gs[0])
        im1 = ax1.imshow(real_mean, cmap=cmap, vmin=vmin, vmax=vmax)
        ax1.set_title("Real Correlation (Mean over Time)", fontsize=12, fontweight='bold')
        self._add_labels(ax1)
        
        # 2. Generated
        ax2 = fig.add_subplot(gs[1])
        im2 = ax2.imshow(gen_mean, cmap=cmap, vmin=vmin, vmax=vmax)
        ax2.set_title("Generated Correlation", fontsize=12, fontweight='bold')
        self._add_labels(ax2)
        
        # 3. Difference
        ax3 = fig.add_subplot(gs[2])
        im3 = ax3.imshow(diff, cmap='YlOrRd', vmin=0, vmax=0.3)
        ax3.set_title(f"Absolute Difference (MAE: {np.nanmean(diff):.4f})", fontsize=12, fontweight='bold')
        self._add_labels(ax3)
        
        # Colorbar
        cbar_ax = fig.add_subplot(gs[3])
        fig.colorbar(im1, cax=cbar_ax)
        
        plt.suptitle(f"Connectivity Fingerprint (Sample {sample_idx})", fontsize=16, y=1.05)
        
        out_path = os.path.join(self.output_dir, f"fingerprint_sample_{sample_idx}.png")
        plt.savefig(out_path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"  [Saved] {out_path}")

    def plot_connectogram(self, real_seq, gen_seq, sample_idx=0, threshold_percentile=90):
        """
        Plot 2: Topological Connectogram
        Draws the head and strongest connections.
        """
        real_corr = cov_to_corr(real_seq).mean(dim=0).cpu().numpy() # (N, N)
        gen_corr = cov_to_corr(gen_seq).mean(dim=0).cpu().numpy()   # (N, N)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        self._draw_head(axes[0], real_corr, "Real Top 10% Connections", threshold_percentile)
        self._draw_head(axes[1], gen_corr, "Generated Top 10% Connections", threshold_percentile)
        
        plt.suptitle(f"Topological Connectogram (Sample {sample_idx})", fontsize=16)
        
        out_path = os.path.join(self.output_dir, f"connectogram_sample_{sample_idx}.png")
        plt.savefig(out_path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"  [Saved] {out_path}")

    def plot_temporal_dynamics(self, real_seq, gen_seq, sample_idx=0):
        """
        Plot 3: Temporal Trajectories of Key Connections
        Tracks how correlation between specific pairs evolves over time.
        Focuses on C3-C4 (Inter-hemispheric motor) and Cz-Pz (Central-Parietal).
        """
        # Indices
        idx_C3 = CHANNEL_NAMES.index('C3')
        idx_C4 = CHANNEL_NAMES.index('C4')
        idx_Cz = CHANNEL_NAMES.index('Cz')
        idx_Pz = CHANNEL_NAMES.index('Pz')
        
        real_corr = cov_to_corr(real_seq).cpu().numpy() # (T, N, N)
        gen_corr = cov_to_corr(gen_seq).cpu().numpy()   # (T, N, N)
        
        # Extract trajectories
        traj_real_C3C4 = real_corr[:, idx_C3, idx_C4]
        traj_gen_C3C4 = gen_corr[:, idx_C3, idx_C4]
        
        traj_real_CzPz = real_corr[:, idx_Cz, idx_Pz]
        traj_gen_CzPz = gen_corr[:, idx_Cz, idx_Pz]
        
        # Plot
        fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        t = np.arange(len(traj_real_C3C4))
        
        # C3-C4
        ax = axes[0]
        ax.plot(t, traj_real_C3C4, 'b-', linewidth=2, label='Real', alpha=0.7)
        ax.plot(t, traj_gen_C3C4, 'r--', linewidth=2, label='Generated', alpha=0.7)
        ax.set_ylabel("Correlation")
        ax.set_title("C3-C4 Connectivity (Inter-hemispheric Motor)", fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Cz-Pz
        ax = axes[1]
        ax.plot(t, traj_real_CzPz, 'b-', linewidth=2, label='Real', alpha=0.7)
        ax.plot(t, traj_gen_CzPz, 'r--', linewidth=2, label='Generated', alpha=0.7)
        ax.set_ylabel("Correlation")
        ax.set_xlabel("Timesteps")
        ax.set_title("Cz-Pz Connectivity (Central-Parietal)", fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.suptitle(f"Temporal Dynamics Analysis (Sample {sample_idx})", fontsize=16)
        
        out_path = os.path.join(self.output_dir, f"dynamics_sample_{sample_idx}.png")
        plt.savefig(out_path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"  [Saved] {out_path}")

    # --- Helpers ---
    def _add_labels(self, ax):
        ax.set_xticks(range(0, len(CHANNEL_NAMES), 2))
        ax.set_yticks(range(0, len(CHANNEL_NAMES), 2))
        ax.set_xticklabels(CHANNEL_NAMES[::2], rotation=90, fontsize=8)
        ax.set_yticklabels(CHANNEL_NAMES[::2], fontsize=8)

    def _draw_head(self, ax, corr_matrix, title, percentile=95):
        """Custom Head Plot using Matplotlib (No MNE required)"""
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Draw Head Circle
        circle = patches.Circle((0, 0), 1.0, edgecolor='black', facecolor='none', linewidth=2)
        ax.add_patch(circle)
        
        # Draw Nose
        ax.plot([-0.1, 0, 0.1], [1.0, 1.1, 1.0], 'k-', linewidth=2)
        
        # Plot Nodes
        x_coords = [CHANNEL_POS[ch][0] for ch in CHANNEL_NAMES]
        y_coords = [CHANNEL_POS[ch][1] for ch in CHANNEL_NAMES]
        ax.scatter(x_coords, y_coords, s=100, c='lightgray', edgecolors='black', zorder=10)
        
        for i, ch in enumerate(CHANNEL_NAMES):
            ax.text(x_coords[i], y_coords[i], ch, fontsize=8, ha='center', va='center', zorder=11)
            
        # Draw Edges (Thresholded)
        threshold = np.percentile(np.abs(corr_matrix), percentile)
        n = len(CHANNEL_NAMES)
        
        # Only iterate upper triangle
        for i in range(n):
            for j in range(i + 1, n):
                val = corr_matrix[i, j]
                if abs(val) >= threshold:
                    # Color based on sign (Red=Pos, Blue=Neg)
                    color = 'red' if val > 0 else 'blue'
                    # Width based on magnitude
                    width = (abs(val) - threshold) / (1 - threshold) * 3 + 0.5
                    
                    x_vals = [x_coords[i], x_coords[j]]
                    y_vals = [y_coords[i], y_coords[j]]
                    
                    ax.plot(x_vals, y_vals, color=color, linewidth=width, alpha=0.6, zorder=5)
                    
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['vae', 'gru'], default='vae', help="Latent source")
    parser.add_argument('--samples', type=int, default=3, help="Number of samples to visualize")
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print(f"Qualitative Visualization Pipeline (Mode: {args.mode.upper()})")
    print("="*60)
    
    # 1. Load Models & Data
    # (Assuming checkpoints exist, otherwise simplified loading)
    print("Loading models and data...")
    vae = ImprovedVAE(input_dim=253, latent_dim=32).to(DEVICE)
    vae.load_state_dict(torch.load("checkpoints/vae/vae_dynamics_latent32_best.pth", map_location=DEVICE))
    vae.eval()
    
    diffusion = TangentDiffusion(tangent_dim=253, hidden_dim=512, condition_dim=32, n_steps=1000).to(DEVICE)
    diffusion.load_state_dict(torch.load("checkpoints/diffusion/tangent_diffusion_best.pth", map_location=DEVICE))
    diffusion.eval()
    
    # Load GRU if needed
    gru = None
    if args.mode == 'gru':
        gru = TemporalGRU(latent_dim=32, hidden_dim=256, num_layers=3).to(DEVICE)
        gru.load_state_dict(torch.load("checkpoints/gru/gru_multistep_L32_H256_3L_best.pth", map_location=DEVICE))
        gru.eval()
        
        # Load GRU Latent Stats for Denormalization
        gru_stats = np.load("checkpoints/gru/latent_norm_stats_multistep.npy", allow_pickle=True).item()
        l_mean = torch.tensor(gru_stats['mean']).to(DEVICE)
        l_std = torch.tensor(gru_stats['std']).to(DEVICE)

    # Load Data
    data_norm, stats = load_normalized_dataset()
    
    # Select random samples
    indices = np.random.choice(len(data_norm), args.samples, replace=False)
    
    visualizer = QualitativeVisualizer(f"checkpoints/qualitative_vis_{args.mode}")
    
    for i, idx in enumerate(indices):
        print(f"\nProcessing Sample {i+1}/{args.samples} (Index {idx})...")
        
        # Prepare Real Data
        real_vec = torch.from_numpy(data_norm[idx]).float().to(DEVICE).unsqueeze(0) # (1, T, 253)
        
        # --- Generate Condition ---
        if args.mode == 'vae':
            # VAE Condition: Encode real data, per-frame conditioning
            with torch.no_grad():
                mu, _ = vae.encode(real_vec.view(-1, 253))
                # Per-frame conditioning preserves temporal dynamics
                condition = mu.view(1, SEQUENCE_LENGTH, 32)  # (1, T, 32)
                
        elif args.mode == 'gru':
            # GRU Condition: Seed -> Predict
            with torch.no_grad():
                # 1. Get real latents
                z_real_mu, _ = vae.encode(real_vec.view(-1, 253))
                z_real_seq = z_real_mu.view(1, SEQUENCE_LENGTH, 32)
                
                # 2. Normalize for GRU
                z_norm = (z_real_seq - l_mean) / (l_std + 1e-8)
                
                # 3. Predict (Seed 8 steps)
                SEED = 8
                z_seed = z_norm[:, :SEED, :]
                z_pred_norm = gru.generate_sequence(z_seed, SEQUENCE_LENGTH - SEED)
                
                # 4. Concatenate & Denormalize
                z_full_norm = torch.cat([z_seed, z_pred_norm], dim=1)
                z_full = z_full_norm * (l_std + 1e-8) + l_mean
                
                # 5. Per-frame condition preserves temporal dynamics
                condition = z_full  # (1, T, 32)

        # --- Diffusion Generation ---
        with torch.no_grad():
            # Sample (1 sample)
            gen_vec_norm = diffusion.sample_ddim((1, SEQUENCE_LENGTH, 253), condition=condition, steps=50)
            
        # --- Post-process ---
        # Denormalize Data (Norm -> Tangent)
        real_tan = real_vec[0] * stats['std'] + stats['mean']
        gen_tan = gen_vec_norm[0] * stats['std'] + stats['mean']
        
        # Tangent -> SPD Matrices
        real_spd = exp_euclidean_map(vec_to_sym_matrix(real_tan, 22))
        gen_spd = exp_euclidean_map(vec_to_sym_matrix(gen_tan, 22))
        
        # --- Visualize ---
        visualizer.plot_fingerprint(real_spd, gen_spd, sample_idx=i)
        visualizer.plot_connectogram(real_spd, gen_spd, sample_idx=i)
        visualizer.plot_temporal_dynamics(real_spd, gen_spd, sample_idx=i)
        
    print(f"\nDone! Results saved in {visualizer.output_dir}")

if __name__ == "__main__":
    main()