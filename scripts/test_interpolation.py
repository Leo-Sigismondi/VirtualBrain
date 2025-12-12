"""
Latent Space Interpolation Test
"Dynamic Validation" - Demonstrates smooth transitions between brain states
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from src.preprocessing.geometry_utils import vec_to_sym_matrix, exp_euclidean_map
from src.models.vae import ImprovedVAE
from src.models.diffusion import TangentDiffusion
from src.data.data_utils import get_normalization_stats, load_normalized_dataset, encode_to_latent

# Config
VAE_PATH = "checkpoints/vae/vae_dynamics_latent32_best.pth"
DIFFUSION_PATH = "checkpoints/diffusion/tangent_diffusion_best.pth"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

NODE_NAMES = ['Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz', 
              'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz', 
              'P2', 'POz']

def load_models(device):
    print("Loading models...")
    vae = ImprovedVAE(input_dim=253, latent_dim=32, hidden_dims=[256, 128, 64]).to(device)
    vae.load_state_dict(torch.load(VAE_PATH, map_location=device))
    vae.eval()
    
    diffusion = TangentDiffusion(tangent_dim=253, condition_dim=32).to(device)
    diffusion.load_state_dict(torch.load(DIFFUSION_PATH, map_location=device))
    diffusion.eval()
    print("✓ Models loaded")
    return vae, diffusion

def interpolate_latents(z_start, z_end, num_steps=10):
    """Linear interpolation between two latent vectors."""
    alphas = torch.linspace(0, 1, num_steps, device=z_start.device)
    z_interpolated = []
    for alpha in alphas:
        z_t = (1 - alpha) * z_start + alpha * z_end
        z_interpolated.append(z_t)
    return torch.stack(z_interpolated, dim=0)  # (num_steps, latent_dim)

def plot_connectome_simple(con_matrix, title, ax):
    """Simplified connectome plot without MNE for reliability."""
    np.fill_diagonal(con_matrix, 0)
    
    # Create circular layout
    n_nodes = len(NODE_NAMES)
    angles = np.linspace(0, 2*np.pi, n_nodes, endpoint=False)
    
    # Plot nodes
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.axis('off')
    
    node_x = np.cos(angles)
    node_y = np.sin(angles)
    
    # Plot connections (top 20% strongest)
    threshold = np.percentile(np.abs(con_matrix), 80)
    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            strength = abs(con_matrix[i, j])
            if strength > threshold:
                alpha = min(1.0, strength / con_matrix.max())
                ax.plot([node_x[i], node_x[j]], [node_y[i], node_y[j]], 
                       'b-', alpha=alpha*0.7, linewidth=1)
    
    # Plot nodes
    ax.scatter(node_x, node_y, s=100, c='red', zorder=5)
    for i, name in enumerate(NODE_NAMES):
        ax.annotate(name, (node_x[i]*1.15, node_y[i]*1.15), 
                   ha='center', va='center', fontsize=6)
    ax.set_title(title, fontsize=10)

def main():
    print("="*60)
    print("Latent Interpolation - Dynamic Validation")
    print("="*60)
    
    vae, diffusion = load_models(DEVICE)
    
    # Load two different sequences
    print("Loading data...")
    from src.data.dataset import BCIDataset
    from torch.utils.data import DataLoader
    dataset = BCIDataset("data/processed/train", sequence_length=64)
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    seqs, _ = next(iter(loader))
    seq1 = seqs[0:1].to(DEVICE)
    seq2 = seqs[1:2].to(DEVICE)
    
    # Norm stats
    stats = get_normalization_stats("data/processed/normalization_stats.npy")
    mu = torch.tensor(stats['mean']).to(DEVICE)
    std = torch.tensor(stats['std']).to(DEVICE)
    
    # Encode to latent
    with torch.no_grad():
        z1 = encode_to_latent(vae, seq1, DEVICE).to(DEVICE)  # (1, 64, 32)
        z2 = encode_to_latent(vae, seq2, DEVICE).to(DEVICE)
        
        # Use specific time points
        z_start = z1[0, 32, :]  # Middle of sequence 1 (32,)
        z_end = z2[0, 32, :]    # Middle of sequence 2
        
        print(f"Z_start stats: mean={z_start.mean():.4f}, std={z_start.std():.4f}")
        print(f"Z_end stats: mean={z_end.mean():.4f}, std={z_end.std():.4f}")
        
        # Interpolate
        num_interp = 8
        z_interp = interpolate_latents(z_start, z_end, num_interp)  # (8, 32)
        
        print(f"Generating {num_interp} interpolated samples...")
        
        # Generate for each interpolated latent
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        for idx in range(num_interp):
            cond = z_interp[idx:idx+1]  # (1, 32)
            
            # Generate tangent vector
            x_0 = diffusion.sample_ddim(shape=(1, 1, 253), condition=cond, steps=50)
            
            # Denormalize
            x_0_denorm = x_0[0, 0] * std + mu
            
            # Convert to SPD
            mat_tangent = vec_to_sym_matrix(x_0_denorm, n=22)
            mat_spd = exp_euclidean_map(mat_tangent)
            con_matrix = mat_spd.cpu().numpy()
            
            # Plot
            alpha = idx / (num_interp - 1)
            plot_connectome_simple(con_matrix, f"α = {alpha:.2f}", axes[idx])
        
        plt.suptitle("Latent Space Interpolation: State A → State B", fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        out_path = "checkpoints/latent_interpolation.png"
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {out_path}")
        plt.close()
        
        # Eigenvalue stability plot
        print("\\nGenerating eigenvalue stability plot...")
        eigenvalues_over_time = []
        
        for idx in range(num_interp):
            cond = z_interp[idx:idx+1]
            x_0 = diffusion.sample_ddim(shape=(1, 1, 253), condition=cond, steps=50)
            x_0_denorm = x_0[0, 0] * std + mu
            
            mat_tangent = vec_to_sym_matrix(x_0_denorm, n=22)
            mat_spd = exp_euclidean_map(mat_tangent)
            eigvals = torch.linalg.eigvalsh(mat_spd).cpu().numpy()
            eigenvalues_over_time.append(eigvals)
        
        eigenvalues_over_time = np.array(eigenvalues_over_time)  # (num_interp, 22)
        
        plt.figure(figsize=(10, 6))
        for i in range(22):
            plt.plot(range(num_interp), eigenvalues_over_time[:, i], 'o-', alpha=0.5, markersize=4)
        plt.xlabel("Interpolation Step (α)")
        plt.ylabel("Eigenvalue")
        plt.title("Eigenvalue Stability During Interpolation\\n(All 22 eigenvalues should vary smoothly)")
        plt.grid(True, alpha=0.3)
        
        out_path2 = "checkpoints/eigenvalue_stability.png"
        plt.savefig(out_path2, dpi=150, bbox_inches='tight')
        print(f"Saved: {out_path2}")
        plt.close()
        
    print("\\n" + "="*60)
    print("Interpolation Test Complete!")
    print("="*60)

if __name__ == "__main__":
    main()
