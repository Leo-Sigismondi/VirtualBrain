"""
Visualize Generated Brain States using MNE Connectivity Circles
"The Brain" - Visual Validation for Thesis
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
import numpy as np
import matplotlib.pyplot as plt
from mne_connectivity.viz import plot_connectivity_circle
from src.preprocessing.geometry_utils import vec_to_sym_matrix, exp_euclidean_map
from src.models.vae import ImprovedVAE
from src.models.gru import TemporalGRU
from src.models.diffusion import TangentDiffusion
from src.data.data_utils import get_normalization_stats, load_normalized_dataset, encode_to_latent

# Config
VAE_PATH = "checkpoints/vae/vae_dynamics_latent32_best.pth"
GRU_PATH = "checkpoints/gru/gru_multistep_L32_H128_2L_best.pth"
DIFFUSION_PATH = "checkpoints/diffusion/tangent_diffusion_best.pth"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

NODE_NAMES = ['Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz', 
              'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz', 
              'P2', 'POz']

def load_models_full(device):
    print("Loading models...")
    vae = ImprovedVAE(input_dim=253, latent_dim=32, hidden_dims=[256, 128, 64]).to(device)
    vae.load_state_dict(torch.load(VAE_PATH, map_location=device))
    vae.eval()
    
    gru = TemporalGRU(latent_dim=32, hidden_dim=128, num_layers=2).to(device)
    gru.load_state_dict(torch.load(GRU_PATH, map_location=device))
    gru.eval()
    
    # CORRECT: TangentDiffusion uses tangent_dim, not input_dim
    diffusion = TangentDiffusion(tangent_dim=253, condition_dim=32).to(device)
    diffusion.load_state_dict(torch.load(DIFFUSION_PATH, map_location=device))
    diffusion.eval()
    print("✓ Models loaded")
    return vae, gru, diffusion

def plot_generated_connectome(tangent_vector, title="Generated Brain State"):
    """
    Visualize a tangent vector as a connectivity circle using MNE.
    tangent_vector: (253,) torch tensor or numpy array
    """
    if isinstance(tangent_vector, np.ndarray):
        tangent_vector = torch.from_numpy(tangent_vector).float()
    
    # Vector to SPD Matrix
    mat_tangent = vec_to_sym_matrix(tangent_vector, n=22)
    mat_spd = exp_euclidean_map(mat_tangent)
    con_matrix = mat_spd.cpu().numpy()
    
    # Zero diagonal to focus on connectivity
    con_matrix_viz = con_matrix.copy()
    np.fill_diagonal(con_matrix_viz, 0)
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    plot_connectivity_circle(con_matrix_viz, NODE_NAMES, n_lines=50, 
                             node_angles=None, node_colors=None,
                             title=title, ax=ax, show=False,
                             vmin=con_matrix_viz.min(), vmax=con_matrix_viz.max(),
                             facecolor='white', textcolor='black')
    return fig

def main():
    print("="*60)
    print("Creating 'Vital Signs' - Brain Connectograms")
    print("="*60)
    
    vae, gru, diffusion = load_models_full(DEVICE)
    
    # Get one sample from dataset to seed the GRU
    print("Getting seed data...")
    from src.data.dataset import BCIDataset
    from torch.utils.data import DataLoader
    dataset = BCIDataset("data/processed/train", sequence_length=64)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    real_seq, _ = next(iter(loader))
    real_seq = real_seq.to(DEVICE)
    
    # Norm stats
    stats = get_normalization_stats("data/processed/normalization_stats.npy")
    mu = torch.tensor(stats['mean']).to(DEVICE)
    std = torch.tensor(stats['std']).to(DEVICE)
    
    with torch.no_grad():
        z_seq_raw = encode_to_latent(vae, real_seq, DEVICE).to(DEVICE)
        
        # Normalize for GRU
        l_stats = np.load("checkpoints/gru/latent_norm_stats_multistep.npy", allow_pickle=True).item()
        l_mean = torch.tensor(l_stats['mean']).to(DEVICE)
        l_std = torch.tensor(l_stats['std']).to(DEVICE)
        z_seq_norm = (z_seq_raw - l_mean) / (l_std + 1e-8)
        
        # Predict using GRU
        z_seed = z_seq_norm[:, :5, :]
        z_pred_norm = gru.generate_sequence(z_seed, 59)
        
        # Combine and denormalize
        z_full_norm = torch.cat([z_seed, z_pred_norm], dim=1)
        z_full = z_full_norm * (l_std + 1e-8) + l_mean
        
        # Generate at multiple time points
        times_to_plot = [10, 40]
        
        for t in times_to_plot:
            print(f"Generating for t={t}...")
            cond = z_full[:, min(t, 63), :].view(1, 32)
            
            # Generate tangent vector (shape: batch, seq_len, tangent_dim)
            x_0 = diffusion.sample_ddim(shape=(1, 1, 253), condition=cond, steps=50)
            x_0_denorm = x_0[0, 0] * std + mu  # (253,)
            
            # Plot
            fig = plot_generated_connectome(x_0_denorm, title=f"Generated Brain State (t={t})")
            
            out_path = f"checkpoints/generated_connectome_t{t}.png"
            fig.savefig(out_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved: {out_path}")
            plt.close(fig)
            
    print("\nVisualization Complete!")

if __name__ == "__main__":
    main()
