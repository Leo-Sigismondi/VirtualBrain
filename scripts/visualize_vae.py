"""
VAE Reconstruction Visualization
Shows original vs reconstructed covariance matrices to evaluate VAE quality
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
import matplotlib.pyplot as plt
import numpy as np
from src.data.dataset import BCIDataset
from torch.utils.data import DataLoader
from src.models.vae import VAE
from src.preprocessing.geometry_utils import vec_to_sym_matrix

# --- CONFIG ---
CHECKPOINT_PATH = "checkpoints/vae/vae_temporal_latent32_best.pth"  # Use best model
LATENT_DIM = 32 # Match your training config
INPUT_DIM = 253
N_CHANNELS = 22  # Calculated from 253 -> 22*23/2 = 253

def visualize_reconstruction():
    """
    Visualize how well the VAE reconstructs covariance matrices
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load Data
    print("Loading dataset...")
    ds = BCIDataset("data/processed/train")
    dl = DataLoader(ds, batch_size=1, shuffle=True)
    
    # 2. Load Model
    print(f"Loading model from {CHECKPOINT_PATH}...")
    model = VAE(input_dim=INPUT_DIM, latent_dim=LATENT_DIM).to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()
    
    # 3. Load Normalization Stats
    stats_path = f"checkpoints/vae/vae_norm_stats_latent{LATENT_DIM}.npy"
    print(f"Loading normalization stats from {stats_path}...")
    norm_stats = np.load(stats_path, allow_pickle=True).item()
    mean = torch.tensor(norm_stats['mean']).to(device)
    std = torch.tensor(norm_stats['std']).to(device)

    # 4. Get a random sample
    # Dataloader returns sequences (Batch, Seq_Len, Feat). Take first frame.
    seq_vectors, _ = next(iter(dl)) 
    original_vec = seq_vectors[0, 0, :].to(device)  # First frame of sequence
    
    # 5. Pass through VAE
    print("Generating reconstruction...")
    with torch.no_grad():
        # Normalize input
        norm_vec = (original_vec - mean) / std
        
        # Forward pass
        recon_norm, mu, _ = model(norm_vec.unsqueeze(0))  # Add batch dim
        
        # Denormalize output
        recon_vec = (recon_norm * std) + mean
        
    # 6. Reconstruct Matrices (Vector -> Symmetric Tangent Matrix)
    # Note: We're visualizing Tangent Space (Log-Euclidean), not SPD, 
    # but the structure is similar.
    orig_matrix = vec_to_sym_matrix(original_vec, N_CHANNELS).cpu().numpy()
    recon_matrix = vec_to_sym_matrix(recon_vec.squeeze(0), N_CHANNELS).cpu().numpy()
    
    # 6. Plotting
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original
    im1 = axes[0].imshow(orig_matrix, cmap='viridis', interpolation='nearest')
    axes[0].set_title("Original (Tangent Space)", fontsize=14, fontweight='bold')
    axes[0].set_xlabel("Channel")
    axes[0].set_ylabel("Channel")
    fig.colorbar(im1, ax=axes[0])
    
    # Reconstructed
    im2 = axes[1].imshow(recon_matrix, cmap='viridis', interpolation='nearest')
    axes[1].set_title(f"Reconstruction (Latent Dim={LATENT_DIM})", fontsize=14, fontweight='bold')
    axes[1].set_xlabel("Channel")
    axes[1].set_ylabel("Channel")
    fig.colorbar(im2, ax=axes[1])
    
    # Difference (Error)
    diff = np.abs(orig_matrix - recon_matrix)
    mse = np.mean(diff**2)
    im3 = axes[2].imshow(diff, cmap='inferno', interpolation='nearest')
    axes[2].set_title(f"Absolute Difference (MSE: {mse:.4f})", fontsize=14, fontweight='bold')
    axes[2].set_xlabel("Channel")
    axes[2].set_ylabel("Channel")
    fig.colorbar(im3, ax=axes[2])
    
    plt.suptitle(f"VAE Reconstruction Quality (Latent Dim={LATENT_DIM})", 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save figure
    output_path = f"checkpoints/vae/reconstruction_latent{LATENT_DIM}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")
    
    # Print statistics
    print(f"\nReconstruction Quality:")
    print(f"  MSE: {mse:.6f}")
    print(f"  MAE: {np.mean(diff):.6f}")
    print(f"  Max Error: {np.max(diff):.6f}")
    
    plt.show()

def compare_latent_dims():
    """
    Compare reconstructions with different latent dimensions
    Useful for choosing the right latent dimension
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load data
    ds = BCIDataset("data/processed/train")
    dl = DataLoader(ds, batch_size=1, shuffle=True)
    seq_vectors, _ = next(iter(dl))
    original_vec = seq_vectors[0, 0, :].to(device)
    
    # Try different latent dimensions
    latent_dims = [8, 16, 32, 64]
    
    fig, axes = plt.subplots(2, len(latent_dims) + 1, figsize=(20, 8))
    
    # Original in first column
    orig_matrix = vec_to_sym_matrix(original_vec, N_CHANNELS).cpu().numpy()
    for row in range(2):
        axes[row, 0].imshow(orig_matrix, cmap='viridis')
        axes[row, 0].set_title("Original")
        axes[row, 0].axis('off')
    
    # Reconstructions for each latent dim
    for idx, latent_dim in enumerate(latent_dims):
        try:
            checkpoint = f"checkpoints/vae/vae_encoder_latent{latent_dim}.pth"
            model = VAE(input_dim=INPUT_DIM, latent_dim=latent_dim).to(device)
            model.load_state_dict(torch.load(checkpoint, map_location=device))
            model.eval()
            
            with torch.no_grad():
                recon_vec, _, _ = model(original_vec.unsqueeze(0))
            
            recon_matrix = vec_to_sym_matrix(recon_vec.squeeze(0), N_CHANNELS).cpu().numpy()
            diff = np.abs(orig_matrix - recon_matrix)
            mse = np.mean(diff**2)
            
            # Reconstruction
            axes[0, idx + 1].imshow(recon_matrix, cmap='viridis')
            axes[0, idx + 1].set_title(f"Latent={latent_dim}\nMSE={mse:.4f}")
            axes[0, idx + 1].axis('off')
            
            # Difference
            axes[1, idx + 1].imshow(diff, cmap='inferno')
            axes[1, idx + 1].set_title(f"Error")
            axes[1, idx + 1].axis('off')
            
        except FileNotFoundError:
            axes[0, idx + 1].text(0.5, 0.5, f"No model\nfor latent={latent_dim}", 
                                  ha='center', va='center')
            axes[0, idx + 1].axis('off')
            axes[1, idx + 1].axis('off')
    
    plt.tight_layout()
    plt.savefig("checkpoints/vae/latent_dim_comparison.png", dpi=150)
    print("Comparison saved to: checkpoints/vae/latent_dim_comparison.png")
    plt.show()

def visualize_latent_dynamics():
    """
    Visualize latent space trajectories to check for smoothness
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("\n" + "="*60)
    print("Latent Dynamics Visualization")
    print("="*60)
    
    # 1. Load Data
    ds = BCIDataset("data/processed/train")
    dl = DataLoader(ds, batch_size=1, shuffle=True)
    
    # 2. Load Model
    model = VAE(input_dim=INPUT_DIM, latent_dim=LATENT_DIM).to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()
    
    # 3. Load Stats
    stats_path = f"checkpoints/vae/vae_norm_stats_latent{LATENT_DIM}.npy"
    norm_stats = np.load(stats_path, allow_pickle=True).item()
    mean = torch.tensor(norm_stats['mean']).to(device)
    std = torch.tensor(norm_stats['std']).to(device)
    
    # 4. Get a sequence
    seq_vectors, _ = next(iter(dl))
    seq_vectors = seq_vectors.to(device) # (1, Seq_Len, Feat)
    
    # 5. Encode sequence
    latent_seq = []
    with torch.no_grad():
        for t in range(seq_vectors.shape[1]):
            frame = seq_vectors[:, t, :]
            norm_frame = (frame - mean) / std
            mu, _ = model.encode(norm_frame)
            latent_seq.append(mu)
    
    latent_seq = torch.cat(latent_seq, dim=0).cpu().numpy() # (Seq_Len, Latent)
    
    # 6. Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    timesteps = np.arange(latent_seq.shape[0])
    
    # Plot first 5 dimensions
    for i in range(min(5, LATENT_DIM)):
        ax.plot(timesteps, latent_seq[:, i], 'o-', label=f'Dim {i}')
        
    ax.set_title("Latent Trajectories (First 5 Dims)")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Latent Value")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    output_path = "checkpoints/vae/latent_dynamics.png"
    plt.savefig(output_path, dpi=150)
    print(f"Visualization saved to: {output_path}")
    
    # Calculate smoothness metric (mean squared velocity)
    velocity = np.diff(latent_seq, axis=0)
    smoothness = np.mean(velocity**2)
    print(f"Smoothness Metric (lower is smoother): {smoothness:.6f}")

if __name__ == "__main__":
    print("="*60)
    print("VAE Reconstruction Visualization")
    print("="*60)
    visualize_reconstruction()
    visualize_latent_dynamics()
    
    # Uncomment to compare different latent dimensions
    # compare_latent_dims()
