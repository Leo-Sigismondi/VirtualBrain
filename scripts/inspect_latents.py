import torch
import numpy as np
import os
import sys
from pathlib import Path
from torch.utils.data import DataLoader

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.models.vae import VAE
from src.data.dataset import BCIDataset

def inspect_actual_latents():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    VAE_PATH = "checkpoints/vae/vae_temporal_latent32_best.pth"
    LATENT_DIM = 32
    INPUT_DIM = 253
    BATCH_SIZE = 1024 # Large batch for good stats

    print(f"Loading VAE from {VAE_PATH}...")
    vae = VAE(input_dim=INPUT_DIM, latent_dim=LATENT_DIM).to(device)
    try:
        vae.load_state_dict(torch.load(VAE_PATH, map_location=device))
    except FileNotFoundError:
        print("Model file not found!")
        return
    vae.eval()

    print("Loading dataset...")
    # We only need a subset to check stats
    dataset = BCIDataset("data/processed/train", sequence_length=64)
    # Just take the first 5000 samples approx
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Load input normalization stats (to normalize input correctly)
    stats_path = "checkpoints/vae/vae_norm_stats_latent32.npy"
    norm_stats = np.load(stats_path, allow_pickle=True).item()
    input_mean = torch.tensor(norm_stats['mean']).to(device)
    input_std = torch.tensor(norm_stats['std']).to(device)

    print("Encoding data...")
    all_mus = []
    
    with torch.no_grad():
        for i, (batch_seq, _) in enumerate(dataloader):
            if i > 10: break # Check ~10k samples
            
            batch_seq = batch_seq.to(device)
            # Flatten: (B, T, D) -> (B*T, D)
            x = batch_seq.view(-1, INPUT_DIM)
            
            # Normalize
            x = (x - input_mean) / input_std
            
            mu, _ = vae.encode(x)
            all_mus.append(mu.cpu())
            
    latents = torch.cat(all_mus, dim=0).numpy()
    print(f"Encoded {latents.shape[0]} samples.")

    # Calculate stats
    mean = latents.mean(axis=0)
    std = latents.std(axis=0)
    
    print("\nACTUAL LATENT SPACE STATISTICS (32 Dims)")
    print(f"{'Dim':<5} | {'Mean':<10} | {'Std':<10}")
    print("-" * 30)
    
    active_dims = 0
    # Threshold for "active": distinct from 0 (if trained with KL, 'inactive' usually means std~1 but mean~0, 
    # OR if posterior collapse, it matches prior N(0,1). 
    # Actually, posterior collapse usually means the posterior ignore input, so mu(x) is constant -> std=0.
    # Wait, if mode collapses to prior, std should be close to 0? No, if it matches prior N(0,1), std is 1.
    # But usually "collapse" in VAE means the encoder outputs u=0, sigma=1 always. 
    # If the encoder simply ignores input, all x map to same mu. So std(mu) across dataset = 0.
    
    for i in range(LATENT_DIM):
        is_active = std[i] > 0.01 # Arbitrary small threshold
        marker = "*" if std[i] > 0.1 else ("." if std[i] > 0.01 else "DEAD")
        if std[i] > 0.01: active_dims += 1
        print(f"{i:<5} | {mean[i]:<10.4f} | {std[i]:<10.4f} {marker}")
        
    print("-" * 30)
    print(f"Active Dimensions (>0.01 std): {active_dims}/{LATENT_DIM}")
    
    if active_dims < 5:
        print("\nWARNING: Posterior Collapse likely! Most dimensions have near-zero variance across the dataset.")

if __name__ == "__main__":
    inspect_actual_latents()
