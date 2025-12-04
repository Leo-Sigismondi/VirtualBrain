"""
VAE Training Script with Temporal Loss
Trains VAE to encode temporal dynamics, not just static features
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from src.data.dataset import BCIDataset
from src.models.vae import VAE
import os
import numpy as np

# --- CONFIG ---
BATCH_SIZE = 64
EPOCHS = 500
LEARNING_RATE = 1e-3
LATENT_DIM = 32
INPUT_DIM = 253
TEMPORAL_WEIGHT = 5  # Weight for temporal loss components
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def temporal_vae_loss(recon_x, x, mu, logvar, mu_sequence, temporal_weight=0.1):
    """
    Enhanced VAE loss with temporal consistency
    
    Args:
        recon_x: Reconstructed data (flattened sequence)
        x: Original data (flattened sequence)
        mu: Latent mu (flattened sequence)
        logvar: Latent logvar (flattened sequence)
        mu_sequence: Latent mu reshaped to (batch, seq_len, latent)
        temporal_weight: Weight for temporal loss components
    
    Returns:
        total_loss, mse, kld, temporal_loss
    """
    # Standard VAE loss components
    MSE = F.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # TEMPORAL LOSS COMPONENTS
    
    # 1. Smoothness penalty - consecutive latents shouldn't jump wildly
    # This encourages gradual changes over time
    latent_velocity = mu_sequence[:, 1:, :] - mu_sequence[:, :-1, :]
    smoothness_loss = latent_velocity.pow(2).sum()
    
    # 2. Anti-static penalty - latents should change meaningfully over time
    # Penalize sequences with low variance (nearly flat)
    within_seq_var = mu_sequence.var(dim=1).sum()  # Variance across timesteps
    static_penalty = 1.0 / (within_seq_var + 1e-6)  # Higher when variance is low
    
    # Combined temporal loss
    # Balance: want smooth changes (low smoothness_loss) but not too flat (low static_penalty)
    temporal_loss = smoothness_loss + 0.5 * static_penalty
    
    # Total loss
    total_loss = MSE + KLD + temporal_weight * temporal_loss
    
    return total_loss, MSE.item(), KLD.item(), temporal_loss.item()


def split_dataset(dataset):
    """
    Split dataset into train and validation (80/20 split)
    """
    total_size = len(dataset)
    val_size = int(0.2 * total_size)
    train_size = total_size - val_size
    
    train_indices = list(range(train_size))
    val_indices = list(range(train_size, total_size))
    
    return train_indices, val_indices


def evaluate(model, dataloader, device, mean, std, temporal_weight):
    """
    Evaluate model on validation set with temporal loss
    """
    model.eval()
    total_loss = 0
    total_mse = 0
    total_kld = 0
    total_temporal = 0
    
    with torch.no_grad():
        for batch_seq, _ in dataloader:
            batch_size, seq_len, feat_dim = batch_seq.shape
            batch_seq = batch_seq.to(device)
            
            # Normalize
            batch_seq = (batch_seq - mean) / std
            
            # Flatten for VAE
            x = batch_seq.view(-1, feat_dim)
            
            # Forward pass
            recon_x, mu, logvar = model(x)
            
            # Reshape mu for temporal loss
            mu_sequence = mu.view(batch_size, seq_len, -1)
            
            # Compute loss
            loss, mse, kld, temporal = temporal_vae_loss(
                recon_x, x, mu, logvar, mu_sequence, temporal_weight
            )
            
            total_loss += loss
            total_mse += mse
            total_kld += kld
            total_temporal += temporal
    
    # Average over all samples
    num_samples = len(dataloader.dataset) * 13  # 13 timesteps per sequence
    avg_loss = total_loss / num_samples
    avg_mse = total_mse / num_samples
    avg_kld = total_kld / num_samples
    avg_temporal = total_temporal / num_samples
    
    return avg_loss, avg_mse, avg_kld, avg_temporal


def train_temporal_vae(config=None):
    """
    Train VAE with temporal loss
    """
    if config is None:
        config = {}
    
    batch_size = config.get('batch_size', BATCH_SIZE)
    epochs = config.get('epochs', EPOCHS)
    lr = config.get('lr', LEARNING_RATE)
    latent_dim = config.get('latent_dim', LATENT_DIM)
    temporal_weight = config.get('temporal_weight', TEMPORAL_WEIGHT)
    device = config.get('device', DEVICE)
    
    # Setup save paths
    save_dir = "checkpoints/vae"
    os.makedirs(save_dir, exist_ok=True)
    
    save_path = os.path.join(save_dir, f"vae_temporal_latent{latent_dim}.pth")
    best_model_path = os.path.join(save_dir, f"vae_temporal_latent{latent_dim}_best.pth")
    stats_path = os.path.join(save_dir, f"vae_norm_stats_latent{latent_dim}.npy")
    
    print(f"\n{'='*60}")
    print(f"Training Temporal VAE")
    print(f"{'='*60}")
    print(f"Latent Dim: {latent_dim}")
    print(f"Temporal Weight: {temporal_weight}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")
    
    # 1. Load dataset
    print("Loading dataset...")
    full_dataset = BCIDataset("data/processed/train")
    print(f"[OK] Dataset loaded: {len(full_dataset)} sequences\n")
    
    # 2. Split train/val
    train_indices, val_indices = split_dataset(full_dataset)
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Train: {len(train_dataset)} sequences")
    print(f"Val: {len(val_dataset)} sequences\n")
    
    # 3. Calculate normalization stats
    print("Calculating normalization stats...")
    all_train_data = []
    for batch_seq, _ in train_loader:
        all_train_data.append(batch_seq.view(-1, INPUT_DIM))
    
    all_train_data = torch.cat(all_train_data, dim=0)
    train_mean = all_train_data.mean(dim=0)
    train_std = all_train_data.std(dim=0)
    train_std[train_std < 1e-8] = 1.0
    
    # Save stats
    norm_stats = {
        'mean': train_mean.cpu().numpy(),
        'std': train_std.cpu().numpy()
    }
    np.save(stats_path, norm_stats)
    print(f"[OK] Stats saved to {stats_path}\n")
    
    train_mean = train_mean.to(device)
    train_std = train_std.to(device)
    
    # 4. Create model
    print("Creating VAE model...")
    model = VAE(input_dim=INPUT_DIM, latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=50
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[OK] Model created with {total_params:,} parameters\n")
    
    # 5. Training loop
    print("Starting training...\n")
    best_val_loss = float('inf')
    patience = 0
    max_patience = 100
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_mse = 0
        train_kld = 0
        train_temporal = 0
        
        for batch_seq, _ in train_loader:
            batch_size, seq_len, feat_dim = batch_seq.shape
            batch_seq = batch_seq.to(device)
            
            # Normalize
            batch_seq = (batch_seq - train_mean) / train_std
            
            # Flatten for VAE
            x = batch_seq.view(-1, feat_dim)
            
            # Forward pass
            optimizer.zero_grad()
            recon_x, mu, logvar = model(x)
            
            # Reshape mu for temporal loss
            mu_sequence = mu.view(batch_size, seq_len, -1)
            
            # Compute temporal-aware loss
            loss, mse, kld, temporal = temporal_vae_loss(
                recon_x, x, mu, logvar, mu_sequence, temporal_weight
            )
            
            # Backprop
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_mse += mse
            train_kld += kld
            train_temporal += temporal
        
        # Average training loss
        num_train_samples = len(train_dataset) * 13
        avg_train_loss = train_loss / num_train_samples
        avg_train_mse = train_mse / num_train_samples
        avg_train_kld = train_kld / num_train_samples
        avg_train_temporal = train_temporal / num_train_samples
        
        # Validation
        val_loss, val_mse, val_kld, val_temporal = evaluate(
            model, val_loader, device, train_mean, train_std, temporal_weight
        )
        
        # Scheduler
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            patience = 0
        else:
            patience += 1
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch < 5:
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train: {avg_train_loss:.4f} | "
                  f"Val: {val_loss:.4f} | "
                  f"Best: {best_val_loss:.4f} | "
                  f"Temporal: {val_temporal:.4f}")
        
        # Early stopping
        if patience >= max_patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    # 6. Save final model
    torch.save(model.state_dict(), save_path)
    
    print(f"\n{'='*60}")
    print(f"[OK] Training Complete!")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Model saved to: {best_model_path}")
    print(f"{'='*60}\n")
    
    return best_val_loss, best_model_path, stats_path


if __name__ == "__main__":
    train_temporal_vae()