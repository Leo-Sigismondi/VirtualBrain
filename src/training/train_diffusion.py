"""
Training script for Tangent Space Diffusion Model.

This script trains a DDPM diffusion model DIRECTLY in the Log-Euclidean 
tangent space of SPD matrices. No VAE is used.

Pipeline:
    1. Load SPD covariance matrices
    2. Apply log() to get tangent space representation
    3. Train diffusion to denoise noisy tangent vectors
    4. At generation: sample → exp() → guaranteed SPD matrix!
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.diffusion import TangentDiffusion
from src.models.vae import ImprovedVAE
from src.data.data_utils import (
    get_data_loaders, load_normalized_dataset, get_normalization_stats,
    INPUT_DIM, N_CHANNELS, SEQUENCE_LENGTH, encode_to_latent
)
from src.preprocessing.geometry_utils import (
    validate_spd, exp_euclidean_map, vec_to_sym_matrix
)


# =============================================================================
# Configuration
# =============================================================================

# Data dimensions (imported from data_utils for consistency)
# INPUT_DIM = 253, N_CHANNELS = 22, SEQUENCE_LENGTH = 64

# Diffusion config
DIFFUSION_STEPS = 1000
DIFFUSION_HIDDEN_DIM = 512  # Larger for 253-dim input
SCHEDULE = 'cosine'  # 'linear' or 'cosine'

# Conditioning - condition on VAE latent vectors (32-dim)
# At inference, GRU-predicted latents will be used as conditions
CONDITION_DIM = 32  # VAE latent dimension
VAE_PATH = "checkpoints/vae/vae_dynamics_latent32.pth"
VAE_LATENT_DIM = 32

# Training config
BATCH_SIZE = 32  # Smaller due to larger model
EPOCHS = 200
LEARNING_RATE = 1e-4

# Paths
SAVE_DIR = "checkpoints/diffusion"
SAVE_PATH = f"{SAVE_DIR}/tangent_diffusion_best.pth"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =============================================================================
# Training Functions
# =============================================================================

def train_epoch(diffusion, vae, dataloader, optimizer, device, epoch, total_epochs):
    """
    Train diffusion model for one epoch with VAE latent conditioning.
    
    The model learns to generate tangent-space vectors conditioned on VAE latents.
    """
    diffusion.train()
    vae.eval()  # VAE is frozen, only used for encoding
    total_loss = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{total_epochs}", leave=False)
    
    for batch_idx, batch in enumerate(pbar):
        sequences = batch[0].to(device)  # (B, T, D) already in tangent space!
        
        # Encode tangent sequences to get latent conditions
        batch_size, seq_len, input_dim = sequences.shape
        flat_sequences = sequences.view(-1, input_dim)  # (B*T, 253)
        
        with torch.no_grad():
            # Only get mu (don't bother computing logvar)
            latent_mu, _ = vae.encode(flat_sequences)  # (B*T, 32)
            # Per-frame condition preserves temporal dynamics
            condition = latent_mu.view(batch_size, seq_len, -1)  # (B, T, 32)
        
        # Free intermediate tensors
        del flat_sequences, latent_mu
        
        optimizer.zero_grad()
        
        # Diffusion training loss with VAE latent condition
        loss = diffusion.training_loss(sequences, condition=condition)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(diffusion.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Clear CUDA cache periodically to prevent memory buildup
        if batch_idx % 500 == 0:
            torch.cuda.empty_cache()
    
    return total_loss / num_batches


@torch.no_grad()
def evaluate(diffusion, vae, dataloader, device, num_batches=10):
    """
    Evaluate diffusion model by computing validation loss with VAE conditioning.
    """
    diffusion.eval()
    vae.eval()
    total_loss = 0
    count = 0
    
    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break
        sequences = batch[0].to(device)
        
        # Encode for condition - per-frame
        batch_size, seq_len, input_dim = sequences.shape
        flat_sequences = sequences.view(-1, input_dim)
        latent_mu, _ = vae.encode(flat_sequences)
        condition = latent_mu.view(batch_size, seq_len, -1)  # (B, T, 32)
        
        # Free intermediate tensors
        del flat_sequences, latent_mu
        
        loss = diffusion.training_loss(sequences, condition=condition)
        total_loss += loss.item()
        count += 1
    
    return total_loss / count if count > 0 else float('inf')


@torch.no_grad()
def validate_spd_samples(diffusion, device, num_samples=10, seq_len=64):
    """
    Generate samples and validate they produce valid SPD matrices.
    
    Uses random latent vectors as conditions to simulate GRU outputs.
    End-to-end test: sample tangent vectors → reshape → exp() → SPD
    """
    diffusion.eval()
    
    # Create temporally coherent conditions (not fully random per-frame)
    # This mimics real VAE latent trajectories which are smooth over time
    # Start with a base latent and add small temporal perturbations
    base_condition = torch.randn(num_samples, 1, CONDITION_DIM, device=device) * 0.3
    temporal_noise = torch.randn(num_samples, seq_len, CONDITION_DIM, device=device) * 0.1
    # Smooth the temporal noise with cumsum to create coherent trajectories
    temporal_noise = temporal_noise.cumsum(dim=1) * 0.05
    random_conditions = base_condition + temporal_noise  # (B, T, 32)
    
    # Generate samples in tangent space with per-frame conditioning
    samples = diffusion.sample_ddim(
        (num_samples, seq_len, INPUT_DIM), 
        condition=random_conditions,
        device=device, 
        steps=50
    )
    
    all_valid = []
    all_min_eigvals = []
    
    for t in range(seq_len):
        # Reshape to matrix form
        tangent_vecs = samples[:, t, :]  # (B, 253)
        matrices = vec_to_sym_matrix(tangent_vecs, N_CHANNELS)  # (B, 22, 22)
        
        # Apply exp() to ensure SPD
        spd_matrices = exp_euclidean_map(matrices)
        
        # Validate
        is_valid, details = validate_spd(spd_matrices, return_details=True)
        all_valid.append(is_valid)
        all_min_eigvals.append(details['min_eigenvalue'])
    
    valid_tensor = torch.stack(all_valid, dim=1)  # (B, T)
    min_eigvals = torch.stack(all_min_eigvals, dim=1)  # (B, T)
    
    spd_valid_rate = valid_tensor.float().mean().item()
    min_eigenvalue = min_eigvals.min().item()
    
    return spd_valid_rate, min_eigenvalue


# =============================================================================
# Main Training Function
# =============================================================================

def train_diffusion(config=None):
    """
    Main training function for tangent space diffusion.
    """
    if config is None:
        config = {}
    
    epochs = config.get('epochs', EPOCHS)
    lr = config.get('lr', LEARNING_RATE)
    device = config.get('device', DEVICE)
    
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Training Tangent Space Diffusion Model")
    print(f"{'='*60}")
    print(f"Input Dim: {INPUT_DIM} (tangent space)")
    print(f"Diffusion Steps: {DIFFUSION_STEPS}")
    print(f"Schedule: {SCHEDULE}")
    print(f"Hidden Dim: {DIFFUSION_HIDDEN_DIM}")
    print(f"Conditioning: VAE latent ({CONDITION_DIM}-dim)")
    print(f"Device: {device}")
    print(f"{'='*60}\n")
    
    # 1. Load pre-normalized dataset using shared utilities
    print("Loading pre-normalized dataset...")
    train_loader, val_loader, norm_stats = get_data_loaders(batch_size=BATCH_SIZE)
    print(f"[OK] Train: {len(train_loader.dataset)} sequences")
    print(f"[OK] Val: {len(val_loader.dataset)} sequences\n")
    
    # 2. Load pre-trained VAE for conditioning
    print(f"Loading pre-trained VAE from {VAE_PATH}...")
    vae = ImprovedVAE(input_dim=INPUT_DIM, latent_dim=VAE_LATENT_DIM).to(device)
    vae.load_state_dict(torch.load(VAE_PATH, map_location=device))
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False  # Freeze VAE
    print(f"[OK] VAE loaded (latent_dim={VAE_LATENT_DIM})\n")
    
    # 3. Create diffusion model
    print("Creating diffusion model...")
    diffusion = TangentDiffusion(
        tangent_dim=INPUT_DIM,
        hidden_dim=DIFFUSION_HIDDEN_DIM,
        condition_dim=CONDITION_DIM,  # Now using 32-dim VAE latent
        n_steps=DIFFUSION_STEPS,
        schedule=SCHEDULE
    ).to(device)
    
    total_params = sum(p.numel() for p in diffusion.parameters())
    print(f"[OK] Model created with {total_params:,} parameters\n")
    
    # Optimizer
    optimizer = optim.AdamW(diffusion.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    # 4. Training loop
    print("Starting training...\n")
    best_val_loss = float('inf')
    patience = 0
    max_patience = 30
    
    history = {'train_loss': [], 'val_loss': [], 'spd_valid_rate': []}
    
    for epoch in range(epochs):
        # Train
        train_loss = train_epoch(
            diffusion, vae, train_loader, 
            optimizer, device, epoch, epochs
        )
        
        # Evaluate
        val_loss = evaluate(diffusion, vae, val_loader, device)
        
        # SPD validation on small sample every 10 epochs
        if epoch % 10 == 0 or epoch == epochs - 1:
            spd_rate, min_eigval = validate_spd_samples(diffusion, device, num_samples=10)
            print(f"         SPD Valid: {spd_rate*100:.1f}% | Min Eigval: {min_eigval:.6f}")
        else:
            spd_rate, min_eigval = history['spd_valid_rate'][-1] if history['spd_valid_rate'] else 1.0, 0.0
        
        scheduler.step()
        
        # Track history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['spd_valid_rate'].append(spd_rate)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(diffusion.state_dict(), SAVE_PATH)
            patience = 0
        else:
            patience += 1
        
        # Print progress
        if (epoch + 1) % 5 == 0 or epoch < 5:
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train: {train_loss:.4f} | "
                  f"Val: {val_loss:.4f} | "
                  f"Best: {best_val_loss:.4f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Early stopping
        if patience >= max_patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    # Save history
    np.save(f"{SAVE_DIR}/training_history_tangent.npy", history)
    
    print(f"\n{'='*60}")
    print(f"[OK] Training Complete!")
    print(f"Best Validation Loss: {best_val_loss:.6f}")
    print(f"Model saved to: {SAVE_PATH}")
    print(f"{'='*60}\n")
    
    return best_val_loss, SAVE_PATH


if __name__ == "__main__":
    train_diffusion()
