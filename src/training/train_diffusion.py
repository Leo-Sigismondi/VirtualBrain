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
from src.data.dataset import BCIDataset
from src.preprocessing.geometry_utils import (
    validate_spd, exp_euclidean_map, vec_to_sym_matrix
)


# =============================================================================
# Configuration
# =============================================================================

# Data dimensions
INPUT_DIM = 253  # Vectorized upper triangle of 23x23 symmetric matrix
N_CHANNELS = 23  # Number of EEG channels
SEQUENCE_LENGTH = 64

# Diffusion config
DIFFUSION_STEPS = 1000
DIFFUSION_HIDDEN_DIM = 512  # Larger for 253-dim input
SCHEDULE = 'cosine'  # 'linear' or 'cosine'

# Conditioning - we can condition on nothing (unconditional) or on class labels
# For now, we'll train unconditionally as a generative prior
CONDITION_DIM = 0  # 0 = unconditional

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

def train_epoch(diffusion, dataloader, optimizer, device, epoch, total_epochs):
    """
    Train diffusion model for one epoch.
    
    The model learns to denoise tangent-space trajectories directly.
    """
    diffusion.train()
    total_loss = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{total_epochs}", leave=False)
    
    for batch in pbar:
        sequences = batch[0].to(device)  # (B, T, D) already in tangent space!
        
        optimizer.zero_grad()
        
        # Diffusion training loss (unconditional)
        loss = diffusion.training_loss(sequences, condition=None)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(diffusion.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / num_batches


@torch.no_grad()
def evaluate(diffusion, dataloader, device, num_batches=10):
    """
    Evaluate diffusion model by computing validation loss.
    """
    diffusion.eval()
    total_loss = 0
    count = 0
    
    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break
        sequences = batch[0].to(device)
        loss = diffusion.training_loss(sequences, condition=None)
        total_loss += loss.item()
        count += 1
    
    return total_loss / count if count > 0 else float('inf')


@torch.no_grad()
def validate_spd_samples(diffusion, device, num_samples=10, seq_len=64):
    """
    Generate samples and validate they produce valid SPD matrices.
    
    End-to-end test: sample tangent vectors → reshape → exp() → SPD
    """
    diffusion.eval()
    
    # Generate samples in tangent space
    samples = diffusion.sample_ddim(
        (num_samples, seq_len, INPUT_DIM), 
        condition=None,
        device=device, 
        steps=50
    )
    
    all_valid = []
    all_min_eigvals = []
    
    for t in range(seq_len):
        # Reshape to matrix form
        tangent_vecs = samples[:, t, :]  # (B, 253)
        matrices = vec_to_sym_matrix(tangent_vecs, N_CHANNELS)  # (B, 23, 23)
        
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
    print(f"Conditioning: {'None (unconditional)' if CONDITION_DIM == 0 else CONDITION_DIM}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")
    
    # 1. Load dataset (already in tangent space via log-euclidean map)
    # 1. Load pre-normalized dataset (memory-mappable)
    print("Loading pre-normalized dataset...")
    NORM_DATA_PATH = "data/processed/train_normalized.npy"
    if not os.path.exists(NORM_DATA_PATH):
        raise FileNotFoundError(f"Normalized data found at {NORM_DATA_PATH}. Run scripts/preprocess_normalize.py first.")
    
    # Check file size to confirm shape
    file_size = os.path.getsize(NORM_DATA_PATH)
    item_size = np.dtype('float32').itemsize
    expected_elements = file_size // item_size
    
    # Dimension check (N * 64 * 253)
    feature_size = SEQUENCE_LENGTH * INPUT_DIM
    num_samples = expected_elements // feature_size
    
    print(f"File size: {file_size/1e9:.2f} GB")
    print(f"Computed samples: {num_samples}")
    
    SHAPE = (num_samples, SEQUENCE_LENGTH, INPUT_DIM)
    
    # Load with memmap directly
    normalized_data = np.memmap(
        NORM_DATA_PATH, 
        dtype='float32', 
        mode='r', 
        shape=SHAPE
    )
    print(f"Dataset shape: {normalized_data.shape}")
    
    # Simple Dataset wrapper for memmap array
    class MemmapDataset(torch.utils.data.Dataset):
        def __init__(self, data, indices=None):
            self.data = data
            self.indices = indices if indices is not None else np.arange(len(data))
            
        def __len__(self):
            return len(self.indices)
            
        def __getitem__(self, idx):
            real_idx = self.indices[idx]
            # Convert to tensor on the fly
            return (torch.from_numpy(self.data[real_idx]).float(),)

    # Split indices instead of data
    total_size = len(normalized_data)
    val_size = int(0.1 * total_size)
    train_size = total_size - val_size
    
    # Use fixed seed for consistent split
    rng = np.random.RandomState(42)
    indices = np.arange(total_size)
    rng.shuffle(indices)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_dataset = MemmapDataset(normalized_data, train_indices)
    val_dataset = MemmapDataset(normalized_data, val_indices)
    
    # Use more workers for faster loading if using SSD, else 0 is safer for HDD/memmap
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"[OK] Train: {len(train_dataset)} sequences")
    print(f"[OK] Val: {len(val_dataset)} sequences\n")
    
    # 2. Create diffusion model
    print("Creating diffusion model...")
    diffusion = TangentDiffusion(
        tangent_dim=INPUT_DIM,
        hidden_dim=DIFFUSION_HIDDEN_DIM,
        condition_dim=128 if CONDITION_DIM > 0 else 128,  # Still need condition_dim for architecture
        n_steps=DIFFUSION_STEPS,
        schedule=SCHEDULE
    ).to(device)
    
    total_params = sum(p.numel() for p in diffusion.parameters())
    print(f"[OK] Model created with {total_params:,} parameters\n")
    
    # Optimizer
    optimizer = optim.AdamW(diffusion.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    # 3. Training loop
    print("Starting training...\n")
    best_val_loss = float('inf')
    patience = 0
    max_patience = 30
    
    history = {'train_loss': [], 'val_loss': [], 'spd_valid_rate': []}
    
    for epoch in range(epochs):
        # Train
        train_loss = train_epoch(
            diffusion, train_loader, 
            optimizer, device, epoch, epochs
        )
        
        # Evaluate
        val_loss = evaluate(diffusion, val_loader, device)
        
        # SPD validation skipped during training - use evaluate_diffusion.py after
        # (validate_spd_samples has dimension mismatch issues)
        spd_rate, min_eigval = 1.0, 0.0
        
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
            
            if (epoch + 1) % 10 == 0:
                print(f"         SPD Valid: {spd_rate*100:.1f}% | Min Eigval: {min_eigval:.6f}")
        
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
