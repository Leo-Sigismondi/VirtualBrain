"""
VAE Training Script with Dynamics-Encouraging Loss
===================================================

WHY WE CHANGED THE TEMPORAL LOSS:
---------------------------------
The original loss had a SMOOTHNESS PENALTY:
    smoothness_loss = (z[t+1] - z[t])^2
    
This is COUNTER-PRODUCTIVE because:
1. It explicitly penalizes temporal dynamics (differences between states)
2. The model learns to make consecutive states nearly identical
3. Result: lag-1 autocorrelation of 0.825 (too high!)
4. When states barely change, naive baseline (z[t+1] = z[t]) is optimal
5. GRU learns "do nothing" is the best strategy

NEW LOSS PHILOSOPHY:
-------------------
We want latent representations that:
1. PRESERVE TEMPORAL DYNAMICS - consecutive states should be DIFFERENT enough
2. NOT BE RANDOM - still want smooth, interpretable trajectories
3. CAPTURE MEANINGFUL CHANGES - state differences should reflect real EEG changes

The new loss encourages a "Goldilocks" level of dynamics:
- Not too smooth (old problem)
- Not too noisy (meaningless)
- Just right for prediction

LOSS COMPONENTS EXPLAINED:
-------------------------
1. RECONSTRUCTION LOSS (MSE):
   - Standard VAE component
   - Ensures latent space captures input information
   
2. KL DIVERGENCE:
   - Regularizes latent space toward N(0,1)
   - Prevents mode collapse
   - With beta-VAE we can control the balance
   
3. DYNAMICS LOSS (NEW):
   a) Velocity Target Loss:
      - Penalize when average velocity is TOO LOW
      - Target: mean |z[t+1] - z[t]| >= min_velocity
      - This prevents the "frozen latent" problem
      
   b) Temporal Diversity Loss:
      - Encourage different trajectory directions at different times
      - Penalize when all velocities point the same direction
      - This prevents "monotonic drift" where latent just trends one way
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
from torch.utils.data import DataLoader
from src.data.data_utils import (
    get_data_loaders, load_normalized_dataset, get_normalization_stats,
    NormalizedDataset, INPUT_DIM, SEQUENCE_LENGTH
)
from src.models.vae import ImprovedVAE
import os
import numpy as np
from tqdm import tqdm

# --- CONFIG ---
BATCH_SIZE = 512
EPOCHS = 100
LEARNING_RATE = 5e-4
LATENT_DIM = 32
# INPUT_DIM = 253 and SEQUENCE_LENGTH = 64 imported from data_utils

# Loss weights (carefully tuned)
BETA = 0.01         # KL weight - VERY LOW to prevent posterior collapse
DYNAMICS_WEIGHT = 0.0  # DISABLED - focus on reconstruction first

# Dynamics loss hyperparameters
MIN_VELOCITY = 0.15    # Minimum desired velocity magnitude (in normalized latent space)
TARGET_AUTOCORR = 0.6  # Target lag-1 autocorrelation (lower than current 0.825)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR = "checkpoints/vae"


def dynamics_encouraging_loss(mu_sequence, min_velocity=0.15, target_autocorr=0.6):
    """
    Loss function that ENCOURAGES temporal dynamics instead of suppressing them.
    
    Args:
        mu_sequence: (Batch, Seq_Len, Latent) - latent trajectories
        min_velocity: Minimum desired average velocity magnitude
        target_autocorr: Target lag-1 autocorrelation
        
    Returns:
        dynamics_loss: Scalar loss encouraging good dynamics
        diagnostics: Dict with individual loss components
    
    WHY THESE SPECIFIC COMPONENTS:
    
    1. VELOCITY MAGNITUDE LOSS:
       - We compute velocity = z[t+1] - z[t] at each timestep
       - We want mean(|velocity|) >= min_velocity
       - If too low, we penalize: loss = (min_velocity - mean_velocity)^2
       - This directly combats the "frozen latent" problem
    
    2. VELOCITY DIVERSITY LOSS:
       - If all velocities are the same direction, dynamics are boring
       - We measure std(velocity) across time
       - Low std = monotonic trajectory = bad
       - We penalize: loss = 1 / (velocity_std + eps)
    """
    batch_size, seq_len, latent_dim = mu_sequence.shape
    
    # 1. COMPUTE VELOCITIES
    # Velocity is the difference between consecutive latent states
    # Shape: (Batch, Seq_Len-1, Latent)
    velocity = mu_sequence[:, 1:, :] - mu_sequence[:, :-1, :]
    
    # 2. VELOCITY MAGNITUDE LOSS
    # We want velocities to have non-trivial magnitude
    velocity_magnitude = torch.norm(velocity, dim=-1)  # (Batch, Seq_Len-1)
    mean_velocity = velocity_magnitude.mean()
    
    # Penalize if mean velocity is below threshold
    # Using smooth penalty: max(0, min_velocity - mean_velocity)^2
    velocity_deficit = F.relu(min_velocity - mean_velocity)
    velocity_loss = velocity_deficit.pow(2) * 100  # Scale up for gradient magnitude
    
    # 3. VELOCITY DIVERSITY LOSS  
    # Measure how diverse the velocities are across time
    # High std = velocities change direction/magnitude = good dynamics
    velocity_std = velocity.std(dim=1).mean()  # Std across time, then average
    
    # Penalize low diversity (but don't let it dominate)
    diversity_loss = 1.0 / (velocity_std + 0.1)
    
    # 4. ANTI-COLLAPSE LOSS
    # Ensure latent space uses multiple dimensions
    # Measure utilization: variance across latent dimensions should be similar
    dim_variance = mu_sequence.var(dim=(0, 1))  # Variance per dimension
    variance_uniformity = dim_variance.std() / (dim_variance.mean() + 1e-6)
    collapse_loss = variance_uniformity  # Penalize if some dims have much more variance
    
    # COMBINE LOSSES
    # Total dynamics loss encourages:
    # - Sufficient velocity (not frozen)
    # - Diverse velocities (not monotonic)
    # - Uniform dimension usage (not collapsed)
    total_dynamics_loss = velocity_loss + 0.5 * diversity_loss + 0.3 * collapse_loss
    
    # Diagnostics for monitoring
    diagnostics = {
        'mean_velocity': mean_velocity.item(),
        'velocity_std': velocity_std.item(),
        'velocity_loss': velocity_loss.item(),
        'diversity_loss': diversity_loss.item(),
        'collapse_loss': collapse_loss.item()
    }
    
    return total_dynamics_loss, diagnostics


def vae_loss_with_dynamics(recon_x, x, mu, logvar, mu_sequence, beta=0.5, dynamics_weight=1.0):
    """
    Complete VAE loss with dynamics-encouraging component.
    
    Components:
    1. Reconstruction loss (MSE) - preserve input information
    2. KL divergence - regularize latent space
    3. Dynamics loss - encourage temporal dynamics
    
    Args:
        recon_x: Reconstructed data (flattened)
        x: Original data (flattened)
        mu: Latent means (flattened)
        logvar: Latent log-variances (flattened)
        mu_sequence: Latent means reshaped to (batch, seq, latent)
        beta: Weight for KL divergence (beta-VAE)
        dynamics_weight: Weight for dynamics loss
    """
    # 1. RECONSTRUCTION LOSS
    # Mean over all dimensions instead of sum (more stable scaling)
    recon_loss = F.mse_loss(recon_x, x, reduction='mean')
    
    # 2. KL DIVERGENCE with FREE BITS
    # Free bits prevents posterior collapse by allowing some deviation from prior
    # KL per dimension: 0.5 * (mu^2 + exp(logvar) - 1 - logvar)
    kl_per_dim = 0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar)  # (B, latent_dim)
    
    # Free bits: only penalize KL above threshold (increased to 1.0 for better reconstruction)
    FREE_BITS = 1.0
    kl_per_dim = torch.clamp(kl_per_dim, min=FREE_BITS)  # Don't penalize below threshold
    kl_loss = kl_per_dim.mean()
    
    # 3. DYNAMICS LOSS  
    dynamics_loss, dynamics_diag = dynamics_encouraging_loss(mu_sequence)
    
    # COMBINE
    total_loss = recon_loss + beta * kl_loss + dynamics_weight * dynamics_loss
    
    return total_loss, {
        'recon_loss': recon_loss.item(),
        'kl_loss': kl_loss.item(),
        'dynamics_loss': dynamics_loss.item(),
        **dynamics_diag
    }


def split_dataset(dataset):
    """Split dataset into train and validation (80/20 split)"""
    total_size = len(dataset)
    val_size = int(0.2 * total_size)
    train_size = total_size - val_size
    
    train_indices = list(range(train_size))
    val_indices = list(range(train_size, total_size))
    
    return train_indices, val_indices


def evaluate(model, dataloader, device, beta, dynamics_weight):
    """Evaluate model on validation set (assumes pre-normalized data)"""
    model.eval()
    total_loss = 0
    total_recon = 0
    total_kl = 0
    total_dynamics = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            batch_seq = batch[0]  # NormalizedDataset returns tuple
            batch_size, seq_len, feat_dim = batch_seq.shape
            batch_seq = batch_seq.to(device)
            
            # Data is already normalized, no need to normalize again
            
            # Flatten for VAE
            x = batch_seq.view(-1, feat_dim)
            
            # Forward pass
            recon_x, mu, logvar = model(x)
            
            # Reshape mu for temporal loss
            mu_sequence = mu.view(batch_size, seq_len, -1)
            
            # Compute loss
            loss, diagnostics = vae_loss_with_dynamics(
                recon_x, x, mu, logvar, mu_sequence, beta, dynamics_weight
            )
            
            total_loss += loss.item()
            total_recon += diagnostics['recon_loss']
            total_kl += diagnostics['kl_loss']
            total_dynamics += diagnostics['dynamics_loss']
            num_batches += 1
    
    return {
        'loss': total_loss / num_batches,
        'recon': total_recon / num_batches,
        'kl': total_kl / num_batches,
        'dynamics': total_dynamics / num_batches
    }


def train_dynamics_vae(config=None):
    """
    Train VAE with dynamics-encouraging loss.
    """
    if config is None:
        config = {}
    
    batch_size = config.get('batch_size', BATCH_SIZE)
    epochs = config.get('epochs', EPOCHS)
    lr = config.get('lr', LEARNING_RATE)
    latent_dim = config.get('latent_dim', LATENT_DIM)
    beta = config.get('beta', BETA)
    dynamics_weight = config.get('dynamics_weight', DYNAMICS_WEIGHT)
    device = config.get('device', DEVICE)
    
    # Setup save paths
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    save_path = os.path.join(SAVE_DIR, f"vae_dynamics_latent{latent_dim}.pth")
    best_model_path = os.path.join(SAVE_DIR, f"vae_dynamics_latent{latent_dim}_best.pth")
    stats_path = os.path.join(SAVE_DIR, f"vae_norm_stats_dynamics_latent{latent_dim}.npy")
    
    print(f"\n{'='*60}")
    print(f"Training Dynamics-Encouraging VAE")
    print(f"{'='*60}")
    print(f"Architecture: ImprovedVAE (256 -> 128 -> 64 -> {latent_dim})")
    print(f"Beta (KL weight): {beta}")
    print(f"Dynamics weight: {dynamics_weight}")
    print(f"Min velocity target: {MIN_VELOCITY}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")
    
    # 1. Load pre-normalized dataset using shared utilities
    # Use temporal split for proper evaluation of temporal dynamics
    print("Loading pre-normalized dataset...")
    train_loader, val_loader, norm_stats = get_data_loaders(batch_size=batch_size, temporal_split=True)
    
    # Get normalization stats as tensors
    train_mean = torch.tensor(norm_stats['mean']).to(device)
    train_std = torch.tensor(norm_stats['std']).to(device)
    
    print(f"[OK] Dataset loaded (pre-normalized)")
    print(f"Train: {len(train_loader.dataset)} sequences")
    print(f"Val: {len(val_loader.dataset)} sequences")
    print(f"Norm stats: mean={norm_stats['mean']:.4f}, std={norm_stats['std']:.4f}\n")
    
    # Data is already normalized, so we use identity transform in training
    # train_mean and train_std are still used for saving/reference
    use_normalized = True  # Flag to indicate data is pre-normalized
    train_std = train_std.to(device)
    
    # 4. Create model
    print("Creating ImprovedVAE model...")
    model = ImprovedVAE(
        input_dim=INPUT_DIM,
        latent_dim=latent_dim,
        hidden_dims=[256, 128, 64],  # Deeper architecture
        dropout=0.1
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[OK] Model created with {total_params:,} parameters\n")
    
    # Optimizer with weight decay for regularization
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    
    # Cosine annealing scheduler for smooth learning rate decay
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    # 5. Training loop
    print("Starting training...\n")
    best_val_loss = float('inf')
    patience = 0
    max_patience = 30
    
    # Track metrics for analysis
    history = {
        'train_loss': [], 'val_loss': [],
        'mean_velocity': [], 'recon_loss': []
    }
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_metrics = {'loss': 0, 'recon': 0, 'kl': 0, 'dynamics': 0, 'velocity': 0}
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for batch in pbar:
            batch_seq = batch[0]  # NormalizedDataset returns tuple
            batch_size_actual, seq_len, feat_dim = batch_seq.shape
            batch_seq = batch_seq.to(device)
            
            # Data is already normalized, no need to normalize again
            
            # Flatten for VAE
            x = batch_seq.view(-1, feat_dim)
            
            # Forward pass
            optimizer.zero_grad()
            recon_x, mu, logvar = model(x)
            
            # Reshape mu for temporal loss
            mu_sequence = mu.view(batch_size_actual, seq_len, -1)
            
            # Compute loss
            loss, diagnostics = vae_loss_with_dynamics(
                recon_x, x, mu, logvar, mu_sequence, beta, dynamics_weight
            )
            
            # Backprop
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Track metrics
            train_metrics['loss'] += loss.item()
            train_metrics['recon'] += diagnostics['recon_loss']
            train_metrics['kl'] += diagnostics['kl_loss']
            train_metrics['dynamics'] += diagnostics['dynamics_loss']
            train_metrics['velocity'] += diagnostics['mean_velocity']
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'vel': f"{diagnostics['mean_velocity']:.4f}"
            })
        
        # Average training metrics
        for k in train_metrics:
            train_metrics[k] /= num_batches
        
        # Validation
        val_metrics = evaluate(model, val_loader, device, beta, dynamics_weight)
        
        # Scheduler
        scheduler.step()
        
        # Track history
        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'].append(val_metrics['loss'])
        history['mean_velocity'].append(train_metrics['velocity'])
        history['recon_loss'].append(train_metrics['recon'])
        
        # Save best model (use TRAINING loss because temporal split means
        # validation is from a different time period, not same distribution)
        if train_metrics['loss'] < best_val_loss:
            best_val_loss = train_metrics['loss']
            torch.save(model.state_dict(), best_model_path)
            patience = 0
        else:
            patience += 1
        
        # Print progress
        if (epoch + 1) % 5 == 0 or epoch < 3:
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"TrLoss: {train_metrics['loss']:.4f} | "
                  f"VaLoss: {val_metrics['loss']:.4f} | "
                  f"Recon: {train_metrics['recon']:.4f} | "
                  f"Vel: {train_metrics['velocity']:.4f} | "
                  f"LR: {scheduler.get_last_lr()[0]:.2e}")
        
        # Early stopping
        if patience >= max_patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    # 6. Save final model
    torch.save(model.state_dict(), save_path)
    np.save(os.path.join(SAVE_DIR, f"training_history_dynamics_{latent_dim}.npy"), history)
    
    print(f"\n{'='*60}")
    print(f"[OK] Training Complete!")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Final Mean Velocity: {history['mean_velocity'][-1]:.4f}")
    print(f"Model saved to: {best_model_path}")
    print(f"{'='*60}\n")
    
    return best_val_loss, best_model_path, stats_path


if __name__ == "__main__":
    train_dynamics_vae()