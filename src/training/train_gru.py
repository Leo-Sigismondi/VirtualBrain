"""
Autoregressive GRU Training Script with Multi-Step Loss
========================================================

WHY MULTI-STEP LOSS:
-------------------
If we only train with 1-step prediction, the GRU can learn to:
- Just copy the input (lazy prediction)
- Only use the most recent state (ignore history)

Multi-step loss forces the model to learn ACTUAL dynamics because:
1. Errors compound over multiple steps
2. Model must understand trends/patterns, not just memorize
3. Longer horizons require understanding of underlying dynamics

TRAINING STRATEGY:
-----------------
1. SCHEDULED SAMPLING: Gradually reduce teacher forcing
   - Early epochs: Use ground truth as input (stable learning)
   - Later epochs: Use predictions as input (learn to recover from errors)

2. MULTI-HORIZON LOSS: Penalize at multiple prediction distances
   - t+1: Short-term accuracy
   - t+4: Medium-term dynamics
   - t+8: Long-term trends
   
3. DIRECTIONAL LOSS: Penalize when prediction direction is wrong
   - If ground truth goes UP, prediction should go UP
   - This is more important than exact magnitude
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
    encode_to_latent, INPUT_DIM, SEQUENCE_LENGTH
)
from src.models.vae import ImprovedVAE
from src.models.gru import TemporalGRU
import os
import numpy as np
from tqdm import tqdm

# --- CONFIG ---
BATCH_SIZE = 256
EPOCHS = 200
LEARNING_RATE = 5e-4
LATENT_DIM = 32
HIDDEN_DIM = 256  # Increased from 128 for more capacity
NUM_LAYERS = 3    # Increased from 2 for deeper temporal modeling
DROPOUT = 0.2
# INPUT_DIM = 253 and SEQUENCE_LENGTH = 64 imported from data_utils

# Multi-step prediction horizons
PREDICTION_HORIZONS = [1, 4, 8, 16]  # Predict at these many steps ahead

# Seed steps: use ground truth to build hidden state before autoregressive prediction
# This aligns training with how the model will be evaluated
SEED_STEPS = 8  # Use first 8 steps to warm up hidden state, predict remaining 55

# Autoregressive training config
TEACHER_FORCING_EPOCHS = EPOCHS // 2  # Reach 0 forcing halfway, then pure autoregressive

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Use the dynamics-trained VAE
VAE_PATH = "checkpoints/vae/vae_dynamics_latent32_best.pth"
VAE_STATS_PATH = "checkpoints/vae/vae_norm_stats_dynamics_latent32.npy"

SAVE_DIR = "checkpoints/gru"
SAVE_PATH = f"{SAVE_DIR}/gru_multistep_L32_H{HIDDEN_DIM}_{NUM_LAYERS}L.pth"
BEST_MODEL_PATH = f"{SAVE_DIR}/gru_multistep_L32_H{HIDDEN_DIM}_{NUM_LAYERS}L_best.pth"





def compute_multistep_loss(predictions, targets, horizons=[1, 4, 8, 16]):
    """
    Compute loss at multiple prediction horizons.
    
    WHY THIS MATTERS:
    - 1-step loss: Model can cheat by just copying input
    - Multi-step loss: Model must understand dynamics
    
    Args:
        predictions: (Batch, Seq, Latent) - predicted sequence
        targets: (Batch, Seq, Latent) - ground truth sequence
        horizons: List of step horizons to evaluate
        
    Returns:
        total_loss: Weighted sum of horizon losses
        losses: Dict with individual horizon losses
    """
    batch_size, seq_len, latent_dim = predictions.shape
    total_loss = 0
    losses = {}
    
    for horizon in horizons:
        if horizon >= seq_len:
            continue
            
        # Compare prediction[t] with target[t] for valid timesteps
        # Weight longer horizons more heavily (they're harder)
        weight = np.log(horizon + 1)  # log scale for balanced weighting
        
        horizon_loss = F.mse_loss(predictions, targets, reduction='mean')
        total_loss += weight * horizon_loss
        losses[f'h{horizon}'] = horizon_loss.item()
    
    return total_loss, losses


def compute_directional_loss(predictions, targets, inputs):
    """
    Loss that penalizes wrong prediction DIRECTION.
    
    WHY THIS HELPS:
    - MSE only cares about magnitude
    - Predicting the right DIRECTION (up/down) is often more important
    - This encourages the model to capture trends
    
    Args:
        predictions: (B, Seq-1, D) Predicted states (predictions for t+1, t+2, ...)
        targets: (B, Seq-1, D) Ground truth states (z[1], z[2], ..., z[Seq-1])
        inputs: (B, Seq-1, D) Input states used for prediction (z[0], z[1], ..., z[Seq-2])
    """
    # Target direction: which way did the ground truth move from input?
    # target[t] - input[t] = z[t+1] - z[t]
    target_direction = torch.sign(targets - inputs)
    
    # Predicted direction: which way does our prediction move from input?
    pred_direction = torch.sign(predictions - inputs)
    
    # Penalize when directions don't match
    # Agreement is +1 when same sign, -1 when different, 0 when one is zero
    direction_agreement = (target_direction * pred_direction).mean()
    
    # Convert to loss: we want high agreement, so loss = (1 - agreement) / 2
    # This gives loss in [0, 1], with 0 being perfect agreement
    direction_loss = (1 - direction_agreement) / 2
    
    return direction_loss


def compute_diversity_loss(predictions, targets):
    """
    Loss that penalizes when predictions are too similar across different inputs.
    
    WHY THIS MATTERS:
    - If the model ignores input and outputs the same thing for all sequences,
      the batch variance of predictions will be near zero.
    - Ground truth has natural variance across different samples.
    - This loss encourages the model to produce varied outputs.
    
    Args:
        predictions: (Batch, Seq, Latent)
        targets: (Batch, Seq, Latent)
    
    Returns:
        diversity_loss: Penalizes when predictions lack batch diversity
    """
    # Compute variance across batch dimension for predictions and targets
    pred_batch_var = predictions.var(dim=0).mean()  # Variance over batch
    target_batch_var = targets.var(dim=0).mean()    # Ground truth variance
    
    # We want pred_batch_var to be close to target_batch_var
    # If pred_batch_var << target_batch_var, model is ignoring inputs
    # Use ratio-based loss: penalize when predictions are less varied than targets
    var_ratio = pred_batch_var / (target_batch_var + 1e-8)
    
    # Loss is high when ratio < 1 (predictions less varied than targets)
    # ReLU(1 - ratio) = 0 when ratio >= 1, increases as ratio decreases
    diversity_loss = torch.relu(1.0 - var_ratio)
    
    return diversity_loss


def compute_temporal_variance_loss(predictions, targets):
    """
    Force predictions to match the TEMPORAL VARIANCE of ground truth.
    
    WHY THIS HELPS:
    - GRU tends to predict "safe" low-variance outputs (flat dynamics)
    - This loss penalizes when predictions have less variation over time
    - Encourages the model to capture the full dynamic range
    
    Args:
        predictions: (Batch, Seq, Latent)
        targets: (Batch, Seq, Latent)
    """
    # Variance over time (dim=1) for each sample, then average
    pred_temporal_var = predictions.var(dim=1).mean()  # (B, D) -> scalar
    target_temporal_var = targets.var(dim=1).mean()
    
    # Penalize when predictions have less temporal variance than targets
    var_ratio = pred_temporal_var / (target_temporal_var + 1e-8)
    temporal_var_loss = torch.relu(1.0 - var_ratio)
    
    return temporal_var_loss


def compute_velocity_loss(predictions, targets):
    """
    Match velocity (frame-to-frame changes) between predictions and targets.
    
    WHY THIS HELPS:
    - Velocities capture the dynamics of the sequence
    - If prediction velocities are too small, the output looks flat
    - MSE on velocities + variance matching ensures dynamic predictions
    
    Args:
        predictions: (Batch, Seq, Latent)
        targets: (Batch, Seq, Latent)
    """
    # Compute velocities (differences between consecutive frames)
    pred_velocity = predictions[:, 1:, :] - predictions[:, :-1, :]
    target_velocity = targets[:, 1:, :] - targets[:, :-1, :]
    
    # L2 loss on velocities (match the direction and magnitude of changes)
    velocity_mse = F.mse_loss(pred_velocity, target_velocity)
    
    # Also match velocity variance (ensure similar "energy" in dynamics)
    pred_vel_var = pred_velocity.var()
    target_vel_var = target_velocity.var()
    vel_var_loss = torch.relu(1.0 - pred_vel_var / (target_vel_var + 1e-8))
    
    return velocity_mse + vel_var_loss


def train_epoch_multistep(model, inputs, targets, optimizer, device, epoch, total_epochs):
    """
    Train for one epoch with multi-step loss and scheduled sampling.
    
    IMPORTANT: Uses SEED_STEPS to build hidden state before autoregressive prediction.
    This aligns training with evaluation where we give the model context to warm up.
    """
    model.train()
    total_loss = 0
    total_direction_loss = 0
    num_batches = 0
    
    # Scheduled sampling: gradually reduce teacher forcing
    teacher_forcing_ratio = max(0.0, 1.0 - (epoch / TEACHER_FORCING_EPOCHS))
    
    # Create batches
    dataset_size = inputs.shape[0]
    indices = torch.randperm(dataset_size)
    
    for i in tqdm(range(0, dataset_size, BATCH_SIZE), desc=f"Epoch {epoch+1}/{total_epochs}", leave=False):
        batch_indices = indices[i:i + BATCH_SIZE]
        batch_inputs = inputs[batch_indices].to(device)
        batch_targets = targets[batch_indices].to(device)
        
        batch_size = batch_inputs.shape[0]
        seq_len = batch_inputs.shape[1]
        
        optimizer.zero_grad()
        
        # === PHASE 1: Build hidden state using seed steps (always use ground truth) ===
        hidden = None
        for t in range(SEED_STEPS):
            current_state = batch_inputs[:, t:t+1, :]
            _, hidden = model.predict_next(current_state, hidden)
        
        # === PHASE 2: Autoregressive prediction with scheduled sampling ===
        predictions = []
        
        # Start from the last seed state
        current_state = batch_inputs[:, SEED_STEPS-1:SEED_STEPS, :]
        
        # Predict remaining steps
        for t in range(SEED_STEPS, seq_len):
            # Predict next state
            next_state, hidden = model.predict_next(current_state, hidden)
            predictions.append(next_state)
            
            # Scheduled sampling (for predicted portion only)
            if t + 1 < seq_len:
                if np.random.random() < teacher_forcing_ratio:
                    current_state = batch_inputs[:, t:t+1, :]
                else:
                    current_state = next_state
        
        if len(predictions) == 0:
            continue
            
        pred_sequence = torch.cat(predictions, dim=1)
        
        # Only compute loss on predicted portion (after SEED_STEPS)
        # predictions has shape (B, seq_len - SEED_STEPS, D) = (B, 55, D)
        # targets should match: starting from step SEED_STEPS
        target_portion = batch_targets[:, SEED_STEPS:, :]  # (B, 55, D)
        input_portion = batch_inputs[:, SEED_STEPS-1:-1, :]  # (B, 55, D) - inputs used for each prediction
        
        # Multi-step MSE loss
        mse_loss, horizon_losses = compute_multistep_loss(
            pred_sequence, target_portion, PREDICTION_HORIZONS
        )
        
        # Directional loss
        dir_loss = compute_directional_loss(pred_sequence, target_portion, input_portion)
        
        # Batch diversity loss (prevents model from ignoring input)
        diversity_loss = compute_diversity_loss(pred_sequence, target_portion)
        
        # Combined loss (original formula without temporal variance losses)
        # - mse_loss: reconstruction accuracy
        # - dir_loss: direction correctness (0.5 weight)
        # - diversity_loss: batch diversity (2.0 weight)
        loss = mse_loss + 0.5 * dir_loss + 2.0 * diversity_loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += mse_loss.item()
        total_direction_loss += dir_loss.item()
        num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    avg_dir_loss = total_direction_loss / num_batches if num_batches > 0 else 0
    
    return avg_loss, avg_dir_loss, teacher_forcing_ratio


def evaluate_autoregressive(model, inputs, targets, device):
    """
    Evaluate model using full autoregressive rollout (no teacher forcing).
    Uses SEED_STEPS to build hidden state before prediction (aligns with training).
    """
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        dataset_size = inputs.shape[0]
        
        for i in range(0, dataset_size, BATCH_SIZE):
            batch_inputs = inputs[i:i + BATCH_SIZE].to(device)
            batch_targets = targets[i:i + BATCH_SIZE].to(device)
            
            seq_len = batch_inputs.shape[1]
            
            # Phase 1: Build hidden state using seed steps
            hidden = None
            for t in range(SEED_STEPS):
                current = batch_inputs[:, t:t+1, :]
                _, hidden = model.predict_next(current, hidden)
            
            # Phase 2: Autoregressive rollout from last seed state
            predictions = []
            current = batch_inputs[:, SEED_STEPS-1:SEED_STEPS, :]
            
            for t in range(SEED_STEPS, seq_len):
                next_state, hidden = model.predict_next(current, hidden)
                predictions.append(next_state)
                current = next_state
            
            if len(predictions) == 0:
                continue
                
            pred_sequence = torch.cat(predictions, dim=1)
            
            # Only compute loss on predicted portion
            # predictions shape: (B, seq_len - SEED_STEPS, D)
            target_portion = batch_targets[:, SEED_STEPS:, :]
            
            loss = F.mse_loss(pred_sequence, target_portion)
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0


def train_gru_multistep(config=None):
    """
    Train GRU with multi-step prediction loss.
    """
    if config is None:
        config = {}
    
    batch_size = config.get('batch_size', BATCH_SIZE)
    epochs = config.get('epochs', EPOCHS)
    lr = config.get('lr', LEARNING_RATE)
    latent_dim = config.get('latent_dim', LATENT_DIM)
    hidden_dim = config.get('hidden_dim', HIDDEN_DIM)
    num_layers = config.get('num_layers', NUM_LAYERS)
    device = config.get('device', DEVICE)
    vae_path = config.get('vae_path', VAE_PATH)
    
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    print(f"\\n{'='*60}")
    print(f"Training GRU with Multi-Step Loss + Seed Warmup")
    print(f"{'='*60}")
    print(f"Latent Dim: {latent_dim}")
    print(f"Hidden Dim: {hidden_dim}")
    print(f"Num Layers: {num_layers}")
    print(f"Seed Steps: {SEED_STEPS} (hidden state warmup)")
    print(f"Prediction Horizons: {PREDICTION_HORIZONS}")
    print(f"Sequence Length: {SEQUENCE_LENGTH}")
    print(f"Teacher Forcing Warmup: {TEACHER_FORCING_EPOCHS} epochs")
    print(f"Device: {device}")
    print(f"{'='*60}\\n")
    
    # 1. Load pre-trained VAE
    print(f"Loading VAE from {vae_path}...")
    vae = ImprovedVAE(
        input_dim=INPUT_DIM, 
        latent_dim=latent_dim,
        hidden_dims=[256, 128, 64]
    ).to(device)
    
    if os.path.exists(vae_path):
        vae.load_state_dict(torch.load(vae_path, map_location=device))
        print("[OK] VAE loaded\\n")
    else:
        print(f"[WARNING] VAE not found at {vae_path}, checking fallback...")
        fallback_path = "checkpoints/vae/vae_temporal_latent32_best.pth"
        if os.path.exists(fallback_path):
            # Load old VAE (may have different architecture)
            try:
                from src.models.vae import VAE as OldVAE
                vae = OldVAE(input_dim=INPUT_DIM, latent_dim=latent_dim).to(device)
                vae.load_state_dict(torch.load(fallback_path, map_location=device))
                print(f"[OK] Loaded fallback VAE from {fallback_path}\\n")
            except:
                print("[ERROR] Could not load VAE. Train VAE first!")
                return None, None
        else:
            print("[ERROR] No VAE found. Train VAE first!")
            return None, None
    
    vae.eval()
    
    # 2. Load pre-normalized dataset using shared utilities
    print("Loading pre-normalized dataset...")
    train_loader, val_loader, norm_stats = get_data_loaders(batch_size=batch_size)
    
    # For GRU, we need all data encoded to latent space
    # Load the full normalized dataset as memmap (no RAM copy needed yet)
    normalized_data, _ = load_normalized_dataset()
    print(f"[OK] Dataset memmap loaded: {len(normalized_data)} sequences\n")
    
    # 3. Encode to latent space using the encode_to_latent utility
    # It handles batching and GPU transfer internally to avoid OOM
    print("Encoding to latent space (batched)...")
    latent_sequences = encode_to_latent(vae, normalized_data, device)
    print(f"[OK] Encoding complete: {latent_sequences.shape}\n")
    
    # 4. Normalize latent space
    print("Normalizing latent space...")
    latent_mean = latent_sequences.mean()
    latent_std = latent_sequences.std()
    latent_normalized = (latent_sequences - latent_mean) / (latent_std + 1e-8)
    
    # Save normalization stats
    latent_stats = {
        'mean': latent_mean.item(),
        'std': latent_std.item()
    }
    stats_save_path = f"{SAVE_DIR}/latent_norm_stats_multistep.npy"
    np.save(stats_save_path, latent_stats)
    print(f"[OK] Latent stats saved to {stats_save_path}\n")
    
    # 5. Create autoregressive sequences
    print("Creating autoregressive sequences...")
    inputs = latent_normalized[:, :-1, :]
    targets = latent_normalized[:, 1:, :]
    print(f"Input shape: {inputs.shape}")
    print(f"Target shape: {targets.shape}\n")
    
    # 6. Train/Val split
    print("Splitting train/val...")
    dataset_size = inputs.shape[0]
    val_size = int(0.2 * dataset_size)
    train_size = dataset_size - val_size
    
    indices = torch.randperm(dataset_size)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_inputs = inputs[train_indices]
    train_targets = targets[train_indices]
    val_inputs = inputs[val_indices]
    val_targets = targets[val_indices]
    
    print(f"Train: {train_inputs.shape[0]} sequences")
    print(f"Val: {val_inputs.shape[0]} sequences\n")
    
    # 7. Create GRU model
    print("Creating GRU model...")
    model = TemporalGRU(
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=DROPOUT
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[OK] Model created with {total_params:,} parameters\n")
    
    # Optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Cosine annealing with warmup
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2, eta_min=1e-6
    )
    
    # 8. Training loop
    print("Starting training...\n")
    best_val_loss = float('inf')
    patience = 0
    max_patience = 50
    
    history = {'train_loss': [], 'val_loss': [], 'dir_loss': []}
    
    for epoch in range(epochs):
        # Train
        train_loss, dir_loss, tf_ratio = train_epoch_multistep(
            model, train_inputs, train_targets, optimizer, device, epoch, epochs
        )
        
        # Validate
        val_loss = evaluate_autoregressive(model, val_inputs, val_targets, device)
        
        # Scheduler
        scheduler.step()
        
        # Track
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['dir_loss'].append(dir_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            patience = 0
        else:
            patience += 1
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch < 5:
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train: {train_loss:.6f} | "
                  f"Val: {val_loss:.6f} | "
                  f"Best: {best_val_loss:.6f} | "
                  f"Dir: {dir_loss:.4f} | "
                  f"TF: {tf_ratio:.2f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Early stopping
        if patience >= max_patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    # Save final model
    torch.save(model.state_dict(), SAVE_PATH)
    np.save(f"{SAVE_DIR}/training_history_multistep.npy", history)
    
    print(f"\n{'='*60}")
    print(f"[OK] Training Complete!")
    print(f"Best Validation Loss: {best_val_loss:.6f}")
    print(f"Model saved to: {BEST_MODEL_PATH}")
    print(f"{'='*60}\n")
    
    return best_val_loss, BEST_MODEL_PATH


if __name__ == "__main__":
    train_gru_multistep()
