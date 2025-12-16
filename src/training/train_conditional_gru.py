"""
Conditional GRU Training Script for Class-Specific Trajectory Generation
=========================================================================

Trains a conditional GRU that can generate brain state trajectories
conditioned on motor imagery class labels.

KEY DIFFERENCES FROM BASE GRU TRAINING:
- Loads window-level class labels alongside latent sequences
- Passes class labels to the conditional model
- Uses same multi-step loss strategy as base GRU

USAGE:
    python src/training/train_conditional_gru.py

OUTPUT:
    checkpoints/gru/conditional_gru_*.pth
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
    load_normalized_dataset_with_labels, encode_to_latent, 
    LabeledLatentDataset, INPUT_DIM, SEQUENCE_LENGTH, NUM_CLASSES
)
from src.models.vae import ImprovedVAE
from src.models.conditional_gru import ConditionalTemporalGRU
import os
import numpy as np
from tqdm import tqdm

# --- CONFIG ---
BATCH_SIZE = 256
EPOCHS = 200
LEARNING_RATE = 5e-4
LATENT_DIM = 32
HIDDEN_DIM = 256
NUM_LAYERS = 3
CLASS_EMBED_DIM = 16
DROPOUT = 0.2

# Multi-step prediction horizons
PREDICTION_HORIZONS = [1, 4, 8, 16]

# Seed steps: use ground truth to build hidden state before autoregressive prediction
SEED_STEPS = 8

# Autoregressive training config
TEACHER_FORCING_EPOCHS = EPOCHS // 2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Use the dynamics-trained VAE (same as base GRU)
VAE_PATH = "checkpoints/vae/vae_dynamics_latent32_best.pth"

SAVE_DIR = "checkpoints/gru"
SAVE_PATH = f"{SAVE_DIR}/conditional_gru_L{LATENT_DIM}_H{HIDDEN_DIM}_{NUM_LAYERS}L.pth"
BEST_MODEL_PATH = f"{SAVE_DIR}/conditional_gru_L{LATENT_DIM}_H{HIDDEN_DIM}_{NUM_LAYERS}L_best.pth"


def compute_multistep_loss(predictions, targets, horizons=[1, 4, 8, 16]):
    """
    Compute loss at multiple prediction horizons.
    Same as base GRU training.
    """
    batch_size, seq_len, latent_dim = predictions.shape
    total_loss = 0
    losses = {}
    
    for horizon in horizons:
        if horizon >= seq_len:
            continue
            
        weight = np.log(horizon + 1)
        horizon_loss = F.mse_loss(predictions, targets, reduction='mean')
        total_loss += weight * horizon_loss
        losses[f'h{horizon}'] = horizon_loss.item()
    
    return total_loss, losses


def compute_directional_loss(predictions, targets, inputs):
    """Penalize wrong prediction direction."""
    target_direction = torch.sign(targets - inputs)
    pred_direction = torch.sign(predictions - inputs)
    direction_agreement = (target_direction * pred_direction).mean()
    direction_loss = (1 - direction_agreement) / 2
    return direction_loss


def compute_diversity_loss(predictions, targets):
    """Penalize when predictions lack batch diversity."""
    pred_batch_var = predictions.var(dim=0).mean()
    target_batch_var = targets.var(dim=0).mean()
    var_ratio = pred_batch_var / (target_batch_var + 1e-8)
    diversity_loss = torch.relu(1.0 - var_ratio)
    return diversity_loss


def train_epoch_conditional(model, inputs, targets, labels, optimizer, device, epoch, total_epochs):
    """
    Train for one epoch with class conditioning.
    """
    model.train()
    total_loss = 0
    total_direction_loss = 0
    num_batches = 0
    
    # Scheduled sampling
    teacher_forcing_ratio = max(0.0, 1.0 - (epoch / TEACHER_FORCING_EPOCHS))
    
    # Create batches
    dataset_size = inputs.shape[0]
    indices = torch.randperm(dataset_size)
    
    for i in tqdm(range(0, dataset_size, BATCH_SIZE), desc=f"Epoch {epoch+1}/{total_epochs}", leave=False):
        batch_indices = indices[i:i + BATCH_SIZE]
        batch_inputs = inputs[batch_indices].to(device)
        batch_targets = targets[batch_indices].to(device)
        batch_labels = labels[batch_indices].to(device)  # Class labels
        
        batch_size = batch_inputs.shape[0]
        seq_len = batch_inputs.shape[1]
        
        optimizer.zero_grad()
        
        # === PHASE 1: Build hidden state using seed steps ===
        hidden = None
        for t in range(SEED_STEPS):
            current_state = batch_inputs[:, t:t+1, :]
            _, hidden = model.predict_next(current_state, batch_labels, hidden)
        
        # === PHASE 2: Autoregressive prediction with scheduled sampling ===
        predictions = []
        current_state = batch_inputs[:, SEED_STEPS-1:SEED_STEPS, :]
        
        for t in range(SEED_STEPS, seq_len):
            next_state, hidden = model.predict_next(current_state, batch_labels, hidden)
            predictions.append(next_state)
            
            if t + 1 < seq_len:
                if np.random.random() < teacher_forcing_ratio:
                    current_state = batch_inputs[:, t:t+1, :]
                else:
                    current_state = next_state
        
        if len(predictions) == 0:
            continue
            
        pred_sequence = torch.cat(predictions, dim=1)
        
        # Compute losses on predicted portion
        target_portion = batch_targets[:, SEED_STEPS:, :]
        input_portion = batch_inputs[:, SEED_STEPS-1:-1, :]
        
        mse_loss, _ = compute_multistep_loss(pred_sequence, target_portion, PREDICTION_HORIZONS)
        dir_loss = compute_directional_loss(pred_sequence, target_portion, input_portion)
        diversity_loss = compute_diversity_loss(pred_sequence, target_portion)
        
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


def evaluate_conditional(model, inputs, targets, labels, device):
    """
    Evaluate model using full autoregressive rollout.
    """
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        dataset_size = inputs.shape[0]
        
        for i in range(0, dataset_size, BATCH_SIZE):
            batch_inputs = inputs[i:i + BATCH_SIZE].to(device)
            batch_targets = targets[i:i + BATCH_SIZE].to(device)
            batch_labels = labels[i:i + BATCH_SIZE].to(device)
            
            seq_len = batch_inputs.shape[1]
            
            # Phase 1: Build hidden state
            hidden = None
            for t in range(SEED_STEPS):
                current = batch_inputs[:, t:t+1, :]
                _, hidden = model.predict_next(current, batch_labels, hidden)
            
            # Phase 2: Autoregressive rollout
            predictions = []
            current = batch_inputs[:, SEED_STEPS-1:SEED_STEPS, :]
            
            for t in range(SEED_STEPS, seq_len):
                next_state, hidden = model.predict_next(current, batch_labels, hidden)
                predictions.append(next_state)
                current = next_state
            
            if len(predictions) == 0:
                continue
                
            pred_sequence = torch.cat(predictions, dim=1)
            target_portion = batch_targets[:, SEED_STEPS:, :]
            
            loss = F.mse_loss(pred_sequence, target_portion)
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0


def train_conditional_gru(config=None):
    """
    Train Conditional GRU for class-specific trajectory generation.
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
    
    print(f"\n{'='*60}")
    print(f"Training Conditional GRU for Data Augmentation")
    print(f"{'='*60}")
    print(f"Latent Dim: {latent_dim}")
    print(f"Hidden Dim: {hidden_dim}")
    print(f"Num Layers: {num_layers}")
    print(f"Num Classes: {NUM_CLASSES}")
    print(f"Class Embed Dim: {CLASS_EMBED_DIM}")
    print(f"Seed Steps: {SEED_STEPS}")
    print(f"Teacher Forcing Warmup: {TEACHER_FORCING_EPOCHS} epochs")
    print(f"Device: {device}")
    print(f"{'='*60}\n")
    
    # 1. Load pre-trained VAE
    print(f"Loading VAE from {vae_path}...")
    vae = ImprovedVAE(
        input_dim=INPUT_DIM, 
        latent_dim=latent_dim,
        hidden_dims=[256, 128, 64]
    ).to(device)
    
    if os.path.exists(vae_path):
        vae.load_state_dict(torch.load(vae_path, map_location=device))
        print("[OK] VAE loaded\n")
    else:
        print(f"[ERROR] VAE not found at {vae_path}")
        return None, None
    
    vae.eval()
    
    # 2. Load normalized dataset WITH LABELS
    print("Loading pre-normalized dataset with labels...")
    normalized_data, labels, norm_stats = load_normalized_dataset_with_labels()
    print(f"[OK] Dataset loaded: {len(normalized_data)} sequences")
    print(f"[OK] Labels loaded: {len(labels)} labels")
    
    # Print class distribution
    unique, counts = np.unique(labels, return_counts=True)
    print(f"Class distribution:")
    for u, c in zip(unique, counts):
        print(f"  Class {u}: {c:6d} ({100*c/len(labels):5.1f}%)")
    print()
    
    # 3. Encode to latent space
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
    stats_save_path = f"{SAVE_DIR}/latent_norm_stats_conditional.npy"
    np.save(stats_save_path, latent_stats)
    print(f"[OK] Latent stats saved to {stats_save_path}\n")
    
    # 5. Create autoregressive sequences
    print("Creating autoregressive sequences...")
    inputs = latent_normalized[:, :-1, :]
    targets = latent_normalized[:, 1:, :]
    labels_tensor = torch.from_numpy(labels).long()
    print(f"Input shape: {inputs.shape}")
    print(f"Target shape: {targets.shape}")
    print(f"Labels shape: {labels_tensor.shape}\n")
    
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
    train_labels = labels_tensor[train_indices]
    val_inputs = inputs[val_indices]
    val_targets = targets[val_indices]
    val_labels = labels_tensor[val_indices]
    
    print(f"Train: {train_inputs.shape[0]} sequences")
    print(f"Val: {val_inputs.shape[0]} sequences\n")
    
    # 7. Create Conditional GRU model
    print("Creating Conditional GRU model...")
    model = ConditionalTemporalGRU(
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_classes=NUM_CLASSES,
        class_embed_dim=CLASS_EMBED_DIM,
        dropout=DROPOUT
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[OK] Model created with {total_params:,} parameters\n")
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
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
        train_loss, dir_loss, tf_ratio = train_epoch_conditional(
            model, train_inputs, train_targets, train_labels, optimizer, device, epoch, epochs
        )
        
        # Validate
        val_loss = evaluate_conditional(model, val_inputs, val_targets, val_labels, device)
        
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
    np.save(f"{SAVE_DIR}/training_history_conditional.npy", history)
    
    print(f"\n{'='*60}")
    print(f"[OK] Training Complete!")
    print(f"Best Validation Loss: {best_val_loss:.6f}")
    print(f"Model saved to: {BEST_MODEL_PATH}")
    print(f"{'='*60}\n")
    
    return best_val_loss, BEST_MODEL_PATH


if __name__ == "__main__":
    train_conditional_gru()
