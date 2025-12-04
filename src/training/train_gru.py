"""
Autoregressive GRU Training Script
Trains temporal model to predict brain state dynamics using autoregressive rollouts
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
from src.data.dataset import BCIDataset
from src.models.vae import VAE
from src.models.gru import TemporalGRU
import os
import numpy as np
from tqdm import tqdm

# --- CONFIG ---
BATCH_SIZE = 128
EPOCHS = 500
LEARNING_RATE = 1e-3
LATENT_DIM = 32
HIDDEN_DIM = 64
NUM_LAYERS = 4
DROPOUT = 0.1
INPUT_DIM = 325

# Autoregressive training config
TEACHER_FORCING_EPOCHS = 400  # Warmup epochs with full teacher forcing

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VAE_PATH = "checkpoints/vae/vae_latent32_best.pth"
SAVE_DIR = "checkpoints/gru"
SAVE_PATH = f"{SAVE_DIR}/gru_autoregressive_L32_H64_4L.pth"
BEST_MODEL_PATH = f"{SAVE_DIR}/gru_autoregressive_L32_H64_4L_best.pth"


def encode_to_latent(vae_model, dataloader, device, norm_stats=None):
    """
    Encode all data to latent space using pre-trained VAE
    """
    vae_model.eval()
    all_latents = []
    all_labels = []
    
    # Prepare normalization tensors
    if norm_stats is not None:
        mean = torch.tensor(norm_stats['mean']).to(device)
        std = torch.tensor(norm_stats['std']).to(device)
        print("Using VAE normalization stats for encoding")
    
    print("Encoding data to latent space...")
    with torch.no_grad():
        for batch_seq, batch_labels in tqdm(dataloader, desc="Encoding"):
            # batch_seq: (Batch, Seq_Len, Features)
            batch_size, seq_len, feat_dim = batch_seq.shape
            
            # Flatten to encode each frame
            x = batch_seq.view(-1, feat_dim).to(device)
            
            # Normalize input if stats provided
            if norm_stats is not None:
                x = (x - mean) / std
            
            # Encode through VAE (only need mu, not the full sample)
            mu, _ = vae_model.encode(x)
            
            # Reshape back to sequences
            latent_seq = mu.view(batch_size, seq_len, -1)
            
            all_latents.append(latent_seq.cpu())
            all_labels.append(batch_labels)
    
    latent_sequences = torch.cat(all_latents, dim=0)
    labels = torch.cat(all_labels, dim=0)
    
    print(f"Encoded shape: {latent_sequences.shape}")
    return latent_sequences, labels


def create_sequences(latent_sequences):
    """
    Create input-target pairs for autoregressive training
    
    For autoregressive training, we predict one step ahead:
    Input: [z1, z2, z3, z4]
    Target: [z2, z3, z4, z5]
    
    Args:
        latent_sequences: (N, Seq_Len, Latent) tensor
    
    Returns:
        inputs:  Sequences without the last timestep
        targets: Sequences without the first timestep (shifted by 1)
    """
    # Input: all timesteps except the last
    # Target: all timesteps except the first (shifted by 1)
    inputs = latent_sequences[:, :-1, :]
    targets = latent_sequences[:, 1:, :]
    
    return inputs, targets


def train_epoch_autoregressive(model, inputs, targets, optimizer, device, epoch, total_epochs):
    """
    Train for one epoch with autoregressive rollouts and scheduled sampling
    
    Strategy:
    - Early epochs: Use mostly ground truth (teacher forcing)
    - Later epochs: Use mostly model predictions (autoregressive)
    - This prevents training instability while learning robust dynamics
    """
    model.train()
    total_loss = 0
    num_batches = 0
    
    # Scheduled sampling: gradually reduce teacher forcing
    # Ratio = 1.0 at start (full teacher forcing), 0.0 at end (full autoregressive)
    teacher_forcing_ratio = max(0.0, 1.0 - (epoch / TEACHER_FORCING_EPOCHS))
    
    # Create batches manually
    dataset_size = inputs.shape[0]
    indices = torch.randperm(dataset_size)
    
    for i in range(0, dataset_size, BATCH_SIZE):
        batch_indices = indices[i:i + BATCH_SIZE]
        batch_inputs = inputs[batch_indices].to(device)  # (B, Seq-1, Latent)
        batch_targets = targets[batch_indices].to(device)  # (B, Seq-1, Latent)
        
        batch_size = batch_inputs.shape[0]
        seq_len = batch_inputs.shape[1]
        
        optimizer.zero_grad()
        
        # Autoregressive training with scheduled sampling
        predictions = []
        hidden = None
        
        # Start with the first timestep
        current_state = batch_inputs[:, 0:1, :]  # (B, 1, Latent)
        
        for t in range(seq_len):
            # Predict next state using GRU's predict_next (which applies residual connection)
            next_state, hidden = model.predict_next(current_state, hidden)
            predictions.append(next_state)
            
            # Scheduled sampling: choose between ground truth and prediction
            if t + 1 < seq_len:  # If not at the end
                if np.random.random() < teacher_forcing_ratio:
                    # Use ground truth
                    current_state = batch_inputs[:, t+1:t+2, :]
                else:
                    # Use prediction (autoregressive)
                    current_state = next_state
        
        # Stack all predictions
        pred_sequence = torch.cat(predictions, dim=1)  # (B, Seq-1, Latent)
        
        # Calculate loss on all predictions
        loss = F.mse_loss(pred_sequence, batch_targets)
        
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    avg_loss = total_loss / num_batches
    
    return avg_loss, teacher_forcing_ratio


def evaluate_autoregressive(model, inputs, targets, device):
    """
    Evaluate model on validation set using full autoregressive rollout
    This gives us a realistic measure of generalization
    """
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        dataset_size = inputs.shape[0]
        
        for i in range(0, dataset_size, BATCH_SIZE):
            batch_inputs = inputs[i:i + BATCH_SIZE].to(device)
            batch_targets = targets[i:i + BATCH_SIZE].to(device)
            
            batch_size = batch_inputs.shape[0]
            seq_len = batch_inputs.shape[1]
            
            # Full autoregressive rollout (no teacher forcing)
            predictions = []
            hidden = None
            current_state = batch_inputs[:, 0:1, :]
            
            for t in range(seq_len):
                next_state, hidden = model.predict_next(current_state, hidden)
                predictions.append(next_state)
                current_state = next_state  # Always use prediction
            
            pred_sequence = torch.cat(predictions, dim=1)
            
            # Loss
            loss = F.mse_loss(pred_sequence, batch_targets)
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches


def train_gru(config=None):
    """
    Train GRU with autoregressive rollouts
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
    
    # Create save directory
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Training Autoregressive GRU")
    print(f"{'='*60}")
    print(f"Latent Dim: {latent_dim}")
    print(f"Hidden Dim: {hidden_dim}")
    print(f"Num Layers: {num_layers}")
    print(f"Teacher Forcing Warmup: {TEACHER_FORCING_EPOCHS} epochs")
    print(f"Device: {device}")
    print(f"{'='*60}\n")
    
    # 1. Load pre-trained VAE
    print(f"Loading VAE from {vae_path}...")
    vae = VAE(input_dim=INPUT_DIM, latent_dim=latent_dim).to(device)
    vae.load_state_dict(torch.load(vae_path, map_location=device))
    vae.eval()
    print("[OK] VAE loaded\n")
    
    # 2. Load dataset
    print("Loading dataset...")
    full_dataset = BCIDataset("data/processed/train")
    full_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False)
    print(f"[OK] Dataset loaded: {len(full_dataset)} sequences\n")
    
    # 3. Encode to latent space
    # Load VAE normalization stats
    vae_stats_path = f"checkpoints/vae/vae_norm_stats_latent{latent_dim}.npy"
    if not os.path.exists(vae_stats_path):
        vae_stats_path = "checkpoints/vae/vae_norm_stats.npy"
    
    vae_norm_stats = np.load(vae_stats_path, allow_pickle=True).item()
    latent_sequences, labels = encode_to_latent(vae, full_loader, device, vae_norm_stats)
    print(f"[OK] Encoding complete\n")
    
    # 4. NORMALIZE LATENT SPACE
    print("Normalizing latent space...")
    latent_mean = latent_sequences.mean()
    latent_std = latent_sequences.std()
    latent_normalized = (latent_sequences - latent_mean) / (latent_std + 1e-8)
    
    # Save normalization stats for inference
    latent_stats = {
        'mean': latent_mean.item(),
        'std': latent_std.item()
    }
    stats_save_path = f"{SAVE_DIR}/latent_norm_stats_autoregressive.npy"
    np.save(stats_save_path, latent_stats)
    print(f"[OK] Latent stats saved to {stats_save_path}\n")
    
    # 5. Create sequences for autoregressive training
    print("Creating autoregressive sequences...")
    inputs, targets = create_sequences(latent_normalized)
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
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=50
    )
    
    # 8. Training loop
    print("Starting training...\n")
    best_val_loss = float('inf')
    patience = 0
    max_patience = 200
    
    for epoch in range(epochs):
        # Train with autoregressive rollouts and scheduled sampling
        train_loss, tf_ratio = train_epoch_autoregressive(
            model, train_inputs, train_targets, optimizer, device, epoch, epochs
        )
        
        # Validate with full autoregressive rollout
        val_loss = evaluate_autoregressive(model, val_inputs, val_targets, device)
        
        # Scheduler
        scheduler.step(val_loss)
        
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
                  f"TF: {tf_ratio:.2f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Early stopping
        if patience >= max_patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    # Save final model
    torch.save(model.state_dict(), SAVE_PATH)
    
    print(f"\n{'='*60}")
    print(f"[OK] Training Complete!")
    print(f"Best Validation Loss: {best_val_loss:.6f}")
    print(f"Model saved to: {BEST_MODEL_PATH}")
    print(f"{'='*60}\n")
    
    return best_val_loss, BEST_MODEL_PATH


if __name__ == "__main__":
    train_gru()
