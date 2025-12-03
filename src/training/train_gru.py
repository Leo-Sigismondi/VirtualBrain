"""
GRU Training Script with Many-to-Many Strategy
Trains temporal model to predict brain state dynamics in latent space
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
from src.models.gru import TemporalGRU
import os
import numpy as np
from tqdm import tqdm

# --- CONFIG ---
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 1e-3
LATENT_DIM = 128
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.1
INPUT_DIM = 325
SKIP_STEPS = 4  # Predict 4 steps ahead (1 second) to force learning dynamics
LAZY_PENALTY_WEIGHT = 0.05 # Penalize small prediction vectors to prevent lazy prediction

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VAE_PATH = "checkpoints/vae/vae_latent128_best.pth"
SAVE_PATH = "checkpoints/gru/gru_L128_H128_Lay2.pth"
BEST_MODEL_PATH = "checkpoints/gru/gru_L128_H128_Lay2_best.pth"

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

def create_shifted_sequences(latent_sequences, skip_steps=4):
    """
    Create input-target pairs for many-to-many training with skip steps
    
    Instead of predicting the very next timestep (0.25s ahead),
    predict skip_steps ahead (e.g., 4 steps = 1 second ahead)
    
    This forces the model to learn meaningful temporal dynamics
    instead of just copying the input (lazy predictor problem)
    
    Args:
        latent_sequences: (N, Seq_Len, Latent) tensor
        skip_steps: How many steps ahead to predict (default=4 for 1 second)
    
    Returns:
        inputs:  Valid sequences that have enough future context
        targets: Corresponding target states skip_steps ahead
    """
    seq_len = latent_sequences.shape[1]
    
    # We need at least skip_steps+1 timesteps to make one prediction
    if seq_len <= skip_steps:
        raise ValueError(f"Sequence length {seq_len} is too short for skip_steps={skip_steps}")
    
    # Input: all timesteps that have a valid target skip_steps ahead
    # Target: the state skip_steps in the future
    inputs = latent_sequences[:, :-skip_steps, :]
    targets = latent_sequences[:, skip_steps:, :]
    
    return inputs, targets

def train_epoch(model, inputs, targets, optimizer, device, lazy_penalty=0.01):
    """
    Train for one epoch with many-to-many strategy
    Added anti-lazy predictor regularization
    """
    model.train()
    total_loss = 0
    total_mse = 0
    total_lazy_penalty = 0
    num_batches = 0
    
    # Create batches manually
    dataset_size = inputs.shape[0]
    indices = torch.randperm(dataset_size)
    
    for i in range(0, dataset_size, BATCH_SIZE):
        batch_indices = indices[i:i + BATCH_SIZE]
        batch_inputs = inputs[batch_indices].to(device)
        batch_targets = targets[batch_indices].to(device)
        
        optimizer.zero_grad()
        
        # Forward: predict delta at each timestep
        delta_pred, _ = model(batch_inputs)
        
        # Target delta = target_state - current_state
        target_delta = batch_targets - batch_inputs
        
        # MSE loss
        mse_loss = F.mse_loss(delta_pred, target_delta)
        
        # ANTI-LAZY PENALTY
        # Penalize predictions that are too close to zero
        # This forces the model to actually predict dynamics
        lazy_loss = -torch.mean(torch.norm(delta_pred, dim=-1))
        
        # Combined loss
        loss = mse_loss + lazy_penalty * lazy_loss
        
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        total_mse += mse_loss.item()
        total_lazy_penalty += lazy_loss.item()
        num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_mse = total_mse / num_batches
    avg_lazy = total_lazy_penalty / num_batches
    
    return avg_loss, avg_mse, avg_lazy

def evaluate(model, inputs, targets, device):
    """
    Evaluate model on validation set
    """
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        dataset_size = inputs.shape[0]
        
        for i in range(0, dataset_size, BATCH_SIZE):
            batch_inputs = inputs[i:i + BATCH_SIZE].to(device)
            batch_targets = targets[i:i + BATCH_SIZE].to(device)
            
            # Predict delta
            delta_pred, _ = model(batch_inputs)
            
            # Target delta
            target_delta = batch_targets - batch_inputs
            
            # Loss
            loss = F.mse_loss(delta_pred, target_delta)
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches

def train_gru(config=None):
    """
    Train GRU with given configuration
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
    stats_path = config.get('stats_path', "checkpoints/gru/latent_norm_stats.npy")
    
    # Update paths based on config
    save_dir = "checkpoints/gru"
    os.makedirs(save_dir, exist_ok=True)
    
    model_name = f"gru_L{latent_dim}_H{hidden_dim}_Lay{num_layers}"
    save_path = os.path.join(save_dir, f"{model_name}.pth")
    best_model_path = os.path.join(save_dir, f"{model_name}_best.pth")

    print(f"\n--- Training GRU ({model_name}) ---")
    
    # 1. Load pre-trained VAE
    vae = VAE(input_dim=INPUT_DIM, latent_dim=latent_dim).to(device)
    vae.load_state_dict(torch.load(vae_path, map_location=device))
    vae.eval()
    
    # 2. Load dataset
    full_dataset = BCIDataset("data/processed/train")
    full_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False)
    
    # 3. Encode to latent space
    # Load VAE normalization stats
    vae_norm_stats = np.load(stats_path, allow_pickle=True).item()
    latent_sequences, labels = encode_to_latent(vae, full_loader, device, vae_norm_stats)
    
    # 4. NORMALIZE LATENT SPACE
    latent_mean = latent_sequences.mean()
    latent_std = latent_sequences.std()
    latent_normalized = (latent_sequences - latent_mean) / (latent_std + 1e-8)
    
    # 5. Create shifted sequences
    inputs, targets = create_shifted_sequences(latent_normalized, skip_steps=SKIP_STEPS)
    
    # 6. Train/Val split
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
    
    # 7. Create GRU model
    model = TemporalGRU(
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=DROPOUT
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    # 8. Training loop
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Train
        train_loss, train_mse, train_lazy = train_epoch(
            model, train_inputs, train_targets, optimizer, device, lazy_penalty=LAZY_PENALTY_WEIGHT
        )
        
        # Validate
        val_loss = evaluate(model, val_inputs, val_targets, device)
        
        # Scheduler
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
        
        # Print progress
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Val Loss: {val_loss:.6f} | Best: {best_val_loss:.6f}")
    
    # Save final model
    torch.save(model.state_dict(), save_path)
    
    print(f"✓ GRU Training Complete. Best Val Loss: {best_val_loss:.6f}")
    return best_val_loss, best_model_path

if __name__ == "__main__":
    train_gru()
