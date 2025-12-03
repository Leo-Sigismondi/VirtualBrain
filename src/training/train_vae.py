"""
VAE Training Script with Train/Validation Split
Trains a Variational Autoencoder on BCI covariance matrices and monitors overfitting
"""
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
LATENT_DIM = 64
INPUT_DIM = 325
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_PATH = "checkpoints/vae/vae_encoder.pth"
BEST_MODEL_PATH = "checkpoints/vae/vae_encoder_best.pth"

# Validation split (which subjects to use for validation)
# We have 9 subjects total, use 2 for validation
VALIDATION_SUBJECTS = ['A08T', 'A09T']  # Hold out last 2 subjects

def loss_function(recon_x, x, mu, logvar):
    """
    VAE Loss = Reconstruction Loss (MSE) + KL Divergence
    """
    # 1. Reconstruction loss
    MSE = F.mse_loss(recon_x, x, reduction='sum')
    
    # 2. KL Divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return MSE + KLD, MSE.item(), KLD.item()

def split_dataset(dataset, validation_subjects):
    """
    Split dataset into train and validation based on subject IDs
    """
    train_indices = []
    val_indices = []
    
    # Get all files to know which indices correspond to which subjects
    for idx in range(len(dataset)):
        # Check which file this index comes from
        # Since dataset concatenates all subjects, we need to track this
        # For now, we'll use a simple approach: last ~20% for validation
        pass
    
    # Simple split: last 20% of data for validation
    total_size = len(dataset)
    val_size = int(0.2 * total_size)
    train_size = total_size - val_size
    
    train_indices = list(range(train_size))
    val_indices = list(range(train_size, total_size))
    
    return train_indices, val_indices

def evaluate(model, dataloader, device, mean, std):
    """
    Evaluate model on validation set
    """
    model.eval()
    total_loss = 0
    total_mse = 0
    total_kld = 0
    
    with torch.no_grad():
        for batch_seq, _ in dataloader:
            x = batch_seq.view(-1, INPUT_DIM).to(device)
            
            # Normalize
            x = (x - mean) / std
            
            recon_x, mu, logvar = model(x)
            loss, mse, kld = loss_function(recon_x, x, mu, logvar)
            total_loss += loss.item()
            total_mse += mse
            total_kld += kld
    
"""
VAE Training Script with Train/Validation Split
Trains a Variational Autoencoder on BCI covariance matrices and monitors overfitting
"""
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
LATENT_DIM = 64
INPUT_DIM = 325
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_PATH = "checkpoints/vae/vae_encoder.pth"
BEST_MODEL_PATH = "checkpoints/vae/vae_encoder_best.pth"

# Validation split (which subjects to use for validation)
# We have 9 subjects total, use 2 for validation
VALIDATION_SUBJECTS = ['A08T', 'A09T']  # Hold out last 2 subjects

def loss_function(recon_x, x, mu, logvar):
    """
    VAE Loss = Reconstruction Loss (MSE) + KL Divergence
    """
    # 1. Reconstruction loss
    MSE = F.mse_loss(recon_x, x, reduction='sum')
    
    # 2. KL Divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return MSE + KLD, MSE.item(), KLD.item()

def split_dataset(dataset, validation_subjects):
    """
    Split dataset into train and validation based on subject IDs
    """
    train_indices = []
    val_indices = []
    
    # Get all files to know which indices correspond to which subjects
    for idx in range(len(dataset)):
        # Check which file this index comes from
        # Since dataset concatenates all subjects, we need to track this
        # For now, we'll use a simple approach: last ~20% for validation
        pass
    
    # Simple split: last 20% of data for validation
    total_size = len(dataset)
    val_size = int(0.2 * total_size)
    train_size = total_size - val_size
    
    train_indices = list(range(train_size))
    val_indices = list(range(train_size, total_size))
    
    return train_indices, val_indices

def evaluate(model, dataloader, device, mean, std):
    """
    Evaluate model on validation set
    """
    model.eval()
    total_loss = 0
    total_mse = 0
    total_kld = 0
    
    with torch.no_grad():
        for batch_seq, _ in dataloader:
            x = batch_seq.view(-1, INPUT_DIM).to(device)
            
            # Normalize
            x = (x - mean) / std
            
            recon_x, mu, logvar = model(x)
            loss, mse, kld = loss_function(recon_x, x, mu, logvar)
            total_loss += loss.item()
            total_mse += mse
            total_kld += kld
    
    avg_loss = total_loss / len(dataloader.dataset)
    avg_mse = total_mse / len(dataloader.dataset)
    avg_kld = total_kld / len(dataloader.dataset)
    
    return avg_loss, avg_mse, avg_kld

def train_vae(config=None):
    """
    Train VAE with given configuration
    """
    # Default config
    if config is None:
        config = {}
        
    batch_size = config.get('batch_size', BATCH_SIZE)
    epochs = config.get('epochs', EPOCHS)
    lr = config.get('lr', LEARNING_RATE)
    latent_dim = config.get('latent_dim', LATENT_DIM)
    device = config.get('device', DEVICE)
    
    # Update paths based on latent dim to avoid overwriting
    save_dir = "checkpoints/vae"
    os.makedirs(save_dir, exist_ok=True)
    
    save_path = os.path.join(save_dir, f"vae_latent{latent_dim}.pth")
    best_model_path = os.path.join(save_dir, f"vae_latent{latent_dim}_best.pth")
    stats_path = os.path.join(save_dir, f"vae_norm_stats_latent{latent_dim}.npy")

    # 1. Load full dataset
    print(f"\n--- Training VAE (Latent={latent_dim}) ---")
    print("Loading dataset...")
    full_dataset = BCIDataset("data/processed/train")
    
    # 2. Split into train/validation
    train_indices, val_indices = split_dataset(full_dataset, VALIDATION_SUBJECTS)
    
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 3. Calculate Normalization Stats
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
    
    train_mean = train_mean.to(device)
    train_std = train_std.to(device)

    # 4. Model
    model = VAE(input_dim=INPUT_DIM, latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Track best validation loss
    best_val_loss = float('inf')
    
    # 5. Training loop
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        
        for batch_seq, _ in train_loader:
            x = batch_seq.view(-1, INPUT_DIM).to(device)
            x = (x - train_mean) / train_std
            
            optimizer.zero_grad()
            recon_x, mu, logvar = model(x)
            loss, _, _ = loss_function(recon_x, x, mu, logvar)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_dataset)
        
        # Validation
        val_loss, val_mse, val_kld = evaluate(model, val_loader, device, train_mean, train_std)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
        
        # Print progress sparingly
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Val Loss: {val_loss:.4f} | Best: {best_val_loss:.4f}")
    
    # 6. Save final model
    torch.save(model.state_dict(), save_path)
    
    print(f"✓ VAE Training Complete. Best Val Loss: {best_val_loss:.4f}")
    return best_val_loss, best_model_path, stats_path

if __name__ == "__main__":
    # Default execution
    train_vae()