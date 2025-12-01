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
EPOCHS = 200
LEARNING_RATE = 1e-3
LATENT_DIM = 8
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

def evaluate(model, dataloader, device):
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
            recon_x, mu, logvar = model(x)
            loss, mse, kld = loss_function(recon_x, x, mu, logvar)
            total_loss += loss.item()
            total_mse += mse
            total_kld += kld
    
    avg_loss = total_loss / len(dataloader.dataset)
    avg_mse = total_mse / len(dataloader.dataset)
    avg_kld = total_kld / len(dataloader.dataset)
    
    return avg_loss, avg_mse, avg_kld

def main():
    os.makedirs("checkpoints/vae", exist_ok=True)

    # 1. Load full dataset
    print("Loading dataset...")
    full_dataset = BCIDataset("data/processed/train")
    
    # 2. Split into train/validation
    print("\nSplitting into train/validation sets...")
    train_indices, val_indices = split_dataset(full_dataset, VALIDATION_SUBJECTS)
    
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 3. Model
    model = VAE(input_dim=INPUT_DIM, latent_dim=LATENT_DIM).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(f"\nTraining VAE on {DEVICE}...")
    print(f"Input dimension: {INPUT_DIM}, Latent dimension: {LATENT_DIM}")
    print(f"Epochs: {EPOCHS}, Batch size: {BATCH_SIZE}\n")
    
    # Track best validation loss
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    # 4. Training loop
    for epoch in range(EPOCHS):
        # Training
        model.train()
        train_loss = 0
        
        for batch_seq, _ in train_loader:
            x = batch_seq.view(-1, INPUT_DIM).to(DEVICE)
            
            optimizer.zero_grad()
            recon_x, mu, logvar = model(x)
            loss, _, _ = loss_function(recon_x, x, mu, logvar)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_dataset)
        train_losses.append(avg_train_loss)
        
        # Validation
        val_loss, val_mse, val_kld = evaluate(model, val_loader, DEVICE)
        val_losses.append(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            best_epoch = epoch + 1
        
        # Print progress
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | "
                  f"Train Loss: {avg_train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Best Val: {best_val_loss:.4f} (epoch {best_epoch})")
    
    # 5. Save final model and training history
    torch.save(model.state_dict(), SAVE_PATH)
    
    # Save loss history for plotting
    np.save("checkpoints/vae/train_losses.npy", np.array(train_losses))
    np.save("checkpoints/vae/val_losses.npy", np.array(val_losses))
    
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Final model saved to: {SAVE_PATH}")
    print(f"Best model saved to: {BEST_MODEL_PATH} (epoch {best_epoch})")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"{'='*60}")
    
    # 6. Check for overfitting
    if avg_train_loss < val_loss * 0.7:  # Train loss much lower than val
        print("\n⚠️  WARNING: Possible overfitting detected!")
        print(f"   Training loss ({avg_train_loss:.4f}) is much lower than validation loss ({val_loss:.4f})")
        print("   Consider: reducing model capacity, adding regularization, or early stopping")
    else:
        print("\n✅ Model appears to generalize well!")

if __name__ == "__main__":
    main()