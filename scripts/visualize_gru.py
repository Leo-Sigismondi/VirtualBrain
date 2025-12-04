"""
Autoregressive GRU Visualization Script
Visualizes how well the GRU predicts future brain states using autoregressive rollouts
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from src.data.dataset import BCIDataset
from torch.utils.data import DataLoader
from src.models.vae import VAE
from src.models.gru import TemporalGRU

# Force unbuffered output
sys.stdout.reconfigure(encoding='utf-8')

# --- CONFIG ---
VAE_PATH = "checkpoints/vae/vae_latent32_best.pth"
GRU_PATH = "checkpoints/gru/gru_autoregressive_L32_H64_4L_best.pth"
LATENT_DIM = 32
HIDDEN_DIM = 64
INPUT_DIM = 325
NUM_LAYERS = 4


def load_models(device):
    """Load VAE and GRU models"""
    print("Loading models...")
    
    # Load VAE
    vae = VAE(input_dim=INPUT_DIM, latent_dim=LATENT_DIM).to(device)
    vae.load_state_dict(torch.load(VAE_PATH, map_location=device))
    vae.eval()
    
    # Load GRU
    gru = TemporalGRU(
        latent_dim=LATENT_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS
    ).to(device)
    gru.load_state_dict(torch.load(GRU_PATH, map_location=device))
    gru.eval()
    
    print("✓ Models loaded\n")
    return vae, gru


def encode_sequence(vae, seq_vectors, vae_mean, vae_std, device):
    """Encode a sequence to latent space"""
    latent_seq = []
    with torch.no_grad():
        for t in range(seq_vectors.shape[0]):
            frame = seq_vectors[t:t+1]
            
            # Normalize input
            frame = (frame - vae_mean) / vae_std
            
            # Encode
            mu, _ = vae.encode(frame)
            latent_seq.append(mu)
        
        latent_seq = torch.cat(latent_seq, dim=0).unsqueeze(0)
    
    return latent_seq


def visualize_one_step_predictions():
    """
    Visualize one-step-ahead predictions
    Shows how well the model predicts the immediate next state
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("="*60)
    print("One-Step-Ahead Prediction Visualization")
    print("="*60 + "\n")
    
    # Load models
    vae, gru = load_models(device)
    
    # Load data
    print("Loading dataset...")
    dataset = BCIDataset("data/processed/train")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    seq_vectors, _ = next(iter(dataloader))
    seq_vectors = seq_vectors[0].to(device)
    print(f"✓ Sequence loaded: {seq_vectors.shape}\n")
    
    # Load normalization stats
    vae_norm_stats = np.load("checkpoints/vae/vae_norm_stats_latent32.npy", allow_pickle=True).item()
    vae_mean = torch.tensor(vae_norm_stats['mean']).to(device)
    vae_std = torch.tensor(vae_norm_stats['std']).to(device)
    
    latent_stats = np.load("checkpoints/gru/latent_norm_stats_autoregressive.npy", allow_pickle=True).item()
    latent_mean = torch.tensor(latent_stats['mean']).to(device)
    latent_std = torch.tensor(latent_stats['std']).to(device)
    
    # Encode sequence
    print("Encoding to latent space...")
    latent_seq = encode_sequence(vae, seq_vectors, vae_mean, vae_std, device)
    
    # Normalize latent sequence
    latent_seq_norm = (latent_seq - latent_mean) / (latent_std + 1e-8)
    
    seq_len = latent_seq_norm.shape[1]
    print(f"✓ Encoded: {latent_seq_norm.shape}\n")
    
    # One-step predictions
    print("Generating one-step predictions...")
    with torch.no_grad():
        predictions = []
        hidden = None
        
        for t in range(seq_len - 1):
            current = latent_seq_norm[:, t:t+1, :]
            next_pred, hidden = gru.predict_next(current, hidden)
            predictions.append(next_pred)
        
        predictions = torch.cat(predictions, dim=1)  # (1, seq_len-1, latent)
    
    # Convert to numpy
    ground_truth = latent_seq_norm[0].cpu().numpy()
    predictions = predictions[0].cpu().numpy()
    
    # Calculate errors
    # predictions[t] predicts ground_truth[t+1]
    errors = np.abs(ground_truth[1:, :] - predictions)
    mean_error_per_dim = errors.mean(axis=0)
    mean_error_per_timestep = errors.mean(axis=1)
    
    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Plot 1: First 6 dimensions over time
    ax = axes[0, 0]
    timesteps = np.arange(seq_len)
    for dim in range(min(6, LATENT_DIM)):
        ax.plot(timesteps, ground_truth[:, dim], 'o-', label=f'GT Dim {dim}', markersize=6, alpha=0.7)
        # predictions start at t=1
        ax.plot(timesteps[1:], predictions[:, dim], 'x--', markersize=8, alpha=0.7)
    
    ax.set_xlabel('Timestep', fontsize=11)
    ax.set_ylabel('Latent Value (Normalized)', fontsize=11)
    ax.set_title('One-Step-Ahead Predictions\n(Solid=Ground Truth, Dashed=Predictions)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Error per dimension
    ax = axes[0, 1]
    ax.bar(range(LATENT_DIM), mean_error_per_dim, color='coral', alpha=0.7)
    ax.set_xlabel('Latent Dimension', fontsize=11)
    ax.set_ylabel('Mean Absolute Error', fontsize=11)
    ax.set_title(f'Prediction Error by Dimension\n(Overall MAE: {mean_error_per_dim.mean():.6f})', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Error over time
    ax = axes[1, 0]
    ax.plot(timesteps[1:], mean_error_per_timestep, 'o-', color='red', markersize=8, linewidth=2)
    ax.set_xlabel('Timestep', fontsize=11)
    ax.set_ylabel('Mean Absolute Error', fontsize=11)
    ax.set_title('Prediction Error Over Time', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Error distribution
    ax = axes[1, 1]
    ax.hist(errors.flatten(), bins=50, color='skyblue', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Absolute Error', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title(f'Error Distribution\n(Mean: {errors.mean():.6f}, Std: {errors.std():.6f})', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f'Autoregressive GRU - One-Step Prediction Quality',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # Save
    output_path = "checkpoints/gru/one_step_predictions.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Visualization saved: {output_path}\n")
    
    # Statistics
    print("One-Step Prediction Quality:")
    print(f"  Overall MAE: {mean_error_per_dim.mean():.6f}")
    print(f"  Max Error (worst dim): {mean_error_per_dim.max():.6f}")
    print(f"  Min Error (best dim): {mean_error_per_dim.min():.6f}")
    print(f"  Error Std: {errors.std():.6f}\n")
    
    plt.show()
    return mean_error_per_dim.mean()


def visualize_autoregressive_rollout():
    """
    Visualize multi-step autoregressive rollout
    Given the first N states, predict the remaining states autoregressively
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("="*60)
    print("Multi-Step Autoregressive Rollout Visualization")
    print("="*60 + "\n")
    
    # Load models
    vae, gru = load_models(device)
    
    # Load data
    print("Loading dataset...")
    dataset = BCIDataset("data/processed/train")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    seq_vectors, _ = next(iter(dataloader))
    seq_vectors = seq_vectors[0].to(device)
    print(f"✓ Sequence loaded: {seq_vectors.shape}\n")
    
    # Load normalization stats
    vae_norm_stats = np.load("checkpoints/vae/vae_norm_stats_latent32.npy", allow_pickle=True).item()
    vae_mean = torch.tensor(vae_norm_stats['mean']).to(device)
    vae_std = torch.tensor(vae_norm_stats['std']).to(device)
    
    latent_stats = np.load("checkpoints/gru/latent_norm_stats_autoregressive.npy", allow_pickle=True).item()
    latent_mean = torch.tensor(latent_stats['mean']).to(device)
    latent_std = torch.tensor(latent_stats['std']).to(device)
    
    # Encode sequence
    print("Encoding to latent space...")
    latent_seq = encode_sequence(vae, seq_vectors, vae_mean, vae_std, device)
    latent_seq_norm = (latent_seq - latent_mean) / (latent_std + 1e-8)
    seq_len = latent_seq_norm.shape[1]
    print(f"✓ Encoded: {latent_seq_norm.shape}\n")
    
    # Use first 2 states, predict the rest autoregressively
    num_seed = 2
    num_predict = seq_len - num_seed
    
    print(f"Autoregressive rollout: Given first {num_seed} states, predict next {num_predict} states...")
    with torch.no_grad():
        # Start with first state
        predictions = []
        hidden = None
        current = latent_seq_norm[:, 0:1, :]
        predictions.append(current)
        
        # Process second state to build up hidden state
        if num_seed > 1:
            current, hidden = gru.predict_next(current, hidden)
            predictions.append(current)
        
        # Autoregressive rollout
        for step in range(num_predict):
            next_state, hidden = gru.predict_next(current, hidden)
            predictions.append(next_state)
            current = next_state
        
        predicted_trajectory = torch.cat(predictions, dim=1)[0].cpu().numpy()
    
    ground_truth = latent_seq_norm[0].cpu().numpy()
    
    # Calculate error for predicted portion
    predicted_portion = predicted_trajectory[num_seed:]
    gt_portion = ground_truth[num_seed:]
    errors = np.abs(predicted_portion - gt_portion)
    error_per_step = errors.mean(axis=1)
    
    # Naive baseline: persistence (z[t+1] = z[t])
    naive_predictions = ground_truth[num_seed-1:-1]  # Shift by 1
    naive_errors = np.abs(gt_portion - naive_predictions)
    naive_error_per_step = naive_errors.mean(axis=1)
    
    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Plot 1: Trajectories
    ax = axes[0, 0]
    timesteps = np.arange(seq_len)
    for dim in range(min(6, LATENT_DIM)):
        ax.plot(timesteps, ground_truth[:, dim], 'o-', label=f'GT Dim {dim}', 
                markersize=6, linewidth=2, alpha=0.7)
        ax.plot(timesteps, predicted_trajectory[:, dim], 'x--', 
                markersize=8, linewidth=2, alpha=0.7)
    
    ax.axvline(x=num_seed-0.5, color='red', linestyle=':', linewidth=2.5, 
               label=f'Prediction starts (given first {num_seed})')
    ax.set_xlabel('Timestep', fontsize=11)
    ax.set_ylabel('Latent Value (Normalized)', fontsize=11)
    ax.set_title(f'Autoregressive Rollout\n(Solid=Ground Truth, Dashed=Predictions)', 
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Error accumulation
    ax = axes[0, 1]
    pred_timesteps = timesteps[num_seed:]
    ax.plot(pred_timesteps, error_per_step, 'o-', color='red', 
            label='GRU', markersize=8, linewidth=2)
    ax.plot(pred_timesteps, naive_error_per_step, 's--', color='gray', 
            label='Naive Baseline', markersize=6, linewidth=2, alpha=0.7)
    ax.set_xlabel('Timestep', fontsize=11)
    ax.set_ylabel('Mean Absolute Error', fontsize=11)
    ax.set_title(f'Error Accumulation Over Rollout\n(GRU vs Baseline)', 
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Cumulative error
    ax = axes[1, 0]
    cumulative_gru = np.cumsum(error_per_step)
    cumulative_naive = np.cumsum(naive_error_per_step)
    ax.plot(pred_timesteps, cumulative_gru, 'o-', color='red', 
            label='GRU', markersize=8, linewidth=2)
    ax.plot(pred_timesteps, cumulative_naive, 's--', color='gray', 
            label='Naive Baseline', markersize=6, linewidth=2, alpha=0.7)
    ax.set_xlabel('Timestep', fontsize=11)
    ax.set_ylabel('Cumulative Error', fontsize=11)
    ax.set_title('Cumulative Error Over Time', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Error comparison
    ax = axes[1, 1]
    gru_mae = errors.mean()
    naive_mae = naive_errors.mean()
    
    bars = ax.bar(['GRU\nAutoregressive', 'Naive\nPersistence'], 
                  [gru_mae, naive_mae], 
                  color=['coral', 'lightgray'], alpha=0.8, edgecolor='black', linewidth=2)
    
    # Add values on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.6f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    improvement = (naive_mae - gru_mae) / naive_mae * 100
    ax.set_ylabel('Mean Absolute Error', fontsize=11)
    ax.set_title(f'Overall Performance Comparison\n(GRU improves by {improvement:.1f}%)', 
                 fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f'Autoregressive GRU - Multi-Step Rollout Quality',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # Save
    output_path = "checkpoints/gru/autoregressive_rollout.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Visualization saved: {output_path}\n")
    
    # Statistics
    print("Autoregressive Rollout Quality:")
    print(f"  GRU MAE: {gru_mae:.6f}")
    print(f"  Naive MAE: {naive_mae:.6f}")
    print(f"  Improvement: {improvement:.2f}%")
    print(f"  Final step error (GRU): {error_per_step[-1]:.6f}")
    print(f"  Final step error (Naive): {naive_error_per_step[-1]:.6f}\n")
    
    plt.show()
    return gru_mae, naive_mae


def evaluate_full_dataset():
    """
    Evaluate autoregressive performance on full dataset
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("="*60)
    print("Full Dataset Autoregressive Evaluation")
    print("="*60 + "\n")
    
    # Load models
    vae, gru = load_models(device)
    
    # Load data
    print("Loading full dataset...")
    dataset = BCIDataset("data/processed/train")
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
    print(f"✓ Dataset loaded: {len(dataset)} sequences\n")
    
    # Load normalization stats
    vae_norm_stats = np.load("checkpoints/vae/vae_norm_stats_latent32.npy", allow_pickle=True).item()
    vae_mean = torch.tensor(vae_norm_stats['mean']).to(device)
    vae_std = torch.tensor(vae_norm_stats['std']).to(device)
    
    latent_stats = np.load("checkpoints/gru/latent_norm_stats_autoregressive.npy", allow_pickle=True).item()
    latent_mean = torch.tensor(latent_stats['mean']).to(device)
    latent_std = torch.tensor(latent_stats['std']).to(device)
    
    # Encode all sequences
    print("Encoding full dataset...")
    all_latents = []
    with torch.no_grad():
        for batch_seq, _ in dataloader:
            batch_seq = batch_seq.to(device)
            batch_size, seq_len, feat_dim = batch_seq.shape
            
            # Flatten
            x = batch_seq.view(-1, feat_dim)
            
            # Normalize
            x = (x - vae_mean) / vae_std
            
            # Encode
            mu, _ = vae.encode(x)
            
            # Reshape
            latent_seq = mu.view(batch_size, seq_len, -1)
            all_latents.append(latent_seq)
    
    latent_sequences = torch.cat(all_latents, dim=0)
    
    # Normalize latent space
    latent_normalized = (latent_sequences - latent_mean) / (latent_std + 1e-8)
    print(f"✓ Encoded: {latent_normalized.shape}\n")
    
    # Evaluate autoregressive prediction
    print("Evaluating autoregressive predictions...")
    total_gru_error = 0
    total_naive_error = 0
    total_elements = 0
    
    batch_size = 64
    with torch.no_grad():
        for i in range(0, latent_normalized.shape[0], batch_size):
            batch = latent_normalized[i:i+batch_size].to(device)
            b_size, seq_len = batch.shape[0], batch.shape[1]
            
            # Autoregressive prediction
            predictions = []
            hidden = None
            current = batch[:, 0:1, :]
            
            for t in range(seq_len - 1):
                next_state, hidden = gru.predict_next(current, hidden)
                predictions.append(next_state)
                current = next_state
            
            pred_seq = torch.cat(predictions, dim=1)
            
            # GRU error
            target = batch[:, 1:, :]
            gru_error = F.l1_loss(pred_seq, target, reduction='sum')
            total_gru_error += gru_error.item()
            
            # Naive error (persistence)
            naive_pred = batch[:, :-1, :]
            naive_error = F.l1_loss(naive_pred, target, reduction='sum')
            total_naive_error += naive_error.item()
            
            total_elements += target.numel()
    
    gru_mae = total_gru_error / total_elements
    naive_mae = total_naive_error / total_elements
    improvement = (naive_mae - gru_mae) / naive_mae * 100
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"GRU MAE (autoregressive): {gru_mae:.6f}")
    print(f"Naive Baseline MAE: {naive_mae:.6f}")
    print(f"Improvement: {improvement:.2f}%")
    
    if improvement > 0:
        print("✅ SUCCESS: GRU beats naive baseline!")
    else:
        print("❌ WARNING: GRU does not beat baseline")
    
    print("="*60 + "\n")
    
    return gru_mae, naive_mae


if __name__ == "__main__":
    # Run all visualizations
    print("\n" + "🎨 "*20)
    print("Autoregressive GRU Visualization Suite")
    print("🎨 "*20 + "\n")
    
    # 1. One-step predictions
    one_step_error = visualize_one_step_predictions()
    
    print("\n" + "-"*60 + "\n")
    
    # 2. Multi-step rollout
    gru_rollout_mae, naive_rollout_mae = visualize_autoregressive_rollout()
    
    print("\n" + "-"*60 + "\n")
    
    # 3. Full dataset evaluation
    gru_full_mae, naive_full_mae = evaluate_full_dataset()
    
    print("\n" + "🎉 "*20)
    print("All visualizations complete!")
    print("🎉 "*20 + "\n")
