"""
GRU Temporal Prediction Visualization
Shows how well the GRU predicts future brain states in latent space
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import os
import sys
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
VAE_PATH = "checkpoints/vae/vae_latent16_best.pth"
GRU_PATH = "checkpoints/gru/gru_L16_H128_Lay1_best.pth"
LATENT_DIM = 16
HIDDEN_DIM = 128
INPUT_DIM = 325
NUM_LAYERS = 1
SKIP_STEPS = 4  # Match training config

def visualize_predictions():
    """
    Visualize GRU predictions vs ground truth
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("="*60)
    print("GRU Temporal Prediction Visualization")
    print("="*60)
    
    # 1. Load models
    print(f"\nLoading models...")
    vae = VAE(input_dim=INPUT_DIM, latent_dim=LATENT_DIM).to(device)
    vae.load_state_dict(torch.load(VAE_PATH, map_location=device))
    vae.eval()
    
    gru = TemporalGRU(
        latent_dim=LATENT_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS
    ).to(device)
    gru.load_state_dict(torch.load(GRU_PATH, map_location=device))
    gru.eval()
    print("✓ Models loaded")
    
    # 2. Load data
    print("\nLoading dataset...")
    dataset = BCIDataset("data/processed/train")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # 3. Get a random sequence
    seq_vectors, _ = next(iter(dataloader))
    seq_vectors = seq_vectors[0].to(device)  # (Seq_Len=5, Features=325)
    
    # Load VAE normalization stats
    vae_norm_stats = np.load("checkpoints/vae/vae_norm_stats_latent128.npy", allow_pickle=True).item()
    vae_mean = torch.tensor(vae_norm_stats['mean']).to(device)
    vae_std = torch.tensor(vae_norm_stats['std']).to(device)

    # 4. Encode to latent space
    print("Encoding to latent space...")
    with torch.no_grad():
        latent_seq = []
        for t in range(seq_vectors.shape[0]):
            frame = seq_vectors[t:t+1]  # (1, 325)
            
            # Normalize input
            frame = (frame - vae_mean) / vae_std
            
            mu, _ = vae.encode(frame)
            latent_seq.append(mu)
        latent_seq = torch.cat(latent_seq, dim=0).unsqueeze(0)  # (1, 5, 16)
    
    # Load normalization stats
    stats_path = "checkpoints/gru/latent_norm_stats.npy"
    if os.path.exists(stats_path):
        print(f"Loading normalization stats from {stats_path}")
        stats = np.load(stats_path, allow_pickle=True).item()
        latent_mean = torch.tensor(stats['mean']).to(device)
        latent_std = torch.tensor(stats['std']).to(device)
    else:
        print("WARNING: No normalization stats found! Predictions may be poor.")
        latent_mean = torch.tensor(0.0).to(device)
        latent_std = torch.tensor(1.0).to(device)

    # 5. Make predictions
    print("Generating predictions...")
    with torch.no_grad():
        # Normalize latent sequence for the GRU
        latent_seq_norm = (latent_seq - latent_mean) / (latent_std + 1e-8)
        
        # Use first 4 timesteps to predict the rest
        input_seq = latent_seq_norm[:, :-SKIP_STEPS, :]  # (1, 4, 16)
        
        # Predict deltas
        delta_pred, _ = gru(input_seq)
        
        # Apply residual connection to get predictions (in normalized space)
        predicted_states_norm = input_seq + delta_pred  # (1, 4, 16)
        
        # Let's visualize in NORMALIZED space
        predicted_states = predicted_states_norm
        
        # Also update ground truth to be normalized for comparison
        ground_truth_tensor = latent_seq_norm
    
    # Convert to numpy
    ground_truth = ground_truth_tensor[0].cpu().numpy()  # (Seq_Len, 16)
    predictions = predicted_states[0].cpu().numpy()  # (Seq_Len-1, 16)
    
    # Analyze Ground Truth Dynamics
    gt_std = np.std(ground_truth, axis=0)
    gt_range = np.ptp(ground_truth, axis=0)  # Peak-to-peak (max - min)
    
    with open("variance_analysis.txt", "w", encoding="utf-8") as f:
        f.write("Ground Truth Dynamics Analysis:\n")
        f.write(f"  Avg Std across dims: {np.mean(gt_std):.6f}\n")
        f.write(f"  Max Std in any dim: {np.max(gt_std):.6f}\n")
        f.write(f"  Avg Range (max-min): {np.mean(gt_range):.6f}\n")
        f.write(f"  Max Range in any dim: {np.max(gt_range):.6f}\n")
        
        if np.max(gt_range) < 0.1:
            f.write("\nWARNING: Ground truth is extremely flat!\n")
            f.write("   The VAE might be collapsing the temporal dimension.\n")
            f.write("   This explains why the lines look straight.\n")

            
    print("Analysis written to variance_analysis.txt")


    predictions = predicted_states[0].cpu().numpy()  # (Seq_Len-1, 16)
    
    # Get sequence length automatically
    seq_len = ground_truth.shape[0]
    pred_len = predictions.shape[0]
    
    # 6. Visualize
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: All latent dimensions over time
    ax = axes[0, 0]
    timesteps_gt = np.arange(seq_len)
    timesteps_pred = np.arange(SKIP_STEPS, seq_len)  # Predictions start at t=SKIP_STEPS
    
    for dim in range(LATENT_DIM):
        ax.plot(timesteps_gt, ground_truth[:, dim], 
                'o-', alpha=0.3, color='gray', markersize=4)
        ax.plot(timesteps_pred, predictions[:, dim], 
                'x--', alpha=0.8, markersize=6)
    
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Latent Value')
    ax.set_title('All Latent Dimensions\n(Gray=Ground Truth, Colored=Predictions)')
    ax.grid(True, alpha=0.3)
    ax.legend(['Ground Truth', 'Predictions'], loc='upper right')
    
    # Plot 2: First 4 latent dimensions (zoomed)
    ax = axes[0, 1]
    for dim in range(min(4, LATENT_DIM)):
        ax.plot(timesteps_gt, ground_truth[:, dim], 
                'o-', label=f'GT Dim {dim}', markersize=6)
        ax.plot(timesteps_pred, predictions[:, dim], 
                'x--', label=f'Pred Dim {dim}', markersize=8)
    
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Latent Value')
    ax.set_title('First 4 Latent Dimensions (Zoomed)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Prediction error per dimension
    ax = axes[1, 0]
    # Calculate error for timesteps where we have predictions
    # ground_truth starts at t=0. predictions start at t=SKIP_STEPS.
    # So we compare ground_truth[SKIP_STEPS:] with predictions
    errors = np.abs(ground_truth[SKIP_STEPS:, :] - predictions)
    mean_errors = errors.mean(axis=0)
    
    ax.bar(range(LATENT_DIM), mean_errors, color='coral', alpha=0.7)
    ax.set_xlabel('Latent Dimension')
    ax.set_ylabel('Mean Absolute Error')
    ax.set_title(f'Prediction Error per Dimension\n(Overall MAE: {mean_errors.mean():.6f})')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Error over time
    ax = axes[1, 1]
    errors_per_timestep = errors.mean(axis=1)
    ax.plot(timesteps_pred, errors_per_timestep, 'o-', 
            color='red', markersize=8, linewidth=2)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Mean Absolute Error')
    ax.set_title('Prediction Error Over Time')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'GRU Temporal Prediction Quality (Latent Dim={LATENT_DIM}, Hidden Dim={HIDDEN_DIM})',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # Save
    output_path = f"checkpoints/gru/prediction_visualization.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {output_path}")
    
    # Print statistics
    print(f"\nPrediction Quality:")
    print(f"  Overall MAE: {mean_errors.mean():.6f}")
    print(f"  Max Error (worst dim): {mean_errors.max():.6f}")
    print(f"  Min Error (best dim): {mean_errors.min():.6f}")
    print(f"  Error at t=1: {errors_per_timestep[0]:.6f}")
    print(f"  Error at t=4: {errors_per_timestep[-1]:.6f}")
    
    plt.show()

def visualize_multi_step():
    """
    Test multi-step autoregressive prediction
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("\n" + "="*60)
    print("Multi-Step Autoregressive Prediction")
    print("="*60)
    
    # Load models
    vae = VAE(input_dim=INPUT_DIM, latent_dim=LATENT_DIM).to(device)
    vae.load_state_dict(torch.load(VAE_PATH, map_location=device))
    vae.eval()
    
    gru = TemporalGRU(
        latent_dim=LATENT_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS
    ).to(device)
    gru.load_state_dict(torch.load(GRU_PATH, map_location=device))
    gru.eval()
    
    # Load data
    dataset = BCIDataset("data/processed/train")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    seq_vectors, _ = next(iter(dataloader))
    seq_vectors = seq_vectors[0].to(device)
    
    # Load VAE normalization stats
    vae_norm_stats = np.load("checkpoints/vae/vae_norm_stats_latent128.npy", allow_pickle=True).item()
    vae_mean = torch.tensor(vae_norm_stats['mean']).to(device)
    vae_std = torch.tensor(vae_norm_stats['std']).to(device)

    # Encode
    with torch.no_grad():
        latent_seq = []
        for t in range(seq_vectors.shape[0]):
            frame = seq_vectors[t:t+1]
            
            # Normalize input
            frame = (frame - vae_mean) / vae_std
            
            mu, _ = vae.encode(frame)
            latent_seq.append(mu)
        latent_seq = torch.cat(latent_seq, dim=0).unsqueeze(0)  # (1, Seq_Len, 16)
    
    # Load normalization stats
    stats_path = "checkpoints/gru/latent_norm_stats.npy"
    if os.path.exists(stats_path):
        stats = np.load(stats_path, allow_pickle=True).item()
        latent_mean = torch.tensor(stats['mean']).to(device)
        latent_std = torch.tensor(stats['std']).to(device)
    else:
        latent_mean = torch.tensor(0.0).to(device)
        latent_std = torch.tensor(1.0).to(device)

    seq_len = latent_seq.shape[1]
    num_pred_steps = seq_len - 2  # Use first 2, predict the rest
    
    # Use first 2 timesteps and predict the rest
    with torch.no_grad():
        # Normalize latent sequence
        latent_seq_norm = (latent_seq - latent_mean) / (latent_std + 1e-8)
        
        z_start = latent_seq_norm[:, :2, :]  # First 2 timesteps
        
        # Autoregressive prediction
        predictions = [z_start[:, 0, :].unsqueeze(1), 
                      z_start[:, 1, :].unsqueeze(1)]
        hidden = None
        
        for step in range(num_pred_steps):  # Predict remaining steps
            # Take last predicted state
            z_current = predictions[-1]
            
            # Predict delta
            delta, hidden = gru(z_current, hidden)
            
            # Apply residual
            z_next = z_current + delta
            predictions.append(z_next)
        
        predicted_trajectory = torch.cat(predictions, dim=1)[0].cpu().numpy()
    
    ground_truth = latent_seq_norm[0].cpu().numpy()
    seq_len = ground_truth.shape[0]
    
    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    for dim in range(min(6, LATENT_DIM)):
        ax.plot(range(seq_len), ground_truth[:, dim], 
                'o-', label=f'GT Dim {dim}', markersize=8, linewidth=2)
        ax.plot(range(seq_len), predicted_trajectory[:, dim], 
                'x--', alpha=0.7, markersize=10, linewidth=2)
    
    ax.axvline(x=1.5, color='red', linestyle=':', linewidth=2, 
               label='Prediction starts here')
    ax.set_xlabel('Timestep', fontsize=12)
    ax.set_ylabel('Latent Value', fontsize=12)
    ax.set_title(f'Multi-Step Autoregressive Prediction\n(Given t=0,1 → Predict t=2...{seq_len-1})', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("checkpoints/gru/multistep_prediction.png", dpi=150, bbox_inches='tight')
    print("\n✓ Multi-step visualization saved")
    plt.show()

def check_naive_baseline(latent_sequences, device, skip_steps=1):
    """
    Calculate error if we simply predicted: "The future is the same as the present".
    Z_{t+skip} = Z_t
    """
    print(f"\n--- Naive Baseline Calculation (Skip={skip_steps}) ---")
    
    # Input: Current state (all except last skip_steps)
    current_state = latent_sequences[:, :-skip_steps, :].to(device)
    
    # Target: Future state (all starting from skip_steps)
    future_real = latent_sequences[:, skip_steps:, :].to(device)
    
    # Naive Prediction: Copy current state
    naive_pred = current_state
    
    # Calculate MSE
    mse = F.mse_loss(naive_pred, future_real)
    print(f"Naive Baseline MSE: {mse.item():.6f}")
    return mse.item()

def evaluate_model_vs_baseline():
    """
    Compare GRU model performance against Naive Baseline
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("="*60)
    print("Model vs Baseline Comparison")
    print("="*60)
    
    # 1. Load VAE
    print(f"Loading VAE from {VAE_PATH}...")
    vae = VAE(input_dim=INPUT_DIM, latent_dim=LATENT_DIM).to(device)
    vae.load_state_dict(torch.load(VAE_PATH, map_location=device))
    vae.eval()
    
    # 2. Load GRU
    print(f"Loading GRU from {GRU_PATH}...")
    gru = TemporalGRU(
        latent_dim=LATENT_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS
    ).to(device)
    gru.load_state_dict(torch.load(GRU_PATH, map_location=device))
    gru.eval()
    
    # 3. Load Data
    print("Loading dataset...")
    dataset = BCIDataset("data/processed/train")
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
    
    # 4. Encode to Latent Space
    print("Encoding data to latent space...")
    
    # Load stats
    stats_path = f"checkpoints/vae/vae_norm_stats_latent{LATENT_DIM}.npy"
    if not os.path.exists(stats_path):
        # Fallback to generic name if specific not found
        stats_path = "checkpoints/vae/vae_norm_stats.npy"
        
    print(f"Loading normalization stats from {stats_path}")
    vae_norm_stats = np.load(stats_path, allow_pickle=True).item()
    vae_mean = torch.tensor(vae_norm_stats['mean']).to(device)
    vae_std = torch.tensor(vae_norm_stats['std']).to(device)
    
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
    
    # 5. Normalize Latent Space (Same as training)
    latent_mean = latent_sequences.mean()
    latent_std = latent_sequences.std()
    latent_normalized = (latent_sequences - latent_mean) / (latent_std + 1e-8)
    
    print(f"Latent Data Shape: {latent_normalized.shape}")
    
    # 6. Calculate Baseline
    # Assuming SKIP_STEPS=4 based on training config, but let's check what the user wants.
    # The user's snippet had skip_steps=1 default. 
    # We should use the SKIP_STEPS constant if defined, or default to 4.
    skip = 4 # Default to 4 as per our training
    baseline_mse = check_naive_baseline(latent_normalized, device, skip_steps=skip)
    
    # 7. Calculate GRU MSE
    print(f"\n--- Calculating GRU MSE (Skip={skip}) ---")
    inputs = latent_normalized[:, :-skip, :]
    targets = latent_normalized[:, skip:, :]
    
    # Process in batches to avoid OOM
    total_mse = 0
    count = 0
    batch_size = 64
    
    with torch.no_grad():
        for i in range(0, inputs.shape[0], batch_size):
            batch_input = inputs[i:i+batch_size].to(device)
            batch_target = targets[i:i+batch_size].to(device)
            
            delta_pred, _ = gru(batch_input)
            
            # GRU predicts delta, so predicted_next = current + delta
            # But we are comparing against target (which is next state)
            # Wait, train_gru calculates MSE on DELTA.
            # target_delta = batch_target - batch_input
            # loss = F.mse_loss(delta_pred, target_delta)
            
            # Let's calculate MSE on the actual state reconstruction to be comparable to baseline
            # Pred State = Input + Delta
            pred_state = batch_input + delta_pred
            
            mse = F.mse_loss(pred_state, batch_target, reduction='sum')
            total_mse += mse.item()
            count += batch_input.numel()
            
    gru_mse = total_mse / count
    print(f"GRU Model MSE: {gru_mse:.6f}")
    
    # 8. Conclusion
    print("\n" + "-"*30)
    print("RESULTS")
    print("-" * 30)
    print(f"Baseline MSE: {baseline_mse:.6f}")
    print(f"GRU Model MSE: {gru_mse:.6f}")
    
    if gru_mse < baseline_mse:
        improvement = (baseline_mse - gru_mse) / baseline_mse * 100
        print(f"✅ SUCCESS: GRU beats baseline by {improvement:.2f}%")
    else:
        print("❌ FAILURE: GRU is worse or equal to baseline (Lazy Predictor)")

if __name__ == "__main__":
    visualize_predictions()
    visualize_multi_step()
    evaluate_model_vs_baseline()
