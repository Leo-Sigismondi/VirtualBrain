"""
Fair Baseline Comparison Script
================================
This script compares our GRU model against baselines in a FAIR manner:

The key insight is that we need to compare "apples to apples":
1. ONE-STEP PREDICTION: Given z[t], predict z[t+1]
   - This is the fundamental unit of temporal prediction
   - All models predict just 1 step ahead from ground truth
   
2. SHORT ROLLOUT (e.g., 4 steps): Given z[0], predict z[1]..z[4]
   - Tests error accumulation over a few steps
   - More realistic than 62-step rollout

3. LONG ROLLOUT: Full autoregressive rollout (very hard)
   - Only useful to show how errors compound
   - Not a fair comparison if baselines use ground truth
"""

import torch
import numpy as np
import sys
from pathlib import Path
from torch.utils.data import DataLoader

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.models.vae import ImprovedVAE
from src.models.gru import TemporalGRU
from src.data.dataset import BCIDataset


def compare():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    LATENT_DIM = 32
    HIDDEN_DIM = 128
    INPUT_DIM = 253
    NUM_LAYERS = 2
    SEQUENCE_LENGTH = 64
    
    # Use the new dynamics-trained VAE and multistep GRU
    VAE_PATH = "checkpoints/vae/vae_dynamics_latent32_best.pth"
    GRU_PATH = "checkpoints/gru/gru_multistep_L32_H128_2L_best.pth"

    print("Loading models...")
    vae = ImprovedVAE(input_dim=INPUT_DIM, latent_dim=LATENT_DIM, hidden_dims=[256, 128, 64]).to(device)
    vae.load_state_dict(torch.load(VAE_PATH, map_location=device))
    vae.eval()

    gru = TemporalGRU(latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS).to(device)
    gru.load_state_dict(torch.load(GRU_PATH, map_location=device))
    gru.eval()

    print("Loading dataset...")
    dataset = BCIDataset("data/processed/train", sequence_length=SEQUENCE_LENGTH)
    dataloader = DataLoader(dataset, batch_size=200, shuffle=True)
    
    seq_vectors, _ = next(iter(dataloader))
    seq_vectors = seq_vectors.to(device)
    
    # Load stats
    vae_stats = np.load("checkpoints/vae/vae_norm_stats_dynamics_latent32.npy", allow_pickle=True).item()
    vae_mean = torch.tensor(vae_stats['mean']).to(device)
    vae_std = torch.tensor(vae_stats['std']).to(device)
    
    latent_stats = np.load("checkpoints/gru/latent_norm_stats_multistep.npy", allow_pickle=True).item()
    latent_mean = torch.tensor(latent_stats['mean']).to(device)
    latent_std = torch.tensor(latent_stats['std']).to(device)

    print("Encoding to latent space...")
    with torch.no_grad():
        x = seq_vectors.view(-1, INPUT_DIM)
        x = (x - vae_mean) / vae_std
        mu, _ = vae.encode(x)
        latent_seq = mu.view(seq_vectors.shape[0], SEQUENCE_LENGTH, LATENT_DIM)
        
        # Normalize latents
        latent_norm = (latent_seq - latent_mean) / (latent_std + 1e-8)
    
    ground_truth = latent_norm.cpu().numpy()
    batch_size = ground_truth.shape[0]
    
    print("\n" + "="*60)
    print("FAIR BASELINE COMPARISON")
    print("="*60)
    
    # ==========================================
    # TEST 1: ONE-STEP PREDICTION (FAIR)
    # ==========================================
    print("\n1. ONE-STEP PREDICTION (Given z[t], predict z[t+1])")
    print("-" * 50)
    
    with torch.no_grad():
        gru_one_step_preds = []
        hidden = None
        
        # For each timestep, predict the next one using ground truth as input
        for t in range(SEQUENCE_LENGTH - 1):
            current = latent_norm[:, t:t+1, :]
            next_pred, hidden = gru.predict_next(current, hidden)
            gru_one_step_preds.append(next_pred)
        
        gru_one_step = torch.cat(gru_one_step_preds, dim=1).cpu().numpy()
    
    # Ground truth targets: z[1] to z[T-1]
    gt_one_step = ground_truth[:, 1:, :]
    
    # Naive: predict z[t+1] = z[t]
    naive_one_step = ground_truth[:, :-1, :]
    
    # Momentum: predict z[t+1] = 2*z[t] - z[t-1]
    # Valid from t=1 onwards (need t-1)
    momentum_one_step = np.zeros_like(gt_one_step)
    momentum_one_step[:, 0, :] = naive_one_step[:, 0, :]  # First step same as naive
    momentum_one_step[:, 1:, :] = 2 * ground_truth[:, 1:-1, :] - ground_truth[:, :-2, :]
    
    # Calculate errors
    gru_mae = np.mean(np.abs(gt_one_step - gru_one_step))
    naive_mae = np.mean(np.abs(gt_one_step - naive_one_step))
    momentum_mae = np.mean(np.abs(gt_one_step - momentum_one_step))
    
    print(f"   GRU MAE:      {gru_mae:.6f}")
    print(f"   Naive MAE:    {naive_mae:.6f}")
    print(f"   Momentum MAE: {momentum_mae:.6f}")
    
    gru_vs_naive = (naive_mae - gru_mae) / naive_mae * 100
    print(f"   GRU vs Naive: {gru_vs_naive:+.2f}%")
    
    if gru_mae < naive_mae:
        print("   Result: GRU BEATS Naive! ✅")
        test1_pass = True
    else:
        print("   Result: GRU loses to Naive ❌")
        test1_pass = False
    
    # ==========================================
    # TEST 2: SHORT ROLLOUT (4 steps)
    # ==========================================
    print("\n2. SHORT ROLLOUT (Given z[t], predict z[t+1]...z[t+4])")
    print("-" * 50)
    
    rollout_length = 4
    
    with torch.no_grad():
        all_gru_errors = []
        all_naive_errors = []
        
        # For each starting point, do a short rollout
        for start_t in range(0, SEQUENCE_LENGTH - rollout_length):
            # GRU rollout
            current = latent_norm[:, start_t:start_t+1, :]
            hidden = None
            
            gru_rollout = [current]
            for step in range(rollout_length):
                next_pred, hidden = gru.predict_next(current, hidden)
                gru_rollout.append(next_pred)
                current = next_pred
            
            gru_rollout = torch.cat(gru_rollout, dim=1).cpu().numpy()
            
            # Ground truth for this segment
            gt_segment = ground_truth[:, start_t:start_t+rollout_length+1, :]
            
            # Errors (skip first which is seed)
            gru_error = np.mean(np.abs(gt_segment[:, 1:, :] - gru_rollout[:, 1:, :]))
            
            # Naive rollout: each step uses the previous prediction
            naive_error = 0
            for step in range(1, rollout_length + 1):
                naive_pred = gt_segment[:, 0, :]  # All predictions are just the first state
                target = gt_segment[:, step, :]
                naive_error += np.mean(np.abs(target - naive_pred))
            naive_error /= rollout_length
            
            all_gru_errors.append(gru_error)
            all_naive_errors.append(naive_error)
    
    avg_gru_error = np.mean(all_gru_errors)
    avg_naive_error = np.mean(all_naive_errors)
    
    print(f"   GRU MAE:   {avg_gru_error:.6f}")
    print(f"   Naive MAE: {avg_naive_error:.6f}")
    
    improvement = (avg_naive_error - avg_gru_error) / avg_naive_error * 100
    print(f"   GRU vs Naive: {improvement:+.2f}%")
    
    if avg_gru_error < avg_naive_error:
        print("   Result: GRU BEATS Naive! ✅")
        test2_pass = True
    else:
        print("   Result: GRU loses to Naive ❌")
        test2_pass = False
    
    # ==========================================
    # TEST 3: DIRECTION ACCURACY
    # ==========================================
    print("\n3. DIRECTION PREDICTION (Does z[t+1] - z[t] have correct sign?)")
    print("-" * 50)
    
    with torch.no_grad():
        hidden = None
        gru_preds = []
        
        for t in range(SEQUENCE_LENGTH - 1):
            current = latent_norm[:, t:t+1, :]
            next_pred, hidden = gru.predict_next(current, hidden)
            gru_preds.append(next_pred)
        
        gru_preds = torch.cat(gru_preds, dim=1).cpu().numpy()
    
    # True directions
    true_velocity = ground_truth[:, 1:, :] - ground_truth[:, :-1, :]
    true_direction = np.sign(true_velocity)
    
    # GRU predicted directions
    pred_velocity = gru_preds - ground_truth[:, :-1, :]
    pred_direction = np.sign(pred_velocity)
    
    # Naive direction: always 0 (predicts no change)
    naive_direction = np.zeros_like(true_direction)
    
    # Agreement rate
    gru_agreement = np.mean(true_direction == pred_direction)
    naive_agreement = np.mean(true_direction == naive_direction)
    
    print(f"   GRU direction agreement:   {gru_agreement:.4f}")
    print(f"   Naive direction agreement: {naive_agreement:.4f}")
    
    if gru_agreement > naive_agreement:
        print("   Result: GRU has better direction prediction! ✅")
        test3_pass = True
    else:
        print("   Result: GRU does not have better direction ❌")
        test3_pass = False
    
    # ==========================================
    # SUMMARY
    # ==========================================
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    tests_passed = sum([test1_pass, test2_pass, test3_pass])
    print(f"Tests Passed: {tests_passed}/3")
    
    if test1_pass:
        print("✅ GRU beats naive on 1-step prediction")
    else:
        print("❌ GRU loses to naive on 1-step prediction")
    
    if test2_pass:
        print("✅ GRU beats naive on 4-step rollout")
    else:
        print("❌ GRU loses to naive on 4-step rollout")
    
    if test3_pass:
        print("✅ GRU has better direction prediction")
    else:
        print("❌ GRU does not have better direction prediction")
    
    print("="*60)
    
    if tests_passed >= 2:
        print("OVERALL: Model shows improvement over baseline! 🎉")
    else:
        print("OVERALL: Model needs more work. 📝")
    

if __name__ == "__main__":
    compare()
