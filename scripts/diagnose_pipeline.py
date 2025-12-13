"""
Pipeline Diagnostics - Isolate VAE vs GRU vs Diffusion Issues
Tests each component separately to find the bottleneck.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import matplotlib.pyplot as plt
from src.preprocessing.geometry_utils import vec_to_sym_matrix, exp_euclidean_map, riemannian_distance
from src.models.vae import ImprovedVAE
from src.models.gru import TemporalGRU
from src.models.diffusion import TangentDiffusion
from src.data.dataset import BCIDataset

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    print("="*60)
    print("PIPELINE DIAGNOSTICS")
    print("="*60)
    
    # Load models
    print("\nLoading models...")
    vae = ImprovedVAE(input_dim=253, latent_dim=32).to(DEVICE)
    vae.load_state_dict(torch.load("checkpoints/vae/vae_dynamics_latent32_best.pth", map_location=DEVICE))
    vae.eval()
    
    gru = TemporalGRU(latent_dim=32, hidden_dim=128, num_layers=2).to(DEVICE)
    gru.load_state_dict(torch.load("checkpoints/gru/gru_multistep_L32_H128_2L_best.pth", map_location=DEVICE))
    gru.eval()
    
    diffusion = TangentDiffusion(tangent_dim=253, hidden_dim=512, condition_dim=32).to(DEVICE)
    diffusion.load_state_dict(torch.load("checkpoints/diffusion/tangent_diffusion_best.pth", map_location=DEVICE))
    diffusion.eval()
    
    # Load data
    dataset = BCIDataset("data/processed/train", sequence_length=64)
    real_seq = dataset[0][0].to(DEVICE)  # (64, 253)
    
    # Load GRU normalization stats
    l_stats = np.load("checkpoints/gru/latent_norm_stats_multistep.npy", allow_pickle=True).item()
    l_mean = torch.tensor(l_stats['mean']).to(DEVICE)
    l_std = torch.tensor(l_stats['std']).to(DEVICE)
    
    print("[OK] Setup complete\n")
    
    # =========================================================================
    # TEST 1: VAE Reconstruction Quality
    # =========================================================================
    print("="*60)
    print("TEST 1: VAE Reconstruction (Is the VAE working?)")
    print("="*60)
    
    with torch.no_grad():
        # Encode and decode
        z, _ = vae.encode(real_seq)  # (64, 32)
        reconstructed = vae.decode(z)  # (64, 253)
        
        # Compare
        recon_mse = ((real_seq - reconstructed)**2).mean().item()
        recon_corr = torch.corrcoef(torch.stack([real_seq.flatten(), reconstructed.flatten()]))[0,1].item()
        
        print(f"  Reconstruction MSE:  {recon_mse:.6f}")
        print(f"  Reconstruction Corr: {recon_corr:.4f}")
        
        if recon_corr > 0.9:
            print("  ✅ VAE GOOD: High reconstruction correlation")
        elif recon_corr > 0.7:
            print("  ⚠️ VAE OK: Moderate reconstruction")
        else:
            print("  ❌ VAE PROBLEM: Poor reconstruction!")
    
    # =========================================================================
    # TEST 2: Diffusion with REAL VAE Latents (Skip GRU)
    # =========================================================================
    print("\n" + "="*60)
    print("TEST 2: Diffusion with REAL Latents (Bypass GRU)")
    print("="*60)
    
    with torch.no_grad():
        # Use REAL VAE latents (not GRU predicted)
        real_z = z[32:33]  # Take middle frame latent (1, 32)
        
        # Generate with diffusion conditioned on REAL latent
        gen_real_cond = diffusion.sample_ddim(shape=(1, 1, 253), condition=real_z, steps=50)
        gen_real_cond = gen_real_cond.squeeze()  # (253,)
        
        # Compare to actual real frame
        real_frame = real_seq[32]  # (253,)
        
        diff_mse = ((real_frame - gen_real_cond)**2).mean().item()
        diff_corr = torch.corrcoef(torch.stack([real_frame.flatten(), gen_real_cond.flatten()]))[0,1].item()
        
        # SPD validity check
        gen_spd = exp_euclidean_map(vec_to_sym_matrix(gen_real_cond, 22))
        eigvals = torch.linalg.eigvalsh(gen_spd)
        is_spd = (eigvals > 0).all().item()
        
        print(f"  Generated vs Real MSE:  {diff_mse:.4f}")
        print(f"  Generated vs Real Corr: {diff_corr:.4f}")
        print(f"  SPD Valid: {is_spd}")
        
        if diff_corr > 0.5:
            print("  ✅ DIFFUSION GOOD: Generates similar to real when given real latents")
        elif diff_corr > 0.2:
            print("  ⚠️ DIFFUSION OK: Moderate similarity")
        else:
            print("  ❌ DIFFUSION PROBLEM: Poor generation even with real latents!")
    
    # =========================================================================
    # TEST 3: GRU Latent Prediction Quality
    # =========================================================================
    print("\n" + "="*60)
    print("TEST 3: GRU Prediction Quality")
    print("="*60)
    
    with torch.no_grad():
        # Normalize latents for GRU
        z_norm = (z - l_mean) / (l_std + 1e-8)
        
        # Predict from seed
        SEED = 5
        z_seed = z_norm[:SEED].unsqueeze(0)  # (1, 5, 32)
        z_pred_norm = gru.generate_sequence(z_seed, 64 - SEED)  # (1, 59, 32)
        
        # Denormalize - ensure proper broadcasting
        z_pred_all = z_pred_norm.squeeze(0) * (l_std + 1e-8) + l_mean  
        z_real = z[SEED:].float()  # (59, 32) - ground truth
        
        # Truncate predictions to match real sequence length
        z_pred = z_pred_all[:len(z_real)].float()  # Match length
        
        gru_mse = ((z_real - z_pred)**2).mean().item()
        gru_corr = torch.corrcoef(torch.stack([z_real.flatten(), z_pred.flatten()]))[0,1].item()
        
        # Check dynamics (velocity)
        real_vel = (z_real[1:] - z_real[:-1]).abs().mean().item()
        pred_vel = (z_pred[1:] - z_pred[:-1]).abs().mean().item()
        vel_ratio = pred_vel / (real_vel + 1e-8)
        
        print(f"  GRU vs Real Latent MSE:  {gru_mse:.6f}")
        print(f"  GRU vs Real Latent Corr: {gru_corr:.4f}")
        print(f"  Velocity Ratio (pred/real): {vel_ratio:.2f}")
        
        if gru_corr > 0.5 and 0.5 < vel_ratio < 2.0:
            print("  ✅ GRU GOOD: Accurate latent predictions")
        elif gru_corr > 0.2:
            print("  ⚠️ GRU OK: Moderate prediction")
        else:
            print("  ❌ GRU PROBLEM: Poor latent predictions!")
    
    # =========================================================================
    # TEST 4: Full Pipeline (GRU Latents → Diffusion)
    # =========================================================================
    print("\n" + "="*60)
    print("TEST 4: Full Pipeline (GRU → Diffusion)")
    print("="*60)
    
    with torch.no_grad():
        # Use GRU-predicted latent at frame 32
        gru_z = z_pred[32-SEED:33-SEED]  # (1, 32)
        
        # Generate with diffusion
        gen_gru_cond = diffusion.sample_ddim(shape=(1, 1, 253), condition=gru_z, steps=50)
        gen_gru_cond = gen_gru_cond.squeeze()
        
        # Compare to real frame
        full_mse = ((real_frame - gen_gru_cond)**2).mean().item()
        full_corr = torch.corrcoef(torch.stack([real_frame.flatten(), gen_gru_cond.flatten()]))[0,1].item()
        
        print(f"  Full Pipeline vs Real MSE:  {full_mse:.4f}")
        print(f"  Full Pipeline vs Real Corr: {full_corr:.4f}")
        
        # Compare Test 2 vs Test 4 to see how much GRU hurts
        degradation = diff_corr - full_corr
        print(f"\n  Degradation from GRU: {degradation:.4f}")
        
        if degradation < 0.1:
            print("  ✅ GRU adds minimal error")
        else:
            print("  ❌ GRU is the bottleneck!")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"  VAE Reconstruction:     {recon_corr:.3f} (>0.9 good)")
    print(f"  Diffusion (real cond):  {diff_corr:.3f} (>0.5 good)")
    print(f"  GRU Latent Prediction:  {gru_corr:.3f} (>0.5 good)")
    print(f"  Full Pipeline:          {full_corr:.3f}")
    print()
    
    bottleneck = min([
        ("VAE", recon_corr),
        ("Diffusion", diff_corr),
        ("GRU", gru_corr)
    ], key=lambda x: x[1])
    print(f"  🎯 BOTTLENECK: {bottleneck[0]} (lowest correlation: {bottleneck[1]:.3f})")

if __name__ == "__main__":
    main()
