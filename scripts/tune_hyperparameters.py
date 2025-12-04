"""
Hyperparameter Tuning Script
Automates the search for the best VAE and GRU configuration.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd
from src.training.train_vae import train_vae
from src.training.train_gru import train_gru

def main():
    print("="*60)
    print("STARTING HYPERPARAMETER TUNING")
    print("="*60)
    
    # Search Space
    LATENT_DIMS = [16, 32]
    HIDDEN_DIMS = [32, 64, 128]
    NUM_LAYERS = [1, 2]
    
    results = []
    
    # 1. Iterate over VAE Latent Dimensions
    for latent_dim in LATENT_DIMS:
        print(f"\n\n>>> TUNING VAE (Latent Dim = {latent_dim}) <<<")
        
        # Train VAE
        vae_config = {
            'latent_dim': latent_dim,
            'epochs': 500,  # Reduced for speed during tuning
            'batch_size': 64,
            'lr': 1e-3
        }
        vae_loss, vae_path, stats_path = train_vae(vae_config)
        
        # 2. Iterate over GRU Hyperparameters
        for hidden_dim in HIDDEN_DIMS:
            for layers in NUM_LAYERS:
                print(f"\n   >>> TUNING GRU (Hidden={hidden_dim}, Layers={layers}) <<<")
                
                gru_config = {
                    'latent_dim': latent_dim,
                    'hidden_dim': hidden_dim,
                    'num_layers': layers,
                    'epochs': 200,  # Reduced for speed
                    'vae_path': vae_path,
                    'stats_path': stats_path,
                    'lr': 1e-3
                }
                
                gru_loss, gru_path = train_gru(gru_config)
                
                # Log results
                results.append({
                    'latent_dim': latent_dim,
                    'hidden_dim': hidden_dim,
                    'num_layers': layers,
                    'vae_loss': vae_loss,
                    'gru_loss': gru_loss,
                    'vae_path': vae_path,
                    'gru_path': gru_path
                })
                
                # Save intermediate results
                df = pd.DataFrame(results)
                df.to_csv("tuning_results.csv", index=False)

    # 3. Report Best Configuration
    print("\n" + "="*60)
    print("TUNING COMPLETE")
    print("="*60)
    
    df = pd.DataFrame(results)
    print("\nAll Results:")
    print(df)
    
    # Find best GRU model (lowest GRU val loss)
    best_run = df.loc[df['gru_loss'].idxmin()]
    
    print("\n🏆 BEST CONFIGURATION 🏆")
    print(f"Latent Dim: {best_run['latent_dim']}")
    print(f"Hidden Dim: {best_run['hidden_dim']}")
    print(f"Num Layers: {best_run['num_layers']}")
    print(f"GRU Val Loss: {best_run['gru_loss']:.6f}")
    print(f"VAE Val Loss: {best_run['vae_loss']:.4f}")
    
    print(f"\nBest VAE Path: {best_run['vae_path']}")
    print(f"Best GRU Path: {best_run['gru_path']}")

if __name__ == "__main__":
    main()
