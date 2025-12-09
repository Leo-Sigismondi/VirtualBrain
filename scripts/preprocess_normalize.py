"""
Pre-normalize the vectorized dataset ONCE and save it.

This creates normalized vectorized data that can be used by all models
(VAE, GRU, Diffusion) without recomputing normalization each time.

Run this ONCE before training:
    python scripts/preprocess_normalize.py

Output:
    - data/processed/train_normalized.pt - normalized vectorized data
    - data/processed/normalization_stats.npy - mean/std for denormalization
"""

import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import BCIDataset

# Paths
STATS_PATH = "data/processed/normalization_stats.npy"
NORMALIZED_DATA_PATH = "data/processed/train_normalized.pt"


def main():
    print("\n" + "="*60)
    print("Pre-normalizing Vectorized Dataset (Memory Efficient)")
    print("="*60)
    
    # Check for existing stats
    if os.path.exists(STATS_PATH):
        print("\nFound existing normalization stats!")
        stats = np.load(STATS_PATH, allow_pickle=True).item()
        global_mean = stats['mean']
        global_std = stats['std']
        print(f"Loaded stats - Mean: {global_mean:.6f}, Std: {global_std:.6f}")
    else:
        # Compute global statistics (Streaming approach)
        print("\nComputing normalization stats (Streaming)...")
        dataset = BCIDataset("data/processed/train", sequence_length=64)
        loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0)
        
        count = 0
        mean_sum = 0.0
        var_sum = 0.0
        
        # Pass 1: Mean
        for batch in tqdm(loader, desc="Computing Mean"):
            data = batch[0].numpy()
            mean_sum += data.sum()
            count += data.size
        
        global_mean = mean_sum / count
        
        # Pass 2: Std
        for batch in tqdm(loader, desc="Computing Std"):
            data = batch[0].numpy()
            var_sum += ((data - global_mean) ** 2).sum()
            
        global_std = np.sqrt(var_sum / count)
        
        # Save stats
        stats = {'mean': float(global_mean), 'std': float(global_std)}
        np.save(STATS_PATH, stats)
        print(f"Computed and saved stats - Mean: {global_mean:.6f}, Std: {global_std:.6f}")

    # Create memmap for validation/verification of size
    print("\nPreparing to save normalized data...")
    dataset = BCIDataset("data/processed/train", sequence_length=64)
    N_SAMPLES = len(dataset)
    SHAPE = (N_SAMPLES, 64, 253)
    
    # Use .npy format but write via memmap
    # First create the file with header by saving a dummy array then reopening as memmap
    normalized_path_npy = "data/processed/train_normalized.npy"
    
    # We'll use a streaming write approach with memmap
    print(f"Output file: {normalized_path_npy}")
    print(f"Total shape: {SHAPE}")
    
    fp = np.memmap(normalized_path_npy, dtype='float32', mode='w+', shape=SHAPE)
    
    loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=0)
    
    idx = 0
    for batch in tqdm(loader, desc="Normalizing & Saving"):
        data = batch[0].numpy()  # (B, T, D)
        B = data.shape[0]
        
        # Normalize
        normalized_batch = (data - global_mean) / (global_std + 1e-8)
        
        # Write to memmap
        fp[idx : idx + B] = normalized_batch.astype(np.float32)
        idx += B
        
        # Flush periodically
        if idx % 5000 == 0:
            fp.flush()
            
    fp.flush()
    del fp  # Close memmap
    
    print("\n" + "="*60)
    print("✓ Pre-normalization Complete!")
    print("="*60 + "\n")



if __name__ == "__main__":
    main()
