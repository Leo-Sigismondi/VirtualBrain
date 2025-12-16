"""
Pre-normalize the vectorized dataset ONCE and save it.

This creates normalized vectorized data that can be used by all models
(VAE, GRU, Diffusion) without recomputing normalization each time.

Run this ONCE before training:
    python scripts/preprocess_normalize.py

Output:
    - data/processed/train_normalized.npy - normalized vectorized data (memmap)
    - data/processed/train_labels_windows.npy - per-window activity labels
    - data/processed/normalization_stats.npy - mean/std for denormalization

Label Strategy (Primary Activity):
    - If window contains any activity (class 1-4), use that activity class
    - If window contains multiple activities, use majority among activities
    - If window contains only rest (class 0), label as 0
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
NORMALIZED_DATA_PATH = "data/processed/train_normalized.npy"
LABELS_PATH = "data/processed/train_labels_windows.npy"

# Minimum activity steps to consider a window as having that activity
# Activity segments are ~15-16 steps, so 8 means at least half an activity period
MIN_ACTIVITY_STEPS = 8


def get_primary_activity_label(label_sequence, min_activity_steps=MIN_ACTIVITY_STEPS):
    """
    Get the primary activity label for a window.
    
    Strategy:
    - If any activity (class 1-4) has >= min_activity_steps, use that activity
    - If multiple activities meet threshold, use the one with most steps
    - If no activity meets threshold, label as rest (0)
    
    Args:
        label_sequence: numpy array of shape (seq_len,) with class labels
        min_activity_steps: minimum steps of activity required to label as that class
        
    Returns:
        int: Primary activity label for this window
    """
    # Count steps per class
    unique_labels, counts = np.unique(label_sequence, return_counts=True)
    label_counts = dict(zip(unique_labels, counts))
    
    # Find activity classes (1-4) that meet the threshold
    valid_activities = []
    for label in range(1, 5):  # Classes 1-4
        count = label_counts.get(label, 0)
        if count >= min_activity_steps:
            valid_activities.append((label, count))
    
    if len(valid_activities) == 0:
        # No activity meets threshold - label as rest
        return 0
    elif len(valid_activities) == 1:
        # Single activity meets threshold
        return valid_activities[0][0]
    else:
        # Multiple activities meet threshold - use the one with most steps
        valid_activities.sort(key=lambda x: x[1], reverse=True)
        return valid_activities[0][0]


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
    print("\nPreparing to save normalized data and labels...")
    dataset = BCIDataset("data/processed/train", sequence_length=64)
    N_SAMPLES = len(dataset)
    SHAPE = (N_SAMPLES, 64, 253)
    
    # We'll use a streaming write approach with memmap
    print(f"Output file: {NORMALIZED_DATA_PATH}")
    print(f"Labels file: {LABELS_PATH}")
    print(f"Total shape: {SHAPE}")
    
    fp = np.memmap(NORMALIZED_DATA_PATH, dtype='float32', mode='w+', shape=SHAPE)
    
    # Also collect labels
    all_labels = []
    
    loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=0)
    
    idx = 0
    for batch in tqdm(loader, desc="Normalizing & Saving"):
        data = batch[0].numpy()  # (B, T, D)
        labels = batch[1].numpy()  # (B, T) - per-step labels
        B = data.shape[0]
        
        # Normalize
        normalized_batch = (data - global_mean) / (global_std + 1e-8)
        
        # Write to memmap
        fp[idx : idx + B] = normalized_batch.astype(np.float32)
        idx += B
        
        # Extract primary activity label for each window
        for window_labels in labels:
            primary_label = get_primary_activity_label(window_labels)
            all_labels.append(primary_label)
        
        # Flush periodically
        if idx % 5000 == 0:
            fp.flush()
            
    fp.flush()
    del fp  # Close memmap
    
    # Save labels
    all_labels = np.array(all_labels, dtype=np.int64)
    np.save(LABELS_PATH, all_labels)
    
    # Print label distribution
    print("\n" + "-"*40)
    print("Window Label Distribution:")
    unique, counts = np.unique(all_labels, return_counts=True)
    for u, c in zip(unique, counts):
        pct = 100 * c / len(all_labels)
        print(f"  Class {u}: {c:6d} ({pct:5.1f}%)")
    print("-"*40)
    
    print("\n" + "="*60)
    print("✓ Pre-normalization Complete!")
    print(f"  Data saved to: {NORMALIZED_DATA_PATH}")
    print(f"  Labels saved to: {LABELS_PATH}")
    print("="*60 + "\n")



if __name__ == "__main__":
    main()
