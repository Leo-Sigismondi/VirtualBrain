"""
Shared Data Utilities for VirtualBrain

This module provides centralized functions for:
- Loading pre-normalized datasets (memmap)
- Loading normalization statistics
- Encoding tangent data to VAE latent space

All models (VAE, GRU, Diffusion) should use these utilities
to ensure consistent data handling.
"""

import os
import numpy as np
import torch
from typing import Dict, Tuple, Optional


# ============================================================================
# Configuration - Single source of truth for data paths
# ============================================================================

DATA_DIR = "data/processed"
NORMALIZED_DATA_PATH = f"{DATA_DIR}/train_normalized.npy"
NORM_STATS_PATH = f"{DATA_DIR}/normalization_stats.npy"
LABELS_PATH = f"{DATA_DIR}/train_labels_windows.npy"

# Data dimensions
INPUT_DIM = 253  # Vectorized lower triangle of 22x22 symmetric matrix
N_CHANNELS = 22
SEQUENCE_LENGTH = 64
NUM_CLASSES = 5  # 0=rest, 1-4=motor imagery classes


# ============================================================================
# Data Loading Functions
# ============================================================================

def get_normalization_stats(path: str = NORM_STATS_PATH) -> Dict[str, float]:
    """
    Load normalization statistics (mean, std) from saved file.
    
    Returns:
        Dict with 'mean' and 'std' keys
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Normalization stats not found at {path}. "
            "Run scripts/preprocess_normalize.py first."
        )
    
    stats = np.load(path, allow_pickle=True).item()
    return stats


def load_normalized_dataset(
    path: str = NORMALIZED_DATA_PATH,
    return_stats: bool = True,
    sequence_length: int = SEQUENCE_LENGTH,
    input_dim: int = INPUT_DIM
) -> Tuple[np.memmap, Optional[Dict[str, float]]]:
    """
    Load pre-normalized dataset as memory-mapped array.
    
    Args:
        path: Path to normalized .npy file
        return_stats: Whether to also load normalization stats
        sequence_length: Expected sequence length per sample
        input_dim: Expected feature dimension
        
    Returns:
        Tuple of (data, stats) if return_stats=True, else just data
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Normalized dataset not found at {path}. "
            "Run scripts/preprocess_normalize.py first."
        )
    
    # Calculate shape from file size
    file_size = os.path.getsize(path)
    element_size = 4  # float32
    feature_size = sequence_length * input_dim * element_size
    num_samples = file_size // feature_size
    
    shape = (num_samples, sequence_length, input_dim)
    
    data = np.memmap(path, dtype='float32', mode='r', shape=shape)
    
    if return_stats:
        stats = get_normalization_stats()
        return data, stats
    
    return data


def load_window_labels(path: str = LABELS_PATH) -> np.ndarray:
    """
    Load per-window activity labels.
    
    Labels use "primary activity" strategy:
    - If window has >= 8 steps of an activity, labeled as that activity
    - Otherwise labeled as rest (0)
    
    Args:
        path: Path to labels .npy file
        
    Returns:
        numpy array of shape (num_samples,) with int64 labels
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Window labels not found at {path}. "
            "Run scripts/preprocess_normalize.py first."
        )
    
    return np.load(path)


def load_normalized_dataset_with_labels(
    data_path: str = NORMALIZED_DATA_PATH,
    labels_path: str = LABELS_PATH,
    return_stats: bool = True
) -> Tuple[np.memmap, np.ndarray, Optional[Dict[str, float]]]:
    """
    Load normalized dataset along with window labels.
    
    Args:
        data_path: Path to normalized data
        labels_path: Path to window labels
        return_stats: Whether to also return normalization stats
        
    Returns:
        Tuple of (data, labels, stats) if return_stats=True
        Tuple of (data, labels) otherwise
    """
    data, stats = load_normalized_dataset(data_path, return_stats=True)
    labels = load_window_labels(labels_path)
    
    assert len(data) == len(labels), \
        f"Data/label mismatch: {len(data)} samples vs {len(labels)} labels"
    
    if return_stats:
        return data, labels, stats
    return data, labels


def denormalize(data: torch.Tensor, stats: Dict[str, float]) -> torch.Tensor:
    """
    Denormalize data from normalized space back to original scale.
    
    Args:
        data: Normalized tensor (any shape)
        stats: Dict with 'mean' and 'std' keys
        
    Returns:
        Denormalized tensor
    """
    return data * stats['std'] + stats['mean']


def normalize(data: torch.Tensor, stats: Dict[str, float]) -> torch.Tensor:
    """
    Normalize data to zero mean, unit variance.
    
    Args:
        data: Original tensor (any shape)
        stats: Dict with 'mean' and 'std' keys
        
    Returns:
        Normalized tensor
    """
    return (data - stats['mean']) / stats['std']


# ============================================================================
# VAE Encoding Functions
# ============================================================================

def encode_to_latent(
    vae,
    data,
    device: str = 'cuda',
    batch_size: int = 512
) -> torch.Tensor:
    """
    Encode tangent space data to VAE latent space in batches.
    
    Args:
        vae: Trained VAE model (in eval mode)
        data: Tensor or numpy array of shape (batch, seq_len, input_dim)
        device: Device to use for encoding
        batch_size: Batch size for processing
        
    Returns:
        Latent tensor of shape (batch, seq_len, latent_dim)
    """
    vae.eval()
    num_samples = len(data)
    all_latents = []
    
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            batch = data[i:i + batch_size]
            
            # Convert to tensor if needed
            if isinstance(batch, (np.ndarray, np.memmap)):
                batch = torch.from_numpy(batch).float()
            
            batch = batch.to(device)
            current_batch_size, seq_len, input_dim = batch.shape
            
            # Flatten for VAE: (B*T, D)
            flat = batch.view(-1, input_dim)
            latent_mu, _ = vae.encode(flat)
            
            # Reshape back: (B, T, latent_dim)
            latent_dim = latent_mu.shape[-1]
            batch_latents = latent_mu.view(current_batch_size, seq_len, latent_dim)
            
            # Move to CPU to conserve GPU memory
            all_latents.append(batch_latents.cpu())
            
    return torch.cat(all_latents, dim=0)


def decode_from_latent(
    vae,
    latent: torch.Tensor,
    device: str = 'cuda'
) -> torch.Tensor:
    """
    Decode VAE latent vectors back to tangent space.
    
    Args:
        vae: Trained VAE model (in eval mode)
        latent: Tensor of shape (batch, seq_len, latent_dim)
        device: Device to use
        
    Returns:
        Reconstructed tensor of shape (batch, seq_len, input_dim)
    """
    batch_size, seq_len, latent_dim = latent.shape
    
    # Flatten: (B*T, latent_dim)
    flat = latent.view(-1, latent_dim).to(device)
    
    with torch.no_grad():
        reconstructed = vae.decode(flat)
    
    # Reshape back: (B, T, input_dim)
    input_dim = reconstructed.shape[-1]
    return reconstructed.view(batch_size, seq_len, input_dim)


# ============================================================================
# Dataset Classes
# ============================================================================

class NormalizedDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset wrapper for normalized memmap data.
    """
    def __init__(self, data: np.memmap, indices: np.ndarray = None):
        self.data = data
        self.indices = indices if indices is not None else np.arange(len(data))
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        return (torch.from_numpy(self.data[real_idx].copy()).float(),)


class LatentDataset(torch.utils.data.Dataset):
    """
    Dataset of pre-encoded VAE latent sequences.
    
    Useful for training GRU without re-encoding each batch.
    """
    def __init__(self, latent_data: torch.Tensor, indices: np.ndarray = None):
        self.data = latent_data
        self.indices = indices if indices is not None else np.arange(len(latent_data))
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        return (self.data[real_idx],)


class LabeledLatentDataset(torch.utils.data.Dataset):
    """
    Dataset of pre-encoded VAE latent sequences WITH class labels.
    
    For training conditional GRU that generates class-specific trajectories.
    """
    def __init__(
        self, 
        latent_data: torch.Tensor,
        labels: np.ndarray,
        indices: np.ndarray = None
    ):
        self.data = latent_data
        self.labels = torch.from_numpy(labels).long() if isinstance(labels, np.ndarray) else labels
        self.indices = indices if indices is not None else np.arange(len(latent_data))
        
        assert len(self.data) == len(self.labels), \
            f"Data/label size mismatch: {len(self.data)} vs {len(self.labels)}"
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        return self.data[real_idx], self.labels[real_idx]


# ============================================================================
# Helper Functions
# ============================================================================

def create_train_val_split(
    total_size: int,
    val_ratio: float = 0.1,
    seed: int = 42,
    temporal_split: bool = True,
    sequence_length: int = SEQUENCE_LENGTH
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create train/val index split with optional temporal splitting to prevent leakage.
    
    With stride=1, consecutive windows [0:64] and [1:65] share 63 samples.
    Random shuffling causes data leakage between train/val sets.
    
    Args:
        total_size: Total number of samples
        val_ratio: Fraction for validation
        seed: Random seed (only used if temporal_split=False)
        temporal_split: If True, val comes from end of data to prevent leakage
        sequence_length: Window size (used to create gap between train/val)
        
    Returns:
        Tuple of (train_indices, val_indices)
    """
    if temporal_split:
        # Temporal split: last val_ratio of data is validation
        # Add sequence_length gap to prevent any overlap
        val_size = int(val_ratio * total_size)
        gap = sequence_length  # Ensure no overlap between train/val windows
        
        train_end = total_size - val_size - gap
        train_indices = np.arange(train_end)
        val_indices = np.arange(total_size - val_size, total_size)
        
        print(f"[Data Split] Temporal split: train[0:{train_end}], val[{total_size - val_size}:{total_size}]")
        print(f"             Gap of {gap} samples prevents window overlap")
    else:
        # Original random shuffle (for backward compatibility)
        rng = np.random.RandomState(seed)
        indices = np.arange(total_size)
        rng.shuffle(indices)
        
        val_size = int(val_ratio * total_size)
        train_indices = indices[val_size:]
        val_indices = indices[:val_size]
        
        print(f"[Data Split] Random shuffle (WARNING: may cause leakage with overlapping windows)")
    
    return train_indices, val_indices


def get_data_loaders(
    batch_size: int = 32,
    val_ratio: float = 0.1,
    num_workers: int = 0,
    temporal_split: bool = True
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, Dict]:
    """
    Convenience function to get train/val DataLoaders and stats.
    
    Args:
        batch_size: Batch size for DataLoaders
        val_ratio: Fraction of data for validation
        num_workers: Number of worker processes
        temporal_split: If True, use temporal split (for sequence models).
                       If False, use random split (for frame-level models like VAE).
    
    Returns:
        Tuple of (train_loader, val_loader, norm_stats)
    """
    data, stats = load_normalized_dataset()
    train_idx, val_idx = create_train_val_split(len(data), val_ratio, temporal_split=temporal_split)
    
    train_dataset = NormalizedDataset(data, train_idx)
    val_dataset = NormalizedDataset(data, val_idx)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    return train_loader, val_loader, stats
