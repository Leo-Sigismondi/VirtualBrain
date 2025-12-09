import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from src.preprocessing.geometry_utils import log_euclidean_map, sym_matrix_to_vec

class BCIDataset(Dataset):
    def __init__(self, data_dir, sequence_length=64, stride=1):
        """
        Loads continuous subject data and prepares it for sliding window training.
        
        Args:
            data_dir: Path to processed data
            sequence_length: Length of sequences to slice (e.g., 64 steps)
            stride: Step size between consecutive windows (default 1 = max overlap)
                   Higher values = less overlap = more distinct consecutive samples
                   
        WHY STRIDE MATTERS:
        - stride=1: Windows [0:64], [1:65], [2:66]... → consecutive very similar
        - stride=16: Windows [0:64], [16:80], [32:96]... → more distinct samples
        - Higher stride = harder prediction task = GRU must learn real dynamics
        """
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.stride = stride
        self.files = [f for f in os.listdir(data_dir) if f.endswith('_cov.npy')]
        
        self.subjects_data = [] # List of tensors (T_i, Features)
        self.subjects_labels = [] # List of tensors (T_i,)
        
        # Index mapping: maps global_idx -> (subject_idx, start_offset)
        self.valid_indices = []
        
        print(f"Loading continuous dataset from {data_dir}...")
        
        total_timesteps = 0
        
        for i, f_name in enumerate(self.files):
            # Load covariances (N_Windows, 22, 22)
            covs = np.load(os.path.join(data_dir, f_name))
            
            # Load labels
            lbl_name = f_name.replace('_cov.npy', '_labels.npy')
            lbls = np.load(os.path.join(data_dir, lbl_name))
            
            # Convert to Tensor
            covs_tensor = torch.from_numpy(covs).float()
            lbls_tensor = torch.from_numpy(lbls).long()
            
            # Geometric Preprocessing: SPD -> Tangent Space -> Vector
            # 1. Log Map
            tangent_matrices = log_euclidean_map(covs_tensor)
            # 2. Vectorize (Input size becomes 22*23/2 = 253)
            vectors = sym_matrix_to_vec(tangent_matrices)
            
            self.subjects_data.append(vectors)
            self.subjects_labels.append(lbls_tensor)
            
            n_samples = vectors.shape[0]
            total_timesteps += n_samples
            
            # Calculate valid start indices for this subject
            # We can start from 0 up to n_samples - sequence_length
            if n_samples >= sequence_length:
                # Add (subject_idx, start_offset) for every valid window
                # Use stride to control overlap between consecutive windows
                for start in range(0, n_samples - sequence_length + 1, self.stride):
                    self.valid_indices.append((i, start))
            else:
                print(f"Warning: Subject {f_name} has {n_samples} samples, shorter than seq_len {sequence_length}")

        print(f"Dataset Loaded.")
        print(f"  Total Subjects: {len(self.subjects_data)}")
        print(f"  Total Timesteps: {total_timesteps}")
        print(f"  Total Training Sequences (Slices): {len(self.valid_indices)}")
        print(f"  Feature Dimension: {self.subjects_data[0].shape[-1]}")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        # Map global index to subject and offset
        subj_idx, start_t = self.valid_indices[idx]
        end_t = start_t + self.sequence_length
        
        # Slice
        data_seq = self.subjects_data[subj_idx][start_t:end_t]
        label_seq = self.subjects_labels[subj_idx][start_t:end_t]
        
        return data_seq, label_seq

# --- TEST ---
if __name__ == "__main__":
    # Quick test
    ds = BCIDataset("data/processed/train", sequence_length=64)
    if len(ds) > 0:
        dl = DataLoader(ds, batch_size=32, shuffle=True)
        batch_data, batch_labels = next(iter(dl))
        print(f"\nBatch Shape: {batch_data.shape}")
        print(f"Label Shape: {batch_labels.shape}")
        
        if torch.isnan(batch_data).any():
            print("ATTENTION: NaNs found!")
        else:
            print("Data clean and ready.")
    else:
        print("Dataset empty.")