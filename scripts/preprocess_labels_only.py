"""
Pre-extract labels from the dataset and save them associated with normalized data.
This ensures we have a separate labels file that is perfectly aligned with `train_normalized.npy`.

Output:
    - data/processed/train_labels.npy - Class labels (N, 64)
"""

import os
import sys
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import BCIDataset

def main():
    print("\n" + "="*60)
    print("Extracting aligned labels from BCIDataset")
    print("="*60)
    
    # 1. Load Dataset (Identical settings to preprocess_normalize.py)
    dataset = BCIDataset("data/processed/train", sequence_length=64)
    N_SAMPLES = len(dataset)
    SHAPE = (N_SAMPLES, 64)
    
    print(f"\nDataset loaded. Total samples: {N_SAMPLES}")
    
    # 2. Prepare output file (Memmap)
    output_path = "data/processed/train_labels.npy"
    print(f"Output file: {output_path}")
    print(f"Shape: {SHAPE}")
    
    # Save as int8 (classes 0-4 fits easily)
    fp = np.memmap(output_path, dtype='int8', mode='w+', shape=SHAPE)
    
    # 3. Iterate and Save
    # Must use same batch size/shuffle settings as normalization script
    loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=0)
    
    idx = 0
    for batch in tqdm(loader, desc="Extracting Labels"):
        # batch is [data, labels]
        labels = batch[1].numpy()  # (B, T)
        B = labels.shape[0]
        
        fp[idx : idx + B] = labels.astype(np.int8)
        idx += B
        
        if idx % 5000 == 0:
            fp.flush()
            
    fp.flush()
    del fp
    
    print("\n" + "="*60)
    print("✓ Label Extraction Complete!")
    print(f"Saved to {output_path}")
    print("="*60 + "\n")

    # Verify
    labels = np.load(output_path, mmap_mode='r')
    print("Verification:")
    print(f"  Shape: {labels.shape}")
    print(f"  Unique classes: {np.unique(labels)}")
    print(f"  First sample classes: {labels[0]}")

if __name__ == "__main__":
    main()
