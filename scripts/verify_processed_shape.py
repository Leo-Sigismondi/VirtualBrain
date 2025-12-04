import numpy as np
import os
import glob

def check_processed_shape():
    train_files = glob.glob("data/processed/train/*_cov.npy")
    if not train_files:
        print("No processed training files found.")
        return

    # Check first file
    f = train_files[0]
    data = np.load(f)
    print(f"Checking file: {f}")
    print(f"Shape: {data.shape}")
    
    # Expected shape: (N_trials, N_windows, Channels, Channels)
    # If we have 22 channels, the last two dims should be 22x22
    
    if len(data.shape) == 4:
        n_trials, n_wins, n_ch1, n_ch2 = data.shape
        print(f"Trials: {n_trials}, Windows: {n_wins}, Channels: {n_ch1}x{n_ch2}")
        
        if n_ch1 == 22 and n_ch2 == 22:
            print("SUCCESS: Channel count is 22.")
        else:
            print(f"FAILURE: Channel count is {n_ch1} (expected 22).")
    else:
        print(f"Unexpected shape: {data.shape}")

if __name__ == "__main__":
    check_processed_shape()
