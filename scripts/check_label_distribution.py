import numpy as np
import os
from glob import glob

def check_distribution():
    data_dir = "data/processed/train"
    files = glob(os.path.join(data_dir, "*_labels.npy"))
    
    total_frames = 0
    total_rest = 0
    total_task = 0
    
    print(f"Checking distribution in {data_dir}...")
    
    for f in files:
        labels = np.load(f)
        n_frames = len(labels)
        n_rest = np.sum(labels == 0)
        n_task = np.sum(labels != 0)
        
        total_frames += n_frames
        total_rest += n_rest
        total_task += n_task
        
        print(f"{os.path.basename(f)}: {n_frames} frames | Rest: {n_rest} ({n_rest/n_frames:.1%}) | Task: {n_task} ({n_task/n_frames:.1%})")
        
    print("\n" + "="*40)
    print("OVERALL STATISTICS")
    print("="*40)
    print(f"Total Frames: {total_frames}")
    print(f"Rest Frames:  {total_rest} ({total_rest/total_frames:.1%})")
    print(f"Task Frames:  {total_task} ({total_task/total_frames:.1%})")

if __name__ == "__main__":
    check_distribution()
