import os
import zipfile
import numpy as np
import mne
from sklearn.covariance import LedoitWolf
from tqdm import tqdm

# Import ensure_spd from centralized geometry_utils
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessing.geometry_utils import ensure_spd

# --- CONFIGURAZIONE ---
DATASET_PATH = "data/BCICIV_2a_gdf.zip"
EXTRACT_PATH = "data/raw/gdf"
OUTPUT_PATH_TRAIN = "data/processed/train"
OUTPUT_PATH_EVAL = "data/processed/eval"

# Parametri di Preprocessing (MODIFY THESE TO CHANGE PREPROCESSING)
SFREQ = 250          # Sampling frequency (don't change - from dataset)
F_MIN, F_MAX = 4, 38 # Frequency band for Motor Imagery (Theta, Alpha, Beta)
WINDOW_SIZE = 1.0    # Window size in seconds (affects input dimension)
STRIDE = 0.25         # Sliding window stride in seconds (affects # of timesteps)

# ID Eventi standard del dataset BCI IV 2a
# 769: Left Hand, 770: Right Hand, 771: Foot, 772: Tongue
# MNE reads GDF annotations as strings, e.g.: '769', '770'
TARGET_EVENT_IDS = ['769', '770', '771', '772'] 

def extract_and_process_subject(filename, subject_id):
    """
    Loads a GDF file, applies filters, and extracts continuous sliding windows.
    Returns:
        covariances: (N_Windows, Channels, Channels)
        labels: (N_Windows,) - 0 for rest, 1-4 for tasks
    """
    # 1. Load Data
    try:
        raw = mne.io.read_raw_gdf(filename, preload=True, verbose='ERROR')
    except Exception as e:
        print(f"Error loading MNE for {filename}: {e}")
        return np.array([]), np.array([])
    
    # 2. Channel Selection and Filtering
    # Select first 22 EEG channels
    raw.pick_channels(raw.ch_names[:22]) 
    raw.filter(F_MIN, F_MAX, fir_design='firwin', verbose='ERROR')
    
    # 3. Extract Events for Labeling
    events, event_id_map = mne.events_from_annotations(raw, verbose='ERROR')
    
    # Create a label array for the entire raw duration
    # Default to 0 (Rest)
    raw_labels = np.zeros(len(raw.times), dtype=int)
    
    # Map events to class IDs (1-4)
    # 769->1, 770->2, 771->3, 772->4
    event_mapping = {
        '769': 1, '770': 2, '771': 3, '772': 4
    }
    
    # Fill labels based on events
    # Standard trial duration is 4 seconds (from t=0 to t=4 relative to event)
    # But we want to capture the transition, so we mark the 4s window as the task.
    sfreq = raw.info['sfreq']
    trial_samples = int(4.0 * sfreq)
    
    for desc, int_id in event_id_map.items():
        if desc in event_mapping:
            class_id = event_mapping[desc]
            # Find all timestamps for this event
            event_indices = events[events[:, 2] == int_id, 0]
            
            for start_idx in event_indices:
                end_idx = start_idx + trial_samples
                if end_idx <= len(raw_labels):
                    raw_labels[start_idx:end_idx] = class_id
    
    # 4. Sliding Window & Covariance Estimation
    # We slide over the CONTINUOUS raw data
    data = raw.get_data() # (Channels, Time)
    n_samples = data.shape[1]
    
    win_samples = int(WINDOW_SIZE * sfreq)
    step_samples = int(STRIDE * sfreq)
    estimator = LedoitWolf(store_precision=False, assume_centered=False)
    
    covariances = []
    window_labels = []
    
    # Slide window
    for t in range(0, n_samples - win_samples, step_samples):
        window = data[:, t : t + win_samples]
        
        # Label for this window: use the mode (most frequent label) or the center label
        # Let's use the label at the center of the window
        center_idx = t + win_samples // 2
        label = raw_labels[center_idx]
        
        # Compute Covariance
        try:
            cov = estimator.fit(window.T).covariance_
            cov, _ = ensure_spd(cov)
            covariances.append(cov)
            window_labels.append(label)
        except Exception:
            continue
            
    return np.array(covariances), np.array(window_labels)

def main():
    # 0. Setup Folders
    if not os.path.exists(EXTRACT_PATH):
        os.makedirs(EXTRACT_PATH)
        print(f"Extracting {DATASET_PATH}...")
        with zipfile.ZipFile(DATASET_PATH, 'r') as zip_ref:
            zip_ref.extractall(EXTRACT_PATH)
            
    os.makedirs(OUTPUT_PATH_TRAIN, exist_ok=True)
    os.makedirs(OUTPUT_PATH_EVAL, exist_ok=True)

    gdf_files = [f for f in os.listdir(EXTRACT_PATH) if f.endswith('.gdf')]
    gdf_files.sort()
    
    train_count = 0
    eval_count = 0

    for f_name in tqdm(gdf_files, desc="Processing Files"):
        full_path = os.path.join(EXTRACT_PATH, f_name)
        subj_id = f_name.split('.')[0] 
        
        # Determine if Training or Evaluation
        if subj_id.endswith('T'):
            output_dir = OUTPUT_PATH_TRAIN
            dataset_type = "Training"
        elif subj_id.endswith('E'):
            output_dir = OUTPUT_PATH_EVAL
            dataset_type = "Evaluation"
        else:
            print(f"Warning: Unknown type for {f_name}")
            continue
        
        save_file = os.path.join(output_dir, f"{subj_id}_cov.npy")
        label_file = os.path.join(output_dir, f"{subj_id}_labels.npy")
        
        # FORCE OVERWRITE for new continuous data
        # if os.path.exists(save_file):
        #     continue
            
        cov_seqs, labels = extract_and_process_subject(full_path, subj_id)
        
        if len(cov_seqs) > 0:
            np.save(save_file, cov_seqs)
            np.save(label_file, labels)
            if subj_id.endswith('T'):
                train_count += 1
            else:
                eval_count += 1
        else:
            print(f"Warning: No valid data extracted for {subj_id}")

    print("\n--- PREPROCESSING COMPLETE ---")
    print(f"Training files processed: {train_count}/9")
    print(f"Evaluation files processed: {eval_count}/9")
    print(f"Training saved in: {OUTPUT_PATH_TRAIN}")
    print(f"Evaluation saved in: {OUTPUT_PATH_EVAL}")

if __name__ == "__main__":
    main()