import os
import zipfile
import numpy as np
import mne
from sklearn.covariance import LedoitWolf
from tqdm import tqdm

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

def ensure_spd(matrix, epsilon=1e-5):
    """
    Ensures the matrix is SPD (Symmetric Positive Definite).
    """
    if not np.allclose(matrix, matrix.T):
        # Force symmetry if numerical error
        matrix = (matrix + matrix.T) / 2
    
    eigvals = np.linalg.eigvalsh(matrix)
    
    if np.all(eigvals > 0):
        return matrix, True 
    
    # Regularization (Shrinkage/Jitter)
    min_eig = np.min(eigvals)
    jitter = abs(min_eig) + epsilon
    matrix_fixed = matrix + np.eye(matrix.shape[0]) * jitter
    
    return matrix_fixed, False 

def extract_and_process_subject(filename, subject_id):
    """
    Loads a GDF file, applies filters, extracts windows and computes covariances.
    """
    # 1. Load Data (FIXED: removed eog=True)
    # MNE will read channel types directly from GDF header
    try:
        raw = mne.io.read_raw_gdf(filename, preload=True, verbose='ERROR')
    except Exception as e:
        print(f"Error loading MNE for {filename}: {e}")
        return np.array([]), np.array([])
    
    # 2. Channel Selection and Filtering
    # BCI IV 2a has 22 EEG channels. Select only those.
    raw.pick('eeg', exclude='bads') 
    raw.filter(F_MIN, F_MAX, fir_design='firwin', verbose='ERROR')
    
    # 3. Event Extraction (IMPROVED)
    # Extract annotations (e.g. '769') and map to integer IDs
    events, event_id_map = mne.events_from_annotations(raw, verbose='ERROR')
    
    # We only want events corresponding to Motor Imagery (769...772)
    # Filter event_id_map to find integer IDs that MNE assigned
    selected_ids = []
    selected_descriptions = []
    
    for desc, int_id in event_id_map.items():
        if desc in TARGET_EVENT_IDS:
            selected_ids.append(int_id)
            selected_descriptions.append(desc)
    
    if not selected_ids:
        print(f"No target events found in {subject_id}. Events found: {list(event_id_map.keys())}")
        return np.array([]), np.array([])

    # Create Epochs only for selected events
    # tmin=0, tmax=4.0 (standard MI trial duration in this dataset)
    epochs = mne.Epochs(raw, events, event_id=selected_ids, 
                        tmin=0, tmax=4.0, baseline=None, preload=True, verbose='ERROR')
    
    # 4. Sliding Window & Covariance Estimation
    subject_covariances = []
    subject_labels = []
    
    win_samples = int(WINDOW_SIZE * raw.info['sfreq'])
    step_samples = int(STRIDE * raw.info['sfreq'])
    estimator = LedoitWolf(store_precision=False, assume_centered=False)  # FIXED: EEG is NOT centered!
    
    n_epochs = len(epochs)
    if n_epochs == 0:
        return np.array([]), np.array([])

    epoch_data = epochs.get_data(copy=False) # Shape: (N_Trials, Channels, Time)
    epoch_events = epochs.events[:, 2]       # Shape: (N_Trials,)

    for i in range(n_epochs):
        trial_data = epoch_data[i] # (22, 1001)
        trial_label = epoch_events[i]
        
        trial_sequence = []
        
        # Sliding Window
        for t in range(0, trial_data.shape[1] - win_samples, step_samples):
            window = trial_data[:, t : t + win_samples]
            
            # Covariance
            try:
                cov = estimator.fit(window.T).covariance_
                cov, _ = ensure_spd(cov)
                trial_sequence.append(cov)
            except Exception:
                continue # Skip if LedoitWolf fails (rare)
            
        if len(trial_sequence) > 0:
            subject_covariances.append(np.array(trial_sequence))
            subject_labels.append(trial_label)

    return np.array(subject_covariances), np.array(subject_labels)

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
        
        if os.path.exists(save_file):
            continue
            
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