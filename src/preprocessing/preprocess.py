import os

# ID Eventi standard del dataset BCI IV 2a
# 769: Left Hand, 770: Right Hand, 771: Foot, 772: Tongue
# MNE legge le annotazioni GDF come stringhe, es: '769', '770'
TARGET_EVENT_IDS = ['769', '770', '771', '772']

def ensure_spd(matrix, epsilon=1e-5):
    """
    Assicura che la matrice sia SPD (Symmetric Positive Definite).
    """
    if not np.allclose(matrix, matrix.T):
        # Forza simmetria se errore numerico minimo
        matrix = (matrix + matrix.T) / 2
    
    eigvals = np.linalg.eigvalsh(matrix)
    
    if np.all(eigvals > 0):
        return matrix, True 
    
    # Regolarizzazione (Shrinkage/Jitter)
    min_eig = np.min(eigvals)
    jitter = abs(min_eig) + epsilon
    matrix_fixed = matrix + np.eye(matrix.shape[0]) * jitter
    
    return matrix_fixed, False 

def extract_and_process_subject(filename, subject_id):
    """
    Carica un file GDF, applica filtri, estrae finestre e calcola covarianze.
    """
    # 1. Caricamento Dati (CORRETTO: rimosso eog=True)
    # MNE leggerà i tipi di canale direttamente dall'header del GDF
    try:
        raw = mne.io.read_raw_gdf(filename, preload=True, verbose='ERROR')
    except Exception as e:
        print(f"Errore caricamento MNE per {filename}: {e}")
        return np.array([]), np.array([])
    
    # 2. Selezione Canali e Filtro
    # BCI IV 2a ha 22 canali EEG. Selezioniamo solo quelli.
    raw.pick_types(eeg=True, eog=False, stim=False, exclude='bads') 
    raw.filter(F_MIN, F_MAX, fir_design='firwin', verbose='ERROR')
    
    # 3. Estrazione Eventi (MIGLIORATA)
    # Estraiamo le annotazioni (es. '769') e le mappiamo a ID interi
    events, event_id_map = mne.events_from_annotations(raw, verbose='ERROR')
    
    # Vogliamo solo gli eventi che corrispondono a Motor Imagery (769...772)
    # Filtriamo l'event_id_map per trovare gli ID interi che MNE ha assegnato
    selected_ids = []
    selected_descriptions = []
    
    for desc, int_id in event_id_map.items():
        if desc in TARGET_EVENT_IDS:
            selected_ids.append(int_id)
            selected_descriptions.append(desc)
    
    if not selected_ids:
        print(f"Nessun evento target trovato in {subject_id}. Eventi trovati: {list(event_id_map.keys())}")
        return np.array([]), np.array([])

    # Creiamo Epochs solo per gli eventi selezionati
    # tmin=0, tmax=4.0 (durata standard trial MI in questo dataset)
    epochs = mne.Epochs(raw, events, event_id=selected_ids, 
                        tmin=0, tmax=4.0, baseline=None, preload=True, verbose='ERROR')
    
    # 4. Sliding Window & Covariance Estimation
    subject_covariances = []
    subject_labels = []
    
    win_samples = int(WINDOW_SIZE * raw.info['sfreq'])
    step_samples = int(STRIDE * raw.info['sfreq'])
    estimator = LedoitWolf(store_precision=False, assume_centered=True)
    
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
                continue # Skip se LedoitWolf fallisce (raro)
            
        if len(trial_sequence) > 0:
            subject_covariances.append(np.array(trial_sequence))
            subject_labels.append(trial_label)

    return np.array(subject_covariances), np.array(subject_labels)

def main():
    # 0. Setup Cartelle
    if not os.path.exists(EXTRACT_PATH):
        os.makedirs(EXTRACT_PATH)
        print(f"Estraendo {DATASET_PATH}...")
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
        
        # Determina se è Training o Evaluation
        if subj_id.endswith('T'):
            output_dir = OUTPUT_PATH_TRAIN
            dataset_type = "Training"
        elif subj_id.endswith('E'):
            output_dir = OUTPUT_PATH_EVAL
            dataset_type = "Evaluation"
        else:
            print(f"Warning: Tipo sconosciuto per {f_name}")
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
            print(f"Warning: Nessun dato valido estratto per {subj_id}")

    print("\n--- PREPROCESSING COMPLETATO ---")
    print(f"File Training processati: {train_count}/9")
    print(f"File Evaluation processati: {eval_count}/9")
    print(f"Training salvati in: {OUTPUT_PATH_TRAIN}")
    print(f"Evaluation salvati in: {OUTPUT_PATH_EVAL}")

if __name__ == "__main__":
    main()