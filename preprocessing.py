import os
import zipfile
import numpy as np
import mne
from sklearn.covariance import LedoitWolf
from tqdm import tqdm

# --- CONFIGURAZIONE ---
DATASET_PATH = "datasets/BCICIV_2a_gdf.zip"
EXTRACT_PATH = "datasets/raw_gdf"
OUTPUT_PATH = "datasets/processed_covariances"

# Parametri di Preprocessing (Quelli discussi)
SFREQ = 250          # Frequenza di campionamento BCI-IV 2a
F_MIN, F_MAX = 4, 38 # Banda per Motor Imagery (Theta, Alpha, Beta)
WINDOW_SIZE = 2.0    # Secondi (500 samples)
STRIDE = 0.5         # Secondi (Step di avanzamento, 125 samples)

# Codici eventi BCI-IV 2a (Motor Imagery)
# 769: Left Hand, 770: Right Hand, 771: Foot, 772: Tongue
EVENT_ID = {'Left Hand': 769, 'Right Hand': 770, 'Foot': 771, 'Tongue': 772}

def ensure_spd(matrix, epsilon=1e-5):
    """
    Controlla se una matrice è SPD. Se ha autovalori piccoli/negativi 
    (dovuti a errori numerici), aggiunge una regolarizzazione (jitter).
    """
    if not np.allclose(matrix, matrix.T):
        raise ValueError("La matrice non è simmetrica!")
    
    # Calcola autovalori
    eigvals = np.linalg.eigvalsh(matrix)
    
    if np.all(eigvals > 0):
        return matrix, True # È già perfetta
    
    # Se non è definita positiva, regolarizza
    # Aggiunge epsilon alla diagonale finché min(eig) > 0
    min_eig = np.min(eigvals)
    jitter = abs(min_eig) + epsilon
    matrix_fixed = matrix + np.eye(matrix.shape[0]) * jitter
    
    return matrix_fixed, False # Restituisce la versione corretta

def extract_and_process_subject(filename, subject_id):
    """
    Carica un file GDF, applica filtri, estrae finestre e calcola covarianze.
    """
    print(f"\nProcessing Subject {subject_id} from {filename}...")
    
    # 1. Caricamento Dati (MNE gestisce i GDF automaticamente)
    # EOG=True serve per rimuoverli dopo, preload=True carica in RAM
    raw = mne.io.read_raw_gdf(filename, preload=True, verbose='ERROR')
    
    # 2. Selezione Canali e Filtro
    # Teniamo solo i 22 canali EEG, scartiamo EOG
    raw.pick_types(eeg=True, eog=False) 
    raw.filter(F_MIN, F_MAX, fir_design='firwin', verbose='ERROR')
    
    # 3. Estrazione Eventi (Trials)
    events, _ = mne.events_from_annotations(raw, verbose='ERROR')
    
    # Filtriamo solo gli eventi che ci interessano (Motor Imagery)
    # Nota: MNE mappa i codici GDF (che sono strani) a interi progressivi. 
    # Bisognerebbe controllare il mapping specifico, ma per ora usiamo 
    # il metodo epoching basato sugli ID standard se presenti.
    # Per semplicità in questo script, tagliamo i trial standard di 4 secondi.
    
    # Creiamo Epochs: prendiamo da 0s a 4s per ogni evento MI
    tmin, tmax = 0, 4.0 
    # Mappiamo gli ID annotazione agli ID interi che vogliamo noi
    # (Questa parte spesso richiede tuning specifico per file GDF, 
    #  qui assumiamo di prendere tutti gli eventi MI standard)
    # Per sicurezza, prendiamo gli eventi che corrispondono ai codici standard
    # Se i codici non matchano direttamente, MNE usa 1,2,3,4. 
    # Qui usiamo un approccio generico: prendiamo tutti gli eventi.
    
    epochs = mne.Epochs(raw, events, tmin=tmin, tmax=tmax, 
                        baseline=None, preload=True, verbose='ERROR')
    
    # 4. Sliding Window & Covariance Estimation
    # Output list
    subject_covariances = []
    subject_labels = []
    
    # Parametri in samples
    win_samples = int(WINDOW_SIZE * raw.info['sfreq'])
    step_samples = int(STRIDE * raw.info['sfreq'])
    
    estimator = LedoitWolf(store_precision=False, assume_centered=True)
    
    n_trials = len(epochs)
    
    for i in range(n_trials):
        # Dati del singolo trial: (Canali, Tempo)
        trial_data = epochs[i].get_data(copy=False)[0] # Shape (22, 1001)
        
        trial_sequence = []
        
        # Scorriamo dentro il trial (Sliding Window)
        for t in range(0, trial_data.shape[1] - win_samples, step_samples):
            window = trial_data[:, t : t + win_samples]
            
            # Calcolo Covarianza (Ledoit-Wolf per stabilità)
            # Input atteso da sklearn: (Samples, Features) -> Quindi trasponiamo (.T)
            cov = estimator.fit(window.T).covariance_
            
            # 5. Controllo e Fix SPD
            cov, was_perfect = ensure_spd(cov)
            
            trial_sequence.append(cov)
            
        if len(trial_sequence) > 0:
            subject_covariances.append(np.array(trial_sequence))
            # Salviamo anche l'evento (label) associato se servisse per classificazione
            subject_labels.append(events[i, 2])

    return np.array(subject_covariances), np.array(subject_labels)

def main():
    # 0. Setup Cartelle
    if not os.path.exists(EXTRACT_PATH):
        os.makedirs(EXTRACT_PATH)
        print(f"Estraendo {DATASET_PATH}...")
        with zipfile.ZipFile(DATASET_PATH, 'r') as zip_ref:
            zip_ref.extractall(EXTRACT_PATH)
            
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    # Trova tutti i file .gdf (es. A01T.gdf, A02T.gdf...)
    # BCI IV 2a ha file tipo A01T.gdf (Training) e A01E.gdf (Evaluation)
    gdf_files = [f for f in os.listdir(EXTRACT_PATH) if f.endswith('.gdf')]
    gdf_files.sort()
    
    all_data_shape = None

    for f_name in tqdm(gdf_files, desc="Processing Files"):
        full_path = os.path.join(EXTRACT_PATH, f_name)
        subj_id = f_name.split('.')[0] # es. A01T
        save_file = os.path.join(OUTPUT_PATH, f"{subj_id}_cov.npy")
        
        # Check se esiste già
        if os.path.exists(save_file):
            print(f"File {save_file} già esistente. Skipping.")
            continue
            
        # Processa
        try:
            cov_seqs, labels = extract_and_process_subject(full_path, subj_id)
            
            # cov_seqs shape: (Num_Trials, Num_Windows_Per_Trial, 22, 22)
            np.save(save_file, cov_seqs)
            np.save(os.path.join(OUTPUT_PATH, f"{subj_id}_labels.npy"), labels)
            
            print(f"Salvato {subj_id}: Shape {cov_seqs.shape}")
            all_data_shape = cov_seqs.shape
            
        except Exception as e:
            print(f"Errore processando {f_name}: {e}")

    print("\n--- PREPROCESSING COMPLETATO ---")
    print(f"I dati sono salvati in: {OUTPUT_PATH}")
    if all_data_shape:
        print(f"Formato tipico dei dati salvati (N_Trials, Windows, Ch, Ch): {all_data_shape}")
        print("NOTA: Ogni file .npy contiene le TRAIETTORIE di covarianza per un soggetto.")

if __name__ == "__main__":
    main()