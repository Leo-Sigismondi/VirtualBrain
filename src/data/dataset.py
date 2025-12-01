import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from src.preprocessing.geometry_utils import log_euclidean_map, sym_matrix_to_vec

class BCIDataset(Dataset):
    def __init__(self, data_dir, sequence_length=5):
        """
        Carica tutti i file dei soggetti e prepara le sequenze.
        """
        self.data_dir = data_dir
        self.files = [f for f in os.listdir(data_dir) if f.endswith('_cov.npy')]
        
        self.sequences = []
        self.labels = []
        
        print(f"Caricamento dataset da {data_dir}...")
        
        for f_name in self.files:
            # Carica covarianze (N_Trials, Seq_Len_Originale, 22, 22)
            covs = np.load(os.path.join(data_dir, f_name))
            
            # Carica label (opzionale, per ora non le usiamo nel training self-supervised)
            lbl_name = f_name.replace('_cov.npy', '_labels.npy')
            lbls = np.load(os.path.join(data_dir, lbl_name))
            
            # Converto in Tensor subito
            covs_tensor = torch.from_numpy(covs).float()
            
            # Preprocessing Geometrico: SPD -> Tangent Space -> Vector
            # Lo facciamo qui una volta per tutte per velocizzare il training
            # 1. Log Map
            tangent_matrices = log_euclidean_map(covs_tensor)
            # 2. Vectorize (Input size diventa 22*23/2 = 253)
            vectors = sym_matrix_to_vec(tangent_matrices)
            
            # Ora vectors ha shape (N_Trials, Seq_Len_Originale, 253)
            
            # Creiamo le sottosequenze per l'RNN
            # Se un trial ha 5 finestre, è già una sequenza perfetta.
            # Se vogliamo fare training "Many-to-Many" (predirre il prossimo step)
            self.sequences.append(vectors)
            self.labels.append(torch.from_numpy(lbls))

        # Concateniamo tutti i soggetti
        self.all_data = torch.cat(self.sequences, dim=0) # (Total_Trials, Seq_Len, Feature_Dim)
        self.all_labels = torch.cat(self.labels, dim=0)
        
        print(f"Dataset Caricato. Shape totale: {self.all_data.shape}")
        # Esempio shape: (2500 trials, 5 steps, 253 features)

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        # Ritorna la sequenza intera. 
        # L'RNN imparerà: Dato x[t], predici x[t+1]
        return self.all_data[idx], self.all_labels[idx]

# --- TEST ---
if __name__ == "__main__":
    # Test veloce
    ds = BCIDataset("data/processed/train")
    dl = DataLoader(ds, batch_size=32, shuffle=True)
    
    batch_data, batch_labels = next(iter(dl))
    print(f"\nBatch Shape: {batch_data.shape}")
    print(f"Feature Dimension (Input per Encoder): {batch_data.shape[-1]}")
    
    # Verifica integrità
    if torch.isnan(batch_data).any():
        print("ATTENZIONE: Trovati NaN nel dataset!")
    else:
        print("Dati puliti e pronti per il training.")