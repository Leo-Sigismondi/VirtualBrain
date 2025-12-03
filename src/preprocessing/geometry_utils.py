import torch

def sym_matrix_to_vec(matrix):
    """
    Prende una matrice simmetrica (N, N) e la appiattisce in un vettore.
    Prende solo il triangolo inferiore per evitare duplicati.
    Input: (..., N, N)
    Output: (..., N*(N+1)/2)
    """
    # Estraiamo gli indici del triangolo inferiore
    n = matrix.shape[-1]
    # tril_indices restituisce gli indici (riga, colonna) del triangolo inferiore
    rows, cols = torch.tril_indices(n, n)
    
    # Se l'input è un batch (Batch, N, N), usiamo indicizzazione avanzata
    return matrix[..., rows, cols]

def vec_to_sym_matrix(vec, n):
    """
    Operazione inversa: dal vettore ricostruisce la matrice simmetrica.
    Input: (..., D) dove D = N*(N+1)/2
    Output: (..., N, N)
    """
    # Prepariamo la matrice vuota
    # Gestisce sia singoli vettori che batch
    batch_shape = vec.shape[:-1]
    matrix = torch.zeros(*batch_shape, n, n, device=vec.device, dtype=vec.dtype)
    
    rows, cols = torch.tril_indices(n, n)
    matrix[..., rows, cols] = vec
    matrix[..., cols, rows] = vec # Simmetria
    
    return matrix

def log_euclidean_map(spd_matrix):
    """
    Mappa dal Manifold SPD allo Spazio Tangente (Euclideo).
    Log(C) = U * log(S) * U.T
    """
    # Decomposizione agli autovalori (più stabile di matrix_log generico)
    # eigh è specifico per matrici Hermitiane/Simmetriche (più veloce)
    eigvals, eigvecs = torch.linalg.eigh(spd_matrix)
    
    # Logaritmo degli autovalori (aggiungiamo epsilon per sicurezza numerica)
    log_eigvals = torch.log(torch.clamp(eigvals, min=1e-20))
    
    # Ricostruzione: U * diag(log_s) * U.T
    # Usiamo matmul per gestire i batch correttamente
    log_matrix = eigvecs @ torch.diag_embed(log_eigvals) @ eigvecs.transpose(-2, -1)
    
    return log_matrix

def exp_euclidean_map(tangent_matrix):
    """
    Mappa dallo Spazio Tangente al Manifold SPD.
    Exp(X) = U * exp(S) * U.T
    """
    eigvals, eigvecs = torch.linalg.eigh(tangent_matrix)
    exp_eigvals = torch.exp(eigvals)
    spd_matrix = eigvecs @ torch.diag_embed(exp_eigvals) @ eigvecs.transpose(-2, -1)
    
    return spd_matrix