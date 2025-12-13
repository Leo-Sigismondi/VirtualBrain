import torch
import math

# √2 constant for isometric scaling
_SQRT2 = math.sqrt(2.0)

def sym_matrix_to_vec(matrix):
    """
    Convert symmetric matrix to vector with ISOMETRIC scaling.
    
    Off-diagonal elements are scaled by √2 to preserve Frobenius norm:
    ||vec||² = ||matrix||_F²
    
    Input: (..., N, N)
    Output: (..., N*(N+1)/2)
    """
    n = matrix.shape[-1]
    rows, cols = torch.tril_indices(n, n, device=matrix.device)
    
    # Extract lower triangle
    vec = matrix[..., rows, cols].clone()
    
    # Create mask for off-diagonal elements (where row != col)
    off_diag_mask = rows != cols
    
    # Scale off-diagonal elements by √2 for isometry
    vec[..., off_diag_mask] = vec[..., off_diag_mask] * _SQRT2
    
    return vec

def vec_to_sym_matrix(vec, n):
    """
    Reconstruct symmetric matrix from vector with ISOMETRIC scaling.
    
    Off-diagonal elements are divided by √2 to reverse the isometric transform.
    
    Input: (..., D) where D = N*(N+1)/2
    Output: (..., N, N)
    """
    batch_shape = vec.shape[:-1]
    rows, cols = torch.tril_indices(n, n, device=vec.device)
    
    # Create mask for off-diagonal elements
    off_diag_mask = rows != cols
    
    # Reverse the √2 scaling for off-diagonal elements
    vec_unscaled = vec.clone()
    vec_unscaled[..., off_diag_mask] = vec_unscaled[..., off_diag_mask] / _SQRT2
    
    # Build matrix
    matrix = torch.zeros(*batch_shape, n, n, device=vec.device, dtype=vec.dtype)
    matrix[..., rows, cols] = vec_unscaled
    
    # Make symmetric: M + M.T - diag(M) to avoid doubling diagonal
    matrix = matrix + matrix.transpose(-2, -1) - torch.diag_embed(matrix.diagonal(dim1=-2, dim2=-1))
    
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

def ensure_spd(matrix, epsilon=1e-8):
    """
    Ensures the matrix is SPD (Symmetric Positive Definite).
    
    Handles numerical errors by:
    1. Forcing symmetry if needed
    2. Adding jitter regularization if eigenvalues are non-positive
    
    Works with both numpy arrays and torch tensors.
    
    Args:
        matrix: (..., N, N) matrix to fix
        epsilon: Minimum eigenvalue threshold
        
    Returns:
        matrix_fixed: SPD-guaranteed matrix
        was_valid: Boolean indicating if matrix was already valid
    """
    is_torch = isinstance(matrix, torch.Tensor)
    
    if is_torch:
        # Force symmetry
        matrix = (matrix + matrix.transpose(-2, -1)) / 2
        
        # Check eigenvalues
        eigvals = torch.linalg.eigvalsh(matrix)
        min_eig = eigvals.min()
        
        if min_eig > 0:
            return matrix, True
        
        # Add jitter to fix non-positive eigenvalues
        jitter = abs(min_eig.item()) + epsilon
        n = matrix.shape[-1]
        eye = torch.eye(n, device=matrix.device, dtype=matrix.dtype)
        matrix_fixed = matrix + eye * jitter
        
        return matrix_fixed, False
    else:
        import numpy as np
        # Numpy version
        if not np.allclose(matrix, matrix.T):
            matrix = (matrix + matrix.T) / 2
        
        eigvals = np.linalg.eigvalsh(matrix)
        
        if np.all(eigvals > 0):
            return matrix, True
        
        min_eig = np.min(eigvals)
        jitter = abs(min_eig) + epsilon
        matrix_fixed = matrix + np.eye(matrix.shape[0]) * jitter
        
        return matrix_fixed, False


def exp_euclidean_map(tangent_matrix, ensure_valid=True):
    """
    Mappa dallo Spazio Tangente al Manifold SPD.
    Exp(X) = U * exp(S) * U.T
    
    IMPORTANT: This guarantees the output is SPD!
    - Symmetric: by construction (U @ diag @ U.T)
    - Positive Definite: exp(eigenvalues) > 0 always
    
    Args:
        tangent_matrix: Input tangent space matrix
        ensure_valid: If True, apply ensure_spd for extra numerical safety
    
    Uses float64 for numerical stability. Output is also float64.
    """
    # Use float64 for numerical stability
    tangent_matrix_64 = tangent_matrix.double()
    
    eigvals, eigvecs = torch.linalg.eigh(tangent_matrix_64)
    exp_eigvals = torch.exp(eigvals)
    
    # Clamp to ensure strictly positive (handles any remaining precision issues)
    exp_eigvals = exp_eigvals.clamp(min=1e-15)
    
    spd_matrix = eigvecs @ torch.diag_embed(exp_eigvals) @ eigvecs.transpose(-2, -1)
    
    # Apply ensure_spd for extra numerical safety if requested
    if ensure_valid:
        spd_matrix, _ = ensure_spd(spd_matrix)
    
    # Keep in float64 for numerical stability
    return spd_matrix



# =============================================================================
# Riemannian Geometry Utilities for SPD Matrices
# =============================================================================

def validate_spd(matrix, eps=1e-6, return_details=False):
    """
    Validate that a matrix is Symmetric Positive Definite.
    
    Checks:
    1. Symmetry: ||M - M.T|| < eps
    2. Positive Definiteness: all eigenvalues > eps
    
    Args:
        matrix: (..., N, N) matrix to validate
        eps: Tolerance for numerical checks
        return_details: If True, return detailed diagnostics
    
    Returns:
        is_valid: Boolean (or tensor of booleans for batch)
        details: (optional) dict with min_eigenvalue, max_eigenvalue, symmetry_error
    """
    # Check symmetry
    symmetry_error = torch.norm(matrix - matrix.transpose(-2, -1), dim=(-2, -1))
    is_symmetric = symmetry_error < eps
    
    # Check positive definiteness via eigenvalues (use float64 for precision)
    eigvals = torch.linalg.eigvalsh(matrix.double())
    min_eigval = eigvals.min(dim=-1).values.to(matrix.dtype)
    max_eigval = eigvals.max(dim=-1).values.to(matrix.dtype)
    is_positive = min_eigval > eps
    
    is_valid = is_symmetric & is_positive
    
    if return_details:
        details = {
            'min_eigenvalue': min_eigval,
            'max_eigenvalue': max_eigval,
            'condition_number': max_eigval / (min_eigval + 1e-10),
            'symmetry_error': symmetry_error,
            'is_symmetric': is_symmetric,
            'is_positive': is_positive,
        }
        return is_valid, details
    
    return is_valid


def riemannian_distance(P1, P2, metric='log_euclidean'):
    """
    Compute distance between SPD matrices on the Riemannian manifold.
    
    Args:
        P1: (..., N, N) first SPD matrix
        P2: (..., N, N) second SPD matrix
        metric: 'log_euclidean' or 'affine_invariant'
    
    Returns:
        distance: (...,) geodesic distance
    """
    if metric == 'log_euclidean':
        # Log-Euclidean distance: ||log(P1) - log(P2)||_F
        log_P1 = log_euclidean_map(P1)
        log_P2 = log_euclidean_map(P2)
        distance = torch.norm(log_P1 - log_P2, dim=(-2, -1))
    
    elif metric == 'affine_invariant':
        # Affine-invariant distance: ||log(P1^{-1/2} @ P2 @ P1^{-1/2})||_F
        # More geometrically correct but more expensive
        
        # Compute P1^{-1/2}
        eigvals, eigvecs = torch.linalg.eigh(P1)
        P1_inv_sqrt = eigvecs @ torch.diag_embed(1.0 / torch.sqrt(eigvals + 1e-10)) @ eigvecs.transpose(-2, -1)
        
        # Compute P1^{-1/2} @ P2 @ P1^{-1/2}
        inner = P1_inv_sqrt @ P2 @ P1_inv_sqrt
        
        # Compute log and norm
        log_inner = log_euclidean_map(inner)
        distance = torch.norm(log_inner, dim=(-2, -1))
    
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    return distance


def frechet_mean(matrices, weights=None, max_iter=100, tol=1e-6):
    """
    Compute the Fréchet mean (Riemannian centroid) of SPD matrices.
    
    Uses the iterative algorithm for the Log-Euclidean mean,
    which is simply the matrix exponential of the arithmetic mean of log matrices.
    
    Args:
        matrices: (N, D, D) batch of SPD matrices
        weights: (N,) optional weights for weighted mean
        max_iter: Maximum iterations (for affine-invariant, not used for log-euclidean)
        tol: Convergence tolerance
    
    Returns:
        mean: (D, D) Fréchet mean SPD matrix
    """
    if weights is None:
        weights = torch.ones(matrices.shape[0], device=matrices.device)
    weights = weights / weights.sum()
    
    # Log-Euclidean mean: exp(weighted mean of logs)
    log_matrices = log_euclidean_map(matrices)
    
    # Weighted average in tangent space
    weighted_log_mean = torch.einsum('n,nij->ij', weights, log_matrices)
    
    # Map back to SPD manifold
    mean = exp_euclidean_map(weighted_log_mean)
    
    return mean


def tangent_space_projection(matrices, reference=None):
    """
    Project SPD matrices to tangent space at a reference point.
    
    This is useful for applying Euclidean operations to SPD data.
    
    Args:
        matrices: (..., N, N) SPD matrices
        reference: (N, N) reference point (default: identity matrix)
    
    Returns:
        tangent_vectors: (..., N, N) vectors in tangent space
    """
    if reference is None:
        # Use identity as reference - equivalent to log_euclidean_map
        return log_euclidean_map(matrices)
    
    # General case: log at reference point
    # tangent = reference^{1/2} @ log(reference^{-1/2} @ M @ reference^{-1/2}) @ reference^{1/2}
    eigvals, eigvecs = torch.linalg.eigh(reference)
    ref_sqrt = eigvecs @ torch.diag_embed(torch.sqrt(eigvals)) @ eigvecs.transpose(-2, -1)
    ref_inv_sqrt = eigvecs @ torch.diag_embed(1.0 / torch.sqrt(eigvals + 1e-10)) @ eigvecs.transpose(-2, -1)
    
    inner = ref_inv_sqrt @ matrices @ ref_inv_sqrt
    log_inner = log_euclidean_map(inner)
    tangent = ref_sqrt @ log_inner @ ref_sqrt
    
    return tangent