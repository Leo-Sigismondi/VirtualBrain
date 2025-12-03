"""
Direct inspection of saved covariance files
"""
import numpy as np

# Load processed data
data = np.load('data/processed/train/A01T_cov.npy')
print(f"Loaded data shape: {data.shape}")
print(f"Data type: {data.dtype}")
print(f"Data min/max: [{data.min()}, {data.max()}]")
print(f"Data mean: {data.mean()}")
print(f"Data std: {data.std()}")

# Check a single covariance matrix
single_cov = data[0, 0, :, :]  # First trial, first timestep
print(f"\nFirst covariance matrix shape: {single_cov.shape}")
print(f"First covariance min/max: [{single_cov.min()}, {single_cov.max()}]")
print(f"First covariance mean: {single_cov.mean()}")

# Show the actual values
print(f"\nFirst 5x5 of covariance matrix:")
print(single_cov[:5, :5])

# Check if it's symmetric
is_symmetric = np.allclose(single_cov, single_cov.T)
print(f"\nIs symmetric: {is_symmetric}")

# Check eigenvalues
eigvals = np.linalg.eigvalsh(single_cov)
print(f"Eigenvalues min/max: [{eigvals.min()}, {eigvals.max()}]")
print(f"Is positive definite: {np.all(eigvals > 0)}")
