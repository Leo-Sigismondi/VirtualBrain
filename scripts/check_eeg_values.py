"""
Check raw EEG signal values
"""
import mne
import numpy as np

# Load raw data
raw = mne.io.read_raw_gdf("data/raw/gdf/A01T.gdf", preload=True, verbose='ERROR')

# Pick EEG
raw.pick('eeg', exclude='bads')
raw.filter(4, 38, fir_design='firwin', verbose='ERROR')

# Get the data
data = raw.get_data()  # (channels, samples)

print(f"Raw EEG data shape: {data.shape}")
print(f"Data type: {data.dtype}")
print(f"\n=== RAW SIGNAL VALUES ===")
print(f"Min: {data.min()}")
print(f"Max: {data.max()}")
print(f"Mean: {data.mean()}")
print(f"Std: {data.std()}")
print(f"Range: {data.max() - data.min()}")

# The issue: EEG in GDF is in VOLTS, which are tiny values!
# MNE should convert to microvolts automatically, but let's check
print(f"\n=== EXPECTED VALUES ===")
print("EEG should be in microvolts (µV): typically -100 to +100 µV")
print("In volts: -0.0001 to +0.0001 V")
print(f"\nYour data appears to be in: {'Volts (10^-6)' if abs(data.mean()) < 1e-3 else 'microVolts'}")

# Manual conversion if needed
data_uv = data * 1e6  # Convert to microvolts
print(f"\n=== AFTER CONVERTING TO µV ===")
print(f"Min: {data_uv.min():.2f} µV")
print(f"Max: {data_uv.max():.2f} µV")
print(f"Mean: {data_uv.mean():.2f} µV")
print(f"Std: {data_uv.std():.2f} µV")

# Now compute covariance
from sklearn.covariance import LedoitWolf

window = data[:, :250]  # First 1 second
cov_volts = LedoitWolf().fit(window.T).covariance_

window_uv = data_uv[:, :250]
cov_uv = LedoitWolf().fit(window_uv.T).covariance_

print(f"\n=== COVARIANCE VALUES ===")
print(f"From Volts:      {cov_volts.mean():.6e} (diagonal: {np.diag(cov_volts).mean():.6e})")
print(f"From microVolts: {cov_uv.mean():.6e} (diagonal: {np.diag(cov_uv).mean():.6e})")
print(f"\nRatio: {cov_uv.mean() / cov_volts.mean():.0f}x (should be 10^12 since covariance scales as value^2)")
