"""
Debug preprocessing - check what's actually happening
"""
import os
import numpy as np
import mne
from sklearn.covariance import LedoitWolf

# Paths
gdf_file = "data/raw/gdf/A01T.gdf"
print(f"Loading: {gdf_file}")

# Load
raw = mne.io.read_raw_gdf(gdf_file, preload=True, verbose='ERROR')
print(f"Raw data loaded: {raw.info['nchan']} channels, {raw.info['sfreq']} Hz")
print(f"Channel names: {raw.ch_names[:5]}...")

# Pick EEG
raw.pick_types(eeg=True, eog=False, stim=False, exclude='bads')
print(f"After picking EEG: {raw.info['nchan']} channels")
print(f"Channel names: {raw.ch_names}")

# Filter
raw.filter(4, 38, fir_design='firwin', verbose='ERROR')

# Get events
events, event_id_map = mne.events_from_annotations(raw, verbose='ERROR')
print(f"\nEvents found: {event_id_map}")

# Create epochs
selected_ids = []
for desc, int_id in event_id_map.items():
    if desc in ['769', '770', '771', '772']:
        selected_ids.append(int_id)

epochs = mne.Epochs(raw, events, event_id=selected_ids,
                    tmin=0, tmax=4.0, baseline=None, preload=True, verbose='ERROR')

print(f"\nEpochs created: {len(epochs)} trials")

# Get first trial
trial_data = epochs.get_data(copy=False)[0]  # (Channels, Time)
print(f"Trial data shape: {trial_data.shape}")
print(f"Trial data range: [{trial_data.min():.6f}, {trial_data.max():.6f}]")
print(f"Trial data mean: {trial_data.mean():.6f}")
print(f"Trial data std: {trial_data.std():.6f}")

# Extract first window
win_samples = int(1.0 * raw.info['sfreq'])  # 1 second window
window = trial_data[:, 0:win_samples]

print(f"\n--- Computing Covariance ---")
print(f"Window shape: {window.shape}")  # Should be (22, 250)
print(f"Window range: [{window.min():.6f}, {window.max():.6f}]")

# Compute covariance
estimator = LedoitWolf(store_precision=False, assume_centered=False)  # FIXED!
cov = estimator.fit(window.T).covariance_

print(f"\nCovariance matrix shape: {cov.shape}")
print(f"Covariance range: [{cov.min()}, {cov.max()}]")
print(f"Covariance mean: {cov.mean()}")
print(f"Covariance diagonal (variances): {np.diag(cov)[:5]}")

# Check if SPD
eigvals = np.linalg.eigvalsh(cov)
print(f"\nEigenvalues range: [{eigvals.min()}, {eigvals.max()}]")
print(f"Is positive definite: {np.all(eigvals > 0)}")

# Show a sample
print(f"\nFirst 3x3 of covariance matrix:")
print(cov[:3, :3])
