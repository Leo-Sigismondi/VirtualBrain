"""
Quick diagnostic to check if data has temporal variation
"""
import numpy as np
import matplotlib.pyplot as plt

# Load one subject's data
data = np.load('data/processed/train/A01T_cov.npy')
labels = np.load('data/processed/train/A01T_labels.npy')

print(f"Data shape: {data.shape}")  # (N_trials, N_timesteps, 22, 22)

# Pick first trial
trial_0 = data[0]  # (13, 22, 22)

print(f"\nTrial 0 shape: {trial_0.shape}")
print(f"Label: {labels[0]}")

# Check variance across time
temporal_variance = []
for i in range(trial_0.shape[0]):
    cov_matrix = trial_0[i]
    temporal_variance.append(np.mean(np.abs(cov_matrix)))

print(f"\nMean covariance values across timesteps:")
for t, val in enumerate(temporal_variance):
    print(f"  t={t}: {val:.6f}")

# Calculate change between timesteps
changes = np.diff(temporal_variance)
print(f"\nChange between consecutive timesteps:")
for t, change in enumerate(changes):
    print(f"  t={t}→{t+1}: {change:.8f} ({abs(change/temporal_variance[t])*100:.2f}% change)")

# Visualize one covariance matrix
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for i, t in enumerate([0, 4, 8, 12]):
    axes[i].imshow(trial_0[t], cmap='viridis')
    axes[i].set_title(f"t={t} ({t*0.25:.2f}s)")
    axes[i].axis('off')

plt.suptitle("Covariance Matrices Across Time (Trial 0)")
plt.tight_layout()
plt.savefig('checkpoints/covariance_temporal_check.png', dpi=150)
print(f"\nVisualization saved to: checkpoints/covariance_temporal_check.png")
plt.show()

# Check if different trials have different patterns
print(f"\n{'='*60}")
print("Checking inter-trial vs intra-trial variance...")
print(f"{'='*60}")

# Variance within trial 0 (across time)
within_trial_var = np.var([trial_0[t].flatten() for t in range(13)])
print(f"Variance within trial 0 (temporal): {within_trial_var:.8f}")

# Variance across different trials (spatial)
first_timepoint = data[:, 0, :, :]  # All trials, first timestep
across_trials_var = np.var([first_timepoint[i].flatten() for i in range(min(10, len(data)))])
print(f"Variance across trials (spatial): {across_trials_var:.8f}")

if across_trials_var > within_trial_var * 10:
    print("\n⚠️  DIAGNOSIS: Trials differ more than timesteps!")
    print("   → Covariance matrices are stable features (good for classification)")
    print("   → But they don't have temporal dynamics (bad for GRU prediction)")
    print("\n💡 SOLUTION: Use different features for temporal modeling:")
    print("   - Power Spectral Density over time")
    print("   - Band power features")  
    print("   - Raw filtered EEG (not covariance)")
else:
    print("\n✓ Temporal variation exists")
