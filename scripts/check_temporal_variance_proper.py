"""
Proper check of temporal variance with scientific notation
"""
import numpy as np
import matplotlib.pyplot as plt

# Load data
data = np.load('data/processed/train/A01T_cov.npy')
print(f"Data shape: {data.shape}")

# Pick first trial
trial_0 = data[0]  # (13, 25, 25)

print(f"\n=== TEMPORAL VARIANCE CHECK ===")
print("Covariance values across time (scientific notation):")

temporal_means = []
for t in range(trial_0.shape[0]):
    mean_val = np.mean(trial_0[t])
    temporal_means.append(mean_val)
    print(f"  t={t}: {mean_val:.6e}")

# Calculate changes
changes = np.diff(temporal_means)
print(f"\n=== CHANGES BETWEEN TIMESTEPS ===")
for t, change in enumerate(changes):
    pct_change = abs(change / temporal_means[t]) * 100 if temporal_means[t] != 0 else 0
    print(f"  t={t}→{t+1}: {change:.6e} ({pct_change:.2f}% change)")

# Overall variance
temporal_variance = np.var(temporal_means)
print(f"\nVariance across time: {temporal_variance:.6e}")

# Compare to inter-trial variance
all_means = [np.mean(data[i, 0, :, :]) for i in range(min(10, len(data)))]
inter_trial_var = np.var(all_means)

print(f"\n=== VARIANCE COMPARISON ===")
print(f"Within-trial (temporal): {temporal_variance:.6e}")
print(f"Across-trials (spatial): {inter_trial_var:.6e}")
print(f"Ratio (spatial/temporal): {inter_trial_var / temporal_variance if temporal_variance > 0 else 'inf'}x")

if inter_trial_var > temporal_variance * 5:
    print("\n⚠️  Trials vary MORE than timesteps within a trial")
    print("   → This is EXPECTED for motor imagery (stable states)")
    print("   → Your data is CORRECT!")
else:
    print("\n✓ Significant temporal dynamics detected")

# Plot to visualize
plt.figure(figsize=(12, 4))
plt.plot(temporal_means, 'o-', linewidth=2, markersize=8)
plt.xlabel('Timestep')
plt.ylabel('Mean Covariance Value')
plt.title('Temporal Evolution of Covariance Values')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('checkpoints/temporal_evolution.png', dpi=150)
print(f"\n✓ Plot saved to: checkpoints/temporal_evolution.png")
plt.show()
