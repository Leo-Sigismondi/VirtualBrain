import numpy as np

r = np.load('results/conditional_gru_eval/evaluation_results.npy', allow_pickle=True).item()

print("=" * 60)
print("CONDITIONAL GRU EVALUATION RESULTS")
print("=" * 60)

print("\n=== CLASS CONDITIONING IMPORTANCE ===")
print("(Importance ratio > 1 means class embedding helps)")
for c in range(5):
    imp = r['class_conditioning_importance'][c]
    print(f"Class {c}: mse_correct={imp['mse_correct_class']:.6f}, mse_wrong={imp['mse_wrong_class']:.6f}, ratio={imp['importance_ratio']:.2f}")

avg_ratio = np.mean([r['class_conditioning_importance'][c]['importance_ratio'] for c in range(5)])
print(f"\nAverage importance ratio: {avg_ratio:.2f}")

print(f"\n=== CLASS SEPARABILITY ===")
print(f"Silhouette Score: {r['silhouette_score']:.4f}")
print("(Range: -1 to 1, higher = better class separation)")

print("\n=== STATISTICAL SIMILARITY ===")
for c in range(5):
    s = r['statistics'][c]
    print(f"Class {c}: mean_mse={s['mean_mse']:.6f}, std_mse={s['std_mse']:.6f}, vel_ratio={s['velocity_ratio']:.2f}")
