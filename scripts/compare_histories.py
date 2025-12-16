import numpy as np

print("=== CONDITIONAL GRU (if exists) ===")
try:
    h = np.load('checkpoints/gru/training_history_conditional.npy', allow_pickle=True).item()
    print(f"Train losses (first 10): {[f'{x:.6f}' for x in h['train_loss'][:10]]}")
    print(f"Val losses (first 10): {[f'{x:.6f}' for x in h['val_loss'][:10]]}")
    print(f"Min val loss: {min(h['val_loss']):.8f}")
except Exception as e:
    print(f"Error: {e}")

print("\n=== BASE GRU MULTISTEP ===")
try:
    h2 = np.load('checkpoints/gru/training_history_multistep.npy', allow_pickle=True).item()
    print(f"Train losses (first 10): {[f'{x:.6f}' for x in h2['train_loss'][:10]]}")
    print(f"Val losses (first 10): {[f'{x:.6f}' for x in h2['val_loss'][:10]]}")
    print(f"Min val loss: {min(h2['val_loss']):.8f}")
except Exception as e:
    print(f"Error: {e}")
