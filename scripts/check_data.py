import numpy as np

print("=== CHECKING train_normalized.npy ===")

# Try loading as memmap first
try:
    data = np.memmap('data/processed/train_normalized.npy', dtype='float32', mode='r', shape=(96272, 64, 253))
    print(f"Memmap load SUCCESS")
    print(f"  Shape: {data.shape}")
    print(f"  Dtype: {data.dtype}")
    print(f"  Sample [0,0,:5]: {data[0,0,:5]}")
    print(f"  Min: {data[:100].min():.4f}, Max: {data[:100].max():.4f}, Mean: {data[:100].mean():.4f}")
except Exception as e:
    print(f"Memmap load FAILED: {e}")

# Try regular load with pickle
try:
    data2 = np.load('data/processed/train_normalized.npy', allow_pickle=True)
    print(f"\nPickle load result:")
    print(f"  Type: {type(data2)}")
    if hasattr(data2, 'shape'):
        print(f"  Shape: {data2.shape}")
        print(f"  Dtype: {data2.dtype}")
except Exception as e:
    print(f"Pickle load FAILED: {e}")

# Check file header
print("\n=== FILE HEADER (first 50 bytes) ===")
with open('data/processed/train_normalized.npy', 'rb') as f:
    header = f.read(50)
    print(f"Bytes: {header[:20]}")
    print(f"Magic: {header[:6]}")  # Should be \x93NUMPY
