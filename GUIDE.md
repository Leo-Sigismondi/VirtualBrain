# VirtualBrain - Beginner's Guide

## 📁 Understanding the Project Structure

Think of your project like organizing a kitchen:
- **Ingredients** (data) go in one place
- **Recipes** (code) go in another
- **Finished dishes** (trained models) in another

### The Folders Explained

```
VirtualBrain/
├── src/                    # 🧑‍💻 YOUR CODE (the recipes)
│   ├── preprocessing/      # 📊 Data preparation tools
│   ├── models/            # 🧠 Neural network designs
│   ├── data/              # 📦 Data loading tools
│   └── training/          # 🏋️ Training programs
│
├── data/                   # 💾 YOUR DATA (the ingredients)
│   ├── raw/               # Original EEG files
│   └── processed/         # Cleaned, ready-to-use data
│
├── checkpoints/           # 🎯 TRAINED MODELS (finished dishes)
│   └── vae/               # Saved VAE models
│
├── scripts/               # ⚙️ HELPER SCRIPTS (quick shortcuts)
└── configs/               # ⚙️ SETTINGS (recipe variations)
```

---

## 🔄 The Complete Pipeline (How Data Flows)

### Step 1: Raw Data → Processed Data
**What happens:** Convert raw brain signals into mathematics the computer can learn from

```
Raw EEG Files (.gdf)
        ↓
  [preprocessing]
        ↓
Covariance Matrices (.npy)
```

**How to run:**
```bash
python scripts/run_preprocessing.py
```

**What it does:**
- Reads raw brain signals from `data/raw/BCICIV_2a_gdf.zip`
- Filters the signals (removes noise)
- Calculates covariance matrices (mathematical summaries)
- Saves to `data/processed/train/` and `data/processed/eval/`

---

### Step 2: Processed Data → Trained VAE Model
**What happens:** Train a neural network to compress and understand the data

```
Covariance Matrices
        ↓
     [VAE Model]
        ↓
Compressed Representation (latent space)
```

**How to run:**
```bash
python -m src.training.train_vae
```

**What it does:**
- Loads processed data from `data/processed/train/`
- Trains a VAE (Variational Autoencoder) to compress the data
- Saves the trained model to `checkpoints/vae/vae_encoder.pth`

---

### Step 3 (Future): Add GRU for Time
**What happens:** Learn how brain activity changes over time

```
Latent Space Sequence
        ↓
     [GRU Model]
        ↓
Temporal Predictions
```

---

### Step 4 (Future): Add Diffusion for Generation
**What happens:** Generate new, realistic brain signals

```
Compressed + Temporal Info
        ↓
  [Diffusion Model]
        ↓
Generated Brain Signals
```

---

## 📝 Understanding the Code Files

### `src/preprocessing/preprocess.py`
**What it does:** The "chef" that prepares raw data
- Reads `.gdf` brain signal files
- Applies bandpass filters (keeps important frequencies)
- Splits into time windows
- Calculates covariance matrices

**When to use:** Run this ONCE when you get new data

---

### `src/preprocessing/geometry_utils.py`
**What it does:** Math tools for working with covariance matrices
- `log_euclidean_map()`: Converts matrices to a special mathematical space
- `sym_matrix_to_vec()`: Flattens matrices into vectors

**When to use:** Automatically used by other scripts, you don't call it directly

---

### `src/models/vae.py`
**What it does:** Defines the VAE neural network architecture
- **Encoder:** Compresses 325 numbers → 64 numbers
- **Decoder:** Expands 64 numbers → 325 numbers back
- **Reparameterize:** Adds randomness for better learning

**When to use:** Automatically loaded when training

---

### `src/data/dataset.py`
**What it does:** Loads and prepares data for training
- Reads all `.npy` files from a folder
- Applies geometric preprocessing (Log-Euclidean mapping)
- Creates batches for training

**When to use:** Automatically used during training

---

### `src/training/train_vae.py`
**What it does:** The main training script for VAE
- Loads data using `dataset.py`
- Creates VAE model from `models/vae.py`
- Trains for 50 epochs (50 complete passes through data)
- Saves the trained model

**When to use:** Run this to train your VAE

---

## 🎯 How Python Imports Work

### The Old Way (Before Reorganization)
```python
from utils_geometry import log_euclidean_map  # ❌ Confusing, where is this file?
```

### The New Way (After Reorganization)
```python
from src.preprocessing.geometry_utils import log_euclidean_map  # ✅ Clear path!
```

**Why this is better:**
- You can see exactly where each function comes from
- No confusion about which file contains what
- Works from anywhere in the project

---

## 🚀 How to Use Your Project

### Option 1: Step-by-Step (Recommended for Learning)

**Step 1:** Place your raw data
```bash
# Put BCICIV_2a_gdf.zip in:
data/raw/BCICIV_2a_gdf.zip
```

**Step 2:** Preprocess the data
```bash
python scripts/run_preprocessing.py
```
**Output:** `data/processed/train/` will contain 9 `.npy` files

**Step 3:** Train the VAE
```bash
python -m src.training.train_vae
```
**Output:** `checkpoints/vae/vae_encoder.pth` will be created

---

### Option 2: Using as a Python Package

Because everything is in `src/`, you can import from anywhere:

```python
# In a Jupyter notebook or another script:
from src.models.vae import VAE
from src.data.dataset import BCIDataset

# Load data
dataset = BCIDataset("data/processed/train")

# Create model
model = VAE(input_dim=325, latent_dim=64)

# Do your own experiments!
```

---

## 🔍 What Each Command Does

### `python scripts/run_preprocessing.py`
- **Starts from:** Project root (`VirtualBrain/`)
- **Runs:** `src/preprocessing/preprocess.py`
- **Reads from:** `data/raw/`
- **Writes to:** `data/processed/`
- **Time:** ~30 seconds

---

### `python -m src.training.train_vae`
- **Starts from:** Project root (`VirtualBrain/`)
- **Runs:** `src/training/train_vae.py`
- **Reads from:** `data/processed/train/`
- **Writes to:** `checkpoints/vae/`
- **Time:** ~5-10 minutes (depends on GPU)

**Note:** The `-m` flag tells Python to run it as a module, which makes imports work correctly

---

## 💡 Tips for Adding New Features

### Adding a New Model (e.g., GRU)

**1. Create the model file:**
```bash
# Create: src/models/gru.py
```

**2. Define your model:**
```python
import torch.nn as nn

class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim)
        # ... rest of your model
```

**3. Create a training script:**
```bash
# Create: src/training/train_gru.py
```

**4. Import and use:**
```python
from src.models.gru import GRUModel
from src.models.vae import VAE  # Use the VAE encoder

# Your training code here
```

---

## 🆘 Troubleshooting

### Import Error: "No module named 'src'"
**Problem:** You're not in the right folder
**Solution:** 
```bash
cd E:/Programming/VirtualBrain  # Go to project root
python -m src.training.train_vae  # Run from here
```

---

### Import Error: "No module named 'utils_geometry'"
**Problem:** Your file hasn't been updated to the new structure
**Solution:** Update the import:
```python
# Change from:
from utils_geometry import something

# To:
from src.preprocessing.geometry_utils import something
```

---

### File Not Found: "data/processed/train"
**Problem:** You haven't preprocessed the data yet
**Solution:**
```bash
python scripts/run_preprocessing.py
```

---

## 📚 Next Steps for Learning

1. **Understand one file at a time**
   - Start with `src/data/dataset.py` (simplest)
   - Then `src/models/vae.py` (see the neural network)
   - Finally `src/training/train_vae.py` (see how training works)

2. **Experiment with parameters**
   - Change `LATENT_DIM` in `train_vae.py` (try 32, 64, 128)
   - Change `EPOCHS` (try 10, 50, 100)
   - See how it affects the results

3. **Add visualization**
   - Create `notebooks/` folder
   - Make a Jupyter notebook to visualize the latent space

4. **Read the saved models**
   ```python
   import torch
   model = VAE(input_dim=325, latent_dim=64)
   model.load_state_dict(torch.load("checkpoints/vae/vae_encoder.pth"))
   # Now you can use the trained model!
   ```

---

## 🎓 Key Concepts

### What is a "Module"?
A module is just a Python file with reusable code. `src/models/vae.py` is a module.

### What is a "Package"?
A package is a folder with an `__init__.py` file. `src/` is a package containing sub-packages.

### Why the dots in imports?
```python
from src.preprocessing.geometry_utils import log_euclidean_map
#     ^           ^               ^
#  package   sub-package        module
```

Think of it like a folder path, but with dots instead of slashes!

---

## ✨ Summary

**Your project is now organized like a professional Python package:**
- ✅ Clear separation of data, code, and outputs
- ✅ Easy to navigate and understand
- ✅ Ready to add more models (GRU, Diffusion)
- ✅ Can be used as a library in other projects

**The two main commands you need:**
```bash
# 1. Prepare data (run once)
python scripts/run_preprocessing.py

# 2. Train model (run whenever you want to train)
python -m src.training.train_vae
```

That's it! You're ready to build your brain decoding pipeline! 🧠✨
