# VirtualBrain - BCI Brain Decoding Pipeline

Deep learning pipeline for decoding brain signals from EEG covariance matrices using Variational Autoencoders, GRU temporal models, and Diffusion decoders.

## Project Structure

```
VirtualBrain/
├── src/                    # Source code
│   ├── preprocessing/      # Data preprocessing
│   ├── models/            # Neural network architectures
│   ├── data/              # Dataset loaders
│   └── training/          # Training scripts
│
├── data/                   # Data directory (gitignored)
│   ├── raw/               # Raw GDF files
│   └── processed/         # Processed covariance matrices
│
├── checkpoints/           # Trained models (gitignored)
│   ├── vae/
│   ├── gru/
│   └── diffusion/
│
├── scripts/               # Utility scripts
├── configs/               # Configuration files
└── notebooks/             # Jupyter notebooks
```

## Pipeline

1. **Preprocessing** → Extract covariance matrices from BCI Competition IV 2a dataset
2. **VAE Encoder** → Compress covariance matrices to latent space
3. **GRU Temporal** → Model temporal dynamics (future)
4. **Diffusion Decoder** → Generate reconstructions (future)

## Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\\Scripts\\activate

# Install dependencies
pip install torch numpy mne scikit-learn tqdm

# Add data
# Place BCICIV_2a_gdf.zip in data/raw/
```

## Usage

### 1. Preprocess Data
```bash
python scripts/run_preprocessing.py
```

### 2. Train VAE
```bash
python -m src.training.train_vae
```

## Dataset

BCI Competition IV Dataset 2a - Motor Imagery (4 classes)
- 9 subjects
- Training and evaluation sets
- 22 EEG channels
- 250 Hz sampling rate

## Requirements

- Python 3.8+
- PyTorch
- MNE-Python
- NumPy
- scikit-learn
- tqdm