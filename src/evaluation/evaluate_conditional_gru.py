"""
Evaluation Script for Conditional GRU
======================================

Evaluates how well the conditional GRU generates class-specific trajectories.

Metrics:
1. Statistical Similarity - Compare distributions of real vs generated
2. Class Separability - Can we distinguish between generated classes?
3. Temporal Dynamics - Do generated trajectories have realistic dynamics?
4. Visual Comparison - Side-by-side plots of real vs generated

Usage:
    python src/evaluation/evaluate_conditional_gru.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import os

from src.models.conditional_gru import ConditionalTemporalGRU
from src.models.vae import ImprovedVAE
from src.data.data_utils import (
    load_normalized_dataset_with_labels, encode_to_latent,
    INPUT_DIM, NUM_CLASSES
)


# Config
LATENT_DIM = 32
HIDDEN_DIM = 256
NUM_LAYERS = 3
CLASS_EMBED_DIM = 16
SEED_STEPS = 8

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "checkpoints/gru/conditional_gru_L32_H256_3L_best.pth"
VAE_PATH = "checkpoints/vae/vae_dynamics_latent32_best.pth"
LATENT_STATS_PATH = "checkpoints/gru/latent_norm_stats_conditional.npy"

SAVE_DIR = "results/conditional_gru_eval"

CLASS_NAMES = {0: "Rest", 1: "Left Hand", 2: "Right Hand", 3: "Both Feet", 4: "Tongue"}


def load_models():
    """Load trained VAE and Conditional GRU."""
    # Load VAE
    vae = ImprovedVAE(
        input_dim=INPUT_DIM,
        latent_dim=LATENT_DIM,
        hidden_dims=[256, 128, 64]
    ).to(DEVICE)
    vae.load_state_dict(torch.load(VAE_PATH, map_location=DEVICE))
    vae.eval()
    
    # Load Conditional GRU
    model = ConditionalTemporalGRU(
        latent_dim=LATENT_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        num_classes=NUM_CLASSES,
        class_embed_dim=CLASS_EMBED_DIM
    ).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    # Load latent normalization stats
    latent_stats = np.load(LATENT_STATS_PATH, allow_pickle=True).item()
    
    return vae, model, latent_stats


def get_real_samples_by_class(latent_sequences, labels, n_samples=100):
    """Get real latent sequences grouped by class."""
    samples_by_class = {}
    for c in range(NUM_CLASSES):
        class_mask = labels == c
        class_indices = np.where(class_mask)[0]
        if len(class_indices) > n_samples:
            selected = np.random.choice(class_indices, n_samples, replace=False)
        else:
            selected = class_indices
        samples_by_class[c] = latent_sequences[selected]
    return samples_by_class


def generate_samples_by_class(model, real_samples, latent_stats, n_samples=100):
    """Generate trajectories for each class using real starting points."""
    generated_by_class = {}
    
    model.eval()
    with torch.no_grad():
        for c in range(NUM_CLASSES):
            # Use real samples as seed
            real = real_samples[c][:n_samples]
            
            # Normalize
            real_norm = (real - latent_stats['mean']) / (latent_stats['std'] + 1e-8)
            
            # Take seed portion
            seed = torch.from_numpy(real_norm[:, :SEED_STEPS, :]).float().to(DEVICE)
            class_labels = torch.full((len(seed),), c, dtype=torch.long).to(DEVICE)
            
            # Generate using the model
            generated = model.generate_full_trajectory(
                seed, class_labels, seed_steps=SEED_STEPS, total_steps=64
            )
            
            # Denormalize
            generated_denorm = generated.cpu().numpy() * latent_stats['std'] + latent_stats['mean']
            generated_by_class[c] = generated_denorm
    
    return generated_by_class


def compute_statistics(real_samples, generated_samples):
    """Compute statistical comparison metrics."""
    results = {}
    
    for c in range(NUM_CLASSES):
        real = real_samples[c]
        gen = generated_samples[c]
        
        # Flatten to (N, features) for comparison
        real_flat = real.reshape(-1, real.shape[-1])
        gen_flat = gen.reshape(-1, gen.shape[-1])
        
        # Mean comparison
        real_mean = np.mean(real_flat, axis=0)
        gen_mean = np.mean(gen_flat, axis=0)
        mean_mse = np.mean((real_mean - gen_mean) ** 2)
        
        # Std comparison
        real_std = np.std(real_flat, axis=0)
        gen_std = np.std(gen_flat, axis=0)
        std_mse = np.mean((real_std - gen_std) ** 2)
        
        # Temporal dynamics: compute velocities
        real_vel = np.diff(real, axis=1)
        gen_vel = np.diff(gen, axis=1)
        
        real_vel_std = np.std(real_vel)
        gen_vel_std = np.std(gen_vel)
        vel_ratio = gen_vel_std / (real_vel_std + 1e-8)
        
        results[c] = {
            'mean_mse': mean_mse,
            'std_mse': std_mse,
            'velocity_ratio': vel_ratio,
            'real_vel_std': real_vel_std,
            'gen_vel_std': gen_vel_std
        }
    
    return results


def compute_class_separability(generated_samples, n_components=2):
    """Check if generated samples of different classes are distinguishable."""
    # Collect all samples with labels
    all_samples = []
    all_labels = []
    
    for c in range(NUM_CLASSES):
        samples = generated_samples[c]
        # Use mean of each trajectory as representation
        trajectory_means = samples.mean(axis=1)  # (N, latent_dim)
        all_samples.append(trajectory_means)
        all_labels.extend([c] * len(trajectory_means))
    
    all_samples = np.concatenate(all_samples, axis=0)
    all_labels = np.array(all_labels)
    
    # Compute silhouette score (higher = better class separation)
    silhouette = silhouette_score(all_samples, all_labels)
    
    # t-SNE for visualization
    if len(all_samples) > 50:
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        embedded = tsne.fit_transform(all_samples)
    else:
        embedded = all_samples[:, :2]
    
    return silhouette, embedded, all_labels


def compute_class_conditioning_importance(model, real_samples, latent_stats, n_samples=50):
    """
    Measure how much the class embedding contributes to generation quality.
    
    Approach:
    - Generate with CORRECT class label
    - Generate with WRONG class label (use a different class)
    - Compare MSE to real data for both
    
    If class conditioning matters:
        MSE(correct_class) << MSE(wrong_class)
    
    Returns:
        Dict with MSE for correct vs wrong class per class
    """
    results = {}
    
    model.eval()
    with torch.no_grad():
        for true_class in range(NUM_CLASSES):
            real = real_samples[true_class][:n_samples]
            
            # Normalize
            real_norm = (real - latent_stats['mean']) / (latent_stats['std'] + 1e-8)
            seed = torch.from_numpy(real_norm[:, :SEED_STEPS, :]).float().to(DEVICE)
            
            # === Generate with CORRECT class ===
            correct_labels = torch.full((len(seed),), true_class, dtype=torch.long).to(DEVICE)
            gen_correct = model.generate_full_trajectory(
                seed, correct_labels, seed_steps=SEED_STEPS, total_steps=64
            )
            gen_correct_np = gen_correct.cpu().numpy() * latent_stats['std'] + latent_stats['mean']
            
            # === Generate with WRONG class ===
            wrong_class = (true_class + 1) % NUM_CLASSES  # Pick next class as wrong
            wrong_labels = torch.full((len(seed),), wrong_class, dtype=torch.long).to(DEVICE)
            gen_wrong = model.generate_full_trajectory(
                seed, wrong_labels, seed_steps=SEED_STEPS, total_steps=64
            )
            gen_wrong_np = gen_wrong.cpu().numpy() * latent_stats['std'] + latent_stats['mean']
            
            # Compute MSE to real (only on generated portion, after seed)
            real_post_seed = real[:, SEED_STEPS:, :]
            gen_correct_post = gen_correct_np[:, SEED_STEPS:, :]
            gen_wrong_post = gen_wrong_np[:, SEED_STEPS:, :]
            
            mse_correct = np.mean((gen_correct_post - real_post_seed) ** 2)
            mse_wrong = np.mean((gen_wrong_post - real_post_seed) ** 2)
            
            # Importance ratio: how much worse is wrong class?
            importance = mse_wrong / (mse_correct + 1e-8)
            
            results[true_class] = {
                'mse_correct_class': mse_correct,
                'mse_wrong_class': mse_wrong,
                'importance_ratio': importance,  # >1 means class matters
                'wrong_class_used': wrong_class
            }
    
    return results


def plot_class_conditioning_importance(importance_results, save_path):
    """Plot bar chart comparing correct vs wrong class MSE."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    classes = list(range(NUM_CLASSES))
    class_labels = [CLASS_NAMES[c] for c in classes]
    
    mse_correct = [importance_results[c]['mse_correct_class'] for c in classes]
    mse_wrong = [importance_results[c]['mse_wrong_class'] for c in classes]
    importance = [importance_results[c]['importance_ratio'] for c in classes]
    
    # Plot MSE comparison
    x = np.arange(len(classes))
    width = 0.35
    axes[0].bar(x - width/2, mse_correct, width, label='Correct Class', color='forestgreen')
    axes[0].bar(x + width/2, mse_wrong, width, label='Wrong Class', color='crimson')
    axes[0].set_xlabel('True Class')
    axes[0].set_ylabel('MSE to Real')
    axes[0].set_title('Generation MSE: Correct vs Wrong Class Label')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(class_labels, rotation=45, ha='right')
    axes[0].legend()
    
    # Plot importance ratio
    colors = ['forestgreen' if r > 1.0 else 'crimson' for r in importance]
    axes[1].bar(class_labels, importance, color=colors)
    axes[1].axhline(y=1.0, color='gray', linestyle='--', label='No difference')
    axes[1].set_xlabel('True Class')
    axes[1].set_ylabel('Importance Ratio (Wrong/Correct)')
    axes[1].set_title('Class Conditioning Importance\n(>1 = class embedding helps)')
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved class conditioning importance to {save_path}")


def plot_trajectory_comparison(real_samples, generated_samples, save_path):
    """Plot side-by-side comparison of real vs generated trajectories."""
    fig, axes = plt.subplots(NUM_CLASSES, 3, figsize=(15, 4*NUM_CLASSES))
    
    for c in range(NUM_CLASSES):
        real = real_samples[c]
        gen = generated_samples[c]
        
        # Select a random sample
        idx = np.random.randint(0, min(len(real), len(gen)))
        
        # Plot first 3 latent dimensions
        for dim in range(3):
            ax = axes[c, dim]
            ax.plot(real[idx, :, dim], label='Real', alpha=0.8, linewidth=2)
            ax.plot(gen[idx, :, dim], label='Generated', alpha=0.8, linewidth=2, linestyle='--')
            ax.axvline(x=SEED_STEPS, color='gray', linestyle=':', label='Seed end' if dim==0 else None)
            ax.set_title(f'{CLASS_NAMES[c]} - Dim {dim}')
            ax.set_xlabel('Timestep')
            ax.set_ylabel('Latent value')
            if c == 0 and dim == 0:
                ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved trajectory comparison to {save_path}")


def plot_class_embedding(embedded, labels, save_path):
    """Plot t-SNE of generated samples colored by class."""
    plt.figure(figsize=(10, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, NUM_CLASSES))
    
    for c in range(NUM_CLASSES):
        mask = labels == c
        plt.scatter(embedded[mask, 0], embedded[mask, 1], 
                   c=[colors[c]], label=CLASS_NAMES[c], alpha=0.6, s=50)
    
    plt.legend()
    plt.title('t-SNE of Generated Trajectories by Class')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved class embedding to {save_path}")


def plot_statistics_summary(stats_results, save_path):
    """Plot bar charts of statistical metrics."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    classes = list(range(NUM_CLASSES))
    class_labels = [CLASS_NAMES[c] for c in classes]
    
    # Mean MSE
    mean_mses = [stats_results[c]['mean_mse'] for c in classes]
    axes[0].bar(class_labels, mean_mses, color='steelblue')
    axes[0].set_title('Mean MSE (Real vs Generated)')
    axes[0].set_ylabel('MSE')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Std MSE
    std_mses = [stats_results[c]['std_mse'] for c in classes]
    axes[1].bar(class_labels, std_mses, color='darkorange')
    axes[1].set_title('Std MSE (Real vs Generated)')
    axes[1].set_ylabel('MSE')
    axes[1].tick_params(axis='x', rotation=45)
    
    # Velocity ratio
    vel_ratios = [stats_results[c]['velocity_ratio'] for c in classes]
    axes[2].bar(class_labels, vel_ratios, color='forestgreen')
    axes[2].axhline(y=1.0, color='red', linestyle='--', label='Ideal (1.0)')
    axes[2].set_title('Velocity Ratio (Gen/Real)')
    axes[2].set_ylabel('Ratio')
    axes[2].tick_params(axis='x', rotation=45)
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved statistics summary to {save_path}")


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    print("="*60)
    print("Evaluating Conditional GRU")
    print("="*60)
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model not found at {MODEL_PATH}")
        print("Please train the model first with train_conditional_gru.py")
        return
    
    # Load models
    print("\n1. Loading models...")
    vae, model, latent_stats = load_models()
    print(f"   Latent stats: mean={latent_stats['mean']:.4f}, std={latent_stats['std']:.4f}")
    
    # Load data
    print("\n2. Loading data...")
    normalized_data, labels, _ = load_normalized_dataset_with_labels()
    
    # Encode to latent
    print("\n3. Encoding to latent space...")
    latent_sequences = encode_to_latent(vae, normalized_data[:5000], DEVICE)  # Subset for speed
    latent_sequences = latent_sequences.numpy()
    labels_subset = labels[:5000]
    
    # Get real samples by class
    print("\n4. Sampling real trajectories by class...")
    real_samples = get_real_samples_by_class(latent_sequences, labels_subset, n_samples=100)
    for c in range(NUM_CLASSES):
        print(f"   {CLASS_NAMES[c]}: {len(real_samples[c])} samples")
    
    # Generate samples
    print("\n5. Generating trajectories for each class...")
    generated_samples = generate_samples_by_class(model, real_samples, latent_stats, n_samples=100)
    
    # Compute statistics
    print("\n6. Computing statistics...")
    stats_results = compute_statistics(real_samples, generated_samples)
    
    print("\n   Results:")
    print(f"   {'Class':<12} {'Mean MSE':<12} {'Std MSE':<12} {'Vel Ratio':<12}")
    print("   " + "-"*48)
    for c in range(NUM_CLASSES):
        r = stats_results[c]
        print(f"   {CLASS_NAMES[c]:<12} {r['mean_mse']:<12.6f} {r['std_mse']:<12.6f} {r['velocity_ratio']:<12.3f}")
    
    # Class separability
    print("\n7. Computing class separability...")
    silhouette, embedded, embed_labels = compute_class_separability(generated_samples)
    print(f"   Silhouette score: {silhouette:.4f}")
    print(f"   (Higher = better class separation, range: -1 to 1)")
    
    # Class conditioning importance
    print("\n8. Computing class conditioning importance...")
    importance_results = compute_class_conditioning_importance(model, real_samples, latent_stats)
    
    print("\n   Results:")
    print(f"   {'Class':<12} {'MSE Correct':<14} {'MSE Wrong':<14} {'Importance':<12}")
    print("   " + "-"*52)
    for c in range(NUM_CLASSES):
        r = importance_results[c]
        print(f"   {CLASS_NAMES[c]:<12} {r['mse_correct_class']:<14.6f} {r['mse_wrong_class']:<14.6f} {r['importance_ratio']:<12.2f}")
    
    avg_importance = np.mean([importance_results[c]['importance_ratio'] for c in range(NUM_CLASSES)])
    print(f"\n   Average importance ratio: {avg_importance:.2f}")
    if avg_importance > 1.0:
        print("   ✓ Class embedding IS contributing to generation quality!")
    else:
        print("   ✗ Class embedding may NOT be learning class-specific patterns")
    
    # Generate plots
    print("\n9. Generating plots...")
    plot_trajectory_comparison(real_samples, generated_samples, 
                               f"{SAVE_DIR}/trajectory_comparison.png")
    plot_class_embedding(embedded, embed_labels, 
                         f"{SAVE_DIR}/class_tsne.png")
    plot_statistics_summary(stats_results, 
                            f"{SAVE_DIR}/statistics_summary.png")
    plot_class_conditioning_importance(importance_results,
                                       f"{SAVE_DIR}/class_conditioning_importance.png")
    
    # Save numerical results
    print("\n10. Saving results...")
    results = {
        'statistics': stats_results,
        'silhouette_score': silhouette,
        'class_conditioning_importance': importance_results,
        'class_names': CLASS_NAMES
    }
    np.save(f"{SAVE_DIR}/evaluation_results.npy", results)
    
    print("\n" + "="*60)
    print("Evaluation Complete!")
    print(f"Results saved to: {SAVE_DIR}/")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
