"""
Tangent Space Diffusion Model for SPD Matrix Trajectory Generation.

This module implements a DDPM (Denoising Diffusion Probabilistic Model) that operates
DIRECTLY in the Log-Euclidean tangent space of SPD matrices.

Pipeline:
    SPD → log() → Tangent Space → Diffusion (denoise) → exp() → SPD

Key Design Choices:
    - Diffusion in tangent space (253-dim for 23×23 matrices)
    - Works directly with Log-Euclidean mapped covariance matrices
    - exp() at generation time guarantees SPD output
    - No VAE needed - preserves full geometric structure
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


# =============================================================================
# Noise Schedule
# =============================================================================

def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    """
    Linear noise schedule as in original DDPM paper.
    
    Args:
        timesteps: Number of diffusion steps
        beta_start: Starting noise level
        beta_end: Ending noise level
    
    Returns:
        betas: (T,) noise schedule
    """
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    Cosine schedule as proposed in "Improved DDPM".
    Produces smoother noise schedule that works better in practice.
    
    Args:
        timesteps: Number of diffusion steps
        s: Small offset to prevent singularity at t=0
    
    Returns:
        betas: (T,) noise schedule
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    # Clip betas to reasonable range - max 0.02 prevents instability at high t
    return torch.clip(betas, 0.0001, 0.02)


# =============================================================================
# Time Embedding
# =============================================================================

class SinusoidalPositionEmbeddings(nn.Module):
    """
    Sinusoidal embeddings for diffusion timestep, similar to Transformer.
    Allows the model to know "how noisy" the current input is.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


# =============================================================================
# Denoiser Network
# =============================================================================

class ConditionalMLP(nn.Module):
    """
    MLP-based denoiser with FiLM conditioning.
    
    FiLM (Feature-wise Linear Modulation) conditioning is more powerful than
    simple additive conditioning. The condition produces scale (gamma) and 
    shift (beta) parameters: h = h * gamma + beta
    
    Conditioned on:
    - Timestep (via sinusoidal embedding)
    - VAE latent / GRU hidden state (via FiLM)
    """
    def __init__(
        self, 
        input_dim,          # Tangent space dimension (253)
        hidden_dim=256,     # Hidden layer size
        time_dim=64,        # Time embedding dimension
        condition_dim=128,  # VAE latent / GRU hidden dimension
        num_layers=4,
        dropout=0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.GELU(),
            nn.Linear(time_dim * 2, time_dim),
        )
        
        # FiLM condition projection: produces gamma (scale) and beta (shift)
        # for each layer's hidden dimension
        self.film_proj = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
        )
        # This outputs (hidden_dim * 2): first half is gamma, second half is beta
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Main network blocks
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            self.blocks.append(
                ResidualBlock(hidden_dim, time_dim, dropout)
            )
        
        # Output projection (predicts noise)
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        )
    
    def forward(self, x, t, condition=None):
        """
        Predict noise added to x at timestep t with FiLM conditioning.
        
        Args:
            x: (B, T, D) noisy tangent vectors
            t: (B,) diffusion timestep
            condition: (B, T, H) per-frame condition, OR (B, H) sequence-level condition
        
        Returns:
            noise_pred: (B, T, D) predicted noise
        """
        batch_size, seq_len, latent_dim = x.shape
        
        # Time embedding: (B, time_dim)
        t_emb = self.time_mlp(t)
        
        # Flatten sequence for processing: (B*T, D)
        x_flat = x.view(-1, latent_dim)
        
        # Input projection: (B*T, hidden_dim)
        h = self.input_proj(x_flat)
        
        # FiLM conditioning
        gamma = None
        beta = None
        if condition is not None:
            # Check for per-frame conditioning (B, T, H)
            if condition.dim() == 3 and condition.shape[1] == seq_len:
                # Per-frame condition: flatten directly to (B*T, H)
                cond_flat = condition.reshape(-1, condition.shape[-1])
            elif condition.dim() == 3:
                # Multi-layer hidden states (B, num_layers, H) -> take last layer
                cond_flat = condition[:, -1, :]
                cond_flat = cond_flat.unsqueeze(1).expand(-1, seq_len, -1)
                cond_flat = cond_flat.reshape(-1, condition.shape[-1])
            else:
                # Sequence-level condition (B, H) - expand to all frames
                cond_flat = condition.unsqueeze(1).expand(-1, seq_len, -1)
                cond_flat = cond_flat.reshape(-1, condition.shape[-1])
            
            # Project to FiLM parameters: (B*T, hidden_dim * 2)
            film_params = self.film_proj(cond_flat)
            
            # Split into gamma (scale) and beta (shift)
            gamma = film_params[:, :self.hidden_dim]  # (B*T, hidden_dim)
            beta = film_params[:, self.hidden_dim:]   # (B*T, hidden_dim)
            
            # Apply FiLM: h = h * (1 + gamma) + beta
            # Using (1 + gamma) for stability (identity initialization)
            h = h * (1.0 + gamma) + beta
        
        # Expand time embedding to match sequence: (B*T, time_dim)
        t_emb = t_emb.unsqueeze(1).expand(-1, seq_len, -1)
        t_emb = t_emb.reshape(-1, t_emb.shape[-1])
        
        # Apply residual blocks
        for block in self.blocks:
            h = block(h, t_emb)
        
        # Output projection: (B*T, D)
        out = self.output_proj(h)
        
        # Reshape back: (B, T, D)
        out = out.view(batch_size, seq_len, latent_dim)
        
        return out


class ResidualBlock(nn.Module):
    """
    Residual block with time conditioning.
    """
    def __init__(self, hidden_dim, time_dim, dropout=0.1):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        
        self.time_proj = nn.Linear(time_dim, hidden_dim)
        
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, t_emb):
        # First layer
        h = self.norm1(x)
        h = F.gelu(h)
        h = self.linear1(h)
        
        # Add time embedding
        h = h + self.time_proj(F.gelu(t_emb))
        
        # Second layer
        h = self.norm2(h)
        h = F.gelu(h)
        h = self.dropout(h)
        h = self.linear2(h)
        
        return x + h


# =============================================================================
# Diffusion Model
# =============================================================================

class TangentDiffusion(nn.Module):
    """
    DDPM-style diffusion model operating DIRECTLY in tangent space.
    
    Works with Log-Euclidean mapped SPD matrices. The exp() operation
    at generation time guarantees the output is a valid SPD matrix.
    
    Args:
        tangent_dim: Dimension of vectorized tangent space (253 for 23×23 matrices)
        hidden_dim: Hidden dimension of denoiser
        condition_dim: Dimension of conditioning (e.g., class label embedding)
        n_steps: Number of diffusion steps (1000 for training, can use fewer for sampling)
        schedule: 'linear' or 'cosine' beta schedule
    """
    def __init__(
        self,
        tangent_dim=253,  # 23*24/2 for upper triangle, or 23*22/2 + 23 for lower + diag
        hidden_dim=512,   # Larger for higher-dim input
        condition_dim=128,
        n_steps=1000,
        schedule='cosine'
    ):
        super().__init__()
        
        self.tangent_dim = tangent_dim
        self.n_steps = n_steps
        
        # Denoiser network - larger for tangent space
        self.denoiser = ConditionalMLP(
            input_dim=tangent_dim,
            hidden_dim=hidden_dim,
            condition_dim=condition_dim,
            num_layers=6,  # More layers for higher-dim input
        )
        
        # Noise schedule
        if schedule == 'linear':
            betas = linear_beta_schedule(n_steps)
        else:
            betas = cosine_beta_schedule(n_steps)
        
        # Precompute diffusion parameters
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Register as buffers (not trainable, but saved with model)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas', torch.sqrt(1.0 / alphas))
        
        # For posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
    
    def forward_diffusion(self, x_0, t, noise=None):
        """
        Forward diffusion process: q(x_t | x_0).
        
        Adds noise to clean data according to the noise schedule.
        
        Args:
            x_0: (B, T, D) clean latent trajectories
            t: (B,) timesteps
            noise: Optional pre-generated noise
        
        Returns:
            x_t: (B, T, D) noisy trajectories
            noise: (B, T, D) the noise that was added
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        
        # Get schedule parameters for timestep t
        sqrt_alpha_cumprod = self._extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alpha_cumprod = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
        
        # x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        x_t = sqrt_alpha_cumprod * x_0 + sqrt_one_minus_alpha_cumprod * noise
        
        return x_t, noise
    
    def _extract(self, a, t, x_shape):
        """
        Extract values from a at indices t, and reshape for broadcasting.
        """
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
    
    def training_loss(self, x_0, condition=None):
        """
        Compute training loss (simplified DDPM loss).
        
        We predict the noise and minimize MSE between predicted and actual noise.
        
        Args:
            x_0: (B, T, D) clean latent trajectories
            condition: (B, H) GRU hidden state for conditioning
        
        Returns:
            loss: Scalar loss value
        """
        batch_size = x_0.shape[0]
        device = x_0.device
        
        # Sample random timesteps
        t = torch.randint(0, self.n_steps, (batch_size,), device=device, dtype=torch.long)
        
        # Forward diffusion
        x_t, noise = self.forward_diffusion(x_0, t)
        
        # Predict noise
        noise_pred = self.denoiser(x_t, t, condition)
        
        # MSE loss
        loss = F.mse_loss(noise_pred, noise)
        
        return loss
    
    @torch.no_grad()
    def sample(self, shape, condition=None, device='cuda'):
        """
        Generate samples via reverse diffusion process.
        
        Args:
            shape: (B, T, D) shape of samples to generate
            condition: (B, H) GRU hidden state for conditioning
            device: Device to generate on
        
        Returns:
            samples: (B, T, D) generated latent trajectories
        """
        batch_size = shape[0]
        
        # Start from pure noise
        x = torch.randn(shape, device=device)
        
        # Reverse diffusion
        for t in reversed(range(self.n_steps)):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            x = self._reverse_step(x, t_batch, condition)
        
        return x
    
    def _reverse_step(self, x_t, t, condition=None):
        """
        Single reverse diffusion step: p(x_{t-1} | x_t).
        """
        # Predict noise
        noise_pred = self.denoiser(x_t, t, condition)
        
        # Get schedule parameters
        beta_t = self._extract(self.betas, t, x_t.shape)
        sqrt_one_minus_alpha_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        sqrt_recip_alpha_t = self._extract(self.sqrt_recip_alphas, t, x_t.shape)
        
        # Predict x_0 from x_t and predicted noise
        # x_0 = (x_t - sqrt(1-alpha_bar) * noise) / sqrt(alpha_bar)
        # Then compute mean of p(x_{t-1} | x_t, x_0)
        model_mean = sqrt_recip_alpha_t * (x_t - beta_t * noise_pred / sqrt_one_minus_alpha_cumprod_t)
        
        if t[0] == 0:
            return model_mean
        else:
            # Add noise for stochastic sampling
            posterior_variance_t = self._extract(self.posterior_variance, t, x_t.shape)
            noise = torch.randn_like(x_t)
            return model_mean + torch.sqrt(posterior_variance_t) * noise
    
    @torch.no_grad()
    def sample_ddim(self, shape, condition=None, device='cuda', steps=50, eta=0.0):
        """
        DDIM sampling for faster generation.
        
        Uses a subset of timesteps for faster sampling while maintaining quality.
        
        Args:
            shape: (B, T, D) shape of samples to generate
            condition: (B, H) GRU hidden state for conditioning
            device: Device to generate on
            steps: Number of sampling steps (fewer = faster)
            eta: Stochasticity parameter (0 = deterministic, 1 = DDPM)
        
        Returns:
            samples: (B, T, D) generated latent trajectories
        """
        batch_size = shape[0]
        
        # Create timestep schedule (evenly spaced)
        timesteps = torch.linspace(self.n_steps - 1, 0, steps, dtype=torch.long, device=device)
        
        # Start from pure noise
        x = torch.randn(shape, device=device)
        
        for i, t in enumerate(timesteps):
            t_batch = torch.full((batch_size,), t.item(), device=device, dtype=torch.long)
            
            # Get next timestep (or 0 if at the end)
            t_next = timesteps[i + 1] if i + 1 < len(timesteps) else torch.tensor(0, device=device)
            
            x = self._ddim_step(x, t_batch, t_next, condition, eta)
        
        return x
    
    def _ddim_step(self, x_t, t, t_next, condition=None, eta=0.0):
        """
        Single DDIM reverse step.
        """
        # Predict noise
        noise_pred = self.denoiser(x_t, t, condition)
        
        # Get alpha values
        alpha_t = self._extract(self.alphas_cumprod, t, x_t.shape)
        alpha_next = self.alphas_cumprod[t_next] if t_next > 0 else torch.tensor(1.0, device=x_t.device)
        alpha_next = alpha_next.view(1, 1, 1).expand_as(x_t[:1, :1, :1]).expand_as(x_t)
        
        # Predict x_0
        x_0_pred = (x_t - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
        
        # Compute sigma for stochasticity
        sigma = eta * torch.sqrt((1 - alpha_next) / (1 - alpha_t)) * torch.sqrt(1 - alpha_t / alpha_next)
        
        # Direction pointing to x_t
        dir_xt = torch.sqrt(1 - alpha_next - sigma ** 2) * noise_pred
        
        # Sample noise
        noise = torch.randn_like(x_t) if t[0] > 0 else 0
        
        # Compute x_{t-1}
        x_next = torch.sqrt(alpha_next) * x_0_pred + dir_xt + sigma * noise
        
        return x_next


# =============================================================================
# Utility Functions
# =============================================================================

def count_parameters(model):
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick test
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model for tangent space (253-dim for 23x23 matrices)
    tangent_dim = 253  # Upper triangle of 23x23 symmetric matrix
    model = TangentDiffusion(
        tangent_dim=tangent_dim,
        hidden_dim=512,
        condition_dim=128,
        n_steps=1000
    ).to(device)
    
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Test forward pass
    batch_size = 4
    seq_len = 64
    condition_dim = 128
    
    x_0 = torch.randn(batch_size, seq_len, tangent_dim, device=device)
    condition = torch.randn(batch_size, condition_dim, device=device)
    
    # Training loss
    loss = model.training_loss(x_0, condition)
    print(f"Training loss: {loss.item():.4f}")
    
    # Sampling (DDIM for speed)
    samples = model.sample_ddim((2, seq_len, tangent_dim), condition[:2], device, steps=50)
    print(f"Generated samples shape: {samples.shape}")
    
    print("✓ Tangent space diffusion model test passed!")
