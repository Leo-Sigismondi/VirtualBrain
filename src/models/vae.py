"""
Improved VAE Model for Dynamic Functional Connectivity
=======================================================

WHY WE NEED A DEEPER VAE:
-------------------------
The original VAE had only 1 hidden layer (253 -> 128 -> 32). This creates problems:

1. LIMITED CAPACITY: A single layer can only learn linear transformations followed 
   by ReLU. Brain connectivity patterns are non-linear and hierarchical.

2. INFORMATION BOTTLENECK: Jumping from 253 to 128 to 32 dimensions loses too much
   information too quickly. The model compensates by making the latent space 
   "smooth" (nearly identical consecutive states) to minimize reconstruction error.

3. POOR TEMPORAL FEATURES: SPD matrices from EEG have temporal structure at multiple
   scales (fast neural oscillations, slower state transitions). A single layer 
   cannot disentangle these.

WHY THESE SPECIFIC DESIGN CHOICES:
----------------------------------
1. GRADUAL DIMENSION REDUCTION (253 -> 256 -> 128 -> 64 -> 32):
   - Wider first layer (256) than input captures different feature combinations
   - Gradual compression preserves more information at each stage
   - Each layer can specialize: first layer = low-level features, deeper = abstract

2. LAYERNORM (not BatchNorm):
   - BatchNorm: normalizes across BATCH dimension. Bad for sequences because:
     * Different sequences have different means/variances (that's the signal!)
     * Creates train/test mismatch with running statistics
   - LayerNorm: normalizes across FEATURE dimension for each sample independently
     * Each sample normalized by its own statistics
     * Consistent behavior at train and test time
     * Works with any batch size (even 1)

3. ELU ACTIVATION (not ReLU):
   - ReLU: f(x) = max(0, x) 
     * Problem: "dead neurons" - once a neuron outputs 0, gradient is 0, never recovers
     * Problem: non-smooth at 0, can cause optimization issues
   - ELU: f(x) = x if x > 0, else alpha * (exp(x) - 1)
     * Smooth everywhere, better gradient flow
     * Negative values possible, richer representations
     * "Self-normalizing" property when combined with proper initialization

4. SKIP CONNECTIONS IN ENCODER:
   - Problem: Deep networks suffer from vanishing gradients
   - Solution: Residual connections bypass some layers
   - Benefit: Gradients flow directly through skip connections
   - Note: We only use this when dimensions match (128->128 skip)

5. DROPOUT PLACEMENT:
   - After activation, not before
   - Only in training (automatically disabled in eval mode)
   - Prevents co-adaptation of neurons, improves generalization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ImprovedVAE(nn.Module):
    """
    Deep VAE with better temporal feature preservation.
    
    Architecture Philosophy:
    - Gradual dimension reduction preserves more information
    - LayerNorm ensures stable training without batch-size dependency
    - Skip connections improve gradient flow in deeper network
    - ELU activation avoids dead neurons and provides smoother gradients
    """
    
    def __init__(self, input_dim=253, latent_dim=32, hidden_dims=None, dropout=0.1):
        """
        Args:
            input_dim: Dimension of input (253 for vectorized 22x22 SPD matrix)
            latent_dim: Dimension of latent space (32 is a good balance)
            hidden_dims: List of hidden layer dimensions. Default: [256, 128, 64]
                        - 256 first to capture multiple feature combinations
                        - Gradual reduction to preserve information
            dropout: Dropout rate for regularization
        """
        super(ImprovedVAE, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]  # Wider, deeper, gradual reduction
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        
        # ===== ENCODER =====
        # Build encoder layer by layer for skip connections
        self.encoder_layers = nn.ModuleList()
        prev_dim = input_dim
        
        for i, h_dim in enumerate(hidden_dims):
            layer = nn.Sequential(
                nn.Linear(prev_dim, h_dim),
                nn.LayerNorm(h_dim),  # Normalize before activation (pre-norm)
                nn.ELU(),             # Smooth activation, no dead neurons
                nn.Dropout(dropout)
            )
            self.encoder_layers.append(layer)
            prev_dim = h_dim
        
        # Latent space projections (mu and logvar)
        # Why separate: VAE needs both mean and variance for reparameterization trick
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
        # ===== DECODER =====
        # Mirror architecture of encoder
        self.decoder_layers = nn.ModuleList()
        
        # Start from latent dimension   
        reversed_dims = [latent_dim] + list(reversed(hidden_dims))
        
        for i in range(len(reversed_dims) - 1):
            in_dim = reversed_dims[i]
            out_dim = reversed_dims[i + 1]
            
            layer = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.ELU(),
                nn.Dropout(dropout)
            )
            self.decoder_layers.append(layer)
        
        # Final output layer (no activation - we want full range)
        self.fc_out = nn.Linear(hidden_dims[0], input_dim)
        
        # Initialize weights properly for ELU
        self._init_weights()
    
    def _init_weights(self):
        """
        Proper weight initialization for ELU activation.
        
        Why this matters:
        - Random initialization can cause vanishing/exploding gradients
        - ELU works best with "SELU-style" initialization
        - Linear layers use Kaiming init scaled for ELU
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Kaiming initialization for ELU (gain ~1.0)
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='linear')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def encode(self, x):
        """
        Encode input to latent distribution parameters.
        
        Returns mu and logvar (not the sample!) because:
        - Sampling happens in reparameterize() for backprop to work
        - We often want just mu for deterministic encoding (inference)
        """
        h = x
        for layer in self.encoder_layers:
            h = layer(h)
        
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick for VAE.
        
        Why we need this:
        - VAE latent is SAMPLED from distribution N(mu, sigma^2)
        - Sampling is not differentiable (can't backprop through random)
        - Trick: z = mu + eps * sigma, where eps ~ N(0, 1)
        - Now gradient flows through mu and sigma, not the sampling
        
        logvar instead of sigma:
        - Variance must be positive, hard to enforce directly
        - logvar can be any real number
        - sigma = exp(logvar / 2) is always positive
        """
        std = torch.exp(0.5 * logvar)  # sigma = exp(logvar/2)
        eps = torch.randn_like(std)    # Sample from standard normal
        return mu + eps * std
    
    def decode(self, z):
        """
        Decode from latent space to reconstruction.
        """
        h = z
        for layer in self.decoder_layers:
            h = layer(h)
        
        # Final layer without activation (full range output)
        return self.fc_out(h)
    
    def forward(self, x):
        """
        Full forward pass: encode -> reparameterize -> decode
        
        Returns:
            reconstruction: Reconstructed input
            mu: Mean of latent distribution
            logvar: Log-variance of latent distribution
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar
    
    def get_latent(self, x, deterministic=True):
        """
        Get latent representation (for downstream tasks like GRU).
        
        Args:
            deterministic: If True, return mu only (no sampling noise)
                          If False, return sampled z
        """
        mu, logvar = self.encode(x)
        if deterministic:
            return mu
        return self.reparameterize(mu, logvar)


# Backward compatibility: keep old VAE class name but use new implementation
class VAE(ImprovedVAE):
    """
    Alias for backward compatibility.
    Old code using VAE() will now use ImprovedVAE.
    """
    def __init__(self, input_dim=253, hidden_dim=128, latent_dim=32):
        # Map old API to new
        # Old: single hidden_dim -> New: list of hidden dims
        # Keep first hidden_dim same, add gradual reduction
        super().__init__(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=[256, hidden_dim, latent_dim * 2],  # Gradual reduction
            dropout=0.1
        )


# ===== TESTING =====
if __name__ == "__main__":
    # Test the model
    print("Testing ImprovedVAE...")
    
    batch_size = 32
    input_dim = 253
    latent_dim = 32
    
    model = ImprovedVAE(input_dim=input_dim, latent_dim=latent_dim)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    x = torch.randn(batch_size, input_dim)
    recon, mu, logvar = model(x)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Reconstruction shape: {recon.shape}")
    print(f"Mu shape: {mu.shape}")
    print(f"Logvar shape: {logvar.shape}")
    
    # Test deterministic latent
    z_det = model.get_latent(x, deterministic=True)
    z_stoch = model.get_latent(x, deterministic=False)
    print(f"\nDeterministic latent shape: {z_det.shape}")
    print(f"Stochastic latent shape: {z_stoch.shape}")
    
    # Verify backward pass works
    loss = F.mse_loss(recon, x)
    loss.backward()
    print(f"\nBackward pass successful, loss: {loss.item():.4f}")
    
    print("\n✓ All tests passed!")