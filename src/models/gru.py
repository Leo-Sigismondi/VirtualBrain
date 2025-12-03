"""
Temporal GRU Model for Brain State Prediction
Predicts future latent states using residual (delta) prediction
"""
import torch
import torch.nn as nn

class TemporalGRU(nn.Module):
    """
    GRU-based temporal model for predicting brain state dynamics in latent space.
    
    Key Features:
    - Residual (delta) prediction: predicts change from current state
    - Many-to-many training: learns at every timestep
    - Autoregressive generation capability
    
    Architecture:
        Input: Latent sequence [z₁, z₂, ..., zₜ]
        GRU: Captures temporal dynamics
        Head: Predicts Δz (change in latent state)
        Output: z_{t+1} = z_t + Δz_t
    """
    
    def __init__(self, latent_dim=16, hidden_dim=64, num_layers=2, dropout=0.1):
        """
        Args:
            latent_dim: Dimension of VAE latent space
            hidden_dim: Size of GRU hidden state
            num_layers: Number of stacked GRU layers
            dropout: Dropout rate between GRU layers
        """
        super(TemporalGRU, self).__init__()
        
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # GRU Layer - captures temporal dependencies
        self.gru = nn.GRU(
            input_size=latent_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output Head - predicts delta (velocity/change)
        # Using MLP instead of single linear layer for better expressivity
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, latent_dim)
        )

    def forward(self, z_sequence, hidden=None):
        """
        Forward pass for training (many-to-many)
        
        Args:
            z_sequence: (Batch, Seq_Len, Latent) - input latent sequences
            hidden: GRU hidden state (optional)
            
        Returns:
            delta: (Batch, Seq_Len, Latent) - predicted changes in latent state
            hidden: GRU hidden state for next iteration
        """
        # 1. Pass through GRU to get temporal features
        gru_out, hidden = self.gru(z_sequence, hidden)
        
        # 2. Predict Delta (velocity/change) at each timestep
        # This is the KEY for residual prediction: we predict CHANGE not absolute state
        delta = self.head(gru_out)
        
        return delta, hidden
    
    def predict_next(self, z_current, hidden=None):
        """
        Predict next single timestep (for autoregressive generation)
        
        Args:
            z_current: (Batch, 1, Latent) - current latent state
            hidden: GRU hidden state
            
        Returns:
            z_next: (Batch, 1, Latent) - predicted next state
            hidden: Updated hidden state
        """
        # Get delta prediction
        delta, hidden = self.forward(z_current, hidden)
        
        # Apply residual connection: next_state = current_state + delta
        z_next = z_current + delta
        
        return z_next, hidden
    
    def generate_sequence(self, z_start, num_steps, hidden=None):
        """
        Autoregressively generate a sequence of future states
        
        Args:
            z_start: (Batch, 1, Latent) - starting latent state
            num_steps: Number of future steps to predict
            hidden: Initial hidden state (optional)
            
        Returns:
            sequence: (Batch, num_steps, Latent) - predicted trajectory
        """
        predictions = []
        z_current = z_start
        
        for _ in range(num_steps):
            z_next, hidden = self.predict_next(z_current, hidden)
            predictions.append(z_next)
            z_current = z_next  # Feed prediction as next input
        
        # Stack all predictions
        sequence = torch.cat(predictions, dim=1)
        return sequence

    def init_hidden(self, batch_size, device):
        """
        Initialize hidden state with zeros
        
        Args:
            batch_size: Batch size
            device: Device to create tensor on
            
        Returns:
            hidden: (num_layers, batch_size, hidden_dim)
        """
        return torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)


# Example usage and testing
if __name__ == "__main__":
    # Test the model
    batch_size = 32
    seq_len = 4
    latent_dim = 16
    
    # Create model
    model = TemporalGRU(latent_dim=latent_dim, hidden_dim=64, num_layers=2)
    
    # Create dummy input (batch of latent sequences)
    z_sequence = torch.randn(batch_size, seq_len, latent_dim)
    
    # Forward pass
    delta, hidden = model(z_sequence)
    
    print(f"Input shape: {z_sequence.shape}")
    print(f"Delta shape: {delta.shape}")
    print(f"Hidden shape: {hidden.shape}")
    
    # Test autoregressive generation
    z_start = torch.randn(batch_size, 1, latent_dim)
    generated = model.generate_sequence(z_start, num_steps=10)
    print(f"Generated sequence shape: {generated.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
