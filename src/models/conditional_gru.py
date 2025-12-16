"""
Conditional Temporal GRU Model for Class-Specific Brain State Generation
=========================================================================

Extends TemporalGRU with class conditioning for data augmentation.
Given a class label, generates trajectories typical of that class.

Key Features:
- Learned class embeddings concatenated with input at each timestep
- Same residual (delta) prediction architecture as base GRU
- Compatible with existing VAE latent space
"""
import torch
import torch.nn as nn


class ConditionalTemporalGRU(nn.Module):
    """
    GRU-based temporal model with class conditioning for generating
    class-specific brain state trajectories.
    
    Architecture:
        Input: Latent sequence [z₁, z₂, ..., zₜ] + Class label c
        Embedding: class c → learned embedding e_c
        Conditioned Input: [z_t; e_c] at each timestep
        GRU: Captures class-conditional temporal dynamics
        Head: Predicts Δz (change in latent state)
        Output: z_{t+1} = z_t + Δz_t
    
    For Data Augmentation:
        1. Provide a starting state z_0 (e.g., random or from real data)
        2. Provide target class label c
        3. Call generate_sequence() to produce class-typical trajectory
    """
    
    def __init__(
        self, 
        latent_dim: int = 32, 
        hidden_dim: int = 256, 
        num_layers: int = 3,
        num_classes: int = 5,
        class_embed_dim: int = 16,
        dropout: float = 0.2
    ):
        """
        Args:
            latent_dim: Dimension of VAE latent space
            hidden_dim: Size of GRU hidden state
            num_layers: Number of stacked GRU layers
            num_classes: Number of class labels (5 for BCI: 0=rest, 1-4=activities)
            class_embed_dim: Dimension of learned class embeddings
            dropout: Dropout rate between GRU layers
        """
        super(ConditionalTemporalGRU, self).__init__()
        
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.class_embed_dim = class_embed_dim
        
        # Class embedding layer
        # Learns a dense representation for each class
        self.class_embedding = nn.Embedding(num_classes, class_embed_dim)
        
        # GRU Layer - takes latent + class embedding as input
        self.gru = nn.GRU(
            input_size=latent_dim + class_embed_dim,  # Concatenated input
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output Head - predicts delta (velocity/change)
        # Same architecture as base GRU, outputs latent_dim
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, latent_dim)
        )

    def forward(self, z_sequence: torch.Tensor, class_labels: torch.Tensor, hidden=None):
        """
        Forward pass for training (many-to-many)
        
        Args:
            z_sequence: (Batch, Seq_Len, Latent) - input latent sequences
            class_labels: (Batch,) - class label for each sequence (integer 0-4)
            hidden: GRU hidden state (optional)
            
        Returns:
            delta: (Batch, Seq_Len, Latent) - predicted changes in latent state
            hidden: GRU hidden state for next iteration
        """
        batch_size, seq_len, _ = z_sequence.shape
        
        # 1. Get class embeddings: (Batch,) -> (Batch, embed_dim)
        class_embed = self.class_embedding(class_labels)
        
        # 2. Expand to match sequence length: (Batch, embed_dim) -> (Batch, Seq_Len, embed_dim)
        class_embed_expanded = class_embed.unsqueeze(1).expand(-1, seq_len, -1)
        
        # 3. Concatenate with latent input: (Batch, Seq_Len, latent_dim + embed_dim)
        conditioned_input = torch.cat([z_sequence, class_embed_expanded], dim=-1)
        
        # 4. Pass through GRU
        gru_out, hidden = self.gru(conditioned_input, hidden)
        
        # 5. Predict Delta at each timestep
        delta = self.head(gru_out)
        
        return delta, hidden
    
    def predict_next(self, z_current: torch.Tensor, class_labels: torch.Tensor, hidden=None):
        """
        Predict next single timestep (for autoregressive generation)
        
        Args:
            z_current: (Batch, 1, Latent) - current latent state
            class_labels: (Batch,) - class labels
            hidden: GRU hidden state
            
        Returns:
            z_next: (Batch, 1, Latent) - predicted next state
            hidden: Updated hidden state
        """
        # Get delta prediction
        delta, hidden = self.forward(z_current, class_labels, hidden)
        
        # Apply residual connection: next_state = current_state + delta
        z_next = z_current + delta
        
        return z_next, hidden
    
    def generate_sequence(
        self, 
        z_start: torch.Tensor, 
        class_labels: torch.Tensor,
        num_steps: int, 
        hidden=None
    ):
        """
        Autoregressively generate a sequence of future states for a given class.
        
        This is the main method for DATA AUGMENTATION:
        - Provide a starting point and target class
        - Model generates a class-typical trajectory
        
        Args:
            z_start: (Batch, 1, Latent) - starting latent state
            class_labels: (Batch,) - target class for generation
            num_steps: Number of future steps to predict
            hidden: Initial hidden state (optional)
            
        Returns:
            sequence: (Batch, num_steps, Latent) - predicted trajectory
        """
        predictions = []
        z_current = z_start
        
        for _ in range(num_steps):
            z_next, hidden = self.predict_next(z_current, class_labels, hidden)
            predictions.append(z_next)
            z_current = z_next  # Feed prediction as next input
        
        # Stack all predictions
        sequence = torch.cat(predictions, dim=1)
        return sequence
    
    def generate_full_trajectory(
        self,
        z_seed: torch.Tensor,
        class_labels: torch.Tensor,
        seed_steps: int = 8,
        total_steps: int = 64
    ):
        """
        Generate a full trajectory with seed warmup, matching training procedure.
        
        Args:
            z_seed: (Batch, seed_steps, Latent) - seed sequence for hidden state warmup
            class_labels: (Batch,) - target class
            seed_steps: Number of steps used to warm up hidden state
            total_steps: Total sequence length to generate
            
        Returns:
            full_sequence: (Batch, total_steps, Latent) - complete trajectory
        """
        # Phase 1: Build hidden state from seed
        hidden = None
        for t in range(seed_steps):
            current = z_seed[:, t:t+1, :]
            _, hidden = self.predict_next(current, class_labels, hidden)
        
        # Phase 2: Generate remaining steps autoregressively
        generate_steps = total_steps - seed_steps
        z_current = z_seed[:, -1:, :]  # Start from last seed step
        
        generated = self.generate_sequence(z_current, class_labels, generate_steps, hidden)
        
        # Combine seed + generated
        full_sequence = torch.cat([z_seed, generated], dim=1)
        
        return full_sequence

    def init_hidden(self, batch_size: int, device: torch.device):
        """
        Initialize hidden state with zeros
        
        Args:
            batch_size: Batch size
            device: Device to create tensor on
            
        Returns:
            hidden: (num_layers, batch_size, hidden_dim)
        """
        return torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)

    def generate_from_class(
        self,
        class_labels: torch.Tensor,
        num_steps: int = 64,
        z_start: torch.Tensor = None,
        start_mode: str = 'zero'
    ):
        """
        Generate trajectories from JUST a class label - main method for DATA AUGMENTATION.
        
        No seed sequence required - starts from zero or random and generates the full trajectory.
        
        Args:
            class_labels: (Batch,) - target class for generation
            num_steps: Total number of steps to generate
            z_start: Optional (Batch, 1, Latent) starting point. If None, uses start_mode.
            start_mode: How to initialize starting point if z_start is None
                       - 'zero': Start from zeros (default)
                       - 'random': Start from random normal
                       - 'random_small': Start from small random values (0.1 * randn)
            
        Returns:
            trajectory: (Batch, num_steps, Latent) - complete generated trajectory
        
        Example:
            # Generate 10 left-hand trajectories
            class_labels = torch.full((10,), fill_value=1, dtype=torch.long)
            trajectories = model.generate_from_class(class_labels, num_steps=64)
        """
        batch_size = class_labels.shape[0]
        device = class_labels.device
        
        # Initialize starting point
        if z_start is None:
            if start_mode == 'zero':
                z_start = torch.zeros(batch_size, 1, self.latent_dim).to(device)
            elif start_mode == 'random':
                z_start = torch.randn(batch_size, 1, self.latent_dim).to(device)
            elif start_mode == 'random_small':
                z_start = 0.1 * torch.randn(batch_size, 1, self.latent_dim).to(device)
            else:
                raise ValueError(f"Unknown start_mode: {start_mode}")
        
        # Generate autoregressively from scratch (no hidden state warmup)
        hidden = None
        predictions = [z_start]
        z_current = z_start
        
        for _ in range(num_steps - 1):  # -1 because z_start is the first step
            z_next, hidden = self.predict_next(z_current, class_labels, hidden)
            predictions.append(z_next)
            z_current = z_next
        
        trajectory = torch.cat(predictions, dim=1)
        return trajectory


# Example usage and testing
if __name__ == "__main__":
    # Test the conditional model
    batch_size = 32
    seq_len = 64
    latent_dim = 32
    num_classes = 5
    
    # Create model
    model = ConditionalTemporalGRU(
        latent_dim=latent_dim, 
        hidden_dim=256, 
        num_layers=3,
        num_classes=num_classes,
        class_embed_dim=16
    )
    
    # Create dummy input
    z_sequence = torch.randn(batch_size, seq_len, latent_dim)
    class_labels = torch.randint(0, num_classes, (batch_size,))
    
    # Forward pass
    delta, hidden = model(z_sequence, class_labels)
    
    print(f"Input shape: {z_sequence.shape}")
    print(f"Class labels shape: {class_labels.shape}")
    print(f"Delta shape: {delta.shape}")
    print(f"Hidden shape: {hidden.shape}")
    
    # Test generation for a specific class
    z_start = torch.randn(batch_size, 1, latent_dim)
    target_class = torch.full((batch_size,), fill_value=2, dtype=torch.long)  # Generate class 2
    generated = model.generate_sequence(z_start, target_class, num_steps=63)
    print(f"\nGenerated sequence shape (class 2): {generated.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    
    # Compare with base GRU
    from gru import TemporalGRU
    base_model = TemporalGRU(latent_dim=latent_dim, hidden_dim=256, num_layers=3)
    base_params = sum(p.numel() for p in base_model.parameters())
    print(f"Base GRU parameters: {base_params:,}")
    print(f"Additional params from conditioning: {total_params - base_params:,}")
