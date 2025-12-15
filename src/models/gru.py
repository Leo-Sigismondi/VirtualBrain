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


class ConditionalGRU(nn.Module):
    """
    Class-Conditional GRU for Latent Dynamics.
    Allows predicting trajectories specific to a class (e.g., Left Hand vs Right Hand).
    
    Architecture:
    - Embedding: Maps class ID -> Vector
    - Input: Concat[Latent, Embedding] -> GRU
    - Head: GRU_Out -> Delta
    """
    def __init__(self, num_classes=5, class_emb_dim=8, latent_dim=16, hidden_dim=64, num_layers=2, dropout=0.1):
        super(ConditionalGRU, self).__init__()
        
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.class_emb_dim = class_emb_dim
        
        # Class Embedding
        self.embedding = nn.Embedding(num_classes, class_emb_dim)
        
        # GRU Layer
        input_size = latent_dim + class_emb_dim
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output Head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, latent_dim)
        )

    def forward(self, z_sequence, labels, hidden=None):
        """
        Args:
            z_sequence: (Batch, Seq_Len, Latent)
            labels: (Batch, Seq_Len) - Integer class labels per timestep
            hidden: Initial hidden state
        """
        B, T, D = z_sequence.shape
        
        # Embed labels
        # labels: (B, T) -> (B, T, Emb)
        emb = self.embedding(labels)
        
        # Concatenate inputs
        # (B, T, Latent) + (B, T, Emb) -> (B, T, Latent + Emb)
        rnn_input = torch.cat([z_sequence, emb], dim=-1)
        
        # GRU Pass
        gru_out, hidden = self.gru(rnn_input, hidden)
        
        # Predict Delta
        delta = self.head(gru_out)
        
        return delta, hidden

    def predict_next(self, z_current, label, hidden=None):
        """
        Symbolic 1-step prediction for generation.
        label: (Batch, 1) or (Batch,)
        """
        if label.dim() == 1:
            label = label.unsqueeze(1) # (B, 1)
            
        emb = self.embedding(label) # (B, 1, Emb)
        
        rnn_input = torch.cat([z_current, emb], dim=-1)
        
        gru_out, hidden = self.gru(rnn_input, hidden)
        delta = self.head(gru_out)
        
        z_next = z_current + delta
        return z_next, hidden

    def generate_sequence(self, z_start, class_idx, num_steps, hidden=None):
        """
        Generate sequence for a specific class.
        class_idx: Integer or (Batch,) tensor
        """
        B = z_start.shape[0]
        if isinstance(class_idx, int):
            label = torch.full((B, 1), class_idx, device=z_start.device, dtype=torch.long)
        else:
            label = class_idx.view(B, 1)
            
        predictions = []
        z_current = z_start
        
        for _ in range(num_steps):
            z_next, hidden = self.predict_next(z_current, label, hidden)
            predictions.append(z_next)
            z_current = z_next
            
        return torch.cat(predictions, dim=1)
