import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_dim=325, hidden_dim=128, latent_dim=64):
        super(VAE, self).__init__()
        
        # --- ENCODER ---
        # Comprime da 325 -> 128 -> (Mu, LogVar)
        self.enc1 = nn.Linear(input_dim, hidden_dim)
        self.enc2_mu = nn.Linear(hidden_dim, latent_dim)
        self.enc2_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # --- DECODER (Temporaneo) ---
        # Serve solo per allenare l'encoder ora. 
        # Nella fase finale sarà sostituito dal Diffusion Model.
        self.dec1 = nn.Linear(latent_dim, hidden_dim)
        self.dec2 = nn.Linear(hidden_dim, input_dim)
        
        self.dropout = nn.Dropout(0.1)

    def encode(self, x):
        h = F.relu(self.enc1(x))
        h = self.dropout(h)
        mu = self.enc2_mu(h)
        logvar = self.enc2_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.dec1(z))
        h = self.dropout(h)
        return self.dec2(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar