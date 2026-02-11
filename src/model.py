import torch
import torch.nn as nn
import torch.nn.functional as F

class EcoSieveVAE(nn.Module):
    def __init__(self, vocab_size, embedding_dim=256, hidden_dim=512, latent_dim=128):
        """
        Arquitectura VAE Flexible.
        Permite ajustar el tamaño de las capas (hidden_dim) y el espacio latente (latent_dim).
        """
        super(EcoSieveVAE, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # --- 1. Encoder ---
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # GRU ahora usa hidden_dim variable
        self.encoder_rnn = nn.GRU(embedding_dim, self.hidden_dim, batch_first=True)
        
        # --- 2. Espacio Latente ---
        # Comprimimos la salida de la RNN al vector latente
        self.fc_mu = nn.Linear(self.hidden_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(self.hidden_dim, self.latent_dim)
        
        # --- 3. Decoder ---
        # Expandimos del latente al tamaño de la RNN
        self.decoder_input = nn.Linear(self.latent_dim, self.hidden_dim)
        
        self.decoder_rnn = nn.GRU(embedding_dim, self.hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(self.hidden_dim, vocab_size)
        
    def reparameterize(self, mu, logvar):
        """Truco de reparametrizacion para poder entrenar con backpropagation"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # x shape: [batch, seq_len]
        
        # A. Codificación
        embedded = self.embedding(x)
        # Solo necesitamos el ultimo estado oculto (h_n) del encoder
        _, h_n = self.encoder_rnn(embedded) 
        h_n = h_n.squeeze(0) # [batch, hidden_dim]
        
        # B. Espacio Latente
        mu = self.fc_mu(h_n)
        logvar = self.fc_logvar(h_n)
        z = self.reparameterize(mu, logvar)
        
        # C. Decodificación
        # Preparamos el estado inicial del decoder basado en z
        d_hidden = self.decoder_input(z).unsqueeze(0) # [1, batch, hidden_dim]
        
        # Pasamos toda la secuencia (Teacher Forcing)
        output, _ = self.decoder_rnn(embedded, d_hidden)
        
        # Proyectamos a vocabulario
        recon_batch = self.fc_out(output) # [batch, seq_len, vocab_size]
        
        return recon_batch, mu, logvar