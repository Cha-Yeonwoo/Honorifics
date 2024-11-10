import torch
import torch.nn as nn

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim=64, n_heads=4, ff_dim=128, num_layers=2, output_dim=3):
        super(TransformerEncoder, self).__init__()
        
        # Token embedding layer
        self.embedding = nn.Embedding(input_dim, embed_dim)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads, dim_feedforward=ff_dim)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection layer to get desired output dimension (length 3)
        self.output_layer = nn.Linear(embed_dim, output_dim)
    
    def forward(self, x):
        # x: (batch_size, seq_length) -> token indices

        # Embedding: (batch_size, seq_length, embed_dim)
        x = self.embedding(x)

        x = x.mean(dim=2) 

        # Transformer expects (seq_length, batch_size, embed_dim)
        x = x.permute(1, 0, 2)
        
        # Encoder output: (seq_length, batch_size, embed_dim)
        x = self.encoder(x)

        # Project to desired output dimension
        x = self.output_layer(x)  # (seq_length, batch_size, output_dim)
        
        # Permute back to (batch_size, seq_length, output_dim)
        x = x.permute(1, 0, 2)
        
        return x  # (batch_size, seq_length, output_dim)
