import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from attention import MultiHeadAttention


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()

        # Multi-head attention
        self.self_attention = MultiHeadAttention(d_model, n_heads)

        # Feed forward
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model)
        )

        # Normalization layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        # Self attention
        attn_output = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))  # residual connection

        # Feed forward
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))  # residual connection

        return x


class Encoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, num_layers, dropout=0.1):
        super(Encoder, self).__init__()

        # Stack of encoder layers
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(num_layers)]
        )

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        # Initialize three components:
        # 1. Masked multi-head self-attention (for target sequence)
        self.self_attention = MultiHeadAttention(d_model, n_heads)

        # 2. Multi-head cross-attention (to attend to encoder outputs)
        self.cross_attention = MultiHeadAttention(d_model, n_heads)

        # 3. Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model)
        )
        # 4. Initialize normalization layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)  # One more for the cross-attention

        # 5. Initialize dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        # First sub-layer - masked self-attention on target sequence
        # Remember to apply residual connection and layer normalization
        attn_output = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Second sub-layer - cross-attention with encoder outputs
        # Remember to apply residual connection and layer normalization
        cross_attn_output = self.cross_attention(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))  # residual connection

        # Third sub-layer - feed-forward network
        # Remember to apply residual connection and layer normalization
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))  # this is the residual connection

        return x


class Decoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, num_layers, dropout=0.1):
        super(Decoder, self).__init__()

        # Stack of decoder layers
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(num_layers)]
        )

    def forward(self, x, enc_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model,  # embedding dimension
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        d_ff,
        max_seq_length,
        dropout=0.1,
    ):
        super(Transformer, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        # Create embedding layers
        self.src_embedding = nn.Embedding(
            src_vocab_size, d_model
        )  # Question: Are source embeddings and target embeddings required here?
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        # Add positional encoding
        self.positional_encoding = self.create_positional_encoding(
            max_seq_length, d_model
        )
        self.dropout = nn.Dropout(dropout)

        # Create encoder
        self.encoder = Encoder(d_model, num_heads, d_ff, num_encoder_layers, dropout)

        # Create decoder
        self.decoder = Decoder(d_model, num_heads, d_ff, num_decoder_layers, dropout)

        # Create final output layer
        self.output_layer = nn.Linear(d_model, tgt_vocab_size)

    def create_positional_encoding(self, max_seq_length, d_model):
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # Add batch dimension

        # Register as buffer (not a parameter, but should be saved and moved with the model)
        return nn.Parameter(pe, requires_grad=False)

    def forward(self, src, tgt, src_mask, tgt_mask):
        # Get sequence lengths for positional encoding
        src_seq_len = src.size(1)
        tgt_seq_len = tgt.size(1)

        # Embed source and add positional encoding
        src_embedded = self.dropout(
            self.src_embedding(src) * math.sqrt(self.d_model)
            + self.positional_encoding[:, :src_seq_len, :]
        )

        # Pass through encoder
        enc_output = self.encoder(src_embedded, src_mask)

        # Embed target and add positional encoding
        tgt_embedded = self.dropout(
            self.tgt_embedding(tgt) * math.sqrt(self.d_model)
            + self.positional_encoding[:, :tgt_seq_len, :]
        )

        # Pass through decoder (with encoder output)
        dec_output = self.decoder(tgt_embedded, enc_output, src_mask, tgt_mask)

        # Pass through final output layer and return raw logits
        logits = self.output_layer(dec_output)

        return logits
