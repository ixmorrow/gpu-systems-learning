import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from attention import MultiHeadAttention


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        # TODO: Initialize three components:
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
