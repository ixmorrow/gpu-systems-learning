import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from attention import MultiHeadAttention
from encoder import Encoder
from decoder import Decoder


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
