import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model  # embedding dimension (e.g., 512)
        self.num_heads = num_heads  # number of attention heads (e.g., 8)
        self.head_dim = d_model // num_heads  # dimension per head (e.g., 64)

        # Create the Q, K, V projection layers
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        # Final output projection
        self.out_proj = nn.Linear(d_model, d_model)

    # scaled dot product attention
    def attention(self, Q, K, V, mask=None):
        """
        Q, K, V are expected to be of shape:
          [batch_size, seq_len, d_k]
        or possibly
          [batch_size, num_heads, seq_len, d_k]
        if you’re already doing multi-head splitting.
        """
        d_k = K.shape[-1]
        scores = Q @ K.transpose(-2, -1)
        scores = scores / math.sqrt(d_k)
        # Apply mask if provided
        if mask is not None:
            # Expand mask to match the batch size and number of heads
            # mask shape: [batch_size, 1, seq_len, seq_len] or [batch_size, 1, 1, seq_len]
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        output = attention_weights @ V

        return output, attention_weights

    def transopse_akv(self, output, attention_weights, batch_size, seq_len, d_model):
        # re-order dimensions back to original
        output = torch.permute(output, (0, 2, 1, 3))
        # reshape the dimensions to "combine" the attention heads outputs
        output = output.reshape(batch_size, seq_len, d_model)
        # attention_weights has shape [batch_size, num_heads, seq_len, seq_len]
        # Average across the heads dimension (dim=1)
        attention_weights = attention_weights.mean(dim=1)

        return output, attention_weights

    def forward(self, q, k, v, mask=None):
        batch_size, seq_len, d_model = q.shape
        Q = self.q_proj(q)
        K = self.k_proj(k)
        V = self.v_proj(v)

        # Reshape to separate the heads
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim)

        # re-order dimensions to be compatible with attention method
        Q = torch.permute(Q, (0, 2, 1, 3))
        K = torch.permute(K, (0, 2, 1, 3))
        V = torch.permute(V, (0, 2, 1, 3))

        output, attention_weights = self.attention(Q, K, V, mask)
        output, attention_weights = self.transopse_akv(
            output, attention_weights, batch_size, seq_len, d_model
        )

        return self.out_proj(output)
