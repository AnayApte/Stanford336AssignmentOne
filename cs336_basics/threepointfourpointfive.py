import torch
import torch.nn as nn
from cs336_basics.threepointthreepointtwo import Linear
from cs336_basics.threepointfourpointthree import Rope
from cs336_basics.threepointfourpointfour import scaled_dot_product_attention

# torch.triu to construct causal masking.
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, theta: float = None, max_seq_len: int = None):
        super().__init__()
        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.o_proj = Linear(d_model, d_model)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dk = d_model // num_heads
        self.rope = Rope(theta, self.dk, max_seq_len) if theta and max_seq_len else None

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor = None) -> torch.Tensor:
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        Q = Q.reshape(*Q.shape[:-1], self.num_heads, self.dk).transpose(-2,-3)
        K = K.reshape(*K.shape[:-1], self.num_heads, self.dk).transpose(-2,-3)
        V = V.reshape(*V.shape[:-1], self.num_heads, self.dk).transpose(-2,-3)
        if self.rope:
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)
        seq_len = x.shape[-2]
        if token_positions is None:
            token_positions = torch.arange(seq_len, device=x.device)
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))
        scaled_attention = scaled_dot_product_attention(Q, K, V, mask)
        scaled_attention = scaled_attention.transpose(-2,-3).reshape(*x.shape[:-1], self.d_model)
        return self.o_proj(scaled_attention)
