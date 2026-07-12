import torch
import torch.nn as nn
from cs336_basics.threepointfourpointone import RMSNorm
from cs336_basics.threepointfourpointfive import CausalSelfAttention
from cs336_basics.threepointfourpointtwo import FFN

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, theta: float, max_seq_len: int):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, num_heads, theta, max_seq_len)
        self.ln1 = RMSNorm(d_model)
        self.ffn = FFN(d_model, d_ff)
        self.ln2 = RMSNorm(d_model)
    
    def forward(self, x: torch.Tensor):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x
