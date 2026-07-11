import torch
import torch.nn as nn

class Rope(nn.Module):
    # theta is the float angle for RoPE
    # d_k is the int dimension for query and key vectors
    # max_seq_len is the int maximum sequence length
    # device is a torch device to store buffer on
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.d_k = d_k
        # Gets an array of the positions from 0 to max_seq_len.
        positions = torch.arange(max_seq_len, device = device)
        # Gets an array of every other dimension. Needed cause we rotate pairs.
        dims = torch.arange(0, d_k, 2, device = device)
        # Gets an array of the frequencies.
        frequencies = 1 / theta**(dims / d_k)
        # Gets the angles.
        self.angles = torch.outer(positions, frequencies)
        # These values are always the exact same - don't need to be modified so no need to store to state.
        self.register_buffer('cos', torch.cos(self.angles), persistent=False)
        self.register_buffer('sin', torch.sin(self.angles), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # Takes the last dimension and splits into pairs
        x_pairs = x.reshape(*x.shape[:-1], self.d_k//2, 2)
        x1 = x_pairs[..., 0]
        x2 = x_pairs[..., 1]
        # Apply RoPE
        x1_new = x1 * self.cos[token_positions] - x2 * self.sin[token_positions]
        x2_new = x1 * self.sin[token_positions] + x2 * self.cos[token_positions]
        x_pairs = torch.stack((x1_new, x2_new), dim = -1)
        return x_pairs.reshape(x.shape)
