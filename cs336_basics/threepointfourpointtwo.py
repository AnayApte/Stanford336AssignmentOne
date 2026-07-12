import torch
import torch.nn as nn
from cs336_basics.threepointthreepointtwo import Linear

class FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int = None):
        super().__init__()
        self.d_ff = d_ff if d_ff is not None else int(8/3 * d_model / 64) * 64
        self.w1 = Linear(d_model, self.d_ff)
        self.w2 = Linear(self.d_ff, d_model)
        self.w3 = Linear(d_model, self.d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first_transform = self.w1(x)
        activated = first_transform * torch.sigmoid(first_transform)
        third_transform = self.w3(x)
        gated = activated * third_transform
        output = self.w2(gated)
        return output
