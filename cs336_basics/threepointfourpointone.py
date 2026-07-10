import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    # d_model is an int that takes the dimension of the model.
    # eps is a float that represents the epsilon value for numerical stability.
    # device specifies the device on which the module's parameters will be allocated.
    # dtype specifies the data type of the module's parameters.
    def __init__(self, d_model: int, eps: float = 1e-5, device = None, dtype = None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(torch.mean(x**2, dim = -1, keepdim = True)+self.eps)
        result = x / rms * self.weight
        return result.to(in_dtype)
