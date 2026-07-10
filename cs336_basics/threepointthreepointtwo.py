import math
import torch
import torch.nn as nn

class Linear(nn.Module):
    
    # in_features is an int with the final dimension of the input.
    # out_features is an int with the final dimension of the output.
    # device is type torch.device for device to store parameters on.
    # dtype is type torch.dtype that has data type of parameters.
    def __init__(self, in_features, out_features, device = None, dtype = None):
        # Calls the super constructor to initialize the nn.Module class.
        super().__init__()
        # Creates the weight matrix of shape given by the arguments.
        # Makes sure its a parameter so it can be updated during training.
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))     
        std = math.sqrt(2 / (in_features + out_features))
        # Apply our initialization to the weight matrix
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a= -3*std, b=3*std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Basic Matmul using Einops        
        return torch.einsum('...i,oi->...o',x,self.weight)
