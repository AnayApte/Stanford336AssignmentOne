import torch

def softmax(tensor: torch.Tensor, i: int) -> torch.Tensor:
    max_val = torch.max(tensor, dim = i, keepdim = True).values
    exp = torch.exp(tensor - max_val)
    return exp / torch.sum(exp, dim = i, keepdim = True)

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    product = torch.einsum('...qd, ...kd -> ...qk', Q, K) / Q.shape[-1] ** 0.5
    if mask is not None:
        product = product.masked_fill(mask == 0, float('-inf'))
    attention = softmax(product, i = -1)
    return torch.einsum('...qk, ...kd -> ...qd', attention, V)
