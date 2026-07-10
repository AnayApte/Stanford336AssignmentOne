import math
import torch
import torch.nn as nn

class Embedding(nn.Module):
    # num_embeddings is an int as the vocab size.
    # embedding_dim is the size of an embedding vector.
    # device is type torch.device for device to store parameters on.
    # dtype is type torch.dtype that has data type of parameters.
    def __init__(self, num_embeddings, embedding_dim, device = None, dtype = None):
        # Again just call the super constructor to initialize the nn.Module class.
        super().__init__()
        # Create an embedding matrix.
        self.embedding = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        # Apply embedding initialization.
        nn.init.trunc_normal_(self.embedding, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # Since vectors are each token, we can just index into the embedding matrix.
        return self.embedding[token_ids]
