import torch
import numpy as np
import math

def data_loading(x: np.ndarray, batch_size, context_length, device) -> tuple[torch.Tensor, torch.Tensor]:
    start_indices = torch.randint(0, len(x) - context_length, (batch_size,))
    inputs = torch.zeros((batch_size, context_length), dtype=torch.long)
    targets = torch.zeros((batch_size, context_length), dtype=torch.long)
    for batch_idx, start in enumerate(start_indices):
        inputs[batch_idx] = torch.from_numpy(x[start : start + context_length])
        targets[batch_idx] = torch.from_numpy(x[start + 1 : start + 1 + context_length])

    return inputs.to(device), targets.to(device)

def save_checkpoint(model, optimizer, iteration, out):
    model.state

def load_checkpoint(src, model, optimizer):
    pass
