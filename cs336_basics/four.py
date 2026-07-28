import torch
import math
from collections.abc import Callable, Iterable
from typing import Optional

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr = 1e-3, betas = (0.9, 0.999), eps = 1e-8, weight_decay = 0.01):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        # torch.optim.Optimizer takes a defaults dictionary instead of params
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)
        
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get("t", 0)
                if t == 0:
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)
                t += 1
                state["t"] = t

                g = p.grad.data
                lr_t = lr * math.sqrt(1 - group["betas"][1]**t) / (1 - group["betas"][0]**t)
                p.data -= lr*group["weight_decay"]*p.data
                state["m"] = group["betas"][0] * state["m"] + (1 - group["betas"][0]) * g
                state["v"] = group["betas"][1] * state["v"] + (1 - group["betas"][1]) * torch.mul(g, g)
                p.data -= lr_t * state["m"] / (torch.sqrt(state["v"]) + group["eps"])

        return loss

def cross_entropy_loss(logits, targets):
    # Need to find the largest element for numerical stability
    logits_at_targets = logits.gather(-1, index = targets.unsqueeze(-1)).squeeze(-1)
    loss = torch.logsumexp(logits, dim = -1) - logits_at_targets
    return torch.mean(loss)

def cosine_learning_rate(t, a_max, a_min, T_w, T_c):
    # t is the current iteration
    # a_max is the maximum learning rate
    # a_min is the minimum final learning rate
    # T_w is the number of warmup iterations
    # T_c is the final iteration of cosine annealing
    if t < T_w:
        return a_max * t / T_w
    elif t <= T_c:
        return a_min + (a_max - a_min) * (1 + math.cos(math.pi * (t - T_w) / (T_c - T_w))) / 2
    else:
        return a_min

def gradient_clipping(parameters, max_norm):
    total_norm = torch.sqrt(sum(p.grad.data.norm()**2 for p in parameters if p.grad is not None))
    if total_norm >= max_norm:
        for p in parameters:
            if p.grad is not None:
                p.grad.data.mul_(max_norm / (total_norm + 1e-6))

    return parameters
