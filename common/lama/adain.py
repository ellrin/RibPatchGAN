"""Adaptive Instance Normalization conditioned on a class label."""
import torch
import torch.nn as nn


class LabelAdaIN(nn.Module):
    """AdaIN(x, label) = gamma(label) * IN(x) + beta(label).
    Initialised near identity so class modulation is learned gradually."""

    def __init__(self, num_features: int, num_classes: int, cond_dim: int = 64):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.embed = nn.Embedding(num_classes, cond_dim)
        self.gamma = nn.Linear(cond_dim, num_features)
        self.beta = nn.Linear(cond_dim, num_features)
        nn.init.zeros_(self.gamma.weight)
        nn.init.ones_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

    def forward(self, x: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        e = self.embed(label)
        g = self.gamma(e)
        b = self.beta(e)
        if x.dim() == 4:
            g = g.view(-1, x.shape[1], 1, 1)
            b = b.view(-1, x.shape[1], 1, 1)
            return self.norm(x) * g + b
        return x * g + b
