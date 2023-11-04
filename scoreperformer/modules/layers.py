""" PyTorch modules used by the models."""

from typing import Optional

import torch
from torch import nn as nn, Tensor

from scoreperformer.utils import exists


# residual

class Residual(nn.Module):
    def __init__(self, dim: int, scale_residual: bool = False, scale_residual_constant: float = 1.):
        super().__init__()
        self.residual_scale = nn.Parameter(torch.ones(dim)) if scale_residual else None
        self.scale_residual_constant = scale_residual_constant

    def forward(self, x, residual):
        if exists(self.residual_scale):
            residual = residual * self.residual_scale

        if self.scale_residual_constant != 1:
            residual = residual * self.scale_residual_constant

        return x + residual


# adaptive layer normalization

class AdaptiveLayerNorm(nn.Module):
    def __init__(self, dim: int, condition_dim: int, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)

        self.linear = nn.Linear(condition_dim, dim * 2)
        self.linear.bias.data[:dim] = 1
        self.linear.bias.data[dim:] = 0

    def forward(self, x: Tensor, condition: Optional[Tensor] = None):
        if condition is None:
            gamma, beta = x.new_ones(1), x.new_zeros(1)
        else:
            condition = condition.unsqueeze(1) if condition.ndim == 2 else condition
            gamma, beta = self.linear(condition).chunk(2, dim=-1)
        return gamma * self.norm(x) + beta
