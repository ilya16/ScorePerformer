"""
Transformer FeedForward layer.

Adapted from: https://github.com/lucidrains/x-transformers
"""
from dataclasses import dataclass

import torch.nn as nn

from ..constructor import Constructor, ModuleConfig


class GLU(nn.Module):
    def __init__(self, dim_in, dim_out, activation):
        super().__init__()
        self.act = activation
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * self.act(gate)


@dataclass
class FeedForwardConfig(ModuleConfig):
    dim: int = 512
    mult: int = 4
    glu: bool = False
    swish: bool = False
    post_act_ln: bool = False
    dropout: float = 0.
    no_bias: bool = True


class FeedForward(nn.Module, Constructor):
    def __init__(
            self,
            dim: int = 512,
            mult: int = 4,
            glu: bool = False,
            swish: bool = False,
            post_act_ln: bool = False,
            dropout: float = 0.,
            no_bias: bool = True
    ):
        super().__init__()

        inner_dim = int(dim * mult)
        activation = nn.SiLU() if swish else nn.GELU()

        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim, bias=not no_bias),
            activation
        ) if not glu else GLU(dim, inner_dim, activation)

        self.ff = nn.Sequential(
            project_in,
            nn.LayerNorm(inner_dim) if post_act_ln else nn.Identity(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim, bias=not no_bias)
        )

    def forward(self, x):
        return self.ff(x)
