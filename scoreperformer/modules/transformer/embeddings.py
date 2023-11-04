""" General purpose transformer embeddings. """
import math
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class DiscreteContinuousEmbedding(nn.Module):
    def __init__(
            self,
            num_embeddings: int,
            embedding_dim: int,
            discrete: bool = True,
            continuous: bool = True,
            discrete_ids: Optional[Union[list, Tensor]] = None,
            token_values: Optional[Union[list, Tensor]] = None,
            padding_idx: Optional[int] = None,
            activation=None,
            _weight: Optional[Tensor] = None,
            device=None,
            dtype=None
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        if discrete_ids is not None:
            if not isinstance(discrete_ids, Tensor):
                discrete_ids = torch.Tensor(discrete_ids)
            discrete_ids = discrete_ids.reshape(-1).to(device=device, dtype=torch.long)
        self.discrete_ids = discrete_ids

        if padding_idx is not None:
            if padding_idx > 0:
                assert padding_idx < self.num_embeddings, 'Padding_idx must be within num_embeddings'
            elif padding_idx < 0:
                assert padding_idx >= -self.num_embeddings, 'Padding_idx must be within num_embeddings'
                padding_idx = self.num_embeddings + padding_idx
        self.padding_idx = padding_idx

        assert discrete or continuous, '`DiscreteContinuousEmbedding` should be at least discrete or continuous'
        self.discrete = discrete
        self.continuous = continuous

        self.index_weight = None
        if self.has_discrete:
            if _weight is None:
                self.index_weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim), **factory_kwargs))
            else:
                assert list(_weight.shape) == [num_embeddings, embedding_dim], \
                    'Shape of weight does not match num_embeddings and embedding_dim'
                self.index_weight = nn.Parameter(_weight)

        self.value_layer = None
        self.activation = None
        if self.continuous:
            if token_values is not None:
                if not isinstance(token_values, Tensor):
                    token_values = torch.Tensor(token_values)
            else:
                token_values = torch.linspace(0., 1., self.num_embeddings)
            token_values = token_values.reshape(-1, 1).to(**factory_kwargs)

            self.value_layer = nn.Linear(1, embedding_dim, bias=False, **factory_kwargs)
            self.activation = activation
        self.register_buffer('token_values', token_values)

        self._value_weight = None
        if _weight is None:
            self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.has_discrete:
            nn.init.normal_(self.index_weight, std=1e-2)
        if self.continuous:
            nn.init.normal_(self.value_layer.weight, std=1e-2)
        self._fill_padding_idx_with_zero()

    def _fill_padding_idx_with_zero(self) -> None:
        if self.padding_idx is not None:
            with torch.no_grad():
                if self.has_discrete:
                    self.index_weight[self.padding_idx].fill_(0)
                if self.continuous and self.token_values is not None:
                    self.token_values[self.padding_idx].fill_(0)

    def forward(self, tokens: Optional[Tensor] = None, values: Optional[Tensor] = None) -> Tensor:
        assert not self.has_discrete or tokens is not None
        if values is None:  # use fixed `token_values`
            assert self.token_values is not None, \
                "`DiscreteContinuousEmbedding.token_values` cannot be empty when no `values` are provided"
            token_weight = self.token_weight if self.has_discrete else 0
            value_weight = self.value_weight if self.continuous else 0
            weight = token_weight + value_weight
            return F.embedding(tokens, weight, self.padding_idx)
        else:
            token_emb = F.embedding(tokens, self.token_weight, self.padding_idx) if self.has_discrete else 0
            value_emb = self.forward_values(tokens, values=values) if self.continuous else 0
            return token_emb + value_emb

    def forward_indices(self, tokens: Tensor) -> Tensor:
        assert self.has_discrete
        return F.embedding(tokens, self.token_weight, self.padding_idx)

    def forward_values(self, tokens: Tensor, values: Optional[Tensor] = None) -> Tensor:
        assert self.continuous
        if values is None:  # use fixed `token_values`
            assert self.token_values is not None, \
                "`DiscreteContinuousEmbedding.token_values` cannot be empty when no `values` are provided"
            return F.embedding(tokens, self.value_weight, self.padding_idx)
        else:
            return self._compute_value_embeddings(values)

    def _compute_value_embeddings(self, values: Tensor) -> Tensor:
        assert self.continuous
        values_emb = self.value_layer(values.view(-1, 1))
        values_emb = values_emb if self.activation is None else self.activation(values_emb)
        return values_emb

    @property
    def token_weight(self):
        if self.discrete:
            return self.index_weight
        elif self.discrete_ids is not None:
            index_weight = torch.zeros_like(self.index_weight)
            index_weight[self.discrete_ids] = self.index_weight[self.discrete_ids]
            return index_weight

    @property
    def value_weight(self):
        if self.continuous:
            if self.token_values is None:
                return None
            if self._value_weight is None:
                value_weight = self._compute_value_embeddings(self.token_values)
                if self.discrete_ids is not None:
                    value_weight[self.discrete_ids] = 0.
                return value_weight
            return self._value_weight

    @property
    def weight(self):
        if self.has_discrete:
            if self.token_values is None:
                return self.token_weight
            return self.token_weight + self.value_weight
        else:
            return self.value_weight

    @property
    def has_discrete(self):  # fully discrete or has discrete some token indices
        return self.discrete or self.discrete_ids is not None

    def train(self, mode=True):
        if mode or self.token_values is None:
            self._value_weight = None
        elif self.continuous:
            self._value_weight = self.value_weight.detach().to(device=self.token_values.device)
        return super().train(mode)

    def extra_repr(self) -> str:
        s = '{num_embeddings}, {embedding_dim}'
        if self.padding_idx is not None:
            s += ', padding_idx={padding_idx}'
        return s.format(**self.__dict__)


class DiscreteDenseContinuousEmbedding(DiscreteContinuousEmbedding):
    def __init__(
            self,
            num_embeddings: int,
            embedding_dim: int,
            depth: int = 2,
            discrete: bool = True,
            continuous: bool = True,
            discrete_ids: Optional[Union[list, Tensor]] = None,
            token_values: Optional[Union[list, Tensor]] = None,
            padding_idx: Optional[int] = None,
            _weight: Optional[Tensor] = None,
            device=None,
            dtype=None
    ) -> None:
        super().__init__(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            discrete=discrete,
            continuous=continuous,
            discrete_ids=discrete_ids,
            token_values=token_values,
            padding_idx=padding_idx,
            device=device,
            dtype=dtype
        )

        if self.continuous:
            factory_kwargs = {'device': device, 'dtype': dtype}

            layers = [nn.Sequential(
                nn.Linear(1, embedding_dim, **factory_kwargs),
                nn.Mish() if depth > 1 else nn.Identity()
            )]

            for i in range(depth - 1):
                layers.append(nn.Sequential(
                    nn.Linear(embedding_dim, embedding_dim, **factory_kwargs),
                    nn.Mish() if i < depth - 2 else nn.Identity()
                ))

            self.value_layer = nn.Sequential(*layers)

    def reset_parameters(self) -> None:
        if self.has_discrete:
            nn.init.normal_(self.index_weight, std=1e-2)
        if self.continuous:
            for module in self.value_layer.modules():
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, std=1e-2)
        self._fill_padding_idx_with_zero()


class AbsolutePositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len):
        super().__init__()
        self.dim = dim
        self.scale = dim ** -0.5
        self.max_seq_len = max_seq_len
        self.emb = nn.Embedding(max_seq_len, dim)

    def forward(self, x: Tensor, pos: Optional[Tensor] = None):
        seq_len = x.shape[1]
        assert seq_len <= self.max_seq_len

        if pos is None:
            pos = torch.arange(seq_len, device=x.device)

        pos_emb = self.emb(pos)
        pos_emb = pos_emb * self.scale
        return pos_emb

    def extra_repr(self) -> str:
        return f'dim={self.dim}'


class FixedPositionalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x: Tensor, pos: Optional[Tensor] = None, seq_dim: int = 1, offset: int = 0):
        if pos is None:
            pos = torch.arange(x.shape[seq_dim], device=x.device)

        pos = pos.type_as(self.inv_freq) + offset
        sinusoid_inp = pos.unsqueeze(-1) * self.inv_freq
        pos_emb = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
        return pos_emb

    def extra_repr(self) -> str:
        return f'dim={self.dim}'


class ALiBiPositionalBias(nn.Module):
    def __init__(self, heads: int, total_heads: int, symmetric: bool = True):
        super().__init__()
        self.heads = heads
        self.total_heads = total_heads
        self.symmetric = symmetric

        slopes = torch.Tensor(self._compute_slopes(heads)).view(-1, 1, 1)
        if not symmetric:
            slopes = torch.stack([slopes, torch.roll(slopes, -1)])
        self.register_buffer('slopes', slopes, persistent=False)

    @staticmethod
    def _compute_slopes(heads):
        def slopes_power_of_2(n):
            start = (2 ** (-2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio ** i for i in range(n)]

        if math.log2(heads).is_integer():
            return slopes_power_of_2(heads)

        closest_power_of_2 = 2 ** math.floor(math.log2(heads))
        return slopes_power_of_2(closest_power_of_2) \
            + slopes_power_of_2(2 * closest_power_of_2)[0::2][:heads - closest_power_of_2]

    def get_bias(self, i: int, j: int, k: int = 0):
        i_arange = torch.arange(k, i + k, dtype=torch.int, device=self.slopes.device)
        j_arange = torch.arange(j, dtype=torch.int, device=self.slopes.device)
        return -torch.abs(j_arange[None, None, :] - i_arange[None, :, None])

    def get_slopes(self):
        return self.slopes

    def forward(self, i: int, j: int, k: int = 0, bias: Optional[Tensor] = None):
        if bias is not None and bias.shape[-2] >= i and bias.shape[-1] >= j - k:
            bias = bias[..., :i, :j]
        else:
            bias = self.get_bias(i, j, k)

        slopes = self.get_slopes()
        if self.total_heads - slopes.shape[-3] > 0:
            slopes = F.pad(slopes, (0, 0, 0, 0, 0, self.total_heads - slopes.shape[-3]))

        if self.symmetric:
            return slopes * bias
        else:
            return slopes[0] * torch.tril(bias) + slopes[1] * torch.triu(bias)


class LearnedALiBiPositionalBias(ALiBiPositionalBias):
    def __init__(self, heads: int, total_heads: int, symmetric: bool = True):
        super().__init__(heads, total_heads, symmetric)
        log_slopes = torch.log(self.slopes)
        self.learned_logslopes = nn.Parameter(log_slopes)

    def get_slopes(self):
        return self.learned_logslopes.exp()
