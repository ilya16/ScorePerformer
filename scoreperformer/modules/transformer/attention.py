"""
Transformer Attention with data caching support for inference.

Adapted from: https://github.com/lucidrains/x-transformers
"""
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor

from scoreperformer.utils import default, or_reduce
from .attend import AttentionIntermediates, Attend
from .embeddings import ALiBiPositionalBias, LearnedALiBiPositionalBias
from ..constructor import Constructor, ModuleConfig


@dataclass
class AttentionSharedIntermediates:
    rel_pos_bias: Optional[Tensor] = None


@dataclass
class AttentionConfig(ModuleConfig):
    dim: int = 512
    dim_head: int = 64
    heads: int = 8
    causal: bool = False
    dropout: float = 0.
    one_kv_head: bool = False
    num_mem_kv: int = 0
    shared_kv: bool = False
    value_dim_head: Optional[int] = None
    max_attend_past: Optional[int] = None
    alibi_pos_bias: bool = False
    alibi_num_heads: Optional[int] = None
    alibi_symmetric: bool = True
    alibi_learned: bool = False


class Attention(nn.Module, Constructor):
    def __init__(
            self,
            dim: int,
            dim_head: int = 64,
            heads: int = 8,
            causal: bool = False,
            dropout: float = 0.,
            one_kv_head: bool = False,
            num_mem_kv: int = 0,
            max_attend: Optional[int] = None,
            alibi_pos_bias: bool = False,
            alibi_num_heads: Optional[int] = None,
            alibi_symmetric: bool = True,
            alibi_learned: bool = False,
    ):
        super().__init__()
        self.scale = dim_head ** -0.5

        self.heads = heads
        self.causal = causal
        self.max_attend = max_attend

        self.one_kv_head = one_kv_head
        out_dim = q_dim = dim_head * heads
        kv_dim = dim_head if one_kv_head else dim_head * heads

        self.to_q = nn.Linear(dim, q_dim, bias=False)
        self.to_k = nn.Linear(dim, kv_dim, bias=False)
        self.to_v = nn.Linear(dim, kv_dim, bias=False)

        # relative positional bias

        self.rel_pos = None
        if alibi_pos_bias:
            alibi_num_heads = default(alibi_num_heads, heads)
            assert alibi_num_heads <= heads, 'number of ALiBi heads must be less than the total number of heads'
            alibi_pos_klass = LearnedALiBiPositionalBias if alibi_learned else ALiBiPositionalBias
            self.rel_pos = alibi_pos_klass(
                heads=alibi_num_heads,
                total_heads=heads,
                symmetric=alibi_symmetric or causal
            )

        # attend class - includes core attention algorithm + talking heads

        self.attend = Attend(
            causal=causal,
            dropout=dropout,
            scale=self.scale
        )

        # add memory key / values

        self.num_mem_kv = num_mem_kv
        if num_mem_kv > 0:
            self.mem_k = nn.Parameter(torch.randn(heads, num_mem_kv, dim_head))
            self.mem_v = nn.Parameter(torch.randn(heads, num_mem_kv, dim_head))

        # output layer

        self.to_out = nn.Linear(out_dim, dim, bias=False)

    def forward(
            self,
            x: Tensor,
            context: Optional[Tensor] = None,
            mask: Optional[Tensor] = None,
            context_mask: Optional[Tensor] = None,
            attn_mask: Optional[Tensor] = None,
            prev_attn: Optional[Tensor] = None,
            mem: Optional[Tensor] = None,
            cache: Optional[AttentionIntermediates] = None,
            shared_cache: Optional[AttentionSharedIntermediates] = None
    ):
        b, n = x.shape[:2]
        h, scale, device = self.heads, self.scale, x.device
        has_context, has_mem, has_cache = context is not None, mem is not None, cache is not None
        assert not (has_mem and has_cache), 'cache is not compatible with memory keys'
        assert not (has_context and has_cache), 'cache is not compatible with context yet'

        kv_input = default(context, x)

        q_input = x
        k_input = kv_input
        v_input = kv_input

        if has_mem:
            k_input = torch.cat((mem, k_input), dim=-2)
            v_input = torch.cat((mem, v_input), dim=-2)

        q = self.to_q(q_input)
        k = self.to_k(k_input)
        v = self.to_v(v_input) if self.to_v is not None else k

        q = rearrange(q, 'b n (h d) -> b h n d', h=h)

        if not self.one_kv_head:
            k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (k, v))

        input_mask = mask if context_mask is None else context_mask

        if self.num_mem_kv > 0:
            mem_k, mem_v = map(lambda t: repeat(t, 'h n d -> b h n d', b=b), (self.mem_k, self.mem_v))

            k = torch.cat((mem_k, k), dim=-2)
            v = torch.cat((mem_v, v), dim=-2)

            if input_mask is not None:
                input_mask = F.pad(input_mask, (self.num_mem_kv, 0), value=True)

        k = torch.cat([cache.keys, k], dim=-2) if has_cache else k
        v = torch.cat([cache.values, v], dim=-2) if has_cache else v

        i, j = map(lambda t: t.shape[-2], (q, k))

        # determine masking

        masks = []
        final_attn_mask = None

        if input_mask is not None:
            input_mask = rearrange(input_mask, 'b j -> b 1 1 j')
            masks.append(~input_mask)

        if attn_mask is not None:
            assert 2 <= attn_mask.ndim <= 4, \
                'attention mask must have greater than 2 dimensions but less than or equal to 4'
            if attn_mask.ndim == 2:
                attn_mask = rearrange(attn_mask, 'i j -> 1 1 i j')
            elif attn_mask.ndim == 3:
                attn_mask = rearrange(attn_mask, 'h i j -> 1 h i j')
            attn_mask = attn_mask[:, :, -1:] if has_cache else attn_mask
            masks.append(~attn_mask)

        if self.max_attend is not None:
            range_q = torch.arange(j - i, j, device=device)
            range_k = torch.arange(j, device=device)
            dist = rearrange(range_q, 'i -> 1 1 i 1') - rearrange(range_k, 'j -> 1 1 1 j')
            max_attend_mask = torch.logical_or(-self.max_attend < dist, dist > self.max_attend)
            masks.append(max_attend_mask)

        if len(masks) > 0:
            final_attn_mask = ~or_reduce(masks)

        # prepare relative positional bias, if needed

        rel_pos_bias, attn_bias = None, None
        if self.rel_pos is not None:
            if shared_cache is not None and shared_cache.rel_pos_bias is not None:
                rel_pos_bias = shared_cache.rel_pos_bias
            else:
                rel_pos_bias = self.rel_pos.get_bias(i, j, k=j - i).to(dtype=q.dtype)
            attn_bias = self.rel_pos(i, j, k=j - i, bias=rel_pos_bias)

        # attention is all we need

        out, intermediates = self.attend(
            q, k, v,
            mask=final_attn_mask,
            attn_bias=attn_bias,
            prev_attn=prev_attn
        )

        # merge heads

        out = rearrange(out, 'b h n d -> b n (h d)')

        # combine the heads

        out = self.to_out(out)

        if mask is not None:
            mask = mask[:, -1:] if has_cache else mask
            out = out * mask[..., None]

        shared_intermediates = AttentionSharedIntermediates(rel_pos_bias=rel_pos_bias)

        return out, intermediates, shared_intermediates
