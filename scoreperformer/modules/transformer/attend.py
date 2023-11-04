"""
Attention with efficient memory attention support.

Adapted from: https://github.com/lucidrains/x-transformers
"""

from collections import namedtuple
from dataclasses import dataclass
from functools import partial
from typing import Optional

import torch
import torch.nn.functional as F
from einops import rearrange
from packaging import version
from torch import nn, einsum, Tensor

from scoreperformer.utils import exists, default

# constants

EfficientAttentionConfig = namedtuple('EfficientAttentionConfig',
                                      ['enable_flash', 'enable_math', 'enable_mem_efficient'])


@dataclass
class AttentionIntermediates:
    keys: Optional[Tensor] = None
    values: Optional[Tensor] = None
    qk_similarities: Optional[Tensor] = None

    def to_tuple(self):
        return self.keys, self.values, self.qk_similarities


# main class

class Attend(nn.Module):
    def __init__(
            self,
            *,
            dropout: float = 0.,
            causal: bool = False,
            scale: Optional[float] = None,
    ):
        super().__init__()
        self.scale = scale
        self.causal = causal
        self.attn_fn = partial(F.softmax, dtype=torch.float32)

        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)

        # efficient attention
        self.efficient = version.parse(torch.__version__) >= version.parse('2.0.0')
        self.config = EfficientAttentionConfig(enable_flash=False, enable_math=True, enable_mem_efficient=True)

    def efficient_attn(
            self,
            q, k, v,
            mask=None,
            attn_bias=None
    ):
        batch, heads, q_len, _, k_len, is_cuda, device = *q.shape, k.shape[-2], q.is_cuda, q.device

        # Recommended for multi-query single-key-value attention by Tri Dao
        # kv shape torch.Size([1, 512, 64]) -> torch.Size([1, 8, 512, 64])

        intermediates = AttentionIntermediates(keys=k, values=v)

        if k.ndim == 3:
            k = rearrange(k, 'b ... -> b 1 ...').expand(-1, q.shape[1], -1, -1)

        if v.ndim == 3:
            v = rearrange(v, 'b ... -> b 1 ...').expand(-1, q.shape[1], -1, -1)

        # Check if mask exists and expand to compatible shape
        # The mask is B L, so it would have to be expanded to B H N L

        causal = self.causal

        if exists(mask):
            assert mask.ndim == 4
            mask = mask.expand(batch, heads, q_len, k_len)

            # manually handle causal mask, if another mask was given

            if causal:
                causal_mask = torch.ones((q_len, k_len), dtype=torch.bool, device=device).triu(k_len - q_len + 1)
                mask = mask & ~causal_mask
                causal = False

        # handle alibi positional bias
        # convert from bool to float

        if exists(attn_bias):
            attn_bias = rearrange(attn_bias, 'h i j -> 1 h i j').expand(batch, -1, -1, -1)

            # if mask given, the mask would already contain the causal mask from above logic
            # otherwise, if no mask given but still causal, mask out alibi positional bias to a large negative number

            mask_value = -torch.finfo(q.dtype).max

            if exists(mask):
                attn_bias = attn_bias.masked_fill(~mask, mask_value // 2)
            elif causal:
                causal_mask = torch.ones((q_len, k_len), dtype=torch.bool, device=device).triu(k_len - q_len + 1)
                attn_bias = attn_bias.masked_fill(causal_mask, mask_value // 2)
                causal = False

            # scaled_dot_product_attention handles attn_mask either as bool or additive bias
            # make it an additive bias here

            mask = attn_bias

        # pytorch 2.0 attention: q, k, v, mask, dropout, causal, softmax_scale

        with torch.backends.cuda.sdp_kernel(**self.config._asdict()):
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=mask,
                dropout_p=self.dropout if self.training else 0.,
                is_causal=causal and q_len != 1
            )

        return out, intermediates

    def forward(
            self,
            q, k, v,
            mask=None,
            attn_bias=None,
            prev_attn=None
    ):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        if self.efficient:
            assert not exists(prev_attn), 'residual attention not compatible with efficient attention'
            return self.efficient_attn(q, k, v, mask=mask, attn_bias=attn_bias)

        n, device = q.shape[-2], q.device
        scale = default(self.scale, q.shape[-1] ** -0.5)

        kv_einsum_eq = 'b j d' if k.ndim == 3 else 'b h j d'

        dots = einsum(f'b h i d, {kv_einsum_eq} -> b h i j', q, k) * scale

        if exists(prev_attn):
            dots = dots + prev_attn

        qk_similarities = dots.clone()

        if exists(attn_bias):
            dots = dots + attn_bias

        dtype = dots.dtype
        mask_value = -torch.finfo(dots.dtype).max

        if exists(mask):
            dots = dots.masked_fill(~mask, mask_value)

        if self.causal:
            i, j = dots.shape[-2:]
            causal_mask = torch.ones((i, j), dtype=torch.bool, device=device).triu(j - i + 1)
            dots = dots.masked_fill(causal_mask, mask_value)

        attn = self.attn_fn(dots, dim=-1)
        attn = attn.type(dtype)

        attn = self.attn_dropout(attn)

        out = einsum(f'b h i j, {kv_einsum_eq} -> b h i d', attn, v)

        intermediates = AttentionIntermediates(
            keys=k,
            values=v,
            qk_similarities=qk_similarities
        )

        return out, intermediates
