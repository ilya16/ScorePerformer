"""
Transformer Attention Layers with data caching support for inference.

Adapted from: https://github.com/lucidrains/x-transformers
"""

import copy
from dataclasses import dataclass, field
from functools import partial
from typing import Optional, Union, List

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor

from scoreperformer.utils import equals
from .attend import AttentionIntermediates
from .attention import Attention, AttentionConfig
from .feedforward import FeedForward, FeedForwardConfig
from ..constructor import VariableModuleConfig, Constructor, Registry
from ..layers import Residual, AdaptiveLayerNorm


@dataclass
class TransformerIntermediates:
    hiddens: Optional[List[Tensor]] = None
    attention: Optional[List[AttentionIntermediates]] = None


TransformerRegistry = type("_TransformerRegistry", (Registry,), {})()


@dataclass
class TransformerConfig(VariableModuleConfig):
    _target_: str = "default"

    dim: int = 512
    depth: int = 4
    heads: int = 8

    attention: Union[AttentionConfig, DictConfig] = None
    feed_forward: Union[FeedForwardConfig, DictConfig] = None

    causal: bool = False
    cross_attend: bool = False
    only_cross: bool = False

    pre_norm: bool = True
    use_adanorm: bool = False
    style_emb_dim: Optional[int] = None


@TransformerRegistry.register("default")
class Transformer(nn.Module, Constructor):
    def __init__(
            self,
            dim: int = 512,
            depth: int = 4,
            heads: int = 8,
            attention: Union[AttentionConfig, DictConfig] = None,
            feed_forward: Union[FeedForwardConfig, DictConfig] = None,
            causal: bool = False,
            cross_attend: bool = False,
            only_cross: bool = False,
            pre_norm: bool = True,
            use_adanorm: bool = False,
            style_emb_dim: Optional[int] = None
    ):
        super().__init__()

        attention = attention if attention else AttentionConfig()
        feed_forward = feed_forward if feed_forward else FeedForwardConfig()

        self.dim = dim
        self.depth = depth
        self.layers = nn.ModuleList([])

        # normalization

        self.pre_norm = pre_norm
        self.ada_norm = use_adanorm

        assert not use_adanorm or style_emb_dim is not None, 'condition_dim should be provided with adanorm'

        norm_fn = partial(AdaptiveLayerNorm, dim, style_emb_dim) if use_adanorm else partial(nn.LayerNorm, dim)

        # layers

        self.cross_attend = cross_attend

        if cross_attend and not only_cross:
            default_block = ('a', 'c', 'f')
        elif cross_attend and only_cross:
            default_block = ('c', 'f')
        else:
            default_block = ('a', 'f')

        # calculate layer block order

        self.layer_types = default_block * depth
        self.num_attn_layers = len(list(filter(equals('a'), self.layer_types)))

        # whether it has post norm

        self.final_norm = norm_fn() if pre_norm else nn.Identity()

        # iterate and construct layers

        for ind, layer_type in enumerate(self.layer_types):

            if layer_type == 'a':
                layer = Attention.init(config=attention, dim=dim, heads=heads, causal=causal)
            elif layer_type == 'c':
                layer = Attention.init(config=attention, dim=dim, heads=heads)
            elif layer_type == 'f':
                layer = FeedForward.init(config=feed_forward, dim=dim)
            else:
                raise Exception(f'invalid layer type {layer_type}')

            residual = Residual(dim)

            pre_branch_norm = norm_fn() if pre_norm else None
            post_branch_norm = None
            post_main_norm = norm_fn() if not pre_norm else None

            norms = nn.ModuleList([
                pre_branch_norm,
                post_branch_norm,
                post_main_norm
            ])

            self.layers.append(nn.ModuleList([
                norms,
                layer,
                residual
            ]))

    def forward(
            self,
            x: Tensor,
            mask: Optional[Tensor] = None,
            context: Optional[Tensor] = None,
            context_mask: Optional[Tensor] = None,
            attn_mask: Optional[Tensor] = None,
            style_embeddings: Optional[Tensor] = None,
            mems: Optional[List[Tensor]] = None,
            intermediates_cache: Optional[TransformerIntermediates] = None,
            return_hiddens: bool = False
    ):
        assert not (self.cross_attend ^ (context is not None)), \
            'context must be passed in if cross_attend is set to True'
        assert not self.ada_norm or style_embeddings is not None, \
            'style_embeddings must be passed for AdaLayerNorm'

        hiddens = []
        attn_intermediates = []

        mems = mems.copy() if mems is not None else [None] * self.num_attn_layers

        has_cache = intermediates_cache is not None
        intermediates_cache = copy.copy(intermediates_cache) if has_cache else None

        x = x[:, -1:] if has_cache else x
        style_embeddings = style_embeddings[:, -1:] if has_cache and style_embeddings is not None else style_embeddings

        attn_shared_cache = None

        for ind, (layer_type, (norm, block, residual_fn)) in enumerate(zip(self.layer_types, self.layers)):
            cache = None

            if layer_type == 'a':
                if has_cache:
                    cache = intermediates_cache.hiddens.pop(0)
                    x = torch.cat([cache, x], dim=1)

                if return_hiddens:
                    hiddens.append(x)

                x = x[:, -1:] if has_cache else x

                layer_mem = mems.pop(0) if mems else None

            if has_cache:
                if layer_type in ('a', 'c'):
                    cache = intermediates_cache.attention.pop(0)

            residual = x

            pre_norm, post_branch_norm, post_main_norm = norm

            if pre_norm is not None:
                x = pre_norm(x, condition=style_embeddings) if self.ada_norm else pre_norm(x)

            if layer_type == 'a':
                out, inter, attn_shared_cache = block(
                    x, mask=mask, attn_mask=attn_mask,
                    mem=layer_mem, cache=cache, shared_cache=attn_shared_cache,
                )
            elif layer_type == 'c':
                out, inter, _ = block(x, context=context, mask=mask, context_mask=context_mask)
            elif layer_type == 'f':
                out = block(x)

            if post_branch_norm is not None:
                out = post_branch_norm(x, condition=style_embeddings) if self.ada_norm else post_branch_norm(out)

            x = residual_fn(out, residual)

            if return_hiddens:
                if layer_type in ('a', 'c'):
                    attn_intermediates.append(inter)

            if post_main_norm is not None:
                x = post_main_norm(x, condition=style_embeddings) if self.ada_norm else post_main_norm(x)

        x = self.final_norm(x, condition=style_embeddings) if self.ada_norm else self.final_norm(x)

        if has_cache:
            cache = intermediates_cache.hiddens.pop(0)
            x = torch.cat([cache, x], dim=1)

        if return_hiddens:
            hiddens.append(x)
            intermediates = TransformerIntermediates(
                hiddens=hiddens,
                attention=attn_intermediates
            )

            return x, intermediates

        return x


@dataclass
class EncoderConfig(TransformerConfig):
    _target_: str = "encoder"
    causal: bool = False


@TransformerRegistry.register("encoder")
class Encoder(Transformer):
    def __init__(self, **kwargs):
        super().__init__(causal=False, **kwargs)


@dataclass
class DecoderConfig(TransformerConfig):
    _target_: str = "decoder"
    causal: bool = True


@TransformerRegistry.register("decoder")
class Decoder(Transformer):
    def __init__(self, **kwargs):
        super().__init__(causal=True, **kwargs)
