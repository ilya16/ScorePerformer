""" TupleTransformer with MMD-VAE output embedding heads. """

from dataclasses import dataclass
from typing import Optional, Union, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import Tensor

from scoreperformer.modules.transformer import TransformerConfig
from scoreperformer.utils import ExplicitEnum
from .embeddings import TupleTokenEmbeddingsConfig, TupleTokenHeadsConfig, TupleTokenRegressionHeadConfig
from .transformer import TupleTransformerOutput, TupleTransformerConfig, TupleTransformer


class EmbeddingAggregateModes(ExplicitEnum):
    SAME = "same"
    MEAN = "mean"
    BEAT_MEAN = "beat_mean"
    BAR_MEAN = "bar_mean"
    ONSET_MEAN = "onset_mean"
    ISOLATED_BAR_MEAN = "isolated_bar_mean"


@dataclass
class MMDTupleTransformerOutput(TupleTransformerOutput):
    latents: Optional[Union[Tensor, List[Tensor]]] = None
    embeddings: Optional[Tensor] = None
    full_embeddings: Optional[Tensor] = None
    dropout_mask: Optional[Tensor] = None
    loss: Optional[Tensor] = None
    losses: Optional[Dict[str, Tensor]] = None


@dataclass
class MMDTupleTransformerConfig(TupleTransformerConfig):
    latent_dim: Union[int, List[int]] = 64
    aggregate_mode: Union[EmbeddingAggregateModes, List[EmbeddingAggregateModes]] = EmbeddingAggregateModes.MEAN
    hierarchical: bool = False
    hierarchical_with_context: bool = True
    latent_dropout: Union[float, List[float]] = 0.
    inclusive_latent_dropout: bool = True
    deadpan_zero_latent: bool = False
    loss_weight: float = 1.0


class MMDVAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.linear = nn.Linear(input_dim, latent_dim)

    def forward(self, inputs: Tensor):
        return self.linear(inputs)


class MMDTupleTransformer(TupleTransformer):
    def __init__(
            self,
            num_tokens: Dict[str, int],
            dim: int = 512,
            max_seq_len: int = 1024,
            transformer: Union[DictConfig, TransformerConfig] = None,
            token_embeddings: Union[DictConfig, TupleTokenEmbeddingsConfig] = None,
            use_abs_pos_emb: bool = True,
            emb_norm: bool = False,
            emb_dropout: float = 0.0,
            context_emb_dim: Optional[int] = None,
            context_emb_mode: str = "attention",
            style_emb_dim: Optional[int] = None,
            style_emb_mode: str = "cat",
            lm_head: Optional[Union[DictConfig, TupleTokenHeadsConfig]] = None,
            regression_head: Optional[Union[DictConfig, TupleTokenRegressionHeadConfig]] = None,
            latent_dim: Union[int, List[int]] = 64,
            aggregate_mode: Union[
                EmbeddingAggregateModes, List[EmbeddingAggregateModes]
            ] = EmbeddingAggregateModes.MEAN,
            hierarchical: bool = False,
            hierarchical_with_context: bool = True,  # condition of context and all preceding latents
            latent_dropout: Union[float, List[float]] = 0.,  # dropout whole latent vectors with given probability
            inclusive_latent_dropout: bool = True,  # dropout all lower-level latents for dropped out segment
            deadpan_zero_latent: bool = False,  # optimize latents to zero for deadpan performances
            loss_weight: float = 1.0
    ):
        if transformer is None:
            transformer = TransformerConfig(_target_="default")
        if token_embeddings is None:
            token_embeddings = TupleTokenEmbeddingsConfig()
        super().__init__(
            num_tokens=num_tokens,
            dim=dim,
            max_seq_len=max_seq_len,
            transformer=transformer,
            token_embeddings=token_embeddings,
            use_abs_pos_emb=use_abs_pos_emb,
            emb_norm=emb_norm,
            emb_dropout=emb_dropout,
            context_emb_dim=context_emb_dim,
            context_emb_mode=context_emb_mode,
            style_emb_dim=style_emb_dim,
            style_emb_mode=style_emb_mode,
            lm_head=lm_head,
            regression_head=regression_head
        )

        if not isinstance(latent_dim, int):
            if isinstance(aggregate_mode, str):
                aggregate_mode = [aggregate_mode] * len(latent_dim)
            else:
                aggregate_mode = list(aggregate_mode)

        if isinstance(aggregate_mode, str):
            assert EmbeddingAggregateModes.has_value(aggregate_mode), \
                f'`{aggregate_mode}` is not a valid aggregate_mode`, available modes: {EmbeddingAggregateModes.list()}'
        else:
            latent_dim = [latent_dim] * len(aggregate_mode) if isinstance(latent_dim, int) else latent_dim
            for mode in aggregate_mode:
                assert EmbeddingAggregateModes.has_value(mode), \
                    f'`{mode}` is not a valid aggregate_mode`, available modes: {EmbeddingAggregateModes.list()}'

        assert not hierarchical or isinstance(aggregate_mode, list), \
            '`hierarchical` mode can only be used with multiple VAE heads'
        self.hierarchical = hierarchical
        self.hierarchical_with_context = hierarchical_with_context

        if not isinstance(latent_dim, int):
            latent_dropout = [latent_dropout] * len(latent_dim) if isinstance(latent_dropout, float) else latent_dropout

        self.aggregate_mode = aggregate_mode
        self.latent_dim = latent_dim
        self.latent_dropout = latent_dropout
        self.inclusive_latent_dropout = inclusive_latent_dropout
        self.deadpan_zero_latent = deadpan_zero_latent

        if isinstance(latent_dim, int):
            self.vae_head = MMDVAE(
                input_dim=dim,
                latent_dim=latent_dim
            )
            self.embedding_dim = latent_dim
        else:
            self.vae_head = nn.ModuleDict()
            input_dim = dim
            for mode, latent_dim_i in zip(aggregate_mode, latent_dim):
                self.vae_head[mode] = MMDVAE(
                    input_dim=input_dim,
                    latent_dim=latent_dim_i
                )
                if self.hierarchical:
                    if self.hierarchical_with_context:
                        input_dim += latent_dim_i
                    else:
                        input_dim = latent_dim_i
            self.embedding_dim = sum(latent_dim)

        self.criterion = MMDLoss()
        self.loss_weight = loss_weight

        # TODO: make configurable
        self.pad_token_id = 0
        self.mask_token_id = 1
        self.sos_token_id = 2
        self.eos_token_id = 3

        self._mask_bars = False

    def forward(
            self,
            x: Tensor,
            mask: Optional[Tensor] = None,
            x_extra: Optional[Union[Tensor, List[Tensor]]] = None,
            latents: Optional[Union[Tensor, List[Tensor]]] = None,
            bars: Optional[Tensor] = None,
            beats: Optional[Tensor] = None,
            onsets: Optional[Tensor] = None,
            deadpan_mask: Optional[Tensor] = None,
            return_embeddings: bool = False,
            return_attn: bool = False,
            compute_loss: bool = True,
            **kwargs
    ):
        # first aggregate_mode is the dominant
        main_aggregate_mode = self.aggregate_mode[0] if isinstance(self.aggregate_mode, list) else self.aggregate_mode

        x_input = x
        if main_aggregate_mode == EmbeddingAggregateModes.ISOLATED_BAR_MEAN or self._mask_bars:
            x_input = x.clone().detach()
            x_input[..., 0][x_input[..., 0] > self.eos_token_id] = self.mask_token_id

        attn_mask = None
        if main_aggregate_mode == EmbeddingAggregateModes.ISOLATED_BAR_MEAN:
            masks = []
            for bars_i in bars:
                unique_bars, counts = torch.unique(bars_i, return_counts=True)
                counts = counts[unique_bars > self.pad_token_id]
                pad = x.shape[1] - counts.sum()
                blocks = [torch.ones(c, c, device=x.device) for c in counts] + [torch.zeros(pad, pad, device=x.device)]
                masks.append(torch.block_diag(*blocks))

            attn_mask = torch.stack(masks)[:, None].bool()

        transformer_outputs = super().forward(
            x=x_input, mask=mask, x_extra=x_extra, attn_mask=attn_mask,
            return_embeddings=return_embeddings, return_attn=return_attn,
            **kwargs
        )

        out = transformer_outputs.hidden_state

        if mask is None:
            mask = torch.ones_like(out[..., :1], dtype=torch.bool, device=x.device)
        else:
            mask = mask.unsqueeze(-1)
            out = out * mask

        assert not self.deadpan_zero_latent or deadpan_mask is not None

        loss, losses = None, {}
        if isinstance(self.aggregate_mode, str):
            segments = self._get_segments(self.aggregate_mode, bars=bars, beats=beats, onsets=onsets)
            latents, latents_mask, embeddings, drop_mask = self._forward_latents(
                out, mask, self.aggregate_mode, latent_dropout=self.latent_dropout,
                segments=segments, latents=latents
            )
            drop_mask = drop_mask.expand_as(embeddings)

            if compute_loss:
                losses[f'MMD/{self.aggregate_mode}'] = self.loss_weight * self.criterion(latents, mask=latents_mask)

                if self.deadpan_zero_latent:
                    deadpan_latents = latents[deadpan_mask[:, None] * latents_mask]
                    if torch.any(deadpan_latents):
                        losses[f'MMD/{self.aggregate_mode}/deadpan'] = F.mse_loss(
                            deadpan_latents, torch.zeros_like(deadpan_latents)
                        )
        else:
            _latents = latents
            prior_drop_mask = None
            latents, embeddings, drop_masks = [], [], []
            for i, aggregate_mode in enumerate(self.aggregate_mode):
                segments = self._get_segments(aggregate_mode, bars=bars, beats=beats, onsets=onsets)
                latents_i, latents_mask_i, embeddings_i, drop_mask_i = self._forward_latents(
                    out, mask, aggregate_mode, latent_dropout=self.latent_dropout[i],
                    segments=segments, latents=None if _latents is None else _latents[i]
                )

                if self.training and self.inclusive_latent_dropout:
                    if prior_drop_mask is None:
                        prior_drop_mask = drop_mask_i
                    elif drop_mask_i is not None:
                        prior_drop_mask = drop_mask_i = prior_drop_mask + drop_mask_i

                latents.append(latents_i)
                embeddings.append(embeddings_i)
                drop_masks.append(drop_mask_i.expand_as(embeddings_i))

                if self.hierarchical:
                    if self.hierarchical_with_context:
                        out = torch.concatenate([out, embeddings_i], dim=-1)
                    else:
                        out = embeddings_i

                if compute_loss:
                    losses[f'MMD/{aggregate_mode}'] = self.loss_weight * self.criterion(latents_i, mask=latents_mask_i)

                    if self.deadpan_zero_latent:
                        deadpan_latents_i = latents_i[deadpan_mask[:, None] * latents_mask_i]
                        if torch.any(deadpan_latents_i):
                            losses[f'MMD/{aggregate_mode}/deadpan'] = F.mse_loss(
                                deadpan_latents_i, torch.zeros_like(deadpan_latents_i)
                            )

            embeddings = torch.cat(embeddings, dim=-1)
            drop_mask = torch.cat(drop_masks, dim=-1)

        embeddings = embeddings * mask

        if self.training:
            full_embeddings = embeddings.clone()
            drop_mask = drop_mask * mask * (~deadpan_mask[:, None, None])
            embeddings = embeddings * (~drop_mask)
        else:
            full_embeddings = embeddings
            drop_mask = None

        if compute_loss:
            loss = sum(losses.values())
            losses['MMD'] = loss

        return MMDTupleTransformerOutput(
            hidden_state=transformer_outputs.hidden_state,
            logits=transformer_outputs.logits,
            attentions=transformer_outputs.attentions,
            latents=latents,
            embeddings=embeddings,
            full_embeddings=full_embeddings,
            dropout_mask=drop_mask,
            loss=loss,
            losses=losses
        )

    def _forward_latents(
            self,
            out: Tensor,
            mask: Tensor,
            aggregate_mode: str,
            latent_dropout: float = 0.,
            segments: Optional[Tensor] = None,
            latents: Optional[Tensor] = None
    ):
        b, t = out.shape[:2]

        segment_mode = aggregate_mode in (
            EmbeddingAggregateModes.ISOLATED_BAR_MEAN,
            EmbeddingAggregateModes.BAR_MEAN,
            EmbeddingAggregateModes.BEAT_MEAN,
            EmbeddingAggregateModes.ONSET_MEAN
        )

        latents_mask = None
        if latents is None:
            if aggregate_mode == EmbeddingAggregateModes.MEAN:
                out = out.sum(dim=1) / mask.sum(dim=1)
                out = out.unsqueeze(1)
                latents_mask = torch.ones_like(out[..., 0], dtype=torch.bool)
            elif segment_mode:
                # build segment alignment
                alignment = torch.zeros(b, t, segments.max() + 1, device=out.device)
                indices = (
                    torch.arange(b).repeat_interleave(t),
                    torch.arange(t).repeat(b),
                    segments.view(-1)
                )
                alignment[indices] = 1.

                # aggregate output embeddings by segments
                counts = torch.maximum(torch.tensor(1), alignment.sum(dim=1))[..., None]
                out = (out.transpose(1, 2) @ alignment).transpose(1, 2) / counts

                latents_mask = torch.all(out != 0., dim=-1)
            else:
                latents_mask = mask[..., 0]

            latents = self.vae_head(out) if isinstance(self.vae_head, MMDVAE) else self.vae_head[aggregate_mode](out)
            latents = latents * latents_mask[..., None]

        embeddings = latents

        if aggregate_mode != EmbeddingAggregateModes.MEAN and self.training and latent_dropout > 0.:
            drop_mask = dropout_latent_mask(latents_mask, latent_dropout)
        else:
            drop_mask = torch.zeros_like(latents_mask[..., None], dtype=torch.bool)

        if aggregate_mode == EmbeddingAggregateModes.MEAN:
            embeddings = embeddings.expand(-1, out.shape[1], -1)
            if drop_mask is not None:
                drop_mask = drop_mask.expand(-1, out.shape[1], -1)
        elif segment_mode:
            # distribute embeddings
            embeddings = embeddings[(torch.arange(b).repeat_interleave(t), segments.view(-1))].view(b, t, -1)
            if drop_mask is not None:
                drop_mask = drop_mask[(torch.arange(b).repeat_interleave(t), segments.view(-1))].view(b, t, -1)

        embeddings = embeddings * mask

        return latents, latents_mask, embeddings, drop_mask

    @staticmethod
    def _get_segments(
            aggregate_mode: str,
            bars: Optional[Tensor] = None,
            beats: Optional[Tensor] = None,
            onsets: Optional[Tensor] = None
    ):
        if aggregate_mode in (EmbeddingAggregateModes.BAR_MEAN, EmbeddingAggregateModes.ISOLATED_BAR_MEAN):
            assert bars is not None, f'`bars` should be provided as inputs for aggregate_mode `{aggregate_mode}`'
            return bars
        elif aggregate_mode == EmbeddingAggregateModes.BEAT_MEAN:
            assert beats is not None, f'`beats` should be provided as inputs for aggregate_mode `{aggregate_mode}`'
            return beats
        elif aggregate_mode == EmbeddingAggregateModes.ONSET_MEAN:
            assert onsets is not None, f'`onsets` should be provided as inputs for aggregate_mode `{aggregate_mode}`'
            return onsets
        return None

    def embeddings_to_latents(
            self,
            embeddings: Tensor,
            mask: Optional[Tensor] = None,
            bars: Optional[Tensor] = None,
            beats: Optional[Tensor] = None,
            onsets: Optional[Tensor] = None
    ):
        if isinstance(self.aggregate_mode, str):
            segments = self._get_segments(self.aggregate_mode, bars=bars, beats=beats, onsets=onsets)
            latents = self._embeddings_to_latents(
                embeddings, self.aggregate_mode, segments=segments, mask=mask
            )
        else:
            embeddings = embeddings.split(list(self.latent_dim), dim=-1)
            latents = []
            for i, aggregate_mode in enumerate(self.aggregate_mode):
                segments = self._get_segments(aggregate_mode, bars=bars, beats=beats, onsets=onsets)
                latents_i = self._embeddings_to_latents(
                    embeddings[i], aggregate_mode, segments=segments, mask=mask
                )
                latents.append(latents_i)

        return latents

    @staticmethod
    def _embeddings_to_latents(
            embeddings: Tensor,
            aggregate_mode: str,
            mask: Optional[Tensor] = None,
            segments: Optional[Tensor] = None
    ):
        b, t = embeddings.shape[:2]

        segment_mode = aggregate_mode in (
            EmbeddingAggregateModes.ISOLATED_BAR_MEAN,
            EmbeddingAggregateModes.BAR_MEAN,
            EmbeddingAggregateModes.BEAT_MEAN,
            EmbeddingAggregateModes.ONSET_MEAN
        )

        if aggregate_mode == EmbeddingAggregateModes.MEAN:
            if mask is None:
                latents = embeddings.mean(dim=1)
            else:
                latents = embeddings.sum(dim=1) / mask.sum(dim=1)
            latents = latents.unsqueeze(1)
        elif segment_mode:
            # build segment alignment
            alignment = torch.zeros(b, t, segments.max() + 1, device=embeddings.device)
            indices = (
                torch.arange(b).repeat_interleave(t),
                torch.arange(t).repeat(b),
                segments.view(-1)
            )
            alignment[indices] = 1.

            # aggregate output embeddings by segments
            counts = torch.maximum(torch.tensor(1), alignment.sum(dim=1))[..., None]
            latents = (embeddings.transpose(1, 2) @ alignment).transpose(1, 2) / counts
        else:
            latents = embeddings

        return latents

    def latents_to_embeddings(
            self,
            latents: Tensor,
            seq_len: Tensor,
            bars: Optional[Tensor] = None,
            beats: Optional[Tensor] = None,
            onsets: Optional[Tensor] = None
    ):
        if isinstance(self.aggregate_mode, str):
            segments = self._get_segments(self.aggregate_mode, bars=bars, beats=beats, onsets=onsets)
            embeddings = self._latents_to_embeddings(
                latents, seq_len, self.aggregate_mode, segments=segments
            )
        else:
            embeddings = []
            for i, aggregate_mode in enumerate(self.aggregate_mode):
                segments = self._get_segments(aggregate_mode, bars=bars, beats=beats, onsets=onsets)
                embeddings_i = self._latents_to_embeddings(
                    latents[i], seq_len, aggregate_mode, segments=segments
                )
                embeddings.append(embeddings_i)

            embeddings = torch.cat(embeddings, dim=-1)

        return embeddings

    @staticmethod
    def _latents_to_embeddings(
            latents: Tensor,
            seq_len: Tensor,
            aggregate_mode: str,
            segments: Optional[Tensor] = None
    ):
        b, t = latents.shape[0], seq_len

        segment_mode = aggregate_mode in (
            EmbeddingAggregateModes.ISOLATED_BAR_MEAN,
            EmbeddingAggregateModes.BAR_MEAN,
            EmbeddingAggregateModes.BEAT_MEAN,
            EmbeddingAggregateModes.ONSET_MEAN
        )

        embeddings = latents
        if aggregate_mode == EmbeddingAggregateModes.MEAN:
            embeddings = embeddings.expand(-1, t, -1)
        elif segment_mode:
            # distribute embeddings
            embeddings = embeddings[(torch.arange(b).repeat_interleave(t), segments.view(-1))].view(b, t, -1)

        return embeddings


class MMDLoss(nn.Module):
    def __init__(self, num_samples: int = 256, max_num_latents: int = 4096):
        super().__init__()
        self.num_samples = num_samples
        self.max_num_latents = max_num_latents

    def forward(self, latents, mask=None):
        if mask is not None:
            latents = latents[mask]

        if latents.shape[0] > self.max_num_latents:
            # avoid memory overflow: sample `max_num_latents` and compute loss only for them
            latents = latents[torch.randperm(latents.shape[0])[:self.max_num_latents]]

        z = torch.randn(self.num_samples, latents.shape[-1], device=latents.device, dtype=latents.dtype)
        return self.compute_mmd(z, latents)

    @staticmethod
    def gaussian_kernel(x, y):
        x_core = x.unsqueeze(1).expand(-1, y.size(0), -1)  # (x_dim, y_dim, dim)
        y_core = y.unsqueeze(0).expand(x.size(0), -1, -1)  # (x_dim, y_dim, dim)
        numerator = (x_core - y_core).pow(2).mean(2) / x.size(-1)  # (x_dim, y_dim)
        return torch.exp(-numerator)

    @staticmethod
    def compute_mmd(x, y):
        x_kernel = MMDLoss.gaussian_kernel(x, x)
        y_kernel = MMDLoss.gaussian_kernel(y, y)
        xy_kernel = MMDLoss.gaussian_kernel(x, y)
        return x_kernel.mean() + y_kernel.mean() - 2 * xy_kernel.mean()


def dropout_latent_mask(mask, dropout):
    bs, ts = torch.where(mask)
    drop_ids = torch.rand(bs.shape[0]) < dropout
    drop_mask = torch.zeros_like(mask, dtype=torch.bool)
    drop_mask[bs[drop_ids], ts[drop_ids]] = True
    return drop_mask[..., None]
