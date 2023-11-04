""" ScorePerformer Wrappers for Language Modeling tasks. """

import warnings
from dataclasses import dataclass
from typing import Optional, Dict, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from tqdm.auto import tqdm

from scoreperformer.data.tokenizers import OctupleM
from scoreperformer.modules.sampling import top_k, filter_logits_and_sample
from scoreperformer.utils import ExplicitEnum, exists
from .transformer import TupleTransformer, TupleTransformerOutput, TupleTransformerCaches


class LMWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.max_seq_len = self.model.max_seq_len

    def forward(self, seq, labels=None, **kwargs):
        ...


@dataclass
class ScorePerformerLMOutput(TupleTransformerOutput):
    loss: Optional[Tensor] = None
    losses: Optional[Dict[str, Tensor]] = None


class ScorePerformerLMWrapper(LMWrapper):
    def __init__(
            self,
            model: TupleTransformer,
            ignore_index: int = -100
    ):
        super().__init__(model=model)
        self.ignore_index = ignore_index

    def forward(self, seq: Tensor, labels: Optional[Tensor] = None, **kwargs):
        out = self.model(seq, **kwargs)

        loss = losses = None
        if exists(labels):
            losses = {
                key: F.cross_entropy(
                    out.logits[key].transpose(1, 2),
                    labels[..., i],
                    ignore_index=self.ignore_index
                )
                for i, (key, logits) in enumerate(out.logits.items())
                if torch.any(labels[..., i] != self.ignore_index)
            }

            loss = sum(losses.values()) / len(losses)

            if exists(out.reg_values) and self.model.token_emb.continuous:
                reg_losses = {}
                for i, key in enumerate(out.logits.keys()):
                    if key not in out.reg_values:
                        continue

                    label_mask = labels[..., i] > 3  # special tokens, no regression

                    predictions = out.reg_values[key][label_mask]
                    targets = F.embedding(
                        labels[..., i][label_mask],
                        self.model.token_emb.embs[key].token_values
                    )

                    reg_losses[f"{key}/l1"] = F.l1_loss(predictions, targets)

                loss += sum(reg_losses.values()) / len(reg_losses)
                losses.update(**reg_losses)

        return ScorePerformerLMOutput(
            loss=loss,
            losses=losses,
            **out.__dict__
        )


class ScorePerformerMLMWrapper(ScorePerformerLMWrapper):
    def __init__(
            self,
            model: TupleTransformer,
            mask_token_id: int = 1,
            num_special_tokens: int = 4,
            ignore_index: int = -100
    ):
        super().__init__(model=model, ignore_index=ignore_index)
        self.mask_token_id = mask_token_id
        self.num_special_tokens = num_special_tokens

    @torch.inference_mode()
    def unmask_tokens(
            self,
            tokens: Tensor,
            single_run: bool = True,
            temperature: float = 1.,
            filter_logits_fn: Callable = top_k,
            filter_kwargs: Optional[Dict[str, object]] = None,
            filter_key_ids: Optional[Dict[str, list]] = None,
            disable_tqdm: bool = False,
            **kwargs
    ):
        assert callable(filter_logits_fn)

        was_training = self.model.training
        if was_training:
            self.model.eval()

        num_dims = len(tokens.shape)
        if num_dims == 2:
            tokens = tokens[None, :]

        out = tokens.clone().detach()
        mask = kwargs.pop('mask', None)

        if mask is None:
            mask = torch.full_like(out[..., 0], True, dtype=torch.bool, device=out.device)

        filter_key_ids = filter_key_ids or dict()
        mask_value = -float("Inf")

        def _unmask(inputs, idx, mask, logits_keys, **kwargs):
            outputs = self(inputs, mask=mask, return_embeddings=True, **kwargs)
            logits = self.model.lm_head(outputs.hidden_state[:, idx - 1], keys=logits_keys)

            samples = []
            for key, logits_i in logits.items():
                logits_i[:, :self.num_special_tokens] = mask_value
                filter_ids = filter_key_ids.get(key, None)
                if filter_ids is not None:
                    logits_i[:, filter_ids] = mask_value
                sample = filter_logits_and_sample(
                    logits_i, filter_logits_fn, filter_kwargs=filter_kwargs, temperature=temperature
                )
                samples.append(sample)

            return torch.cat(samples, dim=-1)[None]

        unmask_mask = out == self.mask_token_id

        if single_run:
            # TODO: support single run sampling
            warnings.warn("`single_run` unmasking with sampling is not yet implemented, using argmax.")

            outputs = self.model(out, mask=mask, **kwargs)
            samples = torch.cat(
                list(map(lambda l: torch.argmax(l, dim=-1, keepdim=True), outputs.logits.values())),
                dim=-1
            )
            out[unmask_mask] = samples[unmask_mask]
        else:
            unmask_ids = torch.where(torch.any(unmask_mask, dim=2))[1]

            pbar = unmask_ids if disable_tqdm else tqdm(unmask_ids, leave=False)
            for idx in pbar:
                # get samples
                type_mask = unmask_mask[:, idx][0]
                logits_keys = torch.where(type_mask)[0].tolist()

                samples = _unmask(
                    out[:, :idx + 1], idx,
                    mask=mask[:, :idx + 1], logits_keys=logits_keys, **kwargs
                )

                # fill input sequence
                out[:, idx, type_mask] = samples

        if num_dims == 2:
            out = out.squeeze(0)

        if was_training:
            self.model.train(was_training)

        return out


class ScorePerformerARWrapper(ScorePerformerLMWrapper):
    def __init__(
            self,
            model: TupleTransformer,
            pad_token_id: int = 0,
            eos_token_id: int = 3,
            num_special_tokens: int = 4,
            ignore_index: int = -100
    ):
        super().__init__(model=model, ignore_index=ignore_index)
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.num_special_tokens = num_special_tokens

    @torch.inference_mode()
    def generate(
            self,
            start_tokens: Tensor,
            seq_len: int,
            max_bar: Optional[int] = None,
            temperature: float = 1.,
            filter_logits_fn: Callable = top_k,
            filter_kwargs: Optional[Dict[str, object]] = None,
            caches: Optional[TupleTransformerCaches] = None,
            return_caches: bool = False,
            tokenizer: OctupleM = None,
            fix_errors: bool = True,
            disable_tqdm: bool = False,
            **kwargs
    ):
        assert callable(filter_logits_fn)

        was_training = self.model.training
        if was_training:
            self.model.eval()

        num_dims = len(start_tokens.shape)
        if num_dims == 2:
            start_tokens = start_tokens[None, :]

        b, t = start_tokens.shape[:2]

        out = start_tokens
        mask = kwargs.pop('mask', None)

        if mask is None:
            mask = torch.full_like(out[..., 0], True, dtype=torch.bool, device=out.device)

        pbar = range(t, seq_len + 1)
        pbar = pbar if disable_tqdm else tqdm(pbar, leave=False)
        for _ in pbar:
            x = out[:, -self.max_seq_len:]
            mask = mask[:, -self.max_seq_len:]

            outputs = self(x, mask=mask, caches=caches, return_embeddings=True, return_caches=True, **kwargs)
            logits = self.model.lm_head(outputs.hidden_state[:, -1])
            caches = outputs.caches

            samples = {}
            for key, logits_i in logits.items():
                do_sample = True
                if fix_errors and exists(tokenizer):
                    if key == 'Bar':
                        last_bar = out[:, -1, tokenizer.vocab_types_idx['Bar']]
                        logits_i[:, 4:last_bar] = -float("Inf")
                    same_bar = samples.get('Bar', -1) == out[:, -1, tokenizer.vocab_types_idx['Bar']]
                    if (key == 'Tempo' and same_bar) or key == 'TimeSig':
                        sample = out[:, -1, tokenizer.vocab_types_idx[key]][None]
                        do_sample = False
                if do_sample:
                    logits_i[:, :2] = -float("Inf")
                    sample = filter_logits_and_sample(
                        logits_i, filter_logits_fn, filter_kwargs=filter_kwargs, temperature=temperature
                    )
                samples[key] = sample
            samples = torch.cat(list(samples.values()), dim=-1)[None]

            out = torch.cat((out, samples), dim=1)
            mask = F.pad(mask, (0, 1), value=True)

            if exists(self.eos_token_id):
                is_eos_tokens = (out[..., -1, 0] == self.eos_token_id)  # eos only in `Bar`
                if is_eos_tokens.any(dim=-1):
                    out[:, -1, 1:] = self.pad_token_id
                    pbar.close()
                    break
            elif exists(max_bar):
                is_max_bar_tokens = (out[..., -1, 0] > max_bar)
                if is_max_bar_tokens.any(dim=-1):
                    out = out[:, :-1, :]
                    pbar.close()
                    break

        out = out[:, t:]

        if num_dims == 2:
            out = out.squeeze(0)

        if was_training:
            self.model.train(was_training)

        if return_caches:
            return out, caches
        return out

    def forward(self, seq: Tensor, labels: Optional[Tensor] = None, **kwargs):
        seq = seq[:, :-1]
        labels = labels[:, 1:] if exists(labels) else None

        context = kwargs.get("context", None)
        if exists(context) and self.model.context_emb_mode == "cat":
            kwargs["context"] = context[:, 1:]

        style_embeddings = kwargs.get("style_embeddings", None)
        if exists(style_embeddings):
            kwargs["style_embeddings"] = style_embeddings[:, 1:]

        mask = kwargs.get('mask', None)
        if exists(mask) and mask.shape[1] == seq.shape[1] + 1:
            mask = mask[:, :-1]
            kwargs['mask'] = mask

        return super().forward(seq, labels=labels, **kwargs)


class ScorePerformerMixedLMWrapper(ScorePerformerLMWrapper):
    def __init__(
            self,
            model: TupleTransformer,
            pad_token_id: int = 0,
            mask_token_id: int = 1,
            num_special_tokens: int = 4,
            ignore_index: int = -100
    ):
        super().__init__(model=model, ignore_index=ignore_index)
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id
        self.num_special_tokens = num_special_tokens

    @torch.inference_mode()
    def unmask_tokens(
            self,
            tokens: Tensor,
            tokens_masked,
            temperature: float = 1.,
            filter_logits_fn: Callable = top_k,
            filter_kwargs: Optional[Dict[str, object]] = None,
            filter_key_ids: Optional[Dict[str, list]] = None,
            caches: Optional[TupleTransformerCaches] = None,
            return_caches: bool = False,
            disable_tqdm: bool = False,
            **kwargs
    ):
        assert callable(filter_logits_fn)

        was_training = self.model.training
        if was_training:
            self.model.eval()

        num_dims = len(tokens.shape)
        if num_dims == 2:
            tokens = tokens[None, :]
            tokens_masked = tokens_masked[None, :]

        out = tokens.clone().detach()
        mask = kwargs.pop('mask', None)

        if mask is None:
            mask = torch.full_like(out[..., 0], True, dtype=torch.bool, device=out.device)

        filter_key_ids = filter_key_ids or dict()
        mask_value = -float("Inf")

        def _unmask(inputs, inputs_masked, idx, mask, logits_keys, **kwargs):
            # shifting is handled in forward
            outputs = self(
                inputs, seq_masked=inputs_masked, mask=mask,
                return_embeddings=True, return_caches=True, **kwargs
            )
            logits = self.model.lm_head(outputs.hidden_state[:, idx - 1], keys=logits_keys)

            samples = []
            for key, logits_i in logits.items():
                logits_i[:, self.pad_token_id] = mask_value
                logits_i[:, self.mask_token_id] = mask_value

                filter_ids = filter_key_ids.get(key, None)
                if filter_ids is not None:
                    logits_i[:, filter_ids] = mask_value

                sample = filter_logits_and_sample(
                    logits_i, filter_logits_fn, filter_kwargs=filter_kwargs, temperature=temperature
                )
                samples.append(sample)

            return torch.cat(samples, dim=-1)[None], outputs.caches

        unmask_mask = out == self.mask_token_id
        unmask_ids = torch.where(torch.any(unmask_mask, dim=2))[1]

        pbar = unmask_ids if disable_tqdm else tqdm(unmask_ids, leave=False)
        for idx in pbar:
            # get samples
            type_mask = unmask_mask[:, idx][0]
            logits_keys = torch.where(type_mask)[0].tolist()

            samples, caches = _unmask(
                out[:, :idx + 1], tokens_masked[:, :idx + 1], idx, mask=mask[:, :idx + 1],
                logits_keys=logits_keys, caches=caches, **kwargs
            )

            # fill input sequence
            out[:, idx, type_mask] = samples

        if num_dims == 2:
            out = out.squeeze(0)

        if was_training:
            self.model.train(was_training)

        if return_caches:
            return out, caches
        return out

    def forward(self, seq: Tensor, labels: Optional[Tensor] = None, **kwargs):
        seq = seq[:, :-1]
        labels = labels[:, 1:] if exists(labels) else None

        seq_masked = kwargs.pop("seq_masked", None)
        if exists(seq_masked):
            seq_masked = seq_masked[:, 1:]

        context = kwargs.get("context", None)
        if exists(context) and self.model.context_emb_mode == "cat":
            kwargs["context"] = context[:, 1:]

        style_embeddings = kwargs.get("style_embeddings", None)
        if exists(style_embeddings):
            kwargs["style_embeddings"] = style_embeddings[:, 1:]

        mask = kwargs.get("mask", None)
        if exists(mask) and mask.shape[1] == seq.shape[1] + 1:
            kwargs["mask"] = mask[:, :-1]

        out = super().forward(seq, labels=labels, x_extra=seq_masked, **kwargs)

        return out


class ScorePerformerLMModes(ExplicitEnum):
    MLM = "mlm"
    CLM = "clm"
    MixedLM = "mixlm"


ScorePerformerLMWrappers = {
    ScorePerformerLMModes.MLM: ScorePerformerMLMWrapper,
    ScorePerformerLMModes.CLM: ScorePerformerARWrapper,
    ScorePerformerLMModes.MixedLM: ScorePerformerMixedLMWrapper
}
