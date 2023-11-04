""" Token sequence generators. """

from dataclasses import dataclass
from typing import Optional, Union, Callable, Dict

import numpy as np
import torch
from miditok.constants import TEMPO
from torch import Tensor

from scoreperformer.data.collators import MixedLMScorePerformanceCollator
from scoreperformer.data.datasets import ScorePerformanceDataset, ScorePerformanceSampleMeta
from scoreperformer.data.helpers import TokenSequenceAugmentations
from scoreperformer.data.tokenizers import SPMuple2
from scoreperformer.data.tokenizers.constants import SOS_TOKEN, EOS_TOKEN
from scoreperformer.models.scoreperformer import ScorePerformer, TupleTransformerCaches
from scoreperformer.modules.sampling import top_k
from scoreperformer.modules.transformer import AttentionIntermediates, TransformerIntermediates
from scoreperformer.utils import find_closest
from .messengers import SPMupleMessenger, IntermediateData, SPMuple2IntermediateData


@dataclass
class PerformanceData:
    perf_seq: Optional[np.ndarray] = None
    notes: Optional[Tensor] = None
    embeddings: Optional[Tensor] = None
    context: Optional[Tensor] = None
    gen_seq: Optional[Tensor] = None
    intermediates: Optional[IntermediateData] = None
    caches: Optional[TupleTransformerCaches] = None
    reached_eos: bool = False


class ScorePerformerGenerator:
    def __init__(
            self,
            model: ScorePerformer,
            dataset: ScorePerformanceDataset,
            collator: MixedLMScorePerformanceCollator,
            messenger: SPMupleMessenger,
            device: Optional[Union[str, torch.device]] = None
    ):
        self.model = model
        assert model.perf_decoder is not None

        self.dataset = dataset
        self.tokenizer = dataset.tokenizer
        self.collator = collator

        self.sos_token_id = self.tokenizer[0, SOS_TOKEN]
        self.eos_token_id = self.tokenizer[0, EOS_TOKEN]

        self.messenger = messenger

        self.device = device
        self._init_variables()
        self.perf_data = PerformanceData()

    def _init_variables(self):
        num_dims = len(self.tokenizer.sizes)
        mask_dims = set(range(num_dims)).difference(self.collator.mask_ignore_token_dims)
        self.mask_dims = torch.tensor(list(mask_dims))

    def reset(self):
        self.perf_data = PerformanceData()

    def prepare_performance_notes(
            self,
            perf_idx: int,
            score_embeddings: Optional[Tensor] = None,
            perf_embeddings: Optional[Tensor] = None,
            overlay_bars: float = 0.5
    ):
        # get performance sequence (and notes) from dataset
        perf_seq = self.dataset.performances[perf_idx]
        self.perf_data.perf_seq = perf_seq

        initial_tempo = TEMPO
        if isinstance(self.tokenizer, SPMuple2) and hasattr(self.dataset, 'initial_tempos'):
            initial_tempo = self.dataset.initial_tempos[self.dataset.performance_names[perf_idx]]

        # add SOS/EOS tokens
        perf_seq = self.dataset.processor.add_sos_token(perf_seq)
        perf_seq = self.dataset.processor.add_eos_token(perf_seq)

        # compute performance embeddings if not provided
        compute_embeddings = self.model.perf_encoder is not None and perf_embeddings is None
        compute_embeddings = compute_embeddings or (self.model.score_encoder is not None and score_embeddings is None)
        if compute_embeddings:
            score_embeddings, perf_embeddings, _ = self.encode_embeddings(perf_idx, overlay_bars=overlay_bars)

        # prepare sequence and move to device
        perf_notes = torch.from_numpy(perf_seq).to(self.device)
        perf_notes[1:-1, self.mask_dims] = self.collator.mask_token_id

        self.perf_data.notes = perf_notes
        self.perf_data.embeddings = perf_embeddings
        self.perf_data.context = score_embeddings

        if isinstance(self.tokenizer, SPMuple2):
            self.perf_data.intermediates = SPMuple2IntermediateData(initial_tempo=initial_tempo)

        return self.perf_data

    def generate_performance_notes(
            self,
            start_time: float = 0.,
            time_window: float = 0.2,
            time_window_overflow: float = 0.1,
            delta_embedding: Optional[Tensor] = None,
            max_context_len: int = 512,
            group_chord_notes: bool = True,
            time_messages: bool = True,
            sort_messages: bool = False,
            filter_logits_fn: Callable = top_k,
            filter_kwargs: Optional[Dict[str, object]] = None,
            disable_tqdm: bool = True,
            disable_caches: bool = False
    ):
        perf_notes = self.perf_data.notes
        perf_seq = self.perf_data.gen_seq
        has_perf_emb = self.perf_data.embeddings is not None
        has_score_emb = self.perf_data.context is not None
        perf_embeddings = self.perf_data.embeddings.clone().detach() if has_perf_emb else None
        score_embeddings = self.perf_data.context.clone().detach() if has_score_emb else None

        # prepare position counters
        if perf_seq is None:
            perf_seq = perf_notes[:1]
            self.perf_data.gen_seq = perf_seq

        current_note_idx = perf_seq.shape[0]

        start_idx = 0
        if current_note_idx >= max_context_len - 1:
            next_bar_idx = torch.where(torch.diff(perf_seq[1:, 0]))[0]
            if len(next_bar_idx) > 0:
                fits_context = torch.where(current_note_idx - (next_bar_idx + 1) < max_context_len)[0]
                start_idx = 0 if len(fits_context) == 0 else next_bar_idx[fits_context[0]] + 2

        # take known sequence as context
        input_seq = perf_seq[start_idx:].clone().detach()
        known_input_len = input_seq.shape[0]  # cut by `max_context_len`

        # process sos
        has_sos = input_seq[0, 0] == self.sos_token_id
        first_note_idx = int(has_sos)

        # move delta embedding to device if present
        delta_embedding = None if delta_embedding is None else delta_embedding.to(self.device)

        # add notes one by one, predict and check the timing, break if went behind window
        gen_seq = None
        caches, intermediates = self.perf_data.caches, self.perf_data.intermediates
        all_token_times, all_gen_tokens = [], []
        while not self.perf_data.reached_eos:
            # add notes to predict
            if group_chord_notes:
                end = current_note_idx + 1
                while end < len(perf_notes) and torch.all(perf_notes[current_note_idx, :2] == perf_notes[end, :2]):
                    end += 1
                new_notes = perf_notes[current_note_idx:end]
            else:
                new_notes = perf_notes[current_note_idx:current_note_idx + 1]
            num_new_notes = new_notes.shape[0]

            if isinstance(self.tokenizer, SPMuple2) and not self.tokenizer.vocab_types_idx['Tempo'] in self.mask_dims:
                # update tempo tokens with current tempos only if tempos are not predicted
                tempo = intermediates.tempos[-1, 0] if intermediates.tempos is not None else intermediates.initial_tempo
                tempo_token = find_closest(self.tokenizer.tempos, tempo) + self.tokenizer.zero_token
                new_notes[:, self.tokenizer.vocab_types_idx['Tempo']] = tempo_token

            has_eos = new_notes[-1, 0] == self.eos_token_id
            if has_eos:
                self.perf_data.reached_eos = True
                break

            input_seq = torch.cat([input_seq, new_notes], dim=0)
            last_note_idx = input_seq.shape[0] - 1 if has_eos else input_seq.shape[0]

            # cut input sequence if exceeds `max_context_len`
            input_len = input_seq.shape[0]
            if input_len >= max_context_len:
                next_bar_idx = torch.where(torch.diff(input_seq[first_note_idx:last_note_idx, 0]))[0]
                shift = 1
                if len(next_bar_idx) > 0:
                    fits_context = torch.where(input_len - (next_bar_idx + first_note_idx) < max_context_len)[0]
                    if len(fits_context) > 0 and next_bar_idx[fits_context[0]] + 1 + first_note_idx != input_len - 1:
                        shift = next_bar_idx[fits_context[0]] + 1 + first_note_idx

                input_seq = input_seq[shift:]
                known_input_len -= shift
                last_note_idx -= shift
                start_idx += shift
                has_sos, first_note_idx = False, 0
                caches = None

                if known_input_len < max_context_len / 8:
                    break  # more notes in `time_window` than `max_context_len` can handle, leave  # TODO

            # shift bars to zero before computation
            bar_shift_to_zero = input_seq[first_note_idx, 0] - self.tokenizer.zero_token
            input_seq[first_note_idx:last_note_idx, 0] -= bar_shift_to_zero

            # add masked sequence
            input_seq_doubled = input_seq.clone().detach()
            input_seq_doubled[first_note_idx:last_note_idx, self.mask_dims] = self.collator.mask_token_id

            # add delta embedding if present
            if has_perf_emb and delta_embedding is not None:
                perf_embeddings[current_note_idx:current_note_idx + num_new_notes] += delta_embedding

            # get score and performance embeddings
            score_embs = None
            if has_score_emb:
                score_embs = score_embeddings[start_idx:current_note_idx + num_new_notes].unsqueeze(0)
            perf_embs = None
            if has_perf_emb:
                perf_embs = perf_embeddings[start_idx:current_note_idx + num_new_notes].unsqueeze(0)

            if caches is not None:
                if input_seq.shape[0] - 1 - num_new_notes != caches.token_emb.shape[1] \
                        or caches.token_emb.shape[1] == 0 \
                        or len(caches.transformer.attention) == 0:
                    caches = None

            # generate notes
            with torch.inference_mode():
                gen_seq, caches = self.model.perf_decoder.unmask_tokens(
                    input_seq,
                    input_seq_doubled,
                    context=score_embs,
                    style_embeddings=perf_embs,
                    caches=caches if not disable_caches else None,
                    return_caches=True,
                    filter_logits_fn=filter_logits_fn,
                    filter_kwargs=filter_kwargs,
                    disable_tqdm=disable_tqdm
                )
                input_seq[first_note_idx:last_note_idx, 0] += bar_shift_to_zero
                gen_seq = gen_seq[known_input_len:last_note_idx]
                gen_seq[:, 0] += bar_shift_to_zero

            # get token time and stop if needed
            gen_tokens = gen_seq[-num_new_notes:].cpu().numpy()
            # gen_tokens = self.perf_data.perf_seq[current_note_idx - 1:current_note_idx - 1 + num_new_notes]
            token_times, intermediates = self.messenger.tokens_to_messages(
                gen_tokens, note_attributes=False, note_off_events=False,
                intermediates=intermediates, return_intermediates=True, sort=False
            )

            all_token_times.extend(token_times.tolist())
            all_gen_tokens.append(gen_tokens)

            if token_times.max() >= start_time + time_window + time_window_overflow:
                break

            # add generated notes
            input_seq[-num_new_notes:] = gen_seq[-num_new_notes:]
            current_note_idx += num_new_notes

        if gen_seq is None:
            return gen_seq, []

        # cut notes fitting `time_window`
        cut_idx = np.where(np.array(all_token_times) <= start_time + time_window)[0]
        cut_idx = 0 if len(cut_idx) == 0 else cut_idx[-1] + 1

        if cut_idx == 0:
            return None, []

        # compute new messages
        gen_tokens = np.concatenate(all_gen_tokens, axis=0)[:cut_idx]
        messages, self.perf_data.intermediates = self.messenger.tokens_to_messages(
            gen_tokens, intermediates=self.perf_data.intermediates, return_intermediates=True,
            to_times=time_messages, sort=sort_messages
        )

        # update performance embeddings for the generated notes
        if has_perf_emb and delta_embedding is not None:
            total_len = self.perf_data.gen_seq.shape[0]
            self.perf_data.embeddings[total_len:total_len + cut_idx] = perf_embeddings[total_len:total_len + cut_idx]

        # update total generated sequence
        gen_seq = torch.from_numpy(gen_tokens).to(device=gen_seq.device)
        self.perf_data.gen_seq = torch.cat([self.perf_data.gen_seq, gen_seq])

        # update caches
        if caches is not None:
            cut_len = caches.token_emb.shape[1] - (len(all_token_times) - cut_idx)
            caches = self.cut_caches(caches, right_idx=cut_len)
        self.perf_data.caches = caches

        return gen_seq, messages

    def predict_number_of_notes(
            self,
            start_time: float = 0.,
            time_window: float = 0.2,
            max_notes: int = 32,
    ):
        num_gen_notes = len(self.perf_data.gen_seq) - 1 if self.perf_data.gen_seq is not None else 0
        future_notes = self.perf_data.perf_seq[num_gen_notes:num_gen_notes + max_notes]
        if len(future_notes) == 0:
            return 0.

        if self.perf_data.intermediates is not None:  # adjust tempos
            tempo_index = self.tokenizer.vocab_types_idx['Tempo']
            tempo_token = self.tokenizer[tempo_index, f'Tempo_{int(self.perf_data.intermediates.tempos[-1, 0])}']
            shift = tempo_token - self.perf_data.perf_seq[num_gen_notes - 1, tempo_index]
            future_notes[:, tempo_index] += shift

        times = self.messenger.tokens_to_messages(
            future_notes, note_attributes=False, note_off_events=False,
            intermediates=self.perf_data.intermediates, sort=False
        )
        return (times <= start_time + time_window).sum()

    def encode_embeddings(
            self,
            perf_idx: int,
            compute_latents: bool = False,
            overlay_bars: float = 0.,
            augmentations: Optional[TokenSequenceAugmentations] = None
    ):
        # get score sequence and its data
        perf = self.dataset.performance_names[perf_idx]
        score, _ = self.dataset._performance_map[perf]
        score_idx = self.dataset.scores._name_to_idx[score]
        score_indices = self.dataset._score_indices[score_idx]
        if score_indices is None:
            score_indices = self.dataset.indexer.compute_bar_indices(self.dataset.scores[score_idx])
            self.dataset._score_indices[score_idx] = score_indices

        # get initial meta and sample
        from scoreperformer.data.datasets.utils import get_end_bar
        start_bar = 0
        end_bar = get_end_bar(score_indices, start_bar, self.dataset.max_seq_len, self.dataset.max_bar)
        meta = ScorePerformanceSampleMeta(
            idx=None, score_idx=score_idx, perf_idx=perf_idx,
            start_bar=start_bar, end_bar=end_bar,
            augmentations=augmentations,
        )
        sample = self.dataset.get(meta=meta)

        # get current last bar and total number of bars
        bar_idx = self.tokenizer.vocab_types_idx['Bar']
        _bar_0 = self.tokenizer.zero_token
        score_seq = self.dataset.scores[score_idx]
        has_sos = sample.score[0, 0] == self.sos_token_id
        has_eos = sample.score[-1, 0] == self.eos_token_id
        first_note_idx, last_note_idx = int(has_sos), sample.score.shape[0] - int(has_eos)
        last_perf_note_idx = sample.perf.shape[0] - int(has_eos)
        last_bar = sample.score[-1 - int(has_eos), bar_idx] - _bar_0
        total_bars = score_seq[-1, bar_idx] - _bar_0

        emb_start_bar = start_bar
        score_embeddings, perf_embeddings = [], []
        while last_bar <= total_bars:
            _inputs = self.collator((sample,))
            inputs = self.model.allocate_inputs(self.model.prepare_inputs(_inputs), self.device)

            # move bars to zero
            bar_shift_to_zero = inputs['score'][:, first_note_idx, bar_idx] - _bar_0
            inputs['score'][:, first_note_idx:last_note_idx, bar_idx] -= bar_shift_to_zero
            inputs['perf'][:, first_note_idx:last_perf_note_idx, bar_idx] -= bar_shift_to_zero

            with torch.inference_mode():
                # get encoder embeddings
                enc_out = self.model.forward_encoders(
                    score=inputs['score'], score_mask=inputs['score_mask'],
                    perf=inputs['perf'], perf_mask=inputs['perf_mask'],
                    bars=inputs['bars'], beats=inputs['beats'], onsets=inputs['onsets'],
                    deadpan_mask=inputs['deadpan_mask'],
                    compute_loss=False
                )

            # append new note embeddings
            note_cut_idx = 0
            if overlay_bars:
                note_cut_idx = np.where(sample.score[:, bar_idx] - _bar_0 >= emb_start_bar)[0][0] - first_note_idx

            if enc_out.score_embeddings is not None:
                score_embeddings.append(enc_out.score_embeddings[0, note_cut_idx:])
            if enc_out.perf_embeddings is not None:
                perf_embeddings.append(enc_out.perf_embeddings[0, note_cut_idx:])

            if has_eos:
                break

            # move to the next bars
            if overlay_bars:
                start_bar = sample.score[int(sample.score.shape[0] * (1 - overlay_bars)), 0] - _bar_0
                emb_start_bar = end_bar + 1
            else:
                emb_start_bar = start_bar = end_bar + 1
            end_bar = get_end_bar(score_indices, start_bar, self.dataset.max_seq_len, self.dataset.max_bar)

            # get next sample
            meta.start_bar, meta.end_bar = start_bar, end_bar
            sample = self.dataset.get(meta=meta)

            # process EOS, get new last bar
            has_sos = sample.score[0, 0] == self.sos_token_id
            has_eos = sample.score[-1, 0] == self.eos_token_id
            first_note_idx, last_note_idx = int(has_sos), sample.score.shape[0] - int(has_eos)
            last_perf_note_idx = sample.perf.shape[0] - int(has_eos)
            last_bar = sample.score[last_note_idx - 1, bar_idx] - _bar_0

        score_embeddings = torch.cat(score_embeddings, dim=0) if score_embeddings else None
        perf_embeddings = torch.cat(perf_embeddings, dim=0) if perf_embeddings else None

        latents = None
        if perf_embeddings is not None and compute_latents:
            bars, beats = score_seq[:, 0], self.dataset._beat_maps[score_idx]
            onsets = self.dataset._onset_maps[score_idx]
            bars, beats, onsets = map(
                lambda s: torch.from_numpy(np.concatenate([[s[0]], s, [s[-1]]]))[None].to(self.device),
                (bars, beats, onsets)
            )
            latents = self.model.perf_encoder.embeddings_to_latents(
                embeddings=perf_embeddings[None], bars=bars, beats=beats, onsets=onsets
            )

        return score_embeddings, perf_embeddings, latents

    @staticmethod
    def cut_caches(caches, left_idx=0, right_idx=None):
        right_idx = caches.token_emb.shape[-1] if right_idx is None else right_idx
        caches.token_emb = caches.token_emb[:, left_idx:right_idx]
        caches.transformer = TransformerIntermediates(
            hiddens=[
                t[..., left_idx:right_idx, :] for t in caches.transformer.hiddens
            ],
            attention=[
                AttentionIntermediates(
                    *(tuple(map(lambda t: t[..., left_idx:right_idx, :], (inter.keys, inter.values))) + (None,))
                )
                for inter in caches.transformer.attention
            ]
        )
        return caches
