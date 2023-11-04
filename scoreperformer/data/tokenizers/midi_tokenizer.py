""" MIDI encoding base class and methods. """

from abc import ABC

from miditok import MIDITokenizer as _MIDITokenizer
from miditok.constants import TIME_SIGNATURE
from miditok.utils import remove_duplicated_notes, merge_same_program_tracks
from miditoolkit import MidiFile, TimeSignature


class MIDITokenizer(_MIDITokenizer, ABC):
    r"""MIDI tokenizer base class, containing common methods and attributes for all tokenizers.

    See :class:`miditok.MIDITokenizer` for a detailed documentation.
    """

    def preprocess_midi(self, midi: MidiFile):
        r"""Pre-process (in place) a MIDI file to quantize its time and note attributes
        before tokenizing it. Its notes attribute (times, pitches, velocities) will be
        quantized and sorted, duplicated notes removed, as well as tempos. Empty tracks
        (with no note) will be removed from the MIDI object. Notes with pitches outside
        of self.pitch_range will be deleted.

        :param midi: MIDI object to preprocess.
        """
        # Merge instruments of the same program / inst before preprocessing them
        # This allows to avoid potential duplicated notes in some multitrack settings
        if self.config.use_programs and self.one_token_stream:
            merge_same_program_tracks(midi.instruments)

        t = 0
        while t < len(midi.instruments):
            # quantize notes attributes
            self._quantize_notes(midi.instruments[t].notes, midi.ticks_per_beat)
            # sort notes
            midi.instruments[t].notes.sort(key=lambda x: (x.start, x.pitch, x.end))
            # remove possible duplicated notes
            if self.config.additional_params.get("remove_duplicates", False):
                remove_duplicated_notes(midi.instruments[t].notes)
            if len(midi.instruments[t].notes) == 0:
                del midi.instruments[t]
                continue

            # Quantize sustain pedal and pitch bend
            if self.config.use_sustain_pedals:
                self._quantize_sustain_pedals(
                    midi.instruments[t].pedals, midi.ticks_per_beat
                )
            if self.config.use_pitch_bends:
                self._quantize_pitch_bends(
                    midi.instruments[t].pitch_bends, midi.ticks_per_beat
                )
            t += 1

        # Recalculate max_tick is this could have changed after notes quantization
        if len(midi.instruments) > 0:
            midi.max_tick = max(
                [max([note.end for note in track.notes]) for track in midi.instruments]
            )

        if self.config.use_tempos:
            self._quantize_tempos(midi.tempo_changes, midi.ticks_per_beat)

        if len(midi.time_signature_changes) == 0:  # can sometimes happen
            midi.time_signature_changes.append(
                TimeSignature(*TIME_SIGNATURE, 0)
            )  # 4/4 by default in this case
        if self.config.use_time_signatures:
            self._quantize_time_signatures(
                midi.time_signature_changes, midi.ticks_per_beat
            )
