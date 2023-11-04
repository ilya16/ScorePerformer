import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
import pretty_midi

from scoreperformer.data.tokenizers import SPMuple


def plot_performance_parameter(tokenizer: SPMuple, total_seq, perf_seq, token_type="Tempo"):
    type_idx = tokenizer.vocab_types_idx[token_type]

    preds = total_seq[:, type_idx] - tokenizer.zero_token
    targets = perf_seq[:total_seq.shape[0], type_idx] - tokenizer.zero_token

    if token_type == "Velocity":
        values_map = tokenizer.velocities
    elif token_type == "Tempo":
        values_map = tokenizer.tempos
    elif token_type == "OnsetDev":
        nb_positions = max(tokenizer.beat_res.values()) * 2  # up to two quarter notes
        values_map = np.arange(-nb_positions, nb_positions + 1) / nb_positions / 2
    elif token_type == "PerfDuration":
        values_map = np.array([
            (beat * res + pos) / res if res > 0 else 0
            for beat, pos, res in tokenizer.durations
        ])
    elif token_type == "RelOnsetDev":
        values_map = tokenizer.rel_onset_deviations
    elif token_type == "RelPerfDuration":
        values_map = tokenizer.rel_performed_durations
    else:
        return

    preds, targets = values_map[preds], values_map[targets]

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(16, 12))

    fig.suptitle(f"Performance Notes, {token_type}", fontsize=20)
    ax0.plot(preds)
    ax0.plot(targets)
    ax1.plot(preds - targets)

    ax0.legend(["Generated", "Target"], fontsize=18)
    ax1.legend(["Difference"], fontsize=18)

    ax0.get_xaxis().set_visible(False)
    ax1.set_xlabel("note id", fontsize=16)
    for ax in (ax0, ax1):
        ax.tick_params(labelsize=14)
        ax.set_ylabel(token_type.lower(), fontsize=16)

    fig.tight_layout()


_colors = plt.get_cmap('Reds', 256)(np.linspace(0, 1, 256))
_colors[:1, :] = np.array([1, 1, 1, 1])
pianoroll_cmap = ListedColormap(_colors)


def plot_pianoroll(pm, min_pitch=21, max_pitch=109, min_velocity=0., max_velocity=127.,
                   fs=100, max_time=None, pad_time=0.2, xticks_time=1., figsize=(14, 6), fig=None, ax=None):
    if fig is None or ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    # get numpy array from pianoroll
    arr = pm.get_piano_roll(fs)[min_pitch:max_pitch + 1]
    arr[arr > max_velocity] = max_velocity

    # pad with a few steps
    max_time = pm.get_end_time() if max_time is None else max_time
    pad_steps = int(fs * pad_time)
    pad_l, pad_r = pad_steps, pad_steps + max(0, int(fs * max_time) - arr.shape[1])
    arr = np.pad(arr, ((0, 0), (pad_l, pad_r)), 'constant')

    # plot pianoroll
    x_coords = np.arange(-pad_time, arr.shape[1] / fs - pad_time, 1 / fs)
    y_coords = np.arange(min_pitch, max_pitch + 1, 1)
    librosa.display.specshow(
        arr, cmap=pianoroll_cmap, ax=ax, x_coords=x_coords, y_coords=y_coords,
        hop_length=1, sr=fs, x_axis='time', y_axis='linear', bins_per_octave=12,
        fmin=pretty_midi.note_number_to_hz(min_pitch), vmin=min_velocity, vmax=max_velocity
    )

    # plot colorbar
    cbar = fig.colorbar(ax.get_children()[0], ax=ax, fraction=0.15, pad=0.02, aspect=15)
    cbar.ax.tick_params(labelsize=14)
    cbar.set_ticks(np.arange(0, max_velocity, 12))

    # axis labels
    ax.set_xlabel('time (s)', fontsize=16)
    ax.set_ylabel('pitch', fontsize=16)
    ax.tick_params(labelsize=14)

    # x-axis
    ax.set_xticks(np.arange(0, ax.get_xticks()[-1], xticks_time))
    ax.set_xlim(xmax=max_time + pad_time)

    # y-axis
    yticks = np.arange(min_pitch + 12 - min_pitch % 12, max_pitch, 12)
    ax.set_yticks(yticks - 0.5)
    ax.set_yticklabels(yticks)

    # removing empty pitch lines
    has_notes = min_pitch + np.where(np.any(arr != 0., axis=1))[0]
    if len(has_notes) > 0:
        ymin, ymax = has_notes[0], has_notes[-1]
        ymin = max(min_pitch, ymin - ymin % 12) - 2.5
        ymax = min(max_pitch, ymax + 12 - ymax % 12) + 1.5
        ax.set_ylim(ymin, ymax)

    ax.grid(alpha=0.5)

    return fig, ax
