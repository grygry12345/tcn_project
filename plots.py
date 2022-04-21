"""
Created by Arman Savran at 2022-04-16
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm as mpcm


def plot_est_im(eventspiketensor, frame_size_ms, idx):
    im = eventspiketensor[idx].sum(axis=0)
    lim_intensity = frame_size_ms * 5e-2
    fig, ax = plt.subplots()
    ax.imshow(im, clim=(0, lim_intensity), origin='lower', interpolation='nearest', cmap=mpcm.Blues)
    return fig, ax


def plot_timeseq(eventspiketensor, frame_size_ms, segment, name):
    event_activation = eventspiketensor.sum(axis=(1, 2, 3))
    step = frame_size_ms * 1e-3  # Convert to second
    t_offset = step / 2
    t_frames = t_offset + step * np.arange(event_activation.shape[0])
    fig, ax = plt.subplots(figsize=(10, 2))
    ax.plot(t_frames, event_activation)
    if segment is not None and (segment[0] >= 0 and segment[1] >= 0):
        # print(segment)
        ax.axvline(segment[0], c='k')
        ax.axvline(segment[1], c='k')
    ax.set_xlim((t_frames[0], t_frames[-1]))
    max_y = event_activation.max()
    ax.set_ylim((0, max_y))
    return fig, ax

