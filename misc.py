"""
Created by Arman Savran at 2022-04-16
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm as mpcm
import h5py


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

def plot_axample():
    with h5py.File('data/roi_mouth/avds000-lab010-01.h5', 'r') as f:
        train_x = f['data']
        train_x = np.array(train_x)
    
    fig, _ = plot_timeseq(train_x, 10, [1.1624487967229904, 3.7137179979518695], 'test')
    plt.show()
    # save figure
    fig.savefig('test.png')

if __name__ == '__main__':
    plot_axample()

