import numpy as np
import h5py
import plots
import matplotlib.pyplot as plt
import json

def get_relative_path(data, directory):
    keys = list(data.keys())
    for i in range(len(data)):
        keys[i] = directory + '/' + keys[i] + '.h5'
    return keys

def combine_dataset():
    directory = 'data/roi_mouth'
    height = 48
    width = 64
    channels = 2
    total_frames = 0

    # get sets json file
    with open('data/sets.json') as f:
        sets = json.load(f)

    train_list = sets['train']
    val_list = sets['validation']
    test_list = sets['test']
    
    # get train data lists in sets
    train_list_paths = get_relative_path(train_list, directory)
    val_list_paths = get_relative_path(val_list, directory)
    test_list_paths = get_relative_path(test_list, directory)
    
    train_y = train_list.values()
    val_y = val_list.values()
    test_y = test_list.values()


    train_x = np.zeros((total_frames, channels, height, width))
    for i in range(len(train_list_paths)):
        with h5py.File(train_list_paths[i], 'r') as f:
            data = f['data']
            # append frames to train_x array
            train_x = np.append(train_x, data, axis=0)
            print("iteration: " + str(i) + "/" + str(len(train_list_paths)))
    
    print('train_x shape: ', train_x.shape)


def plot_axample():
    with h5py.File('data/roi_mouth/avds000-lab010-01.h5', 'r') as f:
        train_x = f['data']
        train_x = np.array(train_x)
    
    print(train_x.shape)
    plots.plot_timeseq(train_x, 10, [1.1624487967229904, 3.7137179979518695], 'test')
    plt.show()