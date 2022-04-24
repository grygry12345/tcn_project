"""
Created by Arman Savran at 2022-04-16
"""

from statistics import mode
import h5py
import numpy as np
import matplotlib.pyplot as plt
import json
import glob
import torch
from torch.utils.data import DataLoader, TensorDataset
import plots

def get_relative_path(data, directory):
    keys = list(data.keys())
    for i in range(len(data)):
        keys[i] = directory + '/' + keys[i] + '.h5'
    return keys

def test():
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

# if __name__ == '__main__':
    

from torch.utils.data import DataLoader, Dataset

class HDF5Dataset(Dataset):
    def __init__(self, file_path='data', transform=None, group='train', file_name='avds000-lab010-01'):
        self.file_path = file_path
        self.file_name = file_name
        self.transform = transform
        self.mode = group
        self.data = None
        self.labels = None
        self.__load_data()
        self.__load_labels()

    def __load_data(self):
        data_file_path = self.file_path + '/roi_mouth/' + self.file_name + '.h5'

        # data
        with h5py.File(data_file_path, 'r') as f:
            self.data = f['data']
            self.data = np.array(self.data)

    def __load_labels(self):
        label_file_path = self.file_path + '/sets.json'
        # labels
        with open(label_file_path, 'r') as f:
            segment = json.load(f)
            if self.mode == 'train':
                segment = self.labels['train'][self.file_name]
            elif self.mode == 'val':
                segment = self.labels['validation'][self.file_name]
            elif self.mode == 'test':
                segment = self.labels['test'][self.file_name]

        
    
    
def plot_axample():
    with h5py.File('data/roi_mouth/avds000-lab010-01.h5', 'r') as f:
        train_x = f['data']
        train_x = np.array(train_x)
    
    print(train_x.shape)
    plots.plot_timeseq(train_x, 10, [1.1624487967229904, 3.7137179979518695], 'test')
    plt.show()
            
if __name__ == '__main__':
    plot_axample()
