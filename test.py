import h5py
import numpy as np
import matplotlib.pyplot as plt
import json
import torch
from torch.utils.data import DataLoader
import plots
from torch.utils.data import DataLoader, Dataset

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
    


class HDF5Dataset(Dataset):
    def __init__(self, file_path='data', transform=None, group='train', file_name='avds000-lab010-01'):
        self.file_path = file_path
        self.file_name = file_name
        self.transform = transform
        self.group = group
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
            self.data = self.data.reshape(self.data.shape[0], self.data.shape[1] * self.data.shape[2] * self.data.shape[3])



    def __load_labels(self):
        label_file_path = self.file_path + '/sets.json'
        # labels
        with open(label_file_path, 'r') as f:
            segments = json.load(f)
            if self.group == 'train':
                segments = segments['train'][self.file_name]
            elif self.group == 'val':
                segments = segments['validation'][self.file_name]
            elif self.group == 'test':
                segments = segments['test'][self.file_name]
            else:
                raise Exception('Invalid mode')
            
            if (segments[0] != -1 and segments[1] != -1):
                self.labels = np.zeros(len(self.data))
                start = int(segments[0] * 100)
                end = int(segments[1] * 100)
                self.labels[start:end] = 1
            else:
                self.labels = np.zeros(len(self.data))    

    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample, label

        
    
def plot_axample():
    with h5py.File('data/roi_mouth/avds000-lab010-01.h5', 'r') as f:
        train_x = f['data']
        train_x = np.array(train_x)
    
    print(train_x.shape)
    plots.plot_timeseq(train_x, 10, [1.1624487967229904, 3.7137179979518695], 'test')
    plt.show()

from torch.utils.data import DataLoader

import model as m  
import torch.nn as nn

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 4
    lr = 1e-3
    epochs = 1000

    training_data = HDF5Dataset(file_path='data', group='train', file_name='avds000-lab010-01')
    training_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    frame_number = training_loader.dataset.data.shape[0]
    frame_features = training_loader.dataset.data.shape[1]

    model = m.DilatedResidualLayer(channels=4, output=1, dim=frame_features).to(device)
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):   
        print(f"Epoch {epoch+1}\n-------------------------------")
        size = len(training_loader.dataset)
        model.train()
        for batch, (X, y) in enumerate(training_loader):
            X, y= X.to(device), y.to(device)

            pred = model(X)
            loss = loss_fn(pred.squeeze(), y.squeeze())
            

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 10 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"\rBatch {batch+1}/{len(training_loader)} \
                | Loss: {loss:.4f} | {current*100/size:.2f}%", end="")
