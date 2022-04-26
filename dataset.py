from torch.utils.data import Dataset
import h5py
import json
import numpy as np

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