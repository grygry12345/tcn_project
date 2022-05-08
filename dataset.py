from torch.utils.data import Dataset
import h5py
import json
import numpy as np

# from eray.project import t_frames

class HDF5Dataset(Dataset):
    def __init__(self, file_path='data', data_path="data/roi_mouth", transform=None, group='train'):
        self.file_path = file_path
        self.transform = transform
        self.group = group
        self.data = None
        self.labels = None
        self.data_path = data_path
        self._frame_lengths = {} # needed for label concatenation


    def _concatenate_frames(self, sets, group):
        i = 0
        for file_name in sets[group].keys():
            
            with h5py.File(self.data_path + '/' + file_name + '.h5', 'r') as f:
                d = f['data']
                d = np.array(d)

                # "file_name" : frame_length
                self._frame_lengths[file_name] = d.shape[0]
                # append d to self.data
                if self.data is None:
                    self.data = d
                else:
                    self.data = np.concatenate((self.data, d))
                    
            # progress bar
            i += 1
            print('\r', end='')
            print('Creating data {}/{}'.format(i, len(sets[group])), end='')
            print('\r', end='')

    def _concatenate_labels(self, sets, group):
        i = 0
        for file_name in sets[group].keys():
            segments = sets[group][file_name]
            frame_length = self._frame_lengths[file_name]

            if (segments[0] != -1 and segments[1] != -1):
                l = np.zeros(frame_length)
                start = int(segments[0] * 100)
                end = int(segments[1] * 100)
                l[start:end] = 1
            else:
                l = np.zeros(frame_length)
            
            # append l to self.labels
            if self.labels is None:
                self.labels = l
            else:
                self.labels = np.concatenate((self.labels, l))
            
            # progress bar
            i += 1
            print('\r', end='')
            print('Creating labels {}/{}'.format(i, len(sets[group])), end='')
            print('\r', end='')

    def create_data(self):
        label_file_path = self.file_path + '/sets.json'
        
        # assign file_names in folder using sets.json
        with open(label_file_path, 'r') as l:
            sets = json.load(l)
            if self.group == 'train':
                self._concatenate_frames(sets, 'train')
            elif self.group == 'val':
                self._concatenate_frames(sets, 'validation')
            elif self.group == 'test':
                self._concatenate_frames(sets, 'test')
            else:
                raise Exception('Invalid group')
               
            # reshape after concanating
            self.data = self.data.reshape(self.data.shape[0], -1)

    def create_one_data(self, file_name):
        with h5py.File(self.data_path + '/' + file_name + '.h5', 'r') as f:
            self.data = np.array(f['data'])
            

    def create_labels(self, frame_size_ms=10): # ? Maybe add frame_size_ms for the labels
        label_file_path = self.file_path + '/sets.json'

        # labels
        with open(label_file_path, 'r') as l:
            segments = json.load(l)
            if self.group == 'train':
                self._concatenate_labels(segments, 'train')
            elif self.group == 'val':
                self._concatenate_labels(segments, 'validation')
            elif self.group == 'test':
                self._concatenate_labels(segments, 'test')
            else:
                raise Exception('Invalid group')  

    # save data and labels to file
    def save_varibles(self, file_path):
        np.save('data/var/data.npy', self.data)
        np.save('data/var/labels.npy', self.labels)

    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample, label
