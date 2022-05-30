import torch
from torch.utils.data import Dataset
import h5py
import json
import numpy as np
from model import BasicLinear

class HDF5Dataset(Dataset):
    def __init__(self, file_path='data', data_path="data/roi_mouth", transform=None, group='train', \
    frame_count: int = 3, step_size: int = 1, device = 'cpu', 
    model_feature = None, cnn_hidden_size: int = 4, use_baseline = True):
        self.file_path = file_path
        self.transform = transform
        self.group = group
        self.data_path = data_path
        self.frame_count = frame_count
        self.step_size = step_size
        self.device = device
        self.labels = np.array([])
        self.data = None
        self._frame_lengths = {}
        
        self.cnn_hidden_size = cnn_hidden_size
        self.model_feature = model_feature
        self.use_baseline = use_baseline

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx] 
        label = self.labels[idx]
        return sample, label

    def _concatenate_frames(self, sets, group):
        i = 0
        file_list = sets[group].keys()
        
        for file_name in file_list:
            with h5py.File(self.data_path + '/' + file_name + '.h5', 'r') as f:
                d = f['data']
                d = np.array(d)

                d_len = d.shape[0]


                for frame in range(0, d_len, self.step_size):

                    if frame + self.frame_count > d_len:
                        break
                    else:
                        seq = d[frame:frame+self.frame_count]

                        seq = torch.from_numpy(seq).to(self.device)
                        seq = self._features(self.model_feature, seq)
                        seq = seq.detach().cpu().numpy()
                        
                        seq = np.expand_dims(seq, axis=0)


                        if self.data is None:
                            self.data = seq.copy()
                        else:
                            self.data = np.concatenate((self.data, seq), axis=0)
            self._frame_lengths[file_name] = d_len
            
            # progress bar
            i += 1
            print (f'Creating {group} data {i}/{len(sets[group])}', end='\r')

        print()

    def _features(self, model, seq):
        features = model(seq)
        return features

    def _concatenate_labels(self, sets, group):
        i = 0
        file_list = sets[group].keys()
        for file_name in file_list:
            segments = sets[group][file_name]
            segments = np.array(segments)
            frame_length = self._frame_lengths[file_name]


            if segments[0] == -1 and segments[1] == -1:
                frame_label = np.zeros(frame_length)
            else:
                frame_label = np.zeros(frame_length)
                start = int(segments[0] * 100)
                end = int(segments[1] * 100)
                frame_label[start:end] = 1
            
            for frame in range(0, frame_length, self.step_size):
                if frame + self.frame_count > frame_label.shape[0]:
                    break
                else:
                    seq = frame_label[frame:frame+self.frame_count]

                    # mean of the sequence
                    seq = np.mean(seq)
                    seq = np.round(seq)

                    # Append to labels
                    self.labels = np.append(self.labels, seq)
                
            # progress bar
            i += 1
            print (f'Creating {group} labels {i}/{len(sets[group])}', end='\r')
        print()

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
               
    
    def create_labels(self):
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