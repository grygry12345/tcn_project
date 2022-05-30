import torch
from torch.utils.data import Dataset
import h5py
import json
import numpy as np


class HDF5Dataset(Dataset):
    def __init__(self, file_path='data', data_path="data/roi_mouth", transform=None, group='train', \
                 frame_count: int = 3, step_size: int = 1, device='cpu'):
        self.file_path = file_path
        self.transform = transform
        self.group = group
        self.data_path = data_path
        self.frame_count = frame_count
        self.step_size = step_size
        self.device = device
        self.labels = None
        self.data = None
        self._frame_lengths = {}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample, label

    def _concatenate_frames(self, sets, group):
        i = 0
        file_list = sets[group].keys()
        frame_lengths = []
        for file_name in file_list:
            with h5py.File(self.data_path + '/' + file_name + '.h5', 'r') as f:
                d = f['data']
                d = np.array(d)
                d = torch.tensor(d, dtype=torch.float32, device=self.device)

                d = d.squeeze()
                d_len = d.shape[0]

                frames = []
                for frame in range(0, d_len, self.step_size):

                    if frame + self.frame_count > d_len:
                        break
                    else:
                        seq = d[frame:frame + self.frame_count]

                        if self.data is None:
                            self.data = seq.clone()
                            frames.append(seq)
                        else:
                            frames.append(seq)
                self.data = torch.stack(frames)
            self._frame_lengths[file_name] = d_len
            frame_lengths.append(d_len)

            # progress bar
            i += 1
            print(f'Creating {group} data {i}/{len(sets[group])}', end='\r')
        print()
        print(f'The number of {group} labels {len(frames)}', end='\r')
        print()
        print(f'The average time of {group} frame {np.mean(frame_lengths)}', end='\r')
        print()

    def _concatenate_labels(self, sets, group):
        i = 0
        file_list = sets[group].keys()
        no_speak = []
        frame_lengths = []
        for file_name in file_list:
            segments = sets[group][file_name]
            segments = np.array(segments)
            segments = torch.tensor(segments, dtype=torch.float32, device=self.device)
            frame_length = self._frame_lengths[file_name]
            frame_lengths.append(frame_length)

            if segments[0] == -1 and segments[1] == -1:
                l = torch.zeros(frame_length, dtype=torch.float32, device=self.device)
                no_speak.append(file_name)
            else:
                l = torch.zeros(frame_length, dtype=torch.float32, device=self.device)
                start = int(segments[0] * 100)
                end = int(segments[1] * 100)
                l[start:end] = 1

            labels = []
            for frame in range(0, frame_length, self.step_size):
                if frame + self.frame_count > l.shape[0]:
                    break
                else:
                    seq = l[frame:frame + self.frame_count]

                    if self.labels is None:
                        self.labels = seq.clone()
                        labels.append(seq)
                    else:
                        labels.append(seq)
            self.labels = torch.stack(labels)

            # progress bar
            i += 1
            print(f'Creating {group} labels {i}/{len(sets[group])}', end='\r')
        print()
        print(f'The number of {group} labels {len(labels)}', end='\r')
        print()
        print(f'The average spoken time of {group} labels {np.mean(frame_lengths)}', end='\r')
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
        
    
                       
    def load_variables(self, file_path, file_name_data, file_name_label):
        self.data = torch.load(file_path + file_name_data + '.pt')
        self.labels = torch.load(file_path + file_name_label + '.pt')
    # save data and labels to file
    def save_varibles(self, file_path, file_name_data, file_name_label):
        torch.save(self.data, file_path + file_name_data + '.pt')
        torch.save(self.labels, file_path + file_name_label + '.pt')


