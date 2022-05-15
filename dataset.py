from lib2to3.pytree import Node
from torch.utils.data import Dataset
import h5py
import json
import numpy as np
import pickle
# from eray.project import t_frames

class HDF5Dataset(Dataset):
    def __init__(self, file_path='data', data_path="data/roi_mouth", transform=None, group='train'):
        self.file_path = file_path
        self.transform = transform
        self.group = group
        self.data = None
        self.labels = None
        self.data_path = data_path
        # self._frame_segments = None
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
        total_frame = 0
        firstTime = True
        file_list = sets[group].keys()
        for file_name in file_list:
            with h5py.File(self.data_path + '/' + file_name + '.h5', 'r') as f:
                d = f['data']
                d = np.array(d)
                d = d.reshape(d.shape[0], -1)

                # video_length = d.shape[0]         
                # if (firstTime):
                #     frame_start = total_frame
                #     firstTime = False 
                # else:
                #     frame_start = total_frame + 1

                
                # frame_end = video_length + total_frame
                # total_frame += video_length

                self._frame_lengths[file_name] = d.shape[0]
                    
                if self.data is None:
                    self.data = d
                    # self.data = {0: d}
                    # self._frame_segments = [[frame_start, frame_end]]
                else:
                    self.data = np.concatenate((self.data, d))
                    
                    # data dictinoary append
                    # self.data[i] = d
                    # self._frame_segments = np.append(self._frame_segments, [[frame_start, frame_end]], axis=0)
                    
            # progress bar
            i += 1

            print('Creating data {}/{}'.format(i, len(sets[group])), end='\r')


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
                # start = segments[0]
                # end = segments[1]
                # l = [[start, end]]

            else:
                l = np.zeros(frame_length)
                #l = [[-1, -1]]
            
            # append l to self.labels
            if self.labels is None:
                self.labels = l
            else:
                self.labels = np.concatenate((self.labels, l))
            
            # progress bar
            i += 1

            print('Creating labels {}/{}'.format(i, len(sets[group])), end='\r')


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
            # self.data = self.data.reshape(self.data.shape[0], -1)
    
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
  
    def create_one_data(self, file_name):
        with h5py.File(self.data_path + '/' + file_name + '.h5', 'r') as f:
            self.data = np.array(f['data'])
            self.data = self.data.reshape(self.data.shape[0], -1)
    
    def create_one_labels(self, file_name, group):
        with open(self.file_path + '/sets.json', 'r') as l:
            labels = json.load(l)
            if group == 'train':
                segments = labels['train'][file_name]
            elif group == 'val':
                segments = labels['validation'][file_name]
            elif group == 'test':
                segments = labels['test'][file_name]
            
            
            start = int(segments[0] * 100)
            end = int(segments[1] * 100)
            if (segments[0] != -1 and segments[1] != -1):
                self.labels = np.zeros(self.data.shape[0])
                self.labels[start:end] = 1
            else:
                self.labels = np.zeros(self.data.shape[0])
            
            #if (segments[0] != -1 and segments[1] != -1):
            #     self.labels = [-1, -1]
            # else:
            #     start = segments[0]
            #     end = segments[1]
            #     self.labels = [start, end]
                       
    def load_variables(self, file_path, file_name_data, file_name_label):
        self.data = np.load(file_path + file_name_data + '.npy') # data
        # with open(file_path + file_name_data + '.pkl', 'rb') as f:
        #     self.data = pickle.load(f)
        self.labels = np.load(file_path + file_name_label + '.npy') # labels


    # save data and labels to file
    def save_varibles(self, file_path, file_name_data, file_name_label):
        np.save(file_path + file_name_data, self.data) # data
        # with open(file_path + file_name_data + '.pkl' , 'wb') as f:
        #     pickle.dump(self.data, f)
        np.save(file_path + file_name_label, self.labels) # labels


