import torch
import torch.nn as nn
from torch import device, optim
import copy
from torchsummary import summary

# Temporal Convolutional Network only one refinment layer
class TConvNet(nn.Module):
    def __init__(self, num_layers, num_feature_maps, num_classes, input_dim):
        super(TConvNet, self).__init__()
        # input convolution
        self.conv_in = nn.Conv1d(input_dim, num_feature_maps, 1)
        # Dialeted Residual Layers
        self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(dilation=2**i, c_in=num_feature_maps, c_out=num_feature_maps)) for i in range(num_layers)])
        # output convolution
        self.conv_out = nn.Conv1d(num_feature_maps, num_classes, 1)
    
    def forward(self, x):
        # input convolution
        x = self.conv_in(x)
        # Dialeted Residual Layers
        for layer in self.layers:
            x = layer(x)
        # output convolution
        x = self.conv_out(x)
        return x

class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, c_in, c_out):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(c_in, c_out, 3, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(c_out, c_out, 1)

    def forward(self, x):
        # Dilated convolution with residual connection
        out = self.conv_dilated(x)
        out = self.conv_1x1(out)
        return out + x
    

import json
import h5py
import numpy as np
from torch.utils.data import DataLoader, TensorDataset


def train_validate(device, epochs, learning_rate, num_layers, batch_size, num_channels):
    with open('data/sets.json') as f:
        sets = json.load(f)
    
    # avds000-lab010-01 as train data
    train_y = sets['train']["avds000-lab010-01"]
    train_y = np.array(train_y)
    train_y = torch.from_numpy(train_y).to(device)

    with h5py.File('data/roi_mouth/avds000-lab010-01.h5', 'r') as f:
        train_x = f['data']
        train_x = np.array(train_x)
        train_x = torch.from_numpy(train_x).to(device)
        train_x = train_x.view(train_x.shape[0], -1)
        # reduce train_x dimension by 2

    val_y = sets['validation']["avds012-lab010-01"]
    val_y = np.array(val_y)
    val_y = torch.from_numpy(val_y).to(device)

    with h5py.File('data/roi_mouth/avds012-lab010-01.h5', 'r') as f:
        val_x = f['data']
        val_x = np.array(val_x)
        val_x = torch.from_numpy(val_x).to(device)
        val_x = val_x.view(val_x.shape[0], -1)
        # reduce val_x dimension by 2
    


    # model
    model = TConvNet(num_layers, num_channels, train_y.shape[0], train_x.shape[1])
    
    # Loss and Optimizer loss cross entropy
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the Model
    for epoch in range(epochs):
        for x, y in next_batch(train_x, train_y, batch_size):
            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, epochs, loss.item()))

    
        


def next_batch(x, y, batch_size):
    for i in range(0, x.shape[0], batch_size):
        yield x[i:i+batch_size], y[i:i+batch_size]

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epochs = 1000
    learning_rate = 1e-3
    batch_size = 4
    num_layers = 1 # number of layers in the network
    num_channels = 16 # number of feature maps

    train_validate(device, epochs, learning_rate, num_layers, batch_size, num_channels)

