import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import os
import numpy as np

import model as m  
from dataset import HDF5Dataset
import trainer as t

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 64
    lr = 1e-3
    epochs = 100
    printBatch = True

    onlyOneFile = False
    file_train = "avds000-lab010-01"
    file_val = "avds012-lab010-01"

    training_data = HDF5Dataset(file_path='data', group='train')
    val_data = HDF5Dataset(file_path='data', group='val')

    # if var folder not empty then load data and labels from saved numpy arrays
    # else create data and labels
    if onlyOneFile == False:
        if not os.path.exists('data/var'):
            os.mkdir('data/var')
        
        if not os.path.exists('data/var/data.npy'):
            training_data.create_data()
            val_data.create_data()
        else:
            print('Loading data...')
            training_data.data = np.load('data/var/data.npy')
            val_data.data = np.load('data/var/data.npy')
            print('Data loaded.')

        
        if not os.path.exists('data/var/labels.npy'):
            training_data.create_labels()
            val_data.create_labels()
        else:
            print('Loading labels...')
            training_data.labels = np.load('data/var/labels.npy')
            val_data.labels = np.load('data/var/labels.npy')
            print('Labels loaded.')
        
        training_data.save_varibles('data/var')
        val_data.save_varibles('data/var')

    else:
        training_data.create_one_data(file_train)
        training_data.create_one_labels(file_train, 'train')
        val_data.create_one_data(file_val)
        val_data.create_one_labels(file_val, 'val')
    


    training_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

    # frame_number = training_loader.dataset.data.shape[0]
    frame_features_train = training_loader.dataset.data.shape[1]

    model = m.DilatedResidualLayer(channels=2, output=1, dim=frame_features_train).to(device)
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    trainer = t.Trainer(model, loss_fn, optimizer, epochs=epochs, train_dataloader=training_loader, \
                        val_dataloader=val_loader, printBatch=printBatch, device=device)
    trainer.train()
           
        
