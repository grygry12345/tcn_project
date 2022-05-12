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
    epochs = 20
    lr = 1e-3

    onlyOneFile = True
    file_train = "avds000-lab010-01"
    file_val = "avds012-lab010-01"
    file_test = "avds014-lab010-01"

    training_data = HDF5Dataset(file_path='data', group='train')
    val_data = HDF5Dataset(file_path='data', group='val')
    test_data = HDF5Dataset(file_path='data', group='test')

    # if var folder not empty then load data and labels from saved numpy arrays
    # else create data and labels
    if onlyOneFile == False:
        if not os.path.exists('data/var'):
            os.mkdir('data/var')
        
        # Check if data already saved in var folder 

        if not os.path.exists('data/var/data_train.npy'):
            training_data.create_data()
            training_data.create_labels()
            training_data.save_varibles('data/var/', 'data_train', 'label_train')
        else:
            print('Loading train data...')
            training_data.load_variables('data/var/', 'data_train', 'label_train')
            print('Train data loaded.')

        if not os.path.exists('data/var/data_val.npy'):
            val_data.create_data()
            val_data.create_labels()
            val_data.save_varibles('data/var/', 'data_val', 'label_val')
        else:
            print('Loading val data...')
            val_data.load_variables('data/var/', 'data_val', 'label_val')
            print('Val data loaded.')
        
        if not os.path.exists('data/var/data_test.npy'):
            test_data.create_data()
            test_data.create_labels()
            test_data.save_varibles('data/var/', 'data_test', 'label_test')
        else:
            print('Loading test data...')
            test_data.load_variables('data/var/', 'data_test', 'label_test')
            print('Test data loaded.')
    else:
        training_data.create_one_data(file_train)
        training_data.create_one_labels(file_train, 'train')
        val_data.create_one_data(file_val)
        val_data.create_one_labels(file_val, 'val')
        test_data.create_one_data(file_test)
        test_data.create_one_labels(file_test, 'test')
    


    training_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, shuffle=True)

    # frame_number = training_loader.dataset.data.shape[0]
    frame_features_train = training_loader.dataset.data.shape[1]

    model = m.DilatedResidualLayer(channels=2, output=10, dim=frame_features_train).to(device)
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    trainer = t.Trainer(model, loss_fn, optimizer, epochs=epochs, train_dataloader=training_loader, \
                        val_dataloader=val_loader, test_dataloader=test_loader, device=device)
    trainer.train()

    # Predict on test data
    trainer.predict()
           
        
