import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import os
import model as m  
from dataset import HDF5Dataset
import trainer as t
import eval



if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 64
    epochs = 100
    lr = 1e-3
    step_size = 3
    frame_count = 3
    filter_size = 4
    
    num_class = 2

    training_data = HDF5Dataset(file_path='data', group='train', device=device, frame_count=frame_count, step_size=step_size)
    val_data = HDF5Dataset(file_path='data', group='val', device=device, frame_count=frame_count, step_size=step_size)
    test_data = HDF5Dataset(file_path='data', group='test', device=device, frame_count=frame_count, step_size=step_size)

    # if var folder not empty then load data and labels from saved numpy arrays
    # else create data and labels
    # if not os.path.exists('data/var'):
    #     os.mkdir('data/var')
    
    # # Check if data already saved in var folder 

    # if not os.path.exists('data/var/data_train.pt'):
    #     training_data.create_data()
    #     training_data.create_labels()
    #     training_data.save_varibles('data/var/', 'data_train', 'label_train')
    # else:
    #     print('Loading train data...')
    #     training_data.load_variables('data/var/', 'data_train', 'label_train')
    #     print('Train data loaded.')

    # if not os.path.exists('data/var/data_val.pt'):
    #     val_data.create_data()
    #     val_data.create_labels()
    #     val_data.save_varibles('data/var/', 'data_val', 'label_val')
    # else:
    #     print('Loading val data...')
    #     val_data.load_variables('data/var/', 'data_val', 'label_val')
    #     print('Val data loaded.')
    
    # if not os.path.exists('data/var/data_test.pt'):
    #     test_data.create_data()
    #     test_data.create_labels()
    #     test_data.save_varibles('data/var/', 'data_test', 'label_test')
    # else:
    #     print('Loading test data...')
    #     test_data.load_variables('data/var/', 'data_test', 'label_test')
    #     print('Test data loaded.')

    training_data.create_data()
    training_data.create_labels()
    val_data.create_data()
    val_data.create_labels()
    test_data.create_data()
    test_data.create_labels()
    


    training_loader = DataLoader(training_data, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data,batch_size=batch_size, shuffle=True)

    model = m.DilatedResidualLayer(output_channels=num_class, input_channels=frame_count, filter_size=filter_size).to(device) # ? assigning batch size could be wrong
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    trainer = t.Trainer(model, loss_fn, optimizer, epochs=epochs, train_dataloader=training_loader, \
                        val_dataloader=val_loader, test_dataloader=test_loader, device=device)
    trainer.train()
    print('Training complete.')

    eval = eval.Eval(model, test_loader, device)
    eval.eval()
