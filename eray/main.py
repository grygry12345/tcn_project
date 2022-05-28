import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import os
import model2 as m2
from dataset import HDF5Dataset
import trainer as t
import eval
import itertools
from torch.utils.tensorboard import SummaryWriter
import csv

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    batch_size = 16
    epochs = 30
    lr = [1e-3, 1e-6]
    step_size = [5 , 10, 20]
    frame_count = [16]
    hidden_size = [64]
    number_layers = [2]
    
    # get combinations of parameters lr, step_size, frame_count, filter_size
    lr_step_size_frame_filter = list(itertools.product(frame_count, step_size, lr, hidden_size, number_layers))

    num_class = 2

    i = 0
    frame_count_prev = 0
    step_size_prev = 0

    # create csv file for storing results
    results_file = open('results.csv', 'w')
    results_writer = csv.writer(results_file)
    # write header
    results_writer.writerow(['lr', 'step_size', 'frame_count', 'hidden_size', 'acc', 'f1', \
                            'precision', 'recall'])

    model_dir = f'models/lr_{lr}_step_size_{step_size}_frame_count_{frame_count}_filter_size_{hidden_size}_num_layer_{number_layers}.pt'
    
    # iterate through all combinations
    for frame_count, step_size, lr, filter_size, number_layers in lr_step_size_frame_filter:
        
        writer = SummaryWriter(f"runs/frame_count_{frame_count}_step_size_{step_size}_lr_{lr}_filter_size_{filter_size}")
        
        if frame_count_prev == frame_count and step_size_prev == step_size:
            print('Same parameters "frame_size" and "step_size" skipping')
        else:
            

            train = HDF5Dataset(file_path='data', group='train', device=device, frame_count=frame_count, step_size=step_size)
            val = HDF5Dataset(file_path='data', group='val', device=device, frame_count=frame_count, step_size=step_size)
            test = HDF5Dataset(file_path='data', group='test', device=device, frame_count=frame_count, step_size=step_size)

            train.create_data()
            train.create_labels()
            val.create_data()
            val.create_labels()
            test.create_data()
            test.create_labels()

            training_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test,batch_size=batch_size, shuffle=True)
            

        # if model is not already created, create model
        if not os.path.exists(f'new_var'):
            print('Creating new model')
            model = m2.CNN(output_dim=num_class).to(device)
            loss_fn = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            trainer = t.Trainer(model, loss_fn, optimizer, epochs=epochs, train_dataloader=training_loader, \
                                val_dataloader=val_loader, test_dataloader=test_loader, device=device, writer=writer)
            trainer.train()
            print('Training complete.')
        else:
            print('Already has model. Skipping training.')



        # Evaluate model on test data
        evaluation = eval.Eval(model, test_loader, device)
        acc, f1_arr, precision_list, recall_list = evaluation.eval()


        # append results in csv file
        results_writer.writerow([lr, step_size, frame_count, filter_size, acc, f1_arr[0], \
                                precision_list[0], recall_list[0]])

        i += 1
        print(f"{i}/{len(lr_step_size_frame_filter)} combinations")
        
        frame_count_prev = frame_count
        step_size_prev = step_size
    
    # close csv file
    results_file.close()
