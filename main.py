import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import os
import model as m  
from dataset import HDF5Dataset
import trainer as t
import eval
import itertools
from torch.utils.tensorboard import SummaryWriter
import csv

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 64
    epochs = 30
    lr = [1e-3]
    step_size = [5]
    frame_count = [2, 4, 8, 16, 32, 64]
    filter_size = [16]
    number_layers = [1]
    
    # get combinations of parameters lr, step_size, frame_count, filter_size
    lr_step_size_frame_filter = list(itertools.product(frame_count, step_size, lr, filter_size, number_layers))

    num_class = 2

    i = 0
    frame_count_prev = 0
    step_size_prev = 0

    # create csv file for storing results
    results_file = open('results.csv', 'w')
    results_writer = csv.writer(results_file)
    # write header
    # results_writer.writerow(['lr', 'step_size', 'frame_count', 'filter_size', 'acc', 'f1_0.1', 'f1_0.25', 'f1_0.5', \
    #                         'precision_0.1', 'precision_0.25', 'precision_0.5', 'recall_0.1', 'recall_0.25', 'recall_0.5'])
    results_writer.writerow(['lr', 'step_size', 'frame_count', 'filter_size', 'acc', 'f1_0.1', 'f1_0.25', 'f1_0.5', \
                            'precision_0.1', 'precision_0.25', 'precision_0.5', 'recall_0.1', 'recall_0.25', 'recall_0.5'])

    model_dir = f'models/lr_{lr}_step_size_{step_size}_frame_count_{frame_count}_filter_size_{filter_size}_num_layer_{number_layers}.pt'
    
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
            # model = m.DilatedResidualLayer(output_channels=num_class, input_channels=frame_count, filter_size=filter_size).to(device)
            model = m.SS_TCN(num_classes=num_class, num_input=frame_count, filter_size=filter_size, num_layers=number_layers).to(device)
            loss_fn = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            trainer = t.Trainer(model, loss_fn, optimizer, epochs=epochs, train_dataloader=training_loader, \
                                val_dataloader=val_loader, test_dataloader=test_loader, device=device, writer=writer)
            trainer.train()
            print('Training complete.')
            # saves model
            torch.save(model.state_dict(), model_dir)
        else:
            print('Already has model. Skipping training.')
            # model = m.DilatedResidualLayer(output_channels=num_class, input_channels=frame_count, filter_size=filter_size).to(device)
            model = m.SS_TCN(num_classes=num_class, num_input=frame_count, filter_size=filter_size, num_layers=number_layers).to(device)
            model.load_state_dict(torch.load(model_dir))



        # Evaluate model on test data
        evaluation = eval.Eval(model, test_loader, device)
        # acc, edit, f1_arr, precision_list, recall_list = evaluation.eval()
        acc, f1_arr, precision_list, recall_list = evaluation.eval()
        # acc = evaluation.eval()


        # append results in csv file
        # results_writer.writerow([lr, step_size, frame_count, filter_size, acc, edit, f1_arr[0], f1_arr[1], f1_arr[2], \
        #                         precision_list[0], precision_list[1], precision_list[2], recall_list[0], recall_list[1], recall_list[2]])
        results_writer.writerow([lr, step_size, frame_count, filter_size, acc, f1_arr[0], f1_arr[1], f1_arr[2], \
                                precision_list[0], precision_list[1], precision_list[2], recall_list[0], recall_list[1], recall_list[2]])
        # results_writer.writerow([lr, step_size, frame_count, filter_size, acc]) # only accuracy

        i += 1
        print(f"{i}/{len(lr_step_size_frame_filter)} combinations")
        
        frame_count_prev = frame_count
        step_size_prev = step_size
    
    # close csv file
    results_file.close()
