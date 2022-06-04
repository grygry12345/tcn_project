from dataset import HDF5Dataset
import trainer as t
import eval
from model import ResCNN, Linear, BasicLinear

from torch.utils.tensorboard import SummaryWriter
import torch
from torch.utils.data import DataLoader
import os
import itertools
import torch.nn as nn

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 64
    epochs = 100
    lr = [1e-3]
    
    step_size = [8]
    frame_count = [16]
    channel_size_feature = [64]
    channel_size_model = [16]
    num_layers = [1]
    use_conv = [False]
    

    # get combinations of parameters lr, step_size, frame_count, filter_size

    num_channel = 2
    lr_prev = 0
    filter_size_model_prev = 0
    filter_size_feature_prev = 0
    number_layers_prev = 0

    i = 0
    frame_count_prev = 0
    step_size_prev = 0

    lr_step_size_frame_filter = list(itertools.product(num_layers, use_conv, frame_count, step_size, lr, channel_size_feature, channel_size_model))
    
      
    # if runs folder does not exist, create it
    if not os.path.exists('runs'):
        os.makedirs('runs')
    
    # iterate through all combinations
    for num_layers, use_conv, frame_count, step_size, lr, channel_size_feature, channel_size_model in lr_step_size_frame_filter:
                
        writer = SummaryWriter(f"runs/frame_count-{frame_count}_step_size-{step_size}_lr-{lr}_filter_size_feature-{channel_size_feature}_filter_size_model-{channel_size_model}")
        
        feature_output = channel_size_feature
        
        if use_conv == True:
            model_feature = ResCNN(num_input=num_channel, hidden_size=channel_size_feature, num_layers=num_layers).to(device)
        else:
            model_feature = BasicLinear(num_input=num_channel, hidden_size=channel_size_feature).to(device)
        
        
        model = Linear(num_input=(feature_output), hidden_size=channel_size_model, num_output=num_channel).to(device)
        if frame_count_prev == frame_count and step_size_prev == step_size and channel_size_feature == filter_size_feature_prev:
            print('Same parameters "frame_size" and "step_size" skipping')
        else:
            train = HDF5Dataset(file_path='data', group='train', device=device, frame_count=frame_count, step_size=step_size, \
                                model_feature=model_feature)
            val = HDF5Dataset(file_path='data', group='val', device=device, frame_count=frame_count, step_size=step_size, \
                                model_feature=model_feature)
            test = HDF5Dataset(file_path='data', group='test', device=device, frame_count=frame_count, step_size=step_size, \
                                model_feature=model_feature)

            train.create_data()
            train.create_labels()
            val.create_data()
            val.create_labels()
            test.create_data()
            test.create_labels()

            training_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test,batch_size=batch_size, shuffle=True)

            
            

        print('Creating new model')
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        trainer = t.Trainer(model, loss_fn, optimizer, epochs=epochs, train_dataloader=training_loader, \
                            val_dataloader=val_loader, device=device, writer=writer)
        trainer.train()
        print('Training complete.')


        # Evaluate model on test data
        evaluation = eval.Eval(model, test_loader, device, frame_count=frame_count)
        acc, f1, precision, recall = evaluation.eval()
        evaluation.show_best_worst_clips()

        # round to 2 decimal places
        acc = round(acc, 2) * 100
        f1 = round(f1, 2) * 100
        precision = round(precision, 2) * 100
        recall = round(recall, 2) * 100

        # Save hyperparameters
        writer.add_hparams({"num_layers": num_layers, "use_conv": use_conv, "frame_count": frame_count, "step_size": step_size, "lr": lr, "filter_size_feature": channel_size_feature, "filter_size_model": channel_size_model}, \
                            {"acc": acc, "f1": f1, "precision": precision, "recall": recall})




        i += 1
        print(f"{i}/{len(lr_step_size_frame_filter)} combinations")
        
        print(f"acc: {acc}, f1: {f1}, precision: {precision}, recall: {recall}")

        
        frame_count_prev = frame_count
        step_size_prev = step_size
        
        lr_prev = lr
        filter_size_model_prev = channel_size_model
        filter_size_feature_prev = channel_size_feature

    
    # results_file.close()
