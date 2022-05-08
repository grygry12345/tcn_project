from numpy import dtype
import torch.nn as nn 
import torch
from torch.utils.data import DataLoader

# TODO: design device selection and add default optimizer and loss function
class Trainer(nn.Module):
    def __init__(self, model, loss_fn, optimizer, epochs: int, train_dataloader: DataLoader, val_dataloader: DataLoader, device: str = "cuda:0", printBatch: bool = False):
        super(Trainer, self).__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.epochs = epochs
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.printBatch = printBatch
    

    def _step_train(self):
        train_loss = 0.0
        size = len(self.train_dataloader.dataset)
        self.model.train()
        for batch, (X, y) in enumerate(self.train_dataloader):
            X = X.to(self.device)
            y = y.type(torch.LongTensor)
            y = y.to(self.device)

            # Compute prediction error
            pred = self.model(X).to(dtype=torch.float64)

            loss = self.loss_fn(pred.squeeze(), y.squeeze())

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()

            # print batch progress 
            if batch % 100 == 0 and self.printBatch == True:
                print(f"Batch {batch}/{len(self.train_dataloader)}", end="\r")
        
        train_loss /= size

        return train_loss
            
    
    def _step_val(self):
        size = len(self.val_dataloader.dataset)
        num_batches = len(self.val_dataloader)
        self.model.eval()
        i, val_loss, correct = 0, 0, 0
        with torch.no_grad():
            for _ , (X, y) in enumerate(self.val_dataloader):
                X = X.to(self.device)
                y = y.type(torch.LongTensor)
                y = y.to(self.device)

                # Compute prediction error
                pred = self.model(X)
                pred = pred.squeeze()

                val_loss += self.loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.FloatTensor).sum().item()
        val_loss /= num_batches
        correct /= size

        return val_loss, correct





    def train(self):
        for t in range(self.epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            train_loss = self._step_train()
            val_loss, correct = self._step_val()
            print("Train error:", train_loss)
            print("Val error:", val_loss)
            # double digit precision
            print("Val accuracy:", round(correct, 2) * 100, "%")
            print("\n")

