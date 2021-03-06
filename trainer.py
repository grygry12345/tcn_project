from pyexpat import features
import torch.nn as nn 
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

class Trainer(nn.Module):
    def __init__(self, model, loss_fn, optimizer, epochs: int, train_dataloader: DataLoader, val_dataloader: DataLoader, \
    device: str = "cpu", writer: SummaryWriter = None):
        super(Trainer, self).__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.epochs = epochs
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.writer = writer
    

    def _step_train(self):
        train_loss, train_correct = 0.0, 0.0
        num_batches = len(self.train_dataloader)
        size = len(self.train_dataloader.dataset)
        
        self.model.train()
        for _, (X, y) in enumerate(self.train_dataloader):
            X = X.to(self.device)
            y = y.type(torch.LongTensor).to(self.device)


            # Compute prediction error
            pred = self.model(X)
            loss = self.loss_fn(pred, y)


            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            pred_target = pred.argmax(1)


            train_correct += torch.sum(pred_target == y).item()
            train_loss += loss.item()
        
        train_loss /= num_batches
        train_correct /= size

        return train_loss, train_correct
            
    
    def _step_val(self):
        
        size = len(self.val_dataloader.dataset)
        num_batches = len(self.val_dataloader)
        val_loss, val_correct = 0, 0

        self.model.eval()
        with torch.no_grad():
            for _ , (X, y) in enumerate(self.val_dataloader):
                X = X.to(self.device)
                y = y.type(torch.LongTensor).to(self.device)

                pred = self.model(X)
                
                loss = self.loss_fn(pred, y)

                val_loss += loss.item()

                pred_target = pred.argmax(1)

                val_correct += torch.sum(pred_target == y).item()


        val_loss /= num_batches
        val_correct /= size

        return val_loss, val_correct

    def train(self):
        for t in range(self.epochs):
            print(f"Epoch {t}/{self.epochs}", end="\r")
            train_loss, train_correct = self._step_train()
            val_loss, val_correct = self._step_val()
            self.writer.add_scalar('train_loss', train_loss, t)
            self.writer.add_scalar('val_loss', val_loss, t)
            self.writer.add_scalar('train_acc', train_correct, t)
            self.writer.add_scalar('val_acc', val_correct, t)
        print()













