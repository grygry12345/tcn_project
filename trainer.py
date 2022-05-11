from numpy import dtype
import torch.nn as nn 
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

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
    

    def _step_train(self):
        train_loss, train_correct = 0.0, 0.0
        num_batches = len(self.train_dataloader)
        size = len(self.train_dataloader.dataset)
        self.model.train()
        for batch, (X, y) in enumerate(self.train_dataloader):
            X = X.to(self.device)
            y = y.type(torch.LongTensor)
            y = y.to(self.device)

            # Compute prediction error
            pred = self.model(X)

            loss = self.loss_fn(pred.squeeze(), y)
            train_correct += (pred.argmax(1) == y).type(torch.FloatTensor).sum().item()


            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
        
        train_loss /= num_batches
        train_correct /= size

        return train_loss, train_correct
            
    
    def _step_val(self):
        size = len(self.val_dataloader.dataset)
        num_batches = len(self.val_dataloader)
        self.model.eval()
        val_loss, val_correct = 0, 0
        with torch.no_grad():
            for _ , (X, y) in enumerate(self.val_dataloader):
                X = X.to(self.device)
                y = y.type(torch.LongTensor)
                y = y.to(self.device)

                # Compute prediction error
                pred = self.model(X)
                
                pred = pred.squeeze(-1)
                val_loss += self.loss_fn(pred, y).item()

                val_correct += (pred.argmax(1) == y).type(torch.FloatTensor).sum().item()
        val_loss /= num_batches
        val_correct /= size

        return val_loss, val_correct





    def train(self):
        

        writer = SummaryWriter(f'run/test_{self.model.__class__.__name__}')
        for t in range(self.epochs):
            train_loss, train_correct = self._step_train()
            val_loss, val_correct = self._step_val()
            writer.add_scalar('train_loss', train_loss, t)
            writer.add_scalar('train_acc', train_correct, t)
            writer.add_scalar('val_loss', val_loss, t)
            writer.add_scalar('val_acc', val_correct, t)

        
    

