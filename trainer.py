import torch.nn as nn 
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

class Trainer(nn.Module):
    def __init__(self, model, loss_fn, optimizer, epochs: int, train_dataloader: DataLoader, val_dataloader: DataLoader, \
    test_dataloader: DataLoader, device: str = "cuda:0", writer: SummaryWriter = None):
        super(Trainer, self).__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.epochs = epochs
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.device = device
        # Summary writer with date and model name
        self.writer = writer
    

    def _step_train(self):
        train_loss = 0.0
        num_batches = len(self.train_dataloader)
        self.model.train()
        for batch, (X, y) in enumerate(self.train_dataloader):
            X = X.to(self.device)
            y = y.type(torch.LongTensor).to(self.device)


            # Compute prediction error
            pred = self.model(X)

            loss = self.loss_fn(pred, y.squeeze())

            
            self.optimizer.zero_grad()
            
            # Backpropagation
            loss.backward()
            self.optimizer.step()
            


            train_loss += loss.item()
        
        train_loss /= num_batches

        return train_loss
            
    
    def _step_val(self):
        
        size = len(self.val_dataloader.dataset)
        num_batches = len(self.val_dataloader)
        self.model.eval()
        val_loss, val_correct = 0, 0
        with torch.no_grad():
            for _ , (X, y) in enumerate(self.val_dataloader):
                X = X.to(self.device)
                
                y = y.type(torch.LongTensor).to(self.device)

                # Compute prediction error
                pred = self.model(X)
                
                loss = self.loss_fn(pred, y.squeeze())

                val_loss += loss.item()

                pred_target = pred.argmax(1)

                if pred_target.unique().shape[0] == 1:
                    if pred_target.unique() == 0:
                            pred_target  = torch.tensor([0], dtype=torch.float32, device=self.device)
                    elif pred_target.unique() == 1:
                            pred_target  = torch.tensor([1], dtype=torch.float32, device=self.device)
                elif pred_target.unique().shape[0] == 2:
                    # count zero values
                    c0 = torch.sum(pred_target == 0)
                    c1 = torch.sum(pred_target == 1)
                    if c0 > c1:
                        pred_target = torch.tensor([0], dtype=torch.float32, device=self.device)
                    elif c1 > c0 or c1 == c0:
                        pred_target = torch.tensor([1], dtype=torch.float32, device=self.device)

                val_correct += torch.sum(pred_target == y).item()


        val_loss /= num_batches
        val_correct /= size

        return val_loss, val_correct

    def train(self):
        for t in range(self.epochs):
            print(f"Epoch {t}/{self.epochs}", end="\r")
            train_loss = self._step_train()
            val_loss, val_correct = self._step_val()
            self.writer.add_scalar('train_loss', train_loss, t)
            self.writer.add_scalar('val_loss', val_loss, t)
            self.writer.add_scalar('val_acc', val_correct, t)
        print()













