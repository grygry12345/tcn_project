import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Referance : https://github.com/yabufarha/ms-tcn/blob/master/eval.py
# Y. Abu Farha and J. Gall.
# MS-TCN: Multi-Stage Temporal Convolutional Network for Action Segmentation.
# In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2019

# S. Li, Y. Abu Farha, Y. Liu, MM. Cheng,  and J. Gall.
# MS-TCN++: Multi-Stage Temporal Convolutional Network for Action Segmentation.
# In IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2020

class Eval():
    def __init__(self, model, test_dataloader: DataLoader, device):
        self.model = model
        self.test_dataloader = test_dataloader
        self.device = device
        self._preds = np.empty((0))
        self._gts = np.empty((0))

    def eval(self):
        self.model.eval()
        with torch.no_grad():
            for _, (X, y) in enumerate(self.test_dataloader):
                X = X.to(self.device)
                y = y.type(torch.LongTensor).to(self.device)
                ground_truth = y

                pred = self.model(X)

                pred_target = pred.argmax(1)
                
                self._gts = np.append(self._gts, ground_truth.cpu().numpy())
                self._preds = np.append(self._preds, pred_target.cpu().numpy())
        
        acc = accuracy_score(self._gts, self._preds)
        f1 = f1_score(self._gts, self._preds, average='macro')
        precision = precision_score(self._gts, self._preds, average='macro')
        recall = recall_score(self._gts, self._preds, average='macro')

        
        return acc, f1, precision, recall
