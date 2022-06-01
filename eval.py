from turtle import color
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt


class Eval():
    def __init__(self, model, test_dataloader: DataLoader, device, frame_count):
        self.model = model
        self.test_dataloader = test_dataloader
        self.device = device
        self.frame_count = frame_count
        # self.step_size = step_size
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

    def _predict(self):
        gts, preds = np.array([]), np.array([])

        self.model.eval()
        with torch.no_grad():
            for _, (X, y) in enumerate(self.test_dataloader):
                X = X.to(self.device)
                y = y.type(torch.LongTensor).to(self.device)
                ground_truth = y
                ground_truth = ground_truth.cpu().numpy()

                pred = self.model(X)
                pred_target = pred.argmax(1)
                pred_target = pred_target.cpu().numpy()

                pred, ground_truth = self._expand_frames(pred_target, ground_truth)

                # append to preds and gts
                gts = np.append(gts, ground_truth)
                preds = np.append(preds, pred)
        
        return gts, preds

    def predict_show(self, start, end):
        gts, preds = self._predict()
        gts = gts[start:end]
        preds = preds[start:end]

        # matplotlib
        plt.figure(figsize=(10, 5))
        plt.plot(gts, label='ground truth')
        plt.plot(preds, label='prediction')
        plt.legend()
        plt.show()                
                
    # exapnd frames by frame count
    def _expand_frames(self, pred, gt):
        pred = np.repeat(pred, self.frame_count)
        gt = np.repeat(gt, self.frame_count)

        return pred, gt


        
            

