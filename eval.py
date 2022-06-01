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


        
    def _find_best_worst_clips(self):
        gts, preds = np.array([]), np.array([])
        minCorrect = 999999
        maxCorrect = 0

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

                correct = np.sum(pred_target == ground_truth)
                
                if correct > maxCorrect:
                    maxCorrect = correct
                    best_pred = pred_target
                    best_gts = ground_truth

                if correct < minCorrect:
                    minCorrect = correct
                    worst_pred = pred_target
                    worst_gts = ground_truth

        return best_pred, best_gts, worst_pred, worst_gts, minCorrect, maxCorrect



    def show_best_worst_clips(self):
        best_pred, best_gts, worst_pred, worst_gts, minCorrect, maxCorrect = self._find_best_worst_clips()


        # figure 1 only best clip
        plt.figure(figsize=(20, 10))
        plt.plot(best_pred, label='pred')
        plt.plot(best_gts, label='gts')
        plt.legend()
        plt.xlabel('frames')
        plt.ylabel('class')
        plt.title('Best clip')
        plt.savefig('best_clip.png') 

        # figure 2 only worst clip
        plt.figure(figsize=(20, 10))
        plt.plot(worst_pred, label='pred')
        plt.plot(worst_gts, label='gts')
        plt.legend()
        plt.xlabel('frames')
        plt.ylabel('class')
        plt.title('Worst clip')
        plt.savefig('worst_clip.png')

        print('Best clip accuracy: {}'.format(maxCorrect))
        print('Worst clip accuracy: {}'.format(minCorrect))

        
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

