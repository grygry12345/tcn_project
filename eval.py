import numpy as np
import torch
from torch.utils.data import DataLoader

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

    def _levenstein(self, p, y, norm=False):
        m_row = len(p)
        n_col = len(y)
        D = np.zeros([m_row+1, n_col+1], np.float)
        for i in range(m_row+1):
            D[i, 0] = i
        for i in range(n_col+1):
            D[0, i] = i


        for j in range(1, n_col+1):
            for i in range(1, m_row+1):
                if y[j-1] == p[i-1]:
                    D[i, j] = D[i-1, j-1]
                else:
                    D[i, j] = min(D[i-1, j] + 1,
                                D[i, j-1] + 1,
                                D[i-1, j-1] + 1)
            
            print(f"Levenstein {j}/{n_col}", end="\r")

        if norm:
            score = (1 - D[-1, -1]/max(m_row, n_col)) * 100
        else:
            score = D[-1, -1]

        return score
    
    def _get_labels_start_end_time(self, frame_wise_labels, bg_class=[-1]):
        labels = []
        starts = []
        ends = []
        last_label = frame_wise_labels[0]
        if frame_wise_labels[0] not in bg_class:
            labels.append(frame_wise_labels[0])
            starts.append(0)
        for i in range(len(frame_wise_labels)):
            if frame_wise_labels[i] != last_label:
                if frame_wise_labels[i] not in bg_class:
                    labels.append(frame_wise_labels[i])
                    starts.append(i)
                if last_label not in bg_class:
                    ends.append(i)
                last_label = frame_wise_labels[i]
        if last_label not in bg_class:
            ends.append(i)
        return labels, starts, ends

    
    def _edit_score(self, recognized, ground_truth, norm=True, bg_class=[-1]): # ! not sure it works
        P, _, _ = self._get_labels_start_end_time(recognized, bg_class)
        Y, _, _ = self._get_labels_start_end_time(ground_truth, bg_class)
        return self._levenstein(P, Y, norm)
    
    def _f_score(self, recognized, ground_truth, overlap, bg_class=[-1]):
        p_label, p_start, p_end = self._get_labels_start_end_time(recognized, bg_class)
        y_label, y_start, y_end = self._get_labels_start_end_time(ground_truth, bg_class)

        p_label = torch.tensor(p_label, dtype=torch.int64, device=self.device)
        y_label = torch.tensor(y_label, dtype=torch.int64, device=self.device)

        tp = 0
        fp = 0
        i = 0
        hits = np.zeros(len(y_label))

        for j in range(len(p_label)):
            i += 1
            intersection = np.minimum(p_end[j], y_end) - np.maximum(p_start[j], y_start)
            union = np.maximum(p_end[j], y_end) - np.minimum(p_start[j], y_start)

            # convert intersection to torch array device cuda
            intersection = torch.from_numpy(intersection).to(self.device)
            union = torch.from_numpy(union).to(self.device)

            
            IoU = (1.0 * intersection / union) * (p_label[j] == y_label[:]) # !


            # Get the best scoring segment
            idx = IoU.argmax()

            if IoU[idx] >= overlap and not hits[idx]:
                tp += 1
                hits[idx] = 1
            else:
                fp += 1
            print(f"F1 score {i}/{len(p_label)}", end="\r")
        
        
        fn = len(y_label) - sum(hits)
        
        return float(tp), float(fp), float(fn)

    def eval(self):
    
        overlap = [.1, .25, .5]
        tp, fp, fn = np.zeros(len(overlap)), np.zeros(len(overlap)), np.zeros(len(overlap))
        size = len(self.test_dataloader.dataset) # for test
        correct = 0

        self.model.eval()
        with torch.no_grad():
            for _, (X, y) in enumerate(self.test_dataloader):
                X = X.to(self.device)

                y = y.type(torch.LongTensor).to(self.device)
                ground_truth = y

                pred = self.model(X)
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


                correct += (pred_target == ground_truth).sum().item()
                self._gts = np.append(self._gts, ground_truth.cpu().numpy())
                
                self._preds = np.append(self._preds, pred_target.cpu().numpy())


            acc = (correct / size) * 100

            # edit_score = self._edit_score(self._preds, self._gts)
            
            precision_list = []
            recall_list = []
            f1_list = []
            for i in range(len(overlap)):
                tp, fp, fn = self._f_score(self._preds, self._gts, overlap[i])

                precision = tp / (tp + fp)
                recall = tp / (tp + fn)
                precision_list.append(precision)
                recall_list.append(recall)


                if (precision + recall) != 0:
                    f1 = 2 * precision * recall / (precision + recall)
                else:
                    f1 = 0.0

                f1_list.append(f1)
        
        # return acc, edit, f1_list, precision_list, recall_list
        return acc, f1_list, precision_list, recall_list
