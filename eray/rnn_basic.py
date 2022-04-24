import json
import math
import h5py
from sklearn.metrics import accuracy_score, r2_score
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

def plot_timeseq(eventspiketensor, frame_size_ms, segment, name):
    event_activation = eventspiketensor.sum(axis=(1, 2, 3))
    step = frame_size_ms * 1e-3  # Convert to second
    t_offset = step / 2
    t_frames = t_offset + step * np.arange(event_activation.shape[0])
    fig, ax = plt.subplots(figsize=(10, 2))
    ax.plot(t_frames, event_activation)
    if segment is not None and (segment[0] >= 0 and segment[1] >= 0):
        # print(segment)
        ax.axvline(segment[0], c='k')
        ax.axvline(segment[1], c='k')
    ax.set_xlim((t_frames[0], t_frames[-1]))
    max_y = event_activation.max()
    ax.set_ylim((0, max_y))
    return fig, ax

Y=[]
with open('sets.json', 'r') as myfile:
    data=myfile.read()
    obj = json.loads(data)
    for element in obj['train']:
        continue
    with h5py.File("roi_mouth/avds004-lab015-01.h5", "r") as f:
        a_group_key = list(f.keys())[0]
        train_data=np.array(list(f[a_group_key]))
        event_activation = train_data.sum(axis=(1, 2, 3))
        step = 10 * 1e-3  # Convert to second
        t_offset = step / 2
        t_frames = t_offset + step * np.arange(event_activation.shape[0])
        y=obj['train']['avds004-lab015-01']
        for i in t_frames:
            if i >= math.floor(y[0]) and i <= math.floor(y[1]):
                Y.append(1) # Talking
            else:
                Y.append(-1) # Not Talking
        print(y)
        X = np.array(train_data)
        Y = np.array(Y)
        X_train = X.reshape(X.shape[0], X.shape[1] * X.shape[2] * X.shape[3])
        print(X_train.shape, Y.shape)
        x_1 = []
        y_1 = []
        for i in range(0, np.array(X_train).shape[0] - 10):
            list1 = []
            for j in range(i, i + 10):
                list1.append(Y[j])
            x_1.append(list1)
            y_1.append(Y[j + 1])
        x_train = np.array(x_1)
        y_train = np.array(y_1)

plot_timeseq(torch.tensor(train_data, dtype=torch.float32),10,y,None)
plt.show(block=True)

class LSTM(nn.Module):
    def __init__(self):
        super(LSTM,self).__init__()
        self.lstm = nn.LSTM(input_size=1,hidden_size=5,num_layers=1,batch_first=True)
        self.fc1 = nn.Linear(in_features=5,out_features=1)

    def forward(self,x):
        output,_status = self.lstm(x)
        output = output[:,-1,:]
        output = self.fc1(torch.relu(output))
        return output

model = LSTM()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 150


class timeseries(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.len = x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return self.len


dataset = timeseries(x_train, y_train)

train_loader = DataLoader(dataset, shuffle=True, batch_size=256)

for i in range(epochs):
    for j,data in enumerate(train_loader):
        y_pred = model(data[:][0].view(-1,10,1)).reshape(-1)
        loss = criterion(y_pred,data[:][1])
        y_pred2 = y_pred.tolist()
        y_true = data[:][1].tolist()
        accuracy=r2_score(y_true, y_pred2)
        loss.backward()
        optimizer.step()

    if i%50 == 0:
        print(f'epoch: {i:3} loss: {loss.item():10.8f} R2 score: {accuracy:10.8f}')

test_set = timeseries(x_train,y_train)
test_pred = model(test_set[:][0].view(-1,10,1)).view(-1)
plt.plot(event_activation,label='Event')
plt.plot(test_pred.detach().numpy(),label='predicted_time')
plt.plot(test_set[:][1].view(-1),label='original_time')
plt.legend()
plt.show(block=True)