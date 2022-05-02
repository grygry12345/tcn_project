import json
import math
from datetime import datetime
import h5py
from sklearn.metrics import r2_score
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset
import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision.models import resnet152

writer = SummaryWriter()

def plot_timeseq(eventspiketensor, frame_size_ms, segment, name):
    event_activation = eventspiketensor.sum(axis=(1, 2, 3))
    step = frame_size_ms * 1e-6  # Convert to millisecond
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


def t_frames(event_activation,values):
    Y = []
    step = 10 * 1e-6  # Convert to millisecond
    t_offset = step / 2
    t_frames = t_offset + step * np.arange(event_activation.shape[0])
    for i in t_frames:
        if i >= math.floor(values[0]) and i <= math.floor(values[1]):
            Y.append(1)  # Talking
        else:
            Y.append(-1)  # Not Talking
    Y = np.array(Y)
    return Y


with open('sets.json', 'r') as myfile:
    data=myfile.read()
    obj = json.loads(data)
    # train values
    for element in obj['train']:
        with h5py.File("roi_mouth/avds004-lab015-01.h5", "r") as f:
            a_group_key = list(f.keys())[0]
            train_data=np.array(list(f[a_group_key]))
            train_values = obj['train']['avds004-lab015-01']
            train_event_activation = train_data.sum(axis=(1, 2, 3))
            y_train = t_frames(train_event_activation,train_values)
            x_train=train_data.reshape(train_data.shape[0], train_data.shape[1] * train_data.shape[2] * train_data.shape[3])
    # test values
    for element in obj['test']:
        with h5py.File("roi_mouth/avds000-lab022-02.h5", "r") as f:
            a_group_key = list(f.keys())[0]
            test_data=np.array(list(f[a_group_key]))
            test_values = obj['train']['avds000-lab022-02']
            test_event_activation = test_data.sum(axis=(1, 2, 3))
            y_test = t_frames(test_event_activation,test_values)
            x_test=test_data.reshape(test_data.shape[0], test_data.shape[1] * test_data.shape[2] * test_data.shape[3])
    #validation values
    for element in obj['validation']:
        with h5py.File("roi_mouth/avds000-lab022-02.h5", "r") as f:
            a_group_key = list(f.keys())[0]
            valid_data=np.array(list(f[a_group_key]))
            valid_values = obj['train']['avds000-lab022-02']
            valid_event_activation = valid_data.sum(axis=(1, 2, 3))
            y_valid = t_frames(valid_event_activation,valid_values)
            x_valid=valid_data.reshape(valid_data.shape[0], valid_data.shape[1] * valid_data.shape[2] * valid_data.shape[3])

#plot example values in train data
plot_timeseq(torch.tensor(train_data, dtype=torch.float32),10,train_values,None)
plt.show(block=True)
plt.plot(train_event_activation/100,label='Event')
plt.plot(np.array(y_train).view(-1),label='original_time')
plt.legend()
plt.show(block=True)

class LSTM(nn.Module):
    def __init__(self,input_dim=1,hidden_dim=5,num_layers=1,lstm=False,dropout=0.2):
        super(LSTM,self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        # GRU layers & LSTM layers
        if lstm:
          self.rnn = nn.LSTM(input_size=input_dim,hidden_size=hidden_dim,num_layers=num_layers,batch_first=True,dropout=dropout)
        else:
          self.rnn = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True,dropout=dropout)
        # Fully connected layer
        self.fc1 = nn.Linear(in_features=hidden_dim,out_features=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self,x):
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        # Forward propagation by passing in the input and hidden state into the model
        output,_status = self.rnn(x, h0.detach())
        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        output = output[:,-1,:]
        # Convert the final state to our desired output shape (batch_size, output_dim)
        output = self.fc1(torch.relu(output))
        return output

class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        resnet = resnet152(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.final = nn.Sequential(
            nn.Linear(resnet.fc.in_features, latent_dim), nn.BatchNorm1d(latent_dim, momentum=0.01)
        )

    def forward(self, x):
        with torch.no_grad():
            x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        return self.final(x)

class LSTM(nn.Module):
    def __init__(self, latent_dim, num_layers, hidden_dim, bidirectional):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(latent_dim, hidden_dim, num_layers, batch_first=True, bidirectional=bidirectional)
        self.hidden_state = None

    def reset_hidden_state(self):
        self.hidden_state = None

    def forward(self, x):
        x, self.hidden_state = self.lstm(x, self.hidden_state)
        return x

class ConvLSTM(nn.Module):
    def __init__(
        self, latent_dim=512, lstm_layers=1, hidden_dim=1024, bidirectional=True):
        super(ConvLSTM, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.lstm = LSTM(latent_dim, lstm_layers, hidden_dim, bidirectional)
        self.output_layers = nn.Sequential(
            nn.Linear(2 * hidden_dim if bidirectional else hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim, momentum=0.01),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        x = self.encoder(x)
        x = x.view(batch_size, seq_length, -1)
        x = self.lstm(x)
        x = x[:, -1]
        return self.output_layers(x)

batch_size = 64

#Features
train_features = torch.Tensor(x_train)
train_targets = torch.Tensor(y_train)
val_features = torch.Tensor(x_valid)
val_targets = torch.Tensor(y_valid)
test_features = torch.Tensor(x_test)
test_targets = torch.Tensor(y_test)


train = TensorDataset(train_features, train_targets)
val = TensorDataset(val_features, val_targets)
test = TensorDataset(test_features, test_targets)

train_loader = DataLoader(train, batch_size=batch_size, shuffle=False, drop_last=True)
val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, drop_last=True)
test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, drop_last=True)
test_loader_one = DataLoader(test, batch_size=1, shuffle=False, drop_last=True)


class Optimization:
    def __init__(self, model, loss_fn, optimizer):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_losses = []
        self.val_losses = []

    def train_step(self, x, y):
        # Sets model to train mode
        self.model.train()

        # Makes predictions
        yhat = self.model(x)

        # Computes loss
        loss = self.loss_fn(y, yhat)

        # Computes gradients
        loss.backward()

        # Updates parameters and zeroes gradients
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Returns the loss
        return loss.item()
    def train(self,train_loader, val_loader, batch_size=64, n_epochs=50, n_features=1):
        model_path = f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'


        for epoch in range(1, n_epochs + 1):
            batch_losses = []
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.view([batch_size, -1, n_features])
                y_batch = y_batch
                loss = self.train_step(x_batch, y_batch)
                batch_losses.append(loss)
            training_loss = np.mean(batch_losses)
            self.train_losses.append(training_loss)

            with torch.no_grad():
                batch_val_losses = []
                for x_val, y_val in val_loader:
                    x_val = x_val.view([batch_size, -1, n_features])
                    y_val = y_val
                    self.model.eval()
                    yhat = self.model(x_val)
                    val_loss = self.loss_fn(y_val, yhat).item()
                    batch_val_losses.append(val_loss)
                validation_loss = np.mean(batch_val_losses)
                self.val_losses.append(validation_loss)

            #TensorBoard
            writer.add_scalar('Loss/train', training_loss, epoch)
            writer.add_scalar('Loss/validation', val_loss, epoch)
            if (epoch <= 10) | (epoch % 50 == 0):
                print(
                    f"[{epoch}/{n_epochs}] Training loss: {training_loss:.4f}\t Validation loss: {validation_loss:.4f}"
                )

    def evaluate(self, test_loader, batch_size=1, n_features=1):
        with torch.no_grad():
            predictions = []
            values = []
            for x_test, y_test in test_loader:
                x_test = x_test.view([batch_size, -1, n_features])
                y_test = y_test
                self.model.eval()
                yhat = self.model(x_test)
                predictions.append(yhat.detach().numpy())
                values.append(y_test.detach().numpy())

        return predictions, values

    def plot_losses(self):
        plt.plot(self.train_losses, label="Training loss")
        plt.plot(self.val_losses, label="Validation loss")
        plt.legend()
        plt.title("Losses")
        plt.show()
        plt.close()

input_dim = 1
output_dim = 1
hidden_dim = 64
layer_dim = 1
batch_size = 64
dropout = 0.2
n_epochs = 10
learning_rate = 1e-3
weight_decay = 1e-6

model = LSTM()

loss_fn = nn.MSELoss(reduction="mean")
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
opt = Optimization(model=model, loss_fn=loss_fn, optimizer=optimizer)
opt.train(train_loader, val_loader, batch_size=batch_size, n_epochs=n_epochs, n_features=input_dim)
opt.plot_losses()
predictions, values = opt.evaluate(test_loader, batch_size=1, n_features=input_dim)
#plot example values in train data
print(np.array(predictions).view(-1))
print(np.array(y_test).view(-1))

plt.plot(test_event_activation/100,label='Event')
plt.plot(np.array(predictions).view(-1),label='predicted_time')
plt.plot(np.array(y_test).view(-1),label='original_time')
plt.legend()
plt.show(block=True)
