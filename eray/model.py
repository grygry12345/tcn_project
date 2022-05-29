import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision.models import  resnet152


def num_params(net):
    count = sum(p.numel() for p in net.parameters())
    return count
"""
Old Models

class RNN(nn.Module):
    def __init__(self, input_dim,output_size,lstm=False,hidden_dim=64):
        super(RNN, self).__init__()
        self.lstm = lstm
        self.hidden_dim=hidden_dim
        self.input = input_dim
        self.output_size=output_size

        if self.lstm:
            self.lstm = nn.LSTM(input_size=self.input, hidden_size=self.hidden_dim, batch_first=True,num_layers=2)
        else:
            self.gru = nn.GRU(input_size=self.input, hidden_size=self.hidden_dim, batch_first=True,num_layers=2)
        self.linear = nn.Linear(self.hidden_dim, self.output_size)

    def init_hidden(self,x):
        h = torch.zeros(2, x.size(0), self.hidden_dim).requires_grad_()
        c = torch.zeros(2, x.size(0), self.hidden_dim).requires_grad_()
        return h, c

    def forward(self, x):
        if self.lstm:
            (h, c) = self.init_hidden(x)
            output, _ = self.lstm(x, (h, c))
        else:
            output, _ = self.gru(x)
        x = F.relu(self.linear(output))
        return x

class RNN_Attention(nn.Module):
    def __init__(self,input_dim,output_size,hidden_dim=64 ,lstm=False, embedding_dim=2):
        super(RNN_Attention, self,).__init__()
        self.lstm=lstm
        self.input_dim=input_dim
        self.hidden_dim=hidden_dim
        self.embedding_dim=embedding_dim
        self.output_size=output_size

        self.embedding = nn.Embedding(self.input_dim, self.embedding_dim)
        if self.lstm==True:
            self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, bidirectional=True)
        else:
            self.gru = nn.GRU(self.embedding_dim, self.hidden_dim, bidirectional=True)
        self.out = nn.Linear(self.hidden_dim*2, output_size)

    def attention(self, output, state):
        hidden = state.view(-1, self.hidden_dim * 2 , 1)
        soft_attn_weights = F.softmax(torch.bmm(output, hidden).squeeze(2), 1)
        context = torch.bmm(output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return context, soft_attn_weights.data

    def init_hidden(self,X):
        h = Variable(torch.zeros(2, len(X), self.hidden_dim))
        c = Variable(torch.zeros(2, len(X), self.hidden_dim))
        return h, c

    def forward(self, X):
        input = self.embedding(X.to(torch.int64))
        input = input.permute(1, 0, 2)
        if self.lstm==True:
            h,c = self.init_hidden(X)
            output, (h, _) = self.lstm(input, (h, c))
        else:
            output, h = self.gru(input)
        output = output.permute(1, 0, 2)
        attn_output, _ = self.attention(output, h)
        return self.out(attn_output)
"""

class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.final = nn.Sequential(
            nn.Linear(6144, 256),
            nn.BatchNorm1d(256, momentum=0.01)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.final(x)

class GRU(nn.Module):
    def __init__(self, latent_dim, num_layers, hidden_dim, bidirectional):
        super(GRU, self).__init__()
        self.gru = nn.GRU(latent_dim, hidden_dim, num_layers, batch_first=True, bidirectional=bidirectional)

    def forward(self, x):
        x, _ = self.gru(x)
        return x

class Net(nn.Module):
    def __init__(
        self, output_dim, latent_dim=256, gru_layers=1, hidden_dim=1024, bidirectional=True):
        super(Net, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.gru = GRU(latent_dim, gru_layers, hidden_dim, bidirectional)
        self.output_layers = nn.Sequential(
            nn.Linear(2 * hidden_dim if bidirectional else hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim, momentum=0.01),
            nn.ReLU(),
            nn.Linear(hidden_dim,output_dim),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        x = self.encoder(x)
        x = x.view(batch_size, seq_length, -1)
        x = self.gru(x)
        x = x[:, -1]
        return self.output_layers(x)
