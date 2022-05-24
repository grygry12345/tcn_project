import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


def num_params(net):
    count = sum(p.numel() for p in net.parameters())
    return count

class RNN(nn.Module):
    def __init__(self, input_dim,output_size,lstm=False,hidden_dim=64):
        super(RNN, self).__init__()
        self.lstm = lstm
        self.hidden_dim=hidden_dim
        self.input = input_dim
        self.output_size=output_size

        if self.lstm:
            self.lstm = nn.LSTM(input_size=self.input, hidden_size=self.hidden_dim, num_layers=1)
        else:
            self.gru = nn.GRU(input_size=self.input, hidden_size=self.hidden_dim, num_layers=1)
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


class CNN(nn.Module):
    def __init__(self, input_size, output_size,hidden_size=64, n_layers=1):
        super(CNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.c1 = nn.Conv1d(input_size, hidden_size, 1)
        self.c2 = nn.Conv1d(hidden_size, hidden_size, 1)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=0.01)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, inputs):
        inputs = inputs.unsqueeze(2)
        c = self.c1(inputs)
        c = self.c2(c)
        p = c.squeeze(2)
        output, _ = self.gru(p)
        output = F.relu(self.out(output))
        return output