import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

class ResCNN(nn.Module):
    def __init__(self, num_input=2, hidden_size=4, num_layers=1):
        super(ResCNN, self).__init__()
        self.conv_in = nn.Conv2d(num_input, hidden_size, kernel_size=3, padding=1)
        self.maxpool_in = nn.MaxPool2d(kernel_size=2, stride=4)
        
        self.layers = nn.ModuleList([deepcopy(CNN(hidden_size)) for _ in range(num_layers)])
        
        self.conv_res = nn.Conv2d(num_input, hidden_size, kernel_size=1)
        
        # Global average pooling
        self.gap = nn.AdaptiveAvgPool3d(1)
    
    def forward(self, x):
        connection = x
        connection = F.relu(self.conv_res(connection))

        out = F.relu(self.conv_in(x))
        out = self.maxpool_in(out)
        for layers in self.layers:
            out = F.relu(layers(out))
        
        out = out.permute(1, 0, 2, 3)
        out = self.gap(out)
        out = out.squeeze(1)
        out = out.squeeze(1)
        out = out.squeeze(1)

        connection = connection.permute(1, 0, 2, 3)
        connection = self.gap(connection)
        connection = connection.squeeze(1)
        connection = connection.squeeze(1)
        connection = connection.squeeze(1)

        out = out + connection

        return out

class CNN(nn.Module):
    def __init__(self, hidden_size=4):
        super(CNN, self).__init__()
        self.conv_in = nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1) 
            
    def forward(self, x):
        out = F.relu(self.conv_in(x))

        return out



class Linear(nn.Module):
    def __init__(self, num_input=2, hidden_size=4, num_output=2):
        super(Linear, self).__init__()
        self.linear_1 = nn.Linear(num_input, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, num_output)
    
    def forward(self, x):
        out = F.relu(self.linear_1(x))
        out = self.linear_2(out)
        return out

class BasicLinear(nn.Module):
    def __init__(self, num_input=2, hidden_size=2):
        super(BasicLinear, self).__init__()
        self.linear_1 = nn.Linear(num_input, hidden_size)
        # Global average pooling
        self.gap = nn.AdaptiveAvgPool3d(1)
    
    def forward(self, x):
        
        out = x.permute(1, 0, 2, 3)
        out = self.gap(out)
        out = out.squeeze(1)
        out = out.squeeze(1)
        out = out.squeeze(1)

        out = self.linear_1(out)
       
        return out