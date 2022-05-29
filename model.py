import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, num_input=2, hidden_size=4):
        super(CNN, self).__init__()
        self.conv_1 = nn.Conv2d(num_input, hidden_size, kernel_size=3, padding=1)
        self.maxpool_1 = nn.MaxPool2d(kernel_size=2, stride=4)
        self.conv_2 = nn.Conv2d(hidden_size, hidden_size * 2, kernel_size=3, padding=1)
        # Global average pooling
        self.gap = nn.AdaptiveAvgPool3d(1)
    
    def forward(self, x):
        out = F.relu(self.conv_1(x))
        out = self.maxpool_1(out)
        out = F.relu(self.conv_2(out))

        # Reshape N x C x H x W -> C X N X H X W
        out = out.permute(1, 0, 2, 3)
        out = self.gap(out)

        # sqeeze C X 1 X 1 X 1 -> C
        out = out.squeeze(1)
        out = out.squeeze(1)
        out = out.squeeze(1)

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
