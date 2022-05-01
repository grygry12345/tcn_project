import torch
import torch.nn as nn
import torch.nn.functional as F

# Temporal Convolutional Network only one refinment layer
class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation=1, channels=4, output=1, dim=6144):
        super(DilatedResidualLayer, self).__init__()
        self.conv_input = nn.Conv1d(dim, channels, kernel_size=1)
        self.conv_dilated = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=1)
        self.conv_1x1 = nn.Conv1d(channels, channels, 1)
        self.conv_out = nn.Conv1d(channels, output, 1)

        self.shortcut = nn.Conv1d(dim, channels, 1) # TODO write a identity if else statemnent

    def forward(self, x):
        
        x = torch.unsqueeze(x, 1)
        x = x.permute(0, 2, 1)
        
        residual = self.shortcut(x)
        
        out = self.conv_input(x)
        out = F.relu(self.conv_dilated(out))
        out = self.conv_1x1(out) + residual

        return out