import torch.nn as nn
import copy

# Temporal Convolutional Network only one refinment layer
class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation=1, channels=4, output=1, dim=6144):
        super(DilatedResidualLayer, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, channels, kernel_size=1)
        self.conv_dilated = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=1)
        self.conv_1x1 = nn.Conv1d(channels, channels, 1)
        self.conv_out = nn.Conv1d(channels, output, 1)

    def forward(self, x):
        out = self.conv_1x1(x)
        out = self.conv_dilated(out)
        out = self.conv_1x1(out) + x
        out = self.conv_out(out)

        return out