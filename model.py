import torch
import torch.nn as nn
import torch.nn.functional as F

# Temporal Convolutional Network dialutional layer
class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation=1, output_channels=1, input_channels=1, filter_size=4):
        super(DilatedResidualLayer, self).__init__()
        self.conv_in = nn.Conv1d(input_channels, filter_size, 1)
        self.conv_dilated = nn.Conv1d(filter_size, filter_size, 3, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(filter_size, filter_size, 1)
        self.dropout = nn.Dropout()
        self.conv_out = nn.Conv1d(filter_size, output_channels, 1)

    def forward(self, x):
        out = x.unsqueeze(2)

        out = self.conv_in(out)
        out = F.relu(self.conv_dilated(out))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        out = self.conv_out(out)

        out = out.squeeze(2)

        return out
        # return x + out # will be used when using multiple dilated layers
