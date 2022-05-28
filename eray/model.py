import torch.nn as nn
import torch.nn.functional as F
import copy
class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation=1, num_output=1, num_input=2):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(num_input, num_output, 3, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(num_output, num_output, 1)
        self.dropout = nn.Dropout()


        self.shortcut = nn.Conv1d(num_input, num_output, 1)

    def forward(self, x):
                
        residual = self.shortcut(x)
        
        out = F.relu(self.conv_dilated(x))
        out = self.dropout(out)
        out = self.conv_1x1(out) + residual

        return out

class SS_TCN(nn.Module): # 
    def __init__(self, num_layers, filter_size, num_input, num_classes):
        super(SS_TCN, self).__init__()
        self.conv_1x1 = nn.Conv1d(num_input, filter_size, 1)
        self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(2 ** i, filter_size, filter_size)) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(filter_size, num_classes, 1)

    def forward(self, x):
        out = x.unsqueeze(2)

        out = self.conv_1x1(out)
        for layer in self.layers:
            out = layer(out)
        
        out = self.conv_out(out) 

        out = out.squeeze(2)

        return out