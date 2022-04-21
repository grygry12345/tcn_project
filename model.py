import torch
import torch.nn as nn
from torch import device, optim
import copy
from torchsummary import summary

# Temporal Convolutional Network only one refinment layer
class TConvNet(nn.Module):
    def __init__(self, num_layers, num_feature_maps, num_classes, input_dim):
        super(TConvNet, self).__init__()
        # input convolution
        self.conv_in = nn.Conv1d(input_dim, num_feature_maps, 1)
        # Dialeted Residual Layers
        self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(dilation=2**i, c_in=num_feature_maps, c_out=num_feature_maps)) for i in range(num_layers)])
        # output convolution
        self.conv_out = nn.Conv1d(num_feature_maps, num_classes, 1)
    
    def forward(self, x):
        # input convolution
        x = self.conv_in(x)
        # Dialeted Residual Layers
        for layer in self.layers:
            x = layer(x)
        # output convolution
        x = self.conv_out(x)
        return x

class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, c_in, c_out):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(c_in, c_out, 3, padding=dilation, dilation=dilation)
        self.batchNorm = nn.BatchNorm1d(c_out)
        self.relu = nn.ReLU()
        self.conv_1x1 = nn.Conv1d(c_out, c_out, 1)

    def forward(self, x):
        # Dilated convolution with residual connection
        out = self.conv_dilated(x)
        out = self.batchNorm(out)
        out = self.relu(out)
        out = self.conv_1x1(out)
        return out + x
    

def torch_Summary_model(num_samples, num_feature_maps, num_layers, input_dim, num_classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0  

    model = TConvNet(num_layers, num_feature_maps, num_classes, input_dim).to(device)
    summary(model, input_size=(input_dim, num_samples))


def train_validate_plot(device, epochs, learning_rate, val_test_percentage, batch_size, num_feature_maps, num_samples, num_layers, input_dim, num_classes):
    # Model    
    model = TConvNet(num_layers, num_feature_maps, num_classes, input_dim).to(device)
    
    # Random data
    train_x = torch.randn(batch_size, input_dim, num_samples, device=device)
    train_y = torch.randint(0, num_classes, (batch_size, num_samples), device=device)
    val_x = torch.randn(batch_size, input_dim, int(num_samples * (val_test_percentage / 100)), device=device)
    val_y = torch.randint(0, num_classes, (batch_size, int(num_samples * (val_test_percentage / 100))), device=device)

    # Loss and Optimizer loss cross entropy
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    val_losses = []
     
    for epoch in range(epochs):
        # Training
        optimizer.zero_grad()
        outputs = model(train_x)
        loss = criterion(outputs, train_y)
        loss.backward()
        optimizer.step()

        # Validation
        val_outputs = model(val_x)
        val_loss = criterion(val_outputs, val_y)
        train_losses.append(loss.item())
        val_losses.append(val_loss.item())

        
    # Save the model
    torch.save(model.state_dict(), 'model.ckpt')
    

     # plot the training loss and validation loss
    import matplotlib.pyplot as plt
    plt.plot(train_losses, label='Training loss')
    plt.plot(val_losses, label='Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss.png')
    plt.show()

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epochs = 1000
    learning_rate = 1e-3
    val_test_percentage = 25 # pertentage between train and validation set
    batch_size = 32 # number of samples per batch
    
    num_samples = 10000 # number of samples (Time steps)
    num_feature_maps = 4 # channel number
    num_layers = 2 # number of layers in the network
    input_dim = 16 # input dimensions
    num_classes = 2 # output classes 

    train_validate_plot(device, epochs, learning_rate, val_test_percentage, batch_size, num_feature_maps, num_samples, num_layers, input_dim, num_classes)
    torch_Summary_model(num_samples, num_feature_maps, num_layers, input_dim, num_classes)