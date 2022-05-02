import torch
from torch.utils.data import DataLoader
import model as m  
import torch.nn as nn

from dataset import HDF5Dataset

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 64
    lr = 1e-3
    epochs = 100

    training_data = HDF5Dataset(file_path='data', group='train', file_name='avds000-lab010-01')
    training_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    frame_number = training_loader.dataset.data.shape[0]
    frame_features = training_loader.dataset.data.shape[1]

    model = m.DilatedResidualLayer(channels=4, output=5, dim=frame_features).to(device)
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):   
        size = len(training_loader.dataset)
        model.train()
        for batch, (X, y) in enumerate(training_loader):
            X = X.to(device) 
            y = y.type(torch.LongTensor)
            y = y.to(device)

            pred = model(X)
            loss = loss_fn(pred.squeeze(), y.squeeze())
            

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoach {epoch+1} Loss: {loss.item()}")
        
        
