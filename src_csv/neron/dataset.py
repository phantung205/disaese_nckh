import torch
from torch.utils.data import TensorDataset, DataLoader
from configs import config_csv


def create_dataloader(x,y,batch_size=config_csv.batch_size,shuffle=True):
    # DataFrame / Series -> NumPy
    if hasattr(x, "to_numpy"):
        x = x.to_numpy()

    if hasattr(y, "to_numpy"):
        y = y.to_numpy()

    # NumPy -> Tensor
    x_tensor = torch.tensor(x,dtype=torch.float32)

    y_tensor = torch.tensor(y,dtype=torch.float32).reshape(-1, 1)

    dataset = TensorDataset(x_tensor,y_tensor)

    dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=shuffle)

    return dataloader