import torch
from torch.utils.data import Dataset

class StudentDataset(Dataset):
    def __init__(self):
        # Normalize inputs (VERY IMPORTANT)
        self.x = torch.tensor([[30.], [40.], [50.], [60.], [70.], [80.], [90.]]) / 100.0
        self.y = torch.tensor([[0.], [0.], [0.], [1.], [1.], [1.], [1.]])

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
