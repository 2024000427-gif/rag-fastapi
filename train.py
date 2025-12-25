import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import SimpleDataset

dataset = SimpleDataset()
loader = DataLoader(dataset, batch_size=2, shuffle=True)

model = nn.Linear(1, 1)
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(200):
    for x, y in loader:
        y_pred = model(x)
        loss = loss_fn(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss {loss.item()}")

print("Final Weight:", model.weight.item())
print("Final Bias:", model.bias.item())
