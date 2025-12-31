import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import StudentDataset

dataset = StudentDataset()
loader = DataLoader(dataset, batch_size=2, shuffle=True)

model = nn.Linear(1, 1)


loss_fn = nn.BCEWithLogitsLoss()


optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

for epoch in range(300):
    for x, y in loader:
        logits = model(x)
        loss = loss_fn(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss {loss.item():.4f}")


with torch.no_grad():
    test_score = torch.tensor([[0.65]])  # NOT 65.0
    prob = torch.sigmoid(model(test_score))
    print("Pass probability:", prob.item())
