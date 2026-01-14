import torch
import torch.nn as nn
import torch.nn.functional as F

k = 10


torch.manual_seed(0)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Embedding(k, 128)
        self.l2 = nn.Linear(128, k)

    def forward(self, x):
        x = self.l1(x)
        x = F.relu(x)
        x = self.l2(x)
        return x


model = Model()

inputs = torch.arange(k)
targets = inputs
batch_sz = 4
n_steps = 1000

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for step in range(n_steps):
    optimizer.zero_grad()
    indices_sample = torch.randperm(inputs.size(0))[:batch_sz]
    inputs_sample = inputs[indices_sample]
    targets_sample = targets[indices_sample]
    out = model(inputs_sample)
    loss = F.cross_entropy(out, targets_sample)
    loss.backward()
    optimizer.step()
    if step % 100 == 0:
        print(loss.item())


print(model(inputs))
