# don't show any cases where second item of pair is 0 during training
# model fails to learn as relevant entries in embedding matrix never seen during training

import torch
import torch.nn as nn
import torch.nn.functional as F

k = 5


torch.manual_seed(0)


class Model(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.l1 = nn.Embedding(k**2, dim)
        self.embed_attribute_index = nn.Embedding(2, dim)
        self.l2 = nn.Linear(dim, k)

    def forward(self, pair_int, attribute_index):
        x = self.l1(pair_int) + self.embed_attribute_index(attribute_index)
        x = F.relu(x)
        x = self.l2(x)
        return x


model = Model()
inputs = torch.cartesian_prod(torch.arange(k), torch.arange(k), torch.arange(2))
targets = torch.where(inputs[:, 2] == 0, inputs[:, 0], inputs[:, 1])

pairs_int = inputs[:, 0] * k + inputs[:, 1]

# model(pairs_int, inputs[:, 2])
attribute_index = inputs[:, 2]

batch_sz = 4
n_steps = 10000

train_mask = ~(inputs[:, 1] == 0)
print(train_mask)
inputs_train = inputs[train_mask]
targets_train = targets[train_mask]
attribute_index_train = attribute_index[train_mask]
pairs_int_train = pairs_int[train_mask]

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.1)

for step in range(n_steps):
    optimizer.zero_grad()
    indices_sample = torch.randperm(pairs_int_train.size(0))[:batch_sz]
    pairs_int_sample = pairs_int_train[indices_sample]
    attribute_index_sample = attribute_index_train[indices_sample]
    targets_sample = targets_train[indices_sample]
    out = model(pairs_int_sample, attribute_index_sample)
    loss = F.cross_entropy(out, targets_sample)
    loss.backward()
    optimizer.step()
    if step % 100 == 0:
        print(loss.item())

out = model(pairs_int, attribute_index)
loss = F.cross_entropy(out, targets)
print(out)
print(loss.item())
