import torch
import torch.nn as nn
import torch.nn.functional as F

k = 5


class Model(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.dim = dim

        self.l1 = nn.Embedding(k, dim)
        self.l2 = nn.Linear(dim, dim)
        self.l3 = nn.Linear(dim, k)
        self.embed_attribute_index = nn.Embedding(2, dim)

    def encoder(self, item, prior_state):
        x = self.l1(item)
        y = x + prior_state
        y = F.relu(y)
        y = self.l2(y)
        return y

    def decoder(self, state, attribute_index):
        y = state + self.embed_attribute_index(attribute_index)
        y = F.relu(y)
        y = self.l3(y)
        return y


model = Model()


inputs = torch.cartesian_prod(torch.arange(k), torch.arange(k), torch.arange(2))
targets = torch.where(inputs[:, 2] == 0, inputs[:, 0], inputs[:, 1])

# pairs_int = inputs[:, 0] * k + inputs[:, 1]
print(inputs)

out = model.encoder(inputs[:, 0], torch.zeros(model.dim))
out = model.encoder(inputs[:, 1], out)
out = model.decoder(out, inputs[:, 2])

train_mask = ~(inputs[:, 1] == 0)
print(train_mask)
inputs_train = inputs[train_mask]
targets_train = targets[train_mask]

batch_sz = 4
n_steps = 100000

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.1)

for step in range(n_steps):
    optimizer.zero_grad()
    indices_sample = torch.randperm(inputs_train.size(0))[:batch_sz]
    inputs_sample = inputs_train[indices_sample]
    targets_sample = targets_train[indices_sample]

    out = model.encoder(inputs_sample[:, 0], torch.zeros(model.dim))
    out = model.encoder(inputs_sample[:, 1], out)
    out = model.decoder(out, inputs_sample[:, 2])

    loss = F.cross_entropy(out, targets_sample)
    loss.backward()
    optimizer.step()
    # if step % 100 == 0:
    #     print(step, loss.item())

    if step % 1000 == 0 or step == n_steps - 1:
        with torch.no_grad():
            out = model.encoder(inputs[:, 0], torch.zeros(model.dim))
            out = model.encoder(inputs[:, 1], out)
            out = model.decoder(out, inputs[:, 2])
            val_loss = F.cross_entropy(out, targets)

            out = model.encoder(inputs_train[:, 0], torch.zeros(model.dim))
            out = model.encoder(inputs_train[:, 1], out)
            out = model.decoder(out, inputs_train[:, 2])
            train_loss = F.cross_entropy(out, targets_train)
            print(step, "val loss", val_loss.item(), "train loss", train_loss.item())

    # out = model.encoder(inputs_train[:, 0], torch.zeros(model.dim))
    # out = model.encoder(inputs_train[:, 1], out)
    # out = model.decoder(out, inputs_train[:, 2])
    # loss = F.cross_entropy(out, targets_train)
    # print(step, "loss", loss.item())

# out = model.encoder(inputs[:, 0], torch.zeros(model.dim))
# out = model.encoder(inputs[:, 1], out)
# out = model.decoder(out, inputs[:, 2])
# loss = F.cross_entropy(out, targets)
# print(out)
# print(loss.item())
