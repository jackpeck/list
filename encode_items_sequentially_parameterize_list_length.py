import torch
import torch.nn as nn
import torch.nn.functional as F

k = 5
list_len = 2

torch.manual_seed(0)


class Model(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.dim = dim

        self.l1 = nn.Embedding(k, dim)
        self.l2 = nn.Linear(dim, dim)
        self.l3 = nn.Linear(dim, k)
        self.embed_attribute_index = nn.Embedding(list_len, dim)

    def encoder(self, item, prior_state):
        x = self.l1(item)
        y = x + prior_state
        y = F.relu(y)
        y = self.l2(y)
        return y

    def enc_seq(self, items):
        initial_state = torch.zeros(items.size(0), self.dim)
        state = initial_state
        for i in range(list_len):
            state = model.encoder(items[:, i], state)
        return state

    def decoder(self, state, attribute_index):
        y = state + self.embed_attribute_index(attribute_index)
        y = F.relu(y)
        y = self.l3(y)
        return y


model = Model()


inputs = torch.cartesian_prod(
    *[torch.arange(k) for _ in range(list_len)], torch.arange(list_len)
)
targets = inputs[torch.arange(inputs.size(0)), inputs[:, list_len]]


# out = model.encoder(inputs[:, 0], torch.zeros(model.dim))
# out = model.encoder(inputs[:, 1], out)
# out = model.encoder(inputs[:, 2], out)

out = model.enc_seq(inputs)
out = model.decoder(out, inputs[:, list_len])

train_mask = ~(inputs[:, 1] == 0)
print(train_mask)
inputs_train = inputs[train_mask]
targets_train = targets[train_mask]

batch_sz = 4
n_steps = 10000

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.1)

for step in range(n_steps):
    optimizer.zero_grad()
    indices_sample = torch.randperm(inputs_train.size(0))[:batch_sz]
    inputs_sample = inputs_train[indices_sample]
    targets_sample = targets_train[indices_sample]

    out = model.enc_seq(inputs_sample)
    out = model.decoder(out, inputs_sample[:, list_len])

    loss = F.cross_entropy(out, targets_sample)
    loss.backward()
    optimizer.step()
    # if step % 100 == 0:
    #     print(step, loss.item())

    if step % 1000 == 0 or step == n_steps - 1:
        with torch.no_grad():
            out = model.enc_seq(inputs)
            out = model.decoder(out, inputs[:, list_len])
            val_loss = F.cross_entropy(out, targets)

            out = model.enc_seq(inputs_train)
            out = model.decoder(out, inputs_train[:, list_len])
            train_loss = F.cross_entropy(out, targets_train)
            print(step, "val loss", val_loss.item(), "train loss", train_loss.item())

    # out = model.enc_seq(inputs_train)
    # out = model.decoder(out, inputs_train[:, list_len])
    # loss = F.cross_entropy(out, targets_train)
    # print(step, "loss", loss.item())

# out = model.enc_seq(inputs)
# out = model.decoder(out, inputs[:, list_len])
# loss = F.cross_entropy(out, targets)
# print(out)
# print(loss.item())
