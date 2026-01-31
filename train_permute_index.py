import os
import subprocess
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

k = 7
list_len = 3

torch.manual_seed(0)

device = torch.device("mps")


class Model(nn.Module):
    def __init__(self, dim=3):
        super().__init__()
        self.dim = dim

        self.l1 = nn.Embedding(k, dim)
        self.l2 = nn.Linear(dim, dim, bias=False)
        self.l3 = nn.Linear(dim, k, bias=False)
        self.embed_attribute_index = nn.Embedding(list_len, dim)

        self.l4 = nn.Linear(dim, dim * 4, bias=True)
        self.l5 = nn.Linear(dim * 4, dim, bias=False)

        # self.l6up = nn.Linear(dim, dim * 4, bias=True)
        # self.l6down = nn.Linear(dim * 4, dim, bias=True)
        # self.l7up = nn.Linear(dim, dim * 20, bias=True)
        # self.l7down = nn.Linear(dim * 20, dim, bias=True)
        self.l8up = nn.Linear(dim, dim * 4, bias=True)
        self.l8down = nn.Linear(dim * 4, dim, bias=True)
        # self.l4 = nn.Linear(dim, dim * 4, bias=True)
        # self.l5 = nn.Linear(dim * 4, dim, bias=False)

        self.embed_permute_index = nn.Embedding(list_len, dim * 4)

    def encoder(self, item, prior_state):
        x = self.l1(item)
        y = x + prior_state
        y = self.l2(y)
        return y

    def enc_seq(self, items):
        initial_state = torch.zeros(items.size(0), self.dim, device=device)
        state = initial_state
        for i in range(list_len):
            state = self.encoder(items[:, i], state)
        return state

    def decoder(self, state, attribute_index, permute_index):
        y = state + self.embed_attribute_index(attribute_index)
        # y = y + self.embed_permute_index(permute_index)
        z = self.l4(y)
        z = F.relu(z)
        z = self.l5(z)
        y = y + z

        # z = self.l6up(y)
        # z = F.relu(z)
        # z = self.l6down(z)
        # y = y + z

        # y = y + self.embed_permute_index(permute_index)

        # z = self.l7up(y)
        # z = F.relu(z)
        # z = self.l7down(z)
        # y = y + z

        z = self.l8up(y)
        z = z + self.embed_permute_index(permute_index)
        z = F.relu(z)
        z = self.l8down(z)
        y = y + z

        y = self.l3(y)
        return y


model = Model().to(device)


inputs = torch.cartesian_prod(
    *[torch.arange(k) for _ in range(list_len)],
    torch.arange(list_len),
    torch.arange(list_len),
).to(device)
print(inputs[100:113])

perm = torch.tensor([5, 3, 2, 4, 0, 1, 6]).to(device)
# perm = torch.randn(k).sort().indices.to(device)
# assert (perm == torch.tensor([5, 3, 2, 4, 0, 1, 6]).to(device)).all()
# inputs[torch.arange(inputs.size(0)), inputs[:, list_len + 1]] = perm[
#     inputs[torch.arange(inputs.size(0)), inputs[:, list_len + 1]]
# ]

targets = inputs[torch.arange(inputs.size(0)), inputs[:, list_len]]


target_is_permuted_index = inputs[:, list_len] == inputs[:, list_len + 1]
print(target_is_permuted_index[100:113])
targets = torch.where(target_is_permuted_index, perm[targets], targets)
print(targets[100:113])


train_mask = ~(inputs[:, 1] == 0)
print(train_mask, train_mask.sum(), inputs.size(0))
inputs_train = inputs[train_mask]
targets_train = targets[train_mask]

batch_sz = 999999999
# batch_sz = 64
n_steps = 1000000

optimizer = torch.optim.AdamW(
    model.parameters(),
    # lr=3e-4,
    lr=1e-3,
    weight_decay=0.004,
    # betas=(0.9, 0.99),
)


os.makedirs("runs", exist_ok=True)

day = datetime.now().strftime("%Y%m%d")
time = datetime.now().strftime("%H%M%S")
run_prefix_name = "encode_items_sequentially"
run_dir = Path("runs") / run_prefix_name / day / time
os.makedirs(run_dir, exist_ok=True)
git_diff = subprocess.run(["git", "diff"], capture_output=True, text=True).stdout
(run_dir / "diff.patch").write_text(git_diff)

print("run_dir", run_dir)

for step in range(n_steps):
    optimizer.zero_grad()
    indices_sample = torch.randperm(inputs_train.size(0))[:batch_sz]
    inputs_sample = inputs_train[indices_sample]
    targets_sample = targets_train[indices_sample]

    out = model.enc_seq(inputs_sample)
    out = model.decoder(out, inputs_sample[:, list_len], inputs_sample[:, list_len + 1])

    loss = F.cross_entropy(out, targets_sample)
    loss.backward()
    optimizer.step()
    if step % 100 == 0:
        print(
            step, loss.item(), (out.argmax(-1) == targets_sample).float().mean().item()
        )

    if step % 100 == 0 or step == n_steps - 1:
        with torch.no_grad():
            out = model.enc_seq(inputs)
            out = model.decoder(out, inputs[:, list_len], inputs[:, list_len + 1])
            val_loss = F.cross_entropy(out, targets)

            out = model.enc_seq(inputs_train)
            out = model.decoder(
                out, inputs_train[:, list_len], inputs_train[:, list_len + 1]
            )
            train_loss = F.cross_entropy(out, targets_train)

            out = model.enc_seq(inputs[~train_mask])
            out = model.decoder(
                out,
                inputs[~train_mask][:, list_len],
                inputs[~train_mask][:, list_len + 1],
            )
            test_loss = F.cross_entropy(out, targets[~train_mask])
            test_acc = (out.argmax(-1) == targets[~train_mask]).float().mean()

            print(
                step,
                f"val_loss={val_loss.item():.6g} train_loss={train_loss.item():.6g} test_loss={test_loss.item():.6g} test_acc={test_acc.item():.4f}",
                # f"test_loss={test_loss.item():.6g} test_acc={test_acc.item():.4f}",
            )

        checkpoint_path = run_dir / f"step_{step}.pt"

        if step > 0 and step % 10000 == 0 or step == n_steps - 1:
            torch.save(
                {
                    "step": step,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "rng_state": torch.random.get_rng_state(),
                },
                checkpoint_path,
            )
