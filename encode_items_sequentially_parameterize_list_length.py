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

        self.l4 = nn.Linear(dim, dim * 10, bias=False)
        self.l5 = nn.Linear(dim * 10, dim, bias=False)

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

    def decoder(self, state, attribute_index):
        y = state + self.embed_attribute_index(attribute_index)
        z = self.l4(y)
        z = F.relu(z)
        z = self.l5(z)
        y = y + z
        y = self.l3(y)
        return y


model = Model().to(device)


inputs = torch.cartesian_prod(
    *[torch.arange(k) for _ in range(list_len)], torch.arange(list_len)
).to(device)
targets = inputs[torch.arange(inputs.size(0)), inputs[:, list_len]]


train_mask = ~(inputs[:, 1] == 0)
print(train_mask, train_mask.sum(), inputs.size(0))
inputs_train = inputs[train_mask]
targets_train = targets[train_mask]

batch_sz = 999999999
n_steps = 1000000

optimizer = torch.optim.AdamW(
    # model.parameters(), lr=1e-5, weight_decay=1.0, betas=(0.9, 0.95)
    model.parameters(),
    lr=3e-4,
    weight_decay=0.2,
    betas=(0.9, 0.95),
)


import os
import subprocess
from datetime import datetime
from pathlib import Path

os.makedirs("runs", exist_ok=True)

day = datetime.now().strftime("%Y%m%d")
time = datetime.now().strftime("%H%M%S")
run_prefix_name = "encode_items_sequentially"
run_dir = Path("runs") / run_prefix_name / day / time
os.makedirs(run_dir, exist_ok=True)
git_diff = subprocess.run(["git", "diff"], capture_output=True, text=True).stdout
(run_dir / "diff.patch").write_text(git_diff)

print("run_dir", run_dir)


# if os.path.exists(checkpoint):
#     checkpoint = torch.load(checkpoint, map_location=device)
#     model.load_state_dict(checkpoint["model"])
#     optimizer.load_state_dict(checkpoint["optimizer"])
#     start_step = checkpoint["step"] + 1
#     torch.random.set_rng_state(checkpoint["rng_state"])
#     print(f"resumed from step {start_step}")


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
    if step % 100 == 0:
        print(
            step, loss.item(), (out.argmax(-1) == targets_sample).float().mean().item()
        )

    if step % 100 == 0 or step == n_steps - 1:
        with torch.no_grad():
            out = model.enc_seq(inputs)
            out = model.decoder(out, inputs[:, list_len])
            val_loss = F.cross_entropy(out, targets)

            out = model.enc_seq(inputs_train)
            out = model.decoder(out, inputs_train[:, list_len])
            train_loss = F.cross_entropy(out, targets_train)

            out = model.enc_seq(inputs[~train_mask])
            out = model.decoder(out, inputs[~train_mask][:, list_len])
            test_loss = F.cross_entropy(out, targets[~train_mask])
            test_acc = (out.argmax(-1) == targets[~train_mask]).float().mean()

            print(
                step,
                f"val_loss={val_loss.item():.6g} train_loss={train_loss.item():.6g} test_loss={test_loss.item():.6g} test_acc={test_acc.item():.4f}",
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
