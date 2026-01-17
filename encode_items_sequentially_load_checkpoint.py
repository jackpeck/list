import os
from datetime import datetime

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA

device = torch.device("mps")


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

        self.l4 = nn.Linear(dim, dim * 4)
        self.l5 = nn.Linear(dim * 4, dim)

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
# print(train_mask, train_mask.sum(), inputs.size(0))
inputs_train = inputs[train_mask]
targets_train = targets[train_mask]


checkpoint_path = "runs/encode_items_sequentially/20260116/175325/step_20000.pt"


if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # exit()
    model.load_state_dict(checkpoint["model"])
    # optimizer.load_state_dict(checkpoint["optimizer"])
    start_step = checkpoint["step"] + 1
    torch.random.set_rng_state(checkpoint["rng_state"].cpu())
    # print(f"resumed from step {start_step}")


# Permute l1 embedding weights: reorder rows from 0152346 to 0123456
perm = [
    0,
    1,
    5,
    2,
    3,
    4,
    6,
]  # this is the order the values seems to be encoded in. plotting model.l1.weight shows a straight line in 3d. see plots/l1_weight_3d-20260116-175750.png
model.l1.weight.data = model.l1.weight.data[perm]
model.l3.weight.data = model.l3.weight.data[perm]

out = model.enc_seq(inputs[~train_mask])
out = model.decoder(out, inputs[~train_mask][:, list_len])
test_loss = F.cross_entropy(out, targets[~train_mask])
test_acc = (out.argmax(-1) == targets[~train_mask]).float().mean()

print(f"test_loss={test_loss.item():.6g}, test_acc={test_acc.item():.6f}")


out_enc = model.enc_seq(inputs[1:2])
out_logits = model.decoder(out_enc, inputs[1:2][:, list_len])
print(inputs[1:2])
print(out_enc)
print(out_logits, F.softmax(out_logits, dim=-1))


# embeddings = model.l1.weight.detach().cpu().numpy()
# pca = PCA(n_components=3)
# embeddings_pca = pca.fit_transform(embeddings)


# os.makedirs("plots", exist_ok=True)

# timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
# save_path = f"plots/pca-{timestamp}.png"

# plt.figure(figsize=(8, 8))
# plt.scatter(embeddings_pca[:, 1], embeddings_pca[:, 2], s=100)
# for i in range(k):
#     plt.annotate(
#         str(i),
#         (embeddings_pca[i, 1], embeddings_pca[i, 2]),
#         fontsize=12,
#         ha="center",
#         va="bottom",
#     )
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.savefig(save_path)


# print(torch.clamp(model.l2.weight, min=0))
# print(model.l2.weight)
print(model.l1.weight)

# # 3D plot of l1 embeddings
# embeddings = model.l1.weight

# embeddings2 = model.l2(embeddings)
# embeddings3 = model.l2(embeddings2)
# embeddings = torch.cat([embeddings, embeddings2, embeddings3])

embeddings = model.enc_seq(inputs[:])

state = embeddings
y = state + model.embed_attribute_index(inputs[:, list_len])
# z = model.l4(y)
# z = F.relu(z)
# z = model.l5(z)
# y = y + z
# y = z
# embeddings = y

print(inputs[27 : 27 + 4])
print(inputs[:, list_len][27 : 27 + 4])
print(targets[27 : 27 + 4])
print(embeddings[27 : 27 + 4])
# exit()

# embeddings = embeddings[:10]


# embeddings = model.embed_attribute_index.weight

# print("l3", model.l3.weight)
# embeddings = model.l3.weight

# embeddings = torch.cat([embeddings, model.l3.weight])


embeddings = embeddings.detach().cpu().numpy()


fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

# embeddings = embeddings[:10]

ax.scatter(
    embeddings[:, 0],
    embeddings[:, 1],
    embeddings[:, 2],
    s=100,
    # c=range(embeddings.shape[0]),
    # c=inputs[: embeddings.shape[0], list_len].cpu().numpy(),
    # c=targets[: embeddings.shape[0]].cpu().numpy(),
    c=torch.stack(
        [
            inputs[: embeddings.shape[0], 0],
            inputs[: embeddings.shape[0], 1],
            inputs[: embeddings.shape[0], 2],
        ],
        dim=1,
    )
    .float()
    .cpu()
    .numpy()
    / (k - 1),
    cmap="viridis",
)

# for i in range(embeddings.shape[0]):
#     # s = str(i)
#     s = str(inputs[i].cpu().numpy())
#     ax.text(embeddings[i, 0], embeddings[i, 1], embeddings[i, 2], s, fontsize=12)

ax.set_xlabel("Dim 0")
ax.set_ylabel("Dim 1")
ax.set_zlabel("Dim 2")
# ax.set_title("model.l1.weight (Embeddings)")

os.makedirs("plots", exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
save_path = f"plots/encodings-3d-{timestamp}.png"
plt.savefig(save_path)
plt.show()
