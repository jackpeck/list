import os
from datetime import datetime

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA

k = 7
list_len = 3

torch.manual_seed(0)

device = torch.device("cpu")


class Model(nn.Module):
    def __init__(self, dim=3):
        super().__init__()
        self.dim = dim

        self.l1 = nn.Embedding(k, dim)
        self.l2 = nn.Linear(dim, dim, bias=False)
        self.l3 = nn.Linear(dim, k, bias=False)
        self.embed_attribute_index = nn.Embedding(list_len, dim)

        up_mult = 4
        self.l4 = nn.Linear(dim, dim * up_mult, bias=True)
        self.l5 = nn.Linear(dim * up_mult, dim, bias=False)

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


# checkpoint_path = "runs/encode_items_sequentially/20260116/175325/step_20000.pt"
# checkpoint_path = "runs/encode_items_sequentially/20260119/140951/step_120000.pt"
# checkpoint_path = "runs/encode_items_sequentially/20260119/152649/step_20000.pt"
checkpoint_path = "runs/encode_items_sequentially/20260119/153356/step_230000.pt"


if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # exit()
    model.load_state_dict(checkpoint["model"])
    # optimizer.load_state_dict(checkpoint["optimizer"])
    start_step = checkpoint["step"] + 1
    torch.random.set_rng_state(checkpoint["rng_state"].cpu())
    # print(f"resumed from step {start_step}")


# Permute l1 embedding weights: reorder rows from 0152346 to 0123456
# perm = [
#     0,
#     1,
#     5,
#     2,
#     3,
#     4,
#     6,
# ]  # this is the order the values seems to be encoded in. plotting model.l1.weight shows a straight line in 3d. see plots/l1_weight_3d-20260116-175750.png
# perm = [0, 1, 2, 5, 3, 4, 6]
perm = [1, 0, 5, 2, 3, 4, 6]
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
# print(model.l1.weight)

# # 3D plot of l1 embeddings
# embeddings = model.l1.weight

# embeddings2 = model.l2(embeddings)
# embeddings3 = model.l2(embeddings2)
# embeddings = torch.cat([embeddings, embeddings2, embeddings3])

embeddings = model.enc_seq(inputs[:])

state = embeddings
y = state + model.embed_attribute_index(inputs[:, list_len])
z = model.l4(y)
z = F.relu(z)
z = model.l5(z)
y = y + z
# # y = z
embeddings = y


mask = inputs[:, list_len] == 0
# mask = inputs[:, list_len] != 99
# mask = torch.ones(embeddings.shape[0]).bool()

# print(inputs[27 : 27 + 4])
# print(inputs[:, list_len][27 : 27 + 4])
# print(targets[27 : 27 + 4])
# print(embeddings[27 : 27 + 4])
# exit()

# embeddings = embeddings[:10]


# embeddings = model.embed_attribute_index.weight

# print("l3", model.l3.weight)
# embeddings = model.l3.weight

# embeddings = torch.cat([embeddings, model.l3.weight])


embeddings = embeddings.detach().cpu().numpy()
mask = mask.cpu().numpy()

print(embeddings.shape)


# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection="3d")

# # embeddings = embeddings[:10]
# ax.scatter(
#     embeddings[mask][:, 0],
#     embeddings[mask][:, 1],
#     embeddings[mask][:, 2],
#     s=100,
#     # c=range(embeddings.shape[0]),
#     # c=inputs[: embeddings.shape[0], list_len].cpu().numpy(),
#     # c=targets[mask][: embeddings[mask].shape[0]].cpu().numpy(),
#     c=targets[mask].cpu().numpy(),
#     # c=torch.stack(
#     #     [
#     #         inputs[: embeddings.shape[0], 0],
#     #         inputs[: embeddings.shape[0], 1],
#     #         inputs[: embeddings.shape[0], 2],
#     #     ],
#     #     dim=1,
#     # )
#     # .float()
#     # .cpu()
#     # .numpy()
#     # / (k - 1),
#     cmap="viridis",
# )

# # for i in range(embeddings.shape[0]):
# #     s = str(i)
# #     # s = str(inputs[i].cpu().numpy())
# #     ax.text(embeddings[i, 0], embeddings[i, 1], embeddings[i, 2], s, fontsize=12)

# ax.set_xlabel("Dim 0")
# ax.set_ylabel("Dim 1")
# ax.set_zlabel("Dim 2")
# # ax.set_title("model.l1.weight (Embeddings)")

# os.makedirs("plots", exist_ok=True)
# timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
# save_path = f"plots/encodings-3d-{timestamp}.png"
# # plt.savefig(save_path)
# plt.show()


# # Compute y before and after MLP
# y_before = state + model.embed_attribute_index(inputs[:, list_len])
# z = model.l4(y_before)
# z = F.relu(z)
# z = model.l5(z)
# y_after = y_before + z

# y_before_np = y_before.detach().cpu().numpy()
# y_after_np = y_after.detach().cpu().numpy()

# y_before_np = y_before_np[mask]
# y_after_np = y_after_np[mask] / 50

# # Plot lines showing transformation
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection="3d")


# targets_sub = targets[mask].cpu().numpy()
# norm = mcolors.Normalize(vmin=min(targets_sub), vmax=max(targets_sub))
# cmap = plt.get_cmap("viridis")

# # Draw lines from before to after for each point
# for i in range(y_before_np.shape[0]):
#     ax.plot(
#         [y_before_np[i, 0], y_after_np[i, 0]],
#         [y_before_np[i, 1], y_after_np[i, 1]],
#         [y_before_np[i, 2], y_after_np[i, 2]],
#         # c="gray",
#         c=cmap(norm(targets_sub[i])),
#         alpha=0.3,
#         linewidth=0.5,
#     )

# # Scatter before points
# ax.scatter(
#     y_before_np[:, 0],
#     y_before_np[:, 1],
#     y_before_np[:, 2],
#     s=30,
#     # c="blue",
#     c=targets[mask].cpu().numpy(),
#     alpha=0.6,
#     label="before MLP",
# )

# # Scatter after points
# ax.scatter(
#     y_after_np[:, 0],
#     y_after_np[:, 1],
#     y_after_np[:, 2],
#     s=30,
#     # c="red",
#     c=targets[mask].cpu().numpy(),
#     alpha=0.6,
#     label="after MLP",
# )

# ax.set_xlabel("Dim 0")
# ax.set_ylabel("Dim 1")
# ax.set_zlabel("Dim 2")
# ax.legend()

# # make aspect ratio the same
# all_points = np.vstack([y_before_np[:, :3], y_after_np[:, :3]])
# max_range = np.ptp(all_points, axis=0).max() / 2
# mid = all_points.mean(axis=0)
# ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
# ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
# ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

# plt.show()


# print("l4", model.l4.weight)
# print(model.l5.weight)


y_before = state + model.embed_attribute_index(inputs[:, list_len])
# z = model.l4(y_before)
# print(model.l4.weight.shape, y_before.shape)
# print(z[mask][:10])
# print(z[mask].shape)
z = y_before @ model.l4.weight.T + model.l4.bias


for i in [0, 1, 4, 6, 8]:
    z[:, i] = 0
# z[:, 10] = 0

# z[:, 1] = 0
# z[:, 4] = 0
# z[:, 6] = 0
# z[:, 8] = 0
# z[:, 10] = z[:, 10] * 10
# z[:, 2] = 0


# plt.plot(z[mask].detach().cpu() > 0)
sns.heatmap(z[mask].detach().cpu())
# sns.heatmap(z[mask].detach().cpu())
# sns.heatmap(
#     torch.cat([z[mask].detach().cpu(), targets[mask].unsqueeze(-1).cpu() * 10], dim=-1)
# )

# sns.heatmap(torch.cat([targets[mask].unsqueeze(-1).cpu()], dim=-1))
# print(
#     torch.cat([z[mask].detach().cpu(), targets[mask].unsqueeze(-1).cpu()], dim=-1).shape
# )
# plt.plot(targets[mask].detach().cpu(), label="targets")


# z = y_before @ model.l4.weight.T + model.l4.bias
# for i in [0, 1, 4, 6, 8]:
#     z[:, i] = 0
# z = z @ model.l5.weight.T
# y = y_before + z

live_indices_mask = torch.ones(
    model.l4.weight.size(0), dtype=torch.bool, device=model.l4.weight.device
)
live_indices_mask[[0, 1, 4, 6, 8]] = False

l4_linear_for_group_0 = model.l4.weight[live_indices_mask]
l5_linear_for_group_0 = model.l5.weight[:, live_indices_mask]
print(l4_linear_for_group_0.shape, l5_linear_for_group_0.shape)


# z = y_before @ l4_linear_for_group_0.T + model.l4.bias[live_indices_mask]
# z = z @ l5_linear_for_group_0.T
# y = y_before + z

# z = (
#     y_before @ l4_linear_for_group_0.T + model.l4.bias[live_indices_mask]
# ) @ l5_linear_for_group_0.T
# y = y_before + z


# z = (
#     y_before @ (l4_linear_for_group_0.T @ l5_linear_for_group_0.T)
#     + model.l4.bias[live_indices_mask] @ l5_linear_for_group_0.T
# )
# y = y_before + z

l5_l4_combined_w = l4_linear_for_group_0.T @ l5_linear_for_group_0.T + torch.eye(
    3, device=device
)
l5_l4_combined_bias = model.l4.bias[live_indices_mask] @ l5_linear_for_group_0.T

z = y_before @ l5_l4_combined_w + l5_l4_combined_bias
y = z

print(l5_l4_combined_w)
# exit()


svd = torch.linalg.svd(l5_l4_combined_w.cpu())

eig = torch.linalg.eig(l5_l4_combined_w.cpu())

assert (eig.eigenvectors.imag == 0).all()
assert (eig.eigenvalues.imag == 0).all()

print(eig)

print(
    eig.eigenvectors.real
    @ torch.diag(eig.eigenvalues.real)
    @ eig.eigenvectors.real.inverse()
)

# print(svd.U @ torch.diag(svd.S) @ svd.Vh)

# svd.S[0] = 0


eig.eigenvalues[0] = 0

# z = y_before @ svd.U @ torch.diag(svd.S) @ svd.Vh + l5_l4_combined_bias
z = (
    y_before
    @ eig.eigenvectors.real
    @ torch.diag(eig.eigenvalues.real)
    @ eig.eigenvectors.real.inverse()
    + l5_l4_combined_bias
)
y = z

embeddings = y

# # state = embeddings
# # y = state + model.embed_attribute_index(inputs[:, list_len])
# # z = model.l4(y)
# z = F.relu(z)
# z = model.l5(z)
# z = z @ model.l5.weight.T
# y = y_before + z
y = model.l3(y)
print((y.argmax(-1) == targets)[mask].float().mean())


embeddings = embeddings.detach().cpu()
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

# embeddings = embeddings[:10]
ax.scatter(
    embeddings[mask][:, 0],
    embeddings[mask][:, 1],
    embeddings[mask][:, 2],
    s=100,
    # c=range(embeddings.shape[0]),
    # c=inputs[: embeddings.shape[0], list_len].cpu().numpy(),
    # c=targets[mask][: embeddings[mask].shape[0]].cpu().numpy(),
    c=targets[mask].cpu().numpy(),
    # c=torch.stack(
    #     [
    #         inputs[: embeddings.shape[0], 0],
    #         inputs[: embeddings.shape[0], 1],
    #         inputs[: embeddings.shape[0], 2],
    #     ],
    #     dim=1,
    # )
    # .float()
    # .cpu()
    # .numpy()
    # / (k - 1),
    cmap="viridis",
)
ax.set_xlabel("Dim 0")
ax.set_ylabel("Dim 1")
ax.set_zlabel("Dim 2")
plt.show()


# plt.legend()
# plt.show()

# print(model.l4.bias)
# print(model.l4.weight.shape)
# print(model.l4.weight.inverse())


# # print(model.l5.weight @ model.l4.weight)
# # print(
# #     model.l5.weight[:, [0, 2, 3, 5, 7, 9, 10, 11]]
# #     @ model.l4.weight[[0, 2, 3, 5, 7, 9, 10, 11]]
# # )
# # # print(model.l5.weight[:, [0, 2, 3, 5, 7, 9, 10, 11]].shape)


# # mlp_linear_for_index_0 = (
# #     model.l5.weight[:, [0, 2, 3, 5, 7, 9, 10, 11]]
# #     @ model.l4.weight[[0, 2, 3, 5, 7, 9, 10, 11]]
# # )

# # print((y_before @ mlp_linear_for_index_0).shape)


# z = model.l4(y_before)
# z = F.relu(z)
# z = model.l5(z)
# y = y_before + z
# y = model.l3(y)
# print((y.argmax(-1) == targets)[mask].float().mean())

# print(model.l4.weight.shape, y_before.shape)
# print(model.l4.bias)
# z = model.l4(y_before)
# # z = y_before @ model.l4.weight.T
# z = F.relu(z)
# z = model.l5(z)
# # z = z @ model.l5.weight.T
# y = y_before + z
# y = model.l3(y)
# print((y.argmax(-1) == targets)[mask].float().mean())
