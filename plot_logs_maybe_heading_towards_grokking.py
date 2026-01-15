import numpy as np
from matplotlib import pyplot as plt

a = None
log_path = "runs/encode_items_sequentially/20260114/174036/partial_log_245000.log"
with open(log_path) as f:
    a = f.read()

lines = a.split("\n")
eval_lines = [line for line in lines if "val_loss" in line]
train_loss = np.array(
    [float(line.split("train_loss=")[1].split(" ")[0]) for line in eval_lines]
)
test_loss = np.array(
    [float(line.split("test_loss=")[1].split(" ")[0]) for line in eval_lines]
)

print(test_loss[:10])

import os
from datetime import datetime

os.makedirs("plots", exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
save_path = f"plots/maybe-heading-towards-grokking-{timestamp}.png"

plt.plot(train_loss, label="train_loss")
plt.plot(test_loss, label="test_loss")
# plt.yscale("log")
plt.title(log_path)
plt.savefig(save_path)
