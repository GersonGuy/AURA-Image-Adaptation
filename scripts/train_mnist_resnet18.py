"""Train ResNet-18-MNIST on clean MNIST (mnist_c/identity). Saves ckpt for TTA runs."""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from models.resnet_mnist import build_resnet18_mnist


class MnistNpyDataset(Dataset):
    def __init__(self, x_path, y_path):
        self.x = np.load(x_path)  # (N, 28, 28, 1) uint8
        self.y = np.load(y_path).astype(np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        img = self.x[idx].astype(np.float32) / 255.0  # (28,28,1)
        img = torch.from_numpy(img).permute(2, 0, 1)  # (1,28,28)
        return img, int(self.y[idx])


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def main():
    root = os.environ.get(
        "MNIST_C_IDENTITY_DIR",
        os.path.join(REPO_ROOT, "data", "mnist_c", "identity"),
    )
    train_ds = MnistNpyDataset(os.path.join(root, "train_images.npy"),
                               os.path.join(root, "train_labels.npy"))
    test_ds = MnistNpyDataset(os.path.join(root, "test_images.npy"),
                              os.path.join(root, "test_labels.npy"))

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=2, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_resnet18_mnist(num_classes=10).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
    crit = nn.CrossEntropyLoss()

    for epoch in range(10):
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = crit(out, y)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * x.size(0)
            total += x.size(0)
            correct += (out.argmax(1) == y).sum().item()
        sched.step()

        model.eval()
        t_total, t_correct = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                t_total += x.size(0)
                t_correct += (out.argmax(1) == y).sum().item()
        print(f"epoch {epoch+1}: train_loss={loss_sum/total:.4f} train_acc={correct/total:.4f} test_acc={t_correct/t_total:.4f}", flush=True)

    out_path = os.path.join(REPO_ROOT, "ckpt", "resnet18_mnist.pth")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save({"state_dict": model.state_dict()}, out_path)
    print(f"saved → {out_path}")


if __name__ == "__main__":
    main()
