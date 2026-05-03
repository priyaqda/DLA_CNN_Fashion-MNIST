"""
anggota4_quantization/train_lenet.py — Training LeNet-5 pada Fashion-MNIST.

Jalankan:
    cd dla_level1
    python -m anggota4_quantization.train_lenet

Output:
    weights/lenet5_fp32.npy   — weights FP32 (untuk referensi akurasi)
    weights/accuracy.txt      — log akurasi training

Butuh: pip install torch torchvision
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ──────────────────────────────────────────────
# CLASS NAMES — 10 kategori Fashion-MNIST
# ──────────────────────────────────────────────
CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# ──────────────────────────────────────────────
# LeNet-5 — arsitektur PERSIS SAMA dengan simulasi kita
#
# Ini penting: arsitektur di sini harus identik dengan
# apa yang di-model di pe_array.py dan conv_engine.py,
# supaya weight yang dihasilkan bisa dipakai di Verilog
# tanpa perubahan dimensi.
#
# Layer map:
#   conv1:  1ch → 6ch,  kernel 5×5, stride 1  → (6, 24, 24)
#   pool1:  AvgPool 2×2                       → (6, 12, 12)
#   conv2:  6ch → 16ch, kernel 5×5, stride 1  → (16, 8, 8)
#   pool2:  AvgPool 2×2                       → (16, 4, 4) = 256
#   fc1:    256 → 120
#   fc2:    120 → 84
#   fc3:    84  → 10
# ──────────────────────────────────────────────
class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.pool  = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc1   = nn.Linear(16 * 4 * 4, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)
        self.relu  = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))   # (B,6,12,12)
        x = self.pool(self.relu(self.conv2(x)))   # (B,16,4,4)
        x = x.view(x.size(0), -1)                # (B,256)
        x = self.relu(self.fc1(x))                # (B,120)
        x = self.relu(self.fc2(x))                # (B,84)
        x = self.fc3(x)                           # (B,10)
        return x


def train(epochs=10, batch_size=64, lr=1e-3, data_dir="./data"):
    """
    Training pipeline lengkap.

    Parameters
    ----------
    epochs     : jumlah epoch training (10 sudah cukup)
    batch_size : ukuran batch (64 standar untuk LeNet)
    lr         : learning rate (Adam, 1e-3 default)
    data_dir   : tempat download dataset Fashion-MNIST

    Returns
    -------
    model      : LeNet5 yang sudah terlatih
    accuracy   : akurasi test set (%)
    """
    # ── 1. Persiapan dataset ─────────────────────────────────
    #
    # Normalisasi: (pixel - 0.5) / 0.5  → range [-1, 1]
    # Ini penting karena weight akan di-quantize nanti,
    # dan distribusi yang centered di 0 lebih ramah quantization.
    transform = transforms.Compose([
        transforms.ToTensor(),           # PIL → tensor [0,1]
        transforms.Normalize((0.5,), (0.5,))  # → [-1, 1]
    ])

    print("=" * 55)
    print("  TRAINING LeNet-5 pada Fashion-MNIST")
    print("=" * 55)
    print(f"\n  Loading dataset dari '{data_dir}'...")

    use_synthetic = False
    try:
        train_set = datasets.FashionMNIST(
            root=data_dir, train=True, download=True, transform=transform)
        test_set = datasets.FashionMNIST(
            root=data_dir, train=False, download=True, transform=transform)
        dataset_name = "Fashion-MNIST (real)"
    except Exception as e:
        # Fallback: kalau download gagal (misal tidak ada internet),
        # coba load dari file lokal yang sudah ada
        try:
            train_set = datasets.FashionMNIST(
                root=data_dir, train=True, download=False, transform=transform)
            test_set = datasets.FashionMNIST(
                root=data_dir, train=False, download=False, transform=transform)
            dataset_name = "Fashion-MNIST (local cache)"
        except Exception:
            print(f"\n  ⚠ Fashion-MNIST tidak tersedia: {e}")
            print(f"  → Menggunakan synthetic dataset (untuk testing pipeline)")
            print(f"  → Di laptop kamu, download akan otomatis berhasil.\n")
            use_synthetic = True
            dataset_name = "Synthetic (pipeline test only)"

    if use_synthetic:
        # Buat synthetic dataset dengan shape yang sama persis
        from torch.utils.data import TensorDataset
        n_train, n_test = 2000, 500
        X_train = torch.randn(n_train, 1, 28, 28)
        y_train = torch.randint(0, 10, (n_train,))
        X_test  = torch.randn(n_test, 1, 28, 28)
        y_test  = torch.randint(0, 10, (n_test,))
        train_set = TensorDataset(X_train, y_train)
        test_set  = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True, num_workers=0)
    test_loader  = DataLoader(test_set, batch_size=256,
                              shuffle=False, num_workers=0)

    print(f"  Dataset          : {dataset_name}")
    print(f"  Training samples : {len(train_set):,}")
    print(f"  Test samples     : {len(test_set):,}")
    print(f"  Input shape      : (1, 28, 28)")
    print(f"  Classes          : {len(CLASS_NAMES)}")
    print(f"\n  Hyperparameters:")
    print(f"    Epochs         : {epochs}")
    print(f"    Batch size     : {batch_size}")
    print(f"    Learning rate  : {lr}")
    print(f"    Optimizer      : Adam")
    print(f"    Loss           : CrossEntropyLoss")

    # ── 2. Inisialisasi model ────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"    Device         : {device}")

    model     = LeNet5().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Hitung total parameter
    total_params = sum(p.numel() for p in model.parameters())
    print(f"    Parameters     : {total_params:,}")

    # ── 3. Training loop ─────────────────────────────────────
    print(f"\n  {'Epoch':>7} | {'Loss':>8} | {'Train Acc':>10} | {'Test Acc':>10}")
    print(f"  {'-'*7}-+-{'-'*8}-+-{'-'*10}-+-{'-'*10}")

    log_lines = []
    best_acc = 0

    for epoch in range(1, epochs + 1):
        # -- Train --
        model.train()
        total_loss   = 0
        train_correct = 0
        train_total   = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss    += loss.item()
            train_correct += (outputs.argmax(dim=1) == labels).sum().item()
            train_total   += labels.size(0)

        avg_loss  = total_loss / len(train_loader)
        train_acc = train_correct / train_total * 100

        # -- Evaluate --
        model.eval()
        test_correct = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                preds = model(images).argmax(dim=1)
                test_correct += (preds == labels).sum().item()

        test_acc = test_correct / len(test_set) * 100

        print(f"  {epoch:>7} | {avg_loss:>8.4f} | {train_acc:>9.2f}% | {test_acc:>9.2f}%")
        log_lines.append(f"Epoch {epoch}: loss={avg_loss:.4f} "
                         f"train_acc={train_acc:.2f}% test_acc={test_acc:.2f}%")

        if test_acc > best_acc:
            best_acc = test_acc

    # ── 4. Per-class accuracy ────────────────────────────────
    print(f"\n  Per-class accuracy:")
    class_correct = [0] * 10
    class_total   = [0] * 10
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(dim=1)
            for p, t in zip(preds, labels):
                class_correct[t.item()] += (p == t).item()
                class_total[t.item()]   += 1

    for i in range(10):
        acc = class_correct[i] / class_total[i] * 100
        bar = "█" * int(acc / 5) + "░" * (20 - int(acc / 5))
        print(f"    {CLASS_NAMES[i]:<14} {bar} {acc:>5.1f}%")

    # ── 5. Simpan weights ke .npy ────────────────────────────
    os.makedirs("weights", exist_ok=True)
    model_cpu = model.cpu()

    weights_dict = {
        # Conv layers: shape (C_out, C_in, K, K)
        "conv1_weight": model_cpu.conv1.weight.detach().numpy(),  # (6,1,5,5)
        "conv1_bias":   model_cpu.conv1.bias.detach().numpy(),    # (6,)
        "conv2_weight": model_cpu.conv2.weight.detach().numpy(),  # (16,6,5,5)
        "conv2_bias":   model_cpu.conv2.bias.detach().numpy(),    # (16,)
        # FC layers: shape (out_features, in_features)
        "fc1_weight":   model_cpu.fc1.weight.detach().numpy(),    # (120,256)
        "fc1_bias":     model_cpu.fc1.bias.detach().numpy(),      # (120,)
        "fc2_weight":   model_cpu.fc2.weight.detach().numpy(),    # (84,120)
        "fc2_bias":     model_cpu.fc2.bias.detach().numpy(),      # (84,)
        "fc3_weight":   model_cpu.fc3.weight.detach().numpy(),    # (10,84)
        "fc3_bias":     model_cpu.fc3.bias.detach().numpy(),      # (10,)
    }
    np.save("weights/lenet5_fp32.npy", weights_dict)

    # Simpan log akurasi
    with open("weights/accuracy.txt", "w") as f:
        f.write(f"Model: LeNet-5\n")
        f.write(f"Dataset: Fashion-MNIST\n")
        f.write(f"Final test accuracy: {test_acc:.2f}%\n")
        f.write(f"Best test accuracy: {best_acc:.2f}%\n")
        f.write(f"Total parameters: {total_params:,}\n\n")
        for line in log_lines:
            f.write(line + "\n")

    print(f"\n  ✅ Weights tersimpan → weights/lenet5_fp32.npy")
    print(f"  ✅ Log akurasi     → weights/accuracy.txt")
    print(f"\n  Final test accuracy : {test_acc:.2f}%")
    print(f"  Best test accuracy  : {best_acc:.2f}%")
    print(f"  Total parameters    : {total_params:,}")
    print(f"\n  Next step: python -m anggota4_quantization.export_weights\n")

    return model, test_acc


if __name__ == "__main__":
    train()
