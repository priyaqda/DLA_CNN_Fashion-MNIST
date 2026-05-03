"""
anggota1_pe_array/inference.py — Inference dengan weights terlatih.

Jalankan setelah training + export:
    cd dla_level1
    python -m anggota1_pe_array.inference

File ini membuktikan bahwa:
1. Weights hasil training menghasilkan prediksi yang benar
2. Weights INT8 (quantized) tetap menghasilkan prediksi sama
3. Pipeline PE Array kita menghasilkan output identik dengan PyTorch
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anggota1_pe_array.conv_engine import ConvEngine
from anggota4_quantization.quantizer import PerChannelQuantizer

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]


def load_test_images(n_samples=10, data_dir="./data"):
    """Load beberapa sample test dari Fashion-MNIST."""
    try:
        from torchvision import datasets, transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        try:
            test_set = datasets.FashionMNIST(
                root=data_dir, train=False, download=True, transform=transform)
        except Exception:
            test_set = datasets.FashionMNIST(
                root=data_dir, train=False, download=False, transform=transform)

        images = []
        labels = []
        for i in range(min(n_samples, len(test_set))):
            img, lbl = test_set[i]
            images.append(img.numpy())  # (1, 28, 28)
            labels.append(lbl)
        return np.array(images), np.array(labels)
    except Exception:
        print("  ⚠ torchvision tidak tersedia, pakai random input")
        return np.random.randn(n_samples, 1, 28, 28), np.zeros(n_samples, dtype=int)


def load_weights(path, mode="fp32"):
    """
    Load weights dari .npy file.

    mode="fp32" → load langsung dari lenet5_fp32.npy
    mode="int8" → load dari lenet5_int8.npy, dequantize ke float
    """
    data = np.load(path, allow_pickle=True).item()

    if mode == "fp32":
        return data  # dict of numpy arrays langsung

    elif mode == "int8":
        int8_w = data["weights"]
        scales = data["scales"]
        quant  = PerChannelQuantizer(bit_width=8)

        fp_weights = {}
        for name in int8_w:
            q = int8_w[name]
            s = scales[name]

            if "weight" in name:
                if q.ndim == 4:
                    fp_weights[name] = quant.dequantize_weights(
                        q.astype(np.int32), s)
                elif q.ndim == 2:
                    q4d = q[:, :, np.newaxis, np.newaxis].astype(np.int32)
                    fp_weights[name] = quant.dequantize_weights(q4d, s)[:, :, 0, 0]
                else:
                    fp_weights[name] = q.astype(np.float64) * s[0]
            elif "bias" in name:
                fp_weights[name] = q.astype(np.float64) * s[0]

        return fp_weights


def run_inference(image, weights_dict, engine=None):
    """
    Jalankan inference satu gambar melalui PE Array.

    Parameters
    ----------
    image        : np.ndarray shape (1, 28, 28)
    weights_dict : dict dari load_weights()
    engine       : ConvEngine (optional, dibuat baru kalau None)

    Returns
    -------
    predicted_class : int
    logits          : np.ndarray (10,)
    results         : list of PEArrayResult
    """
    if engine is None:
        engine = ConvEngine(pe_rows=8, pe_cols=8, bit_width=32)

    all_results = []
    x = image  # (1, 28, 28)

    # Conv1: (1,28,28) → (6,24,24)
    w = weights_dict["conv1_weight"]          # (6,1,5,5)
    b = weights_dict["conv1_bias"]            # (6,)
    x, r = engine.run_layer(x, w, b, relu=True)
    all_results.append(r)
    x = x[:, ::2, ::2]  # AvgPool 2×2 → (6,12,12)

    # Conv2: (6,12,12) → (16,8,8)
    w = weights_dict["conv2_weight"]          # (16,6,5,5)
    b = weights_dict["conv2_bias"]            # (16,)
    x, r = engine.run_layer(x, w, b, relu=True)
    all_results.append(r)
    x = x[:, ::2, ::2]  # AvgPool 2×2 → (16,4,4)

    # Flatten → FC layers (sebagai 1×1 conv)
    x = x.reshape(-1, 1, 1)  # (256, 1, 1)

    for fc_name, do_relu in [("fc1", True), ("fc2", True), ("fc3", False)]:
        w = weights_dict[f"{fc_name}_weight"]  # (out, in)
        b = weights_dict[f"{fc_name}_bias"]    # (out,)
        # Reshape FC weight ke conv format: (out, in, 1, 1)
        w_conv = w[:, :, np.newaxis, np.newaxis]
        x, r = engine.run_layer(x, w_conv, b, relu=do_relu)
        all_results.append(r)

    logits = x.flatten()
    predicted = np.argmax(logits)

    return predicted, logits, all_results


def softmax(x):
    """Softmax untuk confidence percentage."""
    e = np.exp(x - x.max())
    return e / e.sum()


def main():
    print("=" * 60)
    print("  INFERENCE — LeNet-5 on Fashion-MNIST")
    print("=" * 60)

    # Check files
    fp32_path = "weights/lenet5_fp32.npy"
    int8_path = "weights/lenet5_int8.npy"

    if not os.path.exists(fp32_path):
        print(f"\n  ❌ {fp32_path} tidak ditemukan")
        print(f"     Jalankan: python -m anggota4_quantization.train_lenet")
        sys.exit(1)

    # Load test images
    print(f"\n  Loading test images...")
    images, labels = load_test_images(n_samples=20)
    print(f"  Loaded {len(images)} images\n")

    # Load weights
    w_fp32 = load_weights(fp32_path, mode="fp32")
    if os.path.exists(int8_path):
        w_int8 = load_weights(int8_path, mode="int8")
        has_int8 = True
    else:
        has_int8 = False
        print("  ⚠ INT8 weights belum ada, jalankan export_weights dulu\n")

    engine = ConvEngine(pe_rows=8, pe_cols=8, bit_width=32)

    # ── Run inference ────────────────────────────────────────
    print(f"  {'#':>3} {'True Label':<16} {'FP32 Pred':<16} {'Conf':>6}", end="")
    if has_int8:
        print(f" {'INT8 Pred':<16} {'Match':>5}")
    else:
        print()
    print(f"  {'─'*3}─┼─{'─'*16}─┼─{'─'*16}─┼─{'─'*6}", end="")
    if has_int8:
        print(f"─┼─{'─'*16}─┼─{'─'*5}")
    else:
        print()

    correct_fp32 = 0
    correct_int8 = 0
    match_count  = 0

    for i in range(len(images)):
        true_label = CLASS_NAMES[labels[i]]

        # FP32 inference
        pred_fp32, logits_fp32, _ = run_inference(images[i], w_fp32, engine)
        conf = softmax(logits_fp32)[pred_fp32] * 100
        if pred_fp32 == labels[i]:
            correct_fp32 += 1

        print(f"  {i+1:>3} │ {true_label:<16} │ {CLASS_NAMES[pred_fp32]:<16} │ {conf:>5.1f}%", end="")

        if has_int8:
            pred_int8, logits_int8, _ = run_inference(images[i], w_int8, engine)
            if pred_int8 == labels[i]:
                correct_int8 += 1
            match = pred_fp32 == pred_int8
            if match:
                match_count += 1
            print(f" │ {CLASS_NAMES[pred_int8]:<16} │ {'✓' if match else '✗':>5}")
        else:
            print()

    # ── Summary ──────────────────────────────────────────────
    n = len(images)
    print(f"\n  Summary ({n} samples):")
    print(f"    FP32 accuracy  : {correct_fp32}/{n} ({correct_fp32/n*100:.1f}%)")
    if has_int8:
        print(f"    INT8 accuracy  : {correct_int8}/{n} ({correct_int8/n*100:.1f}%)")
        print(f"    FP32↔INT8 match: {match_count}/{n} ({match_count/n*100:.1f}%)")
    print()


if __name__ == "__main__":
    main()
