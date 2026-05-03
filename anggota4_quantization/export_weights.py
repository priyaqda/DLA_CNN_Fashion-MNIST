"""
anggota4_quantization/export_weights.py — Quantize & export weights.

Jalankan SETELAH training selesai:
    cd dla_level1
    python -m anggota4_quantization.export_weights

Input:
    weights/lenet5_fp32.npy     — dari train_lenet.py

Output:
    weights/lenet5_int8.npy     — INT8 weights + scales (untuk Python sim)
    weights/lenet5_int8.hex     — HEX format (untuk Verilog $readmemh)
    weights/lenet5_scales.hex   — scale factors per channel (untuk dequant di Verilog)
    weights/weight_summary.txt  — ringkasan human-readable
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anggota4_quantization.quantizer import PerChannelQuantizer, Quantizer


# Urutan layer — harus sama dengan urutan address di Verilog ROM
LAYER_ORDER = [
    "conv1_weight", "conv1_bias",
    "conv2_weight", "conv2_bias",
    "fc1_weight",   "fc1_bias",
    "fc2_weight",   "fc2_bias",
    "fc3_weight",   "fc3_bias",
]


def quantize_all_weights(fp32_path="weights/lenet5_fp32.npy", bit_width=8):
    """
    Quantize semua weights dari FP32 ke INT8.

    Returns
    -------
    int8_weights : dict — quantized weight arrays
    scales       : dict — scale factor per layer (per-channel untuk conv/fc)
    fp32_weights : dict — original weights untuk perbandingan
    """
    fp32_weights = np.load(fp32_path, allow_pickle=True).item()
    quant = PerChannelQuantizer(bit_width=bit_width)

    int8_weights = {}
    scales = {}

    for name in LAYER_ORDER:
        w = fp32_weights[name]

        if "weight" in name:
            if w.ndim == 4:
                # Conv layer: (C_out, C_in, K, K) — per-channel quantization
                q, s = quant.quantize_weights(w)
            elif w.ndim == 2:
                # FC layer: (out, in) — reshape ke 4D, quantize, reshape balik
                w4d = w[:, :, np.newaxis, np.newaxis]
                q4d, s = quant.quantize_weights(w4d)
                q = q4d[:, :, 0, 0]
            else:
                q_scalar = Quantizer(bit_width=bit_width)
                q, s_val = q_scalar.quantize(w)
                s = np.array([s_val])

            int8_weights[name] = q.astype(np.int8)
            scales[name] = s

        elif "bias" in name:
            # Bias disimpan sebagai INT32 (akumulasi presisi tinggi)
            abs_max = float(np.abs(w).max())
            if abs_max == 0 or not np.isfinite(abs_max):
                s_bias = 1.0
            else:
                s_bias = abs_max / (2**30)  # pakai 2^30 bukan 2^31-1 untuk safety margin
            q_bias_f = np.round(w / s_bias)
            # Clip secara eksplisit ke int32 range sebelum cast
            q_bias_f = np.clip(q_bias_f, -(2**31), 2**31 - 1)
            int8_weights[name] = q_bias_f.astype(np.int32)
            scales[name] = np.array([s_bias])

    return int8_weights, scales, fp32_weights


def export_npy(int8_weights, scales, output_path="weights/lenet5_int8.npy"):
    """Simpan sebagai .npy untuk Python simulation."""
    np.save(output_path, {
        "weights": int8_weights,
        "scales": scales,
    })
    print(f"  ✅ {output_path}")


def export_hex(int8_weights, output_path="weights/lenet5_int8.hex"):
    """
    Export weights ke HEX untuk Verilog $readmemh.

    Format output:
        // layer_name (shape) — start_addr
        AB        ← satu byte per baris, signed INT8 as unsigned hex
        ...

    Di Verilog:
        reg [7:0] weight_rom [0:TOTAL-1];
        initial $readmemh("lenet5_int8.hex", weight_rom);
    """
    total_bytes = 0
    addr = 0

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("// =============================================\n")
        f.write("// LeNet-5 INT8 Weights - Fashion-MNIST\n")
        f.write("// Format: 1 byte per line, signed->unsigned hex\n")
        f.write("// Untuk Verilog: $readmemh(\"lenet5_int8.hex\", rom)\n")
        f.write("// =============================================\n\n")

        for name in LAYER_ORDER:
            q = int8_weights[name]
            flat = q.flatten()

            f.write(f"// {name} — shape={q.shape} — "
                    f"addr=[0x{addr:04X} .. 0x{addr + len(flat) - 1:04X}] "
                    f"({len(flat)} elements)\n")

            if "bias" in name:
                # Bias INT32: simpan sebagai 4 bytes per elemen (little-endian)
                for val in flat:
                    val_int = int(val)
                    b = val_int.to_bytes(4, byteorder='little', signed=True)
                    for byte in b:
                        f.write(f"{byte:02X}\n")
                        total_bytes += 1
                        addr += 1
            else:
                # Weights INT8: satu byte per elemen
                for val in flat:
                    # Two's complement: -1 → FF, -128 → 80, 127 → 7F
                    hex_val = int(val) & 0xFF
                    f.write(f"{hex_val:02X}\n")
                    total_bytes += 1
                    addr += 1

            f.write("\n")

        f.write(f"// Total: {total_bytes} bytes ({total_bytes/1024:.1f} KB)\n")
        f.write(f"// Address range: 0x0000 — 0x{total_bytes-1:04X}\n")

    print(f"  ✅ {output_path} ({total_bytes:,} bytes)")
    return total_bytes


def export_scales(scales, output_path="weights/lenet5_scales.hex"):
    """
    Export scale factors untuk dequantization di Verilog.
    Scale factors disimpan sebagai FP32 (IEEE 754) → 4 bytes hex.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("// Scale factors per layer (IEEE 754 FP32, little-endian)\n")
        f.write("// Di Verilog: gunakan fixed-point atau lookup table\n\n")

        for name in LAYER_ORDER:
            if name not in scales:
                continue
            s = scales[name]
            f.write(f"// {name} - {len(s)} scale(s)\n")
            for val in s:
                # FP32 -> 4 bytes IEEE 754
                fp_bytes = np.float32(val).tobytes()
                for b in fp_bytes:
                    f.write(f"{b:02X}\n")
            f.write("\n")

    print(f"  ✅ {output_path}")


def export_summary(int8_weights, scales, fp32_weights, total_bytes,
                    output_path="weights/weight_summary.txt"):
    """Ringkasan human-readable untuk dokumentasi."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("  WEIGHT SUMMARY - LeNet-5 Fashion-MNIST\n")
        f.write("=" * 60 + "\n\n")

        total_params = 0
        for name in LAYER_ORDER:
            q = int8_weights[name]
            orig = fp32_weights[name]
            s = scales[name]
            n = q.size
            total_params += n

            f.write(f"Layer: {name}\n")
            f.write(f"  Shape          : {q.shape}\n")
            f.write(f"  Elements       : {n:,}\n")
            f.write(f"  FP32 range     : [{orig.min():.4f}, {orig.max():.4f}]\n")
            f.write(f"  INT8 range     : [{q.min()}, {q.max()}]\n")
            f.write(f"  Scale(s)       : {s[:4]}{'...' if len(s) > 4 else ''}\n")

            if "weight" in name:
                # Compute quantization error
                if q.ndim == 4:
                    from anggota4_quantization.quantizer import PerChannelQuantizer
                    pc = PerChannelQuantizer(bit_width=8)
                    dq = pc.dequantize_weights(q.astype(np.int32), s)
                elif q.ndim == 2:
                    from anggota4_quantization.quantizer import PerChannelQuantizer
                    pc = PerChannelQuantizer(bit_width=8)
                    q4d = q[:, :, np.newaxis, np.newaxis].astype(np.int32)
                    dq = pc.dequantize_weights(q4d, s)[:, :, 0, 0]
                else:
                    dq = q * s[0]
                mse = np.mean((orig - dq) ** 2)
                f.write(f"  Quant MSE      : {mse:.8f}\n")

            f.write("\n")

        f.write(f"{'='*60}\n")
        f.write(f"Total parameters : {total_params:,}\n")
        f.write(f"FP32 size        : {total_params * 4 / 1024:.1f} KB\n")
        f.write(f"INT8 size        : {total_bytes / 1024:.1f} KB\n")
        f.write(f"Compression      : {total_params * 4 / total_bytes:.1f}x\n")
        f.write(f"{'='*60}\n\n")

        f.write("FILE MAP - untuk Verilog integration:\n\n")
        f.write("  lenet5_int8.hex    -> $readmemh ke weight ROM\n")
        f.write("  lenet5_scales.hex  -> scale factors untuk dequant output\n")
        f.write("  lenet5_int8.npy    -> Python verification (numpy)\n")
        f.write("  lenet5_fp32.npy    -> FP32 baseline (referensi)\n")

    print(f"  ✅ {output_path}")


def verify_quantization(int8_weights, scales, fp32_weights):
    """
    Verifikasi: dequantize INT8 → FP32, bandingkan dengan original.
    Ini adalah sanity check untuk memastikan quantization benar.
    """
    print(f"\n  Quantization verification:")
    print(f"  {'Layer':<16} {'MSE':>12} {'Cosine Sim':>12} {'MaxErr':>10}")
    print(f"  {'-'*16}-+-{'-'*12}-+-{'-'*12}-+-{'-'*10}")

    from anggota4_quantization.quantizer import PerChannelQuantizer
    pc = PerChannelQuantizer(bit_width=8)

    for name in LAYER_ORDER:
        if "bias" in name:
            continue

        orig = fp32_weights[name]
        q = int8_weights[name].astype(np.int32)
        s = scales[name]

        # Dequantize
        if orig.ndim == 4:
            dq = pc.dequantize_weights(q, s)
        elif orig.ndim == 2:
            q4d = q[:, :, np.newaxis, np.newaxis]
            dq = pc.dequantize_weights(q4d, s)[:, :, 0, 0]
        else:
            dq = q * s[0]

        mse = np.mean((orig - dq) ** 2)
        cos = np.sum(orig * dq) / (np.linalg.norm(orig) * np.linalg.norm(dq) + 1e-12)
        maxerr = np.abs(orig - dq).max()

        status = "✓" if cos > 0.999 else "⚠"
        print(f"  {name:<16} {mse:>12.8f} {cos:>12.6f} {maxerr:>10.6f}  {status}")


def main():
    print("=" * 55)
    print("  EXPORT WEIGHTS — FP32 → INT8 → HEX")
    print("=" * 55)

    fp32_path = "weights/lenet5_fp32.npy"
    if not os.path.exists(fp32_path):
        print(f"\n  ❌ File tidak ditemukan: {fp32_path}")
        print(f"     Jalankan dulu: python -m anggota4_quantization.train_lenet")
        sys.exit(1)

    print(f"\n  Loading {fp32_path}...")

    # 1. Quantize
    int8_weights, scales, fp32_weights = quantize_all_weights(fp32_path)

    # 2. Verify
    verify_quantization(int8_weights, scales, fp32_weights)

    # 3. Export semua format
    print(f"\n  Exporting files:")
    export_npy(int8_weights, scales)
    total_bytes = export_hex(int8_weights)
    export_scales(scales)
    export_summary(int8_weights, scales, fp32_weights, total_bytes)

    # 4. Print Verilog usage guide
    print(f"\n  ─────────────────────────────────────────────")
    print(f"  Verilog usage (Level 2):")
    print(f"  ─────────────────────────────────────────────")
    print(f"  reg [7:0] w_rom [0:{total_bytes-1}];")
    print(f"  initial $readmemh(\"lenet5_int8.hex\", w_rom);")
    print(f"  ─────────────────────────────────────────────")
    print(f"\n  Next step: python -m anggota5_integration.run_demo\n")


if __name__ == "__main__":
    main()
