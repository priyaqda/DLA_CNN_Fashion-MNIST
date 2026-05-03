"""
anggota4_quantization/accuracy_analysis.py — Accuracy impact analysis.

Mengukur accuracy drop ketika weights/activations di-quantize
dari FP32 ke INT8/INT16/INT4.

Menggunakan simple LeNet-like model dengan random weights
(karena kita belum train). Yang diukur: output divergence.
"""

import numpy as np
from .quantizer import Quantizer, PerChannelQuantizer


def simple_conv2d(input_act, weights, bias=None):
    """Naive conv2d tanpa dependencies ke modul lain."""
    C_out, C_in, K, _ = weights.shape
    _, H, W = input_act.shape
    H_out = H - K + 1
    W_out = W - K + 1
    output = np.zeros((C_out, H_out, W_out))
    for f in range(C_out):
        for c in range(C_in):
            for i in range(H_out):
                for j in range(W_out):
                    output[f, i, j] += np.sum(
                        weights[f, c] * input_act[c, i:i+K, j:j+K])
    if bias is not None:
        for f in range(C_out):
            output[f] += bias[f]
    return np.maximum(output, 0)  # ReLU


def run_accuracy_comparison(bit_widths=None, seed=42):
    """
    Jalankan LeNet inference dengan berbagai precision level.
    Bandingkan output terhadap FP32 baseline.
    
    Returns
    -------
    list of dict, satu per bit_width
    """
    if bit_widths is None:
        bit_widths = [4, 8, 16]

    np.random.seed(seed)

    # Generate random "trained" weights
    w1 = np.random.randn(6, 1, 5, 5) * 0.5
    w2 = np.random.randn(16, 6, 5, 5) * 0.3

    input_img = np.random.randn(1, 28, 28)

    # FP32 baseline
    x_fp32 = simple_conv2d(input_img, w1)
    x_fp32 = x_fp32[:, ::2, ::2]  # pooling
    x_fp32 = simple_conv2d(x_fp32, w2)
    baseline_output = x_fp32

    results = []
    for bw in bit_widths:
        quant = PerChannelQuantizer(bit_width=bw)

        # Quantize weights
        q_w1, s1 = quant.quantize_weights(w1)
        q_w2, s2 = quant.quantize_weights(w2)
        dq_w1 = quant.dequantize_weights(q_w1, s1)
        dq_w2 = quant.dequantize_weights(q_w2, s2)

        # Run inference with quantized weights
        x_q = simple_conv2d(input_img, dq_w1)
        x_q = x_q[:, ::2, ::2]
        x_q = simple_conv2d(x_q, dq_w2)

        # Compute divergence
        mse = np.mean((baseline_output - x_q) ** 2)
        cosine_sim = (np.sum(baseline_output * x_q) /
                      (np.linalg.norm(baseline_output) * np.linalg.norm(x_q) + 1e-10))

        # Output magnitude comparison
        baseline_mean = np.abs(baseline_output).mean()
        quantized_mean = np.abs(x_q).mean()
        magnitude_ratio = quantized_mean / baseline_mean if baseline_mean > 0 else 1.0

        results.append({
            "bit_width": bw,
            "mse_vs_fp32": round(mse, 6),
            "cosine_similarity": round(cosine_sim, 6),
            "magnitude_ratio": round(magnitude_ratio, 4),
            "weight_compression": round(32 / bw, 1),
        })

    return results
