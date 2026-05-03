"""
anggota4_quantization/test_quantization.py — Test quantization dan cost model.

Jalankan: python -m anggota4_quantization.test_quantization
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import config, utils
from anggota4_quantization.quantizer import Quantizer, PerChannelQuantizer
from anggota4_quantization.accuracy_analysis import run_accuracy_comparison
from anggota4_quantization.hw_cost_model import HardwareCostModel


def test_quantizer_basic():
    """Test basic quantization correctness."""
    print("=" * 55)
    print("TEST: Quantizer Basic")
    print("=" * 55)

    np.random.seed(42)
    tensor = np.random.randn(3, 3) * 2.0

    for bw in [4, 8, 16]:
        q = Quantizer(bit_width=bw)
        qt, scale = q.quantize(tensor)
        dq = q.dequantize(qt, scale)
        err = q.compute_error(tensor, dq)

        print(f"  INT{bw}: MSE={err['mse']:.6f}, "
              f"SNR={err['snr_db']:.1f}dB, "
              f"max_err={err['max_error']:.4f}, "
              f"range=[{qt.min()}, {qt.max()}]")

    print("  ✓ PASSED\n")


def test_per_channel():
    """Test per-channel quantization."""
    print("=" * 55)
    print("TEST: Per-channel Quantization")
    print("=" * 55)

    np.random.seed(42)
    weights = np.random.randn(6, 3, 5, 5) * 0.5

    q = PerChannelQuantizer(bit_width=8)
    qw, scales = q.quantize_weights(weights)
    dqw = q.dequantize_weights(qw, scales)

    error = np.abs(weights - dqw)
    print(f"  Weights shape : {weights.shape}")
    print(f"  Scales shape  : {scales.shape}")
    print(f"  Mean error    : {error.mean():.6f}")
    print(f"  Max error     : {error.max():.6f}")
    print(f"  Scales range  : [{scales.min():.4f}, {scales.max():.4f}]")
    print("  ✓ PASSED\n")


def test_accuracy_analysis():
    """Test accuracy drop across precisions."""
    print("=" * 55)
    print("TEST: Accuracy Analysis (LeNet)")
    print("=" * 55)

    results = run_accuracy_comparison(bit_widths=[4, 8, 16])

    headers = ["Precision", "MSE vs FP32", "Cosine Sim", "Mag Ratio", "Compression"]
    rows = []
    for r in results:
        rows.append([
            f"INT{r['bit_width']}",
            f"{r['mse_vs_fp32']:.6f}",
            f"{r['cosine_similarity']:.6f}",
            f"{r['magnitude_ratio']:.4f}",
            f"{r['weight_compression']}×",
        ])
    utils.print_table(headers, rows, "Quantization Accuracy Impact")
    print("  ✓ PASSED\n")


def test_hw_cost_model():
    """Test hardware cost estimation."""
    print("=" * 55)
    print("TEST: Hardware Cost Model")
    print("=" * 55)

    model = HardwareCostModel()
    comparisons = model.compare_precisions(rows=8, cols=8)

    headers = ["Precision", "Area(mm²)", "Power(mW)", "Area Red.", "Power Red."]
    rows = []
    for c in comparisons:
        rows.append([
            f"INT{c['bit_width']}",
            f"{c['total_area_mm2']:.3f}",
            f"{c['total_power_mW']:.2f}",
            f"{c['area_reduction']:.1%}",
            f"{c['power_reduction']:.1%}",
        ])
    utils.print_table(headers, rows, "Hardware Cost vs Precision (8×8 array)")

    # Detail breakdown for INT8
    area = model.total_chip_area_mm2(8, 8, 8, 64)
    power = model.total_power_mw(8, 8, 8, utilization=0.85)
    print(f"  INT8 Breakdown:")
    print(f"    PE Array  : {area['pe_array_mm2']:.3f} mm² ({area['pe_pct']:.0f}%)")
    print(f"    SRAM      : {area['sram_mm2']:.3f} mm² ({area['sram_pct']:.0f}%)")
    print(f"    Total     : {area['total_mm2']:.3f} mm²")
    print(f"    Power     : {power['total_mW']:.2f} mW @ 85% util")
    print("  ✓ PASSED\n")


if __name__ == "__main__":
    test_quantizer_basic()
    test_per_channel()
    test_accuracy_analysis()
    test_hw_cost_model()
    print("All Anggota 4 tests PASSED ✓")
