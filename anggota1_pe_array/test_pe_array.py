"""
anggota1_pe_array/test_pe_array.py — Unit tests + profiling.

Jalankan: python -m anggota1_pe_array.test_pe_array
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.interfaces import LayerSpec
from common import config, utils
from anggota1_pe_array.mac_unit import MACUnit
from anggota1_pe_array.pe_array import PEArray
from anggota1_pe_array.conv_engine import ConvEngine


def test_mac_unit():
    """Test single MAC unit correctness."""
    print("=" * 50)
    print("TEST: MAC Unit")
    print("=" * 50)

    mac = MACUnit(bit_width=8)
    
    # Test dot product: [1,2,3] · [4,5,6] = 4+10+18 = 32
    w = np.array([1, 2, 3], dtype=float)
    a = np.array([4, 5, 6], dtype=float)
    result = mac.mac_vector(w, a)
    expected = np.dot(w, a)
    
    print(f"  Dot product: {result} (expected: {expected})")
    print(f"  MAC count  : {mac.mac_count}")
    print(f"  Energy     : {mac.energy_pj:.2f} pJ")
    assert abs(result - expected) < 1e-6, "MAC unit FAILED"
    print("  ✓ PASSED\n")


def test_pe_array_conv2d():
    """Test PE array convolution against numpy."""
    print("=" * 50)
    print("TEST: PE Array Conv2D")
    print("=" * 50)

    pe = PEArray(rows=4, cols=4, bit_width=32)
    
    np.random.seed(0)
    C_in, H, W = 3, 8, 8
    C_out, K = 4, 3
    
    input_act = np.random.randn(C_in, H, W)
    weights = np.random.randn(C_out, C_in, K, K)
    
    result = pe.compute_conv2d(input_act, weights)
    
    # Reference: manual convolution
    H_out = H - K + 1
    W_out = W - K + 1
    ref = np.zeros((C_out, H_out, W_out))
    for f in range(C_out):
        for c in range(C_in):
            for i in range(H_out):
                for j in range(W_out):
                    ref[f, i, j] += np.sum(
                        weights[f, c] * input_act[c, i:i+K, j:j+K]
                    )
    
    error = np.abs(result.output_feature_map - ref).max()
    print(f"  Output shape : {result.output_feature_map.shape}")
    print(f"  Total MACs   : {result.total_macs:,}")
    print(f"  Total cycles : {result.total_cycles:,}")
    print(f"  Utilization  : {result.utilization:.1%}")
    print(f"  Max error    : {error:.2e}")
    assert error < 1e-10, "Conv2D FAILED"
    print("  ✓ PASSED\n")


def test_profiling():
    """Profile LeNet-5 layers on different array sizes."""
    print("=" * 50)
    print("TEST: LeNet-5 Profiling")
    print("=" * 50)

    layers = [LayerSpec(*l) for l in config.LENET5_LAYERS]

    # Profile on 8x8 array
    engine = ConvEngine(pe_rows=8, pe_cols=8)
    profiles = engine.profile_network(layers)

    headers = ["Layer", "MACs", "Cycles", "Util%", "Latency(μs)", "GOPS"]
    rows = []
    for p in profiles:
        rows.append([
            p["layer"],
            f"{p['total_macs']:,}",
            f"{p['actual_cycles']:,}",
            f"{p['utilization']:.1%}",
            f"{p['latency_us']:.3f}",
            f"{p['throughput_gops']:.2f}",
        ])

    utils.print_table(headers, rows, title=f"LeNet-5 on {engine.pe_array}")

    # Compare array sizes
    print("\n  Array size sweep:")
    for size in [4, 8, 16, 32]:
        eng = ConvEngine(pe_rows=size, pe_cols=size)
        profs = eng.profile_network(layers)
        total_macs = sum(p["total_macs"] for p in profs)
        total_cycles = sum(p["actual_cycles"] for p in profs)
        gops = total_macs / total_cycles * config.CLOCK_FREQ_MHZ * 1e6 / 1e9
        print(f"    {size:>2}×{size:<2} array: "
              f"{total_cycles:>8,} cycles, "
              f"{gops:.2f} GOPS, "
              f"peak={eng.pe_array.peak_gops:.1f} GOPS")

    print("  ✓ PASSED\n")


def test_lenet5_inference():
    """Run LeNet-5 inference end-to-end."""
    print("=" * 50)
    print("TEST: LeNet-5 Inference")
    print("=" * 50)

    engine = ConvEngine(pe_rows=8, pe_cols=8, bit_width=32)
    input_img = np.random.randn(1, 28, 28)

    logits, results = engine.run_lenet5_inference(input_img)

    print(f"  Output logits: {logits.round(3)}")
    print(f"  Predicted class: {np.argmax(logits)}")
    total_macs = sum(r.total_macs for r in results)
    total_cycles = sum(r.total_cycles for r in results)
    print(f"  Total MACs   : {total_macs:,}")
    print(f"  Total cycles : {total_cycles:,}")
    print("  ✓ PASSED\n")


if __name__ == "__main__":
    test_mac_unit()
    test_pe_array_conv2d()
    test_profiling()
    test_lenet5_inference()
    print("All Anggota 1 tests PASSED ✓")
