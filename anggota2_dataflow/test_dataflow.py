"""
anggota2_dataflow/test_dataflow.py — Test dan perbandingan dataflow.

Jalankan: python -m anggota2_dataflow.test_dataflow
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.interfaces import LayerSpec
from common import config
from anggota2_dataflow.dataflow_analyzer import DataflowAnalyzer


def test_single_layer():
    """Test dataflow comparison on a single conv layer."""
    print("=" * 60)
    print("TEST: Single Layer Dataflow Comparison")
    print("=" * 60)

    layer = LayerSpec("conv_test", C_in=3, H_in=32, W_in=32, 
                      C_out=16, K=3, stride=1, padding=0)
    
    analyzer = DataflowAnalyzer(pe_rows=8, pe_cols=8)
    comp = analyzer.compare_layer(layer)
    
    ws, os_ = comp["ws"], comp["os"]
    print(f"  Layer: {layer.name}")
    print(f"  MACs : {layer.total_macs:,}")
    print(f"\n  Weight Stationary:")
    print(f"    DRAM reads : {ws.total_dram_reads:,}")
    print(f"    SRAM reads : {ws.total_sram_reads:,}")
    print(f"    Data reuse : {ws.data_reuse_factor:.1f}x")
    print(f"    Bandwidth  : {ws.bandwidth_required_GBs:.2f} GB/s")
    print(f"\n  Output Stationary:")
    print(f"    DRAM reads : {os_.total_dram_reads:,}")
    print(f"    SRAM reads : {os_.total_sram_reads:,}")
    print(f"    Data reuse : {os_.data_reuse_factor:.1f}x")
    print(f"    Bandwidth  : {os_.bandwidth_required_GBs:.2f} GB/s")
    print("  ✓ PASSED\n")


def test_lenet5_comparison():
    """Compare dataflows across all LeNet-5 layers."""
    print("=" * 60)
    print("TEST: LeNet-5 Full Network Comparison")
    print("=" * 60)

    analyzer = DataflowAnalyzer(pe_rows=8, pe_cols=8)
    layers = [LayerSpec(*l) for l in config.LENET5_LAYERS]
    analyzer.print_comparison(layers)

    # Energy breakdown for conv1
    energy = analyzer.compute_energy_breakdown(layers[0])
    print(f"  Energy breakdown for {layers[0].name}:")
    for strategy in ["ws_energy", "os_energy"]:
        e = energy[strategy]
        label = "WS" if "ws" in strategy else "OS"
        print(f"    {label}: compute={e['compute_pJ']:.0f}pJ, "
              f"SRAM={e['sram_read_pJ']:.0f}pJ, "
              f"DRAM={e['dram_pJ']:.0f}pJ, "
              f"total={e['total_pJ']:.0f}pJ")

    print("  ✓ PASSED\n")


def test_sram_size_sweep():
    """Sweep SRAM size to see impact on DRAM traffic."""
    print("=" * 60)
    print("TEST: SRAM Size Sweep")
    print("=" * 60)

    layer = LayerSpec("conv2", C_in=6, H_in=12, W_in=12,
                      C_out=16, K=5, stride=1, padding=0)

    print(f"  Layer: {layer.name} ({layer.total_macs:,} MACs)")
    print(f"  {'SRAM(KB)':>10} | {'WS DRAM':>10} | {'OS DRAM':>10} | {'Winner':>8}")
    print(f"  {'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}")

    for sram_kb in [8, 16, 32, 64, 128, 256]:
        analyzer = DataflowAnalyzer(pe_rows=8, pe_cols=8, sram_size_kb=sram_kb)
        comp = analyzer.compare_layer(layer)
        ws_dr = comp["ws"].total_dram_reads
        os_dr = comp["os"].total_dram_reads
        winner = "WS" if ws_dr <= os_dr else "OS"
        print(f"  {sram_kb:>10} | {ws_dr:>10,} | {os_dr:>10,} | {winner:>8}")

    print("\n  ✓ PASSED\n")


if __name__ == "__main__":
    test_single_layer()
    test_lenet5_comparison()
    test_sram_size_sweep()
    print("All Anggota 2 tests PASSED ✓")
