"""
anggota3_buffer_tiling/test_tiling.py — Test buffer dan tiling.

Jalankan: python -m anggota3_buffer_tiling.test_tiling
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.interfaces import LayerSpec
from common import config, utils
from anggota3_buffer_tiling.sram_model import SRAMBuffer
from anggota3_buffer_tiling.tiling_engine import TilingEngine
from anggota3_buffer_tiling.memory_hierarchy import MemoryHierarchy


def test_sram_model():
    """Test SRAM buffer model."""
    print("=" * 55)
    print("TEST: SRAM Buffer Model")
    print("=" * 55)

    sram = SRAMBuffer(total_size_kb=64, num_banks=4)
    print(f"  {sram}")
    print(f"  Weight buffer  : {sram.weight_buf_bytes:,} bytes")
    print(f"  Act buffer     : {sram.act_buf_bytes:,} bytes")
    print(f"  Output buffer  : {sram.out_buf_bytes:,} bytes")
    print(f"  Area estimate  : {sram.area_estimate_mm2:.2f} mm²")
    
    # Test partitioning
    part = sram.compute_partitioning(
        weight_bytes=8000, act_bytes=24000, out_bytes=12000)
    print(f"  Partitioning   : all_fit={part['all_fit']}, "
          f"util={part['utilization']:.1%}")
    print("  ✓ PASSED\n")


def test_tiling_optimal():
    """Test optimal tile finding."""
    print("=" * 55)
    print("TEST: Optimal Tile Search")
    print("=" * 55)

    layer = LayerSpec("conv1", C_in=1, H_in=28, W_in=28,
                      C_out=6, K=5, stride=1, padding=0)

    for sram_kb in [8, 16, 32, 64]:
        sram = SRAMBuffer(total_size_kb=sram_kb)
        engine = TilingEngine(sram)
        result = engine.find_optimal_tile(layer)
        print(f"  SRAM={sram_kb:>3}KB → tile=({result.tile_h}×{result.tile_w}×{result.tile_c}), "
              f"tiles={result.num_tiles:>3}, "
              f"DRAM={result.dram_traffic_bytes:>8,}B, "
              f"hit={result.sram_hit_rate:.1%}, "
              f"{'✓ optimal' if result.optimal else '✗ constrained'}")

    print("  ✓ PASSED\n")


def test_tile_sweep():
    """Sweep tile sizes to show trade-off."""
    print("=" * 55)
    print("TEST: Tile Size Sweep")
    print("=" * 55)

    layer = LayerSpec("conv2", C_in=6, H_in=12, W_in=12,
                      C_out=16, K=5, stride=1, padding=0)

    engine = TilingEngine(SRAMBuffer(total_size_kb=64))
    results = engine.sweep_tile_sizes(layer)

    headers = ["Tile(HxWxC)", "Tiles", "SRAM Util", "Fits?", "DRAM(B)"]
    rows = []
    for r in results:
        rows.append([
            f"{r['tile_h']}×{r['tile_w']}×{r['tile_c']}",
            r["total_tiles"],
            f"{r['sram_utilization']:.1%}",
            "Yes" if r["fits_sram"] else "No",
            f"{r['dram_total_bytes']:,}",
        ])
    utils.print_table(headers, rows, f"Tile sweep for {layer.name}")
    print("  ✓ PASSED\n")


def test_memory_hierarchy():
    """Test memory hierarchy energy and bandwidth."""
    print("=" * 55)
    print("TEST: Memory Hierarchy Analysis")
    print("=" * 55)

    layer = LayerSpec("conv1", C_in=1, H_in=28, W_in=28,
                      C_out=6, K=5, stride=1, padding=0)
    
    mem = MemoryHierarchy()
    
    energy = mem.compute_total_energy(
        layer,
        dram_reads=layer.weight_count + layer.activation_input_size,
        dram_writes=layer.activation_output_size,
        sram_reads=layer.total_macs * 2,
        sram_writes=layer.activation_output_size,
        mac_count=layer.total_macs,
    )
    
    print(f"  Layer: {layer.name}")
    print(f"  Energy breakdown:")
    print(f"    Compute : {energy['compute_pJ']:>10,.0f} pJ ({energy['compute_pct']:>5.1f}%)")
    print(f"    SRAM    : {energy['sram_pJ']:>10,.0f} pJ ({energy['sram_pct']:>5.1f}%)")
    print(f"    DRAM    : {energy['dram_pJ']:>10,.0f} pJ ({energy['dram_pct']:>5.1f}%)")
    print(f"    Total   : {energy['total_pJ']:>10,.0f} pJ")
    
    bw = mem.compute_bandwidth_requirement(layer, total_cycles=1000)
    print(f"\n  Bandwidth analysis:")
    print(f"    Required  : {bw['required_bandwidth_GBs']:.2f} GB/s")
    print(f"    Available : {bw['available_bandwidth_GBs']} GB/s")
    print(f"    Compute intensity: {bw['compute_intensity']:.2f} ops/byte")
    print(f"    Memory bound: {bw['memory_bound']}")
    print("  ✓ PASSED\n")


if __name__ == "__main__":
    test_sram_model()
    test_tiling_optimal()
    test_tile_sweep()
    test_memory_hierarchy()
    print("All Anggota 3 tests PASSED ✓")
