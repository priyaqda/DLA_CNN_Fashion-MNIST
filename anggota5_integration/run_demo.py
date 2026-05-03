"""
anggota5_integration/run_demo.py — Main demo: jalankan seluruh simulasi.

Jalankan dari root directory:
    python -m anggota5_integration.run_demo

Script ini mengintegrasikan semua modul dan menghasilkan:
1. Chip specs
2. Per-layer simulation results
3. Array size comparison
4. Precision comparison
5. Dataflow comparison
6. DLA vs GPU comparison
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.interfaces import LayerSpec
from common import config, utils
from anggota5_integration.dla_simulator import DLASimulator
from anggota5_integration.benchmark import (
    benchmark_array_sizes,
    benchmark_precisions,
    benchmark_dataflows,
    print_benchmark_summary,
)
from anggota5_integration.gpu_baseline import compare_dla_vs_gpu


def main():
    print("\n")
    print("╔" + "═" * 63 + "╗")
    print("║   DLA (Deep Learning Accelerator) — Level 1 Simulation Demo   ║")
    print("║   CNN Accelerator with Weight-Stationary Dataflow             ║")
    print("╚" + "═" * 63 + "╝")

    # ── Prepare workload ──
    layer_specs = [LayerSpec(*l) for l in config.LENET5_LAYERS]
    total_macs = sum(l.total_macs for l in layer_specs)
    print(f"\n  Workload: LeNet-5 ({len(layer_specs)} layers, {total_macs:,} MACs)")

    # ══════════════════════════════════════════
    # 1. Default configuration benchmark
    # ══════════════════════════════════════════
    print("\n" + "─" * 65)
    print("  [1/6] DEFAULT CONFIGURATION")
    print("─" * 65)

    sim = DLASimulator(
        pe_rows=8, pe_cols=8,
        bit_width=8, sram_kb=64,
        dataflow="weight_stationary"
    )
    benchmark = sim.simulate_network(layer_specs)
    chip_specs = sim.get_chip_specs()
    print_benchmark_summary(benchmark, chip_specs)

    # ══════════════════════════════════════════
    # 2. Array size comparison
    # ══════════════════════════════════════════
    print("─" * 65)
    print("  [2/6] ARRAY SIZE COMPARISON")
    print("─" * 65)

    arr_results = benchmark_array_sizes(layer_specs)
    headers = ["Array", "Peak(GOPS)", "Actual(GOPS)", "Util",
               "Energy(μJ)", "Eff(GOPS/W)", "Area(mm²)"]
    rows = []
    for r in arr_results:
        rows.append([
            r["config"],
            f"{r['peak_gops']:.2f}",
            f"{r['actual_gops']:.3f}",
            f"{r['utilization']:.1%}",
            f"{r['energy_uJ']:.3f}",
            f"{r['efficiency']:.2f}",
            f"{r['area_mm2']:.3f}",
        ])
    utils.print_table(headers, rows, "PE Array Size Sweep")

    # ══════════════════════════════════════════
    # 3. Precision comparison
    # ══════════════════════════════════════════
    print("─" * 65)
    print("  [3/6] PRECISION COMPARISON")
    print("─" * 65)

    prec_results = benchmark_precisions(layer_specs)
    headers = ["Precision", "Peak(GOPS)", "Actual(GOPS)",
               "Energy(μJ)", "Eff(GOPS/W)", "Area(mm²)", "Power(mW)"]
    rows = []
    for r in prec_results:
        rows.append([
            r["precision"],
            f"{r['peak_gops']:.2f}",
            f"{r['actual_gops']:.3f}",
            f"{r['energy_uJ']:.3f}",
            f"{r['efficiency']:.2f}",
            f"{r['area_mm2']:.3f}",
            f"{r['power_mW']:.2f}",
        ])
    utils.print_table(headers, rows, "Precision Sweep")

    # ══════════════════════════════════════════
    # 4. Dataflow comparison
    # ══════════════════════════════════════════
    print("─" * 65)
    print("  [4/6] DATAFLOW COMPARISON")
    print("─" * 65)

    df_results = benchmark_dataflows(layer_specs)
    headers = ["Dataflow", "Cycles", "GOPS", "Energy(μJ)", "Eff(GOPS/W)", "Util"]
    rows = []
    for r in df_results:
        rows.append([
            r["dataflow"],
            f"{r['total_cycles']:,}",
            f"{r['throughput_gops']:.3f}",
            f"{r['energy_uJ']:.3f}",
            f"{r['efficiency']:.2f}",
            f"{r['utilization']:.1%}",
        ])
    utils.print_table(headers, rows, "Weight Stationary vs Output Stationary")

    # ══════════════════════════════════════════
    # 5. DLA vs GPU comparison
    # ══════════════════════════════════════════
    print("─" * 65)
    print("  [5/6] DLA vs GPU COMPARISON")
    print("─" * 65)

    gpu_comp = compare_dla_vs_gpu(benchmark, total_macs)
    headers = ["Platform", "Precision", "Peak(GOPS)", "Actual(GOPS)",
               "Time(ms)", "Energy(μJ)", "Eff(GOPS/W)"]
    rows = []
    for name, data in gpu_comp.items():
        rows.append([
            name,
            data["precision"],
            f"{data['peak_gops']:.2f}",
            f"{data['actual_gops']:.3f}",
            f"{data['inference_ms']:.4f}",
            f"{data['energy_uJ']:.3f}",
            f"{data['efficiency']:.2f}",
        ])
    utils.print_table(headers, rows, "DLA vs GPU (LeNet-5 Inference)")

    # ══════════════════════════════════════════
    # 6. Energy breakdown
    # ══════════════════════════════════════════
    print("─" * 65)
    print("  [6/6] ENERGY BREAKDOWN")
    print("─" * 65)

    total_compute = 0
    total_sram = 0
    total_dram = 0
    for lr in benchmark.layer_results:
        eb = lr["energy_breakdown"]
        total_compute += eb["compute_pJ"]
        total_sram += eb["sram_pJ"]
        total_dram += eb["dram_pJ"]
    total = total_compute + total_sram + total_dram

    print(f"\n  Energy Distribution:")
    print(f"    Compute (MACs) : {total_compute/1e6:>8.3f} μJ  "
          f"({total_compute/total*100:>5.1f}%)")
    print(f"    SRAM access    : {total_sram/1e6:>8.3f} μJ  "
          f"({total_sram/total*100:>5.1f}%)")
    print(f"    DRAM access    : {total_dram/1e6:>8.3f} μJ  "
          f"({total_dram/total*100:>5.1f}%)")
    print(f"    ─────────────────────────────────")
    print(f"    Total          : {total/1e6:>8.3f} μJ")

    insight = "DRAM-dominated" if total_dram > total_compute else "Compute-dominated"
    print(f"\n  → Insight: Energy is {insight}.")
    if total_dram > total_compute:
        print(f"    Reducing DRAM access (better tiling, larger SRAM) is key.")
    else:
        print(f"    Lower precision (INT4) could further reduce compute energy.")

    # ── Final summary ──
    print("\n" + "═" * 65)
    print("  SIMULATION COMPLETE")
    print(f"  Config: {chip_specs['array_size']} {chip_specs['precision']}, "
          f"{chip_specs['sram']}, {chip_specs['dataflow']}")
    print(f"  Result: {benchmark.throughput_gops:.3f} GOPS, "
          f"{benchmark.energy_efficiency_gops_w:.2f} GOPS/W, "
          f"{chip_specs['total_area_mm2']:.3f} mm²")
    print("═" * 65 + "\n")


if __name__ == "__main__":
    main()
