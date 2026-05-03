"""
anggota5_integration/benchmark.py — Benchmarking utilities.

Membandingkan konfigurasi DLA yang berbeda dan menyajikan hasilnya.
"""

import numpy as np
from common.interfaces import LayerSpec, DLABenchmark
from common import config, utils
from .dla_simulator import DLASimulator


def benchmark_array_sizes(layer_specs, sizes=None):
    """Benchmark berbagai ukuran PE array."""
    if sizes is None:
        sizes = [(4, 4), (8, 8), (16, 16), (32, 32)]

    results = []
    for rows, cols in sizes:
        sim = DLASimulator(pe_rows=rows, pe_cols=cols)
        bench = sim.simulate_network(layer_specs)
        specs = sim.get_chip_specs()
        results.append({
            "config": f"{rows}×{cols}",
            "peak_gops": specs["peak_gops"],
            "actual_gops": bench.throughput_gops,
            "utilization": bench.pe_utilization,
            "energy_uJ": bench.energy_total_uJ,
            "efficiency": bench.energy_efficiency_gops_w,
            "area_mm2": specs["total_area_mm2"],
            "cycles": bench.total_cycles,
        })
    return results


def benchmark_precisions(layer_specs, precisions=None):
    """Benchmark berbagai precision levels."""
    if precisions is None:
        precisions = [4, 8, 16, 32]

    results = []
    for bw in precisions:
        sim = DLASimulator(bit_width=bw)
        bench = sim.simulate_network(layer_specs)
        specs = sim.get_chip_specs()
        results.append({
            "precision": f"INT{bw}",
            "peak_gops": specs["peak_gops"],
            "actual_gops": bench.throughput_gops,
            "energy_uJ": bench.energy_total_uJ,
            "efficiency": bench.energy_efficiency_gops_w,
            "area_mm2": specs["total_area_mm2"],
            "power_mW": specs["total_power_mW"],
        })
    return results


def benchmark_dataflows(layer_specs):
    """Benchmark Weight Stationary vs Output Stationary."""
    results = []
    for df in ["weight_stationary", "output_stationary"]:
        sim = DLASimulator(dataflow=df)
        bench = sim.simulate_network(layer_specs)
        results.append({
            "dataflow": df,
            "total_cycles": bench.total_cycles,
            "throughput_gops": bench.throughput_gops,
            "energy_uJ": bench.energy_total_uJ,
            "efficiency": bench.energy_efficiency_gops_w,
            "utilization": bench.pe_utilization,
        })
    return results


def print_benchmark_summary(benchmark: DLABenchmark, chip_specs: dict):
    """Print a comprehensive benchmark report."""
    print("\n" + "=" * 65)
    print("  DLA BENCHMARK REPORT")
    print("=" * 65)

    print(f"\n  Chip Configuration:")
    for k, v in chip_specs.items():
        print(f"    {k:<20}: {v}")

    print(f"\n  Performance Summary:")
    print(f"    Total MACs         : {benchmark.total_macs:,}")
    print(f"    Total cycles       : {benchmark.total_cycles:,}")
    print(f"    Throughput         : {benchmark.throughput_gops:.3f} GOPS")
    print(f"    PE utilization     : {benchmark.pe_utilization:.1%}")
    print(f"    Memory bound       : {'Yes' if benchmark.memory_bound else 'No'}")

    print(f"\n  Energy:")
    print(f"    Total energy       : {benchmark.energy_total_uJ:.3f} μJ")
    print(f"    Efficiency         : {benchmark.energy_efficiency_gops_w:.2f} GOPS/W")

    if benchmark.layer_results:
        print(f"\n  Per-layer Breakdown:")
        headers = ["Layer", "MACs", "Cycles", "Util", "MemBound", "Energy(pJ)"]
        rows = []
        for lr in benchmark.layer_results:
            rows.append([
                lr["layer"],
                f"{lr['total_macs']:,}",
                f"{lr['actual_cycles']:,}",
                f"{lr['pe_utilization']:.1%}",
                "Yes" if lr["memory_bound"] else "No",
                f"{lr['energy_pJ']:,.0f}",
            ])
        utils.print_table(headers, rows)

    print("=" * 65 + "\n")
