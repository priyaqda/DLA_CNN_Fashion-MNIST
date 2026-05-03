"""
common/interfaces.py — Dataclass interfaces antar subsistem.

File ini adalah "kontrak" yang mengikat semua modul.
Ketika nanti ke Level 2 (RTL), dataclass ini menjadi port list Verilog.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


@dataclass
class LayerSpec:
    """Spesifikasi satu layer CNN."""
    name: str
    C_in: int       # input channels
    H_in: int       # input height
    W_in: int       # input width
    C_out: int      # output channels (filters)
    K: int          # kernel size (KxK)
    stride: int = 1
    padding: int = 0

    @property
    def H_out(self) -> int:
        return (self.H_in + 2 * self.padding - self.K) // self.stride + 1

    @property
    def W_out(self) -> int:
        return (self.W_in + 2 * self.padding - self.K) // self.stride + 1

    @property
    def total_macs(self) -> int:
        """Total MAC operations for this layer."""
        return self.C_out * self.C_in * self.K * self.K * self.H_out * self.W_out

    @property
    def weight_count(self) -> int:
        return self.C_out * self.C_in * self.K * self.K

    @property
    def activation_input_size(self) -> int:
        return self.C_in * self.H_in * self.W_in

    @property
    def activation_output_size(self) -> int:
        return self.C_out * self.H_out * self.W_out


@dataclass
class PEArrayResult:
    """Output dari PE Array model (Anggota 1 → Anggota 2, 5)."""
    output_feature_map: np.ndarray   # computed output
    total_macs: int                  # total MAC operations
    total_cycles: int                # estimated clock cycles
    utilization: float               # PE utilization (0-1)
    pe_rows: int                     # array dimensions used
    pe_cols: int


@dataclass
class DataflowResult:
    """Output dari Dataflow Simulation (Anggota 2 → Anggota 3, 5)."""
    strategy_name: str               # "weight_stationary" or "output_stationary"
    total_dram_reads: int            # jumlah DRAM access (mahal)
    total_sram_reads: int            # jumlah SRAM access
    total_sram_writes: int
    data_reuse_factor: float         # berapa kali data di-reuse sebelum evict
    bandwidth_required_GBs: float    # bandwidth DRAM yang dibutuhkan
    cycle_count: int                 # total cycles


@dataclass
class TilingResult:
    """Output dari Buffer & Tiling (Anggota 3 → Anggota 5)."""
    tile_h: int                      # tile height
    tile_w: int                      # tile width
    tile_c: int                      # tile channels
    num_tiles: int                   # total tile count
    sram_hit_rate: float             # cache hit rate (0-1)
    dram_traffic_bytes: int          # total DRAM traffic
    optimal: bool                    # apakah ini konfigurasi optimal


@dataclass
class QuantizationResult:
    """Output dari Quantization Analysis (Anggota 4 → Anggota 5)."""
    bit_width: int                   # quantized precision
    accuracy_fp32: float             # baseline accuracy (FP32)
    accuracy_quantized: float        # accuracy setelah quantization
    accuracy_drop: float             # penurunan accuracy (%)
    area_reduction_factor: float     # estimasi area savings vs FP32
    energy_reduction_factor: float   # estimasi energy savings vs FP32


@dataclass 
class DLABenchmark:
    """Final benchmark result (Anggota 5 output)."""
    total_macs: int
    total_cycles: int
    throughput_gops: float           # Giga-OPS
    energy_total_uJ: float          # total energy (microJoule)
    energy_efficiency_gops_w: float  # GOPS/W
    pe_utilization: float
    memory_bound: bool               # True if memory-bound, False if compute-bound
    layer_results: list              # per-layer breakdown
