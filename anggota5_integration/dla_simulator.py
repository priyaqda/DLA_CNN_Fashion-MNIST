"""
anggota5_integration/dla_simulator.py — End-to-end DLA simulator.

Mengintegrasikan semua subsistem:
  Anggota 1: PE Array → compute cycles, MACs
  Anggota 2: Dataflow → data movement, bandwidth
  Anggota 3: Buffer   → tiling, SRAM hit rate
  Anggota 4: Quant    → precision, accuracy/area trade-off
  
Simulator ini menjadi "chip simulator" — mensimulasikan bagaimana
DLA memproses CNN inference secara end-to-end.
"""

import numpy as np
from common.interfaces import LayerSpec, DLABenchmark
from common import config

from anggota1_pe_array.pe_array import PEArray
from anggota2_dataflow.weight_stationary import WeightStationaryDataflow
from anggota2_dataflow.output_stationary import OutputStationaryDataflow
from anggota3_buffer_tiling.tiling_engine import TilingEngine
from anggota3_buffer_tiling.sram_model import SRAMBuffer
from anggota3_buffer_tiling.memory_hierarchy import MemoryHierarchy
from anggota4_quantization.hw_cost_model import HardwareCostModel


class DLASimulator:
    """
    Full DLA chip simulator.
    
    Parameters
    ----------
    pe_rows, pe_cols : int
        PE array dimensions
    bit_width : int
        Precision (8, 16, 32)
    sram_kb : int
        On-chip SRAM size
    dataflow : str
        "weight_stationary" or "output_stationary"
    """

    def __init__(self, pe_rows=None, pe_cols=None,
                 bit_width=None, sram_kb=None,
                 dataflow="weight_stationary"):
        self.pe_rows = pe_rows or config.PE_ARRAY_ROWS
        self.pe_cols = pe_cols or config.PE_ARRAY_COLS
        self.bit_width = bit_width or config.DEFAULT_PRECISION
        self.sram_kb = sram_kb or config.SRAM_SIZE_KB

        # Instantiate subsystems
        self.pe_array = PEArray(self.pe_rows, self.pe_cols, self.bit_width)
        self.sram = SRAMBuffer(total_size_kb=self.sram_kb)
        self.tiling = TilingEngine(self.sram)
        self.mem_hierarchy = MemoryHierarchy(self.sram)
        self.hw_cost = HardwareCostModel()

        if dataflow == "weight_stationary":
            self.dataflow = WeightStationaryDataflow(
                self.pe_rows, self.pe_cols, self.sram_kb)
        else:
            self.dataflow = OutputStationaryDataflow(
                self.pe_rows, self.pe_cols, self.sram_kb)
        
        self.dataflow_name = dataflow

    def simulate_layer(self, layer: LayerSpec) -> dict:
        """
        Simulasi satu layer CNN pada DLA.
        Menggabungkan hasil dari semua subsistem.
        """
        # 1. PE Array profiling
        pe_profile = self.pe_array.profile_layer(layer)

        # 2. Dataflow analysis
        df_result = self.dataflow.simulate(layer)

        # 3. Tiling optimization
        tile_result = self.tiling.find_optimal_tile(layer)

        # 4. Energy computation
        energy = self.mem_hierarchy.compute_total_energy(
            layer,
            dram_reads=df_result.total_dram_reads,
            dram_writes=layer.activation_output_size,
            sram_reads=df_result.total_sram_reads,
            sram_writes=df_result.total_sram_writes,
            mac_count=layer.total_macs,
        )

        # 5. Bandwidth analysis
        bw = self.mem_hierarchy.compute_bandwidth_requirement(
            layer, df_result.cycle_count)

        # Actual cycle count = max(compute, memory)
        actual_cycles = max(pe_profile["actual_cycles"], df_result.cycle_count)

        return {
            "layer": layer.name,
            "total_macs": layer.total_macs,
            "compute_cycles": pe_profile["actual_cycles"],
            "memory_cycles": df_result.cycle_count,
            "actual_cycles": actual_cycles,
            "pe_utilization": pe_profile["utilization"],
            "dram_reads": df_result.total_dram_reads,
            "data_reuse": df_result.data_reuse_factor,
            "tile_size": (tile_result.tile_h, tile_result.tile_w, tile_result.tile_c),
            "sram_hit_rate": tile_result.sram_hit_rate,
            "energy_pJ": energy["total_pJ"],
            "energy_breakdown": energy,
            "memory_bound": bw["memory_bound"],
            "bandwidth_GBs": df_result.bandwidth_required_GBs,
        }

    def simulate_network(self, layer_specs) -> DLABenchmark:
        """
        Simulasi seluruh CNN network pada DLA.
        
        Parameters
        ----------
        layer_specs : list of LayerSpec or tuples
        
        Returns
        -------
        DLABenchmark
        """
        layers = []
        for spec in layer_specs:
            if isinstance(spec, tuple):
                spec = LayerSpec(*spec)
            layers.append(spec)

        layer_results = []
        total_macs = 0
        total_cycles = 0
        total_energy_pJ = 0

        for layer in layers:
            result = self.simulate_layer(layer)
            layer_results.append(result)
            total_macs += result["total_macs"]
            total_cycles += result["actual_cycles"]
            total_energy_pJ += result["energy_pJ"]

        # Throughput
        total_time_s = total_cycles / (config.CLOCK_FREQ_MHZ * 1e6)
        throughput_gops = total_macs / total_time_s / 1e9 if total_time_s > 0 else 0

        # Energy efficiency
        total_energy_uJ = total_energy_pJ / 1e6
        power_W = total_energy_uJ / 1e6 / total_time_s if total_time_s > 0 else 0
        efficiency = throughput_gops / power_W if power_W > 0 else 0

        # Overall utilization
        avg_util = np.mean([r["pe_utilization"] for r in layer_results])

        # Memory bound?
        n_mem_bound = sum(1 for r in layer_results if r["memory_bound"])
        mostly_mem_bound = n_mem_bound > len(layer_results) / 2

        return DLABenchmark(
            total_macs=total_macs,
            total_cycles=total_cycles,
            throughput_gops=round(throughput_gops, 3),
            energy_total_uJ=round(total_energy_uJ, 3),
            energy_efficiency_gops_w=round(efficiency, 2),
            pe_utilization=round(avg_util, 3),
            memory_bound=mostly_mem_bound,
            layer_results=layer_results,
        )

    def get_chip_specs(self) -> dict:
        """Return chip-level specifications."""
        area = self.hw_cost.total_chip_area_mm2(
            self.pe_rows, self.pe_cols, self.bit_width, self.sram_kb)
        power = self.hw_cost.total_power_mw(
            self.pe_rows, self.pe_cols, self.bit_width, utilization=0.85)

        return {
            "array_size": f"{self.pe_rows}×{self.pe_cols}",
            "precision": f"INT{self.bit_width}",
            "sram": f"{self.sram_kb} KB",
            "clock": f"{config.CLOCK_FREQ_MHZ} MHz",
            "dataflow": self.dataflow_name,
            "peak_gops": self.pe_array.peak_gops,
            "total_area_mm2": area["total_mm2"],
            "total_power_mW": power["total_mW"],
        }
