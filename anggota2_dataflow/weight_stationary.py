"""
anggota2_dataflow/weight_stationary.py — Weight Stationary Dataflow.

Dalam Weight Stationary (WS):
- Weights TETAP di PE selama memproses semua activation
- Activations mengalir masuk dari buffer, partial sums mengalir keluar
- Keuntungan: minimize weight read → hemat DRAM bandwidth
- Kerugian: activation harus di-stream berkali-kali

Referensi: Eyeriss (MIT, 2016) — Row Stationary
"""

import numpy as np
from common.interfaces import LayerSpec, DataflowResult
from common import config


class WeightStationaryDataflow:
    """
    Simulasi dataflow weight-stationary pada PE array.
    Menghitung data movement, DRAM access, dan bandwidth.
    """

    def __init__(self, pe_rows=None, pe_cols=None, sram_size_kb=None):
        self.pe_rows = pe_rows or config.PE_ARRAY_ROWS
        self.pe_cols = pe_cols or config.PE_ARRAY_COLS
        self.sram_size_bytes = (sram_size_kb or config.SRAM_SIZE_KB) * 1024
        self.bit_width = config.DEFAULT_PRECISION

    def _bytes_per_element(self):
        return self.bit_width // 8

    def simulate(self, layer: LayerSpec) -> DataflowResult:
        """
        Simulasi data movement untuk satu conv layer.
        
        Weight Stationary strategy:
        1. Load weights ke PE (sekali per filter group)
        2. Stream activations melalui PE
        3. Collect partial sums → accumulate → output
        """
        bpe = self._bytes_per_element()
        
        # === Weight loading ===
        # Setiap PE menyimpan 1 weight. Array M×N → M×N weights loaded.
        # Total weights per layer: C_out × C_in × K × K
        total_weights = layer.weight_count
        weights_per_load = self.pe_rows * self.pe_cols
        num_weight_loads = int(np.ceil(total_weights / weights_per_load))
        
        # Cek apakah semua weights muat di SRAM
        weight_bytes = total_weights * bpe
        weights_fit_sram = weight_bytes <= self.sram_size_bytes * 0.5  # 50% for weights
        
        if weights_fit_sram:
            # Weights di-load dari DRAM ke SRAM sekali, 
            # lalu dari SRAM ke PE berkali-kali
            dram_weight_reads = total_weights
            sram_weight_reads = total_weights * num_weight_loads
        else:
            # Weights harus di-reload dari DRAM setiap tile
            dram_weight_reads = total_weights * num_weight_loads
            sram_weight_reads = dram_weight_reads
        
        # === Activation streaming ===
        # Setiap activation element dipakai oleh K×K filter positions
        # Dalam WS: activation di-stream per output row
        total_act_elements = layer.activation_input_size
        output_elements = layer.activation_output_size
        
        # Spatial reuse: each activation read from SRAM 
        # K times (reused across K filter positions in same row)
        spatial_reuse = layer.K
        
        # Activations fit in remaining SRAM?
        act_bytes = total_act_elements * bpe
        act_sram_budget = self.sram_size_bytes - min(weight_bytes, 
                                                      self.sram_size_bytes * 0.5)
        
        if act_bytes <= act_sram_budget:
            # All activations fit → 1 DRAM load
            dram_act_reads = total_act_elements
            sram_act_reads = total_act_elements * layer.C_out  # read for each filter
        else:
            # Must re-fetch from DRAM for each filter group
            n_filter_groups = int(np.ceil(layer.C_out / self.pe_rows))
            dram_act_reads = total_act_elements * n_filter_groups
            sram_act_reads = dram_act_reads * spatial_reuse

        # === Output writes ===
        # Partial sums written back to SRAM, then to DRAM
        sram_output_writes = output_elements * layer.C_in  # partial sums
        dram_output_writes = output_elements  # final output
        
        # === Totals ===
        total_dram_reads = dram_weight_reads + dram_act_reads
        total_sram_reads = sram_weight_reads + sram_act_reads
        total_sram_writes = sram_output_writes
        
        # Data reuse factor
        total_macs = layer.total_macs
        data_reuse = total_macs / total_dram_reads if total_dram_reads > 0 else 0
        
        # Cycle count (compute-bound or memory-bound)
        compute_cycles = int(np.ceil(
            total_macs / (self.pe_rows * self.pe_cols)))
        
        dram_bytes = (total_dram_reads + dram_output_writes) * bpe
        memory_cycles = int(np.ceil(
            dram_bytes / (config.DRAM_BANDWIDTH_GBs * 1e9 
                          / config.CLOCK_FREQ_MHZ / 1e6)))
        
        actual_cycles = max(compute_cycles, memory_cycles)
        
        bandwidth_required = (dram_bytes * config.CLOCK_FREQ_MHZ * 1e6 
                              / actual_cycles / 1e9)
        
        return DataflowResult(
            strategy_name="weight_stationary",
            total_dram_reads=total_dram_reads,
            total_sram_reads=total_sram_reads,
            total_sram_writes=total_sram_writes,
            data_reuse_factor=round(data_reuse, 2),
            bandwidth_required_GBs=round(bandwidth_required, 3),
            cycle_count=actual_cycles,
        )

    def generate_access_pattern(self, layer: LayerSpec):
        """
        Generate 2D heatmap of PE activation access pattern.
        
        Returns (H_out, W_out) array — berapa kali tiap posisi diakses.
        Berguna untuk visualisasi dan memahami data locality.
        """
        H_out, W_out = layer.H_out, layer.W_out
        access_map = np.zeros((H_out, W_out))
        
        # Dalam WS: setiap output position diakses C_in × K × K kali
        # tapi PE yang berbeda mengerjakan filter berbeda
        accesses_per_pos = layer.C_in * layer.K * layer.K
        
        # Simulate tiling: positions mapped to PE columns
        for j_start in range(0, H_out * W_out, self.pe_cols):
            for j in range(j_start, min(j_start + self.pe_cols, H_out * W_out)):
                oh = j // W_out
                ow = j % W_out
                access_map[oh, ow] = accesses_per_pos
        
        return access_map
