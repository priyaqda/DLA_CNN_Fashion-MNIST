"""
anggota2_dataflow/output_stationary.py — Output Stationary Dataflow.

Dalam Output Stationary (OS):
- Partial sums (output) TETAP di PE accumulator
- Weights dan activations di-stream masuk
- Keuntungan: minimize partial sum movement → hemat energy untuk accumulation
- Kerugian: weights dan activations harus di-fetch berkali-kali

Contoh chip: ShiDianNao (2015), NVDLA
"""

import numpy as np
from common.interfaces import LayerSpec, DataflowResult
from common import config


class OutputStationaryDataflow:
    """
    Simulasi dataflow output-stationary pada PE array.
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
        Simulasi data movement untuk output-stationary.
        
        Output Stationary strategy:
        1. Assign output positions ke PE (tetap sampai selesai)
        2. Stream (weight, activation) pairs masuk ke PE
        3. Accumulator di PE mengumpulkan partial sum
        4. Setelah semua C_in × K × K selesai, output di-flush
        """
        bpe = self._bytes_per_element()
        
        total_macs = layer.total_macs
        total_weights = layer.weight_count
        total_act = layer.activation_input_size
        output_elements = layer.activation_output_size
        
        # === Output assignment ===
        # Setiap PE meng-hold 1 output element sampai fully accumulated
        # MxN PE → MxN outputs computed simultaneously
        outputs_per_batch = self.pe_rows * self.pe_cols
        num_output_batches = int(np.ceil(output_elements / outputs_per_batch))
        
        # === Weight streaming ===
        # Untuk setiap output batch, SEMUA weights harus di-stream masuk
        # (karena weights tidak di-store di PE)
        weight_bytes = total_weights * bpe
        weights_fit_sram = weight_bytes <= self.sram_size_bytes * 0.4
        
        if weights_fit_sram:
            dram_weight_reads = total_weights  # load once to SRAM
            sram_weight_reads = total_weights * num_output_batches
        else:
            dram_weight_reads = total_weights * num_output_batches
            sram_weight_reads = dram_weight_reads
        
        # === Activation streaming ===
        # Setiap output batch butuh subset of activations (receptive field)
        # Overlap antara adjacent output positions → ada reuse
        
        # Unique activations per output position: K × K
        # Overlap with neighbors: (K-stride) pixels shared
        # Total unique activations needed: all input activations
        
        # Dalam OS: activations di-stream bersama weights
        # Setiap output batch: perlu C_in × K × K activations per output
        act_per_output_batch = min(
            total_act,
            outputs_per_batch * layer.C_in * layer.K * layer.K
        )
        
        act_bytes = total_act * bpe
        acts_fit_sram = act_bytes <= self.sram_size_bytes * 0.4
        
        if acts_fit_sram:
            dram_act_reads = total_act  # load once
            sram_act_reads = act_per_output_batch * num_output_batches
        else:
            dram_act_reads = act_per_output_batch * num_output_batches
            sram_act_reads = dram_act_reads
        
        # === Output writes ===
        # In OS: partial sums stay in PE → no intermediate SRAM writes
        # Only final flush to SRAM then DRAM
        sram_output_writes = output_elements  # only final values!
        dram_output_writes = output_elements
        
        # === Totals ===
        total_dram_reads = dram_weight_reads + dram_act_reads
        total_sram_reads = sram_weight_reads + sram_act_reads
        total_sram_writes = sram_output_writes
        
        data_reuse = total_macs / total_dram_reads if total_dram_reads > 0 else 0
        
        # Cycles
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
            strategy_name="output_stationary",
            total_dram_reads=total_dram_reads,
            total_sram_reads=total_sram_reads,
            total_sram_writes=total_sram_writes,
            data_reuse_factor=round(data_reuse, 2),
            bandwidth_required_GBs=round(bandwidth_required, 3),
            cycle_count=actual_cycles,
        )

    def generate_access_pattern(self, layer: LayerSpec):
        """
        Generate heatmap: berapa kali tiap input position diakses.
        Dalam OS, setiap input diakses oleh semua filter yang receptive field-nya
        mencakup posisi tersebut.
        """
        H_in, W_in = layer.H_in, layer.W_in
        access_map = np.zeros((H_in, W_in))
        
        H_out, W_out = layer.H_out, layer.W_out
        K, S = layer.K, layer.stride
        
        for oh in range(H_out):
            for ow in range(W_out):
                ih_start = oh * S
                iw_start = ow * S
                # This input patch is accessed C_out times (once per filter)
                access_map[ih_start:ih_start+K, iw_start:iw_start+K] += layer.C_out
        
        return access_map
