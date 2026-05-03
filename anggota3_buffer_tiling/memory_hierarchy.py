"""
anggota3_buffer_tiling/memory_hierarchy.py — DRAM ↔ SRAM ↔ PE data movement.

Model hierarki memori lengkap:
  DRAM (off-chip, besar, lambat, mahal energi)
    ↕
  SRAM (on-chip, kecil, cepat, murah energi)
    ↕
  Register File di PE (sangat kecil, 0-cycle access)
"""

import numpy as np
from common.interfaces import LayerSpec
from common import config
from .sram_model import SRAMBuffer


class MemoryHierarchy:
    """
    Model lengkap hierarki memori DLA.
    """

    def __init__(self, sram: SRAMBuffer = None):
        self.sram = sram or SRAMBuffer()

    def compute_total_energy(self, layer: LayerSpec,
                              dram_reads: int, dram_writes: int,
                              sram_reads: int, sram_writes: int,
                              mac_count: int) -> dict:
        """
        Hitung total energy breakdown.
        
        Returns dict with energy per component (in pJ and percentage).
        """
        bpe = config.DEFAULT_PRECISION // 8

        e_mac = mac_count * config.ENERGY_MAC_INT8_PJ
        e_sram_r = sram_reads * config.ENERGY_SRAM_READ_PJ
        e_sram_w = sram_writes * config.ENERGY_SRAM_WRITE_PJ
        e_dram_r = dram_reads * config.ENERGY_DRAM_READ_PJ
        e_dram_w = dram_writes * config.ENERGY_DRAM_READ_PJ  # write ≈ read energy
        
        total = e_mac + e_sram_r + e_sram_w + e_dram_r + e_dram_w

        return {
            "compute_pJ": e_mac,
            "sram_pJ": e_sram_r + e_sram_w,
            "dram_pJ": e_dram_r + e_dram_w,
            "total_pJ": total,
            "compute_pct": e_mac / total * 100 if total > 0 else 0,
            "sram_pct": (e_sram_r + e_sram_w) / total * 100 if total > 0 else 0,
            "dram_pct": (e_dram_r + e_dram_w) / total * 100 if total > 0 else 0,
        }

    def compute_bandwidth_requirement(self, layer: LayerSpec, 
                                        total_cycles: int) -> dict:
        """
        Hitung bandwidth DRAM yang dibutuhkan dan apakah memory-bound.
        """
        bpe = config.DEFAULT_PRECISION // 8
        
        # Minimal DRAM traffic (no tiling overhead)
        min_dram_bytes = (layer.weight_count + layer.activation_input_size 
                          + layer.activation_output_size) * bpe
        
        # Available bandwidth
        cycle_time_s = 1.0 / (config.CLOCK_FREQ_MHZ * 1e6)
        total_time_s = total_cycles * cycle_time_s
        required_bw = min_dram_bytes / total_time_s / 1e9  # GB/s
        
        # Compute intensity (MACs per byte of DRAM)
        compute_intensity = layer.total_macs / min_dram_bytes
        
        # Roofline: memory-bound if compute_intensity < peak_ops/bandwidth
        pe_array_size = config.PE_ARRAY_ROWS * config.PE_ARRAY_COLS
        peak_ops_per_byte = (pe_array_size * config.CLOCK_FREQ_MHZ * 1e6 
                             / (config.DRAM_BANDWIDTH_GBs * 1e9))
        
        memory_bound = compute_intensity < peak_ops_per_byte
        
        return {
            "min_dram_bytes": min_dram_bytes,
            "required_bandwidth_GBs": round(required_bw, 3),
            "available_bandwidth_GBs": config.DRAM_BANDWIDTH_GBs,
            "compute_intensity": round(compute_intensity, 2),
            "roofline_threshold": round(peak_ops_per_byte, 2),
            "memory_bound": memory_bound,
        }
