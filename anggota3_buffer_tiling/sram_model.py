"""
anggota3_buffer_tiling/sram_model.py — Model SRAM on-chip buffer.

SRAM adalah jembatan antara DRAM (lambat, murah) dan PE (cepat, mahal).
Desain buffer menentukan seberapa sering PE harus menunggu (stall).

Di Level 2 (RTL): ini menjadi SRAM macro atau register file.
Di Level 3 (Physical): ini menjadi hardmacro yang di-place secara manual.
"""

import numpy as np
from common import config


class SRAMBuffer:
    """
    Model SRAM buffer multi-bank.
    
    Parameters
    ----------
    total_size_kb : int
        Total SRAM capacity dalam KB
    num_banks : int
        Jumlah SRAM bank (parallel access)
    read_latency : int
        Read latency dalam clock cycles
    write_latency : int
        Write latency dalam clock cycles
    """

    def __init__(self, total_size_kb=None, num_banks=None,
                 read_latency=None, write_latency=None):
        self.total_size_kb = total_size_kb or config.SRAM_SIZE_KB
        self.total_size_bytes = self.total_size_kb * 1024
        self.num_banks = num_banks or config.SRAM_BANKS
        self.bank_size_bytes = self.total_size_bytes // self.num_banks
        self.read_latency = read_latency or config.SRAM_READ_LATENCY
        self.write_latency = write_latency or config.SRAM_WRITE_LATENCY

        # Partitioning: weight buffer, activation buffer, output buffer
        # Default: 25% weights, 50% activations, 25% output
        self.weight_buf_bytes = self.total_size_bytes // 4
        self.act_buf_bytes = self.total_size_bytes // 2
        self.out_buf_bytes = self.total_size_bytes // 4

        # Statistics
        self.stats = {
            "reads": 0,
            "writes": 0,
            "read_hits": 0,
            "read_misses": 0,
            "stall_cycles": 0,
        }

    def reset_stats(self):
        for k in self.stats:
            self.stats[k] = 0

    def can_fit(self, data_type: str, size_bytes: int) -> bool:
        """Check apakah data muat di buffer partition."""
        budget = {
            "weight": self.weight_buf_bytes,
            "activation": self.act_buf_bytes,
            "output": self.out_buf_bytes,
        }
        return size_bytes <= budget.get(data_type, 0)

    def compute_partitioning(self, weight_bytes, act_bytes, out_bytes):
        """
        Hitung partitioning optimal berdasarkan kebutuhan aktual.
        
        Returns
        -------
        dict dengan alokasi dan apakah masing-masing fit
        """
        total_needed = weight_bytes + act_bytes + out_bytes

        if total_needed <= self.total_size_bytes:
            # Semua muat — ideal
            return {
                "weight_alloc": weight_bytes,
                "act_alloc": act_bytes,
                "out_alloc": out_bytes,
                "all_fit": True,
                "utilization": total_needed / self.total_size_bytes,
            }

        # Tidak muat semua — prioritize berdasarkan reuse
        # Strategy: weights paling sering di-reuse → prioritas tertinggi
        weight_alloc = min(weight_bytes, int(self.total_size_bytes * 0.4))
        remaining = self.total_size_bytes - weight_alloc
        act_alloc = min(act_bytes, int(remaining * 0.7))
        out_alloc = remaining - act_alloc

        return {
            "weight_alloc": weight_alloc,
            "act_alloc": act_alloc,
            "out_alloc": out_alloc,
            "all_fit": False,
            "utilization": 1.0,
            "weight_fit": weight_bytes <= weight_alloc,
            "act_fit": act_bytes <= act_alloc,
            "out_fit": out_bytes <= out_alloc,
        }

    @property
    def energy_per_read_pJ(self):
        return config.ENERGY_SRAM_READ_PJ

    @property
    def energy_per_write_pJ(self):
        return config.ENERGY_SRAM_WRITE_PJ

    @property
    def area_estimate_mm2(self):
        """
        Estimasi area SRAM untuk 180nm.
        Rule of thumb: ~1 mm² per 64KB di 180nm.
        """
        return self.total_size_kb / 64.0

    def __repr__(self):
        return (f"SRAMBuffer({self.total_size_kb}KB, "
                f"{self.num_banks} banks, "
                f"~{self.area_estimate_mm2:.2f} mm²)")
