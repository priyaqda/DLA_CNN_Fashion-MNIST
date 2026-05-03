"""
anggota5_integration/gpu_baseline.py — GPU theoretical baseline.

Menyediakan estimasi performa GPU sebagai pembanding DLA.
Bukan simulasi GPU aktual, tapi model analitik berdasarkan specs.

Tujuan: menunjukkan bahwa DLA (walaupun clock-nya jauh lebih rendah)
bisa lebih energy-efficient untuk workload CNN spesifik.
"""

import numpy as np
from common import config


class GPUBaseline:
    """
    Model analitik performa GPU untuk perbandingan.
    Menggunakan spesifikasi entry-level GPU (mirip level mobile GPU).
    """

    # Reference: entry-level mobile GPU (Adreno 610-class)
    PRESETS = {
        "mobile_gpu": {
            "name": "Mobile GPU (entry-level)",
            "clock_mhz": 850,
            "cuda_cores": 128,  # shader processors
            "ops_per_core_per_cycle": 2,  # FMA = 2 ops
            "tdp_w": 3.0,
            "mem_bw_gbs": 13,
            "precision": "FP16",
        },
        "desktop_gpu": {
            "name": "Desktop GPU (mid-range)",
            "clock_mhz": 1500,
            "cuda_cores": 1024,
            "ops_per_core_per_cycle": 2,
            "tdp_w": 75,
            "mem_bw_gbs": 192,
            "precision": "FP16",
        },
        "dla_equivalent": {
            "name": "DLA (this project)",
            "clock_mhz": config.CLOCK_FREQ_MHZ,
            "cuda_cores": config.PE_ARRAY_ROWS * config.PE_ARRAY_COLS,
            "ops_per_core_per_cycle": 1,  # 1 MAC = 1 op
            "tdp_w": None,  # computed from model
            "mem_bw_gbs": config.DRAM_BANDWIDTH_GBs,
            "precision": f"INT{config.DEFAULT_PRECISION}",
        }
    }

    def __init__(self, preset="mobile_gpu"):
        self.specs = self.PRESETS[preset].copy()

    @property
    def peak_gops(self):
        return (self.specs["cuda_cores"]
                * self.specs["ops_per_core_per_cycle"]
                * self.specs["clock_mhz"] * 1e6 / 1e9)

    def estimate_inference(self, total_macs: int) -> dict:
        """
        Estimasi inference time dan energy.
        Asumsi: 60% utilization (typical for small models on GPU).
        """
        utilization = 0.6
        effective_gops = self.peak_gops * utilization
        
        time_s = total_macs / (effective_gops * 1e9) if effective_gops > 0 else 0
        throughput = total_macs / time_s / 1e9 if time_s > 0 else 0
        
        tdp = self.specs["tdp_w"]
        if tdp is None:
            from anggota4_quantization.hw_cost_model import HardwareCostModel
            hw = HardwareCostModel()
            power = hw.total_power_mw(
                config.PE_ARRAY_ROWS, config.PE_ARRAY_COLS,
                config.DEFAULT_PRECISION, utilization)
            tdp = power["total_mW"] / 1000
        
        energy_j = tdp * time_s
        efficiency = throughput / tdp if tdp > 0 else 0

        return {
            "name": self.specs["name"],
            "precision": self.specs["precision"],
            "peak_gops": round(self.peak_gops, 2),
            "actual_gops": round(throughput, 3),
            "utilization": utilization,
            "inference_time_ms": round(time_s * 1000, 4),
            "energy_uJ": round(energy_j * 1e6, 3),
            "efficiency_gops_w": round(efficiency, 2),
            "tdp_w": round(tdp, 4),
        }


def compare_dla_vs_gpu(dla_benchmark, total_macs: int) -> dict:
    """
    Perbandingan langsung DLA vs Mobile GPU vs Desktop GPU.
    
    Parameters
    ----------
    dla_benchmark : DLABenchmark dari simulator
    total_macs : int
    
    Returns
    -------
    dict with comparison table data
    """
    mobile = GPUBaseline("mobile_gpu").estimate_inference(total_macs)
    desktop = GPUBaseline("desktop_gpu").estimate_inference(total_macs)

    dla_time_s = dla_benchmark.total_cycles / (config.CLOCK_FREQ_MHZ * 1e6)

    results = {
        "DLA (ours)": {
            "precision": f"INT{config.DEFAULT_PRECISION}",
            "peak_gops": round(config.PE_ARRAY_ROWS * config.PE_ARRAY_COLS 
                               * config.CLOCK_FREQ_MHZ * 1e6 / 1e9, 2),
            "actual_gops": dla_benchmark.throughput_gops,
            "inference_ms": round(dla_time_s * 1000, 4),
            "energy_uJ": dla_benchmark.energy_total_uJ,
            "efficiency": dla_benchmark.energy_efficiency_gops_w,
        },
        "Mobile GPU": {
            "precision": mobile["precision"],
            "peak_gops": mobile["peak_gops"],
            "actual_gops": mobile["actual_gops"],
            "inference_ms": mobile["inference_time_ms"],
            "energy_uJ": mobile["energy_uJ"],
            "efficiency": mobile["efficiency_gops_w"],
        },
        "Desktop GPU": {
            "precision": desktop["precision"],
            "peak_gops": desktop["peak_gops"],
            "actual_gops": desktop["actual_gops"],
            "inference_ms": desktop["inference_time_ms"],
            "energy_uJ": desktop["energy_uJ"],
            "efficiency": desktop["efficiency_gops_w"],
        },
    }

    return results
