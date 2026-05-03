"""
anggota4_quantization/hw_cost_model.py — Hardware cost model vs precision.

Estimasi area dan power dari MAC unit berdasarkan bit-width.
Sumber: scaling rules dari literatur VLSI.

Rules of thumb (untuk 180nm):
- Area MAC ∝ (bit_width)²  (karena multiplier area scales quadratically)
- Power MAC ∝ (bit_width)²  
- SRAM per element ∝ bit_width (linear)
"""

import numpy as np
from common import config


class HardwareCostModel:
    """
    Estimasi area dan power berdasarkan bit-width dan array size.
    Semua angka di-normalize relatif terhadap INT8 baseline.
    """

    # Reference: INT8 MAC unit di 180nm
    REF_BIT_WIDTH = 8
    REF_MAC_AREA_UM2 = 2500        # ~50μm × 50μm per INT8 MAC
    REF_MAC_POWER_UW = 50          # μW per MAC at 200MHz
    REF_SRAM_AREA_PER_BIT_UM2 = 1  # μm² per bit (180nm 6T SRAM)

    def mac_area_um2(self, bit_width: int) -> float:
        """Area satu MAC unit (μm²)."""
        scale = (bit_width / self.REF_BIT_WIDTH) ** 2
        return self.REF_MAC_AREA_UM2 * scale

    def mac_power_uw(self, bit_width: int) -> float:
        """Dynamic power satu MAC unit (μW) at CLOCK_FREQ."""
        scale = (bit_width / self.REF_BIT_WIDTH) ** 2
        return self.REF_MAC_POWER_UW * scale

    def array_area_mm2(self, rows: int, cols: int, bit_width: int) -> float:
        """Total area PE array (mm²)."""
        n_pes = rows * cols
        mac_area = self.mac_area_um2(bit_width) * n_pes
        # Add accumulator registers: ACC_WIDTH bits per PE
        acc_area = config.ACCUMULATOR_WIDTH * self.REF_SRAM_AREA_PER_BIT_UM2 * n_pes
        # Overhead: interconnect, routing (~30%)
        total = (mac_area + acc_area) * 1.3
        return total / 1e6  # μm² → mm²

    def sram_area_mm2(self, size_kb: int) -> float:
        """SRAM area (mm²)."""
        size_bits = size_kb * 1024 * 8
        area_um2 = size_bits * self.REF_SRAM_AREA_PER_BIT_UM2
        return area_um2 / 1e6 * 1.2  # 20% overhead for peripherals

    def total_chip_area_mm2(self, rows, cols, bit_width, sram_kb) -> dict:
        """Total chip area estimation."""
        pe_area = self.array_area_mm2(rows, cols, bit_width)
        sram_area = self.sram_area_mm2(sram_kb)
        ctrl_area = pe_area * 0.15  # controller ~15% of PE area
        io_area = 0.5  # I/O pads estimate
        total = pe_area + sram_area + ctrl_area + io_area

        return {
            "pe_array_mm2": round(pe_area, 3),
            "sram_mm2": round(sram_area, 3),
            "controller_mm2": round(ctrl_area, 3),
            "io_mm2": io_area,
            "total_mm2": round(total, 3),
            "pe_pct": round(pe_area / total * 100, 1),
            "sram_pct": round(sram_area / total * 100, 1),
        }

    def total_power_mw(self, rows, cols, bit_width, utilization=1.0) -> dict:
        """Total power estimation (mW)."""
        n_pes = rows * cols
        pe_dynamic = self.mac_power_uw(bit_width) * n_pes * utilization / 1000
        pe_leakage = pe_dynamic * 0.1  # leakage ~10% of dynamic at 180nm
        sram_power = 5.0  # mW estimate for 64KB SRAM
        ctrl_power = pe_dynamic * 0.05
        total = pe_dynamic + pe_leakage + sram_power + ctrl_power

        return {
            "pe_dynamic_mW": round(pe_dynamic, 2),
            "pe_leakage_mW": round(pe_leakage, 2),
            "sram_mW": sram_power,
            "controller_mW": round(ctrl_power, 2),
            "total_mW": round(total, 2),
        }

    def compare_precisions(self, rows=None, cols=None) -> list:
        """
        Bandingkan semua supported precision levels.
        Output: tabel perbandingan area, power, dan relative savings.
        """
        rows = rows or config.PE_ARRAY_ROWS
        cols = cols or config.PE_ARRAY_COLS
        sram_kb = config.SRAM_SIZE_KB
        results = []

        for bw in config.SUPPORTED_PRECISIONS:
            area = self.total_chip_area_mm2(rows, cols, bw, sram_kb)
            power = self.total_power_mw(rows, cols, bw)

            results.append({
                "bit_width": bw,
                "total_area_mm2": area["total_mm2"],
                "total_power_mW": power["total_mW"],
                "area_vs_fp32": area["total_mm2"],  # will be normalized below
                "power_vs_fp32": power["total_mW"],
            })

        # Normalize to FP32
        fp32 = [r for r in results if r["bit_width"] == 32][0]
        for r in results:
            r["area_reduction"] = round(1 - r["total_area_mm2"] / fp32["total_area_mm2"], 3)
            r["power_reduction"] = round(1 - r["total_power_mW"] / fp32["total_power_mW"], 3)

        return results
