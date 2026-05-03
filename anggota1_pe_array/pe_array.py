"""
anggota1_pe_array/pe_array.py — Model NxM PE Array.

PE Array adalah jantung DLA. Array berukuran M×N bekerja paralel:
- M baris  → memproses M output channel secara paralel
- N kolom  → memproses N output pixel secara paralel

Setiap PE melakukan MAC secara independen dalam 1 cycle.
Total throughput ideal: M × N MACs/cycle.

Untuk konvolusi C_out filter × C_in channels × K×K kernel:
- Outer loop: iterate over C_in (input channels)
- Setiap iterasi: seluruh array aktif, masing-masing PE
  mengakumulasi partial sum dari channel tersebut
"""

import numpy as np
from .mac_unit import MACUnit
from common import config
from common.interfaces import PEArrayResult


class PEArray:
    """
    Model M×N Processing Element Array.
    
    Parameters
    ----------
    rows : int
        Jumlah PE baris (dimensi output channel)
    cols : int
        Jumlah PE kolom (dimensi spatial output)
    bit_width : int
        Presisi per operand
    """

    def __init__(self, rows=None, cols=None, bit_width=None):
        self.rows = rows or config.PE_ARRAY_ROWS
        self.cols = cols or config.PE_ARRAY_COLS
        self.bit_width = bit_width or config.DEFAULT_PRECISION

        # Inisialisasi grid PE
        self.pe_grid = [
            [MACUnit(bit_width=self.bit_width) for _ in range(self.cols)]
            for _ in range(self.rows)
        ]
        self.total_cycles = 0

    @property
    def peak_throughput_mac_per_cycle(self) -> int:
        """MACs per cycle jika 100% utilized."""
        return self.rows * self.cols

    @property
    def peak_gops(self) -> float:
        """Peak throughput dalam GOPS."""
        return (self.peak_throughput_mac_per_cycle 
                * config.CLOCK_FREQ_MHZ * 1e6) / 1e9

    def reset_all(self):
        """Reset semua PE."""
        for row in self.pe_grid:
            for pe in row:
                pe.reset()
        self.total_cycles = 0

    def compute_conv2d(self, input_act, weights, stride=1, padding=0):
        """
        Simulasi konvolusi 2D pada PE array.
        
        Parameters
        ----------
        input_act : np.ndarray, shape (C_in, H, W)
            Input activation (feature map)
        weights : np.ndarray, shape (C_out, C_in, K, K)
            Convolution filters
        stride : int
        padding : int
        
        Returns
        -------
        PEArrayResult
        """
        C_out, C_in, K, _ = weights.shape
        _, H, W = input_act.shape

        # Apply padding
        if padding > 0:
            input_act = np.pad(input_act,
                               ((0, 0), (padding, padding), (padding, padding)))
            _, H, W = input_act.shape

        H_out = (H - K) // stride + 1
        W_out = (W - K) // stride + 1
        output = np.zeros((C_out, H_out, W_out))

        total_macs = 0
        total_cycles = 0

        # Tiling: map computation ke PE array
        # M PE rows → handle M output filters at a time
        # N PE cols → handle N output positions at a time
        for f_start in range(0, C_out, self.rows):
            f_end = min(f_start + self.rows, C_out)
            n_filters = f_end - f_start

            for pos_start in range(0, H_out * W_out, self.cols):
                pos_end = min(pos_start + self.cols, H_out * W_out)
                n_positions = pos_end - pos_start

                # Untuk setiap (filter, position) pair yang di-assign ke PE:
                for c in range(C_in):
                    for ki in range(K):
                        for kj in range(K):
                            # 1 cycle: semua PE aktif secara paralel
                            total_cycles += 1

                            for fi in range(n_filters):
                                f_idx = f_start + fi
                                for pi in range(n_positions):
                                    p_idx = pos_start + pi
                                    oh = p_idx // W_out
                                    ow = p_idx % W_out

                                    ih = oh * stride + ki
                                    iw = ow * stride + kj

                                    w = weights[f_idx, c, ki, kj]
                                    a = input_act[c, ih, iw]
                                    output[f_idx, oh, ow] += w * a
                                    total_macs += 1

        # Hitung utilization
        ideal_cycles = total_macs / self.peak_throughput_mac_per_cycle
        utilization = ideal_cycles / total_cycles if total_cycles > 0 else 0

        self.total_cycles = total_cycles

        return PEArrayResult(
            output_feature_map=output,
            total_macs=total_macs,
            total_cycles=total_cycles,
            utilization=min(utilization, 1.0),
            pe_rows=self.rows,
            pe_cols=self.cols
        )

    def profile_layer(self, layer_spec):
        """
        Quick profiling tanpa compute — hanya hitung MACs dan cycles.
        
        Parameters
        ----------
        layer_spec : LayerSpec
        
        Returns
        -------
        dict with profiling metrics
        """
        total_macs = layer_spec.total_macs
        ideal_cycles = total_macs / self.peak_throughput_mac_per_cycle

        # Overhead: tiling inefficiency (edge tiles underutilized)
        n_filter_tiles = int(np.ceil(layer_spec.C_out / self.rows))
        n_spatial_tiles = int(np.ceil(
            layer_spec.H_out * layer_spec.W_out / self.cols))

        # Effective utilization
        last_filter_util = (layer_spec.C_out % self.rows or self.rows) / self.rows
        last_spatial_util = ((layer_spec.H_out * layer_spec.W_out) 
                             % self.cols or self.cols) / self.cols
        avg_util = (1.0 - (1 - last_filter_util) / n_filter_tiles) * \
                   (1.0 - (1 - last_spatial_util) / n_spatial_tiles)

        actual_cycles = int(np.ceil(ideal_cycles / avg_util))

        latency_us = actual_cycles / (config.CLOCK_FREQ_MHZ * 1e6) * 1e6

        return {
            "layer": layer_spec.name,
            "total_macs": total_macs,
            "ideal_cycles": int(ideal_cycles),
            "actual_cycles": actual_cycles,
            "utilization": round(avg_util, 3),
            "latency_us": round(latency_us, 3),
            "throughput_gops": round(
                total_macs / actual_cycles 
                * config.CLOCK_FREQ_MHZ * 1e6 / 1e9, 2),
        }

    def __repr__(self):
        return (f"PEArray({self.rows}×{self.cols}, "
                f"bit_width={self.bit_width}, "
                f"peak={self.peak_gops:.1f} GOPS)")
