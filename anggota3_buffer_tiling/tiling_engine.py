"""
anggota3_buffer_tiling/tiling_engine.py — Tile size optimizer.

Tiling = membagi feature map besar menjadi tile kecil yang muat di SRAM.
Ukuran tile menentukan trade-off:
- Tile besar  → lebih banyak data reuse, tapi butuh SRAM lebih besar
- Tile kecil  → muat di SRAM, tapi DRAM traffic meningkat (redundant loads)

Tugas utama: cari tile size optimal yang minimize total DRAM traffic
given SRAM budget.
"""

import numpy as np
from common.interfaces import LayerSpec, TilingResult
from common import config
from .sram_model import SRAMBuffer


class TilingEngine:
    """
    Optimizer untuk tile size pada CNN convolution layer.
    """

    def __init__(self, sram: SRAMBuffer = None):
        self.sram = sram or SRAMBuffer()
        self.bit_width = config.DEFAULT_PRECISION

    def _bytes_per_element(self):
        return self.bit_width // 8

    def compute_tile_traffic(self, layer: LayerSpec,
                              tile_h: int, tile_w: int, tile_c: int) -> dict:
        """
        Hitung DRAM traffic untuk tile size tertentu.
        
        Parameters
        ----------
        layer : LayerSpec
        tile_h, tile_w : int
            Tile size untuk output spatial dimensions
        tile_c : int
            Tile size untuk output channels
        
        Returns
        -------
        dict with traffic breakdown
        """
        bpe = self._bytes_per_element()
        K, S = layer.K, layer.stride
        H_out, W_out = layer.H_out, layer.W_out

        # Jumlah tiles
        n_tiles_h = int(np.ceil(H_out / tile_h))
        n_tiles_w = int(np.ceil(W_out / tile_w))
        n_tiles_c = int(np.ceil(layer.C_out / tile_c))
        total_tiles = n_tiles_h * n_tiles_w * n_tiles_c

        # Input tile size (termasuk halo untuk convolution)
        input_tile_h = (tile_h - 1) * S + K
        input_tile_w = (tile_w - 1) * S + K

        # Memory per tile
        weight_per_tile = tile_c * layer.C_in * K * K * bpe
        input_per_tile = layer.C_in * input_tile_h * input_tile_w * bpe
        output_per_tile = tile_c * tile_h * tile_w * bpe
        total_per_tile = weight_per_tile + input_per_tile + output_per_tile

        # Fit in SRAM?
        fits_sram = total_per_tile <= self.sram.total_size_bytes

        # DRAM traffic
        # Weights: loaded once per spatial tile (reused across spatial positions)
        weight_dram = layer.weight_count * bpe * n_tiles_h * n_tiles_w

        # Activations: halo region causes redundant loads
        # Overlap per tile boundary: (K - S) pixels
        halo_overlap = K - S
        unique_input_pixels = layer.C_in * layer.H_in * layer.W_in
        redundant_pixels = (layer.C_in * halo_overlap 
                           * (input_tile_w * (n_tiles_h - 1) 
                              + input_tile_h * (n_tiles_w - 1)))
        act_dram = (unique_input_pixels + redundant_pixels) * bpe * n_tiles_c

        # Output: written once
        output_dram = layer.activation_output_size * bpe

        total_dram = weight_dram + act_dram + output_dram

        return {
            "tile_h": tile_h,
            "tile_w": tile_w,
            "tile_c": tile_c,
            "total_tiles": total_tiles,
            "bytes_per_tile": total_per_tile,
            "fits_sram": fits_sram,
            "dram_weight_bytes": weight_dram,
            "dram_act_bytes": act_dram,
            "dram_output_bytes": output_dram,
            "dram_total_bytes": total_dram,
            "sram_utilization": min(total_per_tile / self.sram.total_size_bytes, 1.0),
        }

    def find_optimal_tile(self, layer: LayerSpec) -> TilingResult:
        """
        Brute-force search untuk tile size optimal.
        Minimize total DRAM traffic subject to SRAM constraint.
        """
        bpe = self._bytes_per_element()
        H_out, W_out = layer.H_out, layer.W_out
        best = None
        best_traffic = float("inf")

        # Search space
        tile_h_range = range(1, H_out + 1)
        tile_w_range = range(1, W_out + 1)
        tile_c_range = [c for c in [1, 2, 4, 8, 16, 32, 64]
                        if c <= layer.C_out]

        for th in tile_h_range:
            for tw in tile_w_range:
                for tc in tile_c_range:
                    result = self.compute_tile_traffic(layer, th, tw, tc)
                    if result["fits_sram"] and result["dram_total_bytes"] < best_traffic:
                        best_traffic = result["dram_total_bytes"]
                        best = result

        if best is None:
            # Nothing fits — use smallest tile
            best = self.compute_tile_traffic(layer, 1, 1, 1)

        # Compute hit rate
        total_data_needed = (layer.total_macs * 2 * bpe)  # weights + activations
        hit_rate = 1.0 - (best["dram_total_bytes"] / total_data_needed) \
            if total_data_needed > 0 else 0

        return TilingResult(
            tile_h=best["tile_h"],
            tile_w=best["tile_w"],
            tile_c=best["tile_c"],
            num_tiles=best["total_tiles"],
            sram_hit_rate=max(0, min(1, hit_rate)),
            dram_traffic_bytes=best["dram_total_bytes"],
            optimal=best["fits_sram"],
        )

    def sweep_tile_sizes(self, layer: LayerSpec) -> list:
        """
        Sweep beberapa tile size representatif dan return hasilnya.
        Berguna untuk plotting trade-off.
        """
        H_out, W_out = layer.H_out, layer.W_out
        results = []

        tile_sizes = [1, 2, 4, 8, 16, 32]
        for t in tile_sizes:
            th = min(t, H_out)
            tw = min(t, W_out)
            tc = min(t, layer.C_out)
            r = self.compute_tile_traffic(layer, th, tw, tc)
            results.append(r)

        return results
