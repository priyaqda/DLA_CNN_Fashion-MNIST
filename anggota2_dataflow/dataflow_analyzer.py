"""
anggota2_dataflow/dataflow_analyzer.py — Analisis perbandingan dataflow.

Membandingkan Weight Stationary vs Output Stationary secara kuantitatif.
"""

import numpy as np
from common.interfaces import LayerSpec, DataflowResult
from common import config, utils
from .weight_stationary import WeightStationaryDataflow
from .output_stationary import OutputStationaryDataflow


class DataflowAnalyzer:
    """
    Membandingkan strategi dataflow pada workload CNN.
    """

    def __init__(self, pe_rows=None, pe_cols=None, sram_size_kb=None):
        self.ws = WeightStationaryDataflow(pe_rows, pe_cols, sram_size_kb)
        self.os = OutputStationaryDataflow(pe_rows, pe_cols, sram_size_kb)

    def compare_layer(self, layer: LayerSpec) -> dict:
        """Bandingkan WS vs OS untuk satu layer."""
        ws_result = self.ws.simulate(layer)
        os_result = self.os.simulate(layer)
        
        return {
            "layer": layer.name,
            "total_macs": layer.total_macs,
            "ws": ws_result,
            "os": os_result,
            "dram_ratio_ws_vs_os": (ws_result.total_dram_reads 
                                     / os_result.total_dram_reads 
                                     if os_result.total_dram_reads > 0 else 0),
        }

    def compare_network(self, layer_specs) -> list:
        """Bandingkan WS vs OS untuk seluruh network."""
        results = []
        for spec in layer_specs:
            if isinstance(spec, tuple):
                spec = LayerSpec(*spec)
            results.append(self.compare_layer(spec))
        return results

    def print_comparison(self, layer_specs):
        """Print tabel perbandingan yang rapi."""
        comparisons = self.compare_network(layer_specs)
        
        headers = [
            "Layer", "MACs",
            "WS DRAM", "OS DRAM",
            "WS Reuse", "OS Reuse",
            "WS BW(GB/s)", "OS BW(GB/s)",
            "Winner"
        ]
        rows = []
        for c in comparisons:
            ws, os = c["ws"], c["os"]
            winner = "WS" if ws.total_dram_reads <= os.total_dram_reads else "OS"
            rows.append([
                c["layer"],
                f"{c['total_macs']:,}",
                f"{ws.total_dram_reads:,}",
                f"{os.total_dram_reads:,}",
                f"{ws.data_reuse_factor:.1f}",
                f"{os.data_reuse_factor:.1f}",
                f"{ws.bandwidth_required_GBs:.2f}",
                f"{os.bandwidth_required_GBs:.2f}",
                winner,
            ])
        
        utils.print_table(headers, rows, "Dataflow Comparison: WS vs OS")

    def compute_energy_breakdown(self, layer: LayerSpec) -> dict:
        """
        Hitung breakdown energi per komponen untuk kedua dataflow.
        
        Output: dict dengan energy breakdown yang bisa di-plot.
        Dipakai Anggota 5 untuk benchmark.
        """
        bpe = config.DEFAULT_PRECISION // 8
        ws = self.ws.simulate(layer)
        os = self.os.simulate(layer)
        
        def energy_for(result: DataflowResult):
            e_mac = layer.total_macs * config.ENERGY_MAC_INT8_PJ
            e_sram_r = result.total_sram_reads * config.ENERGY_SRAM_READ_PJ
            e_sram_w = result.total_sram_writes * config.ENERGY_SRAM_WRITE_PJ
            e_dram = result.total_dram_reads * config.ENERGY_DRAM_READ_PJ
            return {
                "compute_pJ": e_mac,
                "sram_read_pJ": e_sram_r,
                "sram_write_pJ": e_sram_w,
                "dram_pJ": e_dram,
                "total_pJ": e_mac + e_sram_r + e_sram_w + e_dram,
            }
        
        return {
            "layer": layer.name,
            "ws_energy": energy_for(ws),
            "os_energy": energy_for(os),
        }
