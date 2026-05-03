"""
anggota4_quantization/quantizer.py — Quantization engine.

Quantization = konversi dari floating point (FP32) ke integer (INT8/INT16).
Ini krusial untuk DLA karena:
- INT8 MAC unit ~18x lebih kecil dari FP32 MAC unit
- INT8 SRAM access ~4x lebih hemat energi
- Trade-off: accuracy turun sedikit

Metode: Symmetric quantization (paling mudah di hardware)
  x_q = round(x / scale)
  scale = max(|x|) / (2^(b-1) - 1)
"""

import numpy as np


class Quantizer:
    """
    Symmetric quantizer: FP32 → INTn → FP32 (dequantize).
    """

    def __init__(self, bit_width: int = 8):
        self.bit_width = bit_width
        self.qmin = -(2 ** (bit_width - 1))
        self.qmax = (2 ** (bit_width - 1)) - 1

    def compute_scale(self, tensor: np.ndarray) -> float:
        """Compute quantization scale factor."""
        abs_max = np.abs(tensor).max()
        if abs_max == 0:
            return 1.0
        return abs_max / self.qmax

    def quantize(self, tensor: np.ndarray) -> tuple:
        """
        Quantize FP32 tensor ke INT.
        
        Returns
        -------
        q_tensor : np.ndarray (int)
        scale : float
        """
        scale = self.compute_scale(tensor)
        q_tensor = np.clip(
            np.round(tensor / scale), self.qmin, self.qmax
        ).astype(np.int32)
        return q_tensor, scale

    def dequantize(self, q_tensor: np.ndarray, scale: float) -> np.ndarray:
        """Dequantize back to FP32."""
        return q_tensor.astype(np.float64) * scale

    def quantize_dequantize(self, tensor: np.ndarray) -> np.ndarray:
        """Simulate quantization: quantize then immediately dequantize."""
        q, s = self.quantize(tensor)
        return self.dequantize(q, s)

    def compute_error(self, original: np.ndarray, quantized: np.ndarray) -> dict:
        """Compute quantization error metrics."""
        diff = original - quantized
        mse = np.mean(diff ** 2)
        mae = np.mean(np.abs(diff))
        snr = 10 * np.log10(np.mean(original ** 2) / mse) if mse > 0 else float("inf")
        max_err = np.abs(diff).max()

        return {
            "mse": mse,
            "mae": mae,
            "snr_db": snr,
            "max_error": max_err,
            "relative_error_pct": mae / np.abs(original).mean() * 100 
                                   if np.abs(original).mean() > 0 else 0,
        }


class PerChannelQuantizer(Quantizer):
    """
    Per-channel quantization: setiap output channel punya scale sendiri.
    Lebih akurat dari per-tensor, tapi butuh lebih banyak scale storage.
    """

    def quantize_weights(self, weights: np.ndarray) -> tuple:
        """
        Quantize weights per output channel.
        
        Parameters
        ----------
        weights : shape (C_out, C_in, K, K)
        
        Returns
        -------
        q_weights : np.ndarray (int)
        scales : np.ndarray, shape (C_out,)
        """
        C_out = weights.shape[0]
        scales = np.zeros(C_out)
        q_weights = np.zeros_like(weights, dtype=np.int32)

        for c in range(C_out):
            channel_weights = weights[c]
            scales[c] = self.compute_scale(channel_weights)
            q_weights[c] = np.clip(
                np.round(channel_weights / scales[c]),
                self.qmin, self.qmax
            ).astype(np.int32)

        return q_weights, scales

    def dequantize_weights(self, q_weights, scales):
        """Dequantize per-channel weights."""
        result = np.zeros_like(q_weights, dtype=np.float64)
        for c in range(q_weights.shape[0]):
            result[c] = q_weights[c].astype(np.float64) * scales[c]
        return result
