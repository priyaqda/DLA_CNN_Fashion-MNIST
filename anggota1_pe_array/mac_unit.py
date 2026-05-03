"""
anggota1_pe_array/mac_unit.py — Model satu MAC (Multiply-Accumulate) unit.

Ini adalah unit komputasi terkecil di DLA.
Di hardware (Level 2), ini akan menjadi 1 module Verilog.

Rumus dasar:
    acc += weight * activation

Untuk INT8: output = clamp(round(sum(w_i * x_i) * scale))
"""

import numpy as np
from common import config


class MACUnit:
    """
    Model satu Processing Element yang melakukan MAC operation.
    
    Parameters
    ----------
    bit_width : int
        Presisi operand (4, 8, 16, atau 32 bit)
    acc_width : int
        Lebar accumulator (mencegah overflow)
    """

    def __init__(self, bit_width=None, acc_width=None):
        self.bit_width = bit_width or config.DEFAULT_PRECISION
        self.acc_width = acc_width or config.ACCUMULATOR_WIDTH
        self.accumulator = 0
        self.mac_count = 0

        # Range nilai berdasarkan bit-width (signed integer)
        self.val_min = -(2 ** (self.bit_width - 1))
        self.val_max = (2 ** (self.bit_width - 1)) - 1

    def reset(self):
        """Reset accumulator (di hardware: sinyal clear)."""
        self.accumulator = 0

    def mac(self, weight: float, activation: float) -> float:
        """
        Satu operasi multiply-accumulate.
        
        Di hardware ini terjadi dalam 1 clock cycle.
        """
        if self.bit_width < 32:
            weight = np.clip(int(weight), self.val_min, self.val_max)
            activation = np.clip(int(activation), self.val_min, self.val_max)

        product = weight * activation
        self.accumulator += product
        self.mac_count += 1
        return self.accumulator

    def mac_vector(self, weights: np.ndarray, activations: np.ndarray) -> float:
        """
        MAC operation pada vektor (dot product).
        Ini simulasi: di hardware, tiap elemen butuh 1 cycle.
        
        Parameters
        ----------
        weights : np.ndarray, shape (K,)
        activations : np.ndarray, shape (K,)
        
        Returns
        -------
        float : accumulated result
        """
        assert weights.shape == activations.shape, \
            f"Shape mismatch: {weights.shape} vs {activations.shape}"

        self.reset()
        for w, a in zip(weights.flat, activations.flat):
            self.mac(w, a)
        return self.accumulator

    @property
    def energy_pj(self) -> float:
        """Estimasi energi yang dikonsumsi (pJ)."""
        energy_per_mac = {
            8: config.ENERGY_MAC_INT8_PJ,
            16: config.ENERGY_MAC_INT16_PJ,
            32: config.ENERGY_MAC_FP32_PJ,
        }
        e = energy_per_mac.get(self.bit_width, config.ENERGY_MAC_INT8_PJ)
        return self.mac_count * e

    def __repr__(self):
        return (f"MACUnit(bit_width={self.bit_width}, "
                f"macs={self.mac_count}, acc={self.accumulator:.4f})")
