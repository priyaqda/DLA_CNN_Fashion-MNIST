"""
anggota1_pe_array/conv_engine.py — CNN Convolution Engine.

Wraps PE Array untuk menjalankan multi-layer CNN inference.
Menyediakan interface yang dipakai Anggota 5 (Integration).
"""

import numpy as np
from .pe_array import PEArray
from common.interfaces import LayerSpec, PEArrayResult
from common import config


class ConvEngine:
    """
    Engine konvolusi yang menggunakan PE Array.
    Bisa menjalankan sequence of conv layers (CNN inference).
    """

    def __init__(self, pe_rows=None, pe_cols=None, bit_width=None):
        self.pe_array = PEArray(
            rows=pe_rows, cols=pe_cols, bit_width=bit_width
        )

    def run_layer(self, input_act, weights, bias=None,
                  stride=1, padding=0, relu=True):
        """
        Jalankan satu conv layer pada PE array.
        
        Parameters
        ----------
        input_act : np.ndarray, shape (C_in, H, W)
        weights : np.ndarray, shape (C_out, C_in, K, K)
        bias : np.ndarray, shape (C_out,), optional
        stride, padding : int
        relu : bool, apply ReLU activation
        
        Returns
        -------
        output : np.ndarray
        result : PEArrayResult
        """
        self.pe_array.reset_all()
        result = self.pe_array.compute_conv2d(
            input_act, weights, stride, padding
        )

        output = result.output_feature_map

        # Bias addition (1 cycle per output element, tapi bukan MAC)
        if bias is not None:
            for c in range(output.shape[0]):
                output[c] += bias[c]

        # ReLU activation (1 cycle, comparator di hardware)
        if relu:
            output = np.maximum(output, 0)

        result.output_feature_map = output
        return output, result

    def profile_network(self, layer_specs):
        """
        Profile seluruh CNN tanpa compute.
        
        Parameters
        ----------
        layer_specs : list of LayerSpec
        
        Returns
        -------
        list of dict (per-layer profiling)
        """
        results = []
        for spec in layer_specs:
            if isinstance(spec, tuple):
                spec = LayerSpec(*spec)
            profile = self.pe_array.profile_layer(spec)
            results.append(profile)
        return results

    def run_lenet5_inference(self, input_image):
        """
        Jalankan LeNet-5 inference end-to-end.
        
        Parameters
        ----------
        input_image : np.ndarray, shape (1, 28, 28) — grayscale MNIST
        
        Returns
        -------
        output : np.ndarray, shape (10,) — class logits
        all_results : list of PEArrayResult
        """
        np.random.seed(42)  # reproducible random weights
        all_results = []
        x = input_image

        # Conv1: 1→6, 5x5
        w1 = np.random.randn(6, 1, 5, 5) * 0.1
        b1 = np.zeros(6)
        x, r = self.run_layer(x, w1, b1, relu=True)
        all_results.append(r)
        # Pooling: avg pool 2x2 (simplified)
        x = x[:, ::2, ::2]

        # Conv2: 6→16, 5x5
        w2 = np.random.randn(16, 6, 5, 5) * 0.1
        b2 = np.zeros(16)
        x, r = self.run_layer(x, w2, b2, relu=True)
        all_results.append(r)
        x = x[:, ::2, ::2]

        # Flatten → FC layers as 1x1 conv
        x = x.reshape(-1, 1, 1)

        w3 = np.random.randn(120, x.shape[0], 1, 1) * 0.1
        x, r = self.run_layer(x, w3, relu=True)
        all_results.append(r)

        w4 = np.random.randn(84, 120, 1, 1) * 0.1
        x, r = self.run_layer(x, w4, relu=True)
        all_results.append(r)

        w5 = np.random.randn(10, 84, 1, 1) * 0.1
        x, r = self.run_layer(x, w5, relu=False)
        all_results.append(r)

        return x.flatten(), all_results
