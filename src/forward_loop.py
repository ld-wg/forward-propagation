"""Reference non-vectorized forward propagation."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from .activations import numpy_activation

Array = np.ndarray


def forward_loop(
    X: Array,
    weights: Sequence[Array],
    biases: Sequence[Array],
    activations: Sequence[str],
) -> Array:
    """Compute forward propagation using explicit Python loops."""

    if not weights:
        raise ValueError("At least one layer is required.")
    if len(weights) != len(biases) or len(weights) != len(activations):
        raise ValueError("Layer/activation length mismatch.")

    A = X.astype(np.float32, copy=False)
    for layer_idx, (W, b, act_name) in enumerate(zip(weights, biases, activations)):
        samples, prev_units = A.shape
        units = W.shape[0]
        if prev_units != W.shape[1]:
            raise ValueError("Weight shape does not match previous activation dimension.")

        Z = np.zeros((samples, units), dtype=A.dtype)
        for sample_idx in range(samples):
            for unit_idx in range(units):
                acc = 0.0
                for feature_idx in range(prev_units):
                    acc += A[sample_idx, feature_idx] * W[unit_idx, feature_idx]
                Z[sample_idx, unit_idx] = acc + b[unit_idx]

        activation_fn = numpy_activation(act_name)
        A = activation_fn(Z)

    return A


__all__ = ["forward_loop"]
