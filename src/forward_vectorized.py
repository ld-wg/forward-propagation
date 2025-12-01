"""Vectorized forward propagation using NumPy."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from .activations import numpy_activation

Array = np.ndarray


def forward_vectorized(
    X: Array,
    weights: Sequence[Array],
    biases: Sequence[Array],
    activations: Sequence[str],
) -> Array:
    """Compute forward pass using matrix multiplications."""

    if not weights or not biases:
        raise ValueError("Weights and biases must be non-empty sequences.")
    if len(weights) != len(biases) or len(weights) != len(activations):
        raise ValueError("Mismatch between layers and activation list.")

    A = X
    for W, b, act_name in zip(weights, biases, activations):
        if W.shape[1] != A.shape[1]:
            raise ValueError("Incompatible shapes between activations and weights.")
        Z = A @ W.T + b
        activation_fn = numpy_activation(act_name)
        A = activation_fn(Z)
    return A


__all__ = ["forward_vectorized"]
