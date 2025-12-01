"""Network configuration and parameter helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from .activations import SUPPORTED_ACTIVATIONS

Array = np.ndarray


@dataclass(frozen=True)
class NetworkConfig:
    name: str
    input_dim: int
    hidden_layers: Sequence[int]
    output_dim: int
    output_activation: str = "linear"

    def layer_sizes(self) -> List[int]:
        return [self.input_dim, *self.hidden_layers, self.output_dim]


def _fan_in_scale(activation: str, fan_in: int) -> float:
    if activation.lower() == "relu":
        return np.sqrt(2.0 / fan_in)
    return np.sqrt(1.0 / fan_in)


def initialize_parameters(
    config: NetworkConfig,
    hidden_activation: str,
    rng: np.random.Generator,
    dtype: np.dtype | None = None,
) -> Tuple[List[Array], List[Array], List[str]]:
    """Create weight/bias tensors plus activation names per layer."""

    if hidden_activation.lower() not in SUPPORTED_ACTIVATIONS:
        raise ValueError(f"Unsupported activation: {hidden_activation}")

    dtype = dtype or np.float32
    layers = config.layer_sizes()
    weights: List[Array] = []
    biases: List[Array] = []
    activations = [hidden_activation] * len(config.hidden_layers) + [config.output_activation]

    for idx, (fan_in, fan_out) in enumerate(zip(layers[:-1], layers[1:])):
        act_name = activations[idx]
        scale = _fan_in_scale(act_name, fan_in)
        W = rng.standard_normal((fan_out, fan_in)).astype(dtype) * scale
        b = np.zeros(fan_out, dtype=dtype)
        weights.append(W)
        biases.append(b)

    return weights, biases, activations


__all__ = ["NetworkConfig", "initialize_parameters"]
