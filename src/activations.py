"""Activation utilities shared across forward implementations."""
from __future__ import annotations

from typing import Callable

import numpy as np


ActivationFn = Callable[[np.ndarray], np.ndarray]
ArrayLike = np.ndarray


def _relu():
    return lambda z: np.maximum(0, z)


def _sigmoid():
    return lambda z: 1.0 / (1.0 + np.exp(-z))


def _linear():
    return lambda z: z


_ACTIVATION_FACTORIES = {
    "relu": _relu,
    "sigmoid": _sigmoid,
    "linear": _linear,
}


def get_activation(name: str):
    """Return an activation callable for the given backend."""

    key = name.lower()
    if key not in _ACTIVATION_FACTORIES:
        raise ValueError(f"Unsupported activation: {name}")
    return _ACTIVATION_FACTORIES[key]()


def numpy_activation(name: str) -> ActivationFn:
    """Shortcut for NumPy-based activation."""

    return get_activation(name)


SUPPORTED_ACTIVATIONS = tuple(_ACTIVATION_FACTORIES.keys())

__all__ = [
    "ActivationFn",
    "ArrayLike",
    "SUPPORTED_ACTIVATIONS",
    "get_activation",
    "numpy_activation",
]
