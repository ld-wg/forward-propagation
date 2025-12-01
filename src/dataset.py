"""Synthetic dataset helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

Dataset = np.ndarray


@dataclass(frozen=True)
class DatasetSpec:
    samples: int = 1_024
    features: int = 256
    seed: int = 7
    dtype: str = "float32"


_DEFAULT_SPEC = DatasetSpec()


def generate_dataset(spec: DatasetSpec = _DEFAULT_SPEC) -> Dataset:
    """Generate a reproducible Gaussian dataset with zero mean unit variance."""

    rng = np.random.default_rng(spec.seed)
    data = rng.standard_normal((spec.samples, spec.features), dtype=getattr(np, spec.dtype))
    return data.astype(getattr(np, spec.dtype), copy=False)


__all__ = ["Dataset", "DatasetSpec", "generate_dataset"]
