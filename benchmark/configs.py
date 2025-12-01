"""Benchmark configurations for stress testing."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from src.model import NetworkConfig


@dataclass(frozen=True)
class StressConfig:
    tag: str
    config: NetworkConfig


STRESS_CONFIGS: List[StressConfig] = [
    StressConfig(
        tag="small",
        config=NetworkConfig(
            name="Small",
            input_dim=256,
            hidden_layers=[32, 16],
            output_dim=10,
        ),
    ),
    StressConfig(
        tag="medium",
        config=NetworkConfig(
            name="Medium",
            input_dim=256,
            hidden_layers=[128, 64, 32, 16],
            output_dim=10,
        ),
    ),
    StressConfig(
        tag="large",
        config=NetworkConfig(
            name="Large",
            input_dim=256,
            hidden_layers=[128] * 8,
            output_dim=10,
        ),
    ),
    StressConfig(
        tag="extreme",
        config=NetworkConfig(
            name="Extreme",
            input_dim=256,
            hidden_layers=[512, 256, 256, 128, 128, 64, 64, 32, 32, 16],
            output_dim=10,
        ),
    ),
]

__all__ = ["StressConfig", "STRESS_CONFIGS"]
