"""Single entrypoint for the fixed forward-propagation experiment."""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Callable, Dict, List

import matplotlib.pyplot as plt
import numpy as np

from src.dataset import DatasetSpec, generate_dataset
from src.forward_loop import forward_loop
from src.forward_vectorized import forward_vectorized
from src.model import initialize_parameters

from .configs import STRESS_CONFIGS, StressConfig


Array = np.ndarray
IMPLEMENTATIONS: Dict[str, Callable[[Array, List[Array], List[Array], List[str]], Array]] = {
    "vectorized": forward_vectorized,
    "non_vectorized": forward_loop,
}
CONFIG_BY_TAG = {cfg.tag: cfg for cfg in STRESS_CONFIGS}
RESULTS_DIR = Path("results")
DATASET_SPEC = DatasetSpec(samples=1_024, features=256, seed=7)
DATASET_CACHE_PATH = Path("data") / "dataset.npy"
WEIGHT_BASE_SEED = 1234
ACTIVATIONS = ("relu", "sigmoid")
RUNS_PER_JOB = 5


@dataclass(frozen=True)
class Job:
    config_tag: str
    implementation: str
    activations: List[str]
    runs: int


def _build_jobs() -> List[Job]:
    jobs: List[Job] = []
    for cfg in STRESS_CONFIGS:
        for implementation in IMPLEMENTATIONS:
            jobs.append(
                Job(
                    config_tag=cfg.tag,
                    implementation=implementation,
                    activations=list(ACTIVATIONS),
                    runs=RUNS_PER_JOB,
                )
            )
    return jobs


CPU_JOBS: List[Job] = _build_jobs()


def _seed_from_label(base_seed: int, label: str) -> int:
    digest = hashlib.sha256(label.encode()).hexdigest()
    return base_seed + int(digest[:8], 16)


def _load_or_create_dataset() -> Array:
    if DATASET_CACHE_PATH.exists():
        return np.load(DATASET_CACHE_PATH).astype(np.float32, copy=False)

    DATASET_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataset = generate_dataset(DATASET_SPEC).astype(np.float32)
    np.save(DATASET_CACHE_PATH, dataset)
    return dataset


def _measure(fn: Callable[[], Array], runs: int) -> List[float]:
    timings: List[float] = []
    for _ in range(runs):
        timings.append(_time_once(fn))
    return timings


def _time_once(fn: Callable[[], Array]) -> float:
    import time

    start = time.perf_counter()
    fn()
    end = time.perf_counter()
    return end - start


def _parameter_count(config: StressConfig) -> int:
    layers = config.config.layer_sizes()
    return sum((fan_in + 1) * fan_out for fan_in, fan_out in zip(layers[:-1], layers[1:]))


def _reset_results_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _write_csv(rows: List[Dict[str, object]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_metadata(path: Path, runs_by_config: Dict[str, int]) -> None:
    payload = {
        "dataset_samples": DATASET_SPEC.samples,
        "dataset_features": DATASET_SPEC.features,
        "runs_by_config": runs_by_config,
    }
    with path.open("w") as fh:
        json.dump(payload, fh, indent=2)


def _config_table_data() -> List[List[str]]:
    rows: List[List[str]] = []
    for cfg in STRESS_CONFIGS:
        rows.append(
            [
                cfg.config.name,
                str(len(cfg.config.hidden_layers)),
                "[" + ", ".join(str(val) for val in cfg.config.hidden_layers) + "]",
                f"{_parameter_count(cfg):,}",
            ]
        )
    return rows


def _performance_table_data(rows: List[Dict[str, object]]) -> List[List[str]]:
    lookup = {
        (row["config_tag"], row["implementation"], row["activation"]): row
        for row in rows
    }
    result: List[List[str]] = []
    for cfg in STRESS_CONFIGS:
        for impl in IMPLEMENTATIONS:
            for activation in ACTIVATIONS:
                row = lookup.get((cfg.tag, impl, activation))
                time_val = f"{row['mean_time_s']:.3e}" if row else "N/A"
                result.append(
                    [
                        cfg.config.name,
                        impl.replace("_", " ").title(),
                        activation.title(),
                        time_val,
                    ]
                )
    return result


def _plot_summary(rows: List[Dict[str, object]], output_dir: Path) -> None:
    """Render tables and bar chart as independent figures."""

    output_dir.mkdir(parents=True, exist_ok=True)

    def _table_figure(cell_text: List[List[str]], col_labels: List[str], path: Path, scale: float = 1.1) -> None:
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.axis("off")
        table = ax.table(
            cellText=cell_text,
            colLabels=col_labels,
            loc="center",
            cellLoc="center",
        )
        table.scale(1, scale)
        fig.tight_layout()
        fig.savefig(path, dpi=220, bbox_inches="tight")
        plt.close(fig)

    _table_figure(
        _config_table_data(),
        [
            "Configuração",
            "Número de Camadas Ocultas (L)",
            "Neurônios por Camada Oculta",
            "Parâmetros (≈)",
        ],
        output_dir / "config_table.png",
        scale=1.3,
    )

    _table_figure(
        _performance_table_data(rows),
        ["Configuração", "Implementação", "Função de Ativação", "Tempo Médio (s)"],
        output_dir / "performance_table.png",
        scale=1.15,
    )

    lookup = {
        (row["config_tag"], row["implementation"], row["activation"]): row
        for row in rows
    }
    configs = [cfg.tag for cfg in STRESS_CONFIGS]

    fig, ax_bar = plt.subplots(figsize=(11, 5))
    width = 0.18
    x = np.arange(len(configs))
    combo_labels = [
        ("vectorized", "relu", "Vetorizada – ReLU"),
        ("vectorized", "sigmoid", "Vetorizada – Sigmoide"),
        ("non_vectorized", "relu", "Não Vetorizada – ReLU"),
        ("non_vectorized", "sigmoid", "Não Vetorizada – Sigmoide"),
    ]

    offset_center = (len(combo_labels) - 1) / 2
    for idx, (impl, activation, label) in enumerate(combo_labels):
        means = []
        for cfg in configs:
            row = lookup.get((cfg, impl, activation))
            means.append(row["mean_time_s"] if row else np.nan)
        ax_bar.bar(
            x + (idx - offset_center) * width,
            means,
            width=width,
            label=label,
        )

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([cfg.title() for cfg in configs])
    ax_bar.set_ylabel("Tempo médio (s) (escala log)")
    ax_bar.set_title("Tempo de Execução por Configuração / Implementação / Ativação")
    ax_bar.set_yscale("log")
    ax_bar.legend()

    fig.tight_layout()
    fig.savefig(output_dir / "execution_times.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _run_job(job: Job, dataset: Array) -> List[Dict[str, object]]:
    config = CONFIG_BY_TAG[job.config_tag]
    impl_fn = IMPLEMENTATIONS[job.implementation]
    rows: List[Dict[str, object]] = []
    for activation in job.activations:
        seed = _seed_from_label(WEIGHT_BASE_SEED, f"{job.config_tag}-{activation}")
        rng = np.random.default_rng(seed)
        weights, biases, activation_list = initialize_parameters(
            config=config.config,
            hidden_activation=activation,
            rng=rng,
            dtype=np.float32,
        )
        timings = _measure(lambda: impl_fn(dataset, weights, biases, activation_list), job.runs)
        rows.append(
            {
                "config": config.config.name,
                "config_tag": job.config_tag,
                "implementation": job.implementation,
                "activation": activation,
                "mean_time_s": mean(timings),
                "std_time_s": pstdev(timings) if len(timings) > 1 else 0.0,
                "runs": job.runs,
                "parameters": _parameter_count(config),
                "layers": "[" + ", ".join(str(val) for val in config.config.hidden_layers) + "]",
            }
        )
    return rows


def run_experiment() -> None:
    _reset_results_dir(RESULTS_DIR)
    dataset = _load_or_create_dataset()

    cpu_rows: List[Dict[str, object]] = []
    runs_by_config: Dict[str, int] = {}
    for job in CPU_JOBS:
        runs_by_config[job.config_tag] = max(runs_by_config.get(job.config_tag, 0), job.runs)
        cpu_rows.extend(_run_job(job, dataset))

    summary_path = RESULTS_DIR / "summary.csv"
    _write_csv(cpu_rows, summary_path)
    _write_metadata(RESULTS_DIR / "metadata.json", runs_by_config)

    _plot_summary(cpu_rows, RESULTS_DIR / "plots")
    print(f"[experiment] Summary written to {summary_path}")


if __name__ == "__main__":
    run_experiment()
