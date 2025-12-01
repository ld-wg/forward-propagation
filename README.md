# Configurable Forward Propagation

Implementation of configurable forward propagation pipelines (vectorized NumPy and non-vectorized loops) plus benchmarking utilities for stress testing activation functions and execution strategies.

## Environment

- Python >= 3.10
- Install dependencies: `pip install -r requirements.txt`

## Dataset

All experiments use a fixed synthetic dataset with 1,024 samples and 256 features generated via a reproducible NumPy RNG seed. Keeping the dataset constant guarantees fair comparisons across configurations.

## Project Layout

```
src/                Core forward propagation implementations and helpers
benchmark/          Stress-test configurations and benchmark runner
results/            CSV summaries and generated figures
report/             Final report
tests/              Lightweight regression/unit tests
```

## Experiment Runner

`python -m benchmark.run_experiment`

- No flags or parameters: the script always wipes `results/`, executes every stress configuration (small → extreme) with the predefined runs/activations/implementations, and regenerates the consolidated PNG (`results/plots/experiment_summary.png`) plus `results/summary.csv` and `results/metadata.json`.

Outputs:

- `results/summary.csv`: final CPU timings used in the report
- `results/plots/experiment_summary.png`: single image containing the stress table, comparison table, and grouped bar chart

## Tests

Run `pytest` to verify numerical parity between implementations on a small deterministic network.

## Report

Final findings live in `report/forward_propagation.md`, covering design decisions, benchmark methodology, results tables, figures, and conclusions.
