import numpy as np

from src.forward_loop import forward_loop
from src.forward_vectorized import forward_vectorized
from src.model import NetworkConfig, initialize_parameters


def _build_example():
    rng = np.random.default_rng(0)
    config = NetworkConfig(name="Tiny", input_dim=8, hidden_layers=[4, 4], output_dim=2)
    X = rng.standard_normal((5, config.input_dim)).astype(np.float32)
    weights, biases, activations = initialize_parameters(
        config=config,
        hidden_activation="relu",
        rng=np.random.default_rng(1),
        dtype=np.float32,
    )
    return X, weights, biases, activations


def test_loop_matches_vectorized():
    X, weights, biases, activations = _build_example()
    vec = forward_vectorized(X, weights, biases, activations)
    loop = forward_loop(X, weights, biases, activations)
    np.testing.assert_allclose(vec, loop, rtol=1e-5, atol=1e-5)
